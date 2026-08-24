"""Emit contract for ``g.constraints.interface()`` → ``emit_interfaces``
(ADR 0093 S5).

Text-only (no OpenSees backend), in the shape of ``test_contact_emit.py``:
records in, emitter calls out. What is pinned here:

* the per-record emit ORDER — phantom ``node`` → nested ``equalDOF`` →
  the two tributary-scaled materials → the ``zeroLength`` (D5);
* **INV-1's node order, literally**: ``iNode`` is the real continuum
  ``master_node`` and ``jNode`` is the phantom when one exists, else the
  real slave. Swapping them still converges — as a tension-only
  interface — so a byte-golden line is the only honest gate;
* the D1 translation table's scaling (``E ∝ A_trib`` everywhere,
  ``epsyP`` CONSTANT across pairs, ``|Fy| ∝ A_trib``) and its
  sign ownership (``Fy < 0`` **and** ``gap <= 0``, always);
* the emit-time ndf gate, which is the S4 refinement: a 3-dof slave with
  no phantom declared is the exact silent-wrong-model case.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NodePairRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees._internal.build import BridgeError, emit_interfaces
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees._internal.tag_resolution import is_phantom_node
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

# Exact binary values throughout, so the golden lines carry no float
# noise: A_trib ∈ {0.25, 0.5}, k ∈ {1e6, 1e5}, tau_b = 250.
ENT_LAW = NormalLaw(kind="ent", k_per_area=1.0e6)
GAP_LAW = NormalLaw(
    kind="epp_gap", k_per_area=1.0e6, tau_b_n=400.0, gap=-0.001)
ELASTIC_N = NormalLaw(kind="elastic", k_per_area=1.0e6)
EPP_LAW = TangentialLaw(kind="epp", k_per_area=1.0e5, tau_b=250.0)
ELASTIC_T = TangentialLaw(kind="elastic", k_per_area=1.0e5)

ORIENT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class _Fem:
    """The two attributes ``emit_interfaces`` reads."""

    def __init__(self, interfaces):
        self.elements = type("E", (), {"interfaces": interfaces})()


def _rec(
    master: int, slave: int, *,
    a_trib: float,
    normal=ENT_LAW,
    tangential=EPP_LAW,
    phantom: int | None = None,
    coords=(1.0, 0.5, 0.0),
    phantom_ndf: int | None = 2,
    orient=ORIENT,
    name: str | None = None,
) -> InterfaceRecord:
    equal_dofs = []
    if phantom is not None:
        equal_dofs = [NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, name=name,
            master_node=slave, slave_node=phantom, dofs=[1, 2],
        )]
    return InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        name=name,
        master_node=master,
        slave_node=slave,
        backing_element=7,
        orient=orient,
        a_trib=a_trib,
        normal_law=normal,
        tangential_law=tangential,
        phantom_node=phantom,
        phantom_coords=(np.asarray(coords, dtype=float)
                        if phantom is not None else None),
        phantom_ndf=phantom_ndf if phantom is not None else None,
        equal_dof_records=equal_dofs,
    )


def _emit(records, *, ndf=None, envelope=2, ndm=2, emitter=None):
    em = emitter if emitter is not None else RecordingEmitter()
    if emitter is None:
        # Production issues ``model`` first (apeSees.emit step 1), and that
        # is what teaches the emitter its ``ndm`` — hence how many
        # coordinates a ``node`` line carries (``trim_coords_to_ndm``).
        # Dropped from ``calls`` afterwards for the same reason ``_golden``
        # slices off the banner: it is setup, not this pass's output.
        # (``_golden`` issues it itself for emitters passed in.)
        em.model(ndm=ndm, ndf=envelope)
        em.calls.clear()
    if ndf is None:
        ndf = {}
        for r in records:
            ndf[int(r.master_node)] = 2
            ndf[int(r.slave_node)] = 3 if r.phantom_node is not None else 2
    emit_interfaces(
        em, _Fem(list(records)), TagAllocator(),
        effective_ndf=ndf, envelope_ndf=envelope, ndm=ndm,
    )
    return em


def _golden(records, emitter, **kw):
    """The lines ``emit_interfaces`` ADDED — the Tcl / Py emitters open
    with a generated-file banner that is not this pass's business.

    ``model`` is issued first because production always does: it is step 1
    of ``apeSees.emit``, and it is what teaches the emitter its ``ndm`` —
    and therefore how many coordinates a ``node`` line carries
    (``trim_coords_to_ndm``, ADR 0099). A bare emitter has ``ndm = None``
    and trims nothing, so a rig that skips it asserts a deck shape no real
    deck can have: ``node 101 1.0 0.5 0.0 -ndf 2``, whose padded third
    coordinate is exactly what strands the ``-ndf`` token.
    """
    emitter.model(ndm=kw.get("ndm", 2), ndf=kw.get("envelope", 2))
    before = len(emitter.lines())
    _emit(records, emitter=emitter, **kw)
    return emitter.lines()[before:]


# ==========================================================================
# No-op
# ==========================================================================
def test_noop_when_no_interfaces():
    em = _emit([])
    assert em.calls == []


# ==========================================================================
# Equal-ndf — order, node order, distinct per-pair material tags
# ==========================================================================
def test_equal_ndf_emit_order_and_node_order():
    em = _emit([_rec(10, 20, a_trib=0.25)])
    assert [c[0] for c in em.calls] == [
        "uniaxialMaterial", "uniaxialMaterial", "element",
    ]
    # No phantom ⇒ no node / equalDOF lines at all.
    assert not [c for c in em.calls if c[0] in ("node", "equalDOF")]
    ele = [c for c in em.calls if c[0] == "element"][0][1]
    # INV-1: iNode = the real continuum master, jNode = the real slave.
    assert ele[0] == "zeroLength"
    assert ele[2] == 10 and ele[3] == 20


def test_equal_ndf_golden_tcl():
    assert _golden([_rec(10, 20, a_trib=0.25)], TclEmitter()) == [
        "uniaxialMaterial ENT 1 250000.0",
        "uniaxialMaterial ElasticPP 2 25000.0 0.0025",
        "element zeroLength 1 10 20 -mat 1 2 -dir 1 2 "
        "-orient 1.0 0.0 0.0 0.0 1.0 0.0",
    ]


def test_equal_ndf_golden_py():
    assert _golden([_rec(10, 20, a_trib=0.25)], PyEmitter()) == [
        "ops.uniaxialMaterial('ENT', 1, 250000.0)",
        "ops.uniaxialMaterial('ElasticPP', 2, 25000.0, 0.0025)",
        "ops.element('zeroLength', 1, 10, 20, '-mat', 1, 2, '-dir', 1, 2, "
        "'-orient', 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)",
    ]


def test_per_pair_material_tags_are_distinct():
    em = _emit([
        _rec(10, 20, a_trib=0.25),
        _rec(11, 21, a_trib=0.5),
    ])
    mats = [c[1][1] for c in em.calls if c[0] == "uniaxialMaterial"]
    assert mats == [1, 2, 3, 4]
    eles = [c[1] for c in em.calls if c[0] == "element"]
    assert [e[1] for e in eles] == [1, 2]
    # each element references its OWN two materials, in (normal, tangential)
    # order on -dir 1 2.
    assert list(eles[0][4:8]) == ["-mat", 1, 2, "-dir"]
    assert list(eles[1][4:8]) == ["-mat", 3, 4, "-dir"]


def test_orient_is_six_floats_from_the_record():
    curved = (0.6, 0.8, 0.0, -0.8, 0.6, 0.0)
    em = _emit([_rec(10, 20, a_trib=0.25, orient=curved)])
    ele = [c for c in em.calls if c[0] == "element"][0][1]
    i = ele.index("-orient")
    assert tuple(ele[i + 1:i + 7]) == curved
    assert len(ele) == i + 7          # -orient is the last block


def test_name_emits_the_declaration_comment():
    em = _emit([_rec(10, 20, a_trib=0.25, name="RockLiner")])
    assert em.calls[0] == ("mp_constraint_comment", ("RockLiner",), {})


def test_record_without_orient_is_refused():
    # -orient omitted ⇒ the GLOBAL frame ⇒ a normal law along global x.
    rec = _rec(10, 20, a_trib=0.25)
    rec.orient = None
    with pytest.raises(BridgeError, match="GLOBAL frame"):
        _emit([rec])


# ==========================================================================
# Mixed ndf — phantom (ndf=2!) → equalDOF → materials → zeroLength(phantom)
# ==========================================================================
def test_mixed_ndf_emit_order_and_jnode_is_the_phantom():
    em = _emit([_rec(10, 20, a_trib=0.25, phantom=101)])
    assert [c[0] for c in em.calls] == [
        "node", "equalDOF", "uniaxialMaterial", "uniaxialMaterial", "element",
    ]
    node_call = em.calls[0]
    # ndm=2 deck ⇒ TWO coordinates; a third would swallow the -ndf.
    assert node_call[1] == (101, 1.0, 0.5)
    assert node_call[2] == {"ndf": 2}          # NOT the helper's hardcoded 6
    assert em.calls[1][1] == (20, 101, 1, 2)   # retained beam, constrained phantom
    ele = em.calls[-1][1]
    assert ele[2] == 10 and ele[3] == 101      # INV-1: jNode is the phantom


def test_mixed_ndf_golden_tcl():
    assert _golden(
        [_rec(10, 20, a_trib=0.25, phantom=101, name="RockLiner")],
        TclEmitter(),
    ) == [
        "# RockLiner",
        "node 101 1.0 0.5 -ndf 2",
        "equalDOF 20 101 1 2",
        "uniaxialMaterial ENT 1 250000.0",
        "uniaxialMaterial ElasticPP 2 25000.0 0.0025",
        "element zeroLength 1 10 101 -mat 1 2 -dir 1 2 "
        "-orient 1.0 0.0 0.0 0.0 1.0 0.0",
    ]


def test_mixed_ndf_golden_py():
    assert _golden(
        [_rec(10, 20, a_trib=0.25, phantom=101, name="RockLiner")],
        PyEmitter(),
    ) == [
        "# RockLiner",
        "ops.node(101, 1.0, 0.5, '-ndf', 2)",
        "ops.equalDOF(20, 101, 1, 2)",
        "ops.uniaxialMaterial('ENT', 1, 250000.0)",
        "ops.uniaxialMaterial('ElasticPP', 2, 25000.0, 0.0025)",
        "ops.element('zeroLength', 1, 10, 101, '-mat', 1, 2, '-dir', 1, 2, "
        "'-orient', 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)",
    ]


def test_phantom_without_coords_is_refused():
    rec = _rec(10, 20, a_trib=0.25, phantom=101)
    rec.phantom_coords = None
    with pytest.raises(BridgeError, match="phantom_coords"):
        _emit([rec])


# ==========================================================================
# D1 translation table — scaling and signs
# ==========================================================================
def _mat_lines(records, **kw):
    em = _emit(records, **kw)
    return [c[1] for c in em.calls if c[0] == "uniaxialMaterial"]


@pytest.mark.parametrize("normal, token", [
    (ENT_LAW, "ENT"),
    (ELASTIC_N, "Elastic"),
    (GAP_LAW, "ElasticPPGap"),
])
def test_normal_E_scales_with_a_trib(normal, token):
    mats = _mat_lines([
        _rec(10, 20, a_trib=0.25, normal=normal),
        _rec(11, 21, a_trib=0.5, normal=normal),
    ])
    n0, n1 = mats[0], mats[2]
    assert n0[0] == n1[0] == token
    assert n1[2] == pytest.approx(2.0 * n0[2])
    assert n0[2] == pytest.approx(1.0e6 * 0.25)


def test_tangential_elastic_E_scales_with_a_trib():
    mats = _mat_lines([
        _rec(10, 20, a_trib=0.25, tangential=ELASTIC_T),
        _rec(11, 21, a_trib=0.5, tangential=ELASTIC_T),
    ])
    t0, t1 = mats[1], mats[3]
    assert t0[0] == t1[0] == "Elastic"
    assert t1[2] == pytest.approx(2.0 * t0[2])


def test_epp_epsyP_is_constant_while_E_scales():
    # The ADR review's ElasticPP correction: the deck line carries a
    # yield STRAIN, so A_trib cancels there and only E scales. The yield
    # FORCE tau_b * A_trib is the emergent product E * epsyP.
    mats = _mat_lines([
        _rec(10, 20, a_trib=0.25),
        _rec(11, 21, a_trib=0.5),
    ])
    t0, t1 = mats[1], mats[3]
    assert t0[0] == t1[0] == "ElasticPP"
    assert t1[2] == pytest.approx(2.0 * t0[2])       # E ∝ A_trib
    assert t1[3] == pytest.approx(t0[3])             # epsyP constant
    assert t0[3] == pytest.approx(250.0 / 1.0e5)
    # …and the emergent yield force IS tau_b * A_trib.
    assert t0[2] * t0[3] == pytest.approx(250.0 * 0.25)
    assert t1[2] * t1[3] == pytest.approx(250.0 * 0.5)


def test_epp_gap_signs_and_fy_scaling():
    mats = _mat_lines([
        _rec(10, 20, a_trib=0.25, normal=GAP_LAW),
        _rec(11, 21, a_trib=0.5, normal=GAP_LAW),
    ])
    n0, n1 = mats[0], mats[2]
    # uniaxialMaterial ElasticPPGap tag E Fy gap
    for n in (n0, n1):
        assert n[3] < 0.0          # Fy — compression side (INV-1)
        assert n[4] <= 0.0         # gap — never a tension-side offset
    assert abs(n1[3]) == pytest.approx(2.0 * abs(n0[3]))
    assert abs(n0[3]) == pytest.approx(400.0 * 0.25)
    assert n0[4] == pytest.approx(-0.001)


def test_epp_gap_sign_is_owned_by_the_translation_not_the_law():
    # A law spelled with a positive gap magnitude (a future schema
    # relaxation, or a hand-built record) must STILL emit gap <= 0 with
    # Fy < 0 — the translation forces both, so the sign(Fy)/sign(gap)
    # pair the fork only *warns* about can never disagree.
    law = NormalLaw.__new__(NormalLaw)       # bypass the record-side check
    object.__setattr__(law, "kind", "epp_gap")
    object.__setattr__(law, "k_per_area", 1.0e6)
    object.__setattr__(law, "tau_b_n", -400.0)
    object.__setattr__(law, "gap", 0.001)
    n = _mat_lines([_rec(10, 20, a_trib=0.25, normal=law)])[0]
    assert n[3] < 0.0 and n[4] < 0.0


def test_epp_gap_zero_gap_renders_positive_zero():
    law = NormalLaw(kind="epp_gap", k_per_area=1.0e6, tau_b_n=400.0, gap=0.0)
    lines = _golden([_rec(10, 20, a_trib=0.25, normal=law)], TclEmitter())
    assert lines[0] == "uniaxialMaterial ElasticPPGap 1 250000.0 -100.0 0.0"


def test_unknown_law_kind_fails_loud():
    law = NormalLaw.__new__(NormalLaw)
    object.__setattr__(law, "kind", "hysteretic")
    object.__setattr__(law, "k_per_area", 1.0e6)
    with pytest.raises(BridgeError, match="unknown NormalLaw kind"):
        _emit([_rec(10, 20, a_trib=0.25, normal=law)])
    tl = TangentialLaw.__new__(TangentialLaw)
    object.__setattr__(tl, "kind", "coulomb")
    object.__setattr__(tl, "k_per_area", 1.0e5)
    with pytest.raises(BridgeError, match="unknown TangentialLaw kind"):
        _emit([_rec(10, 20, a_trib=0.25, tangential=tl)])


# ==========================================================================
# Emit-time ndf gate (the S4 refinement)
# ==========================================================================
def test_three_dof_slave_without_phantom_fails_loud():
    with pytest.raises(BridgeError, match=r"slave_ndf=3"):
        _emit([_rec(10, 20, a_trib=0.25)], ndf={10: 2, 20: 3})


def test_two_dof_slave_with_phantom_fails_loud():
    with pytest.raises(BridgeError, match="phantom bridge"):
        _emit([_rec(10, 20, a_trib=0.25, phantom=101)], ndf={10: 2, 20: 2})


def test_non_two_dof_master_fails_loud():
    with pytest.raises(BridgeError, match="master node 10 has ndf=3"):
        _emit([_rec(10, 20, a_trib=0.25)], ndf={10: 3, 20: 2})


def test_phantom_ndf_must_match_the_master():
    rec = _rec(10, 20, a_trib=0.25, phantom=101, phantom_ndf=3)
    with pytest.raises(BridgeError, match="phantom node 101 carries ndf=3"):
        _emit([rec])


def test_endpoints_absent_from_the_map_fall_to_the_envelope():
    # An element-less endpoint takes the ops.model envelope; envelope 3
    # against a no-phantom record is the silent-wrong case.
    with pytest.raises(BridgeError, match=r"slave_ndf=3"):
        _emit([_rec(10, 20, a_trib=0.25)], ndf={10: 2}, envelope=3)


def test_ndm_must_be_two():
    with pytest.raises(BridgeError, match="ndm=3"):
        _emit([_rec(10, 20, a_trib=0.25)], ndm=3)


def test_validation_runs_before_any_emission():
    # Two good records then a bad one — nothing may reach the deck.
    em = TclEmitter()
    before = len(em.lines())
    with pytest.raises(BridgeError):
        _emit(
            [_rec(10, 20, a_trib=0.25),
             _rec(11, 21, a_trib=0.25),
             _rec(12, 22, a_trib=0.25)],
            ndf={10: 2, 20: 2, 11: 2, 21: 2, 12: 2, 22: 3},
            emitter=em,
        )
    assert em.lines()[before:] == []


# ==========================================================================
# Phantom-tag predicate (D4(b))
# ==========================================================================
def test_interface_phantoms_reach_the_phantom_tag_predicate():
    em = RecordingEmitter()
    _emit([_rec(10, 20, a_trib=0.25, phantom=101),
           _rec(11, 21, a_trib=0.25, phantom=102)], emitter=em)
    assert is_phantom_node(em, 101)
    assert is_phantom_node(em, 102)
    assert not is_phantom_node(em, 20)


def test_phantom_tag_registration_is_additive():
    # The MP pass installs its own set first (emit_mp_constraints step 0);
    # the interface pass must UNION, never replace.
    from apeGmsh.opensees._internal.tag_resolution import set_phantom_node_tags

    em = RecordingEmitter()
    set_phantom_node_tags(em, {900})
    _emit([_rec(10, 20, a_trib=0.25, phantom=101)], emitter=em)
    assert is_phantom_node(em, 900) and is_phantom_node(em, 101)


def test_h5_emitter_classifies_the_interface_phantom():
    # The predicate's only consumer: H5Emitter.node routes a phantom tag
    # into ``phantom_node_tags`` instead of treating it as a real node.
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    em = H5Emitter(model_name="iface_phantom")
    em.model(ndm=2, ndf=2)
    _emit([_rec(10, 20, a_trib=0.25, phantom=101)], emitter=em)
    assert em._phantom_node_tags == [101]


def test_equal_ndf_model_registers_no_phantoms():
    em = RecordingEmitter()
    _emit([_rec(10, 20, a_trib=0.25)], emitter=em)
    assert not is_phantom_node(em, 20)


# ==========================================================================
# Tag determinism
# ==========================================================================
def test_two_emits_of_the_same_model_are_byte_identical():
    recs = [_rec(10, 20, a_trib=0.25, phantom=101),
            _rec(11, 21, a_trib=0.5, phantom=102)]
    first = _golden(recs, TclEmitter())
    second = _golden(recs, TclEmitter())
    assert first == second


def test_interface_tags_continue_the_shared_namespaces():
    # emit_interfaces draws from the canonical allocator, so its tags
    # start ABOVE whatever the mesh element / material fan-out already
    # consumed (the flat path calls it after both).
    tags = TagAllocator()
    for _ in range(5):
        tags.allocate("element")
    for _ in range(3):
        tags.allocate("uniaxialMaterial")
    em = RecordingEmitter()
    emit_interfaces(
        em, _Fem([_rec(10, 20, a_trib=0.25)]), tags,
        effective_ndf={10: 2, 20: 2}, envelope_ndf=2, ndm=2,
    )
    assert [c[1][1] for c in em.calls if c[0] == "uniaxialMaterial"] == [4, 5]
    assert [c[1][1] for c in em.calls if c[0] == "element"] == [6]
