"""ADR 0092 — 2026-08-13 adversarial-review fixes (F1 / F2 / F3 / F5).

The post-S4 review of the partitioned contact emit (PR #927 + follow-ups
#944/#945/#948) found four holes in the refusal lattice; these tests pin
the fixes:

* **F1** — a pattern-borne ``sp`` on a contact-ghosted node was neither
  mirrored nor refused: pattern sp lines fan out on a node's NATIVE
  ranks only (``_emit_one_pattern_partitioned`` filters on
  ``owned_nodes``), so a displacement-driven contact deck (ordinary
  authoring: the prescribed motion rides the slave surface) constrained
  the DOF on the slave's home rank while the owner rank's ghost copy
  stayed FREE — the exact constrained-DOF disagreement ADR 0027 INV-2
  measured as "Matrix is Singular Numerically". The INTERFACE lane
  already refused this (``_refuse_pattern_sp_on_interface_ghosts``); the
  contact planner now runs the identical sweep (``lane="contact"``).

* **F2** — the INV-4 cut-master + auto backstop silently disengaged when
  the facet → element map was unresolvable, and the map was
  all-or-nothing: ONE ambiguous facet anywhere returned ``None`` for the
  WHOLE surface, skipping the backstop and falling back to the node
  tally (which only refuses the all-shared tie). A genuinely cut master
  + ``kn="auto"`` + one ambiguous facet emitted a deck whose off-rank
  auto-kn facets the fork silently D5.2-skips. The resolver is now
  per-facet; the straddle check runs on the RESOLVED subset; and a
  partial map + an active auto knob + a multi-rank master (node view)
  REFUSES instead of falling back. A cut master with a fully EXPLICIT
  penalty stays permitted — nothing else in the narrow phase reads
  elements (the 2026-08-11 review), and the whole interface is on the
  owner by INV-2.

* **F3** — an ``element_owner.get()`` miss (backing element absent from
  every ``PartitionRecord``) silently degraded the exact owner pick to
  the node tally. Now: loud warning always; folded into the F2 refusal
  when an auto knob is active.

* **F5** — ``soft_family_knobs`` ignored the ``edge_edge`` gate while
  ``contact_args`` drops ALL edge knobs when ``edge_edge`` is falsy, so
  a record carrying ``edge_soft`` with ``edge_edge=False`` (reachable
  via H5 round-trip / hand-built records; the authoring layer refuses
  the combination) was refused under partitioning despite emitting no
  ``-edgeSoft`` token serially. The predicate now mirrors the builder.

Emit-time only — no fork build needed.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import ContactRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Real 2-rank two-body pounding fixture (gmsh-meshed) — the F1 lane
# ---------------------------------------------------------------------------


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-3) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _two_box_fem(model_name: str, **contact_kw):
    """Two stacked boxes, one contact interaction, partitioned into 2."""
    with apeGmsh(model_name=model_name, verbose=False) as g:
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
        g.model.sync()
        master = _face_at_z(box1, 1.0)
        slave = _face_at_z(box2, 1.05)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.physical.add(2, [_face_at_z(box1, 0.0)], name="base")
        g.physical.add(2, [_face_at_z(box2, 2.05)], name="top")
        g.constraints.contact("master", "slave", **contact_kw)
        g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    return ops


# ---------------------------------------------------------------------------
# F1 — pattern-borne sp on a contact ghost refuses, named
# ---------------------------------------------------------------------------


def test_pattern_sp_on_contact_ghost_refuses_named(tmp_path):
    # Displacement-driven contact — ordinary authoring: the prescribed
    # motion rides the slave surface, whose nodes ghost onto the master
    # (owner) rank. Pre-fix this emitted the sp on the native rank only
    # and left the owner-rank ghost DOF free.
    fem = _two_box_fem("f1_disp_driven", formulation="nts", kn="auto",
                       name="pound")
    assert len(fem.partitions) == 2
    ops = _ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="slave", dof=3, value=-0.01)
    with pytest.raises(BridgeError) as excinfo:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    # Names the contact lane (not the interface lane it borrowed from).
    assert "g.constraints.contact" in msg
    assert "g.constraints.interface()" not in msg
    assert "ADR 0092 INV-2" in msg
    assert "ghost-declared" in msg
    # The measured consequence + the honest scope statement.
    assert "ADR 0027 INV-2" in msg
    assert "Matrix is Singular Numerically" in msg
    assert "its own project" in msg          # mirroring is out of scope
    # The advertised fixes.
    assert "uncuttable_elements" in msg
    assert "ops.fix" in msg
    assert "flat=True" in msg


def test_pattern_sp_on_contact_ghost_node_target_refuses(tmp_path):
    # Same refusal through the node= target form, naming the node.
    fem = _two_box_fem("f1_disp_node", formulation="nts", kn="auto")
    slave_node = int(fem.nodes.select(pg="slave").ids[0])
    ops = _ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(node=slave_node, dof=3, value=-0.01)
    with pytest.raises(BridgeError, match=rf"\[{slave_node}\]"):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_same_displacement_driven_model_emits_serial(tmp_path):
    # The sanctioned serial escape hatch (flat=True) on the SAME
    # partition-carrying model emits fine — the refusal is a
    # partitioned-lane statement, not a modelling one.
    fem = _two_box_fem("f1_disp_serial", formulation="nts", kn="auto")
    ops = _ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="slave", dof=3, value=-0.01)
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck), flat=True)               # no BridgeError
    text = deck.read_text(encoding="utf-8")
    sp_lines = [ln.strip() for ln in text.splitlines()
                if ln.strip().startswith("sp ")]
    assert sp_lines, "the serial deck must carry the prescribed sp lines"
    assert [ln for ln in text.splitlines()
            if ln.strip().startswith("contact ")]


def test_load_driven_partitioned_contact_unaffected(tmp_path):
    # The S5 shape — load-driven — must keep emitting: pattern loads
    # bucket to a node's primary rank and never touch the ghost sweep.
    fem = _two_box_fem("f1_load_driven", formulation="nts", kn="auto")
    ops = _ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="top", forces=(0.0, 0.0, -2500.0))
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))                          # no BridgeError
    text = deck.read_text(encoding="utf-8")
    contact_lines = [ln.strip() for ln in text.splitlines()
                     if ln.strip().startswith("contact ")]
    assert len(contact_lines) == 1
    assert [ln for ln in text.splitlines()
            if ln.strip().startswith("load ")]


# ---------------------------------------------------------------------------
# F2 / F3 — partial facet → backing-solid resolution, stub-driven
# ---------------------------------------------------------------------------


class _IterableElementsStub(_ElementsStub):
    """The real ``FEMData`` elements composite iterates its element
    groups (``ids`` + ``connectivity`` per type); the planner's facet →
    backing-solid resolution reads exactly that surface. The base stub
    is deliberately NOT iterable (which is how the undecidable-tie tests
    force the node tally); this one is."""

    def __iter__(self):
        return iter(self._pgs.values())


def _cut_master_stub(*, kn, second_facet: str) -> FEMStub:
    """Two-facet master surface whose facet → element map is PARTIAL.

    Facet A ``[10,11,12,13]`` resolves uniquely to solid 101, owned by
    rank 0. Facet B ``[12,13,14,15]`` is the review's probe, selected by
    ``second_facet``:

    * ``"ambiguous"`` — two candidate solids (102 + 103) contain every
      facet-B node, so the facet is unresolvable (the interior-face
      shape);
    * ``"unowned"`` — exactly one solid (102) backs facet B, but 102 is
      absent from every ``PartitionRecord`` (the element-ownership map
      miss).

    Master nodes 10, 11 are unique to rank 0 and 14, 15 unique to
    rank 1, so the master's node-view span is {0, 1} — the unresolved
    facet COULD hide off-rank backing.
    """
    node_ids = [10, 11, 12, 13, 14, 15,
                20, 21, 22, 23, 24, 25, 26, 27,
                30, 31, 34, 35, 36, 37]
    nodes = _NodesStub(
        ids=node_ids,
        coords=[(float(i), 0.0, 0.0) for i in range(len(node_ids))],
        node_pgs={"Base": [20, 21]},
    )
    solid_groups = {
        "SolidsA": _ElementGroupView(
            ids=(101,),
            connectivity=((10, 11, 12, 13, 20, 21, 22, 23),),
        ),
    }
    if second_facet == "ambiguous":
        solid_groups["SolidsB"] = _ElementGroupView(
            ids=(102, 103),
            connectivity=(
                (12, 13, 14, 15, 24, 25, 26, 27),
                (12, 13, 14, 15, 34, 35, 36, 37),
            ),
        )
        rank1_elements = [2, 102, 103]
    elif second_facet == "unowned":
        solid_groups["SolidsB"] = _ElementGroupView(
            ids=(102,),
            connectivity=((12, 13, 14, 15, 24, 25, 26, 27),),
        )
        rank1_elements = [2]        # 102 in NO PartitionRecord
    else:                            # pragma: no cover - test author error
        raise AssertionError(second_facet)
    elements = _IterableElementsStub(
        elem_pgs={
            "Bars": _ElementGroupView(
                ids=(1, 2),
                connectivity=((20, 21), (30, 31)),
            ),
            **solid_groups,
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.set_partitions([
        (0, [10, 11, 12, 13, 20, 21, 22, 23], [1, 101]),
        (1, [12, 13, 14, 15, 24, 25, 26, 27, 30, 31, 34, 35, 36, 37],
         rank1_elements),
    ])
    fem.elements.contacts = [
        ContactRecord(
            kind="contact",
            formulation="nts",
            master_faces=np.asarray(
                [[10, 11, 12, 13], [12, 13, 14, 15]], dtype=np.int64),
            master_nps=4,
            slave_nodes=[30, 31],
            kn=kn,
            name="cutmaster",
        ),
    ]
    fem.elements.contact_planes = []
    return fem


def _stub_ops(fem):
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    return ops


def test_cut_master_auto_with_ambiguous_facet_refuses(tmp_path):
    # Pre-fix: ONE ambiguous facet returned None for the WHOLE surface,
    # the INV-4 backstop silently disengaged, and the node tally (which
    # only refuses the all-shared tie) let the deck emit — the fork then
    # D5.2-skips the off-rank auto-kn facets, a silently PARTIAL
    # interface.
    fem = _cut_master_stub(kn="auto", second_facet="ambiguous")
    with pytest.raises(BridgeError) as excinfo:
        _stub_ops(fem).tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    assert "g.constraints.contact interaction #1 ('cutmaster')" in msg
    assert "cannot be traced to a partition-owned backing solid" in msg
    assert "kn=" in msg and "'auto'" in msg
    assert "span ranks [0, 1]" in msg
    assert "ADR 0092 INV-4" in msg
    assert "fork ADR-78 D5.2" in msg
    assert "explicit numeric penalty" in msg     # the advertised fix
    assert "uncuttable_elements" in msg


def test_cut_master_explicit_kn_with_ambiguous_facet_permitted(tmp_path):
    # Documented: a cut / partially-resolved master with a fully
    # EXPLICIT penalty is allowed — nothing else in the narrow phase
    # reads elements, and the whole interface is on the owner (INV-2).
    fem = _cut_master_stub(kn=1.0e6, second_facet="ambiguous")
    deck = tmp_path / "deck.tcl"
    _stub_ops(fem).tcl(str(deck))               # no BridgeError
    text = deck.read_text(encoding="utf-8")
    contact_lines = [ln.strip() for ln in text.splitlines()
                     if ln.strip().startswith("contact ")]
    assert len(contact_lines) == 1


def test_unowned_backing_element_warns_and_falls_back(tmp_path):
    # F3, no auto knob: the exact pick degrading to the node tally is
    # never silent any more.
    fem = _cut_master_stub(kn=1.0e6, second_facet="unowned")
    deck = tmp_path / "deck.tcl"
    with pytest.warns(UserWarning,
                      match="absent from every\\s+PartitionRecord"):
        _stub_ops(fem).tcl(str(deck))           # emits (explicit kn)
    text = deck.read_text(encoding="utf-8")
    assert [ln for ln in text.splitlines()
            if ln.strip().startswith("contact ")]


def test_unowned_backing_element_with_auto_refuses(tmp_path):
    # F3, auto knob active: same downstream shape as F2 — refuse, do not
    # fall back.
    fem = _cut_master_stub(kn="auto", second_facet="unowned")
    with pytest.raises(BridgeError) as excinfo:
        _stub_ops(fem).tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    assert "cannot be traced to a partition-owned backing solid" in msg
    assert "absent from every PartitionRecord" in msg
    assert "ADR 0092 INV-4" in msg


def test_fully_resolved_uncut_master_unchanged(tmp_path):
    # Negative control: the real gmsh pounding model (facet map fully
    # resolved, master uncut) keeps emitting exactly as S4 shipped it.
    fem = _two_box_fem("f2_uncut_control", formulation="nts", kn="auto")
    deck = tmp_path / "deck.tcl"
    _ops(fem).tcl(str(deck))                    # no BridgeError
    text = deck.read_text(encoding="utf-8")
    contact_lines = [ln.strip() for ln in text.splitlines()
                     if ln.strip().startswith("contact ")]
    assert len(contact_lines) == 1
    assert contact_lines[0].split()[-1] == "auto"


# ---------------------------------------------------------------------------
# F5 — a dormant edge_soft (edge_edge=False) is partition-safe
# ---------------------------------------------------------------------------


def _mortar_record_stub(**record_kw) -> FEMStub:
    """Single-owner mortar interaction, hand-built record (the authoring
    layer refuses edge_* without edge_edge=True, so the F5 combination
    only exists at the record level — H5 round-trips / composed models)."""
    nodes = _NodesStub(
        ids=[10, 11, 12, 13, 30, 31, 32, 33, 20, 21, 40, 41],
        coords=[(float(i), 0.0, 0.0) for i in range(12)],
        node_pgs={"Base": [20, 21]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Bars": _ElementGroupView(
                ids=(1, 2),
                connectivity=((20, 21), (40, 41)),
            ),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.set_partitions([
        (0, [10, 11, 12, 13, 30, 31, 32, 33, 20, 21], [1]),
        (1, [40, 41], [2]),
    ])
    fem.elements.contacts = [
        ContactRecord(
            kind="contact",
            formulation="mortar",
            master_faces=np.asarray([[10, 11, 12, 13]], dtype=np.int64),
            master_nps=4,
            slave_faces=np.asarray([[30, 31, 32, 33]], dtype=np.int64),
            slave_nps=4,
            eps_n=1.0e6,
            name="tie",
            **record_kw,
        ),
    ]
    fem.elements.contact_planes = []
    return fem


def test_dormant_edge_soft_is_not_refused_under_partitioning(tmp_path):
    # edge_soft set, edge_edge=False: contact_args drops the whole edge
    # block, no -edgeSoft token exists — refusing it under partitioning
    # contradicted the serial deck. Emits now.
    fem = _mortar_record_stub(edge_edge=False, edge_soft=0.1)
    deck = tmp_path / "deck.tcl"
    _stub_ops(fem).tcl(str(deck))               # no BridgeError
    text = deck.read_text(encoding="utf-8")
    contact_lines = [ln.strip() for ln in text.splitlines()
                     if ln.strip().startswith("contact ")]
    assert len(contact_lines) == 1
    assert "-edgeSoft" not in text
    assert "-edgeedge" not in text


def test_live_edge_soft_still_refused_under_partitioning(tmp_path):
    # Control: with edge_edge=True the -edgeSoft token IS emitted and
    # the S3 INV-3 refusal must keep firing.
    fem = _mortar_record_stub(edge_edge=True, edge_soft=0.1)
    with pytest.raises(BridgeError) as excinfo:
        _stub_ops(fem).tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    assert "ADR 0092 INV-3" in msg
    assert "edge_soft=" in msg
