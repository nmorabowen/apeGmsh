"""ADR 0068 — constraint-based non-matching tie via ``equationConstraint``.

The ``enforce="equation"`` route reuses ``resolve_tie``'s projection +
shape-function weights but emits an exact EQ_Constraint per tied DOF
(``u_d(slave) = Σ wᵢ·u_d(masterᵢ)``) instead of the penalty
``ASDEmbeddedNodeElement``. These tests lock:

* the ``enforce=`` field + fail-loud validation on the def,
* the per-DOF ``equationConstraint`` expansion math,
* that the equation route allocates NO element tag (EQ_Constraint is a
  domain command, not an element),
* the penalty route is unchanged (regression),
* the fork ``LadrunoProjection`` handler primitive.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.defs.constraints import TieDef, TiedContactDef
from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh._kernel.records._kinds import ConstraintKind as K
from apeGmsh.opensees._internal.build import _emit_one_interpolation
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# --------------------------------------------------------------------------
# def-level: enforce field + validation
# --------------------------------------------------------------------------

def test_tie_def_enforce_defaults_penalty():
    d = TieDef(master_label="A", slave_label="B")
    assert d.enforce == "penalty"


@pytest.mark.parametrize("mode", ["penalty", "penalty_al", "equation"])
def test_tie_def_accepts_known_enforce(mode):
    assert TieDef(master_label="A", slave_label="B", enforce=mode).enforce == mode
    tc = TiedContactDef(master_label="A", slave_label="B", enforce=mode)
    assert tc.enforce == mode


def test_tie_def_unknown_enforce_raises():
    with pytest.raises(ValueError, match="enforce must be one of"):
        TieDef(master_label="A", slave_label="B", enforce="bogus")


@pytest.mark.parametrize("kw", [
    {"rotational": True},
    {"pressure": True},
    {"pressure": True, "stiffness_p": 1.0e12},
])
def test_equation_rejects_penalty_only_knobs(kw):
    # INV-3: the exact route ties translations only — penalty-only options
    # are meaningless and must fail loud, not be silently ignored.
    with pytest.raises(ValueError, match="penalty-only|enforce='equation'"):
        TieDef(master_label="A", slave_label="B", enforce="equation", **kw)


# --------------------------------------------------------------------------
# record-level + expansion math
# --------------------------------------------------------------------------

def test_interpolation_record_carries_enforce_default():
    rec = InterpolationRecord(kind=K.TIE, slave_node=1, master_nodes=[2, 3, 4])
    assert rec.enforce == "penalty"


def _emit(rec: InterpolationRecord, tags: TagAllocator | None = None):
    e = RecordingEmitter()
    _emit_one_interpolation(e, rec, tags or TagAllocator())
    return e


def test_equation_route_emits_one_equationConstraint_per_dof():
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=7, master_nodes=[2, 3, 4],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3],
        enforce="equation",
    )
    calls = _emit(rec).calls
    eqc = [c for c in calls if c[0] == "equationConstraint"]
    assert len(eqc) == 3                       # one per tied DOF
    assert not [c for c in calls if c[0] == "element"]   # not an element

    # dof 1: 1.0·u1(7) − 0.5·u1(2) − 0.3·u1(3) − 0.2·u1(4) = 0
    cnode, cdof, ccoef, retained = eqc[0][1]
    assert (cnode, cdof, ccoef) == (7, 1, 1.0)
    assert retained == ((2, 1, -0.5), (3, 1, -0.3), (4, 1, -0.2))
    # dof 2 reuses the same weights on component 2
    _, cdof2, _, retained2 = eqc[1][1]
    assert cdof2 == 2
    assert retained2 == ((2, 2, -0.5), (3, 2, -0.3), (4, 2, -0.2))


def test_equation_route_allocates_no_element_tag():
    # The equation route must NOT consume an element tag (it is a domain
    # command), so the element-tag stream is untouched across the emit.
    tags = TagAllocator()
    first = tags.allocate("element")
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=7, master_nodes=[2, 3, 4],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3],
        enforce="equation",
    )
    _emit_one_interpolation(RecordingEmitter(), rec, tags)
    second = tags.allocate("element")
    assert second == first + 1                 # nothing allocated in between


def test_equation_route_accepts_quad4_face_arity():
    # 4 masters would be fine for the embedded guard too, but the point is
    # the equation route never invokes the 3/4-Rnode guard — try a tri6
    # (6 masters), which the penalty embeddedNode path would reject.
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=1,
        master_nodes=[2, 3, 4, 5, 6, 7],
        weights=np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05]),
        dofs=[1, 2, 3], enforce="equation",
    )
    eqc = [c for c in _emit(rec).calls if c[0] == "equationConstraint"]
    assert len(eqc) == 3
    assert len(eqc[0][1][3]) == 6              # 6 retained terms


def test_penalty_route_still_emits_embeddedNode():
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=1, master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]), dofs=[1, 2, 3],
        enforce="penalty",
    )
    calls = _emit(rec).calls
    assert [c for c in calls if c[0] == "embeddedNode"]
    assert not [c for c in calls if c[0] == "equationConstraint"]


def test_equation_route_drops_zero_weight_masters():
    # OpenSees rejects any zero rcoef (EQ_Constraint.cpp:98) and aborts the
    # whole line — a slave on a master face edge legitimately has N_i=0, so
    # zero-weight masters MUST be filtered, not emitted.
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=7, master_nodes=[2, 3, 4],
        weights=np.array([0.6, 0.4, 0.0]), dofs=[1], enforce="equation",
    )
    eqc = [c for c in _emit(rec).calls if c[0] == "equationConstraint"]
    assert len(eqc) == 1
    _, _, _, retained = eqc[0][1]
    assert retained == ((2, 1, -0.6), (3, 1, -0.4))   # node 4 (w=0) dropped


def test_equation_route_all_zero_weights_raises():
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=7, master_nodes=[2, 3],
        weights=np.array([0.0, 0.0]), dofs=[1], enforce="equation",
    )
    with pytest.raises(ValueError, match="all interpolation weights"):
        _emit(rec)


def test_equation_route_rejects_rotational_dofs():
    # The equation route is translations-only (1..3); a rotational DOF in
    # `dofs` is meaningless and OpenSees would fail late.
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=7, master_nodes=[2, 3, 4],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3, 4, 5, 6],
        enforce="equation",
    )
    with pytest.raises(ValueError, match="translations only|out of range"):
        _emit(rec)


def test_equation_route_rejects_self_reference():
    # Slave is also one of its own master face nodes → self-referential EQ.
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=2, master_nodes=[2, 3, 4],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1], enforce="equation",
    )
    with pytest.raises(ValueError, match="self-referential|own"):
        _emit(rec)


def test_fem_has_equation_ties_detector():
    from apeGmsh.opensees.apesees import _fem_has_equation_ties

    class _SC:
        def __init__(self, recs):
            self._r = recs

        def interpolations(self):
            return iter(self._r)

    class _Elems:
        def __init__(self, sc):
            self.constraints = sc

    class _Fem:
        def __init__(self, sc):
            self.elements = _Elems(sc)

    def rec(enforce):
        return InterpolationRecord(
            kind=K.TIE, slave_node=1, master_nodes=[2, 3, 4],
            weights=np.array([0.3, 0.3, 0.4]), enforce=enforce,
        )

    assert _fem_has_equation_ties(_Fem(_SC([rec("penalty")]))) is False
    assert _fem_has_equation_ties(
        _Fem(_SC([rec("penalty"), rec("equation")]))) is True


def test_enforce_survives_h5_record_roundtrip():
    # Persistence (adversarial finding): the enforce field must survive the
    # model.h5 neutral-zone round-trip, else a saved+reloaded equation tie
    # silently downgrades to penalty (handler picks Transformation → drop).
    h5py = pytest.importorskip("h5py")
    import io
    from apeGmsh.mesh._record_h5 import (
        interpolation_payload_dtype, make_record_dtype,
    )
    from apeGmsh.mesh._femdata_h5_io import (
        _encode_interpolation, _decode_interpolation,
    )

    dt = make_record_dtype(interpolation_payload_dtype())

    def _roundtrip(rec):
        rows = np.empty(1, dtype=dt)
        rows[0] = ("node", str(rec.slave_node), "tie",
                   _encode_interpolation(rec))
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("r", data=rows)
        buf.seek(0)
        with h5py.File(buf, "r") as f:
            out = f["r"][:]
        return _decode_interpolation(out[0], InterpolationRecord)

    eq = InterpolationRecord(
        kind=K.TIE, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3], enforce="equation",
    )
    assert _roundtrip(eq).enforce == "equation"

    # default penalty lane still round-trips as penalty
    pen = InterpolationRecord(
        kind=K.TIE, slave_node=5, master_nodes=[1, 2, 3],
        weights=np.array([0.4, 0.4, 0.2]), dofs=[1, 2, 3],
    )
    assert _roundtrip(pen).enforce == "penalty"


def test_penalty_al_route_not_implemented():
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=1, master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]), dofs=[1, 2, 3],
        enforce="penalty_al",
    )
    with pytest.raises(NotImplementedError, match="penalty_al"):
        _emit(rec)


# --------------------------------------------------------------------------
# LadrunoProjection handler primitive
# --------------------------------------------------------------------------

def test_ladruno_projection_emits_minimal():
    from apeGmsh.opensees.analysis.constraint_handler import LadrunoProjection
    e = RecordingEmitter()
    LadrunoProjection()._emit(e, 0)
    assert e.calls == [("constraints", ("LadrunoProjection",), {})]


def test_ladruno_projection_flags():
    from apeGmsh.opensees.analysis.constraint_handler import LadrunoProjection
    e = RecordingEmitter()
    LadrunoProjection(verbose=True, project_ics=True, ic_tol=1.0e-6)._emit(e, 0)
    assert e.calls == [(
        "constraints",
        ("LadrunoProjection", "-verbose", "-projectICs", "-icTol", 1.0e-6),
        {},
    )]


def test_ladruno_projection_ic_tol_requires_project_ics():
    from apeGmsh.opensees.analysis.constraint_handler import LadrunoProjection
    with pytest.raises(ValueError, match="ic_tol"):
        LadrunoProjection(ic_tol=1.0e-6)


# --------------------------------------------------------------------------
# LIVE end-to-end (INV-1): the emitted equationConstraint actually enforces
# u_d(slave) = Σ wᵢ·u_d(masterᵢ) in a real openseespy static solve. Skipped
# when openseespy is not importable (e.g. a plain CI Python).
# --------------------------------------------------------------------------

def test_equation_tie_enforced_in_live_solve():
    ops = pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter
    from apeGmsh.opensees._internal.build import _emit_equation_tie

    # Construct the live emitter FIRST (it wipes on construction); its
    # ._ops is the openseespy singleton we build the model through.
    e = LiveOpsEmitter(wipe=True)

    ops.model("basic", "-ndm", 3, "-ndf", 3)
    masters = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)}
    anchors = {11: (0.0, 0.0, 0.0), 12: (1.0, 0.0, 0.0), 13: (0.0, 1.0, 0.0)}
    coords = {**masters, **anchors, 4: (0.3, 0.3, 0.0)}
    for t, (x, y, z) in coords.items():
        ops.node(t, x, y, z)
    for t in anchors:
        ops.fix(t, 1, 1, 1)

    # Each master gets a 3D elastic spring to a fixed anchor → real free
    # DOFs with stiffness; loads make them displace by known amounts.
    ops.uniaxialMaterial("Elastic", 1, 100.0)
    for ele, (m, a) in enumerate([(1, 11), (2, 12), (3, 13)], start=1):
        ops.element("zeroLength", ele, m, a, "-mat", 1, 1, 1,
                    "-dir", 1, 2, 3)

    # The slave (node 4) is tied to the master triangle through the bridge's
    # own expansion → live ops.equationConstraint. Arbitrary partition-of-
    # unity weights.
    w = np.array([0.5, 0.3, 0.2])
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=4, master_nodes=[1, 2, 3],
        weights=w, dofs=[1, 2, 3], enforce="equation",
    )
    _emit_equation_tie(e, rec)

    # EQ_Constraint needs Lagrange/Penalty/LadrunoProjection (NOT
    # Transformation). Lagrange = exact for this implicit static solve.
    ops.constraints("Lagrange")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Linear")
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(1, 10.0, 0.0, 0.0)
    ops.load(2, 0.0, 5.0, 0.0)
    ops.load(3, 0.0, 0.0, 7.0)
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0

    um = {m: [ops.nodeDisp(m, d) for d in (1, 2, 3)] for m in (1, 2, 3)}
    u4 = [ops.nodeDisp(4, d) for d in (1, 2, 3)]
    for di in range(3):
        expected = sum(w[i] * um[m][di] for i, m in enumerate((1, 2, 3)))
        assert u4[di] == pytest.approx(expected, abs=1.0e-9)
    # sanity: the masters actually moved (non-trivial solve)
    assert any(abs(um[m][di]) > 1e-6 for m in (1, 2, 3) for di in range(3))

    ops.wipe()
