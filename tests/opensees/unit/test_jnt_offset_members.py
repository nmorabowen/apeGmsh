"""``jnt_offset`` where a frame member may be meshed into several elements.

ETABS puts a frame object's end offsets at the object's two ends, and the
frame object is the gmsh curve (maintainer rulings 2026-10-26 on #2470
and #2551). An end-sensitive offset (an end length, an axial component,
unequal ends) is split over each source curve: the curve's i-end element
takes the i-offset, its j-end element the j-offset, interior nodes only
the transverse part. A curved curve carrying end-sensitive offsets, a
curve whose elements run both ways, and a closed curve refuse.

Without each element's source curve (a snapshot with no entity stream:
a pre-2.36 model.h5, a FEM not extracted from gmsh, and every FEMStub
built without ``entity=``) a meshed member cannot be told from a
continuous girder, so the bridge keeps #2470's refusal: an end-sensitive
offset on any collinear run of a transform's elements. What touches the
run (a secondary beam, a spring declared ``pg=`` or ``nodes=``) does not
change that verdict. One-element members, chains that turn a corner, and
a transverse shift equal at both ends keep working either way, byte for
byte.

The live oracle (an n-element member against a hand-built one-element
member) is ``tests/opensees/integration/test_jnt_offset_live.py``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.joint_offset import InsertionPoint

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

#: Cardinal 8 on a 0.6 m deep beam (axis_2 = z, vecxz = +Z): the axis
#: hangs 0.3 m below the joints; 0.3 m rigid zones at both ends.
RULE = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                      end_offset_i=0.3, end_offset_j=0.3)
SHIFT_ONLY = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z")
REFUSAL = r"meet at node \d+ turning .* deg, outside the 30-150 deg.*carries no source curves"


def stub(coords: dict[int, tuple[float, float, float]],
         groups: dict[str, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]],
         entity: dict[int, int] | None = None,
         module_label: dict[int, str] | None = None,
         ) -> FEMStub:
    """``entity`` maps element id -> source curve; ``None`` is a snapshot
    with no entity stream."""
    ids = sorted(coords)
    return FEMStub(
        nodes=_NodesStub(ids=ids, coords=[coords[n] for n in ids],
                         node_pgs={"Base": [ids[0]]}),
        elements=_ElementsStub(elem_pgs={
            name: _ElementGroupView(ids=eids, connectivity=conn)
            for name, (eids, conn) in groups.items()},
            entity=entity, module_label=module_label))


def bridge(fem: FEMStub, rule: Any = RULE, pg: str = "M") -> apeSees:
    ops = apeSees(cast("object", fem), default_orientation=None,
                  element_tags="fem")   # element tag = FEM element id
    ops.model(ndm=3, ndf=6)
    ops.element.elasticBeamColumn(
        pg=pg, transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=rule),
        A=0.01, E=200e9, Iz=1e-4, Iy=2e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1,) * 6)
    return ops


def element_offsets(rec: RecordingEmitter) -> dict[int, tuple[float, ...]]:
    """Beam element id -> the ``-jntOffset`` tail of the transform it uses."""
    lines = {args[1]: tuple(kw.get("jnt_offset", ()))
             for name, args, kw in rec.calls if name == "geomTransf"}
    return {int(args[1]): lines[int(args[-1])]
            for name, args, _kw in rec.calls
            if name == "element" and args[0] == "elasticBeamColumn"}


#: A girder 1 -> 2 -> 3 along +X at z = 3 (two elements, collinear at 2).
GIRDER = {1: (0.0, 0.0, 3.0), 2: (2.0, 0.0, 3.0), 3: (4.0, 0.0, 3.0)}


def test_a_collinear_two_element_run_refuses() -> None:
    fem = stub(GIRDER, {"M": ((101, 102), ((1, 2), (2, 3)))})
    with pytest.raises(BridgeError, match=REFUSAL):
        bridge(fem).build().emit(RecordingEmitter())


@pytest.mark.parametrize("secondary", [False, True], ids=["alone", "secondary_beam"])
def test_girder_with_a_secondary_beam_at_mid_span_refuses_alike(secondary: bool) -> None:
    """The reviewer's repro: a secondary beam framing in at mid-span used
    to split the girder and put rigid zones there. Now the verdict is the
    same with or without it: the collinear girder refuses."""
    coords = dict(GIRDER)
    groups = {"M": ((101, 102), ((1, 2), (2, 3)))}
    if secondary:
        coords[4] = (2.0, 3.0, 3.0)
        groups["S"] = ((201,), ((2, 4),))
    ops = bridge(stub(coords, groups))
    if secondary:
        ops.element.elasticBeamColumn(
            pg="S", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    with pytest.raises(BridgeError, match=r"elements 101 and 102.*node 2"):
        ops.build().emit(RecordingEmitter())


@pytest.mark.parametrize("declared", ["pg", "nodes"])
def test_spring_at_an_interior_node_gives_one_verdict(declared: str) -> None:
    """The reviewer's repro: a ZeroLength spring at the girder's interior
    node split the member when declared ``pg=`` but not ``nodes=``."""
    coords = dict(GIRDER)
    coords[6] = (2.0, 0.0, 3.0)
    groups = {"M": ((101, 102), ((1, 2), (2, 3)))}
    if declared == "pg":
        groups["S"] = ((301,), ((6, 2),))
    ops = bridge(stub(coords, groups))
    mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e8)
    dirs = tuple(ZeroLengthMatDir(material=mat, dof=d) for d in (1, 2, 3))
    if declared == "pg":
        ops.element.ZeroLength(pg="S", mat_dirs=dirs)
    else:
        ops.element.ZeroLength(nodes=(6, 2), mat_dirs=dirs)
    with pytest.raises(BridgeError, match=r"elements 101 and 102.*node 2"):
        ops.build().emit(RecordingEmitter())


def test_two_collinear_curves_meeting_at_a_column_refuse() -> None:
    """Two one-element members meeting at a column are the same mesh as
    one member in two elements; only the curves can tell, and this
    snapshot has none."""
    coords = dict(GIRDER)
    coords[5] = (2.0, 0.0, 0.0)
    ops = bridge(stub(coords, {"M": ((101, 102), ((1, 2), (2, 3))),
                               "C": ((401,), ((5, 2),))}))
    ops.element.elasticBeamColumn(
        pg="C", transf=ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    with pytest.raises(BridgeError, match=REFUSAL):
        ops.build().emit(RecordingEmitter())


def test_two_collinear_members_on_separate_transforms_are_accepted() -> None:
    """The documented way out: one transform per member."""
    fem = stub(GIRDER, {"A": ((101,), ((1, 2),)), "B": ((102,), ((2, 3),))})
    ops = bridge(fem, pg="A")
    ops.element.elasticBeamColumn(
        pg="B", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=RULE),
        A=0.01, E=200e9, Iz=1e-4, Iy=2e-4, G=80e9, J=1e-4)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    got = element_offsets(rec)
    assert got[101] == got[102] == pytest.approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))


@pytest.mark.parametrize("rule", [
    RULE, ((0.25, 0.0, 0.0), (-0.4, 0.0, 0.0)), ((0.0, 0.1, 0.0), (0.0, 0.2, 0.0)),
], ids=["end_lengths", "constant_axial", "unequal_transverse"])
def test_end_sensitive_offsets_refuse_on_a_run(rule: Any) -> None:
    fem = stub(GIRDER, {"M": ((101, 102), ((1, 2), (2, 3)))})
    with pytest.raises(BridgeError, match=REFUSAL):
        bridge(fem, rule=rule).build().emit(RecordingEmitter())


@pytest.mark.parametrize("rule", [SHIFT_ONLY, ((0.0, 0.1, -0.3), (0.0, 0.1, -0.3))],
                         ids=["cardinal_shift", "constant_transverse"])
def test_an_equal_transverse_shift_stays_on_every_element_of_a_run(rule: Any) -> None:
    fem = stub(GIRDER, {"M": ((101, 102), ((1, 2), (2, 3)))})
    rec = RecordingEmitter()
    bridge(fem, rule=rule).build().emit(rec)
    got = element_offsets(rec)
    want = (0.0, 0.0, -0.3) * 2 if rule is SHIFT_ONLY else (0.0, 0.1, -0.3) * 2
    assert got[101] == got[102] == pytest.approx(want)


def _chain(points: list[tuple[float, float, float]],
           curve: int | None = None) -> FEMStub:
    """A chain of line elements 101.. through ``points`` (nodes 1..n);
    ``curve``: every element's source curve (``None``: no stream)."""
    coords = {k + 1: p for k, p in enumerate(points)}
    conn = tuple((k + 1, k + 2) for k in range(len(points) - 1))
    eids = tuple(101 + k for k in range(len(conn)))
    return stub(coords, {"M": (eids, conn)},
                entity=None if curve is None else dict.fromkeys(eids, curve))


def test_a_straight_member_with_rounded_coordinates_refuses() -> None:
    """The reviewer's repro: a straight member along (1, 0.3)/sqrt(1.09),
    3 elements of 1.7, coordinates rounded to the mm (an ETABS or DXF
    import). Its nodes are off a line by far more than round-off, so an
    exact-collinearity test let every element take end zones."""
    import math
    u = (1.0 / math.sqrt(1.09), 0.3 / math.sqrt(1.09))
    pts = [(round(1.7 * k * u[0], 3), round(1.7 * k * u[1], 3), 3.0) for k in range(4)]
    with pytest.raises(BridgeError, match=REFUSAL):
        bridge(_chain(pts)).build().emit(RecordingEmitter())


def test_a_faceted_arc_refuses() -> None:
    """R = 50 m, 3 deg per element: one curved member, not 4 members."""
    import math
    pts = [(50.0 * math.cos(math.radians(3 * k)), 50.0 * math.sin(math.radians(3 * k)), 3.0)
           for k in range(5)]
    with pytest.raises(BridgeError, match=r"turning 3 deg, outside the 30-150 deg"):
        bridge(_chain(pts)).build().emit(RecordingEmitter())


def test_a_20_deg_kink_refuses_naming_the_threshold() -> None:
    import math
    t = math.radians(20.0)
    pts = [(0.0, 0.0, 3.0), (4.0, 0.0, 3.0), (4.0 + 4.0 * math.cos(t), 4.0 * math.sin(t), 3.0)]
    with pytest.raises(BridgeError, match=r"node 2 turning 20 deg, outside the 30-150 deg that marks a frame corner"):
        bridge(_chain(pts)).build().emit(RecordingEmitter())


def test_a_45_deg_knee_is_a_corner() -> None:
    import math
    t = math.radians(45.0)
    pts = [(0.0, 0.0, 3.0), (4.0, 0.0, 3.0), (4.0 + 4.0 * math.cos(t), 4.0 * math.sin(t), 3.0)]
    rec = RecordingEmitter()
    bridge(_chain(pts)).build().emit(rec)
    got = element_offsets(rec)
    assert got[101] == pytest.approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))
    c = math.cos(t) * 0.3
    assert got[102] == pytest.approx((c, c, -0.3, -c, -c, -0.3))


def test_a_branch_does_not_hide_a_straight_pass_through() -> None:
    """The reviewer's repro: girder 1-2-3 straight along X plus a secondary
    beam 2-4 along Y, all on one transform. The girder runs straight
    through node 2, so it may be one member; it refuses."""
    coords = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0), 3: (8.0, 0.0, 3.0), 4: (4.0, 4.0, 3.0)}
    fem = stub(coords, {"M": ((101, 102, 103), ((1, 2), (2, 3), (2, 4)))})
    with pytest.raises(BridgeError, match=r"elements 101 and 102.*node 2 turning 0 deg"):
        bridge(fem).build().emit(RecordingEmitter())


def test_a_hairpin_refuses() -> None:
    """Element 102 folds back over 101 (turn 180 deg): no frame corner."""
    coords = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0), 3: (2.0, 0.0, 3.0)}
    fem = stub(coords, {"M": ((101, 102), ((1, 2), (2, 3)))})
    with pytest.raises(BridgeError, match=r"turning 180 deg, outside the 30-150 deg"):
        bridge(fem).build().emit(RecordingEmitter())


def test_a_y_node_at_120_deg_is_accepted() -> None:
    """Three elements meeting at 120 deg pairwise: every pair is a corner,
    so each element is its own member with end zones at both ends."""
    import math
    def arm(deg: float) -> tuple[float, float, float]:
        t = math.radians(deg)
        return (4.0 * math.cos(t), 4.0 * math.sin(t), 3.0)
    coords = {1: (0.0, 0.0, 3.0), 2: arm(0.0), 3: arm(120.0), 4: arm(240.0)}
    fem = stub(coords, {"M": ((101, 102, 103), ((1, 2), (1, 3), (1, 4)))})
    rec = RecordingEmitter()
    bridge(fem).build().emit(rec)
    got = element_offsets(rec)
    for eid, deg in ((101, 0.0), (102, 120.0), (103, 240.0)):
        t = math.radians(deg)
        c, s_ = 0.3 * math.cos(t), 0.3 * math.sin(t)
        assert got[eid] == pytest.approx((c, s_, -0.3, -c, -s_, -0.3), abs=1e-12)


def test_a_corner_chain_is_accepted_with_each_element_its_own_member() -> None:
    """X then Y at node 2: the corner ends each member, so each element
    takes the full rule on its own axes (left/right cardinal 9 turns with
    the beam), end zones at both its ends."""
    coords = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0), 3: (4.0, 5.0, 3.0)}
    rule = InsertionPoint(cardinal_point=9, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.25, end_offset_j=0.25)
    rec = RecordingEmitter()
    bridge(stub(coords, {"M": ((101, 102), ((1, 2), (2, 3)))}), rule=rule).build().emit(rec)
    got = element_offsets(rec)
    # local 3 = 1 x 2 (2 = +Z): -Y on the X beam, +X on the Y beam; the
    # centroid sits at minus (d/2 along 2, b/2 along 3).
    assert got[101] == pytest.approx((0.25, 0.15, -0.3, -0.25, 0.15, -0.3))
    assert got[102] == pytest.approx((-0.15, 0.25, -0.3, -0.15, -0.25, -0.3))


def test_one_element_members_are_accepted() -> None:
    fem = stub({1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0)}, {"M": ((101,), ((1, 2),))})
    rec = RecordingEmitter()
    bridge(fem).build().emit(rec)
    assert element_offsets(rec)[101] == pytest.approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))


@pytest.mark.parametrize("staged", [False, True], ids=["flat", "staged"])
def test_rank_whose_only_content_is_an_end_zone_load_emits_it(staged: bool) -> None:
    """A load wholly on the i-end zone gives rank 1 no eleLoad, only the
    joint load; the partitioned pattern (and, staged, its bracket
    pre-check) must still open for it."""
    from tests.opensees.unit.test_jnt_offset import _loaded_beam

    rec = RecordingEmitter()
    _loaded_beam(((0.3, 0.0), (-0.3, 0.0)), partitioned=True, staged=staged,
                 params={"wx": 0.0, "wy": -10.0, "wz": 0.0,
                         "aL": 0.0, "bL": 0.05}).build().emit(rec)
    rank = None
    seen = []
    for name, args, _kw in rec.calls:
        if name == "partition_open":
            rank = int(args[0])
        elif name == "partition_close":
            rank = None
        elif name in ("load", "eleLoad"):
            seen.append((rank, name, args[0]))
    assert seen == [(1, "load", 1)]


@pytest.mark.parametrize("staged", [False, True], ids=["flat", "staged"])
def test_h5_replay_writes_the_same_pattern_block(staged: bool, tmp_path: Path) -> None:
    """The bridge writes the end-zone joint loads before the eleLoads, the
    order an H5 replay writes a pattern, so the replayed deck matches."""
    from apeGmsh.opensees import OpenSeesModel

    ops = _real_corner(staged)
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    h5 = tmp_path / "m.h5"
    ops.h5(str(h5))
    replayed = OpenSeesModel.from_h5(str(h5)).build("tcl")

    def block(text: str) -> list[str]:
        lines = [ln.strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln.startswith(("load ", "eleLoad "))]
    want = block(deck.read_text(encoding="utf-8"))
    assert [ln.split()[0] for ln in want] == ["load"] * 4 + ["eleLoad"] * 2
    assert block(replayed) == want


def _real_corner(staged: bool) -> apeSees:
    """Two one-element members meeting at a corner, (0,0)-(2,0)-(2,2), on
    a real ``FEMData`` (``from_h5`` reloads the neutral zone, which
    refuses stub-only files), under a from_model line load with 0.3 m end
    zones on both."""
    import numpy as np

    from apeGmsh._kernel.records._loads import ElementLoadRecord
    from tests.opensees.unit.test_beam_uniform_load import _beam_fem_data

    fem = _beam_fem_data(ndm=2, loads=[
        ElementLoadRecord(element_id=e, load_type="beamUniform",
                          params={"wx": 0.0, "wy": -10.0, "wz": 0.0},
                          pattern="dl") for e in (10, 20)])
    fem.nodes.coords[2] = np.array([2.0, 2.0, 0.0])   # turn the second span
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(
        pg="Beams", transf=ops.geomTransf.Linear(jnt_offset=InsertionPoint(
            depth=0.6, width=0.3, end_offset_i=0.3, end_offset_j=0.3)),
        A=0.01, E=200e9, Iz=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    if staged:
        with ops.stage(name="s1") as st:
            with st.pattern(series=ops.timeSeries.Linear()) as p:
                p.from_model("dl")
            st.analysis(
                system=ops.system.BandGeneral(), numberer=ops.numberer.RCM(),
                constraints=ops.constraints.Plain(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                algorithm=ops.algorithm.Linear(), analysis=ops.analysis.Static())
            st.run(n_increments=1, dt=1.0)
    else:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("dl")
    return ops


# ---------------------------------------------------------------------------
# Curve members: the snapshot carries each element's source curve
# ---------------------------------------------------------------------------

LENGTH = 4.0
CURVE = 7


def member_fem(n: int, *, reverse: bool = False, ndm: int = 3,
               loads: dict | None = None, partitions: Any = None,
               curve: int | None = CURVE) -> FEMStub:
    """A gmsh-like line (0,0,3)-(4,0,3) (2-D: y = 0, z = 0) meshed into
    ``n`` elements of source curve ``curve``: nodes 1..n+1 along +X,
    elements 101.. in curve order (``reverse``: the curve runs -X, so
    every element's node order flips, as gmsh writes it)."""
    from apeGmsh._kernel.record_sets import ElementLoadSet
    from apeGmsh._kernel.records._loads import ElementLoadRecord

    z = 3.0 if ndm == 3 else 0.0
    coords = [(LENGTH * k / n, 0.0, z) for k in range(n + 1)]
    ids = list(range(1, n + 2))
    conn = [(k + 1, k + 2) for k in range(n)]
    if reverse:
        conn = [(b, a) for a, b in reversed(conn)]
    eids = tuple(101 + k for k in range(n))
    nodes = _NodesStub(ids=ids, coords=coords,
                       node_pgs={"Base": [1], "Tip": [n + 1]})
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(
        elem_pgs={"M": _ElementGroupView(ids=eids, connectivity=tuple(conn))},
        entity=None if curve is None else dict.fromkeys(eids, curve)))
    if loads is not None:
        fem.elements.loads = ElementLoadSet([
            ElementLoadRecord(element_id=e, load_type="beamUniform",
                              params=dict(loads), pattern="dl")
            for e in eids])
    if partitions is not None:
        fem.set_partitions(partitions)
    return fem


def member_bridge(fem: FEMStub, *, ndm: int = 3, rule: Any = RULE,
                  loaded: bool = False, staged: bool = False,
                  tip: tuple[float, ...] | None = None) -> apeSees:
    """The member under ``rule`` (fixed at node 1); ``loaded`` applies the
    ``"dl"`` element loads, ``tip`` a nodal load on PG ``"Tip"``."""
    ops = apeSees(cast("object", fem), default_orientation=None,
                  element_tags="fem")   # element tag = FEM element id
    ndf = 6 if ndm == 3 else 3
    ops.model(ndm=ndm, ndf=ndf)
    props: dict[str, float] = dict(A=0.01, E=200e9, Iz=1e-4)
    if ndm == 3:
        props.update(Iy=2e-4, G=80e9, J=1e-4)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=rule)
    else:
        transf = ops.geomTransf.Linear(jnt_offset=rule)
    ops.element.elasticBeamColumn(pg="M", transf=transf, **props)
    ops.fix(pg="Base", dofs=(1,) * ndf)
    if not (loaded or tip):
        return ops

    def fill(p: Any) -> None:
        if loaded:
            p.from_model("dl")
        if tip:
            p.load(pg="Tip", forces=tip)
    if staged:
        with ops.stage(name="s1") as st:
            with st.pattern(series=ops.timeSeries.Linear()) as p:
                fill(p)
            st.analysis(
                system=ops.system.BandGeneral(), numberer=ops.numberer.RCM(),
                constraints=ops.constraints.Plain(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                algorithm=ops.algorithm.Linear(), analysis=ops.analysis.Static())
            st.run(n_increments=1, dt=1.0)
    else:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            fill(p)
    return ops


def _approx(t: tuple[float, ...]) -> Any:
    return pytest.approx(t, abs=1e-12)


def _emit(ops: apeSees) -> RecordingEmitter:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return rec


@pytest.mark.parametrize("n", [1, 2, 5])
def test_end_offsets_sit_on_the_curve_end_elements_only(n: int) -> None:
    """The #2470 review repro (34f0bf3c): n = 5 gave all five elements
    ``0.3 0 -0.3 -0.3 0 -0.3`` (each 0.8 m element 75 % rigid). By hand:
    the i-end element takes the i-offset, the j-end element the j-offset,
    every interior node only the 0.3 m hang (cardinal 8 on 0.6 m)."""
    got = element_offsets(_emit(member_bridge(member_fem(n))))
    hang = (0.0, 0.0, -0.3)
    want = {}
    for k in range(n):
        o_i = (0.3, 0.0, -0.3) if k == 0 else hang
        o_j = (-0.3, 0.0, -0.3) if k == n - 1 else hang
        want[101 + k] = (*o_i, *o_j)
    assert got.keys() == want.keys()
    for eid, tail in want.items():
        assert got[eid] == _approx(tail), eid


def test_reversed_curve_puts_the_i_offset_at_its_own_start() -> None:
    """The curve runs -X: its i-end is the node at x = 4, where the
    element order and the node order of the mesh start."""
    rule = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.5, end_offset_j=0.2)
    got = element_offsets(_emit(member_bridge(member_fem(5, reverse=True), rule=rule)))
    first, last = 101, 105       # 101 runs 6 -> 5 (x = 4 -> 3.2); 105 ends at x = 0
    assert got[first] == _approx((-0.5, 0.0, -0.3, 0.0, 0.0, -0.3))
    assert got[last] == _approx((0.0, 0.0, -0.3, 0.2, 0.0, -0.3))
    for eid in (102, 103, 104):
        assert got[eid] == _approx((0.0, 0.0, -0.3, 0.0, 0.0, -0.3))


def test_constant_pair_with_end_lengths_fans_out_over_the_curve() -> None:
    """A constant pair is one line on one-element members; split over a
    meshed curve it becomes one line per distinct tail, the first under
    the spec's own tag."""
    rec = _emit(member_bridge(member_fem(4), rule=((0.25, 0.1, 0.0), (-0.4, 0.1, 0.0))))
    got = element_offsets(rec)
    assert got[101] == _approx((0.25, 0.1, 0.0, 0.0, 0.1, 0.0))
    assert got[102] == got[103] == _approx((0.0, 0.1, 0.0, 0.0, 0.1, 0.0))
    assert got[104] == _approx((0.0, 0.1, 0.0, -0.4, 0.1, 0.0))
    tags = [int(a[1]) for name, a, _kw in rec.calls if name == "geomTransf"]
    assert tags == [1, 2, 3]


def test_unequal_transverse_ends_interpolate_along_the_curve() -> None:
    """No end lengths, transverse ends 0.1 and 0.3 along +Y: the offset
    axis is the straight line between them, so node x carries
    0.1 + 0.2 x / 4."""
    got = element_offsets(_emit(member_bridge(
        member_fem(4), rule=((0.0, 0.1, 0.0), (0.0, 0.3, 0.0)))))
    y = [0.1, 0.15, 0.2, 0.25, 0.3]
    for k in range(4):
        assert got[101 + k] == _approx((0.0, y[k], 0.0, 0.0, y[k + 1], 0.0))


def test_equal_end_offsets_stay_on_every_element_through_one_line() -> None:
    rec = _emit(member_bridge(member_fem(3), rule=((0.0, 0.1, -0.3), (0.0, 0.1, -0.3))))
    assert len([c for c in rec.calls if c[0] == "geomTransf"]) == 1
    assert set(element_offsets(rec).values()) == {(0.0, 0.1, -0.3, 0.0, 0.1, -0.3)}


def test_2d_member_splits_its_end_lengths() -> None:
    rule = InsertionPoint(cardinal_point=8, depth=0.6, end_offset_i=0.2,
                          end_offset_j=0.4)
    got = element_offsets(_emit(member_bridge(member_fem(3, ndm=2), ndm=2, rule=rule)))
    assert got[101] == _approx((0.2, -0.3, 0.0, -0.3))
    assert got[102] == _approx((0.0, -0.3, 0.0, -0.3))
    assert got[103] == _approx((0.0, -0.3, -0.4, -0.3))


def _arc(deg_per_element: float, n: int) -> list[tuple[float, float, float]]:
    import math
    r = 50.0 if deg_per_element < 10 else 4.0
    return [(r * math.cos(math.radians(deg_per_element * k)),
             r * math.sin(math.radians(deg_per_element * k)), 3.0) for k in range(n + 1)]


@pytest.mark.parametrize("deg", [3.0, 45.0], ids=["faceted_arc", "coarse_arc"])
def test_a_curved_curve_with_end_offsets_refuses_naming_it(deg: float) -> None:
    """Maintainer ruling (2026-10-26, #2551 Q3). The 45 deg coarse arc is
    the case #2470 accepted (each turn read as a frame corner); now one
    curve, it refuses."""
    with pytest.raises(BridgeError, match=(
            r"curve 7 \(elements \[101, 102, 103\]\) is not straight.*"
            r"split the curve into straight curves, or drop the end offsets")):
        member_bridge(_chain(_arc(deg, 3), curve=CURVE)).build().emit(RecordingEmitter())


def test_a_curved_curve_keeps_an_equal_transverse_shift() -> None:
    """A cardinal shift reads the same on every element: a curved curve
    keeps it per element, as without curves."""
    pts = _arc(45.0, 3)
    with_curve = _emit(member_bridge(_chain(pts, curve=CURVE), rule=SHIFT_ONLY))
    without = _emit(member_bridge(_chain(pts), rule=SHIFT_ONLY))
    assert with_curve.calls == without.calls


@pytest.mark.parametrize("case, match", [
    ("mixed", r"elements \[101, 102, 103\] of curve 7 do not all run the same way"),
    ("closed", r"curve 7 \(elements \[101, 102, 103, 104\]\) is closed"),
    ("gap", r"covers elements \[101, 103\] of curve 7 but not its elements \[102\]"),
    ("short", r"element 101 of curve 7 .* is no longer than the end offsets"),
    ("kink", r"curve 7 .* has end lengths along its axis and different transverse"),
])
def test_ambiguous_curves_refuse(case: str, match: str) -> None:
    rule: Any = RULE
    if case == "mixed":      # one element of the line runs backwards
        coords = {1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0), 3: (2.0, 0.0, 3.0), 4: (3.0, 0.0, 3.0)}
        fem = stub(coords, {"M": ((101, 102, 103), ((1, 2), (3, 2), (3, 4)))},
                   entity=dict.fromkeys((101, 102, 103), CURVE))
    elif case == "closed":   # a square ring meshed from one closed curve
        coords = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0), 3: (4.0, 4.0, 3.0), 4: (0.0, 4.0, 3.0)}
        fem = stub(coords, {"M": ((101, 102, 103, 104), ((1, 2), (2, 3), (3, 4), (4, 1)))},
                   entity=dict.fromkeys((101, 102, 103, 104), CURVE))
    elif case == "gap":      # the curve's middle element sits under another transform
        coords = {1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0), 3: (2.0, 0.0, 3.0), 4: (3.0, 0.0, 3.0)}
        fem = stub(coords, {"M": ((101, 103), ((1, 2), (3, 4))), "N": ((102,), ((2, 3),))},
                   entity=dict.fromkeys((101, 102, 103), CURVE))
    elif case == "short":    # 0.3 m end zone on a 0.25 m end element
        fem = member_fem(16)
    else:                    # end lengths and transverse ends 0.1 / 0.3
        fem = member_fem(4)
        rule = ((0.25, 0.1, 0.0), (-0.4, 0.3, 0.0))
    with pytest.raises(BridgeError, match=match):
        member_bridge(fem, rule=rule).build().emit(RecordingEmitter())


def _three_on_one_curve(mine: tuple[int, ...]) -> FEMStub:
    """Curve 7 of three elements 101-103 along +X; ``mine`` under PG "M"
    (the end-sensitive transform), the rest under PG "N"."""
    coords = {1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0), 3: (2.0, 0.0, 3.0), 4: (3.0, 0.0, 3.0)}
    conn = {101: (1, 2), 102: (2, 3), 103: (3, 4)}
    rest = tuple(e for e in conn if e not in mine)
    groups = {"M": (mine, tuple(conn[e] for e in mine))}
    if rest:
        groups["N"] = (rest, tuple(conn[e] for e in rest))
    return stub(coords, groups, entity=dict.fromkeys(conn, CURVE))


def _partial(mine: tuple[int, ...], rule: Any = RULE) -> apeSees:
    ops = bridge(_three_on_one_curve(mine), rule=rule)
    ops.element.elasticBeamColumn(
        pg="N", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    return ops


@pytest.mark.parametrize("mine, missing", [
    ((101, 102), "[103]"), ((102,), "[101, 103]"), ((101,), "[102, 103]"),
    ((101, 103), "[102]")], ids=["first_two", "middle", "first", "both_ends"])
def test_a_partial_curve_with_end_zones_refuses_naming_the_node(
        mine: tuple[int, ...], missing: str) -> None:
    """Maintainer ruling (2026-10-27, #2578 review, option B): part of a
    curve keeps each element's own offset, and refuses where that puts an
    end zone at a node interior to the curve. The review's three cases:
    101+102 (the j-zone landed mid-curve), a lone element (both end zones
    at interior nodes) and 101+103; all put a 0.3 m zone at node 2."""
    with pytest.raises(BridgeError, match=(
            rf"covers elements \[{', '.join(map(str, sorted(mine)))}\] of curve 7 "
            rf"but not its elements {re.escape(missing)}, and would put an end "
            r"zone or axial offset \(-?0\.3 along element 10\d\) at node 2, "
            r"interior to the curve")):
        _partial(mine).build().emit(RecordingEmitter())


@pytest.mark.parametrize("mine, jnt, node", [
    ((101,), ((0.0, 0.1, 0.0), (0.05, 0.1, 0.0)), 2),
    ((103,), ((0.05, 0.1, 0.0), (0.0, 0.1, 0.0)), 3),
], ids=["j_half", "i_half"])
def test_a_partial_curve_refuses_any_axial_part_at_an_interior_node(
        mine: tuple[int, ...], jnt: Any, node: int) -> None:
    """Not only a rigid length: a 0.05 m component along the axis at an
    interior node (either half of the tail) refuses."""
    with pytest.raises(BridgeError, match=rf"\(0\.05 along element {mine[0]}\) at node {node},"):
        _partial(mine, rule=jnt).build().emit(RecordingEmitter())


@pytest.mark.parametrize("mine, rule, want", [
    ((101,), InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                            end_offset_i=0.3),
     {101: (0.3, 0.0, -0.3, 0.0, 0.0, -0.3)}),
    ((102,), ((0.0, 0.1, 0.0), (0.0, 0.2, 0.0)),
     {102: (0.0, 0.1, 0.0, 0.0, 0.2, 0.0)}),
    ((103,), InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                            end_offset_j=0.3),
     {103: (0.0, 0.0, -0.3, -0.3, 0.0, -0.3)}),
], ids=["i_end_element", "interior_transverse", "j_end_element"])
def test_a_partial_curve_with_transverse_interior_nodes_is_accepted(
        mine: tuple[int, ...], rule: Any, want: dict[int, tuple[float, ...]]) -> None:
    """The shape an Assembly rehydrate re-declares (one line per element of
    a split curve): end zones only at the curve's own ends, transverse
    parts (equal or not) at its interior nodes. Each element keeps its
    own offset."""
    got = element_offsets(_emit(_partial(mine, rule=rule)))
    for eid, tail in want.items():
        assert got[eid] == _approx(tail)


def test_a_partial_curve_keeps_an_equal_transverse_shift() -> None:
    """A shift that reads the same on every element stays per element on
    part of a curve, as without curves."""
    def run(fem: FEMStub) -> list[Any]:
        ops = bridge(fem, rule=SHIFT_ONLY)
        ops.element.elasticBeamColumn(
            pg="N", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
        return _emit(ops).calls
    curved = _three_on_one_curve((101, 102))
    bare = stub({1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0), 3: (2.0, 0.0, 3.0), 4: (3.0, 0.0, 3.0)},
                {"M": ((101, 102), ((1, 2), (2, 3))), "N": ((103,), ((3, 4),))})
    assert run(curved) == run(bare)


def test_a_straight_curve_at_utm_coordinates_splits() -> None:
    """A straight 1 m curve at UTM scale (E 5e5, N 1e7 m), 4 elements
    placed by linear interpolation as gmsh does. Its interior nodes sit
    1.4e-9 m off the chord (round-off at 1e7), more than a tolerance of
    1e-9 x its length, so straightness must scale with the coordinates."""
    import math

    import numpy as np

    ang = math.radians(11.4)
    u = np.array([math.cos(ang), math.sin(ang), 0.0])
    p0 = np.array([512347.048, 9876540.5, 3.0])
    pts = [tuple(float(c) for c in p0 + (k / 4.0) * u) for k in range(5)]
    rule = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.05, end_offset_j=0.05)
    got = element_offsets(_emit(bridge(_chain(pts, curve=CURVE), rule=rule)))
    hang = (0.0, 0.0, -0.3)
    z = 0.05 * u
    assert got[101] == pytest.approx((z[0], z[1], -0.3, *hang), abs=1e-8)
    assert got[102] == pytest.approx((*hang, *hang), abs=1e-8)
    assert got[103] == pytest.approx((*hang, *hang), abs=1e-8)
    assert got[104] == pytest.approx((*hang, -z[0], -z[1], -0.3), abs=1e-8)


def test_a_5mm_sagitta_at_utm_coordinates_refuses() -> None:
    """The straightness tolerance is a round-off bound on the coordinates
    (64 eps |x|max, ~1.4e-7 m at 1e7) plus 1e-9 L, never a share of the
    coordinates: a 1 m curve bowed 5 mm at N 1e7 is curved, and refuses."""
    import math

    import numpy as np

    ang = math.radians(11.4)
    u = np.array([math.cos(ang), math.sin(ang), 0.0])
    n = np.array([-math.sin(ang), math.cos(ang), 0.0])
    p0 = np.array([512347.048, 9876540.5, 3.0])
    pts = [tuple(float(c) for c in p0 + s * u + 4.0 * 0.005 * s * (1.0 - s) * n)
           for s in (k / 4.0 for k in range(5))]
    rule = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.05, end_offset_j=0.05)
    with pytest.raises(BridgeError, match=r"curve 7 \(elements \[101, 102, 103, 104\]\) is not straight"):
        bridge(_chain(pts, curve=CURVE), rule=rule).build().emit(RecordingEmitter())


def test_a_lone_reversed_element_of_a_curve_refuses() -> None:
    """Review round 2: element 101 is stored 2 -> 1 on a curve that runs
    1 -> 4. Alone under a transform with only a j-zone, it would put that
    zone at node 1, the curve's i-end; the curve's mixed orientation
    refuses for a part of it too."""
    coords = {1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0), 3: (2.0, 0.0, 3.0), 4: (3.0, 0.0, 3.0)}
    fem = stub(coords, {"M": ((101,), ((2, 1),)), "N": ((102, 103), ((2, 3), (3, 4)))},
               entity=dict.fromkeys((101, 102, 103), CURVE))
    rule = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                          end_offset_j=0.3)
    ops = bridge(fem, rule=rule)
    ops.element.elasticBeamColumn(
        pg="N", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    with pytest.raises(BridgeError, match=r"elements \[101, 102, 103\] of curve 7 do not all run the same way"):
        ops.build().emit(RecordingEmitter())


def test_the_curve_index_is_built_once_per_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 2 (cost): 40 end-sensitive transforms, one per curve,
    plus their line loads, scan the snapshot's line elements once, not
    once per transform (O(transforms x lines) was ~130 s per plan for a
    2000-line rehydrate)."""
    from apeGmsh._kernel.record_sets import ElementLoadSet
    from apeGmsh._kernel.records._loads import ElementLoadRecord

    n = 40
    coords = {}
    groups = {}
    entity = {}
    for k in range(n):
        a, b = 2 * k + 1, 2 * k + 2
        coords[a], coords[b] = (0.0, float(k), 3.0), (4.0, float(k), 3.0)
        groups[f"M{k}"] = ((1000 + k,), ((a, b),))
        entity[1000 + k] = k + 1
    fem = stub(coords, groups, entity=entity)
    fem.elements.loads = ElementLoadSet([
        ElementLoadRecord(element_id=1000 + k, load_type="beamUniform",
                          params={"wx": 0.0, "wy": 0.0, "wz": -10.0}, pattern="dl")
        for k in range(n)])
    scans = []
    select = type(fem.elements).select

    def counting(self: Any, *args: Any, **kw: Any) -> Any:
        if kw.get("dim") == 1 and kw.get("pg") is None:
            scans.append(1)
        return select(self, *args, **kw)
    monkeypatch.setattr(type(fem.elements), "select", counting)
    ops = apeSees(cast("object", fem), default_orientation=None, element_tags="fem")
    ops.model(ndm=3, ndf=6)
    for k in range(n):
        ops.element.elasticBeamColumn(
            pg=f"M{k}", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=RULE),
            A=0.01, E=200e9, Iz=1e-4, Iy=2e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1,) * 6)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.from_model("dl")
    got = element_offsets(_emit(ops))
    assert len(got) == n and all(t == _approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))
                                 for t in got.values())
    assert len(scans) == 1


def test_varying_vecxz_along_a_curve_refuses() -> None:
    """Rows of one straight curve whose vecxz turns: no single frame."""
    import numpy as np

    from apeGmsh.opensees._internal.build.transforms import _split_member
    from apeGmsh.opensees.transform import Linear

    rows = [(101, 1, 2, np.array([0.0, 0.0, 3.0]), np.array([2.0, 0.0, 3.0]), (0.0, 0.0, 1.0)),
            (102, 2, 3, np.array([2.0, 0.0, 3.0]), np.array([4.0, 0.0, 3.0]), (0.0, 1.0, 0.0))]
    with pytest.raises(BridgeError, match=r"elements of curve 7 \(\[101, 102\]\) have different vecxz"):
        _split_member(Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=RULE), rows,  # type: ignore[arg-type]
                      ("", CURVE), 3, "where")


@pytest.mark.parametrize("entity", [{101: CURVE}, {101: CURVE, 102: -1}],
                         ids=["missing", "minus_one"])
def test_an_element_without_a_curve_keeps_the_whole_transform_refused(
        entity: dict[int, int]) -> None:
    """Ratified Q5: any element under the transform lacking an entity
    (pre-2.36 data, a FEM not extracted from gmsh) keeps #2470's refusal
    for all of it."""
    fem = stub(GIRDER, {"M": ((101, 102), ((1, 2), (2, 3)))}, entity=entity)
    with pytest.raises(BridgeError, match=REFUSAL):
        bridge(fem).build().emit(RecordingEmitter())


def test_two_collinear_curves_are_two_members() -> None:
    """What #2470 refused for want of curves: girder 1-2-3 drawn as two
    curves is two members, each with its end zones at both ends."""
    fem = stub(GIRDER, {"M": ((101, 102), ((1, 2), (2, 3)))}, entity={101: 1, 102: 2})
    got = element_offsets(_emit(bridge(fem)))
    assert got[101] == got[102] == _approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))


@pytest.mark.parametrize("labels, split", [
    ({101: "A", 102: "B"}, False), ({101: "A", 102: "A"}, True)],
    ids=["two_instances", "one_instance"])
def test_the_member_key_includes_the_module_label(
        labels: dict[int, str], split: bool) -> None:
    """Curve tag 5 in two Assembly instances is two members; in one
    instance it is one member meshed into two elements."""
    fem = stub(GIRDER, {"M": ((101, 102), ((1, 2), (2, 3)))},
               entity={101: 5, 102: 5}, module_label=labels)
    got = element_offsets(_emit(bridge(fem)))
    if split:
        assert got[101] == _approx((0.3, 0.0, -0.3, 0.0, 0.0, -0.3))
        assert got[102] == _approx((0.0, 0.0, -0.3, -0.3, 0.0, -0.3))
    else:
        assert got[101] == got[102] == _approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))


@pytest.mark.parametrize("touch", ["secondary_beam", "spring_pg", "spring_nodes"])
def test_what_frames_in_at_mid_span_does_not_split_the_curve(touch: str) -> None:
    """Maintainer ruling (2026-10-26): offsets go only at a curve's two
    ends, whatever connects along it."""
    coords = dict(GIRDER)
    groups = {"M": ((101, 102), ((1, 2), (2, 3)))}
    entity = {101: CURVE, 102: CURVE}
    if touch == "secondary_beam":
        coords[4] = (2.0, 3.0, 3.0)
        groups["S"] = ((201,), ((2, 4),))
        entity[201] = 8
    else:
        coords[6] = (2.0, 0.0, 3.0)
        if touch == "spring_pg":
            groups["Z"] = ((301,), ((6, 2),))
    ops = bridge(stub(coords, groups, entity=entity))
    if touch == "secondary_beam":
        ops.element.elasticBeamColumn(
            pg="S", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=RULE),
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    else:
        mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e8)
        dirs = tuple(ZeroLengthMatDir(material=mat, dof=d) for d in (1, 2, 3))
        if touch == "spring_pg":
            ops.element.ZeroLength(pg="Z", mat_dirs=dirs)
        else:
            ops.element.ZeroLength(nodes=(6, 2), mat_dirs=dirs)
    got = element_offsets(_emit(ops))
    assert got[101] == _approx((0.3, 0.0, -0.3, 0.0, 0.0, -0.3))
    assert got[102] == _approx((0.0, 0.0, -0.3, -0.3, 0.0, -0.3))


@pytest.mark.parametrize("staged", [False, True], ids=["flat", "staged"])
def test_partitioned_member_keeps_its_ends(staged: bool) -> None:
    """Two ranks share the curve: the offsets are planned over the whole
    member, and its end-zone loads land once, at the member's two joints
    (w = 10 over 0.3 m: 3.0 each)."""
    loads = {"wx": 0.0, "wy": 0.0, "wz": -10.0}
    parts = [(0, [1, 2, 3], [101, 102]), (1, [3, 4, 5, 6], [103, 104, 105])]
    rec_flat = _emit(member_bridge(member_fem(5, loads=loads), loaded=True))
    rec = _emit(member_bridge(member_fem(5, loads=loads, partitions=parts),
                              loaded=True, staged=staged))
    assert element_offsets(rec) == element_offsets(rec_flat)
    joint_loads = [(int(a[0]), round(float(a[3]), 12)) for n, a, _k in rec.calls
                   if n == "load"]
    assert joint_loads == [(1, -3.0), (6, -3.0)]


def _accepted_by_2470() -> "dict[str, tuple[FEMStub, FEMStub, Any]]":
    """Models #2470 accepts, each as ``(no curves, curves as gmsh would
    give them, rule)``: one curve per member."""
    import math

    def pair(coords: dict, groups: dict, entity: dict, rule: Any = RULE) -> tuple:
        return (stub(coords, groups), stub(coords, groups, entity=entity), rule)
    t = math.radians(45.0)
    knee = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0),
            3: (4.0 + 4.0 * math.cos(t), 4.0 * math.sin(t), 3.0)}
    corner = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0), 3: (4.0, 5.0, 3.0)}

    def arm(deg: float) -> tuple[float, float, float]:
        a = math.radians(deg)
        return (4.0 * math.cos(a), 4.0 * math.sin(a), 3.0)
    y_node = {1: (0.0, 0.0, 3.0), 2: arm(0.0), 3: arm(120.0), 4: arm(240.0)}
    two = {"M": ((101, 102), ((1, 2), (2, 3)))}
    return {
        "one_element": pair({1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0)},
                            {"M": ((101,), ((1, 2),))}, {101: 1}),
        "knee_45": pair(knee, two, {101: 1, 102: 2}),
        "corner_c9": pair(corner, two, {101: 1, 102: 2}, InsertionPoint(
            cardinal_point=9, depth=0.6, width=0.3, axis_2="z",
            end_offset_i=0.25, end_offset_j=0.25)),
        "y_node": pair(y_node, {"M": ((101, 102, 103), ((1, 2), (1, 3), (1, 4)))},
                       {101: 1, 102: 2, 103: 3}),
        "shift_run": pair(GIRDER, two, {101: 1, 102: 1}, SHIFT_ONLY),
        "constant_shift_run": pair(GIRDER, two, {101: 1, 102: 1},
                                   ((0.0, 0.1, -0.3), (0.0, 0.1, -0.3))),
        "constant_corner": pair(corner, two, {101: 1, 102: 2},
                                ((0.25, 0.0, 0.0), (-0.4, 0.0, 0.0))),
    }


@pytest.mark.parametrize("case", sorted(_accepted_by_2470()))
def test_curves_leave_what_2470_accepted_byte_identical(case: str) -> None:
    """Invariant 7 of the #2551 design: giving a model #2470 accepts its
    source curves changes no emitted call."""
    bare, curved, rule = _accepted_by_2470()[case]
    assert _emit(bridge(curved, rule=rule)).calls == _emit(bridge(bare, rule=rule)).calls


def test_end_zones_longer_than_an_element_fit_the_curve() -> None:
    """0.6 m end zones on a 4 m curve of 1 m elements: each element alone
    would carry 1.2 m of zones (refused), but the member puts 0.6 m on
    each end element only. A one-element curve still refuses."""
    rule = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.6, end_offset_j=0.6)
    got = element_offsets(_emit(member_bridge(member_fem(4), rule=rule)))
    assert got[101] == _approx((0.6, 0.0, -0.3, 0.0, 0.0, -0.3))
    assert got[104] == _approx((0.0, 0.0, -0.3, -0.6, 0.0, -0.3))
    one = stub({1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0)}, {"M": ((101,), ((1, 2),))},
               entity={101: CURVE})
    with pytest.raises(BridgeError, match=r"element 101 jnt_offset: .*reach past"):
        bridge(one, rule=rule).build().emit(RecordingEmitter())
