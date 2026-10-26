"""``jnt_offset`` where a frame member may be meshed into several elements.

ETABS puts a frame object's end offsets at the object's two ends. The
mesh alone cannot tell a member meshed into several elements from a
continuous girder of several members (both are a straight collinear run
of line elements), so until elements carry their source curve (#2551)
the bridge refuses an end-sensitive offset (an end length, an axial
component, unequal ends) on any collinear run of a transform's elements
(maintainer ruling 2026-10-26, #2470 round 3). What touches the run
(a secondary beam, a spring declared ``pg=`` or ``nodes=``) does not
change the verdict. One-element members, chains that turn a corner, and
a transverse shift equal at both ends keep working.
"""
from __future__ import annotations

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
REFUSAL = r"meet at node \d+ turning .* deg, under the 30 deg.*source curve.*#2551"


def stub(coords: dict[int, tuple[float, float, float]],
         groups: dict[str, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]],
         ) -> FEMStub:
    ids = sorted(coords)
    return FEMStub(
        nodes=_NodesStub(ids=ids, coords=[coords[n] for n in ids],
                         node_pgs={"Base": [ids[0]]}),
        elements=_ElementsStub(elem_pgs={
            name: _ElementGroupView(ids=eids, connectivity=conn)
            for name, (eids, conn) in groups.items()}))


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
    """Element id -> the ``-jntOffset`` tail of the transform it uses."""
    lines = {args[1]: tuple(kw.get("jnt_offset", ()))
             for name, args, kw in rec.calls if name == "geomTransf"}
    return {int(args[1]): lines[int(args[-1])]
            for name, args, _kw in rec.calls if name == "element"}


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
    one member in two elements; only the curves (#2551) can tell."""
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


def _chain(points: list[tuple[float, float, float]]) -> FEMStub:
    """A chain of line elements 101.. through ``points`` (nodes 1..n)."""
    coords = {k + 1: p for k, p in enumerate(points)}
    conn = tuple((k + 1, k + 2) for k in range(len(points) - 1))
    return stub(coords, {"M": (tuple(101 + k for k in range(len(conn))), conn)})


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
    with pytest.raises(BridgeError, match=r"turning 3 deg, under the 30 deg"):
        bridge(_chain(pts)).build().emit(RecordingEmitter())


def test_a_20_deg_kink_refuses_naming_the_threshold() -> None:
    import math
    t = math.radians(20.0)
    pts = [(0.0, 0.0, 3.0), (4.0, 0.0, 3.0), (4.0 + 4.0 * math.cos(t), 4.0 * math.sin(t), 3.0)]
    with pytest.raises(BridgeError, match=r"node 2 turning 20 deg, under the 30 deg that marks a frame corner"):
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


def test_three_elements_at_a_node_each_end_there() -> None:
    """A node where three of the transform's elements meet is a joint:
    each element is its own member, collinear pair included."""
    coords = {1: (0.0, 0.0, 3.0), 2: (4.0, 0.0, 3.0), 3: (8.0, 0.0, 3.0), 4: (4.0, 4.0, 3.0)}
    fem = stub(coords, {"M": ((101, 102, 103), ((1, 2), (2, 3), (2, 4)))})
    rec = RecordingEmitter()
    bridge(fem).build().emit(rec)
    got = element_offsets(rec)
    assert got[101] == got[102] == pytest.approx((0.3, 0.0, -0.3, -0.3, 0.0, -0.3))
    assert got[103] == pytest.approx((0.0, 0.3, -0.3, 0.0, -0.3, -0.3))


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
