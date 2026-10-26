"""``jnt_offset`` on a frame member meshed into several elements.

ETABS applies a frame object's end offsets at the object's ends (#2470
review of 34f0bf3c, maintainer ruling 2026-10-26): the member's i-end
element takes the i-offset, its j-end element the j-offset, and interior
elements only the offsets' transverse part (a cardinal-point shift), so
the offset axis is the one straight line a one-element member has.

The live oracle (an n-element member against a one-element member with
the same offsets) is ``tests/opensees/integration/test_jnt_offset_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.joint_offset import InsertionPoint

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

#: Cardinal 8 on a 0.6 m deep beam (axis_2 = z, vecxz = +Z): the axis
#: hangs 0.3 m below the joints; 0.3 m rigid zones at both member ends.
RULE = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                      end_offset_i=0.3, end_offset_j=0.3)
LENGTH = 4.0


def member_fem(n: int, *, reverse: bool = False, ndm: int = 3,
               loads: dict | None = None, partitions: Any = None) -> FEMStub:
    """A gmsh-like line (0,0,3)-(4,0,3) meshed into ``n`` elements: nodes
    1..n+1 along +X, elements 101.. in curve order (``reverse``: the
    curve runs -X, so every element's node order flips)."""
    from apeGmsh._kernel.record_sets import ElementLoadSet
    from apeGmsh._kernel.records._loads import ElementLoadRecord

    z = 3.0 if ndm == 3 else 0.0
    coords = [(LENGTH * k / n, 0.0 if ndm == 3 else z, z if ndm == 3 else 0.0)
              for k in range(n + 1)]
    ids = list(range(1, n + 2))
    conn = [(k + 1, k + 2) for k in range(n)]
    if reverse:
        conn = [(b, a) for a, b in reversed(conn)]
    eids = tuple(101 + k for k in range(n))
    nodes = _NodesStub(ids=ids, coords=coords,
                       node_pgs={"Base": [1], "Tip": [n + 1]})
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs={
        "M": _ElementGroupView(ids=eids, connectivity=tuple(conn))}))
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


def element_offsets(rec: RecordingEmitter) -> dict[int, tuple[float, ...]]:
    """Element id -> the ``-jntOffset`` tail of the transform it uses."""
    lines = {args[1]: tuple(kw.get("jnt_offset", ()))
             for name, args, kw in rec.calls if name == "geomTransf"}
    out: dict[int, tuple[float, ...]] = {}
    for name, args, _kw in rec.calls:
        if name == "element":
            out[int(args[1])] = lines[int(args[-1])]
    return out


def _approx(t: tuple[float, ...]) -> Any:
    return pytest.approx(t, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_end_offsets_sit_on_the_member_end_elements_only(n: int) -> None:
    """The review's repro: n = 5 used to give all five elements
    ``0.3 0 -0.3 -0.3 0 -0.3`` (each 0.8 m element 75 % rigid)."""
    rec = RecordingEmitter()
    member_bridge(member_fem(n)).build().emit(rec)
    got = element_offsets(rec)
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
    element order and node order of the mesh start."""
    rule = InsertionPoint(cardinal_point=8, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.5, end_offset_j=0.2)
    rec = RecordingEmitter()
    member_bridge(member_fem(5, reverse=True), rule=rule).build().emit(rec)
    got = element_offsets(rec)
    first, last = 101, 105       # 101 runs 6 -> 5 (x = 4 -> 3.2); 105 ends at x = 0
    assert got[first] == _approx((-0.5, 0.0, -0.3, 0.0, 0.0, -0.3))
    assert got[last] == _approx((0.0, 0.0, -0.3, 0.2, 0.0, -0.3))
    for eid in (102, 103, 104):
        assert got[eid] == _approx((0.0, 0.0, -0.3, 0.0, 0.0, -0.3))


def test_constant_pair_with_end_lengths_splits_over_the_member() -> None:
    rec = RecordingEmitter()
    member_bridge(member_fem(4), rule=((0.25, 0.1, 0.0), (-0.4, 0.1, 0.0))).build().emit(rec)
    got = element_offsets(rec)
    assert got[101] == _approx((0.25, 0.1, 0.0, 0.0, 0.1, 0.0))
    assert got[102] == got[103] == _approx((0.0, 0.1, 0.0, 0.0, 0.1, 0.0))
    assert got[104] == _approx((0.0, 0.1, 0.0, -0.4, 0.1, 0.0))


def test_equal_end_offsets_stay_on_every_element() -> None:
    """A pair equal at both ends is constant along the member: every
    element keeps it, through the transform's own single line."""
    rec = RecordingEmitter()
    member_bridge(member_fem(3), rule=((0.0, 0.1, -0.3), (0.0, 0.1, -0.3))).build().emit(rec)
    assert len([c for c in rec.calls if c[0] == "geomTransf"]) == 1
    assert set(element_offsets(rec).values()) == {(0.0, 0.1, -0.3, 0.0, 0.1, -0.3)}


def test_a_square_ring_is_four_members() -> None:
    """Members are straight: each corner of a ring is a joint even with
    nothing else attached (ETABS: four frame objects)."""
    fem = FEMStub(nodes=_NodesStub(
        ids=[1, 2, 3, 4, 5, 6, 7, 8],
        coords=[(0, 0, 3), (2, 0, 3), (4, 0, 3), (4, 2, 3), (4, 4, 3),
                (2, 4, 3), (0, 4, 3), (0, 2, 3)],
        node_pgs={"Base": [1]}), elements=_ElementsStub(elem_pgs={
            "M": _ElementGroupView(ids=tuple(range(101, 109)), connectivity=tuple(
                (k, k % 8 + 1) for k in range(1, 9)))}))
    rule = InsertionPoint(depth=0.6, width=0.3, axis_2="z", end_offset_i=0.3,
                          end_offset_j=0.3)
    rec = RecordingEmitter()
    member_bridge(fem, rule=rule).build().emit(rec)
    got = element_offsets(rec)
    for first in (101, 103, 105, 107):      # each side's i-end element
        tail = got[first]
        assert tail[3:] == (0.0, 0.0, 0.0) and abs(sum(c * c for c in tail[:3]) - 0.09) < 1e-12
    for last in (102, 104, 106, 108):
        tail = got[last]
        assert tail[:3] == (0.0, 0.0, 0.0) and abs(sum(c * c for c in tail[3:]) - 0.09) < 1e-12


def test_a_joint_splits_the_member(tmp_path: Path) -> None:
    """A node another element touches is a joint: the transform's elements
    on each side are separate members, each with its own end zones."""
    fem = member_fem(4)
    fem.elements = _ElementsStub(elem_pgs={
        "M": _ElementGroupView(ids=(101, 102, 103, 104),
                               connectivity=((1, 2), (2, 3), (3, 4), (4, 5))),
        "Post": _ElementGroupView(ids=(201,), connectivity=((3, 6),))})
    fem.nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6],
        coords=[(0, 0, 3), (1, 0, 3), (2, 0, 3), (3, 0, 3), (4, 0, 3), (2, 0, 0)],
        node_pgs={"Base": [1]})
    fem.elements.loads = None
    ops = member_bridge(fem)
    ops.element.elasticBeamColumn(
        pg="Post", transf=ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    got = element_offsets(rec)
    assert got[101] == got[103] == _approx((0.3, 0.0, -0.3, 0.0, 0.0, -0.3))
    assert got[102] == got[104] == _approx((0.0, 0.0, -0.3, -0.3, 0.0, -0.3))


@pytest.mark.parametrize("case, match", [
    ("loop", "closed loop"),
    ("mixed", "do not all run the same way"),
    ("short", "no longer than the end offsets"),
])
def test_ambiguous_members_refuse(case: str, match: str) -> None:
    if case == "loop":       # two elements on one node pair: no member end
        fem = FEMStub(nodes=_NodesStub(
            ids=[1, 2], coords=[(0, 0, 3), (4, 0, 3)],
            node_pgs={"Base": [1]}), elements=_ElementsStub(elem_pgs={
                "M": _ElementGroupView(ids=(101, 102), connectivity=(
                    (1, 2), (2, 1)))}))
        rule: Any = RULE
    elif case == "mixed":    # one element of the line runs backwards
        fem = member_fem(3)
        fem.elements = _ElementsStub(elem_pgs={"M": _ElementGroupView(
            ids=(101, 102, 103), connectivity=((1, 2), (3, 2), (3, 4)))})
        rule = RULE
    else:                    # 0.3 m end zone on a 0.25 m end element
        fem = member_fem(16)
        rule = RULE
    with pytest.raises(BridgeError, match=match):
        member_bridge(fem, rule=rule).build().emit(RecordingEmitter())


@pytest.mark.parametrize("staged", [False, True], ids=["flat", "staged"])
def test_partitioned_member_keeps_its_ends(staged: bool) -> None:
    """Two ranks share the member: the offsets are planned over the whole
    member, and its end-zone loads land once, at the member's two joints."""
    loads = {"wx": 0.0, "wy": 0.0, "wz": -10.0}
    parts = [(0, [1, 2, 3], [101, 102]), (1, [3, 4, 5, 6], [103, 104, 105])]
    rec_flat = RecordingEmitter()
    member_bridge(member_fem(5, loads=loads), loaded=True).build().emit(rec_flat)
    rec = RecordingEmitter()
    member_bridge(member_fem(5, loads=loads, partitions=parts), loaded=True,
                  staged=staged).build().emit(rec)
    assert element_offsets(rec) == element_offsets(rec_flat)
    joint_loads = [(int(a[0]), round(float(a[3]), 12)) for n, a, _k in rec.calls
                   if n == "load"]
    assert joint_loads == [(1, -3.0), (6, -3.0)]


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

    ops = _real_member(staged)
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    h5 = tmp_path / "m.h5"
    ops.h5(str(h5))
    replayed = OpenSeesModel.from_h5(str(h5)).build("tcl")

    def block(text: str) -> list[str]:
        lines = [ln.strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln.startswith(("load ", "eleLoad "))]
    got, want = block(replayed), block(deck.read_text(encoding="utf-8"))
    assert [ln.split()[0] for ln in want] == ["load", "load", "eleLoad", "eleLoad"]
    assert got == want


def _real_member(staged: bool) -> apeSees:
    """A 2-element member (0,0)-(4,0) on a real ``FEMData`` (``from_h5``
    reloads the neutral zone, which refuses stub-only files), under a
    from_model line load with 0.3 m end zones."""
    from apeGmsh._kernel.records._loads import ElementLoadRecord
    from tests.opensees.unit.test_beam_uniform_load import _beam_fem_data

    fem = _beam_fem_data(ndm=2, loads=[
        ElementLoadRecord(element_id=e, load_type="beamUniform",
                          params={"wx": 0.0, "wy": -10.0, "wz": 0.0},
                          pattern="dl") for e in (10, 20)])
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(
        pg="Beams", transf=ops.geomTransf.Linear(jnt_offset=((0.3, 0.0), (-0.3, 0.0))),
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
