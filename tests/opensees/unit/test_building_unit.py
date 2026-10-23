"""Unit tests of :mod:`apeGmsh.opensees.building` against a fake ops module.

No openseespy: a tiny in-memory domain answers the handful of queries the
module makes, so the free-body arithmetic, the argument checks and the
refusals run in the default suite. The live numeric validation is
``tests/opensees/live/test_building_live.py``.
"""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import building as B
from apeGmsh.opensees._internal.build.errors import BridgeError


class FakeOps:
    """Two columns 0 -> 3 and a beam on top; ``forces`` fixed per element."""

    def __init__(self, ndf: int = 6, ndm: int = 3) -> None:
        self.nodes: dict[int, list[float]] = {
            1: [0.0, 0.0, 0.0], 2: [4.0, 0.0, 0.0],
            3: [0.0, 0.0, 3.0], 4: [4.0, 0.0, 3.0],
        }
        if ndm == 2:
            self.nodes = {k: v[:2] for k, v in self.nodes.items()}
        self._ndf = ndf
        self.eles: dict[int, list[int]] = {1: [1, 3], 2: [2, 4], 3: [3, 4]}
        self.forces: dict[int, list[float]] = {}
        self.retained: dict[int, list[int]] = {}
        self.disp: dict[int, list[float]] = {n: [0.0] * ndf for n in self.nodes}
        self.incr: dict[int, list[float]] = {}   # trial - committed per node
        self.n_eq = 0                 # EQ constraints printModel reports
        self.print_eq_line = True
        self.n_print = 0              # printModel calls

    def printModel(self, flag: str, path: str) -> None:
        assert flag == "-file"
        self.n_print += 1
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("MP_Constraints: numConstraints: 0\n")
            if self.print_eq_line:
                fh.write(f"EQ_Constraints: numConstraints: {self.n_eq}\n")

    def getNodeTags(self) -> list[int]:
        return list(self.nodes)

    def nodeCoord(self, n: int) -> list[float]:
        return self.nodes[n]

    def getNDF(self, n: int) -> list[int]:
        return [self._ndf]

    def getEleTags(self) -> list[int]:
        return list(self.eles)

    def eleNodes(self, e: int) -> list[int]:
        return self.eles[e]

    def eleType(self, e: int) -> str:
        return "FakeBeam"

    def eleForce(self, e: int) -> list[float]:
        return self.forces[e]

    def getRetainedNodes(self, n: int) -> list[int]:
        return self.retained.get(n, [])

    def nodeDisp(self, n: int) -> list[float]:
        return self.disp[n]

    def nodeResponse(self, n: int, dof: int, response: int) -> float:
        assert response == 4                 # IncrDisp: trial - committed
        return self.incr.get(n, [0.0] * 6)[dof - 1]


def _mod(fake: FakeOps) -> ModuleType:
    return cast(ModuleType, fake)


def _column_forces(fake: FakeOps, f_top: np.ndarray, m_top: np.ndarray) -> None:
    """Column forces in equilibrium with a top force ``f_top`` (and moment)."""
    for e in (1, 2):
        # the node applies -f at the bottom, +f at the top of the element
        fe = np.zeros(12)
        fe[0:3] = -f_top / 2
        fe[3:6] = -m_top / 2
        fe[6:9] = f_top / 2
        fe[9:12] = m_top / 2
        fake.forces[e] = fe.tolist()
    fake.forces[3] = [0.0] * 12


def test_story_forces_free_body_sign_and_moments() -> None:
    fake = FakeOps()
    f = np.array([10.0, -2.0, -50.0])
    _column_forces(fake, f, np.zeros(3))
    sf = B.StoryForces((0.0, 3.0), ops=_mod(fake), reference=(1.0, 0.0))
    v = sf(_mod(fake))
    # the part above pushes +f/2 into each base node: VX, VY, P = f
    np.testing.assert_allclose(v[:3], f)
    arms = [np.array([-1.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]
    want_m = sum(np.cross(a, f / 2) for a in arms)
    np.testing.assert_allclose(v[3:], want_m)
    assert sf.labels == tuple(f"Story1:{c}" for c in B.STORY_FORCE_COMPONENTS)


def test_story_forces_top_reads_the_elements_below() -> None:
    fake = FakeOps()
    f = np.array([3.0, 4.0, 0.0])
    _column_forces(fake, f, np.array([0.0, 0.0, 1.0]))
    top = B.StoryForces((0.0, 3.0), location="top", ops=_mod(fake))(_mod(fake))
    np.testing.assert_allclose(top[:3], f)
    # MZ: the nodal moments (1.0) plus the arm of the x = 4 column (4 x 2.0)
    assert top[5] == pytest.approx(9.0)


def test_wrong_force_length_refuses_naming_the_class() -> None:
    fake = FakeOps()
    _column_forces(fake, np.ones(3), np.zeros(3))
    fake.forces[1] = [0.0] * 6
    sf = B.StoryForces((0.0, 3.0), ops=_mod(fake))
    with pytest.raises(BridgeError, match="FakeBeam"):
        sf(_mod(fake))


def test_two_d_model_and_odd_ndf_refuse() -> None:
    with pytest.raises(BridgeError, match="3-D model"):
        B.StoryForces((0.0, 3.0), ops=_mod(FakeOps(ndm=2)))
    with pytest.raises(BridgeError, match="ndf 4"):
        B.StoryForces((0.0, 3.0), ops=_mod(FakeOps(ndf=4)))


def test_mp_constraint_across_the_cut_refuses() -> None:
    fake = FakeOps()
    fake.retained = {3: [1]}
    with pytest.raises(BridgeError, match="MP constraint"):
        B.StoryForces((0.0, 3.0), ops=_mod(fake))
    fake.retained = {4: [3]}                   # inside the level: fine
    B.StoryForces((0.0, 3.0), ops=_mod(fake))


def test_an_eq_constraint_anywhere_refuses_the_story_cut() -> None:
    fake = FakeOps()
    _column_forces(fake, np.ones(3), np.zeros(3))
    B.StoryForces((0.0, 3.0), ops=_mod(fake))
    fake.n_eq = 2
    with pytest.raises(BridgeError, match="2 EQ constraint"):
        B.StoryForces((0.0, 3.0), ops=_mod(fake))


class EqBuild(FakeOps):
    """A build with the ``equationConstraint`` command."""

    def equationConstraint(self, *args: Any) -> None:
        raise AssertionError("not called")


def test_the_eq_count_is_read_once_at_construction_not_per_call() -> None:
    fake = FakeOps()
    _column_forces(fake, np.ones(3), np.zeros(3))
    sf = B.StoryForces((0.0, 3.0), ops=_mod(fake))
    assert fake.n_print == 1
    for _ in range(3):                     # e.g. one call per mode
        sf(_mod(fake))
    sf.table(ops=_mod(fake))
    assert fake.n_print == 1


def test_an_unreadable_eq_count_refuses_when_the_build_has_eq() -> None:
    fake = EqBuild()
    fake.print_eq_line = False
    with pytest.raises(BridgeError, match="EQ constraint count"):
        B.StoryForces((0.0, 3.0), ops=_mod(fake))
    old = FakeOps()                       # openseespy < 3.8: no EQ at all
    old.print_eq_line = False
    B.StoryForces((0.0, 3.0), ops=_mod(old))


@pytest.mark.parametrize("levels", [(0.0,), (0.0, 0.0), (3.0, 1.0), (0.0, float("nan"))])
def test_levels_are_checked(levels: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="levels"):
        B.StoryForces(levels, ops=_mod(FakeOps()))


def test_names_default_and_length_check() -> None:
    sf = B.StoryForces((0.0, 3.0), ops=_mod(FakeOps()))
    assert sf.stories == ("Story1",)
    sf = B.StoryForces((0.0, 3.0), names=("B", "Roof"), ops=_mod(FakeOps()))
    assert sf.stories == ("Roof",)
    with pytest.raises(ValueError, match="one name per level"):
        B.StoryForces((0.0, 3.0), names=("Roof",), ops=_mod(FakeOps()))


def test_drift_joints_and_story_max() -> None:
    fake = FakeOps()
    fake.disp[3] = [0.03, -0.06, 0, 0, 0, 0]
    fake.disp[4] = [0.03, 0.09, 0, 0, 0, 0]
    d = B.StoryDrifts((0.0, 3.0), ops=_mod(fake))
    assert d.labels == ("Story1:3:DX", "Story1:3:DY", "Story1:4:DX", "Story1:4:DY")
    np.testing.assert_allclose(d(_mod(fake)), [0.01, -0.02, 0.01, 0.03])
    t = d.story_max(ops=_mod(fake))
    assert t.columns == ("DriftX", "DriftY", "AvgX", "AvgY")
    # AvgY: the signed ends (x = 0 and x = 4) average |-0.02 + 0.03| / 2
    np.testing.assert_allclose(t.values, [[0.01, 0.03, 0.01, 0.005]])


def test_story_displacements_joints_points_and_story_max() -> None:
    fake = FakeOps()
    fake.disp[3] = [0.03, -0.06, 0, 0, 0, 0]
    fake.disp[4] = [-0.01, 0.09, 0, 0, 0, 0]
    u = B.StoryDisplacements((0.0, 3.0), ops=_mod(fake))
    assert u.labels == ("Story1:3:UX", "Story1:3:UY", "Story1:4:UX", "Story1:4:UY")
    np.testing.assert_allclose(u(_mod(fake)), [0.03, -0.06, -0.01, 0.09])
    t = u.story_max(ops=_mod(fake))
    assert t.columns == ("MaxX", "MaxY", "AvgX", "AvgY")
    # both nodes lie at y = 0, one end for X: AvgX = |0.03 - 0.01| / 2;
    # AvgY = |-0.06 + 0.09| / 2 over the ends x = 0 and x = 4
    np.testing.assert_allclose(t.values, [[0.03, 0.09, 0.01, 0.015]])
    p = B.StoryDisplacements((0.0, 3.0), points=[(4.0, 0.0)], ops=_mod(fake))
    assert p.table(ops=_mod(fake)).rows == ("Story1:P0",)
    np.testing.assert_allclose(p(_mod(fake)), [-0.01, 0.09])
    fake.incr[4] = [0.02, -0.01, 0, 0, 0, 0]          # an uncommitted trial
    np.testing.assert_allclose(p(_mod(fake)), [0.01, 0.08])


def test_a_drift_story_without_a_joint_refuses_naming_it() -> None:
    """Level 6 holds only an offset column top at x = 2: no node of it
    lies over a level-3 node, so story 2 has no joint."""
    fake = FakeOps()
    fake.nodes[5] = [2.0, 0.0, 6.0]
    fake.eles[4] = [4, 5]
    fake.disp[5] = [0.49 * 3.0, 0, 0, 0, 0, 0]
    with pytest.raises(BridgeError, match=r"story 'Story2': node\(s\) \[5\]"):
        B.StoryDrifts((0.0, 3.0, 6.0), ops=_mod(fake))
    # a point read still answers, and the story is not zero
    d = B.StoryDrifts((0.0, 3.0, 6.0), points=[(2.0, 0.0)], method="nearest",
                      max_offset=2.0, ops=_mod(fake))
    assert d.table(ops=_mod(fake)).row("Story2:P0")[0] == pytest.approx(0.49)


def test_a_cantilever_tip_without_a_node_below_refuses_naming_it() -> None:
    """Base nodes at x = 0, 10; the level adds a cantilever tip at x = 14.
    Under a rigid rotation (uy = 1e-3 x) the tip drifts 4.67e-3; dropping
    it read the story maximum as the x = 10 joint's 3.33e-3."""
    fake = _rotated_level((0.0, 10.0), 0.0, 1e-3)
    fake.nodes[103] = [14.0, 0.0, 3.0]
    fake.disp[103] = [0.0, 14e-3, 0, 0, 0, 1e-3]
    with pytest.raises(BridgeError, match=r"story 'Story1': node\(s\) \[103\]"):
        B.StoryDrifts((0.0, 3.0), ops=_mod(fake))
    # the rigid read of the tip over the base gives the true drift
    d = B.StoryDrifts((0.0, 3.0), points=[(14.0, 0.0)], method="rigid",
                      ops=_mod(fake))
    assert d.table(ops=_mod(fake)).row("Story1:P0")[1] == pytest.approx(14e-3 / 3)


def _rotated_level(xs: tuple[float, ...], x0: float, theta: float,
                   axis: str = "x") -> FakeOps:
    """Joints at ``c in xs`` along ``axis`` on z = 0 (fixed) and z = 3, the
    top level rotated rigidly by ``theta`` about the plan point ``c = x0``
    on that axis: ``uy = theta (x - x0)`` along x, ``ux = -theta (y - x0)``
    along y."""
    fake = FakeOps()
    fake.nodes = {}
    fake.eles = {}
    fake.disp = {}
    for k, c in enumerate(xs):
        xy = [c, 0.0] if axis == "x" else [0.0, c]
        u = [0.0, theta * (c - x0)] if axis == "x" else [-theta * (c - x0), 0.0]
        fake.nodes[1 + k] = [*xy, 0.0]
        fake.disp[1 + k] = [0.0] * 6
        fake.nodes[101 + k] = [*xy, 3.0]
        fake.disp[101 + k] = [*u, 0, 0, 0, theta]
    return fake


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(("xs", "x0", "ratio"), [
    # rotation about one end: the ends move 0 and 10 theta, so ASCE's
    # average is 5 theta and max/avg = 2.0 (a mean over the five joints
    # gives 10 / 2.12 = 4.72)
    ((0.0, 0.1, 0.2, 0.3, 10.0), 0.0, 2.0),
    # about an interior point the ends move in opposite directions:
    # -2 and +8 theta, signed average 3 theta, ratio 8 / 3
    ((0.0, 2.0, 10.0), 2.0, 8.0 / 3.0),
    # the review's repro: ends -3.33 and +6.67 theta (drifts -1.11e-3 and
    # +2.22e-3), signed average 1.67 theta, ratio 4.0 (not 1.33)
    ((0.0, 10.0), 10.0 / 3.0, 4.0),
])
def test_story_average_is_the_asce_signed_end_average(
        xs: tuple[float, ...], x0: float, ratio: float, axis: str) -> None:
    theta = 1e-3
    fake = _rotated_level(xs, x0, theta, axis)
    col = [1, 3] if axis == "x" else [0, 2]   # loaded direction: Y, then X
    u = B.StoryDisplacements((0.0, 3.0), ops=_mod(fake)).story_max(ops=_mod(fake))
    max_u, avg_u = u.row("Story1")[col]
    assert max_u == pytest.approx(theta * max(abs(c - x0) for c in xs))
    assert max_u / avg_u == pytest.approx(ratio)
    d = B.StoryDrifts((0.0, 3.0), ops=_mod(fake)).story_max(ops=_mod(fake))
    drift, davg = d.row("Story1")[col]
    assert drift == pytest.approx(max_u / 3.0)
    assert drift / davg == pytest.approx(ratio)


def test_story_table_accessors() -> None:
    t = B.StoryTable(("a", "b"), ("X", "Y"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(t.column("Y"), [2.0, 4.0])
    np.testing.assert_array_equal(t.row("b"), [3.0, 4.0])
    assert list(t.to_frame().index) == ["a", "b"]


def test_pier_from_pg_under_fem_element_tags() -> None:
    sel = SimpleNamespace(ids=[7, 8])
    fem = SimpleNamespace(elements=SimpleNamespace(select=lambda pg: sel))
    bridge = SimpleNamespace(fem=fem, _element_tags="fem")
    pier = B.Pier.from_pg(cast(Any, bridge), "Wall1", axis2=(1.0, 0.0))
    assert pier == B.Pier("Wall1", (7, 8), axis2=(1.0, 0.0))
    with pytest.raises(ValueError, match="empty"):
        B.Pier("W", ())


def test_pier_from_pg_through_the_live_read_session() -> None:
    """Under non-``fem`` element tags the FEM ids go through the bridge's
    read session: ids 7, 8 were emitted as OpenSees elements 100, 101."""
    from apeGmsh.opensees._internal.build.pg_expand import ElementPlanRows
    from apeGmsh.opensees._readback import ReadSession
    from apeGmsh.opensees.emitter.live import domain_epoch

    sel = SimpleNamespace(ids=[8, 7])
    fem = SimpleNamespace(elements=SimpleNamespace(select=lambda pg: sel))
    rows = ElementPlanRows(np.array([7, 8, 9], dtype=np.int64),
                           np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int64),
                           tag_start=100)
    session = ReadSession(emitter=cast(Any, None), ndm=3, epoch=domain_epoch(),
                          element_plan=((cast(Any, "spec"), rows),))
    bridge = SimpleNamespace(fem=fem, _element_tags="sequential",
                             _read_state=session)
    pier = B.Pier.from_pg(cast(Any, bridge), "Wall1", name="W")
    assert pier == B.Pier("W", (101, 100))
    bridge._read_state = None                     # no live analyze yet
    with pytest.raises(BridgeError, match="no live analyze"):
        B.Pier.from_pg(cast(Any, bridge), "Wall1")


def test_levels_from_pgs() -> None:
    coords = {"L0": np.array([[0, 0, 0.0], [1, 0, 0.0]]),
              "L1": np.array([[0, 0, 3.0], [1, 1, 3.0]]),
              "Bad": np.array([[0, 0, 3.0], [1, 1, 3.5]])}
    nodes = SimpleNamespace(
        coords=np.vstack(list(coords.values())),
        select=lambda pg: SimpleNamespace(coords=coords[pg]))
    fem = cast(Any, SimpleNamespace(nodes=nodes))
    assert B.levels_from_pgs(fem, ["L0", "L1"]) == (0.0, 3.0)
    with pytest.raises(ValueError, match="horizontal"):
        B.levels_from_pgs(fem, ["L0", "Bad"])
