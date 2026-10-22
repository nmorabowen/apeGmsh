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
    np.testing.assert_allclose(t.values, [[0.01, 0.03, 0.01, 0.025]])


def test_story_displacements_joints_points_and_story_max() -> None:
    fake = FakeOps()
    fake.disp[3] = [0.03, -0.06, 0, 0, 0, 0]
    fake.disp[4] = [-0.01, 0.09, 0, 0, 0, 0]
    u = B.StoryDisplacements((0.0, 3.0), ops=_mod(fake))
    assert u.labels == ("Story1:3:UX", "Story1:3:UY", "Story1:4:UX", "Story1:4:UY")
    np.testing.assert_allclose(u(_mod(fake)), [0.03, -0.06, -0.01, 0.09])
    t = u.story_max(ops=_mod(fake))
    assert t.columns == ("MaxX", "MaxY", "AvgX", "AvgY")
    np.testing.assert_allclose(t.values, [[0.03, 0.09, 0.02, 0.075]])
    p = B.StoryDisplacements((0.0, 3.0), points=[(4.0, 0.0)], ops=_mod(fake))
    assert p.table(ops=_mod(fake)).rows == ("Story1:P0",)
    np.testing.assert_allclose(p(_mod(fake)), [-0.01, 0.09])


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
