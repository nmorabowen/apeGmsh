"""``g.decouple_node_set`` — decoupled nodes by group name (ADR 0119 D1).

Oracles, independent of the code under test: the source nodes and their
coordinates are read back from the FEM snapshot's own physical group;
the expected offsets are written out here; the ``equal_dof`` records are
compared with the pairs the handle reports.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.records._kinds import ConstraintKind

LX, LY = 4.0, 2.0


def _grid(build):
    """A LX x LY quad grid (1.0 spacing, PG ``Mat``); ``build(g)`` declares."""
    with apeGmsh(model_name="decouple_set", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, LX, LY, label="mat")
        g.physical.add_surface("mat", name="Mat")
        handles = build(g)
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=2)
    return fem, handles


def _xyz(fem) -> dict[int, np.ndarray]:
    return {int(n): np.asarray(c, dtype=float)
            for n, c in zip(fem.nodes.ids, fem.nodes.coords)}


def test_one_node_per_source_node_at_the_offset() -> None:
    fem, gnd = _grid(lambda g: g.decouple_node_set(
        "Mat", offset=(0.0, 0.0, -5.0), label="gnd"))
    src = sorted(int(t) for t in fem.nodes.physical.node_ids("Mat"))
    assert gnd.source_ids == tuple(src)
    assert len(gnd.tags) == len(src) == 15
    assert set(gnd.tags) == {int(t) for t in fem.nodes.decoupled_ids}
    assert min(gnd.tags) > max(src)
    xyz = _xyz(fem)
    for s, n in gnd.pairs().items():
        np.testing.assert_allclose(xyz[n], xyz[s] + [0.0, 0.0, -5.0])


def test_tags_continue_after_single_decoupled_nodes() -> None:
    def build(g):
        one = g.decouple_node(coords=(9.0, 9.0, 9.0), label="lone")
        return one, g.decouple_node_set("Mat", label="side")

    fem, (one, side) = _grid(build)
    assert side.tags[0] == one.tag + 1
    assert list(side.tags) == list(range(one.tag + 1, one.tag + 16))


def test_callable_offset_gets_the_source_coordinates() -> None:
    seen = {}

    def outward(xyz):
        seen["xyz"] = xyz.copy()
        out = np.zeros_like(xyz)
        out[:, 0] = np.where(xyz[:, 0] < LX / 2, -1.0, 1.0)
        return out

    fem, gnd = _grid(lambda g: g.decouple_node_set("Mat", offset=outward))
    xyz = _xyz(fem)
    np.testing.assert_allclose(
        seen["xyz"], np.array([xyz[s] for s in gnd.source_ids]))
    for s, n in gnd.pairs().items():
        dx = -1.0 if xyz[s][0] < LX / 2 else 1.0
        np.testing.assert_allclose(xyz[n], xyz[s] + [dx, 0.0, 0.0])


def test_tie_dofs_add_one_equal_dof_per_pair() -> None:
    fem, side = _grid(lambda g: g.decouple_node_set(
        "Mat", label="side", tie_dofs=(1, 2, 3)))
    recs = [r for r in fem.nodes.constraints.pairs()
            if r.kind == ConstraintKind.EQUAL_DOF]
    assert {(r.master_node, r.slave_node) for r in recs} == set(
        side.pairs().items())
    assert all(list(r.dofs) == [1, 2, 3] for r in recs)


def test_no_tie_dofs_adds_no_constraint() -> None:
    fem, _ = _grid(lambda g: g.decouple_node_set("Mat"))
    assert not list(fem.nodes.constraints.pairs())


def test_nodes_and_ties_survive_the_h5_round_trip(tmp_path) -> None:
    from apeGmsh.mesh.FEMData import FEMData

    fem, side = _grid(lambda g: g.decouple_node_set(
        "Mat", offset=(0.0, 0.0, -1.0), tie_dofs=(1, 2, 3)))
    path = tmp_path / "m.h5"
    fem.to_h5(str(path))
    back = FEMData.from_h5(str(path))
    assert {int(t) for t in back.nodes.decoupled_ids} == set(side.tags)
    pairs = {(r.master_node, r.slave_node)
             for r in back.nodes.constraints.pairs()}
    assert pairs == set(side.pairs().items())


@pytest.mark.filterwarnings("ignore:autosave to")
def test_unknown_source_fails_loud() -> None:
    with pytest.raises(ValueError, match="names no label or physical group"):
        _grid(lambda g: g.decouple_node_set("Nope"))


@pytest.mark.filterwarnings("ignore:autosave to")
def test_bad_offset_callable_fails_loud() -> None:
    with pytest.raises(ValueError, match="offset callable"):
        _grid(lambda g: g.decouple_node_set(
            "Mat", offset=lambda xyz: np.zeros((1, 3))))


@pytest.mark.parametrize("kw, match", [
    ({"offset": (1.0, 2.0)}, "triple"),
    ({"tie_dofs": (0, 1)}, "tie_dofs"),
    ({"tie_dofs": (1, 1)}, "tie_dofs"),
    ({"tie_dofs": ()}, "tie_dofs"),
])
def test_declaration_refusals(kw, match) -> None:
    with apeGmsh(model_name="decouple_set_bad", verbose=False) as g:
        with pytest.raises(ValueError, match=match):
            g.decouple_node_set("Mat", **kw)
        assert not g.decoupled_nodes.node_set_defs


@pytest.mark.filterwarnings("ignore:autosave to")
def test_pairs_before_extraction_raises() -> None:
    with apeGmsh(model_name="decouple_set_early", verbose=False) as g:
        h = g.decouple_node_set("Mat")
        with pytest.raises(ValueError, match="get_fem_data"):
            h.pairs()
