"""Parity: scene_ir adapter + backend reproduce build_fem_scene's grid.

ADR 0042 Phase R-A's acceptance gate. A backend rebuilding the grid
from the IR the adapter emits must match the legacy ``build_fem_scene``
grid: same points, same per-element corner-node sets.  The comparison
is keyed on ``element_id`` (not raw cell order) because the adapter
groups cells by neutral token while ``build_fem_scene`` keeps
group-iteration order — both must describe the *same* elements.

Headless: builds ``pyvista.UnstructuredGrid`` objects only (no GL).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.backends import mesh_layer_to_grid  # noqa: E402
from apeGmsh.viewers.scene._ir_adapter import (  # noqa: E402
    mesh_layer_from_viewer_data,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene  # noqa: E402


# --- Minimal duck-typed ViewerData stand-in ------------------------------
# build_fem_scene / the adapter only touch this surface.


@dataclass
class _EType:
    code: int
    dim: int
    npe: int


@dataclass
class _Group:
    element_type: _EType
    connectivity: np.ndarray
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class _Nodes:
    ids: Sequence[int]
    coords: np.ndarray


@dataclass
class _View:
    nodes: _Nodes
    elements: list


def _fixture_view() -> _View:
    node_ids = [10, 11, 12, 13, 14]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    tri = _Group(_EType(2, 2, 3), np.array([[10, 11, 12]]), np.array([100]))
    tet = _Group(_EType(4, 3, 4), np.array([[10, 11, 12, 13]]), np.array([200]))
    return _View(_Nodes(node_ids, coords), [tri, tet])


def _eid_to_pointset(grid) -> dict[int, frozenset[int]]:
    eids = np.asarray(grid.cell_data["element_id"])
    out: dict[int, frozenset[int]] = {}
    for i in range(grid.n_cells):
        out[int(eids[i])] = frozenset(int(p) for p in grid.get_cell(i).point_ids)
    return out


def test_adapter_grid_matches_legacy() -> None:
    view = _fixture_view()

    legacy = build_fem_scene(view).grid
    ir_grid = mesh_layer_to_grid(mesh_layer_from_viewer_data(view))

    assert ir_grid.n_points == legacy.n_points
    assert ir_grid.n_cells == legacy.n_cells
    np.testing.assert_allclose(
        np.asarray(ir_grid.points), np.asarray(legacy.points)
    )
    assert _eid_to_pointset(ir_grid) == _eid_to_pointset(legacy)


def test_adapter_carries_per_entity_color() -> None:
    view = _fixture_view()
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 2 cells
    layer = mesh_layer_from_viewer_data(view, element_rgb=rgb)
    assert layer.color.mode == "per_entity_rgb"

    grid = mesh_layer_to_grid(layer)
    colors = grid.cell_data["colors"]
    assert colors.dtype == np.uint8
    assert colors.shape == (2, 3)


def _tet10_fork_view() -> _View:
    """A BezierTet10-style group: a non-Gmsh fork code (-33001), 10 nodes.

    Corners first (Gmsh order), then the 6 edge mid-side nodes.  Before the
    ``(3, 10)`` fallback entry this group had no ``GMSH_LINEAR`` code and no
    ``(dim, npe)`` fallback, so it was dropped → an empty viewport.
    """
    node_ids = list(range(20, 30))
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],  # corners
            [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],                   # mid-side
            [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        ]
    )
    tet10 = _Group(_EType(-33001, 3, 10), np.array([list(range(20, 30))]),
                   np.array([300]))
    return _View(_Nodes(node_ids, coords), [tet10])


def test_fork_higher_order_tet10_renders_as_linear_tet() -> None:
    """A fork BezierTet10 (10-node tet, non-Gmsh code) renders as a linear
    VTK_TETRA on its 4 corner nodes — it must NOT be silently dropped.

    Regression for the empty-viewport bug: the results viewer keyed unknown
    element codes through ``GMSH_LINEAR_FALLBACK[(dim, npe)]``, which lacked
    ``(3, 10)``, so every 10-node tet was dropped (0 cells).
    """
    view = _tet10_fork_view()

    ir_grid = mesh_layer_to_grid(mesh_layer_from_viewer_data(view))
    assert ir_grid.n_cells == 1                       # was 0 before the fix
    assert int(ir_grid.celltypes[0]) == 10            # VTK_TETRA
    corners = _eid_to_pointset(ir_grid)[300]
    assert len(corners) == 4                          # linear corner subset

    # parity: the legacy build_fem_scene path renders it identically
    legacy = build_fem_scene(view).grid
    assert legacy.n_cells == 1
    assert _eid_to_pointset(legacy)[300] == corners
