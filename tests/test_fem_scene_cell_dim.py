"""S4a acceptance — FEMSceneData.cell_dim + cell_indices_for_dims.

Headless: constructs FEMSceneData directly with a synthetic per-cell dim
array (the build's cell_dim accumulation is data-only and exercised by
any fem-scene build test). Locks the dim-filter primitive the results
0/1/2/3/4 filter will consume.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from apeGmsh.viewers.scene.fem_scene import FEMSceneData


def _scene(cell_dim) -> FEMSceneData:
    cell_dim = np.asarray(cell_dim, dtype=np.int8)
    n = cell_dim.size
    return FEMSceneData(
        grid=pv.UnstructuredGrid(),          # contents irrelevant to the query
        node_ids=np.array([], dtype=np.int64),
        node_id_to_idx={},
        cell_to_element_id=np.arange(n, dtype=np.int64),
        element_id_to_cell={},
        model_diagonal=1.0,
        cell_dim=cell_dim,
    )


def test_cell_indices_for_dims_selects_matching_cells() -> None:
    scene = _scene([1, 3, 2, 3])            # line, hex, quad, hex
    assert scene.cell_indices_for_dims([3]).tolist() == [1, 3]
    assert scene.cell_indices_for_dims([2, 3]).tolist() == [1, 2, 3]
    assert scene.cell_indices_for_dims([1]).tolist() == [0]
    assert scene.cell_indices_for_dims([0, 1, 2, 3]).tolist() == [0, 1, 2, 3]


def test_cell_indices_for_dims_empty_active_is_empty() -> None:
    scene = _scene([1, 2, 3])
    assert scene.cell_indices_for_dims([]).tolist() == []


def test_cell_indices_for_dims_empty_cell_dim_is_empty() -> None:
    scene = _scene([])
    assert scene.cell_indices_for_dims([1, 2, 3]).tolist() == []


def test_cell_dim_default_is_empty() -> None:
    # Omitting cell_dim (back-compat construction) yields an empty array,
    # not a crash — the field is defaulted.
    s = FEMSceneData(
        grid=pv.UnstructuredGrid(),
        node_ids=np.array([], dtype=np.int64),
        node_id_to_idx={},
        cell_to_element_id=np.array([], dtype=np.int64),
        element_id_to_cell={},
        model_diagonal=1.0,
    )
    assert s.cell_dim.size == 0
    assert s.cell_indices_for_dims([3]).tolist() == []
