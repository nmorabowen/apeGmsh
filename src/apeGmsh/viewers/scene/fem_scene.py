"""FEM scene builder — substrate mesh from a FEMData snapshot.

Parallel to ``mesh_scene.py`` but reads from a ``FEMData`` instead of
a live Gmsh session. Used by ``ResultsViewer``: a Results file embeds
its own FEMData, so the viewer must be able to render without a live
Gmsh model.

Builds:

* One merged ``pyvista.UnstructuredGrid`` containing all elements
  across all FEMData groups, with ``cell_data["element_id"]`` so
  future picking can map a picked cell to a FEM element ID.
* The node coordinate array, with ``point_data["node_id"]`` carrying
  raw FEM node IDs.
* A scipy KD-tree over the nodes for node picking (later phases).
* Model diagonal — used for sizing glyphs / overlays.

Phase 0 keeps the build minimal: one merged grid, cell↔element-id
map, KD-tree, model diagonal. The DimTag-based EntityRegistry from
the pre-solve viewer is **not** built here — Results-side picking
maps cell -> element_id directly. If we later want Part / PG batched
selection in the results viewer, that layer is added on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


# ======================================================================
# Gmsh -> VTK linearization
# ======================================================================
#
# Same table as ``viewers/scene/mesh_scene.py:GMSH_LINEAR``. Higher-
# order Gmsh element types render with their corner-node subset only
# (linear VTK cells); the wireframe / mid-side nodes are not part of
# the substrate mesh. Phase 0 does not render edges separately.

GMSH_LINEAR: dict[int, tuple[int, int]] = {
    # ── points ──────────────────────────────────────────────────
    15: (1,  1),    # 1-node point                   -> VTK_VERTEX
    # ── lines ───────────────────────────────────────────────────
    1:  (3,  2),    # 2-node line                    -> VTK_LINE
    8:  (3,  2),    # 3-node line     (P2)
    26: (3,  2),    # 4-node line     (P3)
    27: (3,  2),    # 5-node line     (P4)
    28: (3,  2),    # 6-node line     (P5)
    # ── triangles ───────────────────────────────────────────────
    2:  (5,  3),    # 3-node tri                     -> VTK_TRIANGLE
    9:  (5,  3),    # 6-node tri      (P2)
    21: (5,  3),    # 10-node tri     (P3)
    23: (5,  3),    # 15-node tri     (P4)
    25: (5,  3),    # 21-node tri     (P5)
    # ── quads ───────────────────────────────────────────────────
    3:  (9,  4),    # 4-node quad                    -> VTK_QUAD
    16: (9,  4),    # 8-node quad     (P2 serendipity)
    10: (9,  4),    # 9-node quad     (P2 + bubble)
    36: (9,  4),    # 16-node quad    (P3)
    37: (9,  4),    # 25-node quad    (P4)
    # ── tets ────────────────────────────────────────────────────
    4:  (10, 4),    # 4-node tet                     -> VTK_TETRA
    11: (10, 4),    # 10-node tet     (P2)
    29: (10, 4),    # 20-node tet     (P3)
    30: (10, 4),    # 35-node tet     (P4)
    # ── hexes ───────────────────────────────────────────────────
    5:  (12, 8),    # 8-node hex                     -> VTK_HEXAHEDRON
    17: (12, 8),    # 20-node hex     (P2 serendipity)
    12: (12, 8),    # 27-node hex     (P2 + bubbles)
    92: (12, 8),    # 64-node hex     (P3)
    93: (12, 8),    # 125-node hex    (P4)
    # ── prisms (wedges) ─────────────────────────────────────────
    6:  (13, 6),    # 6-node prism                   -> VTK_WEDGE
    18: (13, 6),    # 15-node prism   (P2 serendipity)
    13: (13, 6),    # 18-node prism   (P2 + bubbles)
    # ── pyramids ────────────────────────────────────────────────
    7:  (14, 5),    # 5-node pyramid                 -> VTK_PYRAMID
    19: (14, 5),    # 13-node pyramid (P2 serendipity)
    14: (14, 5),    # 14-node pyramid (P2 + bubble)
}


# Fallback for non-Gmsh element codes — MPCO synthesizes ``code =
# -class_tag`` (negative to avoid colliding with positive Gmsh codes;
# see ``mesh/_femdata_mpco_io.py``). Keyed on ``(dim, npe)``; only the
# linear corner subset is rendered.
GMSH_LINEAR_FALLBACK: dict[tuple[int, int], tuple[int, int]] = {
    (0, 1): (1,  1),    # vertex                         -> VTK_VERTEX
    (1, 2): (3,  2),    # 2-node line                    -> VTK_LINE
    (2, 3): (5,  3),    # 3-node tri                     -> VTK_TRIANGLE
    (2, 4): (9,  4),    # 4-node quad                    -> VTK_QUAD
    (3, 4): (10, 4),    # 4-node tet                     -> VTK_TETRA
    (3, 5): (14, 5),    # 5-node pyramid                 -> VTK_PYRAMID
    (3, 6): (13, 6),    # 6-node prism                   -> VTK_WEDGE
    (3, 8): (12, 8),    # 8-node hex                     -> VTK_HEXAHEDRON
}


# ======================================================================
# FEMSceneData
# ======================================================================

@dataclass
class FEMSceneData:
    """Everything the results viewer needs from the FEM scene build.

    Attributes
    ----------
    grid
        Merged ``pyvista.UnstructuredGrid`` of all elements (linear
        VTK cells; higher-order Gmsh nodes dropped).
    node_ids
        ``ndarray(N,)`` of FEM node IDs aligned with ``grid.points``.
    node_id_to_idx
        Mapping from FEM node ID to row index into ``grid.points``.
    cell_to_element_id
        ``ndarray(M,)`` of FEM element IDs aligned with ``grid.cells``.
    element_id_to_cell
        Mapping from FEM element ID to row index into the cell list.
    model_diagonal
        Length of the bounding-box diagonal — used for default glyph
        size, camera reset, etc.
    skipped_types
        Gmsh element type codes that the linearization table does not
        cover (typically exotic high-order types). Reported for
        diagnostics; their elements do not appear in ``grid``.
    actor
        The VTK actor for the substrate mesh, populated when the
        viewer adds it to the plotter.
    node_tree
        scipy ``cKDTree`` over the node coordinates — built lazily.
    """

    grid: pv.UnstructuredGrid
    node_ids: ndarray
    node_id_to_idx: dict[int, int]
    cell_to_element_id: ndarray
    element_id_to_cell: dict[int, int]
    model_diagonal: float
    skipped_types: list[int] = field(default_factory=list)
    actor: Any = None
    node_tree: Any = None      # scipy.spatial.cKDTree, lazy

    def ensure_node_tree(self):
        if self.node_tree is None:
            try:
                from scipy.spatial import cKDTree
                self.node_tree = cKDTree(np.asarray(self.grid.points))
            except ImportError:
                self.node_tree = None
        return self.node_tree


# ======================================================================
# Build
# ======================================================================

def build_fem_scene(
    fem: "FEMData",
    *,
    verbose: bool = False,
) -> FEMSceneData:
    """Build the substrate mesh + identity maps from a ``FEMData``.

    Parameters
    ----------
    fem
        The bound FEMData snapshot.
    verbose
        Log the build summary.

    Returns
    -------
    FEMSceneData
        Substrate mesh and identity maps; not yet attached to a plotter.
    """
    # ── Nodes ────────────────────────────────────────────────────
    raw_node_ids = np.asarray(list(fem.nodes.ids), dtype=np.int64)
    raw_node_coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if raw_node_coords.ndim != 2 or raw_node_coords.shape[1] != 3:
        raise RuntimeError(
            f"FEMData.nodes.coords must be (N, 3); got shape "
            f"{raw_node_coords.shape}."
        )
    if raw_node_ids.shape[0] != raw_node_coords.shape[0]:
        raise RuntimeError(
            f"FEMData node id / coord length mismatch: "
            f"ids={raw_node_ids.shape[0]} coords={raw_node_coords.shape[0]}."
        )

    n_nodes = raw_node_ids.shape[0]
    node_id_to_idx = {int(nid): i for i, nid in enumerate(raw_node_ids)}

    # ── Elements ────────────────────────────────────────────────
    cells_flat: list[int] = []
    cell_types: list[int] = []
    element_ids: list[int] = []
    skipped: dict[int, int] = {}

    for group in fem.elements:
        etype = group.element_type
        code = int(etype.code)
        mapping = GMSH_LINEAR.get(code)
        if mapping is None:
            mapping = GMSH_LINEAR_FALLBACK.get((int(etype.dim), int(etype.npe)))
        if mapping is None:
            skipped[code] = skipped.get(code, 0) + len(group)
            continue
        vtk_type, n_corner = mapping

        # Connectivity is (count, npe) in node-id space; we only keep
        # the first ``n_corner`` columns (linear corners).
        conn = np.asarray(group.connectivity, dtype=np.int64)
        ids = np.asarray(group.ids, dtype=np.int64)
        if conn.ndim != 2 or conn.shape[1] < n_corner:
            if verbose:
                print(
                    f"[fem_scene] skipping group code={code} "
                    f"(connectivity shape={conn.shape}, need >= "
                    f"{n_corner} cols)"
                )
            skipped[code] = skipped.get(code, 0) + len(group)
            continue
        corner = conn[:, :n_corner]

        # Map node IDs -> indices via dense lookup. Build a temporary
        # max-id+1 array; faster than per-element dict lookup.
        if n_nodes > 0:
            max_id = int(raw_node_ids.max())
            id_to_idx_dense = np.full(max_id + 2, -1, dtype=np.int64)
            id_to_idx_dense[raw_node_ids] = np.arange(n_nodes, dtype=np.int64)
            mapped = id_to_idx_dense[corner]
        else:
            mapped = corner

        # VTK cell layout: [npe, idx_0, idx_1, ..., idx_{npe-1}, npe, ...]
        npe_col = np.full((mapped.shape[0], 1), n_corner, dtype=np.int64)
        block = np.hstack([npe_col, mapped]).reshape(-1)
        cells_flat.append(block)
        cell_types.extend([vtk_type] * mapped.shape[0])
        element_ids.extend(int(eid) for eid in ids)

    if cells_flat:
        cells_arr = np.concatenate(cells_flat).astype(np.int64)
    else:
        cells_arr = np.array([], dtype=np.int64)

    cell_types_arr = np.asarray(cell_types, dtype=np.uint8)
    element_id_arr = np.asarray(element_ids, dtype=np.int64)

    grid = pv.UnstructuredGrid(cells_arr, cell_types_arr, raw_node_coords)
    if element_id_arr.size:
        grid.cell_data["element_id"] = element_id_arr
    grid.point_data["node_id"] = raw_node_ids

    # Cell index lookup (cell row in the merged grid -> element ID,
    # and reverse).
    element_id_to_cell = {
        int(eid): i for i, eid in enumerate(element_id_arr)
    }

    # ── Model diagonal ──────────────────────────────────────────
    if n_nodes:
        mins = raw_node_coords.min(axis=0)
        maxs = raw_node_coords.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        if not np.isfinite(diag) or diag <= 0.0:
            diag = 1.0
    else:
        diag = 1.0

    skipped_types = sorted(skipped.keys())
    if verbose:
        print(
            f"[fem_scene] built grid: nodes={n_nodes}, "
            f"cells={cell_types_arr.size}, types={len(set(cell_types))}, "
            f"diagonal={diag:.4g}, skipped={skipped}"
        )

    return FEMSceneData(
        grid=grid,
        node_ids=raw_node_ids,
        node_id_to_idx=node_id_to_idx,
        cell_to_element_id=element_id_arr,
        element_id_to_cell=element_id_to_cell,
        model_diagonal=diag,
        skipped_types=skipped_types,
    )
