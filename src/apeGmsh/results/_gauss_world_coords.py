"""GaussSlab.global_coords backing — shape-fn evaluation per element type.

Maps natural-space Gauss-point coordinates to world coordinates by
evaluating element shape functions against the bound FEMData's node
coordinates.

Backed by :mod:`apeGmsh.results._shape_functions` (a vectorized
shape-function library mined from STKO_to_python and adapted to be
keyed by Gmsh element-type codes). Coverage:

* Hex8 (code 5) — trilinear in ``[-1, +1]^3``
* Quad4 (code 3) — bilinear in ``[-1, +1]^2``
* Tet4 (code 4) — linear barycentric, vertices at origin + unit axes
* Tri3 (code 2) — linear barycentric, vertices at (0,0)/(1,0)/(0,1)
* Line2 (code 1) — linear, ξ ∈ ``[-1, +1]``

For element types not in the catalog (higher-order P2/P3, prisms,
pyramids) we fall back to ``centroid + 0.5 * bbox_span * natural`` —
visualization-faithful for axis-aligned elements while we wait for
explicit shape-fn coverage. The upcoming-work block in
``_shape_functions.py`` lists what's left to add.

Per-element evaluation: rather than rely on STKO's "all elements
same IP layout" batched einsum (which doesn't hold across our
flattened slabs), we evaluate per element using the catalog's
single-IP API. The shape functions accept a ``(1, dim)`` batch and
return ``(1, n_nodes)``; one matrix-vector multiply per element gives
the world coord.

Returns
-------
ndarray (sum_GP, 3)
    World coordinates per GP, in the order of
    ``slab.element_index`` / ``slab.natural_coords``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ._shape_functions import (
    SHAPE_FUNCTIONS_BY_GMSH_CODE,
    get_shape_functions,
)

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ._slabs import GaussSlab


# Re-export for legacy / test imports — these paths existed before the
# shape_functions library was lifted from STKO. They wrap the new
# library to keep call sites that took a single ``(3,)`` natural coord
# working unchanged.

def _hex8_shape_functions(natural_xyz: ndarray) -> ndarray:
    """Single-IP wrapper around ``hex8_N`` — input ``(3,)``, output ``(8,)``."""
    from ._shape_functions import hex8_N
    return hex8_N(np.asarray(natural_xyz, dtype=np.float64).reshape(1, 3))[0]


def _quad4_shape_functions(natural_xy: ndarray) -> ndarray:
    """Single-IP wrapper around ``quad4_N`` — input ``(2,)``, output ``(4,)``."""
    from ._shape_functions import quad4_N
    return quad4_N(np.asarray(natural_xy, dtype=np.float64).reshape(1, 2))[0]


# Corner-node natural coordinates (kept for tests that pinned to the
# old import path).
_HEX8_CORNERS = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=np.float64)

_QUAD4_CORNERS = np.array([
    [-1, -1], [+1, -1], [+1, +1], [-1, +1],
], dtype=np.float64)


# --------------------------------------------------------------------- #
# Centroid + bbox fallback for unsupported types                        #
# --------------------------------------------------------------------- #

def _world_via_bbox(
    natural_coord: ndarray,
    node_coords: ndarray,
) -> ndarray:
    """Approximate world coord: centroid + 0.5 * bbox_span * natural."""
    centroid = node_coords.mean(axis=0)
    span = node_coords.max(axis=0) - node_coords.min(axis=0)
    nat3 = np.zeros(3, dtype=np.float64)
    nat = np.asarray(natural_coord, dtype=np.float64).ravel()
    n = min(nat.size, 3)
    nat3[:n] = nat[:n]
    return centroid + 0.5 * span * nat3


# --------------------------------------------------------------------- #
# Top-level entry                                                       #
# --------------------------------------------------------------------- #

def compute_global_coords(slab: "GaussSlab", fem: "FEMData") -> ndarray:
    """Return ``(sum_GP, 3)`` world coords for every GP in ``slab``."""
    eids = np.asarray(slab.element_index, dtype=np.int64)
    nat = np.asarray(slab.natural_coords, dtype=np.float64)
    if nat.ndim == 1:
        nat = nat[:, None]
    n_gp = eids.size
    out = np.zeros((n_gp, 3), dtype=np.float64)
    if n_gp == 0:
        return out

    # ── Lookup: FEM node id -> coord row ───────────────────────────
    node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
    coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
    if node_ids_arr.size == 0:
        return out
    max_nid = int(node_ids_arr.max())
    nid_to_idx = np.full(max_nid + 2, -1, dtype=np.int64)
    nid_to_idx[node_ids_arr] = np.arange(
        node_ids_arr.size, dtype=np.int64,
    )

    # ── Per-element info: type code + node coords ─────────────────
    needed = set(int(e) for e in np.unique(eids))
    eid_info: dict[int, tuple[int, ndarray]] = {}
    for group in fem.elements:
        type_code = int(group.element_type.code)
        ids = np.asarray(group.ids, dtype=np.int64)
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for k in range(len(group)):
            eid = int(ids[k])
            if eid not in needed:
                continue
            row = conn[k]
            idxs = nid_to_idx[row]
            valid = idxs >= 0
            if not valid.all():
                continue
            eid_info[eid] = (type_code, coords_arr[idxs])

    # ── Per-GP: pick shape fn, evaluate, scatter ───────────────────
    for k in range(n_gp):
        eid = int(eids[k])
        info = eid_info.get(eid)
        if info is None:
            continue
        type_code, node_coords = info
        catalog_entry = get_shape_functions(type_code)
        if catalog_entry is None:
            # Fallback for unsupported types.
            out[k] = _world_via_bbox(nat[k], node_coords)
            continue

        N_fn, _, _, n_corner = catalog_entry
        # Use only the corner nodes for the shape function — higher-
        # order Gmsh types (P2 etc.) include mid-side nodes that the
        # linear shape fn doesn't see. The catalog's ``n_corner`` is
        # the number we need to take from the connectivity.
        corner_coords = node_coords[:n_corner]
        if corner_coords.shape[0] != n_corner:
            # Connectivity smaller than expected — fall back.
            out[k] = _world_via_bbox(nat[k], node_coords)
            continue

        # Trim natural coord to the right parent dim. line2: 1, tri3/
        # quad4: 2, tet4/hex8: 3.
        parent_dim = corner_coords.shape[0]    # not the right metric
        # Determine parent_dim from the catalog's dN result instead.
        # For Phase 4 we just trim to whatever the shape fn expects;
        # peek at ``N_fn`` by checking its first call below.
        # Practically: line2 needs 1-D, tri3/quad4 need 2-D, tet4/hex8
        # need 3-D. Use type_code to map.
        nat_row = nat[k]
        if type_code == 1:                # Line2
            nat_in = nat_row[:1].reshape(1, 1)
        elif type_code in (2, 3):         # Tri3 / Quad4
            nat_in = nat_row[:2].reshape(1, 2)
        else:                              # Tet4 / Hex8
            nat_in = nat_row[:3].reshape(1, 3)

        N = N_fn(nat_in)                  # (1, n_nodes)
        out[k] = (N[0] @ corner_coords)
    return out
