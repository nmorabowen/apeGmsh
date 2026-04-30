"""Gauss-point → nodal extrapolation with cross-element averaging.

Used by ``ContourDiagram`` (and any other diagram that wants to render
continuous GP-valued fields as nodal contours).

Pipeline
--------

For each element in a ``GaussSlab`` with ``n_gp`` Gauss points:

1. Look up the element's shape function ``N`` and corner-node
   connectivity in the bound ``FEMData``.
2. Evaluate ``N`` at the GP natural coords → matrix
   ``A`` shape ``(n_gp, n_corner)``.
3. Compute the Moore–Penrose pseudo-inverse:
   ``M = pinv(A)`` shape ``(n_corner, n_gp)``.
   - When ``n_gp == n_corner`` (e.g. hex8 + 2×2×2 GPs, quad4 + 2×2)
     ``M`` is the exact inverse; constant + linear fields are
     reproduced exactly.
   - When ``n_gp < n_corner`` (e.g. tet4 / tri3 with one GP),
     ``M`` is the least-squares fit and reduces to "assign the GP
     value to every corner node."
   - When ``n_gp > n_corner`` (over-determined integration rule),
     ``M`` gives the least-squares projection.
4. Per timestep ``t``: ``nodal[t, corner] = M @ gp_values[t, :]``.
5. Accumulate each per-element nodal contribution into a global
   per-node sum + count; final nodal value is the mean across
   neighbouring elements.

Smoothing across element boundaries via nodal averaging is the
standard post-processing approach (STKO, ParaView, most academic
viewers). Sharp discontinuities at material interfaces are smeared —
that's a known trade and the price of a single-mesh nodal contour.
A future per-element subdivision path can preserve discontinuities
at the cost of substrate bookkeeping.

Public API
----------

``extrapolate_gauss_slab_to_nodes(slab, fem)``
    Return ``(node_ids, nodal_values)`` with ``nodal_values`` shape
    ``(T, N)`` matching the slab's time axis.

``per_element_max_gp_count(slab)``
    Quick check used by ``ContourDiagram`` to decide whether to take
    the cell-data (``n_gp == 1``) or the extrapolation path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy import ndarray

from ._shape_functions import get_shape_functions

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ._slabs import GaussSlab


# Parent dim per Gmsh type code — see catalog in _shape_functions.py
_PARENT_DIM: dict[int, int] = {
    1: 1,    # Line2
    2: 2,    # Tri3
    3: 2,    # Quad4
    4: 3,    # Tet4
    5: 3,    # Hex8
}


def per_element_max_gp_count(slab: "GaussSlab") -> int:
    """Return the largest number of GPs any single element has.

    Used as a quick discriminator: ``1`` means cell-constant rendering
    is sufficient; anything else needs the extrapolation pipeline.
    """
    eidx = np.asarray(slab.element_index, dtype=np.int64)
    if eidx.size == 0:
        return 0
    _, counts = np.unique(eidx, return_counts=True)
    return int(counts.max())


def _build_extrapolation_matrix(
    natural_gps: ndarray, gmsh_code: int,
) -> Optional[ndarray]:
    """Return M shape ``(n_corner, n_gp)`` or ``None`` if unsupported.

    Built by evaluating shape functions at the GP natural coords
    (matrix ``A`` shape ``(n_gp, n_corner)``) and computing
    ``pinv(A)``.
    """
    catalog = get_shape_functions(int(gmsh_code))
    if catalog is None:
        return None
    N_fn, _, _, _n_corner = catalog
    pdim = _PARENT_DIM.get(int(gmsh_code))
    if pdim is None:
        return None
    nat = np.asarray(natural_gps, dtype=np.float64)
    if nat.ndim == 1:
        nat = nat[:, None]
    nat_in = nat[:, :pdim]
    A = N_fn(nat_in)                # (n_gp, n_corner)
    return np.linalg.pinv(A)        # (n_corner, n_gp)


def _build_element_index(fem: "FEMData") -> dict[int, tuple[int, ndarray]]:
    """Map element ID → ``(gmsh_code, corner_node_ids)`` for the bound FEM."""
    out: dict[int, tuple[int, ndarray]] = {}
    for group in fem.elements:
        type_code = int(group.element_type.code)
        catalog = get_shape_functions(type_code)
        n_corner = catalog[3] if catalog is not None else None
        ids = np.asarray(group.ids, dtype=np.int64)
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for k in range(len(group)):
            eid = int(ids[k])
            row = conn[k]
            if n_corner is not None and row.size >= n_corner:
                row = row[:n_corner]
            out[eid] = (type_code, np.asarray(row, dtype=np.int64))
    return out


def extrapolate_gauss_slab_to_nodes(
    slab: "GaussSlab", fem: "FEMData",
) -> tuple[ndarray, ndarray]:
    """Extrapolate a ``GaussSlab`` to per-node values + average.

    Returns
    -------
    node_ids
        ``(N,)`` int64 — sorted FEM node IDs that received any
        contribution.
    nodal_values
        ``(T, N)`` float64 — averaged nodal values per time step,
        column-aligned with ``node_ids``.

    Empty slab returns ``(empty, (T, 0) array)``.
    """
    eidx = np.asarray(slab.element_index, dtype=np.int64)
    nat = np.asarray(slab.natural_coords, dtype=np.float64)
    values = np.asarray(slab.values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    T = values.shape[0]
    if eidx.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((T, 0), dtype=np.float64),
        )

    # FEM node-id → row index in the global node arrays
    fem_nids = np.asarray(list(fem.nodes.ids), dtype=np.int64)
    if fem_nids.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((T, 0), dtype=np.float64),
        )

    elem_index = _build_element_index(fem)

    # Sparse accumulators keyed by FEM node id. Dict lookups are fast
    # enough at result-viewer scale; avoiding a dense (N_fem, T) array
    # also keeps memory bounded on large meshes.
    sums: dict[int, ndarray] = {}
    counts: dict[int, int] = {}

    # Group slab rows by element id.
    order = np.argsort(eidx, kind="stable")
    eidx_sorted = eidx[order]
    splits = np.where(np.diff(eidx_sorted) != 0)[0] + 1
    groups = np.split(order, splits)

    for rows in groups:
        if rows.size == 0:
            continue
        eid = int(eidx[rows[0]])
        info = elem_index.get(eid)
        if info is None:
            # Element id not in FEM — skip.
            continue
        type_code, corner_nids = info
        n_gp_e = rows.size
        gp_vals = values[:, rows]    # (T, n_gp_e)
        nat_e = nat[rows]            # (n_gp_e, dim)

        if n_gp_e == 1:
            # Constant: assign the GP value to each corner equally.
            per_corner = np.broadcast_to(
                gp_vals, (T, corner_nids.size),
            )
        else:
            M = _build_extrapolation_matrix(nat_e, type_code)
            if M is None or M.shape[0] != corner_nids.size:
                # Unsupported element type — fall back to per-element
                # mean. Loses spatial detail but never wrong shape.
                mean_vals = gp_vals.mean(axis=1, keepdims=True)
                per_corner = np.broadcast_to(
                    mean_vals, (T, corner_nids.size),
                )
            else:
                # nodal[t, c] = sum_g M[c, g] * gp_vals[t, g]
                per_corner = gp_vals @ M.T    # (T, n_corner)

        for c, nid in enumerate(corner_nids):
            nid_i = int(nid)
            col = per_corner[:, c]
            existing = sums.get(nid_i)
            if existing is None:
                sums[nid_i] = col.astype(np.float64).copy()
                counts[nid_i] = 1
            else:
                existing += col
                counts[nid_i] += 1

    if not sums:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((T, 0), dtype=np.float64),
        )

    node_ids = np.fromiter(sums.keys(), dtype=np.int64, count=len(sums))
    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    nodal = np.zeros((T, node_ids.size), dtype=np.float64)
    for j, nid in enumerate(node_ids):
        nodal[:, j] = sums[int(nid)] / counts[int(nid)]
    return node_ids, nodal
