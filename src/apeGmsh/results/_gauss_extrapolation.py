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

``extrapolate_gauss_slab_per_element(slab, fem)``
    Per-element extrapolated corner values, **no cross-element
    averaging**. Returns a dict ``{element_id: (T, n_corner)}``.
    Used by the discrete contour path that wants visible jumps at
    element boundaries.

``extrapolate_gauss_slab_to_nodes(slab, fem)``
    The averaged path. Returns ``(node_ids, nodal_values)`` with
    ``nodal_values`` shape ``(T, N)`` matching the slab's time axis.
    Built on top of the per-element core.

``per_element_max_gp_count(slab)``
    Quick check used by ``ContourDiagram`` to decide whether to take
    the cell-data (``n_gp == 1``) or the extrapolation path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy import ndarray

from ._shape_functions import get_shape_functions

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ._slabs import GaussSlab


# Parent dim per Gmsh type code — see catalog in _shape_functions.py
_PARENT_DIM: dict[int, int] = {
    1: 1,     # Line2
    2: 2,     # Tri3
    3: 2,     # Quad4
    4: 3,     # Tet4
    5: 3,     # Hex8
    6: 3,     # Wedge6
    9: 2,     # Tri6
    10: 2,    # Quad9
    11: 3,    # Tet10
    12: 3,    # Hex27
    16: 2,    # Quad8
    17: 3,    # Hex20
}


# For extrapolation we project GP values onto the **linear** corner
# shape functions only, even when the underlying element is quadratic.
# Reasons:
#   * The viewer's substrate is built with linear cells (mid-side / face
#     / center nodes are dropped in build_fem_scene), so values at
#     non-corner nodes are never painted.
#   * Using the full higher-order N matrix yields a pseudo-inverse that
#     produces non-constant nodal fields for a truly constant GP input
#     (minimum-norm regularization of the under-determined system),
#     which is the wrong behaviour for visualization.
#
# Mapping each higher-order code to its linear counterpart:
_LINEAR_COUNTERPART: dict[int, int] = {
    9: 2,     # Tri6   -> Tri3
    10: 3,    # Quad9  -> Quad4
    11: 4,    # Tet10  -> Tet4
    12: 5,    # Hex27  -> Hex8
    16: 3,    # Quad8  -> Quad4
    17: 5,    # Hex20  -> Hex8
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
    """Return M shape ``(n_real_corners, n_gp)`` or ``None`` if unsupported.

    For higher-order types (Tri6, Tet10, Quad8/9, Hex20/27) the GP
    values are projected onto the **linear** counterpart's corner
    shape functions — see ``_LINEAR_COUNTERPART`` for the mapping and
    rationale. This guarantees a constant GP field is reproduced as
    a constant nodal field at the corners (the only nodes the viewer
    actually paints).
    """
    code = int(gmsh_code)
    effective_code = _LINEAR_COUNTERPART.get(code, code)
    catalog = get_shape_functions(effective_code)
    if catalog is None:
        return None
    N_fn, _, _, _n_corner = catalog
    pdim = _PARENT_DIM.get(code)
    if pdim is None:
        return None
    nat = np.asarray(natural_gps, dtype=np.float64)
    if nat.ndim == 1:
        nat = nat[:, None]
    nat_in = nat[:, :pdim]
    A = N_fn(nat_in)                # (n_gp, n_real_corners)
    return np.linalg.pinv(A)        # (n_real_corners, n_gp)


def _build_element_index(fem: "FEMData") -> dict[int, tuple[int, ndarray]]:
    """Map element ID → ``(gmsh_code, real_corner_node_ids)``.

    Higher-order types are truncated to their linear-counterpart
    corner count (e.g. tri6 → first 3 nodes, hex27 → first 8 nodes)
    because that's what ``_build_extrapolation_matrix`` projects onto
    and what the substrate paints.
    """
    out: dict[int, tuple[int, ndarray]] = {}
    for group in fem.elements:
        type_code = int(group.element_type.code)
        effective_code = _LINEAR_COUNTERPART.get(type_code, type_code)
        catalog = get_shape_functions(effective_code)
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


@dataclass(frozen=True)
class PerElementCornerValues:
    """Per-element extrapolated corner values, no cross-element averaging.

    Attributes
    ----------
    element_ids
        ``(E,)`` int64 — FEM element IDs, in ascending order.
    corner_node_ids
        Length-``E`` list of ``(n_corner_e,)`` int64 arrays — corner
        node IDs in the same order as the element's
        ``group.connectivity[:, :n_corner]``. Matches the corner order
        used by :func:`apeGmsh.viewers.scene.fem_scene.build_fem_scene`,
        so a substrate cell's k-th point is the k-th entry here.
    values
        Length-``E`` list of ``(T, n_corner_e)`` float64 arrays —
        extrapolated values at each corner of each element, before any
        cross-element averaging.
    time_count
        Number of timesteps ``T``.
    """
    element_ids: ndarray
    corner_node_ids: list
    values: list
    time_count: int


def extrapolate_gauss_slab_per_element(
    slab: "GaussSlab", fem: "FEMData",
) -> PerElementCornerValues:
    """Extrapolate a ``GaussSlab`` to per-element corner values.

    No averaging across elements that share a node — each element keeps
    its own corner values. Used by the discrete contour path; the
    averaged path (:func:`extrapolate_gauss_slab_to_nodes`) is built on
    top of this.

    Empty slab returns an empty record with ``time_count`` derived from
    the slab's value array (``T == 1`` for a 1-D values array).
    """
    eidx = np.asarray(slab.element_index, dtype=np.int64)
    nat = np.asarray(slab.natural_coords, dtype=np.float64)
    values = np.asarray(slab.values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    T = values.shape[0]

    if eidx.size == 0:
        return PerElementCornerValues(
            element_ids=np.zeros(0, dtype=np.int64),
            corner_node_ids=[],
            values=[],
            time_count=T,
        )

    fem_nids = np.asarray(list(fem.nodes.ids), dtype=np.int64)
    if fem_nids.size == 0:
        return PerElementCornerValues(
            element_ids=np.zeros(0, dtype=np.int64),
            corner_node_ids=[],
            values=[],
            time_count=T,
        )

    elem_index = _build_element_index(fem)

    # Group slab rows by element id, in ascending element-id order so
    # downstream consumers get a stable iteration ordering.
    order = np.argsort(eidx, kind="stable")
    eidx_sorted = eidx[order]
    splits = np.where(np.diff(eidx_sorted) != 0)[0] + 1
    groups = np.split(order, splits)

    out_eids: list[int] = []
    out_corner_nids: list = []
    out_values: list = []

    for rows in groups:
        if rows.size == 0:
            continue
        eid = int(eidx[rows[0]])
        info = elem_index.get(eid)
        if info is None:
            continue
        type_code, corner_nids = info
        n_gp_e = rows.size
        gp_vals = values[:, rows]    # (T, n_gp_e)
        nat_e = nat[rows]            # (n_gp_e, dim)

        if n_gp_e == 1:
            per_corner = np.broadcast_to(
                gp_vals, (T, corner_nids.size),
            ).astype(np.float64, copy=True)
        else:
            M = _build_extrapolation_matrix(nat_e, type_code)
            if M is None or M.shape[0] != corner_nids.size:
                mean_vals = gp_vals.mean(axis=1, keepdims=True)
                per_corner = np.broadcast_to(
                    mean_vals, (T, corner_nids.size),
                ).astype(np.float64, copy=True)
            else:
                per_corner = gp_vals @ M.T    # (T, n_corner)
                per_corner = np.ascontiguousarray(per_corner, dtype=np.float64)

        out_eids.append(eid)
        out_corner_nids.append(corner_nids)
        out_values.append(per_corner)

    return PerElementCornerValues(
        element_ids=np.asarray(out_eids, dtype=np.int64),
        corner_node_ids=out_corner_nids,
        values=out_values,
        time_count=T,
    )


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
    per_elem = extrapolate_gauss_slab_per_element(slab, fem)
    T = per_elem.time_count
    if per_elem.element_ids.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((T, 0), dtype=np.float64),
        )

    sums: dict[int, ndarray] = {}
    counts: dict[int, int] = {}
    for corner_nids, per_corner in zip(
        per_elem.corner_node_ids, per_elem.values,
    ):
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
