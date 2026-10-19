"""GaussSlab.global_coords backing — shape-fn evaluation per element type.

Maps natural-space Gauss-point coordinates to world coordinates by
evaluating element shape functions against the bound FEMData's node
coordinates.

Backed by :mod:`apeGmsh.fem._shape_functions` (a vectorized
shape-function library mined from STKO_to_python and adapted to be
keyed by Gmsh element-type codes; shared with the input-side mass
resolver). Coverage:

* Hex8 (code 5) — trilinear in ``[-1, +1]^3``
* Quad4 (code 3) — bilinear in ``[-1, +1]^2``
* Tet4 (code 4) — linear barycentric, vertices at origin + unit axes
* Tri3 (code 2) — linear barycentric, vertices at (0,0)/(1,0)/(0,1)
* Line2 (code 1) — linear, ξ ∈ ``[-1, +1]``

For element types not in the catalog (pyramids, P3+; the catalog
covers lines / tris / quads / tets / hexes / wedges including their
P2 forms) we fall back to ``centroid + 0.5 * bbox_span * natural`` —
visualization-faithful for axis-aligned elements while we wait for
explicit shape-fn coverage. The upcoming-work block in
``apeGmsh.fem._shape_functions`` lists what's left to add. Degraded
reconstructions are LOUD (ADR 0056 INV-6): one aggregated
:class:`WarnGaussCoordsApproximate` per call reports how many GPs
took the bbox approximation (and which type codes) or could not be
resolved at all (left at the origin).

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

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ..fem._shape_functions import (
    SHAPE_FUNCTIONS_BY_GMSH_CODE,
    get_shape_functions,
)

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ._slabs import GaussSlab


class WarnGaussCoordsApproximate(UserWarning):
    """GP world-coordinate reconstruction was degraded.

    Raised (as a warning, once per reconstruction call) when Gauss
    points were positioned by the centroid+bbox approximation
    (element type lacks shape-function coverage) or left at the
    origin (parent element missing from the FEMData / connectivity
    references unknown nodes). A silently approximated marker cloud
    is indistinguishable from a correct one — say so (ADR 0056
    INV-6).
    """


# Re-export for legacy / test imports — these paths existed before the
# shape_functions library was lifted from STKO. They wrap the new
# library to keep call sites that took a single ``(3,)`` natural coord
# working unchanged.

def _hex8_shape_functions(natural_xyz: ndarray) -> ndarray:
    """Single-IP wrapper around ``hex8_N`` — input ``(3,)``, output ``(8,)``."""
    from ..fem._shape_functions import hex8_N
    return hex8_N(np.asarray(natural_xyz, dtype=np.float64).reshape(1, 3))[0]


def _quad4_shape_functions(natural_xy: ndarray) -> ndarray:
    """Single-IP wrapper around ``quad4_N`` — input ``(2,)``, output ``(4,)``."""
    from ..fem._shape_functions import quad4_N
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
# Self-describing (Bézier / Bernstein) reconstruction via the neutral    #
# basis library                                                          #
# --------------------------------------------------------------------- #

def _bezier_basis_spec(element_type) -> "tuple[str, str] | None":
    """``(topology, family)`` for a recognized Bézier element, else None.

    The ``.ladruno`` reader's FEMData names these ``BezierTri6`` /
    ``BezierTet10`` (negated class-tag codes), so the linear Gmsh
    shape-function catalog can't reconstruct their Gauss-point world
    coords — they need the Bernstein basis ``x = B(ξ)·X`` over **all**
    control points (the file's ``GP_PARAM`` ξ, never a catalog GP order;
    see the Bézier plan's Tri6 GP-index caveat). This routes only Bézier
    families through :func:`apeGmsh._basis.basis_values`; every other type
    keeps the existing linear-catalog / bbox path unchanged.
    """
    name = (getattr(element_type, "gmsh_name", "") or "").lower()
    if "beziertri" in name:
        return ("tri", "bernstein")
    if "beziertet" in name:
        return ("tet", "bernstein")
    return None


def _world_via_basis(
    natural_coord: ndarray,
    node_coords: ndarray,
    *,
    topology: str,
    family: str,
    order: int,
) -> "ndarray | None":
    """``x = B(ξ)·X`` over the element's control points, or None on a
    basis/shape mismatch (caller falls back).

    ``natural_coord`` is gmsh ``(u, v[, w])`` (the ``GaussSlab`` contract,
    #2228); :func:`apeGmsh._basis.basis_values` takes the element's own
    full barycentric row, so it is rebuilt here: ``(ξ1, ξ2, ξ3) = (u, v,
    1 - u - v)`` for the triangle (``N0 = ξ3²``), ``(L1, L2, L3, L4) =
    (1 - u - v - w, u, v, w)`` for the tet (``N0 = L1²``).
    """
    from .._basis import BasisError, basis_values
    xi = np.asarray(natural_coord, dtype=np.float64).reshape(1, -1)
    rest = 1.0 - xi.sum(axis=1, keepdims=True)
    if topology == "tri" and xi.shape[1] == 2:
        xi = np.hstack([xi, rest])
    elif topology == "tet" and xi.shape[1] == 3:
        xi = np.hstack([rest, xi])
    try:
        r = basis_values(
            topology=topology, family=family, order=order, xi=xi,
        )
    except BasisError:
        return None
    if r.shape[1] != node_coords.shape[0]:
        return None
    return np.asarray(r[0] @ node_coords, dtype=np.float64)


# --------------------------------------------------------------------- #
# Top-level entry                                                       #
# --------------------------------------------------------------------- #

def compute_global_coords(
    slab: "GaussSlab",
    fem: "FEMData",
    *,
    node_coords_override: "ndarray | None" = None,
) -> ndarray:
    """Return ``(sum_GP, 3)`` world coords for every GP in ``slab``.

    ``node_coords_override`` (optional) replaces ``fem.nodes.coords``
    for the shape-function evaluation. The override must be a
    ``(len(fem.nodes.ids), 3)`` array row-aligned with ``fem.nodes.ids``
    — typically the deformed substrate points from
    :class:`FEMSceneData.grid` after a Geometry's deformation has been
    applied. Useful for visualizing GP markers on the deformed shape.
    """
    return compute_global_coords_from_arrays(
        np.asarray(slab.element_index, dtype=np.int64),
        np.asarray(slab.natural_coords, dtype=np.float64),
        fem,
        node_coords_override=node_coords_override,
    )


def compute_global_coords_from_arrays(
    element_index: ndarray,
    natural_coords: ndarray,
    fem: "FEMData",
    *,
    node_coords_override: "ndarray | None" = None,
) -> ndarray:
    """Lower-level entry — same as :func:`compute_global_coords` but
    takes ``element_index`` and ``natural_coords`` directly. Useful
    when a caller has cached those arrays from an earlier slab read
    and wants to recompute world coords against a different node
    coords array (e.g. the deformed substrate)."""
    eids = np.asarray(element_index, dtype=np.int64)
    nat = np.asarray(natural_coords, dtype=np.float64)
    if nat.ndim == 1:
        nat = nat[:, None]
    n_gp = eids.size
    out = np.zeros((n_gp, 3), dtype=np.float64)
    if n_gp == 0:
        return out

    # ── Lookup: FEM node id -> coord row ───────────────────────────
    node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
    coords_arr = (
        np.asarray(node_coords_override, dtype=np.float64)
        if node_coords_override is not None
        else np.asarray(fem.nodes.coords, dtype=np.float64)
    )
    if node_ids_arr.size == 0:
        return out
    if coords_arr.shape[0] != node_ids_arr.shape[0]:
        # Override length must match fem.nodes.ids — silently fall
        # back to the embedded fem coords.
        coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
    max_nid = int(node_ids_arr.max())
    nid_to_idx = np.full(max_nid + 2, -1, dtype=np.int64)
    nid_to_idx[node_ids_arr] = np.arange(
        node_ids_arr.size, dtype=np.int64,
    )

    # ── Per-element info: type code + node coords ─────────────────
    # MPCO-derived FEMData carries synthetic negative codes (e.g.
    # ``-33`` for Tri31) that don't exist in the Gmsh-keyed shape
    # function catalog. Map them to the equivalent Gmsh linear code
    # by (dim, npe) so the shape function evaluation hits instead of
    # falling through to the bbox approximation.
    _GMSH_CODE_BY_DIM_NPE: dict[tuple[int, int], int] = {
        (1, 2): 1,   # Line2
        (2, 3): 2,   # Tri3
        (2, 4): 3,   # Quad4
        (3, 4): 4,   # Tet4
        (3, 8): 5,   # Hex8
    }
    needed = set(int(e) for e in np.unique(eids))
    eid_info: dict[int, tuple[int, ndarray, "tuple[str, str, int] | None"]] = {}
    for group in fem.elements:
        et = group.element_type
        raw_code = int(et.code)
        type_code = (
            raw_code if raw_code >= 0
            else _GMSH_CODE_BY_DIM_NPE.get(
                (int(et.dim), int(et.npe)), raw_code,
            )
        )
        bezier = _bezier_basis_spec(et)
        basis_spec = (
            (bezier[0], bezier[1], int(et.order)) if bezier is not None
            else None
        )
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
            eid_info[eid] = (type_code, coords_arr[idxs], basis_spec)

    # ── Per-GP: pick shape fn, evaluate, scatter ───────────────────
    # Degradation counters — aggregated into ONE warning below.
    n_bbox = 0
    bbox_codes: set[int] = set()
    unresolved_eids: set[int] = set()
    for k in range(n_gp):
        eid = int(eids[k])
        info = eid_info.get(eid)
        if info is None:
            # No parent element / unknown nodes — the GP stays at the
            # ORIGIN, which is worse than approximate. Counted loud.
            unresolved_eids.add(eid)
            continue
        type_code, node_coords, basis_spec = info
        # Self-describing Bézier elements: reconstruct via the Bernstein
        # basis over all control points (the file's GP_PARAM ξ). The linear
        # Gmsh catalog has no entry for them, so without this they would
        # fall through to the bbox approximation.
        if basis_spec is not None:
            topo, fam, order = basis_spec
            world = _world_via_basis(
                nat[k], node_coords,
                topology=topo, family=fam, order=order,
            )
            if world is not None:
                out[k] = world
                continue
        catalog_entry = get_shape_functions(type_code)
        if catalog_entry is None:
            # Fallback for unsupported types.
            out[k] = _world_via_bbox(nat[k], node_coords)
            n_bbox += 1
            bbox_codes.add(type_code)
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
            n_bbox += 1
            bbox_codes.add(type_code)
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
        if type_code == 1:                          # Line2
            nat_in = nat_row[:1].reshape(1, 1)
        elif type_code in (2, 3, 9, 10, 16):        # Tri3, Quad4, Tri6, Quad9, Quad8
            nat_in = nat_row[:2].reshape(1, 2)
        else:                                       # Tet4, Hex8, Wedge6, Tet10, Hex27, Hex20
            nat_in = nat_row[:3].reshape(1, 3)

        N = N_fn(nat_in)                  # (1, n_nodes)
        out[k] = (N[0] @ corner_coords)

    if n_bbox or unresolved_eids:
        parts: list[str] = []
        if n_bbox:
            parts.append(
                f"{n_bbox} GP(s) positioned by the centroid+bbox "
                f"approximation (element type code(s) "
                f"{sorted(bbox_codes)} have no shape-function "
                f"coverage; exact only for axis-aligned elements)"
            )
        if unresolved_eids:
            sample = sorted(unresolved_eids)[:5]
            more = (
                f", ... ({len(unresolved_eids) - 5} more)"
                if len(unresolved_eids) > 5 else ""
            )
            parts.append(
                f"{len(unresolved_eids)} element(s) could not be "
                f"resolved against the FEMData (missing element or "
                f"unknown nodes) — their GPs sit at the ORIGIN; "
                f"eids: {sample}{more}"
            )
        warnings.warn(
            "GP world-coordinate reconstruction degraded: "
            + "; ".join(parts),
            WarnGaussCoordsApproximate,
            stacklevel=2,
        )
    return out
