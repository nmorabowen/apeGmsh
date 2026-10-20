"""GaussSlab.global_coords backing — shape-fn evaluation per element type.

Maps natural-space Gauss-point coordinates to world coordinates by
evaluating element shape functions against the bound FEMData's node
coordinates.

Backed by :mod:`apeGmsh.fem._shape_functions` (a vectorized
shape-function library mined from STKO_to_python and adapted to be
keyed by Gmsh element-type codes; shared with the input-side mass
resolver). The catalog covers lines, tris, quads, tets, hexes and
wedges, P2 forms included, each over its full node row.

Either FEMData: a gmsh-built one keys each group by its gmsh code. One
read from a result file's own ``MODEL`` group keys it by ``-class_tag``,
which names no topology, so such an element is resolved to a gmsh shape
from its node count and the width of its own Gauss rows, by the table
the Gauss-to-node extrapolation uses. Bézier classes go through the
Bernstein basis instead.

Element types with no shape functions (pyramids, P3+) fall back to
``centroid + 0.5 * bbox_span * natural``, exact only for axis-aligned
elements. Degraded reconstructions are LOUD (ADR 0056 INV-6): one
aggregated :class:`WarnGaussCoordsApproximate` per call names each
class that took the bbox approximation and why, and the GPs that could
not be placed at all (left at the origin).

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

import math
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy import ndarray

from ..fem._shape_functions import (
    SHAPE_FUNCTIONS_BY_GMSH_CODE,
    get_shape_functions,
)
from ._gauss_extrapolation import (
    _GMSH_CODE_BY_PARENT_DIM_NPE,
    _PARENT_DIM,
    _gp_parent_dim,
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
    xi = np.atleast_2d(np.asarray(natural_coord, dtype=np.float64))
    single = np.ndim(natural_coord) == 1
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
    world = np.asarray(r @ node_coords, dtype=np.float64)
    return world[0] if single else world


def _simplex_order(topology: str, n_ctrl: int) -> "int | None":
    """The order of a complete simplex basis on ``n_ctrl`` control points.

    Taken from the control-point count, never the FEMData group's
    ``order``: the MPCO reader's FEMData calls a BezierTri6 order 1, and
    an order-1 basis has 3 functions against its 6 control points.
    """
    n_free = 2 if topology == "tri" else 3
    for p in range(1, 21):
        if math.comb(p + n_free, n_free) == n_ctrl:
            return p
    return None


class _ElementInfo(NamedTuple):
    """One FEMData element as the GP placement sees it."""
    code: int            # gmsh code, or -class_tag for a file-coded FEMData
    node_coords: ndarray  # (npe, 3), in connectivity order
    label: str           # names the class in a degradation warning
    dim: "int | None"    # the group's dim; read for file-coded groups only
    bezier: "tuple[str, str] | None"


def _place_element(
    info: _ElementInfo, nat_e: ndarray,
) -> "tuple[ndarray | None, str | None, bool]":
    """``(world, None, True)`` for an element placed by its own basis.

    Otherwise ``(None, reason, by_bbox)``: ``by_bbox`` says whether the
    caller may still approximate the GPs by the centroid+bbox map (their
    natural coordinates are usable) or must leave them at the origin.

    A file-coded element (``code = -class_tag``) is resolved to a gmsh
    shape exactly as the Gauss-to-node extrapolation resolves it: from its
    node count and the width of its own Gauss rows. Every class of these
    shapes writes its nodes in gmsh order (corners first, then mid-edge
    nodes on gmsh's edges), so the gmsh catalog's full P2 shape functions
    are the element's own.
    """
    npe = int(info.node_coords.shape[0])
    if info.bezier is not None:
        topo, fam = info.bezier
        need = 2 if topo == "tri" else 3
        if nat_e.shape[1] < need or not np.isfinite(nat_e[:, :need]).all():
            return None, (
                f"its Gauss rows carry no {need}-D natural coordinates"
            ), False
        p = _simplex_order(topo, npe)
        world = None if p is None else _world_via_basis(
            nat_e[:, :need], info.node_coords,
            topology=topo, family=fam, order=p,
        )
        if world is None:
            return None, (
                f"no {fam} {topo} basis on its {npe} control points"
            ), True
        return world, None, True

    if info.code >= 0:
        gmsh_code: int = info.code
    else:
        pdim = _gp_parent_dim(nat_e)
        if pdim is None:
            return None, "its Gauss rows carry no natural coordinates", False
        if info.dim is not None and pdim > info.dim:
            return None, (
                f"its {pdim}-D Gauss rows contradict its group's "
                f"dim={info.dim}"
            ), True
        resolved = _GMSH_CODE_BY_PARENT_DIM_NPE.get((pdim, npe))
        if resolved is None:
            return None, (
                f"no shape functions for a {npe}-node element with "
                f"{pdim}-D Gauss points"
            ), True
        gmsh_code = resolved

    catalog = get_shape_functions(gmsh_code)
    need_dim = _PARENT_DIM.get(gmsh_code)
    if catalog is None or need_dim is None:
        return None, f"no shape functions for gmsh element type {gmsh_code}", True
    N_fn, _, _, n_nodes = catalog
    if npe < n_nodes:
        return None, (
            f"{npe} nodes against the {n_nodes} its gmsh type "
            f"{gmsh_code} needs"
        ), True
    if (nat_e.shape[1] < need_dim
            or not np.isfinite(nat_e[:, :need_dim]).all()):
        return None, (
            f"its Gauss rows carry no {need_dim}-D natural coordinates"
        ), False
    N = N_fn(nat_e[:, :need_dim])                 # (n_gp, n_nodes)
    return N @ info.node_coords[:n_nodes], None, True


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

    # ── Per-element info: the group's type + the element's nodes ──
    needed = set(int(e) for e in np.unique(eids))
    eid_info: dict[int, _ElementInfo] = {}
    for group in fem.elements:
        et = group.element_type
        code = int(et.code)
        if code >= 0:
            label, dim = f"gmsh element type {code}", None
        else:
            label, dim = f"{et.gmsh_name} (code {code})", int(et.dim)
        bezier = _bezier_basis_spec(et)
        ids = np.asarray(group.ids, dtype=np.int64)
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for k in range(len(group)):
            eid = int(ids[k])
            if eid not in needed:
                continue
            idxs = nid_to_idx[conn[k]]
            if not (idxs >= 0).all():
                continue
            eid_info[eid] = _ElementInfo(
                code, coords_arr[idxs], label, dim, bezier,
            )

    # ── Per element: resolve its shape once, place all its GPs ─────
    # Degradation counters — aggregated into ONE warning below.
    bbox: dict[tuple[str, str], int] = {}
    bbox_codes: set[int] = set()
    at_origin: dict[tuple[str, str], int] = {}
    unresolved_eids: set[int] = set()
    order = np.argsort(eids, kind="stable")
    splits = np.flatnonzero(np.diff(eids[order])) + 1
    for rows in np.split(order, splits):
        eid = int(eids[rows[0]])
        info = eid_info.get(eid)
        if info is None:
            # No parent element / unknown nodes — the GP stays at the
            # ORIGIN, which is worse than approximate. Counted loud.
            unresolved_eids.add(eid)
            continue
        world, reason, by_bbox = _place_element(info, nat[rows])
        if world is not None:
            out[rows] = world
            continue
        key = (info.label, str(reason))
        if not by_bbox:
            at_origin[key] = at_origin.get(key, 0) + rows.size
            continue
        out[rows] = [_world_via_bbox(nat[r], info.node_coords) for r in rows]
        bbox[key] = bbox.get(key, 0) + rows.size
        bbox_codes.add(info.code)

    if bbox or at_origin or unresolved_eids:
        parts: list[str] = []
        if bbox:
            listing = "; ".join(
                f"{n} GP(s) of {label}: {why}"
                for (label, why), n in sorted(bbox.items())
            )
            parts.append(
                f"{sum(bbox.values())} GP(s) positioned by the "
                f"centroid+bbox approximation (element type code(s) "
                f"{sorted(bbox_codes)}, exact only for axis-aligned "
                f"elements; {listing})"
            )
        if at_origin:
            listing = "; ".join(
                f"{n} GP(s) of {label}: {why}"
                for (label, why), n in sorted(at_origin.items())
            )
            parts.append(
                f"{sum(at_origin.values())} GP(s) could not be placed "
                f"and sit at the ORIGIN ({listing})"
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
