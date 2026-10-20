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
not be placed at all. An unplaced GP is NaN in every column, whatever the
cause (an element missing from the FEMData, connectivity naming unknown
nodes, a FEMData with no nodes, a row without natural coordinates): never
the origin, which is a plausible coordinate that a pick or a box-select
would take for a real point.

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
    ``slab.element_index`` / ``slab.natural_coords``. A row that could
    not be placed is NaN in all three columns.
"""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy import ndarray

from ..fem._shape_functions import get_shape_functions
from ..opensees._element_capabilities import GMSH_HEX20_TO_SERENDIPITY
from ._gauss_extrapolation import _GMSH_CODE_BY_PARENT_DIM_NPE, _PARENT_DIM

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ._slabs import GaussSlab


class WarnGaussCoordsApproximate(UserWarning):
    """GP world-coordinate reconstruction was degraded.

    Raised (as a warning, once per reconstruction call) when Gauss
    points were positioned by the centroid+bbox approximation
    (element type lacks shape-function coverage) or could not be
    placed at all and are NaN (parent element missing from the
    FEMData / connectivity references unknown nodes / no natural
    coordinates). A silently approximated marker cloud
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


def _inverse(perm: "tuple[int, ...]") -> "tuple[int, ...]":
    inv = [0] * len(perm)
    for k, j in enumerate(perm):
        inv[j] = k
    return tuple(inv)


#: File-coded classes (keyed by ``-class_tag``) of a quadratic shape whose
#: node order is established, and how to read their row in gmsh order (``gmsh_row = file_row[perm]``;
#: ``None`` is gmsh's own order). A quadratic shape's map runs through its
#: mid-edge nodes, so a class missing here is not guessed: it takes the bbox
#: approximation and is named. Linear shapes need only the corner order every
#: class shares, so they are not listed.
#:
#: * SixNodeTri, LadrunoLST, TenNodeTetrahedron: gmsh order. The fixture solve
#:   in ``tests/fixtures/simplex_gp`` writes them so and each reproduces
#:   ``u = x²/2`` exactly, which a misordered element could not.
#: * EightNodeQuad, NineNodeQuad: gmsh order (their ``shapeFunction``: mid-
#:   edges (1,2) (2,3) (3,4) (4,1), then the centre).
#: * LadrunoBrick20, Twenty_Node_Brick: the serendipity order of ``brcshl``
#:   (``shp3dv``), which ``LadrunoBrick20._emit`` writes by
#:   ``GMSH_HEX20_TO_SERENDIPITY``; read back through its inverse.
_GMSH_ORDER_OF_FILE_CLASS: "dict[int, tuple[int, ...] | None]" = {
    -209: None,      # SixNodeTri
    -33016: None,    # LadrunoLST
    -256: None,      # TenNodeTetrahedron
    -208: None,      # EightNodeQuad
    -207: None,      # NineNodeQuad
    -33018: _inverse(GMSH_HEX20_TO_SERENDIPITY),   # LadrunoBrick20
    -49: _inverse(GMSH_HEX20_TO_SERENDIPITY),      # Twenty_Node_Brick
}

#: Gmsh codes whose map uses only the corners (any class's corner order).
_LINEAR_GMSH_CODES = frozenset({1, 2, 3, 4, 5})


class _Placement(NamedTuple):
    """How one element's GPs were placed.

    ``world`` is ``(n_gp, 3)``; a row is NaN when that GP carries no
    natural coordinates (``n_unplaced``). ``bbox_reason`` is set when the
    placeable rows took the centroid+bbox approximation.
    """
    world: ndarray
    bbox_reason: "str | None"
    n_unplaced: int


def _prefix_widths(nat_e: ndarray) -> ndarray:
    """Per row: how many leading columns are finite, or ``-1`` when a finite
    value follows a NaN (the row is not a padded narrower row)."""
    fin = np.isfinite(nat_e)
    n_col = nat_e.shape[1]
    width = np.where(fin.all(axis=1), n_col, np.argmin(fin, axis=1))
    after = np.arange(n_col)[None, :] >= width[:, None]
    width[(fin & after).any(axis=1)] = -1
    return width


def _place_element(info: _ElementInfo, nat_e: ndarray) -> _Placement:
    """Place one element's GPs by its own basis where it is known.

    A file-coded element (``code = -class_tag``) is resolved to a gmsh shape
    as the Gauss-to-node extrapolation resolves it, from its node count and
    the width of its own Gauss rows. A quadratic shape is evaluated through
    its full shape functions only for a class in
    :data:`_GMSH_ORDER_OF_FILE_CLASS`.

    Rows without natural coordinates in the needed columns are NaN; the
    element's other rows are still placed.
    """
    n_gp = nat_e.shape[0]
    world = np.full((n_gp, 3), np.nan, dtype=np.float64)
    npe = int(info.node_coords.shape[0])
    nodes = info.node_coords
    bbox_reason: "str | None" = None

    if info.bezier is not None:
        topo, fam = info.bezier
        need = 2 if topo == "tri" else 3
        ok = (np.isfinite(nat_e[:, :need]).all(axis=1)
              if nat_e.shape[1] >= need else np.zeros(n_gp, dtype=bool))
        p = _simplex_order(topo, npe)
        placed = None
        if ok.any() and p is not None:
            placed = _world_via_basis(nat_e[ok, :need], nodes,
                                      topology=topo, family=fam, order=p)
        if placed is not None:
            world[ok] = placed
        elif ok.any():
            bbox_reason = f"no {fam} {topo} basis on its {npe} control points"
        return _finish(world, nat_e[:, :need], nodes, ok, bbox_reason)

    if info.code >= 0:
        gmsh_code: "int | None" = info.code
        need_dim = _PARENT_DIM.get(info.code)
    else:
        width = _prefix_widths(nat_e)
        pdim = int(width.max()) if width.size else 0
        if pdim <= 0:
            return _Placement(world, None, n_gp)
        need_dim = pdim
        gmsh_code = None
        if info.dim is not None and pdim > info.dim:
            bbox_reason = (
                f"its {pdim}-D Gauss rows contradict its group's "
                f"dim={info.dim}"
            )
        else:
            gmsh_code = _GMSH_CODE_BY_PARENT_DIM_NPE.get((pdim, npe))
            if gmsh_code is None:
                bbox_reason = (
                    f"no shape functions for a {npe}-node element with "
                    f"{pdim}-D Gauss points"
                )
            elif gmsh_code not in _LINEAR_GMSH_CODES:
                if info.code not in _GMSH_ORDER_OF_FILE_CLASS:
                    bbox_reason = (
                        f"the node order of a {npe}-node {pdim}-D class is "
                        f"not established here"
                    )
                    gmsh_code = None
                else:
                    perm = _GMSH_ORDER_OF_FILE_CLASS[info.code]
                    if perm is not None:
                        nodes = nodes[list(perm)]

    if need_dim is None:
        ok = np.isfinite(nat_e).all(axis=1)
    elif nat_e.shape[1] < need_dim:
        ok = np.zeros(n_gp, dtype=bool)
    else:
        ok = np.isfinite(nat_e[:, :need_dim]).all(axis=1)
    if bbox_reason is not None or not ok.any():
        return _finish(world, _cols(nat_e, need_dim), nodes, ok, bbox_reason)

    assert gmsh_code is not None
    catalog = get_shape_functions(gmsh_code)
    if catalog is None or need_dim is None:
        return _finish(world, _cols(nat_e, need_dim), nodes, ok,
                       f"no shape functions for gmsh element type {gmsh_code}")
    N_fn, _, _, n_nodes = catalog
    if npe < n_nodes:
        return _finish(world, _cols(nat_e, need_dim), nodes, ok, (
            f"{npe} nodes against the {n_nodes} its gmsh type "
            f"{gmsh_code} needs"
        ))
    N = N_fn(nat_e[ok, :need_dim])                 # (n_ok, n_nodes)
    world[ok] = N @ nodes[:n_nodes]
    return _Placement(world, None, int(n_gp - ok.sum()))


def _cols(nat_e: ndarray, need: "int | None") -> ndarray:
    return nat_e if need is None else nat_e[:, :need]


def _finish(
    world: ndarray, nat_e: ndarray, nodes: ndarray, ok: ndarray,
    bbox_reason: "str | None",
) -> _Placement:
    """Fill the placeable rows by the bbox map when a reason says so."""
    if bbox_reason is not None:
        for r in np.flatnonzero(ok):
            world[r] = _world_via_bbox(nat_e[r], nodes)
    return _Placement(world, bbox_reason, int(ok.size - ok.sum()))


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
    # Every row starts unplaced (NaN) and is overwritten only when its
    # element resolves: an unplaced GP is never the origin.
    out = np.full((n_gp, 3), np.nan, dtype=np.float64)
    if n_gp == 0:
        return out

    # ── Lookup: FEM node id -> coord row ───────────────────────────
    node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
    coords_arr = (
        np.asarray(node_coords_override, dtype=np.float64)
        if node_coords_override is not None
        else np.asarray(fem.nodes.coords, dtype=np.float64)
    )
    if coords_arr.shape[0] != node_ids_arr.shape[0]:
        # Override length must match fem.nodes.ids — silently fall
        # back to the embedded fem coords.
        coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
    # A FEMData with no nodes resolves no element: every GP is then
    # counted unresolved below and named in the warning.
    max_nid = int(node_ids_arr.max()) if node_ids_arr.size else -1
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
            row = conn[k]
            if row.size and (row.min() < 0 or row.max() > max_nid):
                continue
            idxs = nid_to_idx[row]
            if not (idxs >= 0).all():
                continue
            eid_info[eid] = _ElementInfo(
                code, coords_arr[idxs], label, dim, bezier,
            )

    # ── Per element: resolve its shape once, place all its GPs ─────
    # Degradation counters — aggregated into ONE warning below.
    bbox: dict[tuple[str, str], int] = {}
    bbox_codes: set[int] = set()
    unplaced: dict[str, int] = {}
    unresolved_eids: set[int] = set()
    order = np.argsort(eids, kind="stable")
    splits = np.flatnonzero(np.diff(eids[order])) + 1
    for rows in np.split(order, splits):
        eid = int(eids[rows[0]])
        info = eid_info.get(eid)
        if info is None:
            # No parent element / unknown nodes — the rows stay NaN
            # (not placed). Counted loud.
            unresolved_eids.add(eid)
            continue
        placed = _place_element(info, nat[rows])
        out[rows] = placed.world
        if placed.n_unplaced:
            unplaced[info.label] = (
                unplaced.get(info.label, 0) + placed.n_unplaced
            )
        if placed.bbox_reason is not None:
            key = (info.label, placed.bbox_reason)
            bbox[key] = bbox.get(key, 0) + rows.size - placed.n_unplaced
            bbox_codes.add(info.code)

    if bbox or unplaced or unresolved_eids:
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
        if unplaced:
            listing = "; ".join(
                f"{n} GP(s) of {label}" for label, n in sorted(unplaced.items())
            )
            parts.append(
                f"{sum(unplaced.values())} GP(s) carry no natural "
                f"coordinates and are NaN ({listing})"
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
                f"unknown nodes) — their GPs are NaN (not placed); "
                f"eids: {sample}{more}"
            )
        warnings.warn(
            "GP world-coordinate reconstruction degraded: "
            + "; ".join(parts),
            WarnGaussCoordsApproximate,
            stacklevel=2,
        )
    return out
