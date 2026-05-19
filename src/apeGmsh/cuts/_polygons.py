"""Polygon helpers on planes — bounding-polygon derivation from a PG.

The first user of these helpers is
:meth:`SectionCutDef.from_planar_pg(..., with_bounding=True)`, which
auto-derives a convex ``bounding_polygon`` from the cut PG's nodes so
``SectionCutSpec`` filters elements crossing the plane *and* falling
inside that region.

Algorithm
---------
1. Fit a plane through the PG nodes (reuses
   :func:`apeGmsh.cuts._planes.plane_from_coords`).
2. Build an orthonormal in-plane basis ``(e1, e2)`` perpendicular to the
   fit normal. Mirrors STKO_to_python's convention: ``e1`` is the cross
   product of the normal and the most-perpendicular global axis.
3. Project all PG nodes to 2-D plane coords.
4. Take the convex hull via :class:`scipy.spatial.ConvexHull`. Lazy-
   imported so the rest of the cuts package stays scipy-free; only
   :func:`bounding_polygon_from_physical_surface` needs it.
5. Re-embed the hull vertices in 3-D from the plane basis. Vertices lie
   exactly on the plane (no projection residual).

Returned polygon is CCW around the plane normal and contains the
minimum vertices to span the PG's footprint — interior mesh nodes are
discarded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from ._defs import Vec3

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


# ---------------------------------------------------------------------- #
# Plane basis + projection
# ---------------------------------------------------------------------- #
def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane basis ``(e1, e2)``.

    Picks the global axis most perpendicular to ``normal`` for a
    numerically stable cross product. ``(e1, e2, normal)`` is
    right-handed.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _project_to_basis(
    coords_3d: np.ndarray,
    point: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    """Project coplanar 3-D points to 2-D plane coords."""
    rel = coords_3d - point
    u = rel @ e1
    v = rel @ e2
    return np.stack([u, v], axis=1)


def _embed_from_basis(
    coords_2d: np.ndarray,
    point: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    """Re-embed 2-D plane-basis points back to 3-D on the plane."""
    return point + coords_2d[:, 0:1] * e1 + coords_2d[:, 1:2] * e2


# ---------------------------------------------------------------------- #
# Convex hull (scipy.spatial.ConvexHull, lazy-imported)
# ---------------------------------------------------------------------- #
def _convex_hull_2d_ccw(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, vertices returned in CCW order.

    Wraps :class:`scipy.spatial.ConvexHull`. Lazy import keeps the
    rest of the cuts package usable without scipy; only the bounding-
    polygon path needs it.

    Raises
    ------
    ImportError
        scipy is not installed. The message points users to
        ``pip install apeGmsh[plot]`` (the existing extra that bundles
        scipy + matplotlib) or plain ``pip install scipy``.
    ValueError
        Fewer than 3 distinct points (after dedup), or the points are
        collinear / degenerate (qhull raises, we re-wrap as
        ``ValueError``).
    """
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError as exc:
        raise ImportError(
            "Convex-hull-based bounding-polygon derivation requires "
            "scipy. Install with `pip install apeGmsh[plot]` (the extra "
            "that bundles scipy + matplotlib) or plain `pip install scipy`."
        ) from exc

    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        raise ValueError(
            f"Convex hull needs at least 3 distinct points; got {pts.shape[0]}."
        )
    # Pre-check collinearity so the user sees a clear message instead
    # of qhull's "input appears to be less than 2 dimensional" precision
    # error. Rank of the mean-centred points is 2 iff the points span
    # the plane.
    centered = pts - pts.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-9) < 2:
        raise ValueError(
            "2-D convex hull is undefined: input points are collinear."
        )
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        # Other qhull failures (extreme degeneracy, numerical edge
        # cases) — surface the message but keep ValueError as the type.
        raise ValueError(
            f"Could not compute 2-D convex hull (qhull): {exc}"
        ) from exc

    # ConvexHull.vertices is in CCW order for 2-D inputs.
    return pts[hull.vertices]


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #
def bounding_polygon_from_physical_surface(
    fem: "FEMData",
    pg_name: str,
    *,
    tol: float = 1e-6,
    normal_hint: Iterable[float] | np.ndarray | None = None,
) -> tuple[Vec3, ...]:
    """Convex hull of a planar PG's nodes, embedded back on the fit plane.

    Suitable as a ``bounding_polygon`` for
    :class:`SectionCutDef`/``SectionCutSpec``: convex, on-plane within
    the same tolerance the plane fit accepts, non-degenerate.

    Parameters
    ----------
    fem:
        Solver-ready :class:`apeGmsh.mesh.FEMData.FEMData`.
    pg_name:
        Physical group whose nodes define the polygon. Must be planar
        within ``tol`` — same check as :func:`plane_from_physical_surface`.
    tol:
        Coplanarity tolerance.
    normal_hint:
        Optional outward direction; the fit normal is flipped to
        agree. Note: hull CCW-ness is reported relative to the fit
        normal, so flipping the normal flips the polygon winding.

    Returns
    -------
    tuple[Vec3, ...]
        Polygon vertices in 3-D, CCW around the plane normal. Each
        vertex lies on the fit plane.

    Raises
    ------
    ValueError
        Empty PG, non-coplanar nodes, or all-collinear nodes (no
        non-degenerate hull).
    """
    from ._planes import plane_from_coords

    coords = fem.nodes.select(pg=pg_name).coords
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.size == 0:
        raise ValueError(
            f"Physical group {pg_name!r} resolves to zero nodes — "
            "cannot build a bounding polygon."
        )

    point, normal = plane_from_coords(
        coords_arr, tol=tol, normal_hint=normal_hint,
    )
    point_arr = np.asarray(point, dtype=float)
    normal_arr = np.asarray(normal, dtype=float)
    e1, e2 = _plane_basis(normal_arr)
    coords_2d = _project_to_basis(coords_arr, point_arr, e1, e2)
    hull_2d = _convex_hull_2d_ccw(coords_2d)
    hull_3d = _embed_from_basis(hull_2d, point_arr, e1, e2)
    return tuple(
        (float(v[0]), float(v[1]), float(v[2])) for v in hull_3d
    )
