"""Plane-builder helpers feeding :class:`SectionCutDef`.

Returns ``(point, normal)`` tuples in the same shape ``SectionCutDef``
takes — so callers can splat them straight in::

    point, normal = plane_horizontal(z=2500.0)
    cut_def = SectionCutDef(
        plane_point=point, plane_normal=normal, element_ids=ids,
    )

Why not return a STKO ``Plane`` directly? Same reason :mod:`._defs`
stores tuples: keep the cuts subpackage importable and constructible
without ``STKO_to_python`` installed. The shape STKO's ``Plane`` takes
is exactly ``(point: Vec3, normal: Vec3)`` anyway, so the round-trip
through ``SectionCutDef.to_spec()`` is lossless.

Coplanarity is checked by SVD: the smallest singular value of the
mean-centred node-coords gives the out-of-plane spread. Cleanly
coplanar inputs → smallest singular value is at numerical noise level
and the corresponding right-singular vector IS the plane normal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------- #
# Convenience constructors — no FEMData needed
# ---------------------------------------------------------------------- #
def plane_horizontal(z: float) -> tuple[Vec3, Vec3]:
    """Plane perpendicular to global Z at elevation ``z``.

    Returns ``((0.0, 0.0, z), (0.0, 0.0, 1.0))``.
    """
    return ((0.0, 0.0, float(z)), (0.0, 0.0, 1.0))


def plane_vertical(*, axis: Literal["x", "y"], at: float) -> tuple[Vec3, Vec3]:
    """Plane perpendicular to global X or Y at offset ``at``.

    Returns the point on the axis and the unit normal along that axis.
    """
    key = axis.strip().lower()
    if key == "x":
        return ((float(at), 0.0, 0.0), (1.0, 0.0, 0.0))
    if key == "y":
        return ((0.0, float(at), 0.0), (0.0, 1.0, 0.0))
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")


def plane_from_three_points(
    p1: Iterable[float] | np.ndarray,
    p2: Iterable[float] | np.ndarray,
    p3: Iterable[float] | np.ndarray,
    *,
    normal_hint: Iterable[float] | np.ndarray | None = None,
) -> tuple[Vec3, Vec3]:
    """Plane through three non-collinear points.

    Returns ``(p1_as_tuple, unit_normal)``. With ``normal_hint``, the
    normal is flipped if ``normal · hint < 0`` — useful when three
    points define the plane but you want the "outward" direction set
    explicitly.

    Raises
    ------
    ValueError
        Points are collinear / coincident (cross product magnitude
        below 1e-12), or any coordinate is non-finite.
    """
    a = _as_vec3(p1, label="p1")
    b = _as_vec3(p2, label="p2")
    c = _as_vec3(p3, label="p3")
    n = np.cross(b - a, c - a)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        raise ValueError(
            "Three points are collinear or coincident — cannot define a plane."
        )
    n_unit = n / n_norm
    if normal_hint is not None:
        hint = _as_vec3(normal_hint, label="normal_hint")
        if float(np.dot(n_unit, hint)) < 0.0:
            n_unit = -n_unit
    return (
        (float(a[0]), float(a[1]), float(a[2])),
        (float(n_unit[0]), float(n_unit[1]), float(n_unit[2])),
    )


# ---------------------------------------------------------------------- #
# SVD-based plane fit
# ---------------------------------------------------------------------- #
def plane_from_coords(
    coords: np.ndarray,
    *,
    tol: float = 1e-6,
    normal_hint: Iterable[float] | np.ndarray | None = None,
) -> tuple[Vec3, Vec3]:
    """Fit a plane to an ``(N, 3)`` array of nominally coplanar points.

    Algorithm: subtract the centroid, run SVD; the singular vector
    matched to the smallest singular value is the plane normal. The
    centroid is returned as the on-plane point.

    Parameters
    ----------
    coords:
        ``(N, 3)`` float array. Must contain at least 3 finite points.
    tol:
        Maximum absolute signed distance any input point may lie from
        the fit plane. Defaults to ``1e-6`` (typical mesh-tolerance
        scale; physical surfaces in apeGmsh should be planar by
        construction, so genuinely-planar inputs will sit well below).
    normal_hint:
        Optional length-3 outward direction. The fit normal is flipped
        to make ``normal · hint >= 0``.

    Raises
    ------
    ValueError
        Fewer than 3 points, non-finite coords, collinear points (no
        unique plane), or out-of-plane deviation exceeds ``tol``.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"coords must be shape (N, 3), got {arr.shape}."
        )
    if arr.shape[0] < 3:
        raise ValueError(
            f"plane fit needs at least 3 points; got {arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("coords must contain only finite values.")

    centroid = arr.mean(axis=0)
    centered = arr - centroid
    # full_matrices=False makes Vh shape (3, 3) regardless of N.
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)

    # Collinearity check: two large singular values define the in-plane
    # spread; if only one is meaningful, the points lie on a line.
    in_plane_scale = float(S[0])
    if in_plane_scale < 1e-15:
        raise ValueError("All input points coincide; cannot define a plane.")
    if S[1] / in_plane_scale < 1e-9:
        raise ValueError(
            "Input points are collinear (second singular value is "
            "numerically zero); cannot define a unique plane."
        )

    normal = Vh[-1]
    # Coplanarity residual: signed distances along the fit normal.
    residual = float(np.max(np.abs(centered @ normal)))
    if residual > tol:
        raise ValueError(
            f"Points are not coplanar within tol={tol}; "
            f"max out-of-plane deviation is {residual}."
        )

    if normal_hint is not None:
        hint = _as_vec3(normal_hint, label="normal_hint")
        if float(np.dot(normal, hint)) < 0.0:
            normal = -normal

    return (
        (float(centroid[0]), float(centroid[1]), float(centroid[2])),
        (float(normal[0]), float(normal[1]), float(normal[2])),
    )


# ---------------------------------------------------------------------- #
# FEMData entry-point
# ---------------------------------------------------------------------- #
def plane_from_physical_surface(
    fem: "FEMData",
    pg_name: str,
    *,
    tol: float = 1e-6,
    normal_hint: Iterable[float] | np.ndarray | None = None,
) -> tuple[Vec3, Vec3]:
    """Fit a plane to the nodes of a planar physical group.

    The PG is expected to be 2-D (a surface PG); coplanarity is
    enforced. The centroid of the PG's nodes is used as the
    on-plane point.

    Parameters
    ----------
    fem:
        Solver-ready :class:`apeGmsh.mesh.FEMData.FEMData`.
    pg_name:
        Name of the physical group whose nodes define the plane.
    tol:
        Coplanarity tolerance; see :func:`plane_from_coords`.
    normal_hint:
        Optional outward direction; the fit normal is flipped to
        agree.

    Returns
    -------
    (point, normal) : tuple[Vec3, Vec3]
        Plug straight into ``SectionCutDef(plane_point=..., plane_normal=...)``.
    """
    coords = fem.nodes.select(pg=pg_name).coords
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.size == 0:
        raise ValueError(
            f"Physical group {pg_name!r} resolves to zero nodes — "
            "check the PG name and that the mesh contains it."
        )
    return plane_from_coords(coords_arr, tol=tol, normal_hint=normal_hint)


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #
def _as_vec3(v: Iterable[float] | np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(f"{label} must be length-3, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite, got {arr.tolist()}.")
    return arr
