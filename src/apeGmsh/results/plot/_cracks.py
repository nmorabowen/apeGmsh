"""Crack-map geometry for :meth:`ResultsPlot.cracks` (pure numpy).

A smeared-crack material (ASDConcrete3D's ``crackPattern``) gives, at
each Gauss point, up to three vectors along the crack normals, each as
long as that crack's opening. Seen in a projection plane, a crack is a
short segment through the Gauss point running ACROSS its projected
normal — the trace of the crack plane on the view.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

#: Canonical Gauss components of the crack vectors, slot-major:
#: ``(crack_c1_x, crack_c1_y, crack_c1_z, crack_c2_x, ...)``.
CRACK_VECTOR_COMPONENTS: tuple[str, ...] = tuple(
    f"crack_c{k}_{axis}" for k in (1, 2, 3) for axis in ("x", "y", "z")
)


def parse_plane(plane: str) -> tuple[int, int]:
    """``"xz"`` → ``(0, 2)``: the (horizontal, vertical) world axes."""
    p = str(plane).strip().lower()
    if (len(p) != 2 or p[0] not in _AXIS_INDEX or p[1] not in _AXIS_INDEX
            or p[0] == p[1]):
        raise ValueError(
            f"plane={plane!r}: give two distinct axes of 'x', 'y', 'z', "
            f"horizontal first (e.g. 'xz' for a side view)."
        )
    return _AXIS_INDEX[p[0]], _AXIS_INDEX[p[1]]


def default_half_length(points_2d: ndarray) -> float:
    """Half a crack segment: 0.4 x the typical projected Gauss spacing.

    The spacing is ``sqrt(area / n)`` over the DISTINCT projected
    positions (points stacked along the view direction collapse to one),
    which is the grid pitch for a regular mesh. Under half the pitch, so
    neighbouring segments do not touch.
    """
    pts = np.asarray(points_2d, dtype=np.float64)
    if pts.shape[0] == 0:
        return 1.0
    span = pts.max(axis=0) - pts.min(axis=0)
    extent = float(span.max())
    if extent <= 0.0:
        return 1.0
    distinct = np.unique(np.round(pts / (extent * 1e-9)), axis=0).shape[0]
    if distinct <= 1:
        return 0.4 * extent
    short, long_ = sorted(float(s) for s in span)
    if short <= 1e-9 * extent:                       # points on one line
        return 0.4 * long_ / (distinct - 1)
    return 0.4 * float(np.sqrt(short * long_ / distinct))


def crack_segments(
    points: ndarray,
    vectors: ndarray,
    *,
    plane: tuple[int, int],
    min_opening: float,
    half_length: float,
) -> tuple[ndarray, ndarray]:
    """``(segments[S, 2, 2], openings[S])`` for one projected crack map.

    ``points`` is ``(P, 3)`` Gauss-point world coords; ``vectors`` is
    ``(P, K, 3)`` crack vectors (``K`` = 3 slots), row-aligned with
    ``points``. A vector is drawn when its length (the opening, in 3-D)
    is at least ``min_opening`` and positive. A crack whose normal is
    perpendicular to the view plane has no trace on it and is skipped.
    """
    h, v = plane
    pts = np.asarray(points, dtype=np.float64)
    vec = np.asarray(vectors, dtype=np.float64)
    opening = np.linalg.norm(vec, axis=2)                       # (P, K)
    normal_2d = np.stack([vec[..., h], vec[..., v]], axis=-1)   # (P, K, 2)
    proj = np.linalg.norm(normal_2d, axis=2)
    keep = (opening > 0.0) & (opening >= min_opening) & (proj > 1e-9 * opening)
    p_idx, _ = np.nonzero(keep)
    n2 = normal_2d[keep] / proj[keep][:, None]
    across = np.stack([-n2[:, 1], n2[:, 0]], axis=1)
    centre = pts[p_idx][:, [h, v]]
    segments = np.stack(
        [centre - half_length * across, centre + half_length * across],
        axis=1,
    )
    return segments, opening[keep]


def convex_hull_2d(points: ndarray) -> ndarray:
    """Counter-clockwise hull of ``(n, 2)`` points (monotone chain)."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts

    def cross(o: ndarray, a: ndarray, b: ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1])
                     - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])
