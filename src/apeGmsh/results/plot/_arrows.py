"""Vector-glyph helpers — auto-scaling and magnitude filtering.

Used by ``vector_glyph``, ``reactions``, and ``loads``. Shared so a
multi-arrow figure (e.g. reactions + applied loads) can use a
consistent visual size.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


def model_diagonal(coords: ndarray) -> float:
    """Return the bbox diagonal of ``coords`` (or 1.0 if empty)."""
    if coords.size == 0:
        return 1.0
    span = coords.max(axis=0) - coords.min(axis=0)
    diag = float(np.linalg.norm(span))
    return diag if diag > 0.0 else 1.0


def auto_arrow_scale(
    vectors: ndarray,
    model_size: float,
    *,
    target_frac: float = 0.07,
) -> float:
    """Scale that maps the largest vector to ``target_frac × model_size``.

    Returns ``1.0`` if all vectors are zero (caller will end up with
    invisible arrows, which is fine).
    """
    if vectors.size == 0:
        return 1.0
    mags = np.linalg.norm(vectors, axis=1)
    m = float(mags.max())
    if m <= 0.0:
        return 1.0
    return target_frac * float(model_size) / m


def filter_significant(
    points: ndarray,
    vectors: ndarray,
    *,
    zero_tol: float = 1e-6,
) -> tuple[ndarray, ndarray]:
    """Drop rows whose vector magnitude is below ``zero_tol × max``.

    The viewer's reactions diagram drops free-interior nodes the same
    way (recorders write effective zero on unconstrained DOFs).
    """
    if vectors.size == 0:
        return points, vectors
    mags = np.linalg.norm(vectors, axis=1)
    m = float(mags.max())
    if m <= 0.0:
        return (
            np.empty((0, 3), dtype=points.dtype),
            np.empty((0, 3), dtype=vectors.dtype),
        )
    mask = mags >= zero_tol * m
    return points[mask], vectors[mask]
