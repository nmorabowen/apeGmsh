"""H5-safe geometry helpers for Results (numpy only).

Promoted from ``results.plot._arrows`` so assess (and any other
non-plot consumer) can measure the model bbox without importing
matplotlib. ``plot._arrows`` re-exports :func:`model_diagonal` for
back-compat.
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


# ADR 0094 Amendment 3: a model is "planar" (for camera-default
# purposes) when its z-extent is negligible next to the bbox
# diagonal. ``coords`` is always ``(N, 3)`` — even a 2-D model stores
# a z column, nominally constant.
_PLANAR_Z_TOL = 1e-9


def is_planar(coords: ndarray, *, tol: float = _PLANAR_Z_TOL) -> bool:
    """True when ``coords``' z-extent is negligible vs. the bbox diagonal."""
    if coords.size == 0:
        return False
    z_extent = float(coords[:, 2].max() - coords[:, 2].min())
    return z_extent <= tol * model_diagonal(coords)
