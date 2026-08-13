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
