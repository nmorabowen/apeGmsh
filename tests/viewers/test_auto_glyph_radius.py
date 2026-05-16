"""``_auto_glyph_radius`` — auto-sizing for node/point glyph spheres.

Replaces the legacy ``0.003 × model_diagonal`` formula whose result
was dominated by the longest bbox axis. Verifies:

* cubic / near-cubic centers keep approximately the same size as the
  legacy formula (calibration target — visual continuity);
* elongated geometry shrinks dramatically;
* planar (zero-extent on one axis) doesn't collapse to 0;
* line-shaped (zero on two axes) doesn't collapse to 0;
* the ``point_size`` scale factor is honored linearly;
* tiny center sets (< 2 points) fall back to the legacy formula.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh.viewers.scene.glyph_points import _auto_glyph_radius


def _legacy_radius(diag: float, point_size: float = 10.0) -> float:
    return diag * 0.003 * max(0.1, point_size / 10.0)


def _cube_centers(side: float) -> np.ndarray:
    """8 corners of a cube of edge length ``side``."""
    s = side
    return np.array(
        [
            [0, 0, 0], [s, 0, 0], [0, s, 0], [s, s, 0],
            [0, 0, s], [s, 0, s], [0, s, s], [s, s, s],
        ],
        dtype=np.float64,
    )


# =====================================================================
# Calibration — cubic models stay close to the legacy size
# =====================================================================


def test_cubic_centers_within_15_percent_of_legacy():
    """For a cube, the new formula should fall within 15% of the
    legacy ``0.003 × diagonal`` value — visual continuity for the
    common case."""
    side = 10.0
    centers = _cube_centers(side)
    diag = math.sqrt(3.0) * side    # 17.32...
    legacy = _legacy_radius(diag)

    new = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    rel = abs(new - legacy) / legacy
    assert rel < 0.15, f"new={new:.4f} vs legacy={legacy:.4f} (rel={rel:.2%})"


# =====================================================================
# The reported pain — elongated geometry shrinks dramatically
# =====================================================================


def test_elongated_beam_shrinks_vs_diagonal():
    """A 100 × 0.5 × 0.5 beam: legacy gives radius ≈ 0.3 (bigger than
    cross-section); the new formula should drop it by ≥ 10×."""
    centers = np.array(
        [
            [0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
            [0.0, 0.5, 0.0], [100.0, 0.5, 0.0],
            [0.0, 0.0, 0.5], [100.0, 0.0, 0.5],
            [0.0, 0.5, 0.5], [100.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    diag = math.sqrt(100.0**2 + 0.5**2 + 0.5**2)    # ≈ 100.0025
    legacy = _legacy_radius(diag)    # ≈ 0.3

    new = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    assert new < legacy / 10.0, (
        f"new radius {new:.4f} not <= legacy/10 = {legacy/10:.4f}"
    )


def test_thin_plate_shrinks_vs_diagonal():
    """10 × 10 × 0.1 plate: another reported pain (thin flat slabs)."""
    centers = np.array(
        [
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0], [10.0, 10.0, 0.0],
            [0.0, 0.0, 0.1], [10.0, 0.0, 0.1],
            [0.0, 10.0, 0.1], [10.0, 10.0, 0.1],
        ],
        dtype=np.float64,
    )
    diag = math.sqrt(10.0**2 + 10.0**2 + 0.1**2)    # ≈ 14.14
    legacy = _legacy_radius(diag)

    new = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    assert new < legacy / 2.0


# =====================================================================
# Degenerate-extent cases
# =====================================================================


def test_planar_centers_do_not_collapse_to_zero():
    """All centers at z=0 — extents = [dx, dy, 0]. Geometric mean
    would be 0 if we didn't substitute the smallest non-zero."""
    centers = np.array(
        [
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0], [10.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    diag = math.sqrt(10.0**2 + 10.0**2)
    r = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    assert r > 0.0


def test_collinear_centers_do_not_collapse_to_zero():
    """All centers on the x-axis — extents = [dx, 0, 0]."""
    centers = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    r = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=10.0)
    assert r > 0.0


def test_single_center_falls_back_to_legacy_diagonal():
    """One center → no extents to measure; use the supplied diag."""
    centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    diag = 5.0
    r = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    assert r == pytest.approx(_legacy_radius(diag))


def test_zero_centers_falls_back_to_legacy_diagonal():
    """Empty input still has to return a finite radius — fall back."""
    centers = np.zeros((0, 3), dtype=np.float64)
    diag = 7.5
    r = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    assert r == pytest.approx(_legacy_radius(diag))


def test_coincident_centers_do_not_collapse_to_zero():
    """All centers at the same point → extents all zero. Should fall
    back to the legacy diagonal, not return 0."""
    centers = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    diag = 3.0
    r = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    assert r == pytest.approx(_legacy_radius(diag))


# =====================================================================
# point_size scaling — linear with floor at 0.1× base
# =====================================================================


def test_point_size_scales_linearly():
    """``point_size=20`` doubles the radius; ``point_size=5`` halves it."""
    centers = _cube_centers(10.0)
    diag = math.sqrt(3.0) * 10.0
    r10 = _auto_glyph_radius(centers, point_size=10.0, fallback_diag=diag)
    r20 = _auto_glyph_radius(centers, point_size=20.0, fallback_diag=diag)
    r5 = _auto_glyph_radius(centers, point_size=5.0, fallback_diag=diag)
    assert r20 == pytest.approx(2.0 * r10)
    assert r5 == pytest.approx(0.5 * r10)


def test_point_size_floor_at_one_tenth_of_base():
    """``point_size <= 1`` clamps at ``0.1 × base`` so the glyph
    doesn't vanish."""
    centers = _cube_centers(10.0)
    diag = math.sqrt(3.0) * 10.0
    r_zero = _auto_glyph_radius(centers, point_size=0.0, fallback_diag=diag)
    r_floor = _auto_glyph_radius(centers, point_size=1.0, fallback_diag=diag)
    assert r_zero == pytest.approx(r_floor)
