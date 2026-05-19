"""Phase-2 unit tests for the plane-builder helpers."""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.cuts import (
    plane_from_coords,
    plane_from_physical_surface,
    plane_from_three_points,
    plane_horizontal,
    plane_vertical,
)


# --------------------------------------------------------------------- #
# Convenience constructors
# --------------------------------------------------------------------- #
def test_plane_horizontal():
    point, normal = plane_horizontal(z=2500.0)
    assert point == (0.0, 0.0, 2500.0)
    assert normal == (0.0, 0.0, 1.0)


def test_plane_vertical_x():
    point, normal = plane_vertical(axis="x", at=100.0)
    assert point == (100.0, 0.0, 0.0)
    assert normal == (1.0, 0.0, 0.0)


def test_plane_vertical_y():
    point, normal = plane_vertical(axis="y", at=-50.0)
    assert point == (0.0, -50.0, 0.0)
    assert normal == (0.0, 1.0, 0.0)


def test_plane_vertical_case_insensitive():
    _, normal = plane_vertical(axis="Y", at=0.0)
    assert normal == (0.0, 1.0, 0.0)


def test_plane_vertical_invalid_axis_raises():
    with pytest.raises(ValueError, match="axis must be"):
        plane_vertical(axis="z", at=0.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# plane_from_three_points
# --------------------------------------------------------------------- #
def test_from_three_points_horizontal():
    point, normal = plane_from_three_points(
        (0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (0.0, 1.0, 10.0),
    )
    assert point == (0.0, 0.0, 10.0)
    np.testing.assert_allclose(normal, (0.0, 0.0, 1.0), atol=1e-12)


def test_from_three_points_oblique():
    # A plane tilted 45° about the x-axis, passing through (0,0,0).
    point, normal = plane_from_three_points(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0),
    )
    assert point == (0.0, 0.0, 0.0)
    expected = np.array([0.0, -1.0, 1.0]) / np.sqrt(2)
    np.testing.assert_allclose(normal, expected, atol=1e-12)


def test_from_three_points_normal_hint_flips():
    # Right-hand rule on these three points yields +z.
    point, normal = plane_from_three_points(
        (0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (0.0, 1.0, 10.0),
        normal_hint=(0.0, 0.0, -1.0),
    )
    assert point == (0.0, 0.0, 10.0)
    np.testing.assert_allclose(normal, (0.0, 0.0, -1.0), atol=1e-12)


def test_from_three_points_collinear_raises():
    with pytest.raises(ValueError, match="collinear"):
        plane_from_three_points(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        )


def test_from_three_points_coincident_raises():
    with pytest.raises(ValueError, match="collinear"):
        plane_from_three_points(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        )


# --------------------------------------------------------------------- #
# plane_from_coords (SVD fit)
# --------------------------------------------------------------------- #
def test_from_coords_horizontal_grid():
    # 9-point grid on z=5, with mild jitter inside tol — should fit cleanly.
    xs, ys = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(9, 5.0)], axis=1)
    point, normal = plane_from_coords(pts)
    np.testing.assert_allclose(point, (1.0, 1.0, 5.0), atol=1e-12)
    # Normal is along z (sign is arbitrary without hint; check magnitude).
    np.testing.assert_allclose(abs(normal[2]), 1.0, atol=1e-9)
    np.testing.assert_allclose(normal[:2], (0.0, 0.0), atol=1e-9)


def test_from_coords_oblique_plane_fit():
    # Plane through origin with normal (1, 1, 1)/sqrt(3).
    n_true = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    # Two in-plane vectors orthogonal to n_true.
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v = np.cross(n_true, u)
    # Sample a 4×4 grid in the plane.
    ts = np.linspace(-1.0, 1.0, 4)
    pts = np.array([s * u + t * v for s in ts for t in ts])
    point, normal = plane_from_coords(pts, normal_hint=n_true)
    np.testing.assert_allclose(point, (0.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(normal, n_true, atol=1e-9)


def test_from_coords_non_coplanar_raises():
    # 4 points: 3 on z=0, 1 at z=1 → out-of-plane deviation > tol.
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],     # off-plane
    ])
    with pytest.raises(ValueError, match="not coplanar"):
        plane_from_coords(pts, tol=1e-9)


def test_from_coords_tolerance_admits_small_noise():
    # Same shape as the previous test, but with the off-plane point sub-tol.
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1e-7],    # off-plane but within tol=1e-6
    ])
    point, normal = plane_from_coords(pts, tol=1e-6)
    # The fit normal lies very close to (0, 0, 1) (or -1).
    np.testing.assert_allclose(abs(normal[2]), 1.0, atol=1e-3)


def test_from_coords_too_few_points_raises():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="at least 3 points"):
        plane_from_coords(pts)


def test_from_coords_collinear_raises():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    with pytest.raises(ValueError, match="collinear"):
        plane_from_coords(pts)


def test_from_coords_non_finite_raises():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, float("inf"), 0.0],
    ])
    with pytest.raises(ValueError, match="finite"):
        plane_from_coords(pts)


def test_from_coords_wrong_shape_raises():
    pts = np.zeros((5, 2))
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        plane_from_coords(pts)


# --------------------------------------------------------------------- #
# plane_from_physical_surface — thin wrapper over plane_from_coords
# --------------------------------------------------------------------- #
class _StubNodes:
    def __init__(self, coords: np.ndarray) -> None:
        self._coords = coords

    def get_coords(self, *, pg: str) -> np.ndarray:
        return self._coords

    def select(self, target=None, *, pg: str | None = None, **_kw):
        """selection-unification v2 P3-R: ``fem.nodes.get_coords(pg=)``
        is removed; ``fem.nodes.select(pg=).coords`` is the migration
        target (P-COORD).  Mirrors the broker — same coords as the
        (removed) ``get_coords`` body via the ``.coords`` terminal."""
        return _SelResult(self._coords)


class _SelResult:
    """The ``fem.nodes.select(...)`` terminal — exposes ``.coords``
    (the only surface PROD reads after the ``get_coords``→``select``
    P3-R migration)."""

    def __init__(self, coords: np.ndarray) -> None:
        self.coords = coords


class _StubFEM:
    def __init__(self, coords: np.ndarray) -> None:
        self.nodes = _StubNodes(coords)


def test_plane_from_physical_surface_happy_path():
    pts = np.array([
        [0.0, 0.0, 100.0],
        [1.0, 0.0, 100.0],
        [0.0, 1.0, 100.0],
        [1.0, 1.0, 100.0],
    ])
    fem = _StubFEM(pts)
    point, normal = plane_from_physical_surface(fem, "diaphragm-3")  # type: ignore[arg-type]
    np.testing.assert_allclose(point, (0.5, 0.5, 100.0), atol=1e-12)
    np.testing.assert_allclose(abs(normal[2]), 1.0, atol=1e-9)


def test_plane_from_physical_surface_with_hint():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    fem = _StubFEM(pts)
    _, normal = plane_from_physical_surface(
        fem, "wall-x0", normal_hint=(0.0, 0.0, -1.0),    # type: ignore[arg-type]
    )
    np.testing.assert_allclose(normal, (0.0, 0.0, -1.0), atol=1e-12)


def test_plane_from_physical_surface_empty_raises():
    fem = _StubFEM(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="zero nodes"):
        plane_from_physical_surface(fem, "missing-pg")  # type: ignore[arg-type]


def test_plane_from_physical_surface_non_coplanar_raises():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],     # genuinely off-plane
    ])
    fem = _StubFEM(pts)
    with pytest.raises(ValueError, match="not coplanar"):
        plane_from_physical_surface(fem, "curved-surface", tol=1e-9)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Output feeds SectionCutDef cleanly
# --------------------------------------------------------------------- #
def test_output_feeds_section_cut_def():
    from apeGmsh.cuts import SectionCutDef

    point, normal = plane_horizontal(z=3000.0)
    d = SectionCutDef(
        plane_point=point,
        plane_normal=normal,
        element_ids=(1, 2, 3),
    )
    assert d.plane_point == (0.0, 0.0, 3000.0)
    assert d.plane_normal == (0.0, 0.0, 1.0)
