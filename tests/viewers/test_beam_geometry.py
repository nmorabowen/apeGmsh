"""Beam local-axis math — pure functions, no Qt or Results.

Verifies the OpenSees ``vecxz`` convention for canonical orientations.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._beam_geometry import (
    COMPONENT_TO_LOCAL_AXIS,
    compute_local_axes,
    default_vecxz,
    fill_axis_for,
    station_position,
)


# =====================================================================
# default_vecxz
# =====================================================================

def test_default_vecxz_horizontal_x():
    v = default_vecxz(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(v, [0.0, 0.0, 1.0])


def test_default_vecxz_horizontal_y():
    v = default_vecxz(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(v, [0.0, 0.0, 1.0])


def test_default_vecxz_vertical():
    v = default_vecxz(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(v, [1.0, 0.0, 0.0])


def test_default_vecxz_near_vertical():
    # Beam slightly off vertical — still treated as vertical
    x_local = np.array([0.001, 0.0, 0.999998])
    x_local /= np.linalg.norm(x_local)
    v = default_vecxz(x_local)
    np.testing.assert_array_equal(v, [1.0, 0.0, 0.0])


# =====================================================================
# compute_local_axes — canonical orientations
# =====================================================================

def test_axes_horizontal_x_default():
    """Horizontal beam along +X with default vecxz: y = -Y, z = +Z."""
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    # y_local = z × x = (0,0,1) × (1,0,0) = (0,1,0)
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0])
    assert L == 10.0


def test_axes_horizontal_y_default():
    """Horizontal beam along +Y: y_local = -X, z_local = +Z."""
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
    )
    np.testing.assert_allclose(x, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    # y_local = z × x = (0,0,1) × (0,1,0) = (-1,0,0)
    np.testing.assert_allclose(y, [-1.0, 0.0, 0.0])
    assert L == 10.0


def test_axes_vertical_z_default():
    """Vertical beam along +Z: default vecxz = +X."""
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 10.0]),
    )
    np.testing.assert_allclose(x, [0.0, 0.0, 1.0])
    # vecxz = +X. Project out x_local component: (1,0,0) - 0 = (1,0,0)
    # so z_local = (1,0,0) (the local "z" in the section frame)
    np.testing.assert_allclose(z, [1.0, 0.0, 0.0])
    # y_local = z × x = (1,0,0) × (0,0,1) = (0,-1,0)
    np.testing.assert_allclose(y, [0.0, -1.0, 0.0])
    assert L == 10.0


def test_axes_skew_with_explicit_vecxz():
    """Skew beam in xy plane with vecxz = global Z."""
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        vecxz=np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(x, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    # y_local = z × x = (0,0,1) × (a,a,0) = (-a, a, 0) where a = 1/sqrt(2)
    np.testing.assert_allclose(
        y, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
    )
    assert L == pytest.approx(np.sqrt(2.0))


def test_axes_orthonormality():
    """All three axes form an orthonormal right-handed triad."""
    np.random.seed(42)
    for _ in range(10):
        ci = np.random.uniform(-5, 5, 3)
        dx = np.random.uniform(-5, 5, 3)
        if np.linalg.norm(dx) < 0.5:
            continue
        cj = ci + dx
        x, y, z, L = compute_local_axes(ci, cj)
        # Unit length
        assert np.linalg.norm(x) == pytest.approx(1.0)
        assert np.linalg.norm(y) == pytest.approx(1.0)
        assert np.linalg.norm(z) == pytest.approx(1.0)
        # Orthogonal
        assert abs(np.dot(x, y)) < 1e-10
        assert abs(np.dot(y, z)) < 1e-10
        assert abs(np.dot(x, z)) < 1e-10
        # Right-handed: y_local == cross(z_local, x_local)
        np.testing.assert_allclose(np.cross(z, x), y, atol=1e-10)


def test_axes_zero_length_raises():
    with pytest.raises(ValueError, match="coincide"):
        compute_local_axes(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_axes_vecxz_parallel_to_x_uses_fallback():
    """vecxz parallel to x_local should still produce a valid frame
    via the default-vecxz fallback path.
    """
    # Horizontal-X beam, but user provides vecxz = +X (parallel).
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        vecxz=np.array([1.0, 0.0, 0.0]),
    )
    # Fallback uses default_vecxz (which gives +Z for horizontal-X),
    # so z_local should end up = +Z.
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0])


# =====================================================================
# station_position
# =====================================================================

def test_station_position_at_endpoints():
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([10.0, 0.0, 0.0])
    np.testing.assert_array_equal(station_position(ci, cj, -1.0), ci)
    np.testing.assert_array_equal(station_position(ci, cj, 1.0), cj)


def test_station_position_at_midpoint():
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([10.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        station_position(ci, cj, 0.0),
        [5.0, 0.0, 0.0],
    )


def test_station_position_lobatto_3pt():
    """3-point Lobatto: xi = -1, 0, +1 (endpoints + midpoint)."""
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([6.0, 0.0, 0.0])
    np.testing.assert_array_equal(station_position(ci, cj, -1.0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(station_position(ci, cj, 0.0), [3.0, 0.0, 0.0])
    np.testing.assert_array_equal(station_position(ci, cj, 1.0), [6.0, 0.0, 0.0])


# =====================================================================
# fill_axis_for
# =====================================================================

def test_fill_axis_for_canonical_components():
    assert fill_axis_for("shear_y") == "y"
    assert fill_axis_for("shear_z") == "z"
    assert fill_axis_for("bending_moment_y") == "z"
    assert fill_axis_for("bending_moment_z") == "y"
    assert fill_axis_for("axial_force") == "z"


def test_fill_axis_for_explicit_override():
    assert fill_axis_for("shear_y", override="z") == "z"
    assert fill_axis_for("bending_moment_y", override="y") == "y"


def test_fill_axis_for_unknown_component_default():
    assert fill_axis_for("some_unknown_component") == "y"


def test_fill_axis_for_invalid_override_raises():
    with pytest.raises(ValueError):
        fill_axis_for("shear_y", override="x")


def test_component_to_local_axis_completeness():
    """Every standard line-diagram component is in the mapping."""
    expected = {
        "axial_force", "shear_y", "shear_z",
        "torsion", "bending_moment_y", "bending_moment_z",
    }
    for c in expected:
        assert c in COMPONENT_TO_LOCAL_AXIS
