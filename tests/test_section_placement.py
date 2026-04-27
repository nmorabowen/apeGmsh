"""
Tests for ``apeGmsh.core._section_placement``.

Pure-math helpers; the only gmsh interaction is the centroid mode of
``compute_anchor_offset``, which is exercised here against an empty
gmsh model to confirm the no-live-entities fallback.
"""
from __future__ import annotations

import math

import gmsh
import pytest

from apeGmsh.core._section_placement import (
    compute_alignment_rotation,
    compute_anchor_offset,
)


# ---------------------------------------------------------------------
# compute_anchor_offset
# ---------------------------------------------------------------------

class TestAnchorOffset:
    def test_start_is_identity(self):
        assert compute_anchor_offset("start", length=1000) == (0.0, 0.0, 0.0)

    def test_start_no_length(self):
        # "start" doesn't actually need a length
        assert compute_anchor_offset("start") == (0.0, 0.0, 0.0)

    def test_end_shifts_far_face_to_origin(self):
        assert compute_anchor_offset("end", length=1000) == (0.0, 0.0, -1000.0)

    def test_midspan_shifts_midpoint(self):
        assert compute_anchor_offset("midspan", length=1000) == (
            0.0, 0.0, -500.0,
        )

    def test_centroid_with_no_entities(self):
        # Empty gmsh model → x/y centroid is 0, z is -length/2
        gmsh.initialize()
        try:
            gmsh.clear()
            assert compute_anchor_offset("centroid", length=1000) == (
                0.0, 0.0, -500.0,
            )
        finally:
            gmsh.finalize()

    def test_explicit_tuple(self):
        assert compute_anchor_offset((10, 20, 30)) == (-10.0, -20.0, -30.0)

    def test_explicit_tuple_no_length_required(self):
        # Tuple form ignores length
        assert compute_anchor_offset((1, 2, 3)) == (-1.0, -2.0, -3.0)

    def test_end_without_length_raises(self):
        with pytest.raises(ValueError, match="requires a length"):
            compute_anchor_offset("end")

    def test_midspan_without_length_raises(self):
        with pytest.raises(ValueError, match="requires a length"):
            compute_anchor_offset("midspan")

    def test_centroid_without_length_raises(self):
        with pytest.raises(ValueError, match="requires a length"):
            compute_anchor_offset("centroid")

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown anchor"):
            compute_anchor_offset("middle", length=1000)

    def test_wrong_length_tuple_raises(self):
        with pytest.raises(ValueError, match="3 components"):
            compute_anchor_offset((1, 2))
        with pytest.raises(ValueError, match="3 components"):
            compute_anchor_offset((1, 2, 3, 4))

    def test_non_iterable_raises(self):
        with pytest.raises(ValueError):
            compute_anchor_offset(42)


# ---------------------------------------------------------------------
# compute_alignment_rotation
# ---------------------------------------------------------------------

class TestAlignmentRotation:
    def test_z_is_identity(self):
        assert compute_alignment_rotation("z") is None

    def test_x_is_120_about_111(self):
        rot = compute_alignment_rotation("x")
        assert rot is not None
        angle, ax, ay, az = rot
        assert angle == pytest.approx(2.0 * math.pi / 3.0)
        assert (ax, ay, az) == (1.0, 1.0, 1.0)

    def test_y_is_180_about_011(self):
        rot = compute_alignment_rotation("y")
        assert rot is not None
        angle, ax, ay, az = rot
        assert angle == pytest.approx(math.pi)
        assert (ax, ay, az) == (0.0, 1.0, 1.0)

    def test_x_named_maps_local_z_to_world_x(self):
        # Rodrigues: 120° about (1,1,1)/√3 maps z=(0,0,1) to (1,0,0)
        rot = compute_alignment_rotation("x")
        out = _rodrigues(rot, (0.0, 0.0, 1.0))
        assert out == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)

    def test_x_named_maps_local_y_to_world_z(self):
        rot = compute_alignment_rotation("x")
        out = _rodrigues(rot, (0.0, 1.0, 0.0))
        assert out == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)

    def test_y_named_maps_local_z_to_world_y(self):
        rot = compute_alignment_rotation("y")
        out = _rodrigues(rot, (0.0, 0.0, 1.0))
        assert out == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)

    def test_tuple_plus_x(self):
        # Shortest arc Z→+X: π/2 about +Y
        rot = compute_alignment_rotation((1, 0, 0))
        assert rot is not None
        angle, ax, ay, az = rot
        assert angle == pytest.approx(math.pi / 2.0)
        # axis = z_hat × x_hat = (0, 1, 0)
        assert (ax, ay, az) == pytest.approx((0.0, 1.0, 0.0))
        # And it actually maps Z to +X
        out = _rodrigues(rot, (0.0, 0.0, 1.0))
        assert out == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)

    def test_tuple_plus_z_is_identity(self):
        assert compute_alignment_rotation((0, 0, 1)) is None
        # Auto-normalization: any positive multiple of +Z
        assert compute_alignment_rotation((0, 0, 5)) is None

    def test_tuple_minus_z_is_180_about_x(self):
        rot = compute_alignment_rotation((0, 0, -1))
        assert rot is not None
        angle, ax, ay, az = rot
        assert angle == pytest.approx(math.pi)
        assert (ax, ay, az) == (1.0, 0.0, 0.0)
        out = _rodrigues(rot, (0.0, 0.0, 1.0))
        assert out == pytest.approx((0.0, 0.0, -1.0), abs=1e-12)

    def test_tuple_auto_normalizes(self):
        rot1 = compute_alignment_rotation((1, 0, 0))
        rot2 = compute_alignment_rotation((5, 0, 0))
        assert rot1 == pytest.approx(rot2)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="nonzero"):
            compute_alignment_rotation((0, 0, 0))

    def test_wrong_length_tuple_raises(self):
        with pytest.raises(ValueError, match="3 components"):
            compute_alignment_rotation((1, 0))
        with pytest.raises(ValueError, match="3 components"):
            compute_alignment_rotation((1, 0, 0, 0))

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown align"):
            compute_alignment_rotation("up")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rodrigues(rot, v):
    """Apply (angle, ax, ay, az) rotation to vector ``v``.

    Auto-normalizes the axis (matches gmsh's convention).
    """
    angle, ax, ay, az = rot
    n = math.sqrt(ax * ax + ay * ay + az * az)
    kx, ky, kz = ax / n, ay / n, az / n
    vx, vy, vz = v
    c = math.cos(angle)
    s = math.sin(angle)
    # v' = v cos + (k × v) sin + k (k·v)(1-c)
    cross_x = ky * vz - kz * vy
    cross_y = kz * vx - kx * vz
    cross_z = kx * vy - ky * vx
    dot = kx * vx + ky * vy + kz * vz
    return (
        vx * c + cross_x * s + kx * dot * (1.0 - c),
        vy * c + cross_y * s + ky * dot * (1.0 - c),
        vz * c + cross_z * s + kz * dot * (1.0 - c),
    )
