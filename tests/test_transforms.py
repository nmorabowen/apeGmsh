"""
Tests for ``g.model.transforms.*`` — translate, rotate, scale, mirror,
copy, extrude, and revolve.
"""
from __future__ import annotations

import math

import gmsh
import numpy as np
import pytest


# =====================================================================
# Helpers
# =====================================================================

def _centroid_3d(tag: int) -> tuple[float, float, float]:
    """Return (cx, cy, cz) for a dim=3 entity via Gmsh OCC."""
    return gmsh.model.occ.getCenterOfMass(3, tag)


def _volume(tag: int) -> float:
    """Return the volume (mass at dim=3) of an entity."""
    return gmsh.model.occ.getMass(3, tag)


def _entity_tags(dim: int) -> list[int]:
    """Return all entity tags at *dim*."""
    return [t for _, t in gmsh.model.getEntities(dim)]


# =====================================================================
# Translate
# =====================================================================

class TestTranslate:

    def test_translate(self, g):
        """Box at origin translated by (10,0,0) has centroid at (10.5,0.5,0.5)."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        cx0, cy0, cz0 = _centroid_3d(box)
        assert cx0 == pytest.approx(0.5, abs=1e-10)

        g.model.transforms.translate(box, 10, 0, 0, dim=3)
        cx, cy, cz = _centroid_3d(box)

        assert cx == pytest.approx(10.5, abs=1e-10)
        assert cy == pytest.approx(0.5, abs=1e-10)
        assert cz == pytest.approx(0.5, abs=1e-10)


# =====================================================================
# Rotate
# =====================================================================

class TestRotate:

    def test_rotate_90_z(self, g):
        """Box centred at (1.5, 0.5, 0.5) rotated 90 deg about Z through
        origin -> centroid at (-0.5, 1.5, 0.5)."""
        # Box from (1,0,0) to (2,1,1) => centroid (1.5, 0.5, 0.5)
        box = g.model.geometry.add_box(1, 0, 0, 1, 1, 1)
        cx0, cy0, _ = _centroid_3d(box)
        assert cx0 == pytest.approx(1.5, abs=1e-10)
        assert cy0 == pytest.approx(0.5, abs=1e-10)

        g.model.transforms.rotate(box, math.pi / 2, az=1, dim=3)
        cx, cy, cz = _centroid_3d(box)

        # Rotation of (1.5, 0.5) by 90 deg CCW -> (-0.5, 1.5)
        assert cx == pytest.approx(-0.5, abs=1e-6)
        assert cy == pytest.approx(1.5, abs=1e-6)
        assert cz == pytest.approx(0.5, abs=1e-6)


# =====================================================================
# Scale
# =====================================================================

class TestScale:

    def test_scale_uniform(self, g):
        """Uniform 2x scale of a unit box -> volume = 8."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        assert _volume(box) == pytest.approx(1.0, abs=1e-10)

        g.model.transforms.scale(box, 2, 2, 2, dim=3)
        assert _volume(box) == pytest.approx(8.0, abs=1e-10)

    def test_scale_nonuniform(self, g):
        """Non-uniform scale (2,1,1) doubles the volume."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        assert _volume(box) == pytest.approx(1.0, abs=1e-10)

        g.model.transforms.scale(box, 2, 1, 1, dim=3)
        assert _volume(box) == pytest.approx(2.0, abs=1e-10)


# =====================================================================
# Mirror
# =====================================================================

class TestMirror:

    def test_mirror_yz_plane(self, g):
        """Copy a box at x=1, mirror the copy through YZ plane -> 2 boxes."""
        box = g.model.geometry.add_box(1, 0, 0, 1, 1, 1)
        assert len(_entity_tags(3)) == 1

        # Copy first, then mirror the copy
        copies = g.model.transforms.copy(box, dim=3)
        assert len(copies) == 1
        copy_tag = copies[0]
        assert len(_entity_tags(3)) == 2

        # Mirror through the YZ plane: 1*x + 0*y + 0*z + 0 = 0
        g.model.transforms.mirror(copy_tag, 1, 0, 0, 0, dim=3)

        # Original centroid at (1.5, 0.5, 0.5)
        cx_orig, _, _ = _centroid_3d(box)
        assert cx_orig == pytest.approx(1.5, abs=1e-6)

        # Mirrored copy centroid at (-1.5, 0.5, 0.5)
        cx_copy, _, _ = _centroid_3d(copy_tag)
        assert cx_copy == pytest.approx(-1.5, abs=1e-6)


# =====================================================================
# Copy
# =====================================================================

class TestCopy:

    def test_copy(self, g):
        """Copy returns new tags; originals remain untouched."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        original_centroid = _centroid_3d(box)

        copies = g.model.transforms.copy(box, dim=3)

        assert len(copies) == 1
        assert copies[0] != box
        assert len(_entity_tags(3)) == 2

        # Original centroid unchanged
        cx, cy, cz = _centroid_3d(box)
        assert cx == pytest.approx(original_centroid[0], abs=1e-10)
        assert cy == pytest.approx(original_centroid[1], abs=1e-10)
        assert cz == pytest.approx(original_centroid[2], abs=1e-10)

        # Copy has the same centroid (clone at the same location)
        ccx, ccy, ccz = _centroid_3d(copies[0])
        assert ccx == pytest.approx(original_centroid[0], abs=1e-10)
        assert ccy == pytest.approx(original_centroid[1], abs=1e-10)
        assert ccz == pytest.approx(original_centroid[2], abs=1e-10)


# =====================================================================
# Extrude
# =====================================================================

class TestExtrude:

    def test_extrude_surface(self, g):
        """Extruding a rectangle along Z produces a volume with correct mass."""
        rect = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)

        result = g.model.transforms.extrude(rect, 0, 0, 1, dim=2)

        # Result is a list of dimtags.  There should be at least one
        # volume (dim=3) entity among them.
        vol_tags = [t for d, t in result if d == 3]
        assert len(vol_tags) >= 1

        # The resulting volume should have volume approx 1.0
        vol = gmsh.model.occ.getMass(3, vol_tags[0])
        assert vol == pytest.approx(1.0, abs=1e-10)

    def test_extrude_with_layers(self, g):
        """Extruding with num_elements produces correct geometry."""
        rect = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)

        result = g.model.transforms.extrude(
            rect, 0, 0, 2,
            dim=2,
            num_elements=[5],
        )

        # Should still produce a volume
        vol_tags = [t for d, t in result if d == 3]
        assert len(vol_tags) >= 1

        vol = gmsh.model.occ.getMass(3, vol_tags[0])
        assert vol == pytest.approx(2.0, abs=1e-10)


# =====================================================================
# Revolve
# =====================================================================

class TestRevolve:

    def test_revolve_surface(self, g):
        """Rectangle revolved 360 deg around an axis produces a volume."""
        rect = g.model.geometry.add_rectangle(2, 0, 0, 1, 1)

        result = g.model.transforms.revolve(
            rect,
            2 * math.pi,
            ay=1,
            dim=2,
        )

        # There should be at least one volume
        vol_tags = [t for d, t in result if d == 3]
        assert len(vol_tags) >= 1

        # Volume should be positive (non-degenerate solid)
        vol = gmsh.model.occ.getMass(3, vol_tags[0])
        assert vol > 0
