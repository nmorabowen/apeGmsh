"""
Tests for the cutting / slicing API on ``g.model.geometry``.

Covers:
- add_cutting_plane
- add_axis_cutting_plane
- cut_by_surface
- cut_by_plane  (above/below classification)
- slice  (atomic cut + cleanup)
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest


# =====================================================================
# add_cutting_plane
# =====================================================================

class TestAddCuttingPlane:
    """Tests for g.model.geometry.add_cutting_plane."""

    def test_returns_surface_tag(self, g):
        """add_cutting_plane returns a valid dim=2 surface tag."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        tag = g.model.geometry.add_cutting_plane(
            point=[0, 0, 0.5],
            normal_vector=[0, 0, 1],
        )
        assert isinstance(tag, int)
        # Tag must exist as a dim=2 entity in the Gmsh model
        surf_tags = [t for _, t in gmsh.model.getEntities(2)]
        assert tag in surf_tags

    def test_metadata_stores_normal_and_point(self, g):
        """Metadata for the cutting plane must contain 'normal' and 'point'."""
        g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
        tag = g.model.geometry.add_cutting_plane(
            point=[1.0, 0.0, 0.0],
            normal_vector=[1.0, 0.0, 0.0],
        )
        entry = g.model._metadata.get((2, tag))
        assert entry is not None, "metadata entry missing for cutting plane"
        assert 'normal' in entry
        assert 'point' in entry
        np.testing.assert_allclose(entry['normal'], (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(entry['point'], (1.0, 0.0, 0.0), atol=1e-12)

    def test_metadata_kind_is_cutting_plane(self, g):
        """The metadata 'kind' field must be 'cutting_plane'."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        tag = g.model.geometry.add_cutting_plane(
            point=[0, 0, 0.5],
            normal_vector=[0, 0, 1],
        )
        entry = g.model._metadata[(2, tag)]
        assert entry['kind'] == 'cutting_plane'

    def test_non_unit_normal_is_normalised(self, g):
        """A non-unit normal_vector must be normalised in metadata."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        tag = g.model.geometry.add_cutting_plane(
            point=[0, 0, 0],
            normal_vector=[0, 3.0, 0],
        )
        stored = np.array(g.model._metadata[(2, tag)]['normal'])
        np.testing.assert_allclose(stored, [0.0, 1.0, 0.0], atol=1e-12)


# =====================================================================
# add_axis_cutting_plane
# =====================================================================

class TestAddAxisCuttingPlane:
    """Tests for g.model.geometry.add_axis_cutting_plane."""

    def test_z_axis_plane_normal(self, g):
        """axis='z' must produce a plane with normal (0,0,1)."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        tag = g.model.geometry.add_axis_cutting_plane('z', offset=0.5)
        entry = g.model._metadata[(2, tag)]
        np.testing.assert_allclose(entry['normal'], (0.0, 0.0, 1.0), atol=1e-12)
        np.testing.assert_allclose(entry['point'], (0.0, 0.0, 0.5), atol=1e-12)

    def test_x_axis_with_offset(self, g):
        """axis='x', offset=2.0 must produce point=(2,0,0), normal=(1,0,0)."""
        g.model.geometry.add_box(0, 0, 0, 4, 1, 1)
        tag = g.model.geometry.add_axis_cutting_plane('x', offset=2.0)
        entry = g.model._metadata[(2, tag)]
        np.testing.assert_allclose(entry['normal'], (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(entry['point'], (2.0, 0.0, 0.0), atol=1e-12)


# =====================================================================
# cut_by_surface
# =====================================================================

class TestCutBySurface:
    """Tests for g.model.geometry.cut_by_surface."""

    def test_box_split_into_two_volumes(self, g):
        """Cutting a box at its midplane must produce exactly 2 volumes."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        plane = g.model.geometry.add_axis_cutting_plane('z', offset=0.5)
        pieces = g.model.geometry.cut_by_surface(box, plane)
        assert len(pieces) == 2
        # Verify they are real volumes in Gmsh
        vol_tags = [t for _, t in gmsh.model.getEntities(3)]
        for p in pieces:
            assert p in vol_tags

    def test_label_inheritance_single_label(self, g):
        """When a labeled box is cut, fragments inherit the label."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="slab")
        plane = g.model.geometry.add_axis_cutting_plane('z', offset=0.5)
        pieces = g.model.geometry.cut_by_surface(box, plane)
        # Both fragments should carry the "slab" label
        for tag in pieces:
            labels = g.labels.labels_for_entity(3, tag)
            assert "slab" in labels, (
                f"Fragment {tag} missing inherited label 'slab'; has {labels}"
            )


# =====================================================================
# cut_by_plane
# =====================================================================

class TestCutByPlane:
    """Tests for g.model.geometry.cut_by_plane."""

    def test_above_below_classification(self, g):
        """cut_by_plane must return (above, below) with at least 1 tag each."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        plane = g.model.geometry.add_axis_cutting_plane('z', offset=1.0)
        above, below = g.model.geometry.cut_by_plane(box, plane)
        assert len(above) >= 1
        assert len(below) >= 1
        assert len(above) + len(below) == 2

    def test_label_above_below(self, g):
        """label_above and label_below are applied correctly."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        plane = g.model.geometry.add_axis_cutting_plane('z', offset=1.0)
        above, below = g.model.geometry.cut_by_plane(
            box, plane,
            label_above="upper", label_below="lower",
        )
        for tag in above:
            assert "upper" in g.labels.labels_for_entity(3, tag)
        for tag in below:
            assert "lower" in g.labels.labels_for_entity(3, tag)


# =====================================================================
# slice
# =====================================================================

class TestSlice:
    """Tests for g.model.geometry.slice (atomic cut + cleanup)."""

    def test_flat_list_mode(self, g):
        """slice with classify=False returns a flat list of tags."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pieces = g.model.geometry.slice(box, axis='z', offset=0.5)
        assert isinstance(pieces, list)
        assert len(pieces) == 2

    def test_classify_mode(self, g):
        """slice with classify=True returns (above, below) tuple."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        result = g.model.geometry.slice(box, axis='z', offset=1.0, classify=True)
        assert isinstance(result, tuple)
        above, below = result
        assert len(above) >= 1
        assert len(below) >= 1

    def test_cleanup_removes_orphaned_geometry(self, g):
        """After slice, no orphaned cutting-plane geometry should remain.

        The cutting plane adds points, lines, and a surface.  After
        slicing, only the volume fragments and their bounding entities
        should survive -- the cutting plane's own corner points, edges,
        and surface must be cleaned up.
        """
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        # Snapshot entity counts before slice
        pre_surfs = set(t for _, t in gmsh.model.getEntities(2))

        pieces = g.model.geometry.slice(box, axis='z', offset=0.5)

        # All surviving surfaces must bound a volume (no free surfaces)
        for _, surf_tag in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surf_tag)
            assert len(up) > 0, (
                f"Surface {surf_tag} is orphaned (does not bound any volume)"
            )

    def test_slice_by_label_string(self, g):
        """slice accepts a label string for the solid argument."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="block")
        pieces = g.model.geometry.slice("block", axis='y', offset=0.5)
        assert len(pieces) == 2
        # All fragments should exist as dim=3 entities
        vol_tags = {t for _, t in gmsh.model.getEntities(3)}
        for p in pieces:
            assert p in vol_tags

    def test_slice_with_label_propagation(self, g):
        """slice with label= assigns the label to every fragment."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pieces = g.model.geometry.slice(
            box, axis='x', offset=0.5, label="half",
        )
        for tag in pieces:
            labels = g.labels.labels_for_entity(3, tag)
            assert "half" in labels
