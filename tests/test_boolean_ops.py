"""
Tests for the boolean operations API on ``g.model.boolean``.

Covers:
- fuse
- cut
- intersect
- fragment
"""
from __future__ import annotations

import gmsh
import pytest


# =====================================================================
# fuse
# =====================================================================

class TestFuse:
    """Tests for g.model.boolean.fuse."""

    def test_fuse_two_overlapping_boxes_one_volume(self, g):
        """Fusing two overlapping boxes must produce exactly 1 volume."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        result = g.model.boolean.fuse(box_a, box_b)
        assert len(result) == 1
        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 1

    def test_fuse_preserves_labels(self, g):
        """Labels on the input objects survive a fuse operation."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="part_A")
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1, label="part_B")
        result = g.model.boolean.fuse(box_a, box_b)
        assert len(result) == 1
        fused_tag = result[0]
        # Both labels should survive on the fused volume
        labels_a = g.labels.labels_for_entity(3, fused_tag)
        labels_b = g.labels.labels_for_entity(3, fused_tag)
        assert "part_A" in labels_a, f"part_A missing after fuse; got {labels_a}"
        assert "part_B" in labels_b, f"part_B missing after fuse; got {labels_b}"


# =====================================================================
# cut
# =====================================================================

class TestCut:
    """Tests for g.model.boolean.cut."""

    def test_cut_box_minus_cylinder(self, g):
        """Cutting a cylinder from a box must reduce the volume count."""
        box = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
        cyl = g.model.geometry.add_cylinder(1, 1, 0, 0, 0, 2, 0.5)
        result = g.model.boolean.cut(box, cyl)
        assert len(result) >= 1
        # The result should be a box with a hole; still 1 volume
        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 1

    def test_cut_removes_consumed_tool_from_metadata(self, g):
        """The consumed tool entity must be purged from _metadata."""
        box = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
        cyl = g.model.geometry.add_cylinder(1, 1, 0, 0, 0, 2, 0.5)
        assert (3, cyl) in g.model._metadata
        g.model.boolean.cut(box, cyl)
        assert (3, cyl) not in g.model._metadata, (
            "Consumed tool should be removed from _metadata"
        )


# =====================================================================
# intersect
# =====================================================================

class TestIntersect:
    """Tests for g.model.boolean.intersect."""

    def test_intersect_two_overlapping_boxes(self, g):
        """Intersection of two overlapping boxes keeps only the overlap."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        result = g.model.boolean.intersect(box_a, box_b)
        assert len(result) == 1
        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 1


# =====================================================================
# fragment
# =====================================================================

class TestFragment:
    """Tests for g.model.boolean.fragment."""

    def test_fragment_two_overlapping_boxes_three_volumes(self, g):
        """Fragmenting two overlapping boxes must produce 3 volumes."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        result = g.model.boolean.fragment(box_a, box_b)
        assert len(result) == 3
        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 3

    def test_fragment_cleanup_free_removes_orphan_surfaces(self, g):
        """With cleanup_free=True, no surface should be unbounded after fragment."""
        box = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
        # A cutting rectangle that extends beyond the box -- the overhang
        # part becomes a "free" surface with no bounding volume.
        plane = g.model.geometry.add_axis_cutting_plane('z', offset=1.0)
        g.model.boolean.fragment(box, plane, cleanup_free=True)
        for _, surf_tag in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surf_tag)
            assert len(up) > 0, (
                f"Surface {surf_tag} is free (unbounded) after fragment cleanup"
            )


# =====================================================================
# Input forms
# =====================================================================

class TestInputForms:
    """Tests that boolean ops accept both DimTag tuples and plain int tags."""

    def test_dimtag_tuple_input(self, g):
        """Passing (3, tag) tuples must work for fuse."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        result = g.model.boolean.fuse((3, box_a), (3, box_b))
        assert len(result) == 1

    def test_list_of_tags_input(self, g):
        """Passing a list of bare int tags must work for fragment."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        box_c = g.model.geometry.add_box(4, 0, 0, 1, 1, 1)
        result = g.model.boolean.fragment([box_a, box_b], box_c)
        # box_a and box_b overlap -> 3 volumes from that pair, plus box_c
        # which does not overlap -> total 4 volumes
        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 4
