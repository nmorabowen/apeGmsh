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

    def test_fuse_box_cylinder_preserves_labels(self, g):
        """Box+cylinder fuse — OCC returns ``result_map=[[], []]`` here, which
        previously caused both labels to be dropped because the absorbed
        fallback had no surviving entities to remap to.  Regression for
        the diagnosis where the two-box fuse test passed by accident."""
        g.model.geometry.add_box(0, 0, 0, 5, 5, 5, label="box")
        g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1, label="cyl")
        result = g.model.boolean.fuse("box", "cyl")
        assert len(result) == 1
        fused_tag = result[0]
        labels = g.labels.labels_for_entity(3, fused_tag)
        assert "box" in labels, f"box label missing after fuse; got {labels}"
        assert "cyl" in labels, f"cyl label missing after fuse; got {labels}"

    def test_intersect_box_cylinder_preserves_labels(self, g):
        """Same shape as the fuse regression — locks the contract for
        intersect, which already worked but only because OCC's intersect
        populates ``result_map`` (unlike fuse)."""
        g.model.geometry.add_box(0, 0, 0, 5, 5, 5, label="box")
        g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1, label="cyl")
        result = g.model.boolean.intersect("box", "cyl")
        assert len(result) == 1
        inter_tag = result[0]
        labels = g.labels.labels_for_entity(3, inter_tag)
        assert "box" in labels, f"box label missing after intersect; got {labels}"
        assert "cyl" in labels, f"cyl label missing after intersect; got {labels}"


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

    def test_fragment_2d_preserves_surfaces_with_default_cleanup(self, g):
        """2D fragment must not drop surfaces under the default
        ``cleanup_free=True`` (topology sweep).

        Regression: the previous centroid-in-bbox heuristic, when
        active, deleted every adjacency-free surface — which in a
        2D model is every surface (there are no volumes to bound).
        Two layers now protect:
        1. ``boolean.fragment`` short-circuits the sweep entirely
           when ``getEntities(3)`` is empty.
        2. Even if it ran, ``sweep_dangling`` would classify every
           ``add_*``-registered surface as user-intentional via
           ``_metadata``.
        """
        p_BL = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=0.5)
        p_BR = g.model.geometry.add_point(2.0, 0.0, 0.0, lc=0.5)
        p_TR = g.model.geometry.add_point(2.0, 1.0, 0.0, lc=0.5)
        p_TL = g.model.geometry.add_point(0.0, 1.0, 0.0, lc=0.5)
        lB = g.model.geometry.add_line(p_BL, p_BR)
        lR = g.model.geometry.add_line(p_BR, p_TR)
        lT = g.model.geometry.add_line(p_TR, p_TL)
        lL = g.model.geometry.add_line(p_TL, p_BL)
        loop = g.model.geometry.add_curve_loop([lB, lR, lT, lL])
        surf = g.model.geometry.add_plane_surface([loop])

        p0 = g.model.geometry.add_point(0.0, 0.5, 0.0, lc=0.5)
        p1 = g.model.geometry.add_point(2.0, 0.5, 0.0, lc=0.5)
        cutting_line = g.model.geometry.add_line(p0, p1)
        g.model.sync()

        g.model.boolean.fragment(
            objects=[(2, surf)],
            tools=[(1, cutting_line)],
            dim=2,
        )
        surfs = [t for _, t in gmsh.model.getEntities(2)]
        assert len(surfs) >= 2, (
            f"2D fragment should split the surface into >=2 pieces, got {surfs}"
        )

    def test_fragment_preserves_embedded_interior_surface(self, g):
        """An embedded interior surface (no upward volume adjacency,
        but lying inside the volume) must survive the cleanup_free
        sweep so it remains addressable for crack/cohesive workflows.
        """
        g.model.geometry.add_box(-50, -50, -50, 100, 100, 100, label='box')
        g.model.geometry.add_rectangle(-10, -10, 0, 20, 20, label='plane')
        g.model.boolean.fragment(objects='box', tools='plane')
        # 'plane' label must still resolve and point at a dim=2 entity
        # that survives in the model.
        plane_tags = g.labels.entities('plane', dim=2)
        assert plane_tags, "embedded 'plane' surface was deleted by cleanup"
        existing = {t for _, t in gmsh.model.getEntities(2)}
        for t in plane_tags:
            assert t in existing, f"label points at non-existent surface {t}"

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

    def test_fragment_cleanup_free_preserves_user_labeled_shell(self, g):
        """The topology-driven sweep that replaced the old centroid-in-
        bbox heuristic preserves user-intentional geometry — any
        surface in ``model._metadata`` (every ``add_*`` primitive
        registers there) or carrying a label survives the sweep,
        regardless of whether its centroid sits inside any volume.

        This is the contract the audit asked for: shell-on-solid
        workflows where the user explicitly created a shell that
        partially overhangs a volume face must not silently lose the
        shell.  The previous heuristic deleted it whenever the
        centroid landed outside every volume bbox.
        """
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label='box')
        # Shell rectangle whose centroid (1.5, 0.5, 0.0) sits OUTSIDE
        # the box bbox in x — under the old centroid heuristic this
        # would have been deleted by ``cleanup_free=True``.
        g.model.geometry.add_rectangle(
            1.0, 0.0, 0.0, 1.0, 1.0, label='overhang',
        )
        g.model.sync()

        g.model.boolean.fragment(
            objects='box', tools='overhang', cleanup_free=True,
        )

        # Some piece of the overhang shell must still resolve as a
        # surviving dim=2 entity (the label may now point at the
        # fragmented sub-piece(s), but at least one surface tag
        # carrying the label exists).
        survivors = g.labels.entities('overhang', dim=2)
        existing_dim2 = {t for _, t in gmsh.model.getEntities(2)}
        live = [t for t in survivors if t in existing_dim2]
        assert live, (
            "topology sweep regressed: user-labeled shell-on-solid "
            "surface was deleted by cleanup_free=True."
        )

    def test_fragment_coincident_face_solids_no_double_surface(self, g):
        """Two abutting boxes that share one face must fragment into
        three surfaces at the shared plane (one outer face from each
        box plus the shared interior face) — without the topology
        sweep, the centroid heuristic dropped the shared face.

        Pins the contract: ``fragment`` plus the default
        ``cleanup_free=True`` produces exactly one shared interior
        surface at the contact plane, and no orphan stranded face.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left")
        g.model.geometry.add_box(1, 0, 0, 1, 1, 1, label="right")
        g.model.boolean.fragment(objects=["left"], tools=["right"])

        # No orphan surfaces.
        orphans = g.model.geometry.find_orphans()
        assert orphans[2] == [], (
            f"fragment of abutting boxes left orphan surfaces: {orphans[2]}"
        )

        # Both volumes survive.
        vols = sorted(t for _, t in gmsh.model.getEntities(3))
        assert len(vols) == 2

        # The shared face is exactly one surface bounded by both
        # volumes — find it via upward adjacencies.
        shared = []
        for _, s in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, s)
            if len(up) == 2:
                shared.append(int(s))
        assert len(shared) == 1, (
            f"expected exactly one shared interior face, got {shared}"
        )

    def test_fragment_cavity_with_coincident_plane_no_orphans(self, g):
        """A swiss-cheese solid fragmented against a cutting plane
        coincident with its cavity face leaves no orphan after the
        topology sweep — even though the coincident-face geometry is
        exactly the failure mode the audit identified.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(-1, -1, -1, 2, 2, 2, label='outer')
        g.model.geometry.add_box(-0.5, -0.5, -0.5, 1, 1, 1, label='inner')
        g.model.boolean.cut(objects=['outer'], tools=['inner'], label='shell')

        # Cutting plane coincident with the cavity bottom (z=-0.5).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', WarnGeomCoincidentFace)
            plane = g.model.geometry.add_axis_cutting_plane('z', offset=-0.5)
            g.model.boolean.fragment(
                objects=['shell'], tools=[(2, plane)],
            )

        orphans = g.model.geometry.find_orphans()
        assert orphans == {0: [], 1: [], 2: []}, (
            f"fragment with coincident cutting plane left orphans: "
            f"{orphans}"
        )


# =====================================================================
# label= override
# =====================================================================

class TestLabelOverride:
    """``label=`` on cut/fuse/intersect drops input labels and attaches
    a single new label to the result entities."""

    def test_fuse_label_replaces_inputs(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 5, label="box")
        g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1, label="cyl")
        result = g.model.boolean.fuse("box", "cyl", label="merged")
        labels = g.labels.labels_for_entity(3, result[0])
        assert "merged" in labels
        assert "box" not in labels
        assert "cyl" not in labels

    def test_cut_label_replaces_object(self, g):
        g.model.geometry.add_box(0, 0, 0, 2, 2, 2, label="box")
        g.model.geometry.add_cylinder(1, 1, 0, 0, 0, 2, 0.5, label="cyl")
        result = g.model.boolean.cut("box", "cyl", label="holey")
        labels = g.labels.labels_for_entity(3, result[0])
        assert "holey" in labels
        assert "box" not in labels

    def test_intersect_label_replaces_inputs(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 5, label="box")
        g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1, label="cyl")
        result = g.model.boolean.intersect("box", "cyl", label="overlap")
        labels = g.labels.labels_for_entity(3, result[0])
        assert "overlap" in labels
        assert "box" not in labels
        assert "cyl" not in labels

    def test_label_override_keeps_unrelated_label_membership(self, g):
        """A label that points at the input AND at an unrelated entity
        must keep the unrelated entity after the override — only the
        result tag should be stripped."""
        # Two boxes share the label "shared"; only one participates in
        # the fuse with the cylinder.
        g.model.geometry.add_box(0, 0, 0, 5, 5, 5, label="shared")
        g.model.geometry.add_box(20, 0, 0, 1, 1, 1, label="shared")
        g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1, label="cyl")
        # The "shared" label now points at both boxes (dim=3).
        before = set(g.labels.entities("shared", dim=3))
        assert len(before) == 2
        result = g.model.boolean.fuse("cyl", g.labels.entities("shared", dim=3)[:1],
                                       label="merged")
        # "shared" should still exist with the unrelated box as its only entity.
        after = set(g.labels.entities("shared", dim=3))
        assert len(after) == 1
        assert result[0] not in after


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
