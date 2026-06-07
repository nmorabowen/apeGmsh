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


# =====================================================================
# Coincident-face orphan-leak regressions  (audit-confirmed bug class)
# =====================================================================

class TestSliceCoincidentFaceOrphans:
    """Slice operations whose cutting plane sits at the same coordinate
    as an existing face used to leave a stranded surface (and lower-dim
    leaks) behind.  Pins the fix added with the :func:`sweep_dangling`
    helper — these are the four failure modes the audit confirmed.
    """

    def test_slice_coincident_with_cavity_face_no_orphans(self, g):
        """Outer box minus inner box, then slice at the cavity bottom.

        Before the fix this left:
        - 1 free-floating surface spanning the full x/y extent at
          z=cavity-bottom
        - 4 stranded boundary curves
        - 4 stranded corner points
        - 9 stale ``_metadata`` entries (consumed cutting-plane tags)

        After the fix: zero of everything.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(-3.3, -0.8, -0.9, 6.6, 1.6, 0.9, label="outer")
        g.model.geometry.add_box(-3.025, -0.675, -0.6, 6.05, 1.35, 0.6, label="inner")
        g.model.boolean.cut(objects=["outer"], tools=["inner"], label="shell")

        # The coincident-face advisory is expected; suppress it from
        # bubbling so the test stays focused on the orphan invariant.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WarnGeomCoincidentFace)
            g.model.geometry.slice(target="shell", axis="z", offset=-0.6)

        orphans = g.model.geometry.find_orphans()
        assert orphans == {0: [], 1: [], 2: []}, (
            f"coincident-face cavity-bottom slice still leaks: {orphans}"
        )

        live = {(d, t) for d in range(4)
                for _, t in gmsh.model.getEntities(d)}
        stale = [dt for dt in g.model._metadata if dt not in live]
        assert not stale, f"stale metadata after slice: {stale}"

    def test_slice_coincident_with_swiss_cheese_inner_face(self, g):
        """Same as above but at the cavity TOP plane (z=0.0 in this
        layout).  Verifies the bug class isn't asymmetric in z.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(-3.3, -0.8, -0.9, 6.6, 1.6, 0.9, label="outer")
        g.model.geometry.add_box(-3.025, -0.675, -0.6, 6.05, 1.35, 0.6, label="inner")
        g.model.boolean.cut(objects=["outer"], tools=["inner"], label="shell")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WarnGeomCoincidentFace)
            g.model.geometry.slice(target="shell", axis="z", offset=0.0)

        orphans = g.model.geometry.find_orphans()
        assert orphans == {0: [], 1: [], 2: []}, (
            f"coincident-face cavity-top slice still leaks: {orphans}"
        )

    def test_slice_idempotent_resliced_same_plane(self, g):
        """Slicing twice at the same offset must not produce new
        entities at ANY dim — the second slice is a no-op beyond the
        (already-present) inter-fragment face.  Snapshot every dim so
        a regression that produced a new bounded surface or stray
        curve slips through dim=3-only checks would still be caught.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(0, 0, 0, 1, 1, 2, label="col")
        g.model.geometry.slice(target="col", axis="z", offset=1.0)

        ents_after_first = {
            d: sorted(t for _, t in gmsh.model.getEntities(d))
            for d in range(4)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WarnGeomCoincidentFace)
            g.model.geometry.slice(target="col", axis="z", offset=1.0)
        ents_after_second = {
            d: sorted(t for _, t in gmsh.model.getEntities(d))
            for d in range(4)
        }

        assert ents_after_first == ents_after_second, (
            f"idempotent slice changed entity set: "
            f"first={ents_after_first} second={ents_after_second}"
        )
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}

    def test_slice_classify_empty_side_warns_but_returns(self, g):
        """When the plane offset sits outside the solid bbox, one side
        of ``classify=True`` is empty.  The op must emit
        :class:`WarnGeomOneSidedCut` (was a silent log line) and still
        return the ``(above, below)`` tuple so callers that pattern-
        match don't crash.
        """
        from apeGmsh.core._geometry_errors import WarnGeomOneSidedCut

        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        with pytest.warns(WarnGeomOneSidedCut):
            result = g.model.geometry.slice(
                target="b", axis="z", offset=10.0, classify=True,
            )
        assert isinstance(result, tuple) and len(result) == 2

    def test_slice_via_label_resolves_multiple_volumes(self, g):
        """Two boxes both labeled ``"deck"`` get sliced together when
        ``slice("deck", ...)`` is called.  Fragments inherit the label,
        each half spans exactly z=0..0.5 or z=0.5..1, and no orphans
        appear.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="deck")
        g.model.geometry.add_box(5, 0, 0, 1, 1, 1, label="deck")
        pieces = g.model.geometry.slice(
            "deck", axis="z", offset=0.5, label="deck",
        )
        assert len(pieces) == 4, (
            f"expected 4 fragments (2 boxes x 2 halves), got {pieces}"
        )
        for tag in pieces:
            assert "deck" in g.labels.labels_for_entity(3, tag)
            _, _, zmin, _, _, zmax = gmsh.model.getBoundingBox(3, tag)
            extent = zmax - zmin
            assert abs(extent - 0.5) < 1e-6, (
                f"fragment {tag} has z-extent {extent}, expected 0.5 — "
                f"slice did not happen at z=0.5"
            )
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}

    def test_slice_no_dim1_or_dim0_orphans(self, g):
        """The audit found dim=1 and dim=0 leaks alongside the surface
        leak.  Walk both dimensions explicitly so a regression that
        only fixes the surface sweep would fail here.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(-3.3, -0.8, -0.9, 6.6, 1.6, 0.9, label="outer")
        g.model.geometry.add_box(-3.025, -0.675, -0.6, 6.05, 1.35, 0.6, label="inner")
        g.model.boolean.cut(objects=["outer"], tools=["inner"], label="shell")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WarnGeomCoincidentFace)
            g.model.geometry.slice(target="shell", axis="z", offset=-0.6)

        gmsh.model.occ.synchronize()
        # Rebuild the volume-boundary closure inline so the test does
        # not depend on the internal helper's exact API.
        keep = set()
        for d_v, v in gmsh.model.getEntities(3):
            keep.add((d_v, v))
            faces = gmsh.model.getBoundary(
                [(3, v)], oriented=False, recursive=False,
            )
            face_dts = [(abs(d), abs(t)) for d, t in faces]
            keep.update(face_dts)
            if face_dts:
                curves = gmsh.model.getBoundary(
                    face_dts, oriented=False, recursive=False, combined=False,
                )
                curve_dts = [(abs(d), abs(t)) for d, t in curves]
                keep.update(curve_dts)
                if curve_dts:
                    points = gmsh.model.getBoundary(
                        curve_dts, oriented=False,
                        recursive=False, combined=False,
                    )
                    keep.update((abs(d), abs(t)) for d, t in points)

        dim1_orphans = [t for _, t in gmsh.model.getEntities(1)
                        if (1, t) not in keep]
        dim0_orphans = [t for _, t in gmsh.model.getEntities(0)
                        if (0, t) not in keep]
        assert dim1_orphans == [], f"dim=1 orphans: {dim1_orphans}"
        assert dim0_orphans == [], f"dim=0 orphans: {dim0_orphans}"


class TestCoincidentFaceWarning:
    """Pin the advisory that fires before the orphan-prone cut runs."""

    def test_coincident_face_warning_fires_for_cavity_bottom(self, g):
        """Slicing at exactly the cavity-bottom z-coordinate of a
        swiss-cheese solid is the canonical coincident-face case;
        :class:`WarnGeomCoincidentFace` must fire so users can choose
        to refactor the offset.
        """
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(-3.3, -0.8, -0.9, 6.6, 1.6, 0.9, label="outer")
        g.model.geometry.add_box(-3.025, -0.675, -0.6, 6.05, 1.35, 0.6, label="inner")
        g.model.boolean.cut(objects=["outer"], tools=["inner"], label="shell")
        with pytest.warns(WarnGeomCoincidentFace):
            g.model.geometry.slice(target="shell", axis="z", offset=-0.6)

    def test_no_coincident_warning_for_clear_plane(self, g):
        """Slicing far from any existing face must NOT trigger the
        coincident-face advisory.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g.model.geometry.slice(axis="z", offset=0.37)
        coincid = [w for w in caught
                   if issubclass(w.category, WarnGeomCoincidentFace)]
        assert coincid == [], (
            f"spurious coincident-face warning on non-coincident slice: "
            f"{[str(w.message) for w in coincid]}"
        )


# =====================================================================
# slice — point= location and dimension-generic cutting (dim 1/2/3/all)
# =====================================================================

class TestSlicePointAndDim:
    """slice() accepts either offset= or point=, and slices any
    dimension via dim=1|2|3|'all' (no longer volume-only).
    """

    def test_point_equivalent_to_offset(self, g):
        """point= places the plane at the point's axis coordinate, the
        same cut offset= would make at that coordinate.
        """
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pieces = g.model.geometry.slice(box, axis="z", point=(0.3, 0.7, 0.5))
        assert len(pieces) == 2
        for tag in pieces:
            _, _, zmin, _, _, zmax = gmsh.model.getBoundingBox(3, tag)
            assert abs((zmax - zmin) - 0.5) < 1e-6, (
                "slice through point z=0.5 did not split the unit box in half"
            )

    def test_point_and_offset_mutually_exclusive(self, g):
        """Passing both a non-zero offset and a point raises."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with pytest.raises(ValueError, match="either point= or offset="):
            g.model.geometry.slice(
                box, axis="z", offset=0.5, point=(0, 0, 0.5),
            )

    def test_invalid_dim_raises(self, g):
        """dim must be 1, 2, 3, or 'all'."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with pytest.raises(ValueError, match="dim must be"):
            g.model.geometry.slice(axis="z", offset=0.5, dim=0)

    def test_dim2_slices_standalone_surface(self, g):
        """dim=2 slices a shell surface into two, leaving no orphans."""
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        pieces = g.model.geometry.slice(axis="x", offset=0.5, dim=2)
        assert len(pieces) == 2
        live_surfs = {t for _, t in gmsh.model.getEntities(2)}
        assert set(pieces) == live_surfs, (
            "sliced surfaces must be exactly the two live surfaces"
        )
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}

    def test_dim1_slices_standalone_curve(self, g):
        """dim=1 slices a curve at the pierce point into two curves."""
        p0 = g.model.geometry.add_point(0, 0, 0)
        p1 = g.model.geometry.add_point(2, 0, 0)
        g.model.geometry.add_line(p0, p1)
        gmsh.model.occ.synchronize()
        pieces = g.model.geometry.slice(axis="x", offset=1.0, dim=1)
        assert len(pieces) == 2
        live_curves = {t for _, t in gmsh.model.getEntities(1)}
        assert set(pieces) == live_curves
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}

    def test_dim_all_defaults_to_maximal_entities(self, g):
        """dim='all' (default) slices a shell model's surfaces even with
        no volumes present — the maximal entities are the surfaces.
        """
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        pieces = g.model.geometry.slice(axis="y", offset=0.5)  # dim='all'
        assert len(pieces) == 2
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}

    def test_dim_all_solid_model_unchanged(self, g):
        """dim='all' on a solid model collapses to volume-only slicing —
        the historical behaviour — leaving no free surfaces.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pieces = g.model.geometry.slice(axis="z", offset=0.5)
        assert len(pieces) == 2
        for _, surf_tag in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surf_tag)
            assert len(up) > 0, f"surface {surf_tag} bounds no volume"

    def test_dim2_classify_and_label(self, g):
        """classify=True + label= works for dim=2 surfaces."""
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        above, below = g.model.geometry.slice(
            axis="x", offset=0.5, dim=2, classify=True, label="half",
        )
        assert len(above) == 1 and len(below) == 1
        for tag in (*above, *below):
            assert "half" in g.labels.labels_for_entity(2, tag)
