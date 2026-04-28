"""
Tests for the ``anchor=`` and ``align=`` kwargs on section factories
and on the in-session ``g.sections.*`` builders.

The factory wiring (Part-returning functions) and the builder wiring
(direct in-session construction) share the same pure-math helpers in
``apeGmsh.core._section_placement``, but they apply them in different
plumbing — one inside a Part's own gmsh session before STEP export,
the other in the parent session against a snapshot delta.  Both paths
need their own coverage.
"""
from __future__ import annotations

import math

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.sections import (
    W_solid, W_shell, W_profile,
    rect_solid, pipe_solid,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_TOL = 1e-6  # OCC bbox precision


def _bbox_of_inst(inst):
    """Return the bbox stored on the Instance after add()."""
    assert inst.bbox is not None, "instance has no bbox"
    return inst.bbox


def _approx(actual, expected, tol=_TOL):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


# ---------------------------------------------------------------------
# Factory: W_solid anchor=
# ---------------------------------------------------------------------

class TestWSolidAnchor:
    def test_default_start_extrudes_along_positive_z(self):
        col = W_solid(bf=150, tf=20, h=300, tw=10, length=1000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                xmin, ymin, zmin, xmax, ymax, zmax = _bbox_of_inst(inst)
                assert abs(zmin - 0.0) <= _TOL
                assert abs(zmax - 1000.0) <= _TOL
        finally:
            col.cleanup()

    def test_midspan_centers_z_about_origin(self):
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000, anchor="midspan",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
                assert abs(zmin - (-500.0)) <= _TOL
                assert abs(zmax - 500.0) <= _TOL
        finally:
            col.cleanup()

    def test_end_pulls_far_face_to_origin(self):
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000, anchor="end",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
                assert abs(zmin - (-1000.0)) <= _TOL
                assert abs(zmax - 0.0) <= _TOL
        finally:
            col.cleanup()

    def test_centroid_centers_xy_and_z(self):
        # W_solid is already symmetric in XY about origin, so the
        # centroid translation degenerates to (0, 0, -length/2) — same
        # as midspan.  The point of this test is that "centroid" runs
        # without errors and matches the symmetry.
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000, anchor="centroid",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                xmin, _, zmin, xmax, _, zmax = _bbox_of_inst(inst)
                assert abs(zmin - (-500.0)) <= _TOL
                assert abs(zmax - 500.0) <= _TOL
                assert abs(xmin + xmax) <= _TOL  # symmetric about x=0
        finally:
            col.cleanup()

    def test_explicit_tuple_anchor_makes_local_point_origin(self):
        # Pick a point that's NOT a named anchor: x=10, y=20, z=250.
        # After anchor, that point in the local frame lands at (0,0,0).
        # The original z=0 face moves to z=-250; far face to z=750.
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000,
            anchor=(10, 20, 250),
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                xmin, ymin, zmin, xmax, ymax, zmax = _bbox_of_inst(inst)
                # X was -75..+75 → -85..+65
                assert abs(xmin - (-85.0)) <= _TOL
                assert abs(xmax - 65.0) <= _TOL
                # Z was 0..1000 → -250..+750
                assert abs(zmin - (-250.0)) <= _TOL
                assert abs(zmax - 750.0) <= _TOL
        finally:
            col.cleanup()

    def test_unknown_anchor_raises(self):
        with pytest.raises(ValueError, match="Unknown anchor"):
            W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="middle",
            )


# ---------------------------------------------------------------------
# Factory: W_solid align=
# ---------------------------------------------------------------------

class TestWSolidAlign:
    def test_default_z_no_rotation(self):
        col = W_solid(bf=150, tf=20, h=300, tw=10, length=1000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
                assert (zmax - zmin) == pytest.approx(1000.0, abs=_TOL)
        finally:
            col.cleanup()

    def test_align_x_extrudes_along_world_x(self):
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000, align="x",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
                # Extrusion 0..length about local Z; align=x maps Z→X.
                assert abs(xmin - 0.0) <= _TOL
                assert abs(xmax - 1000.0) <= _TOL
        finally:
            col.cleanup()

    def test_align_y_extrudes_along_world_y(self):
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000, align="y",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                _, ymin, _, _, ymax, _ = _bbox_of_inst(inst)
                assert abs(ymin - 0.0) <= _TOL
                assert abs(ymax - 1000.0) <= _TOL
        finally:
            col.cleanup()

    def test_align_tuple_plus_x_matches_named_x_extent(self):
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000, align=(1, 0, 0),
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
                # Shortest-arc Z→+X gives the same X extent.
                assert abs(xmin - 0.0) <= _TOL
                assert abs(xmax - 1000.0) <= _TOL
        finally:
            col.cleanup()

    def test_unknown_align_raises(self):
        with pytest.raises(ValueError, match="Unknown align"):
            W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000, align="up",
            )


# ---------------------------------------------------------------------
# Factory: W_solid combined anchor + align (the headline sanity check)
# ---------------------------------------------------------------------

class TestWSolidAnchorPlusAlign:
    def test_midspan_plus_align_x_centers_x_about_origin(self):
        # The headline numerical check called out in the spec:
        #   length=1000, anchor="midspan", align="x"
        # → bbox X range == [-500, +500] within OCC precision (~1e-6).
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000,
            anchor="midspan", align="x",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col)
                xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
                assert abs(xmin - (-500.0)) <= _TOL
                assert abs(xmax - 500.0) <= _TOL
        finally:
            col.cleanup()


# ---------------------------------------------------------------------
# Factory: labels survive anchor/align
# ---------------------------------------------------------------------

class TestLabelsSurvivePlacement:
    def test_w_solid_labels_intact_after_midspan_align_x(self):
        col = W_solid(
            bf=150, tf=20, h=300, tw=10, length=1000,
            anchor="midspan", align="x",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col, label="c")
                labels = g.labels.get_all()
                assert "c.web" in labels
                assert "c.top_flange" in labels
                assert "c.bottom_flange" in labels
                # Volumes still resolvable.
                assert len(g.labels.entities("c.web")) >= 1
        finally:
            col.cleanup()


# ---------------------------------------------------------------------
# Factory: W_shell
# ---------------------------------------------------------------------

class TestWShellPlacement:
    def test_midspan_centers_z(self):
        bm = W_shell(
            bf=190, tf=14, h=428, tw=9, length=4000, anchor="midspan",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(bm)
                _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
                assert abs(zmin - (-2000.0)) <= _TOL
                assert abs(zmax - 2000.0) <= _TOL
        finally:
            bm.cleanup()

    def test_align_x_routes_extrusion_to_world_x(self):
        bm = W_shell(
            bf=190, tf=14, h=428, tw=9, length=4000, align="x",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(bm)
                xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
                assert abs(xmin - 0.0) <= _TOL
                assert abs(xmax - 4000.0) <= _TOL
        finally:
            bm.cleanup()

    def test_labels_survive_align(self):
        bm = W_shell(
            bf=190, tf=14, h=428, tw=9, length=4000,
            anchor="midspan", align="x",
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(bm, label="b")
                labels = g.labels.get_all()
                assert "b.top_flange" in labels
                assert "b.web" in labels
                assert "b.bottom_flange" in labels
        finally:
            bm.cleanup()


# ---------------------------------------------------------------------
# Factory: W_profile
# ---------------------------------------------------------------------

class TestWProfilePlacement:
    def test_default_start_unchanged(self):
        prof = W_profile(bf=150, tf=20, h=300, tw=10)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(prof)
                xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
                # Original profile is centered: x ∈ [-75, 75]
                assert abs(xmin - (-75.0)) <= _TOL
                assert abs(xmax - 75.0) <= _TOL
        finally:
            prof.cleanup()

    def test_tuple_anchor_translates_in_xy(self):
        # Re-origin the profile so its (10, 20, 0) point lands at origin.
        prof = W_profile(
            bf=150, tf=20, h=300, tw=10, anchor=(10, 20, 0),
        )
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(prof)
                xmin, ymin, _, xmax, ymax, _ = _bbox_of_inst(inst)
                # Original X was -75..+75 → shifted -10 → -85..+65
                assert abs(xmin - (-85.0)) <= _TOL
                assert abs(xmax - 65.0) <= _TOL
        finally:
            prof.cleanup()

    def test_named_length_required_anchor_raises_on_profile(self):
        # W_profile has no length, so anchor="end"/"midspan"/"centroid"
        # all bottom out in compute_anchor_offset's "requires a length"
        # branch.
        with pytest.raises(ValueError, match="requires a length"):
            W_profile(bf=150, tf=20, h=300, tw=10, anchor="end")


# ---------------------------------------------------------------------
# Factory: rect_solid + pipe_solid (spot check the simpler factories)
# ---------------------------------------------------------------------

class TestSimpleSolidsPlacement:
    def test_rect_solid_midspan(self):
        bar = rect_solid(b=100, h=200, length=500, anchor="midspan")
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(bar)
                _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
                assert abs(zmin - (-250.0)) <= _TOL
                assert abs(zmax - 250.0) <= _TOL
        finally:
            bar.cleanup()

    def test_pipe_solid_align_y(self):
        rod = pipe_solid(r=50, length=2000, align="y")
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(rod)
                _, ymin, _, _, ymax, _ = _bbox_of_inst(inst)
                assert abs(ymin - 0.0) <= _TOL
                assert abs(ymax - 2000.0) <= _TOL
        finally:
            rod.cleanup()


# ---------------------------------------------------------------------
# Builder: g.sections.W_solid anchor + align
# ---------------------------------------------------------------------

class TestBuilderWSolidPlacement:
    def test_default_extrudes_along_positive_z(self):
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000, label="c",
            )
            _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
            assert abs(zmin - 0.0) <= _TOL
            assert abs(zmax - 1000.0) <= _TOL

    def test_midspan_centers_z(self):
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="midspan", label="c",
            )
            _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
            assert abs(zmin - (-500.0)) <= _TOL
            assert abs(zmax - 500.0) <= _TOL

    def test_align_x_routes_extrusion_to_world_x(self):
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                align="x", label="c",
            )
            xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
            assert abs(xmin - 0.0) <= _TOL
            assert abs(xmax - 1000.0) <= _TOL

    def test_midspan_plus_align_x_centers_x_about_origin(self):
        # Builder mirror of the headline factory sanity check.
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="midspan", align="x", label="c",
            )
            xmin, _, _, xmax, _, _ = _bbox_of_inst(inst)
            assert abs(xmin - (-500.0)) <= _TOL
            assert abs(xmax - 500.0) <= _TOL

    def test_labels_survive_placement(self):
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="midspan", align="x", label="c",
            )
            labels = g.labels.get_all()
            # Builder labels are dotted directly during build_fn.
            assert "c.web" in labels
            assert "c.top_flange" in labels
            assert "c.bottom_flange" in labels
            assert "c.start_face" in labels
            assert "c.end_face" in labels
            assert len(g.labels.entities("c.web")) >= 1
            assert len(g.labels.entities("c.start_face")) >= 1

    def test_unknown_anchor_raises(self):
        with apeGmsh(model_name="t") as g:
            with pytest.raises(ValueError, match="Unknown anchor"):
                g.sections.W_solid(
                    bf=150, tf=20, h=300, tw=10, length=1000,
                    anchor="middle", label="c",
                )


# ---------------------------------------------------------------------
# Builder: g.sections.rect_solid + W_shell
# ---------------------------------------------------------------------

class TestBuilderSimpleSectionsPlacement:
    def test_rect_solid_midspan(self):
        with apeGmsh(model_name="t") as g:
            inst = g.sections.rect_solid(
                b=100, h=200, length=500, anchor="midspan", label="r",
            )
            _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
            assert abs(zmin - (-250.0)) <= _TOL
            assert abs(zmax - 250.0) <= _TOL
            assert "r.body" in g.labels.get_all()

    def test_W_shell_align_y(self):
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_shell(
                bf=190, tf=14, h=428, tw=9, length=4000,
                align="y", label="b",
            )
            _, ymin, _, _, ymax, _ = _bbox_of_inst(inst)
            assert abs(ymin - 0.0) <= _TOL
            assert abs(ymax - 4000.0) <= _TOL
            labels = g.labels.get_all()
            assert "b.top_flange" in labels
            assert "b.web" in labels


# ---------------------------------------------------------------------
# Builder: anchor/align coexists with user translate/rotate and with
# pre-existing parts in the same session.
# ---------------------------------------------------------------------

class TestBuilderPlacementInteractions:
    def test_user_translate_stacks_on_anchor(self):
        # midspan first centers Z about 0, then user translate=(0,0,100)
        # shifts the whole thing — Z becomes [-400, +600].
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="midspan",
                translate=(0.0, 0.0, 100.0),
                label="c",
            )
            _, _, zmin, _, _, zmax = _bbox_of_inst(inst)
            assert abs(zmin - (-400.0)) <= _TOL
            assert abs(zmax - 600.0) <= _TOL

    def test_other_part_in_session_is_not_disturbed(self):
        # Add a plain volume first, then build a section with align=x
        # and verify the plain volume's labels and bbox are intact —
        # the in-session PG snapshot/restore must scope to the section.
        with apeGmsh(model_name="t") as g:
            tag = g.model.geometry.add_box(
                10, 0, 0, 1, 1, 1, label="bystander",
            )
            assert "bystander" in g.labels.get_all()
            bystander_tags_before = g.labels.entities("bystander")

            g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="midspan", align="x", label="c",
            )

            # Bystander label still resolves to the same entity tags.
            assert "bystander" in g.labels.get_all()
            assert g.labels.entities("bystander") == bystander_tags_before
            # And the section's labels are also present.
            assert "c.web" in g.labels.get_all()

    def test_user_rotate_after_anchor_preserves_labels(self):
        # anchor=midspan applies a translate (which apply_placement
        # snapshots/restores).  Then the user passes rotate=, which the
        # builder applies via raw occ.rotate + synchronize().  That
        # second sync is exactly the operation that drops PGs whose
        # boundary sub-topology was renumbered.  If the second pass
        # isn't snapshot-protected, the section's face labels go away.
        with apeGmsh(model_name="t") as g:
            inst = g.sections.W_solid(
                bf=150, tf=20, h=300, tw=10, length=1000,
                anchor="midspan",
                rotate=(math.pi / 2, 1.0, 0.0, 0.0),  # 90° about +X
                label="c",
            )
            labels = g.labels.get_all()
            # Volume labels — these survive rotate (top-dim tags don't
            # get renumbered).
            assert "c.web" in labels
            assert "c.top_flange" in labels
            # Face labels — these reference dim=2 boundary entities
            # whose tag IDs get renumbered by the rotate's OCC sync.
            assert "c.start_face" in labels
            assert "c.end_face" in labels
            # And they must still resolve to live entities.
            assert len(g.labels.entities("c.start_face")) >= 1
            assert len(g.labels.entities("c.end_face")) >= 1
            # Sanity on bbox: a 1000-long bar rotated 90° about X should
            # have its Z extent compressed and its Y extent stretched
            # to length/2 ± length/2 (anchor=midspan put it at Z ∈
            # [-500, 500], and a +90° rotation about X maps +Z → -Y).
            _, ymin, _, _, ymax, _ = _bbox_of_inst(inst)
            assert abs(ymin - (-500.0)) <= _TOL
            assert abs(ymax - 500.0) <= _TOL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
