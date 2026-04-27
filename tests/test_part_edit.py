"""
Tests for ``Part.edit`` — whole-Part transforms, copy, patterns, align.

Pins the public kwarg surface so future renames (``label=``, ``anchor=``,
etc.) get caught loudly instead of slipping through.
"""
from __future__ import annotations

import math
from pathlib import Path

import gmsh
import pytest

from apeGmsh import Part, apeGmsh
from apeGmsh.sections import W_solid


# =====================================================================
# Foundation transforms — translate / rotate / mirror / scale / dilate / affine
# =====================================================================

class TestFoundationTransforms:
    def test_translate_moves_geometry(self):
        p = Part("translate_test")
        with p:
            p.model.geometry.add_box(0, 0, 0, 100, 100, 100)
            p.edit.translate(50, 0, 0)
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[0] == pytest.approx(50, abs=1e-3)
            assert bb[3] == pytest.approx(150, abs=1e-3)

    def test_translate_returns_self_for_chaining(self):
        p = Part("translate_chain")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            ret = p.edit.translate(0, 0, 0)
            # Returns the PartEdit composite, not the Part
            assert ret is p.edit

    def test_translate_zero_is_noop(self):
        p = Part("translate_zero")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            ret = p.edit.translate(0, 0, 0)
            assert ret is p.edit  # still chainable

    def test_rotate_about_origin(self):
        p = Part("rotate_test")
        with p:
            p.model.geometry.add_box(100, 0, 0, 50, 50, 50)
            # Rotate 180° about Z — box should land at (-150,-50,0) — (-100,0,50)
            p.edit.rotate(math.pi, 0, 0, 1)
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[0] == pytest.approx(-150, abs=1e-3)
            assert bb[3] == pytest.approx(-100, abs=1e-3)

    def test_rotate_about_custom_center(self):
        p = Part("rotate_center")
        with p:
            p.model.geometry.add_box(0, 0, 0, 10, 10, 10)
            # Rotate 180° about Z through (10, 10, 0)
            p.edit.rotate(math.pi, 0, 0, 1, center=(10, 10, 0))
            bb = gmsh.model.getBoundingBox(-1, -1)
            # Box was at (0..10, 0..10, 0..10), now at (10..20, 10..20, 0..10)
            assert bb[0] == pytest.approx(10, abs=1e-3)
            assert bb[3] == pytest.approx(20, abs=1e-3)

    def test_mirror_named_plane(self):
        p = Part("mirror_named")
        with p:
            p.model.geometry.add_box(10, 20, 30, 5, 5, 5)
            p.edit.mirror(plane="xz")  # y → -y
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[1] == pytest.approx(-25, abs=1e-3)
            assert bb[4] == pytest.approx(-20, abs=1e-3)

    def test_mirror_explicit_normal(self):
        p = Part("mirror_normal")
        with p:
            p.model.geometry.add_box(10, 0, 0, 5, 5, 5)
            # Reflect about plane with normal +X through origin: x → -x
            p.edit.mirror(normal=(1, 0, 0))
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[0] == pytest.approx(-15, abs=1e-3)
            assert bb[3] == pytest.approx(-10, abs=1e-3)

    def test_mirror_requires_one_of_plane_or_normal(self):
        p = Part("mirror_err")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            with pytest.raises(ValueError, match="exactly one"):
                p.edit.mirror()
            with pytest.raises(ValueError, match="exactly one"):
                p.edit.mirror(plane="xy", normal=(1, 0, 0))

    def test_mirror_unknown_plane_name(self):
        p = Part("mirror_bad")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            with pytest.raises(ValueError, match="plane must be"):
                p.edit.mirror(plane="zw")

    def test_scale_uniform(self):
        p = Part("scale_test")
        with p:
            p.model.geometry.add_box(0, 0, 0, 100, 50, 200)
            p.edit.scale(0.5)
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[3] == pytest.approx(50, abs=1e-3)
            assert bb[4] == pytest.approx(25, abs=1e-3)
            assert bb[5] == pytest.approx(100, abs=1e-3)

    def test_dilate_non_uniform(self):
        p = Part("dilate_test")
        with p:
            p.model.geometry.add_box(0, 0, 0, 100, 100, 100)
            p.edit.dilate(2.0, 1.0, 0.5)
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[3] == pytest.approx(200, abs=1e-3)
            assert bb[4] == pytest.approx(100, abs=1e-3)
            assert bb[5] == pytest.approx(50, abs=1e-3)

    def test_affine_identity_is_noop(self):
        p = Part("affine_id")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            bb_before = gmsh.model.getBoundingBox(-1, -1)
            p.edit.affine([1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, 1])
            bb_after = gmsh.model.getBoundingBox(-1, -1)
            assert bb_before == bb_after

    def test_affine_accepts_nested_4x4(self):
        p = Part("affine_nested")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            p.edit.affine([
                [1, 0, 0, 10],
                [0, 1, 0, 20],
                [0, 0, 1, 30],
                [0, 0, 0,  1],
            ])
            bb = gmsh.model.getBoundingBox(-1, -1)
            assert bb[0] == pytest.approx(10, abs=1e-3)
            assert bb[1] == pytest.approx(20, abs=1e-3)
            assert bb[2] == pytest.approx(30, abs=1e-3)

    def test_affine_wrong_size_raises(self):
        p = Part("affine_bad")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            with pytest.raises(ValueError, match="4x4"):
                p.edit.affine([1, 0, 0])

    def test_chaining_returns_part_edit(self):
        p = Part("chain_test")
        with p:
            p.model.geometry.add_box(0, 0, 0, 100, 50, 200)
            ret = (
                p.edit
                .translate(10, 20, 30)
                .rotate(math.pi / 2, 0, 0, 1)
                .scale(0.5)
                .mirror(plane="xz")
            )
            assert ret is p.edit


# =====================================================================
# Active-session checks
# =====================================================================

class TestActiveSessionGuards:
    @pytest.mark.parametrize("op_name, op_call", [
        ("translate", lambda e: e.translate(0, 0, 1)),
        ("rotate",    lambda e: e.rotate(0.1, 0, 0, 1)),
        ("mirror",    lambda e: e.mirror(plane="xy")),
        ("scale",     lambda e: e.scale(2.0)),
        ("dilate",    lambda e: e.dilate(2, 1, 1)),
        ("affine",    lambda e: e.affine([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])),
        ("delete",    lambda e: e.delete()),
    ])
    def test_active_required_for_inplace_ops(self, op_name, op_call):
        # Build a Part and let auto-persist exit the session
        p = Part(f"guard_{op_name}")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # Now p is inactive — every transform must reject
        with pytest.raises(RuntimeError, match="active session"):
            op_call(p.edit)


# =====================================================================
# Delete
# =====================================================================

class TestDelete:
    def test_delete_clears_all_entities(self):
        p = Part("delete_all")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            p.model.geometry.add_box(2, 0, 0, 1, 1, 1)
            assert len(gmsh.model.getEntities(3)) == 2
            p.edit.delete()
            assert gmsh.model.getEntities() == []

    def test_delete_returns_none(self):
        p = Part("delete_ret")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            assert p.edit.delete() is None


# =====================================================================
# Copy
# =====================================================================

class TestCopy:
    def test_copy_inactive_source(self):
        # Source has been auto-persisted
        p = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            dup = p.edit.copy(label="my_dup")
            assert dup.has_file
            assert dup.file_path != p.file_path
            assert dup.name == "my_dup"
            assert dup.has_file
        finally:
            p.cleanup()
            dup.cleanup()

    def test_copy_active_source(self):
        # gmsh.write fork
        p = Part("active_src")
        with p:
            p.model.geometry.add_box(0, 0, 0, 50, 50, 50, label="body")
            dup = p.edit.copy(label="active_dup")
            assert dup.has_file
            # Source's file_path stayed None inside the with block
            assert p.file_path is None
        try:
            # After exit, dup is still loadable into an assembly
            with apeGmsh(model_name="t", verbose=False) as g:
                g.parts.add(dup)
                assert len(gmsh.model.getEntities(3)) == 1
        finally:
            dup.cleanup()
            p.cleanup()

    def test_copy_requires_label(self):
        p = Part("nolabel")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            with pytest.raises(ValueError, match="non-empty"):
                p.edit.copy(label="")
            with pytest.raises(TypeError):
                p.edit.copy()  # missing required keyword

    def test_copy_label_clash_gets_random_suffix(self):
        # Hold a live Part with a known name to force a clash
        p = Part("name_taken")
        # name registered with PartEdit's process-local table
        try:
            # Build a different source we'll copy from
            src = Part("dup_source")
            with src:
                src.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            # First copy gets the requested name (no conflict yet because
            # we used it for `p` only, not for any copy)
            with pytest.warns(UserWarning, match="already in use"):
                dup = src.edit.copy(label="name_taken")
            assert dup.name.startswith("name_taken_")
            assert dup.name != "name_taken"
        finally:
            try: src.cleanup()
            except: pass
            try: dup.cleanup()
            except: pass


# =====================================================================
# Patterns
# =====================================================================

class TestPatterns:
    def test_pattern_linear_creates_n_copies(self):
        src = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            copies = src.edit.pattern_linear(label="row", n=3, dx=2000, dy=0, dz=0)
            assert len(copies) == 3
            assert [c.name for c in copies] == ["row_1", "row_2", "row_3"]
            # First copy must be at +2000 X (verified via assembly bbox)
            with apeGmsh(model_name="t", verbose=False) as g:
                inst = g.parts.add(copies[0])
                assert inst.bbox is not None
                assert inst.bbox[0] == pytest.approx(2000 - 60, abs=1.0)
        finally:
            src.cleanup()
            for c in copies:
                c.cleanup()

    def test_pattern_polar_creates_n_copies(self):
        src = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            copies = src.edit.pattern_polar(
                label="ring", n=4, axis=(0, 0, 1), total_angle=2 * math.pi,
            )
            assert len(copies) == 4
            assert all(c.name.startswith("ring_") for c in copies)
        finally:
            src.cleanup()
            for c in copies:
                c.cleanup()

    def test_pattern_requires_inactive_source(self):
        p = Part("pattern_active")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            with pytest.raises(RuntimeError, match="CLOSED"):
                p.edit.pattern_linear(label="x", n=2, dx=1, dy=0, dz=0)
            with pytest.raises(RuntimeError, match="CLOSED"):
                p.edit.pattern_polar(label="r", n=2, axis=(0,0,1), total_angle=math.pi)

    def test_pattern_n_must_be_positive(self):
        src = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with pytest.raises(ValueError, match="positive"):
                src.edit.pattern_linear(label="x", n=0, dx=1, dy=0, dz=0)
            with pytest.raises(ValueError, match="positive"):
                src.edit.pattern_linear(label="x", n=-2, dx=1, dy=0, dz=0)
        finally:
            src.cleanup()


# =====================================================================
# Alignment
# =====================================================================

class TestAlign:
    def test_align_to_point_all_axes(self):
        p = Part("a2p_all")
        with p:
            p.model.geometry.add_box(0, 0, 0, 100, 100, 100, label="body")
            p.edit.align_to_point((1000, 2000, 3000), source="body", on="all")
            com = gmsh.model.occ.getCenterOfMass(3, 1)
            assert com[0] == pytest.approx(1000, abs=1e-6)
            assert com[1] == pytest.approx(2000, abs=1e-6)
            assert com[2] == pytest.approx(3000, abs=1e-6)

    def test_align_to_point_single_axis(self):
        p = Part("a2p_single")
        with p:
            p.model.geometry.add_box(0, 0, 0, 100, 100, 100, label="body")
            p.edit.align_to_point((999, 999, 500), source="body", on="z")
            com = gmsh.model.occ.getCenterOfMass(3, 1)
            # X, Y left at original 50 (body centroid); Z aligned to 500
            assert com[0] == pytest.approx(50, abs=1e-6)
            assert com[1] == pytest.approx(50, abs=1e-6)
            assert com[2] == pytest.approx(500, abs=1e-6)

    def test_align_to_point_with_offset(self):
        p = Part("a2p_offset")
        with p:
            p.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="body")
            # Body centroid at (5,5,5). Align Z to 100 with +25 offset → 125
            p.edit.align_to_point((0, 0, 100), source="body", on="z", offset=25)
            com = gmsh.model.occ.getCenterOfMass(3, 1)
            assert com[2] == pytest.approx(125, abs=1e-6)

    def test_align_to_uses_sidecar(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            # Build a small mover, align to beam's top_flange on Z
            mover = Part("mover")
            with mover:
                mover.model.geometry.add_box(-5, -5, -5, 10, 10, 10, label="anchor")
                mover.edit.align_to(
                    beam, source="anchor", target="top_flange", on="z", offset=5,
                )
                com = gmsh.model.occ.getCenterOfMass(3, 1)
                # top_flange Z centroid is at length midspan = 500; +5 offset
                assert com[2] == pytest.approx(505, abs=1.0)
        finally:
            beam.cleanup()
            mover.cleanup()

    def test_align_to_rejects_offset_with_multi_axis(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            mover = Part("mover_off")
            with mover:
                mover.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
                with pytest.raises(ValueError, match="single-axis"):
                    mover.edit.align_to(
                        beam, source="a", target="top_flange",
                        on=("x", "z"), offset=10,
                    )
        finally:
            beam.cleanup()

    def test_align_to_rejects_non_part(self):
        p = Part("rj_np")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
            with pytest.raises(TypeError, match="Part"):
                p.edit.align_to("not_a_part", source="a", target="x", on="z")

    def test_align_to_rejects_self(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            # Need to be inside a with block to test, but `beam` is closed.
            # Open a fresh Part to test on.
            with beam:
                with pytest.raises(ValueError, match="different Part"):
                    beam.edit.align_to(
                        beam, source="top_flange", target="bottom_flange", on="z",
                    )
        finally:
            beam.cleanup()

    def test_align_to_rejects_unsaved_target(self):
        target = Part("unsaved_target")  # never saved
        mover = Part("mover_us")
        with mover:
            mover.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
            with pytest.raises(RuntimeError, match="saved"):
                mover.edit.align_to(target, source="a", target="x", on="z")

    def test_align_axis_validation(self):
        p = Part("align_axes")
        with p:
            p.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
            with pytest.raises(ValueError, match="axis"):
                p.edit.align_to_point((0, 0, 0), source="a", on="w")
            with pytest.raises(ValueError, match="unknown axis"):
                p.edit.align_to_point((0, 0, 0), source="a", on=("x", "q"))
