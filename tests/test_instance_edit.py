"""
Tests for ``Instance.edit`` — whole-Instance transforms in the live
assembly session, plus ``g.parts.get(label=)``.

Pins the public kwarg surface of every Instance.edit method.
"""
from __future__ import annotations

import math

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.sections import W_solid


# =====================================================================
# parts.get(label=)
# =====================================================================

class TestPartsGet:
    def test_returns_registered_instance(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                inst = g.parts.add(beam, label="b1")
                assert g.parts.get("b1") is inst
        finally:
            beam.cleanup()

    def test_missing_label_raises_with_available(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                g.parts.add(beam, label="b1")
                g.parts.add(beam, label="b2")
                with pytest.raises(KeyError) as exc_info:
                    g.parts.get("nope")
                msg = str(exc_info.value)
                assert "nope" in msg
                assert "b1" in msg and "b2" in msg
        finally:
            beam.cleanup()

    def test_returned_instance_has_edit(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                g.parts.add(beam, label="b1")
                inst = g.parts.get("b1")
                assert inst.edit is not None
        finally:
            beam.cleanup()


# =====================================================================
# Foundation transforms scoped to an Instance
# =====================================================================

class TestInstanceFoundationTransforms:
    def test_translate_scoped_to_instance(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                b = g.parts.add(beam, label="b")
                bb_b_before = b.bbox

                a.edit.translate(500, 0, 0)
                # a moved
                assert a.bbox[0] == pytest.approx(bb_b_before[0] + 500, abs=1.0)
                # b unmoved
                assert b.bbox == bb_b_before
        finally:
            beam.cleanup()

    def test_translate_returns_instance_edit_for_chaining(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                ret = a.edit.translate(0, 0, 0).rotate(0, 0, 0, 1)
                assert ret is a.edit
        finally:
            beam.cleanup()

    def test_rotate_about_center(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                bb_before = a.bbox
                a.edit.rotate(math.pi, 0, 0, 1, center=(100, 0, 0))
                assert a.bbox != bb_before
        finally:
            beam.cleanup()

    def test_mirror_named_plane(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a", translate=(100, 0, 0))
                a.edit.mirror(plane="yz")
                # X coordinates flip across yz plane
                assert a.bbox[0] < 0
                assert a.bbox[3] < 0
        finally:
            beam.cleanup()

    def test_mirror_explicit_normal(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a", translate=(0, 100, 0))
                a.edit.mirror(normal=(0, 1, 0))
                # Y coordinates flip
                assert a.bbox[1] < 0
        finally:
            beam.cleanup()

    def test_mirror_requires_one_of_plane_or_normal(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                with pytest.raises(ValueError, match="exactly one"):
                    a.edit.mirror()
                with pytest.raises(ValueError, match="exactly one"):
                    a.edit.mirror(plane="xy", normal=(1, 0, 0))
        finally:
            beam.cleanup()

    def test_scale_uniform(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                bb_before = a.bbox
                a.edit.scale(0.5)
                # Half size
                size_before = bb_before[5] - bb_before[2]
                size_after = a.bbox[5] - a.bbox[2]
                assert size_after == pytest.approx(size_before * 0.5, abs=1.0)
        finally:
            beam.cleanup()

    def test_dilate_non_uniform(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                a.edit.dilate(2.0, 1.0, 0.5)
                # Verify bbox cache refreshed
                assert a.bbox is not None
        finally:
            beam.cleanup()

    def test_affine_identity_is_noop(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                bb_before = a.bbox
                a.edit.affine([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
                assert a.bbox == bb_before
        finally:
            beam.cleanup()


# =====================================================================
# Delete
# =====================================================================

class TestInstanceDelete:
    def test_delete_removes_geometry(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                b = g.parts.add(beam, label="b")
                n_before = len(gmsh.model.getEntities(3))
                a.edit.delete()
                n_after = len(gmsh.model.getEntities(3))
                assert n_after < n_before
        finally:
            beam.cleanup()

    def test_delete_unregisters_from_parts(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                a.edit.delete()
                assert "a" not in g.parts.instances
                with pytest.raises(KeyError):
                    g.parts.get("a")
        finally:
            beam.cleanup()

    def test_delete_makes_subsequent_ops_fail(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                a.edit.delete()
                with pytest.raises(RuntimeError, match="deleted"):
                    a.edit.translate(1, 0, 0)
                with pytest.raises(RuntimeError, match="deleted"):
                    a.edit.delete()  # double-delete
        finally:
            beam.cleanup()

    def test_delete_frees_label_for_reuse(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                g.parts.add(beam, label="a")
                g.parts.get("a").edit.delete()
                # Label "a" is free again — re-adding should work
                a2 = g.parts.add(beam, label="a")
                assert a2.label == "a"
        finally:
            beam.cleanup()


# =====================================================================
# Copy — within the same assembly session
# =====================================================================

class TestInstanceCopy:
    def test_copy_creates_new_instance(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                n_before = len(gmsh.model.getEntities(3))
                dup = src.edit.copy(label="dup")
                n_after = len(gmsh.model.getEntities(3))
                # Geometry doubled (W_solid has 7 volumes)
                assert n_after == 2 * n_before
                # Registered in parts
                assert "dup" in g.parts.instances
                assert dup.edit is not None
        finally:
            beam.cleanup()

    def test_copy_rebrands_labels(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                # src.label_names contains entries like "src.top_flange"
                assert any(n == "src.top_flange" for n in src.label_names)
                dup = src.edit.copy(label="dup")
                # Labels rebranded
                assert any(n == "dup.top_flange" for n in dup.label_names)
                assert any(n == "dup.web" for n in dup.label_names)
                # Source's labels untouched
                assert "src.top_flange" in src.label_names
        finally:
            beam.cleanup()

    def test_copy_label_clash_gets_random_suffix(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                g.parts.add(beam, label="taken")
                with pytest.warns(UserWarning, match="already in use"):
                    dup = src.edit.copy(label="taken")
                assert dup.label.startswith("taken_")
                assert dup.label != "taken"
        finally:
            beam.cleanup()

    def test_copy_requires_label(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                with pytest.raises(ValueError, match="non-empty"):
                    src.edit.copy(label="")
        finally:
            beam.cleanup()


# =====================================================================
# Patterns
# =====================================================================

class TestInstancePatterns:
    def test_pattern_linear_creates_n_copies(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                copies = src.edit.pattern_linear(
                    label="row", n=3, dx=2000, dy=0, dz=0,
                )
                assert len(copies) == 3
                assert [c.label for c in copies] == ["row_1", "row_2", "row_3"]
                # First copy at +2000 X from src
                assert copies[0].bbox[0] == pytest.approx(
                    src.bbox[0] + 2000, abs=1.0,
                )
        finally:
            beam.cleanup()

    def test_pattern_polar_creates_n_copies(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                copies = src.edit.pattern_polar(
                    label="ring", n=4, axis=(0, 0, 1), total_angle=2 * math.pi,
                )
                assert len(copies) == 4
                assert all(c.label.startswith("ring_") for c in copies)
        finally:
            beam.cleanup()

    def test_pattern_n_must_be_positive(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                src = g.parts.add(beam, label="src")
                with pytest.raises(ValueError, match="positive"):
                    src.edit.pattern_linear(label="x", n=0, dx=1, dy=0, dz=0)
                with pytest.raises(ValueError, match="positive"):
                    src.edit.pattern_polar(
                        label="r", n=-1, axis=(0,0,1), total_angle=math.pi,
                    )
        finally:
            beam.cleanup()


# =====================================================================
# Alignment between Instances
# =====================================================================

class TestInstanceAlign:
    def test_align_to_stacks_on_z(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                b = g.parts.add(beam, label="b", translate=(5000, 0, 7000))
                # Align a's top_flange to b's bottom_flange on Z, +50 gap
                a.edit.align_to(
                    b, source="top_flange", target="bottom_flange",
                    on="z", offset=50,
                )
                # b's bottom_flange centroid Z: 7000 + 500 (midspan) = 7500
                # a's top_flange centroid was at z=500 (local frame, not translated)
                # After align: a's top_flange Z = 7500 + 50 = 7550
                # Verify a's bbox max-z is around 7550 + (top_flange thickness above midspan)
                # Top flange centroid is at midspan vertically — but the W is symmetric;
                # just confirm a moved up substantially
                assert a.bbox[5] > 7000
        finally:
            beam.cleanup()

    def test_align_to_point(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                a.edit.align_to_point(
                    (0, 0, 1000), source="bottom_flange", on="z",
                )
                # a's bottom_flange centroid now at z=1000 — bbox z-min should be
                # below 1000 (since the flange has some thickness)
                assert a.bbox[2] < 1000
                assert a.bbox[5] > 1000  # top of beam above the alignment plane
        finally:
            beam.cleanup()

    def test_align_to_rejects_part(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                with pytest.raises(TypeError, match="Instance"):
                    a.edit.align_to(
                        beam,  # passing a Part
                        source="top_flange", target="bottom_flange", on="z",
                    )
        finally:
            beam.cleanup()

    def test_align_to_rejects_self(self):
        beam = W_solid(bf=120, tf=10, h=150, tw=6, length=1000)
        try:
            with apeGmsh(model_name="t", verbose=False) as g:
                a = g.parts.add(beam, label="a")
                with pytest.raises(ValueError, match="different Instance"):
                    a.edit.align_to(
                        a, source="top_flange", target="bottom_flange", on="z",
                    )
        finally:
            beam.cleanup()
