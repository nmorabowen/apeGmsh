"""End-to-end tests for ``g.parts.add_DRM_box`` and :class:`DRMBox`.

These tests exercise the full geometry → STEP → import → PG-tagging
→ transfinite cascade path against a live OCC kernel.  They are
deliberately not OpenSees-aware — the DRM-box primitive is pure
geometry / mesh setup.
"""
from __future__ import annotations

import math

import gmsh
import pytest

from apeGmsh import DRMBox, DRMBoxResult, apeGmsh


# A small, mesh-friendly default that survives 8 distinct test
# variants — total of 8x8x6 = 384 hex elements per build.  Replaces
# the user's 16x16x30 = 7680-element notebook in the test suite
# (which is verified separately in ``test_user_notebook_reproduction``).
TINY = dict(
    x_inner=(50.0, 4),
    x_layer=(10.0, 1),
    x_outer=(20.0, 1),
    y_inner=(50.0, 4),
    y_layer=(10.0, 1),
    y_outer=(20.0, 1),
    z_top=(20.0, 2),
    z_mid=(20.0, 2),
    z_bottom=(40.0, 2),
)
# Per-axis element totals: X = 1+1+4+1+1 = 8, Y = 8, Z = 2+2+2 = 6.
TINY_TOTAL_HEX = 8 * 8 * 6   # 384


def _element_counts(dim: int) -> dict[str, int]:
    etypes, etags, _ = gmsh.model.mesh.getElements(dim=dim)
    out: dict[str, int] = {}
    for et, tags in zip(etypes, etags):
        name, *_ = gmsh.model.mesh.getElementProperties(et)
        out[name] = out.get(name, 0) + len(tags)
    return out


class TestHappyPath:
    def test_smoke_default_pg_names(self):
        g = apeGmsh(model_name="drm_smoke", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(**TINY)
            assert isinstance(res, DRMBoxResult)
            assert res.inner_pg == "inner_box"
            assert res.transition_pg == "transition_box"
            assert res.outer_pg == "outer_box"
            assert res.center == (0.0, 0.0, 0.0)
            assert res.rotation_z == 0.0
            assert set(res.axes) == {"x", "y", "z"}
            # 5x5x3 = 75 sub-volumes on a symmetric layered box
            inst = g.parts.get("drm_box")
            assert len(inst.entities[3]) == 75
        finally:
            g.end()

    def test_mesh_total_hex_count_matches_per_region(self):
        g = apeGmsh(model_name="drm_mesh", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box(**TINY)
            g.mesh.generation.generate(dim=3)
            counts = _element_counts(3)
            assert "Hexahedron 8" in counts
            assert counts["Hexahedron 8"] == TINY_TOTAL_HEX
            # No tets — the structured cascade should hex-mesh
            # every sub-volume.
            assert counts.get("Tetrahedron 4", 0) == 0
        finally:
            g.end()

    def test_volume_pg_partition_is_complete(self):
        g = apeGmsh(model_name="drm_pgs", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(**TINY)
            inst = g.parts.get("drm_box")
            all_vols = set(inst.entities[3])
            inner = set(g.physical.get_entities(
                3, g.physical.get_tag(3, res.inner_pg)
            ))
            trans = set(g.physical.get_entities(
                3, g.physical.get_tag(3, res.transition_pg)
            ))
            outer = set(g.physical.get_entities(
                3, g.physical.get_tag(3, res.outer_pg)
            ))
            assert inner.isdisjoint(trans)
            assert inner.isdisjoint(outer)
            assert trans.isdisjoint(outer)
            assert inner | trans | outer == all_vols
            # Symmetric layered geometry: the inner column is one
            # cell laterally x 3 z layers = 3 sub-volumes.
            assert len(inner) == 3
            # Lateral wrap classes — see comment in ``add_DRM_box``:
            # per z layer there is 1 inner + 8 transition + 16 outer
            # sub-volumes; x 3 z layers => 3, 24, 48.
            assert len(trans) == 24
            assert len(outer) == 48
        finally:
            g.end()


class TestRotation:
    def test_rotation_preserves_element_count(self):
        g = apeGmsh(model_name="drm_rot", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(**TINY, rotation_z_deg=30.0)
            assert res.rotation_z == pytest.approx(math.radians(30.0))
            g.mesh.generation.generate(dim=3)
            counts = _element_counts(3)
            assert counts.get("Tetrahedron 4", 0) == 0
            assert counts["Hexahedron 8"] == TINY_TOTAL_HEX
        finally:
            g.end()

    def test_rotation_rotates_outer_bbox(self):
        g = apeGmsh(model_name="drm_rot_bb", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box(
                **TINY, rotation_z_deg=30.0,
                apply_transfinite=False, tag_line_pgs=False,
            )
            inst = g.parts.get("drm_box")
            # The instance's umbrella bbox expands when rotated.
            # X-extent of the unrotated outer-box is
            # 2 * (50/2 + 10 + 20) = 110.  After a 30° rotation it
            # should be strictly larger.
            assert inst.bbox is not None
            xmin, ymin, _zmin, xmax, ymax, _zmax = inst.bbox
            assert (xmax - xmin) > 110.0
            assert (ymax - ymin) > 110.0
        finally:
            g.end()


class TestTranslation:
    def test_center_shifts_bbox(self):
        g = apeGmsh(model_name="drm_trans", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box(
                **TINY, center=(100.0, 200.0, 50.0),
                apply_transfinite=False, tag_line_pgs=False,
            )
            inst = g.parts.get("drm_box")
            assert inst.bbox is not None
            xmin, ymin, zmin, xmax, ymax, zmax = inst.bbox
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            # The DRM-box's lateral centroid sits at the translation
            # ``center`` xy (the inner box is centred laterally).
            assert cx == pytest.approx(100.0, abs=1e-6)
            assert cy == pytest.approx(200.0, abs=1e-6)
            # The Z stack descends from ``cz``: free surface at
            # cz = 50; bottom at cz - (20 + 20 + 40) = -30.
            assert zmax == pytest.approx(50.0, abs=1e-6)
            assert zmin == pytest.approx(-30.0, abs=1e-6)
        finally:
            g.end()


class TestNamingOverrides:
    def test_name_prefix(self):
        g = apeGmsh(model_name="drm_naming", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(**TINY, name="alice")
            assert res.inner_pg == "alice_inner_box"
            assert res.transition_pg == "alice_transition_box"
            assert res.outer_pg == "alice_outer_box"
            # Line-PG names also gain the prefix.
            assert res.line_pgs["inner_x"] == "alice_lines_inner_x"
        finally:
            g.end()

    def test_per_pg_override(self):
        g = apeGmsh(model_name="drm_override", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(
                **TINY,
                names={
                    "inner_pg": "core",
                    "line_pg_top_z": "top_curves",
                },
            )
            # ``inner_pg`` was overridden directly.
            assert res.inner_pg == "core"
            # Others keep the unprefixed default.
            assert res.transition_pg == "transition_box"
            assert res.outer_pg == "outer_box"
            assert res.line_pgs["top_z"] == "top_curves"
            # Non-overridden line-PG keeps the default.
            assert res.line_pgs["inner_x"] == "lines_inner_x"
        finally:
            g.end()


class TestToggles:
    def test_tag_line_pgs_false_yields_empty_line_pgs(self):
        g = apeGmsh(model_name="drm_no_lines", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(**TINY, tag_line_pgs=False)
            assert res.line_pgs == {}
            # And no ``lines_*`` PGs exist in the model.
            for _d, pg_tag in gmsh.model.getPhysicalGroups(dim=1):
                name = gmsh.model.getPhysicalName(1, pg_tag)
                assert not name.startswith("lines_")
        finally:
            g.end()

    def test_tag_line_pgs_true_creates_curve_pgs(self):
        g = apeGmsh(model_name="drm_lines", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box(**TINY)
            # All 9 expected (region, axis) groups should be present
            # (each is non-empty on this geometry).
            expected = {
                "inner_x", "layer_x", "outer_x",
                "inner_y", "layer_y", "outer_y",
                "top_z", "mid_z", "bottom_z",
            }
            assert set(res.line_pgs) == expected
            # Each PG resolves to at least one curve.
            for key, name in res.line_pgs.items():
                pg_tag = g.physical.get_tag(1, name)
                assert pg_tag is not None, f"line PG {key} missing"
                ents = g.physical.get_entities(1, pg_tag)
                assert len(ents) > 0, f"line PG {key} empty"
        finally:
            g.end()

    def test_apply_transfinite_false_skips_directives(self):
        g = apeGmsh(model_name="drm_no_tf", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box(**TINY, apply_transfinite=False)
            # Mesh should still generate, but without the strict
            # 8x8x6 = 384 hex count the transfinite cascade
            # produces.
            g.mesh.generation.generate(dim=3)
            counts = _element_counts(3)
            total = sum(counts.values())
            # Unstructured tet meshes on this geometry produce far
            # more elements than the structured 384.
            assert total != TINY_TOTAL_HEX
        finally:
            g.end()


class TestUserNotebookReproduction:
    """Pin the user's hand-written notebook example to a single call."""

    PARAMS = dict(
        x_inner=(605.0, 10),
        x_layer=(10.0, 1),
        x_outer=(20.0, 2),
        y_inner=(605.0, 10),
        y_layer=(10.0, 1),
        y_outer=(20.0, 2),
        z_top=(50.0, 5),
        z_mid=(50.0, 5),
        z_bottom=(200.0, 20),
    )
    # Per-axis element totals: X = 2+1+10+1+2 = 16, Y = 16, Z = 30.
    EXPECTED_HEX = 16 * 16 * 30

    def test_total_hex_count(self):
        g = apeGmsh(model_name="drm_notebook", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box(**self.PARAMS)
            g.mesh.generation.generate(dim=3)
            counts = _element_counts(3)
            assert counts.get("Tetrahedron 4", 0) == 0
            assert counts["Hexahedron 8"] == self.EXPECTED_HEX
        finally:
            g.end()


class TestDRMBoxStandalone:
    """The :class:`DRMBox` Part should also build cleanly on its own."""

    def test_part_session_has_75_volumes(self):
        drm = DRMBox(**TINY)
        with drm:
            drm.build()
            n_vols = len(gmsh.model.getEntities(3))
        try:
            assert n_vols == 75
            assert drm.has_file
        finally:
            drm.cleanup()
