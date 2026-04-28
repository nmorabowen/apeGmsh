"""Phase 11c Step 1 — fiber / layer / section_deformation catalogs.

Pure unit tests on ``apeGmsh.solvers._element_response``. No OpenSees,
no h5py, no MPCO files. Verifies:

- Vocabulary additions (``LINE_STATION_DEFORMATIONS`` token set,
  ``section_deformation`` shorthand).
- ``CUSTOM_RULE_CATALOG`` doubled with ``section_deformation`` entries
  for every line-station class.
- New ``FiberSectionLayout`` / ``LayeredShellLayout`` classes and the
  matching ``FIBER_CATALOG`` / ``LAYER_CATALOG`` dicts.
- Routing helpers extended with ``"fibers"`` and ``"layers"``
  topologies, plus the keyword swap (beams →
  ``section.fiber.stress``; shells → ``material.fiber.stress``).
- ``resolve_layout_from_gp_x(..., kind="deformation")`` produces the
  conjugate canonical names from the same OpenSees code vector.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results._vocabulary import (
    LINE_DIAGRAMS,
    LINE_STATION_DEFORMATIONS,
    expand_shorthand,
    is_canonical,
    is_shorthand,
)
from apeGmsh.solvers._element_response import (
    CUSTOM_RULE_CATALOG,
    ELE_TAG_ASDShellQ4,
    ELE_TAG_ASDShellT3,
    ELE_TAG_DispBeamColumn3d,
    ELE_TAG_ForceBeamColumn2d,
    ELE_TAG_ForceBeamColumn3d,
    ELE_TAG_ShellMITC4,
    ELE_TAG_ShellMITC9,
    FIBER_CATALOG,
    LAYER_CATALOG,
    SECTION_RESPONSE_DEFORMATION_TO_CANONICAL,
    SECTION_RESPONSE_TO_CANONICAL,
    CatalogLookupError,
    FiberSectionLayout,
    IntRule,
    LayeredShellLayout,
    catalog_token_for_keyword,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    is_fiber_catalogued,
    is_layer_catalogued,
    lookup_custom_rule,
    lookup_fiber,
    lookup_layer,
    resolve_layout_from_gp_x,
)


# =====================================================================
# Vocabulary
# =====================================================================

class TestLineStationDeformations:
    def test_six_components(self) -> None:
        assert len(LINE_STATION_DEFORMATIONS) == 6

    def test_components_match_line_diagram_pairing(self) -> None:
        # Conjugate work-pair order: positional pairing with LINE_DIAGRAMS.
        assert LINE_STATION_DEFORMATIONS == (
            "axial_strain",
            "shear_strain_y", "shear_strain_z",
            "torsional_strain",
            "curvature_y", "curvature_z",
        )
        assert len(LINE_DIAGRAMS) == len(LINE_STATION_DEFORMATIONS)

    def test_all_canonical(self) -> None:
        for name in LINE_STATION_DEFORMATIONS:
            assert is_canonical(name), f"{name} should be canonical"

    def test_section_deformation_shorthand_expands_to_six(self) -> None:
        assert is_shorthand("section_deformation")
        assert expand_shorthand("section_deformation") == LINE_STATION_DEFORMATIONS

    def test_section_force_shorthand_expands_to_diagrams(self) -> None:
        # Defensive: confirms the symmetric shorthand was added without
        # disturbing the existing one.
        assert expand_shorthand("section_force") == LINE_DIAGRAMS

    def test_shorthand_does_not_clip_by_ndm(self) -> None:
        # Section shorthands are not ndm-clipped (catalog declares
        # which subset each element class actually emits).
        assert expand_shorthand(
            "section_deformation", ndm=2, ndf=3,
        ) == LINE_STATION_DEFORMATIONS


# =====================================================================
# SECTION_RESPONSE_DEFORMATION_TO_CANONICAL
# =====================================================================

class TestSectionResponseDeformationMap:
    def test_keys_match_force_table(self) -> None:
        # Same OpenSees codes (1..6); only canonical names differ.
        assert (
            set(SECTION_RESPONSE_DEFORMATION_TO_CANONICAL.keys())
            == set(SECTION_RESPONSE_TO_CANONICAL.keys())
        )

    @pytest.mark.parametrize("code, expected", [
        (1, "curvature_z"),       # conjugate to bending_moment_z
        (2, "axial_strain"),      # conjugate to axial_force
        (3, "shear_strain_y"),    # conjugate to shear_y
        (4, "curvature_y"),       # conjugate to bending_moment_y
        (5, "shear_strain_z"),    # conjugate to shear_z
        (6, "torsional_strain"),  # conjugate to torsion
    ])
    def test_mapping(self, code: int, expected: str) -> None:
        assert SECTION_RESPONSE_DEFORMATION_TO_CANONICAL[code] == expected

    def test_values_are_canonical(self) -> None:
        for canonical in SECTION_RESPONSE_DEFORMATION_TO_CANONICAL.values():
            assert is_canonical(canonical)


# =====================================================================
# CUSTOM_RULE_CATALOG — section_deformation entries
# =====================================================================

LINE_STATION_CLASSES = (
    "ForceBeamColumn2d", "ForceBeamColumn3d",
    "ForceBeamColumnCBDI2d", "ForceBeamColumnWarping2d",
    "ElasticForceBeamColumn2d", "ElasticForceBeamColumn3d",
    "DispBeamColumn2d", "DispBeamColumn3d",
)


class TestSectionDeformationEntries:
    @pytest.mark.parametrize("class_name", LINE_STATION_CLASSES)
    def test_deformation_entry_present(self, class_name: str) -> None:
        assert (class_name, "section_deformation") in CUSTOM_RULE_CATALOG

    @pytest.mark.parametrize("class_name", LINE_STATION_CLASSES)
    def test_force_and_deformation_share_same_class_tag(
        self, class_name: str,
    ) -> None:
        force = CUSTOM_RULE_CATALOG[(class_name, "section_force")]
        defo = CUSTOM_RULE_CATALOG[(class_name, "section_deformation")]
        assert force.class_tag == defo.class_tag

    def test_lookup_via_helper(self) -> None:
        layout = lookup_custom_rule("ForceBeamColumn3d", "section_deformation")
        assert layout.class_tag == ELE_TAG_ForceBeamColumn3d
        assert layout.coord_system == "isoparametric_1d"


# =====================================================================
# resolve_layout_from_gp_x — kind switching
# =====================================================================

class TestResolveLayoutKind:
    @pytest.fixture
    def custom_3d(self):
        return lookup_custom_rule("ForceBeamColumn3d", "section_deformation")

    def test_force_kind_default_unchanged(self) -> None:
        # Backward-compat: calling without kind= still hits the force table.
        custom = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        layout = resolve_layout_from_gp_x(
            custom, np.array([-1.0, 0.0, 1.0]), (2, 1, 4, 6),
        )
        assert layout.component_layout == (
            "axial_force", "bending_moment_z",
            "bending_moment_y", "torsion",
        )

    def test_deformation_kind_returns_conjugates(self, custom_3d) -> None:
        layout = resolve_layout_from_gp_x(
            custom_3d, np.array([-1.0, 0.0, 1.0]), (2, 1, 4, 6),
            kind="deformation",
        )
        assert layout.component_layout == (
            "axial_strain", "curvature_z",
            "curvature_y", "torsional_strain",
        )

    def test_full_3d_section_with_shears(self, custom_3d) -> None:
        # Six-component aggregate (P, Mz, My, T, Vy, Vz).
        layout = resolve_layout_from_gp_x(
            custom_3d, np.array([-0.5, 0.5]), (2, 1, 4, 6, 3, 5),
            kind="deformation",
        )
        assert layout.component_layout == (
            "axial_strain", "curvature_z", "curvature_y",
            "torsional_strain", "shear_strain_y", "shear_strain_z",
        )

    def test_invalid_kind_raises(self, custom_3d) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            resolve_layout_from_gp_x(
                custom_3d, np.array([0.0]), (2,),
                kind="bogus",
            )

    def test_natural_coords_carry_through(self, custom_3d) -> None:
        gp_x = np.array([-0.7745966, 0.0, 0.7745966])
        layout = resolve_layout_from_gp_x(
            custom_3d, gp_x, (2, 1), kind="deformation",
        )
        assert layout.n_gauss_points == 3
        np.testing.assert_array_almost_equal(
            layout.natural_coords.flatten(), gp_x,
        )


# =====================================================================
# FIBER_CATALOG
# =====================================================================

class TestFiberCatalog:
    def test_size_two_tokens_per_class(self) -> None:
        # 8 line-station beam-column classes × 2 tokens.
        assert len(FIBER_CATALOG) == len(LINE_STATION_CLASSES) * 2

    @pytest.mark.parametrize("class_name", LINE_STATION_CLASSES)
    @pytest.mark.parametrize("token", ["fiber_stress", "fiber_strain"])
    def test_entry_present(self, class_name: str, token: str) -> None:
        assert (class_name, token) in FIBER_CATALOG

    def test_entries_are_fiber_section_layout(self) -> None:
        for entry in FIBER_CATALOG.values():
            assert isinstance(entry, FiberSectionLayout)

    def test_all_isoparametric_1d(self) -> None:
        for entry in FIBER_CATALOG.values():
            assert entry.coord_system == "isoparametric_1d"

    def test_class_tags_match_line_station_constants(self) -> None:
        # FIBER and CUSTOM_RULE catalogs share class identity — same
        # eight beam-columns. Tags must match by class.
        for class_name in LINE_STATION_CLASSES:
            f = FIBER_CATALOG[(class_name, "fiber_stress")]
            c = CUSTOM_RULE_CATALOG[(class_name, "section_force")]
            assert f.class_tag == c.class_tag

    def test_lookup_helper_hit(self) -> None:
        layout = lookup_fiber("DispBeamColumn3d", "fiber_strain")
        assert layout.class_tag == ELE_TAG_DispBeamColumn3d

    def test_lookup_helper_miss_raises(self) -> None:
        with pytest.raises(CatalogLookupError, match="No FiberSectionLayout"):
            lookup_fiber("NotARealBeam", "fiber_stress")

    def test_is_fiber_catalogued(self) -> None:
        assert is_fiber_catalogued("ForceBeamColumn2d", "fiber_stress")
        assert not is_fiber_catalogued("ForceBeamColumn2d", "stress")
        assert not is_fiber_catalogued("ASDShellQ4", "fiber_stress")  # shell


# =====================================================================
# LAYER_CATALOG
# =====================================================================

EXPECTED_LAYER_CLASSES = (
    "ShellMITC4", "ShellDKGQ", "ShellNLDKGQ", "ASDShellQ4",
    "ShellMITC9",
    "ShellDKGT", "ShellNLDKGT", "ASDShellT3",
)


class TestLayerCatalog:
    def test_size(self) -> None:
        # 8 shell classes × 2 tokens.
        assert len(LAYER_CATALOG) == len(EXPECTED_LAYER_CLASSES) * 2

    @pytest.mark.parametrize("class_name", EXPECTED_LAYER_CLASSES)
    @pytest.mark.parametrize("token", ["fiber_stress", "fiber_strain"])
    def test_entry_present(self, class_name: str, token: str) -> None:
        assert (class_name, token) in LAYER_CATALOG

    def test_entries_are_layered_shell_layout(self) -> None:
        for entry in LAYER_CATALOG.values():
            assert isinstance(entry, LayeredShellLayout)

    @pytest.mark.parametrize("class_name, expected_tag", [
        ("ShellMITC4", ELE_TAG_ShellMITC4),
        ("ASDShellQ4", ELE_TAG_ASDShellQ4),
        ("ShellMITC9", ELE_TAG_ShellMITC9),
        ("ASDShellT3", ELE_TAG_ASDShellT3),
    ])
    def test_class_tags(self, class_name: str, expected_tag: int) -> None:
        assert LAYER_CATALOG[(class_name, "fiber_stress")].class_tag == expected_tag

    @pytest.mark.parametrize("class_name, expected_rule", [
        ("ShellMITC4",  IntRule.Quad_GL_2),
        ("ShellDKGQ",   IntRule.Quad_GL_2),
        ("ShellNLDKGQ", IntRule.Quad_GL_2),
        ("ASDShellQ4",  IntRule.Quad_GL_2),
        ("ShellMITC9",  IntRule.Quad_GL_3),
        ("ShellDKGT",   IntRule.Triangle_GL_2C),
        ("ShellNLDKGT", IntRule.Triangle_GL_2C),
        ("ASDShellT3",  IntRule.Triangle_GL_2B),
    ])
    def test_surface_int_rule_per_class(
        self, class_name: str, expected_rule: int,
    ) -> None:
        assert LAYER_CATALOG[
            (class_name, "fiber_stress")
        ].surface_int_rule == expected_rule

    @pytest.mark.parametrize("class_name, expected_cs", [
        ("ASDShellQ4",  "isoparametric"),
        ("ShellMITC4",  "isoparametric"),
        ("ShellMITC9",  "isoparametric"),
        ("ASDShellT3",  "barycentric_tri"),
        ("ShellDKGT",   "barycentric_tri"),
        ("ShellNLDKGT", "barycentric_tri"),
    ])
    def test_coord_system_per_class(
        self, class_name: str, expected_cs: str,
    ) -> None:
        assert LAYER_CATALOG[
            (class_name, "fiber_stress")
        ].coord_system == expected_cs

    def test_force_and_strain_share_layout_fields(self) -> None:
        # A class's two tokens point at structurally identical layouts
        # (only the readout differs).
        for class_name in EXPECTED_LAYER_CLASSES:
            stress = LAYER_CATALOG[(class_name, "fiber_stress")]
            strain = LAYER_CATALOG[(class_name, "fiber_strain")]
            assert stress.class_tag == strain.class_tag
            assert stress.surface_int_rule == strain.surface_int_rule
            assert stress.coord_system == strain.coord_system

    def test_lookup_helper_hit(self) -> None:
        layout = lookup_layer("ASDShellQ4", "fiber_stress")
        assert layout.class_tag == ELE_TAG_ASDShellQ4

    def test_lookup_helper_miss_raises(self) -> None:
        with pytest.raises(CatalogLookupError, match="No LayeredShellLayout"):
            lookup_layer("NotAShell", "fiber_stress")

    def test_is_layer_catalogued(self) -> None:
        assert is_layer_catalogued("ASDShellQ4", "fiber_strain")
        # Beam-column entries belong to FIBER_CATALOG, not LAYER_CATALOG.
        assert not is_layer_catalogued("ForceBeamColumn3d", "fiber_stress")


# =====================================================================
# Routing — fibers / layers topologies + keyword swap
# =====================================================================

class TestFiberTopologyRouting:
    def test_fiber_stress_routes_to_section_fiber_stress(self) -> None:
        assert (
            gauss_keyword_for_canonical("fiber_stress", topology="fibers")
            == "section.fiber.stress"
        )

    def test_fiber_strain_routes_to_section_fiber_strain(self) -> None:
        assert (
            gauss_keyword_for_canonical("fiber_strain", topology="fibers")
            == "section.fiber.strain"
        )

    def test_keyword_to_token(self) -> None:
        assert (
            catalog_token_for_keyword("section.fiber.stress", topology="fibers")
            == "fiber_stress"
        )
        assert (
            catalog_token_for_keyword("section.fiber.strain", topology="fibers")
            == "fiber_strain"
        )

    def test_full_routing(self) -> None:
        assert (
            gauss_routing_for_canonical("fiber_stress", topology="fibers")
            == ("section.fiber.stress", "fiber_stress")
        )
        assert (
            gauss_routing_for_canonical("fiber_strain", topology="fibers")
            == ("section.fiber.strain", "fiber_strain")
        )

    def test_non_fiber_canonical_returns_none(self) -> None:
        # Continuum stress doesn't live at the fibers topology.
        assert (
            gauss_routing_for_canonical("stress_xx", topology="fibers")
            is None
        )


class TestLayerTopologyRouting:
    def test_keyword_swap_for_shells(self) -> None:
        # Shells use ``material.fiber.*`` (the swapped keyword) on disk
        # — not ``section.fiber.*`` like beams.
        assert (
            gauss_keyword_for_canonical("fiber_stress", topology="layers")
            == "material.fiber.stress"
        )
        assert (
            gauss_keyword_for_canonical("fiber_strain", topology="layers")
            == "material.fiber.strain"
        )

    def test_full_routing(self) -> None:
        assert (
            gauss_routing_for_canonical("fiber_stress", topology="layers")
            == ("material.fiber.stress", "fiber_stress")
        )

    def test_layers_distinct_from_fibers(self) -> None:
        # Same canonical, different keyword in each topology.
        beam_keyword = gauss_keyword_for_canonical(
            "fiber_stress", topology="fibers",
        )
        shell_keyword = gauss_keyword_for_canonical(
            "fiber_stress", topology="layers",
        )
        assert beam_keyword != shell_keyword


class TestLineStationDeformationRouting:
    @pytest.mark.parametrize("canonical", [
        "axial_strain", "torsional_strain",
        "shear_strain_y", "shear_strain_z",
        "curvature_y", "curvature_z",
    ])
    def test_deformation_canonical_routes_to_section_deformation(
        self, canonical: str,
    ) -> None:
        assert (
            gauss_keyword_for_canonical(canonical, topology="line_stations")
            == "section.deformation"
        )

    def test_keyword_to_token(self) -> None:
        assert (
            catalog_token_for_keyword(
                "section.deformation", topology="line_stations",
            )
            == "section_deformation"
        )

    def test_force_token_unchanged(self) -> None:
        # Defensive: section.force routing was not disturbed.
        assert (
            gauss_routing_for_canonical("axial_force", topology="line_stations")
            == ("section.force", "section_force")
        )

    def test_full_routing_curvature(self) -> None:
        assert (
            gauss_routing_for_canonical("curvature_y", topology="line_stations")
            == ("section.deformation", "section_deformation")
        )


class TestUnknownTopologyRaises:
    def test_unknown_topology_raises_for_prefix(self) -> None:
        with pytest.raises(ValueError, match="Unknown topology"):
            gauss_keyword_for_canonical("fiber_stress", topology="bogus")

    def test_unknown_topology_raises_for_keyword(self) -> None:
        with pytest.raises(ValueError, match="Unknown topology"):
            catalog_token_for_keyword("section.fiber.stress", topology="bogus")


# =====================================================================
# MPCO group-name aliases for gauss continuum
# =====================================================================
#
# Continuum stress/strain may live under ``stresses`` / ``strains``
# (legacy MPCO) or ``material.stress`` / ``material.strain`` (modern).
# Discovery walks both spellings.

class TestMpcoGaussAliases:
    def test_stresses_alias_includes_modern_keyword(self) -> None:
        from apeGmsh.solvers._element_response import (
            mpco_gauss_group_aliases,
        )
        assert mpco_gauss_group_aliases("stresses") == (
            "stresses", "material.stress",
        )

    def test_strains_alias_includes_modern_keyword(self) -> None:
        from apeGmsh.solvers._element_response import (
            mpco_gauss_group_aliases,
        )
        assert mpco_gauss_group_aliases("strains") == (
            "strains", "material.strain",
        )

    def test_no_alias_for_other_keywords(self) -> None:
        from apeGmsh.solvers._element_response import (
            mpco_gauss_group_aliases,
        )
        # ``axialForce`` / ``section.force`` / etc. have no alternate
        # spellings in MPCO — the function returns just the primary.
        assert mpco_gauss_group_aliases("axialForce") == ("axialForce",)
        assert mpco_gauss_group_aliases("section.force") == ("section.force",)

    def test_primary_keyword_listed_first(self) -> None:
        # Caller relies on this order (first-hit wins in discovery).
        from apeGmsh.solvers._element_response import (
            mpco_gauss_group_aliases,
        )
        assert mpco_gauss_group_aliases("stresses")[0] == "stresses"
