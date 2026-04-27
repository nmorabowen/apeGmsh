"""Phase 11b Step 1 — custom-rule + nodal-force catalogs and topology-aware routing.

Pure unit tests on ``apeGmsh.solvers._element_response``. No OpenSees,
no h5py, no MPCO files — synthetic numpy data only. Verifies the
catalog skeleton plus the per-element resolver that lifts a
``CustomRuleLayout`` to a concrete ``ResponseLayout`` from a
``GP_X`` array and a section-response code vector.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.solvers._element_response import (
    CUSTOM_RULE_CATALOG,
    NODAL_FORCE_CATALOG,
    SECTION_RESPONSE_TO_CANONICAL,
    CatalogLookupError,
    CustomRuleLayout,
    ELE_TAG_DispBeamColumn2d,
    ELE_TAG_DispBeamColumn3d,
    ELE_TAG_ElasticBeam2d,
    ELE_TAG_ElasticBeam3d,
    ELE_TAG_ElasticForceBeamColumn2d,
    ELE_TAG_ElasticForceBeamColumn3d,
    ELE_TAG_ElasticTimoshenkoBeam2d,
    ELE_TAG_ElasticTimoshenkoBeam3d,
    ELE_TAG_ForceBeamColumn2d,
    ELE_TAG_ForceBeamColumn3d,
    ELE_TAG_ForceBeamColumnCBDI2d,
    ELE_TAG_ForceBeamColumnWarping2d,
    ELE_TAG_ModElasticBeam2d,
    NodalForceLayout,
    ResponseLayout,
    catalog_token_for_keyword,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    is_custom_rule_catalogued,
    is_nodal_force_catalogued,
    lookup_custom_rule,
    lookup_nodal_force,
    resolve_layout_from_gp_x,
    unflatten,
)


# =====================================================================
# CUSTOM_RULE_CATALOG — coverage + entry shape
# =====================================================================

EXPECTED_CUSTOM_RULE_KEYS: tuple[tuple[str, str], ...] = (
    ("ForceBeamColumn2d", "section_force"),
    ("ForceBeamColumn3d", "section_force"),
    ("ForceBeamColumnCBDI2d", "section_force"),
    ("ForceBeamColumnWarping2d", "section_force"),
    ("ElasticForceBeamColumn2d", "section_force"),
    ("ElasticForceBeamColumn3d", "section_force"),
    ("DispBeamColumn2d", "section_force"),
    ("DispBeamColumn3d", "section_force"),
)


class TestCustomRuleCatalog:
    def test_all_expected_keys_present(self) -> None:
        for key in EXPECTED_CUSTOM_RULE_KEYS:
            assert key in CUSTOM_RULE_CATALOG, f"missing entry {key}"

    def test_no_unexpected_entries(self) -> None:
        # Round-A scope freeze: catalog grows only via deliberate
        # additions, so a stray entry should fail the test.
        assert set(CUSTOM_RULE_CATALOG.keys()) == set(EXPECTED_CUSTOM_RULE_KEYS)

    @pytest.mark.parametrize("class_name, expected_tag", [
        ("ForceBeamColumn2d", ELE_TAG_ForceBeamColumn2d),
        ("ForceBeamColumn3d", ELE_TAG_ForceBeamColumn3d),
        ("ForceBeamColumnCBDI2d", ELE_TAG_ForceBeamColumnCBDI2d),
        ("ForceBeamColumnWarping2d", ELE_TAG_ForceBeamColumnWarping2d),
        ("ElasticForceBeamColumn2d", ELE_TAG_ElasticForceBeamColumn2d),
        ("ElasticForceBeamColumn3d", ELE_TAG_ElasticForceBeamColumn3d),
        ("DispBeamColumn2d", ELE_TAG_DispBeamColumn2d),
        ("DispBeamColumn3d", ELE_TAG_DispBeamColumn3d),
    ])
    def test_class_tag_matches_constant(
        self, class_name: str, expected_tag: int,
    ) -> None:
        entry = CUSTOM_RULE_CATALOG[(class_name, "section_force")]
        assert entry.class_tag == expected_tag

    def test_all_entries_are_isoparametric_1d(self) -> None:
        for entry in CUSTOM_RULE_CATALOG.values():
            assert entry.coord_system == "isoparametric_1d"

    def test_entries_are_custom_rule_layout_instances(self) -> None:
        for entry in CUSTOM_RULE_CATALOG.values():
            assert isinstance(entry, CustomRuleLayout)


class TestLookupCustomRule:
    def test_hit_returns_entry(self) -> None:
        layout = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        assert layout.class_tag == ELE_TAG_ForceBeamColumn3d
        assert layout.coord_system == "isoparametric_1d"

    def test_miss_class_raises_catalog_lookup_error(self) -> None:
        with pytest.raises(CatalogLookupError, match="No CustomRuleLayout"):
            lookup_custom_rule("NotARealClass", "section_force")

    def test_miss_token_raises(self) -> None:
        # ``section_deformation`` is out of scope for v1.
        with pytest.raises(CatalogLookupError):
            lookup_custom_rule("ForceBeamColumn3d", "section_deformation")

    def test_catalog_lookup_error_is_keyerror_subclass(self) -> None:
        try:
            lookup_custom_rule("Nope", "section_force")
        except KeyError:
            pass
        else:
            pytest.fail("expected KeyError-derived CatalogLookupError")

    def test_is_custom_rule_catalogued_matches_dict(self) -> None:
        assert is_custom_rule_catalogued("ForceBeamColumn3d", "section_force")
        assert not is_custom_rule_catalogued("ForceBeamColumn3d", "garbage")
        assert not is_custom_rule_catalogued("Nope", "section_force")


# =====================================================================
# SECTION_RESPONSE_TO_CANONICAL — code → canonical name
# =====================================================================

class TestSectionResponseDecoder:
    def test_all_six_codes_mapped(self) -> None:
        # Codes 1-6 from SectionForceDeformation.h:52-57.
        expected = {
            1: "bending_moment_z",
            2: "axial_force",
            3: "shear_y",
            4: "bending_moment_y",
            5: "shear_z",
            6: "torsion",
        }
        assert SECTION_RESPONSE_TO_CANONICAL == expected


# =====================================================================
# resolve_layout_from_gp_x — the runtime resolver
# =====================================================================

class TestResolveLayoutFromGpX:
    def test_3d_bare_fiber_section_4_components(self) -> None:
        # FiberSection3d.getType() = [P, Mz, My, T] = (2, 1, 4, 6).
        custom = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        gp_x = np.array([-0.7745966, 0.0, +0.7745966])  # 3 IPs
        layout = resolve_layout_from_gp_x(custom, gp_x, (2, 1, 4, 6))

        assert isinstance(layout, ResponseLayout)
        assert layout.n_gauss_points == 3
        assert layout.n_components_per_gp == 4
        assert layout.flat_size_per_element == 12
        assert layout.class_tag == ELE_TAG_ForceBeamColumn3d
        assert layout.coord_system == "isoparametric_1d"
        assert layout.component_layout == (
            "axial_force",
            "bending_moment_z",
            "bending_moment_y",
            "torsion",
        )
        np.testing.assert_allclose(
            layout.natural_coords, gp_x.reshape(-1, 1),
        )

    def test_3d_aggregated_section_6_components(self) -> None:
        # SectionAggregator(FiberSection3d, Vy, Vz) =
        # [P, Mz, My, T, Vy, Vz] = (2, 1, 4, 6, 3, 5).
        custom = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        gp_x = np.array([-1.0, +1.0])  # endpoints (Lobatto with 2 IPs)
        layout = resolve_layout_from_gp_x(custom, gp_x, (2, 1, 4, 6, 3, 5))

        assert layout.n_components_per_gp == 6
        assert layout.component_layout == (
            "axial_force",
            "bending_moment_z",
            "bending_moment_y",
            "torsion",
            "shear_y",
            "shear_z",
        )

    def test_2d_bare_fiber_section_2_components(self) -> None:
        # FiberSection2d.getType() = [P, Mz] = (2, 1).
        custom = lookup_custom_rule("ForceBeamColumn2d", "section_force")
        gp_x = np.array([-1.0, 0.0, +1.0])
        layout = resolve_layout_from_gp_x(custom, gp_x, (2, 1))

        assert layout.n_gauss_points == 3
        assert layout.n_components_per_gp == 2
        assert layout.component_layout == ("axial_force", "bending_moment_z")
        assert layout.class_tag == ELE_TAG_ForceBeamColumn2d

    def test_2d_aggregated_with_shear(self) -> None:
        # 2D aggregated: [P, Mz, Vy] = (2, 1, 3).
        custom = lookup_custom_rule("DispBeamColumn2d", "section_force")
        gp_x = np.array([-0.57735, +0.57735])
        layout = resolve_layout_from_gp_x(custom, gp_x, (2, 1, 3))

        assert layout.component_layout == (
            "axial_force", "bending_moment_z", "shear_y",
        )
        assert layout.class_tag == ELE_TAG_DispBeamColumn2d

    def test_unknown_section_code_raises_keyerror(self) -> None:
        custom = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        gp_x = np.array([0.0])
        # Code 99 is not in SECTION_RESPONSE_TO_CANONICAL (warping codes
        # like 15 / 18 are also unsupported in v1).
        with pytest.raises(KeyError):
            resolve_layout_from_gp_x(custom, gp_x, (2, 99))

    def test_gp_x_is_normalised_to_column_vector(self) -> None:
        # Accept 1-D or shape-(n,1) input; normalise to (n, 1).
        custom = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        layout_1d = resolve_layout_from_gp_x(
            custom, np.array([0.1, 0.2, 0.3]), (2, 1),
        )
        layout_col = resolve_layout_from_gp_x(
            custom, np.array([[0.1], [0.2], [0.3]]), (2, 1),
        )
        np.testing.assert_array_equal(
            layout_1d.natural_coords, layout_col.natural_coords,
        )
        assert layout_1d.natural_coords.shape == (3, 1)

    def test_resolved_layout_is_unflatten_ready(self) -> None:
        """Round-trip: build flat data, resolve, unflatten — names land right."""
        custom = lookup_custom_rule("ForceBeamColumn3d", "section_force")
        # Aggregated 6-comp section, 2 IPs (Lobatto endpoints).
        gp_x = np.array([-1.0, +1.0])
        codes = (2, 1, 4, 6, 3, 5)
        layout = resolve_layout_from_gp_x(custom, gp_x, codes)

        # Synthesise flat data: 1 step, 1 element, 2 IPs × 6 comps = 12 cols.
        # GP-slowest, comp-fastest packing — a known signature per slot:
        # value = gp * 100 + comp_idx
        flat = np.zeros((1, 1, 12), dtype=np.float64)
        for gp in range(2):
            for k in range(6):
                flat[0, 0, gp * 6 + k] = gp * 100 + k

        components = unflatten(flat, layout)

        # axial_force is component k=0 → values 0 (gp 0) and 100 (gp 1).
        np.testing.assert_array_equal(
            components["axial_force"],
            np.array([[[0.0, 100.0]]]),
        )
        # shear_z is component k=5 → values 5 (gp 0) and 105 (gp 1).
        np.testing.assert_array_equal(
            components["shear_z"],
            np.array([[[5.0, 105.0]]]),
        )
        # bending_moment_z is component k=1 → values 1, 101.
        np.testing.assert_array_equal(
            components["bending_moment_z"],
            np.array([[[1.0, 101.0]]]),
        )


# =====================================================================
# NODAL_FORCE_CATALOG — coverage + entry shape
# =====================================================================

EXPECTED_NODAL_FORCE_KEYS: tuple[tuple[str, str], ...] = (
    ("ElasticBeam2d", "global_force"),
    ("ElasticBeam2d", "local_force"),
    ("ElasticBeam3d", "global_force"),
    ("ElasticBeam3d", "local_force"),
    ("ModElasticBeam2d", "global_force"),
    ("ModElasticBeam2d", "local_force"),
    ("ElasticTimoshenkoBeam2d", "global_force"),
    ("ElasticTimoshenkoBeam2d", "local_force"),
    ("ElasticTimoshenkoBeam3d", "global_force"),
    ("ElasticTimoshenkoBeam3d", "local_force"),
)


class TestNodalForceCatalog:
    def test_all_expected_keys_present(self) -> None:
        for key in EXPECTED_NODAL_FORCE_KEYS:
            assert key in NODAL_FORCE_CATALOG, f"missing entry {key}"

    def test_no_unexpected_entries(self) -> None:
        assert set(NODAL_FORCE_CATALOG.keys()) == set(EXPECTED_NODAL_FORCE_KEYS)

    @pytest.mark.parametrize("class_name, expected_tag", [
        ("ElasticBeam2d", ELE_TAG_ElasticBeam2d),
        ("ElasticBeam3d", ELE_TAG_ElasticBeam3d),
        ("ModElasticBeam2d", ELE_TAG_ModElasticBeam2d),
        ("ElasticTimoshenkoBeam2d", ELE_TAG_ElasticTimoshenkoBeam2d),
        ("ElasticTimoshenkoBeam3d", ELE_TAG_ElasticTimoshenkoBeam3d),
    ])
    def test_class_tag_matches(self, class_name: str, expected_tag: int) -> None:
        for token in ("global_force", "local_force"):
            assert NODAL_FORCE_CATALOG[(class_name, token)].class_tag == expected_tag

    def test_2d_entries_have_3_components_per_node(self) -> None:
        for class_name in ("ElasticBeam2d", "ModElasticBeam2d", "ElasticTimoshenkoBeam2d"):
            for token in ("global_force", "local_force"):
                entry = NODAL_FORCE_CATALOG[(class_name, token)]
                assert entry.n_nodes_per_element == 2
                assert entry.n_components_per_node == 3
                assert entry.flat_size_per_element == 6

    def test_3d_entries_have_6_components_per_node(self) -> None:
        for class_name in ("ElasticBeam3d", "ElasticTimoshenkoBeam3d"):
            for token in ("global_force", "local_force"):
                entry = NODAL_FORCE_CATALOG[(class_name, token)]
                assert entry.n_nodes_per_element == 2
                assert entry.n_components_per_node == 6
                assert entry.flat_size_per_element == 12

    def test_global_3d_layout_uses_canonical_global_names(self) -> None:
        entry = NODAL_FORCE_CATALOG[("ElasticBeam3d", "global_force")]
        assert entry.frame == "global"
        assert entry.component_layout == (
            "nodal_resisting_force_x",
            "nodal_resisting_force_y",
            "nodal_resisting_force_z",
            "nodal_resisting_moment_x",
            "nodal_resisting_moment_y",
            "nodal_resisting_moment_z",
        )

    def test_local_3d_layout_uses_local_names(self) -> None:
        entry = NODAL_FORCE_CATALOG[("ElasticBeam3d", "local_force")]
        assert entry.frame == "local"
        # Every name in a local-frame entry must contain ``_local_``.
        for name in entry.component_layout:
            assert "_local_" in name

    def test_global_2d_layout(self) -> None:
        entry = NODAL_FORCE_CATALOG[("ElasticBeam2d", "global_force")]
        assert entry.component_layout == (
            "nodal_resisting_force_x",
            "nodal_resisting_force_y",
            "nodal_resisting_moment_z",
        )

    def test_local_2d_layout(self) -> None:
        entry = NODAL_FORCE_CATALOG[("ElasticBeam2d", "local_force")]
        assert entry.component_layout == (
            "nodal_resisting_force_local_x",
            "nodal_resisting_force_local_y",
            "nodal_resisting_moment_local_z",
        )


class TestNodalForceLayoutValidation:
    def test_bad_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            NodalForceLayout(
                n_nodes_per_element=2,
                n_components_per_node=3,
                component_layout=("a", "b", "c"),
                class_tag=999,
                frame="weird",
            )

    def test_component_layout_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="component_layout"):
            NodalForceLayout(
                n_nodes_per_element=2,
                n_components_per_node=3,
                component_layout=("a", "b"),  # only 2 names but 3 declared
                class_tag=999,
                frame="global",
            )


class TestLookupNodalForce:
    def test_hit(self) -> None:
        entry = lookup_nodal_force("ElasticBeam3d", "global_force")
        assert entry.class_tag == ELE_TAG_ElasticBeam3d
        assert entry.frame == "global"

    def test_miss_raises_catalog_lookup_error(self) -> None:
        with pytest.raises(CatalogLookupError, match="NodalForceLayout"):
            lookup_nodal_force("Nonexistent", "global_force")

    def test_is_nodal_force_catalogued(self) -> None:
        assert is_nodal_force_catalogued("ElasticBeam3d", "local_force")
        assert not is_nodal_force_catalogued("ElasticBeam3d", "garbage")


# =====================================================================
# Topology-aware keyword routing
# =====================================================================

class TestGaussKeywordDefaultTopology:
    """Phase 11a behaviour must be preserved when ``topology`` is omitted."""

    def test_continuum_stress(self) -> None:
        assert gauss_keyword_for_canonical("stress_xx") == "stresses"
        assert gauss_keyword_for_canonical("stress_yz") == "stresses"

    def test_continuum_strain(self) -> None:
        assert gauss_keyword_for_canonical("strain_xy") == "strains"

    def test_shell_resultants(self) -> None:
        assert gauss_keyword_for_canonical("membrane_force_xx") == "stresses"
        assert gauss_keyword_for_canonical("bending_moment_xy") == "stresses"
        assert gauss_keyword_for_canonical("transverse_shear_yz") == "stresses"

    def test_truss_axial_scalar(self) -> None:
        assert gauss_keyword_for_canonical("axial_force") == "axialForce"

    def test_unknown_returns_none(self) -> None:
        assert gauss_keyword_for_canonical("displacement_x") is None


class TestGaussKeywordLineStationsTopology:
    @pytest.mark.parametrize("name", [
        "axial_force",
        "shear_y",
        "shear_z",
        "torsion",
        "bending_moment_y",
        "bending_moment_z",
    ])
    def test_line_diagram_routes_to_section_force(self, name: str) -> None:
        assert gauss_keyword_for_canonical(
            name, topology="line_stations",
        ) == "section.force"

    def test_axial_force_topology_disambiguates_truss_vs_beam(self) -> None:
        # Same canonical name, different keyword by topology — this is
        # the central design constraint for Phase 11b.
        assert gauss_keyword_for_canonical("axial_force") == "axialForce"
        assert gauss_keyword_for_canonical(
            "axial_force", topology="line_stations",
        ) == "section.force"

    def test_shell_components_not_in_line_station_table(self) -> None:
        # ``bending_moment_xx`` is a shell resultant; should NOT route
        # at the line-stations topology even though "bending_moment"
        # matches the prefix table (suffix is xx, not y/z — and even
        # so, the test here is that the lookup logic returns
        # section.force for ALL bending_moment_* line-station forms;
        # what we want to ensure is that a shell user accidentally
        # passing topology="line_stations" still gets a defined
        # routing for the line-station forms only).
        # Stress/strain prefixes are not in the line-station table:
        assert gauss_keyword_for_canonical(
            "stress_xx", topology="line_stations",
        ) is None
        assert gauss_keyword_for_canonical(
            "membrane_force_xx", topology="line_stations",
        ) is None


class TestGaussKeywordNodalForcesTopology:
    @pytest.mark.parametrize("name, expected", [
        ("nodal_resisting_force_x", "globalForce"),
        ("nodal_resisting_force_y", "globalForce"),
        ("nodal_resisting_force_z", "globalForce"),
        ("nodal_resisting_moment_x", "globalForce"),
        ("nodal_resisting_moment_y", "globalForce"),
        ("nodal_resisting_moment_z", "globalForce"),
        ("nodal_resisting_force_local_x", "localForce"),
        ("nodal_resisting_force_local_y", "localForce"),
        ("nodal_resisting_force_local_z", "localForce"),
        ("nodal_resisting_moment_local_x", "localForce"),
        ("nodal_resisting_moment_local_y", "localForce"),
        ("nodal_resisting_moment_local_z", "localForce"),
    ])
    def test_routes(self, name: str, expected: str) -> None:
        assert gauss_keyword_for_canonical(
            name, topology="nodal_forces",
        ) == expected

    def test_axial_force_not_in_nodal_force_table(self) -> None:
        assert gauss_keyword_for_canonical(
            "axial_force", topology="nodal_forces",
        ) is None


class TestGaussKeywordInvalidTopology:
    def test_unknown_topology_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown topology"):
            gauss_keyword_for_canonical("axial_force", topology="bogus")


# =====================================================================
# catalog_token_for_keyword + gauss_routing_for_canonical
# =====================================================================

class TestCatalogTokenForKeyword:
    def test_default_topology(self) -> None:
        assert catalog_token_for_keyword("stresses") == "stress"
        assert catalog_token_for_keyword("strains") == "strain"
        assert catalog_token_for_keyword("axialForce") == "axial_force"

    def test_line_stations_topology(self) -> None:
        assert catalog_token_for_keyword(
            "section.force", topology="line_stations",
        ) == "section_force"

    def test_nodal_forces_topology(self) -> None:
        assert catalog_token_for_keyword(
            "globalForce", topology="nodal_forces",
        ) == "global_force"
        assert catalog_token_for_keyword(
            "localForce", topology="nodal_forces",
        ) == "local_force"

    def test_unknown_keyword_returns_none(self) -> None:
        assert catalog_token_for_keyword("nonsense") is None

    def test_line_station_keyword_unknown_in_default_topology(self) -> None:
        # ``section.force`` only has a token in the line-stations table.
        assert catalog_token_for_keyword("section.force") is None


class TestGaussRoutingForCanonical:
    def test_default_topology_truss(self) -> None:
        routing = gauss_routing_for_canonical("axial_force")
        assert routing == ("axialForce", "axial_force")

    def test_line_stations_topology_axial_force(self) -> None:
        routing = gauss_routing_for_canonical(
            "axial_force", topology="line_stations",
        )
        assert routing == ("section.force", "section_force")

    def test_line_stations_topology_bending_moment(self) -> None:
        routing = gauss_routing_for_canonical(
            "bending_moment_z", topology="line_stations",
        )
        assert routing == ("section.force", "section_force")

    def test_nodal_forces_topology_global(self) -> None:
        routing = gauss_routing_for_canonical(
            "nodal_resisting_force_x", topology="nodal_forces",
        )
        assert routing == ("globalForce", "global_force")

    def test_nodal_forces_topology_local(self) -> None:
        routing = gauss_routing_for_canonical(
            "nodal_resisting_moment_local_z", topology="nodal_forces",
        )
        assert routing == ("localForce", "local_force")

    def test_unrouted_returns_none(self) -> None:
        assert gauss_routing_for_canonical(
            "stress_xx", topology="line_stations",
        ) is None
