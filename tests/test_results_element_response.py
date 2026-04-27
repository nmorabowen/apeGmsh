"""Phase 11a Step A — element response catalog + unflatten core.

Pure unit tests on ``apeGmsh.solvers._element_response``. No OpenSees,
no h5py, no MPCO files — synthetic numpy data only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh.solvers._element_response import (
    ELE_TAG_ASDShellQ4,
    ELE_TAG_ASDShellT3,
    ELE_TAG_BbarBrick,
    ELE_TAG_Brick,
    ELE_TAG_CorotTruss,
    ELE_TAG_CorotTruss2,
    ELE_TAG_EightNodeQuad,
    ELE_TAG_FourNodeQuad,
    ELE_TAG_FourNodeTetrahedron,
    ELE_TAG_InertiaTruss,
    ELE_TAG_ShellDKGQ,
    ELE_TAG_ShellDKGT,
    ELE_TAG_ShellMITC4,
    ELE_TAG_ShellMITC9,
    ELE_TAG_ShellNLDKGQ,
    ELE_TAG_ShellNLDKGT,
    ELE_TAG_SSPbrick,
    ELE_TAG_SSPquad,
    ELE_TAG_TenNodeTetrahedron,
    ELE_TAG_Tri31,
    ELE_TAG_Truss,
    ELE_TAG_Truss2,
    ELE_TAG_Twenty_Node_Brick,
    CatalogLookupError,
    IntRule,
    MPCOElementKey,
    RESPONSE_CATALOG,
    catalog_token_for_keyword,
    flatten,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    is_catalogued,
    lookup,
    parse_mpco_element_key,
    split_canonical_component,
    unflatten,
)


# =====================================================================
# Catalog lookup
# =====================================================================

class TestLookup:
    def test_four_node_tet_stress(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        assert layout.n_gauss_points == 1
        assert layout.n_components_per_gp == 6
        assert layout.flat_size_per_element == 6
        assert layout.class_tag == ELE_TAG_FourNodeTetrahedron
        assert layout.coord_system == "barycentric_tet"
        # First name should be stress_xx, last stress_xz (per _vocabulary.STRESS).
        assert layout.component_layout[0] == "stress_xx"
        assert layout.component_layout[-1] == "stress_xz"
        assert len(layout.component_layout) == 6

    def test_four_node_tet_centroid_coord(self) -> None:
        """1-GP tet must sit at the volume centroid (1/4, 1/4, 1/4)."""
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        np.testing.assert_allclose(
            layout.natural_coords, [[0.25, 0.25, 0.25]], atol=1e-15,
        )

    def test_ten_node_tet_stress(self) -> None:
        layout = lookup("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress")
        assert layout.n_gauss_points == 4
        assert layout.n_components_per_gp == 6
        assert layout.flat_size_per_element == 24
        assert layout.class_tag == ELE_TAG_TenNodeTetrahedron

    def test_ten_node_tet_gp_locations(self) -> None:
        """4-point Hammer-Stroud rule: alpha/beta arrangement.

        Verified against TenNodeTetrahedron.cpp lines 223–226 + 593–600.
        """
        layout = lookup("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress")
        alpha = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
        beta = (5.0 - math.sqrt(5.0)) / 20.0
        expected = np.array([
            [alpha, beta, beta],
            [beta, alpha, beta],
            [beta, beta, alpha],
            [beta, beta, beta],
        ])
        np.testing.assert_allclose(layout.natural_coords, expected, atol=1e-14)

        # Each row is a valid barycentric coord (4th = 1 - sum), all positive.
        sums = layout.natural_coords.sum(axis=1)
        assert np.all(sums < 1.0 - 1e-12)
        assert np.all(layout.natural_coords > 0.0)

    def test_strain_token_uses_strain_names(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "strain")
        assert layout.component_layout[0] == "strain_xx"
        assert layout.component_layout[-1] == "strain_xz"
        # Same layout shape as stress.
        stress = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        assert layout.n_gauss_points == stress.n_gauss_points
        assert layout.n_components_per_gp == stress.n_components_per_gp

    def test_lookup_miss_is_helpful(self) -> None:
        with pytest.raises(CatalogLookupError) as excinfo:
            lookup("NotAClass", 999, "stress")
        msg = str(excinfo.value)
        assert "NotAClass" in msg
        assert "999" in msg
        assert "RESPONSE_CATALOG" in msg
        assert "_element_response.py" in msg

    def test_lookup_error_is_keyerror(self) -> None:
        """Subclassing KeyError lets callers ``except KeyError`` for skip-on-miss."""
        with pytest.raises(KeyError):
            lookup("NotAClass", 999, "stress")

    def test_is_catalogued(self) -> None:
        assert is_catalogued("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        assert not is_catalogued("FourNodeTetrahedron", IntRule.Tet_GL_1, "globalForce")
        assert not is_catalogued("Unknown", IntRule.Tet_GL_1, "stress")


# =====================================================================
# 3D solid hex extensions (Brick, BbarBrick, SSPbrick)
# =====================================================================

class TestHexLookup:
    def test_brick_uses_hex_gl_2(self) -> None:
        layout = lookup("Brick", IntRule.Hex_GL_2, "stress")
        assert layout.n_gauss_points == 8
        assert layout.flat_size_per_element == 48
        assert layout.coord_system == "isoparametric"
        assert layout.class_tag == ELE_TAG_Brick

    def test_brick_gp_locations_are_pm_one_over_root3(self) -> None:
        """8 GPs at all combinations of (±1/√3, ±1/√3, ±1/√3),
        ordered ξ-slowest / ζ-fastest per Brick.cpp:540–542."""
        layout = lookup("Brick", IntRule.Hex_GL_2, "stress")
        m = -1.0 / math.sqrt(3.0)
        p = +1.0 / math.sqrt(3.0)
        expected = np.array([
            [m, m, m], [m, m, p],
            [m, p, m], [m, p, p],
            [p, m, m], [p, m, p],
            [p, p, m], [p, p, p],
        ])
        np.testing.assert_allclose(layout.natural_coords, expected, atol=1e-14)

    def test_bbarbrick_shares_brick_shape(self) -> None:
        b = lookup("Brick", IntRule.Hex_GL_2, "stress")
        bb = lookup("BbarBrick", IntRule.Hex_GL_2, "stress")
        assert bb.n_gauss_points == b.n_gauss_points
        assert bb.n_components_per_gp == b.n_components_per_gp
        assert bb.coord_system == b.coord_system
        np.testing.assert_array_equal(bb.natural_coords, b.natural_coords)
        # Different formulations → different class_tag.
        assert bb.class_tag == ELE_TAG_BbarBrick
        assert bb.class_tag != b.class_tag

    def test_sspbrick_uses_hex_gl_1_at_origin(self) -> None:
        layout = lookup("SSPbrick", IntRule.Hex_GL_1, "stress")
        assert layout.n_gauss_points == 1
        assert layout.flat_size_per_element == 6
        assert layout.coord_system == "isoparametric"
        assert layout.class_tag == ELE_TAG_SSPbrick
        np.testing.assert_array_equal(
            layout.natural_coords, [[0.0, 0.0, 0.0]],
        )

    def test_strain_entries_present_for_hex_classes(self) -> None:
        for cls, rule in (
            ("Brick", IntRule.Hex_GL_2),
            ("BbarBrick", IntRule.Hex_GL_2),
            ("SSPbrick", IntRule.Hex_GL_1),
        ):
            layout = lookup(cls, rule, "strain")
            assert layout.component_layout[0] == "strain_xx"
            assert layout.component_layout[-1] == "strain_xz"


# =====================================================================
# Catalog coverage summary — guard against accidental regressions
# =====================================================================

# =====================================================================
# 2D solid extensions (FourNodeQuad, Tri31, SSPquad)
# =====================================================================

class TestPlaneLookup:
    def test_four_node_quad_uses_quad_gl_2(self) -> None:
        layout = lookup("FourNodeQuad", IntRule.Quad_GL_2, "stress")
        assert layout.n_gauss_points == 4
        # 3 components per GP for plane stress / plane strain.
        assert layout.n_components_per_gp == 3
        assert layout.flat_size_per_element == 12
        assert layout.coord_system == "isoparametric"
        assert layout.class_tag == ELE_TAG_FourNodeQuad
        assert layout.component_layout == ("stress_xx", "stress_yy", "stress_xy")

    def test_four_node_quad_gp_order_is_counterclockwise(self) -> None:
        """FourNodeQuad.cpp:298–305 stores GPs counter-clockwise around the
        parent square: (−,−), (+,−), (+,+), (−,+)."""
        layout = lookup("FourNodeQuad", IntRule.Quad_GL_2, "stress")
        m = -1.0 / math.sqrt(3.0)
        p = +1.0 / math.sqrt(3.0)
        np.testing.assert_allclose(
            layout.natural_coords,
            [[m, m], [p, m], [p, p], [m, p]],
            atol=1e-14,
        )

    def test_tri31_uses_triangle_gl_1_centroid(self) -> None:
        layout = lookup("Tri31", IntRule.Triangle_GL_1, "stress")
        assert layout.n_gauss_points == 1
        assert layout.flat_size_per_element == 3
        assert layout.coord_system == "barycentric_tri"
        assert layout.class_tag == ELE_TAG_Tri31
        np.testing.assert_allclose(
            layout.natural_coords, [[1.0 / 3.0, 1.0 / 3.0]],
        )

    def test_sspquad_uses_quad_gl_1_at_origin(self) -> None:
        layout = lookup("SSPquad", IntRule.Quad_GL_1, "stress")
        assert layout.n_gauss_points == 1
        assert layout.flat_size_per_element == 3
        assert layout.coord_system == "isoparametric"
        assert layout.class_tag == ELE_TAG_SSPquad
        np.testing.assert_array_equal(
            layout.natural_coords, [[0.0, 0.0]],
        )

    def test_2d_strain_entries_present(self) -> None:
        for cls, rule in (
            ("FourNodeQuad", IntRule.Quad_GL_2),
            ("Tri31", IntRule.Triangle_GL_1),
            ("SSPquad", IntRule.Quad_GL_1),
        ):
            layout = lookup(cls, rule, "strain")
            assert layout.component_layout == (
                "strain_xx", "strain_yy", "strain_xy",
            )

    def test_2d_classes_distinguishable_from_3d_by_dim(self) -> None:
        """All 2D entries have natural_coords of shape (n_gp, 2)."""
        for cls, rule in (
            ("FourNodeQuad", IntRule.Quad_GL_2),
            ("Tri31", IntRule.Triangle_GL_1),
            ("SSPquad", IntRule.Quad_GL_1),
        ):
            layout = lookup(cls, rule, "stress")
            assert layout.natural_coords.shape[1] == 2


# =====================================================================
# Higher-order solid extensions (Twenty_Node_Brick, EightNodeQuad)
# =====================================================================

class TestHigherOrderSolids:
    def test_twenty_node_brick_uses_hex_gl_3(self) -> None:
        layout = lookup("Twenty_Node_Brick", IntRule.Hex_GL_3, "stress")
        assert layout.n_gauss_points == 27
        assert layout.n_components_per_gp == 6
        assert layout.flat_size_per_element == 162
        assert layout.coord_system == "isoparametric"
        assert layout.class_tag == ELE_TAG_Twenty_Node_Brick

    def test_twenty_node_brick_corner_edge_face_centroid(self) -> None:
        """27-GP order: 8 corners, 12 edges, 6 faces, 1 centroid.

        Per ``shp3dv.cpp::brcshl`` (lines 251–277) the GP coordinates
        come from RA/SA/TA × G with G = 2√(3/5). Each value is then
        either ±√(3/5) or 0.
        """
        layout = lookup("Twenty_Node_Brick", IntRule.Hex_GL_3, "stress")
        s = math.sqrt(3.0 / 5.0)

        # First 8 are corners — every coordinate is ±s, none are 0.
        for i in range(8):
            for v in layout.natural_coords[i]:
                assert abs(abs(v) - s) < 1e-12, (
                    f"corner GP {i}: value {v} not in {{-s, +s}}"
                )

        # Last (index 26) is the body centroid — all zeros.
        np.testing.assert_array_equal(
            layout.natural_coords[26], [0.0, 0.0, 0.0],
        )

        # Check that the 8 corners are exactly the 8 sign combinations
        # of (±s, ±s, ±s) — and only those.
        corners = {tuple(layout.natural_coords[i]) for i in range(8)}
        expected_corners = {
            (sx * s, sy * s, sz * s)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        }
        assert corners == expected_corners

        # 12 edge midpoints: exactly ONE coord is 0, other two are ±s.
        for i in range(8, 20):
            zero_count = sum(
                1 for v in layout.natural_coords[i] if abs(v) < 1e-12
            )
            assert zero_count == 1, f"edge GP {i}: expected 1 zero coord"

        # 6 face centers: exactly TWO coords are 0.
        for i in range(20, 26):
            zero_count = sum(
                1 for v in layout.natural_coords[i] if abs(v) < 1e-12
            )
            assert zero_count == 2, f"face GP {i}: expected 2 zero coords"

    def test_eight_node_quad_uses_quad_gl_3(self) -> None:
        layout = lookup("EightNodeQuad", IntRule.Quad_GL_3, "stress")
        assert layout.n_gauss_points == 9
        assert layout.n_components_per_gp == 3
        assert layout.flat_size_per_element == 27
        assert layout.coord_system == "isoparametric"
        assert layout.class_tag == ELE_TAG_EightNodeQuad

    def test_eight_node_quad_gp_layout(self) -> None:
        """9-GP order: 4 corners CCW, 4 edge midpoints CCW, then centroid."""
        layout = lookup("EightNodeQuad", IntRule.Quad_GL_3, "stress")
        s = math.sqrt(3.0 / 5.0)
        expected = np.array([
            [-s, -s], [+s, -s], [+s, +s], [-s, +s],
            [0.0, -s], [+s, 0.0], [0.0, +s], [-s, 0.0],
            [0.0, 0.0],
        ])
        np.testing.assert_allclose(layout.natural_coords, expected, atol=1e-14)


# =====================================================================
# Canonical-component prefix splitter + gauss keyword routing
# =====================================================================

class TestSplitCanonicalComponent:
    @pytest.mark.parametrize("name,expected", [
        ("stress_xx", ("stress", "xx")),
        ("strain_yz", ("strain", "yz")),
        ("membrane_force_xy", ("membrane_force", "xy")),
        ("bending_moment_xx", ("bending_moment", "xx")),
        ("transverse_shear_xz", ("transverse_shear", "xz")),
        ("curvature_yy", ("curvature", "yy")),
        ("transverse_shear_strain_yz", ("transverse_shear_strain", "yz")),
        ("displacement_x", ("displacement", "x")),
        ("rotation_z", ("rotation", "z")),
    ])
    def test_known_suffixes_split_correctly(self, name, expected) -> None:
        assert split_canonical_component(name) == expected

    @pytest.mark.parametrize("scalar", [
        "pore_pressure", "damage", "von_mises_stress",
        "equivalent_plastic_strain",
        # Truss scalar — no axis suffix.
        "axial_force",
    ])
    def test_scalars_return_none(self, scalar) -> None:
        # No _xx/_x suffix → splitter returns None; the keyword
        # routing handles scalars via the full-name fallback.
        assert split_canonical_component(scalar) is None

    def test_longer_suffix_wins(self) -> None:
        """``stress_xz`` must split as (stress, xz), not (stress_x, z)."""
        assert split_canonical_component("stress_xz") == ("stress", "xz")
        assert split_canonical_component("transverse_shear_xz") == (
            "transverse_shear", "xz",
        )


class TestGaussKeywordRouting:
    @pytest.mark.parametrize("name,expected", [
        # Continuum
        ("stress_xx", "stresses"),
        ("strain_xy", "strains"),
        # Shell stress resultants — same MPCO group as stress
        ("membrane_force_xx", "stresses"),
        ("bending_moment_xy", "stresses"),
        ("transverse_shear_xz", "stresses"),
        # Shell generalized strains — same MPCO group as strain
        ("membrane_strain_xx", "strains"),
        ("curvature_yy", "strains"),
        ("transverse_shear_strain_xz", "strains"),
        # Truss scalar — full-name lookup (no axis suffix)
        ("axial_force", "axialForce"),
        # Nodal kinematics — no Gauss routing
        ("displacement_x", None),
        ("rotation_y", None),
    ])
    def test_keyword_for_canonical(self, name, expected) -> None:
        assert gauss_keyword_for_canonical(name) == expected

    def test_catalog_token_for_keyword(self) -> None:
        assert catalog_token_for_keyword("stresses") == "stress"
        assert catalog_token_for_keyword("strains") == "strain"
        assert catalog_token_for_keyword("axialForce") == "axial_force"
        assert catalog_token_for_keyword("globalForce") is None

    def test_routing_pair(self) -> None:
        """``gauss_routing_for_canonical`` returns ``(keyword, catalog_token)``."""
        assert gauss_routing_for_canonical("stress_xx") == ("stresses", "stress")
        assert gauss_routing_for_canonical("membrane_force_xy") == (
            "stresses", "stress",
        )
        assert gauss_routing_for_canonical("curvature_xx") == ("strains", "strain")
        assert gauss_routing_for_canonical("displacement_x") is None


# =====================================================================
# Shell catalog entries
# =====================================================================

class TestShellLayouts:
    @pytest.mark.parametrize("cls,rule,n_gp,expected_class_tag", [
        ("ShellMITC4", IntRule.Quad_GL_2, 4, ELE_TAG_ShellMITC4),
        ("ShellDKGQ", IntRule.Quad_GL_2, 4, ELE_TAG_ShellDKGQ),
        ("ShellNLDKGQ", IntRule.Quad_GL_2, 4, ELE_TAG_ShellNLDKGQ),
        ("ASDShellQ4", IntRule.Quad_GL_2, 4, ELE_TAG_ASDShellQ4),
        ("ShellMITC9", IntRule.Quad_GL_3, 9, ELE_TAG_ShellMITC9),
        ("ShellDKGT", IntRule.Triangle_GL_2C, 4, ELE_TAG_ShellDKGT),
        ("ShellNLDKGT", IntRule.Triangle_GL_2C, 4, ELE_TAG_ShellNLDKGT),
        ("ASDShellT3", IntRule.Triangle_GL_2B, 3, ELE_TAG_ASDShellT3),
    ])
    def test_shell_stress_layout(
        self, cls: str, rule: int, n_gp: int, expected_class_tag: int,
    ) -> None:
        layout = lookup(cls, rule, "stress")
        assert layout.n_gauss_points == n_gp
        # All shell classes share the same 8-component resultant layout.
        assert layout.n_components_per_gp == 8
        assert layout.flat_size_per_element == n_gp * 8
        assert layout.class_tag == expected_class_tag
        assert layout.component_layout == (
            "membrane_force_xx", "membrane_force_yy", "membrane_force_xy",
            "bending_moment_xx", "bending_moment_yy", "bending_moment_xy",
            "transverse_shear_xz", "transverse_shear_yz",
        )

    def test_shell_strain_layout_is_generalized_strains(self) -> None:
        layout = lookup("ShellMITC4", IntRule.Quad_GL_2, "strain")
        assert layout.component_layout == (
            "membrane_strain_xx", "membrane_strain_yy", "membrane_strain_xy",
            "curvature_xx", "curvature_yy", "curvature_xy",
            "transverse_shear_strain_xz", "transverse_shear_strain_yz",
        )

    def test_quad_shells_share_gp_array(self) -> None:
        """All four 4-GP quad shells use the CCW corner-walk array.

        The four classes have different formulations but identical
        Gauss-point coordinates per their source.
        """
        a = lookup("ShellMITC4", IntRule.Quad_GL_2, "stress").natural_coords
        for cls in ("ShellDKGQ", "ShellNLDKGQ", "ASDShellQ4"):
            b = lookup(cls, IntRule.Quad_GL_2, "stress").natural_coords
            np.testing.assert_array_equal(b, a)

    def test_mitc9_uses_alternating_walk_order(self) -> None:
        """ShellMITC9's GP order alternates corner/edge around the parent square,
        ending with the centroid — distinct from EightNodeQuad's order."""
        mitc = lookup("ShellMITC9", IntRule.Quad_GL_3, "stress").natural_coords
        # First three: corner SW, edge S, corner SE.
        s = math.sqrt(3.0 / 5.0)
        np.testing.assert_allclose(
            mitc[:3], [[-s, -s], [0.0, -s], [s, -s]], atol=1e-14,
        )
        # Centroid is the last point.
        np.testing.assert_array_equal(mitc[8], [0.0, 0.0])

    def test_asdshellt3_mid_edge_gps(self) -> None:
        """ASDShellT3 places its 3 GPs at the parent-triangle edge midpoints."""
        layout = lookup("ASDShellT3", IntRule.Triangle_GL_2B, "stress")
        np.testing.assert_array_equal(
            layout.natural_coords,
            [[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]],
        )

    def test_shelldkgt_4_point_rule(self) -> None:
        """ShellDKGT uses the 4-point degree-3 rule: centroid + 3 corner-biased."""
        layout = lookup("ShellDKGT", IntRule.Triangle_GL_2C, "stress")
        np.testing.assert_array_equal(
            layout.natural_coords,
            [[1.0/3.0, 1.0/3.0], [0.2, 0.6], [0.6, 0.2], [0.2, 0.2]],
        )

    def test_layered_and_nonlayered_share_entry(self) -> None:
        """Layered shells use the same catalog entry as their non-layered siblings.

        ShellMITC4 with a regular ``PlateFiberSection`` and ShellMITC4
        with a ``LayeredShellFiberSection`` both use the catalog's
        ``ShellMITC4`` entry — the layered-section behavior is
        invisible at the GP-resultant topology level (per-layer
        through-thickness data is a different topology level).
        """
        # The catalog has only one ShellMITC4 entry per token; the
        # element class is the only key, not the section kind.
        keys_for_mitc4 = [
            (cls, rule, tok)
            for (cls, rule, tok) in RESPONSE_CATALOG
            if cls == "ShellMITC4"
        ]
        assert len(keys_for_mitc4) == 2   # stress + strain only


# =====================================================================
# Truss family — single GP, scalar axial force
# =====================================================================

class TestTrussLookup:
    @pytest.mark.parametrize("cls,expected_class_tag", [
        ("Truss", ELE_TAG_Truss),
        ("CorotTruss", ELE_TAG_CorotTruss),
        ("Truss2", ELE_TAG_Truss2),
        ("CorotTruss2", ELE_TAG_CorotTruss2),
        ("InertiaTruss", ELE_TAG_InertiaTruss),
    ])
    def test_truss_layout(self, cls: str, expected_class_tag: int) -> None:
        layout = lookup(cls, IntRule.Line_GL_1, "axial_force")
        assert layout.n_gauss_points == 1
        assert layout.n_components_per_gp == 1
        assert layout.flat_size_per_element == 1
        assert layout.coord_system == "isoparametric_1d"
        assert layout.class_tag == expected_class_tag
        assert layout.component_layout == ("axial_force",)
        # 1 GP at the parent-line midpoint.
        np.testing.assert_array_equal(layout.natural_coords, [[0.0]])

    def test_axial_force_routes_through_axialforce_keyword(self) -> None:
        """The catalog token ``"axial_force"`` maps to OpenSees keyword
        ``"axialForce"`` and MPCO group ``ON_ELEMENTS/axialForce/``."""
        assert gauss_routing_for_canonical("axial_force") == (
            "axialForce", "axial_force",
        )


def test_catalog_coverage_v1() -> None:
    """The v1 catalog covers the classes Phase 11a is responsible for.

    If you intentionally remove an entry, update this test. If you
    add a new entry, also add a Test*Layout/Lookup case above.
    """
    expected = {
        # 3D solids
        ("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress"),
        ("FourNodeTetrahedron", IntRule.Tet_GL_1, "strain"),
        ("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress"),
        ("TenNodeTetrahedron", IntRule.Tet_GL_2, "strain"),
        ("Brick", IntRule.Hex_GL_2, "stress"),
        ("Brick", IntRule.Hex_GL_2, "strain"),
        ("BbarBrick", IntRule.Hex_GL_2, "stress"),
        ("BbarBrick", IntRule.Hex_GL_2, "strain"),
        ("SSPbrick", IntRule.Hex_GL_1, "stress"),
        ("SSPbrick", IntRule.Hex_GL_1, "strain"),
        ("Twenty_Node_Brick", IntRule.Hex_GL_3, "stress"),
        ("Twenty_Node_Brick", IntRule.Hex_GL_3, "strain"),
        # 2D solids
        ("FourNodeQuad", IntRule.Quad_GL_2, "stress"),
        ("FourNodeQuad", IntRule.Quad_GL_2, "strain"),
        ("Tri31", IntRule.Triangle_GL_1, "stress"),
        ("Tri31", IntRule.Triangle_GL_1, "strain"),
        ("SSPquad", IntRule.Quad_GL_1, "stress"),
        ("SSPquad", IntRule.Quad_GL_1, "strain"),
        ("EightNodeQuad", IntRule.Quad_GL_3, "stress"),
        ("EightNodeQuad", IntRule.Quad_GL_3, "strain"),
        # Shells (8-component stress resultants per surface GP)
        ("ShellMITC4", IntRule.Quad_GL_2, "stress"),
        ("ShellMITC4", IntRule.Quad_GL_2, "strain"),
        ("ShellDKGQ", IntRule.Quad_GL_2, "stress"),
        ("ShellDKGQ", IntRule.Quad_GL_2, "strain"),
        ("ShellNLDKGQ", IntRule.Quad_GL_2, "stress"),
        ("ShellNLDKGQ", IntRule.Quad_GL_2, "strain"),
        ("ASDShellQ4", IntRule.Quad_GL_2, "stress"),
        ("ASDShellQ4", IntRule.Quad_GL_2, "strain"),
        ("ShellMITC9", IntRule.Quad_GL_3, "stress"),
        ("ShellMITC9", IntRule.Quad_GL_3, "strain"),
        ("ShellDKGT", IntRule.Triangle_GL_2C, "stress"),
        ("ShellDKGT", IntRule.Triangle_GL_2C, "strain"),
        ("ShellNLDKGT", IntRule.Triangle_GL_2C, "stress"),
        ("ShellNLDKGT", IntRule.Triangle_GL_2C, "strain"),
        ("ASDShellT3", IntRule.Triangle_GL_2B, "stress"),
        ("ASDShellT3", IntRule.Triangle_GL_2B, "strain"),
        # Trusses (1 GP at midpoint, scalar axial force)
        ("Truss", IntRule.Line_GL_1, "axial_force"),
        ("CorotTruss", IntRule.Line_GL_1, "axial_force"),
        ("Truss2", IntRule.Line_GL_1, "axial_force"),
        ("CorotTruss2", IntRule.Line_GL_1, "axial_force"),
        ("InertiaTruss", IntRule.Line_GL_1, "axial_force"),
    }
    assert set(RESPONSE_CATALOG.keys()) == expected


# =====================================================================
# ResponseLayout invariants
# =====================================================================

class TestLayoutInvariants:
    def test_natural_coords_count_matches_n_gp(self) -> None:
        for key, layout in RESPONSE_CATALOG.items():
            assert layout.natural_coords.shape[0] == layout.n_gauss_points, (
                f"{key}: natural_coords has {layout.natural_coords.shape[0]} "
                f"rows but n_gauss_points = {layout.n_gauss_points}"
            )

    def test_component_layout_length_matches(self) -> None:
        for key, layout in RESPONSE_CATALOG.items():
            assert len(layout.component_layout) == layout.n_components_per_gp, (
                f"{key}: component_layout has {len(layout.component_layout)} "
                f"names but n_components_per_gp = {layout.n_components_per_gp}"
            )

    def test_class_tag_consistent_per_class(self) -> None:
        """Every entry for a given class must use the same class_tag."""
        per_class: dict[str, int] = {}
        for (class_name, _rule, _token), layout in RESPONSE_CATALOG.items():
            existing = per_class.setdefault(class_name, layout.class_tag)
            assert existing == layout.class_tag, (
                f"{class_name}: catalog has inconsistent class_tag "
                f"({existing} vs {layout.class_tag})"
            )


# =====================================================================
# MPCO bracket-key parser
# =====================================================================

class TestParseMPCOElementKey:
    def test_model_form_no_header(self) -> None:
        k = parse_mpco_element_key("179-FourNodeTetrahedron[300:0]")
        assert k == MPCOElementKey(
            class_tag=179, class_name="FourNodeTetrahedron",
            int_rule=300, custom_rule_idx=0, header_idx=0,
        )

    def test_results_form_with_header(self) -> None:
        k = parse_mpco_element_key("31-ASDShellQ4[202:0:0]")
        assert k.class_tag == 31
        assert k.class_name == "ASDShellQ4"
        assert k.int_rule == 202
        assert k.custom_rule_idx == 0
        assert k.header_idx == 0

    def test_custom_rule_force_beam(self) -> None:
        k = parse_mpco_element_key("73-ForceBeamColumn3d[1000:3:0]")
        assert k.class_tag == 73
        assert k.class_name == "ForceBeamColumn3d"
        assert k.int_rule == 1000
        assert k.custom_rule_idx == 3
        assert k.is_custom_rule

    def test_class_name_with_underscores_and_digits(self) -> None:
        k = parse_mpco_element_key("256-TenNodeTetrahedron[301:0:0]")
        assert k.class_name == "TenNodeTetrahedron"
        assert k.int_rule == 301

    def test_malformed_raises(self) -> None:
        for bad in [
            "no-bracket",
            "179-FourNodeTetrahedron",
            "179-FourNodeTetrahedron[300]",   # missing custom idx
            "FourNodeTetrahedron[300:0:0]",   # missing tag
            "[300:0]",
            "",
        ]:
            with pytest.raises(ValueError):
                parse_mpco_element_key(bad)


# =====================================================================
# unflatten — the keystone shape transform
# =====================================================================

class TestUnflatten:
    def test_single_gp_tet(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        # T=2 steps, E_g=3 elements, flat_size=6 (1 GP × 6 components).
        flat = np.arange(2 * 3 * 6, dtype=np.float64).reshape(2, 3, 6)
        out = unflatten(flat, layout)

        # Six components, each shape (2, 3, 1).
        assert set(out.keys()) == set(layout.component_layout)
        for name in layout.component_layout:
            assert out[name].shape == (2, 3, 1)

        # GP slowest, component fastest: at element 0, time 0 the flat
        # array is [0, 1, 2, 3, 4, 5] which is exactly the 6 components
        # at GP 0.
        np.testing.assert_array_equal(out["stress_xx"][0, 0, 0], 0)
        np.testing.assert_array_equal(out["stress_yy"][0, 0, 0], 1)
        np.testing.assert_array_equal(out["stress_zz"][0, 0, 0], 2)
        np.testing.assert_array_equal(out["stress_xy"][0, 0, 0], 3)
        np.testing.assert_array_equal(out["stress_yz"][0, 0, 0], 4)
        np.testing.assert_array_equal(out["stress_xz"][0, 0, 0], 5)

    def test_multi_gp_tet(self) -> None:
        layout = lookup("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress")
        # T=2, E_g=2, flat_size = 4 GPs × 6 components = 24.
        T, E, K = 2, 2, layout.flat_size_per_element
        flat = np.arange(T * E * K, dtype=np.float64).reshape(T, E, K)
        out = unflatten(flat, layout)

        # Each component shape (T, E_g, n_GP) = (2, 2, 4).
        for name in layout.component_layout:
            assert out[name].shape == (T, E, 4), name

        # Verify GP-slowest layout for element 0, time 0:
        # flat[0, 0, :]   = 0..23
        # GP 0 components = 0..5    → stress_xx[0,0,0] == 0
        # GP 1 components = 6..11   → stress_xx[0,0,1] == 6
        # GP 2 components = 12..17  → stress_xx[0,0,2] == 12
        # GP 3 components = 18..23  → stress_xx[0,0,3] == 18
        np.testing.assert_array_equal(out["stress_xx"][0, 0], [0, 6, 12, 18])
        np.testing.assert_array_equal(out["stress_yy"][0, 0], [1, 7, 13, 19])
        np.testing.assert_array_equal(out["stress_xz"][0, 0], [5, 11, 17, 23])

    def test_unflatten_then_flatten_roundtrip(self) -> None:
        """Roundtrip: synthesize per-component, flatten, unflatten, equal."""
        layout = lookup("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress")
        T, E, G = 3, 5, layout.n_gauss_points
        rng = np.random.default_rng(42)
        components = {
            name: rng.standard_normal((T, E, G))
            for name in layout.component_layout
        }
        flat = flatten(components, layout)
        assert flat.shape == (T, E, layout.flat_size_per_element)

        decoded = unflatten(flat, layout)
        for name in layout.component_layout:
            np.testing.assert_array_equal(decoded[name], components[name])

    def test_flatten_then_unflatten_roundtrip_tet1(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        T, E, G = 4, 7, 1
        rng = np.random.default_rng(7)
        components = {
            name: rng.standard_normal((T, E, G))
            for name in layout.component_layout
        }
        decoded = unflatten(flatten(components, layout), layout)
        for name in layout.component_layout:
            np.testing.assert_array_equal(decoded[name], components[name])

    def test_unflatten_wrong_flat_size(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        bad = np.zeros((2, 3, 5))   # 5 ≠ 6
        with pytest.raises(ValueError, match="flat_size"):
            unflatten(bad, layout)

    def test_unflatten_wrong_ndim(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        bad = np.zeros((6,))
        with pytest.raises(ValueError, match="3-D flat array"):
            unflatten(bad, layout)

    def test_flatten_missing_component(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        # Missing stress_xz.
        comps = {
            "stress_xx": np.zeros((1, 1, 1)),
            "stress_yy": np.zeros((1, 1, 1)),
            "stress_zz": np.zeros((1, 1, 1)),
            "stress_xy": np.zeros((1, 1, 1)),
            "stress_yz": np.zeros((1, 1, 1)),
        }
        with pytest.raises(ValueError, match="missing"):
            flatten(comps, layout)

    def test_flatten_extra_component(self) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        comps = {name: np.zeros((1, 1, 1)) for name in layout.component_layout}
        comps["stress_extra"] = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="extra"):
            flatten(comps, layout)
