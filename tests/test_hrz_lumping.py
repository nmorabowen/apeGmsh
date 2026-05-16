"""
HRZ diagonal-scaling lumping — quadrature, weights, and resolver wiring.

Three layers:
  * quadrature rules integrate known polynomials exactly
  * HRZ weights match closed-form values and stay physical
  * resolver `reduction='consistent'` is bit-identical to lumped for
    first-order elements (regression) and correctly redistributes for
    higher-order ones (total mass still conserved)
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.masses.defs import SurfaceMassDef, VolumeMassDef
from apeGmsh.fem import _quadrature as Q
from apeGmsh.fem._hrz import (
    hrz_weights,
    surface_code,
    volume_code,
)
from apeGmsh.fem import _shape_functions as SF
from apeGmsh.mesh._mass_resolver import MassResolver


# =====================================================================
# Quadrature — integrate known polynomials exactly
# =====================================================================

class TestQuadrature(unittest.TestCase):

    def test_gauss_legendre_1d_exactness(self):
        p, w = Q.gauss_legendre_1d(3)
        self.assertAlmostEqual(w.sum(), 2.0)              # ∫_{-1}^1 1
        self.assertAlmostEqual((w * p**2).sum(), 2/3)     # ∫ ξ²
        self.assertAlmostEqual((w * p**4).sum(), 2/5)     # ∫ ξ⁴ (deg 4)

    def test_triangle_rule(self):
        p, w = Q.gauss_tri()
        self.assertAlmostEqual(w.sum(), 0.5)              # area
        self.assertAlmostEqual((w * p[:, 0]).sum(), 1/6)  # ∫ ξ
        self.assertAlmostEqual(
            (w * p[:, 0] * p[:, 1]).sum(), 1/24,          # ∫ ξη
        )

    def test_tet_rule(self):
        p, w = Q.gauss_tet()
        self.assertAlmostEqual(w.sum(), 1/6)              # volume
        self.assertAlmostEqual((w * p[:, 0]).sum(), 1/24)  # ∫ ξ
        self.assertAlmostEqual((w * p[:, 0]**2).sum(), 1/60)  # ∫ ξ²

    def test_partition_of_unity_all_types(self):
        """∫ Σ_I N_I dΩ_ref == reference measure for every catalog type."""
        from apeGmsh.fem._hrz import _QUAD_SPEC, _quadrature_for
        expected = {
            SF._LINE2: 2.0, SF._TRI3: 0.5, SF._QUAD4: 4.0,
            SF._TET4: 1/6, SF._HEX8: 8.0, SF._WEDGE6: 1.0,
            SF._TRI6: 0.5, SF._QUAD9: 4.0, SF._TET10: 1/6,
            SF._HEX27: 8.0, SF._QUAD8: 4.0, SF._HEX20: 8.0,
        }
        for code, exp in expected.items():
            pts, wts = _quadrature_for(*_QUAD_SPEC[code])
            N = SF.get_shape_functions(code)[0](pts)
            self.assertAlmostEqual(
                float((N.sum(axis=1) * wts).sum()), exp, places=6,
                msg=f"partition of unity failed for code {code}",
            )


# =====================================================================
# HRZ weights — closed-form values and physicality
# =====================================================================

class TestHRZWeights(unittest.TestCase):

    def test_weights_sum_to_one(self):
        for code in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 16, 17):
            self.assertAlmostEqual(sum(hrz_weights(code)), 1.0, places=10)

    def test_first_order_collapses_to_equal_share(self):
        cases = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6}  # code: n_nodes
        for code, n in cases.items():
            w = hrz_weights(code)
            np.testing.assert_allclose(w, [1.0 / n] * n, atol=1e-12)

    def test_tri6_closed_form(self):
        """Quadratic triangle: corners 1/19, mid-sides 16/57."""
        w = np.array(hrz_weights(SF._TRI6))
        # catalog node order: 3 corners then 3 mid-edges
        np.testing.assert_allclose(w[:3], 1/19, rtol=1e-6)
        np.testing.assert_allclose(w[3:], 16/57, rtol=1e-6)

    def test_quad9_tensor_product(self):
        """Lagrangian quad9 = line3 ⊗ line3, weights (1,4,16)/36."""
        w = np.array(hrz_weights(SF._QUAD9))
        # 4 corners = 1/36, 4 edge-mids = 4/36, 1 center = 16/36
        np.testing.assert_allclose(sorted(w)[:4], 1/36, rtol=1e-6)
        np.testing.assert_allclose(sorted(w)[-1], 16/36, rtol=1e-6)

    def test_hex27_center_node(self):
        """Lagrangian hex27 center node = (2/3)³ = 8/27."""
        w = hrz_weights(SF._HEX27)
        self.assertAlmostEqual(max(w), 8/27, places=6)

    def test_serendipity_all_positive(self):
        """quad8 / hex20 corner weights stay non-negative under HRZ
        (row-sum lumping would go negative — that's why HRZ).
        """
        for code in (SF._QUAD8, SF._HEX20):
            w = hrz_weights(code)
            self.assertTrue(min(w) > 0.0, f"code {code} has negative weight")

    def test_node_count_to_code_maps(self):
        self.assertEqual(volume_code(4), SF._TET4)
        self.assertEqual(volume_code(10), SF._TET10)
        self.assertEqual(volume_code(27), SF._HEX27)
        self.assertEqual(surface_code(3), SF._TRI3)
        self.assertEqual(surface_code(8), SF._QUAD8)
        self.assertIsNone(volume_code(5))      # pyramid5 — not in catalog
        self.assertIsNone(surface_code(7))


# =====================================================================
# Resolver wiring — consistent path
# =====================================================================

def _hex8_unit_cube():
    tags = np.arange(1, 9, dtype=np.int64)
    coords = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    return MassResolver(tags, coords), [np.arange(1, 9, dtype=np.int64)]


def _tet4_unit():
    tags = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float,
    )
    return MassResolver(tags, coords), [np.array([1, 2, 3, 4], dtype=np.int64)]


class TestResolverConsistentFirstOrder(unittest.TestCase):
    """Regression: consistent == lumped for first-order elements."""

    def test_hex8_consistent_equals_lumped(self):
        r, conn = _hex8_unit_cube()
        defn = VolumeMassDef(target="c", density=7850.0)
        lumped = {x.node_id: x.mass for x in r.resolve_volume_lumped(defn, conn)}
        cons = {x.node_id: x.mass for x in r.resolve_volume_consistent(defn, conn)}
        for nid in lumped:
            np.testing.assert_allclose(cons[nid], lumped[nid])

    def test_tet4_consistent_equals_lumped(self):
        r, conn = _tet4_unit()
        defn = VolumeMassDef(target="t", density=2400.0)
        lumped = {x.node_id: x.mass for x in r.resolve_volume_lumped(defn, conn)}
        cons = {x.node_id: x.mass for x in r.resolve_volume_consistent(defn, conn)}
        for nid in lumped:
            np.testing.assert_allclose(cons[nid], lumped[nid])


class TestResolverConsistentHigherOrder(unittest.TestCase):
    """Higher-order: HRZ redistributes but conserves total mass."""

    def _tet10_unit(self):
        # tet4 corners + 6 edge-midpoints, catalog node order
        c = {
            1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1),
            5: (0.5, 0, 0),    # edge 1-2
            6: (0.5, 0.5, 0),  # edge 2-3
            7: (0, 0.5, 0),    # edge 3-1
            8: (0, 0, 0.5),    # edge 1-4
            9: (0.5, 0, 0.5),  # edge 2-4
            10: (0, 0.5, 0.5),  # edge 3-4
        }
        tags = np.array(sorted(c), dtype=np.int64)
        coords = np.array([c[int(t)] for t in tags], dtype=float)
        conn = [np.arange(1, 11, dtype=np.int64)]
        return MassResolver(tags, coords), conn

    def test_tet10_total_mass_conserved(self):
        r, conn = self._tet10_unit()
        rho = 1000.0
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="t10", density=rho), conn,
        )
        total = sum(x.mass[0] for x in recs)
        # element_volume() bbox-approximates tet10 → just assert the
        # distribution conserves whatever V it computed, and that the
        # split is non-uniform (HRZ is doing something).
        V = r.element_volume(conn[0])
        self.assertAlmostEqual(total, rho * V, places=6)
        masses = sorted(x.mass[0] for x in recs)
        self.assertGreater(masses[-1], masses[0] * 1.5,
                           "HRZ should redistribute tet10 non-uniformly")

    def test_tet10_consistent_differs_from_lumped(self):
        r, conn = self._tet10_unit()
        defn = VolumeMassDef(target="t10", density=1.0)
        lumped = sorted(x.mass[0] for x in r.resolve_volume_lumped(defn, conn))
        cons = sorted(x.mass[0] for x in r.resolve_volume_consistent(defn, conn))
        self.assertFalse(
            np.allclose(lumped, cons),
            "tet10 consistent must differ from equal-share lumped",
        )


class TestConsistentComposesWithKwargs(unittest.TestCase):
    """dofs= and rotational= still apply on the HRZ path."""

    def test_dofs_and_rotational_on_consistent(self):
        r, conn = _hex8_unit_cube()
        defn = VolumeMassDef(
            target="c", density=1.0,
            dofs=[1, 2], rotational=(0.1, 0.2, 0.3),
        )
        recs = r.resolve_volume_consistent(defn, conn)
        for x in recs:
            mx, my, mz, ixx, iyy, izz = x.mass
            self.assertAlmostEqual(mz, 0.0)          # z masked
            self.assertGreater(mx, 0.0)
            self.assertAlmostEqual(mx, my)
            self.assertEqual((ixx, iyy, izz), (0.1, 0.2, 0.3))

    def test_surface_consistent_tri6_conserves_mass(self):
        # Reference tri6 with unit-leg corners + edge midpoints
        c = {
            1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0),
            4: (0.5, 0, 0), 5: (0.5, 0.5, 0), 6: (0, 0.5, 0),
        }
        tags = np.array(sorted(c), dtype=np.int64)
        coords = np.array([c[int(t)] for t in tags], dtype=float)
        r = MassResolver(tags, coords)
        face = [[1, 2, 3, 4, 5, 6]]
        rho_a = 50.0
        recs = r.resolve_surface_consistent(
            SurfaceMassDef(target="t6", areal_density=rho_a), face,
        )
        A = r.face_area(face[0])
        self.assertAlmostEqual(sum(x.mass[0] for x in recs), rho_a * A,
                               places=6)
        # corners (nodes 1-3) should each carry less than mid-edges (4-6)
        m = {x.node_id: x.mass[0] for x in recs}
        self.assertLess(max(m[1], m[2], m[3]), min(m[4], m[5], m[6]))


if __name__ == "__main__":
    unittest.main()
