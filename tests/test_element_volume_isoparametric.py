"""
Isoparametric element-volume integration for higher-order solids.

`MassResolver.element_volume` keeps its exact analytic paths for tet4
(scalar triple product) and hex8 (6-tet decomposition), and now uses
isoparametric ``V = ∫_{Ω_ref} |J(ξ)| dξ`` for every other catalog
solid type (wedge6, tet10, hex20, hex27) instead of the old
bounding-box approximation.  Unknown element types still fall back to
the bbox last resort.

Node orderings below follow the Gmsh convention the shape-function
catalog assumes (see apeGmsh.fem._shape_functions).
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.masses.defs import VolumeMassDef
from apeGmsh.mesh._mass_resolver import MassResolver


def _vol(coords):
    tags = np.arange(1, len(coords) + 1, dtype=np.int64)
    r = MassResolver(tags, np.array(coords, dtype=float))
    return r.element_volume(tags)


# ---------------------------------------------------------------------------
# Canonical node sets (Gmsh ordering)
# ---------------------------------------------------------------------------

# tet10: 4 corners then edges 0-1, 1-2, 0-2, 0-3, 2-3, 1-3
def _tet10(scale=1.0):
    s = scale
    return [
        (0, 0, 0), (s, 0, 0), (0, s, 0), (0, 0, s),
        (s/2, 0, 0), (s/2, s/2, 0), (0, s/2, 0),
        (0, 0, s/2), (0, s/2, s/2), (s/2, 0, s/2),
    ]


def _hex20_box(lx, ly, lz):
    c = [
        (0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0),
        (0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz),
    ]
    e = [
        (lx/2, 0, 0), (0, ly/2, 0), (0, 0, lz/2), (lx, ly/2, 0),
        (lx, 0, lz/2), (lx/2, ly, 0), (lx, ly, lz/2), (0, ly, lz/2),
        (lx/2, 0, lz), (0, ly/2, lz), (lx, ly/2, lz), (lx/2, ly, lz),
    ]
    return c + e


def _hex27_cube(L=1.0):
    c = [
        (0, 0, 0), (L, 0, 0), (L, L, 0), (0, L, 0),
        (0, 0, L), (L, 0, L), (L, L, L), (0, L, L),
    ]
    e = [
        (L/2, 0, 0), (0, L/2, 0), (0, 0, L/2), (L, L/2, 0),
        (L, 0, L/2), (L/2, L, 0), (L, L, L/2), (0, L, L/2),
        (L/2, 0, L), (0, L/2, L), (L, L/2, L), (L/2, L, L),
    ]
    f = [
        (L/2, L/2, 0), (L/2, 0, L/2), (0, L/2, L/2),
        (L, L/2, L/2), (L/2, L, L/2), (L/2, L/2, L),
    ]
    ctr = [(L/2, L/2, L/2)]
    return c + e + f + ctr


def _wedge6(area_tri, h):
    # right triangle base of given area (legs a, a → area = a²/2), height h
    a = (2.0 * area_tri) ** 0.5
    return [
        (0, 0, 0), (a, 0, 0), (0, a, 0),
        (0, 0, h), (a, 0, h), (0, a, h),
    ]


# =====================================================================
# Exact analytic paths preserved (regression)
# =====================================================================

class TestAnalyticPathsUnchanged(unittest.TestCase):

    def test_tet4_unit(self):
        self.assertAlmostEqual(
            _vol([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]), 1/6,
        )

    def test_hex8_box(self):
        self.assertAlmostEqual(
            _vol([
                (0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0),
                (0, 0, 4), (2, 0, 4), (2, 3, 4), (0, 3, 4),
            ]),
            24.0,
        )


# =====================================================================
# Isoparametric paths — exact for affine higher-order elements
# =====================================================================

class TestIsoparametricExactForAffine(unittest.TestCase):

    def test_wedge6_prism(self):
        # triangle area 0.5, height 2 → V = 1.0
        self.assertAlmostEqual(_vol(_wedge6(0.5, 2.0)), 1.0, places=9)

    def test_tet10_straight_equals_tet4(self):
        self.assertAlmostEqual(_vol(_tet10(1.0)), 1/6, places=9)

    def test_tet10_affine_scaling(self):
        # scale ×2 → volume ×8
        self.assertAlmostEqual(_vol(_tet10(2.0)), 8.0 / 6.0, places=9)

    def test_hex20_box(self):
        self.assertAlmostEqual(_vol(_hex20_box(2.0, 3.0, 4.0)), 24.0,
                                places=9)

    def test_hex27_unit_cube(self):
        self.assertAlmostEqual(_vol(_hex27_cube(1.0)), 1.0, places=9)

    def test_hex27_affine_scaling(self):
        self.assertAlmostEqual(_vol(_hex27_cube(3.0)), 27.0, places=9)


# =====================================================================
# Unknown element type → bbox last resort still works
# =====================================================================

class TestUnknownTypeFallback(unittest.TestCase):

    def test_pyramid5_uses_bbox(self):
        # 5-node pyramid is not in the catalog → bbox of the points.
        coords = [
            (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (1, 1, 3),
        ]
        # bbox span = 2 × 2 × 3 = 12
        self.assertAlmostEqual(_vol(coords), 12.0)


# =====================================================================
# Resolver-level — consistent mass uses the isoparametric volume
# =====================================================================

class TestResolverUsesIsoparametricVolume(unittest.TestCase):

    def test_tet10_consistent_total_mass(self):
        coords = _tet10(1.0)
        tags = np.arange(1, 11, dtype=np.int64)
        r = MassResolver(tags, np.array(coords, dtype=float))
        conn = [tags]
        rho = 2400.0
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="t10", density=rho), conn,
        )
        # total mass == ρ · V_isoparametric == ρ · (1/6), exactly
        total = sum(x.mass[0] for x in recs)
        self.assertAlmostEqual(total, rho / 6.0, places=6)

    def test_wedge6_consistent_total_mass(self):
        coords = _wedge6(0.5, 2.0)            # V = 1.0
        tags = np.arange(1, 7, dtype=np.int64)
        r = MassResolver(tags, np.array(coords, dtype=float))
        rho = 1000.0
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="w6", density=rho), [tags],
        )
        self.assertAlmostEqual(sum(x.mass[0] for x in recs), rho * 1.0,
                               places=6)


if __name__ == "__main__":
    unittest.main()
