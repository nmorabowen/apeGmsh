"""
Unit tests for :class:`apeGmsh.mesh._mass_resolver.MassResolver`.
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.masses.defs import (
    LineMassDef,
    PointMassDef,
    SurfaceMassDef,
    VolumeMassDef,
)
from apeGmsh._kernel.resolvers._mass_resolver import MassResolver
from apeGmsh._kernel.records import MassRecord


def _resolver(coords_by_tag, ndf=6):
    tags = np.array(sorted(coords_by_tag), dtype=np.int64)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return MassResolver(tags, coords, ndf=ndf)


def _by_node(records):
    return {rec.node_id: np.array(rec.mass) for rec in records}


# =====================================================================
# resolve_point_lumped
# =====================================================================

class TestResolvePointLumped(unittest.TestCase):

    def test_translational_only(self):
        r = _resolver({1: (0, 0, 0), 2: (1, 0, 0)})
        defn = PointMassDef(target="pt", mass=3.0)
        records = r.resolve_point_lumped(defn, {1, 2})
        self.assertEqual(len(records), 2)
        for rec in records:
            self.assertIsInstance(rec, MassRecord)
            self.assertEqual(rec.mass, (3.0, 3.0, 3.0, 0.0, 0.0, 0.0))

    def test_with_rotational_inertia(self):
        r = _resolver({1: (0, 0, 0)})
        defn = PointMassDef(
            target="pt", mass=2.0, rotational=(0.1, 0.2, 0.3),
        )
        recs = r.resolve_point_lumped(defn, {1})
        self.assertEqual(recs[0].mass, (2.0, 2.0, 2.0, 0.1, 0.2, 0.3))

    def test_consistent_equals_lumped_for_point(self):
        r = _resolver({1: (0, 0, 0)})
        defn = PointMassDef(target="pt", mass=5.0)
        lumped = r.resolve_point_lumped(defn, {1})
        consistent = r.resolve_point_consistent(defn, {1})
        self.assertEqual(lumped[0].mass, consistent[0].mass)


# =====================================================================
# resolve_line_lumped
# =====================================================================

class TestResolveLineLumped(unittest.TestCase):

    def test_single_edge_half_split(self):
        # Length 4, linear density 10 → total 40 → 20 each end.
        r = _resolver({1: (0, 0, 0), 2: (4, 0, 0)})
        defn = LineMassDef(target="edge", linear_density=10.0)
        records = r.resolve_line_lumped(defn, edges=[(1, 2)])
        m = _by_node(records)
        np.testing.assert_allclose(m[1][:3], [20.0, 20.0, 20.0])
        np.testing.assert_allclose(m[2][:3], [20.0, 20.0, 20.0])

    def test_interior_node_gets_double_share(self):
        r = _resolver({1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)})
        defn = LineMassDef(target="edges", linear_density=2.0)
        records = r.resolve_line_lumped(defn, edges=[(1, 2), (2, 3)])
        m = _by_node(records)
        # Each edge L=1, share = ρ·L/2 = 1 per end
        np.testing.assert_allclose(m[1][0], 1.0)
        np.testing.assert_allclose(m[2][0], 2.0)
        np.testing.assert_allclose(m[3][0], 1.0)

    def test_consistent_matches_lumped_for_2node(self):
        # Consistent diagonal sum = ρL/2 = lumped per endpoint
        r = _resolver({1: (0, 0, 0), 2: (3, 0, 0)})
        defn = LineMassDef(target="edge", linear_density=4.0)
        lumped = _by_node(r.resolve_line_lumped(defn, [(1, 2)]))
        consistent = _by_node(r.resolve_line_consistent(defn, [(1, 2)]))
        for nid in (1, 2):
            np.testing.assert_allclose(consistent[nid], lumped[nid])


# =====================================================================
# resolve_surface_lumped
# =====================================================================

class TestResolveSurfaceLumped(unittest.TestCase):

    def test_unit_square(self):
        coords = {
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (1, 1, 0), 4: (0, 1, 0),
        }
        r = _resolver(coords)
        defn = SurfaceMassDef(target="face", areal_density=8.0)
        records = r.resolve_surface_lumped(defn, faces=[[1, 2, 3, 4]])
        m = _by_node(records)
        # A=1, total mass = 8, split /4 → 2 per node
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(m[nid][:3], [2.0, 2.0, 2.0])
            np.testing.assert_allclose(m[nid][3:], [0, 0, 0])

    def test_degenerate_face_ignored(self):
        r = _resolver({1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)})
        defn = SurfaceMassDef(target="face", areal_density=1.0)
        records = r.resolve_surface_lumped(defn, [[1, 2, 3]])
        self.assertEqual(records, [])

    def test_two_triangles_accumulate_on_shared_node(self):
        # Two unit-area triangles sharing an edge — shared nodes get
        # contribution from both.
        coords = {
            1: (0, 0, 0),           # shared
            2: (2, 0, 0),           # shared
            3: (0, 1, 0),           # in tri A only
            4: (2, 1, 0),           # in tri B only
        }
        r = _resolver(coords)
        defn = SurfaceMassDef(target="face", areal_density=3.0)
        records = r.resolve_surface_lumped(
            defn, faces=[[1, 2, 3], [1, 2, 4]],
        )
        m = _by_node(records)
        # Each triangle: base=2, height=1 → A=1; total m = 3, /3 nodes = 1 per
        np.testing.assert_allclose(m[1][0], 2.0)   # two triangles
        np.testing.assert_allclose(m[2][0], 2.0)
        np.testing.assert_allclose(m[3][0], 1.0)
        np.testing.assert_allclose(m[4][0], 1.0)


# =====================================================================
# resolve_volume_lumped
# =====================================================================

class TestResolveVolumeLumped(unittest.TestCase):

    def test_unit_tet(self):
        r = _resolver({
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (0, 1, 0), 4: (0, 0, 1),
        })
        defn = VolumeMassDef(target="vol", density=24.0)
        records = r.resolve_volume_lumped(
            defn, elements=[np.array([1, 2, 3, 4])],
        )
        m = _by_node(records)
        # V = 1/6, total mass = 24 * 1/6 = 4, /4 → 1 each
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(m[nid][:3], [1.0, 1.0, 1.0])

    def test_unit_hex(self):
        r = _resolver({
            1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
            5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1),
        })
        defn = VolumeMassDef(target="vol", density=16.0)
        records = r.resolve_volume_lumped(
            defn, elements=[np.array([1, 2, 3, 4, 5, 6, 7, 8])],
        )
        m = _by_node(records)
        # V=1, total mass=16, /8 → 2 per node
        for nid in range(1, 9):
            np.testing.assert_allclose(m[nid][0], 2.0)

    def test_consistent_matches_lumped_for_tet(self):
        r = _resolver({
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (0, 1, 0), 4: (0, 0, 1),
        })
        defn = VolumeMassDef(target="vol", density=24.0)
        lumped = _by_node(r.resolve_volume_lumped(
            defn, [np.array([1, 2, 3, 4])],
        ))
        consistent = _by_node(r.resolve_volume_consistent(
            defn, [np.array([1, 2, 3, 4])],
        ))
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(consistent[nid], lumped[nid])


if __name__ == "__main__":
    unittest.main()
