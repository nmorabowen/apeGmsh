"""
Tests for ``LoadsComposite.point_closest`` — coordinate-driven point loads.

These exercise the snap helper + resolver dispatch without requiring a
live Gmsh model.  The ``within=`` path is tested by monkeypatching
``_target_nodes`` on the composite, since it normally calls Gmsh.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from apeGmsh.core.LoadsComposite import LoadsComposite
from apeGmsh.solvers.Loads import (
    LoadResolver,
    PointClosestLoadDef,
)


def _resolver(coords_by_tag):
    tags = np.array(sorted(coords_by_tag), dtype=np.int64)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return LoadResolver(tags, coords)


def _composite():
    """LoadsComposite with no parent — fine as long as we don't call
    methods that need session access (e.g. _target_nodes with within=)."""
    return LoadsComposite(parent=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

class TestFactory(unittest.TestCase):

    def test_stores_xyz_and_force(self):
        c = _composite()
        d = c.point_closest((1.0, 2.0, 3.0), force_xyz=(10.0, 0.0, -5.0))
        self.assertIsInstance(d, PointClosestLoadDef)
        self.assertEqual(d.xyz_request, (1.0, 2.0, 3.0))
        self.assertEqual(d.target, (1.0, 2.0, 3.0))
        self.assertEqual(d.target_source, "closest_xyz")
        self.assertEqual(d.force_xyz, (10.0, 0.0, -5.0))
        self.assertIsNone(d.within)
        self.assertIsNone(d.tol)
        self.assertIsNone(d.snap_distance)

    def test_requires_force_or_moment(self):
        c = _composite()
        with self.assertRaises(ValueError):
            c.point_closest((0.0, 0.0, 0.0))

    def test_within_kwargs_are_interchangeable(self):
        c = _composite()
        d1 = c.point_closest((0, 0, 0), within="my_pg",
                             force_xyz=(1.0, 0.0, 0.0))
        d2 = c.point_closest((0, 0, 0), pg="my_pg",
                             force_xyz=(1.0, 0.0, 0.0))
        d3 = c.point_closest((0, 0, 0), label="my_label",
                             force_xyz=(1.0, 0.0, 0.0))
        self.assertEqual(d1.within, "my_pg")
        self.assertEqual(d1.within_source, "auto")
        self.assertEqual(d2.within, "my_pg")
        self.assertEqual(d2.within_source, "pg")
        self.assertEqual(d3.within, "my_label")
        self.assertEqual(d3.within_source, "label")

    def test_pattern_inherited_from_active_context(self):
        c = _composite()
        with c.pattern("live"):
            d = c.point_closest((0, 0, 0), force_xyz=(1.0, 0.0, 0.0))
        self.assertEqual(d.pattern, "live")


# ---------------------------------------------------------------------
# Snap helper
# ---------------------------------------------------------------------

class TestSnapNodeXYZ(unittest.TestCase):

    def setUp(self):
        # 4 nodes at corners of a unit square (z=0)
        self.coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (1.0, 1.0, 0.0),
        }
        self.r = _resolver(self.coords)
        self.c = _composite()

    def test_global_snap_picks_nearest(self):
        nids, snap = self.c._snap_node_xyz(
            (0.1, 0.05, 0.0), within=None, within_source="auto",
            tol=None, resolver=self.r, node_map=None,
        )
        self.assertEqual(nids, [1])
        self.assertAlmostEqual(snap, np.hypot(0.1, 0.05), places=12)

    def test_exact_hit_zero_distance(self):
        nids, snap = self.c._snap_node_xyz(
            (1.0, 1.0, 0.0), within=None, within_source="auto",
            tol=None, resolver=self.r, node_map=None,
        )
        self.assertEqual(nids, [4])
        self.assertEqual(snap, 0.0)

    def test_tol_returns_all_within_radius(self):
        # (0.5, 0, 0) is 0.5 from nodes 1 and 2, ~1.118 from 3,4
        nids, snap = self.c._snap_node_xyz(
            (0.5, 0.0, 0.0), within=None, within_source="auto",
            tol=0.6, resolver=self.r, node_map=None,
        )
        self.assertEqual(sorted(nids), [1, 2])
        self.assertAlmostEqual(snap, 0.5, places=12)

    def test_tol_no_match_raises(self):
        with self.assertRaises(ValueError):
            self.c._snap_node_xyz(
                (10.0, 10.0, 10.0), within=None, within_source="auto",
                tol=0.1, resolver=self.r, node_map=None,
            )

    def test_within_restricts_pool(self):
        # Force the global nearest to be outside the within set.
        # (0.1, 0.05, 0) globally → node 1, but if within={3,4} → node 3.
        self.c._target_nodes = lambda *a, **kw: {3, 4}  # type: ignore[assignment]
        nids, _snap = self.c._snap_node_xyz(
            (0.1, 0.05, 0.0), within="x", within_source="auto",
            tol=None, resolver=self.r, node_map=None,
        )
        self.assertEqual(nids, [3])

    def test_within_empty_raises(self):
        self.c._target_nodes = lambda *a, **kw: set()  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            self.c._snap_node_xyz(
                (0.0, 0.0, 0.0), within="x", within_source="auto",
                tol=None, resolver=self.r, node_map=None,
            )


# ---------------------------------------------------------------------
# Full resolve path (composite.resolve)
# ---------------------------------------------------------------------

class TestResolveDispatch(unittest.TestCase):

    def setUp(self):
        self.coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
        }
        tags = np.array([1, 2], dtype=np.int64)
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        self.tags = tags
        self.coords_arr = coords

    def test_exact_hit_no_warning_force_applied(self):
        c = _composite()
        c.point_closest((1.0, 0.0, 0.0), force_xyz=(0.0, 0.0, -10.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ls = c.resolve(self.tags, self.coords_arr)
        self.assertEqual(len(w), 0)
        recs = list(ls)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].node_id, 2)
        self.assertEqual(recs[0].force_xyz, (0.0, 0.0, -10.0))
        # snap_distance back-written on def
        self.assertEqual(c.load_defs[0].snap_distance, 0.0)

    def test_offset_emits_warning_and_records_distance(self):
        c = _composite()
        c.point_closest((1.0, 0.5, 0.0), force_xyz=(0.0, 0.0, -1.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.resolve(self.tags, self.coords_arr)
        self.assertTrue(any("snapped to node" in str(x.message) for x in w))
        self.assertAlmostEqual(c.load_defs[0].snap_distance, 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
