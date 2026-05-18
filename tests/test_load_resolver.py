"""
Unit tests for :class:`apeGmsh.mesh._load_resolver.LoadResolver`.

The resolver is pure numpy math (no Gmsh).  Tests use hand-built
synthetic meshes: a single beam edge, a single square face, a unit
tet, and a unit hex.
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.loads.defs import (
    BodyLoadDef,
    GravityLoadDef,
    LineLoadDef,
    PointLoadDef,
    SurfaceLoadDef,
)
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver
from apeGmsh._kernel.records import ElementLoadRecord, NodalLoadRecord


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _resolver(coords_by_tag, elem_tags=None, connectivity=None):
    tags = np.array(sorted(coords_by_tag), dtype=np.int64)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return LoadResolver(
        tags, coords, elem_tags=elem_tags,
        connectivity=connectivity,
    )


def _vec6(rec: NodalLoadRecord) -> np.ndarray:
    """Flatten a record into a length-6 array for hand comparisons."""
    out = np.zeros(6, dtype=float)
    if rec.force_xyz  is not None: out[:3] = rec.force_xyz
    if rec.moment_xyz is not None: out[3:] = rec.moment_xyz
    return out


def _by_node(records):
    return {rec.node_id: _vec6(rec) for rec in records}


# =====================================================================
# resolve_point
# =====================================================================

class TestResolvePoint(unittest.TestCase):

    def test_applies_force_to_all_nodes(self):
        coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)}
        r = _resolver(coords)
        defn = PointLoadDef(
            target="pt",
            force_xyz=(10.0, 0.0, -5.0),
        )
        records = r.resolve_point(defn, node_set={1, 2, 3})
        self.assertEqual(len(records), 3)
        for rec in records:
            self.assertIsInstance(rec, NodalLoadRecord)
            self.assertEqual(rec.pattern, "default")
            self.assertEqual(rec.force_xyz, (10.0, 0.0, -5.0))
            self.assertIsNone(rec.moment_xyz)

    def test_moment_carried_through(self):
        coords = {1: (0, 0, 0)}
        r = _resolver(coords)
        defn = PointLoadDef(
            target="n",
            moment_xyz=(0.0, 0.0, 7.5),
        )
        recs = r.resolve_point(defn, {1})
        self.assertIsNone(recs[0].force_xyz)
        self.assertEqual(recs[0].moment_xyz, (0.0, 0.0, 7.5))


# =====================================================================
# resolve_line_tributary
# =====================================================================

class TestResolveLineTributary(unittest.TestCase):

    def test_single_edge_splits_equally(self):
        # Edge from (0,0,0) to (4,0,0), length 4
        coords = {1: (0, 0, 0), 2: (4, 0, 0)}
        r = _resolver(coords)
        defn = LineLoadDef(
            target="edge",
            magnitude=10.0,
            direction=(0.0, 0.0, -1.0),
        )
        records = r.resolve_line_tributary(defn, edges=[(1, 2)])
        forces = _by_node(records)
        # q*L/2 = 10*4/2 = 20 per end node, in -z
        expected = np.array([0, 0, -20.0, 0, 0, 0])
        np.testing.assert_allclose(forces[1], expected)
        np.testing.assert_allclose(forces[2], expected)

    def test_interior_node_gets_double_share(self):
        # Chain of two edges sharing node 2 → node 2 gets 2x the share
        coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)}
        r = _resolver(coords)
        defn = LineLoadDef(
            target="edges",
            q_xyz=(0.0, 0.0, -1.0),
        )
        records = r.resolve_line_tributary(defn, edges=[(1, 2), (2, 3)])
        forces = _by_node(records)
        # Each edge length 1 → half-length 0.5 per endpoint.
        np.testing.assert_allclose(forces[1][2], -0.5)
        np.testing.assert_allclose(forces[2][2], -1.0)      # sum of both
        np.testing.assert_allclose(forces[3][2], -0.5)

    def test_direction_string_resolves(self):
        coords = {1: (0, 0, 0), 2: (2, 0, 0)}
        r = _resolver(coords)
        defn = LineLoadDef(
            target="edge", magnitude=3.0, direction="y",
        )
        records = r.resolve_line_tributary(defn, [(1, 2)])
        forces = _by_node(records)
        # q*L/2 = 3*2/2 = 3 in y
        np.testing.assert_allclose(forces[1][1], 3.0)
        np.testing.assert_allclose(forces[2][1], 3.0)


# =====================================================================
# resolve_surface_tributary
# =====================================================================

class TestResolveSurfaceTributary(unittest.TestCase):

    def test_quad_area_and_split(self):
        # Unit square in the XY plane, normal = +z
        coords = {
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (1, 1, 0), 4: (0, 1, 0),
        }
        r = _resolver(coords)
        defn = SurfaceLoadDef(
            target="face", magnitude=8.0, normal=True,
        )
        records = r.resolve_surface_tributary(defn, faces=[[1, 2, 3, 4]])
        forces = _by_node(records)
        # area=1, mag=8, normal=+z → -magnitude*A*n = (0,0,-8)
        # 4 nodes → -2 per node in z
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(forces[nid][:3], [0, 0, -2.0])

    def test_explicit_direction(self):
        coords = {
            1: (0, 0, 0), 2: (2, 0, 0),
            3: (2, 2, 0), 4: (0, 2, 0),
        }
        r = _resolver(coords)
        defn = SurfaceLoadDef(
            target="face",
            magnitude=5.0,
            normal=False,
            direction=(1.0, 0.0, 0.0),
        )
        records = r.resolve_surface_tributary(defn, [[1, 2, 3, 4]])
        forces = _by_node(records)
        # area=4, total Fx = 5*4 = 20, split /4 → 5 per node
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(forces[nid][:3], [5.0, 0, 0])

    def test_degenerate_face_ignored(self):
        coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)}  # colinear
        r = _resolver(coords)
        defn = SurfaceLoadDef(
            target="face", magnitude=1.0, normal=False,
            direction=(0, 0, 1),
        )
        records = r.resolve_surface_tributary(defn, [[1, 2, 3]])
        self.assertEqual(records, [])


# =====================================================================
# resolve_gravity_tributary
# =====================================================================

class TestResolveGravityTributary(unittest.TestCase):

    def test_unit_tet(self):
        # Tet with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1) → V = 1/6
        coords = {
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (0, 1, 0), 4: (0, 0, 1),
        }
        r = _resolver(coords)
        defn = GravityLoadDef(
            target="vol",
            g=(0.0, 0.0, -10.0),
            density=6.0,    # → ρ·V = 1.0
        )
        records = r.resolve_gravity_tributary(
            defn,
            elements=[np.array([1, 2, 3, 4])],
        )
        forces = _by_node(records)
        # total f3 = ρVg = 1*(0,0,-10) = (0,0,-10); /4 nodes → (0,0,-2.5)
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(forces[nid][:3], [0, 0, -2.5])

    def test_missing_density_raises(self):
        r = _resolver({1: (0, 0, 0), 2: (1, 0, 0),
                       3: (0, 1, 0), 4: (0, 0, 1)})
        defn = GravityLoadDef(target="vol", density=None)
        with self.assertRaises(ValueError):
            r.resolve_gravity_tributary(defn, [np.array([1, 2, 3, 4])])


# =====================================================================
# resolve_body_tributary
# =====================================================================

class TestResolveBodyTributary(unittest.TestCase):

    def test_unit_tet_body_force(self):
        coords = {
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (0, 1, 0), 4: (0, 0, 1),
        }
        r = _resolver(coords)
        defn = BodyLoadDef(
            target="vol",
            force_per_volume=(12.0, 0.0, 0.0),   # V=1/6 → total fx=2
        )
        records = r.resolve_body_tributary(
            defn, elements=[np.array([1, 2, 3, 4])],
        )
        forces = _by_node(records)
        # 2.0 split among 4 nodes → 0.5 each
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(forces[nid][:3], [0.5, 0, 0])


# =====================================================================
# Element-form output
# =====================================================================

class TestElementFormOutput(unittest.TestCase):

    def test_resolve_line_element_emits_beamUniform(self):
        r = _resolver({1: (0, 0, 0), 2: (1, 0, 0)})
        defn = LineLoadDef(
            target="edge", q_xyz=(0.0, -5.0, 0.0),
            pattern="live",
        )
        records = r.resolve_line_element(defn, element_ids=[100, 101])
        self.assertEqual(len(records), 2)
        for rec, eid in zip(records, [100, 101]):
            self.assertIsInstance(rec, ElementLoadRecord)
            self.assertEqual(rec.element_id, eid)
            self.assertEqual(rec.load_type, "beamUniform")
            self.assertEqual(rec.params,
                             {"wx": 0.0, "wy": -5.0, "wz": 0.0})
            self.assertEqual(rec.pattern, "live")

    def test_resolve_surface_element_emits_surfacePressure(self):
        r = _resolver({1: (0, 0, 0)})
        defn = SurfaceLoadDef(
            target="face", magnitude=2.5, normal=True,
        )
        records = r.resolve_surface_element(defn, element_ids=[7])
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.load_type, "surfacePressure")
        self.assertAlmostEqual(rec.params["p"], 2.5)
        self.assertTrue(rec.params["normal"])

    def test_resolve_gravity_element_emits_bodyForce(self):
        r = _resolver({1: (0, 0, 0)})
        defn = GravityLoadDef(
            target="vol", g=(0, 0, -9.81), density=2400.0,
        )
        records = r.resolve_gravity_element(defn, element_ids=[1])
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.load_type, "bodyForce")
        self.assertEqual(rec.params["g"], (0.0, 0.0, -9.81))
        self.assertEqual(rec.params["density"], 2400.0)


# =====================================================================
# Geometry helpers
# =====================================================================

class TestGeometryHelpers(unittest.TestCase):

    def test_face_area_square(self):
        r = _resolver({
            1: (0, 0, 0), 2: (3, 0, 0),
            3: (3, 2, 0), 4: (0, 2, 0),
        })
        self.assertAlmostEqual(r.face_area([1, 2, 3, 4]), 6.0)

    def test_face_normal_is_unit(self):
        r = _resolver({1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0)})
        n = r.face_normal([1, 2, 3])
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0)
        np.testing.assert_allclose(n, [0, 0, 1])

    def test_element_volume_tet(self):
        r = _resolver({
            1: (0, 0, 0), 2: (1, 0, 0),
            3: (0, 1, 0), 4: (0, 0, 1),
        })
        self.assertAlmostEqual(
            r.element_volume(np.array([1, 2, 3, 4])),
            1.0 / 6.0,
        )

    def test_element_volume_unit_hex(self):
        r = _resolver({
            1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
            5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1),
        })
        V = r.element_volume(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertAlmostEqual(V, 1.0)


if __name__ == "__main__":
    unittest.main()
