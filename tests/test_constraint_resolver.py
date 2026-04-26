"""
Unit tests for :class:`apeGmsh.solvers.Constraints.ConstraintResolver`.

The resolver is deliberately decoupled from Gmsh — it operates on raw
numpy arrays of node tags and coordinates.  These tests exercise it
with hand-built synthetic meshes so no Gmsh session is required.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from apeGmsh.solvers.Constraints import (
    ConstraintResolver,
    DistributingCouplingDef,
    EmbeddedDef,
    EqualDOFDef,
    InterpolationRecord,
    KinematicCouplingDef,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceDef,
    NodeToSurfaceRecord,
    PenaltyDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_resolver(coords_by_tag: dict[int, tuple[float, float, float]]):
    tags = np.array(sorted(coords_by_tag), dtype=int)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return ConstraintResolver(tags, coords)


# =====================================================================
# resolve_equal_dof
# =====================================================================

class TestResolveEqualDOF(unittest.TestCase):

    def test_matches_colocated_pairs(self):
        # Master: tags 1..3, slave: 11..13 at the SAME coords
        coords = {
            1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0),
            11: (0, 0, 0), 12: (1, 0, 0), 13: (2, 0, 0),
        }
        r = _make_resolver(coords)
        defn = EqualDOFDef(master_label="A", slave_label="B", dofs=[1, 2, 3])

        records = r.resolve_equal_dof(
            defn,
            master_nodes={1, 2, 3},
            slave_nodes={11, 12, 13},
        )

        self.assertEqual(len(records), 3)
        pairs = {(rec.master_node, rec.slave_node) for rec in records}
        self.assertEqual(pairs, {(1, 11), (2, 12), (3, 13)})
        for rec in records:
            self.assertEqual(rec.kind, "equal_dof")
            self.assertEqual(rec.dofs, [1, 2, 3])
            self.assertIsNone(rec.offset)

    def test_tolerance_excludes_far_nodes(self):
        coords = {
            1: (0, 0, 0),
            2: (1, 0, 0),
            11: (0.0, 0.0, 0.0),    # on top of master 1
            12: (1.0, 0.0, 0.01),   # 10 mm off master 2 → should match
            13: (5.0, 0.0, 0.0),    # far from any master → skipped
        }
        r = _make_resolver(coords)
        defn = EqualDOFDef(
            master_label="A", slave_label="B",
            tolerance=0.05,
        )

        records = r.resolve_equal_dof(
            defn, master_nodes={1, 2}, slave_nodes={11, 12, 13},
        )

        matched = {rec.slave_node for rec in records}
        self.assertEqual(matched, {11, 12})

    def test_default_dofs_is_all_six(self):
        coords = {1: (0, 0, 0), 2: (0, 0, 0)}
        r = _make_resolver(coords)
        defn = EqualDOFDef(master_label="A", slave_label="B")

        records = r.resolve_equal_dof(defn, {1}, {2})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].dofs, [1, 2, 3, 4, 5, 6])

    def test_empty_inputs(self):
        r = _make_resolver({1: (0, 0, 0)})
        defn = EqualDOFDef(master_label="A", slave_label="B")
        self.assertEqual(r.resolve_equal_dof(defn, set(), {1}), [])
        self.assertEqual(r.resolve_equal_dof(defn, {1}, set()), [])

    def test_equal_dof_pair_constraint_matrix_is_selection(self):
        rec = NodePairRecord(
            kind="equal_dof", master_node=1, slave_node=2, dofs=[1, 2, 3],
        )
        C = rec.constraint_matrix(ndof=6)
        expected = np.zeros((3, 6))
        expected[0, 0] = expected[1, 1] = expected[2, 2] = 1.0
        np.testing.assert_array_equal(C, expected)


# =====================================================================
# resolve_penalty
# =====================================================================

class TestResolvePenalty(unittest.TestCase):

    def test_carries_stiffness_and_dofs(self):
        coords = {1: (0, 0, 0), 2: (0, 0, 0)}
        r = _make_resolver(coords)
        defn = PenaltyDef(
            master_label="A", slave_label="B",
            stiffness=5.0e8, dofs=[1, 2],
        )

        records = r.resolve_penalty(defn, {1}, {2})
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.kind, "penalty")
        self.assertEqual(rec.penalty_stiffness, 5.0e8)
        self.assertEqual(rec.dofs, [1, 2])


# =====================================================================
# resolve_rigid_link
# =====================================================================

class TestResolveRigidLink(unittest.TestCase):

    def test_beam_links_master_to_every_slave_with_offset(self):
        coords = {
            1: (0.0, 0.0, 0.0),   # master point
            2: (1.0, 0.0, 0.0),   # slave 1 (offset = +x)
            3: (0.0, 2.0, 0.0),   # slave 2 (offset = +2y)
            4: (0.0, 0.0, 3.0),   # slave 3 (offset = +3z)
        }
        r = _make_resolver(coords)
        defn = RigidLinkDef(
            master_label="A", slave_label="B",
            link_type="beam",
            master_point=(0.0, 0.0, 0.0),
        )

        records = r.resolve_rigid_link(defn, {1}, {2, 3, 4})
        self.assertEqual(len(records), 3)
        for rec in records:
            self.assertEqual(rec.kind, "rigid_beam")
            self.assertEqual(rec.master_node, 1)
            self.assertEqual(rec.dofs, [1, 2, 3, 4, 5, 6])

        by_slave = {rec.slave_node: rec for rec in records}
        np.testing.assert_allclose(by_slave[2].offset, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(by_slave[3].offset, [0.0, 2.0, 0.0])
        np.testing.assert_allclose(by_slave[4].offset, [0.0, 0.0, 3.0])

    def test_rod_uses_translation_dofs_only(self):
        coords = {1: (0, 0, 0), 2: (1, 0, 0)}
        r = _make_resolver(coords)
        defn = RigidLinkDef(
            master_label="A", slave_label="B",
            link_type="rod",
            master_point=(0, 0, 0),
        )
        records = r.resolve_rigid_link(defn, {1}, {2})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].kind, "rigid_rod")
        self.assertEqual(records[0].dofs, [1, 2, 3])

    def test_does_not_pair_master_with_itself(self):
        coords = {1: (0, 0, 0), 2: (1, 0, 0)}
        r = _make_resolver(coords)
        defn = RigidLinkDef(
            master_label="A", slave_label="B",
            link_type="beam",
            master_point=(0, 0, 0),
        )
        # Include master tag in the slave set — resolver must skip it.
        records = r.resolve_rigid_link(defn, {1}, {1, 2})
        slave_ids = {rec.slave_node for rec in records}
        self.assertEqual(slave_ids, {2})

    def test_rigid_beam_constraint_matrix_includes_skew(self):
        # Offset r = (1, 0, 0) → skew matrix:
        #   [[ 0,  0,  0],
        #    [ 0,  0, -1],
        #    [ 0,  1,  0]]
        rec = NodePairRecord(
            kind="rigid_beam",
            master_node=1, slave_node=2,
            dofs=[1, 2, 3, 4, 5, 6],
            offset=np.array([1.0, 0.0, 0.0]),
        )
        C = rec.constraint_matrix(ndof=6)

        # Translation rows (0..2): identity on translations + -skew on rot
        np.testing.assert_allclose(C[:3, :3], np.eye(3))
        expected_skew = np.array([
            [0.0,  0.0, 0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0, 0.0],
        ])
        np.testing.assert_allclose(C[:3, 3:6], -expected_skew)

        # Rotation rows (3..5): identity on rotations
        np.testing.assert_allclose(C[3:6, 3:6], np.eye(3))


# =====================================================================
# resolve_rigid_diaphragm
# =====================================================================

class TestResolveRigidDiaphragm(unittest.TestCase):

    def test_collects_planar_nodes_and_builds_offsets(self):
        # Floor at z=0, one "column" node at z=3.
        coords = {
            1: (0.0, 0.0, 0.0),   # center → master
            2: (5.0, 0.0, 0.0),   # slave
            3: (0.0, 5.0, 0.0),   # slave
            4: (5.0, 5.0, 0.0),   # slave
            5: (2.5, 2.5, 3.0),   # above floor — must be excluded
        }
        r = _make_resolver(coords)
        defn = RigidDiaphragmDef(
            master_label="floor", slave_label="floor",
            master_point=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 0.0, 1.0),
            constrained_dofs=[1, 2, 6],
            plane_tolerance=0.01,
        )

        rec = r.resolve_rigid_diaphragm(defn, all_nodes={1, 2, 3, 4, 5})
        self.assertIsInstance(rec, NodeGroupRecord)
        self.assertEqual(rec.master_node, 1)
        self.assertEqual(sorted(rec.slave_nodes), [2, 3, 4])
        self.assertEqual(rec.dofs, [1, 2, 6])
        self.assertIsNotNone(rec.offsets)
        self.assertEqual(rec.offsets.shape, (3, 3))
        # Plane normal stored (normalised)
        np.testing.assert_allclose(rec.plane_normal, [0.0, 0.0, 1.0])

    def test_empty_when_no_nodes_on_plane(self):
        coords = {1: (0, 0, 5), 2: (1, 0, 5)}
        r = _make_resolver(coords)
        defn = RigidDiaphragmDef(
            master_label="x", slave_label="x",
            master_point=(0, 0, 0),
            plane_normal=(0, 0, 1),
            plane_tolerance=0.1,
        )
        rec = r.resolve_rigid_diaphragm(defn, {1, 2})
        self.assertEqual(rec.slave_nodes, [])

    def test_expand_to_pairs_produces_rigid_beam_pairs(self):
        group = NodeGroupRecord(
            kind="rigid_diaphragm",
            master_node=1,
            slave_nodes=[2, 3],
            dofs=[1, 2, 6],
            offsets=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        pairs = group.expand_to_pairs()
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p.kind == "rigid_beam" for p in pairs))
        self.assertEqual(pairs[0].master_node, 1)
        self.assertEqual(pairs[0].slave_node, 2)
        np.testing.assert_allclose(pairs[0].offset, [1.0, 0.0, 0.0])


# =====================================================================
# resolve_kinematic_coupling / rigid_body
# =====================================================================

class TestResolveKinematicCoupling(unittest.TestCase):

    def test_rigid_body_uses_all_six_dofs(self):
        coords = {
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (0, 1, 0),
            4: (0, 0, 1),
        }
        r = _make_resolver(coords)
        defn = RigidBodyDef(
            master_label="m", slave_label="s",
            master_point=(0, 0, 0),
        )
        rec = r.resolve_kinematic_coupling(
            defn, master_nodes={1}, slave_nodes={2, 3, 4},
        )
        self.assertEqual(rec.kind, "rigid_body")
        self.assertEqual(rec.master_node, 1)
        self.assertEqual(sorted(rec.slave_nodes), [2, 3, 4])
        self.assertEqual(rec.dofs, [1, 2, 3, 4, 5, 6])
        self.assertEqual(rec.offsets.shape, (3, 3))

    def test_kinematic_coupling_respects_custom_dofs(self):
        coords = {1: (0, 0, 0), 2: (1, 0, 0)}
        r = _make_resolver(coords)
        defn = KinematicCouplingDef(
            master_label="m", slave_label="s",
            master_point=(0, 0, 0),
            dofs=[1, 3, 5],
        )
        rec = r.resolve_kinematic_coupling(defn, {1}, {2})
        self.assertEqual(rec.kind, "kinematic_coupling")
        self.assertEqual(rec.dofs, [1, 3, 5])


# =====================================================================
# resolve_distributing
# =====================================================================

class TestResolveDistributing(unittest.TestCase):

    def test_uniform_weights_sum_to_one(self):
        # 4 slave nodes, 1 master. Uniform weights = 1/4 each.
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (-1.0, 0.0, 0.0),
            4: (0.0, 1.0, 0.0),
            5: (0.0, -1.0, 0.0),
        }
        r = _make_resolver(coords)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0, 0, 0),
            weighting="uniform",
        )
        rec = r.resolve_distributing(
            defn, master_nodes={1}, slave_nodes={2, 3, 4, 5},
        )
        self.assertIsInstance(rec, InterpolationRecord)
        self.assertEqual(rec.kind, "distributing")
        self.assertEqual(rec.slave_node, 1)                  # ref point
        self.assertEqual(sorted(rec.master_nodes), [2, 3, 4, 5])
        np.testing.assert_allclose(rec.weights.sum(), 1.0)
        np.testing.assert_allclose(rec.weights, np.full(4, 0.25))

    def test_area_weights_sum_to_one(self):
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (-2.0, 0.0, 0.0),
            4: (0.0, 3.0, 0.0),
        }
        r = _make_resolver(coords)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0, 0, 0),
            weighting="area",
        )
        rec = r.resolve_distributing(defn, {1}, {2, 3, 4})
        np.testing.assert_allclose(rec.weights.sum(), 1.0)
        self.assertEqual(len(rec.weights), 3)


# =====================================================================
# resolve_node_to_surface
# =====================================================================

class TestResolveNodeToSurface(unittest.TestCase):

    def test_generates_phantom_nodes_and_records(self):
        coords = {
            1: (0.0, 0.0, 0.0),   # 6-DOF master
            2: (1.0, 0.0, 0.0),   # 3-DOF slave
            3: (1.0, 1.0, 0.0),   # 3-DOF slave
            4: (0.0, 1.0, 0.0),   # 3-DOF slave
        }
        r = _make_resolver(coords)
        defn = NodeToSurfaceDef(
            master_label="0",
            slave_label="2",
        )
        rec = r.resolve_node_to_surface(
            defn, master_tag=1, slave_nodes={2, 3, 4},
        )

        self.assertIsInstance(rec, NodeToSurfaceRecord)
        self.assertEqual(rec.master_node, 1)
        self.assertEqual(sorted(rec.slave_nodes), [2, 3, 4])

        # Phantom tags allocated above max existing (4)
        self.assertEqual(rec.phantom_nodes, [5, 6, 7])
        self.assertEqual(rec.phantom_coords.shape, (3, 3))

        # 3 rigid beam + 3 equal_dof records
        self.assertEqual(len(rec.rigid_link_records), 3)
        self.assertEqual(len(rec.equal_dof_records), 3)
        for rl in rec.rigid_link_records:
            self.assertEqual(rl.kind, "rigid_beam")
            self.assertEqual(rl.master_node, 1)
            # rigid_beam records carry empty dofs — OpenSees picks
            # DOFs from the model's ndf at emit time.
            self.assertEqual(rl.dofs, [])
        for ed in rec.equal_dof_records:
            self.assertEqual(ed.kind, "equal_dof")
            self.assertEqual(ed.dofs, [1, 2, 3])

        # expand() returns rigid then equalDOF
        expanded = rec.expand()
        self.assertEqual(len(expanded), 6)
        self.assertEqual(expanded[0].kind, "rigid_beam")
        self.assertEqual(expanded[3].kind, "equal_dof")


# =====================================================================
# InterpolationRecord.constraint_matrix
# =====================================================================

class TestInterpolationConstraintMatrix(unittest.TestCase):

    def test_tie_3node_shape_matrix(self):
        # u_slave = 0.5 * u_m1 + 0.3 * u_m2 + 0.2 * u_m3
        rec = InterpolationRecord(
            kind="tie",
            slave_node=99,
            master_nodes=[1, 2, 3],
            weights=np.array([0.5, 0.3, 0.2]),
            dofs=[1, 2, 3],
        )
        C = rec.constraint_matrix(ndof=3)
        self.assertEqual(C.shape, (3, 9))   # (n_dof, n_master * n_dof)
        # Row 0 picks x-DOF of each master
        np.testing.assert_allclose(C[0], [0.5, 0, 0, 0.3, 0, 0, 0.2, 0, 0])
        np.testing.assert_allclose(C[1], [0, 0.5, 0, 0, 0.3, 0, 0, 0.2, 0])
        np.testing.assert_allclose(C[2], [0, 0, 0.5, 0, 0, 0.3, 0, 0, 0.2])


# =====================================================================
# resolve_embedded
# =====================================================================

class TestResolveEmbedded(unittest.TestCase):
    """Embed a point inside a synthetic 2D tri3 mesh and assert the
    returned shape-function weights equal the barycentric coords."""

    def test_tri3_embedded_weights_at_known_xi_eta(self):
        # Single tri3 with corners at (0,0), (1,0), (0,1).
        # Embed a node at (0.25, 0.25) → barycentric (0.5, 0.25, 0.25).
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            99: (0.25, 0.25, 0.0),
        }
        r = _make_resolver(coords)
        host_elems = np.array([[1, 2, 3]], dtype=int)
        defn = EmbeddedDef(
            master_label="host", slave_label="rebar", tolerance=1.0,
        )

        records = r.resolve_embedded(defn, host_elems, {99})
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.kind, "embedded")
        self.assertEqual(rec.slave_node, 99)
        self.assertEqual(rec.master_nodes, [1, 2, 3])
        np.testing.assert_allclose(rec.weights, [0.5, 0.25, 0.25], atol=1e-12)
        # Partition of unity.
        self.assertAlmostEqual(float(np.sum(rec.weights)), 1.0, places=12)
        # Parametric coords (v, w) should correspond to corners 1 and 2.
        np.testing.assert_allclose(rec.parametric_coords, [0.25, 0.25],
                                   atol=1e-12)

    def test_tri3_corner_node_gets_unit_weight(self):
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (2.0, 0.0, 0.0),
            3: (0.0, 2.0, 0.0),
            99: (2.0, 0.0, 0.0),   # sits exactly on corner 2
        }
        r = _make_resolver(coords)
        defn = EmbeddedDef(master_label="h", slave_label="e")
        records = r.resolve_embedded(defn, np.array([[1, 2, 3]]), {99})
        self.assertEqual(len(records), 1)
        np.testing.assert_allclose(records[0].weights, [0, 1, 0], atol=1e-12)

    def test_tet4_centroid_gets_equal_weights(self):
        # Unit tet with corners at the canonical reference positions.
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (0.0, 0.0, 1.0),
            99: (0.25, 0.25, 0.25),   # centroid → weights 1/4 each
        }
        r = _make_resolver(coords)
        defn = EmbeddedDef(master_label="h", slave_label="e")
        records = r.resolve_embedded(
            defn, np.array([[1, 2, 3, 4]]), {99},
        )
        self.assertEqual(len(records), 1)
        np.testing.assert_allclose(records[0].weights, [0.25] * 4, atol=1e-12)
        self.assertEqual(records[0].master_nodes, [1, 2, 3, 4])

    def test_picks_containing_tri_among_many(self):
        # Two tris sharing an edge.  Embedded point clearly inside tri 2.
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (1.0, 1.0, 0.0),
            99: (0.7, 0.7, 0.0),
        }
        r = _make_resolver(coords)
        host = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
        defn = EmbeddedDef(master_label="h", slave_label="e")
        records = r.resolve_embedded(defn, host, {99})
        self.assertEqual(len(records), 1)
        # The containing triangle is (2, 4, 3).
        self.assertEqual(records[0].master_nodes, [2, 4, 3])
        # Weights must be non-negative (point is inside).
        self.assertTrue((records[0].weights >= -1e-12).all())


if __name__ == "__main__":
    unittest.main()
