"""
Unit tests for
:class:`apeGmsh.mesh._constraint_resolver.ConstraintResolver`.

The resolver is deliberately decoupled from Gmsh — it operates on raw
numpy arrays of node tags and coordinates.  These tests exercise it
with hand-built synthetic meshes so no Gmsh session is required.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pytest

from apeGmsh.core.constraints.defs import (
    DistributingCouplingDef,
    EmbeddedDef,
    EqualDOFDef,
    KinematicCouplingDef,
    NodeToSurfaceDef,
    PenaltyDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
)
from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver
from apeGmsh._kernel.records import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
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
# resolve_equal_dof_mixed (ADR 0069)
# =====================================================================

class TestResolveEqualDOFMixed(unittest.TestCase):

    def test_carries_separate_retained_and_constrained_dofs(self):
        from apeGmsh.core.constraints.defs import EqualDOFMixedDef

        # Two co-located pairs; tie master ux→slave rz and master uy→slave uy.
        coords = {
            1: (0, 0, 0), 2: (1, 0, 0),
            11: (0, 0, 0), 12: (1, 0, 0),
        }
        r = _make_resolver(coords)
        defn = EqualDOFMixedDef(
            master_label="A", slave_label="B",
            dof_pairs=[(1, 6), (2, 2)],
        )

        records = r.resolve_equal_dof_mixed(
            defn, master_nodes={1, 2}, slave_nodes={11, 12},
        )

        self.assertEqual(len(records), 2)
        pairs = {(rec.master_node, rec.slave_node) for rec in records}
        self.assertEqual(pairs, {(1, 11), (2, 12)})
        for rec in records:
            self.assertEqual(rec.kind, "equal_dof_mixed")
            self.assertEqual(rec.master_dofs, [1, 2])   # retained (RDOF)
            self.assertEqual(rec.dofs, [6, 2])          # constrained (CDOF)

    def test_tolerance_excludes_far_nodes(self):
        from apeGmsh.core.constraints.defs import EqualDOFMixedDef

        coords = {1: (0, 0, 0), 11: (0, 0, 0), 12: (5, 0, 0)}
        r = _make_resolver(coords)
        defn = EqualDOFMixedDef(
            master_label="A", slave_label="B",
            dof_pairs=[(3, 3)], tolerance=0.05,
        )
        records = r.resolve_equal_dof_mixed(defn, {1}, {11, 12})
        self.assertEqual({rec.slave_node for rec in records}, {11})

    def test_empty_inputs(self):
        from apeGmsh.core.constraints.defs import EqualDOFMixedDef

        r = _make_resolver({1: (0, 0, 0)})
        defn = EqualDOFMixedDef(
            master_label="A", slave_label="B", dof_pairs=[(1, 1)],
        )
        self.assertEqual(r.resolve_equal_dof_mixed(defn, set(), {1}), [])
        self.assertEqual(r.resolve_equal_dof_mixed(defn, {1}, set()), [])


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

    def test_kinematic_coupling_carries_control_onto_record(self):
        from apeGmsh._kernel._coupling_control import CouplingControl
        r = _make_resolver({1: (0, 0, 0), 2: (1, 0, 0)})
        defn = KinematicCouplingDef(
            master_label="m", slave_label="s", master_point=(0, 0, 0),
            control=CouplingControl(k=1e10, enforce="al"),
        )
        rec = r.resolve_kinematic_coupling(defn, {1}, {2})
        self.assertEqual(rec.control, CouplingControl(k=1e10, enforce="al"))

    def test_kinematic_coupling_default_control_becomes_none(self):
        r = _make_resolver({1: (0, 0, 0), 2: (1, 0, 0)})
        defn = KinematicCouplingDef(
            master_label="m", slave_label="s", master_point=(0, 0, 0),
        )
        rec = r.resolve_kinematic_coupling(defn, {1}, {2})
        self.assertIsNone(rec.control)   # no knobs set ⇒ None (emits nothing)


# =====================================================================
# resolve_distributing
# =====================================================================

class TestResolveDistributing(unittest.TestCase):
    """resolve_distributing → InterpolationRecord (RBE3 fork element).

    R = the master-side node closest to ``master_point`` (the dependent
    reference); independents = the slave set minus R; weights ``None``
    (uniform). The record maps onto
    ``element LadrunoDistributingCoupling $tag $R $N $i1..iN``.
    """

    def test_resolve_distributing_builds_interpolation_record(self):
        from apeGmsh._kernel.records._constraints import InterpolationRecord
        r = _make_resolver({1: (0., 0., 0.), 2: (1., 0., 0.),
                            3: (-1., 0., 0.), 4: (0., 1., 0.)})
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0, 0, 0), weighting="uniform",
        )
        rec = r.resolve_distributing(defn, {1}, {2, 3, 4})
        self.assertIsInstance(rec, InterpolationRecord)
        self.assertEqual(rec.kind, "distributing")
        self.assertEqual(rec.slave_node, 1)            # reference node R
        self.assertEqual(rec.master_nodes, [2, 3, 4])  # independents (sorted)
        self.assertIsNone(rec.weights)                 # uniform ⇒ no -w

    def test_resolve_distributing_no_independents_raises(self):
        r = _make_resolver({1: (0., 0., 0.)})
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf", master_point=(0, 0, 0),
        )
        # slave set is only the reference node ⇒ no independents
        with self.assertRaises(ValueError):
            r.resolve_distributing(defn, {1}, {1})

    def test_resolve_distributing_carries_control_onto_record(self):
        from apeGmsh._kernel._coupling_control import CouplingControl
        r = _make_resolver({1: (0., 0., 0.), 2: (1., 0., 0.), 3: (-1., 0., 0.)})
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf", master_point=(0, 0, 0),
            control=CouplingControl(k=5e8, bipenalty_dtcr=2e-6),
        )
        rec = r.resolve_distributing(defn, {1}, {2, 3})
        self.assertEqual(rec.control,
                         CouplingControl(k=5e8, bipenalty_dtcr=2e-6))
        # default control ⇒ None
        defn2 = DistributingCouplingDef(
            master_label="ref", slave_label="surf", master_point=(0, 0, 0),
        )
        self.assertIsNone(r.resolve_distributing(defn2, {1}, {2, 3}).control)


class TestResolveDistributingAreaWeighting(unittest.TestCase):
    """``weighting="area"`` → per-independent tributary areas in ``weights``.

    Each face's area is split equally among its nodes (the ``g.loads``
    surface-tributary lumping model) and accumulated per node; weights
    align with the record's sorted independent order. The fork
    normalizes by ``W = Σw``, so only proportionality matters.
    """

    # Two coplanar quads sharing edge 3-4:
    #   face A (2,3,4,5): 1×1 unit square, area 1   → share 0.25/node
    #   face B (3,6,7,4): 2×1 rectangle,  area 2   → share 0.5/node
    # Ref node 1 floats above the surface.
    _COORDS = {
        1: (0.5, 0.5, 1.0),
        2: (0.0, 0.0, 0.0), 3: (1.0, 0.0, 0.0),
        4: (1.0, 1.0, 0.0), 5: (0.0, 1.0, 0.0),
        6: (3.0, 0.0, 0.0), 7: (3.0, 1.0, 0.0),
    }
    _FACES = np.array([[2, 3, 4, 5], [3, 6, 7, 4]], dtype=int)

    def test_area_weights_are_tributary_and_sorted(self):
        r = _make_resolver(self._COORDS)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0.5, 0.5, 1.0), weighting="area",
        )
        rec = r.resolve_distributing(
            defn, {1}, {2, 3, 4, 5, 6, 7}, slave_face_conn=self._FACES,
        )
        self.assertEqual(rec.master_nodes, [2, 3, 4, 5, 6, 7])
        # nodes 2,5 only on A (0.25); 3,4 shared (0.25+0.5); 6,7 only on B.
        np.testing.assert_allclose(
            rec.weights, [0.25, 0.75, 0.75, 0.25, 0.5, 0.5],
        )

    def test_area_weights_tri_faces(self):
        # One unit right triangle, area 0.5 → each node gets 1/6.
        r = _make_resolver({
            1: (0.0, 0.0, 1.0),
            2: (0.0, 0.0, 0.0), 3: (1.0, 0.0, 0.0), 4: (0.0, 1.0, 0.0),
        })
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0, 0, 1), weighting="area",
        )
        rec = r.resolve_distributing(
            defn, {1}, {2, 3, 4},
            slave_face_conn=np.array([[2, 3, 4]], dtype=int),
        )
        np.testing.assert_allclose(rec.weights, [0.5 / 3] * 3)

    def test_area_without_faces_raises(self):
        r = _make_resolver(self._COORDS)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0.5, 0.5, 1.0), weighting="area",
        )
        with self.assertRaisesRegex(ValueError, "face connectivity"):
            r.resolve_distributing(defn, {1}, {2, 3}, slave_face_conn=None)
        with self.assertRaisesRegex(ValueError, "face connectivity"):
            r.resolve_distributing(
                defn, {1}, {2, 3},
                slave_face_conn=np.empty((0, 4), dtype=int),
            )

    def test_area_node_on_no_face_fails_loud(self):
        # Node 8 is in the independent set but on no slave face.
        coords = dict(self._COORDS)
        coords[8] = (9.0, 9.0, 0.0)
        r = _make_resolver(coords)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0.5, 0.5, 1.0), weighting="area",
        )
        with self.assertRaisesRegex(ValueError, r"\[8\]"):
            r.resolve_distributing(
                defn, {1}, {2, 3, 4, 5, 6, 7, 8},
                slave_face_conn=self._FACES,
            )

    def test_uniform_ignores_faces(self):
        r = _make_resolver(self._COORDS)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(0.5, 0.5, 1.0), weighting="uniform",
        )
        rec = r.resolve_distributing(
            defn, {1}, {2, 3}, slave_face_conn=self._FACES,
        )
        self.assertIsNone(rec.weights)

    def test_ref_node_on_face_keeps_share_unreported(self):
        # The reference node sits ON the slave surface: its face share
        # is not reported (it's not an independent) and the remaining
        # weights are unaffected in order.
        r = _make_resolver(self._COORDS)
        defn = DistributingCouplingDef(
            master_label="ref", slave_label="surf",
            master_point=(1.0, 0.0, 0.0), weighting="area",
        )
        # Node 3 is the closest to master_point ⇒ becomes R, excluded.
        rec = r.resolve_distributing(
            defn, {3}, {2, 3, 4, 5, 6, 7}, slave_face_conn=self._FACES,
        )
        self.assertEqual(rec.slave_node, 3)
        self.assertEqual(rec.master_nodes, [2, 4, 5, 6, 7])
        np.testing.assert_allclose(
            rec.weights, [0.25, 0.75, 0.25, 0.5, 0.5],
        )


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


# ---------------------------------------------------------------------
# Phase 2 — barycentric-excess tolerance gate
# ---------------------------------------------------------------------
#
# ``EmbeddedDef.tolerance`` is the maximum barycentric excess allowed
# when locating an embedded node inside a host element.  Default
# ``0.0`` means strictly inside; the resolver raises ``ValueError`` on
# any off-host node.  Users with CAD-noise geometry set a small
# positive tolerance (e.g. ``1e-3``) to allow controlled extrapolation.


def test_off_host_embedded_node_raises_under_default_tolerance() -> None:
    """An embedded node lying outside every host element fails loud
    under the default tolerance (0.0) — never produces an
    ``InterpolationRecord`` with extrapolation (negative) weights.
    """
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0),
        99: (10.0, 10.0, 10.0),  # way outside the unit tet
    }
    r = _make_resolver(coords)
    defn = EmbeddedDef(master_label="host", slave_label="rebar",
                       name="rebar_node_outside_host")

    with pytest.raises(ValueError, match=r"slave node 99 lies outside"):
        r.resolve_embedded(
            defn, np.array([[1, 2, 3, 4]], dtype=int), {99},
        )


def test_off_host_embedded_node_accepted_when_tolerance_is_wide() -> None:
    """Setting ``tolerance`` wider than the actual excess accepts the
    record (intentional-extrapolation escape hatch for users with
    geometry/mesh noise they understand)."""
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0),
        99: (10.0, 10.0, 10.0),
    }
    r = _make_resolver(coords)
    defn = EmbeddedDef(master_label="host", slave_label="rebar",
                       tolerance=1e9)  # arbitrarily large

    records = r.resolve_embedded(
        defn, np.array([[1, 2, 3, 4]], dtype=int), {99},
    )
    assert len(records) == 1
    # Excess is surfaced on the record (introspectable post-resolve).
    assert records[0].excess is not None
    assert records[0].excess > 0.0


def test_inside_embedded_node_has_zero_excess() -> None:
    """An embedded node strictly inside a host element gets a record
    with ``excess`` ≈ 0 (the short-circuit `bary_tol = 1e-6` threshold).
    """
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0),
        99: (0.25, 0.25, 0.25),  # tet centroid
    }
    r = _make_resolver(coords)
    defn = EmbeddedDef(master_label="host", slave_label="rebar")

    records = r.resolve_embedded(
        defn, np.array([[1, 2, 3, 4]], dtype=int), {99},
    )
    assert len(records) == 1
    assert records[0].excess is not None
    assert records[0].excess <= 1e-6


class TestColocatedPairingFailLoud(unittest.TestCase):
    """Many-to-one co-located matching is degenerate (redundant /
    conflicting MPCs) — must fail loud, not silently emit (PR-D)."""

    def test_master_matched_by_multiple_slaves_raises(self):
        # master 1 and slaves 2,3 all coincide → 2 & 3 both match 1.
        r = _make_resolver({
            1: (0.0, 0.0, 0.0),
            2: (0.0, 0.0, 0.0),
            3: (0.0, 0.0, 0.0),
        })
        with self.assertRaises(ValueError):
            r._match_node_pairs({1}, {2, 3}, tolerance=1e-3)

    def test_clean_one_to_one_still_pairs(self):
        # Distinct co-located pairs (1↔3, 2↔4) — no ambiguity.
        r = _make_resolver({
            1: (0.0, 0.0, 0.0), 3: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0), 4: (1.0, 0.0, 0.0),
        })
        pairs = r._match_node_pairs({1, 2}, {3, 4}, tolerance=1e-6)
        self.assertEqual(sorted(pairs), [(1, 3), (2, 4)])


# ---------------------------------------------------------------------------
# ADR 0035 — ASDEmbeddedNodeElement option exposure
# ---------------------------------------------------------------------------


class TestAsdEmbeddedOptionValidation:
    """The four Defs that map to ASDEmbeddedNodeElement guard against
    the C++ parser's mutual-exclusion rule (``-rot`` vs ``-p`` at
    ASDEmbeddedNodeElement.cpp:276) and against silently-ignored
    ``-KP`` (only consulted when ``-p`` is active)."""

    def test_tie_rot_and_pressure_mutually_exclusive(self):
        from apeGmsh._kernel.defs.constraints import TieDef
        with pytest.raises(ValueError, match="mutually exclusive"):
            TieDef(
                master_label="m", slave_label="s",
                rotational=True, pressure=True,
            )

    def test_embedded_rot_and_pressure_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            EmbeddedDef(
                master_label="m", slave_label="s",
                rotational=True, pressure=True,
            )

    def test_tied_contact_rot_and_pressure_mutually_exclusive(self):
        from apeGmsh._kernel.defs.constraints import TiedContactDef
        with pytest.raises(ValueError, match="mutually exclusive"):
            TiedContactDef(
                master_label="m", slave_label="s",
                rotational=True, pressure=True,
            )

    def test_tie_stiffness_p_without_pressure_raises(self):
        from apeGmsh._kernel.defs.constraints import TieDef
        with pytest.raises(ValueError, match="stiffness_p"):
            TieDef(
                master_label="m", slave_label="s",
                stiffness_p=1.0e6,
            )

    def test_embedded_stiffness_p_with_pressure_accepted(self):
        defn = EmbeddedDef(
            master_label="m", slave_label="s",
            pressure=True, stiffness_p=1.0e6,
        )
        assert defn.pressure is True
        assert defn.stiffness_p == 1.0e6

    def test_default_stiffness_matches_opensees_cpp(self):
        # DistributingCouplingDef is no longer an ASD-embedded constraint
        # (it emits the fork LadrunoDistributingCoupling element, which has
        # its own -k penalty default), so it carries no stiffness/rotational/
        # pressure fields and is excluded here.
        from apeGmsh._kernel.defs.constraints import (
            TieDef, TiedContactDef,
        )
        for D in (TieDef, EmbeddedDef, TiedContactDef):
            d = D(master_label="m", slave_label="s")
            assert d.stiffness == 1.0e18
            assert d.stiffness_p is None
            assert d.rotational is False
            assert d.pressure is False


class TestAsdEmbeddedResolverPropagation:
    """The four flags ride the resolver from the Def into every
    :class:`InterpolationRecord` row so the emit pass can ``-K``/``-KP``/
    ``-rot``/``-p`` per record without re-consulting the Def."""

    def test_resolve_embedded_carries_flags_to_record(self):
        coords = {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (0.0, 0.0, 1.0),
            99: (0.25, 0.25, 0.25),
        }
        r = _make_resolver(coords)
        defn = EmbeddedDef(
            master_label="host", slave_label="rebar",
            stiffness=1.0e8, rotational=True,
        )
        records = r.resolve_embedded(
            defn, np.array([[1, 2, 3, 4]], dtype=int), {99},
        )
        assert len(records) == 1
        assert records[0].stiffness == 1.0e8
        assert records[0].rotational is True
        assert records[0].pressure is False
        assert records[0].stiffness_p is None


if __name__ == "__main__":
    unittest.main()
