"""
Resolver-level tests for recently-added features that shipped without
unit coverage:

* ``face_load``  (LoadResolver.resolve_face_load)
* ``face_sp``    (LoadResolver.resolve_face_sp)
* ``tied_contact`` (ConstraintResolver.resolve_tied_contact)
* ``mortar``       (ConstraintResolver.resolve_mortar)
* ``node_to_surface_spring`` (ConstraintResolver.resolve_node_to_surface_spring)

All tests are resolver-level: synthetic node tags + coordinates, no
Gmsh session required.  They exercise the mathematical guarantees each
resolver must satisfy (force/moment equilibrium, rigid-body kinematics
at a centroid, kind-tag routing for the spring variant, …) rather than
implementation details.
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.solvers.Constraints import (
    ConstraintResolver,
    MortarDef,
    NodePairRecord,
    NodeToSurfaceRecord,
    NodeToSurfaceSpringDef,
    SurfaceCouplingRecord,
    TiedContactDef,
)
from apeGmsh.solvers.Loads import (
    FaceLoadDef,
    FaceSPDef,
    LoadResolver,
    NodalLoadRecord,
    SPRecord,
)
from apeGmsh.solvers._kinds import ConstraintKind


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_resolver(coords_by_tag):
    tags = np.array(sorted(coords_by_tag), dtype=np.int64)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return LoadResolver(tags, coords)


def _constraint_resolver(coords_by_tag):
    tags = np.array(sorted(coords_by_tag), dtype=int)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return ConstraintResolver(tags, coords)


def _face_of_four(side: float = 1.0, z: float = 0.0):
    """Unit square in the xy-plane with corner node tags 1..4."""
    return {
        1: (0.0,   0.0,   z),
        2: (side,  0.0,   z),
        3: (side,  side,  z),
        4: (0.0,   side,  z),
    }


# =====================================================================
# resolve_face_load
# =====================================================================

class TestResolveFaceLoad(unittest.TestCase):

    def test_pure_force_equal_split(self):
        """Force F on a 4-node face -> F/4 on each node, no moment."""
        r = _load_resolver(_face_of_four())
        defn = FaceLoadDef(
            target="face",
            force_xyz=(40.0, -20.0, 8.0),
        )
        records = r.resolve_face_load(defn, face_node_ids=[1, 2, 3, 4])
        self.assertEqual(len(records), 4)

        per_node = (10.0, -5.0, 2.0)
        for rec in records:
            self.assertIsInstance(rec, NodalLoadRecord)
            self.assertIsNone(rec.moment_xyz)
            for i in range(3):
                self.assertAlmostEqual(rec.force_xyz[i], per_node[i], places=10)

    def test_pure_moment_satisfies_equilibrium(self):
        """Moment M on a face -> nodal forces with Σf=0 and Σr×f=M."""
        coords = _face_of_four(side=2.0)
        r = _load_resolver(coords)
        M = np.array([0.0, 0.0, 15.0])   # pure z-moment
        defn = FaceLoadDef(target="face", moment_xyz=tuple(M))
        records = r.resolve_face_load(defn, face_node_ids=[1, 2, 3, 4])

        centroid = np.mean(list(coords.values()), axis=0)
        total_f = np.zeros(3)
        total_m = np.zeros(3)
        for rec in records:
            f = np.asarray(rec.force_xyz, dtype=float)
            xi = np.asarray(coords[rec.node_id], dtype=float)
            r_arm = xi - centroid
            total_f += f
            total_m += np.cross(r_arm, f)

        np.testing.assert_allclose(total_f, 0.0, atol=1e-10)
        np.testing.assert_allclose(total_m, M, atol=1e-10)

    def test_combined_force_and_moment_is_linear_superposition(self):
        coords = _face_of_four()
        r = _load_resolver(coords)
        F = (3.0, 0.0, -6.0)
        M = (0.0, 2.5, 0.0)

        recs_F = r.resolve_face_load(
            FaceLoadDef(target="f", force_xyz=F),
            face_node_ids=[1, 2, 3, 4],
        )
        recs_M = r.resolve_face_load(
            FaceLoadDef(target="f", moment_xyz=M),
            face_node_ids=[1, 2, 3, 4],
        )
        recs_FM = r.resolve_face_load(
            FaceLoadDef(target="f", force_xyz=F, moment_xyz=M),
            face_node_ids=[1, 2, 3, 4],
        )

        def by_node(recs):
            out: dict[int, np.ndarray] = {}
            for rc in recs:
                v = np.zeros(3)
                if rc.force_xyz is not None:
                    v = np.asarray(rc.force_xyz, dtype=float)
                out[rc.node_id] = out.get(rc.node_id, np.zeros(3)) + v
            return out

        a, b, c = by_node(recs_F), by_node(recs_M), by_node(recs_FM)
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(a[nid] + b[nid], c[nid], atol=1e-10)

    def test_empty_face_node_list_returns_empty(self):
        r = _load_resolver({1: (0, 0, 0)})
        recs = r.resolve_face_load(
            FaceLoadDef(target="empty", force_xyz=(1, 0, 0)),
            face_node_ids=[],
        )
        self.assertEqual(recs, [])


# =====================================================================
# resolve_face_sp
# =====================================================================

class TestResolveFaceSP(unittest.TestCase):

    def test_pure_translation_assigns_uniform_displacement(self):
        """Pure disp at centroid -> same (ux, uy, uz) on every node."""
        r = _load_resolver(_face_of_four())
        defn = FaceSPDef(
            target="face",
            dofs=[1, 1, 1],
            disp_xyz=(0.3, -0.1, 2.0),
        )
        recs = r.resolve_face_sp(defn, face_node_ids=[1, 2, 3, 4])
        self.assertEqual(len(recs), 4 * 3)  # 4 nodes × 3 dofs

        by_node: dict[int, dict[int, float]] = {}
        for rc in recs:
            self.assertIsInstance(rc, SPRecord)
            by_node.setdefault(rc.node_id, {})[rc.dof] = rc.value

        for nid in (1, 2, 3, 4):
            self.assertAlmostEqual(by_node[nid][1],  0.3, places=10)
            self.assertAlmostEqual(by_node[nid][2], -0.1, places=10)
            self.assertAlmostEqual(by_node[nid][3],  2.0, places=10)

    def test_pure_rotation_about_z_gives_tangential_disp(self):
        """θz about centroid -> ux_i = -θz·ry, uy_i = +θz·rx."""
        coords = _face_of_four(side=2.0)
        r = _load_resolver(coords)
        theta_z = 0.5
        defn = FaceSPDef(
            target="face",
            dofs=[1, 1, 1],
            rot_xyz=(0.0, 0.0, theta_z),
        )
        recs = r.resolve_face_sp(defn, face_node_ids=[1, 2, 3, 4])

        centroid = np.mean(list(coords.values()), axis=0)
        by_node: dict[int, dict[int, float]] = {}
        for rc in recs:
            by_node.setdefault(rc.node_id, {})[rc.dof] = rc.value

        for nid in (1, 2, 3, 4):
            r_arm = np.asarray(coords[nid], dtype=float) - centroid
            expected_ux = -theta_z * r_arm[1]
            expected_uy =  theta_z * r_arm[0]
            self.assertAlmostEqual(by_node[nid][1], expected_ux, places=10)
            self.assertAlmostEqual(by_node[nid][2], expected_uy, places=10)
            self.assertAlmostEqual(by_node[nid][3], 0.0, places=10)

    def test_dof_mask_skips_free_directions(self):
        """A mask of [1, 0, 1] should emit only DOFs 1 and 3 per node."""
        r = _load_resolver(_face_of_four())
        defn = FaceSPDef(
            target="f",
            dofs=[1, 0, 1],
            disp_xyz=(1.0, 99.0, 2.0),   # 99.0 should never appear
        )
        recs = r.resolve_face_sp(defn, face_node_ids=[1, 2, 3, 4])
        dofs_seen = sorted({rc.dof for rc in recs})
        self.assertEqual(dofs_seen, [1, 3])
        self.assertEqual(len(recs), 4 * 2)
        for rc in recs:
            self.assertNotEqual(rc.value, 99.0)

    def test_homogeneous_fix_flags_is_homogeneous_true(self):
        """disp=None and rot=None -> every SPRecord has value=0 and
        is_homogeneous=True."""
        r = _load_resolver(_face_of_four())
        defn = FaceSPDef(target="f", dofs=[1, 1, 1])
        recs = r.resolve_face_sp(defn, face_node_ids=[1, 2, 3, 4])
        self.assertEqual(len(recs), 12)
        for rc in recs:
            self.assertEqual(rc.value, 0.0)
            self.assertTrue(rc.is_homogeneous)


# =====================================================================
# resolve_tied_contact
# =====================================================================

class TestResolveTiedContact(unittest.TestCase):

    def test_bidirectional_produces_slave_records_from_both_sides(self):
        """Coincident master/slave faces -> both forward and backward
        projections land inside the other face, yielding 8 records."""
        # Master and slave are the same unit square z=0 with different
        # node tags. Forward: slave nodes 11..14 project to master face.
        # Backward: master nodes 1..4 project to slave face.
        coords = {
            1:  (0.0, 0.0, 0.0),
            2:  (1.0, 0.0, 0.0),
            3:  (1.0, 1.0, 0.0),
            4:  (0.0, 1.0, 0.0),
            11: (0.0, 0.0, 0.0),
            12: (1.0, 0.0, 0.0),
            13: (1.0, 1.0, 0.0),
            14: (0.0, 1.0, 0.0),
        }
        r = _constraint_resolver(coords)
        master_face = np.array([[1, 2, 3, 4]], dtype=int)
        slave_face  = np.array([[11, 12, 13, 14]], dtype=int)
        defn = TiedContactDef(
            master_label="A",
            slave_label="B",
            tolerance=0.1,
        )
        rec = r.resolve_tied_contact(
            defn,
            master_face, slave_face,
            master_nodes={1, 2, 3, 4},
            slave_nodes={11, 12, 13, 14},
        )
        self.assertIsInstance(rec, SurfaceCouplingRecord)
        self.assertEqual(rec.kind, ConstraintKind.TIED_CONTACT)
        # 4 slave-to-master projections + 4 master-to-slave projections
        self.assertEqual(len(rec.slave_records), 8)
        # Sanity: weights from Q4 shape functions sum to 1
        for ir in rec.slave_records:
            self.assertAlmostEqual(float(np.sum(ir.weights)), 1.0, places=8)

    def test_tolerance_excludes_distant_faces(self):
        """Slave face offset in z > tolerance -> no records."""
        coords = {
            1: (0.0, 0.0, 0.0),  2: (1.0, 0.0, 0.0),
            3: (1.0, 1.0, 0.0),  4: (0.0, 1.0, 0.0),
            11: (0.0, 0.0, 5.0), 12: (1.0, 0.0, 5.0),
            13: (1.0, 1.0, 5.0), 14: (0.0, 1.0, 5.0),
        }
        r = _constraint_resolver(coords)
        rec = r.resolve_tied_contact(
            TiedContactDef(master_label="A", slave_label="B",
                           tolerance=0.5),
            np.array([[1, 2, 3, 4]], dtype=int),
            np.array([[11, 12, 13, 14]], dtype=int),
            master_nodes={1, 2, 3, 4},
            slave_nodes={11, 12, 13, 14},
        )
        self.assertEqual(rec.slave_records, [])


# =====================================================================
# resolve_mortar
# =====================================================================

class TestResolveMortar(unittest.TestCase):

    def test_operator_matrix_has_correct_shape(self):
        """B should be (n_slave * nd) x (n_master * nd)."""
        coords = {
            1:  (0.0, 0.0, 0.0),  2:  (1.0, 0.0, 0.0),
            3:  (1.0, 1.0, 0.0),  4:  (0.0, 1.0, 0.0),
            11: (0.25, 0.25, 0.0), 12: (0.75, 0.25, 0.0),
            13: (0.75, 0.75, 0.0), 14: (0.25, 0.75, 0.0),
        }
        r = _constraint_resolver(coords)
        defn = MortarDef(master_label="A", slave_label="B")
        rec = r.resolve_mortar(
            defn,
            np.array([[1, 2, 3, 4]], dtype=int),
            np.array([[11, 12, 13, 14]], dtype=int),
            master_nodes={1, 2, 3, 4},
            slave_nodes={11, 12, 13, 14},
        )
        self.assertIsInstance(rec, SurfaceCouplingRecord)
        self.assertEqual(rec.kind, ConstraintKind.MORTAR)
        self.assertIsNotNone(rec.mortar_operator)
        n_s, n_m, nd = 4, 4, 3
        self.assertEqual(rec.mortar_operator.shape, (n_s * nd, n_m * nd))


# =====================================================================
# resolve_node_to_surface_spring
# =====================================================================

class TestResolveNodeToSurfaceSpring(unittest.TestCase):

    def test_rigid_link_records_tagged_as_stiff(self):
        """The spring variant must tag master->phantom links with
        RIGID_BEAM_STIFF so stiff_beam_groups() picks them up and
        rigid_link_groups() skips them."""
        coords = {
            1: (0.0, 0.0, 0.5),   # master above the face
            2: (0.0, 0.0, 0.0),
            3: (1.0, 0.0, 0.0),
            4: (1.0, 1.0, 0.0),
            5: (0.0, 1.0, 0.0),
        }
        r = _constraint_resolver(coords)
        defn = NodeToSurfaceSpringDef(master_label="1", slave_label="face")
        rec = r.resolve_node_to_surface_spring(
            defn, master_tag=1, slave_nodes={2, 3, 4, 5},
        )
        self.assertIsInstance(rec, NodeToSurfaceRecord)
        self.assertEqual(rec.kind, ConstraintKind.NODE_TO_SURFACE_SPRING)

        self.assertEqual(len(rec.rigid_link_records), 4)
        for pair in rec.rigid_link_records:
            self.assertIsInstance(pair, NodePairRecord)
            self.assertEqual(pair.kind, ConstraintKind.RIGID_BEAM_STIFF)

        self.assertEqual(len(rec.equal_dof_records), 4)
        for pair in rec.equal_dof_records:
            self.assertEqual(pair.kind, ConstraintKind.EQUAL_DOF)
            self.assertEqual(pair.dofs, [1, 2, 3])

    def test_phantom_tags_are_contiguous_and_above_existing(self):
        """Phantom node tags must be contiguous and start above the
        highest existing node tag so solvers can ops.node() them."""
        coords = {
            1:  (0.0, 0.0, 0.0),
            10: (1.0, 0.0, 0.0),
            20: (1.0, 1.0, 0.0),
            30: (0.0, 1.0, 0.0),
            42: (0.0, 0.0, 1.0),  # master
        }
        r = _constraint_resolver(coords)
        defn = NodeToSurfaceSpringDef(master_label="42", slave_label="face")
        rec = r.resolve_node_to_surface_spring(
            defn, master_tag=42, slave_nodes={1, 10, 20, 30},
        )
        # Existing max tag = 42 -> phantoms must start at 43
        self.assertEqual(len(rec.phantom_nodes), 4)
        self.assertEqual(min(rec.phantom_nodes), 43)
        self.assertEqual(max(rec.phantom_nodes), 46)
        # Contiguous
        self.assertEqual(
            rec.phantom_nodes,
            list(range(min(rec.phantom_nodes),
                       min(rec.phantom_nodes) + len(rec.phantom_nodes))),
        )

    def test_offsets_are_slave_coords_minus_master(self):
        """Each rigid link's offset must equal (slave_xyz - master_xyz)."""
        coords = {
            1: (2.0, 3.0, 5.0),   # master
            2: (2.0, 3.0, 0.0),
            3: (7.0, 3.0, 0.0),
        }
        r = _constraint_resolver(coords)
        defn = NodeToSurfaceSpringDef(master_label="1", slave_label="face")
        rec = r.resolve_node_to_surface_spring(
            defn, master_tag=1, slave_nodes={2, 3},
        )
        master_xyz = np.asarray(coords[1])
        for pair in rec.rigid_link_records:
            # phantom was created from a slave; find which slave via
            # the corresponding equal_dof record (phantom -> slave)
            edof = next(e for e in rec.equal_dof_records
                        if e.master_node == pair.slave_node)
            slave_xyz = np.asarray(coords[edof.slave_node])
            expected = slave_xyz - master_xyz
            np.testing.assert_allclose(pair.offset, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
