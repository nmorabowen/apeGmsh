"""
Numerical MPC-relation locks (deep-review PR-E).

The constraint *math* — the actual kinematic relation each record
encodes — had zero numerical verification (only smoke / structure
tests).  These lock the exact relations so a sign flip or a wrong
skew term cannot regress silently:

* equal_dof  : selection matrix (u_sᵢ = u_mᵢ)
* rigid_beam : u_s = u_m + θ_m × r,  θ_s = θ_m
* rigid_rod  : u_s = u_m + θ_m × r,  θ_s free
* rigid_body / rigid_diaphragm : expand to rigid_beam pairs
* tie        : u_s = Σ wᵢ u_mᵢ with Σ wᵢ = 1 (partition of unity)
"""
import numpy as np

from apeGmsh.mesh.records._constraints import (
    InterpolationRecord, NodeGroupRecord, NodePairRecord,
)
from apeGmsh.mesh.records._kinds import ConstraintKind as K


def test_equal_dof_is_selection_matrix():
    rec = NodePairRecord(kind=K.EQUAL_DOF, master_node=1, slave_node=2,
                          dofs=[1, 2, 3])
    C = rec.constraint_matrix(ndof=6)
    expected = np.zeros((3, 6))
    expected[0, 0] = expected[1, 1] = expected[2, 2] = 1.0
    np.testing.assert_allclose(C, expected)


def test_rigid_beam_reproduces_u_plus_theta_cross_r():
    r = np.array([2.0, -1.0, 3.0])          # offset = x_s − x_m
    rec = NodePairRecord(kind=K.RIGID_BEAM, master_node=1, slave_node=2,
                          dofs=[1, 2, 3, 4, 5, 6], offset=r)
    C = rec.constraint_matrix(ndof=6)
    u_m = np.array([1.0, 2.0, 3.0])
    th_m = np.array([0.10, -0.20, 0.05])
    out = C @ np.concatenate([u_m, th_m])
    np.testing.assert_allclose(out[:3], u_m + np.cross(th_m, r),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out[3:], th_m,           # θ_s = θ_m
                               rtol=1e-12, atol=1e-12)


def test_rigid_rod_translations_only():
    r = np.array([0.5, 4.0, -2.0])
    rec = NodePairRecord(kind=K.RIGID_ROD, master_node=1, slave_node=2,
                          dofs=[1, 2, 3], offset=r)   # no rotation dofs
    C = rec.constraint_matrix(ndof=6)
    u_m = np.array([0.3, -0.7, 1.1])
    th_m = np.array([0.2, 0.1, -0.4])
    out = C @ np.concatenate([u_m, th_m])
    np.testing.assert_allclose(out, u_m + np.cross(th_m, r),
                               rtol=1e-12, atol=1e-12)
    assert C.shape == (3, 6)               # rotations of slave are free


def test_rigid_body_and_diaphragm_expand_to_rigid_beam():
    r = np.array([1.0, 0.0, -2.0])
    for kind in (K.RIGID_BODY, K.RIGID_DIAPHRAGM):
        ng = NodeGroupRecord(
            kind=kind, master_node=10, slave_nodes=[11],
            dofs=[1, 2, 3, 4, 5, 6], offsets=np.array([r]),
        )
        pairs = ng.expand_to_pairs()
        assert len(pairs) == 1
        p = pairs[0]
        assert p.kind == K.RIGID_BEAM         # rigid-link emission
        assert (p.master_node, p.slave_node) == (10, 11)
        u_m = np.array([2.0, -1.0, 0.5])
        th_m = np.array([0.05, 0.0, -0.1])
        out = p.constraint_matrix(ndof=6) @ np.concatenate([u_m, th_m])
        np.testing.assert_allclose(out[:3], u_m + np.cross(th_m, r),
                                   rtol=1e-12, atol=1e-12)


def test_kinematic_coupling_expand_preserves_dof_subset():
    ng = NodeGroupRecord(
        kind=K.KINEMATIC_COUPLING, master_node=1, slave_nodes=[2, 3],
        dofs=[1, 3, 5],
    )
    pairs = ng.expand_to_pairs()
    assert [p.kind for p in pairs] == [K.KINEMATIC_COUPLING] * 2
    # The DOF subset must survive expansion (the 6-DOF collapse bug
    # destroyed exactly this — see PR-B).
    assert all(p.dofs == [1, 3, 5] for p in pairs)


def test_tie_partition_of_unity_and_interpolation():
    w = np.array([0.5, 0.3, 0.2])
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=9, master_nodes=[1, 2, 3], weights=w,
        dofs=[1, 2, 3],
    )
    np.testing.assert_allclose(w.sum(), 1.0)      # partition of unity
    C = rec.constraint_matrix(ndof=3)
    u1 = np.array([1.0, 0.0, 0.0])
    u2 = np.array([0.0, 2.0, 0.0])
    u3 = np.array([0.0, 0.0, 4.0])
    out = C @ np.concatenate([u1, u2, u3])
    np.testing.assert_allclose(out, 0.5 * u1 + 0.3 * u2 + 0.2 * u3,
                               rtol=1e-12, atol=1e-12)
