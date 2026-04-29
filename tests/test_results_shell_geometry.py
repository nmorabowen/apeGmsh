"""Unit tests for shell local-axes + quaternion utilities.

These tests are pure-numpy: no openseespy, no apeGmsh session.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results._shell_geometry import (
    rotation_matrix_to_quaternion,
    shell_local_axes,
    shell_quaternion,
)


# =====================================================================
# shell_local_axes
# =====================================================================

class TestShellLocalAxesQuad:
    def test_unit_quad_in_xy_plane_aligned(self) -> None:
        """Quad in the xy-plane with edges along x, y → identity-ish frame."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        R = shell_local_axes(coords, "ASDShellQ4")
        np.testing.assert_allclose(R[0], [1, 0, 0])    # x_local
        np.testing.assert_allclose(R[1], [0, 1, 0])    # y_local
        np.testing.assert_allclose(R[2], [0, 0, 1])    # z_local

    def test_quad_rotated_in_xy_plane(self) -> None:
        """Quad rotated 90° about z → x_local along +y."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        R = shell_local_axes(coords, "ASDShellQ4")
        np.testing.assert_allclose(R[0], [0, 1, 0])
        np.testing.assert_allclose(R[2], [0, 0, 1])

    def test_quad_in_xz_plane(self) -> None:
        """Quad lying in the xz-plane → normal along +y."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        R = shell_local_axes(coords, "ASDShellQ4")
        np.testing.assert_allclose(R[0], [1, 0, 0])
        np.testing.assert_allclose(R[2], [0, -1, 0])    # right-handed

    def test_works_for_other_quad_classes(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        for cls in ("ShellMITC4", "ShellDKGQ", "ShellNLDKGQ"):
            R = shell_local_axes(coords, cls)
            np.testing.assert_allclose(R[0], [1, 0, 0])
            np.testing.assert_allclose(R[2], [0, 0, 1])


class TestShellLocalAxesTri:
    def test_unit_triangle_xy(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        R = shell_local_axes(coords, "ASDShellT3")
        np.testing.assert_allclose(R[0], [1, 0, 0])
        np.testing.assert_allclose(R[2], [0, 0, 1])

    def test_works_for_other_tri_classes(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        for cls in ("ShellDKGT", "ShellNLDKGT"):
            R = shell_local_axes(coords, cls)
            np.testing.assert_allclose(R[0], [1, 0, 0])


class TestShellLocalAxes9Node:
    def test_quad9_uses_corner_nodes(self) -> None:
        """ShellMITC9 has 9 nodes — the 4 corners drive the local frame."""
        # 4 corners + 4 midsides + center
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ])
        R = shell_local_axes(coords, "ShellMITC9")
        np.testing.assert_allclose(R[0], [1, 0, 0])
        np.testing.assert_allclose(R[2], [0, 0, 1])


class TestShellLocalAxesValidation:
    def test_unsupported_class_raises(self) -> None:
        coords = np.eye(4)[:, :3]
        with pytest.raises(ValueError, match="Unsupported shell class"):
            shell_local_axes(coords, "NotAShell")

    def test_degenerate_first_edge(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],    # n2 == n1
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        with pytest.raises(ValueError, match="zero length"):
            shell_local_axes(coords, "ASDShellQ4")

    def test_collinear_edges(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        with pytest.raises(ValueError, match="collinear"):
            shell_local_axes(coords, "ASDShellQ4")


# =====================================================================
# rotation_matrix_to_quaternion
# =====================================================================

class TestRotationToQuaternion:
    def test_identity_gives_unit_w(self) -> None:
        q = rotation_matrix_to_quaternion(np.eye(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0])

    def test_180deg_about_x(self) -> None:
        # diag(1, -1, -1)
        R = np.diag([1.0, -1.0, -1.0])
        q = rotation_matrix_to_quaternion(R)
        # Expected: ±(0, 1, 0, 0)
        assert np.isclose(abs(q[1]), 1.0)
        np.testing.assert_allclose(q[[0, 2, 3]], [0, 0, 0], atol=1e-12)

    def test_180deg_about_y(self) -> None:
        R = np.diag([-1.0, 1.0, -1.0])
        q = rotation_matrix_to_quaternion(R)
        assert np.isclose(abs(q[2]), 1.0)
        np.testing.assert_allclose(q[[0, 1, 3]], [0, 0, 0], atol=1e-12)

    def test_180deg_about_z(self) -> None:
        R = np.diag([-1.0, -1.0, 1.0])
        q = rotation_matrix_to_quaternion(R)
        assert np.isclose(abs(q[3]), 1.0)
        np.testing.assert_allclose(q[[0, 1, 2]], [0, 0, 0], atol=1e-12)

    def test_90deg_about_z(self) -> None:
        # Rotation that takes +x → +y, +y → -x, +z → +z.
        R = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Hmm — need to be careful about row vs column convention.
        # Our convention: rows = body basis vectors expressed in world.
        # Body x = +y world, body y = -x world, body z = +z world.
        q = rotation_matrix_to_quaternion(R)
        # Quaternion magnitude must be 1.
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)

    def test_quaternion_norm_is_unit(self) -> None:
        # Random orthonormal frame.
        rng = np.random.default_rng(0)
        for _ in range(20):
            A = rng.normal(size=(3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1.0
            q = rotation_matrix_to_quaternion(Q)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            rotation_matrix_to_quaternion(np.eye(4))


# =====================================================================
# shell_quaternion (composition)
# =====================================================================

class TestShellQuaternion:
    def test_identity_quad_in_xy(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        q = shell_quaternion(coords, "ASDShellQ4")
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0])

    def test_quaternion_is_unit_on_arbitrary_quad(self) -> None:
        coords = np.array([
            [0.5, 0.5, 0.5],
            [1.5, 0.7, 0.3],
            [1.6, 1.7, 0.1],
            [0.6, 1.5, 0.6],
        ])
        q = shell_quaternion(coords, "ASDShellQ4")
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
