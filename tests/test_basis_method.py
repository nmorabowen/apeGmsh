"""Tests for the basis-function (Vandermonde) shape-function derivation.

Verifies:
  - Kronecker-delta property: Nᵢ(ξⱼ) = δᵢⱼ at all reference nodes
  - Partition of unity: Σᵢ Nᵢ(ξ) = 1 at interior points
  - Linear precision: shape functions reproduce linear fields exactly
  - Derivatives match finite differences
  - Shape functions and derivatives match the hard-coded _shape_functions catalog
  - The isoparametric geometry mapping reproduces node coordinates
"""
import numpy as np
import pytest

from apeGmsh.fem._basis_method import (
    BasisDerivedElement,
    tri3_basis_element,
    quad4_basis_element,
    tet4_basis_element,
)
from apeGmsh.fem._shape_functions import (
    tri3_N, tri3_dN,
    quad4_N, quad4_dN,
    tet4_N, tet4_dN,
)


# ------------------------------------------------------------------ helpers --

def fd_dN(elem: BasisDerivedElement, xi: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Finite-difference approximation of dN at a single reference point."""
    N0 = elem.N(xi[None])[0]
    grad = np.zeros((elem.n_nodes, elem.dim))
    for d in range(elem.dim):
        xi_p = xi.copy(); xi_p[d] += h
        grad[:, d] = (elem.N(xi_p[None])[0] - N0) / h
    return grad


def isoparametric_map(elem: BasisDerivedElement, nat: np.ndarray) -> np.ndarray:
    """x(ξ) = Σᵢ Nᵢ(ξ) · xᵢ  using the element's own reference nodes."""
    N = elem.N(nat)                   # (n_ip, n_nodes)
    return N @ elem.ref_nodes         # (n_ip, dim)


# ------------------------------------------------------------------ Tri3 ---

class TestTri3:
    @pytest.fixture
    def elem(self):
        return tri3_basis_element()

    def test_kronecker_delta(self, elem):
        N = elem.N(elem.ref_nodes)            # (3, 3)
        np.testing.assert_allclose(N, np.eye(3), atol=1e-14)

    def test_partition_of_unity(self, elem):
        pts = np.array([[0.2, 0.2], [0.5, 0.3], [0.1, 0.7]])
        np.testing.assert_allclose(elem.N(pts).sum(axis=1), 1.0, atol=1e-14)

    def test_matches_hardcoded_N(self, elem):
        pts = np.array([[0.2, 0.1], [0.5, 0.3], [0.0, 0.5]])
        np.testing.assert_allclose(elem.N(pts), tri3_N(pts), atol=1e-14)

    def test_matches_hardcoded_dN(self, elem):
        pts = np.array([[0.2, 0.1], [0.5, 0.3], [0.0, 0.5]])
        np.testing.assert_allclose(elem.dN(pts), tri3_dN(pts), atol=1e-14)

    def test_derivatives_fd(self, elem):
        xi = np.array([0.3, 0.2])
        np.testing.assert_allclose(elem.dN(xi[None])[0], fd_dN(elem, xi), atol=1e-5)

    def test_linear_precision(self, elem):
        """N @ node_values reproduces any linear field exactly."""
        node_vals = np.array([2.0, -1.0, 3.0])
        pts = np.array([[0.25, 0.25], [0.4, 0.1], [0.0, 0.0]])
        result = elem.N(pts) @ node_vals
        xi, eta = pts[:, 0], pts[:, 1]
        expected = node_vals[0] * (1 - xi - eta) + node_vals[1] * xi + node_vals[2] * eta
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_isoparametric_geometry_map(self, elem):
        """x = N @ X reproduces reference-node coordinates at reference points."""
        pts = np.array([[0.3, 0.3], [0.5, 0.1]])
        x = isoparametric_map(elem, pts)
        np.testing.assert_allclose(x, pts, atol=1e-14)


# ------------------------------------------------------------------ Quad4 ---

class TestQuad4:
    @pytest.fixture
    def elem(self):
        return quad4_basis_element()

    def test_kronecker_delta(self, elem):
        N = elem.N(elem.ref_nodes)            # (4, 4)
        np.testing.assert_allclose(N, np.eye(4), atol=1e-14)

    def test_partition_of_unity(self, elem):
        pts = np.array([[0.0, 0.0], [0.5, 0.3], [-0.4, 0.7]])
        np.testing.assert_allclose(elem.N(pts).sum(axis=1), 1.0, atol=1e-14)

    def test_matches_hardcoded_N(self, elem):
        pts = np.array([[0.0, 0.0], [0.5, -0.3], [-0.1, 0.6]])
        np.testing.assert_allclose(elem.N(pts), quad4_N(pts), atol=1e-14)

    def test_matches_hardcoded_dN(self, elem):
        pts = np.array([[0.0, 0.0], [0.5, -0.3], [-0.1, 0.6]])
        np.testing.assert_allclose(elem.dN(pts), quad4_dN(pts), atol=1e-14)

    def test_derivatives_fd(self, elem):
        xi = np.array([0.3, -0.2])
        np.testing.assert_allclose(elem.dN(xi[None])[0], fd_dN(elem, xi), atol=1e-5)

    def test_isoparametric_geometry_map(self, elem):
        pts = np.array([[0.5, 0.5], [-0.3, 0.7]])
        x = isoparametric_map(elem, pts)
        np.testing.assert_allclose(x, pts, atol=1e-14)


# ------------------------------------------------------------------ Tet4 ---

class TestTet4:
    @pytest.fixture
    def elem(self):
        return tet4_basis_element()

    def test_kronecker_delta(self, elem):
        N = elem.N(elem.ref_nodes)            # (4, 4)
        np.testing.assert_allclose(N, np.eye(4), atol=1e-14)

    def test_partition_of_unity(self, elem):
        pts = np.array([[0.1, 0.1, 0.1], [0.5, 0.2, 0.1]])
        np.testing.assert_allclose(elem.N(pts).sum(axis=1), 1.0, atol=1e-14)

    def test_matches_hardcoded_N(self, elem):
        pts = np.array([[0.2, 0.1, 0.3], [0.1, 0.5, 0.2]])
        np.testing.assert_allclose(elem.N(pts), tet4_N(pts), atol=1e-14)

    def test_matches_hardcoded_dN(self, elem):
        pts = np.array([[0.2, 0.1, 0.3], [0.1, 0.5, 0.2]])
        np.testing.assert_allclose(elem.dN(pts), tet4_dN(pts), atol=1e-14)

    def test_derivatives_fd(self, elem):
        xi = np.array([0.2, 0.1, 0.3])
        np.testing.assert_allclose(elem.dN(xi[None])[0], fd_dN(elem, xi), atol=1e-5)

    def test_isoparametric_geometry_map(self, elem):
        pts = np.array([[0.2, 0.3, 0.1], [0.1, 0.1, 0.6]])
        x = isoparametric_map(elem, pts)
        np.testing.assert_allclose(x, pts, atol=1e-14)


# ------------------------------------------------------------------ generic --

class TestBasisDerivedElementGeneric:
    def test_singular_basis_raises(self):
        """Duplicate node positions make V singular — must raise ValueError."""
        ref_nodes = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        exponents = np.array([[0, 0], [1, 0], [0, 1]])
        with pytest.raises(ValueError, match="near-singular"):
            BasisDerivedElement(ref_nodes, exponents)

    def test_n_nodes_and_dim(self):
        elem = quad4_basis_element()
        assert elem.n_nodes == 4
        assert elem.dim == 2

    def test_V_inv_times_V_is_identity(self):
        """V · V⁻¹ = I (round-trip sanity check)."""
        elem = tri3_basis_element()
        V = elem._eval_monomials(elem.ref_nodes)
        np.testing.assert_allclose(V @ elem.V_inv, np.eye(3), atol=1e-14)
