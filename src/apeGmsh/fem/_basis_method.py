"""Basis-function (Vandermonde) method for deriving FE shape functions.

The **basis-function method** builds shape functions by inverting the
Vandermonde matrix formed by evaluating n polynomial basis functions φₖ at
n reference-element nodes ξᵢ:

    V_ij = φⱼ(ξᵢ),    N(ξ)ᵀ = φ(ξ)ᵀ V⁻¹

Column i of V⁻¹ gives the polynomial coefficients for Nᵢ, ensuring the
Kronecker-delta property  Nᵢ(ξⱼ) = δᵢⱼ.

Isoparametric elements
----------------------
An element is **isoparametric** when the same shape functions N used for
field interpolation

    u(ξ) = Σᵢ Nᵢ(ξ) uᵢ

also map the geometry from the reference domain to the physical domain:

    x(ξ) = Σᵢ Nᵢ(ξ) xᵢ

All elements in this module are isoparametric: the polynomial basis is
chosen in the *natural* (reference) coordinate system, and the resulting
N functions serve both the geometry-mapping and the field-interpolation
roles identically.

Tri3 (3-node triangle)
    Basis [1, ξ, η] on the unit triangle.  The geometry mapping is
    **affine** (constant Jacobian), so any physical triangle is exactly
    representable and the strain is constant within the element (CST).

Quad4 (4-node quadrilateral)
    Basis [1, ξ, η, ξη] on [-1, +1]².  The geometry mapping is
    **bilinear** (Jacobian varies within the element), correctly mapping
    any physical quadrilateral at the cost of a non-constant strain field.

Tet4 (4-node tetrahedron)
    Basis [1, ξ, η, ζ] on the unit tetrahedron.  Affine mapping; constant
    strain throughout the element.

Monomial representation
-----------------------
Each basis function is a monomial encoded by a non-negative integer
exponent vector:

    φₖ(ξ) = ∏ᵢ ξᵢ^eₖᵢ,    exponents[k] = (eₖ₁, eₖ₂, ...)

Derivatives are computed analytically from the exponent array:

    ∂φₖ/∂ξd = eₖd · ξd^(eₖd − 1) · ∏_{i≠d} ξᵢ^eₖᵢ

Output conventions match ``_shape_functions.py``:

    N(nat)   nat ``(n_ip, dim)``  →  ``(n_ip, n_nodes)``
    dN(nat)  nat ``(n_ip, dim)``  →  ``(n_ip, n_nodes, dim)``
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BasisDerivedElement",
    "tri3_basis_element",
    "quad4_basis_element",
    "tet4_basis_element",
]


class BasisDerivedElement:
    """FE shape functions derived via the polynomial basis (Vandermonde) method.

    Parameters
    ----------
    ref_nodes:
        ``(n, dim)`` reference-element node coordinates.
    exponents:
        ``(n, dim)`` non-negative integer array.  Row k encodes
        ``φₖ(ξ) = ∏ᵢ ξᵢ^exponents[k, i]``.

    Attributes
    ----------
    n_nodes : int
    dim : int
    ref_nodes : ndarray  ``(n, dim)``
    exponents : ndarray  ``(n, dim)``
    V_inv : ndarray  ``(n, n)``
        Inverse Vandermonde matrix.  The shape function coefficients are
        its columns: ``N_i(ξ) = φ(ξ) · V_inv[:, i]``.
    """

    def __init__(self, ref_nodes: np.ndarray, exponents: np.ndarray) -> None:
        self.ref_nodes = np.asarray(ref_nodes, dtype=np.float64)
        self.exponents = np.asarray(exponents, dtype=np.int64)
        self.n_nodes, self.dim = self.ref_nodes.shape

        V = self._eval_monomials(self.ref_nodes)  # (n, n)
        cond = np.linalg.cond(V)
        if cond > 1e10:
            raise ValueError(
                f"Vandermonde matrix is near-singular (cond={cond:.2e}). "
                "Check that node positions are consistent with the chosen basis."
            )
        self.V_inv: np.ndarray = np.linalg.inv(V)

    # ------------------------------------------------------------------
    # Public interface — matches _shape_functions.py convention
    # ------------------------------------------------------------------

    def N(self, nat: np.ndarray) -> np.ndarray:
        """Shape functions at ``(n_ip, dim)`` natural-coord points.

        Returns ``(n_ip, n_nodes)``.  Satisfies Nᵢ(ξⱼ) = δᵢⱼ and
        Σᵢ Nᵢ(ξ) = 1 for any ξ inside the reference element.
        """
        nat = np.asarray(nat, dtype=np.float64)
        phi = self._eval_monomials(nat)   # (n_ip, n_basis)
        return phi @ self.V_inv           # (n_ip, n_nodes)

    def dN(self, nat: np.ndarray) -> np.ndarray:
        """Shape-function derivatives at ``(n_ip, dim)`` points.

        Returns ``(n_ip, n_nodes, dim)``.
        ``dN[ip, i, d] = ∂Nᵢ/∂ξd`` at ``nat[ip]``.
        """
        nat = np.asarray(nat, dtype=np.float64)
        dphi = self._eval_mono_derivs(nat)              # (n_ip, n_basis, dim)
        return np.einsum("ki,qkd->qid", self.V_inv, dphi)

    # ------------------------------------------------------------------
    # Internal monomial helpers
    # ------------------------------------------------------------------

    def _eval_monomials(self, nat: np.ndarray) -> np.ndarray:
        """Evaluate all n basis monomials at m points.

        ``nat`` ``(m, dim)``  →  ``(m, n_basis)``

        Uses ``nat[:, None, :] ** exponents[None, :, :]`` with numpy's
        convention that ``0 ** 0 = 1``, which is correct for monomials.
        """
        powers = nat[:, None, :] ** self.exponents[None, :, :]  # (m, n, dim)
        return np.prod(powers, axis=2)                           # (m, n)

    def _eval_mono_derivs(self, nat: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of all n monomials at m points.

        ``nat`` ``(m, dim)``  →  ``(m, n_basis, dim)``

        For each direction d only the monomials with exponents[k, d] > 0
        have a non-zero derivative; the rest stay at zero.
        """
        m = nat.shape[0]
        out = np.zeros((m, self.n_nodes, self.dim))

        for d in range(self.dim):
            mask = self.exponents[:, d] > 0
            if not np.any(mask):
                continue
            # Reduce exponent in direction d by 1, multiply by that exponent.
            exp_reduced = self.exponents[mask].copy()   # (n_mask, dim)
            exp_reduced[:, d] -= 1
            powers = nat[:, None, :] ** exp_reduced[None, :, :]  # (m, n_mask, dim)
            reduced_phi = np.prod(powers, axis=2)                  # (m, n_mask)
            coeffs = self.exponents[mask, d].astype(np.float64)    # (n_mask,)
            out[:, mask, d] = coeffs[None, :] * reduced_phi

        return out


# ------------------------------------------------------------------
# Pre-built standard elements
# ------------------------------------------------------------------

def tri3_basis_element() -> BasisDerivedElement:
    """3-node linear triangle (CST) in natural coords (ξ, η) ∈ unit triangle.

    Polynomial basis: [1, ξ, η].  Isoparametric with affine geometry
    mapping — any physical triangle is exactly represented.

    Node ordering matches Gmsh code 2:
      node 0: (0, 0),  node 1: (1, 0),  node 2: (0, 1)
    """
    ref_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    exponents = np.array([[0, 0], [1, 0], [0, 1]])
    return BasisDerivedElement(ref_nodes, exponents)


def quad4_basis_element() -> BasisDerivedElement:
    """4-node bilinear quadrilateral (Q4) in natural coords (ξ, η) ∈ [-1,+1]².

    Polynomial basis: [1, ξ, η, ξη].  Isoparametric with bilinear geometry
    mapping — any physical quadrilateral is exactly represented, but the
    Jacobian is not constant.

    Node ordering matches Gmsh code 3:
      node 0: (-1,-1),  node 1: (+1,-1),
      node 2: (+1,+1),  node 3: (-1,+1)
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    exponents = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    return BasisDerivedElement(ref_nodes, exponents)


def tet4_basis_element() -> BasisDerivedElement:
    """4-node linear tetrahedron in natural coords (ξ, η, ζ) ∈ unit tet.

    Polynomial basis: [1, ξ, η, ζ].  Isoparametric with affine geometry
    mapping — any physical tetrahedron is exactly represented and the
    strain is constant within the element.

    Node ordering matches Gmsh code 4:
      node 0: (0,0,0),  node 1: (1,0,0),
      node 2: (0,1,0),  node 3: (0,0,1)
    """
    ref_nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    exponents = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return BasisDerivedElement(ref_nodes, exponents)
