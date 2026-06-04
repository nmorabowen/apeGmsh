"""
apeGmsh FEM kernel — shared building blocks used by both the input side
(mesh, mass resolver) and the output side (results, viewers).

This package exists to give shape functions, quadrature rules, and other
element-aware primitives a single home that neither ``mesh/`` nor
``results/`` owns.  Both sides import from here.

Submodules:

``_shape_functions``
    Per-element-type shape functions ``N`` and derivatives ``dN``,
    isoparametric mapping helpers, Jacobian determinants, and the
    ``get_shape_functions(elem_type)`` dispatch.  Covers line2, tri3,
    tri6, quad4, quad8, quad9, tet4, tet10, hex8, hex20, hex27, wedge6.

``_quadrature``
    Minimal Gauss quadrature rules (1-D Gauss-Legendre, tensor-product
    quad/hex, triangle, tet, wedge) on the same reference domains the
    shape functions use.

``_hrz``
    HRZ (Hinton–Rock–Zienkiewicz) diagonal-scaling lumping weights —
    ``hrz_weights(gmsh_code)`` and the ``{volume,surface,line}_code``
    node-count → element-type helpers.  Used by the mass resolver's
    ``reduction='consistent'`` path.

``_basis_method``
    Basis-function (Vandermonde) method for deriving shape functions from
    a polynomial monomial basis.  Provides ``BasisDerivedElement`` plus
    pre-built ``tri3_basis_element``, ``quad4_basis_element``, and
    ``tet4_basis_element`` — all isoparametric, matching the node ordering
    and natural-coordinate conventions of ``_shape_functions``.
"""
from __future__ import annotations

from . import _basis_method, _hrz, _quadrature, _shape_functions

__all__ = ["_shape_functions", "_quadrature", "_hrz", "_basis_method"]
