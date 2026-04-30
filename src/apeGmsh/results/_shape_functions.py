"""Shape functions and derivatives for fixed-class FE elements.

Mined from `STKO_to_python <https://github.com/nmorabowen/STKO_to_python>`_
(MIT, same author) and adapted to be **keyed by Gmsh element-type
codes** (matching apeGmsh's ``ElementTypeInfo.code``) so the catalog
can be looked up directly from a ``FEMData.elements`` group:

* code  1 — Line2     (2-node line)         — line geom
* code  2 — Tri3      (3-node triangle)     — shell geom
* code  3 — Quad4     (4-node bilinear quad) — shell geom
* code  4 — Tet4      (4-node tetrahedron)  — solid geom
* code  5 — Hex8      (8-node trilinear hex) — solid geom

Each catalog entry is a triple ``(N_fn, dN_fn, geom_kind)`` where:

* ``N_fn(nat_coords)`` — ``(n_ip, n_nodes)`` shape-function values
* ``dN_fn(nat_coords)`` — ``(n_ip, n_nodes, parent_dim)`` derivatives
* ``geom_kind`` ∈ ``{"line", "shell", "solid"}`` — selects the
  Jacobian-determinant formula used by ``compute_jacobian_dets``.

Conventions
-----------
* Quad4 / Hex8 — natural coords in ``[-1, +1]^d``, OpenSees CCW node
  order. Hex8 is bottom-face CCW (1..4 at ζ=−1) then top-face CCW
  (5..8 at ζ=+1).
* Tri3 / Tet4 — natural coords in barycentric space with vertices at
  the origin and the unit-axis points (``[0,0]/[1,0]/[0,1]`` and
  ``[0,0,0]/[1,0,0]/[0,1,0]/[0,0,1]``). ``N_0 = 1 − Σ ξ_i``.
* Line2 — natural ξ ∈ ``[-1, +1]``.

If a given mesh's connectivity uses a different node order than the
catalog assumes, override the entry rather than re-deriving the
shape functions globally.

Vectorized helpers
------------------
``compute_physical_coords(natural_coords, element_node_coords, N_fn)``
maps batched (n_ip, parent_dim) IP positions through (n_elements,
n_nodes, 3) node-coords using one ``einsum`` — useful when many
elements share the same IP layout.

``compute_jacobian_dets(...)`` returns per-element-per-IP measures
suitable for integration (volume, surface, length).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


__all__ = [
    "SHAPE_FUNCTIONS_BY_GMSH_CODE",
    "get_shape_functions",
    "compute_physical_coords",
    "compute_jacobian_dets",
    # Reusable shape-function primitives — exported so users can
    # plug them into custom catalog keys when handling element types
    # the catalog doesn't yet cover.
    "line2_N",
    "line2_dN",
    "tri3_N",
    "tri3_dN",
    "quad4_N",
    "quad4_dN",
    "tet4_N",
    "tet4_dN",
    "hex8_N",
    "hex8_dN",
]


ShapeFn = Callable[[np.ndarray], np.ndarray]
GeomKind = str    # "line" | "shell" | "solid"


# Gmsh element-type codes
_LINE2 = 1
_TRI3 = 2
_QUAD4 = 3
_TET4 = 4
_HEX8 = 5


# --------------------------------------------------------------------- #
# Line2 — 2-node linear segment (line)                                  #
# --------------------------------------------------------------------- #
#
# Node ordering (natural):
#   1: -1
#   2: +1


def line2_N(nat: np.ndarray) -> np.ndarray:
    """Linear-line shape functions — shape ``(n_ip, 2)``.

    Accepts ``nat`` as ``(n_ip,)`` or ``(n_ip, 1)``.
    """
    nat = np.asarray(nat, dtype=np.float64)
    if nat.ndim == 1:
        xi = nat
    else:
        xi = nat[:, 0]
    return 0.5 * np.stack([1.0 - xi, 1.0 + xi], axis=1)


def line2_dN(nat: np.ndarray) -> np.ndarray:
    """Line2 derivatives — shape ``(n_ip, 2, 1)`` (constant)."""
    nat = np.asarray(nat, dtype=np.float64)
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 2, 1), dtype=np.float64)
    out[:, 0, 0] = -0.5
    out[:, 1, 0] = +0.5
    return out


# --------------------------------------------------------------------- #
# Tri3 — 3-node linear triangle (shell)                                  #
# --------------------------------------------------------------------- #
#
# Parent: unit triangle with vertices at (0,0), (1,0), (0,1).
# Node ordering (natural):
#   1: (0, 0)
#   2: (1, 0)
#   3: (0, 1)


def tri3_N(nat: np.ndarray) -> np.ndarray:
    """Linear-triangle shape functions — shape ``(n_ip, 3)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    return np.stack([1.0 - xi - eta, xi, eta], axis=1)


def tri3_dN(nat: np.ndarray) -> np.ndarray:
    """Linear-triangle derivatives — shape ``(n_ip, 3, 2)`` (constant)."""
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 3, 2), dtype=np.float64)
    out[:, 0, 0] = -1.0; out[:, 0, 1] = -1.0
    out[:, 1, 0] = +1.0; out[:, 1, 1] = +0.0
    out[:, 2, 0] = +0.0; out[:, 2, 1] = +1.0
    return out


# --------------------------------------------------------------------- #
# Quad4 — 4-node bilinear quadrilateral (shell)                         #
# --------------------------------------------------------------------- #
#
# Node ordering (natural):
#   1: (-1, -1)
#   2: (+1, -1)
#   3: (+1, +1)
#   4: (-1, +1)


_QUAD4_NODE_SIGNS = np.array(
    [
        [-1, -1],
        [+1, -1],
        [+1, +1],
        [-1, +1],
    ],
    dtype=np.float64,
)


def quad4_N(nat: np.ndarray) -> np.ndarray:
    """Bilinear-quad shape functions — shape ``(n_ip, 4)``."""
    factors = 1.0 + _QUAD4_NODE_SIGNS[None, :, :] * nat[:, None, :]
    return 0.25 * np.prod(factors, axis=2)


def quad4_dN(nat: np.ndarray) -> np.ndarray:
    """Bilinear-quad derivatives — shape ``(n_ip, 4, 2)``."""
    factors = 1.0 + _QUAD4_NODE_SIGNS[None, :, :] * nat[:, None, :]
    out = np.empty((nat.shape[0], 4, 2), dtype=np.float64)
    for k in range(2):
        other = np.delete(factors, k, axis=2).prod(axis=2)
        out[:, :, k] = 0.25 * _QUAD4_NODE_SIGNS[None, :, k] * other
    return out


# --------------------------------------------------------------------- #
# Tet4 — 4-node linear tetrahedron (solid)                              #
# --------------------------------------------------------------------- #
#
# Parent: unit tetrahedron with vertices at the origin and the three
# unit-axis points.
# Node ordering (natural):
#   1: (0, 0, 0)
#   2: (1, 0, 0)
#   3: (0, 1, 0)
#   4: (0, 0, 1)


def tet4_N(nat: np.ndarray) -> np.ndarray:
    """Linear-tet shape functions — shape ``(n_ip, 4)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    return np.stack(
        [1.0 - xi - eta - zeta, xi, eta, zeta], axis=1,
    )


def tet4_dN(nat: np.ndarray) -> np.ndarray:
    """Linear-tet derivatives — shape ``(n_ip, 4, 3)`` (constant)."""
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 4, 3), dtype=np.float64)
    out[:, 0, :] = [-1.0, -1.0, -1.0]
    out[:, 1, :] = [+1.0,  0.0,  0.0]
    out[:, 2, :] = [ 0.0, +1.0,  0.0]
    out[:, 3, :] = [ 0.0,  0.0, +1.0]
    return out


# --------------------------------------------------------------------- #
# Hex8 — 8-node trilinear hex (solid)                                   #
# --------------------------------------------------------------------- #
#
# Node ordering (natural):
#   1: (-1, -1, -1)        5: (-1, -1, +1)
#   2: (+1, -1, -1)        6: (+1, -1, +1)
#   3: (+1, +1, -1)        7: (+1, +1, +1)
#   4: (-1, +1, -1)        8: (-1, +1, +1)


_HEX8_NODE_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float64,
)


def hex8_N(nat: np.ndarray) -> np.ndarray:
    """Trilinear-hex shape functions — shape ``(n_ip, 8)``."""
    factors = 1.0 + _HEX8_NODE_SIGNS[None, :, :] * nat[:, None, :]
    return 0.125 * np.prod(factors, axis=2)


def hex8_dN(nat: np.ndarray) -> np.ndarray:
    """Trilinear-hex derivatives — shape ``(n_ip, 8, 3)``."""
    factors = 1.0 + _HEX8_NODE_SIGNS[None, :, :] * nat[:, None, :]
    out = np.empty((nat.shape[0], 8, 3), dtype=np.float64)
    for k in range(3):
        other = np.delete(factors, k, axis=2).prod(axis=2)
        out[:, :, k] = 0.125 * _HEX8_NODE_SIGNS[None, :, k] * other
    return out


# --------------------------------------------------------------------- #
# Catalog                                                               #
# --------------------------------------------------------------------- #

SHAPE_FUNCTIONS_BY_GMSH_CODE: Dict[
    int, Tuple[ShapeFn, ShapeFn, GeomKind, int]
] = {
    # code: (N_fn, dN_fn, geom_kind, n_corner_nodes)
    _LINE2: (line2_N, line2_dN, "line", 2),
    _TRI3:  (tri3_N,  tri3_dN,  "shell", 3),
    _QUAD4: (quad4_N, quad4_dN, "shell", 4),
    _TET4:  (tet4_N,  tet4_dN,  "solid", 4),
    _HEX8:  (hex8_N,  hex8_dN,  "solid", 8),
}


def get_shape_functions(
    gmsh_code: int,
) -> Optional[Tuple[ShapeFn, ShapeFn, GeomKind, int]]:
    """Look up shape functions by Gmsh element-type code.

    Returns ``None`` for codes not in the catalog. Higher-order types
    (P2/P3) and prisms / pyramids fall through to ``None`` in v1; the
    caller can leave physical coords unset or fall back to a centroid
    approximation.
    """
    return SHAPE_FUNCTIONS_BY_GMSH_CODE.get(int(gmsh_code))


# --------------------------------------------------------------------- #
# Vectorized mapping                                                    #
# --------------------------------------------------------------------- #


def compute_physical_coords(
    natural_coords: np.ndarray,
    element_node_coords: np.ndarray,
    N_fn: ShapeFn,
) -> np.ndarray:
    """Map natural-coord IP positions to physical (x, y, z).

    Parameters
    ----------
    natural_coords : np.ndarray, shape ``(n_ip, parent_dim)``
        IP positions in the parent domain.
    element_node_coords : np.ndarray, shape ``(n_elements, n_nodes_per, 3)``
        Physical coordinates of each element's nodes — order must
        match the shape function's node ordering.
    N_fn : callable
        Shape-function evaluator returning ``(n_ip, n_nodes_per)``.

    Returns
    -------
    np.ndarray, shape ``(n_elements, n_ip, 3)``
    """
    N = N_fn(natural_coords)                # (n_ip, n_nodes)
    return np.einsum("in,enj->eij", N, element_node_coords)


def compute_jacobian_dets(
    natural_coords: np.ndarray,
    element_node_coords: np.ndarray,
    dN_fn: ShapeFn,
    geom_kind: GeomKind,
) -> np.ndarray:
    """Per-IP Jacobian determinants (or surface / line measures).

    * ``"solid"``: ``det(J)`` of the 3×3 ∂x/∂ξ matrix.
    * ``"shell"``: ``||∂x/∂ξ × ∂x/∂η||`` (surface area element).
    * ``"line"``:  ``||∂x/∂ξ||`` (length element).

    Always non-negative — sign-of-determinant errors raise here, not
    silently flip the integral.
    """
    dN = dN_fn(natural_coords)              # (n_ip, n_nodes, parent_dim)
    J = np.einsum("ink,ena->eika", dN, element_node_coords)
    # J shape: (n_elements, n_ip, parent_dim, 3)
    if geom_kind == "solid":
        return np.abs(np.linalg.det(J))
    if geom_kind == "shell":
        cross = np.cross(J[..., 0, :], J[..., 1, :])
        return np.linalg.norm(cross, axis=-1)
    if geom_kind == "line":
        return np.linalg.norm(J[..., 0, :], axis=-1)
    raise ValueError(
        f"Unknown geom_kind {geom_kind!r}; expected "
        f"'solid', 'shell', or 'line'."
    )
