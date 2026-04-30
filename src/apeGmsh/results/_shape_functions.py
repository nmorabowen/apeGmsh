"""Shape functions and derivatives for fixed-class FE elements.

Mined from `STKO_to_python <https://github.com/nmorabowen/STKO_to_python>`_
(MIT, same author) and adapted to be **keyed by Gmsh element-type
codes** (matching apeGmsh's ``ElementTypeInfo.code``) so the catalog
can be looked up directly from a ``FEMData.elements`` group:

* code  1 — Line2    (2-node line)             — line geom
* code  2 — Tri3     (3-node linear triangle)  — shell geom
* code  3 — Quad4    (4-node bilinear quad)    — shell geom
* code  4 — Tet4     (4-node linear tet)       — solid geom
* code  5 — Hex8     (8-node trilinear hex)    — solid geom
* code  6 — Wedge6   (6-node linear prism)     — solid geom
* code  9 — Tri6     (6-node quadratic tri)    — shell geom
* code 10 — Quad9    (9-node Lagrangian quad)  — shell geom
* code 11 — Tet10    (10-node quadratic tet)   — solid geom
* code 12 — Hex27    (27-node Lagrangian hex)  — solid geom
* code 16 — Quad8    (8-node serendipity quad) — shell geom
* code 17 — Hex20    (20-node serendipity hex) — solid geom

Each catalog entry is a tuple ``(N_fn, dN_fn, geom_kind, n_corner)`` where:

* ``N_fn(nat_coords)`` — ``(n_ip, n_nodes)`` shape-function values
* ``dN_fn(nat_coords)`` — ``(n_ip, n_nodes, parent_dim)`` derivatives
* ``geom_kind`` ∈ ``{"line", "shell", "solid"}`` — selects the
  Jacobian-determinant formula used by ``compute_jacobian_dets``.
* ``n_corner`` — number of nodes the shape function takes (used to
  truncate connectivity, kept named "n_corner" for the linear case
  but for higher-order types it equals the total node count).

Conventions
-----------
Natural-coord conventions follow Gmsh exactly so a connectivity row
read from a Gmsh-generated mesh works without any reordering. Per-type
node orderings come from the ASCII diagrams in Gmsh's element-class
headers (``src/geo/M{Triangle,Quadrangle,Tetrahedron,Hexahedron,
Prism}.h``):

* Quad4 / Quad8 / Quad9 / Hex8 / Hex20 / Hex27 — natural coords in
  ``[-1, +1]^d``. Hex bottom-face CCW (corners 0..3 at ζ=−1) then
  top-face CCW (corners 4..7 at ζ=+1). Higher-order mid-edge / face
  / center nodes appended in Gmsh's published order.
* Tri3 / Tri6 / Tet4 / Tet10 — barycentric, vertices at the origin
  and the unit-axis points (``[0,0]/[1,0]/[0,1]`` and
  ``[0,0,0]/[1,0,0]/[0,1,0]/[0,0,1]``). ``N_0 = 1 − Σ ξ_i``.
* Wedge6 — tri × line tensor product: ``(ξ, η)`` in barycentric tri
  parent, ``ζ`` in ``[-1, +1]``. Bottom triangle nodes 0..2 at
  ζ=−1, top triangle nodes 3..5 at ζ=+1.
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
    "line2_N", "line2_dN",
    "tri3_N", "tri3_dN",
    "quad4_N", "quad4_dN",
    "tet4_N", "tet4_dN",
    "hex8_N", "hex8_dN",
    "wedge6_N", "wedge6_dN",
    "tri6_N", "tri6_dN",
    "quad9_N", "quad9_dN",
    "tet10_N", "tet10_dN",
    "hex27_N", "hex27_dN",
    "quad8_N", "quad8_dN",
    "hex20_N", "hex20_dN",
]


ShapeFn = Callable[[np.ndarray], np.ndarray]
GeomKind = str    # "line" | "shell" | "solid"


# Gmsh element-type codes
_LINE2 = 1
_TRI3 = 2
_QUAD4 = 3
_TET4 = 4
_HEX8 = 5
_WEDGE6 = 6
_TRI6 = 9
_QUAD9 = 10
_TET10 = 11
_HEX27 = 12
_QUAD8 = 16
_HEX20 = 17


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
# Wedge6 — 6-node linear prism (solid)                                  #
# --------------------------------------------------------------------- #
#
# Tri × line tensor product. Bottom triangle at ζ=−1, top at ζ=+1.
# Per Gmsh MPrism node ordering:
#   0: (0, 0, -1)        3: (0, 0, +1)
#   1: (1, 0, -1)        4: (1, 0, +1)
#   2: (0, 1, -1)        5: (0, 1, +1)


def wedge6_N(nat: np.ndarray) -> np.ndarray:
    """Linear-prism shape functions — shape ``(n_ip, 6)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta
    half_minus = 0.5 * (1.0 - zeta)
    half_plus = 0.5 * (1.0 + zeta)
    return np.stack([
        L0 * half_minus,
        L1 * half_minus,
        L2 * half_minus,
        L0 * half_plus,
        L1 * half_plus,
        L2 * half_plus,
    ], axis=1)


def wedge6_dN(nat: np.ndarray) -> np.ndarray:
    """Linear-prism derivatives — shape ``(n_ip, 6, 3)``."""
    n_ip = nat.shape[0]
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta
    hm = 0.5 * (1.0 - zeta)
    hp = 0.5 * (1.0 + zeta)
    out = np.zeros((n_ip, 6, 3), dtype=np.float64)
    # ∂L0/∂ξ = ∂L0/∂η = -1; ∂L1/∂ξ = 1; ∂L2/∂η = 1
    # Bottom triangle (nodes 0..2)
    out[:, 0, 0] = -hm;  out[:, 0, 1] = -hm;  out[:, 0, 2] = -0.5 * L0
    out[:, 1, 0] = +hm;  out[:, 1, 1] = 0.0;  out[:, 1, 2] = -0.5 * L1
    out[:, 2, 0] = 0.0;  out[:, 2, 1] = +hm;  out[:, 2, 2] = -0.5 * L2
    # Top triangle (nodes 3..5)
    out[:, 3, 0] = -hp;  out[:, 3, 1] = -hp;  out[:, 3, 2] = +0.5 * L0
    out[:, 4, 0] = +hp;  out[:, 4, 1] = 0.0;  out[:, 4, 2] = +0.5 * L1
    out[:, 5, 0] = 0.0;  out[:, 5, 1] = +hp;  out[:, 5, 2] = +0.5 * L2
    return out


# --------------------------------------------------------------------- #
# Tri6 — 6-node quadratic triangle (shell)                              #
# --------------------------------------------------------------------- #
#
# Per Gmsh MTriangle6 node ordering:
#   0: (0, 0)        3: (0.5, 0)        — mid 0-1
#   1: (1, 0)        4: (0.5, 0.5)      — mid 1-2
#   2: (0, 1)        5: (0,   0.5)      — mid 0-2


def tri6_N(nat: np.ndarray) -> np.ndarray:
    """Quadratic-triangle shape functions — shape ``(n_ip, 6)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta
    return np.stack([
        L0 * (2.0 * L0 - 1.0),
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        4.0 * L0 * L1,
        4.0 * L1 * L2,
        4.0 * L0 * L2,
    ], axis=1)


def tri6_dN(nat: np.ndarray) -> np.ndarray:
    """Quadratic-triangle derivatives — shape ``(n_ip, 6, 2)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    L0 = 1.0 - xi - eta
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 6, 2), dtype=np.float64)
    # Corners — chain rule on Li(2Li-1) with dL0=(-1,-1), dL1=(1,0), dL2=(0,1)
    out[:, 0, 0] = -(4.0 * L0 - 1.0);    out[:, 0, 1] = -(4.0 * L0 - 1.0)
    out[:, 1, 0] = +(4.0 * xi - 1.0);    out[:, 1, 1] = 0.0
    out[:, 2, 0] = 0.0;                  out[:, 2, 1] = +(4.0 * eta - 1.0)
    # Mid-edges — 4 L_a L_b
    out[:, 3, 0] = 4.0 * (L0 - xi);      out[:, 3, 1] = -4.0 * xi
    out[:, 4, 0] = 4.0 * eta;            out[:, 4, 1] = 4.0 * xi
    out[:, 5, 0] = -4.0 * eta;           out[:, 5, 1] = 4.0 * (L0 - eta)
    return out


# --------------------------------------------------------------------- #
# Quad9 — 9-node Lagrangian quadrilateral (shell)                       #
# --------------------------------------------------------------------- #
#
# 1-D Lagrange basis at nodes (-1, 0, +1):
#   L_minus(t) = t(t-1)/2     [equals 1 at t=-1, 0 at 0 and +1]
#   L_zero(t)  = 1 - t²        [equals 1 at t=0,  0 at ±1]
#   L_plus(t)  = t(t+1)/2     [equals 1 at t=+1, 0 at -1 and 0]
#
# Per Gmsh MQuadrangle9:
#   0: (-1,-1)   3: (-1,+1)   6: ( 0,+1)
#   1: (+1,-1)   4: ( 0,-1)   7: (-1, 0)
#   2: (+1,+1)   5: (+1, 0)   8: ( 0, 0)


def _lagrange_1d(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        0.5 * t * (t - 1.0),    # L_minus
        1.0 - t * t,            # L_zero
        0.5 * t * (t + 1.0),    # L_plus
    )


def _lagrange_1d_d(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (t - 0.5, -2.0 * t, t + 0.5)


def quad9_N(nat: np.ndarray) -> np.ndarray:
    """Quad9 shape functions — shape ``(n_ip, 9)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    Lxm, Lx0, Lxp = _lagrange_1d(xi)
    Lym, Ly0, Lyp = _lagrange_1d(eta)
    return np.stack([
        Lxm * Lym,    # 0: (-1, -1)
        Lxp * Lym,    # 1: (+1, -1)
        Lxp * Lyp,    # 2: (+1, +1)
        Lxm * Lyp,    # 3: (-1, +1)
        Lx0 * Lym,    # 4: ( 0, -1)
        Lxp * Ly0,    # 5: (+1,  0)
        Lx0 * Lyp,    # 6: ( 0, +1)
        Lxm * Ly0,    # 7: (-1,  0)
        Lx0 * Ly0,    # 8: ( 0,  0)
    ], axis=1)


def quad9_dN(nat: np.ndarray) -> np.ndarray:
    """Quad9 derivatives — shape ``(n_ip, 9, 2)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    Lxm, Lx0, Lxp = _lagrange_1d(xi)
    Lym, Ly0, Lyp = _lagrange_1d(eta)
    dLxm, dLx0, dLxp = _lagrange_1d_d(xi)
    dLym, dLy0, dLyp = _lagrange_1d_d(eta)
    out = np.empty((nat.shape[0], 9, 2), dtype=np.float64)
    pairs = [
        (Lxm, dLxm, Lym, dLym),
        (Lxp, dLxp, Lym, dLym),
        (Lxp, dLxp, Lyp, dLyp),
        (Lxm, dLxm, Lyp, dLyp),
        (Lx0, dLx0, Lym, dLym),
        (Lxp, dLxp, Ly0, dLy0),
        (Lx0, dLx0, Lyp, dLyp),
        (Lxm, dLxm, Ly0, dLy0),
        (Lx0, dLx0, Ly0, dLy0),
    ]
    for i, (Lx, dLx, Ly, dLy) in enumerate(pairs):
        out[:, i, 0] = dLx * Ly
        out[:, i, 1] = Lx * dLy
    return out


# --------------------------------------------------------------------- #
# Tet10 — 10-node quadratic tetrahedron (solid)                         #
# --------------------------------------------------------------------- #
#
# Per Gmsh MTetrahedron10 node ordering:
#   0: (0, 0, 0)         4: (0.5, 0,   0)    — mid 0-1
#   1: (1, 0, 0)         5: (0.5, 0.5, 0)    — mid 1-2
#   2: (0, 1, 0)         6: (0,   0.5, 0)    — mid 0-2
#   3: (0, 0, 1)         7: (0,   0,   0.5)  — mid 0-3
#                        8: (0,   0.5, 0.5)  — mid 2-3
#                        9: (0.5, 0,   0.5)  — mid 1-3


def tet10_N(nat: np.ndarray) -> np.ndarray:
    """Quadratic-tet shape functions — shape ``(n_ip, 10)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    L0 = 1.0 - xi - eta - zeta
    L1 = xi
    L2 = eta
    L3 = zeta
    return np.stack([
        L0 * (2.0 * L0 - 1.0),
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        4.0 * L0 * L1,
        4.0 * L1 * L2,
        4.0 * L0 * L2,
        4.0 * L0 * L3,
        4.0 * L2 * L3,
        4.0 * L1 * L3,
    ], axis=1)


def tet10_dN(nat: np.ndarray) -> np.ndarray:
    """Quadratic-tet derivatives — shape ``(n_ip, 10, 3)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    L0 = 1.0 - xi - eta - zeta
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 10, 3), dtype=np.float64)
    # Corners — d[L_a(2L_a-1)] = (4L_a - 1) · dL_a
    coeff_0 = -(4.0 * L0 - 1.0)
    out[:, 0, 0] = coeff_0;   out[:, 0, 1] = coeff_0;   out[:, 0, 2] = coeff_0
    out[:, 1, 0] = 4.0 * xi - 1.0;   out[:, 1, 1] = 0.0;          out[:, 1, 2] = 0.0
    out[:, 2, 0] = 0.0;              out[:, 2, 1] = 4.0 * eta - 1.0;  out[:, 2, 2] = 0.0
    out[:, 3, 0] = 0.0;              out[:, 3, 1] = 0.0;          out[:, 3, 2] = 4.0 * zeta - 1.0
    # Mid-edges — d[4 L_a L_b] = 4(L_b dL_a + L_a dL_b)
    # Edge 0-1: 4 L0 L1; dL0=(-1,-1,-1), dL1=(1,0,0)
    out[:, 4, 0] = 4.0 * (L0 - xi);      out[:, 4, 1] = -4.0 * xi;        out[:, 4, 2] = -4.0 * xi
    # Edge 1-2: 4 L1 L2; dL1=(1,0,0), dL2=(0,1,0)
    out[:, 5, 0] = 4.0 * eta;            out[:, 5, 1] = 4.0 * xi;         out[:, 5, 2] = 0.0
    # Edge 0-2: 4 L0 L2; dL0=(-1,-1,-1), dL2=(0,1,0)
    out[:, 6, 0] = -4.0 * eta;           out[:, 6, 1] = 4.0 * (L0 - eta); out[:, 6, 2] = -4.0 * eta
    # Edge 0-3: 4 L0 L3; dL0=(-1,-1,-1), dL3=(0,0,1)
    out[:, 7, 0] = -4.0 * zeta;          out[:, 7, 1] = -4.0 * zeta;      out[:, 7, 2] = 4.0 * (L0 - zeta)
    # Edge 2-3: 4 L2 L3; dL2=(0,1,0), dL3=(0,0,1)
    out[:, 8, 0] = 0.0;                  out[:, 8, 1] = 4.0 * zeta;       out[:, 8, 2] = 4.0 * eta
    # Edge 1-3: 4 L1 L3; dL1=(1,0,0), dL3=(0,0,1)
    out[:, 9, 0] = 4.0 * zeta;           out[:, 9, 1] = 0.0;              out[:, 9, 2] = 4.0 * xi
    return out


# --------------------------------------------------------------------- #
# Hex27 — 27-node Lagrangian hex (solid)                                #
# --------------------------------------------------------------------- #
#
# Tensor product of three 1-D Lagrange triples on (ξ, η, ζ).
#
# Per Gmsh MHexahedron27 node ordering: corners 0..7 (same as Hex8),
# then 12 mid-edge nodes 8..19 (same as Hex20), then 6 face-center
# nodes and 1 volume center:
#   20: (0, 0, -1)   bottom face (ζ=-1)
#   21: (0,-1,  0)   back face   (η=-1)
#   22: (-1,0,  0)   left face   (ξ=-1)
#   23: (+1,0,  0)   right face  (ξ=+1)
#   24: (0,+1,  0)   front face  (η=+1)
#   25: (0, 0, +1)   top face    (ζ=+1)
#   26: (0, 0,  0)   volume center


# Per-node 1-D Lagrange index for hex27. Each row picks one of
# {-1, 0, +1} for ξ, η, ζ. The basis is then L[idx_ξ](ξ) · L[idx_η](η)
# · L[idx_ζ](ζ) where L[-1]=Lminus, L[0]=Lzero, L[+1]=Lplus.
_HEX27_LAGRANGE_INDEX = np.array([
    # 8 corners
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1],
    # 12 mid-edges (same order as Hex20)
    [ 0, -1, -1],
    [-1,  0, -1],
    [-1, -1,  0],
    [+1,  0, -1],
    [+1, -1,  0],
    [ 0, +1, -1],
    [+1, +1,  0],
    [-1, +1,  0],
    [ 0, -1, +1],
    [-1,  0, +1],
    [+1,  0, +1],
    [ 0, +1, +1],
    # 6 face centers + 1 volume center
    [ 0,  0, -1],    # 20
    [ 0, -1,  0],    # 21
    [-1,  0,  0],    # 22
    [+1,  0,  0],    # 23
    [ 0, +1,  0],    # 24
    [ 0,  0, +1],    # 25
    [ 0,  0,  0],    # 26
], dtype=np.int64)


def _hex27_eval(nat: np.ndarray, want_deriv: bool):
    """Evaluate L[idx_ξ](ξ) etc. for every (node, ip). Returns
    (Lx, Ly, Lz) arrays of shape (n_ip, 27) — and if ``want_deriv``,
    the same triple of derivatives.
    """
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    Lxm, Lx0, Lxp = _lagrange_1d(xi)
    Lym, Ly0, Lyp = _lagrange_1d(eta)
    Lzm, Lz0, Lzp = _lagrange_1d(zeta)
    L_x_table = np.stack([Lxm, Lx0, Lxp], axis=0)    # (3, n_ip)
    L_y_table = np.stack([Lym, Ly0, Lyp], axis=0)
    L_z_table = np.stack([Lzm, Lz0, Lzp], axis=0)
    # Map index in {-1, 0, +1} to row 0/1/2 in the table.
    rows_x = _HEX27_LAGRANGE_INDEX[:, 0] + 1
    rows_y = _HEX27_LAGRANGE_INDEX[:, 1] + 1
    rows_z = _HEX27_LAGRANGE_INDEX[:, 2] + 1
    Lx = L_x_table[rows_x]    # (27, n_ip)
    Ly = L_y_table[rows_y]
    Lz = L_z_table[rows_z]
    if not want_deriv:
        return Lx, Ly, Lz, None, None, None
    dLxm, dLx0, dLxp = _lagrange_1d_d(xi)
    dLym, dLy0, dLyp = _lagrange_1d_d(eta)
    dLzm, dLz0, dLzp = _lagrange_1d_d(zeta)
    dL_x_table = np.stack([dLxm, dLx0, dLxp], axis=0)
    dL_y_table = np.stack([dLym, dLy0, dLyp], axis=0)
    dL_z_table = np.stack([dLzm, dLz0, dLzp], axis=0)
    dLx = dL_x_table[rows_x]
    dLy = dL_y_table[rows_y]
    dLz = dL_z_table[rows_z]
    return Lx, Ly, Lz, dLx, dLy, dLz


def hex27_N(nat: np.ndarray) -> np.ndarray:
    """Hex27 shape functions — shape ``(n_ip, 27)``."""
    Lx, Ly, Lz, _, _, _ = _hex27_eval(nat, want_deriv=False)
    # (27, n_ip) -> (n_ip, 27)
    return (Lx * Ly * Lz).T


def hex27_dN(nat: np.ndarray) -> np.ndarray:
    """Hex27 derivatives — shape ``(n_ip, 27, 3)``."""
    Lx, Ly, Lz, dLx, dLy, dLz = _hex27_eval(nat, want_deriv=True)
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 27, 3), dtype=np.float64)
    out[:, :, 0] = (dLx * Ly * Lz).T
    out[:, :, 1] = (Lx * dLy * Lz).T
    out[:, :, 2] = (Lx * Ly * dLz).T
    return out


# --------------------------------------------------------------------- #
# Quad8 — 8-node serendipity quadrilateral (shell)                      #
# --------------------------------------------------------------------- #
#
# Per Gmsh MQuadrangle8:
#   0: (-1,-1)   3: (-1,+1)
#   1: (+1,-1)   4: ( 0,-1)    — mid 0-1, varying ξ at η=-1
#   2: (+1,+1)   5: (+1, 0)    — mid 1-2, varying η at ξ=+1
#                6: ( 0,+1)    — mid 2-3, varying ξ at η=+1
#                7: (-1, 0)    — mid 3-0, varying η at ξ=-1


def quad8_N(nat: np.ndarray) -> np.ndarray:
    """Quad8 serendipity shape functions — shape ``(n_ip, 8)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    one_m_xi = 1.0 - xi
    one_p_xi = 1.0 + xi
    one_m_eta = 1.0 - eta
    one_p_eta = 1.0 + eta
    one_m_xi2 = 1.0 - xi * xi
    one_m_eta2 = 1.0 - eta * eta
    return np.stack([
        # Corners: 1/4 (1+ξ_i ξ)(1+η_i η)(ξ_i ξ + η_i η - 1)
        0.25 * one_m_xi * one_m_eta * (-xi - eta - 1.0),
        0.25 * one_p_xi * one_m_eta * (+xi - eta - 1.0),
        0.25 * one_p_xi * one_p_eta * (+xi + eta - 1.0),
        0.25 * one_m_xi * one_p_eta * (-xi + eta - 1.0),
        # Mid-edges
        0.5 * one_m_xi2 * one_m_eta,    # 4: ( 0,-1)
        0.5 * one_p_xi  * one_m_eta2,   # 5: (+1, 0)
        0.5 * one_m_xi2 * one_p_eta,    # 6: ( 0,+1)
        0.5 * one_m_xi  * one_m_eta2,   # 7: (-1, 0)
    ], axis=1)


def quad8_dN(nat: np.ndarray) -> np.ndarray:
    """Quad8 serendipity derivatives — shape ``(n_ip, 8, 2)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 8, 2), dtype=np.float64)
    # Corners — derived by taking d/dξ and d/dη of N_i above
    out[:, 0, 0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
    out[:, 0, 1] = 0.25 * (1.0 - xi)  * (xi + 2.0 * eta)
    out[:, 1, 0] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
    out[:, 1, 1] = 0.25 * (1.0 + xi)  * (-xi + 2.0 * eta)
    out[:, 2, 0] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
    out[:, 2, 1] = 0.25 * (1.0 + xi)  * (xi + 2.0 * eta)
    out[:, 3, 0] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
    out[:, 3, 1] = 0.25 * (1.0 - xi)  * (-xi + 2.0 * eta)
    # Mid-edges
    out[:, 4, 0] = -xi * (1.0 - eta)
    out[:, 4, 1] = -0.5 * (1.0 - xi * xi)
    out[:, 5, 0] = +0.5 * (1.0 - eta * eta)
    out[:, 5, 1] = -eta * (1.0 + xi)
    out[:, 6, 0] = -xi * (1.0 + eta)
    out[:, 6, 1] = +0.5 * (1.0 - xi * xi)
    out[:, 7, 0] = -0.5 * (1.0 - eta * eta)
    out[:, 7, 1] = -eta * (1.0 - xi)
    return out


# --------------------------------------------------------------------- #
# Hex20 — 20-node serendipity hex (solid)                               #
# --------------------------------------------------------------------- #
#
# Per Gmsh MHexahedron20: corners 0..7 (same as Hex8); then 12 mid-
# edge nodes 8..19, each with one natural-coord component = 0:
#   8: ( 0,-1,-1) — varying ξ      14: (+1,+1, 0) — varying ζ
#   9: (-1, 0,-1) — varying η      15: (-1,+1, 0) — varying ζ
#   10:(-1,-1, 0) — varying ζ      16: ( 0,-1,+1) — varying ξ
#   11:(+1, 0,-1) — varying η      17: (-1, 0,+1) — varying η
#   12:(+1,-1, 0) — varying ζ      18: (+1, 0,+1) — varying η
#   13:( 0,+1,-1) — varying ξ      19: ( 0,+1,+1) — varying ξ


# Corner natural-coord signs (8, 3) — reuse the existing Hex8 table.
# Mid-edge entries: (varying-axis-index, sign_a, sign_b) where
# sign_a / sign_b are the ±1 of the two non-varying axes, in axis
# order (e.g. for varying-axis=0, the two non-varying are η, ζ).
_HEX20_MIDEDGE = np.array([
    # node-id, axis, η_or_ξ_sign, ζ_or_other_sign — one row per edge
    # axis: 0 = ξ varies, 1 = η varies, 2 = ζ varies
    [8,  0, -1, -1],   # ( 0,-1,-1)
    [9,  1, -1, -1],   # (-1, 0,-1)
    [10, 2, -1, -1],   # (-1,-1, 0)
    [11, 1, +1, -1],   # (+1, 0,-1)
    [12, 2, +1, -1],   # (+1,-1, 0)
    [13, 0, +1, -1],   # ( 0,+1,-1)
    [14, 2, +1, +1],   # (+1,+1, 0)
    [15, 2, -1, +1],   # (-1,+1, 0)
    [16, 0, -1, +1],   # ( 0,-1,+1)
    [17, 1, -1, +1],   # (-1, 0,+1)
    [18, 1, +1, +1],   # (+1, 0,+1)
    [19, 0, +1, +1],   # ( 0,+1,+1)
], dtype=np.int64)


def hex20_N(nat: np.ndarray) -> np.ndarray:
    """Hex20 serendipity shape functions — shape ``(n_ip, 20)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 20), dtype=np.float64)

    # Corners 0..7
    s = _HEX8_NODE_SIGNS    # (8, 3)
    a = 1.0 + s[:, 0][None, :] * xi[:, None]      # (n_ip, 8)
    b = 1.0 + s[:, 1][None, :] * eta[:, None]
    c = 1.0 + s[:, 2][None, :] * zeta[:, None]
    extra = (
        s[:, 0][None, :] * xi[:, None]
        + s[:, 1][None, :] * eta[:, None]
        + s[:, 2][None, :] * zeta[:, None]
        - 2.0
    )
    out[:, 0:8] = 0.125 * a * b * c * extra

    # Mid-edges 8..19
    for nid, axis, sb, sc in _HEX20_MIDEDGE:
        if axis == 0:
            # varying ξ: 1/4 (1-ξ²)(1 + s_b η)(1 + s_c ζ)
            out[:, nid] = 0.25 * (1.0 - xi * xi) * (1.0 + sb * eta) * (1.0 + sc * zeta)
        elif axis == 1:
            # varying η: 1/4 (1 + s_b ξ)(1-η²)(1 + s_c ζ)
            out[:, nid] = 0.25 * (1.0 + sb * xi) * (1.0 - eta * eta) * (1.0 + sc * zeta)
        else:
            # varying ζ: 1/4 (1 + s_b ξ)(1 + s_c η)(1-ζ²)
            out[:, nid] = 0.25 * (1.0 + sb * xi) * (1.0 + sc * eta) * (1.0 - zeta * zeta)
    return out


def hex20_dN(nat: np.ndarray) -> np.ndarray:
    """Hex20 serendipity derivatives — shape ``(n_ip, 20, 3)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 20, 3), dtype=np.float64)

    # Corners 0..7 — N_i = 1/8 (1+s_x ξ)(1+s_y η)(1+s_z ζ)(s_x ξ + s_y η + s_z ζ - 2)
    # ∂/∂ξ = 1/8 [s_x (1+s_y η)(1+s_z ζ)(s_x ξ + s_y η + s_z ζ - 2)
    #             + (1+s_x ξ)(1+s_y η)(1+s_z ζ) · s_x]
    s = _HEX8_NODE_SIGNS    # (8, 3)
    a = 1.0 + s[:, 0][None, :] * xi[:, None]
    b = 1.0 + s[:, 1][None, :] * eta[:, None]
    c = 1.0 + s[:, 2][None, :] * zeta[:, None]
    extra = (
        s[:, 0][None, :] * xi[:, None]
        + s[:, 1][None, :] * eta[:, None]
        + s[:, 2][None, :] * zeta[:, None]
        - 2.0
    )
    out[:, 0:8, 0] = 0.125 * s[:, 0][None, :] * b * c * (extra + a)
    out[:, 0:8, 1] = 0.125 * s[:, 1][None, :] * a * c * (extra + b)
    out[:, 0:8, 2] = 0.125 * s[:, 2][None, :] * a * b * (extra + c)

    # Mid-edges
    one_m_xi2 = 1.0 - xi * xi
    one_m_eta2 = 1.0 - eta * eta
    one_m_zeta2 = 1.0 - zeta * zeta
    for nid, axis, sb, sc in _HEX20_MIDEDGE:
        if axis == 0:
            # 1/4 (1-ξ²)(1 + s_b η)(1 + s_c ζ)
            out[:, nid, 0] = 0.25 * (-2.0 * xi)         * (1.0 + sb * eta)  * (1.0 + sc * zeta)
            out[:, nid, 1] = 0.25 * one_m_xi2           * sb                * (1.0 + sc * zeta)
            out[:, nid, 2] = 0.25 * one_m_xi2           * (1.0 + sb * eta)  * sc
        elif axis == 1:
            out[:, nid, 0] = 0.25 * sb                  * one_m_eta2        * (1.0 + sc * zeta)
            out[:, nid, 1] = 0.25 * (1.0 + sb * xi)     * (-2.0 * eta)      * (1.0 + sc * zeta)
            out[:, nid, 2] = 0.25 * (1.0 + sb * xi)     * one_m_eta2        * sc
        else:
            out[:, nid, 0] = 0.25 * sb                  * (1.0 + sc * eta)  * one_m_zeta2
            out[:, nid, 1] = 0.25 * (1.0 + sb * xi)     * sc                * one_m_zeta2
            out[:, nid, 2] = 0.25 * (1.0 + sb * xi)     * (1.0 + sc * eta)  * (-2.0 * zeta)
    return out


# --------------------------------------------------------------------- #
# Catalog                                                               #
# --------------------------------------------------------------------- #

SHAPE_FUNCTIONS_BY_GMSH_CODE: Dict[
    int, Tuple[ShapeFn, ShapeFn, GeomKind, int]
] = {
    # code: (N_fn, dN_fn, geom_kind, n_nodes)
    # The fourth field is the total number of nodes the shape function
    # takes (kept named ``n_corner`` historically; for higher-order
    # types it equals the full node count, mid-edge / face / center
    # nodes included).
    _LINE2:  (line2_N,  line2_dN,  "line",  2),
    _TRI3:   (tri3_N,   tri3_dN,   "shell", 3),
    _QUAD4:  (quad4_N,  quad4_dN,  "shell", 4),
    _TET4:   (tet4_N,   tet4_dN,   "solid", 4),
    _HEX8:   (hex8_N,   hex8_dN,   "solid", 8),
    _WEDGE6: (wedge6_N, wedge6_dN, "solid", 6),
    _TRI6:   (tri6_N,   tri6_dN,   "shell", 6),
    _QUAD9:  (quad9_N,  quad9_dN,  "shell", 9),
    _TET10:  (tet10_N,  tet10_dN,  "solid", 10),
    _HEX27:  (hex27_N,  hex27_dN,  "solid", 27),
    _QUAD8:  (quad8_N,  quad8_dN,  "shell", 8),
    _HEX20:  (hex20_N,  hex20_dN,  "solid", 20),
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
