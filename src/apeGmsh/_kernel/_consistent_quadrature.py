"""Gauss quadrature + shape-function integrals for consistent load reduction.

Given a single edge or face element (linear or higher order), compute
``w_i = ∫ N_i(ξ) · |J(ξ)| dξ`` for each node, where N_i is the element
shape function.  For constant distributed load q, the consistent
per-node nodal force is ``F_i = q · w_i``.

For surface elements, also returns ``n_i = ∫ N_i · (∂x/∂ξ × ∂x/∂η) dξdη``,
which multiplied by a pressure magnitude gives the consistent pressure
force on each node (oriented along the varying outward normal).

Supported node counts:
    line:  2 (line2), 3 (line3)
    face:  3 (tri3), 4 (quad4), 6 (tri6), 8 (quad8), 9 (quad9)

Out-of-range node counts raise ``NotImplementedError`` — never silently
fall back to a wrong rule.

Basis (ADR 0091)
----------------
``basis="lagrange"`` (default) integrates against the interpolatory
Lagrange shape functions of the gmsh element — correct when the target
element's DOFs are nodal values (TenNodeTetrahedron, SixNodeTri, ...).

``basis="bernstein"`` integrates against the quadratic Bernstein (Bézier)
basis instead — required when the target element's DOFs are **control
values** (the Ladruno fork's BezierTet10 / BezierTri6). Applying a
Lagrange-consistent vector to control values represents a strongly
oscillatory traction (exact resultant, locally spiking), which can drive
near-surface Gauss points of a pressure-sensitive material into
apex/tension. Node ordering follows the fork convention (verified against
``BezierTet10.cpp`` / ``BezierTri6.cpp``): the gmsh connectivity is used
verbatim, corners carry ``L_a²`` and the midside of edge ``(a, b)``
carries ``2·L_a·L_b`` — the face/edge trace of the volumetric basis. The
bernstein branch supports the simplex family only (line2/line3,
tri3/tri6); a quad face raises ``NotImplementedError``. For degree-1
elements (line2/tri3) the two bases coincide.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

_SUPPORTED_EDGE_NODES: tuple[int, ...] = (2, 3)
_SUPPORTED_FACE_NODES: tuple[int, ...] = (3, 4, 6, 8, 9)
_SUPPORTED_BASES: tuple[str, ...] = ("lagrange", "bernstein")
_SUPPORTED_BERNSTEIN_FACE_NODES: tuple[int, ...] = (3, 6)


# ---------------------------------------------------------------------
# Gauss rules
# ---------------------------------------------------------------------

def _gauss_1d_2() -> tuple[ndarray, ndarray]:
    s = 1.0 / np.sqrt(3.0)
    return np.array([-s, s]), np.array([1.0, 1.0])


def _gauss_1d_3() -> tuple[ndarray, ndarray]:
    s = np.sqrt(3.0 / 5.0)
    return (np.array([-s, 0.0, s]),
            np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]))


def _gauss_tri_6() -> tuple[ndarray, ndarray]:
    """Dunavant 6-point degree-4 rule on reference triangle (area 1/2)."""
    a, b = 0.445948490915965, 0.108103018168070
    c, d = 0.091576213509771, 0.816847572980459
    wa, wc = 0.223381589678011 / 2.0, 0.109951743655322 / 2.0
    pts = np.array([
        [a, b], [a, a], [b, a],
        [c, d], [c, c], [d, c],
    ])
    wts = np.array([wa, wa, wa, wc, wc, wc])
    return pts, wts


def _gauss_quad_3x3() -> tuple[ndarray, ndarray]:
    xi, w = _gauss_1d_3()
    XI, ETA = np.meshgrid(xi, xi, indexing="ij")
    W = np.outer(w, w)
    return np.column_stack([XI.ravel(), ETA.ravel()]), W.ravel()


# ---------------------------------------------------------------------
# Shape functions (N) and derivatives (dN/dξ, dN/dη)
#
# 1D: returns (ng, n_nodes) and (ng, n_nodes)
# 2D: returns (ng, n_nodes) and (ng, n_nodes, 2) where last axis is [dξ, dη]
# ---------------------------------------------------------------------

def _line2(xi: ndarray) -> tuple[ndarray, ndarray]:
    xi = np.atleast_1d(xi)
    N = np.column_stack([(1 - xi) / 2, (1 + xi) / 2])
    dN = np.column_stack([
        -0.5 * np.ones_like(xi), 0.5 * np.ones_like(xi),
    ])
    return N, dN


def _line3(xi: ndarray) -> tuple[ndarray, ndarray]:
    # Gmsh MSH_LINE_3 ordering: [end(ξ=-1), end(ξ=+1), mid(ξ=0)]
    xi = np.atleast_1d(xi)
    N = np.column_stack([
        xi * (xi - 1) / 2,    # end at ξ = -1
        xi * (xi + 1) / 2,    # end at ξ = +1
        1 - xi ** 2,          # mid at ξ =  0
    ])
    dN = np.column_stack([
        (2 * xi - 1) / 2,
        (2 * xi + 1) / 2,
        -2 * xi,
    ])
    return N, dN


def _bernstein_line3(xi: ndarray) -> tuple[ndarray, ndarray]:
    """Quadratic Bernstein edge, same gmsh MSH_LINE_3 node order as
    :func:`_line3`: [end(ξ=-1), end(ξ=+1), mid(ξ=0)].

    With ``t = (1+ξ)/2``: ends carry ``(1-t)²`` / ``t²``, the mid control
    point carries ``2t(1-t)`` (non-interpolatory — peaks at 1/2). On a
    straight edge of length L every function integrates to L/3.
    """
    xi = np.atleast_1d(xi)
    N = np.column_stack([
        (1 - xi) ** 2 / 4,    # end at ξ = -1
        (1 + xi) ** 2 / 4,    # end at ξ = +1
        (1 - xi ** 2) / 2,    # mid control point
    ])
    dN = np.column_stack([
        -(1 - xi) / 2,
        (1 + xi) / 2,
        -xi,
    ])
    return N, dN


def _tri3(xi: ndarray, eta: ndarray) -> tuple[ndarray, ndarray]:
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    N = np.column_stack([1 - xi - eta, xi, eta])
    dN = np.zeros((len(xi), 3, 2))
    dN[:, 0, :] = [-1.0, -1.0]
    dN[:, 1, :] = [1.0, 0.0]
    dN[:, 2, :] = [0.0, 1.0]
    return N, dN


def _tri6(xi: ndarray, eta: ndarray) -> tuple[ndarray, ndarray]:
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    L1 = 1 - xi - eta
    L2 = xi
    L3 = eta
    N = np.column_stack([
        L1 * (2 * L1 - 1),
        L2 * (2 * L2 - 1),
        L3 * (2 * L3 - 1),
        4 * L1 * L2,
        4 * L2 * L3,
        4 * L1 * L3,
    ])
    dN = np.zeros((len(xi), 6, 2))
    # Corner 1: N = L1(2L1 - 1), dL1/dξ = -1, dL1/dη = -1
    dN[:, 0, 0] = 1 - 4 * L1
    dN[:, 0, 1] = 1 - 4 * L1
    # Corner 2: N = L2(2L2 - 1), dL2/dξ = 1, dL2/dη = 0
    dN[:, 1, 0] = 4 * L2 - 1
    dN[:, 1, 1] = 0.0
    # Corner 3: N = L3(2L3 - 1), dL3/dξ = 0, dL3/dη = 1
    dN[:, 2, 0] = 0.0
    dN[:, 2, 1] = 4 * L3 - 1
    # Mid 1-2: 4 L1 L2
    dN[:, 3, 0] = 4 * (L1 - L2)
    dN[:, 3, 1] = -4 * L2
    # Mid 2-3: 4 L2 L3
    dN[:, 4, 0] = 4 * L3
    dN[:, 4, 1] = 4 * L2
    # Mid 3-1: 4 L1 L3
    dN[:, 5, 0] = -4 * L3
    dN[:, 5, 1] = 4 * (L1 - L3)
    return N, dN


def _bernstein_tri6(xi: ndarray, eta: ndarray) -> tuple[ndarray, ndarray]:
    """Quadratic Bernstein triangle, same gmsh tri6 node order as
    :func:`_tri6`: corners 1, 2, 3 then midsides (1-2), (2-3), (3-1).

    ``B_corner_a = L_a²`` and ``B_mid(a,b) = 2·L_a·L_b`` — the face/edge
    trace of the fork's BezierTet10 / BezierTri6 volumetric basis
    (``BezierTet10.cpp`` shapeFunctions; ``apeGmsh._basis`` pins the same
    order). On a flat face of area A every function integrates to A/6, so
    a uniform traction maps to EQUAL control-point loads — unlike the
    Lagrange tri6, whose corners get ~0 and midsides A/3.
    """
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    L1 = 1 - xi - eta
    L2 = xi
    L3 = eta
    N = np.column_stack([
        L1 * L1,
        L2 * L2,
        L3 * L3,
        2 * L1 * L2,
        2 * L2 * L3,
        2 * L1 * L3,
    ])
    dN = np.zeros((len(xi), 6, 2))
    # dL1/dξ = dL1/dη = -1; dL2/dξ = 1; dL3/dη = 1.
    dN[:, 0, 0] = -2 * L1
    dN[:, 0, 1] = -2 * L1
    dN[:, 1, 0] = 2 * L2
    dN[:, 1, 1] = 0.0
    dN[:, 2, 0] = 0.0
    dN[:, 2, 1] = 2 * L3
    dN[:, 3, 0] = 2 * (L1 - L2)
    dN[:, 3, 1] = -2 * L2
    dN[:, 4, 0] = 2 * L3
    dN[:, 4, 1] = 2 * L2
    dN[:, 5, 0] = -2 * L3
    dN[:, 5, 1] = 2 * (L1 - L3)
    return N, dN


def _quad4(xi: ndarray, eta: ndarray) -> tuple[ndarray, ndarray]:
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    N = np.column_stack([
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4,
    ])
    dN = np.zeros((len(xi), 4, 2))
    dN[:, 0, 0] = -(1 - eta) / 4
    dN[:, 0, 1] = -(1 - xi) / 4
    dN[:, 1, 0] = (1 - eta) / 4
    dN[:, 1, 1] = -(1 + xi) / 4
    dN[:, 2, 0] = (1 + eta) / 4
    dN[:, 2, 1] = (1 + xi) / 4
    dN[:, 3, 0] = -(1 + eta) / 4
    dN[:, 3, 1] = (1 - xi) / 4
    return N, dN


def _quad8(xi: ndarray, eta: ndarray) -> tuple[ndarray, ndarray]:
    """8-node serendipity quad.

    Gmsh ordering: corners 0-3 CCW starting (-1,-1), then midsides
    5=(0,-1), 6=(1,0), 7=(0,1), 8=(-1,0) on edges 1-2, 2-3, 3-4, 4-1.
    """
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    N = np.column_stack([
        (1 - xi) * (1 - eta) * (-1 - xi - eta) / 4,
        (1 + xi) * (1 - eta) * (-1 + xi - eta) / 4,
        (1 + xi) * (1 + eta) * (-1 + xi + eta) / 4,
        (1 - xi) * (1 + eta) * (-1 - xi + eta) / 4,
        (1 - xi ** 2) * (1 - eta) / 2,
        (1 + xi) * (1 - eta ** 2) / 2,
        (1 - xi ** 2) * (1 + eta) / 2,
        (1 - xi) * (1 - eta ** 2) / 2,
    ])
    dN = np.zeros((len(xi), 8, 2))
    dN[:, 0, 0] = (1 - eta) * (2 * xi + eta) / 4
    dN[:, 0, 1] = (1 - xi) * (2 * eta + xi) / 4
    dN[:, 1, 0] = (1 - eta) * (2 * xi - eta) / 4
    dN[:, 1, 1] = (1 + xi) * (2 * eta - xi) / 4
    dN[:, 2, 0] = (1 + eta) * (2 * xi + eta) / 4
    dN[:, 2, 1] = (1 + xi) * (2 * eta + xi) / 4
    dN[:, 3, 0] = (1 + eta) * (2 * xi - eta) / 4
    dN[:, 3, 1] = (1 - xi) * (2 * eta - xi) / 4
    dN[:, 4, 0] = -xi * (1 - eta)
    dN[:, 4, 1] = -(1 - xi ** 2) / 2
    dN[:, 5, 0] = (1 - eta ** 2) / 2
    dN[:, 5, 1] = -(1 + xi) * eta
    dN[:, 6, 0] = -xi * (1 + eta)
    dN[:, 6, 1] = (1 - xi ** 2) / 2
    dN[:, 7, 0] = -(1 - eta ** 2) / 2
    dN[:, 7, 1] = -(1 - xi) * eta
    return N, dN


def _quad9(xi: ndarray, eta: ndarray) -> tuple[ndarray, ndarray]:
    """9-node Lagrangian biquadratic quad.

    Gmsh ordering: 4 corners CCW, 4 edge midsides (same order as quad8),
    1 center.
    """
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    # 1D quadratic Lagrange at nodes {-1, 0, +1}
    Lxi = [(xi ** 2 - xi) / 2, 1 - xi ** 2, (xi ** 2 + xi) / 2]
    Leta = [(eta ** 2 - eta) / 2, 1 - eta ** 2, (eta ** 2 + eta) / 2]
    dLxi = [(2 * xi - 1) / 2, -2 * xi, (2 * xi + 1) / 2]
    dLeta = [(2 * eta - 1) / 2, -2 * eta, (2 * eta + 1) / 2]

    # Node (i_ξ, i_η) indices into 1D basis, Gmsh node order
    idx = [
        (0, 0), (2, 0), (2, 2), (0, 2),
        (1, 0), (2, 1), (1, 2), (0, 1),
        (1, 1),
    ]
    ng = len(xi)
    N = np.zeros((ng, 9))
    dN = np.zeros((ng, 9, 2))
    for k, (ix, iy) in enumerate(idx):
        N[:, k] = Lxi[ix] * Leta[iy]
        dN[:, k, 0] = dLxi[ix] * Leta[iy]
        dN[:, k, 1] = Lxi[ix] * dLeta[iy]
    return N, dN


# ---------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------

def _check_basis(basis: str) -> None:
    if basis not in _SUPPORTED_BASES:
        raise ValueError(
            f"Unknown shape-function basis {basis!r}. "
            f"Supported bases: {_SUPPORTED_BASES}."
        )


def _edge_shape(n_nodes: int, basis: str) -> tuple[ndarray, ndarray, ndarray]:
    """Gauss weights + basis values/derivatives for one edge type."""
    _check_basis(basis)
    if n_nodes == 2:
        xi, w = _gauss_1d_2()
        N, dN = _line2(xi)          # degree 1: bernstein ≡ lagrange
    elif n_nodes == 3:
        xi, w = _gauss_1d_3()
        N, dN = (
            _bernstein_line3(xi) if basis == "bernstein" else _line3(xi)
        )
    else:
        raise NotImplementedError(
            f"Consistent line reduction for {n_nodes}-node edges is not "
            f"implemented.  Supported node counts: {_SUPPORTED_EDGE_NODES}."
        )
    return w, N, dN


def _face_shape(n_nodes: int, basis: str) -> tuple[ndarray, ndarray, ndarray]:
    """Gauss weights + basis values/derivatives for one face type."""
    _check_basis(basis)
    if basis == "bernstein" and n_nodes not in _SUPPORTED_BERNSTEIN_FACE_NODES:
        raise NotImplementedError(
            f"basis='bernstein' supports the simplex family only "
            f"(node counts {_SUPPORTED_BERNSTEIN_FACE_NODES}); got a "
            f"{n_nodes}-node face. The Ladruno Bézier elements "
            f"(BezierTri6 / BezierTet10) have no quad faces."
        )
    if n_nodes == 3:
        pts, w = _gauss_tri_6()
        N, dN = _tri3(pts[:, 0], pts[:, 1])   # degree 1: bases coincide
    elif n_nodes == 4:
        pts, w = _gauss_quad_3x3()
        N, dN = _quad4(pts[:, 0], pts[:, 1])
    elif n_nodes == 6:
        pts, w = _gauss_tri_6()
        N, dN = (
            _bernstein_tri6(pts[:, 0], pts[:, 1])
            if basis == "bernstein"
            else _tri6(pts[:, 0], pts[:, 1])
        )
    elif n_nodes == 8:
        pts, w = _gauss_quad_3x3()
        N, dN = _quad8(pts[:, 0], pts[:, 1])
    elif n_nodes == 9:
        pts, w = _gauss_quad_3x3()
        N, dN = _quad9(pts[:, 0], pts[:, 1])
    else:
        raise NotImplementedError(
            f"Consistent surface reduction for {n_nodes}-node faces is not "
            f"implemented.  Supported node counts: {_SUPPORTED_FACE_NODES}."
        )
    return w, N, dN


def integrate_edge(
    coords: ndarray, n_nodes: int, basis: str = "lagrange",
) -> ndarray:
    """Return per-node scalar weights ``∫ N_i · |J| dξ`` over an edge.

    Multiply element-wise by a constant force-per-length vector to
    obtain the consistent nodal force distribution.

    Parameters
    ----------
    coords : (n_nodes, 3) array of 3D physical coordinates.
    n_nodes : 2 (line2) or 3 (line3).
    basis : "lagrange" (interpolatory nodal values) or "bernstein"
        (Bézier control values — Ladruno BezierTri6/BezierTet10 edges).
    """
    w, N, dN = _edge_shape(n_nodes, basis)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    dx = dN @ coords                      # (ng, 3)
    Jmag = np.linalg.norm(dx, axis=1)     # (ng,)
    return (w * Jmag) @ N                 # (n_nodes,)


def integrate_edge_scaled(
    coords: ndarray,
    n_nodes: int,
    scalar_fn,
    basis: str = "lagrange",
) -> ndarray:
    """Per-node weights ``∫ N_i(ξ) · f(x(ξ)) · |J| dξ`` over an edge.

    Same Gauss rule and shape functions as :func:`integrate_edge`, but
    the force-per-length density ``f`` is a **scalar field** sampled at
    each Gauss point (in physical coordinates) rather than assumed
    constant.  Multiply the returned per-node weights by a unit (or
    scaling) direction vector to obtain the consistent nodal force
    distribution of a spatially varying line load — this is the exact
    integral to quadrature order, so it does not over/undershoot the
    way a single midpoint sample does.

    Parameters
    ----------
    coords : (n_nodes, 3) array of 3D physical coordinates.
    n_nodes : 2 (line2) or 3 (line3).
    scalar_fn : callable ``f(xyz) -> float`` evaluated at each physical
        Gauss point ``x(ξ_g) = N(ξ_g) · coords``.
    basis : "lagrange" or "bernstein" — see :func:`integrate_edge`.
    """
    w, N, dN = _edge_shape(n_nodes, basis)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    dx = dN @ coords                          # (ng, 3)
    Jmag = np.linalg.norm(dx, axis=1)         # (ng,)
    xg = N @ coords                           # (ng, 3) physical Gauss pts
    fg = np.array(
        [float(scalar_fn(xg[g])) for g in range(xg.shape[0])],
        dtype=float,
    )                                         # (ng,)
    return (w * Jmag * fg) @ N                 # (n_nodes,)


def integrate_face(coords: ndarray, n_nodes: int, basis: str = "lagrange",
                   ) -> tuple[ndarray, ndarray]:
    """Return per-node scalar weights and pressure-normal integrals.

    weights[i] = ``∫ N_i · |J| dA``
    normals[i] = ``∫ N_i · (∂x/∂ξ × ∂x/∂η) dA``   (3-vector)

    For a pressure magnitude ``p`` acting on the face:
        - Normal pressure (into face): ``F_i = -p · normals[i]``
        - Directional traction d̂:       ``F_i = (p · weights[i]) · d̂``

    Parameters
    ----------
    coords : (n_nodes, 3) physical node coordinates.
    n_nodes : 3, 4, 6, 8 or 9.
    basis : "lagrange" (interpolatory nodal values) or "bernstein"
        (Bézier control values — Ladruno BezierTet10 tri6 faces; the
        simplex family only, quad faces raise ``NotImplementedError``).
    """
    w, N, dN = _face_shape(n_nodes, basis)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    dx_dxi = dN[:, :, 0] @ coords         # (ng, 3)
    dx_deta = dN[:, :, 1] @ coords        # (ng, 3)
    cross = np.cross(dx_dxi, dx_deta)     # (ng, 3)  =  |J| · n̂
    Jmag = np.linalg.norm(cross, axis=1)  # (ng,)
    weights = (w * Jmag) @ N              # (n_nodes,)
    # normals[i] = Σ_g w[g] · N[g,i] · cross[g,:]
    normals = np.einsum("g,gi,gd->id", w, N, cross)
    return weights, normals


__all__ = [
    "integrate_edge",
    "integrate_edge_scaled",
    "integrate_face",
    "_SUPPORTED_BASES",
    "_SUPPORTED_BERNSTEIN_FACE_NODES",
    "_SUPPORTED_EDGE_NODES",
    "_SUPPORTED_FACE_NODES",
]
