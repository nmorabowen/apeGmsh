"""
Geometric primitives for constraint resolution.

Provides:

* Shape function evaluators for tri3, quad4, tri6, quad8 faces
  (used for tie / tied_contact / mortar interpolation).
* :class:`_SpatialIndex` — a thin KD-tree wrapper with a SciPy-free
  fallback so that the resolver works without scipy.
* :func:`_project_point_to_face` — Newton iteration to project a
  point onto an element face and obtain parametric coordinates.
* :func:`_is_inside_parametric` — reference-domain containment check.

These utilities are intentionally solver-agnostic and depend only on
NumPy (SciPy is optional).
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


# ── Shape functions ─────────────────────────────────────────────────

def _shape_tri3(xi: float, eta: float) -> ndarray:
    """Shape functions for 3-node triangle in area coords."""
    return np.array([1.0 - xi - eta, xi, eta])


def _shape_quad4(xi: float, eta: float) -> ndarray:
    """Shape functions for 4-node quad at (ξ, η) ∈ [-1,1]²."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])


def _shape_tri6(xi: float, eta: float) -> ndarray:
    """Shape functions for 6-node triangle."""
    zeta = 1.0 - xi - eta
    return np.array([
        zeta * (2 * zeta - 1),
        xi * (2 * xi - 1),
        eta * (2 * eta - 1),
        4 * zeta * xi,
        4 * xi * eta,
        4 * eta * zeta,
    ])


def _shape_quad8(xi: float, eta: float) -> ndarray:
    """Shape functions for 8-node serendipity quad."""
    N = np.zeros(8)
    # Corner nodes
    for i, (xi_i, eta_i) in enumerate([
        (-1, -1), (1, -1), (1, 1), (-1, 1)
    ]):
        N[i] = 0.25 * (1 + xi_i * xi) * (1 + eta_i * eta) * \
               (xi_i * xi + eta_i * eta - 1)
    # Mid-side nodes
    N[4] = 0.5 * (1 - xi**2) * (1 - eta)
    N[5] = 0.5 * (1 + xi) * (1 - eta**2)
    N[6] = 0.5 * (1 - xi**2) * (1 + eta)
    N[7] = 0.5 * (1 - xi) * (1 - eta**2)
    return N


# Map: number of face nodes -> shape function evaluator
SHAPE_FUNCTIONS = {
    3: _shape_tri3,
    4: _shape_quad4,
    6: _shape_tri6,
    8: _shape_quad8,
}


# ── Spatial index ───────────────────────────────────────────────────

class _SpatialIndex:
    """Small nearest-neighbour wrapper with a SciPy-free fallback."""

    def __init__(self, coords: ndarray) -> None:
        self._coords = np.asarray(coords, dtype=float)

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self._tree = None
        else:
            self._tree = cKDTree(self._coords)

    def query_ball_point(self, point: ndarray, radius: float) -> list[int]:
        point = np.asarray(point, dtype=float)
        if self._tree is not None:
            return list(self._tree.query_ball_point(point, radius))

        dists = np.linalg.norm(self._coords - point, axis=1)
        return np.flatnonzero(dists <= radius).astype(int).tolist()

    def query(self, point: ndarray, k: int = 1):
        point = np.asarray(point, dtype=float)
        if self._tree is not None:
            return self._tree.query(point, k=k)

        dists = np.linalg.norm(self._coords - point, axis=1)
        order = np.argsort(dists)
        if k == 1:
            idx = int(order[0])
            return float(dists[idx]), idx

        order = order[:k]
        return dists[order], order


# ── Projection / containment ────────────────────────────────────────

def _project_point_to_face(
    point: ndarray,
    face_coords: ndarray,
) -> tuple[ndarray, ndarray, float]:
    """
    Project a point onto an element face.

    Uses Newton iteration to find the parametric coordinates (ξ, η)
    that minimise the distance from the point to the face surface.

    Parameters
    ----------
    point : ndarray, shape (3,)
        The point to project.
    face_coords : ndarray, shape (n_nodes, 3)
        Physical coordinates of the face nodes.

    Returns
    -------
    xi_eta : ndarray, shape (2,)
        Parametric coordinates of the projection.
    projected : ndarray, shape (3,)
        Physical coordinates of the projected point on the face.
    distance : float
        Distance from the original point to the projection.
    """
    n_nodes = face_coords.shape[0]
    shape_fn = SHAPE_FUNCTIONS.get(n_nodes)
    if shape_fn is None:
        raise ValueError(
            f"No shape function for {n_nodes}-node face."
        )

    # Initial guess: face centroid in parametric space
    if n_nodes in (3, 6):
        xi, eta = 1.0 / 3.0, 1.0 / 3.0
    else:
        xi, eta = 0.0, 0.0

    # Newton iteration (typically converges in 3-5 iterations)
    for _ in range(20):
        N = shape_fn(xi, eta)
        x_param = N @ face_coords           # (3,)
        residual = x_param - point           # (3,)

        # Numerical derivatives of shape functions
        eps = 1e-8
        N_xi  = (shape_fn(xi + eps, eta) - shape_fn(xi - eps, eta)) / (2 * eps)
        N_eta = (shape_fn(xi, eta + eps) - shape_fn(xi, eta - eps)) / (2 * eps)

        dx_dxi  = N_xi  @ face_coords       # (3,)
        dx_deta = N_eta @ face_coords        # (3,)

        # Gauss–Newton step solved as a least-squares problem on J
        # directly (SVD-based).  Forming JᵀJ squares the condition
        # number (κ(JᵀJ)=κ(J)²), degrading the projection on sliver /
        # high-aspect faces well before any det≈0 guard would trip;
        # lstsq on the (3×2) J is robust there.
        J = np.column_stack([dx_dxi, dx_deta])   # (3, 2)
        try:
            delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        except np.linalg.LinAlgError:
            break
        xi  += delta[0]
        eta += delta[1]

        if np.linalg.norm(delta) < 1e-12:
            break

    N_final = shape_fn(xi, eta)
    projected = np.asarray(N_final @ face_coords)
    distance = float(np.linalg.norm(projected - point))

    return np.array([xi, eta]), projected, distance


def _is_inside_parametric(
    xi_eta: ndarray,
    n_nodes: int,
    tol: float = 0.05,
) -> bool:
    """
    Check if parametric coordinates are inside the element face
    (with a small tolerance for numerical rounding).
    """
    xi, eta = xi_eta
    if n_nodes in (3, 6):
        # Triangle: ξ ≥ 0, η ≥ 0, ξ + η ≤ 1
        return (xi >= -tol and eta >= -tol and
                xi + eta <= 1.0 + tol)
    else:
        # Quad: ξ ∈ [-1,1], η ∈ [-1,1]
        return (abs(xi) <= 1.0 + tol and abs(eta) <= 1.0 + tol)


__all__ = [
    "SHAPE_FUNCTIONS",
    "_SpatialIndex",
    "_project_point_to_face",
    "_is_inside_parametric",
    "_shape_tri3",
    "_shape_quad4",
    "_shape_tri6",
    "_shape_quad8",
]
