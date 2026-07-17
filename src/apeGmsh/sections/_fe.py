"""Per-block Gauss-point FE data for the section solvers (ADR 0078).

Precomputes, for one element block, everything the warping (S2) and
stress (S4) analyses integrate with: shape values ``N``, physical
gradients ``B``, weighted measures ``w·detJ``, and IP coordinates.
Shared by design — the geometric analysis (S1) needs no gradients and
keeps its own lighter loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from apeGmsh.fem._quadrature import gauss_quad_2d, gauss_tri
from apeGmsh.fem._shape_functions import get_shape_functions

from ._snapshot import _Block

_TRI_CODES = frozenset({2, 9})
_QUAD_GAUSS_N = 3


@dataclass(frozen=True, slots=True)
class BlockQuadrature:
    """Gauss-point data for one element block (centroidal coordinates)."""

    block: _Block
    N: ndarray        # (n_ip, npe)          shape values
    B: ndarray        # (E, n_ip, 2, npe)    physical gradients [∂N/∂x; ∂N/∂y]
    wdetj: ndarray    # (E, n_ip)            Gauss weight × |J|
    x: ndarray        # (E, n_ip)            IP x (centroidal)
    y: ndarray        # (E, n_ip)            IP y (centroidal)


def block_quadrature(
    block: _Block, coords: ndarray, *, centroid: tuple[float, float]
) -> BlockQuadrature:
    """Build :class:`BlockQuadrature` for *block* over ``coords (N, 2)``.

    Coordinates are shifted to the given centroid — every warping-side
    integral in the Pilkey formulation lives in centroidal axes.
    """
    if block.code in _TRI_CODES:
        pts, wts = gauss_tri()
    else:
        pts, wts = gauss_quad_2d(_QUAD_GAUSS_N)
    shape = get_shape_functions(block.code)
    assert shape is not None  # gated in the snapshot
    N_fn, dN_fn, _, _ = shape

    N = N_fn(pts)                                   # (n_ip, npe)
    dN = dN_fn(pts)                                 # (n_ip, npe, 2)
    xy = coords[block.conn].astype(np.float64)      # (E, npe, 2)
    xy = xy - np.asarray(centroid, dtype=np.float64)[None, None, :]

    # Jacobian M[k, j] = ∂x_j/∂ξ_k, per element per IP: (E, n_ip, 2, 2)
    M = np.einsum("iak,eaj->eikj", dN, xy)
    detj = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    if np.any(detj <= 0.0):
        # gmsh 2-D elements are CCW in-plane; a negative det means a
        # flipped element — integrate |J| but keep gradients consistent
        # by inverting the true (signed) matrix. abs() only on the
        # measure below.
        pass
    invM = np.linalg.inv(M)                         # (E, n_ip, 2, 2)
    # chain rule: g_ξ = M · g_x  →  ∂N_a/∂x_j = Σ_k (M⁻¹)[j, k] · ∂N_a/∂ξ_k
    B = np.einsum("iak,eijk->eija", dN, invM)

    ip_xy = np.einsum("ia,eaj->eij", N, xy)         # (E, n_ip, 2)
    return BlockQuadrature(
        block=block,
        N=N,
        B=B,
        wdetj=np.abs(detj) * wts[None, :],
        x=ip_xy[..., 0],
        y=ip_xy[..., 1],
    )


# --------------------------------------------------------------------- #
# Nodal-point evaluation (S4 stress recovery)                            #
# --------------------------------------------------------------------- #

# Reference (natural) coordinates of each node, in the SAME gmsh
# ordering the kernel's shape functions use (see _shape_functions.py
# ordering comments — tri: corners then mid-edges 01/12/20; quad:
# corners CCW from (-1,-1), then mid-edges bottom/right/top/left,
# then centre).
_REF_NODES: dict[int, np.ndarray] = {
    2: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    9: np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
    ]),
    3: np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
    16: np.array([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
    ]),
    10: np.array([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        [0.0, 0.0],
    ]),
}


def block_nodal(
    block: _Block, coords: ndarray, *, centroid: tuple[float, float]
) -> BlockQuadrature:
    """Like :func:`block_quadrature` but evaluated AT the element nodes
    (exact nodal recovery — no Gauss→node extrapolation).  ``wdetj``
    carries |J| only (no weights) and must not be used for integration.
    """
    pts = _REF_NODES[block.code]
    shape = get_shape_functions(block.code)
    assert shape is not None
    N_fn, dN_fn, _, _ = shape

    N = N_fn(pts)                                   # ≈ identity
    dN = dN_fn(pts)
    xy = coords[block.conn].astype(np.float64)
    xy = xy - np.asarray(centroid, dtype=np.float64)[None, None, :]
    M = np.einsum("iak,eaj->eikj", dN, xy)
    detj = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    invM = np.linalg.inv(M)
    B = np.einsum("iak,eijk->eija", dN, invM)
    ip_xy = np.einsum("ia,eaj->eij", N, xy)
    return BlockQuadrature(
        block=block,
        N=N,
        B=B,
        wdetj=np.abs(detj),
        x=ip_xy[..., 0],
        y=ip_xy[..., 1],
    )
