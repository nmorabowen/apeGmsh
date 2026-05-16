"""Minimal Gauss quadrature rules for reference-element integration.

Sufficient for HRZ mass lumping (integrate ``N_I^2`` on the reference
element) and other element-aware operations on the input side.  Rules
return ``(points, weights)`` where ``points`` has shape
``(n_qp, parent_dim)`` and ``weights`` has shape ``(n_qp,)``.

Conventions match :mod:`apeGmsh.fem._shape_functions` exactly:

* Line ``ξ ∈ [-1, +1]`` — weights sum to 2.
* Quad / Hex tensor product of ``[-1, +1]^d`` — weights sum to
  ``2^d``.
* Triangle barycentric on the reference triangle
  ``(0,0)/(1,0)/(0,1)`` — weights sum to the area ``1/2``.
* Tet barycentric on the reference tet
  ``(0,0,0)/(1,0,0)/(0,1,0)/(0,0,1)`` — weights sum to the volume
  ``1/6``.
* Wedge ``tri × line`` tensor product — weights sum to ``1/2 · 2 = 1``.

These conventions matter: the HRZ weights are *normalized* (divided by
their own sum), so the absolute scale of the quadrature weights cancels
— but the *points* must live in the same parent domain the shape
functions expect, which is what these conventions guarantee.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------
# 1D Gauss-Legendre on [-1, +1]
# ------------------------------------------------------------------------

def gauss_legendre_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n``-point Gauss-Legendre on ``[-1, +1]``.

    Exact for polynomials of degree ``2n - 1``.  Returns 1-D
    ``points`` of shape ``(n,)`` — the line shape functions accept
    that directly.
    """
    pts, wts = np.polynomial.legendre.leggauss(int(n))
    return pts.astype(np.float64), wts.astype(np.float64)


# ------------------------------------------------------------------------
# Tensor-product rules for [-1, +1]^d  (Quad*, Hex*)
# ------------------------------------------------------------------------

def gauss_quad_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n×n`` tensor-product Gauss rule for the ``[-1, +1]^2`` quad."""
    p, w = gauss_legendre_1d(n)
    pts = np.array([(a, b) for a in p for b in p], dtype=np.float64)
    wts = np.array([wa * wb for wa in w for wb in w], dtype=np.float64)
    return pts, wts


def gauss_hex_3d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n×n×n`` tensor-product Gauss rule for the ``[-1, +1]^3`` hex."""
    p, w = gauss_legendre_1d(n)
    pts = np.array(
        [(a, b, c) for a in p for b in p for c in p], dtype=np.float64,
    )
    wts = np.array(
        [wa * wb * wc for wa in w for wb in w for wc in w],
        dtype=np.float64,
    )
    return pts, wts


# ------------------------------------------------------------------------
# Triangle — 7-point rule, degree 5 exact
# (Strang–Fix / Hammer; weights sum to the reference area 1/2)
# ------------------------------------------------------------------------

_A1 = 0.470142064105115
_B1 = 0.059715871789770
_A2 = 0.101286507323456
_B2 = 0.797426985353087
_W0 = 0.225 * 0.5
_W1 = 0.132394152788506 * 0.5
_W2 = 0.125939180544827 * 0.5

_TRI_PTS = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [_A1, _A1],
    [_A1, _B1],
    [_B1, _A1],
    [_A2, _A2],
    [_A2, _B2],
    [_B2, _A2],
], dtype=np.float64)
_TRI_WTS = np.array([_W0, _W1, _W1, _W1, _W2, _W2, _W2], dtype=np.float64)


def gauss_tri() -> tuple[np.ndarray, np.ndarray]:
    """7-point rule on the reference triangle, degree 5 exact."""
    return _TRI_PTS.copy(), _TRI_WTS.copy()


# ------------------------------------------------------------------------
# Tetrahedron — 11-point rule, degree 4 exact
# (Keast; weights sum to the reference volume 1/6)
# ------------------------------------------------------------------------

def _tet_rule() -> tuple[np.ndarray, np.ndarray]:
    """Keast 11-point degree-4 rule, weights summing to 1/6.

    Built explicitly here (rather than via a table of magic numbers)
    so the construction is auditable: one centroid point + two
    symmetric orbits.
    """
    pts: list[tuple[float, float, float]] = []
    wts: list[float] = []

    # Orbit 1 — centroid
    pts.append((0.25, 0.25, 0.25))
    wts.append(-0.013155555555556)

    # Orbit 2 — points of the form (a,b,b,b) permutations, a + 3b = 1
    a, b = 0.785714285714286, 0.071428571428571
    for p in ((a, b, b), (b, a, b), (b, b, a), (b, b, b)):
        pts.append(p)
        wts.append(0.007622222222222)

    # Orbit 3 — points (c,c,d,d) type, 6 permutations
    c, d = 0.399403576166799, 0.100596423833201
    for p in (
        (c, c, d), (c, d, c), (d, c, c),
        (c, d, d), (d, c, d), (d, d, c),
    ):
        pts.append(p)
        wts.append(0.024888888888889)

    P = np.array(pts, dtype=np.float64)
    W = np.array(wts, dtype=np.float64)
    # Scale so the weights sum to the reference-tet volume 1/6.
    W = W / W.sum() / 6.0
    return P, W


_TET_PTS, _TET_WTS = _tet_rule()


def gauss_tet() -> tuple[np.ndarray, np.ndarray]:
    """11-point rule on the reference tet, degree 4 exact."""
    return _TET_PTS.copy(), _TET_WTS.copy()


# ------------------------------------------------------------------------
# Wedge — tri × line tensor product
# ------------------------------------------------------------------------

def gauss_wedge(n_line: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Tensor product of the triangle rule and a 1-D line rule."""
    pt, wt = gauss_tri()
    pl, wl = gauss_legendre_1d(n_line)
    pts = np.array(
        [(t[0], t[1], z) for t in pt for z in pl], dtype=np.float64,
    )
    wts = np.array(
        [a * b for a in wt for b in wl], dtype=np.float64,
    )
    return pts, wts
