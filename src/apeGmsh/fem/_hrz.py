"""HRZ diagonal-scaling lumping weights for FE mass matrices.

Equal-share lumping (``ρV/n`` per node) is correct only when every
node of an element has the same shape-function "footprint".  That
holds for first-order elements (line2, tri3, quad4, tet4, hex8,
wedge6) but **not** for higher-order ones — for a quadratic line the
mid-node carries four times the corner mass, and equal-share gets it
wrong.

HRZ (Hinton–Rock–Zienkiewicz, 1976) lumping fixes this by
distributing the element mass in proportion to the *diagonal* of the
consistent mass matrix, then rescaling so total mass is preserved:

    ŵ_I = ∫_{Ω_ref} N_I²(ξ) dξ          (diagonal of consistent M)
    w_I = ŵ_I / Σ_J ŵ_J                 (normalized — Σ w_I = 1)
    m_I = ρ V_e · w_I                   (per-node lumped mass)

Why the *diagonal* and not the row-sum ``∫ ρ N_I dΩ``: for serendipity
elements (quad8, hex20) the corner shape functions go negative over
parts of the element, so the row-sum can produce **negative** corner
masses.  ``∫ N_I²`` is non-negative by construction, so HRZ stays
physical for every element type in the catalog.

Affine-mapping simplification
-----------------------------
The weights are computed on the **reference element** (Jacobian
``≡ 1``).  For a physical element with a constant Jacobian
(parallelepiped hex, undistorted prism, affine simplex) this is
*exact*: the Jacobian factors out of both numerator and denominator
and cancels.  For curved or strongly distorted higher-order physical
elements it is a (standard, widely used) approximation — the same
simplification ANSYS / Abaqus / LS-DYNA make for their default lumped
mass.  True per-element Jacobian-weighted HRZ is a future refinement.

First-order collapse
--------------------
For line2 / tri3 / quad4 / tet4 / hex8 / wedge6 the ``∫ N_I²`` are all
equal, so ``w_I = 1/n`` and HRZ is *bit-identical* to equal-share.
Existing first-order callers see no change.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import _quadrature as Q
from . import _shape_functions as SF


# Gmsh code → (quadrature kind, n-arg).  The quadrature must integrate
# N_I² exactly: degree 2p where p is the element's polynomial order.
_QUAD_SPEC: dict[int, tuple[str, int | None]] = {
    SF._LINE2:  ("line", 2),    # N² degree 2  → 2-pt GL (exact deg 3)
    SF._TRI3:   ("tri", None),  # 7-pt tri (exact deg 5)
    SF._QUAD4:  ("quad", 2),    # N² degree 2 per axis
    SF._TET4:   ("tet", None),  # 11-pt tet (exact deg 4)
    SF._HEX8:   ("hex", 2),
    SF._WEDGE6: ("wedge", 3),
    SF._TRI6:   ("tri", None),  # N² degree 4 → 7-pt tri exact
    SF._QUAD9:  ("quad", 3),    # N² degree 4 per axis → 3-pt GL
    SF._TET10:  ("tet", None),  # N² degree 4 → 11-pt tet exact
    SF._HEX27:  ("hex", 3),
    SF._QUAD8:  ("quad", 3),    # serendipity — 3-pt GL covers it
    SF._HEX20:  ("hex", 3),
}


def _quadrature_for(kind: str, n: int | None) -> tuple[np.ndarray, np.ndarray]:
    if kind == "line":
        pts, wts = Q.gauss_legendre_1d(n)
        return pts.reshape(-1, 1), wts
    if kind == "quad":
        return Q.gauss_quad_2d(n)
    if kind == "hex":
        return Q.gauss_hex_3d(n)
    if kind == "tri":
        return Q.gauss_tri()
    if kind == "tet":
        return Q.gauss_tet()
    if kind == "wedge":
        return Q.gauss_wedge(n or 3)
    raise ValueError(f"Unknown quadrature kind {kind!r}")


@lru_cache(maxsize=None)
def hrz_weights(gmsh_code: int) -> tuple[float, ...]:
    """Normalized per-node HRZ weights for a Gmsh element type.

    Returns a tuple of length ``n_nodes`` summing to ``1.0``.  For
    first-order elements every entry is ``1/n`` (equal-share).

    Raises
    ------
    ValueError
        If ``gmsh_code`` is not in the shape-function catalog or has
        no quadrature spec.
    """
    spec = _QUAD_SPEC.get(int(gmsh_code))
    if spec is None:
        raise ValueError(
            f"HRZ weights undefined for Gmsh element code {gmsh_code}."
        )
    catalog = SF.get_shape_functions(int(gmsh_code))
    if catalog is None:
        raise ValueError(
            f"No shape-function catalog entry for Gmsh code {gmsh_code}."
        )
    N_fn = catalog[0]
    pts, wts = _quadrature_for(*spec)
    N = N_fn(pts)                               # (n_qp, n_nodes)
    diag = (N**2 * wts[:, None]).sum(axis=0)    # ∫ N_I² over Ω_ref
    total = float(diag.sum())
    if total <= 0.0:
        raise ValueError(
            f"HRZ trace non-positive for Gmsh code {gmsh_code}: {total}."
        )
    return tuple((diag / total).astype(float).tolist())


# n_nodes → Gmsh code, by parent dimension.  The mass resolver only
# has connectivity (no element-type tag), so it infers type from the
# node count within a known dimension.
_VOLUME_CODE_BY_N: dict[int, int] = {
    4:  SF._TET4,
    8:  SF._HEX8,
    6:  SF._WEDGE6,
    10: SF._TET10,
    20: SF._HEX20,
    27: SF._HEX27,
}
_SURFACE_CODE_BY_N: dict[int, int] = {
    3: SF._TRI3,
    4: SF._QUAD4,
    6: SF._TRI6,
    8: SF._QUAD8,
    9: SF._QUAD9,
}
_LINE_CODE_BY_N: dict[int, int] = {
    2: SF._LINE2,
}


def volume_code(n_nodes: int) -> int | None:
    return _VOLUME_CODE_BY_N.get(int(n_nodes))


def surface_code(n_nodes: int) -> int | None:
    return _SURFACE_CODE_BY_N.get(int(n_nodes))


def line_code(n_nodes: int) -> int | None:
    return _LINE_CODE_BY_N.get(int(n_nodes))
