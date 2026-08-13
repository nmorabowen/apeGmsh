"""Corner-node signed volume / area for linear cells (MESH.INVERTED).

Quadratic / Bézier cells are not judged here — their extra nodes are
control values (ADR 0091), not corner coords. prism6 / pyramid5 are
also not judged in v1: there is no single well-rehearsed corner-volume
formula in this tree that we would trust without an extra test suite.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy import ndarray

# Same hex8 → 6-tet split as MassResolver, but *signed* (MassResolver
# takes abs and cannot see inversion). The kernel list has tet
# (0, 7, 2, 6) in negative orientation — fine under abs(), a false
# MESH.INVERTED here. Flip to (0, 2, 7, 6) so all six tets are
# positive on a right-handed hex (unit cube sums to 1.0).
_HEX8_TETS = (
    (0, 1, 2, 5), (0, 2, 3, 7), (0, 5, 2, 6),
    (0, 5, 6, 7), (0, 2, 7, 6), (0, 4, 5, 7),
)

# Max distance to the best-fit plane, as a fraction of the bbox
# diagonal, for a mesh to count as planar (A2).
_PLANAR_REL_TOL = 1e-8


def _tet_signed(pts: ndarray, a: int, b: int, c: int, d: int) -> float:
    ab = pts[b] - pts[a]
    ac = pts[c] - pts[a]
    ad = pts[d] - pts[a]
    return float(np.dot(ab, np.cross(ac, ad))) / 6.0


def _tri_signed_area(pts: ndarray, a: int, b: int, c: int) -> float:
    """Projected signed area onto the coordinate plane of largest |n|."""
    n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
    k = int(np.argmax(np.abs(n)))
    return 0.5 * float(n[k])


def _signed_tri3(pts: ndarray) -> float:
    return _tri_signed_area(pts, 0, 1, 2)


def _signed_quad4(pts: ndarray) -> float:
    # Two-triangle split. Opposite signs (bowtie) or a non-positive
    # triangle count as inverted — the caller tests the returned value
    # and this helper reports the worse of the two signed areas when
    # they disagree, else the sum.
    a1 = _tri_signed_area(pts, 0, 1, 2)
    a2 = _tri_signed_area(pts, 0, 2, 3)
    if a1 == 0.0 or a2 == 0.0:
        return 0.0
    if (a1 > 0.0) != (a2 > 0.0):
        return 0.0
    return a1 + a2


def _signed_tet4(pts: ndarray) -> float:
    return _tet_signed(pts, 0, 1, 2, 3)


def _signed_hex8(pts: ndarray) -> float:
    return sum(_tet_signed(pts, *tet) for tet in _HEX8_TETS)


# Linear types we will judge. Keyed by ElementTypeInfo.name.
_LINEAR_MEASURE: dict[str, Callable[[ndarray], float]] = {
    "tri3": _signed_tri3,
    "quad4": _signed_quad4,
    "tet4": _signed_tet4,
    "hex8": _signed_hex8,
}


def measure_linear_cell(type_name: str, corner_xyz: ndarray) -> float | None:
    """Signed corner volume/area, or ``None`` if this type is not judged."""
    fn = _LINEAR_MEASURE.get(type_name)
    if fn is None:
        return None
    return fn(np.asarray(corner_xyz, dtype=np.float64))


def coords_are_planar(
    coords: ndarray, *, rel_tol: float = _PLANAR_REL_TOL,
) -> bool:
    """True when all points lie on one plane (2D cells may be judged).

    A 3D shell's orientation is a viewing convention, not inversion.
    Fewer than 4 points are trivially planar.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return True
    from apeGmsh.results._geometry import model_diagonal

    diag = model_diagonal(pts)
    centered = pts - pts.mean(axis=0)
    # Smallest right-singular vector is the best-fit normal.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dist = np.abs(centered @ vt[-1])
    return float(dist.max()) <= float(rel_tol) * diag
