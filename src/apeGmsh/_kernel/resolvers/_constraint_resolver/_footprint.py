"""Overlap areas of mesh faces with a convex planar footprint polygon.

Used by ``distributing_coupling(weighting="footprint")``: the RBE3 weights
are the load of a uniform pressure over the footprint lumped onto the
mesh, each face's overlap area split equally among its nodes. Pure numpy,
no Gmsh.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

#: relative tolerance for the footprint's degeneracy and convexity checks.
_TOL = 1.0e-9
#: relative tolerance for its planarity.
_PLANE_TOL = 1.0e-6
#: an overlap below this fraction of the footprint's area is round-off.
_OVERLAP_TOL = 1.0e-9


def polygon_frame(poly: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Origin, in-plane axes ``e1``, ``e2`` and the 2-D vertices of a
    planar convex polygon, counter-clockwise in ``(e1, e2)``.

    Raises ``ValueError`` on a degenerate, non-planar or non-convex one.
    """
    p = np.asarray(poly, dtype=float)
    o = p.mean(axis=0)
    n = np.zeros(3)
    for i in range(len(p)):  # Newell normal
        n += np.cross(p[i] - o, p[(i + 1) % len(p)] - o)
    size = float(np.ptp(p, axis=0).max())
    if size <= 0.0 or float(np.linalg.norm(n)) <= _TOL * size * size:
        raise ValueError("distributing_coupling: the footprint polygon is degenerate.")
    n /= np.linalg.norm(n)
    if float(np.abs((p - o) @ n).max()) > _PLANE_TOL * size:
        raise ValueError("distributing_coupling: the footprint polygon is not planar.")
    e1 = p[1] - p[0]
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = np.column_stack([(p - o) @ e1, (p - o) @ e2])  # CCW: n is the Newell normal
    m = len(q)
    for i in range(m):
        a, b, c = q[i], q[(i + 1) % m], q[(i + 2) % m]
        turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if turn < -_TOL * size * size:
            raise ValueError(
                "distributing_coupling: the footprint polygon must be convex "
                "and its vertices given in order."
            )
    return o, e1, e2, q


def _clip(subject: ndarray, clip: ndarray) -> ndarray:
    """Sutherland-Hodgman: ``subject`` (a simple polygon) clipped by the
    convex counter-clockwise ``clip``; 2-D vertex arrays."""
    out = np.asarray(subject, dtype=float)
    m = len(clip)
    for i in range(m):
        if len(out) == 0:
            break
        a, b = clip[i], clip[(i + 1) % m]
        d = b - a
        side = d[0] * (out[:, 1] - a[1]) - d[1] * (out[:, 0] - a[0])
        nxt: list[ndarray] = []
        for j in range(len(out)):
            cur, prev = out[j], out[j - 1]
            sc, sp = side[j], side[j - 1]
            if sc >= 0.0:
                if sp < 0.0:
                    nxt.append(prev + (cur - prev) * (sp / (sp - sc)))
                nxt.append(cur)
            elif sp >= 0.0:
                nxt.append(prev + (cur - prev) * (sp / (sp - sc)))
        out = np.array(nxt, dtype=float).reshape(-1, 2)
    return out


def _area(poly: ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def face_overlap_shares(face_coords: ndarray, footprint: ndarray) -> list[tuple[int, float]]:
    """``(f, overlap / nodes_per_face)`` for every face ``f`` the footprint
    overlaps.

    ``face_coords`` is ``(n_faces, nodes_per_face, 3)``. Faces are
    projected onto the footprint's plane; their orientation does not
    matter.
    """
    o, e1, e2, q = polygon_frame(footprint)
    fc = np.asarray(face_coords, dtype=float)
    u = (fc - o) @ e1
    v = (fc - o) @ e2
    lo, hi = q.min(axis=0), q.max(axis=0)
    hit = np.nonzero(
        (u.max(axis=1) > lo[0]) & (u.min(axis=1) < hi[0])
        & (v.max(axis=1) > lo[1]) & (v.min(axis=1) < hi[1])
    )[0]
    npf = fc.shape[1]
    # a face that only touches the footprint along an edge or at a corner
    # clips to a sliver of round-off area: below this it is no overlap
    floor = _OVERLAP_TOL * _area(q)
    out: list[tuple[int, float]] = []
    for f in hit:
        poly = np.column_stack([u[f], v[f]])
        if _area(poly) < 0.0:
            poly = poly[::-1]
        a = _area(_clip(poly, q))
        if a > floor:
            out.append((int(f), a / npf))
    return out
