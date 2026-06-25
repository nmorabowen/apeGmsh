"""
Pure geometry helpers for rebar hook / bend construction (ADR 0066 §3, §5).

No gmsh — just vector math, so the 3-D bend-plane resolution and the
fillet-arc construction are unit-testable off-session. The composite
(:mod:`apeGmsh.core.RebarComposite`) turns the returned primitives into
gmsh points + lines + arcs, reusing point tags between consecutive
primitives so a hook stays topologically welded WITHOUT make_conformal.
"""

from __future__ import annotations

import numpy as np

_TOL = 1e-12
_SEED_AXES = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > _TOL else None


def _project_perp(v, t_hat) -> np.ndarray | None:
    """Unit component of *v* perpendicular to unit vector *t_hat*."""
    v = np.asarray(v, dtype=float)
    return _unit(v - np.dot(v, t_hat) * t_hat)


def _rodrigues(v, axis_hat, angle) -> np.ndarray:
    """Rotate *v* by *angle* (rad) about unit axis *axis_hat* (Rodrigues)."""
    v = np.asarray(v, dtype=float)
    k = np.asarray(axis_hat, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def bend_plane(tangent, turn_dir) -> tuple[np.ndarray, np.ndarray, bool]:
    """Resolve the hook bend plane.

    Returns ``(u_hat, n_hat, fell_back)`` where ``u_hat`` is the in-plane
    turn direction (⊥ tangent, toward ``turn_dir``) and ``n_hat = t × u``
    is the bend normal. If ``turn_dir`` is (near-)parallel to the tangent
    a deterministic seed-axis ladder (+Z, +Y, +X) picks a plane and
    ``fell_back`` is True (caller should warn).
    """
    t = _unit(tangent)
    if t is None:
        raise ValueError("bend_plane: tangent is zero-length.")
    u = _project_perp(turn_dir, t)
    if u is not None:
        return u, _unit(np.cross(t, u)), False
    for seed in _SEED_AXES:
        u = _project_perp(seed, t)
        if u is not None:
            return u, _unit(np.cross(t, u)), True
    raise ValueError("bend_plane: degenerate tangent — no bend plane.")


def _arc_segment(p0, t0_hat, angle, n_hat, radius):
    """A circular arc from ``p0`` with unit tangent ``t0_hat``, turning
    ``angle`` (rad) about ``n_hat`` (⊥ t0), of ``radius``.

    Returns ``(center, p_end, t_end_hat)`` with ``|center-p0| =
    |center-p_end| = radius`` (the add_arc through_point=False invariant).
    """
    to_center = np.cross(n_hat, t0_hat)          # unit, p0 → center
    center = np.asarray(p0, float) + radius * to_center
    p_end = center + _rodrigues(np.asarray(p0, float) - center, n_hat, angle)
    t_end = _unit(_rodrigues(t0_hat, n_hat, angle))
    return center, p_end, t_end


def hook_primitives(anchor, tangent, turn_dir, angle_deg, tail, radius,
                    true_arc):
    """Geometry primitives for a hook at a free bar/stirrup end.

    ``anchor`` is the end point; ``tangent`` is the bar direction pointing
    OUT of that end (where the hook continues). Returns
    ``(prims, fell_back)`` where each prim is
    ``("line", p0, p1)`` or ``("arc", p_start, center, p_end)`` (np arrays).
    A 180° hook is split into two 90° arcs (the add_arc shorter-arc
    boundary). ``true_arc=False`` emits a single sharp tail segment with
    the bend radius carried only as metadata.
    """
    t = _unit(tangent)
    if t is None:
        raise ValueError("hook_primitives: tangent is zero-length.")
    anchor = np.asarray(anchor, dtype=float)
    u, n, fell_back = bend_plane(t, turn_dir)
    theta = np.radians(float(angle_deg))

    if not true_arc:
        d_tail = np.cos(theta) * t + np.sin(theta) * u
        return [("line", anchor, anchor + float(tail) * d_tail)], fell_back

    R = float(radius)
    prims: list = []
    if angle_deg >= 179.9:                       # split 180° → two 90° arcs
        c1, p_mid, t_mid = _arc_segment(anchor, t, theta / 2.0, n, R)
        c2, p_end, t_end = _arc_segment(p_mid, t_mid, theta / 2.0, n, R)
        prims += [("arc", anchor, c1, p_mid), ("arc", p_mid, c2, p_end)]
    else:
        center, p_end, t_end = _arc_segment(anchor, t, theta, n, R)
        prims.append(("arc", anchor, center, p_end))
    prims.append(("line", p_end, p_end + float(tail) * t_end))
    return prims, fell_back


def outward_tangent(points, at_start: bool) -> np.ndarray:
    """Unit bar direction pointing OUT of an end (where a hook continues)."""
    pts = [np.asarray(p, float) for p in points]
    if at_start:
        v = pts[0] - pts[1]
    else:
        v = pts[-1] - pts[-2]
    u = _unit(v)
    if u is None:
        raise ValueError("outward_tangent: zero-length end segment.")
    return u
