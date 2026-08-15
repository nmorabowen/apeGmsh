"""Pure polyline vertex fillet / chamfer expansion (ADR 0097).

No gmsh — the geometry composite turns the returned segments into OCC
points + lines + arcs. Unit-testable off-session.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TOL = 1e-12
# Interior vertices whose turning angle exceeds this, with no fillet
# or chamfer, warn — OCC addPipe kinks or fails at sharp corners.
SHARP_TURN_RAD = math.radians(30.0)


@dataclass(frozen=True)
class PolylineSegment:
    """One output curve of an expanded polyline.

    ``kind`` is ``"line"`` or ``"arc"``. Arcs carry ``center`` (the
    circle centre); lines leave it ``None``.
    """

    kind: str
    start: np.ndarray
    end: np.ndarray
    center: np.ndarray | None = None


def _as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "add_polyline: points must be an Nx3 sequence of (x, y, z); "
            f"got shape {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"add_polyline: need at least 2 points, got {arr.shape[0]}."
        )
    return arr


def normalize_polyline_points(points, *, closed: bool) -> np.ndarray:
    """Drop a repeated closing vertex and validate length."""
    arr = _as_points(points)
    if closed:
        if arr.shape[0] >= 2 and np.linalg.norm(arr[0] - arr[-1]) <= _TOL:
            arr = arr[:-1]
        if arr.shape[0] < 3:
            raise ValueError(
                "add_polyline: a closed polyline needs at least 3 "
                "distinct vertices."
            )
    return np.ascontiguousarray(arr)


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > _TOL else None


def vertex_turn(
    points: np.ndarray, i: int, *, closed: bool,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Incoming / outgoing unit tangents and turning angle at vertex *i*.

    Turning angle is 0 for a collinear joint and π for a U-turn.
    Returns ``None`` when the vertex is an open-polyline endpoint or
    either adjacent segment is degenerate.
    """
    n = int(points.shape[0])
    if not closed and (i <= 0 or i >= n - 1):
        return None
    prev_i = (i - 1) % n
    next_i = (i + 1) % n
    u_in = _unit(points[i] - points[prev_i])
    u_out = _unit(points[next_i] - points[i])
    if u_in is None or u_out is None:
        return None
    turn = float(np.arccos(np.clip(np.dot(u_in, u_out), -1.0, 1.0)))
    return u_in, u_out, turn


def _corner_setback(
    points: np.ndarray, i: int, *, closed: bool, radius: float | None,
    chamfer: float | None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    """Setback *t* along each leg, T1, T2, and fillet centre (or None)."""
    frame = vertex_turn(points, i, closed=closed)
    if frame is None:
        raise ValueError(
            f"add_polyline: vertex {i} cannot take a fillet/chamfer "
            f"(open-polyline endpoints, or a degenerate adjacent segment)."
        )
    u_in, u_out, turn = frame
    # Angle between the two legs measured from the vertex (outgoing).
    a = -u_in
    b = u_out
    phi = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    if phi <= 1e-8:
        raise ValueError(
            f"add_polyline: vertex {i} is a U-turn (leg angle ~0); "
            f"fillet/chamfer is undefined. Split the corner or pick a "
            f"shallower joint."
        )
    if phi >= math.pi - 1e-8:
        raise ValueError(
            f"add_polyline: vertex {i} is collinear; there is no corner "
            f"to fillet or chamfer."
        )
    p = points[i]
    if radius is not None:
        t = float(radius) / math.tan(phi / 2.0)
        u_bis = _unit(a + b)
        if u_bis is None:
            raise ValueError(
                f"add_polyline: vertex {i} has a degenerate angle bisector."
            )
        center = p + u_bis * (float(radius) / math.sin(phi / 2.0))
        t1 = p + a * t
        t2 = p + b * t
        return t, t1, t2, center
    t = float(chamfer)
    t1 = p + a * t
    t2 = p + b * t
    return t, t1, t2, None


def sharp_untreated_vertices(
    points: np.ndarray, *, closed: bool, treated: set[int],
    threshold: float = SHARP_TURN_RAD,
) -> list[tuple[int, float]]:
    """``(vertex_index, turning_deg)`` for untreated sharp interiors."""
    n = int(points.shape[0])
    out: list[tuple[int, float]] = []
    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        if i in treated:
            continue
        frame = vertex_turn(points, i, closed=closed)
        if frame is None:
            continue
        _, _, turn = frame
        if turn > threshold:
            out.append((i, math.degrees(turn)))
    return out


def expand_polyline(
    points,
    *,
    closed: bool = False,
    fillet: dict[int, float] | None = None,
    chamfer: dict[int, float] | None = None,
) -> list[PolylineSegment]:
    """Expand control points into line/arc segments with vertex treatment.

    *fillet* / *chamfer* map vertex index → radius / setback. The same
    vertex cannot appear in both maps. Open-polyline endpoints cannot
    be treated. Setbacks on both ends of a segment must sum to less
    than that segment's length.
    """
    pts = normalize_polyline_points(points, closed=closed)
    n = int(pts.shape[0])
    fillet = dict(fillet or {})
    chamfer = dict(chamfer or {})

    overlap = set(fillet) & set(chamfer)
    if overlap:
        raise ValueError(
            f"add_polyline: vertex {sorted(overlap)[0]} is in both "
            f"fillet= and chamfer=; pass one treatment per vertex."
        )
    for src, name in ((fillet, "fillet"), (chamfer, "chamfer")):
        for i, val in src.items():
            if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= n:
                raise ValueError(
                    f"add_polyline: {name} key {i!r} is not a vertex "
                    f"index in 0..{n - 1}."
                )
            if float(val) <= 0.0:
                raise ValueError(
                    f"add_polyline: {name}[{i}] must be > 0, got {val}."
                )

    treated: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray | None, str]] = {}
    for i, r in fillet.items():
        t, t1, t2, c = _corner_setback(
            pts, i, closed=closed, radius=float(r), chamfer=None,
        )
        treated[i] = (t, t1, t2, c, "fillet")
    for i, d in chamfer.items():
        t, t1, t2, c = _corner_setback(
            pts, i, closed=closed, radius=None, chamfer=float(d),
        )
        treated[i] = (t, t1, t2, c, "chamfer")

    nseg = n if closed else n - 1
    for i in range(nseg):
        j = (i + 1) % n
        length = float(np.linalg.norm(pts[j] - pts[i]))
        t_i = treated[i][0] if i in treated else 0.0
        t_j = treated[j][0] if j in treated else 0.0
        if t_i + t_j >= length - 1e-9:
            raise ValueError(
                f"add_polyline: fillet/chamfer setbacks on vertices "
                f"{i} and {j} ({t_i:g} + {t_j:g}) meet or exceed "
                f"segment length {length:g}. Reduce the radius/setback."
            )

    segments: list[PolylineSegment] = []
    for i in range(nseg):
        j = (i + 1) % n
        start = treated[i][2] if i in treated else pts[i]  # T2 of i
        end = treated[j][1] if j in treated else pts[j]    # T1 of j
        segments.append(PolylineSegment("line", start, end, None))
        if j in treated:
            _, t1, t2, center, kind = treated[j]
            if kind == "fillet":
                segments.append(PolylineSegment("arc", t1, t2, center))
            else:
                segments.append(PolylineSegment("line", t1, t2, None))
    return segments
