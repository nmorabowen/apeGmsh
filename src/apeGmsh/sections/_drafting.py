"""Drafting aids for the section builder canvas (ADR 0080 B5).

**Qt-free, pure functions.** These are the AutoCAD-style input habits
— grid snap, object snap, ortho, typed exact input, and the segment
lock resolver — expressed as pure geometry over a
:class:`~apeGmsh.sections.SectionDocument` (or its plain dict). The Qt
canvas layer draws the marker glyphs and floating length/angle fields
and *calls* these; it owns no geometry logic of its own.

Because they add zero document state (they only decide which
coordinates get written into the document), the ADR 0080 parity law is
untouched — a headless script writes the same polygon by passing the
coordinates directly.

The public surface:

* :func:`snap_candidates` — every object-snap point implied by the
  document's resolved shape outlines (endpoints, midpoints, circle
  centers, circle quadrants, segment–segment intersections).
* :func:`resolve_snap` — pick the winning snap for a cursor position
  (**object snap beats grid snap**), or ``None`` when nothing snaps.
* :func:`ortho_project` — constrain a rubber-band segment to the
  nearest world axis (F8 ortho).
* :func:`constrain_segment` — the length/angle **lock resolver**
  (length-locked → vertex on a circle; angle-locked → on a ray; both →
  fully determined).
* :func:`parse_dynamic_input` — the one-box command-line parser
  (``"length<angle"`` polar, ``"@dx,dy"`` relative, ``"x,y"``
  absolute), raising :class:`DraftingInputError` on rejects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DraftingInputError",
    "GridSpec",
    "SnapCandidate",
    "constrain_segment",
    "ortho_project",
    "parse_dynamic_input",
    "resolve_snap",
    "shape_outlines",
    "snap_candidates",
]

Point = tuple[float, float]

#: Object-snap kinds, most-preferred first — the tie-breaker when two
#: candidates sit the same distance from the cursor (AutoCAD priority:
#: a vertex beats a midpoint at the same range).
_KIND_PRIORITY: tuple[str, ...] = (
    "endpoint", "intersection", "center", "quadrant", "midpoint",
)


class DraftingInputError(ValueError):
    """Typed-input string the dynamic-input box cannot parse."""


@dataclass(frozen=True)
class SnapCandidate:
    """One snap target: world coordinates plus the kind of feature it
    came from (drives which marker glyph the canvas draws)."""

    x: float
    y: float
    kind: str


@dataclass(frozen=True)
class GridSpec:
    """Rectangular snap grid — spacing and origin. Pass ``grid=None``
    to :func:`resolve_snap` to disable grid snapping entirely."""

    spacing: float
    origin: Point = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.spacing) or self.spacing <= 0.0:
            raise DraftingInputError(
                f"grid spacing must be a positive number, got {self.spacing!r}."
            )

    def nearest(self, x: float, y: float) -> Point:
        """The grid intersection closest to ``(x, y)``."""
        ox, oy = self.origin
        return (
            ox + round((x - ox) / self.spacing) * self.spacing,
            oy + round((y - oy) / self.spacing) * self.spacing,
        )


# ─────────────────────────────────────────────────────────────────────
# outline resolution (shared by the snap engine and the canvas)
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShapeOutlines:
    """Resolved world-frame geometry of one document shape.

    ``polylines`` are closed loops of vertices (outer boundary plus any
    holes); ``circles`` are ``(cx, cy, r)`` for round shapes. The canvas
    draws both, and the snap engine harvests candidates from them.
    """

    polylines: tuple[tuple[Point, ...], ...] = ()
    circles: tuple[tuple[float, float, float], ...] = ()


def _transform(
    pts: "list[Point]", translate: Point, rotate: float | None
) -> "list[Point]":
    """Rotate about the origin (degrees) then translate — the exact
    convention :func:`SectionDocument._build_polygon` and the ``*_face``
    placement use, so the canvas matches where the document builds."""
    tx, ty = translate
    if rotate is None:
        return [(x + tx, y + ty) for x, y in pts]
    th = math.radians(rotate)
    c, s = math.cos(th), math.sin(th)
    return [
        (x * c - y * s + tx, x * s + y * c + ty) for x, y in pts
    ]


def _face_local_polylines(shape: str, p: dict[str, Any]) -> "list[list[Point]]":
    """Local-frame outline loops for a parametric ``*_face`` shape,
    mirroring the profiles in :class:`SectionsBuilder`. Circular shapes
    return ``[]`` here and are handled as circles by the caller."""
    if shape == "rect_face":
        b, h = p["b"], p["h"]
        return [_rect(-b / 2, -h / 2, b, h)]
    if shape == "rect_hollow_face":
        b, h, t = p["b"], p["h"], p["t"]
        return [
            _rect(-b / 2, -h / 2, b, h),
            _rect(-b / 2 + t, -h / 2 + t, b - 2 * t, h - 2 * t),
        ]
    if shape == "W_face":
        bf, tf, h, tw = p["bf"], p["tf"], p["h"], p["tw"]
        ho = h / 2 + tf  # outer half-height
        hi = h / 2       # web/flange junction
        return [[
            (-bf / 2, -ho), (bf / 2, -ho), (bf / 2, -hi),
            (tw / 2, -hi), (tw / 2, hi), (bf / 2, hi),
            (bf / 2, ho), (-bf / 2, ho), (-bf / 2, hi),
            (-tw / 2, hi), (-tw / 2, -hi), (-bf / 2, -hi),
        ]]
    if shape == "channel_face":
        bf, tf, h, tw = p["bf"], p["tf"], p["h"], p["tw"]
        ho = h / 2 + tf
        hi = h / 2
        return [[
            (0.0, -ho), (bf, -ho), (bf, -hi), (tw, -hi),
            (tw, hi), (bf, hi), (bf, ho), (0.0, ho),
        ]]
    if shape == "tee_face":
        bf, tf, h, tw = p["bf"], p["tf"], p["h"], p["tw"]
        return [[
            (-tw / 2, -h), (tw / 2, -h), (tw / 2, 0.0),
            (bf / 2, 0.0), (bf / 2, tf), (-bf / 2, tf),
            (-bf / 2, 0.0), (-tw / 2, 0.0),
        ]]
    if shape == "angle_face":
        b, h, t = p["b"], p["h"], p["t"]
        return [[
            (0.0, 0.0), (b, 0.0), (b, t), (t, t), (t, h), (0.0, h),
        ]]
    return []


def _rect(x: float, y: float, dx: float, dy: float) -> "list[Point]":
    return [(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]


def shape_outlines(shape_dict: dict[str, Any]) -> ShapeOutlines:
    """Resolve one document shape entry to world-frame outlines.

    Handles the freehand ``polygon`` shape, the eight parametric
    ``*_face`` shapes, and the two circular shapes (``pipe_face`` /
    ``pipe_hollow_face``, returned as circles)."""
    shape = shape_dict.get("shape")
    translate = tuple(shape_dict.get("translate", (0.0, 0.0)))  # type: ignore[assignment]
    rotate = shape_dict.get("rotate")

    if shape == "polygon":
        pts = [(float(x), float(y)) for x, y in shape_dict["points"]]
        return ShapeOutlines(
            polylines=(tuple(_transform(pts, translate, rotate)),)
        )

    params = shape_dict.get("params", {})
    if shape in ("pipe_face", "pipe_hollow_face"):
        cx, cy = _transform([(0.0, 0.0)], translate, rotate)[0]
        r = float(params["r"])
        circles = [(cx, cy, r)]
        if shape == "pipe_hollow_face":
            circles.append((cx, cy, r - float(params["t"])))
        return ShapeOutlines(circles=tuple(circles))

    loops = _face_local_polylines(str(shape), params)
    return ShapeOutlines(
        polylines=tuple(
            tuple(_transform(loop, translate, rotate)) for loop in loops
        )
    )


def _as_data(document: Any) -> dict[str, Any]:
    return document.to_dict() if hasattr(document, "to_dict") else document


def document_outlines(document: Any) -> "list[ShapeOutlines]":
    """Every shape's resolved outlines (continuum lane; empty for the
    fiber lane, which has no freehand canvas)."""
    data = _as_data(document)
    return [shape_outlines(sh) for sh in data.get("shapes", [])]


# ─────────────────────────────────────────────────────────────────────
# snap candidates
# ─────────────────────────────────────────────────────────────────────


def _seg_intersection(
    a: Point, b: Point, c: Point, d: Point
) -> "Point | None":
    """Proper intersection of segments ``ab`` and ``cd`` within both,
    or ``None`` when parallel/non-crossing."""
    (x1, y1), (x2, y2) = a, b
    (x3, y3), (x4, y4) = c, d
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _dedup(cands: "list[SnapCandidate]") -> "list[SnapCandidate]":
    """Drop coincident candidates, keeping the highest-priority kind at
    each location."""
    rank = {k: i for i, k in enumerate(_KIND_PRIORITY)}
    best: dict[tuple[int, int], SnapCandidate] = {}
    for c in cands:
        key = (round(c.x, 9), round(c.y, 9))
        cur = best.get(key)
        if cur is None or rank.get(c.kind, 99) < rank.get(cur.kind, 99):
            best[key] = c
    return list(best.values())


def snap_candidates(
    document: Any,
    *,
    extra_points: "list[Point] | None" = None,
) -> "list[SnapCandidate]":
    """Object-snap candidates from the document's resolved outlines.

    Vertices → ``endpoint``; segment midpoints → ``midpoint``; circle
    centers → ``center``; circle N/E/S/W → ``quadrant``; segment–segment
    crossings → ``intersection``. At section-builder scale (tens of
    segments) the intersection scan is brute-force by design — no
    spatial index.

    ``extra_points`` are extra ``endpoint`` candidates (the in-progress
    polygon's committed vertices, which are not yet in the document).
    """
    outlines = document_outlines(document)
    cands: list[SnapCandidate] = []
    segments: list[tuple[Point, Point]] = []

    for out in outlines:
        for loop in out.polylines:
            n = len(loop)
            for i, (x, y) in enumerate(loop):
                cands.append(SnapCandidate(x, y, "endpoint"))
                a = loop[i]
                b = loop[(i + 1) % n]
                segments.append((a, b))
                mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
                cands.append(SnapCandidate(mx, my, "midpoint"))
        for cx, cy, r in out.circles:
            cands.append(SnapCandidate(cx, cy, "center"))
            cands.append(SnapCandidate(cx + r, cy, "quadrant"))
            cands.append(SnapCandidate(cx - r, cy, "quadrant"))
            cands.append(SnapCandidate(cx, cy + r, "quadrant"))
            cands.append(SnapCandidate(cx, cy - r, "quadrant"))

    # brute-force segment–segment intersections (skip shared segments)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            hit = _seg_intersection(*segments[i], *segments[j])
            if hit is not None:
                cands.append(SnapCandidate(hit[0], hit[1], "intersection"))

    for x, y in extra_points or ():
        cands.append(SnapCandidate(float(x), float(y), "endpoint"))

    return _dedup(cands)


def resolve_snap(
    cursor: Point,
    candidates: "list[SnapCandidate]",
    grid: "GridSpec | None",
    tolerance: float,
) -> "SnapCandidate | None":
    """Resolve the cursor to a snap point, or ``None`` when nothing
    applies.

    **Object snap beats grid snap**: if any candidate lies within
    ``tolerance`` of the cursor, the nearest one wins (ties broken by
    kind priority — endpoint over midpoint, etc.). Otherwise, when a
    ``grid`` is given, the cursor snaps to the nearest grid
    intersection (kind ``"grid"``). With no candidate in range and no
    grid, returns ``None`` (free cursor).
    """
    cx, cy = cursor
    rank = {k: i for i, k in enumerate(_KIND_PRIORITY)}
    best: SnapCandidate | None = None
    best_key: tuple[float, int] | None = None
    tol2 = tolerance * tolerance
    for c in candidates:
        d2 = (c.x - cx) ** 2 + (c.y - cy) ** 2
        if d2 > tol2:
            continue
        key = (d2, rank.get(c.kind, 99))
        if best_key is None or key < best_key:
            best, best_key = c, key
    if best is not None:
        return best
    if grid is not None:
        gx, gy = grid.nearest(cx, cy)
        return SnapCandidate(gx, gy, "grid")
    return None


# ─────────────────────────────────────────────────────────────────────
# ortho + the segment lock resolver
# ─────────────────────────────────────────────────────────────────────


def ortho_project(anchor: Point, cursor: Point) -> Point:
    """Constrain the rubber-band segment to the nearest world axis
    (AutoCAD F8): horizontal when the cursor's |dx| ≥ |dy|, vertical
    otherwise. Returns the projected cursor position."""
    ax, ay = anchor
    x, y = cursor
    if abs(x - ax) >= abs(y - ay):
        return (x, ay)   # horizontal
    return (ax, y)       # vertical


def constrain_segment(
    anchor: Point,
    cursor: Point,
    *,
    length: "float | None" = None,
    angle: "float | None" = None,
) -> Point:
    """The length/angle lock resolver for a rubber-band segment.

    * both locked → the vertex is fully determined:
      ``anchor + length·(cosθ, sinθ)``.
    * ``length`` only → the vertex slides on the circle of that radius
      about the anchor, in the cursor's direction (cursor on the anchor
      falls back to 0°).
    * ``angle`` only → the vertex slides on the ray from the anchor at
      that heading; its distance is the cursor's projection onto the
      ray (clamped to the ray, never behind the anchor).
    * neither → the cursor passes through unchanged.

    ``angle`` is in degrees, CCW from +x.
    """
    ax, ay = anchor
    cx, cy = cursor
    if length is not None and not (math.isfinite(length) and length >= 0):
        raise DraftingInputError(f"locked length must be ≥ 0, got {length!r}.")
    if angle is not None and not math.isfinite(angle):
        raise DraftingInputError(f"locked angle must be finite, got {angle!r}.")

    if angle is not None:
        th = math.radians(angle)
        dx, dy = math.cos(th), math.sin(th)
        if length is not None:                    # both → determined
            return (ax + length * dx, ay + length * dy)
        proj = (cx - ax) * dx + (cy - ay) * dy    # angle only → ray
        proj = max(0.0, proj)
        return (ax + proj * dx, ay + proj * dy)

    if length is not None:                        # length only → circle
        vx, vy = cx - ax, cy - ay
        norm = math.hypot(vx, vy)
        if norm < 1e-12:
            return (ax + length, ay)
        return (ax + length * vx / norm, ay + length * vy / norm)

    return (cx, cy)


# ─────────────────────────────────────────────────────────────────────
# typed exact (dynamic) input
# ─────────────────────────────────────────────────────────────────────


def _finite(tok: str, what: str) -> float:
    try:
        v = float(tok)
    except (TypeError, ValueError) as e:
        raise DraftingInputError(f"{what}: {tok!r} is not a number.") from e
    if not math.isfinite(v):
        raise DraftingInputError(f"{what}: {tok!r} is not finite.")
    return v


def parse_dynamic_input(text: str, *, anchor: "Point | None" = None) -> Point:
    """Parse a one-box dynamic-input string to a world point.

    Three forms (the AutoCAD command-line habit):

    * ``"length<angle"`` — polar, ``length`` at ``angle`` degrees from
      the anchor (``"35<30"``). Relative to the anchor.
    * ``"@dx,dy"`` — cartesian offset from the anchor (``"@10,-5"``).
    * ``"x,y"`` — absolute cartesian (``"10,-5"``); the anchor is
      ignored.

    Polar and ``@`` relative forms require an ``anchor``. Malformed
    strings (empty, wrong separators, non-numeric, wrong token count,
    non-finite, or a relative form with no anchor) raise
    :class:`DraftingInputError`.
    """
    if not isinstance(text, str):
        raise DraftingInputError(f"input must be a string, got {text!r}.")
    s = text.strip()
    if not s:
        raise DraftingInputError("empty input.")

    relative = s.startswith("@")
    if relative:
        s = s[1:].strip()

    if "<" in s:
        parts = s.split("<")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise DraftingInputError(
                f"polar input must be 'length<angle', got {text!r}."
            )
        if anchor is None:
            raise DraftingInputError(
                f"polar input {text!r} is relative — needs an anchor point."
            )
        length = _finite(parts[0].strip(), "polar length")
        angle = _finite(parts[1].strip(), "polar angle")
        if length < 0:
            raise DraftingInputError(
                f"polar length must be ≥ 0, got {length!r}."
            )
        th = math.radians(angle)
        return (anchor[0] + length * math.cos(th),
                anchor[1] + length * math.sin(th))

    if "," in s:
        parts = s.split(",")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise DraftingInputError(
                f"cartesian input must be 'x,y', got {text!r}."
            )
        a = _finite(parts[0].strip(), "x")
        b = _finite(parts[1].strip(), "y")
        if relative:
            if anchor is None:
                raise DraftingInputError(
                    f"relative input {text!r} needs an anchor point."
                )
            return (anchor[0] + a, anchor[1] + b)
        return (a, b)

    raise DraftingInputError(
        f"unrecognized input {text!r}; expected 'length<angle', "
        f"'@dx,dy', or 'x,y'."
    )
