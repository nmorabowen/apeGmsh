"""ClipGizmoRenderer — the in-viewport section-plane gizmos (ADR 0083 S2).

Per visible plane: a translucent quad sized to the reference bbox's
cross-section, plus an arrow along the plane normal whose cone tip is
the rotation grab handle. Both are ordinary scene-IR ``MeshLayer``s
pushed through the ``RenderBackend`` seam (INV-2: the gizmo goes
*through* the seam, not around it) with two deliberate flags:

* ``clip_exempt=True`` — the backend stamps its clip set onto every
  mesh/glyph actor it creates; a gizmo sliced by its own plane (or by
  another plane) is useless, so gizmo layers opt out.
* ``pickable=False`` — the quad covers the whole cross-section; a
  pickable quad would eat every node/element pick near the plane. The
  gizmo's own gestures live in ``_clip_gizmo_interactor`` at priority
  12 instead.

The arrow is a **mesh**, not a ``GlyphLayer``, on purpose: glyphs have
no in-place ``update_layer`` fast path — every update destroys and
re-creates the actor, which is the per-move flicker ADR 0081 L2
measured on legend drags. Both gizmo layers keep a constant topology
(4 quad points; 11 arrow points), so a drag re-uses the actors and
only moves points.

The renderer is a **projection** of ``ClipPlaneSetController`` state,
exactly like the panel: it reads planes / ``show_gizmos`` /
``gizmo_visible`` on :meth:`refresh` (driven by the viewer's
``CLIP_PLANES_CHANGED`` subscription) and never mutates anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

#: Layer-id prefix for every gizmo layer — namespaced so it can never
#: collide with a diagram's layers.
GIZMO_LAYER_PREFIX = "clip_gizmo:"

#: The quad extends a hair past the bbox cross-section so it reads as
#: a cutting tool, not a face of the model.
QUAD_MARGIN = 1.06
#: Floor on the quad half-extent, as a fraction of the bbox diagonal —
#: a plane whose normal is parallel to a flat model's thin axis still
#: gets a grabbable quad.
MIN_HALF_FRAC = 0.05
#: Arrow length as a fraction of the bbox diagonal.
ARROW_LEN_FRAC = 0.22
#: Cone-tip length as a fraction of the arrow length.
TIP_LEN_FRAC = 0.30
#: Cone-tip radius as a fraction of the arrow length. Doubles as the
#: shaft's grab radius in the interactor's hit test.
TIP_RADIUS_FRAC = 0.09
_TIP_SIDES = 8

QUAD_RGB = (0.35, 0.62, 0.85)
QUAD_OPACITY = 0.25
ARROW_RGB = (1.0, 0.53, 0.0)


@dataclass(frozen=True)
class GizmoGeometry:
    """One gizmo's world-space geometry — shared by the layer builder
    and the interactor's hit test, so what you grab is exactly what you
    see."""

    plane_id: str
    #: Quad centre: the reference-bbox centre projected onto the plane.
    centre: tuple[float, float, float]
    #: The plane's stored unit normal — the axis a drag slides along
    #: (``offset`` is measured along it, flipped or not).
    normal: tuple[float, float, float]
    #: The arrow's direction: the *surviving* half-space (folds
    #: ``flipped`` in), so the arrow always points at what you keep.
    direction: tuple[float, float, float]
    #: In-plane orthonormal basis for the quad.
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    half_u: float
    half_v: float
    arrow_len: float
    tip_len: float
    tip_radius: float

    @property
    def tip_point(self) -> tuple[float, float, float]:
        c, d = np.asarray(self.centre), np.asarray(self.direction)
        return tuple(c + self.arrow_len * d)

    @property
    def shaft_end(self) -> tuple[float, float, float]:
        c, d = np.asarray(self.centre), np.asarray(self.direction)
        return tuple(c + (self.arrow_len - self.tip_len) * d)


def _unit(vec: np.ndarray) -> np.ndarray:
    mag = float(np.linalg.norm(vec))
    return vec / mag if mag > 0.0 else np.array([1.0, 0.0, 0.0])


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal in-plane pair for ``normal``."""
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(normal[2])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = _unit(np.cross(normal, ref))
    v = np.cross(normal, u)
    return u, v


def gizmo_geometry(plane: Any, bbox: Sequence[float]) -> GizmoGeometry:
    """The gizmo for ``plane`` against the reference bbox.

    ``bbox`` is ``(xmin, ymin, zmin, xmax, ymax, zmax)`` — the same
    reference the panel's offset slider maps, so the quad and the
    slider always talk about the same geometry.
    """
    n = _unit(np.asarray(plane.normal, dtype=float))
    d = -n if plane.flipped else n
    lo = np.asarray(bbox[:3], dtype=float)
    hi = np.asarray(bbox[3:], dtype=float)
    diag = float(np.linalg.norm(hi - lo)) or 1.0

    # Quad centre: bbox centre projected onto the plane, so the quad
    # tracks the model's cross-section as the plane slides.
    b = (lo + hi) / 2.0
    centre = b - (float(b @ n) - float(plane.offset)) * n

    u, v = _basis(n)
    corners = np.array([
        (x, y, z)
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ])
    rel = corners - centre
    floor = MIN_HALF_FRAC * diag
    half_u = max(float(np.max(np.abs(rel @ u))) * QUAD_MARGIN, floor)
    half_v = max(float(np.max(np.abs(rel @ v))) * QUAD_MARGIN, floor)

    arrow_len = ARROW_LEN_FRAC * diag
    return GizmoGeometry(
        plane_id=plane.plane_id,
        centre=tuple(centre),
        normal=tuple(n),
        direction=tuple(d),
        u=tuple(u),
        v=tuple(v),
        half_u=half_u,
        half_v=half_v,
        arrow_len=arrow_len,
        tip_len=TIP_LEN_FRAC * arrow_len,
        tip_radius=TIP_RADIUS_FRAC * arrow_len,
    )


# =====================================================================
# Ray hit-testing (world space — the interactor's half of the deal)
# =====================================================================

MODE_TRANSLATE = "translate"
MODE_ROTATE = "rotate"

#: Grab slack on the tip sphere — a handle you must hit exactly is a
#: handle nobody hits.
_TIP_GRAB = 1.8


def ray_hit(
    geom: GizmoGeometry,
    origin: Sequence[float],
    direction: Sequence[float],
) -> "Optional[tuple[str, float]]":
    """``(mode, ray_param)`` where the ray grabs this gizmo, or ``None``.

    The tip handle wins over the quad/shaft regardless of depth: it is
    a deliberately small target (rotate is the rarer gesture), and the
    quad behind it is huge — losing a tip-sized patch of quad costs
    nothing, while a tip you can only grab from the arrow side would.
    """
    o = np.asarray(origin, dtype=float)
    d = _unit(np.asarray(direction, dtype=float))

    # Tip sphere (rotate).
    tip = np.asarray(geom.tip_point)
    w = tip - o
    tca = float(w @ d)
    if tca > 0.0:
        dist2 = float(w @ w) - tca * tca
        r = geom.tip_radius * _TIP_GRAB
        if dist2 <= r * r:
            return (MODE_ROTATE, tca)

    hits: list[float] = []

    # Quad (translate).
    n = np.asarray(geom.normal)
    denom = float(d @ n)
    if abs(denom) > 1e-12:
        t = float((np.asarray(geom.centre) - o) @ n) / denom
        if t > 0.0:
            p = o + t * d - np.asarray(geom.centre)
            if (
                abs(float(p @ np.asarray(geom.u))) <= geom.half_u
                and abs(float(p @ np.asarray(geom.v))) <= geom.half_v
            ):
                hits.append(t)

    # Arrow shaft (translate) — a segment with the tip radius as its
    # grab radius.
    a = np.asarray(geom.centre)
    seg = np.asarray(geom.shaft_end) - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len > 0.0:
        # Closest point on the segment to the ray line, clamped to the
        # segment, then the ray's own closest approach to that point.
        seg_dir = seg / seg_len
        s = axis_param(a, seg_dir, o, d)
        s = min(max(s if s is not None else 0.0, 0.0), seg_len)
        p_seg = a + s * seg_dir
        t = max(float((p_seg - o) @ d), 0.0)
        if float(np.linalg.norm(p_seg - (o + t * d))) <= geom.tip_radius:
            hits.append(t)

    if not hits:
        return None
    return (MODE_TRANSLATE, min(hits))


#: Sensitivity floor for the *drag* projection (S2 review finding C1).
#: ``1 - (n.d)^2`` is ``sin^2`` of the angle between the ray and the
#: drag axis, and it divides the result — so as the camera swings onto
#: the axis the gesture's gain runs away: measured at 1 degree off
#: face-on, one pixel moved the plane 21 % of the model, and at 0.1
#: degrees a single pixel threw it clean off. The raw guard below only
#: catches true degeneracy (~6e-5 degrees), which no user reaches.
#: 0.06 is sin^2(14 degrees): inside that cone the gain saturates
#: instead of exploding. Looking *exactly* down the normal still moves
#: nothing — there is no screen direction that means "along the axis"
#: — but it now degrades to dead rather than to violent.
DRAG_MIN_SEPARATION = 0.06


def axis_param(
    axis_point: Sequence[float],
    axis_dir: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[float],
    *,
    min_separation: Optional[float] = None,
) -> Optional[float]:
    """Signed distance along the axis of the point nearest the ray.

    The translate gesture in one number: the press stores this for the
    press ray, every move recomputes it for the current ray, and the
    offset delta is the difference. ``None`` when the ray runs parallel
    to the axis (the projection is unbounded — keep the last value).

    ``min_separation`` clamps the ``1 - (n.d)^2`` denominator up to that
    floor and therefore never returns ``None`` — the drag path passes
    :data:`DRAG_MIN_SEPARATION` to bound its gain near face-on viewing.
    The hit test leaves it unset and keeps the exact projection.
    """
    n = _unit(np.asarray(axis_dir, dtype=float))
    d = _unit(np.asarray(direction, dtype=float))
    w = np.asarray(axis_point, dtype=float) - np.asarray(origin, dtype=float)
    b = float(n @ d)
    denom = 1.0 - b * b
    if min_separation is not None:
        denom = max(denom, float(min_separation))
    elif denom < 1e-12:
        return None
    return (b * float(w @ d) - float(w @ n)) / denom


def sphere_point(
    centre: Sequence[float],
    radius: float,
    origin: Sequence[float],
    direction: Sequence[float],
) -> Optional[np.ndarray]:
    """Where the ray meets the virtual trackball around ``centre``.

    The near intersection when the ray pierces the sphere; the ray's
    closest approach otherwise — so the rotate gesture keeps working
    smoothly as the cursor slides off the sphere's silhouette. ``None``
    when the whole sphere is behind the ray origin.
    """
    c = np.asarray(centre, dtype=float)
    o = np.asarray(origin, dtype=float)
    d = _unit(np.asarray(direction, dtype=float))
    w = c - o
    tca = float(w @ d)
    if tca <= 0.0:
        return None
    dist2 = float(w @ w) - tca * tca
    r2 = radius * radius
    t = tca - float(np.sqrt(r2 - dist2)) if dist2 < r2 else tca
    return o + t * d


# =====================================================================
# Layer construction
# =====================================================================


def _quad_layer(geom: GizmoGeometry) -> Any:
    from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

    c = np.asarray(geom.centre)
    u = np.asarray(geom.u) * geom.half_u
    v = np.asarray(geom.v) * geom.half_v
    pts = np.array([c - u - v, c + u - v, c + u + v, c - u + v])
    return MeshLayer(
        layer_id=f"{GIZMO_LAYER_PREFIX}{geom.plane_id}:quad",
        points=PointSet(pts),
        cells=CellBlocks({"quad": np.array([[0, 1, 2, 3]])}),
        color=ColorSpec(mode="solid", solid_rgb=QUAD_RGB),
        opacity=QUAD_OPACITY,
        pickable=False,
        clip_exempt=True,
    )


def _arrow_layer(geom: GizmoGeometry) -> Any:
    from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

    c = np.asarray(geom.centre)
    tip = np.asarray(geom.tip_point)
    base = np.asarray(geom.shaft_end)
    u, v = np.asarray(geom.u), np.asarray(geom.v)

    angles = np.linspace(0.0, 2.0 * np.pi, _TIP_SIDES, endpoint=False)
    ring = base + geom.tip_radius * (
        np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    )
    # 0 = shaft base, 1 = cone base centre, 2 = apex, 3.. = ring.
    pts = np.vstack([c, base, tip, ring])

    side = [
        [2, 3 + i, 3 + (i + 1) % _TIP_SIDES] for i in range(_TIP_SIDES)
    ]
    cap = [
        [1, 3 + (i + 1) % _TIP_SIDES, 3 + i] for i in range(_TIP_SIDES)
    ]
    return MeshLayer(
        layer_id=f"{GIZMO_LAYER_PREFIX}{geom.plane_id}:arrow",
        points=PointSet(pts),
        cells=CellBlocks({
            "line": np.array([[0, 1]]),
            "triangle": np.array(side + cap),
        }),
        color=ColorSpec(mode="solid", solid_rgb=ARROW_RGB),
        line_width=3.0,
        pickable=False,
        clip_exempt=True,
    )


# =====================================================================
# Renderer
# =====================================================================


class ClipGizmoRenderer:
    """Projects the plane set onto gizmo layers for one viewport."""

    def __init__(
        self, backend: Any, controller: Any, *, bbox: Sequence[float],
    ) -> None:
        self._backend = backend
        self._controller = controller
        self._bbox = tuple(float(c) for c in bbox)
        # Strong refs, deliberately: the backend's clip registry holds
        # handles weakly, so whoever adds a layer must keep its handle
        # alive (the same contract diagrams follow).
        self._handles: "dict[str, tuple[Any, Any]]" = {}
        self._geoms: "dict[str, GizmoGeometry]" = {}

    def geometries(self) -> "dict[str, GizmoGeometry]":
        """Current gizmo geometry by plane id — the hit-test surface."""
        return dict(self._geoms)

    def refresh(self) -> None:
        """Reconcile gizmo layers against the controller's state.

        Same-topology updates go through ``update_layer``'s in-place
        fast path (both layers keep constant point/cell counts), so a
        drag moves the existing actors instead of rebuilding them —
        the 0081 L2 flicker lesson, applied up front.
        """
        controller = self._controller
        desired: "dict[str, GizmoGeometry]" = {}
        if controller is not None and controller.show_gizmos:
            for plane in controller.planes():
                if plane.gizmo_visible:
                    desired[plane.plane_id] = gizmo_geometry(
                        plane, self._bbox,
                    )

        for plane_id in list(self._handles):
            if plane_id not in desired:
                quad, arrow = self._handles.pop(plane_id)
                for handle in (quad, arrow):
                    try:
                        self._backend.remove_layer(handle)
                    except Exception:
                        pass

        for plane_id, geom in desired.items():
            quad_layer = _quad_layer(geom)
            arrow_layer = _arrow_layer(geom)
            try:
                if plane_id in self._handles:
                    quad, arrow = self._handles[plane_id]
                    self._backend.update_layer(quad, quad_layer)
                    self._backend.update_layer(arrow, arrow_layer)
                else:
                    self._handles[plane_id] = (
                        self._backend.add_layer(quad_layer),
                        self._backend.add_layer(arrow_layer),
                    )
            except Exception:
                # A gizmo that failed to render must not stay in the
                # hit-test surface claiming presses it cannot show.
                self._handles.pop(plane_id, None)

        self._geoms = {
            pid: geom
            for pid, geom in desired.items()
            if pid in self._handles
        }

    def clear(self) -> None:
        """Drop every gizmo layer. Idempotent."""
        for quad, arrow in self._handles.values():
            for handle in (quad, arrow):
                try:
                    self._backend.remove_layer(handle)
                except Exception:
                    pass
        self._handles.clear()
        self._geoms.clear()


__all__ = [
    "GIZMO_LAYER_PREFIX",
    "MODE_ROTATE",
    "MODE_TRANSLATE",
    "ClipGizmoRenderer",
    "GizmoGeometry",
    "axis_param",
    "gizmo_geometry",
    "ray_hit",
    "sphere_point",
]
