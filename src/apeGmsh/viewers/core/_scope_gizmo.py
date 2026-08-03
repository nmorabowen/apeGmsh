"""ScopeGizmoRenderer — the in-viewport scope box (catalog).

The drag half of the scope box (``core/scope_controller``): a wireframe
box with six small square **grips**, one centred on each face, drawn for
the ACTIVE geometry only. Grab a grip, slide it along its own world
axis, and that one bound moves.

**The grip is the grab area, and the grip is drawn.** The first cut of
this gizmo used the six FULL faces as handles, which made the box a
solid claim over its own interior: because enabling the scope seeds the
box to the geometry's own bounds, the box encloses the model, so every
press over the model silhouette became a face drag — node pick, element
pick and rubber-band box-pick all died silently. A grip is instead a
fixed, small square (:data:`GRIP_HALF_FRAC` of the REFERENCE model
diagonal), so:

* a press over the model interior misses every grip and falls through
  un-aborted to the picker at priority 10 — click-pick and rubber-band
  keep working with a scope active;
* the grips do not shrink as the box grows, so a model-sized box is
  still draggable, and they do not vanish as the box collapses, so a
  squashed box is still recoverable;
* the grab region is **exactly** the drawn quad — :func:`ray_hit` and
  :func:`_grips_layer` both read
  :meth:`~ScopeGizmoGeometry.grip_quad` — so a click outside the
  visible handle can never start a drag.

Nothing translucent is drawn over the box interior: a face-sized
translucent quad is what made the old gizmo *look* like it owned every
pixel it was in fact claiming, and dropping it is what lets the grips
sit exactly in their faces' planes without z-fighting against it.

Deliberately a much smaller thing than the clip gizmo it sits next to
(ADR 0083 S2), because :class:`~..scene_ir.BBox` is **axis-aligned by
construction**: there is no rotation, so there is no trackball, no
``sphere_point``, and no pivot re-anchoring. Every gesture is a pure
translate along one world axis — which is exactly what
:func:`~._clip_gizmo.axis_param` already computes, so that function and
its :data:`~._clip_gizmo.DRAG_MIN_SEPARATION` floor are imported rather
than re-derived.

Two scene-IR ``MeshLayer``s go through the ``RenderBackend`` seam (INV-2)
with the same two flags the clip gizmo's layers carry:

* ``clip_exempt=True`` — a scope box sliced by a section plane is
  useless.
* ``pickable=False`` — a gizmo that answered the node/element picker
  would put itself in the pick results. The gestures live in
  ``_scope_gizmo_interactor`` at priority 12 instead.

The gizmo is also immune to the scope filter it drives, and for free:
the filter is an ``ElementVisibility`` layer on a *geometry's* scene
grid, and these layers are backend layers of their own that belong to
no scene.

Both layers keep a constant topology (8 points / 12 lines; 24 points /
6 quads), so a drag re-uses the actors through ``update_layer``'s
in-place fast path and rebuilds nothing — the 0081 L2 flicker lesson.

Like ``ClipGizmoRenderer`` this is a **projection** of controller state:
it reads ``show_gizmo`` and ``box_for(active)`` on :meth:`refresh` and
never mutates anything.

**Active geometry only.** ADR 0058 makes "active" the editing target,
and one box per scoped geometry would be a pile of overlapping faces
with no way to say which one you meant to grab.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..scene_ir import BBox
from ._clip_gizmo import DRAG_MIN_SEPARATION, axis_param

#: Layer-id prefix for every scope-gizmo layer — namespaced so it can
#: never collide with a diagram's layers or the clip gizmo's.
SCOPE_GIZMO_LAYER_PREFIX = "scope_gizmo:"

#: The six face handles. **The order is the tie-break**: when a ray
#: grazes an edge and two faces report the same distance, the earlier
#: one here wins, so an ambiguous press resolves the same way every
#: time instead of following dict iteration luck.
FACE_HANDLES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

#: ``handle -> (axis index, is_max)``.
_FACE_SPEC: "dict[str, tuple[int, bool]]" = {
    handle: (axis, bool(side))
    for axis, name in enumerate("xyz")
    for side, handle in enumerate((f"{name}min", f"{name}max"))
}

#: A grip's in-plane half-extent, as a fraction of the REFERENCE model
#: diagonal — the same shape as the clip gizmo's ``MIN_HALF_FRAC``, and
#: like it, deliberately a fraction of the MODEL rather than of the box.
#:
#: Constant in world units is what makes the grip work at both ends of
#: the range. Sized to the FACE it would grow with the box until it
#: covered the model (the defect this replaces) and shrink to nothing as
#: the box collapsed (a face with no area is a face nobody can grab). A
#: constant is neither: on a model-sized box a grip covers ~2 % of its
#: face's area, so a press over the model interior misses it; on a box
#: squashed to a point the six grips are still full size, so the box is
#: still draggable back open.
GRIP_HALF_FRAC = 0.045

EDGE_RGB = (0.20, 0.90, 0.55)
GRIP_RGB = (0.35, 0.95, 0.60)
GRIP_OPACITY = 0.85

_CORNER_SIGNS = np.array(
    [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ],
    dtype=bool,
)
_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class ScopeGizmoGeometry:
    """One scope box's world-space geometry.

    Shared by the layer builder and the interactor's hit test — and now
    with **no exception at all**: both read :meth:`grip_quad`, so the
    grab region and the drawn handle are the same four corners.
    """

    geometry_id: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    #: A grip's in-plane half-extent (see :data:`GRIP_HALF_FRAC`).
    grip_half: float

    @property
    def corners(self) -> np.ndarray:
        """The 8 corners, ``(8, 3)`` — :attr:`BBox.corners8`'s order."""
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        return np.where(_CORNER_SIGNS, hi, lo)

    def bound(self, handle: str) -> float:
        """The world coordinate the named face currently sits at."""
        axis, is_max = _FACE_SPEC[handle]
        return float((self.hi if is_max else self.lo)[axis])

    def face_centre(self, handle: str) -> tuple[float, float, float]:
        """The face's centre — the drag axis's anchor point."""
        axis, _is_max = _FACE_SPEC[handle]
        centre = (
            np.asarray(self.lo, dtype=float)
            + np.asarray(self.hi, dtype=float)
        ) / 2.0
        centre[axis] = self.bound(handle)
        return tuple(centre)

    def grip_quad(self, handle: str) -> np.ndarray:
        """The named grip's four corners, ``(4, 3)``, wound as a cycle.

        A square of half-extent :attr:`grip_half` centred on the face
        centre and lying IN the face's plane. This is the whole grab
        surface: :func:`ray_hit` tests exactly this rectangle and
        :func:`_grips_layer` draws exactly this rectangle, so a press
        that starts a drag is always a press on something the user can
        see (review finding F8 — the old floored face-containment test
        stayed grabbable well outside a thin box's drawn silhouette).
        """
        centre = np.asarray(self.face_centre(handle), dtype=float)
        half = float(self.grip_half)
        e0, e1 = (np.zeros(3), np.zeros(3))
        a0, a1 = in_plane_axes(handle)
        e0[a0] = half
        e1[a1] = half
        return np.array([
            centre - e0 - e1, centre + e0 - e1,
            centre + e0 + e1, centre - e0 + e1,
        ])

    @property
    def box(self) -> BBox:
        """The box as the canonical value type (ADR 0045 INV-2)."""
        return BBox(self.lo, self.hi)


def face_axis(handle: str) -> int:
    """The world axis index (0/1/2) the named face slides along."""
    return _FACE_SPEC[handle][0]


def in_plane_axes(handle: str) -> "tuple[int, int]":
    """The two world axis indices spanning the named face's plane."""
    axis = face_axis(handle)
    return tuple(a for a in (0, 1, 2) if a != axis)  # type: ignore[return-value]


def face_axis_dir(handle: str) -> tuple[float, float, float]:
    """The unit world axis the named face slides along.

    Always the POSITIVE axis, both for the min face and the max one:
    the drag maps a signed distance along it straight onto a bound, and
    flipping the direction for min faces would invert the gesture.
    """
    vec = [0.0, 0.0, 0.0]
    vec[face_axis(handle)] = 1.0
    return tuple(vec)


def scope_gizmo_geometry(
    geometry_id: str, box: BBox, bbox: Sequence[float],
) -> ScopeGizmoGeometry:
    """The gizmo for ``box`` on ``geometry_id``.

    ``bbox`` is the reference model extent
    ``(xmin, ymin, zmin, xmax, ymax, zmax)`` — used only to size
    :attr:`~ScopeGizmoGeometry.grip_half`, so a grip is a fraction of
    the MODEL rather than of a box the user may have grown to fill the
    viewport or collapsed to nothing.
    """
    lo = np.asarray(bbox[:3], dtype=float)
    hi = np.asarray(bbox[3:], dtype=float)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    return ScopeGizmoGeometry(
        geometry_id=str(geometry_id),
        lo=tuple(float(c) for c in box.min),
        hi=tuple(float(c) for c in box.max),
        grip_half=GRIP_HALF_FRAC * diag,
    )


def rebased(geom: ScopeGizmoGeometry, box: BBox) -> ScopeGizmoGeometry:
    """``geom`` re-read against a (possibly newer) ``box``.

    The drag's answer to review finding F11: a press must freeze the
    axis ANCHOR (or the axis point chases the face it is measuring and
    the gesture drifts), but freezing all six BOUNDS as well means an
    external write landing mid-drag — a panel edit, a session restore —
    is silently reverted by the next mouse-move, which rebuilds the box
    from the press-time snapshot. Only the anchor is frozen; the bounds
    come through here, live, every move.
    """
    return replace(
        geom,
        lo=tuple(float(c) for c in box.min),
        hi=tuple(float(c) for c in box.max),
    )


# =====================================================================
# Ray hit-testing (world space — the interactor's half of the deal)
# =====================================================================


def ray_hit(
    geom: ScopeGizmoGeometry,
    origin: Sequence[float],
    direction: Sequence[float],
) -> "Optional[tuple[str, float]]":
    """``(handle, ray_param)`` where the ray grabs a GRIP, or ``None``.

    Six small axis-aligned squares — the ones :func:`_grips_layer`
    draws, corner for corner — tested independently; the **nearest**
    hit wins, so the grip you can see is the grip you get rather than
    the one behind it. Two determinism rules:

    * A face **edge-on** to the ray (``|d[axis]|`` at the epsilon) is
      not hit at all. It is a zero-pixel target, and admitting it would
      make the winner depend on floating-point noise in a denominator
      that is about to divide.
    * On an exact tie — most often an axis the box has collapsed, where
      the two grips are the same square — the earlier handle in
      :data:`FACE_HANDLES` wins, because the loop only replaces the
      incumbent on a *strictly* nearer hit.

    The in-plane containment test is
    :attr:`~ScopeGizmoGeometry.grip_half` about the FACE CENTRE and is
    **independent of the box's own extent** (review finding F1). That
    independence is the whole design: it is what stops a model-sized
    box from claiming every press over the model, and equally what
    keeps a box collapsed to a point grabbable instead of trapping the
    user in the panel (see :func:`clamp_bound`).
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    mag = float(np.linalg.norm(d))
    if mag <= 0.0:
        return None
    d = d / mag
    half = float(geom.grip_half)

    best: "Optional[tuple[str, float]]" = None
    for handle in FACE_HANDLES:
        axis = face_axis(handle)
        if abs(float(d[axis])) <= 1e-12:
            continue                            # edge-on: not a target
        t = (geom.bound(handle) - float(o[axis])) / float(d[axis])
        if t <= 0.0:
            continue                            # behind the eye
        if best is not None and t >= best[1]:
            continue                            # farther, or a tie
        point = o + t * d
        centre = np.asarray(geom.face_centre(handle), dtype=float)
        other = list(in_plane_axes(handle))
        if bool(np.all(np.abs(point[other] - centre[other]) <= half)):
            best = (handle, t)
    return best


# =====================================================================
# The clamp — a drag must never construct an invalid BBox
# =====================================================================


def resolve_side(
    geom: ScopeGizmoGeometry, handle: str, value: float,
) -> bool:
    """Which bound the drag actually moves — ``True`` for the max one.

    The named one, except on an axis the box has already collapsed to
    **zero extent**. There the two faces sit at the same coordinate and
    are the same pixels, so which of them the hit test named is an
    arbitrary tie-break (declared order, min first). Honouring it would
    make the collapsed box draggable in one direction only — grab the
    face, push outwards, and the clamp below pins it in place — which is
    the "face you can no longer grab" trap in a subtler shape. At
    coincidence the gesture is therefore free to take **either** bound,
    and it takes the one the motion implies.
    """
    axis, is_max = _FACE_SPEC[handle]
    if geom.lo[axis] == geom.hi[axis]:
        return float(value) > float(geom.lo[axis])
    return is_max


def clamp_bound(geom: ScopeGizmoGeometry, handle: str, value: float) -> float:
    """``value`` clamped so the dragged face cannot pass its opposite.

    ``BBox.__post_init__`` RAISES on ``min > max``, so an unclamped drag
    would throw out of a Qt mouse handler the moment a face crossed. The
    rule is **stop, do not cross and do not swap**: a min face clamps at
    the max bound and a max face at the min one, and the box the drag
    hands to the controller is valid by construction.

    **At exact coincidence** the box keeps a zero extent on that axis.
    That is legal (``BBox`` allows ``min == max``) and honest — an
    empty-thickness scope contains no nodes, so the geometry disappears,
    which is precisely what the user just asked for by squashing it.
    Nor is it a dead end, thanks to two rules aimed squarely at that
    state: :func:`resolve_side` keeps both drag directions live, and
    the grip's box-independent size keeps a face with no area both
    drawn and hittable (:meth:`~ScopeGizmoGeometry.grip_quad`).
    """
    axis, _named = _FACE_SPEC[handle]
    is_max = resolve_side(geom, handle, value)
    limit = float((geom.lo if is_max else geom.hi)[axis])
    return max(float(value), limit) if is_max else min(float(value), limit)


def box_with_bound(
    geom: ScopeGizmoGeometry, handle: str, value: float,
) -> BBox:
    """``geom``'s box with the named face moved to a clamped ``value``."""
    lo = list(geom.lo)
    hi = list(geom.hi)
    axis, _named = _FACE_SPEC[handle]
    target = hi if resolve_side(geom, handle, value) else lo
    target[axis] = clamp_bound(geom, handle, value)
    return BBox(lo, hi)


# =====================================================================
# Layer construction
# =====================================================================


def _grips_layer(geom: ScopeGizmoGeometry) -> Any:
    """The six drawn grips — and, corner for corner, the grab surface.

    Six independent quads (24 points, no shared corners) so each grip
    stays a flat square in its own face's plane. Constant topology, so
    a drag takes ``update_layer``'s in-place path like the edges.
    """
    from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

    pts = np.vstack([geom.grip_quad(handle) for handle in FACE_HANDLES])
    quads = np.arange(4 * len(FACE_HANDLES)).reshape(-1, 4)
    return MeshLayer(
        layer_id=f"{SCOPE_GIZMO_LAYER_PREFIX}{geom.geometry_id}:grips",
        points=PointSet(pts),
        cells=CellBlocks({"quad": quads}),
        color=ColorSpec(mode="solid", solid_rgb=GRIP_RGB),
        opacity=GRIP_OPACITY,
        pickable=False,
        clip_exempt=True,
    )


def _edges_layer(geom: ScopeGizmoGeometry) -> Any:
    """The wireframe cage.

    Degenerates to nothing when the box collapses on all three axes —
    12 zero-length lines — which is exactly why the grips layer above
    is not sized from the box: at that point the six full-size grips
    are the ONLY thing drawn, and they are what makes the collapse
    visible and recoverable rather than an empty viewport (review
    finding F3).
    """
    from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

    return MeshLayer(
        layer_id=f"{SCOPE_GIZMO_LAYER_PREFIX}{geom.geometry_id}:edges",
        points=PointSet(geom.corners),
        cells=CellBlocks({"line": np.array(_BOX_EDGES)}),
        color=ColorSpec(mode="solid", solid_rgb=EDGE_RGB),
        line_width=2.0,
        pickable=False,
        clip_exempt=True,
    )


# =====================================================================
# Renderer
# =====================================================================


class ScopeGizmoRenderer:
    """Projects the ACTIVE geometry's scope box onto gizmo layers."""

    def __init__(
        self,
        backend: Any,
        controller: Any,
        *,
        bbox: Sequence[float],
        active_id: Callable[[], Optional[str]],
    ) -> None:
        self._backend = backend
        self._controller = controller
        self._bbox = tuple(float(c) for c in bbox)
        self._active_id = active_id
        # Strong refs, deliberately: the backend's clip registry holds
        # handles weakly, so whoever adds a layer keeps its handle alive.
        self._handles: "dict[str, tuple[Any, Any]]" = {}
        self._geoms: "dict[str, ScopeGizmoGeometry]" = {}

    def geometries(self) -> "dict[str, ScopeGizmoGeometry]":
        """Current gizmo geometry by geometry id — the hit-test surface.

        At most one entry (the active geometry), but kept a dict so the
        reconcile and teardown below read exactly like the clip
        renderer's.
        """
        return dict(self._geoms)

    def refresh(self) -> None:
        """Reconcile the gizmo layers against the controller's state.

        Same-topology updates take ``update_layer``'s in-place fast
        path, so a drag moves the existing actors and rebuilds nothing.
        """
        desired: "dict[str, ScopeGizmoGeometry]" = {}
        controller = self._controller
        if controller is not None and getattr(controller, "show_gizmo", True):
            try:
                gid = self._active_id()
            except Exception:
                gid = None
            box = controller.box_for(gid) if gid is not None else None
            if box is not None:
                desired[str(gid)] = scope_gizmo_geometry(
                    str(gid), box, self._bbox,
                )

        for gid in list(self._handles):
            if gid not in desired:
                for handle in self._handles.pop(gid):
                    try:
                        self._backend.remove_layer(handle)
                    except Exception:
                        pass

        for gid, geom in desired.items():
            grips = _grips_layer(geom)
            edges = _edges_layer(geom)
            try:
                if gid in self._handles:
                    h_grips, h_edges = self._handles[gid]
                    self._backend.update_layer(h_grips, grips)
                    self._backend.update_layer(h_edges, edges)
                else:
                    self._handles[gid] = (
                        self._backend.add_layer(grips),
                        self._backend.add_layer(edges),
                    )
            except Exception:
                # A gizmo that failed to render must not stay in the
                # hit-test surface claiming presses it cannot show.
                self._handles.pop(gid, None)

        self._geoms = {
            gid: geom
            for gid, geom in desired.items()
            if gid in self._handles
        }

    def clear(self) -> None:
        """Drop every gizmo layer. Idempotent."""
        for pair in self._handles.values():
            for handle in pair:
                try:
                    self._backend.remove_layer(handle)
                except Exception:
                    pass
        self._handles.clear()
        self._geoms.clear()


__all__ = [
    "DRAG_MIN_SEPARATION",
    "FACE_HANDLES",
    "GRIP_HALF_FRAC",
    "SCOPE_GIZMO_LAYER_PREFIX",
    "ScopeGizmoGeometry",
    "ScopeGizmoRenderer",
    "axis_param",
    "box_with_bound",
    "clamp_bound",
    "face_axis",
    "face_axis_dir",
    "in_plane_axes",
    "ray_hit",
    "rebased",
    "resolve_side",
    "scope_gizmo_geometry",
]
