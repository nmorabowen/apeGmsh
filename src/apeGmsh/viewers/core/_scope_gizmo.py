"""ScopeGizmoRenderer — the in-viewport scope box (catalog).

The drag half of the scope box (``core/scope_controller``): a wireframe
box with six translucent **face handles**, drawn for the ACTIVE geometry
only. Grab a face, slide it along its own world axis, and that one bound
moves.

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
* ``pickable=False`` — the faces span the whole box; a pickable face
  would eat every node/element pick inside it. The gestures live in
  ``_scope_gizmo_interactor`` at priority 12 instead.

The gizmo is also immune to the scope filter it drives, and for free:
the filter is an ``ElementVisibility`` layer on a *geometry's* scene
grid, and these layers are backend layers of their own that belong to
no scene.

Both layers keep a constant topology (8 points; 6 quads / 12 lines), so
a drag re-uses the actors through ``update_layer``'s in-place fast path
and rebuilds nothing — the 0081 L2 flicker lesson.

Like ``ClipGizmoRenderer`` this is a **projection** of controller state:
it reads ``show_gizmo`` and ``box_for(active)`` on :meth:`refresh` and
never mutates anything.

**Active geometry only.** ADR 0058 makes "active" the editing target,
and one box per scoped geometry would be a pile of overlapping faces
with no way to say which one you meant to grab.
"""
from __future__ import annotations

from dataclasses import dataclass
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

#: Floor on a face's in-plane grab half-extent, as a fraction of the
#: REFERENCE model diagonal — the same shape as the clip gizmo's
#: ``MIN_HALF_FRAC``, and here it is what keeps a degenerate box out of
#: a trap. A box collapsed on two axes has no face with any area, so
#: without a floor there would be nothing left to grab and the only way
#: back would be the panel. See :func:`ray_hit`.
GRAB_MIN_FRAC = 0.02

FACE_RGB = (0.35, 0.80, 0.55)
FACE_OPACITY = 0.16
EDGE_RGB = (0.20, 0.90, 0.55)

_CORNER_SIGNS = np.array(
    [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ],
    dtype=bool,
)
#: Corner indices of each face, in ``FACE_HANDLES`` order, wound as a
#: cycle (indices match :attr:`BBox.corners8`, which is z-slowest).
_FACE_QUADS = (
    (0, 2, 6, 4), (1, 3, 7, 5),
    (0, 1, 5, 4), (2, 3, 7, 6),
    (0, 1, 3, 2), (4, 5, 7, 6),
)
_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class ScopeGizmoGeometry:
    """One scope box's world-space geometry.

    Shared by the layer builder and the interactor's hit test, so what
    you grab is what you see — with one documented exception,
    :attr:`grab_floor`, which only ever widens a face that has collapsed
    to no area at all.
    """

    geometry_id: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    #: Minimum in-plane grab half-extent (see :data:`GRAB_MIN_FRAC`).
    grab_floor: float

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

    @property
    def box(self) -> BBox:
        """The box as the canonical value type (ADR 0045 INV-2)."""
        return BBox(self.lo, self.hi)


def face_axis(handle: str) -> int:
    """The world axis index (0/1/2) the named face slides along."""
    return _FACE_SPEC[handle][0]


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
    :attr:`~ScopeGizmoGeometry.grab_floor`, so the grab slack is a
    fraction of the MODEL rather than of a box the user may have
    collapsed to nothing.
    """
    lo = np.asarray(bbox[:3], dtype=float)
    hi = np.asarray(bbox[3:], dtype=float)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    return ScopeGizmoGeometry(
        geometry_id=str(geometry_id),
        lo=tuple(float(c) for c in box.min),
        hi=tuple(float(c) for c in box.max),
        grab_floor=GRAB_MIN_FRAC * diag,
    )


# =====================================================================
# Ray hit-testing (world space — the interactor's half of the deal)
# =====================================================================


def ray_hit(
    geom: ScopeGizmoGeometry,
    origin: Sequence[float],
    direction: Sequence[float],
) -> "Optional[tuple[str, float]]":
    """``(handle, ray_param)`` where the ray grabs a face, or ``None``.

    Six axis-aligned rectangles, tested independently; the **nearest**
    hit wins, so the face you can see is the face you get rather than
    the one behind it. Two determinism rules, both of which a box
    viewed from a corner exercises constantly:

    * A face **edge-on** to the ray (``|d[axis]|`` at the epsilon) is
      not hit at all. It is a zero-pixel target, and admitting it would
      make the winner depend on floating-point noise in a denominator
      that is about to divide.
    * On an exact tie — the ray through a shared edge or corner hits
      two or three faces at the same distance — the earlier handle in
      :data:`FACE_HANDLES` wins, because the loop only replaces the
      incumbent on a *strictly* nearer hit.

    The in-plane containment test is widened to
    :attr:`~ScopeGizmoGeometry.grab_floor` on each side. For any box
    with real extent that floor is far below the face's own half-extent
    and changes nothing; it exists for the degenerate case, where it is
    the difference between a box the user can drag back open and one
    they can only fix from the panel (see :func:`clamp_bound`).
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    mag = float(np.linalg.norm(d))
    if mag <= 0.0:
        return None
    d = d / mag
    lo = np.asarray(geom.lo, dtype=float)
    hi = np.asarray(geom.hi, dtype=float)
    floor = float(geom.grab_floor)

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
        other = [a for a in (0, 1, 2) if a != axis]
        half = np.maximum((hi[other] - lo[other]) / 2.0, floor)
        mid = (hi[other] + lo[other]) / 2.0
        if bool(np.all(np.abs(point[other] - mid) <= half)):
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
    :attr:`~ScopeGizmoGeometry.grab_floor` keeps a face with no area
    hittable in :func:`ray_hit`.
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


def _faces_layer(geom: ScopeGizmoGeometry) -> Any:
    from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

    return MeshLayer(
        layer_id=f"{SCOPE_GIZMO_LAYER_PREFIX}{geom.geometry_id}:faces",
        points=PointSet(geom.corners),
        cells=CellBlocks({"quad": np.array(_FACE_QUADS)}),
        color=ColorSpec(mode="solid", solid_rgb=FACE_RGB),
        opacity=FACE_OPACITY,
        pickable=False,
        clip_exempt=True,
    )


def _edges_layer(geom: ScopeGizmoGeometry) -> Any:
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
            faces = _faces_layer(geom)
            edges = _edges_layer(geom)
            try:
                if gid in self._handles:
                    h_faces, h_edges = self._handles[gid]
                    self._backend.update_layer(h_faces, faces)
                    self._backend.update_layer(h_edges, edges)
                else:
                    self._handles[gid] = (
                        self._backend.add_layer(faces),
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
    "GRAB_MIN_FRAC",
    "SCOPE_GIZMO_LAYER_PREFIX",
    "ScopeGizmoGeometry",
    "ScopeGizmoRenderer",
    "axis_param",
    "box_with_bound",
    "clamp_bound",
    "face_axis",
    "face_axis_dir",
    "ray_hit",
    "resolve_side",
    "scope_gizmo_geometry",
]
