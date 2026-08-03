"""ScopeGizmoInteractor — dragging the scope box's faces (catalog).

The third instance of the ADR 0081 L2 pattern, after ``_legend_interactor``
(THE reference) and ``_clip_gizmo_interactor``, and the simplest of the
three — see ``_scope_gizmo`` for why an axis-aligned box needs no
trackball:

* **Priority 12** — above navigation (11) and picking (10), installed
  **after** the legend and clip-gizmo interactors so either of those
  wins a literal overlap (VTK invokes same-priority observers in
  insertion order). The press observer hit-tests the cursor against the
  visible gizmo and **returns without aborting on a miss** — camera and
  pick behaviour outside a face is bit-for-bit what it was.
* **Hit test** — a world-space ray from the backend
  (``display_to_world_ray``, the pixel-unprojection service that keeps
  the camera math behind the ADR 0042 seam) against the six small
  GRIPS of the exact :class:`~._scope_gizmo.ScopeGizmoGeometry` the
  renderer drew. Grips, not faces: a handle the size of a face makes
  the enclosing box claim every press over the model and kills picking
  outright — see ``_scope_gizmo``.
* **Drag** — the mouse ray is projected onto the grabbed face's own
  world axis (:func:`~._clip_gizmo.axis_param`, with
  :data:`~._clip_gizmo.DRAG_MIN_SEPARATION` so the gain stays bounded
  as the camera swings face-on: measured at 1 degree off, one pixel
  moved a clip plane 21 % of the model). The bound delta is the
  difference from the press projection, clamped at the opposite face by
  :func:`~._scope_gizmo.box_with_bound` so the ``BBox`` it builds is
  valid by construction.
* **Leave ends the drag** — a release delivered outside the window
  never reaches us (0081 L2's measured defect class).

Every move that actually MOVES the box goes through
``ScopeController.set_scope`` — the single owner — and then announces it
exactly once through :attr:`~ScopeGizmoInteractor.on_changed`. That
callback is how the gesture reaches ``GEOMETRY_SCOPE_CHANGED``; unlike
the clip controller, ``ScopeController`` deliberately takes no
collaborators (it is what makes the scope free to scrub), so the fire is
wired at install by the viewer, the same way the settings panel wires
its own. This module holds no box state beyond the in-flight drag.

Two costs the gesture deliberately does not pay, both mirroring the
clip gizmo's ADR 0083 S3 arrangement:

* **No announcement for a move that changed nothing.** Qt delivers a
  ``MouseMoveEvent`` for sub-pixel motion, so a hand resting on a
  pressed button produces a stream of moves that rebuild the identical
  box — and a clamped face produces the same thing whenever the
  projection lands on the far side of its opposite. Firing for those
  would re-run the whole mask pass for a no-op.
* **The mask pass waits for the release.** The push half is
  :attr:`~ScopeGizmoInteractor.on_drag_end`, the pull half is
  :attr:`~ScopeGizmoInteractor.is_dragging` — a subscriber woken by a
  mid-drag ``GEOMETRY_SCOPE_CHANGED`` asks the latter whether the box
  has settled and skips the O(cells) recompute if it has not (measured
  at 28 ms per mouse-move on a 500k-cell grid, before rendering). The
  gizmo and the panel still follow every tick; only the mask waits.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ._clip_gizmo import DRAG_MIN_SEPARATION, axis_param
from ._scope_gizmo import box_with_bound, face_axis_dir, ray_hit, rebased


def _same_box(box: Any, other: Any) -> bool:
    """Whether two boxes describe the same bounds.

    ``BBox`` is ``eq=False`` (its fields are arrays, so the generated
    ``__eq__`` would raise on the ambiguous truth value), hence the
    explicit component compare rather than ``==``.
    """
    if other is None:
        return False
    return bool(
        np.array_equal(np.asarray(box.min), np.asarray(other.min))
        and np.array_equal(np.asarray(box.max), np.asarray(other.max))
    )


class ScopeGizmoInteractor:
    """Mouse gestures for the scope-box gizmo of one viewport."""

    def __init__(
        self,
        backend: Any,
        controller: Any,
        gizmos: Any,
        *,
        on_changed: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._backend = backend
        self._controller = controller
        self._gizmos = gizmos
        self._iren: Any = None
        self._tags: dict[str, int] = {}
        self._drag: Optional[dict] = None
        self._cursor: Optional[str] = None
        #: Called with the geometry id after every ``set_scope`` a drag
        #: performs. The owner mutation and its announcement stay one
        #: pair, so a move can never move the box without repainting.
        self.on_changed = on_changed
        #: Called with the geometry id once when a gesture ends
        #: (release, or a LeaveEvent standing in for a release outside
        #: the window). The scope MASK pass recomputes here rather than
        #: per tick, and the end of a gesture changes no box state, so
        #: it fires nothing of its own — hence this push. Deliberately a
        #: plain callback, not a dispatcher event: it announces a
        #: *gesture boundary*, not a state change (the clip gizmo's
        #: ``on_drag_end``, same reasoning).
        self.on_drag_end: Optional[Callable[[str], None]] = None

    @property
    def is_dragging(self) -> bool:
        """Whether a face drag is in flight (the pull half of the pair
        above).

        A subscriber woken by a mid-drag ``GEOMETRY_SCOPE_CHANGED`` asks
        here whether the box has settled, instead of sniffing private
        state — same contract as ``ClipGizmoInteractor.is_dragging``.
        """
        return self._drag is not None

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    def install(self) -> bool:
        """Attach the observers. ``False`` when there is no interactor
        or the backend cannot unproject a pixel."""
        if self._tags:
            return True
        if getattr(self._backend, "display_to_world_ray", None) is None:
            return False
        try:
            iren = self._backend.plotter.iren.interactor
        except Exception:
            return False
        if iren is None:
            return False
        self._iren = iren
        self._tags["press"] = iren.AddObserver(
            "LeftButtonPressEvent", self._on_lmb_press, 12.0,
        )
        self._tags["move"] = iren.AddObserver(
            "MouseMoveEvent", self._on_mouse_move, 12.0,
        )
        self._tags["release"] = iren.AddObserver(
            "LeftButtonReleaseEvent", self._on_lmb_release, 12.0,
        )
        # A release delivered outside the window never reaches us, and
        # a drag with no end follows the cursor forever afterwards.
        self._tags["leave"] = iren.AddObserver(
            "LeaveEvent", self._on_leave, 12.0,
        )
        return True

    def uninstall(self) -> None:
        """Remove every observer and withdraw our cursor. Idempotent.

        The cursor withdrawal is not optional (review finding F10): the
        arbiter holds a memo per requester, so an interactor that goes
        away mid-hover leaves a SIZEALL request standing that nothing
        can ever retract. Dropping ``_iren`` afterwards keeps the torn
        -down interactor from holding the window alive and makes a
        second :meth:`install` start clean.
        """
        self._set_cursor(None)
        if self._iren is not None:
            for tag in self._tags.values():
                try:
                    self._iren.RemoveObserver(tag)
                except Exception:
                    pass
        self._tags.clear()
        self._iren = None
        self._drag = None

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _ray(self, x: float, y: float) -> Optional[tuple]:
        try:
            return self._backend.display_to_world_ray(x, y)
        except Exception:
            return None

    def hit(
        self, x: float, y: float,
    ) -> "tuple[Optional[str], Optional[str]]":
        """``(geometry_id, handle)`` under the cursor, or ``(None, None)``.

        ``x`` / ``y`` are VTK event coordinates (pixels, bottom-left
        origin). The nearest face wins, with ties broken by declared
        handle order — see :func:`~._scope_gizmo.ray_hit`.
        """
        gid, handle, _ray = self._hit_with_ray(x, y)
        return gid, handle

    def _hit_with_ray(
        self, x: float, y: float,
    ) -> "tuple[Optional[str], Optional[str], Optional[tuple]]":
        """:meth:`hit` plus the ray it used (the press needs both)."""
        ray = self._ray(x, y)
        if ray is None:
            return None, None, None
        origin, direction = ray
        best: "Optional[tuple[float, str, str]]" = None
        for gid, geom in self._gizmos.geometries().items():
            result = ray_hit(geom, origin, direction)
            if result is None:
                continue
            handle, t = result
            if best is None or t < best[0]:
                best = (t, gid, handle)
        if best is None:
            return None, None, ray
        return best[1], best[2], ray

    # ------------------------------------------------------------------
    # Gesture handlers
    # ------------------------------------------------------------------

    def _on_lmb_press(self, caller: Any, _event: Any) -> None:
        # Shift+LMB is navigation's orbit gesture (priority 11); never
        # take it, even over a face.
        try:
            if caller.GetShiftKey():
                return
        except Exception:
            pass
        x, y = caller.GetEventPosition()
        gid, handle, ray = self._hit_with_ray(x, y)
        if gid is None or ray is None:
            return          # miss: no abort — pick and camera untouched
        geom = self._gizmos.geometries().get(gid)
        if geom is None:
            return
        origin, direction = ray
        axis_dir = face_axis_dir(handle)
        # Floored, like the clip gizmo's translate (ADR 0083 S2 / C1):
        # the projection is then defined for every camera angle, so a
        # press that hit a face always becomes a drag and is always
        # claimed instead of falling through to the picker near face-on.
        s0 = axis_param(
            geom.face_centre(handle), axis_dir, origin, direction,
            min_separation=DRAG_MIN_SEPARATION,
        )
        self._drag = {
            "geometry_id": gid,
            "handle": handle,
            # ONLY the axis anchor is frozen: re-deriving it from the
            # live box each move would let the axis point chase the face
            # it is measuring and drift over a long drag. The six BOUNDS
            # are deliberately NOT frozen — see ``rebased`` (F11).
            "geom": geom,
            "axis_point": geom.face_centre(handle),
            "axis_dir": axis_dir,
            "s0": s0,
            "bound0": geom.bound(handle),
        }
        self._abort(caller, "press")

    def _on_mouse_move(self, caller: Any, _event: Any) -> None:
        x, y = caller.GetEventPosition()
        if self._drag is None:
            self._update_cursor(x, y)
            return
        ray = self._ray(x, y)
        if ray is not None:
            origin, direction = ray
            d = self._drag
            s = axis_param(
                d["axis_point"], d["axis_dir"], origin, direction,
                min_separation=DRAG_MIN_SEPARATION,
            )
            # The bounds come off the OWNER, live, so an external write
            # landing mid-drag survives instead of being reverted by the
            # press-time snapshot (F11). The anchor above is still the
            # press's.
            live = self._current_box(d["geometry_id"])
            geom = d["geom"] if live is None else rebased(d["geom"], live)
            box = box_with_bound(
                geom, d["handle"], d["bound0"] + (s - d["s0"]),
            )
            # A move that rebuilds the identical box — an unchanged
            # pixel, or a projection the clamp pinned — would re-run the
            # mask over every geometry to change nothing, so the whole
            # no-op is dropped here (F2).
            if not _same_box(box, live):
                self._controller.set_scope(d["geometry_id"], box)
                self._announce(d["geometry_id"])
        self._abort(caller, "move")

    def _on_lmb_release(self, caller: Any, _event: Any) -> None:
        if self._drag is None:
            return
        self._end_drag()
        self._abort(caller, "release")

    def _on_leave(self, _caller: Any, _event: Any) -> None:
        """End any drag when the cursor leaves — the release may not come."""
        self._end_drag()
        self._set_cursor(None)

    def _end_drag(self) -> None:
        """Drop the in-flight gesture and announce the boundary once.

        The order matters: ``_drag`` is cleared FIRST, so that by the
        time the callback runs :attr:`is_dragging` already reads False
        and the deferred mask pass it triggers actually runs.
        """
        drag, self._drag = self._drag, None
        if drag is None or self.on_drag_end is None:
            return
        try:
            self.on_drag_end(drag["geometry_id"])
        except Exception:
            pass

    def _current_box(self, geometry_id: str) -> Any:
        try:
            return self._controller.box_for(geometry_id)
        except Exception:
            return None

    def _announce(self, geometry_id: str) -> None:
        if self.on_changed is None:
            return
        try:
            self.on_changed(geometry_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _update_cursor(self, x: float, y: float) -> None:
        _gid, handle = self.hit(x, y)
        self._set_cursor(handle)

    def _set_cursor(self, handle: "Optional[str]") -> None:
        """Ask the arbiter for this hover's cursor.

        Not a direct ``SetCurrentCursor``: three priority-12 interactors
        now hover on the same event and none of them aborts on a miss,
        so a direct write here would erase whatever the legend or the
        clip gizmo just set (ADR 0083 S2 finding C2).
        """
        if handle == self._cursor:
            return
        self._cursor = handle
        try:
            import vtk

            from ._cursor_arbiter import request_cursor

            request_cursor(
                self._backend.plotter.render_window,
                "scope_gizmo",
                None if handle is None else vtk.VTK_CURSOR_SIZEALL,
            )
        except Exception:
            pass

    def _abort(self, caller: Any, tag: str) -> None:
        """Claim the event so navigation and picking never see it."""
        try:
            cmd = caller.GetCommand(self._tags[tag])
            if cmd is not None:
                cmd.SetAbortFlag(1)
        except Exception:
            pass


def install_scope_gizmo_interactor(
    backend: Any,
    controller: Any,
    gizmos: Any,
    *,
    on_changed: Optional[Callable[[str], None]] = None,
) -> "Optional[ScopeGizmoInteractor]":
    """Install scope-box gestures on ``backend``'s interactor.

    Returns the interactor, or ``None`` for headless / offscreen
    backends that have no interactor to observe. Call **after**
    ``install_legend_interactor`` and ``install_clip_gizmo_interactor``
    — same priority, and insertion order is what resolves an overlap.
    """
    interactor = ScopeGizmoInteractor(
        backend, controller, gizmos, on_changed=on_changed,
    )
    return interactor if interactor.install() else None


__all__ = ["ScopeGizmoInteractor", "install_scope_gizmo_interactor"]
