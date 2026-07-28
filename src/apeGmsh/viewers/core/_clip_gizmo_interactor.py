"""ClipGizmoInteractor — dragging and rotating a section plane (ADR 0083 S2).

The second instance of the ADR 0081 L2 pattern (after
``_legend_interactor``, which is THE reference — same priority, same
abort discipline, same leave-event hygiene):

* **Priority 12** — above navigation (11) and picking (10), installed
  **after** the legend interactor so a legend overlapping a gizmo wins
  the press (VTK invokes same-priority observers in insertion order).
  The press observer hit-tests the cursor against the visible gizmos
  and **returns without aborting on a miss** — camera and pick
  behaviour outside a gizmo is bit-for-bit what it was.
* **Hit test** — a world-space ray from the backend
  (``display_to_world_ray``, the pixel-unprojection service that keeps
  the camera math behind the seam) against the exact
  :class:`~._clip_gizmo.GizmoGeometry` the renderer drew: quad and
  arrow shaft grab a **translate**, the cone tip grabs a **rotate**.
* **Translate** — the mouse ray is projected onto the plane's normal
  axis (:func:`~._clip_gizmo.axis_param`); the offset delta is the
  difference from the press projection. One ``set_offset`` per move.
* **Rotate** — a virtual trackball around the quad centre
  (:func:`~._clip_gizmo.sphere_point`): the tip follows the cursor,
  and ``set_pose`` re-anchors the offset so the plane pivots about the
  centre instead of swinging about the world origin.
* **Leave ends the drag** — a release delivered outside the window
  never reaches us (0081 L2's measured defect class).

Every gesture goes through a ``ClipPlaneSetController`` mutator, so
the owner stays single and fires ``CLIP_PLANES_CHANGED`` itself
(ADR 0056 Part 2); the renderer and the panel repaint as subscribers.
This module holds no plane state beyond the in-flight drag.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._clip_gizmo import (
    DRAG_MIN_SEPARATION,
    MODE_ROTATE,
    MODE_TRANSLATE,
    axis_param,
    ray_hit,
    sphere_point,
)


class ClipGizmoInteractor:
    """Mouse gestures for the section-plane gizmos of one viewport."""

    def __init__(self, backend: Any, controller: Any, gizmos: Any) -> None:
        self._backend = backend
        self._controller = controller
        self._gizmos = gizmos
        self._iren: Any = None
        self._tags: dict[str, int] = {}
        self._drag: Optional[dict] = None
        self._cursor: Optional[str] = None

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
        """Remove every observer. Idempotent."""
        if self._iren is not None:
            for tag in self._tags.values():
                try:
                    self._iren.RemoveObserver(tag)
                except Exception:
                    pass
        self._tags.clear()
        self._drag = None

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _ray(self, x: float, y: float) -> Optional[tuple]:
        try:
            return self._backend.display_to_world_ray(x, y)
        except Exception:
            return None

    def hit(self, x: float, y: float) -> "tuple[Optional[str], Optional[str]]":
        """``(plane_id, mode)`` under the cursor, or ``(None, None)``.

        ``x`` / ``y`` are VTK event coordinates (pixels, bottom-left
        origin). Across gizmos the nearest hit along the ray wins; a
        tip (rotate) beats its own quad regardless of depth — see
        :func:`~._clip_gizmo.ray_hit`.
        """
        plane_id, mode, _ray = self._hit_with_ray(x, y)
        return plane_id, mode

    def _hit_with_ray(
        self, x: float, y: float,
    ) -> "tuple[Optional[str], Optional[str], Optional[tuple]]":
        """:meth:`hit` plus the ray it used (the press needs both)."""
        ray = self._ray(x, y)
        if ray is None:
            return None, None, None
        origin, direction = ray
        best: Optional[tuple[float, str, str]] = None
        for plane_id, geom in self._gizmos.geometries().items():
            result = ray_hit(geom, origin, direction)
            if result is None:
                continue
            mode, t = result
            if best is None or t < best[0]:
                best = (t, plane_id, mode)
        if best is None:
            return None, None, ray
        return best[1], best[2], ray

    # ------------------------------------------------------------------
    # Gesture handlers
    # ------------------------------------------------------------------

    def _on_lmb_press(self, caller: Any, _event: Any) -> None:
        # Shift+LMB is navigation's orbit gesture (priority 11); never
        # take it, even over a gizmo.
        try:
            if caller.GetShiftKey():
                return
        except Exception:
            pass
        x, y = caller.GetEventPosition()
        plane_id, mode, ray = self._hit_with_ray(x, y)
        if plane_id is None or ray is None:
            return          # miss: no abort — pick and camera are untouched
        origin, direction = ray
        plane = self._controller.plane(plane_id)
        geom = self._gizmos.geometries().get(plane_id)
        if plane is None or geom is None:
            return
        if mode == MODE_ROTATE:
            self._drag = {
                "mode": MODE_ROTATE,
                "plane_id": plane_id,
                # The pivot is frozen at press: re-deriving it per move
                # from the live plane would let numeric drift walk the
                # plane sideways over a long drag.
                "centre": geom.centre,
                "radius": geom.arrow_len,
                "flipped": bool(plane.flipped),
            }
        else:
            # Floored: the projection is defined for every camera angle,
            # so a press that hit a gizmo always becomes a drag and is
            # always claimed — near face-on it used to bail here without
            # aborting, handing the click to the picker instead (C1).
            s0 = axis_param(
                geom.centre, geom.normal, origin, direction,
                min_separation=DRAG_MIN_SEPARATION,
            )
            self._drag = {
                "mode": MODE_TRANSLATE,
                "plane_id": plane_id,
                "axis_point": geom.centre,
                "axis_dir": geom.normal,
                "s0": s0,
                "offset0": float(plane.offset),
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
            if d["mode"] == MODE_TRANSLATE:
                s = axis_param(
                    d["axis_point"], d["axis_dir"], origin, direction,
                    min_separation=DRAG_MIN_SEPARATION,
                )
                self._controller.set_offset(
                    d["plane_id"], d["offset0"] + (s - d["s0"]),
                )
            else:
                point = sphere_point(
                    d["centre"], d["radius"], origin, direction,
                )
                if point is not None:
                    vec = point - np.asarray(d["centre"], dtype=float)
                    mag = float(np.linalg.norm(vec))
                    if mag > 1e-12:
                        n_eff = vec / mag
                        # The arrow points at the surviving side; fold
                        # the flip back out to get the stored normal.
                        stored = -n_eff if d["flipped"] else n_eff
                        self._controller.set_pose(
                            d["plane_id"],
                            tuple(stored),
                            float(np.asarray(d["centre"]) @ stored),
                        )
        self._abort(caller, "move")

    def _on_lmb_release(self, caller: Any, _event: Any) -> None:
        if self._drag is None:
            return
        self._drag = None
        self._abort(caller, "release")

    def _on_leave(self, _caller: Any, _event: Any) -> None:
        """End any drag when the cursor leaves — the release may not come."""
        self._drag = None
        self._set_cursor(None)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _update_cursor(self, x: float, y: float) -> None:
        _plane_id, mode = self.hit(x, y)
        self._set_cursor(mode)

    def _set_cursor(self, mode: "Optional[str]") -> None:
        """Ask the arbiter for this hover's cursor.

        Not a direct ``SetCurrentCursor``: the legend interactor hovers
        on the same events and neither of us aborts on a miss, so a
        direct write here erased the cursor the legend had just set
        (S2 review finding C2). The arbiter holds the memo and resolves
        the overlap by registration order.
        """
        if mode == self._cursor:
            return
        self._cursor = mode
        try:
            import vtk

            from ._cursor_arbiter import request_cursor

            request_cursor(
                self._backend.plotter.render_window,
                "clip_gizmo",
                {
                    MODE_TRANSLATE: vtk.VTK_CURSOR_SIZEALL,
                    MODE_ROTATE: vtk.VTK_CURSOR_HAND,
                }.get(mode),
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


def install_clip_gizmo_interactor(
    backend: Any, controller: Any, gizmos: Any,
) -> "Optional[ClipGizmoInteractor]":
    """Install gizmo gestures on ``backend``'s interactor.

    Returns the interactor, or ``None`` for headless / offscreen
    backends that have no interactor to observe. Call **after**
    ``install_legend_interactor`` — same priority, and insertion order
    is what lets legends win a literal overlap (ADR 0083 Consequences).
    """
    interactor = ClipGizmoInteractor(backend, controller, gizmos)
    return interactor if interactor.install() else None


__all__ = ["ClipGizmoInteractor", "install_clip_gizmo_interactor"]
