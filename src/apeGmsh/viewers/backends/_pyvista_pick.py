"""``PickBackend`` over a pyvista plotter — shared by both desktop viewers.

The pick-side counterpart of :class:`PyVistaBackend
<apeGmsh.viewers.backends.pyvista_qt.PyVistaBackend>`, implementing the
:class:`PickBackend <apeGmsh.viewers.scene_ir._pick.PickBackend>` Protocol
(ADR 0047, Phase R-D).  It owns *all* the VTK that the two legacy pick
engines (``core/pick_engine.py`` mesh, ``core/results_pick.py`` results)
each used to construct independently:

* the ``vtkCellPicker`` ray-cast (the stateless :meth:`resolve_pick` core,
  reused verbatim by the web request/response face in R-D.3),
* the world→display :meth:`project_points` projection (the shared box-pick
  core both engines already called via ``_project_points_to_display``),
* the screen-box→world :meth:`frustum_planes` un-projection,
* the desktop event face: the LMB press/move/release state machine, the
  rubber-band overlay, and the priority-10 abort chain shared with
  navigation (priority 11).

What stays in the domain (ADR 0047 INV-3): mode routing, ``prop_id`` →
entity resolution (``EntityRegistry`` / the results inventory), box
*candidate* sourcing, highlight, and hover *dedup-by-entity*.  The
backend resolves only geometry and reports a :class:`PickHit` / a
:class:`BoxGesture`; the domain interprets it.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..core.frustum import frustum_planes as _frustum_planes_from_corners
from ..scene_ir import BoxGesture, PickHit, PickModifiers, PickRequest
from ..scene_ir._pick import OnBox, OnHover, OnPick

_PICK_TOLERANCE = 0.005
_HOVER_THROTTLE = 3  # process 1 of every N idle move events


def _project_points_to_display(points: "np.ndarray", renderer: Any) -> "np.ndarray":
    """Project ``(N, 3)`` world points to ``(N, 2)`` display pixels.

    Vectorized via the camera's composite projection matrix (~40x faster
    than per-point ``WorldToDisplay`` at 100k pts), falling back to the
    per-point loop when the renderer doesn't expose the camera-matrix API
    (test stubs).  Relocated verbatim from ``core/results_pick.py`` — the
    canonical home is now the backend so both engines share one copy.
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    try:
        cam = renderer.GetActiveCamera()
        aspect = renderer.GetTiledAspectRatio()
        M = cam.GetCompositeProjectionTransformMatrix(aspect, 0.0, 1.0)
        size = renderer.GetSize()
        vp = renderer.GetViewport()
    except AttributeError:
        return _project_points_to_display_loop(points, renderer)

    arr = np.empty((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            arr[i, j] = M.GetElement(i, j)
    homog = np.empty((n, 4), dtype=np.float64)
    homog[:, :3] = points
    homog[:, 3] = 1.0
    clip = homog @ arr.T
    w = clip[:, 3:4]
    ndc = clip[:, :3] / np.where(w == 0.0, 1.0, w)
    win_w, win_h = size[0], size[1]
    vp_x0 = vp[0] * win_w
    vp_y0 = vp[1] * win_h
    vp_w = (vp[2] - vp[0]) * win_w
    vp_h = (vp[3] - vp[1]) * win_h
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = vp_x0 + (ndc[:, 0] * 0.5 + 0.5) * vp_w
    out[:, 1] = vp_y0 + (ndc[:, 1] * 0.5 + 0.5) * vp_h
    return out


def _project_points_to_display_loop(points: "np.ndarray", renderer: Any) -> "np.ndarray":
    """Per-point ``WorldToDisplay`` fallback. Kept for stub renderers."""
    out = np.empty((points.shape[0], 2), dtype=np.float64)
    for i in range(points.shape[0]):
        renderer.SetWorldPoint(
            float(points[i, 0]), float(points[i, 1]), float(points[i, 2]), 1.0
        )
        renderer.WorldToDisplay()
        d = renderer.GetDisplayPoint()
        out[i, 0] = d[0]
        out[i, 1] = d[1]
    return out


class PyVistaPickBackend:
    """Pixel-precise picking over a pyvista plotter (ADR 0047 ``PickBackend``).

    Constructed lazily by :meth:`PyVistaBackend.picking`.  Stateless core
    (:meth:`resolve_pick` / :meth:`project_points` / :meth:`frustum_planes`)
    is usable without :meth:`install`; the desktop event face layers the
    gesture machine over it.
    """

    def __init__(self, plotter: Any) -> None:
        self._plotter = plotter
        self._iren: Any = None
        self._click_picker: Any = None
        self._hover_picker: Any = None
        # Callbacks (set by install)
        self._on_pick: Optional[OnPick] = None
        self._on_hover: Optional[OnHover] = None
        self._on_box: Optional[OnBox] = None
        # Gesture state
        self._press_pos: Optional[tuple[int, int]] = None
        self._dragging = False
        self._mods = PickModifiers()
        self._hover_throttle = 0
        # Rubber-band overlay
        self._rubberband_actor: Any = None
        self._rubberband_pts: Any = None
        self._tags: dict[str, int] = {}

    # -- stateless geometric core -------------------------------------

    def resolve_pick(self, request: "PickRequest") -> "Optional[PickHit]":
        if self._click_picker is None:
            self._click_picker = self._new_picker()
        return self._resolve(self._click_picker, request.x, request.y)

    def project_points(self, world: "np.ndarray") -> "np.ndarray":
        return _project_points_to_display(np.asarray(world), self._plotter.renderer)

    def frustum_planes(
        self, box: "tuple[int, int, int, int]"
    ) -> "Optional[np.ndarray]":
        """Un-project the display box to 6 inward world frustum planes.

        Honours ``APEGMSH_BOX_2D=1`` (force the 2-D project fallback —
        returns ``None``).  The four box corners are un-projected at near
        (z=0) and far (z=1) clip depths in CCW order and handed to
        ``frustum_planes``.  Returns ``None`` when the camera can't
        un-project."""
        if os.environ.get("APEGMSH_BOX_2D"):
            return None
        renderer = getattr(self._plotter, "renderer", None)
        if renderer is None:
            return None
        x0, y0, x1, y1 = box
        bx0, bx1 = (x0, x1) if x0 <= x1 else (x1, x0)
        by0, by1 = (y0, y1) if y0 <= y1 else (y1, y0)
        corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        try:
            near, far = [], []
            for (x, y) in corners:
                n = self._unproject(renderer, x, y, 0.0)
                f = self._unproject(renderer, x, y, 1.0)
                if n is None or f is None:
                    return None
                near.append(n)
                far.append(f)
            return _frustum_planes_from_corners(np.array(near), np.array(far))
        except Exception:
            return None

    # -- desktop event face -------------------------------------------

    def install(
        self,
        *,
        on_pick: "OnPick",
        on_hover: "Optional[OnHover]" = None,
        on_box: "Optional[OnBox]" = None,
    ) -> None:
        self._on_pick = on_pick
        self._on_hover = on_hover
        self._on_box = on_box

        iren = self._plotter.iren.interactor
        self._iren = iren
        self._click_picker = self._new_picker()
        self._hover_picker = self._new_picker()

        self._tags["lmb_press"] = iren.AddObserver(
            "LeftButtonPressEvent", self._on_lmb_press, 10.0
        )
        self._tags["move"] = iren.AddObserver(
            "MouseMoveEvent", self._on_mouse_move, 9.0
        )
        self._tags["lmb_release"] = iren.AddObserver(
            "LeftButtonReleaseEvent", self._on_lmb_release, 10.0
        )

    def uninstall(self) -> None:
        """Remove observers + the rubber-band actor. Idempotent.

        Closes the observer leak both legacy engines carried (neither had
        a teardown path)."""
        iren = self._iren
        if iren is not None:
            for tag in self._tags.values():
                try:
                    iren.RemoveObserver(tag)
                except Exception:
                    pass
        self._tags.clear()
        if self._rubberband_actor is not None:
            try:
                self._plotter.renderer.RemoveActor2D(self._rubberband_actor)
            except Exception:
                pass
            self._rubberband_actor = None
            self._rubberband_pts = None
        self._press_pos = None
        self._dragging = False

    # -- internals ----------------------------------------------------

    def _new_picker(self) -> Any:
        import vtk

        p = vtk.vtkCellPicker()
        p.SetTolerance(_PICK_TOLERANCE)
        return p

    def _resolve(self, picker: Any, x: int, y: int) -> "Optional[PickHit]":
        renderer = self._plotter.renderer
        picker.Pick(x, y, 0, renderer)
        prop = picker.GetViewProp()
        if prop is None:
            return None
        world = picker.GetPickPosition()
        return PickHit(
            world=(float(world[0]), float(world[1]), float(world[2])),
            cell_id=int(picker.GetCellId()),
            prop_id=id(prop),
        )

    @staticmethod
    def _unproject(renderer: Any, x: float, y: float, z: float):
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        wp = renderer.GetWorldPoint()
        w = wp[3]
        if w == 0.0:
            return None
        return (wp[0] / w, wp[1] / w, wp[2] / w)

    @staticmethod
    def _abort(caller: Any, tag: int) -> None:
        cmd = caller.GetCommand(tag)
        if cmd is not None:
            cmd.SetAbortFlag(1)

    # ── gesture handlers ─────────────────────────────────────────────

    def _on_lmb_press(self, caller: Any, _event: Any) -> None:
        # Shift+LMB is owned by navigation (priority 11) — bail so the
        # rubber-band path doesn't fight the rotate gesture.
        try:
            if caller.GetShiftKey():
                return
        except Exception:
            pass
        x, y = caller.GetEventPosition()
        self._press_pos = (x, y)
        self._dragging = False
        self._mods = PickModifiers(
            ctrl=bool(caller.GetControlKey()),
            alt=bool(caller.GetAltKey()),
        )
        self._abort(caller, self._tags["lmb_press"])

    def _on_mouse_move(self, caller: Any, _event: Any) -> None:
        if self._press_pos is not None:
            px, py = caller.GetEventPosition()
            sx, sy = self._press_pos
            if not self._dragging:
                dist2 = (px - sx) ** 2 + (py - sy) ** 2
                if dist2 > 8 ** 2:
                    self._dragging = True
            if self._dragging:
                self._update_rubberband(sx, sy, px, py)
            self._abort(caller, self._tags["move"])
            return
        # Idle hover — throttled; no abort (navigation still sees moves).
        self._hover_throttle = (self._hover_throttle + 1) % _HOVER_THROTTLE
        if self._hover_throttle != 0 or self._on_hover is None:
            return
        px, py = caller.GetEventPosition()
        self._on_hover(self._resolve(self._hover_picker, px, py))

    def _on_lmb_release(self, caller: Any, _event: Any) -> None:
        if self._press_pos is None:
            return
        x0, y0 = self._press_pos
        x1, y1 = caller.GetEventPosition()
        mods = self._mods
        self._hide_rubberband()
        if self._dragging:
            if self._on_box is not None:
                self._on_box(
                    BoxGesture(
                        box=(x0, y0, x1, y1),
                        crossing=x1 < x0,
                        modifiers=mods,
                    )
                )
        else:
            if self._on_pick is not None:
                self._on_pick(self._resolve(self._click_picker, x1, y1), mods)
        self._press_pos = None
        self._dragging = False
        self._abort(caller, self._tags["lmb_release"])

    # ── rubber-band overlay ──────────────────────────────────────────

    def _ensure_rubberband(self) -> None:
        if self._rubberband_pts is not None:
            return
        import vtk

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(4)
        for i in range(4):
            pts.SetPoint(i, 0, 0, 0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(5)
        for i in [0, 1, 2, 3, 0]:
            lines.InsertCellPoint(i)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetLineWidth(1.5)
        actor.VisibilityOff()
        self._plotter.renderer.AddActor2D(actor)
        self._rubberband_pts = pts
        self._rubberband_actor = actor

    def _update_rubberband(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._ensure_rubberband()
        pts = self._rubberband_pts
        pts.SetPoint(0, x0, y0, 0)
        pts.SetPoint(1, x1, y0, 0)
        pts.SetPoint(2, x1, y1, 0)
        pts.SetPoint(3, x0, y1, 0)
        pts.Modified()
        prop = self._rubberband_actor.GetProperty()
        prop.SetLineStipplePattern(0xAAAA if x1 < x0 else 0xFFFF)
        self._rubberband_actor.VisibilityOn()
        self._plotter.render()

    def _hide_rubberband(self) -> None:
        if self._rubberband_actor is not None:
            self._rubberband_actor.VisibilityOff()
