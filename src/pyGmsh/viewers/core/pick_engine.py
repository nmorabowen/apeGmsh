"""
PickEngine — LMB click, drag (rubber-band), and hover picking.

Installs VTK observers for left-mouse-button events.  Resolves
picks through the :class:`EntityRegistry` and fires callbacks.
Does NOT modify selection state directly — the caller wires
callbacks to :class:`SelectionState`.

Usage::

    engine = PickEngine(plotter, registry)
    engine.on_pick = lambda dt, ctrl: selection.toggle(dt)
    engine.on_hover = lambda dt: color_mgr.set_entity_state(...)
    engine.on_box_select = lambda dts, ctrl: ...
    engine.install()
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from .entity_registry import DimTag, EntityRegistry


class PickEngine:
    """Pixel-precise picking via VTK cell pickers."""

    __slots__ = (
        "_plotter",
        "_registry",
        "_pickable_dims",
        "_hidden_check",
        "_drag_threshold",
        # Callbacks
        "on_pick",
        "on_hover",
        "on_box_select",
        # Internal state
        "_click_picker",
        "_hover_picker",
        "_hover_id",
        "_hover_throttle",
        "_press_pos",
        "_dragging",
        "_ctrl_held",
        "_rubberband_actor",
        "_rubberband_pts",
        "_tags",
    )

    def __init__(
        self,
        plotter: "pv.Plotter",
        registry: "EntityRegistry",
        *,
        drag_threshold: int = 8,
    ) -> None:
        self._plotter = plotter
        self._registry = registry
        self._pickable_dims: set[int] = set(registry.dims)
        self._hidden_check: Callable[["DimTag"], bool] = lambda _: False
        self._drag_threshold = drag_threshold

        self.on_pick: Callable[["DimTag", bool], None] | None = None
        self.on_hover: Callable[["DimTag | None"], None] | None = None
        self.on_box_select: Callable[[list["DimTag"], bool], None] | None = None

        self._click_picker = None
        self._hover_picker = None
        self._hover_id: "DimTag | None" = None
        self._hover_throttle: int = 0
        self._press_pos: tuple[int, int] | None = None
        self._dragging: bool = False
        self._ctrl_held: bool = False
        self._rubberband_actor = None
        self._rubberband_pts = None
        self._tags: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_pickable_dims(self, dims: set[int]) -> None:
        self._pickable_dims = set(dims)

    def set_hidden_check(self, fn: Callable[["DimTag"], bool]) -> None:
        """Set a function that returns True if an entity should be skipped."""
        self._hidden_check = fn

    @property
    def drag_threshold(self) -> int:
        return self._drag_threshold

    @drag_threshold.setter
    def drag_threshold(self, value: int) -> None:
        self._drag_threshold = max(2, value)

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install LMB press/move/release observers."""
        import vtk

        plotter = self._plotter
        iren_wrap = plotter.iren
        iren = iren_wrap.interactor
        renderer = plotter.renderer

        # Lazy-init pickers
        self._click_picker = vtk.vtkCellPicker()
        self._click_picker.SetTolerance(0.005)
        self._hover_picker = vtk.vtkCellPicker()
        self._hover_picker.SetTolerance(0.005)

        engine = self

        def _abort(caller, tag):
            cmd = caller.GetCommand(tag)
            if cmd is not None:
                cmd.SetAbortFlag(1)

        # ── rubberband helpers ──────────────────────────────────────
        def _ensure_rubberband():
            if engine._rubberband_pts is not None:
                return
            import vtk as _vtk
            pts = _vtk.vtkPoints()
            pts.SetNumberOfPoints(4)
            for i in range(4):
                pts.SetPoint(i, 0, 0, 0)
            lines = _vtk.vtkCellArray()
            lines.InsertNextCell(5)
            for i in [0, 1, 2, 3, 0]:
                lines.InsertCellPoint(i)
            poly = _vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(lines)
            mapper = _vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            actor = _vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            actor.GetProperty().SetLineWidth(1.5)
            actor.VisibilityOff()
            renderer.AddActor2D(actor)
            engine._rubberband_pts = pts
            engine._rubberband_actor = actor

        def _update_rubberband(x0, y0, x1, y1):
            _ensure_rubberband()
            pts = engine._rubberband_pts
            pts.SetPoint(0, x0, y0, 0)
            pts.SetPoint(1, x1, y0, 0)
            pts.SetPoint(2, x1, y1, 0)
            pts.SetPoint(3, x0, y1, 0)
            pts.Modified()
            # Change style for crossing (R→L) vs window (L→R)
            prop = engine._rubberband_actor.GetProperty()
            if x1 < x0:  # crossing
                prop.SetLineStipplePattern(0xAAAA)
            else:
                prop.SetLineStipplePattern(0xFFFF)
            engine._rubberband_actor.VisibilityOn()
            plotter.render()

        def _hide_rubberband():
            if engine._rubberband_actor is not None:
                engine._rubberband_actor.VisibilityOff()

        # ── LMB handlers ───────────────────────────────────────────
        def on_lmb_press(caller, _event):
            x, y = caller.GetEventPosition()
            engine._press_pos = (x, y)
            engine._dragging = False
            engine._ctrl_held = bool(caller.GetControlKey())
            _abort(caller, engine._tags["lmb_press"])

        def on_mouse_move(caller, _event):
            # LMB drag → rubberband
            if engine._press_pos is not None:
                px, py = caller.GetEventPosition()
                sx, sy = engine._press_pos
                if not engine._dragging:
                    dist2 = (px - sx) ** 2 + (py - sy) ** 2
                    if dist2 > engine._drag_threshold ** 2:
                        engine._dragging = True
                if engine._dragging:
                    _update_rubberband(sx, sy, px, py)
                _abort(caller, engine._tags["move"])
                return
            # Idle hover
            engine._hover_throttle = (engine._hover_throttle + 1) % 3
            if engine._hover_throttle != 0:
                return
            px, py = caller.GetEventPosition()
            engine._do_hover(px, py)
            # Don't abort — let navigation see move events

        def on_lmb_release(caller, _event):
            if engine._press_pos is None:
                return
            x0, y0 = engine._press_pos
            x1, y1 = caller.GetEventPosition()
            ctrl = engine._ctrl_held
            _hide_rubberband()

            if engine._dragging:
                # Box select
                engine._do_box(x0, y0, x1, y1, ctrl)
            else:
                # Click pick
                engine._do_click(x1, y1, ctrl)

            engine._press_pos = None
            engine._dragging = False
            engine._ctrl_held = False
            _abort(caller, engine._tags["lmb_release"])

        # ── register ───────────────────────────────────────────────
        self._tags["lmb_press"]   = iren.AddObserver("LeftButtonPressEvent",   on_lmb_press,   10.0)
        self._tags["move"]        = iren.AddObserver("MouseMoveEvent",         on_mouse_move,  9.0)
        self._tags["lmb_release"] = iren.AddObserver("LeftButtonReleaseEvent", on_lmb_release, 10.0)

    # ------------------------------------------------------------------
    # Pick logic
    # ------------------------------------------------------------------

    def _do_click(self, x: int, y: int, ctrl: bool) -> None:
        renderer = self._plotter.renderer
        self._click_picker.Pick(x, y, 0, renderer)
        prop = self._click_picker.GetViewProp()
        if prop is None:
            return
        cell_id = self._click_picker.GetCellId()
        dt = self._registry.resolve_pick(id(prop), cell_id)
        if dt is None:
            return
        if dt[0] not in self._pickable_dims:
            return
        if self._hidden_check(dt):
            return
        if self.on_pick is not None:
            self.on_pick(dt, ctrl)

    def _do_hover(self, x: int, y: int) -> None:
        renderer = self._plotter.renderer
        self._hover_picker.Pick(x, y, 0, renderer)
        prop = self._hover_picker.GetViewProp()

        new_dt: "DimTag | None" = None
        if prop is not None:
            cell_id = self._hover_picker.GetCellId()
            candidate = self._registry.resolve_pick(id(prop), cell_id)
            if candidate is not None:
                if (candidate[0] in self._pickable_dims
                        and not self._hidden_check(candidate)):
                    new_dt = candidate

        if new_dt == self._hover_id:
            return
        old_dt = self._hover_id
        self._hover_id = new_dt
        if self.on_hover is not None:
            self.on_hover(new_dt)

    def _do_box(self, x0: int, y0: int, x1: int, y1: int, ctrl: bool) -> None:
        """Box-select with proper window vs crossing modes.

        Left-to-right (window): entity must be **fully enclosed**.
        Right-to-left (crossing): entity must **overlap** the box.
        """
        # DPI scaling
        try:
            rw = self._plotter.render_window
            vw, vh = rw.GetSize()
            aw, ah = rw.GetActualSize()
            sx_ratio = aw / vw if vw else 1.0
            sy_ratio = ah / vh if vh else 1.0
        except Exception:
            sx_ratio = sy_ratio = 1.0

        crossing = x1 < x0  # R→L drag = crossing mode
        bx0 = min(x0, x1) * sx_ratio
        bx1 = max(x0, x1) * sx_ratio
        by0 = min(y0, y1) * sy_ratio
        by1 = max(y0, y1) * sy_ratio

        renderer = self._plotter.renderer
        hits: list["DimTag"] = []

        def _project(xyz):
            renderer.SetWorldPoint(xyz[0], xyz[1], xyz[2], 1.0)
            renderer.WorldToDisplay()
            return renderer.GetDisplayPoint()[:2]

        for dt in self._registry.all_entities():
            if dt[0] not in self._pickable_dims:
                continue
            if self._hidden_check(dt):
                continue

            bbox = self._registry.bbox(dt)
            if bbox is not None:
                # Project all 8 AABB corners to screen
                inside = []
                for corner in bbox:
                    sx, sy = _project(corner)
                    inside.append(bx0 <= sx <= bx1 and by0 <= sy <= by1)

                if crossing:
                    # Crossing: ANY corner inside → selected
                    hit = any(inside)
                else:
                    # Window: ALL corners inside → selected
                    hit = all(inside)
            else:
                # Fallback: centroid only
                xyz = self._registry.centroid(dt)
                if xyz is None:
                    continue
                sx, sy = _project(xyz)
                hit = bx0 <= sx <= bx1 and by0 <= sy <= by1

            if hit:
                hits.append(dt)

        if hits and self.on_box_select is not None:
            self.on_box_select(hits, ctrl)

    @property
    def hover_entity(self) -> "DimTag | None":
        return self._hover_id
