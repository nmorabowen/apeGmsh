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
    from apeGmsh._types import DimTag
    from .entity_registry import EntityRegistry


def _entity_in_box(
    sx: np.ndarray,
    sy: np.ndarray,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    crossing: bool,
) -> bool:
    """Box-select containment predicate for one entity's projected points.

    Window mode:   entity's projected 2D AABB is fully inside the box.
    Crossing mode: any sampled point is inside the box (classic).

    Crossing uses sample-point-inside rather than AABB overlap because
    ``bbox()`` for curves/surfaces returns the 3D AABB's 8 corners,
    whose projected 2D AABB can be much larger than the visible
    silhouette at angled views — pure AABB overlap over-selected.
    The sample-density workaround (64 points per volume) makes the
    classic test tight enough in practice.
    """
    if crossing:
        inside = (bx0 <= sx) & (sx <= bx1) & (by0 <= sy) & (sy <= by1)
        return bool(np.any(inside))
    return bool(
        bx0 <= sx.min() and sx.max() <= bx1
        and by0 <= sy.min() and sy.max() <= by1
    )


class PickEngine:
    """Pixel-precise picking via VTK cell pickers."""

    # Type declarations for __slots__ (consumed by mypy / pyright).
    # VTK pickers and the rubber-band actor/pts are lazy-created by
    # :meth:`install` — they stay None until then.  Typed as ``Any``
    # here so handlers (which only run after install) can call
    # ``.Pick(...)``, ``.SetTolerance(...)`` etc. without assertions.
    _plotter: Any
    _registry: EntityRegistry
    _pickable_dims: set[int]
    _hidden_check: Callable[["DimTag"], bool]
    _drag_threshold: int
    on_pick: Callable[["DimTag", bool], None] | None
    on_hover: Callable[["DimTag | None"], None] | None
    on_box_select: Callable[[list["DimTag"], bool], None] | None
    _click_picker: Any
    _hover_picker: Any
    _hover_id: "DimTag | None"
    _hover_throttle: int
    _press_pos: tuple[int, int] | None
    _dragging: bool
    _ctrl_held: bool
    _rubberband_actor: Any
    _rubberband_pts: Any
    _tags: dict[str, int]

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
            # Change style for crossing (R->L) vs window (L->R)
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
            # Shift+LMB is owned by navigation (drag → rotate, click
            # → on_shift_click). Bail early so the rubber-band path
            # doesn't fight the rotate gesture and so the priority-11
            # navigation observer sees a clean abort chain.
            try:
                if caller.GetShiftKey():
                    return
            except Exception:
                pass
            x, y = caller.GetEventPosition()
            engine._press_pos = (x, y)
            engine._dragging = False
            engine._ctrl_held = bool(caller.GetControlKey())
            _abort(caller, engine._tags["lmb_press"])

        def on_mouse_move(caller, _event):
            # LMB drag -> rubberband
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
        self._hover_id = new_dt
        if self.on_hover is not None:
            self.on_hover(new_dt)

    def _do_box(self, x0: int, y0: int, x1: int, y1: int, ctrl: bool) -> None:
        """Box-select with proper window vs crossing modes.

        Event coordinates from VTK on this build live in the same
        display space as ``renderer.WorldToDisplay`` output and as
        ``vtkActor2D`` (which is what the rubber-band uses — and the
        rubber-band draws correctly). No DPI scaling is applied here;
        if a future build diverges these spaces, the fix is to scale
        click + hover + rubber-band consistently, not to re-introduce
        scaling only in this method.

        Set env var ``APEGMSH_DEBUG_BOX=1`` to log per-entity projection
        and hit results for diagnosing crossing-mode misses.
        """
        import os
        _debug = bool(os.environ.get("APEGMSH_DEBUG_BOX"))

        crossing = x1 < x0
        bx0 = min(x0, x1)
        bx1 = max(x0, x1)
        by0 = min(y0, y1)
        by1 = max(y0, y1)

        renderer = self._plotter.renderer

        if _debug:
            try:
                rw = self._plotter.render_window
                sz = rw.GetSize()
                asz = rw.GetActualSize()
            except Exception:
                sz = asz = "?"
            print(
                f"[box] mode={'crossing' if crossing else 'window'} "
                f"event=({x0},{y0})->({x1},{y1}) "
                f"box=[{bx0}..{bx1}]x[{by0}..{by1}] "
                f"win_size={sz} actual={asz}",
                flush=True,
            )

        hits: list["DimTag"] = []

        # Collect all pickable entities and their representative points
        entities: list["DimTag"] = []
        all_corners: list[np.ndarray] = []
        corner_counts: list[int] = []

        for dt in self._registry.all_entities():
            if dt[0] not in self._pickable_dims:
                continue
            if self._hidden_check(dt):
                continue
            # Prefer actual mesh vertices (tight to silhouette) over
            # 3D AABB corners (loose — projects to a rectangle much
            # wider than curves/surfaces at angled views, causing
            # phantom hits on nearby boxes).
            if dt[0] >= 1:
                pts = self._registry.entity_points(dt)
                if pts is not None and len(pts) > 0:
                    entities.append(dt)
                    all_corners.append(pts)
                    corner_counts.append(len(pts))
                    continue
            bbox = self._registry.bbox(dt)
            if bbox is not None:
                entities.append(dt)
                all_corners.append(bbox)
                corner_counts.append(len(bbox))
            else:
                c = self._registry.centroid(dt)
                if c is not None:
                    entities.append(dt)
                    all_corners.append(c.reshape(1, 3))
                    corner_counts.append(1)

        if entities:
            # Project all corners via VTK WorldToDisplay
            pts_all = np.vstack(all_corners)
            screen_x = np.empty(len(pts_all))
            screen_y = np.empty(len(pts_all))
            for i, p in enumerate(pts_all):
                renderer.SetWorldPoint(p[0], p[1], p[2], 1.0)
                renderer.WorldToDisplay()
                dp = renderer.GetDisplayPoint()
                screen_x[i] = dp[0]
                screen_y[i] = dp[1]

            # Check each entity's points against the box
            offset = 0
            for i, dt in enumerate(entities):
                n = corner_counts[i]
                sx = screen_x[offset:offset + n]
                sy = screen_y[offset:offset + n]
                hit = _entity_in_box(sx, sy, bx0, bx1, by0, by1, crossing)
                if _debug:
                    print(
                        f"[box]   dt={dt} n={n} "
                        f"sx=[{sx.min():.1f}..{sx.max():.1f}] "
                        f"sy=[{sy.min():.1f}..{sy.max():.1f}] "
                        f"-> {'HIT' if hit else 'miss'}",
                        flush=True,
                    )
                if hit:
                    hits.append(dt)
                offset += n

        if _debug:
            print(f"[box] total hits={len(hits)}: {hits}", flush=True)

        if self.on_box_select is not None:
            self.on_box_select(hits, ctrl)

    @property
    def hover_entity(self) -> "DimTag | None":
        return self._hover_id
