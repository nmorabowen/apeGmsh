"""LegendInteractor — dragging, resizing and right-clicking a colour scale.

ADR 0081 L2. The scale presented a drag handle from the day it shipped
and never once moved: `interactive=True` built a ``vtkScalarBarWidget``
that observes the interactor at VTK priority **0.5**, while the pick
engine aborts every plain left-button press at priority **10**. The
widget could not receive a click, and its geometry was a second copy of
the layout that nothing read back.

So the widget is gone (L0) and the gesture lives here instead, in the
event stream the application already owns:

* **Priority 12** — above navigation (11) and picking (10). The press
  observer hit-tests the cursor against the visible legends and
  **returns without aborting on a miss**, so camera and pick behaviour
  outside a legend is bit-for-bit what it was. Only a hit aborts.
* **Move** — drag the body; edges snap; dropping near the home edge
  re-docks into the automatic stack.
* **Resize** — drag the corner opposite the anchor. Because a legend's
  box is *derived from its text* (ADR 0081 Part 3), the gesture drives
  ``font_scale`` and the box follows, which is why it can never be
  dragged to a size its labels don't fit.
* **Right-click** — hands the hit entry to ``on_context_menu``. The
  menu itself is Qt and belongs to the viewer; this module stays
  toolkit-free, the same split ``install_navigation`` uses for
  ``on_shift_click``.

Every gesture goes through a ``LegendController`` mutator, so the
controller stays the single owner and fires ``LEGEND_CHANGED`` itself
(ADR 0056 Part 2) — a dragged legend is real state, not a widget's
private opinion, and survives the next recolour.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

#: Grab region for the resize corner, and the edge-snap radius (px).
HANDLE_PX = 16
SNAP_PX = 14

#: How close to its home edge a legend must be dropped to re-dock.
REDOCK_PX = 48

MODE_MOVE = "move"
MODE_RESIZE = "resize"


class LegendInteractor:
    """Mouse gestures for the legends of one viewport."""

    def __init__(
        self,
        backend: Any,
        legends: Any,
        *,
        on_context_menu: "Optional[Callable[[Any, tuple[int, int]], None]]" = None,
    ) -> None:
        self._backend = backend
        self._legends = legends
        self._on_context_menu = on_context_menu
        self._iren: Any = None
        self._tags: dict[str, int] = {}
        self._drag: Optional[dict] = None
        self._cursor: Optional[str] = None

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    def install(self) -> bool:
        """Attach the observers. ``False`` when there is no interactor."""
        if self._tags:
            return True
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
        self._tags["rmb"] = iren.AddObserver(
            "RightButtonPressEvent", self._on_rmb_press, 12.0,
        )
        # A release delivered outside the window never reaches us, and a
        # drag with no end follows the cursor forever afterwards.
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

    def hit(self, x: float, y: float) -> "tuple[Any, Optional[str]]":
        """``(entry, mode)`` under the cursor, or ``(None, None)``.

        ``x`` / ``y`` are VTK event coordinates — pixels from the
        bottom-left, the same origin the normalized viewport uses.
        Later legends are tested first so the one drawn on top wins.
        """
        # The extents about to be tested must describe THIS viewport;
        # see ``LegendController.ensure_current``. No-op unless the
        # viewport actually moved.
        ensure = getattr(self._legends, "ensure_current", None)
        if ensure is not None:
            try:
                ensure()
            except Exception:
                pass
        vw, vh = self._legends.viewport_px()
        nx, ny = float(x) / vw, float(y) / vh
        hx, hy = HANDLE_PX / vw, HANDLE_PX / vh
        for entry in reversed(self._legends.entries()):
            if not entry.visible:
                continue
            ax, ay = entry.anchor
            w, h = entry.extent
            if not (ax <= nx <= ax + w and ay <= ny <= ay + h):
                continue
            corner = nx >= ax + w - hx and ny >= ay + h - hy
            return entry, (MODE_RESIZE if corner else MODE_MOVE)
        return None, None

    # ------------------------------------------------------------------
    # Gesture handlers
    # ------------------------------------------------------------------

    def _on_lmb_press(self, caller: Any, _event: Any) -> None:
        # Shift+LMB is navigation's orbit gesture (priority 11); never
        # take it, even over a legend.
        try:
            if caller.GetShiftKey():
                return
        except Exception:
            pass
        x, y = caller.GetEventPosition()
        entry, mode = self.hit(x, y)
        if entry is None:
            return          # miss: no abort — pick and camera are untouched
        try:
            if caller.GetRepeatCount():
                # Double-click returns a hand-placed legend to the stack.
                self._legends.redock(entry.key)
                self._abort(caller, "press")
                return
        except Exception:
            pass
        vw, vh = self._legends.viewport_px()
        self._drag = {
            "mode": mode,
            "key": entry.key,
            "press": (float(x) / vw, float(y) / vh),
            "anchor": entry.anchor,
            "extent": entry.extent,
            "scale": self._scale_of(entry),
        }
        self._abort(caller, "press")

    def _on_mouse_move(self, caller: Any, _event: Any) -> None:
        x, y = caller.GetEventPosition()
        if self._drag is None:
            self._update_cursor(x, y)
            return
        vw, vh = self._legends.viewport_px()
        nx, ny = float(x) / vw, float(y) / vh
        d = self._drag
        if d["mode"] == MODE_MOVE:
            ax = d["anchor"][0] + (nx - d["press"][0])
            ay = d["anchor"][1] + (ny - d["press"][1])
            self._legends.set_anchor(
                d["key"], self._snap(ax, ay, d["extent"], vw, vh),
            )
        else:
            self._legends.set_font_scale(
                d["key"], d["scale"] * self._resize_ratio(d, nx, ny),
            )
        self._abort(caller, "move")

    def _on_lmb_release(self, caller: Any, _event: Any) -> None:
        if self._drag is None:
            return
        key, mode = self._drag["key"], self._drag["mode"]
        self._drag = None
        if mode == MODE_MOVE:
            self._maybe_redock(key)
        self._abort(caller, "release")

    def _on_leave(self, _caller: Any, _event: Any) -> None:
        """End any drag when the cursor leaves — the release may not come."""
        if self._drag is not None:
            key, mode = self._drag["key"], self._drag["mode"]
            self._drag = None
            if mode == MODE_MOVE:
                self._maybe_redock(key)
        self._set_cursor(None)

    def _on_rmb_press(self, caller: Any, _event: Any) -> None:
        if self._on_context_menu is None:
            return
        x, y = caller.GetEventPosition()
        entry, _mode = self.hit(x, y)
        if entry is None:
            return          # miss: RMB stays navigation's pan
        try:
            size = self._backend.viewport_size()
            # Qt menus want top-left-origin screen coordinates.
            self._on_context_menu(entry, (int(x), int(size[1] - y)))
        except Exception:
            pass
        self._abort(caller, "rmb")

    # ------------------------------------------------------------------
    # Gesture maths
    # ------------------------------------------------------------------

    @staticmethod
    def _snap(
        ax: float, ay: float, extent: tuple, vw: int, vh: int,
    ) -> tuple[float, float]:
        """Pull an anchor onto a viewport edge when it lands near one."""
        from ._legend import MARGIN_PX

        w, h = extent
        mx, my = MARGIN_PX / vw, MARGIN_PX / vh
        sx, sy = SNAP_PX / vw, SNAP_PX / vh
        if abs(ax - mx) < sx:
            ax = mx
        elif abs((ax + w) - (1.0 - mx)) < sx:
            ax = 1.0 - mx - w
        if abs(ay - my) < sy:
            ay = my
        elif abs((ay + h) - (1.0 - my)) < sy:
            ay = 1.0 - my - h
        return ax, ay

    @staticmethod
    def _resize_ratio(drag: dict, nx: float, ny: float) -> float:
        """Scale factor from the corner drag, relative to the anchor.

        Measured along the box diagonal so both axes contribute and the
        gesture reads the same whichever way the legend is oriented.
        """
        ax, ay = drag["anchor"]
        w, h = drag["extent"]
        start = max(((drag["press"][0] - ax) ** 2
                     + (drag["press"][1] - ay) ** 2) ** 0.5, 1e-6)
        now = ((nx - ax) ** 2 + (ny - ay) ** 2) ** 0.5
        # A drag that crosses the anchor would invert the legend; clamp
        # to a floor instead and let the controller apply its own bounds.
        return max(now / start, 1e-3)

    def _scale_of(self, entry: Any) -> float:
        if entry.font_scale is not None:
            return float(entry.font_scale)
        return float(self._legends.default_font_scale)

    def _maybe_redock(self, key: Any) -> None:
        """Re-dock a legend dropped back onto its home edge."""
        entry = self._legends.entry(key)
        if entry is None or entry.slot is not None:
            return
        vw, vh = self._legends.viewport_px()
        ax, ay = entry.anchor
        w, h = entry.extent
        home = (
            (1.0 - (ax + w)) * vw if entry.vertical    # right edge
            else ay * vh                               # bottom edge
        )
        if home <= REDOCK_PX:
            self._legends.redock(key)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _update_cursor(self, x: float, y: float) -> None:
        _entry, mode = self.hit(x, y)
        self._set_cursor(mode)

    def _set_cursor(self, mode: "Optional[str]") -> None:
        """Ask the arbiter for this hover's cursor.

        Routed through :mod:`._cursor_arbiter` since ADR 0083 S2 added
        a second hover interactor on the same events: a direct write
        here would be overwritten by whichever of us ran last, rather
        than by whichever of us is actually under the pointer.
        """
        if mode == self._cursor:
            return
        self._cursor = mode
        try:
            import vtk

            from ._cursor_arbiter import request_cursor

            request_cursor(
                self._backend.plotter.render_window,
                "legend",
                {
                    MODE_RESIZE: vtk.VTK_CURSOR_SIZENE,
                    MODE_MOVE: vtk.VTK_CURSOR_SIZEALL,
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


def install_legend_interactor(
    backend: Any,
    legends: Any,
    *,
    on_context_menu: "Optional[Callable[[Any, tuple[int, int]], None]]" = None,
) -> "Optional[LegendInteractor]":
    """Install legend gestures on ``backend``'s interactor.

    Returns the interactor, or ``None`` for headless / offscreen
    backends that have no interactor to observe.
    """
    interactor = LegendInteractor(
        backend, legends, on_context_menu=on_context_menu,
    )
    return interactor if interactor.install() else None
