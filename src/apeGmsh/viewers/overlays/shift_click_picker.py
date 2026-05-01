"""ShiftClickPicker — shift+left-click in viewport → callback (B++ §8).

Per the spec: shift-click in the viewport adds a series to the active
plot pane tab. This module owns the low-level VTK observer that detects
the modified click and resolves a world position; the consumer decides
what to do with it (typically: snap to nearest FEM node, then open or
update a time-history plot).

Coexists with :mod:`pick_engine` and the navigation observers at the
default priority — the handler only acts when shift is held, so plain
clicks fall through unchanged. The observer never sets ``AbortFlag``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


class ShiftClickPicker:
    """VTK observer that fires a callback on shift+left-click.

    Parameters
    ----------
    plotter
        The PyVista plotter (its ``renderer`` and ``iren`` are used).
    on_shift_pick
        ``callable(world_pos: ndarray)`` invoked when the user
        shift-clicks somewhere that hits the rendered scene. Not called
        for plain clicks, ctrl/alt clicks, or empty-space picks.
    """

    def __init__(
        self,
        plotter: Any,
        on_shift_pick: Callable[["ndarray"], None],
    ) -> None:
        self._plotter = plotter
        self._on_shift_pick = on_shift_pick
        self._observer_tag: Optional[int] = None

        import vtk
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)

        try:
            iren = plotter.iren.interactor
        except Exception:
            iren = plotter.iren
        self._iren = iren
        self._observer_tag = iren.AddObserver(
            "LeftButtonPressEvent", self._on_press, 0.0,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def detach(self) -> None:
        """Remove the VTK observer. Idempotent."""
        if self._observer_tag is not None and self._iren is not None:
            try:
                self._iren.RemoveObserver(self._observer_tag)
            except Exception:
                pass
        self._observer_tag = None

    # ------------------------------------------------------------------
    # Observer callback
    # ------------------------------------------------------------------

    def _on_press(self, caller: Any, _event: str) -> None:
        # Plain / ctrl / alt clicks fall through. Only shift triggers
        # the series-add path.
        try:
            shift = bool(caller.GetShiftKey())
        except Exception:
            return
        if not shift:
            return

        try:
            x, y = caller.GetEventPosition()
        except Exception:
            return

        renderer = self._plotter.renderer
        if renderer is None:
            return
        try:
            self._picker.Pick(int(x), int(y), 0, renderer)
        except Exception:
            return

        # GetCellId returns -1 if nothing was hit.
        try:
            cell_id = self._picker.GetCellId()
        except Exception:
            return
        if cell_id < 0:
            return

        try:
            world = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
        except Exception:
            return

        try:
            self._on_shift_pick(world)
        except Exception as exc:
            import sys
            print(
                f"[ShiftClickPicker] callback raised: {exc}",
                file=sys.stderr,
            )
