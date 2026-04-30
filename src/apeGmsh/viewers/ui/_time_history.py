"""TimeHistoryPanel — matplotlib-in-Qt time-series plot for one node.

Constructed with ``(director, node_id, component)``. Reads the full
``(T,)`` slab once, caches it for the panel's lifetime, and draws it
once. Subscribes to step changes only to move a vertical "current
step" marker on the chart — the underlying data doesn't refetch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


def _matplotlib_qt():
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.figure import Figure
    return FigureCanvas, Figure


class TimeHistoryPanel:
    """Side panel showing one node's component history vs time."""

    def __init__(
        self,
        director: "ResultsDirector",
        node_id: int,
        component: str,
    ) -> None:
        QtWidgets, _ = _qt()
        FigureCanvas, Figure = _matplotlib_qt()

        self._director = director
        self._node_id = int(node_id)
        self._component = str(component)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QtWidgets.QLabel(
            f"Time history — node {self._node_id} · {self._component}"
        )
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        layout.addWidget(self._title)

        self._fig = Figure(figsize=(5.0, 3.0), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("time")
        self._ax.set_ylabel(self._component)
        self._ax.grid(True, alpha=0.25)

        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        self._widget = widget
        self._marker_line: Any = None
        self._time: "Optional[ndarray]" = None
        self._values: "Optional[ndarray]" = None

        self._unsub_step = director.subscribe_step(self._on_step)
        self._unsub_stage = director.subscribe_stage(
            lambda _id: self.refresh()
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def close(self) -> None:
        for unsub in (self._unsub_step, self._unsub_stage):
            try:
                unsub()
            except Exception:
                pass

    def refresh(self) -> None:
        """Re-fetch the history (e.g., after a stage change)."""
        data = self._director.read_history(self._node_id, self._component)
        if data is None:
            self._render_empty(
                f"No history for node {self._node_id} · {self._component}"
            )
            return
        self._time, self._values = data
        self._draw_plot()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_empty(self, message: str) -> None:
        self._ax.cla()
        self._ax.text(
            0.5, 0.5, message, transform=self._ax.transAxes,
            ha="center", va="center", color="gray", style="italic",
        )
        self._ax.set_xlabel("time")
        self._ax.set_ylabel(self._component)
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _draw_plot(self) -> None:
        if self._time is None or self._values is None:
            return
        self._ax.cla()
        self._ax.plot(
            self._time, self._values,
            linewidth=1.5, color="#1F77B4",
        )
        self._ax.set_xlabel("time")
        self._ax.set_ylabel(self._component)
        self._ax.grid(True, alpha=0.25)
        # Current-step vertical marker
        cur = self._director.current_time()
        if cur is not None:
            self._marker_line = self._ax.axvline(
                cur, color="red", linewidth=1.0, alpha=0.6,
            )
        self._canvas.draw_idle()

    def _on_step(self, _step: int) -> None:
        if self._time is None:
            return
        # Cheap update: just move the marker line; data is cached.
        cur = self._director.current_time()
        if cur is None:
            return
        try:
            if self._marker_line is not None:
                self._marker_line.remove()
        except Exception:
            pass
        self._marker_line = self._ax.axvline(
            cur, color="red", linewidth=1.0, alpha=0.6,
        )
        self._canvas.draw_idle()
