"""Dockable side panel — through-thickness profile for layered shells.

Plots the picked element-GP's layer values against cumulative thickness.
Same lazy matplotlib-Qt pattern as ``FiberSectionPanel``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray
    from ..diagrams._director import ResultsDirector
    from ..diagrams._layer_stack import LayerStackDiagram


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


class LayerThicknessPanel:
    """Side panel widget for ``LayerStackDiagram``."""

    def __init__(
        self,
        diagram: "LayerStackDiagram",
        director: "ResultsDirector",
    ) -> None:
        QtWidgets, _ = _qt()
        FigureCanvas, Figure = _matplotlib_qt()

        self._diagram = diagram
        self._director = director

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── GP picker ──────────────────────────────────────────────
        picker_row = QtWidgets.QHBoxLayout()
        picker_row.addWidget(QtWidgets.QLabel("Shell GP:"))
        self._gp_combo = QtWidgets.QComboBox()
        self._gp_combo.setMinimumWidth(180)
        self._gp_combo.currentIndexChanged.connect(self._on_gp_picked)
        picker_row.addWidget(self._gp_combo, stretch=1)
        layout.addLayout(picker_row)

        self._title = QtWidgets.QLabel(
            f"Through thickness — {diagram.spec.selector.component}"
        )
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        layout.addWidget(self._title)

        # ── Matplotlib canvas ──────────────────────────────────────
        self._fig = Figure(figsize=(3.5, 4.5), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel(diagram.spec.selector.component)
        self._ax.set_ylabel("through-thickness coordinate")
        self._ax.grid(True, alpha=0.25)

        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        self._widget = widget

        self._populate_gp_picker()
        self._unsub_picked = director.subscribe_picked_gp(
            self._on_director_picked
        )
        self._unsub_step = director.subscribe_step(
            lambda _i: self.refresh()
        )
        self._unsub_diagrams = director.subscribe_diagrams(
            self._populate_gp_picker
        )
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def close(self) -> None:
        for unsub in (
            self._unsub_picked, self._unsub_step, self._unsub_diagrams,
        ):
            try:
                unsub()
            except Exception:
                pass

    def refresh(self) -> None:
        picked = self._director.picked_gp
        if picked is None:
            picked = self._fallback_pick()
        if picked is None:
            self._render_empty("No shell GPs available")
            return

        eid, gp_idx = picked
        try:
            data = self._diagram.read_thickness_profile(
                eid, gp_idx, self._director.step_index,
            )
        except Exception as exc:
            self._render_empty(f"Read failed: {exc}")
            return
        if data is None:
            self._render_empty(f"No layer data at elem {eid}, gp {gp_idx}")
            return

        thickness_coord, values = data
        self._render_profile(eid, gp_idx, thickness_coord, values)

    # ------------------------------------------------------------------
    # GP picker
    # ------------------------------------------------------------------

    def _populate_gp_picker(self) -> None:
        if not self._diagram.is_attached:
            return
        gps = self._diagram.available_gps()
        self._gp_combo.blockSignals(True)
        try:
            current = self._director.picked_gp
            self._gp_combo.clear()
            for (eid, gp_idx) in gps:
                self._gp_combo.addItem(
                    f"elem {eid}, gp {gp_idx}", (eid, gp_idx),
                )
            if current is not None:
                for i in range(self._gp_combo.count()):
                    if self._gp_combo.itemData(i) == current:
                        self._gp_combo.setCurrentIndex(i)
                        break
        finally:
            self._gp_combo.blockSignals(False)

    def _on_gp_picked(self, idx: int) -> None:
        if idx < 0:
            return
        data = self._gp_combo.itemData(idx)
        if data is None:
            return
        self._director.set_picked_gp(data)

    def _on_director_picked(
        self, picked: Optional[tuple[int, int]],
    ) -> None:
        if picked is None:
            self._gp_combo.blockSignals(True)
            try:
                self._gp_combo.setCurrentIndex(-1)
            finally:
                self._gp_combo.blockSignals(False)
        else:
            for i in range(self._gp_combo.count()):
                if self._gp_combo.itemData(i) == picked:
                    self._gp_combo.blockSignals(True)
                    try:
                        self._gp_combo.setCurrentIndex(i)
                    finally:
                        self._gp_combo.blockSignals(False)
                    break
        self.refresh()

    def _fallback_pick(self) -> Optional[tuple[int, int]]:
        gps = self._diagram.available_gps() if self._diagram.is_attached else []
        if not gps:
            return None
        first = gps[0]
        self._director.set_picked_gp(first)
        return first

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_empty(self, message: str) -> None:
        self._ax.cla()
        self._ax.text(
            0.5, 0.5, message, transform=self._ax.transAxes,
            ha="center", va="center", color="gray", style="italic",
        )
        self._ax.set_xlabel(self._diagram.spec.selector.component)
        self._ax.set_ylabel("through-thickness coordinate")
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _render_profile(
        self, eid: int, gp_idx: int,
        thickness_coord: "ndarray", values: "ndarray",
    ) -> None:
        self._ax.cla()
        self._ax.plot(
            values, thickness_coord,
            marker="o", markersize=4, linewidth=1.5,
            color="#1F77B4",
        )
        self._ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.6)
        self._ax.set_xlabel(self._diagram.spec.selector.component)
        self._ax.set_ylabel("through-thickness coordinate")
        self._ax.grid(True, alpha=0.25)
        self._ax.set_title(
            f"elem {eid}, gp {gp_idx} — "
            f"step {self._director.step_index}"
        )
        self._canvas.draw_idle()
