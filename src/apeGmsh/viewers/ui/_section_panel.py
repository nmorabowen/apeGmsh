"""Dockable side panel — 2-D fiber section scatter for the picked GP.

Hosts a matplotlib ``FigureCanvasQTAgg`` inside a small Qt widget.
Subscribes to the Director's ``picked_gp`` and ``step`` changes;
re-renders the scatter plot whenever either changes.

The panel reads slab data via the owning ``FiberSectionDiagram``'s
``read_section_at_gp(eid, gp_index)`` helper — that path is already
selector-scoped and stage-scoped, so the panel doesn't need to know
about Results internals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray
    from ..diagrams._director import ResultsDirector
    from ..diagrams._fiber_section import FiberSectionDiagram


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


def _matplotlib_qt():
    """Lazy-import matplotlib's Qt backend.

    Raises ImportError with a clear message if matplotlib isn't
    available — diagrams that need this should refuse to construct.
    """
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.figure import Figure
    return FigureCanvas, Figure


class FiberSectionPanel:
    """Side panel widget for ``FiberSectionDiagram``."""

    def __init__(
        self,
        diagram: "FiberSectionDiagram",
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

        # ── GP picker dropdown ─────────────────────────────────────
        picker_row = QtWidgets.QHBoxLayout()
        picker_row.addWidget(QtWidgets.QLabel("Beam GP:"))
        self._gp_combo = QtWidgets.QComboBox()
        self._gp_combo.setMinimumWidth(180)
        self._gp_combo.currentIndexChanged.connect(self._on_gp_picked)
        picker_row.addWidget(self._gp_combo, stretch=1)
        layout.addLayout(picker_row)

        # ── Title label ────────────────────────────────────────────
        self._title = QtWidgets.QLabel(
            f"Fiber section — {diagram.spec.selector.component}"
        )
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        layout.addWidget(self._title)

        # ── Matplotlib canvas ──────────────────────────────────────
        self._fig = Figure(figsize=(4.0, 4.0), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("y (section local)")
        self._ax.set_ylabel("z (section local)")
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.grid(True, alpha=0.25)

        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        # Empty-state message
        self._empty_text = self._ax.text(
            0.5, 0.5, "No GP selected", transform=self._ax.transAxes,
            ha="center", va="center", color="gray",
            style="italic",
        )
        self._scatter: Any = None      # the matplotlib PathCollection
        self._cbar: Any = None
        self._fixed_clim: Optional[tuple[float, float]] = None

        self._widget = widget

        # Populate GPs and subscribe
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

        # Initial render
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def close(self) -> None:
        """Tear down subscriptions when the panel is disposed."""
        for unsub in (
            self._unsub_picked, self._unsub_step, self._unsub_diagrams,
        ):
            try:
                unsub()
            except Exception:
                pass

    def refresh(self) -> None:
        """Re-render the scatter for the current picked GP and step."""
        picked = self._director.picked_gp
        if picked is None:
            picked = self._fallback_pick()
        if picked is None:
            self._render_empty("No fiber GPs available")
            return

        eid, gp_idx = picked
        try:
            data = self._diagram.read_section_at_gp(
                eid, gp_idx, self._director.step_index,
            )
        except Exception as exc:
            self._render_empty(f"Read failed: {exc}")
            return

        if data is None:
            self._render_empty(f"No fibers at element {eid}, gp {gp_idx}")
            return

        y_arr, z_arr, area_arr, value_arr = data
        self._render_scatter(eid, gp_idx, y_arr, z_arr, area_arr, value_arr)

    # ------------------------------------------------------------------
    # GP picker
    # ------------------------------------------------------------------

    def _populate_gp_picker(self) -> None:
        if not self._diagram.is_attached:
            return
        gps = self._diagram.available_gps()
        # Block signals while repopulating
        self._gp_combo.blockSignals(True)
        try:
            current = self._director.picked_gp
            self._gp_combo.clear()
            for (eid, gp_idx) in gps:
                self._gp_combo.addItem(
                    f"elem {eid}, gp {gp_idx}", (eid, gp_idx),
                )
            # Restore selection if still present
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
        # Sync combo + refresh plot
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
        # Quietly seed the director so the diagram (3-D dot cloud) can
        # also highlight this GP.
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
        self._ax.set_xlabel("y (section local)")
        self._ax.set_ylabel("z (section local)")
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _render_scatter(
        self, eid: int, gp_idx: int,
        y_arr: "ndarray", z_arr: "ndarray",
        area_arr: "ndarray", value_arr: "ndarray",
    ) -> None:
        self._ax.cla()

        style = self._diagram.spec.style
        if getattr(style, "panel_show_areas", True):
            sizes = (
                style.panel_marker_scale * np.clip(area_arr, 1e-12, None)
                / np.maximum(area_arr.mean(), 1e-12)
            )
        else:
            sizes = np.full_like(area_arr, style.panel_marker_scale)

        clim = self._diagram.current_clim()
        sc = self._ax.scatter(
            y_arr, z_arr,
            s=sizes,
            c=value_arr,
            cmap=style.cmap,
            vmin=clim[0] if clim else None,
            vmax=clim[1] if clim else None,
            edgecolors="black",
            linewidths=0.3,
        )
        self._ax.set_xlabel("y (section local)")
        self._ax.set_ylabel("z (section local)")
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.grid(True, alpha=0.25)
        self._ax.set_title(
            f"elem {eid}, gp {gp_idx} — "
            f"step {self._director.step_index}"
        )
        try:
            if self._cbar is None:
                self._cbar = self._fig.colorbar(sc, ax=self._ax)
            else:
                self._cbar.update_normal(sc)
        except Exception:
            pass
        self._canvas.draw_idle()
