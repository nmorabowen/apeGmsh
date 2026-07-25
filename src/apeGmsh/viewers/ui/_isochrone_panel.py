"""IsochroneProfilePanel — the curve-family chart for an isochrone profile.

Hosts a matplotlib ``FigureCanvasQTAgg`` in the results viewer's plot
pane and draws one curve per sampled instant, coloured along time, with
a colourbar in time units. A heavier highlight curve tracks the viewer's
current step so scrubbing shows where "now" sits inside the family.

The family is read once (it is a fixed set of instants) and cached for
the panel's lifetime; only the highlight re-reads on a step change —
one narrow slab read of the path's nodes at a single step, the same
cheap-update discipline as :class:`~._time_history.TimeHistoryPanel`.

Data comes through the owning
:class:`~..diagrams._isochrone_profile.IsochroneProfileDiagram`
(``read_profile`` / ``read_profile_at_step``), which is already
selector- and stage-scoped — the panel never touches Results internals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray
    from ..diagrams._director import ResultsDirector
    from ..diagrams._isochrone_profile import IsochroneProfileDiagram


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


def _matplotlib_qt():
    """Lazy-import matplotlib's Qt backend.

    Raises ImportError when matplotlib is absent; the diagram's
    ``make_side_panel`` catches it and renders the 3-D path only.
    """
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.figure import Figure
    return FigureCanvas, Figure


def _highlight_color() -> str:
    """Colour for the "current step" curve, from the active palette.

    Read live rather than snapshotted at import: a theme switch must
    reach the next redraw (the stale-colour class of bug the
    module-level-constant guard exists to prevent). The ``error`` role
    is the palette's attention colour — it is deliberately outside every
    sequential colormap, so "now" never blends into the family it sits
    in. Falls back to a plain red if the theme isn't importable (a bare
    matplotlib context with no viewer chrome).
    """
    try:
        from .theme import THEME
        return THEME.current.error
    except Exception:
        return "red"


class IsochroneProfilePanel:
    """Side panel: response vs position, one curve per instant."""

    def __init__(
        self,
        diagram: "IsochroneProfileDiagram",
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

        self._title = QtWidgets.QLabel(self._title_text())
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        layout.addWidget(self._title)

        self._fig = Figure(figsize=(4.4, 3.6), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        self._widget = widget
        self._cbar: Any = None
        self._highlight: Any = None
        # Cached family (fixed instants) — read once, redrawn on demand.
        self._family: "Optional[tuple[ndarray, ndarray, ndarray]]" = None

        # Step / stage subscriptions. ``attach_dispatcher`` swaps these
        # for UI-lane coalesced dispatcher subs so a rapid scrubber drag
        # fires one highlight redraw per Qt tick.
        self._unsub_step: Optional[Callable[[], None]] = (
            director.subscribe_step(lambda _i: self._refresh_highlight())
        )
        self._unsub_stage: Optional[Callable[[], None]] = (
            director.subscribe_stage(lambda _id: self.refresh())
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Public API (mirrors TimeHistoryPanel / FiberSectionPanel)
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def close(self) -> None:
        for unsub in (self._unsub_step, self._unsub_stage):
            if unsub is None:
                continue
            try:
                unsub()
            except Exception:
                pass

    def attach_dispatcher(self, dispatcher: Any) -> None:
        """Migrate the step + stage subscriptions onto the dispatcher.

        Same contract as :meth:`._time_history.TimeHistoryPanel.
        attach_dispatcher`: legacy unsub first, idempotent on repeat,
        ``None`` dispatcher is a no-op.
        """
        if dispatcher is None:
            return
        from ..diagrams._dispatch import Lane, STAGE_CHANGED, STEP_CHANGED
        for attr in ("_unsub_step", "_unsub_stage"):
            unsub = getattr(self, attr, None)
            if unsub is not None:
                try:
                    unsub()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._unsub_step = dispatcher.subscribe(
            STEP_CHANGED,
            lambda _kind, _payload: self._refresh_highlight(),
            lane=Lane.UI,
            coalesce=True,
        )
        self._unsub_stage = dispatcher.subscribe(
            STAGE_CHANGED,
            lambda _kind, _payload: self.refresh(),
            lane=Lane.UI,
            coalesce=True,
        )

    def refresh(self) -> None:
        """Re-read the whole curve family and redraw (e.g. stage change)."""
        self._family = self._diagram.read_profile()
        self._title.setText(self._title_text())
        self._draw()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _title_text(self) -> str:
        component = self._diagram.spec.selector.component
        axis = self._diagram.path_axis_name
        return f"Isochrones — {component} vs {axis}"

    def _axis_labels(self) -> tuple[str, str]:
        component = self._diagram.spec.selector.component
        axis = self._diagram.path_axis_name
        if self._diagram.value_on_horizontal():
            return (component, axis)
        return (axis, component)

    def _render_empty(self, message: str) -> None:
        self._drop_colorbar()
        self._ax.cla()
        self._highlight = None
        self._ax.text(
            0.5, 0.5, message, transform=self._ax.transAxes,
            ha="center", va="center", color="gray", style="italic",
        )
        xlabel, ylabel = self._axis_labels()
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _draw(self) -> None:
        if self._family is None:
            self._render_empty(
                "No profile data — check the component and the path "
                "selection."
            )
            return
        position, times, values = self._family
        if position.size == 0 or times.size == 0:
            self._render_empty("Profile is empty.")
            return

        style = self._diagram.spec.style
        self._drop_colorbar()
        self._ax.cla()
        self._highlight = None

        colors, norm, cmap = self._time_colors(times, style.cmap)
        value_h = self._diagram.value_on_horizontal()
        marker = "o" if style.show_markers else None

        for row, color in zip(values, colors):
            if value_h:
                self._ax.plot(
                    row, position, color=color,
                    linewidth=style.line_width, marker=marker,
                    markersize=3.0,
                )
            else:
                self._ax.plot(
                    position, row, color=color,
                    linewidth=style.line_width, marker=marker,
                    markersize=3.0,
                )

        xlabel, ylabel = self._axis_labels()
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.grid(True, alpha=0.25)

        # Colourbar in time units — without it the family is a pile of
        # unlabelled curves.
        if times.size > 1:
            try:
                import matplotlib.cm as mcm
                mappable = mcm.ScalarMappable(norm=norm, cmap=cmap)
                mappable.set_array(times)
                self._cbar = self._fig.colorbar(mappable, ax=self._ax)
                self._cbar.set_label("time")
            except Exception:
                self._cbar = None

        self._refresh_highlight(redraw=False)
        self._canvas.draw_idle()

    @staticmethod
    def _time_colors(times: "ndarray", cmap_name: str):
        """Per-curve colours plus the ``(norm, cmap)`` the bar reuses."""
        import matplotlib.colors as mcolors
        from matplotlib import colormaps

        try:
            cmap = colormaps[cmap_name]
        except (KeyError, TypeError):
            cmap = colormaps["viridis"]
        lo = float(np.min(times))
        hi = float(np.max(times))
        if hi <= lo:
            hi = lo + 1.0
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        return ([cmap(norm(float(t))) for t in times], norm, cmap)

    def _refresh_highlight(self, *, redraw: bool = True) -> None:
        """Redraw the "current step" curve on top of the family."""
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except Exception:
                pass
            self._highlight = None
        style = self._diagram.spec.style
        if not style.mark_current_step or self._family is None:
            if redraw:
                self._canvas.draw_idle()
            return
        data = self._diagram.read_profile_at_step(
            int(self._director.step_index),
        )
        if data is not None:
            position, values = data
            if position.size:
                if self._diagram.value_on_horizontal():
                    lines = self._ax.plot(
                        values, position, color=_highlight_color(),
                        linewidth=max(2.0, style.line_width * 1.8),
                        zorder=5,
                    )
                else:
                    lines = self._ax.plot(
                        position, values, color=_highlight_color(),
                        linewidth=max(2.0, style.line_width * 1.8),
                        zorder=5,
                    )
                self._highlight = lines[0] if lines else None
        if redraw:
            self._canvas.draw_idle()

    def _drop_colorbar(self) -> None:
        """Remove the colourbar so a redraw doesn't stack bars."""
        if self._cbar is None:
            return
        try:
            self._cbar.remove()
        except Exception:
            pass
        self._cbar = None


__all__ = ["IsochroneProfilePanel"]
