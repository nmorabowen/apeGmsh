"""PlotPane — the Qt widget projecting one plot view of a session.

ADR 0098 §6 (S4-2): "A plot is a pane, not a dock of a contour." From
S3 on a plot already tiled, activated, closed and carried a header;
what was missing was the chart. This module is it, plus the thing a
plot pane has no share of: **a repaint path of its own**.

:class:`~._reconciler.SessionReconciler` and :class:`~._pane.MeshPane`
serve mesh views only — ``_resolve_pane`` filters ``MeshView`` and
every layer it emits goes through a ``RenderBackend``. A chart has no
backend and no layers, so :class:`PlotReconciler` restates the same
three disciplines against matplotlib instead of forking it:

* **One gesture, one redraw** (criterion 12). Session ticks mark dirty
  and schedule ONE deferred flush, so an N-write gesture costs one
  ``realize_plot`` and one canvas draw.
* **A signature gate that does not skip the redraw that matters.**
  That is the trap S4-1 fell into — a selection write ticked the
  session, the signature compared equal, and the highlight never
  painted. The signature here is ``(pane_id, kind, series, effective
  instant)``: every input ``realize_plot`` reads, the §7 instant
  included, because the instant IS the cursor and "if the link is on
  and moving the plot does not move the meshes, the link is a lie"
  cuts both ways.
* **A failure is never silent** (ADR 0084 D4). Flush errors report as
  ``pump.session_plot`` — under the ``pump.session`` prefix the window
  surfaces on its status bar and the autouse ``pump_failures`` fixture
  fails tests on.

The gate has two halves because the two repaints cost differently: a
cursor move only slides a vertical marker over cached arrays (the
scrubber drags at interactive rates), while a series change re-reads.
Splitting them is an optimisation of the CHEAP half only — anything
the data signature covers still re-realizes.

What is deliberately NOT in the signature is the §8 **selection**. A
plot's membership is COPIED at creation (§6: "a live alias to the
session selection is not a v1 source"), so a later pick must not
redraw this chart — the opposite of the mesh pane's rule, for the
opposite reason.

``TimeHistoryPanel`` was the template: same matplotlib-in-Qt canvas,
same cache-and-move-the-marker idiom. It is rewritten rather than
adapted because every input it has is a director (``read_history``,
``current_time``, ``subscribe_step``), and the director is the old
ontology. It stays untouched until S6a.

matplotlib is an OPTIONAL dependency (the ``plot`` extra), so a pane
without it renders a named refusal rather than failing to build —
the same reading ``results.plot`` gives.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from qtpy import QtCore, QtWidgets

from .._failures import report
from ._plots import realize_plot

_MPL_MISSING = (
    "This plot pane needs matplotlib, which is an optional dependency "
    "of apeGmsh. Install it with `pip install apeGmsh[plot]` (or `pip "
    "install matplotlib`) and reopen the session. The plot's series "
    "resolve without it — session.realize(pane) returns the arrays."
)

#: Sentinel for "no flush has run yet" — distinct from any real
#: signature, including the ``None`` a removed pane produces.
_NO_SIGNATURE = object()

#: Curve colours, in order. Deliberately a fixed cycle rather than the
#: palette's accent: several series on one chart are one plot view
#: (§6), and they have to be told apart from each other, not from the
#: chrome.
_CURVE_COLORS = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
)

#: Past this many curves the legend is taller than the axes and stops
#: helping. The curves still draw — only the key is dropped.
_MAX_LEGEND_ENTRIES = 8


def _default_defer(fn: Callable[[], None]) -> None:
    """Schedule ``fn`` on the Qt event loop; run inline without Qt.

    The reconciler's idiom, restated (see :mod:`._reconciler`): tests
    inject their own immediate or queued ``defer_fn``.
    """
    try:
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, fn)
    except Exception:
        fn()


def _matplotlib_qt():
    """``(FigureCanvas, Figure)`` for an in-Qt chart, or ``None``."""
    try:
        import matplotlib
        matplotlib.use("QtAgg", force=False)
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
        )
        from matplotlib.figure import Figure
        return FigureCanvas, Figure
    except Exception:
        return None


class PlotChart:
    """The drawing half — a matplotlib canvas and nothing else.

    Session-free on purpose: it takes a
    :class:`~._plots.RealizedPlot` and draws it, which is what makes
    the direct-equality oracle testable (``ax.lines[i].get_ydata()``
    against ``results.plot.history``) without a window or a pixel.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        FigureCanvas, Figure = _matplotlib_qt()
        self._figure = Figure(figsize=(5.0, 3.0), tight_layout=True)
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvas(self._figure)
        if parent is not None:
            self._canvas.setParent(parent)
        self._marker: Any = None
        self._time: "Optional[np.ndarray]" = None
        # Set before theming, not by it: an environment with no theme
        # module still has to be able to draw a refusal message.
        self._text_color = "#000000"
        self._grid_color = "#000000"
        self.apply_theme()

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._canvas

    @property
    def axes(self) -> Any:
        """The live ``Axes``. The test oracle reads its lines; nothing
        else should write to it."""
        return self._axes

    def apply_theme(self) -> None:
        """Follow the window palette (0087: one palette, everything).

        A chart is the one pane whose content is drawn by a library
        with its own default colours, so it is also the one that would
        otherwise sit white inside a dark window.
        """
        try:
            from ..ui.theme import THEME
            palette = THEME.current
        except Exception:
            return
        base = getattr(palette, "base", None) or "#ffffff"
        text = getattr(palette, "text", None) or "#000000"
        self._figure.set_facecolor(base)
        self._axes.set_facecolor(base)
        for spine in self._axes.spines.values():
            spine.set_color(text)
        self._axes.tick_params(colors=text, which="both")
        self._axes.xaxis.label.set_color(text)
        self._axes.yaxis.label.set_color(text)
        self._text_color = text
        self._grid_color = text

    def draw(self, realized: Any) -> None:
        """Paint one resolved plot. Replaces whatever was drawn."""
        self._axes.cla()
        self.apply_theme()
        series = realized.series
        if not series:
            self._empty(
                "This plot has no series yet. Select nodes or Gauss "
                "points and use “New plot from selection”, or "
                "add one from Python.",
            )
            return
        for index, curve in enumerate(series):
            self._axes.plot(
                np.asarray(curve.time), np.asarray(curve.values),
                linewidth=1.4, label=curve.label,
                color=_CURVE_COLORS[index % len(_CURVE_COLORS)],
            )
        self._axes.set_xlabel("time")
        self._axes.set_ylabel(series[0].quantity)
        self._axes.grid(True, alpha=0.25, color=self._grid_color)
        if len(series) <= _MAX_LEGEND_ENTRIES:
            legend = self._axes.legend(fontsize="small")
            for text in legend.get_texts():
                text.set_color(self._text_color)
        self._time = np.asarray(series[0].time)
        self._marker = None
        self.move_cursor(realized.cursor)

    def move_cursor(self, cursor: Any) -> None:
        """Slide the playhead. The cheap half of the repaint: the
        arrays are already drawn, so a scrubber drag costs one line."""
        if self._marker is not None:
            try:
                self._marker.remove()
            except Exception:
                pass
            self._marker = None
        x = self._cursor_x(cursor)
        if x is not None:
            self._marker = self._axes.axvline(
                x, color="#d62728", linewidth=1.0, alpha=0.7,
            )
        self._canvas.draw_idle()

    def show_message(self, text: str) -> None:
        """Draw a named refusal in place of a chart — a plot whose
        series cannot resolve says why, rather than showing empty axes
        that read as "the model did not move"."""
        self._axes.cla()
        self.apply_theme()
        self._time = None
        self._marker = None
        self._empty(text)

    # -- internals -----------------------------------------------------

    def _empty(self, text: str) -> None:
        self._axes.text(
            0.5, 0.5, text, transform=self._axes.transAxes,
            ha="center", va="center", wrap=True,
            color=self._text_color, style="italic", fontsize="small",
        )
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._canvas.draw_idle()

    def _cursor_x(self, cursor: Any) -> Optional[float]:
        """The x of an ``Instant`` on THIS chart's time axis.

        The instant is ``(stage, step)`` and the axis is time, so the
        step indexes the series' own time array — the same array the
        meshes are posed from, which is what keeps the §7 link honest.
        ``Instant`` rejects negative steps by construction, so the only
        out-of-range case is a cursor naming a step this stage does not
        have: no playhead, rather than one at the wrong place.
        """
        if cursor is None or self._time is None or self._time.size == 0:
            return None
        step = int(cursor.step)
        if step >= int(self._time.size):
            return None
        return float(self._time[step])


class PlotReconciler:
    """Keeps one chart painting one plot view of one session."""

    def __init__(
        self,
        session: Any,
        pane_id: str,
        chart: Optional[PlotChart],
        *,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> None:
        self._session = session
        self._pane_id = pane_id
        self._chart = chart
        self._defer = defer_fn or _default_defer
        self._dirty = False
        self._scheduled = False
        self._disposed = False
        self._forced = False
        self._signature: Any = _NO_SIGNATURE
        self._realized: Optional[Any] = None
        session.subscribe(self._on_tick)
        self.schedule()

    # -- surface -------------------------------------------------------

    @property
    def pane_id(self) -> str:
        return self._pane_id

    @property
    def realized(self) -> Optional[Any]:
        """The last flush's ``RealizedPlot`` — ``None`` before the
        first successful flush, or after one that refused. The
        inspector reads the series off it."""
        return self._realized

    def schedule(self) -> None:
        if self._disposed:
            return
        self._dirty = True
        if self._scheduled:
            return
        self._scheduled = True
        self._defer(self._flush)

    def schedule_forced(self) -> None:
        """A repaint the session cannot see — the theme palette the
        chart bakes into its facecolors and its text."""
        self._forced = True
        self.schedule()

    def flush_now(self) -> None:
        self._flush()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._session.unsubscribe(self._on_tick)

    # -- internals -----------------------------------------------------

    def _on_tick(self) -> None:
        self.schedule()

    def _resolve_pane(self) -> Optional[Any]:
        from apeGmsh.results.session import PlotView

        try:
            pane = self._session.pane(self._pane_id)
        except KeyError:
            return None
        return pane if isinstance(pane, PlotView) else None

    def _signature_of(self, plot: Optional[Any]) -> Any:
        """Everything ``realize_plot`` reads, as a comparable tuple.

        ``(data, cursor)``: the halves are separated so a cursor-only
        move can take the cheap path, NOT so the cursor can be left
        out — leaving it out is exactly the S4-1 defect, one layer up.
        ``PlotSeries`` / ``PlotSource`` / ``Instant`` are all frozen
        value types, so ``==`` is exact and total.
        """
        if plot is None:
            return None
        return (
            (plot.id, plot.kind, plot.series),
            self._session.effective_instant(plot),
        )

    def _flush(self) -> None:
        self._scheduled = False
        if self._disposed or not self._dirty:
            return
        self._dirty = False
        forced, self._forced = self._forced, False
        try:
            plot = self._resolve_pane()
            signature = self._signature_of(plot)
        except Exception as exc:
            self._signature = _NO_SIGNATURE
            report("pump.session_plot", exc)
            return
        previous, self._signature = self._signature, signature
        if plot is None or self._chart is None:
            return
        if not forced and signature == previous:
            return                       # nothing this chart draws moved
        if (
            not forced
            and previous is not _NO_SIGNATURE
            and previous is not None
            and signature[0] == previous[0]
            and _same_stage(signature[1], previous[1])
            and self._realized is not None
        ):
            # Only the STEP moved. The arrays on screen are the whole
            # record of this stage (§6), so the playhead slides over
            # cached data and no read happens — the scrubber's drag
            # rate depends on this staying true.
            self._realized = _with_cursor(self._realized, signature[1])
            self._chart.move_cursor(signature[1])
            return
        try:
            self._realized = realize_plot(self._session, plot)
            self._chart.draw(self._realized)
        except Exception as exc:
            # A plot that cannot resolve says so ON the chart — an
            # unreadable component, an over-cap selection and a
            # not-yet-implemented kind are all things the AUTHOR can
            # fix, so the message belongs where they are looking. The
            # signature is dropped so the next tick retries.
            self._realized = None
            self._signature = _NO_SIGNATURE
            self._chart.show_message(f"{type(exc).__name__}: {exc}")
            report("pump.session_plot", exc)


def _same_stage(a: Any, b: Any) -> bool:
    """Whether two instants read the SAME stage — the limit of the
    cheap path.

    ``realize_plot`` resolves every series inside
    ``results.stage(cursor.stage)``, so the cached arrays are a
    function of the STAGE. The step may move freely (that is the
    scrubber, and the whole reason the cheap path exists); a stage
    change must re-read, or the chart paints one stage's record under
    another stage's playhead — a stale picture, which is the failure
    class this reconciler exists to prevent.

    A ``None`` on either side is treated as a change: no instant means
    ``realize_plot`` falls back to the LAST stage, and re-reading once
    is cheaper than reasoning about whether that fallback happened to
    agree.
    """
    if a is None or b is None:
        return a is None and b is None
    return a.stage == b.stage


def _with_cursor(realized: Any, cursor: Any) -> Any:
    """``realized`` with the playhead moved — the cached-arrays path.

    The series are untouched by construction, which is the whole point
    of the cheap half: the record does not change with the cursor.
    """
    from dataclasses import replace

    return replace(realized, cursor=cursor)


class PlotPane(QtWidgets.QWidget):
    """One plot view, projected — the content of a plot pane frame.

    Carries the same small surface as :class:`~._pane.MeshPane`
    (``reconciler`` / ``request_reconcile`` / ``apply_navigation`` /
    ``dispose``, plus ``backend`` and ``surface`` answering ``None``)
    so the frame and the host address the two kinds of pane through
    one protocol instead of branching on kind at every call site.
    """

    def __init__(
        self,
        session: Any,
        pane_id: str,
        *,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._chart: Optional[PlotChart] = (
            PlotChart(parent=self) if _matplotlib_qt() is not None else None
        )
        if self._chart is not None:
            layout.addWidget(self._chart.widget)
        else:
            label = QtWidgets.QLabel(_MPL_MISSING)
            label.setObjectName("SessionPlotPaneUnavailable")
            label.setWordWrap(True)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setEnabled(False)
            layout.addWidget(label)
        self._reconciler = PlotReconciler(
            session, pane_id, self._chart, defer_fn=defer_fn,
        )

    # -- surface -------------------------------------------------------

    @property
    def chart(self) -> Optional[PlotChart]:
        """The chart, or ``None`` without matplotlib."""
        return self._chart

    @property
    def reconciler(self) -> PlotReconciler:
        return self._reconciler

    @property
    def backend(self) -> Any:
        """A chart paints itself — there is no ``RenderBackend``."""
        return None

    @property
    def surface(self) -> Optional[QtWidgets.QWidget]:
        """No GL surface: the toolbar's camera presets and screenshot
        act on mesh panes, and answering ``None`` is how the frame
        tells them so (0087 INV-2)."""
        return None

    def request_reconcile(self) -> None:
        """Forced repaint for a change the session cannot see — the
        palette the chart bakes into its facecolors."""
        self._reconciler.schedule_forced()

    def apply_navigation(self, token: str) -> None:
        """A chart has no camera to navigate."""
        return None

    def dispose(self) -> None:
        self._reconciler.dispose()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt API
        self.dispose()
        super().closeEvent(event)


__all__ = ["PlotChart", "PlotPane", "PlotReconciler"]
