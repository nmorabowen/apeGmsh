"""ProbePaletteHUD — top-right HUD overlay for the 3-D viewport (B++ §4.2).

A vertical strip of mode buttons (Point / Line / Slice) plus Stop /
Clear actions, parented to the QtInteractor's widget so it floats
above the rendered scene. Repositioned on viewport resize via a Qt
event filter — no monkey-patching.

Replaces the right-dock ``ProbesTab`` as the primary probe UX in B3.
The tab itself stays in the codebase but no longer appears in the
right dock — kept for the duration of the migration as a fallback.
Sample count and plane axis use defaults (50 samples, z-normal); a
richer popover lands in B5 if desired.

Result feedback is routed through a caller-supplied status callback
(typically ``ResultsWindow.set_status``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..overlays.probe_overlay import (
        LineProbeResult,
        PlaneProbeResult,
        PointProbeResult,
        ProbeOverlay,
    )


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


# Default settings for the simplified HUD — promote to user controls
# in B5 if the use case demands.
_DEFAULT_LINE_SAMPLES = 50
_DEFAULT_PLANE_AXIS = "z"


class ProbePaletteHUD:
    """Vertical HUD palette in the viewport's top-right corner.

    Parameters
    ----------
    viewport_widget
        The widget over which the HUD floats (the QtInteractor's
        rendering widget).
    overlay
        :class:`ProbeOverlay` driving the actual probe lifecycle.
    on_status
        ``callable(message: str, timeout_ms: int)`` for surfacing
        result summaries (typically the window's status bar).
    """

    _MARGIN = 12

    def __init__(
        self,
        viewport_widget: Any,
        overlay: "ProbeOverlay",
        on_status: Callable[[str, int], None],
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._viewport = viewport_widget
        self._overlay = overlay
        self._on_status = on_status
        self._mode: Optional[str] = None

        widget = QtWidgets.QFrame(parent=viewport_widget)
        widget.setObjectName("ProbeHUD")
        widget.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        widget.setFixedWidth(38)
        # Set early so button factories can parent into it.
        self._widget = widget

        lay = QtWidgets.QVBoxLayout(widget)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        self._btn_point = self._make_mode_btn("◉", "Point probe", "point")
        self._btn_line = self._make_mode_btn("⌒", "Line probe", "line")
        self._btn_slice = self._make_mode_btn("▦", "Slice probe", "slice")
        lay.addWidget(self._btn_point)
        lay.addWidget(self._btn_line)
        lay.addWidget(self._btn_slice)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setObjectName("ProbeHUDSep")
        lay.addWidget(sep)

        self._btn_stop = self._make_action_btn("✕", "Stop active probe", self._on_stop)
        self._btn_clear = self._make_action_btn("⌫", "Clear all probes", self._on_clear)
        lay.addWidget(self._btn_stop)
        lay.addWidget(self._btn_clear)

        # Theme-driven styling lives in viewers/ui/theme.py.

        # Wire overlay callbacks. We capture and forward results to
        # the status bar; we don't suppress callbacks already wired
        # by ProbesTab since it's still around as a fallback.
        self._chain_point = overlay.on_point_result
        self._chain_line = overlay.on_line_result
        self._chain_plane = overlay.on_plane_result
        overlay.on_point_result = self._on_point_result
        overlay.on_line_result = self._on_line_result
        overlay.on_plane_result = self._on_plane_result

        # Reposition on viewport resize via event filter.
        self._filter = _ResizeFilter(self.reposition)
        viewport_widget.installEventFilter(self._filter)

        widget.show()
        widget.raise_()
        self.reposition()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def reposition(self) -> None:
        """Move the HUD into the viewport's top-right corner."""
        try:
            vw = self._viewport.width()
        except Exception:
            return
        x = vw - self._widget.width() - self._MARGIN
        y = self._MARGIN
        self._widget.move(max(0, x), y)
        self._widget.raise_()

    # ------------------------------------------------------------------
    # Button construction
    # ------------------------------------------------------------------

    def _make_mode_btn(self, glyph: str, tooltip: str, mode: str):
        QtWidgets, _ = _qt()
        btn = QtWidgets.QToolButton(self._widget)
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setProperty("active", "false")
        btn.setFixedHeight(28)
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        btn.clicked.connect(lambda _checked=False, m=mode: self._activate_mode(m))
        return btn

    def _make_action_btn(
        self, glyph: str, tooltip: str, callback: Callable[[], None],
    ):
        QtWidgets, _ = _qt()
        btn = QtWidgets.QToolButton(self._widget)
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(22)
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        btn.clicked.connect(callback)
        return btn

    # ------------------------------------------------------------------
    # Mode logic
    # ------------------------------------------------------------------

    def _activate_mode(self, mode: str) -> None:
        # Toggle off if user clicks the active mode again.
        if self._mode == mode:
            self._on_stop()
            return

        self._set_active(mode)
        try:
            if mode == "point":
                self._overlay.start_point_probe()
                self._on_status("Point probe — click a node", 0)
            elif mode == "line":
                self._overlay.start_line_probe()
                self._on_status(
                    "Line probe — click point A, then point B", 0,
                )
            elif mode == "slice":
                # Slice is one-shot; no interactive picking.
                result = self._overlay.probe_with_plane(
                    normal=_DEFAULT_PLANE_AXIS,
                )
                self._set_active(None)
                if result is None:
                    self._on_status("Slice probe — no slice produced", 4000)
                # The result-callback chain reports a richer summary.
        except Exception as exc:
            self._on_status(f"Probe failed: {exc}", 5000)
            self._set_active(None)

    def _on_stop(self) -> None:
        try:
            self._overlay.stop()
        except Exception:
            pass
        self._set_active(None)
        self._on_status("Probe stopped", 1500)

    def _on_clear(self) -> None:
        self._overlay.clear()
        self._set_active(None)
        self._on_status("Probes cleared", 1500)

    def _set_active(self, mode: Optional[str]) -> None:
        self._mode = mode
        for m, btn in (
            ("point", self._btn_point),
            ("line", self._btn_line),
            ("slice", self._btn_slice),
        ):
            btn.setProperty("active", "true" if m == mode else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------
    # Overlay callbacks
    # ------------------------------------------------------------------

    def _on_point_result(self, result: "PointProbeResult") -> None:
        self._on_status(result.summary(), 8000)
        self._set_active(None)
        if self._chain_point is not None:
            try:
                self._chain_point(result)
            except Exception:
                pass

    def _on_line_result(self, result: "LineProbeResult") -> None:
        msg = (
            f"Line probe — {result.n_samples} samples, "
            f"L = {result.total_length:.4g}"
        )
        self._on_status(msg, 8000)
        self._set_active(None)
        if self._chain_line is not None:
            try:
                self._chain_line(result)
            except Exception:
                pass

    def _on_plane_result(self, result: "PlaneProbeResult") -> None:
        msg = (
            f"Slice — n=({result.normal[0]:.2g}, {result.normal[1]:.2g}, "
            f"{result.normal[2]:.2g}), {result.n_points} points"
        )
        self._on_status(msg, 8000)
        if self._chain_plane is not None:
            try:
                self._chain_plane(result)
            except Exception:
                pass


# =====================================================================
# _ResizeFilter — Qt event filter that fires a callback on Resize
# =====================================================================


def _resize_filter_factory():
    """Return a QObject subclass set up for Resize-event forwarding.

    Defined inside a factory because QObject is lazy-imported through
    qtpy — declaring the subclass at module scope would force qtpy
    to import on module load, which the rest of the package avoids.
    """
    from qtpy import QtCore

    class _ResizeFilter(QtCore.QObject):
        def __init__(self, callback: Callable[[], None]) -> None:
            super().__init__()
            self._cb = callback

        def eventFilter(self, _obj, event):   # noqa: N802 — Qt naming
            if event.type() == QtCore.QEvent.Resize:
                try:
                    self._cb()
                except Exception:
                    pass
            return False
    return _ResizeFilter


def _ResizeFilter(callback: Callable[[], None]):  # noqa: N802 — match Qt conv.
    return _resize_filter_factory()(callback)
