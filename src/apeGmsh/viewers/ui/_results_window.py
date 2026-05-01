"""ResultsWindow — Qt shell for the post-solve viewer (B++ design).

Composes :class:`ViewerWindow` and replaces its single-widget central
area with a grid that follows the B++ Implementation Guide:

::

    ┌──────────────────────────────────────────────────────────────┐
    │ title bar                              row 0 · 40px · span 3 │
    ├────────────┬─────────────────────────────┬───────────────────┤
    │ tree       │  3D viewport (centerpiece)  │ plot pane         │
    │ (260px)    │                             │ (380px)           │
    ├────────────┴─────────────────────────────┴───────────────────┤
    │ time scrubber dock                     row 2 · 84px · span 3 │
    └──────────────────────────────────────────────────────────────┘

The class is built up across phases. B0 shipped the title bar +
viewport + scrubber row. B1–B5 filled in the left rail (outline
tree), the right rail (plot pane + details panel), the viewport
HUDs (probe palette, pick readout), and the title-bar breadcrumb.
The legacy right-side QTabWidget dock inherited from
:class:`ViewerWindow` is hidden in B5 — the grid's third cell now
owns the right column directly.

ResultsWindow forwards the small API surface that
:class:`ResultsViewer` consumes (``plotter``, ``window``,
``set_status``, ``exec``) to the wrapped ViewerWindow, so the rest
of the viewer is oblivious to the shell change.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .viewer_window import ViewerWindow


class ResultsWindow:
    """Results-viewer-specific window shell.

    Wraps a :class:`ViewerWindow` and rebuilds its central widget into
    a 3-row grid: title bar, body row (tree | viewport | plot pane),
    bottom scrubber row.

    Parameters
    ----------
    title
        Window title.
    on_close
        Optional callback invoked when the window is closed.
    """

    # Spec dimensions (B++ Implementation Guide §3 "Grid spec").
    _TITLE_HEIGHT = 40
    _SCRUBBER_HEIGHT = 84
    _LEFT_WIDTH = 260
    _RIGHT_WIDTH = 380

    def __init__(
        self,
        *,
        title: str = "Results",
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._vw = ViewerWindow(title=title, on_close=on_close)
        self._title_text = title

        # Populated by _build_grid()
        self._title_bar: Any = None
        self._left_holder: Any = None
        self._right_holder: Any = None
        self._bottom_holder: Any = None

        self._build_grid()

    # ------------------------------------------------------------------
    # Public API (forwards / new)
    # ------------------------------------------------------------------

    @property
    def plotter(self):
        """The PyVista QtInteractor (plotter)."""
        return self._vw.plotter

    @property
    def window(self):
        """The underlying QMainWindow."""
        return self._vw.window

    def set_status(self, text: str, timeout: int = 0) -> None:
        self._vw.set_status(text, timeout)

    def exec(self) -> int:
        return self._vw.exec()

    def set_bottom_widget(self, widget) -> None:
        """Mount a widget in the bottom scrubber row of the grid."""
        self._set_holder_widget(self._bottom_holder, widget)

    def set_left_widget(self, widget) -> None:
        """Mount a widget in the left (tree) column. Used in B1+."""
        self._set_holder_widget(self._left_holder, widget)
        # The left column is empty during B0; show it once populated
        # so the layout doesn't reserve dead space.
        self._left_holder.setVisible(widget is not None)

    def set_right_widget(self, widget) -> None:
        """Mount a widget in the right (plot pane) column. Used in B2+."""
        self._set_holder_widget(self._right_holder, widget)
        self._right_holder.setVisible(widget is not None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        """Replace the wrapped ViewerWindow's central widget with the grid."""
        from qtpy import QtWidgets, QtCore

        win = self._vw.window
        plotter = self._vw.plotter
        interactor_widget = plotter.interactor

        # The wrapped ViewerWindow installs a right-side QDockWidget
        # holding its own QTabWidget. The B++ shell owns the right
        # column directly via the grid's third cell, so the dock is
        # dead weight — hide it to keep the QMainWindow from reserving
        # width for it. Removing the dock outright would also remove
        # the tab widget which other viewers still depend on; hiding
        # is sufficient.
        legacy_dock = getattr(self._vw, "_tabs_dock", None)
        if legacy_dock is not None:
            legacy_dock.hide()

        central = QtWidgets.QWidget()
        central.setObjectName("ResultsWindowCentral")
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        # Row 0 · title bar (span 3)
        self._title_bar = self._make_title_bar(self._title_text)
        grid.addWidget(self._title_bar, 0, 0, 1, 3)

        # Row 1 · left | viewport | right
        self._left_holder = self._make_holder(
            min_width=self._LEFT_WIDTH, name="ResultsLeftHolder",
        )
        self._left_holder.setVisible(False)   # empty until B1
        grid.addWidget(self._left_holder, 1, 0)

        # Re-parent the QtInteractor's widget into the grid. Adding it
        # to the layout reparents it to ``central`` automatically.
        grid.addWidget(interactor_widget, 1, 1)

        self._right_holder = self._make_holder(
            min_width=self._RIGHT_WIDTH, name="ResultsRightHolder",
        )
        self._right_holder.setVisible(False)   # empty until B2
        grid.addWidget(self._right_holder, 1, 2)

        # Row 2 · scrubber (span 3)
        self._bottom_holder = self._make_holder(
            min_height=self._SCRUBBER_HEIGHT, name="ResultsBottomHolder",
        )
        grid.addWidget(self._bottom_holder, 2, 0, 1, 3)

        # Column sizing — fixed left / right, 1fr center.
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        # Row sizing — fixed title / scrubber, 1fr center.
        grid.setRowMinimumHeight(0, self._TITLE_HEIGHT)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        grid.setRowMinimumHeight(2, self._SCRUBBER_HEIGHT)
        grid.setRowStretch(2, 0)

        win.setCentralWidget(central)

    def _make_title_bar(self, title: str):
        """Row-0 widget — stop-light dots, breadcrumb, utility icons.

        Per spec §6 ``TitleBarSpan3``: three colored circles on the
        left (purely decorative — Qt apps already get system
        close/minimize/maximize from the OS chrome), breadcrumb text
        in the middle, and a strip of utility icon buttons on the
        right (theme toggle, screenshot, density toggle, help).
        """
        from qtpy import QtWidgets, QtCore

        bar = QtWidgets.QFrame()
        bar.setObjectName("ResultsTitleBar")
        bar.setFixedHeight(self._TITLE_HEIGHT)
        bar.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(10)

        # ── Stop-light dots (decorative) ───────────────────────────
        dots_holder = QtWidgets.QWidget()
        dots_lay = QtWidgets.QHBoxLayout(dots_holder)
        dots_lay.setContentsMargins(0, 0, 0, 0)
        dots_lay.setSpacing(6)
        for color in ("#ff5f57", "#febc2e", "#28c840"):
            dot = QtWidgets.QLabel()
            dot.setObjectName("ResultsTitleDot")
            dot.setFixedSize(QtCore.QSize(11, 11))
            dot.setStyleSheet(
                f"background-color: {color}; border-radius: 5px;"
            )
            dots_lay.addWidget(dot)
        lay.addWidget(dots_holder)

        # ── Breadcrumb label ───────────────────────────────────────
        label = QtWidgets.QLabel(title)
        label.setObjectName("ResultsTitleLabel")
        label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        lay.addWidget(label)
        lay.addStretch(1)

        # ── Utility icon strip ─────────────────────────────────────
        icons_holder = QtWidgets.QWidget()
        icons_lay = QtWidgets.QHBoxLayout(icons_holder)
        icons_lay.setContentsMargins(0, 0, 0, 0)
        icons_lay.setSpacing(2)

        self._btn_theme = self._make_title_icon_btn(
            "☽", "Toggle theme (dark / light)", self._on_toggle_theme,
        )
        self._btn_screenshot = self._make_title_icon_btn(
            "⎙", "Copy screenshot to clipboard", self._on_screenshot,
        )
        self._btn_density = self._make_title_icon_btn(
            "⚡", "Toggle density (compact / comfortable)",
            self._on_toggle_density,
        )
        self._btn_help = self._make_title_icon_btn(
            "?", "Help / shortcuts", self._on_help,
        )
        for btn in (
            self._btn_theme, self._btn_screenshot,
            self._btn_density, self._btn_help,
        ):
            icons_lay.addWidget(btn)

        lay.addWidget(icons_holder)

        # Theme-driven styling lives in viewers/ui/theme.py
        # (build_stylesheet); the QMainWindow's stylesheet cascades
        # to this widget via its #ResultsTitleBar / #ResultsTitleLabel
        # / #ResultsTitleIconBtn object names.
        self._title_label = label
        return bar

    def _make_title_icon_btn(
        self, glyph: str, tooltip: str, callback,
    ):
        from qtpy import QtWidgets, QtCore
        btn = QtWidgets.QToolButton()
        btn.setObjectName("ResultsTitleIconBtn")
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setFixedSize(QtCore.QSize(26, 24))
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    # ------------------------------------------------------------------
    # Title-bar utility callbacks
    # ------------------------------------------------------------------

    def _on_toggle_theme(self) -> None:
        """Cycle through the registered themes (dark → paper → next)."""
        from .theme import THEME, PALETTES
        names = list(PALETTES.keys())
        if not names:
            return
        try:
            idx = names.index(THEME.current.name)
        except ValueError:
            idx = -1
        THEME.set_theme(names[(idx + 1) % len(names)])

    def _on_screenshot(self) -> None:
        """Forward to the wrapped ViewerWindow's clipboard screenshot."""
        try:
            self._vw._screenshot()
        except Exception:
            pass

    def _on_toggle_density(self) -> None:
        """Cycle compact / comfortable density."""
        try:
            from .density import DENSITY
        except Exception:
            return
        DENSITY.toggle()

    def _on_help(self) -> None:
        """Show a small modal with shortcuts + customisation entry points.

        ResultsViewer has no Session tab (the right dock retired in B5),
        so the Theme editor and Global preferences dialogs — which
        MeshViewer / ModelViewer expose via tab buttons — would be
        unreachable from here without a discovery hook. The help
        dialog doubles as that hook: clear shortcut reference plus two
        buttons launching the respective editors.
        """
        from qtpy import QtWidgets
        dlg = QtWidgets.QDialog(self.window)
        dlg.setWindowTitle("ResultsViewer — shortcuts")
        dlg.setModal(True)
        dlg.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        text = QtWidgets.QLabel(
            "<b>Pick + measure</b><br/>"
            "&nbsp;&nbsp;Click&nbsp;— select node / element<br/>"
            "&nbsp;&nbsp;Shift&nbsp;+&nbsp;click — open time-history "
            "in plot pane<br/><br/>"
            "<b>Layout</b><br/>"
            "&nbsp;&nbsp;Ctrl + Shift + L — collapse / restore left rail"
            "<br/>"
            "&nbsp;&nbsp;Ctrl + Shift + R — collapse / restore right "
            "rail<br/><br/>"
            "<b>Probe modes</b> (top-right HUD)<br/>"
            "&nbsp;&nbsp;Point / Line / Slice — single-click to "
            "activate, click again to stop."
        )
        text.setWordWrap(True)
        # QLabel auto-detects rich text from the <b>/<br> tags; no
        # explicit setTextFormat needed.
        layout.addWidget(text)

        # Customisation actions — same dialogs MeshViewer / ModelViewer
        # surface in their Session tabs.
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)

        btn_theme = QtWidgets.QPushButton("Theme editor…")
        def _open_theme():
            try:
                from .theme_editor_dialog import open_theme_editor
                open_theme_editor(self.window)
            except Exception:
                pass
        btn_theme.clicked.connect(_open_theme)
        btn_row.addWidget(btn_theme)

        btn_prefs = QtWidgets.QPushButton("Global preferences…")
        def _open_prefs():
            try:
                from .preferences_dialog import open_preferences_dialog
                open_preferences_dialog(self.window)
            except Exception:
                pass
        btn_prefs.clicked.connect(_open_prefs)
        btn_row.addWidget(btn_prefs)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_close.setDefault(True)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)
        dlg.exec_()

    def _make_holder(
        self,
        *,
        min_width: int = 0,
        min_height: int = 0,
        name: str = "",
    ):
        """Empty container widget for one grid cell."""
        from qtpy import QtWidgets

        w = QtWidgets.QWidget()
        if name:
            w.setObjectName(name)
        if min_width:
            w.setMinimumWidth(min_width)
        if min_height:
            w.setMinimumHeight(min_height)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        return w

    @staticmethod
    def _set_holder_widget(holder, widget) -> None:
        """Replace whatever's currently inside ``holder`` with ``widget``."""
        layout = holder.layout()
        while layout.count():
            item = layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
        if widget is not None:
            layout.addWidget(widget)

    def set_title_text(self, text: str) -> None:
        """Update the title-bar text. Hooked up in B5 (breadcrumb)."""
        if self._title_label is not None:
            self._title_label.setText(text)
