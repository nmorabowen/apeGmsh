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

The class is built up across phases. **B0 lands here**: title bar +
viewport + scrubber row. The left and right columns are empty until
B1 (outline tree) and B2 (plot pane) ship. The existing right-side
QTabWidget dock from :class:`ViewerWindow` continues to host the
Stages / Diagrams / Settings / Inspector / Probes tabs through B0;
B1 retires that dock when the outline tree replaces it.

ResultsWindow forwards the small API surface that
:class:`ResultsViewer` consumes (``plotter``, ``window``,
``add_tab``, ``set_status``, ``exec``) to the wrapped ViewerWindow,
so the rest of the viewer is oblivious to the shell change.
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

    def add_tab(self, name: str, widget) -> None:
        """Add a tab to the right-side panel.

        B0: forwards to ViewerWindow's tabs dock — preserves the
        existing UX while the new shell takes shape. B1 will replace
        this with placement into the outline tree's left column.
        """
        self._vw.add_tab(name, widget)

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
        """Row-0 widget — breadcrumb-style title text on a raised band."""
        from qtpy import QtWidgets, QtCore

        bar = QtWidgets.QFrame()
        bar.setObjectName("ResultsTitleBar")
        bar.setFixedHeight(self._TITLE_HEIGHT)
        bar.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(8)

        label = QtWidgets.QLabel(title)
        label.setObjectName("ResultsTitleLabel")
        label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        lay.addWidget(label)
        lay.addStretch(1)

        # Lightweight inline style — not yet plumbed through the theme
        # system. B4 ports RV_TOKENS and replaces this with palette-
        # driven styling.
        bar.setStyleSheet(
            "#ResultsTitleBar { "
            "background: rgba(255, 255, 255, 0.04); "
            "border-bottom: 1px solid rgba(255, 255, 255, 0.08); "
            "} "
            "#ResultsTitleLabel { "
            "color: rgba(255, 255, 255, 0.9); "
            "font-size: 12px; "
            "}"
        )
        self._title_label = label
        return bar

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
