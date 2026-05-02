"""PlotPane — right-rail container for 2-D side / history plots (B++ design).

Layout (per ``B++ Implementation Guide.html`` §4.3)::

    ┌─ PLOT PANE ────────────────────────┐
    │ [●] u_x vs t · N412         [×]   │  ← active row
    │ [○] σ vs ε · E1284          [×]   │
    │ [○] Fibers · B-1284         [×]   │
    │ ──────────── + New plot ────────── │
    ├────────────────────────────────────┤
    │                                    │
    │       active body fills here       │
    │                                    │
    └────────────────────────────────────┘

Each tab is keyed by a hashable id supplied by the caller. The
ResultsViewer uses two key shapes:

* side panels   ``("diagram", id(diagram))``
* time history  ``("history", node_id, component)``

The ``+ New plot`` row is a placeholder during B2 — B5 wires it to
an inline picker that opens a new time-history plot for the current
inspector selection.
"""
from __future__ import annotations

from typing import Any, Callable, Hashable, Optional

from ._layout_metrics import LAYOUT


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class PlotPane:
    """Vertical-list tab pane hosting 2-D plot widgets."""

    # Kept as a class attribute for backwards-compat with any external
    # readers; sourced from LayoutMetrics.
    _ROW_HEIGHT = LAYOUT.plot_row_height

    def __init__(self) -> None:
        QtWidgets, QtCore = _qt()

        widget = QtWidgets.QWidget()
        widget.setObjectName("PlotPane")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        header = QtWidgets.QFrame()
        header.setObjectName("PlotPaneHeader")
        header.setFixedHeight(LAYOUT.panel_header_height)
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 0, 6, 0)
        header_lay.setSpacing(6)
        title = QtWidgets.QLabel("PLOT PANE")
        title.setObjectName("PlotPaneHeaderLabel")
        header_lay.addWidget(title)
        header_lay.addStretch(1)
        outer.addWidget(header)

        # ── Tab list (scrollable, capped height) ───────────────────
        # A QScrollArea wrapping a QWidget with QVBoxLayout. Each tab
        # is a custom row (dot · label · ×). Only the rows beyond the
        # cap scroll; layout otherwise stays static.
        tab_holder = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab_holder)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)
        tab_layout.addStretch(1)   # pushes rows to the top
        self._tab_layout = tab_layout

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(tab_holder)
        scroll.setMaximumHeight(self._ROW_HEIGHT * LAYOUT.plot_tab_list_max_rows)
        outer.addWidget(scroll)

        # ── "+ New plot" row (B5 wires this) ───────────────────────
        new_plot_row = QtWidgets.QFrame()
        new_plot_row.setObjectName("PlotPaneNewPlot")
        new_plot_row.setFixedHeight(self._ROW_HEIGHT + 2)
        npl = QtWidgets.QHBoxLayout(new_plot_row)
        npl.setContentsMargins(10, 0, 10, 0)
        self._btn_new_plot = QtWidgets.QPushButton("+ New plot")
        self._btn_new_plot.setFlat(True)
        self._btn_new_plot.setEnabled(False)
        self._btn_new_plot.setToolTip("Inline plot picker — coming")
        npl.addWidget(self._btn_new_plot)
        npl.addStretch(1)
        outer.addWidget(new_plot_row)

        # ── Body (active plot widget) ──────────────────────────────
        self._body = QtWidgets.QStackedWidget()
        self._body.setObjectName("PlotPaneBody")
        outer.addWidget(self._body, stretch=1)

        # ── Empty state ────────────────────────────────────────────
        empty = QtWidgets.QLabel(
            "No plots yet.\n\n"
            "Add a diagram with a side panel (fiber, layer)\n"
            "or use the Inspector to open a time history."
        )
        empty.setAlignment(QtCore.Qt.AlignCenter)
        empty.setWordWrap(True)
        empty.setObjectName("PlotPaneEmpty")
        self._body.addWidget(empty)
        self._empty_widget = empty

        # Theme-driven styling lives in viewers/ui/theme.py.
        self._widget = widget
        self._tabs: list[_TabEntry] = []
        self._on_active_changed: Optional[Callable[[Optional[Hashable]], None]] = None
        # ``on_tabs_changed`` fires after every add/remove so external
        # navigators (the outline tree's Plots group) can stay in sync
        # with the tab list. List of callbacks — one-to-many.
        self._tabs_changed_observers: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def has_tab(self, key: Hashable) -> bool:
        return self._find(key) is not None

    def keys(self) -> list[Hashable]:
        return [t.key for t in self._tabs]

    def add_tab(
        self,
        key: Hashable,
        label: str,
        body: Any,
        *,
        closable: bool = True,
    ) -> None:
        """Add a new tab. The body widget is reparented into the pane.

        ``closable`` controls whether the row shows a ``×`` button.
        Side panels driven by the diagram registry pass ``False`` —
        their lifecycle is owned by the registry, not by the user.
        Time-history plots pass ``True``.
        """
        if self._find(key) is not None:
            self.set_active(key)
            return

        row = _TabRow(
            label,
            on_click=lambda: self.set_active(key),
            on_close=lambda: self._user_close(key),
            closable=closable,
        )
        # Insert just before the trailing stretch.
        self._tab_layout.insertWidget(self._tab_layout.count() - 1, row.widget)
        body_idx = self._body.addWidget(body)

        self._tabs.append(_TabEntry(
            key=key, label=label, row=row, body=body, body_index=body_idx,
        ))
        self.set_active(key)
        self._fire_tabs_changed()

    def remove_tab(self, key: Hashable) -> None:
        entry = self._find(key)
        if entry is None:
            return
        was_active = (
            self._body.currentIndex() == entry.body_index
        )
        self._tab_layout.removeWidget(entry.row.widget)
        entry.row.widget.setParent(None)
        entry.row.widget.deleteLater()
        self._body.removeWidget(entry.body)
        self._tabs.remove(entry)
        # The body widget is now orphaned — caller is responsible for
        # closing/destroying the panel that hosted it (their .close()
        # is what releases plotter callbacks etc).
        if was_active:
            if self._tabs:
                self.set_active(self._tabs[-1].key)
            else:
                self._body.setCurrentWidget(self._empty_widget)
                if self._on_active_changed is not None:
                    self._on_active_changed(None)
        self._fire_tabs_changed()

    def set_active(self, key: Hashable) -> None:
        entry = self._find(key)
        if entry is None:
            return
        self._body.setCurrentIndex(entry.body_index)
        for t in self._tabs:
            t.row.set_active(t is entry)
        if self._on_active_changed is not None:
            self._on_active_changed(key)

    def active_key(self) -> Optional[Hashable]:
        idx = self._body.currentIndex()
        for t in self._tabs:
            if t.body_index == idx:
                return t.key
        return None

    def on_active_changed(
        self, callback: Callable[[Optional[Hashable]], None],
    ) -> None:
        self._on_active_changed = callback

    def on_tabs_changed(
        self, callback: Callable[[], None],
    ) -> Callable[[], None]:
        """Register observer fired after every add_tab / remove_tab.

        Returns an unsubscribe callable. Multi-listener — pile on as
        many navigators as you like.
        """
        self._tabs_changed_observers.append(callback)

        def _unsub() -> None:
            try:
                self._tabs_changed_observers.remove(callback)
            except ValueError:
                pass
        return _unsub

    def _fire_tabs_changed(self) -> None:
        for cb in list(self._tabs_changed_observers):
            try:
                cb()
            except Exception:
                import logging
                logging.getLogger("apeGmsh.viewer.plot_pane").exception(
                    "tabs_changed observer failed: %r", cb,
                )

    def tab_label(self, key: Hashable) -> Optional[str]:
        """Return the visible label for a tab key, or ``None`` if absent."""
        entry = self._find(key)
        return entry.label if entry is not None else None

    def on_user_close(
        self, callback: Callable[[Hashable], None],
    ) -> None:
        """Register a callback fired when the user clicks a tab's ×.

        The callback is responsible for releasing the body widget's
        resources (e.g. ``panel.close()``) and calling
        :meth:`remove_tab`. Without it, the × is a no-op.
        """
        self._on_user_close = callback

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    _on_user_close: Optional[Callable[[Hashable], None]] = None

    def _find(self, key: Hashable) -> "Optional[_TabEntry]":
        for t in self._tabs:
            if t.key == key:
                return t
        return None

    def _user_close(self, key: Hashable) -> None:
        if self._on_user_close is not None:
            self._on_user_close(key)


# =====================================================================
# _TabRow — one row in the tab list (dot · label · close ×)
# =====================================================================


class _TabRow:
    def __init__(
        self,
        label: str,
        *,
        on_click: Callable[[], None],
        on_close: Callable[[], None],
        closable: bool = True,
    ) -> None:
        QtWidgets, QtCore = _qt()
        frame = QtWidgets.QFrame()
        frame.setObjectName("PlotPaneTabRow")
        frame.setFixedHeight(PlotPane._ROW_HEIGHT)
        frame.setProperty("active", False)
        frame.setCursor(QtCore.Qt.PointingHandCursor)

        lay = QtWidgets.QHBoxLayout(frame)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(6)

        dot = QtWidgets.QLabel("●")
        dot.setObjectName("PlotPaneTabDot")
        lay.addWidget(dot)

        text = QtWidgets.QLabel(label)
        text.setObjectName("PlotPaneTabLabel")
        text.setToolTip(label)
        text.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred,
        )
        lay.addWidget(text, stretch=1)

        if closable:
            close_btn = QtWidgets.QToolButton()
            close_btn.setText("×")
            close_btn.setObjectName("PlotPaneTabClose")
            close_btn.setAutoRaise(True)
            close_btn.setFixedSize(16, 16)
            close_btn.clicked.connect(lambda: on_close())
            lay.addWidget(close_btn)

        # Click anywhere on the row (except the × button) → activate.
        # mousePressEvent on the QFrame is the simplest hook.
        def _press(_event, _f=frame, _cb=on_click):
            _cb()
        frame.mousePressEvent = _press   # type: ignore[assignment]

        # Theme-driven styling lives in viewers/ui/theme.py.
        self._frame = frame

    @property
    def widget(self):
        return self._frame

    def set_active(self, active: bool) -> None:
        self._frame.setProperty("active", "true" if active else "false")
        # Force stylesheet re-evaluation after dynamic property change.
        self._frame.style().unpolish(self._frame)
        self._frame.style().polish(self._frame)


# =====================================================================
# _TabEntry — internal record
# =====================================================================


class _TabEntry:
    __slots__ = ("key", "label", "row", "body", "body_index")

    def __init__(self, *, key, label, row, body, body_index):
        self.key = key
        self.label = label
        self.row = row
        self.body = body
        self.body_index = body_index
