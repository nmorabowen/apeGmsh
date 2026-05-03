"""ResultsWindow — Qt shell for the post-solve viewer.

Composes :class:`ViewerWindow` and arranges its central widget +
side docks as a Qt-native dock layout:

::

    ┌──────────────────────────────────────────────────────────────┐
    │ (OS native title bar — filename / minimise / maximise / ×)   │
    ├────────────┬─────────────────────────────┬───────────────────┤
    │ Outline    │                             │ Plots             │
    │ (dock,     │   3D viewport               │ Details           │
    │  left)     │   (central widget)          │ Session           │
    │            │                             │ (docks, right)    │
    ├────────────┴─────────────────────────────┴───────────────────┤
    │ Time Scrubber (dock, bottom — Movable | Floatable, no close) │
    └──────────────────────────────────────────────────────────────┘

The five side panels are :class:`QDockWidget` instances: movable,
floatable, tabifiable, with object names so layout state round-trips
through ``QSettings``. The viewport is the immovable central widget.
There is **no** custom in-window title bar — the OS-supplied window
title bar already shows the filename, and any utility actions live
in the left vertical toolbar (screenshot, camera presets) or the
``Session`` dock (theme picker, …).

Layout persists across launches under ``QSettings('apeGmsh',
'ResultsViewer')`` keys ``layout/state`` and ``layout/geometry``.
``reset_layout()`` restores the default arrangement captured at
startup.

Public API consumed by :class:`ResultsViewer`:
``plotter``, ``window``, ``set_status``, ``exec``,
``set_left_widget``, ``set_right_widget``, ``set_details_widget``,
``set_session_widget``, ``set_bottom_widget``, ``reset_layout``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ._layout_metrics import LAYOUT
from .viewer_window import ViewerWindow


class ResultsWindow:
    """Results-viewer-specific window shell.

    Wraps a :class:`ViewerWindow` and arranges its central widget +
    side docks: viewport as the central widget, with Outline (left),
    Plots / Details / Session (right), and Time Scrubber (bottom) as
    movable :class:`QDockWidget` instances.

    Parameters
    ----------
    title
        Window title (shown in the OS title bar).
    on_close
        Optional callback invoked when the window is closed.
    """

    # Layout schema — bumped whenever the dock set changes (added /
    # removed / renamed). Saved layout state is only restored if the
    # stored version matches; mismatched state is discarded so users
    # don't get a half-broken arrangement after a structural change.
    _LAYOUT_SCHEMA_VERSION = 5

    def __init__(
        self,
        *,
        title: str = "Results",
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        # Wrap user-supplied on_close so we always persist layout on shutdown.
        # ViewerWindow's _MainWindow.closeEvent calls on_close *before* tearing
        # down the VTK interactor, so saveState() still has a live window.
        user_on_close = on_close

        def _wrapped_on_close() -> None:
            try:
                self._save_layout()
            except Exception:
                pass
            if user_on_close is not None:
                user_on_close()

        self._vw = ViewerWindow(title=title, on_close=_wrapped_on_close)

        # Populated by _build_layout()
        self._dock_left: Any = None
        self._dock_right: Any = None
        self._dock_diagram: Any = None
        self._dock_geometry: Any = None
        self._dock_details: Any = None
        self._dock_session: Any = None
        self._dock_bottom: Any = None
        # Host widgets inside each dock's QScrollArea — content swap target.
        self._left_host: Any = None
        self._right_host: Any = None
        self._diagram_host: Any = None
        self._geometry_host: Any = None
        self._details_host: Any = None
        self._session_host: Any = None
        self._bottom_host: Any = None
        # Default state captured after layout is built — target for reset_layout.
        self._default_layout_state: Any = None
        self._default_layout_geometry: Any = None
        # Focus-mode snapshot — populated when Ctrl+H hides everything,
        # consumed when Ctrl+H restores. None = currently NOT in focus mode.
        self._focus_state: Any = None
        # Shortcut objects kept on the instance so Qt's parent-tracked
        # GC doesn't reap them.
        self._focus_shortcut: Any = None
        self._close_shortcut: Any = None

        self._build_layout()

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

    def on_theme_changed(self, callback) -> None:
        """Register a callback fired with the new ``Palette`` on theme change.

        Forwards to the wrapped :class:`ViewerWindow` so consumers
        (e.g. :class:`ResultsViewer` updating substrate colors) don't
        need to reach into private state.
        """
        self._vw.on_theme_changed(callback)

    def set_bottom_widget(self, widget) -> None:
        """Mount a widget in the bottom (time-scrubber) dock."""
        self._set_host_widget(self._bottom_host, widget)

    def set_left_widget(self, widget) -> None:
        """Mount a widget in the left (Outline) dock."""
        self._set_host_widget(self._left_host, widget)
        # Hide the dock while empty so it doesn't reserve dead space.
        self._dock_left.setVisible(widget is not None)

    def set_right_widget(self, widget) -> None:
        """Mount a widget in the right (Plots) dock."""
        self._set_host_widget(self._right_host, widget)
        self._dock_right.setVisible(widget is not None)

    def set_details_widget(self, widget) -> None:
        """Mount a widget in the right-side Details dock (tabified with Plots)."""
        self._set_host_widget(self._details_host, widget)
        self._dock_details.setVisible(widget is not None)

    def set_diagram_widget(self, widget) -> None:
        """Mount a widget in the right-side Diagram dock (layer stack)."""
        self._set_host_widget(self._diagram_host, widget)
        self._dock_diagram.setVisible(widget is not None)

    def set_geometry_widget(self, widget) -> None:
        """Mount a widget in the right-side Geometry dock (geometry settings)."""
        self._set_host_widget(self._geometry_host, widget)
        self._dock_geometry.setVisible(widget is not None)

    def set_session_widget(self, widget) -> None:
        """Mount a widget in the right-side Session dock (viewer-level settings)."""
        self._set_host_widget(self._session_host, widget)
        self._dock_session.setVisible(widget is not None)

    def raise_diagram_dock(self) -> None:
        """Bring the Diagram dock to the front of its tab strip."""
        if self._dock_diagram is not None:
            try:
                self._dock_diagram.raise_()
            except Exception:
                pass

    def raise_geometry_dock(self) -> None:
        """Bring the Geometry dock to the front of its tab strip."""
        if self._dock_geometry is not None:
            try:
                self._dock_geometry.raise_()
            except Exception:
                pass

    def raise_details_dock(self) -> None:
        """Bring the Details dock to the front of its tab strip."""
        if self._dock_details is not None:
            try:
                self._dock_details.raise_()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Replace the wrapped ViewerWindow's central+dock layout.

        - VTK interactor stays where ``ViewerWindow`` put it: as the
          window's central widget.
        - Outline / Plots / Details / Session / Time Scrubber become
          :class:`QDockWidget` instances on the left, right, and bottom
          dock areas.
        - The legacy ``_tabs_dock`` from :class:`ViewerWindow` is removed
          from the layout (its widget stays alive for any external reader).
        """
        from qtpy import QtWidgets, QtCore

        win = self._vw.window

        # ── Retire the legacy right-side tabs dock ──────────────────
        # ViewerWindow installs its own right-side QDockWidget for tabs.
        # We're claiming that area for the Right Rail dock, so remove
        # the legacy one from the layout. Do not delete — the underlying
        # QTabWidget may still be referenced by other code paths.
        legacy_dock = getattr(self._vw, "_tabs_dock", None)
        if legacy_dock is not None:
            try:
                win.removeDockWidget(legacy_dock)
                legacy_dock.hide()
            except Exception:
                pass

        # ── Allow nested + tabified docks for full user control ─────
        win.setDockNestingEnabled(True)
        win.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AnimatedDocks
            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
        )
        # When the user drags one dock onto another's title bar to tabify
        # them, place the tab strip on the LEFT edge of every dock area
        # (sidebar style — consistent regardless of which side the area
        # sits on). Tab text stays horizontal via the proxy style
        # applied below to any QTabBar QMainWindow creates for tabified
        # docks.
        for area in (
            QtCore.Qt.LeftDockWidgetArea,
            QtCore.Qt.RightDockWidgetArea,
            QtCore.Qt.TopDockWidgetArea,
            QtCore.Qt.BottomDockWidgetArea,
        ):
            win.setTabPosition(area, QtWidgets.QTabWidget.TabPosition.West)

        # Apply the horizontal-text proxy style to any QTabBar that
        # QMainWindow creates for tabified docks. The bars don't exist
        # until two docks get tabified, so we hook the QMainWindow's
        # showEvent / a QTimer to scan for them once tabification has
        # happened. For now, install on existing bars too (covers the
        # restored-layout case where docks are already tabified).
        from .viewer_window import _make_horizontal_tab_style
        self._htab_style = _make_horizontal_tab_style()
        # Parent the style to the window so Qt's GC keeps it alive.
        self._htab_style.setParent(win)

        def _apply_htab_style_to_tabbars() -> None:
            for tb in win.findChildren(QtWidgets.QTabBar):
                # Only restyle bars whose shape is West/East (vertical).
                shape = tb.shape()
                if shape in (
                    QtWidgets.QTabBar.Shape.RoundedWest,
                    QtWidgets.QTabBar.Shape.RoundedEast,
                    QtWidgets.QTabBar.Shape.TriangularWest,
                    QtWidgets.QTabBar.Shape.TriangularEast,
                ):
                    tb.setStyle(self._htab_style)

        # Apply now (covers restored layouts with already-tabified docks)
        # and again shortly after Qt finishes laying out.
        _apply_htab_style_to_tabbars()
        QtCore.QTimer.singleShot(0, _apply_htab_style_to_tabbars)
        self._apply_htab_style_to_tabbars = _apply_htab_style_to_tabbars

        # Catch tab bars created LATER (when the user drags one dock onto
        # another to tabify them at runtime). QMainWindow creates the
        # QTabBar as a child as part of that operation, so we listen for
        # ChildAdded events and restyle any new tabbar.
        outer_self = self

        class _TabBarChildFilter(QtCore.QObject):
            def eventFilter(self, _obj, event):
                if event.type() == QtCore.QEvent.Type.ChildAdded:
                    child = event.child()
                    if isinstance(child, QtWidgets.QTabBar):
                        # Defer one tick — the tab bar's shape is set
                        # *after* the ChildAdded event fires.
                        QtCore.QTimer.singleShot(
                            0, outer_self._apply_htab_style_to_tabbars,
                        )
                return False

        self._tabbar_filter = _TabBarChildFilter(win)
        win.installEventFilter(self._tabbar_filter)

        # ── Five docks ──────────────────────────────────────────────
        QDW = QtWidgets.QDockWidget
        movable_floatable = (
            QDW.DockWidgetFeature.DockWidgetMovable
            | QDW.DockWidgetFeature.DockWidgetFloatable
        )
        with_close = movable_floatable | QDW.DockWidgetFeature.DockWidgetClosable

        self._dock_left, self._left_host = self._make_dock(
            "Outline", "dock_results_outline",
            min_width=LAYOUT.outline_min_width,
            features=with_close,
        )
        self._dock_left.setVisible(False)  # empty until ResultsViewer mounts it

        self._dock_right, self._right_host = self._make_dock(
            "Plots", "dock_results_right",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_right.setVisible(False)  # empty until ResultsViewer mounts it

        self._dock_diagram, self._diagram_host = self._make_dock(
            "Diagram", "dock_results_diagram",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_diagram.setVisible(False)  # empty until ResultsViewer mounts it

        self._dock_geometry, self._geometry_host = self._make_dock(
            "Geometry", "dock_results_geometry",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_geometry.setVisible(False)  # empty until ResultsViewer mounts it

        self._dock_details, self._details_host = self._make_dock(
            "Details", "dock_results_details",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_details.setVisible(False)  # empty until ResultsViewer mounts it

        # Session: viewer-level settings (theme picker, ...). Tabified
        # by default with Plots / Details on the right side; user can
        # detach.
        self._dock_session, self._session_host = self._make_dock(
            "Session", "dock_results_session",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_session.setVisible(False)  # empty until ResultsViewer mounts it

        # Bottom: time scrubber. NOT closable — losing the playhead is a
        # usability footgun. Floatable so power users can pop it out.
        self._dock_bottom, self._bottom_host = self._make_dock(
            "Time Scrubber", "dock_results_scrubber",
            min_height=LAYOUT.scrubber_min_height,
            features=movable_floatable,
        )

        win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._dock_left)
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_right)
        # Split the right area so Plots (top) and the dock-cluster
        # (bottom) are independent. Diagram / Geometry / Details /
        # Session start tabified together at the bottom; the user can
        # drag any of them out to detach or re-arrange.
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_diagram)
        win.splitDockWidget(
            self._dock_right, self._dock_diagram, QtCore.Qt.Vertical,
        )
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_geometry)
        win.tabifyDockWidget(self._dock_diagram, self._dock_geometry)
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_details)
        win.tabifyDockWidget(self._dock_diagram, self._dock_details)
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_session)
        win.tabifyDockWidget(self._dock_diagram, self._dock_session)
        # Keep Diagram as the visible tab on first launch.
        self._dock_diagram.raise_()

        win.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._dock_bottom)

        # Initial widths from LayoutMetrics.
        try:
            win.resizeDocks(
                [self._dock_left, self._dock_right],
                [LAYOUT.outline_initial_width, LAYOUT.right_initial_width],
                QtCore.Qt.Horizontal,
            )
        except Exception:
            pass

        # ── Capture default layout, then restore prior user layout ──
        try:
            self._default_layout_state = win.saveState()
            self._default_layout_geometry = win.saveGeometry()
        except Exception:
            self._default_layout_state = None
            self._default_layout_geometry = None
        self._restore_layout()

        # ── Focus-mode shortcut (Ctrl+H) ────────────────────────────
        self._install_focus_shortcut()

    def _make_dock(
        self,
        title: str,
        object_name: str,
        *,
        min_width: int = 0,
        min_height: int = 0,
        features,
    ):
        """Build a ``QDockWidget`` whose content is a scrollable host widget.

        Returns ``(dock, host)``. Mount content with
        ``self._set_host_widget(host, widget)``.
        """
        from qtpy import QtWidgets, QtCore

        host = QtWidgets.QWidget()
        host.setObjectName(f"{object_name}_host")
        host_lay = QtWidgets.QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

        dock = QtWidgets.QDockWidget(title)
        dock.setObjectName(object_name)
        dock.setWidget(scroll)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(features)
        if min_width:
            dock.setMinimumWidth(min_width)
        if min_height:
            dock.setMinimumHeight(min_height)
        return dock, host

    @staticmethod
    def _set_host_widget(host, widget) -> None:
        """Replace whatever's currently inside ``host`` with ``widget``.

        The widget is added with stretch=1 and an Expanding size policy so
        it grows to fill the dock — without this, panels with default
        ``Preferred`` policy sit at the top with empty space below.
        """
        from qtpy import QtWidgets
        layout = host.layout()
        while layout.count():
            item = layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
        if widget is not None:
            try:
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                )
            except Exception:
                pass
            layout.addWidget(widget, stretch=1)

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _layout_settings():
        """Return the QSettings handle used for dock layout persistence."""
        from qtpy.QtCore import QSettings
        return QSettings("apeGmsh", "ResultsViewer")

    def _save_layout(self) -> None:
        """Persist the current dock arrangement and window geometry."""
        win = self._vw.window
        s = self._layout_settings()
        try:
            s.setValue("layout/schema_version", self._LAYOUT_SCHEMA_VERSION)
            s.setValue("layout/geometry", win.saveGeometry())
            s.setValue("layout/state", win.saveState())
        except Exception:
            pass

    def _restore_layout(self) -> None:
        """Restore the user's last dock arrangement, if any.

        Skips restoration if the stored schema version doesn't match the
        current build — applying a v1 state to a v2 layout produces
        broken arrangements (e.g. tabified docks getting un-tabbed).
        """
        win = self._vw.window
        s = self._layout_settings()

        stored_version = s.value("layout/schema_version")
        try:
            stored_version_int = int(stored_version) if stored_version is not None else None
        except (TypeError, ValueError):
            stored_version_int = None
        if stored_version_int != self._LAYOUT_SCHEMA_VERSION:
            # Stored state is from a different layout schema — discard.
            return

        geom = s.value("layout/geometry")
        state = s.value("layout/state")
        if geom is not None:
            try:
                win.restoreGeometry(geom)
            except Exception:
                pass
        if state is not None:
            try:
                win.restoreState(state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Focus mode (Ctrl+H — hide every dock + left toolbar, then restore)
    # ------------------------------------------------------------------

    def _install_focus_shortcut(self) -> None:
        """Wire Ctrl+H to :meth:`toggle_focus_mode` and Q to close.

        Uses ``Qt.ApplicationShortcut`` because the VTK ``QtInteractor``
        intercepts keyboard events at native level — Qt's default
        ``WindowShortcut`` would never see the keys while the viewport
        has focus.
        """
        from qtpy import QtWidgets, QtGui, QtCore
        sc = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+H"), self._vw.window)
        sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(self.toggle_focus_mode)
        self._focus_shortcut = sc

        # Bare-Q closes the window. Bare keys are aggressive — a Spin/
        # LineEdit being focused would still receive the key as input
        # under ApplicationShortcut, so we guard with a focus check.
        sc_q = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self._vw.window)
        sc_q.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        sc_q.activated.connect(self._on_q_pressed)
        self._close_shortcut = sc_q

    def _on_q_pressed(self) -> None:
        """Close the window — but skip if the user is typing in a field."""
        from qtpy import QtWidgets
        fw = QtWidgets.QApplication.focusWidget()
        if isinstance(fw, (
            QtWidgets.QLineEdit,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QComboBox,
        )):
            return
        try:
            self._vw.window.close()
        except Exception:
            pass

    def toggle_focus_mode(self) -> None:
        """Hide every dock + the left toolbar (focus mode), or restore.

        On first call, snapshots which docks / toolbar were visible
        and hides them. On second call, restores exactly that snapshot
        — so a user who manually hid (e.g.) the Session dock before
        entering focus mode comes back to the same arrangement.
        """
        docks = (
            self._dock_left,
            self._dock_right,
            self._dock_diagram,
            self._dock_geometry,
            self._dock_details,
            self._dock_session,
            self._dock_bottom,
        )
        toolbar = getattr(self._vw, "_toolbar", None)

        if self._focus_state is None:
            # Currently in normal mode — capture state, hide everything.
            self._focus_state = {
                "docks": [d for d in docks if d is not None and d.isVisible()],
                "toolbar_visible": (
                    toolbar.isVisible() if toolbar is not None else False
                ),
            }
            for d in docks:
                if d is not None:
                    d.setVisible(False)
            if toolbar is not None:
                toolbar.setVisible(False)
            try:
                self.set_status("Focus mode — Ctrl+H to restore", 3000)
            except Exception:
                pass
        else:
            # Restore the snapshot.
            for d in self._focus_state["docks"]:
                if d is not None:
                    d.setVisible(True)
            if toolbar is not None:
                toolbar.setVisible(self._focus_state["toolbar_visible"])
            self._focus_state = None
            try:
                self.set_status("Focus mode off", 2000)
            except Exception:
                pass

    def reset_layout(self) -> None:
        """Restore the default dock arrangement captured at startup."""
        win = self._vw.window
        if self._default_layout_state is not None:
            try:
                win.restoreState(self._default_layout_state)
            except Exception:
                pass
        # Re-show every dock — a previous run may have closed one via
        # the title-bar × button.
        for dock in (
            self._dock_left,
            self._dock_right,
            self._dock_details,
            self._dock_session,
            self._dock_bottom,
        ):
            if dock is not None:
                dock.setVisible(True)
        try:
            self.set_status("Layout reset to default", 3000)
        except Exception:
            pass

