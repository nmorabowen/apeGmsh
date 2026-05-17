"""
ViewerWindow — Qt window shell hosting a VTK plotter.

Assembled from provided components (tabs, toolbar actions, docks).
Not a base class — takes everything it needs via constructor args.

Layout::

    ┌──────────────────────────────────────────────────┐
    │ Menu Bar                                         │
    ├──────┬───────────────────────────┬───────────────┤
    │ Tool │                           │               │
    │ bar  │    VTK Viewport           │  Tabs dock    │
    │      │                           │               │
    ├──────┴───────────────────────────┴───────────────┤
    │ Status Bar                                       │
    └──────────────────────────────────────────────────┘

Usage::

    from apeGmsh.viewers.ui.viewer_window import ViewerWindow
    win = ViewerWindow(
        title="Model Viewer",
        tabs=[("Browser", browser_widget), ("Prefs", prefs_widget)],
        toolbar_actions=[("New", icon, callback)],
        on_close=lambda: selection.flush_to_gmsh(),
    )
    plotter = win.plotter
    # ... build scene on plotter ...
    win.exec()
"""
from __future__ import annotations

import datetime
from typing import Any, Callable, Optional, Sequence

from ._dock_registry import (
    DockSpec,
    add_view_menu_toggle,
    build_view_menu,
    mount_dock_spec,
)
from .theme import THEME, build_stylesheet


def _lazy_qt():
    """Import Qt + pyvistaqt on first use."""
    try:
        from qtpy import QtWidgets, QtCore, QtGui
        from pyvistaqt import QtInteractor
    except ImportError as err:
        raise ImportError(
            "ViewerWindow requires pyvistaqt + a Qt binding "
            "(PyQt5/PyQt6/PySide2/PySide6).  Install with "
            "`pip install pyvistaqt PyQt5`."
        ) from err
    return QtWidgets, QtCore, QtGui, QtInteractor


def _make_horizontal_tab_style():
    """QProxyStyle that renders West/East tabs with horizontal labels.

    Qt's default ``TabPosition.West`` / ``East`` rotates the label 90°.
    This proxy transposes the tab-cell size hint (so tabs stack vertically
    but each cell is wide enough for its label) and forces the label to
    be drawn in the North (horizontal) shape.
    """
    from qtpy import QtWidgets

    class _HTabStyle(QtWidgets.QProxyStyle):
        def sizeFromContents(self, type_, opt, size, widget):
            s = super().sizeFromContents(type_, opt, size, widget)
            if type_ == QtWidgets.QStyle.ContentsType.CT_TabBarTab:
                s.transpose()
            return s

        def drawControl(self, element, opt, painter, widget):
            if element == QtWidgets.QStyle.ControlElement.CE_TabBarTabLabel:
                if isinstance(opt, QtWidgets.QStyleOptionTab):
                    new_opt = QtWidgets.QStyleOptionTab(opt)
                    new_opt.shape = QtWidgets.QTabBar.Shape.RoundedNorth
                    super().drawControl(element, new_opt, painter, widget)
                    return
            super().drawControl(element, opt, painter, widget)

    return _HTabStyle()


class ViewerWindow:
    """Qt window hosting a PyVista VTK plotter.

    Parameters
    ----------
    title : str
        Window title.
    tabs : list of (name, QWidget)
        Tabs to add to the right-side dock.
    extra_docks : list of QDockWidget, optional
        Additional dock widgets. **Legacy** path — prefer
        ``extension_docks`` for new code.
    toolbar_actions : list of (tooltip, icon_text, callback), optional
        Buttons inserted before the camera controls.
    on_close : callable, optional
        Called when the window is closed (e.g. flush groups to Gmsh).
    show_console : bool
        Whether to include the collapsible console dock.
    extension_docks : sequence of :class:`DockSpec`, optional
        Registry-driven extension docks mounted alongside the built-in
        tabs / console / extra_docks. Same machinery
        :class:`ResultsWindow` uses (plan 08 step 2) — gives
        ``mesh.viewer`` / ``model.viewer`` the same registry-driven
        path. ``tabify_with`` may reference :attr:`DOCK_TABS` /
        :attr:`DOCK_CONSOLE` (when present) or another extension
        ``dock_id``.
    window_key : str, optional
        QSettings sub-key under ``"apeGmsh"`` used to persist dock
        layout + window geometry across launches. When given, the
        View menu gains an enabled "Reset Layout" entry. When ``None``
        (default), the legacy non-persistent behaviour applies —
        every launch starts at the in-code default arrangement and
        "Reset Layout" is disabled.
    """

    # objectName constants for the built-in docks — exposed so
    # extension specs can ``tabify_with`` them by name. (Console only
    # exists when ``show_console=True``; tabify_with that id otherwise
    # will raise at mount time.)
    DOCK_TABS = "dock_viewer_tabs"
    DOCK_CONSOLE = "dock_viewer_console"

    # Bumped whenever the built-in dock set changes (added / removed /
    # renamed). Saved state from a different schema is ignored so a
    # users' restored layout never half-applies to a structurally
    # different window.
    #
    # v2 (2026-05-15): tabs_dock starts hidden when no tabs are
    # supplied; model.viewer migrated all its add_tab calls to
    # tabified extension docks, so v1 state for ModelViewer would
    # surface an empty placeholder dock on restore.
    _LAYOUT_SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        title: str = "Viewer",
        tabs: list[tuple[str, Any]] | None = None,
        extra_docks: list[Any] | None = None,
        toolbar_actions: list[tuple[str, str, Callable]] | None = None,
        on_close: Callable[[], None] | None = None,
        show_console: bool | None = None,
        extension_docks: Optional[Sequence[DockSpec]] = None,
        window_key: Optional[str] = None,
    ) -> None:
        QtWidgets, QtCore, QtGui, QtInteractor = _lazy_qt()
        if show_console is None:
            from .preferences_manager import PREFERENCES as _PREF_SC
            show_console = _PREF_SC.current.show_console
        self._QtWidgets = QtWidgets
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._title = title
        self._on_close = on_close
        # Layout persistence state. ``_default_layout_state`` and
        # ``_default_layout_geometry`` are captured after dock mount;
        # ``reset_layout()`` restores them. Both ``None`` until the
        # post-mount snapshot block runs.
        self._window_key = window_key
        self._default_layout_state: Any = None
        self._default_layout_geometry: Any = None

        # ── QApplication ────────────────────────────────────────────
        app = QtWidgets.QApplication.instance()
        self._own_app = app is None
        if self._own_app:
            app = QtWidgets.QApplication([])
        self._app = app

        # ── Window ──────────────────────────────────────────────────
        ui_self = self

        class _MainWindow(QtWidgets.QMainWindow):  # type: ignore[name-defined]  # Qt lazy-imported
            def closeEvent(self, event):
                # Persist layout BEFORE on_close so a user callback
                # that triggers teardown (e.g. flushes selection,
                # closes resources) doesn't race with saveState.
                try:
                    ui_self._save_layout()
                except Exception:
                    pass
                if ui_self._on_close is not None:
                    try:
                        ui_self._on_close()
                    except Exception as exc:
                        import sys
                        import traceback
                        print(
                            f"[viewer] on_close failed: {exc}",
                            file=sys.stderr,
                        )
                        traceback.print_exc(file=sys.stderr)
                try:
                    ui_self._unsub_theme()
                except Exception:
                    pass
                try:
                    ui_self._qt_interactor.close()
                except Exception:
                    pass
                super().closeEvent(event)

        self._window = _MainWindow()
        self._window.setWindowTitle(title)
        self._window.resize(1600, 1000)

        # Icon-action registry for live theme re-rendering
        self._icon_actions: list[tuple[Any, str]] = []
        # Observers to notify on theme change (VTK content, etc.)
        self._theme_callbacks: list[Callable[[Any], None]] = []

        # ── Central: VTK plotter ────────────────────────────────────
        self._qt_interactor = QtInteractor(parent=self._window)
        self._window.setCentralWidget(self._qt_interactor.interactor)

        from .preferences_manager import PREFERENCES as _PREF
        _p = _PREF.current

        import pyvista as _pv
        _pv.set_plot_theme("dark")
        _pv.global_theme.font.color = "white"
        if _p.anti_aliasing != "none":
            try:
                self._qt_interactor.enable_anti_aliasing(_p.anti_aliasing)
            except Exception:
                pass
        # ── Plan 06 step 5 — Depth peeling ──────────────────────────
        # Order-independent transparency. Without it, two overlapping
        # semi-transparent diagrams composite incorrectly depending on
        # camera angle (the rear actor can paint over the front).
        # The renderer attribute exists on every VTK/PyVista build
        # we ship; wrap in try/except to stay defensive against
        # older / stripped builds. Costs ~0 when nothing in the scene
        # is transparent.
        try:
            renderer = self._qt_interactor.renderer
            renderer.SetUseDepthPeeling(True)
            renderer.SetMaximumNumberOfPeels(4)
            renderer.SetOcclusionRatio(0.0)
        except Exception:
            pass
        _axes_kwargs = dict(
            interactive=False,
            line_width=_p.axis_line_width,
            color="white",            # label text
            # Shaft colors match the local-axes overlay convention
            # (x=red, y=green, z=blue) so the corner gnomon reads as a
            # world reference for sanity-checking element orientation
            # against the per-element triads.
            x_color="#FF4136",
            y_color="#2ECC40",
            z_color="#0074D9",
        )
        if _p.axis_labels_visible:
            _axes_kwargs.update(xlabel="X", ylabel="Y", zlabel="Z")
        else:
            _axes_kwargs.update(xlabel="", ylabel="", zlabel="")
        self._qt_interactor.add_axes(**_axes_kwargs)

        # ── Dock area defaults — vertical tab strip on tabified docks ──
        # When the user drags one dock onto another's title bar to
        # tabify them, Qt creates a QTabBar implicitly. Default is a
        # horizontal strip at the bottom; we override to a sidebar-
        # style West strip with horizontal text via a proxy style.
        # Mirrors :class:`ResultsWindow._build_layout`; lifted here so
        # mesh / model viewers get the same tabified-dock UX.
        self._window.setDockNestingEnabled(True)
        self._window.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AnimatedDocks
            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
        )
        for area in (
            QtCore.Qt.LeftDockWidgetArea,
            QtCore.Qt.RightDockWidgetArea,
            QtCore.Qt.TopDockWidgetArea,
            QtCore.Qt.BottomDockWidgetArea,
        ):
            self._window.setTabPosition(
                area, QtWidgets.QTabWidget.TabPosition.West,
            )

        # Horizontal-label proxy style applied to any QTabBar Qt
        # creates for tabified docks. Parent it to the window so Qt's
        # GC keeps it alive.
        self._htab_style = _make_horizontal_tab_style()
        self._htab_style.setParent(self._window)

        def _apply_htab_style_to_tabbars() -> None:
            for tb in self._window.findChildren(QtWidgets.QTabBar):
                shape = tb.shape()
                if shape in (
                    QtWidgets.QTabBar.Shape.RoundedWest,
                    QtWidgets.QTabBar.Shape.RoundedEast,
                    QtWidgets.QTabBar.Shape.TriangularWest,
                    QtWidgets.QTabBar.Shape.TriangularEast,
                ):
                    tb.setStyle(self._htab_style)

        _apply_htab_style_to_tabbars()
        QtCore.QTimer.singleShot(0, _apply_htab_style_to_tabbars)
        self._apply_htab_style_to_tabbars = _apply_htab_style_to_tabbars

        outer_self = self

        class _TabBarChildFilter(QtCore.QObject):
            def eventFilter(self, _obj, event):
                if event.type() == QtCore.QEvent.Type.ChildAdded:
                    child = event.child()
                    if isinstance(child, QtWidgets.QTabBar):
                        # Defer one tick — the tab bar's shape is set
                        # *after* the ChildAdded event fires.
                        QtCore.QTimer.singleShot(
                            0,
                            outer_self._apply_htab_style_to_tabbars,
                        )
                return False

        self._tabbar_filter = _TabBarChildFilter(self._window)
        self._window.installEventFilter(self._tabbar_filter)

        # ── Right dock: tabs (vertical labels, horizontal text) ─────
        _TAB_POS_MAP = {
            "left":   QtWidgets.QTabWidget.TabPosition.West,
            "right":  QtWidgets.QTabWidget.TabPosition.East,
            "top":    QtWidgets.QTabWidget.TabPosition.North,
            "bottom": QtWidgets.QTabWidget.TabPosition.South,
        }
        _tab_pos = _TAB_POS_MAP.get(
            _p.tab_position, QtWidgets.QTabWidget.TabPosition.West,
        )
        _is_vertical = _tab_pos in (
            QtWidgets.QTabWidget.TabPosition.West,
            QtWidgets.QTabWidget.TabPosition.East,
        )
        self._tab_widget = QtWidgets.QTabWidget()
        self._tab_widget.setTabPosition(_tab_pos)
        if _is_vertical:
            self._tab_widget.tabBar().setStyle(_make_horizontal_tab_style())
        for name, widget in (tabs or []):
            self._tab_widget.addTab(widget, name)

        tabs_dock = QtWidgets.QDockWidget()
        # objectName lets extension docks tabify with it via DOCK_TABS,
        # and survives Qt's saveState/restoreState if a layout-
        # persistence layer is ever added here.
        tabs_dock.setObjectName(self.DOCK_TABS)
        tabs_dock.setTitleBarWidget(QtWidgets.QWidget())  # hide title bar
        tabs_dock.setMinimumWidth(_p.dock_min_width)
        tabs_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        tabs_dock.setWidget(self._tab_widget)
        self._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, tabs_dock)
        self._tabs_dock = tabs_dock
        # Pre-populated tabs make the dock useful immediately; an
        # otherwise-empty tabs_dock reserves dead space when callers
        # route their panels through ``add_extension_dock`` exclusively.
        # ``add_tab`` un-hides the dock on first call.
        if not (tabs or []):
            tabs_dock.setVisible(False)

        # ── Extra docks ─────────────────────────────────────────────
        for dock in (extra_docks or []):
            self._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # ── Console ─────────────────────────────────────────────────
        self._console = None
        self._console_dock = None
        if show_console:
            console = QtWidgets.QTextEdit()
            console.setReadOnly(True)
            console.setFont(QtGui.QFont("Consolas", 9))
            self._console = console
            dock = QtWidgets.QDockWidget("Console")
            dock.setObjectName(self.DOCK_CONSOLE)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            dock.setWidget(console)
            self._window.addDockWidget(
                QtCore.Qt.BottomDockWidgetArea, dock,
            )
            dock.hide()
            self._console_dock = dock

        # Corner assignments
        self._window.setCorner(
            QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea,
        )
        self._window.setCorner(
            QtCore.Qt.TopRightCorner, QtCore.Qt.RightDockWidgetArea,
        )

        # ── Toolbar ─────────────────────────────────────────────────
        bar = QtWidgets.QToolBar("Tools")
        bar.setMovable(True)
        bar.setOrientation(QtCore.Qt.Vertical)
        bar.setFloatable(True)
        bar.setIconSize(QtCore.QSize(28, 28))
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        # Styling handled by global STYLESHEET

        # Subclass-provided actions first
        for tooltip, icon_text, callback in (toolbar_actions or []):
            self._add_toolbar_action(bar, icon_text, tooltip, callback)
        if toolbar_actions:
            bar.addSeparator()

        # Camera controls
        self._act_parallel = self._add_toolbar_action(
            bar, "\u2316", "Ortho / perspective toggle",
            self._toggle_parallel, checkable=True, triggered_signal="toggled",
        )

        self._add_toolbar_action(bar, "\u2922", "Fit view", self._fit_view)

        bar.addSeparator()

        for label, direction in [
            ("T", "top"), ("Bo", "bottom"), ("F", "front"),
            ("Bk", "back"), ("L", "left"), ("R", "right"),
            ("\u25E3", "iso"),
        ]:
            self._add_toolbar_action(
                bar, label, f"{direction.capitalize()} view",
                lambda _=False, d=direction: self._snap_view(d),
            )

        bar.addSeparator()

        self._add_toolbar_action(
            bar, "\u2399", "Copy screenshot to clipboard", self._screenshot,
        )
        self._add_toolbar_action(
            bar, "\u2913", "Save screenshot to file\u2026",
            self._save_screenshot,
        )

        self._toolbar = bar
        self._window.addToolBar(QtCore.Qt.LeftToolBarArea, bar)

        # ── Extension docks (plan 08 — registry-driven extras) ──────
        # Mounted before the View menu so each extension's toggle
        # appears alongside the console / extra_docks toggles in the
        # initial menu population.
        self._extension_specs: list[DockSpec] = (
            list(extension_docks) if extension_docks else []
        )
        self._extension_docks: dict[str, Any] = {}
        for spec in self._extension_specs:
            self._mount_extension_dock_inner(spec)

        # ── Menu bar (auto-populated toggles + Reset Layout) ────────
        # Built once if there are toggleable docks OR layout persistence
        # is enabled (so "Reset Layout" has somewhere to live). Stays
        # absent in the legacy non-persistent + no-toggleable-docks
        # case so we don't surface an empty View menu.
        self._view_menu: Any = None
        self._view_menu_reset_separator: Any = None
        toggleable_docks: list[Any] = []
        if self._console_dock is not None:
            toggleable_docks.append(self._console_dock)
        toggleable_docks.extend(extra_docks or [])
        for spec in self._extension_specs:
            dock = self._extension_docks.get(spec.dock_id)
            if dock is not None:
                toggleable_docks.append(dock)
        if toggleable_docks or self._window_key is not None:
            menu_bar = self._window.menuBar()
            on_reset = (
                self.reset_layout if self._window_key is not None else None
            )
            self._view_menu, self._view_menu_reset_separator = build_view_menu(
                menu_bar,
                docks=toggleable_docks,
                on_reset_layout=on_reset,
            )

        # ── Capture default layout, then restore prior user layout ──
        # ``_save_layout`` (called from closeEvent) reads
        # ``_window_key``; ``_restore_layout`` is a no-op when no key
        # is set, so the legacy non-persistent path stays free of
        # QSettings traffic.
        if self._window_key is not None:
            try:
                self._default_layout_state = self._window.saveState()
                self._default_layout_geometry = self._window.saveGeometry()
            except Exception:
                self._default_layout_state = None
                self._default_layout_geometry = None
            self._restore_layout()

        # ── Status bar ──────────────────────────────────────────────
        self._statusbar = self._window.statusBar()

        # ── Theme: initial apply + subscribe for live switching ────
        self._apply_palette(THEME.current)
        self._unsub_theme = THEME.subscribe(self._apply_palette)

        # ── Density: re-apply the QSS on toggle ────────────────────
        # Density only changes type / row sizing, so we just re-render
        # the stylesheet with the new tokens — same path as a theme
        # change.
        try:
            from .density import DENSITY
            self._unsub_density = DENSITY.subscribe(
                lambda _tok: self._apply_palette(THEME.current),
            )
        except Exception:
            self._unsub_density = lambda: None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def plotter(self):
        """The PyVista QtInteractor (plotter)."""
        return self._qt_interactor

    @property
    def window(self):
        return self._window

    def add_tab(self, name: str, widget) -> None:
        """Add a tab to the right-side panel (after construction).

        Un-hides the tabs dock so that callers who use ``add_tab``
        exclusively get the legacy "tab strip on the right" UI. Mixed
        callers (some tabs, some extension docks) also work — the
        first ``add_tab`` flips visibility on.
        """
        self._tab_widget.addTab(widget, name)
        if self._tabs_dock is not None and not self._tabs_dock.isVisible():
            self._tabs_dock.setVisible(True)

    def focus_tab(self, identifier: str) -> bool:
        """Bring a tab / extension dock to the front.

        Handles both right-side panel patterns: the legacy
        ``QTabWidget`` inside ``_tabs_dock`` (mesh.viewer today) and
        the tabified extension-dock cluster (model.viewer post-plan-08).
        Tries the extension-dock dictionary first (``identifier`` ==
        ``dock_id``); falls back to a tab-text match on ``_tab_widget``.

        Returns ``True`` if the identifier resolved to something
        raisable, ``False`` otherwise — callers can use the bool to
        log or no-op.
        """
        dock = self._extension_docks.get(identifier)
        if dock is not None:
            try:
                dock.setVisible(True)
                dock.raise_()
                return True
            except Exception:
                return False
        # Tab-text fallback for the legacy QTabWidget pattern.
        tw = getattr(self, "_tab_widget", None)
        if tw is not None:
            for i in range(tw.count()):
                if tw.tabText(i) == identifier:
                    tw.setCurrentIndex(i)
                    if self._tabs_dock is not None:
                        self._tabs_dock.setVisible(True)
                        try:
                            self._tabs_dock.raise_()
                        except Exception:
                            pass
                    return True
        return False

    # ------------------------------------------------------------------
    # Extension docks (plan 08 — registry-driven extras)
    # ------------------------------------------------------------------

    _BUILTIN_DOCK_IDS = frozenset({DOCK_TABS, DOCK_CONSOLE})

    def add_extension_dock(self, spec: DockSpec) -> Any:
        """Register and mount an extension dock at runtime.

        Mirrors :meth:`ResultsWindow.add_extension_dock` so mesh /
        model viewers can add panels through the same registry-driven
        path. Returns the mounted ``QDockWidget``. If the View menu
        doesn't yet exist (no prior toggleable docks, no layout
        persistence), it's created lazily so the new toggle has
        somewhere to land. The toggle inserts above the Reset Layout
        separator (when present) so Reset Layout stays pinned at the
        bottom.

        Raises
        ------
        ValueError
            If ``spec.dock_id`` collides with a built-in dock or a
            previously-registered extension, or ``spec.tabify_with``
            doesn't resolve to a mounted ``QDockWidget``.
        """
        dock = self._mount_extension_dock_inner(spec)
        self._extension_specs.append(spec)
        self._ensure_view_menu()
        add_view_menu_toggle(
            self._view_menu,
            self._view_menu_reset_separator,
            dock,
        )
        return dock

    def _mount_extension_dock_inner(self, spec: DockSpec) -> Any:
        """Build a ``QDockWidget`` from ``spec`` and dock it on the window.

        Internal helper shared by the constructor's extension-mount
        loop and :meth:`add_extension_dock`. Validates collisions
        against the built-in objectNames + previously-mounted
        extensions, then delegates to :func:`mount_dock_spec`. Does
        NOT touch ``_extension_specs`` or the View menu — callers
        handle those.
        """
        reserved = self._BUILTIN_DOCK_IDS | set(self._extension_docks)
        if spec.dock_id in self._BUILTIN_DOCK_IDS:
            raise ValueError(
                f"DockSpec.dock_id={spec.dock_id!r} collides with a "
                f"built-in ViewerWindow dock — pick a different id"
            )
        dock = mount_dock_spec(
            self._window, spec, reserved_ids=reserved,
        )
        self._extension_docks[spec.dock_id] = dock
        return dock

    def extension_dock(self, dock_id: str) -> Any:
        """Return the ``QDockWidget`` for an extension ``dock_id``.

        Raises ``KeyError`` if no extension is registered under that id.
        """
        return self._extension_docks[dock_id]

    # ------------------------------------------------------------------
    # Layout persistence (parallel to ResultsWindow's mechanism)
    # ------------------------------------------------------------------

    def _layout_settings(self):
        """Return the QSettings handle for layout persistence.

        Returns ``None`` when no ``window_key`` was supplied —
        callers must guard. Lazy import keeps headless-test code paths
        free of qtpy where the persistence layer isn't exercised.
        """
        if self._window_key is None:
            return None
        from qtpy.QtCore import QSettings
        return QSettings("apeGmsh", self._window_key)

    def _save_layout(self) -> None:
        """Persist the current dock arrangement and window geometry.

        No-op when ``window_key`` is ``None`` — the legacy
        non-persistent path is preserved for callers (and tests) that
        don't want QSettings traffic.
        """
        settings = self._layout_settings()
        if settings is None:
            return
        try:
            settings.setValue(
                "layout/schema_version", self._LAYOUT_SCHEMA_VERSION,
            )
            settings.setValue("layout/geometry", self._window.saveGeometry())
            settings.setValue("layout/state", self._window.saveState())
        except Exception:
            pass

    def _restore_layout(self) -> None:
        """Restore the prior dock arrangement, if any.

        Skips restoration when the stored schema doesn't match the
        current build — applying a v1 state to a v2 layout produces
        half-broken arrangements (e.g. tabified docks getting
        un-tabbed). Discarding the mismatched state silently falls
        back to the captured defaults.
        """
        settings = self._layout_settings()
        if settings is None:
            return
        stored = settings.value("layout/schema_version")
        try:
            stored_int = int(stored) if stored is not None else None
        except (TypeError, ValueError):
            stored_int = None
        if stored_int != self._LAYOUT_SCHEMA_VERSION:
            return

        geom = settings.value("layout/geometry")
        state = settings.value("layout/state")
        if geom is not None:
            try:
                self._window.restoreGeometry(geom)
            except Exception:
                pass
        if state is not None:
            try:
                self._window.restoreState(state)
            except Exception:
                pass

    def reset_layout(self) -> None:
        """Restore the default dock arrangement captured at startup.

        Wired to the "Reset Layout" entry in the View menu when
        ``window_key`` is set (the entry is disabled otherwise — same
        contract :class:`ResultsWindow` uses). Re-shows every dock
        that may have been closed by the user via its title-bar ×
        button so the reset produces a recognizable arrangement.
        """
        if self._default_layout_state is not None:
            try:
                self._window.restoreState(self._default_layout_state)
            except Exception:
                pass
        for dock in (
            self._tabs_dock,
            self._console_dock,
            *self._extension_docks.values(),
        ):
            if dock is not None:
                dock.setVisible(True)
        try:
            self.set_status("Layout reset to default", 3000)
        except Exception:
            pass

    def _ensure_view_menu(self) -> Any:
        """Lazily create the View menu on the window's menu bar.

        ``__init__`` only builds the menu when there's at least one
        toggleable dock or layout persistence is enabled;
        :meth:`add_extension_dock` calls this to materialize the menu
        the first time a runtime addition needs one. Uses
        :func:`build_view_menu` so a Reset Layout entry is always
        present (enabled iff ``window_key`` is set). Idempotent.
        """
        if self._view_menu is not None:
            return self._view_menu
        menu_bar = self._window.menuBar()
        if menu_bar is None:
            return None
        on_reset = (
            self.reset_layout if self._window_key is not None else None
        )
        self._view_menu, self._view_menu_reset_separator = build_view_menu(
            menu_bar,
            docks=[],
            on_reset_layout=on_reset,
        )
        return self._view_menu

    def add_toolbar_button(self, tooltip: str, icon_text: str, callback) -> None:
        """Add a button to the toolbar (after construction).

        Legacy entry point — drops the returned QAction. New callers
        should prefer :meth:`add_toolbar_action`, which returns the
        QAction so it can be checked / toggled / removed later.
        """
        self.add_toolbar_action(tooltip, icon_text, callback)

    def add_toolbar_action(
        self,
        tooltip: str,
        icon_text: str,
        callback,
        *,
        checkable: bool = False,
        triggered_signal: str = "triggered",
    ):
        """Plan 02 — public extensibility hook for diagrams / overlays.

        Adds a button to the viewer's toolbar at runtime and returns
        the underlying ``QAction`` so the caller can:

        * flip ``setChecked(bool)`` to drive a checkable toggle's
          visual state from external events;
        * call :meth:`remove_toolbar_action` to unregister it on
          overlay teardown.

        The action is registered with the theme-refresh list so its
        icon recolors when the palette changes — same lifecycle as
        the chrome's own buttons. Equivalent to ParaView's
        ``pqViewFrame.addTitleBarAction``.

        Parameters
        ----------
        tooltip
            Tooltip text shown on hover. Include the shortcut hint
            (e.g. ``"Reset view (R)"``) when one exists.
        icon_text
            Single-glyph icon. Unicode symbols (`U+2316`, etc.) keep
            the chrome icon-file-free.
        callback
            Called when the action triggers (or toggles, when
            ``checkable=True``).
        checkable
            If ``True``, the button becomes a two-state toggle and
            ``triggered_signal`` defaults to ``"triggered"`` — pass
            ``"toggled"`` when you want the new state as the callback
            argument.
        triggered_signal
            QAction signal name to connect: usually ``"triggered"``
            for momentary buttons, ``"toggled"`` for two-state
            toggles wanting the bool payload.
        """
        return self._add_toolbar_action(
            self._toolbar,
            icon_text, tooltip, callback,
            checkable=checkable,
            triggered_signal=triggered_signal,
        )

    def remove_toolbar_action(self, action) -> None:
        """Remove a toolbar action previously added via
        :meth:`add_toolbar_action`.

        Idempotent — calling twice on the same action, or on an
        action that wasn't registered through this window, is a
        silent no-op. Drops the action from the theme-refresh list so
        a later palette change doesn't poke a destructed QAction.
        """
        if action is None:
            return
        try:
            self._toolbar.removeAction(action)
        except Exception:
            pass
        self._icon_actions = [
            (a, t) for (a, t) in self._icon_actions if a is not action
        ]

    def on_theme_changed(self, callback: Callable[[Any], None]) -> None:
        """Register a viewer-level callback for theme changes.

        The callback receives the new ``Palette``. Called after the
        window stylesheet + viewport have been re-applied.
        """
        self._theme_callbacks.append(callback)

    def add_toolbar_separator(self) -> None:
        """Add a separator to the toolbar."""
        self._toolbar.addSeparator()

    def add_shortcut(self, key: str, callback) -> None:
        """Add a window-level keyboard shortcut (works regardless of focus).

        Parameters
        ----------
        key : str
            Qt key sequence string, e.g. ``"Escape"``, ``"Q"``, ``"H"``.
        callback : callable
            Called when the key is pressed.
        """
        from qtpy.QtWidgets import QShortcut
        from qtpy.QtGui import QKeySequence
        shortcut = QShortcut(QKeySequence(key), self._window)
        shortcut.activated.connect(callback)

    def set_status(self, text: str, timeout: int = 0) -> None:
        self._statusbar.showMessage(text, timeout)

    def log(self, msg: str) -> None:
        """Append a timestamped line to the console."""
        if self._console is None:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._console.append(f"[{ts}] {msg}")
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def exec(self) -> int:
        """Show the window and run the Qt event loop.

        Honors the ``window_maximized`` preference: if ``True`` (default),
        the window opens maximized; otherwise it opens at natural size.
        """
        # Default to orthographic projection
        try:
            self._qt_interactor.enable_parallel_projection()
            self._act_parallel.setChecked(True)
        except Exception:
            pass
        from .preferences_manager import PREFERENCES as _PREF_MAX
        if _PREF_MAX.current.window_maximized:
            self._window.showMaximized()
        else:
            self._window.show()
        # raise_() + activateWindow(): under jupyter %gui qt the kernel's
        # input hook already runs a Qt event loop, so the self._app.exec_()
        # below returns immediately, the cell finishes, and the kernel
        # goes back to idle. The window is alive (pinned by _LIVE_VIEWERS
        # in ResultsViewer) but the OS leaves it behind the browser —
        # show() alone doesn't request foreground. These two calls bring
        # the window to the top of the Z-order and at minimum trigger a
        # taskbar flash. No-op when exec_() actually blocks (terminal
        # python, mesh viewer outside a kernel).
        self._window.raise_()
        self._window.activateWindow()
        try:
            self._qt_interactor.render()
        except Exception as exc:
            import sys
            import traceback
            print(f"[viewer] initial render() failed: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return self._app.exec_()

    # ------------------------------------------------------------------
    # Camera actions
    # ------------------------------------------------------------------

    def _snap_view(self, direction: str) -> None:
        p = self._qt_interactor
        views = {
            "top":    lambda: p.view_xy(negative=False),
            "bottom": lambda: p.view_xy(negative=True),
            "front":  lambda: p.view_xz(negative=False),
            "back":   lambda: p.view_xz(negative=True),
            "right":  lambda: p.view_yz(negative=False),
            "left":   lambda: p.view_yz(negative=True),
            "iso":    lambda: p.view_isometric(),
        }
        try:
            views[direction]()
            p.reset_camera()
            p.render()
        except Exception:
            pass

    def _toggle_parallel(self, checked: bool) -> None:
        try:
            if checked:
                self._qt_interactor.enable_parallel_projection()
            else:
                self._qt_interactor.disable_parallel_projection()
        except Exception:
            pass
        self._qt_interactor.render()

    def _fit_view(self) -> None:
        try:
            self._qt_interactor.reset_camera()
        except Exception:
            pass
        self._qt_interactor.render()

    def _screenshot(self) -> None:
        try:
            img = self._qt_interactor.screenshot(return_img=True)
            from PIL import Image
            import io
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="BMP")
            bmp_data = buf.getvalue()
            QtCore = self._QtCore
            QtGui = self._QtGui
            qimg = QtGui.QImage()
            qimg.loadFromData(QtCore.QByteArray(bmp_data), "BMP")
            clipboard = self._QtWidgets.QApplication.clipboard()
            clipboard.setImage(qimg)
            self.set_status("Screenshot copied to clipboard", 4000)
        except Exception as exc:
            self.set_status(f"Screenshot failed: {exc}", 4000)

    def _save_screenshot(self) -> None:
        QtWidgets = self._QtWidgets
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._window,
            "Save screenshot",
            "viewer.png",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;TIFF image (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            self._qt_interactor.screenshot(path, transparent_background=False)
            self.set_status(f"Screenshot saved: {path}", 4000)
        except Exception as exc:
            self.set_status(f"Save failed: {exc}", 4000)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_toolbar_action(
        self,
        bar,
        icon_text: str,
        tooltip: str,
        callback,
        *,
        checkable: bool = False,
        triggered_signal: str = "triggered",
    ):
        """Add an action and register it for live theme updates."""
        act = bar.addAction(
            self._make_icon(icon_text, THEME.current.icon), "",
        )
        act.setToolTip(tooltip)
        if checkable:
            act.setCheckable(True)
        getattr(act, triggered_signal).connect(callback)
        self._icon_actions.append((act, icon_text))
        return act

    def _apply_palette(self, palette) -> None:
        """Apply *palette* to the window chrome + viewport + icons."""
        try:
            from .density import DENSITY
            density = DENSITY.current
        except Exception:
            density = None
        self._window.setStyleSheet(build_stylesheet(palette, density))
        try:
            from ..scene.background import apply_background
            apply_background(self._qt_interactor, palette)
        except Exception:
            # Fallback to native linear gradient if the scene module or
            # VTK texture path fails for any reason — keeps the viewer
            # usable rather than crashing on palette change.
            try:
                self._qt_interactor.set_background(
                    palette.bg_top, top=palette.bg_bottom,
                )
            except Exception:
                pass
        self._refresh_toolbar_icons(palette.icon)
        for cb in list(self._theme_callbacks):
            try:
                cb(palette)
            except Exception:
                import logging
                logging.getLogger("apeGmsh.viewer.theme").exception(
                    "theme callback failed: %r", cb,
                )
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _refresh_toolbar_icons(self, color: str) -> None:
        """Re-render every registered toolbar icon in *color*."""
        for act, icon_text in self._icon_actions:
            try:
                act.setIcon(self._make_icon(icon_text, color))
            except Exception:
                pass

    def _make_icon(self, text: str, color: str, size: int = 28):
        QtGui = self._QtGui
        QtCore = self._QtCore
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QColor(color))
        font = QtGui.QFont("Segoe UI", 13 if len(text) <= 1 else 10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRect(0, 0, size, size),
            QtCore.Qt.AlignCenter, text,
        )
        painter.end()
        return QtGui.QIcon(pix)
