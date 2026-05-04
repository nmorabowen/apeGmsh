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
from typing import Any, Callable

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
        Additional dock widgets.
    toolbar_actions : list of (tooltip, icon_text, callback), optional
        Buttons inserted before the camera controls.
    on_close : callable, optional
        Called when the window is closed (e.g. flush groups to Gmsh).
    show_console : bool
        Whether to include the collapsible console dock.
    """

    def __init__(
        self,
        *,
        title: str = "Viewer",
        tabs: list[tuple[str, Any]] | None = None,
        extra_docks: list[Any] | None = None,
        toolbar_actions: list[tuple[str, str, Callable]] | None = None,
        on_close: Callable[[], None] | None = None,
        show_console: bool | None = None,
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
        _axes_kwargs = dict(
            interactive=False,
            line_width=_p.axis_line_width,
            color="white",
        )
        if _p.axis_labels_visible:
            _axes_kwargs.update(xlabel="X", ylabel="Y", zlabel="Z")
        else:
            _axes_kwargs.update(xlabel="", ylabel="", zlabel="")
        self._qt_interactor.add_axes(**_axes_kwargs)

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
        tabs_dock.setTitleBarWidget(QtWidgets.QWidget())  # hide title bar
        tabs_dock.setMinimumWidth(_p.dock_min_width)
        tabs_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        tabs_dock.setWidget(self._tab_widget)
        self._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, tabs_dock)
        self._tabs_dock = tabs_dock

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

        # ── Menu bar (only if there are toggleable docks) ───────────
        view_items = []
        if self._console_dock is not None:
            view_items.append(self._console_dock.toggleViewAction())
        for dock in (extra_docks or []):
            view_items.append(dock.toggleViewAction())
        if view_items:
            menu_bar = self._window.menuBar()
            view_menu = menu_bar.addMenu("View")
            for action in view_items:
                view_menu.addAction(action)

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
        """Add a tab to the right-side panel (after construction)."""
        self._tab_widget.addTab(widget, name)

    def add_right_bottom_dock(self, title: str, widget) -> None:
        """Add a dock below the tabs dock on the right side."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        dock = QtWidgets.QDockWidget(title)
        dock.setTitleBarWidget(QtWidgets.QWidget())  # hide title bar
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(widget)
        self._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self._window.splitDockWidget(self._tabs_dock, dock, QtCore.Qt.Vertical)

    def add_toolbar_button(self, tooltip: str, icon_text: str, callback) -> None:
        """Add a button to the toolbar (after construction)."""
        self._add_toolbar_action(self._toolbar, icon_text, tooltip, callback)

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
