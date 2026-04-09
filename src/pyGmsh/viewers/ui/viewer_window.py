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

    from pyGmsh.viewers.ui.viewer_window import ViewerWindow
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


# Background gradient
_BG_TOP    = "#1a1a2e"
_BG_BOTTOM = "#16213e"


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
        show_console: bool = False,
    ) -> None:
        QtWidgets, QtCore, QtGui, QtInteractor = _lazy_qt()
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

        class _MainWindow(QtWidgets.QMainWindow):
            def closeEvent(self, event):
                if ui_self._on_close is not None:
                    try:
                        ui_self._on_close()
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

        # Apply Catppuccin Mocha theme
        from .theme import STYLESHEET
        self._window.setStyleSheet(STYLESHEET)

        # ── Central: VTK plotter ────────────────────────────────────
        self._qt_interactor = QtInteractor(parent=self._window)
        self._window.setCentralWidget(self._qt_interactor.interactor)

        import pyvista as _pv
        _pv.set_plot_theme("dark")
        _pv.global_theme.font.color = "white"
        self._qt_interactor.set_background(_BG_TOP, top=_BG_BOTTOM)
        try:
            self._qt_interactor.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        self._qt_interactor.add_axes(
            interactive=False, line_width=2, color="white",
            xlabel="X", ylabel="Y", zlabel="Z",
        )

        # ── Right dock: tabs ────────────────────────────────────────
        self._tab_widget = QtWidgets.QTabWidget()
        for name, widget in (tabs or []):
            self._tab_widget.addTab(widget, name)

        tabs_dock = QtWidgets.QDockWidget("Panel")
        tabs_dock.setMinimumWidth(320)
        tabs_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        tabs_dock.setWidget(self._tab_widget)
        self._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, tabs_dock)

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
            act = bar.addAction(self._make_icon(icon_text, "#2d2d2d"), "")
            act.setToolTip(tooltip)
            act.triggered.connect(callback)
        if toolbar_actions:
            bar.addSeparator()

        # Camera controls
        _IC = "#2d2d2d"
        self._act_parallel = bar.addAction(self._make_icon("\u2316", _IC), "")
        self._act_parallel.setToolTip("Ortho / perspective toggle")
        self._act_parallel.setCheckable(True)
        self._act_parallel.toggled.connect(self._toggle_parallel)

        act_fit = bar.addAction(self._make_icon("\u2922", _IC), "")
        act_fit.setToolTip("Fit view")
        act_fit.triggered.connect(self._fit_view)

        bar.addSeparator()

        for label, direction in [
            ("T", "top"), ("Bo", "bottom"), ("F", "front"),
            ("Bk", "back"), ("L", "left"), ("R", "right"),
            ("\u25E3", "iso"),
        ]:
            act = bar.addAction(self._make_icon(label, _IC), "")
            act.setToolTip(f"{direction.capitalize()} view")
            act.triggered.connect(lambda _, d=direction: self._snap_view(d))

        bar.addSeparator()

        act_ss = bar.addAction(self._make_icon("\u2399", _IC), "")
        act_ss.setToolTip("Copy screenshot to clipboard")
        act_ss.triggered.connect(self._screenshot)

        self._toolbar = bar
        self._window.addToolBar(QtCore.Qt.LeftToolBarArea, bar)

        # ── Menu bar ────────────────────────────────────────────────
        menu_bar = self._window.menuBar()
        view_menu = menu_bar.addMenu("View")
        if self._console_dock is not None:
            view_menu.addAction(self._console_dock.toggleViewAction())
        for dock in (extra_docks or []):
            view_menu.addAction(dock.toggleViewAction())

        # ── Status bar ──────────────────────────────────────────────
        self._statusbar = self._window.statusBar()

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

    def add_toolbar_button(self, tooltip: str, icon_text: str, callback) -> None:
        """Add a button to the toolbar (after construction)."""
        act = self._toolbar.addAction(self._make_icon(icon_text, "#2d2d2d"), "")
        act.setToolTip(tooltip)
        act.triggered.connect(callback)

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
        """Show the window maximized and run the Qt event loop."""
        # Default to orthographic projection
        try:
            self._qt_interactor.enable_parallel_projection()
            self._act_parallel.setChecked(True)
        except Exception:
            pass
        self._window.showMaximized()
        try:
            self._qt_interactor.render()
        except Exception:
            pass
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
