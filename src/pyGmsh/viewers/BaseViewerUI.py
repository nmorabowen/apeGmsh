"""
BaseViewerUI
============

Generic Qt (pyvistaqt) window base class for 3D viewers.

Provides the shared window shell:

* **Center**: ``pyvistaqt.QtInteractor`` hosting the VTK scene.
* **Right dock -- Tabs**: default *Preferences* tab; subclasses add more
  via :meth:`_build_tabs`.
* **Right dock -- Extra docks**: empty by default; subclasses add via
  :meth:`_build_docks`.
* **Bottom dock -- Console**: collapsible read-only event log.
* **Toolbar**: camera controls (ortho/perspective, fit, standard views),
  screenshot, help.  Subclasses extend via :meth:`_build_toolbar_extra`.
* **Status bar**: generic pick count.

**Not** included here (selection-specific UI lives in
``SelectionPickerUI``):

* Browser tree / model tree
* Physical groups dock
* Entity labels / View tab
* Selection filter tab

Subclass contract
-----------------
Override any combination of the ``_build_*`` / ``_*_extra`` hooks to
inject domain-specific UI.  The most useful hooks:

* :meth:`_build_tabs` -- extend to add tabs before Preferences.
* :meth:`_build_docks` -- return extra ``QDockWidget`` instances.
* :meth:`_build_toolbar_extra` -- append buttons to the toolbar.
* :meth:`_build_prefs_extra` -- append rows to the preferences form.
* :meth:`_help_extra_rows` -- add rows to the shortcuts dialog.
* :meth:`_apply_visual_changes` -- recolor actors after pref changes.
* :meth:`_on_hover_changed_ui` -- respond to hover changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyGmsh.viewers.BaseViewer import BaseViewer


def _lazy_qt():
    """Import Qt + pyvistaqt on first use.  Raises a clear ImportError
    if the deps aren't installed."""
    try:
        from qtpy import QtWidgets, QtCore, QtGui   # noqa: F401
        from pyvistaqt import QtInteractor           # noqa: F401
    except ImportError as err:
        raise ImportError(
            "BaseViewerUI requires pyvistaqt + a Qt binding "
            "(PyQt5 / PyQt6 / PySide2 / PySide6).  Install with "
            "`pip install pyvistaqt PyQt5`."
        ) from err
    return QtWidgets, QtCore, QtGui, QtInteractor


# ======================================================================
# BaseViewerWindow
# ======================================================================

class BaseViewerWindow:
    """QMainWindow hosting the 3D viewport + tabs + console + toolbar.

    Not a direct ``QMainWindow`` subclass because Qt classes must be
    defined after the ``QApplication`` exists.  Instead, this wrapper
    holds an instance of a runtime-created inner ``_PickerMainWindow``
    subclass, and forwards ``.exec()``.
    """

    def __init__(
        self,
        viewer: "BaseViewer",
        *,
        title: str = "Viewer",
        maximized: bool = True,
    ) -> None:
        QtWidgets, QtCore, QtGui, QtInteractor = _lazy_qt()

        # Ensure a QApplication exists
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        self._viewer = viewer
        self._QtWidgets = QtWidgets
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._title = title
        self._maximized = maximized

        # Parallel projection state (mirrors QtInteractor.camera)
        self._parallel = False

        # Label overlays managed by subclasses
        self._view_label_actors: list = []

        # Build the actual QMainWindow
        self._build_window(QtInteractor)

    # ------------------------------------------------------------------
    # Exec / teardown
    # ------------------------------------------------------------------

    def exec(self) -> int:
        """Show the window and run the Qt event loop until closed."""
        from pyGmsh.viewers.BaseViewer import _BG_TOP, _BG_BOTTOM
        try:
            self._qt_interactor.set_background(_BG_TOP, top=_BG_BOTTOM)
        except Exception:
            pass
        if self._maximized:
            self._window.showMaximized()
        else:
            self._window.show()
        try:
            self._qt_interactor.render()
        except Exception:
            pass
        return self._app.exec_()

    # ------------------------------------------------------------------
    # QMainWindow construction
    # ------------------------------------------------------------------

    def _build_window(self, QtInteractor):
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        ui_self = self

        class _PickerMainWindow(QtWidgets.QMainWindow):
            def closeEvent(self, event):
                # Let the viewer commit any pending state before close
                try:
                    ui_self._viewer._commit_active_group()
                except Exception:
                    pass
                # Shut down the QtInteractor cleanly
                try:
                    ui_self._qt_interactor.close()
                except Exception:
                    pass
                ui_self._viewer._plotter = None
                super().closeEvent(event)

        window = _PickerMainWindow()
        window.setWindowTitle(self._title)
        window.resize(1600, 1000)

        # Central: QtInteractor (wraps a VTK render window)
        self._qt_interactor = QtInteractor(parent=window)
        window.setCentralWidget(self._qt_interactor.interactor)

        # Viewer core configures the VTK plotter (scene, picking, hover)
        self._viewer._setup_on(self._qt_interactor)

        # ---- Right panel: tabs ----
        self._tabs = QtWidgets.QTabWidget()
        for tab_name, tab_widget in self._build_tabs():
            self._tabs.addTab(tab_widget, tab_name)

        tabs_dock = QtWidgets.QDockWidget("Panel")
        tabs_dock.setMinimumWidth(320)
        tabs_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        tabs_dock.setWidget(self._tabs)
        window.addDockWidget(QtCore.Qt.RightDockWidgetArea, tabs_dock)

        # ---- Extra docks (subclass-provided, below tabs on the right) ----
        extra_docks = self._build_docks()
        for dock in extra_docks:
            window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # ---- Bottom dock: Console (collapsible, hidden by default) ----
        # Subclasses can set _show_console = False to skip entirely.
        self._console_dock = None
        if getattr(self, "_show_console", True):
            console_dock = QtWidgets.QDockWidget("Console")
            console_dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            console_dock.setWidget(self._build_console_widget())
            window.addDockWidget(
                QtCore.Qt.BottomDockWidgetArea, console_dock,
            )
            self._console_dock = console_dock
            console_dock.hide()  # collapsed by default

        # Corner assignments -- right docks span full height
        window.setCorner(
            QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea,
        )
        window.setCorner(
            QtCore.Qt.TopRightCorner, QtCore.Qt.RightDockWidgetArea,
        )

        # ---- Toolbar (floating, left, vertical) ----
        toolbar = self._build_toolbar()
        toolbar.setOrientation(QtCore.Qt.Vertical)
        toolbar.setFloatable(True)
        window.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)

        # ---- View menu -- toggle dock visibility ----
        menu_bar = window.menuBar()
        view_menu = menu_bar.addMenu("View")
        if self._console_dock is not None:
            view_menu.addAction(self._console_dock.toggleViewAction())
        for dock in extra_docks:
            view_menu.addAction(dock.toggleViewAction())

        # ---- Status bar ----
        self._statusbar = window.statusBar()
        self._refresh_statusbar()

        # ---- Observer callbacks ----
        viewer = self._viewer
        viewer._on_pick_changed.append(self._refresh_statusbar)
        viewer._on_visibility_changed.append(self._refresh_statusbar)
        viewer._on_hover_changed.append(self._on_hover_changed_ui)

        self._window = window

    # ------------------------------------------------------------------
    # Virtual hooks for subclass extension
    # ------------------------------------------------------------------

    def _build_tabs(self) -> list[tuple[str, object]]:
        """Return ``[(tab_name, widget), ...]``.
        Default provides only the Preferences tab.  Subclass should call
        ``super()._build_tabs()`` and prepend its own tabs."""
        return [("Preferences", self._build_prefs_tab())]

    def _build_docks(self) -> list:
        """Return extra ``QDockWidget`` instances to add below the tabs
        in the right dock area.  Default is empty."""
        return []

    def _build_toolbar_extra(self, bar) -> None:
        """Append subclass-specific buttons to *bar*.  Default is empty."""

    def _build_prefs_extra(self, form) -> None:
        """Append subclass-specific rows to the preferences *form*.
        Default is empty."""

    def _help_extra_rows(self) -> list[tuple[str, str]]:
        """Return ``[(key, description), ...]`` to append to the help
        shortcuts dialog.  Default is empty."""
        return []

    def _apply_visual_changes(self) -> None:
        """Called after a preference changes to recolor / update actors.
        Subclass overrides to apply its specific visual logic."""

    def _on_hover_changed_ui(self) -> None:
        """Called when the hover entity changes.  Subclass overrides to
        refresh an info panel or similar."""

    # ------------------------------------------------------------------
    # Icon helper
    # ------------------------------------------------------------------

    def _make_icon(self, text: str, color: str, size: int = 28):
        """Paint *text* into a transparent QPixmap and return a QIcon."""
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

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self):
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        bar = QtWidgets.QToolBar("Tools")
        bar.setMovable(True)
        bar.setIconSize(QtCore.QSize(28, 28))
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        bar.setStyleSheet(
            "QToolBar { spacing: 2px; }"
            "QToolButton { border: none; border-radius: 4px;"
            "  padding: 4px; margin: 1px; }"
            "QToolButton:hover { background: rgba(255,255,255,30); }"
            "QToolButton:pressed { background: rgba(255,255,255,50); }"
            "QToolButton:checked { background: rgba(100,180,255,60);"
            "  border: 1px solid rgba(100,180,255,120); }"
        )

        _IC = "#2d2d2d"  # dark icon colour

        # ---- Camera ----
        self._act_parallel = bar.addAction(
            self._make_icon("\u2316", _IC), "")
        self._act_parallel.setToolTip("Ortho / perspective toggle")
        self._act_parallel.setCheckable(True)
        self._act_parallel.toggled.connect(self._action_toggle_parallel)

        act_fit = bar.addAction(self._make_icon("\u2922", _IC), "")
        act_fit.setToolTip("Fit view")
        act_fit.triggered.connect(self._action_fit_view)

        bar.addSeparator()

        # ---- Standard views ----
        act_top = bar.addAction(self._make_icon("T", _IC), "")
        act_top.setToolTip("Top view  (Z+)")
        act_top.triggered.connect(lambda: self._snap_view("top"))

        act_bottom = bar.addAction(self._make_icon("Bo", _IC), "")
        act_bottom.setToolTip("Bottom view  (Z-)")
        act_bottom.triggered.connect(lambda: self._snap_view("bottom"))

        act_front = bar.addAction(self._make_icon("F", _IC), "")
        act_front.setToolTip("Front view  (Y-)")
        act_front.triggered.connect(lambda: self._snap_view("front"))

        act_back = bar.addAction(self._make_icon("Bk", _IC), "")
        act_back.setToolTip("Back view  (Y+)")
        act_back.triggered.connect(lambda: self._snap_view("back"))

        act_left = bar.addAction(self._make_icon("L", _IC), "")
        act_left.setToolTip("Left view  (X-)")
        act_left.triggered.connect(lambda: self._snap_view("left"))

        act_right = bar.addAction(self._make_icon("R", _IC), "")
        act_right.setToolTip("Right view  (X+)")
        act_right.triggered.connect(lambda: self._snap_view("right"))

        act_iso = bar.addAction(self._make_icon("\u25E3", _IC), "")
        act_iso.setToolTip("Isometric view")
        act_iso.triggered.connect(lambda: self._snap_view("iso"))

        bar.addSeparator()

        # ---- Screenshot ----
        act_screenshot = bar.addAction(
            self._make_icon("\u2399", _IC), "")
        act_screenshot.setToolTip("Copy screenshot to clipboard")
        act_screenshot.triggered.connect(self._action_screenshot)

        bar.addSeparator()

        # ---- Help ----
        act_help = bar.addAction(self._make_icon("?", _IC), "")
        act_help.setToolTip("Shortcuts help")
        act_help.triggered.connect(self._action_show_help)

        # ---- Subclass buttons ----
        self._build_toolbar_extra(bar)

        return bar

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    def _build_console_widget(self):
        """Read-only dark console widget for event logging."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui

        console = QtWidgets.QTextEdit()
        console.setReadOnly(True)
        console.setFont(QtGui.QFont("Consolas", 9))
        console.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #d4d4d4; }"
        )
        self._console = console
        return console

    def log(self, msg: str) -> None:
        """Append a timestamped line to the console widget."""
        if not hasattr(self, "_console") or self._console is None:
            return
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._console.append(f"[{ts}] {msg}")
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Preferences tab
    # ------------------------------------------------------------------

    def _build_prefs_tab(self):
        """Preferences -- visual settings common to all viewers."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        viewer = self._viewer

        # Point size
        self._s_point = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_point.setRange(1, 20)
        self._s_point.setValue(int(viewer._point_size))
        self._s_point.valueChanged.connect(self._on_point_size_changed)
        form.addRow("Point size", self._s_point)

        # Line width
        self._s_line = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_line.setRange(1, 20)
        self._s_line.setValue(int(viewer._line_width))
        self._s_line.valueChanged.connect(self._on_line_width_changed)
        form.addRow("Line width", self._s_line)

        # Surface opacity (0..100 mapping 0.0..1.0)
        self._s_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_alpha.setRange(0, 100)
        self._s_alpha.setValue(int(viewer._surface_opacity * 100))
        self._s_alpha.valueChanged.connect(self._on_opacity_changed)
        form.addRow("Surface \u03b1", self._s_alpha)

        # Show edges
        self._cb_edges = QtWidgets.QCheckBox("Show surface edges")
        self._cb_edges.setChecked(viewer._show_surface_edges)
        self._cb_edges.toggled.connect(self._on_edges_toggled)
        form.addRow(self._cb_edges)

        # Anti-aliasing
        self._cb_aa = QtWidgets.QCheckBox("Anti-aliasing (SSAA)")
        self._cb_aa.setChecked(True)
        self._cb_aa.toggled.connect(self._on_aa_toggled)
        form.addRow(self._cb_aa)

        # Drag threshold (pixels before click becomes a box-select)
        self._s_drag_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_drag_thresh.setRange(2, 30)
        self._s_drag_thresh.setValue(viewer._drag_threshold)
        self._s_drag_thresh.valueChanged.connect(
            self._on_drag_thresh_changed,
        )
        form.addRow("Drag threshold (px)", self._s_drag_thresh)

        # Theme
        form.addRow(QtWidgets.QLabel(""))  # spacer
        self._theme_combo = QtWidgets.QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        form.addRow("Theme", self._theme_combo)

        # ---- Subclass-specific prefs ----
        self._build_prefs_extra(form)

        return panel

    # ------------------------------------------------------------------
    # Preference callbacks
    # ------------------------------------------------------------------

    def _on_point_size_changed(self, value: int) -> None:
        self._viewer._point_size = float(value)
        self._apply_visual_changes()
        self._qt_interactor.render()

    def _on_line_width_changed(self, value: int) -> None:
        self._viewer._line_width = float(value)
        self._apply_visual_changes()
        self._qt_interactor.render()

    def _on_opacity_changed(self, value: int) -> None:
        self._viewer._surface_opacity = float(value) / 100.0
        self._apply_visual_changes()
        self._qt_interactor.render()

    def _on_edges_toggled(self, checked: bool) -> None:
        self._viewer._show_surface_edges = bool(checked)
        self._apply_visual_changes()
        self._qt_interactor.render()

    def _on_aa_toggled(self, checked: bool) -> None:
        try:
            if checked:
                self._qt_interactor.enable_anti_aliasing("ssaa")
            else:
                self._qt_interactor.disable_anti_aliasing()
        except Exception:
            pass
        self._qt_interactor.render()

    def _on_drag_thresh_changed(self, value: int) -> None:
        self._viewer._drag_threshold = value

    def _on_theme_changed(self, value: str) -> None:
        if value == "Light":
            self._qt_interactor.set_background("#eeeeee", top="#ffffff")
        else:
            from pyGmsh.viewers.BaseViewer import _BG_TOP, _BG_BOTTOM
            self._qt_interactor.set_background(_BG_TOP, top=_BG_BOTTOM)
        self._qt_interactor.render()

    # ------------------------------------------------------------------
    # Camera utilities
    # ------------------------------------------------------------------

    def _snap_view(self, direction: str) -> None:
        """Snap the camera to a standard orthogonal or isometric view."""
        try:
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
            views[direction]()
            p.reset_camera()
            p.render()
        except Exception:
            pass

    def _action_toggle_parallel(self, checked: bool) -> None:
        self._parallel = bool(checked)
        try:
            if checked:
                self._qt_interactor.enable_parallel_projection()
            else:
                self._qt_interactor.disable_parallel_projection()
        except Exception:
            try:
                self._qt_interactor.camera.parallel_projection = bool(checked)
            except Exception:
                pass
        self._qt_interactor.render()

    def _action_fit_view(self) -> None:
        try:
            self._qt_interactor.reset_camera()
        except Exception:
            pass
        self._qt_interactor.render()

    def _action_screenshot(self) -> None:
        """Copy a screenshot of the 3D viewport to the system clipboard."""
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
            self.log("Screenshot copied to clipboard")
            self._statusbar.showMessage(
                "Screenshot copied to clipboard", 4000,
            )
        except Exception as exc:
            self._statusbar.showMessage(f"Screenshot failed: {exc}", 4000)

    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------

    def _action_show_help(self) -> None:
        """Show a modeless dialog listing keyboard / mouse shortcuts."""
        QtWidgets = self._QtWidgets
        dlg = QtWidgets.QDialog(self._window)
        dlg.setWindowTitle(f"{self._title} \u2014 shortcuts")
        dlg.setModal(False)
        dlg.resize(520, 460)

        layout = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QTextBrowser()
        text.setOpenExternalLinks(False)
        text.setHtml(self._build_help_html())
        layout.addWidget(text)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.show()
        self._help_dialog = dlg

    def _build_help_html(self) -> str:
        """Assemble the shortcuts help HTML, including extra rows from
        subclasses."""
        extra = self._help_extra_rows()
        extra_html = ""
        if extra:
            extra_html = "\n<h3>Additional shortcuts</h3>\n<table>\n"
            for key, desc in extra:
                extra_html += (
                    f'  <tr><td class="key">{key}</td>'
                    f'<td class="desc">{desc}</td></tr>\n'
                )
            extra_html += "</table>\n"
        return _BASE_SHORTCUTS_HTML + extra_html

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _refresh_statusbar(self) -> None:
        """Generic status bar -- show pick count."""
        viewer = self._viewer
        n = len(viewer._picks)
        self._statusbar.showMessage(f"picks: {n}")


# ======================================================================
# Module-level constants
# ======================================================================

_BASE_SHORTCUTS_HTML = """
<style>
  h3   { margin-top: 12px; margin-bottom: 4px; }
  table { border-spacing: 0; width: 100%; }
  td.key { font-family: Consolas, monospace; color: #bbb;
           padding: 2px 10px 2px 2px; white-space: nowrap;
           vertical-align: top; }
  td.desc { padding: 2px 0; }
</style>

<h3>Mouse -- 3D viewport</h3>
<table>
  <tr><td class="key">Left click</td>
      <td class="desc">Pick entity under cursor.</td></tr>
  <tr><td class="key">Left drag</td>
      <td class="desc">Rubber-band box-select.
            <b>L-R</b> = window, <b>R-L</b> = crossing.</td></tr>
  <tr><td class="key">Ctrl + Left click</td>
      <td class="desc">Unpick (deselect) entity under cursor.</td></tr>
  <tr><td class="key">Middle drag</td>
      <td class="desc">Pan camera.</td></tr>
  <tr><td class="key">Shift + Middle drag</td>
      <td class="desc">Rotate camera (orbit).</td></tr>
  <tr><td class="key">Right drag</td>
      <td class="desc">Pan camera.</td></tr>
  <tr><td class="key">Wheel</td>
      <td class="desc">Zoom in / out.</td></tr>
</table>

<h3>Keyboard</h3>
<table>
  <tr><td class="key">H</td><td class="desc">Hide selected entities</td></tr>
  <tr><td class="key">I</td><td class="desc">Isolate selected entities</td></tr>
  <tr><td class="key">R</td><td class="desc">Reveal all hidden entities</td></tr>
  <tr><td class="key">U</td><td class="desc">Undo last pick</td></tr>
  <tr><td class="key">Esc</td><td class="desc">Deselect all</td></tr>
  <tr><td class="key">Q</td><td class="desc">Close window</td></tr>
</table>
""".strip()
