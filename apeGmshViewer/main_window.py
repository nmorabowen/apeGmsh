"""
Main Window — The primary application window for apeGmshViewer.

Layout:
┌──────────────────────────────────────────────────────┐
│  Menu Bar                                            │
│  Toolbar                                             │
├───────────┬──────────────────────────┬───────────────┤
│           │                          │               │
│  Model    │                          │  Controls     │
│  Tree     │    VTK Viewport          │               │
│           │                          ├───────────────┤
│           │                          │  Properties   │
│           │                          │               │
├───────────┴──────────────────────────┴───────────────┤
│  Status Bar                                          │
└──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QApplication,
)
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QAction, QKeySequence

import pyvista as pv
from pyvistaqt import QtInteractor

from apeGmshViewer.visualization.renderer import ViewportRenderer, DisplayMode
from apeGmshViewer.visualization.probes import ProbeEngine
from apeGmshViewer.panels.model_tree import ModelTree
from apeGmshViewer.panels.controls import ControlsPanel
from apeGmshViewer.panels.properties import PropertiesPanel
from apeGmshViewer.panels.probe_panel import ProbePanel
from apeGmshViewer.loaders.vtu_loader import load_file, MeshData


class MainWindow(QMainWindow):
    """Main application window for apeGmshViewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pyGmsh Viewer")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # State
        self._loaded_meshes: dict[str, MeshData] = {}
        self._active_mesh: str | None = None

        # Build UI
        self._setup_central_widget()
        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_status_bar()
        self._connect_signals()

        # Apply dark theme
        self._apply_theme()

        # Window-level shortcuts
        from qtpy.QtWidgets import QShortcut
        from qtpy.QtGui import QKeySequence as _QKS
        QShortcut(_QKS("Q"), self).activated.connect(self.close)

        self.set_status("Ready — Open a VTU file to begin")

    # ── UI Setup ─────────────────────────────────────────────────────

    def _setup_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        # Main horizontal splitter
        self._hsplitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self._hsplitter)

        # Left: Model tree
        self._model_tree = ModelTree()
        self._model_tree.setMinimumWidth(220)
        self._model_tree.setMaximumWidth(400)
        self._hsplitter.addWidget(self._model_tree)

        # Center: VTK viewport
        self._vtk_frame = QWidget()
        vtk_layout = QVBoxLayout(self._vtk_frame)
        vtk_layout.setContentsMargins(0, 0, 0, 0)

        self._plotter_widget = QtInteractor(self._vtk_frame)
        vtk_layout.addWidget(self._plotter_widget)
        self._hsplitter.addWidget(self._vtk_frame)

        # Right: Controls + Probes + Properties (vertical splitter)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setMinimumWidth(260)
        right_splitter.setMaximumWidth(420)

        self._controls = ControlsPanel()
        right_splitter.addWidget(self._controls)

        self._probe_panel = ProbePanel()
        right_splitter.addWidget(self._probe_panel)

        self._properties = PropertiesPanel()
        right_splitter.addWidget(self._properties)

        right_splitter.setSizes([350, 350, 200])
        self._hsplitter.addWidget(right_splitter)

        # Splitter proportions
        self._hsplitter.setSizes([250, 800, 300])

        # Create the renderer and probe engine
        self._renderer = ViewportRenderer(self._plotter_widget)
        self._probe_engine = ProbeEngine(self._plotter_widget)

        # Install BaseViewer-compatible camera navigation
        from apeGmshViewer.visualization.navigation import install_navigation
        install_navigation(self._plotter_widget)

        # Default to orthographic projection (consistent with model/mesh viewers)
        try:
            self._plotter_widget.enable_parallel_projection()
        except Exception:
            pass

    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open VTU/PVD...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        screenshot_action = QAction("Save Screenshot...", self)
        screenshot_action.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_action.triggered.connect(self._save_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()

        close_action = QAction("Close All", self)
        close_action.triggered.connect(self._close_all)
        file_menu.addAction(close_action)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        for label, view_id in [
            ("Front (XY)", "xy"),
            ("Top (XZ)", "xz"),
            ("Side (YZ)", "yz"),
            ("Isometric", "iso"),
        ]:
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked, v=view_id: self._set_camera_view(v)
            )
            view_menu.addAction(action)

        view_menu.addSeparator()

        reset_action = QAction("Reset Camera", self)
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(self._renderer.reset_camera)
        view_menu.addAction(reset_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        pick_node = QAction("Pick Node", self)
        pick_node.setShortcut(QKeySequence("P"))
        pick_node.triggered.connect(lambda: self._set_picking("point"))
        tools_menu.addAction(pick_node)

        pick_elem = QAction("Pick Element", self)
        pick_elem.setShortcut(QKeySequence("E"))
        pick_elem.triggered.connect(lambda: self._set_picking("cell"))
        tools_menu.addAction(pick_elem)

        no_pick = QAction("Disable Picking", self)
        no_pick.setShortcut(QKeySequence("Escape"))
        no_pick.triggered.connect(lambda: self._set_picking("none"))
        tools_menu.addAction(no_pick)

        tools_menu.addSeparator()

        probe_point = QAction("Probe Point", self)
        probe_point.setShortcut(QKeySequence("Ctrl+P"))
        probe_point.triggered.connect(self._start_point_probe)
        tools_menu.addAction(probe_point)

        probe_line = QAction("Probe Line", self)
        probe_line.setShortcut(QKeySequence("Ctrl+L"))
        probe_line.triggered.connect(self._start_line_probe)
        tools_menu.addAction(probe_line)

        probe_slice = QAction("Interactive Slice", self)
        probe_slice.setShortcut(QKeySequence("Ctrl+Shift+S"))
        probe_slice.triggered.connect(
            lambda: self._start_plane_probe("interactive")
        )
        tools_menu.addAction(probe_slice)

        clear_probes = QAction("Clear Probes", self)
        clear_probes.triggered.connect(self._clear_probes)
        tools_menu.addAction(clear_probes)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self._open_file)
        toolbar.addSeparator()

        # Ortho / perspective toggle
        self._act_ortho = toolbar.addAction("Ortho")
        self._act_ortho.setCheckable(True)
        self._act_ortho.setChecked(True)
        self._act_ortho.setToolTip("Toggle orthographic / perspective projection")
        self._act_ortho.toggled.connect(self._toggle_projection)

        toolbar.addSeparator()
        toolbar.addAction("Front", lambda: self._set_camera_view("xy"))
        toolbar.addAction("Top", lambda: self._set_camera_view("xz"))
        toolbar.addAction("Side", lambda: self._set_camera_view("yz"))
        toolbar.addAction("Iso", lambda: self._set_camera_view("iso"))
        toolbar.addAction("Reset", self._renderer.reset_camera)
        toolbar.addSeparator()
        toolbar.addAction("Screenshot", self._save_screenshot)
        toolbar.addAction("Clipboard", self._copy_screenshot_clipboard)

    def _setup_status_bar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

    def _apply_theme(self):
        """Apply shared Catppuccin Mocha theme."""
        from apeGmsh.viewers.ui.theme import STYLESHEET
        self.setStyleSheet(STYLESHEET)

    # ── Signal Connections ────────────────────────────────────────────

    def _connect_signals(self):
        # Model tree
        self._model_tree.mesh_selected.connect(self._on_mesh_selected)
        self._model_tree.field_selected.connect(self._on_field_selected)

        # Controls
        self._controls.display_mode_changed.connect(self._on_display_mode)
        self._controls.colormap_changed.connect(self._on_colormap)
        self._controls.opacity_changed.connect(self._on_opacity)
        self._controls.scale_factor_changed.connect(self._on_scale_factor)
        self._controls.show_deformed_toggled.connect(self._on_deformed_toggle)
        self._controls.show_undeformed_toggled.connect(self._on_undeformed_toggle)
        self._controls.camera_view_changed.connect(self._set_camera_view)
        self._controls.picking_mode_changed.connect(self._set_picking)
        self._controls.screenshot_requested.connect(self._save_screenshot)
        self._controls.aa_toggled.connect(self._on_aa_toggled)

        # Time-step navigation
        self._model_tree.time_step_selected.connect(self._on_time_step_selected)
        self._controls.time_step_changed.connect(self._on_time_step_slider)

        # Probe panel
        self._probe_panel.probe_point_requested.connect(self._start_point_probe)
        self._probe_panel.probe_line_requested.connect(self._start_line_probe)
        self._probe_panel.probe_plane_requested.connect(self._start_plane_probe)
        self._probe_panel.clear_probes_requested.connect(self._clear_probes)
        self._probe_panel.stop_probe_requested.connect(self._stop_probe)

        # Probe engine callbacks
        self._probe_engine.set_callbacks(
            on_point=self._on_probe_point_result,
            on_line=self._on_probe_line_result,
            on_plane=self._on_probe_plane_result,
        )

    # ── File Operations ──────────────────────────────────────────────

    def _open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Mesh / Results File",
            "",
            "VTK Files (*.vtu *.pvd *.vtk);;Gmsh Files (*.msh);;All Files (*)",
        )
        if not filepath:
            return

        try:
            mesh_data = load_file(filepath)
            name = mesh_data.name

            # Handle duplicate names
            base_name = name
            counter = 1
            while name in self._loaded_meshes:
                name = f"{base_name}_{counter}"
                counter += 1

            self._loaded_meshes[name] = mesh_data
            self._renderer.add_mesh(mesh_data, name)
            self._model_tree.add_mesh(name, mesh_data)
            self._active_mesh = name
            self._properties.show_mesh_info(name, mesh_data)
            self._controls.set_time_steps(
                mesh_data.time_steps if mesh_data.has_time_series else None
            )

            n = mesh_data.n_points
            e = mesh_data.n_cells
            self.set_status(
                f"Loaded: {name} — {n:,} nodes, {e:,} elements"
            )

        except Exception as ex:
            QMessageBox.critical(self, "Error Loading File", str(ex))

    def _close_all(self):
        self._renderer.clear_all()
        self._model_tree.clear()
        self._properties.clear()
        self._loaded_meshes.clear()
        self._active_mesh = None
        self.set_status("All meshes closed")

    def _save_screenshot(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "screenshot.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)",
        )
        if filepath:
            self._renderer.screenshot(filepath)
            self.set_status(f"Screenshot saved: {filepath}", 4000)

    def _copy_screenshot_clipboard(self):
        """Copy the current viewport to the system clipboard."""
        try:
            img = self._plotter_widget.screenshot(return_img=True)
            from PIL import Image
            import io
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="BMP")
            from qtpy.QtCore import QByteArray
            from qtpy.QtGui import QImage
            from qtpy.QtWidgets import QApplication
            qimg = QImage()
            qimg.loadFromData(QByteArray(buf.getvalue()), "BMP")
            QApplication.clipboard().setImage(qimg)
            self.set_status("Screenshot copied to clipboard", 4000)
        except Exception as exc:
            self.set_status(f"Clipboard screenshot failed: {exc}", 4000)

    def _on_aa_toggled(self, checked: bool):
        """Toggle anti-aliasing on the VTK viewport."""
        try:
            if checked:
                self._plotter_widget.enable_anti_aliasing("ssaa")
            else:
                self._plotter_widget.disable_anti_aliasing()
        except Exception:
            pass
        self._plotter_widget.render()

    def _toggle_projection(self, checked: bool):
        """Toggle orthographic / perspective projection."""
        try:
            if checked:
                self._plotter_widget.enable_parallel_projection()
            else:
                self._plotter_widget.disable_parallel_projection()
        except Exception:
            pass
        self._plotter_widget.render()

    def set_status(self, text: str, timeout: int = 0):
        """Update the status bar message."""
        self._statusbar.showMessage(text, timeout)

    # ── Slot Handlers ────────────────────────────────────────────────

    def _on_mesh_selected(self, name: str):
        self._active_mesh = name
        if name in self._loaded_meshes:
            md = self._loaded_meshes[name]
            self._properties.show_mesh_info(name, md)
            self._controls.set_time_steps(
                md.time_steps if md.has_time_series else None
            )

    def _on_field_selected(self, mesh_name: str, field_name: str):
        """Apply contour plot when a field is clicked in the tree."""
        self._active_mesh = mesh_name
        actor = self._renderer.get_actor(mesh_name)
        cmap = "jet"
        if actor:
            cmap = actor.colormap
        self._renderer.set_scalar_field(mesh_name, field_name, colormap=cmap)
        self.set_status(f"Contour: {field_name} on {mesh_name}")

    def _on_display_mode(self, mode_name: str):
        if not self._active_mesh:
            return
        mode = DisplayMode[mode_name]
        self._renderer.set_display_mode(self._active_mesh, mode)

    def _on_colormap(self, cmap: str):
        if not self._active_mesh:
            return
        actor = self._renderer.get_actor(self._active_mesh)
        if actor and actor.active_scalar:
            self._renderer.set_scalar_field(
                self._active_mesh, actor.active_scalar, colormap=cmap
            )

    def _on_opacity(self, value: float):
        if self._active_mesh:
            self._renderer.set_opacity(self._active_mesh, value)

    def _on_scale_factor(self, value: float):
        if self._active_mesh:
            self._renderer.update_scale_factor(self._active_mesh, value)

    def _on_deformed_toggle(self, checked: bool):
        if not self._active_mesh:
            return
        if checked:
            md = self._loaded_meshes.get(self._active_mesh)
            if not md:
                return
            # Find a displacement-like field
            disp_field = None
            for f in md.point_field_names:
                fl = f.lower()
                if "disp" in fl or "modeshape" in fl or f == "Displacement":
                    disp_field = f
                    break
            if disp_field:
                scale = self._controls._scale_spin.value()
                self._renderer.show_deformed(
                    self._active_mesh,
                    displacement_field=disp_field,
                    scale_factor=scale,
                )
                self.set_status(
                    f"Deformed: {disp_field} x {scale}"
                )
            else:
                self.set_status(
                    "No displacement field found in this mesh"
                )
                self._controls.set_deformed_checked(False)
        else:
            self._renderer.hide_deformed(self._active_mesh)

    def _on_undeformed_toggle(self, checked: bool):
        """Toggle undeformed reference visibility."""
        pass  # Handled during deformed update

    # ── Time-Step Handlers ───────────────────────────────────────────

    def _on_time_step_selected(self, mesh_name: str, step: int) -> None:
        """Handle time-step click in the model tree."""
        self._active_mesh = mesh_name
        self._controls.set_time_step_value(step)
        self._renderer.set_active_time_step(mesh_name, step)
        md = self._loaded_meshes.get(mesh_name)
        if md and md.time_steps and step < len(md.time_steps):
            self.set_status(
                f"Time step {step} (t = {md.time_steps[step]:.4g})"
            )

    def _on_time_step_slider(self, step: int) -> None:
        """Handle time-step slider change."""
        if not self._active_mesh:
            return
        self._renderer.set_active_time_step(self._active_mesh, step)
        md = self._loaded_meshes.get(self._active_mesh)
        if md:
            self._properties.show_mesh_info(self._active_mesh, md)
            # Re-set probe engine mesh reference after step change
            self._probe_engine.set_active_mesh(md.mesh, self._active_mesh)
        if md and md.time_steps and step < len(md.time_steps):
            self.set_status(
                f"Time step {step} (t = {md.time_steps[step]:.4g})"
            )

    def _set_camera_view(self, view_id: str):
        views = {
            "xy": self._renderer.view_xy,
            "xz": self._renderer.view_xz,
            "yz": self._renderer.view_yz,
            "iso": self._renderer.view_isometric,
            "reset": self._renderer.reset_camera,
        }
        if view_id in views:
            views[view_id]()

    def _set_picking(self, mode: str):
        nav = getattr(self._plotter_widget, '_nav_set_picking', None)
        if mode == "point":
            self._renderer.enable_point_picking(
                callback=self._on_point_picked
            )
            if nav:
                nav(True)
            self.set_status("Pick mode: Node — click on mesh")
        elif mode == "cell":
            self._renderer.enable_cell_picking(
                callback=self._on_cell_picked
            )
            if nav:
                nav(True)
            self.set_status("Pick mode: Element — click on mesh")
        else:
            self._renderer.disable_picking()
            if nav:
                nav(False)
            self.set_status("Picking disabled")

    def _on_point_picked(self, point):
        """Handle node pick callback."""
        if point is not None:
            coords = point
            self._properties.show_point_info(coords)

    def _on_cell_picked(self, cell):
        """Handle element pick callback."""
        if cell is not None:
            self._properties.show_cell_info(cell_id=0, cell_type="", n_nodes=0)

    # ── Probe Handlers ─────────────────────────────────────────────

    def _ensure_probe_mesh(self) -> bool:
        """Set the active mesh on the probe engine. Returns True if ready."""
        if not self._active_mesh:
            self.set_status("No active mesh — load a file first")
            return False
        md = self._loaded_meshes.get(self._active_mesh)
        if not md:
            return False
        self._probe_engine.set_active_mesh(md.mesh, self._active_mesh)
        return True

    def _start_point_probe(self):
        if not self._ensure_probe_mesh():
            return
        self._probe_engine.start_point_probe()
        self.set_status(
            "Point Probe: click on the mesh to sample field values"
        )

    def _start_line_probe(self):
        if not self._ensure_probe_mesh():
            return
        self._probe_engine.start_line_probe()
        self.set_status(
            "Line Probe: click FIRST point (A) on the mesh"
        )

    def _start_plane_probe(self, normal: str):
        if not self._ensure_probe_mesh():
            return
        if normal == "interactive":
            self._probe_engine.start_interactive_plane()
            self.set_status(
                "Plane Probe: drag the plane widget to slice"
            )
        else:
            result = self._probe_engine.probe_with_plane(normal=normal)
            if result:
                self._probe_panel.show_plane_result(result)
                self.set_status(
                    f"Plane Probe: sliced along {normal.upper()} axis "
                    f"({result.slice_mesh.n_points} pts)"
                )

    def _clear_probes(self):
        self._probe_engine.clear_all()
        self._probe_panel.clear_results()
        self.set_status("All probes cleared")

    def _stop_probe(self):
        self._probe_engine.stop()
        self.set_status("Probe cancelled")

    def _on_probe_point_result(self, result):
        self._probe_panel.show_point_result(result)
        self.set_status(
            f"Point Probe: node {result.closest_point_id}, "
            f"{len(result.field_values)} fields sampled"
        )

    def _on_probe_line_result(self, result):
        self._probe_panel.show_line_result(result)
        self.set_status(
            f"Line Probe: {result.n_samples} samples over "
            f"L = {result.total_length:.4f}"
        )

    def _on_probe_plane_result(self, result):
        self._probe_panel.show_plane_result(result)
        self.set_status(
            f"Plane Probe: {result.slice_mesh.n_points} pts on slice"
        )

    # ── Public API (for programmatic use) ────────────────────────────

    def load_file(self, filepath: str | Path) -> str:
        """Load a file programmatically. Returns the mesh name."""
        mesh_data = load_file(filepath)
        name = mesh_data.name
        self._loaded_meshes[name] = mesh_data
        self._renderer.add_mesh(mesh_data, name)
        self._model_tree.add_mesh(name, mesh_data)
        self._active_mesh = name
        return name

    def load_mesh_data(self, mesh_data: MeshData, name: str | None = None) -> str:
        """Load a MeshData object directly (e.g., from apeGmsh)."""
        name = name or mesh_data.name
        self._loaded_meshes[name] = mesh_data
        self._renderer.add_mesh(mesh_data, name)
        self._model_tree.add_mesh(name, mesh_data)
        self._active_mesh = name
        self._controls.set_time_steps(
            mesh_data.time_steps if mesh_data.has_time_series else None
        )
        return name

    def closeEvent(self, event):
        """Clean up VTK on close."""
        self._plotter_widget.close()
        super().closeEvent(event)
