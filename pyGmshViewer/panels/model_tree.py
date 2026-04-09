"""
Model Tree Panel — Shows loaded meshes, their fields, and physical groups
in a collapsible tree structure.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QLabel,
    QHeaderView,
)
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QIcon, QColor, QBrush

from pyGmshViewer.loaders.vtu_loader import MeshData


class ModelTree(QWidget):
    """Tree widget showing loaded meshes and their data fields."""

    # Signals
    mesh_selected = Signal(str)           # name of selected mesh
    field_selected = Signal(str, str)     # (mesh_name, field_name)
    field_deselected = Signal(str)        # mesh_name (clear contour)
    time_step_selected = Signal(str, int) # (mesh_name, step_index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._mesh_items: dict[str, QTreeWidgetItem] = {}

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Model Tree")
        title.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(title)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Info"])
        self._tree.setColumnCount(2)
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    def add_mesh(self, name: str, mesh_data: MeshData) -> None:
        """Add a mesh entry to the tree."""
        # Root item for this mesh
        root = QTreeWidgetItem(self._tree)
        root.setText(0, name)
        root.setText(1, f"{mesh_data.n_points} pts, {mesh_data.n_cells} cells")
        root.setData(0, Qt.ItemDataRole.UserRole, ("mesh", name))
        root.setForeground(0, QBrush(QColor("#89b4fa")))
        root.setExpanded(True)

        # Geometry info
        info_item = QTreeWidgetItem(root)
        info_item.setText(0, "Geometry")
        bounds = mesh_data.bounds
        info_item.setText(
            1, f"[{bounds[0]:.1f}, {bounds[1]:.1f}] x "
               f"[{bounds[2]:.1f}, {bounds[3]:.1f}] x "
               f"[{bounds[4]:.1f}, {bounds[5]:.1f}]"
        )
        info_item.setForeground(0, QBrush(QColor("#a6adc8")))

        # Point data fields
        if mesh_data.point_field_names:
            pt_root = QTreeWidgetItem(root)
            pt_root.setText(0, "Point Data")
            pt_root.setText(1, f"{len(mesh_data.point_field_names)} fields")
            pt_root.setForeground(0, QBrush(QColor("#a6e3a1")))
            pt_root.setExpanded(True)

            for fname in mesh_data.point_field_names:
                arr = mesh_data.mesh.point_data[fname]
                shape_str = f"({arr.shape[0]},)" if arr.ndim == 1 else str(arr.shape)
                item = QTreeWidgetItem(pt_root)
                item.setText(0, fname)
                item.setText(1, shape_str)
                item.setData(0, Qt.ItemDataRole.UserRole, ("point_field", name, fname))

        # Cell data fields
        if mesh_data.cell_field_names:
            cl_root = QTreeWidgetItem(root)
            cl_root.setText(0, "Cell Data")
            cl_root.setText(1, f"{len(mesh_data.cell_field_names)} fields")
            cl_root.setForeground(0, QBrush(QColor("#fab387")))
            cl_root.setExpanded(True)

            for fname in mesh_data.cell_field_names:
                arr = mesh_data.mesh.cell_data[fname]
                shape_str = f"({arr.shape[0]},)" if arr.ndim == 1 else str(arr.shape)
                item = QTreeWidgetItem(cl_root)
                item.setText(0, fname)
                item.setText(1, shape_str)
                item.setData(0, Qt.ItemDataRole.UserRole, ("cell_field", name, fname))

        # Time series info
        if mesh_data.has_time_series:
            ts_root = QTreeWidgetItem(root)
            ts_root.setText(0, "Time Steps")
            ts_root.setText(1, f"{len(mesh_data.time_steps)} steps")
            ts_root.setForeground(0, QBrush(QColor("#f9e2af")))

            for i, t in enumerate(mesh_data.time_steps):
                item = QTreeWidgetItem(ts_root)
                item.setText(0, f"Step {i}")
                item.setText(1, f"t = {t:.4g}")
                item.setData(0, Qt.ItemDataRole.UserRole, ("time_step", name, i))

        self._mesh_items[name] = root

    def remove_mesh(self, name: str) -> None:
        """Remove a mesh from the tree."""
        if name in self._mesh_items:
            idx = self._tree.indexOfTopLevelItem(self._mesh_items[name])
            if idx >= 0:
                self._tree.takeTopLevelItem(idx)
            del self._mesh_items[name]

    def clear(self) -> None:
        """Remove all items."""
        self._tree.clear()
        self._mesh_items.clear()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle tree item clicks."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        kind = data[0]
        if kind == "mesh":
            self.mesh_selected.emit(data[1])
        elif kind in ("point_field", "cell_field"):
            self.field_selected.emit(data[1], data[2])
        elif kind == "time_step":
            self.time_step_selected.emit(data[1], data[2])
