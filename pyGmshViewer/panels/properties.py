"""
Properties Panel — Shows details about picked nodes/elements and mesh info.
"""

from __future__ import annotations

import numpy as np
from pyGmsh.viewers.ui.theme import BLUE, GREEN, PEACH, RED
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QGroupBox,
)
from qtpy.QtCore import Qt


class PropertiesPanel(QWidget):
    """Bottom-right panel showing picked entity properties and mesh stats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Properties")
        title.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(title)

        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        layout.addWidget(self._info_text)

    def show_mesh_info(self, name: str, mesh_data) -> None:
        """Display mesh summary statistics."""
        mesh = mesh_data.mesh
        bounds = mesh.bounds

        lines = [
            f"<b style='color: {BLUE};'>{name}</b>",
            f"",
            f"<b>Nodes:</b> {mesh.n_points:,}",
            f"<b>Elements:</b> {mesh.n_cells:,}",
            f"",
            f"<b>Bounds:</b>",
            f"  X: [{bounds[0]:.4f}, {bounds[1]:.4f}]",
            f"  Y: [{bounds[2]:.4f}, {bounds[3]:.4f}]",
            f"  Z: [{bounds[4]:.4f}, {bounds[5]:.4f}]",
            f"",
            f"<b>Size:</b>",
            f"  dX = {bounds[1] - bounds[0]:.4f}",
            f"  dY = {bounds[3] - bounds[2]:.4f}",
            f"  dZ = {bounds[5] - bounds[4]:.4f}",
        ]

        if mesh_data.point_field_names:
            lines.append("")
            lines.append(f"<b style='color: {GREEN};'>Point Fields:</b>")
            for f in mesh_data.point_field_names:
                arr = mesh.point_data[f]
                if arr.ndim == 1:
                    lines.append(
                        f"  {f}: [{arr.min():.4e}, {arr.max():.4e}]"
                    )
                else:
                    mag = np.linalg.norm(arr, axis=1)
                    lines.append(
                        f"  {f}: max|v| = {mag.max():.4e}"
                    )

        if mesh_data.cell_field_names:
            lines.append("")
            lines.append(f"<b style='color: {PEACH};'>Cell Fields:</b>")
            for f in mesh_data.cell_field_names:
                arr = mesh.cell_data[f]
                if arr.ndim == 1:
                    lines.append(
                        f"  {f}: [{arr.min():.4e}, {arr.max():.4e}]"
                    )

        self._info_text.setHtml("<br>".join(lines))

    def show_point_info(self, point_coords, point_id: int = -1, field_values: dict | None = None) -> None:
        """Display info about a picked node."""
        lines = [
            f"<b style='color: {RED};'>Picked Node</b>",
            f"",
        ]
        if point_id >= 0:
            lines.append(f"<b>ID:</b> {point_id}")
        lines.extend([
            f"<b>X:</b> {point_coords[0]:.6f}",
            f"<b>Y:</b> {point_coords[1]:.6f}",
            f"<b>Z:</b> {point_coords[2]:.6f}",
        ])

        if field_values:
            lines.append("")
            lines.append(f"<b>Field Values:</b>")
            for name, val in field_values.items():
                if isinstance(val, np.ndarray):
                    lines.append(f"  {name}: {val}")
                else:
                    lines.append(f"  {name}: {val:.6e}")

        self._info_text.setHtml("<br>".join(lines))

    def show_cell_info(self, cell_id: int, cell_type: str = "", n_nodes: int = 0) -> None:
        """Display info about a picked element."""
        lines = [
            f"<b style='color: {RED};'>Picked Element</b>",
            f"",
            f"<b>ID:</b> {cell_id}",
        ]
        if cell_type:
            lines.append(f"<b>Type:</b> {cell_type}")
        if n_nodes:
            lines.append(f"<b>Nodes:</b> {n_nodes}")

        self._info_text.setHtml("<br>".join(lines))

    def clear(self) -> None:
        """Clear the properties display."""
        self._info_text.clear()
