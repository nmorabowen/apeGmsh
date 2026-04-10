"""
Probe Panel — Displays probe results: field values at points,
XY plots along lines, and slice information.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QGroupBox, QComboBox, QSpinBox, QTabWidget,
    QSizePolicy,
)
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor

import numpy as np

from apeGmsh.viewers.ui.theme import RED, GREEN, YELLOW, SURFACE0

from apeGmshViewer.visualization.probes import (
    PointProbeResult, LineProbeResult, PlaneProbeResult,
)


class ProbePanel(QWidget):
    """Panel for probe controls and results display."""

    # Signals
    probe_point_requested = Signal()
    probe_line_requested = Signal()
    probe_plane_requested = Signal(str)       # normal: "x", "y", "z"
    clear_probes_requested = Signal()
    stop_probe_requested = Signal()
    line_samples_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("Probes")
        title.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px;"
        )
        layout.addWidget(title)

        # ── Probe Buttons ────────────────────────────────────────────
        btn_group = self._make_group("Probe Tools")
        btn_layout = QVBoxLayout(btn_group)

        row1 = QHBoxLayout()
        self._btn_point = QPushButton("Point")
        self._btn_point.setToolTip("Click on mesh to sample all field values")
        self._btn_point.clicked.connect(self.probe_point_requested.emit)
        row1.addWidget(self._btn_point)

        self._btn_line = QPushButton("Line")
        self._btn_line.setToolTip("Two clicks define a sampling line")
        self._btn_line.clicked.connect(self.probe_line_requested.emit)
        row1.addWidget(self._btn_line)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setToolTip("Cancel active probe")
        self._btn_stop.clicked.connect(self.stop_probe_requested.emit)
        row1.addWidget(self._btn_stop)
        btn_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_plane_x = QPushButton("Slice X")
        self._btn_plane_x.clicked.connect(lambda: self.probe_plane_requested.emit("x"))
        row2.addWidget(self._btn_plane_x)

        self._btn_plane_y = QPushButton("Slice Y")
        self._btn_plane_y.clicked.connect(lambda: self.probe_plane_requested.emit("y"))
        row2.addWidget(self._btn_plane_y)

        self._btn_plane_z = QPushButton("Slice Z")
        self._btn_plane_z.clicked.connect(lambda: self.probe_plane_requested.emit("z"))
        row2.addWidget(self._btn_plane_z)
        btn_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self._btn_interactive_plane = QPushButton("Interactive Plane")
        self._btn_interactive_plane.setToolTip(
            "Drag an interactive plane widget to slice the mesh"
        )
        self._btn_interactive_plane.clicked.connect(
            lambda: self.probe_plane_requested.emit("interactive")
        )
        row3.addWidget(self._btn_interactive_plane)

        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.clicked.connect(self.clear_probes_requested.emit)
        row3.addWidget(self._btn_clear)
        btn_layout.addLayout(row3)

        # Line samples control
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Line Samples:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(10, 1000)
        self._samples_spin.setValue(100)
        self._samples_spin.setSingleStep(10)
        self._samples_spin.valueChanged.connect(self.line_samples_changed.emit)
        row4.addWidget(self._samples_spin)
        btn_layout.addLayout(row4)

        layout.addWidget(btn_group)

        # ── Results Display ──────────────────────────────────────────
        self._tabs = QTabWidget()

        # Point results tab
        self._point_text = QTextEdit()
        self._point_text.setReadOnly(True)
        self._point_text.setStyleSheet(self._text_style())
        self._tabs.addTab(self._point_text, "Point")

        # Line results tab
        self._line_text = QTextEdit()
        self._line_text.setReadOnly(True)
        self._line_text.setStyleSheet(self._text_style())
        self._tabs.addTab(self._line_text, "Line")

        # Plane results tab
        self._plane_text = QTextEdit()
        self._plane_text.setReadOnly(True)
        self._plane_text.setStyleSheet(self._text_style())
        self._tabs.addTab(self._plane_text, "Plane")

        layout.addWidget(self._tabs)

    def _text_style(self) -> str:
        return """
            QTextEdit {
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 4px;
            }
        """

    def _make_group(self, title: str) -> QGroupBox:
        from apeGmsh.viewers.ui.theme import styled_group
        return styled_group(title)

    # ── Display Results ──────────────────────────────────────────────

    def show_point_result(self, result: PointProbeResult) -> None:
        """Display a point probe result."""
        self._tabs.setCurrentIndex(0)

        lines = [
            f"<b style='color: {RED};'>Point Probe #{len(self._get_existing_probes()) + 1}</b>",
            f"",
            f"<b>Position:</b> ({result.position[0]:.4f}, "
            f"{result.position[1]:.4f}, {result.position[2]:.4f})",
            f"<b>Nearest Node:</b> {result.closest_point_id}",
            f"<b>Distance:</b> {result.distance:.4e}",
            f"",
            f"<b style='color: {GREEN};'>Field Values:</b>",
        ]

        for name, val in result.field_values.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.size > 1:
                mag = np.linalg.norm(val)
                components = ", ".join(f"{v:.6e}" for v in val)
                lines.append(f"  <b>{name}:</b> [{components}]")
                lines.append(f"    |{name}| = {mag:.6e}")
            else:
                scalar_val = float(val) if np.isscalar(val) or val.size == 1 else val
                lines.append(f"  <b>{name}:</b> {scalar_val:.6e}")

        # Append to existing text
        existing = self._point_text.toHtml()
        separator = "<br><hr style='border-color: {SURFACE0};'><br>"
        if existing.strip() and "Point Probe" in existing:
            self._point_text.setHtml(existing + separator + "<br>".join(lines))
        else:
            self._point_text.setHtml("<br>".join(lines))

    def show_line_result(self, result: LineProbeResult) -> None:
        """Display a line probe result with field statistics."""
        self._tabs.setCurrentIndex(1)

        lines = [
            f"<b style='color: {YELLOW};'>Line Probe</b>",
            f"",
            f"<b>Point A:</b> ({result.point_a[0]:.4f}, "
            f"{result.point_a[1]:.4f}, {result.point_a[2]:.4f})",
            f"<b>Point B:</b> ({result.point_b[0]:.4f}, "
            f"{result.point_b[1]:.4f}, {result.point_b[2]:.4f})",
            f"<b>Length:</b> {result.total_length:.4f}",
            f"<b>Samples:</b> {result.n_samples}",
            f"",
            f"<b style='color: {GREEN};'>Field Statistics Along Line:</b>",
        ]

        for name, arr in result.field_values.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                lines.append(
                    f"  <b>{name}:</b> min={arr.min():.4e}, "
                    f"max={arr.max():.4e}, mean={arr.mean():.4e}"
                )
            elif arr.ndim == 2:
                mag = np.linalg.norm(arr, axis=1)
                lines.append(
                    f"  <b>|{name}|:</b> min={mag.min():.4e}, "
                    f"max={mag.max():.4e}, mean={mag.mean():.4e}"
                )

        lines.extend([
            f"",
            f"<i style='color: #6c7086;'>Tip: Use matplotlib to plot "
            f"result.arc_length vs result.field_values['field_name']</i>",
        ])

        self._line_text.setHtml("<br>".join(lines))

    def show_plane_result(self, result: PlaneProbeResult) -> None:
        """Display a plane probe result."""
        self._tabs.setCurrentIndex(2)

        n = result.normal
        lines = [
            f"<b style='color: #cba6f7;'>Plane Probe (Slice)</b>",
            f"",
            f"<b>Origin:</b> ({result.origin[0]:.4f}, "
            f"{result.origin[1]:.4f}, {result.origin[2]:.4f})",
            f"<b>Normal:</b> ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})",
            f"<b>Slice Points:</b> {result.slice_mesh.n_points}",
            f"<b>Slice Cells:</b> {result.slice_mesh.n_cells}",
            f"",
            f"<b style='color: {GREEN};'>Fields on Slice:</b>",
        ]

        for name in result.field_names:
            lines.append(f"  {name}")

        self._plane_text.setHtml("<br>".join(lines))

    def clear_results(self) -> None:
        """Clear all probe result displays."""
        self._point_text.clear()
        self._line_text.clear()
        self._plane_text.clear()

    def _get_existing_probes(self) -> list:
        """Count existing point probe results shown."""
        text = self._point_text.toPlainText()
        return [l for l in text.split("\n") if "Point Probe" in l]

    @property
    def n_line_samples(self) -> int:
        return self._samples_spin.value()
