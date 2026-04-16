"""
Controls Panel — Visualization controls for display mode, contour settings,
deformed shape parameters, and camera views.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSlider, QCheckBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QFrame, QSizePolicy,
)
from qtpy.QtCore import Signal, Qt

from apeGmshViewer.visualization.renderer import DisplayMode, COLORMAPS


class ControlsPanel(QWidget):
    """Right-side panel with visualization controls."""

    # Signals
    display_mode_changed = Signal(str)         # DisplayMode name
    colormap_changed = Signal(str)             # colormap name
    opacity_changed = Signal(float)
    scale_factor_changed = Signal(float)
    show_deformed_toggled = Signal(bool)
    show_undeformed_toggled = Signal(bool)
    camera_view_changed = Signal(str)          # "xy", "xz", "yz", "iso"
    picking_mode_changed = Signal(str)         # "none", "point", "cell"
    screenshot_requested = Signal()
    time_step_changed = Signal(int)            # step index
    aa_toggled = Signal(bool)                  # anti-aliasing

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Display Mode ─────────────────────────────────────────────
        display_group = self._make_group("Display")
        dl = QVBoxLayout(display_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Surface", "Surface + Edges", "Wireframe", "Points"
        ])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        row.addWidget(self._mode_combo)
        dl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.valueChanged.connect(
            lambda v: self.opacity_changed.emit(v / 100.0)
        )
        row.addWidget(self._opacity_slider)
        self._opacity_label = QLabel("1.00")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v/100:.2f}")
        )
        row.addWidget(self._opacity_label)
        dl.addLayout(row)

        layout.addWidget(display_group)

        # ── Contour Settings ─────────────────────────────────────────
        contour_group = self._make_group("Contour")
        cl = QVBoxLayout(contour_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(COLORMAPS.keys()))
        self._cmap_combo.currentTextChanged.connect(
            lambda t: self.colormap_changed.emit(COLORMAPS.get(t, "jet"))
        )
        row.addWidget(self._cmap_combo)
        cl.addLayout(row)

        layout.addWidget(contour_group)

        # ── Deformation ──────────────────────────────────────────────
        deform_group = self._make_group("Deformation")
        dfl = QVBoxLayout(deform_group)

        self._deformed_check = QCheckBox("Show Deformed Shape")
        self._deformed_check.toggled.connect(self.show_deformed_toggled.emit)
        dfl.addWidget(self._deformed_check)

        self._undeformed_check = QCheckBox("Show Undeformed Reference")
        self._undeformed_check.setChecked(True)
        self._undeformed_check.toggled.connect(self.show_undeformed_toggled.emit)
        dfl.addWidget(self._undeformed_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Scale Factor:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.0, 1e6)
        self._scale_spin.setDecimals(1)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setSingleStep(0.5)
        self._scale_spin.valueChanged.connect(self.scale_factor_changed.emit)
        row.addWidget(self._scale_spin)
        dfl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Quick Scale:"))
        for factor in [1, 10, 50, 100, 500]:
            btn = QPushButton(f"{factor}x")
            btn.setFixedWidth(45)
            btn.clicked.connect(lambda checked, f=factor: self._set_scale(f))
            row.addWidget(btn)
        dfl.addLayout(row)

        layout.addWidget(deform_group)

        # ── Time Steps ──────────────────────────────────────────────
        self._timestep_group = self._make_group("Time Steps")
        tsl = QVBoxLayout(self._timestep_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Step:"))
        self._timestep_slider = QSlider(Qt.Orientation.Horizontal)
        self._timestep_slider.setRange(0, 0)
        self._timestep_slider.setValue(0)
        self._timestep_slider.setEnabled(False)
        self._timestep_slider.valueChanged.connect(self._on_timestep_changed)
        row.addWidget(self._timestep_slider)
        tsl.addLayout(row)

        self._timestep_label = QLabel("No time series")
        self._timestep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tsl.addWidget(self._timestep_label)

        self._time_steps: list[float] | None = None
        self._timestep_group.setVisible(False)
        layout.addWidget(self._timestep_group)

        # ── Camera ───────────────────────────────────────────────────
        camera_group = self._make_group("Camera")
        caml = QVBoxLayout(camera_group)

        row = QHBoxLayout()
        for label, view_id in [
            ("XY", "xy"), ("XZ", "xz"), ("YZ", "yz"), ("Iso", "iso")
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(50)
            btn.clicked.connect(
                lambda checked, v=view_id: self.camera_view_changed.emit(v)
            )
            row.addWidget(btn)
        btn = QPushButton("Reset")
        btn.setFixedWidth(50)
        btn.clicked.connect(lambda: self.camera_view_changed.emit("reset"))
        row.addWidget(btn)
        caml.addLayout(row)

        layout.addWidget(camera_group)

        # ── Picking ──────────────────────────────────────────────────
        pick_group = self._make_group("Picking")
        pkl = QVBoxLayout(pick_group)

        row = QHBoxLayout()
        for label, mode in [("None", "none"), ("Node", "point"), ("Element", "cell")]:
            btn = QPushButton(label)
            btn.setFixedWidth(60)
            btn.clicked.connect(
                lambda checked, m=mode: self.picking_mode_changed.emit(m)
            )
            row.addWidget(btn)
        pkl.addLayout(row)

        layout.addWidget(pick_group)

        # ── Rendering ────────────────────────────────────────────────
        render_group = self._make_group("Rendering")
        rl = QVBoxLayout(render_group)
        self._aa_check = QCheckBox("Anti-aliasing (SSAA)")
        self._aa_check.setChecked(True)
        self._aa_check.toggled.connect(self.aa_toggled.emit)
        rl.addWidget(self._aa_check)
        layout.addWidget(render_group)

        # ── Screenshot ───────────────────────────────────────────────
        btn = QPushButton("Save Screenshot")
        btn.clicked.connect(self.screenshot_requested.emit)
        layout.addWidget(btn)

        # Push everything to the top
        layout.addStretch()

    def _make_group(self, title: str) -> QGroupBox:
        """Create a styled group box (inherits global STYLESHEET)."""
        from apeGmsh.viewers.ui.theme import styled_group
        return styled_group(title)

    def _on_mode_changed(self, text: str) -> None:
        mode_map = {
            "Surface + Edges": "SURFACE_WITH_EDGES",
            "Surface": "SURFACE",
            "Wireframe": "WIREFRAME",
            "Points": "POINTS",
        }
        self.display_mode_changed.emit(mode_map.get(text, "SURFACE"))

    def _set_scale(self, factor: float) -> None:
        self._scale_spin.setValue(factor)

    # ── Public API for updating state ────────────────────────────────

    def set_scale_factor(self, value: float) -> None:
        self._scale_spin.setValue(value)

    def set_deformed_checked(self, checked: bool) -> None:
        self._deformed_check.setChecked(checked)

    def set_time_steps(self, time_steps: list[float] | None) -> None:
        """Configure the time-step slider for the active mesh."""
        if time_steps and len(time_steps) > 1:
            self._time_steps = time_steps
            self._timestep_slider.setRange(0, len(time_steps) - 1)
            self._timestep_slider.setValue(0)
            self._timestep_slider.setEnabled(True)
            self._timestep_group.setVisible(True)
            self._update_timestep_label(0)
        else:
            self._time_steps = None
            self._timestep_slider.setEnabled(False)
            self._timestep_group.setVisible(False)
            self._timestep_label.setText("No time series")

    def set_time_step_value(self, step: int) -> None:
        """Set the slider position without emitting time_step_changed."""
        self._timestep_slider.blockSignals(True)
        self._timestep_slider.setValue(step)
        self._update_timestep_label(step)
        self._timestep_slider.blockSignals(False)

    def _on_timestep_changed(self, value: int) -> None:
        self._update_timestep_label(value)
        self.time_step_changed.emit(value)

    def _update_timestep_label(self, step: int) -> None:
        if self._time_steps and step < len(self._time_steps):
            n = len(self._time_steps)
            t = self._time_steps[step]
            self._timestep_label.setText(
                f"Step {step} / {n - 1}  \u2014  t = {t:.4g}"
            )
        else:
            self._timestep_label.setText("No time series")
