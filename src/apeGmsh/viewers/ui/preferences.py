"""
PreferencesTab — Shared visual settings panel.

A standalone QWidget that can be added as a tab in any viewer.
Fires callbacks for each setting change — the viewer wires these
to ColorManager, PickEngine, and actor properties.

Usage::

    from apeGmsh.viewers.ui.preferences import PreferencesTab
    prefs = PreferencesTab(
        point_size=10, line_width=6, surface_opacity=0.35,
        on_point_size=lambda v: ...,
        on_line_width=lambda v: ...,
        on_opacity=lambda v: ...,
    )
"""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

# Qt is lazy-imported at instantiation time via `_qt()`; the TYPE_CHECKING
# block below gives mypy the concrete widget types so class-level
# annotations like ``dict[str, QtWidgets.QSlider]`` resolve correctly.
if TYPE_CHECKING:
    from qtpy import QtWidgets  # noqa: F401


def _qt() -> tuple[Any, Any, Any]:
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


class PreferencesTab:
    """Preferences panel as a standalone QWidget."""

    def __init__(
        self,
        *,
        point_size: float = 10.0,
        line_width: float = 6.0,
        surface_opacity: float = 0.35,
        show_surface_edges: bool = False,
        drag_threshold: int = 8,
        # Callbacks — geometry
        on_point_size: Callable[[float], None] | None = None,
        on_line_width: Callable[[float], None] | None = None,
        on_opacity: Callable[[float], None] | None = None,
        on_edges: Callable[[bool], None] | None = None,
        on_aa: Callable[[bool], None] | None = None,
        on_drag_threshold: Callable[[int], None] | None = None,
        on_theme: Callable[[str], None] | None = None,
        on_pick_color: Callable[[str], None] | None = None,
        # Callbacks — overlay sizing (multipliers, default 1.0×)
        on_overlay_scale: Callable[[str, float], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Geometry group ──────────────────────────────────────────
        geo_group = QtWidgets.QGroupBox("Geometry")
        geo_form = QtWidgets.QFormLayout(geo_group)
        geo_form.setSpacing(4)

        self._s_point = QtWidgets.QDoubleSpinBox()
        self._s_point.setRange(0.1, 9999.0)
        self._s_point.setSingleStep(1.0)
        self._s_point.setDecimals(1)
        self._s_point.setValue(float(point_size))
        if on_point_size:
            self._s_point.valueChanged.connect(on_point_size)
        geo_form.addRow("Point size", self._s_point)

        self._s_line = QtWidgets.QDoubleSpinBox()
        self._s_line.setRange(0.1, 9999.0)
        self._s_line.setSingleStep(0.5)
        self._s_line.setDecimals(1)
        self._s_line.setValue(float(line_width))
        if on_line_width:
            self._s_line.valueChanged.connect(on_line_width)
        geo_form.addRow("Line width", self._s_line)

        self._s_alpha = QtWidgets.QDoubleSpinBox()
        self._s_alpha.setRange(0.0, 1.0)
        self._s_alpha.setSingleStep(0.05)
        self._s_alpha.setDecimals(2)
        self._s_alpha.setValue(float(surface_opacity))
        if on_opacity:
            self._s_alpha.valueChanged.connect(on_opacity)
        geo_form.addRow("Surface \u03b1", self._s_alpha)

        self._cb_edges = QtWidgets.QCheckBox("Show surface edges")
        self._cb_edges.setChecked(show_surface_edges)
        if on_edges:
            self._cb_edges.toggled.connect(on_edges)
        geo_form.addRow(self._cb_edges)

        layout.addWidget(geo_group)

        # ── Rendering group ─────────────────────────────────────────
        render_group = QtWidgets.QGroupBox("Rendering")
        render_form = QtWidgets.QFormLayout(render_group)
        render_form.setSpacing(4)

        self._cb_aa = QtWidgets.QCheckBox("Anti-aliasing (SSAA)")
        self._cb_aa.setChecked(True)
        if on_aa:
            self._cb_aa.toggled.connect(on_aa)
        render_form.addRow(self._cb_aa)

        self._s_drag = QtWidgets.QSpinBox()
        self._s_drag.setRange(2, 30)
        self._s_drag.setValue(drag_threshold)
        if on_drag_threshold:
            self._s_drag.valueChanged.connect(on_drag_threshold)
        render_form.addRow("Drag threshold (px)", self._s_drag)

        layout.addWidget(render_group)

        # ── Theme group ─────────────────────────────────────────────
        theme_group = QtWidgets.QGroupBox("Theme")
        theme_form = QtWidgets.QFormLayout(theme_group)
        theme_form.setSpacing(4)

        self._theme_combo = QtWidgets.QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        if on_theme:
            self._theme_combo.currentTextChanged.connect(on_theme)
        theme_form.addRow("Theme", self._theme_combo)

        # ── Pick color ──────────────────────────────────────────────
        self._btn_pick_color = QtWidgets.QPushButton()
        self._btn_pick_color.setFixedSize(60, 24)
        self._pick_color_hex = "#E74C3C"
        self._btn_pick_color.setStyleSheet(
            f"background-color: {self._pick_color_hex}; border: 1px solid #999;"
        )
        self._on_pick_color = on_pick_color

        def _pick_color_clicked():
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(self._pick_color_hex),
                self.widget,
                "Pick Selection Color",
            )
            if color.isValid():
                self._pick_color_hex = color.name()
                self._btn_pick_color.setStyleSheet(
                    f"background-color: {self._pick_color_hex}; "
                    f"border: 1px solid #999;"
                )
                if self._on_pick_color:
                    self._on_pick_color(self._pick_color_hex)

        self._btn_pick_color.clicked.connect(_pick_color_clicked)
        theme_form.addRow("Selection color", self._btn_pick_color)

        layout.addWidget(theme_group)

        # ── Overlay sizing group ───────────────────────────────────
        overlay_group = QtWidgets.QGroupBox("Overlay Sizing")
        overlay_form = QtWidgets.QFormLayout(overlay_group)
        overlay_form.setSpacing(4)

        # Widget types are resolved only when qtpy is actually imported
        # (at instantiation time). ``Any`` keeps mypy quiet while
        # Pylance / IDEs still pick up the actual QSlider / QLabel types
        # at runtime from the QObject hierarchy.
        self._overlay_sliders: dict[str, Any] = {}
        self._overlay_labels: dict[str, Any] = {}

        _OVERLAY_ITEMS = [
            ("load_arrow",        "Load arrows"),
            ("mass_sphere",       "Mass spheres"),
            ("constraint_marker", "Constraint markers"),
            ("constraint_line",   "Constraint lines"),
        ]

        for key, label in _OVERLAY_ITEMS:
            row = QtWidgets.QHBoxLayout()

            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            # Log-like feel: 1–100 maps to 0.1× – 10×
            # value 10 = 1.0×, value 1 = 0.1×, value 100 = 10.0×
            slider.setRange(1, 100)
            slider.setValue(10)  # default 1.0×
            slider.setTickInterval(10)
            row.addWidget(slider)

            val_label = QtWidgets.QLabel("1.0\u00d7")
            val_label.setMinimumWidth(40)
            row.addWidget(val_label)

            self._overlay_sliders[key] = slider
            self._overlay_labels[key] = val_label

            def _make_handler(k, lbl):
                def _handler(v):
                    mult = v / 10.0  # 1→0.1, 10→1.0, 50→5.0, 100→10.0
                    lbl.setText(f"{mult:.1f}\u00d7")
                    if on_overlay_scale:
                        on_overlay_scale(k, mult)
                return _handler

            slider.valueChanged.connect(_make_handler(key, val_label))
            overlay_form.addRow(label, row)

        layout.addWidget(overlay_group)
        layout.addStretch()

    def add_extra_row(self, label: str, widget) -> None:
        """Add an extra row to the main layout."""
        self.widget.layout().addWidget(widget)

    def add_separator(self, text: str = "") -> None:
        """Add a visual separator with optional label."""
        from qtpy import QtWidgets
        form = self.widget.layout()
        form.addRow(QtWidgets.QLabel(""))
        if text:
            form.addRow(QtWidgets.QLabel(f"--- {text} ---"))
