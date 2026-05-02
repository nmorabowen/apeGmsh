"""SessionPanel — viewer-level settings dock for the post-solve viewer.

Hosts session-scoped controls that don't belong inside the model
(Outline) or per-diagram (Details) flows. Ships:

- Visualization toggles (substrate mesh + node cloud overlay).
- Deformation modifier (global warp of the substrate by a nodal field).
- Theme picker.

Future additions land here as new sections (density, layout reset,
file actions, …).
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class SessionPanel:
    """Right-rail dock for viewer-level settings (visualization, theme).

    Subscribes to :data:`THEME` so the displayed theme stays in sync
    when something outside the panel switches it (e.g. an external
    keyboard shortcut, or another viewer instance updating settings).

    Visualization callbacks are optional — when omitted, the toggles
    still render but flipping them is a no-op. The
    :class:`ResultsViewer` wires them up to the substrate / node-cloud
    actors after constructing the panel.
    """

    def __init__(
        self,
        *,
        on_show_mesh: Optional[Callable[[bool], None]] = None,
        on_show_nodes: Optional[Callable[[bool], None]] = None,
        on_show_node_ids: Optional[Callable[[bool], None]] = None,
        on_show_element_ids: Optional[Callable[[bool], None]] = None,
        on_point_size: Optional[Callable[[float], None]] = None,
        on_line_width: Optional[Callable[[float], None]] = None,
        on_opacity: Optional[Callable[[float], None]] = None,
        show_mesh_initial: bool = True,
        show_nodes_initial: bool = True,
        show_node_ids_initial: bool = False,
        show_element_ids_initial: bool = False,
        point_size_initial: float = 10.0,
        line_width_initial: float = 3.0,
        opacity_initial: float = 1.0,
    ) -> None:
        QtWidgets, QtCore = _qt()
        from .theme import THEME, PALETTES

        self._theme = THEME
        self._palettes = PALETTES
        self._on_show_mesh = on_show_mesh
        self._on_show_nodes = on_show_nodes
        self._on_show_node_ids = on_show_node_ids
        self._on_show_element_ids = on_show_element_ids
        self._on_point_size = on_point_size
        self._on_line_width = on_line_width
        self._on_opacity = on_opacity

        widget = QtWidgets.QWidget()
        widget.setObjectName("SessionPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Visualization section ─────────────────────────────────
        viz_label = QtWidgets.QLabel("Visualization")
        viz_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(viz_label)

        self._cb_show_mesh = QtWidgets.QCheckBox("Show mesh")
        self._cb_show_mesh.setChecked(bool(show_mesh_initial))
        self._cb_show_mesh.toggled.connect(self._fire_show_mesh)
        outer.addWidget(self._cb_show_mesh)

        self._cb_show_nodes = QtWidgets.QCheckBox("Show nodes")
        self._cb_show_nodes.setChecked(bool(show_nodes_initial))
        self._cb_show_nodes.toggled.connect(self._fire_show_nodes)
        outer.addWidget(self._cb_show_nodes)

        self._cb_show_node_ids = QtWidgets.QCheckBox("Show node IDs")
        self._cb_show_node_ids.setChecked(bool(show_node_ids_initial))
        self._cb_show_node_ids.toggled.connect(self._fire_show_node_ids)
        outer.addWidget(self._cb_show_node_ids)

        self._cb_show_element_ids = QtWidgets.QCheckBox("Show element IDs")
        self._cb_show_element_ids.setChecked(bool(show_element_ids_initial))
        self._cb_show_element_ids.toggled.connect(self._fire_show_element_ids)
        outer.addWidget(self._cb_show_element_ids)

        # Sizing knobs (mirror PreferencesTab) — hosted in their own
        # form so labels align cleanly under the show/hide toggles.
        viz_form = QtWidgets.QFormLayout()
        viz_form.setContentsMargins(0, 0, 0, 0)
        viz_form.setSpacing(6)

        self._sb_point_size = QtWidgets.QDoubleSpinBox()
        self._sb_point_size.setRange(0.1, 9999.0)
        self._sb_point_size.setSingleStep(1.0)
        self._sb_point_size.setDecimals(1)
        self._sb_point_size.setValue(float(point_size_initial))
        self._sb_point_size.valueChanged.connect(self._fire_point_size)
        viz_form.addRow("Point size", self._sb_point_size)

        self._sb_line_width = QtWidgets.QDoubleSpinBox()
        self._sb_line_width.setRange(0.1, 9999.0)
        self._sb_line_width.setSingleStep(0.5)
        self._sb_line_width.setDecimals(1)
        self._sb_line_width.setValue(float(line_width_initial))
        self._sb_line_width.valueChanged.connect(self._fire_line_width)
        viz_form.addRow("Line width", self._sb_line_width)

        # Opacity slider — single knob applied to wireframe + node
        # cloud together (the substrate fill opacity is separate and
        # comes from preferences). 0..100 maps to 0.0..1.0.
        opacity_row = QtWidgets.QHBoxLayout()
        self._sl_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sl_opacity.setRange(0, 100)
        self._sl_opacity.setValue(int(round(float(opacity_initial) * 100)))
        self._sl_opacity_label = QtWidgets.QLabel(
            f"{int(round(float(opacity_initial) * 100))}%"
        )
        self._sl_opacity_label.setMinimumWidth(36)
        self._sl_opacity.valueChanged.connect(self._fire_opacity)
        opacity_row.addWidget(self._sl_opacity)
        opacity_row.addWidget(self._sl_opacity_label)
        viz_form.addRow("Opacity", opacity_row)

        outer.addLayout(viz_form)

        # Spacer between sections.
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        outer.addWidget(sep)

        # ── Deformation section ───────────────────────────────────
        # Globally warps scene.grid.points by a nodal vector field.
        # Field combo populated by ``set_deform_options(prefixes)``;
        # empty list → whole section is disabled with an explanatory
        # tooltip.
        self._on_deform_enabled: Optional[Callable[[bool], None]] = None
        self._on_deform_field: Optional[Callable[[str], None]] = None
        self._on_deform_scale: Optional[Callable[[float], None]] = None

        self._deform_label = QtWidgets.QLabel("Deformation")
        self._deform_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(self._deform_label)

        self._cb_deform = QtWidgets.QCheckBox("Deform")
        self._cb_deform.setChecked(False)
        self._cb_deform.toggled.connect(self._fire_deform_enabled)
        outer.addWidget(self._cb_deform)

        deform_form = QtWidgets.QFormLayout()
        deform_form.setContentsMargins(0, 0, 0, 0)
        deform_form.setSpacing(6)

        self._combo_deform_field = QtWidgets.QComboBox()
        self._combo_deform_field.currentIndexChanged.connect(
            self._fire_deform_field,
        )
        deform_form.addRow("Tied to", self._combo_deform_field)

        self._sb_deform_scale = QtWidgets.QDoubleSpinBox()
        self._sb_deform_scale.setRange(0.0, 1e6)
        self._sb_deform_scale.setSingleStep(0.5)
        self._sb_deform_scale.setDecimals(3)
        self._sb_deform_scale.setValue(1.0)
        self._sb_deform_scale.valueChanged.connect(self._fire_deform_scale)
        deform_form.addRow("Scale", self._sb_deform_scale)

        outer.addLayout(deform_form)

        # Section disabled until ``set_deform_options`` is called with
        # a non-empty list. Default-disabled keeps the controls inert
        # for files without nodal vector data.
        self._set_deform_enabled_widgets(False)
        self._deform_disabled_tooltip = (
            "No nodal displacement / velocity / acceleration data "
            "in this file."
        )
        for w in (
            self._deform_label,
            self._cb_deform,
            self._combo_deform_field,
            self._sb_deform_scale,
        ):
            w.setToolTip(self._deform_disabled_tooltip)

        sep_def = QtWidgets.QFrame()
        sep_def.setFrameShape(QtWidgets.QFrame.HLine)
        sep_def.setFrameShadow(QtWidgets.QFrame.Sunken)
        outer.addWidget(sep_def)

        # ── Theme picker ───────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self._theme_combo = QtWidgets.QComboBox()
        for name in sorted(PALETTES.keys()):
            self._theme_combo.addItem(name, name)
        # Reflect the current theme without firing the change signal.
        idx = self._theme_combo.findData(THEME.current.name)
        if idx >= 0:
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(idx)
            self._theme_combo.blockSignals(False)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_chosen)
        form.addRow("Theme:", self._theme_combo)

        outer.addLayout(form)

        # Trailing stretch so controls pack at the top of the dock —
        # any spare vertical space ends up below, not between widgets.
        outer.addStretch(1)

        # Track external theme changes (e.g. another consumer calling
        # THEME.set_theme) so the combo stays accurate.
        self._unsub_theme = THEME.subscribe(self._on_theme_changed_externally)

        self._widget = widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def set_show_mesh_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the show-mesh callback (used by ResultsViewer)."""
        self._on_show_mesh = cb

    def set_show_nodes_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the show-nodes callback."""
        self._on_show_nodes = cb

    def set_point_size_callback(
        self, cb: Optional[Callable[[float], None]],
    ) -> None:
        """Late binding for the node-cloud size callback."""
        self._on_point_size = cb

    def set_line_width_callback(
        self, cb: Optional[Callable[[float], None]],
    ) -> None:
        """Late binding for the substrate edge-width callback."""
        self._on_line_width = cb

    def set_opacity_callback(
        self, cb: Optional[Callable[[float], None]],
    ) -> None:
        """Late binding for the wireframe + node-cloud opacity callback."""
        self._on_opacity = cb

    def set_show_node_ids_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the node-IDs label toggle."""
        self._on_show_node_ids = cb

    def set_show_element_ids_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the element-IDs label toggle."""
        self._on_show_element_ids = cb

    def set_deform_enabled_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the Deform checkbox."""
        self._on_deform_enabled = cb

    def set_deform_field_callback(
        self, cb: Optional[Callable[[str], None]],
    ) -> None:
        """Late binding for the deformation-field combo (vector prefix)."""
        self._on_deform_field = cb

    def set_deform_scale_callback(
        self, cb: Optional[Callable[[float], None]],
    ) -> None:
        """Late binding for the deformation-scale spinner."""
        self._on_deform_scale = cb

    def set_deform_options(self, prefixes: list[str]) -> None:
        """Populate the deformation-field combo.

        Pass an empty list to disable the whole section. The first
        entry becomes the active selection without firing the
        callback (the viewer reads :meth:`current_deform_field` after
        ``show()`` if it needs the initial value).
        """
        QtWidgets, _ = _qt()
        self._combo_deform_field.blockSignals(True)
        try:
            self._combo_deform_field.clear()
            for pfx in prefixes:
                self._combo_deform_field.addItem(pfx, pfx)
        finally:
            self._combo_deform_field.blockSignals(False)
        has_options = bool(prefixes)
        self._set_deform_enabled_widgets(has_options)
        tip = "" if has_options else self._deform_disabled_tooltip
        for w in (
            self._deform_label,
            self._cb_deform,
            self._combo_deform_field,
            self._sb_deform_scale,
        ):
            w.setToolTip(tip)

    def current_deform_field(self) -> Optional[str]:
        """Return the active prefix in the field combo (None if empty)."""
        data = self._combo_deform_field.currentData()
        if data is None:
            return None
        return str(data)

    def current_deform_scale(self) -> float:
        return float(self._sb_deform_scale.value())

    def current_deform_enabled(self) -> bool:
        return bool(self._cb_deform.isChecked())

    def close(self) -> None:
        """Detach observers — call when the host window closes."""
        try:
            self._unsub_theme()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _fire_show_mesh(self, checked: bool) -> None:
        if self._on_show_mesh is not None:
            self._on_show_mesh(bool(checked))

    def _fire_show_nodes(self, checked: bool) -> None:
        if self._on_show_nodes is not None:
            self._on_show_nodes(bool(checked))

    def _fire_point_size(self, value: float) -> None:
        if self._on_point_size is not None:
            self._on_point_size(float(value))

    def _fire_line_width(self, value: float) -> None:
        if self._on_line_width is not None:
            self._on_line_width(float(value))

    def _fire_opacity(self, value: int) -> None:
        frac = float(value) / 100.0
        self._sl_opacity_label.setText(f"{value}%")
        if self._on_opacity is not None:
            self._on_opacity(frac)

    def _fire_show_node_ids(self, checked: bool) -> None:
        if self._on_show_node_ids is not None:
            self._on_show_node_ids(bool(checked))

    def _fire_show_element_ids(self, checked: bool) -> None:
        if self._on_show_element_ids is not None:
            self._on_show_element_ids(bool(checked))

    def _fire_deform_enabled(self, checked: bool) -> None:
        if self._on_deform_enabled is not None:
            self._on_deform_enabled(bool(checked))

    def _fire_deform_field(self, _idx: int) -> None:
        if self._on_deform_field is None:
            return
        data = self._combo_deform_field.currentData()
        if data is None:
            return
        self._on_deform_field(str(data))

    def _fire_deform_scale(self, value: float) -> None:
        if self._on_deform_scale is not None:
            self._on_deform_scale(float(value))

    def _set_deform_enabled_widgets(self, enabled: bool) -> None:
        """Enable / disable every control in the Deformation section."""
        for w in (
            self._cb_deform,
            self._combo_deform_field,
            self._sb_deform_scale,
        ):
            w.setEnabled(bool(enabled))
        if not enabled:
            # Force the checkbox off so a deferred enable doesn't
            # silently re-arm a stale ON state.
            self._cb_deform.blockSignals(True)
            self._cb_deform.setChecked(False)
            self._cb_deform.blockSignals(False)

    def _on_theme_chosen(self, _idx: int) -> None:
        name = self._theme_combo.currentData()
        if name is None:
            return
        self._theme.set_theme(name)

    def _on_theme_changed_externally(self, palette) -> None:
        idx = self._theme_combo.findData(palette.name)
        if idx < 0 or idx == self._theme_combo.currentIndex():
            return
        self._theme_combo.blockSignals(True)
        self._theme_combo.setCurrentIndex(idx)
        self._theme_combo.blockSignals(False)
