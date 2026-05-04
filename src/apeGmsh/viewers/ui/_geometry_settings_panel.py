"""GeometrySettingsPanel — deformation + display editor for a Geometry.

Shown inside the DetailsPanel when the user picks a Geometry row in
the outline. Hosts:

* **Deformation** — toggle / field / scale (per-Geometry warp).
* **Display** — show-mesh / show-nodes toggles + a single opacity
  slider applied to substrate fill, wireframe, and node cloud while
  the geometry is active. These were global SessionPanel knobs until
  the geometry refactor; per-Geometry now lets one view dim the
  substrate beneath a contour while another keeps full alpha.

Available-field detection (Deformation section) is the same as before:
only those vector prefixes (``displacement`` / ``velocity`` /
``acceleration``) that have ≥ 2 axis components recorded on nodes
across any stage are offered. When none qualify, the section is
disabled with a tooltip explaining why.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector
    from ..diagrams._geometries import Geometry


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class GeometrySettingsPanel:
    """Editor for one Geometry's deformation state.

    Parameters
    ----------
    director
        ResultsDirector — used to access ``geometries`` and to detect
        available vector fields from the bound Results.
    available_fields
        Vector prefixes (e.g. ``["displacement"]``) detected at viewer
        open via :func:`_kind_catalog._vector_prefixes`. When empty,
        the controls are disabled.
    """

    def __init__(
        self,
        director: "ResultsDirector",
        available_fields: list[str],
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._director = director
        self._available_fields = list(available_fields)
        self._geom_id: Optional[str] = None
        self._reflecting: bool = False  # block callbacks during sync

        widget = QtWidgets.QWidget()
        widget.setObjectName("GeometrySettingsPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Name field ────────────────────────────────────────────
        name_label = QtWidgets.QLabel("Geometry")
        name_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(name_label)

        self._le_name = QtWidgets.QLineEdit()
        self._le_name.editingFinished.connect(self._fire_name)
        name_form = QtWidgets.QFormLayout()
        name_form.setContentsMargins(0, 0, 0, 0)
        name_form.setSpacing(6)
        name_form.addRow("Name", self._le_name)
        outer.addLayout(name_form)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        outer.addWidget(sep)

        # ── Deformation section ──────────────────────────────────
        deform_label = QtWidgets.QLabel("Deformation")
        deform_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(deform_label)

        self._cb_deform = QtWidgets.QCheckBox("Deform")
        self._cb_deform.toggled.connect(self._fire_deform_enabled)
        outer.addWidget(self._cb_deform)

        deform_form = QtWidgets.QFormLayout()
        deform_form.setContentsMargins(0, 0, 0, 0)
        deform_form.setSpacing(6)

        self._combo_field = QtWidgets.QComboBox()
        for pfx in self._available_fields:
            self._combo_field.addItem(pfx, pfx)
        self._combo_field.currentIndexChanged.connect(self._fire_field)
        deform_form.addRow("Tied to", self._combo_field)

        self._sb_scale = QtWidgets.QDoubleSpinBox()
        self._sb_scale.setRange(0.0, 1e6)
        self._sb_scale.setSingleStep(0.5)
        self._sb_scale.setDecimals(3)
        self._sb_scale.setValue(1.0)
        self._sb_scale.valueChanged.connect(self._fire_scale)
        deform_form.addRow("Scale", self._sb_scale)

        outer.addLayout(deform_form)

        if not self._available_fields:
            tip = (
                "No nodal displacement / velocity / acceleration data "
                "in this file."
            )
            for w in (
                self._cb_deform, self._combo_field, self._sb_scale,
                deform_label,
            ):
                w.setEnabled(False)
                w.setToolTip(tip)

        # ── Display section ──────────────────────────────────────
        # Per-geometry mesh / node visibility + opacity (the global
        # SessionPanel knobs moved here so each Geometry can carry
        # its own substrate look — e.g. dim the mesh beneath a
        # contour layer in one view, full alpha in another).
        sep_disp = QtWidgets.QFrame()
        sep_disp.setFrameShape(QtWidgets.QFrame.HLine)
        sep_disp.setFrameShadow(QtWidgets.QFrame.Sunken)
        outer.addWidget(sep_disp)

        display_label = QtWidgets.QLabel("Display")
        display_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(display_label)

        self._cb_show_mesh = QtWidgets.QCheckBox("Show mesh")
        self._cb_show_mesh.setChecked(True)
        self._cb_show_mesh.toggled.connect(self._fire_show_mesh)
        outer.addWidget(self._cb_show_mesh)

        self._cb_show_nodes = QtWidgets.QCheckBox("Show nodes")
        self._cb_show_nodes.setChecked(True)
        self._cb_show_nodes.toggled.connect(self._fire_show_nodes)
        outer.addWidget(self._cb_show_nodes)

        display_form = QtWidgets.QFormLayout()
        display_form.setContentsMargins(0, 0, 0, 0)
        display_form.setSpacing(6)

        opacity_row = QtWidgets.QHBoxLayout()
        self._sl_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sl_opacity.setRange(0, 100)
        self._sl_opacity.setValue(100)
        self._sl_opacity_label = QtWidgets.QLabel("100%")
        self._sl_opacity_label.setMinimumWidth(36)
        self._sl_opacity.valueChanged.connect(self._fire_opacity)
        opacity_row.addWidget(self._sl_opacity)
        opacity_row.addWidget(self._sl_opacity_label)
        display_form.addRow("Opacity", opacity_row)

        outer.addLayout(display_form)

        outer.addStretch(1)
        self._widget = widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def show_geometry(self, geom_id: str) -> None:
        """Bind the panel to ``geom_id`` and reflect its current state."""
        geom = self._director.geometries.find(geom_id)
        if geom is None:
            return
        self._geom_id = geom_id
        self._reflect(geom)

    def refresh(self) -> None:
        """Re-pull state from the bound geometry (e.g. after rename)."""
        if self._geom_id is None:
            return
        geom = self._director.geometries.find(self._geom_id)
        if geom is None:
            return
        self._reflect(geom)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reflect(self, geom: "Geometry") -> None:
        """Mirror the geometry's state into the controls (no callbacks)."""
        self._reflecting = True
        try:
            self._le_name.blockSignals(True)
            self._le_name.setText(geom.name)
            self._le_name.blockSignals(False)

            self._cb_deform.blockSignals(True)
            self._cb_deform.setChecked(bool(geom.deform_enabled))
            self._cb_deform.blockSignals(False)

            self._combo_field.blockSignals(True)
            if geom.deform_field is not None:
                idx = self._combo_field.findData(geom.deform_field)
                if idx >= 0:
                    self._combo_field.setCurrentIndex(idx)
            self._combo_field.blockSignals(False)

            self._sb_scale.blockSignals(True)
            self._sb_scale.setValue(float(geom.deform_scale))
            self._sb_scale.blockSignals(False)

            self._cb_show_mesh.blockSignals(True)
            self._cb_show_mesh.setChecked(bool(geom.show_mesh))
            self._cb_show_mesh.blockSignals(False)

            self._cb_show_nodes.blockSignals(True)
            self._cb_show_nodes.setChecked(bool(geom.show_nodes))
            self._cb_show_nodes.blockSignals(False)

            pct = int(round(float(geom.display_opacity) * 100))
            self._sl_opacity.blockSignals(True)
            self._sl_opacity.setValue(pct)
            self._sl_opacity.blockSignals(False)
            self._sl_opacity_label.setText(f"{pct}%")
        finally:
            self._reflecting = False

    def _fire_name(self) -> None:
        if self._reflecting or self._geom_id is None:
            return
        name = self._le_name.text()
        self._director.geometries.rename(self._geom_id, name)

    def _fire_deform_enabled(self, checked: bool) -> None:
        if self._reflecting or self._geom_id is None:
            return
        enabled = bool(checked)
        # Coalesce the field from the combo when the user enables
        # deformation without having explicitly picked one — the
        # combo's currentIndexChanged doesn't fire on initial
        # population, so the geometry's deform_field stays None
        # otherwise and the warp would short-circuit to ref points.
        field_to_set: Optional[str] = None
        if enabled:
            geom = self._director.geometries.find(self._geom_id)
            if geom is not None and not geom.deform_field:
                data = self._combo_field.currentData()
                if data is not None:
                    field_to_set = str(data)
        from .._log import log_action
        log_action(
            "ui.geometry", "deform_toggled",
            geom=self._geom_id, enabled=bool(enabled), field=field_to_set,
        )
        self._director.geometries.set_deformation(
            self._geom_id, enabled=enabled, field=field_to_set,
        )

    def _fire_field(self, _idx: int) -> None:
        if self._reflecting or self._geom_id is None:
            return
        data = self._combo_field.currentData()
        if data is None:
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "deform_field_changed",
            geom=self._geom_id, field=str(data),
        )
        self._director.geometries.set_deformation(
            self._geom_id, field=str(data),
        )

    def _fire_scale(self, value: float) -> None:
        if self._reflecting or self._geom_id is None:
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "deform_scale_changed",
            geom=self._geom_id, scale=float(value),
        )
        self._director.geometries.set_deformation(
            self._geom_id, scale=float(value),
        )

    def _fire_show_mesh(self, checked: bool) -> None:
        if self._reflecting or self._geom_id is None:
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "show_mesh_toggled",
            geom=self._geom_id, show=bool(checked),
        )
        self._director.geometries.set_display(
            self._geom_id, show_mesh=bool(checked),
        )

    def _fire_show_nodes(self, checked: bool) -> None:
        if self._reflecting or self._geom_id is None:
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "show_nodes_toggled",
            geom=self._geom_id, show=bool(checked),
        )
        self._director.geometries.set_display(
            self._geom_id, show_nodes=bool(checked),
        )

    def _fire_opacity(self, value: int) -> None:
        # Update the readout immediately so the label tracks the
        # slider regardless of whether we're reflecting (Qt's
        # blockSignals only suppresses valueChanged → our slot, not
        # the visual sync we want here).
        self._sl_opacity_label.setText(f"{value}%")
        if self._reflecting or self._geom_id is None:
            return
        frac = float(value) / 100.0
        from .._log import log_action
        log_action(
            "ui.geometry", "opacity_changed",
            geom=self._geom_id, opacity=frac,
        )
        self._director.geometries.set_display(
            self._geom_id, display_opacity=frac,
        )
