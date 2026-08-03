"""GeometrySettingsPanel — deformation + display editor for a Geometry.

Shown inside the DetailsPanel when the user picks a Geometry row in
the outline. Hosts:

* **Deformation** — toggle / field / scale (per-Geometry warp).
* **Threshold** — ADR 0084 D1. Enable / scalar component / topology /
  min / max, hiding every cell whose values fall outside the range.
  Per-Geometry state like the other two sections, and the same
  ``_fire_*`` idiom: mutate the owner (the director's
  :class:`ThresholdController`), then fire ``STEP_CHANGED`` so the
  dispatcher's one STEP path recomputes the mask and repaints.
* **Display** — show-mesh / show-nodes toggles + a single opacity
  slider applied to substrate fill, wireframe, and node cloud while
  the geometry is active. These were global SessionPanel knobs until
  the geometry refactor; per-Geometry now lets one view dim the
  substrate beneath a contour while another keeps full alpha.

Available-field detection (Deformation section) is the same as before:
only those vector prefixes (``displacement`` / ``velocity`` /
``acceleration``) that have ≥ 2 axis components recorded on nodes
across any stage are offered. When none qualify, the section is
disabled with a tooltip explaining why. The Threshold section is a
SCALAR list instead — the raw ``available_components()`` of the nodes
and gauss composites, the way the Add Diagram dialog resolves them —
and disables itself the same way when the file records none.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..core.threshold_controller import (
    TOPOLOGY_GAUSS,
    TOPOLOGY_NODES,
    GaussValues,
)

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
        the deformation controls are disabled.
    scalar_components
        ``{topology: [component, ...]}`` for the Threshold section,
        keyed by ``TOPOLOGY_NODES`` / ``TOPOLOGY_GAUSS``. Detected at
        viewer open via :func:`_kind_catalog._union_across_stages`.
        ``None`` or all-empty disables the section.
    """

    def __init__(
        self,
        director: "ResultsDirector",
        available_fields: list[str],
        scalar_components: "Optional[dict[str, list[str]]]" = None,
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._director = director
        self._available_fields = list(available_fields)
        self._scalars: dict[str, list[str]] = {
            topo: list((scalar_components or {}).get(topo) or ())
            for topo in (TOPOLOGY_NODES, TOPOLOGY_GAUSS)
        }
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

        # ── Stage pin (ADR 0058 S3b) ──────────────────────────────
        # "Follow active stage" (None = the default) + every REAL
        # stage (the synthetic combined entry is excluded — pinning
        # to "all stages" is meaningless). A pinned geometry shows
        # its stage's state at the global cursor clamped into the
        # pinned range while the viewport scrubs another stage.
        self._combo_stage_pin = QtWidgets.QComboBox()
        self._combo_stage_pin.addItem("Follow active stage", None)
        try:
            stages = [
                s for s in self._director.stages()
                if getattr(s, "kind", None) != "combined"
            ]
        except Exception:
            stages = []
        for s in stages:
            self._combo_stage_pin.addItem(str(s.name), s.id)
        self._combo_stage_pin.currentIndexChanged.connect(
            self._fire_stage_pin,
        )
        self._combo_stage_pin.setToolTip(
            "Pin this geometry to one stage — it shows that stage's "
            "state while the time scrubber drives another."
        )
        stage_pin_form = QtWidgets.QFormLayout()
        stage_pin_form.setContentsMargins(0, 0, 0, 0)
        stage_pin_form.setSpacing(6)
        stage_pin_form.addRow("Stage", self._combo_stage_pin)
        outer.addLayout(stage_pin_form)
        if len(stages) <= 1:
            self._combo_stage_pin.setEnabled(False)
            self._combo_stage_pin.setToolTip(
                "Single-stage file — nothing to pin against."
            )

        # ── Threshold section (ADR 0084 D1) ──────────────────────
        # Hides every cell whose scalar values fall outside [min, max]
        # at the current step. Not a diagram: it writes a visibility
        # LAYER through the geometry's ElementVisibility, so it
        # composes with the manual hide and the dim filter.
        sep_thr = QtWidgets.QFrame()
        sep_thr.setFrameShape(QtWidgets.QFrame.HLine)
        sep_thr.setFrameShadow(QtWidgets.QFrame.Sunken)
        outer.addWidget(sep_thr)

        threshold_label = QtWidgets.QLabel("Threshold")
        threshold_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(threshold_label)

        self._cb_threshold = QtWidgets.QCheckBox("Threshold")
        self._cb_threshold.toggled.connect(self._fire_threshold_enabled)
        outer.addWidget(self._cb_threshold)

        threshold_form = QtWidgets.QFormLayout()
        threshold_form.setContentsMargins(0, 0, 0, 0)
        threshold_form.setSpacing(6)

        self._combo_thr_topology = QtWidgets.QComboBox()
        for topo, caption in (
            (TOPOLOGY_NODES, "Nodes"), (TOPOLOGY_GAUSS, "Gauss"),
        ):
            if self._scalars[topo]:
                self._combo_thr_topology.addItem(caption, topo)
        self._combo_thr_topology.setToolTip(
            "Which table the component is read from. The same name can "
            "exist on both — this is not inferred."
        )
        self._combo_thr_topology.currentIndexChanged.connect(
            self._fire_threshold_topology,
        )
        threshold_form.addRow("Values on", self._combo_thr_topology)
        # One topology available = nothing to choose; the row stays
        # visible so the user can still see WHICH table is being read.
        self._combo_thr_topology.setEnabled(
            self._combo_thr_topology.count() > 1,
        )

        self._combo_thr_component = QtWidgets.QComboBox()
        self._combo_thr_component.currentIndexChanged.connect(
            self._fire_threshold_component,
        )
        threshold_form.addRow("Component", self._combo_thr_component)

        self._sb_thr_min = QtWidgets.QDoubleSpinBox()
        self._sb_thr_max = QtWidgets.QDoubleSpinBox()
        for sb in (self._sb_thr_min, self._sb_thr_max):
            sb.setRange(-1e30, 1e30)
            sb.setDecimals(6)
            sb.valueChanged.connect(self._fire_threshold_range)
        threshold_form.addRow("Min", self._sb_thr_min)
        threshold_form.addRow("Max", self._sb_thr_max)

        outer.addLayout(threshold_form)

        self._btn_thr_reset = QtWidgets.QPushButton("Reset to data range")
        self._btn_thr_reset.setToolTip(
            "Re-read the component's range at the current step and "
            "widen the threshold back to it."
        )
        self._btn_thr_reset.clicked.connect(self._fire_threshold_reset)
        outer.addWidget(self._btn_thr_reset)

        self._threshold_widgets = (
            self._cb_threshold, self._combo_thr_topology,
            self._combo_thr_component, self._sb_thr_min, self._sb_thr_max,
            self._btn_thr_reset, threshold_label,
        )
        self._reload_threshold_components()
        if not any(self._scalars.values()):
            tip = "No scalar node or gauss components in this file."
            for w in self._threshold_widgets:
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

        # Per-geometry spatial offset (ADR 0058 S3a) — rigid X/Y/Z
        # translation in model units, applied at pump time so two
        # geometries can sit side by side.
        offset_row = QtWidgets.QHBoxLayout()
        self._sb_offset: list = []
        for axis in ("X", "Y", "Z"):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-1e9, 1e9)
            sb.setDecimals(3)
            sb.setSingleStep(1.0)
            sb.setValue(0.0)
            sb.setToolTip(f"Offset {axis} (model units)")
            sb.valueChanged.connect(self._fire_offset)
            offset_row.addWidget(sb)
            self._sb_offset.append(sb)
        display_form.addRow("Offset", offset_row)

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

            self._combo_stage_pin.blockSignals(True)
            pin_idx = self._combo_stage_pin.findData(geom.stage_id)
            self._combo_stage_pin.setCurrentIndex(
                pin_idx if pin_idx >= 0 else 0,
            )
            self._combo_stage_pin.blockSignals(False)

            thr = self._director.thresholds.settings_for(geom.id)
            self._cb_threshold.blockSignals(True)
            self._cb_threshold.setChecked(thr is not None)
            self._cb_threshold.blockSignals(False)
            if thr is not None:
                self._combo_thr_topology.blockSignals(True)
                topo_idx = self._combo_thr_topology.findData(thr.topology)
                if topo_idx >= 0:
                    self._combo_thr_topology.setCurrentIndex(topo_idx)
                self._combo_thr_topology.blockSignals(False)
                self._reload_threshold_components()
                self._combo_thr_component.blockSignals(True)
                comp_idx = self._combo_thr_component.findData(thr.component)
                if comp_idx >= 0:
                    self._combo_thr_component.setCurrentIndex(comp_idx)
                self._combo_thr_component.blockSignals(False)
                for sb, value in (
                    (self._sb_thr_min, thr.lo), (self._sb_thr_max, thr.hi),
                ):
                    sb.blockSignals(True)
                    sb.setValue(float(value))
                    sb.blockSignals(False)

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

            for sb, value in zip(self._sb_offset, geom.offset):
                sb.blockSignals(True)
                sb.setValue(float(value))
                sb.blockSignals(False)
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

    def _fire_stage_pin(self, _idx: int) -> None:
        if self._reflecting or self._geom_id is None:
            return
        stage_id = self._combo_stage_pin.currentData()
        from .._log import log_action
        log_action(
            "ui.geometry", "stage_pin_changed",
            geom=self._geom_id,
            stage=None if stage_id is None else str(stage_id),
        )
        self._director.geometries.set_stage_pin(self._geom_id, stage_id)

    def _fire_offset(self, _value: float) -> None:
        if self._reflecting or self._geom_id is None:
            return
        offset = tuple(float(sb.value()) for sb in self._sb_offset)
        from .._log import log_action
        log_action(
            "ui.geometry", "offset_changed",
            geom=self._geom_id, offset=str(offset),
        )
        self._director.geometries.set_offset(self._geom_id, offset)

    # ── Threshold (ADR 0084 D1) ───────────────────────────────────
    # The owner is the director's ThresholdController: it records the
    # per-geometry spec but applies nothing itself, so every handler
    # below ends in one ``STEP_CHANGED`` — the single path that
    # recomputes the mask at the current cursor and repaints.

    def _reload_threshold_components(self) -> None:
        """Repopulate the component combo for the selected topology."""
        topo = self._combo_thr_topology.currentData() or TOPOLOGY_NODES
        self._combo_thr_component.blockSignals(True)
        self._combo_thr_component.clear()
        for comp in self._scalars.get(topo, ()):
            self._combo_thr_component.addItem(comp, comp)
        self._combo_thr_component.blockSignals(False)

    def _threshold_data_range(
        self, component: str, topology: str,
    ) -> "Optional[tuple[float, float]]":
        """Finite (min, max) of ``component`` at the current step.

        Read through the controller's own value reader so the seed
        cannot disagree with what the mask will be computed from.
        Advisory only — it fills the spin boxes; the mask itself is
        always recomputed per geometry with that geometry's pin.
        """
        import numpy as np

        director = self._director
        try:
            values = director.thresholds.read_values(
                component, int(director.local_step_for_active_stage()),
                stage_id=None, topology=topology,
            )
        except Exception:
            return None
        if values is None:
            return None
        if isinstance(values, GaussValues):
            # A gauss read is per INTEGRATION POINT (the all-values rule
            # reduces later, once lo/hi are known) — seed from those.
            values = values.values
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        return (float(finite.min()), float(finite.max()))

    def _seed_range_boxes(self, component: str, topology: str) -> None:
        """Refill the min / max boxes from the component's data range."""
        data_range = self._threshold_data_range(component, topology)
        if data_range is None:
            return
        self._reflecting = True
        try:
            self._sb_thr_min.setValue(data_range[0])
            self._sb_thr_max.setValue(data_range[1])
        finally:
            self._reflecting = False

    def _push_threshold(self, *, seed_range: bool) -> None:
        """Write the combo/spin state onto the owner and repaint.

        ``seed_range`` refills the spin boxes from the component's data
        range first — used when the user enables the section or changes
        component / topology, so they never start from an empty (or a
        previous component's) band.
        """
        component = self._combo_thr_component.currentData()
        if component is None:
            return
        topology = self._combo_thr_topology.currentData() or TOPOLOGY_NODES
        if seed_range:
            self._seed_range_boxes(str(component), topology)
        self._director.thresholds.set_threshold(
            self._geom_id,
            component=str(component),
            lo=float(self._sb_thr_min.value()),
            hi=float(self._sb_thr_max.value()),
            topology=topology,
        )
        self._fire_step_changed()

    def _fire_step_changed(self) -> None:
        from ..diagrams._dispatch import STEP_CHANGED
        self._director.dispatcher.fire(STEP_CHANGED)

    def _fire_threshold_enabled(self, checked: bool) -> None:
        if self._reflecting or self._geom_id is None:
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "threshold_toggled",
            geom=self._geom_id, enabled=bool(checked),
            component=self._combo_thr_component.currentData(),
        )
        if not checked:
            self._director.thresholds.clear_threshold(self._geom_id)
            self._fire_step_changed()
            return
        self._push_threshold(seed_range=True)

    def _fire_threshold_topology(self, _idx: int) -> None:
        if self._reflecting or self._geom_id is None:
            return
        self._reload_threshold_components()
        if not self._cb_threshold.isChecked():
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "threshold_topology_changed",
            geom=self._geom_id,
            topology=str(self._combo_thr_topology.currentData()),
        )
        self._push_threshold(seed_range=True)

    def _fire_threshold_component(self, _idx: int) -> None:
        if self._reflecting or self._geom_id is None:
            return
        if not self._cb_threshold.isChecked():
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "threshold_component_changed",
            geom=self._geom_id,
            component=str(self._combo_thr_component.currentData()),
        )
        self._push_threshold(seed_range=True)

    def _fire_threshold_range(self, _value: float) -> None:
        if self._reflecting or self._geom_id is None:
            return
        if not self._cb_threshold.isChecked():
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "threshold_range_changed",
            geom=self._geom_id,
            lo=float(self._sb_thr_min.value()),
            hi=float(self._sb_thr_max.value()),
        )
        self._push_threshold(seed_range=False)

    def _fire_threshold_reset(self) -> None:
        if self._reflecting or self._geom_id is None:
            return
        from .._log import log_action
        log_action(
            "ui.geometry", "threshold_reset", geom=self._geom_id,
        )
        if self._cb_threshold.isChecked():
            self._push_threshold(seed_range=True)
            return
        # Disabled: re-seed the boxes only — there is no mask to widen.
        component = self._combo_thr_component.currentData()
        if component is not None:
            self._seed_range_boxes(
                str(component),
                self._combo_thr_topology.currentData() or TOPOLOGY_NODES,
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
