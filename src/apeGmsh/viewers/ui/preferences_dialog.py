"""
PreferencesDialog — modal Qt editor for the persistent preferences.

Reads / writes via ``PREFERENCES`` (``PreferencesManager``). Preferences
apply to *new* viewer invocations — already-open viewers keep their
session state. Close and reopen a viewer to pick up the change.

Usage::

    from apeGmsh.viewers.ui.preferences_dialog import open_preferences_dialog
    open_preferences_dialog()          # spins up QApplication if needed
    open_preferences_dialog(parent)    # modal over existing window
"""
from __future__ import annotations

from typing import Any

from .preferences_manager import (
    ANTI_ALIASING_CHOICES,
    DEFAULT_PREFERENCES,
    PREFERENCES,
    TAB_POSITION_CHOICES,
)


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


def _make_dspin(
    QtWidgets: Any,
    *,
    value: float,
    lo: float,
    hi: float,
    decimals: int = 1,
    step: float = 1.0,
) -> Any:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(float(value))
    return s


def _make_ispin(
    QtWidgets: Any, *, value: int, lo: int, hi: int,
) -> Any:
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(int(value))
    return s


def _make_combo(
    QtWidgets: Any, *, choices: tuple[str, ...], value: str,
) -> Any:
    c = QtWidgets.QComboBox()
    c.addItems(list(choices))
    if value in choices:
        c.setCurrentIndex(choices.index(value))
    return c


class PreferencesDialog:
    """Modal dialog — internal QTabWidget groups preferences by domain."""

    def __init__(self, parent: Any = None) -> None:
        QtWidgets, _ = _qt()
        p = PREFERENCES.current

        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("apeGmsh — Global preferences")
        self.dialog.setModal(True)
        self.dialog.resize(520, 560)

        root = QtWidgets.QVBoxLayout(self.dialog)
        tabs = QtWidgets.QTabWidget()

        # ── Rendering tab ───────────────────────────────────────────
        rtab = QtWidgets.QWidget()
        rform = QtWidgets.QFormLayout(rtab)
        self._sp_point = _make_dspin(QtWidgets, value=p.point_size, lo=1, hi=50)
        rform.addRow("BRep point size", self._sp_point)
        self._sp_line = _make_dspin(QtWidgets, value=p.line_width, lo=0.5, hi=20, step=0.5)
        rform.addRow("BRep line width", self._sp_line)
        self._sp_opacity = _make_dspin(
            QtWidgets, value=p.surface_opacity, lo=0, hi=1, decimals=2, step=0.05,
        )
        rform.addRow("BRep surface opacity", self._sp_opacity)
        self._cb_edges = QtWidgets.QCheckBox("Show surface edges on BRep")
        self._cb_edges.setChecked(p.show_surface_edges)
        rform.addRow(self._cb_edges)
        self._cb_smooth = QtWidgets.QCheckBox("Smooth shading (off = flat matte)")
        self._cb_smooth.setChecked(p.smooth_shading)
        rform.addRow(self._cb_smooth)
        self._cmb_aa = _make_combo(
            QtWidgets, choices=ANTI_ALIASING_CHOICES, value=p.anti_aliasing,
        )
        rform.addRow("Anti-aliasing", self._cmb_aa)
        tabs.addTab(rtab, "Rendering")

        # ── Mesh tab ────────────────────────────────────────────────
        mtab = QtWidgets.QWidget()
        mform = QtWidgets.QFormLayout(mtab)
        self._sp_node = _make_dspin(QtWidgets, value=p.node_marker_size, lo=1, hi=50)
        mform.addRow("Node marker size", self._sp_node)
        self._sp_mlw = _make_dspin(
            QtWidgets, value=p.mesh_line_width, lo=0.5, hi=20, step=0.5,
        )
        mform.addRow("Mesh line width", self._sp_mlw)
        self._sp_mop = _make_dspin(
            QtWidgets, value=p.mesh_surface_opacity, lo=0, hi=1, decimals=2, step=0.05,
        )
        mform.addRow("Mesh surface opacity", self._sp_mop)
        self._cb_medges = QtWidgets.QCheckBox("Show mesh element edges")
        self._cb_medges.setChecked(p.mesh_show_surface_edges)
        mform.addRow(self._cb_medges)
        self._sp_fa = _make_dspin(
            QtWidgets, value=p.feature_angle, lo=0, hi=90, step=1,
        )
        mform.addRow("Silhouette feature angle (°)", self._sp_fa)
        tabs.addTab(mtab, "Mesh")

        # ── Labels tab ──────────────────────────────────────────────
        ltab = QtWidgets.QWidget()
        lform = QtWidgets.QFormLayout(ltab)
        self._sp_nlf = _make_ispin(QtWidgets, value=p.node_label_font_size, lo=6, hi=32)
        lform.addRow("Node label font size", self._sp_nlf)
        self._sp_elf = _make_ispin(QtWidgets, value=p.element_label_font_size, lo=6, hi=32)
        lform.addRow("Element label font size", self._sp_elf)
        self._sp_ent = _make_ispin(QtWidgets, value=p.entity_label_font_size, lo=6, hi=32)
        lform.addRow("Entity (BRep) label font size", self._sp_ent)
        self._sp_omf = _make_ispin(QtWidgets, value=p.origin_marker_font_size, lo=6, hi=32)
        lform.addRow("Origin marker label font size", self._sp_omf)
        self._sp_cp = _make_ispin(QtWidgets, value=p.coord_precision, lo=0, hi=8)
        lform.addRow("Coord label decimals", self._sp_cp)
        tabs.addTab(ltab, "Labels")

        # ── Axis & Markers tab ──────────────────────────────────────
        atab = QtWidgets.QWidget()
        aform = QtWidgets.QFormLayout(atab)
        self._sp_alw = _make_dspin(QtWidgets, value=p.axis_line_width, lo=0.5, hi=10, step=0.5)
        aform.addRow("Axis widget line width", self._sp_alw)
        self._cb_al = QtWidgets.QCheckBox("Show axis labels (X/Y/Z)")
        self._cb_al.setChecked(p.axis_labels_visible)
        aform.addRow(self._cb_al)
        self._sp_omsize = _make_dspin(
            QtWidgets, value=p.origin_marker_size, lo=1, hi=100,
        )
        aform.addRow("Origin marker glyph size", self._sp_omsize)
        self._cb_coords = QtWidgets.QCheckBox("Show origin-marker coord labels")
        self._cb_coords.setChecked(p.origin_marker_show_coords)
        aform.addRow(self._cb_coords)
        self._cb_origin = QtWidgets.QCheckBox("Always include world origin (0,0,0)")
        self._cb_origin.setChecked(p.origin_marker_include_world_origin)
        aform.addRow(self._cb_origin)
        tabs.addTab(atab, "Axis & Markers")

        # ── Interaction & UI tab ────────────────────────────────────
        itab = QtWidgets.QWidget()
        iform = QtWidgets.QFormLayout(itab)
        self._sp_dt = _make_ispin(QtWidgets, value=p.drag_threshold, lo=1, hi=50)
        iform.addRow("Drag threshold (px)", self._sp_dt)
        self._cmb_tp = _make_combo(
            QtWidgets, choices=TAB_POSITION_CHOICES, value=p.tab_position,
        )
        iform.addRow("Dock tab position", self._cmb_tp)
        self._sp_dmw = _make_ispin(QtWidgets, value=p.dock_min_width, lo=200, hi=800)
        iform.addRow("Dock min width (px)", self._sp_dmw)
        self._cb_max = QtWidgets.QCheckBox("Open viewers maximized")
        self._cb_max.setChecked(p.window_maximized)
        iform.addRow(self._cb_max)
        self._cb_con = QtWidgets.QCheckBox("Show console dock")
        self._cb_con.setChecked(p.show_console)
        iform.addRow(self._cb_con)
        tabs.addTab(itab, "Interaction & UI")

        root.addWidget(tabs)

        # ── Hint + path + buttons ───────────────────────────────────
        hint = QtWidgets.QLabel(
            "Changes apply to <b>new</b> viewers. Close and reopen to see them."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        file_label = QtWidgets.QLabel(f"File: {PREFERENCES.path}")
        file_label.setWordWrap(True)
        file_label.setStyleSheet("color: gray; font-size: 10px;")
        root.addWidget(file_label)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.dialog.reject)
        btns.button(
            QtWidgets.QDialogButtonBox.StandardButton.Reset
        ).clicked.connect(self._on_reset)
        root.addWidget(btns)

    # ── handlers ─────────────────────────────────────────────────────

    def _collect_overrides(self) -> dict[str, Any]:
        return {
            # BRep
            "point_size": self._sp_point.value(),
            "line_width": self._sp_line.value(),
            "surface_opacity": self._sp_opacity.value(),
            "show_surface_edges": self._cb_edges.isChecked(),
            # Mesh
            "node_marker_size": self._sp_node.value(),
            "mesh_line_width": self._sp_mlw.value(),
            "mesh_surface_opacity": self._sp_mop.value(),
            "mesh_show_surface_edges": self._cb_medges.isChecked(),
            # Rendering
            "smooth_shading": self._cb_smooth.isChecked(),
            "anti_aliasing": self._cmb_aa.currentText(),
            "feature_angle": self._sp_fa.value(),
            # Labels
            "node_label_font_size": self._sp_nlf.value(),
            "element_label_font_size": self._sp_elf.value(),
            "entity_label_font_size": self._sp_ent.value(),
            "origin_marker_font_size": self._sp_omf.value(),
            "coord_precision": self._sp_cp.value(),
            # Axis widget
            "axis_line_width": self._sp_alw.value(),
            "axis_labels_visible": self._cb_al.isChecked(),
            # Origin markers
            "origin_marker_size": self._sp_omsize.value(),
            "origin_marker_show_coords": self._cb_coords.isChecked(),
            "origin_marker_include_world_origin": self._cb_origin.isChecked(),
            # Interaction & UI
            "drag_threshold": self._sp_dt.value(),
            "tab_position": self._cmb_tp.currentText(),
            "dock_min_width": self._sp_dmw.value(),
            "window_maximized": self._cb_max.isChecked(),
            "show_console": self._cb_con.isChecked(),
        }

    def _on_accept(self) -> None:
        PREFERENCES.update(self._collect_overrides())
        self.dialog.accept()

    def _on_reset(self) -> None:
        PREFERENCES.reset()
        d = DEFAULT_PREFERENCES
        # BRep
        self._sp_point.setValue(d.point_size)
        self._sp_line.setValue(d.line_width)
        self._sp_opacity.setValue(d.surface_opacity)
        self._cb_edges.setChecked(d.show_surface_edges)
        # Mesh
        self._sp_node.setValue(d.node_marker_size)
        self._sp_mlw.setValue(d.mesh_line_width)
        self._sp_mop.setValue(d.mesh_surface_opacity)
        self._cb_medges.setChecked(d.mesh_show_surface_edges)
        # Rendering
        self._cb_smooth.setChecked(d.smooth_shading)
        self._cmb_aa.setCurrentIndex(ANTI_ALIASING_CHOICES.index(d.anti_aliasing))
        self._sp_fa.setValue(d.feature_angle)
        # Labels
        self._sp_nlf.setValue(d.node_label_font_size)
        self._sp_elf.setValue(d.element_label_font_size)
        self._sp_ent.setValue(d.entity_label_font_size)
        self._sp_omf.setValue(d.origin_marker_font_size)
        self._sp_cp.setValue(d.coord_precision)
        # Axis widget
        self._sp_alw.setValue(d.axis_line_width)
        self._cb_al.setChecked(d.axis_labels_visible)
        # Origin markers
        self._sp_omsize.setValue(d.origin_marker_size)
        self._cb_coords.setChecked(d.origin_marker_show_coords)
        self._cb_origin.setChecked(d.origin_marker_include_world_origin)
        # Interaction & UI
        self._sp_dt.setValue(d.drag_threshold)
        self._cmb_tp.setCurrentIndex(TAB_POSITION_CHOICES.index(d.tab_position))
        self._sp_dmw.setValue(d.dock_min_width)
        self._cb_max.setChecked(d.window_maximized)
        self._cb_con.setChecked(d.show_console)

    def exec(self) -> int:
        return self.dialog.exec()


def open_preferences_dialog(parent: Any = None) -> int:
    """Open the preferences dialog (spins up a QApplication if none exists).

    Returns the dialog result code (``QDialog.Accepted`` / ``Rejected``).
    """
    QtWidgets, _ = _qt()
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])
    dlg = PreferencesDialog(parent=parent)
    result = dlg.exec()
    if owns_app:
        pass
    return int(result)
