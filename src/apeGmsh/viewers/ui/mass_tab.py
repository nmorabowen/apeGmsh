"""
MassTabPanel — read-only mass display for the model viewer.

Renders a flat list of MassDef objects (mass has no patterns) plus
a stats panel and a single "Show mass overlays" toggle.  Toggling
the checkbox triggers ``on_overlay_changed(bool)`` so the viewer
can show / hide sphere glyphs in the 3-D viewport.

Read-only — never modifies state.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh.core.MassesComposite import MassesComposite
    from apeGmsh.mesh.FEMData import FEMData


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


class MassTabPanel:
    """Read-only mass display: overlay toggle + def list + stats."""

    def __init__(
        self,
        mass_composite: "MassesComposite",
        fem: "FEMData | None" = None,
        *,
        on_overlay_changed: Callable[[bool], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._QtGui = QtGui
        self._mass = mass_composite
        self._fem = fem
        self._on_overlay_changed = on_overlay_changed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._header = QtWidgets.QLabel("Mass")
        self._header.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self._header)

        # ── empty state ──────────────────────────────────────
        self._empty_label = QtWidgets.QLabel(
            "No mass defined.\n\n"
            "Use g.masses.point(), g.masses.volume(), etc."
        )
        self._empty_label.setStyleSheet("color: #6c7086; padding: 12px;")
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        # ── needs-fem warning ────────────────────────────────
        self._fem_warning = QtWidgets.QLabel(
            "Pass fem= to g.model.viewer() to display\n"
            "mass sphere overlays in the 3-D viewport."
        )
        self._fem_warning.setStyleSheet(
            "color: #f9e2af; padding: 6px; "
            "border: 1px solid #f9e2af; border-radius: 3px;"
        )
        self._fem_warning.setWordWrap(True)
        layout.addWidget(self._fem_warning)

        # ── overlay toggle ───────────────────────────────────
        self._show_cb = QtWidgets.QCheckBox("Show mass overlays")
        self._show_cb.setChecked(False)
        self._show_cb.toggled.connect(self._on_toggle)
        layout.addWidget(self._show_cb)

        # ── def list ─────────────────────────────────────────
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Mass def", "Detail"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(False)
        layout.addWidget(self._tree)

        # ── stats panel ──────────────────────────────────────
        stats_box = QtWidgets.QGroupBox("Stats")
        stats_layout = QtWidgets.QFormLayout(stats_box)
        self._lbl_total = QtWidgets.QLabel("—")
        self._lbl_n_nodes = QtWidgets.QLabel("—")
        self._lbl_max = QtWidgets.QLabel("—")
        self._lbl_min = QtWidgets.QLabel("—")
        stats_layout.addRow("Total mass:", self._lbl_total)
        stats_layout.addRow("Nodes with mass:", self._lbl_n_nodes)
        stats_layout.addRow("Max @ node:", self._lbl_max)
        stats_layout.addRow("Min @ node:", self._lbl_min)
        layout.addWidget(stats_box)
        self._stats_box = stats_box

        # ── refresh button ───────────────────────────────────
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        layout.addWidget(btn_refresh)

        self.refresh()

    # ── Build ───────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-read mass_defs and (if available) fem.masses."""
        from qtpy.QtWidgets import QTreeWidgetItem

        self._tree.clear()

        defs = self._mass.mass_defs
        has_defs = bool(defs)
        self._empty_label.setVisible(not has_defs)
        self._tree.setVisible(has_defs)
        self._show_cb.setVisible(has_defs)
        self._stats_box.setVisible(has_defs)
        self._fem_warning.setVisible(self._fem is None and has_defs)

        n = len(defs)
        self._header.setText(
            f"Mass ({n} def{'s' if n != 1 else ''})"
            if has_defs else "Mass"
        )

        if not has_defs:
            return

        for d in defs:
            item = QTreeWidgetItem(self._tree)
            item.setText(0, f"{d.kind} -> {d.target}")
            item.setText(1, self._format_def_detail(d))

        self._tree.resizeColumnToContents(0)

        # Stats from fem.masses (if resolved)
        if self._fem is not None and self._fem.masses:
            self._update_stats()
        else:
            self._lbl_total.setText("—")
            self._lbl_n_nodes.setText("—")
            self._lbl_max.setText("—")
            self._lbl_min.setText("—")

    def _format_def_detail(self, d) -> str:
        from apeGmsh.solvers.Masses import (
            PointMassDef, LineMassDef, SurfaceMassDef, VolumeMassDef,
        )
        if isinstance(d, PointMassDef):
            rot = f", I={tuple(d.rotational)}" if d.rotational else ""
            return f"m={d.mass:.4g} kg{rot}"
        if isinstance(d, LineMassDef):
            return f"ρₗ={d.linear_density:.4g} kg/m"
        if isinstance(d, SurfaceMassDef):
            return f"ρₐ={d.areal_density:.4g} kg/m²"
        if isinstance(d, VolumeMassDef):
            return f"ρ={d.density:.4g} kg/m³"
        return "(unknown)"

    def _update_stats(self) -> None:
        ms = self._fem.masses
        records = list(ms)
        if not records:
            return
        total = ms.total_mass()
        self._lbl_total.setText(f"{total:.4g} kg")
        self._lbl_n_nodes.setText(str(len(records)))
        # Max / min by translational mass (mx)
        max_rec = max(records, key=lambda r: r.mass[0])
        min_rec = min(records, key=lambda r: r.mass[0])
        self._lbl_max.setText(
            f"{max_rec.mass[0]:.4g} kg @ node {max_rec.node_id}"
        )
        self._lbl_min.setText(
            f"{min_rec.mass[0]:.4g} kg @ node {min_rec.node_id}"
        )

    # ── State queries ───────────────────────────────────────

    def show_overlays(self) -> bool:
        return self._show_cb.isChecked()

    # ── Handlers ────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        if self._on_overlay_changed:
            self._on_overlay_changed(checked)

    def set_fem(self, fem) -> None:
        """Update the FEM reference (e.g. after re-meshing)."""
        self._fem = fem
        self.refresh()
