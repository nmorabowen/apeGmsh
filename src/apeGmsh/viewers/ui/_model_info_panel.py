"""
ModelInfoPanel — read-only diagnostic tab for the BRep model viewer.

Shows the current Gmsh model state at a glance:

* Bounding box (min, max, diagonal)
* Entity counts per dimension (P / C / S / V)
* User-facing physical group count (label PGs excluded)
* Part instance count (when ``g.parts`` exists)

The panel does not auto-refresh — call ``refresh()`` (or click the
"Refresh" button) after operations that mutate geometry, e.g. boolean
fragment / fuse, or after rebuilding the scene.
"""
from __future__ import annotations

from typing import Any

import gmsh


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


_DIM_LABEL = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}


class ModelInfoPanel:
    """Read-only diagnostic panel.

    Parameters
    ----------
    parts_registry : optional
        ``g.parts`` if available — exposes ``.instances`` (a dict keyed
        by part label).  ``None`` hides the part-count row.
    """

    def __init__(self, parts_registry: Any = None) -> None:
        QtWidgets, _, _ = _qt()
        self._parts_registry = parts_registry

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Bounding box group ──────────────────────────────────────
        bbox_group = QtWidgets.QGroupBox("Bounding box")
        bbox_form = QtWidgets.QFormLayout(bbox_group)
        bbox_form.setSpacing(4)
        self._lbl_min = QtWidgets.QLabel("—")
        self._lbl_max = QtWidgets.QLabel("—")
        self._lbl_size = QtWidgets.QLabel("—")
        self._lbl_diag = QtWidgets.QLabel("—")
        bbox_form.addRow("min:", self._lbl_min)
        bbox_form.addRow("max:", self._lbl_max)
        bbox_form.addRow("size:", self._lbl_size)
        bbox_form.addRow("diagonal:", self._lbl_diag)
        layout.addWidget(bbox_group)

        # ── Entity count group ──────────────────────────────────────
        ent_group = QtWidgets.QGroupBox("Entity counts")
        ent_form = QtWidgets.QFormLayout(ent_group)
        ent_form.setSpacing(4)
        self._lbl_ents: dict[int, Any] = {}
        for d in (0, 1, 2, 3):
            lbl = QtWidgets.QLabel("0")
            ent_form.addRow(f"{_DIM_LABEL[d]} (dim={d}):", lbl)
            self._lbl_ents[d] = lbl
        layout.addWidget(ent_group)

        # ── Group / instance counts ─────────────────────────────────
        misc_group = QtWidgets.QGroupBox("Groups & instances")
        misc_form = QtWidgets.QFormLayout(misc_group)
        misc_form.setSpacing(4)
        self._lbl_pgs = QtWidgets.QLabel("0")
        misc_form.addRow("Physical groups:", self._lbl_pgs)
        self._lbl_parts: Any | None = None
        if parts_registry is not None:
            self._lbl_parts = QtWidgets.QLabel("0")
            misc_form.addRow("Part instances:", self._lbl_parts)
        layout.addWidget(misc_group)

        # ── Refresh button ──────────────────────────────────────────
        btn = QtWidgets.QPushButton("Refresh")
        btn.clicked.connect(self.refresh)
        layout.addWidget(btn)
        layout.addStretch(1)

        self.refresh()

    def refresh(self) -> None:
        """Re-query Gmsh and update every label."""
        # Bounding box — gmsh raises if the model is empty.
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
            self._lbl_min.setText(_fmt_xyz(bb[0], bb[1], bb[2]))
            self._lbl_max.setText(_fmt_xyz(bb[3], bb[4], bb[5]))
            dx, dy, dz = bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]
            self._lbl_size.setText(_fmt_xyz(dx, dy, dz))
            diag = (dx * dx + dy * dy + dz * dz) ** 0.5
            self._lbl_diag.setText(f"{diag:.6g}")
        except Exception:
            for lbl in (self._lbl_min, self._lbl_max,
                        self._lbl_size, self._lbl_diag):
                lbl.setText("(empty model)")

        # Entity counts
        for d in (0, 1, 2, 3):
            try:
                n = len(gmsh.model.getEntities(d))
            except Exception:
                n = 0
            self._lbl_ents[d].setText(str(n))

        # Physical groups (skip internal label PGs)
        try:
            from apeGmsh.core.Labels import is_label_pg
            n_pgs = 0
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                try:
                    name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
                except Exception:
                    name = ""
                if not is_label_pg(name):
                    n_pgs += 1
            self._lbl_pgs.setText(str(n_pgs))
        except Exception:
            self._lbl_pgs.setText("?")

        # Part instances
        if self._lbl_parts is not None and self._parts_registry is not None:
            try:
                n_parts = len(self._parts_registry.instances)
            except Exception:
                n_parts = 0
            self._lbl_parts.setText(str(n_parts))


def _fmt_xyz(x: float, y: float, z: float) -> str:
    return f"({x:.6g}, {y:.6g}, {z:.6g})"


__all__ = ["ModelInfoPanel"]
