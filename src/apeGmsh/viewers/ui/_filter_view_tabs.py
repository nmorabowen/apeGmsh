"""
FilterTab + ViewTab — small display-toggle panels that sit alongside
the BrowserTab in the model viewer.

* :class:`FilterTab` — per-dimension pick filter. Checkboxes control
  which entity dimensions respond to mouse picks.
* :class:`ViewTab` — label overlay toggles. Controls which dim-tag
  overlays are drawn in the 3D viewport and their styling.
"""
from __future__ import annotations

from typing import Any, Callable


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


# ======================================================================
# FilterTab — Dimension pick filter
# ======================================================================

class FilterTab:
    """Dimension checkboxes for filtering which entities respond to picks."""

    def __init__(
        self,
        dims: list[int],
        *,
        on_filter_changed: Callable[[set[int]], None] | None = None,
    ) -> None:
        QtWidgets, _, _ = _qt()
        self._on_filter_changed = on_filter_changed
        self._active_dims = set(dims)

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)

        filter_group = QtWidgets.QGroupBox("Pick Filter")
        filter_layout = QtWidgets.QVBoxLayout(filter_group)

        self._checkboxes: dict[int, Any] = {}  # QCheckBox (lazy Qt import)
        dim_labels = {0: "Points (dim=0)", 1: "Curves (dim=1)",
                      2: "Surfaces (dim=2)", 3: "Volumes (dim=3)"}
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(dim_labels.get(d, f"dim={d}"))
            cb.setChecked(True)
            cb.toggled.connect(self._on_toggled)
            self._checkboxes[d] = cb
            filter_layout.addWidget(cb)

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("All")
        btn_all.clicked.connect(self._select_all)
        btn_row.addWidget(btn_all)
        btn_none = QtWidgets.QPushButton("None")
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_none)
        filter_layout.addLayout(btn_row)

        layout.addWidget(filter_group)

        layout.addStretch()

    def _on_toggled(self, _checked: bool) -> None:
        active = set()
        for d, cb in self._checkboxes.items():
            if cb.isChecked():
                active.add(d)
        self._active_dims = active
        if self._on_filter_changed:
            self._on_filter_changed(active)

    def _select_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _select_none(self):
        for cb in self._checkboxes.values():
            cb.setChecked(False)


# ======================================================================
# ViewTab — Entity label overlays
# ======================================================================

_DIM_NAMES = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}
_DIM_ABBR = {0: "P", 1: "C", 2: "S", 3: "V"}


class ViewTab:
    """Toggle entity label overlays per dimension in the 3D viewport."""

    def __init__(
        self,
        dims: list[int],
        *,
        on_labels_changed: Callable[..., None] | None = None,
        on_geometry_probes_changed: Callable[[bool, bool], None] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dims : list[int]
            Available dimensions.
        on_labels_changed : callable
            ``fn(active_dims_dict, font_size, use_names, show_parts)`` called when
            any toggle or setting changes.
        on_geometry_probes_changed : callable
            ``fn(show_tangents, show_normals)`` — fires on either toggle.
        """
        QtWidgets, _, _ = _qt()
        self._on_labels_changed = on_labels_changed
        self._on_geometry_probes_changed = on_geometry_probes_changed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Dim checkboxes ──────────────────────────────────────────
        group = QtWidgets.QGroupBox("Show entity labels on screen")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._dim_cbs: dict[int, Any] = {}  # QCheckBox (lazy Qt import)
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(f"{_DIM_NAMES.get(d, f'dim={d}')} tags")
            cb.setChecked(False)
            cb.toggled.connect(self._fire)
            group_layout.addWidget(cb)
            self._dim_cbs[d] = cb

        layout.addWidget(group)

        # ── Instance / entity labels group ─────────────────────────
        labels_group = QtWidgets.QGroupBox("Instance & entity labels")
        labels_layout = QtWidgets.QVBoxLayout(labels_group)

        self._show_parts = QtWidgets.QCheckBox("Show part instance labels")
        self._show_parts.setChecked(False)
        self._show_parts.toggled.connect(self._fire)
        labels_layout.addWidget(self._show_parts)

        self._show_entity_labels = QtWidgets.QCheckBox("Show entity labels (Tier 1)")
        self._show_entity_labels.setChecked(False)
        self._show_entity_labels.toggled.connect(self._fire)
        labels_layout.addWidget(self._show_entity_labels)

        layout.addWidget(labels_group)

        # ── Label style ─────────────────────────────────────────────
        style_group = QtWidgets.QGroupBox("Label style")
        style_layout = QtWidgets.QFormLayout(style_group)
        style_layout.setSpacing(4)

        from .preferences_manager import PREFERENCES as _PREF_FS
        self._font_size = QtWidgets.QSpinBox()
        self._font_size.setRange(6, 24)
        self._font_size.setValue(_PREF_FS.current.entity_label_font_size)
        self._font_size.valueChanged.connect(self._fire)
        style_layout.addRow("Font size", self._font_size)

        self._use_names = QtWidgets.QCheckBox("Show names instead of tags")
        self._use_names.setChecked(False)
        self._use_names.toggled.connect(self._fire)
        style_layout.addRow(self._use_names)

        layout.addWidget(style_group)

        # ── Geometry probes ─────────────────────────────────────────
        probes_group = QtWidgets.QGroupBox("Geometry probes")
        probes_layout = QtWidgets.QVBoxLayout(probes_group)

        self._show_tangents = QtWidgets.QCheckBox("Show curve tangents (dim=1)")
        self._show_tangents.setChecked(False)
        self._show_tangents.toggled.connect(self._fire_probes)
        probes_layout.addWidget(self._show_tangents)

        self._show_normals = QtWidgets.QCheckBox("Show surface normals (dim=2)")
        self._show_normals.setChecked(False)
        self._show_normals.toggled.connect(self._fire_probes)
        probes_layout.addWidget(self._show_normals)

        layout.addWidget(probes_group)
        layout.addStretch(1)

    def _fire_probes(self, *_args) -> None:
        if self._on_geometry_probes_changed is None:
            return
        self._on_geometry_probes_changed(
            self._show_tangents.isChecked(),
            self._show_normals.isChecked(),
        )

    def _fire(self, *_args) -> None:
        if self._on_labels_changed is None:
            return
        active = {d: cb.isChecked() for d, cb in self._dim_cbs.items()}
        font_size = self._font_size.value()
        use_names = self._use_names.isChecked()
        show_parts = self._show_parts.isChecked()
        show_entity_labels = self._show_entity_labels.isChecked()
        self._on_labels_changed(
            active, font_size, use_names, show_parts, show_entity_labels,
        )


__all__ = ["FilterTab", "ViewTab"]
