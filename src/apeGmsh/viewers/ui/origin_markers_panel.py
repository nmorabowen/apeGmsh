"""
OriginMarkersPanel — Qt tab for managing reference-point markers.

Lean v1:
- Show/hide toggle for the overlay
- Show/hide toggle for coord labels
- List of current markers (``(x, y, z)`` text)
- "Add..." opens a coord dialog; "Remove" deletes the selected row

All actions fire callbacks the viewer wires to ``OriginMarkerOverlay``.
The panel holds no scene state of its own.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


def _fmt_point(p: tuple[float, float, float], precision: int = 2) -> str:
    return f"({p[0]:.{precision}f}, {p[1]:.{precision}f}, {p[2]:.{precision}f})"


class _AddMarkerDialog:
    """Modal x/y/z input dialog. Returns (x,y,z) or None on cancel."""

    @staticmethod
    def get_point(parent: Any) -> tuple[float, float, float] | None:
        QtWidgets, _, _ = _qt()
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("Add reference marker")
        form = QtWidgets.QFormLayout(dlg)

        spins: list[Any] = []
        for label in ("X", "Y", "Z"):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-1e9, 1e9)
            s.setDecimals(6)
            s.setSingleStep(1.0)
            s.setValue(0.0)
            form.addRow(label, s)
            spins.append(s)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return (spins[0].value(), spins[1].value(), spins[2].value())


class OriginMarkersPanel:
    """Tab widget for show/hide + add/remove of reference markers."""

    def __init__(
        self,
        initial_points: Sequence[tuple[float, float, float]],
        *,
        initial_visible: bool = True,
        initial_show_coords: bool = True,
        initial_size: float = 10.0,
        on_visible_changed: Callable[[bool], None] | None = None,
        on_show_coords_changed: Callable[[bool], None] | None = None,
        on_marker_added: Callable[[tuple[float, float, float]], None] | None = None,
        on_marker_removed: Callable[[int], None] | None = None,
        on_size_changed: Callable[[float], None] | None = None,
    ) -> None:
        QtWidgets, _, _ = _qt()
        self._on_added = on_marker_added
        self._on_removed = on_marker_removed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Visibility group ─────────────────────────────────────────
        vis_group = QtWidgets.QGroupBox("Visibility")
        vis_layout = QtWidgets.QVBoxLayout(vis_group)

        self._cb_visible = QtWidgets.QCheckBox("Show markers")
        self._cb_visible.setChecked(initial_visible)
        if on_visible_changed:
            self._cb_visible.toggled.connect(on_visible_changed)
        vis_layout.addWidget(self._cb_visible)

        self._cb_labels = QtWidgets.QCheckBox("Show coord labels")
        self._cb_labels.setChecked(initial_show_coords)
        if on_show_coords_changed:
            self._cb_labels.toggled.connect(on_show_coords_changed)
        vis_layout.addWidget(self._cb_labels)

        layout.addWidget(vis_group)

        # ── Markers list group ───────────────────────────────────────
        list_group = QtWidgets.QGroupBox("Markers")
        list_layout = QtWidgets.QVBoxLayout(list_group)

        self._list = QtWidgets.QListWidget()
        for p in initial_points:
            self._list.addItem(_fmt_point(p))
        list_layout.addWidget(self._list)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_add = QtWidgets.QPushButton("Add...")
        self._btn_add.clicked.connect(self._handle_add)
        btn_row.addWidget(self._btn_add)
        self._btn_remove = QtWidgets.QPushButton("Remove")
        self._btn_remove.clicked.connect(self._handle_remove)
        btn_row.addWidget(self._btn_remove)
        list_layout.addLayout(btn_row)

        # ── Size control ─────────────────────────────────────────────
        size_row = QtWidgets.QFormLayout()
        self._spin_size = QtWidgets.QDoubleSpinBox()
        self._spin_size.setRange(1.0, 100.0)
        self._spin_size.setDecimals(1)
        self._spin_size.setSingleStep(1.0)
        self._spin_size.setValue(initial_size)
        if on_size_changed:
            self._spin_size.valueChanged.connect(on_size_changed)
        size_row.addRow("Glyph size", self._spin_size)
        list_layout.addLayout(size_row)

        layout.addWidget(list_group)
        layout.addStretch(1)

    # ── handlers ─────────────────────────────────────────────────────

    def _handle_add(self) -> None:
        p = _AddMarkerDialog.get_point(self.widget)
        if p is None:
            return
        self._list.addItem(_fmt_point(p))
        if self._on_added:
            self._on_added(p)

    def _handle_remove(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._list.takeItem(row)
        if self._on_removed:
            self._on_removed(row)
