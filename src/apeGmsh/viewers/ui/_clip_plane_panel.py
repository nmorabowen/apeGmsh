"""
ClipPlanePanel — Qt tab driving the section-view overlay.

Controls:

* Enable / disable the section
* Axis combo (X / Y / Z) — picks the plane normal
* Position slider — sweeps along the chosen axis between the model's
  world bbox extents (0..1000 integer slider mapped to ``[lo, hi]``)
* Flip checkbox — negates the normal so the opposite half is hidden

After the model bbox changes (e.g. parts fuse), call
:meth:`refresh_bbox` to remap the slider range.
"""
from __future__ import annotations

from typing import Sequence


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


_AXES = ("X", "Y", "Z")
_SLIDER_TICKS = 1000


class ClipPlanePanel:
    """Tab widget for :class:`ClipPlaneOverlay`."""

    def __init__(
        self,
        overlay,
        world_bbox: Sequence[float],
    ) -> None:
        QtWidgets, _, _ = _qt()
        self._overlay = overlay
        self._world_bbox = list(world_bbox)  # (xmin,ymin,zmin,xmax,ymax,zmax)
        self._axis: str = "X"

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Enable ──────────────────────────────────────────────────
        self._cb_enable = QtWidgets.QCheckBox("Enable section plane")
        self._cb_enable.setChecked(False)
        self._cb_enable.toggled.connect(self._on_enable)
        layout.addWidget(self._cb_enable)

        # ── Axis ────────────────────────────────────────────────────
        axis_row = QtWidgets.QHBoxLayout()
        axis_row.addWidget(QtWidgets.QLabel("Axis:"))
        self._combo_axis = QtWidgets.QComboBox()
        self._combo_axis.addItems(list(_AXES))
        self._combo_axis.currentTextChanged.connect(self._on_axis)
        axis_row.addWidget(self._combo_axis, 1)
        layout.addLayout(axis_row)

        # ── Position ────────────────────────────────────────────────
        pos_group = QtWidgets.QGroupBox("Position")
        pos_layout = QtWidgets.QVBoxLayout(pos_group)

        from qtpy import QtCore
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, _SLIDER_TICKS)
        self._slider.setValue(_SLIDER_TICKS // 2)
        self._slider.valueChanged.connect(self._on_slider)
        pos_layout.addWidget(self._slider)

        self._lbl_pos = QtWidgets.QLabel("position: —")
        pos_layout.addWidget(self._lbl_pos)

        self._lbl_range = QtWidgets.QLabel("range: —")
        self._lbl_range.setStyleSheet("color: gray;")
        pos_layout.addWidget(self._lbl_range)

        layout.addWidget(pos_group)

        # ── Flip ────────────────────────────────────────────────────
        self._cb_flip = QtWidgets.QCheckBox("Flip normal (hide opposite side)")
        self._cb_flip.setChecked(False)
        self._cb_flip.toggled.connect(self._on_flip)
        layout.addWidget(self._cb_flip)

        layout.addStretch(1)

        # Apply initial position so the overlay is consistent even
        # before the user touches anything (in case they enable first).
        self._push_position(emit=True)
        self._update_range_label()

    # ------------------------------------------------------------------
    # External
    # ------------------------------------------------------------------

    def refresh_bbox(self, world_bbox: Sequence[float]) -> None:
        """Update the slider's mapping to a new world bbox."""
        self._world_bbox = list(world_bbox)
        self._push_position(emit=True)
        self._update_range_label()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _axis_range(self) -> tuple[float, float]:
        idx = _AXES.index(self._axis)
        return self._world_bbox[idx], self._world_bbox[3 + idx]

    def _slider_to_world(self) -> float:
        lo, hi = self._axis_range()
        if hi <= lo:
            return lo
        t = self._slider.value() / float(_SLIDER_TICKS)
        return lo + t * (hi - lo)

    def _push_position(self, *, emit: bool) -> None:
        pos = self._slider_to_world()
        self._lbl_pos.setText(f"position: {pos:.6g}")
        if emit:
            self._overlay.set_world_position(pos)

    def _update_range_label(self) -> None:
        lo, hi = self._axis_range()
        self._lbl_range.setText(f"range: [{lo:.6g}, {hi:.6g}]")

    # ── Signal handlers ─────────────────────────────────────────────

    def _on_enable(self, checked: bool) -> None:
        self._overlay.set_enabled(checked)

    def _on_axis(self, text: str) -> None:
        if text not in _AXES:
            return
        self._axis = text
        self._overlay.set_axis(text)
        self._push_position(emit=True)
        self._update_range_label()

    def _on_slider(self, _val: int) -> None:
        self._push_position(emit=True)

    def _on_flip(self, checked: bool) -> None:
        self._overlay.set_flipped(checked)


__all__ = ["ClipPlanePanel"]
