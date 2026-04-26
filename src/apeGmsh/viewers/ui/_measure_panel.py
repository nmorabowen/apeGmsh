"""
MeasurePanel — Qt tab driving the entity-distance overlay.

Controls:

* "Measure mode" checkbox — when ON, picks route to the overlay
  instead of toggling selections.  The viewer wires this via
  :meth:`is_active`.
* "Clear" button — wipes any drawn segment and resets the state machine.
* Status block — live readout of how many points are captured, the
  endpoint entity labels, and the resulting distance + delta vector.

The panel itself never touches the overlay or the pick engine; the
viewer wires the callbacks.
"""
from __future__ import annotations

from typing import Callable


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


_HINT = (
    "Click two entities to measure the distance between their centroids.\n"
    "A third click clears the previous result and starts a new measurement."
)


class MeasurePanel:
    """Tab widget for :class:`MeasureOverlay`."""

    def __init__(
        self,
        *,
        on_active_changed: Callable[[bool], None] | None = None,
        on_clear: Callable[[], None] | None = None,
    ) -> None:
        QtWidgets, _, _ = _qt()
        self._on_active_changed = on_active_changed
        self._on_clear = on_clear

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Mode toggle ─────────────────────────────────────────────
        self._cb_active = QtWidgets.QCheckBox("Measure mode")
        self._cb_active.setChecked(False)
        self._cb_active.toggled.connect(self._on_toggle)
        layout.addWidget(self._cb_active)

        hint = QtWidgets.QLabel(_HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        # ── Status group ────────────────────────────────────────────
        status_group = QtWidgets.QGroupBox("Result")
        form = QtWidgets.QFormLayout(status_group)
        form.setSpacing(4)
        self._lbl_state = QtWidgets.QLabel("inactive")
        self._lbl_endpoints = QtWidgets.QLabel("—")
        self._lbl_distance = QtWidgets.QLabel("—")
        self._lbl_delta = QtWidgets.QLabel("—")
        form.addRow("state:", self._lbl_state)
        form.addRow("endpoints:", self._lbl_endpoints)
        form.addRow("|d|:", self._lbl_distance)
        form.addRow("Δ (x,y,z):", self._lbl_delta)
        layout.addWidget(status_group)

        # ── Clear ───────────────────────────────────────────────────
        btn_clear = QtWidgets.QPushButton("Clear measurement")
        btn_clear.clicked.connect(self._on_clear_clicked)
        layout.addWidget(btn_clear)

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # External
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._cb_active.isChecked()

    def update_status(
        self,
        num_points: int,
        endpoints: tuple[str, str] | None,
        distance: float | None,
        delta: tuple[float, float, float] | None,
    ) -> None:
        """Re-render every label from a single update from the viewer."""
        if not self.is_active():
            self._lbl_state.setText("inactive")
        elif num_points == 0:
            self._lbl_state.setText("waiting for first pick")
        elif num_points == 1:
            self._lbl_state.setText("waiting for second pick")
        else:
            self._lbl_state.setText("done — pick again to start over")

        if endpoints is None:
            self._lbl_endpoints.setText("—")
        else:
            self._lbl_endpoints.setText(f"{endpoints[0]}  →  {endpoints[1]}")

        if distance is None:
            self._lbl_distance.setText("—")
        else:
            self._lbl_distance.setText(f"{distance:.6g}")

        if delta is None:
            self._lbl_delta.setText("—")
        else:
            dx, dy, dz = delta
            self._lbl_delta.setText(f"({dx:.4g}, {dy:.4g}, {dz:.4g})")

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _on_toggle(self, checked: bool) -> None:
        if self._on_active_changed is not None:
            self._on_active_changed(checked)

    def _on_clear_clicked(self) -> None:
        if self._on_clear is not None:
            self._on_clear()


__all__ = ["MeasurePanel"]
