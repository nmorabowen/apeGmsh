"""Probes tab — buttons + result display for the probe overlay.

Buttons:

* **Point Probe** — click on the mesh; nearest-node values appear.
* **Line Probe** — click point A, click point B; samples N points
  along the line.
* **Plane Probe** — slice along an axis combo (X / Y / Z).
* **Stop** — cancel any active interactive probe.
* **Clear** — remove all probe markers and result history.

The text area below shows the latest result summary. Multiple results
accumulate on the overlay; only the latest renders here in v1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..overlays.probe_overlay import (
        LineProbeResult,
        PlaneProbeResult,
        PointProbeResult,
        ProbeOverlay,
    )


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class ProbesTab:
    """Right-dock tab with probe controls and result display."""

    def __init__(self, overlay: "ProbeOverlay") -> None:
        QtWidgets, _ = _qt()
        self._overlay = overlay

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Buttons ─────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_point = QtWidgets.QPushButton("Point")
        self._btn_point.setToolTip(
            "Click on the mesh — pick the nearest FEM node and read "
            "values from the active diagrams' components."
        )
        self._btn_point.clicked.connect(self._on_point)
        btn_row.addWidget(self._btn_point)

        self._btn_line = QtWidgets.QPushButton("Line")
        self._btn_line.setToolTip("Click A then B for line sampling.")
        self._btn_line.clicked.connect(self._on_line)
        btn_row.addWidget(self._btn_line)

        # Plane has an axis combo next to the button
        self._plane_axis = QtWidgets.QComboBox()
        for axis in ("x", "y", "z"):
            self._plane_axis.addItem(axis)
        self._btn_plane = QtWidgets.QPushButton("Plane")
        self._btn_plane.setToolTip(
            "Slice the mesh by an axis-aligned plane through the "
            "model center."
        )
        self._btn_plane.clicked.connect(self._on_plane)
        btn_row.addWidget(self._btn_plane)
        btn_row.addWidget(self._plane_axis)
        layout.addLayout(btn_row)

        ctrl_row = QtWidgets.QHBoxLayout()
        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.setToolTip("Cancel any active interactive probe.")
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self._btn_stop)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.setToolTip("Remove all probe markers and history.")
        self._btn_clear.clicked.connect(self._on_clear)
        ctrl_row.addWidget(self._btn_clear)

        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # ── Line-probe sample-count spinner ────────────────────────
        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(QtWidgets.QLabel("Line samples:"))
        self._n_samples_spin = QtWidgets.QSpinBox()
        self._n_samples_spin.setRange(2, 10000)
        self._n_samples_spin.setValue(50)
        self._n_samples_spin.setToolTip(
            "Number of evenly-spaced sample points along a line probe."
        )
        sample_row.addWidget(self._n_samples_spin)
        sample_row.addStretch(1)
        layout.addLayout(sample_row)

        # ── Result display ─────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("Latest result:"))
        self._result_text = QtWidgets.QPlainTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMinimumHeight(160)
        layout.addWidget(self._result_text, stretch=1)

        self._widget = widget

        # Subscribe to overlay callbacks
        overlay.on_point_result = self._on_point_result
        overlay.on_line_result = self._on_line_result
        overlay.on_plane_result = self._on_plane_result

    @property
    def widget(self):
        return self._widget

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_point(self) -> None:
        try:
            self._overlay.start_point_probe()
        except Exception as exc:
            self._result_text.setPlainText(f"Point probe failed: {exc}")

    def _on_line(self) -> None:
        try:
            self._overlay.start_line_probe()
        except Exception as exc:
            self._result_text.setPlainText(f"Line probe failed: {exc}")

    def _on_plane(self) -> None:
        axis = self._plane_axis.currentText()
        try:
            result = self._overlay.probe_with_plane(normal=axis)
        except Exception as exc:
            self._result_text.setPlainText(f"Plane probe failed: {exc}")
            return
        if result is None:
            self._result_text.setPlainText("Plane probe — no slice produced.")
            return
        self._on_plane_result(result)

    def _on_stop(self) -> None:
        try:
            self._overlay.stop()
        except Exception:
            pass

    def _on_clear(self) -> None:
        self._overlay.clear()
        self._result_text.clear()

    # ------------------------------------------------------------------
    # Overlay callbacks
    # ------------------------------------------------------------------

    def _on_point_result(self, result: "PointProbeResult") -> None:
        self._result_text.setPlainText(result.summary())

    def _on_line_result(self, result: "LineProbeResult") -> None:
        head = (
            f"Line probe — {result.n_samples} samples\n"
            f"  A = ({result.point_a[0]:.4g}, {result.point_a[1]:.4g}, "
            f"{result.point_a[2]:.4g})\n"
            f"  B = ({result.point_b[0]:.4g}, {result.point_b[1]:.4g}, "
            f"{result.point_b[2]:.4g})\n"
            f"  L = {result.total_length:.4g}\n"
            f"  step = {result.step_index}"
        )
        if not result.field_values:
            self._result_text.setPlainText(
                head + "\n  (no components — add diagrams to populate)"
            )
            return
        rows = [head]
        for name, values in result.field_values.items():
            rows.append(
                f"  {name:24s}  min={values.min():.6g}  "
                f"max={values.max():.6g}  mean={values.mean():.6g}"
            )
        self._result_text.setPlainText("\n".join(rows))

    def _on_plane_result(self, result: "PlaneProbeResult") -> None:
        text = (
            f"Plane probe\n"
            f"  origin = ({result.origin[0]:.4g}, {result.origin[1]:.4g}, "
            f"{result.origin[2]:.4g})\n"
            f"  normal = ({result.normal[0]:.4g}, {result.normal[1]:.4g}, "
            f"{result.normal[2]:.4g})\n"
            f"  slice points = {result.n_points}\n"
            f"  slice cells  = {result.slice_mesh.n_cells}"
        )
        self._result_text.setPlainText(text)
