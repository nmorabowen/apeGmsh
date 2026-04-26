"""
ClippingTab — Toggle/reset controls for the section clipping plane.
"""
from __future__ import annotations

from typing import Callable

from qtpy import QtWidgets


class ClippingTab:
    """Two-button panel: Enable/Disable + Reset plane."""

    def __init__(
        self,
        *,
        on_toggle: Callable[[], bool],
        on_reset: Callable[[], None],
    ) -> None:
        self._on_toggle = on_toggle
        self._on_reset = on_reset

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        info = QtWidgets.QLabel(
            "Toggle the section plane on, then drag the 3D handle "
            "to slice the mesh."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._btn_toggle = QtWidgets.QPushButton("Enable clipping plane")
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.toggled.connect(self._on_btn_toggled)
        layout.addWidget(self._btn_toggle)

        self._btn_reset = QtWidgets.QPushButton("Reset plane")
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._btn_reset)

        layout.addStretch()

    def _on_btn_toggled(self, checked: bool) -> None:
        new_state = self._on_toggle()
        # Keep the button label/state in sync with the controller's truth
        self._btn_toggle.blockSignals(True)
        self._btn_toggle.setChecked(new_state)
        self._btn_toggle.setText(
            "Disable clipping plane" if new_state else "Enable clipping plane"
        )
        self._btn_toggle.blockSignals(False)

    def _on_reset_clicked(self) -> None:
        self._on_reset()
