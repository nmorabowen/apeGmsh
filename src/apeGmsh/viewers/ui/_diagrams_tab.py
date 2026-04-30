"""Diagrams tab — list of active diagrams with add / remove / reorder.

Phase 1: Add button is enabled and opens the AddDiagramDialog. Row
selection emits ``diagram_selected(diagram)`` so the Settings tab can
swap its content.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from ..diagrams._base import Diagram

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class DiagramsTab:
    """Diagram list + buttons. Phase 1: Add wires to the modal dialog."""

    def __init__(self, director: "ResultsDirector") -> None:
        QtWidgets, QtCore = _qt()
        self._director = director
        self._on_diagram_selected: Optional[Callable[[Optional[Diagram]], None]] = None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Toolbar row ─────────────────────────────────────────
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)

        self._btn_add = QtWidgets.QPushButton("Add…")
        self._btn_add.setToolTip("Add a new diagram")
        self._btn_add.clicked.connect(self._on_add)
        bar.addWidget(self._btn_add)

        self._btn_remove = QtWidgets.QPushButton("Remove")
        self._btn_remove.setEnabled(False)
        self._btn_remove.clicked.connect(self._on_remove)
        bar.addWidget(self._btn_remove)

        self._btn_up = QtWidgets.QPushButton("Up")
        self._btn_up.setEnabled(False)
        self._btn_up.clicked.connect(lambda: self._move_selected(-1))
        bar.addWidget(self._btn_up)

        self._btn_down = QtWidgets.QPushButton("Down")
        self._btn_down.setEnabled(False)
        self._btn_down.clicked.connect(lambda: self._move_selected(+1))
        bar.addWidget(self._btn_down)

        bar.addStretch(1)
        layout.addLayout(bar)

        # ── List ────────────────────────────────────────────────
        self._list = QtWidgets.QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemSelectionChanged.connect(self._on_selection_change)
        self._list.itemChanged.connect(self._on_item_check)
        layout.addWidget(self._list, stretch=1)

        empty_hint = QtWidgets.QLabel(
            "No diagrams yet. Click Add… to create one."
        )
        empty_hint.setWordWrap(True)
        empty_hint.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(empty_hint)
        self._empty_hint = empty_hint

        self._widget = widget

        director.subscribe_diagrams(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def refresh(self) -> None:
        QtWidgets, QtCore = _qt()
        self._list.blockSignals(True)
        try:
            self._list.clear()
            diagrams = self._director.registry.diagrams()
            for d in diagrams:
                item = QtWidgets.QListWidgetItem(d.display_label())
                # Visibility checkbox
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.Checked if d.is_visible else QtCore.Qt.Unchecked
                )
                self._list.addItem(item)
            self._empty_hint.setVisible(len(diagrams) == 0)
        finally:
            self._list.blockSignals(False)
        self._update_button_state()
        # Notify the settings tab of the (possibly cleared) selection
        if self._on_diagram_selected is not None:
            self._on_diagram_selected(self._currently_selected_diagram())

    def on_diagram_selected(
        self, callback: Callable[[Optional[Diagram]], None],
    ) -> None:
        """Register the callback fired when the list selection changes."""
        self._on_diagram_selected = callback

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        from ._add_diagram_dialog import AddDiagramDialog
        dlg = AddDiagramDialog(self._director, parent=self._widget)
        dlg.run()
        # The registry's on_changed observer will refresh the list.

    def _on_remove(self) -> None:
        idx = self._list.currentRow()
        if idx < 0:
            return
        self._director.registry.remove_at(idx)

    def _move_selected(self, delta: int) -> None:
        idx = self._list.currentRow()
        if idx < 0:
            return
        self._director.registry.move(idx, idx + delta)
        new_idx = max(0, min(idx + delta, self._list.count() - 1))
        self._list.setCurrentRow(new_idx)

    def _on_selection_change(self) -> None:
        self._update_button_state()
        if self._on_diagram_selected is not None:
            self._on_diagram_selected(self._currently_selected_diagram())

    def _on_item_check(self, item) -> None:
        QtWidgets, QtCore = _qt()
        idx = self._list.row(item)
        diagram = self._diagram_at(idx)
        if diagram is None:
            return
        visible = item.checkState() == QtCore.Qt.Checked
        if diagram.is_visible != visible:
            self._director.registry.set_visible(diagram, visible)

    def _update_button_state(self) -> None:
        n = self._list.count()
        idx = self._list.currentRow()
        has_sel = idx >= 0
        self._btn_remove.setEnabled(has_sel)
        self._btn_up.setEnabled(has_sel and idx > 0)
        self._btn_down.setEnabled(has_sel and 0 <= idx < n - 1)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _currently_selected_diagram(self) -> Optional[Diagram]:
        return self._diagram_at(self._list.currentRow())

    def _diagram_at(self, idx: int) -> Optional[Diagram]:
        diagrams = self._director.registry.diagrams()
        if 0 <= idx < len(diagrams):
            return diagrams[idx]
        return None

    # ------------------------------------------------------------------
    # Phase 1+ hook — kept for forward-compat (Phase 0 stub kept Add disabled)
    # ------------------------------------------------------------------

    def set_add_enabled(self, enabled: bool, tooltip: str | None = None) -> None:
        self._btn_add.setEnabled(enabled)
        if tooltip is not None:
            self._btn_add.setToolTip(tooltip)
