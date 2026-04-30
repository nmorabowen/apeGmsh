"""Stages tab — list of analysis stages with selection.

For each stage shows:

* name
* kind (``transient`` / ``static`` / ``mode``)
* n_steps
* time range (min … max) for transient/static; eigenvalue/freq for modes

Single-click a row to set it as the active stage. The Director's
``set_stage`` triggers a re-attach of every diagram against the new
scoped Results.

Phase 0 ships the basic list. Modal browsing (next/prev, animation)
piggybacks on this same panel in Phase 6.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class StagesTab:
    """Read-mostly stage list with active-stage selection."""

    def __init__(self, director: "ResultsDirector") -> None:
        QtWidgets, QtCore = _qt()
        self._director = director

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        info = QtWidgets.QLabel(
            "Click a stage to make it active. The active stage drives "
            "the time scrubber and all diagrams."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(
            ["Name", "Kind", "Steps", "Range"]
        )
        self._tree.setRootIsDecorated(False)
        self._tree.setUniformRowHeights(True)
        self._tree.itemActivated.connect(self._on_activate)
        self._tree.itemClicked.connect(self._on_activate)
        layout.addWidget(self._tree, stretch=1)

        self._widget = widget
        self.populate()

        director.subscribe_stage(lambda _id: self._highlight_active())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def populate(self) -> None:
        """Rebuild the tree from the director's stage list."""
        self._tree.clear()
        QtWidgets, _ = _qt()
        for s in self._director.stages():
            item = QtWidgets.QTreeWidgetItem(
                [
                    str(getattr(s, "name", s.id)),
                    str(getattr(s, "kind", "")),
                    str(getattr(s, "n_steps", 0)),
                    self._range_text(s),
                ]
            )
            item.setData(0, 0x100, getattr(s, "id", str(s)))   # Qt.UserRole
            self._tree.addTopLevelItem(item)
        for col in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(col)
        self._highlight_active()

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def _on_activate(self, item, _column: int = 0) -> None:
        if item is None:
            return
        sid = item.data(0, 0x100)
        if not sid:
            return
        if sid == self._director.stage_id:
            return
        self._director.set_stage(sid)

    def _highlight_active(self) -> None:
        active = self._director.stage_id
        QtWidgets, QtCore = _qt()
        for i in range(self._tree.topLevelItemCount()):
            it = self._tree.topLevelItem(i)
            sid = it.data(0, 0x100)
            font = it.font(0)
            font.setBold(sid == active)
            for col in range(self._tree.columnCount()):
                it.setFont(col, font)
        self._tree.viewport().update()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _range_text(s: Any) -> str:
        kind = getattr(s, "kind", "")
        if kind == "mode":
            f = getattr(s, "frequency_hz", None)
            if f is not None:
                return f"f = {f:.4g} Hz"
            ev = getattr(s, "eigenvalue", None)
            if ev is not None:
                return f"λ = {ev:.4g}"
            return ""
        n = int(getattr(s, "n_steps", 0) or 0)
        return f"{n} steps" if n else ""
