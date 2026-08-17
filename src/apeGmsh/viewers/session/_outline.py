"""SessionOutline — the panes group of the session outline (§9).

ADR 0098 S2: one job of the outline's two (panes; the model
composition half — scope checkboxes and select-all actions — lands
with S3/S4). Lists every pane of the session under one "Views"
group; selecting one drives the inspector (and, for mesh views, aims
the single S2 viewport). A projection of the session — it rebuilds
from ``session.panes`` on the change tick and holds no pane state of
its own.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from qtpy import QtCore, QtWidgets

from .._failures import safe_slot

_PANE_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole


class SessionOutline:
    """The outline tree, panes group only.

    ``on_pane_selected`` is called with the pane id whenever the
    user's selection lands on a pane row.
    """

    def __init__(
        self,
        session: Any,
        *,
        on_pane_selected: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._session = session
        self._on_pane_selected = on_pane_selected
        self._listed: tuple[tuple[str, Optional[str]], ...] = ()
        self._syncing = False

        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("session_outline_tree")
        tree.setHeaderHidden(True)
        tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection,
        )
        self._tree = tree
        self._group = QtWidgets.QTreeWidgetItem(tree, ["Views"])
        self._group.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled,  # a header, not a choice
        )
        tree.currentItemChanged.connect(safe_slot(self._on_current_changed))

        session.subscribe(self._on_session_tick)
        self.refresh()

    @property
    def widget(self) -> QtWidgets.QTreeWidget:
        return self._tree

    def dispose(self) -> None:
        self._session.unsubscribe(self._on_session_tick)

    # -- selection -----------------------------------------------------

    def selected_pane_id(self) -> Optional[str]:
        item = self._tree.currentItem()
        if item is None:
            return None
        pane_id = item.data(0, _PANE_ID_ROLE)
        return str(pane_id) if pane_id is not None else None

    def select_pane(self, pane_id: str) -> None:
        """Programmatic selection (boot default: the first pane).
        Fires ``on_pane_selected`` like a user click would."""
        for i in range(self._group.childCount()):
            item = self._group.child(i)
            if item.data(0, _PANE_ID_ROLE) == pane_id:
                self._tree.setCurrentItem(item)
                return
        raise KeyError(f"No pane {pane_id!r} in the outline.")

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the pane rows from ``session.panes``.

        Skips when membership and names are unchanged — the session
        ticks on EVERY mutation (slot fills included), and rebuilding
        rows on each would flicker the selection mid-gesture.
        """
        wanted = tuple(
            (pane.id, pane.name) for pane in self._session.panes
        )
        if wanted == self._listed:
            return
        selected = self.selected_pane_id()
        self._syncing = True
        try:
            self._group.takeChildren()
            for pane_id, name in wanted:
                item = QtWidgets.QTreeWidgetItem(
                    self._group, [name or pane_id],
                )
                item.setData(0, _PANE_ID_ROLE, pane_id)
                if pane_id == selected:
                    self._tree.setCurrentItem(item)
            self._group.setExpanded(True)
            self._listed = wanted
        finally:
            self._syncing = False

    # -- internals -----------------------------------------------------

    def _on_session_tick(self) -> None:
        self.refresh()

    def _on_current_changed(self, current: Any, _previous: Any) -> None:
        if self._syncing or current is None:
            return
        pane_id = current.data(0, _PANE_ID_ROLE)
        if pane_id is None or self._on_pane_selected is None:
            return
        self._on_pane_selected(str(pane_id))


__all__ = ["SessionOutline"]
