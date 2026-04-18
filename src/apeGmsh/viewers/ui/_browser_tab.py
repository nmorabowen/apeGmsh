"""
BrowserTab — physical-group tree for the BRep model viewer.

Tree shows user-facing physical groups (hides ``_label:``-prefixed
internal groups) + staged groups from the current selection state.
Clicking a group activates it for editing; right-clicking opens a
rename/delete context menu.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import gmsh

if TYPE_CHECKING:
    from ..core.selection import SelectionState


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


def _theme():
    from .theme import THEME
    return THEME


class BrowserTab:
    """Tree showing physical groups + unassigned entities.

    The tree supports:
    - Click group header -> activate that group for editing
    - Click entity leaf -> toggle its pick in the working set
    - Right-click group -> rename / delete context menu
    """

    def __init__(
        self,
        selection: "SelectionState",
        *,
        on_group_activated: Callable[[str], None] | None = None,
        on_entity_toggled: Callable[[tuple], None] | None = None,
        on_new_group: Callable[[], None] | None = None,
        on_rename_group: Callable[[str], None] | None = None,
        on_delete_group: Callable[[str], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._selection = selection
        self._on_group_activated = on_group_activated
        self._on_entity_toggled = on_entity_toggled
        self._on_new_group = on_new_group
        self._on_rename_group = on_rename_group
        self._on_delete_group = on_delete_group

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ─────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_new = QtWidgets.QPushButton("New Group")
        btn_new.clicked.connect(self._action_new)
        btn_row.addWidget(btn_new)
        btn_rename = QtWidgets.QPushButton("Rename")
        btn_rename.clicked.connect(self._action_rename)
        btn_row.addWidget(btn_rename)
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.clicked.connect(self._action_delete)
        btn_row.addWidget(btn_delete)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Tree ────────────────────────────────────────────────────
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Count"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemClicked.connect(self._on_tree_click)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree)

        # Populate
        self.refresh()

    def refresh(self) -> None:
        """Full rebuild of the tree from Gmsh physical groups.

        Call only when groups are created, deleted, or renamed.
        For pick changes, use :meth:`update_active` instead.
        """
        QtWidgets, _, QtGui = _qt()
        self._tree.clear()
        # Values are QTreeWidgetItem instances — Qt is lazy-imported so
        # we can't use the real type without circular imports.
        self._group_items: dict[str, Any] = {}

        # Collect groups from Gmsh, keyed by name. A single name may
        # span multiple dims (one gmsh PG per dim) — union their
        # members so the browser reflects the full selection.
        gmsh_groups: dict[str, list[tuple]] = {}
        for name, pg_dim, pg_tag, members in self._collect_groups():
            gmsh_groups.setdefault(name, []).extend(members)

        # Merge with staged groups
        all_groups: dict[str, list[tuple]] = dict(gmsh_groups)
        for name, members in self._selection.staged_groups.items():
            if name not in all_groups:
                all_groups[name] = members

        # Order: follow SelectionState._group_order (creation order),
        # then any Gmsh groups not in the order list (pre-existing)
        order = self._selection.group_order
        ordered_names: list[str] = []
        # Pre-existing Gmsh groups first (by original tag order)
        for name in gmsh_groups:
            if name not in order:
                ordered_names.append(name)
        # Then groups in creation order
        for name in order:
            if name in all_groups and name not in ordered_names:
                ordered_names.append(name)

        active = self._selection.active_group
        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}

        for name in ordered_names:
            members = all_groups[name]
            item = QtWidgets.QTreeWidgetItem(self._tree)
            item.setText(0, name)
            item.setText(1, str(len(members)))
            item.setData(0, 0x0100, ("group", name))

            if name == active:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.success)),
                )
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            else:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
                )

            item.setExpanded(False)
            for dim, tag in members:
                child = QtWidgets.QTreeWidgetItem(item)
                child.setText(0, f"{dim_labels.get(dim, '?')} {tag}")
                child.setData(0, 0x0100, ("entity", (dim, tag)))

            self._group_items[name] = item

    def update_active(self) -> None:
        """Lightweight update: refresh count and highlight of active group.

        Call on pick changes — does NOT rebuild the tree structure.
        """
        _, _, QtGui = _qt()
        active = self._selection.active_group
        active_count = len(self._selection.picks)

        for name, item in self._group_items.items():
            if name == active:
                item.setText(1, str(active_count))
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.success)),
                )
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            else:
                font = item.font(0)
                font.setBold(False)
                item.setFont(0, font)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
                )

    def _collect_groups(self) -> list[tuple[str, int, int, list[tuple]]]:
        """Return user-facing groups (skip internal labels).

        Returns ``[(name, dim, pg_tag, members), ...]`` sorted by tag.
        """
        from apeGmsh.core.Labels import is_label_pg
        raw = []
        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                name = f"Group_{pg_dim}_{pg_tag}"
            if is_label_pg(name):
                continue
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            members = [(pg_dim, int(t)) for t in ents]
            raw.append((name, pg_dim, pg_tag, members))
        raw.sort(key=lambda x: x[2])
        return raw

    def _on_tree_click(self, item, column):
        data = item.data(0, 0x0100)
        if not data:
            return
        kind = data[0]
        if kind == "group" and self._on_group_activated:
            self._on_group_activated(data[1])
        elif kind == "entity" and self._on_entity_toggled:
            self._on_entity_toggled(data[1])

    def _on_context_menu(self, pos):
        QtWidgets, _, _ = _qt()
        item = self._tree.itemAt(pos)
        if not item:
            return
        data = item.data(0, 0x0100)
        if not data or data[0] != "group":
            return
        name = data[1]
        menu = QtWidgets.QMenu()
        act_rename = menu.addAction("Rename")
        act_delete = menu.addAction("Delete")
        action = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if action == act_rename and self._on_rename_group:
            self._on_rename_group(name)
        elif action == act_delete and self._on_delete_group:
            self._on_delete_group(name)

    def _action_new(self):
        if self._on_new_group:
            self._on_new_group()

    def _action_rename(self):
        active = self._selection.active_group
        if active and self._on_rename_group:
            self._on_rename_group(active)

    def _action_delete(self):
        active = self._selection.active_group
        if active and self._on_delete_group:
            self._on_delete_group(active)


__all__ = ["BrowserTab"]
