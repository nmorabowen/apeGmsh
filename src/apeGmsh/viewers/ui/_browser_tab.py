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

from ._eye_icon_delegate import ROLE_VISIBLE, resolve_delegate_class

if TYPE_CHECKING:
    from ..core.selection import SelectionState
    from ..core.visibility import VisibilityManager


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
        on_hide: Callable[[list[tuple[int, int]]], None] | None = None,
        on_isolate: Callable[[list[tuple[int, int]]], None] | None = None,
        on_reveal_all: Callable[[], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._selection = selection
        self._on_group_activated = on_group_activated
        self._on_entity_toggled = on_entity_toggled
        self._on_new_group = on_new_group
        self._on_rename_group = on_rename_group
        self._on_delete_group = on_delete_group
        self._on_hide = on_hide
        self._on_isolate = on_isolate
        self._on_reveal_all = on_reveal_all
        # ParaView-style eye-icon visibility. Wired by
        # :meth:`bind_vis_mgr` after construction because the
        # ``VisibilityManager`` is created later in
        # ``ModelViewer.show()`` (it needs the registry + plotter).
        self._vis_mgr: "VisibilityManager | None" = None
        self._eye_delegate: Any = None

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

    def bind_vis_mgr(self, vis_mgr: "VisibilityManager") -> None:
        """Late-binding hook — wires the eye-icon visibility column.

        ``ModelViewer.show()`` creates the visibility manager AFTER
        the BrowserTab (the manager needs the actor registry and
        plotter, both of which are built later in the same method).
        Calling this hook installs the eye-icon delegate, subscribes
        the tab to the manager's ``on_changed`` chain, and stamps
        ``ROLE_VISIBLE`` on every existing row.
        """
        if self._vis_mgr is not None:
            return    # already bound — idempotent
        self._vis_mgr = vis_mgr
        delegate_cls = resolve_delegate_class()
        self._eye_delegate = delegate_cls(self._tree)
        self._eye_delegate.icon_clicked.connect(self._on_eye_clicked)
        self._tree.setItemDelegateForColumn(0, self._eye_delegate)
        vis_mgr.on_changed.append(self._refresh_eye_states)
        self._refresh_eye_states()

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
            if self._vis_mgr is not None:
                item.setData(0, ROLE_VISIBLE, self._group_is_visible(members))

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
                if self._vis_mgr is not None:
                    child.setData(
                        0,
                        ROLE_VISIBLE,
                        not self._vis_mgr.is_hidden((dim, tag)),
                    )

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
        if not data:
            return
        kind = data[0]
        if kind not in ("group", "entity"):
            return

        # Resolve target DimTag list for visibility actions
        if kind == "group":
            name = data[1]
            dts: list[tuple[int, int]] = []
            for i in range(item.childCount()):
                cd = item.child(i).data(0, 0x0100)
                if cd and cd[0] == "entity":
                    dts.append(cd[1])
        else:  # entity
            name = None
            dts = [data[1]]

        menu = QtWidgets.QMenu()
        act_rename = act_delete = None
        if kind == "group":
            act_rename = menu.addAction("Rename")
            act_delete = menu.addAction("Delete")

        # Visibility actions — only added when callbacks were provided
        act_hide = act_isolate = act_reveal = None
        if self._on_hide or self._on_isolate or self._on_reveal_all:
            if kind == "group":
                menu.addSeparator()
            n = len(dts)
            if self._on_hide and dts:
                act_hide = menu.addAction(f"Hide ({n})")
            if self._on_isolate and dts:
                act_isolate = menu.addAction(f"Isolate ({n})")
            if self._on_reveal_all:
                act_reveal = menu.addAction("Reveal all")

        action = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if action is None:
            return
        if action == act_rename and self._on_rename_group:
            self._on_rename_group(name)
        elif action == act_delete and self._on_delete_group:
            self._on_delete_group(name)
        elif action == act_hide and self._on_hide:
            self._on_hide(dts)
        elif action == act_isolate and self._on_isolate:
            self._on_isolate(dts)
        elif action == act_reveal and self._on_reveal_all:
            self._on_reveal_all()

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

    # ------------------------------------------------------------------
    # Eye-icon visibility (ParaView-style — plan 03 v2 follow-up)
    # ------------------------------------------------------------------

    def _group_is_visible(self, members: list[tuple[int, int]]) -> bool:
        """Return True if ANY member is currently visible. Matches the
        union semantics ``results.viewer``'s outline uses for parent
        rows. Empty groups read as visible so the eye doesn't look
        disabled on a brand-new (empty) group."""
        if not members or self._vis_mgr is None:
            return True
        return any(not self._vis_mgr.is_hidden(dt) for dt in members)

    def _on_eye_clicked(self, item) -> None:
        """Toggle visibility for the clicked row.

        Group rows hide / unhide every member; entity rows toggle just
        the one. The manager's ``set_hidden`` rebuild + ``on_changed``
        fire trigger :meth:`_refresh_eye_states` to repaint the tree.
        """
        if item is None or self._vis_mgr is None:
            return
        data = item.data(0, 0x0100)
        if not data:
            return
        kind = data[0]
        current_hidden = set(self._vis_mgr.hidden)
        if kind == "group":
            members = []
            for i in range(item.childCount()):
                cd = item.child(i).data(0, 0x0100)
                if cd and cd[0] == "entity":
                    members.append(cd[1])
            if not members:
                return
            visible_now = self._group_is_visible(members)
            if visible_now:
                current_hidden.update(members)
            else:
                current_hidden.difference_update(members)
            self._vis_mgr.set_hidden(current_hidden)
        elif kind == "entity":
            dt = data[1]
            if dt in current_hidden:
                current_hidden.discard(dt)
            else:
                current_hidden.add(dt)
            self._vis_mgr.set_hidden(current_hidden)

    def _refresh_eye_states(self) -> None:
        """Update every row's ``ROLE_VISIBLE`` from the manager.

        Called via ``vis_mgr.on_changed`` — covers programmatic
        visibility changes (Hide / Isolate / Reveal all from the
        context menu) so the eye icons stay in sync without a full
        tree rebuild.
        """
        if self._vis_mgr is None:
            return
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            data = item.data(0, 0x0100)
            if data and data[0] == "group":
                members = []
                for j in range(item.childCount()):
                    cd = item.child(j).data(0, 0x0100)
                    if cd and cd[0] == "entity":
                        members.append(cd[1])
                item.setData(
                    0, ROLE_VISIBLE, self._group_is_visible(members),
                )
                for j in range(item.childCount()):
                    child = item.child(j)
                    cd = child.data(0, 0x0100)
                    if cd and cd[0] == "entity":
                        child.setData(
                            0,
                            ROLE_VISIBLE,
                            not self._vis_mgr.is_hidden(cd[1]),
                        )
        try:
            self._tree.viewport().update()
        except Exception:
            pass


__all__ = ["BrowserTab"]
