"""ModelOutlineTree — left-rail navigator for ``model.viewer``.

ParaView-style outline tree showing what's in the model: physical
groups + parts. Sits in the left dock area as primary navigation,
parallel to :class:`OutlineTree` in ``results.viewer``.

The right-side ``BrowserTab`` continues to coexist during this
transition — same data, different surface. Once the outline is the
preferred navigator, the Browser tab can be removed in a follow-up.

Top-level groups
----------------
* **Physical Groups** — user-facing physical groups (skips internal
  ``_label:`` prefixed ones). Click activates the group (same as
  the Browser tab); the eye toggles visibility for every member via
  :class:`VisibilityManager`.
* **Parts** — the session's :class:`PartsRegistry` instances, when
  present. Click a part to select all its entities; the eye toggles
  visibility for the union of its entities.

Each leaf row (entity DimTag) gets its own eye icon for per-entity
control. Right-click menus handle rename/delete on groups and
Hide/Isolate/Reveal-all on any visibility-bearing row.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import gmsh

from ._eye_icon_delegate import ROLE_VISIBLE, resolve_delegate_class

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..core.selection import SelectionState
    from ..core.visibility import VisibilityManager


def _qt():
    from qtpy import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def _theme():
    from .theme import THEME
    return THEME


# Role constants — distinct from the Browser tab's so the two trees
# can coexist without subtle confusion if a future refactor pulls
# items between them.
_ROLE_KIND = int(0x0200)        # "group" | "entity" | "part" | "header"
_ROLE_PAYLOAD = int(0x0201)     # name (for group/part) | DimTag (for entity)


class ModelOutlineTree:
    """Left-rail outline tree for ``model.viewer``.

    Parameters
    ----------
    selection
        The viewer's :class:`SelectionState` — drives the active-group
        highlight + click-to-toggle on entity rows.
    vis_mgr
        The viewer's :class:`VisibilityManager` — read for eye state,
        mutated on click.
    parts_registry
        Optional :class:`PartsRegistry` (``g.parts``). When ``None``,
        the Parts group is hidden.
    on_group_activated
        Callback fired when a Physical Group row is clicked — same
        contract as :class:`BrowserTab.on_group_activated`.
    on_entity_toggled
        Callback fired when an entity leaf is clicked — same contract
        as :class:`BrowserTab.on_entity_toggled`.
    on_rename_group / on_delete_group / on_new_group
        Optional handlers wired into the group-row context menu.
    """

    def __init__(
        self,
        selection: "SelectionState",
        vis_mgr: "VisibilityManager",
        parts_registry: Any = None,
        *,
        on_group_activated: Optional[Callable[[str], None]] = None,
        on_entity_toggled: Optional[Callable[["DimTag"], None]] = None,
        on_rename_group: Optional[Callable[[str], None]] = None,
        on_delete_group: Optional[Callable[[str], None]] = None,
        on_new_group: Optional[Callable[[], None]] = None,
        on_row_focused: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._selection = selection
        self._vis_mgr = vis_mgr
        self._parts = parts_registry
        self._on_group_activated = on_group_activated
        self._on_entity_toggled = on_entity_toggled
        self._on_rename_group = on_rename_group
        self._on_delete_group = on_delete_group
        self._on_new_group = on_new_group
        # Generic row-focused signal — fires for every selectable row
        # with ``(kind, payload)``. Viewers map kinds to tab names and
        # call ``win.focus_tab(...)`` to reveal the property editor.
        self._on_row_focused = on_row_focused

        # ── Outer container + header ────────────────────────────────
        widget = QtWidgets.QWidget()
        widget.setObjectName("ModelOutlineTree")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("OutlineHeader")
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 4, 6, 4)
        label = QtWidgets.QLabel("OUTLINE")
        label.setObjectName("OutlineHeaderLabel")
        header_lay.addWidget(label)
        header_lay.addStretch(1)
        if self._on_new_group is not None:
            btn_new = QtWidgets.QPushButton("+")
            btn_new.setFlat(True)
            btn_new.setFixedWidth(24)
            btn_new.setToolTip("New physical group")
            btn_new.clicked.connect(lambda: self._on_new_group())
            header_lay.addWidget(btn_new)
        layout.addWidget(header)

        # ── Tree ────────────────────────────────────────────────────
        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("ModelOutlineTreeWidget")
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(14)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tree.itemClicked.connect(self._on_item_clicked)
        tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(tree, stretch=1)
        self._tree = tree

        # ── Eye-icon delegate ───────────────────────────────────────
        delegate_cls = resolve_delegate_class()
        self._eye_delegate = delegate_cls(tree)
        self._eye_delegate.icon_clicked.connect(self._on_eye_clicked)
        tree.setItemDelegateForColumn(0, self._eye_delegate)

        # ── Top-level groups ────────────────────────────────────────
        self._group_groups = self._make_header_item("Physical Groups")
        self._group_parts = self._make_header_item("Parts")
        tree.addTopLevelItem(self._group_groups)
        tree.addTopLevelItem(self._group_parts)
        self._group_groups.setExpanded(True)
        self._group_parts.setExpanded(True)

        self._widget = widget

        # ── Subscribe + populate ────────────────────────────────────
        self.refresh()
        vis_mgr.on_changed.append(self._refresh_eye_states)

    @property
    def widget(self) -> Any:
        return self._widget

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Full rebuild — call after groups added / renamed / deleted."""
        self._refresh_groups()
        self._refresh_parts()

    def update_active(self) -> None:
        """Lightweight refresh after pick changes — re-bold the active
        group, update its child count. Doesn't rebuild the tree."""
        _, QtGui, _ = _qt()
        active = self._selection.active_group
        active_count = len(self._selection.picks)
        for i in range(self._group_groups.childCount()):
            item = self._group_groups.child(i)
            if item.data(0, _ROLE_KIND) != "group":
                continue
            name = item.data(0, _ROLE_PAYLOAD)
            font = item.font(0)
            if name == active:
                font.setBold(True)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.success)),
                )
                # Show staged picks count alongside the persisted count.
                base_count = item.data(0, int(0x0202)) or 0
                item.setText(1, f"{active_count}/{base_count}")
            else:
                font.setBold(False)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
                )
            item.setFont(0, font)

    def _refresh_groups(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._group_groups.takeChildren()
        groups = self._collect_groups()
        if not groups:
            empty = QtWidgets.QTreeWidgetItem(self._group_groups)
            empty.setText(0, "(no groups)")
            flags = empty.flags() & ~QtCore.Qt.ItemIsSelectable
            empty.setFlags(flags)
            empty.setForeground(
                0, QtGui.QBrush(QtGui.QColor(_theme().current.overlay)),
            )
            return

        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}
        active = self._selection.active_group
        for name, _dim, _pg_tag, members in groups:
            item = QtWidgets.QTreeWidgetItem(self._group_groups)
            item.setText(0, name)
            item.setText(1, str(len(members)))
            item.setData(0, _ROLE_KIND, "group")
            item.setData(0, _ROLE_PAYLOAD, name)
            item.setData(0, int(0x0202), len(members))    # base count
            item.setData(0, ROLE_VISIBLE, self._group_is_visible(members))
            if name == active:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.success)),
                )
            else:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
                )
            for dim, tag in members:
                child = QtWidgets.QTreeWidgetItem(item)
                child.setText(0, f"{dim_labels.get(dim, '?')} {tag}")
                child.setData(0, _ROLE_KIND, "entity")
                child.setData(0, _ROLE_PAYLOAD, (dim, tag))
                child.setData(
                    0, ROLE_VISIBLE,
                    not self._vis_mgr.is_hidden((dim, tag)),
                )

    def _refresh_parts(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._group_parts.takeChildren()
        if self._parts is None or not getattr(self._parts, "instances", None):
            self._group_parts.setHidden(True)
            return
        self._group_parts.setHidden(False)

        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}
        for name in sorted(self._parts.instances.keys()):
            inst = self._parts.instances[name]
            entities = getattr(inst, "entities", {}) or {}
            flat: list[tuple[int, int]] = []
            for dim, tags in entities.items():
                flat.extend((int(dim), int(t)) for t in tags)
            item = QtWidgets.QTreeWidgetItem(self._group_parts)
            item.setText(0, name)
            item.setText(1, str(len(flat)))
            item.setData(0, _ROLE_KIND, "part")
            item.setData(0, _ROLE_PAYLOAD, name)
            item.setData(0, ROLE_VISIBLE, self._group_is_visible(flat))
            for dim, tag in flat:
                child = QtWidgets.QTreeWidgetItem(item)
                child.setText(0, f"{dim_labels.get(dim, '?')} {tag}")
                child.setData(0, _ROLE_KIND, "entity")
                child.setData(0, _ROLE_PAYLOAD, (dim, tag))
                child.setData(
                    0, ROLE_VISIBLE,
                    not self._vis_mgr.is_hidden((dim, tag)),
                )

    # ------------------------------------------------------------------
    # Eye toggle
    # ------------------------------------------------------------------

    def _group_is_visible(self, dts: list[tuple[int, int]]) -> bool:
        if not dts:
            return True
        return any(not self._vis_mgr.is_hidden(dt) for dt in dts)

    def _on_eye_clicked(self, item: Any) -> None:
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        current_hidden = set(self._vis_mgr.hidden)
        if kind == "group":
            members = self._collect_item_dts(item)
            if not members:
                return
            visible_now = self._group_is_visible(members)
            if visible_now:
                current_hidden.update(members)
            else:
                current_hidden.difference_update(members)
            self._vis_mgr.set_hidden(current_hidden)
        elif kind == "part":
            members = self._collect_item_dts(item)
            if not members:
                return
            visible_now = self._group_is_visible(members)
            if visible_now:
                current_hidden.update(members)
            else:
                current_hidden.difference_update(members)
            self._vis_mgr.set_hidden(current_hidden)
        elif kind == "entity":
            dt = item.data(0, _ROLE_PAYLOAD)
            if dt in current_hidden:
                current_hidden.discard(dt)
            else:
                current_hidden.add(dt)
            self._vis_mgr.set_hidden(current_hidden)

    def _collect_item_dts(self, item: Any) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, _ROLE_KIND) == "entity":
                dt = child.data(0, _ROLE_PAYLOAD)
                if dt is not None:
                    out.append(tuple(dt))
        return out

    def _refresh_eye_states(self) -> None:
        """Repaint eyes after a programmatic visibility change."""
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            for j in range(top.childCount()):
                row = top.child(j)
                kind = row.data(0, _ROLE_KIND)
                if kind in ("group", "part"):
                    members = self._collect_item_dts(row)
                    row.setData(0, ROLE_VISIBLE, self._group_is_visible(members))
                    for k in range(row.childCount()):
                        leaf = row.child(k)
                        if leaf.data(0, _ROLE_KIND) == "entity":
                            dt = leaf.data(0, _ROLE_PAYLOAD)
                            leaf.setData(
                                0, ROLE_VISIBLE,
                                not self._vis_mgr.is_hidden(tuple(dt)),
                            )
        try:
            self._tree.viewport().update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Row selection
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: Any, _column: int) -> None:
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        payload = item.data(0, _ROLE_PAYLOAD)
        if kind == "group" and self._on_group_activated is not None:
            self._on_group_activated(payload)
        elif kind == "entity" and self._on_entity_toggled is not None:
            self._on_entity_toggled(payload)
        # Generic row-focus signal — fires for every kind so the
        # viewer can raise the matching property tab. Header rows
        # (``kind == "header"``) are non-selectable and never reach
        # here, so no guard needed.
        if kind in ("group", "entity", "part") and self._on_row_focused:
            self._on_row_focused(kind, payload)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos: Any) -> None:
        QtCore, _, QtWidgets = _qt()
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        if kind not in ("group", "entity", "part"):
            return

        # Resolve target DimTags for visibility actions.
        if kind == "entity":
            dts = [tuple(item.data(0, _ROLE_PAYLOAD))]
        else:
            dts = self._collect_item_dts(item)

        menu = QtWidgets.QMenu(self._widget)
        act_rename = act_delete = None
        if kind == "group":
            act_rename = menu.addAction("Rename")
            act_delete = menu.addAction("Delete")
            menu.addSeparator()

        n = len(dts)
        act_hide = menu.addAction(f"Hide ({n})") if dts else None
        act_isolate = menu.addAction(f"Isolate ({n})") if dts else None
        act_reveal = menu.addAction("Reveal all")

        chosen = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        name = item.data(0, _ROLE_PAYLOAD) if kind == "group" else None
        if chosen == act_rename and self._on_rename_group is not None and name:
            self._on_rename_group(name)
        elif chosen == act_delete and self._on_delete_group is not None and name:
            self._on_delete_group(name)
        elif chosen == act_hide and dts:
            current = set(self._vis_mgr.hidden)
            current.update(dts)
            self._vis_mgr.set_hidden(current)
        elif chosen == act_isolate and dts:
            self._vis_mgr.isolate_dts(dts)
        elif chosen == act_reveal:
            self._vis_mgr.reveal_all()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_header_item(self, label: str) -> Any:
        QtCore, _, QtWidgets = _qt()
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, _ROLE_KIND, "header")
        flags = item.flags() & ~QtCore.Qt.ItemIsSelectable
        item.setFlags(flags)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item

    @staticmethod
    def _collect_groups() -> list[tuple[str, int, int, list[tuple[int, int]]]]:
        """User-facing physical groups (skips ``_label:`` internals).

        Returns ``[(name, dim, pg_tag, members), ...]`` sorted by tag.
        Same shape as :class:`BrowserTab._collect_groups`.
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


__all__ = ["ModelOutlineTree"]
