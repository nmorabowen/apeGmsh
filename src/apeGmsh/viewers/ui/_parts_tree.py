"""
PartsTreePanel — browsable tree of the part/instance hierarchy.

Each part registered in ``g.parts`` becomes a root node; children are
dimension groups (Points / Curves / Surfaces / Volumes); leaves are
individual entities. An "Untracked" root collects entities present in
the scene that don't belong to any part so stray geometry is still
visible. Supports multi-select + right-click for select/isolate/hide/
rename/delete/fuse operations.
"""
from __future__ import annotations

from typing import Any, Callable

from ._selection_tree import _DIM_ICON_COLOR, _DIM_LABEL


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


def _theme():
    from .theme import THEME
    return THEME


def _part_color() -> str:
    return _theme().current.success


def _untracked_color() -> str:
    return _theme().current.warning


class PartsTreePanel:
    """Shows the Instance hierarchy from ``g.parts`` as a browsable tree.

    Each part is a root node; children are dim-groups; leaves are
    individual entities.  An "Untracked" group shows entities in the
    scene that don't belong to any part.
    """

    _DT_ROLE = 0x0100

    def __init__(
        self,
        parts_registry,
        entity_registry,
        *,
        on_select_only: Callable[[list[tuple[int, int]]], None] | None = None,
        on_add_to_selection: Callable[[list[tuple[int, int]]], None] | None = None,
        on_remove_from_selection: Callable[[list[tuple[int, int]]], None] | None = None,
        on_isolate: Callable[[list[tuple[int, int]]], None] | None = None,
        on_hide: Callable[[list[tuple[int, int]]], None] | None = None,
        on_new_part: Callable[[str, list[tuple[int, int]]], None] | None = None,
        on_rename_part: Callable[[str, str], None] | None = None,
        on_delete_part: Callable[[str], None] | None = None,
        on_fuse_parts: Callable[[list[str], str], None] | None = None,
        get_current_picks: Callable[[], list[tuple[int, int]]] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._QtGui = QtGui
        self._parts = parts_registry
        self._entity_reg = entity_registry
        self._on_select_only = on_select_only
        self._on_add_to_selection = on_add_to_selection
        self._on_remove_from_selection = on_remove_from_selection
        self._on_isolate = on_isolate
        self._on_hide = on_hide
        self._on_new_part = on_new_part
        self._on_rename_part = on_rename_part
        self._on_delete_part = on_delete_part
        self._on_fuse_parts = on_fuse_parts
        self._get_current_picks = get_current_picks

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._header = QtWidgets.QLabel("Parts")
        self._header.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self._header)

        # ── Toolbar ─────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_new = QtWidgets.QPushButton("New Part")
        btn_new.clicked.connect(self._action_new)
        btn_row.addWidget(btn_new)
        btn_rename = QtWidgets.QPushButton("Rename")
        btn_rename.clicked.connect(self._action_rename)
        btn_row.addWidget(btn_rename)
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.clicked.connect(self._action_delete)
        btn_row.addWidget(btn_delete)
        btn_fuse = QtWidgets.QPushButton("Fuse")
        btn_fuse.clicked.connect(self._action_fuse)
        btn_row.addWidget(btn_fuse)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._empty_label = QtWidgets.QLabel(
            "No parts registered.\n\n"
            "Select entities and click 'New Part',\n"
            "or use g.parts.from_model() in code."
        )
        self._empty_label.setStyleSheet(
            f"color: {_theme().current.overlay}; padding: 12px;"
        )
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Count"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(16)
        self._tree.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection,
        )
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemClicked.connect(self._on_tree_click)
        layout.addWidget(self._tree)

        # Map part_label -> root QTreeWidgetItem (for highlight)
        self._part_items: dict[str, Any] = {}  # QTreeWidgetItem (lazy Qt import)

        self.refresh()

    # ── Build tree ──────────────────────────────────────────────

    def refresh(self) -> None:
        """Full rebuild of the parts tree."""
        QtGui = self._QtGui
        from qtpy.QtWidgets import QTreeWidgetItem

        self._tree.clear()
        self._part_items.clear()

        instances = self._parts.instances
        all_scene = set(self._entity_reg.all_entities())

        # Collect all tracked DimTags
        tracked: set[tuple[int, int]] = set()

        for label, inst in instances.items():
            part_dts = []
            for dim, tags in inst.entities.items():
                for t in tags:
                    part_dts.append((dim, t))
                    tracked.add((dim, t))

            # Root: part label
            root = QTreeWidgetItem(self._tree)
            root.setText(0, label)
            root.setText(1, str(len(part_dts)))
            root.setData(0, self._DT_ROLE, ("part", label))
            root.setForeground(0, QtGui.QBrush(QtGui.QColor(_part_color())))
            font = root.font(0)
            font.setBold(True)
            root.setFont(0, font)
            self._part_items[label] = root

            # Dim groups
            for dim in sorted(inst.entities.keys()):
                tags = inst.entities[dim]
                if not tags:
                    continue
                dim_label = _DIM_LABEL.get(dim, f"dim={dim}")
                color = QtGui.QColor(
                    _DIM_ICON_COLOR.get(dim, _theme().current.text)
                )

                group = QTreeWidgetItem(root)
                group.setText(0, f"{dim_label}s")
                group.setText(1, str(len(tags)))
                group.setData(0, self._DT_ROLE, ("dim_group", label, dim))
                group.setForeground(0, QtGui.QBrush(color))

                for t in sorted(tags):
                    leaf = QTreeWidgetItem(group)
                    leaf.setText(0, f"{dim_label} {t}")
                    leaf.setText(1, str(t))
                    leaf.setData(0, self._DT_ROLE, ("entity", (dim, t)))
                    leaf.setForeground(0, QtGui.QBrush(color))

        # Untracked entities
        untracked = all_scene - tracked
        if untracked:
            ucolor = QtGui.QColor(_untracked_color())
            uroot = QTreeWidgetItem(self._tree)
            uroot.setText(0, "Untracked")
            uroot.setText(1, str(len(untracked)))
            uroot.setData(0, self._DT_ROLE, ("untracked", None))
            uroot.setForeground(0, QtGui.QBrush(ucolor))
            font = uroot.font(0)
            font.setBold(True)
            uroot.setFont(0, font)

            by_dim: dict[int, list[int]] = {}
            for dim, tag in untracked:
                by_dim.setdefault(dim, []).append(tag)
            for dim in sorted(by_dim.keys()):
                tags = sorted(by_dim[dim])
                dim_label = _DIM_LABEL.get(dim, f"dim={dim}")
                color = QtGui.QColor(
                    _DIM_ICON_COLOR.get(dim, _theme().current.text)
                )
                group = QTreeWidgetItem(uroot)
                group.setText(0, f"{dim_label}s")
                group.setText(1, str(len(tags)))
                group.setData(0, self._DT_ROLE, ("dim_group", "__untracked__", dim))
                group.setForeground(0, QtGui.QBrush(color))
                for t in tags:
                    leaf = QTreeWidgetItem(group)
                    leaf.setText(0, f"{dim_label} {t}")
                    leaf.setText(1, str(t))
                    leaf.setData(0, self._DT_ROLE, ("entity", (dim, t)))
                    leaf.setForeground(0, QtGui.QBrush(color))

        # Show/hide empty state
        has_content = bool(instances) or bool(untracked)
        self._tree.setVisible(has_content)
        self._empty_label.setVisible(not has_content)
        n = len(instances)
        self._header.setText(
            f"Parts ({n} instance{'s' if n != 1 else ''})"
            if n else "Parts"
        )
        self._tree.resizeColumnToContents(0)

    # ── Highlight part for a picked entity ──────────────────────

    def highlight_part_for_entity(self, dt: tuple[int, int]) -> None:
        """Scroll to and bold the part containing *dt*."""
        QtGui = self._QtGui
        # Reset all part items to normal weight
        for item in self._part_items.values():
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(_part_color())))

        # Find owning part
        for label, inst in self._parts.instances.items():
            dim, tag = dt
            if tag in inst.entities.get(dim, []):
                item = self._part_items.get(label)
                if item:
                    item.setForeground(
                        0, QtGui.QBrush(QtGui.QColor(_theme().current.error)),
                    )
                    self._tree.scrollToItem(item)
                return

    # ── Click handler ───────────────────────────────────────────

    def _on_tree_click(self, item, _column):
        data = item.data(0, self._DT_ROLE)
        if not data or not self._on_select_only:
            return
        dts = self._collect_dimtags_for_item(data)
        if dts:
            self._on_select_only(dts)

    # ── Context menu ────────────────────────────────────────────

    def _on_context_menu(self, pos):
        QtWidgets, _, _ = _qt()
        dts = self._collect_selected_dimtags()
        if not dts:
            return

        n = len(dts)
        menu = QtWidgets.QMenu()
        act_only = menu.addAction(f"Select only ({n})")
        act_add = menu.addAction(f"Add to selection ({n})")
        act_remove = menu.addAction(f"Remove from selection ({n})")

        # Check if any selected item is a part root
        has_part = False
        for item in self._tree.selectedItems():
            d = item.data(0, self._DT_ROLE)
            if d and isinstance(d, tuple) and d[0] == "part":
                has_part = True
                break

        act_isolate = act_hide = act_rename = act_delete = act_fuse = None
        # Count distinct part roots in selection
        part_labels: list[str] = []
        for item in self._tree.selectedItems():
            d = item.data(0, self._DT_ROLE)
            if d and isinstance(d, tuple) and d[0] == "part":
                if d[1] not in part_labels:
                    part_labels.append(d[1])

        if has_part:
            menu.addSeparator()
            act_isolate = menu.addAction("Isolate Part")
            act_hide = menu.addAction("Hide Part")
            part_label = part_labels[0] if part_labels else None
            if part_label:
                menu.addSeparator()
                act_rename = menu.addAction("Rename Part")
                act_delete = menu.addAction("Delete Part")
            if len(part_labels) >= 2:
                menu.addSeparator()
                act_fuse = menu.addAction(f"Fuse Selected Parts ({len(part_labels)})")

        action = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if action == act_only and self._on_select_only:
            self._on_select_only(dts)
        elif action == act_add and self._on_add_to_selection:
            self._on_add_to_selection(dts)
        elif action == act_remove and self._on_remove_from_selection:
            self._on_remove_from_selection(dts)
        elif action == act_isolate and self._on_isolate:
            self._on_isolate(dts)
        elif action == act_hide and self._on_hide:
            self._on_hide(dts)
        elif action == act_rename and part_label:
            self._action_rename(part_label)
        elif action == act_delete and part_label:
            self._action_delete(part_label)
        elif action == act_fuse and len(part_labels) >= 2:
            self._action_fuse()

    # ── Button actions ──────────────────────────────────────────

    def _action_new(self):
        QtWidgets, _, _ = _qt()
        picks = self._get_current_picks() if self._get_current_picks else []
        if not picks:
            QtWidgets.QMessageBox.information(
                self.widget, "New Part",
                "Select entities in the viewport first,\n"
                "then click 'New Part'.",
            )
            return
        label, ok = QtWidgets.QInputDialog.getText(
            self.widget, "New Part", "Part name:",
        )
        if ok and label.strip():
            if self._on_new_part:
                self._on_new_part(label.strip(), picks)

    def _action_rename(self, label: str | None = None):
        QtWidgets, _, _ = _qt()
        # If called from button (no arg), use first selected part
        if label is None:
            for item in self._tree.selectedItems():
                d = item.data(0, self._DT_ROLE)
                if d and isinstance(d, tuple) and d[0] == "part":
                    label = d[1]
                    break
        if label is None:
            return
        new_label, ok = QtWidgets.QInputDialog.getText(
            self.widget, "Rename Part", "New name:", text=label,
        )
        if ok and new_label.strip() and new_label.strip() != label:
            if self._on_rename_part:
                self._on_rename_part(label, new_label.strip())

    def _action_delete(self, label: str | None = None):
        QtWidgets, _, _ = _qt()
        if label is None:
            for item in self._tree.selectedItems():
                d = item.data(0, self._DT_ROLE)
                if d and isinstance(d, tuple) and d[0] == "part":
                    label = d[1]
                    break
        if label is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self.widget, "Delete Part",
            f"Delete part '{label}'?\n\n"
            f"Entities will remain in the session as untracked.",
        )
        if reply == QtWidgets.QMessageBox.Yes:
            if self._on_delete_part:
                self._on_delete_part(label)

    def _action_fuse(self):
        """Fuse 2+ selected parts into a single new part."""
        QtWidgets, _, _ = _qt()
        # Collect labels from selected part-root items
        labels: list[str] = []
        for item in self._tree.selectedItems():
            d = item.data(0, self._DT_ROLE)
            if d and isinstance(d, tuple) and d[0] == "part":
                if d[1] not in labels:
                    labels.append(d[1])
        if len(labels) < 2:
            QtWidgets.QMessageBox.information(
                self.widget, "Fuse Parts",
                "Select at least 2 parts to fuse.\n"
                "Use Ctrl+click on part labels.",
            )
            return
        new_label, ok = QtWidgets.QInputDialog.getText(
            self.widget, "Fuse Parts",
            f"Fuse {len(labels)} parts into:",
            text=labels[0],
        )
        if ok and new_label.strip():
            if self._on_fuse_parts:
                self._on_fuse_parts(labels, new_label.strip())

    # ── Helpers ─────────────────────────────────────────────────

    def _collect_dimtags_for_item(self, data) -> list[tuple[int, int]]:
        """Resolve a tree item's stored data to a list of DimTags."""
        if not data or not isinstance(data, tuple):
            return []
        kind = data[0]
        if kind == "entity":
            return [data[1]]
        if kind == "part":
            label = data[1]
            inst = self._parts.instances.get(label)
            if inst is None:
                return []
            return [(d, t) for d, ts in inst.entities.items() for t in ts]
        if kind == "dim_group":
            label, dim = data[1], data[2]
            if label == "__untracked__":
                tracked = set()
                for inst in self._parts.instances.values():
                    for d, ts in inst.entities.items():
                        for t in ts:
                            tracked.add((d, t))
                return [
                    dt for dt in self._entity_reg.all_entities(dim=dim)
                    if dt not in tracked
                ]
            inst = self._parts.instances.get(label)
            if inst is None:
                return []
            return [(dim, t) for t in inst.entities.get(dim, [])]
        if kind == "untracked":
            tracked = set()
            for inst in self._parts.instances.values():
                for d, ts in inst.entities.items():
                    for t in ts:
                        tracked.add((d, t))
            return [
                dt for dt in self._entity_reg.all_entities()
                if dt not in tracked
            ]
        return []

    def _collect_selected_dimtags(self) -> list[tuple[int, int]]:
        """Collect unique DimTags from all selected tree items."""
        seen: set[tuple[int, int]] = set()
        result: list[tuple[int, int]] = []
        for item in self._tree.selectedItems():
            data = item.data(0, self._DT_ROLE)
            for dt in self._collect_dimtags_for_item(data):
                if dt not in seen:
                    seen.add(dt)
                    result.append(dt)
        return result


__all__ = ["PartsTreePanel"]
