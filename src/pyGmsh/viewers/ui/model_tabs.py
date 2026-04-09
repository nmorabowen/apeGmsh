"""
Model Tabs — UI components for the BRep model viewer.

Standalone QWidget classes, one per tab:

* **BrowserTab**: Physical group tree + group management buttons
* **FilterTab**: Dimension checkboxes for pick filtering

Usage::

    from pyGmsh.viewers.ui.model_tabs import BrowserTab, FilterTab
    browser = BrowserTab(selection_state, on_group_changed=...)
    filter_tab = FilterTab(dims=[0,1,2,3], on_filter_changed=...)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import gmsh

if TYPE_CHECKING:
    from ..core.selection import SelectionState


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


# ======================================================================
# BrowserTab — Physical group tree
# ======================================================================

class BrowserTab:
    """Tree showing physical groups + unassigned entities.

    The tree supports:
    - Click group header → activate that group for editing
    - Click entity leaf → toggle its pick in the working set
    - Right-click group → rename / delete context menu
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
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
                font-size: 12px;
            }
            QTreeWidget::item:selected {
                background-color: #45475a;
            }
            QHeaderView::section {
                background-color: #181825;
                color: #a6adc8;
                border: 1px solid #313244;
                padding: 3px;
                font-weight: bold;
            }
        """)
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
        self._group_items: dict[str, object] = {}

        # Collect all groups (Gmsh + staged)
        groups = self._collect_groups()
        all_names: list[str] = list(groups.keys())

        # Add staged-only groups
        for name in self._selection.staged_groups:
            if name not in groups:
                groups[name] = list(self._selection.staged_groups[name])
                all_names.append(name)

        active = self._selection.active_group
        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}

        for name in all_names:
            members = groups[name]
            item = QtWidgets.QTreeWidgetItem(self._tree)
            item.setText(0, name)
            item.setText(1, str(len(members)))
            item.setData(0, 0x0100, ("group", name))

            # Highlight active group
            if name == active:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor("#a6e3a1")),  # green
                )
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            else:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor("#89b4fa")),  # blue
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
                    0, QtGui.QBrush(QtGui.QColor("#a6e3a1")),
                )
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            else:
                font = item.font(0)
                font.setBold(False)
                item.setFont(0, font)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor("#89b4fa")),
                )

    def _collect_groups(self) -> dict[str, list[tuple]]:
        groups: dict[str, list[tuple]] = {}
        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                name = f"Group_{pg_dim}_{pg_tag}"
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            members = groups.setdefault(name, [])
            for t in ents:
                members.append((pg_dim, int(t)))
        return groups

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


# ======================================================================
# FilterTab — Dimension pick filter
# ======================================================================

class FilterTab:
    """Dimension checkboxes for filtering which entities respond to picks."""

    def __init__(
        self,
        dims: list[int],
        *,
        on_filter_changed: Callable[[set[int]], None] | None = None,
    ) -> None:
        QtWidgets, _, _ = _qt()
        self._on_filter_changed = on_filter_changed
        self._active_dims = set(dims)

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(QtWidgets.QLabel("Pick filter — active dimensions:"))

        self._checkboxes: dict[int, object] = {}
        dim_labels = {0: "Points (dim=0)", 1: "Curves (dim=1)",
                      2: "Surfaces (dim=2)", 3: "Volumes (dim=3)"}
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(dim_labels.get(d, f"dim={d}"))
            cb.setChecked(True)
            cb.toggled.connect(self._on_toggled)
            self._checkboxes[d] = cb
            layout.addWidget(cb)

        # Quick buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("All")
        btn_all.clicked.connect(self._select_all)
        btn_row.addWidget(btn_all)
        btn_none = QtWidgets.QPushButton("None")
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        layout.addStretch()

    def _on_toggled(self, _checked: bool) -> None:
        active = set()
        for d, cb in self._checkboxes.items():
            if cb.isChecked():
                active.add(d)
        self._active_dims = active
        if self._on_filter_changed:
            self._on_filter_changed(active)

    def _select_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _select_none(self):
        for cb in self._checkboxes.values():
            cb.setChecked(False)


# ======================================================================
# ViewTab — Entity label overlays
# ======================================================================

_DIM_NAMES = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}
_DIM_ABBR = {0: "P", 1: "C", 2: "S", 3: "V"}


class ViewTab:
    """Toggle entity label overlays per dimension in the 3D viewport."""

    def __init__(
        self,
        dims: list[int],
        *,
        on_labels_changed: Callable[
            [dict[int, bool], int, bool], None
        ] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dims : list[int]
            Available dimensions.
        on_labels_changed : callable
            ``fn(active_dims_dict, font_size, use_names)`` called when
            any toggle or setting changes.
        """
        QtWidgets, _, _ = _qt()
        self._on_labels_changed = on_labels_changed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Dim checkboxes ──────────────────────────────────────────
        group = QtWidgets.QGroupBox("Show entity labels on screen")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._dim_cbs: dict[int, object] = {}
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(f"{_DIM_NAMES.get(d, f'dim={d}')} tags")
            cb.setChecked(False)
            cb.toggled.connect(self._fire)
            group_layout.addWidget(cb)
            self._dim_cbs[d] = cb

        layout.addWidget(group)

        # ── Label style ─────────────────────────────────────────────
        style_group = QtWidgets.QGroupBox("Label style")
        style_layout = QtWidgets.QFormLayout(style_group)
        style_layout.setSpacing(4)

        self._font_size = QtWidgets.QSpinBox()
        self._font_size.setRange(6, 24)
        self._font_size.setValue(10)
        self._font_size.valueChanged.connect(self._fire)
        style_layout.addRow("Font size", self._font_size)

        self._use_names = QtWidgets.QCheckBox("Show names instead of tags")
        self._use_names.setChecked(False)
        self._use_names.toggled.connect(self._fire)
        style_layout.addRow(self._use_names)

        layout.addWidget(style_group)
        layout.addStretch(1)

    def _fire(self, *_args) -> None:
        if self._on_labels_changed is None:
            return
        active = {d: cb.isChecked() for d, cb in self._dim_cbs.items()}
        font_size = self._font_size.value()
        use_names = self._use_names.isChecked()
        self._on_labels_changed(active, font_size, use_names)
