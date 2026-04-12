"""
Model Tabs — UI components for the BRep model viewer.

Standalone QWidget classes, one per tab:

* **BrowserTab**: Physical group tree + group management buttons
* **FilterTab**: Dimension checkboxes for pick filtering

Usage::

    from apeGmsh.viewers.ui.model_tabs import BrowserTab, FilterTab
    browser = BrowserTab(selection_state, on_group_changed=...)
    filter_tab = FilterTab(dims=[0,1,2,3], on_filter_changed=...)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

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

        # Collect groups from Gmsh, keyed by name
        gmsh_groups = {}
        for name, pg_dim, pg_tag, members in self._collect_groups():
            gmsh_groups[name] = members

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
                    0, QtGui.QBrush(QtGui.QColor("#a6e3a1")),
                )
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            else:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor("#89b4fa")),
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
        layout.setContentsMargins(4, 4, 4, 4)

        filter_group = QtWidgets.QGroupBox("Pick Filter")
        filter_layout = QtWidgets.QVBoxLayout(filter_group)

        self._checkboxes: dict[int, Any] = {}  # QCheckBox (lazy Qt import)
        dim_labels = {0: "Points (dim=0)", 1: "Curves (dim=1)",
                      2: "Surfaces (dim=2)", 3: "Volumes (dim=3)"}
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(dim_labels.get(d, f"dim={d}"))
            cb.setChecked(True)
            cb.toggled.connect(self._on_toggled)
            self._checkboxes[d] = cb
            filter_layout.addWidget(cb)

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("All")
        btn_all.clicked.connect(self._select_all)
        btn_row.addWidget(btn_all)
        btn_none = QtWidgets.QPushButton("None")
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_none)
        filter_layout.addLayout(btn_row)

        layout.addWidget(filter_group)

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
        on_labels_changed: Callable[..., None] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dims : list[int]
            Available dimensions.
        on_labels_changed : callable
            ``fn(active_dims_dict, font_size, use_names, show_parts)`` called when
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

        self._dim_cbs: dict[int, Any] = {}  # QCheckBox (lazy Qt import)
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(f"{_DIM_NAMES.get(d, f'dim={d}')} tags")
            cb.setChecked(False)
            cb.toggled.connect(self._fire)
            group_layout.addWidget(cb)
            self._dim_cbs[d] = cb

        layout.addWidget(group)

        # ── Instance / entity labels group ─────────────────────────
        labels_group = QtWidgets.QGroupBox("Instance & entity labels")
        labels_layout = QtWidgets.QVBoxLayout(labels_group)

        self._show_parts = QtWidgets.QCheckBox("Show part instance labels")
        self._show_parts.setChecked(False)
        self._show_parts.toggled.connect(self._fire)
        labels_layout.addWidget(self._show_parts)

        self._show_entity_labels = QtWidgets.QCheckBox("Show entity labels (Tier 1)")
        self._show_entity_labels.setChecked(False)
        self._show_entity_labels.toggled.connect(self._fire)
        labels_layout.addWidget(self._show_entity_labels)

        layout.addWidget(labels_group)

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
        show_parts = self._show_parts.isChecked()
        show_entity_labels = self._show_entity_labels.isChecked()
        self._on_labels_changed(
            active, font_size, use_names, show_parts, show_entity_labels,
        )


# ======================================================================
# SelectionTreePanel — BRep hierarchy of current selection
# ======================================================================

_DIM_LABEL = {0: "Point", 1: "Curve", 2: "Surface", 3: "Volume"}
_DIM_ICON_COLOR = {
    0: "#f38ba8",   # red (Catppuccin)
    1: "#fab387",   # peach
    2: "#89b4fa",   # blue
    3: "#a6e3a1",   # green
}


class SelectionTreePanel:
    """Shows the BRep hierarchy of the current selection.

    Each selected entity appears as a root node.  Under it, its
    immediate boundary entities (from ``gmsh.model.getBoundary``)
    are listed as children.  The tree updates live on selection change.

    Supports multi-select (Ctrl+click) and right-click context menu
    to narrow the 3D viewer selection to only the highlighted entities.
    """

    # Qt UserRole offset for storing DimTag on items
    _DT_ROLE = 0x0100

    def __init__(
        self,
        *,
        on_select_only: Callable[[list[tuple[int, int]]], None] | None = None,
        on_add_to_selection: Callable[[list[tuple[int, int]]], None] | None = None,
        on_remove_from_selection: Callable[[list[tuple[int, int]]], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._QtGui = QtGui
        self._on_select_only = on_select_only
        self._on_add_to_selection = on_add_to_selection
        self._on_remove_from_selection = on_remove_from_selection

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QtWidgets.QLabel("Selection")
        header.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(header)
        self._header = header

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Entity", "Tag"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(16)
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree)

    def update(self, picks: list[tuple[int, int]]) -> None:
        """Rebuild the tree from the current selection picks."""
        QtGui = self._QtGui
        self._tree.clear()

        if not picks:
            self._header.setText("Selection (empty)")
            return

        self._header.setText(f"Selection ({len(picks)} entities)")

        # Group picks by dimension (descending: volumes first)
        by_dim: dict[int, list[int]] = {}
        for dim, tag in picks:
            by_dim.setdefault(dim, []).append(tag)

        for dim in sorted(by_dim.keys(), reverse=True):
            tags = sorted(by_dim[dim])
            dim_label = _DIM_LABEL.get(dim, f"dim={dim}")
            color = QtGui.QColor(_DIM_ICON_COLOR.get(dim, "#cdd6f4"))

            for tag in tags:
                # Root node: the selected entity
                root = self._make_item(
                    self._tree, dim, dim_label, tag, color, bold=True,
                )

                # Children: immediate boundary
                try:
                    boundary = gmsh.model.getBoundary(
                        [(dim, tag)],
                        combined=False,
                        oriented=False,
                        recursive=False,
                    )
                    # Unique and sorted
                    seen = set()
                    children = []
                    for bd, bt in boundary:
                        bt = abs(bt)
                        if (bd, bt) not in seen:
                            seen.add((bd, bt))
                            children.append((bd, bt))
                    children.sort()

                    for cd, ct in children:
                        child_label = _DIM_LABEL.get(cd, f"dim={cd}")
                        child_color = QtGui.QColor(
                            _DIM_ICON_COLOR.get(cd, "#cdd6f4")
                        )
                        child = self._make_item(
                            root, cd, child_label, ct, child_color,
                        )

                        # Grandchildren: one more level of boundary
                        if cd > 0:
                            try:
                                sub = gmsh.model.getBoundary(
                                    [(cd, ct)],
                                    combined=False,
                                    oriented=False,
                                    recursive=False,
                                )
                                sub_seen = set()
                                sub_children = []
                                for sd, st in sub:
                                    st = abs(st)
                                    if (sd, st) not in sub_seen:
                                        sub_seen.add((sd, st))
                                        sub_children.append((sd, st))
                                sub_children.sort()
                                for sd, st in sub_children:
                                    sl = _DIM_LABEL.get(sd, f"dim={sd}")
                                    sc = QtGui.QColor(
                                        _DIM_ICON_COLOR.get(sd, "#cdd6f4")
                                    )
                                    self._make_item(child, sd, sl, st, sc)
                            except Exception:
                                pass
                except Exception:
                    pass

                root.setExpanded(False)

        self._tree.resizeColumnToContents(0)

    def _on_context_menu(self, pos):
        QtWidgets, _, _ = _qt()
        selected = self._tree.selectedItems()
        if not selected:
            return

        # Collect unique DimTags from selected items
        dts = []
        seen = set()
        for item in selected:
            dt = item.data(0, self._DT_ROLE)
            if dt is not None and dt not in seen:
                seen.add(dt)
                dts.append(dt)
        if not dts:
            return

        n = len(dts)
        menu = QtWidgets.QMenu()
        act_only = menu.addAction(f"Select only ({n})")
        act_add = menu.addAction(f"Add to selection ({n})")
        act_remove = menu.addAction(f"Remove from selection ({n})")

        action = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if action == act_only and self._on_select_only:
            self._on_select_only(list(dts))
        elif action == act_add and self._on_add_to_selection:
            self._on_add_to_selection(list(dts))
        elif action == act_remove and self._on_remove_from_selection:
            self._on_remove_from_selection(list(dts))

    def _make_item(self, parent, dim, label, tag, color, bold=False):
        from qtpy.QtWidgets import QTreeWidgetItem
        QtGui = self._QtGui
        child = QTreeWidgetItem(parent)
        child.setText(0, f"{label} {tag}")
        child.setText(1, str(tag))
        child.setData(0, self._DT_ROLE, (dim, tag))
        child.setForeground(0, QtGui.QBrush(color))
        if bold:
            font = child.font(0)
            font.setBold(True)
            child.setFont(0, font)
        return child


# ======================================================================
# PartsTreePanel — Instance hierarchy from g.parts
# ======================================================================

_PART_COLOR = "#a6e3a1"      # green — Catppuccin
_UNTRACKED_COLOR = "#f9e2af"  # yellow — warning


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
        self._empty_label.setStyleSheet("color: #6c7086; padding: 12px;")
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
            root.setForeground(0, QtGui.QBrush(QtGui.QColor(_PART_COLOR)))
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
                color = QtGui.QColor(_DIM_ICON_COLOR.get(dim, "#cdd6f4"))

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
            ucolor = QtGui.QColor(_UNTRACKED_COLOR)
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
                color = QtGui.QColor(_DIM_ICON_COLOR.get(dim, "#cdd6f4"))
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
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(_PART_COLOR)))

        # Find owning part
        for label, inst in self._parts.instances.items():
            dim, tag = dt
            if tag in inst.entities.get(dim, []):
                item = self._part_items.get(label)
                if item:
                    item.setForeground(
                        0, QtGui.QBrush(QtGui.QColor("#f38ba8")),
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
