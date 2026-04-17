"""
SelectionTreePanel — BRep hierarchy tree of the current viewer selection.

Each selected entity appears as a root node. Under it, its immediate
boundary entities (from ``gmsh.model.getBoundary``) are listed as
children, one more level deep for grandchildren. The tree updates live
on selection change and supports multi-select + right-click context
menu to narrow / extend / subtract the 3D viewer selection.
"""
from __future__ import annotations

from typing import Callable

import gmsh


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


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


__all__ = ["SelectionTreePanel"]
