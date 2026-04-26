"""
MeshBrowserTab — Visibility filtering by physical group and element type.

Two collapsible categories ("Physical Groups" and "Element Types"), each
listing items with a checkbox and an element count. Unchecking an item
hides every BRep entity that belongs to it.

Group / type membership is computed entirely from ``MeshSceneData`` —
no Gmsh round-trip. Hiding is delegated to ``VisibilityManager.set_hidden``,
which recomputes the visible cells in one pass; the tab is responsible
for computing the full unioned hidden set on each toggle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from qtpy import QtWidgets, QtCore

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..scene.mesh_scene import MeshSceneData


_ROLE_DTS = int(QtCore.Qt.UserRole) + 1  # tuple[DimTag, ...]


class MeshBrowserTab:
    """Tree of physical groups + element types with visibility checkboxes."""

    def __init__(
        self,
        scene: "MeshSceneData",
        *,
        on_hidden_changed: Callable[[set["DimTag"]], None],
    ) -> None:
        self._scene = scene
        self._on_hidden_changed = on_hidden_changed
        self._suspend_signals = False

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Item", "Elements"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        layout.addWidget(self._tree)

        self._populate()
        self._tree.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        self._suspend_signals = True
        try:
            self._populate_groups()
            self._populate_types()
            self._tree.expandAll()
            for col in range(self._tree.columnCount()):
                self._tree.resizeColumnToContents(col)
        finally:
            self._suspend_signals = False

    def _populate_groups(self) -> None:
        scene = self._scene
        if not scene.group_to_breps:
            return

        root = QtWidgets.QTreeWidgetItem(self._tree)
        root.setText(0, f"Physical Groups ({len(scene.group_to_breps)})")
        root.setFirstColumnSpanned(True)

        for name in sorted(scene.group_to_breps.keys()):
            breps = scene.group_to_breps[name]
            n_elems = sum(len(scene.brep_to_elems.get(dt, [])) for dt in breps)
            item = QtWidgets.QTreeWidgetItem(root)
            item.setText(0, name)
            item.setText(1, f"{n_elems:,}")
            item.setCheckState(0, QtCore.Qt.Checked)
            item.setData(0, _ROLE_DTS, tuple(breps))

    def _populate_types(self) -> None:
        scene = self._scene
        type_to_breps: dict[str, list["DimTag"]] = {}
        for dt, type_cat in scene.brep_dominant_type.items():
            type_to_breps.setdefault(type_cat, []).append(dt)
        if not type_to_breps:
            return

        root = QtWidgets.QTreeWidgetItem(self._tree)
        root.setText(0, f"Element Types ({len(type_to_breps)})")
        root.setFirstColumnSpanned(True)

        for type_cat in sorted(type_to_breps.keys()):
            breps = type_to_breps[type_cat]
            n_elems = sum(len(scene.brep_to_elems.get(dt, [])) for dt in breps)
            item = QtWidgets.QTreeWidgetItem(root)
            item.setText(0, type_cat)
            item.setText(1, f"{n_elems:,}")
            item.setCheckState(0, QtCore.Qt.Checked)
            item.setData(0, _ROLE_DTS, tuple(breps))

    # ------------------------------------------------------------------
    # Toggle handling
    # ------------------------------------------------------------------

    def _on_item_changed(self, item, _column: int) -> None:
        if self._suspend_signals:
            return
        # Only leaf items carry DimTag data
        if item.data(0, _ROLE_DTS) is None:
            return
        self._fire()

    def _fire(self) -> None:
        hidden: set["DimTag"] = set()
        for i in range(self._tree.topLevelItemCount()):
            root = self._tree.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                if child.checkState(0) == QtCore.Qt.Unchecked:
                    dts = child.data(0, _ROLE_DTS)
                    if dts:
                        hidden.update(dts)
        self._on_hidden_changed(hidden)
