"""
MeshBrowserTab — Visibility filtering by physical group and element type.

Two collapsible categories ("Physical Groups" and "Element Types"), each
listing items with a ParaView-style eye icon and an element count.
Clicking the eye toggles visibility for every BRep entity that belongs
to that group / type.

Group / type membership is computed entirely from ``MeshSceneData`` —
no Gmsh round-trip. Hiding is delegated to ``VisibilityManager.set_hidden``,
which recomputes the visible cells in one pass; the tab is responsible
for computing the full unioned hidden set on each toggle.

The pre-plan-03 implementation used a ``Qt.CheckState`` checkbox on
each leaf row. Replaced 2026-05-16 with the shared
:class:`EyeIconDelegate` so all three viewers (results / mesh /
model) speak the same visibility-UI vocabulary.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from qtpy import QtWidgets, QtCore

from ._eye_icon_delegate import ROLE_VISIBLE, resolve_delegate_class

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

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Item", "Elements"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        layout.addWidget(self._tree)

        # Eye-icon delegate on column 0 — shared with results.viewer +
        # model.viewer for visibility-UI consistency. Tap on the
        # ``ROLE_VISIBLE`` cell to toggle the row.
        delegate_cls = resolve_delegate_class()
        self._eye_delegate = delegate_cls(self._tree)
        self._eye_delegate.icon_clicked.connect(self._on_eye_clicked)
        self._tree.setItemDelegateForColumn(0, self._eye_delegate)

        self._populate()

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        self._populate_groups()
        self._populate_types()
        self._tree.expandAll()
        for col in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(col)

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
            item.setData(0, ROLE_VISIBLE, True)
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
            item.setData(0, ROLE_VISIBLE, True)
            item.setData(0, _ROLE_DTS, tuple(breps))

    # ------------------------------------------------------------------
    # Toggle handling
    # ------------------------------------------------------------------

    def _on_eye_clicked(self, item) -> None:
        """Flip the row's ``ROLE_VISIBLE`` and re-fire the hidden set."""
        if item is None or item.data(0, _ROLE_DTS) is None:
            return
        new_state = not bool(item.data(0, ROLE_VISIBLE))
        item.setData(0, ROLE_VISIBLE, new_state)
        # Force the delegate to repaint this row immediately.
        try:
            self._tree.viewport().update()
        except Exception:
            pass
        self._fire()

    def _fire(self) -> None:
        hidden: set["DimTag"] = set()
        for i in range(self._tree.topLevelItemCount()):
            root = self._tree.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                if not bool(child.data(0, ROLE_VISIBLE)):
                    dts = child.data(0, _ROLE_DTS)
                    if dts:
                        hidden.update(dts)
        self._on_hidden_changed(hidden)
