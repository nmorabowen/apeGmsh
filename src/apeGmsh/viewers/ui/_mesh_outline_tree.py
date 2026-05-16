"""MeshOutlineTree — left-rail navigator for ``mesh.viewer``.

ParaView-style outline tree mirroring ``model.viewer``'s outline.
Sits in the left dock area as primary navigation; the right-side
``MeshBrowserTab`` continues to coexist during this transition.

Top-level groups
----------------
* **Physical Groups** — from ``MeshSceneData.group_to_breps``. Each
  row shows the group name + element count. Eye toggles visibility
  for every BRep in the group via :class:`VisibilityManager`.
* **Element Types** — from ``MeshSceneData.brep_dominant_type``,
  grouped by category (Hex, Tet, Quad, Tri, Line, ...). Eye toggles
  visibility for every BRep of that type.
* **Parts** — when ``g.parts`` is set, lists each instance and its
  DimTags. Eye toggles visibility for the union.

Loads / Masses / Constraints rows are deliberately **not** in this
first cut — they have their own composite panels with richer
interactions that don't map cleanly onto a single eye-icon column.
Follow-up if a clear UX emerges.

Same machinery as :class:`ModelOutlineTree` — share-by-copy until
both viewers have settled enough that a base class makes sense.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ._eye_icon_delegate import ROLE_VISIBLE, resolve_delegate_class

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..core.selection import SelectionState
    from ..core.visibility import VisibilityManager
    from ..scene.mesh_scene import MeshSceneData


def _qt():
    from qtpy import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def _theme():
    from .theme import THEME
    return THEME


# Role constants — distinct from MeshBrowserTab's so the two trees
# can coexist without subtle confusion if items ever move between
# them.
_ROLE_KIND = int(0x0210)       # "group" | "type" | "part" | "entity"
_ROLE_PAYLOAD = int(0x0211)    # name (group/type/part) | DimTag (entity)


class MeshOutlineTree:
    """Left-rail outline tree for ``mesh.viewer``.

    Parameters
    ----------
    scene
        :class:`MeshSceneData` — read for group / type / element-count
        membership. Built once at viewer open; cached.
    selection
        The viewer's :class:`SelectionState` — currently unused for
        the outline's display but reserved for future "active group"
        highlighting (mirror of :class:`ModelOutlineTree`).
    vis_mgr
        The viewer's :class:`VisibilityManager` — eye state + toggle.
    parts_registry
        Optional :class:`PartsRegistry` (``g.parts``). When ``None``,
        the Parts group is hidden.
    on_group_activated
        Callback fired when a Physical Group row is clicked. Same
        contract as :class:`ModelOutlineTree.on_group_activated`.
    """

    def __init__(
        self,
        scene: "MeshSceneData",
        selection: "SelectionState",
        vis_mgr: "VisibilityManager",
        parts_registry: Any = None,
        *,
        on_group_activated: Optional[Callable[[str], None]] = None,
        on_row_focused: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        QtCore, _, QtWidgets = _qt()
        self._scene = scene
        self._selection = selection
        self._vis_mgr = vis_mgr
        self._parts = parts_registry
        self._on_group_activated = on_group_activated
        # Generic row-focused signal — fires for every selectable row
        # with ``(kind, payload)``. Viewers map kinds to tab names and
        # call ``win.focus_tab(...)`` to reveal the property editor.
        self._on_row_focused = on_row_focused

        # ── Outer container + header ────────────────────────────────
        widget = QtWidgets.QWidget()
        widget.setObjectName("MeshOutlineTree")
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
        layout.addWidget(header)

        # ── Tree ────────────────────────────────────────────────────
        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("MeshOutlineTreeWidget")
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
        self._group_types = self._make_header_item("Element Types")
        self._group_parts = self._make_header_item("Parts")
        tree.addTopLevelItem(self._group_groups)
        tree.addTopLevelItem(self._group_types)
        tree.addTopLevelItem(self._group_parts)
        self._group_groups.setExpanded(True)
        self._group_types.setExpanded(True)
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
        """Full rebuild — call when scene data changes."""
        self._refresh_groups()
        self._refresh_types()
        self._refresh_parts()

    def _refresh_groups(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._group_groups.takeChildren()
        if not self._scene.group_to_breps:
            self._group_groups.setHidden(True)
            return
        self._group_groups.setHidden(False)
        for name in sorted(self._scene.group_to_breps.keys()):
            breps = list(self._scene.group_to_breps[name])
            n_elems = sum(
                len(self._scene.brep_to_elems.get(dt, [])) for dt in breps
            )
            item = QtWidgets.QTreeWidgetItem(self._group_groups)
            item.setText(0, name)
            item.setText(1, f"{n_elems:,}")
            item.setData(0, _ROLE_KIND, "group")
            item.setData(0, _ROLE_PAYLOAD, name)
            item.setData(0, ROLE_VISIBLE, self._group_is_visible(breps))

    def _refresh_types(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._group_types.takeChildren()
        type_to_breps: dict[str, list[tuple[int, int]]] = {}
        for dt, type_cat in self._scene.brep_dominant_type.items():
            type_to_breps.setdefault(type_cat, []).append(dt)
        if not type_to_breps:
            self._group_types.setHidden(True)
            return
        self._group_types.setHidden(False)
        for type_cat in sorted(type_to_breps.keys()):
            breps = type_to_breps[type_cat]
            n_elems = sum(
                len(self._scene.brep_to_elems.get(dt, [])) for dt in breps
            )
            item = QtWidgets.QTreeWidgetItem(self._group_types)
            item.setText(0, type_cat)
            item.setText(1, f"{n_elems:,}")
            item.setData(0, _ROLE_KIND, "type")
            item.setData(0, _ROLE_PAYLOAD, type_cat)
            item.setData(0, ROLE_VISIBLE, self._group_is_visible(breps))

    def _refresh_parts(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._group_parts.takeChildren()
        if self._parts is None or not getattr(self._parts, "instances", None):
            self._group_parts.setHidden(True)
            return
        self._group_parts.setHidden(False)
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

    # ------------------------------------------------------------------
    # Eye toggle
    # ------------------------------------------------------------------

    def _group_is_visible(self, dts: list[tuple[int, int]]) -> bool:
        if not dts:
            return True
        return any(not self._vis_mgr.is_hidden(dt) for dt in dts)

    def _resolve_dts(self, item: Any) -> list[tuple[int, int]]:
        """Resolve a header / leaf row to its underlying DimTag list."""
        kind = item.data(0, _ROLE_KIND)
        payload = item.data(0, _ROLE_PAYLOAD)
        if kind == "group":
            return list(self._scene.group_to_breps.get(payload, []))
        if kind == "type":
            return [
                dt
                for dt, cat in self._scene.brep_dominant_type.items()
                if cat == payload
            ]
        if kind == "part" and self._parts is not None:
            inst = self._parts.instances.get(payload)
            if inst is None:
                return []
            out: list[tuple[int, int]] = []
            for dim, tags in (inst.entities or {}).items():
                out.extend((int(dim), int(t)) for t in tags)
            return out
        return []

    def _on_eye_clicked(self, item: Any) -> None:
        if item is None:
            return
        dts = self._resolve_dts(item)
        if not dts:
            return
        current_hidden = set(self._vis_mgr.hidden)
        if self._group_is_visible(dts):
            current_hidden.update(dts)
        else:
            current_hidden.difference_update(dts)
        self._vis_mgr.set_hidden(current_hidden)

    def _refresh_eye_states(self) -> None:
        """Repaint every row's eye after a programmatic visibility change."""
        for top in (self._group_groups, self._group_types, self._group_parts):
            for i in range(top.childCount()):
                row = top.child(i)
                dts = self._resolve_dts(row)
                row.setData(0, ROLE_VISIBLE, self._group_is_visible(dts))
        try:
            self._tree.viewport().update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Row click — group activation
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: Any, _column: int) -> None:
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        payload = item.data(0, _ROLE_PAYLOAD)
        if kind == "group" and self._on_group_activated is not None:
            self._on_group_activated(payload)
        # Generic row-focus signal — fires for every kind so the
        # viewer can raise the matching property tab.
        if kind in ("group", "type", "part") and self._on_row_focused:
            self._on_row_focused(kind, payload)

    # ------------------------------------------------------------------
    # Right-click context menu — Hide / Isolate / Reveal-all
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos: Any) -> None:
        _, _, QtWidgets = _qt()
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        if kind not in ("group", "type", "part"):
            return
        dts = self._resolve_dts(item)
        n = len(dts)

        menu = QtWidgets.QMenu(self._widget)
        act_hide = menu.addAction(f"Hide ({n})") if dts else None
        act_isolate = menu.addAction(f"Isolate ({n})") if dts else None
        act_reveal = menu.addAction("Reveal all")

        chosen = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_hide and dts:
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
        flags = item.flags() & ~QtCore.Qt.ItemIsSelectable
        item.setFlags(flags)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item


__all__ = ["MeshOutlineTree"]
