"""SessionOutline — the outline's two jobs, one tree (ADR 0098 §9).

* **Panes** — every mesh view and plot view under "Views". Selecting
  one is the same state as ``host.active_pane_id`` and the inspector's
  context (Amendment 1 A1.5: one state, three renderings). "New mesh
  view" / "New plot" live here and nowhere else — no View-menu
  duplicates, one spine. Both render DISABLED with the reason in the
  tooltip when ``T(N+1)`` breaches a pane floor (A1.4 Add gate, the
  shipped ``_SlotRow(can_add=False)`` idiom restated; 0087 INV-2: a
  control that cannot act does not pretend to).
* **Model composition** — physical groups, materials, element types.
  Checking names sets the ACTIVE mesh view's **scope**: v1 is ONE axis
  plus one or more checked names, never a boolean ``concrete AND
  hex``, so checking a name on a second axis MOVES the scope to that
  axis rather than intersecting. Unchecking the last name clears the
  scope back to the whole analysis mesh.

  The `materials` axis renders present but disabled: the h5 bridge
  stores material *definitions*, and each element's material tag is
  positional inside its element args, so no element→material index
  exists to check against (see :mod:`._scope`). Offering it enabled
  would promise a picture the realize layer refuses.

The select-all-nodes / select-all-Gauss half of §9's model tree is
selection work and lands with S4-2.

A projection throughout: it rebuilds from ``session.panes`` and reads
``view.scope`` on the change tick, and holds no state of its own.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from qtpy import QtCore, QtWidgets

from .._failures import safe_slot

_PANE_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole
_ACTION_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_AXIS_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2

#: The §9 composition axes, in outline order, with the token
#: :class:`~apeGmsh.results.session.Scope` stores.
_AXES: "tuple[tuple[str, str], ...]" = (
    ("physical_groups", "Physical groups"),
    ("materials", "Materials"),
    ("element_types", "Element types"),
)

_MATERIALS_DISABLED = (
    "The materials axis needs an element→material index, which the "
    "h5 bridge does not publish yet (material definitions are stored "
    "by name; each element's material tag is positional inside its "
    "element args). Scope by physical group or element type meanwhile."
)


class SessionOutline:
    """The outline tree — panes group + model-composition group."""

    def __init__(
        self,
        session: Any,
        *,
        on_pane_selected: Optional[Callable[[str], None]] = None,
        on_new_view: Optional[Callable[[], None]] = None,
        on_new_plot: Optional[Callable[[], None]] = None,
        on_close_pane: Optional[Callable[[str], None]] = None,
        can_add: Optional[Callable[[], "tuple[bool, str]"]] = None,
    ) -> None:
        self._session = session
        self._on_pane_selected = on_pane_selected
        self._on_new_view = on_new_view
        self._on_new_plot = on_new_plot
        self._on_close_pane = on_close_pane
        self._can_add = can_add
        self._listed: tuple[tuple[str, Optional[str]], ...] = ()
        self._syncing = False
        self._active_pane_id: Optional[str] = None

        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("SessionOutlineTree")
        tree.setHeaderHidden(True)
        tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection,
        )
        tree.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu,
        )
        tree.customContextMenuRequested.connect(
            safe_slot(self._on_context_menu),
        )
        self._tree = tree
        self._group = QtWidgets.QTreeWidgetItem(tree, ["Views"])
        self._group.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled,  # a header, not a choice
        )
        self._new_view_item = self._make_action_item("New mesh view", "view")
        self._new_plot_item = self._make_action_item("New plot", "plot")

        self._model_group = QtWidgets.QTreeWidgetItem(tree, ["Model"])
        self._model_group.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        self._axis_items: "dict[str, QtWidgets.QTreeWidgetItem]" = {}
        self._name_items: "dict[str, dict[str, QtWidgets.QTreeWidgetItem]]" = {}
        self._build_model_group()

        tree.currentItemChanged.connect(safe_slot(self._on_current_changed))
        tree.itemClicked.connect(safe_slot(self._on_item_clicked))
        tree.itemChanged.connect(safe_slot(self._on_item_changed))

        session.subscribe(self._on_session_tick)
        self.refresh()

    @property
    def widget(self) -> QtWidgets.QTreeWidget:
        return self._tree

    @property
    def new_view_item(self) -> QtWidgets.QTreeWidgetItem:
        """The "New mesh view" action row (the Add gate's subject)."""
        return self._new_view_item

    @property
    def new_plot_item(self) -> QtWidgets.QTreeWidgetItem:
        """The "New plot" action row (the Add gate's subject)."""
        return self._new_plot_item

    def dispose(self) -> None:
        self._session.unsubscribe(self._on_session_tick)

    # -- selection -----------------------------------------------------

    def selected_pane_id(self) -> Optional[str]:
        item = self._tree.currentItem()
        if item is None:
            return None
        pane_id = item.data(0, _PANE_ID_ROLE)
        return str(pane_id) if pane_id is not None else None

    def select_pane(self, pane_id: str) -> None:
        """Programmatic selection (the host's active pane, the boot
        default). Fires ``on_pane_selected`` like a user click would."""
        for i in range(self._group.childCount()):
            item = self._group.child(i)
            if item.data(0, _PANE_ID_ROLE) == pane_id:
                self._tree.setCurrentItem(item)
                return
        raise KeyError(f"No pane {pane_id!r} in the outline.")

    def set_active_pane(self, pane_id: Optional[str]) -> None:
        """Mirror the host's active pane onto the tree (A1.5).

        Idempotent and echo-free: selecting the row that is already
        current does not re-fire ``on_pane_selected``.
        """
        self._active_pane_id = pane_id
        if pane_id is None or self.selected_pane_id() == pane_id:
            self.refresh_scope()
            return
        self._syncing = True
        try:
            for i in range(self._group.childCount()):
                item = self._group.child(i)
                if item.data(0, _PANE_ID_ROLE) == pane_id:
                    self._tree.setCurrentItem(item)
                    break
        finally:
            self._syncing = False
        self.refresh_scope()

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the pane rows from ``session.panes``.

        Skips when membership and names are unchanged — the session
        ticks on EVERY mutation (slot fills included), and rebuilding
        rows on each would flicker the selection mid-gesture. The gate
        and the scope checkboxes are re-read either way: both are
        functions of state a slot fill can move.
        """
        wanted = tuple(
            (pane.id, pane.name) for pane in self._session.panes
        )
        if wanted != self._listed:
            selected = self.selected_pane_id()
            self._syncing = True
            try:
                self._group.takeChildren()
                for pane_id, name in wanted:
                    item = QtWidgets.QTreeWidgetItem(
                        self._group, [name or pane_id],
                    )
                    item.setData(0, _PANE_ID_ROLE, pane_id)
                    if pane_id == selected:
                        self._tree.setCurrentItem(item)
                self._group.addChild(self._new_view_item)
                self._group.addChild(self._new_plot_item)
                self._group.setExpanded(True)
                self._listed = wanted
            finally:
                self._syncing = False
        self.refresh_gate()
        self.refresh_scope()

    def refresh_gate(self) -> None:
        """Re-evaluate the A1.4 Add gate against the host's size."""
        allowed, reason = (
            self._can_add() if self._can_add is not None else (True, "")
        )
        for item in (self._new_view_item, self._new_plot_item):
            item.setDisabled(not allowed)
            item.setToolTip(0, reason)

    def refresh_scope(self) -> None:
        """Project the active mesh view's scope onto the checkboxes."""
        view = self._active_mesh_view()
        scope = None if view is None else view.scope
        self._syncing = True
        try:
            for axis, items in self._name_items.items():
                if scope is None or scope.axis != axis:
                    checked: set = set()
                elif scope.names is None:
                    checked = set(items)  # names=None means every name
                else:
                    checked = set(scope.names)
                for name, item in items.items():
                    item.setCheckState(
                        0,
                        QtCore.Qt.CheckState.Checked if name in checked
                        else QtCore.Qt.CheckState.Unchecked,
                    )
                    item.setDisabled(view is None or axis == "materials")
        finally:
            self._syncing = False

    # -- internals -----------------------------------------------------

    def _make_action_item(
        self, label: str, action: str,
    ) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(self._group, [f"+ {label}"])
        item.setData(0, _ACTION_ROLE, action)
        return item

    def _build_model_group(self) -> None:
        for axis, label in _AXES:
            axis_item = QtWidgets.QTreeWidgetItem(self._model_group, [label])
            axis_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            axis_item.setData(0, _AXIS_ROLE, axis)
            self._axis_items[axis] = axis_item
            names = self._axis_names(axis)
            self._name_items[axis] = {}
            if axis == "materials":
                axis_item.setToolTip(0, _MATERIALS_DISABLED)
                axis_item.setDisabled(True)
            for name in names:
                item = QtWidgets.QTreeWidgetItem(axis_item, [name])
                item.setData(0, _AXIS_ROLE, axis)
                item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable,
                )
                item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                if axis == "materials":
                    item.setToolTip(0, _MATERIALS_DISABLED)
                    item.setDisabled(True)
                self._name_items[axis][name] = item
            axis_item.setExpanded(axis == "physical_groups")
        self._model_group.setExpanded(True)

    def _axis_names(self, axis: str) -> "list[str]":
        """The names on one composition axis, from the live model.

        Empty (an axis with no rows) whenever the data cannot answer —
        a model with no physical groups, or the unpublished
        element→material index. An empty axis is not an error.
        """
        results = self._session.results
        if results is None or results.fem is None:
            return []
        try:
            from ..data import ViewerData

            data = ViewerData.from_fem(results.fem)
            if axis == "physical_groups":
                return list(data.elements.physical.names())
            if axis == "element_types":
                return [t.name for t in data.elements.types]
        except Exception:
            return []
        return []

    def _active_mesh_view(self) -> Any:
        from apeGmsh.results.session import MeshView

        pane_id = self._active_pane_id or self.selected_pane_id()
        if pane_id is None:
            return None
        try:
            pane = self._session.pane(pane_id)
        except KeyError:
            return None
        return pane if isinstance(pane, MeshView) else None

    def _on_session_tick(self) -> None:
        self.refresh()

    def _on_current_changed(self, current: Any, _previous: Any) -> None:
        if self._syncing or current is None:
            return
        pane_id = current.data(0, _PANE_ID_ROLE)
        if pane_id is None or self._on_pane_selected is None:
            return
        self._on_pane_selected(str(pane_id))

    def _on_item_clicked(self, item: Any, _column: int) -> None:
        action = item.data(0, _ACTION_ROLE)
        if action is None or item.isDisabled():
            return
        if action == "view" and self._on_new_view is not None:
            self._on_new_view()
        elif action == "plot" and self._on_new_plot is not None:
            self._on_new_plot()

    def _on_item_changed(self, item: Any, _column: int) -> None:
        """A scope checkbox moved → rewrite the active view's scope."""
        if self._syncing:
            return
        axis = item.data(0, _AXIS_ROLE)
        # Only a NAME row writes scope: the axis header carries the same
        # role (so a context menu can read it) but is not checkable.
        if axis is None or item not in self._name_items[axis].values():
            return
        view = self._active_mesh_view()
        if view is None:
            return
        from apeGmsh.results.session import Scope

        names = tuple(
            name for name, node in self._name_items[axis].items()
            if node.checkState(0) == QtCore.Qt.CheckState.Checked
        )
        view.scope = Scope(axis=axis, names=names) if names else None

    def _on_context_menu(self, point: Any) -> None:
        item = self._tree.itemAt(point)
        if item is None:
            return
        pane_id = item.data(0, _PANE_ID_ROLE)
        if pane_id is None or self._on_close_pane is None:
            return
        menu = QtWidgets.QMenu(self._tree)
        action = menu.addAction("Close pane")
        chosen = menu.exec_(self._tree.viewport().mapToGlobal(point))
        if chosen is action:
            self._on_close_pane(str(pane_id))


__all__ = ["SessionOutline"]
