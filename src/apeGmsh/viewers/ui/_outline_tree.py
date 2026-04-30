"""OutlineTree — left-rail navigator for ResultsViewer (B++ design).

A single QTreeWidget with four top-level groups (per
``B++ Implementation Guide.html`` §4.1):

* **Stages** — list of analysis stages; click to activate.
* **Diagrams** — active diagrams with visibility checkbox; selection
  drives the contextual details panel.
* **Probes** — placeholder until probes are first-class objects.
* **Plots** — placeholder until time-history plots are first-class
  objects.

The tree subscribes to Director observers (stage / diagrams) and
refreshes its rows when the model changes. Selecting a Diagram row
fires ``on_diagram_selected(diagram)`` so the host can drive the
DiagramSettingsTab (B1) or the details panel (B2+).

Replaces the right-dock ``StagesTab`` and ``DiagramsTab`` from B0.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from .._failures import safe_slot
from ..diagrams._base import Diagram

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


# Roles for stashing data on tree items. Qt.UserRole == 0x100; we
# use distinct subroles for the two leaf kinds so iteration code can
# tell them apart without inspecting the parent.
_ROLE_STAGE_ID = 0x100
_ROLE_DIAGRAM_OBJ = 0x101
_ROLE_GROUP_KEY = 0x102


class OutlineTree:
    """Left-rail outline tree.

    Parameters
    ----------
    director
        The ResultsDirector — used for stage/diagram queries and
        observer registration.
    """

    def __init__(self, director: "ResultsDirector") -> None:
        QtWidgets, QtCore = _qt()
        self._director = director
        self._on_diagram_selected: Optional[
            Callable[[Optional[Diagram]], None]
        ] = None

        # ── Outer container ─────────────────────────────────────────
        widget = QtWidgets.QWidget()
        widget.setObjectName("OutlineTree")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header row: "OUTLINE" label + Insert button ─────────────
        header = QtWidgets.QFrame()
        header.setObjectName("OutlineHeader")
        header.setFixedHeight(28)
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 0, 6, 0)
        header_lay.setSpacing(6)

        label = QtWidgets.QLabel("OUTLINE")
        label.setObjectName("OutlineHeaderLabel")
        header_lay.addWidget(label)
        header_lay.addStretch(1)

        self._btn_insert = QtWidgets.QPushButton("+ Insert")
        self._btn_insert.setObjectName("OutlineInsertButton")
        self._btn_insert.setToolTip("Add a new diagram")
        self._btn_insert.setFlat(True)
        self._btn_insert.clicked.connect(self._on_insert)
        header_lay.addWidget(self._btn_insert)

        layout.addWidget(header)

        # ── Tree ────────────────────────────────────────────────────
        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("OutlineTreeWidget")
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(14)
        tree.setUniformRowHeights(True)
        tree.setExpandsOnDoubleClick(False)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tree.itemClicked.connect(self._on_item_clicked)
        tree.itemChanged.connect(self._on_item_changed)
        tree.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(tree, stretch=1)
        self._tree = tree

        # ── Group items (one each for Stages / Diagrams / Probes / Plots) ──
        self._group_stages = self._make_group("Stages", "stages")
        self._group_diagrams = self._make_group("Diagrams", "diagrams")
        self._group_probes = self._make_group("Probes", "probes")
        self._group_plots = self._make_group("Plots", "plots")
        for g in (
            self._group_stages,
            self._group_diagrams,
            self._group_probes,
            self._group_plots,
        ):
            tree.addTopLevelItem(g)
            g.setExpanded(True)

        # Theme-driven styling lives in viewers/ui/theme.py.
        self._widget = widget

        # ── Initial population + observer wiring ───────────────────
        self._refresh_stages()
        self._refresh_diagrams()
        self._refresh_placeholders()

        director.subscribe_stage(lambda _id: self._refresh_stages())
        director.subscribe_diagrams(self._refresh_diagrams)

        # Refresh on theme change so placeholder-row colours follow
        # the active palette.
        from .theme import THEME
        self._unsub_theme = THEME.subscribe(lambda _p: self._on_theme_changed())

    def _on_theme_changed(self) -> None:
        self._refresh_stages()
        self._refresh_diagrams()
        self._refresh_placeholders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def on_diagram_selected(
        self, callback: Callable[[Optional[Diagram]], None],
    ) -> None:
        """Register the callback fired when a Diagram row is selected.

        The callback receives the selected ``Diagram`` instance, or
        ``None`` if the selection moved off all Diagram rows.
        """
        self._on_diagram_selected = callback

    # ------------------------------------------------------------------
    # Group building
    # ------------------------------------------------------------------

    def _make_group(self, label: str, key: str):
        QtWidgets, QtCore = _qt()
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, _ROLE_GROUP_KEY, key)
        # Group rows are not selectable — clicking them only toggles
        # expand/collapse. Selection is reserved for leaf rows.
        flags = item.flags() & ~QtCore.Qt.ItemIsSelectable
        item.setFlags(flags)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item

    # ------------------------------------------------------------------
    # Refresh — Stages
    # ------------------------------------------------------------------

    def _refresh_stages(self) -> None:
        QtWidgets, QtCore = _qt()
        self._group_stages.takeChildren()
        active = self._director.stage_id
        stages = self._director.stages()
        for s in stages:
            sid = getattr(s, "id", str(s))
            label = self._format_stage_label(s)
            item = QtWidgets.QTreeWidgetItem([label])
            item.setData(0, _ROLE_STAGE_ID, sid)
            if sid == active:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            self._group_stages.addChild(item)

        if not stages:
            empty = QtWidgets.QTreeWidgetItem(["(no stages)"])
            flags = empty.flags() & ~QtCore.Qt.ItemIsSelectable
            empty.setFlags(flags)
            empty.setForeground(0, self._dim_brush())
            self._group_stages.addChild(empty)

    @staticmethod
    def _format_stage_label(s: Any) -> str:
        name = str(getattr(s, "name", getattr(s, "id", "?")))
        kind = getattr(s, "kind", "")
        n_steps = int(getattr(s, "n_steps", 0) or 0)
        if kind == "mode":
            f = getattr(s, "frequency_hz", None)
            if f is not None:
                return f"{name} · f = {f:.4g} Hz"
            return f"{name} · mode"
        if n_steps:
            return f"{name} · {n_steps} steps"
        return name

    # ------------------------------------------------------------------
    # Refresh — Diagrams
    # ------------------------------------------------------------------

    def _refresh_diagrams(self) -> None:
        QtWidgets, QtCore = _qt()
        # Preserve current selection across rebuilds so the details
        # panel doesn't blink to "nothing selected" on every visibility
        # toggle.
        previously_selected = self._currently_selected_diagram()

        self._tree.blockSignals(True)
        try:
            self._group_diagrams.takeChildren()
            diagrams = self._director.registry.diagrams()
            for d in diagrams:
                item = QtWidgets.QTreeWidgetItem([d.display_label()])
                item.setFlags(
                    item.flags() | QtCore.Qt.ItemIsUserCheckable
                )
                item.setCheckState(
                    0,
                    QtCore.Qt.Checked if d.is_visible
                    else QtCore.Qt.Unchecked,
                )
                # Stash the Diagram instance directly. A registry-index
                # would drift if move/remove happens between paint and
                # click; the instance reference is stable.
                item.setData(0, _ROLE_DIAGRAM_OBJ, d)
                self._group_diagrams.addChild(item)

            if not diagrams:
                empty = QtWidgets.QTreeWidgetItem(
                    ["(none — click + Insert)"],
                )
                flags = empty.flags() & ~QtCore.Qt.ItemIsSelectable
                empty.setFlags(flags)
                empty.setForeground(0, self._dim_brush())
                self._group_diagrams.addChild(empty)
        finally:
            self._tree.blockSignals(False)

        # Restore selection if the previously-selected diagram still
        # exists in the registry.
        if previously_selected is not None:
            self._select_diagram(previously_selected)

    # ------------------------------------------------------------------
    # Refresh — Probes / Plots placeholders
    # ------------------------------------------------------------------

    def _refresh_placeholders(self) -> None:
        QtWidgets, _ = _qt()
        for group, msg in (
            (self._group_probes, "(see Probes tab — coming inline)"),
            (self._group_plots, "(time-history plots — coming inline)"),
        ):
            group.takeChildren()
            empty = QtWidgets.QTreeWidgetItem([msg])
            flags = empty.flags() & ~self._unselectable_mask()
            empty.setFlags(flags)
            empty.setForeground(0, self._dim_brush())
            group.addChild(empty)

    @staticmethod
    def _unselectable_mask():
        from qtpy import QtCore
        return QtCore.Qt.ItemIsSelectable

    def _dim_brush(self):
        """Muted foreground brush for placeholder rows.

        Reads ``THEME.current.overlay`` so placeholder rows respect
        the active palette. The brush is computed at refresh time —
        callers re-call this on each :meth:`_refresh_*` so theme
        changes that fire a refresh pick up the new color.
        """
        from qtpy import QtGui
        from .theme import THEME
        return QtGui.QBrush(QtGui.QColor(THEME.current.overlay))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item, _column: int) -> None:
        # Group toggle on click: feels right in an outline tree.
        if item.data(0, _ROLE_GROUP_KEY) is not None:
            item.setExpanded(not item.isExpanded())
            return

        # Stage activation on single click (mirrors the old StagesTab).
        sid = item.data(0, _ROLE_STAGE_ID)
        if sid is not None:
            if sid != self._director.stage_id:
                self._director.set_stage(sid)
            return

    def _on_item_changed(self, item, column: int) -> None:
        QtWidgets, QtCore = _qt()
        diagram = item.data(0, _ROLE_DIAGRAM_OBJ)
        if diagram is None:
            return
        visible = item.checkState(0) == QtCore.Qt.Checked
        if diagram.is_visible != visible:
            self._director.registry.set_visible(diagram, visible)

    def _on_current_item_changed(self, current, _previous) -> None:
        if self._on_diagram_selected is None:
            return
        diagram = (
            current.data(0, _ROLE_DIAGRAM_OBJ) if current is not None else None
        )
        # Be permissive — only Diagram leaves carry _ROLE_DIAGRAM_OBJ;
        # all other rows resolve to None and clear the details panel.
        self._on_diagram_selected(diagram)

    @safe_slot
    def _on_insert(self) -> None:
        from ._add_diagram_dialog import AddDiagramDialog
        dlg = AddDiagramDialog(self._director, parent=self._widget)
        dlg.run()
        # Registry's on_changed observer will refresh the diagrams group.

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _currently_selected_diagram(self) -> Optional[Diagram]:
        item = self._tree.currentItem()
        if item is None:
            return None
        return item.data(0, _ROLE_DIAGRAM_OBJ)

    def _select_diagram(self, diagram: Diagram) -> bool:
        """Select the row showing ``diagram``. Returns True if found."""
        for i in range(self._group_diagrams.childCount()):
            child = self._group_diagrams.child(i)
            if child.data(0, _ROLE_DIAGRAM_OBJ) is diagram:
                self._tree.setCurrentItem(child)
                return True
        return False
