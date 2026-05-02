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
from ._layout_metrics import LAYOUT

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


# Roles for stashing data on tree items. Qt.UserRole == 0x100; we
# use distinct subroles for the leaf kinds so iteration code can
# tell them apart without inspecting the parent.
_ROLE_STAGE_ID = 0x100
_ROLE_DIAGRAM_OBJ = 0x101         # legacy — single-layer rows; unused after pivot
_ROLE_GROUP_KEY = 0x102
_ROLE_PLOT_KEY = 0x103
_ROLE_COMPOSITION_KEY = 0x104     # one Diagram = a stack of layers


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
        self._on_composition_selected: Optional[
            Callable[[Optional[str]], None]
        ] = None
        self._plot_pane: Any = None
        self._unsub_plot_tabs: Optional[Callable[[], None]] = None

        # ── Outer container ─────────────────────────────────────────
        widget = QtWidgets.QWidget()
        widget.setObjectName("OutlineTree")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header row: "OUTLINE" label ─────────────────────────────
        header = QtWidgets.QFrame()
        header.setObjectName("OutlineHeader")
        header.setFixedHeight(LAYOUT.panel_header_height)
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 0, 6, 0)
        header_lay.setSpacing(6)

        label = QtWidgets.QLabel("OUTLINE")
        label.setObjectName("OutlineHeaderLabel")
        header_lay.addWidget(label)
        header_lay.addStretch(1)

        # + Add diagram — also reachable via right-click on the
        # Diagrams group header in the tree.
        self._btn_add_diagram = QtWidgets.QPushButton("+")
        self._btn_add_diagram.setObjectName("OutlineAddButton")
        self._btn_add_diagram.setFlat(True)
        self._btn_add_diagram.setFixedWidth(24)
        self._btn_add_diagram.setToolTip("Add a new diagram")
        self._btn_add_diagram.clicked.connect(self._on_add_diagram)
        header_lay.addWidget(self._btn_add_diagram)

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
        # Right-click context menu — composition rows get rename /
        # duplicate / delete; the Diagrams group header gets Add
        # diagram. Other rows: no menu.
        tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(tree, stretch=1)
        self._tree = tree

        # ── Group items ────────────────────────────────────────────
        # Layers replaces the prior Catalog + Diagrams split: the
        # creation flow lives in the DetailsPanel (+ Add layer
        # button → settings-panel creation mode). The outline only
        # shows what's *attached* — Stages, Layers, Probes, Plots.
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
        # Composition changes (add / rename / set_active) fire here too.
        self._unsub_compositions = director.compositions.subscribe(
            self._refresh_diagrams,
        )

        # F2 on a composition row → enter edit mode for inline rename.
        f2_sc = QtWidgets.QShortcut(QtCore.Qt.Key.Key_F2, tree)
        f2_sc.activated.connect(self._begin_inline_rename)

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

        Legacy single-layer event — kept around so existing wiring
        compiles. After the v2 pivot, the outline doesn't list
        individual diagrams anymore (one ``Diagram 1`` row groups all
        layers); use :meth:`on_composition_selected` instead.
        """
        self._on_diagram_selected = callback

    def on_composition_selected(
        self, callback: Callable[[Optional[str]], None],
    ) -> None:
        """Register the callback fired when a composition row is selected.

        v2: one outline row per *diagram* (composition) — each diagram
        is the stack of layers shown in the details dock. Today the
        only composition is keyed ``"default"``; multi-diagram support
        is a planned extension.
        """
        self._on_composition_selected = callback


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
        """Render one row per composition (Geometry + user-added).

        The active composition is shown bold. Each row also shows its
        layer count in parentheses when non-zero. Geometry is locked
        (no Delete in its context menu).
        """
        QtWidgets, QtCore = _qt()
        from ..diagrams._compositions import GEOMETRY_ID

        self._tree.blockSignals(True)
        try:
            self._group_diagrams.takeChildren()
            mgr = self._director.compositions
            active_id = mgr.active_id
            for comp in mgr.compositions:
                n = len(comp.layers)
                label = comp.name if n == 0 else f"{comp.name} ({n})"
                item = QtWidgets.QTreeWidgetItem([label])
                item.setData(0, _ROLE_COMPOSITION_KEY, comp.id)
                # Make rows editable so F2 / double-click can rename
                # in place. Geometry is editable too (renamable but
                # not deletable).
                item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                if comp.id == active_id:
                    font = item.font(0)
                    font.setBold(True)
                    item.setFont(0, font)
                if comp.locked:
                    item.setToolTip(
                        0,
                        f"{comp.name} — base view (locked, can be "
                        f"renamed). F2 / right-click to rename.",
                    )
                else:
                    item.setToolTip(
                        0,
                        f"{comp.name} — F2 / right-click to rename, "
                        f"duplicate, or delete.",
                    )
                self._group_diagrams.addChild(item)
        finally:
            self._tree.blockSignals(False)

    # ------------------------------------------------------------------
    # Refresh — Probes / Plots placeholders
    # ------------------------------------------------------------------

    def _refresh_placeholders(self) -> None:
        """Render the static placeholder rows.

        Probes is still a placeholder pending B5+ inline migration.
        The Plots group is populated dynamically from the bound plot
        pane (see :meth:`bind_plot_pane`); when nothing is bound yet
        or no non-diagram plots exist, the placeholder is shown.
        """
        QtWidgets, _ = _qt()
        self._group_probes.takeChildren()
        empty = QtWidgets.QTreeWidgetItem(
            ["(see Probes tab — coming inline)"]
        )
        flags = empty.flags() & ~self._unselectable_mask()
        empty.setFlags(flags)
        empty.setForeground(0, self._dim_brush())
        self._group_probes.addChild(empty)
        # Plots group is owned by _refresh_plots; render its empty
        # state here for the unbound case so the tree is never blank.
        self._refresh_plots()

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

    def _accent_brush(self):
        """Accent brush for the active-plot indicator."""
        from qtpy import QtGui
        from .theme import THEME
        return QtGui.QBrush(QtGui.QColor(THEME.current.accent))

    # ------------------------------------------------------------------
    # Plot-pane binding (B++ §7 two-way tree ↔ tab binding)
    # ------------------------------------------------------------------

    def bind_plot_pane(self, plot_pane: Any) -> None:
        """Wire the Plots group to a :class:`PlotPane` instance.

        Subscribes to the pane's ``on_tabs_changed`` and
        ``on_active_changed`` so the tree stays in sync, and routes
        clicks on Plots-group leaves back to ``plot_pane.set_active``.
        Idempotent — calling it twice rewires cleanly.
        """
        if self._unsub_plot_tabs is not None:
            try:
                self._unsub_plot_tabs()
            except Exception:
                pass
            self._unsub_plot_tabs = None
        self._plot_pane = plot_pane
        if plot_pane is None:
            self._refresh_plots()
            return
        self._unsub_plot_tabs = plot_pane.on_tabs_changed(self._refresh_plots)
        plot_pane.on_active_changed(self._on_plot_active_changed)
        self._refresh_plots()

    def _refresh_plots(self) -> None:
        """Repopulate the Plots group from the bound plot pane.

        Diagram side-panel tabs (key tuple starting with ``"diagram"``)
        are skipped — those already appear under the Diagrams group.
        Other tabs (history plots from shift-click, future user-created
        plots) become first-class navigation rows.
        """
        QtWidgets, _ = _qt()
        self._group_plots.takeChildren()

        keys = self._collect_plot_keys()
        if not keys:
            empty = QtWidgets.QTreeWidgetItem(
                ["(shift-click a node to plot a time-history)"]
            )
            flags = empty.flags() & ~self._unselectable_mask()
            empty.setFlags(flags)
            empty.setForeground(0, self._dim_brush())
            self._group_plots.addChild(empty)
            return

        active = (
            self._plot_pane.active_key()
            if self._plot_pane is not None else None
        )
        for key in keys:
            label = self._plot_pane.tab_label(key) or str(key)
            item = QtWidgets.QTreeWidgetItem([label])
            item.setData(0, _ROLE_PLOT_KEY, key)
            self._group_plots.addChild(item)
            if key == active:
                self._mark_plot_active(item, True)

    def _on_plot_active_changed(self, key: Any) -> None:
        """Update the active-row indicator when the plot pane switches tabs."""
        for i in range(self._group_plots.childCount()):
            item = self._group_plots.child(i)
            row_key = item.data(0, _ROLE_PLOT_KEY)
            self._mark_plot_active(item, row_key == key and key is not None)

    def _mark_plot_active(self, item: Any, active: bool) -> None:
        font = item.font(0)
        font.setBold(active)
        item.setFont(0, font)
        item.setForeground(
            0, self._accent_brush() if active else self._normal_brush(),
        )

    def _normal_brush(self):
        from qtpy import QtGui
        from .theme import THEME
        return QtGui.QBrush(QtGui.QColor(THEME.current.text))

    def _collect_plot_keys(self) -> list[Any]:
        """Plot-pane keys that should appear under the Plots group.

        Diagram side panels share their lifecycle with the diagrams
        listed under the Diagrams group; surfacing them again here
        would be redundant and double-list every fiber / layer panel.
        """
        if self._plot_pane is None:
            return []
        out: list[Any] = []
        for key in self._plot_pane.keys():
            if isinstance(key, tuple) and len(key) > 0 and key[0] == "diagram":
                continue
            out.append(key)
        return out

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item, _column: int) -> None:
        if item is None:
            return
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

        # Plot row click → focus the matching plot-pane tab. The pane
        # fires on_active_changed back to us, which re-renders the
        # active-row indicator (idempotent).
        plot_key = item.data(0, _ROLE_PLOT_KEY)
        if plot_key is not None and self._plot_pane is not None:
            self._plot_pane.set_active(plot_key)
            return

    def _on_item_changed(self, item, _column: int) -> None:
        """Handle inline-rename commit on a composition row."""
        if item is None:
            return
        comp_id = item.data(0, _ROLE_COMPOSITION_KEY)
        if comp_id is None:
            return
        # Strip any trailing layer-count suffix the user may have left
        # in the editor (e.g. "Geometry (3)" — drop the count).
        text = item.text(0).strip()
        if "(" in text and text.endswith(")"):
            text = text[: text.rfind("(")].strip() or text
        mgr = self._director.compositions
        mgr.rename(comp_id, text)
        # Manager observer fires _refresh_diagrams which restamps the
        # display label with the right (count) suffix.

    def _on_current_item_changed(self, current, _previous) -> None:
        # Composition row → set it as the active composition and
        # route to the details dock. Off the row → no change to
        # active composition; just clear the panel listener.
        composition_key = (
            current.data(0, _ROLE_COMPOSITION_KEY)
            if current is not None else None
        )
        if composition_key is not None:
            # Set as active in the manager so all downstream
            # consumers (details panel, + Add layer) see it.
            try:
                self._director.compositions.set_active(composition_key)
            except Exception:
                pass
            if self._on_composition_selected is not None:
                self._on_composition_selected(composition_key)
            return
        # Anywhere else: notify listeners so they idle.
        if self._on_composition_selected is not None:
            self._on_composition_selected(None)
        if self._on_diagram_selected is not None:
            self._on_diagram_selected(None)

    # ------------------------------------------------------------------
    # Composition row actions (context menu, F2 rename, + Add diagram)
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos) -> None:
        """Composition rows → rename / duplicate / delete.
        Diagrams group header → Add diagram."""
        QtWidgets, QtCore = _qt()
        item = self._tree.itemAt(pos)
        if item is None:
            return
        global_pos = self._tree.viewport().mapToGlobal(pos)

        # Diagrams group header → Add diagram only.
        if item.data(0, _ROLE_GROUP_KEY) == "diagrams":
            menu = QtWidgets.QMenu(self._widget)
            act_add = menu.addAction("+ Add diagram")
            chosen = menu.exec_(global_pos)
            if chosen == act_add:
                self._on_add_diagram()
            return

        comp_id = item.data(0, _ROLE_COMPOSITION_KEY)
        if comp_id is None:
            return

        comp = self._director.compositions.find(comp_id)
        if comp is None:
            return

        menu = QtWidgets.QMenu(self._widget)
        act_rename = menu.addAction("Rename… (F2)")
        act_duplicate = menu.addAction("Duplicate")
        act_delete = menu.addAction("Delete")
        if comp.locked:
            act_delete.setEnabled(False)
            act_delete.setToolTip("Geometry is locked and cannot be deleted")
        menu.addSeparator()
        act_add = menu.addAction("+ Add diagram")
        chosen = menu.exec_(global_pos)
        if chosen == act_rename:
            self._tree.editItem(item, 0)
        elif chosen == act_duplicate:
            self._director.compositions.duplicate(comp_id)
        elif chosen == act_delete:
            self._on_delete_composition(comp_id)
        elif chosen == act_add:
            self._on_add_diagram()

    def _begin_inline_rename(self) -> None:
        """F2 → enter edit mode on the currently-selected composition row."""
        item = self._tree.currentItem()
        if item is None:
            return
        if item.data(0, _ROLE_COMPOSITION_KEY) is None:
            return
        self._tree.editItem(item, 0)

    def _on_add_diagram(self) -> None:
        """Header `+` / context menu → create a new composition."""
        comp = self._director.compositions.add(name="Diagram", make_active=True)
        # Refresh and highlight the new row so the user can immediately
        # F2 to rename, or click + Add layer to start populating.
        self._refresh_diagrams()
        for i in range(self._group_diagrams.childCount()):
            child = self._group_diagrams.child(i)
            if child.data(0, _ROLE_COMPOSITION_KEY) == comp.id:
                self._tree.setCurrentItem(child)
                break

    def _on_delete_composition(self, comp_id: str) -> None:
        """Delete a composition + every layer belonging to it."""
        mgr = self._director.compositions
        comp = mgr.find(comp_id)
        if comp is None or comp.locked:
            return
        # Tear down layers via the registry first; the manager just
        # tracks groupings.
        for layer in list(comp.layers):
            try:
                self._director.registry.remove(layer)
            except Exception:
                pass
            mgr.remove_layer(layer)
        mgr.remove(comp_id)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _currently_selected_diagram(self) -> Optional[Diagram]:
        item = self._tree.currentItem()
        if item is None:
            return None
        return item.data(0, _ROLE_DIAGRAM_OBJ)

    def _select_composition(self, comp_id: str) -> bool:
        """Move the outline selection to the row for ``comp_id``."""
        for i in range(self._group_diagrams.childCount()):
            child = self._group_diagrams.child(i)
            if child.data(0, _ROLE_COMPOSITION_KEY) == comp_id:
                self._tree.setCurrentItem(child)
                return True
        return False

    def _select_diagram(self, diagram: Diagram) -> bool:
        """Select the row showing ``diagram``. Returns True if found."""
        for i in range(self._group_diagrams.childCount()):
            child = self._group_diagrams.child(i)
            if child.data(0, _ROLE_DIAGRAM_OBJ) is diagram:
                self._tree.setCurrentItem(child)
                return True
        return False
