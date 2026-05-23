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
                               # | "load_pattern" | "mass" | "constraint_kind"
_ROLE_PAYLOAD = int(0x0211)    # name (group/type/part/pattern/kind) | DimTag


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
    loads_composite / mass_composite / constraints_composite
        Optional FEM-input composites. When given, the outline surfaces
        a top-level section per category with per-pattern (Loads),
        per-kind (Constraints), or single-row (Masses) entries. Eye
        click computes the new active set and fires the corresponding
        rebuild callback.
    on_group_activated
        Callback fired when a Physical Group row is clicked.
    on_load_patterns_changed
        Callback fired with the set of currently-visible load pattern
        names whenever a Loads row's eye is toggled. Mirrors the
        existing :attr:`LoadsTabPanel.on_patterns_changed` contract;
        wired in mesh.viewer to ``_rebuild_loads_overlay``.
    on_mass_visibility_changed
        Callback fired with ``bool`` (overlay shown / hidden) when the
        Masses row's eye is toggled. Wired to ``_rebuild_mass_overlay``.
    on_constraint_kinds_changed
        Callback fired with the set of currently-visible constraint
        kinds whenever a Constraints row's eye is toggled. Mirrors
        :attr:`ConstraintsTabPanel.on_kinds_changed`; wired to
        ``_rebuild_constraints_overlay``.

    Known limitation (2026-05-16)
    -----------------------------
    The right-side ``LoadsTab`` / ``MassTab`` / ``ConstraintsTab``
    keep their own visibility state. Toggling in only one surface
    works; alternating between the outline eye and the tab checkbox
    causes the overlay to flip to whichever surface fired last. State
    unification (single source-of-truth across both surfaces) is a
    deliberate follow-up — the design discussion lives in the PR
    description.
    """

    def __init__(
        self,
        scene: "MeshSceneData",
        selection: "SelectionState",
        vis_mgr: "VisibilityManager",
        parts_registry: Any = None,
        *,
        loads_composite: Any = None,
        mass_composite: Any = None,
        constraints_composite: Any = None,
        on_group_activated: Optional[Callable[[str], None]] = None,
        on_load_patterns_changed: Optional[
            Callable[[set[str]], None]
        ] = None,
        on_mass_visibility_changed: Optional[Callable[[bool], None]] = None,
        on_constraint_kinds_changed: Optional[
            Callable[[set[str]], None]
        ] = None,
        on_row_focused: Optional[Callable[[str, Any], None]] = None,
        overlay_model: Any = None,
    ) -> None:
        QtCore, _, QtWidgets = _qt()
        self._scene = scene
        self._selection = selection
        self._vis_mgr = vis_mgr
        self._parts = parts_registry
        self._loads = loads_composite
        self._masses = mass_composite
        self._constraints = constraints_composite
        self._on_group_activated = on_group_activated
        self._on_load_patterns_changed = on_load_patterns_changed
        self._on_mass_visibility_changed = on_mass_visibility_changed
        self._on_constraint_kinds_changed = on_constraint_kinds_changed
        # PR5 — optional :class:`OverlayVisibilityModel`.  When
        # supplied, the outline subscribes to model changes and
        # refreshes its eye-icons to match the model state.  This
        # closes the cross-surface UI sync gap: a tab checkbox write
        # now visually updates the corresponding outline eye-icon.
        # Back-compat: ``None`` → behaviour unchanged (legacy callbacks
        # are the only path).
        self._overlay_model = overlay_model
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
        self._group_loads = self._make_header_item("Loads")
        self._group_masses = self._make_header_item("Masses")
        self._group_constraints = self._make_header_item("Constraints")
        for header in (
            self._group_groups, self._group_types, self._group_parts,
            self._group_loads, self._group_masses, self._group_constraints,
        ):
            tree.addTopLevelItem(header)
            header.setExpanded(True)

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
        self._refresh_loads()
        self._refresh_masses()
        self._refresh_constraints()
        # PR5 — wire UI sync to the OverlayVisibilityModel after the
        # rows exist.  Subscribed once; subsequent writes from the
        # tab panels propagate here automatically.
        if self._overlay_model is not None and not getattr(
            self, "_overlay_model_wired", False,
        ):
            self._overlay_model.subscribe(self._sync_from_overlay_model)
            self._overlay_model_wired = True

    def _sync_from_overlay_model(self) -> None:
        """Refresh eye-icon visual state from the OverlayVisibilityModel.

        For load_pattern / constraint_kind rows: ROLE_VISIBLE =
        (payload in model.{load_patterns | constraint_kinds}).
        For the mass row: ROLE_VISIBLE = model.mass_visible.

        The repaint is triggered via ``viewport().update()``; the
        eye-icon delegate reads ROLE_VISIBLE on paint.  No signal
        round-trip — ROLE_VISIBLE is a data role, not a widget state.
        """
        if self._overlay_model is None:
            return
        load_patterns = self._overlay_model.load_patterns
        for i in range(self._group_loads.childCount()):
            row = self._group_loads.child(i)
            if row.data(0, _ROLE_KIND) == "load_pattern":
                payload = row.data(0, _ROLE_PAYLOAD)
                row.setData(0, ROLE_VISIBLE, payload in load_patterns)
        mass_visible = self._overlay_model.mass_visible
        for i in range(self._group_masses.childCount()):
            row = self._group_masses.child(i)
            if row.data(0, _ROLE_KIND) == "mass":
                row.setData(0, ROLE_VISIBLE, mass_visible)
        constraint_kinds = self._overlay_model.constraint_kinds
        for i in range(self._group_constraints.childCount()):
            row = self._group_constraints.child(i)
            if row.data(0, _ROLE_KIND) == "constraint_kind":
                payload = row.data(0, _ROLE_PAYLOAD)
                row.setData(0, ROLE_VISIBLE, payload in constraint_kinds)
        self._tree.viewport().update()

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
    # FEM-input sections — Loads / Masses / Constraints
    # ------------------------------------------------------------------

    def _refresh_loads(self) -> None:
        """One row per :class:`LoadPattern` in the composite.

        Each row carries the pattern name as its payload; eye toggle
        recomputes the active-patterns set and fires the rebuild
        callback. All patterns start visible (matches the
        ``LoadsTab.refresh()`` default).
        """
        _, _, QtWidgets = _qt()
        self._group_loads.takeChildren()
        if self._loads is None:
            self._group_loads.setHidden(True)
            return
        try:
            patterns = list(self._loads.patterns())
        except Exception:
            patterns = []
        if not patterns:
            self._group_loads.setHidden(True)
            return
        self._group_loads.setHidden(False)
        for name in patterns:
            item = QtWidgets.QTreeWidgetItem(self._group_loads)
            item.setText(0, name)
            item.setData(0, _ROLE_KIND, "load_pattern")
            item.setData(0, _ROLE_PAYLOAD, name)
            item.setData(0, ROLE_VISIBLE, True)

    def _refresh_masses(self) -> None:
        """Single row when the masses composite has any entries."""
        _, _, QtWidgets = _qt()
        self._group_masses.takeChildren()
        if self._masses is None:
            self._group_masses.setHidden(True)
            return
        # The composite doesn't expose a count cheaply; if the user
        # gave us a composite at all, surface the row. The eye toggle
        # is a boolean.
        self._group_masses.setHidden(False)
        item = QtWidgets.QTreeWidgetItem(self._group_masses)
        item.setText(0, "Masses overlay")
        item.setData(0, _ROLE_KIND, "mass")
        item.setData(0, _ROLE_PAYLOAD, None)
        item.setData(0, ROLE_VISIBLE, True)

    def _refresh_constraints(self) -> None:
        """One row per **distinct kind** present in ``constraint_defs``.

        Kinds follow the same resolution rule as
        :func:`apeGmsh.viewers.ui.constraints_tab._def_kind_key` —
        ``RigidLinkDef`` resolves to ``rigid_beam`` / ``rigid_rod``
        based on ``link_type`` so the outline rows match the
        rebuild-callback filter.
        """
        _, _, QtWidgets = _qt()
        self._group_constraints.takeChildren()
        if self._constraints is None:
            self._group_constraints.setHidden(True)
            return
        defs = getattr(self._constraints, "constraint_defs", None) or []
        if not defs:
            self._group_constraints.setHidden(True)
            return
        # Count defs per kind for the "Detail" column.
        from .constraints_tab import _def_kind_key
        counts: dict[str, int] = {}
        for d in defs:
            key = _def_kind_key(d)
            counts[key] = counts.get(key, 0) + 1
        self._group_constraints.setHidden(False)
        for kind in sorted(counts.keys()):
            item = QtWidgets.QTreeWidgetItem(self._group_constraints)
            item.setText(0, kind)
            item.setText(1, str(counts[kind]))
            item.setData(0, _ROLE_KIND, "constraint_kind")
            item.setData(0, _ROLE_PAYLOAD, kind)
            item.setData(0, ROLE_VISIBLE, True)

    def _active_load_patterns(self) -> set[str]:
        """Pattern names whose eye is currently on."""
        out: set[str] = set()
        for i in range(self._group_loads.childCount()):
            row = self._group_loads.child(i)
            if (row.data(0, _ROLE_KIND) == "load_pattern"
                    and bool(row.data(0, ROLE_VISIBLE))):
                out.add(row.data(0, _ROLE_PAYLOAD))
        return out

    def _active_constraint_kinds(self) -> set[str]:
        """Constraint kinds whose eye is currently on."""
        out: set[str] = set()
        for i in range(self._group_constraints.childCount()):
            row = self._group_constraints.child(i)
            if (row.data(0, _ROLE_KIND) == "constraint_kind"
                    and bool(row.data(0, ROLE_VISIBLE))):
                out.add(row.data(0, _ROLE_PAYLOAD))
        return out

    def _mass_overlay_visible(self) -> bool:
        for i in range(self._group_masses.childCount()):
            row = self._group_masses.child(i)
            if row.data(0, _ROLE_KIND) == "mass":
                return bool(row.data(0, ROLE_VISIBLE))
        return False

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
        kind = item.data(0, _ROLE_KIND)
        # FEM-input rows don't have DimTags — they flip their own
        # ROLE_VISIBLE and fire the matching rebuild callback.
        if kind == "load_pattern":
            self._toggle_simple_row(item)
            if self._on_load_patterns_changed is not None:
                self._on_load_patterns_changed(self._active_load_patterns())
            return
        if kind == "mass":
            self._toggle_simple_row(item)
            if self._on_mass_visibility_changed is not None:
                self._on_mass_visibility_changed(self._mass_overlay_visible())
            return
        if kind == "constraint_kind":
            self._toggle_simple_row(item)
            if self._on_constraint_kinds_changed is not None:
                self._on_constraint_kinds_changed(
                    self._active_constraint_kinds(),
                )
            return

        # Substrate-mesh rows route through the visibility manager.
        dts = self._resolve_dts(item)
        if not dts:
            return
        current_hidden = set(self._vis_mgr.hidden)
        if self._group_is_visible(dts):
            current_hidden.update(dts)
        else:
            current_hidden.difference_update(dts)
        self._vis_mgr.set_hidden(current_hidden)

    def _toggle_simple_row(self, item: Any) -> None:
        """Flip ROLE_VISIBLE in place + repaint. Used by FEM-input rows
        whose visibility is owned by the outline (not the vis_mgr)."""
        item.setData(0, ROLE_VISIBLE, not bool(item.data(0, ROLE_VISIBLE)))
        try:
            self._tree.viewport().update()
        except Exception:
            pass

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
        # Generic row-focus signal — fires for every selectable kind
        # so the viewer can raise the matching property tab. Header
        # rows are non-selectable and never reach here.
        if kind in (
            "group", "type", "part",
            "load_pattern", "mass", "constraint_kind",
        ) and self._on_row_focused:
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
