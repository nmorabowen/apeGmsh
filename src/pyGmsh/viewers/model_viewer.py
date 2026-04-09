"""
ModelViewer — Interactive BRep model viewer.

Assembled from independent core modules — no inheritance.
Provides the same public API as the old ``SelectionPicker``:

    viewer = ModelViewer(parent, model)
    viewer.show()
    print(viewer.tags)           # list[DimTag]
    print(viewer.selection)      # Selection object
    print(viewer.active_group)   # str | None
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

from .core.entity_registry import DimTag

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase
    from pyGmsh.core.Model import Model


class ModelViewer:
    """Interactive BRep model viewer with physical group management.

    Parameters
    ----------
    parent : _SessionBase
        The pyGmsh session (provides ``model_name``, ``_verbose``).
    model : Model
        The pyGmsh model (provides ``sync()``).
    physical_group : str, optional
        Auto-activate this physical group on open.
    dims : list[int], optional
        Which entity dimensions to show (default: ``[0, 1, 2, 3]``).
    point_size, line_width, surface_opacity, show_surface_edges
        Visual properties forwarded to the scene builder.
    fast : bool
        Ignored (always fast). Kept for backward compatibility.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        model: "Model",
        *,
        physical_group: str | None = None,
        dims: list[int] | None = None,
        point_size: float = 10.0,
        line_width: float = 6.0,
        surface_opacity: float = 0.35,
        show_surface_edges: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ) -> None:
        self._parent = parent
        self._model = model
        self._dims = dims if dims is not None else [0, 1, 2, 3]
        self._physical_group = physical_group

        # Visual props
        self._point_size = point_size
        self._line_width = line_width
        self._surface_opacity = surface_opacity
        self._show_surface_edges = show_surface_edges

        # Populated during show()
        self._selection_state = None
        self._registry = None

    # ------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------

    def show(self, *, title: str | None = None, maximized: bool = True):
        """Open the viewer window, block until closed."""
        from .core.navigation import install_navigation
        from .core.color_manager import ColorManager
        from .core.pick_engine import PickEngine
        from .core.visibility import VisibilityManager
        from .core.selection import SelectionState
        from .scene.brep_scene import build_brep_scene
        from .ui.viewer_window import ViewerWindow
        from .ui.preferences import PreferencesTab
        from .ui.model_tabs import BrowserTab, FilterTab, ViewTab

        # Ensure geometry is synced
        gmsh.model.occ.synchronize()

        # ── Window (creates QApplication + plotter) ─────────────────
        default_title = (
            f"ModelViewer — {self._parent.model_name}"
            + (f" → {self._physical_group}" if self._physical_group else "")
        )

        # ── Selection state ─────────────────────────────────────────
        sel = SelectionState()
        self._selection_state = sel

        # Seed group order with pre-existing Gmsh groups (by tag)
        for pg_dim, pg_tag in sorted(gmsh.model.getPhysicalGroups(), key=lambda x: x[1]):
            try:
                pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
                if pg_name and pg_name not in sel._group_order:
                    sel._group_order.append(pg_name)
            except Exception:
                pass

        if self._physical_group is not None:
            sel.set_active_group(self._physical_group)

        def _on_close():
            n = sel.flush_to_gmsh()
            if self._parent._verbose:
                print(f"[viewer] closed — {n} physical group(s) written, "
                      f"{len(sel.picks)} picks in working set")

        # Create window FIRST so QApplication exists for Qt widgets
        win = ViewerWindow(
            title=title or default_title,
            on_close=_on_close,
        )

        # ── UI tabs (created AFTER QApplication exists) ─────────────
        prefs = PreferencesTab(
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
        )

        def _on_new_group():
            from qtpy import QtWidgets
            name, ok = QtWidgets.QInputDialog.getText(
                win.window, "New Physical Group",
                "Group name:",
            )
            if ok and name.strip():
                n = name.strip()
                current_picks = list(sel._picks)
                # Stage current picks as the new group directly
                sel._staged_groups[n] = current_picks
                # Switch to the new group (loads picks from staged)
                sel.set_active_group(n)
                browser.refresh()
                if current_picks:
                    win.set_status(
                        f"Group '{n}' created with {len(current_picks)} entities"
                    )
                else:
                    win.set_status(f"Active group: {n} — pick entities to add")

        def _on_rename_group(old_name: str):
            from qtpy import QtWidgets
            new_name, ok = QtWidgets.QInputDialog.getText(
                win.window, "Rename Group",
                "New name:", text=old_name,
            )
            if ok and new_name.strip():
                sel.rename_group(old_name, new_name.strip())
                browser.refresh()

        def _on_delete_group(name: str):
            from qtpy import QtWidgets
            reply = QtWidgets.QMessageBox.question(
                win.window, "Delete Group",
                f"Delete physical group '{name}'?",
            )
            if reply == QtWidgets.QMessageBox.Yes:
                sel.delete_group(name)
                from .core.selection import _delete_group_by_name
                _delete_group_by_name(name)
                browser.refresh()
                win.set_status(f"Deleted group: {name}")

        def _on_group_activated(name: str):
            sel.set_active_group(name)
            browser.refresh()
            n = len(sel.picks)
            win.set_status(f"Active group: {name} ({n} entities)")

        browser = BrowserTab(
            sel,
            on_group_activated=_on_group_activated,
            on_entity_toggled=lambda dt: sel.toggle(dt),
            on_new_group=_on_new_group,
            on_rename_group=_on_rename_group,
            on_delete_group=_on_delete_group,
        )

        filter_tab = FilterTab(self._dims)

        # ── View tab (entity labels) ────────────────────────────────
        _label_actors: list = []
        _DIM_ABBR = {0: "P", 1: "C", 2: "S", 3: "V"}

        def _on_labels_changed(active_dims, font_size, use_names):
            # Remove existing labels
            for a in _label_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _label_actors.clear()

            for dim, show in active_dims.items():
                if not show:
                    continue
                points = []
                labels = []
                for _, tag in gmsh.model.getEntities(dim=dim):
                    try:
                        bb = gmsh.model.getBoundingBox(dim, tag)
                        cx = (bb[0] + bb[3]) * 0.5
                        cy = (bb[1] + bb[4]) * 0.5
                        cz = (bb[2] + bb[5]) * 0.5
                        points.append([cx, cy, cz])
                    except Exception:
                        continue
                    if use_names:
                        # Try to get physical group name
                        name = None
                        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(dim):
                            try:
                                ents = gmsh.model.getEntitiesForPhysicalGroup(
                                    pg_dim, pg_tag,
                                )
                                if tag in ents:
                                    name = gmsh.model.getPhysicalName(
                                        pg_dim, pg_tag,
                                    )
                                    break
                            except Exception:
                                pass
                        labels.append(
                            name or f"{_DIM_ABBR[dim]}{tag}"
                        )
                    else:
                        labels.append(f"{_DIM_ABBR[dim]}{tag}")

                if not points:
                    continue

                try:
                    actor = plotter.add_point_labels(
                        np.array(points), labels,
                        font_size=font_size,
                        text_color="white",
                        shape_color="#333333",
                        shape_opacity=0.6,
                        show_points=False,
                        always_visible=True,
                        name=f"_labels_dim{dim}",
                    )
                    _label_actors.append(actor)
                except Exception:
                    pass

            plotter.render()

        view_tab = ViewTab(
            self._dims,
            on_labels_changed=_on_labels_changed,
        )

        # Add tabs to window
        win.add_tab("Browser", browser.widget)
        win.add_tab("View", view_tab.widget)
        win.add_tab("Filter", filter_tab.widget)
        win.add_tab("Preferences", prefs.widget)

        plotter = win.plotter

        # ── Build scene ─────────────────────────────────────────────
        registry = build_brep_scene(
            plotter, self._dims,
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
        )
        self._registry = registry

        # ── Core modules ────────────────────────────────────────────
        color_mgr = ColorManager(registry)
        vis_mgr = VisibilityManager(registry, color_mgr, sel)
        pick_engine = PickEngine(plotter, registry)

        # ── Wire callbacks ──────────────────────────────────────────

        # Pick → selection
        def _on_pick(dt: DimTag, ctrl: bool):
            if ctrl:
                sel.unpick(dt)
            else:
                sel.toggle(dt)

        pick_engine.on_pick = _on_pick
        pick_engine.set_hidden_check(vis_mgr.is_hidden)

        # Filter → pick engine + visual dim feedback
        def _on_filter(active_dims: set[int]):
            pick_engine.set_pickable_dims(active_dims)
            # Dim non-pickable dimension actors
            for dim in registry.dims:
                actor = registry.dim_actors.get(dim)
                if actor is None:
                    continue
                if dim in active_dims:
                    actor.GetProperty().SetOpacity(
                        self._surface_opacity if dim >= 2 else 1.0
                    )
                else:
                    actor.GetProperty().SetOpacity(0.1)
            plotter.render()

        filter_tab._on_filter_changed = _on_filter

        # Hover → color
        _prev_hover: list[DimTag | None] = [None]

        def _on_hover(dt: DimTag | None):
            old = _prev_hover[0]
            _prev_hover[0] = dt
            if old is not None and old != dt:
                is_picked = old in sel._picks
                color_mgr.set_entity_state(old, picked=is_picked)
            if dt is not None:
                is_picked = dt in sel._picks
                if not is_picked:
                    color_mgr.set_entity_state(dt, hovered=True)
            plotter.render()

        pick_engine.on_hover = _on_hover

        # Selection changed → recolor + refresh UI
        def _on_sel_changed():
            # Recolor all entities based on current state
            for entity_dt in registry.all_entities():
                is_picked = entity_dt in sel._picks
                is_hidden = vis_mgr.is_hidden(entity_dt)
                color_mgr.set_entity_state(
                    entity_dt, picked=is_picked, hidden=is_hidden,
                )
            plotter.render()
            n = len(sel.picks)
            grp = sel.active_group or "none"
            win.set_status(f"{n} picked | group: {grp}")

        sel.on_changed.append(_on_sel_changed)
        sel.on_changed.append(lambda: browser.update_active())
        # Write active group to Gmsh on every pick change
        sel.on_changed.append(lambda: sel.commit_active_group())

        # Visibility changed → render
        vis_mgr.on_changed.append(lambda: plotter.render())

        # Box select
        def _on_box(dts: list[DimTag], ctrl: bool):
            if ctrl:
                sel.box_remove(dts)
            else:
                sel.box_add(dts)

        pick_engine.on_box_select = _on_box

        # ── Navigation ──────────────────────────────────────────────
        install_navigation(
            plotter,
            get_orbit_pivot=lambda: sel.centroid(registry),
        )

        # ── Install pick engine ─────────────────────────────────────
        pick_engine.install()

        # ── Toolbar buttons for visibility ──────────────────────────
        win.add_toolbar_separator()
        win.add_toolbar_button(
            "Hide selected (H)", "H",
            lambda: (vis_mgr.hide(), plotter.render()),
        )
        win.add_toolbar_button(
            "Isolate selected (I)", "I",
            lambda: (vis_mgr.isolate(), plotter.render()),
        )
        win.add_toolbar_button(
            "Reveal all (R)", "R",
            lambda: (vis_mgr.reveal_all(), plotter.render()),
        )

        # ── Keybindings ─────────────────────────────────────────────
        # VTK-level (only when 3D viewport has focus)
        plotter.add_key_event("h", lambda: (vis_mgr.hide(), plotter.render()))
        plotter.add_key_event("i", lambda: (vis_mgr.isolate(), plotter.render()))
        plotter.add_key_event("r", lambda: (vis_mgr.reveal_all(), plotter.render()))
        plotter.add_key_event("u", lambda: sel.undo())

        for key, dim_set in [
            ("1", {0}), ("2", {1}), ("3", {2}), ("4", {3}),
            ("0", set(self._dims)),
        ]:
            plotter.add_key_event(
                key,
                lambda ds=dim_set: pick_engine.set_pickable_dims(ds),
            )

        # Window-level (work regardless of focus / mouse position)
        win.add_shortcut("Escape", lambda: sel.clear())
        win.add_shortcut("Q", lambda: win.window.close())

        # ── Pre-load group if specified ─────────────────────────────
        if self._physical_group is not None and sel.picks:
            _on_sel_changed()

        # ── Run ─────────────────────────────────────────────────────
        win.exec()
        return self

    # ------------------------------------------------------------------
    # Public API (preserved from SelectionPicker)
    # ------------------------------------------------------------------

    @property
    def selection(self):
        """The current working set as a :class:`Selection` object."""
        from pyGmsh.viz.Selection import Selection
        picks = self._selection_state.picks if self._selection_state else []
        return Selection(picks, self._parent)

    @property
    def tags(self) -> list[DimTag]:
        """The current working set as a list of DimTags."""
        return self._selection_state.picks if self._selection_state else []

    @property
    def active_group(self) -> str | None:
        """The name of the physical group currently receiving picks."""
        if self._selection_state is None:
            return None
        return self._selection_state.active_group

    def to_physical(self, name: str | None = None) -> int | None:
        """Write the current picks as a Gmsh physical group."""
        if self._selection_state is None:
            return None
        sel = self._selection_state
        group_name = name or self._physical_group
        if not group_name:
            return None
        sel.apply_group(group_name)
        return sel.flush_to_gmsh()
