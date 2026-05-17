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

from apeGmsh._types import DimTag

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _SessionBase
    from apeGmsh.core.Model import Model
    from .core.entity_registry import EntityRegistry
    from .core.selection import SelectionState


class ModelViewer:
    """Interactive BRep model viewer with physical group management.

    Displays BRep geometry, parts, physical groups, and labels.
    This is a **geometry-only** viewer — loads, constraints, and masses
    are mesh-resolved concepts and live on ``g.mesh.viewer()`` instead.

    Parameters
    ----------
    parent : _SessionBase
        The apeGmsh session (provides ``name``, ``_verbose``).
    model : Model
        The apeGmsh model (provides ``sync()``).
    physical_group : str, optional
        Auto-activate this physical group on open.
    dims : list[int], optional
        Which entity dimensions to show (default: ``[0, 1, 2, 3]``).
    point_size, line_width, surface_opacity, show_surface_edges
        Visual properties forwarded to the scene builder.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        model: "Model",
        *,
        physical_group: str | None = None,
        dims: list[int] | None = None,
        point_size: float | None = None,
        line_width: float | None = None,
        surface_opacity: float | None = None,
        show_surface_edges: bool | None = None,
        origin_markers: list[tuple[float, float, float]] | None = None,
        origin_marker_show_coords: bool | None = None,
    ) -> None:
        from .ui.preferences_manager import PREFERENCES
        p = PREFERENCES.current

        self._parent = parent
        self._model = model
        self._dims = dims if dims is not None else [0, 1, 2, 3]
        self._physical_group = physical_group

        # Visual props — explicit kwarg wins, otherwise pull user preference.
        self._point_size = point_size if point_size is not None else p.point_size
        self._line_width = line_width if line_width is not None else p.line_width
        self._surface_opacity = (
            surface_opacity if surface_opacity is not None else p.surface_opacity
        )
        self._show_surface_edges = (
            show_surface_edges if show_surface_edges is not None
            else p.show_surface_edges
        )

        # Origin marker overlay. User preference controls whether the
        # default is ``[(0,0,0)]`` or ``[]``; explicit kwarg wins.
        if origin_markers is not None:
            self._origin_markers: list[tuple[float, float, float]] = list(origin_markers)
        elif p.origin_marker_include_world_origin:
            self._origin_markers = [(0.0, 0.0, 0.0)]
        else:
            self._origin_markers = []
        self._origin_marker_show_coords = (
            origin_marker_show_coords if origin_marker_show_coords is not None
            else p.origin_marker_show_coords
        )

        # Populated during show()
        self._selection_state: "SelectionState | None" = None
        self._registry: "EntityRegistry | None" = None

        # Plan 04 step 4 — per-viewer ActiveObjects coordinator.
        # Constructed once a QApplication / window exists (in show()).
        # ModelViewer has no pick-mode concept — only the
        # ``selectionChanged`` bridge is wired today. The legacy
        # ``sel.on_changed`` cascade (recolor → tree → browser →
        # parts_tree → commit_active_group) stays untouched per the
        # plan doc's one-release compatibility shim policy; the bridge
        # gives future panels a Qt-signal entry point without forcing
        # them through ``SelectionState``'s internal callback list.
        self._active: Any = None

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
        from .ui.model_tabs import (
            FilterTab, ViewTab, SelectionTreePanel, PartsTreePanel,
        )

        # Ensure geometry is synced
        gmsh.model.occ.synchronize()

        # ── Window (creates QApplication + plotter) ─────────────────
        default_title = (
            f"ModelViewer — {self._parent.name}"
            + (f" -> {self._physical_group}" if self._physical_group else "")
        )

        # ── Selection state ─────────────────────────────────────────
        sel = SelectionState()
        self._selection_state = sel

        # Seed group order with pre-existing user-facing PGs (skip labels)
        from apeGmsh.core.Labels import is_label_pg
        for pg_dim, pg_tag in sorted(gmsh.model.getPhysicalGroups(), key=lambda x: x[1]):
            try:
                pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
                if pg_name and not is_label_pg(pg_name) and pg_name not in sel._group_order:
                    sel._group_order.append(pg_name)
            except Exception:
                pass

        if self._physical_group is not None:
            sel.set_active_group(self._physical_group)

        def _on_close():
            try:
                n = sel.flush_to_gmsh()
            except Exception as exc:
                # Log the full traceback so the user can debug, then surface
                # a dialog. Do NOT re-raise — the user is closing the window;
                # crashing their program after-the-fact loses session state.
                import sys
                import traceback
                print(
                    f"[viewer] flush_to_gmsh failed on close: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                try:
                    from qtpy import QtWidgets
                    QtWidgets.QMessageBox.critical(
                        win.window,
                        "Failed to write physical groups",
                        f"{exc}\n\nSee console for full traceback. "
                        "Pending picks were not committed.",
                    )
                except Exception:
                    pass
                return
            if self._parent._verbose:
                print(f"[viewer] closed — {n} physical group(s) written, "
                      f"{len(sel.picks)} picks in working set")

        # Create window FIRST so QApplication exists for Qt widgets.
        # ``window_key`` opts into layout persistence under
        # ``QSettings("apeGmsh", "ModelViewer")`` (plan 08 follow-up).
        win = ViewerWindow(
            title=title or default_title,
            on_close=_on_close,
            window_key="ModelViewer",
        )

        # ── Plan 04 step 4 — ActiveObjects coordinator ──────────────
        # One per viewer. Provides the ``selectionChanged`` signal that
        # future panels subscribe to; the legacy ``sel.on_changed``
        # cascade installed further below stays as the compatibility
        # path per the plan doc. The bridge into ActiveObjects is
        # registered alongside the cascade in the "Wire callbacks"
        # section so all selection observers are co-located.
        from .core._active_objects import ActiveObjects
        self._active = ActiveObjects(parent=win.window)

        # ── UI tabs (created AFTER QApplication exists) ─────────────
        # NOTE: PreferencesTab is created AFTER scene build (needs registry).
        # See "Preferences" block below build_brep_scene().

        _DIM_NAMES = {0: "points", 1: "curves", 2: "surfaces", 3: "volumes"}

        def _on_new_group():
            from qtpy import QtWidgets
            current_picks = list(sel._picks)
            # A Gmsh physical group is dimension-scoped. A mixed-dim
            # selection would be written as one PG per dimension under
            # the same name (looks duplicated, wrong for FEM export),
            # so reject it up front rather than silently splitting.
            dims = sorted({dt[0] for dt in current_picks})
            if len(dims) > 1:
                QtWidgets.QMessageBox.warning(
                    win.window,
                    "Mixed-dimension selection",
                    "A physical group must contain entities of a "
                    "single dimension.\n\nThe current selection spans: "
                    + ", ".join(_DIM_NAMES.get(d, str(d)) for d in dims)
                    + ".\n\nRefine it to one dimension and try again.",
                )
                return
            name, ok = QtWidgets.QInputDialog.getText(
                win.window, "New Physical Group",
                "Group name:",
            )
            if ok and name.strip():
                n = name.strip()
                # Stage current picks as the new group directly
                sel._staged_groups[n] = current_picks
                # Switch to the new group (loads picks from staged)
                sel.set_active_group(n)
                outline.refresh()
                if current_picks:
                    win.set_status(
                        f"Group '{n}' created with {len(current_picks)} entities"
                    )
                else:
                    win.set_status(f"Active group: {n} — pick entities to add")

        def _on_new_label():
            # The multi-dimensional counterpart to _on_new_group. A
            # label IS allowed to span dimensions — it is backed by one
            # ``_label:`` PG per dimension (PGs are dimension-scoped),
            # which the outline merges into one row.
            from qtpy import QtWidgets
            picks = list(sel._picks)
            if not picks:
                QtWidgets.QMessageBox.information(
                    win.window, "New Label",
                    "Select one or more entities first — a label "
                    "groups the current selection (any mix of "
                    "dimensions).",
                )
                return
            labels_api = getattr(self._parent, "labels", None)
            if labels_api is None:
                QtWidgets.QMessageBox.warning(
                    win.window, "New Label",
                    "This session exposes no labels API.",
                )
                return
            name, ok = QtWidgets.QInputDialog.getText(
                win.window, "New Label",
                "Label name (groups the selection across all its "
                "dimensions):",
            )
            if not (ok and name.strip()):
                return
            n = name.strip()
            by_dim: dict[int, list[int]] = {}
            for d, t in picks:
                by_dim.setdefault(int(d), []).append(int(t))
            try:
                # ``labels.add`` warns about cross-dim "ambiguous
                # lookups" when the same name spans dimensions — which
                # is precisely the intent of a multi-dim label, so
                # silence that one warning for this deliberate add.
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*already exists at dim=.*",
                    )
                    for d, tags in sorted(by_dim.items()):
                        labels_api.add(d, tags, n)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    win.window, "New Label",
                    f"Could not create label '{n}':\n{exc}",
                )
                return
            outline.refresh()
            dims_txt = ", ".join(
                _DIM_NAMES.get(d, str(d)) for d in sorted(by_dim)
            )
            win.set_status(
                f"Label '{n}' created from {len(picks)} entities "
                f"({dims_txt})"
            )

        def _on_rename_label(name: str):
            from qtpy import QtWidgets
            labels_api = getattr(self._parent, "labels", None)
            if labels_api is None:
                return
            new_name, ok = QtWidgets.QInputDialog.getText(
                win.window, "Rename Label",
                "New label name:", text=name,
            )
            if not (ok and new_name.strip()):
                return
            nn = new_name.strip()
            if nn == name:
                return
            try:
                # dim=None → rename across every dimension the label
                # spans (a label is multi-dimensional).
                labels_api.rename(name, nn)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    win.window, "Rename Label",
                    f"Could not rename label '{name}':\n{exc}",
                )
                return
            outline.refresh()
            win.set_status(f"Label '{name}' renamed to '{nn}'")

        def _on_delete_label(name: str):
            from qtpy import QtWidgets
            labels_api = getattr(self._parent, "labels", None)
            if labels_api is None:
                return
            reply = QtWidgets.QMessageBox.question(
                win.window, "Delete Label",
                f"Delete label '{name}' (all dimensions)?",
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            try:
                labels_api.remove(name)        # dim=None → all dims
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    win.window, "Delete Label",
                    f"Could not delete label '{name}':\n{exc}",
                )
                return
            outline.refresh()
            win.set_status(f"Deleted label: {name}")

        def _on_rename_group(old_name: str):
            from qtpy import QtWidgets
            new_name, ok = QtWidgets.QInputDialog.getText(
                win.window, "Rename Group",
                "New name:", text=old_name,
            )
            if ok and new_name.strip():
                sel.rename_group(old_name, new_name.strip())
                outline.refresh()

        def _on_delete_group(name: str):
            from qtpy import QtWidgets
            reply = QtWidgets.QMessageBox.question(
                win.window, "Delete Group",
                f"Delete physical group '{name}'?",
            )
            # Qt6 uses QMessageBox.StandardButton.Yes; Qt5 had the
            # top-level alias. Compare via the enum member to stay
            # portable across PyQt5/PySide2/PyQt6/PySide6.
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                sel.delete_group(name)
                from .core.selection import _delete_group_by_name
                _delete_group_by_name(name)
                outline.refresh()
                win.set_status(f"Deleted group: {name}")

        def _on_group_activated(name: str):
            sel.set_active_group(name)
            # In-place active-row restyle only. A full refresh()
            # (takeChildren + rebuild) would reset scroll/expansion
            # and make rows visibly jump on every click; the
            # structure is unchanged here, only which group is active.
            outline.update_active()
            n = len(sel.picks)
            win.set_status(f"Active group: {name} ({n} entities)")

        # Filter -> pick engine + visual dim feedback. The closure references
        # plotter / registry / pick_engine which are bound later in this
        # method; safe because the callback only fires after ``win.exec()``.
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

        filter_tab = FilterTab(self._dims, on_filter_changed=_on_filter)

        # ── View tab (entity labels) ────────────────────────────────
        _label_actors: list = []
        _DIM_ABBR = {0: "P", 1: "C", 2: "S", 3: "V"}

        def _on_labels_changed(
            active_dims, font_size, use_names,
            show_parts=False, show_entity_labels=False,
        ):
            from apeGmsh.core.Labels import is_label_pg, strip_prefix

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
                    dt = (dim, tag)
                    c = registry.centroid(dt)
                    if c is not None:
                        points.append(c)
                    else:
                        try:
                            bb = gmsh.model.getBoundingBox(dim, tag)
                            cx = (bb[0] + bb[3]) * 0.5 - registry.origin_shift[0]
                            cy = (bb[1] + bb[4]) * 0.5 - registry.origin_shift[1]
                            cz = (bb[2] + bb[5]) * 0.5 - registry.origin_shift[2]
                            points.append([cx, cy, cz])
                        except Exception:
                            continue
                    if use_names:
                        name = None
                        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(dim):
                            try:
                                ents = gmsh.model.getEntitiesForPhysicalGroup(
                                    pg_dim, pg_tag,
                                )
                                if tag in ents:
                                    pg_name = gmsh.model.getPhysicalName(
                                        pg_dim, pg_tag,
                                    )
                                    # Skip label PGs here — they show
                                    # in the dedicated entity-label
                                    # overlay below.
                                    if not is_label_pg(pg_name):
                                        name = pg_name
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

                from .ui.theme import THEME as _THEME
                try:
                    actor = plotter.add_point_labels(
                        np.array(points), labels,
                        font_size=font_size,
                        text_color=_THEME.current.text,
                        shape_color=_THEME.current.mantle,
                        shape_opacity=0.6,
                        show_points=False,
                        always_visible=True,
                        name=f"_labels_dim{dim}",
                    )
                    _label_actors.append(actor)
                except Exception:
                    pass

            # ── Part labels (one per instance, at centroid) ─────────
            parts_reg_local = getattr(self._parent, 'parts', None)
            if show_parts and parts_reg_local is not None:
                part_points = []
                part_labels = []
                for label, inst in parts_reg_local.instances.items():
                    # Use highest-dim entity centroid for placement
                    placed = False
                    for d in (3, 2, 1, 0):
                        for t in inst.entities.get(d, []):
                            c = registry.centroid((d, t))
                            if c is not None:
                                part_points.append(c)
                                part_labels.append(label)
                                placed = True
                                break
                        if placed:
                            break
                    if not placed and inst.bbox is not None:
                        bb = inst.bbox
                        part_points.append([
                            (bb[0] + bb[3]) * 0.5 - registry.origin_shift[0],
                            (bb[1] + bb[4]) * 0.5 - registry.origin_shift[1],
                            (bb[2] + bb[5]) * 0.5 - registry.origin_shift[2],
                        ])
                        part_labels.append(label)

                if part_points:
                    try:
                        actor = plotter.add_point_labels(
                            np.array(part_points), part_labels,
                            font_size=font_size + 2,
                            text_color=_THEME.current.success,
                            shape_color=_THEME.current.base,
                            shape_opacity=0.85,
                            show_points=False,
                            always_visible=True,
                            bold=True,
                            name="_labels_parts",
                        )
                        _label_actors.append(actor)
                    except Exception:
                        pass

            # ── Entity labels (Tier 1 — from g.labels) ────────────
            if show_entity_labels:
                label_points = []
                label_texts = []
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                    pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
                    if not is_label_pg(pg_name):
                        continue
                    display_name = strip_prefix(pg_name)
                    ent_tags = gmsh.model.getEntitiesForPhysicalGroup(
                        pg_dim, pg_tag,
                    )
                    for tag in ent_tags:
                        dt = (pg_dim, int(tag))
                        c = registry.centroid(dt)
                        if c is not None:
                            label_points.append(c)
                        else:
                            try:
                                bb = gmsh.model.getBoundingBox(pg_dim, int(tag))
                                cx = (bb[0] + bb[3]) * 0.5 - registry.origin_shift[0]
                                cy = (bb[1] + bb[4]) * 0.5 - registry.origin_shift[1]
                                cz = (bb[2] + bb[5]) * 0.5 - registry.origin_shift[2]
                                label_points.append([cx, cy, cz])
                            except Exception:
                                continue
                        label_texts.append(display_name)

                if label_points:
                    try:
                        actor = plotter.add_point_labels(
                            np.array(label_points), label_texts,
                            font_size=font_size,
                            text_color=_THEME.current.warning,
                            shape_color=_THEME.current.base,
                            shape_opacity=0.75,
                            show_points=False,
                            always_visible=True,
                            italic=True,
                            name="_labels_entities",
                        )
                        _label_actors.append(actor)
                    except Exception:
                        pass

            plotter.render()

        # ``tn_overlay`` is constructed later in this method (it needs the
        # registry's origin shift, only known after ``build_brep_scene``).
        # The closure resolves it lazily — safe because the callback only
        # fires after ``win.exec()``.
        def _on_geometry_probes_changed(show_tangents: bool, show_normals: bool):
            tn_overlay.set_show_tangents(show_tangents)
            tn_overlay.set_show_normals(show_normals)

        view_tab = ViewTab(
            self._dims,
            on_labels_changed=_on_labels_changed,
            on_geometry_probes_changed=_on_geometry_probes_changed,
        )

        # ── Selection tree panel ────────────────────────────────────
        def _tree_select_only(dts):
            sel.select_batch(dts, replace=True)

        def _tree_add(dts):
            sel.select_batch(dts)

        def _tree_remove(dts):
            sel.box_remove(dts)

        # Visibility callbacks — late-binding on vis_mgr (defined later
        # in this same method).
        def _tree_hide(dts):
            vis_mgr.hide_dts(dts)
            plotter.render()

        def _tree_isolate(dts):
            vis_mgr.isolate_dts(dts)
            plotter.render()

        def _tree_reveal_all():
            vis_mgr.reveal_all()
            plotter.render()

        sel_tree = SelectionTreePanel(
            on_select_only=_tree_select_only,
            on_add_to_selection=_tree_add,
            on_remove_from_selection=_tree_remove,
            on_hide=_tree_hide,
            on_isolate=_tree_isolate,
            on_reveal_all=_tree_reveal_all,
        )

        # Plan 08 follow-up — every right-side panel is now its own
        # ``QDockWidget`` tabified together by default. Users can drag
        # any panel out, dock it elsewhere, close it from the title
        # bar, and the arrangement persists via ``window_key``.
        # ``_FIRST_DOCK`` anchors the tabify chain so subsequent calls
        # land next to it instead of fanning out across dock areas.
        from .ui._dock_registry import DockSpec
        # Right-side tool group. ``_FIRST_DOCK`` anchors the tabify
        # chain so the rest land as tabs next to it. View is the
        # anchor now that the Browser is retired (Outline + Labels
        # supersede it); Selection is no longer here — it lives in the
        # left column under the Outline (see below).
        _FIRST_DOCK = "dock_model_view"

        def _add_panel(dock_id: str, title: str, widget) -> Any:
            return win.add_extension_dock(DockSpec(
                dock_id=dock_id,
                title=title,
                factory=lambda _p: widget,
                tabify_with=(
                    None if dock_id == _FIRST_DOCK else _FIRST_DOCK
                ),
            ))

        _add_panel(_FIRST_DOCK, "View", view_tab.widget)
        _add_panel("dock_model_filter", "Filter", filter_tab.widget)

        plotter = win.plotter

        # ── Build scene ─────────────────────────────────────────────
        _verbose = getattr(self._parent, '_verbose', False)
        registry = build_brep_scene(
            plotter, self._dims,
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
            verbose=_verbose,
        )
        self._registry = registry

        def _compute_model_diagonal() -> float:
            try:
                bb = gmsh.model.getBoundingBox(-1, -1)
                return float(np.linalg.norm(
                    [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
                )) or 1.0
            except Exception:
                return 1.0

        from .overlays.origin_markers_overlay import OriginMarkerOverlay
        from .ui.origin_markers_panel import OriginMarkersPanel
        from .ui.preferences_manager import PREFERENCES as _PREF
        _marker_size = _PREF.current.origin_marker_size
        origin_overlay = OriginMarkerOverlay(
            plotter,
            origin_shift=registry.origin_shift,
            model_diagonal=_compute_model_diagonal(),
            points=self._origin_markers,
            show_coords=self._origin_marker_show_coords,
            size=_marker_size,
        )
        origin_panel = OriginMarkersPanel(
            initial_points=self._origin_markers,
            initial_visible=True,
            initial_show_coords=self._origin_marker_show_coords,
            initial_size=_marker_size,
            on_visible_changed=origin_overlay.set_visible,
            on_show_coords_changed=origin_overlay.set_show_coords,
            on_marker_added=origin_overlay.add,
            on_marker_removed=origin_overlay.remove,
            on_size_changed=origin_overlay.set_size,
        )
        _add_panel("dock_model_markers", "Markers", origin_panel.widget)

        # ── Model info panel (read-only diagnostics) ──────────────
        # No longer a dock tab — surfaced via the top-level "Info"
        # menu as a standalone non-modal window (wired further down,
        # once ``win`` + the menu bar are available).
        from .ui._model_info_panel import ModelInfoPanel
        info_panel = ModelInfoPanel(parts_registry=getattr(self._parent, 'parts', None))

        # ── Section / clipping plane ────────────────────────────────
        from .overlays.clip_plane_overlay import ClipPlaneOverlay
        from .ui._clip_plane_panel import ClipPlanePanel
        clip_overlay = ClipPlaneOverlay(
            plotter, registry, origin_shift=registry.origin_shift,
        )

        def _world_bbox() -> tuple[float, float, float, float, float, float]:
            try:
                return tuple(gmsh.model.getBoundingBox(-1, -1))
            except Exception:
                return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        clip_panel = ClipPlanePanel(clip_overlay, world_bbox=_world_bbox())
        _add_panel("dock_model_section", "Section", clip_panel.widget)

        # ── Measure tool (entity-centroid distance) ─────────────────
        from .overlays.measure_overlay import MeasureOverlay
        from .ui._measure_panel import MeasurePanel
        measure_overlay = MeasureOverlay(plotter, registry)

        def _push_measure_status() -> None:
            measure_panel.update_status(
                num_points=measure_overlay.num_points,
                endpoints=measure_overlay.last_endpoints,
                distance=measure_overlay.last_distance,
                delta=measure_overlay.last_delta,
            )

        def _on_measure_active(active: bool) -> None:
            # Leaving measure mode wipes any in-flight measurement so
            # the next time the user enters they start fresh.
            if not active:
                measure_overlay.reset()
            _push_measure_status()
            win.set_status(
                "Measure mode ON — click two entities" if active
                else "Measure mode off",
                3000,
            )

        def _on_measure_clear() -> None:
            measure_overlay.reset()
            _push_measure_status()

        measure_panel = MeasurePanel(
            on_active_changed=_on_measure_active,
            on_clear=_on_measure_clear,
        )
        _add_panel("dock_model_measure", "Measure", measure_panel.widget)

        # ── Tangent / normal overlay (geometry probes in View tab) ──
        from .overlays.tangent_normal_overlay import TangentNormalOverlay
        tn_overlay = TangentNormalOverlay(
            plotter,
            origin_shift=registry.origin_shift,
            model_diagonal=_compute_model_diagonal(),
            scale=_PREF.current.tangent_normal_scale,
        )

        # ── Preferences (created AFTER scene — needs registry) ─────
        from .overlays.pref_helpers import make_line_width_cb, make_opacity_cb, make_edges_cb
        from .overlays.glyph_helpers import rebuild_brep_point_glyphs

        def _pref_point_size(v: float):
            kw = registry._add_mesh_kwargs.get(0, {})
            kw['point_size'] = v
            registry._add_mesh_kwargs[0] = kw
            rebuild_brep_point_glyphs(plotter, registry)
            plotter.render()

        _pref_line_width = make_line_width_cb(registry, plotter)
        _pref_opacity = make_opacity_cb(registry, plotter)
        _pref_edges = make_edges_cb(registry, plotter)

        def _pref_pick_color(hex_str: str):
            h = hex_str.lstrip("#")
            try:
                rgb = np.array(
                    [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
                    dtype=np.uint8,
                )
            except ValueError:
                return
            color_mgr.set_pick_color(rgb)
            color_mgr.recolor_all(
                picks=set(sel._picks),
                hidden=vis_mgr.hidden,
                hover=pick_engine.hover_entity,
            )
            plotter.render()

        from .ui.theme import THEME
        prefs = PreferencesTab(
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
            on_point_size=_pref_point_size,
            on_line_width=_pref_line_width,
            on_opacity=_pref_opacity,
            on_edges=_pref_edges,
            on_pick_color=_pref_pick_color,
            on_theme=lambda name: THEME.set_theme(name),
        )
        # Session tab (formerly "Preferences") — runtime tweaks that reset
        # next session. The "Global preferences…" button at the bottom opens
        # the persistent-prefs dialog.
        from qtpy import QtWidgets as _QtW
        from .ui.preferences_dialog import open_preferences_dialog
        from .ui.theme_editor_dialog import open_theme_editor
        _btn_global = _QtW.QPushButton("Global preferences…")
        _btn_global.clicked.connect(
            lambda: open_preferences_dialog(win.window)
        )
        prefs.widget.layout().addWidget(_btn_global)
        _btn_theme = _QtW.QPushButton("Theme editor…")
        _btn_theme.clicked.connect(
            lambda: open_theme_editor(win.window)
        )
        prefs.widget.layout().addWidget(_btn_theme)
        # Wrap in a scroll area so the (tall) Session panel never
        # forces a minimum size on the shared right-side tab group —
        # it scrolls instead of stretching its neighbours.
        _sess_scroll = _QtW.QScrollArea()
        _sess_scroll.setWidgetResizable(True)
        _sess_scroll.setFrameShape(_QtW.QFrame.NoFrame)
        _sess_scroll.setWidget(prefs.widget)
        _add_panel("dock_model_session", "Session", _sess_scroll)

        # Set generous clipping range for shifted coords
        try:
            plotter.reset_camera()
            cam = plotter.renderer.GetActiveCamera()
            cam.SetClippingRange(0.01, 1e6)
        except Exception:
            pass

        # ── Core modules ────────────────────────────────────────────
        color_mgr = ColorManager(registry)
        vis_mgr = VisibilityManager(registry, color_mgr, sel, plotter, verbose=_verbose)
        from .ui.preferences_manager import PREFERENCES as _PREF_DT
        pick_engine = PickEngine(
            plotter, registry, drag_threshold=_PREF_DT.current.drag_threshold,
        )

        # ── Left column — primary navigation ───────────────────────
        # The outline (Physical Groups / Labels / Parts, ParaView-
        # style) is the model navigator; the Browser panel it once
        # mirrored has been retired. Selection sits directly below it
        # (vertical split) so picks stay visible while you browse.
        parts_reg = getattr(self._parent, 'parts', None)
        from .ui._model_outline_tree import ModelOutlineTree

        outline = ModelOutlineTree(
            selection=sel,
            vis_mgr=vis_mgr,
            parts_registry=parts_reg,
            on_group_activated=_on_group_activated,
            on_entity_toggled=lambda dt: sel.toggle(dt),
            on_new_group=_on_new_group,
            on_new_label=_on_new_label,
            on_rename_label=_on_rename_label,
            on_delete_label=_on_delete_label,
            on_rename_group=_on_rename_group,
            on_delete_group=_on_delete_group,
        )
        outline_dock = win.add_extension_dock(DockSpec(
            dock_id="dock_model_outline",
            title="Outline",
            factory=lambda _p: outline.widget,
            default_area="left",
        ))
        sel_dock = win.add_extension_dock(DockSpec(
            dock_id="dock_model_selection",
            title="Selection",
            factory=lambda _p: sel_tree.widget,
            default_area="left",
        ))
        # Stack Selection under the Outline in the left column.
        from qtpy import QtCore as _QtC_split
        win.window.splitDockWidget(
            outline_dock, sel_dock, _QtC_split.Qt.Vertical,
        )

        # ── Info menu — model diagnostics as a standalone window ────
        # Replaces the old "Info" dock tab. Lazily builds one
        # non-modal window the first time it's opened; reuses it
        # afterwards. Parented to the main window so it closes with
        # the viewer but never blocks it.
        from qtpy import QtWidgets as _QtW_info, QtCore as _QtC_info
        _model_name = getattr(self._parent, "model_name", None) or "model"
        _info_window: list[Any] = []

        def _open_model_info() -> None:
            w = _info_window[0] if _info_window else None
            if w is None:
                w = _QtW_info.QMainWindow(win.window)
                w.setWindowFlag(_QtC_info.Qt.Window, True)
                w.setWindowTitle(f"Model info — {_model_name}")
                w.setCentralWidget(info_panel.widget)
                w.resize(420, 620)
                _info_window.append(w)
            info_panel.refresh()
            w.show()
            w.raise_()
            w.activateWindow()

        _info_menu = win.window.menuBar().addMenu("Info")
        _act_model_info = _info_menu.addAction("Model info…")
        _act_model_info.triggered.connect(_open_model_info)

        # ── File menu — CAD geometry import / export ────────────────
        # Import is additive: g.model.io.load_step / load_dxf add to
        # the current model, then the scene rebuilds. Export writes
        # the current model to STEP. Errors surface in a dialog (same
        # as the Boolean / Transform panels). Inserted leftmost so it
        # reads as a conventional File menu.
        from qtpy import QtWidgets as _QtW_file

        def _import_step() -> None:
            path, _f = _QtW_file.QFileDialog.getOpenFileName(
                win.window, "Import STEP", "",
                "STEP (*.step *.stp);;All files (*)",
            )
            if not path:
                return
            try:
                imported = self._model.io.load_step(path)
            except Exception as exc:
                _QtW_file.QMessageBox.warning(
                    win.window, "Import STEP", str(exc)
                )
                return
            n = sum(len(v) for v in (imported or {}).values())
            _rebuild_scene()
            win.set_status(
                f"Imported STEP — {n} entit"
                f"{'y' if n == 1 else 'ies'}"
            )

        def _import_dxf() -> None:
            path, _f = _QtW_file.QFileDialog.getOpenFileName(
                win.window, "Import DXF", "",
                "DXF (*.dxf);;All files (*)",
            )
            if not path:
                return
            try:
                self._model.io.load_dxf(path)
            except Exception as exc:
                _QtW_file.QMessageBox.warning(
                    win.window, "Import DXF", str(exc)
                )
                return
            _rebuild_scene()
            win.set_status("Imported DXF")

        def _export_step() -> None:
            path, _f = _QtW_file.QFileDialog.getSaveFileName(
                win.window, "Export STEP", "",
                "STEP (*.step);;All files (*)",
            )
            if not path:
                return
            try:
                self._model.io.save_step(path)
            except Exception as exc:
                _QtW_file.QMessageBox.warning(
                    win.window, "Export STEP", str(exc)
                )
                return
            win.set_status("Exported STEP")

        _file_menu = _QtW_file.QMenu("File", win.window)
        _file_menu.addAction("Import STEP…").triggered.connect(
            _import_step
        )
        _file_menu.addAction("Import DXF…").triggered.connect(
            _import_dxf
        )
        _file_menu.addSeparator()
        _file_menu.addAction("Export STEP…").triggered.connect(
            _export_step
        )
        _mb = win.window.menuBar()
        _mb_acts = _mb.actions()
        if _mb_acts:
            _mb.insertMenu(_mb_acts[0], _file_menu)   # File leftmost
        else:
            _mb.addMenu(_file_menu)

        # Scene rebuild after any geometry mutation (parts fuse,
        # boolean ops, transforms). Hoisted to show() scope so it
        # exists even without a parts registry.
        def _rebuild_scene():
            """Tear down VTK actors and rebuild from current Gmsh state.

            Mutates ``registry`` in-place so all closures over it
            (color_mgr, vis_mgr, pick_engine) keep working.
            """
            # Save camera state
            cam = plotter.renderer.GetActiveCamera()
            cam_pos = cam.GetPosition()
            cam_fp = cam.GetFocalPoint()
            cam_up = cam.GetViewUp()
            cam_clip = cam.GetClippingRange()

            # Remove old actors
            for actor in list(registry.dim_actors.values()):
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass

            # Build fresh scene
            fresh = build_brep_scene(
                plotter, self._dims,
                point_size=self._point_size,
                line_width=self._line_width,
                surface_opacity=self._surface_opacity,
                show_surface_edges=self._show_surface_edges,
                verbose=_verbose,
            )

            # Mutate existing registry in place — preserves closures
            for slot in registry.__slots__:
                setattr(registry, slot, getattr(fresh, slot))

            # Re-sync origin markers with the fresh registry's shift
            origin_overlay.set_origin_shift(registry.origin_shift)
            tn_overlay.set_model_diagonal(_compute_model_diagonal())
            tn_overlay.set_origin_shift(registry.origin_shift)

            # Clear stale selection / active group
            sel.clear()

            # Refresh UI panels
            if parts_tree is not None:
                parts_tree.refresh()
            outline.refresh()
            sel_tree.update(sel.picks)
            info_panel.refresh()

            # Re-bind the clip plane to the fresh mappers + new bbox
            clip_overlay.set_origin_shift(registry.origin_shift)
            clip_overlay.rebind()
            clip_panel.refresh_bbox(_world_bbox())

            # Stored centroids are stale after a rebuild
            measure_overlay.reset()
            _push_measure_status()

            # Restore camera
            cam.SetPosition(*cam_pos)
            cam.SetFocalPoint(*cam_fp)
            cam.SetViewUp(*cam_up)
            cam.SetClippingRange(*cam_clip)
            plotter.render()

        parts_tree = None
        if parts_reg is not None:
            def _parts_select_only(dts):
                sel.select_batch(dts, replace=True)

            def _parts_add(dts):
                sel.select_batch(dts)

            def _parts_remove(dts):
                sel.box_remove(dts)

            def _parts_isolate(dts):
                sel.select_batch(dts, replace=True)
                vis_mgr.isolate()
                plotter.render()

            def _parts_hide(dts):
                sel.select_batch(dts, replace=True)
                vis_mgr.hide()
                plotter.render()

            def _parts_new(label, picks):
                from qtpy.QtWidgets import QMessageBox
                try:
                    parts_reg.register(label, picks)
                except ValueError as e:
                    QMessageBox.warning(win.window, "Ownership conflict", str(e))
                    return
                parts_tree.refresh()

            def _parts_rename(old_label, new_label):
                from qtpy.QtWidgets import QMessageBox
                try:
                    parts_reg.rename(old_label, new_label)
                except (KeyError, ValueError) as e:
                    QMessageBox.warning(win.window, "Rename failed", str(e))
                    return
                parts_tree.refresh()

            def _parts_delete(label):
                parts_reg.delete(label)
                parts_tree.refresh()

            def _parts_fuse(labels, new_label):
                from qtpy.QtWidgets import QMessageBox
                try:
                    parts_reg.fuse_group(labels, label=new_label)
                except (ValueError, RuntimeError) as e:
                    QMessageBox.warning(win.window, "Fuse failed", str(e))
                    return
                _rebuild_scene()

            parts_tree = PartsTreePanel(
                parts_reg, registry,
                on_select_only=_parts_select_only,
                on_add_to_selection=_parts_add,
                on_remove_from_selection=_parts_remove,
                on_isolate=_parts_isolate,
                on_hide=_parts_hide,
                on_new_part=_parts_new,
                on_rename_part=_parts_rename,
                on_delete_part=_parts_delete,
                on_fuse_parts=_parts_fuse,
                get_current_picks=lambda: sel.picks,
            )
            # Insert after Browser tab (position 1)
            win._tab_widget.insertTab(1, parts_tree.widget, "Parts")

        # ── Wire callbacks ──────────────────────────────────────────

        # Pick -> selection (or measure overlay when measure mode is on)
        def _on_pick(dt: DimTag, ctrl: bool):
            if measure_panel.is_active():
                measure_overlay.add_entity(dt)
                _push_measure_status()
                return
            if ctrl:
                sel.unpick(dt)
            else:
                sel.toggle(dt)

        pick_engine.on_pick = _on_pick
        pick_engine.set_hidden_check(vis_mgr.is_hidden)

        # Hover -> color
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

        # Selection changed -> batch recolor + refresh UI
        def _on_sel_changed():
            color_mgr.recolor_all(
                picks=set(sel._picks),
                hidden=vis_mgr.hidden,
                hover=pick_engine.hover_entity,
            )
            plotter.render()
            n = len(sel.picks)
            grp = sel.active_group or "none"
            win.set_status(f"{n} picked | group: {grp}")

        sel.on_changed.append(_on_sel_changed)
        # Repaint idle colors when the theme palette changes
        win.on_theme_changed(lambda _p: _on_sel_changed())
        win.on_theme_changed(lambda _p: tn_overlay.refresh_theme())
        sel.on_changed.append(lambda: sel_tree.update(sel.picks))
        sel.on_changed.append(lambda: outline.update_active())
        if parts_tree is not None:
            sel.on_changed.append(
                lambda: parts_tree.highlight_part_for_entity(sel.picks[-1])
                if sel.picks else None
            )
        # Write active group to Gmsh on every pick change
        sel.on_changed.append(lambda: sel.commit_active_group())
        # Plan 04 step 4 — selection bridge into ActiveObjects.
        # Same pattern as mesh.viewer: emit a fresh tuple of picks on
        # every mutation so ``ActiveObjects``' identity short-circuit
        # doesn't suppress in-place changes. Subscribers reach for
        # ``viewer._active.selection`` (a tuple snapshot) or, for
        # richer state, hold a viewer reference and inspect
        # ``viewer._selection_state``.
        _active_ref = self._active
        sel.on_changed.append(
            lambda: _active_ref.set_selection(tuple(sel.picks)),
        )

        # Visibility changed -> render
        vis_mgr.on_changed.append(lambda: plotter.render())

        # Box select
        def _on_box(dts: list[DimTag], ctrl: bool):
            if ctrl:
                n = sel.box_remove(dts)
                verb = "removed"
            else:
                n = sel.box_add(dts)
                verb = "added"
            if n:
                noun = "entity" if n == 1 else "entities"
                win.set_status(f"Box select: {verb} {n} {noun}", 2000)
            else:
                win.set_status("Box select: 0 entities", 2000)

        pick_engine.on_box_select = _on_box

        # ── Boolean / Transform panels (live OCC editing) ───────────
        # Pure-UI panels; these callbacks own the library call +
        # _rebuild_scene (mirrors _parts_fuse). The selection feeds
        # operands; OCC renumbers after each op, so captured operands
        # are dropped and the rebuild clears the selection.
        import math as _math
        from .ui._boolean_panel import BooleanPanel
        from .ui._transform_panel import TransformPanel

        def _on_boolean(op, objects, tools, opts):
            from qtpy import QtWidgets
            if not objects:
                _boolean_panel.set_hint(
                    "Set the Objects slot from a selection first."
                )
                return
            if op in ("fuse", "cut", "intersect") and not tools:
                _boolean_panel.set_hint(
                    f"{op} needs both Objects and Tools."
                )
                return
            bx = self._model.boolean
            try:
                if op == "fragment":
                    res = bx.fragment(
                        objects, tools,
                        remove_object=opts["remove_object"],
                        remove_tool=opts["remove_tool"],
                        cleanup_free=opts["cleanup_free"],
                    )
                else:
                    kw = dict(
                        remove_object=opts["remove_object"],
                        remove_tool=opts["remove_tool"],
                    )
                    if opts["label"]:
                        kw["label"] = opts["label"]
                    res = getattr(bx, op)(objects, tools, **kw)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    win.window, f"Boolean: {op}", str(exc)
                )
                _boolean_panel.set_hint(f"{op} failed: {exc}")
                return
            _boolean_panel.clear_operands()
            _rebuild_scene()
            n = len(res) if res else 0
            _boolean_panel.set_hint(f"{op} OK → {n} result(s)")
            win.set_status(f"Boolean {op}: {n} result(s)")

        def _on_transform(op, params, duplicate):
            from qtpy import QtWidgets
            tags = list(sel.picks)
            tx = self._model.transforms
            geo = self._model.geometry
            if op != "thru_sections" and not tags:
                _transform_panel.set_hint("Select entities first.")
                return
            try:
                if op in ("translate", "rotate", "scale", "mirror"):
                    if duplicate:
                        dims = {d for d, _ in tags}
                        if len(dims) != 1:
                            _transform_panel.set_hint(
                                "'Keep original' needs a single-"
                                "dimension selection."
                            )
                            return
                        dim0 = dims.pop()
                        target = [(dim0, t) for t in tx.copy(tags)]
                    else:
                        target = tags
                    if op == "translate":
                        tx.translate(target, params["dx"],
                                     params["dy"], params["dz"])
                    elif op == "rotate":
                        tx.rotate(
                            target, _math.radians(params["angle"]),
                            ax=params["ax"], ay=params["ay"],
                            az=params["az"], cx=params["cx"],
                            cy=params["cy"], cz=params["cz"],
                        )
                    elif op == "scale":
                        tx.scale(
                            target, params["sx"], params["sy"],
                            params["sz"], cx=params["cx"],
                            cy=params["cy"], cz=params["cz"],
                        )
                    else:  # mirror
                        tx.mirror(target, params["a"], params["b"],
                                  params["c"], params["d"])
                elif op == "copy":
                    tx.copy(tags)
                elif op == "extrude":
                    ne = [params["layers"]] if params["layers"] else None
                    tx.extrude(tags, params["dx"], params["dy"],
                               params["dz"], num_elements=ne,
                               recombine=params["recombine"])
                elif op == "revolve":
                    ne = [params["layers"]] if params["layers"] else None
                    tx.revolve(
                        tags, _math.radians(params["angle"]),
                        x=params["x"], y=params["y"], z=params["z"],
                        ax=params["ax"], ay=params["ay"],
                        az=params["az"], num_elements=ne,
                        recombine=params["recombine"],
                    )
                elif op == "sweep":
                    pc = params.get("path_curves") or []
                    if not pc:
                        _transform_panel.set_hint(
                            "Set the sweep path from selected curves."
                        )
                        return
                    wire = geo.add_wire(pc)
                    tx.sweep(tags, wire, trihedron=params["trihedron"])
                elif op == "thru_sections":
                    secs = params.get("sections") or []
                    if len(secs) < 2:
                        _transform_panel.set_hint(
                            "Add at least 2 sections."
                        )
                        return
                    wires = [geo.add_wire(c) for c in secs]
                    tx.thru_sections(
                        wires, make_solid=params["make_solid"],
                        make_ruled=params["make_ruled"],
                    )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    win.window, f"Transform: {op}", str(exc)
                )
                _transform_panel.set_hint(f"{op} failed: {exc}")
                return
            _transform_panel.reset_captures()
            _rebuild_scene()
            _transform_panel.set_hint(f"{op} OK")
            win.set_status(f"Transform {op} applied")

        _boolean_panel = BooleanPanel(
            get_selection=lambda: list(sel.picks),
            on_apply=_on_boolean,
        )
        _transform_panel = TransformPanel(
            get_selection=lambda: list(sel.picks),
            on_apply=_on_transform,
        )
        _add_panel("dock_model_boolean", "Boolean", _boolean_panel.widget)
        _add_panel(
            "dock_model_transform", "Transform", _transform_panel.widget
        )

        # ── Navigation ──────────────────────────────────────────────
        install_navigation(
            plotter,
            get_orbit_pivot=lambda: sel.centroid(registry),
        )

        # ── Motion LOD ──────────────────────────────────────────────
        # The per-dim silhouette actors are ``vtkPolyDataSilhouette`` —
        # view-dependent, so they re-execute every frame the camera
        # moves (the dominant per-orbit cost on a complex CAD part,
        # on top of what the navigation bounds-cache already removes).
        # Hide them during any camera gesture and restore ~120 ms
        # after it settles — same interactive-LOD trick mesh.viewer
        # uses for its node cloud. The lambda is re-evaluated per
        # gesture so it always targets the live silhouette actors
        # (they're rebuilt by the visibility hide/show path).
        from .core.motion_lod import MotionLOD
        self._motion_lod = MotionLOD(
            plotter,
            lambda: list(registry.dim_silhouette_actors.values()),
        )
        self._motion_lod.install()

        # ── Install pick engine ─────────────────────────────────────
        pick_engine.install()

        # ── Visibility action helpers (shared between toolbar + keys) ──
        def _act_hide() -> None:
            vis_mgr.hide()
            plotter.render()

        def _act_isolate() -> None:
            vis_mgr.isolate()
            plotter.render()

        def _act_reveal_all() -> None:
            vis_mgr.reveal_all()
            plotter.render()

        # ── Toolbar buttons for visibility ──────────────────────────
        win.add_toolbar_separator()
        win.add_toolbar_button("Hide selected (H)", "H", _act_hide)
        win.add_toolbar_button("Isolate selected (I)", "I", _act_isolate)
        win.add_toolbar_button("Reveal all (R)", "R", _act_reveal_all)

        # ── Keybindings ─────────────────────────────────────────────
        # VTK-level (only when 3D viewport has focus)
        plotter.add_key_event("h", _act_hide)
        plotter.add_key_event("i", _act_isolate)
        plotter.add_key_event("r", _act_reveal_all)
        plotter.add_key_event("u", lambda: sel.undo())

        # Dim filters: 0=points, 1=curves, 2=surfaces, 3=volumes
        for key, dim_set in [
            ("0", {0}), ("1", {1}), ("2", {2}), ("3", {3}),
        ]:
            plotter.add_key_event(
                key,
                lambda ds=dim_set: _on_filter(ds),
            )
        # 4 = all dims
        plotter.add_key_event(
            "4", lambda: _on_filter(set(self._dims)),
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
        from apeGmsh.viz.Selection import Selection
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
