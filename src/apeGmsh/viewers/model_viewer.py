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
    from apeGmsh.mesh.FEMData import FEMData
    from .core.entity_registry import EntityRegistry
    from .core.selection import SelectionState


class ModelViewer:
    """Interactive BRep model viewer with physical group management.

    Displays BRep geometry, parts, physical groups, and optional
    load/mass overlays when a resolved FEMData snapshot is provided.

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
    fast : bool
        Ignored (always fast). Kept for backward compatibility.
    fem : FEMData, optional
        Resolved FEM snapshot from :meth:`Mesh.get_fem_data`.  Required
        to enable the **Loads** and **Mass** tabs' 3-D glyph overlays.
        See :meth:`Model.viewer` for the rationale behind this design
        choice — in short, snapshot semantics ensure the viewer shows
        exactly what was resolved, never drifting out of sync with
        the live session state.  Without ``fem``, the Loads/Mass tabs
        still display the definition list but the overlays are
        disabled with an amber warning.
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
        fem: "FEMData | None" = None,
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

        # Optional resolved FEM data for loads/mass overlays
        self._fem: "FEMData | None" = fem

        # Populated during show()
        self._selection_state: "SelectionState | None" = None
        self._registry: "EntityRegistry | None" = None

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
            BrowserTab, FilterTab, ViewTab, SelectionTreePanel, PartsTreePanel,
        )
        from .ui.loads_tab import LoadsTabPanel, pattern_color
        from .ui.mass_tab import MassTabPanel
        from .ui.constraints_tab import (
            ConstraintsTabPanel, constraint_color,
            NODE_PAIR_KINDS, SURFACE_KINDS,
        )
        import pyvista as pv

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
        # Preference callbacks — update live actors AND stored kwargs
        # (so that VisibilityManager recreates actors with current settings)
        def _pref_point_size(v: float):
            for dim, actor in registry.dim_actors.items():
                if dim == 0:
                    actor.GetProperty().SetPointSize(v)
                    # Also update stored kwargs for actor recreation
                    kw = registry._add_mesh_kwargs.get(dim, {})
                    kw['point_size'] = v
                    registry._add_mesh_kwargs[dim] = kw
            plotter.render()

        def _pref_line_width(v: float):
            for dim, actor in registry.dim_actors.items():
                if dim == 1:
                    actor.GetProperty().SetLineWidth(v)
                    kw = registry._add_mesh_kwargs.get(dim, {})
                    kw['line_width'] = v
                    registry._add_mesh_kwargs[dim] = kw
            plotter.render()

        def _pref_opacity(v: float):
            for dim, actor in registry.dim_actors.items():
                if dim >= 2:
                    actor.GetProperty().SetOpacity(v)
                    kw = registry._add_mesh_kwargs.get(dim, {})
                    kw['opacity'] = v
                    registry._add_mesh_kwargs[dim] = kw
            plotter.render()

        def _pref_edges(show: bool):
            for dim, actor in registry.dim_actors.items():
                if dim >= 2:
                    actor.GetProperty().SetEdgeVisibility(show)
                    kw = registry._add_mesh_kwargs.get(dim, {})
                    kw['show_edges'] = show
                    registry._add_mesh_kwargs[dim] = kw
            plotter.render()

        def _pref_overlay_scale(key: str, mult: float):
            _overlay_scales[key] = mult
            # Re-trigger the relevant overlay with current state
            if key == 'load_arrow' and loads_tab is not None:
                _on_loads_patterns_changed(loads_tab.active_patterns())
            elif key == 'mass_sphere' and mass_tab is not None:
                # Check if mass overlay is currently shown
                show_cb = getattr(mass_tab, '_show_cb', None)
                if show_cb is not None:
                    _on_mass_overlay_changed(show_cb.isChecked())
            elif key.startswith('constraint') and constraints_tab is not None:
                _on_constraint_kinds_changed(constraints_tab.active_kinds())

        prefs = PreferencesTab(
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
            on_point_size=_pref_point_size,
            on_line_width=_pref_line_width,
            on_opacity=_pref_opacity,
            on_edges=_pref_edges,
            on_overlay_scale=_pref_overlay_scale,
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
            # Qt6 uses QMessageBox.StandardButton.Yes; Qt5 had the
            # top-level alias. Compare via the enum member to stay
            # portable across PyQt5/PySide2/PyQt6/PySide6.
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
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
                            text_color="#a6e3a1",
                            shape_color="#1e1e2e",
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
                            text_color="#f9e2af",     # warm yellow
                            shape_color="#1e1e2e",
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

        view_tab = ViewTab(
            self._dims,
            on_labels_changed=_on_labels_changed,
        )

        # ── Selection tree panel ────────────────────────────────────
        def _tree_select_only(dts):
            sel.select_batch(dts, replace=True)

        def _tree_add(dts):
            sel.select_batch(dts)

        def _tree_remove(dts):
            sel.box_remove(dts)

        sel_tree = SelectionTreePanel(
            on_select_only=_tree_select_only,
            on_add_to_selection=_tree_add,
            on_remove_from_selection=_tree_remove,
        )

        # Add tabs to window
        win.add_tab("Browser", browser.widget)
        win.add_tab("View", view_tab.widget)
        win.add_tab("Filter", filter_tab.widget)
        win.add_tab("Preferences", prefs.widget)

        # Add selection tree as bottom dock
        win.add_right_bottom_dock("Selection", sel_tree.widget)

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
        pick_engine = PickEngine(plotter, registry)

        # ── Parts tree panel (needs registry) ──────────────────────
        parts_reg = getattr(self._parent, 'parts', None)
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

                # Clear stale selection / active group
                sel.clear()

                # Refresh UI panels
                if parts_tree is not None:
                    parts_tree.refresh()
                browser.refresh()
                sel_tree.update(sel.picks)

                # Restore camera
                cam.SetPosition(*cam_pos)
                cam.SetFocalPoint(*cam_fp)
                cam.SetViewUp(*cam_up)
                cam.SetClippingRange(*cam_clip)
                plotter.render()

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

        # ── Loads tab (read-only overlays) ──────────────────────────
        loads_comp = getattr(self._parent, 'loads', None)
        loads_tab = None
        _load_actors: list = []

        def _characteristic_length() -> float:
            """Geometric mean of significant bounding box spans.

            Better than diagonal for elongated/flat models — not
            dominated by the largest dimension.
            """
            try:
                dims_present = sorted(registry.dim_meshes.keys())
                if not dims_present:
                    return 1.0
                mesh = registry.dim_meshes[dims_present[-1]]
                pts = mesh.points
                if len(pts) == 0:
                    return 1.0
                span = pts.max(axis=0) - pts.min(axis=0)
                max_span = float(span.max())
                if max_span < 1e-12:
                    return 1.0
                # Keep only significant dims (> 1% of largest)
                sig = span[span > max_span * 0.01]
                return float(np.prod(sig) ** (1.0 / len(sig)))
            except Exception:
                return 1.0

        # ── Overlay scale multipliers (updated by Preferences) ─────
        _overlay_scales = {
            'load_arrow':         1.0,
            'mass_sphere':        1.0,
            'constraint_marker':  1.0,
            'constraint_line':    1.0,
        }

        def _on_loads_patterns_changed(active_patterns):
            for a in _load_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _load_actors.clear()

            fem = self._fem
            if not active_patterns or fem is None or not fem.nodes.loads:
                plotter.render()
                return

            char_len = _characteristic_length()
            base_len = char_len * 0.05 * _overlay_scales['load_arrow']
            origin = registry.origin_shift

            for pat in active_patterns:
                positions = []
                directions = []
                for r in fem.nodes.loads:
                    if r.pattern != pat:
                        continue
                    try:
                        xyz = fem.nodes.coords[fem.nodes.index(int(r.node_id))] - origin
                    except Exception:
                        continue
                    fxyz = np.array(r.forces[:3], dtype=float)
                    if not np.any(fxyz):
                        continue
                    positions.append(xyz)
                    directions.append(fxyz)

                if not positions:
                    continue

                positions_arr = np.array(positions, dtype=float)
                directions_arr = np.array(directions, dtype=float)

                cloud = pv.PolyData(positions_arr)
                cloud['vectors'] = directions_arr
                glyphs = cloud.glyph(
                    orient='vectors', scale=False, factor=base_len,
                )
                color = pattern_color(pat)
                actor = plotter.add_mesh(
                    glyphs, color=color,
                    name=f"_loads_pattern_{pat}",
                    reset_camera=False,
                    pickable=False,
                )
                _load_actors.append(actor)

            plotter.render()

        if loads_comp is not None:
            loads_tab = LoadsTabPanel(
                loads_comp, fem=self._fem,
                on_patterns_changed=_on_loads_patterns_changed,
            )
            # Insert after Parts (position 2 if parts exists, else 1)
            insert_pos = 2 if parts_reg is not None else 1
            win._tab_widget.insertTab(insert_pos, loads_tab.widget, "Loads")

        # ── Mass tab (read-only overlays) ───────────────────────────
        mass_comp = getattr(self._parent, 'masses', None)
        mass_tab = None
        _mass_actors: list = []

        def _on_mass_overlay_changed(show: bool):
            for a in _mass_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _mass_actors.clear()
            try:
                plotter.remove_scalar_bar()
            except Exception:
                pass

            fem = self._fem
            if not show or fem is None or not fem.nodes.masses:
                plotter.render()
                return

            positions = []
            masses = []
            origin = registry.origin_shift
            for r in fem.nodes.masses:
                try:
                    xyz = fem.nodes.coords[fem.nodes.index(int(r.node_id))] - origin
                except Exception:
                    continue
                m = float(r.mass[0])
                if m <= 0:
                    continue
                positions.append(xyz)
                masses.append(m)

            if not positions:
                plotter.render()
                return

            char_len = _characteristic_length()
            max_mass = max(masses) if masses else 1.0
            base_r = char_len * 0.005 * _overlay_scales['mass_sphere']

            cloud = pv.PolyData(np.array(positions, dtype=float))
            cloud['mass'] = np.array(masses, dtype=float)
            sphere = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
            scale_factor = base_r / (max_mass ** (1.0 / 3.0))
            glyphs = cloud.glyph(
                geom=sphere, scale='mass', factor=scale_factor,
            )

            actor = plotter.add_mesh(
                glyphs, scalars='mass', cmap='viridis',
                scalar_bar_args={'title': 'Nodal mass [kg]'},
                name="_mass_overlays",
                reset_camera=False,
                pickable=False,
            )
            _mass_actors.append(actor)
            plotter.render()

        if mass_comp is not None:
            mass_tab = MassTabPanel(
                mass_comp, fem=self._fem,
                on_overlay_changed=_on_mass_overlay_changed,
            )
            insert_pos = 3 if (parts_reg is not None and loads_tab is not None) \
                else (2 if loads_tab is not None or parts_reg is not None else 1)
            win._tab_widget.insertTab(insert_pos, mass_tab.widget, "Mass")

        # ── Constraints tab (read-only overlays) ───────────────────
        constraints_comp = getattr(self._parent, 'constraints', None)
        constraints_tab = None
        _constraint_actors: list = []

        def _on_constraint_kinds_changed(active_kinds: set[str]):
            # 1. Clear old actors
            for a in _constraint_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _constraint_actors.clear()

            fem = self._fem
            if not active_kinds or fem is None:
                plotter.render()
                return

            char_len = _characteristic_length()
            origin = registry.origin_shift
            marker_r = char_len * 0.003 * _overlay_scales['constraint_marker']

            def _node_xyz(nid: int):
                try:
                    return fem.nodes.coords[fem.nodes.index(int(nid))] - origin
                except (KeyError, IndexError):
                    return None

            # ── Node-pair kinds ────────────────────────────────
            node_pair_kinds = active_kinds & NODE_PAIR_KINDS
            if node_pair_kinds:
                for kind in node_pair_kinds:
                    line_pts: list = []
                    line_cells: list = []
                    master_positions: dict[int, np.ndarray] = {}
                    idx = 0

                    for rec in fem.nodes.constraints.node_pairs():
                        if rec.kind != kind:
                            continue
                        p1 = _node_xyz(rec.master_node)
                        p2 = _node_xyz(rec.slave_node)
                        if p1 is None or p2 is None:
                            continue
                        line_pts.extend([p1, p2])
                        line_cells.extend([2, idx, idx + 1])
                        idx += 2
                        if rec.master_node not in master_positions:
                            master_positions[rec.master_node] = p1

                    color = constraint_color(kind)

                    # Line segments (batched)
                    cst_lw = max(1, int(3 * _overlay_scales['constraint_line']))

                    if line_pts:
                        pts_arr = np.array(line_pts, dtype=float)
                        cells_arr = np.array(line_cells, dtype=np.int64)
                        poly = pv.PolyData(pts_arr, lines=cells_arr)
                        actor = plotter.add_mesh(
                            poly, color=color, line_width=cst_lw,
                            render_lines_as_tubes=True,
                            name=f"_cst_lines_{kind}",
                            reset_camera=False, pickable=False,
                        )
                        _constraint_actors.append(actor)

                    # Master node spheres
                    if master_positions:
                        cloud = pv.PolyData(
                            np.array(list(master_positions.values()),
                                     dtype=float))
                        sphere = pv.Sphere(
                            radius=marker_r,
                            theta_resolution=8, phi_resolution=8)
                        glyphs = cloud.glyph(
                            geom=sphere, orient=False, scale=False)
                        actor = plotter.add_mesh(
                            glyphs, color=color, smooth_shading=True,
                            name=f"_cst_masters_{kind}",
                            reset_camera=False, pickable=False,
                        )
                        _constraint_actors.append(actor)

                    # Phantom nodes for node_to_surface
                    if kind == "node_to_surface":
                        phantom_pts = []
                        for nid, xyz in fem.nodes.constraints.extra_nodes():
                            shifted = np.array(xyz, dtype=float) - origin
                            phantom_pts.append(shifted)
                        if phantom_pts:
                            cloud = pv.PolyData(
                                np.array(phantom_pts, dtype=float))
                            diamond = pv.Octahedron(radius=marker_r * 0.7)
                            glyphs = cloud.glyph(
                                geom=diamond, orient=False, scale=False)
                            actor = plotter.add_mesh(
                                glyphs, color=color, smooth_shading=True,
                                name=f"_cst_phantoms_{kind}",
                                reset_camera=False, pickable=False,
                            )
                            _constraint_actors.append(actor)

            # ── Surface kinds ──────────────────────────────────
            surface_kinds = active_kinds & SURFACE_KINDS
            if surface_kinds:
                for kind in surface_kinds:
                    interp_pts: list = []
                    interp_cells: list = []
                    idx = 0

                    for rec in fem.elements.constraints.interpolations():
                        if rec.kind != kind:
                            continue
                        slave_pt = _node_xyz(rec.slave_node)
                        if slave_pt is None:
                            continue
                        master_pts = []
                        for mnid in rec.master_nodes:
                            mp = _node_xyz(mnid)
                            if mp is not None:
                                master_pts.append(mp)
                        if not master_pts:
                            continue
                        weights = rec.weights
                        if (weights is not None
                                and len(weights) == len(master_pts)):
                            centroid = np.average(
                                master_pts, axis=0, weights=weights)
                        else:
                            centroid = np.mean(master_pts, axis=0)
                        interp_pts.extend([slave_pt, centroid])
                        interp_cells.extend([2, idx, idx + 1])
                        idx += 2

                    color = constraint_color(kind)

                    # Interpolation lines
                    cst_lw = max(1, int(2 * _overlay_scales['constraint_line']))

                    if interp_pts:
                        pts_arr = np.array(interp_pts, dtype=float)
                        cells_arr = np.array(interp_cells, dtype=np.int64)
                        poly = pv.PolyData(pts_arr, lines=cells_arr)
                        actor = plotter.add_mesh(
                            poly, color=color, line_width=cst_lw,
                            render_lines_as_tubes=True, opacity=0.7,
                            name=f"_cst_interp_{kind}",
                            reset_camera=False, pickable=False,
                        )
                        _constraint_actors.append(actor)

                    # Surface coupling highlights
                    for coup in fem.elements.constraints.couplings():
                        if coup.kind != kind:
                            continue
                        for node_set, suffix, opac in [
                            (coup.master_nodes, "master", 0.25),
                            (coup.slave_nodes, "slave", 0.25),
                        ]:
                            face_pts = []
                            for nid in node_set:
                                pt = _node_xyz(nid)
                                if pt is not None:
                                    face_pts.append(pt)
                            if len(face_pts) >= 3:
                                cloud = pv.PolyData(
                                    np.array(face_pts, dtype=float))
                                try:
                                    surf = cloud.delaunay_2d()
                                    actor = plotter.add_mesh(
                                        surf, color=color, opacity=opac,
                                        name=(f"_cst_surf_{kind}_{suffix}"
                                              f"_{id(coup)}"),
                                        reset_camera=False, pickable=False,
                                    )
                                    _constraint_actors.append(actor)
                                except Exception:
                                    pass

            plotter.render()

        if constraints_comp is not None:
            constraints_tab = ConstraintsTabPanel(
                constraints_comp, fem=self._fem,
                on_kinds_changed=_on_constraint_kinds_changed,
            )
            n_cond = sum([
                parts_reg is not None,
                loads_tab is not None,
                mass_tab is not None,
            ])
            win._tab_widget.insertTab(
                1 + n_cond, constraints_tab.widget, "Constraints")

        # ── Wire callbacks ──────────────────────────────────────────

        # Pick -> selection
        def _on_pick(dt: DimTag, ctrl: bool):
            if ctrl:
                sel.unpick(dt)
            else:
                sel.toggle(dt)

        pick_engine.on_pick = _on_pick
        pick_engine.set_hidden_check(vis_mgr.is_hidden)

        # Filter -> pick engine + visual dim feedback
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
        sel.on_changed.append(lambda: sel_tree.update(sel.picks))
        sel.on_changed.append(lambda: browser.update_active())
        if parts_tree is not None:
            sel.on_changed.append(
                lambda: parts_tree.highlight_part_for_entity(sel.picks[-1])
                if sel.picks else None
            )
        # Write active group to Gmsh on every pick change
        sel.on_changed.append(lambda: sel.commit_active_group())

        # Visibility changed -> render
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
        # 9 = all dims
        plotter.add_key_event(
            "9", lambda: _on_filter(set(self._dims)),
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
