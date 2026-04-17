"""
MeshViewer — Interactive mesh viewer with overlay support.

Assembled from independent core modules — no inheritance.
Uses the same core (navigation, pick engine, color manager) as
ModelViewer but with mesh-specific scene builder and UI tabs.

Loads, constraints, and masses are displayed here (not in the model
viewer) because they are mesh-resolved concepts.

Usage::

    viewer = MeshViewer(parent)
    viewer.show()
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

from apeGmsh._types import DimTag

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _SessionBase
    from apeGmsh.mesh.FEMData import FEMData
    from .core.selection import SelectionState
    from .scene.mesh_scene import MeshSceneData


class MeshViewer:
    """Interactive mesh viewer with element/node picking.

    Displays mesh elements and nodes with optional load, constraint,
    and mass overlays.  Overlay data comes from a resolved ``FEMData``
    snapshot — either passed explicitly or auto-resolved from the
    session at show time.

    Parameters
    ----------
    parent : _SessionBase
        The apeGmsh session.
    dims : list[int], optional
        Which mesh dimensions to show (default: ``[1, 2, 3]``).
    point_size, line_width, surface_opacity, show_surface_edges
        Visual properties.
    fem : FEMData, optional
        Pre-resolved FEM snapshot.  If not provided, the viewer calls
        ``get_fem_data()`` automatically when the window opens.
    fast : bool
        Ignored (always fast). Kept for backward compatibility.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        *,
        dims: list[int] | None = None,
        point_size: float = 6.0,
        line_width: float = 3.0,
        surface_opacity: float = 1.0,
        show_surface_edges: bool = True,
        fem: "FEMData | None" = None,
        fast: bool = True,
        **kwargs: Any,
    ) -> None:
        self._parent = parent
        self._dims = dims if dims is not None else [1, 2, 3]

        self._point_size = point_size
        self._line_width = line_width
        self._surface_opacity = surface_opacity
        self._show_surface_edges = show_surface_edges
        self._fem: "FEMData | None" = fem

        # Populated during show()
        self._selection_state: "SelectionState | None" = None
        self._scene_data: "MeshSceneData | None" = None

    def show(self, *, title: str | None = None, maximized: bool = True):
        """Open the viewer window, block until closed."""
        from .core.navigation import install_navigation
        from .core.color_manager import ColorManager
        from .core.pick_engine import PickEngine
        from .core.visibility import VisibilityManager
        from .core.selection import SelectionState
        from .scene.mesh_scene import build_mesh_scene
        from .ui.viewer_window import ViewerWindow
        from .ui.preferences import PreferencesTab
        from .ui.mesh_tabs import MeshInfoTab, DisplayTab, MeshFilterTab

        gmsh.model.occ.synchronize()

        # ── Auto-filter requested dims to those with actual elements ──
        meshed_dims: set[int] = set()
        try:
            types_all, _, _ = gmsh.model.mesh.getElements(dim=-1, tag=-1)
            for etype in types_all:
                try:
                    _, edim, *_ = gmsh.model.mesh.getElementProperties(int(etype))
                    meshed_dims.add(int(edim))
                except Exception:
                    pass
        except Exception:
            pass

        if meshed_dims:
            filtered = [d for d in self._dims if d in meshed_dims]
            if filtered != list(self._dims):
                if getattr(self._parent, "_verbose", False):
                    skipped = sorted(set(self._dims) - meshed_dims)
                    print(
                        f"[MeshViewer] auto-filter: requested dims={self._dims}, "
                        f"meshed dims={sorted(meshed_dims)}, "
                        f"skipping empty {skipped}"
                    )
                self._dims = filtered if filtered else list(self._dims)

        # ── Selection state ─────────────────────────────────────────
        sel = SelectionState()
        self._selection_state = sel

        # ── Window (creates QApplication) ───────────────────────────
        default_title = f"MeshViewer — {self._parent.name}"
        win = ViewerWindow(title=title or default_title)

        # ── UI tabs (AFTER QApplication exists) ─────────────────────
        info_tab = MeshInfoTab()

        # ── Label state ─────────────────────────────────────────────
        _label_actors: list = []

        def _toggle_node_labels(checked: bool):
            for a in _label_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _label_actors.clear()
            if checked and scene.node_coords is not None and len(scene.node_coords) > 0:
                labels = [str(int(t)) for t in scene.node_tags]
                try:
                    actor = plotter.add_point_labels(
                        scene.node_coords, labels,
                        font_size=8,
                        text_color="white",
                        shape_color="#333333",
                        shape_opacity=0.6,
                        show_points=False,
                        always_visible=True,
                        name="_node_labels",
                    )
                    _label_actors.append(actor)
                except Exception:
                    pass
            # (phantom node labels removed — phantom nodes are now
            #  conditional on the Constraints tab checkbox)
            plotter.render()

        def _toggle_elem_labels(checked: bool):
            for a in _label_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _label_actors.clear()
            if checked:
                centers = []
                labels = []
                for elem_tag, info in scene.elem_data.items():
                    nodes = info.get("nodes", [])
                    if nodes:
                        coords = []
                        for nid in nodes:
                            idx = scene.node_tag_to_idx.get(int(nid))
                            if idx is not None:
                                coords.append(scene.node_coords[idx])
                        if coords:
                            center = np.mean(coords, axis=0)
                            centers.append(center)
                            labels.append(str(elem_tag))
                if centers:
                    try:
                        actor = plotter.add_point_labels(
                            np.array(centers), labels,
                            font_size=8,
                            text_color="#a6e3a1",
                            shape_color="#333333",
                            shape_opacity=0.6,
                            show_points=False,
                            always_visible=True,
                            name="_elem_labels",
                        )
                        _label_actors.append(actor)
                    except Exception:
                        pass
            plotter.render()

        def _toggle_wireframe(checked: bool):
            for dim in registry.dims:
                actor = registry.dim_actors.get(dim)
                if actor is None:
                    continue
                if checked:
                    actor.GetProperty().SetRepresentationToWireframe()
                else:
                    actor.GetProperty().SetRepresentationToSurface()
            plotter.render()

        def _toggle_edges(checked: bool):
            for dim in registry.dims:
                actor = registry.dim_actors.get(dim)
                if actor is None or dim < 2:
                    continue
                prop = actor.GetProperty()
                prop.SetEdgeVisibility(checked)
                if checked:
                    prop.SetEdgeColor(0.17, 0.29, 0.43)  # #2C4A6E
                    prop.SetLineWidth(0.5)
            plotter.render()

        display_tab = DisplayTab(
            on_node_labels=_toggle_node_labels,
            on_elem_labels=_toggle_elem_labels,
            on_wireframe=_toggle_wireframe,
            on_show_edges=_toggle_edges,
        )
        def _on_mesh_filter(active_dims: set[int]):
            for dim in registry.dims:
                actor = registry.dim_actors.get(dim)
                if actor is None:
                    continue
                actor.SetVisibility(dim in active_dims)
            plotter.render()

        filter_tab = MeshFilterTab(self._dims, on_filter_changed=_on_mesh_filter)

        win.add_tab("Info", info_tab.widget)
        win.add_tab("Display", display_tab.widget)
        win.add_tab("Filter", filter_tab.widget)

        plotter = win.plotter

        # ── Build scene ─────────────────────────────────────────────
        _verbose = getattr(self._parent, '_verbose', False)
        scene = build_mesh_scene(
            plotter, self._dims,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
            node_marker_size=self._point_size,
            verbose=_verbose,
        )
        self._scene_data = scene
        registry = scene.registry

        # ── Resolve FEM snapshot for overlays ───────────────────────
        fem = self._fem
        if fem is None:
            try:
                fem = self._parent.mesh.queries.get_fem_data(
                    dim=max(self._dims))
            except Exception:
                fem = None

        import pyvista as pv

        # ── Overlay infrastructure ──────────────────────────────────
        from .ui.loads_tab import LoadsTabPanel, pattern_color
        from .ui.mass_tab import MassTabPanel
        from .ui.constraints_tab import ConstraintsTabPanel, constraint_color
        from .overlays.constraint_overlay import (
            build_node_pair_actors, build_surface_actors,
        )
        from apeGmsh.mesh._record_set import ConstraintKind

        _load_actors: list = []
        _mass_actors: list = []
        _constraint_actors: list = []

        def _characteristic_length() -> float:
            """Geometric mean of significant bounding box spans."""
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
                sig = span[span > max_span * 0.01]
                return float(np.prod(sig) ** (1.0 / len(sig)))
            except Exception:
                return 1.0

        _overlay_scales = {
            'force_arrow':        1.0,
            'moment_arrow':       1.0,
            'mass_sphere':        1.0,
            'constraint_marker':  1.0,
            'constraint_line':    1.0,
        }

        _moment_template = None

        def _get_moment_template(radius: float):
            nonlocal _moment_template
            if _moment_template is None:
                from .overlays.moment_glyph import make_moment_glyph
                _moment_template = make_moment_glyph(
                    radius=1.0, tube_radius=0.08,
                    arc_degrees=270, resolution=24)
            return _moment_template

        # ── Loads overlay ───────────────────────────────────────────
        def _on_loads_patterns_changed(active_patterns):
            for a in _load_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _load_actors.clear()

            if not active_patterns or fem is None or not fem.nodes.loads:
                plotter.render()
                return

            char_len = _characteristic_length()
            force_len = char_len * 0.05 * _overlay_scales['force_arrow']
            moment_len = char_len * 0.05 * _overlay_scales['moment_arrow']
            origin = registry.origin_shift

            by_pat: dict[str, list] = {}
            for r in fem.nodes.loads:
                if r.pattern in active_patterns:
                    by_pat.setdefault(r.pattern, []).append(r)

            for pat, records in by_pat.items():
                f_positions, f_dirs, f_mags = [], [], []
                m_positions, m_dirs, m_mags = [], [], []

                for r in records:
                    try:
                        xyz = fem.nodes.coords[
                            fem.nodes.index(int(r.node_id))] - origin
                    except Exception:
                        continue

                    if r.force_xyz is not None:
                        fxyz = np.array(r.force_xyz, dtype=float)
                        fmag = float(np.linalg.norm(fxyz))
                        if fmag > 1e-30:
                            f_positions.append(xyz)
                            f_dirs.append(fxyz / fmag)
                            f_mags.append(fmag)

                    if r.moment_xyz is not None:
                        mxyz = np.array(r.moment_xyz, dtype=float)
                        mmag = float(np.linalg.norm(mxyz))
                        if mmag > 1e-30:
                            m_positions.append(xyz)
                            m_dirs.append(mxyz / mmag)
                            m_mags.append(mmag)

                color = pattern_color(pat)

                if f_positions:
                    pos_arr = np.array(f_positions, dtype=float)
                    dir_arr = np.array(f_dirs, dtype=float)
                    mag_arr = np.array(f_mags, dtype=float)
                    max_mag = mag_arr.max()
                    scale_arr = (mag_arr / max_mag
                                 if max_mag > 0
                                 else np.ones_like(mag_arr))

                    cloud = pv.PolyData(pos_arr)
                    cloud['vectors'] = (
                        dir_arr * scale_arr[:, np.newaxis] * force_len)
                    glyphs = cloud.glyph(
                        orient='vectors', scale='vectors', factor=1.0)
                    actor = plotter.add_mesh(
                        glyphs, color=color, lighting=False,
                        name=f"_loads_force_{pat}",
                        reset_camera=False, pickable=False,
                    )
                    _load_actors.append(actor)

                if m_positions:
                    pos_arr = np.array(m_positions, dtype=float)
                    dir_arr = np.array(m_dirs, dtype=float)
                    mag_arr = np.array(m_mags, dtype=float)
                    max_mag = mag_arr.max()
                    scale_arr = (mag_arr / max_mag
                                 if max_mag > 0
                                 else np.ones_like(mag_arr))

                    cloud = pv.PolyData(pos_arr)
                    cloud['vectors'] = (
                        dir_arr * scale_arr[:, np.newaxis]
                        * moment_len * 0.6)
                    template = _get_moment_template(moment_len)
                    glyphs = cloud.glyph(
                        geom=template, orient='vectors',
                        scale='vectors', factor=1.0)
                    actor = plotter.add_mesh(
                        glyphs, color=color, lighting=False,
                        opacity=0.85,
                        name=f"_loads_moment_{pat}",
                        reset_camera=False, pickable=False,
                    )
                    _load_actors.append(actor)

            plotter.render()

        # ── Mass overlay ────────────────────────────────────────────
        _mass_scalar_bar_title = 'Nodal mass'

        def _on_mass_overlay_changed(show: bool):
            for a in _mass_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _mass_actors.clear()
            try:
                plotter.remove_scalar_bar(_mass_scalar_bar_title)
            except Exception:
                pass

            if not show or fem is None or not fem.nodes.masses:
                plotter.render()
                return

            positions = []
            masses = []
            origin = registry.origin_shift
            for r in fem.nodes.masses:
                try:
                    xyz = fem.nodes.coords[
                        fem.nodes.index(int(r.node_id))] - origin
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
            sphere = pv.Sphere(
                radius=1.0, theta_resolution=10, phi_resolution=10)
            scale_factor = base_r / (max_mass ** (1.0 / 3.0))
            glyphs = cloud.glyph(
                geom=sphere, scale='mass', factor=scale_factor,
            )

            actor = plotter.add_mesh(
                glyphs, scalars='mass', cmap='viridis',
                scalar_bar_args={'title': _mass_scalar_bar_title},
                name="_mass_overlays",
                reset_camera=False,
                pickable=False,
            )
            _mass_actors.append(actor)
            plotter.render()

        # ── Constraints overlay ─────────────────────────────────────
        def _on_constraint_kinds_changed(active_kinds: set[str]):
            for a in _constraint_actors:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
            _constraint_actors.clear()

            if (not active_kinds or fem is None
                    or (not fem.nodes.constraints
                        and not fem.elements.constraints)):
                plotter.render()
                return

            char_len = _characteristic_length()
            origin = registry.origin_shift
            marker_r = (char_len * 0.003
                        * _overlay_scales['constraint_marker'])
            cst_lw = max(1, int(
                3 * _overlay_scales['constraint_line']))

            np_kinds = active_kinds & ConstraintKind.NODE_PAIR_KINDS
            if np_kinds:
                for mesh, kwargs in build_node_pair_actors(
                    fem, np_kinds, origin, marker_r, cst_lw,
                    constraint_color,
                ):
                    actor = plotter.add_mesh(mesh, **kwargs)
                    _constraint_actors.append(actor)

            s_kinds = active_kinds & ConstraintKind.SURFACE_KINDS
            if s_kinds:
                interp_lw = max(1, int(
                    2 * _overlay_scales['constraint_line']))
                for mesh, kwargs in build_surface_actors(
                    fem, s_kinds, origin, interp_lw,
                    constraint_color,
                ):
                    actor = plotter.add_mesh(mesh, **kwargs)
                    _constraint_actors.append(actor)

            # Phantom nodes (dark grey spheres)
            if (ConstraintKind.NODE_TO_SURFACE in active_kinds
                    and fem.nodes.constraints):
                pn = fem.nodes.constraints.phantom_nodes()
                if len(pn) > 0:
                    pn_coords = pn.coords - origin
                    sphere = pv.Sphere(
                        radius=marker_r * 0.7,
                        theta_resolution=8, phi_resolution=8)
                    cloud = pv.PolyData(pn_coords)
                    glyphs = cloud.glyph(
                        geom=sphere, orient=False, scale=False)
                    actor = plotter.add_mesh(
                        glyphs, color="#555555", lighting=False,
                        name="_phantom_nodes",
                        reset_camera=False, pickable=False,
                    )
                    _constraint_actors.append(actor)

            plotter.render()

        # ── Insert overlay tabs ─────────────────────────────────────
        loads_comp = getattr(self._parent, 'loads', None)
        loads_tab = None
        if loads_comp is not None:
            def _on_force_scale(v: float):
                _overlay_scales['force_arrow'] = v
                # loads_tab is assigned below before these callbacks fire.
                assert loads_tab is not None
                _on_loads_patterns_changed(loads_tab.active_patterns())

            def _on_moment_scale(v: float):
                _overlay_scales['moment_arrow'] = v
                assert loads_tab is not None
                _on_loads_patterns_changed(loads_tab.active_patterns())

            loads_tab = LoadsTabPanel(
                loads_comp, fem=fem,
                on_patterns_changed=_on_loads_patterns_changed,
                on_force_scale=_on_force_scale,
                on_moment_scale=_on_moment_scale,
            )
            win.add_tab("Loads", loads_tab.widget)

        mass_comp = getattr(self._parent, 'masses', None)
        mass_tab = None
        if mass_comp is not None:
            mass_tab = MassTabPanel(
                mass_comp, fem=fem,
                on_overlay_changed=_on_mass_overlay_changed,
            )
            win.add_tab("Mass", mass_tab.widget)

        constraints_comp = getattr(self._parent, 'constraints', None)
        constraints_tab = None
        if constraints_comp is not None:
            constraints_tab = ConstraintsTabPanel(
                constraints_comp, fem=fem,
                on_kinds_changed=_on_constraint_kinds_changed,
            )
            win.add_tab("Constraints", constraints_tab.widget)

        # ── Preferences (created AFTER scene — needs registry) ─────
        from .overlays.pref_helpers import make_line_width_cb, make_opacity_cb, make_edges_cb
        from .overlays.glyph_helpers import rebuild_node_cloud

        def _pref_point_size(v: float):
            rebuild_node_cloud(plotter, scene, v)
            plotter.render()

        _pref_line_width = make_line_width_cb(registry, plotter)
        _pref_opacity = make_opacity_cb(registry, plotter)
        _pref_edges = make_edges_cb(registry, plotter)

        def _pref_overlay_scale(key: str, mult: float):
            _overlay_scales[key] = mult
            if key in ('force_arrow', 'moment_arrow') and loads_tab is not None:
                _on_loads_patterns_changed(loads_tab.active_patterns())
            elif key == 'mass_sphere' and mass_tab is not None:
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
        win.add_tab("Preferences", prefs.widget)

        # ── Core modules ────────────────────────────────────────────
        color_mgr = ColorManager(registry)
        # Mesh scene uses a single uniform color for all dims — override
        # the default per-dim idle palette so hover→unhover restores the
        # correct colour instead of a different shade.
        from .scene.mesh_scene import DEFAULT_MESH_RGB
        color_mgr.set_idle_fn(lambda _dt: DEFAULT_MESH_RGB)
        vis_mgr = VisibilityManager(registry, color_mgr, sel, plotter, verbose=_verbose)
        pick_engine = PickEngine(plotter, registry)

        # ── Pick mode state ─────────────────────────────────────────
        _pick_mode: list[str] = ["brep"]  # "brep", "element", "node"
        _picked_elems: list[int] = []
        _picked_nodes: list[int] = []

        # ── Wire callbacks ──────────────────────────────────────────

        def _on_pick(dt: DimTag, ctrl: bool):
            mode = _pick_mode[0]
            if mode == "brep":
                if ctrl:
                    sel.unpick(dt)
                else:
                    sel.toggle(dt)
            elif mode == "element":
                dim = dt[0]
                cell_map = scene.batch_cell_to_elem.get(dim, {})
                picker = pick_engine._click_picker
                cell_id = picker.GetCellId()
                elem_tag = cell_map.get(cell_id)
                if elem_tag is not None:
                    if elem_tag in _picked_elems:
                        _picked_elems.remove(elem_tag)
                    else:
                        _picked_elems.append(elem_tag)
                    edata = scene.elem_data.get(elem_tag, {})
                    info_tab.show_element(elem_tag, edata)
                    win.set_status(f"Element {elem_tag} | {len(_picked_elems)} picked")
            elif mode == "node":
                if scene.node_tree is not None:
                    picker = pick_engine._click_picker
                    pos = picker.GetPickPosition()
                    if pos:
                        _, idx = scene.node_tree.query(pos)
                        node_tag = int(scene.node_tags[idx])
                        if node_tag in _picked_nodes:
                            _picked_nodes.remove(node_tag)
                        else:
                            _picked_nodes.append(node_tag)
                        coords = scene.node_coords[idx]
                        info_tab.show_node(node_tag, coords)
                        win.set_status(
                            f"Node {node_tag} | {len(_picked_nodes)} picked"
                        )

        pick_engine.on_pick = _on_pick
        pick_engine.set_hidden_check(vis_mgr.is_hidden)

        # Filter
        filter_tab._on_filter = pick_engine.set_pickable_dims

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

        # Selection changed -> recolor
        def _on_sel_changed():
            for entity_dt in registry.all_entities():
                is_picked = entity_dt in sel._picks
                is_hidden = vis_mgr.is_hidden(entity_dt)
                color_mgr.set_entity_state(
                    entity_dt, picked=is_picked, hidden=is_hidden,
                )
            plotter.render()
            n = len(sel.picks)
            win.set_status(f"{n} BRep entities picked")

        sel.on_changed.append(_on_sel_changed)
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
        # VTK-level
        plotter.add_key_event("h", _act_hide)
        plotter.add_key_event("i", _act_isolate)
        plotter.add_key_event("r", _act_reveal_all)
        plotter.add_key_event("u", lambda: sel.undo())

        # Window-level (work regardless of focus)
        win.add_shortcut("Escape", lambda: sel.clear())
        win.add_shortcut("Q", lambda: win.window.close())

        # Pick mode toggles
        def _set_mode(m):
            _pick_mode[0] = m
            win.set_status(f"Pick mode: {m.upper()}")

        plotter.add_key_event("e", lambda: _set_mode("element"))
        plotter.add_key_event("n", lambda: _set_mode("node"))
        plotter.add_key_event("b", lambda: _set_mode("brep"))

        # Show summary
        n_nodes = len(scene.node_tags)
        n_elems = sum(len(v) for v in scene.brep_to_elems.values())
        info_tab.show_summary(n_nodes, n_elems)
        win.set_status(
            f"Mesh: {n_nodes:,} nodes, {n_elems:,} elements | "
            f"Mode: BRep (press E=element, N=node, B=brep)"
        )

        # ── Run ─────────────────────────────────────────────────────
        win.exec()
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selection(self):
        from apeGmsh.viz.Selection import Selection
        picks = self._selection_state.picks if self._selection_state else []
        return Selection(picks, self._parent)

    @property
    def tags(self) -> list[DimTag]:
        return self._selection_state.picks if self._selection_state else []
