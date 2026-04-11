"""
MeshViewer — Interactive mesh viewer.

Assembled from independent core modules — no inheritance.
Uses the same core (navigation, pick engine, color manager) as
ModelViewer but with mesh-specific scene builder and UI tabs.

Usage::

    viewer = MeshViewer(parent, mesh_composite)
    viewer.show()
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

from .core.entity_registry import DimTag

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _SessionBase
    from apeGmsh.mesh.Mesh import Mesh
    from .core.selection import SelectionState
    from .scene.mesh_scene import MeshSceneData


class MeshViewer:
    """Interactive mesh viewer with element/node picking.

    Parameters
    ----------
    parent : _SessionBase
        The apeGmsh session.
    mesh_composite : Mesh
        The apeGmsh mesh object.
    dims : list[int], optional
        Which mesh dimensions to show (default: ``[1, 2, 3]``).
    point_size, line_width, surface_opacity, show_surface_edges
        Visual properties.
    fast : bool
        Ignored (always fast). Kept for backward compatibility.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        mesh_composite: "Mesh",
        *,
        dims: list[int] | None = None,
        point_size: float = 6.0,
        line_width: float = 3.0,
        surface_opacity: float = 1.0,
        show_surface_edges: bool = True,
        fast: bool = True,
        **kwargs: Any,
    ) -> None:
        self._parent = parent
        self._mesh = mesh_composite
        self._dims = dims if dims is not None else [1, 2, 3]

        self._point_size = point_size
        self._line_width = line_width
        self._surface_opacity = surface_opacity
        self._show_surface_edges = show_surface_edges

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
            # Remove existing
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
        prefs = PreferencesTab(
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
        )
        win.add_tab("Info", info_tab.widget)
        win.add_tab("Display", display_tab.widget)
        win.add_tab("Filter", filter_tab.widget)
        win.add_tab("Preferences", prefs.widget)

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

        # ── Core modules ────────────────────────────────────────────
        color_mgr = ColorManager(registry)
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
                # Resolve cell → element tag
                dim = dt[0]
                cell_map = scene.batch_cell_to_elem.get(dim, {})
                # Get the actual cell_id from the pick engine's last pick
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
                # Find nearest node to pick position
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

        # Selection changed → recolor
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
