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
    from apeGmsh.viewers.data import ViewerData
    from .core.color_manager import ColorManager
    from .core.color_mode_controller import ColorModeController
    from .core.entity_registry import EntityRegistry
    from .core.pick_engine import PickEngine
    from .core.selection import SelectionState
    from .core.visibility import VisibilityManager
    from .overlays.mesh_tangent_normal_overlay import MeshTangentNormalOverlay
    from .scene.mesh_scene import MeshSceneData
    from .ui.mesh_tabs import MeshInfoTab


class MeshViewer:
    """Interactive mesh viewer with element/node picking.

    Displays mesh elements and nodes with optional load, constraint,
    and mass overlays.  Overlay data comes from a resolved
    :class:`apeGmsh.viewers.data.ViewerData` snapshot — either passed
    explicitly or auto-resolved from the session at show time.

    Parameters
    ----------
    parent : _SessionBase
        The apeGmsh session.
    dims : list[int], optional
        Which mesh dimensions to show (default: ``[1, 2, 3]``).
    point_size, line_width, surface_opacity, show_surface_edges
        Visual properties.
    view : ViewerData, optional
        Pre-resolved structural snapshot.  If not provided, the
        viewer calls ``get_fem_data()`` automatically when the
        window opens and wraps the resulting FEMData.  Phase 8.7
        commit 6 renamed this kwarg from ``fem`` to ``view``.
    fast : bool
        Ignored (always fast). Kept for backward compatibility.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        *,
        dims: list[int] | None = None,
        point_size: float | None = None,
        line_width: float | None = None,
        surface_opacity: float | None = None,
        show_surface_edges: bool | None = None,
        origin_markers: list[tuple[float, float, float]] | None = None,
        origin_marker_show_coords: bool | None = None,
        view: "ViewerData | None" = None,
        fast: bool = True,
        **kwargs: Any,
    ) -> None:
        from .ui.preferences_manager import PREFERENCES
        p = PREFERENCES.current

        self._parent = parent
        self._dims = dims if dims is not None else [1, 2, 3]

        # Mesh viewer keeps its own pref-sourced visual defaults. Explicit
        # kwarg still wins; falling back to the user's persisted preference
        # otherwise. Historic hard-coded fallbacks (6.0/3.0/1.0/True) match
        # ``Preferences``'s ``node_marker_size``/``line_width`` defaults.
        self._point_size = (
            point_size if point_size is not None else p.node_marker_size
        )
        self._line_width = (
            line_width if line_width is not None else p.mesh_line_width
        )
        self._surface_opacity = (
            surface_opacity if surface_opacity is not None
            else p.mesh_surface_opacity
        )
        self._show_surface_edges = (
            show_surface_edges if show_surface_edges is not None
            else p.mesh_show_surface_edges
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
        self._view: "ViewerData | None" = view

        # Populated during show()
        self._selection_state: "SelectionState | None" = None
        self._scene_data: "MeshSceneData | None" = None

        # Runtime state (populated in show()) — pre-declared for clarity
        self._plotter: Any = None
        self._win: Any = None
        self._scene: "MeshSceneData | None" = None
        self._registry: "EntityRegistry | None" = None
        self._sel: "SelectionState | None" = None
        self._color_mgr: "ColorManager | None" = None
        self._vis_mgr: "VisibilityManager | None" = None
        self._pick_engine: "PickEngine | None" = None
        self._color_mode_ctrl: "ColorModeController | None" = None
        self._info_tab: "MeshInfoTab | None" = None
        self._mesh_tn_overlay: "MeshTangentNormalOverlay | None" = None

        # UI tabs (resolved after construction)
        self._loads_tab: Any = None
        self._mass_tab: Any = None
        self._constraints_tab: Any = None

        # Mutable per-show state buckets
        self._label_actors: list = []
        self._load_actors: list = []
        self._mass_actors: list = []
        self._constraint_actors: list = []
        self._overlay_scales: dict[str, float] = {
            'force_arrow':           1.0,
            'moment_arrow':          1.0,
            'mass_sphere':           1.0,
            'constraint_marker':     1.0,
            'constraint_line':       1.0,
            'tangent_normal_arrow':  1.0,
        }
        self._moment_template: Any = None
        self._pick_mode: list[str] = ["brep"]   # "brep", "element", "node"
        self._picked_elems: list[int] = []
        self._picked_nodes: list[int] = []
        self._prev_hover: list[DimTag | None] = [None]
        self._hover_label: Any = None

        # Plan 04 step 3 — per-viewer ActiveObjects coordinator.
        # Populated by ``show()`` once a QApplication exists. Same
        # design as ``ResultsViewer._active``: a single source of
        # truth for "what is currently selected / which pick mode" so
        # panels subscribe instead of wiring direct callbacks.
        self._active: Any = None
        # Subscription handle for the SelectionState bridge; cleared
        # in ``_on_close``-equivalent paths.
        self._sel_bridge_unsub: Any = None

    # ==================================================================
    # Entry point
    # ==================================================================

    def show(self, *, title: str | None = None, maximized: bool = True):
        """Open the viewer window, block until closed."""
        from .core.navigation import install_navigation
        from .core.color_manager import ColorManager
        from .core.color_mode_controller import ColorModeController
        from .core.pick_engine import PickEngine
        from .core.visibility import VisibilityManager
        from .core.selection import SelectionState
        from .scene.mesh_scene import build_mesh_scene
        from .ui.viewer_window import ViewerWindow
        from .ui.preferences import PreferencesTab
        from .ui.mesh_tabs import MeshInfoTab, DisplayTab, MeshFilterTab
        from .ui.preferences_manager import PREFERENCES as _PREF
        from .ui.theme import THEME

        gmsh.model.occ.synchronize()

        self._dims = self._auto_filter_dims(self._dims)

        # ── Selection state ─────────────────────────────────────────
        sel = SelectionState()
        self._selection_state = sel
        self._sel = sel

        # ── Window (creates QApplication) ───────────────────────────
        # ``window_key`` opts into layout persistence under
        # ``QSettings("apeGmsh", "MeshViewer")`` (plan 08 follow-up).
        default_title = f"MeshViewer — {self._parent.name}"
        win = ViewerWindow(
            title=title or default_title, window_key="MeshViewer",
        )
        self._win = win

        # ── Plan 04 step 3 — ActiveObjects coordinator ──────────────
        # One per viewer. Pick mode + selection get their canonical
        # signal surface here; existing per-instance state (the
        # ``_pick_mode[0]`` cache, ``sel.on_changed`` callbacks) stays
        # in lockstep via two bridges installed below.
        from .core._active_objects import ActiveObjects
        self._active = ActiveObjects(parent=win.window)
        # Pick mode bridge — subscribers update the legacy cache + the
        # status bar. ``_set_pick_mode`` now flows through
        # ``set_active_pick_mode``, so any future panel that wants to
        # react to pick-mode flips can subscribe to
        # ``activePickModeChanged`` without touching this file.
        self._active.activePickModeChanged.connect(self._on_active_pick_mode)
        # Seed the active pick mode with whatever the constructor /
        # __init__ set on the legacy cache (default "brep"). This keeps
        # ``self._active.active_pick_mode`` aligned with
        # ``_pick_mode[0]`` from the start — code that subscribes to
        # ``activePickModeChanged`` won't see a phantom "" state before
        # the first user key-press.
        try:
            self._active.set_active_pick_mode(self._pick_mode[0])
        except Exception:
            pass

        # ── UI tabs (AFTER QApplication exists) ─────────────────────
        info_tab = MeshInfoTab()
        self._info_tab = info_tab

        display_tab = DisplayTab(
            on_color_mode=self._on_color_mode,
            on_node_labels=self._toggle_node_labels,
            on_elem_labels=self._toggle_elem_labels,
            on_wireframe=self._toggle_wireframe,
            on_show_edges=self._toggle_edges,
        )

        filter_tab = MeshFilterTab(
            self._dims,
            on_filter_changed=self._on_mesh_filter,
            on_mesh_probes_changed=self._on_mesh_probes_changed,
        )

        win.add_tab("Info", info_tab.widget)
        win.add_tab("Display", display_tab.widget)
        win.add_tab("Filter", filter_tab.widget)

        plotter = win.plotter
        self._plotter = plotter

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
        self._scene = scene
        registry = scene.registry
        self._registry = registry

        # ── Hover tooltip overlay (Qt label on the plotter widget) ──
        from qtpy import QtWidgets as _QtW, QtCore as _QtC
        _interactor = getattr(plotter, "interactor", None)
        if _interactor is not None:
            self._hover_label = _QtW.QLabel(_interactor)
            self._hover_label.setStyleSheet(
                "QLabel { background-color: rgba(40, 40, 40, 220); "
                "color: #eee; padding: 4px 6px; border: 1px solid #555; "
                "border-radius: 3px; }"
            )
            self._hover_label.setAttribute(
                _QtC.Qt.WA_TransparentForMouseEvents
            )
            self._hover_label.hide()

        # ── Origin markers ──────────────────────────────────────────
        from .overlays.origin_markers_overlay import OriginMarkerOverlay
        from .ui.origin_markers_panel import OriginMarkersPanel
        _marker_size = _PREF.current.origin_marker_size
        origin_overlay = OriginMarkerOverlay(
            plotter,
            origin_shift=registry.origin_shift,
            model_diagonal=scene.model_diagonal,
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
        win.add_tab("Markers", origin_panel.widget)

        # ── Mesh tangent / normal overlay ───────────────────────────
        from .overlays.mesh_tangent_normal_overlay import (
            MeshTangentNormalOverlay,
        )
        self._mesh_tn_overlay = MeshTangentNormalOverlay(
            plotter,
            origin_shift=registry.origin_shift,
            model_diagonal=scene.model_diagonal,
            scale=_PREF.current.tangent_normal_scale,
        )

        # ── Resolve FEM snapshot for overlays ───────────────────────
        view = self._view
        if view is None:
            try:
                fem = self._parent.mesh.queries.get_fem_data(
                    dim=max(self._dims))
            except Exception:
                fem = None
            if fem is not None:
                from .data import ViewerData
                view = ViewerData.from_fem(fem)
        self._view = view

        # ── Insert overlay tabs (loads/mass/constraints) ────────────
        self._build_overlay_tabs(win)

        # ── Preferences (created AFTER scene — needs registry) ─────
        self._build_preferences_tab(win)

        # ── Core modules ────────────────────────────────────────────
        color_mgr = ColorManager(registry)
        self._color_mgr = color_mgr
        # With the CAD-neutral palette (dim_pt/crv black, dim_srf/vol gray)
        # the default per-dim idle function already gives a uniform look
        # while keeping nodes black — no override needed.
        vis_mgr = VisibilityManager(registry, color_mgr, sel, plotter, verbose=_verbose)
        self._vis_mgr = vis_mgr
        pick_engine = PickEngine(
            plotter, registry,
            drag_threshold=_PREF.current.drag_threshold,
        )
        self._pick_engine = pick_engine

        # ── Color mode controller ───────────────────────────────────
        self._color_mode_ctrl = ColorModeController(
            color_mgr=color_mgr,
            registry=registry,
            scene=scene,
            sel=sel,
            vis_mgr=vis_mgr,
            plotter=plotter,
        )

        # ── Browser tab (groups + element types visibility) ─────────
        from .ui.mesh_browser_tab import MeshBrowserTab
        if scene.group_to_breps or scene.brep_dominant_type:
            self._browser_tab = MeshBrowserTab(
                scene, on_hidden_changed=vis_mgr.set_hidden,
            )
            win.add_tab("Browser", self._browser_tab.widget)

        # ── Left-rail outline tree — primary navigation ────────────
        # ParaView-style alternative to the right-side Browser tab.
        # Lists Physical Groups + Element Types + Parts with eye-icon
        # visibility. Browser tab still coexists; future PR may
        # deprecate one.
        from .ui._mesh_outline_tree import MeshOutlineTree
        from .ui._dock_registry import DockSpec
        parts_reg = getattr(self._parent, 'parts', None)

        # Map outline row kinds to the right-side tab names whose
        # contents serve as the property editor for that row type.
        # mesh.viewer's right side is the legacy ``QTabWidget`` (not
        # tabified extension docks), so we identify tabs by their
        # text label.
        _OUTLINE_TAB_MAP = {
            "group": "Browser",
            "type":  "Browser",
            "part":  "Browser",
        }

        def _on_outline_row_focused(kind: str, _payload) -> None:
            tab_name = _OUTLINE_TAB_MAP.get(kind)
            if tab_name is not None:
                win.focus_tab(tab_name)

        self._outline_tree = MeshOutlineTree(
            scene=scene,
            selection=sel,
            vis_mgr=vis_mgr,
            parts_registry=parts_reg,
            on_row_focused=_on_outline_row_focused,
        )
        win.add_extension_dock(DockSpec(
            dock_id="dock_mesh_outline",
            title="Outline",
            factory=lambda _p: self._outline_tree.widget,
            default_area="left",
        ))

        # ── Clipping tab ────────────────────────────────────────────
        from .core.clipping_controller import ClippingController
        from .ui.clipping_tab import ClippingTab
        self._clipping_ctrl = ClippingController(plotter, registry)
        self._clipping_tab = ClippingTab(
            on_toggle=self._clipping_ctrl.toggle,
            on_reset=self._clipping_ctrl.reset,
        )
        win.add_tab("Clipping", self._clipping_tab.widget)

        # ── Wire callbacks ──────────────────────────────────────────
        pick_engine.on_pick = self._handle_pick
        pick_engine.on_hover = self._handle_hover
        pick_engine.on_box_select = self._handle_box_select
        pick_engine.set_hidden_check(vis_mgr.is_hidden)

        # Dim checkboxes: drive both actor visibility (the user-visible
        # effect) and the pick-engine pickable-dims mask (so picks ignore
        # hidden dims). Earlier this only set the pickable mask, making
        # the checkboxes appear to do nothing visually.
        def _on_dim_filter(active_dims: set[int]) -> None:
            self._on_mesh_filter(active_dims)
            pick_engine.set_pickable_dims(active_dims)
        filter_tab._on_filter = _on_dim_filter

        # Selection changed -> recolor
        sel.on_changed.append(self._handle_sel_changed)
        # Plan 04 step 3 — selection bridge into ActiveObjects.
        # ``SelectionState`` keeps its legacy ``on_changed`` list (the
        # plan doc marks it as a one-release compatibility shim); this
        # bridge fans the same event out via ``selectionChanged``
        # so new subscribers don't need to know about SelectionState's
        # internal callback list. The payload is an immutable tuple of
        # picks — fresh per emit, so ``ActiveObjects``' identity check
        # doesn't suppress repeat fires when picks mutate in place,
        # and downstream subscribers get a stable snapshot they can
        # cache without worrying about later mutation. Subscribers
        # needing more (centroid, parent shapes) reach for
        # ``viewer._sel`` via the viewer reference.
        def _sel_bridge() -> None:
            if self._active is not None and self._sel is not None:
                self._active.set_selection(tuple(self._sel.picks))
        sel.on_changed.append(_sel_bridge)
        self._sel_bridge_unsub = _sel_bridge
        vis_mgr.on_changed.append(lambda: plotter.render())
        # Repaint mesh idle colors when the theme palette changes
        win.on_theme_changed(lambda _p: self._handle_sel_changed())
        # Refresh tangent / normal arrows when palette changes
        win.on_theme_changed(
            lambda _p: self._mesh_tn_overlay.refresh_theme()
            if self._mesh_tn_overlay is not None else None
        )

        # ── Navigation ──────────────────────────────────────────────
        install_navigation(
            plotter,
            get_orbit_pivot=lambda: sel.centroid(registry),
        )

        # ── Install pick engine ─────────────────────────────────────
        pick_engine.install()

        # ── Toolbar buttons for visibility ──────────────────────────
        win.add_toolbar_separator()
        win.add_toolbar_button("Hide selected (H)", "H", self._act_hide)
        win.add_toolbar_button("Isolate selected (I)", "I", self._act_isolate)
        win.add_toolbar_button("Reveal all (R)", "R", self._act_reveal_all)
        win.add_toolbar_separator()
        win.add_toolbar_button("Save image…", "Img", self._act_screenshot)

        # ── Keybindings ─────────────────────────────────────────────
        plotter.add_key_event("h", self._act_hide)
        plotter.add_key_event("i", self._act_isolate)
        plotter.add_key_event("r", self._act_reveal_all)
        plotter.add_key_event("u", lambda: sel.undo())

        win.add_shortcut("Escape", lambda: sel.clear())
        win.add_shortcut("Q", lambda: win.window.close())

        plotter.add_key_event("e", lambda: self._set_pick_mode("element"))
        plotter.add_key_event("n", lambda: self._set_pick_mode("node"))
        plotter.add_key_event("b", lambda: self._set_pick_mode("brep"))

        # ── Show summary ────────────────────────────────────────────
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

    # ==================================================================
    # Setup helpers
    # ==================================================================

    def _auto_filter_dims(self, dims: list[int]) -> list[int]:
        """Drop requested dims that have no mesh elements."""
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

        if not meshed_dims:
            return list(dims)
        filtered = [d for d in dims if d in meshed_dims]
        if filtered == list(dims):
            return list(dims)
        if getattr(self._parent, "_verbose", False):
            skipped = sorted(set(dims) - meshed_dims)
            print(
                f"[MeshViewer] auto-filter: requested dims={dims}, "
                f"meshed dims={sorted(meshed_dims)}, "
                f"skipping empty {skipped}"
            )
        return filtered if filtered else list(dims)

    def _build_overlay_tabs(self, win) -> None:
        """Construct loads/mass/constraints tabs if their components exist."""
        from .ui.loads_tab import LoadsTabPanel
        from .ui.mass_tab import MassTabPanel
        from .ui.constraints_tab import ConstraintsTabPanel

        loads_comp = getattr(self._parent, 'loads', None)
        if loads_comp is not None:
            self._loads_tab = LoadsTabPanel(
                loads_comp, view=self._view,
                on_patterns_changed=self._rebuild_loads_overlay,
                on_force_scale=self._on_force_scale,
                on_moment_scale=self._on_moment_scale,
            )
            win.add_tab("Loads", self._loads_tab.widget)

        mass_comp = getattr(self._parent, 'masses', None)
        if mass_comp is not None:
            self._mass_tab = MassTabPanel(
                mass_comp, view=self._view,
                on_overlay_changed=self._rebuild_mass_overlay,
            )
            win.add_tab("Mass", self._mass_tab.widget)

        constraints_comp = getattr(self._parent, 'constraints', None)
        if constraints_comp is not None:
            self._constraints_tab = ConstraintsTabPanel(
                constraints_comp, view=self._view,
                on_kinds_changed=self._rebuild_constraints_overlay,
            )
            win.add_tab("Constraints", self._constraints_tab.widget)

    def _build_preferences_tab(self, win) -> None:
        """Create the Session preferences tab."""
        from .overlays.pref_helpers import (
            make_line_width_cb, make_opacity_cb, make_edges_cb,
        )
        from .overlays.glyph_helpers import rebuild_node_cloud
        from .ui.preferences import PreferencesTab
        from .ui.theme import THEME
        from .ui.preferences_dialog import open_preferences_dialog
        from .ui.theme_editor_dialog import open_theme_editor
        from qtpy import QtWidgets as _QtW

        registry = self._registry
        plotter = self._plotter
        scene = self._scene
        assert registry is not None and plotter is not None and scene is not None

        def _pref_point_size(v: float):
            rebuild_node_cloud(plotter, scene, v)
            plotter.render()

        prefs = PreferencesTab(
            point_size=self._point_size,
            line_width=self._line_width,
            surface_opacity=self._surface_opacity,
            show_surface_edges=self._show_surface_edges,
            on_point_size=_pref_point_size,
            on_line_width=make_line_width_cb(registry, plotter),
            on_opacity=make_opacity_cb(registry, plotter),
            on_edges=make_edges_cb(registry, plotter),
            on_overlay_scale=self._on_overlay_scale,
            on_theme=lambda name: THEME.set_theme(name),
        )
        # Session tab (formerly "Preferences") — runtime tweaks.
        # "Global preferences…" button opens the persistent-prefs dialog.
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
        win.add_tab("Session", prefs.widget)

    # ==================================================================
    # Display callbacks (Display tab)
    # ==================================================================

    def _toggle_node_labels(self, checked: bool) -> None:
        from .ui.theme import THEME as _THEME
        from .ui.preferences_manager import PREFERENCES as _PREF

        plotter = self._plotter
        scene = self._scene
        if plotter is None or scene is None:
            return

        for a in self._label_actors:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        self._label_actors.clear()

        if checked and scene.node_coords is not None and len(scene.node_coords) > 0:
            labels = [str(int(t)) for t in scene.node_tags]
            try:
                actor = plotter.add_point_labels(
                    scene.node_coords, labels,
                    font_size=_PREF.current.node_label_font_size,
                    text_color=_THEME.current.text,
                    shape_color=_THEME.current.mantle,
                    shape_opacity=0.6,
                    show_points=False,
                    always_visible=True,
                    name="_node_labels",
                )
                self._label_actors.append(actor)
            except Exception:
                pass
        # (phantom node labels removed — phantom nodes are now
        #  conditional on the Constraints tab checkbox)
        plotter.render()

    def _toggle_elem_labels(self, checked: bool) -> None:
        from .ui.theme import THEME as _THEME
        from .ui.preferences_manager import PREFERENCES as _PREF

        plotter = self._plotter
        scene = self._scene
        if plotter is None or scene is None:
            return

        for a in self._label_actors:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        self._label_actors.clear()

        if not checked:
            plotter.render()
            return

        # Group elements by node count (npe) so each group can compute
        # centroids vectorized via fancy-indexing the dense tag_to_idx
        # ndarray. ~10x faster than a Python loop on large meshes.
        by_npe: dict[int, list[tuple[int, list[int]]]] = {}
        for elem_tag, info in scene.elem_data.items():
            nodes = info.get("nodes")
            if not nodes:
                continue
            by_npe.setdefault(len(nodes), []).append((elem_tag, nodes))

        tag_to_idx = scene.tag_to_idx
        node_coords = scene.node_coords
        max_tag = tag_to_idx.shape[0] - 1
        centers_list: list[np.ndarray] = []
        labels: list[str] = []
        for npe, entries in by_npe.items():
            tags = np.fromiter(
                (e[0] for e in entries), dtype=np.int64, count=len(entries),
            )
            conn = np.array([e[1] for e in entries], dtype=np.int64)
            # Mask out-of-range node tags; rows touching any missing
            # node are skipped (matches old `.get(... ) is not None`).
            in_range = (conn >= 0) & (conn <= max_tag)
            if not in_range.all():
                row_ok = in_range.all(axis=1)
                if not row_ok.any():
                    continue
                tags = tags[row_ok]
                conn = conn[row_ok]
            idx_mat = tag_to_idx[conn]
            row_ok = (idx_mat >= 0).all(axis=1)
            if not row_ok.all():
                tags = tags[row_ok]
                idx_mat = idx_mat[row_ok]
            if idx_mat.size == 0:
                continue
            centroids = node_coords[idx_mat].mean(axis=1)
            centers_list.append(centroids)
            labels.extend(str(int(t)) for t in tags)

        if centers_list:
            centers = np.concatenate(centers_list, axis=0)
            try:
                actor = plotter.add_point_labels(
                    centers, labels,
                    font_size=_PREF.current.element_label_font_size,
                    text_color=_THEME.current.success,
                    shape_color=_THEME.current.mantle,
                    shape_opacity=0.6,
                    show_points=False,
                    always_visible=True,
                    name="_elem_labels",
                )
                self._label_actors.append(actor)
            except Exception:
                pass
        plotter.render()

    def _toggle_wireframe(self, checked: bool) -> None:
        registry = self._registry
        plotter = self._plotter
        if registry is None or plotter is None:
            return
        for dim in registry.dims:
            actor = registry.dim_actors.get(dim)
            if actor is None:
                continue
            if checked:
                actor.GetProperty().SetRepresentationToWireframe()
            else:
                actor.GetProperty().SetRepresentationToSurface()
        plotter.render()

    def _toggle_edges(self, checked: bool) -> None:
        registry = self._registry
        plotter = self._plotter
        if registry is None or plotter is None:
            return
        # Mesh edges are drawn by the dedicated wireframe actor (per-dim
        # corner-edge PolyData). Toggling edges = toggling that actor's
        # visibility — color/width are owned by the theme, not this
        # callback.
        for dim in registry.dims:
            wire_actor = registry.dim_wire_actors.get(dim)
            if wire_actor is None:
                continue
            wire_actor.SetVisibility(checked)
        plotter.render()

    def _on_color_mode(self, mode: str) -> None:
        ctrl = self._color_mode_ctrl
        if ctrl is None:
            return
        ctrl.set_mode(mode)
        if self._win is not None:
            self._win.set_status(f"Color mode: {mode}")

    # ==================================================================
    # Filter tab callbacks
    # ==================================================================

    def _on_mesh_filter(self, active_dims: set[int]) -> None:
        registry = self._registry
        plotter = self._plotter
        if registry is None or plotter is None:
            return
        # The set of dims that have any actor (fill, wire, or node cloud)
        # — node-cloud dims may extend beyond fill dims (e.g. dim=0
        # entities only exist as nodes).
        all_dims = (
            set(registry.dims)
            | set(registry.dim_node_actors.keys())
        )
        for dim in all_dims:
            visible = dim in active_dims
            for actor in (
                registry.dim_actors.get(dim),
                registry.dim_wire_actors.get(dim),
                registry.dim_node_actors.get(dim),
            ):
                if actor is None:
                    continue
                actor.SetVisibility(visible)
        plotter.render()

    def _on_mesh_probes_changed(self, show_tangents: bool, show_normals: bool) -> None:
        ov = self._mesh_tn_overlay
        if ov is None:
            return
        ov.set_show_tangents(show_tangents)
        ov.set_show_normals(show_normals)

    # ==================================================================
    # Overlay rebuild helpers (loads / mass / constraints)
    # ==================================================================

    def _characteristic_length(self) -> float:
        """Geometric mean of significant bounding-box spans."""
        registry = self._registry
        if registry is None:
            return 1.0
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

    def _get_moment_template(self, _radius: float):
        if self._moment_template is None:
            from .overlays.moment_glyph import make_moment_glyph
            self._moment_template = make_moment_glyph(
                radius=1.0, tube_radius=0.08,
                arc_degrees=270, resolution=24,
            )
        return self._moment_template

    def _rebuild_loads_overlay(self, active_patterns) -> None:
        import pyvista as pv
        from .ui.loads_tab import pattern_color

        plotter = self._plotter
        registry = self._registry
        view = self._view
        if plotter is None or registry is None:
            return

        for a in self._load_actors:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        self._load_actors.clear()

        if not active_patterns or view is None or not view.nodes.loads:
            plotter.render()
            return

        char_len = self._characteristic_length()
        force_len = char_len * 0.05 * self._overlay_scales['force_arrow']
        moment_len = char_len * 0.05 * self._overlay_scales['moment_arrow']
        origin = registry.origin_shift

        by_pat: dict[str, list] = {}
        for r in view.nodes.loads:
            if r.pattern in active_patterns:
                by_pat.setdefault(r.pattern, []).append(r)

        for pat, records in by_pat.items():
            f_positions, f_dirs, f_mags = [], [], []
            m_positions, m_dirs, m_mags = [], [], []

            for r in records:
                try:
                    xyz = view.nodes.coords[
                        view.nodes.index(int(r.node_id))] - origin
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
                self._load_actors.append(actor)

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
                template = self._get_moment_template(moment_len)
                glyphs = cloud.glyph(
                    geom=template, orient='vectors',
                    scale='vectors', factor=1.0)
                actor = plotter.add_mesh(
                    glyphs, color=color, lighting=False,
                    opacity=0.85,
                    name=f"_loads_moment_{pat}",
                    reset_camera=False, pickable=False,
                )
                self._load_actors.append(actor)

        plotter.render()

    _MASS_SCALAR_BAR_TITLE = 'Nodal mass'

    def _rebuild_mass_overlay(self, show: bool) -> None:
        import pyvista as pv

        plotter = self._plotter
        registry = self._registry
        view = self._view
        if plotter is None or registry is None:
            return

        for a in self._mass_actors:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        self._mass_actors.clear()
        try:
            plotter.remove_scalar_bar(self._MASS_SCALAR_BAR_TITLE)
        except Exception:
            pass

        if not show or view is None or not view.nodes.masses:
            plotter.render()
            return

        positions = []
        masses = []
        origin = registry.origin_shift
        for r in view.nodes.masses:
            try:
                xyz = view.nodes.coords[
                    view.nodes.index(int(r.node_id))] - origin
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

        char_len = self._characteristic_length()
        max_mass = max(masses) if masses else 1.0
        base_r = char_len * 0.005 * self._overlay_scales['mass_sphere']

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
            scalar_bar_args={'title': self._MASS_SCALAR_BAR_TITLE},
            name="_mass_overlays",
            reset_camera=False,
            pickable=False,
        )
        self._mass_actors.append(actor)
        plotter.render()

    def _rebuild_constraints_overlay(self, active_kinds: set[str]) -> None:
        import pyvista as pv
        from .ui.constraints_tab import constraint_color
        from .overlays.constraint_overlay import (
            build_node_pair_actors, build_surface_actors,
        )
        from .data import (
            NODE_PAIR_KINDS, NODE_TO_SURFACE_KIND, SURFACE_KINDS,
        )

        plotter = self._plotter
        registry = self._registry
        view = self._view
        if plotter is None or registry is None:
            return

        for a in self._constraint_actors:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        self._constraint_actors.clear()

        if (not active_kinds or view is None
                or (not view.nodes.constraints
                    and not view.elements.constraints)):
            plotter.render()
            return

        char_len = self._characteristic_length()
        origin = registry.origin_shift
        marker_r = (char_len * 0.003
                    * self._overlay_scales['constraint_marker'])
        cst_lw = max(1, int(
            3 * self._overlay_scales['constraint_line']))

        np_kinds = active_kinds & NODE_PAIR_KINDS
        if np_kinds:
            for mesh, kwargs in build_node_pair_actors(
                view, np_kinds, origin, marker_r, cst_lw,
                constraint_color,
            ):
                actor = plotter.add_mesh(mesh, **kwargs)
                self._constraint_actors.append(actor)

        s_kinds = active_kinds & SURFACE_KINDS
        if s_kinds:
            interp_lw = max(1, int(
                2 * self._overlay_scales['constraint_line']))
            for mesh, kwargs in build_surface_actors(
                view, s_kinds, origin, interp_lw,
                constraint_color,
            ):
                actor = plotter.add_mesh(mesh, **kwargs)
                self._constraint_actors.append(actor)

        # Phantom nodes (dark grey spheres)
        if (NODE_TO_SURFACE_KIND in active_kinds
                and view.nodes.constraints):
            pn_ids, pn_coords_raw = view.nodes.constraints.phantom_nodes()
            if pn_ids.size > 0:
                pn_coords = pn_coords_raw - origin
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
                self._constraint_actors.append(actor)

        plotter.render()

    # ==================================================================
    # Overlay scale callbacks (Session/Preferences tab)
    # ==================================================================

    def _on_force_scale(self, v: float) -> None:
        self._overlay_scales['force_arrow'] = v
        if self._loads_tab is not None:
            self._rebuild_loads_overlay(self._loads_tab.active_patterns())

    def _on_moment_scale(self, v: float) -> None:
        self._overlay_scales['moment_arrow'] = v
        if self._loads_tab is not None:
            self._rebuild_loads_overlay(self._loads_tab.active_patterns())

    def _on_overlay_scale(self, key: str, mult: float) -> None:
        self._overlay_scales[key] = mult
        if key in ('force_arrow', 'moment_arrow') and self._loads_tab is not None:
            self._rebuild_loads_overlay(self._loads_tab.active_patterns())
        elif key == 'mass_sphere' and self._mass_tab is not None:
            show_cb = getattr(self._mass_tab, '_show_cb', None)
            if show_cb is not None:
                self._rebuild_mass_overlay(show_cb.isChecked())
        elif key.startswith('constraint') and self._constraints_tab is not None:
            self._rebuild_constraints_overlay(
                self._constraints_tab.active_kinds()
            )
        elif key == 'tangent_normal_arrow' and self._mesh_tn_overlay is not None:
            from .ui.preferences_manager import PREFERENCES as _PREF
            base = _PREF.current.tangent_normal_scale
            self._mesh_tn_overlay.set_scale(base * mult)

    # ==================================================================
    # Pick / hover / selection callbacks
    # ==================================================================

    def _handle_pick(self, dt: DimTag, ctrl: bool) -> None:
        sel = self._sel
        scene = self._scene
        pick_engine = self._pick_engine
        info_tab = self._info_tab
        win = self._win
        if sel is None or scene is None or pick_engine is None:
            return

        mode = self._pick_mode[0]
        if mode == "brep":
            if ctrl:
                sel.unpick(dt)
            else:
                sel.toggle(dt)
            if info_tab is not None:
                info_tab.append_history(f"BRep {dt}")
        elif mode == "element":
            dim = dt[0]
            cell_map = scene.batch_cell_to_elem.get(dim)
            picker = pick_engine._click_picker
            cell_id = picker.GetCellId()
            elem_tag: int | None = None
            if cell_map is not None and 0 <= cell_id < len(cell_map):
                elem_tag = int(cell_map[cell_id])
            if elem_tag is not None:
                if elem_tag in self._picked_elems:
                    self._picked_elems.remove(elem_tag)
                else:
                    self._picked_elems.append(elem_tag)
                edata = scene.elem_data.get(elem_tag, {})
                if info_tab is not None:
                    info_tab.show_element(elem_tag, edata)
                    info_tab.append_history(
                        f"Elem {elem_tag} ({edata.get('type_name', '?')})"
                    )
                if win is not None:
                    win.set_status(
                        f"Element {elem_tag} | "
                        f"{len(self._picked_elems)} picked"
                    )
        elif mode == "node":
            if scene.node_tree is not None:
                picker = pick_engine._click_picker
                pos = picker.GetPickPosition()
                if pos:
                    _, idx = scene.node_tree.query(pos)
                    node_tag = int(scene.node_tags[idx])
                    if node_tag in self._picked_nodes:
                        self._picked_nodes.remove(node_tag)
                    else:
                        self._picked_nodes.append(node_tag)
                    coords = scene.node_coords[idx]
                    if info_tab is not None:
                        info_tab.show_node(node_tag, coords)
                        info_tab.append_history(
                            f"Node {node_tag} "
                            f"({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
                        )
                    if win is not None:
                        win.set_status(
                            f"Node {node_tag} | "
                            f"{len(self._picked_nodes)} picked"
                        )

    def _handle_hover(self, dt: DimTag | None) -> None:
        sel = self._sel
        color_mgr = self._color_mgr
        plotter = self._plotter
        if sel is None or color_mgr is None or plotter is None:
            return
        old = self._prev_hover[0]
        self._prev_hover[0] = dt
        if old is not None and old != dt:
            is_picked = old in sel._picks
            color_mgr.set_entity_state(old, picked=is_picked)
        if dt is not None:
            is_picked = dt in sel._picks
            if not is_picked:
                color_mgr.set_entity_state(dt, hovered=True)
        plotter.render()
        self._update_hover_tooltip(dt)

    def _update_hover_tooltip(self, dt: DimTag | None) -> None:
        label = self._hover_label
        if label is None:
            return
        text = self._tooltip_text(dt)
        if not text:
            label.hide()
            return
        label.setText(text)
        label.adjustSize()
        from qtpy import QtGui
        plotter = self._plotter
        if plotter is None or getattr(plotter, "interactor", None) is None:
            return
        local = plotter.interactor.mapFromGlobal(QtGui.QCursor.pos())
        label.move(local.x() + 14, local.y() + 18)
        label.raise_()
        label.show()

    def _tooltip_text(self, dt: DimTag | None) -> str:
        if dt is None:
            return ""
        mode = self._pick_mode[0]
        if mode == "brep":
            return f"BRep {dt}"
        scene = self._scene
        pick_engine = self._pick_engine
        if scene is None or pick_engine is None:
            return ""
        if mode == "element":
            cell_map = scene.batch_cell_to_elem.get(dt[0])
            cell_id = pick_engine._hover_picker.GetCellId()
            if cell_map is None or not (0 <= cell_id < len(cell_map)):
                return ""
            elem_tag = int(cell_map[cell_id])
            edata = scene.elem_data.get(elem_tag, {})
            return f"Elem {elem_tag}\n{edata.get('type_name', '?')}"
        if mode == "node":
            if scene.node_tree is None:
                return ""
            pos = pick_engine._hover_picker.GetPickPosition()
            if not pos:
                return ""
            _, idx = scene.node_tree.query(pos)
            node_tag = int(scene.node_tags[idx])
            coords = scene.node_coords[idx]
            return (
                f"Node {node_tag}\n"
                f"({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})"
            )
        return ""

    def _handle_sel_changed(self) -> None:
        sel = self._sel
        registry = self._registry
        color_mgr = self._color_mgr
        vis_mgr = self._vis_mgr
        plotter = self._plotter
        win = self._win
        if (sel is None or registry is None or color_mgr is None
                or vis_mgr is None or plotter is None):
            return
        for entity_dt in registry.all_entities():
            is_picked = entity_dt in sel._picks
            is_hidden = vis_mgr.is_hidden(entity_dt)
            color_mgr.set_entity_state(
                entity_dt, picked=is_picked, hidden=is_hidden,
            )
        plotter.render()
        if win is not None:
            n = len(sel.picks)
            win.set_status(f"{n} BRep entities picked")

    def _handle_box_select(self, dts: list[DimTag], ctrl: bool) -> None:
        sel = self._sel
        if sel is None:
            return
        if ctrl:
            sel.box_remove(dts)
        else:
            sel.box_add(dts)

    # ==================================================================
    # Visibility actions
    # ==================================================================

    def _act_hide(self) -> None:
        if self._vis_mgr is None or self._plotter is None:
            return
        self._vis_mgr.hide()
        self._plotter.render()

    def _act_isolate(self) -> None:
        if self._vis_mgr is None or self._plotter is None:
            return
        self._vis_mgr.isolate()
        self._plotter.render()

    def _act_reveal_all(self) -> None:
        if self._vis_mgr is None or self._plotter is None:
            return
        self._vis_mgr.reveal_all()
        self._plotter.render()

    def _act_screenshot(self) -> None:
        if self._plotter is None or self._win is None:
            return
        from qtpy import QtWidgets
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._win.window,
            "Save viewport image",
            f"{self._parent.name}.png",
            "Images (*.png *.jpg *.jpeg)",
        )
        if not path:
            return
        try:
            self._plotter.screenshot(path)
            self._win.set_status(f"Saved image: {path}")
        except Exception as e:
            self._win.set_status(f"Screenshot failed: {e}")

    def _set_pick_mode(self, m: str) -> None:
        """Switch the active pick mode (``"brep"`` / ``"element"`` / ``"node"``).

        Plan 04 step 3: routes through :class:`ActiveObjects` when one
        is wired (the standard show() path). The subscriber installed
        in show() keeps ``_pick_mode[0]`` and the status bar in sync —
        existing reads of ``_pick_mode[0]`` in ``_handle_pick`` /
        ``_handle_hover`` see the new value at the moment they fire.
        Pre-show (no active wired yet) we mutate the cache directly so
        constructor-time defaults still take effect.
        """
        if self._active is not None:
            self._active.set_active_pick_mode(m)
            return
        # No ActiveObjects yet (constructor-time default): apply the
        # legacy state directly.
        self._pick_mode[0] = m
        if self._hover_label is not None:
            self._hover_label.hide()
        if self._win is not None:
            self._win.set_status(f"Pick mode: {m.upper()}")

    def _on_active_pick_mode(self, mode: str) -> None:
        """Subscriber to ``ActiveObjects.activePickModeChanged``.

        Keeps the legacy ``_pick_mode[0]`` cache and the status bar in
        sync so existing reads — ``_handle_pick``, ``_handle_hover``,
        ``_tooltip_text`` — pick up the new mode immediately.
        """
        self._pick_mode[0] = mode
        if self._hover_label is not None:
            try:
                self._hover_label.hide()
            except Exception:
                pass
        if self._win is not None:
            self._win.set_status(f"Pick mode: {mode.upper()}")

    # ==================================================================
    # Public API
    # ==================================================================

    @property
    def selection(self):
        from apeGmsh.viz.Selection import Selection
        picks = self._selection_state.picks if self._selection_state else []
        return Selection(picks, self._parent)

    @property
    def tags(self) -> list[DimTag]:
        return self._selection_state.picks if self._selection_state else []
