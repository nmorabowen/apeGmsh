"""ResultsViewer — the post-solve interactive viewer.

Opens against a :class:`Results` instance and renders the bound
``FEMData`` mesh. Phase 0 ships the scaffolding only — substrate mesh,
time scrubber, stages tab, empty diagrams tab. Concrete diagrams
(contour, deformed shape, line force, …) arrive in subsequent phases.

Parallel to :class:`MeshViewer` (pre-solve) and :class:`ModelViewer`
(BRep geometry). Reuses the same ``viewers/scene/``, ``viewers/core/``,
and ``viewers/ui/`` infrastructure where possible.

Usage::

    from apeGmsh import Results

    results = Results.from_native("run.h5")
    results.viewer()                       # blocks until window closes
    results.viewer(blocking=False)         # subprocess (Phase 6+)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.mesh.FEMData import FEMData
    from .diagrams._director import ResultsDirector
    from .scene.fem_scene import FEMSceneData
    from .ui._diagram_settings_tab import DiagramSettingsTab


# Module-level strong-reference set so ResultsViewer instances survive
# the duration of their open window even when ``Results.viewer()`` is
# called as an expression statement (the typical jupyter notebook
# pattern). Without this, when ``app.exec_()`` returns immediately —
# which happens whenever a Qt event loop is already running in the
# kernel (``%gui qt``, ipykernel's qt backend, a prior viewer that left
# the loop spinning) — ``show()`` returns, the caller doesn't capture
# the return value, and the viewer is reaped by the garbage collector
# the moment the cell finishes. Adding to this set on ``show()`` and
# removing in ``_on_close`` keeps the window visible until the user
# actually closes it.
_LIVE_VIEWERS: "set[ResultsViewer]" = set()


class ResultsViewer:
    """Post-solve interactive viewer.

    Parameters
    ----------
    results
        A :class:`Results` instance — must have a bound FEMData
        (either from the embedded ``/model/`` snapshot or via
        ``results.bind(fem)``).
    title
        Window title. Defaults to ``"Results — <path>"``.
    """

    def __init__(
        self,
        results: "Results",
        *,
        title: Optional[str] = None,
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
    ) -> None:
        if results.fem is None:
            raise RuntimeError(
                "ResultsViewer requires a Results with a bound FEMData. "
                "Either open with Results.from_native(path) (which auto-"
                "binds the embedded snapshot) or call results.bind(fem)."
            )
        self._results = results
        self._title = title
        self._restore_session = restore_session
        self._save_session = save_session

        # Populated in show()
        self._director: "ResultsDirector | None" = None
        self._scene: "FEMSceneData | None" = None
        self._win: Any = None
        self._plotter: Any = None
        self._settings_tab: "DiagramSettingsTab | None" = None
        self._time_scrubber: Any = None
        self._substrate_actor: Any = None
        self._wireframe_actor: Any = None
        self._node_cloud_actor: Any = None
        self._node_label_actor: Any = None
        self._element_label_actor: Any = None
        self._plot_pane: Any = None
        self._details_panel: Any = None
        self._session_panel: Any = None
        # diagram instance -> side panel; lifecycle tied to registry.
        self._diagram_side_panels: dict = {}
        # (node_id, component) -> TimeHistoryPanel; user-closable from
        # the plot-pane tab × button.
        self._history_panels: dict = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def results(self) -> "Results":
        return self._results

    @property
    def director(self) -> "Optional[ResultsDirector]":
        return self._director

    @property
    def scene(self) -> "Optional[FEMSceneData]":
        return self._scene

    @property
    def plotter(self) -> Any:
        return self._plotter

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def show(self, *, maximized: bool = True):
        """Open the viewer window and run the Qt event loop until close.

        Returns ``self`` so callers can introspect the viewer state
        after the window closes.
        """
        # Pin a strong ref so the window survives even when the kernel
        # has a Qt event loop already running and ``app.exec_()``
        # returns immediately (jupyter ``%gui qt`` and friends).
        _LIVE_VIEWERS.add(self)

        # Lazy imports — keep ``apeGmsh.viewers`` importable in headless
        # environments. Qt / pyvistaqt only loaded when the user opens
        # an actual viewer.
        from .scene.fem_scene import build_fem_scene
        from .diagrams._director import ResultsDirector
        from .ui._results_window import ResultsWindow
        from .ui._diagram_settings_tab import DiagramSettingsTab
        from .ui.preferences_manager import PREFERENCES as _PREF

        # ── Director ────────────────────────────────────────────────
        director = ResultsDirector(self._results)
        self._director = director

        # ── FEM scene ───────────────────────────────────────────────
        fem = self._results.fem
        assert fem is not None    # validated in __init__
        scene = build_fem_scene(fem)
        self._scene = scene

        # ── Window (creates QApplication) ───────────────────────────
        title = self._title or self._default_title()
        win = ResultsWindow(title=title, on_close=self._on_close)
        self._win = win

        # ProbeOverlay needs the plotter, which doesn't exist until
        # ``win`` is constructed below. We initialise to None here so
        # any early access path resolves cleanly; the real overlay is
        # built after ``bind_plotter`` and feeds the viewport HUD.
        self._probe_overlay: Any = None

        # ── Diagram-settings panel (re-hosted by DetailsPanel) ──────
        # InspectorTab and the right-side QTabWidget dock retired in
        # B5: the only surviving widget from the legacy tab set is
        # DiagramSettingsTab, which DetailsPanel embeds.
        settings_tab = DiagramSettingsTab(director)
        self._settings_tab = settings_tab

        # ── Outline tree (left rail) ────────────────────────────────
        from .ui._outline_tree import OutlineTree
        outline = OutlineTree(director)
        win.set_left_widget(outline.widget)
        self._outline = outline

        # ── Right rail (plot pane + details) ────────────────────────
        from .ui._plot_pane import PlotPane
        from .ui._details_panel import DetailsPanel
        plot_pane = PlotPane()
        plot_pane.on_user_close(self._on_plot_user_close)
        # ── Geometry settings panel for the details dock ──────────
        # Built up-front so the details panel can route Geometry-row
        # selections into it. Available-fields detection runs once
        # against the union of every stage's nodal components.
        from .ui._geometry_settings_panel import GeometrySettingsPanel
        from .diagrams._kind_catalog import (
            _vector_prefixes as _vp,
            _union_across_stages as _uas,
        )
        try:
            available_vec_prefixes = set(_vp(_uas(director, "nodes")))
        except Exception:
            available_vec_prefixes = set()
        deform_field_options: list[str] = [
            p for p in ("displacement", "velocity", "acceleration")
            if p in available_vec_prefixes
        ]
        geometry_panel = GeometrySettingsPanel(
            director, deform_field_options,
        )
        details = DetailsPanel(settings_tab, geometry_panel)
        outline.on_composition_selected(self._on_outline_composition_selected)
        outline.on_geometry_selected(self._on_outline_geometry_selected)
        # Two-way binding (B++ §7): the Plots group in the outline
        # tree mirrors the plot pane's tab list; clicking a plot row
        # activates the corresponding tab and vice versa.
        outline.bind_plot_pane(plot_pane)
        win.set_right_widget(plot_pane.widget)
        win.set_details_widget(details.widget)
        self._plot_pane = plot_pane
        self._details_panel = details

        # ── Plotter — substrate mesh ────────────────────────────────
        plotter = win.plotter
        self._plotter = plotter

        prefs = _PREF.current
        # Substrate colors come from the active palette so the FEM mesh
        # recolors when the user switches theme (subscribed below).
        from .ui.theme import THEME, _hex_to_rgb
        palette = THEME.current

        # ── Session dock (viewer-level settings — theme, ...) ──────
        # Constructed BEFORE the substrate so the panel's initial
        # values come from the current preferences. Toggle / size
        # callbacks are wired to the actors after they are built.
        from .ui._session_panel import SessionPanel
        session = SessionPanel(
            point_size_initial=prefs.point_size,
            line_width_initial=prefs.mesh_line_width,
            opacity_initial=1.0,
        )
        win.set_session_widget(session.widget)
        self._session_panel = session

        # Substrate fill — drawn first; edges are rendered separately
        # by the wireframe actor below so they stay on top of any
        # contour diagram added later.
        actor = plotter.add_mesh(
            scene.grid,
            color=palette.substrate_color,
            show_edges=False,
            opacity=prefs.mesh_surface_opacity,
            lighting=True,
            smooth_shading=False,
            name="results_substrate",
            reset_camera=True,
        )
        self._substrate_actor = actor

        # ── Wireframe overlay ──────────────────────────────────────
        # Separate actor (style='wireframe') so the lines render on top
        # of any contour / deformed-shape diagram that draws polygons at
        # the same z-depth. Polygon-offset on the line mapper resolves
        # coincident topology by pulling lines toward the camera, which
        # eliminates z-fighting between the wireframe and the substrate
        # / diagram polygons.
        wireframe_actor = plotter.add_mesh(
            scene.grid,
            style="wireframe",
            color=palette.substrate_edge_color,
            line_width=prefs.mesh_line_width,
            opacity=1.0,
            lighting=False,
            pickable=False,
            name="results_wireframe",
        )
        try:
            wf_mapper = wireframe_actor.GetMapper()
            wf_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            wf_mapper.SetResolveCoincidentTopologyLineOffsetParameters(-1.0, -1.0)
        except Exception:
            pass
        self._wireframe_actor = wireframe_actor

        # ── Node-cloud overlay (matches the pre-solve mesh viewer) ─
        # One sphere glyph per FEM node, drawn over the substrate so
        # the user can see the discretization. Sized off the model
        # diagonal so it scales with bounding box.
        from .scene.glyph_points import build_node_cloud
        try:
            _, node_actor = build_node_cloud(
                plotter,
                scene.grid.points,
                model_diagonal=scene.model_diagonal,
                marker_size=prefs.point_size,
                color=palette.node_accent,
            )
        except Exception:
            node_actor = None
        self._node_cloud_actor = node_actor
        # Capture the glyphed sphere geometry + the centers it was
        # built against. Used by ``_sync_node_cloud`` to translate
        # each sphere when the substrate deforms. Reset whenever the
        # cloud is rebuilt (point-size change, theme retint of glyphs).
        self._capture_node_cloud_base()

        # Re-tint the substrate, wireframe, and node cloud when the
        # theme changes. Hex → 0..1 RGB for vtkProperty.SetColor.
        def _refresh_substrate_colors(p) -> None:
            try:
                if self._substrate_actor is not None:
                    prop = self._substrate_actor.GetProperty()
                    r, g, b = _hex_to_rgb(p.substrate_color)
                    prop.SetColor(r / 255.0, g / 255.0, b / 255.0)
                if self._wireframe_actor is not None:
                    er, eg, eb = _hex_to_rgb(p.substrate_edge_color)
                    self._wireframe_actor.GetProperty().SetColor(
                        er / 255.0, eg / 255.0, eb / 255.0,
                    )
                if self._node_cloud_actor is not None:
                    nr, ng, nb = _hex_to_rgb(p.node_accent)
                    self._node_cloud_actor.GetProperty().SetColor(
                        nr / 255.0, ng / 255.0, nb / 255.0,
                    )
                if plotter is not None:
                    plotter.render()
            except Exception:
                pass
        win.on_theme_changed(_refresh_substrate_colors)

        # ── Bind SessionPanel visualization toggles to the actors ──
        def _render() -> None:
            try:
                if plotter is not None:
                    plotter.render()
            except Exception:
                pass

        def _toggle_show_mesh(checked: bool) -> None:
            v = 1 if checked else 0
            for a in (self._substrate_actor, self._wireframe_actor):
                if a is None:
                    continue
                try:
                    a.SetVisibility(v)
                except Exception:
                    pass
            _render()

        def _toggle_show_nodes(checked: bool) -> None:
            if self._node_cloud_actor is None:
                return
            try:
                self._node_cloud_actor.SetVisibility(1 if checked else 0)
            except Exception:
                pass
            _render()

        def _on_line_width(value: float) -> None:
            if self._wireframe_actor is None:
                return
            try:
                self._wireframe_actor.GetProperty().SetLineWidth(float(value))
            except Exception:
                pass
            _render()

        def _on_opacity(value: float) -> None:
            v = max(0.0, min(1.0, float(value)))
            try:
                if self._wireframe_actor is not None:
                    self._wireframe_actor.GetProperty().SetOpacity(v)
                if self._node_cloud_actor is not None:
                    self._node_cloud_actor.GetProperty().SetOpacity(v)
            except Exception:
                pass
            _render()

        def _on_point_size(value: float) -> None:
            # Glyph spheres ignore SetPointSize — rebuild the cloud at
            # the new marker_size, preserving current visibility / opacity.
            if self._node_cloud_actor is None or plotter is None:
                return
            from .scene.glyph_points import build_node_cloud as _build
            old = self._node_cloud_actor
            try:
                old_visible = bool(old.GetVisibility())
                old_opacity = float(old.GetProperty().GetOpacity())
            except Exception:
                old_visible, old_opacity = True, 1.0
            try:
                plotter.remove_actor(old)
            except Exception:
                pass
            try:
                _, new_actor = _build(
                    plotter,
                    scene.grid.points,
                    model_diagonal=scene.model_diagonal,
                    marker_size=float(value),
                    color=THEME.current.node_accent,
                )
            except Exception:
                new_actor = None
            self._node_cloud_actor = new_actor
            if new_actor is not None:
                try:
                    new_actor.SetVisibility(1 if old_visible else 0)
                    new_actor.GetProperty().SetOpacity(old_opacity)
                except Exception:
                    pass
            # Re-capture base glyph + centers so deformation sync
            # uses the new actor's coordinates as its reference.
            self._capture_node_cloud_base()
            # Re-apply current deformation so the rebuilt cloud
            # immediately tracks the substrate.
            try:
                self._apply_deformation(int(director.step_index))
            except Exception:
                pass

        def _toggle_show_node_ids(checked: bool) -> None:
            self._set_node_id_labels(checked)
            _render()

        def _toggle_show_element_ids(checked: bool) -> None:
            self._set_element_id_labels(checked)
            _render()

        if self._session_panel is not None:
            self._session_panel.set_show_mesh_callback(_toggle_show_mesh)
            self._session_panel.set_show_nodes_callback(_toggle_show_nodes)
            self._session_panel.set_show_node_ids_callback(_toggle_show_node_ids)
            self._session_panel.set_show_element_ids_callback(
                _toggle_show_element_ids,
            )
            self._session_panel.set_point_size_callback(_on_point_size)
            self._session_panel.set_line_width_callback(_on_line_width)
            self._session_panel.set_opacity_callback(_on_opacity)

        # ── Deformation modifier (per-Geometry) ──────────────────
        # Each Geometry owns its own (enabled, field, scale) tuple.
        # The substrate is warped by the *active* Geometry's settings.
        # Reference (undeformed) substrate points are captured once;
        # toggling deformation off / switching to an undeformed
        # Geometry restores them exactly.
        #
        # Per-step mutation propagates to:
        #   - substrate fill + wireframe (they reference scene.grid)
        #   - any layer whose actor renders an UnstructuredGrid /
        #     PolyData with a ``vtkOriginalPointIds`` field (contour,
        #     deformed_shape, …) — points are scattered from the
        #     deformed substrate via the original-point map.
        # Layers that own non-substrate point geometry (vector glyph
        # source, gauss markers, node cloud) don't follow yet.
        import numpy as _np

        self._reference_points = _np.asarray(
            scene.grid.points, dtype=_np.float64,
        ).copy()

        # Dense FEM-id -> substrate-row lookup, built once. Reused by
        # the per-step field reader to scatter slab values back into a
        # row-aligned (N, 3) buffer.
        if scene.node_ids.size:
            _max_id = int(scene.node_ids.max())
            _deform_id_to_idx = _np.full(_max_id + 2, -1, dtype=_np.int64)
            _deform_id_to_idx[scene.node_ids] = _np.arange(
                scene.node_ids.size, dtype=_np.int64,
            )
        else:
            _deform_id_to_idx = _np.array([], dtype=_np.int64)

        def _read_deform_field(field: Optional[str], step: int) -> Optional[Any]:
            """Return ``(N, 3)`` vector field at ``step`` for the active stage.

            Reads ``<field>_x/_y/_z`` for every FEM node aligned to
            ``scene.grid.points``. Pads to 3-D with zeros when an axis
            is missing (e.g. 2-D model with only ``_x`` / ``_y``).
            Returns ``None`` if no field name was given or the read fails.
            """
            if not field or director.stage_id is None:
                return None
            try:
                results = self._results.stage(director.stage_id)
            except Exception:
                return None
            n = scene.node_ids.size
            out = _np.zeros((n, 3), dtype=_np.float64)
            id_to_idx = _deform_id_to_idx
            any_axis = False
            for axis, suf in enumerate(("x", "y", "z")):
                comp = f"{field}_{suf}"
                try:
                    slab = results.nodes.get(
                        ids=scene.node_ids,
                        component=comp,
                        time=[int(step)],
                    )
                except Exception:
                    continue
                if slab.values.size == 0:
                    continue
                slab_ids = _np.asarray(slab.node_ids, dtype=_np.int64)
                slab_vals = _np.asarray(slab.values[0], dtype=_np.float64)
                in_range = (
                    (slab_ids >= 0) & (slab_ids < id_to_idx.size)
                )
                positions = _np.full(slab_ids.shape, -1, dtype=_np.int64)
                positions[in_range] = id_to_idx[slab_ids[in_range]]
                valid = positions >= 0
                out[positions[valid], axis] = slab_vals[valid]
                any_axis = True
            return out if any_axis else None

        def _sync_layer_grids(deformed_pts: "_np.ndarray | None") -> None:
            """Scatter deformed substrate coords into every layer's
            submesh via the ``vtkOriginalPointIds`` map."""
            if self._director is None:
                return
            target_pts = (
                deformed_pts
                if deformed_pts is not None
                else self._reference_points
            )
            import pyvista as _pv
            for d in self._director.registry.diagrams():
                for actor in d._actors:                  # noqa: SLF001
                    try:
                        mapper = actor.GetMapper()
                        if mapper is None:
                            continue
                        raw = mapper.GetInput()
                        if raw is None:
                            continue
                        grid = _pv.wrap(raw)
                        if grid is None or "vtkOriginalPointIds" not in grid.point_data:
                            continue
                        opid = _np.asarray(
                            grid.point_data["vtkOriginalPointIds"],
                            dtype=_np.int64,
                        )
                        if opid.size == 0:
                            continue
                        in_range = (
                            (opid >= 0) & (opid < target_pts.shape[0])
                        )
                        new_pts = _np.asarray(
                            grid.points, dtype=_np.float64,
                        ).copy()
                        new_pts[in_range] = target_pts[opid[in_range]]
                        grid.points = new_pts
                    except Exception:
                        continue

        def _apply_deformation(step: int) -> None:
            """Apply the active Geometry's deformation to the substrate.

            No active geometry / disabled / no field → reset to ref.
            Otherwise: substrate points = ref + scale * field(step),
            then propagate to layer submeshes.
            """
            geom = director.geometries.active
            if (
                geom is None
                or not geom.deform_enabled
                or not geom.deform_field
            ):
                scene.grid.points = self._reference_points.copy()
                _sync_layer_grids(None)
                self._sync_node_cloud(None)
                self._sync_diagram_substrate_points(None)
                _render()
                return
            field_vals = _read_deform_field(geom.deform_field, int(step))
            if field_vals is None:
                scene.grid.points = self._reference_points.copy()
                _sync_layer_grids(None)
                self._sync_node_cloud(None)
                self._sync_diagram_substrate_points(None)
                _render()
                return
            deformed = (
                self._reference_points
                + float(geom.deform_scale) * field_vals
            )
            scene.grid.points = deformed
            _sync_layer_grids(deformed)
            self._sync_node_cloud(deformed)
            self._sync_diagram_substrate_points(deformed)
            _render()

        self._apply_deformation = _apply_deformation

        # Re-apply on step / stage / geometry-state changes. The
        # geometries observer covers active-geometry switches AND
        # in-place deformation edits (set_deformation fires it too).
        director.subscribe_step(
            lambda step: self._apply_deformation(int(step)),
        )
        director.subscribe_stage(
            lambda _sid: self._apply_deformation(int(director.step_index)),
        )
        director.geometries.subscribe(
            lambda: self._apply_deformation(int(director.step_index)),
        )

        # ── Time scrubber row (bottom of grid) ──────────────────────
        from .ui._time_scrubber import TimeScrubberDock
        scrubber = TimeScrubberDock(director)
        self._time_scrubber = scrubber
        win.set_bottom_widget(scrubber.widget)

        # ── Bind director to plotter ────────────────────────────────
        director.bind_plotter(
            plotter,
            scene=scene,
            render_callback=lambda: plotter.render() if plotter else None,
        )

        # ── Subscribe to diagram changes for side-panel docking ─────
        director.subscribe_diagrams(self._sync_side_panels)

        # ── Probe overlay + viewport HUDs ───────────────────────────
        # ProbePaletteHUD: top-right mode strip (point/line/slice).
        # PickReadoutHUD: top-left glass card showing the latest pick
        # and live values. The pick HUD chains into the same
        # on_point_result callback as the palette — both fire on every
        # point pick and remain in sync.
        from .overlays.probe_overlay import ProbeOverlay
        from .ui._viewport_hud import ProbePaletteHUD
        from .ui._pick_readout_hud import PickReadoutHUD
        self._probe_overlay = ProbeOverlay(plotter, scene, director)
        self._probe_hud = ProbePaletteHUD(
            plotter.interactor,
            self._probe_overlay,
            on_status=win.set_status,
        )
        self._pick_hud = PickReadoutHUD(
            plotter.interactor,
            self._probe_overlay,
            director,
        )

        # ── Shift-click → add time-history series (B++ §8) ─────────
        # Shift+left-click anywhere on the substrate snaps to the
        # nearest FEM node and opens (or focuses) a time-history tab
        # in the plot pane. Plain clicks fall through to the existing
        # picker / navigation handlers.
        from .overlays.shift_click_picker import ShiftClickPicker
        self._shift_click_picker = ShiftClickPicker(
            plotter, self._on_shift_click_world,
        )

        # ── Camera / view ──────────────────────────────────────────
        try:
            plotter.enable_parallel_projection()
        except Exception:
            pass

        # ── Status line summary ────────────────────────────────────
        n_nodes = scene.node_ids.size
        n_cells = scene.cell_to_element_id.size
        n_stages = len(director.stages())
        bits = [
            f"Mesh: {n_nodes:,} nodes, {n_cells:,} cells",
            f"Stages: {n_stages}",
        ]
        if scene.skipped_types:
            bits.append(f"Skipped types: {scene.skipped_types}")
        win.set_status(" | ".join(bits))

        # ── Slot-failure handler: route catches to the status bar ───
        # Registered before restore so any failures during the restore
        # path also land as toast messages.
        from ._failures import register_error_handler, unregister_error_handler

        def _slot_failure_to_status(name: str, exc: BaseException) -> None:
            try:
                win.set_status(
                    f"Error in {name}: {type(exc).__name__}: {exc}",
                    timeout=8000,
                )
            except Exception:
                pass

        register_error_handler(_slot_failure_to_status)
        self._slot_failure_handler = _slot_failure_to_status

        # ── Composition viewport gate ──────────────────────────────
        # Only the *active* composition's layers paint into the
        # viewport. When the user switches compositions, every
        # layer not in the active one has its actors hidden. The
        # per-card visibility checkbox controls the user's intent
        # *within* the composition; the composition gate is an
        # independent multiplicative filter.
        def _apply_composition_gate() -> None:
            """Hide every layer that isn't in the active Geometry's
            active Composition.

            Two-level gate: a layer paints only when (a) its parent
            Geometry is the active one, and (b) within that Geometry
            its parent Composition is active. The per-card visibility
            checkbox is an additional multiplicative filter on top.
            """
            geom_mgr = director.geometries
            active_geom = geom_mgr.active
            active_layers: set[int] = set()
            if active_geom is not None:
                active_comp = active_geom.compositions.active
                if active_comp is not None:
                    active_layers = set(map(id, active_comp.layers))
            for d in director.registry.diagrams():
                in_active = id(d) in active_layers
                desired = bool(d.is_visible) and in_active
                for actor in d._actors:                         # noqa: SLF001
                    try:
                        actor.SetVisibility(desired)
                    except Exception:
                        pass
            try:
                if plotter is not None:
                    plotter.render()
            except Exception:
                pass

        director.geometries.subscribe(_apply_composition_gate)
        # Also re-apply after registry changes (new layer added →
        # apply gate so the new actor is hidden if its comp isn't
        # active; layer removed → no-op).
        director.registry.subscribe(_apply_composition_gate)
        self._apply_composition_gate = _apply_composition_gate

        # ── Esc shortcut → return to base view ─────────────────────
        # Esc deselects the outline and drops the details panel into
        # idle state — the viewport is left showing just the substrate
        # mesh + node cloud + whatever active layers are visible.
        # Uses ApplicationShortcut so VTK's QtInteractor doesn't
        # swallow the key when the viewport has focus (same pattern
        # as Ctrl+H / Q in ResultsWindow).
        try:
            from qtpy import QtWidgets, QtGui, QtCore
            esc_sc = QtWidgets.QShortcut(
                QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape),
                win.window,
            )
            esc_sc.setContext(
                QtCore.Qt.ShortcutContext.ApplicationShortcut,
            )
            esc_sc.activated.connect(self._on_escape)
            self._esc_shortcut = esc_sc
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._install_esc_shortcut", exc)

        # ── Restore previous session if requested ───────────────────
        self._maybe_restore_session(win)

        # ── Run ─────────────────────────────────────────────────────
        try:
            win.exec()
        finally:
            unregister_error_handler(_slot_failure_to_status)
        return self

    # ------------------------------------------------------------------
    # Node-cloud deformation sync
    # ------------------------------------------------------------------

    def _capture_node_cloud_base(self) -> None:
        """Snapshot the glyphed sphere geometry + the centers it was
        built against. ``_sync_node_cloud`` reads these to translate
        each sphere when the substrate deforms.

        Called once after the cloud is built and again every time
        :meth:`_on_point_size` rebuilds it. Resets to ``None`` if the
        actor is missing or its mapper input is unwrappable.
        """
        self._node_cloud_base_glyph_pts = None
        self._node_cloud_centers_at_build = None
        self._node_cloud_pts_per_center = 0
        if self._node_cloud_actor is None:
            return
        try:
            import numpy as _np
            import pyvista as _pv
            mapper = self._node_cloud_actor.GetMapper()
            if mapper is None:
                return
            raw = mapper.GetInput()
            if raw is None:
                return
            glyph = _pv.wrap(raw)
            base = _np.asarray(glyph.points, dtype=_np.float64).copy()
            if self._scene is None or self._scene.grid is None:
                return
            n_centers = int(self._scene.grid.n_points)
            if n_centers == 0:
                return
            n_glyph_pts = base.shape[0]
            if n_glyph_pts % n_centers != 0:
                return
            self._node_cloud_base_glyph_pts = base
            self._node_cloud_centers_at_build = _np.asarray(
                self._scene.grid.points, dtype=_np.float64,
            ).copy()
            self._node_cloud_pts_per_center = n_glyph_pts // n_centers
        except Exception:
            self._node_cloud_base_glyph_pts = None
            self._node_cloud_centers_at_build = None
            self._node_cloud_pts_per_center = 0

    def _sync_node_cloud(self, deformed_pts) -> None:
        """Translate each sphere in the node cloud to follow the substrate.

        Each sphere ``i`` has ``pts_per_center`` glyphed points; we
        add ``deformed_pts[i] - centers_at_build[i]`` to every point
        of that sphere. ``deformed_pts=None`` means "reset to the
        reference (undeformed) state".
        """
        if (
            self._node_cloud_actor is None
            or self._node_cloud_base_glyph_pts is None
            or self._node_cloud_centers_at_build is None
            or self._node_cloud_pts_per_center == 0
        ):
            return
        try:
            import numpy as _np
            import pyvista as _pv
            target = (
                _np.asarray(deformed_pts, dtype=_np.float64)
                if deformed_pts is not None
                else self._reference_points
            )
            shifts = target - self._node_cloud_centers_at_build
            shifts_tiled = _np.repeat(
                shifts, self._node_cloud_pts_per_center, axis=0,
            )
            new_pts = self._node_cloud_base_glyph_pts + shifts_tiled
            mapper = self._node_cloud_actor.GetMapper()
            if mapper is None:
                return
            raw = mapper.GetInput()
            if raw is None:
                return
            glyph = _pv.wrap(raw)
            glyph.points = new_pts
        except Exception:
            pass

    def _sync_diagram_substrate_points(self, deformed_pts) -> None:
        """Forward the deformation to every layer's
        :meth:`Diagram.sync_substrate_points` hook.

        OPID-bearing submeshes (contour, etc.) are already handled by
        ``_sync_layer_grids`` walking mapper inputs directly — those
        layers' default no-op override skips this call. Layers that
        own non-substrate point geometry (gauss markers,
        future vector-glyph sync) get their points rewritten here.
        """
        if self._director is None or self._scene is None:
            return
        scene = self._scene
        for d in self._director.registry.diagrams():
            try:
                d.sync_substrate_points(deformed_pts, scene)
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Detach diagrams and release plotter binding before window dies."""
        # Auto-save the session before tearing down — the diagrams
        # still hold their specs at this point.
        if self._save_session:
            self._save_session_to_disk()
        if self._director is not None:
            try:
                self._director.unbind_plotter()
            except Exception:
                pass
        # Drop the strong ref pinned in show() so the viewer can be
        # garbage-collected after the window closes.
        _LIVE_VIEWERS.discard(self)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _save_session_to_disk(self) -> None:
        """Write ``<results>.viewer-session.json`` if the file is on disk.

        Quietly skips for in-memory Results (no ``_path``), and never
        raises — saving is best-effort. Errors land on stderr via the
        slot-failure infrastructure so a corrupt session save doesn't
        prevent the window from closing.
        """
        path = getattr(self._results, "_path", None)
        if path is None or self._director is None:
            return
        try:
            from .diagrams._session import save_session
            specs = [
                d.spec for d in self._director.registry.diagrams()
            ]
            fem = self._results.fem
            save_session(
                specs=specs,
                results_path=path,
                fem_snapshot_id=getattr(fem, "snapshot_id", None),
                active_stage_id=self._director.stage_id,
                active_step=int(self._director.step_index),
            )
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._save_session_to_disk", exc)

    def _maybe_restore_session(self, win: Any) -> None:
        """Apply a saved session if requested by ``restore_session``."""
        if self._restore_session is False or self._director is None:
            return
        path = getattr(self._results, "_path", None)
        if path is None:
            return
        try:
            from .diagrams._session import default_session_path, load_session
            session_path = default_session_path(path)
            if not session_path.exists():
                return
            session = load_session(session_path)
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._maybe_restore_session", exc)
            return

        if not session.diagrams:
            return

        if self._restore_session == "prompt":
            if not self._prompt_restore(win, session):
                return

        self._apply_session(session, win)

    def _prompt_restore(self, win: Any, session: Any) -> bool:
        """Yes/No dialog: restore N diagrams from the saved session?"""
        try:
            from qtpy.QtWidgets import QMessageBox
        except Exception:
            # Headless: don't prompt; conservative default is no.
            return False
        n = len(session.diagrams)
        labels = ", ".join(
            f"{s.kind}:{s.selector.component}" for s in session.diagrams[:5]
        )
        more = "" if n <= 5 else f" (+{n - 5} more)"
        reply = QMessageBox.question(
            win.window if hasattr(win, "window") else None,
            "Restore previous session?",
            f"A saved viewer session is available with {n} diagram(s):\n"
            f"  {labels}{more}\n\n"
            f"Restore them?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def _apply_session(self, session: Any, win: Any) -> None:
        """Reconstruct each spec into a Diagram and add to the registry.

        Restored layers are bundled into a single Composition under
        the active Geometry — old session JSON pre-dates the geometry
        refactor, so there's no per-layer geometry assignment to
        recover. The user can move them around afterwards.
        """
        from .diagrams._base import NoDataError
        from .ui._add_diagram_dialog import _KINDS

        kind_to_class = {entry.kind_id: entry.diagram_class for entry in _KINDS}
        n_added = 0
        n_skipped = 0
        restored_layers: list[Any] = []
        for spec in session.diagrams:
            cls = kind_to_class.get(spec.kind)
            if cls is None:
                n_skipped += 1
                continue
            try:
                diagram = cls(spec, self._results)
                self._director.registry.add(diagram)
                restored_layers.append(diagram)
                n_added += 1
            except NoDataError:
                n_skipped += 1
            except Exception as exc:
                from ._failures import report
                report(
                    f"ResultsViewer._apply_session({spec.kind})", exc,
                )
                n_skipped += 1

        # Bundle restored layers into one composition under the
        # active Geometry. If none was active, do nothing.
        if restored_layers:
            geom = self._director.geometries.active
            if geom is not None:
                comp = geom.compositions.add(
                    name="Restored", make_active=True,
                )
                for d in restored_layers:
                    geom.compositions.add_layer(comp.id, d)

        # Restore active stage / step where possible.
        try:
            if session.active_stage_id and (
                self._director.stage_id != session.active_stage_id
            ):
                self._director.set_stage(session.active_stage_id)
            if session.active_step:
                self._director.set_step(int(session.active_step))
        except Exception:
            pass

        try:
            msg = f"Restored {n_added} diagram(s)"
            if n_skipped:
                msg += f"; {n_skipped} skipped"
            win.set_status(msg, timeout=5000)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Label overlays (node IDs / element IDs)
    # ------------------------------------------------------------------

    def _set_node_id_labels(self, visible: bool) -> None:
        """Build (or remove) the ``add_point_labels`` actor for node IDs."""
        if self._plotter is None or self._scene is None:
            return
        if self._node_label_actor is not None:
            try:
                self._plotter.remove_actor(self._node_label_actor)
            except Exception:
                pass
            self._node_label_actor = None
        if not visible:
            return
        from .ui.theme import THEME
        from .ui.preferences_manager import PREFERENCES as _PREF
        try:
            coords = self._scene.grid.points
            ids = self._scene.node_ids
            if coords is None or len(coords) == 0:
                return
            labels = [str(int(t)) for t in ids]
            self._node_label_actor = self._plotter.add_point_labels(
                coords, labels,
                font_size=_PREF.current.node_label_font_size,
                text_color=THEME.current.text,
                shape_color=THEME.current.mantle,
                shape_opacity=0.6,
                show_points=False,
                always_visible=True,
                pickable=False,
                name="results_node_labels",
            )
        except Exception:
            self._node_label_actor = None

    def _set_element_id_labels(self, visible: bool) -> None:
        """Build (or remove) the element-ID labels at cell centroids."""
        if self._plotter is None or self._scene is None:
            return
        if self._element_label_actor is not None:
            try:
                self._plotter.remove_actor(self._element_label_actor)
            except Exception:
                pass
            self._element_label_actor = None
        if not visible:
            return
        from .ui.theme import THEME
        from .ui.preferences_manager import PREFERENCES as _PREF
        try:
            grid = self._scene.grid
            ids = self._scene.cell_to_element_id
            if grid.n_cells == 0 or ids is None or len(ids) == 0:
                return
            centers = grid.cell_centers().points
            labels = [str(int(t)) for t in ids]
            self._element_label_actor = self._plotter.add_point_labels(
                centers, labels,
                font_size=_PREF.current.element_label_font_size,
                text_color=THEME.current.success,
                shape_color=THEME.current.mantle,
                shape_opacity=0.6,
                show_points=False,
                always_visible=True,
                pickable=False,
                name="results_element_labels",
            )
        except Exception:
            self._element_label_actor = None

    def _default_title(self) -> str:
        path = self._results._reader_path()    # noqa: SLF001 — internal API
        if path == "(in-memory)":
            return "Results"
        from pathlib import Path
        return f"Results — {Path(path).name}"

    # ------------------------------------------------------------------
    # Shift-click → time-history series
    # ------------------------------------------------------------------

    def _on_shift_click_world(self, world_pos) -> None:
        """ShiftClickPicker callback — open a time-history for the picked node.

        Snaps the shift-click world position to the nearest FEM node
        via the probe overlay, picks a default component (the first
        active diagram's component, falling back to the first
        available nodal component), and opens a plot-pane history
        tab.
        """
        if (
            self._director is None
            or self._probe_overlay is None
            or self._plot_pane is None
        ):
            return
        try:
            node_id, _, _ = self._probe_overlay._snap_to_nearest_node(
                world_pos,
            )
        except Exception:
            return
        component = self._default_component_for_history()
        if component is None:
            if self._win is not None:
                self._win.set_status(
                    "Shift-click: no nodal component available — "
                    "add a diagram first.",
                    timeout=4000,
                )
            return
        self._open_time_history(int(node_id), component)

    def _default_component_for_history(self) -> "Optional[str]":
        """Pick a component for shift-click time-history plots.

        Prefers a component already used by an attached diagram so the
        plot matches what the user is looking at; otherwise falls back
        to the first available nodal component for the active stage.
        """
        if self._director is None:
            return None
        for d in self._director.registry.diagrams():
            if not d.is_attached:
                continue
            return d.spec.selector.component
        try:
            scoped = self._director.results.stage(self._director.stage_id)
            available = sorted(scoped.nodes.available_components())
        except Exception:
            return None
        return available[0] if available else None

    def _open_time_history(self, node_id: int, component: str) -> None:
        """Open (or focus) a TimeHistoryPanel as a plot-pane tab.

        Reuses an existing tab if one is already open for the same
        ``(node_id, component)`` so repeated shift-clicks on the same
        node don't multiply tabs.
        """
        if self._director is None or self._plot_pane is None:
            return
        key = ("history", int(node_id), str(component))
        if self._plot_pane.has_tab(key):
            self._plot_pane.set_active(key)
            return
        try:
            from .ui._time_history import TimeHistoryPanel
            panel = TimeHistoryPanel(self._director, node_id, component)
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._open_time_history", exc)
            return
        label = f"u(t) · node {node_id} · {component}"
        self._plot_pane.add_tab(key, label, panel.widget, closable=True)
        self._history_panels[(int(node_id), str(component))] = panel
        self._plot_pane.set_active(key)

    def _sync_side_panels(self) -> None:
        """Add / remove plot-pane side-panel tabs to match the registry.

        Each diagram's ``make_side_panel(director)`` returns a panel
        whose ``.widget`` attribute is hosted as a non-closable tab in
        the plot pane. When a diagram is removed from the registry,
        the tab + panel are torn down. Side-panel tabs are not user-
        closable — their lifecycle is the diagram's.
        """
        if self._director is None or self._plot_pane is None:
            return

        active = set(self._director.registry.diagrams())

        # Remove tabs for diagrams no longer present.
        for d in list(self._diagram_side_panels.keys()):
            if d not in active:
                panel = self._diagram_side_panels.pop(d)
                self._close_diagram_side_panel(d, panel)

        # Add tabs for new panel-bearing diagrams.
        for d in active:
            if d in self._diagram_side_panels:
                continue
            try:
                panel = d.make_side_panel(self._director)
            except Exception as exc:
                from ._failures import report
                report(
                    f"ResultsViewer._sync_side_panels({type(d).__name__})",
                    exc,
                )
                continue
            if panel is None:
                continue
            key = ("diagram", id(d))
            self._plot_pane.add_tab(
                key, d.display_label(), panel.widget, closable=False,
            )
            self._diagram_side_panels[d] = panel

    # ------------------------------------------------------------------
    # Plot pane / details routing
    # ------------------------------------------------------------------

    def _on_plot_user_close(self, key) -> None:
        """User clicked × on a plot-pane tab — only fires for closables."""
        if self._plot_pane is None or not isinstance(key, tuple):
            return
        kind = key[0]
        if kind == "history":
            _, node_id, component = key
            panel = self._history_panels.pop((node_id, component), None)
            if panel is not None:
                try:
                    panel.close()
                except Exception:
                    pass
            self._plot_pane.remove_tab(key)
        # Diagram side-panel tabs use closable=False; no other kinds
        # land here today.

    def _close_diagram_side_panel(self, diagram, panel) -> None:
        """Tear down a side panel + its plot-pane tab."""
        if self._plot_pane is None:
            return
        try:
            if hasattr(panel, "close"):
                panel.close()
        except Exception:
            pass
        self._plot_pane.remove_tab(("diagram", id(diagram)))

    def _on_escape(self) -> None:
        """Esc → return to the active Geometry's base view.

        Deselects the active composition (so just the substrate paints
        per the geometry's deformation), re-selects the active
        Geometry's row in the outline, and routes the details dock to
        the geometry settings panel.
        """
        try:
            from qtpy import QtWidgets
            fw = QtWidgets.QApplication.focusWidget()
            if fw is not None:
                fw.clearFocus()
        except Exception:
            pass
        if self._director is None:
            return
        active_geom = self._director.geometries.active
        if active_geom is None:
            return
        try:
            active_geom.compositions.set_active(None)
        except Exception:
            pass
        try:
            outline_widget = getattr(self, "_outline", None)
            if outline_widget is not None:
                outline_widget._select_geometry(active_geom.id)  # noqa: SLF001
        except Exception:
            pass
        if self._details_panel is not None:
            try:
                self._details_panel.show_geometry(active_geom.id)
            except Exception:
                pass

    def _on_outline_composition_selected(self, key) -> None:
        """Outline tree → composition row selected (or off-row).

        The outline only fires this with a non-None key when a
        composition row becomes the current item; ``None`` only
        arrives via :meth:`_outline._fire_idle` (off any row), in
        which case we drop into the idle (empty) details state.
        """
        if self._details_panel is None:
            return
        if key is None:
            self._details_panel.clear()
            return
        self._details_panel.show_stack()

    def _on_outline_geometry_selected(self, geom_id) -> None:
        """Outline tree → Geometry row selected (or off-row)."""
        if self._details_panel is None:
            return
        if geom_id is None:
            self._details_panel.clear()
            return
        self._details_panel.show_geometry(geom_id)


