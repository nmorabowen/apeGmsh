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
        details = DetailsPanel(settings_tab)
        outline.on_diagram_selected(self._on_outline_diagram_selected)
        # Two-way binding (B++ §7): the Plots group in the outline
        # tree mirrors the plot pane's tab list; clicking a plot row
        # activates the corresponding tab and vice versa.
        outline.bind_plot_pane(plot_pane)
        win.set_right_widget(plot_pane.widget)
        win.set_details_widget(details.widget)
        self._plot_pane = plot_pane
        self._details_panel = details

        # ── Session dock (viewer-level settings — theme, ...) ──────
        from .ui._session_panel import SessionPanel
        session = SessionPanel()
        win.set_session_widget(session.widget)
        self._session_panel = session

        # ── Plotter — substrate mesh ────────────────────────────────
        plotter = win.plotter
        self._plotter = plotter

        prefs = _PREF.current
        # Substrate colors come from the active palette so the FEM mesh
        # recolors when the user switches theme (subscribed below).
        from .ui.theme import THEME, _hex_to_rgb
        palette = THEME.current
        actor = plotter.add_mesh(
            scene.grid,
            color=palette.substrate_color,
            show_edges=True,
            edge_color=palette.substrate_edge_color,
            line_width=prefs.mesh_line_width,
            opacity=prefs.mesh_surface_opacity,
            lighting=True,
            smooth_shading=False,
            name="results_substrate",
            reset_camera=True,
        )
        self._substrate_actor = actor

        # Re-tint the substrate when the theme changes. Hex → 0..1 RGB
        # for vtkProperty.SetColor / SetEdgeColor.
        def _refresh_substrate_colors(p) -> None:
            if self._substrate_actor is None:
                return
            try:
                prop = self._substrate_actor.GetProperty()
                r, g, b = _hex_to_rgb(p.substrate_color)
                er, eg, eb = _hex_to_rgb(p.substrate_edge_color)
                prop.SetColor(r / 255.0, g / 255.0, b / 255.0)
                prop.SetEdgeColor(er / 255.0, eg / 255.0, eb / 255.0)
                if plotter is not None:
                    plotter.render()
            except Exception:
                pass
        win.on_theme_changed(_refresh_substrate_colors)

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

        # ── Restore previous session if requested ───────────────────
        self._maybe_restore_session(win)

        # ── Run ─────────────────────────────────────────────────────
        try:
            win.exec()
        finally:
            unregister_error_handler(_slot_failure_to_status)
        return self

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

        # Snapshot-id mismatch warning.
        current_id = getattr(self._results.fem, "snapshot_id", None)
        if (
            session.fem_snapshot_id is not None
            and current_id is not None
            and session.fem_snapshot_id != current_id
        ):
            try:
                win.set_status(
                    "Saved session was for a different mesh "
                    "(snapshot mismatch); skipping restore.",
                    timeout=8000,
                )
            except Exception:
                pass
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
        """Reconstruct each spec into a Diagram and add to the registry."""
        from .diagrams._base import NoDataError
        from .ui._add_diagram_dialog import _KINDS

        kind_to_class = {entry.kind_id: entry.diagram_class for entry in _KINDS}
        n_added = 0
        n_skipped = 0
        for spec in session.diagrams:
            cls = kind_to_class.get(spec.kind)
            if cls is None:
                n_skipped += 1
                continue
            try:
                diagram = cls(spec, self._results)
                self._director.registry.add(diagram)
                n_added += 1
            except NoDataError:
                # Stale spec (component renamed, fixture changed) — skip.
                n_skipped += 1
            except Exception as exc:
                from ._failures import report
                report(
                    f"ResultsViewer._apply_session({spec.kind})", exc,
                )
                n_skipped += 1

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

    def _on_outline_diagram_selected(self, diagram) -> None:
        """Outline tree's selection moved to/off a Diagram row."""
        if self._details_panel is None:
            return
        if diagram is None:
            self._details_panel.clear()
        else:
            self._details_panel.show_diagram(diagram)

