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
    from .ui.results_tabs import ResultsTabs


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
        self._tabs: "ResultsTabs | None" = None
        self._time_scrubber: Any = None
        self._substrate_actor: Any = None
        # diagram instance -> (QDockWidget, panel) pair; manage lifecycle
        # against registry add/remove events.
        self._diagram_docks: dict = {}
        # (node_id, component) -> (QDockWidget, TimeHistoryPanel)
        self._history_docks: dict = {}

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
        from .ui.results_tabs import build_results_tabs
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
        # built after ``bind_plotter`` and wired into a Probes tab.
        self._probe_overlay: Any = None

        # ── Tabs (Stages, Diagrams, Settings, Inspector, Probes) ────
        # Probes tab is appended after ProbeOverlay is constructed.
        tabs = build_results_tabs(
            director,
            on_open_history=self._open_time_history,
        )
        self._tabs = tabs
        for name, widget in tabs.to_pairs():
            win.add_tab(name, widget)

        # ── Plotter — substrate mesh ────────────────────────────────
        plotter = win.plotter
        self._plotter = plotter

        prefs = _PREF.current
        actor = plotter.add_mesh(
            scene.grid,
            color="lightgray",
            show_edges=True,
            edge_color="#444444",
            line_width=prefs.mesh_line_width,
            opacity=prefs.mesh_surface_opacity,
            lighting=True,
            smooth_shading=False,
            name="results_substrate",
            reset_camera=True,
        )
        self._substrate_actor = actor

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

        # ── Probe overlay + Probes tab ──────────────────────────────
        from .overlays.probe_overlay import ProbeOverlay
        from .ui._probes_tab import ProbesTab
        self._probe_overlay = ProbeOverlay(plotter, scene, director)
        probes_tab = ProbesTab(self._probe_overlay)
        self._tabs.probes = probes_tab
        win.add_tab("Probes", probes_tab.widget)

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

    def _open_time_history(self, node_id: int, component: str) -> None:
        """Inspector callback: dock a TimeHistoryPanel for (node, component).

        Reuses an existing dock if one is open for the same pair; the
        user's repeated "Open time history" clicks don't multiply
        windows.
        """
        if self._director is None or self._win is None:
            return
        from qtpy import QtWidgets, QtCore
        key = (int(node_id), str(component))
        if key in self._history_docks:
            dock, _panel = self._history_docks[key]
            try:
                dock.show()
                dock.raise_()
            except Exception:
                pass
            return
        try:
            from .ui._time_history import TimeHistoryPanel
            panel = TimeHistoryPanel(self._director, node_id, component)
        except Exception as exc:
            import sys
            print(
                f"[ResultsViewer] could not build time-history panel: {exc}",
                file=sys.stderr,
            )
            return

        try:
            dock = QtWidgets.QDockWidget(
                f"History — node {node_id} · {component}"
            )
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            dock.setWidget(panel.widget)
            self._win.window.addDockWidget(
                QtCore.Qt.RightDockWidgetArea, dock,
            )
            self._history_docks[key] = (dock, panel)
        except Exception as exc:
            import sys
            print(
                f"[ResultsViewer] history dock failed: {exc}",
                file=sys.stderr,
            )

    def _sync_side_panels(self) -> None:
        """Add / remove dockable side panels to match the registry.

        Each diagram's ``make_side_panel(director)`` returns a panel
        whose ``.widget`` attribute we wrap in a ``QDockWidget``. When
        a diagram is removed from the registry, we drop its dock.
        """
        if self._director is None or self._win is None:
            return
        from qtpy import QtWidgets, QtCore

        active = set(self._director.registry.diagrams())

        # Remove docks for diagrams no longer present.
        for d in list(self._diagram_docks.keys()):
            if d not in active:
                dock, panel = self._diagram_docks.pop(d)
                try:
                    if hasattr(panel, "close"):
                        panel.close()
                except Exception:
                    pass
                try:
                    self._win.window.removeDockWidget(dock)
                    dock.deleteLater()
                except Exception:
                    pass

        # Add docks for new panel-bearing diagrams.
        for d in active:
            if d in self._diagram_docks:
                continue
            try:
                panel = d.make_side_panel(self._director)
            except Exception as exc:
                import sys
                print(
                    f"[ResultsViewer] make_side_panel failed for "
                    f"{type(d).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            if panel is None:
                continue
            try:
                dock = QtWidgets.QDockWidget(d.display_label())
                dock.setFeatures(
                    QtWidgets.QDockWidget.DockWidgetMovable
                    | QtWidgets.QDockWidget.DockWidgetFloatable
                    | QtWidgets.QDockWidget.DockWidgetClosable
                )
                dock.setWidget(panel.widget)
                self._win.window.addDockWidget(
                    QtCore.Qt.RightDockWidgetArea, dock,
                )
                self._diagram_docks[d] = (dock, panel)
            except Exception as exc:
                import sys
                print(
                    f"[ResultsViewer] could not dock side panel: {exc}",
                    file=sys.stderr,
                )
