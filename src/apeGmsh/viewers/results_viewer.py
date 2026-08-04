"""ResultsViewer — the post-solve interactive viewer.

Opens against a :class:`Results` instance and renders the bound
``FEMData`` mesh.

Parallel to :class:`MeshViewer` (pre-solve) and :class:`ModelViewer`
(BRep geometry). Reuses the same ``viewers/scene/``, ``viewers/core/``,
and ``viewers/ui/`` infrastructure where possible.

Phase 8 (ADR 0020 INV-1) — :class:`Results` carries the
:class:`OpenSeesModel` natively via :attr:`Results.model` (always
non-None post-prune).  The viewer reads structural data from the
chain-forward handle; cuts auto-load and orientation auto-resolve
are gated on ``results.model`` (always populated).

Usage::

    from apeGmsh import Results
    from apeGmsh.opensees import OpenSeesModel

    model = OpenSeesModel.from_h5("model.h5")
    results = Results.from_native("run.h5", model=model)
    results.viewer()                       # blocks until window closes
    results.viewer(blocking=False)         # subprocess
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

# ADR 0084 D7 (PR 7) — the dispatcher pumps and their module-level
# helpers now live in ``_pump_set``; re-exported here because the
# existing import path (``results_viewer._compose_substrate_points`` &
# co.) is what the ADR 0058 tests and the deform fan-out use.
from ._pump_set import (  # noqa: F401
    PumpSet,
    _compose_substrate_points,
    _gate_visible_layer_ids,
    _pump_failed,
    sync_render_surface_points,
)

if TYPE_CHECKING:
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.results.Results import Results
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


def _ensure_qapplication():
    """Return the process ``QApplication``, creating one if absent.

    The viewer builds QWidgets (the Output and Color-Map docks) *before*
    :class:`ViewerWindow` — historically the first and only place a
    ``QApplication`` was created. Constructing any QWidget with no live
    ``QApplication`` aborts the process with::

        QWidget: Must construct a QApplication before a QWidget

    which is fatal in the ``python -m apeGmsh.viewers`` subprocess
    (``viewer(blocking=False)``) and in any in-process blocking
    ``viewer()`` call made before a ``QApplication`` exists. Calling
    this at the top of the launch path guarantees the invariant;
    :class:`ViewerWindow`'s own ``instance() or QApplication([])``
    then simply reuses the returned instance (and still runs the
    event loop unconditionally).

    The platform guard must run *here*, not just in
    :func:`ViewerWindow._lazy_qt`: Qt reads ``QT_QPA_PLATFORM`` /
    ``QT_STYLE_OVERRIDE`` once, at ``QApplication`` construction, and
    on this path that construction happens below — before
    ``ViewerWindow`` ever loads. Without it, Linux Wayland sessions
    crash in VTK's X layer (BadWindow on X_ConfigureWindow).
    """
    from .ui._qt_env import prepare_qt_environment
    prepare_qt_environment()
    from qtpy import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _geometries_occluded_by_diagrams(director: Any) -> "set":
    """Geometry ids owning an attached + visible diagram that paints
    an opaque substrate fill (``occludes_substrate``), e.g. a contour.

    The viewer hides each such geometry grey substrate fill (keeping
    the wireframe) so the contour and the coincident fill do not
    z-fight. Module-level (not a closure) so the truth table is
    testable headless.
    """
    geom_mgr = director.geometries
    out: set = set()
    try:
        diagrams = director.registry.diagrams()
    except Exception:
        return out
    for d in diagrams:
        if not getattr(d, "is_attached", False):
            continue
        if not getattr(d, "is_visible", False):
            continue
        if not getattr(d, "occludes_substrate", False):
            continue
        owner = geom_mgr.geometry_for_layer(d)
        if owner is not None:
            out.add(owner.id)
    return out


def add_substrate_actors(
    plotter: Any,
    g_scene: "FEMSceneData",
    *,
    palette: Any,
    prefs: Any,
    clip_planes: "Sequence[Any]" = (),
    name_suffix: str = "",
    reset_camera: bool = False,
) -> tuple:
    """Build one geometry's substrate fill + wireframe pair.

    ADR 0058 S2a. The fill is drawn first; edges render separately
    (``style='wireframe'``) so the lines stay on top of any contour /
    deformed-shape diagram that draws polygons at the same z-depth.
    Polygon-offset on the line mapper resolves coincident topology by
    pulling lines toward the camera, eliminating z-fighting between the
    wireframe and the substrate / diagram polygons. Actor names are
    unique per geometry — pyvista replaces same-named actors.

    ADR 0083 Part 3: the substrate is **off the seam** (added straight
    to the plotter, never through ``add_layer``), so the section-plane
    cut is applied here, at creation — which is what covers the second
    call site, the per-geometry materialization that runs long after
    the user set up the cut. Module-level for the same reason
    :func:`_compose_substrate_points` is: so both properties are
    testable without a window.

    Render-surface fast path: both actors map ONE pre-extracted
    surface of ``g_scene.grid`` (stashed as ``g_scene.render_surface``)
    rather than the grid itself — ``vtkDataSetMapper`` re-extracts an
    O(volume) surface on every grid MTime bump, which made each deform
    step pay twice (fill + wireframe) for the whole model. The grid
    stays the volumetric source of truth; the DEFORM pump scatters
    deformed points onto the surface via
    :func:`sync_render_surface_points`, and ghost (per-cell
    visibility) changes re-extract it. Falls back to the grid when
    extraction cannot provide the scatter maps.
    """
    from .backends.pyvista_qt import apply_clip_planes, build_render_surface

    rs = getattr(g_scene, "render_surface", None)
    if rs is None:
        rs = build_render_surface(g_scene.grid)
        if rs is not None:
            g_scene.render_surface = rs
    substrate = rs.surface if rs is not None else g_scene.grid

    fill = plotter.add_mesh(
        substrate,
        color=palette.substrate_color,
        show_edges=False,
        opacity=prefs.mesh_surface_opacity,
        lighting=True,
        smooth_shading=False,
        name=f"results_substrate{name_suffix}",
        reset_camera=reset_camera,
    )
    wf = plotter.add_mesh(
        substrate,
        style="wireframe",
        color=palette.substrate_edge_color,
        line_width=prefs.mesh_line_width,
        opacity=1.0,
        lighting=False,
        pickable=False,
        name=f"results_wireframe{name_suffix}",
    )
    try:
        wf_mapper = wf.GetMapper()
        wf_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        wf_mapper.SetResolveCoincidentTopologyLineOffsetParameters(
            -1.0, -1.0,
        )
    except Exception:
        pass
    for actor in (fill, wf):
        apply_clip_planes(actor, clip_planes)
    return fill, wf


def add_node_cloud_actor(
    plotter: Any,
    g_scene: "FEMSceneData",
    *,
    marker_size: float,
    color: Any,
    clip_planes: "Sequence[Any]" = (),
) -> Any:
    """Build the node-cloud overlay actor, cut by ``clip_planes``.

    The node cloud is off the seam like the substrate, and is rebuilt
    from scratch whenever the point size changes — so the cut is
    applied inside this helper rather than at either call site
    (ADR 0083 Part 3). A clipped node renders as a half-sprite at the
    plane; excluding whole nodes by centre would misreport which nodes
    exist near the cut. Returns ``None`` if the cloud cannot be built.
    """
    from .backends.pyvista_qt import apply_clip_planes
    from .scene.glyph_points import build_node_cloud

    try:
        _, actor = build_node_cloud(
            plotter,
            g_scene.grid.points,
            model_diagonal=g_scene.model_diagonal,
            marker_size=float(marker_size),
            color=color,
        )
    except Exception:
        return None
    apply_clip_planes(actor, clip_planes)
    return actor


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
        cuts: "Optional[Sequence[SectionCutDef]]" = None,
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
        # Whether _on_close closes the bound Results' HDF5 handle. True
        # for the interactive viewer (it owns the file for its lifetime);
        # set False by the headless export path, which borrows the
        # caller's live Results and must leave it usable afterwards.
        self._own_results_close = True
        # Re-entrancy guard for the interactive export handler — the
        # capture loop pumps the Qt event queue, so a queued click must
        # not launch a nested export.
        self._exporting = False
        # Section cuts to wire in at boot. The director constructs in
        # ``show()`` — these are queued until then and applied right
        # after the registry is bound, so the cut Layers attach against
        # a live plotter + scene like any other diagram added at boot.
        self._pending_cuts: tuple = tuple(cuts) if cuts else ()

        # Populated in show()
        self._director: "ResultsDirector | None" = None
        # ADR 0058 S2a — the scene built at show() for the boot
        # geometry. Construction-time code (before the director binds)
        # reads this directly; everything display-level goes through
        # the ``_scene`` property (the ACTIVE geometry's scene).
        self._boot_scene: "FEMSceneData | None" = None
        self._win: Any = None
        self._plotter: Any = None
        self._settings_tab: "DiagramSettingsTab | None" = None
        self._color_editor: Any = None
        self._color_editor_action: Any = None
        self._registry_unsub: "Optional[Callable[[], None]]" = None
        self._step_unsub: "Optional[Callable[[], None]]" = None
        self._stage_unsub: "Optional[Callable[[], None]]" = None
        self._legend_pref_unsub: "Optional[Callable[[], None]]" = None
        # Output dock + log router. Constructed lazily in _show_impl
        # so headless usage (Results.from_native + queries) doesn't
        # pull Qt. Lifecycle:
        # - router.install() before window construction
        # - dock mounted via extension_docks=[spec]
        # - router.uninstall() in _on_close
        self._log_router: Any = None
        self._output_dock: Any = None
        self._output_badge: Any = None
        # Plan 04 step 2 — per-viewer ActiveObjects coordinator.
        # Initialised in _show_impl after the window so it can parent
        # to win.window for Qt's GC. Panels subscribe to its signals
        # rather than wiring direct callbacks to each other.
        self._active: Any = None
        self._time_scrubber: Any = None
        self._substrate_actor: Any = None
        self._wireframe_actor: Any = None
        self._node_cloud_actor: Any = None
        self._node_label_actor: Any = None
        self._element_label_actor: Any = None
        self._plot_pane: Any = None
        self._details_panel: Any = None
        self._geometry_panel: Any = None
        self._session_panel: Any = None
        self._definitions_panel: Any = None
        self._clip_planes_panel: Any = None
        self._clip_gizmos: Any = None
        self._clip_gizmo_interactor: Any = None
        self._clip_cut_faces: Any = None
        self._scope_gizmos: Any = None
        self._scope_gizmo_interactor: Any = None
        # diagram instance -> side panel; lifecycle tied to registry.
        self._diagram_side_panels: dict = {}
        # (node_id, component, stage_id) -> TimeHistoryPanel; user-
        # closable from the plot-pane tab × button. ``stage_id`` is the
        # pick's stage pin (None = active stage; ADR 0058 S3b).
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
    def _scene(self) -> "Optional[FEMSceneData]":
        """The ACTIVE geometry's substrate scene (ADR 0058 S2a).

        Every display-level consumer (status line, label overlays,
        probe radius, pick-extract, node cloud) reads ``self._scene``
        — making it a property over ``director.scene_for(active)``
        keeps them all correct unchanged when scenes become
        per-geometry. Falls back to the boot scene before the director
        binds (construction-time reads) and to ``None`` pre-show.
        """
        director = getattr(self, "_director", None)
        if director is not None:
            try:
                scene = director.scene_for(director.geometries.active)
            except Exception:
                scene = None
            if scene is not None:
                return scene
        return getattr(self, "_boot_scene", None)

    @property
    def plotter(self) -> Any:
        return self._plotter

    # ------------------------------------------------------------------
    # Animation export
    # ------------------------------------------------------------------

    def export_animation(
        self,
        path: "str | Any",
        *,
        fps: int = 30,
        step_stride: int = 1,
        progress: "Optional[Callable[[int, int], None]]" = None,
    ) -> Any:
        """Export the time history as an animated MP4 or GIF.

        Format auto-detected from the path suffix (``.mp4`` / ``.gif``).
        See :func:`apeGmsh.viewers.animation.export_animation` for the
        full parameter documentation. Requires the viewer's plotter and
        director to be wired — either via :meth:`show` (interactive) or
        :meth:`show` with ``run_loop=False`` (headless export).
        """
        if self._plotter is None or self._director is None:
            raise RuntimeError(
                "export_animation: call viewer.show() first so the "
                "plotter and director are constructed."
            )
        from .animation import export_animation
        return export_animation(
            self._plotter, self._director, path,
            fps=fps, step_stride=step_stride, progress=progress,
        )

    def _export_animation_interactive(
        self, path: Any, *, fps: int = 30, step_stride: int = 1,
    ) -> None:
        """Run an animation export from the Time Scrubber's Export button.

        Wraps :meth:`export_animation` with a modal progress dialog (the
        capture loop is synchronous and runs on the UI thread; pumping
        events from the progress callback keeps the dialog responsive
        and lets the user cancel). Success / failure / cancel is
        surfaced on the status bar. Cancellation is implemented by
        raising out of the progress callback, which the engine's
        ``finally`` turns into a restored step + a partial file we
        delete.
        """
        # Re-entrancy guard: the capture loop pumps the event queue, so a
        # queued double-click on Export (before the modal grab is fully
        # active) could otherwise launch a nested export.
        if self._exporting:
            return
        self._exporting = True
        scrubber = self._time_scrubber
        if scrubber is not None:
            scrubber.set_export_enabled(False)

        from qtpy import QtWidgets, QtCore
        from pathlib import Path

        out = Path(path)
        win = self._win
        n_steps = int(self._director.n_steps) if self._director else 0
        stride = max(1, int(step_stride))
        # Mirror the engine's frame schedule for an accurate total
        # (the engine always appends the final step if stride skipped
        # it, so add one when the last index isn't already on schedule).
        indices = list(range(0, n_steps, stride))
        if not indices or indices[-1] != n_steps - 1:
            indices.append(n_steps - 1)
        total = max(1, len(indices))

        dlg = QtWidgets.QProgressDialog(
            f"Exporting {out.name} …", "Cancel", 0, total, win.window,
        )
        dlg.setWindowTitle("Export animation")
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        dlg.setValue(0)

        class _Cancelled(Exception):
            pass

        def _progress(done: int, count: int) -> None:
            dlg.setMaximum(count)
            dlg.setValue(done)
            QtWidgets.QApplication.processEvents()
            if dlg.wasCanceled():
                raise _Cancelled()

        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor,
        )
        try:
            self.export_animation(
                out, fps=fps, step_stride=stride, progress=_progress,
            )
            win.set_status(f"Exported animation → {out}", timeout=6000)
        except _Cancelled:
            # Remove the half-written file so a cancel leaves no
            # corrupt artifact behind.
            try:
                out.unlink(missing_ok=True)
            except Exception:
                pass
            win.set_status("Export cancelled", timeout=4000)
        except Exception as exc:
            from ._log import log_error
            log_error("export", "export_animation", exc)
            win.set_status(
                f"Export failed: {type(exc).__name__}: {exc}", timeout=8000,
            )
            QtWidgets.QMessageBox.warning(
                win.window, "Export failed",
                f"Could not export the animation:\n\n{exc}",
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            dlg.close()
            self._exporting = False
            if scrubber is not None:
                scrubber.set_export_enabled(True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def show(
        self,
        *,
        maximized: bool = True,
        run_loop: bool = True,
        window_size: "Optional[tuple[int, int]]" = None,
        enter_loop: bool = True,
    ):
        """Open the viewer window and run the Qt event loop until close.

        Returns ``self`` so callers can introspect the viewer state
        after the window closes.

        When ``run_loop`` is ``False``, the full window + scene are
        built and the render surface is realized (so screenshots work),
        but the blocking ``QApplication`` event loop is **not** entered
        — :meth:`show` returns immediately with the viewer live. This is
        the headless-export path (see :meth:`Results.export_animation`);
        the caller must drive the viewer and call :meth:`close` when
        done.

        ``enter_loop=False`` instead *presents* the window (shown, raised,
        rendered — full chrome) but does not call ``app.exec_()`` — used
        when a Qt event loop is already running and this window should
        join it (File → Open Results… spawning a second viewer). The call
        returns immediately; the viewer stays alive via the module
        ``_LIVE_VIEWERS`` pin.
        """
        # Pin a strong ref so the window survives even when the kernel
        # has a Qt event loop already running and ``app.exec_()``
        # returns immediately (jupyter ``%gui qt`` and friends).
        _LIVE_VIEWERS.add(self)

        # Initialise the action logger early so anything that fires
        # during construction (NoDataError on attach, FEM bind issues,
        # ...) lands in the session log file. session.start is the
        # logger's own header; this line records *which* file we just
        # opened so the log is self-contained.
        from ._log import log_action, log_error
        results_path = getattr(self._results, "_path", None)  # noqa: SLF001
        log_action(
            "session", "open",
            file=str(results_path) if results_path else "<in-memory>",
        )

        try:
            return self._show_impl(
                maximized=maximized, run_loop=run_loop,
                window_size=window_size, enter_loop=enter_loop,
            )
        except BaseException as exc:
            # Anything that escapes ``_show_impl`` — ResultsWindow init
            # failures, VTK render-window pixel-format errors, restore
            # path crashes, Qt resource exhaustion, KeyboardInterrupt —
            # writes to the session log file with full traceback before
            # propagating. The log is on disk by now (session.start
            # flushed); even if the calling terminal closes the user
            # can pull the file from ~/.apegmsh/viewer-logs/ to see
            # what happened. Without this trap the trace went to
            # stderr only and was easy to lose.
            log_error("init", "ResultsViewer.show", exc)
            raise

    # ------------------------------------------------------------------
    # File menu — Open Results…
    # ------------------------------------------------------------------

    def _install_file_menu(self, win: Any) -> None:
        """Add a leftmost **File** menu with an *Open Results…* action.

        Mirrors the File-menu pattern in
        :class:`apeGmsh.viewers.model_viewer.ModelViewer`. Best-effort —
        a menubar failure never blocks the viewer from opening.
        """
        try:
            from qtpy import QtWidgets
            file_menu = QtWidgets.QMenu("File", win.window)
            file_menu.addAction("Open Results…").triggered.connect(
                self._on_open_results
            )
            mb = win.window.menuBar()
            acts = mb.actions()
            if acts:
                mb.insertMenu(acts[0], file_menu)   # File leftmost
            else:
                mb.addMenu(file_menu)
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._install_file_menu", exc)

    def _on_open_results(self) -> None:
        """File → Open Results… — pick a file and open it in a new window."""
        from .ui._open_results import run_open_dialog
        parent = self._win.window if self._win is not None else None
        run_open_dialog(parent, self._open_in_new_window)

    def _open_in_new_window(self, results: "Results") -> None:
        """Show ``results`` in a fresh viewer window beside this one.

        ``enter_loop=False`` so the new window joins the already-running
        Qt event loop instead of nesting a second ``app.exec_()``. The
        new :class:`ResultsViewer` pins itself in ``_LIVE_VIEWERS`` for
        the duration of its window, so it survives this method returning.
        """
        ResultsViewer(results).show(enter_loop=False)

    def _build_viewer_data(self):
        """Build the :class:`ViewerData` scene snapshot.

        Phase 8 (ADR 0020 INV-1) — :attr:`Results.model` is always
        non-None.  When ``results._path`` carries ``/opensees/``
        (Composed-file pattern, the typical Phase-8 landing), build
        from the file via :meth:`ViewerData.from_h5`; otherwise fall
        back to the live :meth:`ViewerData.from_fem` path (degraded
        orientation per ADR 0018 INV-11).  The "should we read from
        the file?" predicate is centralised in
        :func:`resolve_orientation_source` — see ADR 0026.
        """
        from .data import ViewerData
        from .data._h5_probe import resolve_orientation_source
        source = resolve_orientation_source(self._results)
        if source is not None:
            return ViewerData.from_h5(str(source))
        return ViewerData.from_fem(self._results.fem)

    def _show_impl(
        self,
        *,
        maximized: bool = True,
        run_loop: bool = True,
        window_size: "Optional[tuple[int, int]]" = None,
        enter_loop: bool = True,
    ):
        """The actual show() body — see :meth:`show` for the trap wrapper."""
        # Lazy imports — keep ``apeGmsh.viewers`` importable in headless
        # environments. Qt / pyvistaqt only loaded when the user opens
        # an actual viewer.
        from .data import ViewerData
        from .scene.fem_scene import build_fem_scene
        from .diagrams._director import ResultsDirector
        from .ui._results_window import ResultsWindow
        from .ui._diagram_settings_tab import DiagramSettingsTab
        from .ui.preferences_manager import PREFERENCES as _PREF

        # ── QApplication ────────────────────────────────────────────
        # Must exist before the Output / Color-Map dock QWidgets built
        # below; ViewerWindow (which used to create it) is too late.
        # See _ensure_qapplication for the failure mode it prevents.
        _app = _ensure_qapplication()  # noqa: F841 — strong ref for this frame

        # ── Director ────────────────────────────────────────────────
        director = ResultsDirector(self._results)
        self._director = director

        # ── FEM scene ───────────────────────────────────────────────
        # Phase 8 (ADR 0020 INV-1) — ``results.model`` is always
        # populated; the file-mediated read via
        # :meth:`ViewerData.from_h5` is the primary structural-
        # enrichment path (ADR 0014 INV-2 — the viewer never imports
        # ``OpenSeesModel`` directly; it consumes through
        # ``h5_reader``).  Falls back to :meth:`ViewerData.from_fem`
        # when the results file lacks the ``/opensees/`` zone (ADR
        # 0018 INV-11 graceful degrade).
        #
        # Producer-agnostic — ``apeSees(fem).h5()`` and
        # ``ModelData(fem).write()`` produce byte-equivalent zones
        # (ADR 0018 INV-16) so this seam covers both.
        fem = self._results.fem
        assert fem is not None    # validated in __init__
        view = self._build_viewer_data()
        scene = build_fem_scene(view)
        self._boot_scene = scene
        # Pick actor inventory — set on the scene before any diagram
        # attaches so GaussPointDiagram (and future fiber/etc.) can
        # register their actor in their own attach() instead of the
        # picker having to walk every active diagram on every click.
        from .core.results_pick_engine import PickInventory as _PickInventory
        scene.pick_engine = _PickInventory()
        # ElementVisibility — per-cell hide via the substrate
        # ``vtkGhostType`` array. Box-pick consults the ghost mask,
        # renderers / VTK pickers natively skip ghost-hidden cells.
        from .core.element_visibility import ElementVisibility as _ElementVis
        scene.element_visibility = _ElementVis(scene.grid)

        # ── Output dock + log router (before the window so the
        #    LogRouter can capture exceptions raised during window
        #    construction itself — e.g. a Qt error inside
        #    _build_layout). The dock is mounted as an extension dock
        #    so it picks up the View menu toggle + layout persistence
        #    machinery added by plan 08 step 2.
        from .ui._log_router import LogRouter
        from .ui._output_dock import make_output_dock
        log_router = LogRouter()
        log_router.install()
        self._log_router = log_router
        output_dock, output_spec = make_output_dock(log_router)
        self._output_dock = output_dock

        # ── Plan 06 step 4 — Color Map Editor extension dock ────────
        # Constructed up-front so the spec can be passed alongside the
        # Output spec to ``ResultsWindow``. Hidden by default — surfaced
        # via the View menu. Binding to the active layer happens after
        # ``self._active`` is constructed (below).
        from .ui._color_map_editor import make_color_map_editor_dock
        color_editor, color_editor_spec = make_color_map_editor_dock()
        self._color_editor = color_editor

        # ── Definitions extension dock — bridge-side named primitives ──
        # Lists the model's ops.<family>.<Type>(..., name=…) aliases by
        # kind+tag, fed from the ViewerData read seam (H5Model.names).
        # Hidden by default; surfaced via the View menu like the other
        # extension docks. Empty (idle hint) for live-FEM snapshots.
        from .ui._definitions_panel import make_definitions_dock
        definitions_panel, definitions_spec = make_definitions_dock(view)
        self._definitions_panel = definitions_panel

        # ── Section planes extension dock (ADR 0083 Part 4) ─────────
        # Built up-front like the others; bound to its controller after
        # ``bind_plotter`` (the controller is keyed by the backend,
        # which does not exist until then). Tabified with the Diagram
        # dock — a plane is a view annotation, and that is where the
        # other per-view knobs live.
        from .ui._clip_planes_panel import make_clip_planes_dock
        clip_panel, clip_spec = make_clip_planes_dock(
            tabify_with=ResultsWindow.DOCK_DIAGRAM,
        )
        self._clip_planes_panel = clip_panel

        # ── Window (creates QApplication) ───────────────────────────
        title = self._title or self._default_title()
        win = ResultsWindow(
            title=title,
            on_close=self._on_close,
            extension_docks=[
                output_spec, color_editor_spec, definitions_spec, clip_spec,
            ],
        )
        self._win = win

        # ── File menu (Open Results…) ───────────────────────────────
        self._install_file_menu(win)

        # ── Plan 04 step 2 — ActiveObjects coordinator ──────────────
        # Single source of truth for "which composition / geometry /
        # layer is currently active." Panels subscribe to its signals
        # instead of wiring direct callbacks. Parented to win.window
        # so Qt's parent-tracked GC keeps it alive for the viewer's
        # lifetime.
        from .core._active_objects import ActiveObjects
        self._active = ActiveObjects(parent=win.window)

        # ── Status-bar Output badge ─────────────────────────────────
        # Surfaces warning/error counts before the user opens the
        # dock; clicking it raises the dock. Lives in the status bar
        # as a permanent widget. Hidden when counts are zero.
        try:
            from .ui._output_badge import OutputBadge
            output_dock_widget = win.extension_dock("dock_output")
            badge = OutputBadge(output_dock, output_dock_widget)
            self._output_badge = badge
            win.window.statusBar().addPermanentWidget(badge.widget)
        except Exception:
            # Badge is purely peripheral — never let it block viewer
            # startup. The dock + log router work fine without it.
            self._output_badge = None

        # ── Plan 02 — toolbar extensibility demo ────────────────────
        # The color-map editor dock is hidden by default (its
        # discoverability story was originally "find it in the View
        # menu"). Plan 02's extensibility hook lets us add a
        # discoverable toolbar button that toggles the dock open
        # without modifying ``ViewerWindow``'s chrome class. This
        # button is the canonical example of the new API; future
        # diagrams / overlays follow the same registration pattern.
        try:
            color_dock_widget = win.extension_dock("dock_color_map_editor")
        except Exception:
            color_dock_widget = None
        if color_dock_widget is not None:
            self._color_editor_action = win.add_toolbar_action(
                "Color map editor",
                "▩",     # squared diagonal — close enough to "palette"
                lambda checked: color_dock_widget.setVisible(bool(checked)),
                checkable=True,
                triggered_signal="toggled",
            )
            # Keep the button's checked state aligned with the dock —
            # closing the dock via its own X must un-check the button.
            try:
                color_dock_widget.visibilityChanged.connect(
                    lambda visible: self._color_editor_action.setChecked(
                        bool(visible),
                    ),
                )
            except Exception:
                pass
        else:
            self._color_editor_action = None

        # ── Section-planes toolbar button (ADR 0083 Part 4) ─────────
        # Same shape as the colour-map button above: the dock is hidden
        # by default, so it needs a discoverable way in.
        try:
            clip_dock_widget = win.extension_dock("dock_results_clip_planes")
        except Exception:
            clip_dock_widget = None
        if clip_dock_widget is not None:
            from .ui._dock_registry import wire_tabified_dock_action

            # NOT the colour-map wiring: this dock is tabified behind
            # Diagram, where the naive setVisible/setChecked pair is a
            # loop that closes the dock on the click that opened it
            # (S1 review finding B1). The helper shows AND raises.
            self._clip_planes_action = win.add_toolbar_action(
                "Section planes",
                "⧄",     # square with upper-left to lower-right fill
                lambda _checked: None,  # wired below, with the guard
                checkable=True,
                triggered_signal="toggled",
            )
            wire_tabified_dock_action(
                self._clip_planes_action, clip_dock_widget,
            )
        else:
            self._clip_planes_action = None

        # ProbeOverlay needs the plotter, which doesn't exist until
        # ``win`` is constructed below. We initialise to None here so
        # any early access path resolves cleanly; the real overlay is
        # built after ``bind_plotter`` and feeds the viewport HUD.
        self._probe_overlay: Any = None

        # Local-axes overlay: built after ``bind_plotter`` (needs the
        # plotter + scene). Wire the toolbar toggle now; its lambda
        # guards on the None until the real overlay exists.
        self._local_axes_overlay: Any = None
        try:
            self._local_axes_action = win.add_toolbar_action(
                "Local axes",
                "⌖",
                lambda checked: (
                    self._local_axes_overlay.set_visible(bool(checked))
                    if self._local_axes_overlay is not None else None
                ),
                checkable=True,
                triggered_signal="toggled",
            )
        except Exception:
            self._local_axes_action = None

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

        # ── Right rail (plot pane + dedicated Diagram / Geometry /
        #               Details / Session docks) ───────────────────
        from .ui._plot_pane import PlotPane
        from .ui._details_panel import DetailsPanel
        plot_pane = PlotPane()
        plot_pane.on_user_close(self._on_plot_user_close)
        # ── Geometry settings panel ───────────────────────────────
        # Available-fields detection runs once against the union of
        # every stage's nodal components.
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
        # ADR 0084 D1 — the Threshold section wants the raw SCALAR
        # component names (the Add Diagram dialog's list), not the
        # vector-prefix list the Deformation section uses.
        try:
            threshold_components = {
                "nodes": _uas(director, "nodes"),
                "gauss": _uas(director, "gauss"),
            }
        except Exception:
            threshold_components = {}
        geometry_panel = GeometrySettingsPanel(
            director, deform_field_options, threshold_components,
        )
        # DetailsPanel is now a near-empty placeholder for future
        # canvas-click contextual content (contour scale edits, picked
        # node readouts, …). The diagram / geometry editors live in
        # their own dedicated docks.
        details = DetailsPanel(settings_tab, geometry_panel)
        # ── Outline tree → ActiveObjects (plan 04 step 2) ────────────
        # The outline's row-clicked callbacks now feed ActiveObjects
        # state instead of invoking panel methods directly. Multiple
        # subscribers (settings tab, geometry panel, status indicators,
        # future overlays) can react to the same state change.
        #
        # Pattern: outline → set_active_X → activeXChanged signal →
        # whatever's subscribed. The handlers below mirror the old
        # behaviour exactly so this is a transparent refactor.
        outline.on_composition_selected(
            lambda key: self._active.set_active_composition(key),
        )
        outline.on_geometry_selected(
            lambda gid: self._active.set_active_geometry(gid),
        )
        # Plan 03 v2 — outline Layer-row selection drives active_layer
        # alongside the existing settings-tab card-focus path.
        outline.on_diagram_selected(
            lambda layer: self._active.set_active_layer(layer),
        )
        self._active.activeCompositionChanged.connect(
            self._on_active_composition_changed,
        )
        self._active.activeGeometryChanged.connect(
            self._on_active_geometry_changed,
        )
        # ── Plan 06 step 4 — Color Map Editor follows the active layer ──
        # When the active layer changes (driven from the composition
        # handler below), the editor rebinds. ``set_active_layer(None)``
        # collapses the editor to the empty state.
        self._active.activeLayerChanged.connect(self._color_editor.bind_layer)
        # Registry changes (layer added / removed / visibility) may
        # invalidate the currently-active layer or surface a new
        # default one. Re-evaluate when the registry fires.
        self._registry_unsub = director.registry.subscribe(
            self._refresh_active_layer,
        )
        # Plan 04 step 2 cont. — layer card focus drives active_layer.
        # Clicking a specific card in the diagram dock binds the editor
        # to *that* card, not just the first contour layer in the
        # composition. The composition-default path stays as the
        # fallback for plain navigation.
        settings_tab.on_layer_focused(self._active.set_active_layer)
        # Plan 04 step 2 cont. — director step + stage observers route
        # through ActiveObjects so every panel that needs to react to
        # time-scrubber or stage-tab changes can subscribe to the
        # corresponding ``active*Changed`` signal instead of wiring a
        # bespoke director subscription. The bridge callbacks are
        # registered as direct director subscribers so they fire
        # synchronously after the director's own update fan-out,
        # preserving ordering relative to existing observers.
        self._step_unsub = director.subscribe_step(
            lambda step: self._active.set_active_step(int(step)),
        )
        self._stage_unsub = director.subscribe_stage(
            lambda stage_id: self._active.set_active_stage(stage_id),
        )
        # Seed the projection from the owner (ADR 0056 INV-1): the
        # director auto-picks a stage/step at __init__ — before this
        # bridge exists — so on single-stage results no change event
        # ever fires and ActiveObjects would hold None forever.
        if director.stage_id is not None:
            self._active.set_active_stage(director.stage_id)
        self._active.set_active_step(int(director.step_index))
        # Two-way binding (B++ §7): the Plots group in the outline
        # tree mirrors the plot pane's tab list; clicking a plot row
        # activates the corresponding tab and vice versa.
        outline.bind_plot_pane(plot_pane)
        win.set_right_widget(plot_pane.widget)
        win.set_diagram_widget(settings_tab.widget)
        win.set_geometry_widget(geometry_panel.widget)
        win.set_details_widget(details.widget)
        self._plot_pane = plot_pane
        self._details_panel = details
        self._geometry_panel = geometry_panel

        # ── Plotter — substrate mesh ────────────────────────────────
        plotter = win.plotter
        self._plotter = plotter
        # OpacityController — per-actor SetOpacity + depth-peel
        # auto-toggle. Constructed now (after the plotter exists) and
        # stashed on the scene so callers that already hold a scene
        # ref can route through it.
        from .core.opacity_controller import OpacityController as _OpacityCtrl
        scene.opacity_controller = _OpacityCtrl(plotter)

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
        )
        win.set_session_widget(session.widget)
        self._session_panel = session

        # ── Substrate actor pair builder (ADR 0058 S2a) ────────────
        # Thin closure over the module-level builder: called once here
        # for the boot scene and again from the scene factory for every
        # materialized per-geometry scene. The section-plane cut is
        # applied inside the builder (ADR 0083 Part 3), so a geometry
        # materialized while a cut is live arrives already clipped.
        def _add_substrate_actors(
            g_scene: "FEMSceneData",
            *,
            name_suffix: str = "",
            reset_camera: bool = False,
        ):
            return add_substrate_actors(
                plotter, g_scene,
                palette=THEME.current,
                prefs=prefs,
                clip_planes=self._clip_plane_specs(),
                name_suffix=name_suffix,
                reset_camera=reset_camera,
            )

        actor, wireframe_actor = _add_substrate_actors(
            scene, reset_camera=True,
        )
        self._substrate_actor = actor
        self._wireframe_actor = wireframe_actor

        # ── Node-cloud overlay (matches the pre-solve mesh viewer) ─
        # One flat GL point per FEM node, drawn over the substrate so
        # the user can see the discretization.
        node_actor = add_node_cloud_actor(
            plotter, scene,
            marker_size=prefs.point_size,
            color=palette.node_accent,
            clip_planes=self._clip_plane_specs(),
        )
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

        def _apply_geometry_display() -> None:
            """Push per-geometry display state to substrate actors.

            ADR 0058 S2b — concurrent rendering: every geometry with a
            materialized actor pair gets its own state, not just the
            active one:

            * substrate fill + wireframe — visibility is
              ``geometry.visible AND geometry.show_mesh``; opacity
              tracks that geometry's ``display_opacity`` (the
              substrate fill keeps its preferences-driven baseline
              alpha multiplied by ``display_opacity`` so a user who
              starts at 80% baseline and dials the slider to 50%
              ends up at 40%).
            * node cloud — an active-only editing overlay: visibility
              is the ACTIVE geometry's ``visible AND show_nodes``;
              opacity tracks its ``display_opacity``.

            Idempotent — safe to fire on every ``geometries`` change.
            """
            if self._director is None:
                return
            geom_mgr = self._director.geometries
            pairs = getattr(self, "_scene_actors", None) or {}
            # Geometries that own at least one attached + visible
            # diagram which paints an opaque substrate fill (contour).
            # Their grey substrate fill is hidden so the two coincident
            # opaque surfaces do not z-fight (grey bleeding through the
            # colour). The wireframe stays so element edges remain
            # visible. Computed once per sync from the live registry.
            occluding = _geometries_occluded_by_diagrams(self._director)
            for gid, (fill, wf) in list(pairs.items()):
                pair_geom = geom_mgr.find(gid)
                if pair_geom is None:
                    continue
                mesh_v = 1 if (
                    pair_geom.visible and pair_geom.show_mesh
                ) else 0
                # The fill drops only where an occluding diagram covers
                # it; the wireframe keeps mesh_v so edges stay on top.
                fill_v = 0 if (gid in occluding) else mesh_v
                opacity = max(
                    0.0, min(1.0, float(pair_geom.display_opacity)),
                )
                substrate_alpha = (
                    float(prefs.mesh_surface_opacity) * opacity
                )
                try:
                    fill.SetVisibility(fill_v)
                    fill.GetProperty().SetOpacity(substrate_alpha)
                    wf.SetVisibility(mesh_v)
                    wf.GetProperty().SetOpacity(opacity)
                except Exception:
                    pass
            geom = geom_mgr.active
            if geom is None:
                return
            nodes_v = 1 if (geom.visible and geom.show_nodes) else 0
            opacity = max(0.0, min(1.0, float(geom.display_opacity)))
            try:
                if self._node_cloud_actor is not None:
                    self._node_cloud_actor.SetVisibility(nodes_v)
                    self._node_cloud_actor.GetProperty().SetOpacity(opacity)
            except Exception:
                pass

        # Stash on self so other call sites (point-size rebuild,
        # session-restore) can reapply state without re-defining
        # the actor walk.
        self._apply_geometry_display = _apply_geometry_display

        def _on_line_width(value: float) -> None:
            if self._wireframe_actor is None:
                return
            try:
                self._wireframe_actor.GetProperty().SetLineWidth(float(value))
            except Exception:
                pass
            _render()

        def _on_point_size(value: float) -> None:
            # Glyph spheres ignore SetPointSize — rebuild the cloud at
            # the new marker_size. Visibility / opacity come from the
            # active geometry (re-applied below) so the rebuild can't
            # drift away from the per-Geometry display state.
            if self._node_cloud_actor is None or plotter is None:
                return
            old = self._node_cloud_actor
            try:
                plotter.remove_actor(old)
            except Exception:
                pass
            # Rebuild against the ACTIVE geometry's scene (ADR 0058
            # S2a — the node cloud is an active-only display overlay).
            g_scene = self._scene or scene
            new_actor = add_node_cloud_actor(
                plotter, g_scene,
                marker_size=float(value),
                color=THEME.current.node_accent,
                clip_planes=self._clip_plane_specs(),
            )
            self._node_cloud_actor = new_actor
            # Re-capture base glyph + centers so deformation sync
            # uses the new actor's coordinates as its reference.
            self._capture_node_cloud_base()
            # Re-apply the active geometry's display state — pushes
            # show_nodes + display_opacity onto the freshly-built actor.
            self._apply_geometry_display()
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
            # show-mesh / show-nodes / opacity moved to the per-Geometry
            # Display section. SessionPanel keeps the genuinely global
            # cosmetics (point/line size, label toggles, theme).
            self._session_panel.set_show_node_ids_callback(_toggle_show_node_ids)
            self._session_panel.set_show_element_ids_callback(
                _toggle_show_element_ids,
            )
            self._session_panel.set_point_size_callback(_on_point_size)
            self._session_panel.set_line_width_callback(_on_line_width)

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
        #     …) — points are scattered from the
        #     deformed substrate via the original-point map.
        # Layers that own non-substrate point geometry (vector glyph
        # source, gauss markers, node cloud) don't follow yet.
        import numpy as _np

        # ADR 0058 S1: the undeformed baseline lives ON the scene
        # (per-scene, captured at build) — this attribute is an alias
        # to the bound scene's copy for the node-cloud paths.
        self._reference_points = scene.reference_points

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

        # Per-(stage, component) column maps into the director
        # visual float16 cache, built lazily on first use so the
        # DEFORM pump slices a float16 row instead of re-reading
        # displacement_x/y/z from HDF5 every frame.
        _deform_visual_cols: dict = {}
        _deform_visual_ids: dict = {}

        def _read_deform_field(
            field: Optional[str], step: int,
            stage_id: Optional[str] = None,
        ) -> Optional[Any]:
            """Return ``(N, 3)`` vector field at ``step`` for one stage.

            Reads ``<field>_x/_y/_z`` for every FEM node aligned to
            ``scene.grid.points``. Pads to 3-D with zeros when an axis
            is missing (e.g. 2-D model with only ``_x`` / ``_y``).
            ``stage_id`` scopes the read (ADR 0058 S3b — a stage-pinned
            geometry reads its PINNED stage); ``None`` keeps the
            active-stage read. Returns ``None`` if no field name was
            given or the read fails.
            """
            sid = stage_id if stage_id is not None else director.stage_id
            if not field or sid is None:
                return None
            try:
                results = self._results.stage(sid)
            except Exception:
                return None
            n = scene.node_ids.size
            out = _np.zeros((n, 3), dtype=_np.float64)
            id_to_idx = _deform_id_to_idx
            any_axis = False
            for axis, suf in enumerate(("x", "y", "z")):
                comp = f"{field}_{suf}"
                slab_ids = None
                slab_vals = None
                # Fast path: slice a float16 row from the cached
                # full-time node slab (column map memoized per
                # (stage, component)). Zero HDF5 on the hot path.
                vs = director.visual_store
                full = None
                if vs is not None:
                    try:
                        full = vs.nodes_slab(results, sid, comp)
                    except Exception:
                        full = None
                if full is not None:
                    key = (sid, comp)
                    cols = _deform_visual_cols.get(key)
                    if cols is None:
                        full_ids = _np.asarray(
                            full.node_ids, dtype=_np.int64,
                        )
                        cols = _np.where(_np.isin(
                            full_ids,
                            _np.asarray(scene.node_ids, dtype=_np.int64),
                        ))[0]
                        _deform_visual_cols[key] = cols
                        _deform_visual_ids[key] = full_ids[cols]
                    step_i = int(step)
                    if (cols.size and 0 <= step_i
                            < int(full.values.shape[0])):
                        slab_ids = _deform_visual_ids[key]
                        slab_vals = _np.asarray(
                            full.values[step_i], dtype=_np.float64,
                        )[cols]
                # Fallback: per-step HDF5 read.
                if slab_ids is None:
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

        # ── Pipeline primitives — see _dispatch.py for the contract ──
        # ADR 0084 D7 (PR 7): the four pump bodies moved verbatim into
        # ``PumpSet`` (``_pump_set.py``) so they are invocable headless,
        # without a window. Everything they used to capture from this
        # frame is now an explicit argument — that list is the frozen
        # seam PR 8/9/10 build on. The dispatcher below is still the
        # only place that composes them into event-driven sequences;
        # don't call them directly from observers — fire an event.
        # ADR 0084 D1 — the scalar threshold's value reader. The
        # director's controller already exists (constructed with an
        # inert reader); this is where it gets the real one, now that
        # the scene's id→row index arrays do. Every geometry's scene
        # clones the bound one with those arrays SHARED
        # (``clone_scene``), so one reader serves them all — the same
        # reason ``_read_deform_field`` above is geometry-agnostic.
        from .core.threshold_controller import SceneValueReader
        director.thresholds.read_values = SceneValueReader(
            director=director, scene=scene,
        )

        pumps = PumpSet(
            director=director,
            scene=scene,
            read_deform_field=_read_deform_field,
            render_geometries=self._render_geometries,
            sync_node_cloud=self._sync_node_cloud,
            sync_diagram_substrate_points=self._sync_diagram_substrate_points,
            thresholds=director.thresholds,
        )

        # Stash for the rest of the file (probe overlay etc. still
        # call _apply_composition_gate by name in some teardown paths).
        self._apply_composition_gate = pumps.pump_gate

        # ── Dispatcher — single-source event pipeline ────────────────
        from .diagrams._dispatch import (
            STEP_CHANGED, DEFORM_CHANGED, STAGE_CHANGED,
            CLIP_PLANES_CHANGED, ELEMENT_VISIBILITY_CHANGED,
            COMP_ACTIVE_CHANGED, DIAGRAM_ATTACHED,
            DIAGRAM_DETACHED, DIAGRAM_MODIFIED,
            LAYER_VISIBILITY_CHANGED, LAYER_REORDERED, PICK_CLEARED,
            GEOMETRIES_CHANGED,
            GEOMETRY_ACTIVE_CHANGED, GEOMETRY_ADDED,
            GEOMETRY_OFFSET_CHANGED, GEOMETRY_SCOPE_CHANGED,
            GEOMETRY_REMOVED, GEOMETRY_STAGE_PIN_CHANGED,
            GEOMETRY_VISIBILITY_CHANGED, SCOPE_GIZMO_CHANGED, Lane,
        )
        # ADR 0056 Part 3: the director constructed its dispatcher at
        # __init__ (no-op pumps); rebind the real pumps now that the
        # plotter / scene / actor list exist.
        dispatcher = director.dispatcher
        dispatcher.bind(
            pump_step=pumps.pump_step,
            pump_deform=pumps.pump_deform,
            pump_gate=pumps.pump_gate,
            pump_restack=pumps.pump_restack,
            render=_render,
        )
        self._dispatcher = dispatcher
        # (ADR 0047 R-D.2b: the PickInventory no longer carries a
        # dispatcher — the dead set_pick_mode / PICK_MODE_CHANGED path is
        # gone.)
        # ElementVisibility gets the dispatcher so hide/show fires ELEMENT_VISIBILITY_CHANGED.
        scene.element_visibility.dispatcher = dispatcher
        # A ghost (per-cell visibility) change is the one substrate
        # update that can't be scattered onto the pre-extracted render
        # surface — hidden cells change WHICH faces exist — so it
        # re-extracts. RENDER lane: it must land in the frame the
        # dispatcher then paints. Covers manual hide/isolate, the dim
        # filter, and stage activation (all write through
        # ElementVisibility, the sole vtkGhostType writer).
        dispatcher.subscribe(
            ELEMENT_VISIBILITY_CHANGED,
            self._refresh_substrate_surfaces,
            lane=Lane.RENDER,
        )
        # And OpacityController for OPACITY_CHANGED.
        if scene.opacity_controller is not None:
            scene.opacity_controller.dispatcher = dispatcher
        # Migrate the outline tree's geometry subscription onto the
        # UI-lane coalesced dispatcher path (replaces the raw
        # ``director.geometries.subscribe`` wiring set in its __init__).
        # The 6.86 ms/event omnibus storm on session restore collapses
        # to one rebuild per Qt tick per granular kind.
        outline.attach_dispatcher(dispatcher)
        # Same migration for the DiagramSettingsTab (per-diagram
        # styling panel) — its stack rebuild also fires on every
        # geometry mutation; UI-lane coalesce collapses storms.
        settings_tab.attach_dispatcher(dispatcher)

        # ── Observer wiring — every callback fires a dispatcher event.
        # Director's existing observers are preserved (the time scrubber
        # subscribes to on_step_changed for cursor sync); the dispatcher
        # is an additional consumer that runs the matrix.
        director.subscribe_step(
            lambda _step: dispatcher.fire(STEP_CHANGED),
        )
        director.subscribe_stage(
            lambda _sid: dispatcher.fire(STAGE_CHANGED),
        )
        # Geometry-state covers: deform toggle/scale/field, active
        # geometry change, comp create/rename/delete, comp active,
        # layer membership, mesh/node/opacity display state. The
        # granular subscribe_typed wiring below fires the specific
        # event kind (GEOMETRY_ACTIVE_CHANGED, GEOMETRY_DEFORM_CHANGED,
        # …) BEFORE this omnibus runs; the dispatcher's same-tick guard
        # then suppresses the redundant GEOMETRIES_CHANGED pump. Display
        # state (show_mesh / show_nodes / display_opacity) doesn't have
        # a typed kind, so it falls through to the omnibus on its own.
        director.geometries.subscribe_typed(
            lambda kind, payload: dispatcher.fire(kind, payload=payload),
        )
        director.geometries.subscribe(
            lambda: dispatcher.fire(GEOMETRIES_CHANGED),
        )
        # Display state (per-geometry show_mesh / show_nodes /
        # display_opacity) is independent of the dispatch matrix —
        # one direct subscription pushes the active geometry's values
        # to the substrate / wireframe / node-cloud actors. The
        # compound GEOMETRIES_CHANGED handles the heavier render
        # plumbing; this just touches actor properties.
        director.geometries.subscribe(_apply_geometry_display)
        # Registry observer covers add/remove/move/visibility when the
        # call site doesn't dispatch granularly. Conservative: just
        # re-run the gate. Granular events from settings tab carry the
        # right scope when they fire first.
        director.registry.subscribe(
            lambda: dispatcher.fire(COMP_ACTIVE_CHANGED),
        )

        # ── Time scrubber row (bottom of grid) ──────────────────────
        from .ui._time_scrubber import TimeScrubberDock
        scrubber = TimeScrubberDock(
            director, on_export=self._export_animation_interactive,
        )
        self._time_scrubber = scrubber
        win.set_bottom_widget(scrubber.widget)

        # ── Per-geometry scenes + concurrent rendering (ADR 0058
        # S2a/S2b) ──
        # Each geometry's scene owns its substrate fill + wireframe
        # actor pair, added once at materialization; "which geometry
        # renders" is actor VISIBILITY, never actor churn. Every
        # geometry with ``visible=True`` renders concurrently; a
        # pair's visibility is ``visible AND show_mesh``.
        boot_geom = director.geometries.active or (
            director.geometries.geometries[0]
            if director.geometries.geometries else None
        )
        # geometry id -> (fill_actor, wireframe_actor).
        self._scene_actors: dict = {}
        # ADR 0058 S2c — pick disambiguation: ``id(substrate actor) ->
        # (geometry_id, scene)``. A pick hit carries ``prop_id ==
        # id(actor)`` (ADR 0047), so this map resolves any geometry's
        # substrate hit against THAT geometry's grid (its deformed
        # points) + index arrays. Maintained in lockstep with
        # ``_scene_actors``: boot pair here, clone pairs in
        # ``_materialize_scene``, dropped in ``_on_geometry_removed``.
        self._actor_scenes: dict = {}
        if boot_geom is not None:
            self._scene_actors[boot_geom.id] = (actor, wireframe_actor)
            for a in (actor, wireframe_actor):
                self._actor_scenes[id(a)] = (boot_geom.id, scene)

        from .core.element_visibility import (
            apply_dim_filter as _apply_dim_f,
        )
        from .data._stage_activation import LAYER_STAGE as _LAYER_STAGE

        def _materialize_scene(geom) -> "FEMSceneData":
            """``scene_factory`` for ``director.bind_plotter``.

            Clones the boot scene (born undeformed, index arrays
            shared — see ``clone_scene``) and wires the render side
            per the S2 plan's disposition table: per-scene
            ElementVisibility (+ dispatcher), shared plotter-scoped
            pick inventory and opacity controller, the CURRENT
            dim-filter and stage-activation state, and a hidden actor
            pair (the RENDER-lane ``_sync_substrate_visibility``
            subscriber — which runs after the pump that materialized
            this scene — turns it on when the geometry is visible).
            """
            from .scene.fem_scene import clone_scene
            new_scene = clone_scene(scene)
            new_scene.pick_engine = scene.pick_engine
            new_scene.opacity_controller = scene.opacity_controller
            ev = _ElementVis(new_scene.grid)
            ev.dispatcher = dispatcher
            new_scene.element_visibility = ev
            # View-global state, applied at materialization (it is
            # re-applied to every materialized scene on change below).
            flt = getattr(self, "_results_filter", None)
            if flt is not None and flt.dims:
                _apply_dim_f(ev, new_scene.cell_dim, flt.active, flt.dims)
            ctrl = getattr(self, "_stage_activation", None)
            if ctrl is not None:
                # ADR 0058 S3b — pinned-or-active: a stage-pinned
                # geometry materializes wearing its PINNED stage's
                # mask, not the active stage's.
                pin = getattr(geom, "stage_id", None)
                mask = (
                    ctrl.mask_for_stage_id(pin) if pin
                    else ctrl.current_mask()
                )
                if mask is not None:
                    ev.set_layer(_LAYER_STAGE, mask)
            # ADR 0084 D1 — same treatment for the scalar threshold: a
            # geometry materialized mid-session must wear its mask from
            # the first frame, not from the next STEP. Pin-aware step
            # resolution lives in the pump.
            pumps.refresh_threshold_for(geom, new_scene)
            # …and for the spatial scope box. This is the ONLY thing
            # that can apply it to a scene that appears mid-session:
            # the scope is off the STEP path by design (it is invariant
            # under the time cursor), so there is no later tick to
            # rescue a missed seed.
            director.scopes.refresh(geom, new_scene)
            fill, wf = _add_substrate_actors(
                new_scene, name_suffix=f"@{geom.id}",
            )
            try:
                fill.SetVisibility(0)
                wf.SetVisibility(0)
            except Exception:
                pass
            self._scene_actors[geom.id] = (fill, wf)
            for a in (fill, wf):
                self._actor_scenes[id(a)] = (geom.id, new_scene)
            return new_scene

        def _sync_substrate_visibility() -> None:
            """Sync every substrate actor pair to its geometry's flags.

            ADR 0058 S2b — concurrent rendering. Idempotent:
            materializes any visible geometry that lacks an actor pair
            (``director.scene_for``), re-points
            ``self._substrate_actor`` / ``self._wireframe_actor`` at
            the ACTIVE geometry's pair (display-level consumers —
            theme, line-width, status — read those), then applies each
            pair's ``visible AND show_mesh`` + per-geometry opacity
            via ``_apply_geometry_display``, re-applies the current
            palette, and rebuilds the label overlays against the
            active scene when they are visible. The node cloud needs
            no work here — the DEFORM pump (which runs before this
            RENDER-lane subscriber in the same fire) already re-synced
            it against the active geometry.
            """
            for geom in self._render_geometries():
                director.scene_for(geom)   # materialize on demand
            active_geom = director.geometries.active
            if active_geom is not None:
                pair = self._scene_actors.get(active_geom.id)
                if pair is not None:
                    self._substrate_actor, self._wireframe_actor = pair
            self._apply_geometry_display()
            _refresh_substrate_colors(THEME.current)
            if self._node_label_actor is not None:
                self._set_node_id_labels(True)
            if self._element_label_actor is not None:
                self._set_element_id_labels(True)

        def _on_geometry_active_changed(_kind, _payload) -> None:
            _sync_substrate_visibility()

        def _on_geometry_removed(_kind, payload) -> None:
            # The director already dropped its cached scene (typed
            # observer, registered first); remove the actors here.
            pair = self._scene_actors.pop(payload, None)
            if pair is not None:
                for a in pair:
                    self._actor_scenes.pop(id(a), None)
                    try:
                        plotter.remove_actor(a)
                    except Exception:
                        pass
            _sync_substrate_visibility()

        dispatcher.subscribe(
            GEOMETRY_ACTIVE_CHANGED, _on_geometry_active_changed,
            lane=Lane.RENDER,
        )
        # ADR 0058 S2b — a visibility flip re-runs the same idempotent
        # sync: it materializes a newly-shown geometry's scene (the
        # DEFORM pump in the same fire already did, but the sync must
        # not depend on row contents) and flips its actor pair on/off.
        dispatcher.subscribe(
            GEOMETRY_VISIBILITY_CHANGED,
            lambda _kind, _payload: _sync_substrate_visibility(),
            lane=Lane.RENDER,
        )
        dispatcher.subscribe(
            GEOMETRY_REMOVED, _on_geometry_removed, lane=Lane.RENDER,
        )
        # ADR 0084 D3 — a geometry that APPEARS needs the same sync:
        # ``_materialize_scene`` hands every new geometry a hidden actor
        # pair and this is the only thing that turns it on. Without the
        # subscription the sync reached ``add``-ed geometries only by
        # accident (a later active/visible flip) — which is why the
        # session-restore path used to re-run it by hand.
        dispatcher.subscribe(
            GEOMETRY_ADDED,
            lambda _kind, _payload: _sync_substrate_visibility(),
            lane=Lane.RENDER,
        )
        # An occluding diagram (contour) attaching / detaching / toggling
        # visibility must re-sync the substrate fill (hide it where the
        # contour covers the substrate, restore it otherwise) so the two
        # coincident opaque surfaces never z-fight. Rides the RENDER lane
        # after the GATE pump in the same fire.
        for _diag_evt in (DIAGRAM_ATTACHED, DIAGRAM_DETACHED,
                          LAYER_VISIBILITY_CHANGED):
            dispatcher.subscribe(
                _diag_evt,
                lambda _kind, _payload: _sync_substrate_visibility(),
                lane=Lane.RENDER,
            )
        # ADR 0058 S3a — an offset change moves the geometry's grid
        # points (DEFORM pump, same fire). The cached pick KD-tree and
        # the label overlays read those points; refresh them after the
        # pump so node snaps / labels never use the stale frame. No
        # visibility change → no _apply_geometry_display needed.
        dispatcher.subscribe(
            GEOMETRY_OFFSET_CHANGED,
            self._on_geometry_offset_changed,
            lane=Lane.RENDER,
        )
        # The scope box is world-frame while the geometry's offset moves
        # the geometry through it, so an offset change genuinely changes
        # WHICH cells are in scope — one of the four inputs that must
        # recompute the mask (the others: the box itself, scene
        # materialization above, and session restore). Scrubbing is
        # pointedly NOT one of them.
        dispatcher.subscribe(
            (GEOMETRY_OFFSET_CHANGED, GEOMETRY_SCOPE_CHANGED),
            self._on_geometry_scope_changed,
            lane=Lane.RENDER,
        )

        # ── Bind director to plotter ────────────────────────────────
        def _bar_prefix_for(diagram) -> "Optional[str]":
            """Scalar-bar title prefix (ADR 0058 S2b ruling): the
            owning geometry's name, but only while MORE than one
            geometry is visible — single-geometry sessions keep their
            unprefixed titles."""
            geoms = director.geometries
            if sum(1 for g in geoms.geometries if g.visible) <= 1:
                return None
            owner = geoms.geometry_for_layer(diagram)
            return owner.name if owner is not None else None

        def _stage_pin_for(diagram) -> "Optional[str]":
            """Owning geometry's stage pin (ADR 0058 S3b). Stamped on
            each diagram by the registry; ``Diagram._scoped_results``
            resolves it at read time — an explicit ``spec.stage_id``
            wins (the two pins compose)."""
            owner = director.geometries.geometry_for_layer(diagram)
            return getattr(owner, "stage_id", None) if owner is not None else None

        director.bind_plotter(
            plotter,
            scene=scene,
            render_callback=lambda: plotter.render() if plotter else None,
            scene_factory=_materialize_scene,
            bar_prefix_resolver=_bar_prefix_for,
            stage_pin_resolver=_stage_pin_for,
        )

        # ── Section planes (ADR 0083 S1) ────────────────────────────
        # The controller is keyed by the backend, which only exists
        # after bind_plotter. Bind the panel to it, hand it the status
        # bar for the seven-planes refusal, and put the off-seam
        # re-cut on the RENDER lane (the backend re-stamps its own
        # layers; substrate / node cloud have nobody else).
        clip_controller = self._clip_planes()
        if clip_controller is not None:
            clip_controller.on_status = win.set_status
            reference = scene.reference_points
            ref_bbox = (
                tuple(reference.min(axis=0))
                + tuple(reference.max(axis=0))
            )
            self._clip_planes_panel.bind(
                clip_controller,
                bbox=ref_bbox,
                view_normal=self._camera_view_normal,
            )
            # ── Gizmos (ADR 0083 S2) — a second projection of the
            # same state: translucent quad + normal arrow per visible
            # plane, rendered as clip-exempt scene-IR layers through
            # the backend. Same reference bbox as the panel's slider,
            # so the quad and the slider talk about one geometry.
            from .core._clip_gizmo import ClipGizmoRenderer
            self._clip_gizmos = ClipGizmoRenderer(
                director.registry.backend, clip_controller, bbox=ref_bbox,
            )
            self._clip_gizmos.refresh()
            # ── Cut faces (ADR 0083 S3) — a third projection: the
            # contoured field painted on each active plane, so a solid
            # opens into its interior instead of an empty box. Driven
            # off the same event plus the ones that move the data under
            # the cut (a step, a warp, a diagram appearing or being
            # re-styled); a mid-drag fire is deferred to the release by
            # the ``settled`` predicate.
            from .core._clip_cut_face import ClipCutFaceRenderer
            self._clip_cut_faces = ClipCutFaceRenderer(
                director.registry.backend, clip_controller,
                sources=self._cut_face_sources,
                settled=self._clip_gesture_settled,
            )
            self._clip_cut_faces.refresh()
            dispatcher.subscribe(
                CLIP_PLANES_CHANGED, self._reapply_clip_planes,
                lane=Lane.RENDER,
            )
            dispatcher.subscribe(
                CLIP_PLANES_CHANGED,
                lambda _kind, _payload: self._clip_gizmos.refresh(),
                lane=Lane.RENDER,
            )
            dispatcher.subscribe(
                (
                    CLIP_PLANES_CHANGED, STEP_CHANGED, DEFORM_CHANGED,
                    DIAGRAM_ATTACHED, DIAGRAM_DETACHED, DIAGRAM_MODIFIED,
                ),
                lambda _kind, _payload: self._clip_cut_faces.refresh(),
                lane=Lane.RENDER,
            )
            dispatcher.subscribe(
                CLIP_PLANES_CHANGED,
                lambda _kind, _payload: self._clip_planes_panel.refresh(),
            )

        # ── Scope gizmo — the box, draggable in the viewport ────────
        # A second projection of the same ScopeController state (the
        # panel's six spin boxes are the first), drawn for the ACTIVE
        # geometry only: ADR 0058 makes "active" the editing target, and
        # a box per scoped geometry would be a pile of overlapping faces
        # with no way to say which one you meant to grab. Clip-exempt
        # scene-IR layers through the backend, like the clip gizmos —
        # and built here, AFTER ``bind_plotter``, because that is when
        # the registry has a live backend to hand it.
        from .core._scope_gizmo import ScopeGizmoRenderer
        scope_reference = scene.reference_points
        self._scope_gizmos = ScopeGizmoRenderer(
            director.registry.backend,
            director.scopes,
            bbox=(
                tuple(scope_reference.min(axis=0))
                + tuple(scope_reference.max(axis=0))
            ),
            active_id=lambda: director.geometries.active_id,
        )
        self._scope_gizmos.refresh()
        dispatcher.subscribe(
            (
                GEOMETRY_SCOPE_CHANGED,
                GEOMETRY_ACTIVE_CHANGED,
                SCOPE_GIZMO_CHANGED,
            ),
            lambda _kind, _payload: self._scope_gizmos.refresh(),
            lane=Lane.RENDER,
        )
        # ``ScopeController`` takes no collaborators, so — like the drag
        # gizmo's ``on_changed`` — the show/hide announcement is wired
        # here. Its own event, deliberately: the gizmo is a view
        # preference, and putting it on GEOMETRY_SCOPE_CHANGED made a
        # checkbox re-run the mask over every geometry (F7).
        director.scopes.on_show_gizmo_changed = (
            self._fire_scope_gizmo_changed
        )
        # The box now has two editors, so the numbers have to follow the
        # mouse. UI lane, so a drag's per-move fire collapses to one
        # repaint per Qt tick; the panel refuses to rewrite a FOCUSED
        # editor, which is what keeps typing from being eaten (the ADR
        # 0083 B3 shape).
        dispatcher.subscribe(
            GEOMETRY_SCOPE_CHANGED,
            lambda _kind, _payload: geometry_panel.refresh_scope_bounds(),
        )

        # ── Wire any pending section cuts (programmatic ingress) ────
        # Done after bind_plotter so each cut Layer's attach() lands
        # against a live plotter + scene; done before session restore
        # so the restore path sees pre-existing layers and can decide
        # whether to replace or augment them.
        self._apply_pending_cuts()

        # ── Subscribe to diagram changes for side-panel docking ─────
        # Side-panel sync stays as its own observer — purely UI;
        # doesn't touch the rendering pipeline.
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
        from .ui._shortcuts_help import add_help_shortcuts_menu
        self._probe_overlay = ProbeOverlay(plotter, scene, director)

        # Local-axes overlay (beam-element geomTransf triads). Built
        # here so it sees the bound plotter + substrate; honours a
        # pre-checked toolbar toggle (e.g. restored session state).
        from .overlays.local_axes_overlay import LocalAxesOverlay
        self._local_axes_overlay = LocalAxesOverlay(plotter, scene, director)
        if getattr(self, "_local_axes_action", None) is not None:
            try:
                if self._local_axes_action.isChecked():
                    self._local_axes_overlay.set_visible(True)
            except Exception:
                pass
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
        # Swap the HUD's step/stage subscriptions onto the dispatcher's
        # UI lane so a rapid scrubber drag collapses to one HDF5
        # re-read per Qt tick instead of one per slider tick.
        self._pick_hud.attach_dispatcher(dispatcher)
        # EmptyStateHUD: bottom-center starter card, hidden until the
        # post-restore emptiness check (R1.4) decides the boot is
        # diagram-less. The first DIAGRAM_ATTACHED from ANY creation
        # path (starter button, settings tab, dialog, session) hides
        # it; UI lane so the hide rides the same Qt tick as the tree
        # rebuild rather than the pump path.
        from .ui._empty_state_hud import EmptyStateHUD
        self._empty_state_hud = EmptyStateHUD(
            plotter.interactor,
            on_primary=self._on_empty_state_add_contour,
            on_secondary=self._on_empty_state_enable_deform,
        )
        dispatcher.subscribe(
            DIAGRAM_ATTACHED,
            lambda _kind, _payload: self._empty_state_hud.hide(),
        )
        add_help_shortcuts_menu(
            win.window,
            entries=[
                ("N / E / G", "Pick mode — node / element / gauss point"),
                ("Shift+LMB drag", "Turntable (yaw-only around up axis)"),
                ("Shift+MMB drag", "Orbit (yaw + pitch, no-roll)"),
                ("MMB / RMB drag", "Pan"),
                ("Scroll", "Zoom (focal point fixed)"),
                ("Shift+click", "Time-history at node"),
                ("Toolbar ⧄", "Section planes — cut the model open"),
                ("F2", "Rename outline item"),
                ("Ctrl+H", "Toggle focus mode"),
                ("Esc", "Deselect"),
                ("Q", "Close window"),
            ],
        )

        # ── Navigation: Shift+LMB drag = orbit, click = time-history.
        # ``install_navigation`` binds the no-roll quaternion orbit
        # (Shift+LMB and Shift+MMB), focal-point-anchored scroll zoom,
        # and the Shift+LMB drag-detect / click split.
        from .core.navigation import install_navigation
        install_navigation(
            plotter,
            on_shift_click=self._on_shift_click_world,
        )

        # ── Colour legends (ADR 0081 L1 + L2) ───────────────────────
        # The LegendController's boxes are derived from pixel font
        # metrics, so its viewer-wide text size is a user preference —
        # bound here and re-applied whenever preferences change, the
        # same shape as the label font sizes above it. The interactor
        # goes on at priority 12, above navigation and picking, and
        # aborts only when the cursor is actually over a legend.
        self._bind_legend_preferences()
        self._install_legend_interactor()
        # The gizmo interactor shares priority 12 but installs AFTER
        # the legend interactor: VTK invokes same-priority observers in
        # insertion order, so a legend overlapping a gizmo wins the
        # press (ADR 0083 Consequences).
        self._install_clip_gizmo_interactor()
        # Third at priority 12, and last on purpose: a legend or a
        # section-plane gizmo drawn over the scope box wins the press.
        self._install_scope_gizmo_interactor()

        # ── Motion LOD ──────────────────────────────────────────────
        # Hide the FE node cloud (one sphere-sprite per node — 600k+ on
        # large results models) while the camera is moving; restore
        # ~120 ms after the gesture settles. Same interactive-LOD model
        # as the pre-solve mesh viewer. The getter reads
        # ``_node_cloud_actor`` fresh so it tracks the point-size
        # rebuild (the actor handle is replaced there).
        from .core.motion_lod import MotionLOD
        self._motion_lod = MotionLOD(
            plotter,
            lambda: (
                [self._node_cloud_actor]
                if self._node_cloud_actor is not None else []
            ),
        )
        self._motion_lod.install()

        # ── Plain LMB pick — node by default, element via E, GP via G.
        # The pick observer absorbs plain LMB so the trackball does
        # not also rotate on drag. Mode is toggled by ``N`` / ``E`` /
        # ``G`` keypresses while the viewport has focus. Drag draws a
        # rubber-band rectangle and on release picks all nodes /
        # elements (per current mode) inside it (GP box-pick is not
        # yet wired — the rectangle still renders but the release is
        # a no-op in GP mode).
        from .core.results_pick import (
            install_results_pick, MODE_NODE, MODE_ELEMENT, MODE_GP,
        )
        self._pick_controller = install_results_pick(
            plotter,
            on_pick=self._on_results_pick,
            on_box_pick=self._on_results_box_pick,
            gp_candidates=self._collect_gp_candidates,
            scene=scene,
            scene_resolver=self._resolve_pick_scene,
            clip_planes=self._clip_planes(),
        )
        plotter.add_key_event(
            "n", lambda: self._set_pick_mode(MODE_NODE),
        )
        plotter.add_key_event(
            "e", lambda: self._set_pick_mode(MODE_ELEMENT),
        )
        plotter.add_key_event(
            "g", lambda: self._set_pick_mode(MODE_GP),
        )

        # ── Dimensional pick filter (ADR 0045 S4b + visual dim-hide) ─
        # 0/1/2/3/4 gate which element dims respond to picks, via the
        # shared FilterController (multi-select toggle, same semantics as
        # the model/mesh viewers). Inactive dims are also ghost-HIDDEN on
        # the single shared substrate actor: the dim filter owns the
        # LAYER_DIM layer of ElementVisibility, which ORs it with the
        # manual hide/isolate layer (LAYER_MANUAL) — so the filter and a
        # user isolate compose instead of clobbering each other. Hidden
        # cells are non-pickable for free (VTK skips HIDDENCELL), so the
        # active_dims gate and the visual hide stay consistent.
        from .core.filter_controller import FilterController
        from .core.element_visibility import apply_dim_filter
        _dim_vals = (
            sorted({int(d) for d in scene.cell_dim.tolist()})
            if scene.cell_dim.size else []
        )

        def _apply_results_filter(active) -> None:
            self._pick_controller.active_dims = frozenset(active)
            # Visual ghost-hide: hide cells whose dim is inactive via the
            # dedicated dim layer (composes with manual/isolate hides).
            # ADR 0058 S2a: the filter is view-global — re-apply to
            # every materialized per-geometry scene (scenes not yet
            # materialized pick it up at materialization).
            for g_scene in self._iter_scenes():
                ev = g_scene.element_visibility
                if ev is not None:
                    apply_dim_filter(ev, g_scene.cell_dim, active, _dim_vals)
            try:
                plotter.render()
            except Exception:
                pass
            # A filter change can leave a stale element highlight on cells
            # the new filter excludes; clear it so the on-screen selection
            # never contradicts the active filter (review nit).
            self._clear_element_highlight()
            # Element-scoped, all-active-aware status (the gate only
            # affects ELEMENT picks; node/gp are ungated by design).
            try:
                if set(active) == set(_dim_vals):
                    win.set_status("Element pick filter: all dims")
                elif not active:
                    win.set_status("Element pick filter: none")
                else:
                    dims_txt = ", ".join(str(d) for d in sorted(active))
                    win.set_status(f"Element pick filter: dims {{{dims_txt}}}")
            except Exception:
                pass

        self._results_filter = FilterController(
            _dim_vals, on_change=_apply_results_filter
        )
        if _dim_vals:
            for _k, _d in [("0", 0), ("1", 1), ("2", 2), ("3", 3)]:
                plotter.add_key_event(
                    _k, lambda dd=_d: self._results_filter.toggle(dd)
                )
            plotter.add_key_event(
                "4", lambda: self._results_filter.select_all()
            )

        # ── Stage activation filter (ADR 0055 viewer-consume V1) ────
        # When the Composed file carries a staged-analysis program
        # (``/opensees/stages`` → ``results.model.stages()``), hide
        # elements owned by not-yet-reached stages — and elements
        # removed by an earlier-or-current stage — while a stage is
        # selected. Owns the LAYER_STAGE layer of ElementVisibility,
        # composing with the dim filter and manual hides. Program
        # stages pair with capture stages BY NAME; unmatched stages
        # render unfiltered (fail-soft — the viewer is a read-only
        # consumer). Vanilla files (no program stages) skip all of
        # this — no controller, no toolbar button.
        from .data._stage_activation import (
            LAYER_STAGE,
            StageActivationController,
            build_from_model,
            pair_capture_to_program,
        )
        from .diagrams._director import COMBINED_STAGE_ID
        self._stage_activation = None
        _act_map = build_from_model(
            self._results.model,
            scene.element_id_to_cell,
            int(scene.grid.n_cells),
        )
        if _act_map is not None:
            # Name pairing with a positional fallback: MPCO/Ladruno
            # capture stages are named ``MODEL_STAGE[<stamp>]`` (never
            # equal to program stage names), so when no name matches
            # and the counts line up, capture stage i pairs with
            # program stage i.
            _stage_names = pair_capture_to_program(
                [
                    (s.id, str(s.name)) for s in director.stages()
                    if s.id != COMBINED_STAGE_ID
                ],
                list(_act_map.hidden_by_name),
            )
            _ctrl = StageActivationController(
                scene.element_visibility,
                _act_map,
                stage_name_for_id=_stage_names.get,
                combined_stage_id=COMBINED_STAGE_ID,
            )
            self._stage_activation = _ctrl

            _hinted_unmatched: set = set()

            def _sync_stage_layers() -> None:
                """Apply per-geometry LAYER_STAGE masks onto every
                materialized scene (ADR 0058 S3b — pinned-or-active: a
                stage-PINNED geometry's scene wears its pinned stage's
                mask while the active stage scrubs; unpinned
                geometries mirror the active stage's mask). Scenes not
                yet materialized pick their mask up at
                materialization. Runs after the controller's own
                ``_apply()`` (which writes the active mask to the boot
                scene), overriding per-pin — for an unpinned boot
                geometry the write is idempotent, so the two never
                fight."""
                for geom in director.geometries.geometries:
                    g_scene = director._scenes.get(geom.id)  # noqa: SLF001 — materialized-only walk, never materializes
                    if g_scene is None:
                        continue
                    ev = g_scene.element_visibility
                    if ev is None:
                        continue
                    pin = getattr(geom, "stage_id", None)
                    mask = (
                        _ctrl.mask_for_stage_id(pin) if pin
                        else _ctrl.current_mask()
                    )
                    if mask is None:
                        ev.clear_layer(LAYER_STAGE)
                    else:
                        ev.set_layer(LAYER_STAGE, mask)

            def _apply_stage_activation(sid) -> None:
                _ctrl.on_stage_changed(sid)
                _sync_stage_layers()
                if (
                    _ctrl.enabled
                    and sid is not None
                    and sid != COMBINED_STAGE_ID
                    and _ctrl.current_mask() is None
                    and sid not in _hinted_unmatched
                ):
                    # Hint once per stage id — combined-mode scrubbing
                    # re-fires real stage ids on every boundary cross.
                    _hinted_unmatched.add(sid)
                    win.set_status(
                        "Stage activation: no program stage named "
                        f"{_stage_names.get(sid, sid)!r} — showing "
                        "all elements",
                        timeout=6000,
                    )
                try:
                    plotter.render()
                except Exception:
                    pass

            director.subscribe_stage(_apply_stage_activation)
            # ADR 0058 S3b — a stage-pin change swaps which stage's
            # mask the geometry's scene wears. The matrix row only
            # pumps STEP + DEFORM, so the mask resync rides the RENDER
            # lane (after the pumps, before the closing render).
            dispatcher.subscribe(
                GEOMETRY_STAGE_PIN_CHANGED,
                lambda _kind, _payload: _sync_stage_layers(),
                lane=Lane.RENDER,
            )
            # Apply once at wiring time. Single-stage files arrive with
            # the director's stage pre-seeded (seed fires no observer);
            # multi-stage files start stage-LESS (``stage_id`` is None
            # until the user picks a stage, or session restore does) —
            # None means no stage context, so the model renders
            # unfiltered until a stage is selected.
            _apply_stage_activation(director.stage_id)

            def _apply_stage_toggle(checked: bool) -> None:
                _ctrl.set_enabled(bool(checked))
                _sync_stage_layers()
                try:
                    plotter.render()
                except Exception:
                    pass

            self._stage_activation_action = win.add_toolbar_action(
                "Stage activation — hide elements not yet "
                "activated in the selected stage",
                "⧉",
                _apply_stage_toggle,
                checkable=True,
            )
            try:
                # Visual state only — QAction.setChecked does not emit
                # ``triggered``; the controller starts enabled.
                self._stage_activation_action.setChecked(True)
            except Exception:
                pass

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

        # Composition viewport gate is now ``PumpSet.pump_gate`` and
        # fires via the dispatcher (no embedded render). Subscriptions
        # also wired earlier — see ``_on_geometries_changed``.

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

        # ── First-run empty state (R1.4) ────────────────────────────
        # Zero diagrams AFTER the restore ran = a fresh boot with no
        # saved layers; float the starter card so the grey mesh isn't
        # a dead end.
        self._maybe_show_empty_state()

        # ── Headless export path — build only, no event loop ────────
        # Realize the render surface so screenshots capture real pixels
        # (the GL context isn't created until the widget is shown), but
        # do NOT enter the blocking loop. The caller drives the viewer
        # and tears it down via :meth:`close`. The slot-failure handler
        # stays registered until close (unregistered in _on_close).
        if not run_loop:
            self._realize_headless(win, window_size=window_size)
            return self

        # ── Run ─────────────────────────────────────────────────────
        # enter_loop=False: a Qt loop is already running (we were opened
        # from another viewer's File → Open Results…). Present the window
        # so it joins the existing loop and return without blocking; the
        # slot-failure handler stays registered for the window's life.
        if not enter_loop:
            win.present()
            return self
        try:
            win.exec()
        finally:
            unregister_error_handler(_slot_failure_to_status)
        return self

    def _realize_headless(
        self, win: Any, *, window_size: "Optional[tuple[int, int]]" = None,
    ) -> None:
        """Realize the window's render surface without an event loop.

        Off-screen-style export reuses the full Qt viewer (reuse, don't
        reimplement the deform pump / contour pipeline). The
        QtInteractor's OpenGL context is only created once the widget is
        shown, so we ``show()`` it (un-maximized, not raised — it must
        not steal foreground), hide the docks + toolbar so captured
        frames are a clean viewport, size the window, and pump events
        until the first render lands. The window is closed by
        :meth:`close`.
        """
        from qtpy import QtWidgets
        try:
            # Clean viewport: hide every dock + the toolbar so the
            # screenshot is just the 3D scene (focus mode is the
            # existing all-docks-hidden snapshot).
            try:
                win.toggle_focus_mode()
            except Exception:
                pass
            if window_size is not None:
                try:
                    win.window.resize(int(window_size[0]), int(window_size[1]))
                except Exception:
                    pass
            win.window.show()
            # Pump the queued show/expose/resize events so the GL
            # context realizes (and the resize lands) before the first
            # screenshot.
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()
            win.plotter.render()
            if app is not None:
                app.processEvents()
        except Exception as exc:
            from ._log import log_error
            log_error("export", "realize_headless", exc)
            raise

    def close(self) -> None:
        """Tear down a headless viewer (built via ``show(run_loop=False)``).

        Closes the window — which fires :meth:`_on_close` for the usual
        diagram / plotter / log-router teardown. Safe to call once.
        """
        win = self._win
        if win is None:
            return
        try:
            win.window.close()
        except Exception:
            pass
        self._win = None

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

    def _apply_pending_cuts(self) -> None:
        """Wire ``cuts=`` constructor kwarg and the symmetric
        cuts/orientation auto-load into the director immediately after
        the registry binds to a live plotter.

        Phase 8 (ADR 0020 INV-1 / INV-5) — cuts auto-load is gated on
        ``results.model`` (always populated post-prune). When the
        results file carries ``/opensees/cuts/`` and
        ``/opensees/sweeps/``, they are read and attached
        automatically. Explicit ``cuts=`` wins over h5 persistence —
        kwarg-precedence contract from ARCHITECTURE.md H14.

        Failures here are logged but non-fatal — a malformed cut
        shouldn't prevent the rest of the viewer from opening. The user
        sees the error in the session log and can construct the cut by
        hand afterwards.
        """
        if self._director is None:
            return
        from .data._h5_probe import resolve_orientation_source
        from ._log import log_action, log_error
        # Resolve the file path the director should bind for FemToOps
        # tag mapping.  Phase 8: ``results._path`` is the canonical
        # source (Composed-file pattern carries ``/opensees/``).
        # Centralised in :func:`resolve_orientation_source` — see
        # ADR 0026 for the H5ModelReader Protocol that supersedes the
        # path-return contract.
        bind_path: Optional[Path] = resolve_orientation_source(self._results)
        # Whether auto-load should fire (no explicit cuts; cuts
        # source present).
        model = self._results.model
        has_persisted_cuts = (
            len(model.cuts()) > 0 or len(model.sweeps()) > 0
        )
        # Early-out: nothing to do.
        if (
            not self._pending_cuts
            and bind_path is None
            and not has_persisted_cuts
        ):
            return
        # Bind the director — ADR 0026 PR7-d collapsed the dual
        # ``set_model`` + ``_bind_model_h5`` ceremony into one call.
        # ``bind_results(results)`` derives both binding sources off
        # the Results: ``results.model`` for cuts iteration and
        # ``resolve_orientation_source(results)`` for the tag-map path.
        try:
            self._director.bind_results(self._results)
        except Exception as exc:
            log_error("init", "ResultsViewer.bind_results", exc)
        if self._pending_cuts:
            # Explicit cuts kwarg — wins over auto-load (H14).
            for i, cut in enumerate(self._pending_cuts):
                try:
                    self._director.add_section_cut(cut)
                except Exception as exc:
                    log_action(
                        "session", "section_cut_add_failed",
                        index=i, label=str(getattr(cut, "label", "")),
                        error=type(exc).__name__,
                    )
                    log_error("init", f"ResultsViewer.cut[{i}]", exc)
        elif has_persisted_cuts and bind_path is not None:
            # Auto-load — gated on ``results.model`` carrying cuts
            # (INV-5).  Requires a bound file path for the tag_map.
            try:
                self._director.load_cuts_from_h5()
            except Exception as exc:
                log_action(
                    "session", "section_cut_autoload_failed",
                    model_h5=str(bind_path),
                    error=type(exc).__name__,
                )
                log_error("init", "ResultsViewer.load_cuts_from_h5", exc)
        # Clear the queue so a future re-show (test contexts mainly)
        # doesn't double-add.
        self._pending_cuts = ()

    def _render_geometries(self) -> list:
        """Geometries whose scene participates in rendering this frame.

        ADR 0058 S2b: every geometry whose ``visible`` flag is on —
        all of them render concurrently, each at its own deform state.
        The DEFORM pump loops this; "active" is only the editing
        target.
        """
        if self._director is None:
            return []
        return [
            g for g in self._director.geometries.geometries if g.visible
        ]

    def _on_geometry_offset_changed(self, _kind, payload) -> None:
        """RENDER-lane subscriber for ``GEOMETRY_OFFSET_CHANGED``
        (ADR 0058 S3a).

        Runs after the DEFORM pump moved the geometry's grid points to
        their offset positions:

        * drops that scene's cached pick KD-tree (``node_tree`` —
          rebuilt lazily over the NEW points on the next node snap;
          per-step deform staleness is pre-existing and out of scope);
        * rebuilds the node / element label overlays when visible
          (they bake the active scene's ``grid.points`` at build —
          mirrors ``_sync_substrate_visibility``).
        """
        director = self._director
        if director is None:
            return
        geom = director.geometries.find(payload)
        g_scene = director.scene_for(geom) if geom is not None else None
        if g_scene is not None:
            g_scene.node_tree = None
        if getattr(self, "_node_label_actor", None) is not None:
            self._set_node_id_labels(True)
        if getattr(self, "_element_label_actor", None) is not None:
            self._set_element_id_labels(True)

    def _on_geometry_scope_changed(self, _kind, _payload) -> None:
        """RENDER-lane subscriber that recomputes the SCOPE BOX masks.

        Wired to ``GEOMETRY_SCOPE_CHANGED`` (the box was set / moved /
        cleared, or a session restored one) and to
        ``GEOMETRY_OFFSET_CHANGED`` (the box is world-frame, so moving
        the geometry moves it through the box).

        Payload-INDEPENDENT on purpose: inside a gesture batch the
        RENDER lane replays one call per handler with the LAST payload
        (ADR 0084 D3), so a per-geometry handler would silently skip
        every other geometry a session restore touched. Recomputing
        every already-materialized scene is idempotent, and outside a
        drag it is the honest answer — the mask is pure geometry, no
        read and no step.

        **Inside a drag it is not cheap**, which is what
        :meth:`_scope_gesture_settled` is for: this pass is O(cells)
        per materialized geometry plus a ghost recompose (measured at
        28 ms on a 500k-cell grid), and the gizmo delivers it on every
        mouse-move. The box, the gizmo and the panel all still follow
        the cursor live; only the mask waits for the release, when
        :meth:`_on_scope_drag_end` fires the event once more with the
        gesture settled. The same push/pull pair the clip gizmo uses
        for its cut faces (ADR 0083 S3).
        """
        director = self._director
        if director is None or not director.scopes.needs_refresh():
            return
        if not self._scope_gesture_settled():
            return
        for geom in director.geometries.geometries:
            g_scene = director.materialized_scene_for(geom)
            if g_scene is not None:
                director.scopes.refresh(geom, g_scene)

    def _iter_scenes(self) -> list:
        """Every materialized per-geometry scene (ADR 0058 S2a).

        View-global state (dim filter, stage activation) loops this
        to re-apply on change. Includes the boot scene; never
        materializes anything.
        """
        director = self._director
        scenes: list = (
            list(director.materialized_scenes())
            if director is not None else []
        )
        boot = self._boot_scene
        if boot is not None and all(s is not boot for s in scenes):
            scenes.append(boot)
        return scenes

    def _resolve_pick_scene(self, prop_id) -> tuple:
        """``(geometry_id, scene)`` a pick hit resolves against (ADR
        0058 S2c).

        ``prop_id`` is the hit actor's ``id()`` (``PickHit.prop_id``):

        * a registered substrate actor → its owning geometry + THAT
          geometry's scene (the hit grid, with its own deformed
          points);
        * ``None`` (box gesture — no single hit actor) → the ACTIVE
          geometry + its scene;
        * an unregistered actor (GP glyphs, overlay props) → the
          active scene for coordinate reads, but ``geometry_id=None``
          (the geometry is not actually known).
        """
        if prop_id is not None:
            entry = (
                getattr(self, "_actor_scenes", None) or {}
            ).get(int(prop_id))
            if entry is not None:
                return entry
        director = self._director
        active = (
            director.geometries.active if director is not None else None
        )
        active_scene = self._scene
        if prop_id is None and active is not None:
            return (active.id, active_scene)
        return (None, active_scene)

    def _scene_for_geometry_id(self, geometry_id):
        """Scene carried by a pick's ``geometry_id``; falls back to the
        active scene (ADR 0058 S2c — coordinate reads must follow the
        hit geometry's grid, not the boot scene)."""
        director = self._director
        if geometry_id is not None and director is not None:
            geom = director.geometries.find(geometry_id)
            if geom is not None:
                g_scene = director.scene_for(geom)
                if g_scene is not None:
                    return g_scene
        return self._scene

    def _pick_geometry_label(self, geometry_id) -> "Optional[str]":
        """Geometry name for pick reporting — only while MORE than one
        geometry is visible (mirrors the S2b scalar-bar prefix rule);
        ``None`` otherwise."""
        director = self._director
        if geometry_id is None or director is None:
            return None
        geoms = director.geometries
        if sum(1 for g in geoms.geometries if g.visible) <= 1:
            return None
        geom = geoms.find(geometry_id)
        return geom.name if geom is not None else None

    def _sync_diagram_substrate_points(
        self, deformed_pts, *, geometry=None, scene=None,
    ) -> None:
        """Forward the deformation to each layer's
        :meth:`Diagram.sync_substrate_points` hook.

        This is the ONLY deformation fan-out: post-ADR-0042 every
        diagram emits backend-owned dataset COPIES (the old
        ``_sync_layer_grids`` walk over ``d._actors`` was dead code —
        migrated diagrams never populate ``_actors``). Substrate-
        extracted layers re-sample via their cached
        ``vtkOriginalPointIds`` rows; owned-geometry layers recompute
        their points.

        ADR 0058 S1: when ``geometry`` is given, the fan-out is scoped
        to that geometry's layers (a layer with no recorded membership
        counts as the active geometry's — freshly attached diagrams
        land there). ``geometry=None`` keeps the legacy all-layers
        fan-out against the bound scene.
        """
        if self._director is None or self._scene is None:
            return
        g_scene = scene if scene is not None else self._scene
        geoms = self._director.geometries
        for d in self._director.registry.diagrams():
            if geometry is not None:
                owner = geoms.geometry_for_layer(d) or geoms.active
                if owner is not geometry:
                    continue
            try:
                d.sync_substrate_points(deformed_pts, g_scene)
            except Exception as exc:
                # Loud, but keep fanning out — one bad layer must not
                # leave the others at reference positions.
                _pump_failed("deform_fanout", exc, diagram=id(d))
                continue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Detach diagrams and release plotter binding before window dies."""
        from ._log import log_action, shutdown as _log_shutdown
        log_action("session", "close")
        # Unregister the slot-failure handler here so it is released on
        # BOTH paths: the interactive path also drops it in the
        # win.exec() finally (a double-unregister is a no-op), but the
        # headless run_loop=False path never reaches that finally — so
        # without this the closure (over the whole window graph) would
        # leak into the process-global handler list, once per export.
        handler = getattr(self, "_slot_failure_handler", None)
        if handler is not None:
            try:
                from ._failures import unregister_error_handler
                unregister_error_handler(handler)
            except Exception:
                pass
            self._slot_failure_handler = None
        # Auto-save the session before tearing down — the diagrams
        # still hold their specs at this point.
        if self._save_session:
            self._save_session_to_disk()
        # Tear down the log router BEFORE the director / plotter shut
        # down, so any teardown-time exceptions get captured (or at
        # least don't fire into a half-disconnected signal). Then sever
        # the output dock's connection to the router.
        if self._output_dock is not None:
            try:
                self._output_dock.close()
            except Exception:
                pass
        if self._log_router is not None:
            try:
                self._log_router.uninstall()
            except Exception:
                pass
        # Tear down the gizmo interactors while the VTK interactor and
        # render window are still alive: RemoveObserver needs the iren,
        # and the cursor withdrawal needs the arbiter's window (F10 —
        # a requester that dies mid-hover leaves a SIZEALL memo
        # standing that nothing can ever retract). Then drop the gizmo
        # overlay actors; their refresh() subscriptions die with the
        # dispatcher, but the layers sit in the plotter.
        for attr in ("_clip_gizmo_interactor", "_scope_gizmo_interactor"):
            interactor = getattr(self, attr, None)
            if interactor is not None:
                try:
                    interactor.uninstall()
                except Exception:
                    pass
                setattr(self, attr, None)
        # The refs stay standing (cleared, not nulled): the RENDER-lane
        # ``self._clip_gizmos.refresh()`` lambdas dereference them, and
        # ``clear()`` is idempotent while ``None`` would raise.
        for gizmos in (self._clip_gizmos, self._scope_gizmos):
            if gizmos is not None:
                try:
                    gizmos.clear()
                except Exception:
                    pass
        # Drop the registry observer (plan 06 step 4) so subsequent
        # registry mutations during director shutdown don't poke a
        # half-dismantled editor.
        unsub = getattr(self, "_registry_unsub", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
            self._registry_unsub = None
        # Drop director step / stage bridges (plan 04 step 2 cont.).
        # Without this, a final step/stage fire during director
        # teardown would emit an active*Changed signal at a moment
        # when subscribers may already be partially destructed.
        # Same for the legend text-size preference (ADR 0081 L1).
        # PREFERENCES is a module-level singleton, so a subscription
        # whose closure captures ``self`` pins the whole viewer for the
        # life of the process — and stacks one per viewer opened.
        for attr in ("_legend_pref_unsub", "_step_unsub", "_stage_unsub"):
            u = getattr(self, attr, None)
            if u is not None:
                try:
                    u()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._director is not None:
            try:
                self._director.unbind_plotter()
            except Exception:
                pass
        # Release the HDF5 file handle so the user can re-run the
        # capture (overwrite the same path) without a PermissionError
        # from the still-open reader. Skipped on the headless export
        # path, which borrows the caller's live Results.
        if self._own_results_close:
            try:
                self._results.close()
            except Exception as exc:
                from ._log import log_error
                log_error("session", "results.close", exc)
        # Drop the strong ref pinned in show() so the viewer can be
        # garbage-collected after the window closes.
        _LIVE_VIEWERS.discard(self)
        # Flush + close log handlers so the session file is complete
        # on disk by the time the user looks for it.
        _log_shutdown()

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
            from .diagrams._session import (
                save_session, GeometrySnapshot, CompositionSnapshot,
                ScopeSnapshot, ThresholdSnapshot,
            )
            # Flat list of every Diagram instance in the registry, in
            # registry order. Compositions reference these by index.
            registry_diagrams = list(self._director.registry.diagrams())
            id_to_index: dict[int, int] = {
                id(d): i for i, d in enumerate(registry_diagrams)
            }
            specs = [d.spec for d in registry_diagrams]

            # Geometry → Composition tree mirroring the live manager.
            geom_mgr = self._director.geometries
            geom_snapshots: list[GeometrySnapshot] = []
            for geom in geom_mgr.geometries:
                comp_snapshots: list[CompositionSnapshot] = []
                for comp in geom.compositions.compositions:
                    comp_snapshots.append(CompositionSnapshot(
                        id=comp.id,
                        name=comp.name,
                        layer_indices=tuple(
                            id_to_index[id(d)]
                            for d in comp.layers
                            if id(d) in id_to_index
                        ),
                    ))
                # ADR 0084 D1 — the threshold rides INSIDE the geometry
                # snapshot (its id is stale by the time the session is
                # restored; the geometry is re-matched by name).
                thr = self._director.thresholds.settings_for(geom.id)
                # …and so does the scope box, for the same reason.
                box = self._director.scopes.box_for(geom.id)
                geom_snapshots.append(GeometrySnapshot(
                    id=geom.id,
                    name=geom.name,
                    deform_enabled=bool(geom.deform_enabled),
                    deform_field=geom.deform_field,
                    deform_scale=float(geom.deform_scale),
                    offset=tuple(float(c) for c in geom.offset),
                    stage_id=geom.stage_id,
                    threshold=None if thr is None else ThresholdSnapshot(
                        component=thr.component, lo=thr.lo, hi=thr.hi,
                        topology=thr.topology,
                    ),
                    scope=None if box is None else ScopeSnapshot(
                        min=tuple(float(c) for c in box.min),
                        max=tuple(float(c) for c in box.max),
                    ),
                    visible=bool(geom.visible),
                    show_mesh=bool(geom.show_mesh),
                    show_nodes=bool(geom.show_nodes),
                    display_opacity=float(geom.display_opacity),
                    active_composition_id=geom.compositions.active_id,
                    compositions=tuple(comp_snapshots),
                ))

            fem = self._results.fem
            # ADR 0026 PR-stretch — the director no longer stores a
            # ``_model_h5`` field; the tag-map path is derived on
            # demand from the bound :class:`Results`.  We still emit
            # ``model_h5`` in the session payload for back-compat
            # (older save_session signatures, older readers) — but
            # source it from the same probe the director uses.
            from .data._h5_probe import resolve_orientation_source
            model_h5_path = resolve_orientation_source(self._results)
            save_session(
                specs=specs,
                results_path=path,
                fem_snapshot_id=getattr(fem, "snapshot_id", None),
                geometries=geom_snapshots,
                active_geometry_id=geom_mgr.active_id,
                active_stage_id=self._director.stage_id,
                active_step=int(self._director.step_index),
                model_h5=model_h5_path,
                legends=self._legend_snapshots(),
                clip_planes=self._clip_plane_snapshots(),
                clip_apply_cuts=bool(
                    getattr(self._clip_planes(), "apply_cuts", True),
                ),
                clip_show_gizmos=bool(
                    getattr(self._clip_planes(), "show_gizmos", True),
                ),
                clip_cut_face=bool(
                    getattr(self._clip_planes(), "cut_face", False),
                ),
                scope_show_gizmo=bool(
                    getattr(self._director.scopes, "show_gizmo", True),
                ),
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
            from .diagrams._session import (
                default_session_path,
                load_session,
                session_restorable,
            )
            session_path = default_session_path(path)
            if not session_path.exists():
                return
            session = load_session(session_path)
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._maybe_restore_session", exc)
            return

        if not session_restorable(session):
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
        # Holes (ADR 0084 D6) are positional padding, not restorable
        # layers — don't count or list them.
        real = [s for s in session.diagrams if s is not None]
        n = len(real)
        labels = ", ".join(
            f"{s.kind}:{s.selector.component}" for s in real[:5]
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
        """Apply a saved session (ADR 0084 D7 — delegates to
        :class:`._session_apply.SessionController`).

        The restore body moved out of the shell verbatim; this wrapper
        only names the collaborators it used to read off ``self``.
        ``_prompt_restore`` stays here — it is Qt dialog code.
        """
        from ._session_apply import SessionController

        SessionController(
            director=self._director,
            results=self._results,
            legends=self._legends,
            clip_planes=self._clip_planes,
        ).apply(session, win)

    # ------------------------------------------------------------------
    # First-run empty state (R1.4)
    # ------------------------------------------------------------------

    def _maybe_show_empty_state(self) -> None:
        """Show the starter card iff the boot ended with zero diagrams.

        Runs AFTER ``_maybe_restore_session`` so a restored session's
        layers count; polling the registry length is the robust
        emptiness test (the restore path reports nothing).
        """
        hud = getattr(self, "_empty_state_hud", None)
        director = self._director
        if hud is None or director is None:
            return
        if len(director.registry) != 0:
            return
        from .diagrams._kind_catalog import (
            _union_across_stages,
            _vector_prefixes,
        )
        from .diagrams._starter import default_contour_component
        comp = default_contour_component(director)
        if comp is None:
            # Nothing contourable — the card still shows title +
            # dismiss so the user knows why the viewport is bare.
            hud.set_primary("")
        elif comp.startswith("displacement_"):
            hud.set_primary("Add displacement contour")
        else:
            hud.set_primary(f"Add {comp} contour")
        # Deform starter needs a displacement vector field (same
        # availability recipe as the geometry panel's Deformation
        # section above).
        try:
            has_disp = "displacement" in _vector_prefixes(
                _union_across_stages(director, "nodes"),
            )
        except Exception:
            has_disp = False
        hud.set_secondary("Enable deform ×1" if has_disp else "")
        hud.show()

    def _on_empty_state_add_contour(self) -> None:
        """Starter-card primary action — add the default contour.

        DIAGRAM_ATTACHED (fired inside the starter) hides the HUD; on
        failure (e.g. NoDataError) the HUD stays up and the error goes
        to the status bar, loud on purpose.
        """
        director = self._director
        if director is None:
            return
        from .diagrams._starter import add_default_contour
        try:
            add_default_contour(director)
        except Exception as exc:
            if self._win is not None:
                self._win.set_status(
                    f"Could not add contour: {exc}", timeout=6000,
                )

    def _on_empty_state_enable_deform(self) -> None:
        """Starter-card secondary action — deform ×1 on displacement.

        Deform alone still leaves zero diagrams, so the HUD stays up
        until a diagram attaches or the user dismisses it.
        """
        director = self._director
        if director is None:
            return
        geom = director.geometries.active
        if geom is None:
            return
        director.geometries.set_deformation(
            geom.id, enabled=True, field="displacement", scale=1.0,
        )

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
    # Plain-LMB pick — dispatch on mode (node / element)
    # ------------------------------------------------------------------

    def _on_results_pick(self, result) -> None:
        """Dispatch a :class:`PickResult` based on its ``kind``."""
        # Pause animation so the user can inspect the picked position.
        if self._time_scrubber is not None:
            self._time_scrubber.stop_animation()
        from .core.results_pick import MODE_NODE, MODE_ELEMENT, MODE_GP
        from ._log import log_action
        if result.kind == MODE_NODE:
            log_action(
                "pick", "node",
                world=tuple(round(c, 3) for c in result.world),
                geometry_id=result.geometry_id,
            )
            self._on_node_pick(
                result.world, geometry_id=result.geometry_id,
            )
        elif result.kind == MODE_ELEMENT:
            log_action(
                "pick", "element",
                element_id=result.element_id, cell_id=result.cell_id,
                geometry_id=result.geometry_id,
            )
            self._on_element_pick(
                result.element_id, result.cell_id,
                geometry_id=result.geometry_id,
            )
        elif result.kind == MODE_GP:
            log_action(
                "pick", "gp",
                element_id=result.element_id, gp_index=result.gp_index,
            )
            self._on_gp_pick(
                result.element_id, result.gp_index, result.world,
            )

    def _on_node_pick(self, world_pos, *, geometry_id=None) -> None:
        """Drop a probe marker at the nearest node + refresh the HUD.

        ADR 0058 S2c: the snap reads coordinates off the HIT
        geometry's scene (its deformed grid), not the boot/active
        scene — ``geometry_id`` arrives from the pick result.
        """
        if self._probe_overlay is None:
            return
        try:
            import numpy as np
            point = self._probe_overlay.probe_at_point(
                np.asarray(world_pos),
                scene=self._scene_for_geometry_id(geometry_id),
                geometry_id=geometry_id,
            )
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._on_node_pick", exc)
            return
        cb = self._probe_overlay.on_point_result
        if cb is not None:
            try:
                cb(point)
            except Exception as exc:
                from ._failures import report
                report("ResultsViewer._on_node_pick.on_point_result", exc)

    def _on_element_pick(self, element_id, cell_id, *, geometry_id=None) -> None:
        """Highlight the picked element + post a status message.

        ADR 0058 S2c: the highlight extracts the cell from the HIT
        geometry's grid (its deformed points); the status names the
        geometry while more than one is visible.
        """
        if element_id is None or cell_id is None:
            return
        self._highlight_element_cells(
            [int(cell_id)],
            scene=self._scene_for_geometry_id(geometry_id),
        )
        if self._win is not None:
            label = self._pick_geometry_label(geometry_id)
            suffix = f" on {label}" if label else ""
            self._win.set_status(
                f"Picked element {int(element_id)} "
                f"(cell {int(cell_id)}){suffix}.",
                timeout=4000,
            )


    def _collect_gp_candidates(self):
        """Aggregate GP centers across active GaussPointDiagrams.

        Returns ``(centers, element_ids, gp_indices)`` arrays usable
        directly by the box-pick path in ``results_pick.py``. Empty
        arrays mean "no GP markers on screen". ``None`` on failure.
        """
        if self._director is None:
            return None
        import numpy as np
        from .diagrams._gauss_marker import GaussPointDiagram
        centers_list: list = []
        eids_list: list = []
        idxs_list: list = []
        try:
            diagrams = list(self._director.registry.diagrams())
        except Exception:
            return None
        for d in diagrams:
            if not isinstance(d, GaussPointDiagram):
                continue
            cloud = getattr(d, "_cloud", None)
            eidx = getattr(d, "_gp_element_index", None)
            if cloud is None or eidx is None:
                continue
            try:
                pts = np.asarray(cloud.points, dtype=np.float64)
                eidx_arr = np.asarray(eidx, dtype=np.int64)
            except Exception:
                continue
            if pts.shape[0] != eidx_arr.size or pts.shape[0] == 0:
                continue
            centers_list.append(pts)
            eids_list.append(eidx_arr)
            idxs_list.append(np.arange(pts.shape[0], dtype=np.int64))
        if not centers_list:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
        return (
            np.concatenate(centers_list, axis=0),
            np.concatenate(eids_list, axis=0),
            np.concatenate(idxs_list, axis=0),
        )

    def _on_gp_pick(self, element_id, gp_index, world_pos) -> None:
        """Highlight a halo at the picked GP + post a status message."""
        if element_id is None or gp_index is None:
            return
        self._highlight_gp_world(world_pos)
        if self._win is not None:
            self._win.set_status(
                f"Picked GP {int(gp_index)} on element {int(element_id)}.",
                timeout=4000,
            )

    def _highlight_gp_world(self, world_pos) -> None:
        """Drop a single GP highlight at ``world_pos``."""
        import numpy as np
        try:
            pos = np.asarray(world_pos, dtype=np.float64).reshape(1, 3)
        except Exception:
            return
        self._highlight_gps(pos)

    def _highlight_gps(self, world_positions) -> None:
        """Render bright spheres at one or more GP world coords.

        ``world_positions`` is ``(N, 3)`` (or empty). Replaces any
        prior GP highlight; empty input clears the overlay. Cleared
        on next GP pick / Esc / mode change.
        """
        if self._scene is None or self._plotter is None:
            return
        try:
            import numpy as np
            import pyvista as pv
        except Exception:
            return
        try:
            pts = np.asarray(world_positions, dtype=np.float64)
        except Exception:
            return
        self._clear_gp_highlight()
        if pts.size == 0:
            return
        if pts.ndim != 2 or pts.shape[1] != 3:
            return
        diag = float(getattr(self._scene, "model_diagonal", 0.0)) or 1.0
        radius = 0.008 * diag
        try:
            cloud = pv.PolyData(pts)
            sphere = pv.Sphere(
                radius=radius, theta_resolution=14, phi_resolution=14,
            )
            glyphs = cloud.glyph(geom=sphere, scale=False, orient=False)
            actor = self._plotter.add_mesh(
                glyphs,
                color="#ffd400",
                name="_results_pick_gp_highlight",
                pickable=False,
                reset_camera=False,
                lighting=True,
                smooth_shading=True,
            )
            self._gp_pick_highlight_actor = actor
        except Exception:
            self._gp_pick_highlight_actor = None

    def _clear_gp_highlight(self) -> None:
        if (
            self._plotter is None
            or getattr(self, "_gp_pick_highlight_actor", None) is None
        ):
            return
        try:
            self._plotter.remove_actor("_results_pick_gp_highlight")
        except Exception:
            pass
        self._gp_pick_highlight_actor = None

    def _on_results_box_pick(self, box_result) -> None:
        """Dispatch a :class:`BoxPickResult` based on its ``kind``."""
        from .core.results_pick import MODE_NODE, MODE_ELEMENT, MODE_GP
        if box_result.kind == MODE_ELEMENT:
            self._highlight_element_cells(
                box_result.cell_ids,
                scene=self._scene_for_geometry_id(box_result.geometry_id),
            )
            count = int(box_result.ids.size)
            if self._win is not None:
                self._win.set_status(
                    f"Box-picked {count} element"
                    f"{'' if count == 1 else 's'}.",
                    timeout=4000,
                )
        elif box_result.kind == MODE_NODE:
            count = int(box_result.ids.size)
            if self._win is not None:
                self._win.set_status(
                    f"Box-picked {count} node{'' if count == 1 else 's'}.",
                    timeout=4000,
                )
        elif box_result.kind == MODE_GP:
            count = int(box_result.ids.size)
            # Build the world-position array by mapping the picked
            # (element_id, gp_index) pairs back through the candidate
            # closure — much cheaper to re-query than to thread world
            # coords through the BoxPickResult.
            self._highlight_gps_for_box(box_result)
            if self._win is not None:
                self._win.set_status(
                    f"Box-picked {count} GP{'' if count == 1 else 's'}.",
                    timeout=4000,
                )

    def _highlight_gps_for_box(self, box_result) -> None:
        """Look up world coords for a GP BoxPickResult and highlight them."""
        if box_result.ids.size == 0:
            self._clear_gp_highlight()
            return
        cand = self._collect_gp_candidates()
        if cand is None:
            return
        import numpy as np
        centers, eids, gp_idxs = cand
        # Match (element_id, gp_index) pairs against the candidate
        # arrays. Both sides are small enough for a python loop.
        wanted = set(
            (int(e), int(g))
            for e, g in zip(box_result.ids, box_result.gp_indices)
        )
        pos = np.array(
            [
                centers[i]
                for i in range(centers.shape[0])
                if (int(eids[i]), int(gp_idxs[i])) in wanted
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        self._highlight_gps(pos)

    def _highlight_element_cell(self, cell_id: int) -> None:
        """Render a wireframe overlay around the picked substrate cell.

        Replaces any prior highlight so only the latest pick is
        visible. ``Esc`` clears via :meth:`_on_escape` (which already
        also clears outline / probe state).
        """
        self._highlight_element_cells([int(cell_id)])

    def _highlight_element_cells(self, cell_ids, *, scene=None) -> None:
        """Render a wireframe overlay around one or more picked cells.

        ``cell_ids`` may be a sequence or numpy array; empty ⇒ clear
        the current highlight. Each call replaces the prior highlight
        — picks don't accumulate. ``Esc`` also clears. ``scene``
        selects whose grid the cells extract from (ADR 0058 S2c —
        the hit geometry's); the active scene by default.
        """
        scene = scene if scene is not None else self._scene
        if scene is None or self._plotter is None:
            return
        try:
            import numpy as np
            ids = np.asarray(cell_ids, dtype=np.int64).ravel()
        except Exception:
            return
        self._clear_element_highlight()
        if ids.size == 0:
            return
        try:
            sub = scene.grid.extract_cells(ids)
        except Exception:
            return
        try:
            actor = self._plotter.add_mesh(
                sub,
                color="#ffd400",
                style="wireframe",
                line_width=3,
                name="_results_pick_element_highlight",
                pickable=False,
                reset_camera=False,
            )
            self._element_pick_highlight_actor = actor
        except Exception:
            self._element_pick_highlight_actor = None

    def _clear_element_highlight(self) -> None:
        if (
            self._plotter is None
            or getattr(self, "_element_pick_highlight_actor", None) is None
        ):
            return
        try:
            self._plotter.remove_actor("_results_pick_element_highlight")
        except Exception:
            pass
        self._element_pick_highlight_actor = None

    def _set_pick_mode(self, mode: str) -> None:
        """Toggle pick mode and post a status hint."""
        if getattr(self, "_pick_controller", None) is None:
            return
        try:
            self._pick_controller.set_mode(mode)
        except ValueError:
            return
        if mode != "element":
            self._clear_element_highlight()
        if mode != "gp":
            self._clear_gp_highlight()
        if self._win is not None:
            self._win.set_status(
                f"Pick mode: {mode}.", timeout=2000,
            )

    # ------------------------------------------------------------------
    # Shift-click → time-history series
    # ------------------------------------------------------------------

    def _bind_legend_preferences(self) -> None:
        """Drive the legend controller's default text size from prefs.

        ADR 0081 open question 3: a viewer-wide default with a
        per-legend override. A legend the user has resized by hand keeps
        its own ``font_scale`` and is untouched by this.
        """
        from .ui.preferences_manager import PREFERENCES as _PREF

        def _apply(prefs) -> None:
            director = getattr(self, "_director", None)
            registry = getattr(director, "registry", None)
            legends = getattr(registry, "legends", None)
            if legends is None:
                return
            legends.default_font_scale = float(
                getattr(prefs, "legend_font_scale", 1.0) or 1.0
            )

        _apply(_PREF.current)
        self._legend_pref_unsub = _PREF.subscribe(_apply)

    # ------------------------------------------------------------------
    # Colour-legend direct manipulation (ADR 0081 L2)
    # ------------------------------------------------------------------

    def _legends(self):
        """The viewport's ``LegendController``, or ``None``."""
        director = getattr(self, "_director", None)
        registry = getattr(director, "registry", None)
        return getattr(registry, "legends", None)

    def _legend_snapshots(self) -> list:
        """Colour-scale placement for the session file (ADR 0081 L3)."""
        legends = self._legends()
        if legends is None:
            return []
        from .diagrams._session import LegendSnapshot
        return [LegendSnapshot(**raw) for raw in legends.snapshot()]

    # ------------------------------------------------------------------
    # Section planes (ADR 0083 S1)
    # ------------------------------------------------------------------

    def _clip_planes(self):
        """The viewport's ``ClipPlaneSetController``, or ``None``."""
        director = getattr(self, "_director", None)
        registry = getattr(director, "registry", None)
        return getattr(registry, "clip_planes", None)

    def _clip_plane_specs(self) -> tuple:
        """The active half-spaces, for the off-seam actor builders."""
        controller = self._clip_planes()
        if controller is None:
            return ()
        try:
            return controller.specs()
        except Exception:
            return ()

    def _clip_plane_snapshots(self) -> list:
        """The section-plane set for the session file (ADR 0083 P5)."""
        controller = self._clip_planes()
        if controller is None:
            return []
        from .diagrams._session import ClipPlaneSnapshot
        return [ClipPlaneSnapshot(**raw) for raw in controller.snapshot()]

    def _reapply_clip_planes(self, _kind=None, _payload=None) -> None:
        """Re-cut the off-seam actors (ADR 0083 Part 3).

        The backend re-stamps everything that went through
        ``add_layer`` on its own; substrate fill / wireframe / node
        cloud never do, so they are re-cut here on every
        ``CLIP_PLANES_CHANGED``. RENDER lane: the flags must land in
        the same frame the dispatcher then paints.
        """
        from .backends.pyvista_qt import apply_clip_planes

        specs = self._clip_plane_specs()
        # The boot pair is listed explicitly as well as through the
        # per-geometry map: a viewer that booted without a geometry
        # never registered it there, and re-cutting an actor twice is
        # a no-op (the helper replaces the set rather than appending).
        actors: list = [
            self._node_cloud_actor,
            self._substrate_actor,
            self._wireframe_actor,
        ]
        for pair in getattr(self, "_scene_actors", {}).values():
            actors.extend(pair)
        for actor in actors:
            if actor is not None:
                apply_clip_planes(actor, specs)

    def _cut_face_sources(self) -> list:
        """Diagrams that can back a cut face (ADR 0083 S3).

        Duck-typed on ``cut_face_source`` rather than on a diagram
        kind: what the cap needs is a volumetric layer plus the colour
        scale to share, and any diagram that can offer both is welcome
        to. Today that is the contour.
        """
        director = getattr(self, "_director", None)
        registry = getattr(director, "registry", None)
        if registry is None:
            return []
        sources = []
        for diagram in registry.diagrams():
            provider = getattr(diagram, "cut_face_source", None)
            if provider is None:
                continue
            try:
                source = provider()
            except Exception:
                source = None
            if source is not None:
                sources.append(source)
        return sources

    def _clip_gesture_settled(self) -> bool:
        """``False`` while a gizmo drag is in flight (ADR 0083 S3)."""
        return not getattr(
            self._clip_gizmo_interactor, "is_dragging", False,
        )

    def _on_clip_drag_end(self) -> None:
        """Recompute the cut faces the moment the gizmo is released.

        The gesture's end changes no plane state, so it fires no
        ``CLIP_PLANES_CHANGED`` — hence the direct push, and the direct
        render that goes with it.
        """
        if self._clip_cut_faces is None:
            return
        try:
            self._clip_cut_faces.refresh(force=True)
        except Exception:
            return
        plotter = getattr(self, "_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    def _refresh_substrate_surfaces(self, _kind=None, _payload=None) -> None:
        """Re-extract every substrate render surface whose ghosts moved.

        The ghost half of the substrate fast path (see
        :func:`add_substrate_actors`): a per-cell visibility change
        must re-run the O(volume) extraction so dropped faces vanish
        and interior faces reveal exactly as the volumetric mapper
        would. ``refresh_render_surface`` compares the ghost bytes
        first, so scenes the event didn't touch cost a memcmp. The
        boot pair is listed explicitly as well as through the
        per-geometry map, mirroring ``_reapply_clip_planes``.
        """
        from .backends.pyvista_qt import refresh_render_surface

        entries: list = []
        pairs = getattr(self, "_scene_actors", None) or {}
        actor_scenes = getattr(self, "_actor_scenes", None) or {}
        for pair in pairs.values():
            entry = actor_scenes.get(id(pair[0]))
            if entry is not None:
                entries.append((entry[1], pair))
        if not pairs:
            entries.append((
                self._scene,
                (self._substrate_actor, self._wireframe_actor),
            ))
        for g_scene, pair in entries:
            if g_scene is None:
                continue
            rs = getattr(g_scene, "render_surface", None)
            if rs is None:
                continue
            refresh_render_surface(
                g_scene.grid, rs, [a for a in pair if a is not None],
            )

    def _camera_view_normal(self):
        """The camera's view direction — the "current view" add choice."""
        plotter = getattr(self, "_plotter", None)
        if plotter is None:
            return None
        try:
            fx, fy, fz = plotter.camera.focal_point
            px, py, pz = plotter.camera.position
        except Exception:
            return None
        return (fx - px, fy - py, fz - pz)

    def _install_legend_interactor(self) -> None:
        from .core._legend_interactor import install_legend_interactor

        legends = self._legends()
        if legends is None:
            return
        self._legend_interactor = install_legend_interactor(
            legends._backend, legends,
            on_context_menu=self._show_legend_menu,
        )

    def _install_clip_gizmo_interactor(self) -> None:
        """Gizmo gestures (ADR 0083 S2) — call after the legend one."""
        from .core._clip_gizmo_interactor import (
            install_clip_gizmo_interactor,
        )

        controller = self._clip_planes()
        gizmos = self._clip_gizmos
        if controller is None or gizmos is None:
            return
        registry = getattr(self._director, "registry", None)
        backend = getattr(registry, "backend", None)
        if backend is None:
            return
        self._clip_gizmo_interactor = install_clip_gizmo_interactor(
            backend, controller, gizmos,
        )
        if self._clip_gizmo_interactor is not None:
            # The cut-face pass recomputes on release, not per tick
            # (ADR 0083 S3) — this is the release half of that deal.
            self._clip_gizmo_interactor.on_drag_end = self._on_clip_drag_end

    def _install_scope_gizmo_interactor(self) -> None:
        """Scope-box gestures — call after the clip gizmo's."""
        from .core._scope_gizmo_interactor import (
            install_scope_gizmo_interactor,
        )

        gizmos = self._scope_gizmos
        director = self._director
        if gizmos is None or director is None:
            return
        backend = getattr(getattr(director, "registry", None), "backend", None)
        if backend is None:
            return
        self._scope_gizmo_interactor = install_scope_gizmo_interactor(
            backend, director.scopes, gizmos,
            on_changed=self._fire_geometry_scope_changed,
        )
        if self._scope_gizmo_interactor is not None:
            # The mask pass recomputes on release, not per tick — this
            # is the release half of that deal (see
            # ``_on_geometry_scope_changed``).
            self._scope_gizmo_interactor.on_drag_end = (
                self._on_scope_drag_end
            )

    def _scope_gesture_settled(self) -> bool:
        """``False`` while a scope-gizmo drag is in flight."""
        return not getattr(
            self._scope_gizmo_interactor, "is_dragging", False,
        )

    def _on_scope_drag_end(self, geometry_id: str) -> None:
        """Recompute the scope masks the moment the gizmo is released.

        The gesture's end changes no box state, so it announces nothing
        of its own; firing the same event once more is what lets the
        pass that :meth:`_on_geometry_scope_changed` skipped during the
        drag finally run — and it rides the dispatcher's own coalesced
        render rather than calling ``render()`` here (ADR 0056 INV-5).
        """
        self._fire_geometry_scope_changed(geometry_id)

    def _fire_scope_gizmo_changed(self) -> None:
        """Announce a show/hide of the scope gizmo (F7).

        Its OWN render-only event, not the mask one: the box has not
        moved, so there is nothing to recompute — only the gizmo layers
        to reconcile and the scene to paint.
        """
        from .diagrams._dispatch import SCOPE_GIZMO_CHANGED

        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is None:
            return
        dispatcher.fire(SCOPE_GIZMO_CHANGED)

    def _fire_geometry_scope_changed(self, geometry_id: str) -> None:
        """Announce a gizmo-driven box move.

        ``ScopeController`` deliberately takes no collaborators — that
        is what keeps the scope off the STEP path — so unlike
        ``ClipPlaneSetController`` it cannot fire its own event. The
        drag therefore announces the same way the settings panel does,
        and the RENDER-lane subscriber above recomputes the mask.
        """
        from .diagrams._dispatch import GEOMETRY_SCOPE_CHANGED

        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is None:
            return
        dispatcher.fire(GEOMETRY_SCOPE_CHANGED, payload=geometry_id)

    def _diagram_for_legend(self, entry):
        """The attached diagram feeding ``entry``, or ``None``.

        A legend can have several sources; the menu's colour actions
        only need one of them, since they all share the LUT.
        """
        director = getattr(self, "_director", None)
        registry = getattr(director, "registry", None)
        if registry is None:
            return None
        for d in registry.diagrams():
            if getattr(d, "_legend_key", None) == entry.key:
                return d
        return None

    def _show_legend_menu(self, entry, pos) -> None:
        """Right-click menu on a colour scale.

        The discoverable surface for the scale: the settings tab keeps
        the same knobs for precise and scriptable edits, but nobody
        finds them by right-clicking the thing they want to change.
        """
        from qtpy import QtCore, QtWidgets

        legends = self._legends()
        if legends is None:
            return
        menu = QtWidgets.QMenu(self._plotter_widget())
        key = entry.key

        menu.addAction("Hide this scale").triggered.connect(
            lambda: legends.set_visible(key, False)
        )
        orient = menu.addAction("Horizontal" if entry.vertical else "Vertical")
        orient.triggered.connect(
            lambda: legends.set_vertical(key, not entry.vertical)
        )
        reset = menu.addAction("Reset size")
        reset.triggered.connect(lambda: legends.set_font_scale(key, 1.0))
        if entry.slot is None:
            menu.addAction("Snap back to the edge").triggered.connect(
                lambda: legends.redock(key)
            )
        menu.addSeparator()

        fmt_action = menu.addAction("Number format…")
        fmt_action.triggered.connect(lambda: self._prompt_legend_fmt(entry))

        diagram = self._diagram_for_legend(entry)
        if diagram is not None:
            if hasattr(diagram, "autofit_clim_at_current_step"):
                menu.addAction("Rescale to data").triggered.connect(
                    lambda: diagram.autofit_clim_at_current_step()
                )
            menu.addAction("Edit colour map…").triggered.connect(
                lambda: self._open_color_map_editor(diagram)
            )

        menu.exec_(self._plotter_widget().mapToGlobal(
            QtCore.QPoint(int(pos[0]), int(pos[1]))
        ))

    def _prompt_legend_fmt(self, entry) -> None:
        from qtpy import QtWidgets

        legends = self._legends()
        if legends is None:
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self._plotter_widget(), "Number format",
            "printf format for the tick labels (e.g. %.3g, %.2e, %.4f):",
            text=entry.fmt,
        )
        if ok and text:
            legends.set_fmt(entry.key, text)

    def _plotter_widget(self):
        """The Qt widget the plotter renders into, for menu parenting.

        ``QtInteractor`` *is* the widget; its ``.interactor`` attribute
        is the VTK interactor, which has no ``mapToGlobal``.
        """
        plotter = getattr(self, "_plotter", None)
        if plotter is not None and hasattr(plotter, "mapToGlobal"):
            return plotter
        return getattr(self, "_window", None)

    def _open_color_map_editor(self, diagram) -> None:
        """Focus the colour-map editor dock on ``diagram``."""
        editor = getattr(self, "_color_editor", None)
        if editor is None:
            return
        try:
            editor.bind_layer(diagram)
        except Exception:
            return
        action = getattr(self, "_color_editor_action", None)
        if action is not None and not action.isChecked():
            try:
                action.trigger()
            except Exception:
                pass

    def _on_shift_click_world(self, world_pos, prop=None) -> None:
        """Shift-click callback — open a time-history for the picked node.

        Snaps the shift-click world position to the nearest FEM node
        via the probe overlay, picks a default component (the first
        active diagram's component, falling back to the first
        available nodal component), and opens a plot-pane history
        tab.

        ADR 0058 S3b — pin-aware: ``prop`` (the picked actor, handed
        through by ``install_navigation``) resolves to the hit
        geometry via the S2c actor→scene map; the snap reads THAT
        geometry's grid and the history is scoped to its stage pin
        (``None`` pin / unattributed prop = active stage, the legacy
        behavior).
        """
        if (
            self._director is None
            or self._probe_overlay is None
            or self._plot_pane is None
        ):
            return
        geometry_id, _scene = self._resolve_pick_scene(
            id(prop) if prop is not None else None,
        )
        try:
            node_id, _, _ = self._probe_overlay._snap_to_nearest_node(
                world_pos,
                scene=self._scene_for_geometry_id(geometry_id),
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
        stage_pin = None
        if geometry_id is not None:
            geom = self._director.geometries.find(geometry_id)
            stage_pin = getattr(geom, "stage_id", None) if geom else None
        self._open_time_history(
            int(node_id), component, stage_id=stage_pin,
        )

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

    def _open_time_history(
        self, node_id: int, component: str, *,
        stage_id: "Optional[str]" = None,
    ) -> None:
        """Open (or focus) a TimeHistoryPanel as a plot-pane tab.

        Reuses an existing tab if one is already open for the same
        ``(node_id, component, stage_id)`` so repeated shift-clicks on
        the same node don't multiply tabs. ``stage_id`` scopes the
        history read to one stage (ADR 0058 S3b — the picked
        geometry's stage pin); ``None`` keeps the active-stage read.
        """
        if self._director is None or self._plot_pane is None:
            return
        key = ("history", int(node_id), str(component), stage_id)
        if self._plot_pane.has_tab(key):
            self._plot_pane.set_active(key)
            return
        try:
            from .ui._time_history import TimeHistoryPanel
            panel = TimeHistoryPanel(
                self._director, node_id, component, stage_id=stage_id,
            )
            # Migrate the panel's step / stage subscriptions onto the
            # dispatcher's UI lane so rapid scrubber drags collapse to
            # one marker redraw per Qt tick.
            if self._dispatcher is not None:
                panel.attach_dispatcher(self._dispatcher)
        except Exception as exc:
            from ._failures import report
            report("ResultsViewer._open_time_history", exc)
            return
        label = f"u(t) · node {node_id} · {component}"
        if stage_id is not None:
            label += f" · {stage_id}"
        self._plot_pane.add_tab(key, label, panel.widget, closable=True)
        self._history_panels[
            (int(node_id), str(component), stage_id)
        ] = panel
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
            # Migrate the panel's legacy director subs onto the
            # dispatcher's UI lane if it advertises attach_dispatcher.
            # Duck-typed: panels that haven't been migrated (no method)
            # keep their legacy wiring; migrated ones swap to coalesced
            # dispatcher subs so a rapid scrubber drag collapses to one
            # redraw per Qt tick.
            if self._dispatcher is not None and hasattr(
                panel, "attach_dispatcher",
            ):
                try:
                    panel.attach_dispatcher(self._dispatcher)
                except Exception as exc:
                    from ._failures import report
                    report(
                        f"ResultsViewer._sync_side_panels.attach_dispatcher"
                        f"({type(d).__name__})",
                        exc,
                    )
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
            _, node_id, component, stage_id = key
            panel = self._history_panels.pop(
                (node_id, component, stage_id), None,
            )
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
        """Esc → clear pick visuals.

        Drops every probe marker (P1, P2, …), element / GP highlights,
        and the picked-readout HUD. Leaves the active composition
        unchanged so layered diagrams stay visible — clearing
        composition selection here previously hid every diagram via
        the composition gate, which read as "diagrams broken".
        """
        from ._log import log_action
        log_action("ui.shortcut", "escape")
        try:
            from qtpy import QtWidgets
            fw = QtWidgets.QApplication.focusWidget()
            if fw is not None:
                fw.clearFocus()
        except Exception:
            pass
        if self._director is None:
            return
        if self._probe_overlay is not None:
            try:
                self._probe_overlay.clear()
            except Exception:
                pass
        self._clear_element_highlight()
        self._clear_gp_highlight()
        try:
            if self._plotter is not None:
                self._plotter.render()
        except Exception:
            pass

    def _on_active_composition_changed(self, key) -> None:
        """ActiveObjects → composition selection changed.

        Subscribed in :meth:`_show_impl`. ``key`` is the composition
        id from the outline (or ``None`` when cleared). Routes the
        layer-stack view into the dedicated Diagram dock. Pre-plan-04
        this lived as ``_on_outline_composition_selected`` directly
        wired from the outline's callback registry; now it's just one
        of N possible subscribers.

        Plan 06 step 4: also resolves a default active layer for the
        Color Map Editor (first LUT-bearing layer in the composition,
        or ``None``).
        """
        if key is None:
            # Composition cleared — drop the active layer too so the
            # editor collapses to its empty state.
            self._active.set_active_layer(None)
            return
        try:
            self._settings_tab.show_stack()
        except Exception:
            pass
        win = self._win
        if win is not None:
            win.raise_diagram_dock()
        self._refresh_active_layer()

    def _refresh_active_layer(self) -> None:
        """Pick a sensible default active layer and broadcast.

        Preserves the user's explicit pick (card focus from plan 04
        step 2 cont.) when that layer is still in the active
        composition. Falls back to the first layer with a LUT when the
        prior pick is gone — typically after add/remove churn or when
        a fresh composition becomes active. Clearing to ``None`` means
        the composition has no LUT-bearing layers; the editor goes
        empty.
        """
        if self._director is None:
            return
        try:
            comp = self._director.compositions.active
        except Exception:
            comp = None
        layers = list(getattr(comp, "layers", []) or [])
        current = self._active.active_layer
        # User-picked layer still valid in this composition? Keep it.
        if current is not None and current in layers:
            return
        chosen = None
        for layer in layers:
            if getattr(layer, "lut", None) is not None:
                chosen = layer
                break
        # set_active_layer's identity-based no-op short-circuits when
        # nothing changed, so repeated calls during refresh storms are
        # cheap.
        self._active.set_active_layer(chosen)

    def _on_active_geometry_changed(self, geom_id) -> None:
        """ActiveObjects → geometry selection changed.

        Routes the geometry settings view into the dedicated Geometry
        dock.
        """
        if geom_id is None:
            return
        if self._geometry_panel is not None:
            try:
                self._geometry_panel.show_geometry(geom_id)
            except Exception:
                pass
        win = self._win
        if win is not None:
            win.raise_geometry_dock()


