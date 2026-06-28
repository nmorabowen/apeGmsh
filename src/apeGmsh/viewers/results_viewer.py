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
    """
    from qtpy import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _gate_visible_layer_ids(geom_mgr: Any) -> "set[int]":
    """Layer ids (``id(layer)``) the composition gate may show.

    ADR 0058 S2b — concurrent rendering: every geometry with
    ``visible=True`` contributes its layers. Per geometry the existing
    composition-gate semantics are preserved: when a composition is
    active there, only that composition's layers; otherwise all of its
    compositions' layers. A layer is then shown iff
    ``layer.is_visible AND id(layer) in this set`` (the GATE pump
    composes the two).

    Module-level (not a ``show()`` closure) so the truth table is
    testable headless.
    """
    visible_layers: set[int] = set()
    for geom in geom_mgr.geometries:
        if not geom.visible:
            continue
        active_comp = geom.compositions.active
        if active_comp is not None:
            visible_layers.update(map(id, active_comp.layers))
        else:
            for c in geom.compositions.compositions:
                visible_layers.update(map(id, c.layers))
    return visible_layers


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


def _compose_substrate_points(
    reference_points: Any,
    offset: Any,
    field_vals: Any,
    scale: float,
) -> Any:
    """DEFORM pump point composition (ADR 0058 S3a).

    ``reference + offset + scale·field`` — the geometry's spatial
    offset is a pump-time term, never an actor transform and never
    baked into ``reference_points`` (world coordinates stay grid
    coordinates — the S2c picking invariant).

    Returns ``None`` only when there is nothing to apply (no field AND
    zero offset) — the byte-identical legacy fast-path: the pump then
    resets ``grid.points`` to reference and tells diagrams "back to
    reference". A deform-off geometry with a non-zero offset returns
    ``reference + offset``.

    Module-level (not a ``show()`` closure) so the composition rule is
    testable headless.
    """
    import numpy as np

    off = (
        np.asarray(offset, dtype=np.float64)
        if offset is not None else None
    )
    has_offset = off is not None and bool(np.any(off != 0.0))
    if field_vals is None and not has_offset:
        return None
    pts = np.asarray(reference_points, dtype=np.float64).copy()
    if has_offset:
        pts += off
    if field_vals is not None:
        pts += float(scale) * np.asarray(field_vals, dtype=np.float64)
    return pts


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
                window_size=window_size,
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

        # ── Window (creates QApplication) ───────────────────────────
        title = self._title or self._default_title()
        win = ResultsWindow(
            title=title,
            on_close=self._on_close,
            extension_docks=[output_spec, color_editor_spec, definitions_spec],
        )
        self._win = win

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
        geometry_panel = GeometrySettingsPanel(
            director, deform_field_options,
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
        # One fill + wireframe pair per scene. The fill is drawn
        # first; edges render separately (style='wireframe') so the
        # lines stay on top of any contour / deformed-shape diagram
        # that draws polygons at the same z-depth. Polygon-offset on
        # the line mapper resolves coincident topology by pulling
        # lines toward the camera, eliminating z-fighting between the
        # wireframe and the substrate / diagram polygons. Called once
        # here for the boot scene and again from the scene factory for
        # every materialized per-geometry scene (unique actor names —
        # pyvista replaces same-named actors).
        def _add_substrate_actors(
            g_scene: "FEMSceneData",
            *,
            name_suffix: str = "",
            reset_camera: bool = False,
        ):
            p = THEME.current
            fill = plotter.add_mesh(
                g_scene.grid,
                color=p.substrate_color,
                show_edges=False,
                opacity=prefs.mesh_surface_opacity,
                lighting=True,
                smooth_shading=False,
                name=f"results_substrate{name_suffix}",
                reset_camera=reset_camera,
            )
            wf = plotter.add_mesh(
                g_scene.grid,
                style="wireframe",
                color=p.substrate_edge_color,
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
            return fill, wf

        actor, wireframe_actor = _add_substrate_actors(
            scene, reset_camera=True,
        )
        self._substrate_actor = actor
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
            from .scene.glyph_points import build_node_cloud as _build
            old = self._node_cloud_actor
            try:
                plotter.remove_actor(old)
            except Exception:
                pass
            # Rebuild against the ACTIVE geometry's scene (ADR 0058
            # S2a — the node cloud is an active-only display overlay).
            g_scene = self._scene or scene
            try:
                _, new_actor = _build(
                    plotter,
                    g_scene.grid.points,
                    model_diagonal=g_scene.model_diagonal,
                    marker_size=float(value),
                    color=THEME.current.node_accent,
                )
            except Exception:
                new_actor = None
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
        # The dispatcher below is the only place that composes these
        # into event-driven sequences. Don't call them directly from
        # observers — fire a dispatcher event instead.

        def _compute_deformed_pts(geom, step: int) -> "_np.ndarray | None":
            """Resolve ``geom``'s substrate points at ``step``.

            ADR 0058 S3a: ``reference + offset + scale·field``. Returns
            non-None whenever deformation contributes OR the geometry
            carries a non-zero spatial ``offset`` (a deform-off offset
            geometry yields ``reference + offset``). Returns None only
            for the legacy fast-path — no deformation (disabled / no
            field / unreadable) AND zero offset; the caller then resets
            the substrate to reference.

            ADR 0058 S1: geometry-parameterized — the baseline comes
            from the geometry's own scene (today the single bound
            scene; per-geometry in S2). The field reader stays
            geometry-agnostic: every geometry's scene indexes the same
            model, so ``node_ids`` / the id→row map are shared.

            ADR 0058 S3b: pinned-or-active — a geometry with a stage
            pin reads its PINNED stage's field at the global cursor
            clamped into the pinned range
            (``director.local_step_for_stage``); unpinned geometries
            keep the active stage + raw step.
            """
            if geom is None:
                return None
            field_vals = None
            if geom.deform_enabled and geom.deform_field:
                pin = getattr(geom, "stage_id", None)
                if pin:
                    field_vals = _read_deform_field(
                        geom.deform_field,
                        int(director.local_step_for_stage(pin)),
                        stage_id=pin,
                    )
                else:
                    field_vals = _read_deform_field(
                        geom.deform_field, int(step),
                    )
            g_scene = director.scene_for(geom) or scene
            return _compose_substrate_points(
                g_scene.reference_points,
                getattr(geom, "offset", None),
                field_vals,
                float(geom.deform_scale),
            )

        def _effective_step_for(layer) -> int:
            """Step to push to ``layer`` (ADR 0058 S3b — pin-aware).

            A layer owned by a stage-PINNED geometry steps through the
            pinned stage: the global cursor clamped into its range via
            ``director.local_step_for_stage``. Unpinned (or ownerless)
            layers receive the raw global step.
            """
            geoms = director.geometries
            owner = geoms.geometry_for_layer(layer)
            pin = getattr(owner, "stage_id", None) if owner is not None else None
            if pin:
                return int(director.local_step_for_stage(pin))
            return int(director.step_index)

        def _pump_step(layer) -> None:
            """STEP primitive — push current step values.

            ADR 0058 S3b: pin-aware — the full path loops the registry
            pushing each diagram's effective step (clamped for layers
            of pinned geometries, raw otherwise); the layer-scoped
            path resolves its one owner the same way.
            """
            if layer is not None:
                try:
                    layer.update_to_step(_effective_step_for(layer))
                except Exception:
                    pass
                return
            for d in director.registry.diagrams():
                if not (d.is_attached and d.is_visible):
                    continue
                try:
                    d.update_to_step(_effective_step_for(d))
                except Exception:
                    pass

        def _pump_deform(layer) -> None:
            """DEFORM primitive.

            ``layer=None``: full pump — for every rendering geometry
            (ADR 0058 S2b: every geometry with ``visible=True``),
            recompute its deformed_pts from ITS deform state, mutate
            its scene's ``grid.points``, sync the node cloud (active
            geometry only — it is the editing overlay), and fan out
            to THAT geometry's diagrams. Hidden geometries' diagrams
            are gate-hidden and re-pumped when shown again
            (``GEOMETRY_VISIBILITY_CHANGED`` runs DEFORM), so they
            never show stale positions.

            ``layer=<diagram>``: scoped — sync that one diagram against
            its OWNING geometry's state. Used after a single layer's
            attach / re-attach so existing diagrams aren't re-pumped.
            """
            step = int(director.step_index)
            geoms = director.geometries
            if layer is not None:
                geom = geoms.geometry_for_layer(layer) or geoms.active
                g_scene = director.scene_for(geom) or scene
                deformed_pts = _compute_deformed_pts(geom, step)
                try:
                    layer.sync_substrate_points(deformed_pts, g_scene)
                except Exception:
                    pass
                return
            for geom in self._render_geometries():
                g_scene = director.scene_for(geom) or scene
                deformed_pts = _compute_deformed_pts(geom, step)
                if deformed_pts is None:
                    g_scene.grid.points = g_scene.reference_points.copy()
                else:
                    g_scene.grid.points = deformed_pts
                if geom is geoms.active:
                    self._sync_node_cloud(deformed_pts)
                self._sync_diagram_substrate_points(
                    deformed_pts, geometry=geom, scene=g_scene,
                )

        def _pump_gate() -> None:
            """GATE primitive — composition-based actor visibility.

            ADR 0058 S2b: a layer is shown iff ``layer.is_visible AND
            composition gate AND owning_geometry.visible``. Every
            geometry with ``visible=True`` contributes layers (per
            geometry: the active composition's layers when one is
            active there, else all of its compositions' layers — see
            :func:`_gate_visible_layer_ids`). Layers owned by hidden
            geometries are gate-hidden, so toggling a geometry's
            ``visible`` flag drops / restores its diagrams without
            touching their user-intent ``is_visible`` flags.

            No render here; the dispatcher coalesces RENDER per event.
            """
            visible_layers = _gate_visible_layer_ids(director.geometries)
            for d in director.registry.diagrams():
                in_active = id(d) in visible_layers
                desired = bool(d.is_visible) and in_active
                # Polymorphic: migrated diagrams route backend layer
                # handles, legacy ones flip raw actors. The user-intent
                # flag (is_visible) is preserved — only the rendered
                # artifacts follow the gate.
                try:
                    d.apply_effective_visibility(desired)
                except Exception:
                    pass

        def _pump_restack() -> None:
            """Re-stack actors so paint order matches the layer order
            in the registry.

            VTK paints actors in the order they were added; reordering
            via the ↑ / ↓ buttons updates the registry list but doesn't
            move the actors. Detach + re-attach each diagram in order
            (cheap when the polydata is cached on the instance).
            """
            # Re-attach through the registry's RenderBackend (ADR 0042
            # R-B.final) — attach injects a backend, not a raw plotter.
            # ADR 0058 S2a: resolve each diagram's scene through its
            # owning geometry so a restack doesn't re-bind a layer to
            # the wrong substrate.
            backend = director.registry.backend
            for d in list(director.registry.diagrams()):
                if not d.is_attached:
                    continue
                try:
                    d.detach()
                    d.attach(
                        backend, director.view,
                        director._scene_for_diagram(d),  # noqa: SLF001
                    )
                except Exception as exc:
                    # d is now detached but failed to re-attach; log the
                    # error so the broken state is visible rather than
                    # silently swallowed.
                    from ._log import log_error
                    log_error("dispatch", "restack", exc, diagram=id(d))
                    continue

        # Stash for the rest of the file (probe overlay etc. still
        # call _apply_composition_gate by name in some teardown paths).
        self._apply_composition_gate = _pump_gate

        # ── Dispatcher — single-source event pipeline ────────────────
        from .diagrams._dispatch import (
            STEP_CHANGED, DEFORM_CHANGED, STAGE_CHANGED,
            COMP_ACTIVE_CHANGED, DIAGRAM_ATTACHED,
            DIAGRAM_DETACHED, DIAGRAM_MODIFIED,
            LAYER_VISIBILITY_CHANGED, LAYER_REORDERED, PICK_CLEARED,
            GEOMETRIES_CHANGED,
            GEOMETRY_ACTIVE_CHANGED, GEOMETRY_OFFSET_CHANGED,
            GEOMETRY_REMOVED, GEOMETRY_STAGE_PIN_CHANGED,
            GEOMETRY_VISIBILITY_CHANGED, Lane,
        )
        # ADR 0056 Part 3: the director constructed its dispatcher at
        # __init__ (no-op pumps); rebind the real pumps now that the
        # plotter / scene / actor list exist.
        dispatcher = director.dispatcher
        dispatcher.bind(
            pump_step=_pump_step,
            pump_deform=_pump_deform,
            pump_gate=_pump_gate,
            pump_restack=_pump_restack,
            render=_render,
        )
        self._dispatcher = dispatcher
        # (ADR 0047 R-D.2b: the PickInventory no longer carries a
        # dispatcher — the dead set_pick_mode / PICK_MODE_CHANGED path is
        # gone.)
        # ElementVisibility gets the dispatcher so hide/show fires ELEMENT_VISIBILITY_CHANGED.
        scene.element_visibility.dispatcher = dispatcher
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

        # Stashed so _apply_session can run it once after its
        # suppressed batch flushes (RENDER-lane subscribers don't
        # replay on batch exit).
        self._sync_substrate_visibility = _sync_substrate_visibility

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
        add_help_shortcuts_menu(
            win.window,
            entries=[
                ("N / E / G", "Pick mode — node / element / gauss point"),
                ("Shift+LMB drag", "Turntable (yaw-only around up axis)"),
                ("Shift+MMB drag", "Orbit (yaw + pitch, no-roll)"),
                ("MMB / RMB drag", "Pan"),
                ("Scroll", "Zoom (focal point fixed)"),
                ("Shift+click", "Time-history at node"),
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

        # Composition viewport gate is now ``_pump_gate`` above and
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
            except Exception:
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
        for attr in ("_step_unsub", "_stage_unsub"):
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
                geom_snapshots.append(GeometrySnapshot(
                    id=geom.id,
                    name=geom.name,
                    deform_enabled=bool(geom.deform_enabled),
                    deform_field=geom.deform_field,
                    deform_scale=float(geom.deform_scale),
                    offset=tuple(float(c) for c in geom.offset),
                    stage_id=geom.stage_id,
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
        """Reconstruct each spec into a Diagram and rebuild the
        Geometry → Composition → Layer hierarchy.

        v2 sessions carry the full hierarchy (with deformation state
        per geometry). v1 sessions (no ``geometries`` block) bundle
        every restored layer into one "Restored" composition under
        the active geometry — same fallback as before.
        """
        from .diagrams._base import NoDataError
        from .diagrams._kinds import kind_def

        # Suppress per-add registry pumps during the bulk restore.
        # Without this, the registry observer fires K times for K
        # layers, and each fire re-pumps every other layer
        # (K(K+1)/2 cost). The batch context runs one full pump on
        # exit instead. The dispatcher always exists (ADR 0056 Part 3).
        _batch_cm = self._director.dispatcher.session_batch()
        _batch_cm.__enter__()
        # ADR 0026 PR-stretch — the director's tag-map source is now
        # derived from the bound :class:`Results`, not a separate
        # ``_model_h5`` path field.  bind_results(self._results)
        # ensures the director's tag_map property resolves correctly
        # at construction time for any SectionCutDiagram restored
        # below.  The session payload's ``model_h5`` field is now
        # informational only (kept in the schema for one cycle so
        # older session JSONs still parse).
        try:
            self._director.bind_results(self._results)
        except Exception as exc:
            from ._failures import report
            report(
                "ResultsViewer._apply_session(bind_results)", exc,
            )
        n_added = 0
        n_skipped = 0
        # Build every Diagram instance and stash it at the same index
        # the session JSON used (None for skipped specs so layer_indices
        # references stay aligned).
        restored_layers: list[Any] = []
        for spec in session.diagrams:
            kdef = kind_def(spec.kind)
            cls = kdef.diagram_class if kdef is not None else None
            if cls is None:
                n_skipped += 1
                restored_layers.append(None)
                continue
            try:
                if spec.kind == "section_cut":
                    tag_map = self._director.tag_map
                    diagram = cls(
                        spec, self._results, tag_map=tag_map,
                    )
                else:
                    diagram = cls(spec, self._results)
                self._director.registry.add(diagram)
                restored_layers.append(diagram)
                n_added += 1
            except NoDataError:
                n_skipped += 1
                restored_layers.append(None)
            except Exception as exc:
                from ._failures import report
                report(
                    f"ResultsViewer._apply_session({spec.kind})", exc,
                )
                n_skipped += 1
                restored_layers.append(None)

        geom_mgr = self._director.geometries
        geom_snapshots = getattr(session, "geometries", ()) or ()

        if geom_snapshots:
            # ── v2: rebuild the full geometry tree ────────────────
            # The bootstrap already created one "Geometry 1"; reuse
            # it for the first snapshot and add() for the rest.
            # ADR 0058 S2b — sessions saved before the ``visible``
            # flag existed (snapshot field None) map to "visible iff
            # active", reproducing their previous active-only render.
            # The active pointer is restored after this loop, so the
            # legacy mapping is deferred until then.
            legacy_visible_ids: list[str] = []
            for gi, gsnap in enumerate(geom_snapshots):
                if gi == 0 and len(geom_mgr.geometries) >= 1:
                    geom = geom_mgr.geometries[0]
                    geom_mgr.rename(geom.id, gsnap.name)
                else:
                    geom = geom_mgr.add(name=gsnap.name, make_active=False)
                geom_mgr.set_deformation(
                    geom.id,
                    enabled=bool(gsnap.deform_enabled),
                    field=gsnap.deform_field,
                    scale=float(gsnap.deform_scale),
                )
                # ADR 0058 S3a — restore via the owner mutator (no-op
                # at the zero default; legacy sessions read (0,0,0)).
                geom_mgr.set_offset(geom.id, tuple(gsnap.offset))
                geom_mgr.set_display(
                    geom.id,
                    show_mesh=bool(gsnap.show_mesh),
                    show_nodes=bool(gsnap.show_nodes),
                    display_opacity=float(gsnap.display_opacity),
                )
                if gsnap.visible is None:
                    legacy_visible_ids.append(geom.id)
                else:
                    geom_mgr.set_visible(geom.id, bool(gsnap.visible))
                # Compositions inside this geometry.
                for csnap in gsnap.compositions:
                    comp = geom.compositions.add(
                        name=csnap.name, make_active=False,
                    )
                    for idx in csnap.layer_indices:
                        if 0 <= idx < len(restored_layers):
                            d = restored_layers[idx]
                            if d is not None:
                                geom.compositions.add_layer(comp.id, d)
                # Restore active composition pointer.
                if gsnap.active_composition_id is not None:
                    # Find the composition we just added by its
                    # POSITION — saved ids are stale UUIDs.
                    matches = [
                        c for c in geom.compositions.compositions
                        if c.name == self._comp_name_for(gsnap, gsnap.active_composition_id)
                    ]
                    if matches:
                        geom.compositions.set_active(matches[0].id)
                # Heal stale sessions: pre-#71/#72 the active id could
                # be saved as None (Esc / Geometry-row click). Without
                # an active comp the gate hides every layer on restore,
                # so default to the first composition when the user
                # didn't explicitly pick one.
                if (
                    geom.compositions.active is None
                    and geom.compositions.compositions
                ):
                    first = geom.compositions.compositions[0]
                    geom.compositions.set_active(first.id)
                # ADR 0058 S3b — restore the stage pin via the owner
                # mutator, AFTER the composition/layer loop so the
                # director's reattach observer fires against recorded
                # membership, once, instead of churning per layer.
                # No-op at the None default (legacy sessions).
                geom_mgr.set_stage_pin(
                    geom.id, getattr(gsnap, "stage_id", None),
                )
            # Restore active geometry pointer (by name match — saved
            # UUIDs don't survive a re-bootstrap).
            saved_active = self._geom_name_for(
                geom_snapshots, session.active_geometry_id,
            )
            if saved_active:
                for g in geom_mgr.geometries:
                    if g.name == saved_active:
                        geom_mgr.set_active(g.id)
                        break
            # Legacy (pre-S2b) snapshots: visible = is-active, now
            # that the active pointer reflects the saved session.
            for gid in legacy_visible_ids:
                geom_mgr.set_visible(gid, gid == geom_mgr.active_id)
        else:
            # ── v1 fallback: bundle into one "Restored" composition ─
            real_layers = [d for d in restored_layers if d is not None]
            if real_layers:
                geom = geom_mgr.active
                if geom is not None:
                    comp = geom.compositions.add(
                        name="Restored", make_active=True,
                    )
                    for d in real_layers:
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

        # Flush the batch — runs STEP + DEFORM + GATE + RENDER once
        # against everything that was added during the loop above.
        try:
            _batch_cm.__exit__(None, None, None)
        except Exception:
            pass

        # ADR 0058 S2a/S2b — active-geometry / geometry-visibility
        # changes inside the suppressed batch never reach the
        # RENDER-lane substrate visibility subscriber; run the
        # idempotent sync once now.
        sync = getattr(self, "_sync_substrate_visibility", None)
        if sync is not None:
            try:
                sync()
            except Exception:
                pass

        try:
            msg = f"Restored {n_added} diagram(s)"
            if n_skipped:
                msg += f"; {n_skipped} skipped"
            win.set_status(msg, timeout=5000)
        except Exception:
            pass

    @staticmethod
    def _comp_name_for(
        gsnap: Any, comp_id: Optional[str],
    ) -> Optional[str]:
        """Look up a composition's name in a snapshot by its saved id."""
        if comp_id is None:
            return None
        for c in gsnap.compositions:
            if c.id == comp_id:
                return c.name
        return None

    @staticmethod
    def _geom_name_for(
        geom_snapshots: Any, geom_id: Optional[str],
    ) -> Optional[str]:
        if geom_id is None:
            return None
        for g in geom_snapshots:
            if g.id == geom_id:
                return g.name
        return None

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


