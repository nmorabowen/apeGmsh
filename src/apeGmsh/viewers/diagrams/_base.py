"""Diagram base + DiagramSpec record.

A Diagram is a renderable layer driven by one or more Results slabs.
Subclasses implement the four-method protocol:

* ``attach(plotter, view, scene)`` — build initial actors at step 0,
  resolve the selector to concrete IDs (once).  ``view`` is a
  :class:`apeGmsh.viewers.data.ViewerData` snapshot (Phase 8.7
  commit 5 renamed the parameter from ``fem``).
* ``update_to_step(step_index)`` — refresh actors for a new step.
  Must mutate VTK arrays in place; never re-add actors.
* ``detach()`` — remove actors and release cached arrays.
* ``settings_widget()`` — return a Qt widget for the diagram's
  per-diagram settings tab.

Phase 0 ships only the base. Concrete diagrams arrive in Phase 1+.

The ``DiagramSpec`` record is the persistable form of a Diagram —
``(kind, style, selector, stage_id)``. The Director can serialize a
list of specs to JSON; reconstructing diagrams from specs lets us
save / restore the active set of diagrams across sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ._selectors import SlabSelector
from ._styles import DiagramStyle

if TYPE_CHECKING:
    from numpy import ndarray
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


# =====================================================================
# Errors
# =====================================================================


class NoDataError(RuntimeError):
    """A diagram's ``attach()`` couldn't find any data to render.

    Raised when the slab read for the requested ``(component, stage,
    selector)`` returns empty — typically because the component name
    isn't recorded in this stage, or the selector resolved to elements
    that have no slab coverage. Caught by the Diagrams tab and surfaced
    as a status-bar message; without it the diagram would silently
    add a blank actor and look like a successful operation.
    """


# =====================================================================
# DiagramSpec — persistable form
# =====================================================================

@dataclass(frozen=True)
class DiagramSpec:
    """Persistable record describing a diagram instance.

    Frozen — saved/loaded by the Director, never mutated. The
    ``kind`` field is the registered diagram kind name (e.g.
    ``"contour"``, ``"line_force"``); the registry maps it back
    to the concrete class.
    """
    kind: str
    selector: SlabSelector
    style: DiagramStyle
    stage_id: Optional[str] = None
    visible: bool = True
    label: Optional[str] = None     # user-supplied display name


# =====================================================================
# Diagram base class
# =====================================================================

class Diagram:
    """Base class for all renderable result diagrams.

    Subclasses set the class-level ``kind`` attribute (string,
    registry key) and override ``attach``/``update_to_step``/
    ``detach`` (mandatory) plus ``settings_widget`` (optional —
    returns ``None`` for a blank settings panel).

    Lifecycle::

        diagram = ContourDiagram(spec, results)
        diagram.attach(plotter, view, scene) # builds actors
        diagram.update_to_step(0)            # initial render
        ...
        diagram.update_to_step(i)            # on time change
        ...
        diagram.detach()                     # on remove

    Performance contract (locked in Phase 0, enforced thereafter):

    * ``attach`` resolves the selector to concrete IDs **once**;
      subsequent ``update_to_step`` calls reuse them.
    * ``update_to_step`` mutates existing actor scalar arrays in
      place. Re-creating actors per step is forbidden — see the
      perf gates in ``plan_results_viewer.md``.
    * Slabs read for one step are released after the update; the
      diagram does not cache ``(T, …)`` arrays.
    """

    kind: str = ""        # subclasses must set
    topology: str = ""    # subclasses must set — names the Results
                          # composite that owns this diagram's data
                          # ("nodes", "line_stations", "fibers", "layers",
                          # "gauss", "springs"). Read by AddDiagramDialog
                          # to populate the Component combo from
                          # ``available_components()`` on that composite.

    # True for diagrams that paint an opaque filled surface extracted
    # from the substrate (contour). The viewer hides the geometry grey
    # substrate FILL when such a diagram is attached + visible so the
    # two coincident opaque surfaces do not z-fight (the grey would
    # bleed through the colour). The substrate WIREFRAME stays so
    # element edges remain visible. Non-occluding diagrams (glyphs,
    # line forces, ...) leave the fill alone.
    occludes_substrate: bool = False

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not self.kind:
            raise TypeError(
                f"{type(self).__name__} must declare a class-level "
                f"'kind' attribute (e.g. kind='contour')."
            )
        if spec.kind != self.kind:
            raise ValueError(
                f"{type(self).__name__} expects kind={self.kind!r}; "
                f"got spec.kind={spec.kind!r}."
            )
        self.spec = spec
        self._results = results

        # Attached state (populated by attach, cleared by detach)
        self._attached: bool = False
        self._plotter: Any = None
        # Render seam (ADR 0042, R-B): the RenderBackend migrated
        # diagrams emit SceneLayers through. Built from the plotter in
        # attach. Un-migrated diagrams ignore it and use self._plotter.
        self._backend: Any = None
        self._view: "ViewerData | None" = None
        self._scene: "FEMSceneData | None" = None
        self._resolved_node_ids: "ndarray | None" = None
        self._resolved_element_ids: "ndarray | None" = None
        self._actors: list[Any] = []
        self._visible: bool = spec.visible
        # Optional visual float16 cache stamped by the registry at
        # attach (None for headless tests / pre-bind). When present,
        # per-step diagrams slice a float16 row from RAM instead of
        # re-reading HDF5 every frame; when absent they fall back to
        # the per-step read path. See viewers.diagrams._visual_store.
        self._visual_store: Any = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def selector(self) -> SlabSelector:
        return self.spec.selector

    @property
    def style(self) -> DiagramStyle:
        return self.spec.style

    @property
    def stage_id(self) -> Optional[str]:
        return self.spec.stage_id

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def is_visible(self) -> bool:
        return self._visible

    def display_label(self) -> str:
        """User-facing label for the Diagrams tab list."""
        if self.spec.label:
            return self.spec.label
        return f"{self.kind} — {self.selector.short_label()}"

    def _effective_stage_id(self) -> Optional[str]:
        """Stage id this diagram's reads scope to, or ``None``.

        ADR 0058 S3b — the two pins compose: an explicit per-diagram
        ``spec.stage_id`` wins; otherwise the owning geometry's stage
        pin, resolved through the registry-stamped
        ``_stage_pin_resolver`` (``None`` when unstamped, ownerless,
        or the geometry is unpinned). The single home of the
        effective-stage lookup so the base and the
        ``ReactionsDiagram`` override can't drift.
        """
        if self.spec.stage_id is not None:
            return self.spec.stage_id
        resolver = getattr(self, "_stage_pin_resolver", None)
        if resolver is None:
            return None
        try:
            return resolver(self)
        except Exception:
            return None

    def _scoped_results(self) -> "Optional[Results]":
        """Return a Results scoped to the diagram's effective stage.

        A ``spec.stage_id`` pins the diagram to one stage
        (``Results.stage`` view); without one, the owning geometry's
        stage pin applies (ADR 0058 S3b, via
        :meth:`_effective_stage_id`); without either, reads go through
        the Results as-is. Shared by every concrete diagram's attach /
        update read path. (``ReactionsDiagram`` overrides with a
        defensive variant that returns ``None`` on a bad stage id.)
        """
        stage_id = self._effective_stage_id()
        if stage_id is not None:
            return self._results.stage(stage_id)
        try:
            return self._results
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Visual float16 cache (opt-in, stamped by the registry)
    # ------------------------------------------------------------------
    def _visual_stage_id(self):
        """Explicit stage pin (spec or geometry) for the visual cache.

        Same composition as _effective_stage_id (explicit spec pin
        wins; else the geometry stage pin). None when unpinned - the
        resolved stage then falls back to the scoped Results active
        stage via _visual_resolved_stage_id.
        """
        return self._effective_stage_id()

    def _visual_resolved_stage_id(self):
        """Real stage id the visual cache should key on.

        Resolves the stage the diagram actually reads from: an
        explicit spec/geometry pin wins; otherwise the scoped
        Results active stage (set by the director on the unscoped
        handle); otherwise the single stage of a one-stage file.
        None only when no stage can be resolved at all.
        """
        scoped = self._scoped_results()
        if scoped is None:
            return None
        sid = self._visual_stage_id()
        if sid is not None:
            return sid
        sid = getattr(scoped, "_stage_id", None)
        if sid is not None:
            return sid
        try:
            stages = list(scoped.stages)
            if len(stages) == 1:
                return stages[0].id
        except Exception:
            pass
        return None

    def _visual_nodes_slab(self):
        """Full-time cached NodeSlab (float16 values) for this
        diagram component and resolved stage, or None when no visual
        store is stamped or the component is not cached.

        Returning None makes the caller fall back to the per-step
        results.nodes.get(...) read path, so headless tests and
        pre-bind diagrams keep working byte-identically.
        """
        store = getattr(self, "_visual_store", None)
        if store is None:
            return None
        scoped = self._scoped_results()
        if scoped is None:
            return None
        stage_id = self._visual_resolved_stage_id()
        if stage_id is None:
            return None
        component = self.spec.selector.component
        if not component:
            return None
        try:
            return store.nodes_slab(scoped, stage_id, component)
        except Exception:
            return None

    def _visual_gauss_slab(self):
        """Full-time cached GaussSlab (float16 values), or None."""
        store = getattr(self, "_visual_store", None)
        if store is None:
            return None
        scoped = self._scoped_results()
        if scoped is None:
            return None
        stage_id = self._visual_resolved_stage_id()
        if stage_id is None:
            return None
        component = self.spec.selector.component
        if not component:
            return None
        try:
            return store.gauss_slab(scoped, stage_id, component)
        except Exception:
            return None

    def _visual_color_limits(self):
        """Global (vmin, vmax) for this component across the whole time
        history, from the visual store, or None.

        Consumed by the contour's clim path to give a STABLE colour scale:
        the per-step auto range is computed at attach against step 0, which
        is often the undeformed/zero state and yields a degenerate (0, 1)
        scale for the whole animation. The store tracks the finite min/max
        over all steps during its single load pass (no rescan), so this
        anchors the scale to the real demand range. None when no store is
        stamped (headless / pre-bind) or the component is not cached, in
        which case the caller falls back to the per-step range.
        """
        store = getattr(self, "_visual_store", None)
        if store is None:
            return None
        stage_id = self._visual_resolved_stage_id()
        if stage_id is None:
            return None
        component = self.spec.selector.component
        if not component:
            return None
        try:
            return store.color_limits(stage_id, component)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Subclass hooks (override these)
    # ------------------------------------------------------------------

    def attach(
        self,
        backend: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        """Build initial actors. Subclass override.

        The default implementation stores ``backend`` / ``view`` /
        ``scene`` and resolves the selector to concrete IDs.
        Subclasses should ``super().attach(backend, view, scene)``
        first, then build their actors.

        ``view`` is the :class:`ViewerData` structural snapshot;
        a raw FEMData still works at runtime via duck typing on
        the read accessors the diagrams exercise (kept for the
        transition while tests / external callers haven't switched).
        ``scene`` is the substrate FEMSceneData built once at viewer
        open. Diagrams that paint on the substrate (Contour,
        LineForce, …) require it; stub diagrams may pass ``None``.
        """
        # Render seam (ADR 0042). ``backend`` is always a RenderBackend:
        # production binds through ``DiagramRegistry.bind`` (wraps the raw
        # pyvista plotter into a PyVistaQtBackend once), and diagram tests
        # pass an offscreen PyVistaQtBackend (the shared ``headless_plotter``
        # fixture). Diagrams emit SceneLayers via ``self._backend`` and never
        # import pyvista (INV-2 / test_diagrams_pure_no_pyvista). The seam is
        # now airtight — there is no raw-plotter escape hatch. ``self._plotter``
        # is the raw plotter behind the backend, kept only for the legacy
        # ``self._actors`` teardown path in ``detach`` (migrated diagrams hold
        # handles, not actors, so they never populate it).
        self._backend = backend
        self._plotter = getattr(backend, "plotter", backend)
        self._view = view
        self._scene = scene
        # Subclasses decide which axis matters; resolving both is
        # cheap and avoids subclass boilerplate.
        try:
            self._resolved_node_ids = self.selector.resolve_node_ids(view)
        except Exception:
            self._resolved_node_ids = None
        try:
            self._resolved_element_ids = self.selector.resolve_element_ids(view)
        except Exception:
            self._resolved_element_ids = None
        self._attached = True

    def update_to_step(self, step_index: int) -> None:
        """Refresh actors for ``step_index``. Subclass override."""
        raise NotImplementedError(
            f"{type(self).__name__}.update_to_step must be overridden."
        )

    def detach(self) -> None:
        """Remove actors. Subclass override.

        The default implementation removes every actor in
        ``self._actors`` from the plotter and clears state.
        Subclasses should ``super().detach()`` last.
        """
        if self._plotter is not None:
            for actor in self._actors:
                try:
                    self._plotter.remove_actor(actor)
                except Exception:
                    pass
        self._actors = []
        self._plotter = None
        self._backend = None
        self._view = None
        self._scene = None
        self._resolved_node_ids = None
        self._resolved_element_ids = None
        self._attached = False

    def settings_widget(self) -> Any:
        """Return a Qt widget for the per-diagram settings tab.

        Default: ``None`` — no per-diagram settings. Subclasses that
        expose styling controls return a ``QWidget``.
        """
        return None

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Re-position the layer's geometry against the deformed
        substrate.

        Default: no-op — only correct for diagrams with no point
        geometry of their own (e.g. section cuts). Every rendering
        diagram must override: post-ADR-0042 the backend dataset is a
        COPY (extracted submeshes included — they do NOT share points
        with ``scene.grid``), so a diagram that skips this hook stays
        pinned at the reference configuration while the substrate
        warps. Substrate-extracted layers re-sample via their cached
        ``vtkOriginalPointIds`` rows; owned-geometry layers (gauss
        markers, glyph anchors, fiber clouds) recompute their points.

        ``deformed_pts`` is ``(num_substrate_points, 3)`` row-aligned
        with ``scene.node_ids`` (and ``fem.nodes.ids``). ``None``
        means "reset to the reference / undeformed state" (the pump
        resets ``scene.grid.points`` before fanning out, so sampling
        the scene grid is equivalent).
        """
        return None

    def make_side_panel(self, director: Any) -> Any:
        """Construct a dockable side-panel widget for this diagram.

        Default: ``None`` — no side panel. Diagrams with auxiliary
        2-D plots (fiber section scatter, layer through-thickness
        profile) override this to return a panel object whose
        ``.widget`` attribute is a ``QWidget``. The viewer docks the
        widget when the diagram is added and removes it on detach.
        """
        return None

    # ------------------------------------------------------------------
    # Visibility (default impl — subclasses can override for actor-
    # specific behaviour)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        """Show / hide all actors without detaching."""
        self._visible = visible
        for actor in self._actors:
            try:
                actor.SetVisibility(bool(visible))
            except Exception:
                pass

    def apply_effective_visibility(self, effective: bool) -> None:
        """Apply gate-computed visibility without touching user intent.

        The composition gate computes ``effective = is_visible AND
        in_active_composition`` and pushes it onto the rendered
        artifacts. Routing through :meth:`set_visible` reuses each
        subclass's artifact path (backend layer handles for migrated
        diagrams, raw actors otherwise); restoring ``_visible``
        afterwards keeps ``is_visible`` as the user-intent flag so the
        next gate run recomputes from unchanged inputs.
        """
        saved = self._visible
        try:
            self.set_visible(bool(effective))
        finally:
            self._visible = saved

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        attached = "attached" if self._attached else "detached"
        return f"<{type(self).__name__} kind={self.kind} {attached}>"
