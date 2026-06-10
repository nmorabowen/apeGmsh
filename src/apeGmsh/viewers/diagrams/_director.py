"""ResultsDirector — single source of truth for stage / step / registry.

The Director owns:

* ``results`` — the bound Results object.
* ``stage_id`` — currently active stage (None until set).
* ``step_index`` — currently active time-step index within that stage.
* ``time_mode`` — ``"single" | "range" | "envelope" | "animation"``
  (Phase 0 ships ``"single"`` only; the others arrive in Phase 6).
* ``registry`` — the DiagramRegistry.

UI tabs subscribe to Director observers (``on_step_changed``,
``on_stage_changed``, ``on_diagrams_changed``); user actions flow back
through Director methods.

Observer chain is **UI -> Director -> Diagrams**. A diagram never
calls ``director.set_step(...)``; that would create a feedback loop.

Phase 8 (ADR 0020) — :meth:`set_model` is the chain-forward binder
that accepts an :class:`OpenSeesModel` handle (cuts iteration source).
The director keeps a ``_model_h5`` file path internally for the
:class:`FemToOpsTagMap` build (still path-based by current contract),
bound through the private :meth:`_bind_model_h5` helper at viewer
boot / session restore.  AST guard (ADR 0014 INV-2): the
``OpenSeesModel`` type is referenced only under ``TYPE_CHECKING`` with
a string annotation, never imported at module level.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import numpy as np
from numpy import ndarray

from ._compositions import CompositionManager
from ._geometries import GeometryManager
from ._registry import DiagramRegistry


# Synthetic stage id surfaced when 2+ real stages exist. Selecting it
# walks the concatenated time vector across every real stage; the
# director silently swaps the underlying Results stage as the user
# scrubs across boundaries.
COMBINED_STAGE_ID = "__all__"
COMBINED_STAGE_NAME = "All stages"

if TYPE_CHECKING:
    from apeGmsh.cuts import FemToOpsTagMap, SectionCutDef, SectionSweepDef
    from apeGmsh.results.Results import Results
    from apeGmsh.results.readers._protocol import StageInfo
    from apeGmsh.viewers.data import ViewerData
    from ._base import Diagram
    from ..scene.fem_scene import FEMSceneData
    # AST guard (ADR 0014 INV-2) — :class:`OpenSeesModel` is referenced
    # only as a string in annotations. The module is NEVER imported
    # at all from ``viewers/`` (not even under ``TYPE_CHECKING``); the
    # ``set_model`` parameter is duck-typed. Annotations use the
    # qualified string ``"apeGmsh.opensees.opensees_model.OpenSeesModel"``
    # so static checkers can still resolve the type without breaking
    # the AST scan at ``tests/test_viewers_pure_h5_consumer.py``.


class TimeMode(str, Enum):
    """How the Director interprets ``step_index`` for diagram updates."""
    SINGLE = "single"
    RANGE = "range"
    ENVELOPE = "envelope"
    ANIMATION = "animation"


class ResultsDirector:
    """Composite owning stage / step / time-mode / registry.

    Phase 0 surface: stage selection, single-step navigation, registry
    routing. Range / envelope / animation arrive in Phase 6.

    Observers fire with explicit arguments so the UI doesn't have to
    poll the Director for the new value:

    * ``on_step_changed(step_index: int)``
    * ``on_stage_changed(stage_id: str)``
    * ``on_diagrams_changed()`` — re-fired by the registry; subscribed
      here to keep UI subscriptions on the Director surface.
    """

    def __init__(self, results: "Results") -> None:
        self._results = results
        self._stage_id: Optional[str] = None
        self._step_index: int = 0
        self._time_mode: TimeMode = TimeMode.SINGLE

        # Currently picked beam / shell integration point — feeds the
        # fiber section and layer through-thickness side panels. Tuple
        # ``(element_id, gp_index)`` or ``None`` if nothing picked.
        self._picked_gp: Optional[tuple[int, int]] = None

        self._registry = DiagramRegistry()
        self._registry.subscribe(self._fire_diagrams_changed)

        # Geometry manager — bootstraps one "Geometry 1" that owns its
        # own (initially empty) CompositionManager. Each geometry holds
        # the deformation state for its child compositions.
        self._geometries = GeometryManager()

        # Section-cut tag-map state. Populated lazily on first
        # ``tag_map`` access; the path is derived on demand via
        # :func:`apeGmsh.viewers.data._h5_probe.resolve_orientation_source`
        # against the bound :class:`Results`.  ADR 0026 PR-stretch
        # closed the previous ``_model_h5: Path`` field and the
        # ``_bind_model_h5(path)`` private setter — the bound Results
        # is now the single source of truth for the tag-map path.
        self._tag_map_cache: "Optional[FemToOpsTagMap]" = None

        # Chain-forward handle.  The OpenSeesModel (when supplied via
        # :meth:`set_model` or :meth:`bind_results`) is the source for
        # :meth:`load_cuts_from_h5` iteration.
        # AST guard: duck-typed storage, no runtime import.
        self._opensees_model: "Optional[Any]" = None

        # Combined-stage state. ``_combined_active`` mirrors the
        # public ``stage_id == COMBINED_STAGE_ID`` view; ``_real_stages``
        # is the snapshot of real stages at activation time;
        # ``_combined_boundaries`` is the cumulative-step prefix array
        # used to translate global step → (real_stage_id, local_step).
        self._combined_active: bool = False
        self._real_stages: "list[StageInfo]" = []
        self._combined_boundaries: ndarray = np.array([], dtype=np.int64)
        self._combined_time: ndarray = np.array([], dtype=np.float64)

        self.on_step_changed: list[Callable[[int], None]] = []
        self.on_stage_changed: list[Callable[[str], None]] = []
        self.on_diagrams_changed: list[Callable[[], None]] = []
        self.on_picked_gp_changed: list[
            Callable[[Optional[tuple[int, int]]], None]
        ] = []

        self._render_callback: Optional[Callable[[], None]] = None

        # ADR 0056 Part 3 — the dispatcher ALWAYS exists. Constructed
        # here with no-op pumps; ``ResultsViewer.show()`` rebinds the
        # real four pipeline pumps (STEP / DEFORM / GATE / RENDER) via
        # ``dispatcher.bind(...)`` once the plotter/scene exist. UI
        # call sites and owner mutators fire events via
        # ``director.dispatcher.fire(...)`` so a single matrix decides
        # what primitives run — never guard with
        # ``getattr(director, "dispatcher", None)``.
        from ._dispatch import Dispatcher
        self.dispatcher: "Dispatcher" = Dispatcher(self)
        # Owner-fired events (ADR 0056 Part 2): the registry fires
        # LAYER_VISIBILITY_CHANGED itself on visibility mutations.
        self._registry.dispatcher = self.dispatcher

        # Pick a default stage if there is exactly one (matches
        # Results._resolve_stage's "auto" behaviour). Park the time
        # cursor at the last step of that stage so freshly-attached
        # diagrams paint at the end of history (final state) instead
        # of the near-zero first increment.
        stages = self._all_stages()
        if len(stages) == 1:
            self._stage_id = stages[0].id
            try:
                n = int(self._scoped_results().n_steps)
            except Exception:
                n = 0
            if n > 0:
                self._step_index = n - 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def results(self) -> "Results":
        return self._results

    @property
    def view(self) -> "Optional[ViewerData]":
        """Cached :class:`ViewerData` wrap of the bound results' FEMData.

        Phase 8.7 commit 5 — the canonical viewer-facing structural
        snapshot.  Cached lazily; callers should treat it as immutable
        for the lifetime of the director.  Returns ``None`` if no
        FEMData is bound.
        """
        from apeGmsh.viewers.data import ViewerData
        cached: "Optional[ViewerData]" = getattr(self, "_view_cache", None)
        if cached is not None:
            return cached
        fem = self._results.fem
        if fem is None:
            return None
        new_view = ViewerData.from_fem(fem)
        self._view_cache: "Optional[ViewerData]" = new_view
        return new_view

    @property
    def registry(self) -> DiagramRegistry:
        return self._registry

    @property
    def geometries(self) -> GeometryManager:
        return self._geometries

    @property
    def compositions(self) -> Optional[CompositionManager]:
        """The active geometry's CompositionManager (or None).

        Back-compat property — most call sites still want "the
        compositions the user is currently editing", which after the
        Geometry refactor lives on the active Geometry. Subscribers
        that need to track *every* state change (geometry list,
        active geometry, composition list, rename) should use
        :attr:`geometries` and ``geometries.subscribe`` instead — that
        observer fires for any model change while this property's
        value can switch out from under them on geometry change.
        """
        active = self._geometries.active
        return active.compositions if active is not None else None

    # ------------------------------------------------------------------
    # Section-cut tag map (cuts subpackage integration)
    # ------------------------------------------------------------------

    def bind_results(self, results: "Any") -> None:
        """Bind a :class:`Results` handle — the canonical single-call
        binder (ADR 0026 PR7-d, PR-stretch made it the only path).

        Pulls both binding sources off the Results in one step:

        * ``results.model`` — the chain-forward OpenSeesModel handle
          (cuts iteration source; duck-typed per ADR 0014 INV-2).
        * ``results._path`` — the file path the
          :class:`FemToOpsTagMap` builds against, derived on demand
          inside :attr:`tag_map` via
          :func:`resolve_orientation_source`.  No separate
          ``_model_h5`` field is kept on the director.

        Cuts auto-load (``_apply_pending_cuts`` in ResultsViewer)
        and session restore both flow through this entry point.

        Idempotent: re-binding the same Results is a no-op on the
        chain-forward handle; the tag_map cache is invalidated so
        the next access re-derives against the freshly-bound
        Results.
        """
        # Replace the bound Results so resolve_orientation_source picks
        # it up next access.  The director's __init__ stored an initial
        # ``self._results``; bind_results swaps it (typically the same
        # object on re-bind, but tests can swap to alternates).
        self._results = results
        self._opensees_model = getattr(results, "model", None)
        # Invalidate the tag_map cache so the next access re-derives
        # from the new Results' _path via resolve_orientation_source.
        self._tag_map_cache = None

    def set_model(
        self, opensees_model: "Optional[Any]",
    ) -> None:
        """Bind an :class:`OpenSeesModel` for cuts iteration.

        The chain-forward handle (duck-typed per ADR 0014 INV-2; no
        runtime import of ``apeGmsh.opensees.opensees_model``) feeds
        :meth:`load_cuts_from_h5`, which iterates
        ``opensees_model.cuts()`` / ``opensees_model.sweeps()`` instead
        of re-walking the file.

        ADR 0026 — prefer :meth:`bind_results` for the canonical
        single-call binding from a Results handle.  ``set_model``
        stays for external callers that already hold an
        OpenSeesModel directly (without a Results wrapper) — it does
        NOT touch the tag-map state because the tag-map path lives
        on the bound :class:`Results`.

        Pass ``None`` to clear the bound handle.
        """
        self._opensees_model = opensees_model

    @property
    def opensees_model(self) -> "Optional[Any]":
        """The bound :class:`OpenSeesModel` handle, or ``None``.

        Annotation is ``Any`` to keep the AST scan at
        ``tests/test_viewers_pure_h5_consumer.py`` happy (the
        ``OpenSeesModel`` class is NEVER imported into ``viewers/``,
        not even under ``TYPE_CHECKING``; callers use duck-typed
        access to ``om.cuts()`` / ``om.sweeps()`` — see ADR 0014
        INV-2).
        """
        return self._opensees_model

    @property
    def tag_map(self) -> "Optional[FemToOpsTagMap]":
        """Lazy-built :class:`FemToOpsTagMap` for the bound Results.

        Returns ``None`` when the bound :class:`Results` has no path
        carrying an ``/opensees/`` orientation zone (in-memory
        Results, recorder flavours, or bare neutral-only files).
        The first access opens the file via
        :meth:`FemToOpsTagMap.from_h5`; subsequent accesses reuse
        the cached map.  Propagates any reader exception so callers
        see the underlying error (missing file, schema mismatch).

        ADR 0026 PR-stretch — the path is derived on demand via
        :func:`resolve_orientation_source` instead of being stored
        in a separate ``_model_h5`` field.  The bound Results is the
        single source of truth.
        """
        from ..data._h5_probe import resolve_orientation_source
        source = resolve_orientation_source(self._results)
        if source is None:
            return None
        if self._tag_map_cache is None:
            from apeGmsh.cuts import FemToOpsTagMap
            self._tag_map_cache = FemToOpsTagMap.from_h5(source)
        return self._tag_map_cache

    def add_section_cut(
        self,
        cut: "SectionCutDef",
        *,
        geometry_id: Optional[str] = None,
        composition_id: Optional[str] = None,
        composition_name: str = "Section cuts",
        label: Optional[str] = None,
        style: "Optional[Any]" = None,
        model_h5: "Optional[str | Path]" = None,
    ) -> "Diagram":
        """Wire a :class:`SectionCutDef` into the registry as a Layer.

        Defaults: routes to the active geometry's first matching
        composition (by ``composition_name``); creates that composition
        if absent. Pass ``geometry_id`` / ``composition_id`` to target
        an existing pair explicitly.

        Parameters
        ----------
        cut
            The cut definition to render.
        geometry_id
            Target geometry. ``None`` → the active geometry.
        composition_id
            Target composition. ``None`` → reuse / create
            ``composition_name`` on the geometry.
        composition_name
            Auto-created composition name when neither
            ``composition_id`` nor an existing match is found. Default
            ``"Section cuts"``.
        label
            Override the Layer's user-facing label. Defaults to
            ``cut.label`` (or ``"section cut"`` when the def has none).
        style
            Override the :class:`SectionCutStyle`. Defaults to one
            wrapping ``cut`` with the standard colors.
        model_h5
            Convenience — bind a ``model.h5`` path for the
            :class:`FemToOpsTagMap`.  Set once for the first cut, then
            omit for subsequent adds against the same run.

        Returns
        -------
        Diagram
            The attached :class:`SectionCutDiagram` instance.
        """
        # ADR 0026 PR-stretch: ``model_h5=`` no longer mutates director
        # state.  When supplied it builds a transient :class:`FemToOpsTagMap`
        # for this cut only; otherwise we fall back to ``self.tag_map``
        # (which derives the path from the bound :class:`Results`).
        if model_h5 is not None:
            from apeGmsh.cuts import FemToOpsTagMap
            tag_map = FemToOpsTagMap.from_h5(model_h5)
        else:
            tag_map = self.tag_map
        if tag_map is None:
            raise RuntimeError(
                "add_section_cut: no model.h5 source available. Bind "
                "the director to a Results carrying an /opensees/ "
                "orientation zone via director.bind_results(results), "
                "or pass model_h5= to this call for a one-shot tag map."
            )

        from ._base import DiagramSpec
        from ._section_cut import SectionCutDiagram
        from ._selectors import SlabSelector
        from ._styles import SectionCutStyle

        if style is None:
            style = SectionCutStyle(cut=cut)
        elif not isinstance(style, SectionCutStyle):
            raise TypeError(
                "style must be a SectionCutStyle (or None); "
                f"got {type(style).__name__}."
            )
        elif style.cut is not cut:
            # Caller passed a custom style without wiring the cut in —
            # rebuild so the rendered cut matches the parameter.
            style = SectionCutStyle(
                cut=cut,
                kept_color=style.kept_color,
                discarded_color=style.discarded_color,
                quad_opacity=style.quad_opacity,
                edge_color=style.edge_color,
                show_edges=style.show_edges,
                show_normal_arrow=style.show_normal_arrow,
                normal_arrow_fraction=style.normal_arrow_fraction,
                highlight_color=style.highlight_color,
                highlight_opacity=style.highlight_opacity,
                show_filter_initially=style.show_filter_initially,
            )

        layer_label = label if label is not None else (cut.label or "section cut")
        component = layer_label or "section_cut"
        spec = DiagramSpec(
            kind="section_cut",
            selector=SlabSelector(component=component),
            style=style,
            stage_id=self._stage_id,
            visible=True,
            label=layer_label,
        )
        diagram = SectionCutDiagram(spec, self._results, tag_map=tag_map)

        target_geom = self._resolve_target_geometry(geometry_id)
        target_comp = self._resolve_target_composition(
            target_geom, composition_id, composition_name,
        )
        self._registry.add(diagram)
        target_geom.compositions.add_layer(target_comp.id, diagram)
        return diagram

    def load_cuts_from_h5(self) -> "list[Diagram]":
        """Read ``/opensees/cuts/`` and ``/opensees/sweeps/`` and attach
        each as a Diagram (Layer).

        v4 of the apeGmsh.cuts roadmap — cuts persisted in ``model.h5``
        flow back into the viewer through this hook. Returns the list
        of attached diagrams (standalone cuts first in writer order,
        followed by each sweep's fan-out flattened in sweep order).

        The cuts source is the bound :class:`OpenSeesModel` handle
        when available (:meth:`set_model` or :meth:`bind_results`);
        otherwise falls back to reading the bound Results' file via
        :func:`read_cuts_and_sweeps`.  The :class:`FemToOpsTagMap`
        is built lazily inside :attr:`tag_map` from the same source.

        ADR 0026 PR-stretch — the fallback path was previously gated
        on ``self._model_h5``; it is now derived on demand via
        :func:`resolve_orientation_source` against the bound Results.

        Pre-v4 files (no ``/opensees/cuts/``) produce an empty result.
        """
        if self._opensees_model is not None:
            # Chain-forward path — cuts iteration via the handle.
            cuts = tuple(self._opensees_model.cuts())
            sweeps = tuple(self._opensees_model.sweeps())
        else:
            # File-fallback path — read cuts directly off the bound
            # Results' file.  Note: cuts persistence is independent
            # of the orientation zone (a file can carry
            # /opensees/cuts/ without /opensees/transforms/), so we
            # use the raw ``_path`` here rather than
            # :func:`resolve_orientation_source`.
            path = getattr(self._results, "_path", None) if self._results else None
            if path is None:
                raise RuntimeError(
                    "load_cuts_from_h5: no cuts source bound. Call "
                    "director.bind_results(results) (or set_model) "
                    "with a Results pointing at a file with "
                    "/opensees/cuts/ first."
                )
            from apeGmsh.cuts import read_cuts_and_sweeps
            cuts, sweeps = read_cuts_and_sweeps(path)
        attached: list = []
        for cut in cuts:
            attached.append(self.add_section_cut(cut))
        for sweep in sweeps:
            attached.extend(self.add_section_cut_sweep(sweep))
        return attached

    def add_section_cut_sweep(
        self,
        sweep: "SectionSweepDef",
        *,
        geometry_id: Optional[str] = None,
        composition_id: Optional[str] = None,
        composition_name: str = "Section cuts",
        label_prefix: Optional[str] = None,
        style: "Optional[Any]" = None,
        model_h5: "Optional[str | Path]" = None,
    ) -> "list[Diagram]":
        """Fan a :class:`SectionSweepDef` into one Layer per cut.

        All cuts land in the same Composition so the user can toggle
        the sweep as a unit (per-Layer checkboxes) or select an
        individual cut for inspection. Labels default to each cut's
        own label; pass ``label_prefix`` to override (becomes
        ``f"{prefix}[{i}]"``).
        """
        # ADR 0026 PR-stretch: forward ``model_h5=`` to every per-cut
        # add_section_cut call instead of mutating director state.
        # Each cut builds its own transient tag map; cheap because
        # FemToOpsTagMap.from_h5 only walks /opensees/element_meta.
        diagrams: list = []
        for i, cut in enumerate(sweep):
            cut_label = (
                f"{label_prefix}[{i}]" if label_prefix is not None
                else None
            )
            diagrams.append(self.add_section_cut(
                cut,
                geometry_id=geometry_id,
                composition_id=composition_id,
                composition_name=composition_name,
                label=cut_label,
                style=style,
                model_h5=model_h5,
            ))
        return diagrams

    def _resolve_target_geometry(
        self, geometry_id: Optional[str],
    ):
        if geometry_id is not None:
            target = self._geometries.find(geometry_id)
            if target is None:
                raise KeyError(
                    f"Geometry id {geometry_id!r} not found. "
                    f"Available: {[g.id for g in self._geometries.geometries]}"
                )
            return target
        target = self._geometries.active
        if target is None:
            # Bootstrap guarantees at least one geometry, but the
            # active pointer may be None if the user explicitly
            # cleared it — fall back to the first.
            geoms = self._geometries.geometries
            if not geoms:
                raise RuntimeError(
                    "No geometries available — GeometryManager bootstrap "
                    "should have created one."
                )
            return geoms[0]
        return target

    def _resolve_target_composition(
        self,
        geom,
        composition_id: Optional[str],
        composition_name: str,
    ):
        comps = geom.compositions
        if composition_id is not None:
            target = comps.find(composition_id)
            if target is None:
                raise KeyError(
                    f"Composition id {composition_id!r} not found "
                    f"on geometry {geom.id!r}."
                )
            return target
        for c in comps.compositions:
            if c.name == composition_name:
                return c
        return comps.add(composition_name)

    @property
    def stage_id(self) -> Optional[str]:
        return self._stage_id

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def time_mode(self) -> TimeMode:
        return self._time_mode

    @property
    def n_steps(self) -> int:
        if self._combined_active:
            return int(self._combined_time.size)
        if self._stage_id is None:
            return 0
        try:
            return int(self._scoped_results().n_steps)
        except Exception:
            return 0

    @property
    def time_vector(self) -> ndarray:
        if self._combined_active:
            return self._combined_time
        if self._stage_id is None:
            return np.array([], dtype=np.float64)
        try:
            return np.asarray(self._scoped_results().time, dtype=np.float64)
        except Exception:
            return np.array([], dtype=np.float64)

    def stages(self) -> "list[StageInfo]":
        """Real stages, plus a synthetic combined entry when ≥ 2 exist."""
        from apeGmsh.results.readers._protocol import StageInfo
        real = self._all_stages()
        if len(real) <= 1:
            return real
        total_steps = int(sum(int(s.n_steps or 0) for s in real))
        combined = StageInfo(
            id=COMBINED_STAGE_ID,
            name=COMBINED_STAGE_NAME,
            kind="combined",
            n_steps=total_steps,
        )
        return real + [combined]

    @property
    def combined_active(self) -> bool:
        """True when the user has the synthetic combined stage selected."""
        return self._combined_active

    def current_time(self) -> Optional[float]:
        """Time value at the current step (combined- or stage-local).

        In combined mode this is the offset-shifted concatenated time;
        in single-stage mode it's the stage's own time vector entry.
        """
        tv = self.time_vector
        if tv.size == 0:
            return None
        idx = max(0, min(self._step_index, tv.size - 1))
        return float(tv[idx])

    # ------------------------------------------------------------------
    # Plotter binding (registry forward)
    # ------------------------------------------------------------------

    def bind_plotter(
        self,
        plotter: Any,
        *,
        scene: "FEMSceneData | None" = None,
        render_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Bind the Director (and its registry) to a plotter.

        Parameters
        ----------
        plotter
            The PyVista plotter (Qt-backed for the real viewer; an
            offscreen plotter or stub for tests).
        scene
            The substrate ``FEMSceneData`` built once at viewer open.
            Diagrams that paint on the substrate (Contour, Deformed,
            …) require it. ``None`` is acceptable for tests with stub
            diagrams that ignore the scene.
        render_callback
            Invoked once per logical step / stage / diagram change so
            all UI / overlay state coalesces into one ``plotter.render()``
            per event. If ``None``, the registry is bound but no
            auto-render fires (test mode).
        """
        view = self.view
        if view is None:
            raise RuntimeError(
                "Cannot bind a ResultsDirector without a bound FEMData. "
                "Construct Results with fem= or call results.bind(fem)."
            )
        self._registry.bind(plotter, view, scene)
        self._render_callback = render_callback

    def unbind_plotter(self) -> None:
        self._registry.unbind()
        self._render_callback = None

    # ------------------------------------------------------------------
    # Stage / step actions
    # ------------------------------------------------------------------

    def set_stage(self, stage_id_or_name: str) -> None:
        """Switch the active stage. Re-attaches every diagram against
        the new scoped Results.

        Picking the synthetic ``COMBINED_STAGE_ID`` activates combined
        mode: the time scrubber walks the concatenated time vector
        across every real stage; the director silently swaps the
        underlying Results stage on every boundary crossing.
        """
        # Combined-stage entry — never matches a real StageInfo.
        if stage_id_or_name == COMBINED_STAGE_ID:
            self._activate_combined_mode()
            return

        info = self._lookup_stage(stage_id_or_name)
        leaving_combined = self._combined_active
        if (
            info.id == self._stage_id
            and not leaving_combined
        ):
            return
        self._combined_active = False
        self._real_stages = []
        self._combined_boundaries = np.array([], dtype=np.int64)
        self._combined_time = np.array([], dtype=np.float64)
        self._stage_id = info.id
        # Mirror to the unscoped Results so layer reads with no
        # explicit stage_id route to this stage.
        self._set_results_default_stage(info.id)
        # Land on the last step of the new stage by default — see the
        # constructor's note for why the end of history is the
        # better starting point than step 0.
        try:
            n = int(self._scoped_results().n_steps)
        except Exception:
            n = 0
        self._step_index = max(0, n - 1)
        self._registry.reattach_all()
        self._fire_stage_changed(info.id)
        self._registry.update_to_step(self._step_index)
        self._render()

    def _activate_combined_mode(self) -> None:
        """Enter combined mode using the current real-stage list."""
        real = list(self._all_stages())
        if len(real) <= 1:
            # Nothing to combine; treat as a no-op.
            return
        # Cumulative step counts: boundaries[i] = sum(n_steps[0..i)).
        # Length is len(real)+1 so binary search ``searchsorted`` lands
        # on the correct stage for any global step in [0, total).
        counts = np.asarray(
            [int(s.n_steps or 0) for s in real], dtype=np.int64,
        )
        boundaries = np.concatenate([[0], np.cumsum(counts)])
        # Concatenated time vector with monotone offsets so the
        # scrubber x-axis stays single-valued. Offset for stage i is
        # max(time[i-1]) + small epsilon to keep values strictly
        # increasing across boundaries.
        time_chunks: list[ndarray] = []
        offset = 0.0
        for s in real:
            try:
                tv = np.asarray(
                    self._results.stage(s.id).time, dtype=np.float64,
                )
            except Exception:
                tv = np.zeros(int(s.n_steps or 0), dtype=np.float64)
            if tv.size == 0:
                continue
            shifted = tv + offset
            time_chunks.append(shifted)
            # Bump offset past the last value so the next stage's
            # times don't overlap. Add a small epsilon (last delta or
            # 1.0) to keep monotonicity strict.
            last = float(shifted[-1]) if shifted.size else offset
            tail_step = (
                float(tv[-1] - tv[0]) / max(1, tv.size - 1)
                if tv.size > 1 else 1.0
            )
            offset = last + max(tail_step, 1e-9)
        combined_time = (
            np.concatenate(time_chunks) if time_chunks
            else np.array([], dtype=np.float64)
        )

        self._combined_active = True
        self._real_stages = real
        self._combined_boundaries = boundaries
        self._combined_time = combined_time
        # Land on the last global step in combined mode so the user
        # sees end-of-history (final stage's final increment) by default.
        last_global = int(combined_time.size) - 1 if combined_time.size else 0
        self._step_index = max(0, last_global)
        # Bind to the real stage that owns the last global step.
        last_stage_idx = max(0, len(real) - 1)
        first_id = real[last_stage_idx].id
        self._stage_id = first_id
        self._set_results_default_stage(first_id)
        self._registry.reattach_all()
        self._fire_stage_changed(COMBINED_STAGE_ID)
        self._registry.update_to_step(self._step_index)
        self._render()

    def _combined_translate(self, global_step: int) -> "tuple[str, int]":
        """Map a combined-mode global step to ``(real_stage_id, local_step)``."""
        if not self._real_stages:
            return (self._stage_id or "", int(global_step))
        # Find the stage whose half-open interval contains ``global_step``.
        # boundaries[i] is the cumulative count BEFORE stage i; we want
        # the largest i with boundaries[i] <= global_step.
        stage_idx = int(
            np.searchsorted(
                self._combined_boundaries, int(global_step), side="right",
            ) - 1
        )
        stage_idx = max(0, min(stage_idx, len(self._real_stages) - 1))
        local = int(global_step) - int(
            self._combined_boundaries[stage_idx]
        )
        return (self._real_stages[stage_idx].id, local)

    def set_step(self, step_index: int) -> None:
        """Move the active step. Coalesces into one render.

        In combined mode the global step is translated to
        ``(real_stage_id, local_step)`` — when crossing a boundary
        the underlying real stage is silently swapped (no
        ``on_stage_changed`` fires; the user remains "on" the
        combined view).
        """
        n = self.n_steps
        if n == 0:
            return
        clamped = max(0, min(int(step_index), n - 1))
        if clamped == self._step_index:
            return
        self._step_index = clamped

        if self._combined_active:
            real_id, local = self._combined_translate(clamped)
            if real_id != self._stage_id:
                self._stage_id = real_id
                self._set_results_default_stage(real_id)
                # Diagrams cached step-0 data against the previous
                # stage; rebuild against the new stage.
                self._registry.reattach_all()
                self._registry.update_to_step(local)
                # Notify subscribers — combined mode previously hid
                # the boundary cross from observers, leaving stale
                # stage metadata in any UI that reflects it.
                self._fire_stage_changed(real_id)
            else:
                self._registry.update_to_step(local)
        else:
            self._registry.update_to_step(clamped)

        self._fire_step_changed(clamped)
        self._render()

    def _set_results_default_stage(self, stage_id: Optional[str]) -> None:
        """Mirror the active stage onto the unscoped Results so layer
        reads with no pinned ``spec.stage_id`` route correctly.

        Layers without an explicit stage typically aren't useful in
        a multi-stage file (the read raises on ambiguity); combined
        mode flips that — those layers become useful again because
        the director re-binds the default per step.
        """
        try:
            self._results._stage_id = stage_id  # noqa: SLF001
        except Exception:
            pass

    def read_at_pick(
        self,
        node_id: int,
        components: "Iterable[str]",
        *,
        step: Optional[int] = None,
    ) -> dict[str, float]:
        """Read scalar values for a single node at a single step.

        Used by the Inspector tab to populate the picked-entity panel.
        Returns ``{component: value}``; missing components are silently
        skipped.
        """
        if self._stage_id is None:
            return {}
        results = self._scoped_results()
        target_step = self._step_index if step is None else int(step)
        out: dict[str, float] = {}
        for component in components:
            try:
                slab = results.nodes.get(
                    ids=[int(node_id)],
                    component=component,
                    time=[target_step],
                )
            except Exception:
                continue
            if slab.values.size == 0:
                continue
            out[component] = float(np.asarray(slab.values).ravel()[0])
        return out

    def read_history(
        self,
        node_id: int,
        component: str,
    ) -> "Optional[tuple[ndarray, ndarray]]":
        """Read ``(time, values)`` for one node + one component over the stage.

        Used by ``TimeHistoryPanel``. Returns ``None`` if the read
        fails or the slab is empty.
        """
        if self._stage_id is None:
            return None
        results = self._scoped_results()
        try:
            slab = results.nodes.get(
                ids=[int(node_id)],
                component=component,
            )
        except Exception:
            return None
        if slab.values.size == 0:
            return None
        time = np.asarray(slab.time, dtype=np.float64)
        values = np.asarray(slab.values, dtype=np.float64).ravel()
        if time.size != values.size:
            # Defensive: shapes should match for a single-node read,
            # but trim to common length if not.
            n = min(time.size, values.size)
            time = time[:n]
            values = values[:n]
        return time, values

    def step_to_time(self, t: float) -> None:
        """Snap to the nearest step for time ``t``."""
        tv = self.time_vector
        if tv.size == 0:
            return
        idx = int(np.argmin(np.abs(tv - float(t))))
        self.set_step(idx)

    @property
    def picked_gp(self) -> Optional[tuple[int, int]]:
        """Currently picked ``(element_id, gp_index)`` or None."""
        return self._picked_gp

    def set_picked_gp(
        self, picked: Optional[tuple[int, int]],
    ) -> None:
        """Update the picked beam / shell GP. Fires observers."""
        if picked is None:
            new = None
        else:
            new = (int(picked[0]), int(picked[1]))
        if new == self._picked_gp:
            return
        self._picked_gp = new
        self._fire_picked_gp_changed(new)

    def subscribe_picked_gp(
        self,
        callback: Callable[[Optional[tuple[int, int]]], None],
    ) -> Callable[[], None]:
        """Register an observer for picked-GP changes."""
        self.on_picked_gp_changed.append(callback)
        def _unsub() -> None:
            if callback in self.on_picked_gp_changed:
                self.on_picked_gp_changed.remove(callback)
        return _unsub

    def set_time_mode(self, mode: TimeMode | str) -> None:
        """Switch the time mode. Phase 0 only honours ``SINGLE``;
        non-single modes raise ``NotImplementedError`` until Phase 6.
        """
        m = TimeMode(mode) if not isinstance(mode, TimeMode) else mode
        if m is not TimeMode.SINGLE:
            raise NotImplementedError(
                f"time_mode={m.value!r} arrives in Phase 6 (range / "
                f"envelope / animation). Phase 0 only supports 'single'."
            )
        self._time_mode = m

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scoped_results(self) -> "Results":
        if self._stage_id is None:
            raise RuntimeError("No stage set on ResultsDirector.")
        return self._results.stage(self._stage_id)

    def _all_stages(self) -> "list[StageInfo]":
        return list(self._results.stages)

    def _lookup_stage(self, name_or_id: str) -> "StageInfo":
        for s in self._all_stages():
            if s.id == name_or_id or s.name == name_or_id:
                return s
        raise KeyError(
            f"No stage matches {name_or_id!r}. "
            f"Available: {[s.name for s in self._all_stages()]}"
        )

    def _render(self) -> None:
        if self._render_callback is not None:
            try:
                self._render_callback()
            except Exception as exc:
                import sys
                print(
                    f"[ResultsDirector] render_callback raised: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Observer plumbing
    # ------------------------------------------------------------------

    def _fire_step_changed(self, step: int) -> None:
        for cb in list(self.on_step_changed):
            try:
                cb(step)
            except Exception as exc:
                import sys
                print(
                    f"[ResultsDirector] step observer raised: {exc}",
                    file=sys.stderr,
                )

    def _fire_stage_changed(self, stage_id: str) -> None:
        for cb in list(self.on_stage_changed):
            try:
                cb(stage_id)
            except Exception as exc:
                import sys
                print(
                    f"[ResultsDirector] stage observer raised: {exc}",
                    file=sys.stderr,
                )

    def _fire_diagrams_changed(self) -> None:
        for cb in list(self.on_diagrams_changed):
            try:
                cb()
            except Exception as exc:
                import sys
                print(
                    f"[ResultsDirector] diagrams observer raised: {exc}",
                    file=sys.stderr,
                )

    def _fire_picked_gp_changed(
        self, picked: Optional[tuple[int, int]],
    ) -> None:
        for cb in list(self.on_picked_gp_changed):
            try:
                cb(picked)
            except Exception as exc:
                import sys
                print(
                    f"[ResultsDirector] picked_gp observer raised: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Public subscribe helpers
    # ------------------------------------------------------------------

    def subscribe_step(
        self, callback: Callable[[int], None]
    ) -> Callable[[], None]:
        self.on_step_changed.append(callback)
        return lambda: self.on_step_changed.remove(callback) \
            if callback in self.on_step_changed else None  # type: ignore[func-returns-value]

    def subscribe_stage(
        self, callback: Callable[[str], None]
    ) -> Callable[[], None]:
        self.on_stage_changed.append(callback)
        return lambda: self.on_stage_changed.remove(callback) \
            if callback in self.on_stage_changed else None  # type: ignore[func-returns-value]

    def subscribe_diagrams(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
        self.on_diagrams_changed.append(callback)
        return lambda: self.on_diagrams_changed.remove(callback) \
            if callback in self.on_diagrams_changed else None  # type: ignore[func-returns-value]
