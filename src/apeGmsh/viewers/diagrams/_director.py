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
"""
from __future__ import annotations

from enum import Enum
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
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from apeGmsh.results.readers._protocol import StageInfo
    from ..scene.fem_scene import FEMSceneData


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
    def fem(self) -> "Optional[FEMData]":
        return self._results.fem

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
        fem = self._results.fem
        if fem is None:
            raise RuntimeError(
                "Cannot bind a ResultsDirector without a bound FEMData. "
                "Construct Results with fem= or call results.bind(fem)."
            )
        self._registry.bind(plotter, fem, scene)
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
