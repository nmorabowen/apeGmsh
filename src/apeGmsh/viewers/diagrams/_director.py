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

from ._registry import DiagramRegistry

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

        self.on_step_changed: list[Callable[[int], None]] = []
        self.on_stage_changed: list[Callable[[str], None]] = []
        self.on_diagrams_changed: list[Callable[[], None]] = []
        self.on_picked_gp_changed: list[
            Callable[[Optional[tuple[int, int]]], None]
        ] = []

        self._render_callback: Optional[Callable[[], None]] = None

        # Pick a default stage if there is exactly one (matches
        # Results._resolve_stage's "auto" behaviour).
        stages = self._all_stages()
        if len(stages) == 1:
            self._stage_id = stages[0].id

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
        if self._stage_id is None:
            return 0
        try:
            return int(self._scoped_results().n_steps)
        except Exception:
            return 0

    @property
    def time_vector(self) -> ndarray:
        if self._stage_id is None:
            return np.array([], dtype=np.float64)
        try:
            return np.asarray(self._scoped_results().time, dtype=np.float64)
        except Exception:
            return np.array([], dtype=np.float64)

    def stages(self) -> "list[StageInfo]":
        return self._all_stages()

    def current_time(self) -> Optional[float]:
        """Time value at the current step, or None if no stage."""
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
        """
        info = self._lookup_stage(stage_id_or_name)
        if info.id == self._stage_id:
            return
        self._stage_id = info.id
        self._step_index = 0
        # Re-attach happens implicitly because diagrams hold references
        # to the Results object (which is stage-aware via stage_id at
        # read time). For Phase 0 we only have empty diagrams, but the
        # contract is in place.
        self._registry.reattach_all()
        self._fire_stage_changed(info.id)
        self._registry.update_to_step(self._step_index)
        self._render()

    def set_step(self, step_index: int) -> None:
        """Move the active step. Coalesces into one render."""
        n = self.n_steps
        if n == 0:
            return
        clamped = max(0, min(int(step_index), n - 1))
        if clamped == self._step_index:
            return
        self._step_index = clamped
        self._registry.update_to_step(clamped)
        self._fire_step_changed(clamped)
        self._render()

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
