"""ResultsViewer event-loop dispatcher.

A single-source pipeline for the four primitives that drive what the
viewport paints:

* **STEP**  — push current step values to one or all diagrams
              (``Diagram.update_to_step(step_index)``).
* **DEFORM** — recompute deformed substrate points and call
              ``Diagram.sync_substrate_points(deformed_pts, scene)`` on
              one or all diagrams. Also mutates ``scene.grid.points``
              in place when the scope is "all" so substrate-bound
              actors follow.
* **GATE**  — run the composition gate: each actor's visibility is
              ``d.is_visible AND (no_active_comp OR id(d) in active_layers)``.
* **RENDER** — single coalesced ``plotter.render()``.

Every UI gesture / observer / shortcut funnels through
``Dispatcher.fire(event_kind, ...)`` which selects the right primitive
sequence from the event matrix. This is the only place those four
primitives may run.

Every dispatch fires through ``apeGmsh.viewers._log.log_action``
(category ``dispatch``). The session log file captures the full
sequence with timestamps + duration; bug reports attach the most
recent file and we replay every gesture.

Event matrix (mirrors the contract locked in PR review):

| event                       | scope         | STEP | DEFORM | GATE | RENDER |
|-----------------------------|---------------|------|--------|------|--------|
| step_changed                | all           |  ✓   |   ✓    |  -   |   ✓    |
| deform_changed              | all           |  -   |   ✓    |  -   |   ✓    |
| stage_changed               | all (re-attach + step) | ✓ | ✓ | ✓ |   ✓    |
| comp_active_changed         | -             |  -   |   -    |  ✓   |   ✓    |
| diagram_attached            | this layer    |  ✓   |   ✓    |  ✓   |   ✓    |
| diagram_detached            | -             |  -   |   -    |  ✓   |   ✓    |
| diagram_modified            | this layer    |  ✓   |   ✓    |  -   |   ✓    |
| layer_visibility_changed    | -             |  -   |   -    |  ✓   |   ✓    |
| layer_reordered             | -             |  -   |   -    |  ✓ + restack | ✓ |
| pick_cleared                | -             |  -   |   -    |  -   |   ✓    |

``session_batch(...)`` is a context manager that suppresses every
primitive in between, then runs one full pump on exit. Use it during
``_apply_session`` to kill the N-squared registry pump.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from .._log import log_action

# Public event kinds
STEP_CHANGED = "step_changed"
DEFORM_CHANGED = "deform_changed"
STAGE_CHANGED = "stage_changed"
COMP_ACTIVE_CHANGED = "comp_active_changed"
DIAGRAM_ATTACHED = "diagram_attached"
DIAGRAM_DETACHED = "diagram_detached"
DIAGRAM_MODIFIED = "diagram_modified"
LAYER_VISIBILITY_CHANGED = "layer_visibility_changed"
LAYER_REORDERED = "layer_reordered"
PICK_CLEARED = "pick_cleared"
# Compound event covering any change to the geometry tree:
# deform toggle/scale/field, active geometry, comp create/rename/
# delete, comp active, layer membership. Granular dispatches from
# individual call sites (toggle, composition click) take precedence
# when they fire first; this is the catch-all so the trace covers
# every geometry observer fire.
GEOMETRIES_CHANGED = "geometries_changed"

class Dispatcher:
    """Event-loop pipeline for ResultsViewer.

    Constructed by the viewer once at ``show()``; injected into the
    director (``director.dispatcher``) so call sites that don't hold a
    viewer reference (settings tab, outline tree, …) can fire events.

    Pump callables are supplied by the viewer because they touch the
    plotter / scene / actor list — state the dispatcher itself doesn't
    own.
    """

    def __init__(
        self,
        director: Any,
        *,
        pump_step: Callable[[Optional[Any]], None],
        pump_deform: Callable[[Optional[Any]], None],
        pump_gate: Callable[[], None],
        pump_restack: Callable[[], None],
        render: Callable[[], None],
    ) -> None:
        self._director = director
        self._pump_step = pump_step
        self._pump_deform = pump_deform
        self._pump_gate = pump_gate
        self._pump_restack = pump_restack
        self._render = render
        self._suppress_depth: int = 0
        self._suppressed_kinds: set[str] = set()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def fire(self, kind: str, *, layer: Any = None) -> None:
        """Run the event matrix entry for ``kind``.

        ``layer`` is consulted only by events whose matrix row scopes
        the pump to one diagram (``diagram_attached``,
        ``diagram_modified``). Other events ignore it.
        """
        if self._suppress_depth > 0:
            self._suppressed_kinds.add(kind)
            log_action(
                "dispatch", "suppressed",
                kind=kind, layer=_layer_id(layer), _level="debug",
            )
            return

        t0 = time.perf_counter()

        if kind == STEP_CHANGED:
            self._pump_step(None)
            self._pump_deform(None)
        elif kind == DEFORM_CHANGED:
            self._pump_deform(None)
        elif kind == GEOMETRIES_CHANGED:
            # Compound: deform may have changed (scale/toggle/field)
            # AND composition active may have changed. Run both pumps;
            # they're idempotent.
            self._pump_deform(None)
            self._pump_gate()
        elif kind == STAGE_CHANGED:
            # The director itself runs reattach_all + update_to_step
            # before firing this event; the dispatcher just refreshes
            # gate + deformation + render so the new attach lands on
            # the deformed substrate with correct composition filtering.
            self._pump_step(None)
            self._pump_deform(None)
            self._pump_gate()
        elif kind == COMP_ACTIVE_CHANGED:
            self._pump_gate()
        elif kind == DIAGRAM_ATTACHED:
            if layer is not None:
                self._pump_step(layer)
                self._pump_deform(layer)
            self._pump_gate()
        elif kind == DIAGRAM_DETACHED:
            self._pump_gate()
        elif kind == DIAGRAM_MODIFIED:
            if layer is not None:
                self._pump_step(layer)
                self._pump_deform(layer)
        elif kind == LAYER_VISIBILITY_CHANGED:
            self._pump_gate()
        elif kind == LAYER_REORDERED:
            self._pump_restack()
            self._pump_gate()
        elif kind == PICK_CLEARED:
            pass    # only RENDER fires
        else:
            log_action(
                "dispatch", "unknown_kind", kind=kind, _level="warning",
            )

        self._render()

        dt_ms = (time.perf_counter() - t0) * 1000.0
        log_action(
            "dispatch", kind, layer=_layer_id(layer), duration_ms=round(dt_ms, 2),
        )

    @contextmanager
    def session_batch(self) -> Iterator[None]:
        """Suppress all dispatch inside the block; one full pump on exit.

        Use during multi-layer restore / bulk-add flows so the registry
        observer doesn't pump ``K(K+1)/2`` times for K layers.
        """
        self._suppress_depth += 1
        log_action(
            "dispatch", "batch_start", depth=self._suppress_depth,
            _level="debug",
        )
        try:
            yield
        finally:
            self._suppress_depth -= 1
            if self._suppress_depth == 0 and self._suppressed_kinds:
                kinds = sorted(self._suppressed_kinds)
                self._suppressed_kinds.clear()
                log_action(
                    "dispatch", "batch_flush", suppressed=str(kinds),
                )
                # One full pump matching STAGE_CHANGED semantics —
                # everything was potentially mutated.
                self._pump_step(None)
                self._pump_deform(None)
                self._pump_gate()
                self._render()
            log_action(
                "dispatch", "batch_end", depth=self._suppress_depth,
                _level="debug",
            )


def _layer_id(layer: Any) -> str:
    if layer is None:
        return "<none>"
    try:
        return f"{type(layer).__name__}#{id(layer):x}"
    except Exception:
        return "<unknown>"
