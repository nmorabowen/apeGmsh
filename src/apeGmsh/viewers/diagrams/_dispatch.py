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
| geometries_changed          | (omnibus)     |  -   |   ✓    |  ✓   |   ✓    |
| geometry_active_changed     | payload geom  |  -   |   ✓    |  ✓   |   ✓    |
| geometry_visibility_changed | payload geom  |  -   |   ✓    |  ✓   |   ✓    |
| geometry_deform_changed     | payload geom  |  -   |   ✓    |  -   |   ✓    |
| geometry_offset_changed     | payload geom  |  -   |   ✓    |  -   |   ✓    |
| geometry_stage_pin_changed  | payload geom  |  ✓   |   ✓    |  -   |   ✓    |
| geometry_added              | payload geom  |  -   |   -    |  ✓   |   ✓    |
| geometry_removed            | payload geom  |  -   |   ✓    |  ✓   |   ✓    |
| geometry_renamed            | payload geom  |  -   |   -    |  -   |   ✓    |
| composition_changed         | payload comp  |  -   |   -    |  ✓   |   ✓    |
| element_visibility_changed  | payload pick  |  -   |   -    |  -   |   ✓    |
| opacity_changed             | payload actor |  -   |   -    |  -   |   ✓    |
| pick_mode_changed           | payload mode  |  -   |   -    |  -   |   -    |

The granular ``geometry_*`` / ``composition_changed`` rows fire from
``GeometryManager.subscribe_typed`` / ``CompositionManager`` mutations
with the relevant id as ``payload``. When one of them fires in the
same notification chain as the omnibus ``geometries_changed``, the
omnibus is suppressed — the granular row already runs the right pump.

The Phase-3 ``element_visibility_changed`` / ``opacity_changed`` /
``pick_mode_changed`` events run no pump. They exist to fan-out to
RENDER-lane subscribers — cell-ghost flips, actor opacity updates,
``SetPickable`` toggles. ``pick_mode_changed`` further skips the
closing ``render()`` because pickability isn't visually observable
(the next pick gesture sees the new flags).

``session_batch(...)`` is a context manager that suppresses every
primitive in between, then runs one full pump on exit. Use it during
``_apply_session`` to kill the N-squared registry pump.
``gesture_batch(...)`` is its interactive-cascade sibling: on exit it
replays only the matrix-row **union** of the kinds suppressed inside
the block (ADR 0056 Part 2) — an N-layer eye cascade costs one gate
pump + one render, not N.

ADR 0056 Part 3: the Director constructs this dispatcher at
``__init__`` with no-op pumps, so ``director.dispatcher`` always
exists; the viewer rebinds the real pumps via :meth:`Dispatcher.bind`
at ``show()``. Owner mutators (``DiagramRegistry.set_visible``, the
geometry/composition managers, ``ElementVisibility``) fire their own
events — call sites only call mutators.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Optional

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

# Granular geometry events — emitted by GeometryManager /
# CompositionManager call sites that know exactly what changed. Each
# carries the relevant ``geom_id`` or ``composition_id`` as ``payload``
# so UI subscribers can coalesce on identity.
# When a granular event fires in the same notification chain as the
# omnibus GEOMETRIES_CHANGED, the omnibus is suppressed (one-tick
# dedup) to keep the heavy pump from running twice for the same logical
# change.
GEOMETRY_ACTIVE_CHANGED = "geometry_active_changed"
# ADR 0058 S2b — a geometry's ``visible`` flag flipped (concurrent
# rendering). Runs DEFORM (a newly-shown geometry materializes /
# re-syncs its scene) + GATE (its layers join / leave the viewport).
GEOMETRY_VISIBILITY_CHANGED = "geometry_visibility_changed"
GEOMETRY_DEFORM_CHANGED = "geometry_deform_changed"
# ADR 0058 S3a — a geometry's spatial ``offset`` changed. Runs DEFORM
# only (the offset is a pump-time term added to the computed points —
# ``reference + offset + scale·field``); the node-tree invalidation +
# label-overlay rebuild ride a RENDER-lane subscriber.
GEOMETRY_OFFSET_CHANGED = "geometry_offset_changed"
# ADR 0058 S3b — a geometry's ``stage_id`` pin changed. Runs STEP +
# DEFORM (the pinned geometry's diagrams step through the pinned
# stage's clamped cursor; its substrate re-warps from the pinned
# stage's field). GATE is untouched — layer visibility doesn't depend
# on the pin; the per-scene LAYER_STAGE mask resync + render ride a
# RENDER-lane subscriber. The director's typed observer (registered
# before the viewer's dispatcher bridge) re-attaches that geometry's
# diagrams first, so the pumps land on fresh attachments.
GEOMETRY_STAGE_PIN_CHANGED = "geometry_stage_pin_changed"
GEOMETRY_ADDED = "geometry_added"
GEOMETRY_REMOVED = "geometry_removed"
GEOMETRY_RENAMED = "geometry_renamed"
COMPOSITION_CHANGED = "composition_changed"

_GRANULAR_GEOMETRY_KINDS = frozenset({
    GEOMETRY_ACTIVE_CHANGED,
    GEOMETRY_VISIBILITY_CHANGED,
    GEOMETRY_DEFORM_CHANGED,
    GEOMETRY_OFFSET_CHANGED,
    GEOMETRY_STAGE_PIN_CHANGED,
    GEOMETRY_ADDED,
    GEOMETRY_REMOVED,
    GEOMETRY_RENAMED,
    COMPOSITION_CHANGED,
})

# Lightweight events used by Phase 3 (interior-selection feature). The
# matrix entries below run no pumps — these events exist solely to
# fan-out to RENDER-lane subscribers (cell-ghost flips, actor opacity
# updates, SetPickable toggles). PICK_MODE_CHANGED additionally skips
# the closing render() because changing pickability isn't visually
# observable; the next pick gesture sees the new flags.
ELEMENT_VISIBILITY_CHANGED = "element_visibility_changed"
OPACITY_CHANGED = "opacity_changed"
PICK_MODE_CHANGED = "pick_mode_changed"

_NO_RENDER_KINDS = frozenset({PICK_MODE_CHANGED})

# ── Mesh-viewer event kinds (ADR 0056 V3) ───────────────────────────
# The Dispatcher is viewer-agnostic (ADR 0056 Part 6 / open question 1
# resolved: one shared module, kinds are data). The mesh viewer binds
# its own pumps onto two mesh primitives:
#
# * MESH_ENTITY_VISIBILITY_CHANGED — VisibilityManager state changed
#   (hide / isolate / reveal / set_hidden). Pump = the actor rebuild
#   that used to run inline inside the owner's mutators.
# * MESH_OVERLAY_CHANGED — OverlayVisibilityModel state changed
#   (pattern / kind / mass / boundary flags, or an overlay scale).
#   ``layer`` carries the overlay key ("loads" / "mass" /
#   "constraints" / "boundary" / "tangent"); ``None`` means ALL —
#   which is exactly what a batch replay needs, so this kind is
#   pass-through-scoped, not skip-scoped.
MESH_ENTITY_VISIBILITY_CHANGED = "mesh_entity_visibility_changed"
MESH_OVERLAY_CHANGED = "mesh_overlay_changed"

# ── The event matrix as data (ADR 0056) ─────────────────────────────
# kind -> primitives to run, executed in fixed order:
#   step → deform → restack → gate → entities → overlays.
# Every row in the docstring table above is encoded here; ``fire()``
# and ``gesture_batch()`` both read this table so the two can't drift.
# Kinds in ``_LAYER_SCOPED`` pump step/deform scoped to the fired
# ``layer`` and skip them when no layer is supplied; kinds in
# ``_LAYER_PASSTHROUGH`` hand ``layer`` to their pump as an opaque
# scope key (None = all) without the skip.
_STEP, _DEFORM, _RESTACK, _GATE = "step", "deform", "restack", "gate"
_ENTITIES, _OVERLAYS = "entities", "overlays"

_MATRIX: dict[str, frozenset[str]] = {
    STEP_CHANGED: frozenset({_STEP, _DEFORM}),
    DEFORM_CHANGED: frozenset({_DEFORM}),
    GEOMETRIES_CHANGED: frozenset({_DEFORM, _GATE}),
    GEOMETRY_ACTIVE_CHANGED: frozenset({_DEFORM, _GATE}),
    GEOMETRY_VISIBILITY_CHANGED: frozenset({_DEFORM, _GATE}),
    GEOMETRY_DEFORM_CHANGED: frozenset({_DEFORM}),
    GEOMETRY_OFFSET_CHANGED: frozenset({_DEFORM}),
    GEOMETRY_STAGE_PIN_CHANGED: frozenset({_STEP, _DEFORM}),
    GEOMETRY_ADDED: frozenset({_GATE}),
    GEOMETRY_REMOVED: frozenset({_DEFORM, _GATE}),
    GEOMETRY_RENAMED: frozenset(),
    COMPOSITION_CHANGED: frozenset({_GATE}),
    ELEMENT_VISIBILITY_CHANGED: frozenset(),
    OPACITY_CHANGED: frozenset(),
    PICK_MODE_CHANGED: frozenset(),
    STAGE_CHANGED: frozenset({_STEP, _DEFORM, _GATE}),
    COMP_ACTIVE_CHANGED: frozenset({_GATE}),
    DIAGRAM_ATTACHED: frozenset({_STEP, _DEFORM, _GATE}),
    DIAGRAM_DETACHED: frozenset({_GATE}),
    DIAGRAM_MODIFIED: frozenset({_STEP, _DEFORM}),
    LAYER_VISIBILITY_CHANGED: frozenset({_GATE}),
    LAYER_REORDERED: frozenset({_RESTACK, _GATE}),
    PICK_CLEARED: frozenset(),
    MESH_ENTITY_VISIBILITY_CHANGED: frozenset({_ENTITIES}),
    MESH_OVERLAY_CHANGED: frozenset({_OVERLAYS}),
}

# Kinds whose step/deform pumps are scoped to the fired ``layer`` —
# and skipped entirely when no layer is supplied (matches the legacy
# ``if layer is not None`` guards).
_LAYER_SCOPED = frozenset({DIAGRAM_ATTACHED, DIAGRAM_MODIFIED})

# Kinds whose pump receives ``layer`` as an opaque scope key with NO
# skip on None (None = all): the mesh overlay pump rebuilds one
# overlay when keyed, every overlay on a batch replay.
_LAYER_PASSTHROUGH = frozenset({MESH_OVERLAY_CHANGED})


def _noop_pump(layer: Any = None) -> None:
    """Default pump — does nothing. ADR 0056 Part 3: the dispatcher
    always exists (constructed by the Director with these defaults);
    the viewer rebinds real pumps via :meth:`Dispatcher.bind` at
    ``show()``."""
    return None


def _noop() -> None:
    return None


class Lane(str, Enum):
    """Subscriber dispatch lane.

    * ``RENDER`` — synchronous, fires inside ``fire()`` after the pump
      matrix runs and before ``render()``. No coalescing. Use for cheap
      side-effects that must be visible at the next render of the same
      tick (toggling ``SetPickable`` flags, flipping cell ghosts).
    * ``UI`` — deferred, posted to the Qt event loop via the injected
      ``defer_fn`` (default ``QTimer.singleShot(0, _flush)``). Optionally
      coalesces by ``(handler, kind, key_fn(payload))`` with last-wins:
      a storm of N events that all key to the same value invokes the
      handler at most once per flush. Use for tree rebuilds / panel
      refreshes that only need to reflect the latest state.
    """
    RENDER = "render"
    UI = "ui"


def _default_defer(fn: Callable[[], None]) -> None:
    """Default UI-lane scheduler: QTimer.singleShot(0, fn).

    Falls back to immediate execution when Qt isn't available (pure
    unit tests / library use). Tests that want explicit control over
    flush timing should inject their own ``defer_fn``.
    """
    try:
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, fn)
    except Exception:
        fn()


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
        pump_step: Optional[Callable[[Optional[Any]], None]] = None,
        pump_deform: Optional[Callable[[Optional[Any]], None]] = None,
        pump_gate: Optional[Callable[[], None]] = None,
        pump_restack: Optional[Callable[[], None]] = None,
        pump_entities: Optional[Callable[[], None]] = None,
        pump_overlays: Optional[Callable[[Optional[Any]], None]] = None,
        render: Optional[Callable[[], None]] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> None:
        # ADR 0056 Part 3: pumps are optional — the Director constructs
        # the dispatcher at __init__ with these no-op defaults so
        # ``director.dispatcher`` is never None; the viewer rebinds the
        # real pumps via :meth:`bind` at show(). Headless / unit-test
        # contexts exercise the same event path with no-op pumps.
        # step/deform/restack/gate are the results-viewer pumps;
        # entities/overlays are the mesh-viewer pumps (ADR 0056 V3) —
        # each viewer binds the slots it owns and leaves the rest no-op.
        self._director = director
        self._pump_step = pump_step or _noop_pump
        self._pump_deform = pump_deform or _noop_pump
        self._pump_gate = pump_gate or _noop
        self._pump_restack = pump_restack or _noop
        self._pump_entities = pump_entities or _noop
        self._pump_overlays = pump_overlays or _noop_pump
        self._render = render or _noop
        self._defer_fn = defer_fn or _default_defer
        self._suppress_depth: int = 0
        self._suppressed_kinds: set[str] = set()
        # Set by any granular geometry fire; cleared by the next
        # omnibus GEOMETRIES_CHANGED, which is then suppressed. Lets
        # legacy call sites that wire both granular + omnibus skip the
        # heavy redundant pump.
        self._granular_geometry_seen: bool = False

        # Lane subscriber tables.
        # RENDER: kind -> list[handler]. Synchronous, no coalesce.
        # UI:     kind -> list[(handler, key_fn or None, coalesce)].
        self._render_subs: dict[
            str, list[Callable[[str, Any], None]]
        ] = {}
        self._ui_subs: dict[
            str, list[tuple[Callable[[str, Any], None], Optional[Callable[[Any], Any]], bool]]
        ] = {}
        # Coalesced UI queue: ordered list of (handler, kind, payload).
        # Dedup map (id(handler), kind, key) -> index in _ui_pending so
        # last-wins replaces the payload in place without re-ordering.
        self._ui_pending: list[
            tuple[Callable[[str, Any], None], str, Any]
        ] = []
        self._ui_dedup: dict[tuple[int, str, Any], int] = {}
        self._ui_flush_scheduled: bool = False

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def bind(
        self,
        *,
        pump_step: Optional[Callable[[Optional[Any]], None]] = None,
        pump_deform: Optional[Callable[[Optional[Any]], None]] = None,
        pump_gate: Optional[Callable[[], None]] = None,
        pump_restack: Optional[Callable[[], None]] = None,
        pump_entities: Optional[Callable[[], None]] = None,
        pump_overlays: Optional[Callable[[Optional[Any]], None]] = None,
        render: Optional[Callable[[], None]] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> None:
        """Rebind real pumps + render (+ optional defer_fn).

        Called by a viewer at ``show()`` once the plotter / scene /
        actor list exist. Until then the owner-constructed dispatcher
        runs no-op pumps (ADR 0056 Part 3) so owner-fired events are
        cheap and the event path is identical headless. Only the slots
        passed are rebound — each viewer binds the pumps it owns
        (results: step/deform/restack/gate; mesh: entities/overlays).
        """
        if pump_step is not None:
            self._pump_step = pump_step
        if pump_deform is not None:
            self._pump_deform = pump_deform
        if pump_gate is not None:
            self._pump_gate = pump_gate
        if pump_restack is not None:
            self._pump_restack = pump_restack
        if pump_entities is not None:
            self._pump_entities = pump_entities
        if pump_overlays is not None:
            self._pump_overlays = pump_overlays
        if render is not None:
            self._render = render
        if defer_fn is not None:
            self._defer_fn = defer_fn

    def fire(self, kind: str, *, layer: Any = None, payload: Any = None) -> None:
        """Run the event matrix entry for ``kind``.

        ``layer`` is consulted only by events whose matrix row scopes
        the pump to one diagram (``diagram_attached``,
        ``diagram_modified``). Other events ignore it.

        ``payload`` is forwarded to every lane subscriber (RENDER + UI).
        It carries event-specific data — e.g., ``geom_id`` for the
        granular geometry events, ``composition_id`` for COMPOSITION_CHANGED.
        Subscribers receive ``handler(kind, payload)``.
        """
        if self._suppress_depth > 0:
            self._suppressed_kinds.add(kind)
            # Still queue UI subscribers so coalesce can collapse a
            # storm; session_batch drains the queue on exit. Don't
            # schedule a flush — the batch context owns flushing.
            self._enqueue_ui(kind, payload)
            log_action(
                "dispatch", "suppressed",
                kind=kind, layer=_layer_id(layer), _level="debug",
            )
            return

        # Omnibus guard — if a granular geometry kind already fired in
        # the current notification chain (manager._notify runs typed
        # subscribers before the legacy one), skip the redundant pump.
        # Clears the flag so the *next* omnibus fires normally.
        if kind == GEOMETRIES_CHANGED and self._granular_geometry_seen:
            self._granular_geometry_seen = False
            log_action(
                "dispatch", "geometries_changed_suppressed",
                _level="debug",
            )
            return
        if kind in _GRANULAR_GEOMETRY_KINDS:
            self._granular_geometry_seen = True

        t0 = time.perf_counter()

        # Table-driven primitive selection (ADR 0056) — one row per
        # kind, executed in fixed order step → deform → restack → gate.
        # The per-kind rationale lives in the module-docstring table.
        row = _MATRIX.get(kind)
        if row is None:
            log_action(
                "dispatch", "unknown_kind", kind=kind, _level="warning",
            )
            row = frozenset()
        self._run_primitives(row, kind=kind, layer=layer)

        # RENDER lane: synchronous, before plotter.render() so any
        # actor-flag updates land in the same frame.
        for handler in self._render_subs.get(kind, ()):
            try:
                handler(kind, payload)
            except Exception as exc:
                log_action(
                    "dispatch", "render_sub_error",
                    kind=kind, exc=type(exc).__name__, _level="warning",
                )

        # UI lane: enqueue (coalesce last-wins), schedule a flush if
        # there's pending work and one isn't already in flight.
        self._enqueue_ui(kind, payload)
        if self._ui_pending and not self._ui_flush_scheduled:
            self._ui_flush_scheduled = True
            self._defer_fn(self._flush_ui_lane)

        if kind not in _NO_RENDER_KINDS:
            self._render()

        dt_ms = (time.perf_counter() - t0) * 1000.0
        log_action(
            "dispatch", kind, layer=_layer_id(layer), duration_ms=round(dt_ms, 2),
        )

    def subscribe(
        self,
        kinds: "str | Iterable[str]",
        handler: Callable[[str, Any], None],
        *,
        lane: Lane = Lane.UI,
        coalesce: bool = True,
        key_fn: Optional[Callable[[Any], Any]] = None,
    ) -> Callable[[], None]:
        """Subscribe ``handler`` to one or more event kinds on ``lane``.

        Parameters
        ----------
        kinds
            One event kind string, or an iterable of them. Subscribing
            to multiple kinds with one call returns a single
            unsubscribe that drops the handler from all of them.
        handler
            Called as ``handler(kind, payload)``.
        lane
            ``Lane.RENDER`` runs synchronously inside ``fire()``;
            ``Lane.UI`` posts to the Qt event loop via ``defer_fn``.
        coalesce
            UI-lane only. When ``True``, multiple fires with the same
            ``(kind, key_fn(payload))`` collapse to one handler call
            with the last payload. Ignored on the RENDER lane.
        key_fn
            UI-lane only. Maps payload → coalesce key. ``None`` is
            equivalent to ``lambda p: None`` (collapse all events of
            the same kind to one).

        Returns
        -------
        Callable[[], None]
            Unsubscribe callable.
        """
        kinds_t: tuple[str, ...] = (
            (kinds,) if isinstance(kinds, str) else tuple(kinds)
        )
        for k in kinds_t:
            if lane is Lane.RENDER:
                self._render_subs.setdefault(k, []).append(handler)
            else:
                self._ui_subs.setdefault(k, []).append(
                    (handler, key_fn, bool(coalesce)),
                )

        def _unsub() -> None:
            for k in kinds_t:
                if lane is Lane.RENDER:
                    lst_r = self._render_subs.get(k)
                    if lst_r is not None:
                        lst_r[:] = [h for h in lst_r if h is not handler]
                else:
                    lst_u = self._ui_subs.get(k)
                    if lst_u is not None:
                        lst_u[:] = [
                            (h, kf, c) for (h, kf, c) in lst_u
                            if h is not handler
                        ]
        return _unsub

    # ------------------------------------------------------------------
    # Internal — primitive execution
    # ------------------------------------------------------------------

    def _run_primitives(
        self, row: "frozenset[str]", *, kind: str, layer: Any,
    ) -> None:
        """Run the primitives in ``row`` in the fixed order
        step → deform → restack → gate → entities → overlays.

        Kinds in ``_LAYER_SCOPED`` pump step/deform scoped to
        ``layer`` and skip them when ``layer`` is None (legacy
        ``if layer is not None`` semantics); kinds in
        ``_LAYER_PASSTHROUGH`` hand ``layer`` to their pump as an
        opaque scope key with no skip (None = all); all other kinds
        pump unscoped (``layer=None`` → all diagrams).
        """
        scoped = kind in _LAYER_SCOPED
        target = layer if scoped else None
        if _STEP in row and not (scoped and layer is None):
            self._pump_step(target)
        if _DEFORM in row and not (scoped and layer is None):
            self._pump_deform(target)
        if _RESTACK in row:
            self._pump_restack()
        if _GATE in row:
            self._pump_gate()
        if _ENTITIES in row:
            self._pump_entities()
        if _OVERLAYS in row:
            self._pump_overlays(
                layer if kind in _LAYER_PASSTHROUGH else None
            )

    # ------------------------------------------------------------------
    # Internal — UI lane plumbing
    # ------------------------------------------------------------------

    def _enqueue_ui(self, kind: str, payload: Any) -> None:
        """Push UI subscribers for ``kind`` onto the pending queue.

        When a subscriber opted into coalesce, replace any earlier
        entry with the same ``(handler, kind, key_fn(payload))`` so the
        handler ends up called once with the latest payload.
        """
        subs = self._ui_subs.get(kind)
        if not subs:
            return
        for handler, key_fn, coalesce in subs:
            if coalesce:
                key = key_fn(payload) if key_fn is not None else None
                dedup_key = (id(handler), kind, key)
                idx = self._ui_dedup.get(dedup_key)
                if idx is not None:
                    self._ui_pending[idx] = (handler, kind, payload)
                    continue
                self._ui_dedup[dedup_key] = len(self._ui_pending)
                self._ui_pending.append((handler, kind, payload))
            else:
                self._ui_pending.append((handler, kind, payload))

    def _flush_ui_lane(self) -> None:
        """Drain the UI queue. Called via ``defer_fn`` (QTimer) or
        directly by ``session_batch`` on exit."""
        self._ui_flush_scheduled = False
        if self._suppress_depth > 0:
            # The active session_batch owns draining — bail out so we
            # don't fire UI handlers in the middle of a suppressed run.
            return
        queue = self._ui_pending
        if not queue:
            return
        self._ui_pending = []
        self._ui_dedup = {}
        n = len(queue)
        t0 = time.perf_counter()
        for handler, kind, payload in queue:
            try:
                handler(kind, payload)
            except Exception as exc:
                log_action(
                    "dispatch", "ui_sub_error",
                    kind=kind, exc=type(exc).__name__, _level="warning",
                )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        log_action(
            "dispatch", "ui_flush", n=n, duration_ms=round(dt_ms, 2),
            _level="debug",
        )

    @contextmanager
    def gesture_batch(self) -> Iterator[None]:
        """Suppress dispatch inside the block; replay the **matrix-row
        union** of the suppressed kinds once on exit, plus one render.

        The interactive-cascade sibling of :meth:`session_batch`
        (ADR 0056 Part 2): an N-layer eye cascade that fires
        ``LAYER_VISIBILITY_CHANGED`` N times inside the block costs one
        gate pump + one render on exit — the same as a single fire.
        ``session_batch`` replays a *full* pump instead; use that for
        restore-scale bulk where anything may have changed.

        Layer-scoped rows (``diagram_modified`` / ``diagram_attached``)
        degrade to unscoped in the union — conservative (pumps all
        diagrams instead of one), never wrong. Batches share one
        suppress counter; when nested, the outermost batch's exit
        semantics win.
        """
        self._suppress_depth += 1
        log_action(
            "dispatch", "gesture_start", depth=self._suppress_depth,
            _level="debug",
        )
        try:
            yield
        finally:
            self._suppress_depth -= 1
            if self._suppress_depth == 0 and self._suppressed_kinds:
                kinds = sorted(self._suppressed_kinds)
                self._suppressed_kinds.clear()
                union: frozenset[str] = frozenset().union(
                    *(_MATRIX.get(k, frozenset()) for k in kinds)
                )
                log_action(
                    "dispatch", "gesture_flush",
                    suppressed=str(kinds), primitives=str(sorted(union)),
                )
                self._run_primitives(
                    union, kind="<gesture_batch>", layer=None,
                )
                self._flush_ui_lane()
                self._render()
            log_action(
                "dispatch", "gesture_end", depth=self._suppress_depth,
                _level="debug",
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
                # Drain the UI lane synchronously — fires were queued
                # during the batch (coalesced last-wins) so we don't
                # leave stale work behind. RENDER-lane subs aren't
                # queued; the batch's matrix-equivalent pump above is
                # the analogue.
                self._flush_ui_lane()
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
