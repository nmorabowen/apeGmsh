"""Dispatcher contract — ADR 0056 V1 (dispatcher-always + owners-fire).

Locks:

* The event matrix as data: firing each kind runs exactly the
  primitives the docstring table promises (the table-driven refactor
  must not drift from the legacy elif chain).
* Optional no-op pumps + ``bind()`` — the Director constructs the
  dispatcher before the viewer exists; ``show()`` rebinds.
* ``gesture_batch()`` — N same-gesture fires collapse to the
  matrix-row union once, plus one render (the owners-fire perf gate:
  an N-layer eye cascade costs one gate pump, not N).
* ``DiagramRegistry.set_visible`` owner-fires
  ``LAYER_VISIBILITY_CHANGED`` and is idempotent per call.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from apeGmsh.viewers.diagrams._dispatch import (
    COMP_ACTIVE_CHANGED,
    DEFORM_CHANGED,
    DIAGRAM_ATTACHED,
    DIAGRAM_DETACHED,
    DIAGRAM_MODIFIED,
    Dispatcher,
    ELEMENT_VISIBILITY_CHANGED,
    GEOMETRIES_CHANGED,
    GEOMETRY_ACTIVE_CHANGED,
    GEOMETRY_ADDED,
    GEOMETRY_DEFORM_CHANGED,
    GEOMETRY_OFFSET_CHANGED,
    GEOMETRY_REMOVED,
    GEOMETRY_RENAMED,
    GEOMETRY_STAGE_PIN_CHANGED,
    GEOMETRY_VISIBILITY_CHANGED,
    LAYER_REORDERED,
    LAYER_VISIBILITY_CHANGED,
    OPACITY_CHANGED,
    PICK_CLEARED,
    PICK_MODE_CHANGED,
    STAGE_CHANGED,
    STEP_CHANGED,
)


class _Recorder:
    """Records pump invocations as ('step', layer) / ('gate',) … tuples."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def make_dispatcher(self) -> Dispatcher:
        return Dispatcher(
            MagicMock(),
            pump_step=lambda layer: self.calls.append(("step", layer)),
            pump_deform=lambda layer: self.calls.append(("deform", layer)),
            pump_gate=lambda: self.calls.append(("gate",)),
            pump_restack=lambda: self.calls.append(("restack",)),
            render=lambda: self.calls.append(("render",)),
            defer_fn=lambda fn: fn(),    # synchronous UI lane
        )

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


# =====================================================================
# Matrix equivalence — one assertion per docstring-table row
# =====================================================================

# kind -> (expected primitive names in order, fire kwargs)
_EXPECTED: list[tuple[str, dict, list[str]]] = [
    (STEP_CHANGED, {}, ["step", "deform", "render"]),
    (DEFORM_CHANGED, {}, ["deform", "render"]),
    (GEOMETRIES_CHANGED, {}, ["deform", "gate", "render"]),
    (GEOMETRY_ACTIVE_CHANGED, {}, ["deform", "gate", "render"]),
    (GEOMETRY_VISIBILITY_CHANGED, {}, ["deform", "gate", "render"]),
    (GEOMETRY_DEFORM_CHANGED, {}, ["deform", "render"]),
    (GEOMETRY_OFFSET_CHANGED, {}, ["deform", "render"]),
    (GEOMETRY_STAGE_PIN_CHANGED, {}, ["step", "deform", "render"]),
    (GEOMETRY_ADDED, {}, ["gate", "render"]),
    (GEOMETRY_REMOVED, {}, ["deform", "gate", "render"]),
    (GEOMETRY_RENAMED, {}, ["render"]),
    (ELEMENT_VISIBILITY_CHANGED, {}, ["render"]),
    (OPACITY_CHANGED, {}, ["render"]),
    (PICK_MODE_CHANGED, {}, []),                      # no pumps, no render
    (STAGE_CHANGED, {}, ["step", "deform", "gate", "render"]),
    (COMP_ACTIVE_CHANGED, {}, ["gate", "render"]),
    (DIAGRAM_DETACHED, {}, ["gate", "render"]),
    (LAYER_VISIBILITY_CHANGED, {}, ["gate", "render"]),
    (LAYER_REORDERED, {}, ["restack", "gate", "render"]),
    (PICK_CLEARED, {}, ["render"]),
]


def test_matrix_rows_run_expected_primitives():
    for kind, kwargs, expected in _EXPECTED:
        rec = _Recorder()
        rec.make_dispatcher().fire(kind, **kwargs)
        assert rec.names() == expected, f"matrix row drifted for {kind!r}"


def test_layer_scoped_kinds_pump_the_layer():
    layer = object()
    rec = _Recorder()
    rec.make_dispatcher().fire(DIAGRAM_MODIFIED, layer=layer)
    assert rec.calls == [("step", layer), ("deform", layer), ("render",)]

    rec2 = _Recorder()
    rec2.make_dispatcher().fire(DIAGRAM_ATTACHED, layer=layer)
    assert rec2.calls == [
        ("step", layer), ("deform", layer), ("gate",), ("render",),
    ]


def test_layer_scoped_kinds_skip_step_deform_without_layer():
    rec = _Recorder()
    rec.make_dispatcher().fire(DIAGRAM_MODIFIED)
    assert rec.names() == ["render"]

    rec2 = _Recorder()
    rec2.make_dispatcher().fire(DIAGRAM_ATTACHED)
    assert rec2.names() == ["gate", "render"]


def test_unscoped_kinds_pump_all_even_with_layer():
    """A layer passed to an unscoped kind is ignored (legacy fire()
    consulted ``layer`` only for the two diagram events)."""
    rec = _Recorder()
    rec.make_dispatcher().fire(STEP_CHANGED, layer=object())
    assert rec.calls[0] == ("step", None)
    assert rec.calls[1] == ("deform", None)


# =====================================================================
# Dispatcher-always: no-op pumps + bind
# =====================================================================

def test_constructs_with_noop_pumps_and_fires_safely():
    disp = Dispatcher(MagicMock(), defer_fn=lambda fn: fn())
    # Every kind fires without error against the no-op defaults.
    disp.fire(STEP_CHANGED)
    disp.fire(LAYER_VISIBILITY_CHANGED)
    with disp.gesture_batch():
        disp.fire(LAYER_VISIBILITY_CHANGED)
    with disp.session_batch():
        disp.fire(STEP_CHANGED)


def test_bind_rebinds_real_pumps():
    rec = _Recorder()
    disp = Dispatcher(MagicMock(), defer_fn=lambda fn: fn())
    disp.fire(LAYER_VISIBILITY_CHANGED)        # no-op pumps, no recording
    assert rec.calls == []
    disp.bind(
        pump_step=lambda layer: rec.calls.append(("step", layer)),
        pump_deform=lambda layer: rec.calls.append(("deform", layer)),
        pump_gate=lambda: rec.calls.append(("gate",)),
        pump_restack=lambda: rec.calls.append(("restack",)),
        render=lambda: rec.calls.append(("render",)),
    )
    disp.fire(LAYER_VISIBILITY_CHANGED)
    assert rec.names() == ["gate", "render"]


# =====================================================================
# gesture_batch — the owners-fire perf gate
# =====================================================================

def test_gesture_batch_collapses_cascade_to_one_gate_pump():
    """The ADR 0056 V1 perf gate: an N-layer eye cascade fires
    LAYER_VISIBILITY_CHANGED N times inside the batch and costs ONE
    gate pump + ONE render — same as a single fire."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    with disp.gesture_batch():
        for _ in range(25):
            disp.fire(LAYER_VISIBILITY_CHANGED)
    assert rec.names().count("gate") == 1
    assert rec.names().count("render") == 1
    assert rec.names().count("step") == 0
    assert rec.names().count("deform") == 0


def test_gesture_batch_replays_matrix_union():
    rec = _Recorder()
    disp = rec.make_dispatcher()
    with disp.gesture_batch():
        disp.fire(LAYER_VISIBILITY_CHANGED)    # gate
        disp.fire(DEFORM_CHANGED)              # deform
    # Union = {deform, gate}, fixed order step→deform→restack→gate.
    assert rec.names() == ["deform", "gate", "render"]


def test_gesture_batch_noop_when_nothing_fired():
    rec = _Recorder()
    disp = rec.make_dispatcher()
    with disp.gesture_batch():
        pass
    assert rec.calls == []


def test_gesture_batch_inside_session_batch_defers_to_outer():
    """Nested batches: the outermost batch's exit semantics win —
    a gesture inside a session flush must not pump early."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    with disp.session_batch():
        with disp.gesture_batch():
            disp.fire(LAYER_VISIBILITY_CHANGED)
        assert rec.calls == []     # inner exit pumps nothing
    # Outer session flush = full pump.
    assert "step" in rec.names() and "gate" in rec.names()
    assert rec.names().count("render") == 1


# =====================================================================
# Batch exit replays the RENDER lane (ADR 0084 D3, fork F-A)
# =====================================================================

def _render_spy(disp, kinds, seen, name="a"):
    """Subscribe a RENDER-lane handler that records ``(name, kind, payload)``."""
    from apeGmsh.viewers.diagrams._dispatch import Lane

    disp.subscribe(
        kinds,
        lambda kind, payload: seen.append((name, kind, payload)),
        lane=Lane.RENDER,
    )


def test_gesture_batch_replays_render_lane_once_per_handler():
    """N qualifying events inside one batch replay the subscriber ONCE
    — the id-coalescing that keeps an eye cascade from paying N tree
    rebuilds — and the batch still costs exactly one render."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    seen: list = []
    _render_spy(disp, LAYER_VISIBILITY_CHANGED, seen)

    with disp.gesture_batch():
        for _ in range(25):
            disp.fire(LAYER_VISIBILITY_CHANGED)

    assert len(seen) == 1
    assert rec.names().count("render") == 1
    assert rec.names().count("gate") == 1


def test_render_lane_replay_lands_between_the_pump_and_the_render():
    """Ordering contract, unchanged from an unsuppressed fire:
    primitives → RENDER lane → render."""
    order: list[str] = []
    disp = Dispatcher(
        MagicMock(),
        pump_gate=lambda: order.append("gate"),
        render=lambda: order.append("render"),
        defer_fn=lambda fn: fn(),
    )
    from apeGmsh.viewers.diagrams._dispatch import Lane
    disp.subscribe(
        LAYER_VISIBILITY_CHANGED,
        lambda _k, _p: order.append("render_sub"),
        lane=Lane.RENDER,
    )

    with disp.gesture_batch():
        disp.fire(LAYER_VISIBILITY_CHANGED)
        disp.fire(LAYER_VISIBILITY_CHANGED)

    assert order == ["gate", "render_sub", "render"]


def test_render_lane_replay_coalesces_per_handler_not_per_kind():
    """One handler on two kinds replays once; two handlers on one kind
    replay twice. Identity is the subscriber, not the event."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    from apeGmsh.viewers.diagrams._dispatch import Lane

    both: list = []
    handler = lambda kind, _p: both.append(kind)      # noqa: E731
    disp.subscribe(
        (LAYER_VISIBILITY_CHANGED, DIAGRAM_DETACHED), handler,
        lane=Lane.RENDER,
    )
    solo: list = []
    _render_spy(disp, LAYER_VISIBILITY_CHANGED, solo, name="second")

    with disp.gesture_batch():
        disp.fire(LAYER_VISIBILITY_CHANGED)
        disp.fire(DIAGRAM_DETACHED)
        disp.fire(LAYER_VISIBILITY_CHANGED)

    assert len(both) == 1          # one handler, one replay
    assert len(solo) == 1          # a different handler, its own replay


def test_bound_method_subscriber_coalesces_on_the_stored_record():
    """A bound method mints a fresh object on every attribute access,
    so ``id(self.handler)`` computed at replay time would never match.
    The dedup key is the callable the subscription table HOLDS."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    from apeGmsh.viewers.diagrams._dispatch import Lane

    class _Sub:
        def __init__(self) -> None:
            self.calls: list = []

        def on_event(self, kind, payload) -> None:
            self.calls.append((kind, payload))

    sub = _Sub()
    # The premise: two accesses are two distinct objects.
    assert sub.on_event is not sub.on_event
    disp.subscribe(LAYER_VISIBILITY_CHANGED, sub.on_event, lane=Lane.RENDER)

    with disp.gesture_batch():
        for _ in range(10):
            disp.fire(LAYER_VISIBILITY_CHANGED)

    assert len(sub.calls) == 1


def test_render_lane_replay_keeps_the_last_payload():
    """Last-wins, mirroring the UI lane: a replayed handler sees the
    most recent payload of its kind."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    seen: list = []
    _render_spy(disp, GEOMETRY_VISIBILITY_CHANGED, seen)

    with disp.gesture_batch():
        disp.fire(GEOMETRY_VISIBILITY_CHANGED, payload="g0")
        disp.fire(GEOMETRY_VISIBILITY_CHANGED, payload="g1")

    assert seen == [("a", GEOMETRY_VISIBILITY_CHANGED, "g1")]


def test_render_lane_subscribers_of_unfired_kinds_are_not_replayed():
    """The replay set is 'what fired', not 'everything subscribed'."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    fired: list = []
    quiet: list = []
    _render_spy(disp, LAYER_VISIBILITY_CHANGED, fired, name="fired")
    _render_spy(disp, GEOMETRY_REMOVED, quiet, name="quiet")

    with disp.gesture_batch():
        disp.fire(LAYER_VISIBILITY_CHANGED)

    assert len(fired) == 1
    assert quiet == []


def test_session_batch_replays_the_render_lane_too():
    """``session_batch`` is the restore-scale sibling; ADR 0084 D3
    retires its hand-patched compensations the same way."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    seen: list = []
    _render_spy(disp, GEOMETRY_ACTIVE_CHANGED, seen)

    with disp.session_batch():
        disp.fire(GEOMETRY_ACTIVE_CHANGED, payload="g0")
        disp.fire(GEOMETRY_ACTIVE_CHANGED, payload="g1")

    assert len(seen) == 1
    assert rec.names().count("render") == 1


def test_nested_batches_replay_once_at_the_outer_exit():
    rec = _Recorder()
    disp = rec.make_dispatcher()
    seen: list = []
    _render_spy(disp, LAYER_VISIBILITY_CHANGED, seen)

    with disp.session_batch():
        with disp.gesture_batch():
            disp.fire(LAYER_VISIBILITY_CHANGED)
        assert seen == []          # the inner exit replays nothing
        disp.fire(LAYER_VISIBILITY_CHANGED)

    assert len(seen) == 1


def test_unsuppressed_fires_still_run_the_render_lane_every_time():
    """Coalescing is a batch-exit property only — outside a batch each
    fire runs its RENDER-lane subscribers, as it always did."""
    rec = _Recorder()
    disp = rec.make_dispatcher()
    seen: list = []
    _render_spy(disp, LAYER_VISIBILITY_CHANGED, seen)

    for _ in range(3):
        disp.fire(LAYER_VISIBILITY_CHANGED)

    assert len(seen) == 3


# =====================================================================
# Registry owner-fire (ADR 0056 Part 2)
# =====================================================================

class _FakeDiagram:
    def __init__(self, visible: bool = True) -> None:
        self.is_visible = bool(visible)

    def set_visible(self, v: bool) -> None:
        self.is_visible = bool(v)


def test_registry_set_visible_owner_fires():
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    reg = DiagramRegistry()
    rec = _Recorder()
    reg.dispatcher = rec.make_dispatcher()
    d = _FakeDiagram(visible=True)

    reg.set_visible(d, False)
    assert d.is_visible is False
    assert rec.names() == ["gate", "render"]


def test_registry_set_visible_idempotent_skip():
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    reg = DiagramRegistry()
    rec = _Recorder()
    reg.dispatcher = rec.make_dispatcher()
    notified: list[bool] = []
    reg.on_changed.append(lambda: notified.append(True))
    d = _FakeDiagram(visible=True)

    reg.set_visible(d, True)       # no-op write
    assert rec.calls == []
    assert notified == []


def test_registry_without_dispatcher_still_works():
    """Standalone registries (unit tests) have no dispatcher injected."""
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    reg = DiagramRegistry()
    d = _FakeDiagram(visible=True)
    reg.set_visible(d, False)
    assert d.is_visible is False


# =====================================================================
# Mesh-viewer kinds (ADR 0056 V3)
# =====================================================================


def _make_mesh_recorder():
    """Recorder dispatcher with the mesh pumps bound."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    calls: list[tuple] = []
    disp = Dispatcher(
        MagicMock(),
        pump_entities=lambda: calls.append(("entities",)),
        pump_overlays=lambda key=None: calls.append(("overlays", key)),
        render=lambda: calls.append(("render",)),
        defer_fn=lambda fn: fn(),
    )
    return disp, calls


def test_mesh_entity_visibility_runs_entities_pump():
    from apeGmsh.viewers.diagrams._dispatch import (
        MESH_ENTITY_VISIBILITY_CHANGED,
    )

    disp, calls = _make_mesh_recorder()
    disp.fire(MESH_ENTITY_VISIBILITY_CHANGED)
    assert calls == [("entities",), ("render",)]


def test_mesh_overlay_changed_passes_key_through():
    """MESH_OVERLAY_CHANGED is pass-through-scoped: the overlay key
    rides ``layer`` (None = all), with no skip on None."""
    from apeGmsh.viewers.diagrams._dispatch import MESH_OVERLAY_CHANGED

    disp, calls = _make_mesh_recorder()
    disp.fire(MESH_OVERLAY_CHANGED, layer="loads")
    disp.fire(MESH_OVERLAY_CHANGED)            # no key -> all
    assert calls == [
        ("overlays", "loads"), ("render",),
        ("overlays", None), ("render",),
    ]


def test_mesh_gesture_batch_replays_overlays_unscoped_once():
    """A cascade of keyed overlay fires inside a gesture batch replays
    ONE unscoped overlays pump (key=None -> rebuild all) + one render."""
    from apeGmsh.viewers.diagrams._dispatch import MESH_OVERLAY_CHANGED

    disp, calls = _make_mesh_recorder()
    with disp.gesture_batch():
        disp.fire(MESH_OVERLAY_CHANGED, layer="loads")
        disp.fire(MESH_OVERLAY_CHANGED, layer="mass")
        disp.fire(MESH_OVERLAY_CHANGED, layer="constraints")
    assert calls == [("overlays", None), ("render",)]


# =====================================================================
# Owner-fires: VisibilityManager + OverlayVisibilityModel (V3)
# =====================================================================


def test_visibility_manager_owner_fires_and_defers_rebuild():
    """With a dispatcher injected the mutator fires
    MESH_ENTITY_VISIBILITY_CHANGED and the rebuild runs as the pump;
    without one the legacy inline rebuild runs (model viewer until V4)."""
    from apeGmsh.viewers.core.visibility import VisibilityManager
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    rebuilt: list[str] = []

    # Subclass to stub the heavy internals (the class uses __slots__,
    # so instance-level method patching is impossible) — we're
    # testing the propagation shape, not the VTK rebuild.
    class _StubVM(VisibilityManager):
        __slots__ = ()

        def _rebuild_actors(self) -> None:
            rebuilt.append("actors")

        def _reset_colors(self) -> None:
            rebuilt.append("colors")

    vm = _StubVM(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # Legacy path (no dispatcher): inline rebuild.
    vm.set_hidden({(3, 1)})
    assert rebuilt == ["actors", "colors"]

    # Dispatcher path: the pump does the rebuild, exactly once,
    # synchronously inside the fire.
    rebuilt.clear()
    order: list[str] = []
    disp = Dispatcher(
        MagicMock(),
        pump_entities=vm.rebuild_now,
        render=lambda: order.append("render"),
        defer_fn=lambda fn: fn(),
    )
    vm.dispatcher = disp
    vm.on_changed.append(lambda: order.append("observer"))
    vm.set_hidden({(3, 2)})
    assert rebuilt == ["actors", "colors"]
    # Render precedes the on_changed observers (post-rebuild state).
    assert order == ["render", "observer"]


def test_overlay_model_owner_fires_keyed():
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel

    m = OverlayVisibilityModel()
    fired: list[tuple] = []
    m.dispatcher = MagicMock()
    m.dispatcher.fire = lambda kind, layer=None: fired.append((kind, layer))

    m.set_load_patterns({"dead"})
    m.set_mass_visible(True)
    m.set_constraint_kinds({"rigid_link"})
    m.set_boundary_nodes_visible(True)
    keys = [k for _, k in fired]
    assert keys == ["loads", "mass", "constraints", "boundary"]

    # Idempotent: re-writing the same state fires nothing.
    fired.clear()
    m.set_load_patterns({"dead"})
    m.set_mass_visible(True)
    assert fired == []


def test_overlay_model_owns_scales():
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel

    m = OverlayVisibilityModel()
    fired: list[tuple] = []
    m.dispatcher = MagicMock()
    m.dispatcher.fire = lambda kind, layer=None: fired.append((kind, layer))

    assert m.scale("force_arrow") == 1.0
    m.set_scale("force_arrow", 2.5)
    assert m.scale("force_arrow") == 2.5
    m.set_scale("mass_sphere", 0.5)
    m.set_scale("tangent_normal_arrow", 3.0)
    assert [k for _, k in fired] == ["loads", "mass", "tangent"]

    # Idempotent per call.
    fired.clear()
    m.set_scale("force_arrow", 2.5)
    assert fired == []

    # Unknown keys fail loud (INV-6 — no silent no-op scales).
    import pytest
    with pytest.raises(KeyError, match="Unknown overlay scale"):
        m.set_scale("typo_key", 2.0)
