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
    GEOMETRY_REMOVED,
    GEOMETRY_RENAMED,
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
    (GEOMETRY_DEFORM_CHANGED, {}, ["deform", "render"]),
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
