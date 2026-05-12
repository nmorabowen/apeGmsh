"""PickEngine actor-inventory handshake (Phase 3.1).

The engine is a thin dict keyed by ``id(actor)`` → ``(kind,
reverse_map_fn, actor)``. Diagrams register on attach and unregister
on detach so the click controller can route a vtkCellPicker hit
directly without walking every active diagram.

Phase 3.2 wires ``set_pick_mode`` on top of this inventory; for 3.1
we just verify the inventory itself + the GaussPointDiagram handshake.
"""
from __future__ import annotations

from typing import Any, Optional

import pytest

from apeGmsh.viewers.core.results_pick_engine import PickEngine, PickMode


class _StubActor:
    """Minimal stand-in for a vtkProp / vtkActor."""
    def __init__(self) -> None:
        self._pickable = True

    def SetPickable(self, v: bool) -> None:    # noqa: N802 — VTK API name
        self._pickable = bool(v)

    def GetPickable(self) -> bool:             # noqa: N802 — VTK API name
        return self._pickable


# ---------------------------------------------------------------------
# Core inventory contract
# ---------------------------------------------------------------------

def test_register_actor_records_kind_and_reverse_map():
    pe = PickEngine()
    actor = _StubActor()

    def _rev(cell_id: int) -> Optional[tuple]:
        return (100 + cell_id, 0, (0.0, 0.0, 0.0))

    pe.register_actor(actor, "gp", _rev)
    assert pe.is_registered(actor) is True
    assert pe.kind_for_actor(actor) == "gp"
    assert len(pe) == 1


def test_resolve_pick_dispatches_to_reverse_map():
    pe = PickEngine()
    actor = _StubActor()

    def _rev(cell_id: int) -> Optional[tuple]:
        return (200, cell_id, (1.0, 2.0, 3.0))

    pe.register_actor(actor, "gp", _rev)
    result = pe.resolve_pick(actor, 7)
    assert result == (200, 7, (1.0, 2.0, 3.0))


def test_resolve_pick_returns_none_for_unregistered_actor():
    pe = PickEngine()
    actor = _StubActor()
    assert pe.resolve_pick(actor, 0) is None


def test_resolve_pick_returns_none_when_reverse_map_returns_none():
    pe = PickEngine()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _cell: None)
    assert pe.resolve_pick(actor, 5) is None


def test_resolve_pick_swallows_reverse_map_exception():
    pe = PickEngine()
    actor = _StubActor()

    def _raise(_cell: int) -> Optional[tuple]:
        raise RuntimeError("boom")

    pe.register_actor(actor, "gp", _raise)
    # The picker doesn't want to crash on a bad reverse map — log/swallow.
    assert pe.resolve_pick(actor, 0) is None


def test_unregister_drops_actor():
    pe = PickEngine()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _c: None)
    pe.unregister_actor(actor)
    assert pe.is_registered(actor) is False
    assert len(pe) == 0


def test_unregister_unknown_actor_is_noop():
    pe = PickEngine()
    actor = _StubActor()
    # No registration first — should not raise.
    pe.unregister_actor(actor)
    assert len(pe) == 0


def test_register_overwrites_prior_entry():
    """Diagrams that re-attach should not duplicate entries."""
    pe = PickEngine()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _c: (1, 0))
    pe.register_actor(actor, "gp", lambda _c: (2, 0))
    result = pe.resolve_pick(actor, 0)
    assert result == (2, 0)
    assert len(pe) == 1


def test_register_actor_with_none_is_noop():
    pe = PickEngine()
    pe.register_actor(None, "gp", lambda _c: None)
    assert len(pe) == 0


def test_registered_actors_snapshot():
    pe = PickEngine()
    a1 = _StubActor()
    a2 = _StubActor()
    pe.register_actor(a1, "gp", lambda _c: None)
    pe.register_actor(a2, "fiber", lambda _c: None)
    pairs = pe.registered_actors()
    kinds = sorted(k for (k, _a) in pairs)
    assert kinds == ["fiber", "gp"]
    assert len(pairs) == 2


# ---------------------------------------------------------------------
# Integration — GaussPointDiagram attach/detach round-trip
# ---------------------------------------------------------------------

@pytest.fixture
def _gp_setup(g, tmp_path):
    """Solid hex mesh + 1-GP gauss results — mirrors the pattern in
    test_gauss_marker.py:gauss_results, kept local so this file doesn't
    cross-import test-module fixtures."""
    import numpy as np
    from pathlib import Path
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    eids: list[int] = []
    for group in fem.elements:
        if group.element_type.dim == 3:
            eids.extend(int(x) for x in group.ids)
    eids = sorted(eids)
    assert eids, "no 3-D elements meshed"

    n_elem = len(eids)
    values = np.zeros((1, n_elem, 1), dtype=np.float64)
    for ei in range(n_elem):
        values[0, ei, 0] = float(ei)
    natural = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    path = tmp_path / "pe_handshake.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="s", kind="transient",
            time=np.zeros(1, dtype=np.float64),
        )
        w.write_gauss_group(
            sid, "partition_0", group_id="g0",
            class_tag=10, int_rule=0,
            element_index=np.asarray(eids, dtype=np.int64),
            natural_coords=natural,
            local_axes_quaternion=None,
            components={"stress_xx": values},
        )
        w.end_stage()
    return Results.from_native(path), fem, eids


def test_gauss_diagram_registers_on_attach_unregisters_on_detach(_gp_setup):
    """Attach a GaussPointDiagram against a headless plotter and verify
    it lands in (and leaves) the scene's PickEngine inventory."""
    import pyvista as pv
    from apeGmsh.viewers.diagrams import (
        DiagramSpec,
        GaussPointDiagram,
        GaussMarkerStyle,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    results, fem, eids = _gp_setup
    scene = build_fem_scene(fem)
    # Mount a PickEngine on the scene exactly as ResultsViewer does.
    pe = PickEngine()
    scene.pick_engine = pe

    plotter = pv.Plotter(off_screen=True)
    diagram = GaussPointDiagram(
        DiagramSpec(
            kind="gauss_marker",
            selector=SlabSelector(component="stress_xx"),
            style=GaussMarkerStyle(),
        ),
        results,
    )
    try:
        diagram.attach(plotter, fem, scene)
        # The diagram registered its actor on attach.
        assert len(pe) == 1
        pairs = pe.registered_actors()
        kinds = [k for (k, _a) in pairs]
        assert kinds == ["gp"]
        # Resolve a pick on cell 0 → (eid, gp_index, world).
        actor = pairs[0][1]
        result = pe.resolve_pick(actor, 0)
        assert result is not None
        eid_resolved, gp_idx, world = result
        assert eid_resolved in eids
        assert gp_idx == 0
        assert len(world) == 3

        diagram.detach()
        # Detach removes the actor from inventory.
        assert len(pe) == 0
    finally:
        plotter.close()


# ---------------------------------------------------------------------
# 3.2 — PickMode enum + mode-routed SetPickable
# ---------------------------------------------------------------------

def test_default_mode_is_node():
    pe = PickEngine()
    assert pe.mode is PickMode.NODE


def test_set_pick_mode_to_gp_makes_gp_actors_pickable():
    pe = PickEngine()
    gp_actor = _StubActor()
    fiber_actor = _StubActor()
    pe.register_actor(gp_actor, "gp", lambda _c: None)
    pe.register_actor(fiber_actor, "fiber", lambda _c: None)

    pe.set_pick_mode(PickMode.GP)
    assert gp_actor.GetPickable() is True
    assert fiber_actor.GetPickable() is False
    assert pe.mode is PickMode.GP


def test_set_pick_mode_to_fiber_makes_fiber_actors_pickable():
    pe = PickEngine()
    gp_actor = _StubActor()
    fiber_actor = _StubActor()
    pe.register_actor(gp_actor, "gp", lambda _c: None)
    pe.register_actor(fiber_actor, "fiber", lambda _c: None)

    pe.set_pick_mode(PickMode.FIBER)
    assert gp_actor.GetPickable() is False
    assert fiber_actor.GetPickable() is True


def test_set_pick_mode_to_node_drops_all_inventory_pickability():
    """NODE / ELEMENT modes target the substrate, not inventory."""
    pe = PickEngine()
    gp_actor = _StubActor()
    pe.register_actor(gp_actor, "gp", lambda _c: None)
    # GP-mode first so the actor is pickable.
    pe.set_pick_mode(PickMode.GP)
    assert gp_actor.GetPickable() is True
    # Switch to NODE → GP actor should drop.
    pe.set_pick_mode(PickMode.NODE)
    assert gp_actor.GetPickable() is False
    pe.set_pick_mode(PickMode.ELEMENT)
    assert gp_actor.GetPickable() is False


def test_set_pick_mode_accepts_string():
    """Convenience: keyboard shortcuts may pass the string value."""
    pe = PickEngine()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _c: None)
    pe.set_pick_mode("gp")    # type: ignore[arg-type]
    assert pe.mode is PickMode.GP
    assert actor.GetPickable() is True


def test_set_pick_mode_no_op_when_unchanged():
    """Same-mode set should not fire PICK_MODE_CHANGED (avoid storms)."""
    pe = PickEngine()
    fires: list[tuple[str, Any]] = []

    class _StubDispatcher:
        def fire(self, kind: str, *, payload: Any = None) -> None:
            fires.append((kind, payload))

    pe.dispatcher = _StubDispatcher()
    pe.set_pick_mode(PickMode.GP)
    assert len(fires) == 1
    pe.set_pick_mode(PickMode.GP)    # same mode — no event
    assert len(fires) == 1


def test_set_pick_mode_publishes_event():
    """Dispatcher receives PICK_MODE_CHANGED with the mode value as payload."""
    pe = PickEngine()
    fires: list[tuple[str, Any]] = []

    class _StubDispatcher:
        def fire(self, kind: str, *, payload: Any = None) -> None:
            fires.append((kind, payload))

    pe.dispatcher = _StubDispatcher()
    pe.set_pick_mode(PickMode.GP)
    from apeGmsh.viewers.diagrams._dispatch import PICK_MODE_CHANGED
    assert fires == [(PICK_MODE_CHANGED, "gp")]


def test_register_actor_respects_current_mode():
    """A diagram that attaches into GP-mode lands pickable; one that
    attaches into NODE-mode lands NOT pickable. The engine applies
    the current allow-list at registration time."""
    pe = PickEngine()
    pe.set_pick_mode(PickMode.GP)
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _c: None)
    assert actor.GetPickable() is True

    pe.set_pick_mode(PickMode.NODE)
    actor2 = _StubActor()
    pe.register_actor(actor2, "gp", lambda _c: None)
    assert actor2.GetPickable() is False


def test_with_pick_through_temporarily_enables_all():
    """Alt-modifier context manager flips every actor to pickable,
    restores on exit."""
    pe = PickEngine()
    gp_actor = _StubActor()
    fiber_actor = _StubActor()
    pe.register_actor(gp_actor, "gp", lambda _c: None)
    pe.register_actor(fiber_actor, "fiber", lambda _c: None)
    pe.set_pick_mode(PickMode.GP)
    assert gp_actor.GetPickable() is True
    assert fiber_actor.GetPickable() is False

    with pe.with_pick_through():
        assert gp_actor.GetPickable() is True
        assert fiber_actor.GetPickable() is True

    # Restored to GP-mode state.
    assert gp_actor.GetPickable() is True
    assert fiber_actor.GetPickable() is False


def test_with_pick_through_nested_restores_outer_state():
    """Nested with_pick_through preserves whatever was set when the
    outer block entered, not the very-original state."""
    pe = PickEngine()
    actor = _StubActor()
    pe.register_actor(actor, "fiber", lambda _c: None)
    pe.set_pick_mode(PickMode.NODE)    # actor not pickable

    with pe.with_pick_through():
        assert actor.GetPickable() is True
        with pe.with_pick_through():
            assert actor.GetPickable() is True
        # Inner exit restores to "true" (outer state when it entered).
        assert actor.GetPickable() is True
    # Outer exit restores to "false" (the truly original state).
    assert actor.GetPickable() is False


def test_pickmode_string_values():
    """PickMode enum values match the expected string keys used by
    keyboard shortcuts."""
    assert PickMode.NODE.value == "node"
    assert PickMode.ELEMENT.value == "element"
    assert PickMode.GP.value == "gp"
    assert PickMode.FIBER.value == "fiber"


def test_gauss_diagram_attach_with_no_pick_engine_is_noop(_gp_setup):
    """Diagrams must tolerate ``scene.pick_engine is None`` (headless
    test contexts, mesh-only flows). The attach should still succeed."""
    import pyvista as pv
    from apeGmsh.viewers.diagrams import (
        DiagramSpec,
        GaussPointDiagram,
        GaussMarkerStyle,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    results, fem, _eids = _gp_setup
    scene = build_fem_scene(fem)
    # No pick_engine set on scene.
    assert getattr(scene, "pick_engine", None) is None

    plotter = pv.Plotter(off_screen=True)
    diagram = GaussPointDiagram(
        DiagramSpec(
            kind="gauss_marker",
            selector=SlabSelector(component="stress_xx"),
            style=GaussMarkerStyle(),
        ),
        results,
    )
    try:
        # Should not raise even though pick_engine is None.
        diagram.attach(plotter, fem, scene)
        diagram.detach()
    finally:
        plotter.close()
