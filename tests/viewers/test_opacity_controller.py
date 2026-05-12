"""OpacityController + depth-peel auto-toggle (Phase 3.4).

Tracks per-actor opacity overrides and flips
``plotter.enable_depth_peeling`` / ``disable_depth_peeling`` only on
the 0↔1 boundary so the GPU cost is paid exactly when needed.

Picker correctness does not depend on depth peeling — that's the
mode-routed ``SetPickable`` job (Phase 3.2). Peeling here is purely
about rendering fidelity when translucent surfaces overlap.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.viewers.core.opacity_controller import OpacityController


class _StubProperty:
    def __init__(self) -> None:
        self._opacity = 1.0

    def SetOpacity(self, v: float) -> None:    # noqa: N802 — VTK API name
        self._opacity = float(v)

    def GetOpacity(self) -> float:             # noqa: N802 — VTK API name
        return self._opacity


class _StubActor:
    def __init__(self) -> None:
        self._prop = _StubProperty()

    def GetProperty(self) -> _StubProperty:    # noqa: N802 — VTK API name
        return self._prop


class _StubPlotter:
    def __init__(self) -> None:
        self.peeling: bool = False
        self.peel_n: int = 0
        self.peel_occ: float = 0.0
        self.enable_calls: int = 0
        self.disable_calls: int = 0

    def enable_depth_peeling(
        self, *, number_of_peels: int, occlusion_ratio: float,
    ) -> None:
        self.peeling = True
        self.peel_n = int(number_of_peels)
        self.peel_occ = float(occlusion_ratio)
        self.enable_calls += 1

    def disable_depth_peeling(self) -> None:
        self.peeling = False
        self.disable_calls += 1


@pytest.fixture
def setup():
    plotter = _StubPlotter()
    ctrl = OpacityController(plotter)
    return ctrl, plotter


# ---------------------------------------------------------------------
# Per-actor SetOpacity
# ---------------------------------------------------------------------

def test_set_opacity_writes_to_actor_property(setup):
    ctrl, _ = setup
    actor = _StubActor()
    ctrl.set_opacity(actor, 0.5)
    assert actor.GetProperty().GetOpacity() == 0.5
    assert ctrl.opacity_of(actor) == 0.5


def test_set_opacity_clamps_to_unit_interval(setup):
    ctrl, _ = setup
    actor = _StubActor()
    ctrl.set_opacity(actor, 1.5)
    assert actor.GetProperty().GetOpacity() == 1.0
    ctrl.set_opacity(actor, -0.3)
    assert actor.GetProperty().GetOpacity() == 0.0


def test_set_opacity_idempotent_on_same_value(setup):
    ctrl, plotter = setup
    actor = _StubActor()
    ctrl.set_opacity(actor, 0.5)
    enable_after_first = plotter.enable_calls
    ctrl.set_opacity(actor, 0.5)
    # No additional peeling call on the no-op re-set.
    assert plotter.enable_calls == enable_after_first


# ---------------------------------------------------------------------
# Depth-peel auto-toggle on the 0↔1 boundary
# ---------------------------------------------------------------------

def test_first_translucent_actor_enables_peeling(setup):
    ctrl, plotter = setup
    actor = _StubActor()
    assert not plotter.peeling
    ctrl.set_opacity(actor, 0.4)
    assert plotter.peeling
    assert plotter.peel_n == 4
    assert plotter.peel_occ == pytest.approx(0.1)


def test_returning_to_opaque_disables_peeling(setup):
    ctrl, plotter = setup
    actor = _StubActor()
    ctrl.set_opacity(actor, 0.4)
    assert plotter.peeling
    ctrl.set_opacity(actor, 1.0)
    assert not plotter.peeling
    assert ctrl.n_translucent() == 0


def test_second_translucent_actor_does_not_re_enable(setup):
    """Peeling is already on; another translucent actor shouldn't
    trigger a redundant enable_depth_peeling call."""
    ctrl, plotter = setup
    a1 = _StubActor()
    a2 = _StubActor()
    ctrl.set_opacity(a1, 0.4)
    assert plotter.enable_calls == 1
    ctrl.set_opacity(a2, 0.6)
    assert plotter.enable_calls == 1
    assert ctrl.n_translucent() == 2


def test_one_returns_to_opaque_but_others_stay_translucent(setup):
    """Peeling stays on as long as ANY actor is translucent."""
    ctrl, plotter = setup
    a1 = _StubActor()
    a2 = _StubActor()
    ctrl.set_opacity(a1, 0.4)
    ctrl.set_opacity(a2, 0.6)
    assert plotter.peeling
    ctrl.set_opacity(a1, 1.0)
    assert plotter.peeling
    assert ctrl.n_translucent() == 1
    ctrl.set_opacity(a2, 1.0)
    assert not plotter.peeling


def test_restore_all_resets_every_actor(setup):
    ctrl, plotter = setup
    a1 = _StubActor()
    a2 = _StubActor()
    ctrl.set_opacity(a1, 0.5)
    ctrl.set_opacity(a2, 0.3)
    assert plotter.peeling
    ctrl.restore_all()
    assert a1.GetProperty().GetOpacity() == 1.0
    assert a2.GetProperty().GetOpacity() == 1.0
    assert ctrl.n_translucent() == 0
    assert not plotter.peeling


# ---------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------

def test_set_opacity_fires_opacity_changed(setup):
    ctrl, _ = setup
    fires: list[tuple[str, Any]] = []

    class _StubDispatcher:
        def fire(self, kind, *, payload=None):
            fires.append((kind, payload))

    ctrl.dispatcher = _StubDispatcher()
    actor = _StubActor()
    ctrl.set_opacity(actor, 0.5)
    from apeGmsh.viewers.diagrams._dispatch import OPACITY_CHANGED
    assert len(fires) == 1
    assert fires[0][0] == OPACITY_CHANGED


def test_no_fire_on_idempotent_set(setup):
    ctrl, _ = setup
    fires: list[Any] = []

    class _StubDispatcher:
        def fire(self, kind, *, payload=None):
            fires.append(kind)

    ctrl.dispatcher = _StubDispatcher()
    actor = _StubActor()
    ctrl.set_opacity(actor, 0.5)
    ctrl.set_opacity(actor, 0.5)
    assert len(fires) == 1


def test_restore_all_fires_once(setup):
    ctrl, _ = setup
    fires: list[Any] = []

    class _StubDispatcher:
        def fire(self, kind, *, payload=None):
            fires.append(kind)

    actor = _StubActor()
    ctrl.set_opacity(actor, 0.5)
    ctrl.dispatcher = _StubDispatcher()
    ctrl.restore_all()
    assert fires.count("opacity_changed") == 1


# ---------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------

def test_set_opacity_with_none_is_noop(setup):
    ctrl, plotter = setup
    ctrl.set_opacity(None, 0.5)    # type: ignore[arg-type]
    assert ctrl.n_translucent() == 0
    assert not plotter.peeling


def test_no_plotter_does_not_crash():
    """OpacityController may be constructed before the plotter exists
    (e.g., headless tests). Set_opacity must still update the actor."""
    ctrl = OpacityController(None)
    actor = _StubActor()
    ctrl.set_opacity(actor, 0.5)
    assert actor.GetProperty().GetOpacity() == 0.5
    # Peeling can't enable without a plotter, but the state stays sane.
    assert ctrl.peeling_enabled is False


def test_peeling_enabled_property_reflects_state(setup):
    ctrl, _ = setup
    actor = _StubActor()
    assert ctrl.peeling_enabled is False
    ctrl.set_opacity(actor, 0.5)
    assert ctrl.peeling_enabled is True
    ctrl.set_opacity(actor, 1.0)
    assert ctrl.peeling_enabled is False
