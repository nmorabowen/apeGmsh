"""Gate-channel visibility — ``Diagram.apply_effective_visibility``.

The composition gate computes ``effective = is_visible AND
in_active_composition`` and must push it onto the rendered artifacts
of *every* diagram — including the backend-routed (ADR 0042 R-B) ones
that hold layer handles instead of raw actors — without clobbering
``is_visible`` (the user-intent flag the gate itself reads).

Regression context: ``pump_gate`` used to iterate ``d._actors``
directly, which is empty for every migrated diagram, so the
composition gate was a silent no-op post-R-B migration.

Also covers the outline tree's eye-toggle routing: it must fire
``LAYER_VISIBILITY_CHANGED`` through the dispatcher (same path as the
settings-tab visibility checkbox) so the gate re-runs, falling back to
a raw render only when no dispatcher is installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from apeGmsh.viewers.diagrams._base import Diagram, DiagramSpec
from apeGmsh.viewers.diagrams._selectors import SlabSelector
from apeGmsh.viewers.diagrams._styles import DiagramStyle


# =====================================================================
# Stub diagram — legacy actor path
# =====================================================================


class _FakeActor:
    def __init__(self) -> None:
        self.visible = True

    def SetVisibility(self, v: bool) -> None:
        self.visible = bool(v)


class _StubDiagram(Diagram):
    kind = "stub"
    topology = "nodes"

    def update_to_step(self, step_index: int) -> None:
        pass


def _make_stub(visible: bool = True) -> _StubDiagram:
    spec = DiagramSpec(
        kind="stub",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
        visible=visible,
    )
    d = _StubDiagram(spec, MagicMock())
    d._actors = [_FakeActor(), _FakeActor()]
    return d


# =====================================================================
# apply_effective_visibility — contract
# =====================================================================


def test_gate_off_hides_actors_but_preserves_intent():
    d = _make_stub(visible=True)
    d.apply_effective_visibility(False)
    assert all(a.visible is False for a in d._actors)
    # User intent untouched — the next gate run recomputes from it.
    assert d.is_visible is True


def test_gate_on_shows_actors_without_flipping_intent():
    d = _make_stub(visible=False)
    # A gate should never be asked to show a user-hidden layer
    # (desired = is_visible AND in_active), but the channel contract
    # holds regardless: artifacts follow ``effective``, intent doesn't.
    d.apply_effective_visibility(True)
    assert all(a.visible is True for a in d._actors)
    assert d.is_visible is False


def test_set_visible_still_owns_the_intent_flag():
    d = _make_stub(visible=True)
    d.set_visible(False)
    assert d.is_visible is False
    assert all(a.visible is False for a in d._actors)


def test_gate_routes_subclass_override():
    """The gate channel reuses the subclass's set_visible artifact
    path — a backend-routed override sees the effective value."""
    seen: list[bool] = []

    class _Routed(_StubDiagram):
        kind = "stub"

        def set_visible(self, visible: bool) -> None:
            self._visible = visible
            seen.append(bool(visible))

    spec = DiagramSpec(
        kind="stub",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
    )
    d = _Routed(spec, MagicMock())
    d.apply_effective_visibility(False)
    assert seen == [False]
    assert d.is_visible is True


# =====================================================================
# Outline eye-toggle — dispatcher routing (ADR 0056 V1)
# =====================================================================
#
# V0 routed eye-toggles through an OutlineTree._fire_layer_visibility
# helper (with a raw-render fallback for headless contexts). V1
# deleted both: the registry mutator owner-fires
# LAYER_VISIBILITY_CHANGED itself and bulk cascades wrap in
# dispatcher.gesture_batch() — see test_dispatcher_contract.py for the
# owner-fire and batch coverage. Here we lock that the deleted bypass
# helpers stay deleted (a re-introduced fallback is a dispatcher
# bypass, INV-4/INV-5).


def test_outline_has_no_raw_render_helpers():
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    assert not hasattr(OutlineTree, "_fire_layer_visibility")
    assert not hasattr(OutlineTree, "_fire_render")
