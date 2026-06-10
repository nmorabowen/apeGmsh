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

from types import SimpleNamespace
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
# Outline eye-toggle — dispatcher routing
# =====================================================================


def _bind_fire_layer_visibility(director: object):
    """Bind OutlineTree._fire_layer_visibility to a stub namespace —
    same headless trick as test_outline_visibility_state.py (the real
    __init__ needs a QApplication)."""
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    ns = SimpleNamespace()
    ns._director = director
    ns._fire_render = MagicMock()
    ns._fire_layer_visibility = (
        OutlineTree._fire_layer_visibility.__get__(ns)
    )
    return ns


def test_eye_toggle_fires_layer_visibility_changed():
    from apeGmsh.viewers.diagrams._dispatch import LAYER_VISIBILITY_CHANGED

    director = SimpleNamespace(dispatcher=MagicMock())
    ns = _bind_fire_layer_visibility(director)

    ns._fire_layer_visibility()

    director.dispatcher.fire.assert_called_once_with(
        LAYER_VISIBILITY_CHANGED
    )
    ns._fire_render.assert_not_called()


def test_eye_toggle_falls_back_to_render_without_dispatcher():
    director = SimpleNamespace()    # no .dispatcher attribute
    ns = _bind_fire_layer_visibility(director)

    ns._fire_layer_visibility()

    ns._fire_render.assert_called_once()
