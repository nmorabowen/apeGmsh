"""DiagramRegistry — add/remove/move/visibility/observers (Phase 0).

Pure-Python tests with stub Diagrams that record their lifecycle calls
so we can assert attach/detach/update happens at the right times. No
Qt, no plotter — the registry doesn't render itself.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.viewers.diagrams import (
    Diagram,
    DiagramRegistry,
    DiagramSpec,
    DiagramStyle,
    SlabSelector,
)


# =====================================================================
# Stub Diagram for testing
# =====================================================================

class _StubDiagram(Diagram):
    kind = "stub"

    def __init__(self, spec, results=None):
        super().__init__(spec, results=results)
        self.attach_calls: list[Any] = []
        self.detach_calls: int = 0
        self.update_calls: list[int] = []

    def attach(self, plotter, fem, scene=None):
        super().attach(plotter, fem, scene)
        self.attach_calls.append((id(plotter), id(fem)))

    def update_to_step(self, step_index: int) -> None:
        self.update_calls.append(int(step_index))

    def detach(self) -> None:
        self.detach_calls += 1
        super().detach()


def _make_stub(label: str = "x") -> _StubDiagram:
    spec = DiagramSpec(
        kind="stub",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
        label=label,
    )
    return _StubDiagram(spec, results=None)


class _DummyPlotter:
    """Minimal stub: just needs ``remove_actor`` (called by detach)."""
    def remove_actor(self, *args, **kwargs):
        pass


class _DummyFem:
    """Marker — the Director's bind requires fem is not None."""
    pass


# =====================================================================
# CRUD without binding
# =====================================================================

def test_empty_registry_iter_len():
    r = DiagramRegistry()
    assert len(r) == 0
    assert list(r) == []


def test_add_without_bind_does_not_attach():
    r = DiagramRegistry()
    d = _make_stub()
    r.add(d)
    assert len(r) == 1
    assert d.attach_calls == []
    assert not d.is_attached


def test_remove_drops_from_list():
    r = DiagramRegistry()
    d = _make_stub()
    r.add(d)
    r.remove(d)
    assert len(r) == 0


def test_remove_nonexistent_is_noop():
    r = DiagramRegistry()
    d = _make_stub()
    r.remove(d)    # should not raise
    assert len(r) == 0


def test_clear():
    r = DiagramRegistry()
    for _ in range(3):
        r.add(_make_stub())
    r.clear()
    assert len(r) == 0


def test_move_up_and_down():
    r = DiagramRegistry()
    a = _make_stub("a")
    b = _make_stub("b")
    c = _make_stub("c")
    for d in (a, b, c):
        r.add(d)
    r.move(0, 2)        # a -> end
    assert [d.spec.label for d in r] == ["b", "c", "a"]
    r.move(2, 0)        # a -> start
    assert [d.spec.label for d in r] == ["a", "b", "c"]


def test_move_clamps_out_of_range():
    r = DiagramRegistry()
    a = _make_stub("a")
    b = _make_stub("b")
    r.add(a)
    r.add(b)
    r.move(0, 99)       # clamp to last
    assert [d.spec.label for d in r] == ["b", "a"]


# =====================================================================
# Bind / unbind lifecycle
# =====================================================================

def test_bind_attaches_existing_diagrams():
    r = DiagramRegistry()
    d = _make_stub()
    r.add(d)
    r.bind(_DummyPlotter(), _DummyFem())
    assert d.is_attached
    assert len(d.attach_calls) == 1


def test_add_after_bind_attaches_immediately():
    r = DiagramRegistry()
    r.bind(_DummyPlotter(), _DummyFem())
    d = _make_stub()
    r.add(d)
    assert d.is_attached
    assert len(d.attach_calls) == 1


def test_unbind_detaches_all():
    r = DiagramRegistry()
    r.bind(_DummyPlotter(), _DummyFem())
    a = _make_stub()
    b = _make_stub()
    r.add(a)
    r.add(b)
    r.unbind()
    assert a.detach_calls == 1
    assert b.detach_calls == 1
    assert not a.is_attached
    assert not b.is_attached


def test_remove_attached_calls_detach():
    r = DiagramRegistry()
    r.bind(_DummyPlotter(), _DummyFem())
    d = _make_stub()
    r.add(d)
    r.remove(d)
    assert d.detach_calls == 1


def test_reattach_all_cycles_each_diagram():
    r = DiagramRegistry()
    r.bind(_DummyPlotter(), _DummyFem())
    d = _make_stub()
    r.add(d)
    r.reattach_all()
    assert d.detach_calls == 1
    assert len(d.attach_calls) == 2


# =====================================================================
# Step routing
# =====================================================================

def test_update_to_step_only_visible_attached():
    r = DiagramRegistry()
    r.bind(_DummyPlotter(), _DummyFem())
    a = _make_stub("a")
    b = _make_stub("b")
    r.add(a)
    r.add(b)
    r.set_visible(b, False)
    r.update_to_step(7)
    assert a.update_calls == [7]
    assert b.update_calls == []


def test_visibility_toggle_actor_via_default():
    r = DiagramRegistry()
    d = _make_stub()
    r.add(d)
    assert d.is_visible
    r.set_visible(d, False)
    assert not d.is_visible
    r.set_visible(d, True)
    assert d.is_visible


# =====================================================================
# Observers
# =====================================================================

def test_observer_fires_on_add():
    r = DiagramRegistry()
    calls = []
    r.subscribe(lambda: calls.append("changed"))
    r.add(_make_stub())
    assert calls == ["changed"]


def test_observer_fires_on_remove():
    r = DiagramRegistry()
    d = _make_stub()
    r.add(d)
    calls = []
    r.subscribe(lambda: calls.append("changed"))
    r.remove(d)
    assert calls == ["changed"]


def test_observer_fires_on_move():
    r = DiagramRegistry()
    r.add(_make_stub("a"))
    r.add(_make_stub("b"))
    calls = []
    r.subscribe(lambda: calls.append("changed"))
    r.move(0, 1)
    assert calls == ["changed"]


def test_unsubscribe_stops_notifications():
    r = DiagramRegistry()
    calls = []
    unsub = r.subscribe(lambda: calls.append("x"))
    unsub()
    r.add(_make_stub())
    assert calls == []
