"""PR5 — :class:`apeGmsh.viewers.core.overlay_visibility.OverlayVisibilityModel`.

The MVC seam that unifies mesh.viewer outline-eye and tab-checkbox
state.  Pure Python — no Qt — so the contract is testable without
``QApplication``.

The critical property the model guarantees: idempotent setters.  A
write that reflects state already set by the other surface is a
no-op and does not fire observers.  This is what breaks the
documented oscillation (``_mesh_outline_tree.py:96-104``) where two
independent writers ping-pong each other through the rebuild
callbacks.
"""
from __future__ import annotations

from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel


# =====================================================================
# Initial state
# =====================================================================


def test_fresh_model_is_empty():
    m = OverlayVisibilityModel()
    assert m.load_patterns == frozenset()
    assert m.constraint_kinds == frozenset()
    assert m.mass_visible is False


# =====================================================================
# Setters update state and fire observers
# =====================================================================


def test_set_load_patterns_updates_and_fires():
    m = OverlayVisibilityModel()
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_load_patterns(["dead", "live"])

    assert m.load_patterns == frozenset({"dead", "live"})
    assert len(calls) == 1


def test_set_constraint_kinds_updates_and_fires():
    m = OverlayVisibilityModel()
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_constraint_kinds({"rigid_link", "node_to_surface"})

    assert m.constraint_kinds == frozenset({"rigid_link", "node_to_surface"})
    assert len(calls) == 1


def test_set_mass_visible_updates_and_fires():
    m = OverlayVisibilityModel()
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_mass_visible(True)

    assert m.mass_visible is True
    assert len(calls) == 1


def test_setters_accept_any_iterable():
    """Lists, sets, tuples, generators — all coerce to frozenset."""
    m = OverlayVisibilityModel()
    m.set_load_patterns(("a", "b"))
    assert m.load_patterns == frozenset({"a", "b"})
    m.set_load_patterns(x for x in ["c"])
    assert m.load_patterns == frozenset({"c"})


# =====================================================================
# Idempotency — the oscillation fix
# =====================================================================


def test_no_op_set_does_not_fire():
    """Setting a value equal to the current state is silent — this is
    what breaks the outline-eye ↔ tab-checkbox oscillation."""
    m = OverlayVisibilityModel()
    m.set_load_patterns({"dead"})
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_load_patterns({"dead"})           # mirror write
    m.set_constraint_kinds(frozenset())     # already empty
    m.set_mass_visible(False)               # already False

    assert calls == []


def test_idempotent_across_iterable_types():
    """{'dead'}, ['dead'], ('dead',) all hash equal as frozenset — a
    mirror write through any of them is a no-op."""
    m = OverlayVisibilityModel()
    m.set_load_patterns(["dead"])
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_load_patterns({"dead"})
    m.set_load_patterns(("dead",))

    assert calls == []


# =====================================================================
# Subscribe / unsubscribe
# =====================================================================


def test_multiple_observers_all_fire():
    m = OverlayVisibilityModel()
    calls: dict[str, int] = {"a": 0, "b": 0}
    m.subscribe(lambda: calls.__setitem__("a", calls["a"] + 1))
    m.subscribe(lambda: calls.__setitem__("b", calls["b"] + 1))

    m.set_mass_visible(True)

    assert calls == {"a": 1, "b": 1}


def test_unsubscribe_silences_observer():
    m = OverlayVisibilityModel()
    calls: list[None] = []

    def cb():
        calls.append(None)

    m.subscribe(cb)
    m.set_mass_visible(True)
    assert len(calls) == 1

    m.unsubscribe(cb)
    m.set_mass_visible(False)
    assert len(calls) == 1   # second change does NOT fire


def test_unsubscribe_missing_is_noop():
    m = OverlayVisibilityModel()
    # Should not raise.
    m.unsubscribe(lambda: None)


def test_observer_can_unsubscribe_itself_during_fire():
    """Observers that mutate the subscription list inside their own
    callback don't corrupt the iteration."""
    m = OverlayVisibilityModel()
    calls: list[str] = []

    def first():
        calls.append("first")
        m.unsubscribe(first)

    def second():
        calls.append("second")

    m.subscribe(first)
    m.subscribe(second)
    m.set_mass_visible(True)

    assert calls == ["first", "second"]

    # Second write — only `second` remains.
    m.set_mass_visible(False)
    assert calls == ["first", "second", "second"]
