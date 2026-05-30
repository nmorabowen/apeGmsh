"""S2 acceptance — the ADR 0045 FilterController state machine.

Fully headless (pure Python, no Qt/VTK): exercises the ratified
multi-select keystroke semantics (toggle), the checkbox-panel path
(set_active replace), select_all/clear, the on_change emission contract
(fires only on real change), and out-of-range-dim no-ops. The viewer
wiring (keys, panel sync, visibility feedback) is eyeball-verified; this
locks the source-of-truth logic.
"""
from __future__ import annotations

from apeGmsh.viewers.core.filter_controller import FilterController


def _recorder():
    seen: list[frozenset] = []
    return seen, seen.append


def test_defaults_to_all_dims_active() -> None:
    fc = FilterController([0, 1, 2, 3])
    assert fc.dims == [0, 1, 2, 3]
    assert fc.active == frozenset({0, 1, 2, 3})


def test_initial_subset_honored_and_filtered() -> None:
    fc = FilterController([1, 2, 3], initial=[2, 3, 9])  # 9 dropped
    assert fc.active == frozenset({2, 3})


def test_toggle_is_multiselect() -> None:
    seen, sink = _recorder()
    fc = FilterController([0, 1, 2, 3], initial=[], on_change=sink)
    fc.toggle(2)
    assert fc.active == frozenset({2})
    fc.toggle(0)
    assert fc.active == frozenset({0, 2})   # ADDS, not replaces
    fc.toggle(2)
    assert fc.active == frozenset({0})      # toggles back off
    assert seen == [frozenset({2}), frozenset({0, 2}), frozenset({0})]


def test_set_active_replaces_wholesale() -> None:
    fc = FilterController([0, 1, 2, 3])
    fc.set_active([1])
    assert fc.active == frozenset({1})
    fc.set_active([2, 3])
    assert fc.active == frozenset({2, 3})


def test_select_all_and_clear() -> None:
    fc = FilterController([1, 2, 3], initial=[])
    fc.select_all()
    assert fc.active == frozenset({1, 2, 3})
    fc.clear()
    assert fc.active == frozenset()


def test_on_change_fires_only_on_real_change() -> None:
    seen, sink = _recorder()
    fc = FilterController([0, 1, 2, 3], initial=[1], on_change=sink)
    fc.set_active([1])          # unchanged -> no emit
    fc.toggle(9)                # out of range -> no emit
    assert seen == []
    fc.toggle(1)                # 1 -> {} : real change
    assert seen == [frozenset()]


def test_out_of_range_dim_is_noop() -> None:
    fc = FilterController([1, 2, 3])
    fc.toggle(0)                # no dim-0 substrate -> harmless no-op
    assert fc.active == frozenset({1, 2, 3})
    fc.set_active([0, 5])       # all dropped
    assert fc.active == frozenset()


def test_emit_false_suppresses_callback() -> None:
    seen, sink = _recorder()
    fc = FilterController([0, 1, 2, 3], on_change=sink)
    fc.set_active([1], emit=False)
    fc.toggle(2, emit=False)
    assert seen == []
    assert fc.active == frozenset({1, 2})


def test_is_active() -> None:
    fc = FilterController([0, 1, 2, 3], initial=[1, 3])
    assert fc.is_active(1) and fc.is_active(3)
    assert not fc.is_active(0) and not fc.is_active(2)
