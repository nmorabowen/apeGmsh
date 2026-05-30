"""S3a acceptance — SelectionState undo/redo on the op-log.

Headless: the pick / box / undo / redo / clear paths touch no gmsh
(only group methods do). Exercises the gesture-scoped undo + new redo +
the on_changed fire contract, end to end through SelectionState.
"""
from __future__ import annotations

from apeGmsh.viewers.core.selection import SelectionState

A, B, C = (2, 1), (2, 2), (3, 5)


def test_pick_undo_redo() -> None:
    sel = SelectionState()
    sel.pick(A)
    sel.pick(B)
    assert sel.picks == [A, B]
    assert sel.undo() and sel.picks == [A]
    assert sel.undo() and sel.picks == []
    assert sel.undo() is False               # nothing left
    assert sel.redo() and sel.picks == [A]
    assert sel.redo() and sel.picks == [A, B]
    assert sel.redo() is False


def test_box_add_is_one_undo_gesture() -> None:
    sel = SelectionState()
    n = sel.box_add([A, B, C])
    assert n == 3 and sel.picks == [A, B, C]
    # One undo reverts the whole box (the per-entity-granularity fix).
    assert sel.undo() and sel.picks == []
    assert sel.redo() and sel.picks == [A, B, C]


def test_box_add_returns_only_new_count() -> None:
    sel = SelectionState()
    sel.pick(A)
    assert sel.box_add([A, B]) == 1          # A already present
    assert sel.picks == [A, B]


def test_new_gesture_after_undo_truncates_redo() -> None:
    sel = SelectionState()
    sel.pick(A)
    sel.pick(B)
    sel.undo()                                # picks == [A], B in redo tail
    sel.pick(C)                               # truncates redo
    assert sel.picks == [A, C]
    assert sel.redo() is False


def test_clear_is_undoable() -> None:
    sel = SelectionState()
    sel.box_add([A, B])
    sel.clear()
    assert sel.picks == []
    assert sel.undo() and sel.picks == [A, B]


def test_select_batch_replace_is_undoable() -> None:
    sel = SelectionState()
    sel.box_add([A, B])
    sel.select_batch([C], replace=True)
    assert sel.picks == [C]
    assert sel.undo() and sel.picks == [A, B]


def test_box_add_count_dedups_duplicate_input() -> None:
    # Duplicate DimTags in the input must not over-count (review fix).
    sel = SelectionState()
    assert sel.box_add([A, A, B]) == 2     # two distinct entities added
    assert sel.picks == [A, B]
    sel2 = SelectionState()
    sel2.box_add([A, B])
    assert sel2.box_remove([A, A]) == 1     # one distinct entity removed


def test_noop_select_batch_leaves_no_dead_undo_step() -> None:
    # Re-selecting an already-picked entity records no gesture (review fix),
    # so a single undo reverts the original pick, not a phantom no-op.
    sel = SelectionState()
    sel.pick(A)
    sel.select_batch([A])                   # no change → no op recorded
    assert sel.picks == [A]
    assert sel.undo() and sel.picks == []   # the pick, not a phantom step
    assert sel.undo() is False


def test_on_changed_fires_per_gesture() -> None:
    sel = SelectionState()
    fires = []
    sel.on_changed.append(lambda: fires.append(len(sel.picks)))
    sel.pick(A)                # 1
    sel.box_add([B, C])        # 1 (one gesture)
    sel.undo()                 # 1
    assert fires == [1, 3, 1]
