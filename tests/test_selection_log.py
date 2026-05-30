"""S3a acceptance — the SelectionLog serialization engine.

Fully headless (pure Python). Locks the replay/undo/redo/truncation
contract: the working set is a pure function of the active op prefix, so
undo/redo are cursor moves + replay. Uses plain DimTag-style tuples as
the generic target identity.
"""
from __future__ import annotations

from apeGmsh.viewers.core.selection_log import OpKind, SelectionOp, SelectionLog

A, B, C = (2, 1), (2, 2), (3, 5)


def _add(*ts):
    return SelectionOp(OpKind.ADD, tuple(ts))


def test_empty_log() -> None:
    log = SelectionLog()
    assert log.replay() == []
    assert not log.can_undo() and not log.can_redo()


def test_add_then_undo_redo() -> None:
    log = SelectionLog()
    log.record(_add(A))
    log.record(_add(B))
    assert log.replay() == [A, B]
    assert log.undo() and log.replay() == [A]
    assert log.undo() and log.replay() == []
    assert not log.undo()                 # nothing left
    assert log.redo() and log.replay() == [A]
    assert log.redo() and log.replay() == [A, B]
    assert not log.redo()


def test_box_add_is_one_gesture() -> None:
    log = SelectionLog()
    log.record(SelectionOp(OpKind.BOX_ADD, (A, B, C)))
    assert log.replay() == [A, B, C]
    # One undo reverts the whole box, not one entity at a time.
    assert log.undo() and log.replay() == []


def test_new_gesture_truncates_redo_tail() -> None:
    log = SelectionLog()
    log.record(_add(A))
    log.record(_add(B))
    log.undo()                            # active = [A], redo tail = [add B]
    log.record(_add(C))                   # truncates the redo tail
    assert log.replay() == [A, C]
    assert not log.can_redo()             # B is gone from the future


def test_remove_and_box_remove() -> None:
    log = SelectionLog()
    log.record(SelectionOp(OpKind.BOX_ADD, (A, B, C)))
    log.record(SelectionOp(OpKind.REMOVE, (B,)))
    assert log.replay() == [A, C]
    log.record(SelectionOp(OpKind.BOX_REMOVE, (A, C)))
    assert log.replay() == []
    log.undo()                            # back to [A, C]
    assert log.replay() == [A, C]


def test_clear_is_undoable() -> None:
    log = SelectionLog()
    log.record(SelectionOp(OpKind.BOX_ADD, (A, B)))
    log.record(SelectionOp(OpKind.CLEAR))
    assert log.replay() == []
    assert log.undo() and log.replay() == [A, B]   # clear is a gesture


def test_set_replaces_wholesale_and_is_undoable() -> None:
    log = SelectionLog()
    log.record(SelectionOp(OpKind.BOX_ADD, (A, B)))
    log.record(SelectionOp(OpKind.SET, (C,)))
    assert log.replay() == [C]
    assert log.undo() and log.replay() == [A, B]


def test_dedup_and_insertion_order() -> None:
    log = SelectionLog()
    log.record(_add(B))
    log.record(_add(A))
    log.record(_add(B))                   # already present → no dup, order kept
    assert log.replay() == [B, A]


def test_baseline_and_reset() -> None:
    log = SelectionLog([A, B])
    assert log.replay() == [A, B]
    log.record(_add(C))
    assert log.replay() == [A, B, C]
    log.reset([C])                        # group-load: new floor, no undo across it
    assert log.replay() == [C]
    assert not log.can_undo()


def test_all_ops_vs_active_ops() -> None:
    log = SelectionLog()
    log.record(_add(A))
    log.record(_add(B))
    log.undo()
    assert [o.targets for o in log.active_ops] == [(A,)]
    assert [o.targets for o in log.all_ops] == [(A,), (B,)]   # full serialised record
