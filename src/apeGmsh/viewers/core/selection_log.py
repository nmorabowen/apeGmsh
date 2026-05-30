"""SelectionLog — the serialized selection-operation engine (ADR 0045 S3).

The serialization backbone for requirement 5: every selection gesture is
one ordered, replayable :class:`SelectionOp`; the working set is a *pure
function* of the active op prefix, so undo/redo are just cursor moves +
replay (no fragile per-op inverses). The log is **generic over the target
identity** — it stores whatever hashable, value-equal token the caller
uses (today a ``DimTag`` tuple; a :class:`~apeGmsh.viewers.scene_ir.
SelectionTarget` once that migration lands) — so this engine never has to
change when the identity type does.

Pure Python, no Qt/VTK/gmsh — fully headless-testable, which is the point
of moving the undo backbone here.

S3a implements the gesture vocabulary the viewer pick paths use today
(``ADD`` / ``REMOVE`` / ``BOX_ADD`` / ``BOX_REMOVE`` / ``CLEAR`` /
``SET``). The contract's ``GROUP_ACTIVATE`` / ``GROUP_COMMIT`` (S3c) and
``MODE_SET`` (when the FilterController feeds the log) join the enum +
reducer when their slices wire them; they are intentionally absent here
rather than defined-but-unimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable, Sequence


class OpKind(Enum):
    """The selection-gesture vocabulary. One op per user gesture."""

    ADD = "add"            # click-add one (or a small set of) targets
    REMOVE = "remove"      # click-remove
    BOX_ADD = "box_add"    # rubber-band add (one gesture, N targets)
    BOX_REMOVE = "box_remove"
    CLEAR = "clear"        # empty the working set
    SET = "set"            # replace the working set wholesale (replace-drag, tab-cycle)


@dataclass(frozen=True)
class SelectionOp:
    """One selection gesture: a kind + the targets it acted on.

    Frozen + value-typed so the whole log serialises. ``targets`` is a
    tuple of the caller's identity tokens (``DimTag`` today)."""

    kind: OpKind
    targets: tuple = ()


def _reduce(baseline: Sequence[Hashable], ops: Iterable[SelectionOp]) -> list:
    """The pure reducer: baseline + ops → ordered, de-duplicated picks.

    Order is insertion order (mirrors the legacy ``_picks`` list);
    membership is by value equality."""
    picks: list = list(baseline)
    for op in ops:
        k = op.kind
        if k in (OpKind.ADD, OpKind.BOX_ADD):
            for t in op.targets:
                if t not in picks:
                    picks.append(t)
        elif k in (OpKind.REMOVE, OpKind.BOX_REMOVE):
            for t in op.targets:
                if t in picks:
                    picks.remove(t)
        elif k is OpKind.CLEAR:
            picks = []
        elif k is OpKind.SET:
            picks = []
            for t in op.targets:
                if t not in picks:
                    picks.append(t)
    return picks


class SelectionLog:
    """Ordered, append-only op log with an undo/redo cursor.

    The cursor splits the op list into the *active* prefix (``ops[:cursor]``,
    which defines the current working set) and a *redo tail*
    (``ops[cursor:]``). Recording a new gesture truncates the redo tail —
    the standard "new action after undo discards the redo future" rule.

    A ``baseline`` set is the replay starting point. Group activation
    (S3c) resets the baseline to the loaded members, so undo does not
    step back across a group switch (gesture-scoped undo).
    """

    __slots__ = ("_baseline", "_ops", "_cursor")

    def __init__(self, baseline: Iterable[Hashable] = ()) -> None:
        self._baseline: tuple = tuple(baseline)
        self._ops: list[SelectionOp] = []
        self._cursor: int = 0

    # -- introspection -------------------------------------------------

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def active_ops(self) -> list[SelectionOp]:
        """The ops currently in effect (``ops[:cursor]``)."""
        return list(self._ops[: self._cursor])

    @property
    def all_ops(self) -> list[SelectionOp]:
        """Every recorded op, including the redo tail — the serialised
        whole-session record."""
        return list(self._ops)

    def replay(self) -> list:
        """The current working set = baseline + active ops."""
        return _reduce(self._baseline, self._ops[: self._cursor])

    # -- mutation ------------------------------------------------------

    def record(self, op: SelectionOp) -> None:
        """Append a gesture, truncating any redo tail."""
        del self._ops[self._cursor:]
        self._ops.append(op)
        self._cursor += 1

    def can_undo(self) -> bool:
        return self._cursor > 0

    def can_redo(self) -> bool:
        return self._cursor < len(self._ops)

    def undo(self) -> bool:
        """Step the cursor back one gesture. Returns False if nothing to undo."""
        if self._cursor == 0:
            return False
        self._cursor -= 1
        return True

    def redo(self) -> bool:
        """Step the cursor forward one gesture. Returns False if nothing to redo."""
        if self._cursor >= len(self._ops):
            return False
        self._cursor += 1
        return True

    def reset(self, baseline: Iterable[Hashable] = ()) -> None:
        """Drop all ops and set a new baseline (e.g. on group load).

        Undo cannot cross a reset — the new baseline is the floor."""
        self._baseline = tuple(baseline)
        self._ops = []
        self._cursor = 0


__all__ = ["OpKind", "SelectionOp", "SelectionLog"]
