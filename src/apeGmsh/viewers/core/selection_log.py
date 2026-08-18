"""SelectionLog — the serialized selection-operation engine (ADR 0045 S3).

The serialization backbone for requirement 5: every selection gesture is
one ordered, replayable :class:`SelectionOp`; the working set is a *pure
function* of the active op prefix, so undo/redo are just cursor moves +
replay (no fragile per-op inverses). The log is **generic over the target
identity** — it stores whatever hashable, value-equal token the caller
uses (a ``DimTag`` tuple or a :class:`~apeGmsh.viewers.scene_ir.
SelectionTarget`).

Pure Python, no Qt/VTK/gmsh — fully headless-testable, which is the point
of moving the undo backbone here.

S3a implemented the pick-gesture vocabulary (``ADD`` / ``REMOVE`` /
``BOX_ADD`` / ``BOX_REMOVE`` / ``CLEAR`` / ``SET``). S3c-2 adds the
**group** vocabulary (``GROUP_ACTIVATE`` / ``GROUP_COMMIT`` /
``GROUP_STAGE`` / ``GROUP_RENAME`` / ``GROUP_DELETE``) so physical-group
staging is replayable too — undo/redo now cross a group switch. The
reducer reconstructs a full :class:`GroupSnapshot` (working set + active
group + staged groups + creation order + delete tombstones); ``replay()``
returns just the working set (the S3a contract, unchanged), while
``replay_state()`` exposes the whole snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable, Optional


class OpKind(Enum):
    """The selection-gesture vocabulary. One op per user gesture."""

    # -- working-set (pick) gestures (S3a) --
    ADD = "add"            # click-add one (or a small set of) targets
    REMOVE = "remove"      # click-remove
    BOX_ADD = "box_add"    # rubber-band add (one gesture, N targets)
    BOX_REMOVE = "box_remove"
    CLEAR = "clear"        # empty the working set
    SET = "set"            # replace the working set wholesale (replace-drag, tab-cycle)
    # -- physical-group gestures (S3c-2) --
    GROUP_ACTIVATE = "group_activate"  # switch active group (name=None deactivates)
    GROUP_COMMIT = "group_commit"      # snapshot working into the active group
    GROUP_STAGE = "group_stage"        # stage an explicit member set under a name
    GROUP_APPLY = "group_apply"        # stage working under a name (does not activate)
    GROUP_RENAME = "group_rename"      # rename a staged group (name->name2)
    GROUP_DELETE = "group_delete"      # remove a staged group (tombstoned for flush)


@dataclass(frozen=True)
class SelectionOp:
    """One selection gesture: a kind plus its payload.

    Frozen + value-typed so the whole log serialises. ``targets`` carries
    the identity tokens for pick/stage gestures; ``name`` / ``name2``
    carry group names for the group gestures."""

    kind: OpKind
    targets: tuple = ()
    name: Optional[str] = None
    name2: Optional[str] = None


class GroupSnapshot:
    """The full selection state at a point in the op stream.

    Recomputed fresh on every replay, so it is treated as a value: the
    reducer copies the baseline, applies ops, and returns the result.
    """

    __slots__ = ("working", "active", "staged", "order", "pending")

    def __init__(
        self,
        working: Iterable[Hashable] = (),
        active: Optional[str] = None,
        staged: Optional[dict] = None,
        order: Iterable[str] = (),
        pending: Iterable[str] = (),
    ) -> None:
        self.working: list = list(working)
        self.active: Optional[str] = active
        self.staged: dict = {k: list(v) for k, v in (staged or {}).items()}
        self.order: list = list(order)
        self.pending: set = set(pending)

    def copy(self) -> "GroupSnapshot":
        return GroupSnapshot(
            self.working, self.active, self.staged, self.order, self.pending,
        )


def _dedup_extend(picks: list, targets) -> None:
    """Append ``targets`` to ``picks``, first occurrence wins.

    Set-backed rather than a list scan: the reducer runs on every
    gesture, and one ADR 0098 §8 "select all" op carries a whole
    physical group, so ``t not in picks`` would make a single replay
    quadratic in the set size. Tokens are required to be hashable —
    the log's stated contract, and what ``box_add`` already assumes.
    """
    seen = set(picks)
    for t in targets:
        if t not in seen:
            seen.add(t)
            picks.append(t)


def _apply(st: GroupSnapshot, op: SelectionOp) -> None:
    """Apply one op to the snapshot in place (the pure reducer step)."""
    k = op.kind
    # -- working-set gestures --
    if k in (OpKind.ADD, OpKind.BOX_ADD):
        _dedup_extend(st.working, op.targets)
    elif k in (OpKind.REMOVE, OpKind.BOX_REMOVE):
        # Same result as removing each target's first occurrence in
        # turn (the working set is deduped by construction), without
        # the O(n^2) scan-and-shift a big box-remove would pay.
        drop = set(op.targets)
        st.working = [t for t in st.working if t not in drop]
    elif k is OpKind.CLEAR:
        st.working = []
    elif k is OpKind.SET:
        new: list = []
        _dedup_extend(new, op.targets)
        st.working = new
    # -- group gestures --
    elif k is OpKind.GROUP_ACTIVATE:
        # Stage the outgoing group's working set, then load the incoming.
        if st.active is not None:
            st.staged[st.active] = list(st.working)
        name = op.name
        if name is not None:
            if name not in st.order:
                st.order.append(name)
            st.pending.discard(name)         # reactivating revives the name
        st.active = name
        # Staging is authoritative: a name not in staging loads empty
        # (a fresh / tombstoned group), never from a stale gmsh PG.
        st.working = list(st.staged.get(name, [])) if name is not None else []
    elif k is OpKind.GROUP_COMMIT:
        if st.active is not None:
            st.staged[st.active] = list(st.working)
    elif k is OpKind.GROUP_STAGE:
        # Stage an explicit member set under a name (the new-group flow).
        st.staged[op.name] = list(op.targets)
        if op.name not in st.order:
            st.order.append(op.name)
        st.pending.discard(op.name)
    elif k is OpKind.GROUP_APPLY:
        st.staged[op.name] = list(st.working)
        if op.name not in st.order:
            st.order.append(op.name)
        st.pending.discard(op.name)
    elif k is OpKind.GROUP_RENAME:
        old, new = op.name, op.name2
        # Reject a rename that would clobber a DISTINCT live group — the
        # caller (UI) should prevent collisions; the reducer never
        # silently merges or loses members.
        if new != old and new in st.staged:
            return
        if old in st.staged:
            st.staged[new] = st.staged.pop(old)
        if st.active == old:
            st.active = new
        # Update creation order in place (preserve position), without
        # ever duplicating a name.
        if old in st.order:
            if new in st.order:
                st.order.remove(old)
            else:
                st.order[st.order.index(old)] = new
        elif new not in st.order:
            st.order.append(new)
        st.pending.add(old)
        st.pending.discard(new)
    elif k is OpKind.GROUP_DELETE:
        st.staged.pop(op.name, None)
        if op.name in st.order:
            st.order.remove(op.name)
        st.pending.add(op.name)
        if st.active == op.name:
            st.active = None
            st.working = []


def _reduce_state(
    baseline: GroupSnapshot, ops: Iterable[SelectionOp],
) -> GroupSnapshot:
    """The pure reducer: baseline snapshot + ops → resulting snapshot."""
    st = baseline.copy()
    for op in ops:
        _apply(st, op)
    # Materialize the active group's LIVE working set into staging, so
    # readers (browser/outline/flush) always see its current members
    # without a per-pick commit gesture (S3c-2). An inactive working set
    # (active is None) is loose picks, staged under no name.
    if st.active is not None:
        st.staged[st.active] = list(st.working)
    return st


def _as_baseline(baseline) -> GroupSnapshot:
    """Coerce a baseline argument to a GroupSnapshot. Accepts a snapshot
    (full state) or a plain iterable of targets (the S3a working-set
    baseline)."""
    if isinstance(baseline, GroupSnapshot):
        return baseline
    return GroupSnapshot(working=baseline)


class SelectionLog:
    """Ordered, append-only op log with an undo/redo cursor.

    The cursor splits the op list into the *active* prefix (``ops[:cursor]``,
    which defines the current state) and a *redo tail* (``ops[cursor:]``).
    Recording a new gesture truncates the redo tail — the standard "new
    action after undo discards the redo future" rule.

    A ``baseline`` snapshot is the replay starting point. ``reset`` swaps
    the baseline (e.g. to seed pre-existing groups) and drops all ops.
    """

    __slots__ = ("_baseline", "_ops", "_cursor")

    def __init__(self, baseline=()) -> None:
        self._baseline: GroupSnapshot = _as_baseline(baseline)
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

    def replay_state(self) -> GroupSnapshot:
        """The current full snapshot = baseline + active ops."""
        return _reduce_state(self._baseline, self._ops[: self._cursor])

    def replay(self) -> list:
        """The current working set (the S3a contract — just the picks)."""
        return self.replay_state().working

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

    def reset(self, baseline=()) -> None:
        """Drop all ops and set a new baseline (snapshot or pick list).

        Undo cannot cross a reset — the new baseline is the floor."""
        self._baseline = _as_baseline(baseline)
        self._ops = []
        self._cursor = 0


__all__ = ["OpKind", "SelectionOp", "SelectionLog", "GroupSnapshot"]
