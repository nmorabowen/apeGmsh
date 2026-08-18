"""
SelectionState — Pure pick state management.

Manages the working set of picked entities, undo history, tab-cycling,
and physical group staging.  No VTK dependency — fires callbacks so
the viewer can wire colors and UI updates.

Usage::

    sel = SelectionState()
    sel.on_changed.append(lambda: print("picks changed"))
    sel.toggle((2, 5))   # pick surface 5
    sel.toggle((2, 5))   # unpick surface 5
    sel.undo()            # re-pick surface 5
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import gmsh
import numpy as np

from .selection_log import OpKind, SelectionOp, SelectionLog, GroupSnapshot
from ..scene_ir import SelectionTarget

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from .entity_registry import EntityRegistry


def _as_target(x: "DimTag | SelectionTarget") -> SelectionTarget:
    """Normalise a caller token to a ``SelectionTarget``.

    Accepts an already-built target or a legacy BREP ``(dim, tag)``
    DimTag (wrapped as ``MODEL_BREP``). The single front door that lets
    ``SelectionState`` hold targets while existing callers still pass
    DimTags (ADR 0045 keystone)."""
    if isinstance(x, SelectionTarget):
        return x
    return SelectionTarget.from_dimtag(x)

_log = logging.getLogger("apeGmsh.viewer.selection")


# ======================================================================
# Gmsh physical-group I/O helpers
# ======================================================================

def _load_group_members(name: str) -> list["DimTag"]:
    """Load existing physical group members from Gmsh."""
    members: list["DimTag"] = []
    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        except Exception:
            _log.debug("getPhysicalName failed for (%d, %d)", pg_dim, pg_tag)
            continue
        if pg_name == name:
            for etag in gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag):
                members.append((pg_dim, int(etag)))
    return members


def _delete_group_by_name(name: str) -> None:
    """Remove all physical groups with *name* from Gmsh."""
    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            if gmsh.model.getPhysicalName(pg_dim, pg_tag) == name:
                gmsh.model.removePhysicalGroups([(pg_dim, pg_tag)])
        except Exception:
            _log.debug(
                "removePhysicalGroups failed for %r (%d, %d)",
                name, pg_dim, pg_tag,
            )


def _load_targets(name: str) -> list[SelectionTarget]:
    """Load existing physical group members as ``MODEL_BREP`` targets."""
    return [_as_target(dt) for dt in _load_group_members(name)]


def _write_group(
    name: str, members: list["SelectionTarget | DimTag"],
) -> None:
    """Write a physical group to Gmsh (replaces existing with same name).

    A physical-group name maps to a single dimension.  Members
    spanning more than one dimension are rejected — multi-dimensional
    physical groups are not supported.

    Accepts ``SelectionTarget`` (the ``SelectionState`` internal form)
    or a bare ``(dim, tag)`` DimTag (the direct-call/test contract).
    """
    by_dim: dict[int, list[int]] = {}
    for m in members:
        dim, tag = _as_target(m).dimtag
        by_dim.setdefault(dim, []).append(tag)
    if len(by_dim) > 1:
        raise ValueError(
            f"Physical group {name!r} would span dimensions "
            f"{sorted(by_dim)}.  Pick entities of a single dimension "
            f"per group — multi-dimensional physical groups are not "
            f"supported."
        )
    _delete_group_by_name(name)
    for dim, tags in by_dim.items():
        pg_tag = gmsh.model.addPhysicalGroup(dim, tags)
        gmsh.model.setPhysicalName(dim, pg_tag, name)


# ======================================================================
# SelectionState
# ======================================================================

class SelectionState:
    """Working set of picked entities + physical group staging."""

    __slots__ = (
        "_picks",
        "_log",
        "_active_group",
        "_staged_groups",
        "_group_order",
        "_pending_deletes",
        "_tab_candidates",
        "_tab_index",
        "on_changed",
    )

    def __init__(self) -> None:
        # ADR 0045 keystone: ``_picks`` holds ``SelectionTarget`` (the
        # unified, substrate-tagged identity), not bare DimTags. Callers
        # may still pass DimTags — they are normalised to ``MODEL_BREP``
        # targets at the front door (``_as_target``).
        self._picks: list[SelectionTarget] = []
        # ADR 0045 S3a: the serialized op-log replaces the flat
        # per-entity ``_history`` LIFO. ``_picks`` is always == the
        # log's replay of its active op prefix.
        self._log: SelectionLog = SelectionLog()
        # ADR 0045 S3c-2: the group fields below are CACHES of the log's
        # snapshot (``replay_state()``), refreshed by ``_sync`` after every
        # gesture — the log is the single source of truth. Group edits
        # (activate/apply/stage/rename/delete) record replayable ops, so
        # undo/redo cross a group switch. gmsh is written exactly once, at
        # ``flush_to_gmsh`` (the freeze boundary); deleted / renamed-away
        # names are tombstoned in the snapshot's ``pending`` set.
        self._active_group: str | None = None
        self._staged_groups: dict[str, list[SelectionTarget]] = {}
        self._group_order: list[str] = []  # creation order
        self._pending_deletes: set[str] = set()
        self._tab_candidates: list[SelectionTarget] = []
        self._tab_index: int = 0
        self.on_changed: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Pick operations
    # ------------------------------------------------------------------

    @property
    def picks(self) -> list["DimTag"]:
        """BREP-compat view: the picked targets as gmsh DimTags.

        Shim for one release (ADR 0045 keystone). New, substrate-aware
        consumers should read :attr:`targets`; this raises if a non-BREP
        target is present (only BREP targets have a DimTag)."""
        return [t.dimtag for t in self._picks]

    @property
    def targets(self) -> list[SelectionTarget]:
        """The picked entities as unified ``SelectionTarget`` values."""
        return list(self._picks)

    def __len__(self) -> int:
        """How many entities are picked, without copying them.

        ``targets`` hands out a defensive copy, which is right for a
        reader and wrong for a count: ADR 0098's outline re-reads the
        size on every session tick, and after a "select all nodes" on a
        100k-node group that copy is the whole set, per tick, for a
        label."""
        return len(self._picks)

    def _sync(self) -> None:
        """Re-materialise the cached state from the log — the single
        source of truth (ADR 0045 S3c-2). ``_picks`` and the group fields
        (``_active_group`` / ``_staged_groups`` / ``_group_order`` /
        ``_pending_deletes``) are all caches of ``replay_state()`` kept
        O(1) for reads, refreshed after every gesture."""
        st = self._log.replay_state()
        self._picks = st.working
        self._active_group = st.active
        self._staged_groups = st.staged
        self._group_order = st.order
        self._pending_deletes = st.pending

    def pick(self, dt: "DimTag | SelectionTarget") -> None:
        t = _as_target(dt)
        if t not in self._picks:
            self._log.record(SelectionOp(OpKind.ADD, (t,)))
            self._sync()
            self._fire()

    def unpick(self, dt: "DimTag | SelectionTarget") -> None:
        t = _as_target(dt)
        if t in self._picks:
            self._log.record(SelectionOp(OpKind.REMOVE, (t,)))
            self._sync()
            self._fire()

    def toggle(self, dt: "DimTag | SelectionTarget") -> None:
        t = _as_target(dt)
        if t in self._picks:
            self.unpick(t)
        else:
            self.pick(t)

    def clear(self) -> None:
        """Clear the working selection — a replayable gesture (ADR 0045
        S3c-2; no direct cache mutation).

        With an active group, deactivating it (``GROUP_ACTIVATE`` None)
        stages its current members and empties the working set, so the
        group's stored members survive. With no active group, just empty
        the loose picks."""
        if self._active_group is not None:
            self._log.record(SelectionOp(OpKind.GROUP_ACTIVATE, name=None))
            self._sync()
            self._fire()
        elif self._picks:
            self._log.record(SelectionOp(OpKind.CLEAR))
            self._sync()
            self._fire()

    def undo(self) -> bool:
        """Undo the most recent gesture (whole gesture, not per-entity).

        Returns whether anything was undone. (Legacy callers ignored the
        old per-entity return value.)"""
        if not self._log.undo():
            return False
        self._sync()
        self._fire()
        return True

    def redo(self) -> bool:
        """Re-apply the next undone gesture. Returns whether anything was
        redone (ADR 0045 S3a — redo is new; the old flat history had none)."""
        if not self._log.redo():
            return False
        self._sync()
        self._fire()
        return True

    def select_batch(
        self, dts: list["DimTag | SelectionTarget"], *, replace: bool = False,
    ) -> None:
        targets = [_as_target(d) for d in dts]
        # Compute the prospective result and only record a gesture when
        # it actually changes the working set — otherwise a no-op batch
        # (e.g. re-selecting already-picked entities by double-clicking a
        # tree/part item) would leave a dead undo step behind.
        #
        # The dedup is set-backed, not a list scan. Order is preserved
        # (first occurrence wins) because the working set IS an ordered
        # record — tab-cycling and the highlight both read it in order
        # — but ``t not in new`` over a growing list is O(n^2), and
        # ADR 0098 §8's outline "select all" is the first writer that
        # hands this method a whole physical group at once. Measured
        # before the fix: 1k targets 0.10 s, 4k 1.66 s, 16k 27 s.
        if replace:
            new: list[SelectionTarget] = list(dict.fromkeys(targets))
        else:
            new = list(self._picks)
            seen = set(new)
            for t in targets:
                if t not in seen:
                    seen.add(t)
                    new.append(t)
        if new == self._picks:
            return
        kind = OpKind.SET if replace else OpKind.ADD
        self._log.record(SelectionOp(kind, tuple(targets)))
        self._sync()
        self._fire()

    def box_add(self, dts: list["DimTag | SelectionTarget"]) -> int:
        """Add entities from box-select. Returns count added.

        Counts the *distinct* new entities (== the state delta), so a
        duplicate-bearing input list does not over-count."""
        targets = [_as_target(d) for d in dts]
        old = set(self._picks)
        added = len(set(targets) - old)
        if added:
            self._log.record(SelectionOp(OpKind.BOX_ADD, tuple(targets)))
            self._sync()
            self._fire()
        return added

    def box_remove(self, dts: list["DimTag | SelectionTarget"]) -> int:
        """Remove entities from Ctrl+box-select. Returns count removed."""
        targets = [_as_target(d) for d in dts]
        old = set(self._picks)
        removed = len(set(targets) & old)
        if removed:
            self._log.record(SelectionOp(OpKind.BOX_REMOVE, tuple(targets)))
            self._sync()
            self._fire()
        return removed

    # ------------------------------------------------------------------
    # Tab cycling
    # ------------------------------------------------------------------

    def set_tab_candidates(
        self, candidates: list["DimTag | SelectionTarget"],
    ) -> None:
        self._tab_candidates = [_as_target(c) for c in candidates]
        self._tab_index = 0

    def cycle_tab(self) -> "DimTag | None":
        cands = self._tab_candidates
        if len(cands) < 2:
            return None
        new = list(self._picks)
        cur = cands[self._tab_index]
        if cur in new:
            new.remove(cur)
        self._tab_index = (self._tab_index + 1) % len(cands)
        nxt = cands[self._tab_index]
        if nxt not in new:
            new.append(nxt)
        # Tab index always advances (cycling state); record/fire only when
        # the working set actually changed — no phantom undo step.
        if new != self._picks:
            # One undoable gesture: the cycle's resulting set.
            self._log.record(SelectionOp(OpKind.SET, tuple(new)))
            self._sync()
            self._fire()
        return nxt.dimtag

    # ------------------------------------------------------------------
    # Physical group management
    # ------------------------------------------------------------------

    @property
    def active_group(self) -> str | None:
        return self._active_group

    @property
    def staged_groups(self) -> dict[str, list[SelectionTarget]]:
        return dict(self._staged_groups)

    @property
    def group_order(self) -> list[str]:
        """Group names in creation order."""
        return list(self._group_order)

    def seed_from_gmsh(self) -> None:
        """Load existing user-facing physical groups into staging as the
        log BASELINE (ADR 0045 S3c-2). Staging is authoritative, so the
        seeded snapshot is the replay floor; later group gestures replay
        on top of it (undo cannot cross the seed). Skips internal
        ``_label:`` PGs, which the viewer never manages as groups."""
        from apeGmsh.core.Labels import is_label_pg
        staged: dict[str, list[SelectionTarget]] = {}
        order: list[str] = []
        for pg_dim, pg_tag in sorted(
            gmsh.model.getPhysicalGroups(), key=lambda x: x[1]
        ):
            try:
                name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                continue
            if not name or is_label_pg(name) or name in staged:
                continue
            staged[name] = _load_targets(name)
            order.append(name)
        self._log.reset(GroupSnapshot(staged=staged, order=order))
        self._sync()
        self._fire()

    def set_active_group(self, name: str | None) -> None:
        """Switch active group — records a replayable ``GROUP_ACTIVATE``
        gesture so undo crosses the switch (ADR 0045 S3c-2). No gmsh
        write; staging is flushed at :meth:`flush_to_gmsh`."""
        if name == self._active_group:
            return  # already active — no-op (state stays log-consistent)
        self._log.record(SelectionOp(OpKind.GROUP_ACTIVATE, name=name))
        self._sync()
        self._fire()

    def commit_active_group(self) -> None:
        """No-op, kept for API compatibility (ADR 0045 S3c-2).

        The reducer auto-materialises the active group's working set into
        staging on every replay, so an explicit commit gesture is no
        longer needed — the active group's members are always live."""
        return

    def apply_group(self, name: str) -> None:
        """Stage the current working set under *name* without activating
        — records a replayable ``GROUP_APPLY`` gesture. No gmsh write;
        ``model_viewer``'s apply path flushes right after."""
        self._log.record(SelectionOp(OpKind.GROUP_APPLY, name=name))
        self._sync()
        self._fire()

    def stage_group(self, name: str, members: list) -> None:
        """Stage an explicit member set under *name* (the new-group flow)
        — records a replayable ``GROUP_STAGE`` gesture. No gmsh write."""
        targets = tuple(_as_target(m) for m in members)
        self._log.record(
            SelectionOp(OpKind.GROUP_STAGE, targets=targets, name=name)
        )
        self._sync()
        self._fire()

    def rename_group(self, old: str, new: str) -> None:
        """Rename a staged group — records a replayable ``GROUP_RENAME``
        gesture (old name tombstoned for flush). No gmsh write."""
        if old == new:
            return
        self._log.record(
            SelectionOp(OpKind.GROUP_RENAME, name=old, name2=new)
        )
        self._sync()
        self._fire()

    def delete_group(self, name: str) -> None:
        """Remove a group from staging and tombstone it for flush —
        records a replayable ``GROUP_DELETE`` gesture (undoable; the gmsh
        PG is dropped at :meth:`flush_to_gmsh`)."""
        self._log.record(SelectionOp(OpKind.GROUP_DELETE, name=name))
        self._sync()
        self._fire()

    def group_exists(self, name: str) -> bool:
        # Staging is authoritative (ADR 0045 S3c): a tombstoned name no
        # longer staged is gone, even if its gmsh PG lingers until flush.
        if name in self._pending_deletes and name not in self._staged_groups:
            return False
        if name in self._staged_groups:
            return True
        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == name:
                    return True
            except Exception:
                pass
        return False

    def flush_to_gmsh(self) -> int:
        """Reconcile the staged group state into Gmsh — the single freeze
        boundary (ADR 0045 S3c). This is the ONLY method that writes PGs.

        Tombstoned names (deleted / renamed-away originals) are removed,
        then each staged group is (re)written if it has members, or
        deleted from gmsh if empty. Empty staged groups are *kept* in
        staging (an active group mid-build is legitimately empty); they
        simply have no gmsh PG. Returns the count of groups *written*
        (deletions not counted).
        """
        # Read the authoritative snapshot from the log (the active
        # group's live working set is already materialised into staging).
        st = self._log.replay_state()
        # Remove tombstoned names, except any name reborn in staging (a
        # rename A->B->A, or delete-then-recreate, leaves the live group).
        for name in st.pending:
            if name not in st.staged:
                _delete_group_by_name(name)
        n = 0
        for name, members in st.staged.items():
            if members:
                _write_group(name, members)
                n += 1
            else:
                _delete_group_by_name(name)
        return n

    # ------------------------------------------------------------------
    # Centroid for orbit pivot
    # ------------------------------------------------------------------

    def centroid(self, registry: "EntityRegistry") -> tuple | None:
        """Compute centroid of current picks (for orbit pivot)."""
        if not self._picks:
            return None
        pts = []
        for t in self._picks:
            c = registry.centroid(t.dimtag)
            if c is not None:
                pts.append(c)
        if not pts:
            return None
        arr = np.mean(pts, axis=0)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self) -> None:
        for cb in self.on_changed:
            try:
                cb()
            except Exception:
                _log.exception("on_changed callback failed: %r", cb)

    def __repr__(self) -> str:
        return (
            f"<SelectionState {len(self._picks)} picks, "
            f"group={self._active_group!r}>"
        )
