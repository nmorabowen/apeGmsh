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

from .selection_log import OpKind, SelectionOp, SelectionLog
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
        self._active_group: str | None = None
        # ADR 0045 S3c: staging is the single source of truth. Group
        # edits (create/commit/rename/delete) mutate ``_staged_groups``
        # in memory ONLY; gmsh is written exactly once, at
        # ``flush_to_gmsh`` (the single freeze boundary). Names that must
        # be removed from gmsh at flush (deletes + renamed-away originals)
        # are tombstoned here.
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

    def _sync(self) -> None:
        """Re-materialise ``_picks`` from the log (the single source of
        truth). Keeps the legacy ``picks`` reads O(1) without drift."""
        self._picks = self._log.replay()

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
        """Clear picks without affecting the active group's stored members."""
        if self._picks:
            self._log.record(SelectionOp(OpKind.CLEAR))
            self._sync()
            # Deactivate group so commit doesn't overwrite with empty
            self._active_group = None
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
        if replace:
            new: list[SelectionTarget] = []
            for t in targets:
                if t not in new:
                    new.append(t)
        else:
            new = list(self._picks)
            for t in targets:
                if t not in new:
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
        """Load existing user-facing physical groups into staging.

        ADR 0045 S3c: staging is authoritative, so every group the viewer
        shows must live in ``_staged_groups``. Call once at viewer init to
        pull pre-existing PGs in (skipping internal ``_label:`` PGs, which
        the viewer never manages as groups)."""
        from apeGmsh.core.Labels import is_label_pg
        for pg_dim, pg_tag in sorted(
            gmsh.model.getPhysicalGroups(), key=lambda x: x[1]
        ):
            try:
                name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                continue
            if not name or is_label_pg(name) or name in self._staged_groups:
                continue
            self._staged_groups[name] = _load_targets(name)
            if name not in self._group_order:
                self._group_order.append(name)

    def set_active_group(self, name: str | None) -> None:
        """Switch active group. Pure in-memory staging — no gmsh write
        (writes are deferred to :meth:`flush_to_gmsh`, ADR 0045 S3c).

        If *name* is the same as the current active group, reloads from
        staging without re-staging the working set.
        """
        # Reload same group -> just reload its staged members.
        if name is not None and name == self._active_group:
            if name in self._staged_groups:
                self._picks = list(self._staged_groups[name])
            else:
                self._picks = _load_targets(name)
            self._log.reset(self._picks)
            self._fire()
            return

        # Stage the outgoing group's working set in memory.
        if self._active_group is not None:
            self._staged_groups[self._active_group] = list(self._picks)

        self._active_group = name
        if name is not None and name not in self._group_order:
            self._group_order.append(name)
        if name is None:
            self._picks = []
        elif name in self._staged_groups:
            self._picks = list(self._staged_groups[name])
        else:
            self._picks = _load_targets(name)
        # Group load is the new undo floor — undo does not cross a switch.
        self._log.reset(self._picks)
        self._fire()

    def commit_active_group(self) -> None:
        """Stage the current active group's working set (in memory).

        ADR 0045 S3c: no gmsh write — the staged state is flushed at
        :meth:`flush_to_gmsh`. An empty active group stays staged as an
        empty list and is deleted from gmsh at flush."""
        if self._active_group is None:
            return
        self._staged_groups[self._active_group] = list(self._picks)

    def apply_group(self, name: str) -> None:
        """Stage current picks as group *name* (in memory).

        ADR 0045 S3c: no gmsh write here — call :meth:`flush_to_gmsh` to
        persist. ``model_viewer``'s apply path flushes right after."""
        self._staged_groups[name] = list(self._picks)
        if name not in self._group_order:
            self._group_order.append(name)
        self._pending_deletes.discard(name)

    def rename_group(self, old: str, new: str) -> None:
        """Rename a staged group. In-memory only; the old name is
        tombstoned so its gmsh PG is removed at the next flush (ADR 0045
        S3c)."""
        if old == new:
            return

        if old in self._staged_groups:
            self._staged_groups[new] = self._staged_groups.pop(old)

        if self._active_group == old:
            self._active_group = new

        try:
            idx = self._group_order.index(old)
            self._group_order[idx] = new
        except ValueError:
            if new not in self._group_order:
                self._group_order.append(new)

        # Tombstone the old name (drop gmsh PG at flush); the new name is
        # live again, so clear any stale tombstone on it.
        self._pending_deletes.add(old)
        self._pending_deletes.discard(new)

    def delete_group(self, name: str) -> None:
        """Remove a group from staging and tombstone it for flush
        (ADR 0045 S3c — the gmsh PG is dropped at :meth:`flush_to_gmsh`)."""
        self._staged_groups.pop(name, None)
        try:
            self._group_order.remove(name)
        except ValueError:
            pass
        self._pending_deletes.add(name)
        if self._active_group == name:
            self._active_group = None
            self._log.reset(())
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
        then every staged group is (re)written; an empty staged group is
        deleted. Returns the count of groups *written* (deletions not
        counted).
        """
        # Stage current active group's working set.
        if self._active_group is not None:
            self._staged_groups[self._active_group] = list(self._picks)
        # Remove tombstoned names, except any name reborn in staging (a
        # rename A->B->A, or delete-then-recreate, leaves the live group).
        for name in self._pending_deletes:
            if name not in self._staged_groups:
                _delete_group_by_name(name)
        self._pending_deletes.clear()
        n = 0
        for name, members in self._staged_groups.items():
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
