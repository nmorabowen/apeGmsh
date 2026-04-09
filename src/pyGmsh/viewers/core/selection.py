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

from typing import TYPE_CHECKING, Any, Callable

import gmsh
import numpy as np

if TYPE_CHECKING:
    from .entity_registry import DimTag, EntityRegistry


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
            pass


def _write_group(name: str, members: list["DimTag"]) -> None:
    """Write a physical group to Gmsh (replaces existing with same name)."""
    _delete_group_by_name(name)
    by_dim: dict[int, list[int]] = {}
    for dim, tag in members:
        by_dim.setdefault(dim, []).append(tag)
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
        "_history",
        "_active_group",
        "_staged_groups",
        "_tab_candidates",
        "_tab_index",
        "on_changed",
    )

    def __init__(self) -> None:
        self._picks: list["DimTag"] = []
        self._history: list["DimTag"] = []
        self._active_group: str | None = None
        self._staged_groups: dict[str, list["DimTag"]] = {}
        self._group_order: list[str] = []  # creation order
        self._tab_candidates: list["DimTag"] = []
        self._tab_index: int = 0
        self.on_changed: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Pick operations
    # ------------------------------------------------------------------

    @property
    def picks(self) -> list["DimTag"]:
        return list(self._picks)

    def pick(self, dt: "DimTag") -> None:
        if dt not in self._picks:
            self._picks.append(dt)
            self._history.append(dt)
            self._fire()

    def unpick(self, dt: "DimTag") -> None:
        if dt in self._picks:
            self._picks.remove(dt)
            self._history = [d for d in self._history if d != dt]
            self._fire()

    def toggle(self, dt: "DimTag") -> None:
        if dt in self._picks:
            self.unpick(dt)
        else:
            self.pick(dt)

    def clear(self) -> None:
        """Clear picks without affecting the active group's stored members."""
        if self._picks:
            self._picks.clear()
            self._history.clear()
            # Deactivate group so commit doesn't overwrite with empty
            self._active_group = None
            self._fire()

    def undo(self) -> "DimTag | None":
        if not self._history:
            return None
        dt = self._history.pop()
        if dt in self._picks:
            self._picks.remove(dt)
        self._fire()
        return dt

    def select_batch(
        self, dts: list["DimTag"], *, replace: bool = False,
    ) -> None:
        if replace:
            self._picks.clear()
            self._history.clear()
        for dt in dts:
            if dt not in self._picks:
                self._picks.append(dt)
                self._history.append(dt)
        self._fire()

    def box_add(self, dts: list["DimTag"]) -> int:
        """Add entities from box-select. Returns count added."""
        added = 0
        for dt in dts:
            if dt not in self._picks:
                self._picks.append(dt)
                self._history.append(dt)
                added += 1
        if added:
            self._fire()
        return added

    def box_remove(self, dts: list["DimTag"]) -> int:
        """Remove entities from Ctrl+box-select. Returns count removed."""
        removed = 0
        for dt in dts:
            if dt in self._picks:
                self._picks.remove(dt)
                self._history = [d for d in self._history if d != dt]
                removed += 1
        if removed:
            self._fire()
        return removed

    # ------------------------------------------------------------------
    # Tab cycling
    # ------------------------------------------------------------------

    def set_tab_candidates(self, candidates: list["DimTag"]) -> None:
        self._tab_candidates = list(candidates)
        self._tab_index = 0

    def cycle_tab(self) -> "DimTag | None":
        cands = self._tab_candidates
        if len(cands) < 2:
            return None
        cur = cands[self._tab_index]
        if cur in self._picks:
            self._picks.remove(cur)
            self._history = [d for d in self._history if d != cur]
        self._tab_index = (self._tab_index + 1) % len(cands)
        nxt = cands[self._tab_index]
        if nxt not in self._picks:
            self._picks.append(nxt)
            self._history.append(nxt)
        self._fire()
        return nxt

    # ------------------------------------------------------------------
    # Physical group management
    # ------------------------------------------------------------------

    @property
    def active_group(self) -> str | None:
        return self._active_group

    @property
    def staged_groups(self) -> dict[str, list["DimTag"]]:
        return dict(self._staged_groups)

    @property
    def group_order(self) -> list[str]:
        """Group names in creation order."""
        return list(self._group_order)

    def set_active_group(self, name: str | None) -> None:
        """Switch active group, writing the outgoing group to Gmsh.

        If *name* is the same as the current active group, reloads
        without writing (avoids overwriting with stale picks).
        """
        # Reload same group → don't write outgoing (would overwrite)
        if name is not None and name == self._active_group:
            # Reload from staged or Gmsh
            if name in self._staged_groups:
                self._picks = list(self._staged_groups[name])
            else:
                self._picks = _load_group_members(name)
            self._history = list(self._picks)
            self._fire()
            return

        # Write outgoing group to Gmsh immediately
        if self._active_group is not None:
            members = list(self._picks)
            self._staged_groups[self._active_group] = members
            if members:
                _write_group(self._active_group, members)

        self._active_group = name
        if name is not None and name not in self._group_order:
            self._group_order.append(name)
        if name is None:
            self._picks = []
        elif name in self._staged_groups:
            self._picks = list(self._staged_groups[name])
        else:
            self._picks = _load_group_members(name)
        self._history = list(self._picks)
        self._fire()

    def commit_active_group(self) -> None:
        """Write the current active group to Gmsh now."""
        if self._active_group is not None and self._picks:
            self._staged_groups[self._active_group] = list(self._picks)
            _write_group(self._active_group, self._picks)

    def apply_group(self, name: str) -> None:
        """Stage current picks as group *name* and write to Gmsh."""
        self._staged_groups[name] = list(self._picks)
        if self._picks:
            _write_group(name, self._picks)

    def rename_group(self, old: str, new: str) -> None:
        if old in self._staged_groups:
            self._staged_groups[new] = self._staged_groups.pop(old)
        if self._active_group == old:
            self._active_group = new

    def delete_group(self, name: str) -> None:
        self._staged_groups.pop(name, None)
        if self._active_group == name:
            self._active_group = None
            self._picks.clear()
            self._history.clear()
            self._fire()

    def group_exists(self, name: str) -> bool:
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
        """Write all staged groups to Gmsh. Returns count written."""
        # Stage current active group
        if self._active_group is not None:
            self._staged_groups[self._active_group] = list(self._picks)
        n = 0
        for name, members in self._staged_groups.items():
            if members:
                _write_group(name, members)
                n += 1
        return n

    # ------------------------------------------------------------------
    # Centroid for orbit pivot
    # ------------------------------------------------------------------

    def centroid(self, registry: "EntityRegistry") -> tuple | None:
        """Compute centroid of current picks (for orbit pivot)."""
        if not self._picks:
            return None
        pts = []
        for dt in self._picks:
            c = registry.centroid(dt)
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
                pass

    def __repr__(self) -> str:
        return (
            f"<SelectionState {len(self._picks)} picks, "
            f"group={self._active_group!r}>"
        )
