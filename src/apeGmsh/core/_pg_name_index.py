"""Name -> ``{dim: tag}`` index over gmsh physical groups.

``g.physical`` and ``g.labels`` find a group by name.  gmsh has no
name -> tag query, so both used to list every group and read every
name, which made creating or resolving G named groups cost O(G^2).
Each composite keeps one :class:`PGNameIndex` over the names it owns:
user names for ``g.physical``, ``_label:`` names for ``g.labels``.

Staleness rule.  Raw ``gmsh.model.*`` calls (inside apeGmsh, such as the
boolean remap, or in user code) change groups without telling the
index, so the index is never trusted on its own.  Every lookup asks
gmsh at which dims the name exists (one ``getEntitiesForPhysicalName``,
a single C++ call) and confirms each indexed tag (one
``getPhysicalName`` per dim).  If gmsh disagrees, the index is rebuilt
by one full scan and the answer comes from the rebuilt index.  A stale
index therefore costs one rebuild, never a wrong answer.
"""
from __future__ import annotations

from collections.abc import Callable

import gmsh


def _kernel_dims(name: str) -> set[int]:
    """The dims at which gmsh holds a physical group named *name*.

    gmsh keeps no empty physical group, so the dims of the group's
    entities are the dims of the group.  ``getEntitiesForPhysicalName``
    raises for an unknown name and gmsh prints that as an error, so the
    terminal is muted for the probe (as ``Labels._is_bound`` does).
    """
    terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        return {int(d) for d, _ in gmsh.model.getEntitiesForPhysicalName(name)}
    except Exception:
        return set()
    finally:
        gmsh.option.setNumber("General.Terminal", terminal)


class PGNameIndex:
    """``name -> {dim: tag}`` for the physical-group names *owns* accepts.

    Read through :meth:`lookup`, which checks every answer against gmsh
    (see the module docstring).  :meth:`note` and :meth:`drop` keep the
    index warm after apeGmsh's own writes, so a lookup right after one
    needs no rebuild.
    """

    def __init__(self, owns: Callable[[str], bool]) -> None:
        self._owns = owns
        self._by_name: dict[str, dict[int, int]] = {}
        self._by_tag: dict[tuple[int, int], str] = {}

    def lookup(self, name: str) -> dict[int, int]:
        """``{dim: tag}`` of the groups named *name* (empty if none)."""
        if not self._owns(name):
            return {}
        hit = self._by_name.get(name, {})
        if set(hit) == _kernel_dims(name) and all(
            gmsh.model.getPhysicalName(d, t) == name for d, t in hit.items()
        ):
            return dict(hit)
        self.rebuild()
        return dict(self._by_name.get(name, {}))

    def rebuild(self) -> None:
        self.clear()
        for d, t in gmsh.model.getPhysicalGroups():
            pg_name = gmsh.model.getPhysicalName(d, t)
            if self._owns(pg_name):
                self._add(int(d), int(t), pg_name)

    def note(self, dim: int, tag: int) -> None:
        """Index group ``(dim, tag)`` under the name gmsh now holds for it.

        Read back rather than passed in: ``setPhysicalName`` keeps a
        name the group already has.
        """
        self.drop(dim, tag)
        held = gmsh.model.getPhysicalName(dim, tag)
        if self._owns(held):
            self._add(int(dim), int(tag), held)

    def drop(self, dim: int, tag: int) -> None:
        name = self._by_tag.pop((int(dim), int(tag)), None)
        if name is None:
            return
        by_dim = self._by_name[name]
        del by_dim[int(dim)]
        if not by_dim:
            del self._by_name[name]

    def forget(self, name: str) -> None:
        for d, t in self._by_name.pop(name, {}).items():
            self._by_tag.pop((d, t), None)

    def clear(self) -> None:
        self._by_name.clear()
        self._by_tag.clear()

    def _add(self, dim: int, tag: int, name: str) -> None:
        by_dim = self._by_name.setdefault(name, {})
        # The first group per dim wins, as the old scans did.
        if dim not in by_dim:
            by_dim[dim] = tag
            self._by_tag[(dim, tag)] = name
