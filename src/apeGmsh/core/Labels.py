"""
Labels — geometry-time entity naming via internal physical groups.
==================================================================

Accessed via ``g.labels``.  Labels are human-readable names for
geometry entities that survive slicing, boolean operations, and
the STEP round-trip.  They are backed by Gmsh physical groups
whose names carry an internal ``_label:`` prefix so they are
invisible to solver-facing code (``g.physical``, ``fem.physical``,
the OpenSees exporter).

Two-tier naming
---------------
* **Labels** (``g.labels``) — geometry-time bookkeeping.
  Created automatically when ``label=`` is passed to any
  ``g.model.geometry.add_*`` method.  Used for addressing
  entities by name during boolean ops, slicing, and Part
  composition.  NOT visible to the solver.

* **Physical groups** (``g.physical``) — solver-time naming.
  Created explicitly by the user for boundary conditions,
  materials, and loads.  Visible to ``fem.physical`` and the
  OpenSees exporter.

The user promotes a label to a physical group when they are
ready to expose it to the solver::

    # Build and slice
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="shaft")
    g.model.geometry.slice("shaft", axis='z', offset=1.5)

    # Promote to a solver-facing PG
    tags = g.labels.entities("shaft")
    g.physical.add_volume(tags, name="column_shaft")

Naming convention
-----------------
At the Gmsh level, a label named ``"shaft"`` is stored as a
physical group with name ``"_label:shaft"``.  The prefix is
stripped by every method on this composite so the user never
sees it.  ``g.physical`` filters OUT any PG whose name starts
with ``_label:`` so the two tiers do not interfere.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase


# The internal prefix that distinguishes label PGs from user PGs.
LABEL_PREFIX = "_label:"


def is_label_pg(name: str) -> bool:
    """Return True if *name* is an internal label PG name."""
    return name.startswith(LABEL_PREFIX)


def strip_prefix(name: str) -> str:
    """Strip the ``_label:`` prefix from a label PG name."""
    if name.startswith(LABEL_PREFIX):
        return name[len(LABEL_PREFIX):]
    return name


def add_prefix(name: str) -> str:
    """Add the ``_label:`` prefix to a bare label name."""
    if name.startswith(LABEL_PREFIX):
        return name
    return LABEL_PREFIX + name


class Labels:
    """Geometry-time entity naming composite (``g.labels``).

    Backed by Gmsh physical groups with an internal ``_label:``
    prefix.  See the module docstring for the two-tier naming
    architecture.
    """

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Labels] {msg}")

    # ------------------------------------------------------------------
    # Create / update
    # ------------------------------------------------------------------

    def add(self, dim: int, tags: list[int], name: str) -> int:
        """Create a label for the given entities.

        If a label with the same name and dimension already exists,
        the tags are **merged** into the existing PG rather than
        creating a duplicate.

        Parameters
        ----------
        dim : int
            Entity dimension (0–3).
        tags : list[int]
            Entity tags to label.
        name : str
            Human-readable label name (without prefix).

        Returns
        -------
        int
            The Gmsh physical-group tag backing this label.
        """
        prefixed = add_prefix(name)

        # Check if this label already exists at this dim — merge
        # rather than duplicate.
        existing_tag = self._find_pg_tag(dim, prefixed)
        if existing_tag is not None:
            # Merge: get existing entity list, add new tags, recreate.
            existing_ents = list(
                gmsh.model.getEntitiesForPhysicalGroup(dim, existing_tag)
            )
            merged = sorted(set(existing_ents) | set(int(t) for t in tags))
            gmsh.model.removePhysicalGroups([(dim, existing_tag)])
            pg_tag = gmsh.model.addPhysicalGroup(dim, merged)
            gmsh.model.setPhysicalName(dim, pg_tag, prefixed)
            self._log(f"add({name!r}, dim={dim}) → merged into pg_tag={pg_tag}")
            return pg_tag

        pg_tag = gmsh.model.addPhysicalGroup(dim, [int(t) for t in tags])
        gmsh.model.setPhysicalName(dim, pg_tag, prefixed)
        self._log(f"add({name!r}, dim={dim}, tags={tags}) → pg_tag={pg_tag}")
        return pg_tag

    def _find_pg_tag(self, dim: int, prefixed_name: str) -> int | None:
        """Find the PG tag for a label at a given dim, or None."""
        for d, t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(d, t) == prefixed_name:
                return t
        return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def entities(self, name: str, *, dim: int | None = None) -> list[int]:
        """Return entity tags for a label.

        Parameters
        ----------
        name : str
            Label name (without prefix).
        dim : int, optional
            Restrict to a single dimension.  When None, searches
            all dimensions and returns the first match.

        Returns
        -------
        list[int]
            Entity tags.

        Raises
        ------
        KeyError
            When no label with this name exists.
        """
        prefixed = add_prefix(name)
        dims = [dim] if dim is not None else [0, 1, 2, 3]
        for d in dims:
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(d):
                pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
                if pg_name == prefixed:
                    return [
                        int(t)
                        for t in gmsh.model.getEntitiesForPhysicalGroup(
                            pg_dim, pg_tag,
                        )
                    ]
        available = self.get_all()
        raise KeyError(
            f"no label {name!r} found. Available labels: {available}"
        )

    def get_all(self, *, dim: int = -1) -> list[str]:
        """Return all label names (without prefix).

        Parameters
        ----------
        dim : int, default -1
            Filter by dimension.  ``-1`` returns all dimensions.
        """
        names: list[str] = []
        for d, t in gmsh.model.getPhysicalGroups(dim):
            pg_name = gmsh.model.getPhysicalName(d, t)
            if is_label_pg(pg_name):
                names.append(strip_prefix(pg_name))
        return sorted(set(names))

    def has(self, name: str, *, dim: int | None = None) -> bool:
        """Return True if a label with this name exists."""
        try:
            self.entities(name, dim=dim)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Promote to physical group
    # ------------------------------------------------------------------

    def promote_to_physical(
        self,
        label_name: str,
        *,
        pg_name: str | None = None,
        dim: int | None = None,
    ) -> int:
        """Copy a label's entities into a solver-facing physical group.

        The label remains intact — this is a **copy**, not a move.
        The new PG is visible to ``g.physical``, ``fem.physical``,
        and the OpenSees exporter.

        Parameters
        ----------
        label_name : str
            Label to promote.
        pg_name : str, optional
            Name for the new physical group.  Defaults to the
            label name (without prefix).
        dim : int, optional
            Dimension to promote.  Required when the label exists
            at multiple dimensions.

        Returns
        -------
        int
            Physical-group tag of the new PG.
        """
        tags = self.entities(label_name, dim=dim)
        out_name = pg_name or label_name

        # Resolve the dim from the label's PG
        prefixed = add_prefix(label_name)
        resolved_dim = dim
        if resolved_dim is None:
            for d in [3, 2, 1, 0]:
                for pd, pt in gmsh.model.getPhysicalGroups(d):
                    if gmsh.model.getPhysicalName(pd, pt) == prefixed:
                        resolved_dim = d
                        break
                if resolved_dim is not None:
                    break
        if resolved_dim is None:
            raise KeyError(f"label {label_name!r} not found")

        pg_tag = gmsh.model.addPhysicalGroup(resolved_dim, tags)
        gmsh.model.setPhysicalName(resolved_dim, pg_tag, out_name)
        self._log(
            f"promote_to_physical({label_name!r}) → "
            f"PG {out_name!r} (dim={resolved_dim}, {len(tags)} entities)"
        )
        return pg_tag

    def __repr__(self) -> str:
        try:
            labels = self.get_all()
            return f"Labels({len(labels)} labels: {labels[:5]}{'...' if len(labels) > 5 else ''})"
        except Exception:
            return "Labels(session closed)"
