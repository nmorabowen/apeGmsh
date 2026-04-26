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

Boolean safety
--------------
Both label PGs and user PGs survive every OCC boolean operation
(``fragment``, ``fuse``, ``cut``, ``intersect``).  Two module-level
functions — :func:`snapshot_physical_groups` and
:func:`remap_physical_groups` — implement a snapshot-then-remap
pattern that every boolean call site wraps around the OCC call.
Users never need to re-resolve labels after a boolean.

Naming convention
-----------------
At the Gmsh level, a label named ``"shaft"`` is stored as a
physical group with name ``"_label:shaft"``.  The prefix is
stripped by every method on this composite so the user never
sees it.  ``g.physical`` filters OUT any PG whose name starts
with ``_label:`` so the two tiers do not interfere.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import gmsh

from apeGmsh._logging import _HasLogging
from apeGmsh._types import DimTag

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase

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


# =====================================================================
# Boolean-safe PG preservation
# =====================================================================
#
# Gmsh's OCC boolean operations (fragment, fuse, cut, intersect)
# destroy physical group membership after synchronize().  These two
# functions implement a snapshot-then-remap pattern that every boolean
# call site must wrap around the OCC call:
#
#     snap = snapshot_physical_groups()
#     result, result_map = gmsh.model.occ.fragment(obj, tool, ...)
#     gmsh.model.occ.synchronize()
#     remap_physical_groups(snap, obj + tool, result_map)
#
# Or use the context manager for simpler call sites:
#
#     with pg_preserved() as pg:
#         result, result_map = gmsh.model.occ.fragment(obj, tool, ...)
#         pg.set_result(obj + tool, result_map)
# =====================================================================

class _PGPreserver:
    """Collects boolean result info and remaps PGs on context exit."""

    __slots__ = ('_snap', '_input_dts', '_result_map', '_result', '_absorbed')

    def __init__(self, snap: list[dict]) -> None:
        self._snap = snap
        self._input_dts: list[DimTag] | None = None
        self._result_map: list[list[DimTag]] | None = None
        self._result: list[DimTag] | None = None
        self._absorbed = False

    def set_result(
        self,
        input_dimtags: list[DimTag],
        result_map: list[list[DimTag]],
        *,
        result: list[DimTag] | None = None,
        absorbed_into_result: bool = False,
    ) -> None:
        self._input_dts = input_dimtags
        self._result_map = result_map
        self._result = result
        self._absorbed = absorbed_into_result


@contextmanager
def pg_preserved() -> Iterator[_PGPreserver]:
    """Context manager that snapshots PGs on entry and remaps on exit.

    Usage::

        with pg_preserved() as pg:
            result, result_map = gmsh.model.occ.fragment(obj, tool, ...)
            pg.set_result(obj + tool, result_map, result=result)
        # PGs are automatically remapped here
    """
    snap = snapshot_physical_groups()
    ctx = _PGPreserver(snap)
    yield ctx
    if ctx._input_dts is not None and ctx._result_map is not None:
        remap_physical_groups(
            ctx._snap, ctx._input_dts, ctx._result_map,
            result=ctx._result,
            absorbed_into_result=ctx._absorbed,
        )


def snapshot_physical_groups() -> list[dict]:
    """Capture every physical group (both ``_label:*`` and user PGs).

    Call this **before** any OCC boolean + synchronize sequence.

    Returns
    -------
    list[dict]
        One entry per PG: ``{'dim', 'pg_tag', 'name', 'entity_tags'}``.
    """
    snapshot: list[dict] = []
    for dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        snapshot.append({
            'dim':         dim,
            'pg_tag':      pg_tag,
            'name':        name,
            'entity_tags': [int(t) for t in ent_tags],
        })
    return snapshot


def remap_physical_groups(
    snapshot: list[dict],
    input_dimtags: list[DimTag],
    result_map: list[list[DimTag]],
    *,
    result: list[DimTag] | None = None,
    absorbed_into_result: bool = False,
) -> None:
    """Recreate PGs with remapped entity tags after a boolean operation.

    Must be called **after** ``occ.synchronize()``.

    Parameters
    ----------
    snapshot
        Value returned by :func:`snapshot_physical_groups`.
    input_dimtags
        The ``obj + tool`` dimtags that were passed to the OCC boolean
        call — same ordering as ``result_map``.
    result_map
        The second return value of the OCC boolean call.  ``result_map[i]``
        lists the dimtags that ``input_dimtags[i]`` became.
    result
        The first return value of the OCC boolean call — the list of
        all surviving dimtags.  Required for the absorbed-fallback
        when ``absorbed_into_result=True``: OCC's fuse returns
        ``result_map=[[], []]`` even though the merged volume exists,
        so ``result_map`` alone cannot tell us where to remap to.
    absorbed_into_result : bool, default False
        When True, entities whose ``result_map`` entry is empty are
        remapped to the **result** entities at the same dimension
        (the material merged, so the PG should follow).  Use this
        for ``fuse`` and ``intersect``.  When False (the default,
        appropriate for ``cut``), empty mappings mean the entity was
        consumed and a warning is emitted.

    Notes
    -----
    * Entities in a PG that were **inputs** to the boolean are remapped
      through ``result_map``.
    * Entities that were **not** inputs are kept if they still exist in
      the model.  If they disappeared (sub-topology casualty of a
      higher-dim boolean), a warning is emitted and they are dropped.
    * If a PG becomes entirely empty, a warning is emitted and the PG
      is not recreated.
    """
    if not snapshot:
        return

    # -- Build old-dimtag → [new-dimtags] mapping ----------------------
    dt_map: dict[DimTag, list[DimTag]] = {}
    for old_dt, new_dts in zip(input_dimtags, result_map):
        key = (int(old_dt[0]), int(old_dt[1]))
        dt_map[key] = [(int(d), int(t)) for d, t in new_dts]

    # -- Collect surviving entities per dimension (for absorbed-entity
    # fallback).  Prefer ``result`` (the first OCC return value) when
    # available because OCC's ``fuse`` leaves ``result_map`` empty
    # even when the merged entity exists in ``result``.  Fall back to
    # what ``result_map`` exposes for callers that did not pass it.
    result_by_dim: dict[int, list[int]] = {}
    source = result if result is not None else [
        dt for new_dts in result_map for dt in new_dts
    ]
    for d, t in source:
        result_by_dim.setdefault(int(d), []).append(int(t))
    for d in result_by_dim:
        result_by_dim[d] = sorted(set(result_by_dim[d]))

    # -- Current model entities (post-synchronize) ---------------------
    current_entities: set[DimTag] = set()
    for d in range(4):
        for _, t in gmsh.model.getEntities(d):
            current_entities.add((d, int(t)))

    # -- Remove surviving stale PGs, then recreate ---------------------
    for entry in snapshot:
        dim = entry['dim']
        old_pg_tag = entry['pg_tag']
        name = entry['name']
        old_tags = entry['entity_tags']

        # Remove the old PG if it survived synchronize with stale data.
        # Expected to fail when the PG was already destroyed by
        # synchronize() — Gmsh raises bare Exception, so we can't
        # narrow the catch.
        try:
            gmsh.model.removePhysicalGroups([(dim, old_pg_tag)])
        except Exception:
            pass

        # Remap each entity tag
        new_tags: list[int] = []
        for et in old_tags:
            old_dt = (dim, et)
            if old_dt in dt_map:
                # Entity was a boolean input — remap via result_map
                mapped = [t for d, t in dt_map[old_dt] if d == dim]
                if mapped:
                    new_tags.extend(mapped)
                else:
                    if absorbed_into_result:
                        # Entity was absorbed (e.g. fuse tool merged
                        # into the union).  Fall back to the result
                        # entities at this dim — material still there.
                        fallback = result_by_dim.get(dim, [])
                        if fallback:
                            new_tags.extend(fallback)
                            continue
                    warnings.warn(
                        f"Physical group '{name}' (dim={dim}): entity "
                        f"{et} was consumed by the boolean operation.",
                        stacklevel=3,
                    )
            elif old_dt in current_entities:
                # Not a boolean input and still exists — keep as-is
                new_tags.append(et)
            else:
                # Disappeared as sub-topology side-effect
                warnings.warn(
                    f"Physical group '{name}' (dim={dim}): entity {et} "
                    f"was lost (sub-topology renumbering). Cannot remap.",
                    stacklevel=3,
                )

        new_tags = sorted(set(new_tags))
        if new_tags:
            new_pg = gmsh.model.addPhysicalGroup(dim, new_tags)
            gmsh.model.setPhysicalName(dim, new_pg, name)
        elif old_tags:
            warnings.warn(
                f"Physical group '{name}' (dim={dim}) is now empty — "
                f"all its entities were consumed by the boolean.",
                stacklevel=3,
            )


# =====================================================================
# Label PG cleanup after entity removal
# =====================================================================


def cleanup_label_pgs(removed_dimtags: list[DimTag]) -> None:
    """Remove deleted entity tags from label PGs.

    Call after ``occ.remove()`` + ``synchronize()`` when the caller
    knows exactly which entities were deleted.  Drops any label PG
    that becomes empty after the cleanup.

    Only touches ``_label:*`` PGs — user-facing PGs are left alone
    because the caller may want different semantics there.
    """
    if not removed_dimtags:
        return
    removed_set = {(int(d), int(t)) for d, t in removed_dimtags}
    for dim, pg_tag in list(gmsh.model.getPhysicalGroups()):
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        if not is_label_pg(name):
            continue
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        new_tags = [int(t) for t in ent_tags if (dim, int(t)) not in removed_set]
        if len(new_tags) == len(ent_tags):
            continue  # nothing changed
        # Gmsh raises bare Exception — can't narrow the catch.
        try:
            gmsh.model.removePhysicalGroups([(dim, pg_tag)])
        except Exception:
            pass
        if new_tags:
            new_pg = gmsh.model.addPhysicalGroup(dim, new_tags)
            gmsh.model.setPhysicalName(dim, new_pg, name)


def reconcile_label_pgs() -> None:
    """Remove stale entity tags from ALL label PGs.

    Walks every ``_label:*`` PG and drops any tag whose entity no
    longer exists in the Gmsh model.  Use this after operations
    like ``removeAllDuplicates()`` that renumber entities without
    providing a result map.
    """
    current: set[DimTag] = set()
    for d in range(4):
        for _, t in gmsh.model.getEntities(d):
            current.add((d, int(t)))

    for dim, pg_tag in list(gmsh.model.getPhysicalGroups()):
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        if not is_label_pg(name):
            continue
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        new_tags = [int(t) for t in ent_tags if (dim, int(t)) in current]
        if len(new_tags) == len(ent_tags):
            continue  # nothing changed
        # Gmsh raises bare Exception — can't narrow the catch.
        try:
            gmsh.model.removePhysicalGroups([(dim, pg_tag)])
        except Exception:
            pass
        if new_tags:
            new_pg = gmsh.model.addPhysicalGroup(dim, new_tags)
            gmsh.model.setPhysicalName(dim, new_pg, name)


class Labels(_HasLogging):
    """Geometry-time entity naming composite (``g.labels``).

    Backed by Gmsh physical groups with an internal ``_label:``
    prefix.  See the module docstring for the two-tier naming
    architecture.
    """

    _log_prefix = "Labels"

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent

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

        # Build a name→(dim, pg_tag) index in one pass over all label
        # PGs.  This replaces up to 4 separate _find_pg_tag scans with
        # a single O(n) scan + O(1) dict lookups.
        label_index = self._label_index()

        # Check if this label already exists at this dim — merge
        # rather than duplicate.
        existing_tag = label_index.get((dim, prefixed))
        if existing_tag is not None:
            existing_ents = list(
                gmsh.model.getEntitiesForPhysicalGroup(dim, existing_tag)
            )
            new_tags = set(int(t) for t in tags)
            truly_new = new_tags - set(existing_ents)
            if truly_new:
                warnings.warn(
                    f"Label {name!r} (dim={dim}) already exists with "
                    f"{len(existing_ents)} entity(ies). Merging "
                    f"{len(truly_new)} new tag(s) into it. If this is "
                    f"unintentional, use a different label name.",
                    stacklevel=3,
                )
            merged = sorted(set(existing_ents) | new_tags)
            gmsh.model.removePhysicalGroups([(dim, existing_tag)])
            pg_tag = gmsh.model.addPhysicalGroup(dim, merged)
            gmsh.model.setPhysicalName(dim, pg_tag, prefixed)
            self._log(f"add({name!r}, dim={dim}) merged into pg_tag={pg_tag}")
            return pg_tag

        # Check if the same label name exists at a DIFFERENT dim —
        # warn about cross-dim shadowing.
        for other_dim in range(4):
            if other_dim == dim:
                continue
            if (other_dim, prefixed) in label_index:
                warnings.warn(
                    f"Label {name!r} already exists at dim={other_dim}, "
                    f"now also being created at dim={dim}. This may "
                    f"cause ambiguous lookups when dim= is not specified.",
                    stacklevel=3,
                )
                break

        pg_tag = gmsh.model.addPhysicalGroup(dim, [int(t) for t in tags])
        gmsh.model.setPhysicalName(dim, pg_tag, prefixed)
        self._log(f"add({name!r}, dim={dim}, tags={tags}) -> pg_tag={pg_tag}")
        return pg_tag

    @staticmethod
    def _label_index() -> dict[tuple[int, str], int]:
        """Build ``(dim, prefixed_name) -> pg_tag`` for all label PGs.

        One pass over ``getPhysicalGroups(-1)`` replaces repeated
        per-dimension scans.
        """
        index: dict[tuple[int, str], int] = {}
        for d, t in gmsh.model.getPhysicalGroups(-1):
            pg_name = gmsh.model.getPhysicalName(d, t)
            if is_label_pg(pg_name):
                index[(d, pg_name)] = t
        return index

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
            all dimensions.  If the label exists at exactly one
            dimension, returns those entities.  If it exists at
            multiple dimensions, raises ``ValueError`` asking the
            caller to specify ``dim=``.

        Returns
        -------
        list[int]
            Entity tags.

        Raises
        ------
        KeyError
            When no label with this name exists.
        ValueError
            When ``dim=None`` and the label exists at multiple
            dimensions.
        """
        prefixed = add_prefix(name)

        if dim is not None:
            # Direct lookup at a specific dimension
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(dim):
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
                f"no label {name!r} found at dim={dim}. "
                f"Available labels: {available}"
            )

        # dim=None — search all dimensions, require unambiguous match
        matches: list[tuple[int, int]] = []  # (pg_dim, pg_tag)
        for d in range(4):
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(d):
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                    matches.append((pg_dim, pg_tag))

        if not matches:
            available = self.get_all()
            raise KeyError(
                f"no label {name!r} found. Available labels: {available}"
            )

        if len(matches) == 1:
            pg_dim, pg_tag = matches[0]
            return [
                int(t)
                for t in gmsh.model.getEntitiesForPhysicalGroup(
                    pg_dim, pg_tag,
                )
            ]

        dims_found = sorted(set(d for d, _ in matches))
        raise ValueError(
            f"Label {name!r} exists at multiple dimensions "
            f"{dims_found}. Specify dim= to disambiguate, e.g. "
            f"g.labels.entities({name!r}, dim={dims_found[-1]})"
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

    def summary(self):
        """DataFrame describing every label in the model.

        Mirrors :meth:`PhysicalGroups.summary` but returns only the
        internal label PGs (with the ``_label:`` prefix stripped).

        Returns
        -------
        pd.DataFrame  indexed by ``(dim, pg_tag)`` with columns
        ``name``, ``n_entities``, ``entity_tags``.
        """
        import pandas as pd
        rows: list[dict] = []
        for d, t in gmsh.model.getPhysicalGroups():
            pg_name = gmsh.model.getPhysicalName(d, t)
            if not is_label_pg(pg_name):
                continue
            entities = gmsh.model.getEntitiesForPhysicalGroup(d, t)
            rows.append({
                'dim'        : d,
                'pg_tag'     : t,
                'name'       : strip_prefix(pg_name),
                'n_entities' : len(entities),
                'entity_tags': ", ".join(str(x) for x in entities),
            })
        if not rows:
            return pd.DataFrame(
                columns=['dim', 'pg_tag', 'name', 'n_entities', 'entity_tags']
            )
        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'pg_tag'])
            .sort_index()
        )

    def has(self, name: str, *, dim: int | None = None) -> bool:
        """Return True if a label with this name exists."""
        try:
            self.entities(name, dim=dim)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Remove / rename
    # ------------------------------------------------------------------

    def remove(self, name: str, *, dim: int | None = None) -> None:
        """Delete a label (and its backing physical group).

        Parameters
        ----------
        name : str
            Label name (without prefix).
        dim : int, optional
            Restrict to a single dimension.  When None, removes the
            label at **all** dimensions where it exists.

        Raises
        ------
        KeyError
            When no label with this name exists.
        """
        prefixed = add_prefix(name)
        dims = [dim] if dim is not None else [0, 1, 2, 3]
        removed = False
        for d in dims:
            for pg_dim, pg_tag in list(gmsh.model.getPhysicalGroups(d)):
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                    gmsh.model.removePhysicalGroups([(pg_dim, pg_tag)])
                    removed = True
        if not removed:
            raise KeyError(
                f"no label {name!r} found"
                + (f" at dim={dim}" if dim is not None else "")
                + f". Available labels: {self.get_all()}"
            )
        self._log(f"remove({name!r}, dim={dim})")

    def rename(self, old_name: str, new_name: str, *, dim: int | None = None) -> None:
        """Rename a label in place, preserving its entity membership.

        Parameters
        ----------
        old_name : str
            Current label name (without prefix).
        new_name : str
            New label name (without prefix).
        dim : int, optional
            Restrict to a single dimension.  When None, renames the
            label at **all** dimensions where it exists.

        Raises
        ------
        KeyError
            When no label with *old_name* exists.
        """
        old_prefixed = add_prefix(old_name)
        new_prefixed = add_prefix(new_name)
        dims = [dim] if dim is not None else [0, 1, 2, 3]
        renamed = False
        for d in dims:
            for pg_dim, pg_tag in list(gmsh.model.getPhysicalGroups(d)):
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == old_prefixed:
                    # Read entities, remove old PG, create new one
                    ent_tags = list(
                        gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
                    )
                    gmsh.model.removePhysicalGroups([(pg_dim, pg_tag)])
                    new_pg = gmsh.model.addPhysicalGroup(pg_dim, [int(t) for t in ent_tags])
                    gmsh.model.setPhysicalName(pg_dim, new_pg, new_prefixed)
                    renamed = True
        if not renamed:
            raise KeyError(
                f"no label {old_name!r} found"
                + (f" at dim={dim}" if dim is not None else "")
                + f". Available labels: {self.get_all()}"
            )
        self._log(f"rename({old_name!r} -> {new_name!r}, dim={dim})")

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
            f"promote_to_physical({label_name!r}) -> "
            f"PG {out_name!r} (dim={resolved_dim}, {len(tags)} entities)"
        )
        return pg_tag

    def reverse_map(self, *, dim: int = -1) -> dict[DimTag, str]:
        """Build a ``(dim, tag) -> label_name`` reverse lookup.

        Useful when callers need to find labels for many entities at
        once without repeated ``entities()`` calls.

        Parameters
        ----------
        dim : int, default -1
            Filter by dimension.  ``-1`` returns all dimensions.
        """
        result: dict[DimTag, str] = {}
        for d, pg_tag in gmsh.model.getPhysicalGroups(dim):
            pg_name = gmsh.model.getPhysicalName(d, pg_tag)
            if not is_label_pg(pg_name):
                continue
            name = strip_prefix(pg_name)
            for t in gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag):
                result[(int(d), int(t))] = name
        return result

    def labels_for_entity(self, dim: int, tag: int) -> list[str]:
        """Return all label names that contain the given entity."""
        names: list[str] = []
        for d, pg_tag in gmsh.model.getPhysicalGroups(dim):
            pg_name = gmsh.model.getPhysicalName(d, pg_tag)
            if not is_label_pg(pg_name):
                continue
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag)
            if tag in ent_tags:
                names.append(strip_prefix(pg_name))
        return names

    def __repr__(self) -> str:
        try:
            labels = self.get_all()
            return f"Labels({len(labels)} labels: {labels[:5]}{'...' if len(labels) > 5 else ''})"
        except Exception:
            return "Labels(session closed)"
