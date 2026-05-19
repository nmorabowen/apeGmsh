"""ViewerNodes — node-side composite of :class:`ViewerData`.

Mirrors only the FEMData node-side accessors audited in
[phase-8.7-scope.md §2](../../opensees/architecture/phase-8.7-scope.md)
— the surface the viewer actually exercises.  Anything outside that
slice (e.g. the ``fem.nodes.select(...)`` selection API, partitions)
stays on the FEMData and is not migrated.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy import ndarray

from typing import Any

from ._records import (
    ConstraintRow,
    MassRow,
    NodalLoadRow,
    NodePairRow,
    SPRow,
    mass_row_from_record,
    nodal_load_row_from_record,
    sp_row_from_record,
)
from ._records import (
    constraint_row_from_record as _constraint_row_from_record,
)


# =====================================================================
# Selection sub-views
# =====================================================================


class _NamedNodeSelection:
    """Per-name id lookup for physical groups / labels / mesh selections.

    Constructed once per ``(side, source)`` pair, where *side* is
    ``"node"`` or ``"element"`` and decides what the map holds.  Both
    :meth:`node_ids` and :meth:`element_ids` read the same underlying
    map — call whichever fits the side you're consuming.  In practice
    ``view.nodes.physical.node_ids(...)`` and
    ``view.elements.physical.element_ids(...)`` are constructed
    against different maps and pick the matching method name.

    ``raise_on_missing=True`` mirrors :meth:`PhysicalGroupSet.node_ids` /
    :meth:`LabelSet.node_ids` (KeyError on unknown name).
    ``raise_on_missing=False`` mirrors
    :meth:`MeshSelectionStore.node_ids` for the selection-set path
    (returns empty array on unknown name — the documented gap in
    schema 2.4.0).
    """

    __slots__ = ("_by_name", "_raise_on_missing", "_label")

    def __init__(
        self,
        by_name: dict[str, ndarray],
        *,
        raise_on_missing: bool,
        label: str,
    ) -> None:
        self._by_name = by_name
        self._raise_on_missing = raise_on_missing
        self._label = label

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def _lookup(self, name: str) -> ndarray:
        try:
            return self._by_name[name]
        except KeyError:
            if self._raise_on_missing:
                raise KeyError(
                    f"No {self._label} named {name!r}. "
                    f"Available: {self.names()}"
                ) from None
            return np.array([], dtype=np.int64)

    def node_ids(self, name: str) -> ndarray:
        return self._lookup(name)

    def element_ids(self, name: str) -> ndarray:
        return self._lookup(name)


# =====================================================================
# Record sub-views (node-side)
# =====================================================================


class NodalLoadView:
    """Iterable view over node-side load rows."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[NodalLoadRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[NodalLoadRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def patterns(self) -> list[str]:
        """All unique pattern names in insertion order (mirrors :class:`NodalLoadSet`)."""
        seen: list[str] = []
        for r in self._rows:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list[NodalLoadRow]:
        return [r for r in self._rows if r.pattern == name]


class MassView:
    """Iterable view over mass rows."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[MassRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[MassRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)


class SPView:
    """Iterable view over SP (single-point constraint) rows."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[SPRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[SPRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)


class NodeConstraintView:
    """Iterable view over node-side constraint rows.

    Mirrors the subset of :class:`NodeConstraintSet` the viewer
    exercises today: ``__iter__`` over all rows, :meth:`pairs` for the
    flattened pair stream, :meth:`phantom_nodes` for the
    ``node_to_surface`` companion nodes.
    """

    __slots__ = ("_rows",)

    def __init__(self, rows: list[ConstraintRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[ConstraintRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def pairs(self) -> Iterator[NodePairRow]:
        """Yield every constraint as a flat sequence of pairs.

        :class:`NodeGroupRow` and :class:`NodeToSurfaceRow` are
        expanded into per-slave :class:`NodePairRow` instances —
        matching the natural OpenSees emission order documented in
        :class:`apeGmsh.mesh._record_set.NodeConstraintSet.pairs`.
        """
        from ._records import NodeGroupRow, NodeToSurfaceRow
        for row in self._rows:
            if isinstance(row, NodePairRow):
                yield row
            elif isinstance(row, NodeGroupRow):
                yield from _expand_group_to_pairs(row)
            elif isinstance(row, NodeToSurfaceRow):
                yield from row.rigid_link_records
                yield from row.equal_dof_records

    def phantom_nodes(self) -> tuple[ndarray, ndarray]:
        """Return ``(ids, coords)`` for every phantom node in this view.

        Phantom nodes are created by ``node_to_surface`` constraints;
        the live API's :meth:`NodeConstraintSet.phantom_nodes` returns
        a NodeResult that yields ``(id, xyz)`` tuples on iteration.
        Viewer code only needs the arrays, so this returns them
        directly; callers can ``zip`` them to recover the legacy
        pair-iter shape.
        """
        from ._records import NodeToSurfaceRow
        ids: list[int] = []
        coords: list[list[float]] = []
        for row in self._rows:
            if not isinstance(row, NodeToSurfaceRow):
                continue
            if row.phantom_coords is None:
                continue
            for tag, xyz in zip(row.phantom_nodes, row.phantom_coords):
                ids.append(int(tag))
                coords.append(list(xyz))
        if not ids:
            return (
                np.array([], dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
            )
        return (
            np.asarray(ids, dtype=np.int64),
            np.asarray(coords, dtype=np.float64),
        )


def _expand_group_to_pairs(row: Any) -> Iterator[NodePairRow]:
    """Expand a NodeGroupRow into per-slave NodePairRow instances.

    Mirrors :meth:`NodeGroupRecord.expand_to_pairs`: the per-pair kind
    becomes ``rigid_beam`` for rigid_diaphragm / rigid_body, or
    ``kinematic_coupling`` otherwise; the per-slave offset (when
    present) goes into the pair's ``offset`` field.
    """
    for i, sn in enumerate(row.slave_nodes):
        offset = None
        if row.offsets is not None:
            o = row.offsets[i]
            offset = (float(o[0]), float(o[1]), float(o[2]))
        if row.kind in ("rigid_diaphragm", "rigid_body"):
            pair_kind = "rigid_beam"
        else:
            pair_kind = "kinematic_coupling"
        yield NodePairRow(
            kind=pair_kind,
            master_node=int(row.master_node),
            slave_node=int(sn),
            dofs=row.dofs,
            offset=offset,
            name=row.name,
        )


# =====================================================================
# ViewerNodes
# =====================================================================


class ViewerNodes:
    """Read-only node-side composite consumed by the viewer.

    Construct via :class:`ViewerData`'s builders — never directly.
    """

    __slots__ = (
        "_ids",
        "_coords",
        "_id_to_idx",
        "physical",
        "labels",
        "selection",
        "loads",
        "sp",
        "masses",
        "constraints",
    )

    def __init__(
        self,
        *,
        ids: ndarray,
        coords: ndarray,
        physical: _NamedNodeSelection,
        labels: _NamedNodeSelection,
        selection: _NamedNodeSelection,
        loads: NodalLoadView,
        sp: SPView,
        masses: MassView,
        constraints: NodeConstraintView,
    ) -> None:
        self._ids = np.asarray(ids, dtype=np.int64)
        self._coords = np.asarray(coords, dtype=np.float64)
        self._id_to_idx: dict[int, int] | None = None
        self.physical = physical
        self.labels = labels
        self.selection = selection
        self.loads = loads
        self.sp = sp
        self.masses = masses
        self.constraints = constraints

    @property
    def ids(self) -> ndarray:
        return self._ids

    @property
    def coords(self) -> ndarray:
        return self._coords

    def index(self, nid: int) -> int:
        """Array index for a node ID.  O(1) after first call."""
        if self._id_to_idx is None:
            self._id_to_idx = {int(n): i for i, n in enumerate(self._ids)}
        try:
            return self._id_to_idx[int(nid)]
        except KeyError:
            raise KeyError(f"Node ID {nid} not found") from None

    def __len__(self) -> int:
        return int(self._ids.size)


# =====================================================================
# Builders — populated from a FEMData snapshot
# =====================================================================


def viewer_nodes_from_fem(fem: Any) -> ViewerNodes:
    """Build :class:`ViewerNodes` from a live :class:`FEMData`.

    Eagerly converts every record-side iterable into row tuples so the
    viewer surface is uniform regardless of source.
    """
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)

    physical = _named_selection_from_groupset(
        fem.nodes.physical, label="physical group", side="node",
        raise_on_missing=True,
    )
    labels = _named_selection_from_groupset(
        fem.nodes.labels, label="label", side="node",
        raise_on_missing=True,
    )
    selection = _named_selection_from_store(
        getattr(fem, "mesh_selection", None), side="node",
    )

    loads = NodalLoadView([
        nodal_load_row_from_record(r) for r in (fem.nodes.loads or [])
    ])
    sp = SPView([
        sp_row_from_record(r) for r in (fem.nodes.sp or [])
    ])
    masses = MassView([
        mass_row_from_record(r) for r in (fem.nodes.masses or [])
    ])

    cs_rows: list[ConstraintRow] = []
    if fem.nodes.constraints is not None:
        for rec in fem.nodes.constraints:
            cs_rows.append(_constraint_row_from_record(rec))
    constraints = NodeConstraintView(cs_rows)

    return ViewerNodes(
        ids=ids, coords=coords,
        physical=physical, labels=labels, selection=selection,
        loads=loads, sp=sp, masses=masses,
        constraints=constraints,
    )


def _named_selection_from_groupset(
    group_set: Any,
    *,
    label: str,
    side: str,
    raise_on_missing: bool,
) -> _NamedNodeSelection:
    """Snapshot the {name → node_ids} (or element_ids) map from a
    :class:`PhysicalGroupSet` / :class:`LabelSet`."""
    by_name: dict[str, ndarray] = {}
    if group_set is None:
        return _NamedNodeSelection(
            by_name, raise_on_missing=raise_on_missing, label=label,
        )
    try:
        keys = group_set.get_all()
    except (AttributeError, TypeError):
        return _NamedNodeSelection(
            by_name, raise_on_missing=raise_on_missing, label=label,
        )
    for dim, tag in keys:
        try:
            name = group_set.get_name(int(dim), int(tag))
        except (AttributeError, KeyError, ValueError):
            continue
        if not name:
            continue
        try:
            if side == "node":
                arr = group_set.node_ids(name)
            else:
                arr = group_set.element_ids(name)
        except (AttributeError, KeyError, ValueError):
            continue
        by_name[name] = np.asarray(arr, dtype=np.int64)
    return _NamedNodeSelection(
        by_name, raise_on_missing=raise_on_missing, label=label,
    )


def _named_selection_from_store(store: Any, *, side: str) -> _NamedNodeSelection:
    """Snapshot the {name → node_ids} (or element_ids) map from a
    :class:`MeshSelectionStore`.

    Mesh selections never raise on missing names — they return an
    empty array per the live store's documented behavior.  (The viewer
    relies on this for the ``selection=`` selector across the live and
    h5 paths.)
    """
    by_name: dict[str, ndarray] = {}
    if store is None:
        return _NamedNodeSelection(
            by_name, raise_on_missing=False, label="selection",
        )
    try:
        keys = store.get_all()
    except (AttributeError, TypeError):
        return _NamedNodeSelection(
            by_name, raise_on_missing=False, label="selection",
        )
    for dim, tag in keys:
        d, t = int(dim), int(tag)
        try:
            name = store.get_name(d, t)
        except (AttributeError, KeyError, ValueError):
            continue
        if not name:
            continue
        if side == "node":
            try:
                arr = store.node_ids(name)
            except (AttributeError, KeyError, ValueError):
                continue
        else:
            if d < 1:
                continue
            try:
                arr = store.element_ids(name)
            except (AttributeError, KeyError, ValueError):
                continue
        by_name[name] = np.asarray(arr, dtype=np.int64)
    return _NamedNodeSelection(
        by_name, raise_on_missing=False, label="selection",
    )
