"""
FEMData — Solver-ready FEM mesh broker.
========================================

The main output of apeGmsh's meshing pipeline.  Organized by what the
engineer needs: **Nodes** and **Elements** — with selections, BCs,
loads, and masses as sub-composites.

Top-level composites::

    fem.nodes       → NodeComposite   (IDs, coords, nodal loads, masses, node constraints)
    fem.elements    → ElementComposite (per-type element groups, surface constraints, element loads)
    fem.info        → MeshInfo        (mesh statistics)
    fem.inspect     → InspectComposite (introspection and summaries)

Construction::

    fem = FEMData.from_gmsh(dim=3, session=g, ndf=3)
    fem = FEMData.from_gmsh(session=g)          # all dims
    fem = FEMData.from_msh("bridge.msh", dim=2)
    fem = FEMData(nodes=..., elements=..., info=...)   # direct

Usage::

    # Domain nodes
    for nid, xyz in zip(*fem.nodes.get()):
        ops.node(nid, *xyz)

    # Supports
    for nid in fem.nodes.get_ids(pg="Base"):
        ops.fix(nid, 1, 1, 1)

    # Elements (iterate by type)
    for group in fem.elements:
        for eid, conn in group:
            ops.element(group.type_name, eid, *conn, mat_tag)

    # Elements (resolve to flat arrays — single type)
    ids, conn = fem.elements.get(label="col.web").resolve()

    # Constraints
    K = fem.nodes.constraints.Kind
    for c in fem.nodes.constraints.node_pairs():
        if c.kind == K.RIGID_BEAM:
            ops.rigidLink("beam", c.master_node, c.slave_node)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy import ndarray

from ._group_set import (
    NamedGroupSet, PhysicalGroupSet, LabelSet, _to_object,
)
from ._record_set import (
    ConstraintKind, LoadKind,
    NodeConstraintSet, SurfaceConstraintSet,
    NodalLoadSet, ElementLoadSet, MassSet,
)
from ._element_types import (
    ElementTypeInfo, ElementGroup, GroupResult,
    resolve_type_filter,
)

if TYPE_CHECKING:
    import pandas as pd
    from .MeshSelectionSet import MeshSelectionStore


# =====================================================================
# Result NamedTuples
# =====================================================================

class NodeResult(NamedTuple):
    """Bundled node IDs and coordinates.

    Destructurable::

        ids, coords = fem.nodes.get(pg="Base")

    Or use as an object::

        result = fem.nodes.get(pg="Base")
        result.ids       # ndarray(N,) object dtype
        result.coords    # ndarray(N, 3) float64
        result.to_dataframe()
    """
    ids:    ndarray
    coords: ndarray

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(
            self.coords,
            index=pd.Index(
                [int(x) for x in self.ids], name='node_id'),
            columns=['x', 'y', 'z'],
        )


# =====================================================================
# Bandwidth computation
# =====================================================================

def _compute_bandwidth(groups: dict[int, ElementGroup]) -> int:
    """Semi-bandwidth = max over all elements of (max_node - min_node).

    Iterates over per-type groups and takes the global maximum.
    """
    bw = 0
    for g in groups.values():
        if g.connectivity.size == 0:
            continue
        c = np.asarray(g.connectivity, dtype=np.int64)
        row_bw = int((c.max(axis=1) - c.min(axis=1)).max())
        bw = max(bw, row_bw)
    return bw


# =====================================================================
# MeshInfo
# =====================================================================

class MeshInfo:
    """Read-only summary of mesh statistics.

    Accessed via ``fem.info``.

    Attributes
    ----------
    n_nodes : int
    n_elems : int
    bandwidth : int
    types : list[ElementTypeInfo]
        Element types present in the mesh.
    """

    __slots__ = ('n_nodes', 'n_elems', 'bandwidth', 'types')

    def __init__(
        self,
        n_nodes: int,
        n_elems: int,
        bandwidth: int,
        types: list[ElementTypeInfo] | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_elems = n_elems
        self.bandwidth = bandwidth
        self.types = types or []

    # ── Backward compat ─────────────────────────────────────
    @property
    def nodes_per_elem(self) -> int:
        """First type's npe, or 0 if empty."""
        return self.types[0].npe if self.types else 0

    @property
    def elem_type_name(self) -> str:
        """First type's name, or empty string."""
        return self.types[0].name if self.types else ""

    def __repr__(self) -> str:
        parts = (
            f"MeshInfo(n_nodes={self.n_nodes}, n_elems={self.n_elems}, "
            f"bandwidth={self.bandwidth}"
        )
        if self.types:
            names = [t.name for t in self.types]
            parts += f", types={names}"
        return parts + ")"

    def summary(self) -> str:
        """One-line summary string."""
        s = f"{self.n_nodes} nodes, {self.n_elems} elements"
        if self.types:
            type_parts = [f"{t.name}:{t.count}" for t in self.types]
            s += f" ({', '.join(type_parts)})"
        s += f", bandwidth={self.bandwidth}"
        return s


# =====================================================================
# NodeComposite
# =====================================================================

class NodeComposite:
    """Access and query nodes from the FEM mesh.

    Primary interface::

        fem.nodes.get(pg="Base")    → NodeResult(ids, coords)
        fem.nodes.get()             → all domain nodes

    Sub-composites::

        fem.nodes.constraints       → NodeConstraintSet
        fem.nodes.loads             → NodalLoadSet
        fem.nodes.masses            → MassSet

    Public properties for raw array access::

        fem.nodes.ids               → ndarray(N,) object dtype
        fem.nodes.coords            → ndarray(N, 3) float64
    """

    def __init__(
        self,
        node_ids: ndarray,
        node_coords: ndarray,
        physical: PhysicalGroupSet,
        labels: LabelSet,
        constraints=None,
        loads=None,
        masses=None,
        partitions: dict[int, dict] | None = None,
    ) -> None:
        self._ids    = _to_object(node_ids)
        self._coords = np.asarray(node_coords, dtype=np.float64)
        self.physical = physical
        self.labels   = labels

        self.constraints = NodeConstraintSet(constraints)
        self.loads       = NodalLoadSet(loads)
        self.masses      = MassSet(masses)

        self._partitions: dict[int, dict] = partitions or {}
        self._id_to_idx: dict[int, int] | None = None

    # ── Public properties ───────────────────────────────────

    @property
    def ids(self) -> ndarray:
        """All domain node IDs.  ``ndarray(N,)`` object dtype."""
        return self._ids

    @property
    def coords(self) -> ndarray:
        """All domain node coordinates.  ``ndarray(N, 3)`` float64."""
        return self._coords

    @property
    def partitions(self) -> list[int]:
        """Sorted list of partition IDs (empty if not partitioned)."""
        return sorted(self._partitions.keys())

    # ── Selection API ───────────────────────────────────────

    def get(
        self,
        target: str | int | tuple[int, int] | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | tuple[int, int] | None = None,
        partition: int | None = None,
    ) -> NodeResult:
        """Bundled ``(ids, coords)`` for a selection.

        Parameters
        ----------
        target : str, int, or (dim, tag), optional
            Shorthand — searches PGs first, then labels.
        pg : str, optional
            Physical group name (explicit).
        label : str, optional
            Label name (explicit).
        tag : int or (dim, tag), optional
            Direct Gmsh physical-group tag lookup.
        partition : int, optional
            Partition ID (intersection filter).

        Returns
        -------
        NodeResult
        """
        if tag is not None:
            ids = self.physical.node_ids(tag)
            coords = self.physical.node_coords(tag)
        elif pg is not None:
            ids = self.physical.node_ids(pg)
            coords = self.physical.node_coords(pg)
        elif label is not None:
            ids = self.labels.node_ids(label)
            coords = self.labels.node_coords(label)
        elif target is not None:
            try:
                ids = self.physical.node_ids(target)
                coords = self.physical.node_coords(target)
            except (KeyError, ValueError):
                ids = self.labels.node_ids(target)
                coords = self.labels.node_coords(target)
        else:
            ids, coords = self._ids, self._coords

        if partition is not None:
            ids, coords = self._intersect_partition(
                ids, coords, partition)

        return NodeResult(ids, coords)

    def _intersect_partition(
        self, ids: ndarray, coords: ndarray, partition: int,
    ) -> tuple[ndarray, ndarray]:
        pdata = self._partitions.get(partition)
        if pdata is None:
            raise KeyError(
                f"Partition {partition} not found. "
                f"Available: {self.partitions}")
        mask = np.isin(
            np.asarray(ids, dtype=np.int64), pdata['node_ids'])
        return ids[mask], coords[mask]

    def get_ids(self, target=None, *, pg=None, label=None,
                tag=None, partition=None) -> ndarray:
        """Node IDs only."""
        return self.get(target, pg=pg, label=label, tag=tag,
                        partition=partition).ids

    def get_coords(self, target=None, *, pg=None, label=None,
                   tag=None, partition=None) -> ndarray:
        """Coordinates only."""
        return self.get(target, pg=pg, label=label, tag=tag,
                        partition=partition).coords

    # ── Lookups ─────────────────────────────────────────────

    def index(self, nid: int) -> int:
        """Array index for a node ID.  O(1) after first call."""
        if self._id_to_idx is None:
            self._id_to_idx = {
                int(n): i for i, n in enumerate(self._ids)}
        try:
            return self._id_to_idx[int(nid)]
        except KeyError:
            if len(self._ids) > 0:
                msg = (f"Node ID {nid} not found. "
                       f"Valid range: {int(self._ids.min())}-"
                       f"{int(self._ids.max())} "
                       f"({len(self._ids)} nodes)")
            else:
                msg = f"Node ID {nid} not found (no nodes)"
            raise KeyError(msg) from None

    # ── Dunder ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        parts = [f"NodeComposite({len(self._ids)} nodes)"]
        if self.constraints:
            parts.append(f"  constraints: {self.constraints!r}")
        if self.loads:
            parts.append(f"  loads: {self.loads!r}")
        if self.masses:
            parts.append(f"  masses: {self.masses!r}")
        return "\n".join(parts)


# =====================================================================
# ElementComposite
# =====================================================================

class ElementComposite:
    """Access and query elements from the FEM mesh.

    Iterable — yields ``ElementGroup`` objects::

        for group in fem.elements:
            print(group.type_name, len(group))

    Selection API::

        result = fem.elements.get(label="col.web")
        ids, conn = result.resolve()           # single-type
        ids, conn = result.resolve(element_type='tet4')  # pick one

    Sub-composites::

        fem.elements.constraints     → SurfaceConstraintSet
        fem.elements.loads           → ElementLoadSet
    """

    def __init__(
        self,
        groups: dict[int, ElementGroup],
        physical: PhysicalGroupSet,
        labels: LabelSet,
        constraints=None,
        loads=None,
        partitions: dict[int, dict] | None = None,
    ) -> None:
        self._groups: dict[int, ElementGroup] = dict(groups)
        self.physical = physical
        self.labels   = labels

        self.constraints = SurfaceConstraintSet(constraints)
        self.loads       = ElementLoadSet(loads)

        self._partitions: dict[int, dict] = partitions or {}

        # Lazy caches
        self._cached_ids: ndarray | None = None
        self._id_to_idx: dict[int, int] | None = None

    # ── Iteration ───────────────────────────────────────────

    def __iter__(self):
        """Yield ``ElementGroup`` objects (one per element type)."""
        return iter(self._groups.values())

    def __len__(self) -> int:
        """Total element count across all groups."""
        return sum(len(g) for g in self._groups.values())

    def __bool__(self) -> bool:
        return len(self) > 0

    # ── Public properties ───────────────────────────────────

    @property
    def ids(self) -> ndarray:
        """All element IDs concatenated.  ``ndarray(E,)`` int64."""
        if self._cached_ids is None:
            if not self._groups:
                self._cached_ids = np.array([], dtype=np.int64)
            else:
                self._cached_ids = np.concatenate(
                    [g.ids for g in self._groups.values()])
        return self._cached_ids

    @property
    def connectivity(self) -> ndarray:
        """Flat connectivity — only if all elements are the same type.

        Raises
        ------
        TypeError
            If multiple element types are present.
        """
        if not self._groups:
            return np.empty((0, 0), dtype=np.int64)
        if len(self._groups) > 1:
            names = [g.type_name for g in self._groups.values()]
            raise TypeError(
                f"Cannot return flat connectivity: "
                f"{len(self._groups)} element types present "
                f"({', '.join(names)}). "
                f"Use .resolve(element_type=...) or iterate groups."
            )
        return next(iter(self._groups.values())).connectivity

    @property
    def types(self) -> list[ElementTypeInfo]:
        """Element types present in the mesh."""
        return [g.element_type for g in self._groups.values()]

    @property
    def partitions(self) -> list[int]:
        """Sorted list of partition IDs."""
        return sorted(self._partitions.keys())

    @property
    def is_homogeneous(self) -> bool:
        """True if all elements are the same type."""
        return len(self._groups) <= 1

    # ── Type introspection ──────────────────────────────────

    def type_table(self) -> "pd.DataFrame":
        """DataFrame of element types in the mesh."""
        import pandas as pd
        rows = []
        for g in self._groups.values():
            t = g.element_type
            rows.append({
                'code': t.code,
                'name': t.name,
                'gmsh_name': t.gmsh_name,
                'dim': t.dim,
                'order': t.order,
                'npe': t.npe,
                'count': t.count,
            })
        return pd.DataFrame(rows)

    # ── Selection API ───────────────────────────────────────

    def get(
        self,
        target: str | int | tuple[int, int] | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | tuple[int, int] | None = None,
        dim: int | None = None,
        element_type: str | int | None = None,
        partition: int | None = None,
    ) -> GroupResult:
        """Select elements by PG, label, dim, element type, partition.

        All filters compose as AND intersections.  Returns a
        ``GroupResult`` — iterable of ``ElementGroup`` objects.

        Parameters
        ----------
        target : str, int, or (dim, tag), optional
            Shorthand — PGs first, then labels.
        pg, label, tag : optional
            Explicit selectors (same as ``NodeComposite.get``).
        dim : int, optional
            Filter by element dimension (0–3).
        element_type : str or int, optional
            Filter by element type alias (``'tet4'``), Gmsh code
            (``4``), or Gmsh name (``'Tetrahedron 4'``).
        partition : int, optional
            Filter by partition ID.

        Returns
        -------
        GroupResult
        """
        # Step 1: resolve ID set from PG/label/tag
        id_set = None
        if tag is not None:
            id_set = set(int(x) for x in self.physical.element_ids(tag))
        elif pg is not None:
            id_set = set(int(x) for x in self.physical.element_ids(pg))
        elif label is not None:
            id_set = set(int(x) for x in self.labels.element_ids(label))
        elif target is not None:
            try:
                id_set = set(
                    int(x) for x in self.physical.element_ids(target))
            except (KeyError, ValueError):
                id_set = set(
                    int(x) for x in self.labels.element_ids(target))

        # Step 2: partition filter
        if partition is not None:
            pdata = self._partitions.get(partition)
            if pdata is None:
                raise KeyError(
                    f"Partition {partition} not found. "
                    f"Available: {self.partitions}")
            pset = set(int(x) for x in pdata['element_ids'])
            id_set = pset if id_set is None else (id_set & pset)

        # Step 3: build filtered groups
        result_groups: list[ElementGroup] = []
        for g in self._groups.values():
            # dim filter
            if dim is not None and g.dim != dim:
                continue
            # element_type filter
            if element_type is not None:
                codes = resolve_type_filter(
                    element_type, list(self._groups.values()))
                if g.type_code not in codes:
                    continue
            # ID filter (from PG/label/partition)
            if id_set is not None:
                mask = np.isin(g.ids, list(id_set))
                if not mask.any():
                    continue
                filtered = ElementGroup(
                    element_type=g.element_type,
                    ids=g.ids[mask],
                    connectivity=g.connectivity[mask],
                )
                result_groups.append(filtered)
            else:
                result_groups.append(g)

        return GroupResult(result_groups)

    def get_ids(
        self,
        target=None, *, pg=None, label=None, tag=None,
        dim=None, element_type=None, partition=None,
    ) -> ndarray:
        """Element IDs only for a selection."""
        return self.get(
            target, pg=pg, label=label, tag=tag,
            dim=dim, element_type=element_type,
            partition=partition,
        ).ids

    def resolve(
        self,
        target=None, *, pg=None, label=None, tag=None,
        dim=None, element_type=None, partition=None,
    ) -> tuple[ndarray, ndarray]:
        """Flat ``(ids, connectivity)`` — convenience for single-type results.

        Delegates to ``.get(...).resolve()``.
        """
        return self.get(
            target, pg=pg, label=label, tag=tag,
            dim=dim, element_type=element_type,
            partition=partition,
        ).resolve()

    # ── Lookups ─────────────────────────────────────────────

    def index(self, eid: int) -> int:
        """Array index for an element ID.  O(1) after first call."""
        if self._id_to_idx is None:
            self._id_to_idx = {
                int(e): i for i, e in enumerate(self.ids)}
        try:
            return self._id_to_idx[int(eid)]
        except KeyError:
            ids = self.ids
            if len(ids) > 0:
                msg = (f"Element ID {eid} not found. "
                       f"Valid range: {int(ids.min())}-"
                       f"{int(ids.max())} "
                       f"({len(ids)} elements)")
            else:
                msg = f"Element ID {eid} not found (no elements)"
            raise KeyError(msg) from None

    # ── Dunder ──────────────────────────────────────────────

    def __repr__(self) -> str:
        type_parts = [f"{g.type_name}:{len(g)}"
                      for g in self._groups.values()]
        parts = [f"ElementComposite({len(self)} elements: "
                 f"{', '.join(type_parts)})"]
        if self.constraints:
            parts.append(f"  constraints: {self.constraints!r}")
        if self.loads:
            parts.append(f"  loads: {self.loads!r}")
        return "\n".join(parts)


# =====================================================================
# InspectComposite
# =====================================================================

class InspectComposite:
    """Introspection and summary methods.

    Accessed via ``fem.inspect``.
    """

    def __init__(self, fem: "FEMData") -> None:
        self._fem = fem

    # ── Tables ──────────────────────────────────────────────

    def summary(self) -> str:
        """One-line mesh summary plus sub-composite counts."""
        f = self._fem
        lines = [f.info.summary()]

        # Physical groups
        pg = f.nodes.physical
        if pg:
            lines.append(f"  Physical groups ({len(pg)}):")
            for (d, t), info in sorted(pg._groups.items()):
                name = info.get('name', '')
                n_n = len(info['node_ids'])
                eids = info.get('element_ids')
                n_e = len(eids) if eids is not None else 0
                lbl = f'"{name}"' if name else f"tag={t}"
                parts = f"{n_n} nodes"
                if n_e:
                    parts += f", {n_e} elems"
                lines.append(f"    ({d}) {lbl:24s} {parts}")

        # Labels
        lb = f.nodes.labels
        if lb:
            lines.append(f"  Labels ({len(lb)}):")
            for (d, t), info in sorted(lb._groups.items()):
                name = info.get('name', '')
                n_n = len(info['node_ids'])
                eids = info.get('element_ids')
                n_e = len(eids) if eids is not None else 0
                parts = f"{n_n} nodes"
                if n_e:
                    parts += f", {n_e} elems"
                lines.append(f"    ({d}) {name!r:24s} {parts}")

        # Element types
        if f.info.types:
            lines.append(f"  Element types ({len(f.info.types)}):")
            for t in f.info.types:
                lines.append(
                    f"    {t.name:12s} dim={t.dim}, "
                    f"order={t.order}, npe={t.npe}, "
                    f"count={t.count}")

        # Constraints
        nc = f.nodes.constraints
        sc = f.elements.constraints
        if nc:
            lines.append(f"  Node constraints: {nc!r}")
        if sc:
            lines.append(f"  Surface constraints: {sc!r}")
        if f.nodes.loads:
            lines.append(f"  Nodal loads: {f.nodes.loads!r}")
        if f.elements.loads:
            lines.append(f"  Element loads: {f.elements.loads!r}")
        if f.nodes.masses:
            lines.append(f"  {f.nodes.masses!r}")

        return "\n".join(lines)

    def node_table(self) -> "pd.DataFrame":
        """DataFrame of all nodes."""
        import pandas as pd
        f = self._fem
        return pd.DataFrame(
            f.nodes.coords,
            index=pd.Index(
                [int(x) for x in f.nodes.ids], name='node_id'),
            columns=['x', 'y', 'z'],
        )

    def element_table(self) -> "pd.DataFrame":
        """DataFrame of all elements with a ``type`` column."""
        import pandas as pd
        rows = []
        for group in self._fem.elements:
            for eid, conn_row in group:
                row: dict = {'elem_id': eid, 'type': group.type_name}
                for j, nid in enumerate(conn_row):
                    row[f'n{j}'] = int(nid)
                rows.append(row)
        return pd.DataFrame(rows).set_index('elem_id')

    def physical_table(self) -> "pd.DataFrame":
        return self._fem.nodes.physical.summary()

    def label_table(self) -> "pd.DataFrame":
        return self._fem.nodes.labels.summary()

    # ── Constraint/Load/Mass introspection ──────────────────

    def constraint_summary(self) -> str:
        """Human-readable breakdown of all constraints."""
        f = self._fem
        lines = []

        def _kind_summary(record_set, header):
            if not record_set:
                return
            lines.append(f"{header} ({len(record_set)} records):")
            counts: dict[str, int] = {}
            names: dict[str, str] = {}
            for r in record_set:
                k = r.kind
                counts[k] = counts.get(k, 0) + 1
                if k not in names and getattr(r, 'name', None):
                    names[k] = r.name
            for k, count in sorted(counts.items()):
                hint = f"  (source: {names[k]!r})" if k in names else ""
                lines.append(f"  {k:24s} {count:>4d}{hint}")

        _kind_summary(f.nodes.constraints, "Node constraints")
        nc = f.nodes.constraints
        if nc:
            n_phantom = sum(1 for _ in nc.extra_nodes())
            if n_phantom:
                lines.append(
                    f"  {'phantom nodes':24s} {n_phantom:>4d}"
                    f"  (created by node_to_surface)")
        _kind_summary(f.elements.constraints, "Surface constraints")

        if not lines:
            return "No constraints."
        return "\n".join(lines)

    def load_summary(self) -> str:
        """Human-readable breakdown of all loads."""
        f = self._fem
        lines = []

        nl = f.nodes.loads
        if nl:
            lines.append(f"Nodal loads ({len(nl)} records):")
            for pat in nl.patterns():
                recs = nl.by_pattern(pat)
                name_hint = ""
                for r in recs:
                    if getattr(r, 'name', None):
                        name_hint = f"  (source: {r.name!r})"
                        break
                lines.append(
                    f"  Pattern {pat!r:16s} {len(recs):>4d} "
                    f"nodal{name_hint}")

        el = f.elements.loads
        if el:
            lines.append(f"Element loads ({len(el)} records):")
            for pat in el.patterns():
                recs = el.by_pattern(pat)
                name_hint = ""
                for r in recs:
                    if getattr(r, 'name', None):
                        name_hint = f"  (source: {r.name!r})"
                        break
                ltype = getattr(recs[0], 'load_type', 'element') if recs else 'element'
                lines.append(
                    f"  Pattern {pat!r:16s} {len(recs):>4d} "
                    f"{ltype}{name_hint}")

        if not lines:
            return "No loads."
        return "\n".join(lines)

    def mass_summary(self) -> str:
        """Human-readable breakdown of masses."""
        f = self._fem
        ms = f.nodes.masses
        if not ms:
            return "No masses."
        lines = [f"Nodal masses ({len(ms)} nodes):"]
        lines.append(f"  Total mass: {ms.total_mass():.6g}")
        for r in ms:
            if getattr(r, 'name', None):
                lines.append(f"  Source: {r.name!r}")
                break
        return "\n".join(lines)


# =====================================================================
# FEMData — top-level broker
# =====================================================================

class FEMData:
    """Solver-ready FEM mesh broker.

    Organized by what the user needs::

        fem.nodes       → NodeComposite
        fem.elements    → ElementComposite
        fem.info        → MeshInfo
        fem.inspect     → InspectComposite
    """

    def __init__(
        self,
        nodes: NodeComposite,
        elements: ElementComposite,
        info: MeshInfo,
        mesh_selection: "MeshSelectionStore | None" = None,
    ) -> None:
        self.nodes    = nodes
        self.elements = elements
        self.info     = info
        self.mesh_selection = mesh_selection
        self.inspect  = InspectComposite(self)

    @property
    def partitions(self) -> list[int]:
        """Sorted list of partition IDs."""
        return self.nodes.partitions

    @classmethod
    def from_gmsh(
        cls,
        dim: int | None = None,
        *,
        session=None,
        ndf: int = 6,
        remove_orphans: bool = False,
    ):
        """Extract FEMData from a live Gmsh session.

        Parameters
        ----------
        dim : int or None
            Element dimension to extract.  ``None`` = all dims.
        session : apeGmsh session, optional
            When provided, auto-resolves constraints, loads, masses.
        ndf : int
            DOFs per node for load/mass vector padding.
        remove_orphans : bool
            If True, remove mesh nodes not connected to any element.
        """
        from ._fem_factory import _from_gmsh
        return _from_gmsh(
            cls, dim=dim, session=session, ndf=ndf,
            remove_orphans=remove_orphans)

    @classmethod
    def from_msh(
        cls,
        path: str,
        dim: int | None = 2,
        *,
        remove_orphans: bool = False,
    ):
        """Load FEMData from an external ``.msh`` file."""
        from ._fem_factory import _from_msh
        return _from_msh(cls, path=path, dim=dim,
                         remove_orphans=remove_orphans)

    def viewer(self, *, blocking: bool = False) -> None:
        """Open a non-interactive mesh viewer from this snapshot."""
        from ..results.Results import Results
        r = Results.from_fem(self, name="FEMData")
        r.viewer(blocking=blocking)

    def __repr__(self) -> str:
        return self.inspect.summary()
