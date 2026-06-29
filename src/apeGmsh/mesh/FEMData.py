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

    # Domain nodes — MeshSelection iterates as (node_id, xyz) pairs
    for nid, xyz in fem.nodes.select():
        ops.node(nid, *xyz)

    # Supports
    for nid in fem.nodes.select(pg="Base").ids:
        ops.fix(nid, 1, 1, 1)

    # Elements (iterate by type)
    for group in fem.elements:
        for eid, conn in group:
            ops.element(group.type_name, eid, *conn, mat_tag)

    # Elements (resolve to flat arrays — single type; .resolve() on the
    # GroupResult that .result() returns)
    ids, conn = fem.elements.select(label="col.web").result().resolve()

    # Constraints
    K = fem.nodes.constraints.Kind
    for c in fem.nodes.constraints.pairs():
        if c.kind == K.RIGID_BEAM:
            ops.rigidLink("beam", c.master_node, c.slave_node)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ._group_set import (
    PhysicalGroupSet, LabelSet, _to_object,
)
from .._kernel.record_sets import (
    NodeConstraintSet, SurfaceConstraintSet,
    NodalLoadSet, SPSet, ElementLoadSet, MassSet,
    PartitionSet, ComposeSet,
)
from .._kernel.records._compose import ComposeRecord
from .._kernel.records._partitions import PartitionRecord
from ._element_types import ElementTypeInfo
from .._kernel.payloads import (
    NodeResult, ElementGroup, GroupResult,
    resolve_type_filter,
)

if TYPE_CHECKING:
    import pandas as pd
    from .MeshSelectionSet import MeshSelectionStore


# =====================================================================
# NodeResult — pair-iterating node view
# =====================================================================
#
# RELOCATED to ``apeGmsh._kernel.payloads`` (selection-unification-v2
# P1-K, the keystone cycle-break — closes HT1/HT8/R3-B).  Class
# identity is unchanged; only the module path moved.  ``NodeResult`` is
# re-exported above via ``from .._kernel.payloads import NodeResult``
# (a downward ``mesh`` → ``_kernel`` edge — the intended layering
# direction) so ``from apeGmsh.mesh.FEMData import NodeResult`` and the
# byte-unchanged contract/pin tests keep resolving.  Flagged as a
# P3/P4 internal-cleanup candidate.


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

    # Per-slot type annotations (class-level so mypy picks them up;
    # no assignments because that would clash with __slots__).
    n_nodes: int
    n_elems: int
    bandwidth: int
    types: list[ElementTypeInfo]

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

#: Per-node provenance codes (ADR 0049).  Stored as ``int8`` aligned
#: 1:1 with ``NodeComposite.ids``.  ``MESH`` is an ordinary Gmsh-vertex
#: node; ``DECOUPLED`` is an auxiliary node declared via
#: ``g.decouple_node(...)``.  The all-``MESH`` case is encoded as a
#: ``None`` provenance array so it stays hash- and byte-identical to a
#: model with no decoupled nodes.
PROVENANCE_MESH: int = 0
PROVENANCE_DECOUPLED: int = 1


class NodeComposite:
    """Access and query nodes from the FEM mesh.

    Primary interface::

        fem.nodes.select(pg="Base")   → MeshSelection (iterates (id, xyz));
                                        .result() → NodeResult, .ids, .coords
        fem.nodes.select()            → all domain nodes

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
        sp=None,
        masses=None,
        partitions: dict[int, dict] | None = None,
        part_node_map: dict | None = None,
        ndf: ndarray | None = None,
        module_label: ndarray | None = None,
        provenance: ndarray | None = None,
    ) -> None:
        self._ids    = _to_object(node_ids)
        self._coords = np.asarray(node_coords, dtype=np.float64)
        self.physical = physical
        self.labels   = labels

        self.constraints = NodeConstraintSet(constraints)
        self.loads       = NodalLoadSet(loads)
        self.sp          = SPSet(sp)
        self.masses      = MassSet(masses)

        self._partitions: dict[int, dict] = partitions or {}
        self._id_to_idx: dict[int, int] | None = None
        # Snapshot of ``g.parts.build_node_map(...)`` at FEM-build
        # time — lets ``get(target=part_label)`` resolve without
        # needing a live Gmsh session (parts registry may be gone
        # by the time the user queries). Dict of ``str -> set[int]``.
        self._part_node_map: dict[str, set[int]] = part_node_map or {}

        # Per-node ``ndf`` (DOF count) — int8 array aligned 1:1 with
        # ``self._ids``.  Sentinel ``0`` means "undeclared".  In the
        # current model (ADR 0048/0049) per-node ndf is **inferred** by
        # the ``apeSees`` bridge from the incident element classes, with
        # ``ops.ndf(handle, ndf=K)`` the one explicit channel for an
        # element-less decoupled node; this array is only populated when
        # such metadata is supplied.  ``None`` means the broker carries
        # no per-node ndf metadata (e.g. a direct test fixture).
        self._ndf: ndarray | None
        if ndf is None:
            self._ndf = None
        else:
            arr = np.asarray(ndf, dtype=np.int8)
            if arr.shape != self._ids.shape:
                raise ValueError(
                    f"NodeComposite: ndf array shape {arr.shape} does "
                    f"not match node_ids shape {self._ids.shape}."
                )
            self._ndf = arr

        # Per-node ``module_label`` (Phase 3B.2c / ADR 0038) — object
        # ndarray of compose labels aligned 1:1 with ``self._ids``.
        # Empty string for host-owned rows; populated by the compose
        # merge engine for rows that came from a composed source
        # module.  ``None`` means the broker was constructed without
        # any module-label metadata (the uncomposed case + legacy
        # fixtures); the H5 writer falls back to an empty-string
        # parallel dataset in that case.
        self._module_label: ndarray | None
        if module_label is None:
            self._module_label = None
        else:
            mlbl = np.asarray(module_label, dtype=object)
            if mlbl.shape != self._ids.shape:
                raise ValueError(
                    f"NodeComposite: module_label array shape "
                    f"{mlbl.shape} does not match node_ids shape "
                    f"{self._ids.shape}."
                )
            self._module_label = mlbl

        # Per-node ``provenance`` (decoupled nodes — ADR 0049) — int8
        # array aligned 1:1 with ``self._ids``.  ``PROVENANCE_MESH`` (0)
        # = an ordinary Gmsh-vertex node; ``PROVENANCE_DECOUPLED`` (1) =
        # an auxiliary node declared via ``g.decouple_node(...)`` and
        # appended by the FEM factory above every mesh tag (dedup-immune
        # by construction).  ``None`` means the broker carries no
        # decoupled nodes (the common case + every import path that has
        # no notion of them); the all-mesh case is encoded as ``None``
        # so the snapshot_id hash + H5 bytes stay identical to a model
        # with no decoupled nodes at all (see ``_femdata_hash``).
        self._provenance: ndarray | None
        if provenance is None:
            self._provenance = None
        else:
            prov = np.asarray(provenance, dtype=np.int8)
            if prov.shape != self._ids.shape:
                raise ValueError(
                    f"NodeComposite: provenance array shape "
                    f"{prov.shape} does not match node_ids shape "
                    f"{self._ids.shape}."
                )
            self._provenance = prov

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
    def module_label(self) -> ndarray | None:
        """Per-node compose labels aligned 1:1 with :attr:`ids`.

        Object ndarray of source-module labels (empty string for
        host-owned rows), populated by the ``g.compose`` merge engine
        (ADR 0038 §"Schema").  ``None`` when the broker carries no
        module-label metadata (the uncomposed case).  Read-only view
        of the underlying array; consumers must not mutate it.
        """
        return self._module_label

    @property
    def provenance(self) -> ndarray | None:
        """Per-node provenance aligned 1:1 with :attr:`ids` (ADR 0049).

        ``int8`` array where ``PROVENANCE_MESH`` (0) marks an ordinary
        Gmsh-vertex node and ``PROVENANCE_DECOUPLED`` (1) marks an
        auxiliary node declared via ``g.decouple_node(...)``.  ``None``
        when the broker carries no decoupled nodes (the common case).
        Read-only view; consumers must not mutate it.
        """
        return self._provenance

    @property
    def decoupled_ids(self) -> ndarray:
        """Node IDs whose provenance is ``decoupled`` (ADR 0049).

        Empty ``ndarray`` when the broker carries no decoupled nodes.
        """
        if self._provenance is None:
            return np.array([], dtype=self._ids.dtype)
        return self._ids[self._provenance == PROVENANCE_DECOUPLED]

    @property
    def partitions(self) -> list[int]:
        """Sorted list of partition IDs (empty if not partitioned)."""
        return sorted(self._partitions.keys())

    # ── Selection API ───────────────────────────────────────

    def select(
        self,
        target=None,
        *,
        pg=None,
        label=None,
        tag=None,
        partition: int | None = None,
        dim: int | None = None,
        ids=None,
    ):
        """Select a subset of nodes from this FEM snapshot.

        Returns a :class:`~apeGmsh.mesh._mesh_selection.MeshSelection`
        (point family — ``.in_box`` tests node **coordinates**) that
        chains spatial-refinement verbs and terminates at ``.ids`` /
        ``.coords`` / ``.result()``::

            # seed by PG, read bulk arrays directly
            base = fem.nodes.select(pg="Base")
            base.ids      # list[int]
            base.coords   # ndarray (N, 3)

            # chain verbs, then drive a loop or feed ops
            for nid, xyz in fem.nodes.select(pg="Base").result():
                ops.node(nid, *xyz)

            # spatial narrowing
            corner = (fem.nodes.select(pg="Body")
                          .in_box((0, 0, 0), (1, 1, 1))
                          .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))

            # set algebra
            all_bcs = (fem.nodes.select(pg="Base")
                       | fem.nodes.select(pg="Wall"))

        No arguments seeds every domain node.

        .. note::
            **Point family** — ``.in_box`` tests node coordinates
            against a half-open box ``[lo, hi)`` (pass
            ``inclusive=True`` for the closed box).  For
            geometry-level selection use :meth:`g.model.select`.

        Parameters
        ----------
        target :
            Label name, physical group name, part name,
            ``(dim, tag)`` pair, raw int tag, or a list thereof.
            A string resolves through label → PG → part name in
            that order.
        pg :
            Physical group name or list of names.
        label :
            Geometry-time label name or list.  Labels survive
            boolean operations.
        tag :
            Raw physical group tag (int or list).
        partition :
            Restrict to nodes that belong to this partition number.
        dim :
            Restrict to nodes on entities of this topological
            dimension (0=point, 1=curve, 2=surface, 3=volume).
        ids :
            Explicit node id list.  When given, all other
            selectors are ignored.

        Refining verbs
        --------------
        Each returns a new ``MeshSelection`` and composes freely:

        - ``.in_box(lo, hi, *, inclusive=False)`` — half-open
          ``[lo, hi)`` by default; ``inclusive=True`` for
          ``[lo, hi]``.
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — ``tol=`` is
          **required**; raises ``TypeError`` if omitted.
        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)`` — callable ``xyz → bool``.
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminals
        ---------
        - ``.ids`` — ``list[int]`` of selected node IDs.
        - ``.coords`` — ``ndarray (N, 3)`` of coordinates.
        - ``.result()`` → :class:`~apeGmsh._kernel.payloads.NodeResult`;
          iterate as ``(nid, xyz)`` pairs, read ``.ids`` / ``.coords``
          arrays, or call ``.to_dataframe()``.
        """
        # selection-unification-v2: the host hook returns the v2
        # terminal ``MeshSelection`` (the point-family chain==terminal).
        # Same deferred-import idiom — ``_mesh_selection`` imports only
        # the package-root leaf ``_kernel.chain`` at load, so no new
        # eager cross-package edge (one declared downward BASELINE
        # triple; tripwire stays green).
        from ._mesh_selection import MeshSelection  # deferred — plan §3

        if ids is not None:
            atoms = [int(n) for n in ids]
        elif (
            target is None
            and pg is None
            and label is None
            and tag is None
            and partition is None
            and dim is None
        ):
            atoms = [int(n) for n in self._ids]
        else:
            seed_ids, seed_coords = self._resolve_nodes(
                target, pg=pg, label=label, tag=tag, dim=dim
            )
            if partition is not None:
                seed_ids, seed_coords = self._intersect_partition(
                    seed_ids, seed_coords, partition
                )
            atoms = [int(n) for n in seed_ids]
        return MeshSelection(atoms, _engine=self)

    def _resolve_nodes(self, target, *, pg, label, tag, dim=None):
        """Resolve single or list selectors to (ids, coords).

        Explicit keyword modes (``label=``, ``pg=``, ``tag=``) do a
        direct lookup with no fallback. The ``target=`` auto-resolve
        path tries **label → physical group → part label** for string
        items (matching ``LoadsComposite._resolve_target``), and
        resolves raw ``(dim, tag)`` DimTag tuples through a live Gmsh
        session.
        """
        if tag is not None:
            return self._union_nodes(
                tag, self.physical.node_ids,
                self.physical.node_coords, dim=dim)
        if pg is not None:
            return self._union_nodes(
                pg, self.physical.node_ids,
                self.physical.node_coords, dim=dim)
        if label is not None:
            return self._union_nodes(
                label, self.labels.node_ids,
                self.labels.node_coords, dim=dim)
        if target is not None:
            items = self._normalise_target(target)
            id_parts, coord_parts = [], []
            for t in items:
                ids, coords = self._resolve_one_target(t, dim=dim)
                id_parts.append(ids)
                coord_parts.append(coords)
            return self._dedupe_node_parts(id_parts, coord_parts)
        return self._ids, self._coords

    @staticmethod
    def _is_dimtag_tuple(x) -> bool:
        """True if *x* looks like a ``(dim, tag)`` DimTag pair."""
        return (
            isinstance(x, tuple)
            and len(x) == 2
            and all(isinstance(v, (int, np.integer)) for v in x)
        )

    @classmethod
    def _normalise_target(cls, target) -> list:
        """Coerce a target (single item or list) into a flat list of items."""
        if isinstance(target, str):
            return [target]
        if cls._is_dimtag_tuple(target):
            return [target]
        return list(target)

    def _resolve_one_target(self, t, *, dim=None):
        """Resolve one item (string name or ``(dim, tag)``) to (ids, coords)."""
        # Raw DimTag — query the live Gmsh session for the mesh nodes
        # on that geometry entity. Matches ``LoadsComposite`` semantics.
        if self._is_dimtag_tuple(t):
            if dim is not None and int(t[0]) != int(dim):
                raise ValueError(
                    f"tuple target {t} has dim={t[0]} but dim={dim} was "
                    f"passed — remove `dim=` or pass a string name."
                )
            return self._nodes_on_dimtag(int(t[0]), int(t[1]))
        if not isinstance(t, str):
            raise TypeError(
                f"target items must be strings or (dim, tag) tuples, "
                f"got {type(t).__name__}: {t!r}"
            )
        # String — try label, then PG, then part label (matches the
        # LoadsComposite._resolve_target precedence chain).  Catch only
        # KeyError ("not found at this tier"): a string node lookup
        # never raises ValueError, so a ValueError here is a real bug
        # and must NOT silently shadow the tier with a same-named PG.
        try:
            return (self.labels.node_ids(t, dim=dim),
                    self.labels.node_coords(t, dim=dim))
        except KeyError:
            pass
        try:
            return (self.physical.node_ids(t, dim=dim),
                    self.physical.node_coords(t, dim=dim))
        except KeyError:
            pass
        # Part labels are not dim-scoped; only honour this branch when
        # the caller did not restrict by dim.
        if dim is None and t in self._part_node_map:
            return self._nodes_from_ids(self._part_node_map[t])
        where = f" at dim={dim}" if dim is not None else ""
        raise KeyError(
            f"No label, physical group, or part named {t!r}{where}. "
            f"Labels: {list(self.labels._groups) if hasattr(self.labels, '_groups') else '?'}; "
            f"parts: {list(self._part_node_map)}"
        )

    def _nodes_on_dimtag(self, dim: int, tag: int):
        """Mesh nodes on a raw geometry entity via live Gmsh."""
        import gmsh
        try:
            nt, _, _ = gmsh.model.mesh.getNodes(
                dim=dim, tag=tag, includeBoundary=True,
                returnParametricCoord=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"could not resolve raw DimTag ({dim}, {tag}) — the "
                f"Gmsh session may have been closed. Pass the target "
                f"through an explicit `label=` or `pg=` that was "
                f"tagged before the session exited."
            ) from e
        return self._nodes_from_ids({int(n) for n in nt})

    def _nodes_from_ids(self, id_set):
        """Restrict the composite's (ids, coords) to a given ID set."""
        if not id_set:
            return np.array([], dtype=np.int64), np.zeros((0, 3))
        tag_to_idx = {int(t): i for i, t in enumerate(self._ids)}
        idxs = [tag_to_idx[i] for i in id_set if i in tag_to_idx]
        if not idxs:
            return np.array([], dtype=np.int64), np.zeros((0, 3))
        idx_arr = np.asarray(idxs, dtype=np.int64)
        kept_ids = np.asarray([self._ids[i] for i in idxs])
        return kept_ids, self._coords[idx_arr]

    @staticmethod
    def _union_nodes(selector, id_fn, coord_fn, *, dim=None):
        """Call id_fn/coord_fn for each item in selector, union results."""
        items = ([selector] if isinstance(selector, (str, int, tuple))
                 else list(selector))
        kw = {'dim': dim} if dim is not None else {}
        if len(items) == 1:
            return id_fn(items[0], **kw), coord_fn(items[0], **kw)
        id_parts = [id_fn(s, **kw) for s in items]
        coord_parts = [coord_fn(s, **kw) for s in items]
        return NodeComposite._dedupe_node_parts(id_parts, coord_parts)

    @staticmethod
    def _dedupe_node_parts(id_parts, coord_parts):
        """Concatenate and deduplicate by node ID."""
        if len(id_parts) == 1:
            return id_parts[0], coord_parts[0]
        all_ids = np.concatenate(id_parts)
        all_coords = np.concatenate(coord_parts)
        _, unique_idx = np.unique(
            np.asarray(all_ids, dtype=np.int64),
            return_index=True)
        unique_idx.sort()  # preserve original order
        return all_ids[unique_idx], all_coords[unique_idx]

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

    # ── ndf (DOF count) ─────────────────────────────────────

    def ndf_for(self, nid: int) -> int:
        """Return the per-node ``ndf`` (DOF count) for *nid*, if stored.

        Per-node ``ndf`` is **inferred** by the ``apeSees`` bridge from
        the incident element classes (ADR 0048); an element-less
        decoupled node states it explicitly with ``ops.ndf(handle,
        ndf=K)`` (ADR 0049).  This accessor only returns a value when
        such metadata was attached to the snapshot; the broker does not
        infer ndf on its own.

        Raises
        ------
        KeyError
            If ``nid`` is not a known node ID.
        LookupError
            If ``nid`` exists but the snapshot carries no ndf for it —
            the bridge infers it from elements (or, for an element-less
            node, ``ops.ndf(handle, ndf=K)``).
        """
        idx = self.index(nid)
        if self._ndf is None:
            raise LookupError(
                f"node {nid}: ndf not present on this snapshot — the "
                f"apeSees bridge infers it from incident element classes; "
                f"for an element-less decoupled node use ops.ndf(handle, ndf=K)."
            )
        val = int(self._ndf[idx])
        if val == 0:
            raise LookupError(
                f"node {nid}: ndf not present on this snapshot — the "
                f"apeSees bridge infers it from incident element classes; "
                f"for an element-less decoupled node use ops.ndf(handle, ndf=K)."
            )
        return val

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

        result = fem.elements.select(label="col.web").result()
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
        part_elem_map: dict | None = None,
        module_label: dict[int, ndarray] | None = None,
        reinforce_ties=None,
        embed_ties=None,
        contacts=None,
        contact_planes=None,
        rebar_elements=None,
    ) -> None:
        self._groups: dict[int, ElementGroup] = dict(groups)
        self.physical = physical
        self.labels   = labels

        self.constraints = SurfaceConstraintSet(constraints)
        self.loads       = ElementLoadSet(loads)

        # Embedded-reinforcement ties (g.reinforce, ADR 20 / R2b). A plain
        # list of ReinforceTieRecord — one LadrunoEmbeddedRebar coupling
        # per rebar node. The bridge build step consumes it
        # (opensees._internal.build.emit_reinforce_ties); it also round-trips
        # through the neutral model.h5 (/reinforce_ties group, neutral schema
        # 2.15.0, ADR 0067 P5.1). A model with no ties omits the group, so its
        # snapshot stays byte-identical (the snapshot_id hash excludes ties).
        self.reinforce_ties: list = list(reinforce_ties or [])

        # General node-to-host embedment ties (g.embed). A plain list of
        # EmbedTieRecord — one LadrunoEmbeddedNode coupling per node. The
        # bridge build step consumes it
        # (opensees._internal.build.emit_embed_ties); it also round-trips
        # through the neutral model.h5 (/embed_ties group, neutral schema
        # 2.22.0, ADR 0073). A model with no ties omits the group, so its
        # snapshot stays byte-identical (the snapshot_id hash excludes ties).
        self.embed_ties: list = list(embed_ties or [])

        # Face-to-face contact interactions (g.constraints.contact). A plain
        # list of ContactRecord. Consumed by
        # opensees._internal.build.emit_contacts; serial-only; it also
        # round-trips through the neutral model.h5 (/contacts group, neutral
        # schema 2.21.0, ADR 0073), mirroring /contact_planes. A model with no
        # contacts omits the group, so its snapshot stays byte-identical.
        self.contacts: list = list(contacts or [])

        # Rigid analytical-plane contacts (g.constraints.contact_plane). A plain
        # list of ContactPlaneRecord — one fork `contactPlane` per record.
        # Consumed by opensees._internal.build.emit_contact_planes; serial-only;
        # round-trips through the neutral model.h5 (/contact_planes group),
        # mirroring /contacts.
        self.contact_planes: list = list(contact_planes or [])
        # Structural rebar elements (ADR 0067 P5.2 / B1): the cage's
        # auto-emitted CorotTruss/dispBeamColumn intents from
        # g.rebar.place(emit_elements=True). A plain list of
        # RebarElementRecord — one per placed bar PG. The bridge build step
        # consumes it (opensees._internal.build.emit_rebar_elements), fanning
        # each across its PG's line cells. Empty unless emit_elements was set.
        self.rebar_elements: list = list(rebar_elements or [])

        self._partitions: dict[int, dict] = partitions or {}

        # Lazy caches
        self._cached_ids: ndarray | None = None
        self._id_to_idx: dict[int, int] | None = None

        # Snapshot of ``part_label -> set[element_id]`` built at
        # FEM-build time. Lets ``get(target=part_label)`` resolve
        # without a live Gmsh session.
        self._part_elem_map: dict[str, set[int]] = part_elem_map or {}

        # Per-element ``module_label`` (Phase 3B.2c / ADR 0038) —
        # dict keyed by element-type code, each value an object
        # ndarray of compose labels aligned 1:1 with the type's
        # ``ids`` / ``connectivity`` rows.  Empty string for
        # host-owned rows.  ``None`` (or missing key) means no
        # module-label metadata is carried for that type; the H5
        # writer falls back to empty strings.
        self._module_label: dict[int, ndarray] | None
        if module_label is None:
            self._module_label = None
        else:
            self._module_label = {}
            for code, arr in module_label.items():
                if arr is None:
                    continue
                mlbl = np.asarray(arr, dtype=object)
                grp = self._groups.get(int(code))
                expected = grp.ids.shape if grp is not None else None
                if expected is not None and mlbl.shape != expected:
                    raise ValueError(
                        f"ElementComposite: module_label[{code}] shape "
                        f"{mlbl.shape} does not match ids shape "
                        f"{expected}."
                    )
                self._module_label[int(code)] = mlbl

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
    def module_label(self) -> "dict[int, ndarray] | None":
        """Per-element compose labels keyed by element-type code.

        Each value is an object ndarray aligned 1:1 with that type's
        group ``ids`` (empty string for host-owned rows), populated by
        the ``g.compose`` merge engine (ADR 0038 §"Schema").  ``None``
        when the broker carries no module-label metadata (the
        uncomposed case).  Read-only view; consumers must not mutate.
        """
        return self._module_label

    def module_label_by_id(self) -> "dict[int, str] | None":
        """Flat ``element-id -> compose label`` map across all types.

        Concatenates the per-type :attr:`module_label` arrays against
        each type's ``ids``.  ``None`` when no module-label metadata is
        carried (the uncomposed case).  Used by the split-emit path to
        bucket each element into its owning module fragment.
        """
        if self._module_label is None:
            return None
        out: dict[int, str] = {}
        for code, grp in self._groups.items():
            labels = self._module_label.get(int(code))
            if labels is None:
                # No metadata for this type — host-owned by convention.
                for eid in grp.ids:
                    out[int(eid)] = ""
                continue
            for eid, lbl in zip(grp.ids, labels):
                out[int(eid)] = str(lbl)
        return out

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

    def _filtered_groups(
        self,
        target=None,
        *,
        pg=None,
        label=None,
        tag=None,
        dim: int | None = None,
        element_type: str | int | None = None,
        partition: int | None = None,
    ) -> GroupResult:
        """Internal element filter — Step-1/2/3 of the (P3-R-removed)
        public ``get``, relocated **verbatim** (selection-unification
        v2 P3-R / §6.3 M-STOP-1).  Sole consumer: ``select``'s
        auxiliary ``dim``/``element_type``/``partition`` branch — so
        ``select(element_type=|dim=|partition=)`` stays byte-behaviour-
        identical to the pre-P3-R ``get(...)`` path, including the
        §3.1(a) silent-empty ``dim=`` post-filter (mismatch → empty
        ``GroupResult``, no raise).
        """
        # Step 1: resolve ID set from PG/label/tag (union for lists)
        id_set = self._resolve_elem_ids(target, pg=pg, label=label, tag=tag)

        # Step 2: partition filter
        if partition is not None:
            pdata = self._partitions.get(partition)
            if pdata is None:
                raise KeyError(
                    f"Partition {partition} not found. "
                    f"Available: {self.partitions}")
            pset = set(int(x) for x in pdata['element_ids'])
            id_set = pset if id_set is None else (id_set & pset)

        # Resolve element_type filter once (not per-group)
        type_codes: set[int] | None = None
        if element_type is not None:
            type_codes = resolve_type_filter(
                element_type, list(self._groups.values()))

        # Step 3: build filtered groups
        result_groups: list[ElementGroup] = []
        for g in self._groups.values():
            # dim filter
            if dim is not None and g.dim != dim:
                continue
            # element_type filter
            if type_codes is not None and g.type_code not in type_codes:
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

    def _resolve_elem_ids(self, target, *, pg, label, tag):
        """Resolve single or list selectors to an element ID set (or None).

        Explicit keyword modes (``label=``, ``pg=``, ``tag=``) do a
        direct lookup. The ``target=`` auto-resolve path tries
        **label → physical group → part label** for string items
        (matching ``LoadsComposite._resolve_target``) and resolves
        raw ``(dim, tag)`` DimTag tuples through a live Gmsh session.
        """
        if tag is not None:
            return self._union_elem_ids(tag, self.physical.element_ids)
        if pg is not None:
            return self._union_elem_ids(pg, self.physical.element_ids)
        if label is not None:
            return self._union_elem_ids(label, self.labels.element_ids)
        if target is not None:
            items = NodeComposite._normalise_target(target)
            combined: set[int] = set()
            for t in items:
                combined.update(self._resolve_one_elem_target(t))
            return combined
        return None

    def _resolve_one_elem_target(self, t) -> set[int]:
        """Resolve one target item to an element-ID set."""
        # Raw DimTag — query live Gmsh
        if NodeComposite._is_dimtag_tuple(t):
            return self._elements_on_dimtag(int(t[0]), int(t[1]))
        if not isinstance(t, str):
            raise TypeError(
                f"target items must be strings or (dim, tag) tuples, "
                f"got {type(t).__name__}: {t!r}"
            )
        # String — label -> PG -> part, matching LoadsComposite.
        # ValueError is intentionally tolerated here (unlike the node
        # path): element_ids() raises ValueError for a dim-0 group
        # ("no element data"), and a points-only label legitimately
        # falling through to a same-named PG that *does* have elements
        # is the desired behaviour.
        try:
            return {int(x) for x in self.labels.element_ids(t)}
        except (KeyError, ValueError):
            pass
        try:
            return {int(x) for x in self.physical.element_ids(t)}
        except (KeyError, ValueError):
            pass
        if t in self._part_elem_map:
            return set(self._part_elem_map[t])
        raise KeyError(
            f"No label, physical group, or part named {t!r}. "
            f"Parts: {list(self._part_elem_map)}"
        )

    @staticmethod
    def _elements_on_dimtag(dim: int, tag: int) -> set[int]:
        """Element IDs on a raw geometry entity via live Gmsh."""
        import gmsh
        try:
            _, etags_list, _ = gmsh.model.mesh.getElements(dim, tag)
        except Exception as e:
            raise RuntimeError(
                f"could not resolve raw DimTag ({dim}, {tag}) — the "
                f"Gmsh session may have been closed."
            ) from e
        out: set[int] = set()
        for arr in etags_list:
            out.update(int(x) for x in arr)
        return out

    @staticmethod
    def _union_elem_ids(selector, id_fn) -> set[int]:
        """Call id_fn for each item in selector, union the ID sets."""
        items = ([selector] if isinstance(selector, (str, int, tuple))
                 else list(selector))
        combined: set[int] = set()
        for s in items:
            combined.update(int(x) for x in id_fn(s))
        return combined

    def select(
        self,
        target=None,
        *,
        pg=None,
        label=None,
        tag=None,
        dim: int | None = None,
        element_type: str | int | None = None,
        partition: int | None = None,
        ids=None,
    ):
        """Select a subset of elements from this FEM snapshot.

        Returns a :class:`~apeGmsh.mesh._mesh_selection.MeshSelection`
        (point family — spatial verbs test element **centroids**) that
        chains spatial-refinement verbs and terminates at ``.ids`` /
        ``.connectivity`` / ``.result()``::

            # seed by PG, read element ids
            body = fem.elements.select(pg="Body")
            body.ids          # list[int]
            body.connectivity # ndarray (N, npe) — homogeneous mesh only

            # filter to one element type in a spatial region
            tets = (fem.elements.select(pg="Body", element_type="tet4")
                        .in_box((0, 0, 0), (5, 5, 3)))

            # mixed mesh — call .resolve() on the GroupResult
            gr = fem.elements.select(label="col.web").result()
            ids, conn = gr.resolve()                      # single type
            ids, conn = gr.resolve(element_type="hex8")   # pick from mixed

        No arguments seeds every element.

        .. note::
            **Point family** — ``.in_box`` tests element centroids
            against a half-open box ``[lo, hi)`` (pass
            ``inclusive=True`` for the closed box).  For
            geometry-level selection use :meth:`g.model.select`.

        Parameters
        ----------
        target :
            Label name, physical group name, part name,
            ``(dim, tag)`` pair, raw int tag, or a list thereof.
            A string resolves through label → PG → part name in
            that order.
        pg :
            Physical group name or list of names.
        label :
            Geometry-time label name or list.  Labels survive
            boolean operations.
        tag :
            Raw physical group tag (int or list).
        dim :
            Restrict to elements of this topological dimension
            (1=line, 2=surface, 3=volume).
        element_type :
            Restrict to a specific element type by name (e.g.
            ``"tet4"``, ``"hex8"``) or Gmsh type code (int).
        partition :
            Restrict to elements belonging to this partition
            number.
        ids :
            Explicit element id list.  When given, all other
            selectors are ignored.

        Refining verbs
        --------------
        Each returns a new ``MeshSelection`` and composes freely
        (spatial verbs test element **centroids**):

        - ``.in_box(lo, hi, *, inclusive=False)`` — half-open
          ``[lo, hi)`` by default; ``inclusive=True`` for
          ``[lo, hi]``.
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — ``tol=`` is
          **required**; raises ``TypeError`` if omitted.
        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)`` — callable ``centroid_xyz → bool``.
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminals
        ---------
        - ``.ids`` — ``list[int]`` of selected element IDs.
        - ``.coords`` — ``ndarray (N, 3)`` of element centroids.
        - ``.connectivity`` — ``ndarray`` connectivity; **homogeneous
          selections only** (raises ``TypeError`` on mixed element
          types).
        - ``.groups()`` / ``.result()`` →
          :class:`~apeGmsh._kernel.payloads.GroupResult`.
          Call ``.resolve()`` **on the** ``GroupResult`` to get
          ``(ids, connectivity)``; pass ``element_type=`` to pick
          one type from a mixed selection.
        """
        # selection-unification-v2: the host hook returns the v2
        # terminal ``MeshSelection`` (the point-family chain==terminal).
        # Same deferred-import idiom; no new eager cross-package edge.
        from ._mesh_selection import MeshSelection  # deferred — plan §3

        if ids is not None:
            atoms = [int(e) for e in ids]
        elif (
            target is None
            and pg is None
            and label is None
            and tag is None
            and dim is None
            and element_type is None
            and partition is None
        ):
            atoms = [int(e) for e in self.ids]
        elif dim is None and element_type is None and partition is None:
            # Pure name/target seed — delegate to the exact resolver
            # `.get()` uses (FP-4 element-path swallow preserved by
            # reuse).  `None` means "all" (no PG/label/tag/target).
            id_set = self._resolve_elem_ids(
                target, pg=pg, label=label, tag=tag
            )
            atoms = (
                [int(e) for e in self.ids]
                if id_set is None
                else [int(e) for e in id_set]
            )
        else:
            # Auxiliary dim/element_type/partition filter present —
            # reuse the verbatim filter helper `_filtered_groups`
            # (selection-unification v2 P3-R / §6.3 M-STOP-1: the exact
            # body the now-removed public `get` used), so select(...)
            # stays byte-identical to the pre-P3-R get(...) path.
            atoms = [
                int(e)
                for e in self._filtered_groups(
                    target, pg=pg, label=label, tag=tag, dim=dim,
                    element_type=element_type, partition=partition,
                ).ids
            ]
        return MeshSelection(atoms, _engine=self)

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
            for etype in f.info.types:
                lines.append(
                    f"    {etype.name:12s} dim={etype.dim}, "
                    f"order={etype.order}, npe={etype.npe}, "
                    f"count={etype.count}")

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
            n_phantom = len(nc.phantom_nodes())
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
                erecs = el.by_pattern(pat)
                name_hint = ""
                for er in erecs:
                    if getattr(er, 'name', None):
                        name_hint = f"  (source: {er.name!r})"
                        break
                ltype = getattr(erecs[0], 'load_type', 'element') if erecs else 'element'
                lines.append(
                    f"  Pattern {pat!r:16s} {len(erecs):>4d} "
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

    # ── Topology diagnostics ────────────────────────────────

    def find_coincident_node_pairs(
        self,
        *,
        tol: float = 1e-6,
        pg: str | None = None,
    ) -> dict[tuple[int, int], list[str]]:
        """Find distinct nodes that share an XYZ within tolerance.

        Opt-in diagnostic for suspect topology — most commonly the
        arc-line junction case, where OCC builds an arc-bounded wire
        without welding the arc endpoints onto the joining line's point
        tags. The mesh then carries two distinct nodes at every
        junction with no element or constraint bridging them.

        Returns a dict mapping each coincident pair
        ``(tag_a, tag_b)`` — sorted so ``tag_a < tag_b`` — to a list
        of references that touch the pair:

        * ``"element <type>#<eid>"`` — both nodes appear in the same
          element's connectivity (legitimate for ``zeroLength``,
          tied interfaces, etc.)
        * ``"constraint <kind>"`` — equalDOF / rigidLink / diaphragm /
          kinematic / node_to_surface bridges the pair

        An **empty list** is the smoking gun: the pair is coincident
        but nothing references them together — i.e. an unbridged
        duplicate (the cimbra arc-line corner). An entry with only
        ``constraint`` refs means the user has explicitly tied the
        pair; an entry with an ``element zeroLength*`` ref is the
        canonical legitimate case.

        Parameters
        ----------
        tol : float
            Maximum Euclidean distance between two nodes to consider
            them coincident. Default ``1e-6``.  Single value — if
            you need per-constraint-type tolerances (each constraint
            kind has its own physical tol), drive the resolver's
            preflight directly instead.
        pg : str | None
            If given, restrict the scan to nodes belonging to this
            physical group. Otherwise scan all domain nodes.

        Returns
        -------
        dict[tuple[int, int], list[str]]
            ``{(tag_a, tag_b): [ref, ...]}`` for every coincident
            pair.  Empty dict if no coincident pairs are found.

        Warnings
        --------
        Builds a full-model KDTree on each call (SciPy ``cKDTree``,
        with a NumPy O(N²) fallback if SciPy is unavailable). Avoid
        invoking inside tight loops on million-node models — cache
        the result yourself if you need it repeatedly.

        Examples
        --------
        ::

            pairs = fem.inspect.find_coincident_node_pairs(tol=1e-6)
            for (a, b), refs in pairs.items():
                if not refs:
                    print(f"UNBRIDGED coincident pair: {a}, {b}")
                else:
                    print(f"Pair {a},{b} bridged by: {refs}")
        """
        from apeGmsh._kernel.resolvers._constraint_resolver._geom import (
            _SpatialIndex,
        )

        f = self._fem
        all_ids: ndarray = f.nodes.ids
        all_coords: ndarray = f.nodes.coords

        # Optional PG restriction.
        if pg is not None:
            sel_ids = f.nodes.select(pg=pg).ids
            keep: set[int] = {int(n) for n in sel_ids}
            mask = np.array([int(t) in keep for t in all_ids], dtype=bool)
            ids = all_ids[mask]
            coords = all_coords[mask]
        else:
            ids = all_ids
            coords = all_coords

        n = len(ids)
        if n < 2:
            return {}

        index = _SpatialIndex(np.asarray(coords, dtype=float))

        # Walk every node, query the ball, register every distinct
        # neighbour as a coincident pair (sorted to dedupe).
        pairs: dict[tuple[int, int], list[str]] = {}
        for i in range(n):
            hits = index.query_ball_point(coords[i], float(tol))
            for j in hits:
                if int(j) == i:
                    continue
                ta = int(ids[i])
                tb = int(ids[j])
                key = (ta, tb) if ta < tb else (tb, ta)
                if key not in pairs:
                    pairs[key] = []

        if not pairs:
            return pairs

        # Cross-reference: which elements / constraints touch each pair.
        # Build a node -> set[pairs] inverted index for O(1) lookup.
        node_to_pairs: dict[int, list[tuple[int, int]]] = {}
        for key in pairs:
            node_to_pairs.setdefault(key[0], []).append(key)
            node_to_pairs.setdefault(key[1], []).append(key)

        # Element scan — credit a ref when BOTH endpoints of a pair
        # appear in the same connectivity row.
        for group in f.elements:
            type_name = group.type_name
            for eid, conn in group:
                conn_set = {int(n) for n in conn}
                seen: set[tuple[int, int]] = set()
                for nid in conn_set:
                    for key in node_to_pairs.get(nid, ()):
                        if key in seen:
                            continue
                        if key[0] in conn_set and key[1] in conn_set:
                            pairs[key].append(f"element {type_name}#{int(eid)}")
                            seen.add(key)

        # Constraint scan — flat pairs() expands every constraint kind
        # (equal_dof, rigid_*, diaphragm, kinematic, node_to_surface)
        # into NodePairRecords. A ref counts when the constraint's
        # (master, slave) hits our coincident pair (either order).
        try:
            constraint_pairs = list(f.nodes.constraints.pairs())
        except Exception:
            constraint_pairs = []
        for cp in constraint_pairs:
            a = int(getattr(cp, "master_node"))
            b = int(getattr(cp, "slave_node"))
            key = (a, b) if a < b else (b, a)
            if key in pairs:
                kind = getattr(cp, "kind", "constraint")
                pairs[key].append(f"constraint {kind}")

        return pairs


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
        composed_from: "ComposeSet | tuple[ComposeRecord, ...] | None" = None,
    ) -> None:
        self.nodes    = nodes
        self.elements = elements
        self.info     = info
        self.mesh_selection = mesh_selection
        self.inspect  = InspectComposite(self)
        # ── Compose provenance (Phase 3A.1 / ADR 0038) ───────────
        # ``fem.composed_from`` is a :class:`ComposeSet` exposing one
        # record per composed source module (label-keyed, sorted).
        # Default is an empty set — the canonical "uncomposed" signal.
        # Phase 3B's ``Compose`` facade is the producer; the H5 reader
        # rehydrates it from the optional ``/composed_from/`` sub-group
        # when present.
        if composed_from is None:
            self.composed_from: ComposeSet = ComposeSet(())
        elif isinstance(composed_from, ComposeSet):
            self.composed_from = composed_from
        else:
            self.composed_from = ComposeSet(tuple(composed_from))
        # Wire the sibling NodeComposite onto the ElementComposite so
        # fem.elements.select(...) can compute element centroids
        # in-memory (no live Gmsh session needed — works for
        # import-origin FEMData too). Every construction path funnels
        # through this __init__, so one wiring line covers from_gmsh /
        # from_msh / from_h5 / from_native / from_mpco / direct. The
        # attribute name is the contract shared with
        # mesh/_mesh_selection.NODES_REF_ATTR.
        elements._apegmsh_nodes_ref = nodes

        # ── Partitions composite ─────────────────────────────
        # ``fem.partitions`` is a :class:`PartitionSet` (P2 broker
        # composite) built once from the same dicts the extractor
        # already populated on ``nodes._partitions`` /
        # ``elements._partitions``.  Those private back-stores are
        # left untouched — they still power ``select(partition=N)``;
        # this set is a Python-side ergonomic layer only.
        node_parts: dict[int, dict] = getattr(nodes, "_partitions", {}) or {}
        elem_parts: dict[int, dict] = (
            getattr(elements, "_partitions", {}) or {})
        pids = sorted(set(node_parts.keys()) | set(elem_parts.keys()))
        records: dict[int, PartitionRecord] = {}
        empty_i64 = np.array([], dtype=np.int64)
        for pid in pids:
            n_ids = np.asarray(
                node_parts.get(pid, {}).get("node_ids", empty_i64),
                dtype=np.int64,
            )
            e_ids = np.asarray(
                elem_parts.get(pid, {}).get("element_ids", empty_i64),
                dtype=np.int64,
            )
            records[int(pid)] = PartitionRecord(
                id=int(pid), node_ids=n_ids, element_ids=e_ids,
            )
        self.partitions: PartitionSet = PartitionSet(records)

    @property
    def snapshot_id(self) -> str:
        """Deterministic content hash identifying this FEMData snapshot.

        Computed once and cached. Used by the Results module to bind
        result files to their producing geometry — see
        ``internal_docs/Results_architecture.md`` § "FEMData embedding
        & binding".
        """
        cached = getattr(self, "_snapshot_id_cache", None)
        if cached is not None:
            return cached
        from ._femdata_hash import compute_snapshot_id
        digest = compute_snapshot_id(self)
        self._snapshot_id_cache = digest
        return digest

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

    @classmethod
    def from_h5(cls, path: str, *, root: str = "/") -> "FEMData":
        """Load a :class:`FEMData` snapshot from a root-layout ``model.h5``.

        Inverse of :meth:`to_h5`.  Reads the seven neutral-zone groups
        plus ``/meta`` and rebuilds nodes, elements (per type),
        physical groups, labels, mesh selections, constraints, loads,
        and masses — everything the writer round-trips.

        Parameters
        ----------
        path : str
            Path to a model.h5 written by :meth:`to_h5`, ``g.save()``,
            or ``apeSees(fem).h5(path)``.
        root : str, default ``"/"``
            Sub-group root inside ``path`` to read from.  Default
            rehydrates from the file root (standalone ``model.h5``
            shape).  Per ADR 0020 (Phase 4 cleanup), composed
            ``results.h5`` files carry the same rich layout under
            ``/model/``; pass ``root="/model"`` to rehydrate from a
            composed file.  Backcompat: ``root="/"`` produces
            byte-identical behaviour to the pre-refactor reader.

        Use this to resume a session-saved model in a later script::

            # script 1 — build & save
            with apeGmsh(model_name="m", save_to="m.h5") as g:
                ...

            # script 2 — analyse
            fem = FEMData.from_h5("m.h5")
            apeSees(fem).h5("m.h5")     # enrich with /opensees/...
        """
        from ._femdata_h5_io import read_fem_h5
        return read_fem_h5(path, root=root)

    def to_native_h5(self, group) -> None:
        """Embed this FEMData into an open HDF5 group (``/model/``).

        Used by ``NativeWriter`` to snapshot the geometry alongside
        results.  The reconstructed FEMData (via ``from_native_h5``)
        will produce the same ``snapshot_id`` — this is the linking
        contract for ``Results.bind()``.

        Phase 4 cleanup (ADR 0020): writes the rich neutral-zone
        layout :func:`write_fem_h5` produces at the file root, but
        under ``group`` instead.  The composed ``results.h5`` thus
        carries ``/model/meta``, ``/model/nodes``, ``/model/elements``,
        etc. — the same layout :func:`read_fem_h5(path, root="/model")`
        rehydrates from.  This eliminates the ``/opensees_archive/``
        zone that the previous lean embedding required to round-trip
        the full :class:`OpenSeesModel`.
        """
        from ._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group)

    def to_h5(
        self,
        path: str,
        *,
        model_name: str = "",
        apegmsh_version: str = "",
        ndf: int = 0,
    ) -> None:
        """Write a fresh ``model.h5`` containing the neutral zone.

        Phase 8.5 entry point: dumps everything the broker knows about
        the model (nodes, elements per type, physical groups, labels,
        constraints, loads, masses) into a root-level
        ``model.h5``.  No ``/opensees/`` content is emitted — absent
        enrichment is the right "no solver loaded" signal.

        Use ``apeSees(fem).h5(path)`` instead to get a fully enriched
        file (neutral zone + ``/opensees/...``).
        """
        # g.reinforce (ADR 20 / R2b → ADR 0067 P5.1): LadrunoEmbeddedRebar
        # ties now round-trip through the neutral model.h5 (persisted into the
        # /reinforce_ties group, neutral schema 2.15.0). No deferral warning.
        #
        # ADR 0067 P5.2 / B1a.2: the cage's auto-emitted structural rebar
        # elements (g.rebar.place(emit_elements=True)) now round-trip through
        # the neutral model.h5 (persisted into the /rebar_elements group,
        # neutral schema 2.16.0). No deferral warning.
        from ._femdata_h5_io import write_fem_h5
        write_fem_h5(
            self, path,
            model_name=model_name,
            apegmsh_version=apegmsh_version,
            ndf=ndf,
        )

    @classmethod
    def from_native_h5(cls, group) -> "FEMData":
        """Reconstruct a FEMData from its embedded ``/model/`` group.

        Phase 4 cleanup (ADR 0020): production writers (:meth:`to_native_h5`
        via :class:`NativeWriter`) embed the rich neutral zone — full
        constraints, loads, masses, mesh selections and partitions
        round-trip alongside nodes/elements/PGs.  ``snapshot_id`` of
        the rebuilt FEM matches the source's ``/meta/snapshot_id``
        attribute (the linking contract :class:`Results.bind` relies
        on).
        """
        from ._femdata_h5_io import read_neutral_zone_from_group

        return read_neutral_zone_from_group(
            group, label=getattr(group, "name", "<h5 group>"),
        )

    @classmethod
    def from_mpco_model(cls, group) -> "FEMData":
        """Synthesize a partial FEMData from an MPCO ``MODEL/`` group.

        Carries: nodes, elements (per OpenSees class tag), physical
        groups derived from MPCO Regions (``MODEL/SETS``).

        Missing vs. native:
        - apeGmsh-specific ``labels``
        - Pre-mesh declarations (loads / masses / constraints)
        - STKO named selection sets (those live in ``.cdata`` sidecars)
        - Gmsh-style element type codes (uses negated class_tag instead)

        ``snapshot_id`` will not match a native FEMData of the same
        mesh — that's expected. ``Results.bind()`` will refuse such
        mismatches.
        """
        from ._femdata_mpco_io import read_fem_from_mpco
        return read_fem_from_mpco(group)

    @classmethod
    def from_ladruno_model(cls, group) -> "FEMData":
        """Synthesize a partial FEMData from a ``.ladruno`` ``MODEL/`` group.

        Sibling of :meth:`from_mpco_model` for the self-describing
        ``.ladruno`` layout (element groups carry a ``CONNECTIVITY``
        dataset + BASIS attrs). Carries nodes, elements (per OpenSees
        class tag), and physical groups from ``MODEL/SETS``. Missing vs.
        native: apeGmsh ``labels``, pre-mesh declarations, selection-set
        names. ``snapshot_id`` will not match a native FEMData of the
        same mesh — expected.
        """
        from ._femdata_ladruno_io import read_fem_from_ladruno
        return read_fem_from_ladruno(group)

    # ------------------------------------------------------------------
    # Pure transforms — Phase 3B.2b-prep / ADR 0038
    # ------------------------------------------------------------------
    #
    # ``with_constraint`` / ``with_load`` / ``with_mass`` append a single
    # already-resolved record to the right sub-composite and return a
    # new :class:`FEMData` snapshot.  ``self`` is unchanged; the result
    # shares the heavy node / element arrays via shallow-copy but
    # carries a fresh record set (and a fresh
    # :class:`NodeComposite` / :class:`ElementComposite` wrapper).  The
    # :attr:`snapshot_id` cache is dropped on the result so the new
    # records actually fold into the next hash.
    #
    # These are the canonical mutation primitives behind the upcoming
    # ``FEMData.compose(...)`` engine (Phase 3B.2c) and the chain-phase
    # session shim retrofit.  They never touch gmsh — the caller is
    # responsible for producing a fully-resolved record.

    def _replaced_nodes(self, **overrides):
        """Return a shallow-copied :class:`NodeComposite` with overrides.

        Internal helper for the ``with_*`` transforms.  Copies are kept
        shallow so the heavy ``_ids`` / ``_coords`` / ``_ndf`` /
        ``physical`` / ``labels`` references are shared with the
        predecessor — only the named composite attribute is swapped.
        """
        import copy
        new_nodes = copy.copy(self.nodes)
        for k, v in overrides.items():
            setattr(new_nodes, k, v)
        return new_nodes

    def _replaced_elements(self, **overrides):
        """Return a shallow-copied :class:`ElementComposite` with overrides.

        Mirrors :meth:`_replaced_nodes` for the element-side
        sub-composite.
        """
        import copy
        new_elements = copy.copy(self.elements)
        for k, v in overrides.items():
            setattr(new_elements, k, v)
        return new_elements

    def _replaced(self, *, nodes=None, elements=None) -> "FEMData":
        """Shallow-copy this :class:`FEMData` with one or both of
        ``nodes`` / ``elements`` swapped.

        Pure transform helper.  Drops the cached ``snapshot_id`` on the
        new instance so the changed records actually fold into the
        next hash.  All other attributes (``info`` / ``inspect`` /
        ``composed_from`` / ``partitions`` / ``mesh_selection``) are
        carried over by reference — they are not affected by a single
        record append.
        """
        import copy
        new = copy.copy(self)
        if nodes is not None:
            new.nodes = nodes
        if elements is not None:
            new.elements = elements
        # Rewire the ElementComposite → NodeComposite back-reference
        # (FEMData.__init__ sets ``elements._apegmsh_nodes_ref = nodes``;
        # since we may have swapped one or both, re-pin it from the
        # final attributes).
        new.elements._apegmsh_nodes_ref = new.nodes
        # The InspectComposite holds a back-ref to the predecessor;
        # rebind it so e.g. ``new.inspect.summary()`` sees the new
        # record sets.
        new.inspect = InspectComposite(new)
        # Drop the snapshot_id cache — record set changed.
        if hasattr(new, "_snapshot_id_cache"):
            del new._snapshot_id_cache
        return new

    def with_constraint(self, record) -> "FEMData":
        """Return a new :class:`FEMData` with ``record`` appended.

        Pure transform.  ``self`` is unchanged.  Dispatch is by record
        type:

        =====================================  =================================
        Record subclass                        Appended to
        =====================================  =================================
        ``NodePairRecord``                     ``nodes.constraints``
        ``NodeGroupRecord``                    ``nodes.constraints``
        ``NodeToSurfaceRecord``                ``nodes.constraints``
        ``InterpolationRecord``                ``elements.constraints``
        ``SurfaceCouplingRecord``              ``elements.constraints``
        ``SPRecord``                           ``nodes.sp``
        =====================================  =================================

        Routing an unknown record subclass raises ``TypeError`` — this
        is a fail-loud contract because the compose engine needs every
        record to land in a known broker bucket.
        """
        # Imports are deferred so callers using only ``with_load`` /
        # ``with_mass`` don't pay the cost.
        from .._kernel.records._constraints import (
            NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
            InterpolationRecord, SurfaceCouplingRecord,
        )
        from .._kernel.records._loads import SPRecord

        if isinstance(record, (NodePairRecord, NodeGroupRecord,
                               NodeToSurfaceRecord)):
            new_set = self.nodes.constraints._with_record(record)
            return self._replaced(
                nodes=self._replaced_nodes(constraints=new_set))
        if isinstance(record, (InterpolationRecord, SurfaceCouplingRecord)):
            new_set = self.elements.constraints._with_record(record)
            return self._replaced(
                elements=self._replaced_elements(constraints=new_set))
        if isinstance(record, SPRecord):
            new_set = self.nodes.sp._with_record(record)
            return self._replaced(
                nodes=self._replaced_nodes(sp=new_set))
        raise TypeError(
            f"FEMData.with_constraint: unsupported record type "
            f"{type(record).__name__!r} — expected a ConstraintRecord "
            f"subclass or SPRecord."
        )

    def with_load(self, record) -> "FEMData":
        """Return a new :class:`FEMData` with ``record`` appended.

        Pure transform.  ``self`` is unchanged.  Dispatch is by record
        type:

        =====================================  =================================
        Record subclass                        Appended to
        =====================================  =================================
        ``NodalLoadRecord``                    ``nodes.loads``
        ``ElementLoadRecord``                  ``elements.loads``
        ``SPRecord``                           ``nodes.sp``
        =====================================  =================================
        """
        from .._kernel.records._loads import (
            NodalLoadRecord, ElementLoadRecord, SPRecord,
        )

        if isinstance(record, NodalLoadRecord):
            new_set = self.nodes.loads._with_record(record)
            return self._replaced(
                nodes=self._replaced_nodes(loads=new_set))
        if isinstance(record, ElementLoadRecord):
            new_set = self.elements.loads._with_record(record)
            return self._replaced(
                elements=self._replaced_elements(loads=new_set))
        if isinstance(record, SPRecord):
            new_set = self.nodes.sp._with_record(record)
            return self._replaced(
                nodes=self._replaced_nodes(sp=new_set))
        raise TypeError(
            f"FEMData.with_load: unsupported record type "
            f"{type(record).__name__!r} — expected NodalLoadRecord, "
            f"ElementLoadRecord, or SPRecord."
        )

    # ------------------------------------------------------------------
    # Compose primitive — Phase 3B.2c / ADR 0038
    # ------------------------------------------------------------------

    def compose(
        self,
        source: "str | Path",
        *,
        label: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float, float] | None = None,
        anchor: str | None = None,
        partition_rank: int | None = None,
        properties: "dict | None" = None,
        compose_size_per_module: int | None = None,
        max_compose_depth: int | None = None,
    ) -> "FEMData":
        """Return a new :class:`FEMData` extending this chain with a
        composed module.

        Pure transformation. ``self`` is unchanged.  The returned
        FEMData's ``composed_from`` chain is
        ``self.composed_from + (new_record,)`` and every IMPORT-verdict
        record from the source H5 surfaces on the result, namespaced
        with ``label`` and offset into a non-overlapping tag window
        per ADR 0038 §"Tag-offset scheme".

        Geometry / mesh build-phase operations are not part of this
        primitive — the merge runs entirely against the FEMData
        broker; no live Gmsh state is touched.  The compose API is
        the canonical entry point for cross-session composition:
        ``FEMData.from_h5(path).compose("module.h5", label="A")``.

        Drift hazard: the returned FEMData is decoupled from any
        live gmsh state on the producing session.  Mutating the
        session's mesh/PG/label/parts AFTER calling compose without
        also re-extracting + re-applying the bundle drops the
        composed module's records on the floor.  The
        :meth:`apeGmsh.compose` shim handles this via session-level
        bundle-replay; if you call this primitive directly, replay
        is your responsibility.

        See :func:`apeGmsh.mesh._compose.Compose.compose` for the full
        parameter contract; ``max_compose_depth`` is the only
        ``Compose.compose`` parameter intentionally absent here
        (depth checks come in Phase 3E.1).
        """
        from ._compose import (
            Compose, ComposeAnchorError,
            _compute_source_span, _compute_reservation,
            _rewrite_source_for_compose, _merge_bundle_into_fem,
            _emit_filter_warnings, _host_max_tag,
        )

        # 1. Validate inputs — reuse the Compose facade's static
        #    validators so the contract stays single-sourced.
        Compose._validate_label(label)
        Compose._validate_translate_rotate_anchor(translate, anchor)
        Compose._validate_partition_rank(partition_rank)
        Compose._validate_compose_size(compose_size_per_module)

        # 2. Resolve anchor → translate when anchor is given.  Anchor
        #    + non-zero translate already raised above; an unknown
        #    PG raises ComposeAnchorError with an actionable message.
        if anchor is not None:
            translate = _resolve_anchor_to_translate(self, anchor)

        # 3. Compute reservation against host's max_tag.  The merge
        #    engine doesn't track explicit (base, size) pairs on
        #    composed_from in this slice (deferred to 3D); each
        #    new compose simply rounds host_max_tag up to the next
        #    granularity boundary, which advances naturally as each
        #    bundle's tags fold into the host's id ranges.
        host_max_tag = _host_max_tag(self)
        source_span, source_min_tag, _source_max_tag = (
            _compute_source_span(source)
        )
        base, size = _compute_reservation(
            source_span=source_span,
            host_max_tag=host_max_tag,
            previous_reservations=(),
            granularity=Compose.RESERVATION_GRANULARITY,
            compose_size_per_module=compose_size_per_module,
        )

        # Capacity check (ADR 0038 §"Tag-collision verifier" check 5).
        if compose_size_per_module is not None and source_span > size:
            from ._compose import ComposeCapacityError
            raise ComposeCapacityError(
                f"compose(compose_size_per_module="
                f"{compose_size_per_module}) is smaller than the "
                f"source's tag span ({source_span}); reservation size "
                f"{size} would not fit the imported tags."
            )

        # 4. Rewrite the source into an offset/namespaced bundle.
        #    Phase 3E.1: forward max_compose_depth so the rewriter's
        #    depth gate fires before any rewrite work runs.  ``None``
        #    means "use the default" (DEFAULT_MAX_COMPOSE_DEPTH = 3).
        from ._compose import DEFAULT_MAX_COMPOSE_DEPTH as _DEFAULT_MAX_DEPTH
        depth_cap = (
            max_compose_depth
            if max_compose_depth is not None
            else _DEFAULT_MAX_DEPTH
        )
        bundle = _rewrite_source_for_compose(
            source_path=source,
            label=label,
            translate=translate,
            rotate=rotate,
            partition_rank=partition_rank,
            properties=dict(properties or {}),
            base=base,
            size=size,
            source_span=source_span,
            source_min_tag=source_min_tag,
            max_compose_depth=depth_cap,
        )

        # 5. Emit FILTER warnings (stages / time-series / patterns).
        _emit_filter_warnings(source, label)

        # 5b. Interface-size advisory (Phase 3F.1 / ADR 0038
        #     §"v1 scope gate").  Emits one ComposeInterfaceSizeWarning
        #     when the bundle's MP-style constraint count exceeds
        #     WARN_INTERFACE_SIZE = 50_000 — silenceable per-call via
        #     ``warnings.simplefilter("ignore", ComposeInterfaceSizeWarning)``.
        from ._compose import WARN_INTERFACE_SIZE, _warn_interface_size
        _warn_interface_size(bundle, threshold=WARN_INTERFACE_SIZE)

        # 6. Merge bundle into self → new FEMData.  The merge engine
        #    runs the Phase 2.2 tag-collision verifier first
        #    (ADR 0038 §"Tag-collision verifier") and raises a typed
        #    error on violation.  Cache the bundle on the result for
        #    the session shim's replay machinery.
        new_fem = _merge_bundle_into_fem(
            self, bundle, compose_size_per_module=compose_size_per_module,
        )

        # 7. Phase 3B.2d / ADR 0038 §"Rank model" — eager populator.
        #    Assign one partition per composed module (Layer 1) +
        #    honour partition_rank hints (Layer 2).  Existing
        #    METIS-driven partitions trigger a UserWarning (Layer 3).
        from ._compose import _rebuild_partitions_from_modules
        new_fem = _rebuild_partitions_from_modules(new_fem)

        new_fem._last_compose_bundle = bundle  # type: ignore[attr-defined]
        return new_fem

    def compose_tree(self) -> "tuple":
        """Derived nested-compose tree view of ``self.composed_from``.

        Reconstructs the nested-compose hierarchy from this FEMData's
        flat ``composed_from`` chain (per PR #369's flat-graft
        storage).  Returns a tuple of root
        :class:`~apeGmsh.mesh._compose.ComposeTreeNode` instances;
        each root carries its :class:`ComposeRecord` plus any direct
        children parsed from joined labels via the separator-
        alternation rule (depth-1 ``.``, depth-2 ``/``, depth-3
        ``.``, ...).

        Empty tuple when ``self.composed_from`` is empty (an
        uncomposed FEMData).
        """
        from ._compose import _build_compose_tree
        return _build_compose_tree(tuple(self.composed_from))

    def with_mass(self, record) -> "FEMData":
        """Return a new :class:`FEMData` with ``record`` appended to
        ``nodes.masses``.

        Pure transform.  ``self`` is unchanged.  Only
        :class:`~apeGmsh._kernel.records._masses.MassRecord` is
        accepted; anything else raises ``TypeError``.
        """
        from .._kernel.records._masses import MassRecord

        if isinstance(record, MassRecord):
            new_set = self.nodes.masses._with_record(record)
            return self._replaced(
                nodes=self._replaced_nodes(masses=new_set))
        raise TypeError(
            f"FEMData.with_mass: unsupported record type "
            f"{type(record).__name__!r} — expected MassRecord."
        )

    def viewer(self, *, blocking: bool = False) -> None:
        """Open a non-interactive mesh viewer from this snapshot.

        Currently disabled — the legacy ``Results.from_fem(...).viewer()``
        path was removed when the Results module was rebuilt. The new
        flow is being designed as part of the viewer rebuild project;
        see ``internal_docs/Results_architecture.md`` (Phase 9).
        """
        raise NotImplementedError(
            "fem.viewer() relied on the legacy Results class which has "
            "been rebuilt. The replacement is part of the viewer rebuild "
            "project (Phase 9 in Results_architecture.md). For mesh-only "
            "viewing in the meantime, use g.mesh.viewer() with no args."
        )

    def __repr__(self) -> str:
        return self.inspect.summary()


# =====================================================================
# Compose anchor resolution helper (Phase 3B.2c / ADR 0038)
# =====================================================================


def _resolve_anchor_to_translate(
    fem: FEMData,
    anchor: str,
) -> tuple[float, float, float]:
    """Resolve an anchor PG / label name to its centroid translation.

    Looks up ``anchor`` via the host's physical groups first, then
    labels.  Centroid is the mean of the resolved node coordinates.
    Raises :class:`ComposeAnchorError` when the name does not resolve.
    """
    from ._compose import ComposeAnchorError

    coords: np.ndarray | None = None
    try:
        coords = fem.nodes.physical.node_coords(anchor)
    except (KeyError, ValueError):
        coords = None
    if coords is None or len(coords) == 0:
        try:
            coords = fem.nodes.labels.node_coords(anchor)
        except (KeyError, ValueError):
            coords = None

    if coords is None or len(coords) == 0:
        raise ComposeAnchorError(
            f"compose(anchor={anchor!r}) did not resolve to a "
            f"non-empty physical group or label on the host. "
            f"Available PGs: "
            f"{list(getattr(fem.nodes.physical, '_groups', {}).keys())}; "
            f"labels: "
            f"{list(getattr(fem.nodes.labels, '_groups', {}).keys())}."
        )
    centroid = np.asarray(coords, dtype=np.float64).mean(axis=0)
    return (
        float(centroid[0]),
        float(centroid[1]),
        float(centroid[2]),
    )
