"""
Hand-rolled FEMData-shaped stub for Phase 4 integration / parity tests.

Mirrors the surface the build pipeline uses:

  * ``fem.nodes.index(node_id)`` -> int (array index)
  * ``fem.nodes.coords`` -> ndarray (N, 3)
  * ``fem.nodes.select(pg=name).ids/.coords`` -> ndarray
    (selection-unification v2 P3-R migration target; ``.get`` removed)
  * ``fem.elements.select(pg=name).groups()`` -> iterable yielding
    objects that iterate as ``(eid, conn_tuple)`` pairs; the same
    terminal also answers ``.result().resolve()`` -> ``(ids, conn)``

Building a real :class:`apeGmsh.mesh.FEMData` requires a live Gmsh
session; the bridge talks to FEMData through a narrow surface, so a
stub keeps tests fast and pure-Python.

The stub is intentionally read-only and mutation-free; tests build a
fresh instance per test or reuse a session-scoped one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass(frozen=True)
class _NodeResult:
    """Stand-in for :class:`apeGmsh.mesh.FEMData.NodeResult`.

    Carries the ``ids`` and ``coords`` arrays; nothing else is read by
    the build pipeline.
    """

    ids: ndarray
    coords: ndarray


@dataclass(frozen=True)
class _ElementGroupView:
    """Stand-in for :class:`apeGmsh.mesh._element_types.ElementGroup`.

    Iterates as ``(eid, conn_tuple)`` pairs, matching ElementGroup's
    iter contract used by :func:`expand_pg_to_elements`.
    """

    ids: tuple[int, ...]
    connectivity: tuple[tuple[int, ...], ...]

    def __iter__(self):
        for eid, conn in zip(self.ids, self.connectivity):
            yield eid, conn

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class _GroupResultView:
    """Iterable of :class:`_ElementGroupView` — one per element type.

    The build pipeline iterates this directly.  selection-unification
    v2 P3-R: the broker's removed ``fem.elements.get(...)`` is replaced
    by ``fem.elements.select(...)`` whose terminal exposes ``.groups()``
    (iterable of per-type groups) and ``.result().resolve()`` →
    ``(ids, conn)``.  This view backs both: it *is* the iterable of
    groups (legacy iter contract, unchanged) and also answers
    ``.groups()`` / ``.result()`` / ``.resolve()`` for the new surface.
    """

    _groups: tuple[_ElementGroupView, ...]

    def __iter__(self):
        return iter(self._groups)

    def __bool__(self) -> bool:
        return len(self._groups) > 0

    # ── new broker surface (P3-R: .select(...).groups() / .result()) ──
    def groups(self) -> tuple[_ElementGroupView, ...]:
        return self._groups

    def result(self) -> "_GroupResultView":
        return self

    def resolve(self):
        import numpy as _np

        ids: list[int] = []
        conn: list[tuple[int, ...]] = []
        for grp in self._groups:
            ids.extend(int(e) for e in grp.ids)
            conn.extend(tuple(int(n) for n in c) for c in grp.connectivity)
        return (
            _np.asarray(ids, dtype=_np.int64),
            _np.asarray(conn, dtype=_np.int64) if conn else _np.empty((0, 0), dtype=_np.int64),
        )


class _NodeLabelsStub:
    """Stand-in for ``fem.nodes.labels`` — exposes ``node_ids(name)``."""

    def __init__(self, label_to_ids: dict[str, list[int]]) -> None:
        self._labels = {k: list(v) for k, v in label_to_ids.items()}

    def node_ids(self, name: str) -> ndarray:
        if name not in self._labels:
            raise KeyError(
                f"node label {name!r} not in fixture; have "
                f"{sorted(self._labels.keys())}"
            )
        return np.asarray(self._labels[name], dtype=np.int64)


class _ElementLabelsStub:
    """Stand-in for ``fem.elements.labels`` — exposes ``element_ids(name)``."""

    def __init__(self, label_to_ids: dict[str, list[int]]) -> None:
        self._labels = {k: list(v) for k, v in label_to_ids.items()}

    def element_ids(self, name: str) -> ndarray:
        if name not in self._labels:
            raise KeyError(
                f"element label {name!r} not in fixture; have "
                f"{sorted(self._labels.keys())}"
            )
        return np.asarray(self._labels[name], dtype=np.int64)


class _MeshSelectionStub:
    """Stand-in for ``fem.mesh_selection`` — exposes ``node_ids`` and
    ``element_ids`` keyed by selection set name."""

    def __init__(
        self,
        nodes: dict[str, list[int]] | None = None,
        elements: dict[str, list[int]] | None = None,
    ) -> None:
        self._nodes = {k: list(v) for k, v in (nodes or {}).items()}
        self._elements = {k: list(v) for k, v in (elements or {}).items()}

    def node_ids(self, name: str) -> ndarray:
        if name not in self._nodes:
            raise KeyError(
                f"node selection {name!r} not in fixture; have "
                f"{sorted(self._nodes.keys())}"
            )
        return np.asarray(self._nodes[name], dtype=np.int64)

    def element_ids(self, name: str) -> ndarray:
        if name not in self._elements:
            raise KeyError(
                f"element selection {name!r} not in fixture; have "
                f"{sorted(self._elements.keys())}"
            )
        return np.asarray(self._elements[name], dtype=np.int64)


class _NodesStub:
    """Stand-in for :class:`apeGmsh.mesh.FEMData.NodeComposite`."""

    def __init__(
        self,
        ids: list[int],
        coords: list[tuple[float, float, float]],
        node_pgs: dict[str, list[int]],
        labels: dict[str, list[int]] | None = None,
    ) -> None:
        self._ids = np.asarray(ids, dtype=np.int64)
        self._coords = np.asarray(coords, dtype=np.float64)
        self._id_to_idx = {int(n): i for i, n in enumerate(self._ids)}
        self._pgs = {k: list(v) for k, v in node_pgs.items()}
        self.labels = _NodeLabelsStub(labels or {})

    @property
    def ids(self) -> ndarray:
        return self._ids

    @property
    def coords(self) -> ndarray:
        return self._coords

    def index(self, node_id: int) -> int:
        try:
            return self._id_to_idx[int(node_id)]
        except KeyError as e:
            raise KeyError(
                f"node id {node_id} not in fixture; have "
                f"{sorted(self._id_to_idx.keys())}"
            ) from e

    def get(
        self,
        target: object | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | None = None,
        partition: int | None = None,
        dim: int | None = None,
    ) -> _NodeResult:
        if pg is None:
            raise ValueError("fem-stub.nodes.get: only pg= is supported")
        if pg not in self._pgs:
            raise KeyError(
                f"node pg {pg!r} not in fixture; have {sorted(self._pgs.keys())}"
            )
        ids = np.asarray(self._pgs[pg], dtype=object)
        idxs = [self._id_to_idx[int(n)] for n in ids]
        coords = self._coords[idxs] if idxs else np.empty((0, 3))
        return _NodeResult(ids=ids, coords=np.asarray(coords))

    def select(
        self,
        target: object | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | None = None,
        partition: int | None = None,
        dim: int | None = None,
    ) -> _NodeResult:
        """selection-unification v2 P3-R: ``fem.nodes.get`` is removed;
        ``fem.nodes.select(pg=).ids/.coords`` is the migration target
        (P-NODE/P-COORD).  The stub mirrors the broker — behaviour
        byte-identical to the (now removed) ``.get`` body."""
        return self.get(
            target, pg=pg, label=label, tag=tag,
            partition=partition, dim=dim,
        )

    def ndf_for(self, nid: int) -> int:
        """Mirror the real :meth:`NodeComposite.ndf_for` fail-loud
        behaviour for stub-based bridge tests (S2).

        The stub never carries declared ndf — there is no
        ``g.node_ndf`` composite behind it.  Raises ``LookupError`` so
        the S2 emit-site ``try/except LookupError`` path fires and
        falls back to the envelope ``ndf=K`` from ``apeSees.model``.
        DO NOT return 0 or the envelope value — that would mask
        emit-side bugs that the real broker would have caught.
        """
        raise LookupError(
            f"test stub: no per-node ndf declared (node {nid})"
        )


class _ElementsStub:
    """Stand-in for :class:`apeGmsh.mesh.FEMData.ElementComposite`."""

    def __init__(
        self,
        elem_pgs: dict[str, _ElementGroupView],
        labels: dict[str, list[int]] | None = None,
    ) -> None:
        self._pgs = dict(elem_pgs)
        self.labels = _ElementLabelsStub(labels or {})

    def get(
        self,
        target: object | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | None = None,
        dim: int | None = None,
        element_type: str | int | None = None,
        partition: int | None = None,
    ) -> _GroupResultView:
        if pg is None:
            raise ValueError("fem-stub.elements.get: only pg= is supported")
        if pg not in self._pgs:
            raise KeyError(
                f"element pg {pg!r} not in fixture; have "
                f"{sorted(self._pgs.keys())}"
            )
        return _GroupResultView(_groups=(self._pgs[pg],))

    def select(
        self,
        target: object | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | None = None,
        dim: int | None = None,
        element_type: str | int | None = None,
        partition: int | None = None,
    ) -> _GroupResultView:
        """selection-unification v2 P3-R: ``fem.elements.get`` is
        removed; ``fem.elements.select(pg=).groups()`` /
        ``.result().resolve()`` is the migration target
        (P-GROUPRESULT).  The stub mirrors the broker — behaviour
        byte-identical to the (now removed) ``.get`` body; the returned
        view answers both the legacy iter and the new
        ``.groups()``/``.result()``/``.resolve()`` surface."""
        return self.get(
            target, pg=pg, label=label, tag=tag, dim=dim,
            element_type=element_type, partition=partition,
        )


class _ConstraintsStub:
    """Stand-in for a constraint set (NodeConstraintSet or
    SurfaceConstraintSet).

    Carries a flat list of records and mirrors the broker's typed
    iterators :meth:`pairs` / :meth:`equal_dofs` /
    :meth:`rigid_link_groups` / :meth:`rigid_diaphragms` /
    :meth:`node_to_surfaces` / :meth:`stiff_beam_groups` /
    :meth:`phantom_nodes` (node side) and :meth:`interpolations` /
    :meth:`couplings` (surface side).  Implementation copied from
    :mod:`apeGmsh._kernel.record_sets` so the bridge build pipeline
    sees the same surface in tests as in production.
    """

    def __init__(self, records: list | None = None) -> None:
        self._records: list = list(records) if records else []

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0

    def by_kind(self, kind: str) -> list:
        return [r for r in self._records if getattr(r, "kind", None) == kind]

    # ── Node-side iterators ────────────────────────────────────

    def pairs(self):
        from apeGmsh._kernel.records._constraints import (
            NodeGroupRecord, NodePairRecord, NodeToSurfaceRecord,
        )
        for rec in self._records:
            if isinstance(rec, NodePairRecord):
                yield rec
            elif isinstance(rec, NodeGroupRecord):
                yield from rec.expand_to_pairs()
            elif isinstance(rec, NodeToSurfaceRecord):
                yield from rec.expand()

    def node_to_surfaces(self):
        from apeGmsh._kernel.records._constraints import NodeToSurfaceRecord
        for rec in self._records:
            if isinstance(rec, NodeToSurfaceRecord):
                yield rec

    def equal_dofs(self):
        from apeGmsh._kernel.records._constraints import (
            NodePairRecord, NodeToSurfaceRecord,
        )
        from apeGmsh._kernel.records._kinds import ConstraintKind
        for rec in self._records:
            if (
                isinstance(rec, NodePairRecord)
                and rec.kind == ConstraintKind.EQUAL_DOF
            ):
                yield rec
            elif isinstance(rec, NodeToSurfaceRecord):
                yield from rec.equal_dof_records

    def rigid_diaphragms(self):
        from apeGmsh._kernel.records._constraints import NodeGroupRecord
        from apeGmsh._kernel.records._kinds import ConstraintKind
        from apeGmsh._kernel.record_sets import _perp_dirn
        for rec in self._records:
            if (
                isinstance(rec, NodeGroupRecord)
                and rec.kind == ConstraintKind.RIGID_DIAPHRAGM
            ):
                yield (
                    _perp_dirn(rec.plane_normal),
                    int(rec.master_node),
                    [int(s) for s in rec.slave_nodes],
                )

    # ── Surface-side iterators ─────────────────────────────────

    def interpolations(self):
        from apeGmsh._kernel.records._constraints import (
            InterpolationRecord, SurfaceCouplingRecord,
        )
        for rec in self._records:
            if isinstance(rec, InterpolationRecord):
                yield rec
            elif isinstance(rec, SurfaceCouplingRecord):
                yield from rec.slave_records


@dataclass(frozen=True)
class _PartitionStub:
    """Stand-in for :class:`apeGmsh._kernel.records._partitions.PartitionRecord`.

    Used by ADR 0027 (P4) integration tests that drive
    :class:`apeSees` over a partitioned FEM stub.
    """
    id: int
    node_ids: tuple[int, ...]
    element_ids: tuple[int, ...]


class _PartitionSetStub:
    """Stand-in for :class:`apeGmsh._kernel.record_sets.PartitionSet`.

    Iterates in ascending id order; supports ``len()`` for the
    ``len(fem.partitions) > 1`` partitioned-emit gate.
    """
    def __init__(self, records: list[_PartitionStub] | None = None) -> None:
        recs = list(records or [])
        recs.sort(key=lambda r: int(r.id))
        self._records = recs

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)


class FEMStub:
    """Hand-rolled FEMData-shaped fixture for Phase-4 tests."""

    def __init__(
        self,
        nodes: _NodesStub,
        elements: _ElementsStub,
        mesh_selection: _MeshSelectionStub | None = None,
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.mesh_selection = mesh_selection
        # Phase 7b — constraint sub-composites.  Tests that exercise
        # the MP-constraint emission pass populate these via
        # :meth:`add_node_constraints` / :meth:`add_surface_constraints`.
        # Default-empty so existing tests that don't touch constraints
        # see a vacant surface (the build pipeline no-ops on empty
        # constraint sets).
        self.nodes.constraints = _ConstraintsStub()
        self.elements.constraints = _ConstraintsStub()
        # ADR 0027 (P4) — partitions composite. Default empty
        # (unpartitioned model). Tests that need a partitioned FEM
        # populate it via :meth:`set_partitions`.
        self.partitions: _PartitionSetStub = _PartitionSetStub()

    def add_node_constraints(self, records: list) -> None:
        """Install a list of node-side constraint records.

        Records may mix :class:`NodePairRecord`, :class:`NodeGroupRecord`,
        and :class:`NodeToSurfaceRecord`.  The build pipeline walks
        them via the typed iterators on :class:`_ConstraintsStub`.
        """
        self.nodes.constraints = _ConstraintsStub(records)

    def add_surface_constraints(self, records: list) -> None:
        """Install a list of element-side surface-coupling records.

        Records may mix :class:`InterpolationRecord` and
        :class:`SurfaceCouplingRecord`.
        """
        self.elements.constraints = _ConstraintsStub(records)

    def set_partitions(
        self, partitions: list[tuple[int, list[int], list[int]]],
    ) -> None:
        """Install partitions on the stub.

        Each ``(id, node_ids, element_ids)`` tuple becomes one
        :class:`_PartitionStub`; the resulting :class:`_PartitionSetStub`
        replaces ``self.partitions``. The unpartitioned default (empty)
        keeps the flat-emit code path; setting two or more partitions
        flips the bridge into the partition-emit branch (ADR 0027).
        """
        self.partitions = _PartitionSetStub([
            _PartitionStub(
                id=int(pid),
                node_ids=tuple(int(n) for n in nids),
                element_ids=tuple(int(e) for e in eids),
            )
            for pid, nids, eids in partitions
        ])


def make_two_node_beam() -> FEMStub:
    """Two nodes + one line element, both in PGs ``"Cols"`` and base.

    Geometry:
      * node 1 at origin
      * node 2 at (0, 0, 1) — vertical column

    PGs:
      * ``"Cols"``: element 1 (the vertical line)
      * ``"Base"``: node 1
      * ``"Top"``:  node 2
    """
    nodes = _NodesStub(
        ids=[1, 2],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        node_pgs={"Base": [1], "Top": [2]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1,), connectivity=((1, 2),),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)


def make_two_column_frame() -> FEMStub:
    """Two parallel columns sharing a common base PG.

    Geometry:
      * node 1 at (0, 0, 0)   - base of column A
      * node 2 at (0, 0, 1)   - top of column A
      * node 3 at (1, 0, 0)   - base of column B
      * node 4 at (1, 0, 1)   - top of column B

    PGs:
      * ``"Cols"``: elements 1 and 2 (both vertical columns)
      * ``"Base"``: nodes 1 and 3
      * ``"Top"``:  nodes 2 and 4
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
        ],
        node_pgs={"Base": [1, 3], "Top": [2, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1, 2),
                connectivity=((1, 2), (3, 4)),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)


def make_two_column_frame_with_labels_and_selection() -> FEMStub:
    """Same geometry as :func:`make_two_column_frame`, plus apeGmsh
    labels and a mesh_selection store.

    Adds:
      * node labels: ``"east_column"`` -> nodes 3, 4
      * element labels: ``"east_column"`` -> element 2
      * mesh_selection nodes: ``"upper_band"`` -> nodes 2, 4
      * mesh_selection elements: ``"upper_band"`` -> elements 1, 2
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
        ],
        node_pgs={"Base": [1, 3], "Top": [2, 4]},
        labels={"east_column": [3, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1, 2),
                connectivity=((1, 2), (3, 4)),
            ),
        },
        labels={"east_column": [2]},
    )
    mesh_selection = _MeshSelectionStub(
        nodes={"upper_band": [2, 4]},
        elements={"upper_band": [1, 2]},
    )
    return FEMStub(
        nodes=nodes,
        elements=elements,
        mesh_selection=mesh_selection,
    )


def make_two_column_frame_partitioned() -> FEMStub:
    """Same geometry as :func:`make_two_column_frame`, partitioned into 2.

    Partition 0 owns column A (nodes 1, 2; element 1).
    Partition 1 owns column B (nodes 3, 4; element 2).
    Used by ADR 0027 (P4) integration tests.
    """
    stub = make_two_column_frame()
    stub.set_partitions([
        (0, [1, 2], [1]),
        (1, [3, 4], [2]),
    ])
    return stub


def make_arch_with_orientation_fan_out() -> FEMStub:
    """Three-segment arch — non-collinear elements drive distinct vecxz under
    cylindrical orientation.

    Geometry: three line elements arranged so the unit tangent rotates
    between elements — a coarse approximation of an arch in the X-Z
    plane.

      * node 1 at (1, 0, 0)
      * node 2 at (cos 30°, 0, sin 30°)
      * node 3 at (cos 60°, 0, sin 60°)
      * node 4 at (0, 0, 1)

    Each consecutive pair forms one element.

    PGs:
      * ``"Arch"``: elements 1, 2, 3 (the three arch segments)
    """
    import math
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (1.0, 0.0, 0.0),
            (math.cos(math.pi / 6), 0.0, math.sin(math.pi / 6)),
            (math.cos(math.pi / 3), 0.0, math.sin(math.pi / 3)),
            (0.0, 0.0, 1.0),
        ],
        node_pgs={"Crown": [4], "Springing": [1]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Arch": _ElementGroupView(
                ids=(1, 2, 3),
                connectivity=((1, 2), (2, 3), (3, 4)),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)
