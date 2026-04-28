"""Multi-partition MPCO reader.

OpenSees parallel runs (OpenSeesMP / OpenSeesSP) produce one ``.mpco``
file per MPI rank, named ``<stem>.part-<N>.mpco`` with ``<N>`` from 0
to (n_ranks − 1). Each partition holds the slice of nodes and
elements that lived on its rank, plus its own ``MODEL`` group (so
mid-analysis a partition's MODEL describes only the rank-local
topology) and the corresponding ``RESULTS`` slice.

Boundary nodes are typically replicated across partitions — each rank
keeps a local copy with identical kinematics. Elements are disjoint:
each element lives on exactly one rank.

This module exposes :class:`MPCOMultiPartitionReader`, a thin façade
that wraps N :class:`MPCOReader` instances and implements the same
``ResultsReader`` protocol with stitching at read time:

- **Stages / time vector** — validated to match across partitions
  (a divergent stage list aborts construction loudly).
- **FEM** — union of nodes by ID (boundary duplicates collapse to a
  single row, first partition wins on coords) + concatenated
  per-element-class groups.
- **Component discovery** — union across partitions.
- **Slab reads** — node reads merge by ID; element/gauss/line-station/
  fiber/layer reads concatenate along the spatial axis.

Per the existing protocol, ``partitions(stage_id)`` returns one
``"partition_<i>"`` per wrapped reader. Callers that want
per-partition data can subset by element ID; the merge logic is
internal.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy import ndarray

from .._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
)
from ._mpco import MPCOReader
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData


# ``Recorder.part-0.mpco`` style; the partition number is captured.
_PARTITION_FILENAME_RE = re.compile(
    r"^(?P<stem>.+?)\.part-(?P<idx>\d+)\.mpco$"
)


def discover_partition_files(path: str | Path) -> list[Path]:
    """Find every ``<stem>.part-<N>.mpco`` sibling of ``path``.

    Returns the sorted list (by partition index). If ``path`` does
    not match the partition naming convention, returns ``[path]``
    unchanged.

    Examples
    --------
    Given ``Recorder.part-0.mpco`` in a directory that also holds
    ``Recorder.part-1.mpco`` and ``Recorder.part-2.mpco``, returns
    all three sorted by index.
    """
    p = Path(path)
    m = _PARTITION_FILENAME_RE.match(p.name)
    if m is None:
        return [p]
    stem = m.group("stem")
    parent = p.parent
    matches: list[tuple[int, Path]] = []
    for sib in parent.glob(f"{stem}.part-*.mpco"):
        sm = _PARTITION_FILENAME_RE.match(sib.name)
        if sm is None or sm.group("stem") != stem:
            continue
        matches.append((int(sm.group("idx")), sib))
    if not matches:
        return [p]
    matches.sort(key=lambda pair: pair[0])
    # Verify partitions are contiguous from 0; fall back to "just
    # this file" if anything is missing — partial partition sets are
    # almost certainly user error and shouldn't be silently merged.
    indices = [i for i, _ in matches]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"Partition files for stem {stem!r} are not contiguous "
            f"from 0 — found indices {indices}. Expected "
            f"{list(range(max(indices) + 1))}."
        )
    return [path for _, path in matches]


class MPCOMultiPartitionReader:
    """Façade over N :class:`MPCOReader` instances.

    Implements the :class:`apeGmsh.results.readers._protocol.ResultsReader`
    protocol structurally — duck-typed alongside ``MPCOReader`` and
    ``NativeReader``.
    """

    def __init__(self, paths: list[str | Path]) -> None:
        if not paths:
            raise ValueError(
                "MPCOMultiPartitionReader requires at least one path."
            )
        self._paths = [Path(p) for p in paths]
        self._readers: list[MPCOReader] = [
            MPCOReader(p) for p in self._paths
        ]
        try:
            self._validate_consistency()
        except Exception:
            for r in self._readers:
                r.close()
            raise
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _validate_consistency(self) -> None:
        """Stages and time vectors must match across all partitions."""
        per_reader_stages = [
            tuple((s.id, s.name, s.kind, s.n_steps) for s in r.stages())
            for r in self._readers
        ]
        first = per_reader_stages[0]
        for i, sigs in enumerate(per_reader_stages[1:], start=1):
            if sigs != first:
                raise ValueError(
                    f"Partition {i} ({self._paths[i].name}) reports "
                    f"different stage signatures than partition 0: "
                    f"got {sigs}, expected {first}."
                )
        # Time vectors must match per stage (same step count + values).
        for stage in self._readers[0].stages():
            t0 = self._readers[0].time_vector(stage.id)
            for i, r in enumerate(self._readers[1:], start=1):
                ti = r.time_vector(stage.id)
                if ti.shape != t0.shape:
                    raise ValueError(
                        f"Partition {i} time vector for stage "
                        f"{stage.name!r} has shape {ti.shape}; "
                        f"expected {t0.shape}."
                    )
                if not np.allclose(ti, t0):
                    raise ValueError(
                        f"Partition {i} time vector for stage "
                        f"{stage.name!r} differs from partition 0."
                    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        for r in self._readers:
            r.close()

    def __enter__(self) -> "MPCOMultiPartitionReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Protocol — stage / time / partitions
    # ------------------------------------------------------------------

    def stages(self) -> list[StageInfo]:
        return self._readers[0].stages()

    def time_vector(self, stage_id: str) -> ndarray:
        return self._readers[0].time_vector(stage_id)

    def partitions(self, stage_id: str) -> list[str]:
        return [f"partition_{i}" for i in range(len(self._readers))]

    # ------------------------------------------------------------------
    # FEM — merge across partitions
    # ------------------------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        if not isinstance(self._fem_cache, _Sentinel):
            return self._fem_cache
        per_partition = [r.fem() for r in self._readers]
        # If every partition is missing FEM, surface None.
        if all(f is None for f in per_partition):
            self._fem_cache = None
            return None
        merged = _merge_partition_fems(per_partition)
        self._fem_cache = merged
        return merged

    # ------------------------------------------------------------------
    # Component discovery — union across partitions
    # ------------------------------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        out: set[str] = set()
        for r in self._readers:
            out.update(r.available_components(stage_id, level))
        return sorted(out)

    # ------------------------------------------------------------------
    # Slab reads
    # ------------------------------------------------------------------

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        slabs = [
            r.read_nodes(
                stage_id, component,
                node_ids=node_ids, time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _merge_node_slabs(slabs, component)

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        slabs = [
            r.read_elements(
                stage_id, component,
                element_ids=element_ids, time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _concat_element_slabs(slabs, component)

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        slabs = [
            r.read_line_stations(
                stage_id, component,
                element_ids=element_ids, time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _concat_line_station_slabs(slabs, component)

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        slabs = [
            r.read_gauss(
                stage_id, component,
                element_ids=element_ids, time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _concat_gauss_slabs(slabs, component)

    def read_fibers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> FiberSlab:
        slabs = [
            r.read_fibers(
                stage_id, component,
                element_ids=element_ids,
                gp_indices=gp_indices,
                time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _concat_fiber_slabs(slabs, component)

    def read_layers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        layer_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LayerSlab:
        slabs = [
            r.read_layers(
                stage_id, component,
                element_ids=element_ids,
                gp_indices=gp_indices,
                layer_indices=layer_indices,
                time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _concat_layer_slabs(slabs, component)


# =====================================================================
# Stitching helpers — node merge, element concat
# =====================================================================

def _first_nonempty_time(slabs) -> ndarray:
    for s in slabs:
        if s.time.size:
            return s.time
    return np.array([], dtype=np.float64)


def _merge_node_slabs(slabs: list[NodeSlab], component: str) -> NodeSlab:
    """Union node IDs across partitions, fill values per-partition.

    Boundary nodes appear in multiple partitions with identical
    kinematics; we keep one row per unique ID. First partition that
    has the node wins (subsequent duplicates are ignored).
    """
    if not slabs:
        return _empty_node(component)
    time = _first_nonempty_time(slabs)
    T = int(time.size)
    pieces: list[tuple[ndarray, ndarray]] = []
    for sl in slabs:
        if sl.node_ids.size:
            pieces.append((sl.node_ids, sl.values))
    if not pieces:
        return NodeSlab(
            component=component,
            values=np.zeros((T, 0), dtype=np.float64),
            node_ids=np.array([], dtype=np.int64),
            time=time,
        )
    all_ids = np.concatenate([ids for ids, _ in pieces])
    master_ids = np.unique(all_ids)
    n_total = int(master_ids.size)
    id_to_col = {int(n): i for i, n in enumerate(master_ids)}
    values = np.full((T, n_total), np.nan, dtype=np.float64)
    for ids, vals in pieces:
        cols = np.array(
            [id_to_col[int(n)] for n in ids], dtype=np.int64,
        )
        # First-write wins: only fill columns that are still NaN.
        mask = np.isnan(values[0, cols])
        if mask.any():
            target_cols = cols[mask]
            values[:, target_cols] = vals[:, mask]
    return NodeSlab(
        component=component, values=values,
        node_ids=master_ids, time=time,
    )


def _concat_element_slabs(
    slabs: list[ElementSlab], component: str,
) -> ElementSlab:
    """Concatenate element slabs along the element axis (disjoint by partition)."""
    time = _first_nonempty_time(slabs)
    nonempty = [s for s in slabs if s.element_ids.size]
    if not nonempty:
        return ElementSlab(
            component=component,
            values=np.zeros((time.size, 0, 0), dtype=np.float64),
            element_ids=np.array([], dtype=np.int64),
            time=time,
        )
    return ElementSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_ids=np.concatenate([s.element_ids for s in nonempty]),
        time=time,
    )


def _concat_line_station_slabs(
    slabs: list[LineStationSlab], component: str,
) -> LineStationSlab:
    time = _first_nonempty_time(slabs)
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return LineStationSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            station_natural_coord=np.array([], dtype=np.float64),
            time=time,
        )
    return LineStationSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        station_natural_coord=np.concatenate(
            [s.station_natural_coord for s in nonempty],
        ),
        time=time,
    )


def _concat_gauss_slabs(
    slabs: list[GaussSlab], component: str,
) -> GaussSlab:
    time = _first_nonempty_time(slabs)
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return GaussSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            natural_coords=np.zeros((0, 3), dtype=np.float64),
            local_axes_quaternion=None,
            time=time,
        )
    # natural_coords may have different last-axis sizes if shells +
    # solids both contribute (3D vs 2D parent domains). Pad to the
    # widest dimension with zeros so the column-stack works.
    max_dim = max(s.natural_coords.shape[1] for s in nonempty)
    padded_coords: list[ndarray] = []
    for s in nonempty:
        if s.natural_coords.shape[1] == max_dim:
            padded_coords.append(s.natural_coords)
        else:
            pad_cols = max_dim - s.natural_coords.shape[1]
            padded_coords.append(np.hstack([
                s.natural_coords,
                np.zeros((s.natural_coords.shape[0], pad_cols), dtype=np.float64),
            ]))
    return GaussSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        natural_coords=np.concatenate(padded_coords, axis=0),
        local_axes_quaternion=None,
        time=time,
    )


def _concat_fiber_slabs(
    slabs: list[FiberSlab], component: str,
) -> FiberSlab:
    time = _first_nonempty_time(slabs)
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return FiberSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            gp_index=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.float64),
            z=np.array([], dtype=np.float64),
            area=np.array([], dtype=np.float64),
            material_tag=np.array([], dtype=np.int64),
            time=time,
        )
    return FiberSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        gp_index=np.concatenate([s.gp_index for s in nonempty]),
        y=np.concatenate([s.y for s in nonempty]),
        z=np.concatenate([s.z for s in nonempty]),
        area=np.concatenate([s.area for s in nonempty]),
        material_tag=np.concatenate([s.material_tag for s in nonempty]),
        time=time,
    )


def _concat_layer_slabs(
    slabs: list[LayerSlab], component: str,
) -> LayerSlab:
    time = _first_nonempty_time(slabs)
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return LayerSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            gp_index=np.array([], dtype=np.int64),
            layer_index=np.array([], dtype=np.int64),
            sub_gp_index=np.array([], dtype=np.int64),
            thickness=np.array([], dtype=np.float64),
            local_axes_quaternion=np.zeros((0, 4), dtype=np.float64),
            time=time,
        )
    return LayerSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        gp_index=np.concatenate([s.gp_index for s in nonempty]),
        layer_index=np.concatenate([s.layer_index for s in nonempty]),
        sub_gp_index=np.concatenate([s.sub_gp_index for s in nonempty]),
        thickness=np.concatenate([s.thickness for s in nonempty]),
        local_axes_quaternion=np.concatenate(
            [s.local_axes_quaternion for s in nonempty], axis=0,
        ),
        time=time,
    )


# =====================================================================
# FEM merge
# =====================================================================

def _merge_partition_fems(
    fems: list["Optional[FEMData]"],
) -> "Optional[FEMData]":
    """Merge per-partition FEMData snapshots into one model-wide snapshot.

    Strategy:

    - **Nodes**: union by ID. First partition that owns a node ID
      keeps its coordinates; later duplicates (boundary copies) are
      dropped.
    - **Elements**: per-class concatenation. Different partitions may
      hold disjoint elements of the same class — merge their groups.
    - **Physical groups / labels**: dropped from the merged FEMData.
      Per-partition MPCO regions describe rank-local node sets that
      don't correspond to anything meaningful at the merged level.
    """
    from ...mesh._element_types import ElementGroup
    from ...mesh._group_set import LabelSet, PhysicalGroupSet
    from ...mesh.FEMData import (
        ElementComposite, FEMData, MeshInfo, NodeComposite,
    )

    real = [f for f in fems if f is not None]
    if not real:
        return None
    if len(real) == 1:
        return real[0]

    # Node merge — union by ID, first-occurrence wins on coords.
    seen_ids: dict[int, int] = {}     # node_id → row in merged array
    merged_ids: list[int] = []
    merged_coords: list[ndarray] = []
    for f in real:
        ids = np.asarray(f.nodes.ids, dtype=np.int64)
        coords = np.asarray(f.nodes.coords, dtype=np.float64)
        for i, nid in enumerate(ids):
            int_nid = int(nid)
            if int_nid in seen_ids:
                continue
            seen_ids[int_nid] = len(merged_ids)
            merged_ids.append(int_nid)
            merged_coords.append(coords[i])
    node_ids_arr = np.array(merged_ids, dtype=np.int64)
    coords_arr = (
        np.stack(merged_coords, axis=0)
        if merged_coords
        else np.zeros((0, 3), dtype=np.float64)
    )

    # Element merge — per-class concatenation. ``code`` is the
    # negated class_tag (synthetic) and matches across partitions for
    # the same OpenSees class. ``ElementComposite`` exposes its
    # underlying type-keyed dict via ``_groups`` (no public accessor).
    merged_groups: dict[int, ElementGroup] = {}
    for f in real:
        for code, grp in f.elements._groups.items():
            if code in merged_groups:
                existing = merged_groups[code]
                merged_ids_e = np.concatenate(
                    [existing.ids, grp.ids],
                ).astype(np.int64)
                merged_conn = np.concatenate(
                    [existing.connectivity, grp.connectivity], axis=0,
                ).astype(np.int64)
                # Element type info — npe and dim are immutable; bump
                # the count on a fresh ElementTypeInfo.
                from ...mesh._element_types import make_type_info
                info = make_type_info(
                    code=existing.element_type.code,
                    gmsh_name=existing.element_type.gmsh_name,
                    dim=existing.element_type.dim,
                    order=existing.element_type.order,
                    npe=existing.element_type.npe,
                    count=int(merged_ids_e.size),
                )
                merged_groups[code] = ElementGroup(
                    element_type=info,
                    ids=merged_ids_e,
                    connectivity=merged_conn,
                )
            else:
                merged_groups[code] = grp

    nodes = NodeComposite(
        node_ids=node_ids_arr,
        node_coords=coords_arr,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups=merged_groups,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    n_elems = sum(int(g.ids.size) for g in merged_groups.values())
    info = MeshInfo(
        n_nodes=int(node_ids_arr.size),
        n_elems=n_elems,
        bandwidth=0,
        types=[g.element_type for g in merged_groups.values()],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


# =====================================================================
# Internal sentinel + empty helpers
# =====================================================================

class _Sentinel:
    pass


_SENTINEL = _Sentinel()


def _empty_node(component: str) -> NodeSlab:
    return NodeSlab(
        component=component,
        values=np.zeros((0, 0), dtype=np.float64),
        node_ids=np.array([], dtype=np.int64),
        time=np.array([], dtype=np.float64),
    )


__all__ = [
    "MPCOMultiPartitionReader",
    "discover_partition_files",
]
