"""Multi-partition MPCO reader.

OpenSees parallel runs (OpenSeesMP / OpenSeesSP) produce one ``.mpco``
file per MPI rank, named ``<stem>.part-<N>.mpco`` with ``<N>`` from 0
to (n_ranks − 1). Each partition holds the slice of nodes and
elements that lived on its rank, plus its own ``MODEL`` group (so
mid-analysis a partition's MODEL describes only the rank-local
topology) and the corresponding ``RESULTS`` slice.

Boundary nodes are typically replicated across partitions — each rank
keeps a local copy with identical kinematics, but a reaction or an
unbalanced load there is only that rank's share. Elements are disjoint:
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
- **Slab reads** — node reads merge by ID, keeping the first copy of
  kinematics and summing the copies of reactions and unbalanced loads
  (see :func:`_merge_node_slabs`); element/gauss/line-station/
  fiber/layer reads concatenate along the spatial axis.

Per the existing protocol, ``partitions(stage_id)`` returns one
``"partition_<i>"`` per wrapped reader. Callers that want
per-partition data can subset by element ID; the merge logic is
internal.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
from numpy import ndarray

from .._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
    SpringSlab,
)
from ._buckets import (
    RotationMask,
    _concat_element_slabs,
    _concat_fiber_slabs,
    _concat_gauss_slabs,
    _concat_layer_slabs,
    _concat_line_station_slabs,
    _concat_spring_slabs,
    _merge_node_slabs,
    _merge_partition_fems,
    _node_reduction,
    _validate_warning_once_per_mode,
)
from ._mpco import MPCOGaussDimensionMixed, MPCOReader
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData
    from ._buckets import RolesSource


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
        # Bound first, so ``close()`` / ``__del__`` find a list on a
        # facade whose construction failed.
        self._readers: list[MPCOReader] = []
        if not paths:
            raise ValueError(
                "MPCOMultiPartitionReader requires at least one path."
            )
        self._paths = [Path(p) for p in paths]
        # A part that fails to open, or a failed (or interrupted)
        # consistency read, closes every part opened so far and re-raises
        # the original error, so the part files stay unlocked.
        try:
            for p in self._paths:
                self._readers.append(MPCOReader(p))
            _validate_warning_once_per_mode(
                self._validate_consistency, self._paths,
            )
        except BaseException:
            self.close()
            raise
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL
        # One AbsentDofWarning per façade, not one per part.
        self._absent_dof_warned = [False]

    def attach_tag_map(self, tag_map: "Any") -> None:
        """Forward the fem_eid↔ops-tag translator to every child reader.

        ADR 0043 slice 1.3 — each child :class:`MPCOReader` relabels its
        own ops-keyed slabs to fem_eid space; this façade stitches the
        already-fem-keyed partition slabs, so it needs no translation of
        its own. The map is mirrored locally so introspection
        (``_tag_map``) sees the same state as on a single-file reader.
        """
        self._tag_map = tag_map
        for r in self._readers:
            r.attach_tag_map(tag_map)

    def attach_nodes_ndf(
        self, nodes_ndf: "dict[int, int] | None", ndm: int, *,
        roles: "RolesSource",
    ) -> None:
        """Forward the model's per-node ndf and DOF roles to every part
        (#2028), so a partitioned rotation read masks as a one-file one
        does. Each part checks the ndf against its own ``MODEL/NODES``;
        ``roles`` is one shared source, read once for every part."""
        for r in self._readers:
            r.attach_nodes_ndf(nodes_ndf, ndm, roles=roles)

    def spatial_dim(self) -> int:
        """Coordinate column count of the capture; partition 0 answers (#1393)."""
        return self._readers[0].spatial_dim()

    def attach_stage_names(
        self, names: "Sequence[str]", kinds: "Sequence[str]",
    ) -> None:
        """Forward the program's stage names and kinds to every child
        reader (#1324, #1948).

        :meth:`stages` answers from partition 0, but every child keeps
        its own stage list, so each is renamed — the construction-time
        consistency check already proved they agree.
        """
        for r in self._readers:
            r.attach_stage_names(names, kinds)

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
            try:
                r.close()
            except Exception:
                pass

    def __enter__(self) -> "MPCOMultiPartitionReader":
        return self

    def __del__(self) -> None:
        # Release every per-partition h5py handle on GC so jupyter
        # cell re-runs don't hold Windows file locks on the .mpco.
        self.close()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Protocol — stage / time / partitions
    # ------------------------------------------------------------------

    def stages(self) -> list[StageInfo]:
        return self._readers[0].stages()

    def time_vector(self, stage_id: str) -> ndarray:
        return self._readers[0].time_vector(stage_id)

    def _part_names(self) -> list[str]:
        """Each part's file name, in ``self._readers`` order: the labels
        a stitch refusal names."""
        return [p.name for p in self._paths]

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

    def opensees_model(self):
        """MPCO files carry no ``/opensees/`` zone — always ``None``.

        Mirrors :meth:`MPCOReader.opensees_model`. Multi-partition
        MPCO runs share the same constraint: the zone is unique to
        apeGmsh-produced native files, so any model handle must be
        threaded through ``model_h5=`` on :meth:`Results.from_mpco`.
        """
        return None

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
        with RotationMask.one_warning(self._absent_dof_warned):
            slabs = [
                r.read_nodes(
                    stage_id, component,
                    node_ids=node_ids, time_slice=time_slice,
                )
                for r in self._readers
            ]
        return _merge_node_slabs(
            slabs, component,
            _node_reduction(self._readers, self._paths, stage_id, component),
        )

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
        return _concat_element_slabs(slabs, component, self._part_names())

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
        return _concat_line_station_slabs(slabs, component, self._part_names())

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
        return _concat_gauss_slabs(
            slabs, component, self._part_names(),
            reader="MPCO", mixed_error=MPCOGaussDimensionMixed,
        )

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
        return _concat_fiber_slabs(slabs, component, self._part_names())

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
        return _concat_layer_slabs(slabs, component, self._part_names())

    def read_springs(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> SpringSlab:
        slabs = [
            r.read_springs(
                stage_id, component,
                element_ids=element_ids,
                time_slice=time_slice,
            )
            for r in self._readers
        ]
        return _concat_spring_slabs(slabs, component, self._part_names())


# =====================================================================
# Stitching helpers — node merge, element concat
# =====================================================================


# =====================================================================
# FEM merge
# =====================================================================


# =====================================================================
# Internal sentinel + empty helpers
# =====================================================================

class _Sentinel:
    pass


_SENTINEL = _Sentinel()


__all__ = [
    "MPCOMultiPartitionReader",
    "discover_partition_files",
]
