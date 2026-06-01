"""Multi-partition ``.ladruno`` reader (recorder plan L2b-2).

Parallel OpenSees runs write one ``.ladruno`` per MPI rank, named
``<stem>.part-<N>.ladruno`` (``INFO/PARTITIONED=1`` +
``PARTITION_ID``/``NUM_PARTITIONS`` manifest). This façade wraps N
:class:`LadrunoReader` instances and implements the ``ResultsReader``
protocol with read-time stitching — the sibling of
:class:`apeGmsh.results.readers._mpco_multi.MPCOMultiPartitionReader`.

The stitch logic (node-union, element-concat, FEM merge) is solver
neutral, so the heavy lifting is **reused verbatim** from ``_mpco_multi``
rather than re-implemented. What differs from MPCO is only the per-file
reader class and the ``.ladruno`` partition-filename grammar.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

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
from ._ladruno import LadrunoReader
from ._mpco_multi import (
    _concat_element_slabs,
    _concat_fiber_slabs,
    _concat_gauss_slabs,
    _concat_layer_slabs,
    _concat_line_station_slabs,
    _concat_spring_slabs,
    _merge_node_slabs,
    _merge_partition_fems,
)
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData


_PARTITION_FILENAME_RE = re.compile(
    r"^(?P<stem>.+?)\.part-(?P<idx>\d+)\.ladruno$"
)


def discover_partition_files(path: str | Path) -> list[Path]:
    """Find every ``<stem>.part-<N>.ladruno`` sibling of ``path``.

    Returns the sorted (by index) list. If ``path`` doesn't follow the
    partition naming convention, returns ``[path]`` unchanged. A gap in
    the partition indices raises (a partial set is almost certainly user
    error and shouldn't be silently merged) — mirrors ``_mpco_multi``.
    """
    p = Path(path)
    m = _PARTITION_FILENAME_RE.match(p.name)
    if m is None:
        return [p]
    stem = m.group("stem")
    matches: list[tuple[int, Path]] = []
    for sib in p.parent.glob(f"{stem}.part-*.ladruno"):
        sm = _PARTITION_FILENAME_RE.match(sib.name)
        if sm is None or sm.group("stem") != stem:
            continue
        matches.append((int(sm.group("idx")), sib))
    if not matches:
        return [p]
    matches.sort(key=lambda pair: pair[0])
    indices = [i for i, _ in matches]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"Partition files for stem {stem!r} are not contiguous from 0 "
            f"— found indices {indices}. Expected "
            f"{list(range(max(indices) + 1))}."
        )
    return [path for _, path in matches]


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class LadrunoMultiPartitionReader:
    """Façade over N :class:`LadrunoReader` instances (structural protocol)."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError(
                "LadrunoMultiPartitionReader requires at least one path."
            )
        self._paths = [Path(p) for p in paths]
        self._readers: list[LadrunoReader] = [
            LadrunoReader(p) for p in self._paths
        ]
        try:
            self._validate_consistency()
        except Exception:
            for r in self._readers:
                r.close()
            raise
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL

    def attach_tag_map(self, tag_map) -> None:
        for r in self._readers:
            r.attach_tag_map(tag_map)

    def _validate_consistency(self) -> None:
        sigs = [
            tuple((s.id, s.name, s.kind, s.n_steps) for s in r.stages())
            for r in self._readers
        ]
        first = sigs[0]
        for i, s in enumerate(sigs[1:], start=1):
            if s != first:
                raise ValueError(
                    f"Partition {i} ({self._paths[i].name}) reports different "
                    f"stage signatures than partition 0: {s} vs {first}."
                )
        for stage in self._readers[0].stages():
            t0 = self._readers[0].time_vector(stage.id)
            for i, r in enumerate(self._readers[1:], start=1):
                ti = r.time_vector(stage.id)
                if ti.shape != t0.shape or not np.allclose(ti, t0):
                    raise ValueError(
                        f"Partition {i} time vector for stage {stage.name!r} "
                        f"differs from partition 0."
                    )

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        for r in self._readers:
            try:
                r.close()
            except Exception:
                pass

    def __enter__(self) -> "LadrunoMultiPartitionReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # -- stages / time / partitions ------------------------------------

    def stages(self) -> list[StageInfo]:
        return self._readers[0].stages()

    def time_vector(self, stage_id: str) -> ndarray:
        return self._readers[0].time_vector(stage_id)

    def partitions(self, stage_id: str) -> list[str]:
        return [f"partition_{i}" for i in range(len(self._readers))]

    # -- model / fem ---------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        if not isinstance(self._fem_cache, _Sentinel):
            return self._fem_cache  # type: ignore[return-value]
        per = [r.fem() for r in self._readers]
        merged = None if all(f is None for f in per) else _merge_partition_fems(per)
        self._fem_cache = merged
        return merged

    def opensees_model(self):
        """The minimal broker is built from partition 0's MODEL.

        Mirrors the single-file self-sufficient path; richer lineage
        still comes via ``model_h5=`` on :meth:`Results.from_ladruno`.
        """
        return self._readers[0].opensees_model()

    # -- components / reads --------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        out: set[str] = set()
        for r in self._readers:
            out.update(r.available_components(stage_id, level))
        return sorted(out)

    def read_nodes(
        self, stage_id: str, component: str, *,
        node_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> NodeSlab:
        return _merge_node_slabs(
            [r.read_nodes(stage_id, component, node_ids=node_ids,
                          time_slice=time_slice) for r in self._readers],
            component,
        )

    def read_elements(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> ElementSlab:
        return _concat_element_slabs(
            [r.read_elements(stage_id, component, element_ids=element_ids,
                             time_slice=time_slice) for r in self._readers],
            component,
        )

    def read_line_stations(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        return _concat_line_station_slabs(
            [r.read_line_stations(stage_id, component, element_ids=element_ids,
                                  time_slice=time_slice) for r in self._readers],
            component,
        )

    def read_gauss(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> GaussSlab:
        return _concat_gauss_slabs(
            [r.read_gauss(stage_id, component, element_ids=element_ids,
                          time_slice=time_slice) for r in self._readers],
            component,
        )

    def read_fibers(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> FiberSlab:
        return _concat_fiber_slabs(
            [r.read_fibers(stage_id, component, element_ids=element_ids,
                           gp_indices=gp_indices, time_slice=time_slice)
             for r in self._readers],
            component,
        )

    def read_layers(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        layer_indices: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> LayerSlab:
        return _concat_layer_slabs(
            [r.read_layers(stage_id, component, element_ids=element_ids,
                           gp_indices=gp_indices, layer_indices=layer_indices,
                           time_slice=time_slice) for r in self._readers],
            component,
        )

    def read_springs(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> SpringSlab:
        return _concat_spring_slabs(
            [r.read_springs(stage_id, component, element_ids=element_ids,
                            time_slice=time_slice) for r in self._readers],
            component,
        )


__all__ = ["LadrunoMultiPartitionReader", "discover_partition_files"]
