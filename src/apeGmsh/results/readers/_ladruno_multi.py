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
reader class and the ``.ladruno`` partition-filename grammar. The node
merge takes each result's ``PARTITION_REDUCTION`` attribute from the part
files (reactions sum across partitions, kinematics keep one copy).
"""
from __future__ import annotations

import re
import warnings
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
from ._ladruno_element_io import MissingElementResults
from ._mpco_multi import (
    _concat_element_slabs,
    _concat_fiber_slabs,
    _concat_gauss_slabs,
    _concat_layer_slabs,
    _concat_line_station_slabs,
    _concat_spring_slabs,
    _merge_node_slabs,
    _merge_partition_fems,
    _node_reduction,
)
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData


class StageOrderMatchWarning(UserWarning):
    """Part files named their stages differently; stages were paired by order.

    Fork WP-165 starts a ``MODEL_STAGE`` only on a topology change, but a
    rank-local change can still give ranks different ``MODEL_STAGE[<n>]``
    stamps for one logical stage. With equal stage counts the reader pairs
    the stages by numeric stamp order and warns with this class.
    """


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
        # Mirror the map locally so introspection (``_tag_map``) sees the
        # same state on the façade as on a single-file reader.
        self._tag_map = tag_map
        for r in self._readers:
            r.attach_tag_map(tag_map)

    def _validate_part_set(self) -> None:
        """Refuse a part set mixed from different runs (fork WP-165).

        Every part file must report the same ``NUM_PARTITIONS``, equal to
        the number of files, and ``PARTITION_ID`` must cover ``0..N-1``.
        When the files carry a shared run identity (``RUN_ID_SCOPE`` other
        than ``"process"``), ``RUN_ID`` must match too. A mismatch means
        stale ``.part-N`` files from an earlier run sit next to the new
        ones. Files that predate an attribute skip that check.
        """
        mans = [r.partition_manifest() for r in self._readers]
        names = [p.name for p in self._paths]
        n = len(self._readers)

        def listing(key: str) -> str:
            return ", ".join(f"{nm}={m[key]!r}" for nm, m in zip(names, mans))

        nums = [m["NUM_PARTITIONS"] for m in mans]
        if any(v is not None for v in nums):
            if any(v != n for v in nums):
                raise ValueError(
                    f"Partitioned .ladruno set of {n} file(s) disagrees on "
                    f"NUM_PARTITIONS (expected {n} in every file): "
                    f"{listing('NUM_PARTITIONS')}. Stale part files from "
                    "an earlier run are probably mixed in; delete them and "
                    "re-run, or pass the exact file list."
                )
            ids = [m["PARTITION_ID"] for m in mans]
            if all(v is not None for v in ids) and sorted(ids) != list(range(n)):
                raise ValueError(
                    f"Partitioned .ladruno set has PARTITION_ID values "
                    f"{listing('PARTITION_ID')}; expected each of "
                    f"0..{n - 1} exactly once."
                )

        scopes = [m["RUN_ID_SCOPE"] for m in mans]
        run_ids = [m["RUN_ID"] for m in mans]
        if all(v is not None for v in run_ids) and not any(
            s is None or s == "process" for s in scopes
        ):
            if len(set(run_ids)) > 1:
                raise ValueError(
                    "Partitioned .ladruno files come from different runs "
                    f"(RUN_ID differs): {listing('RUN_ID')}. Stale part "
                    "files from an earlier run are mixed in; delete them "
                    "and re-run, or pass the exact file list."
                )

    def _validate_consistency(self) -> None:
        self._validate_part_set()
        self._pair_stages()
        # Step counts and time vectors are compared only across the parts
        # that hold results for the stage: an EMPTY_PARTITION stage (fork
        # WP-165) may carry no TIME axis at all.
        for stage in self._readers[0].stages():
            live = self._live(stage.id)
            ref_i, ref = live[0]
            t0 = ref.time_vector(stage.id)
            for i, r in live[1:]:
                ti = r.time_vector(stage.id)
                if ti.shape != t0.shape or not np.allclose(ti, t0):
                    raise ValueError(
                        f"Partition {i} ({self._paths[i].name}) time vector "
                        f"for stage {stage.name!r} differs from partition "
                        f"{ref_i} ({self._paths[ref_i].name})."
                    )

    def _pair_stages(self) -> None:
        """Pair each part's stages with the other parts' (fork WP-165).

        Every reader numbers its stages ``stage_0..`` in order of the
        integer stamp inside ``MODEL_STAGE[<n>]``, and reads go by that
        id, so pairing is by ordinal position. When every part has the
        same stage names this is pairing by name. A rank-local topology
        change can make ranks stamp different numbers for one logical
        stage: then, if the stage counts agree, the stages are matched by
        order with a :class:`StageOrderMatchWarning`. Different counts are
        refused. The name exposed to callers is the first non-empty
        part's (see :meth:`stages`).
        """
        per = [r.stages() for r in self._readers]
        names = [[s.name for s in st] for st in per]

        def listing() -> str:
            return "; ".join(
                f"{p.name}: {nm}" for p, nm in zip(self._paths, names)
            )

        if len({len(nm) for nm in names}) > 1:
            raise ValueError(
                "Partitioned .ladruno files have different stage counts "
                f"({listing()}). The part files cannot be paired stage by "
                "stage; they are probably not from one run."
            )
        if any(nm != names[0] for nm in names[1:]):
            warnings.warn(
                "Partitioned .ladruno files name their stages differently "
                f"({listing()}); stages were matched by order (numeric "
                "MODEL_STAGE stamp). A rank-local topology change stamps "
                "ranks differently; if the files are not from one run, the "
                "pairing is wrong.",
                StageOrderMatchWarning,
                stacklevel=4,
            )
        # Paired stages must agree on KIND; EMPTY_PARTITION stages hold
        # their ordinal slot but have no say.
        for k in range(len(per[0])):
            kinds = [
                (i, st[k]) for i, st in enumerate(per)
                if not self._readers[i].is_empty_partition(st[k].id)
            ]
            for i, s in kinds[1:]:
                i0, s0 = kinds[0]
                if s.kind != s0.kind:
                    raise ValueError(
                        f"Stage {k} is {s0.kind!r} in {self._paths[i0].name} "
                        f"({s0.name}) but {s.kind!r} in "
                        f"{self._paths[i].name} ({s.name})."
                    )

    def _live(self, stage_id: str) -> "list[tuple[int, LadrunoReader]]":
        """``(index, reader)`` of the parts that hold results for a stage.

        Parts whose stage is marked ``EMPTY_PARTITION`` drop out. If every
        part is empty, all of them are returned, so reads still answer
        with the empty slab a single empty file gives.
        """
        pairs = list(enumerate(self._readers))
        live = [(i, r) for i, r in pairs if not r.is_empty_partition(stage_id)]
        return live or pairs

    def _live_readers(self, stage_id: str) -> "list[LadrunoReader]":
        return [r for _, r in self._live(stage_id)]

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
        # Step counts come from a part that holds the stage's results:
        # partition 0 may be an EMPTY_PARTITION with no TIME axis.
        out: list[StageInfo] = []
        for s in self._readers[0].stages():
            ref = self._live_readers(s.id)[0]
            out.append(s if ref is self._readers[0] else next(
                rs for rs in ref.stages() if rs.id == s.id
            ))
        return out

    def time_vector(self, stage_id: str) -> ndarray:
        return self._live_readers(stage_id)[0].time_vector(stage_id)

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
        """The minimal broker is built from the first part with a MODEL.

        Partition 0 unless its MODEL is empty (``EMPTY_PARTITION``).
        Mirrors the single-file self-sufficient path; richer lineage
        still comes via ``model_h5=`` on :meth:`Results.from_ladruno`.
        """
        for r in self._readers:
            if r.fem() is not None:
                return r.opensees_model()
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
        live = self._live(stage_id)
        readers = [r for _, r in live]
        return _merge_node_slabs(
            [r.read_nodes(stage_id, component, node_ids=node_ids,
                          time_slice=time_slice) for r in readers],
            component,
            _node_reduction(
                readers, [self._paths[i] for i, _ in live],
                stage_id, component,
            ),
        )

    def read_energy(self, stage_id: str, **_kw):
        """Refused: energy balance is ``PARTITION_REDUCTION=UNSUPPORTED``.

        Each rank's ``energyBalance`` (``ON_DOMAIN`` and ``ON_REGIONS``)
        covers only its own elements, and the error terms do not add, so
        no stitched value is correct. Record energy in a serial run, or
        read one rank's balance with
        ``Results.from_ladruno(<part file>, merge_partitions=False)``.
        """
        raise ValueError(
            "energyBalance is PARTITION_REDUCTION=UNSUPPORTED in a "
            "partitioned .ladruno: each part file holds only its own "
            "rank's balance, and no sum of the parts gives the model's. "
            "Record energy in a serial run, or read one rank's balance "
            "with Results.from_ladruno(<part file>, merge_partitions=False)."
        )

    def _per_partition(self, stage_id: str, read) -> list:
        """Run ``read(reader)`` on every partition that holds results for
        the stage (``EMPTY_PARTITION`` parts drop out), tolerating ranks
        that record no element results.

        A rank owning none of the recorded elements legitimately writes a
        file with no ``ON_ELEMENTS`` group, and
        :class:`MissingElementResults` fires there. That is only a real
        loss when **every** rank is missing it — otherwise the rank is
        simply empty and drops out of the stitch.
        """
        out: list = []
        missing: "Optional[MissingElementResults]" = None
        for r in self._live_readers(stage_id):
            try:
                out.append(read(r))
            except MissingElementResults as exc:
                missing = missing or exc
        if not out and missing is not None:
            raise missing
        return out

    def read_elements(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> ElementSlab:
        return _concat_element_slabs(
            self._per_partition(
                stage_id,
                lambda r: r.read_elements(
                    stage_id, component, element_ids=element_ids,
                    time_slice=time_slice,
                ),
            ),
            component,
        )

    def read_line_stations(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        return _concat_line_station_slabs(
            self._per_partition(
                stage_id,
                lambda r: r.read_line_stations(
                    stage_id, component, element_ids=element_ids,
                    time_slice=time_slice,
                ),
            ),
            component,
        )

    def read_gauss(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> GaussSlab:
        return _concat_gauss_slabs(
            self._per_partition(
                stage_id,
                lambda r: r.read_gauss(
                    stage_id, component, element_ids=element_ids,
                    time_slice=time_slice,
                ),
            ),
            component,
        )

    def read_fibers(
        self, stage_id: str, component: str, *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None, time_slice: TimeSlice = None,
    ) -> FiberSlab:
        return _concat_fiber_slabs(
            self._per_partition(
                stage_id,
                lambda r: r.read_fibers(
                    stage_id, component, element_ids=element_ids,
                    gp_indices=gp_indices, time_slice=time_slice,
                ),
            ),
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


__all__ = [
    "LadrunoMultiPartitionReader",
    "StageOrderMatchWarning",
    "discover_partition_files",
]
