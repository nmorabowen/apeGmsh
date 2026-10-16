"""``LadrunoReader`` — reads a single ``.ladruno`` HDF5 file.

The Ladruno recorder is the fork's *canonical* recorder; this reader is
the sibling of :class:`apeGmsh.results.readers._mpco.MPCOReader`. Per the
canonicity contract it reads the file **direct** — HDF5 groups → the
``ResultsReader`` protocol slabs, lazily, with no transcode to a derived
on-disk representation.

Reads cover: file identity + ``FORMAT_VERSION`` window, stage enumeration,
time vectors, the self-describing FEM (``MODEL/`` →
:meth:`FEMData.from_ladruno_model`), nodal reads (``RESULTS/ON_NODES`` —
chunked ``DATA[T×nIds×nComp]``), element/gauss/line-station reads (L2b-2),
section-level line stations + fiber stress/strain (L2b-3), and layered-shell
layer state (written under the fiber tokens with several components per
layer, read into the MPCO reader's ``LayerSlab``, #1995). Spring reads
return empty slabs by design — a ``.ladruno`` has no distinct spring level
(zeroLength state flows through the element/gauss reads).

Key layout facts (verified against fork build ``605affeb``, FORMAT_VERSION 1):

* ``INFO`` attrs ``GENERATOR="Ladruno"``, ``FORMAT_VERSION`` (int).
* ``MODEL_STAGE[<k>]`` — integer-indexed stage groups, attr ``KIND``
  (``static``/``transient``/``eigen``).
* ``RESULTS/ON_NODES/<RESULT>`` (e.g. ``DISPLACEMENT``) — attr
  ``COMPONENTS`` (comma-joined labels ``"Ux,Uy"``); datasets ``ID[n,1]``,
  ``DATA[T,n,nComp]``, ``TIME[T]``, ``STEP[T]``.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence

import numpy as np
from numpy import ndarray

from ..._vocabulary import PER_ELEMENT_NODAL_FORCES
from .._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
    SpringSlab,
)
from .._time import resolve_time_slice
from ..schema._versions import LADRUNO_SUPPORTED_FORMAT_VERSIONS
from . import _ladruno_element_io as _eio
from . import _mpco_nodal_io as _mnodal
from ._ladruno_hyperslab import read_hyperslab
from ._mpco import (
    _PROGRAM_STAGE_KINDS,
    _empty_element_slab,
    _empty_fiber_slab,
    _empty_gauss_slab,
    _empty_layer_slab,
    _empty_line_station_slab,
    _empty_node_slab,
    _empty_spring_slab,
)
from ._mpco_translation import (
    PARTITION_REDUCTION_NONE,
    PARTITION_REDUCTIONS,
    canonical_node_component,
    partition_reduction_from_name,
)
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData
    from ...opensees.opensees_model import OpenSeesModel
    from .._slabs import LocalAxes
    from ._tag_translation import ElementTagTranslator


_STAGE_PREFIX = "MODEL_STAGE["

_DIM_UNKNOWN = (
    "no INFO/SPATIAL_DIM, and no single MODEL/NODES/COORDINATES width "
    "of 1, 2 or 3"
)


class SpatialDimUnknown(ValueError):
    """A read that needs the spatial dimension, on a file that gives none.

    Raised when the file has no ``INFO/SPATIAL_DIM`` and its
    ``MODEL/NODES/COORDINATES`` widths name no one dimension, by
    ``opensees_model`` and by a ``nodal_resisting_*`` read over indexed
    ``P<a>_<b>`` ``globalForce`` columns (#2081). It is a ``ValueError``
    and no ``MissingElementResults``, so a partition set reports it from
    any rank instead of dropping that rank's elements.
    """


def _stage_sort_key(name: str) -> "tuple[int, str]":
    """Numeric-stamp sort key for ``MODEL_STAGE[<k>]`` group names."""
    try:
        return (int(name[len(_STAGE_PREFIX):].rstrip("]")), name)
    except ValueError:
        return (2**31, name)


# Ladruno stage KIND → protocol StageInfo.kind ("transient"|"static"|"mode").
# ``eigen`` is not a mode: the recorder promotes EVERY stage of a recorder
# that asks for ``modesOfVibration`` to ``KIND=eigen`` (it overwrites the
# ``-kind`` option, default ``static``), so a gravity stage recorded beside
# modes says ``eigen`` too. The stage keeps its steps as a ``static`` stage
# and each recorded mode becomes its own ``kind="mode"`` stage.
_KIND_MAP = {"static": "static", "transient": "transient", "eigen": "static"}


class StageKindMismatchWarning(UserWarning):
    """The program's stage kinds disagree with the ``.ladruno``'s ``KIND``s.

    The fork stamps each ``MODEL_STAGE`` with the kind it ran, so a
    disagreement means the program stages were paired onto the wrong
    file stages (order or count).  Per #2079 Q7 the file keeps its own
    ``MODEL_STAGE[<k>]`` names and kinds; select stages by those names
    or by id.
    """


def warn_stage_kind_conflicts(conflicts: "Sequence[str]") -> None:
    """Warn :class:`StageKindMismatchWarning` over every conflict."""
    warnings.warn(
        StageKindMismatchWarning(
            "attach_stage_names: the program's stage kinds disagree with "
            f"the kinds the file records ({'; '.join(conflicts)}). The "
            "pairing is misaligned (stage order or count), so no stage is "
            "renamed: the file keeps its MODEL_STAGE names and kinds "
            "(#2079 Q7)."
        ),
        stacklevel=3,
    )


class NodeEnvelope(NamedTuple):
    """Per-node time-reduced extremes for one component (recorder
    ``-envelope``). All arrays are aligned by ``node_ids``; ``arg_step`` is
    the 0-based step index at which ``absmax`` occurred (no TIME axis is
    stored in an envelope file, so the step index — the recorder's
    ``commitTag`` — is reported instead of a time)."""
    node_ids: ndarray
    min: ndarray
    max: ndarray
    absmax: ndarray
    arg_step: ndarray


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


def _decode(value) -> str:
    """Coerce an h5py string attr (bytes / 1-elem array / str) to str."""
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _attr_int(attrs, name: str, default: int) -> int:
    if name not in attrs:
        return default
    v = attrs[name]
    if isinstance(v, np.ndarray):
        return int(v.flat[0]) if v.size else default
    return int(v)


def _child(group: "h5py.Group", path: str):
    """Safe multi-segment traversal; ``None`` if any segment is missing."""
    cur = group
    for seg in path.split("/"):
        if seg == "" or seg not in cur:
            return None
        cur = cur[seg]
    return cur


class LadrunoReader:
    """Reads one ``.ladruno`` file into the ``ResultsReader`` protocol."""

    def __init__(self, path: "str | Path") -> None:
        import h5py

        self._path = Path(path)
        self._h5: "h5py.File" = h5py.File(self._path, "r")
        self._validate_identity()
        self._stage_cache: Optional[list[StageInfo]] = None
        self._stage_to_grp: dict[str, str] = {}
        # Mode stage id -> (host MODEL_STAGE group, its ModeEntry).
        # Mode stages are not MODEL_STAGE groups, so they stay out of
        # ``_stage_to_grp`` (which lists the host stages, ``stage_<i>``).
        self._mode_refs: dict[str, tuple[str, _mnodal.ModeEntry]] = {}
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL
        self._tag_map: "Optional[ElementTagTranslator]" = None
        self._rotation_mask = _mnodal.RotationMask(self._path.name)

    # -- identity / lifecycle ------------------------------------------

    def _validate_identity(self) -> None:
        info = (self._h5["INFO"] if "INFO" in self._h5 else None)
        if info is None or "GENERATOR" not in info.attrs:
            raise ValueError(
                f"{self._path} is not a Ladruno file: missing INFO/GENERATOR. "
                "from_ladruno expects a '.ladruno' written by the Ladruno "
                "recorder."
            )
        gen = _decode(info.attrs["GENERATOR"])
        if gen != "Ladruno":
            raise ValueError(
                f"{self._path}: INFO/GENERATOR is {gen!r}, expected 'Ladruno' "
                "(a '.mpco' or foreign HDF5 was passed to from_ladruno)."
            )
        ver = _attr_int(info.attrs, "FORMAT_VERSION", default=-1)
        if ver not in LADRUNO_SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"{self._path}: INFO/FORMAT_VERSION={ver} is not supported by "
                f"this reader (supported: {LADRUNO_SUPPORTED_FORMAT_VERSIONS}). "
                "Update apeGmsh, or the file was written by a newer/older "
                "Ladruno format."
            )

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def __enter__(self) -> "LadrunoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def attach_tag_map(self, tag_map: "ElementTagTranslator") -> None:
        """Store the fem_eid↔ops-tag translator (ADR 0043 slice 1.3).

        Element/gauss/line reads relabel ops↔fem_eid through it whenever
        the bound model records an element_meta pairing (composed models
        AND sparsely-renumbered meshes), exactly like :class:`MPCOReader`.
        The translator is told the ops tags this file records (#2066).
        """
        tag_map.declare_recorded(self._recorded_element_tags())
        self._tag_map = tag_map

    def _recorded_element_tags(self) -> "ndarray":
        """Ops tags of every element the file records, over all stages:
        the ``MODEL/ELEMENTS`` listing (column 0 of each class group's
        ``CONNECTIVITY``; a recorder ``-R`` region writes only its
        elements there) and each ``RESULTS/ON_ELEMENTS`` bucket ``ID``."""
        import h5py

        if self._stage_cache is None:
            self.stages()
        parts = [np.empty(0, dtype=np.int64)]
        for name in self._stage_to_grp.values():
            stage = self._h5[name]
            elements = _child(stage, "MODEL/ELEMENTS")
            for grp in elements.values() if elements is not None else ():
                if isinstance(grp, h5py.Group) and "CONNECTIVITY" in grp:
                    conn = grp["CONNECTIVITY"]
                    if conn.ndim == 2 and conn.shape[0]:
                        parts.append(np.asarray(conn[:, 0], dtype=np.int64))
            on_elements = _child(stage, "RESULTS/ON_ELEMENTS")
            for token in on_elements.values() if on_elements is not None else ():
                for bucket in token.values() if isinstance(token, h5py.Group) else ():
                    if isinstance(bucket, h5py.Group) and "ID" in bucket:
                        parts.append(np.asarray(
                            bucket["ID"][...], dtype=np.int64).ravel())
        return np.unique(np.concatenate(parts))

    def attach_nodes_ndf(
        self, nodes_ndf: "dict[int, int] | None", ndm: int,
    ) -> None:
        """Store the model's per-node ndf: rotation reads (time series and
        envelopes) then give NaN at a node whose DOF layout has no such
        rotation (#1971). ``None`` is an archive older than nodes_ndf."""
        if self._stage_cache is None:
            self.stages()
        self._rotation_mask.attach(nodes_ndf, ndm, _mnodal.model_node_ids(
            [self._h5[name] for name in self._stage_to_grp.values()],
        ))

    def _element_read_translation(
        self, element_ids: "Optional[ndarray]",
    ) -> "tuple[Optional[ndarray], Any]":
        """Per-read ``(bucket_filter, index_to_fem_fn)`` — consistent pair.

        Both directions of one read are decided together (see
        :meth:`ElementTagTranslator.read_translation`): each filter id
        translates on its own, and an id the files cannot return warns.
        """
        if self._tag_map is None:
            return element_ids, (lambda index: index)
        return self._tag_map.read_translation(element_ids)

    # -- stages / time -------------------------------------------------

    def stage_kind_conflicts(
        self, names: "Sequence[str]", kinds: "Sequence[str]",
    ) -> list[str]:
        """The host stages whose recorded ``KIND`` disagrees with ``kinds``.

        It refuses what :meth:`MPCOReader.attach_stage_names` refuses: a
        count mismatch over the non-mode stages, or a kind outside
        ``static``/``transient``.  Then it compares each host stage's raw
        ``KIND`` attribute with the program kind paired onto it.  Only a
        raw ``static`` or ``transient`` is evidence.  ``eigen`` is not: the
        recorder writes it on every stage of a recorder that asks for
        modes, transient ones included (see ``_KIND_MAP``).  A stage with
        no ``KIND`` (older builds), with ``eigen``, or marked
        ``EMPTY_PARTITION`` (it has no say, as in the part pairing) is not
        compared.  Any other ``KIND`` counts as a conflict.
        """
        hosts = [s for s in self.stages() if s.kind != "mode"]
        if not len(names) == len(kinds) == len(hosts):
            raise ValueError(
                f"attach_stage_names: {len(names)} names and "
                f"{len(kinds)} kinds for {len(hosts)} MODEL_STAGE groups "
                f"in {self._path.name}."
            )
        bad = sorted({str(k) for k in kinds} - _PROGRAM_STAGE_KINDS)
        if bad:
            raise ValueError(
                f"attach_stage_names: stage kinds {bad} are not program "
                f"stage kinds {sorted(_PROGRAM_STAGE_KINDS)}."
            )
        out = []
        for s, n, k in zip(hosts, names, kinds):
            attrs = self._h5[self._stage_to_grp[s.id]].attrs
            if "KIND" not in attrs or self.is_empty_partition(s.id):
                continue
            raw = _decode(attrs["KIND"])
            if raw != "eigen" and raw != str(k):
                out.append(
                    f"{self._path.name} {s.name} records KIND {raw!r} but "
                    f"program stage {str(n)!r} is {str(k)!r}"
                )
        return out

    def attach_stage_names(
        self, names: "Sequence[str]", kinds: "Sequence[str]",
    ) -> None:
        """Name and kind the capture stages after the program's stages.

        The sibling of :meth:`MPCOReader.attach_stage_names` (#2100): the
        ``ops.stage(name=...)`` names and each stage's ``analysis`` kind
        live in the sibling ``model.h5``, and the caller pairs them
        positionally over the non-mode stages once it has checked the
        counts.  A count mismatch or a kind outside ``static``/
        ``transient`` raises.  Stage ids are untouched; the
        ``MODEL_STAGE[<k>]`` name moves to :attr:`StageInfo.aliases`.
        Mode stages keep their own names and kind.

        Unlike an ``.mpco``, a ``.ladruno`` stage records its ``KIND``.
        Where it disagrees with the program kind the pairing is
        misaligned (wrong stage order or count), so per #2079 Q7 nothing
        is renamed: the file keeps its names and kinds and
        :class:`StageKindMismatchWarning` lists the conflicts.  The
        program kind fills in only where a stage records no ``KIND``, or
        ``eigen`` (no evidence of static or transient).
        """
        conflicts = self.stage_kind_conflicts(names, kinds)
        if conflicts:
            warn_stage_kind_conflicts(conflicts)
            return
        current = self.stages()
        hosts = [s for s in current if s.kind != "mode"]
        new = {s.id: (str(n), str(k)) for s, n, k in zip(hosts, names, kinds)}
        self._stage_cache = [
            replace(s, name=new[s.id][0], kind=new[s.id][1],
                    aliases=(s.name, *s.aliases))
            if s.id in new else s
            for s in current
        ]

    def stages(self) -> list[StageInfo]:
        if self._stage_cache is not None:
            return list(self._stage_cache)
        hosts: list[StageInfo] = []
        # Sort by the numeric stamp inside ``MODEL_STAGE[<k>]`` —
        # lexicographic sorting would put MODEL_STAGE[10] before
        # MODEL_STAGE[2] once a run reaches ten stages.
        names = sorted(
            (k for k in self._h5.keys() if k.startswith(_STAGE_PREFIX)),
            key=_stage_sort_key,
        )
        host_sets = []
        for i, name in enumerate(names):
            stage_id = f"stage_{i}"
            self._stage_to_grp[stage_id] = name
            grp = self._h5[name]
            kind_raw = _decode(grp.attrs["KIND"]) if "KIND" in grp.attrs else "static"
            kind = _KIND_MAP.get(kind_raw, "static")
            time = self._time_for_group(grp)
            hosts.append(
                StageInfo(
                    id=stage_id, name=name, kind=kind, n_steps=int(time.size),
                )
            )
            on_nodes = _child(grp, "RESULTS/ON_NODES")
            host_sets.append(
                [] if on_nodes is None else _mnodal.discover_modes(
                    on_nodes, source=f"{self._path.name} {name}",
                )
            )
        # Each recorded mode is its own kind="mode" stage, listed
        # right after the MODEL_STAGE that holds it (the layout of a
        # native results file).
        modes_after: dict[int, list[StageInfo]] = {}
        for h, sid, mname, entry in _mnodal.mode_stage_layout(
            [s.id for s in hosts], names, host_sets,
        ):
            self._mode_refs[sid] = (names[h], entry)
            modes_after.setdefault(h, []).append(StageInfo(
                id=sid, name=mname, kind="mode", n_steps=1,
                eigenvalue=entry.eigenvalue,
                frequency_hz=entry.frequency_hz,
                period_s=entry.period_s,
                mode_index=entry.mode_index,
            ))
        out: list[StageInfo] = []
        for h, host in enumerate(hosts):
            out.append(host)
            out.extend(modes_after.get(h, []))
        self._stage_cache = out
        return list(out)

    def _refuse_on_mode(self, stage_id: str, what: str) -> None:
        """Fail loud on a non-nodal read of a mode stage."""
        if self._stage_cache is None:
            self.stages()
        if stage_id in self._mode_refs:
            raise ValueError(
                f"{self._path}: stage {stage_id!r} is a mode stage; it holds "
                f"the mode shape at nodes only, so {what} cannot be read "
                "from it. Read the stage that recorded the analysis steps."
            )

    def _resolve_stage_group(self, stage_id: str) -> "h5py.Group":
        if self._stage_cache is None:
            self.stages()
        if stage_id in self._mode_refs:
            # A mode stage reads its host's group (model, partition flag,
            # modal groups); its own reads branch on ``_mode_refs``.
            return self._h5[self._mode_refs[stage_id][0]]
        if stage_id not in self._stage_to_grp:
            raise KeyError(
                f"Unknown stage_id {stage_id!r}; available: "
                f"{sorted([*self._stage_to_grp, *self._mode_refs])}."
            )
        return self._h5[self._stage_to_grp[stage_id]]

    def _time_for_group(self, grp: "h5py.Group") -> ndarray:
        """Read a stage's TIME axis from the first result that has one."""
        results = _child(grp, "RESULTS")
        if results is None:
            return np.array([], dtype=np.float64)
        # ON_NODES/<res>/TIME is the canonical source; fall back to any
        # ON_* result group carrying a TIME dataset.
        for bucket in ("ON_NODES", "ON_ELEMENTS", "ON_DOMAIN", "ON_REGIONS"):
            bgrp = results.get(bucket)
            if bgrp is None:
                continue
            t = self._find_time_dataset(bgrp)
            if t is not None:
                return np.asarray(t[...], dtype=np.float64).flatten()
        return np.array([], dtype=np.float64)

    @staticmethod
    def _find_time_dataset(group: "h5py.Group"):
        """Depth-first search for a ``TIME`` dataset under ``group``."""
        import h5py

        for key in group:
            item = group[key]
            if key == "TIME" and isinstance(item, h5py.Dataset):
                return item
            if isinstance(item, h5py.Group):
                found = LadrunoReader._find_time_dataset(item)
                if found is not None:
                    return found
        return None

    def time_vector(self, stage_id: str) -> ndarray:
        grp = self._resolve_stage_group(stage_id)
        if stage_id in self._mode_refs:
            return np.array([0.0], dtype=np.float64)
        return self._time_for_group(grp)

    def partitions(self, stage_id: str) -> list[str]:
        return ["partition_0"]

    def is_empty_partition(self, stage_id: str) -> bool:
        """``True`` when the stage is marked ``EMPTY_PARTITION = 1``.

        Fork WP-165: a part file whose process held no node of the
        recorded set writes zero-length ``MODEL/NODES`` and no node or
        element results for that stage. It is a valid empty contribution
        to a partitioned set, not a broken file.
        """
        grp = self._resolve_stage_group(stage_id)
        return _attr_int(grp.attrs, "EMPTY_PARTITION", default=0) == 1

    def partition_manifest(self) -> "dict[str, Any]":
        """The ``INFO`` partition and run-identity attributes.

        Keys ``PARTITIONED``, ``PARTITION_ID``, ``NUM_PARTITIONS`` (int or
        ``None``) and ``RUN_ID``, ``RUN_ID_SCOPE`` (str or ``None``). A
        missing attribute is ``None``: files written before the manifest
        (or before fork WP-165, for the run identity) do not carry it.
        """
        info = self._h5["INFO"]
        out: "dict[str, Any]" = {}
        for name in ("PARTITIONED", "PARTITION_ID", "NUM_PARTITIONS"):
            out[name] = (
                _attr_int(info.attrs, name, default=-1)
                if name in info.attrs else None
            )
        for name in ("RUN_ID", "RUN_ID_SCOPE"):
            out[name] = (
                _decode(info.attrs[name]).strip("\x00 ")
                if name in info.attrs else None
            )
        return out

    # -- model / fem ---------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        if not isinstance(self._fem_cache, _Sentinel):
            return self._fem_cache  # type: ignore[return-value]
        if self._stage_cache is None:
            self.stages()
        if not self._stage_to_grp:
            self._fem_cache = None
            return None
        # Use the last stage's MODEL (most up-to-date geometry), skipping
        # stages marked EMPTY_PARTITION (zero-length NODES, fork WP-165):
        # a part file that never held a node contributes no FEM.
        grp = None
        for i in reversed(range(len(self._stage_to_grp))):
            cand = self._h5[self._stage_to_grp[f"stage_{i}"]]
            if _attr_int(cand.attrs, "EMPTY_PARTITION", default=0) != 1:
                grp = cand
                break
        if grp is None:
            self._fem_cache = None
            return None
        model_grp = (grp["MODEL"] if "MODEL" in grp else None)
        if model_grp is None:
            self._fem_cache = None
            return None
        from ...mesh.FEMData import FEMData
        self._fem_cache = FEMData.from_ladruno_model(model_grp)
        return self._fem_cache  # type: ignore[return-value]

    def opensees_model(self) -> "Optional[OpenSeesModel]":
        """Build a **minimal** in-memory ``OpenSeesModel`` from the
        self-describing ``MODEL`` group — the canonical/self-sufficient
        path (schema Principle 0: a ``.ladruno`` *is* the native path,
        no sibling ``model.h5`` required).

        A ``.ladruno`` carries no ``/opensees`` bridge zone, so the bridge
        collections (materials/sections/transforms/elements/…) are empty;
        the broker wraps the file's geometry + inferred ``ndm``/``ndf``.
        ``from_ladruno`` prefers an explicit ``model_h5=`` when richer
        lineage / bridge records are wanted. This is **read-time
        interpretation**, not a transcode — nothing is re-encoded to disk.
        """
        fem = self.fem()
        if fem is None:
            return None
        from types import MappingProxyType

        from ...opensees.opensees_model import OpenSeesModel
        ndm = self._spatial_dim()
        if ndm is None:
            raise SpatialDimUnknown(
                f"{self._path.name}: the spatial dimension is unknown "
                f"({_DIM_UNKNOWN}), so the model's ndm cannot be built. "
                f"Pass model_h5= to supply the model."
            )
        # A ``.ladruno`` does NOT record ndf — the DISPLACEMENT field is
        # always 3 translations regardless of rotational DOFs, so it can't
        # be inferred from result widths. The minimal broker uses ndm as a
        # safe default; pass ``model_h5=`` when the exact ndf matters
        # (e.g. reaction/rotation shorthand on a 6-DOF model).
        ndf = ndm
        return OpenSeesModel(
            _fem=fem,
            _model_name="ladruno",
            _ndm=ndm,
            _ndf=ndf,
            _snapshot_id="",
            _materials_by_family=MappingProxyType({}),
            _sections=(),
            _transforms=(),
            _beam_integration=(),
            _time_series=(),
            _elements=(),
            _fixes=(),
            _masses=(),
            _patterns=(),
            _recorders=(),
            _analysis_attrs=MappingProxyType({}),
            _analyze_call=None,
            _cuts=(),
            _sweeps=(),
        )

    def _spatial_dim(self) -> Optional[int]:
        """The file's spatial dimension, or ``None`` when it is unknown.

        ``INFO/SPATIAL_DIM`` when the file carries it. Otherwise the width
        of ``MODEL/NODES/COORDINATES``, which the writer sizes by the
        spatial dimension, when every stage that has one agrees on a
        width of 1, 2 or 3. Otherwise ``None``: a read that depends on
        the dimension refuses or warns, naming the file (#2081).
        """
        info = (self._h5["INFO"] if "INFO" in self._h5 else None)
        if info is not None and "SPATIAL_DIM" in info.attrs:
            raw = np.asarray(info.attrs["SPATIAL_DIM"])
            if raw.size:
                return int(raw.flat[0])
        widths: set[int] = set()
        for name in self._h5:
            if not name.startswith(_STAGE_PREFIX):
                continue
            coords = _child(self._h5[name], "MODEL/NODES/COORDINATES")
            if coords is not None and len(coords.shape) == 2:
                widths.add(int(coords.shape[1]))
        if len(widths) == 1 and widths <= {1, 2, 3}:
            return widths.pop()
        return None

    # -- components / reads --------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        grp = self._resolve_stage_group(stage_id)
        if stage_id in self._mode_refs:
            on_nodes = _child(grp, "RESULTS/ON_NODES")
            if level is not ResultLevel.NODES or on_nodes is None:
                return []
            return _mnodal.modal_components(on_nodes)
        if level is ResultLevel.NODES:
            on_nodes = _child(grp, "RESULTS/ON_NODES")
            if on_nodes is None:
                return []
            return self._node_components_of(on_nodes)

        on_elements = _child(grp, "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        if level is ResultLevel.GAUSS:
            return sorted(_eio.gauss_available(on_elements))
        if level is ResultLevel.LINE_STATIONS:
            return sorted(_eio.line_station_available(on_elements))
        if level is ResultLevel.ELEMENTS:
            ndm = self._spatial_dim()
            if ndm is not None:
                return sorted(_eio.element_available(on_elements, ndm=ndm))
            indexed = _eio.indexed_nodal_force_buckets(on_elements)
            # Without indexed P<a>_<b> buckets no decode consults ndm
            # (axis-named columns name their quantity), so 3 never
            # reaches an answer here.
            listed = _eio.element_available(on_elements, ndm=3)
            if not indexed:
                return sorted(listed)
            # An indexed column is a translation only up to ndm. The read
            # refuses these names, so the listing leaves them out.
            nodal = listed.intersection(PER_ELEMENT_NODAL_FORCES)
            warnings.warn(
                f"{self._path.name}: the spatial dimension is unknown "
                f"({_DIM_UNKNOWN}), so the element components leave out "
                f"{sorted(nodal)}: the globalForce bucket(s) {indexed} "
                f"name their columns by DOF index, which reads as a "
                f"translation only up to that dimension. The raw "
                f"globalForce token stays readable.",
                UserWarning,
                stacklevel=2,
            )
            return sorted(listed.difference(PER_ELEMENT_NODAL_FORCES))
        if level is ResultLevel.FIBERS:
            return sorted(_eio.fiber_available(on_elements))
        if level is ResultLevel.LAYERS:
            return sorted(_eio.layer_available(on_elements))
        # SPRINGS: a .ladruno has no distinct spring level — zeroLength
        # material/force state is reachable through read_gauss /
        # read_elements. See _ladruno_element_io.
        return []

    def _locate_node_component(
        self, on_nodes: "h5py.Group", component: str,
    ) -> "Optional[tuple[str, int]]":
        for res_name in on_nodes:
            res = on_nodes[res_name]
            if "COMPONENTS" not in res.attrs:
                continue
            labels = _decode(res.attrs["COMPONENTS"]).split(",")
            for col, label in enumerate(labels):
                if canonical_node_component(res_name, label.strip()) == component:
                    return res_name, col
        return None

    def node_partition_reduction(
        self, stage_id: str, component: str,
    ) -> "Optional[str]":
        """How the partition copies of ``component`` combine.

        Returns ``"NONE"``, ``"SUM"`` or ``"UNSUPPORTED"``, read from the
        ``PARTITION_REDUCTION`` attribute of the result group that holds
        the component, or from the group name when the file predates the
        attribute. ``None`` when the file does not record the component.
        """
        on_nodes = _child(self._resolve_stage_group(stage_id), "RESULTS/ON_NODES")
        if on_nodes is None:
            return None
        if stage_id in self._mode_refs:
            # A mode shape is the same at every copy of a node.
            if component in _mnodal.modal_components(on_nodes):
                return PARTITION_REDUCTION_NONE
            return None
        loc = self._locate_node_component(on_nodes, component)
        if loc is None:
            return None
        res_name, _ = loc
        res = on_nodes[res_name]
        if "PARTITION_REDUCTION" not in res.attrs:
            return partition_reduction_from_name(res_name)
        kind = _decode(res.attrs["PARTITION_REDUCTION"]).strip("\x00 ").upper()
        if kind not in PARTITION_REDUCTIONS:
            raise ValueError(
                f"{self._path}: {res_name}/PARTITION_REDUCTION is {kind!r}; "
                f"expected one of {PARTITION_REDUCTIONS}."
            )
        return kind

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: "Optional[ndarray]" = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        slab = self._read_nodes_as_recorded(
            stage_id, component, node_ids=node_ids, time_slice=time_slice,
        )
        return self._rotation_mask.apply(slab, _child(
            self._resolve_stage_group(stage_id), "MODEL/ELEMENTS",
        ))

    def _read_nodes_as_recorded(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: "Optional[ndarray]" = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        grp = self._resolve_stage_group(stage_id)
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        on_nodes = _child(grp, "RESULTS/ON_NODES")
        if on_nodes is None:
            return _empty_node_slab(component, time, t_idx)
        if stage_id in self._mode_refs:
            got = _mnodal.read_mode_shape(
                on_nodes, self._mode_refs[stage_id][1], component,
                node_ids=node_ids,
            )
            if got is None:
                return _empty_node_slab(component, time, t_idx)
            shape, shape_ids = got
            return NodeSlab(
                component=component,
                values=np.repeat(shape[None, :], t_idx.size, axis=0),
                node_ids=shape_ids, time=time[t_idx],
            )
        loc = self._locate_node_component(on_nodes, component)
        if loc is None:
            return _empty_node_slab(component, time, t_idx)
        res_name, col = loc
        res = on_nodes[res_name]
        ids = np.asarray(res["ID"][...], dtype=np.int64).flatten()
        # DATA is chunked [T × nIds × nComp]: read only the requested
        # steps, nodes and column (a hyperslab), never the whole array.
        rows: "Optional[ndarray]" = None
        sel_ids = ids
        if node_ids is not None:
            want = np.asarray(node_ids, dtype=np.int64)
            rows = np.flatnonzero(np.isin(ids, want))
            sel_ids = ids[rows]
        vals = read_hyperslab(
            res["DATA"], t_idx, rows, np.array([col]),
        )[:, :, 0]  # (T, N)
        return NodeSlab(
            component=component, values=vals, node_ids=sel_ids, time=time[t_idx],
        )

    # -- local axes (beam/shell orientation — Ladruno-only extension) --

    def read_local_axes(
        self,
        stage_id: str,
        *,
        element_ids: "Optional[ndarray]" = None,
    ) -> "LocalAxes":
        """Per-element local frames from ``MODEL/LOCAL_AXES``.

        Ladruno groups frames by class (``<classTag>-<ClassName>/{ID,
        FRAME}``); this flattens them into one ``{element_id: quaternion}``
        lookup. ``element_ids=None`` returns every recorded frame (sorted
        by id); an explicit list returns those ids with an identity-frame
        fallback for any element without a recorded frame.
        """
        from .._slabs import LocalAxes

        grp = self._resolve_stage_group(stage_id)
        la = _child(grp, "MODEL/LOCAL_AXES")
        id_to_quat: dict[int, ndarray] = {}
        if la is not None:
            for cls_name in la:
                sub = la[cls_name]
                if "ID" not in sub or "FRAME" not in sub:
                    continue
                ids = np.asarray(sub["ID"][...]).flatten().astype(np.int64)
                frames = np.asarray(
                    sub["FRAME"][...], dtype=np.float64,
                ).reshape(-1, 4)
                for i, eid in enumerate(ids):
                    id_to_quat[int(eid)] = frames[i]
        if element_ids is None:
            out_ids = np.array(sorted(id_to_quat), dtype=np.int64)
        else:
            out_ids = np.asarray(element_ids, dtype=np.int64).flatten()
        quats = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (out_ids.size, 1),
        )
        for k, eid in enumerate(out_ids):
            q = id_to_quat.get(int(eid))
            if q is not None:
                quats[k, :] = q
        return LocalAxes(element_ids=out_ids, quaternions=quats)

    # -- energy balance (Ladruno-only extension, not in the protocol) --

    def read_energy(
        self,
        stage_id: str,
        *,
        region: "Optional[int]" = None,
        time_slice: TimeSlice = None,
    ) -> "tuple[list[str], ndarray, ndarray]":
        """Read the energy-balance time history (recorder ``-G energy``).

        Returns ``(component_names, values[T, nComp], time[T])``.
        ``region=None`` → whole-domain ``RESULTS/ON_DOMAIN/energyBalance``;
        ``region=<tag>`` → the matching row of
        ``RESULTS/ON_REGIONS/energyBalance``. Raises if the channel is
        absent (energy was not requested at record time).
        """
        self._refuse_on_mode(stage_id, "the energy balance")
        grp = self._resolve_stage_group(stage_id)
        bucket = "ON_REGIONS" if region is not None else "ON_DOMAIN"
        eg = _child(grp, f"RESULTS/{bucket}/energyBalance")
        if eg is None:
            raise ValueError(
                f"This .ladruno has no {bucket}/energyBalance channel — "
                "energy was not recorded. Add the recorder's '-G energy' "
                "verb at record time."
            )
        cols = [c.strip() for c in _decode(eg.attrs["COMPONENTS"]).split(",")]
        ids = np.asarray(eg["ID"][...], dtype=np.int64).flatten()
        time = np.asarray(eg["TIME"][...], dtype=np.float64).flatten()
        if region is None:
            row = 0
        else:
            matches = np.where(ids == int(region))[0]
            if matches.size == 0:
                raise ValueError(
                    f"region {region} is not in this .ladruno's per-region "
                    f"energy (recorded region tags: {ids.tolist()})."
                )
            row = int(matches[0])
        t_idx = resolve_time_slice(time_slice, time)
        # (T, nComp): one region row of DATA[T × nIds × nComp].
        values = read_hyperslab(eg["DATA"], t_idx, np.array([row]))[:, 0, :]
        return cols, values, time[t_idx]

    def available_energy_regions(self, stage_id: str) -> list[int]:
        """Region tags for which per-region energy was recorded.

        Returns the ``RESULTS/ON_REGIONS/energyBalance/ID`` tags in
        recording order (the recorder's ``-G energy <tag...>`` regions),
        so a caller can discover which tag to pass to :meth:`read_energy`.
        Empty when only the whole-model balance (``ON_DOMAIN``) was
        recorded, or when energy was not recorded at all.
        """
        grp = self._resolve_stage_group(stage_id)
        eg = _child(grp, "RESULTS/ON_REGIONS/energyBalance")
        if eg is None or stage_id in self._mode_refs:
            return []
        ids = np.asarray(eg["ID"][...], dtype=np.int64).flatten()
        return [int(t) for t in ids]

    # -- node envelopes (recorder -envelope, Ladruno-only extension) ---

    def available_node_envelope_components(self, stage_id: str) -> list[str]:
        """Canonical node components present under ``RESULTS/ENVELOPES/
        ON_NODES`` (the recorder's ``-envelope`` mode). Empty when the file
        was not recorded with ``-envelope`` (the time-series ``ON_NODES``
        path then carries the data instead)."""
        grp = self._resolve_stage_group(stage_id)
        env = _child(grp, "RESULTS/ENVELOPES/ON_NODES")
        if env is None or stage_id in self._mode_refs:
            return []
        return self._node_components_of(env)

    def read_node_envelope(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: "Optional[ndarray]" = None,
    ) -> NodeEnvelope:
        """Per-node time-reduced extremes for ``component`` (recorder
        ``-envelope``).

        Reads ``RESULTS/ENVELOPES/ON_NODES/<RESULT>/{MIN,MAX,ABSMAX,
        ARG_STEP}`` — componentwise running extremes the recorder
        accumulates *instead of* a time series when ``-envelope`` is set.
        Raises :class:`ValueError` if the file has no node envelopes (it
        was recorded as a plain time series) or the component is absent.
        A rotation a node does not have reads NaN, with ``arg_step``
        ``-1``, as :meth:`read_nodes` masks it.
        """
        self._refuse_on_mode(stage_id, "node envelopes")
        grp = self._resolve_stage_group(stage_id)
        env = _child(grp, "RESULTS/ENVELOPES/ON_NODES")
        if env is None:
            raise ValueError(
                f"This .ladruno has no RESULTS/ENVELOPES/ON_NODES — it was "
                "not recorded with the '-envelope' flag (the time-series "
                "ON_NODES path carries the data; use read_nodes instead)."
            )
        loc = self._locate_node_component(env, component)
        if loc is None:
            avail = self._node_components_of(env)
            raise ValueError(
                f"component {component!r} is not in this .ladruno's node "
                f"envelopes (available: {avail})."
            )
        res_name, col = loc
        res = env[res_name]
        ids = np.asarray(res["ID"][...], dtype=np.int64).flatten()
        if node_ids is not None:
            mask = np.isin(ids, np.asarray(node_ids, dtype=np.int64))
        else:
            mask = np.ones(ids.size, dtype=bool)

        absent = self._rotation_mask.absent(
            component, ids[mask], _child(grp, "MODEL/ELEMENTS"),
        )

        def _col(name: str, dtype) -> ndarray:
            out = np.asarray(res[name][...], dtype=dtype)[mask, col]
            if absent is not None:
                out[absent] = np.nan if dtype is np.float64 else -1
            return out

        return NodeEnvelope(
            node_ids=ids[mask],
            min=_col("MIN", np.float64),
            max=_col("MAX", np.float64),
            absmax=_col("ABSMAX", np.float64),
            arg_step=_col("ARG_STEP", np.int64),
        )

    @staticmethod
    def _node_components_of(on_nodes: "h5py.Group") -> list[str]:
        """Canonical node components carried by an ``ON_NODES``-shaped group
        (shared by the time-series and envelope paths — both key result
        groups by ``COMPONENTS``)."""
        out: list[str] = []
        for res_name in on_nodes:
            res = on_nodes[res_name]
            if "COMPONENTS" not in res.attrs:
                continue
            for label in _decode(res.attrs["COMPONENTS"]).split(","):
                canon = canonical_node_component(res_name, label.strip())
                if canon is not None and canon not in out:
                    out.append(canon)
        return out

    # -- element-level reads (L2b-2) -----------------------------------

    def _on_elements(self, stage_id: str) -> "Optional[h5py.Group]":
        grp = self._resolve_stage_group(stage_id)
        return _child(grp, "RESULTS/ON_ELEMENTS")

    def _require_element_results(
        self, on_elements: "Optional[h5py.Group]", *,
        family: str, component: str, stage_id: str,
    ) -> "h5py.Group":
        """Fail loud when the stage records no element results at all.

        The empty-container answer this replaces once cost a full field
        capture — see :func:`_eio.require_element_results` for the scope
        rule (missing *group*, not missing component).
        """
        return _eio.require_element_results(
            on_elements, family=family, component=component,
            source=self._path, stage_id=stage_id,
        )

    def read_elements(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> ElementSlab:
        self._refuse_on_mode(stage_id, f"element component {component!r}")
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        on_e = self._require_element_results(
            self._on_elements(stage_id),
            family="element", component=component, stage_id=stage_id,
        )
        bucket_ids, index_to_fem = self._element_read_translation(element_ids)
        # A ``nodal_resisting_*`` name decodes ``globalForce`` per node by
        # the spatial dimension, and refuses by name when the file is
        # unable to answer. A token read does not use the dimension.
        ndm: Optional[int] = None
        if component in PER_ELEMENT_NODAL_FORCES:
            ndm = self._spatial_dim()
            if ndm is None:
                indexed = _eio.indexed_nodal_force_buckets(on_e)
                if indexed:
                    raise SpatialDimUnknown(
                        f"{self._path.name}: element component "
                        f"{component!r} decodes the globalForce bucket(s) "
                        f"{indexed}, whose columns are named by DOF index "
                        f"and read as a translation only up to the spatial "
                        f"dimension. The file gives none ({_DIM_UNKNOWN}). "
                        f"Read the globalForce token for the raw columns."
                    )
                # No bucket is indexed, so the decode answers the same at
                # any ndm: axis-named columns name their quantity, and a
                # mixed or unparseable bucket is refused either way.
                ndm = 3
        result = _eio.read_element_slab(
            on_e, component,
            t_idx=t_idx, element_ids=bucket_ids, ndm=ndm,
        )
        if result is None:
            return _empty_element_slab(component, time, time_slice)
        values, eids = result
        return ElementSlab(
            component=component, values=values,
            element_ids=index_to_fem(eids), time=time[t_idx],
        )

    def read_line_stations(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        self._refuse_on_mode(stage_id, f"line-station component {component!r}")
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = self._require_element_results(
            _child(grp, "RESULTS/ON_ELEMENTS"),
            family="line-station", component=component, stage_id=stage_id,
        )
        bucket_ids, index_to_fem = self._element_read_translation(element_ids)
        result = _eio.read_line_station_slab(
            on_e, component,
            t_idx=t_idx, element_ids=bucket_ids,
            model_elements=_child(grp, "MODEL/ELEMENTS"),
        )
        if result is None:
            return _empty_line_station_slab(component, time, t_idx)
        values, element_index, station_coord = result
        fem_index = index_to_fem(element_index)
        return LineStationSlab(
            component=component, values=values,
            element_index=fem_index,
            station_natural_coord=station_coord, time=time[t_idx],
            local_axes_quaternion=self._line_station_quaternions(
                stage_id, fem_index,
            ),
        )

    def _line_station_quaternions(
        self, stage_id: str, element_index: ndarray,
    ) -> "Optional[ndarray]":
        """Per-row beam quaternion for a line-station slab, or ``None``.

        Maps each row's element to its ``MODEL/LOCAL_AXES`` frame. Returns
        ``None`` when the stage records no frames at all (the diagram then
        falls back to node geometry); otherwise a ``(sum_S, 4)`` array with
        NaN rows for elements that have no recorded frame (per-element
        fallback).
        """
        recorded = self.read_local_axes(stage_id)  # only recorded ids
        if recorded.element_ids.size == 0:
            return None
        id_to_quat = {
            int(e): recorded.quaternions[k]
            for k, e in enumerate(recorded.element_ids)
        }
        nan_row = np.full(4, np.nan, dtype=np.float64)
        return np.stack([
            id_to_quat.get(int(e), nan_row) for e in element_index
        ])

    def read_gauss(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> GaussSlab:
        self._refuse_on_mode(stage_id, f"gauss component {component!r}")
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = self._require_element_results(
            _child(grp, "RESULTS/ON_ELEMENTS"),
            family="gauss (stress / strain)", component=component,
            stage_id=stage_id,
        )
        bucket_ids, index_to_fem = self._element_read_translation(element_ids)
        result = _eio.read_gauss_slab(
            on_e, _child(grp, "MODEL/ELEMENTS"), component,
            t_idx=t_idx, element_ids=bucket_ids,
        )
        if result is None:
            return _empty_gauss_slab(component, time, t_idx)
        values, element_index, natural_coords = result
        return GaussSlab(
            component=component, values=values,
            element_index=index_to_fem(element_index),
            natural_coords=natural_coords, local_axes_quaternion=None,
            time=time[t_idx],
        )

    def read_fibers(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None,
        gp_indices: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> FiberSlab:
        self._refuse_on_mode(stage_id, f"fiber component {component!r}")
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = self._require_element_results(
            _child(grp, "RESULTS/ON_ELEMENTS"),
            family="fiber", component=component, stage_id=stage_id,
        )
        bucket_ids, index_to_fem = self._element_read_translation(element_ids)
        result = _eio.read_fiber_slab(
            on_e, _child(grp, "MODEL/ELEMENTS"),
            _child(grp, "MODEL/SECTION_ASSIGNMENTS"), component,
            t_idx=t_idx, element_ids=bucket_ids,
            gp_indices=gp_indices,
        )
        if result is None:
            return _empty_fiber_slab(component, time, t_idx)
        values, ei, gpi, station_xi, y, z, area, mtag = result
        return FiberSlab(
            component=component, values=values,
            element_index=index_to_fem(ei), gp_index=gpi,
            y=y, z=z, area=area, material_tag=mtag, time=time[t_idx],
            station_natural_coord=station_xi,
        )

    def read_layers(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None,
        gp_indices: "Optional[ndarray]" = None,
        layer_indices: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> LayerSlab:
        """Layered-shell layer state, as the MPCO reader returns it.

        A layered shell writes under the fiber tokens with several
        components per layer (``fiber_stress_<c>`` / ``fiber_strain_<c>``);
        the frame comes from the file's node coordinates, since a
        ``.ladruno`` stores no shell ``LOCAL_AXES`` (#1995). A stage with
        no ``RESULTS/ON_ELEMENTS`` reads an empty slab and lists no layer
        component: the partitioned reader concatenates every rank's layer
        read, and a rank may hold no element.
        """
        self._refuse_on_mode(stage_id, f"layer component {component!r}")
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = _child(grp, "RESULTS/ON_ELEMENTS")
        if on_e is None:
            return _empty_layer_slab(component, time, t_idx)
        bucket_ids, index_to_fem = self._element_read_translation(element_ids)
        result = _eio.read_layer_slab(
            on_e, _child(grp, "MODEL"), component,
            t_idx=t_idx, element_ids=bucket_ids,
            gp_indices=gp_indices, layer_indices=layer_indices,
        )
        if result is None:
            return _empty_layer_slab(component, time, t_idx)
        values, ei, gpi, lyri, subi, thick, quat = result
        return LayerSlab(
            component=component, values=values,
            element_index=index_to_fem(ei), gp_index=gpi,
            layer_index=lyri, sub_gp_index=subi, thickness=thick,
            local_axes_quaternion=quat, time=time[t_idx],
        )

    def read_springs(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> SpringSlab:
        # A .ladruno has no distinct spring level: zeroLength force and
        # material stress/strain are already reachable via read_elements
        # (the ``force`` token) and read_gauss (``material.stress`` →
        # stress_xx). No bespoke spring slab. Always empty here.
        time = self.time_vector(stage_id)
        return _empty_spring_slab(
            component, time, resolve_time_slice(time_slice, time),
        )


def attach_model_nodes_ndf(
    reader: object, model_h5: "str | Path", ndm: int,
) -> None:
    """Attach ``model_h5``'s per-node ndf to a one-file reader (#1971);
    see :func:`_mpco_nodal_io.attach_nodes_ndf`."""
    _mnodal.attach_nodes_ndf(reader, LadrunoReader, model_h5, ndm)
