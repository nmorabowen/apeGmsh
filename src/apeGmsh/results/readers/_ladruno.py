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
section-level line stations + fiber stress/strain (L2b-3). Layer and spring
reads return empty slabs by design — a ``.ladruno`` has no distinct layer or
spring level (layered shells serialise as fiber sections; zeroLength state
flows through the element/gauss reads).

Key layout facts (verified against fork build ``605affeb``, FORMAT_VERSION 1):

* ``INFO`` attrs ``GENERATOR="Ladruno"``, ``FORMAT_VERSION`` (int).
* ``MODEL_STAGE[<k>]`` — integer-indexed stage groups, attr ``KIND``
  (``static``/``transient``/``eigen``).
* ``RESULTS/ON_NODES/<RESULT>`` (e.g. ``DISPLACEMENT``) — attr
  ``COMPONENTS`` (comma-joined labels ``"Ux,Uy"``); datasets ``ID[n,1]``,
  ``DATA[T,n,nComp]``, ``TIME[T]``, ``STEP[T]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional

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
from .._time import resolve_time_slice
from ..schema._versions import LADRUNO_SUPPORTED_FORMAT_VERSIONS
from . import _ladruno_element_io as _eio
from ._mpco import (
    _empty_element_slab,
    _empty_fiber_slab,
    _empty_gauss_slab,
    _empty_layer_slab,
    _empty_line_station_slab,
    _empty_node_slab,
    _empty_spring_slab,
)
from ._mpco_translation import canonical_node_component
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData
    from ...opensees.opensees_model import OpenSeesModel
    from .._slabs import LocalAxes
    from ._tag_translation import ElementTagTranslator


_STAGE_PREFIX = "MODEL_STAGE["


def _stage_sort_key(name: str) -> "tuple[int, str]":
    """Numeric-stamp sort key for ``MODEL_STAGE[<k>]`` group names."""
    try:
        return (int(name[len(_STAGE_PREFIX):].rstrip("]")), name)
    except ValueError:
        return (2**31, name)


# Ladruno stage KIND → protocol StageInfo.kind ("transient"|"static"|"mode").
_KIND_MAP = {"static": "static", "transient": "transient", "eigen": "mode"}


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
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL
        self._tag_map: "Optional[ElementTagTranslator]" = None

    # -- identity / lifecycle ------------------------------------------

    def _validate_identity(self) -> None:
        info = self._h5.get("INFO")
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
        """Store the fem_eid↔ops-tag translator (composed models, ADR 0043).

        Element/gauss/line reads relabel ops↔fem_eid through it for a
        composed model (ADR 0043), exactly like :class:`MPCOReader`.
        """
        self._tag_map = tag_map

    def _ids_to_ops(
        self, element_ids: "Optional[ndarray]",
    ) -> "Optional[ndarray]":
        """Translate an incoming ``fem_eid`` filter to ops tags."""
        if self._tag_map is None:
            return element_ids
        return self._tag_map.to_ops(element_ids)

    def _index_to_fem(self, element_index: ndarray) -> ndarray:
        """Relabel an ops-keyed ``element_index`` back to ``fem_eid``."""
        if self._tag_map is None:
            return element_index
        out = self._tag_map.to_fem(element_index)
        assert out is not None  # non-None input → non-None (to_fem contract)
        return out

    # -- stages / time -------------------------------------------------

    def stages(self) -> list[StageInfo]:
        if self._stage_cache is not None:
            return list(self._stage_cache)
        out: list[StageInfo] = []
        # Sort by the numeric stamp inside ``MODEL_STAGE[<k>]`` —
        # lexicographic sorting would put MODEL_STAGE[10] before
        # MODEL_STAGE[2] once a run reaches ten stages.
        names = sorted(
            (k for k in self._h5.keys() if k.startswith(_STAGE_PREFIX)),
            key=_stage_sort_key,
        )
        for i, name in enumerate(names):
            stage_id = f"stage_{i}"
            self._stage_to_grp[stage_id] = name
            grp = self._h5[name]
            kind_raw = _decode(grp.attrs["KIND"]) if "KIND" in grp.attrs else "static"
            kind = _KIND_MAP.get(kind_raw, "static")
            time = self._time_for_group(grp)
            out.append(
                StageInfo(
                    id=stage_id, name=name, kind=kind, n_steps=int(time.size),
                )
            )
        self._stage_cache = out
        return list(out)

    def _resolve_stage_group(self, stage_id: str) -> "h5py.Group":
        if self._stage_cache is None:
            self.stages()
        if stage_id not in self._stage_to_grp:
            raise KeyError(
                f"Unknown stage_id {stage_id!r}; available: "
                f"{sorted(self._stage_to_grp)}."
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
        return self._time_for_group(self._resolve_stage_group(stage_id))

    def partitions(self, stage_id: str) -> list[str]:
        return ["partition_0"]

    # -- model / fem ---------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        if not isinstance(self._fem_cache, _Sentinel):
            return self._fem_cache  # type: ignore[return-value]
        if self._stage_cache is None:
            self.stages()
        if not self._stage_to_grp:
            self._fem_cache = None
            return None
        # Use the last stage's MODEL (most up-to-date geometry).
        last_id = f"stage_{len(self._stage_to_grp) - 1}"
        grp = self._h5[self._stage_to_grp[last_id]]
        model_grp = grp.get("MODEL")
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

    def _spatial_dim(self) -> int:
        info = self._h5.get("INFO")
        if info is not None and "SPATIAL_DIM" in info.attrs:
            return _attr_int(info.attrs, "SPATIAL_DIM", default=3)
        return 3

    # -- components / reads --------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        grp = self._resolve_stage_group(stage_id)
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
            return sorted(_eio.element_available(on_elements))
        if level is ResultLevel.FIBERS:
            return sorted(_eio.fiber_available(on_elements))
        # LAYERS / SPRINGS: a .ladruno has no distinct layer or spring
        # level — layered shells serialise as fiber sections (read via
        # read_fibers) and zeroLength material/force state is reachable
        # through read_gauss / read_elements. See _ladruno_element_io.
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

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: "Optional[ndarray]" = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        grp = self._resolve_stage_group(stage_id)
        time = self._time_for_group(grp)
        t_idx = resolve_time_slice(time_slice, time)
        on_nodes = _child(grp, "RESULTS/ON_NODES")
        if on_nodes is None:
            return _empty_node_slab(component, time, t_idx)
        loc = self._locate_node_component(on_nodes, component)
        if loc is None:
            return _empty_node_slab(component, time, t_idx)
        res_name, col = loc
        res = on_nodes[res_name]
        ids = np.asarray(res["ID"][...], dtype=np.int64).flatten()
        # DATA is chunked [T × nIds × nComp]; read the requested column,
        # then the requested time steps + node mask in numpy (files are
        # small and h5py fancy-indexing across axes is limited).
        data = np.asarray(res["DATA"][...], dtype=np.float64)
        if node_ids is not None:
            want = np.asarray(node_ids, dtype=np.int64)
            mask = np.isin(ids, want)
        else:
            mask = np.ones(ids.size, dtype=bool)
        sel_ids = ids[mask]
        vals = data[t_idx][:, :, col][:, mask]  # (T, N)
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
        data = np.asarray(eg["DATA"][...], dtype=np.float64)  # (T, nIds, nComp)
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
        values = data[t_idx][:, row, :]  # (T, nComp)
        return cols, values, time[t_idx]

    # -- node envelopes (recorder -envelope, Ladruno-only extension) ---

    def available_node_envelope_components(self, stage_id: str) -> list[str]:
        """Canonical node components present under ``RESULTS/ENVELOPES/
        ON_NODES`` (the recorder's ``-envelope`` mode). Empty when the file
        was not recorded with ``-envelope`` (the time-series ``ON_NODES``
        path then carries the data instead)."""
        grp = self._resolve_stage_group(stage_id)
        env = _child(grp, "RESULTS/ENVELOPES/ON_NODES")
        if env is None:
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
        """
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

        def _col(name: str, dtype) -> ndarray:
            return np.asarray(res[name][...], dtype=dtype)[mask, col]

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

    def read_elements(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> ElementSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        on_e = self._on_elements(stage_id)
        if on_e is None:
            return _empty_element_slab(component, time, time_slice)
        result = _eio.read_element_slab(
            on_e, component,
            t_idx=t_idx, element_ids=self._ids_to_ops(element_ids),
        )
        if result is None:
            return _empty_element_slab(component, time, time_slice)
        values, eids = result
        return ElementSlab(
            component=component, values=values,
            element_ids=self._index_to_fem(eids), time=time[t_idx],
        )

    def read_line_stations(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = _child(grp, "RESULTS/ON_ELEMENTS")
        if on_e is None:
            return _empty_line_station_slab(component, time, t_idx)
        result = _eio.read_line_station_slab(
            on_e, component,
            t_idx=t_idx, element_ids=self._ids_to_ops(element_ids),
            model_elements=_child(grp, "MODEL/ELEMENTS"),
        )
        if result is None:
            return _empty_line_station_slab(component, time, t_idx)
        values, element_index, station_coord = result
        fem_index = self._index_to_fem(element_index)
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
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = _child(grp, "RESULTS/ON_ELEMENTS")
        if on_e is None:
            return _empty_gauss_slab(component, time, t_idx)
        result = _eio.read_gauss_slab(
            on_e, _child(grp, "MODEL/ELEMENTS"), component,
            t_idx=t_idx, element_ids=self._ids_to_ops(element_ids),
        )
        if result is None:
            return _empty_gauss_slab(component, time, t_idx)
        values, element_index, natural_coords = result
        return GaussSlab(
            component=component, values=values,
            element_index=self._index_to_fem(element_index),
            natural_coords=natural_coords, local_axes_quaternion=None,
            time=time[t_idx],
        )

    def read_fibers(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None,
        gp_indices: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> FiberSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        grp = self._resolve_stage_group(stage_id)
        on_e = _child(grp, "RESULTS/ON_ELEMENTS")
        if on_e is None:
            return _empty_fiber_slab(component, time, t_idx)
        result = _eio.read_fiber_slab(
            on_e, _child(grp, "MODEL/ELEMENTS"),
            _child(grp, "MODEL/SECTION_ASSIGNMENTS"), component,
            t_idx=t_idx, element_ids=self._ids_to_ops(element_ids),
            gp_indices=gp_indices,
        )
        if result is None:
            return _empty_fiber_slab(component, time, t_idx)
        values, ei, gpi, station_xi, y, z, area, mtag = result
        return FiberSlab(
            component=component, values=values,
            element_index=self._index_to_fem(ei), gp_index=gpi,
            y=y, z=z, area=area, material_tag=mtag, time=time[t_idx],
            station_natural_coord=station_xi,
        )

    def read_layers(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None,
        gp_indices: "Optional[ndarray]" = None,
        layer_indices: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> LayerSlab:
        # A .ladruno has no distinct layer level: a layered-shell section
        # serialises as a fiber section (KIND='fiber', layers == fibers),
        # so layer state is read through read_fibers. As of recorder PR #200
        # layered shells emit per-layer material stress under the
        # ``material.fiber.<resp>`` verb (shells swap section→material); the
        # Ladruno fiber token map now keys on both spellings, so read_fibers
        # picks shells up. Always empty here.
        time = self.time_vector(stage_id)
        return _empty_layer_slab(
            component, time, resolve_time_slice(time_slice, time),
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
