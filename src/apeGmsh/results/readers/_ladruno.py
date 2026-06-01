"""``LadrunoReader`` — reads a single ``.ladruno`` HDF5 file.

The Ladruno recorder is the fork's *canonical* recorder; this reader is
the sibling of :class:`apeGmsh.results.readers._mpco.MPCOReader`. Per the
canonicity contract it reads the file **direct** — HDF5 groups → the
``ResultsReader`` protocol slabs, lazily, with no transcode to a derived
on-disk representation.

L2a scope (this slice): file identity + ``FORMAT_VERSION`` window, stage
enumeration, time vectors, the self-describing FEM (``MODEL/`` →
:meth:`FEMData.from_ladruno_model`), and **nodal** result reads
(``RESULTS/ON_NODES`` — chunked ``DATA[T×nIds×nComp]``). Element / gauss /
line-station / fiber / layer / spring reads return empty slabs until L2b.

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
    SpringSlab,
)
from .._time import resolve_time_slice
from ..schema._versions import LADRUNO_SUPPORTED_FORMAT_VERSIONS
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
    from ._tag_translation import ElementTagTranslator


_STAGE_PREFIX = "MODEL_STAGE["

# Ladruno stage KIND → protocol StageInfo.kind ("transient"|"static"|"mode").
_KIND_MAP = {"static": "static", "transient": "transient", "eigen": "mode"}


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

        L2a does no element reads, so the map is held but unused; L2b's
        element/gauss reads consult it like :class:`MPCOReader`.
        """
        self._tag_map = tag_map

    # -- stages / time -------------------------------------------------

    def stages(self) -> list[StageInfo]:
        if self._stage_cache is not None:
            return list(self._stage_cache)
        out: list[StageInfo] = []
        names = sorted(k for k in self._h5.keys() if k.startswith(_STAGE_PREFIX))
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
        if level is not ResultLevel.NODES:
            return []  # element/gauss/line/... land in L2b
        grp = self._resolve_stage_group(stage_id)
        on_nodes = _child(grp, "RESULTS/ON_NODES")
        if on_nodes is None:
            return []
        out: list[str] = []
        for res_name in on_nodes:
            res = on_nodes[res_name]
            if "COMPONENTS" not in res.attrs:
                continue
            labels = _decode(res.attrs["COMPONENTS"]).split(",")
            for label in labels:
                canon = canonical_node_component(res_name, label.strip())
                if canon is not None and canon not in out:
                    out.append(canon)
        return out

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

    # -- element-level reads (empty until L2b) -------------------------

    def read_elements(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> ElementSlab:
        return _empty_element_slab(
            component, self.time_vector(stage_id), time_slice,
        )

    def read_line_stations(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        time = self.time_vector(stage_id)
        return _empty_line_station_slab(
            component, time, resolve_time_slice(time_slice, time),
        )

    def read_gauss(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> GaussSlab:
        time = self.time_vector(stage_id)
        return _empty_gauss_slab(
            component, time, resolve_time_slice(time_slice, time),
        )

    def read_fibers(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None,
        gp_indices: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> FiberSlab:
        time = self.time_vector(stage_id)
        return _empty_fiber_slab(
            component, time, resolve_time_slice(time_slice, time),
        )

    def read_layers(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None,
        gp_indices: "Optional[ndarray]" = None,
        layer_indices: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> LayerSlab:
        time = self.time_vector(stage_id)
        return _empty_layer_slab(
            component, time, resolve_time_slice(time_slice, time),
        )

    def read_springs(
        self, stage_id: str, component: str, *,
        element_ids: "Optional[ndarray]" = None, time_slice: TimeSlice = None,
    ) -> SpringSlab:
        time = self.time_vector(stage_id)
        return _empty_spring_slab(
            component, time, resolve_time_slice(time_slice, time),
        )
