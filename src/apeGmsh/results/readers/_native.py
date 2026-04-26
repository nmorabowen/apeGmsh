"""NativeReader — reads apeGmsh native HDF5 result files.

Implements the ``ResultsReader`` protocol. The composite layer above
(Phase 2) calls these methods without knowing the backend type.

Phase 1 implementation is correct but not maximally lazy: each
``read_*`` reads the full dataset for its category into memory and
slices. Phase 2+ benchmarking can drive optimizations (chunked
reads, fancy h5py indexing) if needed.
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
)
from .._time import resolve_time_slice
from ..schema import _native
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData


class NativeReader:
    """Reader for apeGmsh native HDF5 result files."""

    def __init__(self, path: str | Path) -> None:
        import h5py

        self._path = Path(path)
        self._h5: "h5py.File" = h5py.File(self._path, "r")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "NativeReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Stage discovery
    # ------------------------------------------------------------------

    def stages(self) -> list[StageInfo]:
        if _native.STAGES_GROUP[1:] not in self._h5:
            return []
        out: list[StageInfo] = []
        stages_grp = self._h5[_native.STAGES_GROUP[1:]]
        for sid in stages_grp.keys():
            grp = stages_grp[sid]
            attrs = grp.attrs
            kind = str(attrs.get(_native.ATTR_STAGE_KIND, ""))
            time = grp[_native.DSET_TIME]
            info = StageInfo(
                id=sid,
                name=str(attrs.get(_native.ATTR_STAGE_NAME, "")),
                kind=kind,
                n_steps=int(time.shape[0]),
            )
            if kind == _native.KIND_MODE:
                info = StageInfo(
                    id=info.id,
                    name=info.name,
                    kind=info.kind,
                    n_steps=info.n_steps,
                    eigenvalue=float(attrs[_native.ATTR_EIGENVALUE]),
                    frequency_hz=float(attrs[_native.ATTR_FREQUENCY_HZ]),
                    period_s=float(attrs[_native.ATTR_PERIOD_S]),
                    mode_index=(
                        int(attrs[_native.ATTR_MODE_INDEX])
                        if _native.ATTR_MODE_INDEX in attrs else None
                    ),
                )
            out.append(info)
        return out

    def time_vector(self, stage_id: str) -> ndarray:
        return np.asarray(
            self._h5[_native.stage_time_path(stage_id)][...], dtype=np.float64,
        )

    def partitions(self, stage_id: str) -> list[str]:
        path = _native.partitions_path(stage_id)[1:]
        if path not in self._h5:
            return []
        return sorted(self._h5[path].keys())

    # ------------------------------------------------------------------
    # FEM access
    # ------------------------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        if _native.MODEL_GROUP[1:] not in self._h5:
            return None
        from ...mesh.FEMData import FEMData
        return FEMData.from_native_h5(self._h5[_native.MODEL_GROUP[1:]])

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        seen: set[str] = set()
        for pid in self.partitions(stage_id):
            seen.update(self._partition_components(stage_id, pid, level))
        return sorted(seen)

    def _partition_components(
        self, stage_id: str, partition_id: str, level: ResultLevel,
    ) -> set[str]:
        # Compare by enum *value* (string), not identity. Tests that
        # purge and re-import apeGmsh modules can produce two distinct
        # ``ResultLevel`` classes with the same names — identity
        # comparison fails between them. Values are stable.
        level_value = level.value

        if level_value == "nodes":
            path = _native.nodes_path(stage_id, partition_id)[1:]
            if path not in self._h5:
                return set()
            grp = self._h5[path]
            # Component datasets are not underscore-prefixed; underscore
            # names are reserved for index/metadata (e.g. _ids).
            return {k for k in grp.keys() if not k.startswith("_")}

        # Element-level: components live under groups inside the category.
        category_map = {
            "elements": _native.GROUP_NODAL_FORCES,
            "line_stations": _native.GROUP_LINE_STATIONS,
            "gauss": _native.GROUP_GAUSS_POINTS,
            "fibers": _native.GROUP_FIBERS,
            "layers": _native.GROUP_LAYERS,
        }
        category = category_map.get(level_value)
        if category is None:
            return set()
        cat_path = (
            f"{_native.elements_path(stage_id, partition_id)}/{category}"[1:]
        )
        if cat_path not in self._h5:
            return set()
        cat_grp = self._h5[cat_path]
        out: set[str] = set()
        for group_id in cat_grp.keys():
            sub = cat_grp[group_id]
            for k in sub.keys():
                if not k.startswith("_"):
                    out.add(k)
        return out

    # ------------------------------------------------------------------
    # Slab reads — nodes
    # ------------------------------------------------------------------

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        node_ids_filter = (
            None if node_ids is None
            else np.asarray(node_ids, dtype=np.int64)
        )

        chunks_values: list[ndarray] = []
        chunks_ids: list[ndarray] = []
        for pid in self.partitions(stage_id):
            nodes_path = _native.nodes_path(stage_id, pid)[1:]
            if nodes_path not in self._h5:
                continue
            nodes_grp = self._h5[nodes_path]
            if component not in nodes_grp:
                continue
            all_ids = np.asarray(
                nodes_grp[_native.DSET_IDS][...], dtype=np.int64,
            )

            if node_ids_filter is None:
                sel = slice(None)
                sel_ids = all_ids
            else:
                mask = np.isin(all_ids, node_ids_filter)
                if not mask.any():
                    continue
                sel = np.where(mask)[0]
                sel_ids = all_ids[sel]

            full = nodes_grp[component][...]                 # (T, N)
            values = full[t_idx]
            if not isinstance(sel, slice):
                values = values[:, sel]

            chunks_values.append(values)
            chunks_ids.append(sel_ids)

        values = (
            np.concatenate(chunks_values, axis=1) if chunks_values
            else np.zeros((t_idx.size, 0), dtype=np.float64)
        )
        ids_out = (
            np.concatenate(chunks_ids) if chunks_ids
            else np.array([], dtype=np.int64)
        )
        return NodeSlab(
            component=component, values=values,
            node_ids=ids_out, time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Slab reads — Gauss points
    # ------------------------------------------------------------------

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        eid_filter = (
            None if element_ids is None
            else np.asarray(element_ids, dtype=np.int64)
        )

        chunks_v, chunks_eidx, chunks_nc = [], [], []
        chunks_quat: list[ndarray] = []
        any_quat = False

        for pid in self.partitions(stage_id):
            gauss_root_path = (
                f"{_native.elements_path(stage_id, pid)}/"
                f"{_native.GROUP_GAUSS_POINTS}"
            )[1:]
            if gauss_root_path not in self._h5:
                continue
            gauss_root = self._h5[gauss_root_path]
            for group_id in gauss_root.keys():
                grp = gauss_root[group_id]
                if component not in grp:
                    continue
                eidx = np.asarray(
                    grp[_native.DSET_ELEMENT_INDEX][...], dtype=np.int64,
                )
                nc = np.asarray(
                    grp[_native.DSET_NATURAL_COORDS][...], dtype=np.float64,
                )
                n_gp = nc.shape[0]

                if eid_filter is None:
                    elem_sel = np.arange(eidx.size)
                else:
                    mask = np.isin(eidx, eid_filter)
                    if not mask.any():
                        continue
                    elem_sel = np.where(mask)[0]

                sel_eidx = eidx[elem_sel]
                full = grp[component][...]                  # (T, E_g, n_gp)
                values_t = full[t_idx]
                values_es = values_t[:, elem_sel, :]
                T = values_es.shape[0]
                E_sel = sel_eidx.size
                flat = values_es.reshape(T, E_sel * n_gp)

                row_eidx = np.repeat(sel_eidx, n_gp)
                row_nc = np.tile(nc, (E_sel, 1))

                chunks_v.append(flat)
                chunks_eidx.append(row_eidx)
                chunks_nc.append(row_nc)

                if _native.DSET_LOCAL_AXES_QUATERNION in grp:
                    any_quat = True
                    quat = np.asarray(
                        grp[_native.DSET_LOCAL_AXES_QUATERNION][...],
                        dtype=np.float64,
                    )
                    chunks_quat.append(np.repeat(quat[elem_sel], n_gp, axis=0))

        values = (
            np.concatenate(chunks_v, axis=1) if chunks_v
            else np.zeros((t_idx.size, 0), dtype=np.float64)
        )
        eidx_out = (
            np.concatenate(chunks_eidx) if chunks_eidx
            else np.array([], dtype=np.int64)
        )
        nc_out = (
            np.concatenate(chunks_nc, axis=0) if chunks_nc
            else np.zeros((0, 0), dtype=np.float64)
        )
        quat_out = (
            np.concatenate(chunks_quat, axis=0) if any_quat and chunks_quat
            else None
        )
        return GaussSlab(
            component=component, values=values,
            element_index=eidx_out, natural_coords=nc_out,
            local_axes_quaternion=quat_out,
            time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Slab reads — fibers
    # ------------------------------------------------------------------

    def read_fibers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> FiberSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        eid_filter = (
            None if element_ids is None
            else np.asarray(element_ids, dtype=np.int64)
        )
        gp_filter = (
            None if gp_indices is None
            else np.asarray(gp_indices, dtype=np.int64)
        )

        v, eidx, gpi, ys, zs, areas, mtags = [], [], [], [], [], [], []

        for pid in self.partitions(stage_id):
            root_path = (
                f"{_native.elements_path(stage_id, pid)}/"
                f"{_native.GROUP_FIBERS}"
            )[1:]
            if root_path not in self._h5:
                continue
            root = self._h5[root_path]
            for group_id in root.keys():
                grp = root[group_id]
                if component not in grp:
                    continue
                e = np.asarray(grp[_native.DSET_ELEMENT_INDEX][...], dtype=np.int64)
                g = np.asarray(grp[_native.DSET_GP_INDEX][...], dtype=np.int64)
                y = np.asarray(grp[_native.DSET_Y][...], dtype=np.float64)
                z = np.asarray(grp[_native.DSET_Z][...], dtype=np.float64)
                a = np.asarray(grp[_native.DSET_AREA][...], dtype=np.float64)
                m = np.asarray(grp[_native.DSET_MATERIAL_TAG][...], dtype=np.int64)

                mask = np.ones(e.size, dtype=bool)
                if eid_filter is not None:
                    mask &= np.isin(e, eid_filter)
                if gp_filter is not None:
                    mask &= np.isin(g, gp_filter)
                if not mask.any():
                    continue
                sel = np.where(mask)[0]

                full = grp[component][...]                   # (T, sum_F)
                v.append(full[t_idx][:, sel])
                eidx.append(e[sel])
                gpi.append(g[sel])
                ys.append(y[sel])
                zs.append(z[sel])
                areas.append(a[sel])
                mtags.append(m[sel])

        return FiberSlab(
            component=component,
            values=(
                np.concatenate(v, axis=1) if v
                else np.zeros((t_idx.size, 0), dtype=np.float64)
            ),
            element_index=_concat_or_empty(eidx, np.int64),
            gp_index=_concat_or_empty(gpi, np.int64),
            y=_concat_or_empty(ys, np.float64),
            z=_concat_or_empty(zs, np.float64),
            area=_concat_or_empty(areas, np.float64),
            material_tag=_concat_or_empty(mtags, np.int64),
            time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Slab reads — layers
    # ------------------------------------------------------------------

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
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        e_filter = element_ids if element_ids is None else np.asarray(element_ids, dtype=np.int64)
        g_filter = gp_indices if gp_indices is None else np.asarray(gp_indices, dtype=np.int64)
        l_filter = layer_indices if layer_indices is None else np.asarray(layer_indices, dtype=np.int64)

        v: list[ndarray] = []
        e_idx: list[ndarray] = []
        gp_i: list[ndarray] = []
        l_idx: list[ndarray] = []
        sub_gp: list[ndarray] = []
        thk: list[ndarray] = []
        quat: list[ndarray] = []

        for pid in self.partitions(stage_id):
            root_path = (
                f"{_native.elements_path(stage_id, pid)}/"
                f"{_native.GROUP_LAYERS}"
            )[1:]
            if root_path not in self._h5:
                continue
            root = self._h5[root_path]
            for group_id in root.keys():
                grp = root[group_id]
                if component not in grp:
                    continue
                e = np.asarray(grp[_native.DSET_ELEMENT_INDEX][...], dtype=np.int64)
                g = np.asarray(grp[_native.DSET_GP_INDEX][...], dtype=np.int64)
                lay = np.asarray(grp[_native.DSET_LAYER_INDEX][...], dtype=np.int64)
                sg = np.asarray(grp[_native.DSET_SUB_GP_INDEX][...], dtype=np.int64)
                th = np.asarray(grp[_native.DSET_THICKNESS][...], dtype=np.float64)
                q = np.asarray(grp[_native.DSET_LOCAL_AXES_QUATERNION][...], dtype=np.float64)

                mask = np.ones(e.size, dtype=bool)
                if e_filter is not None:
                    mask &= np.isin(e, e_filter)
                if g_filter is not None:
                    mask &= np.isin(g, g_filter)
                if l_filter is not None:
                    mask &= np.isin(lay, l_filter)
                if not mask.any():
                    continue
                sel = np.where(mask)[0]

                full = grp[component][...]
                v.append(full[t_idx][:, sel])
                e_idx.append(e[sel])
                gp_i.append(g[sel])
                l_idx.append(lay[sel])
                sub_gp.append(sg[sel])
                thk.append(th[sel])
                quat.append(q[sel])

        return LayerSlab(
            component=component,
            values=(
                np.concatenate(v, axis=1) if v
                else np.zeros((t_idx.size, 0), dtype=np.float64)
            ),
            element_index=_concat_or_empty(e_idx, np.int64),
            gp_index=_concat_or_empty(gp_i, np.int64),
            layer_index=_concat_or_empty(l_idx, np.int64),
            sub_gp_index=_concat_or_empty(sub_gp, np.int64),
            thickness=_concat_or_empty(thk, np.float64),
            local_axes_quaternion=(
                np.concatenate(quat, axis=0) if quat
                else np.zeros((0, 4), dtype=np.float64)
            ),
            time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Slab reads — line stations
    # ------------------------------------------------------------------

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        eid_filter = (
            None if element_ids is None
            else np.asarray(element_ids, dtype=np.int64)
        )

        v, e_idx, snc_rows = [], [], []
        for pid in self.partitions(stage_id):
            root_path = (
                f"{_native.elements_path(stage_id, pid)}/"
                f"{_native.GROUP_LINE_STATIONS}"
            )[1:]
            if root_path not in self._h5:
                continue
            root = self._h5[root_path]
            for group_id in root.keys():
                grp = root[group_id]
                if component not in grp:
                    continue
                eidx = np.asarray(grp[_native.DSET_ELEMENT_INDEX][...], dtype=np.int64)
                snc = np.asarray(grp[_native.DSET_STATION_NATURAL_COORD][...], dtype=np.float64)
                n_st = snc.size

                if eid_filter is None:
                    elem_sel = np.arange(eidx.size)
                else:
                    mask = np.isin(eidx, eid_filter)
                    if not mask.any():
                        continue
                    elem_sel = np.where(mask)[0]

                sel_eidx = eidx[elem_sel]
                full = grp[component][...]                    # (T, E_g, n_st)
                values_t = full[t_idx]
                values_es = values_t[:, elem_sel, :]
                T = values_es.shape[0]
                E_sel = sel_eidx.size
                flat = values_es.reshape(T, E_sel * n_st)

                v.append(flat)
                e_idx.append(np.repeat(sel_eidx, n_st))
                snc_rows.append(np.tile(snc, E_sel))

        return LineStationSlab(
            component=component,
            values=(
                np.concatenate(v, axis=1) if v
                else np.zeros((t_idx.size, 0), dtype=np.float64)
            ),
            element_index=_concat_or_empty(e_idx, np.int64),
            station_natural_coord=_concat_or_empty(snc_rows, np.float64),
            time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Slab reads — element-level (per-element-node)
    # ------------------------------------------------------------------

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        eid_filter = (
            None if element_ids is None
            else np.asarray(element_ids, dtype=np.int64)
        )

        # Only one (class_tag, npe) group can be returned per call because
        # the values shape is (T, E, npe). Mixing different npes would
        # produce ragged arrays. If multiple groups carry the same component,
        # we concat along the element axis only when their npes match.
        combined: dict[int, dict] = {}    # npe -> {"v": [...], "e": [...]}

        for pid in self.partitions(stage_id):
            root_path = (
                f"{_native.elements_path(stage_id, pid)}/"
                f"{_native.GROUP_NODAL_FORCES}"
            )[1:]
            if root_path not in self._h5:
                continue
            root = self._h5[root_path]
            for group_id in root.keys():
                grp = root[group_id]
                if component not in grp:
                    continue
                eidx = np.asarray(grp[_native.DSET_ELEMENT_INDEX][...], dtype=np.int64)
                full = grp[component][...]                     # (T, E_g, npe)
                npe = full.shape[2]

                if eid_filter is None:
                    elem_sel = np.arange(eidx.size)
                else:
                    mask = np.isin(eidx, eid_filter)
                    if not mask.any():
                        continue
                    elem_sel = np.where(mask)[0]

                sel_eidx = eidx[elem_sel]
                values_es = full[t_idx][:, elem_sel, :]
                bucket = combined.setdefault(npe, {"v": [], "e": []})
                bucket["v"].append(values_es)
                bucket["e"].append(sel_eidx)

        if not combined:
            return ElementSlab(
                component=component,
                values=np.zeros((t_idx.size, 0, 0), dtype=np.float64),
                element_ids=np.array([], dtype=np.int64),
                time=time[t_idx],
            )

        if len(combined) > 1:
            raise ValueError(
                f"read_elements: component {component!r} spans multiple "
                f"npes ({sorted(combined.keys())}); element-level slabs "
                f"require a single npe. Filter element_ids to a single "
                f"element class first."
            )

        only_npe = next(iter(combined))
        bucket = combined[only_npe]
        return ElementSlab(
            component=component,
            values=np.concatenate(bucket["v"], axis=1),
            element_ids=np.concatenate(bucket["e"]),
            time=time[t_idx],
        )


# =====================================================================
# Helpers
# =====================================================================

def _concat_or_empty(chunks: list[ndarray], dtype) -> ndarray:
    if chunks:
        return np.concatenate(chunks)
    return np.array([], dtype=dtype)
