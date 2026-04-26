"""NativeWriter — produces apeGmsh native HDF5 result files.

Bulk-write API. The user typically calls higher-level entry points
(``Results.from_recorders`` in Phase 6, ``DomainCapture`` in Phase 7),
which use this writer internally.

Usage
-----
::

    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture")

        # Stage 1 — transient
        sid = w.begin_stage(name="gravity", kind="transient",
                            time=time_grav)
        w.write_nodes(sid, "partition_0",
                      node_ids=ids,
                      components={"displacement_x": ux, ...})
        w.write_gauss_group(sid, "partition_0", "group_0",
                            class_tag=4, int_rule=1,
                            element_index=eidx,
                            natural_coords=nc,
                            components={"stress_xx": sxx})
        w.end_stage()

        # Stage 2 — mode shape (T=1, kind="mode")
        sid = w.begin_stage(name="mode_1", kind="mode",
                            time=np.array([0.0]),
                            eigenvalue=158.7,
                            frequency_hz=2.005,
                            period_s=0.499,
                            mode_index=1)
        w.write_nodes(sid, "partition_0",
                      node_ids=ids,
                      components={"displacement_x": shape_x[None, :], ...})
        w.end_stage()
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ..schema import _native, _versions

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData


class NativeWriter:
    """Bulk writer for apeGmsh native HDF5 result files.

    The writer holds an open ``h5py.File`` for its lifetime; use as a
    context manager or call ``open()`` / ``close()`` explicitly.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._h5: Optional["h5py.File"] = None
        self._current_stage: Optional[str] = None
        self._stage_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "NativeWriter":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def open(
        self,
        *,
        fem: Optional["FEMData"] = None,
        source_type: str = _native.SOURCE_DOMAIN_CAPTURE,
        source_path: str = "",
        analysis_label: str = "",
    ) -> None:
        """Create the file, write root attrs, embed FEMData if provided."""
        import h5py

        if self._h5 is not None:
            raise RuntimeError(f"NativeWriter for {self._path} already open.")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self._path, "w")

        h5 = self._h5
        h5.attrs[_native.ATTR_SCHEMA_VERSION] = _versions.SCHEMA_VERSION
        h5.attrs[_native.ATTR_SOURCE_TYPE] = source_type
        h5.attrs[_native.ATTR_SOURCE_PATH] = source_path
        h5.attrs[_native.ATTR_CREATED_AT] = (
            datetime.now(tz=timezone.utc).isoformat()
        )
        h5.attrs[_native.ATTR_APEGMSH_VERSION] = _apegmsh_version()
        h5.attrs[_native.ATTR_ANALYSIS_LABEL] = analysis_label

        # Empty stages container
        h5.create_group(_native.STAGES_GROUP[1:])

        if fem is not None:
            self.write_model(fem)

    def close(self) -> None:
        if self._h5 is None:
            return
        self._h5.close()
        self._h5 = None
        self._current_stage = None

    # ------------------------------------------------------------------
    # Embedded FEMData snapshot
    # ------------------------------------------------------------------

    def write_model(self, fem: "FEMData") -> None:
        h5 = self._require_open()
        if _native.MODEL_GROUP[1:] in h5:
            raise RuntimeError("/model/ already written.")
        model_grp = h5.create_group(_native.MODEL_GROUP[1:])
        fem.to_native_h5(model_grp)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def begin_stage(
        self,
        *,
        name: str,
        kind: str,
        time: ndarray,
        stage_id: Optional[str] = None,
        eigenvalue: Optional[float] = None,
        frequency_hz: Optional[float] = None,
        period_s: Optional[float] = None,
        mode_index: Optional[int] = None,
    ) -> str:
        """Open a new stage. Returns the stage_id (auto-generated if omitted)."""
        h5 = self._require_open()
        if self._current_stage is not None:
            raise RuntimeError(
                f"Stage {self._current_stage!r} still open — call end_stage()."
            )
        if kind not in _native.ALL_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_native.ALL_KINDS)} (got {kind!r})."
            )

        if stage_id is None:
            stage_id = _native.stage_id(self._stage_count)
        self._stage_count += 1

        stage_grp = h5.create_group(_native.stage_path(stage_id)[1:])
        stage_grp.attrs[_native.ATTR_STAGE_NAME] = name
        stage_grp.attrs[_native.ATTR_STAGE_KIND] = kind

        time_arr = np.asarray(time, dtype=np.float64)
        stage_grp.create_dataset(_native.DSET_TIME, data=time_arr)

        if kind == _native.KIND_MODE:
            if eigenvalue is None or frequency_hz is None or period_s is None:
                raise ValueError(
                    "kind='mode' requires eigenvalue, frequency_hz, period_s."
                )
            stage_grp.attrs[_native.ATTR_EIGENVALUE] = float(eigenvalue)
            stage_grp.attrs[_native.ATTR_FREQUENCY_HZ] = float(frequency_hz)
            stage_grp.attrs[_native.ATTR_PERIOD_S] = float(period_s)
            if mode_index is not None:
                stage_grp.attrs[_native.ATTR_MODE_INDEX] = int(mode_index)

        # Pre-create the partitions container so writes can require_group it.
        stage_grp.create_group(_native.GROUP_PARTITIONS)

        self._current_stage = stage_id
        return stage_id

    def end_stage(self) -> None:
        if self._current_stage is None:
            raise RuntimeError("No stage open.")
        self._current_stage = None

    # ------------------------------------------------------------------
    # Bulk writes — nodes
    # ------------------------------------------------------------------

    def write_nodes(
        self,
        stage_id: str,
        partition_id: str,
        *,
        node_ids: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        """Write nodal results for one partition.

        ``components[name]`` must have shape ``(T, N)`` matching the
        stage's time vector length and the node count.
        """
        nodes_grp = self._require_partition(stage_id, partition_id).require_group(
            _native.GROUP_NODES,
        )
        node_ids = np.asarray(node_ids, dtype=np.int64)
        _write_or_validate_ids(nodes_grp, _native.DSET_IDS, node_ids)
        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != node_ids.size:
                raise ValueError(
                    f"Component {comp_name!r} has {arr.shape[1]} nodes but "
                    f"node_ids has {node_ids.size}."
                )
            nodes_grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — Gauss points
    # ------------------------------------------------------------------

    def write_gauss_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        class_tag: int,
        int_rule: int = 0,
        custom_rule_idx: int = 0,
        element_index: ndarray,
        natural_coords: ndarray,
        components: dict[str, ndarray],
        local_axes_quaternion: Optional[ndarray] = None,
    ) -> None:
        """Write one ``(class_tag, int_rule)`` Gauss group.

        Shapes:
        - ``element_index``: ``(E_g,)``
        - ``natural_coords``: ``(n_GP_g, dim)``
        - ``components[name]``: ``(T, E_g, n_GP_g)``
        - ``local_axes_quaternion``: ``(E_g, 4)``, optional (shells)
        """
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_GAUSS_POINTS, group_id,
        )
        grp.attrs[_native.ATTR_CLASS_TAG] = int(class_tag)
        grp.attrs[_native.ATTR_INT_RULE] = int(int_rule)
        grp.attrs[_native.ATTR_CUSTOM_RULE_IDX] = int(custom_rule_idx)

        eidx = np.asarray(element_index, dtype=np.int64)
        nc = np.asarray(natural_coords, dtype=np.float64)
        grp.create_dataset(_native.DSET_ELEMENT_INDEX, data=eidx)
        grp.create_dataset(_native.DSET_NATURAL_COORDS, data=nc)
        if local_axes_quaternion is not None:
            grp.create_dataset(
                _native.DSET_LOCAL_AXES_QUATERNION,
                data=np.asarray(local_axes_quaternion, dtype=np.float64),
            )

        n_gp = nc.shape[0]
        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != eidx.size or arr.shape[2] != n_gp:
                raise ValueError(
                    f"Component {comp_name!r} has shape {arr.shape}; expected "
                    f"(T, {eidx.size}, {n_gp})."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — fibers
    # ------------------------------------------------------------------

    def write_fibers_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        section_tag: int,
        section_class: str,
        element_index: ndarray,
        gp_index: ndarray,
        y: ndarray,
        z: ndarray,
        area: ndarray,
        material_tag: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_FIBERS, group_id,
        )
        grp.attrs[_native.ATTR_SECTION_TAG] = int(section_tag)
        grp.attrs[_native.ATTR_SECTION_CLASS] = section_class

        eidx = np.asarray(element_index, dtype=np.int64)
        n = eidx.size
        for name, arr, dtype in [
            (_native.DSET_ELEMENT_INDEX, eidx, np.int64),
            (_native.DSET_GP_INDEX, gp_index, np.int64),
            (_native.DSET_Y, y, np.float64),
            (_native.DSET_Z, z, np.float64),
            (_native.DSET_AREA, area, np.float64),
            (_native.DSET_MATERIAL_TAG, material_tag, np.int64),
        ]:
            a = np.asarray(arr, dtype=dtype)
            if a.size != n:
                raise ValueError(
                    f"Fiber index dataset {name!r} has size {a.size}; "
                    f"expected {n}."
                )
            grp.create_dataset(name, data=a)

        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != n:
                raise ValueError(
                    f"Fiber component {comp_name!r} has {arr.shape[1]} fibers; "
                    f"expected {n}."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — layers
    # ------------------------------------------------------------------

    def write_layers_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        element_index: ndarray,
        gp_index: ndarray,
        layer_index: ndarray,
        sub_gp_index: ndarray,
        thickness: ndarray,
        local_axes_quaternion: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_LAYERS, group_id,
        )
        eidx = np.asarray(element_index, dtype=np.int64)
        n = eidx.size
        for name, arr, dtype in [
            (_native.DSET_ELEMENT_INDEX, eidx, np.int64),
            (_native.DSET_GP_INDEX, gp_index, np.int64),
            (_native.DSET_LAYER_INDEX, layer_index, np.int64),
            (_native.DSET_SUB_GP_INDEX, sub_gp_index, np.int64),
            (_native.DSET_THICKNESS, thickness, np.float64),
        ]:
            a = np.asarray(arr, dtype=dtype)
            if a.size != n:
                raise ValueError(
                    f"Layer index dataset {name!r} has size {a.size}; "
                    f"expected {n}."
                )
            grp.create_dataset(name, data=a)

        quat = np.asarray(local_axes_quaternion, dtype=np.float64)
        if quat.shape != (n, 4):
            raise ValueError(
                f"local_axes_quaternion must have shape ({n}, 4); "
                f"got {quat.shape}."
            )
        grp.create_dataset(_native.DSET_LOCAL_AXES_QUATERNION, data=quat)

        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != n:
                raise ValueError(
                    f"Layer component {comp_name!r} has {arr.shape[1]} entries; "
                    f"expected {n}."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — line stations
    # ------------------------------------------------------------------

    def write_line_stations_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        class_tag: int,
        int_rule: int = 0,
        element_index: ndarray,
        station_natural_coord: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_LINE_STATIONS, group_id,
        )
        grp.attrs[_native.ATTR_CLASS_TAG] = int(class_tag)
        grp.attrs[_native.ATTR_INT_RULE] = int(int_rule)

        eidx = np.asarray(element_index, dtype=np.int64)
        snc = np.asarray(station_natural_coord, dtype=np.float64)
        grp.create_dataset(_native.DSET_ELEMENT_INDEX, data=eidx)
        grp.create_dataset(_native.DSET_STATION_NATURAL_COORD, data=snc)

        n_stations = snc.size
        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != eidx.size or arr.shape[2] != n_stations:
                raise ValueError(
                    f"Line-station component {comp_name!r} has shape "
                    f"{arr.shape}; expected (T, {eidx.size}, {n_stations})."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — nodal forces (per-element-node)
    # ------------------------------------------------------------------

    def write_nodal_forces_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        class_tag: int,
        frame: str,                        # "global" or "local"
        element_index: ndarray,
        components: dict[str, ndarray],    # (T, E_g, npe_g)
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_NODAL_FORCES, group_id,
        )
        grp.attrs[_native.ATTR_CLASS_TAG] = int(class_tag)
        grp.attrs[_native.ATTR_FRAME] = frame

        eidx = np.asarray(element_index, dtype=np.int64)
        grp.create_dataset(_native.DSET_ELEMENT_INDEX, data=eidx)

        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != eidx.size:
                raise ValueError(
                    f"Nodal-force component {comp_name!r} has {arr.shape[1]} "
                    f"elements; expected {eidx.size}."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Element ID writing (per partition)
    # ------------------------------------------------------------------

    def write_element_ids(
        self,
        stage_id: str,
        partition_id: str,
        ids: ndarray,
    ) -> None:
        """Write the partition's flat element ID list at ``elements/_ids``."""
        grp = self._require_partition(stage_id, partition_id).require_group(
            _native.GROUP_ELEMENTS,
        )
        _write_or_validate_ids(
            grp, _native.DSET_IDS, np.asarray(ids, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> "h5py.File":
        if self._h5 is None:
            raise RuntimeError(
                f"NativeWriter for {self._path} is not open. "
                f"Call open() or use as context manager."
            )
        return self._h5

    def _require_partition(self, stage_id: str, partition_id: str) -> Any:
        h5 = self._require_open()
        partitions_grp = h5.require_group(
            _native.partitions_path(stage_id)[1:],
        )
        return partitions_grp.require_group(partition_id)

    def _require_element_subgroup(
        self,
        stage_id: str,
        partition_id: str,
        category: str,
        group_id: str,
    ) -> Any:
        elem_grp = self._require_partition(stage_id, partition_id).require_group(
            _native.GROUP_ELEMENTS,
        )
        cat_grp = elem_grp.require_group(category)
        if group_id in cat_grp:
            raise RuntimeError(
                f"Group {group_id!r} already exists under "
                f"{stage_id}/{partition_id}/elements/{category}/."
            )
        return cat_grp.create_group(group_id)

    def _validate_time_axis(self, stage_id: str, arr: ndarray) -> None:
        h5 = self._require_open()
        time = h5[_native.stage_time_path(stage_id)]
        n_steps = time.shape[0]
        if arr.ndim < 1 or arr.shape[0] != n_steps:
            raise ValueError(
                f"Component array has shape {arr.shape}; expected leading "
                f"dim {n_steps} (matching stage time vector)."
            )


# =====================================================================
# Module helpers
# =====================================================================

def _write_or_validate_ids(group: Any, key: str, ids: ndarray) -> None:
    """Write ``ids`` to ``group[key]`` if absent; otherwise validate equal."""
    if key in group:
        existing = group[key][...]
        if not np.array_equal(np.asarray(existing, dtype=np.int64), ids):
            raise ValueError(
                f"{group.name}/{key} mismatch: existing {existing} vs new {ids}."
            )
        return
    group.create_dataset(key, data=ids)


def _apegmsh_version() -> str:
    try:
        from apeGmsh import __version__ as v
        return str(v)
    except Exception:
        return ""
