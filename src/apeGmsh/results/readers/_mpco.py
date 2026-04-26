"""MPCOReader — reads STKO ``.mpco`` files through the unified protocol.

Phase 3 scope: nodal results (DISPLACEMENT, ROTATION, VELOCITY,
ACCELERATION, REACTION_FORCE, …) translated to canonical apeGmsh
names. Element-level reads (gauss, fibers, layers, line stations)
are stubbed — the META unflattening is non-trivial and lands in a
later phase. Empty slabs are returned for those levels with a
clear "not supported in Phase 3" log message via the slab itself
(values is empty (T, 0)).
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
from . import _mpco_translation as _mtr
from ._protocol import ResultLevel, StageInfo, TimeSlice

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData


# Stage groups in MPCO are ``MODEL_STAGE[<stamp>]``
_STAGE_PREFIX = "MODEL_STAGE["


class MPCOReader:
    """Reader for STKO ``.mpco`` HDF5 files."""

    def __init__(self, path: str | Path) -> None:
        import h5py

        self._path = Path(path)
        self._h5: "h5py.File" = h5py.File(self._path, "r")
        self._stage_cache: Optional[list[StageInfo]] = None
        # MPCO: each apeGmsh stage_id maps to one MPCO group name
        self._stage_to_mpco: dict[str, str] = {}
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "MPCOReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Stage discovery
    # ------------------------------------------------------------------

    def stages(self) -> list[StageInfo]:
        if self._stage_cache is not None:
            return list(self._stage_cache)

        out: list[StageInfo] = []
        # Map MPCO group names to apeGmsh stage_ids in iteration order.
        for name in sorted(self._h5.keys()):
            if not name.startswith(_STAGE_PREFIX):
                continue
            grp = self._h5[name]
            time = self._build_time_vector_for_mpco_stage(grp)
            stage_id = f"stage_{len(out)}"
            self._stage_to_mpco[stage_id] = name
            out.append(StageInfo(
                id=stage_id,
                name=name,
                kind="transient",       # MPCO stages aren't tagged; assume transient
                n_steps=int(time.size),
            ))
        self._stage_cache = out
        return list(out)

    def _build_time_vector_for_mpco_stage(
        self, stage_grp: "h5py.Group",
    ) -> ndarray:
        """Aggregate TIME attrs from all STEP_<k> datasets in the stage.

        Picks the first ``RESULTS/ON_NODES/<X>`` group with STEP data
        and reads the TIME attribute on each step. Different result
        groups are assumed to share the same step cadence (which is
        STKO's invariant).
        """
        on_nodes = stage_grp.get("RESULTS/ON_NODES")
        if on_nodes is None:
            return np.array([], dtype=np.float64)
        for result_name in on_nodes:
            data_grp = on_nodes[result_name].get("DATA")
            if data_grp is None:
                continue
            step_keys = sorted(
                [k for k in data_grp.keys() if k.startswith("STEP_")],
                key=lambda s: int(s.split("_", 1)[1]),
            )
            if not step_keys:
                continue
            return np.array(
                [float(_attr_scalar(data_grp[k].attrs.get("TIME", 0.0)))
                 for k in step_keys],
                dtype=np.float64,
            )
        return np.array([], dtype=np.float64)

    def time_vector(self, stage_id: str) -> ndarray:
        self._ensure_stages()
        mpco_name = self._stage_to_mpco[stage_id]
        return self._build_time_vector_for_mpco_stage(self._h5[mpco_name])

    def partitions(self, stage_id: str) -> list[str]:
        # MPCO file = single partition. Multi-partition runs use multiple
        # ``.part-<n>.mpco`` files; merging across them lands in a later
        # phase (open MPCOReader per file then stitch at the composite
        # layer).
        return ["partition_0"]

    # ------------------------------------------------------------------
    # FEM access
    # ------------------------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        if not isinstance(self._fem_cache, _Sentinel):
            return self._fem_cache
        self._ensure_stages()
        if not self._stage_cache:
            self._fem_cache = None
            return None
        # Use the LAST stage's MODEL — most up-to-date geometry.
        mpco_name = self._stage_to_mpco[self._stage_cache[-1].id]
        model_grp = self._h5[mpco_name].get("MODEL")
        if model_grp is None:
            self._fem_cache = None
            return None
        from ...mesh.FEMData import FEMData
        self._fem_cache = FEMData.from_mpco_model(model_grp)
        return self._fem_cache

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        if level.value != "nodes":
            # Element-level not supported in Phase 3.
            return []
        self._ensure_stages()
        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return []
        on_nodes = self._h5[mpco_name].get("RESULTS/ON_NODES")
        if on_nodes is None:
            return []

        out: set[str] = set()
        for result_name in on_nodes:
            grp = on_nodes[result_name]
            comps = _parse_components_attr(grp.attrs.get("COMPONENTS"))
            for comp_label in comps:
                canonical = _mtr.canonical_node_component(
                    result_name, comp_label,
                )
                if canonical is not None:
                    out.add(canonical)
        return sorted(out)

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
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco[stage_id]
        on_nodes = self._h5[mpco_name].get("RESULTS/ON_NODES")
        if on_nodes is None:
            return _empty_node_slab(component, time, t_idx)

        # Find the (mpco_result_name, column_index) for the requested canonical.
        loc = _locate_canonical_in_mpco(on_nodes, component)
        if loc is None:
            return _empty_node_slab(component, time, t_idx)
        mpco_result_name, col_idx = loc

        result_grp = on_nodes[mpco_result_name]
        data_grp = result_grp.get("DATA")
        if data_grp is None:
            return _empty_node_slab(component, time, t_idx)

        # Read the result's ID array (subset of model nodes for this result).
        all_ids = (
            np.asarray(result_grp["ID"][...], dtype=np.int64).flatten()
            if "ID" in result_grp
            else self._fallback_node_ids(mpco_name)
        )

        # Apply node_ids filter
        if node_ids is None:
            sel = np.arange(all_ids.size)
            sel_ids = all_ids
        else:
            requested = np.asarray(node_ids, dtype=np.int64)
            mask = np.isin(all_ids, requested)
            if not mask.any():
                return NodeSlab(
                    component=component,
                    values=np.zeros((t_idx.size, 0), dtype=np.float64),
                    node_ids=np.array([], dtype=np.int64),
                    time=time[t_idx],
                )
            sel = np.where(mask)[0]
            sel_ids = all_ids[sel]

        # Read selected steps and selected node rows.
        step_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("STEP_")],
            key=lambda s: int(s.split("_", 1)[1]),
        )
        values = np.empty((t_idx.size, sel_ids.size), dtype=np.float64)
        for i, k in enumerate(t_idx):
            step_ds = data_grp[step_keys[int(k)]]
            row = step_ds[:, col_idx]                 # (nNodes,)
            values[i, :] = row[sel]

        return NodeSlab(
            component=component, values=values,
            node_ids=sel_ids, time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Element-level reads — Phase 3 stubs
    # ------------------------------------------------------------------

    def read_elements(self, stage_id, component, *, element_ids=None,
                       time_slice=None) -> ElementSlab:
        return _empty_element_slab(component, self.time_vector(stage_id),
                                    time_slice)

    def read_line_stations(self, stage_id, component, *, element_ids=None,
                            time_slice=None) -> LineStationSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        return LineStationSlab(
            component=component,
            values=np.zeros((t_idx.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            station_natural_coord=np.array([], dtype=np.float64),
            time=time[t_idx],
        )

    def read_gauss(self, stage_id, component, *, element_ids=None,
                    time_slice=None) -> GaussSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        return GaussSlab(
            component=component,
            values=np.zeros((t_idx.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            natural_coords=np.zeros((0, 3), dtype=np.float64),
            local_axes_quaternion=None,
            time=time[t_idx],
        )

    def read_fibers(self, stage_id, component, *, element_ids=None,
                     gp_indices=None, time_slice=None) -> FiberSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        return FiberSlab(
            component=component,
            values=np.zeros((t_idx.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            gp_index=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.float64),
            z=np.array([], dtype=np.float64),
            area=np.array([], dtype=np.float64),
            material_tag=np.array([], dtype=np.int64),
            time=time[t_idx],
        )

    def read_layers(self, stage_id, component, *, element_ids=None,
                     gp_indices=None, layer_indices=None,
                     time_slice=None) -> LayerSlab:
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)
        return LayerSlab(
            component=component,
            values=np.zeros((t_idx.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            gp_index=np.array([], dtype=np.int64),
            layer_index=np.array([], dtype=np.int64),
            sub_gp_index=np.array([], dtype=np.int64),
            thickness=np.array([], dtype=np.float64),
            local_axes_quaternion=np.zeros((0, 4), dtype=np.float64),
            time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_stages(self) -> None:
        if self._stage_cache is None:
            self.stages()

    def _fallback_node_ids(self, mpco_stage_name: str) -> ndarray:
        """When a result group has no ID, fall back to MODEL/NODES/ID."""
        return np.asarray(
            self._h5[mpco_stage_name]["MODEL/NODES/ID"][...],
            dtype=np.int64,
        ).flatten()


# =====================================================================
# Module helpers
# =====================================================================

class _Sentinel:
    pass


_SENTINEL = _Sentinel()


def _attr_scalar(value):
    """Coerce an h5py attribute value (scalar or 1-element array) to a scalar."""
    if hasattr(value, "size") and getattr(value, "size", 0) == 1:
        return value.item() if hasattr(value, "item") else value[0]
    if hasattr(value, "__len__") and len(value) == 1:
        return value[0]
    return value


def _parse_components_attr(value) -> list[str]:
    """Parse the COMPONENTS attribute into a list of component labels.

    The attribute is typically a 1-element array of fixed-length bytes,
    e.g. ``array([b'Ux,Uy,Uz'])``. Some files may store it as a plain
    bytes / str.
    """
    if value is None:
        return []
    raw = _attr_scalar(value)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _locate_canonical_in_mpco(
    on_nodes_grp, canonical_component: str,
) -> tuple[str, int] | None:
    """Find ``(mpco_result_name, column_index)`` for a canonical component.

    Iterates the MPCO result groups under ``ON_NODES``, parsing each
    group's COMPONENTS attribute and translating each column to its
    canonical name. Returns the first match.
    """
    for mpco_result_name in on_nodes_grp:
        grp = on_nodes_grp[mpco_result_name]
        comps = _parse_components_attr(grp.attrs.get("COMPONENTS"))
        for col_idx, comp_label in enumerate(comps):
            canonical = _mtr.canonical_node_component(
                mpco_result_name, comp_label,
            )
            if canonical == canonical_component:
                return (mpco_result_name, col_idx)
    return None


def _empty_node_slab(component: str, time: ndarray, t_idx) -> NodeSlab:
    if t_idx is None:
        t_idx = np.arange(time.size, dtype=np.int64)
    return NodeSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        node_ids=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_element_slab(component: str, time: ndarray, time_slice) -> ElementSlab:
    t_idx = resolve_time_slice(time_slice, time)
    return ElementSlab(
        component=component,
        values=np.zeros((t_idx.size, 0, 0), dtype=np.float64),
        element_ids=np.array([], dtype=np.int64),
        time=time[t_idx],
    )
