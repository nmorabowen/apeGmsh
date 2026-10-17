"""Unit-aware scaling and reader wrapper for :class:`~apeGmsh.results.Results` (Slice U2).

When a model or results file declares its consistent unit system (or when the
user supplies ``source=...`` to :meth:`Results.to_units`), this module wraps
the underlying :class:`ResultsReader` and scales every dimensional result
slab, energy balance, node envelope, and bound ``FEMData`` coordinate array
into the requested target consistent unit system.

Mode-shape stages (``stage.kind == "mode"``), time vectors, natural
coordinates, and dimensionless components (strains, rotations, damage,
utilization ratios, energy-balance ``ERR``) stay unscaled.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import ndarray

from ..units import (
    UnitSystemId,
    UnitSystemInput,
    conversion_factor,
    resolve_field_units,
    resolve_unit_system,
)
from ._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
    SpringSlab,
)
from .readers._protocol import ResultLevel, ResultsReader, StageInfo, TimeSlice

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from .Results import Results

__all__ = [
    "_UnitScaledResultsReader",
    "build_scaled_results",
    "compute_results_unit_factor",
    "compute_results_unit_symbol",
    "resolve_results_source_units",
    "resolve_results_units",
    "scale_fem_units",
]

_LEVEL_MAP: dict[str, str] = {
    "nodes": "node",
    "node": "node",
    "nodal": "node",
    "elements": "element",
    "element": "element",
    "line_stations": "line",
    "line_station": "line",
    "line": "line",
    "gauss": "gauss",
    "fibers": "fiber",
    "fiber": "fiber",
    "layers": "layer",
    "layer": "layer",
    "springs": "spring",
    "spring": "spring",
    "energy": "energy",
}


def _normalize_level(level: str | ResultLevel) -> str:
    if isinstance(level, ResultLevel):
        raw = level.value
    else:
        raw = str(level).strip().lower()
    if raw not in _LEVEL_MAP:
        raise ValueError(
            f"Unknown result level {level!r}; expected one of "
            f"{sorted(_LEVEL_MAP)}."
        )
    return _LEVEL_MAP[raw]


def _scale_named_group_set(group_set: Any, length_factor: float) -> Any:
    if group_set is None or not hasattr(group_set, "_groups"):
        return group_set
    cloned = copy.copy(group_set)
    new_groups: dict[Any, dict[str, Any]] = {}
    for key, info in group_set._groups.items():
        d = dict(info)
        if "node_coords" in d and d["node_coords"] is not None:
            d["node_coords"] = np.asarray(d["node_coords"], dtype=np.float64) * length_factor
        new_groups[key] = d
    cloned._groups = new_groups
    cloned._merged_cache = {}
    return cloned


def scale_fem_units(
    fem: "FEMData | None",
    source_units: UnitSystemId,
    target_units: UnitSystemId,
) -> "FEMData | None":
    """Return a shallow-cloned :class:`FEMData` whose node coordinates are
    scaled from ``source_units`` to ``target_units`` and whose ``.units`` is
    ``target_units``."""
    if fem is None:
        return None
    length_factor = conversion_factor(source_units, target_units, "length")
    if length_factor == 1.0:
        return fem.with_units(target_units)

    nodes = copy.copy(fem.nodes)
    nodes._coords = np.asarray(fem.nodes._coords, dtype=np.float64) * length_factor
    nodes.physical = _scale_named_group_set(fem.nodes.physical, length_factor)
    nodes.labels = _scale_named_group_set(fem.nodes.labels, length_factor)
    elements = copy.copy(fem.elements)
    elements.physical = _scale_named_group_set(fem.elements.physical, length_factor)
    elements.labels = _scale_named_group_set(fem.elements.labels, length_factor)
    scaled = fem._replaced(nodes=nodes, elements=elements)
    scaled.units = target_units
    return scaled


class _UnitScaledResultsReader:
    """Decorator around a :class:`ResultsReader` that scales dimensional
    values from ``source_units`` to ``target_units``."""

    def __init__(
        self,
        base_reader: ResultsReader,
        *,
        source_units: UnitSystemId,
        target_units: UnitSystemId,
    ) -> None:
        if isinstance(base_reader, _UnitScaledResultsReader):
            base_reader = base_reader.base_reader
        self._base: ResultsReader = base_reader
        self._source_units: UnitSystemId = source_units
        self._target_units: UnitSystemId = target_units
        self._mode_stage_ids: set[str] | None = None
        self._scaled_fem: "FEMData | None" = None
        self._fem_cached: bool = False

    @property
    def base_reader(self) -> ResultsReader:
        return self._base

    @property
    def units(self) -> UnitSystemId:
        return self._target_units

    @property
    def source_units(self) -> UnitSystemId:
        return self._source_units

    def _is_mode_stage(self, stage_id: str) -> bool:
        if self._mode_stage_ids is None:
            self._mode_stage_ids = {
                s.id for s in self._base.stages() if s.kind == "mode"
            }
        return stage_id in self._mode_stage_ids

    def _factor(self, component: str, level: str, stage_id: str) -> float:
        stage_kind = "mode" if self._is_mode_stage(stage_id) else None
        return resolve_field_units(
            component,
            level,
            self._source_units,
            self._target_units,
            stage_kind=stage_kind,
        ).factor

    # -- Stage & metadata delegation -----------------------------------

    def stages(self) -> list[StageInfo]:
        return self._base.stages()

    def time_vector(self, stage_id: str) -> ndarray:
        return self._base.time_vector(stage_id)

    def partitions(self, stage_id: str) -> list[str]:
        return self._base.partitions(stage_id)

    def fem(self) -> "FEMData | None":
        if not self._fem_cached:
            self._scaled_fem = scale_fem_units(
                self._base.fem(), self._source_units, self._target_units,
            )
            self._fem_cached = True
        return self._scaled_fem

    def opensees_model(self) -> Any:
        return self._base.opensees_model()

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        return self._base.available_components(stage_id, level)

    # -- Scaled slab reads ---------------------------------------------

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        slab = self._base.read_nodes(
            stage_id, component, node_ids=node_ids, time_slice=time_slice,
        )
        factor = self._factor(component, "node", stage_id)
        if factor == 1.0:
            return slab
        return NodeSlab(
            component=slab.component,
            values=slab.values * factor,
            node_ids=slab.node_ids,
            time=slab.time,
        )

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        slab = self._base.read_elements(
            stage_id, component, element_ids=element_ids, time_slice=time_slice,
        )
        factor = self._factor(component, "element", stage_id)
        if factor == 1.0:
            return slab
        return ElementSlab(
            component=slab.component,
            values=slab.values * factor,
            element_ids=slab.element_ids,
            time=slab.time,
        )

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        slab = self._base.read_line_stations(
            stage_id, component, element_ids=element_ids, time_slice=time_slice,
        )
        factor = self._factor(component, "line", stage_id)
        if factor == 1.0:
            return slab
        return LineStationSlab(
            component=slab.component,
            values=slab.values * factor,
            element_index=slab.element_index,
            station_natural_coord=slab.station_natural_coord,
            time=slab.time,
            local_axes_quaternion=slab.local_axes_quaternion,
        )

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        slab = self._base.read_gauss(
            stage_id, component, element_ids=element_ids, time_slice=time_slice,
        )
        factor = self._factor(component, "gauss", stage_id)
        if factor == 1.0:
            return slab
        return GaussSlab(
            component=slab.component,
            values=slab.values * factor,
            element_index=slab.element_index,
            natural_coords=slab.natural_coords,
            local_axes_quaternion=slab.local_axes_quaternion,
            time=slab.time,
        )

    def read_fibers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: ndarray | None = None,
        gp_indices: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> FiberSlab:
        slab = self._base.read_fibers(
            stage_id,
            component,
            element_ids=element_ids,
            gp_indices=gp_indices,
            time_slice=time_slice,
        )
        factor = self._factor(component, "fiber", stage_id)
        len_factor = conversion_factor(
            self._source_units, self._target_units, "length",
        )
        if factor == 1.0 and len_factor == 1.0:
            return slab
        area_factor = len_factor * len_factor
        return FiberSlab(
            component=slab.component,
            values=slab.values * factor if factor != 1.0 else slab.values,
            element_index=slab.element_index,
            gp_index=slab.gp_index,
            y=slab.y * len_factor if len_factor != 1.0 else slab.y,
            z=slab.z * len_factor if len_factor != 1.0 else slab.z,
            area=slab.area * area_factor if area_factor != 1.0 else slab.area,
            material_tag=slab.material_tag,
            time=slab.time,
            station_natural_coord=slab.station_natural_coord,
        )

    def read_layers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: ndarray | None = None,
        gp_indices: ndarray | None = None,
        layer_indices: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> LayerSlab:
        slab = self._base.read_layers(
            stage_id,
            component,
            element_ids=element_ids,
            gp_indices=gp_indices,
            layer_indices=layer_indices,
            time_slice=time_slice,
        )
        factor = self._factor(component, "layer", stage_id)
        len_factor = conversion_factor(
            self._source_units, self._target_units, "length",
        )
        if factor == 1.0 and len_factor == 1.0:
            return slab
        return LayerSlab(
            component=slab.component,
            values=slab.values * factor if factor != 1.0 else slab.values,
            element_index=slab.element_index,
            gp_index=slab.gp_index,
            layer_index=slab.layer_index,
            sub_gp_index=slab.sub_gp_index,
            thickness=(
                slab.thickness * len_factor
                if len_factor != 1.0
                else slab.thickness
            ),
            local_axes_quaternion=slab.local_axes_quaternion,
            time=slab.time,
        )

    def read_springs(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: ndarray | None = None,
        time_slice: TimeSlice = None,
    ) -> SpringSlab:
        slab = self._base.read_springs(
            stage_id, component, element_ids=element_ids, time_slice=time_slice,
        )
        factor = self._factor(component, "spring", stage_id)
        if factor == 1.0:
            return slab
        return SpringSlab(
            component=slab.component,
            values=slab.values * factor,
            element_index=slab.element_index,
            time=slab.time,
        )

    # -- Ladruno reader extensions (energy & node_envelope) ------------

    def _read_energy(
        self,
        stage_id: str,
        *,
        region: int | None = None,
        time_slice: TimeSlice = None,
    ) -> tuple[list[str], ndarray, ndarray]:
        fn = getattr(self._base, "read_energy")
        cols, values, time = fn(stage_id, region=region, time_slice=time_slice)
        scaled = np.array(values, dtype=np.float64, copy=True)
        for j, col in enumerate(cols):
            factor = self._factor(col, "energy", stage_id)
            if factor != 1.0:
                scaled[:, j] *= factor
        return cols, scaled, time

    def _read_node_envelope(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: ndarray | None = None,
    ) -> Any:
        fn = getattr(self._base, "read_node_envelope")
        env = fn(stage_id, component, node_ids=node_ids)
        factor = self._factor(component, "node", stage_id)
        if factor == 1.0:
            return env
        return type(env)(
            node_ids=env.node_ids,
            min=env.min * factor,
            max=env.max * factor,
            absmax=env.absmax * factor,
            arg_step=env.arg_step,
        )

    def close(self) -> None:
        if hasattr(self._base, "close"):
            self._base.close()

    def __getattr__(self, name: str) -> Any:
        if name == "read_energy":
            if getattr(self._base, "read_energy", None) is None:
                raise AttributeError(
                    f"{type(self._base).__name__!r} object has no attribute 'read_energy'"
                )
            return self._read_energy
        if name == "read_node_envelope":
            if getattr(self._base, "read_node_envelope", None) is None:
                raise AttributeError(
                    f"{type(self._base).__name__!r} object has no attribute 'read_node_envelope'"
                )
            return self._read_node_envelope
        return getattr(self._base, name)


def resolve_results_units(results: "Results") -> UnitSystemId | None:
    """Resolve the active unit system of ``results`` (or ``None`` if undeclared).

    Raises :class:`ValueError` when the reader, bound ``OpenSeesModel``, and
    bound ``FEMData`` declare conflicting non-``None`` unit systems.
    """
    override = getattr(results, "_units_override", None)
    if override is not None:
        return resolve_unit_system(override)

    sources: list[tuple[str, UnitSystemId]] = []
    reader_units = getattr(results._reader, "units", None)
    if reader_units is not None:
        r_id = resolve_unit_system(reader_units)
        if r_id is not None:
            sources.append(("reader", r_id))

    model = getattr(results, "_model", None)
    if model is not None and getattr(model, "units", None) is not None:
        m_id = resolve_unit_system(model.units)
        if m_id is not None:
            sources.append(("model", m_id))

    fem = getattr(results, "_fem", None)
    if fem is not None and getattr(fem, "units", None) is not None:
        f_id = resolve_unit_system(fem.units)
        if f_id is not None:
            sources.append(("fem", f_id))

    if not sources:
        return None

    first_name, first_unit = sources[0]
    for other_name, other_unit in sources[1:]:
        if other_unit != first_unit:
            raise ValueError(
                f"Conflicting unit declarations on Results: "
                f"{first_name} declares {first_unit!r}, but "
                f"{other_name} declares {other_unit!r}."
            )
    return first_unit


def resolve_results_source_units(results: "Results") -> UnitSystemId | None:
    """Resolve the native (as-recorded) unit system of ``results``."""
    src_override = getattr(results, "_source_units_override", None)
    if src_override is not None:
        return resolve_unit_system(src_override)
    if isinstance(results._reader, _UnitScaledResultsReader):
        return results._reader.source_units
    return resolve_results_units(results)


def compute_results_unit_factor(
    results: "Results",
    component: str,
    *,
    level: str | ResultLevel = "nodes",
    target: UnitSystemInput | None = None,
    source: UnitSystemInput | None = None,
) -> float:
    """Return the multiplicative conversion factor for ``component`` at ``level``."""
    norm_level = _normalize_level(level)
    stage_kind: str | None = None
    if results._stage_id is not None:
        for s in results._all_stages():
            if s.id == results._stage_id:
                stage_kind = s.kind
                break
    if target is None:
        return 1.0
    dst_id = resolve_unit_system(target)
    if dst_id is None:
        raise ValueError("target unit system cannot be None.")
    src_id = (
        resolve_unit_system(source)
        if source is not None
        else resolve_results_units(results)
    )
    if src_id is None:
        raise ValueError(
            f"Cannot compute unit conversion factor for {component!r} to "
            f"{dst_id!r}: no unit system was declared on this Results object. "
            "Pass source=<unit_system> explicitly (e.g. source='kN_m_s')."
        )
    return resolve_field_units(
        component, norm_level, src_id, dst_id, stage_kind=stage_kind,
    ).factor


def compute_results_unit_symbol(
    results: "Results",
    component: str,
    *,
    level: str | ResultLevel = "nodes",
) -> str:
    """Return the display unit string for ``component`` in ``results.units``
    (or ``""`` when ``results.units is None`` or the field is dimensionless)."""
    active = resolve_results_units(results)
    if active is None:
        return ""
    norm_level = _normalize_level(level)
    stage_kind: str | None = None
    if results._stage_id is not None:
        for s in results._all_stages():
            if s.id == results._stage_id:
                stage_kind = s.kind
                break
    unit = resolve_field_units(
        component, norm_level, active, active, stage_kind=stage_kind,
    ).unit
    return unit or ""


def build_scaled_results(
    results: "Results",
    target: UnitSystemInput,
    *,
    source: UnitSystemInput | None = None,
) -> "Results":
    """Return a new :class:`Results` view whose slabs and bound ``FEMData``
    are expressed in ``target`` consistent units."""
    if target is None:
        raise ValueError(
            "Results.to_units(target) requires a non-None target unit system."
        )
    target_id = resolve_unit_system(target)
    assert target_id is not None

    declared_src = resolve_results_source_units(results)
    if source is not None:
        src_id = resolve_unit_system(source)
        if src_id is None:
            raise ValueError("source unit system cannot be None when supplied.")
        if declared_src is not None and src_id != declared_src:
            raise ValueError(
                f"Results.to_units: explicit source={src_id!r} conflicts with "
                f"declared unit system {declared_src!r}."
            )
    else:
        if declared_src is None:
            raise ValueError(
                "Results has no declared unit system (/meta/@units or "
                "INFO/PROVENANCE@units is absent). Pass source=<unit_system> "
                "explicitly to Results.to_units(target, source=...) to "
                "translate units."
            )
        src_id = declared_src

    base_reader = (
        results._reader.base_reader
        if isinstance(results._reader, _UnitScaledResultsReader)
        else results._reader
    )
    base_fem = getattr(results, "_base_fem", None)
    if base_fem is None:
        base_fem = results._fem

    scaled_reader = _UnitScaledResultsReader(
        base_reader, source_units=src_id, target_units=target_id,
    )
    embedded_base_fem = None
    try:
        embedded_base_fem = base_reader.fem()
    except Exception:
        embedded_base_fem = None
    if base_fem is not None and base_fem is embedded_base_fem:
        scaled_fem = scaled_reader.fem()
    else:
        scaled_fem = scale_fem_units(base_fem, src_id, target_id)

    derived = results._derive(fem=scaled_fem)
    derived._reader = scaled_reader
    derived._units_override = target_id
    derived._source_units_override = src_id
    derived._base_fem = base_fem
    return derived
