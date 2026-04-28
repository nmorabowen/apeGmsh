"""MPCOReader — reads STKO ``.mpco`` files through the unified protocol.

Phase 3 covered nodal results (DISPLACEMENT, ROTATION, VELOCITY,
ACCELERATION, REACTION_FORCE, …) translated to canonical apeGmsh
names. Phase 11a wires Gauss-level reads through the shared response
catalog in :mod:`apeGmsh.solvers._element_response`; fibers, layers,
line stations, and per-element-node forces remain stubbed (empty
slabs) until their catalog entries land.
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
from . import _mpco_element_io as _melem
from . import _mpco_fiber_io as _mfiber
from . import _mpco_layer_io as _mlayer
from . import _mpco_line_io as _mline
from . import _mpco_material_io as _mmat
from . import _mpco_nodal_io as _mnodal
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
        self._ensure_stages()
        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return []

        if level.value == "nodes":
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

        if level.value == "gauss":
            return self._gauss_available_components(mpco_name)

        if level.value == "line_stations":
            return self._line_stations_available_components(mpco_name)

        if level.value == "elements":
            return self._elements_available_components(mpco_name)

        if level.value == "fibers":
            return self._fibers_available_components(mpco_name)

        if level.value == "layers":
            return self._layers_available_components(mpco_name)

        return []

    def _gauss_available_components(self, mpco_name: str) -> list[str]:
        on_elements = self._h5[mpco_name].get("RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        # Tensor canonicals (catalog-driven, fixed shape): any axis
        # suffix discovers the same buckets, so probe with ``_xx``.
        for prefix in ("stress", "strain"):
            _, buckets = _melem.discover_gauss_buckets(
                on_elements, canonical_component=f"{prefix}_xx",
            )
            for b in buckets:
                out.update(b.layout.component_layout)
        # Material-state canonicals (META-driven, variable shape).
        # Probe each parent token; surface every per-segment canonical
        # the bucket's META declares.
        for parent in ("damage", "equivalent_plastic_strain"):
            _, mat_buckets = _mmat.discover_material_state_buckets(
                on_elements, canonical_component=parent,
            )
            for mb in mat_buckets:
                bucket_grp = on_elements[mb.mpco_group_name][mb.bracket_key]
                try:
                    canonicals = _mmat.material_state_canonicals_in_bucket(
                        bucket_grp, mb,
                    )
                except ValueError:
                    # Malformed META — skip this bucket for discovery
                    # rather than crash the whole listing.
                    continue
                out.update(canonicals)
        return sorted(out)

    def _elements_available_components(
        self, mpco_name: str,
    ) -> list[str]:
        on_elements = self._h5[mpco_name].get("RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        # Probe each frame via a representative canonical name; the
        # discover helper handles the canonical→MPCO group-name
        # routing and bucket filtering against NODAL_FORCE_CATALOG.
        for probe in (
            "nodal_resisting_force_x",       # globalForce
            "nodal_resisting_force_local_x", # localForce
        ):
            _, buckets = _mnodal.discover_nodal_force_buckets(
                on_elements, canonical_component=probe,
            )
            for b in buckets:
                out.update(b.layout.component_layout)
        return sorted(out)

    def _fibers_available_components(self, mpco_name: str) -> list[str]:
        on_elements = self._h5[mpco_name].get("RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        for canonical in ("fiber_stress", "fiber_strain"):
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elements, canonical_component=canonical,
            )
            if buckets:
                out.add(canonical)
        return sorted(out)

    def _layers_available_components(self, mpco_name: str) -> list[str]:
        stage_grp = self._h5[mpco_name]
        on_elements = stage_grp.get("RESULTS/ON_ELEMENTS")
        section_assignments = stage_grp.get("MODEL/SECTION_ASSIGNMENTS")
        if on_elements is None or section_assignments is None:
            return []
        out: set[str] = set()
        # For each parent token, walk every catalogued bucket and
        # surface its META-resolved per-cell canonicals (single-
        # component buckets keep the bare name; multi-component
        # buckets expose ``fiber_stress_<i>`` per index).
        for parent in ("fiber_stress", "fiber_strain"):
            _, buckets = _mlayer.discover_layer_buckets(
                on_elements, canonical_component=parent,
            )
            for b in buckets:
                bucket_grp = on_elements[b.mpco_group_name][b.bracket_key]
                try:
                    layout = _mlayer.resolve_layer_bucket_layout(
                        section_assignments, bucket_grp, b,
                    )
                except ValueError:
                    continue
                out.update(layout.component_layout)
        return sorted(out)

    def _line_stations_available_components(
        self, mpco_name: str,
    ) -> list[str]:
        stage_grp = self._h5[mpco_name]
        on_elements = stage_grp.get("RESULTS/ON_ELEMENTS")
        model_elements = stage_grp.get("MODEL/ELEMENTS")
        if on_elements is None or model_elements is None:
            return []
        # Discover line-stations buckets via any line-station canonical
        # (the routing only depends on the topology, not the specific
        # component name); ``axial_force`` is always present in v1.
        _, buckets = _mline.discover_line_station_buckets(
            on_elements, canonical_component="axial_force",
        )
        out: set[str] = set()
        for bucket in buckets:
            try:
                token_grp = on_elements["section.force"][bucket.bracket_key]
                layout, _ = _mline.resolve_bucket_layout(
                    model_elements, token_grp, bucket,
                )
            except (ValueError, KeyError):
                # Unsupported section codes / missing GP_X — silent skip
                # at availability discovery (still raises on .get()).
                continue
            out.update(layout.component_layout)
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

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_element_slab(component, time, time_slice)
        on_elements = self._h5[mpco_name].get("RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return _empty_element_slab(component, time, time_slice)

        token, buckets = _mnodal.discover_nodal_force_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_element_slab(component, time, time_slice)
        token_grp = on_elements[token]

        # Read per-bucket slabs and stitch on the element axis.
        values_parts: list[ndarray] = []
        element_id_parts: list[ndarray] = []

        for bucket in buckets:
            bucket_grp = token_grp[bucket.bracket_key]
            result = _mnodal.read_nodal_bucket_slab(
                bucket_grp, bucket, component,
                t_idx=t_idx, element_ids=element_ids,
            )
            if result is None:
                continue
            values, eids = result
            values_parts.append(values)
            element_id_parts.append(eids)

        if not values_parts:
            return _empty_element_slab(component, time, time_slice)

        return ElementSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_ids=np.concatenate(element_id_parts),
            time=time[t_idx],
        )

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_line_station_slab(component, time, t_idx)
        stage_grp = self._h5[mpco_name]
        on_elements = stage_grp.get("RESULTS/ON_ELEMENTS")
        model_elements = stage_grp.get("MODEL/ELEMENTS")
        if on_elements is None or model_elements is None:
            return _empty_line_station_slab(component, time, t_idx)

        token, buckets = _mline.discover_line_station_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_line_station_slab(component, time, t_idx)
        token_grp = on_elements[token]

        # Read per-bucket slabs and stitch on the station axis.
        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        station_coord_parts: list[ndarray] = []

        for bucket in buckets:
            bucket_grp = token_grp[bucket.bracket_key]
            result = _mline.read_line_bucket_slab(
                bucket_grp, model_elements, bucket, component,
                t_idx=t_idx, element_ids=element_ids,
            )
            if result is None:
                continue
            values, element_index, station_coord = result
            values_parts.append(values)
            element_index_parts.append(element_index)
            station_coord_parts.append(station_coord)

        if not values_parts:
            return _empty_line_station_slab(component, time, t_idx)

        return LineStationSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=np.concatenate(element_index_parts),
            station_natural_coord=np.concatenate(station_coord_parts),
            time=time[t_idx],
        )

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_gauss_slab(component, time, t_idx)
        on_elements = self._h5[mpco_name].get("RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return _empty_gauss_slab(component, time, t_idx)

        # Dispatch on canonical type: material-state tokens
        # (``damage`` / ``equivalent_plastic_strain`` and their
        # ``_tension``/``_compression`` variants) take the META-driven
        # path; everything else uses the catalog-driven path.
        if _mmat.parent_token_for_canonical(component) is not None:
            return self._read_gauss_material_state(
                on_elements, component, time, t_idx, element_ids,
            )

        token, buckets = _melem.discover_gauss_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_gauss_slab(component, time, t_idx)

        # Read per-bucket slabs and stitch on the GP/element axis.
        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        natural_coords_parts: list[ndarray] = []

        for bucket in buckets:
            # Buckets may live under different alias group names
            # (e.g. legacy ``stresses`` vs. modern ``material.stress``)
            # — use each bucket's own source group.
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            result = _melem.read_bucket_slab(
                bucket_grp, bucket, component,
                t_idx=t_idx, element_ids=element_ids,
            )
            if result is None:
                continue
            values, element_index, natural_coords = result
            values_parts.append(values)
            element_index_parts.append(element_index)
            natural_coords_parts.append(natural_coords)

        if not values_parts:
            return _empty_gauss_slab(component, time, t_idx)

        return GaussSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=np.concatenate(element_index_parts),
            natural_coords=np.concatenate(natural_coords_parts, axis=0),
            local_axes_quaternion=None,
            time=time[t_idx],
        )

    def _read_gauss_material_state(
        self,
        on_elements: "h5py.Group",
        component: str,
        time: ndarray,
        t_idx: ndarray,
        element_ids: Optional[ndarray],
    ) -> GaussSlab:
        """META-driven material-state read — separate from the catalog
        path because ``n_components_per_gp`` depends on the assigned
        constitutive model, not just the element class."""
        token, buckets = _mmat.discover_material_state_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_gauss_slab(component, time, t_idx)

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        natural_coords_parts: list[ndarray] = []

        for bucket in buckets:
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            try:
                result = _mmat.read_material_state_bucket_slab(
                    bucket_grp, bucket, component,
                    t_idx=t_idx, element_ids=element_ids,
                )
            except ValueError:
                # Malformed META — skip rather than crash the read.
                continue
            if result is None:
                continue
            values, element_index, natural_coords = result
            values_parts.append(values)
            element_index_parts.append(element_index)
            natural_coords_parts.append(natural_coords)

        if not values_parts:
            return _empty_gauss_slab(component, time, t_idx)

        return GaussSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=np.concatenate(element_index_parts),
            natural_coords=np.concatenate(natural_coords_parts, axis=0),
            local_axes_quaternion=None,
            time=time[t_idx],
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
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_fiber_slab(component, time, t_idx)
        stage_grp = self._h5[mpco_name]
        on_elements = stage_grp.get("RESULTS/ON_ELEMENTS")
        model_elements = stage_grp.get("MODEL/ELEMENTS")
        section_assignments = stage_grp.get("MODEL/SECTION_ASSIGNMENTS")
        if (
            on_elements is None
            or model_elements is None
            or section_assignments is None
        ):
            return _empty_fiber_slab(component, time, t_idx)

        token, buckets = _mfiber.discover_fiber_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_fiber_slab(component, time, t_idx)
        token_grp = on_elements[token]

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        gp_index_parts: list[ndarray] = []
        y_parts: list[ndarray] = []
        z_parts: list[ndarray] = []
        area_parts: list[ndarray] = []
        material_tag_parts: list[ndarray] = []

        for bucket in buckets:
            bucket_grp = token_grp[bucket.bracket_key]
            result = _mfiber.read_fiber_bucket_slab(
                bucket_grp, model_elements, section_assignments, bucket,
                t_idx=t_idx,
                element_ids=element_ids,
                gp_indices=gp_indices,
            )
            if result is None:
                continue
            values, ei, gpi, y, z, area, mtag = result
            values_parts.append(values)
            element_index_parts.append(ei)
            gp_index_parts.append(gpi)
            y_parts.append(y)
            z_parts.append(z)
            area_parts.append(area)
            material_tag_parts.append(mtag)

        if not values_parts:
            return _empty_fiber_slab(component, time, t_idx)

        return FiberSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=np.concatenate(element_index_parts),
            gp_index=np.concatenate(gp_index_parts),
            y=np.concatenate(y_parts),
            z=np.concatenate(z_parts),
            area=np.concatenate(area_parts),
            material_tag=np.concatenate(material_tag_parts),
            time=time[t_idx],
        )

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
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_layer_slab(component, time, t_idx)
        stage_grp = self._h5[mpco_name]
        on_elements = stage_grp.get("RESULTS/ON_ELEMENTS")
        section_assignments = stage_grp.get("MODEL/SECTION_ASSIGNMENTS")
        if on_elements is None or section_assignments is None:
            return _empty_layer_slab(component, time, t_idx)
        local_axes = stage_grp.get("MODEL/LOCAL_AXES")  # may be None

        token, buckets = _mlayer.discover_layer_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_layer_slab(component, time, t_idx)

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        gp_index_parts: list[ndarray] = []
        layer_index_parts: list[ndarray] = []
        sub_gp_index_parts: list[ndarray] = []
        thickness_parts: list[ndarray] = []
        quaternion_parts: list[ndarray] = []

        for bucket in buckets:
            # Buckets may live under either ``material.fiber.<X>``
            # (swapped) or ``section.fiber.<X>`` (unswapped). Use
            # each bucket's own source group.
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            try:
                result = _mlayer.read_layer_bucket_slab(
                    bucket_grp, section_assignments, local_axes, bucket,
                    component,
                    t_idx=t_idx,
                    element_ids=element_ids,
                    gp_indices=gp_indices,
                    layer_indices=layer_indices,
                )
            except ValueError:
                # Malformed META — skip rather than crash.
                continue
            if result is None:
                continue
            (values, ei, gpi, lyri, subi,
             thick, quat) = result
            values_parts.append(values)
            element_index_parts.append(ei)
            gp_index_parts.append(gpi)
            layer_index_parts.append(lyri)
            sub_gp_index_parts.append(subi)
            thickness_parts.append(thick)
            quaternion_parts.append(quat)

        if not values_parts:
            return _empty_layer_slab(component, time, t_idx)

        return LayerSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=np.concatenate(element_index_parts),
            gp_index=np.concatenate(gp_index_parts),
            layer_index=np.concatenate(layer_index_parts),
            sub_gp_index=np.concatenate(sub_gp_index_parts),
            thickness=np.concatenate(thickness_parts),
            local_axes_quaternion=np.concatenate(quaternion_parts, axis=0),
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


def _empty_gauss_slab(component: str, time: ndarray, t_idx) -> GaussSlab:
    return GaussSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        natural_coords=np.zeros((0, 3), dtype=np.float64),
        local_axes_quaternion=None,
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_line_station_slab(
    component: str, time: ndarray, t_idx,
) -> LineStationSlab:
    return LineStationSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        station_natural_coord=np.array([], dtype=np.float64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_fiber_slab(component: str, time: ndarray, t_idx) -> FiberSlab:
    return FiberSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        gp_index=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.float64),
        z=np.array([], dtype=np.float64),
        area=np.array([], dtype=np.float64),
        material_tag=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_layer_slab(component: str, time: ndarray, t_idx) -> LayerSlab:
    return LayerSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        gp_index=np.array([], dtype=np.int64),
        layer_index=np.array([], dtype=np.int64),
        sub_gp_index=np.array([], dtype=np.int64),
        thickness=np.array([], dtype=np.float64),
        local_axes_quaternion=np.zeros((0, 4), dtype=np.float64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )
