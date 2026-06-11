"""FiberSectionDiagram — 3-D dot cloud + 2-D side panel for fiber sections.

Per fiber:

* World position = beam-GP position + ``y * y_local + z * z_local`` in
  the section local frame.
* Per-step value = ``fiber_stress`` (or whatever component the
  selector specifies) at the active step.

The 2-D side panel is a matplotlib scatter showing fibers at their
``(y, z)`` section coordinates, colored by the same value. Pick a beam
GP via the panel's dropdown (or programmatically through the
Director's ``picked_gp``) and the scatter updates.

Render seam (ADR 0042, R-B Wave 2 #2). The 3-D dot cloud is emitted as
a vertex-cell point-cloud :class:`MeshLayer` through the backend
(GPU sphere billboards via ``point_size`` + ``render_points_as_spheres``,
``pickable=False`` so clicks pass through to the substrate). The diagram
holds no VTK objects. The matplotlib side panel stays OUT of the IR —
it remains a diagram-owned ``make_side_panel`` hook.

Performance contract: endpoints, local frames, and per-fiber world
positions are computed **once at attach**. Per-step update is one h5py
read + a backend ``update_layer`` whose in-place fast path mutates the
bound dataset's scalar array (topology unchanged) — actor + dataset
identity stable across steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._beam_geometry import (
    collect_endpoints_with_substrate_rows,
    compute_local_axes,
    recorder_z_axes,
    station_position,
)
from ._scalar_color_support import ScalarColorSupport
from ._styles import FiberSectionStyle
from ..scene_ir import (
    CellBlocks,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
    ScalarField,
)

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


class FiberSectionDiagram(ScalarColorSupport, Diagram):
    """Per-fiber dot cloud + 2-D section panel for fiber-section beams."""

    kind = "fiber_section"
    topology = "fibers"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, FiberSectionStyle):
            raise TypeError(
                "FiberSectionDiagram requires a FiberSectionStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        # Populated by attach()
        self._layer: Optional[MeshLayer] = None
        self._handle: Any = None
        self._points: Optional[PointSet] = None     # cached fiber positions
        self._cells: Optional[CellBlocks] = None     # cached vertex cells
        self._fiber_values: Optional[ndarray] = None  # current-step scalar
        self._element_ids_to_read: tuple[int, ...] = ()

        # Per-fiber metadata (row-aligned with the emitted point cloud)
        self._slab_eid: Optional[ndarray] = None      # (n_fibers,)
        self._slab_gp: Optional[ndarray] = None
        self._slab_y: Optional[ndarray] = None        # section-local y
        self._slab_z: Optional[ndarray] = None
        self._slab_area: Optional[ndarray] = None
        # Mapping slab row -> our point index. The reader returns slab
        # rows in file-storage order; we lock that order at attach so
        # later reads can be used directly without permutation.
        self._row_order_locked = False

        # Per-(eid, gp) pre-computed start/end into the slab arrays
        # so the panel can look up fibers for a picked GP O(1).
        self._gp_to_indices: dict[tuple[int, int], ndarray] = {}

        # Deformation-following caches (sync_substrate_points): per-beam
        # endpoint substrate rows, the resolved vecxz (recorder z /
        # model vecxz / None), and the per-fiber resolved station ξ.
        self._endpoint_subs: dict[int, tuple[int, int]] = {}
        self._element_vecxz: dict[int, "ndarray | None"] = {}
        self._station_xi_res: Optional[ndarray] = None

        # Scalar-bar + runtime colour state + LUT mirror (mixin).
        self._init_scalar_color_state()

    # ------------------------------------------------------------------
    # Attach / detach / update
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "FiberSectionDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, view, scene)
        style: FiberSectionStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (line elements only) ────────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_line_element_ids(view)
        if element_ids.size == 0:
            return
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Step-0 read to discover fibers ──────────────────────────
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.fibers.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[0],
            )
        except Exception as exc:
            raise RuntimeError(
                f"FiberSectionDiagram could not read fibers slab: {exc}"
            )
        if slab.values.size == 0:
            return

        slab_eid = np.asarray(slab.element_index, dtype=np.int64)
        slab_gp = np.asarray(slab.gp_index, dtype=np.int64)
        slab_y = np.asarray(slab.y, dtype=np.float64)
        slab_z = np.asarray(slab.z, dtype=np.float64)
        slab_area = np.asarray(slab.area, dtype=np.float64)
        n = slab_eid.size

        # ── Endpoints lookup for unique beam IDs ────────────────────
        unique_eids = np.unique(slab_eid)
        endpoints, endpoint_subs = collect_endpoints_with_substrate_rows(
            view, scene, unique_eids,
        )
        self._endpoint_subs = endpoint_subs

        # ── Cache per-beam local frames (computed once) ─────────────
        # Roll source, best first: the recorder's true frame (.ladruno
        # MODEL/LOCAL_AXES — its z-axis is the geomTransf-equivalent
        # vecxz), then the model's geomTransf vecxz, then the geometric
        # default. The fiber (y, z) section coords only land in the
        # right plane when the roll matches the analysis frame. The
        # resolved vecxz is cached so sync_substrate_points re-derives
        # the same roll on the deformed chord.
        recorder_z = self._recorder_z_axes(results, unique_eids)
        _vecxz_for = getattr(view.elements, "vecxz_for", None)
        local_frames: dict[int, tuple[ndarray, ndarray, ndarray, ndarray]] = {}
        for eid_int in (int(e) for e in unique_eids):
            if eid_int not in endpoints:
                continue
            ci, cj = endpoints[eid_int]
            vecxz = recorder_z.get(eid_int)
            if vecxz is None and _vecxz_for is not None:
                vecxz = _vecxz_for(eid_int)
            try:
                _, y_local, z_local, _ = compute_local_axes(ci, cj, vecxz)
            except ValueError:
                continue
            local_frames[eid_int] = (ci, cj, y_local, z_local)
            self._element_vecxz[eid_int] = vecxz

        # ── Compute world positions for each fiber ──────────────────
        world_pts = np.zeros((n, 3), dtype=np.float64)
        valid_mask = np.zeros(n, dtype=bool)

        # Station ξ per row: the slab's TRUE integration-point
        # coordinate (MPCO ``GP_X`` / .ladruno ``GP_PARAM`` / live
        # ``integrationPoints``). Rows without one — files written
        # before the field existed, or a failed capture-side geometry
        # probe — fall back to a uniform spread inferred from the
        # element's GP count, and say so (ADR 0056 INV-6): for
        # force-based beams with non-uniform rules the inferred
        # positions are approximate.
        slab_xi = (
            np.asarray(slab.station_natural_coord, dtype=np.float64)
            if slab.station_natural_coord is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        gp_count_per_beam: dict[int, int] = {}
        for eid_int in (int(e) for e in unique_eids):
            mask = slab_eid == eid_int
            if not mask.any():
                continue
            gp_count_per_beam[eid_int] = int(slab_gp[mask].max() + 1)

        n_inferred = 0
        xi_res = np.zeros(n, dtype=np.float64)
        for k in range(n):
            eid = int(slab_eid[k])
            if eid not in local_frames:
                continue
            ci, cj, y_local, z_local = local_frames[eid]
            xi = float(slab_xi[k])
            if not np.isfinite(xi):
                n_gp = gp_count_per_beam.get(eid, 1)
                if n_gp <= 1:
                    xi = 0.0
                else:
                    # Uniform spread: -1 .. +1 (approximation).
                    xi = -1.0 + 2.0 * float(slab_gp[k]) / (n_gp - 1)
                n_inferred += 1
            xi_res[k] = xi
            base = station_position(ci, cj, xi)
            world_pts[k] = (
                base + slab_y[k] * y_local + slab_z[k] * z_local
            )
            valid_mask[k] = True

        if n_inferred:
            from .._log import log_action
            log_action(
                "diagram", "fiber_station_xi_inferred",
                n_fibers=n_inferred,
                component=str(self.spec.selector.component),
                _level="warning",
            )

        if not valid_mask.any():
            return

        # Trim to valid fibers
        if not valid_mask.all():
            world_pts = world_pts[valid_mask]
            slab_eid = slab_eid[valid_mask]
            slab_gp = slab_gp[valid_mask]
            slab_y = slab_y[valid_mask]
            slab_z = slab_z[valid_mask]
            slab_area = slab_area[valid_mask]
            xi_res = xi_res[valid_mask]
            slab_values_step0 = np.asarray(slab.values[0])[valid_mask]
        else:
            slab_values_step0 = np.asarray(slab.values[0])
        self._station_xi_res = xi_res

        n_valid = world_pts.shape[0]
        self._slab_eid = slab_eid
        self._slab_gp = slab_gp
        self._slab_y = slab_y
        self._slab_z = slab_z
        self._slab_area = slab_area
        self._row_order_locked = True

        # ── Cache point-cloud geometry (one vertex cell per fiber) ──
        self._points = PointSet(world_pts)
        self._cells = CellBlocks(
            {"vertex": np.arange(n_valid, dtype=np.int64).reshape(-1, 1)}
        )
        self._fiber_values = slab_values_step0.astype(np.float64)

        # ── Build (eid, gp) -> indices map for the side panel ──────
        keys = list(zip(slab_eid.tolist(), slab_gp.tolist()))
        gp_map: dict[tuple[int, int], list[int]] = {}
        for idx, key in enumerate(keys):
            gp_map.setdefault(key, []).append(idx)
        self._gp_to_indices = {
            k: np.asarray(v, dtype=np.int64) for k, v in gp_map.items()
        }

        # ── Initial clim ────────────────────────────────────────────
        if style.clim is not None:
            self._initial_clim = (
                float(style.clim[0]), float(style.clim[1]),
            )
        else:
            finite = slab_values_step0[np.isfinite(slab_values_step0)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        self._layer = self._build_layer(self._fiber_values)
        self._handle = self._backend.add_layer(self._layer)

        self._init_lut()
        if self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle,
                ScalarBarSpec(
                    layer_id=self._handle.layer_id,
                    title=self._scalar_bar_title(),
                    lut=self._current_lutspec(),
                ),
            )

    def update_to_step(self, step_index: int) -> None:
        if (
            self._layer is None
            or self._handle is None
            or self._fiber_values is None
        ):
            return
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.fibers.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return
        if slab.values.size == 0:
            return
        # Slab row order should match attach-time order — verify cheaply
        # and skip on mismatch (which would mean a partition / file change).
        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if slab_values.size != self._fiber_values.size:
            return
        self._fiber_values = slab_values
        self._layer = self._build_layer(self._fiber_values)
        # Topology is unchanged across steps, so the backend's in-place
        # fast path recolours the bound dataset without re-adding the
        # actor (perf contract).
        self._backend.update_layer(self._handle, self._layer)

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Move the fiber cloud with the deformed substrate.

        Re-samples each beam's endpoints from the (deformed) substrate,
        rebuilds the frame from the new chord + the cached vecxz
        (recorder z / model vecxz / geometric default), and replaces the
        cloud points — values are untouched.
        """
        if (
            self._handle is None
            or self._points is None
            or self._fiber_values is None
            or self._slab_eid is None
            or self._slab_y is None
            or self._slab_z is None
            or self._station_xi_res is None
            or not self._endpoint_subs
        ):
            return
        try:
            target = (
                np.asarray(deformed_pts, dtype=np.float64)
                if deformed_pts is not None
                else np.asarray(scene.grid.points, dtype=np.float64)
            )
        except Exception:
            return

        frames: dict[int, tuple[ndarray, ndarray, ndarray, ndarray]] = {}
        for eid, (si, sj) in self._endpoint_subs.items():
            if si >= target.shape[0] or sj >= target.shape[0]:
                continue
            ci = target[si]
            cj = target[sj]
            try:
                _, y_local, z_local, _ = compute_local_axes(
                    ci, cj, self._element_vecxz.get(int(eid)),
                )
            except ValueError:
                continue
            frames[int(eid)] = (ci, cj, y_local, z_local)
        if not frames:
            return

        pts = np.asarray(self._points.coords, dtype=np.float64).copy()
        for k in range(self._slab_eid.size):
            frame = frames.get(int(self._slab_eid[k]))
            if frame is None:
                continue
            ci, cj, y_local, z_local = frame
            base = station_position(
                ci, cj, float(self._station_xi_res[k]),
            )
            pts[k] = (
                base
                + self._slab_y[k] * y_local
                + self._slab_z[k] * z_local
            )
        self._points = PointSet(pts)
        self._layer = self._build_layer(self._fiber_values)
        # Same fast path as a step change — points swap in place.
        self._backend.update_layer(self._handle, self._layer)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._points = None
        self._cells = None
        self._fiber_values = None
        self._element_ids_to_read = ()
        self._slab_eid = None
        self._slab_gp = None
        self._slab_y = None
        self._slab_z = None
        self._slab_area = None
        self._row_order_locked = False
        self._gp_to_indices = {}
        self._endpoint_subs = {}
        self._element_vecxz = {}
        self._station_xi_res = None
        self._initial_clim = None
        super().detach()

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def make_side_panel(self, director: Any) -> Any:
        if not self.is_attached:
            return None
        try:
            from ..ui._section_panel import FiberSectionPanel
        except ImportError:
            return None
        return FiberSectionPanel(self, director)

    # ------------------------------------------------------------------
    # Side-panel helpers
    # ------------------------------------------------------------------

    def available_gps(self) -> list[tuple[int, int]]:
        """List of ``(element_id, gp_index)`` pairs that have fibers."""
        return sorted(self._gp_to_indices.keys())

    def read_section_at_gp(
        self, element_id: int, gp_index: int, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray, ndarray, ndarray]]:
        """Return ``(y, z, area, values)`` for one beam GP at one step.

        Returns ``None`` if the (eid, gp) pair has no fibers.
        """
        if (
            self._slab_y is None or self._slab_z is None
            or self._slab_area is None
        ):
            return None
        key = (int(element_id), int(gp_index))
        idxs = self._gp_to_indices.get(key)
        if idxs is None or idxs.size == 0:
            return None

        # Read this step's values (cheap — only reads the fibers slab)
        results = self._scoped_results()
        if results is None:
            return None
        try:
            slab = results.elements.fibers.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return None
        if slab.values.size == 0:
            return None

        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if slab_values.size != (self._slab_y.size):
            return None

        return (
            self._slab_y[idxs],
            self._slab_z[idxs],
            self._slab_area[idxs],
            slab_values[idxs],
        )

    # ------------------------------------------------------------------
    # Runtime style — clim/cmap/LUT from ScalarColorSupport
    # ------------------------------------------------------------------

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._fiber_values

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"fiber_{id(self):x}"

    def _color_array_name(self) -> str:
        return self.spec.selector.component or "_fiber_value"

    def _build_layer(self, fiber_values: ndarray) -> MeshLayer:
        """Point-cloud MeshLayer: GPU sphere billboards coloured by the
        fiber value through the LUT; decorative (pickable=False)."""
        style: FiberSectionStyle = self.spec.style    # type: ignore[assignment]
        assert self._points is not None and self._cells is not None
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        name = self._color_array_name()
        color = ColorSpec(
            mode="by_array",
            array_name=name,
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return MeshLayer(
            layer_id=self._layer_id(),
            points=self._points,
            cells=self._cells,
            fields=(ScalarField(name, fiber_values, "point"),),
            color=color,
            opacity=style.opacity,
            point_size=10.0,
            render_points_as_spheres=True,
            pickable=False,
        )

    @staticmethod
    def _recorder_z_axes(
        results: "Results", element_ids: ndarray,
    ) -> dict[int, ndarray]:
        """``eid -> recorder local z-axis`` — see
        :func:`~apeGmsh.viewers.diagrams._beam_geometry.recorder_z_axes`
        (shared with the line-force diagram and local-axes overlay)."""
        return recorder_z_axes(results, element_ids)

    @staticmethod
    def _collect_line_element_ids(view: "ViewerData") -> ndarray:
        ids: list[int] = []
        for group in view.elements:
            if group.element_type.dim == 1:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

