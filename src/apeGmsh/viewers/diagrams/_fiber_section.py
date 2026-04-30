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

Performance contract: same as Phase 1 — endpoints, local frames, and
per-fiber world positions are computed **once at attach**. Per-step
update is one h5py read + numpy scatter into the existing
``point_data`` array; actor + polydata identity stable across steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._beam_geometry import compute_local_axes, station_position
from ._styles import FiberSectionStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_SCALAR_NAME = "_fiber_value"


class FiberSectionDiagram(Diagram):
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
        self._cloud: Optional[pv.PolyData] = None
        self._actor: Any = None
        self._scalar_array: Optional[ndarray] = None
        self._element_ids_to_read: tuple[int, ...] = ()

        # Per-fiber metadata (in the order they appear in self._cloud.points)
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

        self._initial_clim: Optional[tuple[float, float]] = None
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_cmap: Optional[str] = None

    # ------------------------------------------------------------------
    # Attach / detach / update
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        fem: "FEMData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "FiberSectionDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, fem, scene)
        style: FiberSectionStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (line elements only) ────────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_line_element_ids(fem)
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
        endpoints = self._collect_endpoints(fem, unique_eids)

        # ── Cache per-beam local frames (computed once) ─────────────
        local_frames: dict[int, tuple[ndarray, ndarray, ndarray, ndarray]] = {}
        for eid_int in (int(e) for e in unique_eids):
            if eid_int not in endpoints:
                continue
            ci, cj = endpoints[eid_int]
            try:
                _, y_local, z_local, _ = compute_local_axes(ci, cj)
            except ValueError:
                continue
            local_frames[eid_int] = (ci, cj, y_local, z_local)

        # ── Compute world positions for each fiber ──────────────────
        world_pts = np.zeros((n, 3), dtype=np.float64)
        valid_mask = np.zeros(n, dtype=bool)

        # We need the GP natural coord too. The fibers slab gives gp_index
        # but not natural_coord directly. To map gp_index -> natural coord,
        # we'd need an integration-rule lookup. For Phase 3 we use a uniform
        # spread when the rule is unknown: natural_coord = -1 + 2*gp/(N-1).
        gp_count_per_beam: dict[int, int] = {}
        for eid_int in (int(e) for e in unique_eids):
            mask = slab_eid == eid_int
            if not mask.any():
                continue
            gp_count_per_beam[eid_int] = int(slab_gp[mask].max() + 1)

        for k in range(n):
            eid = int(slab_eid[k])
            if eid not in local_frames:
                continue
            ci, cj, y_local, z_local = local_frames[eid]
            n_gp = gp_count_per_beam.get(eid, 1)
            if n_gp <= 1:
                xi = 0.0
            else:
                # Lobatto-style spread: -1 .. +1
                xi = -1.0 + 2.0 * float(slab_gp[k]) / (n_gp - 1)
            base = station_position(ci, cj, xi)
            world_pts[k] = (
                base + slab_y[k] * y_local + slab_z[k] * z_local
            )
            valid_mask[k] = True

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
            slab_values_step0 = np.asarray(slab.values[0])[valid_mask]
        else:
            slab_values_step0 = np.asarray(slab.values[0])

        n_valid = world_pts.shape[0]
        self._slab_eid = slab_eid
        self._slab_gp = slab_gp
        self._slab_y = slab_y
        self._slab_z = slab_z
        self._slab_area = slab_area
        self._row_order_locked = True

        # ── Build PolyData (point cloud) ────────────────────────────
        cloud = pv.PolyData(world_pts)
        cloud.point_data[_SCALAR_NAME] = slab_values_step0.astype(np.float64)
        self._cloud = cloud
        self._scalar_array = cloud.point_data[_SCALAR_NAME]

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

        # Point size in world units
        point_size = max(
            scene.model_diagonal * style.point_size_fraction, 1e-6,
        )

        actor = plotter.add_mesh(
            cloud,
            scalars=_SCALAR_NAME,
            cmap=self._runtime_cmap or style.cmap,
            clim=self._runtime_clim or self._initial_clim,
            opacity=style.opacity,
            render_points_as_spheres=True,
            point_size=10.0,        # screen-space; the dot cloud rendering
            show_scalar_bar=style.show_scalar_bar,
            scalar_bar_args={
                "title": self.spec.selector.component,
            } if style.show_scalar_bar else None,
            name=self._actor_name(),
            reset_camera=False,
            lighting=False,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if self._cloud is None or self._scalar_array is None:
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
        if slab_values.size != self._scalar_array.size:
            return
        self._scalar_array[:] = slab_values
        try:
            self._cloud.Modified()
        except Exception:
            pass

    def detach(self) -> None:
        self._cloud = None
        self._actor = None
        self._scalar_array = None
        self._element_ids_to_read = ()
        self._slab_eid = None
        self._slab_gp = None
        self._slab_y = None
        self._slab_z = None
        self._slab_area = None
        self._row_order_locked = False
        self._gp_to_indices = {}
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
    # Runtime style adjustments
    # ------------------------------------------------------------------

    def set_clim(self, vmin: float, vmax: float) -> None:
        if vmin == vmax:
            vmax = vmin + 1.0
        self._runtime_clim = (float(vmin), float(vmax))
        if self._actor is not None:
            try:
                mapper = self._actor.GetMapper()
                mapper.SetScalarRange(*self._runtime_clim)
            except Exception:
                pass

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        if self._scalar_array is None:
            return None
        data = np.asarray(self._scalar_array)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    def set_cmap(self, cmap: str) -> None:
        self._runtime_cmap = cmap
        if self._actor is None:
            return
        try:
            lut = pv.LookupTable(cmap)
            clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
            lut.scalar_range = clim
            mapper = self._actor.GetMapper()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(*clim)
        except Exception:
            pass

    def current_clim(self) -> Optional[tuple[float, float]]:
        return self._runtime_clim or self._initial_clim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_fiber_{id(self):x}"

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _collect_line_element_ids(fem: "FEMData") -> ndarray:
        ids: list[int] = []
        for group in fem.elements:
            if group.element_type.dim == 1:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _collect_endpoints(
        fem: "FEMData", element_ids: ndarray,
    ) -> dict[int, tuple[ndarray, ndarray]]:
        eid_set = {int(e) for e in element_ids}
        node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
        coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
        if node_ids_arr.size == 0:
            return {}
        max_nid = int(node_ids_arr.max())
        nid_to_idx = np.full(max_nid + 2, -1, dtype=np.int64)
        nid_to_idx[node_ids_arr] = np.arange(node_ids_arr.size, dtype=np.int64)
        out: dict[int, tuple[ndarray, ndarray]] = {}
        for group in fem.elements:
            if group.element_type.dim != 1:
                continue
            ids = np.asarray(group.ids, dtype=np.int64)
            conn = np.asarray(group.connectivity, dtype=np.int64)
            for k in range(len(group)):
                eid = int(ids[k])
                if eid not in eid_set:
                    continue
                nid_i = int(conn[k, 0])
                nid_j = int(conn[k, 1])
                ii = nid_to_idx[nid_i]
                jj = nid_to_idx[nid_j]
                if ii < 0 or jj < 0:
                    continue
                out[eid] = (
                    coords_arr[ii].copy(), coords_arr[jj].copy(),
                )
        return out
