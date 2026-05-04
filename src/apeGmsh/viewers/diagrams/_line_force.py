"""LineForceDiagram — beam internal-force / strain diagrams.

Renders the classic textbook fill perpendicular to each beam axis,
amplitude proportional to the value at the integration station, with
linear interpolation between stations.

Performance contract:

* Endpoint coordinates and per-station fill geometry are built **once
  at attach** from a step-0 read of the line-stations slab. The
  topology — quads between adjacent stations — never changes.
* Per-step update reads only the values, scatters them into the
  per-station ordering computed at attach, and mutates the *top*
  half of the polydata's points array in place.
* The actor and PolyData object identities are stable across step
  changes.

Component-to-axis mapping defaults to the structural convention
(``shear_y`` and ``bending_moment_z`` along ``y_local``; ``shear_z`` and
``bending_moment_y`` along ``z_local``; axial / torsion along ``z_local``).
The user can override via ``LineForceStyle.fill_axis``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._beam_geometry import (
    compute_local_axes,
    normalize_fill_axis_spec,
    resolve_fill_direction,
    station_position,
)
from ._styles import LineForceStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


class LineForceDiagram(Diagram):
    """Per-beam fill diagram driven by a ``LineStationSlab``."""

    kind = "line_force"
    topology = "line_stations"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, LineForceStyle):
            raise TypeError(
                "LineForceDiagram requires a LineForceStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        # Populated by attach()
        self._fill_polydata: Optional[pv.PolyData] = None
        self._fill_actor: Any = None
        self._n_stations: int = 0
        self._base_points: Optional[ndarray] = None
        self._fill_directions: Optional[ndarray] = None
        self._our_to_slab_index: Optional[ndarray] = None
        self._element_ids_to_read: tuple[int, ...] = ()
        self._initial_scale: float = 1.0
        self._last_step: int = 0
        # Per-station metadata used by sync_substrate_points to rebuild
        # base points and local fill directions when the substrate warps.
        self._station_eid: Optional[ndarray] = None
        self._station_xi: Optional[ndarray] = None
        # Endpoint substrate indices: eid -> (i_idx, j_idx) into
        # ``scene.grid.points``. Cached at attach.
        self._endpoint_subs_idx: dict[int, tuple[int, int]] = {}

        # Mutable runtime overrides
        self._runtime_scale: Optional[float] = None
        # Forces full re-attach. ``None`` means "fall through to
        # spec.style.fill_axis"; a non-None value (str or tuple) wins.
        self._runtime_axis: "str | tuple[float, float, float] | None" = None
        self._runtime_flip: Optional[bool] = None

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
                "LineForceDiagram.attach requires a FEMSceneData. The "
                "Director must call bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, fem, scene)
        style: LineForceStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (line elements only) ────────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_line_element_ids(fem)
        if element_ids.size == 0:
            from ._base import NoDataError
            raise NoDataError(
                f"LineForceDiagram: selector resolved to no line "
                f"elements (selector={self.spec.selector!r})."
            )
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Step-0 line-stations read to discover topology ──────────
        results = self._scoped_results()
        if results is None:
            from ._base import NoDataError
            raise NoDataError(
                "LineForceDiagram: results scope unresolved (no stage)."
            )
        try:
            slab = results.elements.line_stations.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LineForceDiagram could not read line_stations for "
                f"component {self.spec.selector.component!r}: {exc}"
            )
        if slab.values.size == 0 or slab.element_index.size == 0:
            from ._base import NoDataError
            raise NoDataError(
                f"LineForceDiagram: no line-station data for component "
                f"{self.spec.selector.component!r} on the selected "
                f"elements (slab is empty). Use "
                f"`results.inspect.diagnose({self.spec.selector.component!r})` "
                f"to see which buckets were checked."
            )

        slab_eids = np.asarray(slab.element_index, dtype=np.int64)
        slab_xi = np.asarray(slab.station_natural_coord, dtype=np.float64)

        # ── Endpoint lookup for each unique beam ────────────────────
        unique_eids = np.unique(slab_eids)
        endpoints, endpoint_subs = self._collect_endpoints_with_subs(
            fem, scene, unique_eids,
        )
        self._endpoint_subs_idx = endpoint_subs

        # ── Build per-beam geometry ─────────────────────────────────
        n_total = slab_eids.size
        base_points = np.zeros((n_total, 3), dtype=np.float64)
        fill_dirs = np.zeros((n_total, 3), dtype=np.float64)
        our_to_slab = np.zeros(n_total, dtype=np.int64)
        station_eid = np.zeros(n_total, dtype=np.int64)
        station_xi = np.zeros(n_total, dtype=np.float64)

        axis_spec = (
            self._runtime_axis if self._runtime_axis is not None
            else style.fill_axis
        )

        faces_list: list[int] = []
        running = 0
        for eid in unique_eids:
            eid_int = int(eid)
            if eid_int not in endpoints:
                continue
            ci, cj = endpoints[eid_int]
            try:
                x_local, y_local, z_local, _ = compute_local_axes(ci, cj)
            except ValueError:
                continue

            fill_dir = resolve_fill_direction(
                self.spec.selector.component,
                axis_spec, x_local, y_local, z_local,
            )

            slab_idx_for_beam = np.where(slab_eids == eid)[0]
            xi_values = slab_xi[slab_idx_for_beam]
            sort_order = np.argsort(xi_values, kind="stable")
            sorted_slab_indices = slab_idx_for_beam[sort_order]
            sorted_xi = xi_values[sort_order]
            n = sorted_slab_indices.size

            for k in range(n):
                our_idx = running + k
                base_points[our_idx] = station_position(
                    ci, cj, float(sorted_xi[k]),
                )
                fill_dirs[our_idx] = fill_dir
                our_to_slab[our_idx] = sorted_slab_indices[k]
                station_eid[our_idx] = eid_int
                station_xi[our_idx] = float(sorted_xi[k])

            if n >= 2:
                for k in range(n - 1):
                    b_k = running + k
                    b_k1 = running + k + 1
                    # Top indices live at offset n_total above the base
                    # block — but the *final* offset is the value of
                    # ``running`` after the loop. We patch it later.
                    faces_list.extend([
                        4, b_k, b_k1,
                        -(b_k1 + 1),    # placeholder, fixed up below
                        -(b_k + 1),
                    ])
            running += n

        if running == 0 or not faces_list:
            return

        # Trim to actual count (in case some beams were skipped).
        base_points = base_points[:running]
        fill_dirs = fill_dirs[:running]
        our_to_slab = our_to_slab[:running]
        station_eid = station_eid[:running]
        station_xi = station_xi[:running]

        # Patch placeholder -(idx+1) -> idx + running (top index).
        faces_arr = np.asarray(faces_list, dtype=np.int64)
        mask = faces_arr < 0
        faces_arr[mask] = -faces_arr[mask] - 1 + running

        # Initial top points = base (zero magnitude); we update right after.
        all_points = np.vstack([base_points, base_points.copy()])

        polydata = pv.PolyData(all_points, faces_arr)
        self._fill_polydata = polydata
        self._n_stations = running
        self._base_points = base_points
        self._fill_directions = fill_dirs
        self._our_to_slab_index = our_to_slab
        self._station_eid = station_eid
        self._station_xi = station_xi

        # ── Auto-fit scale from global max across all steps ─────────
        if style.scale is None:
            max_abs = float(np.abs(slab.values).max()) if slab.values.size else 0.0
            if max_abs > 0.0 and scene.model_diagonal > 0.0:
                self._initial_scale = (
                    style.auto_scale_fraction * scene.model_diagonal / max_abs
                )
            else:
                self._initial_scale = 1.0
        else:
            self._initial_scale = float(style.scale)

        # Apply step 0 values (initial display state; update_to_step
        # will refresh once the director's current step is pushed)
        self._apply_values(np.asarray(slab.values[0], dtype=np.float64))

        actor = plotter.add_mesh(
            polydata,
            color=style.fill_color,
            opacity=style.opacity,
            show_edges=style.show_edges,
            edge_color=style.edge_color,
            line_width=1,
            lighting=False,
            smooth_shading=False,
            name=self._actor_name(),
            reset_camera=False,
        )
        self._fill_actor = actor
        self._actors = [actor]
        self._last_step = 0

    def update_to_step(self, step_index: int) -> None:
        if self._fill_polydata is None or self._our_to_slab_index is None:
            return
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.line_stations.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return
        if slab.values.size == 0:
            return
        self._apply_values(np.asarray(slab.values[0], dtype=np.float64))
        self._last_step = int(step_index)

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Move the diagram fills to follow the deformed substrate.

        For each beam in the slab, fetch the deformed endpoint coords,
        rebuild ``base_points`` (and the per-station ``fill_directions``
        from the new local axes), then re-apply the current step's
        values so the top-of-fill points refresh against the new bases.
        """
        if (
            self._fill_polydata is None
            or self._base_points is None
            or self._fill_directions is None
            or self._station_eid is None
            or self._station_xi is None
            or not self._endpoint_subs_idx
        ):
            return
        try:
            target_pts = (
                np.asarray(deformed_pts, dtype=np.float64)
                if deformed_pts is not None
                else np.asarray(scene.grid.points, dtype=np.float64)
            )
        except Exception:
            return

        style: LineForceStyle = self.spec.style    # type: ignore[assignment]
        axis_spec = (
            self._runtime_axis if self._runtime_axis is not None
            else style.fill_axis
        )

        new_base = self._base_points.copy()
        new_dirs = self._fill_directions.copy()
        for eid, (si, sj) in self._endpoint_subs_idx.items():
            if si >= target_pts.shape[0] or sj >= target_pts.shape[0]:
                continue
            ci = target_pts[si]
            cj = target_pts[sj]
            try:
                x_local, y_local, z_local, _ = compute_local_axes(ci, cj)
            except ValueError:
                continue
            fill_dir = resolve_fill_direction(
                self.spec.selector.component,
                axis_spec, x_local, y_local, z_local,
            )
            mask = self._station_eid == eid
            if not np.any(mask):
                continue
            xi = self._station_xi[mask]
            for k_idx, xi_val in zip(np.where(mask)[0], xi):
                new_base[k_idx] = station_position(ci, cj, float(xi_val))
                new_dirs[k_idx] = fill_dir

        self._base_points = new_base
        self._fill_directions = new_dirs
        # Rewrite the base block of the polydata in place; the top
        # block is then refreshed by re-applying the last step's values.
        try:
            all_pts = np.asarray(self._fill_polydata.points)
            all_pts[: self._n_stations] = new_base
            self._fill_polydata.Modified()
        except Exception:
            return
        self._reapply_last_step()

    def detach(self) -> None:
        self._fill_polydata = None
        self._fill_actor = None
        self._n_stations = 0
        self._base_points = None
        self._fill_directions = None
        self._our_to_slab_index = None
        self._element_ids_to_read = ()
        self._station_eid = None
        self._station_xi = None
        self._endpoint_subs_idx = {}
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        """Update the fill amplification factor; live re-render."""
        self._runtime_scale = float(scale)
        if self._fill_polydata is not None:
            self._reapply_last_step()

    def set_flip_sign(self, flip: bool) -> None:
        self._runtime_flip = bool(flip)
        if self._fill_polydata is not None:
            self._reapply_last_step()

    def set_fill_axis(
        self, axis: "str | tuple[float, float, float] | None",
    ) -> None:
        """Switch the fill direction.

        Accepts:

        * ``None`` — clear the runtime override and fall back to the
          spec's ``style.fill_axis`` (or the component default if that
          is also ``None``).
        * ``"y"`` / ``"z"`` — local-frame axis.
        * ``"global_x" | "global_y" | "global_z"`` — global axis,
          projected perpendicular to each beam axis.
        * ``(dx, dy, dz)`` — explicit world-frame direction.

        Triggers a full re-attach because the per-station fill
        directions are baked at attach.
        """
        normalized = normalize_fill_axis_spec(axis)
        if self._runtime_axis == normalized:
            return
        self._runtime_axis = normalized
        if self.is_attached and self._fem is not None:
            # Capture before detach() — that call clears _fem/_plotter/_scene.
            plotter = self._plotter
            scene = self._scene
            fem = self._fem
            last_step = self._last_step
            self.detach()
            self.attach(plotter, fem, scene)
            # Re-attach starts at step-0 values against the undeformed
            # FEM coords. Push the step the user was on, then sync to
            # the current (possibly deformed) substrate so the diagram
            # lands aligned without requiring an extra UI gesture.
            try:
                self.update_to_step(int(last_step))
            except Exception:
                pass
            if scene is not None:
                try:
                    self.sync_substrate_points(None, scene)
                except Exception:
                    pass

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_line_force_{id(self):x}"

    def _reapply_last_step(self) -> None:
        # Re-fetch values for the current step. Cheap; one h5py read.
        self.update_to_step(self._last_step)

    def _apply_values(self, slab_values: ndarray) -> None:
        """Mutate the top half of the polydata points in place."""
        if (
            self._fill_polydata is None
            or self._base_points is None
            or self._fill_directions is None
            or self._our_to_slab_index is None
        ):
            return

        style: LineForceStyle = self.spec.style    # type: ignore[assignment]
        flip = (
            self._runtime_flip
            if self._runtime_flip is not None else style.flip_sign
        )
        ours = slab_values[self._our_to_slab_index]
        if flip:
            ours = -ours
        scale = self.current_scale()
        offsets = (scale * ours)[:, None] * self._fill_directions
        top_points = self._base_points + offsets

        # Layout: [base_0..base_{n-1}, top_0..top_{n-1}]
        all_pts = np.asarray(self._fill_polydata.points)
        all_pts[self._n_stations:] = top_points
        try:
            self._fill_polydata.Modified()
        except Exception:
            pass

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _collect_line_element_ids(fem: "FEMData") -> ndarray:
        """Return all 1-D element IDs in the FEM."""
        ids: list[int] = []
        for group in fem.elements:
            if group.element_type.dim == 1:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _collect_endpoints_with_subs(
        fem: "FEMData",
        scene: "FEMSceneData",
        element_ids: ndarray,
    ) -> "tuple[dict[int, tuple[ndarray, ndarray]], dict[int, tuple[int, int]]]":
        """Return ``(eid -> (ci, cj), eid -> (i_sub, j_sub))``.

        ``i_sub`` / ``j_sub`` are row indices into ``scene.grid.points``
        (and ``scene.node_ids``) — what ``sync_substrate_points``
        re-samples from to follow the deformed substrate.
        """
        eid_set = {int(e) for e in element_ids}
        node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
        coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
        if node_ids_arr.size == 0:
            return {}, {}
        max_nid = int(node_ids_arr.max())
        nid_to_idx = np.full(max_nid + 2, -1, dtype=np.int64)
        nid_to_idx[node_ids_arr] = np.arange(
            node_ids_arr.size, dtype=np.int64,
        )

        # Substrate lookup: FEM node id -> row in scene.grid.points.
        scene_ids = np.asarray(scene.node_ids, dtype=np.int64)
        max_sid = int(scene_ids.max()) if scene_ids.size else -1
        nid_to_sub = np.full(max(max_sid, max_nid) + 2, -1, dtype=np.int64)
        nid_to_sub[scene_ids] = np.arange(scene_ids.size, dtype=np.int64)

        coords_out: dict[int, tuple[ndarray, ndarray]] = {}
        subs_out: dict[int, tuple[int, int]] = {}
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
                coords_out[eid] = (
                    coords_arr[ii].copy(),
                    coords_arr[jj].copy(),
                )
                si = int(nid_to_sub[nid_i]) if nid_i <= nid_to_sub.size - 1 else -1
                sj = int(nid_to_sub[nid_j]) if nid_j <= nid_to_sub.size - 1 else -1
                if si >= 0 and sj >= 0:
                    subs_out[eid] = (si, sj)
        return coords_out, subs_out
