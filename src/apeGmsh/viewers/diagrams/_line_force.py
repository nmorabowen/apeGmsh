"""LineForceDiagram — beam internal-force / strain diagrams.

Renders the classic textbook fill perpendicular to each beam axis,
amplitude proportional to the value at the integration station, with
linear interpolation between stations.

Render seam (ADR 0042, R-B Wave 3 #1). The fill is emitted as a
quad-cell :class:`MeshLayer` (solid fill colour, styled edges) through
``self._backend``; the diagram holds no VTK objects.

Performance contract:

* Endpoint coordinates and per-station fill geometry are built **once
  at attach** from a step-0 read of the line-stations slab. The
  topology — quads between adjacent stations — never changes, so the
  quad :class:`CellBlocks` is cached.
* Per-step update reads only the values, recomputes the ``(2N, 3)``
  points (top half = base + scale·value·dir), and emits through
  ``backend.update_layer`` — its mesh fast path replaces the bound
  dataset's points in place (the property-setter path VTK reliably
  re-reads), keeping the actor + dataset identity stable across steps.

Component-to-axis mapping defaults to the structural convention
(``shear_y`` and ``bending_moment_z`` along ``y_local``; ``shear_z`` and
``bending_moment_y`` along ``z_local``; axial / torsion along ``z_local``).
The user can override via ``LineForceStyle.fill_axis``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._beam_geometry import (
    axes_from_quaternion,
    collect_endpoints_with_substrate_rows,
    compute_local_axes,
    normalize_fill_axis_spec,
    resolve_fill_direction,
    station_position,
)
from ._kinds import register_diagram_kind
from ._styles import LineForceStyle
from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


def _default_style(component: str) -> LineForceStyle:
    """Default style for a line-force diagram.

    Bending moments default to ``flip_sign=True`` so the diagram
    renders on the tension side of the beam (sagging-positive
    convention universally used by structural engineers). Axial
    force, shear, and torsion keep the natural sign — those have no
    "tension side" tradition to follow.
    """
    return LineForceStyle(flip_sign=component.startswith("bending_moment"))


@register_diagram_kind(
    label="Line force diagram",
    style_class=LineForceStyle,
    style_factory=_default_style,
    order=30,
)
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
        self._handle: Any = None
        self._layer: Optional[MeshLayer] = None
        self._cells: Optional[CellBlocks] = None      # quad fill (stable)
        self._current_points: Optional[ndarray] = None  # (2N, 3), per step
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

        # eid -> geomTransf vecxz (3,) or None. Cached at attach from
        # ViewerData; drives the real fill orientation. When the slab
        # carries a recorder frame (.ladruno MODEL/LOCAL_AXES) the entry
        # is overlaid with that frame's z-axis — the true cross-section
        # roll wins over the model's vecxz / geometric default.
        self._element_vecxz: dict[int, "ndarray | None"] = {}

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
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "LineForceDiagram.attach requires a FEMSceneData. The "
                "Director must call bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, view, scene)
        style: LineForceStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (line elements only) ────────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_line_element_ids(view)
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
            view, scene, unique_eids,
        )
        self._endpoint_subs_idx = endpoint_subs

        # Real per-element geomTransf vecxz (h5 path). ``vecxz_for`` is a
        # ViewerData accessor; ``view`` may also be a raw FEMData (the
        # duck-typed legacy path used in tests) which lacks it — probe
        # defensively. Absent vecxz → None → compute_local_axes falls
        # back to the structural default, preserving prior behaviour.
        _vecxz_for = getattr(view.elements, "vecxz_for", None)
        self._element_vecxz = {
            int(e): (_vecxz_for(int(e)) if _vecxz_for is not None else None)
            for e in unique_eids
        }

        # Recorder beam frame (ADR 0056): where the slab carries a finite
        # per-row quaternion (.ladruno ``MODEL/LOCAL_AXES``), its z-axis is
        # the geomTransf-equivalent vecxz — compute_local_axes Gram-Schmidts
        # it against the chord, reproducing the recorder's true cross-section
        # roll here AND in sync_substrate_points' deformed re-derivation.
        # This is the frame a model-less ``from_ladruno`` open has no vecxz
        # for; NaN rows (element without a recorded frame) keep the fallback.
        if slab.local_axes_quaternion is not None:
            slab_quat = np.asarray(
                slab.local_axes_quaternion, dtype=np.float64,
            )
            for eid in unique_eids:
                rows = np.where(slab_eids == eid)[0]
                if rows.size and np.all(np.isfinite(slab_quat[rows[0]])):
                    _, _, z_rec = axes_from_quaternion(slab_quat[rows[0]])
                    self._element_vecxz[int(eid)] = z_rec

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

        # Quad connectivity (n_quads, 4) into the (2N, 3) point block.
        # Base points occupy rows [0, N); top points [N, 2N). Negative
        # placeholders mark "this index lives in the top block" — patched
        # to ``idx + running`` once the final N is known (same trick the
        # old pyvista faces array used).
        quads_list: list[list[int]] = []
        running = 0
        for eid in unique_eids:
            eid_int = int(eid)
            if eid_int not in endpoints:
                continue
            ci, cj = endpoints[eid_int]
            try:
                x_local, y_local, z_local, _ = compute_local_axes(
                    ci, cj, self._element_vecxz.get(eid_int),
                )
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
                    # Top indices live at offset N above the base block —
                    # but the *final* N is ``running`` after the loop.
                    # Patch the placeholders below.
                    quads_list.append([
                        b_k, b_k1,
                        -(b_k1 + 1),    # placeholder, fixed up below
                        -(b_k + 1),
                    ])
            running += n

        if running == 0 or not quads_list:
            return

        # Trim to actual count (in case some beams were skipped).
        base_points = base_points[:running]
        fill_dirs = fill_dirs[:running]
        our_to_slab = our_to_slab[:running]
        station_eid = station_eid[:running]
        station_xi = station_xi[:running]

        # Patch placeholder -(idx+1) -> idx + running (top index).
        quads = np.asarray(quads_list, dtype=np.int64)
        mask = quads < 0
        quads[mask] = -quads[mask] - 1 + running

        self._cells = CellBlocks({"quad": quads})
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
        self._compute_points(np.asarray(slab.values[0], dtype=np.float64))

        self._layer = self._build_layer()
        self._handle = self._backend.add_layer(self._layer)
        self._last_step = 0

    def update_to_step(self, step_index: int) -> None:
        if self._handle is None or self._our_to_slab_index is None:
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
        self._compute_points(np.asarray(slab.values[0], dtype=np.float64))
        self._push_update()
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
            self._handle is None
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
                x_local, y_local, z_local, _ = compute_local_axes(
                    ci, cj, self._element_vecxz.get(int(eid)),
                )
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
        # Re-applying the step recomputes the full points array and emits
        # through the backend's points fast path, pushing the change to
        # the rendered actor.
        self._reapply_last_step()

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    def detach(self) -> None:
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._handle = None
        self._layer = None
        self._cells = None
        self._current_points = None
        self._n_stations = 0
        self._base_points = None
        self._fill_directions = None
        self._our_to_slab_index = None
        self._element_ids_to_read = ()
        self._station_eid = None
        self._station_xi = None
        self._endpoint_subs_idx = {}
        self._element_vecxz = {}
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        """Update the fill amplification factor; live re-render."""
        self._runtime_scale = float(scale)
        if self._handle is not None:
            self._reapply_last_step()

    def set_flip_sign(self, flip: bool) -> None:
        self._runtime_flip = bool(flip)
        if self._handle is not None:
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
        if self.is_attached and self._view is not None:
            # Capture before detach() — that call clears _view/_backend/_scene.
            # Re-attach through the same RenderBackend (ADR 0042 R-B.final;
            # attach injects a backend, not a raw plotter).
            backend = self._backend
            scene = self._scene
            view = self._view
            last_step = self._last_step
            self.detach()
            self.attach(backend, view, scene)
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

    def _layer_id(self) -> str:
        return f"line_force_{id(self):x}"

    def _build_layer(self) -> MeshLayer:
        style: LineForceStyle = self.spec.style    # type: ignore[assignment]
        assert self._cells is not None and self._current_points is not None
        return MeshLayer(
            layer_id=self._layer_id(),
            points=PointSet(self._current_points),
            cells=self._cells,
            color=ColorSpec(mode="solid", solid_rgb=style.fill_color),
            opacity=style.opacity,
            show_edges=style.show_edges,
            edge_color=style.edge_color,
            # Decorative overlay — picks pass through to the substrate so
            # node/element/shift-click resolve to the real mesh, not the
            # fill quad in front of it.
            pickable=False,
        )

    def _push_update(self) -> None:
        if self._handle is None:
            return
        self._layer = self._build_layer()
        # Topology stable → backend mesh fast path replaces the bound
        # dataset's points in place (the property-setter path VTK reliably
        # re-reads), keeping the actor + dataset identity stable.
        self._backend.update_layer(self._handle, self._layer)

    def _reapply_last_step(self) -> None:
        # Re-fetch values for the current step. Cheap; one h5py read.
        self.update_to_step(self._last_step)

    def _compute_points(self, slab_values: ndarray) -> None:
        """Recompute the full ``(2N, 3)`` points for the new step values.

        Layout: ``[base_0..base_{N-1}, top_0..top_{N-1}]`` where each
        top point = ``base + scale·value·fill_dir``. Stored in
        ``self._current_points`` for the next layer emit.
        """
        if (
            self._base_points is None
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

        new_pts = np.empty((2 * self._n_stations, 3), dtype=np.float64)
        new_pts[: self._n_stations] = self._base_points
        new_pts[self._n_stations:] = top_points
        self._current_points = new_pts

    @staticmethod
    def _collect_line_element_ids(view: "ViewerData") -> ndarray:
        """Return all 1-D element IDs in the FEM."""
        ids: list[int] = []
        for group in view.elements:
            if group.element_type.dim == 1:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _collect_endpoints_with_subs(
        view: "ViewerData",
        scene: "FEMSceneData",
        element_ids: ndarray,
    ) -> "tuple[dict[int, tuple[ndarray, ndarray]], dict[int, tuple[int, int]]]":
        """``(eid -> (ci, cj), eid -> (i_sub, j_sub))`` — see
        :func:`~apeGmsh.viewers.diagrams._beam_geometry.collect_endpoints_with_substrate_rows`
        (shared with the fiber-section diagram)."""
        return collect_endpoints_with_substrate_rows(
            view, scene, element_ids,
        )
