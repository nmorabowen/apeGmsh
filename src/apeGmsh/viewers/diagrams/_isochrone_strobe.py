"""IsochroneStrobeDiagram — the deformed shape at several instants at once.

A strobe (motion-trail) view: instead of animating one deformed frame,
draw ``n_frames`` of them simultaneously as wireframes coloured by their
own time, so the *shape of the motion* is legible in a still image —
which is what you want in a report, and what scrubbing can never show.

Why this is a diagram and not several geometries: the geometry manager
deliberately shares ONE global step cursor across all geometries (ADR
0058 S3b rejected per-geometry cursors), so "the same stage at six
different steps" is not expressible as six geometries. A diagram that
owns its own frame set is.

All frames live in **one** :class:`MeshLayer`: the selected submesh is
replicated once per frame and each copy's points carry that frame's
time as a point scalar. One layer means one colour scale, one scalar
bar (reporting time), and a topology that never changes — so the
backend's in-place fast path applies and there is no per-frame actor
churn.

Deformation baseline: frames are built on the substrate points the
DEFORM pump hands the diagram, so a spatial offset on the owning
geometry carries the whole strobe with it. This diagram is meant for a
**deform-off** geometry — it *is* the deformation display. On a
deform-on geometry each frame stacks on top of that geometry's current
warp, double-counting the motion; the style docstring says so.

The frame field is read **once at attach** (one slab read per axis
covering all frames) and the per-frame warp is cached, so
``update_to_step`` does no work at all — the strobe is a whole-history
view, like the isochrone map.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec, NoDataError
from ._isochrone_math import pick_step_indices
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import IsochroneStrobeStyle
from ..scene_ir import (
    CellBlocks,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarField,
)

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


#: Emitted scalar array name (LUT / bar key) — frame time, not a response.
_FRAME_TIME_ARRAY = "frame_time"


@register_diagram_kind(
    label="Isochrone strobe (motion trail)",
    style_class=IsochroneStrobeStyle,
    order=16,
)
class IsochroneStrobeDiagram(ScalarColorSupport, Diagram):
    """Stacked deformed wireframes, one per sampled instant."""

    kind = "isochrone_strobe"
    topology = "nodes"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, IsochroneStrobeStyle):
            raise TypeError(
                "IsochroneStrobeDiagram requires an IsochroneStrobeStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._handle: Any = None
        self._layer: Optional[MeshLayer] = None
        self._points: Optional[PointSet] = None
        self._cells: Optional[CellBlocks] = None
        # Per-merged-point frame time (the painted scalar).
        self._scalar_values: Optional[ndarray] = None
        # Submesh-point -> substrate-row map (one frame's worth).
        self._substrate_rows: Optional[ndarray] = None
        # (F, P, 3) per-frame warp vectors as READ (unscaled), plus the
        # scaled copy the geometry is composed from. Keeping the raw
        # field is what makes ``set_scale`` reversible: deriving each
        # new scale from the previous *scaled* array would multiply by
        # ``new/old``, and a pass through zero would erase the field
        # with no way back.
        self._raw_offsets: Optional[ndarray] = None
        self._frame_offsets: Optional[ndarray] = None
        # The instants actually drawn.
        self._frame_steps: Optional[ndarray] = None
        self._frame_times: Optional[ndarray] = None
        self._scale_used: float = 1.0

        self._init_scalar_color_state()
        self._runtime_opacity: Optional[float] = None

    # ------------------------------------------------------------------
    # Attach / update / detach
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "IsochroneStrobeDiagram.attach requires a FEMSceneData "
                "(the viewer's substrate mesh). The Director must call "
                "bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, view, scene)
        style: IsochroneStrobeStyle = self.spec.style  # type: ignore[assignment]

        # ── Submesh for the selection (one frame's geometry) ─────────
        node_ids = self._resolved_node_ids
        if node_ids is None:
            point_indices = np.arange(scene.grid.n_points, dtype=np.int64)
        else:
            point_indices = self._fem_ids_to_substrate_indices(
                scene, node_ids,
            )
            if point_indices.size == 0:
                raise NoDataError(
                    f"IsochroneStrobeDiagram: selector resolved to "
                    f"{node_ids.size} node(s) but none of them are in "
                    f"the substrate mesh (selector={self.spec.selector!r})."
                )
        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0 or submesh.n_cells == 0:
            raise NoDataError(
                "IsochroneStrobeDiagram: substrate submesh has no cells "
                "for this selector — nothing to strobe."
            )

        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        fem_ids = scene.node_ids[orig_indices]
        n_points = int(submesh.n_points)

        # ── Which instants? ─────────────────────────────────────────
        steps, times = self._resolve_frames(int(fem_ids[0]))
        n_frames = int(steps.size)
        budget = int(style.max_points)
        if n_frames * n_points > budget:
            raise NoDataError(
                f"IsochroneStrobeDiagram: {n_frames} frames × "
                f"{n_points} points = {n_frames * n_points} replicated "
                f"points exceeds the {budget} budget. Lower n_frames, "
                f"restrict the diagram with a selector (physical group "
                f"/ label), or raise IsochroneStrobeStyle.max_points."
            )

        # ── Per-frame warp vectors ──────────────────────────────────
        raw = self._read_frame_field(fem_ids, steps)
        self._raw_offsets = raw
        self._scale_used = self._resolve_scale(raw, scene, style)
        self._frame_offsets = raw * self._scale_used
        self._frame_steps = steps
        self._frame_times = times

        self._substrate_rows = orig_indices
        self._cells = self._replicated_cells(submesh, n_frames, n_points)
        self._scalar_values = np.repeat(times, n_points).astype(np.float64)
        self._points = PointSet(
            self._compose_points(np.asarray(submesh.points, dtype=np.float64))
        )

        self._initial_clim = self._compute_initial_clim(times)
        self._layer = self._build_layer()
        self._handle = self._backend.add_layer(self._layer)

        self._init_lut()
        if self._handle is not None and self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle, self._make_scalar_bar_spec(),
            )

    def update_to_step(self, step_index: int) -> None:
        """No-op — the strobe shows its own fixed set of instants.

        The frames were chosen at attach and their warps cached; moving
        the time cursor doesn't change which instants are drawn. (The
        strobe still follows a substrate move via
        :meth:`sync_substrate_points`.)
        """
        return None

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Re-compose every frame against the (moved) substrate.

        The frames are ``base + scale·field(step_k)``; this re-samples
        ``base`` from the substrate through the cached
        ``vtkOriginalPointIds`` rows and re-adds the cached per-frame
        warps, so a rigid geometry offset carries the whole strobe.
        """
        if (
            self._handle is None
            or self._substrate_rows is None
            or self._frame_offsets is None
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
        rows = self._substrate_rows
        if rows.size == 0 or int(rows.max()) >= target.shape[0]:
            return
        self._points = PointSet(self._compose_points(target[rows]))
        self._push_update()

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._handle = None
        self._layer = None
        self._points = None
        self._cells = None
        self._scalar_values = None
        self._substrate_rows = None
        self._raw_offsets = None
        self._frame_offsets = None
        self._frame_steps = None
        self._frame_times = None
        self._initial_clim = None
        super().detach()

    # ------------------------------------------------------------------
    # Visibility / runtime style
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    def set_opacity(self, opacity: float) -> None:
        self._runtime_opacity = float(opacity)
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_opacity(self._handle, float(opacity))

    def set_scale(self, scale: float) -> None:
        """Re-warp every frame at a new amplification, live.

        Always recomputed from the raw (unscaled) field, so any sequence
        of scales is reachable — including going to ``0`` and back.
        """
        self._scale_used = float(scale)
        if self._raw_offsets is None:
            return
        self._frame_offsets = self._raw_offsets * self._scale_used
        scene = getattr(self, "_scene", None)
        if scene is not None:
            self.sync_substrate_points(None, scene)

    @property
    def scale_used(self) -> float:
        """The warp amplification in force (auto-fitted when unset)."""
        return self._scale_used

    @property
    def frame_times(self) -> "ndarray | None":
        """The instants actually drawn, or ``None`` before attach."""
        return self._frame_times

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._scalar_values

    # ------------------------------------------------------------------
    # Frame resolution + field read
    # ------------------------------------------------------------------

    def _resolve_frames(self, probe_id: int) -> tuple[ndarray, ndarray]:
        """``(step_indices, times)`` for the instants to draw."""
        if self._scoped_results() is None:
            raise NoDataError(
                "IsochroneStrobeDiagram: could not scope Results to a "
                "stage — the diagram needs a resolvable stage."
            )
        time = self._stage_time_vector(
            self.spec.selector.component, int(probe_id),
        )
        if time is None:
            raise NoDataError(
                f"IsochroneStrobeDiagram: could not read the stage's "
                f"time vector via component "
                f"{self.spec.selector.component!r}. Pick a component "
                f"this stage records."
            )
        if time.size < 2:
            raise NoDataError(
                f"IsochroneStrobeDiagram: a motion trail needs a "
                f"history — this stage has {time.size} step(s). Pick a "
                f"transient stage."
            )
        style: IsochroneStrobeStyle = self.spec.style  # type: ignore[assignment]
        steps = pick_step_indices(int(time.size), int(style.n_frames))
        return (steps, time[steps])

    def _read_frame_field(
        self, fem_ids: ndarray, steps: ndarray,
    ) -> ndarray:
        """``(F, P, 3)`` unscaled warp vectors for ``fem_ids`` at ``steps``.

        One slab read per axis covering every frame. Missing axes stay
        zero (2-D models record only x / y), matching the substrate
        deform pump's tolerance.
        """
        results = self._scoped_results()
        if results is None:
            raise NoDataError(
                "IsochroneStrobeDiagram: could not scope Results to a "
                "stage."
            )
        style: IsochroneStrobeStyle = self.spec.style  # type: ignore[assignment]
        field = str(style.field or "").strip()
        if not field:
            raise NoDataError(
                "IsochroneStrobeDiagram: IsochroneStrobeStyle.field is "
                "empty — set a nodal vector prefix (e.g. "
                "'displacement')."
            )

        n_frames = int(steps.size)
        n_points = int(fem_ids.size)
        out = np.zeros((n_frames, n_points, 3), dtype=np.float64)
        step_list = [int(s) for s in steps]
        any_axis = False
        for axis, suffix in enumerate(("x", "y", "z")):
            component = f"{field}_{suffix}"
            try:
                slab = results.nodes.get(
                    ids=fem_ids, component=component, time=step_list,
                )
            except Exception:
                continue
            values = np.asarray(slab.values, dtype=np.float64)
            if values.size == 0 or values.shape[0] != n_frames:
                continue
            # Scatter slab columns into fem_ids order — the reader may
            # return its own node ordering / subset.
            slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
            cols = self._align_columns(fem_ids, slab_ids)
            valid = cols >= 0
            if not valid.any():
                continue
            out[:, valid, axis] = values[:, cols[valid]]
            any_axis = True

        if not any_axis:
            raise NoDataError(
                f"IsochroneStrobeDiagram: no nodal data for any of "
                f"{field}_x / _y / _z in this stage. Pick a recorded "
                f"vector field (e.g. 'displacement')."
            )
        return out

    @staticmethod
    def _align_columns(fem_ids: ndarray, slab_ids: ndarray) -> ndarray:
        """For each ``fem_ids`` entry, its column in the slab (or -1)."""
        if slab_ids.size == 0:
            return np.full(fem_ids.size, -1, dtype=np.int64)
        max_id = int(max(int(fem_ids.max()), int(slab_ids.max()))) + 1
        col_of_id = np.full(max_id + 1, -1, dtype=np.int64)
        col_of_id[slab_ids] = np.arange(slab_ids.size, dtype=np.int64)
        safe = np.clip(fem_ids, 0, max_id)
        cols = col_of_id[safe]
        cols[(fem_ids < 0) | (fem_ids > max_id)] = -1
        return cols

    def _resolve_scale(
        self,
        raw: ndarray,
        scene: "FEMSceneData",
        style: IsochroneStrobeStyle,
    ) -> float:
        """Explicit ``style.scale``, else auto-fit off the model diagonal."""
        if style.scale is not None:
            return float(style.scale)
        finite = raw[np.isfinite(raw)]
        peak = float(np.abs(finite).max()) if finite.size else 0.0
        if peak <= 0.0:
            return 1.0
        diag = float(getattr(scene, "model_diagonal", 0.0)) or 1.0
        return float(style.auto_scale_fraction) * diag / peak

    # ------------------------------------------------------------------
    # Geometry assembly
    # ------------------------------------------------------------------

    def _compose_points(self, base: ndarray) -> ndarray:
        """``(F·P, 3)`` merged points: ``base + offset_f`` per frame."""
        assert self._frame_offsets is not None
        base = np.asarray(base, dtype=np.float64)
        return (base[None, :, :] + self._frame_offsets).reshape(-1, 3)

    @staticmethod
    def _replicated_cells(
        submesh: Any, n_frames: int, n_points: int,
    ) -> CellBlocks:
        """The submesh's connectivity, tiled once per frame with offsets."""
        from ..backends.pyvista_qt import cellblocks_from_grid
        base = cellblocks_from_grid(submesh)
        shifts = (
            np.arange(n_frames, dtype=np.int64) * int(n_points)
        ).reshape(-1, 1, 1)
        blocks: dict[str, ndarray] = {}
        for token, conn in base.blocks.items():
            conn = np.asarray(conn, dtype=np.int64)
            blocks[token] = (conn[None, :, :] + shifts).reshape(
                -1, conn.shape[1],
            )
        return CellBlocks(blocks)

    # ------------------------------------------------------------------
    # Layer build / emit
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"isochrone_strobe_{id(self):x}"

    def _color_array_name(self) -> str:
        return _FRAME_TIME_ARRAY

    def _scalar_bar_base_title(self) -> str:
        style: IsochroneStrobeStyle = self.spec.style  # type: ignore[assignment]
        return f"t ({style.field} strobe)"

    def _effective_opacity(self) -> float:
        style: IsochroneStrobeStyle = self.spec.style  # type: ignore[assignment]
        return (
            self._runtime_opacity
            if self._runtime_opacity is not None else style.opacity
        )

    def _build_layer(self) -> MeshLayer:
        style: IsochroneStrobeStyle = self.spec.style  # type: ignore[assignment]
        assert (
            self._points is not None
            and self._cells is not None
            and self._scalar_values is not None
        )
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        color = ColorSpec(
            mode="by_array",
            array_name=_FRAME_TIME_ARRAY,
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return MeshLayer(
            layer_id=self._layer_id(),
            points=self._points,
            cells=self._cells,
            fields=(
                ScalarField(_FRAME_TIME_ARRAY, self._scalar_values, "point"),
            ),
            color=color,
            opacity=self._effective_opacity(),
            wireframe=True,
            line_width=style.line_width,
            pickable=False,
        )

    def _push_update(self) -> None:
        if self._handle is None:
            return
        self._layer = self._build_layer()
        self._backend.update_layer(self._handle, self._layer)

    @staticmethod
    def _compute_initial_clim(times: ndarray) -> tuple[float, float]:
        """The colour range is always the drawn frames' time span.

        Unlike the value-painting diagrams there is no ``clim`` style
        knob: the scale's job is to separate the frames from each other,
        and the frames' own times are exactly that range.
        """
        finite = np.asarray(times, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if lo == hi:
                hi = lo + 1.0
            return (lo, hi)
        return (0.0, 1.0)

    @staticmethod
    def _fem_ids_to_substrate_indices(
        scene: "FEMSceneData", fem_ids: ndarray,
    ) -> ndarray:
        """Map a FEM-id array to substrate point indices, dropping misses."""
        if fem_ids.size == 0 or scene.node_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        max_id = max(int(fem_ids.max()), int(scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(
            scene.node_ids.size, dtype=np.int64,
        )
        idx = lookup[fem_ids]
        return idx[idx >= 0]
