"""IsochroneMapDiagram — paint *when*, not *how much*.

An isochrone map colours each node by the time at which its history
satisfies an arrival criterion: the first crossing of a threshold, or
the instant of its peak. Iso-lines of the resulting field are the
wavefronts — the reason to reach for this instead of a contour when the
question is "where has it got to by now" (wave propagation from a
moment-tensor / DRM source, first-yield onset, damage initiation).

Two consequences shape the implementation:

* **The scalar is a time.** The colour scale, the scalar bar, and the
  LUT mirror are all in the stage's time units, and the bar is titled
  after the time quantity rather than the tracked component (via the
  ``_scalar_bar_base_title`` hook).
* **The map is static.** It is a reduction over the *whole* history, so
  ``update_to_step`` is a deliberate no-op — scrubbing the time cursor
  moves the deformation and every other diagram, and leaves this one
  alone. That is the diagram's whole point; a per-step isochrone map
  would just be a contour.

The history is read **once at attach** to compute the ``(N,)`` arrival
array; the ``(T, N)`` slab is released immediately and never cached, so
the standing per-step performance contract is untouched (there are no
per-step reads at all).

In ``"first_crossing"`` mode, nodes whose history never reaches the
threshold are **excluded from the painted submesh** rather than painted
with a sentinel colour — leaving a hole is the honest rendering of "the
front has not arrived here". Note this diagram sets
``occludes_substrate`` (like the contour), so the viewer hides the grey
substrate *fill* while it is visible: the un-arrived region reads as
substrate **wireframe**, not as grey shading. If no selected node ever
crosses, attach raises :class:`NoDataError` naming the threshold that
was applied.

Render seam (ADR 0042): emits one substrate-submesh :class:`MeshLayer`
with point-located scalars through ``self._backend``; the diagram holds
no VTK objects. Mirrors the contour's nodes path, including the cached
``vtkOriginalPointIds`` rows that let ``sync_substrate_points``
re-sample a deformed substrate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec, NoDataError
from ._isochrone_math import (
    MODE_FIRST_CROSSING,
    MODE_TIME_TO_PEAK,
    ARRIVAL_MODES,
    arrival_times,
)
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import IsochroneMapStyle
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


#: Name of the emitted scalar array (and the LUT / bar key). Deliberately
#: not the component: the painted values are times.
_ARRIVAL_ARRAY = "arrival_time"


@register_diagram_kind(
    label="Isochrone map (arrival time)",
    style_class=IsochroneMapStyle,
    order=12,
)
class IsochroneMapDiagram(ScalarColorSupport, Diagram):
    """Per-node arrival time painted on a slice of the substrate."""

    kind = "isochrone_map"
    topology = "nodes"
    # Same reasoning as the contour: an opaque filled surface coincident
    # with the grey substrate fill, so the viewer hides that fill while
    # this layer is visible to avoid z-fighting.
    occludes_substrate = True

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, IsochroneMapStyle):
            raise TypeError(
                "IsochroneMapDiagram requires an IsochroneMapStyle; "
                f"got {type(spec.style).__name__}."
            )
        mode = getattr(spec.style, "mode", MODE_FIRST_CROSSING)
        if mode not in ARRIVAL_MODES:
            raise ValueError(
                f"IsochroneMapStyle.mode must be one of {ARRIVAL_MODES}; "
                f"got {mode!r}."
            )
        super().__init__(spec, results)

        self._handle: Any = None
        self._layer: Optional[MeshLayer] = None
        self._points: Optional[PointSet] = None
        self._cells: Optional[CellBlocks] = None
        # Per-submesh-point arrival times (the painted scalar).
        self._scalar_values: Optional[ndarray] = None
        # Submesh-point -> substrate-row map for deformation follow.
        self._substrate_rows: Optional[ndarray] = None
        # The threshold actually applied (nan in time_to_peak mode) —
        # surfaced by ``describe_criterion`` for the settings panel.
        self._threshold_used: float = float("nan")
        # Node counts for the criterion readout: how many of the
        # selected nodes ever arrived.
        self._n_selected: int = 0
        self._n_arrived: int = 0

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
                "IsochroneMapDiagram.attach requires a FEMSceneData (the "
                "viewer's substrate mesh). The Director must call "
                "bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, view, scene)
        style: IsochroneMapStyle = self.spec.style    # type: ignore[assignment]

        node_ids, arrival = self._compute_arrival()

        # Drop the never-arrived nodes; the grey substrate showing
        # through is the rendering of "not yet".
        finite = np.isfinite(arrival)
        self._n_selected = int(arrival.size)
        self._n_arrived = int(finite.sum())
        if not finite.any():
            raise NoDataError(
                f"IsochroneMapDiagram: no node in the selection ever "
                f"reached the arrival criterion "
                f"({self._describe_criterion_short()}). Lower the "
                f"threshold, or switch mode to 'time_to_peak' which is "
                f"always defined."
            )
        node_ids = node_ids[finite]
        arrival = arrival[finite]

        point_indices = self._fem_ids_to_substrate_indices(scene, node_ids)
        if point_indices.size == 0:
            raise NoDataError(
                f"IsochroneMapDiagram: {node_ids.size} node(s) satisfied "
                f"the arrival criterion but none of them are in the "
                f"substrate mesh (selector={self.spec.selector!r})."
            )

        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0:
            raise NoDataError(
                "IsochroneMapDiagram: substrate submesh is empty for "
                "this selector — nothing to color."
            )

        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        fem_ids_in_submesh = scene.node_ids[orig_indices]

        # FEM id -> arrival lookup, then scatter into submesh order.
        max_id = int(fem_ids_in_submesh.max()) + 1
        arrival_of_id = np.full(max_id + 1, np.nan, dtype=np.float64)
        in_range = node_ids <= max_id
        arrival_of_id[node_ids[in_range]] = arrival[in_range]

        self._points = PointSet(np.asarray(submesh.points))
        self._cells = self._cellblocks(submesh)
        self._scalar_values = arrival_of_id[fem_ids_in_submesh]
        self._substrate_rows = orig_indices

        self._initial_clim = self._compute_initial_clim(
            self._scalar_values, style,
        )
        self._layer = self._build_layer()
        self._handle = self._backend.add_layer(self._layer)

        self._init_lut()
        if self._handle is not None and self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle, self._make_scalar_bar_spec(),
            )

    def update_to_step(self, step_index: int) -> None:
        """No-op — an isochrone map is a whole-history reduction.

        The painted field answers "at what time did each node arrive",
        which does not depend on where the cursor sits. Scrubbing still
        moves the substrate (and hence this layer, via
        :meth:`sync_substrate_points`); only the colours are frozen.
        """
        return None

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Re-sample the submesh points from the (deformed) substrate.

        Identical to the contour's hook: the emitted submesh is a COPY,
        so without this the arrival map stays at the reference
        configuration while the substrate warps.
        """
        if (
            self._handle is None
            or self._points is None
            or self._substrate_rows is None
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
        self._points = PointSet(target[rows])
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

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._scalar_values

    # ------------------------------------------------------------------
    # Criterion readout (settings panel / status)
    # ------------------------------------------------------------------

    @property
    def threshold_used(self) -> float:
        """The arrival level actually applied (``nan`` for time-to-peak).

        With ``IsochroneMapStyle.threshold=None`` the level is derived
        from the data, so the user needs a way to see what it came out
        as — otherwise the map is uninterpretable.
        """
        return self._threshold_used

    def describe_criterion(self) -> str:
        """One-line human summary of the criterion and its coverage."""
        base = self._describe_criterion_short()
        if self._n_selected:
            return (
                f"{base} — {self._n_arrived}/{self._n_selected} nodes "
                f"arrived"
            )
        return base

    def _describe_criterion_short(self) -> str:
        style: IsochroneMapStyle = self.spec.style    # type: ignore[assignment]
        component = self.spec.selector.component
        if style.mode == MODE_TIME_TO_PEAK:
            what = f"|{component}|" if style.use_abs else component
            return f"time of peak {what}"
        what = f"|{component}|" if style.use_abs else component
        level = self._threshold_used
        if np.isfinite(level):
            derived = "" if style.threshold is not None else (
                f", auto = {style.threshold_fraction:g}×peak"
            )
            return f"first {what} ≥ {level:.6g}{derived}"
        return f"first {what} ≥ (threshold pending)"

    # ------------------------------------------------------------------
    # History reduction
    # ------------------------------------------------------------------

    def _compute_arrival(self) -> tuple[ndarray, ndarray]:
        """Read the whole history once and reduce it to ``(ids, times)``.

        The ``(T, N)`` slab is local to this call — the arrival array is
        the only thing retained, so the diagram never holds a full-time
        cache (the standing perf contract).
        """
        results = self._scoped_results()
        if results is None:
            raise NoDataError(
                "IsochroneMapDiagram: could not scope Results to a "
                "stage — the diagram needs a resolvable stage."
            )
        component = self.spec.selector.component
        ids = self._resolved_node_ids
        style: IsochroneMapStyle = self.spec.style    # type: ignore[assignment]
        self._check_history_budget(component, ids, style)
        try:
            slab = results.nodes.get(
                ids=ids, component=component, time=None,
            )
        except Exception as exc:
            raise NoDataError(
                f"IsochroneMapDiagram: could not read the history of "
                f"{component!r}: {exc}"
            ) from exc
        values = np.asarray(slab.values, dtype=np.float64)
        if values.size == 0:
            raise NoDataError(
                f"IsochroneMapDiagram: no nodal data for component "
                f"{component!r} in this stage. Use "
                f"`results.inspect.diagnose({component!r})` to see "
                f"which buckets were checked."
            )
        if values.shape[0] < 2:
            raise NoDataError(
                f"IsochroneMapDiagram: an arrival time needs a history — "
                f"this stage has {values.shape[0]} step(s). Pick a "
                f"transient stage."
            )

        try:
            arrival, level = arrival_times(
                values,
                np.asarray(slab.time, dtype=np.float64),
                mode=style.mode,
                threshold=style.threshold,
                threshold_fraction=style.threshold_fraction,
                use_abs=style.use_abs,
                interpolate=style.interpolate,
            )
        except ValueError as exc:
            raise NoDataError(f"IsochroneMapDiagram: {exc}") from exc
        self._threshold_used = float(level)
        return (np.asarray(slab.node_ids, dtype=np.int64), arrival)

    def _check_history_budget(
        self,
        component: str,
        ids: "ndarray | None",
        style: IsochroneMapStyle,
    ) -> None:
        """Refuse a whole-history read that would not fit in memory.

        The reduction needs every step of every selected node, so the
        read is inherently ``(T, N)``. Sized in advance from a one-node
        probe (for ``T``) and the resolved selection (for ``N``) — both
        cheap — so an over-budget request fails with an actionable
        message instead of the viewer dying inside the h5 read.

        Silently unknown sizes (no probe, no bound view) skip the check
        rather than guess: a wrong refusal is worse than the read.
        """
        budget = int(getattr(style, "max_history_samples", 0) or 0)
        if budget <= 0:
            return
        n_nodes = self._selected_node_count(ids)
        if n_nodes <= 0:
            return
        probe = self._probe_node_id(ids)
        if probe is None:
            return
        time = self._stage_time_vector(component, probe)
        if time is None:
            return
        samples = int(time.size) * int(n_nodes)
        if samples <= budget:
            return
        gib = samples * 8 / 1024 ** 3
        raise NoDataError(
            f"IsochroneMapDiagram: an arrival map reduces the WHOLE "
            f"history, so it would read {time.size} steps × {n_nodes} "
            f"nodes = {samples:,} samples (~{gib:.1f} GiB) — over the "
            f"{budget:,}-sample budget. Restrict the diagram with a "
            f"selector (physical group / label), or raise "
            f"IsochroneMapStyle.max_history_samples if you have the RAM."
        )

    def _selected_node_count(self, ids: "ndarray | None") -> int:
        """How many nodes the read will cover (``ids=None`` = all)."""
        if ids is not None:
            return int(ids.size)
        scene = getattr(self, "_scene", None)
        node_ids = getattr(scene, "node_ids", None)
        if node_ids is not None:
            return int(np.asarray(node_ids).size)
        try:
            return int(np.asarray(self._view.nodes.ids).size)  # type: ignore[union-attr]
        except Exception:
            return 0

    def _probe_node_id(self, ids: "ndarray | None") -> Optional[int]:
        """One node id the component is recorded at, for the time probe."""
        if ids is not None and ids.size:
            return int(ids[0])
        scene = getattr(self, "_scene", None)
        node_ids = getattr(scene, "node_ids", None)
        if node_ids is not None and np.asarray(node_ids).size:
            return int(np.asarray(node_ids)[0])
        try:
            arr = np.asarray(self._view.nodes.ids)  # type: ignore[union-attr]
            return int(arr[0]) if arr.size else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Layer build / emit
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"isochrone_map_{id(self):x}"

    def _color_array_name(self) -> str:
        return _ARRIVAL_ARRAY

    def _scalar_bar_base_title(self) -> str:
        style: IsochroneMapStyle = self.spec.style    # type: ignore[assignment]
        if style.mode == MODE_TIME_TO_PEAK:
            return f"t_peak ({self.spec.selector.component})"
        return f"t_arrival ({self.spec.selector.component})"

    def _effective_opacity(self) -> float:
        style: IsochroneMapStyle = self.spec.style    # type: ignore[assignment]
        return (
            self._runtime_opacity
            if self._runtime_opacity is not None else style.opacity
        )

    def _build_layer(self) -> MeshLayer:
        style: IsochroneMapStyle = self.spec.style    # type: ignore[assignment]
        assert (
            self._points is not None
            and self._cells is not None
            and self._scalar_values is not None
        )
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        color = ColorSpec(
            mode="by_array",
            array_name=_ARRIVAL_ARRAY,
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return MeshLayer(
            layer_id=self._layer_id(),
            points=self._points,
            cells=self._cells,
            fields=(
                ScalarField(_ARRIVAL_ARRAY, self._scalar_values, "point"),
            ),
            color=color,
            opacity=self._effective_opacity(),
            show_edges=style.show_edges,
            pickable=False,
        )

    def _push_update(self) -> None:
        if self._handle is None:
            return
        self._layer = self._build_layer()
        self._backend.update_layer(self._handle, self._layer)

    @staticmethod
    def _compute_initial_clim(
        data: ndarray, style: IsochroneMapStyle,
    ) -> tuple[float, float]:
        if style.clim is not None:
            return (float(style.clim[0]), float(style.clim[1]))
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if lo == hi:
                hi = lo + 1.0
            return (lo, hi)
        return (0.0, 1.0)

    @staticmethod
    def _cellblocks(submesh: Any) -> CellBlocks:
        from ..backends.pyvista_qt import cellblocks_from_grid
        return cellblocks_from_grid(submesh)

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
