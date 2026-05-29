"""LayerStackDiagram — shell mid-surface contour + through-thickness panel.

3-D rendering: per shell-cell scalar value, aggregated from the
``LayerSlab`` rows that share ``element_id`` (and the cell's GP).
Three aggregations are supported via ``LayerStackStyle.aggregation``:

* ``"mid_layer"`` — the sub-GP nearest the mid-thickness coordinate
* ``"mean"``      — average over all layers and sub-GPs at the GP
* ``"max_abs"``   — signed value of largest magnitude

2-D side panel: at the picked ``(element_id, gp_index)``, plot the
component value vs cumulative thickness coordinate (one point per
layer × sub-GP).

Render seam (ADR 0042, R-B Wave 2 #3). The mid-surface contour is
emitted as a substrate-submesh :class:`MeshLayer` carrying a per-cell
:class:`ScalarField` through the backend; the diagram holds no VTK
objects. The submesh is still extracted via the handed scene grid
(``scene.grid.extract_cells`` — a method call, no pyvista import) and
re-expressed as neutral IR via ``cellblocks_from_grid``; the matplotlib
through-thickness panel stays OUT of the IR (own ``make_side_panel``).

Performance: aggregation is done in numpy via grouped reduction. At
attach we precompute per-cell row indices into the layer slab and the
sub-GP weights for the chosen aggregation; per-step update is a
dot/argmax over the precomputed groups, then a backend ``update_layer``
whose in-place mesh fast path recolours the bound dataset (topology
unchanged) — actor + dataset identity stable across steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._scalar_bar_support import ScalarBarSupport
from ._styles import LayerStackStyle
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


_AGGREGATIONS = ("mid_layer", "mean", "max_abs")


class LayerStackDiagram(ScalarBarSupport, Diagram):
    """Shell mid-surface contour + through-thickness side panel."""

    kind = "layer_stack"
    topology = "layers"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, LayerStackStyle):
            raise TypeError(
                "LayerStackDiagram requires a LayerStackStyle; "
                f"got {type(spec.style).__name__}."
            )
        if spec.style.aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"LayerStackStyle.aggregation must be one of "
                f"{_AGGREGATIONS}; got {spec.style.aggregation!r}."
            )
        super().__init__(spec, results)

        self._layer: Optional[MeshLayer] = None
        self._handle: Any = None
        self._points: Optional[PointSet] = None     # cached submesh points
        self._cells: Optional[CellBlocks] = None     # cached cell blocks
        self._cell_values: Optional[ndarray] = None  # per-cell scalar (grouped)
        self._n_cells: int = 0
        self._element_ids_to_read: tuple[int, ...] = ()

        # Per-layer-row metadata (locked at attach)
        self._slab_eid: Optional[ndarray] = None
        self._slab_gp: Optional[ndarray] = None
        self._slab_layer: Optional[ndarray] = None
        self._slab_sub_gp: Optional[ndarray] = None
        self._slab_thickness: Optional[ndarray] = None

        # Per-cell aggregation map: cell_idx -> indices into layer slab
        self._cell_to_slab_rows: dict[int, ndarray] = {}
        # For mid-layer aggregation, per-cell single row index
        self._cell_to_mid_row: Optional[ndarray] = None

        # (eid, gp) -> sorted slab-row indices (for thickness panel)
        self._gp_to_indices: dict[tuple[int, int], ndarray] = {}

        self._initial_clim: Optional[tuple[float, float]] = None
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_cmap: Optional[str] = None
        self._init_scalar_bar_state()

        # Plan 06 — LUT mirror built at the tail of attach().
        self._lut: Any = None
        self._lut_conn: Any = None

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
                "LayerStackDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, view, scene)
        style: LayerStackStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (shell elements only) ───────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_shell_element_ids(view)
        if element_ids.size == 0:
            return
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Step-0 read to discover layer rows ──────────────────────
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.layers.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[0],
            )
        except Exception as exc:
            raise RuntimeError(
                f"LayerStackDiagram could not read layers slab: {exc}"
            )
        if slab.values.size == 0:
            return

        slab_eid = np.asarray(slab.element_index, dtype=np.int64)
        slab_gp = np.asarray(slab.gp_index, dtype=np.int64)
        slab_layer = np.asarray(slab.layer_index, dtype=np.int64)
        slab_sub_gp = np.asarray(slab.sub_gp_index, dtype=np.int64)
        slab_thickness = np.asarray(slab.thickness, dtype=np.float64)

        self._slab_eid = slab_eid
        self._slab_gp = slab_gp
        self._slab_layer = slab_layer
        self._slab_sub_gp = slab_sub_gp
        self._slab_thickness = slab_thickness

        # ── Build (eid, gp) -> indices map for the panel ────────────
        keys = list(zip(slab_eid.tolist(), slab_gp.tolist()))
        gp_map: dict[tuple[int, int], list[int]] = {}
        for idx, key in enumerate(keys):
            gp_map.setdefault(key, []).append(idx)
        self._gp_to_indices = {
            k: np.asarray(v, dtype=np.int64) for k, v in gp_map.items()
        }

        # ── Extract substrate sub-mesh for selected shell elements ──
        present_eids = np.unique(slab_eid)
        cell_indices = self._element_ids_to_substrate_cells(
            scene, present_eids,
        )
        if cell_indices.size == 0:
            return
        # Submesh extraction is via the handed scene grid (transitional;
        # a method call, not a pyvista import) and re-expressed as
        # neutral IR through cellblocks_from_grid below.
        submesh = scene.grid.extract_cells(cell_indices)
        if submesh.n_cells == 0:
            return

        # vtkOriginalCellIds maps submesh cell index -> substrate cell index
        # which we can map back to FEM element_id via scene.cell_to_element_id.
        orig_cell_ids = np.asarray(
            submesh.cell_data["vtkOriginalCellIds"], dtype=np.int64,
        )
        cell_eids = scene.cell_to_element_id[orig_cell_ids]

        # ``CellBlocks`` (and the grid the backend rebuilds from it)
        # groups cells by VTK type, so for a mixed tri+quad shell mesh the
        # rebuilt cell order differs from the extracted submesh order.
        # Reorder the per-cell metadata into that grouped order ONCE so
        # the per-cell ScalarField stays aligned with the CellBlocks.
        # ``cellblocks_from_grid`` iterates ``submesh.cells_dict.items()``;
        # we mirror that iteration here (homogeneous meshes → identity).
        from ..backends.pyvista_qt import cellblocks_from_grid

        celltypes = np.asarray(submesh.celltypes, dtype=np.int64)
        group_to_orig = np.concatenate(
            [np.where(celltypes == t)[0] for t in submesh.cells_dict]
        ).astype(np.int64) if submesh.n_cells else np.empty(0, np.int64)
        cell_eids = cell_eids[group_to_orig]

        self._cells = cellblocks_from_grid(submesh)
        self._points = PointSet(np.asarray(submesh.points))
        self._n_cells = int(submesh.n_cells)

        # Per-cell aggregation rows from the layer slab (grouped order)
        cell_to_rows: dict[int, ndarray] = {}
        for cell_idx, eid in enumerate(cell_eids):
            mask = slab_eid == int(eid)
            if mask.any():
                cell_to_rows[cell_idx] = np.where(mask)[0]
        self._cell_to_slab_rows = cell_to_rows

        # Pre-pick mid-layer row index per cell for "mid_layer"
        # aggregation. Mid layer = the sub-GP closest to the
        # cumulative thickness midpoint.
        if style.aggregation == "mid_layer":
            cell_to_mid = np.full(self._n_cells, -1, dtype=np.int64)
            for cell_idx, rows in cell_to_rows.items():
                t = slab_thickness[rows]
                cum = np.cumsum(t)
                if cum[-1] > 0:
                    target = cum[-1] / 2.0
                    mid_local = int(np.argmin(np.abs(cum - target)))
                else:
                    mid_local = len(rows) // 2
                cell_to_mid[cell_idx] = rows[mid_local]
            self._cell_to_mid_row = cell_to_mid

        # ── Initial scalar array filled from step 0 ─────────────────
        self._cell_values = np.zeros(self._n_cells, dtype=np.float64)
        slab_values_step0 = np.asarray(slab.values[0], dtype=np.float64)
        self._aggregate_into_scalars(slab_values_step0)

        # ── Initial clim ────────────────────────────────────────────
        if style.clim is not None:
            self._initial_clim = (
                float(style.clim[0]), float(style.clim[1]),
            )
        else:
            finite = self._cell_values[np.isfinite(self._cell_values)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        self._layer = self._build_layer(self._cell_values)
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
            or self._cell_values is None
        ):
            return
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.layers.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return
        if slab.values.size == 0:
            return
        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if self._slab_eid is None or slab_values.size != self._slab_eid.size:
            return
        self._aggregate_into_scalars(slab_values)
        self._layer = self._build_layer(self._cell_values)
        # Topology unchanged across steps → the backend's in-place mesh
        # fast path recolours the bound dataset without re-adding the actor.
        self._backend.update_layer(self._handle, self._layer)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        if self._lut is not None and self._lut_conn is not None:
            try:
                self._lut.changed.disconnect(self._lut_conn)
            except (TypeError, RuntimeError):
                pass
        self._lut = None
        self._lut_conn = None
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._points = None
        self._cells = None
        self._cell_values = None
        self._n_cells = 0
        self._element_ids_to_read = ()
        self._slab_eid = None
        self._slab_gp = None
        self._slab_layer = None
        self._slab_sub_gp = None
        self._slab_thickness = None
        self._cell_to_slab_rows = {}
        self._cell_to_mid_row = None
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
            from ..ui._thickness_panel import LayerThicknessPanel
        except ImportError:
            return None
        return LayerThicknessPanel(self, director)

    # ------------------------------------------------------------------
    # Side-panel helpers
    # ------------------------------------------------------------------

    def available_gps(self) -> list[tuple[int, int]]:
        return sorted(self._gp_to_indices.keys())

    def read_thickness_profile(
        self, element_id: int, gp_index: int, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray]]:
        """Return ``(thickness_coord, values)`` sorted bottom -> top."""
        if (
            self._slab_layer is None or self._slab_sub_gp is None
            or self._slab_thickness is None
        ):
            return None
        key = (int(element_id), int(gp_index))
        idxs = self._gp_to_indices.get(key)
        if idxs is None or idxs.size == 0:
            return None

        # Read step values for this slab
        results = self._scoped_results()
        if results is None:
            return None
        try:
            slab = results.elements.layers.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return None
        if slab.values.size == 0:
            return None

        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if slab_values.size != self._slab_layer.size:
            return None

        # Sort by (layer_index, sub_gp_index) for a clean bottom->top profile
        layers = self._slab_layer[idxs]
        sub_gps = self._slab_sub_gp[idxs]
        thickness = self._slab_thickness[idxs]
        values = slab_values[idxs]
        order = np.lexsort((sub_gps, layers))

        ordered_thickness = thickness[order]
        ordered_values = values[order]
        # Use cumulative thickness as the through-thickness coordinate,
        # placing each sample at its layer's mid-point.
        midpoints = np.cumsum(ordered_thickness) - 0.5 * ordered_thickness
        return midpoints, ordered_values

    # ------------------------------------------------------------------
    # Runtime style adjustments
    # ------------------------------------------------------------------

    @property
    def lut(self) -> Any:
        """Shared lookup-table mirror; ``None`` outside attach."""
        return self._lut

    def set_clim(self, vmin: float, vmax: float) -> None:
        if vmin == vmax:
            vmax = vmin + 1.0
        if self._lut is not None:
            self._lut.set_range(float(vmin), float(vmax))
            return
        self._runtime_clim = (float(vmin), float(vmax))

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        if self._cell_values is None:
            return None
        finite = self._cell_values[np.isfinite(self._cell_values)]
        if finite.size == 0:
            return None
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    def set_cmap(self, cmap: str) -> None:
        self._runtime_cmap = cmap
        if self._lut is not None:
            self._lut.set_preset(cmap)

    def current_clim(self) -> Optional[tuple[float, float]]:
        return self._runtime_clim or self._initial_clim

    # ------------------------------------------------------------------
    # LUT mirror (diagram-side; changes pushed through the backend)
    # ------------------------------------------------------------------

    def _init_lut(self) -> None:
        from ..core._lut_manager import LUT

        style: LayerStackStyle = self.spec.style    # type: ignore[assignment]
        preset = self._runtime_cmap or style.cmap or "viridis"
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        try:
            self._lut = LUT(
                array_name=self.spec.selector.component,
                preset=preset,
                vmin=float(clim[0]),
                vmax=float(clim[1]),
                show_scalar_bar=self._effective_show_scalar_bar(),
            )
            self._lut_conn = self._lut.changed.connect(self._on_lut_changed)
        except Exception:
            self._lut = None
            self._lut_conn = None

    def _on_lut_changed(self) -> None:
        if self._lut is None or self._handle is None or self._backend is None:
            return
        self._runtime_cmap = self._lut.preset
        self._runtime_clim = (self._lut.vmin, self._lut.vmax)
        color = ColorSpec(
            mode="by_array",
            array_name=self._color_array_name(),
            lut=self._current_lutspec(),
        )
        self._backend.set_layer_color(self._handle, color)
        if self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle,
                ScalarBarSpec(
                    layer_id=self._handle.layer_id,
                    title=self._scalar_bar_title(),
                    lut=self._current_lutspec(),
                ),
            )

    # ------------------------------------------------------------------
    # Internal — aggregation
    # ------------------------------------------------------------------

    def _aggregate_into_scalars(self, slab_values: ndarray) -> None:
        if self._cell_values is None:
            return
        style: LayerStackStyle = self.spec.style    # type: ignore[assignment]
        out = self._cell_values

        if style.aggregation == "mid_layer":
            assert self._cell_to_mid_row is not None
            mid = self._cell_to_mid_row
            valid = mid >= 0
            out[valid] = slab_values[mid[valid]]
            out[~valid] = 0.0
        elif style.aggregation == "mean":
            for cell_idx, rows in self._cell_to_slab_rows.items():
                vals = slab_values[rows]
                out[cell_idx] = float(vals.mean()) if vals.size else 0.0
        elif style.aggregation == "max_abs":
            for cell_idx, rows in self._cell_to_slab_rows.items():
                vals = slab_values[rows]
                if vals.size == 0:
                    out[cell_idx] = 0.0
                else:
                    j = int(np.argmax(np.abs(vals)))
                    out[cell_idx] = float(vals[j])

    # ------------------------------------------------------------------
    # Internal — layer build + geometry helpers
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"layer_{id(self):x}"

    def _color_array_name(self) -> str:
        return self.spec.selector.component or "_layer_value"

    def _build_layer(self, cell_values: ndarray) -> MeshLayer:
        """Submesh MeshLayer with a per-cell ScalarField; decorative
        (pickable=False) so picks pass through to the substrate."""
        style: LayerStackStyle = self.spec.style    # type: ignore[assignment]
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
            fields=(ScalarField(name, cell_values, "cell"),),
            color=color,
            opacity=style.opacity,
            show_edges=style.show_edges,
            pickable=False,
        )

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _collect_shell_element_ids(view: "ViewerData") -> ndarray:
        """All 2-D element IDs (shells / plates / membranes)."""
        ids: list[int] = []
        for group in view.elements:
            if group.element_type.dim == 2:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _element_ids_to_substrate_cells(
        scene: "FEMSceneData", element_ids: ndarray,
    ) -> ndarray:
        """Map FEM element IDs to substrate cell indices (drop misses)."""
        out: list[int] = []
        for eid in element_ids:
            cell_idx = scene.element_id_to_cell.get(int(eid))
            if cell_idx is not None:
                out.append(int(cell_idx))
        return np.asarray(out, dtype=np.int64)
