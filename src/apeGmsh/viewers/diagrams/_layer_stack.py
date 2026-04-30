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

Performance: aggregation is done in numpy via grouped reduction. At
attach we precompute per-cell row indices into the layer slab and the
sub-GP weights for the chosen aggregation; per-step update is a
dot/argmax over the precomputed groups + an in-place scalar mutation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import LayerStackStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_SCALAR_NAME = "_layer_value"
_AGGREGATIONS = ("mid_layer", "mean", "max_abs")


class LayerStackDiagram(Diagram):
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

        self._submesh: Optional[pv.UnstructuredGrid] = None
        self._actor: Any = None
        self._scalar_array: Optional[ndarray] = None
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
                "LayerStackDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, fem, scene)
        style: LayerStackStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (shell elements only) ───────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_shell_element_ids(fem)
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
        submesh = scene.grid.extract_cells(cell_indices)
        if submesh.n_cells == 0:
            return

        # vtkOriginalCellIds maps submesh cell index -> substrate cell index
        # which we can map back to FEM element_id via scene.cell_to_element_id.
        orig_cell_ids = np.asarray(
            submesh.cell_data["vtkOriginalCellIds"], dtype=np.int64,
        )
        cell_eids = scene.cell_to_element_id[orig_cell_ids]

        # Per-cell aggregation rows from the layer slab
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
            cell_to_mid = np.full(submesh.n_cells, -1, dtype=np.int64)
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
        scalars = np.zeros(submesh.n_cells, dtype=np.float64)
        submesh.cell_data[_SCALAR_NAME] = scalars
        self._scalar_array = submesh.cell_data[_SCALAR_NAME]

        slab_values_step0 = np.asarray(slab.values[0], dtype=np.float64)
        self._aggregate_into_scalars(slab_values_step0)
        self._submesh = submesh

        # ── Initial clim ────────────────────────────────────────────
        if style.clim is not None:
            self._initial_clim = (
                float(style.clim[0]), float(style.clim[1]),
            )
        else:
            data = np.asarray(self._scalar_array)
            finite = data[np.isfinite(data)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        actor = plotter.add_mesh(
            submesh,
            scalars=_SCALAR_NAME,
            cmap=self._runtime_cmap or style.cmap,
            clim=self._runtime_clim or self._initial_clim,
            opacity=style.opacity,
            show_edges=style.show_edges,
            show_scalar_bar=style.show_scalar_bar,
            scalar_bar_args={
                "title": self.spec.selector.component,
            } if style.show_scalar_bar else None,
            name=self._actor_name(),
            reset_camera=False,
            lighting=True,
            smooth_shading=False,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if self._submesh is None or self._scalar_array is None:
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

    def detach(self) -> None:
        self._submesh = None
        self._actor = None
        self._scalar_array = None
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
    # Internal — aggregation
    # ------------------------------------------------------------------

    def _aggregate_into_scalars(self, slab_values: ndarray) -> None:
        if self._scalar_array is None:
            return
        style: LayerStackStyle = self.spec.style    # type: ignore[assignment]
        out = np.asarray(self._scalar_array)

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
        try:
            self._submesh.Modified()    # type: ignore[union-attr]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal — geometry helpers
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_layer_{id(self):x}"

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _collect_shell_element_ids(fem: "FEMData") -> ndarray:
        """All 2-D element IDs (shells / plates / membranes)."""
        ids: list[int] = []
        for group in fem.elements:
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
