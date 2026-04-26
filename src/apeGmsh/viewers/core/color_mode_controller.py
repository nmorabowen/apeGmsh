"""
ColorModeController — Drives mesh-viewer color modes.

Two coloring strategies coexist:

* **Idle-fn strategy** (Default / Element Type / Physical Group)
  Swaps ``ColorManager``'s idle-color function. Hover/pick/hidden state
  layers correctly on top, so mode switches preserve interaction.

* **Scalar-mapper strategy** (Quality)
  Mutates each per-dim actor's VTK mapper to display a per-cell scalar
  with a lookup table + colorbar. Hover/pick writes still go to
  ``cell_data["colors"]`` but are visually masked while Quality is
  active; switching back to an idle-fn mode triggers a full repaint
  so no stale state leaks.

Supported modes
---------------
- ``Default``        — per-dim theme colors (ColorManager's built-in default)
- ``Element Type``   — color per entity based on dominant element type
- ``Physical Group`` — color per entity based on physical group membership
- ``Quality``        — per-cell scalar from gmsh.model.mesh.getElementQualities
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..scene.mesh_scene import MeshSceneData
    from .color_manager import ColorManager
    from .entity_registry import EntityRegistry
    from .selection import SelectionState
    from .visibility import VisibilityManager


_FALLBACK_RGB = np.array([136, 136, 136], dtype=np.uint8)


# Element-type category -> RGB. Matches ELEM_TYPE_COLORS in scene/mesh_scene.py.
_ELEM_TYPE_RGB: dict[str, np.ndarray] = {
    "Triangle":      np.array([67, 99, 216], dtype=np.uint8),
    "Quadrilateral": np.array([60, 180, 75], dtype=np.uint8),
    "Quad":          np.array([60, 180, 75], dtype=np.uint8),
    "Tetrahedron":   np.array([230, 25, 75], dtype=np.uint8),
    "Hexahedron":    np.array([245, 130, 49], dtype=np.uint8),
    "Prism":         np.array([145, 30, 180], dtype=np.uint8),
    "Pyramid":       np.array([66, 212, 244], dtype=np.uint8),
    "Line":          np.array([170, 170, 170], dtype=np.uint8),
    "Point":         np.array([255, 255, 255], dtype=np.uint8),
}


_GROUP_PALETTE_HEX = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
)


def _hex_to_rgb(s: str) -> np.ndarray:
    s = s.lstrip("#")
    return np.array(
        [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)],
        dtype=np.uint8,
    )


_GROUP_PALETTE_RGB: list[np.ndarray] = [_hex_to_rgb(h) for h in _GROUP_PALETTE_HEX]


_DEFAULT_QUALITY_METRIC = "minSICN"
_QUALITY_CMAP = "viridis"


class ColorModeController:
    """Switch the mesh viewer between named color modes."""

    __slots__ = (
        "_color_mgr", "_registry", "_scene", "_sel", "_vis_mgr",
        "_plotter", "_mode",
        "_quality_metric", "_quality_mapper_state", "_quality_bar_title",
    )

    def __init__(
        self,
        *,
        color_mgr: "ColorManager",
        registry: "EntityRegistry",
        scene: "MeshSceneData",
        sel: "SelectionState",
        vis_mgr: "VisibilityManager",
        plotter,
    ) -> None:
        self._color_mgr = color_mgr
        self._registry = registry
        self._scene = scene
        self._sel = sel
        self._vis_mgr = vis_mgr
        self._plotter = plotter
        self._mode = "Default"
        self._quality_metric = _DEFAULT_QUALITY_METRIC
        # Per-dim VTK mapper state captured on entry to Quality mode,
        # restored on exit. Keys: dim -> {color_mode, scalar_mode, ...}.
        self._quality_mapper_state: dict[int, dict[str, Any]] = {}
        self._quality_bar_title: str | None = None

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        # If leaving Quality, revert mapper state first so the next
        # _repaint sees rgb-mode mappers.
        if self._mode == "Quality" and mode != "Quality":
            self._exit_quality_mode()

        if mode == "Default":
            self._color_mgr.reset_idle_fn()
        elif mode == "Element Type":
            self._color_mgr.set_idle_fn(self._elem_type_idle)
        elif mode == "Physical Group":
            self._color_mgr.set_idle_fn(self._phys_group_idle)
        elif mode == "Quality":
            self._enter_quality_mode()
            self._mode = mode
            return  # _enter_quality_mode renders; skip _repaint
        else:
            return

        self._mode = mode
        self._repaint()

    def refresh(self) -> None:
        """Re-apply the current mode (e.g. after theme change)."""
        if self._mode == "Quality":
            return  # quality colors are theme-independent
        self._repaint()

    # ------------------------------------------------------------------
    # Idle-fn strategy (Default / Element Type / Physical Group)
    # ------------------------------------------------------------------

    def _elem_type_idle(self, dt: "DimTag") -> np.ndarray:
        cat = self._scene.brep_dominant_type.get(dt, "Line")
        return _ELEM_TYPE_RGB.get(cat, _FALLBACK_RGB)

    def _phys_group_idle(self, dt: "DimTag") -> np.ndarray:
        name = self._scene.brep_to_group.get(dt)
        if name is None:
            return _FALLBACK_RGB
        idx = abs(hash(name)) % len(_GROUP_PALETTE_RGB)
        return _GROUP_PALETTE_RGB[idx]

    def _repaint(self) -> None:
        picks = set(self._sel.picks)
        for dt in self._registry.all_entities():
            self._color_mgr.set_entity_state(
                dt,
                picked=dt in picks,
                hidden=self._vis_mgr.is_hidden(dt),
            )
        self._plotter.render()

    # ------------------------------------------------------------------
    # Quality strategy (scalar mapper)
    # ------------------------------------------------------------------

    def _enter_quality_mode(self) -> None:
        import pyvista as pv

        metric = self._quality_metric

        # Compute per-dim quality and the global value range. Lines
        # (dim=1) are skipped — gmsh's quality metrics return 0 for
        # straight 2-node line elements, which would flag every edge as
        # "worst" and compress the meaningful colorbar range.
        per_dim: dict[int, tuple[np.ndarray, Any]] = {}
        for dim in self._registry.dims:
            if dim < 2:
                continue
            grid = self._registry.dim_meshes.get(dim)
            actor = self._registry.dim_actors.get(dim)
            if grid is None or actor is None:
                continue
            qarr = self._get_quality(metric, dim)
            if qarr is None or len(qarr) != grid.n_cells:
                continue
            per_dim[dim] = (qarr, grid)

        if not per_dim:
            return

        all_q = np.concatenate([q for q, _ in per_dim.values()])
        qmin = float(all_q.min())
        qmax = float(all_q.max())
        if qmax - qmin < 1e-12:
            qmax = qmin + 1e-6

        lut = pv.LookupTable(cmap=_QUALITY_CMAP)
        lut.scalar_range = (qmin, qmax)

        scalar_name = f"_quality_{metric}"

        first_mapper = None
        for dim, (qarr, grid) in per_dim.items():
            actor = self._registry.dim_actors[dim]
            grid.cell_data[scalar_name] = qarr
            mapper = actor.GetMapper()
            self._quality_mapper_state.setdefault(dim, {
                "color_mode":         mapper.GetColorMode(),
                "scalar_mode":        mapper.GetScalarMode(),
                "array_name":         mapper.GetArrayName() or "colors",
                "scalar_visibility":  bool(mapper.GetScalarVisibility()),
                "lut":                mapper.GetLookupTable(),
            })
            mapper.SelectColorArray(scalar_name)
            mapper.SetScalarModeToUseCellData()
            mapper.SetColorModeToMapScalars()
            mapper.SetLookupTable(lut)
            mapper.SetUseLookupTableScalarRange(True)
            mapper.ScalarVisibilityOn()
            if first_mapper is None:
                first_mapper = mapper

        title = f"Quality ({metric})"
        self._quality_bar_title = title
        try:
            self._plotter.add_scalar_bar(title=title, mapper=first_mapper)
        except Exception:
            pass

        self._plotter.render()

    def _exit_quality_mode(self) -> None:
        for dim, state in self._quality_mapper_state.items():
            actor = self._registry.dim_actors.get(dim)
            if actor is None:
                continue
            mapper = actor.GetMapper()
            mapper.SelectColorArray(state["array_name"])
            mapper.SetScalarMode(state["scalar_mode"])
            mapper.SetColorMode(state["color_mode"])
            if state["lut"] is not None:
                mapper.SetLookupTable(state["lut"])
            mapper.SetScalarVisibility(int(state["scalar_visibility"]))
        self._quality_mapper_state.clear()

        if self._quality_bar_title is not None:
            try:
                self._plotter.remove_scalar_bar(self._quality_bar_title)
            except Exception:
                pass
            self._quality_bar_title = None

    def _get_quality(self, metric: str, dim: int) -> np.ndarray | None:
        """Return per-cell quality for *dim*, computing & caching on first call."""
        cache = self._scene.quality.setdefault(metric, {})
        if dim in cache:
            return cache[dim]

        cell_map = self._scene.batch_cell_to_elem.get(dim)
        if cell_map is None or len(cell_map) == 0:
            return None

        elem_tags = cell_map.tolist()
        try:
            qlist = gmsh.model.mesh.getElementQualities(
                elem_tags, qualityName=metric,
            )
            qarr = np.asarray(qlist, dtype=np.float64)
        except Exception:
            return None

        cache[dim] = qarr
        return qarr
