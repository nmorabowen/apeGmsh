"""
ColorManager — Single source of truth for per-cell RGB colors.

Modifies ``cell_data["colors"]`` arrays on the merged meshes.
**Never calls** ``plotter.render()`` — the caller is responsible
for batching recolor operations and rendering once at the end.

Usage::

    cm = ColorManager(registry)
    cm.set_entity_state(dt, picked=True)   # sets cells to pick color
    cm.set_entity_state(dt, hidden=True)   # sets cells to black
    cm.reset_all_idle()                     # restore all to dim defaults
    plotter.render()                        # caller renders
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .entity_registry import DimTag, EntityRegistry


# ======================================================================
# Default color palette (Catppuccin-inspired, protanopia-safe)
# ======================================================================

PICK_RGB   = np.array([231, 76, 60],  dtype=np.uint8)   # #E74C3C red
HOVER_RGB  = np.array([255, 215, 0],  dtype=np.uint8)   # #FFD700 gold
HIDDEN_RGB = np.array([0, 0, 0],      dtype=np.uint8)   # invisible on dark bg

IDLE_COLORS: dict[int, np.ndarray] = {
    0: np.array([232, 213, 183], dtype=np.uint8),  # #E8D5B7 warm white
    1: np.array([170, 170, 170], dtype=np.uint8),  # #AAAAAA mid grey
    2: np.array([91, 141, 184],  dtype=np.uint8),  # #5B8DB8 steel blue
    3: np.array([90, 110, 130],  dtype=np.uint8),  # #5A6E82 slate
}


class ColorManager:
    """Manages per-cell RGB arrays on batched meshes.

    State priority per entity:  hidden > picked > hovered > idle.
    """

    __slots__ = (
        "_registry",
        "_pick_rgb",
        "_hover_rgb",
        "_hidden_rgb",
        "_idle_fn",
    )

    def __init__(
        self,
        registry: "EntityRegistry",
        *,
        pick_color: np.ndarray | None = None,
    ) -> None:
        self._registry = registry
        self._pick_rgb = pick_color if pick_color is not None else PICK_RGB
        self._hover_rgb = HOVER_RGB
        self._hidden_rgb = HIDDEN_RGB
        self._idle_fn: Callable[["DimTag"], np.ndarray] = self._default_idle

    # ------------------------------------------------------------------
    # Idle color strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _default_idle(dt: "DimTag") -> np.ndarray:
        return IDLE_COLORS.get(dt[0], IDLE_COLORS[2])

    def set_idle_fn(self, fn: Callable[["DimTag"], np.ndarray]) -> None:
        """Set a custom idle-color function (e.g. partition or group colors)."""
        self._idle_fn = fn

    def reset_idle_fn(self) -> None:
        """Restore the default per-dimension idle colors."""
        self._idle_fn = self._default_idle

    # ------------------------------------------------------------------
    # Entity state
    # ------------------------------------------------------------------

    def set_entity_state(
        self,
        dt: "DimTag",
        *,
        picked: bool = False,
        hovered: bool = False,
        hidden: bool = False,
    ) -> None:
        """Update the color of all cells belonging to entity *dt*.

        Priority: hidden > picked > hovered > idle.
        Does NOT call ``plotter.render()``.
        """
        if hidden:
            rgb = self._hidden_rgb
        elif picked:
            rgb = self._pick_rgb
        elif hovered:
            rgb = self._hover_rgb
        else:
            rgb = self._idle_fn(dt)

        self._set_cells_rgb(dt, rgb)

    def recolor_entity(self, dt: "DimTag", rgb: np.ndarray) -> None:
        """Set a specific RGB on all cells of entity *dt*."""
        self._set_cells_rgb(dt, np.asarray(rgb, dtype=np.uint8))

    def reset_all_idle(self) -> None:
        """Reset every entity to its idle color (vectorized)."""
        reg = self._registry
        for dim in reg.dims:
            mesh = reg.dim_meshes.get(dim)
            if mesh is None:
                continue
            colors = mesh.cell_data.get("colors")
            if colors is None:
                continue
            idle = IDLE_COLORS.get(dim, IDLE_COLORS[2])
            colors[:] = idle  # broadcast fill — instant
            mesh.cell_data["colors"] = colors

    def recolor_all(
        self,
        picks: set["DimTag"],
        hidden: set["DimTag"] | None = None,
        hover: "DimTag | None" = None,
    ) -> None:
        """Batch recolor all entities in one pass per dimension.

        Uses numpy fancy indexing — single ``cell_data`` assignment
        per dimension instead of per-entity.
        """
        reg = self._registry
        hidden = hidden or set()

        for dim in reg.dims:
            mesh = reg.dim_meshes.get(dim)
            if mesh is None:
                continue
            colors = mesh.cell_data.get("colors")
            if colors is None:
                continue

            # Fill all with idle color for this dim
            idle = IDLE_COLORS.get(dim, IDLE_COLORS[2])
            colors[:] = idle

            # Overlay pick color on picked entities
            for dt in picks:
                if dt[0] != dim:
                    continue
                cells = reg.cells_for_entity(dt)
                if cells:
                    colors[cells] = self._pick_rgb

            # Overlay hover color
            if hover is not None and hover[0] == dim and hover not in picks:
                cells = reg.cells_for_entity(hover)
                if cells:
                    colors[cells] = self._hover_rgb

            # Single VTK update per dimension
            mesh.cell_data["colors"] = colors

    def set_pick_color(self, rgb: np.ndarray) -> None:
        """Change the pick highlight color."""
        self._pick_rgb = np.asarray(rgb, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_cells_rgb(self, dt: "DimTag", rgb: np.ndarray) -> None:
        """Write *rgb* to cells of entity *dt* (numpy fancy indexing)."""
        reg = self._registry
        cells = reg.cells_for_entity(dt)
        if not cells:
            return
        mesh = reg.mesh_for_entity(dt)
        if mesh is None:
            return
        colors = mesh.cell_data.get("colors")
        if colors is None:
            return
        colors[cells] = rgb  # numpy fancy indexing (vectorized)
        mesh.cell_data["colors"] = colors
