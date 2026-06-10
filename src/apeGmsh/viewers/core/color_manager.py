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
    from apeGmsh._types import DimTag
    from .entity_registry import EntityRegistry


# ======================================================================
# Palette-sourced colors
#
# Every RGB value flows from the active ``Palette`` at the moment it is
# needed — no module-level snapshots, no hardcoded hex fallbacks. This
# makes theme switches coherent: one source of truth.
# ======================================================================


def _theme_idle_colors() -> dict[int, np.ndarray]:
    """Per-dim idle RGBs sourced from the active viewer theme."""
    from apeGmsh.viewers.ui.theme import THEME
    p = THEME.current
    return {
        0: np.array(p.dim_pt,  dtype=np.uint8),
        1: np.array(p.dim_crv, dtype=np.uint8),
        2: np.array(p.dim_srf, dtype=np.uint8),
        3: np.array(p.dim_vol, dtype=np.uint8),
    }


def _theme_hover_rgb() -> np.ndarray:
    from apeGmsh.viewers.ui.theme import THEME
    return np.array(THEME.current.hover_rgb, dtype=np.uint8)


def _theme_pick_rgb() -> np.ndarray:
    from apeGmsh.viewers.ui.theme import THEME
    return np.array(THEME.current.pick_rgb, dtype=np.uint8)


def _theme_hidden_rgb() -> np.ndarray:
    from apeGmsh.viewers.ui.theme import THEME
    return np.array(THEME.current.hidden_rgb, dtype=np.uint8)


class ColorManager:
    """Manages per-cell RGB arrays on batched meshes.

    State priority per entity:  hidden > picked > hovered > idle.
    """

    __slots__ = (
        "_registry",
        "_pick_override",
        "_idle_fn",
    )

    def __init__(
        self,
        registry: "EntityRegistry",
        *,
        pick_color: np.ndarray | None = None,
    ) -> None:
        self._registry = registry
        # ``_pick_override`` lets callers pin a pick colour (e.g. user pref).
        # When ``None``, the active palette's ``pick_rgb`` is used.
        self._pick_override: np.ndarray | None = pick_color
        self._idle_fn: Callable[["DimTag"], np.ndarray] = self._default_idle

    @property
    def _pick_rgb(self) -> np.ndarray:
        return self._pick_override if self._pick_override is not None else _theme_pick_rgb()

    @property
    def pick_rgb(self) -> np.ndarray:
        """Effective pick colour — the user override when set, else the
        active theme's ``pick_rgb``. Public read path so UI projections
        can initialize from the owner (ADR 0056 INV-1)."""
        return self._pick_rgb

    @property
    def _hover_rgb(self) -> np.ndarray:
        return _theme_hover_rgb()

    @property
    def _hidden_rgb(self) -> np.ndarray:
        return _theme_hidden_rgb()

    # ------------------------------------------------------------------
    # Idle color strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _default_idle(dt: "DimTag") -> np.ndarray:
        colors = _theme_idle_colors()
        return colors.get(dt[0], colors[2])

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
        """Reset every entity to its idle color.

        Uses the custom ``_idle_fn`` if set (e.g. mesh viewer's
        uniform scene color), otherwise falls back to per-dim defaults.
        """
        reg = self._registry
        for dim in reg.dims:
            mesh = reg.dim_meshes.get(dim)
            if mesh is None:
                continue
            colors = mesh.cell_data.get("colors")
            if colors is None:
                continue
            # Use _idle_fn for the first entity at this dim to get
            # the correct color (handles custom idle functions).
            idle = self._idle_fn((dim, 0))
            colors[:] = idle
            mesh.cell_data["colors"] = colors

    def recolor_all(
        self,
        picks: set["DimTag"],
        hidden: set["DimTag"] | None = None,
        hover: "DimTag | None" = None,
    ) -> None:
        """Batch recolor every entity in one pass per dimension.

        Walks the registry once per dim, computes each entity's idle
        color (per-entity ``_idle_fn`` — covers Element-Type and
        Physical-Group modes where colors vary by entity), scatters
        into the dim's ``colors`` buffer, then overlays hidden / pick
        / hover. Exactly one ``cell_data["colors"] = colors`` rebind
        per dim — vs one per entity in the legacy per-entity path.
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

            # Per-entity idle scatter — preserves correctness for
            # idle_fns whose output depends on the entity, not just dim.
            for dt in reg.all_entities():
                if dt[0] != dim:
                    continue
                cells = reg.cells_for_entity(dt)
                if not cells:
                    continue
                colors[cells] = self._idle_fn(dt)

            # Overlay hidden first (lowest priority below pick/hover).
            for dt in hidden:
                if dt[0] != dim:
                    continue
                cells = reg.cells_for_entity(dt)
                if cells:
                    colors[cells] = self._hidden_rgb

            # Overlay pick color on picked entities (above hidden).
            for dt in picks:
                if dt[0] != dim:
                    continue
                cells = reg.cells_for_entity(dt)
                if cells:
                    colors[cells] = self._pick_rgb

            # Overlay hover color (above pick? matches set_entity_state
            # priority: hidden > picked > hovered > idle, so hover is
            # NOT applied if the entity is picked or hidden).
            if (
                hover is not None
                and hover[0] == dim
                and hover not in picks
                and hover not in hidden
            ):
                cells = reg.cells_for_entity(hover)
                if cells:
                    colors[cells] = self._hover_rgb

            # Single VTK update per dimension.
            mesh.cell_data["colors"] = colors

    def set_pick_color(self, rgb: np.ndarray) -> None:
        """Override the palette's pick highlight color (user preference)."""
        self._pick_override = np.asarray(rgb, dtype=np.uint8)

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
