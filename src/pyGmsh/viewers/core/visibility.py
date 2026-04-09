"""
VisibilityManager — Hide / isolate / reveal entities.

Uses ``extract_cells`` to create sub-meshes with only visible cells.
Hidden cells are truly removed from the render pipeline — no black
silhouettes, no rendering overhead.

The original (full) meshes are stored in the EntityRegistry for
``reveal_all`` to restore them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from .color_manager import ColorManager
    from .entity_registry import DimTag, EntityRegistry
    from .selection import SelectionState


class VisibilityManager:
    """Cell-extraction-based entity visibility."""

    __slots__ = (
        "_registry",
        "_color_mgr",
        "_selection",
        "_plotter",
        "_hidden",
        "on_changed",
    )

    def __init__(
        self,
        registry: "EntityRegistry",
        color_mgr: "ColorManager",
        selection: "SelectionState",
        plotter: "pv.Plotter",
    ) -> None:
        self._registry = registry
        self._color_mgr = color_mgr
        self._selection = selection
        self._plotter = plotter
        self._hidden: set["DimTag"] = set()
        self.on_changed: list[Callable[[], None]] = []

    @property
    def hidden(self) -> frozenset["DimTag"]:
        return frozenset(self._hidden)

    def is_hidden(self, dt: "DimTag") -> bool:
        return dt in self._hidden

    def hide(self) -> None:
        """Hide every currently picked entity, then clear picks."""
        picks = self._selection.picks
        if not picks:
            return
        for dt in picks:
            self._hidden.add(dt)
        self._selection.clear()
        self._rebuild_actors()
        self._fire()

    def isolate(self) -> None:
        """Hide everything except the currently picked entities."""
        picks = set(self._selection.picks)
        if not picks:
            return
        for dt in self._registry.all_entities():
            if dt not in picks:
                self._hidden.add(dt)
        self._rebuild_actors()
        self._fire()

    def reveal_all(self) -> None:
        """Restore all hidden entities from the original meshes."""
        if not self._hidden:
            return
        self._hidden.clear()
        self._rebuild_actors()
        self._fire()

    def _rebuild_actors(self) -> None:
        """Extract visible cells per dimension and swap actors."""
        plotter = self._plotter
        reg = self._registry

        for dim in reg.dims:
            full_mesh = reg._full_meshes.get(dim)
            if full_mesh is None:
                continue

            kwargs = reg._add_mesh_kwargs.get(dim, {})

            if not self._hidden:
                # No hidden entities — restore full mesh
                visible = full_mesh
            else:
                # Build mask: keep cells not in hidden entities
                entity_tags = full_mesh.cell_data.get("entity_tag")
                if entity_tags is None:
                    continue
                hidden_tags = {dt[1] for dt in self._hidden if dt[0] == dim}
                if not hidden_tags:
                    visible = full_mesh
                else:
                    mask = np.isin(
                        np.asarray(entity_tags), list(hidden_tags),
                        invert=True,
                    )
                    if mask.all():
                        visible = full_mesh
                    elif not mask.any():
                        # All cells hidden — remove actor
                        old = reg.dim_actors.get(dim)
                        if old is not None:
                            try:
                                plotter.remove_actor(old)
                            except Exception:
                                pass
                        continue
                    else:
                        visible = full_mesh.extract_cells(
                            np.where(mask)[0]
                        )

            # Remove old actor
            old = reg.dim_actors.get(dim)
            if old is not None:
                try:
                    plotter.remove_actor(old)
                except Exception:
                    pass

            # Add new actor with same visual properties
            new_actor = plotter.add_mesh(
                visible,
                reset_camera=False,
                **kwargs,
            )
            reg.swap_dim(dim, visible, new_actor)

    def _fire(self) -> None:
        for cb in self.on_changed:
            try:
                cb()
            except Exception:
                pass
