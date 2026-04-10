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
        "_verbose",
        "on_changed",
    )

    def __init__(
        self,
        registry: "EntityRegistry",
        color_mgr: "ColorManager",
        selection: "SelectionState",
        plotter: "pv.Plotter",
        *,
        verbose: bool = False,
    ) -> None:
        self._registry = registry
        self._color_mgr = color_mgr
        self._selection = selection
        self._plotter = plotter
        self._hidden: set["DimTag"] = set()
        self._verbose = verbose
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
        self._reset_colors()
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
        self._reset_colors()
        self._fire()

    def reveal_all(self) -> None:
        """Restore all hidden entities from the original meshes."""
        if not self._hidden:
            return
        self._hidden.clear()
        self._rebuild_actors()
        self._reset_colors()
        self._fire()

    def _reset_colors(self) -> None:
        """Reset all visible entity colors to idle, re-apply pick highlights."""
        self._color_mgr.reset_all_idle()
        # Re-apply pick state for any remaining picks
        for dt in self._selection._picks:
            self._color_mgr.set_entity_state(dt, picked=True)

    def _rebuild_actors(self) -> None:
        """Extract visible cells per dimension and swap actors.

        Only rebuilds dimensions that have hidden entities (or all
        if revealing).
        """
        if self._verbose:
            import time
            _t0 = time.perf_counter()
        from .color_manager import IDLE_COLORS
        plotter = self._plotter
        reg = self._registry

        # Which dims are affected by hidden entities?
        affected_dims = set()
        if not self._hidden:
            # Revealing all — rebuild every dim that was previously affected
            affected_dims = set(reg.dims)
        else:
            for dt in self._hidden:
                affected_dims.add(dt[0])
            # Also need dims that were previously hidden but now aren't
            for dim in reg.dims:
                if reg.dim_meshes.get(dim) is not reg._full_meshes.get(dim):
                    affected_dims.add(dim)

        for dim in reg.dims:
            if dim not in affected_dims:
                continue

            full_mesh = reg._full_meshes.get(dim)
            if full_mesh is None:
                continue

            # Reset colors on full mesh to idle before extracting
            idle_rgb = IDLE_COLORS.get(dim, IDLE_COLORS[2])
            colors = full_mesh.cell_data.get("colors")
            if colors is not None:
                colors[:] = idle_rgb
                full_mesh.cell_data["colors"] = colors

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
                show_scalar_bar=False,
                **kwargs,
            )
            reg.swap_dim(dim, visible, new_actor)

        if self._verbose:
            import time as _time
            print(f"[visibility] _rebuild_actors: {(_time.perf_counter()-_t0)*1000:.1f}ms  "
                  f"({len(affected_dims)} dims, {len(self._hidden)} hidden)")

    def _fire(self) -> None:
        for cb in self.on_changed:
            try:
                cb()
            except Exception:
                pass
