"""
VisibilityManager — Hide / isolate / reveal entities.

Uses ``extract_cells`` to create sub-meshes with only visible cells.
Hidden cells are truly removed from the render pipeline — no black
silhouettes, no rendering overhead.

The original (full) meshes are stored in the EntityRegistry for
``reveal_all`` to restore them.

Filter state model
------------------
The viewer has **two independent filter states** that do not share a
backing store:

1. **Dim filter** (cosmetic) — driven by FilterTab / MeshFilterTab dim
   checkboxes. Calls ``actor.SetVisibility(bool)`` on the fill *and*
   wire actors of a whole dimension. Does NOT enter ``_hidden``. Cheap:
   no mesh rebuild. Toggling a dim back on restores whatever state the
   actor was in (hidden entities stay hidden).

2. **Entity hide** (this class) — driven by browser-tab checkboxes
   (groups / element types) and pick-driven actions (``_act_hide`` /
   ``_act_isolate`` / ``_act_reveal_all``). Stores a ``frozenset[DimTag]``
   in ``_hidden`` and rebuilds the affected dim's fill + wire actor via
   ``extract_cells(mask)``.

These two states are deliberately independent. The dim filter is a
view-time toggle; entity hide is a model-state edit. Re-enabling a dim
does NOT clear the hidden set — that's a separate user action
(``reveal_all``).

A third mechanism exists in :class:`ClippingController` (render-time
clipping plane) which is independent of both above.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from apeGmsh._types import DimTag
    from .color_manager import ColorManager
    from .entity_registry import EntityRegistry
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

    def hide_dts(self, dts) -> None:
        """Add *dts* to the hidden set (programmatic, no pick dependency).

        Counterpart of :meth:`hide` for tree right-click menus and other
        callers that already know which entities to hide.
        """
        new = self._hidden | {dt for dt in dts}
        if new == self._hidden:
            return
        self.set_hidden(new)

    def isolate_dts(self, dts) -> None:
        """Hide everything except *dts*.

        Counterpart of :meth:`isolate` for tree right-click menus.
        """
        keep = {dt for dt in dts}
        if not keep:
            return
        new = {d for d in self._registry.all_entities() if d not in keep}
        self.set_hidden(new)

    def set_hidden(self, dts) -> None:
        """Replace the hidden set with *dts* and rebuild affected dims.

        Programmatic counterpart of hide/isolate/reveal_all — used by
        the browser tab to express "the current hidden set is exactly
        these entities", with no dependency on selection state.
        """
        new_hidden = {dt for dt in dts}
        if new_hidden == self._hidden:
            return
        self._hidden = new_hidden
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

        # Internal metadata keys that must not be forwarded to pyvista
        _INTERNAL_KEYS = frozenset({
            'model_diagonal', '_tags_d0', '_centers_d0',
        })

        for dim in reg.dims:
            if dim not in affected_dims:
                continue

            # Skip dim=0 (point glyphs) — they use a special rendering
            # path (build_brep_point_glyphs) that can't be reconstructed
            # by simple extract_cells + add_mesh.
            if dim == 0:
                continue

            full_mesh = reg._full_meshes.get(dim)
            if full_mesh is None:
                continue

            # Reset colors on full mesh to idle before extracting
            idle_rgb = self._color_mgr._idle_fn((dim, 0))
            colors = full_mesh.cell_data.get("colors")
            if colors is not None:
                colors[:] = idle_rgb
                full_mesh.cell_data["colors"] = colors

            kwargs = {k: v for k, v in
                      reg._add_mesh_kwargs.get(dim, {}).items()
                      if k not in _INTERNAL_KEYS}

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
                        # All cells hidden — remove fill and wire actors
                        old = reg.dim_actors.get(dim)
                        if old is not None:
                            try:
                                plotter.remove_actor(old)
                            except Exception:
                                pass
                        old_wire = reg.dim_wire_actors.get(dim)
                        if old_wire is not None:
                            try:
                                plotter.remove_actor(old_wire)
                            except Exception:
                                pass
                            reg.dim_wire_actors.pop(dim, None)
                            reg.dim_wire_meshes.pop(dim, None)
                        continue
                    else:
                        visible = full_mesh.extract_cells(
                            np.where(mask)[0]
                        )

            # Remove old fill actor
            old = reg.dim_actors.get(dim)
            if old is not None:
                try:
                    plotter.remove_actor(old)
                except Exception:
                    pass

            # Add new fill actor with same visual properties
            new_actor = plotter.add_mesh(
                visible,
                reset_camera=False,
                show_scalar_bar=False,
                **kwargs,
            )
            reg.swap_dim(dim, visible, new_actor)

            # Rebuild wire actor from the visible (post-mask) grid so
            # hidden entities also lose their wireframe. Only rebuild
            # if the dim already had a registered wire — the mesh
            # viewer registers one during scene build, the BRep model
            # viewer does not (it uses silhouette + dim=1 curves).
            old_wire = reg.dim_wire_actors.get(dim)
            if old_wire is not None:
                try:
                    plotter.remove_actor(old_wire)
                except Exception:
                    pass
                try:
                    new_wire_mesh = visible.extract_all_edges()
                    wire_color = "#666666"
                    wire_width = 0.5
                    try:
                        prop = old_wire.GetProperty()
                        rgb = prop.GetColor()
                        wire_color = (
                            float(rgb[0]),
                            float(rgb[1]),
                            float(rgb[2]),
                        )
                        wire_width = float(prop.GetLineWidth()) or 0.5
                    except Exception:
                        pass
                    new_wire_actor = plotter.add_mesh(
                        new_wire_mesh,
                        color=wire_color,
                        line_width=wire_width,
                        pickable=False,
                        reset_camera=False,
                    )
                    reg.register_wire(dim, new_wire_mesh, new_wire_actor)
                except Exception:
                    reg.dim_wire_actors.pop(dim, None)
                    reg.dim_wire_meshes.pop(dim, None)

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
