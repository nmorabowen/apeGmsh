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

from typing import TYPE_CHECKING, Any, Callable

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
        "dispatcher",
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
        # Injected by the mesh viewer (V3) and the model viewer (V4),
        # ADR 0056: when set, mutators owner-fire
        # MESH_ENTITY_VISIBILITY_CHANGED and the actor rebuild runs as
        # the dispatcher's ``entities`` pump (one coalesced render per
        # gesture). Both production viewers inject; None is the
        # standalone / unit-test mode (inline reconcile, no render —
        # there is no plotter.render() anywhere in this class).
        self.dispatcher: Any = None

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
        self._after_mutation()

    def isolate(self) -> None:
        """Hide everything except the currently picked entities."""
        picks = set(self._selection.picks)
        if not picks:
            return
        for dt in self._registry.all_entities():
            if dt not in picks:
                self._hidden.add(dt)
        self._after_mutation()

    def reveal_all(self) -> None:
        """Restore all hidden entities from the original meshes."""
        if not self._hidden:
            return
        self._hidden.clear()
        self._after_mutation()

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
        self._after_mutation()

    def _after_mutation(self) -> None:
        """Post-mutation propagation (ADR 0056 V3, owner-fires).

        With a dispatcher injected (mesh viewer V3, model viewer V4),
        fire ``MESH_ENTITY_VISIBILITY_CHANGED`` — the actor rebuild +
        recolor run synchronously as the dispatcher's ``entities``
        pump (:meth:`rebuild_now`) and the render coalesces; the
        ``on_changed`` observers then see post-rebuild state. Without
        a dispatcher (standalone / unit-test mode only — both
        production viewers inject) the inline reconcile runs, with no
        render.
        """
        if self.dispatcher is not None:
            from ..diagrams._dispatch import MESH_ENTITY_VISIBILITY_CHANGED
            self.dispatcher.fire(MESH_ENTITY_VISIBILITY_CHANGED)
            self._fire()
            return
        self.rebuild_now()
        self._fire()

    def rebuild_now(self) -> None:
        """Reconcile actors + colors against the current hidden set.

        The designated ``entities`` pump callee (ADR 0056 Part 4) —
        invoked by the mesh dispatcher, or inline on the legacy
        no-dispatcher path.
        """
        self._rebuild_actors()
        self._reset_colors()

    def _reset_colors(self) -> None:
        """Reset all visible entity colors to idle, re-apply pick highlights."""
        self._color_mgr.reset_all_idle()
        # Re-apply pick state for any remaining picks
        for dt in self._selection.picks:
            self._color_mgr.set_entity_state(dt, picked=True)

    def _expanded_hidden(self) -> set["DimTag"]:
        """User-hidden entities + every lower-dim entity that *only*
        bounds hidden entities.

        The BRep scene draws a solid's faces twice — once as the
        dim-2 surface mesh, once as the dim-3 volume-boundary mesh —
        and again as dim-1 edges / dim-0 corner points. Masking only
        the entity the user clicked leaves those duplicates on screen
        ("doesn't disappear completely"). Cascading down the topology
        makes a hidden body vanish entirely, while a face/edge shared
        with a still-visible body stays — it has a parent that is not
        hidden.

        ``self._hidden`` stays the canonical record of what the user
        explicitly hid (the outline eye state derives from it); this
        expansion is used only for what gets drawn.
        """
        if not self._hidden:
            return set()
        import gmsh
        hidden = set(self._hidden)
        # Top-down so a surface added by the volume pass is itself
        # considered when cascading into curves, etc.
        for parent_dim in (3, 2, 1):
            for _, ptag in gmsh.model.getEntities(parent_dim):
                if (parent_dim, ptag) not in hidden:
                    continue
                try:
                    _up, down = gmsh.model.getAdjacencies(parent_dim, ptag)
                except Exception:
                    continue
                for child in down:
                    cdt = (parent_dim - 1, int(child))
                    if cdt in hidden:
                        continue
                    try:
                        up, _d = gmsh.model.getAdjacencies(
                            parent_dim - 1, int(child),
                        )
                    except Exception:
                        up = []
                    # Hide the child only when every higher-dim entity
                    # it bounds is hidden too (shared boundary rule).
                    if len(up) == 0 or all(
                        (parent_dim, int(p)) in hidden for p in up
                    ):
                        hidden.add(cdt)
        return hidden

    def _rebuild_actors(self) -> None:
        """Extract visible cells per dimension and swap actors.

        Only rebuilds dimensions that have hidden entities (or all
        if revealing). Uses :meth:`_expanded_hidden` so a hidden body
        disappears completely, not just its top-dimensional shell.
        """
        if self._verbose:
            import time
            _t0 = time.perf_counter()
        plotter = self._plotter
        reg = self._registry
        effective = self._expanded_hidden()

        # Which dims are affected by hidden entities?
        affected_dims = set()
        if not effective:
            # Revealing all — rebuild every dim that was previously affected
            affected_dims = set(reg.dims)
        else:
            for dt in effective:
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

            # dim=0 is a glyph layer, not an extract_cells mesh — it
            # needs its own filtered rebuild so hidden points vanish.
            if dim == 0:
                self._rebuild_point_glyphs(effective)
                # The per-dim node cloud for dim=0 (registered in
                # ``mesh_scene.py``) is a second sphere-glyph actor at
                # the same locations as the dim-0 fill actor — without
                # its own rebuild it would resurrect every hidden point.
                self._rebuild_node_cloud(0, effective)
                continue

            # Drop this dim's prior silhouette. pyvista's silhouette is
            # a separate actor that ``remove_actor(fill)`` does NOT take
            # down — leaving it makes a hidden body keep its outline.
            # Recreated from the visible subset after the fill rebuild.
            old_sil = reg.dim_silhouette_actors.get(dim)
            if old_sil is not None:
                try:
                    plotter.remove_actor(old_sil)
                except Exception:
                    pass
                reg.dim_silhouette_actors.pop(dim, None)

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

            if not effective:
                # No hidden entities — restore full mesh
                visible = full_mesh
            else:
                # Build mask: keep cells not in hidden entities
                entity_tags = full_mesh.cell_data.get("entity_tag")
                if entity_tags is None:
                    continue
                hidden_tags = {dt[1] for dt in effective if dt[0] == dim}
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
                        # All cells hidden — drop the fill actor.
                        # Edges are rendered by the fill mapper itself
                        # (vtkProperty::EdgeVisibility on the fill kwargs),
                        # so no separate wire actor exists to remove.
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

            # Remove old fill actor
            old = reg.dim_actors.get(dim)
            if old is not None:
                try:
                    plotter.remove_actor(old)
                except Exception:
                    pass

            # Add new fill actor with same visual properties — its
            # mapper inherits ``show_edges=True`` for dim>=2 from the
            # original ``add_mesh`` kwargs, so element edges follow
            # the mask automatically without a second pipeline pass.
            new_actor = plotter.add_mesh(
                visible,
                reset_camera=False,
                show_scalar_bar=False,
                **kwargs,
            )
            reg.swap_dim(dim, visible, new_actor)

            # Recreate the silhouette from the visible subset so the
            # outline tracks the hide (only dims that had one — 2/3).
            sil_kw = reg.dim_silhouette_kwargs.get(dim)
            if sil_kw:
                try:
                    new_sil = plotter.add_silhouette(visible, **sil_kw)
                    reg.set_silhouette(dim, new_sil, sil_kw)
                except Exception:
                    pass

            # Rebuild the per-dim node cloud from the visible-node
            # subset.  Without this the per-dim sphere-glyph actor
            # (registered globally in ``mesh_scene.py``) keeps drawing
            # every node of this dim, leaving ghost-node sprites on top
            # of the freshly-filtered fill actor when entities are
            # hidden.  Honors the same ``effective`` set used above so
            # the shared-boundary rule is preserved: a node owned by a
            # still-visible entity (e.g. on a shared face) stays drawn.
            self._rebuild_node_cloud(dim, effective)

        if self._verbose:
            import time as _time
            print(f"[visibility] _rebuild_actors: {(_time.perf_counter()-_t0)*1000:.1f}ms  "
                  f"({len(affected_dims)} dims, {len(effective)} hidden)")

    def _rebuild_point_glyphs(self, effective: set["DimTag"]) -> None:
        """Rebuild the dim-0 glyph actor from the *visible* points.

        Point entities render as sphere glyphs (a special path that
        ``extract_cells`` can't touch), so to hide a point we rebuild
        the glyph set from the subset whose tag is not in
        *effective*. All points hidden → blank the existing actor but
        keep the registry entry so ``reveal_all`` rebuilds it.
        """
        reg = self._registry
        plotter = self._plotter
        kw = reg._add_mesh_kwargs.get(0)
        if not kw:
            return
        centers = kw.get("_centers_d0")
        tags = kw.get("_tags_d0", [])
        if centers is None or len(tags) == 0:
            return
        centers = np.asarray(centers)
        tags_arr = np.asarray(tags)
        hidden_pts = [dt[1] for dt in effective if dt[0] == 0]
        keep = (
            ~np.isin(tags_arr, hidden_pts)
            if hidden_pts else np.ones(len(tags_arr), dtype=bool)
        )
        old = reg.dim_actors.get(0)
        if not keep.any():
            # Every point hidden — blanking the actor is enough; the
            # registry keeps dim 0 so a later reveal rebuilds it.
            if old is not None:
                try:
                    old.SetVisibility(False)
                except Exception:
                    pass
            return
        if old is not None:
            try:
                plotter.remove_actor(old)
            except Exception:
                pass
        from ..scene.glyph_points import build_point_glyphs
        from ..ui.theme import THEME
        mesh, actor, _, _ = build_point_glyphs(
            plotter,
            centers[keep],
            [int(t) for t, k in zip(tags, keep) if k],
            model_diagonal=kw.get("model_diagonal", 1.0),
            point_size=kw.get("point_size", 10.0),
            idle_color=np.array(THEME.current.dim_pt, dtype=np.uint8),
        )
        reg.swap_dim(0, mesh, actor)

    def _rebuild_node_cloud(self, dim: int, effective: set["DimTag"]) -> None:
        """Rebuild ``registry.dim_node_actors[dim]`` from the *visible*
        node subset.

        The per-dim sphere-glyph node cloud is registered once in
        ``mesh_scene.py`` and was never touched by the fill-rebuild
        loop — every node of every dim stayed on screen regardless of
        hide state.  This method closes that gap.

        Filtering rule (matches the fill rebuild):

        - Look at ``registry.dim_node_entity_pairs[dim]`` — rows are
          ``(node_idx, entity_tag)`` for every (node, owning-entity)
          pair within ``dim``.
        - A node is **visible** if at least one of its owning entities
          is not in ``effective``.  This implements the shared-boundary
          rule: when a node is shared between hidden entity A and
          visible entity B (both of the same dim), B keeps it alive.
        - Rebuild the cloud from the surviving node indices, re-using
          the build kwargs stashed at scene-build time so the visual
          styling (color / marker size) matches the original.

        No-op when the registry has no pair data for *dim* — the
        scene-build path falls through to a gmsh-fallback for dims
        whose centroid pass failed, which doesn't carry entity tags;
        the prior all-nodes-visible behaviour is preserved for those.
        """
        reg = self._registry
        pairs = reg.dim_node_entity_pairs.get(dim)
        node_coords = reg._node_coords
        if pairs is None or len(pairs) == 0 or node_coords is None:
            # Locked fallback: keep all nodes visible rather than wipe
            # the cloud. But say so — silent ghost nodes after a hide
            # are indistinguishable from a visibility bug.
            if any(dt[0] == dim for dt in effective):
                from .._log import log_action
                log_action(
                    "visibility", "node_cloud_no_ownership_data",
                    dim=dim,
                    n_hidden=sum(1 for dt in effective if dt[0] == dim),
                    _level="warning",
                )
            return

        if effective:
            hidden_tags = [dt[1] for dt in effective if dt[0] == dim]
        else:
            hidden_tags = []

        if hidden_tags:
            visible_pairs_mask = ~np.isin(pairs[:, 1], hidden_tags)
            if not visible_pairs_mask.any():
                # Every node has at least one hidden owner and no
                # visible owner — blank the actor (preserve the entry
                # so reveal_all can resurrect it).
                old = reg.dim_node_actors.get(dim)
                if old is not None:
                    try:
                        old.SetVisibility(False)
                    except Exception:
                        pass
                return
            visible_node_indices = np.unique(pairs[visible_pairs_mask, 0])
        else:
            visible_node_indices = np.unique(pairs[:, 0])

        visible_coords = node_coords[visible_node_indices]
        if len(visible_coords) == 0:
            return

        plotter = self._plotter
        old = reg.dim_node_actors.get(dim)
        if old is not None:
            try:
                plotter.remove_actor(old)
            except Exception:
                pass

        from ..scene.glyph_points import build_node_cloud
        kw = reg._node_cloud_kwargs or {}
        new_cloud, new_actor = build_node_cloud(
            plotter,
            visible_coords,
            model_diagonal=kw.get("model_diagonal", 1.0),
            marker_size=kw.get("marker_size", 6.0),
            color=kw.get("color"),
        )
        reg.register_node_cloud(dim, new_cloud, new_actor)

    def _fire(self) -> None:
        for cb in self.on_changed:
            try:
                cb()
            except Exception:
                pass
