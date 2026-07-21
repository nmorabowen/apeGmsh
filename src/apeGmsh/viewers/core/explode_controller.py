"""ExplodeController — separates mesh blocks by the active color mode.

Supports independent X/Y/Z magnitudes so the user can spread blocks
along any single axis or all three simultaneously.  Explosion is
skipped for modes that don't define groups ("Default", "Quality").

Mesh sources:
- dim 1/2: EntityRegistry._full_meshes[dim] (or dim_meshes as fallback)
           plus batch_cell_to_elem[dim]
- dim 3  : vol_grids[3] plus vol_cell_to_elem[3], captured before the
           render-only boundary extraction in mesh_scene.py

Groups are collected across every displayed element dimension.  A category
shared by several dimensions receives one offset, while each dimensional
block is rendered separately so line, surface, and volume styles remain
independent.

Per-axis offset:
    delta_K = centroid_K[axis] - global_center[axis]
    offset_K[axis] = (delta_K / max_delta) * diagonal * 0.35 * magnitude

Groups with no centroid separation on an axis (max_delta < 1e-12) fall
back to a uniform spread.

G-RENDER contract (ADR 0056 INV-5): apply() never calls plotter.render()
— PyVista auto-renders after add_mesh/remove_actor in the Qt event loop.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_NO_EXPLODE_MODES = {"Default", "Quality"}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _fallback_spread(n: int, i: int, axis_idx: int) -> float:
    """Uniform spread when all groups share the same centroid on an axis."""
    if n <= 1:
        return 0.0
    return (i - (n - 1) / 2.0) / max(1, n - 1)


# ======================================================================
# Module-level helpers (pure functions, no Qt dependency)
# ======================================================================


def _build_dim_elem_colors(
    scene: Any,
    dim: int,
    mesh: Any = None,
) -> dict[int, np.ndarray]:
    """Build elem_tag → RGB(3,) uint8 from one rendered dimension."""
    if mesh is None:
        mesh = scene.registry.dim_meshes.get(dim)
    if mesh is None:
        return {}
    colors = mesh.cell_data.get("colors")
    if colors is None:
        return {}
    c2e = scene.batch_cell_to_elem.get(dim)
    if c2e is None:
        return {}
    result: dict[int, np.ndarray] = {}
    for cell_idx, elem_tag in enumerate(c2e):
        if cell_idx < len(colors):
            result[int(elem_tag)] = colors[cell_idx]
    return result


def _build_surf_elem_colors(scene: Any) -> dict[int, np.ndarray]:
    """Backward-compatible dim=3 color lookup used by existing callers."""
    return _build_dim_elem_colors(scene, 3)


def _apply_block_colors(
    block: Any,
    cell_indices: list[int],
    cell_to_elem: "np.ndarray | None",
    elem_colors: dict[int, np.ndarray],
) -> None:
    """Paint block UNIFORMLY with the group color.

    The first category-colored cell wins.  Uniform painting also ensures
    interior 3D cells inherit the group color instead of falling back to
    grey when they have no counterpart on the rendered boundary shell.
    """
    fallback = np.array([136, 136, 136], dtype=np.uint8)
    group_color = fallback
    if cell_to_elem is not None and elem_colors:
        for orig_ci in cell_indices:
            if orig_ci < len(cell_to_elem):
                c = elem_colors.get(int(cell_to_elem[orig_ci]))
                if c is not None:
                    group_color = c
                    break
    block.cell_data["colors"] = np.tile(group_color, (block.n_cells, 1))


# ======================================================================
# ExplodeController
# ======================================================================


class ExplodeController:
    """Separates 1D, 2D, and 3D mesh blocks by the active color mode.

    Parameters
    ----------
    registry : EntityRegistry
        Entity registry (provides dim_meshes for color lookup).
    scene : MeshSceneData
        Built mesh scene (provides vol_grids, vol_cell_to_elem, …).
    plotter : pyvista.Plotter
        The viewer's plotter (actors are added/removed here).
    view : ViewerData, optional
        FEM snapshot (needed for Partition / Module grouping).
    """

    __slots__ = (
        "_registry", "_scene", "_view", "_plotter",
        "_mode", "_magnitudes", "_explode_actors", "_original_visibility",
        "_active", "_vis_mgr", "_color_mgr",
    )

    def __init__(
        self,
        *,
        registry: Any,
        scene: Any,
        plotter: Any,
        view: Any = None,
        vis_mgr: Any = None,
        color_mgr: Any = None,
    ) -> None:
        self._registry = registry
        self._scene = scene
        self._view = view
        self._plotter = plotter
        self._vis_mgr = vis_mgr
        self._color_mgr = color_mgr
        self._mode: str = "Default"
        self._magnitudes: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._explode_actors: list[Any] = []
        # (actor-dict name, dim) -> visibility before explosion. Keyed by
        # name+dim (not id(actor)) so swaps during explosion resolve live.
        self._original_visibility: dict[tuple[str, int], bool] = {}
        self._active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_axis(self, axis: str, value: float) -> None:
        self._magnitudes[axis] = _clamp01(value)
        self.apply()

    def set_value(self, v: float) -> None:
        """Backward-compat: set all three axes to the same value."""
        v = _clamp01(v)
        for k in self._magnitudes:
            self._magnitudes[k] = v
        self.apply()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.apply()

    def apply(self) -> None:
        if not self._should_explode() or self._mode in _NO_EXPLODE_MODES:
            self._clear_explode()
            return

        # Restore the original actors before rebuilding. This makes repeated
        # slider updates read the current actor style and visibility instead
        # of treating the actors hidden by the previous explode as user-hidden.
        self._clear_explode()

        sources = self._mesh_sources()
        groups = self._cell_groups(sources)
        if len(groups) <= 1:
            return

        offsets = self._group_offsets_multi(sources, groups)
        participating_dims = {
            dim
            for cells_by_dim in groups.values()
            for dim in cells_by_dim
        }
        styles = {
            dim: self._render_kwargs(dim)
            for dim in participating_dims
        }
        elem_colors = {
            dim: self._build_elem_colors(
                dim,
                None if dim == 3 else sources[dim][0],
            )
            for dim in participating_dims
        }

        self._save_and_hide_actors(participating_dims)
        self._active = True
        try:
            for key, cells_by_dim in groups.items():
                offset = offsets.get(key, np.zeros(3))
                for dim in sorted(cells_by_dim):
                    cell_indices = cells_by_dim[dim]
                    mesh, cell_to_elem = sources[dim]
                    idxs = np.asarray(cell_indices, dtype=np.int64)
                    block = mesh.extract_cells(idxs).copy()
                    block.translate(offset, inplace=True)
                    _apply_block_colors(
                        block,
                        cell_indices,
                        cell_to_elem,
                        elem_colors[dim],
                    )
                    actor = self._plotter.add_mesh(
                        block,
                        **styles[dim],
                    )
                    self._explode_actors.append(actor)
        except Exception:
            # Leave the viewer in its pre-explode state if a VTK actor cannot
            # be created. The original exception still reaches the caller.
            self._clear_explode()
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mesh_sources(self) -> dict[int, tuple[Any, np.ndarray]]:
        """Return full cell grids and aligned element-tag arrays per dim."""
        sources: dict[int, tuple[Any, np.ndarray]] = {}
        full_meshes = getattr(self._registry, "_full_meshes", {})

        for dim in sorted(self._registry.dim_meshes):
            if dim not in (1, 2, 3):
                continue
            if dim == 3:
                mesh = self._scene.vol_grids.get(3)
                cell_to_elem = self._scene.vol_cell_to_elem.get(3)
            else:
                mesh = full_meshes.get(dim)
                if mesh is None:
                    mesh = self._registry.dim_meshes.get(dim)
                cell_to_elem = self._scene.batch_cell_to_elem.get(dim)

            if mesh is None or cell_to_elem is None:
                continue
            cell_to_elem = np.asarray(cell_to_elem, dtype=np.int64)
            if getattr(mesh, "n_cells", 0) <= 0 or len(cell_to_elem) == 0:
                continue
            sources[dim] = (mesh, cell_to_elem)

        return sources

    def _render_kwargs(self, dim: int) -> dict[str, Any]:
        """Clone the dimension style and overlay current actor toggles."""
        internal_keys = frozenset({
            "model_diagonal", "_tags_d0", "_centers_d0",
        })
        kwargs = {
            key: value
            for key, value in getattr(
                self._registry, "_add_mesh_kwargs", {},
            ).get(dim, {}).items()
            if key not in internal_keys
        }
        kwargs.update(
            pickable=False,
            reset_camera=False,
            show_scalar_bar=False,
        )

        actor = self._registry.dim_actors.get(dim)
        if actor is not None:
            try:
                prop = actor.GetProperty()
                wireframe = prop.GetRepresentation() == 1
                kwargs["style"] = "wireframe" if wireframe else "surface"
                if dim >= 2:
                    kwargs["show_edges"] = (
                        bool(prop.GetEdgeVisibility()) and not wireframe
                    )
            except Exception:
                pass
        return kwargs

    def _build_elem_colors(
        self,
        dim: int = 3,
        mesh: Any = None,
    ) -> dict[int, np.ndarray]:
        """Build elem_tag → RGB from the active color mode's idle function.

        Reads directly from ``scene.elem_to_brep`` + ``ColorManager._idle_fn``
        so interior elements (not visible on the surface of the global mesh)
        get their correct Physical-Group / Element-Type color. Falls back to
        the rendered dimension's color array when no color manager is injected.
        """
        if self._color_mgr is not None:
            idle_fn = self._color_mgr._idle_fn
            result: dict[int, np.ndarray] = {}
            for elem_tag, brep_dt in self._scene.elem_to_brep.items():
                if brep_dt[0] == dim:
                    result[int(elem_tag)] = idle_fn(brep_dt)
            return result
        return _build_dim_elem_colors(self._scene, dim, mesh)

    def _actor_dicts(self) -> "dict[str, dict]":
        """The four per-dim actor dicts by stable name.

        Saved visibility is keyed by ``(name, dim)`` rather than
        ``id(actor)`` so that an actor SWAPPED in mid-explosion (e.g. a
        point-size rebuild replacing the node cloud, or a fill re-add) is
        resolved live from the registry at hide/restore time — id() identity
        is fragile across swaps (and can even alias a freed object).
        """
        r = self._registry
        return {
            "fill": r.dim_actors,
            "wire": getattr(r, "dim_wire_actors", {}),
            "node": getattr(r, "dim_node_actors", {}),
            "silhouette": getattr(r, "dim_silhouette_actors", {}),
        }

    def enforce_hiding(self) -> None:
        """Re-assert actor hiding after an external visibility operation.

        Call from any viewer callback (filter, node-toggle) that may have
        restored actors that the explode should keep hidden.
        """
        if not self._active:
            return
        dicts = self._actor_dicts()
        # Resolve each saved (name, dim) to the CURRENT actor and hide it —
        # robust to swaps because we never hold a stale actor reference.
        for name, dim in self._original_visibility:
            actor = dicts.get(name, {}).get(dim)
            if actor is not None:
                actor.SetVisibility(False)
        # Node-cloud actors are hidden unconditionally (covers a node dim that
        # appeared only after the snapshot was taken).
        for actor in dicts["node"].values():
            if actor is not None:
                actor.SetVisibility(False)

    def sync_node_visibility(self, visible: bool) -> None:
        """Update the intended visibility of node-cloud actors.

        During explosion node actors are always hidden (wrong positions).
        This stores the user's intent so restoration on explosion-end
        reflects the Show nodes checkbox state.
        """
        if not self._active:
            return
        for key in list(self._original_visibility):
            if key[0] == "node":
                self._original_visibility[key] = visible

    def _should_explode(self) -> bool:
        return any(v > 0.0 for v in self._magnitudes.values())

    def _save_and_hide_actors(self, exploded_dims: set[int]) -> None:
        # Hide each exploded dimension's original geometry. Node clouds are
        # hidden globally because their points cannot follow category-specific
        # offsets and would otherwise remain at misleading original positions.
        # Saved by (name, dim) — see _actor_dicts.
        for name, actor_dict in self._actor_dicts().items():
            for dim, actor in actor_dict.items():
                if actor is None:
                    continue
                if name != "node" and dim not in exploded_dims:
                    continue
                self._original_visibility[(name, dim)] = bool(actor.GetVisibility())
                actor.SetVisibility(False)

    def _clear_explode(self) -> None:
        for actor in self._explode_actors:
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass
        self._explode_actors.clear()

        if self._active:
            # Restore saved visibility, resolving each (name, dim) to the
            # CURRENT actor so a swap during explosion restores correctly.
            dicts = self._actor_dicts()
            for (name, dim), was in self._original_visibility.items():
                actor = dicts.get(name, {}).get(dim)
                if actor is not None:
                    actor.SetVisibility(was)

        self._original_visibility.clear()
        self._active = False

    def _cell_groups_vol(self) -> dict[Any, list[int]]:
        """Map each volume cell index to its group key under the current mode.

        Cells belonging to hidden entities (via VisibilityManager) are
        excluded so they don't reappear inside the exploded view.
        """
        vol_to_elem = self._scene.vol_cell_to_elem.get(3)
        if vol_to_elem is None:
            return {}
        return self._cell_groups_for_dim(3, vol_to_elem)

    def _cell_groups_for_dim(
        self,
        dim: int,
        cell_to_elem: np.ndarray,
        *,
        n_cells: int | None = None,
    ) -> dict[Any, list[int]]:
        """Group one dimension's visible cells under the active color mode."""
        cell_to_elem = np.asarray(cell_to_elem, dtype=np.int64)
        if n_cells is None:
            n_cells = len(cell_to_elem)

        hidden_entity_tags: set[int] = set()
        if self._vis_mgr is not None:
            hidden_entity_tags = {
                dt[1] for dt in self._vis_mgr.hidden if dt[0] == dim
            }

        groups: dict[Any, list[int]] = {}
        for cell_idx, elem_tag in enumerate(cell_to_elem[:n_cells]):
            if hidden_entity_tags:
                brep = self._scene.elem_to_brep.get(int(elem_tag))
                if (
                    brep is not None
                    and brep[0] == dim
                    and brep[1] in hidden_entity_tags
                ):
                    continue
            key = self._elem_category_key(int(elem_tag))
            if key is None:
                continue
            groups.setdefault(key, []).append(cell_idx)
        return groups

    def _cell_groups(
        self,
        sources: dict[int, tuple[Any, np.ndarray]],
    ) -> dict[Any, dict[int, list[int]]]:
        """Collect category groups across all available element dimensions."""
        groups: dict[Any, dict[int, list[int]]] = {}
        for dim, (mesh, cell_to_elem) in sources.items():
            groups_for_dim = self._cell_groups_for_dim(
                dim,
                cell_to_elem,
                n_cells=int(mesh.n_cells),
            )
            for key, cell_indices in groups_for_dim.items():
                groups.setdefault(key, {})[dim] = cell_indices
        return groups

    def _elem_category_key(self, elem_tag: int) -> Any:
        mode = self._mode
        scene = self._scene
        if mode == "Partition":
            view = self._view
            if view is not None and view.elements.has_partitions:
                return view.elements.partition_for(elem_tag)
            return None
        if mode == "Physical Group":
            dt = scene.elem_to_brep.get(elem_tag)
            if dt is not None:
                return scene.brep_to_group.get(dt)
            return None
        if mode == "Module":
            view = self._view
            if view is not None and view.elements.has_modules:
                return view.elements.module_for(elem_tag)
            return None
        if mode == "Element Type":
            ed = scene.elem_data.get(elem_tag)
            if ed is not None:
                return ed.get("type_name")
            return None
        # Fallback: group by BRep entity
        return scene.elem_to_brep.get(elem_tag)

    def _group_offsets(
        self, vol: Any, groups: dict[Any, list[int]]
    ) -> dict[Any, np.ndarray]:
        keys = list(groups.keys())
        centers_all = vol.cell_centers().points
        centroids = {
            k: centers_all[np.asarray(groups[k], dtype=np.int64)].mean(axis=0)
            for k in keys
        }
        return self._offsets_from_centroids(centroids)

    def _group_offsets_multi(
        self,
        sources: dict[int, tuple[Any, np.ndarray]],
        groups: dict[Any, dict[int, list[int]]],
    ) -> dict[Any, np.ndarray]:
        """Compute one shared offset per category across all dimensions."""
        centers_by_dim = {
            dim: mesh.cell_centers().points
            for dim, (mesh, _cell_to_elem) in sources.items()
        }
        centroids: dict[Any, np.ndarray] = {}
        for key, cells_by_dim in groups.items():
            points = [
                centers_by_dim[dim][np.asarray(indices, dtype=np.int64)]
                for dim, indices in cells_by_dim.items()
                if indices
            ]
            if points:
                centroids[key] = np.concatenate(points, axis=0).mean(axis=0)
        return self._offsets_from_centroids(centroids)

    def _offsets_from_centroids(
        self,
        centroids: dict[Any, np.ndarray],
    ) -> dict[Any, np.ndarray]:
        """Convert category centroids into normalized per-axis offsets."""
        if not centroids:
            return {}
        diag = self._scene.model_diagonal
        keys = list(centroids)
        global_center = np.mean(list(centroids.values()), axis=0)
        offsets: dict[Any, np.ndarray] = {k: np.zeros(3) for k in keys}

        for axis_idx, axis_key in [(0, "x"), (1, "y"), (2, "z")]:
            mag = self._magnitudes[axis_key]
            if mag <= 0.0:
                continue
            deltas = {
                k: centroids[k][axis_idx] - global_center[axis_idx]
                for k in keys
            }
            max_delta = max(abs(d) for d in deltas.values())
            if max_delta < 1e-12:
                for i, key in enumerate(keys):
                    offsets[key][axis_idx] = (
                        _fallback_spread(len(keys), i, axis_idx) * diag * 0.35 * mag
                    )
            else:
                for key in keys:
                    offsets[key][axis_idx] = (
                        deltas[key] / max_delta * diag * 0.35 * mag
                    )

        return offsets
