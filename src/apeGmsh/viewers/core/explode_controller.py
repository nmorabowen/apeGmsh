"""ExplodeController — separates mesh blocks by the active color mode.

Supports independent X/Y/Z magnitudes so the user can spread blocks
along any single axis or all three simultaneously.  Explosion is
skipped for modes that don't define groups ("Default", "Quality").

Architecture:
- vol_grids[3]        : full UnstructuredGrid (hex/tet cells) captured
                        BEFORE _extract_surface_fast() in mesh_scene.py
- vol_cell_to_elem[3] : maps volume cell index → gmsh elem_tag
- dim_meshes[3]       : surface PolyData shell (post-extraction) whose
                        "colors" array is the source of truth for colors

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


def _build_surf_elem_colors(scene: Any) -> dict[int, np.ndarray]:
    """Build elem_tag → RGB(3,) uint8 from the surface mesh color array."""
    surf = scene.registry.dim_meshes.get(3)
    if surf is None:
        return {}
    surf_colors = surf.cell_data.get("colors")
    if surf_colors is None:
        return {}
    c2e = scene.batch_cell_to_elem.get(3)
    if c2e is None:
        return {}
    result: dict[int, np.ndarray] = {}
    for cell_idx, elem_tag in enumerate(c2e):
        if cell_idx < len(surf_colors):
            result[int(elem_tag)] = surf_colors[cell_idx]
    return result


def _apply_block_colors(
    block: Any,
    cell_indices: list[int],
    vol_to_elem: "np.ndarray | None",
    surf_colors: dict[int, np.ndarray],
) -> None:
    """Paint block UNIFORMLY with the group color.

    First surface-colored cell wins.  This fixes grey boundary faces:
    interior-to-global-mesh cells that are not in surf_colors inherit
    the group color rather than falling back to grey.
    """
    fallback = np.array([136, 136, 136], dtype=np.uint8)
    group_color = fallback
    if vol_to_elem is not None and surf_colors:
        for orig_ci in cell_indices:
            if orig_ci < len(vol_to_elem):
                c = surf_colors.get(int(vol_to_elem[orig_ci]))
                if c is not None:
                    group_color = c
                    break
    block.cell_data["colors"] = np.tile(group_color, (block.n_cells, 1))


# ======================================================================
# ExplodeController
# ======================================================================


class ExplodeController:
    """Separates mesh volume blocks by the active color mode.

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
        "_active",
    )

    def __init__(
        self,
        *,
        registry: Any,
        scene: Any,
        plotter: Any,
        view: Any = None,
    ) -> None:
        self._registry = registry
        self._scene = scene
        self._view = view
        self._plotter = plotter
        self._mode: str = "Default"
        self._magnitudes: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._explode_actors: list[Any] = []
        self._original_visibility: dict[Any, bool] = {}
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

        vol = self._scene.vol_grids.get(3)
        if vol is None:
            self._clear_explode()
            return

        groups = self._cell_groups_vol()
        if len(groups) <= 1:
            self._clear_explode()
            return

        offsets = self._group_offsets(vol, groups)
        surf_colors = _build_surf_elem_colors(self._scene)
        vol_to_elem = self._scene.vol_cell_to_elem.get(3)

        # Read display style from dim=3 actor BEFORE hiding it so explode
        # blocks inherit the user's current wireframe / edge-visibility state.
        actor3 = self._registry.dim_actors.get(3)
        _wireframe = False
        _show_edges = True
        if actor3 is not None:
            try:
                _wireframe = (actor3.GetProperty().GetRepresentation() == 1)
                _show_edges = bool(actor3.GetProperty().GetEdgeVisibility())
            except Exception:
                pass

        self._clear_explode()
        self._save_and_hide_actors()

        for key, cell_indices in groups.items():
            offset = offsets.get(key, np.zeros(3))
            idxs = np.asarray(cell_indices, dtype=np.int64)
            block = vol.extract_cells(idxs).copy()
            block.translate(offset, inplace=True)
            _apply_block_colors(block, cell_indices, vol_to_elem, surf_colors)
            actor = self._plotter.add_mesh(
                block,
                scalars="colors",
                rgb=True,
                style="wireframe" if _wireframe else "surface",
                show_edges=_show_edges and not _wireframe,
                edge_color="#333333",
                smooth_shading=False,
                pickable=False,
                reset_camera=False,
            )
            self._explode_actors.append(actor)

        self._active = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def enforce_hiding(self) -> None:
        """Re-assert actor hiding after an external visibility operation.

        Call from any viewer callback (filter, node-toggle) that may have
        restored actors that the explode should keep hidden.
        """
        if not self._active:
            return
        registry = self._registry
        for actor_dict in (
            registry.dim_actors,
            getattr(registry, "dim_wire_actors", {}),
            getattr(registry, "dim_node_actors", {}),
            getattr(registry, "dim_silhouette_actors", {}),
        ):
            for actor in actor_dict.values():
                if actor is not None and id(actor) in self._original_visibility:
                    actor.SetVisibility(False)

    def sync_node_visibility(self, visible: bool) -> None:
        """Update the intended visibility of node-cloud actors.

        During explosion node actors are always hidden (wrong positions).
        This stores the user's intent so restoration on explosion-end
        reflects the Show nodes checkbox state.
        """
        if not self._active:
            return
        for actor in getattr(self._registry, "dim_node_actors", {}).values():
            if actor is not None and id(actor) in self._original_visibility:
                self._original_visibility[id(actor)] = visible

    def _should_explode(self) -> bool:
        return any(v > 0.0 for v in self._magnitudes.values())

    def _save_and_hide_actors(self) -> None:
        registry = self._registry
        # Hide all fill, wire, node-cloud, and silhouette actors so nothing
        # from the original mesh floats at its original position during explosion.
        for actor_dict in (
            registry.dim_actors,
            getattr(registry, "dim_wire_actors", {}),
            getattr(registry, "dim_node_actors", {}),
            getattr(registry, "dim_silhouette_actors", {}),
        ):
            for actor in actor_dict.values():
                if actor is None:
                    continue
                self._original_visibility[id(actor)] = bool(actor.GetVisibility())
                actor.SetVisibility(False)

    def _clear_explode(self) -> None:
        for actor in self._explode_actors:
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass
        self._explode_actors.clear()

        if self._active:
            # Restore original visibility for every saved actor.
            registry = self._registry
            for actor_dict in (
                registry.dim_actors,
                getattr(registry, "dim_wire_actors", {}),
                getattr(registry, "dim_node_actors", {}),
                getattr(registry, "dim_silhouette_actors", {}),
            ):
                for actor in actor_dict.values():
                    if actor is None:
                        continue
                    was = self._original_visibility.get(id(actor), True)
                    actor.SetVisibility(was)

        self._original_visibility.clear()
        self._active = False

    def _cell_groups_vol(self) -> dict[Any, list[int]]:
        """Map each volume cell index to its group key under the current mode."""
        vol_to_elem = self._scene.vol_cell_to_elem.get(3)
        if vol_to_elem is None:
            return {}

        groups: dict[Any, list[int]] = {}
        for cell_idx, elem_tag in enumerate(vol_to_elem):
            key = self._elem_category_key(int(elem_tag))
            if key is None:
                continue
            groups.setdefault(key, []).append(cell_idx)
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
        diag = self._scene.model_diagonal
        keys = list(groups.keys())
        centers_all = vol.cell_centers().points
        centroids = {
            k: centers_all[np.asarray(groups[k], dtype=np.int64)].mean(axis=0)
            for k in keys
        }
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
