"""
EntityRegistry — Central data structure for viewer entity management.

Maps ``DimTag`` tuples ``(dimension, gmsh_tag)`` to VTK cell indices
within per-dimension merged meshes.  Replaces both the old
``_actor_to_id`` / ``_id_to_actor`` per-entity registration system
and the ``_batch_*`` dict family.

Usage::

    registry = EntityRegistry()
    registry.register_dim(dim=2, mesh=poly, actor=vtk_actor,
                          cell_to_dt={0: (2, 1), 1: (2, 1), 2: (2, 5), ...},
                          centroids={(2, 1): [0, 0, 0], (2, 5): [1, 0, 0]})
    dt = registry.resolve_pick(id(vtk_actor), cell_id=2)
    # -> (2, 5)
"""
from __future__ import annotations

from typing import Any

import numpy as np


# Type alias used throughout the viewer system
DimTag = tuple[int, int]


class EntityRegistry:
    """Unified entity ↔ cell mapping for batched VTK actors.

    Stores one merged mesh and one VTK actor per entity dimension.
    Provides O(1) lookup in both directions:
    - pick resolution: ``(actor_id, cell_id)`` → ``DimTag``
    - recolor / hide: ``DimTag`` → ``list[cell_idx]``
    """

    __slots__ = (
        "dim_meshes",
        "dim_actors",
        "_actor_id_to_dim",
        "_cell_to_dt",
        "_dt_to_cells",
        "centroids",
        "_bboxes",
    )

    def __init__(self) -> None:
        self.dim_meshes: dict[int, Any] = {}          # dim → PolyData | UnstructuredGrid
        self.dim_actors: dict[int, Any] = {}           # dim → vtkActor
        self._actor_id_to_dim: dict[int, int] = {}     # id(actor) → dim
        self._cell_to_dt: dict[int, dict[int, DimTag]] = {}   # dim → {cell_idx: DimTag}
        self._dt_to_cells: dict[DimTag, list[int]] = {}       # DimTag → [cell_indices]
        self.centroids: dict[DimTag, np.ndarray] = {}          # DimTag → (3,) xyz
        self._bboxes: dict[DimTag, np.ndarray] = {}            # DimTag → (8, 3) corners

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_dim(
        self,
        dim: int,
        mesh: Any,
        actor: Any,
        cell_to_dt: dict[int, DimTag],
        centroids: dict[DimTag, np.ndarray] | None = None,
        bboxes: dict[DimTag, np.ndarray] | None = None,
    ) -> None:
        """Register a merged mesh + actor for one dimension.

        Parameters
        ----------
        dim : int
            Entity dimension (0=points, 1=curves, 2=surfaces, 3=volumes).
        mesh : pv.PolyData or pv.UnstructuredGrid
            The merged VTK mesh containing all entities of this dim.
        actor : vtkActor
            The actor returned by ``plotter.add_mesh(mesh, ...)``.
        cell_to_dt : dict
            Mapping from cell index (within *mesh*) to ``DimTag``.
        centroids : dict, optional
            Mapping from ``DimTag`` to 3D centroid coordinates.
        bboxes : dict, optional
            Mapping from ``DimTag`` to ``ndarray (8, 3)`` AABB corners.
            If not provided, computed from centroids (degenerate boxes).
        """
        self.dim_meshes[dim] = mesh
        self.dim_actors[dim] = actor
        self._actor_id_to_dim[id(actor)] = dim
        self._cell_to_dt[dim] = cell_to_dt

        # Build inverse mapping: DimTag → list[cell_idx]
        inv: dict[DimTag, list[int]] = {}
        for cell_idx, dt in cell_to_dt.items():
            inv.setdefault(dt, []).append(cell_idx)
        self._dt_to_cells.update(inv)

        if centroids:
            self.centroids.update(centroids)

        if bboxes:
            self._bboxes.update(bboxes)
        elif centroids:
            # Fallback: degenerate bbox from centroid
            for dt, c in centroids.items():
                if dt not in self._bboxes:
                    self._bboxes[dt] = np.tile(c, (8, 1))

    # ------------------------------------------------------------------
    # Pick resolution
    # ------------------------------------------------------------------

    def resolve_pick(self, actor_id: int, cell_id: int) -> DimTag | None:
        """Resolve a VTK pick event to a ``DimTag``.

        Parameters
        ----------
        actor_id : int
            ``id(vtk_actor)`` from the picker's ``GetViewProp()``.
        cell_id : int
            Cell index from ``GetCellId()``.

        Returns
        -------
        DimTag or None
        """
        dim = self._actor_id_to_dim.get(actor_id)
        if dim is None:
            return None
        cell_map = self._cell_to_dt.get(dim)
        if cell_map is None:
            return None
        return cell_map.get(cell_id)

    # ------------------------------------------------------------------
    # Entity queries
    # ------------------------------------------------------------------

    def cells_for_entity(self, dt: DimTag) -> list[int]:
        """Return cell indices for entity *dt* (empty list if unknown)."""
        return self._dt_to_cells.get(dt, [])

    def mesh_for_entity(self, dt: DimTag) -> Any | None:
        """Return the merged mesh that contains entity *dt*."""
        return self.dim_meshes.get(dt[0])

    def actor_for_entity(self, dt: DimTag) -> Any | None:
        """Return the VTK actor that renders entity *dt*."""
        return self.dim_actors.get(dt[0])

    def all_entities(self, dim: int | None = None) -> list[DimTag]:
        """Return all registered entity DimTags.

        Parameters
        ----------
        dim : int, optional
            Filter to a specific dimension.  ``None`` returns all.
        """
        if dim is not None:
            cell_map = self._cell_to_dt.get(dim, {})
            return list(set(cell_map.values()))
        result: list[DimTag] = []
        for d_map in self._cell_to_dt.values():
            result.extend(set(d_map.values()))
        return result

    def centroid(self, dt: DimTag) -> np.ndarray | None:
        """Return the 3D centroid of entity *dt*, or ``None``."""
        return self.centroids.get(dt)

    def bbox(self, dt: DimTag) -> np.ndarray | None:
        """Return the 8 corners of the 3D AABB for entity *dt*.

        Returns ``ndarray (8, 3)`` or ``None`` if unknown.
        Corners are ordered: all combinations of (xmin/xmax, ymin/ymax, zmin/zmax).
        """
        return self._bboxes.get(dt)

    @property
    def dims(self) -> list[int]:
        """Registered dimensions (sorted)."""
        return sorted(self.dim_meshes.keys())

    def __len__(self) -> int:
        return len(self._dt_to_cells)

    def __contains__(self, dt: DimTag) -> bool:
        return dt in self._dt_to_cells

    def __repr__(self) -> str:
        counts = {d: len(set(m.values())) for d, m in self._cell_to_dt.items()}
        return f"<EntityRegistry dims={counts}>"
