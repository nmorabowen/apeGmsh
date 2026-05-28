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


from apeGmsh._types import DimTag


class EntityRegistry:
    """Unified entity ↔ cell mapping for batched VTK actors.

    Stores one merged mesh and one VTK actor per entity dimension.
    Provides O(1) lookup in both directions:
    - pick resolution: ``(actor_id, cell_id)`` -> ``DimTag``
    - recolor / hide: ``DimTag`` -> ``list[cell_idx]``
    """

    __slots__ = (
        "dim_meshes",
        "dim_actors",
        "dim_wire_meshes",
        "dim_wire_actors",
        "dim_node_clouds",
        "dim_node_actors",
        "dim_silhouette_actors",
        "dim_silhouette_kwargs",
        "_full_meshes",
        "_actor_id_to_dim",
        "_cell_to_dt",
        "_dt_to_cells",
        "centroids",
        "_bboxes",
        "_add_mesh_kwargs",
        "origin_shift",
        # Node-cloud rebuild support — visibility manager needs the
        # full node coord array, per-dim (node_idx, entity_tag) pairs,
        # and the build-time kwargs to re-render a filtered subset.
        "_node_coords",
        "dim_node_entity_pairs",
        "_node_cloud_kwargs",
    )

    def __init__(self) -> None:
        self.dim_meshes: dict[int, Any] = {}          # dim -> PolyData | UnstructuredGrid
        self.dim_actors: dict[int, Any] = {}           # dim -> vtkActor (fill)
        # Wireframe layer: corner-edge PolyData per dim>=2. Independent
        # of the fill actor; participates in clipping and dim filtering
        # but not in color modes or pick resolution.
        self.dim_wire_meshes: dict[int, Any] = {}      # dim -> PolyData
        self.dim_wire_actors: dict[int, Any] = {}      # dim -> vtkActor
        # Node cloud layer: one glyph actor per dim, each containing the
        # nodes used by entities of that dim (incl. boundary). Nodes
        # shared across dims appear in each owner's cloud — overlapping
        # at the same coords but invisible. Lets the dim filter scope
        # node visibility (hide 1D → 1D-only nodes go away).
        self.dim_node_clouds: dict[int, Any] = {}      # dim -> PolyData
        self.dim_node_actors: dict[int, Any] = {}      # dim -> vtkActor
        # Silhouette/outline layer (dim>=2). pyvista's ``silhouette=``
        # actor is separate from the fill actor and is NOT torn down by
        # ``remove_actor(fill)`` — we track it explicitly so the
        # visibility rebuild can drop a hidden body's outline too.
        self.dim_silhouette_actors: dict[int, Any] = {}   # dim -> vtkActor
        self.dim_silhouette_kwargs: dict[int, dict] = {}  # dim -> add_silhouette kwargs
        self._full_meshes: dict[int, Any] = {}         # dim -> original (unfiltered) mesh
        self._actor_id_to_dim: dict[int, int] = {}     # id(actor) -> dim
        self._cell_to_dt: dict[int, dict[int, DimTag]] = {}   # dim -> {cell_idx: DimTag}
        self._dt_to_cells: dict[DimTag, list[int]] = {}       # DimTag -> [cell_indices]
        self.centroids: dict[DimTag, np.ndarray] = {}          # DimTag -> (3,) xyz
        self._bboxes: dict[DimTag, np.ndarray] = {}            # DimTag -> (8, 3) corners
        self._add_mesh_kwargs: dict[int, dict] = {}            # dim -> kwargs for add_mesh
        self.origin_shift: np.ndarray = np.zeros(3)            # subtracted from world coords
        # Node-cloud rebuild support (populated by build_mesh_scene via
        # ``register_node_cloud_data``).  ``dim_node_entity_pairs[d]`` is
        # an (K, 2) int64 array of (node_idx_into_node_coords, entity_tag)
        # pairs — a node shared by N entities of the same dim contributes
        # N rows.  Lets ``VisibilityManager._rebuild_node_cloud`` filter
        # the cloud by the *expanded* hidden set while preserving the
        # shared-boundary rule (a node owned by a still-visible entity
        # stays drawn).
        self._node_coords: np.ndarray | None = None
        self.dim_node_entity_pairs: dict[int, np.ndarray] = {}
        self._node_cloud_kwargs: dict = {}

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
        add_mesh_kwargs: dict | None = None,
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
        add_mesh_kwargs : dict, optional
            The kwargs used in ``plotter.add_mesh`` (for actor recreation).
        """
        self.dim_meshes[dim] = mesh
        self._full_meshes[dim] = mesh  # store original for reveal
        self.dim_actors[dim] = actor
        self._actor_id_to_dim[id(actor)] = dim
        self._cell_to_dt[dim] = cell_to_dt
        if add_mesh_kwargs is not None:
            self._add_mesh_kwargs[dim] = add_mesh_kwargs

        # Build inverse mapping: DimTag -> list[cell_idx]
        inv: dict[DimTag, list[int]] = {}
        for cell_idx, dt in cell_to_dt.items():
            inv.setdefault(dt, []).append(cell_idx)
        self._dt_to_cells.update(inv)

        if centroids:
            self.centroids.update(centroids)

        if bboxes:
            self._bboxes.update(bboxes)
        elif centroids:
            for dt, c in centroids.items():
                if dt not in self._bboxes:
                    self._bboxes[dt] = np.tile(c, (8, 1))

    def register_wire(self, dim: int, mesh: Any, actor: Any) -> None:
        """Register the wireframe (corner-edge) layer for *dim*.

        The wireframe is a separate actor that draws one line segment
        per FE element edge. It is built from the linearized fill cells
        via ``extract_all_edges`` and is not part of pick resolution.
        """
        self.dim_wire_meshes[dim] = mesh
        self.dim_wire_actors[dim] = actor

    def register_node_cloud(self, dim: int, cloud: Any, actor: Any) -> None:
        """Register a per-dim node-glyph actor."""
        self.dim_node_clouds[dim] = cloud
        self.dim_node_actors[dim] = actor

    def register_node_cloud_data(
        self,
        node_coords: np.ndarray,
        dim_node_entity_pairs: dict[int, np.ndarray],
        kwargs: dict,
    ) -> None:
        """Stash the inputs ``VisibilityManager`` needs to rebuild node
        clouds after a hide.

        ``node_coords`` is the full ``(N_nodes, 3)`` coordinate array
        (the same one ``build_mesh_scene`` slices to feed
        :func:`build_node_cloud`).  ``dim_node_entity_pairs[d]`` is an
        ``(K_d, 2)`` int64 array whose rows are
        ``(node_idx_into_node_coords, entity_tag)`` — *one row per
        (node, owning-entity) pair*, so a node shared by two entities
        of the same dim contributes two rows.  ``kwargs`` holds the
        keyword args passed to :func:`build_node_cloud` at scene-build
        time (``model_diagonal``, ``marker_size``, ``color``); the
        rebuild reuses them so the rebuilt cloud matches the original
        visual styling exactly.
        """
        self._node_coords = node_coords
        self.dim_node_entity_pairs = dict(dim_node_entity_pairs)
        self._node_cloud_kwargs = dict(kwargs)

    def set_silhouette(self, dim: int, actor: Any, kwargs: dict) -> None:
        """Track the explicit silhouette actor + its ``add_silhouette``
        kwargs for *dim* so the visibility rebuild can remove and
        recreate it from the visible subset (pyvista does not tear the
        silhouette down with the fill actor)."""
        self.dim_silhouette_actors[dim] = actor
        self.dim_silhouette_kwargs[dim] = kwargs

    def swap_dim(self, dim: int, new_mesh: Any, new_actor: Any) -> None:
        """Replace the mesh+actor for *dim* after extract_cells.

        Rebuilds ``cell_to_dt`` from the preserved ``entity_tag`` cell_data.
        """
        # Clean up old actor id
        old_actor = self.dim_actors.get(dim)
        if old_actor is not None:
            self._actor_id_to_dim.pop(id(old_actor), None)

        self.dim_meshes[dim] = new_mesh
        self.dim_actors[dim] = new_actor
        self._actor_id_to_dim[id(new_actor)] = dim

        # Rebuild cell_to_dt from preserved entity_tag cell_data
        entity_tags = new_mesh.cell_data.get("entity_tag")
        new_cell_to_dt: dict[int, DimTag] = {}
        if entity_tags is not None:
            for ci, etag in enumerate(entity_tags):
                new_cell_to_dt[ci] = (dim, int(etag))
        self._cell_to_dt[dim] = new_cell_to_dt

        # Rebuild dt_to_cells for this dim
        for dt in [d for d in self._dt_to_cells if d[0] == dim]:
            del self._dt_to_cells[dt]
        inv: dict[DimTag, list[int]] = {}
        for ci, dt in new_cell_to_dt.items():
            inv.setdefault(dt, []).append(ci)
        self._dt_to_cells.update(inv)

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

    def entity_points(self, dt: DimTag, max_points: int = 64) -> np.ndarray | None:
        """Return representative mesh points for entity *dt*.

        For accurate box-selection of concave shapes (especially
        volumes), returns actual mesh vertices rather than just AABB
        corners.  If the entity has more than *max_points* unique
        vertices, a uniform subsample is returned.

        Uses ``mesh.cell_connectivity`` + ``mesh.offset`` to slice
        point ids directly — ~20x faster than per-cell ``get_cell()``
        VTK round-trips at 50k cells (validated in
        ``test_core_perf.py``). Falls back to ``get_cell()`` if either
        array isn't available on the wrapper.
        """
        mesh = self.dim_meshes.get(dt[0])
        if mesh is None:
            return None
        cells = self._dt_to_cells.get(dt)
        if not cells:
            return None
        try:
            cell_arr = np.asarray(cells, dtype=np.int64)
            connectivity = getattr(mesh, "cell_connectivity", None)
            offset = getattr(mesh, "offset", None)
            if connectivity is not None and offset is not None:
                # Vectorized path: offset[ci+1]-offset[ci] point ids per
                # cell, gathered via a single fancy-index on the flat
                # connectivity stream.
                starts = offset[cell_arr]
                ends = offset[cell_arr + 1]
                counts = ends - starts
                if counts.sum() == 0:
                    return None
                # Build flat index list pointing into `connectivity`.
                # Trick: idx[i] = starts[k] + (i - cum_start[k]) where k
                # is the cell containing flat-position i.
                total = int(counts.sum())
                flat = np.empty(total, dtype=np.int64)
                pos = 0
                for s, c in zip(starts.tolist(), counts.tolist()):
                    flat[pos:pos + c] = np.arange(s, s + c, dtype=np.int64)
                    pos += c
                pt_ids = connectivity[flat]
                unique = np.unique(pt_ids)
            elif (
                (rf := getattr(mesh, "regular_faces", None)) is not None
                and len(rf) > 0
                and int(cell_arr.max(initial=-1)) < len(rf)
            ):
                # PolyData uniform-face fast path. ``dim_meshes`` is a
                # PolyData when the dim>=3 actor renders the pre-extracted
                # boundary surface (all quads for a hex mesh). PolyData
                # exposes neither ``cell_connectivity`` nor ``offset``, so
                # the UG path above is skipped; ``regular_faces`` is the
                # vectorized equivalent — (n_cells, k) point ids — and
                # avoids the ~20x-slower per-cell ``get_cell()`` round
                # trip (82k calls / 3.5s per box-pick on the 607k mesh).
                unique = np.unique(np.asarray(rf)[cell_arr].ravel())
            else:
                # Fallback: per-cell VTK round-trip (mixed faces or an
                # unrecognised mesh type).
                pt_ids_set: set[int] = set()
                for ci in cell_arr:
                    pt_ids_set.update(mesh.get_cell(int(ci)).point_ids)
                if not pt_ids_set:
                    return None
                unique = np.array(sorted(pt_ids_set), dtype=np.int64)

            pts = np.asarray(mesh.points[unique])
            if len(pts) > max_points:
                step = len(pts) // max_points
                pts = pts[::step]
            return pts
        except Exception:
            return None

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
