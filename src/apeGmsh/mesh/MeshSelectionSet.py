"""
MeshSelectionSet — Post-mesh selection system complementary to PhysicalGroups.

Uses the same ``(dim, tag) + name`` identity contract as
:class:`PhysicalGroups`, but operates on *mesh* entities (nodes and
elements) rather than geometric entities.

* ``dim=0`` -> node set
* ``dim=1`` -> 1-D element set (line elements)
* ``dim=2`` -> 2-D element set (tri/quad)
* ``dim=3`` -> 3-D element set (tet/hex)

The mutable composite lives on ``g.mesh_selection``; an immutable
snapshot :class:`MeshSelectionStore` is captured into ``FEMData`` at
``get_fem_data()`` time.

Usage::

    g.mesh.generation.generate(2)

    g.mesh_selection.add_nodes(on_plane=("z", 0.0, 1e-3), name="base")
    g.mesh_selection.add_elements(dim=2, in_box=[0,0,-1, 10,10,1], name="zone")

    fem = g.mesh.queries.get_fem_data(dim=2)
    fem.mesh_selection.get_nodes(0, 1)  # same shape as fem.physical.get_nodes()
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import gmsh
import numpy as np
import pandas as pd

from . import _mesh_filters as _flt

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase

from apeGmsh._logging import _HasLogging
from apeGmsh._types import DimTag

_DIM_LABEL = {0: "nodes", 1: "line_elems", 2: "surf_elems", 3: "vol_elems"}


# ======================================================================
# MeshSelectionSet — mutable composite (lives on g.mesh_selection)
# ======================================================================


class MeshSelectionSet(_HasLogging):
    """Post-mesh selection composite — complementary to PhysicalGroups.

    Attached to ``g.mesh_selection`` by the session framework.
    Stores sets of mesh node/element IDs, resolved by spatial queries
    or explicit tag lists.  Mirrors the PhysicalGroups API shape so
    downstream consumers (solvers, FEMData) can treat both identically.
    """

    _log_prefix = "MeshSelection"

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent
        self._sets: dict[DimTag, dict] = {}
        self._next_tag: dict[int, int] = {0: 1, 1: 1, 2: 1, 3: 1}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alloc_tag(self, dim: int, tag: int = -1) -> int:
        """Return *tag* if positive, else auto-allocate next free tag."""
        if tag > 0:
            self._next_tag[dim] = max(self._next_tag[dim], tag + 1)
            return tag
        t = self._next_tag[dim]
        self._next_tag[dim] = t + 1
        return t

    def _get_mesh_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Fetch all mesh nodes as (ids, coords(N,3)) from Gmsh."""
        tags, coords, _ = gmsh.model.mesh.getNodes()
        ids = np.asarray(tags, dtype=np.int64)
        xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        return ids, xyz

    def _get_mesh_elements(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Fetch element IDs and connectivity for dimension *dim*."""
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(dim=dim, tag=-1)
        if not etypes:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.int64)
        # Assume single element type per dim (most common case)
        all_ids = []
        all_conn = []
        for etype, etags, enodes in zip(etypes, etags_list, enodes_list):
            _, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(etype)
            ids = np.asarray(etags, dtype=np.int64)
            conn = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
            all_ids.append(ids)
            all_conn.append(conn)
        if len(all_ids) == 1:
            return all_ids[0], all_conn[0]
        # Mixed types: pad to max npe
        max_npe = max(c.shape[1] for c in all_conn)
        padded = []
        for c in all_conn:
            if c.shape[1] < max_npe:
                pad = np.full((c.shape[0], max_npe - c.shape[1]), -1, dtype=np.int64)
                c = np.hstack([c, pad])
            padded.append(c)
        return np.concatenate(all_ids), np.vstack(padded)

    def _build_node_lookup(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (node_ids, coords, id_to_idx) for the current mesh."""
        ids, coords = self._get_mesh_nodes()
        id_to_idx = {int(nid): i for i, nid in enumerate(ids)}
        return ids, coords, id_to_idx

    def _store_node_set(
        self, tag: int, name: str, node_ids: np.ndarray, node_coords: np.ndarray,
    ) -> None:
        self._sets[(0, tag)] = {
            "name": name,
            "node_ids": node_ids.copy(),
            "node_coords": node_coords.copy(),
            "element_ids": None,
            "connectivity": None,
        }

    def _store_element_set(
        self, dim: int, tag: int, name: str,
        elem_ids: np.ndarray, connectivity: np.ndarray,
        node_ids: np.ndarray, node_coords: np.ndarray,
    ) -> None:
        self._sets[(dim, tag)] = {
            "name": name,
            "node_ids": node_ids.copy(),
            "node_coords": node_coords.copy(),
            "element_ids": elem_ids.copy(),
            "connectivity": connectivity.copy(),
        }

    # ------------------------------------------------------------------
    # Creation — explicit
    # ------------------------------------------------------------------

    def add(
        self,
        dim: int,
        tags: list[int],
        *,
        name: str = "",
        tag: int = -1,
    ) -> int:
        """Add a mesh selection set from explicit node/element IDs.

        Parameters
        ----------
        dim : 0 for node set, 1/2/3 for element set
        tags : list of node IDs (dim=0) or element IDs (dim>=1)
        name : optional label
        tag : explicit tag (auto-allocated if -1)

        Returns
        -------
        int — the allocated set tag
        """
        t = self._alloc_tag(dim, tag)
        all_ids, all_coords = self._get_mesh_nodes()

        if dim == 0:
            # Node set
            arr = np.array(tags, dtype=np.int64)
            mask = np.isin(all_ids, arr)
            self._store_node_set(t, name, all_ids[mask], all_coords[mask])
            self._log(f"add(dim=0, tag={t}, name='{name}') -> {int(mask.sum())} nodes")
        else:
            # Element set
            elem_ids_all, conn_all = self._get_mesh_elements(dim)
            arr = np.array(tags, dtype=np.int64)
            mask = np.isin(elem_ids_all, arr)
            sel_ids = elem_ids_all[mask]
            sel_conn = conn_all[mask]
            # Gather nodes from connectivity
            used = set(int(n) for n in sel_conn.ravel() if n >= 0)
            nmask = np.isin(all_ids, list(used))
            self._store_element_set(
                dim, t, name, sel_ids, sel_conn,
                all_ids[nmask], all_coords[nmask],
            )
            self._log(f"add(dim={dim}, tag={t}, name='{name}') -> "
                       f"{len(sel_ids)} elements, {int(nmask.sum())} nodes")
        return t

    # ------------------------------------------------------------------
    # Creation — spatial queries
    # ------------------------------------------------------------------

    def add_nodes(
        self,
        *,
        name: str = "",
        tag: int = -1,
        on_plane: tuple | None = None,
        in_box: tuple | list | None = None,
        in_sphere: tuple | None = None,
        closest_to: tuple | None = None,
        count: int = 1,
        predicate: Callable[[np.ndarray], np.ndarray] | None = None,
        inclusive: bool = False,
    ) -> int:
        """Create a node set (dim=0) from spatial queries.

        Parameters
        ----------
        name : label for the set
        tag : explicit tag (-1 = auto)
        on_plane : (axis, value, atol) — e.g. ("z", 0.0, 1e-3)
        in_box : (xmin, ymin, zmin, xmax, ymax, zmax). Half-open on the
            upper side by default (a node exactly on an upper face is
            excluded, matching results-side box semantics).
        in_sphere : (cx, cy, cz, radius)
        closest_to : (x, y, z) — use *count* to select N nearest
        count : number of nearest nodes (used with closest_to)
        predicate : fn(coords(N,3)) -> bool mask(N,)
        inclusive : if True, ``in_box`` uses a closed upper bound
            (restores the pre-S2 closed-closed behavior).

        Returns
        -------
        int — set tag
        """
        all_ids, all_coords = self._get_mesh_nodes()
        mask = np.ones(len(all_ids), dtype=bool)

        if on_plane is not None:
            axis, value = on_plane[0], on_plane[1]
            atol = on_plane[2] if len(on_plane) > 2 else 1e-6
            mask &= _flt.nodes_on_plane(all_coords, axis, value, atol)

        if in_box is not None:
            mask &= _flt.nodes_in_box(all_coords, in_box, inclusive=inclusive)

        if in_sphere is not None:
            cx, cy, cz, r = in_sphere
            mask &= _flt.nodes_in_sphere(all_coords, (cx, cy, cz), r)

        if closest_to is not None:
            mask &= _flt.nodes_nearest(all_coords, closest_to, count)

        if predicate is not None:
            mask &= predicate(all_coords)

        t = self._alloc_tag(0, tag)
        self._store_node_set(t, name, all_ids[mask], all_coords[mask])
        self._log(f"add_nodes(tag={t}, name='{name}') -> {int(mask.sum())} nodes")
        return t

    def add_elements(
        self,
        dim: int = 2,
        *,
        name: str = "",
        tag: int = -1,
        in_box: tuple | list | None = None,
        on_plane: tuple | None = None,
        predicate: Callable[[np.ndarray], np.ndarray] | None = None,
        inclusive: bool = False,
    ) -> int:
        """Create an element set from spatial queries.

        Parameters
        ----------
        dim : element dimension (1, 2, or 3)
        name : label
        tag : explicit tag (-1 = auto)
        in_box : select elements whose centroid is inside box. Half-open
            on the upper side by default (a centroid exactly on an upper
            face is excluded, matching results-side box semantics).
        on_plane : select elements with all nodes on plane
        predicate : fn(centroids(E,3)) -> bool mask(E,)
        inclusive : if True, ``in_box`` uses a closed upper bound
            (restores the pre-S2 closed-closed behavior).

        Returns
        -------
        int — set tag
        """
        all_node_ids, all_coords = self._get_mesh_nodes()
        id_to_idx = {int(nid): i for i, nid in enumerate(all_node_ids)}
        elem_ids, conn = self._get_mesh_elements(dim)

        if len(elem_ids) == 0:
            t = self._alloc_tag(dim, tag)
            self._store_element_set(
                dim, t, name,
                np.array([], dtype=np.int64),
                np.empty((0, 0), dtype=np.int64),
                np.array([], dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
            )
            self._log(f"add_elements(dim={dim}, tag={t}) -> 0 elements (none found)")
            return t

        mask = np.ones(len(elem_ids), dtype=bool)

        if in_box is not None:
            centroids = _flt.element_centroids(conn, id_to_idx, all_coords)
            mask &= _flt.elements_in_box(centroids, in_box, inclusive=inclusive)

        if on_plane is not None:
            axis, value = on_plane[0], on_plane[1]
            atol = on_plane[2] if len(on_plane) > 2 else 1e-6
            mask &= _flt.elements_on_plane(conn, id_to_idx, all_coords, axis, value, atol)

        if predicate is not None:
            centroids = _flt.element_centroids(conn, id_to_idx, all_coords)
            mask &= predicate(centroids)

        sel_ids = elem_ids[mask]
        sel_conn = conn[mask]
        used = set(int(n) for n in sel_conn.ravel() if n >= 0)
        nmask = np.isin(all_node_ids, list(used))

        t = self._alloc_tag(dim, tag)
        self._store_element_set(
            dim, t, name, sel_ids, sel_conn,
            all_node_ids[nmask], all_coords[nmask],
        )
        self._log(f"add_elements(dim={dim}, tag={t}, name='{name}') -> "
                   f"{len(sel_ids)} elements, {int(nmask.sum())} nodes")
        return t

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def set_name(self, dim: int, tag: int, name: str) -> "MeshSelectionSet":
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(f"No mesh selection (dim={dim}, tag={tag})")
        info["name"] = name
        return self

    def remove_name(self, name: str) -> "MeshSelectionSet":
        for info in self._sets.values():
            if info["name"] == name:
                info["name"] = ""
        return self

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, dim_tags: list[DimTag]) -> "MeshSelectionSet":
        for dt in dim_tags:
            self._sets.pop(dt, None)
        return self

    def remove_all(self) -> "MeshSelectionSet":
        self._sets.clear()
        self._next_tag = {0: 1, 1: 1, 2: 1, 3: 1}
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all(self, dim: int = -1) -> list[DimTag]:
        if dim == -1:
            return sorted(self._sets.keys())
        return sorted(k for k in self._sets if k[0] == dim)

    def get_entities(self, dim: int, tag: int) -> list[int]:
        """Return node IDs (dim=0) or element IDs (dim>=1)."""
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(f"No mesh selection (dim={dim}, tag={tag})")
        if dim == 0:
            return list(int(n) for n in info["node_ids"])
        return list(int(e) for e in info["element_ids"])

    def get_name(self, dim: int, tag: int) -> str:
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(f"No mesh selection (dim={dim}, tag={tag})")
        return info.get("name", "")

    def get_tag(self, dim: int, name: str) -> int | None:
        for (d, t), info in self._sets.items():
            if d == dim and info.get("name", "") == name:
                return t
        return None

    def get_nodes(self, dim: int, tag: int) -> dict:
        """Return ``{'tags': ndarray, 'coords': ndarray(N,3)}``."""
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(f"No mesh selection (dim={dim}, tag={tag})")
        return {
            "tags": np.asarray(info["node_ids"]).astype(object),
            "coords": np.asarray(info["node_coords"], dtype=np.float64),
        }

    def get_elements(self, dim: int, tag: int) -> dict:
        """Return ``{'element_ids': ndarray, 'connectivity': ndarray(E,npe)}``."""
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(f"No mesh selection (dim={dim}, tag={tag})")
        eids = info.get("element_ids")
        conn = info.get("connectivity")
        if eids is None or conn is None:
            name = info.get("name", f"(dim={dim}, tag={tag})")
            raise ValueError(
                f"Mesh selection '{name}' has no element data. "
                f"Element data is only available for dim >= 1 sets."
            )
        return {
            "element_ids": np.asarray(eids).astype(object),
            "connectivity": np.asarray(conn).astype(object),
        }

    # ------------------------------------------------------------------
    # Set algebra
    # ------------------------------------------------------------------

    def union(
        self, dim: int, tag_a: int, tag_b: int,
        *, name: str = "", tag: int = -1,
    ) -> int:
        """Create a new set = A ∪ B."""
        a = self._sets.get((dim, tag_a))
        b = self._sets.get((dim, tag_b))
        if a is None or b is None:
            raise KeyError(f"Set (dim={dim}, tag={tag_a}) or tag={tag_b} not found")
        if dim == 0:
            ids = np.union1d(a["node_ids"], b["node_ids"])
            return self.add(0, list(ids), name=name, tag=tag)
        ids = np.union1d(a["element_ids"], b["element_ids"])
        return self.add(dim, list(ids), name=name, tag=tag)

    def intersection(
        self, dim: int, tag_a: int, tag_b: int,
        *, name: str = "", tag: int = -1,
    ) -> int:
        """Create a new set = A ∩ B."""
        a = self._sets.get((dim, tag_a))
        b = self._sets.get((dim, tag_b))
        if a is None or b is None:
            raise KeyError(f"Set (dim={dim}, tag={tag_a}) or tag={tag_b} not found")
        if dim == 0:
            ids = np.intersect1d(a["node_ids"], b["node_ids"])
            return self.add(0, list(ids), name=name, tag=tag)
        ids = np.intersect1d(a["element_ids"], b["element_ids"])
        return self.add(dim, list(ids), name=name, tag=tag)

    def difference(
        self, dim: int, tag_a: int, tag_b: int,
        *, name: str = "", tag: int = -1,
    ) -> int:
        """Create a new set = A \\ B."""
        a = self._sets.get((dim, tag_a))
        b = self._sets.get((dim, tag_b))
        if a is None or b is None:
            raise KeyError(f"Set (dim={dim}, tag={tag_a}) or tag={tag_b} not found")
        if dim == 0:
            ids = np.setdiff1d(a["node_ids"], b["node_ids"])
            return self.add(0, list(ids), name=name, tag=tag)
        ids = np.setdiff1d(a["element_ids"], b["element_ids"])
        return self.add(dim, list(ids), name=name, tag=tag)

    # ------------------------------------------------------------------
    # Bridges
    # ------------------------------------------------------------------

    def from_physical(
        self, dim: int, name_or_tag: str | int,
        *, ms_name: str = "", ms_tag: int = -1,
    ) -> int:
        """Import a physical group as a mesh selection set.

        Parameters
        ----------
        dim : dimension of the physical group
        name_or_tag : physical group name or tag
        ms_name : name for the new mesh selection
        ms_tag : tag for the new mesh selection (-1 = auto)

        Returns
        -------
        int — mesh selection tag
        """
        # Resolve name -> tag
        if isinstance(name_or_tag, str):
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(dim):
                try:
                    if gmsh.model.getPhysicalName(pg_dim, pg_tag) == name_or_tag:
                        name_or_tag = pg_tag
                        break
                except Exception:
                    pass
            else:
                raise KeyError(f"Physical group '{name_or_tag}' not found at dim={dim}")

        pg_tag = int(name_or_tag)
        node_tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(dim, pg_tag)
        t = self._alloc_tag(0, ms_tag)
        nids = np.asarray(node_tags, dtype=np.int64)
        ncoords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        label = ms_name or gmsh.model.getPhysicalName(dim, pg_tag)
        self._store_node_set(t, label, nids, ncoords)
        self._log(f"from_physical(dim={dim}, pg_tag={pg_tag}) -> "
                   f"node set tag={t}, {len(nids)} nodes")
        return t

    def from_geometric(
        self,
        selection,
        *,
        kind: str = "nodes",
        name: str = "",
        tag: int = -1,
    ) -> int:
        """Seed a mesh selection from a geometric :class:`Selection`.

        Bridge between the pre-mesh ``g.model.selection`` system and
        the post-mesh ``g.mesh_selection`` system.

        Parameters
        ----------
        selection : Selection
            A geometric selection from ``g.model.selection.select_*``.
        kind : "nodes" or "elements"
            Whether to extract mesh nodes or elements of the entities.
            ``"elements"`` requires homogeneous-dim selection at dim≥1.
        name, tag : identifier for the resulting mesh selection set.

        Returns
        -------
        int — mesh selection set tag

        Example
        -------
        ::

            top = g.model.selection.select_surfaces(on_plane=("z", 10))
            g.mesh_selection.from_geometric(top, name="top_nodes")
        """
        if kind == "nodes":
            data = selection.to_mesh_nodes()
            t = self._alloc_tag(0, tag)
            self._store_node_set(t, name, data['tags'], data['coords'])
            self._log(
                f"from_geometric(kind=nodes) -> tag={t}, "
                f"{len(data['tags'])} nodes"
            )
            return t
        elif kind == "elements":
            data = selection.to_mesh_elements()
            sel_dim = selection._dim
            t = self._alloc_tag(sel_dim, tag)
            # Element set storage
            conn = data['connectivity']
            unique_node_ids = np.unique(conn[conn >= 0]) if conn.size else \
                np.array([], dtype=np.int64)
            self._sets[(sel_dim, t)] = {
                "name": name,
                "node_ids": unique_node_ids,
                "node_coords": np.empty((0, 3), dtype=np.float64),
                "element_ids": data['element_ids'],
                "connectivity": conn,
            }
            self._log(
                f"from_geometric(kind=elements, dim={sel_dim}) -> tag={t}, "
                f"{len(data['element_ids'])} elements"
            )
            return t
        else:
            raise ValueError(f"kind must be 'nodes' or 'elements', got {kind!r}")

    # ------------------------------------------------------------------
    # Refinement (filter / sort an existing set into a new one)
    # ------------------------------------------------------------------

    def filter_set(
        self,
        dim: int,
        tag: int,
        *,
        name: str = "",
        new_tag: int = -1,
        on_plane: tuple | None = None,
        in_box: tuple | list | None = None,
        in_sphere: tuple | None = None,
        closest_to: tuple | None = None,
        count: int = 1,
        predicate: Callable[[np.ndarray], np.ndarray] | None = None,
        inclusive: bool = False,
    ) -> int:
        """Refine an existing set with spatial filters -> create a new set.

        For node sets, filters apply to node coordinates.  For element
        sets, filters apply to element centroids.  All filters are
        AND-combined.

        Parameters
        ----------
        dim, tag : source set identifier
        name, new_tag : identifier for the resulting set
        on_plane, in_box, in_sphere, closest_to, predicate :
            same semantics as :meth:`add_nodes` (``in_box`` is half-open
            on the upper side by default)
        inclusive : if True, ``in_box`` uses a closed upper bound
            (restores the pre-S2 closed-closed behavior).

        Returns
        -------
        int — tag of the new (filtered) set
        """
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No mesh selection (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )

        if dim == 0:
            ids = np.asarray(info["node_ids"])
            coords = np.asarray(info["node_coords"], dtype=np.float64)
        else:
            # Element set: centroids drive the filtering
            elem_ids = np.asarray(info.get("element_ids", []))
            conn = np.asarray(info.get("connectivity"))
            node_ids, node_coords = self._get_mesh_nodes()
            id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
            coords = _flt.element_centroids(conn, id_to_idx, node_coords)
            ids = elem_ids

        mask = np.ones(len(ids), dtype=bool)
        if on_plane is not None:
            axis, value = on_plane[0], on_plane[1]
            atol = on_plane[2] if len(on_plane) > 2 else 1e-6
            mask &= _flt.nodes_on_plane(coords, axis, value, atol)
        if in_box is not None:
            mask &= _flt.nodes_in_box(coords, in_box, inclusive=inclusive)
        if in_sphere is not None:
            cx, cy, cz, r = in_sphere
            mask &= _flt.nodes_in_sphere(coords, (cx, cy, cz), r)
        if closest_to is not None:
            mask &= _flt.nodes_nearest(coords, closest_to, count)
        if predicate is not None:
            mask &= predicate(coords)

        t = self._alloc_tag(dim, new_tag)
        if dim == 0:
            self._store_node_set(t, name, ids[mask], coords[mask])
        else:
            self._sets[(dim, t)] = {
                "name": name,
                "node_ids": np.unique(conn[mask].ravel()),
                "node_coords": np.array([], dtype=np.float64).reshape(0, 3),
                "element_ids": ids[mask],
                "connectivity": conn[mask],
            }
        self._log(
            f"filter_set(dim={dim}, src={tag}) -> tag={t}, "
            f"{int(mask.sum())}/{len(mask)} kept"
        )
        return t

    # ------------------------------------------------------------------
    # Daisy-chainable selection (S3d additive + name-seed follow-on;
    # pre-snapshot live mesh)
    # ------------------------------------------------------------------

    def _seed_ids_by_name(self, dim: int, name: str) -> np.ndarray:
        """Seed ids for an **existing** mesh-selection set ``name``.

        Pure delegation to the surfaces ``MeshSelectionSet`` already
        exposes — :meth:`get_tag` (the ``_sets`` name→tag lookup) plus
        :meth:`get_nodes` / :meth:`get_elements`.  It introduces **no
        new resolver** and **never writes** :attr:`_sets`.

        Fails loud (``KeyError``) when no set of dimension ``dim``
        carries ``name`` — never a silent empty / full-universe seed
        (resolution-contract Rule 6, ``tests/test_resolution_contract``).
        The message names the existing register-then-select route for
        the two name kinds that have **no** non-registering resolution
        surface on ``MeshSelectionSet`` (a gmsh physical group →
        :meth:`from_physical`; a label / geometry selection →
        :meth:`from_geometric`), so the contract that a mesh-selection
        name is *not* a geometry-resolver tier
        (``test_meshselection_name_is_not_a_helpers_tier``) is upheld.
        """
        tag = self.get_tag(dim, name)
        if tag is None:
            avail = sorted(
                nm for (d, t) in self.get_all(dim)
                if (nm := self.get_name(d, t))
            )
            where = "node" if dim == 0 else f"element dim={dim}"
            raise KeyError(
                f"No mesh-selection set named {name!r} at the {where} "
                f"level. select(name=) resolves an EXISTING "
                f"g.mesh_selection set via the get_tag/_sets surface; "
                f"to seed from a gmsh physical group call "
                f"from_physical(...) first, or from a label / geometry "
                f"selection call from_geometric(...) first, then "
                f"select(name=...). Available {where} sets: {avail}"
            )
        if dim == 0:
            return self.get_nodes(0, tag)["tags"]
        return self.get_elements(dim, tag)["element_ids"]

    def select(
        self,
        *,
        level: str = "node",
        dim: int = 2,
        ids=None,
        name=None,
    ):
        """Start a daisy-chainable selection over the **live** mesh.

        Returns a
        :class:`~apeGmsh.mesh._mesh_selection_chain.MeshSelectionChain`
        (point family) — the fluent equivalent of the eager
        :meth:`add_nodes` / :meth:`add_elements`::

            g.mesh_selection.select().in_box(lo, hi).on_plane(p, n, tol=1e-6)
            g.mesh_selection.select(level="element", dim=3).in_box(lo, hi)
            g.mesh_selection.select(ids=a) | g.mesh_selection.select(ids=b)
            g.mesh_selection.select(name="base").in_sphere(c, r)

        The chain seeds its atoms and then the standard point-family
        verbs (``in_box`` / ``on_plane`` / ``in_sphere`` /
        ``nearest_to`` / ``where`` + ``| & - ^``) narrow it, operating
        on the same live-mesh coordinates the eager API uses (node
        coords for ``level="node"``, element centroids for
        ``level="element"``) — so
        ``select().in_box(b).on_plane(p, n, tol=t)`` selects the same
        nodes/elements as ``add_nodes(in_box=b, on_plane=(...))``.

        Parameters
        ----------
        level : ``"node"`` (default) — atoms are mesh node ids; or
            ``"element"`` — atoms are element ids for ``dim``.
        dim : element dimension (1/2/3) used when
            ``level == "element"`` (ignored for the node level);
            mirrors :meth:`add_elements`'s ``dim``.
        ids : optional explicit id list to seed from.  Omitted (and
            no ``name``) → the full live-mesh node universe
            (``level="node"``) or the full live-mesh element universe
            of ``dim`` (``level="element"``), exactly the universe
            :meth:`add_nodes` / :meth:`add_elements` start from before
            their filters.
        name : optional name of an **existing** ``g.mesh_selection``
            set to seed from (its node ids for ``level="node"``, its
            element ids for ``level="element"``), so
            ``select(name=N).<spatial>`` is the id-for-id fluent
            equivalent of :meth:`filter_set` over that set.  Mutually
            exclusive with ``ids=``.  Resolution **delegates verbatim**
            to the existing :meth:`get_tag` / :meth:`get_nodes` /
            :meth:`get_elements` surface (no new resolver); an unknown
            name fails loud.

        Notes
        -----
        ``.select()`` is **additive**: it does not register a set into
        :attr:`_sets`, does not allocate a tag, and does not perturb
        the eager API (``name=`` only *reads* :attr:`_sets`).
        ``MeshSelectionChain`` is imported **deferred** (mirrors
        ``mesh/_mesh_structured.py``); the chain module imports only
        the package-root leaf ``apeGmsh._chain`` + numpy, so this adds
        no eager cross-package edge
        (``tests/test_import_dag_polarity.py`` baseline unchanged).

        ``name=`` seeds from an **already-registered** mesh-selection
        set only.  Seeding *directly* from a raw gmsh physical-group
        name or an apeGmsh label is **not** a ``select()`` parameter:
        ``MeshSelectionSet`` has no non-registering, non-reimplementing
        resolver for those — its only PG bridge,
        :meth:`from_physical`, *registers* a set + allocates a tag
        (which ``select()`` must not do) and is node-only; label /
        geometry resolution lives off ``MeshSelectionSet`` entirely,
        and a mesh-selection name is deliberately *not* a
        geometry-resolver tier (see
        ``docs/plans/selection-unification.md`` §9 and
        ``tests/test_resolution_contract.py``).  The supported,
        existing-surface route is the documented two step
        ``from_physical(dim, "PG", ms_name="foo")`` /
        ``from_geometric(sel, name="foo")`` **then**
        ``select(name="foo")``.  Persistence (``.save_as(name)`` /
        round-trip as ``selection=``) remains deferred and out of
        S3d scope.
        """
        from ._mesh_selection_chain import (  # deferred — see plan §3
            MeshSelectionChain,
            engine_for,
        )

        if level not in ("node", "element"):
            raise ValueError(
                f"select(level=) must be 'node' or 'element', "
                f"got {level!r}."
            )
        if ids is not None and name is not None:
            raise ValueError(
                "select(ids=, name=) are mutually exclusive — pass an "
                "explicit id list OR an existing selection-set name, "
                "not both."
            )

        if level == "node":
            eng = engine_for(self, "node", 0)
            if ids is not None:
                atoms = [int(n) for n in ids]
            elif name is not None:
                atoms = [int(n) for n in self._seed_ids_by_name(0, name)]
            else:
                all_ids, _ = self._get_mesh_nodes()
                atoms = [int(n) for n in all_ids]
        else:
            d = int(dim)
            eng = engine_for(self, "element", d)
            if ids is not None:
                atoms = [int(e) for e in ids]
            elif name is not None:
                atoms = [int(e) for e in self._seed_ids_by_name(d, name)]
            else:
                elem_ids, _ = self._get_mesh_elements(d)
                atoms = [int(e) for e in elem_ids]

        return MeshSelectionChain(atoms, _engine=eng)

    def sort_set(
        self,
        dim: int,
        tag: int,
        *,
        by: str = "x",
        descending: bool = False,
    ) -> None:
        """Sort the entries of a set in place by coordinate axis.

        Parameters
        ----------
        dim, tag : set identifier
        by : "x", "y", or "z" — axis to sort along
        descending : reverse the order
        """
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(f"No mesh selection (dim={dim}, tag={tag}).")

        axis_idx = {"x": 0, "y": 1, "z": 2}[by.lower()]

        if dim == 0:
            coords = np.asarray(info["node_coords"], dtype=np.float64)
            order = np.argsort(coords[:, axis_idx])
            if descending:
                order = order[::-1]
            info["node_ids"] = np.asarray(info["node_ids"])[order]
            info["node_coords"] = coords[order]
        else:
            conn = np.asarray(info.get("connectivity"))
            elem_ids = np.asarray(info.get("element_ids", []))
            node_ids, node_coords = self._get_mesh_nodes()
            id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
            cents = _flt.element_centroids(conn, id_to_idx, node_coords)
            order = np.argsort(cents[:, axis_idx])
            if descending:
                order = order[::-1]
            info["element_ids"] = elem_ids[order]
            info["connectivity"] = conn[order]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """DataFrame describing all mesh selection sets."""
        rows: list[dict] = []
        for (dim, t), info in sorted(self._sets.items()):
            eids = info.get("element_ids")
            rows.append({
                "dim": dim,
                "tag": t,
                "name": info.get("name", ""),
                "n_nodes": len(info["node_ids"]),
                "n_elems": len(eids) if eids is not None else 0,
            })
        if not rows:
            return pd.DataFrame(columns=["dim", "tag", "name", "n_nodes", "n_elems"])
        return pd.DataFrame(rows).set_index(["dim", "tag"]).sort_index()

    def to_dataframe(self, dim: int, tag: int) -> pd.DataFrame:
        """Return a DataFrame of the entries of a single set.

        For dim=0 (node set): columns ``[node_id, x, y, z]``.
        For dim>0 (element set): columns ``[element_id, cx, cy, cz, n_nodes]``.
        """
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No mesh selection (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        if dim == 0:
            return pd.DataFrame({
                "node_id": np.asarray(info["node_ids"]),
                "x": info["node_coords"][:, 0],
                "y": info["node_coords"][:, 1],
                "z": info["node_coords"][:, 2],
            })
        elem_ids = np.asarray(info.get("element_ids", []))
        conn = np.asarray(info.get("connectivity"))
        node_ids, node_coords = self._get_mesh_nodes()
        id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
        cents = _flt.element_centroids(conn, id_to_idx, node_coords)
        return pd.DataFrame({
            "element_id": elem_ids,
            "cx": cents[:, 0],
            "cy": cents[:, 1],
            "cz": cents[:, 2],
            "n_nodes": [conn.shape[1]] * len(elem_ids),
        })

    # ------------------------------------------------------------------
    # Snapshot (for FEMData)
    # ------------------------------------------------------------------

    def _snapshot(self) -> "MeshSelectionStore":
        """Return an immutable copy of all sets for FEMData."""
        import copy
        return MeshSelectionStore(copy.deepcopy(self._sets))

    def __len__(self) -> int:
        return len(self._sets)

    def __repr__(self) -> str:
        return f"MeshSelectionSet({len(self._sets)} sets)"


# ======================================================================
# MeshSelectionStore — immutable snapshot (lives on fem.mesh_selection)
# ======================================================================

class MeshSelectionStore:
    """Immutable snapshot of mesh selections captured at ``get_fem_data()`` time.

    Accessed via ``fem.mesh_selection``.  Mirrors the query API of
    :class:`MeshSelectionSet` and :class:`PhysicalGroupSet`.

    Example
    -------
    ::

        fem = g.mesh.queries.get_fem_data(dim=2)
        fem.mesh_selection.get_all()
        fem.mesh_selection.get_nodes(0, 1)
        fem.mesh_selection.get_elements(2, 1)
        fem.mesh_selection.summary()
    """

    def __init__(self, sets: dict[DimTag, dict]) -> None:
        self._sets = sets

    # ── Queries ───────────────────────────────────────────────

    def get_all(self, dim: int = -1) -> list[DimTag]:
        if dim == -1:
            return sorted(self._sets.keys())
        return sorted(k for k in self._sets if k[0] == dim)

    def get_name(self, dim: int, tag: int) -> str:
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No mesh selection (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        return info.get("name", "")

    def get_tag(self, dim: int, name: str) -> int | None:
        for (d, t), info in self._sets.items():
            if d == dim and info.get("name", "") == name:
                return t
        return None

    # ── Mesh data ─────────────────────────────────────────────

    def get_nodes(self, dim: int, tag: int) -> dict:
        """Return ``{'tags': ndarray, 'coords': ndarray(N,3)}``."""
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No mesh selection (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        return {
            "tags": np.asarray(info["node_ids"]).astype(object),
            "coords": np.asarray(info["node_coords"], dtype=np.float64),
        }

    def get_elements(self, dim: int, tag: int) -> dict:
        """Return ``{'element_ids': ndarray, 'connectivity': ndarray(E,npe)}``."""
        info = self._sets.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No mesh selection (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        eids = info.get("element_ids")
        conn = info.get("connectivity")
        if eids is None or conn is None:
            name = info.get("name", f"(dim={dim}, tag={tag})")
            raise ValueError(
                f"Mesh selection '{name}' has no element data. "
                f"Element data is only available for dim >= 1 sets."
            )
        return {
            "element_ids": np.asarray(eids).astype(object),
            "connectivity": np.asarray(conn).astype(object),
        }

    # ── Name-based lookups (mirrors PhysicalGroupSet API) ─────

    def names(self, dim: int = -1) -> list[str]:
        """All non-empty selection names, optionally filtered by dim."""
        out = []
        for (d, _), info in sorted(self._sets.items()):
            n = info.get("name", "")
            if n and (dim == -1 or d == dim):
                out.append(n)
        return out

    def node_ids(self, name: str) -> np.ndarray:
        """Node IDs for any selection matching ``name`` (any dim).

        For ``dim=0`` selections returns the explicit node IDs.
        For ``dim>=1`` selections returns the node IDs implied by the
        elements' connectivity.

        Raises ``KeyError`` if no selection matches ``name``.
        """
        for (dim, tag), info in self._sets.items():
            if info.get("name", "") == name:
                return np.asarray(info["node_ids"], dtype=np.int64)
        raise KeyError(
            f"No mesh selection named {name!r}. "
            f"Available: {self.names()}"
        )

    def element_ids(self, name: str) -> np.ndarray:
        """Element IDs for an element-bearing selection matching ``name``.

        Searches ``dim>=1`` selections only (``dim=0`` selections have
        no element data).

        Raises ``KeyError`` if no element-bearing selection matches.
        """
        for (dim, tag), info in self._sets.items():
            if dim < 1:
                continue
            if info.get("name", "") == name:
                eids = info.get("element_ids")
                if eids is None:
                    continue
                return np.asarray(eids, dtype=np.int64)
        raise KeyError(
            f"No element-bearing mesh selection named {name!r}. "
            f"Available (dim>=1): "
            f"{self.names(1) + self.names(2) + self.names(3)}"
        )

    # ── Display ───────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        rows: list[dict] = []
        for (dim, t), info in sorted(self._sets.items()):
            eids = info.get("element_ids")
            rows.append({
                "dim": dim,
                "tag": t,
                "name": info.get("name", ""),
                "n_nodes": len(info["node_ids"]),
                "n_elems": len(eids) if eids is not None else 0,
            })
        if not rows:
            return pd.DataFrame(columns=["dim", "tag", "name", "n_nodes", "n_elems"])
        return pd.DataFrame(rows).set_index(["dim", "tag"]).sort_index()

    def __len__(self) -> int:
        return len(self._sets)

    def __repr__(self) -> str:
        return f"MeshSelectionStore({len(self._sets)} sets)"
