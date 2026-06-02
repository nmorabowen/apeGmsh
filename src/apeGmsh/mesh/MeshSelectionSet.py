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
        """Build a named mesh-selection set from the live mesh.

        Returns a :class:`~apeGmsh.mesh._mesh_selection.MeshSelection`
        (point family) that chains spatial-refinement verbs.  Finish
        with ``.save_as(name)`` to register the result — it then
        snapshots into ``FEMData`` and round-trips via HDF5::

            # capture all base nodes and persist
            (g.mesh_selection.select()
                 .in_box((0, 0, 0), (10, 10, 0.01))
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
                 .save_as("base_nodes"))

            # 3-D solid elements in a region (dim=3 required for tets)
            (g.mesh_selection.select(level="element", dim=3)
                 .in_box(lo, hi)
                 .save_as("core_tets"))

            # narrow an existing set further
            g.mesh_selection.select(name="base_nodes").in_sphere(c, r)

            # set algebra
            g.mesh_selection.select(ids=a) | g.mesh_selection.select(ids=b)

        No arguments seeds the full live-mesh universe for the chosen
        ``level`` and ``dim``.

        Parameters
        ----------
        level : ``"node"`` | ``"element"``
            ``"node"`` (default) — atoms are node ids.
            ``"element"`` — atoms are element ids for ``dim``.
        dim :
            Element dimension (1=line, 2=surface, **3=volume**) used
            when ``level="element"`` (ignored for nodes).
            Defaults to 2 (surface elements) — **pass** ``dim=3``
            **for solid/tet meshes.**
        ids :
            Explicit id list to seed from.  Mutually exclusive
            with ``name=``.
        name :
            Name of an **existing** ``g.mesh_selection`` set to
            seed from.  Node ids are used for ``level="node"``,
            element ids for ``level="element"``.  Mutually
            exclusive with ``ids=``.  An unknown name raises.

        Refining verbs
        --------------
        Each returns a new ``MeshSelection`` and composes freely:

        - ``.in_box(lo, hi, *, inclusive=False)``
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — ``tol=`` is
          **required**; raises ``TypeError`` if omitted.
        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)``
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminals
        ---------
        - ``.ids`` — ``list[int]``.
        - ``.coords`` — ``ndarray (N, 3)``.
        - ``.save_as(name)`` — registers the narrowed id set as a
          named mesh-selection, so it snapshots into ``FEMData``
          and round-trips via HDF5 as ``selection=``.

        Notes
        -----
        To seed from a physical group or label, register it first::

            g.mesh_selection.from_physical(2, "BottomFace",
                                           ms_name="bottom")
            g.mesh_selection.select(name="bottom").in_sphere(c, r)
        """
        # selection-unification-v2 P2-I (§6.1 STOP-2): return the v2
        # terminal ``MeshSelection`` (legacy ``MeshSelectionChain`` left
        # defined-but-unwired; P3 deletes it).  ``engine_for`` is kept
        # verbatim — the per-``(level, dim)`` memoised ``_LiveMeshEngine``
        # singleton is what ``SelectionChain._compatible`` gates
        # set-algebra identity on, and ``MeshSelection`` delegates the
        # per-engine live-mesh read back to a ``MeshSelectionChain``
        # built from this exact same engine (byte-faithful).  Same
        # deferred-import idiom; ``_mesh_selection`` imports only the
        # package-root leaf ``_kernel.chain`` at load (one declared
        # downward BASELINE triple; no new eager cross-package edge).
        from ._live_engine import engine_for            # deferred — plan §3
        from ._mesh_selection import MeshSelection      # deferred — plan §3

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

        return MeshSelection(atoms, _engine=eng)

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
