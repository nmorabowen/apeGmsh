"""
FEMData — Solver-ready mesh data container.
============================================

Returned by ``Mesh.get_fem_data()``, this module holds everything
needed to build a solver model: node IDs, coordinates, element IDs,
connectivity, mesh statistics, and physical group snapshots.

The data is fully self-contained — once extracted, no live Gmsh
session is required.

Classes
-------
FEMData
    Top-level container with ``.info`` and ``.physical`` sub-objects.
MeshInfo
    Read-only mesh statistics (n_nodes, n_elems, bandwidth).
PhysicalGroupSet
    Snapshot of physical groups mirroring the ``g.physical`` API.

Usage
-----
::

    g.mesh.renumber_mesh(method="rcm", base=1)
    fem = g.mesh.get_fem_data(dim=2)

    # Mesh statistics
    print(fem.info)

    # Physical groups
    fem.physical.get_all()
    fem.physical.get_name(0, 1)
    base = fem.physical.get_nodes(0, 1)

    # Build solver model
    for i in range(fem.info.n_nodes):
        ops.node(int(fem.node_ids[i]),
                 *fem.node_coords[i])

    for i in range(fem.info.n_elems):
        ops.element('ShellMITC4', int(fem.element_ids[i]),
                    *fem.connectivity[i], sec_tag)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


# =====================================================================
# Bandwidth computation
# =====================================================================

def _compute_bandwidth(connectivity: ndarray) -> int:
    """
    Compute the semi-bandwidth of the mesh.

    bandwidth = max over all elements of (max_node_id - min_node_id)

    This is the semi-bandwidth of the assembled stiffness matrix.
    Lower is better for direct solvers.
    """
    if connectivity.size == 0:
        return 0
    row_max = connectivity.max(axis=1)
    row_min = connectivity.min(axis=1)
    return int((row_max - row_min).max())


# =====================================================================
# Mesh info
# =====================================================================

class MeshInfo:
    """
    Read-only summary of mesh statistics.

    Accessed via ``fem.info``.

    Attributes
    ----------
    n_nodes : int
    n_elems : int
    bandwidth : int
    """

    __slots__ = ('n_nodes', 'n_elems', 'bandwidth')

    def __init__(self, n_nodes: int, n_elems: int, bandwidth: int) -> None:
        object.__setattr__(self, 'n_nodes', n_nodes)
        object.__setattr__(self, 'n_elems', n_elems)
        object.__setattr__(self, 'bandwidth', bandwidth)

    def __repr__(self) -> str:
        return (
            f"MeshInfo(n_nodes={self.n_nodes}, n_elems={self.n_elems}, "
            f"bandwidth={self.bandwidth})"
        )

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"{self.n_nodes} nodes, {self.n_elems} elements, "
            f"bandwidth={self.bandwidth}"
        )


# =====================================================================
# Physical group snapshot
# =====================================================================

class PhysicalGroupSet:
    """
    Snapshot of physical groups captured at ``get_fem_data()`` time.

    Accessed via ``fem.physical``.  Mirrors the query API of
    :class:`PhysicalGroups` (``g.physical``) but is fully
    self-contained — no live Gmsh session required after construction.

    Provides:

    * **Queries** — ``get_all``, ``get_name``, ``get_tag``, ``summary``
    * **Mesh nodes** — ``get_nodes``
    * **Mesh elements** — ``get_elements``

    Example
    -------
    ::

        fem = g.mesh.get_fem_data(dim=2)

        # Inspect
        fem.physical.get_all()           # [(0, 1), (1, 2), (2, 3)]
        fem.physical.get_name(0, 1)      # "base_supports"
        fem.physical.get_tag(0, "base_supports")  # 1
        fem.physical.summary()           # DataFrame

        # Retrieve nodes
        nodes = fem.physical.get_nodes(0, 1)
        nodes['tags']    # ndarray of node IDs
        nodes['coords']  # ndarray(N, 3)

        # Retrieve elements
        elems = fem.physical.get_elements(2, 3)
        elems['element_ids']    # ndarray(E,)
        elems['connectivity']   # ndarray(E, npe)
    """

    def __init__(
        self,
        groups: dict[tuple[int, int], dict],
    ) -> None:
        # groups: {(dim, pg_tag): {'name': str,
        #                          'node_ids': ndarray,
        #                          'node_coords': ndarray,
        #                          'element_ids': ndarray (optional),
        #                          'connectivity': ndarray (optional)}}
        self._groups = groups

    # ── Queries ───────────────────────────────────────────────

    def get_all(self, dim: int = -1) -> list[tuple[int, int]]:
        """
        Return all physical groups as ``(dim, tag)`` pairs.

        Parameters
        ----------
        dim : filter to a single dimension (``-1`` = all)
        """
        if dim == -1:
            return sorted(self._groups.keys())
        return sorted(k for k in self._groups if k[0] == dim)

    def get_name(self, dim: int, tag: int) -> str:
        """
        Return the name of a physical group, or ``""`` if unnamed.

        Parameters
        ----------
        dim : dimension of the physical group
        tag : physical-group tag
        """
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No physical group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        return info.get('name', '')

    def get_tag(self, dim: int, name: str) -> int | None:
        """
        Look up the tag of a named physical group.

        Returns ``None`` if no group with that name and dimension exists.

        Parameters
        ----------
        dim  : dimension to search
        name : human-readable label
        """
        for (d, pg_tag), info in self._groups.items():
            if d == dim and info.get('name', '') == name:
                return pg_tag
        return None

    # ── Mesh nodes ────────────────────────────────────────────

    def get_nodes(self, dim: int, tag: int) -> dict:
        """
        Return the mesh nodes belonging to a physical group.

        Parameters
        ----------
        dim : dimension of the physical group
        tag : physical-group tag

        Returns
        -------
        dict
            ``'tags'``   : ndarray(N,)   — node IDs (Python ``int``)
            ``'coords'`` : ndarray(N, 3) — XYZ coordinates (``float``)

        Note
        ----
        Tags are returned as ``object`` dtype so iteration yields
        Python ``int``, not ``numpy.int64`` — safe for OpenSees.
        """
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No physical group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        return {
            'tags':   np.asarray(info['node_ids']).astype(object),
            'coords': np.asarray(info['node_coords'], dtype=np.float64),
        }

    # ── Mesh elements ────────────────────────────────────────

    def get_elements(self, dim: int, tag: int) -> dict:
        """
        Return element IDs and connectivity for a physical group.

        Only available for physical groups with ``dim >= 1`` (curves,
        surfaces, volumes).  Point groups (``dim=0``) have no elements.

        Parameters
        ----------
        dim : dimension of the physical group
        tag : physical-group tag

        Returns
        -------
        dict
            ``'element_ids'``    : ndarray(E,)      — element IDs
            ``'connectivity'``   : ndarray(E, npe)   — node IDs per element

        Raises
        ------
        KeyError
            If the physical group does not exist.
        ValueError
            If the physical group has no element data (dim=0 groups,
            or data was not captured at extraction time).
        """
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No physical group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}"
            )
        elem_ids = info.get('element_ids')
        conn     = info.get('connectivity')
        if elem_ids is None or conn is None:
            name = info.get('name', f'(dim={dim}, tag={tag})')
            raise ValueError(
                f"Physical group '{name}' has no element data. "
                f"Element data is only available for dim >= 1 groups."
            )
        return {
            'element_ids':  np.asarray(elem_ids).astype(object),
            'connectivity': np.asarray(conn).astype(object),
        }

    # ── Display ───────────────────────────────────────────────

    def summary(self):
        """
        Build a DataFrame describing every physical group.

        Returns
        -------
        pd.DataFrame  indexed by ``(dim, pg_tag)`` with columns:

        ``name``       label (empty string if unnamed)
        ``n_nodes``    number of mesh nodes in the group
        ``n_elems``    number of elements in the group (0 for dim=0)
        """
        import pandas as pd

        rows: list[dict] = []
        for (dim, pg_tag), info in sorted(self._groups.items()):
            elem_ids = info.get('element_ids')
            rows.append({
                'dim':     dim,
                'pg_tag':  pg_tag,
                'name':    info.get('name', ''),
                'n_nodes': len(info['node_ids']),
                'n_elems': len(elem_ids) if elem_ids is not None else 0,
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'pg_tag', 'name', 'n_nodes', 'n_elems']
            )

        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'pg_tag'])
            .sort_index()
        )

    def __len__(self) -> int:
        return len(self._groups)

    def __repr__(self) -> str:
        return f"PhysicalGroupSet({len(self._groups)} groups)"


# =====================================================================
# FEM data container
# =====================================================================

@dataclass
class FEMData:
    """
    Solver-ready FEM mesh data.

    Returned by ``Mesh.get_fem_data()``.  Contains everything needed
    to build a solver model: node IDs, coordinates, element IDs,
    connectivity, mesh statistics, and physical group data.

    Attributes
    ----------
    node_ids : ndarray(N,)
        Node IDs (contiguous if ``renumber_mesh()`` was called first).
    node_coords : ndarray(N, 3)
        Nodal coordinates, same order as ``node_ids``.
    element_ids : ndarray(E,)
        Element IDs (contiguous if ``renumber_mesh()`` was called first).
    connectivity : ndarray(E, npe)
        Element-to-node connectivity in terms of ``node_ids``.
    info : MeshInfo
        Mesh statistics: ``n_nodes``, ``n_elems``, ``bandwidth``.
    physical : PhysicalGroupSet
        Physical group introspection and node retrieval.

    Example
    -------
    ::

        g.mesh.renumber_mesh(method="rcm", base=1)
        fem = g.mesh.get_fem_data(dim=2)

        # Mesh stats
        print(fem.info)

        # Physical groups (mirrors g.physical API)
        fem.physical.get_all()              # [(0, 1), (1, 2), (2, 3)]
        fem.physical.get_name(0, 1)         # "base_supports"
        base = fem.physical.get_nodes(0, 1) # {'tags': ..., 'coords': ...}
        fem.physical.summary()              # DataFrame

        # Build solver model
        for i in range(fem.info.n_nodes):
            ops.node(int(fem.node_ids[i]), *fem.node_coords[i])

        # Quick coordinate lookup by node ID
        x, y, z = fem.get_node_coords(42)

        # Array index lookup (node or element)
        idx = fem.node_index(42)
        fem.node_coords[idx]       # same as above

        # Elements by physical group
        cols = fem.physical.get_elements(1, 2)   # columns (dim=1)
        slab = fem.physical.get_elements(2, 3)   # slab (dim=2)
        cols['element_ids'], cols['connectivity']
        slab['element_ids'], slab['connectivity']
    """
    node_ids:      ndarray
    node_coords:   ndarray
    element_ids:   ndarray
    connectivity:  ndarray
    info:          MeshInfo          = field(repr=False)
    physical:      PhysicalGroupSet  = field(repr=False)

    # ── Lazy lookup caches (not part of __init__) ────────────

    _node_id_to_idx: dict = field(default=None, init=False, repr=False)
    _elem_id_to_idx: dict = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Cast ID/connectivity arrays to ``object`` dtype so iteration
        yields Python ``int``, not ``numpy.int64``.

        OpenSees and other C-extension solvers reject numpy integer
        types.  ``object`` dtype keeps full numpy functionality
        (``np.isin``, ``np.unique``, arithmetic) while making
        ``for x in arr`` yield plain ``int``.
        """
        object.__setattr__(
            self, 'node_ids', np.asarray(self.node_ids).astype(object))
        object.__setattr__(
            self, 'element_ids', np.asarray(self.element_ids).astype(object))
        object.__setattr__(
            self, 'connectivity', np.asarray(self.connectivity).astype(object))
        object.__setattr__(
            self, 'node_coords',
            np.asarray(self.node_coords, dtype=np.float64))

    # ── Solver-friendly iterators ────────────────────────────

    def nodes(self):
        """Iterate ``(node_id, coords)`` as Python-native types.

        Yields ``(int, list[float])`` — safe for OpenSees.

        Example
        -------
        ::

            for nid, xyz in fem.nodes():
                ops.node(nid, *xyz)
        """
        for i in range(len(self.node_ids)):
            yield int(self.node_ids[i]), self.node_coords[i].tolist()

    def elements(self):
        """Iterate ``(element_id, connectivity)`` as Python-native types.

        Yields ``(int, list[int])`` — safe for OpenSees.

        Example
        -------
        ::

            for eid, conn in fem.elements():
                ops.element("tri31", eid, *conn, thk, "PlaneStrain", 1)
        """
        for i in range(len(self.element_ids)):
            yield int(self.element_ids[i]), [int(n) for n in self.connectivity[i]]

    # ── Lookup maps ──────────────────────────────────────────

    def _build_node_map(self) -> dict[int, int]:
        """Build and cache the node-ID → array-index map."""
        if self._node_id_to_idx is None:
            self._node_id_to_idx = {
                int(nid): i for i, nid in enumerate(self.node_ids)
            }
        return self._node_id_to_idx

    def _build_elem_map(self) -> dict[int, int]:
        """Build and cache the element-ID → array-index map."""
        if self._elem_id_to_idx is None:
            self._elem_id_to_idx = {
                int(eid): i for i, eid in enumerate(self.element_ids)
            }
        return self._elem_id_to_idx

    # ── Node lookups ─────────────────────────────────────────

    def node_index(self, nid: int) -> int:
        """
        Return the array index for a node ID.

        Parameters
        ----------
        nid : int
            Node ID (as stored in ``node_ids``).

        Returns
        -------
        int
            Index into ``node_ids``, ``node_coords``.

        Raises
        ------
        KeyError
            If ``nid`` is not in ``node_ids``.

        Example
        -------
        ::

            idx = fem.node_index(42)
            fem.node_coords[idx]   # → array([x, y, z])
        """
        m = self._build_node_map()
        try:
            return m[int(nid)]
        except KeyError:
            raise KeyError(
                f"Node ID {nid} not found. "
                f"Valid range: {int(self.node_ids.min())}–"
                f"{int(self.node_ids.max())} "
                f"({len(self.node_ids)} nodes)"
            ) from None

    def get_node_coords(self, nid: int) -> ndarray:
        """
        Return the coordinates of a node by its ID.

        Parameters
        ----------
        nid : int
            Node ID (as stored in ``node_ids``).

        Returns
        -------
        ndarray(3,)
            ``[x, y, z]`` coordinates.

        Example
        -------
        ::

            x, y, z = fem.get_node_coords(42)
        """
        return self.node_coords[self.node_index(nid)]

    # ── Element lookups ──────────────────────────────────────

    def elem_index(self, eid: int) -> int:
        """
        Return the array index for an element ID.

        Parameters
        ----------
        eid : int
            Element ID (as stored in ``element_ids``).

        Returns
        -------
        int
            Index into ``element_ids``, ``connectivity``.

        Raises
        ------
        KeyError
            If ``eid`` is not in ``element_ids``.

        Example
        -------
        ::

            idx = fem.elem_index(10)
            fem.connectivity[idx]   # → array([n1, n2, n3])
        """
        m = self._build_elem_map()
        try:
            return m[int(eid)]
        except KeyError:
            raise KeyError(
                f"Element ID {eid} not found. "
                f"Valid range: {int(self.element_ids.min())}–"
                f"{int(self.element_ids.max())} "
                f"({len(self.element_ids)} elements)"
            ) from None

    def get_elem_connectivity(self, eid: int) -> ndarray:
        """
        Return the connectivity (node IDs) of an element by its ID.

        Parameters
        ----------
        eid : int
            Element ID (as stored in ``element_ids``).

        Returns
        -------
        ndarray(npe,)
            Node IDs forming the element.

        Example
        -------
        ::

            n1, n2, n3 = fem.get_elem_connectivity(10)
        """
        return self.connectivity[self.elem_index(eid)]