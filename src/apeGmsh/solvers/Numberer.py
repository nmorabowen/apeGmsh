"""
Numberer — Contiguous, solver-ready node & element renumbering.
================================================================

Gmsh produces non-contiguous tags (1, 5, 47, 200…).  FEM solvers
need either contiguous IDs or — better yet — optimally ordered IDs
that minimise the bandwidth of the stiffness matrix.

This module provides:

* **Simple** renumbering:  Gmsh tags -> contiguous 1-based IDs
  (preserving relative order).
* **RCM** (Reverse Cuthill-McKee):  bandwidth-optimal ordering
  for direct solvers (Cholesky, LDL, skyline).
* **Bidirectional maps** so post-processing can translate solver
  results back to Gmsh tags for visualisation.
* **Rewritten connectivity** in terms of the new IDs — ready to
  feed directly to ``ops.node()`` / ``ops.element()`` calls.

Usage
-----
::

    fem  = g.mesh.queries.get_fem_data(dim=2)
    numb = Numberer(fem)

    # Contiguous (default)
    data = numb.renumber()

    # Bandwidth-optimised
    data = numb.renumber(method="rcm")

    # Use in OpenSees
    for i in range(data.n_nodes):
        ops.node(data.node_ids[i],
                 *data.node_coords[i])

    for i in range(data.n_elems):
        ops.element('ShellMITC4', data.elem_ids[i],
                    *data.connectivity[i], sec_tag)

    # Post-processing: translate solver ID back to Gmsh tag
    gmsh_tag = data.solver_to_gmsh_node[solver_id]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


# =====================================================================
# Result container
# =====================================================================

@dataclass
class NumberedMesh:
    """
    Solver-ready mesh with contiguous IDs and bidirectional maps.

    All IDs are **1-based** (the standard in structural FEM solvers
    like OpenSees, Abaqus, SAP2000).  Set ``base=0`` in
    :meth:`Numberer.renumber` for 0-based if your solver needs it.

    Attributes
    ----------
    node_ids : ndarray(N,)
        New contiguous node IDs.
    node_coords : ndarray(N, 3)
        Nodal coordinates, same order as ``node_ids``.
    elem_ids : ndarray(E,)
        New contiguous element IDs.
    connectivity : ndarray(E, npe)
        Element connectivity in terms of new node IDs.
    n_nodes : int
    n_elems : int
    bandwidth : int
        Semi-bandwidth of the resulting adjacency.
    method : str
        Numbering method used (``"simple"`` or ``"rcm"``).

    Maps
    ~~~~
    gmsh_to_solver_node : dict[int, int]
        Gmsh node tag -> solver node ID.
    solver_to_gmsh_node : dict[int, int]
        Solver node ID -> Gmsh node tag.
    gmsh_to_solver_elem : dict[int, int]
        Gmsh element tag -> solver element ID.
    solver_to_gmsh_elem : dict[int, int]
        Solver element ID -> Gmsh element tag.
    """
    node_ids: ndarray
    node_coords: ndarray
    elem_ids: ndarray
    connectivity: ndarray
    n_nodes: int = 0
    n_elems: int = 0
    bandwidth: int = 0
    method: str = "simple"

    gmsh_to_solver_node: dict[int, int] = field(default_factory=dict)
    solver_to_gmsh_node: dict[int, int] = field(default_factory=dict)
    gmsh_to_solver_elem: dict[int, int] = field(default_factory=dict)
    solver_to_gmsh_elem: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"NumberedMesh({self.method}): "
            f"{self.n_nodes} nodes, {self.n_elems} elements, "
            f"bandwidth={self.bandwidth}"
        )


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
# RCM (Reverse Cuthill-McKee) — pure numpy, no scipy needed
# =====================================================================

def _build_adjacency(
    n_nodes: int,
    connectivity: ndarray,
) -> list[list[int]]:
    """
    Build an adjacency list from element connectivity.

    Two nodes are adjacent if they share at least one element.
    Node IDs here are 0-based array indices.
    """
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for elem in connectivity:
        nodes = [int(n) for n in elem]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                adj[nodes[i]].add(nodes[j])
                adj[nodes[j]].add(nodes[i])

    # Convert sets to sorted lists for deterministic ordering
    return [sorted(s) for s in adj]


def _pseudo_peripheral_node(adj: list[list[int]]) -> int:
    """
    Find a pseudo-peripheral node using the George-Liu algorithm.

    A peripheral node has high eccentricity (far from all other nodes).
    Starting RCM from such a node tends to give better bandwidth.

    Algorithm:
    1. Start from node 0 (or the lowest-degree node)
    2. BFS to find the farthest node -> candidate
    3. BFS again from candidate to find a farther one
    4. Repeat until eccentricity stops increasing
    """
    n = len(adj)
    if n == 0:
        return 0

    # Start from lowest-degree node
    start = min(range(n), key=lambda i: len(adj[i]))

    for _ in range(10):  # safety limit
        # BFS from start
        visited = [-1] * n
        visited[start] = 0
        queue = deque([start])
        last = start

        while queue:
            u = queue.popleft()
            last = u
            for v in adj[u]:
                if visited[v] < 0:
                    visited[v] = visited[u] + 1
                    queue.append(v)

        # last is the farthest node from start
        if last == start:
            break
        start = last

    return start


def _cm_from_start(
    start: int,
    adj: list[list[int]],
    visited: np.ndarray,
) -> list[int]:
    """
    Run one Cuthill-McKee BFS pass from *start*, respecting
    the *visited* array.  Returns the CM ordering (not reversed).
    """
    queue = deque([start])
    visited[start] = True
    component: list[int] = []

    while queue:
        u = queue.popleft()
        component.append(u)

        # Collect unvisited neighbours, sort by degree (ascending)
        neighbours = [v for v in adj[u] if not visited[v]]
        neighbours.sort(key=lambda v: len(adj[v]))

        for v in neighbours:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    return component


def _rcm_ordering(
    n_nodes: int,
    connectivity: ndarray,
) -> ndarray:
    """
    Compute the Reverse Cuthill-McKee ordering.

    Tries multiple starting nodes (all minimum-degree nodes) and
    picks the one producing the lowest bandwidth.  Falls back to
    identity if RCM doesn't improve over natural ordering.

    Returns
    -------
    ndarray of shape (n_nodes,)
        ``perm[new_index] = old_index``
    """
    if n_nodes == 0:
        return np.array([], dtype=int)

    adj = _build_adjacency(n_nodes, connectivity)

    # ── Find candidate starting nodes ─────────────────────────
    degrees = [len(a) for a in adj]
    min_deg = min(degrees)
    candidates = [i for i, d in enumerate(degrees) if d == min_deg]

    # Also try the pseudo-peripheral node
    pp = _pseudo_peripheral_node(adj)
    if pp not in candidates:
        candidates.append(pp)

    # ── Try each candidate, keep the best ─────────────────────
    best_bw = float('inf')
    best_perm = np.arange(n_nodes, dtype=int)

    for cand in candidates:
        visited = np.full(n_nodes, False)
        ordering: list[int] = []

        # Start from this candidate
        component = _cm_from_start(cand, adj, visited)
        ordering.extend(reversed(component))

        # Handle remaining disconnected components
        while not all(visited):
            unvisited = [i for i in range(n_nodes) if not visited[i]]
            if not unvisited:
                break
            comp = _cm_from_start(unvisited[0], adj, visited)
            ordering.extend(reversed(comp))

        perm = np.array(ordering, dtype=int)
        inv_perm = np.empty(n_nodes, dtype=int)
        inv_perm[perm] = np.arange(n_nodes)
        test_conn = inv_perm[connectivity]
        bw = _compute_bandwidth(test_conn)

        if bw < best_bw:
            best_bw = bw
            best_perm = perm

    # ── Fallback: don't make things worse ─────────────────────
    original_bw = _compute_bandwidth(connectivity)
    if best_bw >= original_bw:
        return np.arange(n_nodes, dtype=int)  # identity

    return best_perm


# =====================================================================
# Numberer class
# =====================================================================

class Numberer:
    """
    Renumbers a FEM mesh for solver consumption.

    Parameters
    ----------
    fem_data : dict
        Output of ``Mesh.get_fem_data()``.  Must contain:
        ``node_tags``, ``node_coords``, ``elem_tags``,
        ``connectivity``, ``used_tags``.
    """

    def __init__(self, fem_data: dict) -> None:
        self._raw = fem_data
        self._node_tags   = np.asarray(fem_data['node_tags'], dtype=int)
        self._node_coords = np.asarray(fem_data['node_coords'], dtype=float)
        self._elem_tags   = np.asarray(fem_data['elem_tags'], dtype=int)
        self._connectivity = np.asarray(fem_data['connectivity'], dtype=int)
        self._used_tags   = fem_data.get('used_tags', set(self._connectivity.flatten()))

        # Build Gmsh tag -> raw array index
        self._tag_to_raw_idx: dict[int, int] = {
            int(t): i for i, t in enumerate(self._node_tags)
        }

    def renumber(
        self,
        method: str = "simple",
        *,
        base: int = 1,
        used_only: bool = True,
    ) -> NumberedMesh:
        """
        Produce a solver-ready mesh with contiguous IDs.

        Parameters
        ----------
        method : ``"simple"`` or ``"rcm"``
            ``"simple"``  — preserves relative order, just makes IDs
            contiguous.  Fast, no optimisation.

            ``"rcm"``  — Reverse Cuthill-McKee bandwidth minimisation.
            Reorders nodes so that the assembled stiffness matrix has
            minimal bandwidth.  Recommended for direct solvers.

        base : int
            Starting ID (default 1 = Fortran/OpenSees convention;
            use 0 for C/Python convention).

        used_only : bool
            If True (default), only include nodes that appear in at
            least one element (skip orphan nodes).  Set False to
            include all nodes from the mesh.

        Returns
        -------
        NumberedMesh
        """
        # ── Filter nodes ──────────────────────────────────────────
        if used_only:
            mask = np.isin(self._node_tags, list(self._used_tags))
            n_total   = len(self._node_tags)
            gmsh_tags = self._node_tags[mask]
            coords    = self._node_coords[mask]
            n_orphans = n_total - len(gmsh_tags)
            if n_orphans > 0:
                orphan_tags = self._node_tags[~mask]
                print(
                    f"[Numberer] WARNING: {n_orphans} orphan node(s) "
                    f"skipped (not connected to any element). "
                    f"Tags: {orphan_tags.tolist()[:20]}"
                    + (f" ... (+{n_orphans - 20} more)"
                       if n_orphans > 20 else "")
                )
        else:
            gmsh_tags = self._node_tags.copy()
            coords    = self._node_coords.copy()

        n_nodes = len(gmsh_tags)

        # ── Temporary 0-based indexing ────────────────────────────
        # Map Gmsh tags -> 0-based indices for internal work
        gtag_to_tmp: dict[int, int] = {
            int(t): i for i, t in enumerate(gmsh_tags)
        }

        # Rewrite connectivity in 0-based tmp indices
        conn_tmp = np.zeros_like(self._connectivity)
        for i in range(self._connectivity.shape[0]):
            for j in range(self._connectivity.shape[1]):
                conn_tmp[i, j] = gtag_to_tmp[int(self._connectivity[i, j])]

        # ── Compute permutation ───────────────────────────────────
        if method == "rcm":
            perm = _rcm_ordering(n_nodes, conn_tmp)
        elif method == "simple":
            perm = np.arange(n_nodes, dtype=int)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'simple' or 'rcm'."
            )

        # perm[new_pos] = old_pos
        # inverse: inv_perm[old_pos] = new_pos
        inv_perm = np.empty(n_nodes, dtype=int)
        inv_perm[perm] = np.arange(n_nodes)

        # ── Apply permutation ─────────────────────────────────────
        inv_perm + base              # contiguous IDs
        new_coords   = coords[perm]                 # reordered coords
        gmsh_tags[perm]            # Gmsh tags in new order

        # Rewrite connectivity with new IDs
        new_conn = np.zeros_like(conn_tmp)
        for i in range(conn_tmp.shape[0]):
            for j in range(conn_tmp.shape[1]):
                new_conn[i, j] = inv_perm[conn_tmp[i, j]] + base

        # Element IDs: simple contiguous
        new_elem_ids = np.arange(base, base + len(self._elem_tags), dtype=int)

        # ── Bandwidth ─────────────────────────────────────────────
        bw = _compute_bandwidth(new_conn)

        # ── Build maps ────────────────────────────────────────────
        g2s_node: dict[int, int] = {}
        s2g_node: dict[int, int] = {}
        for new_pos in range(n_nodes):
            old_pos = perm[new_pos]
            gtag = int(gmsh_tags[old_pos])
            sid  = int(inv_perm[old_pos]) + base
            g2s_node[gtag] = sid
            s2g_node[sid]  = gtag

        g2s_elem: dict[int, int] = {}
        s2g_elem: dict[int, int] = {}
        for i, etag in enumerate(self._elem_tags):
            eid = int(new_elem_ids[i])
            g2s_elem[int(etag)] = eid
            s2g_elem[eid]       = int(etag)

        # ── Rewrite node_ids array in order ───────────────────────
        # node_ids[i] = solver ID of the i-th node (in new ordering)
        solver_node_ids = np.arange(base, base + n_nodes, dtype=int)

        result = NumberedMesh(
            node_ids=solver_node_ids,
            node_coords=new_coords,
            elem_ids=new_elem_ids,
            connectivity=new_conn,
            n_nodes=n_nodes,
            n_elems=len(self._elem_tags),
            bandwidth=bw,
            method=method,
            gmsh_to_solver_node=g2s_node,
            solver_to_gmsh_node=s2g_node,
            gmsh_to_solver_elem=g2s_elem,
            solver_to_gmsh_elem=s2g_elem,
        )

        return result

    def compare_methods(self) -> dict[str, int]:
        """
        Compare bandwidth for all available methods.

        Returns
        -------
        dict[str, int]
            ``{"simple": bw1, "rcm": bw2}``
        """
        results = {}
        for method in ("simple", "rcm"):
            data = self.renumber(method=method)
            results[method] = data.bandwidth
        return results
