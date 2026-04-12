"""
Spatial filter engine — pure numpy functions for mesh node/element queries.

No Gmsh dependency.  Every function takes raw arrays and returns a boolean
mask or index array.  Used by :class:`MeshSelectionSet` to resolve
``add_nodes(on_plane=...)`` and ``add_elements(in_box=...)`` queries.

Usage::

    from apeGmsh.mesh._mesh_filters import nodes_on_plane, nodes_in_box
    mask = nodes_on_plane(coords, axis="z", value=0.0, atol=1e-3)
    selected_ids = node_ids[mask]
"""
from __future__ import annotations

import numpy as np


# ======================================================================
# Node filters
# ======================================================================

def nodes_on_plane(
    coords: np.ndarray,
    axis: str | int,
    value: float,
    atol: float = 1e-6,
) -> np.ndarray:
    """Return boolean mask for nodes within *atol* of a coordinate plane.

    Parameters
    ----------
    coords : ndarray (N, 3)
    axis : "x" | "y" | "z" or 0 | 1 | 2
    value : coordinate value of the plane
    atol : absolute tolerance

    Returns
    -------
    ndarray (N,) bool
    """
    col = _axis_index(axis)
    return np.abs(coords[:, col] - value) <= atol


def nodes_in_box(
    coords: np.ndarray,
    bbox: tuple | list,
) -> np.ndarray:
    """Return boolean mask for nodes inside an axis-aligned bounding box.

    Parameters
    ----------
    coords : ndarray (N, 3)
    bbox : (xmin, ymin, zmin, xmax, ymax, zmax)

    Returns
    -------
    ndarray (N,) bool
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return (
        (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax)
        & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        & (coords[:, 2] >= zmin) & (coords[:, 2] <= zmax)
    )


def nodes_in_sphere(
    coords: np.ndarray,
    center: tuple | list | np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return boolean mask for nodes within *radius* of *center*.

    Parameters
    ----------
    coords : ndarray (N, 3)
    center : (x, y, z)
    radius : float

    Returns
    -------
    ndarray (N,) bool
    """
    c = np.asarray(center, dtype=np.float64)
    dist2 = np.sum((coords - c) ** 2, axis=1)
    return dist2 <= radius * radius


def nodes_nearest(
    coords: np.ndarray,
    point: tuple | list | np.ndarray,
    count: int = 1,
) -> np.ndarray:
    """Return boolean mask for the *count* nearest nodes to *point*.

    Parameters
    ----------
    coords : ndarray (N, 3)
    point : (x, y, z)
    count : number of nearest nodes to select

    Returns
    -------
    ndarray (N,) bool
    """
    p = np.asarray(point, dtype=np.float64)
    dist2 = np.sum((coords - p) ** 2, axis=1)
    idx = np.argpartition(dist2, min(count, len(dist2)) - 1)[:count]
    mask = np.zeros(len(coords), dtype=bool)
    mask[idx] = True
    return mask


# ======================================================================
# Element filters
# ======================================================================

def element_centroids(
    connectivity: np.ndarray,
    node_id_to_idx: dict,
    node_coords: np.ndarray,
) -> np.ndarray:
    """Compute centroid of each element.

    Parameters
    ----------
    connectivity : ndarray (E, npe) — node IDs per element
    node_id_to_idx : dict — maps node ID -> row index in node_coords
    node_coords : ndarray (N, 3)

    Returns
    -------
    ndarray (E, 3) — centroid coordinates
    """
    E, npe = connectivity.shape
    centroids = np.zeros((E, 3), dtype=np.float64)
    for j in range(npe):
        col_ids = connectivity[:, j]
        # Vectorised lookup: map node IDs -> coord indices
        idx = np.array([node_id_to_idx.get(int(nid), 0) for nid in col_ids])
        centroids += node_coords[idx]
    centroids /= npe
    return centroids


def elements_in_box(
    centroids: np.ndarray,
    bbox: tuple | list,
) -> np.ndarray:
    """Return boolean mask for elements whose centroid is inside a box.

    Parameters
    ----------
    centroids : ndarray (E, 3)
    bbox : (xmin, ymin, zmin, xmax, ymax, zmax)

    Returns
    -------
    ndarray (E,) bool
    """
    return nodes_in_box(centroids, bbox)


def elements_on_plane(
    connectivity: np.ndarray,
    node_id_to_idx: dict,
    node_coords: np.ndarray,
    axis: str | int,
    value: float,
    atol: float = 1e-6,
) -> np.ndarray:
    """Return boolean mask for elements with ALL nodes on a plane.

    Parameters
    ----------
    connectivity : ndarray (E, npe)
    node_id_to_idx : dict
    node_coords : ndarray (N, 3)
    axis, value, atol : plane specification

    Returns
    -------
    ndarray (E,) bool
    """
    col = _axis_index(axis)
    E, npe = connectivity.shape
    all_on = np.ones(E, dtype=bool)
    for j in range(npe):
        col_ids = connectivity[:, j]
        idx = np.array([node_id_to_idx.get(int(nid), 0) for nid in col_ids])
        on_plane = np.abs(node_coords[idx, col] - value) <= atol
        all_on &= on_plane
    return all_on


def boundary_nodes_of(
    elem_ids: np.ndarray,
    connectivity: np.ndarray,
    all_connectivity: np.ndarray,
) -> np.ndarray:
    """Return node IDs that lie on the boundary of a set of elements.

    Boundary nodes are those connected to at least one element outside
    the set.  Useful for applying boundary conditions to a region.

    Parameters
    ----------
    elem_ids : ndarray (E_sel,) — selected element IDs
    connectivity : ndarray (E_sel, npe) — connectivity of selected elements
    all_connectivity : ndarray (E_all, npe) — full mesh connectivity

    Returns
    -------
    ndarray of unique boundary node IDs
    """
    sel_nodes = set(int(n) for n in connectivity.ravel())
    sel_elem_set = set(int(e) for e in elem_ids)

    # Find all elements that share any node with the selection
    # but are NOT in the selection
    boundary = set()
    for i, row in enumerate(all_connectivity):
        row_nodes = set(int(n) for n in row)
        if row_nodes & sel_nodes and i not in sel_elem_set:
            boundary |= row_nodes & sel_nodes
    return np.array(sorted(boundary), dtype=np.int64)


# ======================================================================
# Helpers
# ======================================================================

def _axis_index(axis: str | int) -> int:
    """Convert axis name or index to column index."""
    if isinstance(axis, int):
        return axis
    return {"x": 0, "X": 0, "y": 1, "Y": 1, "z": 2, "Z": 2}[axis]
