"""
Glyph-based point rendering — single actor for all point entities.

Builds one ``pv.PolyData`` glyph actor from an array of point centres.
Each point gets ``n_cells_per_glyph`` cells in the merged mesh, with
``cell_data["entity_tag"]`` and ``cell_data["colors"]`` for picking
and per-entity recoloring.

Usage::

    from apeGmsh.viewers.scene.glyph_points import build_point_glyphs
    mesh, actor, cell_to_dt, centroids = build_point_glyphs(
        plotter, centers, tags, model_diagonal=diag,
        point_size=10.0, idle_color=idle_rgb,
    )
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

DimTag = tuple[int, int]


def build_point_glyphs(
    plotter: pv.Plotter,
    centers: np.ndarray,
    tags: list[int],
    *,
    model_diagonal: float,
    point_size: float = 10.0,
    idle_color: np.ndarray,
) -> tuple[pv.PolyData, Any, dict[int, DimTag], dict[DimTag, np.ndarray]]:
    """Build a single glyph actor for all point entities.

    Parameters
    ----------
    plotter : pv.Plotter
        Target plotter.
    centers : ndarray (N, 3)
        Point coordinates.
    tags : list[int]
        Gmsh entity tags, one per centre.
    model_diagonal : float
        Bounding-box diagonal for auto-sizing.
    point_size : float
        Relative size (default 10 = 0.003 * diag).
    idle_color : ndarray (3,) uint8
        Default RGB for unpicked points.

    Returns
    -------
    mesh : pv.PolyData
        The glyph PolyData (with cell_data).
    actor : vtkActor
        The rendered actor.
    cell_to_dt : dict
        Cell index -> DimTag mapping.
    centroids : dict
        DimTag -> (3,) centroid coordinates.
    """
    cloud = pv.PolyData(centers)
    base_r = model_diagonal * 0.003 * max(0.1, point_size / 10.0)
    sphere_src = pv.Sphere(
        radius=base_r,
        theta_resolution=8,
        phi_resolution=8,
    )
    glyphs = cloud.glyph(geom=sphere_src, orient=False, scale=False)

    n_pts = len(tags)
    n_cells_per_pt = glyphs.n_cells // n_pts if n_pts else 1

    entity_tags = np.empty(glyphs.n_cells, dtype=np.int64)
    colors = np.tile(idle_color, (glyphs.n_cells, 1))
    cell_to_dt: dict[int, DimTag] = {}
    centroids: dict[DimTag, np.ndarray] = {}

    for i, tag in enumerate(tags):
        start = i * n_cells_per_pt
        end = start + n_cells_per_pt
        entity_tags[start:end] = tag
        dt: DimTag = (0, tag)
        for ci in range(start, end):
            cell_to_dt[ci] = dt
        centroids[dt] = centers[i]

    glyphs.cell_data["entity_tag"] = entity_tags
    glyphs.cell_data["colors"] = colors

    actor = plotter.add_mesh(
        glyphs,
        scalars="colors",
        rgb=True,
        smooth_shading=True,
        pickable=True,
        reset_camera=False,
    )

    return glyphs, actor, cell_to_dt, centroids


def build_node_cloud(
    plotter: pv.Plotter,
    node_coords: np.ndarray,
    *,
    model_diagonal: float,
    marker_size: float = 6.0,
    color: str = "#FF6600",
) -> tuple[pv.PolyData, Any]:
    """Build a node-cloud glyph overlay (not pickable).

    Returns (cloud_polydata, actor).
    """
    cloud = pv.PolyData(node_coords)
    glyph_r = 0.003 * model_diagonal * max(0.1, marker_size / 10.0)
    sphere_src = pv.Sphere(
        radius=glyph_r,
        theta_resolution=8,
        phi_resolution=8,
    )
    glyphs = cloud.glyph(geom=sphere_src, orient=False, scale=False)
    actor = plotter.add_mesh(
        glyphs,
        color=color,
        smooth_shading=True,
        pickable=False,
        opacity=1.0,
    )
    return cloud, actor
