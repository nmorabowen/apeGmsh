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

from apeGmsh._types import DimTag


def _auto_glyph_radius(
    centers: np.ndarray,
    point_size: float,
    fallback_diag: float,
) -> float:
    """Auto-scale a sphere glyph radius from the centers' own extents.

    The legacy ``0.003 * model_diagonal`` formula uses ``√(dx²+dy²+dz²)``
    which is dominated by the longest axis. On elongated geometry
    (long thin beam, large flat plate) the resulting glyphs are
    bigger than the cross-section. Using the geometric mean of the
    extents pulls the size toward the smaller dimensions while
    matching the legacy size on roughly cubic models — the constant
    ``0.0052`` is calibrated so ``0.0052 × side ≈ 0.003 × (√3 × side)``
    when ``dx = dy = dz = side``.

    Zero-extent dimensions (planar / line-shaped centers) are
    substituted with the smallest non-zero extent so the geometric
    mean doesn't collapse to 0.
    """
    scale = max(0.1, point_size / 10.0)
    if len(centers) < 2:
        return fallback_diag * 0.003 * scale
    extents = centers.max(axis=0) - centers.min(axis=0)
    nonzero = extents[extents > 0]
    if nonzero.size == 0:
        return fallback_diag * 0.003 * scale
    safe = np.where(extents > 0, extents, nonzero.min())
    gmean = float(np.prod(safe) ** (1.0 / safe.size))
    return 0.0052 * gmean * scale


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
    base_r = _auto_glyph_radius(centers, point_size, model_diagonal)
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
    color: str | None = None,
) -> tuple[pv.PolyData, Any]:
    """Build a node-cloud overlay (not pickable).

    Renders each node as a flat GL point (``style="points"``) — no
    glyph geometry is generated, so cold start / per-frame cost stays
    O(0) in triangles even at 600k+ nodes.

    ``render_points_as_spheres`` is deliberately **off**: the shaded
    point-sprite path needs a GL sphere-impostor shader that is absent
    on software / remote / older-driver GL stacks, where it silently
    rasterises nothing (nodes vanish while line/surface actors still
    draw). Plain ``GL_POINTS`` is the most universally supported
    primitive and always renders. See the same rationale in
    ``diagrams/_gauss_marker.py``.

    ``marker_size`` is screen pixels (constant under zoom). The
    legacy sphere-mesh path scaled with ``model_diagonal``; that arg
    stays in the signature for caller compat but is no longer read.
    ``color`` defaults to the active palette's ``node_accent``.
    Returns (cloud_polydata, actor).
    """
    if color is None:
        from ..ui.theme import THEME
        color = THEME.current.node_accent
    cloud = pv.PolyData(node_coords)
    actor = plotter.add_mesh(
        cloud,
        color=color,
        style="points",
        point_size=float(marker_size),
        render_points_as_spheres=False,
        pickable=False,
        opacity=1.0,
        reset_camera=False,
    )
    return cloud, actor
