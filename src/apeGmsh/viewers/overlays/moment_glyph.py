"""
moment_glyph — Curved arrow geometry for moment visualization.
===============================================================

Creates a 270° arc tube with a cone arrowhead, oriented with its
rotation axis along X.  When used with ``pv.PolyData.glyph(orient=...)``
the arc aligns perpendicular to the moment vector direction.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


def make_moment_glyph(
    radius: float = 1.0,
    tube_radius: float = 0.08,
    arc_degrees: float = 270.0,
    resolution: int = 24,
) -> pv.PolyData:
    """Build a curved arrow glyph for moment/rotation visualization.

    The arc lies in the YZ plane with the rotation axis along +X.
    PyVista's ``glyph(orient='vectors')`` will rotate +X to align
    with the moment direction vector.

    Parameters
    ----------
    radius : float
        Arc radius (1.0 = unit; will be scaled by glyph factor).
    tube_radius : float
        Tube cross-section radius relative to arc radius.
    arc_degrees : float
        Arc sweep in degrees (270 = three-quarter turn).
    resolution : int
        Number of points along the arc.

    Returns
    -------
    pv.PolyData
        Combined tube + cone mesh.
    """
    # Arc points in YZ plane, axis along X
    angles = np.linspace(0, np.radians(arc_degrees), resolution)
    pts = np.column_stack([
        np.zeros(resolution),
        radius * np.cos(angles),
        radius * np.sin(angles),
    ])

    # Build tube along the arc
    spline = pv.Spline(pts, resolution)
    tube = spline.tube(radius=tube_radius, n_sides=8)

    # Arrowhead cone at the arc tip
    end = pts[-1]
    tangent = pts[-1] - pts[-2]
    tangent = tangent / np.linalg.norm(tangent)

    cone_height = tube_radius * 5
    cone_radius = tube_radius * 2.5
    cone = pv.Cone(
        center=end + tangent * cone_height * 0.5,
        direction=tangent,
        height=cone_height,
        radius=cone_radius,
        resolution=10,
    )

    return tube + cone
