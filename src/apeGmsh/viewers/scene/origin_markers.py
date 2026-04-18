"""
Origin markers — optional reference-point overlay.

Draws one sphere glyph per user-provided world coordinate as a purely
visual artifact. Does not mutate geometry, mesh, or ``EntityRegistry``.
Optionally renders a ``(x, y, z)`` text label next to each marker.

Usage::

    from apeGmsh.viewers.scene.origin_markers import build_origin_markers
    glyph_actor, label_actor = build_origin_markers(
        plotter,
        points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        origin_shift=registry.origin_shift,
        model_diagonal=registry_or_scene.model_diagonal,
    )
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pyvista as pv


def build_origin_markers(
    plotter: pv.Plotter,
    points: Sequence[tuple[float, float, float]],
    *,
    origin_shift: np.ndarray,
    model_diagonal: float,
    show_coords: bool = True,
    color: str | None = None,
    size: float = 10.0,
    coord_precision: int = 2,
) -> tuple[Any, Any]:
    """Render sphere-glyph markers at the given world coordinates.

    Markers are drawn at ``point - origin_shift`` in render space so
    they land at the true world coordinate after the scene's numerical
    stability shift.

    Parameters
    ----------
    plotter : pv.Plotter
    points : sequence of (x, y, z)
        World coordinates. Empty sequence returns ``(None, None)``.
    origin_shift : ndarray (3,)
        Subtracted from each world coordinate before rendering (the same
        shift the scene builder applies to geometry).
    model_diagonal : float
        Bounding-box diagonal — drives glyph auto-size.
    show_coords : bool
        If ``True``, adds a ``(x, y, z)`` label next to each marker.
    color : str, optional
        Hex color for the glyphs. Defaults to the active palette's
        ``origin_marker_color`` when ``None``.
    size : float
        Relative marker size (default 10 = ~0.005 × diagonal).
    coord_precision : int
        Decimal places in the coord label.

    Returns
    -------
    glyph_actor : vtkActor | None
        The sphere-glyph actor, or ``None`` if ``points`` was empty.
    label_actor : vtkActor | None
        The point-labels actor, or ``None`` if labels are off / empty.
    """
    if len(points) == 0:
        return None, None

    from ..ui.preferences_manager import PREFERENCES
    _pref = PREFERENCES.current
    if color is None:
        from ..ui.theme import THEME
        color = THEME.current.origin_marker_color
    # Preference wins when caller used the default of 2
    if coord_precision == 2:
        coord_precision = _pref.coord_precision

    world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rendered = world - np.asarray(origin_shift, dtype=np.float64)

    radius = 0.005 * model_diagonal * max(0.1, size / 10.0)
    cloud = pv.PolyData(rendered)
    sphere = pv.Sphere(radius=radius, theta_resolution=12, phi_resolution=12)
    glyphs = cloud.glyph(geom=sphere, orient=False, scale=False)
    glyph_actor = plotter.add_mesh(
        glyphs,
        color=color,
        smooth_shading=True,
        pickable=False,
        reset_camera=False,
    )

    label_actor = None
    if show_coords:
        fmt = f"({{:.{coord_precision}f}}, {{:.{coord_precision}f}}, {{:.{coord_precision}f}})"
        labels = [fmt.format(*p) for p in world]
        from ..ui.theme import THEME
        pal = THEME.current
        label_actor = plotter.add_point_labels(
            rendered,
            labels,
            font_size=_pref.origin_marker_font_size,
            text_color=pal.text,
            shape_color=pal.mantle,
            point_color=color,
            point_size=1,
            always_visible=True,
            reset_camera=False,
        )

    return glyph_actor, label_actor
