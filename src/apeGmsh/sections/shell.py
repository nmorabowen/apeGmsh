"""
Shell section factories — 3D mid-surface rectangles for shell elements.
======================================================================

Shell sections model the structural member by its mid-surfaces
rather than its solid material boundary.  Each flange and web is
a single rectangular surface at the theoretical centroidal position.

These produce surfaces (dim=2) suitable for meshing with
``ShellMITC4``, ``ASDShellQ4``, or any quad/tri shell element.
"""
from __future__ import annotations

from apeGmsh.core.Part import Part


def _build_rect_surface(geo, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, *, label=None):
    """Build a planar quad surface from 4 corner points.

    Syncs before assigning the label so the entity exists in the
    Gmsh model topology when the label PG references it.
    """
    p1 = geo.add_point(x0, y0, z0, sync=False)
    p2 = geo.add_point(x1, y1, z1, sync=False)
    p3 = geo.add_point(x2, y2, z2, sync=False)
    p4 = geo.add_point(x3, y3, z3, sync=False)
    l1 = geo.add_line(p1, p2, sync=False)
    l2 = geo.add_line(p2, p3, sync=False)
    l3 = geo.add_line(p3, p4, sync=False)
    l4 = geo.add_line(p4, p1, sync=False)
    loop = geo.add_curve_loop([l1, l2, l3, l4], sync=False)
    # Sync so the surface exists in the model before the label
    # PG references it (label= triggers addPhysicalGroup).
    return geo.add_plane_surface(loop, sync=True, label=label)


def W_shell(
    bf: float,
    tf: float,
    h: float,
    tw: float,
    length: float,
    *,
    name: str = "W_shell",
) -> Part:
    """Create a W-shape as 3 mid-surface shell rectangles.

    The I-section is represented by:

    * **top flange** — horizontal rectangle at ``y = h/2 + tf/2``
      (flange mid-plane), width ``bf``, length ``length``.
    * **bottom flange** — horizontal rectangle at ``y = -(h/2 + tf/2)``.
    * **web** — vertical rectangle at ``x = 0``, height ``h``,
      length ``length``.

    Parameters
    ----------
    bf : float
        Flange width.
    tf : float
        Flange thickness (positions the mid-surface; the shell
        element's section definition carries the actual thickness).
    h : float
        Clear web height (between flange mid-surfaces).
    tw : float
        Web thickness (informational — the mid-surface is at x=0).
    length : float
        Extrusion length along Z.
    name : str
        Part name.

    Labels created
    --------------
    ``top_flange``
        The top flange mid-surface.
    ``bottom_flange``
        The bottom flange mid-surface.
    ``web``
        The web mid-surface.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        geo = part.model.geometry

        y_top = h / 2 + tf / 2
        y_bot = -(h / 2 + tf / 2)

        # Top flange: rectangle in XZ plane at y = y_top
        _build_rect_surface(
            geo,
            -bf / 2, y_top, 0,
            bf / 2,  y_top, 0,
            bf / 2,  y_top, length,
            -bf / 2, y_top, length,
            label="top_flange",
        )

        # Bottom flange: rectangle in XZ plane at y = y_bot
        _build_rect_surface(
            geo,
            -bf / 2, y_bot, 0,
            bf / 2,  y_bot, 0,
            bf / 2,  y_bot, length,
            -bf / 2, y_bot, length,
            label="bottom_flange",
        )

        # Web: rectangle in YZ plane at x = 0
        _build_rect_surface(
            geo,
            0, -h / 2, 0,
            0,  h / 2, 0,
            0,  h / 2, length,
            0, -h / 2, length,
            label="web",
        )

        part.model.sync()

    return part
