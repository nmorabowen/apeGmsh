"""
Profile-only section factories — 2D cross-sections.
====================================================

Returns a :class:`Part` containing only the 2D cross-section
surface (no extrusion).  Useful for:

* Fiber section analysis in OpenSees
* ``transforms.sweep(profile, curved_path)`` for curved members
* Visualization / inspection of the section shape
"""
from __future__ import annotations

from apeGmsh.core.Part import Part


def W_profile(
    bf: float,
    tf: float,
    h: float,
    tw: float,
    *,
    name: str = "W_profile",
) -> Part:
    """Create a W-shape 2D cross-section (no extrusion).

    The I-shaped surface sits in the XY plane at z=0, centered
    on the origin.

    Parameters
    ----------
    bf : float
        Flange width.
    tf : float
        Flange thickness.
    h : float
        Clear web height.
    tw : float
        Web thickness.
    name : str
        Part name.

    Labels created
    --------------
    ``profile`` — the I-shaped surface.

    Returns
    -------
    Part

    Example
    -------
    ::

        section = W_profile(bf=150, tf=20, h=300, tw=10)
        # section.has_file → True (auto-persisted)
        # Use for fiber analysis or sweep along a path
    """
    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean

        total_h = 2 * tf + h
        outer  = geo.add_rectangle(x=-bf / 2, y=-total_h / 2, z=0, dx=bf, dy=total_h)
        void_l = geo.add_rectangle(x=-bf / 2, y=-h / 2, z=0, dx=bf / 2 - tw / 2, dy=h)
        void_r = geo.add_rectangle(x=tw / 2,  y=-h / 2, z=0, dx=bf / 2 - tw / 2, dy=h)
        boo.cut(outer, [void_l, void_r], dim=2)

        # Label the surviving surface
        import gmsh
        for _, tag in gmsh.model.getEntities(2):
            part.labels.add(2, [tag], name="profile")
            break

    return part
