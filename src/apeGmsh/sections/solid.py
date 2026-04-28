"""
Solid section factories — 3D volumes, hex-compatible via slicing.
================================================================

Each function returns a :class:`~apeGmsh.core.Part.Part` containing
axis-aligned rectangular volumes that are ready for transfinite
hex meshing via ``g.mesh.structured.set_transfinite_automatic()``.
"""
from __future__ import annotations

import gmsh

from apeGmsh.core.Part import Part
from apeGmsh.core._section_placement import apply_placement
from ._classify import (
    classify_angle_outer_faces,
    classify_end_faces,
    classify_tee_outer_faces,
    classify_w_outer_faces,
    classify_w_web_side_faces,
)


# =====================================================================
# W-shape (wide flange / I-beam)
# =====================================================================

def W_solid(
    bf: float,
    tf: float,
    h: float,
    tw: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "W_solid",
) -> Part:
    """Create a W-shape (wide flange) as a 3D solid Part.

    The section is built as an extruded I-profile, then sliced
    into **7 hex-compatible volumes** by 4 axis-aligned cuts.

    Parameters
    ----------
    bf : float
        Flange width.
    tf : float
        Flange thickness.
    h : float
        Clear web height (between flanges, not including flanges).
    tw : float
        Web thickness.
    length : float
        Extrusion length along Z.
    anchor : str or (x, y, z), default ``"start"``
        Re-origin the section in its local frame before optional align.
        See :func:`apeGmsh.core._section_placement.compute_anchor_offset`.
    align : str or (ax, ay, az), default ``"z"``
        Reorient the local +Z axis to a world direction.
        See :func:`apeGmsh.core._section_placement.compute_alignment_rotation`.
    name : str, default "W_solid"
        Part name.

    Labels created
    --------------
    ``top_flange``
        3 volumes forming the top flange (y > h/2).
    ``bottom_flange``
        3 volumes forming the bottom flange (y < −h/2).
    ``web``
        1 volume for the web (|y| ≤ h/2).
    ``top_flange_face``
        Outer +y skin surfaces of the top flange (face-to-face
        stacking target for ``align_to``).
    ``bottom_flange_face``
        Outer −y skin surfaces of the bottom flange.
    ``web_left_face``, ``web_right_face``
        Exposed −x / +x outer faces of the web (within |y| ≤ h/2).
    ``start_face``, ``end_face``
        Cross-section profile faces at z=0 and z=length.

    Returns
    -------
    Part

    Example
    -------
    ::

        col = W_solid(bf=150, tf=20, h=300, tw=10, length=3000)

        with apeGmsh("frame") as g:
            g.parts.add(col, label="col_A")
            g.mesh.structured.set_transfinite_automatic()
            g.mesh.sizing.set_global_size(50)
            g.mesh.generation.generate(3)
    """
    from ._classify import classify_w_volumes

    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean
        tr  = part.model.transforms

        # Profile: outer rectangle minus two voids
        total_h = 2 * tf + h
        outer  = geo.add_rectangle(x=-bf / 2, y=-total_h / 2, z=0, dx=bf, dy=total_h)
        void_l = geo.add_rectangle(x=-bf / 2, y=-h / 2, z=0, dx=bf / 2 - tw / 2, dy=h)
        void_r = geo.add_rectangle(x=tw / 2,  y=-h / 2, z=0, dx=bf / 2 - tw / 2, dy=h)
        profile = boo.cut(outer, [void_l, void_r], dim=2)

        # Extrude along Z
        tr.extrude(profile, 0, 0, length)

        # Slice into 7 hex-compatible regions
        geo.slice(axis='x', offset=-tw / 2)
        geo.slice(axis='x', offset=tw / 2)
        geo.slice(axis='y', offset=h / 2)
        geo.slice(axis='y', offset=-h / 2)

        # Label by structural role
        classify_w_volumes(h, tw, tf, bf, part.labels)

        # Label end-cap surfaces for BC / load application
        classify_end_faces(length, part.labels)

        # Label outer +y / -y skin surfaces for face-to-face stacking
        classify_w_outer_faces(h, tf, part.labels)
        classify_w_web_side_faces(h, tw, part.labels)

        # Apply anchor + align AFTER classification so the z=0/length
        # end-face heuristic still matches the as-extruded geometry.
        apply_placement(anchor, align, length=length)

    return part


# =====================================================================
# Rectangular solid
# =====================================================================

def rect_solid(
    b: float,
    h: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "rect_solid",
) -> Part:
    """Create a solid rectangular bar as a 3D Part.

    Parameters
    ----------
    b : float
        Width (X-direction).
    h : float
        Height (Y-direction).
    length : float
        Length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``body`` — the single volume.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        part.model.geometry.add_box(-b / 2, -h / 2, 0, b, h, length, label="body")
        classify_end_faces(length, part.labels)
        apply_placement(anchor, align, length=length)
    return part


# =====================================================================
# Hollow rectangular (HSS rect)
# =====================================================================

def rect_hollow(
    b: float,
    h: float,
    t: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "rect_hollow",
) -> Part:
    """Create a hollow rectangular tube (HSS) as a 3D solid Part.

    Parameters
    ----------
    b : float
        Outer width (X-direction).
    h : float
        Outer height (Y-direction).
    t : float
        Wall thickness.
    length : float
        Length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``body`` — the hollow tube volume.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean

        outer = geo.add_box(-b / 2, -h / 2, 0, b, h, length)
        inner = geo.add_box(-b / 2 + t, -h / 2 + t, 0,
                            b - 2 * t, h - 2 * t, length)
        boo.cut(outer, [inner])

        # Label the surviving volume
        for _, tag in gmsh.model.getEntities(3):
            part.labels.add(3, [tag], name="body")
            break
        classify_end_faces(length, part.labels)
        apply_placement(anchor, align, length=length)

    return part


# =====================================================================
# Circular pipe (solid)
# =====================================================================

def pipe_solid(
    r: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "pipe_solid",
) -> Part:
    """Create a solid circular bar as a 3D Part.

    Parameters
    ----------
    r : float
        Radius.
    length : float
        Length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``body`` — the single cylinder volume.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        part.model.geometry.add_cylinder(0, 0, 0, 0, 0, length, r, label="body")
        classify_end_faces(length, part.labels)
        apply_placement(anchor, align, length=length)
    return part


# =====================================================================
# Hollow circular pipe
# =====================================================================

def pipe_hollow(
    r_outer: float,
    t: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "pipe_hollow",
) -> Part:
    """Create a hollow circular pipe as a 3D solid Part.

    Parameters
    ----------
    r_outer : float
        Outer radius.
    t : float
        Wall thickness.
    length : float
        Length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``body`` — the hollow pipe volume.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean

        outer = geo.add_cylinder(0, 0, 0, 0, 0, length, r_outer)
        inner = geo.add_cylinder(0, 0, 0, 0, 0, length, r_outer - t)
        boo.cut(outer, [inner])

        for _, tag in gmsh.model.getEntities(3):
            part.labels.add(3, [tag], name="body")
            break
        classify_end_faces(length, part.labels)
        apply_placement(anchor, align, length=length)

    return part


# =====================================================================
# L-shape (angle)
# =====================================================================

def angle_solid(
    b: float,
    h: float,
    t: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "angle_solid",
) -> Part:
    """Create an L-shape (angle) as a 3D solid Part.

    The angle is placed with its corner at the origin, legs
    extending in +X and +Y.  Sliced at the corner junction
    for hex-compatible meshing.

    Parameters
    ----------
    b : float
        Horizontal leg width (X-direction).
    h : float
        Vertical leg height (Y-direction).
    t : float
        Thickness of both legs.
    length : float
        Extrusion length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``horizontal_leg`` — volumes in the horizontal leg (y < t).
    ``vertical_leg`` — volumes in the vertical leg (x < t).
    ``horizontal_leg_face`` — underside of h-leg at y=0.
    ``vertical_leg_face`` — back of v-leg at x=0.
    ``start_face``, ``end_face`` — profile faces at z=0 and z=length.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean
        tr  = part.model.transforms

        # Build L-profile as two rectangles fused
        h_leg = geo.add_rectangle(x=0, y=0, z=0, dx=b, dy=t)
        v_leg = geo.add_rectangle(x=0, y=0, z=0, dx=t, dy=h)
        boo.fuse([h_leg], [v_leg], dim=2)

        # Extrude
        surfs = gmsh.model.getEntities(2)
        if surfs:
            tr.extrude(surfs[0], 0, 0, length)

        # Slice at the corner junction
        geo.slice(axis='x', offset=t)
        geo.slice(axis='y', offset=t)

        # Label by structural role
        h_tags = []
        v_tags = []
        for _, tag in gmsh.model.getEntities(3):
            com = gmsh.model.occ.getCenterOfMass(3, tag)
            if com[1] < t:
                h_tags.append(tag)
            else:
                v_tags.append(tag)
        if h_tags:
            part.labels.add(3, h_tags, name="horizontal_leg")
        if v_tags:
            part.labels.add(3, v_tags, name="vertical_leg")
        classify_end_faces(length, part.labels)
        classify_angle_outer_faces(part.labels)
        apply_placement(anchor, align, length=length)

    return part


# =====================================================================
# C-shape (channel)
# =====================================================================

def channel_solid(
    bf: float,
    tf: float,
    h: float,
    tw: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "channel_solid",
) -> Part:
    """Create a C-shape (channel) as a 3D solid Part.

    The channel opens in the +X direction.  The web is at x=0,
    flanges extend in the +X direction from the web.

    Parameters
    ----------
    bf : float
        Flange width (depth of flanges in X).
    tf : float
        Flange thickness.
    h : float
        Clear web height (between flanges).
    tw : float
        Web thickness.
    length : float
        Extrusion length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``top_flange`` — volumes in the top flange (y > h/2).
    ``bottom_flange`` — volumes in the bottom flange (y < −h/2).
    ``web`` — volumes in the web.
    ``top_flange_face`` — outer +y skin of top flange.
    ``bottom_flange_face`` — outer −y skin of bottom flange.
    ``start_face``, ``end_face`` — profile faces at z=0 and z=length.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean
        tr  = part.model.transforms

        total_h = h + 2 * tf

        # C-profile: outer rectangle minus one void (the open side)
        outer = geo.add_rectangle(x=0, y=-total_h / 2, z=0, dx=bf, dy=total_h)
        void  = geo.add_rectangle(x=tw, y=-h / 2, z=0, dx=bf - tw, dy=h)
        boo.cut(outer, [void], dim=2)

        # Extrude
        surfs = gmsh.model.getEntities(2)
        if surfs:
            tr.extrude(surfs[0], 0, 0, length)

        # Slice for hex readiness
        geo.slice(axis='x', offset=tw)
        geo.slice(axis='y', offset=h / 2)
        geo.slice(axis='y', offset=-h / 2)

        # Label by role
        from ._classify import classify_w_volumes
        classify_w_volumes(h, tw, tf, bf, part.labels)
        classify_end_faces(length, part.labels)
        classify_w_outer_faces(h, tf, part.labels)
        apply_placement(anchor, align, length=length)

    return part


# =====================================================================
# T-shape (tee / WT)
# =====================================================================

def tee_solid(
    bf: float,
    tf: float,
    h: float,
    tw: float,
    length: float,
    *,
    anchor="start",
    align="z",
    name: str = "tee_solid",
) -> Part:
    """Create a T-shape (tee / WT) as a 3D solid Part.

    The flange is at the top (+Y), the stem hangs down.  Centered
    on the web at x=0.

    Parameters
    ----------
    bf : float
        Flange width.
    tf : float
        Flange thickness.
    h : float
        Stem height (from bottom of flange to bottom of stem).
    tw : float
        Stem (web) thickness.
    length : float
        Extrusion length (Z-direction).
    name : str
        Part name.

    Labels created
    --------------
    ``flange`` — volumes in the flange.
    ``stem`` — volumes in the stem.
    ``flange_face`` — outer +y skin of the flange (top).
    ``stem_face`` — outer −y skin of the stem (bottom).
    ``start_face``, ``end_face`` — profile faces at z=0 and z=length.

    Returns
    -------
    Part
    """
    part = Part(name)
    with part:
        geo = part.model.geometry
        boo = part.model.boolean
        tr  = part.model.transforms

        # T-profile: flange rectangle + stem rectangle, fused
        flange = geo.add_rectangle(x=-bf / 2, y=0, z=0, dx=bf, dy=tf)
        stem   = geo.add_rectangle(x=-tw / 2, y=-h, z=0, dx=tw, dy=h)
        boo.fuse([flange], [stem], dim=2)

        surfs = gmsh.model.getEntities(2)
        if surfs:
            tr.extrude(surfs[0], 0, 0, length)

        # Slice at the flange-stem junction
        geo.slice(axis='x', offset=-tw / 2)
        geo.slice(axis='x', offset=tw / 2)
        geo.slice(axis='y', offset=0)

        # Label by role
        flange_tags = []
        stem_tags = []
        for _, tag in gmsh.model.getEntities(3):
            com = gmsh.model.occ.getCenterOfMass(3, tag)
            if com[1] >= 0:
                flange_tags.append(tag)
            else:
                stem_tags.append(tag)
        if flange_tags:
            part.labels.add(3, flange_tags, name="flange")
        if stem_tags:
            part.labels.add(3, stem_tags, name="stem")
        classify_end_faces(length, part.labels)
        classify_tee_outer_faces(h, tf, part.labels)
        apply_placement(anchor, align, length=length)

    return part
