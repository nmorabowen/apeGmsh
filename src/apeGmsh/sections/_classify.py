"""
Shared helpers for classifying volumes/surfaces after slicing.

Used by the solid and shell factories to label entities by their
structural role (top flange, web, bottom flange, etc.) based on
centroid position relative to the section geometry.
"""
from __future__ import annotations

import gmsh


def classify_w_volumes(
    h: float,
    tw: float,
    tf: float,
    bf: float,
    labels_comp,
) -> None:
    """Label the 7 hex-ready volumes of a sliced W-section.

    Groups into three structural regions:

    * ``top_flange``    — volumes whose centroid y > h/2
    * ``bottom_flange`` — volumes whose centroid y < −h/2
    * ``web``           — volumes whose centroid |y| ≤ h/2
    """
    top_tags: list[int] = []
    bot_tags: list[int] = []
    web_tags: list[int] = []

    for _, tag in gmsh.model.getEntities(3):
        com = gmsh.model.occ.getCenterOfMass(3, tag)
        y = com[1]
        if y > h / 2:
            top_tags.append(tag)
        elif y < -h / 2:
            bot_tags.append(tag)
        else:
            web_tags.append(tag)

    if top_tags:
        labels_comp.add(3, top_tags, name="top_flange")
    if bot_tags:
        labels_comp.add(3, bot_tags, name="bottom_flange")
    if web_tags:
        labels_comp.add(3, web_tags, name="web")


def _surface_exposed(tag: int) -> bool:
    """True if a 2D entity bounds exactly one volume (open to air).

    Used to skip internal cut surfaces produced by slicing: those
    bound two adjacent sub-volumes, exposed faces bound only one.
    """
    upward, _ = gmsh.model.getAdjacencies(2, tag)
    return len(upward) == 1


def classify_w_outer_faces(
    h: float,
    tf: float,
    labels_comp,
    *,
    tol: float = 1e-3,
) -> None:
    """Label the outer +y / -y skin surfaces of a W- or C-section.

    * ``top_flange_face``    — surfaces with centroid y ≈ +(h/2 + tf)
    * ``bottom_flange_face`` — surfaces with centroid y ≈ -(h/2 + tf)

    These are the face-to-face stacking targets for ``align_to``:
    a planar face's centroid lies on its plane, so matching two
    such centroids puts the surfaces in contact with no offset.
    """
    y_top = h / 2 + tf
    y_bot = -(h / 2 + tf)
    top_tags: list[int] = []
    bot_tags: list[int] = []

    for _, tag in gmsh.model.getEntities(2):
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
        except Exception:
            continue
        y = com[1]
        if abs(y - y_top) < tol:
            top_tags.append(tag)
        elif abs(y - y_bot) < tol:
            bot_tags.append(tag)

    if top_tags:
        labels_comp.add(2, top_tags, name="top_flange_face")
    if bot_tags:
        labels_comp.add(2, bot_tags, name="bottom_flange_face")


def classify_w_web_side_faces(
    h: float,
    tw: float,
    labels_comp,
    *,
    tol: float = 1e-3,
) -> None:
    """Label the web's exposed +x / -x outer faces of a W-section.

    * ``web_left_face``  — exposed surfaces with centroid x ≈ -tw/2 and |y| ≤ h/2
    * ``web_right_face`` — exposed surfaces with centroid x ≈ +tw/2 and |y| ≤ h/2

    The slicing at x=±tw/2 produces internal cut faces between
    adjacent flange sub-volumes at the same x-plane, so we filter
    by exposure (single bounding volume).  The |y|≤h/2 constraint
    further excludes the inner sides of flange wings, which are
    exposed but lie outside the web's vertical extent.
    """
    left_tags: list[int] = []
    right_tags: list[int] = []
    for _, tag in gmsh.model.getEntities(2):
        if not _surface_exposed(tag):
            continue
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
        except Exception:
            continue
        x, y = com[0], com[1]
        if abs(y) > h / 2 + tol:
            continue
        if abs(x - tw / 2) < tol:
            right_tags.append(tag)
        elif abs(x + tw / 2) < tol:
            left_tags.append(tag)
    if left_tags:
        labels_comp.add(2, left_tags, name="web_left_face")
    if right_tags:
        labels_comp.add(2, right_tags, name="web_right_face")


def classify_tee_outer_faces(
    h: float,
    tf: float,
    labels_comp,
    *,
    tol: float = 1e-3,
) -> None:
    """Label the tee's outer +y flange skin and -y stem-bottom skin.

    Tee local frame: flange occupies y ∈ [0, tf], stem hangs to y=-h.

    * ``flange_face`` — surfaces with centroid y ≈ +tf (top of flange).
    * ``stem_face``   — surfaces with centroid y ≈ -h  (bottom of stem).
    """
    flange_tags: list[int] = []
    stem_tags: list[int] = []
    for _, tag in gmsh.model.getEntities(2):
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
        except Exception:
            continue
        y = com[1]
        if abs(y - tf) < tol:
            flange_tags.append(tag)
        elif abs(y + h) < tol:
            stem_tags.append(tag)
    if flange_tags:
        labels_comp.add(2, flange_tags, name="flange_face")
    if stem_tags:
        labels_comp.add(2, stem_tags, name="stem_face")


def classify_angle_outer_faces(
    labels_comp,
    *,
    tol: float = 1e-3,
) -> None:
    """Label the angle's outer-skin faces of each leg.

    Angle local frame: corner at the origin, h-leg in +x, v-leg in +y.

    * ``horizontal_leg_face`` — surfaces with centroid y ≈ 0 (underside of h-leg).
    * ``vertical_leg_face``   — surfaces with centroid x ≈ 0 (back of v-leg).
    """
    h_tags: list[int] = []
    v_tags: list[int] = []
    for _, tag in gmsh.model.getEntities(2):
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
        except Exception:
            continue
        x, y = com[0], com[1]
        if abs(y) < tol:
            h_tags.append(tag)
        elif abs(x) < tol:
            v_tags.append(tag)
    if h_tags:
        labels_comp.add(2, h_tags, name="horizontal_leg_face")
    if v_tags:
        labels_comp.add(2, v_tags, name="vertical_leg_face")


def classify_end_faces(
    length: float,
    labels_comp,
    *,
    tol: float = 1e-3,
) -> None:
    """Label the end-cap surfaces at z=0 and z=length.

    * ``start_face`` — surfaces whose centroid z ≈ 0
    * ``end_face``   — surfaces whose centroid z ≈ length

    These are the natural targets for node-to-surface couplings
    (reference points for applying forces/moments or BCs at the
    member ends).
    """
    start_tags: list[int] = []
    end_tags: list[int] = []

    for _, tag in gmsh.model.getEntities(2):
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
        except Exception:
            continue
        z = com[2]
        if abs(z) < tol:
            start_tags.append(tag)
        elif abs(z - length) < tol:
            end_tags.append(tag)

    if start_tags:
        labels_comp.add(2, start_tags, name="start_face")
    if end_tags:
        labels_comp.add(2, end_tags, name="end_face")
