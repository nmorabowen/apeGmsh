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
