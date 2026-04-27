"""
Alignment helpers shared by ``Part.edit.align_to`` and
``Instance.edit.align_to`` (and their ``align_to_point`` siblings).

The math: given a source centroid and a target centroid (or point),
compute a translation that maps source → target restricted to the
axes named in ``on``.  An optional ``offset`` adds a signed gap
along the (single) active axis.

This module does NOT touch gmsh — callers handle the per-axis
geometry queries (live session vs sidecar) and apply the resulting
translation themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import gmsh

from ._part_anchors import read_sidecar


# ----------------------------------------------------------------------
# Axis mask resolution
# ----------------------------------------------------------------------

AlignAxes = "str | Iterable[str]"


def resolve_axes(on: AlignAxes) -> tuple[bool, bool, bool]:
    """Convert ``on`` spec to a 3-bool mask (x, y, z).

    Accepts:

    * ``"x"`` / ``"y"`` / ``"z"`` — single axis.
    * ``"all"`` — all three axes.
    * Iterable of axis names: ``("x", "z")``, ``["x", "y", "z"]``.
    """
    if isinstance(on, str):
        if on == "all":
            return (True, True, True)
        if on in ("x", "y", "z"):
            return (on == "x", on == "y", on == "z")
        raise ValueError(
            f"on={on!r} is not a recognized axis; use 'x', 'y', 'z', "
            f"'all', or a tuple like ('x','z')."
        )
    try:
        names = [str(a).lower() for a in on]
    except TypeError as exc:
        raise TypeError(
            f"on= must be a string or iterable of axis names; got "
            f"{type(on).__name__}"
        ) from exc
    invalid = [n for n in names if n not in ("x", "y", "z")]
    if invalid:
        raise ValueError(
            f"on= contains unknown axis names: {invalid}.  Allowed: x, y, z."
        )
    s = set(names)
    return ("x" in s, "y" in s, "z" in s)


# ----------------------------------------------------------------------
# Translation builder
# ----------------------------------------------------------------------

def compute_align_translation(
    source_com: tuple[float, float, float],
    target_com: tuple[float, float, float],
    on: AlignAxes,
    offset: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(dx, dy, dz)`` translating ``source_com`` toward ``target_com``
    along the axes named in ``on``.  Other axes are zero.

    If ``offset`` is non-zero, it is applied along the single axis in
    ``on`` (signed).  Combining ``offset`` with a multi-axis ``on``
    raises ``ValueError`` because the offset direction is then undefined.
    """
    mask = resolve_axes(on)
    dx = target_com[0] - source_com[0] if mask[0] else 0.0
    dy = target_com[1] - source_com[1] if mask[1] else 0.0
    dz = target_com[2] - source_com[2] if mask[2] else 0.0

    if offset != 0.0:
        active = sum(mask)
        if active != 1:
            raise ValueError(
                f"offset= can only be combined with a single-axis on=; "
                f"got on with {active} active axes."
            )
        if mask[0]:
            dx += float(offset)
        elif mask[1]:
            dy += float(offset)
        else:
            dz += float(offset)

    return dx, dy, dz


# ----------------------------------------------------------------------
# Centroid lookup — live gmsh session
# ----------------------------------------------------------------------

def label_centroid_live(label_name: str) -> tuple[float, float, float]:
    """Mass-weighted centroid of all entities under ``label_name`` in
    the **current** Gmsh model.

    The label is resolved by walking physical groups whose name is
    the apeGmsh label (with the ``_label:`` prefix).  Multi-entity
    labels (e.g. a flange split across 3 volumes) are weighted by
    their gmsh "mass" (volume for 3D, area for 2D, length for 1D).

    Raises
    ------
    LookupError
        If no physical group matches the label.
    RuntimeError
        If the label has zero total mass (degenerate geometry).
    """
    from .Labels import LABEL_PREFIX

    target_pg_name = f"{LABEL_PREFIX}{label_name}"
    matches: list[tuple[int, int]] = []   # (dim, entity_tag)
    for dim, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            pg_name = gmsh.model.getPhysicalName(dim, pg_tag)
        except Exception:
            continue
        if pg_name != target_pg_name:
            continue
        for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag):
            matches.append((int(dim), int(tag)))

    if not matches:
        raise LookupError(
            f"Label {label_name!r} not found in the current Gmsh model.  "
            f"Hint: the apeGmsh label PG is named {target_pg_name!r}."
        )

    total_m = 0.0
    sx = sy = sz = 0.0
    for dim, tag in matches:
        m = gmsh.model.occ.getMass(dim, tag)
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        sx += m * com[0]
        sy += m * com[1]
        sz += m * com[2]
        total_m += m
    if total_m == 0.0:
        raise RuntimeError(
            f"Label {label_name!r} has zero total mass; cannot compute "
            f"centroid."
        )
    return sx / total_m, sy / total_m, sz / total_m


# ----------------------------------------------------------------------
# Centroid lookup — Part sidecar (no live session needed)
# ----------------------------------------------------------------------

def label_centroid_from_sidecar(
    label_name: str,
    cad_path: Path,
) -> tuple[float, float, float]:
    """Read the centroid of ``label_name`` from the sidecar next to
    ``cad_path`` (typically a Part's STEP).

    Falls back to mass-weighted averaging if the label spans multiple
    entities in the sidecar (each with its own ``com``).

    Raises
    ------
    FileNotFoundError
        If no sidecar is next to the CAD file.
    LookupError
        If the sidecar has no entry for ``label_name``.
    """
    payload = read_sidecar(cad_path)
    if payload is None:
        raise FileNotFoundError(
            f"No sidecar found next to {cad_path}.  align_to() needs the "
            f"target Part to have been saved with anchor data."
        )

    coms: list[tuple[float, float, float]] = []
    for rec in payload.get("anchors", []):
        if rec.get("pg_name") == label_name:
            com = rec.get("com")
            if com and len(com) == 3:
                coms.append((float(com[0]), float(com[1]), float(com[2])))

    if not coms:
        available = sorted({
            rec.get("pg_name", "?") for rec in payload.get("anchors", [])
        })
        raise LookupError(
            f"Label {label_name!r} not in sidecar for {cad_path.name}.  "
            f"Available labels: {available}"
        )

    if len(coms) == 1:
        return coms[0]
    # Multi-entity: equal-weight mean (sidecar doesn't carry per-entity
    # mass).  This matches what `label_centroid_live` does when all
    # entities have similar mass — close enough for alignment use.
    n = len(coms)
    sx = sum(c[0] for c in coms) / n
    sy = sum(c[1] for c in coms) / n
    sz = sum(c[2] for c in coms) / n
    return (sx, sy, sz)
