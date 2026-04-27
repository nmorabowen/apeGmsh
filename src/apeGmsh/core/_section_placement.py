"""
Section placement helpers — pure math for ``anchor=`` and ``align=``.
=====================================================================

These helpers compute the translation and rotation that section
factories apply after building geometry, to support the ``anchor=``
and ``align=`` keyword arguments.

Two pure functions:

* :func:`compute_anchor_offset` — translation in the section's local
  frame.  Geometry built 0 .. +length along Z is shifted so the
  chosen anchor lands at the origin.

* :func:`compute_alignment_rotation` — rotation about origin applied
  *after* the anchor translation, to reorient the extrusion axis from
  +Z to whichever world axis the user wants.

Both raise :class:`ValueError` on bad input.  ``compute_anchor_offset``
reads ``gmsh.model.occ`` only for the ``"centroid"`` mode (mass-weighted
XY centroid over the active model's top-dimension entities).
"""
from __future__ import annotations

import math
from typing import Sequence

import gmsh


# Tolerance for "is this vector parallel to ±Z?" checks in the tuple
# branch of compute_alignment_rotation.  Tighter than typical bbox
# tolerances (1e-3) since we're operating on normalized direction
# vectors, but loose enough to handle float round-trip from user input.
_PARALLEL_TOL = 1e-9


# ---------------------------------------------------------------------
# Anchor (translation)
# ---------------------------------------------------------------------

def compute_anchor_offset(
    anchor,
    *,
    length: float | None = None,
    dimtags: Sequence[tuple[int, int]] | None = None,
) -> tuple[float, float, float]:
    """Return ``(dx, dy, dz)`` translation for the requested anchor.

    The translation is applied in the section's **local frame** (the
    frame in which the factory just built geometry: extrusion runs
    0 → +length along Z).  Named modes other than ``"start"`` require
    a non-None ``length``.

    Parameters
    ----------
    anchor : str or (x, y, z) tuple
        One of ``"start"``, ``"end"``, ``"midspan"``, ``"centroid"``,
        or a 3-tuple of floats specifying an explicit local point that
        should become the new origin.
    length : float, optional
        Extrusion length.  Required for ``"end"``, ``"midspan"``,
        ``"centroid"``.  Ignored for ``"start"`` and tuple anchors.
    dimtags : list of (dim, tag), optional
        Restrict the centroid computation to these entities.  When
        ``None`` (the default), walks all entities of the highest
        dimension present in the active gmsh model.

    Returns
    -------
    (dx, dy, dz) : tuple of float

    Raises
    ------
    ValueError
        Unknown anchor string, named mode passed without a length,
        or wrong-length tuple.
    """
    if isinstance(anchor, str):
        if anchor == "start":
            return (0.0, 0.0, 0.0)
        if anchor in ("end", "midspan", "centroid"):
            if length is None:
                raise ValueError(
                    f"anchor={anchor!r} requires a length; "
                    f"got length=None."
                )
            if anchor == "end":
                return (0.0, 0.0, -float(length))
            if anchor == "midspan":
                return (0.0, 0.0, -float(length) / 2.0)
            # centroid
            cx, cy = _xy_centroid(dimtags)
            return (-cx, -cy, -float(length) / 2.0)
        raise ValueError(
            f"Unknown anchor {anchor!r}; expected one of "
            f"'start', 'end', 'midspan', 'centroid', or an "
            f"(x, y, z) tuple."
        )

    # Tuple form
    try:
        seq = tuple(float(v) for v in anchor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"anchor must be a string or (x, y, z) tuple of floats; "
            f"got {anchor!r}."
        ) from exc
    if len(seq) != 3:
        raise ValueError(
            f"anchor tuple must have exactly 3 components; "
            f"got {len(seq)} ({anchor!r})."
        )
    x, y, z = seq
    return (-x, -y, -z)


def _xy_centroid(
    dimtags: Sequence[tuple[int, int]] | None,
) -> tuple[float, float]:
    """Mass-weighted XY centroid over the chosen entities.

    When ``dimtags`` is None, walks all entities at the highest
    dimension present in the active gmsh model.  Returns (0.0, 0.0)
    if no entities are found or total mass is zero.
    """
    if dimtags is None:
        all_ents = gmsh.model.getEntities()
        if not all_ents:
            return (0.0, 0.0)
        top_dim = max(d for d, _ in all_ents)
        targets = [(d, t) for d, t in all_ents if d == top_dim]
    else:
        targets = [(int(d), int(t)) for d, t in dimtags]
        if not targets:
            return (0.0, 0.0)

    total_mass = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for d, t in targets:
        try:
            m = float(gmsh.model.occ.getMass(d, t))
            cx, cy, _cz = gmsh.model.occ.getCenterOfMass(d, t)
        except Exception:
            continue
        if m <= 0.0:
            continue
        total_mass += m
        sum_x += m * float(cx)
        sum_y += m * float(cy)

    if total_mass <= 0.0:
        return (0.0, 0.0)
    return (sum_x / total_mass, sum_y / total_mass)


# ---------------------------------------------------------------------
# Align (rotation)
# ---------------------------------------------------------------------

def compute_alignment_rotation(
    align,
) -> tuple[float, float, float, float] | None:
    """Return ``(angle, ax, ay, az)`` for the requested alignment.

    Applied as a rotation about the origin AFTER the anchor
    translation.  The rotation maps the section's local +Z (the
    extrusion axis) to the requested world direction.

    Parameters
    ----------
    align : str or (ax, ay, az) tuple
        * ``"z"`` (default in callers) — identity; returns ``None``.
        * ``"x"`` — 120° about (1, 1, 1).  Cycles X→Y, Y→Z, Z→X.
        * ``"y"`` — 180° about (0, 1, 1).  Maps Z→Y, Y→Z, X→-X.
        * ``(ax, ay, az)`` — shortest-arc rotation from +Z to the
          (auto-normalized) direction.  Special cases: parallel to +Z
          returns ``None``; parallel to -Z returns 180° about +X.

    Returns
    -------
    (angle, ax, ay, az) or None
        ``None`` for the identity (so callers can skip the gmsh call).

    Raises
    ------
    ValueError
        Unknown align string, wrong-length tuple, or zero vector.
    """
    if isinstance(align, str):
        if align == "z":
            return None
        if align == "x":
            return (2.0 * math.pi / 3.0, 1.0, 1.0, 1.0)
        if align == "y":
            return (math.pi, 0.0, 1.0, 1.0)
        raise ValueError(
            f"Unknown align {align!r}; expected one of 'x', 'y', 'z', "
            f"or an (ax, ay, az) tuple."
        )

    # Tuple form
    try:
        seq = tuple(float(v) for v in align)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"align must be a string or (ax, ay, az) tuple of floats; "
            f"got {align!r}."
        ) from exc
    if len(seq) != 3:
        raise ValueError(
            f"align tuple must have exactly 3 components; "
            f"got {len(seq)} ({align!r})."
        )
    vx, vy, vz = seq
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0.0:
        raise ValueError("align direction must be nonzero.")
    nx, ny, nz = vx / norm, vy / norm, vz / norm

    # Shortest-arc rotation from +Z = (0, 0, 1) to (nx, ny, nz).
    # axis = z_hat × n  = (-ny, nx, 0); angle = acos(nz)
    if nz >= 1.0 - _PARALLEL_TOL:
        return None  # parallel to +Z, no rotation needed
    if nz <= -1.0 + _PARALLEL_TOL:
        # Antiparallel — 180° about any axis perpendicular to Z.
        return (math.pi, 1.0, 0.0, 0.0)
    angle = math.acos(nz)
    return (angle, -ny, nx, 0.0)


# ---------------------------------------------------------------------
# Apply (translate + rotate)
# ---------------------------------------------------------------------

def apply_placement(
    anchor,
    align,
    length: float | None = None,
    *,
    dimtags: Sequence[tuple[int, int]] | None = None,
    affected: Sequence[tuple[int, int]] | None = None,
) -> None:
    """Apply ``anchor`` translation then ``align`` rotation in-place.

    Convenience wrapper that resolves both kwargs via the pure helpers
    above and calls ``gmsh.model.occ.translate`` / ``rotate`` on the
    chosen entities.  Skips the gmsh call when the resolved transform
    is the identity.

    Parameters
    ----------
    anchor : str or (x, y, z) tuple
        Passed through to :func:`compute_anchor_offset`.
    align : str or (ax, ay, az) tuple
        Passed through to :func:`compute_alignment_rotation`.
    length : float, optional
        Required for ``"end"``, ``"midspan"``, and ``"centroid"``
        anchors.  Ignored for ``"start"`` and tuple anchors.
    dimtags : list of (dim, tag), optional
        Entities to transform.  When ``None``, walks all entities of
        the highest dimension present in the active gmsh model — the
        right default for section factories building in their own
        Part session.  Builders that share a session with other parts
        must pass an explicit list to avoid moving unrelated geometry.
    affected : list of (dim, tag), optional
        Full set of entities whose tag IDs may be invalidated by the
        transform — usually ``dimtags`` plus the recursive boundary
        sub-topology (faces of rotated volumes get renumbered even
        though the volumes themselves keep their tag).  Used to scope
        the PG snapshot/restore so untouched PGs in a shared parent
        session are left alone.  Defaults to ``None`` — every entity
        in the model is treated as affected, which is correct for the
        Part-only case where the section has the session to itself.
    """
    if dimtags is None:
        all_ents = gmsh.model.getEntities()
        if not all_ents:
            return
        top_dim = max(d for d, _ in all_ents)
        dimtags = [(d, t) for d, t in all_ents if d == top_dim]
    else:
        dimtags = [(int(d), int(t)) for d, t in dimtags]
    if not dimtags:
        return

    dx, dy, dz = compute_anchor_offset(anchor, length=length, dimtags=dimtags)
    rot = compute_alignment_rotation(align)
    needs_translate = (dx != 0.0 or dy != 0.0 or dz != 0.0)
    needs_rotate = rot is not None
    if not (needs_translate or needs_rotate):
        return

    if affected is None:
        affected_set: set[tuple[int, int]] | None = None
    else:
        affected_set = {(int(d), int(t)) for d, t in affected}

    # Rigid OCC transforms followed by synchronize() drop the
    # physical groups whose entities were touched.  Top-dim entity
    # tags survive translate/rotate, but the OCC kernel renumbers
    # the boundary sub-topology (faces of rotated volumes).  So we
    # snapshot per-entity COMs before the transform and re-find each
    # entity by COM after — same matching strategy that the import
    # path uses, just in-process.  ``affected_set`` (when given) keeps
    # us from touching PGs in a shared session that the transform
    # didn't move.
    snap = _snapshot_physical_groups(affected_set)

    if needs_translate:
        gmsh.model.occ.translate(dimtags, dx, dy, dz)
        gmsh.model.occ.synchronize()

    if needs_rotate:
        angle, ax, ay, az = rot
        gmsh.model.occ.rotate(dimtags, 0.0, 0.0, 0.0, ax, ay, az, angle)
        gmsh.model.occ.synchronize()

    _restore_physical_groups(snap, (dx, dy, dz), rot, affected_set)


def _snapshot_physical_groups(
    affected_set: set[tuple[int, int]] | None = None,
) -> list[dict]:
    """Capture PGs (label and user) before a placement transform.

    Records ``{dim, pg_tag, name, entity_coms}`` per group, where
    ``entity_coms`` is a list of ``(tag, (cx, cy, cz), is_affected)``
    for every entity in the group at snapshot time.  The COM is read
    fresh so the post-transform restore can match by transformed-COM
    rather than by potentially-renumbered entity tag.

    When ``affected_set`` is given, only PGs that contain at least one
    affected entity are snapshotted.  Untouched PGs are left alone —
    the OCC sync did not drop them.
    """
    snap: list[dict] = []
    for d, pg in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(d, pg)
        ents = list(gmsh.model.getEntitiesForPhysicalGroup(d, pg))
        coms: list[tuple[int, tuple[float, float, float], bool]] = []
        any_affected = False
        for t in ents:
            try:
                cx, cy, cz = gmsh.model.occ.getCenterOfMass(int(d), int(t))
            except Exception:
                continue
            is_aff = (
                affected_set is None
                or (int(d), int(t)) in affected_set
            )
            if is_aff:
                any_affected = True
            coms.append(
                (int(t), (float(cx), float(cy), float(cz)), is_aff),
            )
        if affected_set is not None and not any_affected:
            continue
        snap.append({
            'dim':          int(d),
            'pg_tag':       int(pg),
            'name':         name,
            'entity_coms':  coms,
        })
    return snap


def _restore_physical_groups(
    snap: list[dict],
    translate: tuple[float, float, float],
    rotate: tuple[float, float, float, float] | None,
    affected_set: set[tuple[int, int]] | None = None,
) -> None:
    """Recreate snapshot PGs by matching transformed COMs.

    For each snapshotted entity COM, applies the placement transform
    only to entities that were ``affected``, then finds the live
    entity at that dim whose current COM is closest.  Entities that
    were not affected keep their tag and original COM as-is.
    """
    if not snap:
        return

    # Cache live (tag, com) per dim so we don't re-walk for every PG.
    live_by_dim: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for d, t in gmsh.model.getEntities():
        try:
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(int(d), int(t))
        except Exception:
            continue
        live_by_dim.setdefault(int(d), []).append(
            (int(t), (float(cx), float(cy), float(cz))),
        )

    for entry in snap:
        d = entry['dim']
        try:
            gmsh.model.removePhysicalGroups([(d, entry['pg_tag'])])
        except Exception:
            pass

        new_tags: list[int] = []
        live = live_by_dim.get(d, [])
        for old_tag, com, is_aff in entry['entity_coms']:
            if is_aff:
                expected = _transform_point(com, translate, rotate)
                best_tag = _nearest_tag(live, expected)
                if best_tag is not None:
                    new_tags.append(best_tag)
            else:
                # Entity was not transformed — keep the original tag.
                new_tags.append(old_tag)

        new_tags = sorted(set(new_tags))
        if not new_tags:
            continue
        new_pg = gmsh.model.addPhysicalGroup(d, new_tags)
        if entry['name']:
            gmsh.model.setPhysicalName(d, new_pg, entry['name'])


def _transform_point(
    p: tuple[float, float, float],
    translate: tuple[float, float, float],
    rotate: tuple[float, float, float, float] | None,
) -> tuple[float, float, float]:
    """Apply translate-then-rotate (the order ``apply_placement`` uses)
    to a point.  Rotation is about the world origin via Rodrigues.
    """
    px, py, pz = p
    dx, dy, dz = translate
    px, py, pz = px + dx, py + dy, pz + dz
    if rotate is None:
        return (px, py, pz)
    angle, ax, ay, az = rotate
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0.0:
        return (px, py, pz)
    kx, ky, kz = ax / norm, ay / norm, az / norm
    c, s = math.cos(angle), math.sin(angle)
    # v' = v c + (k × v) s + k (k·v)(1-c)
    cross_x = ky * pz - kz * py
    cross_y = kz * px - kx * pz
    cross_z = kx * py - ky * px
    dot = kx * px + ky * py + kz * pz
    return (
        px * c + cross_x * s + kx * dot * (1.0 - c),
        py * c + cross_y * s + ky * dot * (1.0 - c),
        pz * c + cross_z * s + kz * dot * (1.0 - c),
    )


def _nearest_tag(
    live: list[tuple[int, tuple[float, float, float]]],
    target: tuple[float, float, float],
    *,
    tol: float = 1e-3,
) -> int | None:
    """Return the live entity tag whose COM is nearest ``target``.

    Returns None if no live entity is within ``tol`` distance.
    """
    if not live:
        return None
    best_tag = None
    best_d2 = float('inf')
    tx, ty, tz = target
    for tag, (cx, cy, cz) in live:
        dx, dy, dz = cx - tx, cy - ty, cz - tz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best_tag = tag
    if best_d2 > tol * tol:
        return None
    return best_tag
