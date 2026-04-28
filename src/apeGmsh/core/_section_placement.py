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

:func:`apply_placement` composes anchor + align + optional user
translate/rotate into a single 4×4 affine matrix and applies it via
one ``gmsh.model.occ.affineTransform`` call.  This deliberately avoids
the chained ``occ.translate`` → ``synchronize`` → ``occ.rotate`` →
``synchronize`` pattern: each intermediate sync renumbers boundary
sub-topology, so chaining transforms means each step has to re-derive
the section's current entity set and re-find every PG entity by COM.
With matrix composition, there is exactly one sync and one PG
snapshot/restore — every label survives by construction.
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

# Tolerance for treating a composed 4x4 as identity (skip the
# affineTransform call entirely).
_IDENTITY_TOL = 1e-12


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
# Apply (compose all transforms into one matrix, single OCC call)
# ---------------------------------------------------------------------

def apply_placement(
    anchor=None,
    align=None,
    length: float | None = None,
    *,
    user_translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    user_rotate: tuple[float, ...] | None = None,
    dimtags: Sequence[tuple[int, int]] | None = None,
    affected: Sequence[tuple[int, int]] | None = None,
) -> None:
    """Compose anchor + align + user transforms into one affine matrix
    and apply it with a single ``gmsh.model.occ.affineTransform`` call.

    Order of application (left to right; rightmost is applied first):

        M = T_user · R_user · R_align · T_anchor

    A point ``p`` in the section's local frame is mapped to
    ``M · p`` in world coordinates.  Anchor translates first (re-origin
    in the local frame), align rotates the extrusion axis, then the
    user's rotate/translate stack as the last operations — matching
    the historical behavior where user kwargs ran *after* anchor/align.

    Composing into one matrix means exactly one ``synchronize()`` and
    one PG snapshot/restore: no chained sync cycles, no stale
    boundary-tag bookkeeping between steps.

    Parameters
    ----------
    anchor : str or (x, y, z), optional
        Passed to :func:`compute_anchor_offset`.  ``None`` is treated
        as ``"start"`` (identity translate).
    align : str or (ax, ay, az), optional
        Passed to :func:`compute_alignment_rotation`.  ``None`` is
        treated as ``"z"`` (identity rotate).
    length : float, optional
        Required for ``"end"``, ``"midspan"``, ``"centroid"`` anchors.
    user_translate : (dx, dy, dz), default ``(0, 0, 0)``
        Translation applied AFTER anchor + align + user_rotate.
    user_rotate : (angle, ax, ay, az) or (angle, ax, ay, az, cx, cy, cz), optional
        4-tuple rotates about the world origin; 7-tuple rotates about
        ``(cx, cy, cz)``.  Applied AFTER anchor + align, BEFORE
        user_translate.  ``None`` means no user rotation.
    dimtags : list of (dim, tag), optional
        Entities to transform.  When ``None``, walks all entities of
        the highest dimension present in the active gmsh model — the
        right default for section factories with the session to
        themselves.  Builders sharing a session must pass an explicit
        list.
    affected : list of (dim, tag), optional
        Set of (dim, tag) whose PG membership might be affected by the
        transform.  Used to scope the PG snapshot/restore so untouched
        PGs in a shared parent session are left alone.  ``None`` means
        "every PG in the model is potentially affected" — correct when
        the section owns the session.
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

    # Build the four constituent matrices.
    offset = compute_anchor_offset(
        anchor if anchor is not None else "start",
        length=length,
        dimtags=dimtags,
    )
    T_anchor = _translate_matrix(*offset)

    align_rot = compute_alignment_rotation(align if align is not None else "z")
    R_align = (
        _identity_matrix()
        if align_rot is None
        else _rotate_matrix(*align_rot)
    )

    if user_rotate is None:
        R_user = _identity_matrix()
    elif len(user_rotate) == 4:
        angle, ax, ay, az = (float(v) for v in user_rotate)
        R_user = _rotate_matrix(angle, ax, ay, az)
    elif len(user_rotate) == 7:
        angle, ax, ay, az, cx, cy, cz = (float(v) for v in user_rotate)
        R_user = _rotate_matrix(angle, ax, ay, az, cx=cx, cy=cy, cz=cz)
    else:
        raise ValueError(
            f"user_rotate must be a 4-tuple (angle, ax, ay, az) or a "
            f"7-tuple (angle, ax, ay, az, cx, cy, cz); got "
            f"{len(user_rotate)} components."
        )

    T_user = _translate_matrix(*(float(v) for v in user_translate))

    # Compose: M = T_user · R_user · R_align · T_anchor.  Right-most
    # matrix applies first to a point.
    M = _matmul(T_user, _matmul(R_user, _matmul(R_align, T_anchor)))

    if _is_identity(M):
        return

    affected_set: set[tuple[int, int]] | None = (
        None if affected is None
        else {(int(d), int(t)) for d, t in affected}
    )

    # Rigid OCC transforms followed by synchronize() drop physical
    # groups whose entities were touched: top-dim tags survive but
    # boundary sub-topology gets renumbered.  Snapshot per-entity
    # COMs first; after the single sync, re-find each entity by the
    # transformed COM.  affected_set scopes this to the section's
    # PGs so a shared parent session's untouched PGs are left alone.
    snap = _snapshot_physical_groups(affected_set)

    gmsh.model.occ.affineTransform(dimtags, _matrix_to_gmsh(M))
    gmsh.model.occ.synchronize()

    _restore_physical_groups(snap, M, affected_set)


# ---------------------------------------------------------------------
# 4×4 matrix helpers (plain Python, no numpy dependency)
# ---------------------------------------------------------------------

# Matrix layout: list of 4 rows, each a list of 4 floats. (Row-major.)
_Mat4 = list


def _identity_matrix() -> _Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translate_matrix(dx: float, dy: float, dz: float) -> _Mat4:
    return [
        [1.0, 0.0, 0.0, float(dx)],
        [0.0, 1.0, 0.0, float(dy)],
        [0.0, 0.0, 1.0, float(dz)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotate_matrix(
    angle: float,
    ax: float, ay: float, az: float,
    *,
    cx: float = 0.0, cy: float = 0.0, cz: float = 0.0,
) -> _Mat4:
    """Rotation by ``angle`` radians about axis ``(ax, ay, az)`` through
    point ``(cx, cy, cz)``.  Implemented as T(c) · R · T(-c).
    """
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0.0:
        return _identity_matrix()
    kx, ky, kz = ax / norm, ay / norm, az / norm
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    R = [
        [c + kx * kx * C,      kx * ky * C - kz * s, kx * kz * C + ky * s, 0.0],
        [ky * kx * C + kz * s, c + ky * ky * C,      ky * kz * C - kx * s, 0.0],
        [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C,      0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    if cx == 0.0 and cy == 0.0 and cz == 0.0:
        return R
    return _matmul(
        _translate_matrix(cx, cy, cz),
        _matmul(R, _translate_matrix(-cx, -cy, -cz)),
    )


def _matmul(A: _Mat4, B: _Mat4) -> _Mat4:
    """4×4 matrix product."""
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                A[i][0] * B[0][j]
                + A[i][1] * B[1][j]
                + A[i][2] * B[2][j]
                + A[i][3] * B[3][j]
            )
    return out


def _apply_matrix(
    M: _Mat4, p: tuple[float, float, float],
) -> tuple[float, float, float]:
    px, py, pz = p
    return (
        M[0][0] * px + M[0][1] * py + M[0][2] * pz + M[0][3],
        M[1][0] * px + M[1][1] * py + M[1][2] * pz + M[1][3],
        M[2][0] * px + M[2][1] * py + M[2][2] * pz + M[2][3],
    )


def _is_identity(M: _Mat4, tol: float = _IDENTITY_TOL) -> bool:
    I = _identity_matrix()
    for i in range(4):
        for j in range(4):
            if abs(M[i][j] - I[i][j]) > tol:
                return False
    return True


def _matrix_to_gmsh(M: _Mat4) -> list[float]:
    """Flatten the first 3 rows into a 12-element list.

    ``gmsh.model.occ.affineTransform`` takes the first 3 rows of the
    4×4 affine matrix in row-major order; the implicit last row
    ``[0, 0, 0, 1]`` is added by gmsh.
    """
    return [
        M[0][0], M[0][1], M[0][2], M[0][3],
        M[1][0], M[1][1], M[1][2], M[1][3],
        M[2][0], M[2][1], M[2][2], M[2][3],
    ]


# ---------------------------------------------------------------------
# Physical-group snapshot/restore
# ---------------------------------------------------------------------

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
    M: _Mat4,
    affected_set: set[tuple[int, int]] | None = None,
) -> None:
    """Recreate snapshot PGs by matching transformed COMs.

    For each snapshotted entity COM, applies ``M`` only to entities
    that were ``affected``, then finds the live entity at that dim
    whose current COM is closest.  Entities that were not affected
    keep their tag and original COM as-is.
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
                expected = _apply_matrix(M, com)
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
