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
