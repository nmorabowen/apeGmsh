"""
Part label anchors — carry Part entity labels across the STEP/IGES
round-trip so imported instances can be addressed by the names the
user gave them when the Part was built.

Architecture
------------
STEP files don't carry apeGmsh label strings, so we write a JSON
sidecar next to the CAD file::

    {name}.step               ← the geometry
    {name}.step.apegmsh.json  ← the label -> COM map

Each entry in the sidecar stores:

* the user's label
* the entity's dimension (0/1/2/3)
* the entity's center of mass in the Part's local coordinate system
* the registry ``kind`` for debugging

At import time, :func:`rebind_labels` transforms the stored COMs
by the same translate + rotate the caller passed to
``parts.add()``, then matches each transformed COM to the nearest
imported entity of the same dim.

Why COMs, not embedded STEP metadata
------------------------------------
1. Gmsh's ``gmsh.write()`` goes through OCC's plain STEP writer,
   which does not expose XDE / TDataStd_Name.  Embedding labels
   would require bypassing Gmsh entirely.
2. Third-party STEP readers (FreeCAD, Fusion, etc.) silently drop
   XDE anyway, so the labels would not travel.
3. A JSON sidecar is grep-able, extensible, and trivially
   version-controlled if the user wants.

Limitations
-----------
* Only *user-named* entities are anchored.  Labels that match the
  auto-generated ``f"{kind}_{tag}"`` pattern are filtered out.
* Rebinding is valid at ``parts.add()`` time.  Subsequent boolean
  operations (``fragment_all``, ``fuse_group``, etc.) renumber
  entities and the stored ``label_to_tag`` map becomes stale by
  design.  Users who need post-fragment labels should re-resolve.
* Symmetric Parts may produce multiple entities equidistant from
  a stored COM.  We keep the first by Gmsh tag and emit a warning.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gmsh as _gmsh_t


# =====================================================================
# Sidecar file naming
# =====================================================================

_SIDECAR_SUFFIX = ".apegmsh.json"


def sidecar_path(cad_path: Path | str) -> Path:
    """Return the sidecar path for a given CAD file.

    Keeps the full CAD extension so ``foo.step`` and ``foo.iges``
    in the same directory do not collide.
    """
    cad = Path(cad_path)
    return cad.with_name(cad.name + _SIDECAR_SUFFIX)


# =====================================================================
# Writing anchors (Part side)
# =====================================================================

def collect_anchors(gmsh_module: "_gmsh_t") -> list[dict]:
    """Walk the Gmsh label PGs (``_label:*``) and return serialisable
    anchor records — one per entity in a named label.

    Only **label** PGs (Tier 1) are captured, not user-facing PGs
    (Tier 2).  The sidecar carries labels because they are the
    geometry-time naming system; solver-facing PGs are created
    explicitly by the user in the Assembly and don't need to
    travel through files.

    Parameters
    ----------
    gmsh_module : module
        The ``gmsh`` module — passed in so this function stays
        unit-testable without a live import.

    Returns
    -------
    list[dict]
        One record per entity in a named label PG.  Each record
        has ``pg_name`` (the label name WITHOUT the ``_label:``
        prefix), ``dim``, and ``com``.
    """
    from .Labels import is_label_pg, strip_prefix

    records: list[dict] = []
    for dim, pg_tag in gmsh_module.model.getPhysicalGroups():
        raw_name = gmsh_module.model.getPhysicalName(dim, pg_tag)
        if not raw_name or not is_label_pg(raw_name):
            continue
        label_name = strip_prefix(raw_name)
        ent_tags = gmsh_module.model.getEntitiesForPhysicalGroup(dim, pg_tag)
        for tag in ent_tags:
            try:
                com = gmsh_module.model.occ.getCenterOfMass(int(dim), int(tag))
            except Exception:
                continue
            records.append({
                'pg_name': str(label_name),
                'dim':     int(dim),
                'com':     [float(com[0]), float(com[1]), float(com[2])],
            })
    return records


def write_sidecar(
    cad_path: Path | str,
    anchors: list[dict],
    *,
    part_name: str | None = None,
    format_version: int = 1,
) -> Path | None:
    """Write a JSON sidecar next to ``cad_path`` containing anchors.

    Returns the sidecar path, or ``None`` if ``anchors`` is empty
    (no user-named entities -> no sidecar).
    """
    if not anchors:
        return None
    cad = Path(cad_path)
    out = sidecar_path(cad)
    payload = {
        'format_version': format_version,
        'part_name':      part_name or cad.stem,
        'anchors':        anchors,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# =====================================================================
# Reading anchors (Assembly side)
# =====================================================================

def read_sidecar(cad_path: Path | str) -> dict | None:
    """Read the sidecar next to ``cad_path`` if one exists.

    Returns
    -------
    dict or None
        Parsed sidecar payload, or ``None`` when the sidecar is
        missing or malformed.  Malformed sidecars emit a warning;
        they are never a hard error because the geometry import
        must still succeed even if rebinding is unavailable.
    """
    out = sidecar_path(cad_path)
    if not out.exists():
        return None
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(
            f"apeGmsh: sidecar {out.name} is unreadable ({exc!r}); "
            f"label rebinding disabled for this import.",
            stacklevel=2,
        )
        return None


# =====================================================================
# Transform application
# =====================================================================

def _rodrigues_rotate(
    point: np.ndarray,
    axis:  np.ndarray,
    angle: float,
    center: np.ndarray,
) -> np.ndarray:
    """Rotate ``point`` around an axis through ``center`` by ``angle``
    radians using Rodrigues' formula.
    """
    # Translate to origin, rotate, translate back
    p = point - center
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    rotated = p * c + np.cross(k, p) * s + k * np.dot(k, p) * (1.0 - c)
    return rotated + center


def apply_transform_to_com(
    com: np.ndarray,
    translate: tuple[float, float, float],
    rotate: tuple[float, ...] | None,
) -> np.ndarray:
    """Apply the same ``rotate -> translate`` sequence that
    :func:`_parts_registry._apply_transforms` applies to the
    imported geometry.

    The order matters: rotate first, then translate.  This matches
    the implementation in ``_parts_registry.PartsRegistry._apply_transforms``.
    """
    p = np.asarray(com, dtype=float)
    if rotate is not None:
        if len(rotate) == 4:
            angle, ax, ay, az = rotate
            cx = cy = cz = 0.0
        elif len(rotate) == 7:
            angle, ax, ay, az, cx, cy, cz = rotate
        else:
            raise ValueError(
                "rotate must be (angle, ax, ay, az) or "
                "(angle, ax, ay, az, cx, cy, cz)"
            )
        axis_v  = np.array([ax, ay, az], dtype=float)
        center  = np.array([cx, cy, cz], dtype=float)
        p = _rodrigues_rotate(p, axis_v, float(angle), center)

    dx, dy, dz = translate
    return p + np.array([dx, dy, dz], dtype=float)


# =====================================================================
# Label rebinding (the big one)
# =====================================================================

def rebind_physical_groups(
    anchors: list[dict],
    imported_entities: dict[int, list[int]],
    translate: tuple[float, float, float],
    rotate: tuple[float, ...] | None,
    gmsh_module: "_gmsh_t",
    *,
    tolerance: float = 1e-4,
    characteristic_length: float | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Match stored PG anchors against imported entities by COM.

    Parameters
    ----------
    anchors : list[dict]
        Anchors as produced by :func:`collect_anchors` — each must
        have ``pg_name``, ``dim``, and ``com``.
    imported_entities : dict
        The ``entities`` dict produced by ``_import_cad`` — maps
        ``dim`` to the list of imported entity tags at that dim.
    translate, rotate :
        Placement transforms applied to the import (identical to
        the arguments passed to ``parts.add()``).
    gmsh_module : module
        The ``gmsh`` module — passed in so this function stays
        unit-testable.
    tolerance : float
        Matching radius as a fraction of ``characteristic_length``.
    characteristic_length : float, optional
        Model-scale length used to size ``tolerance``.  Defaults
        to the bounding-box diagonal of the imported entities, or
        ``1.0`` when the geometry is empty.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        ``pg_name -> [(dim, tag), ...]`` map.  A PG with multiple
        entities (e.g. two surfaces in the same group) gets
        multiple matches.  PGs with no match are omitted.
    """
    if not anchors or not imported_entities:
        return {}

    if characteristic_length is None:
        characteristic_length = _imported_characteristic_length(
            imported_entities, gmsh_module,
        )
    abs_tol = float(tolerance) * max(characteristic_length, 1.0)

    # Cache imported COMs per dim.
    imported_coms: dict[int, list[tuple[int, np.ndarray]]] = {}
    for dim, tags in imported_entities.items():
        for tag in tags:
            try:
                com = gmsh_module.model.occ.getCenterOfMass(int(dim), int(tag))
            except Exception:
                continue
            imported_coms.setdefault(dim, []).append(
                (int(tag), np.asarray(com, dtype=float)),
            )

    pg_matches: dict[str, list[tuple[int, int]]] = {}

    for anchor in anchors:
        pg_name = anchor.get('pg_name') or anchor.get('label', '')
        dim = int(anchor['dim'])
        com = np.asarray(anchor['com'], dtype=float)
        expected = apply_transform_to_com(com, translate, rotate)

        candidates = imported_coms.get(dim, [])
        if not candidates:
            continue

        best_tag: int | None = None
        best_dist = float('inf')
        for tag, cand_com in candidates:
            d = float(np.linalg.norm(cand_com - expected))
            if d < best_dist:
                best_tag = tag
                best_dist = d

        if best_tag is None or best_dist > abs_tol:
            warnings.warn(
                f"apeGmsh: PG anchor {pg_name!r} (dim={dim}) has no "
                f"import match within {abs_tol:.3e} — best distance "
                f"was {best_dist:.3e}.",
                stacklevel=2,
            )
            continue

        pg_matches.setdefault(pg_name, []).append((dim, int(best_tag)))

    return pg_matches


def _imported_characteristic_length(
    imported_entities: dict[int, list[int]],
    gmsh_module: "_gmsh_t",
) -> float:
    """Return the bounding-box diagonal of the imported entities
    to use as the tolerance scale for :func:`rebind_labels`.
    """
    xmin = ymin = zmin = float('inf')
    xmax = ymax = zmax = float('-inf')
    any_entity = False
    for dim, tags in imported_entities.items():
        for tag in tags:
            try:
                bb = gmsh_module.model.getBoundingBox(int(dim), int(tag))
            except Exception:
                continue
            any_entity = True
            xmin = min(xmin, bb[0])
            ymin = min(ymin, bb[1])
            zmin = min(zmin, bb[2])
            xmax = max(xmax, bb[3])
            ymax = max(ymax, bb[4])
            zmax = max(zmax, bb[5])
    if not any_entity:
        return 1.0
    return float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
