"""Live-kernel name index for the envelope projector (ADR 0095 S1 host).

Queries gmsh physical groups: ``_label:`` names become labels, the rest
become physical groups. Tags stay evidence. Imported only by the host.

``collect_manifest`` / ``write_names`` dump ``.apegmsh/names.json`` at a
``run_until`` stop — what exists, not what was clicked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gmsh

from apeGmsh._kernel._label_prefix import is_label_pg, strip_prefix

from ._contract import CONTRACT_VERSION
from ._envelope import NameRecord

NAMES_SCHEMA = 1


def lookup_from_gmsh(dim: int, tag: int) -> NameRecord:
    """Resolve labels / PGs / bbox for one BREP ``(dim, tag)``."""
    labels: list[str] = []
    pgs: list[str] = []
    try:
        pg_tags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
    except Exception:
        pg_tags = []
    for pg_tag in pg_tags:
        try:
            name = gmsh.model.getPhysicalName(dim, int(pg_tag))
        except Exception:
            continue
        if not name:
            continue
        if is_label_pg(name):
            labels.append(strip_prefix(name))
        else:
            pgs.append(name)
    bbox = None
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        bbox = (
            float(xmin), float(ymin), float(zmin),
            float(xmax), float(ymax), float(zmax),
        )
    except Exception:
        bbox = None
    return NameRecord(
        labels=tuple(labels),
        physical_groups=tuple(pgs),
        bbox=bbox,
    )


def _unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _element_counts() -> dict[str, int]:
    counts = {"1": 0, "2": 0, "3": 0}
    for dim in (1, 2, 3):
        try:
            _etypes, etags, _enodes = gmsh.model.mesh.getElements(dim)
            counts[str(dim)] = int(sum(len(t) for t in etags))
        except Exception:
            counts[str(dim)] = 0
    return counts


def collect_manifest(*, phase: str) -> dict[str, Any]:
    """Snapshot labels / PGs / counts / bboxes from the live kernel.

    Carries ``contract_version`` (ADR 0095 Amendment 11): ``names.json``
    is the file an out-of-process consumer polls first, and until 1.8.0
    nothing on disk said which contract wrote it — so a consumer's
    INV-17 major gate could never engage against a file, only against a
    live ``status`` call it may not be able to make.
    """
    entities: list[dict[str, Any]] = []
    labels: list[str] = []
    pgs: list[str] = []
    entity_counts = {"0": 0, "1": 0, "2": 0, "3": 0}
    try:
        dimtags = gmsh.model.getEntities()
    except Exception:
        dimtags = []
    for dim, tag in dimtags:
        d, t = int(dim), int(tag)
        key = str(d)
        entity_counts[key] = entity_counts.get(key, 0) + 1
        rec = lookup_from_gmsh(d, t)
        entities.append(
            {
                "dim": d,
                "tag": t,
                "labels": list(rec.labels),
                "physical_groups": list(rec.physical_groups),
                "bbox": list(rec.bbox) if rec.bbox is not None else None,
            }
        )
        labels.extend(rec.labels)
        pgs.extend(rec.physical_groups)
    return {
        "schema": NAMES_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "phase": phase,
        "labels": _unique(labels),
        "physical_groups": _unique(pgs),
        "counts": {"entities": entity_counts, "elements": _element_counts()},
        "entities": entities,
    }


def write_names(path: Path, manifest: dict[str, Any]) -> Path:
    from ._paths import atomic_write_text

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(
        path, json.dumps(manifest, indent=2) + "\n",
    )
