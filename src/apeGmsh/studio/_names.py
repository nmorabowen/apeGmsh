"""Live-kernel name index for the envelope projector (ADR 0095 S1 host).

Queries gmsh physical groups: ``_label:`` names become labels, the rest
become physical groups. Tags stay evidence. Imported only by the host.
"""

from __future__ import annotations

import gmsh

from apeGmsh._kernel._label_prefix import is_label_pg, strip_prefix

from ._envelope import NameRecord


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
