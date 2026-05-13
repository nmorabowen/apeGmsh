"""``apeGmsh.cuts`` — section-cut spec producer.

apeGmsh produces :class:`SectionCutDef` objects from CAD-level geometry
and physical groups; STKO_to_python's kernel consumes them. See
``ARCHITECTURE.md`` next to this file for the full design note.

Public surface (v1)::

    from apeGmsh.cuts import SectionCutDef

    cut_def = SectionCutDef(
        plane_point=(0.0, 0.0, 2500.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(101, 102, 103),
        label="Story 3 base shear",
    )

    spec = cut_def.to_spec()    # requires STKO_to_python installed

Phases 2–5 will add ``plane_from_physical_surface``,
``FemToOpsTagMap``, ``SectionCutDef.from_planar_pg``, and
``SectionSweepDef``.
"""
from __future__ import annotations

from ._defs import SectionCutDef
from ._drift import DriftDef, DriftSweepDef
from ._h5_io import persist_to_h5, read_cuts_and_sweeps
from ._planes import (
    plane_from_coords,
    plane_from_physical_surface,
    plane_from_three_points,
    plane_horizontal,
    plane_vertical,
)
from ._polygons import bounding_polygon_from_physical_surface
from ._preflight import PreflightError, PreflightIssue, PreflightReport
from ._sweeps import SectionSweepDef
from ._tag_map import FemToOpsTagMap

__all__ = [
    "SectionCutDef",
    "SectionSweepDef",
    "DriftDef",
    "DriftSweepDef",
    "FemToOpsTagMap",
    "PreflightError",
    "PreflightIssue",
    "PreflightReport",
    "persist_to_h5",
    "read_cuts_and_sweeps",
    "plane_horizontal",
    "plane_vertical",
    "plane_from_three_points",
    "plane_from_coords",
    "plane_from_physical_surface",
    "bounding_polygon_from_physical_surface",
]
