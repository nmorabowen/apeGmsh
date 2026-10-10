"""
apeGmsh.parts — parametric part primitives.

This package collects reusable :class:`~apeGmsh.core.Part.Part`
subclasses that build common parametric geometry layouts.  The
flagship primitive is :class:`DRMBox`, the layered soil box used by
the Domain Reduction Method workflow.

Public API
----------

* :class:`Axis1D` — 1-D layered axis description
* :class:`DRMBox` — Domain-Reduction-Method box (Part subclass)
* :class:`DRMBoxResult` — frozen summary returned by
  ``g.parts.add_DRM_box(...)``
* :class:`SoilLattice`, :class:`NearField`, :class:`Pit`, :class:`Exterior`
  — inputs of ``g.parts.add_station_soil_box(...)`` (ADR 0118)
"""
from __future__ import annotations

from ._axis1d import Axis1D
from .drm_box import DRMBox, DRMBoxResult
from .station_box import (
    Exterior,
    NearField,
    Pit,
    SoilLattice,
    StationSoilBoxResult,
)

__all__ = [
    "Axis1D",
    "DRMBox",
    "DRMBoxResult",
    "Exterior",
    "NearField",
    "Pit",
    "SoilLattice",
    "StationSoilBoxResult",
]
