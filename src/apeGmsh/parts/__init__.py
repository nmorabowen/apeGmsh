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

The station-box names are resolved lazily (PEP 562): importing
``apeGmsh`` must not pull ``station_box`` (and through it
``plane_wave_box``) into the import graph.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._axis1d import Axis1D
from .drm_box import DRMBox, DRMBoxResult

if TYPE_CHECKING:
    from .station_box import (
        Exterior,
        NearField,
        Pit,
        SoilLattice,
        StationSoilBoxResult,
    )

_LAZY: dict[str, str] = {
    "Exterior": ".station_box",
    "NearField": ".station_box",
    "Pit": ".station_box",
    "SoilLattice": ".station_box",
    "StationSoilBoxResult": ".station_box",
}

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


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
