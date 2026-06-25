"""apeGmsh reinforcement-cage authoring (ADR 0066).

Public L1 surface: the frozen spec objects (re-exported from
:mod:`apeGmsh._kernel.defs.rebar`) and the detailing standards / bar
catalogue. The L2 ``g.rebar`` composite lives in
:mod:`apeGmsh.core.RebarComposite`.
"""

from __future__ import annotations

from .._kernel.defs.rebar import (
    Bar, BarBuilder, BarLayout, Cage, Hook, Path, Stirrup, TieLayout,
    METADATA, Vec3,
)
from .detailing import (
    ACI318, ACI318_seismic, BarCatalog, DetailingError, DetailingStandard, Raw,
)

__all__ = [
    "Bar", "BarBuilder", "BarLayout", "Cage", "Hook", "Path", "Stirrup",
    "TieLayout", "METADATA", "Vec3",
    "ACI318", "ACI318_seismic", "BarCatalog", "DetailingError",
    "DetailingStandard", "Raw",
]
