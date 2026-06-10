"""
Per-family bridge namespaces.

Each namespace lives in its own module so that parallel slice agents
can extend them without contending for a single shared file. The
module re-exports every namespace class for the bridge's import side.
"""
from __future__ import annotations

from ._base import _BridgeNamespace
from .analysis import (
    _AlgorithmNS,
    _AnalysisNS,
    _ConstraintsNS,
    _IntegratorNS,
    _NumbererNS,
    _SystemNS,
    _TestNS,
)
from .beam_integration import _BeamIntegrationNS
from .damping import _DampingNS, _StageDampingNS
from .element import _ElementNS
from .geom_transf import _GeomTransfNS
from .nd import _NDMaterialNS
from .pattern import _PatternNS
from .profiler import _ProfilerNS
from .recorder import _RecorderNS
from .section import _SectionNS
from .strategy import _StrategyNS
from .time_series import _TimeSeriesNS
from .uniaxial import _UniaxialMaterialNS


__all__ = [
    "_BridgeNamespace",
    "_UniaxialMaterialNS",
    "_NDMaterialNS",
    "_SectionNS",
    "_GeomTransfNS",
    "_BeamIntegrationNS",
    "_DampingNS",
    "_StageDampingNS",
    "_TimeSeriesNS",
    "_PatternNS",
    "_ElementNS",
    "_ProfilerNS",
    "_RecorderNS",
    "_ConstraintsNS",
    "_NumbererNS",
    "_SystemNS",
    "_TestNS",
    "_AlgorithmNS",
    "_IntegratorNS",
    "_AnalysisNS",
    "_StrategyNS",
]
