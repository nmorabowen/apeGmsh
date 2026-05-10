"""
``_TimeSeriesNS`` — backs ``ops.timeSeries.<Type>(...)``.

Phase 0 ships the empty stub; Phase 1D-extra populates it with one
typed method per OpenSees TimeSeries (Linear, Path, ASCE41Protocol, …).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_TimeSeriesNS"]


class _TimeSeriesNS(_BridgeNamespace):
    """``ops.timeSeries.<Type>(...)`` — Phase 1D-extra populates."""
