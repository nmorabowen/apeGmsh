"""
``_RecorderNS`` — backs ``ops.recorder.<Type>(...)``.

Phase 0 ships the empty stub; Phase 3B populates it with one typed
method per recorder kind (Node, Element, MPCO).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_RecorderNS"]


class _RecorderNS(_BridgeNamespace):
    """``ops.recorder.<Type>(...)`` — Phase 3B populates."""
