"""
``_GeomTransfNS`` — backs ``ops.geomTransf.<Type>(...)``.

Phase 0 ships the empty stub; Phase 1D populates it with one typed
method per OpenSees geomTransf (Linear, PDelta, Corotational).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_GeomTransfNS"]


class _GeomTransfNS(_BridgeNamespace):
    """``ops.geomTransf.<Type>(...)`` — Phase 1D populates."""
