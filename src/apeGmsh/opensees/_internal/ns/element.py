"""
``_ElementNS`` — backs ``ops.element.<Type>(pg=..., ...)``.

Phase 0 ships the empty stub; Phase 2 populates it with one typed
method per OpenSees element (forceBeamColumn, FourNodeTetrahedron, …).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_ElementNS"]


class _ElementNS(_BridgeNamespace):
    """``ops.element.<Type>(pg=..., ...)`` — Phase 2 populates."""
