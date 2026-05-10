"""
``_UniaxialMaterialNS`` — backs ``ops.uniaxialMaterial.<Type>(...)``.

Phase 0 ships the empty stub; Phase 1A populates it with one typed
method per OpenSees uniaxial material (Steel02, Concrete02, …).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_UniaxialMaterialNS"]


class _UniaxialMaterialNS(_BridgeNamespace):
    """``ops.uniaxialMaterial.<Type>(...)`` — Phase 1A populates."""
