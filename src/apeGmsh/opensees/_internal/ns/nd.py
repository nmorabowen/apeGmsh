"""
``_NDMaterialNS`` — backs ``ops.nDMaterial.<Type>(...)``.

Phase 0 ships the empty stub; Phase 1B populates it with one typed
method per OpenSees nD material (ElasticIsotropic, J2Plasticity, …).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_NDMaterialNS"]


class _NDMaterialNS(_BridgeNamespace):
    """``ops.nDMaterial.<Type>(...)`` — Phase 1B populates."""
