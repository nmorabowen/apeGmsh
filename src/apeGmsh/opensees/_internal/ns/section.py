"""
``_SectionNS`` — backs ``ops.section.<Type>(...)``.

Phase 0 ships the empty stub; Phase 1C populates it with one typed
method per OpenSees section (Fiber, ElasticMembranePlateSection, …).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_SectionNS"]


class _SectionNS(_BridgeNamespace):
    """``ops.section.<Type>(...)`` — Phase 1C populates."""
