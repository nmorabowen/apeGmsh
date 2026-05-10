"""
``_PatternNS`` — backs ``ops.pattern.<Type>(...)``.

Phase 0 ships the empty stub; Phase 3A populates it with the pattern
context-manager-producing methods (Plain, UniformExcitation, …).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_PatternNS"]


class _PatternNS(_BridgeNamespace):
    """``ops.pattern.<Type>(...)`` — Phase 3A populates."""
