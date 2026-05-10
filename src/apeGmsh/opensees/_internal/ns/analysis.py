"""
Analysis-chain namespaces — backed by Phase 3C.

Holds the seven analysis-component family namespaces in one module
because Phase 3C ships them as a single slice (one agent owns all
seven; they share patterns and the volume per file is small).
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = [
    "_ConstraintsNS",
    "_NumbererNS",
    "_SystemNS",
    "_TestNS",
    "_AlgorithmNS",
    "_IntegratorNS",
    "_AnalysisNS",
]


class _ConstraintsNS(_BridgeNamespace):
    """``ops.constraints.<Type>(...)`` — Phase 3C populates."""


class _NumbererNS(_BridgeNamespace):
    """``ops.numberer.<Type>()`` — Phase 3C populates."""


class _SystemNS(_BridgeNamespace):
    """``ops.system.<Type>(...)`` — Phase 3C populates."""


class _TestNS(_BridgeNamespace):
    """``ops.test.<Type>(...)`` — Phase 3C populates."""


class _AlgorithmNS(_BridgeNamespace):
    """``ops.algorithm.<Type>(...)`` — Phase 3C populates."""


class _IntegratorNS(_BridgeNamespace):
    """``ops.integrator.<Type>(...)`` — Phase 3C populates."""


class _AnalysisNS(_BridgeNamespace):
    """``ops.analysis.<Type>()`` — Phase 3C populates."""
