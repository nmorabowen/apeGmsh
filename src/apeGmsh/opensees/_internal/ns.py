"""
Namespace base class plus the per-family namespace stubs.

The bridge exposes a method-namespace per OpenSees command that has
type variants: ``ops.uniaxialMaterial.Steel02(...)``,
``ops.element.forceBeamColumn(pg=..., ...)``, etc. This module ships
the **stubs** for those namespaces — they hold a back-reference to
the bridge for registration but carry no concrete type methods yet.

Phase 1+ agents add type methods (``Steel02``, ``Concrete02``, …) to
each namespace; the namespace classes themselves stay here so the
bridge can instantiate them in Phase 0.

The base class :class:`_BridgeNamespace` is the only one that holds
state; the per-family classes are empty subclasses used purely for
type identification and method-dispatch namespacing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..apesees import apeSees


__all__ = [
    "_BridgeNamespace",
    "_UniaxialMaterialNS",
    "_NDMaterialNS",
    "_SectionNS",
    "_GeomTransfNS",
    "_TimeSeriesNS",
    "_PatternNS",
    "_ElementNS",
    "_RecorderNS",
    "_ConstraintsNS",
    "_NumbererNS",
    "_SystemNS",
    "_TestNS",
    "_AlgorithmNS",
    "_IntegratorNS",
    "_AnalysisNS",
]


class _BridgeNamespace:
    """Base for bridge namespaces.

    Each namespace holds a back-reference to its owning bridge so
    that namespace methods can call ``self._bridge._register(...)``
    when they construct a typed primitive.
    """

    __slots__ = ("_bridge",)

    def __init__(self, bridge: "apeSees") -> None:
        self._bridge = bridge


# ---------------------------------------------------------------------------
# Per-family namespace stubs.
#
# Phase 0: empty bodies. Phase 1+ agents add typed methods to each
# class (one method per OpenSees type, e.g. _UniaxialMaterialNS.Steel02).
# ---------------------------------------------------------------------------

class _UniaxialMaterialNS(_BridgeNamespace):
    """``ops.uniaxialMaterial.<Type>(...)`` — Phase 1A populates."""


class _NDMaterialNS(_BridgeNamespace):
    """``ops.nDMaterial.<Type>(...)`` — Phase 1B populates."""


class _SectionNS(_BridgeNamespace):
    """``ops.section.<Type>(...)`` — Phase 1C populates."""


class _GeomTransfNS(_BridgeNamespace):
    """``ops.geomTransf.<Type>(...)`` — Phase 1D populates."""


class _TimeSeriesNS(_BridgeNamespace):
    """``ops.timeSeries.<Type>(...)`` — Phase 1D-extra populates."""


class _PatternNS(_BridgeNamespace):
    """``ops.pattern.<Type>(...)`` — Phase 3A populates."""


class _ElementNS(_BridgeNamespace):
    """``ops.element.<Type>(pg=..., ...)`` — Phase 2 populates."""


class _RecorderNS(_BridgeNamespace):
    """``ops.recorder.<Type>(...)`` — Phase 3B populates."""


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
