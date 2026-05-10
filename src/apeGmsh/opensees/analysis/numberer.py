"""
Typed ``numberer`` primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``numberer <Type>`` command. The matching
:class:`apeGmsh.opensees._internal.ns.analysis._NumbererNS` methods
take no parameters (these numberers are flag-only) and call
``self._bridge._register(Cls())``.

Numberers are singletons in OpenSees — no tag in the command syntax.
The ``tag`` parameter to :meth:`_emit` is consumed by the allocator
but not rendered.

OpenSees command shapes::

    numberer Plain
    numberer RCM
    numberer AMD

Note: ``numberer ParallelPlain`` (and other parallel variants) are
deferred — they apply only to OpenSeesSP/MP builds, are rarely used,
and add complexity without payoff for the v1 bridge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Numberer, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Plain",
    "RCM",
    "AMD",
]


# ---------------------------------------------------------------------------
# Plain — sequential numbering in node-add order
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Plain(Numberer):
    """``numberer Plain`` — number DOFs in node-add order."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag  # singletons; no tag in the OpenSees command
        emitter.numberer("Plain")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# RCM — reverse Cuthill-McKee bandwidth-reducing permutation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class RCM(Numberer):
    """``numberer RCM`` — reverse Cuthill-McKee bandwidth reduction."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.numberer("RCM")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# AMD — approximate minimum degree fill-reducing permutation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class AMD(Numberer):
    """``numberer AMD`` — approximate minimum degree fill reduction."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.numberer("AMD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
