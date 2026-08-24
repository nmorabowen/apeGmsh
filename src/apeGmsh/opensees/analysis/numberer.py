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
    numberer ParallelPlain
    numberer ParallelRCM
    numberer LadrunoParallelRCM

Note: the parallel variants (:class:`ParallelPlain`, :class:`ParallelRCM`)
are only compiled into OpenSees builds with ``_PARALLEL_INTERPRETERS``
(typically ``OpenSeesMP``). Emitting them and running through a serial
``OpenSees.exe`` produces ``WARNING No Numberer type exists``.
``LadrunoParallelRCM`` additionally requires the Ladruno fork (ADR-74):
a stock ``OpenSeesMP`` errors the same way.
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
    "ParallelPlain",
    "ParallelRCM",
    "LadrunoParallelRCM",
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


# ---------------------------------------------------------------------------
# ParallelPlain — node-add-order numbering on partitioned domains
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ParallelPlain(Numberer):
    """``numberer ParallelPlain`` — parallel plain numbering.

    Only available in OpenSees builds with ``_PARALLEL_INTERPRETERS``
    (e.g. ``OpenSeesMP``). Use with MP-partitioned models; running
    against a serial ``OpenSees.exe`` raises ``WARNING No Numberer
    type exists``.
    """

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.numberer("ParallelPlain")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ParallelRCM — reverse Cuthill-McKee on partitioned domains
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ParallelRCM(Numberer):
    """``numberer ParallelRCM`` — parallel reverse Cuthill-McKee.

    Only available in OpenSees builds with ``_PARALLEL_INTERPRETERS``
    (e.g. ``OpenSeesMP``). Use with MP-partitioned models; running
    against a serial ``OpenSees.exe`` raises ``WARNING No Numberer
    type exists``.
    """

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.numberer("ParallelRCM")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoParallelRCM — O(V) parallel RCM (Ladruno fork, ADR-74)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoParallelRCM(Numberer):
    """``numberer LadrunoParallelRCM`` — parallel RCM, O(V) merge (**Ladruno fork**).

    **Fork-only** (Ladruno OpenSees, ADR-74). Drop-in replacement for
    :class:`ParallelRCM` that replaces the stock ``ParallelNumberer``'s
    O(V²) linear-scan subgraph merge with an owned dense tag→Vertex*
    merge, so first-step numbering setup stays near-linear at large node
    counts instead of degrading quadratically. Only available in OpenSees
    builds with ``_PARALLEL_INTERPRETERS`` (e.g. ``OpenSeesMP``); emission
    works on any build, but running against a stock build or a serial
    ``OpenSees.exe`` raises ``WARNING No Numberer type exists``.
    """

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.numberer("LadrunoParallelRCM")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
