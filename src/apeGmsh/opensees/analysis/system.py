"""
Typed ``system`` primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``system <Type>`` command. The matching
:class:`apeGmsh.opensees._internal.ns.analysis._SystemNS` methods take
no parameters (the seven core variants ship as flag-only) and call
``self._bridge._register(Cls())``.

Linear-system solvers are singletons in OpenSees (no tag in the
command). The ``tag`` parameter to :meth:`_emit` is consumed by the
allocator but not rendered.

OpenSees command shapes::

    system BandGeneral
    system BandSPD
    system ProfileSPD
    system UmfPack
    system Mumps
    system SparseGeneral
    system FullGeneral
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import LinearSystem, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "BandGeneral",
    "BandSPD",
    "ProfileSPD",
    "UmfPack",
    "Mumps",
    "SparseGeneral",
    "FullGeneral",
]


# ---------------------------------------------------------------------------
# BandGeneral — banded, general (non-symmetric)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class BandGeneral(LinearSystem):
    """``system BandGeneral`` — banded general (non-symmetric) solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("BandGeneral")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# BandSPD — banded, symmetric positive-definite
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class BandSPD(LinearSystem):
    """``system BandSPD`` — banded symmetric-positive-definite solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("BandSPD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ProfileSPD — variable-bandwidth (skyline) symmetric positive-definite
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ProfileSPD(LinearSystem):
    """``system ProfileSPD`` — skyline symmetric-positive-definite solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("ProfileSPD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# UmfPack — direct sparse LU (UMFPACK)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class UmfPack(LinearSystem):
    """``system UmfPack`` — direct sparse LU via UMFPACK."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("UmfPack")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Mumps — multifrontal massively-parallel sparse direct solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Mumps(LinearSystem):
    """``system Mumps`` — multifrontal massively-parallel sparse solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("Mumps")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# SparseGeneral — generic sparse general (SuperLU-backed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class SparseGeneral(LinearSystem):
    """``system SparseGeneral`` — generic sparse general solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("SparseGeneral")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# FullGeneral — dense, non-symmetric (small-system fallback)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class FullGeneral(LinearSystem):
    """``system FullGeneral`` — dense general solver (small systems only)."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("FullGeneral")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
