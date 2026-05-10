"""
Typed ``analysis`` (analysis-type) primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``analysis <Type>`` command. The matching
:class:`apeGmsh.opensees._internal.ns.analysis._AnalysisNS` methods
take no parameters (all three variants are flag-only) and call
``self._bridge._register(Cls())``.

The analysis-type primitive is a singleton in OpenSees (no tag). The
``tag`` parameter to :meth:`_emit` is consumed by the allocator but
not rendered in the emitted command.

OpenSees command shapes::

    analysis Static
    analysis Transient
    analysis VariableTransient

The actual ``analyze N [dt]`` driver lives on the bridge as
``apeSees.analyze(steps=, dt=)`` (see ``architecture/api-design.md``);
the ``analysis`` directive only configures which DirectIntegration /
StaticAnalysis subclass OpenSees instantiates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Analysis, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Static",
    "Transient",
    "VariableTransient",
]


# ---------------------------------------------------------------------------
# Static — pseudo-time / load-control / displacement-control / arc-length
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Static(Analysis):
    """``analysis Static`` — static analysis (pseudo-time)."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.analysis("Static")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Transient — fixed-step time integration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Transient(Analysis):
    """``analysis Transient`` — fixed-step transient analysis."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.analysis("Transient")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# VariableTransient — adaptive-step time integration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class VariableTransient(Analysis):
    """``analysis VariableTransient`` — adaptive-step transient analysis."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.analysis("VariableTransient")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
