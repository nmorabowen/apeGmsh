"""
Base classes for typed primitives.

``Primitive`` is the abstract root. Concrete leaf classes (``Steel02``,
``Fiber``, ``forceBeamColumn``, …) inherit from one of the family
bases (``UniaxialMaterial``, ``Section``, ``Element``, …) and become
``@dataclass(frozen=True, kw_only=True, slots=True)``.

Combining ``abc.ABC`` with slotted frozen dataclasses requires care:
slots conflict with ``__dict__``, and frozen dataclasses inherit their
slot layout. Rather than fighting that, we keep the ABC parents
**non-dataclass** plain classes that only declare the abstract method
surface. Concrete subclasses become dataclasses one level down. This
is the pattern used elsewhere in the apeGmsh codebase.

The Protocol type is imported behind ``TYPE_CHECKING`` to avoid a
runtime cycle (``emitter`` package re-exports nothing from here, so
the static reference is sufficient).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Primitive",
    "UniaxialMaterial",
    "NDMaterial",
    "Section",
    "GeomTransf",
    "Element",
    "TimeSeries",
    "Pattern",
    "Recorder",
    "ConstraintHandler",
    "Numberer",
    "LinearSystem",
    "ConvergenceTest",
    "SolutionAlgorithm",
    "Integrator",
    "Analysis",
]


class Primitive(ABC):
    """Abstract root of the typed-primitive hierarchy.

    Every concrete primitive class:
      * inherits from a family base (``UniaxialMaterial``, ``Section``,
        …) below this class.
      * is decorated ``@dataclass(frozen=True, kw_only=True,
        slots=True)``.
      * implements :meth:`_emit` to push its OpenSees command into
        an :class:`~apeGmsh.opensees.emitter.base.Emitter`.
      * implements :meth:`dependencies` to expose the other primitives
        it composes (sections return their materials, elements return
        their section + transform, leaves return ``()``).
    """

    @abstractmethod
    def _emit(self, emitter: "Emitter", tag: int) -> None:
        """Push this primitive's OpenSees command(s) into ``emitter``.

        ``tag`` is allocated by the bridge's :class:`TagAllocator` at
        build time. Primitives never know their tag at construction.
        """

    @abstractmethod
    def dependencies(self) -> tuple["Primitive", ...]:
        """Other primitives this one composes.

        Leaves (``Steel02``, ``Linear`` time series, …) return ``()``.
        Composers (``Fiber`` section, elements) return their direct
        children; transitive resolution is the bridge's job.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


# ---------------------------------------------------------------------------
# Material family
# ---------------------------------------------------------------------------

class UniaxialMaterial(Primitive):
    """Abstract base for ``uniaxialMaterial <Type>`` primitives."""


class NDMaterial(Primitive):
    """Abstract base for ``nDMaterial <Type>`` primitives."""


# ---------------------------------------------------------------------------
# Section, GeomTransf
# ---------------------------------------------------------------------------

class Section(Primitive):
    """Abstract base for ``section <Type>`` primitives."""


class GeomTransf(Primitive):
    """Abstract base for ``geomTransf <Type>`` primitives.

    Concrete transforms (Linear, PDelta, Corotational) live in
    ``opensees.transform``; coordinate-system helpers (Cartesian,
    Cylindrical, Spherical) are re-exported from
    :mod:`apeGmsh.solvers._opensees_csys` per ADR 0010.
    """


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------

class Element(Primitive):
    """Abstract base for element specifications.

    The typed dataclass IS the spec. The user-facing object returned
    by ``ops.element.X(pg=...)`` is an ``ElementGroup`` (apeGmsh-native
    aggregate) that wraps one element spec applied to a physical
    group; that aggregation lands in Phase 5. Phase 0 exposes only
    the abstract base so element primitives in Phase 2 have something
    to inherit from.
    """


# ---------------------------------------------------------------------------
# TimeSeries, Pattern
# ---------------------------------------------------------------------------

class TimeSeries(Primitive):
    """Abstract base for ``timeSeries <Type>`` primitives."""


class Pattern(Primitive):
    """Abstract base for ``pattern <Type>`` primitives.

    Concrete patterns are also context managers: ``__enter__`` returns
    ``self``, ``__exit__`` finalizes (the bridge typically writes
    ``pattern_close`` at exit). Phase 3 implementations supply the
    full ``__enter__`` / ``__exit__`` protocol; Phase 0 only declares
    the abstract base so primitive-level tests can reference it.
    """


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder(Primitive):
    """Abstract base for ``recorder <Type>`` primitives."""


# ---------------------------------------------------------------------------
# Analysis chain — one base per OpenSees analysis-component family
# ---------------------------------------------------------------------------

class ConstraintHandler(Primitive):
    """Abstract base for ``constraints <Type>`` primitives."""


class Numberer(Primitive):
    """Abstract base for ``numberer <Type>`` primitives."""


class LinearSystem(Primitive):
    """Abstract base for ``system <Type>`` primitives."""


class ConvergenceTest(Primitive):
    """Abstract base for ``test <Type>`` primitives."""


class SolutionAlgorithm(Primitive):
    """Abstract base for ``algorithm <Type>`` primitives."""


class Integrator(Primitive):
    """Abstract base for ``integrator <Type>`` primitives."""


class Analysis(Primitive):
    """Abstract base for ``analysis <Type>`` primitives."""
