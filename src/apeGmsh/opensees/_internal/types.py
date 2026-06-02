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
    from apeGmsh.mesh.FEMData import FEMData

    from ..emitter.base import Emitter
    from .tag_allocator import TagAllocator


__all__ = [
    "Primitive",
    "UniaxialMaterial",
    "NDMaterial",
    "Section",
    "GeomTransf",
    "BeamIntegration",
    "Element",
    "TimeSeries",
    "Pattern",
    "Recorder",
    "Damping",
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
    ``opensees.transform``; orientation helpers (Cartesian,
    Cylindrical, Spherical) are exported from
    :mod:`apeGmsh.opensees._orientation` per ADR 0010.
    """


# ---------------------------------------------------------------------------
# BeamIntegration
# ---------------------------------------------------------------------------

class BeamIntegration(Primitive):
    """Abstract base for ``beamIntegration <Type>`` primitives.

    BeamIntegration rules sit between sections and beam-column elements.
    A rule (Lobatto, HingeRadau, etc.) composes one or more sections
    and tells the force-/disp-based beam-column element how many
    integration points to place and where. The element references the
    rule by tag rather than carrying a ``section`` + ``n_ip`` pair
    directly.

    This split mirrors modern OpenSees (and is what openseespy requires
    for ``forceBeamColumn`` / ``dispBeamColumn`` to parse). It also
    makes concentrated-plasticity rules (Hinge*) and per-IP
    heterogeneous sections (UserDefined / FixedLocation) first-class.
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
    """Abstract base for ``recorder <Type>`` primitives.

    Recorders may carry build-time selectors (``pg=``, ``nodes_pg=``,
    ``elements_pg=``) that must be resolved against the FEM before
    ``_emit`` runs.  :meth:`materialize` is the hook each concrete
    recorder overrides to do that resolution; the default returns
    ``self`` unchanged (no resolution needed).  Recorders that
    require auxiliary OpenSees declarations (e.g. MPCO's
    ``region`` line) emit them on the supplied ``emitter`` from
    inside ``materialize`` and return a clone of themselves with
    the build-time selectors cleared and any tag fields populated.
    """

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
    ) -> "Recorder":
        """Resolve build-time selectors against the FEM.

        Default no-op for recorders that carry no ``pg=``-style
        selectors (e.g. a ``Node`` recorder built with explicit
        ``nodes=``).  Concrete subclasses override to fan out
        ``pg``/``nodes_pg``/``elements_pg`` into explicit id lists
        and, when applicable, emit auxiliary declarations
        (MPCO emits a ``region`` line) on ``emitter``.

        The returned recorder must satisfy its own ``_emit`` contract
        directly — i.e. with no remaining ``pg=`` form set.  ``tags``
        is the bridge's :class:`TagAllocator`; recorders that need
        a fresh tag (e.g. MPCO's region tag) allocate from it.

        ``fem_eid_to_ops_tag`` is the bridge-built ``{fem_eid: ops_tag}``
        map for element fan-out — needed by element-targeting recorders
        (``Element``, ``MPCO`` with element filters) to translate
        FEM-side element ids resolved from ``pg=`` into the actual
        OpenSees element tags emitted by the element fan-out (which
        differ whenever an element primitive consumed an allocator
        slot in ``_register``).  ``None`` means the bridge did not
        supply it (legacy direct callers); element-targeting recorders
        with ``pg=`` then fall back to the FEM eids verbatim.
        """
        return self


# ---------------------------------------------------------------------------
# Damping objects (ADR 0053)
# ---------------------------------------------------------------------------

class Damping(Primitive):
    """Abstract base for ``damping <Type>`` objects (ADR 0053).

    A tagged frequency-band viscous damping object (``Uniform`` /
    ``SecStif`` / ``URD`` / ``URDbeta``).  Inert on its own — it does
    nothing until attached to elements via a ``region $tag -damp $tag``
    line (the bridge emits that attach from ``ops.damping.<type>(on=...)``).
    Emits as a *definition* primitive in the pre-element group, like a
    material or section, so its tag exists before any region references it.
    """


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
