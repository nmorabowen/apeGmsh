"""
``apeSees`` — the bridge class.

Takes a :class:`~apeGmsh.mesh.FEMData` snapshot at construction. Never
imports gmsh. Holds the user's typed primitive declarations and
delegates emission to a separate :class:`BuiltModel` produced by
:meth:`apeSees.build`.

Phase 0 ships:
  * Construction with a FEM snapshot.
  * Namespace stubs (``ops.uniaxialMaterial``, ``ops.element``, …).
  * Tag allocator.
  * ``register()`` / ``_register()`` — adds a primitive, allocates
    its tag.
  * Stub flat methods (``model``, ``fix``, ``mass``, ``analyze``,
    ``tcl``, ``py``, ``run``, ``h5``).
  * ``build()`` returns a minimal :class:`BuiltModel`.

Concrete primitive type methods on the namespaces, real emission, and
real analysis dispatch land in Phase 1+.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, TypeVar

from ._internal.ns import (
    _AlgorithmNS,
    _AnalysisNS,
    _ConstraintsNS,
    _ElementNS,
    _GeomTransfNS,
    _IntegratorNS,
    _NDMaterialNS,
    _NumbererNS,
    _PatternNS,
    _RecorderNS,
    _SectionNS,
    _SystemNS,
    _TestNS,
    _TimeSeriesNS,
    _UniaxialMaterialNS,
)
from ._internal.tag_allocator import TagAllocator
from ._internal.types import (
    Analysis,
    ConstraintHandler,
    ConvergenceTest,
    Element,
    GeomTransf,
    Integrator,
    LinearSystem,
    NDMaterial,
    Numberer,
    Pattern,
    Primitive,
    Recorder,
    Section,
    SolutionAlgorithm,
    TimeSeries,
    UniaxialMaterial,
)
from .emitter.base import Emitter

if TYPE_CHECKING:
    # FEMData is the only mesh symbol the bridge depends on (P3, P9).
    # Imported under TYPE_CHECKING so that constructing apeSees does
    # not transitively import gmsh during static analysis.
    from apeGmsh.mesh import FEMData


__all__ = ["apeSees", "BuiltModel"]


# Bound to Primitive so namespace methods preserve the concrete type:
#   def Steel02(self, ...) -> Steel02:
#       return self._bridge._register(Steel02(...))
_P = TypeVar("_P", bound=Primitive)


# ---------------------------------------------------------------------------
# Tag-allocation kind dispatch
# ---------------------------------------------------------------------------
#
# The TagAllocator is per-kind. Every primitive lands in exactly one
# kind, determined by the family base class it inherits from. This map
# is the bridge's authoritative source of "what kind is this primitive."
# Pattern matching against MRO covers the case where Phase 1+ slices
# add new family bases (currently they shouldn't — the foundation is
# read-only after Phase 0 — but the pattern is robust either way).
# ---------------------------------------------------------------------------

_KIND_BY_FAMILY: tuple[tuple[type[Primitive], str], ...] = (
    (UniaxialMaterial, "uniaxialMaterial"),
    (NDMaterial,       "nDMaterial"),
    (Section,          "section"),
    (GeomTransf,       "geomTransf"),
    (TimeSeries,       "timeSeries"),
    (Pattern,          "pattern"),
    (Element,          "element"),
    (Recorder,         "recorder"),
    (ConstraintHandler, "constraints"),
    (Numberer,         "numberer"),
    (LinearSystem,     "system"),
    (ConvergenceTest,  "test"),
    (SolutionAlgorithm, "algorithm"),
    (Integrator,       "integrator"),
    (Analysis,         "analysis"),
)


def _kind_of(prim: Primitive) -> str:
    """Return the tag-allocator kind string for ``prim``."""
    for base, kind in _KIND_BY_FAMILY:
        if isinstance(prim, base):
            return kind
    raise TypeError(
        f"Primitive {type(prim).__name__} does not inherit from any "
        f"recognized family base (UniaxialMaterial, Section, ...)."
    )


# ---------------------------------------------------------------------------
# BuiltModel — the immutable read-only artifact emitters consume
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuiltModel:
    """Immutable snapshot of declared primitives + tag assignments.

    Phase 0 keeps this minimal. Later phases extend with dependency-
    sorted ordering, named-group lookups, and the model-level
    ``model(ndm, ndf)`` / ``fix`` / ``mass`` / analysis-chain records.
    """

    primitives: tuple[Primitive, ...]
    tag_for: dict[int, int]
    ndm: int
    ndf: int

    def emit(self, emitter: Emitter) -> None:
        """Drive ``emitter`` over the model.

        Phase 0 emits only the model directive plus each primitive's
        ``_emit`` (in registration order — Phase 1+ replaces this with
        dependency-sorted ordering). The flat methods (``fix``,
        ``mass``, the analysis chain, ``analyze``) are not yet wired.
        """
        emitter.model(ndm=self.ndm, ndf=self.ndf)
        for prim in self.primitives:
            tag = self.tag_for[id(prim)]
            prim._emit(emitter, tag)


# ---------------------------------------------------------------------------
# apeSees — the bridge
# ---------------------------------------------------------------------------

class apeSees:
    """The OpenSees bridge.

    Construct with a :class:`~apeGmsh.mesh.FEMData` snapshot:

    .. code-block:: python

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
        ...

    The bridge holds **declared** state. ``apeSees.build()`` returns a
    :class:`BuiltModel` (immutable) that emitters consume. See ADR
    0001 for the FEM-as-snapshot contract and ADR 0006 for the class
    name.
    """

    def __init__(self, fem: "FEMData") -> None:
        self._fem: "FEMData" = fem
        self._primitives: list[Primitive] = []
        self._tags = TagAllocator()
        self._ndm: int | None = None
        self._ndf: int | None = None

        # Namespaces — Phase 0 ships stub classes; Phase 1+ fills in
        # concrete type methods on each.
        self.uniaxialMaterial = _UniaxialMaterialNS(self)
        self.nDMaterial       = _NDMaterialNS(self)
        self.section          = _SectionNS(self)
        self.geomTransf       = _GeomTransfNS(self)
        self.timeSeries       = _TimeSeriesNS(self)
        self.pattern          = _PatternNS(self)
        self.element          = _ElementNS(self)
        self.recorder         = _RecorderNS(self)
        self.constraints      = _ConstraintsNS(self)
        self.numberer         = _NumbererNS(self)
        self.system           = _SystemNS(self)
        self.test             = _TestNS(self)
        self.algorithm        = _AlgorithmNS(self)
        self.integrator       = _IntegratorNS(self)
        self.analysis         = _AnalysisNS(self)

    # -- Read-only access to the FEM snapshot ----------------------------
    @property
    def fem(self) -> "FEMData":
        return self._fem

    # -- Flat methods (Phase 0 stubs) -----------------------------------
    #
    # These accept their typed signatures but defer concrete behavior
    # to Phase 3+ slices. They exist now so:
    #   1. The user-facing surface is locked at the type-system level.
    #   2. Phase 1+ tests can construct an ``apeSees`` and call
    #      ``ops.model(...)`` without ImportError.
    # -------------------------------------------------------------------

    def model(self, *, ndm: int, ndf: int) -> None:
        """Set the model dimensionality (``ndm``) and DOFs/node (``ndf``)."""
        self._ndm = ndm
        self._ndf = ndf

    def fix(
        self,
        *,
        pg: str | None = None,
        nodes: Iterable[int] | None = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Apply homogeneous SP constraints (Phase 3 fills emit logic)."""
        raise NotImplementedError(
            "ops.fix() is declared in Phase 0 but emit logic lands in "
            "Phase 3A (patterns-and-constraints slice)."
        )

    def mass(
        self,
        *,
        pg: str | None = None,
        nodes: Iterable[int] | None = None,
        values: tuple[float, ...],
    ) -> None:
        """Attach lumped nodal mass (Phase 3 fills emit logic)."""
        raise NotImplementedError(
            "ops.mass() is declared in Phase 0 but emit logic lands "
            "in Phase 3A."
        )

    def analyze(self, *, steps: int, dt: float | None = None) -> int:
        """Run the analysis chain for ``steps`` steps (Phase 4 fills)."""
        raise NotImplementedError(
            "ops.analyze() is declared in Phase 0 but execution lands "
            "in Phase 4 (concrete emitters)."
        )

    def tcl(
        self,
        path: str,
        *,
        run: bool = False,
        bin: str | None = None,
    ) -> None:
        """Emit a Tcl deck (Phase 4A fills)."""
        raise NotImplementedError(
            "ops.tcl() is declared in Phase 0 but TclEmitter lands "
            "in Phase 4A."
        )

    def py(self, path: str, *, run: bool = False) -> None:
        """Emit an openseespy Python deck (Phase 4B fills)."""
        raise NotImplementedError(
            "ops.py() is declared in Phase 0 but PyEmitter lands "
            "in Phase 4B."
        )

    def run(self, *, wipe: bool = True) -> None:
        """Drive an in-process LiveOpsEmitter (Phase 4C fills)."""
        raise NotImplementedError(
            "ops.run() is declared in Phase 0 but LiveOpsEmitter "
            "lands in Phase 4C."
        )

    def h5(self, path: str) -> None:
        """Emit a model-definition HDF5 archive (Phase 6 fills)."""
        raise NotImplementedError(
            "ops.h5() is declared in Phase 0 but H5Emitter lands "
            "in Phase 6."
        )

    # -- Registration -----------------------------------------------------

    def _register(self, prim: _P) -> _P:
        """Add ``prim`` to the bridge, allocate its tag, return it.

        Internal — namespace methods call this to register the
        primitives they construct. Public callers should use
        :meth:`register` (P11 standalone-then-register flow).

        Returns the same instance (typed-narrowed via ``TypeVar``) so
        callsites read as ``Steel02 = ops._register(Steel02(...))``.
        """
        kind = _kind_of(prim)
        self._tags.allocate_for(prim, kind)
        self._primitives.append(prim)
        return prim

    def register(self, prim: _P) -> _P:
        """Register a standalone primitive with the bridge (P11).

        This is the supported path for primitives the user constructed
        outside the namespace API (e.g. for material studies). The
        primitive's tag is assigned at this call, identical to what
        the namespace methods do internally.
        """
        return self._register(prim)

    def tag_for(self, prim: Primitive) -> int | None:
        """Return ``prim``'s allocated tag, or ``None`` if unregistered."""
        return self._tags.tag_for(prim)

    # -- Build -----------------------------------------------------------

    def build(self) -> BuiltModel:
        """Freeze the declarations into a :class:`BuiltModel`.

        The returned object is immutable and is the only thing
        emitters see. Calling ``build()`` does not modify the bridge —
        further declarations after a build produce a new build on the
        next call.
        """
        if self._ndm is None or self._ndf is None:
            raise RuntimeError(
                "apeSees.model(ndm=..., ndf=...) must be called before "
                "build()."
            )

        tag_for: dict[int, int] = {
            id(p): self._tags.tag_for(p) or 0 for p in self._primitives
        }
        return BuiltModel(
            primitives=tuple(self._primitives),
            tag_for=tag_for,
            ndm=self._ndm,
            ndf=self._ndf,
        )
