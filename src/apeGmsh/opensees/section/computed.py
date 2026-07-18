"""
``ComputedSection`` — declarative binding of a section analyzer to the
bridge (ADR 0078 S5).

Holds a reference to a :class:`~apeGmsh.sections.SectionProperties`
declaration and resolves it **at emit time** through the single shared
lowering (:mod:`apeGmsh.sections._lowering`) into a plain ``section
Elastic`` line — byte-identical to a hand-typed
:class:`~apeGmsh.opensees.section.beam.ElasticSection`.  Because it
subclasses the same :class:`Section` base, it slots into every
consumer (``beamIntegration``, ``Aggregator.base_section``,
``zeroLengthSection``, element ``section=`` fields) with zero consumer
changes.

The analyzer memoizes its solves, so N ``ComputedSection`` references
to one analyzer cost one solve.  Analysis failure at emit
(disconnected domain, missing reference moduli on a composite) fails
loud with the analyzer's ``name`` handle — never a silent fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._internal.types import Primitive, Section
from .beam import ElasticSection

if TYPE_CHECKING:
    from apeGmsh.sections._analysis import SectionProperties

    from ..emitter.base import Emitter


__all__ = ["ComputedSection"]


@dataclass(frozen=True, kw_only=True, slots=True)
class ComputedSection(Section):
    """``section Elastic`` lowered lazily from a section analyzer.

    Parameters
    ----------
    analysis
        The :class:`~apeGmsh.sections.SectionProperties` declaration.
        Held by reference (identity semantics) — its memoized
        ``geometric()`` / ``warping()`` results are computed at emit
        if not already cached.
    E, G
        Reference moduli for the lowering.  Optional for a homogeneous
        analyzer (default from its single material); **required** for
        composite (transformed-section ``EA/E``, ``EI/E``, ``GJ/G``)
        and geometric-only analyzers — omitting them there fails loud
        at emit naming the analyzer.
    ndm
        ``ElasticSection`` form to emit: ``3`` (default) is the 3-D
        form ``E A Iz Iy G J alphaY alphaZ``; ``2`` is the 2-D
        shear-flexible form ``E A Iz G alphaY``.  Match the model's
        ``ops.model(ndm=...)`` envelope.
    """

    analysis: "SectionProperties"
    E: float | None = None
    G: float | None = None
    ndm: Literal[2, 3] = 3

    def __post_init__(self) -> None:
        if self.E is not None and self.E <= 0:
            raise ValueError(
                f"ComputedSection: E must be > 0, got {self.E}."
            )
        if self.G is not None and self.G <= 0:
            raise ValueError(
                f"ComputedSection: G must be > 0, got {self.G}."
            )
        if self.ndm not in (2, 3):
            raise ValueError(
                f"ComputedSection: ndm must be 2 or 3, got {self.ndm}."
            )

    def resolve(self) -> ElasticSection:
        """Run the lowering now and return the plain
        :class:`ElasticSection` this primitive emits as."""
        from apeGmsh.sections._lowering import lower_to_elastic

        params = lower_to_elastic(self.analysis, E=self.E, G=self.G)
        return ElasticSection(**params.section_kwargs(self.ndm))

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        self.resolve()._emit(emitter, tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
