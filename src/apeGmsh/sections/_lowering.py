"""The single authoring-axes → OpenSees ``ElasticSection`` lowering
(ADR 0078 S5).

Both bridge paths — the declarative ``ops.section.ComputedSection`` and
the eager ``SectionProperties.to_elastic_section()`` — call
:func:`lower_to_elastic`.  No other code performs this mapping.

Axis contract (ADR 0078 "Axis and load conventions"): the section is
authored in the gmsh (x, y) plane; OpenSees frame sections live in
local (y, z) with local-y "up".  The identification is *authoring x ≡
local z, authoring y ≡ local y*, hence:

=========================  ==========================
analyzer (authoring axes)  OpenSees ``ElasticSection``
=========================  ==========================
``Ixx_c``                  ``Iz``
``Iyy_c``                  ``Iy``
``J``                      ``J``
``As_y / A``               ``alphaY``
``As_x / A``               ``alphaZ``
=========================  ==========================

Reference-moduli rules:

* **geometric-only** analyzer (no ``materials=``): the classic
  geometric numbers are emitted verbatim and the deck moduli ``E`` /
  ``G`` are **required** — there is no material to default them from.
* **homogeneous** (one modulus pair): ``E`` / ``G`` default from the
  single material; the emitted section constants are the true geometry.
* **composite**: explicit reference ``E`` and ``G`` are **required**
  (fail-loud naming the section handle) and the emitted constants are
  the transformed-section values ``EA/E``, ``EI/E``, ``GJ/G`` — the
  deck reproduces the analyzer's rigidities exactly.

Explicit ``E`` / ``G`` on a homogeneous analyzer follow the same
transformed-section semantics: the rigidities are preserved and the
section constants are re-referenced to the given moduli.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._errors import CompositeSectionError

if TYPE_CHECKING:  # pragma: no cover
    from ._analysis import SectionProperties


@dataclass(frozen=True, slots=True)
class ElasticLoweringParams:
    """The full ``section Elastic`` parameter set in OpenSees axes."""

    E: float
    A: float
    Iz: float
    Iy: float
    G: float
    J: float
    alphaY: float
    alphaZ: float

    def section_kwargs(self, ndm: int) -> dict[str, float]:
        """Constructor kwargs for :class:`ElasticSection` at ``ndm``.

        ``ndm=3`` selects the 3-D form (``E A Iz Iy G J alphaY
        alphaZ``); ``ndm=2`` the 2-D shear-flexible form (``E A Iz G
        alphaY``).  Any other value raises ``ValueError``.
        """
        if ndm == 3:
            return {
                "E": self.E, "A": self.A, "Iz": self.Iz, "Iy": self.Iy,
                "G": self.G, "J": self.J,
                "alphaY": self.alphaY, "alphaZ": self.alphaZ,
            }
        if ndm == 2:
            return {
                "E": self.E, "A": self.A, "Iz": self.Iz,
                "G": self.G, "alphaY": self.alphaY,
            }
        raise ValueError(f"ndm must be 2 or 3, got {ndm}.")


def lower_to_elastic(
    analysis: "SectionProperties",
    *,
    E: float | None = None,
    G: float | None = None,
) -> ElasticLoweringParams:
    """Lower one analyzer into ``section Elastic`` parameters.

    Triggers the (memoized) ``geometric()`` + ``warping()`` analyses;
    warping's own gates apply (disconnected sections under the default
    policy fail loud here).

    Raises
    ------
    ValueError
        Geometric-only analyzer without explicit ``E`` and ``G``.
    CompositeSectionError
        Composite analyzer without explicit reference ``E`` and ``G``.
    """
    handle = analysis.name or "section"
    geo = analysis.geometric()
    warp = analysis.warping()

    if analysis.geometric_only:
        # Classic geometric numbers (unit-modulus placeholders divide
        # out via the unprefixed accessors); deck moduli must be given.
        missing = [k for k, v in (("E", E), ("G", G)) if v is None]
        if missing:
            raise ValueError(
                f"{handle}: geometric-only analyzer has no material to "
                f"default the deck moduli from — pass "
                f"{' and '.join(f'{m}=' for m in missing)} to "
                f"ComputedSection / to_elastic_section."
            )
        assert E is not None and G is not None
        area = geo.area
        return ElasticLoweringParams(
            E=E, A=area, Iz=geo.Ixx_c, Iy=geo.Iyy_c,
            G=G, J=warp.J,
            alphaY=warp.As_y / area, alphaZ=warp.As_x / area,
        )

    composite = geo.e_ref is None or warp.g_ref is None
    if composite:
        missing = [k for k, v in (("E", E), ("G", G)) if v is None]
        if missing:
            raise CompositeSectionError(
                f"{handle}: composite section — the ElasticSection "
                f"lowering needs explicit reference moduli "
                f"({' and '.join(f'{m}=' for m in missing)} on "
                f"ComputedSection / to_elastic_section) to produce the "
                f"transformed-section constants EA/E, EI/E, GJ/G. A "
                f"reference modulus is never picked silently."
            )
    e_out = E if E is not None else geo.e_ref
    g_out = G if G is not None else warp.g_ref
    assert e_out is not None and g_out is not None

    area = geo.EA / e_out
    return ElasticLoweringParams(
        E=e_out,
        A=area,
        Iz=geo.EIxx_c / e_out,
        Iy=geo.EIyy_c / e_out,
        G=g_out,
        J=warp.GJ / g_out,
        alphaY=(warp.GAs_y / g_out) / area,
        alphaZ=(warp.GAs_x / g_out) / area,
    )


__all__ = ["ElasticLoweringParams", "lower_to_elastic"]
