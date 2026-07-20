"""
``ComputedSection`` — declarative binding of a section analyzer to the
bridge (ADR 0078 S5 + Amendment A2).

Holds a reference to a :class:`~apeGmsh.sections.SectionProperties`
declaration and resolves it **at emit time** through the shared
lowerings (:mod:`apeGmsh.sections._lowering`):

* ``kind="elastic"`` (default) — a plain ``section Elastic`` line,
  byte-identical to a hand-typed
  :class:`~apeGmsh.opensees.section.beam.ElasticSection`.
* ``kind="fiber"`` (Amendment A2) — an auto-generated ``section
  Fiber`` block: one fiber per Gauss point of the analyzer mesh, with
  **user-supplied** per-region uniaxial materials
  (``fibers={pg: UniaxialMaterial}`` — the material law is a modeling
  decision, never inferred) and ``-GJ`` defaulting from the analyzer's
  ``warp.GJ``.

Because it subclasses the same :class:`Section` base as
``ElasticSection`` / ``Fiber``, it slots into every consumer
(``beamIntegration``, ``Aggregator.base_section``,
``zeroLengthSection``, element ``section=`` fields) with zero consumer
changes.

The analyzer memoizes its solves, so N ``ComputedSection`` references
to one analyzer cost one solve.  Analysis failure at emit
(disconnected domain, missing reference moduli on a composite) fails
loud with the analyzer's ``name`` handle — never a silent fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping

from .._internal.types import Primitive, Section, UniaxialMaterial
from .beam import ElasticSection
from .fiber import Fiber, FiberPoint

if TYPE_CHECKING:
    from apeGmsh.sections._analysis import SectionProperties

    from ..emitter.base import Emitter


__all__ = ["Bar", "ComputedSection"]


@dataclass(frozen=True, kw_only=True, slots=True)
class Bar:
    """One discrete reinforcement bar overlaid on an analyzer section
    (ADR 0080 B3).

    A value object like :class:`~.fiber.FiberPoint`, but its
    coordinates are the analyzer's **authoring (x, y) axes** — the
    ``kind="fiber"`` lowering maps them through the same
    *authoring x ≡ local z, authoring y ≡ local y* identification as
    the Gauss fibers, about the elastic centroid (gate G-E covers the
    signed mapping). Concrete area is NOT deducted at bar locations
    (standard fiber-section practice; the ~ρ·(1−Ec/Es) error is
    documented, not knobbed).
    """

    material: UniaxialMaterial
    x: float
    y: float
    area: float

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError(
                f"Bar: area must be > 0, got {self.area}."
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class ComputedSection(Section):
    """``section Elastic`` / ``section Fiber`` lowered lazily from a
    section analyzer.

    Parameters
    ----------
    analysis
        The :class:`~apeGmsh.sections.SectionProperties` declaration.
        Held by reference (identity semantics) — its memoized analyses
        are computed at emit if not already cached.
    kind
        The lowering: ``"elastic"`` (default) emits ``section
        Elastic``; ``"fiber"`` emits a ``section Fiber`` block with
        one fiber per Gauss point of the analyzer mesh.  Argument
        families are validated per kind at construction.
    E, G
        (``kind="elastic"`` only) Reference moduli for the lowering.
        Optional for a homogeneous analyzer (default from its single
        material); **required** for composite (transformed-section
        ``EA/E``, ``EI/E``, ``GJ/G``) and geometric-only analyzers —
        omitting them there fails loud at emit naming the analyzer.
    ndm
        (``kind="elastic"`` only) ``ElasticSection`` form to emit:
        ``3`` (default) is the 3-D form ``E A Iz Iy G J alphaY
        alphaZ``; ``2`` is the 2-D shear-flexible form ``E A Iz G
        alphaY``.  Match the model's ``ops.model(ndm=...)`` envelope.
    fibers
        (``kind="fiber"`` only, required) Analyzer material-PG name →
        :class:`UniaxialMaterial`.  Must **exactly cover** the
        analyzer's material regions — missing or unknown names raise
        at construction.  Geometric-only analyzers are rejected (no
        PGs to key by).  The materials are ordinary dependencies:
        construct them through the bridge
        (``ops.uniaxialMaterial.*``) so they are registered (P11).
    GJ
        (``kind="fiber"`` only) Torsional stiffness for the ``-GJ``
        flag.  ``None`` (default) resolves from the analyzer's
        ``warping().GJ`` — a rigidity-form value, valid in every mode
        with no reference modulus.  The flag is always emitted
        (harmless but inert in 2-D models).
    bars
        (``kind="fiber"`` only) Discrete :class:`Bar` overlay (ADR
        0080 B3) — rebar on top of the meshed face, in the analyzer's
        **authoring (x, y) axes**.  Appended to the Gauss fibers
        through the same axis mapping about the elastic centroid;
        concrete area is not deducted (documented).
    """

    analysis: "SectionProperties"
    kind: Literal["elastic", "fiber"] = "elastic"
    E: float | None = None
    G: float | None = None
    ndm: Literal[2, 3] = 3
    fibers: Mapping[str, UniaxialMaterial] | None = None
    GJ: float | None = None
    bars: tuple[Bar, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("elastic", "fiber"):
            raise ValueError(
                f"ComputedSection: kind must be 'elastic' or 'fiber', "
                f"got {self.kind!r}."
            )
        if self.kind == "elastic":
            if self.fibers is not None or self.GJ is not None or self.bars:
                raise ValueError(
                    "ComputedSection: fibers=, GJ=, and bars= are "
                    "fiber-only arguments — not valid on the default "
                    "elastic lowering (pass kind='fiber')."
                )
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
            return

        # -- kind="fiber" (Amendment A2) ------------------------------
        handle = self.analysis.name or "section"
        if self.E is not None or self.G is not None:
            raise ValueError(
                f"ComputedSection: kind='fiber' takes no reference "
                f"moduli — the fiber materials own the constitutive "
                f"response (remove E=/G= on {handle!r})."
            )
        if self.ndm != 3:
            raise ValueError(
                "ComputedSection: ndm= applies to the elastic lowering "
                "only — section Fiber has no per-ndm form."
            )
        if not self.fibers:
            raise ValueError(
                f"ComputedSection: kind='fiber' requires "
                f"fibers={{pg: UniaxialMaterial}} covering the "
                f"analyzer's material regions ({handle!r})."
            )
        if self.GJ is not None and self.GJ <= 0:
            raise ValueError(
                f"ComputedSection: GJ must be > 0 if supplied, "
                f"got {self.GJ}."
            )
        if self.analysis.geometric_only:
            raise ValueError(
                f"{handle}: kind='fiber' needs material regions — a "
                f"geometric-only analyzer has no PGs to key fibers= "
                f"by. Construct SectionProperties with materials=."
            )
        # defensive copy + exact-cover gate (fail at construction,
        # not at emit — the analyzer's regions are already known)
        object.__setattr__(self, "fibers", dict(self.fibers))
        assert self.fibers is not None
        pgs = set(self.analysis.materials)
        given = set(self.fibers)
        missing = sorted(pgs - given)
        unknown = sorted(given - pgs)
        if missing or unknown:
            raise ValueError(
                f"{handle}: fibers= must exactly cover the analyzer's "
                f"material regions {sorted(pgs)} — "
                f"missing {missing}, unknown {unknown}."
            )

    def resolve(self) -> ElasticSection | Fiber:
        """Run the lowering now and return the plain primitive this
        section emits as (:class:`ElasticSection` or :class:`Fiber`)."""
        if self.kind == "elastic":
            from apeGmsh.sections._lowering import lower_to_elastic

            params = lower_to_elastic(self.analysis, E=self.E, G=self.G)
            return ElasticSection(**params.section_kwargs(self.ndm))

        from apeGmsh.sections._lowering import lower_to_fiber

        data = lower_to_fiber(self.analysis)
        assert self.fibers is not None  # gated in __post_init__
        mats = tuple(self.fibers[name] for name in data.region_names)
        points = tuple(
            FiberPoint(
                material=mats[r],
                y=float(y),
                z=float(z),
                area=float(a),
            )
            for y, z, a, r in zip(data.y, data.z, data.area, data.region)
        )
        if self.bars:
            geo = self.analysis.geometric()
            bar_points = tuple(
                FiberPoint(
                    material=b.material,
                    y=float(b.y - geo.cy),
                    z=float(b.x - geo.cx),
                    area=float(b.area),
                )
                for b in self.bars
            )
            points = points + bar_points
        gj = self.GJ if self.GJ is not None else self.analysis.warping().GJ
        return Fiber(fibers=points, GJ=gj)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        self.resolve()._emit(emitter, tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.kind == "elastic":
            return ()
        seen: dict[int, UniaxialMaterial] = {}
        for mat in (self.fibers or {}).values():
            seen.setdefault(id(mat), mat)
        for b in self.bars:
            seen.setdefault(id(b.material), b.material)
        return tuple(seen.values())
