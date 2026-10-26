"""
Frame property modifiers — ETABS-style stiffness / mass multipliers.

:class:`FrameModifiers` is a frozen value object (not a primitive: it emits
nothing and holds no tag) passed as ``modifiers=`` on the frame element
specs. Each field multiplies one section property of the element it is
attached to, in **OpenSees local axes**:

==========  ==================================================  ============
field       multiplies                                          ETABS name
==========  ==================================================  ============
``A``       axial area ``A``                                    ``A``
``Avy``     shear area along local y                            ``As2``
``Avz``     shear area along local z                            ``As3``
``J``       torsion constant ``J``                              ``J``
``Iy``      second moment about local y                         ``I22``
``Iz``      second moment about local z                         ``I33``
``mass``    mass per unit length (``-mass``)                    ``mass``
``weight``  frame self-weight                                   ``weight``
==========  ==================================================  ============

The modifier reaches the emitted command, never the declaration: an
``elasticBeamColumn`` emits ``A * m.A`` and so on, and a force- or
displacement-based element whose integration holds
:class:`~apeGmsh.opensees.section.beam.ElasticSection` s gets a modified
*copy* of each section (and of the integration rule). The shared section the
user declared is never mutated, and one copy is made per distinct
``(section, modifiers)`` value within a session, so two element groups with
the same section and the same modifiers share one modified section. Each
bridge keeps its own copies (:class:`ModifiedCopies`); two sessions never
share one.

Fail loud: a non-unit modifier on a property the element does not have
(``Avy`` on an Euler-Bernoulli ``elasticBeamColumn``, ``Iy`` on a 2-D
element, ``mass`` on an element declared without ``mass=``) is refused at
construction, as is a non-unit ``weight``: apeGmsh synthesises no frame
self-weight (OpenSees beam-columns accept no ``eleLoad -selfWeight``; frame
dead load is a user ``beam_uniform``), so there is nothing for it to scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Protocol

from .._internal.types import BeamIntegration, Section
from ..section.beam import ElasticSection


__all__ = ["FrameModifiers"]


#: Field order of :class:`FrameModifiers` (OpenSees names).
_NAMES: tuple[str, ...] = (
    "A", "Avy", "Avz", "J", "Iy", "Iz", "mass", "weight",
)
#: The stiffness modifiers (strictly positive: a zero would make the
#: element stiffness singular in that mode).
_STIFFNESS: frozenset[str] = frozenset(_NAMES[:6])


@dataclass(frozen=True, kw_only=True, slots=True)
class FrameModifiers:
    """Frame section property modifiers, in OpenSees local axes.

    Every field defaults to ``1.0`` (gross section). The six stiffness
    modifiers (``A``, ``Avy``, ``Avz``, ``J``, ``Iy``, ``Iz``) must be
    finite and ``> 0``; ``mass`` and ``weight`` finite and ``>= 0``.

    Attach to an element spec with ``modifiers=``::

        cracked = FrameModifiers.etabs(I22=0.7, I33=0.7)   # ACI 318-25 columns
        ops.element.elasticBeamColumn(pg="Cols", transf=t, A=..., E=...,
                                      Iz=..., Iy=..., G=..., J=...,
                                      modifiers=cracked)

    Use :meth:`etabs` to construct from ETABS-named modifiers.
    """

    A: float = 1.0
    Avy: float = 1.0
    Avz: float = 1.0
    J: float = 1.0
    Iy: float = 1.0
    Iz: float = 1.0
    mass: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in _NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"FrameModifiers: {name} must be a number, got "
                    f"{value!r}."
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"FrameModifiers: {name} must be finite, got {value!r}."
                )
            if name in _STIFFNESS and not value > 0.0:
                raise ValueError(
                    f"FrameModifiers: the stiffness modifier {name} must be "
                    f"> 0, got {value!r} (a zero stiffness makes the element "
                    "singular in that mode)."
                )
            if value < 0.0:
                raise ValueError(
                    f"FrameModifiers: {name} must be >= 0, got {value!r}."
                )

    @classmethod
    def etabs(
        cls,
        *,
        A: float = 1.0,
        As2: float = 1.0,
        As3: float = 1.0,
        J: float = 1.0,
        I22: float = 1.0,
        I33: float = 1.0,
        mass: float = 1.0,
        weight: float = 1.0,
    ) -> "FrameModifiers":
        """Build from ETABS frame property modifier names.

        Axis mapping: ETABS local 1 runs along the member, as OpenSees
        local x does. Choose the element's ``geomTransf`` ``vecxz`` along
        **ETABS local 3**: OpenSees local z then equals ETABS 3, and since
        both triads are right-handed with a common first axis, OpenSees
        local y = z x x = 3 x 1 = ETABS 2. Under that ``vecxz`` the mapping
        is ``As2 -> Avy``, ``As3 -> Avz``, ``I22 -> Iy`` (bending about
        local 2 / y), ``I33 -> Iz`` (bending about local 3 / z, the major
        axis of an ETABS frame section). ``A``, ``J``, ``mass`` and
        ``weight`` carry no axis. With any other ``vecxz`` the 2 and 3
        modifiers land on whichever OpenSees axes it defines, so check it.

        For a vertical ETABS column (1 = +Z, 2 = +X, 3 = +Y) that is
        ``vecxz=(0, 1, 0)``; for a beam along +X (1 = +X, 2 = +Z,
        3 = -Y) it is ``vecxz=(0, -1, 0)``.

        **The ETABS importer does not use this convention.**
        ``interop/etabs_import.py`` (``_VECXZ``) gives a vertical member
        ``vecxz=(1, 0, 0)`` and any other member ``(0, 0, 1)``: for a member
        at its default orientation both are ETABS **2**, not 3, and the
        importer ignores ``Frame.rotation``. Under that ``vecxz`` OpenSees
        local z is ETABS 2, so ``I33`` lands on the minor axis (and ``I22``
        on the major one). The importer passes no ``modifiers=`` today;
        wiring it to this method needs ``vecxz`` along ETABS 3 first.
        """
        return cls(
            A=A, Avy=As2, Avz=As3, J=J, Iy=I22, Iz=I33,
            mass=mass, weight=weight,
        )

    @property
    def is_identity(self) -> bool:
        """True when every modifier is ``1.0`` (gross section)."""
        return all(getattr(self, n) == 1.0 for n in _NAMES)

    def non_unit(self) -> dict[str, float]:
        """The modifiers that differ from ``1.0``, by OpenSees name."""
        return {
            n: getattr(self, n) for n in _NAMES if getattr(self, n) != 1.0
        }


# ---------------------------------------------------------------------------
# Element-side checks (pure functions, used by the element specs).
# ---------------------------------------------------------------------------

def check_frame_modifiers(
    owner: str,
    mods: FrameModifiers | None,
    *,
    lacks: dict[str, str],
    has_mass: bool,
) -> None:
    """Refuse a non-unit modifier that the element ``owner`` cannot apply.

    ``lacks`` maps a modifier name to why the element has no such
    property. ``weight`` and (when ``has_mass`` is false) ``mass`` are
    refused for every element; see the module docstring.
    """
    if mods is None:
        return
    if not isinstance(mods, FrameModifiers):
        raise TypeError(
            f"{owner}: modifiers must be a FrameModifiers, got "
            f"{type(mods).__name__}."
        )
    for name, value in mods.non_unit().items():
        if name in lacks:
            raise ValueError(
                f"{owner}: modifiers.{name}={value!r} acts on a property this "
                f"element does not have ({lacks[name]}). Set it to 1.0 or "
                "use an element that carries the property."
            )
    if mods.weight != 1.0:
        raise ValueError(
            f"{owner}: modifiers.weight={mods.weight!r} cannot be applied: "
            "apeGmsh synthesises no frame self-weight (OpenSees beam-columns "
            "accept no eleLoad -selfWeight), so frame dead load is the "
            "beam_uniform you declare. Scale that load by the weight "
            "modifier yourself and set weight=1.0."
        )
    if mods.mass != 1.0 and not has_mass:
        raise ValueError(
            f"{owner}: modifiers.mass={mods.mass!r} has no element mass to "
            "scale: the element was declared without mass=. Pass mass= (mass "
            "per unit length) or set modifiers.mass=1.0; nodal masses are "
            "not element properties and are not scaled."
        )


def scaled(value: float, factor: float) -> float:
    """``value * factor``, returning ``value`` itself for a unit factor."""
    return value if factor == 1.0 else value * factor


def scaled_mass(
    mass: float | None, mods: FrameModifiers | None,
) -> float | None:
    """The element ``-mass`` after the ``mass`` modifier."""
    if mass is None or mods is None:
        return mass
    return scaled(mass, mods.mass)

# ---------------------------------------------------------------------------
# Section-side copies for the force-/displacement-based elastic route.
# ---------------------------------------------------------------------------

def is_reference_stub(prim: object) -> bool:
    """True for a primitive with an unset dataclass field.

    A constructed frozen dataclass always has every field set; only the
    replay dry run's stand-in (``_internal/replay_types._stub``, made with
    ``object.__new__``) lacks them, and only a tag lookup may use it.
    """
    return any(not hasattr(prim, f.name) for f in fields(prim))  # type: ignore[arg-type]


class ModifiedCopies:
    """A memo of modified copies: one per distinct ``(base, modifiers)``
    value, so specs that share a section and a modifier set reference the
    very same copy (the bridge deduplicates by identity).

    Each bridge owns one (``apeSees._frame_modifier_copies``), so two
    sessions never share a copy and the memo dies with its session. A
    throwaway instance serves the construction-time checks.
    """

    __slots__ = ("_sections", "_integrations")

    def __init__(self) -> None:
        self._sections: dict[tuple[Section, FrameModifiers], Section] = {}
        self._integrations: dict[
            tuple[BeamIntegration, FrameModifiers], BeamIntegration
        ] = {}

    def section(
        self, owner: str, sec: Section, mods: FrameModifiers,
    ) -> Section:
        """A copy of the ``ElasticSection`` ``sec`` with ``mods`` applied."""
        if not isinstance(sec, ElasticSection):
            raise TypeError(
                f"{owner}: modifiers reach only an ElasticSection inside the "
                f"integration rule, but it holds a {type(sec).__name__}, "
                "whose stiffness comes from its materials. Scale those, or "
                "use an ElasticSection (or elasticBeamColumn) for the "
                "modified member."
            )
        key = (sec, mods)
        hit = self._sections.get(key)
        if hit is not None:
            return hit
        out = _modified_elastic_section(owner, sec, mods)
        self._sections[key] = out
        return out

    def integration(
        self, owner: str, integ: BeamIntegration, mods: FrameModifiers,
    ) -> BeamIntegration:
        """A copy of ``integ`` whose every section carries ``mods``.

        Every :class:`Section` field of the rule (``section`` of a uniform
        rule, ``section_i`` / ``section_j`` / ``section_interior`` of a
        hinge rule) must be an ``ElasticSection``; anything else is refused.
        """
        changes: dict[str, Section] = {}
        for f in fields(integ):  # type: ignore[arg-type]
            value = getattr(integ, f.name)
            if isinstance(value, Section):
                changes[f.name] = self.section(owner, value, mods)
        if not changes:
            raise TypeError(
                f"{owner}: the integration rule {type(integ).__name__} holds "
                "no section a modifier could reach."
            )
        key = (integ, mods)
        hit = self._integrations.get(key)
        if hit is not None:
            return hit
        out: BeamIntegration = replace(integ, **changes)  # type: ignore[type-var]
        self._integrations[key] = out
        return out


def _modified_elastic_section(
    owner: str, sec: ElasticSection, mods: FrameModifiers,
) -> Section:
    """A new ``ElasticSection``: ``sec`` with ``mods`` applied (no memo)."""
    three_d = sec._is_3d()
    if three_d:
        shear = sec.alphaY is not None or sec.alphaZ is not None
    else:
        shear = sec.G is not None and sec.alphaY is not None
    lacks: dict[str, str] = {}
    if not three_d:
        for n, what in (("Iy", "second moment about local y"),
                        ("J", "torsion constant"),
                        ("Avz", "shear area along local z")):
            lacks[n] = f"a 2-D ElasticSection has no {what}"
    if not shear:
        why = ("the ElasticSection has no shear flexibility: pass alphaY "
               "(and G in 2-D) to make it a shear section")
        lacks.setdefault("Avy", why)
        lacks.setdefault("Avz", why)
    for name, value in mods.non_unit().items():
        if name in lacks:
            raise ValueError(
                f"{owner}: modifiers.{name}={value!r} acts on a property the "
                f"section does not have ({lacks[name]})."
            )

    changes: dict[str, float] = {
        "A": scaled(sec.A, mods.A),
        "Iz": scaled(sec.Iz, mods.Iz),
    }
    if three_d:
        assert sec.Iy is not None and sec.J is not None
        changes["Iy"] = scaled(sec.Iy, mods.Iy)
        changes["J"] = scaled(sec.J, mods.J)
    if shear:
        # Shear stiffness is G * alpha * A: the shear area is alpha * A, so
        # alpha' = alpha * m_Av / m_A keeps an A modifier off the shear
        # area (ETABS scales them independently). An absent alpha of a 3-D
        # shear section emits as OpenSees's default 1.0.
        if mods.Avy != 1.0 or mods.A != 1.0:
            changes["alphaY"] = (
                (sec.alphaY if sec.alphaY is not None else 1.0)
                * mods.Avy / mods.A
            )
        if three_d and (mods.Avz != 1.0 or mods.A != 1.0):
            changes["alphaZ"] = (
                (sec.alphaZ if sec.alphaZ is not None else 1.0)
                * mods.Avz / mods.A
            )
    out: Section = replace(sec, **changes)
    return out


class SectionRouteSpec(Protocol):
    """A frame spec whose ``modifiers`` reach the sections of its
    ``integration`` (``forceBeamColumn``, ``dispBeamColumn``)."""

    @property
    def integration(self) -> BeamIntegration: ...

    @property
    def modifiers(self) -> FrameModifiers | None: ...


#: Instance-dict slot (outside the dataclass fields, so outside equality,
#: ``repr`` and the archive) holding the modified rule a spec references.
_BOUND = "_frame_modifier_integration_"


def bind_integration(
    spec: SectionRouteSpec, copies: ModifiedCopies,
) -> BeamIntegration | None:
    """Fix the modified rule ``spec`` references, taken from ``copies``.

    ``None`` when the spec needs no copy (no or identity modifiers, or a
    replay reference stub). Binding happens once: a spec already bound
    (registered on an earlier bridge, or emitted standalone) keeps its
    copy, since an emitted tag is looked up by the copy's identity.
    """
    integ, mods = spec.integration, spec.modifiers
    if mods is None or mods.is_identity or is_reference_stub(integ):
        return None
    slot = spec.__dict__
    bound: BeamIntegration | None = slot.get(_BOUND)
    if bound is None:
        bound = copies.integration(type(spec).__name__, integ, mods)
        slot[_BOUND] = bound
    return bound


def emitted_integration(spec: SectionRouteSpec) -> BeamIntegration:
    """The rule ``spec`` references: its bound modified copy, else its
    declared ``integration``.

    The bridge binds the copy when it registers the spec; a spec never
    registered binds its own copy here, on first use.
    """
    mods = spec.modifiers
    if mods is None or mods.is_identity:
        return spec.integration
    bound: BeamIntegration | None = spec.__dict__.get(_BOUND)
    if bound is None:
        bound = bind_integration(spec, ModifiedCopies())
    return spec.integration if bound is None else bound


def copies_of(integ: BeamIntegration) -> tuple[Section | BeamIntegration, ...]:
    """The modified copies a bound rule ``integ`` brings: its sections
    first, then the rule itself, deduplicated by identity."""
    seen: dict[int, Section | BeamIntegration] = {}
    for f in fields(integ):  # type: ignore[arg-type]
        value = getattr(integ, f.name)
        if isinstance(value, Section):
            seen.setdefault(id(value), value)
    seen.setdefault(id(integ), integ)
    return tuple(seen.values())
