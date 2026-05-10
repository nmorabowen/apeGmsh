"""
Typed ``beamIntegration`` primitives — rules that drive force-/disp-
based beam-column elements.

A :class:`BeamIntegration` composes one or more :class:`Section`
primitives and tells the beam-column element how many integration
points to place and where. The beam-column element then references
the rule **by tag**, not by composition.

OpenSees command shapes
=======================

::

    beamIntegration Lobatto      tag secTag N
    beamIntegration Legendre     tag secTag N
    beamIntegration NewtonCotes  tag secTag N
    beamIntegration Radau        tag secTag N
    beamIntegration Trapezoidal  tag secTag N

    # Concentrated-plasticity rules
    beamIntegration HingeRadau     tag secITag lpI secJTag lpJ secETag
    beamIntegration HingeRadauTwo  tag secITag lpI secJTag lpJ secETag
    beamIntegration HingeMidpoint  tag secITag lpI secJTag lpJ secETag
    beamIntegration HingeEndpoint  tag secITag lpI secJTag lpJ secETag

    # Per-IP heterogeneous sections (rare; deferred)
    beamIntegration UserDefined    tag N secTag_1 ... secTag_N x_1 ... x_N w_1 ... w_N
    beamIntegration FixedLocation  tag N secTag_1 ... secTag_N x_1 ... x_N

This module ships the uniform-section rules (Lobatto / Legendre /
NewtonCotes / Radau / Trapezoidal) plus the four concentrated-
plasticity hinge variants. UserDefined / FixedLocation are deferred
to a follow-up — their parameter shapes are arrays-of-arrays and
deserve their own typed value objects.

See also
--------
:class:`apeGmsh.opensees.element.beam_column.forceBeamColumn` and
:class:`~apeGmsh.opensees.element.beam_column.dispBeamColumn`, which
compose a :class:`BeamIntegration` via their ``integration`` field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._internal.tag_resolution import resolve_tag
from ._internal.types import BeamIntegration, Primitive, Section

if TYPE_CHECKING:
    from .emitter.base import Emitter


__all__ = [
    # Uniform-section quadrature rules
    "Lobatto",
    "Legendre",
    "NewtonCotes",
    "Radau",
    "Trapezoidal",
    # Concentrated-plasticity hinge rules
    "HingeRadau",
    "HingeRadauTwo",
    "HingeMidpoint",
    "HingeEndpoint",
]


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _check_n_ip(type_name: str, n_ip: int, min_n: int) -> None:
    if n_ip < min_n:
        raise ValueError(
            f"{type_name}: n_ip must be >= {min_n}, got {n_ip}."
        )


def _check_lp(type_name: str, lp_i: float, lp_j: float) -> None:
    if lp_i <= 0:
        raise ValueError(f"{type_name}: lp_i must be > 0, got {lp_i}.")
    if lp_j <= 0:
        raise ValueError(f"{type_name}: lp_j must be > 0, got {lp_j}.")


# ---------------------------------------------------------------------------
# Uniform-section quadrature rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Lobatto(BeamIntegration):
    """``beamIntegration Lobatto tag secTag N`` — Gauss-Lobatto quadrature.

    Lobatto quadrature **includes both element endpoints** as
    integration points. Convergence is :math:`O(h^{2N-2})` for smooth
    integrands, plus the IP at each end means localized plasticity at
    the ends is sampled directly. The most common choice for
    distributed-plasticity beam-columns.

    Parameters
    ----------
    section
        Single section integrated at every IP.
    n_ip
        Number of integration points. Must be >= 2 (Lobatto's two
        endpoints).
    """

    section: Section
    n_ip: int

    def __post_init__(self) -> None:
        _check_n_ip("Lobatto", self.n_ip, min_n=2)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        sec_tag = resolve_tag(emitter, self.section)
        emitter.beamIntegration("Lobatto", tag, sec_tag, self.n_ip)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


@dataclass(frozen=True, kw_only=True, slots=True)
class Legendre(BeamIntegration):
    """``beamIntegration Legendre tag secTag N`` — Gauss-Legendre quadrature.

    Legendre quadrature is **interior only** (no IPs at endpoints).
    Convergence is :math:`O(h^{2N})` for smooth integrands, but
    localized plasticity at the ends is NOT sampled — usually inferior
    to Lobatto for nonlinear beam-columns.

    Parameters
    ----------
    section
        Single section integrated at every IP.
    n_ip
        Number of integration points. Must be >= 1.
    """

    section: Section
    n_ip: int

    def __post_init__(self) -> None:
        _check_n_ip("Legendre", self.n_ip, min_n=1)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        sec_tag = resolve_tag(emitter, self.section)
        emitter.beamIntegration("Legendre", tag, sec_tag, self.n_ip)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


@dataclass(frozen=True, kw_only=True, slots=True)
class NewtonCotes(BeamIntegration):
    """``beamIntegration NewtonCotes tag secTag N`` — closed Newton-Cotes.

    Closed Newton-Cotes places IPs at evenly spaced points including
    both endpoints. Convergence order depends on N (parity dependent).
    Rarely used for nonlinear beam-columns; included for completeness.

    Parameters
    ----------
    section
        Single section integrated at every IP.
    n_ip
        Number of integration points. Must be >= 2.
    """

    section: Section
    n_ip: int

    def __post_init__(self) -> None:
        _check_n_ip("NewtonCotes", self.n_ip, min_n=2)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        sec_tag = resolve_tag(emitter, self.section)
        emitter.beamIntegration("NewtonCotes", tag, sec_tag, self.n_ip)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


@dataclass(frozen=True, kw_only=True, slots=True)
class Radau(BeamIntegration):
    """``beamIntegration Radau tag secTag N`` — Gauss-Radau quadrature.

    Radau quadrature includes one endpoint (typically the i-end) as
    an integration point. Used internally by the Hinge* rules; rarely
    invoked directly.

    Parameters
    ----------
    section
        Single section integrated at every IP.
    n_ip
        Number of integration points. Must be >= 1.
    """

    section: Section
    n_ip: int

    def __post_init__(self) -> None:
        _check_n_ip("Radau", self.n_ip, min_n=1)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        sec_tag = resolve_tag(emitter, self.section)
        emitter.beamIntegration("Radau", tag, sec_tag, self.n_ip)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


@dataclass(frozen=True, kw_only=True, slots=True)
class Trapezoidal(BeamIntegration):
    """``beamIntegration Trapezoidal tag secTag N`` — trapezoidal rule.

    Composite trapezoidal rule with N intervals (N+1 IPs including
    both ends). Low-order; included for completeness.

    Parameters
    ----------
    section
        Single section integrated at every IP.
    n_ip
        Number of integration points. Must be >= 2.
    """

    section: Section
    n_ip: int

    def __post_init__(self) -> None:
        _check_n_ip("Trapezoidal", self.n_ip, min_n=2)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        sec_tag = resolve_tag(emitter, self.section)
        emitter.beamIntegration("Trapezoidal", tag, sec_tag, self.n_ip)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


# ---------------------------------------------------------------------------
# Concentrated-plasticity hinge rules
# ---------------------------------------------------------------------------
#
# Hinge* rules place plastic-hinge regions of length lp_i / lp_j at the
# element ends (with their own sections), and an elastic interior
# section in the middle. The four variants differ only in the
# quadrature used inside each plastic hinge:
#
#   HingeRadau     — 2-point Radau in each hinge (4 IPs total)
#   HingeRadauTwo  — 2-point Radau, both endpoints (4 IPs total)
#   HingeMidpoint  — 1-point midpoint of each hinge (2 IPs total)
#   HingeEndpoint  — 1-point at element end (2 IPs total)
#
# Shape is identical across the four: (sec_i, lp_i, sec_j, lp_j, sec_e).
# We share validation and dependency declaration via a common emit
# helper inside each class.
# ---------------------------------------------------------------------------

def _emit_hinge(
    emitter: "Emitter",
    tag: int,
    type_token: str,
    sec_i: Section,
    lp_i: float,
    sec_j: Section,
    lp_j: float,
    sec_e: Section,
) -> None:
    """Common emit logic for Hinge* rules. Args layout per OpenSees Tcl::

        beamIntegration HingeRadau tag secI_tag lp_i secJ_tag lp_j secE_tag
    """
    sec_i_tag = resolve_tag(emitter, sec_i)
    sec_j_tag = resolve_tag(emitter, sec_j)
    sec_e_tag = resolve_tag(emitter, sec_e)
    emitter.beamIntegration(
        type_token, tag,
        sec_i_tag, lp_i,
        sec_j_tag, lp_j,
        sec_e_tag,
    )


def _hinge_deps(
    sec_i: Section, sec_j: Section, sec_e: Section,
) -> tuple[Primitive, ...]:
    """Dedupe-by-id dependency tuple for a Hinge* rule.

    The three sections may be the same instance (e.g. user passes the
    same Fiber section for i / j and an Elastic section for interior);
    return only unique instances.
    """
    seen: dict[int, Primitive] = {}
    for s in (sec_i, sec_j, sec_e):
        seen.setdefault(id(s), s)
    return tuple(seen.values())


@dataclass(frozen=True, kw_only=True, slots=True)
class HingeRadau(BeamIntegration):
    """``beamIntegration HingeRadau tag secI lpI secJ lpJ secE``.

    2-point Radau quadrature in each plastic hinge zone (lp_i at i,
    lp_j at j) plus the elastic interior. Standard concentrated-
    plasticity rule for capacity-design nonlinear beam-columns.

    Parameters
    ----------
    section_i, section_j
        Plastic-hinge sections at the i / j ends. Typically fiber
        sections capturing the inelastic constitutive response.
    lp_i, lp_j
        Plastic-hinge lengths. Both must be > 0.
    section_interior
        Elastic section spanning the middle of the element.
    """

    section_i:        Section
    lp_i:             float
    section_j:        Section
    lp_j:             float
    section_interior: Section

    def __post_init__(self) -> None:
        _check_lp("HingeRadau", self.lp_i, self.lp_j)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _emit_hinge(
            emitter, tag, "HingeRadau",
            self.section_i, self.lp_i,
            self.section_j, self.lp_j,
            self.section_interior,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return _hinge_deps(self.section_i, self.section_j, self.section_interior)


@dataclass(frozen=True, kw_only=True, slots=True)
class HingeRadauTwo(BeamIntegration):
    """``beamIntegration HingeRadauTwo tag secI lpI secJ lpJ secE``.

    Variant of :class:`HingeRadau` with Radau quadrature anchored at
    both endpoints. Slightly different IP placement; same parameter
    shape.
    """

    section_i:        Section
    lp_i:             float
    section_j:        Section
    lp_j:             float
    section_interior: Section

    def __post_init__(self) -> None:
        _check_lp("HingeRadauTwo", self.lp_i, self.lp_j)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _emit_hinge(
            emitter, tag, "HingeRadauTwo",
            self.section_i, self.lp_i,
            self.section_j, self.lp_j,
            self.section_interior,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return _hinge_deps(self.section_i, self.section_j, self.section_interior)


@dataclass(frozen=True, kw_only=True, slots=True)
class HingeMidpoint(BeamIntegration):
    """``beamIntegration HingeMidpoint tag secI lpI secJ lpJ secE``.

    1-point quadrature at the midpoint of each plastic-hinge zone.
    Simplest concentrated-plasticity rule; same parameter shape as
    :class:`HingeRadau`.
    """

    section_i:        Section
    lp_i:             float
    section_j:        Section
    lp_j:             float
    section_interior: Section

    def __post_init__(self) -> None:
        _check_lp("HingeMidpoint", self.lp_i, self.lp_j)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _emit_hinge(
            emitter, tag, "HingeMidpoint",
            self.section_i, self.lp_i,
            self.section_j, self.lp_j,
            self.section_interior,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return _hinge_deps(self.section_i, self.section_j, self.section_interior)


@dataclass(frozen=True, kw_only=True, slots=True)
class HingeEndpoint(BeamIntegration):
    """``beamIntegration HingeEndpoint tag secI lpI secJ lpJ secE``.

    1-point quadrature at the element end of each plastic-hinge zone
    (the extreme fibre). Same parameter shape as :class:`HingeRadau`.
    """

    section_i:        Section
    lp_i:             float
    section_j:        Section
    lp_j:             float
    section_interior: Section

    def __post_init__(self) -> None:
        _check_lp("HingeEndpoint", self.lp_i, self.lp_j)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _emit_hinge(
            emitter, tag, "HingeEndpoint",
            self.section_i, self.lp_i,
            self.section_j, self.lp_j,
            self.section_interior,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return _hinge_deps(self.section_i, self.section_j, self.section_interior)
