"""
``_BeamIntegrationNS`` — backs ``ops.beamIntegration.<Type>(...)``.

Constructs and registers a typed :class:`BeamIntegration` primitive
for each OpenSees ``beamIntegration <Type>`` rule. The bridge
allocates a tag, the rule is referenced by tag from force-/disp-based
beam-column elements (see ADR 0011 / Phase 4.5).
"""
from __future__ import annotations

from ...integration import (
    HingeEndpoint,
    HingeMidpoint,
    HingeRadau,
    HingeRadauTwo,
    Legendre,
    Lobatto,
    NewtonCotes,
    Radau,
    Trapezoidal,
)
from ..types import Section
from ._base import _BridgeNamespace


__all__ = ["_BeamIntegrationNS"]


class _BeamIntegrationNS(_BridgeNamespace):
    """``ops.beamIntegration.<Type>(...)``.

    Each method constructs a typed :class:`BeamIntegration` rule,
    registers it with the bridge (allocating its tag), and returns
    the typed instance.
    """

    # -- Uniform-section quadrature rules -------------------------------

    def Lobatto(self, *, section: Section, n_ip: int) -> Lobatto:
        """``beamIntegration Lobatto`` — Gauss-Lobatto, IPs include both ends."""
        return self._bridge._register(Lobatto(section=section, n_ip=n_ip))

    def Legendre(self, *, section: Section, n_ip: int) -> Legendre:
        """``beamIntegration Legendre`` — Gauss-Legendre, interior IPs only."""
        return self._bridge._register(Legendre(section=section, n_ip=n_ip))

    def NewtonCotes(self, *, section: Section, n_ip: int) -> NewtonCotes:
        """``beamIntegration NewtonCotes`` — closed Newton-Cotes."""
        return self._bridge._register(NewtonCotes(section=section, n_ip=n_ip))

    def Radau(self, *, section: Section, n_ip: int) -> Radau:
        """``beamIntegration Radau`` — Gauss-Radau."""
        return self._bridge._register(Radau(section=section, n_ip=n_ip))

    def Trapezoidal(self, *, section: Section, n_ip: int) -> Trapezoidal:
        """``beamIntegration Trapezoidal`` — composite trapezoidal."""
        return self._bridge._register(Trapezoidal(section=section, n_ip=n_ip))

    # -- Concentrated-plasticity hinge rules ----------------------------

    def HingeRadau(
        self, *,
        section_i: Section, lp_i: float,
        section_j: Section, lp_j: float,
        section_interior: Section,
    ) -> HingeRadau:
        """``beamIntegration HingeRadau`` — 2-point Radau in each hinge."""
        return self._bridge._register(
            HingeRadau(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            )
        )

    def HingeRadauTwo(
        self, *,
        section_i: Section, lp_i: float,
        section_j: Section, lp_j: float,
        section_interior: Section,
    ) -> HingeRadauTwo:
        """``beamIntegration HingeRadauTwo`` — endpoint-anchored Radau."""
        return self._bridge._register(
            HingeRadauTwo(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            )
        )

    def HingeMidpoint(
        self, *,
        section_i: Section, lp_i: float,
        section_j: Section, lp_j: float,
        section_interior: Section,
    ) -> HingeMidpoint:
        """``beamIntegration HingeMidpoint`` — 1-point at hinge midpoint."""
        return self._bridge._register(
            HingeMidpoint(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            )
        )

    def HingeEndpoint(
        self, *,
        section_i: Section, lp_i: float,
        section_j: Section, lp_j: float,
        section_interior: Section,
    ) -> HingeEndpoint:
        """``beamIntegration HingeEndpoint`` — 1-point at element end."""
        return self._bridge._register(
            HingeEndpoint(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            )
        )
