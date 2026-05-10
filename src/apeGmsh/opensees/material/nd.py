"""
Typed primitives for OpenSees ``nDMaterial`` commands.

Phase 1B ships the priority-1 set: ``ElasticIsotropic``,
``J2Plasticity``, ``DruckerPrager``. The exotic soil and ASD damage
models (``PressureIndepMultiYield``, ``PM4Sand``, ``ASDConcrete3D``)
are deferred — their parameter sets are large, version-dependent,
and would benefit from an OpenSees expert sign-off before being
locked in.

Per P12, every user-facing parameter is a fully typed keyword on the
matching dataclass and on the namespace method. The OpenSees-vocabulary
varargs only appear inside ``_emit`` where the boundary is internal.

The Tcl signatures these classes emit:

* ``nDMaterial ElasticIsotropic tag E nu rho``
* ``nDMaterial J2Plasticity tag K G sig0 sigInf delta H eta``
* ``nDMaterial DruckerPrager tag K G sigmaY rho rhoBar Kinf Ko delta1 delta2 H theta``
"""
from __future__ import annotations

from dataclasses import dataclass

from .._internal.types import NDMaterial, Primitive
from ..emitter.base import Emitter


__all__ = [
    "ElasticIsotropic",
    "J2Plasticity",
    "DruckerPrager",
]


# ---------------------------------------------------------------------------
# ElasticIsotropic — 3-D / 2-D linear elastic continuum material
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ElasticIsotropic(NDMaterial):
    """Linear-elastic isotropic continuum material.

    Tcl signature::

        nDMaterial ElasticIsotropic $tag $E $nu <$rho>

    Parameters
    ----------
    E
        Young's modulus. Must be strictly positive.
    nu
        Poisson's ratio. OpenSees enforces ``0 <= nu < 0.5``.
    rho
        Mass density. Defaults to ``0.0`` (statics). Must be ``>= 0``.
    """

    E: float
    nu: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(
                f"ElasticIsotropic: E must be > 0, got {self.E!r}"
            )
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(
                f"ElasticIsotropic: nu must be in [0, 0.5), got {self.nu!r}"
            )
        if self.rho < 0:
            raise ValueError(
                f"ElasticIsotropic: rho must be >= 0, got {self.rho!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial("ElasticIsotropic", tag, self.E, self.nu, self.rho)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# J2Plasticity — von Mises plasticity with isotropic + nonlinear hardening
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class J2Plasticity(NDMaterial):
    """von-Mises (J2) plasticity with combined nonlinear hardening.

    Tcl signature::

        nDMaterial J2Plasticity $tag $K $G $sig0 $sigInf $delta $H <$eta>

    Parameters
    ----------
    K
        Bulk modulus. Must be strictly positive.
    G
        Shear modulus. Must be strictly positive.
    sig0
        Initial yield stress (von-Mises radius at zero plastic strain).
        Must be strictly positive.
    sigInf
        Saturation yield stress (asymptote of the exponential hardening
        term). ``sigInf >= sig0`` for monotonic hardening.
    delta
        Exponential decay rate for the saturation term. Must be ``>= 0``.
    H
        Linear isotropic hardening modulus. Must be ``>= 0``.
    eta
        Viscoplastic regularization parameter. Defaults to ``0.0``
        (rate-independent). Must be ``>= 0``.
    """

    K: float
    G: float
    sig0: float
    sigInf: float
    delta: float
    H: float
    eta: float = 0.0

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"J2Plasticity: K must be > 0, got {self.K!r}")
        if self.G <= 0:
            raise ValueError(f"J2Plasticity: G must be > 0, got {self.G!r}")
        if self.sig0 <= 0:
            raise ValueError(
                f"J2Plasticity: sig0 must be > 0, got {self.sig0!r}"
            )
        if self.delta < 0:
            raise ValueError(
                f"J2Plasticity: delta must be >= 0, got {self.delta!r}"
            )
        if self.H < 0:
            raise ValueError(
                f"J2Plasticity: H must be >= 0, got {self.H!r}"
            )
        if self.eta < 0:
            raise ValueError(
                f"J2Plasticity: eta must be >= 0, got {self.eta!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "J2Plasticity",
            tag,
            self.K,
            self.G,
            self.sig0,
            self.sigInf,
            self.delta,
            self.H,
            self.eta,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# DruckerPrager — pressure-dependent plasticity for soils / concrete
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class DruckerPrager(NDMaterial):
    """Drucker-Prager elasto-plastic continuum material.

    Tcl signature::

        nDMaterial DruckerPrager $tag $K $G $sigmaY \\
            $rho $rhoBar $Kinf $Ko $delta1 $delta2 $H $theta

    Parameters
    ----------
    K
        Bulk modulus. Must be strictly positive.
    G
        Shear modulus. Must be strictly positive.
    sigmaY
        Initial cohesive yield strength (von-Mises radius at zero
        plastic strain). Must be strictly positive.
    rho
        Drucker-Prager friction parameter (yield surface slope).
        Must be ``>= 0``.
    rhoBar
        Plastic-flow direction parameter (associated when
        ``rhoBar == rho``). Must be ``>= 0``.
    Kinf
        Saturation isotropic hardening parameter. Must be ``>= 0``.
    Ko
        Initial isotropic hardening parameter. Must be ``>= 0``.
    delta1
        Exponential rate for the saturation hardening term. Must be ``>= 0``.
    delta2
        Tension-cap exponential evolution parameter. Must be ``>= 0``.
    H
        Linear isotropic hardening modulus. Must be ``>= 0``.
    theta
        Mixed isotropic / kinematic hardening fraction
        (``0`` = purely kinematic, ``1`` = purely isotropic). OpenSees
        accepts ``0 <= theta <= 1``.
    """

    K: float
    G: float
    sigmaY: float
    rho: float
    rhoBar: float
    Kinf: float
    Ko: float
    delta1: float
    delta2: float
    H: float
    theta: float

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"DruckerPrager: K must be > 0, got {self.K!r}")
        if self.G <= 0:
            raise ValueError(f"DruckerPrager: G must be > 0, got {self.G!r}")
        if self.sigmaY <= 0:
            raise ValueError(
                f"DruckerPrager: sigmaY must be > 0, got {self.sigmaY!r}"
            )
        for name, value in (
            ("rho", self.rho),
            ("rhoBar", self.rhoBar),
            ("Kinf", self.Kinf),
            ("Ko", self.Ko),
            ("delta1", self.delta1),
            ("delta2", self.delta2),
            ("H", self.H),
        ):
            if value < 0:
                raise ValueError(
                    f"DruckerPrager: {name} must be >= 0, got {value!r}"
                )
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(
                f"DruckerPrager: theta must be in [0, 1], got {self.theta!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "DruckerPrager",
            tag,
            self.K,
            self.G,
            self.sigmaY,
            self.rho,
            self.rhoBar,
            self.Kinf,
            self.Ko,
            self.delta1,
            self.delta2,
            self.H,
            self.theta,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
