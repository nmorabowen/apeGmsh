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

import warnings
from dataclasses import dataclass
from typing import ClassVar

from . import _asdconcrete_laws as _laws
from . import _ladruno_j2 as _lj2
from .._internal.tag_resolution import resolve_tag
from .._internal.types import NDMaterial, Primitive
from ..emitter.base import Emitter


__all__ = [
    "ElasticIsotropic",
    "J2Plasticity",
    "DruckerPrager",
    "ASDPlasticMaterial3D",
    "MohrCoulombSoil",
    "PlaneStrain",
    "ASDConcrete3D",
    "ASDRegularizationWarning",
    "LadrunoJ2",
    "LadrunoJ2Finite",
    "LadrunoConcrete3D",
    "LadrunoRCConcrete",
    "LadrunoRCFiniteStrain",
    "LadrunoCohesiveHingeBiaxial",
    "LogStrain",
    "InitDefGrad",
    "StagedStrain",
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


# ---------------------------------------------------------------------------
# ASDPlasticMaterial3D — templated YF / PF / EL / IV plasticity (Abell + Petracca)
# ---------------------------------------------------------------------------
#
# The Tcl card is a 4-string-type header followed by three keyed
# blocks.  The parser
# (``SRC/material/nD/ASDPlasticMaterial3D/OPS_AllASDPlasticMaterial3Ds.cpp``)
# reads:
#
#     nDMaterial ASDPlasticMaterial3D $tag
#       $yf $pf $el $iv
#       Begin_Internal_Variables   <name> v1 [v2 v3 ...]   ... End_Internal_Variables
#       Begin_Model_Parameters     <name> value            ... End_Model_Parameters
#       Begin_Integration_Options  <name> value            ... End_Integration_Options
#
# Internal-variable values are per-name N-tuples (size determined by
# the IV type — BackStress is 6, scalar IVs are 1).  Model parameters
# are always scalar.  Integration options carry mixed value types
# (doubles, ints, string enums) keyed by ``param_name``.
#
# Phase SSI-1: the SSI MohrCoulomb soil case lives in :class:`MohrCoulombSoil`
# below.  This generic class is the escape hatch for other YF / PF / EL
# combinations and is also what :class:`MohrCoulombSoil` constructs
# internally.
#
# Valid combinations are produced by
# ``SRC/material/nD/ASDPlasticMaterial3D/gen_ASD_material_definitions_CPP.py``;
# unsupported triples cause an OpenSees runtime error (the factory
# returns ``nullptr``).  apeGmsh does not enforce client-side: any
# ``(yf, pf, el, iv)`` shape is accepted at registration time; the
# OpenSees binary is the source of truth on which combinations exist
# in this build.


@dataclass(frozen=True, kw_only=True, slots=True)
class ASDPlasticMaterial3D(NDMaterial):
    """Generic templated ASD plasticity material (Abell / Petracca / Camata).

    Tcl signature (verbatim — line breaks for readability only)::

        nDMaterial ASDPlasticMaterial3D $tag \\
            $yf $pf $el $iv \\
            Begin_Internal_Variables  ... End_Internal_Variables \\
            Begin_Model_Parameters    ... End_Model_Parameters   \\
            Begin_Integration_Options ... End_Integration_Options

    The four type strings select the templated implementation; the
    three dict blocks populate it.  ``commitStressIncrementXX/YY/ZZ
    /XY/YZ/XZ`` responses (used by :func:`apeSees.initial_stress`)
    are defined on every ASDPlasticMaterial3D instantiation —
    independent of the YF / PF / EL / IV chosen.

    Parameters
    ----------
    yf
        Yield-function type name, e.g. ``"MohrCoulomb_YF"`` /
        ``"DruckerPrager_YF"`` / ``"VonMises_YF"`` /
        ``"HoekBrown_YF"``.
    pf
        Plastic-flow direction type name (typically matches ``yf``
        for associated flow; e.g. ``"MohrCoulomb_PF"``).
    el
        Elasticity model type name, e.g.
        ``"LinearIsotropic3D_EL"``.
    iv
        Internal-variable composition string, e.g.
        ``"BackStress(NullHardeningTensorFunction):"`` (NOTE the
        trailing colon — required by the parser's name-match).
    internal_variables
        ``{name: scalar | tuple}`` — values keyed by internal-variable
        name.  Tuple length must match the IV's declared size
        (e.g. BackStress is 6-vector; scalar IVs accept a single
        value or a 1-tuple).
    model_parameters
        ``{name: scalar}`` — model-parameter dictionary.  All values
        are stored as floats.  Unknown keys are silently consumed by
        the OpenSees parser (it forwards via ``setParameterByName``);
        prefer the typed :class:`MohrCoulombSoil` helper for the SSI
        case.
    integration_options
        ``{name: scalar | str}`` — keyed by parser option name.
        Mixed types: ``f_absolute_tol`` / ``stress_absolute_tol`` /
        ``rk45_dT_min`` are floats; ``n_max_iterations`` /
        ``rk45_niter_max`` are ints; ``integration_method`` /
        ``tangent_type`` / ``return_to_yield_surface`` are string
        enums (see the OpenSees source for valid tokens).  Empty
        dict = all defaults (Backward_Euler / Secant / 1e-6 /
        100 / Disabled / 0.01 / 110).
    """

    yf: str
    pf: str
    el: str
    iv: str
    internal_variables: tuple[tuple[str, tuple[float, ...]], ...] = ()
    model_parameters: tuple[tuple[str, float], ...] = ()
    integration_options: tuple[tuple[str, float | int | str], ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("yf", self.yf), ("pf", self.pf),
            ("el", self.el), ("iv", self.iv),
        ):
            if not value:
                raise ValueError(
                    f"ASDPlasticMaterial3D: {label}= must be non-empty"
                )
        # Internal-variable values must be per-tuple of floats.
        for name, values in self.internal_variables:
            if not name:
                raise ValueError(
                    "ASDPlasticMaterial3D: internal_variables key "
                    "must be non-empty"
                )
            if not values:
                raise ValueError(
                    "ASDPlasticMaterial3D: internal_variables "
                    f"{name!r} must have at least one value"
                )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[float | int | str] = [self.yf, self.pf, self.el, self.iv]
        args.append("Begin_Internal_Variables")
        for name, values in self.internal_variables:
            args.append(name)
            args.extend(float(v) for v in values)
        args.append("End_Internal_Variables")
        args.append("Begin_Model_Parameters")
        for name, value in self.model_parameters:
            args.append(name)
            args.append(float(value))
        args.append("End_Model_Parameters")
        args.append("Begin_Integration_Options")
        for opt_name, opt_value in self.integration_options:
            args.append(opt_name)
            # Preserve int / float / str distinction so the Tcl emit
            # renders enums (e.g. ``Backward_Euler``) as tokens, not
            # as the float ``Backward_Euler`` would coerce to NaN.
            if isinstance(opt_value, str):
                args.append(opt_value)
            elif isinstance(opt_value, bool):
                # bool BEFORE int — Python's bool isinstance(int) is True.
                args.append(1 if opt_value else 0)
            elif isinstance(opt_value, int):
                args.append(int(opt_value))
            else:
                args.append(float(opt_value))
        args.append("End_Integration_Options")
        emitter.nDMaterial("ASDPlasticMaterial3D", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# MohrCoulombSoil — typed convenience helper for the SSI rock / soil case
# ---------------------------------------------------------------------------
#
# Constructs an ASDPlasticMaterial3D with the standard
# MohrCoulomb_YF + MohrCoulomb_PF + LinearIsotropic3D_EL
# + BackStress(NullHardeningTensorFunction): composition.  Zero-fills
# the non-MohrCoulomb model parameters (AF_*, DP_*, DuncanChang_*,
# etc.) defensively — STKO's emit does the same; the OpenSees parser
# accepts unknown names without erroring (forwarded via
# ``setParameterByName``).


def MohrCoulombSoil(
    *,
    c: float,
    phi: float,
    psi: float,
    E: float,
    nu: float,
    rho: float = 0.0,
    ds: float = 1e-5,
    yield_stress: float = 1e10,
    initial_p0: float = 0.0,
    integration_method: str = "Backward_Euler",
    tangent_type: str = "Secant",
    f_absolute_tol: float = 1e-6,
    stress_absolute_tol: float = 1e-6,
    n_max_iterations: int = 100,
    return_to_yield_surface: str = "Disabled",
    rk45_dT_min: float = 0.01,
    rk45_niter_max: int = 100,
) -> ASDPlasticMaterial3D:
    """Build an ASDPlasticMaterial3D wired for Mohr-Coulomb soil / rock.

    Replaces the ~30-line dict-of-parameters call to the generic
    :class:`ASDPlasticMaterial3D` for the SSI Cerro Lindo / rock-mass
    case.

    Parameters
    ----------
    c, phi, psi
        Mohr-Coulomb cohesion (stress units), friction angle (degrees),
        dilation angle (degrees).
    E, nu, rho
        Linear-elastic Young's modulus, Poisson's ratio, mass density.
        ``rho`` defaults to ``0.0`` (static analysis).
    ds
        Mohr-Coulomb rounding parameter (small number; default ``1e-5``
        matches STKO).
    yield_stress
        Initial scalar yield stress for the ``YieldStress`` internal
        variable.  Default ``1e10`` (effectively unbounded — pure
        Mohr-Coulomb with no scalar hardening cap).
    initial_p0
        Initial confining pressure offset.  Defaults to ``0.0``.
    integration_method
        One of ``"Forward_Euler"``, ``"Forward_Euler_Subincrement"``,
        ``"Modified_Euler_Error_Control"``,
        ``"Runge_Kutta_45_Error_Control"``, ``"Backward_Euler"``
        (default), ``"Backward_Euler_LineSearch"``.
    tangent_type
        One of ``"Elastic"``, ``"Continuum"``, ``"Secant"`` (default),
        ``"Numerical_Algorithmic_FirstOrder"``,
        ``"Numerical_Algorithmic_SecondOrder"``.
    f_absolute_tol, stress_absolute_tol, n_max_iterations
        Integration solver tolerances + iteration cap.
    return_to_yield_surface
        ``"Disabled"`` (default — STKO behavior), ``"One_Step_Return"``,
        or ``"Iterative_Return"``.
    rk45_dT_min, rk45_niter_max
        RK45 sub-step controls (only used when ``integration_method``
        is an RK45 variant).

    Returns
    -------
    ASDPlasticMaterial3D
        Frozen generic-class instance ready to register via
        ``ops.nDMaterial.ASDPlasticMaterial3D(...)`` or to pass
        directly to ``ops.register(...)``.
    """
    if c < 0:
        raise ValueError(f"MohrCoulombSoil: c must be >= 0, got {c!r}")
    if not (0.0 <= phi < 90.0):
        raise ValueError(
            f"MohrCoulombSoil: phi must be in [0, 90) degrees, got {phi!r}"
        )
    if not (0.0 <= psi <= phi):
        raise ValueError(
            "MohrCoulombSoil: psi must be in [0, phi] (associated flow "
            f"is psi=phi; non-associated requires psi<phi). Got "
            f"psi={psi!r}, phi={phi!r}."
        )
    if E <= 0:
        raise ValueError(f"MohrCoulombSoil: E must be > 0, got {E!r}")
    if not (0.0 <= nu < 0.5):
        raise ValueError(
            f"MohrCoulombSoil: nu must be in [0, 0.5), got {nu!r}"
        )
    if rho < 0:
        raise ValueError(f"MohrCoulombSoil: rho must be >= 0, got {rho!r}")

    return ASDPlasticMaterial3D(
        yf="MohrCoulomb_YF",
        pf="MohrCoulomb_PF",
        el="LinearIsotropic3D_EL",
        iv="BackStress(NullHardeningTensorFunction):",
        # Only ``BackStress`` is a valid IV for this YF/PF/IV combination
        # — the MohrCoulomb_YF declares one internal variable
        # (BackStress, size 6).  DP_cohesion / YieldStress are accepted
        # by the parser but silently dropped because
        # ``getInternalVariableSizeByName(name)`` returns 0 for unknown
        # names.  Emit only the recognized IV to keep the deck minimal.
        internal_variables=(
            ("BackStress", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ),
        model_parameters=(
            # Required MohrCoulomb + elastic + density.
            ("AF_cr", 0.0),
            ("AF_ha", 0.0),
            ("DP_eta", 0.0),
            ("DP_etabar", 0.0),
            ("DP_xi_c", 0.0),
            ("Dilatancy", 0.0),
            ("DuncanChang_MaxSigma3", 0.0),
            ("DuncanChang_n", 0.0),
            ("InitialP0", initial_p0),
            ("MC_c", c),
            ("MC_ds", ds),
            ("MC_phi", phi),
            ("MC_psi", psi),
            ("MassDensity", rho),
            ("PoissonsRatio", nu),
            ("ReferencePressure", 0.0),
            ("ReferenceYoungsModulus", 0.0),
            ("ScalarLinearHardeningParameter", 0.0),
            ("TC_min_stress", 0.0),
            ("TensorLinearHardeningParameter", 0.0),
            ("YoungsModulus", E),
        ),
        integration_options=(
            ("f_absolute_tol", f_absolute_tol),
            ("stress_absolute_tol", stress_absolute_tol),
            ("n_max_iterations", n_max_iterations),
            ("rk45_dT_min", rk45_dT_min),
            ("rk45_niter_max", rk45_niter_max),
            ("return_to_yield_surface", return_to_yield_surface),
            ("integration_method", integration_method),
            ("tangent_type", tangent_type),
        ),
    )


# ---------------------------------------------------------------------------
# PlaneStrain — wraps a 3-D nDMaterial as a 2-D plane-strain material
# ---------------------------------------------------------------------------
#
# OpenSees command::
#
#     nDMaterial PlaneStrain $tag $base3d_tag
#
# Required wrapping for the SSI rock case: ASDPlasticMaterial3D is
# strictly 3D — passing its tag directly to ``element quad ... PlaneStrain
# $matTag`` triggers ``ASDPlasticMaterial3D::getCopy("PlaneStrain") --
# Only 3D is currently supported.``  The PlaneStrain wrapper bridges
# the 2D constitutive interface the quad element expects.


@dataclass(frozen=True, kw_only=True, slots=True)
class PlaneStrain(NDMaterial):
    """``nDMaterial PlaneStrain`` — 2-D plane-strain wrapper around a 3-D material.

    Tcl signature::

        nDMaterial PlaneStrain $tag $base3d_tag

    Parameters
    ----------
    base
        The 3-D :class:`NDMaterial` (e.g. :class:`ASDPlasticMaterial3D`)
        that supplies the 3-D constitutive law.  The wrapper exposes
        a 2-D plane-strain view by constraining ε_zz = 0 and projecting
        the stress to the in-plane components.

    Notes
    -----
    Use this whenever an apeGmsh 2-D element (``FourNodeQuad``,
    ``Tri31``) needs to consume a strictly-3-D material.  For natively
    2-D materials (``ElasticIsotropic``), the quad's ``plane_type=``
    argument selects the 2-D view directly and no wrapping is needed.
    """

    base: NDMaterial

    def _emit(self, emitter: Emitter, tag: int) -> None:
        base_tag = resolve_tag(emitter, self.base)
        emitter.nDMaterial("PlaneStrain", tag, base_tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.base,)


# ---------------------------------------------------------------------------
# ASDConcrete3D — Petracca plastic-damage, crack-band-regularized concrete
# ---------------------------------------------------------------------------
#
# See ADR 0044 (asdconcrete-regularization-contract). Two facts drive the
# design, both source-verified against OpenSees 7c92197:
#
#   * Regularization is per-ELEMENT (one material clone per Gauss point,
#     Brick.cpp:190-197); a tag shared across a graded mesh self-regularizes
#     correctly per element.
#   * The native ``-fc`` command CANNOT take a user fracture energy (no
#     ``-Gf``/``-Gc`` token; it derives them from ``fc`` via CEB-FIP). To honour
#     a user-supplied ``Gf``/``Gc`` — the physical regularization input — apeGmsh
#     OWNS the backbone (see :mod:`._asdconcrete_laws`) and emits the explicit
#     ``-Te/-Ts/-Td/-Ce/-Cs/-Cd`` points. The solver integrates exactly those
#     points, so there is no parity-drift surface.
#
# ``-autoRegularization`` requires an explicit ``$lch_ref`` value (the bare flag
# is a parser error); the curve and the emitted ``lch_ref`` share one reference
# length so ``area*lch_ref == Gf`` per element.


class ASDRegularizationWarning(UserWarning):
    """Raised (as a warning) when an element exceeds the crack-band ceiling.

    Subclass of :class:`UserWarning` so it can be silenced per-call or
    promoted to an error in CI via
    ``pytest -W error::...ASDRegularizationWarning`` — the warn-as-contract
    idiom (cf. ``ComposeInterfaceSizeWarning``). Over-ceiling elements yield
    an over-brittle, mesh-dependent response (the binary floors the fracture
    energy); the model is still well-formed, so this never blocks emit.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class ASDConcrete3D(NDMaterial):
    """``nDMaterial ASDConcrete3D`` — Petracca plastic-damage concrete.

    Prefer the :meth:`from_fc` constructor (physical inputs ``fc, ft, Gf,
    Gc``); the raw constructor takes pre-built backbones for
    test-calibrated or Mander-confined curves.

    Parameters
    ----------
    E, v, rho
        Young's modulus (``>0``), Poisson's ratio (``[0, 0.5)``), density.
    Te, Ts, Td / Ce, Cs, Cd
        Tension / compression backbone points: total strain, nominal
        stress, damage ``d in [0, 1)``. The three lists in each triple
        must share length (``>= 2``) and start at the origin.
    lch_ref
        Reference band width (``>0``) the backbone's fracture energy is
        calibrated to; emitted to ``-autoRegularization``. The physics is
        invariant to its value (ADR 0044), but it must be supplied — the
        bare flag is a parser error.
    Kc
        Lubliner triaxial shape ratio, ``[2/3, 1]`` (confinement
        sensitivity; default ``2/3``).
    eta, cdf, implex
        Rate-dependent viscosity, tension/compression cross-damage factor,
        IMPL-EX integration flag.
    auto_regularize
        Emit ``-autoRegularization $lch_ref`` (default ``True``). Disable
        only to deliberately opt out of mesh regularization.
    ft, Gf
        Provenance from :meth:`from_fc` (tensile strength, tensile fracture
        energy per area) — used by :meth:`l_max` / :meth:`check_element_size`.
        ``None`` for raw-curve construction (then :meth:`l_max` returns
        ``None``).

    Notes
    -----
    3-D only — for a 2-D/shell element wrap in :class:`PlaneStrain`. The
    1-D sibling (fibers) is confinement-blind; bake Mander into its
    backbone yourself (ADR 0044, deferred ``ConfinedConcrete`` helper).
    """

    E: float
    v: float
    Te: tuple[float, ...]
    Ts: tuple[float, ...]
    Td: tuple[float, ...]
    Ce: tuple[float, ...]
    Cs: tuple[float, ...]
    Cd: tuple[float, ...]
    lch_ref: float
    rho: float = 0.0
    Kc: float = 2.0 / 3.0
    eta: float = 0.0
    cdf: float = 0.0
    implex: bool = False
    auto_regularize: bool = True
    ft: float | None = None
    Gf: float | None = None

    @classmethod
    def from_fc(
        cls, *,
        E: float,
        v: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        rho: float = 0.0,
        Kc: float = 2.0 / 3.0,
        eta: float = 0.0,
        cdf: float = 0.0,
        implex: bool = False,
    ) -> "ASDConcrete3D":
        """Build from physical inputs, generating the backbone in Python.

        ``ft`` defaults to ``0.1*fc``; ``Gf`` (tensile) and ``Gc``
        (compressive) fracture energies per area default to the CEB-FIP
        correlations (``Gf = 0.073 fc^0.18``, ``Gc = 2 Gf (fc/ft)^2``).
        ``lch_ref`` defaults to the native self-derived ``min(hmin_t,
        hmin_c)``; pass a representative element size for better-conditioned
        softening (ADR 0044).
        """
        if E <= 0:
            raise ValueError(f"ASDConcrete3D.from_fc: E must be > 0, got {E!r}")
        if fc <= 0:
            raise ValueError(f"ASDConcrete3D.from_fc: fc must be > 0, got {fc!r}")
        for label, val in (("ft", ft), ("Gf", Gf), ("Gc", Gc),
                           ("lch_ref", lch_ref)):
            if val is not None and val <= 0:
                raise ValueError(
                    f"ASDConcrete3D.from_fc: {label} must be > 0 if supplied, "
                    f"got {val!r}"
                )
        ft_ = ft if ft is not None else _laws.default_ft(fc)
        Gf_ = Gf if Gf is not None else _laws.ceb_fip_Gf(fc)
        Gc_ = Gc if Gc is not None else _laws.ceb_fip_Gc(fc, ft_, Gf_)
        lch = lch_ref if lch_ref is not None else _laws.auto_lch_ref(
            E, fc, ft_, Gf_, Gc_)
        Te, Ts, Td = _laws.make_tension(E, ft_, Gf_, lch)
        Ce, Cs, Cd = _laws.make_compression(E, fc, Gc_, lch)
        return cls(
            E=E, v=v,
            Te=tuple(Te), Ts=tuple(Ts), Td=tuple(Td),
            Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd),
            lch_ref=lch, rho=rho, Kc=Kc, eta=eta, cdf=cdf, implex=implex,
            ft=ft_, Gf=Gf_,
        )

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"ASDConcrete3D: E must be > 0, got {self.E!r}")
        if not (0.0 <= self.v < 0.5):
            raise ValueError(
                f"ASDConcrete3D: v must be in [0, 0.5), got {self.v!r}"
            )
        if self.lch_ref <= 0:
            raise ValueError(
                f"ASDConcrete3D: lch_ref must be > 0, got {self.lch_ref!r}"
            )
        if not (2.0 / 3.0 <= self.Kc <= 1.0):
            raise ValueError(
                f"ASDConcrete3D: Kc must be in [2/3, 1], got {self.Kc!r}"
            )
        for label, val in (("rho", self.rho), ("eta", self.eta),
                           ("cdf", self.cdf)):
            if val < 0:
                raise ValueError(
                    f"ASDConcrete3D: {label} must be >= 0, got {val!r}"
                )
        for side, (e, s, d) in (("tension", (self.Te, self.Ts, self.Td)),
                                ("compression", (self.Ce, self.Cs, self.Cd))):
            if not (len(e) == len(s) == len(d)):
                raise ValueError(
                    f"ASDConcrete3D: {side} backbone lists must share length, "
                    f"got {len(e)}/{len(s)}/{len(d)}"
                )
            if len(e) < 2:
                raise ValueError(
                    f"ASDConcrete3D: {side} backbone needs >= 2 points, "
                    f"got {len(e)}"
                )
        for dmg in (*self.Td, *self.Cd):
            if not (0.0 <= dmg < 1.0):
                raise ValueError(
                    f"ASDConcrete3D: damage must be in [0, 1), got {dmg!r}"
                )

    def preview_backbone(self) -> dict[str, tuple[float, ...] | float]:
        """The exact backbone that will be emitted (read-only, for plotting)."""
        return {
            "Te": self.Te, "Ts": self.Ts, "Td": self.Td,
            "Ce": self.Ce, "Cs": self.Cs, "Cd": self.Cd,
            "lch_ref": self.lch_ref,
        }

    def l_max(self) -> float | None:
        """Crack-band snapback ceiling ``2*E*Gf/ft^2``, or ``None`` if ``Gf``/``ft`` unknown."""
        if self.ft is None or self.Gf is None:
            return None
        return _laws.l_max(self.E, self.Gf, self.ft)

    def check_element_size(self, lch: float, *, pg: str | None = None) -> bool:
        """Warn (never raise) if ``lch`` exceeds :meth:`l_max`; return ``True`` if OK.

        Intended to be called per-element at bind/emit time once realized
        geometry is available (ADR 0044, Decision 5). Returns ``True`` when
        no ceiling is known or the element is within it.
        """
        lm = self.l_max()
        if lm is not None and lch > lm:
            where = f", PG {pg!r}" if pg is not None else ""
            warnings.warn(
                f"ASDConcrete3D: element size lch={lch:g} exceeds the "
                f"crack-band snapback ceiling l_max=2*E*Gf/ft^2={lm:g} "
                f"(ratio {lch / lm:.2f}{where}). The softening fracture energy "
                f"will be floored and the response is no longer mesh-objective; "
                f"refine the mesh or increase Gf.",
                ASDRegularizationWarning,
                stacklevel=2,
            )
            return False
        return True

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | int | str] = [
            self.E, self.v,
            "-Te", *self.Te, "-Ts", *self.Ts, "-Td", *self.Td,
            "-Ce", *self.Ce, "-Cs", *self.Cs, "-Cd", *self.Cd,
            "-rho", self.rho, "-Kc", self.Kc,
        ]
        if self.eta:
            args += ["-eta", self.eta]
        if self.cdf:
            args += ["-cdf", self.cdf]
        if self.implex:
            args.append("-implex")
        if self.auto_regularize:
            args += ["-autoRegularization", self.lch_ref]
        emitter.nDMaterial("ASDConcrete3D", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoJ2 — combined-hardening (Voce + Chaboche) von Mises (Ladruno fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoJ2(NDMaterial):
    r"""``nDMaterial LadrunoJ2`` — combined-hardening von Mises (Ladruno fork).

    OpenSees command (Ladruno fork, ``ND_TAG`` **33011**)::

        nDMaterial LadrunoJ2 tag K G \
            -iso voce sig0 Qinf b Hiso \
            [-kin N C1 g1 C2 g2 ...] \
            [-damage lemaitre r s pD Dc] \
            [-rho rho] [-autoRegularization lch_ref] [-implex]

    The fork's flagship rate-independent von Mises ``nDMaterial`` unifying
    nonlinear **isotropic** (Voce + linear) and nonlinear **kinematic**
    (Chaboche / Armstrong-Frederick) hardening — the OpenSees analogue of
    Abaqus ``*PLASTIC, COMBINED``. One class serves all five dimensional
    views (3D / PlaneStrain / AxiSymm / PlateFiber / PlaneStress).

    .. note::
       Fork-only. Emission produces a deck line on any build; the material
       is unavailable on stock ``openseespy`` and bites only at
       ``ops.run()`` (a "requires the Ladruno fork build" error).

    Parameters
    ----------
    K, G
        Bulk and shear moduli (both must be > 0).
    sig0
        Initial yield stress (Voce ``sigma_0``). Must be > 0.
    Qinf, b, Hiso
        Voce saturation stress, saturation rate (``>= 0``), and linear
        isotropic hardening modulus. All default ``0.0`` (perfectly
        plastic when also no kinematic hardening).
    backstresses
        Chaboche kinematic backstress pairs ``[(C1, gamma1), ...]`` — at
        most 8 (the fork ``MAXBACK``). Each ``C_k > 0``, ``gamma_k >= 0``.
        Empty (default) emits no ``-kin`` (pure isotropic / ``J2Plasticity``
        limit).
    rho
        Mass density (``-rho``; ``>= 0``). Emitted only when nonzero.
    lch_ref
        Characteristic-length reference for mesh-objective damage
        regularization (``-autoRegularization``; must be > 0 if supplied).
        Only meaningful together with ``damage``.
    damage
        Optional Lemaitre ductile-damage parameters ``(r, s, pD, Dc)``
        (``-damage lemaitre``). The fork requires ``r > 0`` and
        ``0 < Dc <= 1``. ``None`` (default) = no damage (byte-identical to
        the undamaged material).
    implex
        Emit ``-implex`` for the IMPL-EX (extrapolated) integration — an
        SPD tangent for explicit / softening robustness.
    """

    K: float
    G: float
    sig0: float
    Qinf: float = 0.0
    b: float = 0.0
    Hiso: float = 0.0
    backstresses: tuple[tuple[float, float], ...] = ()
    rho: float = 0.0
    lch_ref: float | None = None
    damage: tuple[float, float, float, float] | None = None
    implex: bool = False

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"LadrunoJ2: K must be > 0, got {self.K!r}")
        if self.G <= 0:
            raise ValueError(f"LadrunoJ2: G must be > 0, got {self.G!r}")
        _lj2.validate_iso("LadrunoJ2", self.sig0, self.Qinf, self.b, self.Hiso)
        _lj2.validate_backstresses("LadrunoJ2", self.backstresses)
        if self.rho < 0:
            raise ValueError(f"LadrunoJ2: rho must be >= 0, got {self.rho!r}")
        if self.lch_ref is not None and self.lch_ref <= 0:
            raise ValueError(
                f"LadrunoJ2: lch_ref must be > 0 if supplied, got "
                f"{self.lch_ref!r}"
            )
        if self.damage is not None:
            _lj2.validate_lemaitre("LadrunoJ2", self.damage)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | int | str] = [self.K, self.G]
        args += _lj2.iso_args(self.sig0, self.Qinf, self.b, self.Hiso)
        args += _lj2.kin_args(self.backstresses)
        if self.rho:
            args += ["-rho", self.rho]
        if self.lch_ref is not None:
            args += ["-autoRegularization", self.lch_ref]
        if self.damage is not None:
            args += _lj2.lemaitre_args(self.damage)
        if self.implex:
            args.append("-implex")
        emitter.nDMaterial("LadrunoJ2", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoJ2Finite — finite-strain-native combined J2 (Ladruno fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoJ2Finite(NDMaterial):
    r"""``nDMaterial LadrunoJ2Finite`` — finite-strain-native combined J2.

    OpenSees command (Ladruno fork, ``ND_TAG`` **33012**)::

        nDMaterial LadrunoJ2Finite tag K G \
            -iso voce sig0 Qinf b Hiso \
            [-kin N C1 g1 ...] [-rho rho] [-implex]

    A ``FiniteStrainNDMaterial`` that does combined-hardening J2 at finite
    strain **natively** (co-rotating the backstress each step). Use it when
    you need **combined (kinematic) hardening AND large rotation** — finite
    cyclic / buckling-brace loops. For *isotropic* hardening at finite
    strain the wrapper path ``LogStrain(LadrunoJ2 -kin 0)`` is already exact
    and simpler. 3-D only; the sole consumer is
    ``LadrunoBrick ... -geom finite`` (the F-interface).

    Unlike :class:`LadrunoJ2`, the finite-strain material has **no**
    ``-damage`` and **no** ``-autoRegularization`` flags (the fork parser
    rejects them here).

    .. note::
       Fork-only. Emission works on any build; the material errors at
       ``ops.run()`` on stock ``openseespy``.

    Parameters
    ----------
    K, G
        Bulk and shear moduli (both > 0).
    sig0
        Initial yield stress (> 0).
    Qinf, b, Hiso
        Voce saturation stress, saturation rate (``>= 0``), linear
        isotropic hardening modulus (default ``0.0``).
    backstresses
        Chaboche backstress pairs ``[(C, gamma), ...]`` — at most 8.
    rho
        Mass density (``-rho``; ``>= 0``). Emitted only when nonzero.
    implex
        Emit ``-implex`` (constant SPD elastic tangent for explicit /
        quasi-static use).
    """

    is_finite_strain: ClassVar[bool] = True

    K: float
    G: float
    sig0: float
    Qinf: float = 0.0
    b: float = 0.0
    Hiso: float = 0.0
    backstresses: tuple[tuple[float, float], ...] = ()
    rho: float = 0.0
    implex: bool = False

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"LadrunoJ2Finite: K must be > 0, got {self.K!r}")
        if self.G <= 0:
            raise ValueError(f"LadrunoJ2Finite: G must be > 0, got {self.G!r}")
        _lj2.validate_iso(
            "LadrunoJ2Finite", self.sig0, self.Qinf, self.b, self.Hiso
        )
        _lj2.validate_backstresses("LadrunoJ2Finite", self.backstresses)
        if self.rho < 0:
            raise ValueError(
                f"LadrunoJ2Finite: rho must be >= 0, got {self.rho!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | int | str] = [self.K, self.G]
        args += _lj2.iso_args(self.sig0, self.Qinf, self.b, self.Hiso)
        args += _lj2.kin_args(self.backstresses)
        if self.rho:
            args += ["-rho", self.rho]
        if self.implex:
            args.append("-implex")
        emitter.nDMaterial("LadrunoJ2Finite", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LogStrain — Hencky finite-strain lift wrapper (Ladruno fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class LogStrain(NDMaterial):
    r"""``nDMaterial LogStrain`` — Hencky finite-strain lift (Ladruno fork).

    OpenSees command (Ladruno fork, ``ND_TAG`` **33010**)::

        nDMaterial LogStrain tag innerTag

    The material-side adaptor that lifts an *unchanged* small-strain 3-D
    ``nDMaterial`` to a genuine finite-strain (large rotation + large
    strain) material by the logarithmic (Hencky) strain-space technique
    (de Souza Neto Box 14.3). The inner return map is reused **verbatim** —
    the wrapper does the spectral pre/post-processing and returns the
    constitutive spatial tangent; the element owns the geometric stiffness.
    The result is a ``FiniteStrainNDMaterial`` (driven by ``setTrialF``),
    consumable by ``LadrunoBrick ... -geom finite``.

    Exact and objective only for the **isotropic** spine: pair it with an
    isotropic inner (e.g. ``LadrunoJ2(-kin 0)``, ``ElasticIsotropic``,
    ``DruckerPrager``). For combined (kinematic) hardening at finite strain
    use the native :class:`LadrunoJ2Finite` instead (the backstress doesn't
    co-rotate through the wrapper — dSNPO §14.11).

    .. note::
       Fork-only. The inner must yield a 3-D (order-6) copy — the fork
       parser rejects a non-3-D inner. Emission works on any build; errors
       at ``ops.run()`` on stock ``openseespy``.

    Parameters
    ----------
    inner
        The wrapped small-strain 3-D :class:`NDMaterial`. Held by reference;
        its tag is resolved at emit time and the bridge emits it **before**
        the wrapper (via :meth:`dependencies`).
    """

    is_finite_strain: ClassVar[bool] = True

    inner: NDMaterial

    def __post_init__(self) -> None:
        if not isinstance(self.inner, NDMaterial):
            raise TypeError(
                "LogStrain: inner must be an NDMaterial primitive, got "
                f"{type(self.inner).__name__!r}."
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        inner_tag = resolve_tag(emitter, self.inner)
        emitter.nDMaterial("LogStrain", tag, inner_tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.inner,)


# ---------------------------------------------------------------------------
# InitDefGrad — finite staged stress-free birth wrapper (Ladruno fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class InitDefGrad(NDMaterial):
    r"""``nDMaterial InitDefGrad`` — finite staged stress-free birth.

    OpenSees command (Ladruno fork, ``ND_TAG`` **33013**)::

        nDMaterial InitDefGrad tag innerTag [-noInitF] \
            [-F0 f11 f12 f13 f21 f22 f23 f31 f32 f33]

    A ``FiniteStrainNDMaterial`` wrapper that makes a continuum element
    **born stress-free at the current deformed geometry** in a staged
    analysis (a new member, a concrete lift, a backfill layer). It captures
    the per-Gauss-point birth deformation gradient ``F0`` on the first
    ``setTrialF`` and feeds the inner the relative gradient
    ``F_rel = F · F0^-1`` (objective by construction). The inner **must**
    itself be a finite-strain material (e.g. :class:`LogStrain` or
    :class:`LadrunoJ2Finite`). Also registered as ``StagedDefGrad``.

    .. note::
       Fork-only. The fork parser rejects a non-``FiniteStrainNDMaterial``
       inner. Emission works on any build; errors at ``ops.run()`` on stock
       ``openseespy``. A supplied singular ``F0`` (``det = 0``) aborts the
       fork at construction.

    Parameters
    ----------
    inner
        The wrapped finite-strain :class:`NDMaterial` (``LogStrain`` /
        ``LadrunoJ2Finite``). Emitted before the wrapper.
    no_init_f
        Emit ``-noInitF`` to opt out of birth capture (the wrapper then
        behaves as the bare inner). Defaults to ``False``.
    F0
        Optional known birth deformation gradient as **9 row-major**
        components ``(F11, F12, F13, F21, F22, F23, F31, F32, F33)``
        (``-F0``). Omit (default) for auto-capture at birth.
    """

    is_finite_strain: ClassVar[bool] = True

    inner: NDMaterial
    no_init_f: bool = False
    F0: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inner, NDMaterial):
            raise TypeError(
                "InitDefGrad: inner must be an NDMaterial primitive, got "
                f"{type(self.inner).__name__!r}."
            )
        if self.F0 is not None and len(self.F0) != 9:
            raise ValueError(
                "InitDefGrad: F0 must have 9 row-major components "
                f"(F11..F33), got {len(self.F0)}."
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        inner_tag = resolve_tag(emitter, self.inner)
        args: list[float | int | str] = [inner_tag]
        if self.no_init_f:
            args.append("-noInitF")
        if self.F0 is not None:
            args += ["-F0", *self.F0]
        emitter.nDMaterial("InitDefGrad", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.inner,)


# ---------------------------------------------------------------------------
# StagedStrain — small-strain staged stress-free birth wrapper (Ladruno fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class StagedStrain(NDMaterial):
    r"""``nDMaterial StagedStrain`` — small-strain staged stress-free birth.

    OpenSees command (Ladruno fork, ``ND_TAG`` **33014**)::

        nDMaterial StagedStrain tag innerTag [-noInit] [-eps0 e1 ... e6]

    The **small-strain** member of the ``Staged*`` family (the additive
    analog of :class:`InitDefGrad`). Captures the birth strain ``eps0`` at
    the first ``setTrialStrain`` and feeds the inner
    ``eps_rel = eps - eps0``, so at birth the element is **genuinely
    virgin** (zero stress *and* zero plastic history). The everyday
    staged-build case in 2-D or 3-D. The inner may be any 3-D-capable
    ``nDMaterial`` (the fork coerces it to a 3-D view).

    .. note::
       Fork-only. ``eps0`` is read **greedily** by the parser (all remaining
       tokens) and must match the inner's 3-D order (6 Voigt components),
       else the fork silently discards it and falls back to auto-capture —
       so apeGmsh requires exactly 6 components. Emission works on any
       build; errors at ``ops.run()`` on stock ``openseespy``.

    Parameters
    ----------
    inner
        The wrapped 3-D-capable :class:`NDMaterial`. Emitted before the
        wrapper.
    no_init
        Emit ``-noInit`` to opt out of birth capture. Defaults to ``False``.
    eps0
        Optional known birth strain as **6 Voigt components**
        ``(eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx)``
        (``-eps0``). Omit (default) for auto-capture at birth.
    """

    inner: NDMaterial
    no_init: bool = False
    eps0: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inner, NDMaterial):
            raise TypeError(
                "StagedStrain: inner must be an NDMaterial primitive, got "
                f"{type(self.inner).__name__!r}."
            )
        if self.eps0 is not None and len(self.eps0) != 6:
            raise ValueError(
                "StagedStrain: eps0 must have 6 Voigt components (matching "
                f"the inner's 3-D order), got {len(self.eps0)}."
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        inner_tag = resolve_tag(emitter, self.inner)
        args: list[float | int | str] = [inner_tag]
        if self.no_init:
            args.append("-noInit")
        # -eps0 is greedy on the parser side: it must be the LAST flag.
        if self.eps0 is not None:
            args += ["-eps0", *self.eps0]
        emitter.nDMaterial("StagedStrain", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.inner,)


# ---------------------------------------------------------------------------
# LadrunoConcrete3D — CDPM2-grade solid plastic-damage concrete (Ladruno fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoConcrete3D(NDMaterial):
    r"""``nDMaterial LadrunoConcrete3D`` — CDPM2-grade plastic-damage concrete.

    OpenSees command (Ladruno fork, ``ND_TAG`` **33017**)::

        nDMaterial LadrunoConcrete3D tag E nu fc ft Gf Gc \
            [-e e | -kupfer fcc/fc] [-Df Df] [-As As] [-rho rho] \
            [-hardening qh0 Hp] [-ductility Ah Bh Ch Dh] [-lch lch] \
            [-autoRegularization] [-implex] [-eta eta] \
            [-ctTemper none|alphat|proj] [-hoop K [-hoopFy fy]]

    The fork's flagship solid-concrete material: a CDPM2-grade isotropic
    plastic-damage model with a single Lubliner/Lee-Fenves yield surface,
    separate tension/compression damage, fracture-energy regularization
    (``Gf``/``Gc``), and an optional IMPL-EX integration. 3-D / plane /
    BeamFiber views are served from one class.

    .. warning::
       The consistent tangent is **non-symmetric** (non-associated flow);
       drive it with an unsymmetric solver (``system UmfPack`` or
       ``system FullGeneral``). ``-implex`` gives a symmetric-part-SPD
       secant on single-sign states but an unsymmetric solver is still the
       safe default.

    .. note::
       Fork-only. Emission produces a deck line on any build; the material
       is unavailable on stock ``openseespy`` and bites only at
       ``ops.run()``.

    Parameters
    ----------
    E, nu, rho
        Young's modulus (``> 0``), Poisson's ratio (``[0, 0.5)``), density
        (``>= 0``; ``-rho``, emitted only when nonzero).
    fc, ft
        Uniaxial compressive and tensile strengths as **positive
        magnitudes** (both ``> 0``; the fork requires ``ft < fc``).
    Gf, Gc
        Tensile and compressive fracture energies per unit area (both
        ``> 0``).
    e, kupfer
        Yield-surface eccentricity. Supply ``e`` directly (``-e``; must be
        in ``(0.5, 1]``) **or** let it derive from the biaxial/uniaxial
        strength ratio ``kupfer`` (``-kupfer``; ``> 1``, default 1.16).
        Supplying both an explicit ``e`` and a non-default ``kupfer`` is a
        construction error.
    Df
        Dilatancy / flow-shape factor (``-Df``; ``> 0``, default 1.0).
    As
        Compression-ductility amplitude (``-As``; ``>= 1``, default 2.0).
    hardening
        Pre-peak hardening ``(qh0, Hp)`` (``-hardening``; default
        ``(0.3, 0.5)``).
    ductility
        Compression post-peak ductility coefficients
        ``(Ah, Bh, Ch, Dh)`` (``-ductility``; default
        ``(0.08, 0.003, 2.0, 1e-6)``).
    lch
        Fixed characteristic length used when ``auto_regularize`` is off
        (``-lch``; ``> 0``, default 1.0).
    auto_regularize
        Emit the bare ``-autoRegularization`` flag so each element scales
        its softening to its own size (default ``False``).
    implex
        Emit ``-implex`` for the IMPL-EX (extrapolated) integration.
    eta
        Duvaut-Lions viscoplastic relaxation time (``-eta``; ``>= 0``,
        TIME units; needs a positive time increment to bite). Default 0.
    ct_temper
        Compression->tension damage-coupling temper, one of ``"none"``
        (literal CDPM2, default), ``"alphat"``, ``"proj"`` (``-ctTemper``).
    hoop_k, hoop_fy
        Passive transverse-hoop confining stiffness ``K`` (``-hoop``;
        ``>= 0``) and its yield ``fy`` (``-hoopFy``; ``> 0``). Active ONLY
        through the ``BeamFiber`` view (e.g. ``NDFiberSection3d``); inert
        for solid 3-D / plane views.
    """

    _CT_TEMPER: ClassVar[frozenset[str]] = frozenset({"none", "alphat", "proj"})

    E: float
    nu: float
    fc: float
    ft: float
    Gf: float
    Gc: float
    e: float | None = None
    kupfer: float = 1.16
    Df: float = 1.0
    As: float = 2.0
    rho: float = 0.0
    hardening: tuple[float, float] = (0.3, 0.5)
    ductility: tuple[float, float, float, float] = (0.08, 0.003, 2.0, 1.0e-6)
    lch: float = 1.0
    auto_regularize: bool = False
    implex: bool = False
    eta: float = 0.0
    ct_temper: str = "none"
    hoop_k: float = 0.0
    hoop_fy: float = 1.0e30

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"LadrunoConcrete3D: E must be > 0, got {self.E!r}")
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(
                f"LadrunoConcrete3D: nu must be in [0, 0.5), got {self.nu!r}"
            )
        if self.fc <= 0 or self.ft <= 0:
            raise ValueError(
                "LadrunoConcrete3D: fc, ft must be > 0 (positive magnitudes), "
                f"got fc={self.fc!r}, ft={self.ft!r}"
            )
        if self.ft >= self.fc:
            raise ValueError(
                f"LadrunoConcrete3D: need ft < fc, got ft={self.ft!r}, "
                f"fc={self.fc!r}"
            )
        if self.Gf <= 0 or self.Gc <= 0:
            raise ValueError(
                f"LadrunoConcrete3D: Gf, Gc must be > 0, got Gf={self.Gf!r}, "
                f"Gc={self.Gc!r}"
            )
        if self.Df <= 0:
            raise ValueError(f"LadrunoConcrete3D: Df must be > 0, got {self.Df!r}")
        if self.As < 1.0:
            raise ValueError(
                f"LadrunoConcrete3D: As must be >= 1, got {self.As!r}"
            )
        if self.e is not None:
            if not (0.5 < self.e <= 1.0):
                raise ValueError(
                    f"LadrunoConcrete3D: e must be in (0.5, 1], got {self.e!r}"
                )
            if self.kupfer != 1.16:
                raise ValueError(
                    "LadrunoConcrete3D: supply either e or a non-default "
                    "kupfer, not both (the fork's -e overrides -kupfer)."
                )
        elif self.kupfer <= 1.0:
            raise ValueError(
                f"LadrunoConcrete3D: kupfer (fcc/fc) must be > 1, got "
                f"{self.kupfer!r}"
            )
        if self.rho < 0:
            raise ValueError(
                f"LadrunoConcrete3D: rho must be >= 0, got {self.rho!r}"
            )
        if self.lch <= 0:
            raise ValueError(
                f"LadrunoConcrete3D: lch must be > 0, got {self.lch!r}"
            )
        if self.eta < 0:
            raise ValueError(
                f"LadrunoConcrete3D: eta must be >= 0, got {self.eta!r}"
            )
        if self.ct_temper not in self._CT_TEMPER:
            raise ValueError(
                "LadrunoConcrete3D: ct_temper must be one of "
                f"{sorted(self._CT_TEMPER)}, got {self.ct_temper!r}"
            )
        if self.hoop_k < 0:
            raise ValueError(
                f"LadrunoConcrete3D: hoop_k must be >= 0, got {self.hoop_k!r}"
            )
        if self.hoop_fy <= 0:
            raise ValueError(
                f"LadrunoConcrete3D: hoop_fy must be > 0, got {self.hoop_fy!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | int | str] = [
            self.E, self.nu, self.fc, self.ft, self.Gf, self.Gc
        ]
        if self.e is not None:
            args += ["-e", self.e]
        elif self.kupfer != 1.16:
            args += ["-kupfer", self.kupfer]
        if self.Df != 1.0:
            args += ["-Df", self.Df]
        if self.As != 2.0:
            args += ["-As", self.As]
        if self.rho:
            args += ["-rho", self.rho]
        if self.hardening != (0.3, 0.5):
            args += ["-hardening", *self.hardening]
        if self.ductility != (0.08, 0.003, 2.0, 1.0e-6):
            args += ["-ductility", *self.ductility]
        if self.lch != 1.0:
            args += ["-lch", self.lch]
        if self.auto_regularize:
            args.append("-autoRegularization")
        if self.implex:
            args.append("-implex")
        if self.eta:
            args += ["-eta", self.eta]
        if self.ct_temper != "none":
            args += ["-ctTemper", self.ct_temper]
        if self.hoop_k:
            args += ["-hoop", self.hoop_k]
            if self.hoop_fy != 1.0e30:
                args += ["-hoopFy", self.hoop_fy]
        emitter.nDMaterial("LadrunoConcrete3D", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoRCConcrete / LadrunoRCFiniteStrain — RC plastic-damage + MCFT
# ---------------------------------------------------------------------------
#
# The two RC materials share ONE command grammar verified against the fork
# parsers ``OPS_LadrunoRCConcrete`` / ``OPS_LadrunoRCFiniteStrain`` (the
# finite-strain twin is the Hencky view of the same plastic-damage law). The
# grammar is centralized in the ``_LadrunoRC`` base so the two never drift —
# the only differences are the emitted command token (``_type``) and the
# ``is_finite_strain`` flag.
#
# Backbones are the same total-strain / nominal-stress / damage triples the
# fork builds via the ASDConcrete3D HardeningLaw c-tor, so the existing
# :mod:`._asdconcrete_laws` generator drives the :meth:`from_fc` convenience.

_TANGENT_MODES: dict[str, str | None] = {
    "consistent": None,
    "secant": "-secant",
    "numerical": "-numericalTangent",
}
_SHEAR_RETENTION: frozenset[str] = frozenset({"mcft", "const", "dsfm", "rots"})
_TENS_STIFF: frozenset[str] = frozenset({"off", "vc", "cm"})


@dataclass(frozen=True, kw_only=True, slots=True)
class _LadrunoRC(NDMaterial):
    r"""Shared base for the Ladruno-fork RC plastic-damage materials.

    Holds the full command grammar (backbones + the MCFT
    aggregate-interlock / tension-stiffening / IMPL-EX option set) common to
    :class:`LadrunoRCConcrete` and :class:`LadrunoRCFiniteStrain`. Concrete
    subclasses set the command token via ``_type`` only. Not exported / not
    instantiated directly.

    Parameters
    ----------
    E, nu, rho
        Young's modulus (``> 0``), Poisson's ratio (``[0, 0.5)``), density
        (``-rho``; ``>= 0``, emitted only when nonzero).
    Ce, Cs, Cd / Te, Ts, Td
        Compression / tension backbone points: total strain, nominal
        stress, damage ``d in [0, 1)``. ``Ce``/``Cs`` (and ``Te``/``Ts``)
        are required, equal length ``>= 2``; the damage lists ``Cd``/``Td``
        are optional (the fork pads them with zeros) — supply empty
        (default) or a list matching the strain/stress length.
    Kc
        Lubliner triaxial shape ratio ``[2/3, 1]`` (``-Kc``; default 2/3).
    beta
        Emit ``-beta`` (Lubliner dilatancy term).
    beta_floor
        Lower bound on the biaxial reduction factor (``-betaFloor``;
        default 0.1).
    lubliner_reduced
        Emit ``-lublinerReduced`` (reduced tension/compression coupling).
    tangent
        Tangent operator: ``"consistent"`` (default), ``"secant"``
        (``-secant``) or ``"numerical"`` (``-numericalTangent``).
    interlock, cyclic, xcrack
        Aggregate-interlock shear-retention toggles (``-interlock`` /
        ``-cyclic`` / ``-xcrack``). The fork implies the weaker flags from
        the stronger ones; emitted here verbatim as set.
    agg, crack_strain, crack_spacing, lch, beta_sr_min
        Interlock geometry / state inputs (``-agg`` 16.0, ``-crackStrain``
        0, ``-crackSpacing`` 0, ``-lch`` 0, ``-betaSrMin`` 0.01).
    shear_retention, shear_ret_factor
        Crack-shear retention curve ``{"mcft" (default), "const", "dsfm",
        "rots"}`` (``-shearRetention``) and the ``const``-mode retention
        factor (``-shearRetFactor``; default 0.4).
    deg_kappa, deg_slip_ref, deg_min
        Slip-driven interlock-wear law (``-degKappa`` 0.5, ``-degSlipRef``
        0.01, ``-degMin`` 0.1; only meaningful under ``xcrack``).
    implex, implex_alpha, implex_control
        IMPL-EX integration: ``-implex`` flag, extrapolation factor
        (``-implexAlpha``; default 1.0), and the adaptive control
        ``(err_tol, time_red_lim)`` (``-implexControl``; ``None`` = off).
    tens_stiff, tens_stiff_c, tens_stiff_alpha
        Tension stiffening ``{"off" (default), "vc" (Bentz),
        "cm" (Collins-Mitchell)}`` (``-tensStiff``) with its coefficient
        (``-tensStiffC``; ``> 0`` in ``vc`` mode, default 500) and exponent
        (``-tensStiffAlpha``; default 1.0).
    auto_regularization
        Crack-band (Bazant-Oh) reference length (``-autoRegularization
        $lch_ref``; ``> 0``). ``None`` (default) = off / baseline-identical.
    """

    _type: ClassVar[str] = ""

    E: float
    nu: float
    Ce: tuple[float, ...]
    Cs: tuple[float, ...]
    Te: tuple[float, ...]
    Ts: tuple[float, ...]
    Cd: tuple[float, ...] = ()
    Td: tuple[float, ...] = ()
    rho: float = 0.0
    Kc: float = 2.0 / 3.0
    beta: bool = False
    beta_floor: float = 0.1
    lubliner_reduced: bool = False
    tangent: str = "consistent"
    interlock: bool = False
    cyclic: bool = False
    xcrack: bool = False
    agg: float = 16.0
    crack_strain: float = 0.0
    crack_spacing: float = 0.0
    lch: float = 0.0
    beta_sr_min: float = 0.01
    shear_retention: str = "mcft"
    shear_ret_factor: float = 0.4
    deg_kappa: float = 0.5
    deg_slip_ref: float = 0.01
    deg_min: float = 0.1
    implex: bool = False
    implex_alpha: float = 1.0
    implex_control: tuple[float, float] | None = None
    tens_stiff: str = "off"
    tens_stiff_c: float = 500.0
    tens_stiff_alpha: float = 1.0
    auto_regularization: float | None = None

    @classmethod
    def from_fc(
        cls, *,
        E: float,
        nu: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        rho: float = 0.0,
        regularize: bool = True,
        **kwargs: object,
    ) -> "_LadrunoRC":
        """Build from physical inputs, generating the backbones in Python.

        Mirrors :meth:`ASDConcrete3D.from_fc`: ``ft`` defaults to
        ``0.1*fc``; ``Gf``/``Gc`` to the CEB-FIP correlations; ``lch_ref``
        to the native self-derived band width. ``regularize=True`` (default)
        wires ``-autoRegularization $lch_ref`` so the crack-band softening is
        mesh-objective. Extra ``kwargs`` pass straight through to the
        constructor (e.g. ``interlock=True``, ``tens_stiff="vc"``).
        """
        if E <= 0:
            raise ValueError(f"{cls.__name__}.from_fc: E must be > 0, got {E!r}")
        if fc <= 0:
            raise ValueError(
                f"{cls.__name__}.from_fc: fc must be > 0, got {fc!r}"
            )
        for label, val in (("ft", ft), ("Gf", Gf), ("Gc", Gc),
                           ("lch_ref", lch_ref)):
            if val is not None and val <= 0:
                raise ValueError(
                    f"{cls.__name__}.from_fc: {label} must be > 0 if supplied, "
                    f"got {val!r}"
                )
        ft_ = ft if ft is not None else _laws.default_ft(fc)
        Gf_ = Gf if Gf is not None else _laws.ceb_fip_Gf(fc)
        Gc_ = Gc if Gc is not None else _laws.ceb_fip_Gc(fc, ft_, Gf_)
        lch = lch_ref if lch_ref is not None else _laws.auto_lch_ref(
            E, fc, ft_, Gf_, Gc_)
        Te, Ts, Td = _laws.make_tension(E, ft_, Gf_, lch)
        Ce, Cs, Cd = _laws.make_compression(E, fc, Gc_, lch)
        return cls(
            E=E, nu=nu,
            Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd),
            Te=tuple(Te), Ts=tuple(Ts), Td=tuple(Td),
            rho=rho,
            auto_regularization=(lch if regularize else None),
            **kwargs,
        )

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"{self._type}: E must be > 0, got {self.E!r}")
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(
                f"{self._type}: nu must be in [0, 0.5), got {self.nu!r}"
            )
        for side, e, s, d in (("compression", self.Ce, self.Cs, self.Cd),
                              ("tension", self.Te, self.Ts, self.Td)):
            if len(e) != len(s):
                raise ValueError(
                    f"{self._type}: {side} strain/stress lists must share "
                    f"length, got {len(e)}/{len(s)}"
                )
            if len(e) < 2:
                raise ValueError(
                    f"{self._type}: {side} backbone needs >= 2 points, "
                    f"got {len(e)}"
                )
            if d and len(d) != len(e):
                raise ValueError(
                    f"{self._type}: {side} damage list, when given, must "
                    f"match the backbone length, got {len(d)} vs {len(e)}"
                )
            for dmg in d:
                if not (0.0 <= dmg < 1.0):
                    raise ValueError(
                        f"{self._type}: {side} damage must be in [0, 1), "
                        f"got {dmg!r}"
                    )
        if not (2.0 / 3.0 <= self.Kc <= 1.0):
            raise ValueError(
                f"{self._type}: Kc must be in [2/3, 1], got {self.Kc!r}"
            )
        if self.rho < 0:
            raise ValueError(f"{self._type}: rho must be >= 0, got {self.rho!r}")
        if self.tangent not in _TANGENT_MODES:
            raise ValueError(
                f"{self._type}: tangent must be one of "
                f"{sorted(_TANGENT_MODES)}, got {self.tangent!r}"
            )
        if self.shear_retention not in _SHEAR_RETENTION:
            raise ValueError(
                f"{self._type}: shear_retention must be one of "
                f"{sorted(_SHEAR_RETENTION)}, got {self.shear_retention!r}"
            )
        if self.tens_stiff not in _TENS_STIFF:
            raise ValueError(
                f"{self._type}: tens_stiff must be one of "
                f"{sorted(_TENS_STIFF)}, got {self.tens_stiff!r}"
            )
        if self.tens_stiff == "vc" and self.tens_stiff_c <= 0:
            raise ValueError(
                f"{self._type}: tens_stiff_c must be > 0 in 'vc' mode, got "
                f"{self.tens_stiff_c!r}"
            )
        if self.implex_control is not None and len(self.implex_control) != 2:
            raise ValueError(
                f"{self._type}: implex_control must be (err_tol, "
                f"time_red_lim), got {self.implex_control!r}"
            )
        if self.auto_regularization is not None and self.auto_regularization <= 0:
            raise ValueError(
                f"{self._type}: auto_regularization (lch_ref) must be > 0, "
                f"got {self.auto_regularization!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | int | str] = [self.E, self.nu]
        args += ["-Ce", *self.Ce, "-Cs", *self.Cs]
        if self.Cd:
            args += ["-Cd", *self.Cd]
        args += ["-Te", *self.Te, "-Ts", *self.Ts]
        if self.Td:
            args += ["-Td", *self.Td]
        if self.Kc != 2.0 / 3.0:
            args += ["-Kc", self.Kc]
        if self.beta:
            args.append("-beta")
        if self.beta_floor != 0.1:
            args += ["-betaFloor", self.beta_floor]
        if self.lubliner_reduced:
            args.append("-lublinerReduced")
        if self.rho:
            args += ["-rho", self.rho]
        tan_flag = _TANGENT_MODES[self.tangent]
        if tan_flag is not None:
            args.append(tan_flag)
        if self.interlock:
            args.append("-interlock")
        if self.cyclic:
            args.append("-cyclic")
        if self.agg != 16.0:
            args += ["-agg", self.agg]
        if self.crack_strain:
            args += ["-crackStrain", self.crack_strain]
        if self.crack_spacing:
            args += ["-crackSpacing", self.crack_spacing]
        if self.lch:
            args += ["-lch", self.lch]
        if self.beta_sr_min != 0.01:
            args += ["-betaSrMin", self.beta_sr_min]
        if self.xcrack:
            args.append("-xcrack")
        if self.deg_kappa != 0.5:
            args += ["-degKappa", self.deg_kappa]
        if self.deg_slip_ref != 0.01:
            args += ["-degSlipRef", self.deg_slip_ref]
        if self.deg_min != 0.1:
            args += ["-degMin", self.deg_min]
        if self.implex:
            args.append("-implex")
        if self.implex_alpha != 1.0:
            args += ["-implexAlpha", self.implex_alpha]
        if self.implex_control is not None:
            args += ["-implexControl", *self.implex_control]
        if self.shear_retention != "mcft":
            args += ["-shearRetention", self.shear_retention]
        if self.shear_ret_factor != 0.4:
            args += ["-shearRetFactor", self.shear_ret_factor]
        if self.tens_stiff != "off":
            args += ["-tensStiff", self.tens_stiff]
        if self.tens_stiff_c != 500.0:
            args += ["-tensStiffC", self.tens_stiff_c]
        if self.tens_stiff_alpha != 1.0:
            args += ["-tensStiffAlpha", self.tens_stiff_alpha]
        if self.auto_regularization is not None:
            args += ["-autoRegularization", self.auto_regularization]
        emitter.nDMaterial(self._type, tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoRCConcrete(_LadrunoRC):
    r"""``nDMaterial LadrunoRCConcrete`` — RC plastic-damage + MCFT (Ladruno fork).

    OpenSees command (Ladruno fork, ``ND_TAG`` **LadrunoRCConcrete**)::

        nDMaterial LadrunoRCConcrete tag E nu \
            -Ce {..} -Cs {..} [-Cd {..}] -Te {..} -Ts {..} [-Td {..}] \
            [-Kc Kc] [-beta] [-betaFloor f] [-lublinerReduced] [-rho rho] \
            [-secant | -numericalTangent] [interlock/MCFT flags...] \
            [-tensStiff vc|cm ...] [-autoRegularization lch_ref]

    A small-strain solid-concrete plastic-damage material with MCFT-style
    compression softening and (optionally) aggregate-interlock crack-shear
    retention and tension stiffening — the workhorse for cracked RC walls
    and shells. Prefer the :meth:`from_fc` constructor for the everyday
    physical-input path. See :class:`_LadrunoRC` for the full parameter set.

    .. note::
       Fork-only. Emission works on any build; errors at ``ops.run()`` on
       stock ``openseespy``.
    """

    _type: ClassVar[str] = "LadrunoRCConcrete"


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoRCFiniteStrain(_LadrunoRC):
    r"""``nDMaterial LadrunoRCFiniteStrain`` — finite-strain RC plastic-damage.

    OpenSees command (Ladruno fork)::

        nDMaterial LadrunoRCFiniteStrain tag E nu -Ce {..} -Cs {..} ...

    The Hencky (logarithmic) finite-strain view of :class:`LadrunoRCConcrete`
    — identical plastic-damage + MCFT law, evaluated at large rotation /
    large strain. A ``FiniteStrainNDMaterial`` (driven by ``setTrialF``); the
    consumer is ``LadrunoBrick ... -geom finite``. Same command grammar as
    :class:`LadrunoRCConcrete` (see :class:`_LadrunoRC`).

    .. note::
       Fork-only. Emission works on any build; errors at ``ops.run()`` on
       stock ``openseespy``.
    """

    is_finite_strain: ClassVar[bool] = True
    _type: ClassVar[str] = "LadrunoRCFiniteStrain"


# ---------------------------------------------------------------------------
# LadrunoCohesiveHingeBiaxial — coupled Mz-My cohesive hinge surface (fork)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoCohesiveHingeBiaxial(NDMaterial):
    r"""``nDMaterial LadrunoCohesiveHingeBiaxial`` — coupled biaxial cohesive hinge.

    OpenSees command (Ladruno fork, ``ND_TAG`` **33004**)::

        nDMaterial LadrunoCohesiveHingeBiaxial tag Mcz Gfz Mcy Gfy \
            [-exp | -linear] [-penaltyRatio r] [-bk eta]

    The coupled strong-axis/weak-axis (Mz–My) cohesive interaction surface
    that drives the biaxial embedded hinge of
    ``LadrunoDispBeamColumn -hingeBiaxial``. Each axis carries its own
    cohesive moment ``Mc`` and fracture energy ``Gf``; the mixed-mode
    fracture energy follows the Benzeggagh-Kenane law
    ``Gf_mix = Gfz + (Gfy - Gfz)·wy^eta``.

    .. note::
       Fork-only. Emission produces a deck line on any build; the material
       is unavailable on stock ``openseespy`` and bites only at
       ``ops.run()``. Despite being an ``nDMaterial`` it is a hinge-interaction
       law, not a continuum constitutive model — its sole consumer is the
       biaxial ``LadrunoDispBeamColumn`` hinge.

    Parameters
    ----------
    Mcz, Mcy
        Strong-axis (Mz) and weak-axis (My) cohesive moment capacities
        (both > 0).
    Gfz, Gfy
        Strong-/weak-axis fracture energies per hinge (both > 0).
    softening
        Softening envelope shape: ``"exponential"`` (default, ``-exp``) or
        ``"linear"`` (``-linear``).
    penalty_ratio
        Multiplier on the per-axis snapback-floor penalty
        (``-penaltyRatio``; default 1000, must be > 0).
    bk_eta
        Benzeggagh-Kenane mode-mix exponent (``-bk``; default 1.0, must be
        > 0).
    """

    _SOFTENING: ClassVar[frozenset[str]] = frozenset({"exponential", "linear"})

    Mcz: float
    Gfz: float
    Mcy: float
    Gfy: float
    softening: str = "exponential"
    penalty_ratio: float = 1000.0
    bk_eta: float = 1.0

    def __post_init__(self) -> None:
        for label, val in (("Mcz", self.Mcz), ("Gfz", self.Gfz),
                           ("Mcy", self.Mcy), ("Gfy", self.Gfy)):
            if val <= 0:
                raise ValueError(
                    f"LadrunoCohesiveHingeBiaxial: {label} must be > 0, got "
                    f"{val!r}"
                )
        if self.softening not in self._SOFTENING:
            raise ValueError(
                "LadrunoCohesiveHingeBiaxial: softening must be one of "
                f"{sorted(self._SOFTENING)}, got {self.softening!r}"
            )
        if self.penalty_ratio <= 0:
            raise ValueError(
                "LadrunoCohesiveHingeBiaxial: penalty_ratio must be > 0, got "
                f"{self.penalty_ratio!r}"
            )
        if self.bk_eta <= 0:
            raise ValueError(
                "LadrunoCohesiveHingeBiaxial: bk_eta must be > 0, got "
                f"{self.bk_eta!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | str] = [self.Mcz, self.Gfz, self.Mcy, self.Gfy]
        if self.softening == "linear":
            args.append("-linear")
        if self.penalty_ratio != 1000.0:
            args += ["-penaltyRatio", self.penalty_ratio]
        if self.bk_eta != 1.0:
            args += ["-bk", self.bk_eta]
        emitter.nDMaterial("LadrunoCohesiveHingeBiaxial", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
