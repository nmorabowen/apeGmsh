"""Typed primitives for OpenSees uniaxial materials.

Class names match OpenSees type tokens exactly EXCEPT
:class:`ElasticMaterial`, whose OpenSees type token is the bare
``"Elastic"`` (we rename to avoid clashing with the ``timeSeries``
``"Linear"`` naming pattern, and to keep the Python class name from
shadowing the more general "elastic" concept). The :meth:`_emit`
method passes the correct token to the emitter regardless.

Each class is a frozen, slotted, kw-only dataclass that inherits from
:class:`apeGmsh.opensees._internal.types.UniaxialMaterial`. Validation
runs in ``__post_init__``; emission positional order matches the
OpenSees manual command syntax.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

from . import _asdconcrete_laws as _laws
from . import _ladruno_j2 as _lj2
from .nd import ASDRegularizationWarning
from .._internal.tag_resolution import resolve_tag
from .._internal.types import Primitive, UniaxialMaterial
from ..emitter.base import Emitter


__all__ = [
    "Steel01",
    "Steel02",
    "ASDSteel1D",
    "Concrete01",
    "Concrete02",
    "Hysteretic",
    "ElasticMaterial",
    "ENT",
    "Viscous",
    "ViscousDamper",
    "Maxwell",
    "InitialStress",
    "ASDConcrete1D",
    "LadrunoBondSlip",
    "LadrunoUniaxialJ2",
    "LadrunoRebarBuckling",
    "LadrunoCohesiveHinge",
]


_REBAR_BUCKLING_MODELS: tuple[str, ...] = ("dm", "ga")
_REBAR_RESTRAIGHTEN_MODES: tuple[str, ...] = ("lambda", "c")


# ---------------------------------------------------------------------------
# Steel
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Steel01(UniaxialMaterial):
    """``uniaxialMaterial Steel01`` — bilinear with kinematic hardening.

    OpenSees command::

        uniaxialMaterial Steel01 tag fy E b [a1 a2 a3 a4]

    The four optional isotropic-hardening parameters ``a1..a4`` must
    be supplied as a complete quad or omitted entirely; OpenSees
    defaults all four to ``0.0`` when omitted.
    """

    fy: float
    E:  float
    b:  float
    a1: float | None = None
    a2: float | None = None
    a3: float | None = None
    a4: float | None = None

    def __post_init__(self) -> None:
        if self.fy <= 0:
            raise ValueError(f"Steel01: fy must be > 0, got {self.fy!r}")
        if self.E <= 0:
            raise ValueError(f"Steel01: E must be > 0, got {self.E!r}")
        if not (0.0 <= self.b < 1.0):
            raise ValueError(
                f"Steel01: b must be in [0, 1), got {self.b!r}"
            )
        a_provided = sum(
            x is not None for x in (self.a1, self.a2, self.a3, self.a4)
        )
        if a_provided not in (0, 4):
            raise ValueError(
                "Steel01: isotropic-hardening params (a1..a4) must be "
                "supplied as a complete quad or omitted entirely."
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [self.fy, self.E, self.b]
        if self.a1 is not None:
            assert self.a2 is not None
            assert self.a3 is not None
            assert self.a4 is not None
            params += [self.a1, self.a2, self.a3, self.a4]
        emitter.uniaxialMaterial("Steel01", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Steel02(UniaxialMaterial):
    """``uniaxialMaterial Steel02`` — Menegotto-Pinto with isotropic hardening.

    OpenSees command::

        uniaxialMaterial Steel02 tag fy E b R0 cR1 cR2 \
            [a1 a2 a3 a4 [sigInit]]

    ``sig_init`` is only valid when the ``a1..a4`` quad is supplied.
    """

    fy:       float
    E:        float
    b:        float
    R0:       float = 20.0
    cR1:      float = 0.925
    cR2:      float = 0.15
    a1:       float | None = None
    a2:       float | None = None
    a3:       float | None = None
    a4:       float | None = None
    sig_init: float | None = None

    def __post_init__(self) -> None:
        if self.fy <= 0:
            raise ValueError(f"Steel02: fy must be > 0, got {self.fy!r}")
        if self.E <= 0:
            raise ValueError(f"Steel02: E must be > 0, got {self.E!r}")
        if not (0.0 <= self.b < 1.0):
            raise ValueError(
                f"Steel02: b must be in [0, 1), got {self.b!r}"
            )
        if self.R0 <= 0:
            raise ValueError(f"Steel02: R0 must be > 0, got {self.R0!r}")
        a_provided = sum(
            x is not None for x in (self.a1, self.a2, self.a3, self.a4)
        )
        if a_provided not in (0, 4):
            raise ValueError(
                "Steel02: isotropic-hardening params (a1..a4) must be "
                "supplied as a complete quad or omitted entirely."
            )
        if self.sig_init is not None and a_provided != 4:
            raise ValueError(
                "Steel02: sig_init requires the a1..a4 quad to be set."
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [
            self.fy, self.E, self.b, self.R0, self.cR1, self.cR2,
        ]
        if self.a1 is not None:
            assert self.a2 is not None
            assert self.a3 is not None
            assert self.a4 is not None
            params += [self.a1, self.a2, self.a3, self.a4]
            if self.sig_init is not None:
                params.append(self.sig_init)
        emitter.uniaxialMaterial("Steel02", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class ASDSteel1D(UniaxialMaterial):
    """``uniaxialMaterial ASDSteel1D`` — ASDEA plastic-damage steel.

    Unified plastic-damage steel-bar model (fracture, bond-slip and
    buckling via multiscale homogenization), by A. Casalucci,
    M. Petracca, G. Camata — ASDEA Software Technology.

    OpenSees command::

        uniaxialMaterial ASDSteel1D $tag $E $sy $su $eu
            <-implex> <-implexControl $errTol $timeRedLimit>
            <-auto_regularization>
            <-buckling $lch <$r>> <-fracture <$r>> <-slip $matTag <$r>>
            <-K_alpha $K_alpha> <-max_iter $max_iter>
            <-tolU $tolU> <-tolR $tolR>

    The Chaboche two-term hardening backbone is **derived internally**
    by the material from ``(E, sy, su, eu)`` so the initial slope is
    close to ``E`` and the stress approaches ``su`` at strain ``eu``;
    those hardening constants are not user inputs.

    The optional bar ``radius`` is shared by the ``-buckling`` /
    ``-fracture`` / ``-slip`` RVE features — supply it once. It is
    emitted on the first active feature in the order buckling →
    fracture → slip (the C++ parser cross-checks consistency).

    Parameters
    ----------
    E
        Young's modulus (strictly positive).
    sy
        Yield stress (strictly positive).
    su
        Ultimate stress (strictly positive, must exceed ``sy``).
    eu
        Ultimate strain — strain at which the stress reaches ``su``
        (strictly positive).
    implex
        Emit ``-implex`` (IMPL-EX integration).
    implex_control
        ``(errorTolerance, timeReductionLimit)`` for
        ``-implexControl``; ``None`` omits the flag.
    auto_regularization
        Emit ``-auto_regularization``.
    buckling_lch
        Buckling characteristic length ``$lch`` (emits ``-buckling
        $lch``); ``None`` disables buckling.
    fracture
        Emit ``-fracture`` (ductile-fracture homogenization).
    slip_material
        Bond-slip :class:`UniaxialMaterial` (emits ``-slip $matTag``);
        ``None`` disables bond-slip. This is the only dependency.
    radius
        Shared bar radius ``$r`` for the active RVE feature(s).
    K_alpha, max_iter, tolU, tolR
        RVE micro-solver controls; ``None`` leaves the OpenSees
        defaults.
    """

    E:  float
    sy: float
    su: float
    eu: float
    implex: bool = False
    implex_control: tuple[float, float] | None = None
    auto_regularization: bool = False
    buckling_lch: float | None = None
    fracture: bool = False
    slip_material: UniaxialMaterial | None = None
    radius: float | None = None
    K_alpha:  float | None = None
    max_iter: int | None = None
    tolU: float | None = None
    tolR: float | None = None

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"ASDSteel1D: E must be > 0, got {self.E!r}")
        if self.sy <= 0:
            raise ValueError(f"ASDSteel1D: sy must be > 0, got {self.sy!r}")
        if self.su <= 0:
            raise ValueError(f"ASDSteel1D: su must be > 0, got {self.su!r}")
        if self.eu <= 0:
            raise ValueError(f"ASDSteel1D: eu must be > 0, got {self.eu!r}")
        if self.su <= self.sy:
            raise ValueError(
                f"ASDSteel1D: su must be > sy, got su={self.su!r}, "
                f"sy={self.sy!r}"
            )
        if self.buckling_lch is not None and self.buckling_lch <= 0:
            raise ValueError(
                f"ASDSteel1D: buckling_lch must be > 0 if supplied, "
                f"got {self.buckling_lch!r}"
            )
        if self.radius is not None and self.radius <= 0:
            raise ValueError(
                f"ASDSteel1D: radius must be > 0 if supplied, got "
                f"{self.radius!r}"
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.slip_material is not None:
            return (self.slip_material,)
        return ()

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float | str] = [self.E, self.sy, self.su, self.eu]
        if self.implex:
            params.append("-implex")
        if self.implex_control is not None:
            params += ["-implexControl", *self.implex_control]
        if self.auto_regularization:
            params.append("-auto_regularization")
        # The shared bar radius rides on the first active RVE feature.
        radius_used = False

        def _radius_tail() -> list[float | str]:
            nonlocal radius_used
            if self.radius is not None and not radius_used:
                radius_used = True
                return [self.radius]
            return []

        if self.buckling_lch is not None:
            params += ["-buckling", self.buckling_lch, *_radius_tail()]
        if self.fracture:
            params += ["-fracture", *_radius_tail()]
        if self.slip_material is not None:
            slip_tag = resolve_tag(emitter, self.slip_material)
            params += ["-slip", slip_tag, *_radius_tail()]
        if self.K_alpha is not None:
            params += ["-K_alpha", self.K_alpha]
        if self.max_iter is not None:
            params += ["-max_iter", self.max_iter]
        if self.tolU is not None:
            params += ["-tolU", self.tolU]
        if self.tolR is not None:
            params += ["-tolR", self.tolR]
        emitter.uniaxialMaterial("ASDSteel1D", tag, *params)


# ---------------------------------------------------------------------------
# Concrete
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Concrete01(UniaxialMaterial):
    """``uniaxialMaterial Concrete01`` — Kent-Scott-Park, no tensile strength.

    OpenSees command::

        uniaxialMaterial Concrete01 tag fpc epsc0 fpcu epsU

    All four parameters MUST be negative (compression-positive
    convention is rejected — OpenSees uses compression-negative).
    """

    fpc:   float
    epsc0: float
    fpcu:  float
    epsU:  float

    def __post_init__(self) -> None:
        if self.fpc >= 0:
            raise ValueError(
                f"Concrete01: fpc must be < 0 (compression negative), "
                f"got {self.fpc!r}"
            )
        if self.epsc0 >= 0:
            raise ValueError(
                f"Concrete01: epsc0 must be < 0, got {self.epsc0!r}"
            )
        if self.fpcu >= 0:
            raise ValueError(
                f"Concrete01: fpcu must be < 0, got {self.fpcu!r}"
            )
        if self.epsU >= 0:
            raise ValueError(
                f"Concrete01: epsU must be < 0, got {self.epsU!r}"
            )
        if self.epsU >= self.epsc0:
            raise ValueError(
                f"Concrete01: epsU must be more compressive than epsc0 "
                f"(epsU < epsc0), got epsU={self.epsU!r}, "
                f"epsc0={self.epsc0!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial(
            "Concrete01", tag, self.fpc, self.epsc0, self.fpcu, self.epsU,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Concrete02(UniaxialMaterial):
    """``uniaxialMaterial Concrete02`` — Kent-Scott-Park with tensile strength.

    OpenSees command::

        uniaxialMaterial Concrete02 tag fpc epsc0 fpcu epsU \
            lambda ft Ets

    The reload-stiffness ratio is named ``lambda`` in the OpenSees
    manual; the dataclass field is ``lambda_val`` because ``lambda``
    is a Python keyword. The emit order is unchanged.
    """

    fpc:        float
    epsc0:      float
    fpcu:       float
    epsU:       float
    lambda_val: float
    ft:         float
    Ets:        float

    def __post_init__(self) -> None:
        if self.fpc >= 0:
            raise ValueError(
                f"Concrete02: fpc must be < 0 (compression negative), "
                f"got {self.fpc!r}"
            )
        if self.epsc0 >= 0:
            raise ValueError(
                f"Concrete02: epsc0 must be < 0, got {self.epsc0!r}"
            )
        if self.fpcu >= 0:
            raise ValueError(
                f"Concrete02: fpcu must be < 0, got {self.fpcu!r}"
            )
        if self.epsU >= 0:
            raise ValueError(
                f"Concrete02: epsU must be < 0, got {self.epsU!r}"
            )
        if self.epsU >= self.epsc0:
            raise ValueError(
                f"Concrete02: epsU must be more compressive than epsc0 "
                f"(epsU < epsc0), got epsU={self.epsU!r}, "
                f"epsc0={self.epsc0!r}"
            )
        if not (0.0 <= self.lambda_val <= 1.0):
            raise ValueError(
                f"Concrete02: lambda_val must be in [0, 1], got "
                f"{self.lambda_val!r}"
            )
        if self.ft <= 0:
            raise ValueError(f"Concrete02: ft must be > 0, got {self.ft!r}")
        if self.Ets <= 0:
            raise ValueError(
                f"Concrete02: Ets must be > 0, got {self.Ets!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial(
            "Concrete02", tag,
            self.fpc, self.epsc0, self.fpcu, self.epsU,
            self.lambda_val, self.ft, self.Ets,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Hysteretic
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Hysteretic(UniaxialMaterial):
    """``uniaxialMaterial Hysteretic`` — multi-linear hysteretic backbone.

    OpenSees command::

        uniaxialMaterial Hysteretic tag \
            s1p e1p s2p e2p [s3p e3p] \
            s1n e1n s2n e2n [s3n e3n] \
            pinchX pinchY damage1 damage2 [beta]

    The third envelope points (``s3p, e3p, s3n, e3n``) must be
    supplied as a complete quad or omitted. ``beta`` defaults to
    ``0.0`` and is appended only when non-zero.
    """

    s1p: float
    e1p: float
    s2p: float
    e2p: float
    s1n: float
    e1n: float
    s2n: float
    e2n: float
    pinch_x: float
    pinch_y: float
    damage1: float
    damage2: float
    s3p: float | None = None
    e3p: float | None = None
    s3n: float | None = None
    e3n: float | None = None
    beta: float = 0.0

    def __post_init__(self) -> None:
        # Positive backbone strains and stresses.
        if self.e1p <= 0:
            raise ValueError(
                f"Hysteretic: e1p must be > 0, got {self.e1p!r}"
            )
        if self.e2p <= 0:
            raise ValueError(
                f"Hysteretic: e2p must be > 0, got {self.e2p!r}"
            )
        if self.s1p <= 0:
            raise ValueError(
                f"Hysteretic: s1p must be > 0, got {self.s1p!r}"
            )
        if self.s2p <= 0:
            raise ValueError(
                f"Hysteretic: s2p must be > 0, got {self.s2p!r}"
            )
        # Negative backbone strains and stresses.
        if self.e1n >= 0:
            raise ValueError(
                f"Hysteretic: e1n must be < 0, got {self.e1n!r}"
            )
        if self.e2n >= 0:
            raise ValueError(
                f"Hysteretic: e2n must be < 0, got {self.e2n!r}"
            )
        if self.s1n >= 0:
            raise ValueError(
                f"Hysteretic: s1n must be < 0, got {self.s1n!r}"
            )
        if self.s2n >= 0:
            raise ValueError(
                f"Hysteretic: s2n must be < 0, got {self.s2n!r}"
            )
        # Pinch / damage bounds.
        if not (0.0 <= self.pinch_x <= 1.0):
            raise ValueError(
                f"Hysteretic: pinch_x must be in [0, 1], got "
                f"{self.pinch_x!r}"
            )
        if not (0.0 <= self.pinch_y <= 1.0):
            raise ValueError(
                f"Hysteretic: pinch_y must be in [0, 1], got "
                f"{self.pinch_y!r}"
            )
        if self.damage1 < 0:
            raise ValueError(
                f"Hysteretic: damage1 must be >= 0, got {self.damage1!r}"
            )
        if self.damage2 < 0:
            raise ValueError(
                f"Hysteretic: damage2 must be >= 0, got {self.damage2!r}"
            )
        # Monotonic envelope.
        if self.e2p <= self.e1p:
            raise ValueError(
                f"Hysteretic: positive backbone must be monotonic "
                f"(e2p > e1p), got e1p={self.e1p!r}, e2p={self.e2p!r}"
            )
        if self.e2n >= self.e1n:
            raise ValueError(
                f"Hysteretic: negative backbone must be monotonic "
                f"(e2n < e1n), got e1n={self.e1n!r}, e2n={self.e2n!r}"
            )
        # Third-point quad: all-or-none.
        third_provided = sum(
            x is not None for x in (self.s3p, self.e3p, self.s3n, self.e3n)
        )
        if third_provided not in (0, 4):
            raise ValueError(
                "Hysteretic: third envelope point (s3p, e3p, s3n, e3n) "
                "must be supplied as a complete quad or omitted entirely."
            )
        if third_provided == 4:
            # mypy: narrowed by the count above.
            assert self.e3p is not None
            assert self.e3n is not None
            assert self.s3p is not None
            assert self.s3n is not None
            if self.e3p <= self.e2p:
                raise ValueError(
                    f"Hysteretic: positive backbone must be monotonic "
                    f"(e3p > e2p), got e2p={self.e2p!r}, e3p={self.e3p!r}"
                )
            if self.e3n >= self.e2n:
                raise ValueError(
                    f"Hysteretic: negative backbone must be monotonic "
                    f"(e3n < e2n), got e2n={self.e2n!r}, e3n={self.e3n!r}"
                )
            if self.s3p <= 0:
                raise ValueError(
                    f"Hysteretic: s3p must be > 0, got {self.s3p!r}"
                )
            if self.s3n >= 0:
                raise ValueError(
                    f"Hysteretic: s3n must be < 0, got {self.s3n!r}"
                )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [
            self.s1p, self.e1p, self.s2p, self.e2p,
        ]
        if self.s3p is not None:
            assert self.e3p is not None
            params += [self.s3p, self.e3p]
        params += [self.s1n, self.e1n, self.s2n, self.e2n]
        if self.s3n is not None:
            assert self.e3n is not None
            params += [self.s3n, self.e3n]
        params += [self.pinch_x, self.pinch_y, self.damage1, self.damage2]
        if self.beta != 0.0:
            params.append(self.beta)
        emitter.uniaxialMaterial("Hysteretic", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Elastic / ENT
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ElasticMaterial(UniaxialMaterial):
    """``uniaxialMaterial Elastic`` — linear elastic with optional damping.

    OpenSees command::

        uniaxialMaterial Elastic tag E [eta]

    Renamed to ``ElasticMaterial`` on the Python side to avoid a name
    clash with the more general "elastic" concept used elsewhere in
    the bridge. The OpenSees type token emitted is the bare
    ``"Elastic"``.
    """

    E:   float
    eta: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(
                f"ElasticMaterial: E must be > 0, got {self.E!r}"
            )
        if self.eta < 0:
            raise ValueError(
                f"ElasticMaterial: eta must be >= 0, got {self.eta!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [self.E]
        if self.eta != 0.0:
            params.append(self.eta)
        emitter.uniaxialMaterial("Elastic", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class ENT(UniaxialMaterial):
    """``uniaxialMaterial ENT`` — elastic-no-tension.

    OpenSees command::

        uniaxialMaterial ENT tag E
    """

    E: float

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"ENT: E must be > 0, got {self.E!r}")

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial("ENT", tag, self.E)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Dashpot / viscous (rate-dependent) materials
#
# These are the only uniaxials that consume the ``strainRate`` argument of
# ``setTrialStrain(strain, rate)`` and return a velocity-proportional force —
# the constitutive half of a dashpot.  They work ONLY inside a rate-capable
# element (``ZeroLength`` / ``TwoNodeLink`` / ``CoupledZeroLength``), which
# feeds the rate via the 2-arg ``setTrialStrain``.  Inside ``section
# Aggregator`` / ``ZeroLengthSection`` (no rate channel) they are silently
# inert — hence ``is_rate_dependent = True`` so consumers can fail loud.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Viscous(UniaxialMaterial):
    """``uniaxialMaterial Viscous`` — pure dashpot (rate-dependent).

    OpenSees command::

        uniaxialMaterial Viscous tag C alpha [minVel]

    Force law ``F = C·sgn(v)·|v|^alpha`` (``ViscousMaterial.cpp``). This
    is the canonical absorbing-boundary / Lysmer dashpot when dropped
    into a :class:`~apeGmsh.opensees.element.zero_length.ZeroLength`
    ``-mat``/``-dir`` slot: the element feeds it a strain *rate* via the
    2-arg ``setTrialStrain(strain, rate)``, so the force is velocity-
    proportional and active with **no** ``-doRayleigh`` flag.

    .. warning::
       ``Viscous`` has **zero static stiffness** (``getTangent()``
       returns 0); used alone in a ``ZeroLength`` it makes the static
       tangent **singular**. Parallel it with an elastic spring — a
       second ``(material, dof)`` pair on the same DOF — for any
       analysis that forms a static tangent.

    Parameters
    ----------
    C
        Damping coefficient (must be > 0).
    alpha
        Velocity exponent (must be > 0; ``1.0`` = linear dashpot).
    min_vel
        Lower velocity cut-off below which the tangent is frozen
        (OpenSees default ``1e-11``). Emitted only when non-default.
    """

    C: float
    alpha: float = 1.0
    min_vel: float = 1.0e-11

    is_rate_dependent: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError(f"Viscous: C must be > 0, got {self.C!r}")
        if self.alpha <= 0:
            raise ValueError(
                f"Viscous: alpha must be > 0, got {self.alpha!r}"
            )
        if self.min_vel <= 0:
            raise ValueError(
                f"Viscous: min_vel must be > 0, got {self.min_vel!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [self.C, self.alpha]
        if self.min_vel != 1.0e-11:
            params.append(self.min_vel)
        emitter.uniaxialMaterial("Viscous", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class ViscousDamper(UniaxialMaterial):
    """``uniaxialMaterial ViscousDamper`` — Maxwell (spring + dashpot).

    OpenSees command::

        uniaxialMaterial ViscousDamper tag K C alpha [LGap]

    Linear spring ``K`` in series with a nonlinear dashpot
    ``F = C·sgn(v)·|v|^alpha`` (``ViscousDamper.cpp``), integrated by an
    internal adaptive ODE solver off the global time step. Rate-
    dependent, but reports ``getDampTangent() == 0`` and produces force
    only in a **transient** analysis with a defined ``dt``. ``l_gap``
    models slack/gap in the device.

    The advanced solver knobs (``NM/RelTol/AbsTol/MaxHalf``) are left at
    their OpenSees defaults.
    """

    K: float
    C: float
    alpha: float
    l_gap: float | None = None

    is_rate_dependent: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"ViscousDamper: K must be > 0, got {self.K!r}")
        if self.C <= 0:
            raise ValueError(f"ViscousDamper: C must be > 0, got {self.C!r}")
        if self.alpha <= 0:
            raise ValueError(
                f"ViscousDamper: alpha must be > 0, got {self.alpha!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [self.K, self.C, self.alpha]
        if self.l_gap is not None:
            params.append(self.l_gap)
        emitter.uniaxialMaterial("ViscousDamper", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Maxwell(UniaxialMaterial):
    """``uniaxialMaterial Maxwell`` — Maxwell viscoelastic (closed form).

    OpenSees command::

        uniaxialMaterial Maxwell tag K C alpha L

    Spring ``K`` in series with dashpot ``C``, closed-form exponential
    relaxation with relaxation time ``tR = (C/L^alpha)/K``
    (``Maxwell.cpp``). Unlike ``Viscous`` it carries a **nonzero**
    tangent ``K``. ``length`` is the device length ``L``. Its rate
    effect comes from the global ``dt`` (not the ``strainRate`` arg), so
    it is meaningful only in a transient analysis.
    """

    K: float
    C: float
    alpha: float
    length: float

    is_rate_dependent: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"Maxwell: K must be > 0, got {self.K!r}")
        if self.C <= 0:
            raise ValueError(f"Maxwell: C must be > 0, got {self.C!r}")
        if self.alpha <= 0:
            raise ValueError(
                f"Maxwell: alpha must be > 0, got {self.alpha!r}"
            )
        if self.length <= 0:
            raise ValueError(
                f"Maxwell: length must be > 0, got {self.length!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial(
            "Maxwell", tag, self.K, self.C, self.alpha, self.length,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Wrappers — compose other uniaxial materials
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class InitialStress(UniaxialMaterial):
    """``uniaxialMaterial InitialStressMaterial`` — wrap a uniaxial with
    a per-fiber initial stress.

    OpenSees command::

        uniaxialMaterial InitialStressMaterial $tag $base_tag $sigma_init

    The wrapper applies a constant ``sigma_init`` superimposed on the
    base material's stress response. Useful for per-fiber residual
    stress patterns (e.g. ECCS welded-I residual: +0.5/-0.3/+0.2 Fy
    across the web/flange bands) where ``Steel02``'s built-in
    ``sig_init`` field is insufficient because it is per-material, not
    per-fiber.

    The base material is held by reference; its tag is resolved at
    emit time via :func:`resolve_tag`. The base must be a
    :class:`UniaxialMaterial` — nD or section bases raise
    :class:`TypeError` at construction.

    Parameters
    ----------
    base_material
        The wrapped uniaxial material. Must itself be a typed
        :class:`UniaxialMaterial` primitive; its tag is resolved at
        emit time. The bridge emits the base material **before** the
        wrapper via :meth:`dependencies`.
    sigma_init
        Constant initial stress added to the base material's response.
        Sign convention matches the base material (tension positive
        for steels).
    """

    base_material: UniaxialMaterial
    sigma_init: float

    def __post_init__(self) -> None:
        if not isinstance(self.base_material, UniaxialMaterial):
            raise TypeError(
                "InitialStress: base_material must be a UniaxialMaterial "
                f"primitive, got {type(self.base_material).__name__!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.base_material,)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        base_tag = resolve_tag(emitter, self.base_material)
        emitter.uniaxialMaterial(
            "InitialStressMaterial", tag, base_tag, self.sigma_init,
        )


# ---------------------------------------------------------------------------
# ASDConcrete1D — uniaxial sibling of ASDConcrete3D (fibers / trusses)
# ---------------------------------------------------------------------------
#
# See ADR 0044. Same owned-backbone contract as ASDConcrete3D: apeGmsh
# generates the curve (the 1-D ``-fc`` formulas are byte-identical to the
# 3-D ones, so :mod:`._asdconcrete_laws` is reused) and emits the explicit
# ``-Te/-Ts/-Td/-Ce/-Cs/-Cd`` card with ``-autoRegularization $lch_ref``.
#
# CONFINEMENT-BLIND: the uniaxial model has no stress decomposition / Lubliner
# surface, so it cannot see triaxial confinement. For a confined member, bake
# the confinement into the backbone yourself (e.g. a Mander curve) and pass it
# via the explicit constructor — ``from_fc`` produces an UNCONFINED law.


@dataclass(frozen=True, kw_only=True, slots=True)
class ASDConcrete1D(UniaxialMaterial):
    """``uniaxialMaterial ASDConcrete1D`` — Petracca plastic-damage concrete (1-D).

    Uniaxial sibling of :class:`~apeGmsh.opensees.material.nd.ASDConcrete3D`
    for fibers / trusses. Prefer :meth:`from_fc`; the raw constructor takes
    pre-built backbones (the channel for a Mander-confined curve — see
    module note on confinement-blindness).

    Parameters mirror :class:`ASDConcrete3D` minus the continuum-only ones
    (no ``v``, ``Kc``, ``cdf``, ``rho``, ``crackPlanes``). ``lch_ref`` (``>0``)
    is emitted to ``-autoRegularization``; the bare flag is a parser error.
    """

    E: float
    Te: tuple[float, ...]
    Ts: tuple[float, ...]
    Td: tuple[float, ...]
    Ce: tuple[float, ...]
    Cs: tuple[float, ...]
    Cd: tuple[float, ...]
    lch_ref: float
    eta: float = 0.0
    implex: bool = False
    auto_regularize: bool = True
    ft: float | None = None
    Gf: float | None = None

    @classmethod
    def from_fc(
        cls, *,
        E: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        eta: float = 0.0,
        implex: bool = False,
    ) -> "ASDConcrete1D":
        """Build an **unconfined** 1-D law from physical inputs.

        Defaults match :meth:`ASDConcrete3D.from_fc` (``ft=0.1*fc``,
        CEB-FIP ``Gf``/``Gc``, native self-derived ``lch_ref``). For a
        confined member, supply the backbone explicitly instead.
        """
        if E <= 0:
            raise ValueError(f"ASDConcrete1D.from_fc: E must be > 0, got {E!r}")
        if fc <= 0:
            raise ValueError(f"ASDConcrete1D.from_fc: fc must be > 0, got {fc!r}")
        for label, val in (("ft", ft), ("Gf", Gf), ("Gc", Gc),
                           ("lch_ref", lch_ref)):
            if val is not None and val <= 0:
                raise ValueError(
                    f"ASDConcrete1D.from_fc: {label} must be > 0 if supplied, "
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
            E=E,
            Te=tuple(Te), Ts=tuple(Ts), Td=tuple(Td),
            Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd),
            lch_ref=lch, eta=eta, implex=implex, ft=ft_, Gf=Gf_,
        )

    @classmethod
    def from_mander(
        cls, *,
        E: float,
        fc: float,
        eps_cu: float,
        fcc: float | None = None,
        fl: float | None = None,
        eps_co: float = 0.002,
        plastic_ratio: float = 0.7,
        n_comp: int = 12,
        ft: float | None = None,
        Gf: float | None = None,
        lch_ref: float | None = None,
        auto_regularize: bool = False,
        eta: float = 0.0,
        implex: bool = False,
    ) -> "ASDConcrete1D":
        """Build a **Mander confined-concrete** fiber law (Mander et al. 1988).

        The uniaxial model is confinement-blind, so confinement is baked into
        the compression backbone here. Confinement enters as **exactly one** of
        ``fcc`` (confined peak strength) or ``fl`` (effective lateral confining
        pressure → Mander strength). ``eps_cu`` is the confined ultimate strain,
        set by the transverse steel — the caller's responsibility. Tension uses
        the standard CEB-FIP law (confinement barely affects tension).

        ``auto_regularize`` defaults to **False**: the Mander envelope is the
        physical confined response and must NOT be crack-band-rescaled (its
        ductility is confinement-defined, not fracture-energy-defined). With it
        off, ``lch_ref`` only sets the tension-curve reference length. Enable it
        explicitly only if you understand the interaction with the confined
        softening branch.

        Parameters
        ----------
        E, fc
            Concrete modulus (``>0``) and **unconfined** peak strength (``>0``).
        eps_cu
            Confined ultimate (crushing) strain; must exceed the confined peak
            strain ``eps_cc`` (derived internally).
        fcc, fl
            Provide one: confined strength, or effective lateral pressure.
        eps_co
            Unconfined peak strain (default ``0.002``).
        plastic_ratio
            Plastic/damage split of inelastic strain, ``[0, 1]`` (``1`` = pure
            plastic unloading, ``0`` = pure damage). Default ``0.7``.
        n_comp
            Number of points sampling the compression envelope (default ``12``).
        """
        from . import _mander

        if E <= 0:
            raise ValueError(f"ASDConcrete1D.from_mander: E must be > 0, got {E!r}")
        if fc <= 0:
            raise ValueError(f"ASDConcrete1D.from_mander: fc must be > 0, got {fc!r}")
        if (fcc is None) == (fl is None):
            raise ValueError(
                "ASDConcrete1D.from_mander: provide exactly one of fcc "
                "(confined strength) or fl (effective lateral pressure)."
            )
        if fcc is not None and fcc <= fc:
            raise ValueError(
                f"ASDConcrete1D.from_mander: fcc ({fcc!r}) must exceed the "
                f"unconfined fc ({fc!r})."
            )
        if fl is not None and fl <= 0:
            raise ValueError(
                f"ASDConcrete1D.from_mander: fl must be > 0, got {fl!r}"
            )
        if eps_cu <= 0:
            raise ValueError(
                f"ASDConcrete1D.from_mander: eps_cu must be > 0, got {eps_cu!r}"
            )
        if not (0.0 <= plastic_ratio <= 1.0):
            raise ValueError(
                f"ASDConcrete1D.from_mander: plastic_ratio must be in [0, 1], "
                f"got {plastic_ratio!r}"
            )

        if fcc is not None:
            fcc_ = fcc
        else:
            # The exactly-one-of-(fcc, fl) check above guarantees fl here.
            assert fl is not None
            fcc_ = _mander.confined_strength(fc, fl)
        eps_cc = _mander.confined_peak_strain(fc, fcc_, eps_co)
        if eps_cu <= eps_cc:
            raise ValueError(
                f"ASDConcrete1D.from_mander: eps_cu ({eps_cu!r}) must exceed the "
                f"confined peak strain eps_cc ({eps_cc:g})."
            )
        Ce, Cs, Cd = _mander.compression_backbone(
            E, fcc_, eps_cc, eps_cu, n=n_comp, plastic_ratio=plastic_ratio
        )

        ft_ = ft if ft is not None else _laws.default_ft(fc)
        Gf_ = Gf if Gf is not None else _laws.ceb_fip_Gf(fc)
        Gc_ = _laws.ceb_fip_Gc(fc, ft_, Gf_)
        lch = lch_ref if lch_ref is not None else _laws.auto_lch_ref(
            E, fc, ft_, Gf_, Gc_)
        Te, Ts, Td = _laws.make_tension(E, ft_, Gf_, lch)
        return cls(
            E=E,
            Te=tuple(Te), Ts=tuple(Ts), Td=tuple(Td),
            Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd),
            lch_ref=lch, eta=eta, implex=implex, auto_regularize=auto_regularize,
            ft=ft_, Gf=Gf_,
        )

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"ASDConcrete1D: E must be > 0, got {self.E!r}")
        if self.lch_ref <= 0:
            raise ValueError(
                f"ASDConcrete1D: lch_ref must be > 0, got {self.lch_ref!r}"
            )
        if self.eta < 0:
            raise ValueError(
                f"ASDConcrete1D: eta must be >= 0, got {self.eta!r}"
            )
        for side, (e, s, d) in (("tension", (self.Te, self.Ts, self.Td)),
                                ("compression", (self.Ce, self.Cs, self.Cd))):
            if not (len(e) == len(s) == len(d)):
                raise ValueError(
                    f"ASDConcrete1D: {side} backbone lists must share length, "
                    f"got {len(e)}/{len(s)}/{len(d)}"
                )
            if len(e) < 2:
                raise ValueError(
                    f"ASDConcrete1D: {side} backbone needs >= 2 points, "
                    f"got {len(e)}"
                )
        for dmg in (*self.Td, *self.Cd):
            if not (0.0 <= dmg < 1.0):
                raise ValueError(
                    f"ASDConcrete1D: damage must be in [0, 1), got {dmg!r}"
                )

    def preview_backbone(self) -> dict[str, tuple[float, ...] | float]:
        """The exact backbone that will be emitted (read-only, for plotting)."""
        return {
            "Te": self.Te, "Ts": self.Ts, "Td": self.Td,
            "Ce": self.Ce, "Cs": self.Cs, "Cd": self.Cd,
            "lch_ref": self.lch_ref,
        }

    def l_max(self) -> float | None:
        """Crack-band snapback ceiling ``2*E*Gf/ft^2``, or ``None`` if unknown."""
        if self.ft is None or self.Gf is None:
            return None
        return _laws.l_max(self.E, self.Gf, self.ft)

    def check_element_size(self, lch: float, *, pg: str | None = None) -> bool:
        """Warn (never raise) if ``lch`` exceeds :meth:`l_max`; ``True`` if OK."""
        lm = self.l_max()
        if lm is not None and lch > lm:
            where = f", PG {pg!r}" if pg is not None else ""
            warnings.warn(
                f"ASDConcrete1D: element size lch={lch:g} exceeds the "
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
            self.E,
            "-Te", *self.Te, "-Ts", *self.Ts, "-Td", *self.Td,
            "-Ce", *self.Ce, "-Cs", *self.Cs, "-Cd", *self.Cd,
        ]
        if self.eta:
            args += ["-eta", self.eta]
        if self.implex:
            args.append("-implex")
        if self.auto_regularize:
            args += ["-autoRegularization", self.lch_ref]
        emitter.uniaxialMaterial("ASDConcrete1D", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Ladruno fork — bond-slip (the axial law of LadrunoEmbeddedRebar)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoBondSlip(UniaxialMaterial):
    r"""``uniaxialMaterial LadrunoBondSlip`` — 1D bond-slip :math:`\tau`–s law.

    OpenSees command (Ladruno fork, ``MAT_TAG`` **33002**)::

        uniaxialMaterial LadrunoBondSlip tag tau_max s1 s2 s3 tau_f alpha \
            [-Gf Gf] [-s0 s0]

    The CEB-FIP / Model Code 2010 monotonic backbone used as the axial
    slot of :class:`LadrunoEmbeddedRebar`: an optional initial linear
    segment for ``s < s0`` (kills the power-law ``dtau/ds -> inf``
    singularity at the origin), an ascending power law
    ``tau_max·(s/s1)^alpha`` up to ``s1``, a plateau to ``s2``, linear
    softening to the residual ``tau_f`` at ``s3``, then residual ``tau_f``.
    ``-Gf`` regularizes the post-peak softening by a bond fracture energy
    (mirrors ``ASDConcrete1D``'s ``lch_ref/lch`` scaling) so the response
    stays mesh-objective.

    This is a **stress** law (:math:`\tau` vs slip): the embedded-rebar
    element converts it to a nodal force with
    ``F = tau · bondScale`` where ``bondScale = perimeter · L_trib =
    pi·d_b · L_trib`` (the ``g.reinforce`` generator computes this).

    .. warning::
       Past ``tau_max`` the backbone has a **negative tangent** — load
       control diverges. Use ``DisplacementControl`` / ``ArcLength`` /
       IMPLEX on the softening branch (ADR 20 D4.2).

    .. note::
       Fork-only. Emission produces a deck line on any build; the
       material is unavailable on stock ``openseespy`` and bites only at
       ``ops.run()`` (a "requires the Ladruno fork build" error).

    Parameters
    ----------
    tau_max
        Peak bond stress (must be > 0).
    s1, s2, s3
        Slip at peak, end-of-plateau, and residual onset. Must satisfy
        ``0 < s1 <= s2 <= s3``.
    tau_f
        Residual bond stress (``0 <= tau_f <= tau_max``).
    alpha
        Ascending-branch power-law exponent (``0 < alpha <= 1``; MC2010
        well-confined value ``0.4``).
    Gf
        Optional bond fracture energy for softening regularization
        (``-Gf``; must be > 0 if supplied).
    s0
        Optional initial-linear-segment slip (``-s0``; must satisfy
        ``0 < s0 < s1``). Omitted -> the fork default (``~0.1·s1``).
    """

    tau_max: float
    s1: float
    s2: float
    s3: float
    tau_f: float
    alpha: float
    Gf: float | None = None
    s0: float | None = None

    def __post_init__(self) -> None:
        if self.tau_max <= 0:
            raise ValueError(
                f"LadrunoBondSlip: tau_max must be > 0, got {self.tau_max!r}"
            )
        if not (0.0 < self.s1 <= self.s2 <= self.s3):
            raise ValueError(
                "LadrunoBondSlip: slips must satisfy 0 < s1 <= s2 <= s3, "
                f"got s1={self.s1!r}, s2={self.s2!r}, s3={self.s3!r}"
            )
        if not (0.0 <= self.tau_f <= self.tau_max):
            raise ValueError(
                "LadrunoBondSlip: tau_f must be in [0, tau_max], got "
                f"tau_f={self.tau_f!r}, tau_max={self.tau_max!r}"
            )
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(
                f"LadrunoBondSlip: alpha must be in (0, 1], got {self.alpha!r}"
            )
        if self.Gf is not None and self.Gf <= 0:
            raise ValueError(
                f"LadrunoBondSlip: Gf must be > 0, got {self.Gf!r}"
            )
        if self.s0 is not None and not (0.0 < self.s0 < self.s1):
            raise ValueError(
                "LadrunoBondSlip: s0 must satisfy 0 < s0 < s1, got "
                f"s0={self.s0!r}, s1={self.s1!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | str] = [
            self.tau_max, self.s1, self.s2, self.s3, self.tau_f, self.alpha,
        ]
        if self.Gf is not None:
            args += ["-Gf", self.Gf]
        if self.s0 is not None:
            args += ["-s0", self.s0]
        emitter.uniaxialMaterial("LadrunoBondSlip", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Ladruno fork — uniaxial J2 twin (combined Voce + Chaboche hardening)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoUniaxialJ2(UniaxialMaterial):
    r"""``uniaxialMaterial LadrunoUniaxialJ2`` — 1D combined-hardening J2.

    OpenSees command (Ladruno fork, ``MAT_TAG`` **33000**)::

        uniaxialMaterial LadrunoUniaxialJ2 tag E \
            -iso voce sig0 Qinf b Hiso \
            [-kin N C1 g1 ...] [-damage lemaitre r s pD Dc] [-implex]

    The 1-D twin of :class:`apeGmsh.opensees.material.nd.LadrunoJ2`: a
    rate-independent uniaxial J2 ``UniaxialMaterial`` with the same Voce +
    linear isotropic and Chaboche kinematic hardening, consumable by
    **fiber sections, trusses, and zeroLength** elements. It delivers true
    multi-backstress ratcheting (what Menegotto-Pinto / Steel02 cannot) and
    is the calibration oracle for a 3-D ``LadrunoJ2`` model.

    Unlike the 3-D :class:`LadrunoJ2`, the uniaxial twin has **no** ``-rho``
    and **no** ``-autoRegularization`` flags (the fork parser rejects them
    here).

    .. note::
       Fork-only. Emission works on any build; the material errors at
       ``ops.run()`` on stock ``openseespy``.

    Parameters
    ----------
    E
        Young's modulus (must be > 0).
    sig0
        Initial yield stress (Voce ``sigma_0``; must be > 0).
    Qinf, b, Hiso
        Voce saturation stress, saturation rate (``>= 0``), and linear
        isotropic hardening modulus (default ``0.0``).
    backstresses
        Chaboche kinematic backstress pairs ``[(C1, gamma1), ...]`` — at
        most 8 (the fork ``MAXBACK``). Each ``C_k > 0``, ``gamma_k >= 0``.
        Empty (default) = pure isotropic.
    damage
        Optional Lemaitre ductile-damage parameters ``(r, s, pD, Dc)``
        (``-damage lemaitre``; ``r > 0``, ``0 < Dc <= 1``). ``None`` = off.
    implex
        Emit ``-implex`` for IMPL-EX integration (SPD tangent).
    """

    E: float
    sig0: float
    Qinf: float = 0.0
    b: float = 0.0
    Hiso: float = 0.0
    backstresses: tuple[tuple[float, float], ...] = ()
    damage: tuple[float, float, float, float] | None = None
    implex: bool = False

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"LadrunoUniaxialJ2: E must be > 0, got {self.E!r}")
        _lj2.validate_iso(
            "LadrunoUniaxialJ2", self.sig0, self.Qinf, self.b, self.Hiso
        )
        _lj2.validate_backstresses("LadrunoUniaxialJ2", self.backstresses)
        if self.damage is not None:
            _lj2.validate_lemaitre("LadrunoUniaxialJ2", self.damage)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | int | str] = [self.E]
        args += _lj2.iso_args(self.sig0, self.Qinf, self.b, self.Hiso)
        args += _lj2.kin_args(self.backstresses)
        if self.damage is not None:
            args += _lj2.lemaitre_args(self.damage)
        if self.implex:
            args.append("-implex")
        emitter.uniaxialMaterial("LadrunoUniaxialJ2", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Ladruno fork — reinforcing-bar buckling overlay (stress-modifying wrapper)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoRebarBuckling(UniaxialMaterial):
    r"""``uniaxialMaterial LadrunoRebarBuckling`` — reinforcing-bar buckling overlay.

    OpenSees command (Ladruno fork, ``MAT_TAG`` **33001**)::

        uniaxialMaterial LadrunoRebarBuckling tag matTag \
            [-lsr s_over_d] [-model dm|ga] [-alpha a] [-reduction r] \
            [-fsufrac g] [-fy fy] [-E E] [-restraighten lambda | -restraighten c c]

    A **stress-modifying wrapper** around *any* tension-compression
    ``UniaxialMaterial`` (typically :class:`LadrunoUniaxialJ2`, also
    ``Steel02``/``Steel4``). In compression past a slenderness-dependent
    onset it applies a reinforcing-bar **buckling-average** degradation
    (``sigma_buckled = r(e, lambda) · sigma_bare``) while leaving the wrapped
    material byte-untouched — an opt-in geometric overlay (``lsr=0`` is the
    identity gate). Two backbone models: Dhakal-Maekawa (``dm``) and
    Gomes-Appleton (``ga``).

    .. note::
       Fork-only. Emission works on any build; errors at ``ops.run()`` on
       stock ``openseespy``.

    Parameters
    ----------
    material
        The wrapped tension-compression :class:`UniaxialMaterial` (the bar
        steel). Emitted before the wrapper (via :meth:`dependencies`).
    lsr
        Bar slenderness ``s/d`` (tie spacing / bar diameter). ``0.0``
        (default) is the **identity gate** — the wrapper passes the bare
        stress through. Must be ``>= 0``.
    model
        Buckling backbone — ``"dm"`` (Dhakal-Maekawa, default) or ``"ga"``
        (Gomes-Appleton).
    alpha
        DM residual-shape factor (default ``1.0``).
    reduction
        GA blend factor in ``[0, 1]`` (default ``0.0``).
    fsu_frac
        GA ultimate-stress fraction (default ``0.5``).
    fy
        Yield stress (required ``> 0`` for ``model="dm"`` when ``lsr > 0``).
        Default ``0.0``.
    E
        Modulus (required ``> 0`` when ``lsr > 0``; ``0.0`` defers to the
        wrapped bar's initial tangent). Default ``0.0``.
    restraighten
        Cyclic re-straightening control — ``None`` (default; the fork's
        built-in ``c=1.0`` mode), ``"lambda"`` (``-restraighten lambda``),
        or ``"c"`` (``-restraighten c`` with :attr:`restraighten_c`).
    restraighten_c
        The ``c`` coefficient emitted when ``restraighten="c"``
        (default ``1.0``).
    """

    material: UniaxialMaterial
    lsr: float = 0.0
    model: str = "dm"
    alpha: float = 1.0
    reduction: float = 0.0
    fsu_frac: float = 0.5
    fy: float = 0.0
    E: float = 0.0
    restraighten: str | None = None
    restraighten_c: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.material, UniaxialMaterial):
            raise TypeError(
                "LadrunoRebarBuckling: material must be a UniaxialMaterial "
                f"primitive, got {type(self.material).__name__!r}."
            )
        if self.model not in _REBAR_BUCKLING_MODELS:
            raise ValueError(
                f"LadrunoRebarBuckling: model must be one of "
                f"{_REBAR_BUCKLING_MODELS}, got {self.model!r}."
            )
        if (
            self.restraighten is not None
            and self.restraighten not in _REBAR_RESTRAIGHTEN_MODES
        ):
            raise ValueError(
                f"LadrunoRebarBuckling: restraighten must be one of "
                f"{_REBAR_RESTRAIGHTEN_MODES} or None, got "
                f"{self.restraighten!r}."
            )
        if self.lsr < 0:
            raise ValueError(
                f"LadrunoRebarBuckling: lsr must be >= 0, got {self.lsr!r}."
            )
        if not (0.0 <= self.reduction <= 1.0):
            raise ValueError(
                "LadrunoRebarBuckling: reduction must be in [0, 1], got "
                f"{self.reduction!r}."
            )
        # The fork rejects these combinations after parsing (the buckling
        # overlay needs a modulus, and the DM model needs a yield stress).
        if self.lsr > 0 and self.E <= 0:
            raise ValueError(
                "LadrunoRebarBuckling: E must be > 0 when lsr > 0 (the "
                "buckling overlay needs a modulus)."
            )
        if self.lsr > 0 and self.model == "dm" and self.fy <= 0:
            raise ValueError(
                "LadrunoRebarBuckling: fy must be > 0 for model='dm' when "
                "lsr > 0 (the Dhakal-Maekawa backbone needs a yield stress)."
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        mat_tag = resolve_tag(emitter, self.material)
        args: list[float | int | str] = [mat_tag]
        if self.lsr != 0.0:
            args += ["-lsr", self.lsr]
        if self.model != "dm":
            args += ["-model", self.model]
        if self.alpha != 1.0:
            args += ["-alpha", self.alpha]
        if self.reduction != 0.0:
            args += ["-reduction", self.reduction]
        if self.fsu_frac != 0.5:
            args += ["-fsufrac", self.fsu_frac]
        if self.fy != 0.0:
            args += ["-fy", self.fy]
        if self.E != 0.0:
            args += ["-E", self.E]
        if self.restraighten == "c":
            args += ["-restraighten", "c", self.restraighten_c]
        elif self.restraighten == "lambda":
            args += ["-restraighten", "lambda"]
        emitter.uniaxialMaterial("LadrunoRebarBuckling", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)


# ---------------------------------------------------------------------------
# Ladruno fork — discrete cohesive moment-rotation hinge law
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoCohesiveHinge(UniaxialMaterial):
    r"""``uniaxialMaterial LadrunoCohesiveHinge`` — cohesive moment-rotation hinge.

    OpenSees command (Ladruno fork, ``MAT_TAG`` **33003**)::

        uniaxialMaterial LadrunoCohesiveHinge tag Mc Gf \
            [-penalty K] [-penaltyRatio r] [-exp | -linear]

    The discrete cohesive rotation-jump law ``M([[theta]])`` that carries a
    fracture energy ``Gf`` per hinge, used as the embedded strong-discontinuity
    hinge of ``LadrunoDispBeamColumn`` (``-hinge``). A rigid pre-peak penalty
    (guarded so it never falls below the snapback floor ``Mc^2/(2 Gf)``)
    followed by an exponential (default) or linear softening envelope
    calibrated so ``∫ M d[[theta]] == Gf``.

    .. note::
       Fork-only. Emission produces a deck line on any build; the material is
       unavailable on stock ``openseespy`` and bites only at ``ops.run()``.

    Parameters
    ----------
    Mc
        Cohesive (peak) moment capacity. Must be > 0.
    Gf
        Fracture energy per hinge (area under ``M``–``[[theta]]``). Must be > 0.
    penalty
        Pre-peak penalty stiffness ``-penalty`` (must be > 0 if supplied).
        ``None`` (default) lets the fork build it as
        ``penalty_ratio * Mc^2/(2 Gf)``.
    penalty_ratio
        Multiplier on the snapback-floor penalty when ``penalty`` is auto
        (``-penaltyRatio``; default 1000, must be > 0). Inert when an explicit
        ``penalty`` is given.
    softening
        Softening envelope shape: ``"exponential"`` (default, ``-exp``) or
        ``"linear"`` (``-linear``).
    """

    _SOFTENING: ClassVar[frozenset[str]] = frozenset({"exponential", "linear"})

    Mc: float
    Gf: float
    penalty: float | None = None
    penalty_ratio: float = 1000.0
    softening: str = "exponential"

    def __post_init__(self) -> None:
        if self.Mc <= 0:
            raise ValueError(f"LadrunoCohesiveHinge: Mc must be > 0, got {self.Mc!r}")
        if self.Gf <= 0:
            raise ValueError(f"LadrunoCohesiveHinge: Gf must be > 0, got {self.Gf!r}")
        if self.penalty is not None and self.penalty <= 0:
            raise ValueError(
                "LadrunoCohesiveHinge: penalty must be > 0 if supplied, got "
                f"{self.penalty!r}"
            )
        if self.penalty_ratio <= 0:
            raise ValueError(
                "LadrunoCohesiveHinge: penalty_ratio must be > 0, got "
                f"{self.penalty_ratio!r}"
            )
        if self.softening not in self._SOFTENING:
            raise ValueError(
                "LadrunoCohesiveHinge: softening must be one of "
                f"{sorted(self._SOFTENING)}, got {self.softening!r}"
            )

    def _emit(self, emitter: Emitter, tag: int) -> None:
        args: list[float | str] = [self.Mc, self.Gf]
        if self.penalty is not None:
            args += ["-penalty", self.penalty]
        if self.penalty_ratio != 1000.0:
            args += ["-penaltyRatio", self.penalty_ratio]
        if self.softening == "linear":
            args.append("-linear")
        emitter.uniaxialMaterial("LadrunoCohesiveHinge", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
