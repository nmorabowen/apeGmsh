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

from . import _asdconcrete_laws as _laws
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
    "InitialStress",
    "ASDConcrete1D",
]


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

        fcc_ = fcc if fcc is not None else _mander.confined_strength(fc, fl)
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
        for d in (*self.Td, *self.Cd):
            if not (0.0 <= d < 1.0):
                raise ValueError(
                    f"ASDConcrete1D: damage must be in [0, 1), got {d!r}"
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
