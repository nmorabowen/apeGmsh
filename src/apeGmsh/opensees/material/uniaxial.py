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

from dataclasses import dataclass

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
