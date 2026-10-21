"""
Typed primitives for the site-response soil ``nDMaterial`` family.

* ``PressureIndependMultiYield`` — total-stress multi-yield clay (Yang, Elgamal).
* ``PressureDependMultiYield`` / ``PressureDependMultiYield02`` — effective-
  stress multi-yield sand with dilatancy and cyclic mobility.
* ``PM4Sand`` / ``PM4Silt`` — Boulanger and Ziotopoulou's plane-strain
  bounding-surface sand and silt (2-D only).

Each class mirrors its OpenSees parser argument for argument (the parser
sources are cited per class) and emits the full positional list, so a
field never shifts position and the declaration archives with its field
names. Values are in the model's units: the pressure defaults
(``ref_pressure``, ``atm``, ``P_atm``) are OpenSees' kPa defaults, so a
model in other units must set them.

All five honour ``updateMaterialStage`` (stage 0 elastic, 1 plastic; the
pressure-dependent pair also takes stage 2, elastic with
pressure-dependent moduli), so the staged gravity idiom is
``s.update_material_stage(materials=[...], stage=...)``.

The three multi-yield classes compute no stress in ``setTrialStrain``:
they do it in ``getStress()``, and ``commitState`` commits whatever that
last cached. An ``algorithm Linear`` step commits without a second
residual, so it commits a stale stress and the response collapses
(about 2 % of the Newton answer on a 1-D column). The bridge refuses that
pairing at build; use Newton, ``ModifiedNewton`` (``-initial`` for a
cheap stage-0 run) or KrylovNewton.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .._internal.types import NDMaterial, Primitive
from ..emitter.base import Emitter


__all__ = [
    "PressureIndependMultiYield",
    "PressureDependMultiYield",
    "PressureDependMultiYield02",
    "PM4Sand",
    "PM4Silt",
    "MULTI_YIELD_SOIL_CLASSES",
]

#: The classes whose ``commitState`` commits the stress cached by the last
#: ``getStress()`` call (``PressureIndependMultiYield.cpp:326-353`` /
#: ``:493-511`` / ``:558-570`` and the same pattern in both
#: ``PressureDependMultiYield*.cpp``), so a single-solve ``Linear`` step
#: commits a stale stress.
MULTI_YIELD_SOIL_CLASSES: frozenset[str] = frozenset({
    "PressureIndependMultiYield",
    "PressureDependMultiYield",
    "PressureDependMultiYield02",
})

Backbone = tuple[tuple[float, float], ...]


def _check_positive(cls: str, **values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{cls}: {name} must be > 0, got {value!r}")


def _check_nonneg(cls: str, **values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{cls}: {name} must be >= 0, got {value!r}")


def _check_nd(cls: str, nd: int) -> None:
    if isinstance(nd, bool) or nd not in (2, 3):
        raise ValueError(f"{cls}: nd must be 2 or 3, got {nd!r}")


def _check_surfaces(
    cls: str, n_surf: int, backbone: "Backbone | None", *, max_pairs: int,
    max_surf: int,
) -> None:
    """``n_surf`` automatic surfaces, or a user backbone of
    ``(shear strain, G/Gmax)`` pairs (emitted as ``-n`` then the pairs)."""
    if backbone is None:
        if isinstance(n_surf, bool) or not isinstance(n_surf, int) \
                or not 1 <= n_surf <= max_surf:
            raise ValueError(
                f"{cls}: n_surf must be an int in 1..{max_surf}, got "
                f"{n_surf!r} (OpenSees silently resets other values)."
            )
        return
    pairs = tuple(backbone)
    if not 1 <= len(pairs) < max_pairs:
        raise ValueError(
            f"{cls}: backbone must hold 1..{max_pairs - 1} (shear strain, "
            f"G/Gmax) pairs, got {len(pairs)}."
        )
    last = 0.0
    for k, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(
                f"{cls}: backbone[{k}] must be a (shear strain, G/Gmax) "
                f"pair, got {pair!r}."
            )
        strain, ratio = float(pair[0]), float(pair[1])
        if not (math.isfinite(strain) and strain > last):
            raise ValueError(
                f"{cls}: backbone shear strains must be finite, positive "
                f"and strictly increasing; backbone[{k}] = {pair!r}."
            )
        if not (math.isfinite(ratio) and 0 < ratio <= 1):
            raise ValueError(
                f"{cls}: backbone G/Gmax must be in (0, 1]; backbone[{k}] "
                f"= {pair!r}."
            )
        last = strain


def _surface_args(n_surf: int, backbone: "Backbone | None") -> list[float | int]:
    if backbone is None:
        return [n_surf]
    out: list[float | int] = [-len(backbone)]
    for strain, ratio in backbone:
        out += [float(strain), float(ratio)]
    return out


# ---------------------------------------------------------------------------
# PressureIndependMultiYield
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class PressureIndependMultiYield(NDMaterial):
    r"""``nDMaterial PressureIndependMultiYield`` — total-stress clay.

    Tcl signature (``OPS_PressureIndependMultiYield``,
    ``PressureIndependMultiYield.cpp:38-112``)::

        nDMaterial PressureIndependMultiYield $tag $nd $rho $refShearModul \
            $refBulkModul $cohesi $peakShearStra $frictionAng $refPress \
            $pressDependCoe $noYieldSurf <$r1 $Gs1 ...>

    The parser reads the eight doubles and ``noYieldSurf`` as one block,
    so all are emitted. A ``backbone`` replaces the automatic surfaces:
    ``noYieldSurf`` is emitted as minus the number of pairs, followed by
    the ``(shear strain, G/Gmax)`` pairs.

    Parameters
    ----------
    nd
        2 (plane strain) or 3.
    rho
        Saturated mass density. ``>= 0``.
    ref_shear_modulus, ref_bulk_modulus
        Low-strain moduli at ``ref_pressure``. ``> 0``.
    cohesion
        Apparent cohesion (the undrained strength of a ``friction_angle=0``
        clay). ``>= 0``, and ``> 0`` when ``friction_angle`` is 0.
    peak_shear_strain
        Octahedral shear strain at which the strength is reached, default
        ``0.1``. ``> 0``.
    friction_angle
        Degrees, default ``0`` (total stress).
    ref_pressure
        Reference mean effective confinement, default ``100`` (kPa).
    press_depend_coe
        Exponent of the moduli's pressure dependence, default ``0``; it
        needs ``friction_angle > 0``.
    n_surf
        Automatic yield surfaces, default ``20`` (1..100).
    backbone
        Optional user modulus-reduction curve, 1..39 ``(shear strain,
        G/Gmax)`` pairs with increasing strain.
    """

    nd: Literal[2, 3]
    rho: float
    ref_shear_modulus: float
    ref_bulk_modulus: float
    cohesion: float
    peak_shear_strain: float = 0.1
    friction_angle: float = 0.0
    ref_pressure: float = 100.0
    press_depend_coe: float = 0.0
    n_surf: int = 20
    backbone: Backbone | None = None

    def __post_init__(self) -> None:
        cls = "PressureIndependMultiYield"
        _check_nd(cls, self.nd)
        _check_nonneg(cls, rho=self.rho, cohesion=self.cohesion,
                      friction_angle=self.friction_angle,
                      press_depend_coe=self.press_depend_coe)
        _check_positive(cls, ref_shear_modulus=self.ref_shear_modulus,
                        ref_bulk_modulus=self.ref_bulk_modulus,
                        peak_shear_strain=self.peak_shear_strain,
                        ref_pressure=self.ref_pressure)
        if self.friction_angle == 0 and self.cohesion <= 0:
            raise ValueError(
                f"{cls}: with friction_angle=0 the cohesion is the strength "
                f"and must be > 0, got {self.cohesion!r}."
            )
        if self.friction_angle >= 90:
            raise ValueError(
                f"{cls}: friction_angle must be < 90 degrees, got "
                f"{self.friction_angle!r}."
            )
        if self.press_depend_coe > 0 and self.friction_angle == 0:
            raise ValueError(
                f"{cls}: press_depend_coe > 0 needs friction_angle > 0 "
                f"(OpenSees silently resets it to 0)."
            )
        _check_surfaces(cls, self.n_surf, self.backbone,
                        max_pairs=40, max_surf=100)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "PressureIndependMultiYield", tag, int(self.nd), self.rho,
            self.ref_shear_modulus, self.ref_bulk_modulus, self.cohesion,
            self.peak_shear_strain, self.friction_angle, self.ref_pressure,
            self.press_depend_coe,
            *_surface_args(self.n_surf, self.backbone),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# PressureDependMultiYield / PressureDependMultiYield02
# ---------------------------------------------------------------------------

def _check_pdmy_common(cls: str, m: "PressureDependMultiYield | PressureDependMultiYield02") -> None:
    _check_nd(cls, m.nd)
    _check_nonneg(cls, rho=m.rho, press_depend_coe=m.press_depend_coe,
                  cohesion=m.cohesion, e=m.e, vol_limit1=m.vol_limit1)
    _check_positive(cls, ref_shear_modulus=m.ref_shear_modulus,
                    ref_bulk_modulus=m.ref_bulk_modulus,
                    peak_shear_strain=m.peak_shear_strain,
                    ref_pressure=m.ref_pressure, atm=m.atm,
                    friction_angle=m.friction_angle, pt_angle=m.pt_angle)
    if m.friction_angle >= 90:
        raise ValueError(
            f"{cls}: friction_angle must be < 90 degrees, got "
            f"{m.friction_angle!r}."
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class PressureDependMultiYield(NDMaterial):
    r"""``nDMaterial PressureDependMultiYield`` — effective-stress sand.

    Tcl signature (``OPS_PressureDependMultiYield``,
    ``PressureDependMultiYield.cpp:74-187``)::

        nDMaterial PressureDependMultiYield $tag $nd $rho $refShearModul \
            $refBulkModul $frictionAng $peakShearStra $refPress \
            $pressDependCoe $PTAng $contrac $dilat1 $dilat2 $liquefac1 \
            $liquefac2 $liquefac4 <$noYieldSurf <$r $Gs ...> $e $cs1 $cs2 \
            $cs3 $pa <$c <$hv $pv>>>

    Fifteen required values, then the optional block, emitted in full
    with OpenSees' defaults. ``backbone`` replaces the automatic surfaces
    as for :class:`PressureIndependMultiYield` (1..39 pairs). Stage 2
    (``updateMaterialStage ... 2``) is elastic with pressure-dependent
    moduli.

    Parameters
    ----------
    nd, rho, ref_shear_modulus, ref_bulk_modulus, peak_shear_strain,
    ref_pressure, press_depend_coe
        As for :class:`PressureIndependMultiYield`; ``friction_angle`` is
        required here and ``> 0``.
    pt_angle
        Phase-transformation angle, degrees, ``> 0``.
    contrac, dilat1, dilat2
        Contraction and dilation parameters.
    liquefac1, liquefac2, liquefac4
        Cyclic-mobility (liquefaction) parameters.
    n_surf, backbone
        Yield surfaces (default 20, at most 40) or a user curve.
    e
        Initial void ratio, default ``0.6``.
    vol_limit1, vol_limit2, vol_limit3
        Critical-state line parameters, defaults ``0.9``, ``0.02``,
        ``0.7``.
    atm
        Atmospheric pressure, default ``101`` (kPa).
    cohesion
        Apparent cohesion, default ``0.3``.
    hv, pv
        Viscoplastic-style shear-induced volume parameters, defaults
        ``0`` and ``1``.
    """

    nd: Literal[2, 3]
    rho: float
    ref_shear_modulus: float
    ref_bulk_modulus: float
    friction_angle: float
    peak_shear_strain: float
    ref_pressure: float
    press_depend_coe: float
    pt_angle: float
    contrac: float
    dilat1: float
    dilat2: float
    liquefac1: float
    liquefac2: float
    liquefac4: float
    n_surf: int = 20
    backbone: Backbone | None = None
    e: float = 0.6
    vol_limit1: float = 0.9
    vol_limit2: float = 0.02
    vol_limit3: float = 0.7
    atm: float = 101.0
    cohesion: float = 0.3
    hv: float = 0.0
    pv: float = 1.0

    def __post_init__(self) -> None:
        cls = "PressureDependMultiYield"
        _check_pdmy_common(cls, self)
        if self.pt_angle > self.friction_angle:
            raise ValueError(
                f"{cls}: pt_angle ({self.pt_angle!r}) must not exceed "
                f"friction_angle ({self.friction_angle!r})."
            )
        _check_surfaces(cls, self.n_surf, self.backbone,
                        max_pairs=40, max_surf=40)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "PressureDependMultiYield", tag, int(self.nd), self.rho,
            self.ref_shear_modulus, self.ref_bulk_modulus,
            self.friction_angle, self.peak_shear_strain, self.ref_pressure,
            self.press_depend_coe, self.pt_angle, self.contrac, self.dilat1,
            self.dilat2, self.liquefac1, self.liquefac2, self.liquefac4,
            *_surface_args(self.n_surf, self.backbone),
            self.e, self.vol_limit1, self.vol_limit2, self.vol_limit3,
            self.atm, self.cohesion, self.hv, self.pv,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class PressureDependMultiYield02(NDMaterial):
    r"""``nDMaterial PressureDependMultiYield02`` — the 2008 revision.

    Tcl signature (``OPS_PressureDependMultiYield02``,
    ``PressureDependMultiYield02.cpp:77-185``)::

        nDMaterial PressureDependMultiYield02 $tag $nd $rho $refShearModul \
            $refBulkModul $frictionAng $peakShearStra $refPress \
            $pressDependCoe $PTAng $contrac1 $contrac3 $dilat1 $dilat3 \
            <$noYieldSurf <$r $Gs ...> $contrac2 $dilat2 $liquefac1 \
            $liquefac2 $e $cs1 $cs2 $cs3 $pa <$c <$hv $pv>>>

    Thirteen required values, then the optional block, emitted in full
    with OpenSees' defaults. ``backbone``: 1..99 pairs. Unlike
    :class:`PressureDependMultiYield`, the parser does not check
    ``pt_angle <= friction_angle``.

    Parameters
    ----------
    contrac1, contrac3, dilat1, dilat3
        Required contraction / dilation parameters.
    contrac2, dilat2, liquefac1, liquefac2
        Defaults ``5.0``, ``3.0``, ``1.0``, ``0.0``.
    cohesion
        Default ``0.1``.

    The rest are as for :class:`PressureDependMultiYield` (``n_surf`` up
    to 100).
    """

    nd: Literal[2, 3]
    rho: float
    ref_shear_modulus: float
    ref_bulk_modulus: float
    friction_angle: float
    peak_shear_strain: float
    ref_pressure: float
    press_depend_coe: float
    pt_angle: float
    contrac1: float
    contrac3: float
    dilat1: float
    dilat3: float
    n_surf: int = 20
    backbone: Backbone | None = None
    contrac2: float = 5.0
    dilat2: float = 3.0
    liquefac1: float = 1.0
    liquefac2: float = 0.0
    e: float = 0.6
    vol_limit1: float = 0.9
    vol_limit2: float = 0.02
    vol_limit3: float = 0.7
    atm: float = 101.0
    cohesion: float = 0.1
    hv: float = 0.0
    pv: float = 1.0

    def __post_init__(self) -> None:
        cls = "PressureDependMultiYield02"
        _check_pdmy_common(cls, self)
        _check_surfaces(cls, self.n_surf, self.backbone,
                        max_pairs=100, max_surf=100)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "PressureDependMultiYield02", tag, int(self.nd), self.rho,
            self.ref_shear_modulus, self.ref_bulk_modulus,
            self.friction_angle, self.peak_shear_strain, self.ref_pressure,
            self.press_depend_coe, self.pt_angle, self.contrac1,
            self.contrac3, self.dilat1, self.dilat3,
            *_surface_args(self.n_surf, self.backbone),
            self.contrac2, self.dilat2, self.liquefac1, self.liquefac2,
            self.e, self.vol_limit1, self.vol_limit2, self.vol_limit3,
            self.atm, self.cohesion, self.hv, self.pv,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# PM4Sand / PM4Silt
# ---------------------------------------------------------------------------

def _check_pm4_tail(cls: str, int_scheme: int, tan_type: int,
                    tol_f: float, tol_r: float) -> None:
    for name, value in (("int_scheme", int_scheme), ("tan_type", tan_type)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{cls}: {name} must be an int >= 0, got {value!r}")
    _check_positive(cls, tol_f=tol_f, tol_r=tol_r)


def _check_computed(cls: str, **values: float) -> None:
    """A field whose ``-1`` default asks the material to compute it."""
    for name, value in values.items():
        if not (math.isfinite(value) and (value == -1 or value > 0)):
            raise ValueError(
                f"{cls}: {name} must be > 0, or -1 for the value the "
                f"material computes, got {value!r}"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class PM4Sand(NDMaterial):
    r"""``nDMaterial PM4Sand`` — plane-strain sand plasticity (v3.1).

    Tcl signature (``OPS_PM4SandMaterial``, ``PM4Sand.cpp:71-136``)::

        nDMaterial PM4Sand $tag $Dr $G0 $hpo $Den <$P_atm $h0 $e_max \
            $e_min $nb $nd $Ado $z_max $cz $ce $phi_cv $nu $Cgd $Cdr \
            $Ckaf $Q $R $m $Fsed_min $p_sdeo $IntScheme $TanType $TolF \
            $TolR>

    Four required values and the 24 optional ones, emitted in full with
    OpenSees' defaults. A ``-1`` default asks the material for the value
    it computes from ``Dr`` (``h0``, ``Ado``, ``z_max``, ``ce``, ``Cdr``,
    ``Ckaf``, ``Fsed_min``, ``p_sdeo``). 2-D (plane strain) only, so use
    it on quads / triangles. Update ``FirstCall`` / ``PostShake`` with
    ``s.update_parameter(name, value, pg=..., material=sand)``.

    Parameters
    ----------
    Dr
        Relative density, as a fraction (``0 < Dr``).
    G0
        Shear-modulus coefficient. ``> 0``.
    hpo
        Contraction-rate parameter (the calibration knob). ``> 0``.
    rho
        Mass density (Tcl ``$Den``). ``>= 0``.
    """

    Dr: float
    G0: float
    hpo: float
    rho: float
    P_atm: float = 101.3
    h0: float = -1.0
    e_max: float = 0.8
    e_min: float = 0.5
    nb: float = 0.5
    nd: float = 0.1
    Ado: float = -1.0
    z_max: float = -1.0
    cz: float = 250.0
    ce: float = -1.0
    phi_cv: float = 33.0
    nu: float = 0.3
    Cgd: float = 2.0
    Cdr: float = -1.0
    Ckaf: float = -1.0
    Q: float = 10.0
    R: float = 1.5
    m: float = 0.01
    Fsed_min: float = -1.0
    p_sdeo: float = -1.0
    int_scheme: int = 5
    tan_type: int = 0
    tol_f: float = 1.0e-8
    tol_r: float = 1.0e-8

    def __post_init__(self) -> None:
        cls = "PM4Sand"
        _check_positive(cls, Dr=self.Dr, G0=self.G0, hpo=self.hpo,
                        P_atm=self.P_atm, phi_cv=self.phi_cv, cz=self.cz,
                        Q=self.Q, R=self.R, m=self.m, nb=self.nb,
                        nd=self.nd, Cgd=self.Cgd)
        _check_nonneg(cls, rho=self.rho)
        if not self.e_min < self.e_max:
            raise ValueError(
                f"{cls}: e_min ({self.e_min!r}) must be < e_max "
                f"({self.e_max!r})."
            )
        if not 0 <= self.nu < 0.5:
            raise ValueError(f"{cls}: nu must be in [0, 0.5), got {self.nu!r}")
        _check_computed(cls, h0=self.h0, Ado=self.Ado, z_max=self.z_max,
                        ce=self.ce, Cdr=self.Cdr, Ckaf=self.Ckaf,
                        Fsed_min=self.Fsed_min, p_sdeo=self.p_sdeo)
        _check_pm4_tail(cls, self.int_scheme, self.tan_type,
                        self.tol_f, self.tol_r)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "PM4Sand", tag, self.Dr, self.G0, self.hpo, self.rho,
            self.P_atm, self.h0, self.e_max, self.e_min, self.nb, self.nd,
            self.Ado, self.z_max, self.cz, self.ce, self.phi_cv, self.nu,
            self.Cgd, self.Cdr, self.Ckaf, self.Q, self.R, self.m,
            self.Fsed_min, self.p_sdeo, self.int_scheme, self.tan_type,
            self.tol_f, self.tol_r,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class PM4Silt(NDMaterial):
    r"""``nDMaterial PM4Silt`` — plane-strain silt / clay plasticity.

    Tcl signature (``OPS_PM4SiltMaterial``, ``PM4Silt.cpp:72-140``)::

        nDMaterial PM4Silt $tag $S_u $Su_Rat $G_o $h_po $Den <$Su_factor \
            $P_atm $nu $nG $h0 $eInit $lambda $phicv $nb_wet $nb_dry $nd \
            $Ado $ru_max $z_max $cz $ce $cgd $ckaf $m_m $CG_consol \
            $IntScheme $TanType $TolF $TolR>

    Five required values and the 24 optional ones, emitted in full with
    OpenSees' defaults (``-1`` asks the material to compute ``ru_max``,
    ``z_max`` and ``ce``). 2-D (plane strain) only.

    Parameters
    ----------
    Su
        Undrained shear strength, or with ``Su_rate`` the strength ratio
        form (see the OpenSees manual). ``>= 0``.
    Su_rate
        Undrained strength ratio ``Su / sigma'_v``. ``>= 0``.
    G0
        Shear-modulus coefficient. ``> 0``.
    hpo
        Contraction-rate parameter. ``> 0``.
    rho
        Mass density. ``>= 0``.
    """

    Su: float
    Su_rate: float
    G0: float
    hpo: float
    rho: float
    Su_factor: float = 1.0
    P_atm: float = 101.3
    nu: float = 0.3
    nG: float = 0.75
    h0: float = 0.5
    e_init: float = 0.9
    lambda_: float = 0.06
    phi_cv: float = 32.0
    nb_wet: float = 0.8
    nb_dry: float = 0.5
    nd: float = 0.3
    Ado: float = 0.8
    ru_max: float = -1.0
    z_max: float = -1.0
    cz: float = 100.0
    ce: float = -1.0
    Cgd: float = 3.0
    Ckaf: float = 4.0
    m: float = 0.01
    CG_consol: float = 2.0
    int_scheme: int = 5
    tan_type: int = 0
    tol_f: float = 1.0e-7
    tol_r: float = 1.0e-10

    def __post_init__(self) -> None:
        cls = "PM4Silt"
        _check_nonneg(cls, Su=self.Su, Su_rate=self.Su_rate, rho=self.rho)
        if self.Su == 0 and self.Su_rate == 0:
            raise ValueError(f"{cls}: give Su or Su_rate (both are 0).")
        _check_positive(cls, G0=self.G0, hpo=self.hpo, P_atm=self.P_atm,
                        Su_factor=self.Su_factor, e_init=self.e_init,
                        lambda_=self.lambda_, phi_cv=self.phi_cv,
                        cz=self.cz, m=self.m, CG_consol=self.CG_consol)
        if not 0 <= self.nu < 0.5:
            raise ValueError(f"{cls}: nu must be in [0, 0.5), got {self.nu!r}")
        _check_computed(cls, ru_max=self.ru_max, z_max=self.z_max, ce=self.ce)
        _check_pm4_tail(cls, self.int_scheme, self.tan_type,
                        self.tol_f, self.tol_r)

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial(
            "PM4Silt", tag, self.Su, self.Su_rate, self.G0, self.hpo,
            self.rho, self.Su_factor, self.P_atm, self.nu, self.nG, self.h0,
            self.e_init, self.lambda_, self.phi_cv, self.nb_wet,
            self.nb_dry, self.nd, self.Ado, self.ru_max, self.z_max,
            self.cz, self.ce, self.Cgd, self.Ckaf, self.m, self.CG_consol,
            self.int_scheme, self.tan_type, self.tol_f, self.tol_r,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
