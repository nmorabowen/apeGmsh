"""Pure-function backbone generator for ASDConcrete (tension + compression).

This is apeGmsh's OWNED, port of Petracca's reference ``-fc`` law builder
(`ASDConcrete3DMaterial.cpp:345-712`, OpenSees source ``7c92197``). It is
parameterized by the **physical** inputs ``(E, fc, ft, Gf, Gc, lch_ref)`` —
in particular it accepts a user-supplied tensile/compressive fracture energy,
which the native ``-fc`` command CANNOT (that path hard-derives them from
``fc`` via CEB-FIP and exposes no ``-Gf``/``-Gc`` token; see ADR 0044).

The generated ``(Te, Ts, Td, Ce, Cs, Cd)`` points are emitted **verbatim**
as ``-Te/-Ts/-Td/-Ce/-Cs/-Cd``, so the solver integrates exactly this curve —
there is no parity-drift surface (ADR 0044, Fork 1). These functions are also
the read-only ``preview_backbone()`` surface and the substrate for the
invariance / ``l_max`` unit tests.

Fracture energies here are **per unit area** (``Gf``, ``Gc``); the builder
converts to specific (per-volume) energy internally as ``G / lch_ref``,
which is the crack-band reference the regularizer rescales against.

The optional ``fc0`` / ``fcr`` / ``ec`` / ``pscale`` arguments open the same
law to the STKO ``ASDConcrete3D`` presets (``Concrete (1P/4P/6P/9P)``), which
use this algorithm with those four values exposed; their defaults keep the
native ``-fc`` curve unchanged.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
from dataclasses import dataclass
import math
from typing import Any


__all__ = [
    "default_ft",
    "ceb_fip_Gf",
    "ceb_fip_Gc",
    "auto_lch_ref",
    "make_tension",
    "make_compression",
    "l_max",
    "UNIT_PRESETS",
    "AGGREGATE_FACTORS",
    "ConcreteMC2010Params",
    "fib_mc2010_concrete_params",
]


UNIT_PRESETS: dict[str, tuple[float, float]] = {
    "MPa-mm-N": (1.0, 1.0),
    "MPa-mm": (1.0, 1.0),
    "Pa-m-N": (1.0e6, 1.0e-3),
    "Pa-m": (1.0e6, 1.0e-3),
    "kPa-m-kN": (1.0e3, 1.0e-3),
    "kPa-m": (1.0e3, 1.0e-3),
    "psi-in-lbf": (145.03773773, 1.0 / 25.4),
    "psi-in": (145.03773773, 1.0 / 25.4),
    "ksi-in-kip": (0.14503773773, 1.0 / 25.4),
    "ksi-in": (0.14503773773, 1.0 / 25.4),
}

AGGREGATE_FACTORS: dict[str, float] = {
    "basalt": 1.2,
    "quartzite": 1.0,
    "limestone": 0.9,
    "sandstone": 0.7,
}


@dataclass(frozen=True)
class ConcreteMC2010Params(Mapping[str, Any]):
    r"""fib Model Code 2010 / Eurocode 2 / CDPM2 calibrated concrete parameters."""

    E: float
    nu: float
    fc: float
    ft: float
    Gf: float
    Gc: float
    ec: float
    Df: float
    hardening: tuple[float, float]
    ductility: tuple[float, float, float, float]
    As: float
    e: float
    kupfer: float
    units: str
    d_max: float
    aggregate: str
    fck: float
    fcm: float
    l_max_t: float
    l_max_c: float

    @property
    def Gt(self) -> float:
        """Alias for Gf."""
        return self.Gf

    @property
    def eps_c1(self) -> float:
        """Alias for ec."""
        return self.ec

    @property
    def epsc0(self) -> float:
        """Alias for ec (for RC models)."""
        return self.ec

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        fields = [f.name for f in dataclasses.fields(self)]
        yield from fields
        yield "Gt"
        yield "eps_c1"
        yield "epsc0"

    def __len__(self) -> int:
        return len(dataclasses.fields(self)) + 3


def default_ft(fc: float) -> float:
    """Default tensile strength when unspecified (native ``-fc`` rule)."""
    return 0.1 * fc


def ceb_fip_Gf(
    fc: float,
    *,
    units: str = "MPa-mm-N",
    stress_scale: float | None = None,
    length_scale: float | None = None,
) -> float:
    r"""CEB-FIP / fib MC2010 tensile fracture energy per area.

    In fib MC2010 Eq. 5.1-9: :math:`G_f = 0.073 f_{cm}^{0.18}\text{ [N/mm]}`.
    Converted to user units via ``stress_scale * length_scale``
    (:math:`\text{N/mm} \to \text{user force} / \text{user length}`).
    """
    if stress_scale is None or length_scale is None:
        if units in ("MPa-mm-N", "MPa-mm") and fc > 1.0e5:
            units = "Pa-m-N"
        if units not in UNIT_PRESETS:
            raise ValueError(
                f"Unknown units {units!r}, supported: {sorted(UNIT_PRESETS)}"
            )
        s_sig, s_len = UNIT_PRESETS[units]
        stress_scale = s_sig if stress_scale is None else stress_scale
        length_scale = s_len if length_scale is None else length_scale

    fc_mpa = fc / stress_scale
    gf_n_mm = 0.073 * (fc_mpa ** 0.18)
    return float(gf_n_mm * stress_scale * length_scale)


def ceb_fip_Gc(fc: float, ft: float, Gf: float) -> float:
    """Compressive fracture energy per area: ``Gc = 2 Gf (fc/ft)^2``."""
    return 2.0 * Gf * (fc * fc) / (ft * ft)


def fib_mc2010_concrete_params(
    fc: float,
    *,
    units: str = "MPa-mm-N",
    d_max: float | None = None,
    aggregate: str = "quartzite",
) -> ConcreteMC2010Params:
    r"""Derive fib Model Code 2010 / Eurocode 2 / CDPM2 calibrated parameters from fc.

    Parameters
    ----------
    fc : float
        Mean uniaxial compressive cylinder strength in user units (> 0).
    units : str
        Unit system preset ('MPa-mm-N', 'Pa-m-N', 'kPa-m-kN', 'ksi-in-kip', 'psi-in-lbf').
    d_max : float, optional
        Maximum aggregate size in user length units (default 16 mm in user units).
    aggregate : str
        Aggregate type ('quartzite', 'basalt', 'limestone', 'sandstone').
    """
    if fc <= 0:
        raise ValueError(f"fib_mc2010_concrete_params: fc must be > 0, got {fc!r}")
    if units not in UNIT_PRESETS:
        raise ValueError(
            f"fib_mc2010_concrete_params: unknown units {units!r}, supported: {sorted(UNIT_PRESETS)}"
        )
    # If units is default "MPa-mm-N" or alias "MPa-mm" and fc > 1e5, fc is in Pa ("Pa-m-N").
    if units in ("MPa-mm-N", "MPa-mm") and fc > 1.0e5:
        units = "Pa-m-N"
    if aggregate not in AGGREGATE_FACTORS:
        raise ValueError(
            f"fib_mc2010_concrete_params: unknown aggregate {aggregate!r}, supported: {sorted(AGGREGATE_FACTORS)}"
        )
    if d_max is not None and d_max <= 0:
        raise ValueError(f"fib_mc2010_concrete_params: d_max must be > 0, got {d_max!r}")

    s_sig, s_len = UNIT_PRESETS[units]
    s_gf = s_sig * s_len
    alpha_E = AGGREGATE_FACTORS[aggregate]

    # Convert fc to MPa and d_max to mm
    f_cm_mpa = fc / s_sig
    d_max_mm = (d_max / s_len) if d_max is not None else 16.0

    # Characteristic compressive strength fck (MPa)
    f_ck_mpa = max(f_cm_mpa - 8.0, 0.75 * f_cm_mpa)

    # Tensile strength fctm (MPa) per fib MC2010 Eq. 5.1-3a/b
    if f_ck_mpa <= 50.0:
        ft_mpa = 0.30 * (f_ck_mpa ** (2.0 / 3.0))
    else:
        ft_mpa = 2.12 * math.log(1.0 + 0.10 * (f_ck_mpa + 8.0))

    # Young's modulus Eci (MPa) per fib MC2010 Eq. 5.1-20
    ec_mpa = 21500.0 * alpha_E * ((f_cm_mpa / 10.0) ** (1.0 / 3.0))

    # Tensile fracture energy Gf (N/mm) per fib MC2010 Eq. 5.1-9 + aggregate correction
    gf_n_mm = 0.073 * (f_cm_mpa ** 0.18) * ((d_max_mm / 16.0) ** 0.15)

    # Compressive fracture energy Gc (N/mm) per Nakamura & Higai / fib MC2010
    gc_n_mm = 250.0 * gf_n_mm

    # Convert to user units
    fc_u = f_cm_mpa * s_sig
    fck_u = f_ck_mpa * s_sig
    ft_u = ft_mpa * s_sig
    ec_u = ec_mpa * s_sig
    nu = 0.20
    gf_u = gf_n_mm * s_gf
    gc_u = gc_n_mm * s_gf

    # Peak compressive strain eps_c1 (fib MC2010 Table 5.1-8)
    eps0 = fc_u / ec_u
    eps_c1 = max(0.0007 * (f_cm_mpa ** 0.31), 1.6 * eps0)

    # CDPM2 defaults
    Df = 0.85
    hardening = (0.3, 0.01)
    As = 15.0
    kupfer = 1.16
    Bh = max(0.001, eps_c1 - eps0)
    ductility = (0.08, Bh, 2.0, 1.0e-6)

    # Kupfer biaxial eccentricity e in (0.5, 1.0]
    eps_ratio = ft_u / fc_u
    denom = kupfer * kupfer - eps_ratio * eps_ratio
    eps_ecc = eps_ratio * (kupfer * kupfer - 1.0) / denom if denom > 0 else 0.025
    e_ecc = (1.0 + eps_ecc) / (2.0 - eps_ecc)
    e_ecc = max(0.5001, min(1.0, e_ecc))

    # Snap-back ceilings
    hp = hardening[1]  # 0.01
    l_max_t = (20.0 / 21.0) * ec_u * gf_u / (ft_u * ft_u)
    l_max_c = 2.0 * ec_u * gc_u / ((fc_u * fc_u) * (1.0 + hp))

    return ConcreteMC2010Params(
        E=ec_u,
        nu=nu,
        fc=fc_u,
        ft=ft_u,
        Gf=gf_u,
        Gc=gc_u,
        ec=eps_c1,
        Df=Df,
        hardening=hardening,
        ductility=ductility,
        As=As,
        e=e_ecc,
        kupfer=kupfer,
        units=units,
        d_max=d_max_mm * s_len,
        aggregate=aggregate,
        fck=fck_u,
        fcm=fc_u,
        l_max_t=l_max_t,
        l_max_c=l_max_c,
    )


def auto_lch_ref(
    E: float, fc: float, ft: float, Gf: float, Gc: float,
    ec: float | None = None,
) -> float:
    """Native ``-fc`` self-derived reference length ``min(hmin_t, hmin_c)``.

    The smallest band width below which the regularized softening would
    snap back; used by the binary when ``lch_ref`` is unspecified
    (``cpp:613-630``). ``ec`` is the strain at peak compression
    (default ``2 fc / E``).
    """
    et_el = ft / E
    Gt_min = 0.5 * ft * et_el
    hmin_t = 0.01 * Gf / Gt_min

    ec = 2.0 * fc / E if ec is None else ec
    ec1 = fc / E
    ec_pl = (ec - ec1) * 0.4 + ec1
    Gc_min = 0.5 * fc * (ec - ec_pl)
    hmin_c = 0.01 * Gc / Gc_min
    return min(hmin_c, hmin_t)


def _bezier3(
    xi: float,
    x0: float, x1: float, x2: float,
    y0: float, y1: float, y2: float,
) -> float:
    """Quadratic-Bezier ordinate at abscissa ``xi`` (``cpp:345-365``)."""
    A = x0 - 2.0 * x1 + x2
    if abs(A) < 1.0e-12:
        x1 = x1 + 1.0e-6 * (x2 - x0)
        A = x0 - 2.0 * x1 + x2
    if A == 0.0:
        return 0.0
    B = 2.0 * (x1 - x0)
    C = x0 - xi
    D = B * B - 4.0 * A * C
    t = float(D ** 0.5 - B) / (2.0 * A)
    return (y0 - 2.0 * y1 + y2) * t * t + 2.0 * (y1 - y0) * t + y0


def make_tension(
    E: float, ft: float, Gf: float, lch_ref: float,
    pscale: float = 1.0,
) -> tuple[list[float], list[float], list[float]]:
    """Tension backbone ``(Te, Ts, Td)`` — 6 points (``cpp:632-667``).

    ``Gf`` is the tensile fracture energy per area; specific energy is
    ``Gf/lch_ref``. ``pscale`` in ``[0, 1)`` scales the plastic part of the
    inelastic strain (the rest becomes damage); ``1`` keeps the native split.
    """
    Gt = Gf / lch_ref  # per-area -> specific (per-volume)
    f0 = 0.9 * ft
    f1 = ft
    e0 = f0 / E
    e1 = 1.5 * f1 / E
    ep = e1 - f1 / E
    f2 = 0.2 * ft
    f3 = 1.0e-3 * ft
    w2 = Gt / ft
    w3 = 5.0 * w2
    e2 = w2 + f2 / E + ep
    if e2 <= e1:
        e2 = 1.001 * e1
    e3 = w3 + f3 / E + ep
    if e3 <= e2:
        e3 = 1.001 * e2
    e4 = 10.0 * e3

    Te = [0.0, e0, e1, e2, e3, e4]
    Ts = [0.0, f0, f1, f2, f3, f3]
    Td = [0.0] * 6
    Tpl = [0.0, 0.0, ep, 0.9 * e2, 0.8 * e3, 0.8 * e3]
    if 0.0 <= pscale < 1.0:
        Tpl = [pscale * x for x in Tpl]
    for i in range(2, 6):
        xi, si, xipl = Te[i], Ts[i], Tpl[i]
        xipl = min(xipl, xi - si / E)
        qi = (xi - xipl) * E
        d = max(0.0, 1.0 - si / qi)  # clamp FP undershoot at d~0
        Td[i] = min(1.0 - 1e-12, d) if d >= 1.0 else d
    return Te, Ts, Td


def make_compression(
    E: float, fc: float, Gc: float, lch_ref: float,
    fc0: float | None = None, fcr: float | None = None,
    ec: float | None = None, pscale: float = 1.0,
) -> tuple[list[float], list[float], list[float]]:
    """Compression backbone ``(Ce, Cs, Cd)`` — 13 points (``cpp:672-711``).

    ``Gc`` is the compressive fracture energy per area; specific energy is
    ``Gc/lch_ref``. ``fc0`` (end of the linear branch, default ``fc/2``),
    ``fcr`` (residual, default ``fc/10``) and ``ec`` (strain at peak, default
    ``2 fc / E``) shape the curve; ``pscale`` as in :func:`make_tension`.
    """
    Gc_s = Gc / lch_ref  # per-area -> specific
    ec = 2.0 * fc / E if ec is None else ec
    fc0 = 0.5 * fc if fc0 is None else fc0
    ec0 = fc0 / E
    ec1 = fc / E
    fcr = 0.1 * fc if fcr is None else fcr
    ec_pl = (ec - ec1) * 0.4 + ec1
    Gc1 = 0.5 * fc * (ec - ec_pl)
    Gc2 = max(0.01 * Gc1, Gc_s - Gc1)
    ecr = ec + 2.0 * Gc2 / (fc + fcr)

    nc = 10
    n = nc + 3
    Ce = [0.0] * n
    Cs = [0.0] * n
    Cpl = [0.0] * n
    Ce[1], Cs[1] = ec0, fc0
    dec = (ec - ec0) / (nc - 1)
    for i in range(nc - 1):
        iec = ec0 + (i + 1) * dec
        Ce[i + 2] = iec
        Cs[i + 2] = _bezier3(iec, ec0, ec1, ec, fc0, fc, fc)
        Cpl[i + 2] = Cpl[i + 1] + 0.7 * (iec - Cpl[i + 1])
    Ce[nc + 1], Cs[nc + 1] = ecr, fcr
    Cpl[nc + 1] = Cpl[nc] + 0.7 * (ecr - Cpl[nc])
    Ce[nc + 2], Cs[nc + 2] = ecr + ec0, fcr
    Cpl[nc + 2] = Cpl[nc + 1]
    if 0.0 <= pscale < 1.0:
        Cpl = [pscale * x for x in Cpl]

    Cd = [0.0] * n
    for i in range(2, n):
        xi, si, xipl = Ce[i], Cs[i], Cpl[i]
        xipl = min(xipl, xi - si / E)
        qi = (xi - xipl) * E
        d = max(0.0, 1.0 - si / qi)  # clamp FP undershoot at d~0
        Cd[i] = min(1.0 - 1e-12, d) if d >= 1.0 else d
    return Ce, Cs, Cd


def l_max(E: float, Gf: float, ft: float) -> float:
    """Crack-band snapback element-size ceiling ``2 E Gf / ft^2``.

    Above this band width the regularizer floors fracture energy at the
    brittle limit (``cpp:961-965``) and the response is no longer
    mesh-objective. Independent of ``lch_ref`` (ADR 0044, Fact 2).
    """
    return 2.0 * E * Gf / (ft * ft)
