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
"""
from __future__ import annotations


__all__ = [
    "default_ft",
    "ceb_fip_Gf",
    "ceb_fip_Gc",
    "auto_lch_ref",
    "make_tension",
    "make_compression",
    "l_max",
]


def default_ft(fc: float) -> float:
    """Default tensile strength when unspecified (native ``-fc`` rule)."""
    return 0.1 * fc


def ceb_fip_Gf(fc: float) -> float:
    """CEB-FIP tensile fracture energy per area: ``Gf = 0.073 fc^0.18``."""
    return 0.073 * fc ** 0.18


def ceb_fip_Gc(fc: float, ft: float, Gf: float) -> float:
    """Compressive fracture energy per area: ``Gc = 2 Gf (fc/ft)^2``."""
    return 2.0 * Gf * (fc * fc) / (ft * ft)


def auto_lch_ref(E: float, fc: float, ft: float, Gf: float, Gc: float) -> float:
    """Native ``-fc`` self-derived reference length ``min(hmin_t, hmin_c)``.

    The smallest band width below which the regularized softening would
    snap back; used by the binary when ``lch_ref`` is unspecified
    (``cpp:613-630``).
    """
    et_el = ft / E
    Gt_min = 0.5 * ft * et_el
    hmin_t = 0.01 * Gf / Gt_min

    ec = 2.0 * fc / E
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
    t = (D ** 0.5 - B) / (2.0 * A)
    return (y0 - 2.0 * y1 + y2) * t * t + 2.0 * (y1 - y0) * t + y0


def make_tension(
    E: float, ft: float, Gf: float, lch_ref: float,
) -> tuple[list[float], list[float], list[float]]:
    """Tension backbone ``(Te, Ts, Td)`` — 6 points (``cpp:632-667``).

    ``Gf`` is the tensile fracture energy per area; specific energy is
    ``Gf/lch_ref``.
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
    for i in range(2, 6):
        xi, si, xipl = Te[i], Ts[i], Tpl[i]
        xipl = min(xipl, xi - si / E)
        qi = (xi - xipl) * E
        Td[i] = max(0.0, 1.0 - si / qi)  # clamp FP undershoot at d~0
    return Te, Ts, Td


def make_compression(
    E: float, fc: float, Gc: float, lch_ref: float,
) -> tuple[list[float], list[float], list[float]]:
    """Compression backbone ``(Ce, Cs, Cd)`` — 13 points (``cpp:672-711``).

    ``Gc`` is the compressive fracture energy per area; specific energy is
    ``Gc/lch_ref``.
    """
    Gc_s = Gc / lch_ref  # per-area -> specific
    ec = 2.0 * fc / E
    fc0 = 0.5 * fc
    ec0 = fc0 / E
    ec1 = fc / E
    fcr = 0.1 * fc
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

    Cd = [0.0] * n
    for i in range(2, n):
        xi, si, xipl = Ce[i], Cs[i], Cpl[i]
        xipl = min(xipl, xi - si / E)
        qi = (xi - xipl) * E
        Cd[i] = max(0.0, 1.0 - si / qi)  # clamp FP undershoot at d~0
    return Ce, Cs, Cd


def l_max(E: float, Gf: float, ft: float) -> float:
    """Crack-band snapback element-size ceiling ``2 E Gf / ft^2``.

    Above this band width the regularizer floors fracture energy at the
    brittle limit (``cpp:961-965``) and the response is no longer
    mesh-objective. Independent of ``lch_ref`` (ADR 0044, Fact 2).
    """
    return 2.0 * E * Gf / (ft * ft)
