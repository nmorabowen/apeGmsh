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
    "GC_OVER_GF",
    "tension_snapback_factor",
    "rc_regularization_lch_max",
]


#: Compressive-to-tensile fracture-energy ratio used when no ``Gc`` is given.
#: The ~250 ratio is the Nakamura & Higai (2001) estimate (as adopted, e.g.,
#: by the DIANA material library). fib MC2010 gives NO compressive fracture
#: energy, so this is not a Model Code value.
GC_OVER_GF: float = 250.0


#: ``units`` presets: ``(stress per MPa, length per mm)`` in the user system.
#: Units are always explicit; nothing is inferred from the magnitude of ``fc``.
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
    r"""Concrete parameters calibrated from ``fc`` (see :func:`fib_mc2010_concrete_params`).

    Every stress / modulus / energy / length is in the ``units`` system the
    caller declared; strains are dimensionless.
    """

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


def _unit_scales(units: str, who: str) -> tuple[float, float]:
    """``(stress per MPa, length per mm)`` for an explicit ``units`` preset."""
    if units not in UNIT_PRESETS:
        raise ValueError(
            f"{who}: unknown units {units!r}, supported: {sorted(UNIT_PRESETS)}"
        )
    return UNIT_PRESETS[units]


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

    fib MC2010 Eq. 5.1-9: :math:`G_F = 73 f_{cm}^{0.18}` [N/m], i.e.
    :math:`0.073 f_{cm}^{0.18}` [N/mm] with :math:`f_{cm}` in MPa.
    ``fc`` is read in the declared ``units`` (default ``"MPa-mm-N"``, which
    reproduces the historical ``0.073 fc^0.18``); the result is converted to
    user force / user length. Units are never inferred from the magnitude
    of ``fc``.
    """
    if stress_scale is None or length_scale is None:
        s_sig, s_len = _unit_scales(units, "ceb_fip_Gf")
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
    aggregate: str = "quartzite",
) -> ConcreteMC2010Params:
    r"""Calibrate a full concrete parameter set from the mean strength ``fc``.

    Sources, per quantity (``fcm = fc`` in MPa, ``fck = fcm - 8``, floored
    at ``0.75 fcm`` for very weak concrete):

    * ``ft`` = :math:`f_{ctm}`: fib MC2010 Eq. 5.1-3a/b
      (:math:`0.3 f_{ck}^{2/3}` up to C50, :math:`2.12\ln(1 + 0.1 f_{cm})` above).
    * ``E`` = :math:`E_{ci} = 21500\,\alpha_E (f_{cm}/10)^{1/3}`: fib MC2010
      section 5.1.7.2, with the aggregate factor :math:`\alpha_E` of MC2010
      Table 5.1-6 (``aggregate``).
    * ``Gf``: fib MC2010 Eq. 5.1-9, :math:`0.073 f_{cm}^{0.18}` [N/mm]. It has
      no aggregate-size term.
    * ``Gc`` = ``GC_OVER_GF * Gf`` (Nakamura & Higai 2001 ratio; MC2010 has
      no compressive fracture energy).
    * ``ec`` = :math:`\varepsilon_{c1} = 0.7 f_{cm}^{0.31}` per mille, capped
      at 2.8 per mille: EN 1992-1-1 Table 3.1; floored at ``1.6 fc/E`` so the
      pre-peak branch stays above the elastic line.
    * ``Df = 0.85``, ``hardening = (0.3, 0.01)``, ``As = 15``: the CDPM2
      calibration of Grassl et al. (2013). These are **not** the fork's
      constructor defaults (``Df = 1.0``, ``Hp = 0.5``); LadrunoConcrete3D
      always emits ``-Df`` / ``-hardening`` so the deck runs these values.

    ``units`` is an explicit preset from :data:`UNIT_PRESETS`; nothing is
    inferred from the magnitude of ``fc``.
    """
    if fc <= 0:
        raise ValueError(f"fib_mc2010_concrete_params: fc must be > 0, got {fc!r}")
    s_sig, s_len = _unit_scales(units, "fib_mc2010_concrete_params")
    if aggregate not in AGGREGATE_FACTORS:
        raise ValueError(
            f"fib_mc2010_concrete_params: unknown aggregate {aggregate!r}, "
            f"supported: {sorted(AGGREGATE_FACTORS)}"
        )

    s_gf = s_sig * s_len
    alpha_E = AGGREGATE_FACTORS[aggregate]

    f_cm_mpa = fc / s_sig
    f_ck_mpa = max(f_cm_mpa - 8.0, 0.75 * f_cm_mpa)

    if f_ck_mpa <= 50.0:
        ft_mpa = 0.30 * (f_ck_mpa ** (2.0 / 3.0))
    else:
        ft_mpa = 2.12 * math.log(1.0 + 0.10 * f_cm_mpa)

    e_mpa = 21500.0 * alpha_E * ((f_cm_mpa / 10.0) ** (1.0 / 3.0))
    gf_n_mm = 0.073 * (f_cm_mpa ** 0.18)
    gc_n_mm = GC_OVER_GF * gf_n_mm

    fc_u = f_cm_mpa * s_sig
    fck_u = f_ck_mpa * s_sig
    ft_u = ft_mpa * s_sig
    e_u = e_mpa * s_sig
    nu = 0.20
    gf_u = gf_n_mm * s_gf
    gc_u = gc_n_mm * s_gf

    eps0 = fc_u / e_u
    eps_c1 = max(min(0.0007 * (f_cm_mpa ** 0.31), 0.0028), 1.6 * eps0)

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

    # Snap-back ceilings for the CDPM2 bilinear tension law (the fork
    # wrapper default) and the compression law.
    hp = hardening[1]
    l_max_t = tension_snapback_factor("bilinear") * e_u * gf_u / (ft_u * ft_u)
    l_max_c = 2.0 * e_u * gc_u / ((fc_u * fc_u) * (1.0 + hp))

    return ConcreteMC2010Params(
        E=e_u,
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
        aggregate=aggregate,
        fck=fck_u,
        fcm=fc_u,
        l_max_t=l_max_t,
        l_max_c=l_max_c,
    )


def tension_snapback_factor(tension_law: str | None) -> float:
    r"""``k`` in the crack-band snap-back ceiling :math:`h_{max} = k\,E G_f / f_t^2`.

    A band of width ``h`` snaps back when the softening slope in crack
    opening, :math:`|d\sigma/dw|`, exceeds :math:`E/h`, so
    :math:`h_{max} = E / |d\sigma/dw|_{w=0}`. For the fork's
    LadrunoConcrete3D laws (``LadrunoConcrete3DKernel.h``):

    * ``"exp"``: :math:`\sigma = f_t\exp(-w f_t/G_f)`, initial slope
      :math:`f_t^2/G_f`, so ``k = 1``.
    * ``"bilinear"`` (CDPM2, the wrapper default, also for ``None``):
      first branch from :math:`f_t` to :math:`0.3 f_t` at
      :math:`w_1 = 0.15 w_f`, :math:`w_f = G_f/(0.225 f_t)`, initial slope
      :math:`1.05 f_t^2/G_f`, so ``k = 20/21``.
    """
    if tension_law == "exp":
        return 1.0
    if tension_law in (None, "bilinear"):
        return 20.0 / 21.0
    raise ValueError(
        f"tension_snapback_factor: unknown tension_law {tension_law!r} "
        "(expected 'bilinear', 'exp' or None)"
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
        if d >= 1.0:
            raise ValueError(
                f"make_tension: point {i} reaches damage d={d!r} >= 1 "
                f"(stress {si!r} at strain {xi!r}); the inputs give a "
                "backbone with no residual stiffness"
            )
        Td[i] = d
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
        if d >= 1.0:
            raise ValueError(
                f"make_compression: point {i} reaches damage d={d!r} >= 1 "
                f"(stress {si!r} at strain {xi!r}); the inputs give a "
                "backbone with no residual stiffness"
            )
        Cd[i] = d
    return Ce, Cs, Cd


def l_max(E: float, Gf: float, ft: float) -> float:
    """Crack-band snapback element-size ceiling ``2 E Gf / ft^2``.

    Above this band width the regularizer floors fracture energy at the
    brittle limit (``cpp:961-965``) and the response is no longer
    mesh-objective. Independent of ``lch_ref`` (ADR 0044, Fact 2).
    """
    return 2.0 * E * Gf / (ft * ft)


def rc_regularization_lch_max(
    x: tuple[float, ...] | list[float],
    y: tuple[float, ...] | list[float],
    d: tuple[float, ...] | list[float],
    E: float,
    lch_ref: float,
) -> float:
    r"""Band width above which ``-autoRegularization`` floors a backbone.

    A port of the fork's ``ladruno_rc_kernel::fractureEnergy`` /
    ``regularize`` (``LadrunoRCKernel.h``). The kernel rescales the
    post-peak branch to the energy :math:`g_0\,l_{ref}/l_{ch}` and clamps
    it at :math:`g_{min} = 1.01\,y_p x_p / 2` (the peak point) to keep the
    strain abscissa monotone, i.e. to stop snap-back. The clamp engages for

    .. math:: l_{ch} > l_{max} = l_{ref}\,g_0 / g_{min}.

    ``g_0`` is the kernel's energy measure: the unloading triangle at the
    peak (with the damaged modulus) plus the area under the softening
    branch, extended linearly to zero stress when the table ends in
    softening. ``d`` may be empty (the fork pads it with zeros). Returns
    ``inf`` for a backbone with no softening branch.
    """
    n = len(x)
    dd = list(d) if d else [0.0] * n
    q = [y[i] / (1.0 - dd[i]) for i in range(n)]
    pos1 = -1
    for i in range(1, n):
        if (y[i] - y[i - 1]) / (x[i] - x[i - 1]) < 0.0:
            pos1 = i - 1
            break
    if pos1 < 0:
        return math.inf
    pos2 = -1
    for i in range(pos1 + 1, n):
        if (y[i] - y[i - 1]) / (x[i] - x[i - 1]) >= 0.0:
            pos2 = i - 1
            break
    g_add = 0.0
    if pos2 < 0:
        if y[n - 1] > 0.0:
            k = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
            x3 = x[n - 1] - y[n - 1] / k
            g_add = y[n - 1] * (x3 - x[n - 1]) / 2.0
        pos2 = n - 1
    d1 = 1.0 - y[pos1] / q[pos1] if q[pos1] > 0.0 else 0.0
    Ed = (1.0 - d1) * E
    g0 = y[pos1] * y[pos1] / Ed / 2.0 if Ed > 0.0 else 0.0
    for i in range(pos1 + 1, pos2 + 1):
        g0 += (x[i] - x[i - 1]) * (y[i - 1] + y[i]) / 2.0
    g0 += g_add
    g_min = 1.01 * y[pos1] * x[pos1] / 2.0
    return float(lch_ref * g0 / g_min)
