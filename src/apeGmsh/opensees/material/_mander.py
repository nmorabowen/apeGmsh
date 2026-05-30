"""Mander confined-concrete compression envelope (Mander, Priestley & Park 1988).

For fiber sections the uniaxial ``ASDConcrete1D`` cannot see triaxial confinement
(no stress decomposition / Lubliner surface — ADR 0044), so confinement must be
baked into the compression backbone. This module produces that backbone from the
Mander model; :meth:`ASDConcrete1D.from_mander` assembles it into a material via
the explicit-curve channel.

Confinement enters either as the confined peak strength ``fcc`` directly, or as
the effective lateral confining pressure ``fl`` (then ``fcc`` is the Mander
strength). The model:

    fcc      = fc (2.254 sqrt(1 + 7.94 fl/fc) - 2 fl/fc - 1.254)      [if fl given]
    eps_cc   = eps_co [1 + 5 (fcc/fc - 1)]
    fc(eps)  = fcc x r / (r - 1 + x^r),   x = eps/eps_cc,
               r = Ec/(Ec - Esec),   Esec = fcc/eps_cc

The envelope's initial tangent is exactly ``Ec``, so anchoring the first point on
the elastic line at ``0.5 fcc`` keeps the secant slope equal to ``E`` (matching
the tension law's modulus, as the native ``-fc`` builder does).

Stresses/strains are returned as positive magnitudes (ASDConcrete's compression
backbone convention).
"""
from __future__ import annotations


__all__ = [
    "confined_strength",
    "confined_peak_strain",
    "compression_backbone",
]


def confined_strength(fc: float, fl: float) -> float:
    """Mander confined peak strength from effective lateral pressure ``fl``."""
    return fc * (2.254 * (1.0 + 7.94 * fl / fc) ** 0.5 - 2.0 * fl / fc - 1.254)


def confined_peak_strain(fc: float, fcc: float, eps_co: float) -> float:
    """Strain at the confined peak: ``eps_cc = eps_co [1 + 5 (fcc/fc - 1)]``."""
    return eps_co * (1.0 + 5.0 * (fcc / fc - 1.0))


def compression_backbone(
    E: float,
    fcc: float,
    eps_cc: float,
    eps_cu: float,
    *,
    n: int = 12,
    plastic_ratio: float = 0.7,
) -> tuple[list[float], list[float], list[float]]:
    """Mander compression envelope as ASDConcrete ``(Ce, Cs, Cd)`` points.

    ``n`` points sample the curve from the elastic anchor to ``eps_cu`` (the
    confined peak ``eps_cc`` is always included as a vertex). ``plastic_ratio``
    in ``[0, 1]`` splits each point's inelastic strain into plastic vs damage:
    ``1`` = pure plasticity (residual strain, full stiffness), ``0`` = pure
    damage (secant-to-origin unloading). Returns positive magnitudes.
    """
    Esec = fcc / eps_cc
    if E <= Esec:
        raise ValueError(
            f"Mander: E ({E!r}) must exceed the secant modulus Esec=fcc/eps_cc "
            f"({Esec!r}); the peak is too soft for the envelope to be valid."
        )
    r = E / (E - Esec)

    def mander(eps: float) -> float:
        x = eps / eps_cc
        return fcc * x * r / (r - 1.0 + x ** r)

    fc0 = 0.5 * fcc
    ec0 = fc0 / E   # elastic anchor: secant slope is exactly E

    # Strain samples beyond the anchor, with eps_cc forced in as a vertex.
    raw = [ec0 + (eps_cu - ec0) * k / (n - 1) for k in range(n)]
    if ec0 < eps_cc < eps_cu:
        raw.append(eps_cc)
    strains = sorted({e for e in raw if e > ec0})

    Ce = [0.0, ec0]
    Cs = [0.0, fc0]
    for eps in strains:
        Ce.append(eps)
        Cs.append(mander(eps))

    Cd = [0.0] * len(Ce)
    for i in range(2, len(Ce)):
        inel = max(Ce[i] - Cs[i] / E, 0.0)
        xpl = min(plastic_ratio * inel, inel)
        qi = E * (Ce[i] - xpl)
        Cd[i] = max(0.0, 1.0 - Cs[i] / qi) if qi > 0.0 else 0.0
    return Ce, Cs, Cd
