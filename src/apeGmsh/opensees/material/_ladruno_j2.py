"""Shared grammar helpers for the Ladruno-fork J2 plasticity family.

``LadrunoJ2`` (nDMaterial), ``LadrunoJ2Finite`` (nDMaterial) and
``LadrunoUniaxialJ2`` (uniaxialMaterial) share one combined-hardening
flag grammar — verified against the shipped fork parsers
(``OPS_LadrunoJ2`` / ``OPS_LadrunoJ2Finite`` / ``OPS_LadrunoUniaxialJ2``):

* ``-iso voce s0 Qinf b Hiso`` — Voce + linear isotropic hardening
  (always 4 doubles, in that order).
* ``-kin N C1 g1 C2 g2 ...`` — ``N`` Chaboche backstress pairs
  ``(C_k, gamma_k)``; the parser caps ``N`` at 8.
* ``-damage lemaitre r s pD Dc`` — the optional Lemaitre ductile-damage
  mode (only on the two small-strain J2s, **not** ``LadrunoJ2Finite``);
  the parser requires ``r > 0`` and ``0 < Dc <= 1``.

These helpers centralize the validation + arg-building so the three
classes never drift. The fork-side single-kernel design
(``LadrunoJ2Kernel.h`` / ``LadrunoHardening.h``) is mirrored by this
single Python grammar source.
"""
from __future__ import annotations


# The fork parsers cap the number of Chaboche backstress pairs at 8
# (``MAXBACK``); a longer list is rejected at parse time.
_MAX_BACKSTRESSES = 8


def validate_iso(label: str, sig0: float, Qinf: float, b: float,
                 Hiso: float) -> None:
    """Validate the ``-iso voce`` Voce + linear hardening parameters.

    ``sig0`` (initial yield stress) must be strictly positive — a J2
    surface with zero radius is never intended. ``Qinf`` may be negative
    (softening saturation); ``b`` and ``Hiso`` follow the fork's
    lenient parse (no sign constraint enforced there).
    """
    if sig0 <= 0:
        raise ValueError(f"{label}: sig0 must be > 0, got {sig0!r}")
    if b < 0:
        raise ValueError(f"{label}: b (saturation rate) must be >= 0, got {b!r}")


def validate_backstresses(
    label: str, backstresses: tuple[tuple[float, float], ...]
) -> None:
    """Validate the ``-kin`` Chaboche backstress pairs.

    Each entry is a ``(C_k, gamma_k)`` pair; the parser caps the count
    at 8. ``C_k > 0`` and ``gamma_k >= 0`` are the physical Armstrong-
    Frederick constraints (the fork parses leniently, but a non-positive
    modulus or negative recall is always a user error).
    """
    if len(backstresses) > _MAX_BACKSTRESSES:
        raise ValueError(
            f"{label}: at most {_MAX_BACKSTRESSES} kinematic backstress "
            f"pairs are supported (the fork MAXBACK), got {len(backstresses)}"
        )
    for i, pair in enumerate(backstresses):
        if len(pair) != 2:
            raise ValueError(
                f"{label}: backstress {i} must be a (C, gamma) pair, "
                f"got {pair!r}"
            )
        C, gamma = pair
        if C <= 0:
            raise ValueError(
                f"{label}: backstress {i} modulus C must be > 0, got {C!r}"
            )
        if gamma < 0:
            raise ValueError(
                f"{label}: backstress {i} recall gamma must be >= 0, "
                f"got {gamma!r}"
            )


def validate_lemaitre(
    label: str, damage: tuple[float, float, float, float]
) -> None:
    """Validate the ``-damage lemaitre r s pD Dc`` parameters.

    Mirrors the parser's hard guard: ``r > 0`` and ``0 < Dc <= 1``.
    """
    if len(damage) != 4:
        raise ValueError(
            f"{label}: damage must be a 4-tuple (r, s, pD, Dc), got {damage!r}"
        )
    r, _s, _pD, Dc = damage
    if r <= 0:
        raise ValueError(f"{label}: Lemaitre r must be > 0, got {r!r}")
    if not (0.0 < Dc <= 1.0):
        raise ValueError(
            f"{label}: Lemaitre Dc (critical damage) must be in (0, 1], "
            f"got {Dc!r}"
        )


def iso_args(sig0: float, Qinf: float, b: float, Hiso: float) -> list[float | str]:
    """``-iso voce s0 Qinf b Hiso`` as an arg list (always emitted)."""
    return ["-iso", "voce", sig0, Qinf, b, Hiso]


def kin_args(
    backstresses: tuple[tuple[float, float], ...]
) -> list[float | int | str]:
    """``-kin N C1 g1 ...`` as an arg list (empty when no backstresses)."""
    if not backstresses:
        return []
    args: list[float | int | str] = ["-kin", len(backstresses)]
    for C, gamma in backstresses:
        args += [C, gamma]
    return args


def lemaitre_args(
    damage: tuple[float, float, float, float]
) -> list[float | str]:
    """``-damage lemaitre r s pD Dc`` as an arg list."""
    return ["-damage", "lemaitre", *damage]
