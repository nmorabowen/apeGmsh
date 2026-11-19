"""
The period-propagated parity tolerance of a response-spectrum result.

An OpenSees RSA result compared with a reference program (ETABS, say)
never sees exactly the same periods. A flat tolerance then fails for a
difference the period mismatch fully explains. This module turns the
period mismatch into the tolerance a parity gate should use, reported as
its base and period parts compounded, ``total = (1 + base)(1 + period) - 1``
(:class:`ParityTolerance`), so a reader sees why a gate is wider than its
base. Compounding keeps a base on top of an exact period part exact.

* :func:`spectrum_slope` — the local slope ``s = d ln Sa / d ln T``, one-sided
  so a corner of the spectrum gives a defined value;
* :func:`period_tolerance` — one quantity governed by one mode;
* :func:`combined_period_tolerance` — a CQC/SRSS-combined quantity,
  re-evaluated with the reference periods; no dominant mode is picked.

A force-like quantity (an acceleration, a reaction, an element force) of a
mode scales with ``Sa(T)``; a displacement-like one (a displacement, a
drift) with ``Sd = Sa T² / (4π²)``. Relative to the reference, with
``δT/T = T / T_ref - 1``, to first order:

* forces: ``δq/q ≈ s · δT/T``;
* displacements: ``δq/q ≈ (s + 2) · δT/T``.

The exact form is the ratio ``q(T) / q(T_ref) = Sa(T)/Sa(T_ref) ·
(T/T_ref)^p`` with ``p = 0`` for forces and ``2`` for displacements. A
parity gate uses the exact form (the default ``method="exact"``): it has no
linearisation error, and it is the one the combined form generalises. The
first-order form is there to explain a number. Both assume the two programs
share the mode shapes and differ only in the periods.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .spectrum import DesignSpectrum, _sa_function, combine_modes

__all__ = [
    "ParityTolerance",
    "combined_period_tolerance",
    "period_tolerance",
    "spectrum_slope",
]

#: ``"force"`` scales with ``Sa``; ``"displacement"`` with ``Sa T²``.
Quantity = Literal["force", "displacement"]

_T_POWER = {"force": 0, "displacement": 2}


@dataclass(frozen=True, slots=True)
class ParityTolerance:
    """A relative parity tolerance, ``total = (1 + base)(1 + period) - 1``.

    ``base`` is the gate's own tolerance (solver, mesh, rounding);
    ``period`` the part the period mismatch explains, a float for one mode
    or an array (one per value) for a combined quantity. The parts
    compound rather than add: a 1 % base on a 5.27 % period part gives
    6.32 %, not 6.27 %; the plain sum is their first-order form, low by
    ``base · period``. ``method`` is
    ``"exact"``, ``"first_order"`` or ``"combined"``. ``str()`` gives the
    one-line report; nothing is printed.
    """

    base: float
    period: "float | np.ndarray"
    quantity: str
    method: str

    @property
    def total(self) -> "float | np.ndarray":
        """``(1 + base)(1 + period) - 1``: the relative tolerance the gate
        uses."""
        return (1.0 + self.base) * (1.0 + self.period) - 1.0

    def passes(self, value: "npt.ArrayLike", reference: "npt.ArrayLike") -> bool:
        """``|value - reference| <= total · |reference|``, at every value."""
        v = np.asarray(value, dtype=np.float64)
        r = np.asarray(reference, dtype=np.float64)
        return bool(np.all(np.abs(v - r) <= self.total * np.abs(r)))

    def __str__(self) -> str:
        period = float(np.max(self.period))
        worst = " (max)" if np.ndim(self.period) else ""
        total = (1.0 + self.base) * (1.0 + period) - 1.0
        return (
            f"tolerance {total:.2%}{worst} = (1 + base {self.base:.2%})"
            f"(1 + period {period:.2%}) - 1 ({self.quantity}, {self.method})"
        )


def _sa_of(spectrum: Any) -> "Callable[[np.ndarray], np.ndarray]":
    """``Sa(T)`` of a :class:`DesignSpectrum`, or of what a
    :class:`DesignSpectrum` takes as a function (an object with ``.at``, or a
    vectorised callable); refused where it is not positive (a ratio of
    ``Sa`` is taken)."""
    raw = (spectrum.sa if isinstance(spectrum, DesignSpectrum)
           else _sa_function(spectrum))
    if raw is None:
        raise TypeError(
            "spectrum tolerance: spectrum must be a DesignSpectrum, an object "
            f"with .at(T), or a vectorised Sa(T); got {type(spectrum).__name__}."
        )

    def sa(t: "npt.ArrayLike") -> np.ndarray:
        v = np.asarray(raw(np.asarray(t, dtype=np.float64)), dtype=np.float64)
        if np.any(v <= 0.0) or not np.all(np.isfinite(v)):
            raise ValueError(
                "spectrum tolerance: Sa must be finite and > 0 at every "
                "period it is evaluated at."
            )
        return v

    return sa


def _power(quantity: str) -> int:
    if quantity not in _T_POWER:
        raise ValueError(
            "spectrum tolerance: quantity must be 'force' or 'displacement', "
            f"got {quantity!r}."
        )
    return _T_POWER[quantity]


def _base(base: float) -> float:
    if not (math.isfinite(base) and base >= 0.0):
        raise ValueError(
            f"spectrum tolerance: base must be finite and >= 0, got {base!r}."
        )
    return float(base)


def spectrum_slope(
    spectrum: Any,
    period: float,
    *,
    side: Literal["left", "right", "max"] = "max",
    rel_step: float = 1e-6,
) -> float:
    """The local log-log slope ``s = d ln Sa / d ln T`` at ``period``.

    One-sided differences over ``T (1 ± rel_step)``, so a plateau gives 0,
    a ``1/T`` branch -1, and a corner the slope of the chosen side.
    ``side="max"`` takes the side with the larger ``|s|``: the conservative
    choice when the direction of the period error is not known.
    ``spectrum`` is a :class:`DesignSpectrum` or a vectorised ``Sa(T)``.
    """
    t = float(period)
    if not (math.isfinite(t) and t > 0.0):
        raise ValueError(f"spectrum_slope: period must be > 0, got {period!r}.")
    if not 0.0 < rel_step < 1.0:
        raise ValueError(
            f"spectrum_slope: rel_step must be in (0, 1), got {rel_step!r}."
        )
    sa = _sa_of(spectrum)
    lo, mid, hi = sa(np.array([t * (1.0 - rel_step), t, t * (1.0 + rel_step)]))
    left = math.log(mid / lo) / -math.log1p(-rel_step)
    right = math.log(hi / mid) / math.log1p(rel_step)
    if side == "left":
        return left
    if side == "right":
        return right
    if side == "max":
        return left if abs(left) >= abs(right) else right
    raise ValueError(
        f"spectrum_slope: side must be 'left', 'right' or 'max', got {side!r}."
    )


def period_tolerance(
    spectrum: Any,
    period: float,
    period_ref: float,
    *,
    quantity: Quantity,
    method: Literal["exact", "first_order"] = "exact",
    base: float = 0.0,
) -> ParityTolerance:
    """The tolerance of one quantity governed by one mode.

    ``period`` is ours, ``period_ref`` the reference program's. ``"exact"``
    gives ``|Sa(T)/Sa(T_ref) · (T/T_ref)^p - 1|``; ``"first_order"`` gives
    ``|(s + p) · δT/T|``, with ``s`` the one-sided slope at ``T_ref`` on the
    side ``T`` lies (:func:`spectrum_slope`), so a corner is handled.
    """
    p = _power(quantity)
    b = _base(base)
    t, t_ref = float(period), float(period_ref)
    if not (t > 0.0 and t_ref > 0.0):
        raise ValueError(
            "period_tolerance: periods must be > 0, got "
            f"{period!r} and {period_ref!r}."
        )
    if method == "exact":
        sa, sa_ref = _sa_of(spectrum)(np.array([t, t_ref]))
        part = abs(sa / sa_ref * (t / t_ref) ** p - 1.0)
    elif method == "first_order":
        delta = t / t_ref - 1.0
        s = 0.0 if delta == 0.0 else spectrum_slope(
            spectrum, t_ref, side="right" if delta > 0.0 else "left")
        part = abs((s + p) * delta)
    else:
        raise ValueError(
            "period_tolerance: method must be 'exact' or 'first_order', "
            f"got {method!r}."
        )
    return ParityTolerance(b, float(part), quantity, method)


def combined_period_tolerance(
    spectrum: Any,
    per_mode: "npt.ArrayLike",
    periods: "npt.ArrayLike",
    periods_ref: "npt.ArrayLike",
    *,
    quantity: Quantity,
    rule: str = "CQC",
    damping: "float | npt.ArrayLike | None" = None,
    base: float = 0.0,
    sa_periods: "npt.ArrayLike | None" = None,
) -> ParityTolerance:
    """The tolerance of a combined quantity, from its per-mode values.

    ``per_mode`` holds our signed per-mode values, modes on axis 0, as
    :func:`~apeGmsh.opensees.analysis.spectrum.combine_modes` takes them.
    Each mode is rescaled to the reference period by
    ``Sa(T_ref)/Sa(T) · (T_ref/T)^p`` and the combination is re-run with
    the reference ``ω = 2π/T_ref`` (for CQC, ``ρ`` is recomputed with it).
    CQC needs ``damping`` (one ratio, or one per mode); there is no default.
    The period part is ``|C - C_ref| / C_ref``, per value; ``C_ref = 0``
    gives 0 when ``C = 0`` too and ``inf`` otherwise.

    ``sa_periods`` (one per mode, ``0`` for none) marks modes whose Sa was
    read at an override period (``residual_period=``,
    :attr:`~apeGmsh.opensees.analysis.spectrum.SpectrumModalResult.sa_periods`):
    their Sa does not move with the period, so only ``(T_ref/T)^p``
    rescales them, and both combinations take them as one rigid term. The
    ``p = 2`` stays for displacements: the group's sum is the static
    residual ``K⁻¹ M r Sa(T)``, which does not depend on how the pseudo-modes
    split it, but does scale with the stiffness, by exactly ``(T_ref/T)²``
    when the two models share their shapes (the assumption of this module).
    Its force ``M r Sa(T)`` does not, which is ``p = 0``. If the two models
    split the residual differently, the per-mode ratios no longer describe
    the group; compare the two groups' own sums instead.
    """
    p = _power(quantity)
    b = _base(base)
    if rule == "CQC" and damping is None:
        raise ValueError(
            "combined_period_tolerance: rule='CQC' needs damping= (one ratio "
            "or one per mode); pass rule='SRSS' to combine without it."
        )
    v = np.asarray(per_mode, dtype=np.float64)
    t = np.asarray(periods, dtype=np.float64).ravel()
    t_ref = np.asarray(periods_ref, dtype=np.float64).ravel()
    if v.ndim < 1 or not (t.size == t_ref.size == v.shape[0]):
        raise ValueError(
            "combined_period_tolerance: periods and periods_ref need one "
            f"entry per mode ({v.shape[0] if v.ndim else 0}); got "
            f"{t.size} and {t_ref.size}."
        )
    if np.any(t <= 0.0) or np.any(t_ref <= 0.0):
        raise ValueError("combined_period_tolerance: periods must be > 0.")
    sa = _sa_of(spectrum)
    k = sa(t_ref) / sa(t) * (t_ref / t) ** p
    if sa_periods is not None:
        over = np.asarray(sa_periods, dtype=np.float64).ravel() > 0.0
        if over.size != t.size:
            raise ValueError(
                f"combined_period_tolerance: {over.size} Sa periods for "
                f"{t.size} modes.")
        k = np.where(over, (t_ref / t) ** p, k)
    v_ref = v * k.reshape((-1,) + (1,) * (v.ndim - 1))
    ours = combine_modes(v, rule=rule, omega=2.0 * np.pi / t, damping=damping,
                         sa_periods=sa_periods)
    ref = combine_modes(v_ref, rule=rule, omega=2.0 * np.pi / t_ref,
                        damping=damping, sa_periods=sa_periods)
    gap = np.abs(ours - ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(ref > 0.0, gap / ref, np.where(gap > 0.0, np.inf, 0.0))
    return ParityTolerance(
        b, float(part) if part.ndim == 0 else part, quantity, "combined")
