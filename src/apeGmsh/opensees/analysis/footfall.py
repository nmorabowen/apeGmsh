"""
Footfall vibration kernel — AISC Design Guide 11, 2nd ed. (2016), Chapter 7
(ADR 0109, D1).

Pure numpy, no OpenSees import.  These functions evaluate the walking
(§7.4.1) low- and high-frequency floor criteria from an already-known modal
basis and FRF; nothing here builds a model, solves an eigenproblem, or
reads a live domain — that is ``footfall_frf.py`` (S1) and the
``apeSees.footfall_walking`` driver (S2).

**Units are the caller's.**  Every function is dimensionless-in-form: give
it a force in whatever unit the model uses (`body_weight`) and the FRF or
mode shapes in the matching acceleration/force unit, and the accelerations
returned are in that same acceleration unit.  The kernel never divides by
`g` — the driver does, once it knows the model's units.  `tolerance_limit`
is the one exception: Table 4-1 / Fig 2-1 are inherently `%g` values, so it
returns a `%g` number directly (e.g. ``0.5``, not ``0.005``).
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

__all__ = [
    "Occupancy",
    "ToleranceKind",
    "dynamic_coefficient",
    "resonant_buildup_factor",
    "harmonic_for_dominant",
    "effective_impulse",
    "walking_low_frequency",
    "walking_high_frequency",
    "tolerance_limit",
    "dominant_frequency",
]

Occupancy = Literal[
    "office",
    "residence",
    "church",
    "school",
    "shopping",
    "dining",
    "indoor_bridge",
    "outdoor_bridge",
]
ToleranceKind = Literal["curve", "table"]

#: Table 4-1 flat acceptance limits, in `%g`.
_FLAT_LIMIT: dict[str, float] = {
    "office": 0.005,
    "residence": 0.005,
    "church": 0.005,
    "school": 0.005,
    "shopping": 0.015,
    "dining": 0.015,
    "indoor_bridge": 0.015,
    "outdoor_bridge": 0.05,
}

#: Table 7-1 — (upper frequency bound in Hz, harmonic) for the dominant
#: frequency's step-frequency band, 9-20 Hz. A frequency at a table
#: boundary belongs to the *lower* harmonic (its step frequency is then
#: exactly the band's 2.2 Hz upper walking-frequency limit).
_TABLE_7_1: tuple[tuple[float, int], ...] = (
    (11.0, 5),
    (13.2, 6),
    (15.4, 7),
    (17.6, 8),
    (20.0, 9),
)


def dynamic_coefficient(f: float) -> float:
    """Dynamic coefficient α for the resonant walking harmonic (Eq 7-2).

    Parameters
    ----------
    f : float
        Dominant frequency, Hz.

    Returns
    -------
    float
        ``α = 0.09 e^{-0.075 f}``.
    """
    return 0.09 * math.exp(-0.075 * f)


def resonant_buildup_factor(beta: float) -> float:
    """Partial resonant build-up factor ρ (Eq 7-3).

    Parameters
    ----------
    beta : float
        Modal damping ratio (fraction of critical).

    Returns
    -------
    float
        ``ρ = 50β + 0.25`` for ``β < 0.01``, ``ρ = 12.5β + 0.625`` for
        ``0.01 ≤ β < 0.03``, ``ρ = 1.0`` for ``β ≥ 0.03``.
    """
    if beta < 0.01:
        return 50.0 * beta + 0.25
    if beta < 0.03:
        return 12.5 * beta + 0.625
    return 1.0


def harmonic_for_dominant(f: float) -> int:
    """Resonant walking harmonic ``h`` for a dominant frequency (Table 7-1).

    Parameters
    ----------
    f : float
        Dominant frequency, Hz (the high-frequency floor range is 9-20 Hz).

    Returns
    -------
    int
        The harmonic ``h`` from Table 7-1 (5 through 9).

    Raises
    ------
    ValueError
        If ``f`` lies outside Table 7-1's 9-20 Hz range. Below 9 Hz the
        floor is low-frequency (Eq 7-1 applies, not the impulse branch);
        above 20 Hz the Guide sums no modes.
    """
    if not 9.0 <= f <= 20.0:
        raise ValueError(
            f"harmonic_for_dominant: Table 7-1 covers dominant frequencies "
            f"of 9-20 Hz, got {f} Hz."
        )
    for upper, harmonic in _TABLE_7_1:
        if f <= upper:
            return harmonic
    return _TABLE_7_1[-1][1]


def effective_impulse(f_step: float, f_n: float, body_weight: float) -> float:
    """Effective impulse of a footstep (Eq 1-6).

    Parameters
    ----------
    f_step : float
        Step frequency, Hz.
    f_n : float
        Natural frequency of the mode in question, Hz.
    body_weight : float
        Walker weight ``Q`` (force units).

    Returns
    -------
    float
        ``I_eff = (f_step^1.43 / f_n^1.30) · (Q / 17.8)``, in
        ``force · s`` (``Q``'s force unit; 17.8 is the Design Guide's
        constant for ``Q`` in lb).
    """
    return (math.pow(f_step, 1.43) / math.pow(f_n, 1.30)) * (body_weight / 17.8)


def walking_low_frequency(
    frf_max: float, f_dom: float, beta: float, body_weight: float
) -> float:
    """Low-frequency floor peak acceleration (Eq 7-1), dominant frequency < 9 Hz.

    Parameters
    ----------
    frf_max : float
        Peak FRF magnitude (acceleration per unit force) over the band.
    f_dom : float
        Dominant frequency, Hz.
    beta : float
        Modal damping ratio of the dominant mode.
    body_weight : float
        Walker weight ``Q`` (force units).

    Returns
    -------
    float
        ``a_p = FRF_max · α(f_dom) · ρ(β) · Q``.
    """
    return (
        frf_max
        * dynamic_coefficient(f_dom)
        * resonant_buildup_factor(beta)
        * body_weight
    )


def walking_high_frequency(
    f_n: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    f_dom: float,
    beta: float | np.ndarray,
    body_weight: float,
    dt: float = 0.005,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """High-frequency floor response (Eq 7-4 … 7-6), dominant frequency 9-20 Hz.

    Parameters
    ----------
    f_n : numpy.ndarray
        Natural frequencies of the modes to sum, Hz (Eq 7-5 sums every
        mode with ``f_n ≤ 20 Hz``; filter the basis before calling).
    phi_i : numpy.ndarray
        Mass-normalised mode shape component at the excitation node,
        one value per mode in ``f_n``.
    phi_j : numpy.ndarray
        Mass-normalised mode shape component at the response node,
        one value per mode in ``f_n``.
    f_dom : float
        Dominant frequency, Hz — fixes the resonant harmonic (Table 7-1)
        and the step frequency for every mode.
    beta : float or numpy.ndarray
        Modal damping ratio(s): one value applied to every mode, or one
        value per mode in ``f_n``.
    body_weight : float
        Walker weight ``Q`` (force units).
    dt : float, default 0.005
        Time step for sampling ``a(t)``, s (the Design Guide's own value).

    Returns
    -------
    a_espa : float
        Equivalent sinusoidal peak acceleration, ``√2 · RMS(a(t))``
        (Eq 7-6) over one step period ``1 / f_step``.
    a_peak : float
        ``max(|a(t)|)`` over the same window.
    t : numpy.ndarray
        Sample times, s, ``0 ≤ t < 1 / f_step``.
    a_t : numpy.ndarray
        Sampled combined-mode acceleration history (Eq 7-5).
    """
    f_n_arr = np.asarray(f_n, dtype=np.float64)
    phi_i_arr = np.asarray(phi_i, dtype=np.float64)
    phi_j_arr = np.asarray(phi_j, dtype=np.float64)
    beta_arr = np.broadcast_to(np.asarray(beta, dtype=np.float64), f_n_arr.shape)

    harmonic = harmonic_for_dominant(f_dom)
    f_step = f_dom / harmonic
    i_eff = (f_step**1.43 / f_n_arr**1.30) * (body_weight / 17.8)
    a_p = 2.0 * np.pi * f_n_arr * phi_i_arr * phi_j_arr * i_eff

    window = 1.0 / f_step
    t = np.arange(0.0, window, dt)
    decay = np.exp(-2.0 * np.pi * beta_arr[:, None] * f_n_arr[:, None] * t[None, :])
    osc = np.sin(2.0 * np.pi * f_n_arr[:, None] * t[None, :])
    a_t = (a_p[:, None] * decay * osc).sum(axis=0)

    a_peak = float(np.max(np.abs(a_t)))
    a_espa = float(np.sqrt(2.0) * np.sqrt(np.mean(a_t**2)))
    return a_espa, a_peak, t, a_t


def _curve_scale(f: float) -> float:
    """ISO 2631-2 base-curve shape ``s(f)`` used to scale the flat limit."""
    if f < 4.0:
        return math.sqrt(4.0 / f)
    if f <= 8.0:
        return 1.0
    return f / 8.0


def tolerance_limit(occupancy: Occupancy, f: float, kind: ToleranceKind = "curve") -> float:
    """Acceptance limit for the peak/ESPA acceleration (Fig 2-1 / Table 4-1).

    Parameters
    ----------
    occupancy : str
        One of ``"office"``, ``"residence"``, ``"church"``, ``"school"``
        (0.5 %g flat); ``"shopping"``, ``"dining"``, ``"indoor_bridge"``
        (1.5 %g flat); ``"outdoor_bridge"`` (5 %g flat). Values are returned
        as fractions of g (0.005, 0.015, 0.05), the same unit the driver
        reports ``a_p / g`` in (ADR 0109 D1).
    f : float
        Frequency at which the limit is evaluated, Hz (the dominant
        frequency or the frequency of a mode being checked).
    kind : {"curve", "table"}, default "curve"
        ``"curve"`` scales the flat Table 4-1 value by the ISO 2631-2 base
        curve shape ``s(f) = √(4/f)`` (f < 4 Hz), ``1`` (4-8 Hz), ``f/8``
        (f > 8 Hz), matching Fig 2-1. ``"table"`` returns the flat value.

    Returns
    -------
    float
        The limit as a fraction of g (``0.005`` is 0.5 %g).

    Raises
    ------
    ValueError
        If ``occupancy`` is not one of the values above, or ``kind`` is not
        ``"curve"`` or ``"table"``.
    """
    try:
        flat = _FLAT_LIMIT[occupancy]
    except KeyError:
        raise ValueError(
            f"tolerance_limit: unknown occupancy {occupancy!r}, expected "
            f"one of {sorted(_FLAT_LIMIT)}."
        ) from None
    if kind == "table":
        return flat
    if kind == "curve":
        return flat * _curve_scale(f)
    raise ValueError(f"tolerance_limit: kind must be 'curve' or 'table', got {kind!r}.")


def dominant_frequency(freq: np.ndarray, frf_mag: np.ndarray, f_max: float) -> float:
    """Dominant frequency — where ``|FRF|`` peaks over a band (§7.3).

    Not always a natural frequency: close, lightly separated modes with
    enough damping can shift the FRF's maximum off any single ``f_n``.

    Parameters
    ----------
    freq : numpy.ndarray
        Frequencies at which the FRF was evaluated, Hz.
    frf_mag : numpy.ndarray
        ``|FRF|`` at each frequency in ``freq``.
    f_max : float
        Upper frequency to consider, Hz; points above it are ignored (the
        low-frequency search passes ``f_max=9.0``, the high-frequency
        search ``f_max=20.0``).

    Returns
    -------
    float
        The frequency in ``freq`` (at or below ``f_max``) with the largest
        ``|FRF|``, or ``nan`` when no point lies at or below ``f_max`` (a
        floor whose whole FRF sits above 9 Hz has no low-frequency
        dominant frequency — ADR 0109 D5).
    """
    freq_arr = np.asarray(freq, dtype=np.float64)
    frf_mag_arr = np.asarray(frf_mag, dtype=np.float64)
    mask = freq_arr <= f_max
    if not np.any(mask):
        return float("nan")
    idx = int(np.argmax(frf_mag_arr[mask]))
    return float(freq_arr[mask][idx])
