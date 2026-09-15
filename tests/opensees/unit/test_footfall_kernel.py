"""Unit tests for the footfall vibration kernel (ADR 0109, D1).

Pure math — no OpenSees, no live domain. Oracle numbers are AISC Design
Guide 11, 2nd ed., Example 7.1 and Table 7-2 (see the handoff and the ADR
Context items 4-6). Q = 168 lb, beta = 0.025, g = 386 in/s^2 throughout;
"%g" below means ``100 * a / g``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from apeGmsh.opensees.analysis.footfall import (
    dominant_frequency,
    dynamic_coefficient,
    effective_impulse,
    harmonic_for_dominant,
    resonant_buildup_factor,
    tolerance_limit,
    walking_high_frequency,
    walking_low_frequency,
)

G = 386.0  # in/s^2
Q = 168.0  # lb

# Example 7.1, Table 7-2: (mode, f_n Hz, phi at backspan, (in./kip.s^2)^1/2)
TABLE_7_2 = [
    (1, 3.49, 0.449), (2, 3.80, 0.0), (3, 4.89, -0.513), (4, 6.88, 0.0),
    (5, 7.05, 1.67), (6, 7.31, -0.553), (7, 7.72, 0.0), (8, 7.88, 0.0),
    (9, 8.07, -1.47), (10, 8.85, 1.83), (11, 8.98, 0.0), (12, 9.35, -2.57),
    (13, 9.42, 0.565), (14, 9.63, 0.0), (15, 10.1, -1.80), (16, 10.4, -0.649),
    (17, 10.7, 0.0), (18, 10.9, -1.35), (19, 11.2, 0.0), (20, 12.0, 1.93),
    (21, 12.6, 0.0), (22, 12.6, -3.15), (23, 13.1, 0.0), (24, 13.3, 0.286),
    (25, 13.7, 0.0), (26, 14.5, 0.832), (27, 14.6, 0.0), (28, 14.7, 0.044),
    (29, 15.5, 1.00), (30, 16.0, 0.0), (31, 16.9, -1.47), (32, 17.3, 0.0),
    (33, 17.5, 1.18), (34, 17.6, 1.08), (35, 17.7, 0.0), (36, 18.6, 1.93),
    (37, 19.2, 0.0), (38, 19.8, 0.810),
]
F_N = np.array([row[1] for row in TABLE_7_2])
PHI = np.array([row[2] for row in TABLE_7_2])


# ---------------------------------------------------------------------------
# Low-frequency floor (Eq 7-1 ... 7-3)
# ---------------------------------------------------------------------------


def test_dynamic_coefficient_example_7_1() -> None:
    assert dynamic_coefficient(3.49) == pytest.approx(0.069, rel=1e-2)


def test_resonant_buildup_factor_example_7_1() -> None:
    assert resonant_buildup_factor(0.025) == pytest.approx(0.9375)


def test_resonant_buildup_factor_boundary_at_0_01() -> None:
    # The beta < 0.01 and 0.01 <= beta < 0.03 branches agree exactly here.
    assert resonant_buildup_factor(0.01) == pytest.approx(50.0 * 0.01 + 0.25)
    assert resonant_buildup_factor(0.01) == pytest.approx(12.5 * 0.01 + 0.625)


def test_resonant_buildup_factor_boundary_at_0_03() -> None:
    # The 0.01 <= beta < 0.03 branch and the beta >= 0.03 plateau agree here.
    assert resonant_buildup_factor(0.03) == pytest.approx(12.5 * 0.03 + 0.625)
    assert resonant_buildup_factor(0.03) == pytest.approx(1.0)


def test_walking_low_frequency_example_7_1() -> None:
    a_p = walking_low_frequency(0.0344, 3.49, 0.025, Q)
    assert a_p == pytest.approx(0.374, rel=1e-2)


# ---------------------------------------------------------------------------
# High-frequency floor (Eq 1-6, 7-4 ... 7-6, Table 7-1)
# ---------------------------------------------------------------------------


def test_harmonic_for_dominant_example_7_1() -> None:
    assert harmonic_for_dominant(12.6) == 6


@pytest.mark.parametrize(
    "f, expected",
    [(11.0, 5), (13.2, 6), (15.4, 7), (17.6, 8), (20.0, 9)],
)
def test_harmonic_for_dominant_table_7_1_boundaries(f: float, expected: int) -> None:
    # A boundary frequency belongs to the lower harmonic: its step
    # frequency (f / h) is then exactly the 2.2 Hz walking-frequency cap.
    assert harmonic_for_dominant(f) == expected


def test_effective_impulse_example_7_1() -> None:
    f_step = 12.6 / harmonic_for_dominant(12.6)
    assert f_step == pytest.approx(2.1)
    i_eff = effective_impulse(f_step, 12.6, Q)
    assert i_eff == pytest.approx(1.01, rel=1e-2)


def test_mode_22_alone_amplitude_eq_7_4() -> None:
    # Eq 7-4's raw per-mode amplitude, before time sampling introduces the
    # decay factor. Table 7-2's own units (Q in lb, phi in
    # (in./kip.s^2)^1/2) need lb->kip (/1000) then g (/386) to land in %g,
    # exactly as Example 7.1 states it.
    f_step = 12.6 / harmonic_for_dominant(12.6)
    i_eff = effective_impulse(f_step, 12.6, Q)
    phi = -3.15
    a_p_22 = 2.0 * math.pi * 12.6 * phi * phi * i_eff
    a_p_22_pct_g = a_p_22 / 1000.0 / G * 100.0
    assert a_p_22_pct_g == pytest.approx(0.206, rel=1e-2)


def test_walking_high_frequency_all_38_modes_example_7_1() -> None:
    a_espa, a_peak, t, a_t = walking_high_frequency(
        F_N, PHI, PHI, f_dom=12.6, beta=0.025, body_weight=Q, dt=0.005
    )
    a_peak_pct_g = a_peak / 1000.0 / G * 100.0
    a_espa_pct_g = a_espa / 1000.0 / G * 100.0
    assert a_peak_pct_g == pytest.approx(0.865, rel=2e-2)
    assert a_espa_pct_g == pytest.approx(0.314, rel=2e-2)
    # a(t) is sampled over one step period, half-open at the window's end.
    assert t[0] == 0.0
    assert t[-1] < 1.0 / 2.1
    assert a_t.shape == t.shape


def test_walking_high_frequency_scalar_beta_broadcasts_over_modes() -> None:
    # A scalar beta (Example 7.1's uniform damping) must give the same
    # result as the same value repeated once per mode.
    scalar = walking_high_frequency(F_N, PHI, PHI, 12.6, 0.025, Q, dt=0.005)
    per_mode = walking_high_frequency(
        F_N, PHI, PHI, 12.6, np.full_like(F_N, 0.025), Q, dt=0.005
    )
    assert_allclose(scalar[3], per_mode[3])


# ---------------------------------------------------------------------------
# Acceptance limit (Fig 2-1 / Table 4-1)
# ---------------------------------------------------------------------------


def test_tolerance_limit_curve_example_7_1() -> None:
    assert tolerance_limit("office", 8.85) == pytest.approx(0.00553, rel=1e-2)
    assert tolerance_limit("office", 9.35) == pytest.approx(0.00584, rel=1e-2)
    assert tolerance_limit("office", 6.0) == pytest.approx(0.005)


@pytest.mark.parametrize("occupancy", ["office", "residence", "church", "school"])
def test_tolerance_limit_table_flat_half_percent_g(occupancy: str) -> None:
    assert tolerance_limit(occupancy, 8.85, kind="table") == pytest.approx(0.005)


@pytest.mark.parametrize("occupancy", ["shopping", "dining", "indoor_bridge"])
def test_tolerance_limit_table_flat_one_and_half_percent_g(occupancy: str) -> None:
    assert tolerance_limit(occupancy, 8.85, kind="table") == pytest.approx(0.015)


def test_tolerance_limit_table_outdoor_bridge_five_percent_g() -> None:
    assert tolerance_limit("outdoor_bridge", 8.85, kind="table") == pytest.approx(0.05)


def test_tolerance_limit_unknown_occupancy_raises() -> None:
    with pytest.raises(ValueError, match="occupancy"):
        tolerance_limit("gymnasium", 5.0)  # type: ignore[arg-type]


def test_tolerance_limit_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        tolerance_limit("office", 5.0, kind="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dominant frequency (%3.3)
# ---------------------------------------------------------------------------


def test_dominant_frequency_picks_the_frf_peak() -> None:
    freq = np.array([3.0, 3.49, 4.0, 12.6])
    mag = np.array([0.01, 0.0344, 0.02, 0.05])
    assert dominant_frequency(freq, mag, f_max=20.0) == pytest.approx(12.6)


def test_dominant_frequency_ignores_points_above_f_max() -> None:
    freq = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    mag = np.array([1.0, 2.0, 3.0, 100.0, 5.0, 1000.0])
    # The global max (1000 at f=30) is outside f_max=20 and must be ignored.
    assert dominant_frequency(freq, mag, f_max=20.0) == pytest.approx(10.0)


@pytest.mark.parametrize("f", [3.49, 8.99, 20.01])
def test_harmonic_for_dominant_refuses_outside_table_7_1(f: float) -> None:
    """R-A finding 2: a sub-9 Hz dominant frequency must not silently get h=5."""
    with pytest.raises(ValueError, match="9-20 Hz"):
        harmonic_for_dominant(f)


def test_dominant_frequency_is_nan_when_nothing_below_f_max() -> None:
    """R-A finding 3: a high-frequency floor has no < 9 Hz dominant frequency."""
    assert np.isnan(dominant_frequency(np.array([10.0, 12.0]), np.array([1.0, 2.0]), f_max=9.0))
