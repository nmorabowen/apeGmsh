"""Chain A link A6: the period-propagated parity tolerance.

Pure numpy, on a synthetic spectrum with three known branches (a rising
power law ``Sa ∝ T^0.36``, a plateau, a ``1/T`` branch); every oracle is
written here by hand, apart from the module under test:

* gate 1: single mode on the rising branch, ``δT/T = +2.2 %``: the
  first-order and exact tolerances, and a flat 3 % gate that fails on the
  displacement where the period-aware gate passes;
* gate 2: the plateau (``s = 0``) and the ``1/T`` branch (``s = -1``), and
  the one-sided slopes at a corner;
* gate 3: a two-mode CQC re-evaluated with the reference periods, ``ρ``
  recomputed with the reference ``ω``;
* gate 4: the report carries base and period and they compound,
  ``(1 + base)(1 + period) - 1``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh.opensees.analysis.spectrum import DesignSpectrum
from apeGmsh.opensees.analysis.spectrum_tolerance import (
    ParityTolerance,
    combined_period_tolerance,
    period_tolerance,
    spectrum_slope,
)

S_RISE = 0.36
TC1, TC2 = 0.6, 1.0
DT = 0.022                       # our period is 2.2 % longer than the reference


def _sa(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.where(t < TC1, (t / TC1) ** S_RISE,
                    np.where(t <= TC2, 1.0, TC2 / t))


SPEC = DesignSpectrum(_sa, direction=1, scale=9.81)


# --- gate 1: single mode, rising power-law branch --------------------------

def test_slope_on_power_law_branch() -> None:
    assert spectrum_slope(SPEC, 0.4) == pytest.approx(S_RISE, rel=1e-6)


def test_single_mode_first_order() -> None:
    t_ref = 0.4
    t = t_ref * (1.0 + DT)
    force = period_tolerance(SPEC, t, t_ref, quantity="force",
                             method="first_order")
    disp = period_tolerance(SPEC, t, t_ref, quantity="displacement",
                            method="first_order")
    assert float(force.period) == pytest.approx(S_RISE * DT, rel=1e-6)
    assert float(disp.period) == pytest.approx((S_RISE + 2.0) * DT, rel=1e-6)


def test_single_mode_exact_ratio() -> None:
    t_ref = 0.4
    t = t_ref * (1.0 + DT)
    force = period_tolerance(SPEC, t, t_ref, quantity="force")
    disp = period_tolerance(SPEC, t, t_ref, quantity="displacement")
    # hand values: Sa ratio = 1.022**0.36, T² factor = 1.022**2
    assert float(force.period) == pytest.approx(1.022 ** 0.36 - 1.0, rel=1e-9)
    assert float(disp.period) == pytest.approx(1.022 ** 2.36 - 1.0, rel=1e-9)
    assert force.method == disp.method == "exact"


def test_flat_three_percent_fails_period_aware_passes() -> None:
    t_ref = 0.4
    t = t_ref * (1.0 + DT)
    # Sd = Sa T² / (4π²): ours at T, the reference program's at T_ref,
    # identical model otherwise; the whole gap is the period mismatch.
    sd_ours = float(SPEC.sa(t)) * t * t / (4.0 * math.pi ** 2)
    sd_ref = float(SPEC.sa(t_ref)) * t_ref * t_ref / (4.0 * math.pi ** 2)
    gap = abs(sd_ours - sd_ref) / sd_ref
    assert gap == pytest.approx(0.0526, abs=5e-4)
    assert gap > 0.03                                  # a flat 3 % gate fails
    tol = period_tolerance(SPEC, t, t_ref, quantity="displacement", base=0.01)
    assert tol.passes(sd_ours, sd_ref)                 # the period-aware gate
    # ...and a gap the periods do not explain still fails it
    assert not tol.passes(sd_ref * 1.08, sd_ref)


# --- gate 2: plateau, 1/T branch, corners ------------------------------------

def test_plateau_slope_zero() -> None:
    t_ref = 0.8
    t = t_ref * (1.0 + DT)
    assert spectrum_slope(SPEC, t_ref) == pytest.approx(0.0, abs=1e-9)
    force = period_tolerance(SPEC, t, t_ref, quantity="force",
                             method="first_order")
    disp = period_tolerance(SPEC, t, t_ref, quantity="displacement",
                            method="first_order")
    assert float(force.period) == pytest.approx(0.0, abs=1e-9)
    assert float(disp.period) == pytest.approx(2.0 * DT, rel=1e-6)
    exact = period_tolerance(SPEC, t, t_ref, quantity="displacement")
    assert float(exact.period) == pytest.approx(1.022 ** 2 - 1.0, rel=1e-9)


def test_descending_branch_slope_minus_one() -> None:
    t_ref = 2.0
    t = t_ref * (1.0 + DT)
    assert spectrum_slope(SPEC, t_ref) == pytest.approx(-1.0, rel=1e-6)
    force = period_tolerance(SPEC, t, t_ref, quantity="force",
                             method="first_order")
    disp = period_tolerance(SPEC, t, t_ref, quantity="displacement",
                            method="first_order")
    assert float(force.period) == pytest.approx(DT, rel=1e-6)    # |-1·δ|
    assert float(disp.period) == pytest.approx(DT, rel=1e-6)     # (−1+2)·δ
    exact = period_tolerance(SPEC, t, t_ref, quantity="force")
    assert float(exact.period) == pytest.approx(1.0 - 1.0 / 1.022, rel=1e-9)


def test_corner_one_sided_slopes() -> None:
    assert spectrum_slope(SPEC, TC1, side="left") == pytest.approx(
        S_RISE, rel=1e-5)
    assert spectrum_slope(SPEC, TC1, side="right") == pytest.approx(
        0.0, abs=1e-9)
    assert spectrum_slope(SPEC, TC1) == pytest.approx(S_RISE, rel=1e-5)
    assert spectrum_slope(SPEC, TC2) == pytest.approx(-1.0, rel=1e-5)
    # first order at a corner uses the side the period moved to
    up = period_tolerance(SPEC, TC1 * (1.0 + DT), TC1, quantity="force",
                          method="first_order")
    down = period_tolerance(SPEC, TC1 * (1.0 - DT), TC1, quantity="force",
                            method="first_order")
    assert float(up.period) == pytest.approx(0.0, abs=1e-9)
    assert float(down.period) == pytest.approx(S_RISE * DT, rel=1e-5)


def test_table_spectrum_slope() -> None:
    # Sa linear in T from (0.2, 1) to (0.6, 2): s = T Sa'/Sa
    tab = DesignSpectrum([0.2, 0.6, 1.0], [1.0, 2.0, 2.0], direction=1)
    t = 0.4
    assert spectrum_slope(tab, t) == pytest.approx(t * 2.5 / 1.5, rel=1e-5)


# --- gate 3: two-mode combination ---------------------------------------------

def _rho_equal(wi: float, wj: float, xi: float) -> float:
    """Der Kiureghian, equal damping (Chopra eq. 13.7.10)."""
    r = wj / wi
    return (8.0 * xi * xi * (1.0 + r) * r ** 1.5
            / ((1.0 - r * r) ** 2 + 4.0 * xi * xi * r * (1.0 + r) ** 2))


def _cqc2(a: float, b: float, rho: float) -> float:
    return math.sqrt(a * a + b * b + 2.0 * rho * a * b)


def test_two_mode_cqc_with_reference_periods() -> None:
    xi = 0.05
    t_ref = np.array([0.50, 0.46])
    t = np.array([0.50 * 1.022, 0.46 * 1.010])
    d = np.array([1.0, -0.6])                          # signed, ours

    sa = _sa(t)
    sa_ref = _sa(t_ref)
    d_ref = d * (sa_ref / sa) * (t_ref / t) ** 2       # rescaled by hand
    w = 2.0 * math.pi / t
    w_ref = 2.0 * math.pi / t_ref
    c_ours = _cqc2(d[0], d[1], _rho_equal(w[0], w[1], xi))
    c_ref = _cqc2(d_ref[0], d_ref[1], _rho_equal(w_ref[0], w_ref[1], xi))
    hand = abs(c_ours - c_ref) / c_ref

    tol = combined_period_tolerance(SPEC, d, t, t_ref,
                                    quantity="displacement", damping=xi)
    assert float(tol.period) == pytest.approx(hand, rel=1e-9)
    assert tol.method == "combined"
    # ρ with OUR ω on the rescaled values would be a different number
    stale = _cqc2(d_ref[0], d_ref[1], _rho_equal(w[0], w[1], xi))
    assert abs(abs(c_ours - stale) / stale - hand) > 1e-4

    force = combined_period_tolerance(SPEC, d, t, t_ref, quantity="force",
                                      damping=xi)
    f_ref = d * sa_ref / sa
    c_f = _cqc2(f_ref[0], f_ref[1], _rho_equal(w_ref[0], w_ref[1], xi))
    assert float(force.period) == pytest.approx(abs(c_ours - c_f) / c_f,
                                                rel=1e-9)


def test_combined_srss_and_value_axis() -> None:
    t_ref = np.array([0.50, 0.30])
    t = np.array([0.51, 0.31])
    per_mode = np.array([[1.0, 0.0], [0.5, 2.0]])      # (n_modes, n_values)
    k = (_sa(t_ref) / _sa(t)) * (t_ref / t) ** 2
    ref = np.sqrt(((per_mode * k[:, None]) ** 2).sum(axis=0))
    ours = np.sqrt((per_mode ** 2).sum(axis=0))
    tol = combined_period_tolerance(SPEC, per_mode, t, t_ref,
                                    quantity="displacement", rule="SRSS")
    np.testing.assert_allclose(tol.period, np.abs(ours - ref) / ref,
                               rtol=1e-12)


# --- gate 4: the report -------------------------------------------------------

def test_report_base_plus_period() -> None:
    tol = period_tolerance(SPEC, 0.4 * 1.022, 0.4, quantity="displacement",
                           base=0.01)
    assert isinstance(tol, ParityTolerance)
    assert tol.base == 0.01
    assert float(tol.total) == pytest.approx(
        1.01 * (1.0 + float(tol.period)) - 1.0, rel=1e-15)
    # compounding is what keeps it exact: 1 % base on a 5.27 % period part
    # is 6.32 % (1.01 · 1.022**2.36 - 1), not the first-order sum 6.27 %
    assert float(tol.total) == pytest.approx(1.01 * 1.022 ** 2.36 - 1.0,
                                             rel=1e-12)
    assert float(tol.total) == pytest.approx(0.0632, abs=5e-5)
    text = str(tol)
    assert "base" in text and "period" in text and "exact" in text
    assert f"{100 * float(tol.period):.2f}" in text


def test_report_array_total_and_passes() -> None:
    t_ref = np.array([0.50, 0.30])
    t = np.array([0.51, 0.31])
    per_mode = np.array([[1.0, 0.0], [0.5, 2.0]])
    tol = combined_period_tolerance(SPEC, per_mode, t, t_ref,
                                    quantity="force", rule="SRSS", base=0.02)
    np.testing.assert_allclose(tol.total, 1.02 * (1.0 + tol.period) - 1.0,
                               rtol=1e-15)
    ref = np.array([1.0, 2.0])
    assert tol.passes(ref * (1.0 + 0.019), ref)
    assert not tol.passes(ref * (1.0 + tol.total + 0.01), ref)


def test_refusals() -> None:
    with pytest.raises(ValueError, match="quantity"):
        period_tolerance(SPEC, 0.4, 0.4, quantity="stress")
    with pytest.raises(ValueError, match="> 0"):
        spectrum_slope(SPEC, 0.0)
    with pytest.raises(ValueError, match="base"):
        period_tolerance(SPEC, 0.4, 0.4, quantity="force", base=-0.1)
    with pytest.raises(ValueError, match="periods"):
        combined_period_tolerance(SPEC, [1.0, 2.0], [0.4], [0.4, 0.3],
                                  quantity="force", rule="SRSS")


def test_cqc_without_damping_is_refused() -> None:
    with pytest.raises(ValueError, match="rule='CQC' needs damping="):
        combined_period_tolerance(SPEC, [1.0, -0.6], [0.51, 0.46],
                                  [0.50, 0.46], quantity="force")


@pytest.mark.parametrize("step", [0.0, -1e-6, 1.0, 2.0, float("nan")])
def test_slope_rel_step_is_validated(step: float) -> None:
    with pytest.raises(ValueError, match="rel_step"):
        spectrum_slope(SPEC, 0.4, rel_step=step)


def test_spectrum_with_at_method() -> None:
    class AtOnly:                      # a codeSpectra-like object, not callable
        def at(self, t: np.ndarray) -> np.ndarray:
            return _sa(t)

    assert spectrum_slope(AtOnly(), 0.4) == pytest.approx(S_RISE, rel=1e-6)
    tol = period_tolerance(AtOnly(), 0.4 * 1.022, 0.4, quantity="force")
    assert float(tol.period) == pytest.approx(1.022 ** 0.36 - 1.0, rel=1e-9)
    with pytest.raises(TypeError, match="spectrum must be"):
        spectrum_slope(object(), 0.4)
