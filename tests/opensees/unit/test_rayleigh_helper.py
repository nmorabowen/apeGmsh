"""Unit tests for the Rayleigh ratio→coefficient helper (ADR 0053, D1).

Pure math — no bridge, no emitter, no openseespy.
"""
import math

import pytest

from apeGmsh.opensees.analysis.rayleigh import rayleigh_from_ratio


def _hand_fit(ratio: float, f_i: float, f_j: float) -> tuple[float, float]:
    w_i, w_j = 2 * math.pi * f_i, 2 * math.pi * f_j
    alpha = 2 * ratio * w_i * w_j / (w_i + w_j)
    beta = 2 * ratio / (w_i + w_j)
    return alpha, beta


def test_initial_is_default_and_places_beta_in_betaK0() -> None:
    alpha, bk, bk0, bkc = rayleigh_from_ratio(ratio=0.05, f_i=1.0, f_j=10.0)
    exp_alpha, exp_beta = _hand_fit(0.05, 1.0, 10.0)
    assert alpha == pytest.approx(exp_alpha)
    assert bk0 == pytest.approx(exp_beta)   # default slot = betaK0 (initial)
    assert bk == 0.0
    assert bkc == 0.0


def test_current_places_beta_in_betaK() -> None:
    alpha, bk, bk0, bkc = rayleigh_from_ratio(
        ratio=0.05, f_i=1.0, f_j=10.0, stiffness="current",
    )
    _, exp_beta = _hand_fit(0.05, 1.0, 10.0)
    assert bk == pytest.approx(exp_beta)
    assert bk0 == 0.0
    assert bkc == 0.0


def test_committed_places_beta_in_betaKc() -> None:
    alpha, bk, bk0, bkc = rayleigh_from_ratio(
        ratio=0.05, f_i=1.0, f_j=10.0, stiffness="committed",
    )
    _, exp_beta = _hand_fit(0.05, 1.0, 10.0)
    assert bkc == pytest.approx(exp_beta)
    assert bk == 0.0
    assert bk0 == 0.0


def test_realises_target_ratio_at_both_control_frequencies() -> None:
    # The defining property: ζ(ω) = α/(2ω) + β·ω/2 must equal the target
    # at BOTH control frequencies (this is what the two-point fit buys).
    ratio, f_i, f_j = 0.03, 2.0, 15.0
    alpha, _, beta0, _ = rayleigh_from_ratio(ratio=ratio, f_i=f_i, f_j=f_j)
    for f in (f_i, f_j):
        w = 2 * math.pi * f
        zeta = alpha / (2 * w) + beta0 * w / 2
        assert zeta == pytest.approx(ratio)


def test_frequency_order_is_symmetric() -> None:
    a = rayleigh_from_ratio(ratio=0.05, f_i=1.0, f_j=10.0)
    b = rayleigh_from_ratio(ratio=0.05, f_i=10.0, f_j=1.0)
    assert a == pytest.approx(b)


@pytest.mark.parametrize("f_i, f_j", [(0.0, 10.0), (-1.0, 10.0), (1.0, 0.0)])
def test_nonpositive_frequency_raises(f_i: float, f_j: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        rayleigh_from_ratio(ratio=0.05, f_i=f_i, f_j=f_j)


def test_equal_frequencies_raise() -> None:
    with pytest.raises(ValueError, match="differ"):
        rayleigh_from_ratio(ratio=0.05, f_i=5.0, f_j=5.0)


def test_unknown_stiffness_raises() -> None:
    with pytest.raises(ValueError, match="stiffness"):
        rayleigh_from_ratio(
            ratio=0.05, f_i=1.0, f_j=10.0,
            stiffness="tangent",  # type: ignore[arg-type]
        )
