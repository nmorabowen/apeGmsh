"""Chain A link A5 (gaps 8, 10): signed and directional combinations.

Pure numpy, on hand-built per-mode values:

* gate 1: a two-mode CQC with a signed cross term, against the closed form;
* gate 4: the 100/30 sign permutations of a 2-direction case, their count
  and values, and SRSS of directions, by hand;
* gate 5: the cancellation report of two nearly cancelling modes;
* the per-mode storey forces and the inertia / base-reaction balance on a
  hand-built result (the live oracle is
  ``tests/opensees/live/test_spectrum_storey_forces_live.py``).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees.analysis.spectrum import (
    BASE_SHEAR,
    DesignSpectrum,
    DirectionalCases,
    InertiaBalanceWarning,
    NodeDisp,
    SpectrumModalResult,
    combine_directions,
    combine_modes,
    modal_cancellation,
)

W = np.array([2.0 * math.pi / 1.0, 2.0 * math.pi / 0.9])   # close modes
XI = 0.05


def _rho12(wi: float, wj: float, xi: float) -> float:
    """Der Kiureghian's equal-damping form (Chopra, Dynamics of Structures,
    eq. 13.7.10), written apart from the general one that the fork's rhoCQC
    and cqc_correlation use: ``8ξ²(1+r)r^1.5 / ((1-r²)² + 4ξ²r(1+r)²)``."""
    r = wj / wi
    return 8.0 * xi * xi * (1.0 + r) * r ** 1.5 / (
        (1.0 - r * r) ** 2 + 4.0 * xi * xi * r * (1.0 + r) ** 2)


def test_gate1_two_mode_cqc_closed_form_with_signed_cross_term() -> None:
    r = np.array([3.0, -2.0])
    rho = _rho12(W[0], W[1], XI)
    assert 0.3 < rho < 0.9                       # close modes: a real cross term
    want = math.sqrt(3.0 ** 2 + (-2.0) ** 2 + 2.0 * rho * 3.0 * (-2.0))
    got = float(combine_modes(r, rule="CQC", omega=W, damping=XI))
    assert got == pytest.approx(want, rel=1e-12, abs=0.0)
    srss = float(combine_modes(r, rule="SRSS"))
    assert srss == pytest.approx(math.sqrt(13.0), rel=1e-12)
    # opposite signs: the cross term lowers CQC below SRSS, the same values
    # with one sign flipped raise it above
    assert got < srss * (1.0 - 0.1)      # fails if the signs were dropped
    same = float(combine_modes(np.abs(r), rule="CQC", omega=W, damping=XI))
    assert same == pytest.approx(
        math.sqrt(13.0 + 12.0 * rho), rel=1e-12)
    assert same > srss * (1.0 + 0.1)
    # the same through the result (a signed derived quantity, per spectrum)
    # and through the per-mode directional route
    res = _result(F, R, omega=W)
    got_res = res.combine_values(r[None, :, None], damping=XI)
    assert float(got_res[0, 0]) == pytest.approx(want, rel=1e-12, abs=0.0)
    got_dir = combine_directions(r[None, :], rule="CQC", omega=W, damping=XI,
                                 method="SRSS")
    assert float(got_dir) == pytest.approx(want, rel=1e-12, abs=0.0)


def test_gate4_100_30_permutations_count_values_and_srss() -> None:
    # two directions, two values each (say a reaction and a drift)
    ex, ey = np.array([10.0, 2.0]), np.array([4.0, 5.0])
    cases = combine_directions(np.stack([ex, ey]), permutations=True)
    assert isinstance(cases, DirectionalCases)
    assert cases.factors.shape == (8, 2) and cases.values.shape == (8, 2)
    hand = [(sx * (1.0 if lead == 0 else 0.3), sy * (1.0 if lead == 1 else 0.3))
            for lead in (0, 1) for sx in (1.0, -1.0) for sy in (1.0, -1.0)]
    np.testing.assert_array_equal(cases.factors, np.array(hand))
    np.testing.assert_allclose(
        cases.values, [[a * ex[i] + b * ey[i] for i in range(2)]
                       for a, b in hand], rtol=1e-15)
    np.testing.assert_array_equal(cases.leading, [0, 0, 0, 0, 1, 1, 1, 1])
    # every pair of factors appears once, and the envelope is the scalar rule
    assert len({tuple(f) for f in cases.factors.tolist()}) == 8
    np.testing.assert_allclose(cases.envelope, [10.0 + 1.2, 5.0 + 0.6])
    np.testing.assert_allclose(
        cases.envelope, combine_directions(np.stack([ex, ey])))
    np.testing.assert_allclose(
        combine_directions(np.stack([ex, ey]), method="SRSS"),
        [math.sqrt(116.0), math.sqrt(29.0)], rtol=1e-15)
    three = combine_directions(np.ones((3, 1)), permutations=True)
    assert three.factors.shape == (24, 3)
    with pytest.raises(ValueError, match="SRSS of directions"):
        combine_directions(np.stack([ex, ey]), method="SRSS",
                           permutations=True)


def test_combine_directions_takes_per_mode_sets() -> None:
    per_mode = np.array([[[3.0], [-2.0]],                 # direction X
                         [[1.0], [1.5]]])                 # direction Y
    want = np.stack([combine_modes(v, rule="CQC", omega=W, damping=XI)
                     for v in per_mode])
    got = combine_directions(per_mode, rule="CQC", omega=W, damping=XI,
                             method="SRSS")
    np.testing.assert_allclose(got, np.sqrt(np.sum(want ** 2, axis=0)),
                               rtol=1e-15)
    cases = combine_directions(per_mode, rule="SRSS", permutations=True)
    np.testing.assert_allclose(cases.envelope,
                               [math.sqrt(13.0) + 0.3 * math.sqrt(3.25)])
    with pytest.raises(ValueError, match="mode axis"):
        combine_directions(np.ones(2), rule="SRSS")


def test_gate5_cancellation_ratio_and_signed_contributions() -> None:
    w = 2.0 * math.pi / np.array([1.0, 0.98])       # a close pair
    r = np.array([5.0, -4.6])                       # of opposite sign
    rho = _rho12(w[0], w[1], XI)
    combined = math.sqrt(25.0 + 4.6 ** 2 - 2.0 * rho * 5.0 * 4.6)
    rep = modal_cancellation(r, omega=w, damping=XI, angles=[2.0, 91.0])
    assert rep.combined == pytest.approx(combined, rel=1e-12)
    assert rep.abs_sum == pytest.approx(9.6, rel=1e-15)
    assert rep.ratio == pytest.approx(combined / 9.6, rel=1e-12)
    assert rep.ratio < 0.2                          # the terms cancel
    want = np.array([5.0 * (5.0 - rho * 4.6), -4.6 * (rho * 5.0 - 4.6)])
    np.testing.assert_allclose(rep.contributions, want / combined, rtol=1e-12)
    assert rep.contributions[1] < 0.0               # mode 2 cancels mode 1
    assert sum(rep.contributions) == pytest.approx(combined, rel=1e-12)
    np.testing.assert_array_equal(rep.terms, r)
    np.testing.assert_array_equal(rep.angles, [2.0, 91.0])
    assert rep.rule == "CQC"
    srss = modal_cancellation(r, rule="SRSS")
    np.testing.assert_allclose(srss.contributions, r * r / srss.combined)
    assert modal_cancellation(np.zeros(2), rule="SRSS").ratio == 0.0
    with pytest.raises(ValueError, match="one quantity"):
        modal_cancellation(np.ones((2, 2)), rule="SRSS")


# -- a hand-built result -------------------------------------------------------

#: A 2-D stick: nodes 1 (base, fixed), 2 at y = 3, 3 at y = 6; masses
#: (m, m, J) at 2 and 3; two modes, one spectrum along X.
COORDS = np.array([[0.0, 3.0], [1.0, 6.0]])
OMEGA = np.array([5.0, 12.0])


def _result(inertia: np.ndarray, reaction: np.ndarray,
            disp: np.ndarray | None = None,
            gamma: tuple[np.ndarray, np.ndarray] | None = None,
            omega: np.ndarray = OMEGA) -> SpectrumModalResult:
    gx, gy = gamma if gamma is not None else (np.ones(2), np.zeros(2))
    props = SimpleNamespace(
        omega=omega,
        participation_factors=lambda c: {"MX": gx, "MY": gy}[c])
    modal = SimpleNamespace(modal=props, periods=2.0 * np.pi / omega)
    per_mode = {BASE_SHEAR: reaction}
    labels: dict[str, Any] = {BASE_SHEAR: None}
    if disp is not None:
        per_mode["disp"] = disp
        labels["disp"] = NodeDisp(nodes=[2, 3], dofs=[1]).labels
    return SpectrumModalResult(
        modal=cast(Any, modal),
        spectra=(DesignSpectrum([0.0], [1.0], direction=1),),
        gravity_values={}, per_mode=per_mode, labels=labels,
        restore_drift=(0.0, 0.0), inertia=inertia, inertia_nodes=(2, 3),
        inertia_coords=COORDS)


F = np.array([[[[2.0, 0.0, 0.5], [3.0, 0.0, 0.0]],      # mode 1: Fx, Fy, Mz
               [[1.5, 0.0, 0.0], [-1.0, 0.0, 0.0]]]])    # mode 2
R = -F[..., :2].sum(axis=2)                              # (1, 2, 2)


def test_storey_forces_sum_the_inertia_above_each_cut() -> None:
    sf = _result(F, R).storey_forces([0.0, 3.0])
    assert sf.shear.shape == (1, 2, 2, 2) and sf.moment.shape == (1, 2, 2, 1)
    np.testing.assert_allclose(sf.shear[0, :, 0, 0], [5.0, 0.5])
    np.testing.assert_allclose(sf.shear[0, :, 1, 0], [3.0, -1.0])
    # M = Σ (x Fy - (y - z) Fx) + J: base cut, mode 1
    assert sf.moment[0, 0, 0, 0] == pytest.approx(-(3 * 2.0 + 6 * 3.0) + 0.5)
    assert sf.moment[0, 1, 1, 0] == pytest.approx(-(3.0 * -1.0))
    # combined from the per-mode values, not from combined inertia
    res = _result(F, R)
    np.testing.assert_allclose(
        res.combine_values(sf.shear, rule="CQC", damping=XI)[0, 0, 0],
        combine_modes([5.0, 0.5], rule="CQC", omega=OMEGA, damping=XI))


def test_inertia_balance_is_checked_against_the_base_reaction() -> None:
    assert float(np.max(_result(F, R).inertia_balance())) == 0.0
    bad = R.copy()
    bad[0, 1, 0] *= 1.0 + 5e-4                           # mode 2 off by 5e-4
    res = _result(F, bad)
    # relative to the spectrum's largest modal base force (mode 1's 5.0),
    # not to mode 2's own 0.5
    np.testing.assert_allclose(res.inertia_balance()[0],
                               [0.0, 0.5 * 5e-4 / 5.0], rtol=1e-9)
    res.check_inertia_balance(rtol=1e-3)                 # inside the 1e-3
    bad[0, 1, 0] = -0.5 * (1.0 + 2e-2)
    res = _result(F, bad)
    with pytest.raises(RuntimeError, match="mode 2 under spectrum 0"):
        res.storey_forces([0.0])
    with pytest.warns(InertiaBalanceWarning, match="mode 2"):
        res.check_inertia_balance(warn=True)
    res.storey_forces([0.0], check=False)


def test_drifts_are_per_mode_differences() -> None:
    disp = np.array([[[0.01, 0.025], [0.004, -0.003]]])  # (1, 2 modes, u2 u3)
    res = _result(F, R, disp=disp)
    d = res.drifts("disp", [(3, 2)], dof=1, heights=[3.0])
    np.testing.assert_allclose(d[0, :, 0], [0.015 / 3.0, -0.007 / 3.0])
    combined = res.combine_values(d, rule="SRSS")[0, 0]
    differenced = (res.combine("disp", rule="SRSS")[0, 1]
                   - res.combine("disp", rule="SRSS")[0, 0]) / 3.0
    assert combined == pytest.approx(math.hypot(0.015, 0.007) / 3.0)
    assert abs(combined - differenced) > 0.1 * combined
    with pytest.raises(ValueError, match="node1.u1"):
        res.drifts("disp", [(2, 1)], dof=1)
    with pytest.raises(ValueError, match="n_spectra, n_modes"):
        res.combine_values(np.ones((2, 1)), rule="SRSS")


def test_cancellation_reads_the_principal_axis_angles() -> None:
    disp = np.array([[[0.01, 0.025], [0.004, -0.003]]])
    res = _result(F, R, disp=disp,
                  gamma=(np.array([1.0, 0.0]), np.array([1.0, -2.0])))
    rep = res.cancellation("disp", 1, rule="SRSS")
    np.testing.assert_allclose(rep.angles, [45.0, 90.0])   # modulo 180
    np.testing.assert_array_equal(rep.terms, [0.025, -0.003])
