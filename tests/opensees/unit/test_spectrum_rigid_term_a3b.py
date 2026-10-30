"""Chain A link A3b: the Sa-period override, mode kinds, the mode-set options.

* ``_modal_terms`` (A3-5, fork ADR 213 D3b): the modes sharing one
  Sa-period override sum into ONE term, in the slot of the group's first
  mode, at ``ω = 2π/T`` with that mode's damping, before any rule. Checked
  against a hand sum and a hand CQC whose ``ρ`` is Der Kiureghian's
  equal-damping form, written here independently of ``cqc_correlation``.
* ``ModalPropertiesResult.kinds`` (A3-4) reads ``modeKind``, eigen when absent.
* ``modal_after_gravity`` refuses mode-set options that do not fit together
  before anything is built (A3-3).
"""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.modal import ModalPropertiesResult
from apeGmsh.opensees.analysis.spectrum import _modal_terms, combine_modes

from tests.opensees.fixtures.fem_stub import make_two_node_beam

XI = 0.05
T_RIGID = 0.05
# four modes, two values each: two eigenmodes, two pseudo-modes at 0.05 s
R = np.array([[3.0, -1.0], [-2.0, 0.5], [0.7, 0.2], [-0.3, 0.4]])
OMEGA = np.array([6.0, 15.0, 45.0, 90.0])
SA_PERIODS = np.array([0.0, 0.0, T_RIGID, T_RIGID])


def _rho_equal_damping(w: np.ndarray, xi: float) -> np.ndarray:
    """Der Kiureghian (1981), equal damping: 8ξ²(1+r)r^1.5 / ((1-r²)² + 4ξ²r(1+r)²)."""
    n = w.size
    rho = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = w[j] / w[i]
                rho[i, j] = (8 * xi * xi * (1 + r) * r ** 1.5
                             / ((1 - r * r) ** 2 + 4 * xi * xi * r * (1 + r) ** 2))
    return rho


def test_without_override_every_mode_is_its_own_term() -> None:
    for periods in (None, np.zeros(4)):
        r, w, xi = _modal_terms(R, OMEGA, sa_periods=periods, damping=XI)
        assert r is R and w is OMEGA and xi == XI


def test_the_rigid_group_is_one_hand_summed_term_at_2pi_over_t() -> None:
    r, w, xi = _modal_terms(R, OMEGA, sa_periods=SA_PERIODS,
                            damping=[0.02, 0.03, 0.04, 0.05])
    np.testing.assert_array_equal(r, [R[0], R[1], R[2] + R[3]])
    np.testing.assert_allclose(w, [6.0, 15.0, 2 * math.pi / T_RIGID])
    np.testing.assert_array_equal(xi, [0.02, 0.03, 0.04])   # the group's first


def test_cqc_sees_the_rigid_mode_once() -> None:
    terms = np.array([R[0], R[1], R[2] + R[3]])
    w = np.array([6.0, 15.0, 2 * math.pi / T_RIGID])
    rho = _rho_equal_damping(w, XI)
    hand = np.sqrt(np.einsum("iv,ij,jv->v", terms, rho, terms))
    got = combine_modes(R, rule="CQC", omega=OMEGA, damping=XI,
                        sa_periods=SA_PERIODS)
    np.testing.assert_allclose(got, hand, rtol=1e-12)
    # the per-mode CQC differs: the override is what moves the answer
    own = combine_modes(R, rule="CQC", omega=OMEGA, damping=XI)
    assert not np.allclose(own, hand, rtol=1e-3)


def test_srss_adds_the_group_before_squaring() -> None:
    got = combine_modes(R, rule="SRSS", sa_periods=SA_PERIODS)
    hand = np.sqrt(R[0] ** 2 + R[1] ** 2 + (R[2] + R[3]) ** 2)
    np.testing.assert_allclose(got, hand, rtol=1e-12)


def test_distinct_periods_are_distinct_groups() -> None:
    r, w, _ = _modal_terms(R, OMEGA, sa_periods=[0.0, 0.03, 0.05, 0.03])
    np.testing.assert_array_equal(r, [R[0], R[1] + R[3], R[2]])
    np.testing.assert_allclose(w, [6.0, 2 * math.pi / 0.03, 2 * math.pi / 0.05])


def test_sa_periods_must_match_the_modes() -> None:
    with pytest.raises(ValueError, match="3 Sa periods for 4 modes"):
        combine_modes(R, rule="SRSS", sa_periods=[0.0, 0.0, 0.05])


# -- A3-4: mode kinds --------------------------------------------------------

def _props(**extra: Any) -> ModalPropertiesResult:
    return ModalPropertiesResult(
        eigenvalues=np.array([1.0, 4.0, 9.0]),
        properties={"eigenLambda": [1.0, 4.0, 9.0], **extra},
        _live=cast(Any, None),
    )


def test_kinds_default_to_eigen_without_mode_tags() -> None:
    assert _props().kinds == ("eigen", "eigen", "eigen")


def test_kinds_read_mode_kind() -> None:
    assert _props(modeKind=[0, 1, 2]).kinds == ("eigen", "residual", "ritz")
    with pytest.raises(ValueError, match="unknown modeKind 7"):
        _ = _props(modeKind=[0, 7, 0]).kinds


# -- A3-3: the mode-set options ----------------------------------------------

def _bridge() -> apeSees:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    def build() -> None:
        raise AssertionError("refused after the model was built")
    ops.build = build  # type: ignore[method-assign]
    return ops


@pytest.mark.parametrize(("kw", "match"), [
    ({"basis": "lanczos"}, "basis is 'eigen' or 'ritz'"),
    ({"ritz_dirs": (1,)}, "need basis='ritz'"),
    ({"residual_period": 0.05}, "residual_period needs residual="),
    ({"basis": "ritz", "ritz_dirs": (1,), "residual": (1,),
      "residual_period": 0.05}, "refused on a Ritz basis"),
])
def test_mode_set_options_refuse_before_the_build(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _bridge().modal_after_gravity(3, **kw)


def test_omega_must_match_the_modes() -> None:
    with pytest.raises(ValueError, match="3 omegas for 4 modes"):
        combine_modes(R, rule="CQC", omega=OMEGA[:3], damping=XI,
                      sa_periods=SA_PERIODS)


def test_combine_directions_takes_the_rigid_term() -> None:
    from apeGmsh.opensees.analysis.spectrum import combine_directions

    per_dir = np.stack([R, 0.5 * R])               # (direction, mode, value)
    got = combine_directions(per_dir, method="SRSS", rule="CQC", omega=OMEGA,
                             damping=XI, sa_periods=SA_PERIODS)
    each = [combine_modes(v, rule="CQC", omega=OMEGA, damping=XI,
                          sa_periods=SA_PERIODS) for v in per_dir]
    np.testing.assert_allclose(got, np.sqrt(each[0] ** 2 + each[1] ** 2),
                               rtol=1e-12)
    ungrouped = combine_directions(per_dir, method="SRSS", rule="CQC",
                                   omega=OMEGA, damping=XI)
    assert not np.allclose(got, ungrouped, rtol=1e-3)


def test_combined_period_tolerance_keeps_the_rigid_sa_fixed() -> None:
    """A6 with an override: the overridden modes' Sa does not move with the
    period, and both combinations take them as one rigid term."""
    from apeGmsh.opensees.analysis.spectrum_tolerance import (
        combined_period_tolerance,
    )

    tn, sa = [0.0, 0.1, 0.5, 2.0], [4.0, 9.8, 9.8, 2.0]
    spec = lambda t: np.interp(t, tn, sa)  # noqa: E731
    t = 2 * np.pi / OMEGA
    t_ref = t * np.array([1.02, 0.97, 1.05, 0.9])
    tol = combined_period_tolerance(
        spec, R, t, t_ref, quantity="force", rule="SRSS",
        sa_periods=SA_PERIODS)
    k = np.where(SA_PERIODS > 0, 1.0, np.interp(t_ref, tn, sa) / np.interp(t, tn, sa))
    v_ref = R * k[:, None]

    def srss(v: np.ndarray) -> np.ndarray:
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + (v[2] + v[3]) ** 2)

    ours, ref = srss(R), srss(v_ref)
    np.testing.assert_allclose(tol.period, np.abs(ours - ref) / ref, rtol=1e-12)


def _podium_set(alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The V3 podium with ``K`` x ``alpha``: three eigenmodes plus the two
    Rayleigh-Ritz pseudo-modes of the X residual, as ``(T, u1, f, K⁻¹Mr u1)``
    per mode per unit Sa: floor 1's displacement and the base force."""
    m = np.array([40.0] + [1.0] * 9)
    s = alpha * np.array([6e4] + [300.0] * 9)
    k = np.diag(s + np.append(s[1:], 0.0)) - np.diag(s[1:], 1) - np.diag(s[1:], -1)
    w = np.sqrt(m)
    lam, y = np.linalg.eigh(k / np.outer(w, w))
    phi = (y / w[:, None])[:, :3]
    iota = np.ones(10)
    r = iota - phi @ (phi.T @ (m * iota))
    x = np.linalg.solve(k, m * r)
    x = x - phi @ (phi.T @ (m * x))
    v = np.column_stack([r, x])
    # Rayleigh-Ritz on span{r, x}: the fork's pseudo-modes
    b = v.T @ (m[:, None] * v)
    c = np.linalg.cholesky(b)
    ci = np.linalg.inv(c)
    lam2, q = np.linalg.eigh(ci @ (v.T @ k @ v) @ ci.T)
    psi = v @ (ci.T @ q)
    modes = np.column_stack([phi, psi])
    lams = np.concatenate([lam[:3], lam2])
    gamma = modes.T @ (m * iota)
    u1 = modes[0] * gamma / lams
    force = (m[:, None] * modes).sum(axis=0) * gamma
    return (2 * np.pi / np.sqrt(lams), u1, force,
            np.linalg.solve(k, m * r)[:1])


@pytest.mark.parametrize(("quantity", "column"), [("displacement", 1), ("force", 2)])
def test_rigid_term_rescale_matches_a_stiffer_model(quantity: str, column: int) -> None:
    """The hand check behind keeping ``(T_ref/T)^p`` on overridden modes: the
    reference is an independent solve of the podium with ``K`` x 1.3. The
    group's displacement is the static residual, which scales by ``1/1.3``;
    its force does not move."""
    from apeGmsh.opensees.analysis.spectrum_tolerance import (
        combined_period_tolerance,
    )

    t_rigid = 0.05
    sa = np.array([9.81, 5.0])
    tn = [0.0, 10.0]
    spec = lambda t: np.interp(t, tn, sa)  # noqa: E731
    ours, ref = _podium_set(1.0), _podium_set(1.3)
    over = np.array([0.0, 0.0, 0.0, t_rigid, t_rigid])
    at = lambda t: np.where(over > 0.0, spec(t_rigid), spec(t))  # noqa: E731
    v = ours[column] * at(ours[0])
    v_ref = ref[column] * at(ref[0])
    # the group is the static residual: its sum, and how it scales
    group, group_ref = v[3:].sum(), v_ref[3:].sum()
    if quantity == "displacement":
        assert group == pytest.approx(ours[3][0] * spec(t_rigid), rel=1e-9)
        assert group_ref == pytest.approx(group / 1.3, rel=1e-9)
    else:
        assert group_ref == pytest.approx(group, rel=1e-9)
    tol = combined_period_tolerance(
        spec, v, ours[0], ref[0], quantity=quantity, rule="SRSS",
        sa_periods=over)
    hand = combine_modes(v, rule="SRSS", sa_periods=over)
    hand_ref = combine_modes(v_ref, rule="SRSS", sa_periods=over)
    assert float(tol.period) == pytest.approx(
        abs(hand - hand_ref) / hand_ref, rel=1e-8)
