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
