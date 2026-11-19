"""Chain A link A3a: the chain M verbs, run through the bridge on the fork.

The model is the fork's own V3 podium (WP-213), declared through
``apeSees``: a 10-DOF shear building of ``zeroLength`` springs whose first
storey carries 40x the mass and 200x the stiffness of the others, so three
eigenmodes carry only ~18 % of the X mass. On it:

* ``apeSees.residual_modes((1,), num_modes=3)`` appends two pseudo-modes
  and brings the X mass ratio to 100 %; with ``period=0.05`` the set is
  the same (the override changes only what RSA reads);
* ``apeSees.ritz_vectors(3, dirs=(1,))`` installs three Ritz vectors whose
  values bound the eigenvalues from above (Rayleigh-Ritz);
* ``apeSees.response_spectrum_analysis(require_mass=...)`` passes just
  below the three-mode ratio and is refused by the fork just above it.

The lane is ``live-fork`` (``ladruno_fork``); each test skips with the A4
probe's refusal only on a build without chain M, which the pin
(``ladruno-v20261029``) has.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._target import ForkTooOldError, rsa_features_of
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from apeGmsh.opensees.emitter.live import get_ops

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

pytestmark = [pytest.mark.ladruno_fork, pytest.mark.live]

N = 10
M = [40.0] + [1.0] * (N - 1)
K = [300.0 * 200.0] + [300.0] * (N - 1)
TN = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
SA = [4.0, 9.81, 9.81, 5.9, 2.9, 1.2]


@pytest.fixture(autouse=True)
def _wipe() -> Iterator[None]:
    """Leave the shared engine empty for the next test."""
    yield
    get_ops().wipe()


def _require(*features: str) -> None:
    feats = rsa_features_of(get_ops())
    for feature in features:
        try:
            feats.require(feature)
        except ForkTooOldError as exc:
            pytest.skip(str(exc))


def podium() -> apeSees:
    """The V3 podium: node 1 the base, node i+1 floor i, springs in X."""
    nodes = _NodesStub(
        ids=list(range(1, N + 2)),
        coords=[(0.0, 0.0, 0.0)] * (N + 1),
        node_pgs={"Base": [1], "Floors": list(range(2, N + 2))},
    )
    groups = {f"S{i}": _ElementGroupView(ids=(i + 1,), connectivity=((i + 1, i + 2),))
              for i in range(N)}
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    for i in range(N):
        mat = ops.uniaxialMaterial.ElasticMaterial(E=K[i])
        ops.element.ZeroLength(pg=f"S{i}",
                               mat_dirs=(ZeroLengthMatDir(material=mat, dof=1),))
        ops.mass(nodes=[i + 2], values=(M[i], 0.0, 0.0))
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(pg="Floors", dofs=(0, 1, 1))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _hand_eigenvalues() -> np.ndarray:
    k = np.asarray(K)
    kk = np.diag(k + np.append(k[1:], 0.0)) - np.diag(k[1:], 1) - np.diag(k[1:], -1)
    m = np.asarray(M)
    return np.linalg.eigvalsh(kk / np.sqrt(np.outer(m, m)))


def _ratio_x(res: Any) -> float:
    return float(res.cumulative_mass_ratios("MX")[-1]) / 100.0


def test_residual_modes_complete_the_x_mass() -> None:
    _require("residualModes")
    eig = podium().modal_properties(3, solver="-fullGenLapack")
    assert _ratio_x(eig) < 0.25                     # the podium's mass is missing
    res = podium().residual_modes((1,), num_modes=3, solver="-fullGenLapack")
    assert len(res.eigenvalues) == 5                # 3 eigenmodes + 2 pseudo-modes
    np.testing.assert_allclose(res.eigenvalues[:3], _hand_eigenvalues()[:3],
                               rtol=1e-9)
    assert _ratio_x(res) == pytest.approx(1.0, abs=1e-10)
    assert res._live.ops.residualModes("-kinds") == [0, 0, 0, 1, 1]
    fixed = podium().residual_modes((1,), num_modes=3, period=0.05,
                                    solver="-fullGenLapack")
    np.testing.assert_allclose(fixed.eigenvalues, res.eigenvalues, rtol=1e-12)


def test_ritz_vectors_bound_the_eigenvalues() -> None:
    _require("ritzVectors")
    res = podium().ritz_vectors(3, dirs=(1,))
    assert len(res.eigenvalues) == 3
    assert res._live.ops.residualModes("-kinds") == [2, 2, 2]
    for z, e in zip(sorted(res.eigenvalues), _hand_eigenvalues()):
        assert z >= e * (1.0 - 1e-9)
    assert 0.0 < _ratio_x(res) <= 1.0 + 1e-12


def test_require_mass_reaches_the_fork_and_refuses_a_shortfall(
    capfd: pytest.CaptureFixture[str],
) -> None:
    _require("responseSpectrumAnalysis -requireMass")
    ratio = _ratio_x(podium().modal_properties(3, solver="-fullGenLapack"))
    assert 0.18 < ratio < 0.19                      # three modes carry ~18.4 %
    kw: dict[str, Any] = dict(periods=TN, accels=SA, combine="SRSS",
                              num_modes=3, solver="-fullGenLapack")
    podium().response_spectrum_analysis(1, require_mass=0.18, **kw)
    capfd.readouterr()
    with pytest.raises(Exception):
        podium().response_spectrum_analysis(1, require_mass=0.19, **kw)
    err = capfd.readouterr().err
    assert f"the modes carry {100 * ratio:.2f}% of the free mass in direction 1" in err
    assert "below -requireMass 19.00%" in err
