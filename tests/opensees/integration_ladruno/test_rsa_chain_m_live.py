"""Chain A link A4-2: the fork's chain M and P1 commands on the pinned build.

The model is the fork's own V3 podium (WP-213): a 10-DOF shear building
of ``zeroLength`` springs whose first storey carries 40x the mass and
200x the stiffness of the others, so three eigenmodes miss most of the
mass. On it:

* ``residualModes -dir 1`` brings the X mass ratio to 100 %;
* ``ritzVectors`` runs and installs a Ritz set whose values bound the
  eigenvalues from above (Rayleigh-Ritz);
* ``responseSpectrumPerMode``: the base reaction of every eigenmode is
  minus its total modal inertia (iota^T K phi = 0);
* ``responseSpectrumAnalysis -requireMass`` refuses the three-mode
  shortfall and accepts the completed set.

Each test skips with the capability probe's own refusal when the build
lacks the command (``ladruno-v20261023`` has none of them), so the lane
stays green on an older pin and runs them once the pin carries chain M.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees._target import ForkTooOldError, rsa_features_of
from apeGmsh.opensees.emitter.live import get_ops

pytestmark = pytest.mark.ladruno_fork

N = 10
M = [40.0] + [1.0] * (N - 1)
K = [300.0 * 200.0] + [300.0] * (N - 1)
TN = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
SA = [4.0, 9.81, 9.81, 5.9, 2.9, 1.2]
SPEC = ["-Tn", *TN, "-Sa", *SA]


@pytest.fixture(autouse=True)
def _wipe() -> Iterator[None]:
    """Leave the shared engine empty for the next test."""
    yield
    get_ops().wipe()


def engine(*features: str) -> Any:
    """The in-process engine; skip with the probe's refusal if it lacks one."""
    live = get_ops()
    feats = rsa_features_of(live)
    for feature in features:
        try:
            feats.require(feature)
        except ForkTooOldError as exc:
            pytest.skip(str(exc))
    return live


def podium(ops: Any) -> None:
    """The V3 podium with a static analysis defined (``ritzVectors`` needs one)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(0, 0.0, 0.0)
    ops.fix(0, 1, 1, 1)
    for i in range(1, N + 1):
        ops.node(i, 0.0, 0.0)
        ops.fix(i, 0, 1, 1)
        ops.mass(i, M[i - 1], 0.0, 0.0)
        ops.uniaxialMaterial("Elastic", i, K[i - 1])
        ops.element("zeroLength", i, i - 1, i, "-mat", i, "-dir", 1)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def mass_ratio_x(ops: Any) -> float:
    """Cumulative X effective-mass ratio of the installed mode set (0..1)."""
    return float(ops.modalProperties("-return")["partiMassRatiosCumuMX"][-1]) / 100.0


def test_residual_modes_reach_full_mass() -> None:
    ops = engine("residualModes")
    podium(ops)
    ops.eigen("-fullGenLapack", 3)
    assert mass_ratio_x(ops) < 0.25          # the podium's mass is missing
    lam_r = ops.residualModes("-dir", 1)
    assert len(lam_r) == 2
    assert ops.residualModes("-kinds") == [0, 0, 0, 1, 1]
    assert mass_ratio_x(ops) == pytest.approx(1.0, abs=1e-10)


def test_ritz_vectors_run_and_bound_the_eigenvalues() -> None:
    ops = engine("ritzVectors")
    podium(ops)
    lam_e = list(ops.eigen("-fullGenLapack", 3))
    podium(ops)
    lam_z = list(ops.ritzVectors(3, "-dir", 1))
    assert len(lam_z) == 3
    assert ops.residualModes("-kinds") == [2, 2, 2]
    # Rayleigh-Ritz on a subspace: the i-th Ritz value >= the i-th eigenvalue
    for z, e in zip(sorted(lam_z), lam_e):
        assert z >= e * (1.0 - 1e-9)
    assert 0.0 < mass_ratio_x(ops) <= 1.0 + 1e-12


def test_per_mode_base_reaction_balances_the_modal_inertia() -> None:
    ops = engine("responseSpectrumPerMode")
    podium(ops)
    ops.eigen("-fullGenLapack", 3)
    ops.modalProperties()
    out = ops.responseSpectrumPerMode(
        *SPEC, "-dir", 1, "-reactions", "-inertia", "all", *range(1, N + 1),
    )
    nd, nm, nc = (int(v) for v in out["shape"])
    assert (nd, nm, nc) == (1, 3, 3)          # one direction, 3 modes, Fx Fy Mz
    reac = np.array(out["reactions"]).reshape(nd, nm, nc)
    inertia = np.array(out["inertia:all"]).reshape(nd, nm, nc)
    fx = inertia[0, :, 0]
    assert np.all(np.abs(fx) > 0.0)
    np.testing.assert_allclose(reac[0, :, 0], -fx, rtol=1e-9,
                               atol=1e-12 * np.abs(fx).max())


def test_require_mass_refuses_a_shortfall() -> None:
    ops = engine("responseSpectrumAnalysis -requireMass", "residualModes")
    podium(ops)
    ops.eigen("-fullGenLapack", 3)
    ops.modalProperties()
    with pytest.raises(Exception):
        ops.responseSpectrumAnalysis(1, *SPEC, "-mode", 1, "-requireMass", 0.9)
    # the completed set carries the mass, so the same guard passes
    podium(ops)
    ops.eigen("-fullGenLapack", 3)
    ops.residualModes("-dir", 1)
    ops.modalProperties()
    ops.responseSpectrumAnalysis(1, *SPEC, "-mode", 1, "-requireMass", 1.0 - 1e-9)
