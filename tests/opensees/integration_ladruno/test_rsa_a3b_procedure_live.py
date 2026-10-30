"""Chain A link A3b: residual and Ritz mode sets in the Python RSA path.

The model is the fork's own V3 podium (WP-213, ``tests/test_wp213_residual_
modes.py``), declared through ``apeSees`` in both plan directions (``ndf``
2, so only the base node is fixed: the fork test's ``ndf`` 3 with fixed
floor rotations would make ``_TotalReaction`` add the floors' free-DOF
residuals on the ``-increment`` route): a 10-DOF
shear building of ``zeroLength`` springs whose first storey carries 40x the
mass and 200x the stiffness of the others (Y twice as stiff as X), so three
eigenmodes miss most of the mass in both directions. The spectrum is the
fork test's ASCE-shaped table. No client data.

1. ``residual=(1, 2)``: every mode's base shear is its effective mass times
   Sa at its period, the pseudo-modes carry the residual mass, and the
   Python CQC base shear equals the fork's ``responseSpectrumAnalysis
   -combine CQC`` on the same mode set to 1e-6.
2. ``residual_period=0.05``: the pseudo-modes' base shears add to the
   residual mass times Sa(0.05), and the Python CQC still equals the fork's,
   which sums them into one rigid mode at 2π/0.05 (A3-5; without the
   grouping the two differ).
3. A Ritz basis from the gravity pattern: every mode is ``"ritz"`` and the
   static participation in the gravity load is 100 %.
4. ``spectrum_after_gravity(basis="ritz")`` end to end: kinds, the inertia
   balance, and the fork's RSA on the same Ritz set.

In 1, 2 and 4 every mode's storey shear (from ``m φ Γ Sa`` for an
eigenmode, the elastic force ``K ψ Γ Sa / λ`` for a residual or Ritz mode)
equals the storey spring's force.

Lane ``live-fork`` (``ladruno_fork``); each test skips through the A4 probe
only on a build without chain M or ``-increment``, which the pin has.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._target import ForkTooOldError, rsa_features_of
from apeGmsh.opensees.analysis.spectrum import (
    DesignSpectrum,
    InertiaBalanceWarning,
    NodeDisp,
)
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
M = np.array([40.0] + [1.0] * (N - 1))
KX = np.array([300.0 * 200.0] + [300.0] * (N - 1))
KY = 2.0 * KX
G = 9.81
XI = 0.05
N_MODES = 3
H = 3.0
T_RIGID = 0.05
SDS, SD1, TL = 1.0, 0.6, 8.0


def _sa(t: float) -> float:
    ts = SD1 / SDS
    t0 = 0.2 * ts
    if t < t0:
        s = SDS * (0.4 + 0.6 * t / t0)
    elif t <= ts:
        s = SDS
    elif t <= TL:
        s = SD1 / t
    else:
        s = SD1 * TL / t ** 2
    return s * G


TN = np.concatenate([np.linspace(0.0, 2.0, 201), np.linspace(2.05, 10.0, 160)])
SA = np.array([_sa(t) for t in TN])
SPECTRA = [DesignSpectrum(TN.tolist(), SA.tolist(), direction=d) for d in (1, 2)]


@pytest.fixture(autouse=True)
def _wipe() -> Iterator[None]:
    yield
    get_ops().wipe()


def _require(*features: str) -> Any:
    ops = get_ops()
    feats = rsa_features_of(ops)
    for feature in features:
        try:
            feats.require(feature)
        except ForkTooOldError as exc:
            pytest.skip(str(exc))
    return ops


def podium() -> tuple[apeSees, Any]:
    """The V3 podium in X and Y, node 1 the base, node j+1 floor j at
    height j·H (the springs act along their dirs only, so the elevations
    only place the storey cuts), with a gravity pattern (-m g along Y) and
    a static chain."""
    nodes = _NodesStub(
        ids=list(range(1, N + 2)),
        coords=[(0.0, j * H, 0.0) for j in range(N + 1)],
        node_pgs={"Base": [1], "Floors": list(range(2, N + 2))},
    )
    groups = {f"S{i}": _ElementGroupView(ids=(i + 1,), connectivity=((i + 1, i + 2),))
              for i in range(N)}
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    for i in range(N):
        mx = ops.uniaxialMaterial.ElasticMaterial(E=float(KX[i]))
        my = ops.uniaxialMaterial.ElasticMaterial(E=float(KY[i]))
        ops.element.ZeroLength(pg=f"S{i}", mat_dirs=(
            ZeroLengthMatDir(material=mx, dof=1),
            ZeroLengthMatDir(material=my, dof=2)))
        ops.mass(nodes=[i + 2], values=(float(M[i]), float(M[i])))
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear(), name="gravity") as p:
        for i in range(N):
            p.load(node=i + 2, forces=(0.0, -float(M[i]) * G))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops, p


def _fork_combined_base_shear(
    period: float | None, ritz: int | None = None,
) -> list[float]:
    """The fork's own RSA on the same mode set: raw commands, ``-combine CQC``.
    The base spring is the only path to the support, so the combined base
    shear is its stiffness times the combined floor-1 displacement."""
    ops = get_ops()
    out = []
    for d, k in ((1, KX), (2, KY)):
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.node(1, 0.0, 0.0)
        ops.fix(1, 1, 1)
        for i in range(N):
            ops.node(i + 2, 0.0, 0.0)
            ops.mass(i + 2, M[i], M[i])
            ops.uniaxialMaterial("Elastic", 2 * i + 1, KX[i])
            ops.uniaxialMaterial("Elastic", 2 * i + 2, KY[i])
            ops.element("zeroLength", i + 1, i + 1, i + 2,
                        "-mat", 2 * i + 1, 2 * i + 2, "-dir", 1, 2)
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-12, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ritz is None:
            ops.eigen("-fullGenLapack", N_MODES)
            ops.residualModes("-dir", 1, 2,
                              *(() if period is None else ("-period", period)))
        else:
            ops.ritzVectors(ritz, "-dir", 1, 2)
        ops.modalProperties()
        ops.responseSpectrumAnalysis(d, "-Tn", *TN.tolist(), "-Sa", *SA.tolist(),
                                     "-combine", "CQC", "-damp", XI)
        out.append(float(k[0]) * abs(float(ops.nodeDisp(2, d))))
    return out


def _run(period: float | None, **kw: Any) -> Any:
    ops, _ = podium()
    kw = kw or {"residual": (1, 2), "residual_period": period}
    return ops.spectrum_after_gravity(
        N_MODES if "ritz_dirs" not in kw else 2 * N_MODES, spectra=SPECTRA,
        extractors={"u": NodeDisp(list(range(2, N + 2)), dofs=(1, 2))},
        solver="-fullGenLapack", **kw)


def _check_storey_shear_is_the_spring_force(res: Any) -> None:
    """Per mode and spectrum, the storey shear from the modal (inertia or
    elastic) force equals the storey spring's force k (u_j - u_{j-1})."""
    st = res.storey_forces([(j - 1) * H for j in range(1, N + 1)], vertical=2)
    for s in range(2):
        u = res.modes("u", s).reshape(-1, N, 2)          # (mode, floor, dof)
        for d, k in ((0, KX), (1, KY)):
            du = np.diff(np.concatenate(
                [np.zeros((u.shape[0], 1)), u[:, :, d]], axis=1), axis=1)
            spring = du * k[None, :]                       # (mode, storey)
            scale = float(np.max(np.abs(spring)))
            np.testing.assert_allclose(st.shear[s, :, :, d], spring, rtol=1e-6,
                                       atol=1e-7 * scale)


def _check_modal_and_residual(res: Any, sa_of_pseudo: Any) -> None:
    kinds = np.asarray(res.kinds)
    assert list(kinds[:N_MODES]) == ["eigen"] * N_MODES
    pseudo = kinds == "residual"
    assert pseudo.sum() >= 2
    modal = res.modal.modal
    for i, (d, comp) in enumerate(((1, "MX"), (2, "MY"))):
        meff = np.abs(np.asarray(modal.properties[f"partiMass{comp}"]))
        assert float(np.sum(meff)) == pytest.approx(float(M.sum()), rel=1e-9)
        residual_mass = float(M.sum()) - float(np.sum(meff[~pseudo]))
        assert residual_mass > 0.5 * M.sum()           # the podium is missing
        vb = res.modes("base_shear", i)[:, d - 1]      # signed, per mode
        sa = SPECTRA[i].sa(np.where(pseudo, sa_of_pseudo, res.periods))
        # the modal part: each mode's base shear is its effective mass x Sa
        np.testing.assert_allclose(np.abs(vb), meff * sa, rtol=1e-6,
                                   atol=1e-9 * float(np.max(meff * sa)))
        if np.isscalar(sa_of_pseudo):
            # the rigid term: the pseudo-modes add to the residual mass x Sa(T)
            assert abs(float(np.sum(vb[pseudo]))) == pytest.approx(
                residual_mass * float(SPECTRA[i].sa(sa_of_pseudo)), rel=1e-6)


def test_residual_modes_rayleigh_periods_match_the_fork_rsa() -> None:
    _require("residualModes")
    res = _run(None)
    assert res.sa_periods is None
    _check_modal_and_residual(res, res.periods)
    _check_storey_shear_is_the_spring_force(res)
    np.testing.assert_allclose(res.base_shear(rule="CQC", damping=XI),
                               _fork_combined_base_shear(None), rtol=1e-6)


def test_residual_period_is_one_rigid_mode_as_in_the_fork_rsa() -> None:
    _require("residualModes")
    res = _run(T_RIGID)
    pseudo = np.asarray(res.kinds) == "residual"
    np.testing.assert_array_equal(res.sa_periods, np.where(pseudo, T_RIGID, 0.0))
    _check_modal_and_residual(res, T_RIGID)
    _check_storey_shear_is_the_spring_force(res)
    fork = _fork_combined_base_shear(T_RIGID)
    np.testing.assert_allclose(res.base_shear(rule="CQC", damping=XI), fork,
                               rtol=1e-6)
    # without the rigid grouping (each pseudo-mode its own CQC term at its
    # own ω) the Python path would not be the fork's
    ungrouped = np.asarray([
        float(np.sqrt(np.einsum(
            "i,ij,j->", v, _rho(res.omega), v)))
        for v in (res.modes("base_shear", i)[:, i] for i in range(2))])
    assert not np.allclose(ungrouped, fork, rtol=1e-3)


def _rho(w: np.ndarray) -> np.ndarray:
    from apeGmsh.opensees.analysis.spectrum import cqc_correlation
    return cqc_correlation(w, XI)


def test_ritz_basis_from_the_gravity_pattern() -> None:
    _require("ritzVectors")
    ops, gravity = podium()
    res = ops.modal_after_gravity(N_MODES, basis="ritz", ritz_patterns=[gravity])
    modal = res.modal
    assert modal.kinds == ("ritz",) * N_MODES
    # static participation of the gravity load in the Ritz set:
    # Σ (ψᵀf)² / ψᵀKψ over fᵀK⁻¹f, scale-free in ψ
    ky = np.diag(KY + np.append(KY[1:], 0.0)) - np.diag(KY[1:], 1) - np.diag(KY[1:], -1)
    f = -M * G
    psi = np.array([[modal.mode_shape(n, k + 1)[1] for k in range(N_MODES)]
                    for n in range(2, N + 2)])          # (floor, mode), Y
    part = sum(float(psi[:, k] @ f) ** 2 / float(psi[:, k] @ ky @ psi[:, k])
               for k in range(N_MODES))
    assert part / float(f @ np.linalg.solve(ky, f)) == pytest.approx(1.0, abs=1e-9)
    # Ritz values bound the Y eigenvalues from above (Rayleigh-Ritz)
    lam_y = np.linalg.eigvalsh(ky / np.sqrt(np.outer(M, M)))
    assert all(z >= e * (1 - 1e-9) for z, e in zip(sorted(modal.eigenvalues), lam_y))
    assert math.isfinite(res.gravity_drift()[0])


def test_ritz_basis_spectrum_end_to_end() -> None:
    """A3-6: ``spectrum_after_gravity(basis="ritz")`` carries the kinds, its
    storey forces are the spring forces, its inertia balances the base, and
    its CQC base shear is the fork's RSA on the same Ritz set."""
    _require("ritzVectors")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InertiaBalanceWarning)
        res = _run(None, basis="ritz", ritz_dirs=(1, 2))
    assert res.kinds == ("ritz",) * (2 * N_MODES)
    assert res.sa_periods is None
    assert float(np.max(res.inertia_balance())) < 1e-6
    _check_storey_shear_is_the_spring_force(res)
    np.testing.assert_allclose(res.base_shear(rule="CQC", damping=XI),
                               _fork_combined_base_shear(None, ritz=2 * N_MODES),
                               rtol=1e-6)
