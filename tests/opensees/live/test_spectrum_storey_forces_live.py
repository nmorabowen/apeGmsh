"""Chain A link A5-3, live: per-mode drifts and storey forces on a shear
building, against a hand calculation.

A 2-D stick of four storeys: one ``elasticBeamColumn`` per storey whose top
is held against rotation and vertical motion, so storey ``i`` is a shear
spring ``k_i = 12 EI / h³`` and the model is the textbook shear building
``K`` (tridiagonal), ``M = diag(m_i)``. From ``eigh`` by hand:
``u_k = φ_k Γ_k Sa(T_k) / λ_k``, the per-mode drift ``u_k,i - u_k,i-1``,
the per-mode storey shear ``Σ_{j ≥ i} m_j φ_jk Γ_k Sa(T_k)`` and its
overturning moment, each combined by CQC.

* gate 2: the driver's combined drifts and storey forces equal the hand
  CQC; the combined drift is not the difference of the combined
  displacements, and both are asserted;
* gate 3: each mode's base storey shear from inertia equals its base
  reaction (relative ``1e-3``; ``1e-9`` measured), on the static legs
  (stock openseespy) and on the fork's ``-increment`` legs.
"""
from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.spectrum import (
    BASE_SHEAR,
    DesignSpectrum,
    InertiaBalanceWarning,
    NodeDisp,
    SpectrumModalResult,
    combine_modes,
)

openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.live.test_spectrum_after_gravity_live import (  # noqa: E402,F401
    rsa_route,
)

E, I_COL, A_COL = 30e9, 0.0052, 0.25
H = (4.0, 3.0, 3.0, 3.0)                       # storey heights (m)
MASS = (30e3, 25e3, 25e3, 18e3)                # floor masses (kg)
I_SCALE = (1.6, 1.2, 1.0, 0.7)                 # column I per storey, x I_COL
TN = (0.0, 0.1, 0.4, 1.0, 3.0)
SA = (3.0, 7.5, 7.5, 3.0, 1.0)
XI = 0.05
N = len(H)
Z = np.concatenate([[0.0], np.cumsum(H)])      # node elevations, node i+1 at Z[i]


def _building() -> apeSees:
    nodes = _NodesStub(
        ids=list(range(1, N + 2)),
        coords=[(0.0, float(z), 0.0) for z in Z],
        node_pgs={"Base": [1], "Floors": list(range(2, N + 2))},
    )
    groups = {f"S{i}": _ElementGroupView(ids=(i + 1,), connectivity=((i + 1, i + 2),))
              for i in range(N)}
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    for i in range(N):
        ops.element.elasticBeamColumn(pg=f"S{i}", transf=tr, A=A_COL, E=E,
                                      Iz=I_SCALE[i] * I_COL)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(pg="Floors", dofs=(0, 1, 1))
    for i in range(N):
        ops.mass(nodes=[i + 2], values=(MASS[i], 0.0, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Floors", forces=(0.0, -1e4, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()
    return ops


def _hand() -> dict[str, np.ndarray]:
    k = np.array([12.0 * E * s * I_COL / h ** 3 for s, h in zip(I_SCALE, H)])
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += k[i]
        if i + 1 < N:
            K[i, i] += k[i + 1]
            K[i, i + 1] = K[i + 1, i] = -k[i + 1]
    m = np.asarray(MASS)
    lam, psi = np.linalg.eigh(K / np.sqrt(np.outer(m, m)))
    phi = psi / np.sqrt(m)[:, None]                       # (floor, mode)
    gamma = (phi.T @ m) / np.einsum("ik,i,ik->k", phi, m, phi)
    omega = np.sqrt(lam)
    sa = np.interp(2.0 * np.pi / omega, TN, SA)
    u = phi * (gamma * sa / lam)[None, :]                 # (floor, mode)
    drift = np.diff(np.vstack([np.zeros(N), u]), axis=0).T     # (mode, storey)
    force = (m[:, None] * phi * (gamma * sa)[None, :]).T       # (mode, floor)
    shear = np.cumsum(force[:, ::-1], axis=1)[:, ::-1]         # (mode, storey)
    arm = Z[1:][None, :] - Z[:-1][:, None]                     # (cut, floor)
    moment = -np.stack([(force * np.clip(arm[c], 0.0, None)).sum(axis=1)
                        for c in range(N)], axis=1)            # (mode, storey)
    return {"omega": omega, "u": u.T, "drift": drift, "shear": shear,
            "moment": moment, "k": k}


def _cqc(values: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return combine_modes(values, rule="CQC", omega=omega, damping=XI)


def _run() -> SpectrumModalResult:
    spec = DesignSpectrum(TN, SA, direction=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InertiaBalanceWarning)
        return _building().spectrum_after_gravity(
            N, spectra=[spec],
            extractors={"disp": NodeDisp(nodes=range(1, N + 2), dofs=(1,))},
            solver="-fullGenLapack")


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
def test_gate2_per_mode_drifts_and_storey_forces_meet_the_hand_cqc() -> None:
    res = _run()
    ref = _hand()
    np.testing.assert_allclose(res.omega, ref["omega"], rtol=1e-10)
    pairs = [(i + 2, i + 1) for i in range(N)]
    drift = res.drifts("disp", pairs, dof=1)[0]                # (mode, storey)
    np.testing.assert_allclose(drift, ref["drift"], rtol=1e-8,
                               atol=1e-9 * np.abs(ref["drift"]).max())
    combined = res.combine_values(res.drifts("disp", pairs, dof=1),
                                  damping=XI)[0]
    want = _cqc(ref["drift"], ref["omega"])
    np.testing.assert_allclose(combined, want, rtol=1e-8)
    # the difference of the combined displacements is a different number
    u = np.concatenate([[0.0], _cqc(ref["u"], ref["omega"])])
    differenced = np.diff(u)
    got_u = res.combine("disp", damping=XI)[0]
    np.testing.assert_allclose(np.diff(got_u), differenced, rtol=1e-8,
                               atol=1e-12)
    assert np.max(np.abs(combined - differenced) / combined) > 0.02
    sf = res.storey_forces(Z[:-1])
    shear = sf.shear[0, :, :, 0]                               # (mode, storey)
    np.testing.assert_allclose(shear, ref["shear"], rtol=1e-9)
    # equilibrium: the storey shear is the storey spring force, per mode
    np.testing.assert_allclose(shear, ref["drift"] * ref["k"][None, :],
                               rtol=1e-6)
    np.testing.assert_allclose(sf.moment[0, :, :, 0], ref["moment"], rtol=1e-9)
    np.testing.assert_allclose(
        res.combine_values(sf.shear, damping=XI)[0, :, 0],
        _cqc(ref["shear"], ref["omega"]), rtol=1e-9)
    np.testing.assert_allclose(
        res.combine_values(sf.moment, damping=XI)[0, :, 0],
        _cqc(ref["moment"], ref["omega"]), rtol=1e-9)


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
def test_gate3_base_storey_shear_from_inertia_equals_the_base_reaction() -> None:
    res = _run()
    assert res.inertia_nodes == tuple(range(2, N + 2))
    base = res.storey_forces([0.0]).shear[0, :, 0, :]          # (mode, ndm)
    reaction = res.modes(BASE_SHEAR)                           # (mode, ndm)
    assert np.all(np.abs(base[:, 0]) > 0.0)
    np.testing.assert_allclose(base[:, 0], -reaction[:, 0], rtol=1e-3)
    assert float(np.max(res.inertia_balance())) < 1e-9
