"""Live oracles for :meth:`apeSees.modal_after_gravity` (ADR 0123).

1. **P-Delta cantilever with a tip mass.** ``n`` ``elasticBeamColumn``
   elements on ``geomTransf PDelta`` under a tip axial load ``P``. The
   PDelta transformation adds the chord geometric stiffness
   ``-(N/L)[[1, -1], [-1, 1]]`` on the transverse DOFs of each element and
   nothing else, so the exact discrete reference is the numpy assembly of
   the Euler-Bernoulli stiffness plus that term, condensed to the tip
   (the only mass). The driver's ``T/T0`` must equal it to solver
   precision; the continuum tip-load cantilever
   ``k(P) = P / (L (tan(αL)/(αL) - 1))`` and the ``1/sqrt(1 - P/Pcr)``
   trend bound it from outside.
2. **ShellNLDKGQ wall strip.** A cantilever strip in out-of-plane bending
   with a tip mass and a tip axial load at ``P = 0.5 Pcr``, ``Pcr`` the
   Euler load of the strip's own unloaded stiffness. ``T/T0`` meets the
   continuum value (1.4094) and ``sqrt(2)``.
3. ``p_delta=False`` leaves the periods at ``K``; the gravity state is the
   same either way and the eigen solve does not move it.

Everything here is stock OpenSees (PDelta, elasticBeamColumn, ShellNLDKGQ,
eigen, modalProperties), so the file runs on both engines.
"""
from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

E = 200e9
I = 1.0e-4  # noqa: E741 — the second moment of area
A = 1.0e-2
L = 3.0
N_EL = 10
M_TIP = 1000.0
P_CR = math.pi ** 2 * E * I / (4.0 * L ** 2)


def _cantilever(P: float, *, transf: str = "PDelta") -> apeSees:
    nodes = _NodesStub(
        ids=list(range(1, N_EL + 2)),
        coords=[(0.0, i * L / N_EL, 0.0) for i in range(N_EL + 1)],
        node_pgs={"Base": [1], "Top": [N_EL + 1]},
    )
    elements = _ElementsStub(elem_pgs={"Col": _ElementGroupView(
        ids=tuple(range(1, N_EL + 1)),
        connectivity=tuple((i, i + 1) for i in range(1, N_EL + 1)),
    )})
    ops = apeSees(cast("object", FEMStub(nodes=nodes, elements=elements)))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    tr = getattr(ops.geomTransf, transf)()
    ops.element.elasticBeamColumn(pg="Col", transf=tr, A=A, E=E, Iz=I)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass(pg="Top", values=(M_TIP, 1e-9, 1e-9))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Top", forces=(0.0, -P, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


def _discrete_tip_stiffness(P: float) -> float:
    """Tip lateral stiffness of the PDelta-transf cantilever, assembled in
    numpy: cubic Euler-Bernoulli bending plus the chord geometric term."""
    le = L / N_EL
    ndof = 2 * (N_EL + 1)          # (w, theta) per node, axial dropped
    K = np.zeros((ndof, ndof))
    kb = E * I / le ** 3 * np.array([
        [12, 6 * le, -12, 6 * le],
        [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
        [-12, -6 * le, 12, -6 * le],
        [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2],
    ])
    kg = -P / le * np.array([
        [1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0],
    ])
    for e in range(N_EL):
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += kb + kg
    Kf = K[2:, 2:]
    f = np.zeros(ndof - 2)
    f[-2] = 1.0
    return 1.0 / float(np.linalg.solve(Kf, f)[-2])


def _continuum_ratio(P: float, EI: float, H: float) -> float:
    """``T/T0`` of a massless cantilever with a tip mass and tip axial
    load ``P``, from the exact lateral tip stiffness."""
    al = math.sqrt(P / EI)
    k = P / (H * (math.tan(al * H) / (al * H) - 1.0))
    return math.sqrt((3.0 * EI / H ** 3) / k)


@pytest.mark.live
@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_pdelta_cantilever_period_softening_meets_discrete_reference(
    ratio: float,
) -> None:
    t0 = _cantilever(0.0).modal_after_gravity(1, solver="-fullGenLapack")
    res = _cantilever(ratio * P_CR).modal_after_gravity(
        1, solver="-fullGenLapack")
    t_ratio = float(res.periods[0] / t0.periods[0])
    exact = math.sqrt(
        _discrete_tip_stiffness(0.0) / _discrete_tip_stiffness(ratio * P_CR))
    assert t_ratio == pytest.approx(exact, rel=1e-9)
    # The continuum value and the 1/sqrt(1 - P/Pcr) trend bound it.
    assert t_ratio == pytest.approx(
        _continuum_ratio(ratio * P_CR, E * I, L), rel=5e-3)
    assert t_ratio == pytest.approx(1.0 / math.sqrt(1.0 - ratio), rel=1.5e-2)
    assert t_ratio < 1.0 / math.sqrt(1.0 - ratio)


@pytest.mark.live
def test_gravity_state_is_exact_and_untouched_by_eigen() -> None:
    P = 0.5 * P_CR
    res = _cantilever(P).modal_after_gravity(1, solver="-fullGenLapack")
    assert res.p_delta is True
    assert res.gravity_steps == 10
    assert res.gravity.time == 0.0
    # Axial shortening of a straight column, and the base reaction.
    assert float(res.gravity.node_disp(N_EL + 1)[1]) == pytest.approx(
        -P * L / (E * A), rel=1e-10)
    assert res.gravity.total_reaction(2) == pytest.approx(P, rel=1e-10)
    assert res.gravity_drift() == (0.0, 0.0)
    # The participation of the tip-mass mode, from modalProperties.
    assert float(res.modal.mass_ratios("MX")[0]) > 99.9


@pytest.mark.live
def test_p_delta_false_solves_on_k_and_keeps_the_same_gravity_state() -> None:
    P = 0.5 * P_CR
    t0 = _cantilever(0.0).modal_after_gravity(1, solver="-fullGenLapack")
    with_pd = _cantilever(P).modal_after_gravity(1, solver="-fullGenLapack")
    without = _cantilever(P).modal_after_gravity(
        1, solver="-fullGenLapack", p_delta=False)
    assert without.p_delta is False
    np.testing.assert_allclose(without.periods, t0.periods, rtol=1e-12)
    assert float(with_pd.periods[0]) > 1.4 * float(without.periods[0])
    for tag in with_pd.gravity.node_tags:
        np.testing.assert_allclose(
            without.gravity.disp[int(tag)], with_pd.gravity.disp[int(tag)],
            rtol=0.0, atol=1e-15)
    assert without.gravity_drift() == (0.0, 0.0)


@pytest.mark.live
def test_linear_transf_has_no_geometric_stiffness() -> None:
    t0 = _cantilever(0.0, transf="Linear").modal_after_gravity(
        1, solver="-fullGenLapack")
    res = _cantilever(0.5 * P_CR, transf="Linear").modal_after_gravity(
        1, solver="-fullGenLapack")
    np.testing.assert_allclose(res.periods, t0.periods, rtol=1e-10)


# -- ShellNLDKGQ wall strip ---------------------------------------------------

E_W, NU_W, T_W, B_W, H_W = 30e9, 0.2, 0.2, 1.0, 3.0
NX, NZ, M_W = 2, 12, 5000.0


def _strip(P: float) -> apeSees:
    def tag(i: int, k: int) -> int:
        return k * (NX + 1) + i + 1

    ids, coords = [], []
    for k in range(NZ + 1):
        for i in range(NX + 1):
            ids.append(tag(i, k))
            coords.append((i * B_W / NX, 0.0, k * H_W / NZ))
    top = [tag(i, NZ) for i in range(NX + 1)]
    conn = tuple(
        (tag(i, k), tag(i + 1, k), tag(i + 1, k + 1), tag(i, k + 1))
        for k in range(NZ) for i in range(NX)
    )
    nodes = _NodesStub(ids=ids, coords=coords, node_pgs={
        "Base": [tag(i, 0) for i in range(NX + 1)],
        "TopEdge": [top[0], top[-1]], "TopMid": top[1:-1],
    })
    elements = _ElementsStub(elem_pgs={"Wall": _ElementGroupView(
        ids=tuple(range(1, len(conn) + 1)), connectivity=conn)})
    ops = apeSees(cast("object", FEMStub(nodes=nodes, elements=elements)))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=E_W, nu=NU_W, h=T_W)
    ops.element.ShellNLDKGQ(pg="Wall", section=sec)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # Tip mass (out of plane, y) and tip load, tributary along the top edge.
    share = {"TopEdge": 0.5 / NX, "TopMid": 1.0 / NX}
    for pg, w in share.items():
        ops.mass(pg=pg, values=(1e-9, M_W * w, 1e-9, 0.0, 0.0, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for pg, w in share.items():
            p.load(pg=pg, forces=(0.0, 0.0, -P * w, 0.0, 0.0, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=30)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


@pytest.mark.live
def test_shellnldkgq_wall_strip_at_half_pcr_softens_by_sqrt2() -> None:
    r0 = _strip(0.0).modal_after_gravity(1, solver="-fullGenLapack")
    # Pcr from the strip's own unloaded tip stiffness (k0 = w0^2 m).
    k0 = float(r0.eigenvalues[0]) * M_W
    ei = k0 * H_W ** 3 / 3.0
    # Between the beam (Ebt^3/12) and the plate (D b) stiffness.
    assert E_W * B_W * T_W ** 3 / 12 < ei < E_W * T_W ** 3 / 12 / (1 - NU_W ** 2) * B_W
    p_cr = math.pi ** 2 * ei / (4.0 * H_W ** 2)
    res = _strip(0.5 * p_cr).modal_after_gravity(1, solver="-fullGenLapack")
    t_ratio = float(res.periods[0] / r0.periods[0])
    assert t_ratio == pytest.approx(_continuum_ratio(0.5 * p_cr, ei, H_W), rel=2e-3)
    assert t_ratio == pytest.approx(math.sqrt(2.0), rel=5e-3)
    assert res.gravity_drift() == (0.0, 0.0)
