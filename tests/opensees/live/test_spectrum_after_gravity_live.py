"""Live oracles for :meth:`apeSees.spectrum_after_gravity` (ADR 0124 D4–D7).

A 2-D frame, two bays by three stories, lumped floor masses, gravity loads.

(a) **numpy reference.** With ``geomTransf Linear`` the frame is linear, so
    its stiffness ``K`` is assembled in numpy from the textbook frame
    element, the massless DOFs are condensed out, the modes come from
    ``eigh``, and every per-mode response follows by hand:
    ``u_k = φ_k Γ_k Sa(T_k) / λ_k``, reactions ``(K u_k)`` at the supports,
    element end forces ``k_e T u_k``. The driver's per-mode values and their
    CQC match it to ``1e-10``.
(b) **The command's own CQC** (fork-only ``-combine``) of the nodal
    displacements matches the Python CQC of the per-mode displacements.
(c) Each mode's base shear is ``-M_eff,k Sa(T_k)`` (``modalProperties``).
(d) The domain is back at the gravity state afterwards.

The P-Delta variant runs the same checks where they still hold exactly
((c), (d)), and bounds (b) by the PDelta transformation's tangent gap.
"""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.analysis.spectrum import (
    DesignSpectrum,
    ElementResponse,
    NodeDisp,
    NodeReaction,
    combine_modes,
    cqc_correlation,
)

openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

E = 30e9
BAY, STORY = 6.0, 3.0
N_BAY, N_STORY = 2, 3
A_COL, I_COL = 0.25, 0.0052
A_BM, I_BM = 0.18, 0.0054
M_FLOOR = 20e3
G_LOAD = 20e3 * 9.81
TN = (0.0, 0.5, 1.0, 2.0)
SA = (3.0, 6.0, 4.0, 2.0)
SCALE = 0.75
XI = 0.05
N_MODES = 4


def _tag(s: int, c: int) -> int:
    return s * (N_BAY + 1) + c + 1


def _layout() -> tuple[list[tuple[float, float]], list[tuple[int, int, str]]]:
    coords = [(c * BAY, s * STORY) for s in range(N_STORY + 1)
              for c in range(N_BAY + 1)]
    elems: list[tuple[int, int, str]] = []
    for s in range(N_STORY):
        for c in range(N_BAY + 1):
            elems.append((_tag(s, c), _tag(s + 1, c), "col"))
        for c in range(N_BAY):
            elems.append((_tag(s + 1, c), _tag(s + 1, c + 1), "beam"))
    return coords, elems


COORDS, ELEMS = _layout()
BASE = [_tag(0, c) for c in range(N_BAY + 1)]
FLOOR = [_tag(s, c) for s in range(1, N_STORY + 1) for c in range(N_BAY + 1)]
COL_IDS = tuple(i + 1 for i, e in enumerate(ELEMS) if e[2] == "col")
BEAM_IDS = tuple(i + 1 for i, e in enumerate(ELEMS) if e[2] == "beam")
#: An X-brace in the first bay of the first story.
BRACES = ((_tag(0, 0), _tag(1, 1)), (_tag(0, 1), _tag(1, 0)))
BRACE_IDS = (len(ELEMS) + 1, len(ELEMS) + 2)
ROOF = _tag(N_STORY, 0)


def _frame(transf: str = "Linear", *, beam_mass: float | None = None,
           ent_braces: bool = False) -> apeSees:
    nodes = _NodesStub(
        ids=list(range(1, len(COORDS) + 1)),
        coords=[(x, y, 0.0) for x, y in COORDS],
        node_pgs={"Base": BASE, "Floor": FLOOR,
                  "Mid": [_tag(s, 1) for s in range(1, N_STORY + 1)]},
    )
    groups = {
        "Cols": _ElementGroupView(ids=COL_IDS, connectivity=tuple(
            ELEMS[i - 1][:2] for i in COL_IDS)),
        "Beams": _ElementGroupView(ids=BEAM_IDS, connectivity=tuple(
            ELEMS[i - 1][:2] for i in BEAM_IDS)),
    }
    if ent_braces:
        groups["Braces"] = _ElementGroupView(ids=BRACE_IDS,
                                             connectivity=BRACES)
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    tc = getattr(ops.geomTransf, transf)()
    tb = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Cols", transf=tc, A=A_COL, E=E, Iz=I_COL)
    ops.element.elasticBeamColumn(pg="Beams", transf=tb, A=A_BM, E=E,
                                  Iz=I_BM, mass=beam_mass)
    if ent_braces:
        # Elastic-no-tension: stiff in compression, slack in tension.
        ent = ops.uniaxialMaterial.ENT(E=200e9)
        ops.element.Truss(pg="Braces", A=2e-3, material=ent)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass(pg="Floor", values=(M_FLOOR, M_FLOOR, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Floor", forces=(0.0, -2.0 * G_LOAD, 0.0))
        p.load(pg="Mid", forces=(0.0, -2.0 * G_LOAD, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


def _live_column_tags() -> tuple[int, ...]:
    """The bridge allocates its own element tags; read the columns' tags
    off a built domain, in ``ELEMS`` order, by their end nodes."""
    ops = _frame().modal_after_gravity(1, solver="-fullGenLapack")._live.ops
    by_nodes = {tuple(int(n) for n in ops.eleNodes(e)): int(e)
                for e in ops.getEleTags()}
    return tuple(by_nodes[ELEMS[i - 1][:2]] for i in COL_IDS)


EXTRACTORS: dict[str, Any] = {
    "disp": NodeDisp(nodes=[_tag(s, 0) for s in range(1, N_STORY + 1)],
                     dofs=(1, 2, 3)),
    "base": NodeReaction(nodes=BASE, dofs=(1, 2, 3)),
}


@pytest.fixture(scope="module", autouse=True)
def _column_extractor() -> Any:
    EXTRACTORS["cols"] = ElementResponse(
        elements=_live_column_tags(), response="localForce")
    yield
    EXTRACTORS.pop("cols", None)


def _run(transf: str = "Linear", **kw: Any) -> Any:
    spec = DesignSpectrum(TN, SA, direction=1, scale=SCALE)
    return _frame(transf).spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors=EXTRACTORS,
        solver="-fullGenLapack", **kw)


# -- the numpy reference -----------------------------------------------------

def _numpy_reference() -> dict[str, np.ndarray]:
    n = len(COORDS)
    K = np.zeros((3 * n, 3 * n))
    blocks: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
    for i, j, kind in ELEMS:
        (xi, yi), (xj, yj) = COORDS[i - 1], COORDS[j - 1]
        L = math.hypot(xj - xi, yj - yi)
        c, s = (xj - xi) / L, (yj - yi) / L
        a, iz = (A_COL, I_COL) if kind == "col" else (A_BM, I_BM)
        ea, ei = E * a / L, E * iz / L ** 3
        kl = np.array([
            [ea, 0, 0, -ea, 0, 0],
            [0, 12 * ei, 6 * ei * L, 0, -12 * ei, 6 * ei * L],
            [0, 6 * ei * L, 4 * ei * L * L, 0, -6 * ei * L, 2 * ei * L * L],
            [-ea, 0, 0, ea, 0, 0],
            [0, -12 * ei, -6 * ei * L, 0, 12 * ei, -6 * ei * L],
            [0, 6 * ei * L, 2 * ei * L * L, 0, -6 * ei * L, 4 * ei * L * L],
        ])
        r = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T = np.zeros((6, 6))
        T[:3, :3] = T[3:, 3:] = r
        idx = [3 * (i - 1) + d for d in range(3)] + [3 * (j - 1) + d for d in range(3)]
        K[np.ix_(idx, idx)] += T.T @ kl @ T
        blocks.append((kl, T, idx))
    fixed = [3 * (b - 1) + d for b in BASE for d in range(3)]
    free = [d for d in range(3 * n) if d not in fixed]
    mass = np.zeros(3 * n)
    for f in FLOOR:
        mass[3 * (f - 1)] = mass[3 * (f - 1) + 1] = M_FLOOR
    a_dofs = [d for d in free if mass[d] > 0]
    b_dofs = [d for d in free if mass[d] == 0]
    Kaa = K[np.ix_(a_dofs, a_dofs)]
    Kab = K[np.ix_(a_dofs, b_dofs)]
    Kbb = K[np.ix_(b_dofs, b_dofs)]
    Kc = Kaa - Kab @ np.linalg.solve(Kbb, Kab.T)
    m_a = mass[a_dofs]
    lam, psi = np.linalg.eigh(Kc / np.sqrt(np.outer(m_a, m_a)))
    lam, psi = lam[:N_MODES], psi[:, :N_MODES]
    phi_a = psi / np.sqrt(m_a)[:, None]
    phi = np.zeros((3 * n, N_MODES))
    phi[a_dofs] = phi_a
    phi[b_dofs] = -np.linalg.solve(Kbb, Kab.T @ phi_a)
    iota = np.zeros(3 * n)
    iota[0::3] = 1.0
    gamma = (phi.T @ (mass * iota)) / np.einsum("ik,i,ik->k", phi, mass, phi)
    omega = np.sqrt(lam)
    sa = SCALE * np.interp(2 * np.pi / omega, TN, SA)
    u = phi * (gamma * sa / lam)[None, :]          # (3n, modes)
    disp_nodes = [_tag(s, 0) for s in range(1, N_STORY + 1)]
    disp = np.stack([u[[3 * (t - 1) + d for t in disp_nodes for d in range(3)], k]
                     for k in range(N_MODES)])
    reac = np.stack([(K @ u[:, k])[[3 * (b - 1) + d for b in BASE for d in range(3)]]
                     for k in range(N_MODES)])
    cols = np.stack([np.concatenate([
        kl @ T @ u[idx, k]
        for (kl, T, idx), (_, _, kind) in zip(blocks, ELEMS) if kind == "col"
    ]) for k in range(N_MODES)])
    meff = (phi.T @ (mass * iota)) ** 2 / np.einsum("ik,i,ik->k", phi, mass, phi)
    return {"disp": disp, "base": reac, "cols": cols, "omega": omega,
            "meff": meff, "sa": sa}


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.live
def test_a_per_mode_and_cqc_meet_the_numpy_reference() -> None:
    res = _run()
    ref = _numpy_reference()
    np.testing.assert_allclose(res.omega, ref["omega"], rtol=1e-10)
    for name in ("disp", "base", "cols"):
        got = res.modes(name)
        assert _rel(got, ref[name]) < 1e-10, name
        cqc = res.combine(name, rule="CQC", damping=XI)[0]
        want = combine_modes(ref[name], rule="CQC", omega=ref["omega"],
                             damping=XI)
        assert _rel(cqc, want) < 1e-10, name
    # The kernel itself against a hand loop.
    rho = cqc_correlation(ref["omega"], XI)
    r = ref["cols"][:, 0]
    hand = math.sqrt(sum(rho[i, j] * r[i] * r[j]
                         for i in range(N_MODES) for j in range(N_MODES)))
    assert float(res.combine("cols", damping=XI)[0][0]) == pytest.approx(
        hand, rel=1e-10)


@pytest.mark.live
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_c_per_mode_base_shear_is_effective_mass_times_sa(transf: str) -> None:
    res = _run(transf)
    props = res.modal.modal
    meff = props.properties["partiMassMX"]
    sa = SCALE * np.interp(res.periods, TN, SA)
    base = res.modes("base")                     # (modes, 3 * n_base)
    shear = base[:, 0::3].sum(axis=1)
    want = -np.asarray(meff) * sa
    np.testing.assert_allclose(
        shear, want, rtol=1e-9, atol=1e-12 * float(np.max(np.abs(want))))


@pytest.mark.live
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_d_gravity_state_is_restored(transf: str) -> None:
    res = _run(transf)
    assert res.restore_drift[0] < 1e-9 and res.restore_drift[1] < 1e-9
    disp_drift, reac_drift = res.modal.gravity_drift()
    assert disp_drift < 1e-9 and reac_drift < 1e-9
    ops = res.modal._live.ops
    for name, ex in EXTRACTORS.items():
        np.testing.assert_allclose(
            np.asarray(ex(ops)), res.gravity_values[name], rtol=1e-9,
            atol=1e-9 * float(np.max(np.abs(res.gravity_values[name]))))
    # The gravity base reaction carries the whole vertical load.
    total_v = res.gravity_values["base"][1::3].sum()
    assert total_v == pytest.approx(G_LOAD * 2 * (len(FLOOR) + N_STORY), rel=1e-10)


@pytest.mark.live
def test_p_delta_false_responses_are_those_of_k() -> None:
    lin = _run("Linear")
    pd_off = _run("PDelta", p_delta=False)
    pd_on = _run("PDelta")
    # The modes are those of K exactly; the per-mode legs still run on
    # PDelta elements, whose forces past the quadratic term leave a
    # third-order residue at the spectrum amplitude.
    np.testing.assert_allclose(pd_off.periods, lin.periods, rtol=1e-12)
    for name in ("disp", "base", "cols"):
        assert _rel(pd_off.modes(name), lin.modes(name)) < 1e-6, name
    # P-Delta softens the first mode and lengthens its displacement.
    assert float(pd_on.periods[0]) > float(pd_off.periods[0])
    np.testing.assert_allclose(
        pd_off.gravity_values["base"], pd_on.gravity_values["base"],
        rtol=1e-12, atol=1e-6)


@pytest.mark.live
def test_two_directions_combine_100_30_and_srss() -> None:
    ex = {"disp": EXTRACTORS["disp"]}
    sx = DesignSpectrum(TN, SA, direction=1, scale=SCALE)
    sy = DesignSpectrum(TN, SA, direction=2, scale=0.5 * SCALE)
    res = _frame().spectrum_after_gravity(
        N_MODES, spectra=[sx, sy], extractors=ex, solver="-fullGenLapack")
    per_dir = res.combine("disp", damping=XI)
    assert per_dir.shape == (2, 9)
    ex_, ey_ = per_dir
    np.testing.assert_allclose(
        res.directional("disp", damping=XI, method="100-30"),
        np.maximum(ex_ + 0.3 * ey_, 0.3 * ex_ + ey_), rtol=1e-14)
    np.testing.assert_allclose(
        res.directional("disp", damping=XI, method="SRSS"),
        np.hypot(ex_, ey_), rtol=1e-14)
    # One solve pair per mode serves both directions: the X result equals
    # an X-only run.
    only_x = _frame().spectrum_after_gravity(
        N_MODES, spectra=[sx], extractors=ex, solver="-fullGenLapack")
    assert _rel(res.modes("disp", 0), only_x.modes("disp", 0)) < 1e-10


@pytest.mark.live
def test_element_mass_is_refused() -> None:
    spec = DesignSpectrum(TN, SA, direction=1)
    with pytest.raises(BridgeError, match="element mass"):
        _frame(beam_mass=100.0).spectrum_after_gravity(
            N_MODES, spectra=[spec], extractors=EXTRACTORS,
            solver="-fullGenLapack")


@pytest.mark.live
def test_an_extractor_that_changes_length_is_refused() -> None:
    calls = {"n": 0}

    def growing(ops: Any) -> np.ndarray:
        calls["n"] += 1
        return np.zeros(calls["n"])

    spec = DesignSpectrum(TN, SA, direction=1)
    with pytest.raises(ValueError, match="changed length"):
        _frame().spectrum_after_gravity(
            N_MODES, spectra=[spec], extractors={"g": growing},
            solver="-fullGenLapack")


# -- (b) the command's own -combine CQC (fork-only) ---------------------------

@pytest.mark.live
@pytest.mark.ladruno_fork
def test_b_python_cqc_matches_the_commands_combine_linear() -> None:
    res = _run()
    cqc = res.combine("disp", damping=XI)[0]
    ref = _frame().response_spectrum_analysis(
        1, periods=TN, accels=[SCALE * a for a in SA], combine="CQC",
        num_modes=N_MODES, damp=XI, solver="-fullGenLapack")
    nodes = [_tag(s, 0) for s in range(1, N_STORY + 1)]
    cmd = np.asarray([ref.node_disp(n, d) for n in nodes for d in (1, 2, 3)])
    assert _rel(cqc, cmd) < 1e-12


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_b_python_cqc_matches_the_commands_combine_pdelta() -> None:
    """On the preloaded PDelta domain the command combines
    ``φ_k Γ_k Sa / λ_k`` with ``λ_k`` of the PDelta tangent; the Newton
    solve uses the full linearisation of the element forces, which adds the
    axial-force change across the gravity sway. The gap is that term."""
    res = _run("PDelta")
    cqc = res.combine("disp", damping=XI)[0]
    ops = res.modal._live.ops
    ops.responseSpectrumAnalysis(
        1, "-Tn", *TN, "-Sa", *[SCALE * a for a in SA],
        "-combine", "CQC", "-damp", XI)
    nodes = [_tag(s, 0) for s in range(1, N_STORY + 1)]
    cmd = np.asarray([ops.nodeDisp(n, d) for n in nodes for d in (1, 2, 3)])
    assert _rel(cqc, cmd) < 1e-5


@pytest.mark.live
def test_base_shear_is_the_combined_total_reaction() -> None:
    """The driver's ``base_shear`` per mode is the summed support reaction;
    ``V_t`` is its CQC (not the sum of combined reactions)."""
    res = _run()
    np.testing.assert_allclose(
        res.modes("base_shear")[:, 0], res.modes("base")[:, 0::3].sum(axis=1),
        rtol=1e-12, atol=1e-12 * float(np.max(np.abs(res.modes("base")))))
    v_t = float(res.base_shear(damping=XI)[0])
    assert v_t == pytest.approx(
        float(res.combine("base_shear", damping=XI)[0][0]))
    assert v_t < float(res.combine("base", damping=XI)[0][0::3].sum())
    with pytest.raises(ValueError, match="reserved"):
        _frame().spectrum_after_gravity(
            1, spectra=[DesignSpectrum(TN, SA, direction=1)],
            extractors={"base_shear": EXTRACTORS["disp"]},
            solver="-fullGenLapack")


@pytest.mark.live
def test_scale_to_base_shear_lifts_forces_only_below_the_target() -> None:
    res = _run()
    v_t = float(res.base_shear(damping=XI)[0])
    factors, scaled = res.scale_to_base_shear(
        1.3 * v_t, ["cols", "base_shear"], damping=XI)
    assert float(factors[0]) == pytest.approx(1.3)
    np.testing.assert_allclose(
        scaled["cols"], 1.3 * res.combine("cols", damping=XI), rtol=1e-14)
    assert float(scaled["base_shear"][0][0]) == pytest.approx(1.3 * v_t)
    factors, _ = res.scale_to_base_shear(0.5 * v_t, ["cols"], damping=XI)
    assert float(factors[0]) == 1.0


# -- the probe sees the tangent the modes were solved on ----------------------

def _modal_roof(res: Any, spec: DesignSpectrum) -> np.ndarray:
    """``Γ_k Sa(T_k) φ_k / λ_k`` at the roof, from the live domain's own
    modes: the per-mode roof displacement a response spectrum means."""
    ops = res.modal._live.ops
    gamma = np.asarray(res.modal.modal.properties["partiFactorMX"])
    lam = np.asarray(res.modal.eigenvalues)
    sa = spec.sa(res.periods)
    return np.stack([
        gamma[k] * sa[k] / lam[k]
        * np.asarray(ops.nodeEigenvector(ROOF, k + 1))[:2]
        for k in range(N_MODES)])


@pytest.mark.live
def test_ent_braces_give_the_tangent_response_at_design_amplitude() -> None:
    """The X-brace is compressed by gravity, so the modes see both
    diagonals. At the design amplitude (``scale=9.81``) one diagonal would
    go slack; a probe at that amplitude returns the secant, 7-82 % off per
    mode (the review's repro). The per-mode value must be the modal one."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    res = _frame(ent_braces=True).spectrum_after_gravity(
        N_MODES, spectra=[spec],
        extractors={"roof": NodeDisp(nodes=[ROOF], dofs=(1, 2))},
        solver="-fullGenLapack")
    want = _modal_roof(res, spec)
    got = res.modes("roof")
    for k in range(N_MODES):
        assert _rel(got[k], want[k]) < 1e-6, k


@pytest.mark.live
def test_a_member_that_changes_stiffness_at_the_reference_is_refused() -> None:
    """Unloaded (``p_delta=False``), the no-tension braces sit at zero
    strain: the modes see them slack, each probe leg compresses one. No
    amplitude gives the modal response, so the driver refuses."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    with pytest.raises(RuntimeError, match="did not see the tangent"):
        _frame(ent_braces=True).spectrum_after_gravity(
            N_MODES, spectra=[spec],
            extractors={"roof": NodeDisp(nodes=[ROOF], dofs=(1, 2))},
            p_delta=False, solver="-fullGenLapack")
