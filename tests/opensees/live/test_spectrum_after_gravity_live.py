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

import functools
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
           braces: str | None = None, gravity: float = 1.0,
           brace_area: float = 2e-3,
           unbalance_tol: float | None = None) -> apeSees:
    """The test frame; ``braces`` adds the X-brace in ``"ENT"``
    (elastic-no-tension), ``"Elastic"`` or ``"Concrete01"``, and
    ``gravity`` scales the gravity loads."""
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
    if braces is not None:
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
    if braces is not None:
        mat = {
            # Elastic-no-tension: stiff in compression, slack in tension.
            "ENT": lambda: ops.uniaxialMaterial.ENT(E=200e9),
            "Elastic": lambda: ops.uniaxialMaterial.ElasticMaterial(E=200e9),
            # Loads along its backbone, unloads on a stiffer branch.
            "Concrete01": lambda: ops.uniaxialMaterial.Concrete01(
                fpc=-30e6, epsc0=-0.0004, fpcu=-20e6, epsU=-0.004),
        }[braces]()
        ops.element.Truss(pg="Braces", A=brace_area, material=mat)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass(pg="Floor", values=(M_FLOOR, M_FLOOR, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Floor", forces=(0.0, -2.0 * gravity * G_LOAD, 0.0))
        p.load(pg="Mid", forces=(0.0, -2.0 * gravity * G_LOAD, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    if unbalance_tol is None:
        ops.test.NormDispIncr(tol=1e-12, max_iter=25)
    else:
        ops.test.NormUnbalance(tol=unbalance_tol, max_iter=25)
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


@functools.lru_cache(maxsize=1)
def _increment_build() -> bool:
    """Whether the bound build takes ``responseSpectrumAnalysis
    -increment`` (fork WP-207, ladruno ``a8bcc07e``)."""
    from apeGmsh.opensees.procedures import _gravity_modal as gm

    ops = _frame().modal_after_gravity(1, solver="-fullGenLapack")._live.ops
    return gm._rsa_increment_available(ops) is True


@pytest.fixture(params=[
    "static", pytest.param("increment", marks=pytest.mark.ladruno_fork)])
def rsa_route(request: pytest.FixtureRequest,
              monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs a test on each per-mode route: the static legs (the probe
    forced inconclusive) and the fork's ``-increment`` legs, where the probe
    must answer ``True`` (a silent fall-back to the static route would make
    the second run a copy of the first)."""
    from apeGmsh.opensees.procedures import _gravity_modal as gm

    if request.param == "static":
        monkeypatch.setattr(gm, "_rsa_increment_available", lambda ops: None)
        return "static"
    if not _increment_build():
        pytest.skip("this fork build has no responseSpectrumAnalysis "
                    "-increment (fork WP-207, ladruno a8bcc07e)")
    probe = gm._rsa_increment_available

    def must_take_it(ops: Any) -> "bool | None":
        got = probe(ops)
        assert got is True, "the -increment probe fell back on a fork build"
        return got

    monkeypatch.setattr(gm, "_rsa_increment_available", must_take_it)
    return "increment"


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
@pytest.mark.usefixtures("rsa_route")
def test_a_per_mode_and_cqc_meet_the_numpy_reference() -> None:
    res = _run()
    ref = _numpy_reference()
    np.testing.assert_allclose(res.omega, ref["omega"], rtol=1e-10)
    # The probe moves the frame by 1e-8 of its extent about a gravity state
    # ~1e3 times larger, so the central difference carries that much
    # round-off: 3.5e-11 measured.
    for name in ("disp", "base", "cols"):
        got = res.modes(name)
        assert _rel(got, ref[name]) < 1e-9, name
        cqc = res.combine(name, rule="CQC", damping=XI)[0]
        want = combine_modes(ref[name], rule="CQC", omega=ref["omega"],
                             damping=XI)
        assert _rel(cqc, want) < 1e-9, name
    # The kernel itself against a hand loop.
    rho = cqc_correlation(ref["omega"], XI)
    r = ref["cols"][:, 0]
    hand = math.sqrt(sum(rho[i, j] * r[i] * r[j]
                         for i in range(N_MODES) for j in range(N_MODES)))
    assert float(res.combine("cols", damping=XI)[0][0]) == pytest.approx(
        hand, rel=1e-10)


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_c_per_mode_base_shear_is_effective_mass_times_sa(transf: str) -> None:
    """Exact on a linear frame. On a PDelta frame the leg sits at the modal
    displacement of the PDelta tangent, and the element forces there carry
    the axial-force change across the gravity sway that the tangent omits,
    so the support reactions miss ``M_eff Sa`` by that term (7.5e-7 here),
    as ``responseSpectrumAnalysis -mode k`` does."""
    res = _run(transf)
    props = res.modal.modal
    meff = props.properties["partiMassMX"]
    sa = SCALE * np.interp(res.periods, TN, SA)
    base = res.modes("base")                     # (modes, 3 * n_base)
    shear = base[:, 0::3].sum(axis=1)
    want = -np.asarray(meff) * sa
    rtol = 1e-9 if transf == "Linear" else 2e-6
    np.testing.assert_allclose(
        shear, want, rtol=rtol, atol=rtol * float(np.max(np.abs(want))))


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
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
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize("p_delta", [True, False])
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_gravity_drift_is_zero_right_after_the_call(
        transf: str, p_delta: bool) -> None:
    """``gravity_drift()`` promises ``0.0`` while nothing has moved the
    domain. Under ``NormUnbalance 1.0`` the driver settles the gravity state
    with unloaded Linear steps before the per-mode legs, so a snapshot taken
    before settling read 1.5e-7 right after the call (#2545). The snapshot
    is the state the driver leaves the domain at."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    res = _frame(transf, unbalance_tol=1.0).spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors=ROOF_EX, p_delta=p_delta,
        solver="-fullGenLapack")
    assert res.modal.gravity_drift() == (0.0, 0.0)
    # It is still the gravity state: the gravity values agree with it.
    roof = res.modal.gravity.node_disp(ROOF)[:2]
    np.testing.assert_allclose(roof, res.gravity_values["roof"], rtol=1e-9)
    assert res.modal.gravity.total_reaction(2) == pytest.approx(
        G_LOAD * 2 * (len(FLOOR) + N_STORY), rel=1e-6)


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
def test_p_delta_false_responses_are_those_of_k() -> None:
    lin = _run("Linear")
    pd_off = _run("PDelta", p_delta=False)
    pd_on = _run("PDelta")
    # The modes are those of K exactly; the per-mode legs still run on
    # PDelta elements, whose forces past the linear term leave a small
    # residue at the probe amplitude.
    np.testing.assert_allclose(pd_off.periods, lin.periods, rtol=1e-12)
    for name in ("disp", "base", "cols"):
        assert _rel(pd_off.modes(name), lin.modes(name)) < 1e-6, name
    # P-Delta softens the first mode and lengthens its displacement.
    assert float(pd_on.periods[0]) > float(pd_off.periods[0])
    np.testing.assert_allclose(
        pd_off.gravity_values["base"], pd_on.gravity_values["base"],
        rtol=1e-12, atol=1e-6)


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
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
    ``φ_k Γ_k Sa / λ_k`` with ``λ_k`` of the PDelta tangent; the probe legs
    step on that tangent, so the two agree to round-off."""
    res = _run("PDelta")
    cqc = res.combine("disp", damping=XI)[0]
    ops = res.modal._live.ops
    ops.responseSpectrumAnalysis(
        1, "-Tn", *TN, "-Sa", *[SCALE * a for a in SA],
        "-combine", "CQC", "-damp", XI)
    nodes = [_tag(s, 0) for s in range(1, N_STORY + 1)]
    cmd = np.asarray([ops.nodeDisp(n, d) for n in nodes for d in (1, 2, 3)])
    assert _rel(cqc, cmd) < 1e-12


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
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


ROOF_EX = {"roof": NodeDisp(nodes=[ROOF], dofs=(1, 2))}


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
def test_ent_braces_give_the_tangent_response_at_design_amplitude() -> None:
    """The X-brace is compressed by gravity, so the modes see both
    diagonals. At the design amplitude (``scale=9.81``) one diagonal would
    go slack; a probe at that amplitude returns the secant, 8-96 % off per
    mode (the review's repro). The per-mode value must be the modal one."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    res = _frame(braces="ENT").spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors=ROOF_EX, solver="-fullGenLapack")
    want = _modal_roof(res, spec)
    got = res.modes("roof")
    for k in range(N_MODES):
        assert _rel(got[k], want[k]) < 1e-10, k


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize(("transf", "gravity"), [
    ("PDelta", 2.0), ("PDelta", 8.0), ("Corotational", 3.0)])
def test_elastic_p_delta_frames_give_the_modal_displacement(
        transf: str, gravity: float) -> None:
    """The PDelta transformation's tangent omits the axial-force change
    across the gravity sway, so a Newton leg lands off ``a φ / λ`` by a
    first-order gap that grows with the sway (1.2e-3 at twice the gravity,
    5.8e-3 at eight times, the same at every probe size); the review's
    repro refused these elastic frames. The legs step on the tangent the
    modes came from, so the per-mode displacement is the modal one."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    res = _frame(transf, braces="Elastic", gravity=gravity).spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors=ROOF_EX, solver="-fullGenLapack")
    want = _modal_roof(res, spec)
    got = res.modes("roof")
    # Round-off grows as the probe shrinks against the gravity state;
    # 1.3e-8 measured for the Corotational frame's third mode.
    for k in range(N_MODES):
        assert _rel(got[k], want[k]) < 1e-7, k


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize(("braces", "p_delta"), [
    ("ENT", False), ("Concrete01", True)])
def test_a_member_that_changes_stiffness_at_the_reference_is_refused(
        braces: str, p_delta: bool) -> None:
    """Unloaded (``p_delta=False``), the no-tension braces sit at zero
    strain: each probe leg compresses a different one. Concrete01 braces
    compressed by gravity sit on their backbone: one leg loads along it,
    the other unloads on the stiffer branch. Either way the two legs meet
    different stiffnesses, and no amplitude gives a linear response: the
    re-probe at 100 times the amplitude refuses them too."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    with pytest.raises(RuntimeError, match="met different stiffnesses.*"
                                           "at 100 times the probe"):
        _frame(braces=braces).spectrum_after_gravity(
            N_MODES, spectra=[spec], extractors=ROOF_EX, p_delta=p_delta,
            solver="-fullGenLapack")


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
def test_a_slender_kinked_member_is_refused() -> None:
    """No-tension braces a thousand times lighter (``A = 2e-6``) barely
    change the frame's forces, so a global norm let them through and the
    brace's per-mode force came out as the average of its slack and
    compressed legs. Each element is judged on its own forces."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    with pytest.raises(RuntimeError, match="met different stiffnesses in "
                                           "element"):
        _frame(braces="ENT", brace_area=2e-6).spectrum_after_gravity(
            N_MODES, spectra=[spec], extractors=ROOF_EX, p_delta=False,
            solver="-fullGenLapack")


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_a_loose_gravity_tolerance_is_not_a_kink(transf: str) -> None:
    """``NormUnbalance 1.0`` under 2e5 N floor loads leaves the gravity
    state with a residual larger than the probe's forces. Each Linear leg
    cleared it and the kink check counted it twice (refused at 0.1). The
    legs now start from the settled equilibrium."""
    spec = DesignSpectrum(TN, SA, direction=1, scale=9.81)
    res = _frame(transf, unbalance_tol=1.0).spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors=ROOF_EX, solver="-fullGenLapack")
    # The fourth mode is vertical (Γ ~ 0): judged against the largest mode.
    assert _rel(res.modes("roof"), _modal_roof(res, spec)) < 1e-9
    assert max(res.restore_drift) < 1e-9


# -- a torsion mode about a centred mass node ---------------------------------

_CORNERS = ((-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0))


def _diaphragm(transf: str, dx: float, *, link: float = 1.0,
               gravity: float = 1.0, origin: float = 0.0) -> apeSees:
    """Four columns, their tops tied by stiff links to one centre node that
    carries the floor's translational and rotational mass (the review's
    repro). Its pure-torsion mode barely moves the centre node. ``link``
    scales the links' section (``A``, ``J``, ``I``), ``gravity`` the
    loads on the column tops, and ``origin`` shifts every node in x, y
    and z (coarser coordinate ulps, the same span)."""
    o = origin
    coords = ([(x + o, y + o, o) for x, y in _CORNERS]
              + [(x + o, y + o, o + 3.0) for x, y in _CORNERS]
              + [(dx + o, o, o + 3.0)])
    nodes = _NodesStub(ids=list(range(1, 10)), coords=coords,
                       node_pgs={"Base": [1, 2, 3, 4], "Tops": [5, 6, 7, 8],
                                 "Centre": [9]})
    groups = {
        "Cols": _ElementGroupView(ids=(1, 2, 3, 4), connectivity=tuple(
            (i, i + 4) for i in range(1, 5))),
        "Links": _ElementGroupView(ids=(5, 6, 7, 8), connectivity=tuple(
            (9, i) for i in range(5, 9))),
    }
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    tc = getattr(ops.geomTransf, transf)(vecxz=(1.0, 0.0, 0.0))
    tl = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(pg="Cols", transf=tc, A=0.25, E=30e9,
                                  G=12e9, J=0.01, Iy=0.0052, Iz=0.0052)
    ops.element.elasticBeamColumn(pg="Links", transf=tl, A=10.0 * link,
                                  E=30e9, G=12e9, J=10.0 * link,
                                  Iy=10.0 * link, Iz=10.0 * link)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Centre", values=(8e4, 8e4, 0.0, 0.0, 0.0, 8e4 * 6.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tops", forces=(0.0, 0.0, -2e5 * gravity, 0.0, 0.0, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize("dx", [0.0, 1e-9])
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_torsion_mode_at_a_centred_mass_node(transf: str, dx: float) -> None:
    """The torsion mode's Γ is ~1e-12, not 0, and the centre node barely
    translates in it; a probe scaled on that node's translation blew up
    (rc -3). It is skipped as negligible, and the two sway modes are the
    modal ones."""
    spec = [DesignSpectrum((0.0, 2.0), (5.0, 5.0), direction=d)
            for d in (1, 2)]
    res = _diaphragm(transf, dx).spectrum_after_gravity(
        3, spectra=spec, extractors={"c": NodeDisp(nodes=[9], dofs=(1, 2, 6))},
        solver="-fullGenLapack")
    ops = res.modal._live.ops
    props = res.modal.modal.properties
    lam = np.asarray(res.modal.eigenvalues)
    for d, comp in enumerate(("MX", "MY")):
        gamma = np.asarray(props[f"partiFactor{comp}"])
        got = res.modes("c", d)
        for k in range(3):
            want = gamma[k] * 5.0 / lam[k] * np.asarray(
                ops.nodeEigenvector(9, k + 1))[[0, 1, 5]]
            assert np.max(np.abs(got[k] - want)) <= 1e-9 * max(
                float(np.max(np.abs(got))), 1e-300), (comp, k)
    assert max(res.restore_drift) < 1e-9


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_round_off_in_a_near_rigid_link_is_not_a_kink(transf: str) -> None:
    """Links a million times stiffer, under thirty times the gravity: a
    link carries no gravity force, but its resisting force is a huge
    stiffness times the round-off in its nodal displacements, a second
    difference of ~0.04 against a first of 27 at the probe. The linear
    frame was refused at 1.48e-3, blaming a link (#2545). The re-probe at
    100 times the amplitude reads it at ~1e-5, and the per-mode values
    are the modal ones, to the eigenvector's own accuracy on a stiffness
    this ill-conditioned (2.3e-8 Linear, 1.3e-7 PDelta, measured; the same
    at either probe size, so it is not the legs' round-off)."""
    spec = [DesignSpectrum((0.0, 2.0), (5.0, 5.0), direction=d)
            for d in (1, 2)]
    res = _diaphragm(transf, 1e-3, link=1e6, gravity=30.0).spectrum_after_gravity(
        3, spectra=spec, extractors={"c": NodeDisp(nodes=[9], dofs=(1, 2, 6))},
        solver="-fullGenLapack")
    ops = res.modal._live.ops
    props = res.modal.modal.properties
    lam = np.asarray(res.modal.eigenvalues)
    for d, comp in enumerate(("MX", "MY")):
        gamma = np.asarray(props[f"partiFactor{comp}"])
        got = res.modes("c", d)
        for k in range(3):
            want = gamma[k] * 5.0 / lam[k] * np.asarray(
                ops.nodeEigenvector(9, k + 1))[[0, 1, 5]]
            assert np.max(np.abs(got[k] - want)) <= 1e-6 * float(
                np.max(np.abs(got))), (comp, k)
    assert res.modal.gravity_drift() == (0.0, 0.0)


# -- the restore check about an unloaded reference (#2568) --------------------

_DIAPHRAGM_SPEC = [DesignSpectrum((0.0, 2.0), (5.0, 5.0), direction=d)
                   for d in (1, 2)]
_CENTRE_EX = {"c": NodeDisp(nodes=[9], dofs=(1, 2, 6))}


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize(("dx", "link", "gravity", "origin"), [
    (0.0, 1.0, 1.0, 0.0), (1e-9, 1.0, 1.0, 0.0), (1e-3, 1e6, 30.0, 0.0),
    (0.0, 1.0, 1.0, 1e3), (0.0, 1.0, 1.0, 1e5)])
@pytest.mark.parametrize("transf", ["Linear", "PDelta", "Corotational"])
def test_an_unloaded_reference_at_round_off_is_not_a_drift(
        transf: str, dx: float, link: float, gravity: float,
        origin: float) -> None:
    """Under ``p_delta=False`` the per-mode legs run about the unloaded
    state. A Corotational column at rest reads ``6.7e-16`` (one ulp of its
    coordinates, not zero) and reactions of ``3.5e-20``, and the restore
    check divided by them: round-off differences of ``6e-19`` and ``1e-10``
    were refused as a drift of ``9e-4`` and ``3e9``. A reference within
    round-off of the model's extent is undeformed and reports the absolute
    drift, as an exactly zero one always did. Moved 1e3 or 1e5 from the
    origin the model reads the same round-off at rest (the elements work
    on coordinate differences), so the extent is the span, not the
    coordinates' magnitude."""
    res = _diaphragm(transf, dx, link=link, gravity=gravity, origin=origin
                     ).spectrum_after_gravity(
        3, spectra=_DIAPHRAGM_SPEC, extractors=_CENTRE_EX, p_delta=False,
        solver="-fullGenLapack")
    # Absolute: round-off of the displacement (m) and the reaction (N).
    assert res.restore_drift[0] < 1e-15
    assert res.restore_drift[1] < 1e-7
    assert res.modal.gravity_drift() == (0.0, 0.0)


def _left_loaded_drift(monkeypatch: pytest.MonkeyPatch, transf: str,
                       p_delta: bool, load: float) -> tuple[float, float]:
    """Solve a lateral ``load`` on the centre node into the domain just
    before the per-mode loop's restore snapshot, under ``LoadControl 0``
    (no gravity step: the time stays put), and return the refused
    ``(displacement, reaction)`` drift read from the message."""
    import re

    from apeGmsh.opensees.analysis import gravity_modal as gmod

    real = gmod.snapshot_state
    # The restore snapshot is the per-mode loop's second; under p_delta
    # the gravity snapshot and the eigen check's come first.
    restore = 4 if p_delta else 2
    calls = {"n": 0}

    def leave_loaded(ops: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == restore:
            ops.timeSeries("Constant", 9001)
            ops.pattern("Plain", 9001, 9001)
            ops.load(9, load, 0.0, 0.0, 0.0, 0.0, 0.0)
            ops.integrator("LoadControl", 0.0)
            assert ops.analyze(1) == 0
        return real(ops)

    with monkeypatch.context() as m, pytest.raises(
            RuntimeError, match="did not return to its reference state"
    ) as info:
        m.setattr(gmod, "snapshot_state", leave_loaded)
        _diaphragm(transf, 0.0).spectrum_after_gravity(
            3, spectra=_DIAPHRAGM_SPEC, extractors=_CENTRE_EX,
            p_delta=p_delta, solver="-fullGenLapack")
    assert calls["n"] == restore
    got = re.search(r"displacement ([0-9.e+-]+), reaction ([0-9.e+-]+)",
                    str(info.value))
    assert got is not None
    return float(got.group(1)), float(got.group(2))


@pytest.mark.live
@pytest.mark.usefixtures("rsa_route")
@pytest.mark.parametrize("p_delta", [False, True])
@pytest.mark.parametrize("transf", ["Linear", "Corotational"])
def test_a_domain_left_off_its_reference_is_refused(
        monkeypatch: pytest.MonkeyPatch, transf: str, p_delta: bool) -> None:
    """A real drift still refuses, about the unloaded reference and about
    the gravity one: 1 kN on the centre node leaves the domain at another
    equilibrium, as when the legs do not return. About the unloaded
    reference that is 250 N of base shear per column against round-off of
    1e-10 N, reported absolute (the largest base reaction component); about the gravity one,
    a 3.7e-6 m sway against its 8e-5 m. The refused drift is the load's: it
    scales with it, linearly, from 10 N to 1 kN."""
    big = _left_loaded_drift(monkeypatch, transf, p_delta, 1e3)
    small = _left_loaded_drift(monkeypatch, transf, p_delta, 10.0)
    if not p_delta:
        assert 250.0 <= big[1] <= 1e3, big  # absolute, the base reaction
    assert big[0] / small[0] == pytest.approx(100.0, rel=1e-2)
    assert big[1] / small[1] == pytest.approx(100.0, rel=1e-2)


# -- the fork's -increment route (fork WP-207, ADR 0124 amendment) ------------

def _route(monkeypatch: pytest.MonkeyPatch, route: str, run: Any) -> Any:
    from apeGmsh.opensees.procedures import _gravity_modal as gm

    with monkeypatch.context() as m:
        if route == "static":
            m.setattr(gm, "_rsa_increment_available", lambda ops: None)
        return run()


def _per_mode_rel(fast: Any, slow: Any, name: str) -> np.ndarray:
    """Per mode, the largest difference between the routes relative to the
    largest per-mode value of that quantity."""
    a, b = fast.per_mode[name], slow.per_mode[name]
    scale = float(np.max(np.abs(b)))
    return cast("np.ndarray", np.max(np.abs(a - b), axis=(0, 2)) / scale)


_FRAMES: dict[str, Any] = {
    "linear": lambda: _frame("Linear"),
    "pdelta": lambda: _frame("PDelta"),
    "corotational": lambda: _frame("Corotational"),
    "pdelta-braced-x2": lambda: _frame("PDelta", braces="Elastic",
                                       gravity=2.0),
}


def _need_increment() -> None:
    if not _increment_build():
        pytest.skip("this fork build has no responseSpectrumAnalysis "
                    "-increment (fork WP-207, ladruno a8bcc07e)")


@pytest.mark.live
def test_increment_probe_is_none_or_true() -> None:
    """The probe never answers False: a build without the flag is
    inconclusive (``None``) and takes the static route."""
    from apeGmsh.opensees.procedures import _gravity_modal as gm

    ops = _frame().modal_after_gravity(1, solver="-fullGenLapack")._live.ops
    assert gm._rsa_increment_available(ops) in (None, True)


@pytest.mark.live
@pytest.mark.ladruno_fork
@pytest.mark.parametrize("frame", sorted(_FRAMES))
def test_increment_route_equals_the_static_route(
        monkeypatch: pytest.MonkeyPatch, frame: str) -> None:
    """Both routes move the domain by the same ``± a φ / λ`` about the same
    settled reference, one by a Linear step on the eigen tangent and one by
    imposing it, so every per-mode value agrees to round-off."""
    _need_increment()
    spec = DesignSpectrum(TN, SA, direction=1, scale=SCALE)
    # The column tags are read off the unbraced frame; the braced one
    # numbers its elements differently, so it is compared on the nodes.
    names = ("disp", "base", "base_shear") + (
        ("cols",) if frame in ("linear", "pdelta", "corotational") else ())
    ex = {n: EXTRACTORS[n] for n in names if n != "base_shear"}

    def run() -> Any:
        return _FRAMES[frame]().spectrum_after_gravity(
            N_MODES, spectra=[spec], extractors=ex,
            solver="-fullGenLapack")

    fast = _route(monkeypatch, "increment", run)
    slow = _route(monkeypatch, "static", run)
    for name in names:
        rel = _per_mode_rel(fast, slow, name)
        assert float(np.max(rel)) < 1e-9, (name, rel.tolist())
    assert max(fast.restore_drift) < 1e-12


@pytest.mark.live
@pytest.mark.ladruno_fork
@pytest.mark.parametrize("transf", ["Linear", "PDelta"])
def test_increment_route_equals_the_static_route_torsion(
        monkeypatch: pytest.MonkeyPatch, transf: str) -> None:
    """The diaphragm frame: two sway modes in two directions, the
    negligible torsion mode skipped on both routes."""
    _need_increment()
    spec = [DesignSpectrum((0.0, 2.0), (5.0, 5.0), direction=d)
            for d in (1, 2)]
    ex = {"c": NodeDisp(nodes=[9], dofs=(1, 2, 6))}

    def run() -> Any:
        return _diaphragm(transf, 0.0).spectrum_after_gravity(
            3, spectra=spec, extractors=ex, solver="-fullGenLapack")

    fast = _route(monkeypatch, "increment", run)
    slow = _route(monkeypatch, "static", run)
    for name in ("c", "base_shear"):
        rel = _per_mode_rel(fast, slow, name)
        assert float(np.max(rel)) < 1e-9, (name, rel.tolist())


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_an_opaque_extractor_keeps_the_static_route(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A user callable may read ``nodeDisp``, the committed displacement,
    which the ``-increment`` legs leave at the reference: it would return
    zero for every mode. Such an extractor keeps the static route, and its
    per-mode value is the modal one."""
    from apeGmsh.opensees.procedures import _gravity_modal as gm

    _need_increment()
    imposed: list[int] = []
    real = gm._impose_mode

    def counted(*a: Any) -> None:
        imposed.append(1)
        real(*a)

    monkeypatch.setattr(gm, "_impose_mode", counted)

    def roof(ops: Any) -> np.ndarray:
        return np.asarray([ops.nodeDisp(ROOF, 1), ops.nodeDisp(ROOF, 2)])

    spec = DesignSpectrum(TN, SA, direction=1, scale=SCALE)
    res = _frame().spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors={"roof": roof},
        solver="-fullGenLapack")
    assert not imposed
    assert _rel(res.modes("roof"), _modal_roof(res, spec)) < 1e-9
    _frame().spectrum_after_gravity(
        N_MODES, spectra=[spec], extractors=ROOF_EX, solver="-fullGenLapack")
    assert imposed, "a built-in extractor should take the -increment route"


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_an_imposed_state_off_the_mode_is_refused(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The imposed displacement is checked against ``± a φ / λ``: a build
    whose participation factor or spectrum lookup differed (here, ``Sa``
    off by 1e-4) raises instead of scaling every mode. The failed leg is
    still reverted: the domain is back at its committed state, not left at
    the imposed trial one (#2545)."""
    from apeGmsh.opensees.emitter.live import get_ops
    from apeGmsh.opensees.procedures import _gravity_modal as gm

    _need_increment()
    real = gm._impose_mode

    def off(ops: Any, direction: int, period: float, sa: float,
            *rest: Any) -> None:
        real(ops, direction, period, sa * (1.0 + 1e-4), *rest)

    monkeypatch.setattr(gm, "_impose_mode", off)
    with pytest.raises(RuntimeError, match="imposed a displacement"):
        _run()
    ops = get_ops()
    # IncrDisp (response 4) is trial - committed: zero once reverted.
    for node in FLOOR:
        assert [float(ops.nodeResponse(node, d, 4)) for d in (1, 2)] == [
            0.0, 0.0], node
