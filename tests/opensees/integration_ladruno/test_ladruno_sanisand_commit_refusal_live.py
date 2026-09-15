"""Fork-only — F7's commit-time IMPL-EX refusal latch (fork PR #838),
against a REAL post-#838 build (PR #1148 close-out).

Two claims, proved live rather than synthetically:

(a) ``implexRefusals`` is SIX wide on the wire, and
    ``_ladruno_element_io``'s by-position tables (widened 4 -> 6 by
    fork PR #838 / F7-a) resolve it to the six canonical names, from a
    REAL ``.ladruno`` recorder file round-tripped through
    ``Results.from_ladruno`` — not a synthetic block handed to the
    reader directly (that is ``tests/results/readers/test_ladruno_reader.py``'s
    job; this file is the "did the fork actually ship it, and does the
    live emitter -> recorder -> reader chain actually carry it" proof).
(b) A commit-time refusal makes ``analyze()`` return ``-4``, and
    ``check_analyze_rc`` (``_internal.analyze_rc``) turns that into an
    ``AnalysisAbortedError`` raised THROUGH ``Substep.drive`` (ADR 0104)
    BEFORE any subdivision is attempted — not a subdivision signal.

Deck shape: one 1 m³ ``LadrunoBrick`` (MEASURED to propagate a material
refusal — fork PR #838's "Element refusal roster") host to
``LadrunoSANISAND(implex=True, max_substeps=1, int_scheme=1)`` — bare
``-implex`` (no ``-implexControl``/``-implexFactor``) plus a
deliberately starved companion cap, the adoption guide's F7 §1.5
recipe. Isotropic confinement first (elastic, the material's own
construction-time ``alpha = 0`` keeps the yield surface at
``-sqrt(2/3)*m*p`` — with ``m = 0.005`` this sits a hair off the
origin, so ANY deviatoric increment after the stage-1 flip yields
immediately and needs more than one ``ModifiedEuler`` substep — this is
why a cap of 1 trips on the very first deviatoric step and no larger
deck or longer run was needed).

Skips: the root conftest auto-skips ``ladruno_fork`` off the live
backend; ``_require_commit_refusal_support`` additionally probe-skips a
fork build that predates the SIX-wide response (checked by ``len()`` of
the live ``eleResponse`` bucket, not by ``ops.ladrunoBuild()`` — a build
stamp says nothing about which responses that build actually shipped).

Measured on fork build ``634824e1fbcf802bf27c7fbd29c113e5a2d63cb6``
(2026-09-15): the very first post-flip deviatoric increment aborts with
``implexRefusals = [n, 0, 0, n, 1, n]`` (all-companion, this Gauss
point's own commit-latch flag set, ``latched`` growing on every
post-latch Newton iteration) — ``n`` varies a little with the exact
push (4 or 5 across runs at the two control setups below), which is
expected: it is a count of ``ModifiedEuler`` iterations before the
cap, not a fixed constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.analyze_rc import AnalysisAbortedError
from apeGmsh.opensees.analysis.strategy import OpenSeesPyDriver, Substep
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

#: Gorini's calibrated set (ADR 86 §6), copied verbatim from
#: ``test_ladruno_sanisand_live.py``'s ``_GORINI``.
_GORINI = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "z_max": 12.5, "cz": 1100.0, "rho": 2.0,
}

_WANTED_NAMES = (
    "implex_refusals_total", "implex_refusals_sign_change",
    "implex_refusals_control", "implex_refusals_companion",
    "implex_refusals_commit_latched", "implex_refusals_latched",
)

_HAS_SIX_WIDE: bool | None = None


def _single_hex_fem(model_name: str):
    """One 1 m³ 8-node hex, PG 'soil' — the material-driver mesh."""
    with apeGmsh(model_name=model_name, verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="soil")
        return g.mesh.queries.get_fem_data(dim=3)


def _node_coords(fem) -> dict[int, tuple[float, float, float]]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return {
        int(n): (float(p[0]), float(p[1]), float(p[2]))
        for n, p in zip(ids, xyz)
    }


def _plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _require_commit_refusal_support() -> None:
    """Probe-skip a fork build whose ``implexRefusals`` predates F7's
    SIX-wide widening — checked by ``len()`` of the live bucket, not by
    ``ops.ladrunoBuild()``: a build stamp is a configure-time fact and
    says nothing about which responses that particular build shipped.
    """
    global _HAS_SIX_WIDE
    if _HAS_SIX_WIDE is None:
        fem = _single_hex_fem("f7e_probe")
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        sand = ops.nDMaterial.LadrunoSANISAND(
            **_GORINI, int_scheme=1, implex=True, max_substeps=1,
        )
        ops.element.LadrunoBrick(pg="soil", material=sand)
        em = LiveOpsEmitter(wipe=True)
        ops.build().emit(em)
        etag = int(em.ops.getEleTags()[0])
        try:
            resp = em.ops.eleResponse(etag, "material", 1, "implexRefusals")
        except Exception:
            resp = []
        _HAS_SIX_WIDE = len(resp) == 6
        em.ops.wipe()
    if not _HAS_SIX_WIDE:
        pytest.skip(
            "fork build's implexRefusals is not six-wide -- predates "
            "fork PR #838 / F7 (widened 4 -> 6)"
        )


def _build_confined_hex(model_name: str, *, max_substeps: int):
    """Build + confine a LadrunoBrick/LadrunoSANISAND hex; flip to
    stage 1 and freeze the confinement load via ``loadConst``.

    Returns ``(fem, ops, emitter, etag, mtag, top_nodes)``.
    """
    fem = _single_hex_fem(model_name)
    coords = _node_coords(fem)
    bottom = _plane(fem, 2, 0.0)
    top = _plane(fem, 2, 1.0)
    origin = next(n for n, c in coords.items() if c == (0.0, 0.0, 0.0))
    on_x = next(n for n, c in coords.items() if c == (1.0, 0.0, 0.0))

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    sand = ops.nDMaterial.LadrunoSANISAND(
        **_GORINI, int_scheme=1, implex=True, max_substeps=max_substeps,
    )
    ops.element.LadrunoBrick(pg="soil", material=sand)
    ops.fix(nodes=[origin], dofs=(1, 1, 1))
    ops.fix(nodes=[on_x], dofs=(0, 1, 1))
    ops.fix(
        nodes=[n for n in bottom if n not in (origin, on_x)], dofs=(0, 0, 1),
    )

    p_c = 100.0
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid, (x, y, z) in coords.items():
            fx = (p_c / 4.0) if x == 0.0 else (-p_c / 4.0)
            fy = (p_c / 4.0) if y == 0.0 else (-p_c / 4.0)
            fz = (-p_c / 4.0) if z == 1.0 else 0.0
            p.load(node=nid, forces=(fx, fy, fz))

    return ops, sand, coords, bottom, top, p_c


def _confine(ops, sand, coords, bottom, top, p_c, *, recorder_path: str | None):
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormUnbalance(tol=1.0e-4 * p_c, max_iter=200)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    if recorder_path is not None:
        ops.recorder.Ladruno(
            file=recorder_path,
            elem_responses=("stress", "material.implexRefusals"),
        )

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    etag = int(emitter.ops.getEleTags()[0])
    mtag = ops.tag_for(sand)

    for i in range(10):
        rc = emitter.analyze(steps=1)
        assert rc == 0, f"isotropic confinement step {i} failed, rc={rc}"

    # ADR 0057 §2's loadConst contract: freeze the confinement load and
    # reset pseudo time before starting the deviatoric leg, so the
    # (unbounded Linear) confinement pattern does not keep re-inflating.
    emitter.ops.loadConst("-time", 0.0)
    emitter.update_material_stage(mtag, 1)
    return emitter, etag


def test_implex_refusals_six_wide_on_real_recorder_and_reader(tmp_path) -> None:
    """(a) A real ``.ladruno`` file's ``implexRefusals`` block is SIX
    columns wide and the reader resolves it to the six canonical names.
    """
    _require_commit_refusal_support()
    ops, sand, coords, bottom, top, p_c = _build_confined_hex(
        "f7e_reader", max_substeps=1,
    )
    path = str(tmp_path / "f7e_reader.ladruno")
    emitter, etag = _confine(
        ops, sand, coords, bottom, top, p_c, recorder_path=path,
    )

    # The deviatoric leg: the very first increment trips the
    # commit-time latch (see module docstring) -- capture the LIVE
    # eleResponse at the point of the aborted commit, since the guide
    # documents (§1.4(d)) that an aborted commit skips the recorder
    # loop entirely and leaves NO trace in the file.
    emitter.ops.timeSeries("Linear", 2)
    emitter.ops.pattern("Plain", 2, 2)
    for nid in top:
        emitter.ops.load(nid, 200.0, 0.0, -50.0)
    emitter.ops.integrator("LoadControl", 0.05)
    rc = emitter.ops.analyze(1)
    live_refusals = emitter.ops.eleResponse(
        etag, "material", 1, "implexRefusals",
    )
    assert rc == -4, f"expected the commit-time abort rc -4, got {rc}"
    assert len(live_refusals) == 6, live_refusals
    n_total, n_sign, n_ctrl, n_comp, commit_latched, n_latched = live_refusals
    assert commit_latched == 1.0, live_refusals
    assert n_total == n_comp > 0, live_refusals
    assert n_sign == 0.0 and n_ctrl == 0.0, live_refusals

    # Flush the recorder (it only ever saw the 10 elastic confinement
    # rows -- the aborted commit wrote nothing) and round-trip it
    # through the real reader.
    emitter.ops.remove("recorders")
    r = Results.from_ladruno(path)
    available = r.elements.gauss.available_components()
    missing = [c for c in _WANTED_NAMES if c not in available]
    assert not missing, f"{missing} absent from {sorted(available)}"

    for name in _WANTED_NAMES:
        values = r.elements.gauss.get(component=name).values
        # Every recorded row is from the elastic confinement leg (the
        # refusing step never reached the recorder), so the six
        # columns are all zero on file -- still proves the SHAPE and
        # NAMES the guide's §1.4(a) fix targets; the nonzero values
        # above are the live, pre-abort evidence the guide's §1.4(d)
        # says the file itself cannot carry.
        assert np.all(np.asarray(values) == 0.0), (name, values)


def test_commit_time_refusal_aborts_substep_drive_without_subdividing() -> None:
    """(b) analyze() -> -4 raises AnalysisAbortedError through
    Substep.drive BEFORE any halving — not a subdivision signal.
    """
    _require_commit_refusal_support()
    ops, sand, coords, bottom, top, p_c = _build_confined_hex(
        "f7e_substep", max_substeps=1,
    )
    emitter, etag = _confine(
        ops, sand, coords, bottom, top, p_c, recorder_path=None,
    )

    ctrl_node = top[0]
    ctrl_dof = 1  # x, 1-based (OpenSees convention)

    # DisplacementControl needs a reference load pattern for its
    # dU/dP sensitivity direction; shape only, magnitude is irrelevant
    # once a driver drives displacement directly.
    emitter.ops.timeSeries("Linear", 2)
    emitter.ops.pattern("Plain", 2, 2)
    for nid in top:
        emitter.ops.load(nid, 200.0, 0.0, -50.0)

    driver = OpenSeesPyDriver(
        emitter.ops, node=ctrl_node, dof=ctrl_dof, sign=1.0,
    )
    substep = Substep(
        node=ctrl_node, dof=ctrl_dof, target=0.05,
        ds=0.01, ds_min=1e-6, budget=8,
    )

    with pytest.raises(AnalysisAbortedError) as excinfo:
        substep.drive(driver)

    message = str(excinfo.value)
    assert "COMMIT was refused" in message
    assert "ds = 0.01" in message, (
        "the abort must fire on the FIRST attempted step size -- no "
        f"halving should have happened before it: {message!r}"
    )

    live_refusals = emitter.ops.eleResponse(
        etag, "material", 1, "implexRefusals",
    )
    assert len(live_refusals) == 6, live_refusals
    n_total, n_sign, n_ctrl, n_comp, commit_latched, n_latched = live_refusals
    assert commit_latched == 1.0, live_refusals
    assert n_total == n_comp > 0, live_refusals
