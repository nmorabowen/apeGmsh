"""Fork-only — ``LadrunoSANISAND`` against a real fork build (ADR 86 §5.3).

I1  With ``p_residual`` / ``p_min`` pinned to vanilla's hardcoded values
    (``1.0e-2*P_atm``, ``1.0e-4*P_atm``) and ``honor_tol_r=False``, a
    ``LadrunoSANISAND`` deck reproduces the equivalent ``ManzariDafalias``
    deck **bit-identically** — the fork's own G1 gate.  Both primitives
    live in one apeGmsh module, so the test builds ONE parameter dict and
    emits it twice.  If this fails, the emitter is not emitting what it
    thinks it is.
I2  The fork's construction echo names ``p_residual = 0`` and the
    resolved ``p_min`` — what apeGmsh emitted is what the material
    reports running.  Asserting on the echo is the point of the echo.
I3  The confine-first stage ordering (ADR 86 §4.2) produces **zero**
    ``Elastic2Plastic`` inflation warnings; the naive ordering — a
    deviatoric leg during stage 0 — produces some (the ``raising M_c ...
    (Outside Bounding!)`` diagnostic restored by fork PR #714).
I4  A proportional strain ramp at ``p_residual=0`` never yields.  The
    K0/oedometric deck is EXACTLY that ramp: the elastic stress path
    keeps ``σh/σv`` constant, so continuing the same loading after the
    flip is radial through the origin — and stage-0's
    ``elastic_integrator`` welded ``α = s/(p + p_r)`` on every step, so
    with ``p_r = 0`` the weld is ``α = r`` exactly and the yield function
    stays at ``-√(2/3)·m·p < 0`` forever.  The freeze signature is a
    bit-frozen ``α`` and exactly zero plastic strain, with the analysis
    converging and reporting success throughout.  (It is NOT
    displacement equality with a never-flipped run: the stage-0 and
    stage-1 *elastic* code paths integrate the pressure-dependent moduli
    differently, so the two runs differ by a few percent in strain for
    the same stress — neither has yielded.)  Vanilla's ``p_r > 0`` is the
    contrast: ``s/(p + p_r)`` drifts along the ray and the cone is
    eventually crossed.  The test asserts the known freeze so the
    behaviour is documented rather than discovered (ADR 86 §4.3; the
    fork pins the same fact as ``test_radial_ramp_with_pr0_never_yields``;
    the mitigation is a stress-path change — exactly what I1/I3's
    triaxial deck does).  If a future fork revisits the open formulation
    questions and the ramp starts yielding, this test fails loudly —
    rewrite it to assert the yield.

Live-run shape: ``LiveOpsEmitter`` does not execute STAGED decks
(``stage_open`` raises), so these tests use its hand-rolled multi-step
forwarders instead — emit the non-staged deck, analyze the elastic leg,
call ``em.update_material_stage(tag, 1)`` (the same
``updateMaterialStage -material $tag -stage 1`` line the staged emitters
produce), analyze the plastic leg.

Skips: the root conftest auto-skips the ``ladruno_fork`` marker off the
fork backend; on a fork build that predates the material (fork PRs
#767/#768/#771/#772) the in-test probe skips instead.  The probe also
guards the stage semantics these decks rely on: the fork's start-elastic
constructor default (TIMs item 7) landed BEFORE the material, so any
build that has the material also starts its SANISAND family in stage 0.
On an older build materials are born in stage 1, every flip below would
be a silent no-op, and both A/B legs would silently compare the same
run — but such builds fail the probe anyway.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork


#: Gorini's calibrated set (ADR 86 §6).  Units are kPa (and m / Mg for the
#: mesh): the D_factor repair (ADR-86 D5b) is an exact no-op at
#: ``P_atm = 101.0`` and only there.
_GORINI = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "z_max": 12.5, "cz": 1100.0, "rho": 2.0,
}

#: Vanilla's hardcoded low-stress constants, pinned for the I1 A/B — the
#: ONLY thing that may differ between the two legs is the class name.
_VANILLA_P_RESIDUAL = 1.0e-2 * 101.0
_VANILLA_P_MIN = 1.0e-4 * 101.0

_HAS_MATERIAL: bool | None = None


def _require_ladruno_sanisand() -> None:
    """Probe-skip on fork builds that predate the material.

    Probes the RESOLVED backend (``LiveOpsEmitter.ops``), not a bare
    ``import openseespy.opensees``: a Ladruno install's venv boot hook
    binds that name to the installed build at interpreter startup, while
    ``APEGMSH_OPENSEES_BIN`` can point the resolver at a different
    ``dist\bin`` — probing the wrong one would skip (or run) the wrong
    tests.
    """
    global _HAS_MATERIAL
    if _HAS_MATERIAL is None:
        o = LiveOpsEmitter(wipe=True).ops
        o.model("basic", "-ndm", 3, "-ndf", 3)
        try:
            o.nDMaterial(
                "LadrunoSANISAND", 90210,
                _GORINI["G0"], _GORINI["nu"], _GORINI["e_init"],
                _GORINI["Mc"], _GORINI["c"], _GORINI["lambda_c"],
                _GORINI["e0"], _GORINI["ksi"], _GORINI["P_atm"],
                _GORINI["m"], _GORINI["h0"], _GORINI["Ch"],
                _GORINI["nb"], _GORINI["A0"], _GORINI["nd"],
                _GORINI["z_max"], _GORINI["cz"], _GORINI["rho"],
            )
            _HAS_MATERIAL = True
        except Exception:
            _HAS_MATERIAL = False
        finally:
            o.wipe()
    if not _HAS_MATERIAL:
        pytest.skip(
            "fork build predates LadrunoSANISAND "
            "(ADR 86, fork PRs #767/#768/#771/#772)"
        )


def _single_hex_fem():
    """One 1 m³ 8-node hex, PG 'soil' — the material-driver mesh."""
    with apeGmsh(model_name="ls_hex", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="soil")
        return g.mesh.queries.get_fem_data(dim=3)


def _plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _bind_chain(ops: apeSees, *, dlam: float) -> None:
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    # 1e-8, not tighter: the SANISAND consistent tangent leaves a residual
    # floor around 1e-9 on these single-element decks.
    ops.test.NormDispIncr(tol=1e-8, max_iter=200)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=dlam)
    ops.analysis.Static()


_MakeMaterial = Callable[[apeSees], object]


def _make_manzari(ops: apeSees):
    return ops.nDMaterial.ManzariDafalias(**_GORINI)


def _make_ladruno_pinned(ops: apeSees):
    """LadrunoSANISAND with vanilla's constants pinned — the I1 leg."""
    return ops.nDMaterial.LadrunoSANISAND(
        **_GORINI,
        p_residual=_VANILLA_P_RESIDUAL,
        p_min=_VANILLA_P_MIN,
        honor_tol_r=False,
    )


def _make_ladruno_default(ops: apeSees):
    """LadrunoSANISAND at its own defaults: p_residual=0, p_min→0.101."""
    return ops.nDMaterial.LadrunoSANISAND(**_GORINI)


def _node_coords(fem) -> dict[int, tuple[float, float, float]]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return {
        int(n): (float(p[0]), float(p[1]), float(p[2]))
        for n, p in zip(ids, xyz)
    }


def _triaxial_run(
    fem,
    make_material: _MakeMaterial,
    *,
    flip: bool = True,
    p_c: float = 100.0,
    dq: float = 100.0,
    n_confine: int = 10,
    n_dev: int = 40,
) -> float:
    """The §4.2 ordering on a NON-proportional path: isotropic confine
    (η = 0 — the safest possible stage-0 leg), flip, triaxial deviatoric
    ramp (``q: 0 → dq`` at constant cell pressure — the stress-ratio
    rotation that actually exercises the plasticity).  Returns the mean
    top settlement.

    Pressure on the top and the four lateral faces via consistent nodal
    forces (a corner of a unit bilinear face carries A/4); the smooth
    rigid base reacts the sixth face.  3-2-1 in-plane supports so the
    confinement stays statically determinate and exactly homogeneous.
    """
    coords = _node_coords(fem)
    bottom = _plane(fem, 2, 0.0)
    top = _plane(fem, 2, 1.0)
    origin = next(n for n, c in coords.items() if c == (0.0, 0.0, 0.0))
    on_x = next(n for n, c in coords.items() if c == (1.0, 0.0, 0.0))

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    sand = make_material(ops)
    ops.element.stdBrick(pg="soil", material=sand)
    ops.fix(nodes=[origin], dofs=(1, 1, 1))
    ops.fix(nodes=[on_x], dofs=(0, 1, 1))
    ops.fix(
        nodes=[n for n in bottom if n not in (origin, on_x)],
        dofs=(0, 0, 1),
    )

    t1 = n_confine / (n_confine + n_dev)
    # Path series, NOT Linear: a Path holds its last ordinate only up to
    # its final time point (0 after), so both extend well past λ = 1.
    ts_confine = ops.timeSeries.Path(
        values=(0.0, 1.0, 1.0), time=(0.0, t1, 10.0)
    )
    ts_dev = ops.timeSeries.Path(
        values=(0.0, 0.0, 1.0, 1.0), time=(0.0, t1, 1.0, 10.0)
    )
    with ops.pattern.Plain(series=ts_confine) as p:
        for nid, (x, y, z) in coords.items():
            fx = (p_c / 4.0) if x == 0.0 else (-p_c / 4.0)
            fy = (p_c / 4.0) if y == 0.0 else (-p_c / 4.0)
            fz = (-p_c / 4.0) if z == 1.0 else 0.0
            p.load(node=nid, forces=(fx, fy, fz))
    with ops.pattern.Plain(series=ts_dev) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -dq / len(top)))
    _bind_chain(ops, dlam=1.0 / (n_confine + n_dev))

    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    assert em.analyze(steps=n_confine) == 0, "isotropic confinement diverged"
    if flip:
        em.update_material_stage(ops.tag_for(sand), 1)
    assert em.analyze(steps=n_dev) == 0, "deviatoric leg diverged"
    return float(np.mean([em.ops.nodeDisp(int(n), 3) for n in top]))


def _oedometer_run(
    fem,
    make_material: _MakeMaterial,
    *,
    flip: bool = True,
    sigma_v: float = 200.0,
    n_elastic: int = 10,
    n_plastic: int = 20,
) -> dict[str, np.ndarray]:
    """K0 oedometer: confine elastic, flip, CONTINUE the same loading.

    Stage-safe (at elastic ``K0 = nu/(1-nu)`` the stress ratio stays well
    under ``Mc``) — but deliberately PROPORTIONAL: the elastic path keeps
    ``σh/σv`` constant, so the post-flip continuation is radial through
    the origin.  This is the §4.3 freeze deck for I4.  Lateral rollers
    everywhere, base pinned, the top face carries ``sigma_v`` at λ = 1.

    Returns Gauss-point-1 state: ``alpha_flip`` (back-stress ratio at the
    flip), ``alpha_end`` / ``pstrains_end`` after the post-flip leg, and
    the mean top settlement ``uz``.
    """
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    sand = make_material(ops)
    ops.element.stdBrick(pg="soil", material=sand)
    bottom = _plane(fem, 2, 0.0)
    top = _plane(fem, 2, 1.0)
    ops.fix(nodes=bottom, dofs=(1, 1, 1))
    free = [int(n) for n in np.asarray(fem.nodes.ids) if int(n) not in bottom]
    ops.fix(nodes=free, dofs=(1, 1, 0))
    ts = ops.timeSeries.Linear()
    per_node = sigma_v * 1.0 / len(top)  # uniform pressure on a 1 m² face
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -per_node))
    _bind_chain(ops, dlam=1.0 / (n_elastic + n_plastic))

    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    (etag,) = em.ops.getEleTags()

    def gp1(what: str) -> np.ndarray:
        return np.asarray(em.ops.eleResponse(etag, "material", 1, what), float)

    assert em.analyze(steps=n_elastic) == 0, "elastic confinement diverged"
    alpha_flip = gp1("alpha")
    if flip:
        em.update_material_stage(ops.tag_for(sand), 1)
    assert em.analyze(steps=n_plastic) == 0, "post-flip leg diverged"
    return {
        "alpha_flip": alpha_flip,
        "alpha_end": gp1("alpha"),
        "pstrains_end": gp1("pstrains"),
        "uz": np.asarray(
            np.mean([em.ops.nodeDisp(int(n), 3) for n in top]), float
        ),
    }


def _naive_uniaxial_run(
    fem, make_material: _MakeMaterial, *, n_elastic: int = 10
) -> None:
    """The §4.2 ANTI-pattern: a deviatoric leg during stage 0, then flip.

    Sides free, so the elastic leg is uniaxial STRESS — ``η = q/p = 3``,
    far outside ``Mc = 1.331`` — and the flip trips ``Elastic2Plastic``'s
    M_c inflation.  The flip itself fires the diagnostic; no post-flip
    analyze is needed.
    """
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    sand = make_material(ops)
    ops.element.stdBrick(pg="soil", material=sand)
    bottom = _plane(fem, 2, 0.0)
    top = _plane(fem, 2, 1.0)
    ops.fix(nodes=bottom, dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -50.0 / len(top)))
    _bind_chain(ops, dlam=1.0 / n_elastic)

    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    assert em.analyze(steps=n_elastic) == 0, "elastic uniaxial leg diverged"
    em.update_material_stage(ops.tag_for(sand), 1)


# ---------------------------------------------------------------------------
# I1 — the fork's G1 gate, through the emitter
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_i1_pinned_ladruno_sanisand_reproduces_manzari_bit_identically() -> None:
    _require_ladruno_sanisand()
    fem = _single_hex_fem()

    reference = _triaxial_run(fem, _make_manzari)

    # Sanity: the flip must actually engage plasticity, or the A/B only
    # compares two elastic runs and proves nothing about the material.
    # (An un-flipped run integrates the whole deviatoric leg in stage 0.)
    elastic_only = _triaxial_run(fem, _make_manzari, flip=False)
    assert reference != pytest.approx(elastic_only, rel=1e-6), (
        "post-flip leg stayed elastic — the G1 gate would compare nothing"
    )

    got = _triaxial_run(fem, _make_ladruno_pinned)
    # Bit-identical, not approximately equal: with the constants pinned
    # the subclass runs the base's exact code path.
    assert got == reference, (
        f"pinned LadrunoSANISAND diverged from ManzariDafalias: "
        f"{got!r} != {reference!r}"
    )


# ---------------------------------------------------------------------------
# I2 — the construction echo names what apeGmsh emitted
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_i2_construction_echo_names_the_emitted_constants(capfd) -> None:
    _require_ladruno_sanisand()
    fem = _single_hex_fem()
    capfd.readouterr()  # drain — the probe above echoes the same strings

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    sand = _make_ladruno_default(ops)
    ops.element.stdBrick(pg="soil", material=sand)
    ops.fix(nodes=_plane(fem, 2, 0.0), dofs=(1, 1, 1))
    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)

    out, err = capfd.readouterr()
    text = out + err
    # Substring-prefix asserts so the stream's float formatting (0.101 vs
    # 0.101000) cannot break them.
    assert "p_residual = 0" in text
    assert "(default, cohesionless)" in text
    assert "p_min = 0.101" in text  # 1.0e-3 * P_atm, resolved by apeGmsh
    assert "honorTolR = 0" in text


# ---------------------------------------------------------------------------
# I3 — deck ordering: confine-first is clean, shear-during-stage-0 inflates
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_i3_confine_first_is_clean_naive_ordering_inflates_mc(capfd) -> None:
    _require_ladruno_sanisand()
    fem = _single_hex_fem()
    capfd.readouterr()

    _triaxial_run(fem, _make_ladruno_default)
    out, err = capfd.readouterr()
    assert "Outside Bounding" not in out + err, (
        "the confine-first ordering must produce ZERO M_c inflation events"
    )

    _naive_uniaxial_run(fem, _make_ladruno_default)
    out, err = capfd.readouterr()
    assert "Outside Bounding" in out + err, (
        "the naive ordering should trip the Elastic2Plastic inflation "
        "diagnostic (restored by fork PR #714) — if it no longer does, "
        "the fork's stage-switch behaviour changed; re-audit §4.2"
    )


# ---------------------------------------------------------------------------
# I4 — the proportional-ramp freeze at p_residual = 0
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_i4_proportional_ramp_at_zero_p_residual_freezes() -> None:
    _require_ladruno_sanisand()
    fem = _single_hex_fem()

    # The K0 oedometer IS the proportional deck (§4.3): stage-0 gravity,
    # then the same loading continued — the stress path is radial through
    # the origin.  With p_residual = 0 the elastic stage welded α = s/p =
    # r EXACTLY, so after the flip the yield function sits at
    # -√(2/3)·m·p < 0 and stays there for every further step.  The run
    # converges and reports success throughout (asserted inside the
    # helper) — that is the trap this test documents.
    frozen = _oedometer_run(fem, _make_ladruno_default)
    assert np.allclose(
        frozen["alpha_end"], frozen["alpha_flip"], rtol=0.0, atol=1e-12
    ), (
        "α moved after the flip — the proportional ramp yielded: either "
        "the freeze was repaired on the fork side (rewrite this test to "
        "assert the yield) or the deck is no longer proportional"
    )
    assert np.allclose(frozen["pstrains_end"], 0.0, atol=1e-14), (
        "plastic strain accumulated on the proportional ramp at "
        "p_residual = 0 — the freeze was repaired; rewrite to assert yield"
    )

    # Contrast 1 — vanilla's constants on the SAME deck: with p_r > 0 the
    # welded ratio s/(p + p_r) drifts along the ray as p grows, the tiny
    # cone (m = 0.005) is crossed within the leg, and α moves.  The freeze
    # is a property of p_r = 0 on a proportional deck.
    vanilla = _oedometer_run(fem, _make_ladruno_pinned)
    assert not np.allclose(
        vanilla["alpha_end"], vanilla["alpha_flip"], rtol=0.0, atol=1e-6
    ), (
        "vanilla's p_residual should have let α drift off the ray — if "
        "even this froze, the stage flip is not reaching the material"
    )

    # Contrast 2 — same material, same confinement level, NON-proportional
    # path: the triaxial q-ramp rotates the stress ratio and yields at
    # once (§4.3: NTUASand02 ships p_r = 0 and is used for staged
    # foundation work, where the stress direction changes constantly and
    # the freeze never occurs).
    triax_flipped = _triaxial_run(fem, _make_ladruno_default)
    triax_elastic = _triaxial_run(fem, _make_ladruno_default, flip=False)
    assert triax_flipped != pytest.approx(triax_elastic, rel=1e-6), (
        "the rotating stress path should have yielded"
    )
