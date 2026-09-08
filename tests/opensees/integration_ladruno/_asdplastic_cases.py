"""Child-process cases for ``test_asdplastic_live.py`` (ADR 0105 D7).

Run as ``python _asdplastic_cases.py <case>`` in a fresh interpreter with
``stdin`` closed: the fork's ``.pyd`` writes its ``cout`` / ``opserr`` to
the process streams, which pytest's ``capfd`` cannot see but a subprocess
pipe can (both are recorded fork quirks, ``LEDGER_quirks.md``).  Every
case prints one ``RESULT {json}`` line; the parent test asserts on it and
on the captured streams.

Driver (mirrors the fork's own ADR-94 rigs, ``tests/test_adr94_hlist_hb.py``
and ``test_adr94_redblue_numerics.py``): one unit hex meshed through
apeGmsh, 1/8-symmetry restraints, normal strains prescribed through
``sp`` records under ``LoadControl``, ``system UmfPack`` (never
``FullGeneral`` — a fully/near-fully prescribed driver crashes it), some
DOFs left free.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

import numpy as np  # noqa: E402

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

#: The fork's ADR-94 MC reference set (``test_adr94_redblue_numerics.py``):
#: kPa, associated, sharp corners.
MC = dict(c=100.0, phi=30.0, psi=30.0, E=1.0e6, nu=0.25)
MC_DS = 0.0
#: Same physics x1e9 (kPa -> a made-up unit 1e9 smaller) — the fork's M5 deck.
UNIT_GAP = 1.0e9
#: 1e-7 * c*cos(phi) is 8.66e-6 in kPa and 8.66e-3 x1e9: the SAME relative
#: tightness, looser than the 1e-6 absolute the kPa run already passes at.
F_REL = 1.0e-7

#: Hoek-Brown: 50 MPa intact rock, mi 10, GSI 60, D 0 (fork wp/94d numbers).
HB_SIGCI, HB_MI, HB_GSI, HB_D = 50000.0, 10.0, 60.0, 0.0
HB_MB = HB_MI * math.exp((HB_GSI - 100.0) / (28.0 - 14.0 * HB_D))
HB_S = math.exp((HB_GSI - 100.0) / (9.0 - 3.0 * HB_D))
HB_A = 0.5 + (1.0 / 6.0) * (math.exp(-HB_GSI / 15.0) - math.exp(-20.0 / 3.0))
HB_SIGMA_T = HB_S * HB_SIGCI / HB_MB           # textbook tensile strength, > 0
HB_E, HB_NU = 5.0e7, 0.25

MC_IV = "BackStress(NullHardeningTensorFunction):"

#: The pre-ADR-0105 ``MohrCoulombSoil`` deck, VERBATIM (the 21-name
#: superset the helper used to emit, in its emitted order).  A fixture, not
#: regenerated: this is what the fork's ADR-94 parser refuses.
PRE_ADR94_SUPERSET_ARGS = [
    "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL", MC_IV,
    "Begin_Internal_Variables", "BackStress", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    "End_Internal_Variables",
    "Begin_Model_Parameters",
    "AF_cr", 0.0, "AF_ha", 0.0, "DP_eta", 0.0, "DP_etabar", 0.0,
    "DP_xi_c", 0.0, "Dilatancy", 0.0, "DuncanChang_MaxSigma3", 0.0,
    "DuncanChang_n", 0.0, "InitialP0", 0.0, "MC_c", 1014.0, "MC_ds", 1e-05,
    "MC_phi", 45.95, "MC_psi", 11.49, "MassDensity", 4.5,
    "PoissonsRatio", 0.18, "ReferencePressure", 0.0,
    "ReferenceYoungsModulus", 0.0, "ScalarLinearHardeningParameter", 0.0,
    "TC_min_stress", 0.0, "TensorLinearHardeningParameter", 0.0,
    "YoungsModulus", 4080000.0,
    "End_Model_Parameters",
    "Begin_Integration_Options",
    "f_absolute_tol", 1e-06, "stress_absolute_tol", 1e-06,
    "n_max_iterations", 100, "rk45_dT_min", 0.01, "rk45_niter_max", 100,
    "return_to_yield_surface", "Disabled",
    "integration_method", "Backward_Euler", "tangent_type", "Secant",
    "End_Integration_Options",
]


# ---------------------------------------------------------------------------
# mesh + driver
# ---------------------------------------------------------------------------


def _hex_fem(n_cubes: int = 1):
    """``n_cubes`` disjoint unit hexes along x, one PG each (``cube0``, ...)."""
    with apeGmsh(model_name="asdp_hex", verbose=False) as g:
        boxes = []
        for i in range(n_cubes):
            boxes.append(g.model.geometry.add_box(2.0 * i, 0, 0, 1, 1, 1))
        g.model.sync()
        for b in boxes:
            g.mesh.structured.set_transfinite_box(b, n=2)
        g.mesh.generation.generate(3)
        for i, b in enumerate(boxes):
            g.physical.add(3, [b], name=f"cube{i}")
        return g.mesh.queries.get_fem_data(dim=3)


def _plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _fix_eighth(ops, fem) -> None:
    """1/8-symmetry: the x=0 / y=0 / z=0 faces lose their normal DOF."""
    ops.fix(nodes=_plane(fem, 0, 0.0), dofs=(1, 0, 0))
    ops.fix(nodes=_plane(fem, 1, 0.0), dofs=(0, 1, 0))
    ops.fix(nodes=_plane(fem, 2, 0.0), dofs=(0, 0, 1))


def _chain(ops, nsteps: int, tol: float = 1e-10, max_iter: int = 50) -> None:
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=tol, max_iter=max_iter)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / nsteps)
    ops.analysis.Static()


def _build_cube(
    material, *, element: str, nsteps: int,
    eps_x: float, eps_y: float | None = None, eps_z: float | None = None,
):
    """Single hex; ``None`` leaves that face's normal traction FREE."""
    fem = _hex_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = material(ops)
    getattr(ops.element, element)(pg="cube0", material=mat)
    _fix_eighth(ops, fem)
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for n in _plane(fem, 0, 1.0):
            p.sp(node=n, dof=1, value=eps_x)
        if eps_y is not None:
            for n in _plane(fem, 1, 1.0):
                p.sp(node=n, dof=2, value=eps_y)
        if eps_z is not None:
            for n in _plane(fem, 2, 1.0):
                p.sp(node=n, dof=3, value=eps_z)
    _chain(ops, nsteps)
    return ops


def _emit(ops) -> tuple[LiveOpsEmitter, list[str]]:
    """Build + emit live; returns the emitter and the build-gate warnings."""
    em = LiveOpsEmitter(wipe=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ops.build().emit(em)
    return em, [f"{w.category.__name__}: {w.message}" for w in rec]


def _drive(em: LiveOpsEmitter, nsteps: int):
    """Advance up to ``nsteps``; stop at the first non-zero ``analyze()``.

    Returns ``(codes, hist, iters)``: every code up to and including the
    first failure, the committed GP-1 stress (6 Voigt comps, tension
    positive) after each SUCCESSFUL step, and the Newton iterations per
    step.
    """
    o = em.ops
    tags = o.getEleTags()
    tag = int(tags if isinstance(tags, int) else tags[0])
    codes, hist, iters = [], [], []
    for _ in range(nsteps):
        rc = em.analyze(steps=1)
        codes.append(int(rc))
        if rc != 0:
            break
        iters.append(int(o.testIter()))
        o.eleResponse(tag, "forces")            # set the lazy strain/stress
        hist.append([float(v) for v in list(o.eleResponse(tag, "stresses"))[0:6]])
    return codes, hist, iters


def _f_mc(stress6, *, c: float, phi_deg: float) -> float:
    """Sharp Mohr-Coulomb yield function, tension positive, in principal
    stresses: ``(s1 - s3)/2 + (s1 + s3)/2 * sin(phi) - c * cos(phi)``."""
    sxx, syy, szz, sxy, syz, sxz = stress6
    t = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    s1, s3 = float(np.max(np.linalg.eigvalsh(t))), float(np.min(np.linalg.eigvalsh(t)))
    phi = math.radians(phi_deg)
    return 0.5 * (s1 - s3) + 0.5 * (s1 + s3) * math.sin(phi) - c * math.cos(phi)


def _mc(scale: float = 1.0, **kw):
    def make(ops):
        return ops.nDMaterial.MohrCoulombSoil(
            c=MC["c"] * scale, phi=MC["phi"], psi=MC["psi"],
            E=MC["E"] * scale, nu=MC["nu"], ds=MC_DS, **kw,
        )
    return make


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def case_schema_accepted() -> dict:
    """D1 deck accepted; a 20-step deviatoric strain leg on LadrunoBrick
    (``eps = (+0.002, +0.003, -0.010)``: three distinct principal
    stresses, so the state sits on the SMOOTH part of the surface, not a
    corner) yields in shear and commits admissible states — ``f_MC`` on the
    surface, never above ``f_absolute_tol``.

    The helper's default rounding (``ds = 1e-5``) is kept: with ``ds = 0``
    the same leg is refused at step 17 on the triaxial-compression corner.
    """
    nsteps = 20

    def make(ops):
        return ops.nDMaterial.MohrCoulombSoil(
            c=MC["c"], phi=MC["phi"], psi=MC["psi"], E=MC["E"], nu=MC["nu"],
        )

    ops = _build_cube(make, element="LadrunoBrick", nsteps=nsteps,
                      eps_x=0.002, eps_y=0.003, eps_z=-0.01)
    em, warns = _emit(ops)
    codes, hist, iters = _drive(em, nsteps)
    f = [_f_mc(s, c=MC["c"], phi_deg=MC["phi"]) for s in hist]
    return {
        "codes": codes, "f": f, "iters": iters, "warnings": warns,
        "f_absolute_tol": 1e-6,
        "strength_scale": MC["c"] * math.cos(math.radians(MC["phi"])),
        "last_stress": hist[-1] if hist else None,
    }


def _raw_refusal(args: list) -> dict:
    em = LiveOpsEmitter(wipe=True)
    o = em.ops
    o.model("basic", "-ndm", 3, "-ndf", 3)
    try:
        o.nDMaterial("ASDPlasticMaterial3D", 1, *args)
    except Exception as exc:                       # the binding raises
        sys.stdout.flush()
        sys.stderr.flush()
        return {"refused": True, "exception": f"{type(exc).__name__}: {exc}"}
    return {"refused": False, "exception": None}


def case_superset_refused() -> dict:
    """The pre-D1 superset deck (fixture) is refused naming the first foreign
    parameter — ``AF_cr`` in the block's own order."""
    return _raw_refusal(PRE_ADR94_SUPERSET_ARGS)


def case_missing_phi_refused() -> dict:
    """The D1 deck without ``MC_phi`` is refused naming ``MC_phi``."""
    args = [
        "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL", MC_IV,
        "Begin_Internal_Variables", "BackStress", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        "End_Internal_Variables",
        "Begin_Model_Parameters",
        "YoungsModulus", MC["E"], "PoissonsRatio", MC["nu"],
        "MC_c", MC["c"], "MC_ds", MC_DS, "MC_psi", MC["psi"],
        "MassDensity", 0.0, "InitialP0", 0.0,
        "End_Model_Parameters",
        "Begin_Integration_Options", "strict_convergence", 1,
        "End_Integration_Options",
    ]
    return _raw_refusal(args)


def case_strict_hosts() -> dict:
    """``strict_convergence=True`` on a deck the material cannot integrate:
    refused on LadrunoBrick (``analyze() != 0``), swallowed on stdBrick
    (``analyze() == 0`` throughout) — D4's rationale — and the D4 gate
    fires on the stdBrick deck at build time.

    The refusal is provoked the way the fork's own ``test_R2_strict_
    convergence_is_a_noop_on_stdbrick`` provokes it: the same MC problem
    x1e9 under the ABSOLUTE tolerance, unreachable at that scale (fork
    M5).  (A starved ``n_max_iterations = 1`` does NOT refuse on this
    build — Backward_Euler converges the confined path in one iteration —
    so it is not the reproducer.)
    """
    out: dict = {}
    for element in ("LadrunoBrick", "stdBrick"):
        ops = _build_cube(_mc(scale=UNIT_GAP), element=element, nsteps=20,
                          eps_x=0.0, eps_y=0.0, eps_z=0.01)
        em, warns = _emit(ops)
        codes, _hist, _ = _drive(em, 20)
        out[element] = {"codes": codes, "warnings": warns}
    return out


def case_hb_tension() -> dict:
    """``HoekBrownRock`` uniaxial-STRESS tension (lateral faces free) driven
    to 3x the textbook tensile strain.

    With ``strict_convergence=False`` (the fork's own wp/94d rig) the last
    committed state sits on the textbook plateau ``s*sigci/mb``; the next
    step is the composite's non-smooth corner, which Backward_Euler cannot
    return from (the fork's open corner-return follow-up).  With the
    ADR 0105 default ``strict_convergence=True`` that corner step is
    REFUSED instead of committed, so the last committed state is the last
    ELASTIC one, below the plateau — nothing above ``sigma_t`` is ever
    committed.
    """
    nsteps = 60
    eps_end = 3.0 * HB_SIGMA_T / HB_E
    out: dict = {"sigma_t": HB_SIGMA_T, "mb": HB_MB, "s": HB_S, "a": HB_A}
    for strict in (False, True):
        def make(ops, strict=strict):
            return ops.nDMaterial.HoekBrownRock(
                E=HB_E, nu=HB_NU, sigci=HB_SIGCI, mb=HB_MB, s=HB_S, a=HB_A,
                mb_psi=HB_MB, ds=0.0, strict_convergence=strict,
            )
        ops = _build_cube(make, element="LadrunoBrick", nsteps=nsteps,
                          eps_x=eps_end)
        em, warns = _emit(ops)
        codes, hist, _ = _drive(em, nsteps)
        out[f"strict_{int(strict)}"] = {
            "codes": codes, "n_committed": len(hist),
            "sigma_xx": [h[0] for h in hist], "warnings": warns,
        }
    return out


def case_relative_tol_units() -> dict:
    """``f_relative_tol`` on: the same MC problem completes in kPa and x1e9."""
    out: dict = {}
    for label, scale in (("kpa", 1.0), ("x1e9", UNIT_GAP)):
        ops = _build_cube(_mc(scale=scale, f_relative_tol=F_REL),
                          element="LadrunoBrick", nsteps=20,
                          eps_x=0.0, eps_y=0.0, eps_z=0.01)
        em, _ = _emit(ops)
        codes, hist, _ = _drive(em, 20)
        out[label] = {"codes": codes, "last_stress": hist[-1] if hist else None}
    out["unit_gap"] = UNIT_GAP
    return out


#: The fork's two-cube VonMises rig (``test_adr94_redblue_blue.py`` /
#: ``test_adr94_hlist_numerics.py``): isotropic linear hardening, one cube
#: driven plastic and one kept elastic under load control.
VM = dict(E=70000.0, nu=0.3, sy=30.0, h=7000.0)
VM_IV = (
    "BackStress(TensorLinearHardeningFunction):"
    "YieldStress(ScalarLinearHardeningFunction):"
)
VM_LOAD_PL, VM_LOAD_EL, VM_NSTEPS = -60.0, -10.0, 4


def case_continuum_vs_secant() -> dict:
    """Two disjoint cubes, base fixed, top face loaded (one plastic, one
    elastic): total Newton iterations with ``Continuum`` <= with ``Secant``,
    final stresses equal at convergence."""

    def run(tangent: str) -> dict:
        fem = _hex_fem(n_cubes=2)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ASDPlasticMaterial3D(
            yf="VonMises_YF", pf="VonMises_PF", el="LinearIsotropic3D_EL",
            iv=VM_IV,
            internal_variables={"YieldStress": VM["sy"], "BackStress": (0.0,) * 6},
            model_parameters={
                "YoungsModulus": VM["E"], "PoissonsRatio": VM["nu"],
                "ScalarLinearHardeningParameter": VM["h"],
                "TensorLinearHardeningParameter": 0.0, "MassDensity": 0.0,
            },
            integration_options={
                "integration_method": "Backward_Euler",
                "tangent_type": tangent, "n_max_iterations": 100,
            },
        )
        ops.element.LadrunoBrick(pg="cube0", material=mat)
        ops.element.LadrunoBrick(pg="cube1", material=mat)
        ops.fix(nodes=_plane(fem, 2, 0.0), dofs=(1, 1, 1))
        xyz = dict(zip((int(n) for n in fem.nodes.ids), np.asarray(fem.nodes.coords)))
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as pat:
            for n in _plane(fem, 2, 1.0):
                load = VM_LOAD_PL if xyz[n][0] < 1.5 else VM_LOAD_EL
                pat.load(node=n, forces=(0.0, 0.0, load / VM_NSTEPS))
        _chain(ops, 1, tol=1e-9, max_iter=200)      # dlam = 1 per step
        em, _ = _emit(ops)
        o = em.ops
        codes, total = [], 0
        for _ in range(VM_NSTEPS):
            rc = em.analyze(steps=1)
            codes.append(int(rc))
            if rc != 0:
                break
            total += int(o.testIter())
        stresses = {}
        for tag in [int(t) for t in o.getEleTags()]:
            o.eleResponse(tag, "forces")
            stresses[str(tag)] = [
                float(v) for v in list(o.eleResponse(tag, "stresses"))[0:6]
            ]
        return {"codes": codes, "iterations": total, "stresses": stresses}

    return {"Continuum": run("Continuum"), "Secant": run("Secant")}


CASES = {
    name[len("case_"):]: fn
    for name, fn in list(globals().items()) if name.startswith("case_")
}


def main(argv: list[str]) -> int:
    name = argv[1]
    em = LiveOpsEmitter(wipe=True)
    build = em.ops.ladrunoBuild()
    print(f"LADRUNO_BUILD {build}")
    result = CASES[name]()
    result["ladruno_build"] = build
    sys.stdout.flush()
    print("RESULT " + json.dumps(result))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
