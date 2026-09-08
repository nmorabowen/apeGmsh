"""Fork-only — the ADR-95 ``ladrunoBranch`` DruckerPrager diagnostic.

The fork's repaired UW ``DruckerPrager`` (fork PR #803, merged
``61b3efa04``) exposes a per-Gauss-point read-only response,
``eleResponse <e> material <gp> ladrunoBranch`` -> 8 floats
``[branch, gamma0, gamma1, f1_trial, f2_trial, forcedAccept, I1,
detAmin]``.  ADR 0108 makes it a read-back contract; this is its live
gate.

Deck: ONE ``LadrunoBrick`` driven by a prescribed HYDROSTATIC expansion
(every one of the 24 DOFs is an ``sp``), so the run is a material driver
and cannot diverge no matter what the tangent does — the point of the
exercise is the return map, not the solver.  Zero deviator means the
trial state walks straight up the hydrostatic axis and crosses the
tension cutoff / cone apex at ``I1 = sqrt(2/3) * sigmaY / rho``.

Values are cross-checked against ``ops.eleResponse`` on the live domain
— the engine's own flat vector, an authority independent of the reader
under test — exactly as ``test_ladruno_gauss_generic_columns`` does.

**Build gate.** Every engine older than ``61b3efa04`` answers the token
with an EMPTY list (no exception), so the contract tests probe-skip the
way ``test_ladruno_sanisand_responses_live`` does.  Measured on fork
build ``1652f945c`` (the installed one when this was written): the probe
answers ``False``, the ``LadrunoRecorder`` warns "returned no response"
and writes no bucket, and the driver ends with ``I1 = 8.08`` against a
cutoff of ``0.449`` — the dead cutoff branch, unreturned, which is the
defect ADR-95 repaired.  :func:`test_pre_fix_engine_is_self_consistent`
runs on any fork build and pins that correspondence in both directions.
The ASD-DP sibling needs a second floor, ``67474aeb7``; it has no
response token and is not probed here.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import (
    LADRUNO_BRANCH_WIDTH,
    probe_ladruno_branch,
)
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

# ── Material: zero-dilatancy frictional, tiny apex regulariser ────────
# phi_txc = 20 deg -> rho = 2*sqrt(2)*sin(phi) / (3 - sin(phi)).
# sigmaY = 0.2 is the guide's apex REGULARISER, not cohesion.
# rhoBar = 0 (psi = 0) makes the tangent unsymmetric -> UmfPack.
_PHI = math.radians(20.0)
RHO = 2.0 * math.sqrt(2.0) * math.sin(_PHI) / (3.0 - math.sin(_PHI))
SIGMA_Y = 0.2
_E, _NU = 2.0e4, 0.45
BULK_K = _E / (3.0 * (1.0 - 2.0 * _NU))
SHEAR_G = _E / (2.0 * (1.0 + _NU))

#: The tension cutoff, per ``guide_ladruno_adr95_druckerprager_fix.md``
#: section 2: ``I1 = sqrt(2/3) * sigma_y / rho``.  With zero deviator
#: this plane is also the cone apex, so a purely hydrostatic tensile
#: path lands on the CORNER (branch 3) rather than the cutoff alone
#: (branch 2) — the assertions below accept either, as the fork's own
#: active-set bookkeeping decides which.
T_CUTOFF = math.sqrt(2.0 / 3.0) * SIGMA_Y / RHO

#: The eight column names the ``.ladruno`` reader gives the response,
#: in the fill-site order (``DruckerPrager::getLadrunoBranch()``).
#: Spelled out rather than imported from ``_MATERIAL_BUCKET_TOKENS`` —
#: that table is what is under test, so importing it would assert
#: nothing.
BRANCH_COLUMNS = (
    "dp_branch", "dp_gamma_cone", "dp_gamma_cutoff",
    "dp_f1_trial", "dp_f2_trial", "dp_forced_accept",
    "dp_i1", "dp_det_a_min",
)

#: Prescribed hydrostatic strain: at load factor 1 the ELASTIC trial
#: would sit at ``I1 = 8 * T_CUTOFF``.  With ``dlam = 0.05`` step 1 is
#: at ``0.4 * T`` (comfortably elastic) and every step from 3 on is
#: past the cutoff.
_LAMBDA_STEP = 0.05
_N_STEPS = 6
_EPS_AXIAL = 8.0 * T_CUTOFF / (9.0 * BULK_K)


def _driver_fem():
    """One 1 m3 8-node hex, PG ``soil``."""
    with apeGmsh(model_name="dp_branch_hex", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="soil")
        return g.mesh.queries.get_fem_data(dim=3)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    """Run the hydrostatic-tension driver once; keep the live domain.

    Returns ``(emitter, ladruno_path, element_tag, n_gp)``.  Module
    scope because the live cross-check needs the SAME domain the file
    was written from — a re-run would wipe it.
    """
    fem = _driver_fem()
    ids = [int(n) for n in fem.nodes.ids]
    xyz = {
        int(n): tuple(float(v) for v in p)
        for n, p in zip(fem.nodes.ids, fem.nodes.coords)
    }

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.DruckerPrager(
        K=BULK_K, G=SHEAR_G, sigmaY=SIGMA_Y, rho=RHO, rhoBar=0.0,
        Kinf=0.0, Ko=0.0, delta1=0.0, delta2=0.0, H=0.0, theta=0.0,
    )
    ops.element.LadrunoBrick(pg="soil", material=mat)
    # Every DOF prescribed: u_i = eps * x_i, a uniform hydrostatic
    # expansion. No free equations, so the singular apex tangent cannot
    # stall the run.
    ops.fix(nodes=ids, dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in ids:
            x, y, z = xyz[nid]
            for dof, coord in ((1, x), (2, y), (3, z)):
                p.sp(node=nid, dof=dof, value=_EPS_AXIAL * coord)

    path = str(tmp_path_factory.mktemp("dp_branch") / "dp_branch.ladruno")
    ops.recorder.Ladruno(
        file=path, elem_responses=("stress", "material.ladrunoBranch"),
    )
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()          # unsymmetric: rhoBar != rho
    ops.test.NormDispIncr(tol=1.0e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=_LAMBDA_STEP)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for step in range(_N_STEPS):
        assert emitter.analyze(steps=1) == 0, f"driver diverged at {step}"

    tags = emitter.ops.getEleTags()
    etag = int(tags if isinstance(tags, int) else tags[0])
    n_gp = len(emitter.ops.eleResponse(etag, "stress")) // 6
    yield emitter, path, etag, n_gp
    emitter.ops.remove("recorders")


def _flush(driver):
    """Close the recorder so the ``.ladruno`` is readable."""
    emitter, path, _etag, _n_gp = driver
    emitter.ops.remove("recorders")
    return path


def _live_sweep(emitter, etag: int, n_gp: int) -> np.ndarray:
    """``(n_gp, 8)`` of ``ladrunoBranch``, straight off the domain."""
    return np.array(
        [
            emitter.ops.eleResponse(etag, "material", str(gp), "ladrunoBranch")
            for gp in range(1, n_gp + 1)
        ],
        dtype=np.float64,
    )


def _require_post_fix(driver) -> None:
    emitter, _path, etag, _n_gp = driver
    if not probe_ladruno_branch(emitter.ops, etag):
        pytest.skip(
            "engine predates fork 61b3efa04 (ADR-95): "
            "eleResponse(e, 'material', gp, 'ladrunoBranch') is empty"
        )


# ---------------------------------------------------------------------
# Runs on ANY fork build — the probe and the file must tell one story
# ---------------------------------------------------------------------

def test_pre_fix_engine_is_self_consistent(driver) -> None:
    """The capability probe and the recorded file agree about the build.

    Post-fix: eight floats live, and the reader's ``dp_branch`` column
    present in the file.  Pre-fix: an empty reply AND no bucket at all
    (the recorder warns "returned no response" and writes nothing).
    Anything in between — a probe that says yes over a file with no
    column, or the reverse — would mean the recorder and the material
    disagree, which is the failure mode this whole read-back contract
    exists to make loud.
    """
    emitter, _path, etag, _n_gp = driver
    has_response = probe_ladruno_branch(emitter.ops, etag)

    r = Results.from_ladruno(_flush(driver))
    has_column = "dp_branch" in r.elements.gauss.available_components()

    assert has_response == has_column, (
        f"eleResponse says {has_response} but the .ladruno file "
        f"{'has' if has_column else 'has no'} a dp_branch column"
    )


# ---------------------------------------------------------------------
# Post-fix contract
# ---------------------------------------------------------------------

def test_token_returns_eight_floats_in_the_documented_layout(driver) -> None:
    """The response width and the value ranges the layout implies."""
    _require_post_fix(driver)
    emitter, _path, etag, n_gp = driver

    sweep = _live_sweep(emitter, etag, n_gp)
    assert sweep.shape == (n_gp, LADRUNO_BRANCH_WIDTH)
    assert np.all(np.isfinite(sweep))

    branch = sweep[:, 0]
    assert set(np.unique(branch)).issubset({0.0, 1.0, 2.0, 3.0}), (
        f"slot 0 must be a branch code 0/1/2/3, got {sorted(set(branch))}"
    )
    # Slot 5 is a flag; slots 1-2 are plastic multipliers (never negative).
    assert set(np.unique(sweep[:, 5])).issubset({0.0, 1.0})
    assert np.all(sweep[:, 1] >= 0.0) and np.all(sweep[:, 2] >= 0.0)


def test_a_driven_tensile_point_is_returned_to_the_cutoff(driver) -> None:
    """Branch 2 or 3, and ``I1`` back at ``sqrt(2/3)*sigmaY/rho``.

    This is the whole defect: before the fix the cutoff residual row was
    never assembled, so the point kept its UNRETURNED stress — measured
    at ``I1 = 8.08`` on build ``1652f945c`` against this deck's cutoff
    of ``0.449``.
    """
    _require_post_fix(driver)
    emitter, _path, etag, n_gp = driver
    sweep = _live_sweep(emitter, etag, n_gp)

    branch, i1 = sweep[:, 0], sweep[:, 6]
    tensile = np.flatnonzero(np.isin(branch, (2.0, 3.0)))
    assert tensile.size == n_gp, (
        f"a purely hydrostatic tensile path past I1 = {T_CUTOFF:.4f} must "
        f"put EVERY Gauss point on the cutoff or the corner; got branches "
        f"{sorted(set(branch))}"
    )
    np.testing.assert_allclose(i1[tensile], T_CUTOFF, rtol=1e-6)
    # The trial violated the cutoff — otherwise the branch above is a
    # bookkeeping accident rather than a return.
    assert np.all(sweep[tensile, 4] > 0.0), "f2_trial must be > 0 here"


def test_an_elastic_step_reports_branch_zero_with_positive_det(
    driver,
) -> None:
    """Step 1 sits at ``0.4 * T`` — elastic, and ``detAmin`` is O(1).

    Read from the recorded history rather than the live domain: the
    domain only holds the LAST state, and the elastic step is the first
    one.  ``detAmin`` is normalised by ``(2G)^3``, so it is O(1) while
    elastic and goes negative (~-0.06) once the point yields.
    """
    _require_post_fix(driver)
    _emitter, _path, _etag, n_gp = driver
    r = Results.from_ladruno(_flush(driver))
    gauss = r.elements.gauss

    branch_slab = gauss.get(component="dp_branch")
    det_slab = gauss.get(component="dp_det_a_min")
    # An absent bucket reads as an EMPTY slab, and every assertion below
    # would then pass vacuously — pin the width first.
    assert branch_slab.values.shape[1] == n_gp
    assert det_slab.values.shape[1] == n_gp
    assert branch_slab.values.shape[0] >= 2, "need an elastic AND a yielded step"

    first_branch = branch_slab.values[0]
    first_det = det_slab.values[0]
    np.testing.assert_array_equal(first_branch, np.zeros_like(first_branch))
    assert np.all(first_det > 0.0), (
        f"an elastic point must keep det(n.C_ep.n) > 0; got {first_det}"
    )
    # And the acoustic tensor degrades once it yields.
    assert np.all(det_slab.values[-1] < first_det)


def test_reader_columns_match_the_live_eleResponse(driver) -> None:
    """All eight names, and the values the engine itself reports.

    The by-position map in ``_MATERIAL_BUCKET_TOKENS`` is the ONLY
    authority on what ``C1..C8`` mean (the fork sets no ResponseType),
    so this is the check that it is not off by a slot.
    """
    _require_post_fix(driver)
    emitter, _path, etag, n_gp = driver
    live = _live_sweep(emitter, etag, n_gp)

    r = Results.from_ladruno(_flush(driver))
    available = r.elements.gauss.available_components()
    missing = [c for c in BRANCH_COLUMNS if c not in available]
    assert not missing, f"{missing} absent from {sorted(available)}"

    for slot, name in enumerate(BRANCH_COLUMNS):
        slab = r.elements.gauss.get(component=name)
        assert slab.values.shape[1] == n_gp, name
        # One element, so the slab column order is the GP order; compare
        # as a multiset anyway — these buckets carry no GP_PARAM.
        np.testing.assert_allclose(
            np.sort(slab.values[-1]), np.sort(live[:, slot]),
            rtol=1e-10, atol=1e-12, err_msg=f"slot {slot} -> {name}",
        )


def test_censuses_agree_with_a_direct_eleResponse_sweep(driver) -> None:
    """``corner_census`` / ``tension_census`` against the engine.

    ``corner_census`` must equal the count of ``branch == 3`` in the
    live sweep; ``tension_census`` the count of Gauss points whose
    ``I1/3`` is non-negative, taken from the engine's own ``stress``
    vector rather than from the reader.
    """
    _require_post_fix(driver)
    emitter, _path, etag, n_gp = driver
    live = _live_sweep(emitter, etag, n_gp)
    live_corner = int(np.count_nonzero(live[:, 0] == 3.0))

    stress = np.asarray(
        emitter.ops.eleResponse(etag, "stress"), dtype=np.float64,
    ).reshape(n_gp, 6)
    live_tensile = int(np.count_nonzero(stress[:, :3].sum(axis=1) / 3.0 >= 0.0))

    gauss = Results.from_ladruno(_flush(driver)).elements.gauss
    corner = gauss.corner_census()
    tension = gauss.tension_census()

    assert (corner.count, corner.examined) == (live_corner, n_gp)
    assert (tension.count, tension.examined) == (live_tensile, n_gp)
    # The driver is hydrostatic tension: every point is in tension.
    assert tension.count == n_gp
    # And the locations are real ones, not a default fill.
    assert corner.element_index.size == corner.count
    assert set(corner.element_index.tolist()) <= {etag}
