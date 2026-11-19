"""The strut-and-tie FE overlay consumer (apeConcrete ADR-0014 §6).

Fixtures are ``apeConcrete.stm.model_to_dict`` output for the corbel
and four-pile-cap builders; the corbel outline was extended with a
column stub so the far column face is a physical boundary.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.interop import strut_tie as _strut_tie
from apeGmsh.interop.strut_tie import (
    EQUILIBRIUM_TOLERANCE_0_005,
    FALLBACK_TOLERANCE_FACTOR_10,
    PushoverRefusedError,
    _parse_fixed_plane,
    main,
    strut_tie_overlays,
    strut_tie_pushover,
    write_strut_tie_overlays,
)

pytestmark = pytest.mark.live

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_corbel_plane_stress_overlay_balances_the_load() -> None:
    model = _load("corbel.stm.json")
    result = strut_tie_overlays(
        model, mesh_size=40.0, extra_fixed_planes=(("x", -400.0, (0, 1)),)
    )
    segs = result.overlays["fe_trajectories"]["segments"]
    assert len(segs) > 50
    assert result.n_elements > 100
    # Plane model: every glyph lies in the model plane.
    assert all(abs(s[0][2]) < 1e-9 and abs(s[1][2]) < 1e-9 for s in segs)
    # Compression glyphs exist (the inclined strut) and tension glyphs exist (the tie).
    assert any(s[3] < 0.0 for s in segs)
    assert any(s[2] > 0.0 for s in segs)
    # Equilibrium: reactions balance the applied load.
    applied, reactions = np.array(result.applied), np.array(result.reactions)
    assert math.isclose(applied[1], -300.0e3)
    assert result.overlays["fe_summary"]["backend"] in (
        "ladruno-fork",
        "stock-openseespy",
    )
    assert np.allclose(reactions[:2], -applied[:2], rtol=1e-3, atol=1.0)
    json.dumps(result.overlays, allow_nan=False)


def test_corbel_in_an_offset_plane_solves_like_z0() -> None:
    """#1337: the plane region meshes at the STM's z and runs ndm=2.

    A plane at z = 1 drops a CONSTANT z, which is lossless, so the bridge
    must not refuse it, and the answer must match the z = 0 corbel.
    """
    flat = _load("corbel.stm.json")
    lifted = _load("corbel.stm.json")
    for n in lifted["nodes"]:
        n["xyz"][2] += 1.0
    for p in lifted.get("plates", []):
        p["center"][2] += 1.0
    kw = {"mesh_size": 40.0, "extra_fixed_planes": (("x", -400.0, (0, 1)),)}
    r0 = strut_tie_overlays(flat, **kw)
    r1 = strut_tie_overlays(lifted, **kw)
    assert r1.n_elements == r0.n_elements
    assert np.allclose(r1.reactions, r0.reactions, rtol=1e-9, atol=1e-6)
    segs = r1.overlays["fe_trajectories"]["segments"]
    assert segs and all(s[0][2] == 1.0 and s[1][2] == 1.0 for s in segs)


def test_pile_cap_solid_overlay(tmp_path: Path) -> None:
    out = write_strut_tie_overlays(
        FIXTURES / "pile_cap.stm.json", tmp_path / "cap.overlays.json", mesh_size=180.0
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    summary = data["fe_summary"]
    assert summary["n_elements"] > 200
    # The column load goes down into the pile supports; pile loads were skipped.
    assert math.isclose(summary["applied"][2], -2.0e6)
    assert math.isclose(summary["reactions"][2], 2.0e6, rel_tol=1e-3)
    segs = data["fe_trajectories"]["segments"]
    assert segs
    zs = [s[0][2] for s in segs]
    assert min(zs) < 300.0 < max(zs)  # glyphs through the depth, not one plane


def test_schema_and_case_are_checked() -> None:
    model = _load("corbel.stm.json")
    bad = dict(model)
    bad["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        strut_tie_overlays(bad)
    with pytest.raises(StopIteration):
        strut_tie_overlays(model, case="nope", mesh_size=60.0)


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_corbel_pushover_curve_rises_past_cracking_in_equilibrium(
    record_property: Any,
) -> None:
    """The reinforced corbel's curve rises past the plain-concrete cracking
    peak (about 136 kN on this mesh without bars) to more than 150 kN, with
    every accepted step in equilibrium.

    At ``ladruno-v20261029`` and before, ``LadrunoConcrete3D`` returned 0
    from a failed return map, so a step past cracking was accepted out of
    equilibrium and the equilibrium guard refused it (#2602). From
    ``ladruno-v20261030`` (fork PR #1038) it returns a negative code, Newton
    cuts the step, and the run completes in equilibrium (#2622)."""
    result = strut_tie_pushover(
        _load("corbel.stm.json"),
        mesh_size=45.0,
        extra_fixed_planes=(("x", -400.0, (0, 1)),),
        target_displacement=0.3,
        steps=10,
    )
    curve = result.overlays["fe_curve"]
    pts = curve["points"]
    record_property("capacity_N", curve["capacity"])
    record_property("stopped", curve["stopped"])
    record_property("n_points", len(pts))
    record_property(
        "max_relative_unbalance", curve["equilibrium"]["max_relative_unbalance"]
    )
    assert len(pts) >= 4
    assert pts[0] == [0.0, 0.0]
    assert all(b[0] > a[0] for a, b in itertools.pairwise(pts))
    assert curve["capacity"] == max(p[1] for p in pts)
    # Reinforced: the tie bars carry the region well past the plain-concrete
    # cracking peak.
    assert curve["capacity"] > 150.0e3
    assert "ties as Steel02" in curve["source"]
    assert result.overlays["fe_summary"]["ties_modelled"]["T1"] > 0.0
    assert curve["stopped"] in ("target", "post_peak")
    assert curve["equilibrium"]["max_relative_unbalance"] <= 5e-3
    assert pts[1][1] > 0.0  # the elastic first step rises
    json.dumps(result.overlays, allow_nan=False)


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_pile_cap_pushover_writes_both_overlays(tmp_path: Path) -> None:
    """Plumbing: both overlays and the bearing paths reach one file. One
    linear-range step (0.05 mm, λ ≈ 0.6), because past it the pinned fork's
    return map leaves the solid region out of equilibrium and the run
    refuses (#2602)."""
    out = write_strut_tie_overlays(
        FIXTURES / "pile_cap.stm.json",
        tmp_path / "cap.overlays.json",
        mesh_size=300.0,
        pushover=True,
        target_displacement=0.05,
        steps=1,
        max_halvings=3,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "fe_trajectories" in data
    assert data["fe_curve"]["capacity"] > 0.0
    assert "fe_stress_paths" in data  # the bearing paths travel with the file
    assert len(data["fe_summary_nonlinear"]["ties_modelled"]) == 4
    assert data["fe_summary_nonlinear"]["Gc"] > data["fe_summary_nonlinear"]["Gf"]


# ---------------------------------------------------------------------------
# The physics oracle of apeConcrete ADR-0014 §9: a published corbel with
# a measured failure load
# ---------------------------------------------------------------------------
COOK_MITCHELL_STM_PREDICTION_N = 471.0e3  # per side, tie yield (SP-208 Part 3 §2)
COOK_MITCHELL_MEASURED_N = 502.0e3  # per side


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_cook_mitchell_double_corbel_pushover_against_the_paper(
    record_property: Any,
) -> None:
    """ACI SP-208 Part 3 §2, Cook and Mitchell (1988) double corbel:
    strut-and-tie prediction 471 kN per side (tie yield, 4 No. 15 at
    444 MPa), measured 502 kN. The fixture is the apeConcrete anchor's
    ``model_to_dict`` output (``tests/golden/test_sp208_cook_mitchell_
    corbel_anchor.py`` there); the load case is V = 471 kN per side with
    H = 0.2·V outward, so ``capacity_factor`` reads directly as
    FE peak / strut-and-tie prediction.

    State of the bracket on 2026-09-18 (40 mm Tri31 mesh, 718 elements,
    KrylovNewton/ModifiedNewton fallbacks): the FE peaks at ≈369 kN per
    side, 0.78 of the prediction, then softens; with H removed it peaks
    at ≈440 kN (0.93). The lower-bound ordering FE ≥ STM is therefore
    **not met yet**, and the diagnosis is the load introduction, not the
    concrete: in the specimen the plates were welded to the bars, so V
    and H went straight into the tie, while here the bars end at the
    load node and H pulls on the concrete under a 50 × 300 plate. Column
    bars and the two No. 10 ties are absent too. This test pins the
    current state with a wide bracket so a change in either direction —
    a welded-plate option lifting the peak, or a regression lowering
    it — is noticed and re-documented.

    **#2602 (2026-10-29): the run refuses at the pinned fork.** The 287–385
    kN peaks this test saw were not capacities. The #2602 diagnosis traced
    them to a path that branches at the first step plain Newton cannot
    converge, because ``LadrunoConcrete3D`` returns 0 from a failed return
    map; a 1e-12 change in ft moves the peak from 318 to 414 kN on one MKL
    path. The equilibrium guard refuses at that step, step 2, the first one
    a fallback accepts: its unbalanced force is 1.3 % of the load, against
    0.5 % (live-fork run 40826555866). The equilibrated part of the curve
    ends there, below the floor. The bracket is unchanged and waits for an
    equilibrated run: 0.7 × 471 kN (the strut-and-tie prediction, the
    floor) < per side < 1.1 × 502 kN (the measured load), with
    ``stopped == "post_peak"``. Chain F (#2602 route 2) restores it.

    **#2622 (FORK_PIN ladruno-v20261030): the fork refuses the failed
    return map, and the run still refuses, later.** Fork PR #1038 (refusal
    rule B of the amended #2602 ruling) makes ``LadrunoConcrete3D`` return
    a negative code when its last plane-stress pass fails or σ22 stays
    unconverged, so Newton cuts the step instead of accepting a diverged
    state. The run gets past step 2 and then refuses on every MKL path the
    #2602 matrix measured: at steps 65–121, with ``"unbalanced"`` (MKL_CBWR
    unset on AMD) or ``"no_convergence"`` (AVX2, AVX512, COMPATIBLE). The
    step and the reason vary with the MKL path, so neither is pinned; the
    test asserts that the run refuses past the #2602 branch point with one
    of the two reasons and that the error carries the equilibrated curve.
    The return-map robustness behind the refusal, the 2 % MKL agreement and
    the paper bracket are #2621.
    """
    model = _load("cook_mitchell_corbel.stm.json")
    with pytest.raises(PushoverRefusedError) as info:
        strut_tie_pushover(
            model, mesh_size=40.0, target_displacement=3.0, steps=30, max_halvings=5
        )
    err = info.value
    record_property("refused_step", err.step)
    record_property("refusal", str(err))
    record_property("reason", err.reason)
    # The pin before #1038 refused here, at step 2 (live-fork 40826555866).
    assert err.step > 2
    assert err.relative_tolerance == 5e-3
    if err.reason == "unbalanced":
        # An accepted state out of equilibrium: its unbalanced force is kept.
        assert err.unbalance is not None and math.isfinite(err.unbalance)
        assert err.unbalance > 0.0
    elif err.reason == "no_convergence":
        # OpenSees reverted the unconverged step, so there is no state.
        assert err.unbalance is None
    else:
        pytest.fail(f"unknown refusal reason {err.reason!r}")
    assert f"step {err.step} " in str(err)
    # The curve: the origin plus every step before the refused one, each in
    # equilibrium, with the control displacement increasing.
    assert len(err.points) == err.step
    assert err.points[0] == [0.0, 0.0]
    assert all(math.isfinite(d) and math.isfinite(p) for d, p in err.points)
    assert all(b[0] > a[0] for a, b in itertools.pairwise(err.points))
    equilibrated_per_side = max(p[1] for p in err.points) / 2.0
    record_property("equilibrated_per_side_N", equilibrated_per_side)
    assert equilibrated_per_side > 0.0


# ---------------------------------------------------------------------------
# The equilibrium guard (#2602): a step out of equilibrium, or a run that
# does not converge, returns no capacity. These run on stock openseespy
# (ASDConcrete3D in its linear range), so the live-stock lane proves them.
#
# Each case runs in a fresh process. Stock ASDConcrete3D keeps a process-wide
# hardening law per material tag that ``wipe`` does not clear (#2044; fixed
# in the fork only). A pushover on tag 1 in the shared pytest process would
# hand its law to every later tag-1 ASDConcrete3D test, and the other way
# round: on this PR's first live-stock run, test_asdconcrete_fc_golden.py
# read this corbel's Te.
# ---------------------------------------------------------------------------
_ELASTIC_ASD = {
    "mesh_size": 45.0,
    "extra_fixed_planes": (("x", -400.0, (0, 1)),),
    "target_displacement": 0.03,
    "steps": 3,
    "material": "asd",
}


class _InjectingOps:
    """The live module with a known defect from accepted step ``from_step``
    on: ``force`` newtons added to ``nodeReaction`` at one free DOF (an
    unbalanced force of exactly that size), or ``analyze`` failing (a run
    that stops converging; OpenSees reverts a failed step, so the domain
    stays at the last accepted one). ``fail_calls`` fails those 1-based
    ``analyze`` calls only, which forces a halving."""

    def __init__(
        self,
        real: object,
        *,
        from_step: int = 1,
        force: float = 0.0,
        fail: bool = False,
        fail_calls: frozenset[int] = frozenset(),
    ) -> None:
        self._real = real
        self._from_step = from_step
        self._force = force
        self._fail = fail
        self._fail_calls = fail_calls
        self.calls = 0
        self.accepted = 0
        self.node: int | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)

    def analyze(self, *args: object) -> int:
        self.calls += 1
        if self.calls in self._fail_calls:
            return -1
        if self._fail and self.accepted + 1 >= self._from_step:
            return -1
        rc = int(self._real.analyze(*args))
        self.accepted += rc == 0
        return rc

    def nodeReaction(self, tag: int, *args: object) -> list[float]:
        r = [float(x) for x in self._real.nodeReaction(tag, *args)]
        if self.node is None:
            fixed = {int(n) for n in self._real.getFixedNodes()}
            self.node = min(
                int(n) for n in self._real.getNodeTags() if int(n) not in fixed
            )
        if self.accepted >= self._from_step and int(tag) == self.node:
            r[0] += self._force
        return r


def _scenario(model_json: str, inject: dict, run: dict) -> dict:
    """The child process's body: one pushover, with ``inject`` passed to
    :class:`_InjectingOps` (none when empty), and its outcome as JSON data."""
    import json

    import apeGmsh.interop.strut_tie as st

    holder: dict = {}
    if inject:
        if "fail_calls" in inject:
            inject = dict(inject, fail_calls=frozenset(inject["fail_calls"]))

        class _Emitter(st.LiveOpsEmitter):
            @property
            def ops(self):  # type: ignore[override]
                if "ops" not in holder:
                    holder["ops"] = _InjectingOps(super().ops, **inject)
                return holder["ops"]

        st.LiveOpsEmitter = _Emitter  # type: ignore[misc]
    with open(model_json, encoding="utf-8") as fh:
        model = json.load(fh)
    accepted = holder.get
    try:
        result = st.strut_tie_pushover(model, **run)
    except st.PushoverRefusedError as err:
        return {
            "refused": {
                "reason": err.reason,
                "step": err.step,
                "unbalance": err.unbalance,
                "relative_tolerance": err.relative_tolerance,
                "points": err.points,
                "message": str(err),
            },
            "accepted": accepted("ops").accepted if holder else None,
        }
    return {
        "curve": result.overlays["fe_curve"],
        "applied": list(result.applied),
        "accepted": accepted("ops").accepted if holder else None,
    }


def _fresh(tmp_path: Path, inject: dict | None = None, **overrides: object) -> dict:
    """Run :func:`_scenario` on the corbel in a fresh interpreter."""
    run = dict(_ELASTIC_ASD, **overrides)
    out = tmp_path / "out.json"
    script = tmp_path / "run.py"
    model_json = str(FIXTURES / "corbel.stm.json")
    script.write_text(
        "import json\n\n"
        + inspect.getsource(_InjectingOps)
        + "\n\n"
        + inspect.getsource(_scenario)
        + "\n\n"
        + f"res = _scenario({model_json!r}, {inject or {}!r}, {run!r})\n"
        + f"open({str(out)!r}, 'w', encoding='utf-8').write(json.dumps(res))\n",
        encoding="utf-8",
    )
    src = str(Path(_strut_tie.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([src, *filter(None, [env.get("PYTHONPATH")])])
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_pushover_guard_passes_a_converged_linear_range_run(tmp_path: Path) -> None:
    """A linear-range run under plain Newton: the guard reads an unbalance
    thousands of times below its tolerance and returns the capacity."""
    curve = _fresh(tmp_path)["curve"]
    assert curve["stopped"] == "target"
    assert len(curve["points"]) == 4
    assert curve["capacity"] == max(p[1] for p in curve["points"]) > 0.0
    eq = curve["equilibrium"]
    assert eq["tolerance"] == EQUILIBRIUM_TOLERANCE_0_005 == 5e-3
    assert curve["fallback_steps"] == 0
    assert 0.0 <= eq["max_relative_unbalance"] < 1e-6


def test_pushover_measures_the_unbalance_it_is_given(tmp_path: Path) -> None:
    """An injected 10 N below the tolerance passes, and the ratio recorded is
    10 N over the smallest applied load ``|λ|·|P|``, that of step 1."""
    res = _fresh(tmp_path, {"from_step": 1, "force": 10.0})
    curve = res["curve"]
    reference = float(np.linalg.norm(res["applied"][:2]))
    lam1 = curve["load_factors"][1]
    assert curve["equilibrium"]["max_relative_unbalance"] == pytest.approx(
        10.0 / (lam1 * reference), rel=1e-3
    )


def test_pushover_refuses_a_step_out_of_equilibrium(tmp_path: Path) -> None:
    """A step accepted with an unbalanced force above 0.5 % of the applied
    load refuses the run: no result object, an error naming the step, the
    force and the tolerance, and the curve up to the step before."""
    force = 5.0e4
    err = _fresh(tmp_path, {"from_step": 2, "force": force})["refused"]
    assert err["reason"] == "unbalanced"
    assert err["step"] == 2
    assert err["unbalance"] == pytest.approx(force, rel=1e-6)
    assert err["relative_tolerance"] == 5e-3
    assert len(err["points"]) == 2 and err["points"][0] == [0.0, 0.0]
    assert err["points"][1][1] > 0.0
    msg = err["message"]
    assert "step 2" in msg and "tolerance 0.005" in msg and "5e+04 N" in msg
    assert "#2602" in msg


def test_pushover_refuses_a_fallback_accepted_step_out_of_equilibrium(
    tmp_path: Path,
) -> None:
    """Newton fails on step 2 (``analyze`` call 2) and KrylovNewton, the first
    fallback, converges (call 3), so the step that carries the injected force
    was accepted by a fallback: the path the loosened tolerance opened."""
    res = _fresh(tmp_path, {"from_step": 2, "force": 5.0e4, "fail_calls": [2]})
    err = res["refused"]
    assert res["accepted"] == 2
    assert err["reason"] == "unbalanced"
    assert err["step"] == 2
    assert err["unbalance"] == pytest.approx(5.0e4, rel=1e-6)
    assert "accepted by a fallback algorithm" in err["message"]


def test_pushover_defaults_finish_the_stock_asd_corbel(tmp_path: Path) -> None:
    """#2612 review: at the old fallback factor of 100 this run (defaults,
    0.3 mm in 10 steps) was refused at step 5, 0.536 % out of equilibrium. At
    the default of 10 it reaches the target in equilibrium."""
    res = _fresh(tmp_path, target_displacement=0.3, steps=10)
    curve = res["curve"]
    assert curve["stopped"] == "target"
    assert curve["fallback_steps"] > 0  # the fallbacks carry it, not Newton
    assert curve["equilibrium"]["max_relative_unbalance"] < EQUILIBRIUM_TOLERANCE_0_005
    assert curve["capacity"] > 0.0
    assert FALLBACK_TOLERANCE_FACTOR_10 == 10.0


def test_pushover_refuses_at_step_one_with_only_the_origin(tmp_path: Path) -> None:
    """The degenerate end: the first step is refused, and the curve carried
    is the origin alone."""
    err = _fresh(tmp_path, {"from_step": 1, "force": 5.0e4})["refused"]
    assert err["step"] == 1
    assert err["points"] == [[0.0, 0.0]]


def test_pushover_refuses_a_run_that_stops_converging(tmp_path: Path) -> None:
    """Before #2602 a run whose step failed every halving returned the peak
    of the steps before as ``capacity`` with ``stopped="divergence"``."""
    res = _fresh(tmp_path, {"from_step": 3, "fail": True}, max_halvings=2)
    err = res["refused"]
    assert err["reason"] == "no_convergence"
    assert err["step"] == 3
    assert err["unbalance"] is None
    assert len(err["points"]) == 3
    assert res["accepted"] == 2
    assert "step 3 did not converge" in err["message"]
    assert "2 halvings" in err["message"]


def test_pushover_stops_at_the_target_after_halvings(tmp_path: Path) -> None:
    """Two forced halvings (steps 2 and 3, each failing Newton and the three
    fallbacks once) leave increments of 0.01, 0.005, 0.0025, 0.0025, 0.005;
    growing back to 0.01 the last step would end at 0.035 mm, past the
    0.03 mm target. It stops at the target instead."""
    res = _fresh(tmp_path, {"fail_calls": [2, 3, 4, 5, 7, 8, 9, 10]})
    curve = res["curve"]
    assert res["accepted"] == len(curve["points"]) - 1 == 6
    assert curve["stopped"] == "target"
    assert curve["control"]["reached"] == pytest.approx(0.03, rel=1e-9)
    assert curve["reached_fraction"] == pytest.approx(1.0, rel=1e-9)
    assert curve["points"][-1][0] == pytest.approx(0.03, rel=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1e-3, float("nan")])
def test_pushover_equilibrium_tolerance_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="equilibrium_tolerance must be > 0"):
        strut_tie_pushover(_load("corbel.stm.json"), equilibrium_tolerance=bad)


# ---------------------------------------------------------------------------
# The command line (what the office web app launches in the OpenSees venv)
# ---------------------------------------------------------------------------
def test_fixed_plane_syntax() -> None:
    assert _parse_fixed_plane("x=-400:01") == ("x", -400.0, (0, 1))
    assert _parse_fixed_plane("z=0:2") == ("z", 0.0, (2,))
    with pytest.raises(ValueError, match="AXIS=VALUE:DOFS"):
        _parse_fixed_plane("nonsense")


def test_cli_writes_the_linear_overlays(tmp_path: Path) -> None:
    out = tmp_path / "corbel.overlays.json"
    rc = main(
        [
            str(FIXTURES / "corbel.stm.json"),
            str(out),
            "--mesh-size",
            "60",
            "--fix-plane",
            "x=-400:01",
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["fe_trajectories"]["segments"]
    assert "fe_curve" not in data
    assert math.isclose(data["fe_summary"]["applied"][1], -300.0e3)


def test_welded_plates_add_plate_chains_and_welds() -> None:
    """``weld_plates`` places each loaded plate as a stiff bar chain a quarter
    mesh size inside the concrete, welded to the tie node, and routes the
    tangential load into the tie node. Pushover numbers on the Cook and
    Mitchell corbel (40 mm, 2026-09-19): peak 412 kN per side at the 3 mm
    target, 0.87 × the prediction, against 369 kN without the weld — closer,
    still below, with the two No. 10 ties and the column bars still absent."""
    from apeGmsh.interop.strut_tie import _build

    model = _load("cook_mitchell_corbel.stm.json")
    fe = _build(
        model,
        case=None,
        mesh_size=60.0,
        extra_fixed_planes=(),
        verbose=False,
        reinforced=True,
        weld_plates=True,
    )
    assert set(fe.welds) == {"plate_L1", "weld_L1", "plate_L2", "weld_L2"}
    assert fe.welds["plate_L1"] == pytest.approx(25.0 * 300.0)
    assert fe.welds["weld_L1"] == pytest.approx(800.0)  # the tie's bars
    # the outward H goes to the tie node, the vertical V to the plate rows
    horizontal = [r for r, f in fe.nodal.items() if abs(f[0]) > 0.0]
    assert len(horizontal) == 2
    assert math.isclose(fe.applied[1], -2 * 471.0e3)
    with pytest.raises(ValueError, match="weld_plates needs"):
        _build(
            model,
            case=None,
            mesh_size=60.0,
            extra_fixed_planes=(),
            verbose=False,
            reinforced=False,
            weld_plates=True,
        )


def test_extra_bars_are_arranged_and_placed() -> None:
    """Hoops and column bars outside the strut-and-tie model, split where
    they cross the tie, the welds and each other, so gmsh can embed them.

    Cook and Mitchell corbel, 40 mm, welded plates + 2 hoops + 2 column
    bar lines (2026-09-19): implicit integration peaks at 410 kN per side
    (0.87 × the 471 kN prediction) and stops at 0.65 mm; IMPL-EX reaches a
    plateau of 683 kN at 0.1 mm steps and 614 kN at 0.025 mm steps (1.30 ×,
    1.22 × the measured 502 kN). The measurement sits inside that bracket:
    implicit under (premature numerical failure), IMPL-EX over (delayed
    damage by extrapolation)."""
    from apeGmsh.interop.strut_tie import _arrange, _build

    pts, segs = _arrange(
        {
            "a": [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])],
            "b": [np.array([5.0, -5.0, 0.0]), np.array([5.0, 5.0, 0.0])],
            "c": [np.array([10.0, 0.0, 0.0]), np.array([10.0, 5.0, 0.0])],
        },
        tol=1e-6,
    )
    assert len(pts) == 6  # the crossing and the shared end are single points
    assert {k: len(v) for k, v in segs.items()} == {"a": 2, "b": 2, "c": 1}

    model = _load("cook_mitchell_corbel.stm.json")
    bars = _load("cook_mitchell_bars.json")
    fe = _build(
        model,
        case=None,
        mesh_size=60.0,
        extra_fixed_planes=(),
        verbose=False,
        reinforced=True,
        weld_plates=True,
        extra_bars=bars,
    )
    assert set(fe.bars) == {"hoop1", "hoop2", "col_left", "col_right"}
    assert fe.bars["col_left"] == pytest.approx(600.0)
    assert set(fe.welds) == {"plate_L1", "weld_L1", "plate_L2", "weld_L2"}


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_asd_material_rises_past_the_prediction_with_a_bearing_path() -> None:
    """The second concrete model, stock ``ASDConcrete3D``, on the welded and
    reinforced Cook and Mitchell corbel. On 2026-09-19 it reached a plateau
    of about 600 kN per side (601 kN, 1.28 × the prediction, 1.20 × the
    measured 502 kN), where ``LadrunoConcrete3D``'s implicit return map
    stalls at 410 kN. The bearing-element stress path recorded here is the
    data a fork investigation of the Ladruno return map starts from.

    #2612: the run stops at 0.9 mm, on the rising branch, because the
    plateau past about 1 mm is not reached in equilibrium on every CI
    runner, at any fallback factor tried. Every step there is
    fallback-accepted, and the state each fallback lands on depends on the
    runner's path:

    - factor 1: no convergence at 1.1 mm (live-fork 40827694053);
    - factor 10, the default: refused at 1.8 mm, 0.62 % out of equilibrium
      (40836670984), but 2.0 mm reached on the other runner (40836668510);
    - factor 100: 0.1-0.6 % out of equilibrium in local runs, on either side
      of the check.

    Up to 0.9 mm, at the default, the largest ratio is 0.07 % locally, and
    the load is past 1.05 × the prediction (about 560 kN per side)."""
    model = _load("cook_mitchell_corbel.stm.json")
    bars = _load("cook_mitchell_bars.json")
    result = strut_tie_pushover(
        model,
        mesh_size=40.0,
        target_displacement=0.9,
        steps=18,
        max_halvings=4,
        weld_plates=True,
        extra_bars=bars,
        material="asd",
    )
    curve = result.overlays["fe_curve"]
    per_side = curve["capacity"] / 2.0
    assert curve["equilibrium"]["max_relative_unbalance"] < 0.2 * 5e-3
    assert curve["material"] == "asd"
    assert curve["backend"] in ("ladruno-fork", "stock-openseespy")
    assert curve["element"] == "Tri31"
    assert 0.0 < curve["reached_fraction"] <= 1.0 + 1e-9
    assert curve["increment"] <= 0.05 + 1e-12  # max_increment raised the steps
    assert curve["steps"] * curve["increment"] == pytest.approx(0.9)
    assert curve["stopped"] == "target"
    assert result.overlays["fe_summary"]["backend"] == curve["backend"]
    assert "ASDConcrete3D" in curve["source"]
    assert (
        1.05 * COOK_MITCHELL_STM_PREDICTION_N
        < per_side
        < 1.5 * COOK_MITCHELL_STM_PREDICTION_N
    )
    paths = result.overlays["fe_stress_paths"]
    assert set(paths) == {"bearing_L1", "bearing_L2"}
    path = paths["bearing_L1"]["path"]
    assert len(path) >= 10
    assert paths["bearing_L1"]["components"] == ["s11", "s22", "s12"]
    # under the plate: vertical compression that builds up, then the
    # element unloads as it crushes — the path is the record of both
    s22 = [row[3] for row in path]
    assert s22[1] < 0.0
    assert min(s22) < s22[1]
    assert all(
        len(row) == 7 for row in path
    )  # delta, lambda, s11, s22, s12, sigma1, sigma3
    assert min(row[6] for row in path) <= min(
        s22
    )  # sigma3 is the compressive principal
