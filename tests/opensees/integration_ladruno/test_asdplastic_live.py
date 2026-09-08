"""Fork-only — the ASDPlasticMaterial3D deck contract after fork ADR-94
(ADR 0105 D7, the acceptance gate).

Every case runs in a FRESH subprocess (``_asdplastic_cases.py``) with
``stdin=subprocess.DEVNULL``: the fork's ``.pyd`` writes its ``cout`` /
``opserr`` to the process streams, which pytest's ``capfd`` cannot see but
a subprocess pipe can — both are recorded fork quirks (``LEDGER_quirks.md``).
The child prints the engine's build stamp (``ops.ladrunoBuild()``) so a
stale backend is caught in the log, and one ``RESULT {json}`` line the
parent asserts on.

Float pins compared across builds use bounds of global-Newton-tolerance
size (>= 1e-6 relative); 1e-9 pins failed twice on the fork's Linux CI.

1. ``MohrCoulombSoil`` (D1 schema) is accepted by the ADR-94 parser and a
   20-step deviatoric leg on ``LadrunoBrick`` commits admissible states
   (``|f_MC| <= f_absolute_tol`` recomputed in numpy from the committed
   stress).
2. The pre-D1 superset deck (a fixture string) is refused naming the
   first foreign parameter (``AF_cr``).
3. A deck missing ``MC_phi`` is refused naming ``MC_phi``.
4. ``strict_convergence=True`` refuses the step on ``LadrunoBrick`` and is
   swallowed on ``stdBrick`` — D4's rationale; the D4 gate fires on the
   stdBrick deck at build time.
5. ``HoekBrownRock`` uniaxial tension plateaus at ``s*sigci/mb`` (245.0 kPa
   for sigci 50 MPa, mi 10, GSI 60, D 0) within 5 %.
6. ``f_relative_tol`` on: the same MC problem completes in kPa and x1e9.
7. Newton iterations with ``Continuum`` <= with ``Secant`` on the fork's
   two-cube heterogeneous model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ladruno_fork

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_CASES = _HERE / "_asdplastic_cases.py"

#: Minimum fork build for the ADR-94 contract (ADR 0105); the battery was
#: accepted against ``3622d6214`` (ADR-94 closeout, ASDP-equivalent).
_MIN_BUILD = "bbf657d49"


def _run(case: str) -> tuple[dict, str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT / "src"), str(_ROOT)])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, str(_CASES), case],
        cwd=str(_ROOT), env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900, check=False,
    )
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    assert proc.returncode == 0, (
        f"case {case!r} exited {proc.returncode}\n--- stdout ---\n{tail}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
    result_line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
        None,
    )
    assert result_line is not None, f"no RESULT line\n{tail}"
    result = json.loads(result_line[len("RESULT "):])
    build = result.get("ladruno_build")
    assert build, "ops.ladrunoBuild() returned nothing — not a fork build?"
    print(f"[asdplastic_live/{case}] fork build {build}")
    return result, proc.stdout, proc.stderr


def _refusal_text(stdout: str, stderr: str) -> str:
    return stdout + "\n" + stderr


# 1 ─────────────────────────────────────────────────────────────────────────

def test_schema_deck_accepted_and_admissible_on_ladruno_brick() -> None:
    r, _, _ = _run("schema_accepted")
    assert r["codes"] == [0] * 20, r["codes"]
    assert r["warnings"] == [], r["warnings"]
    # Admissibility on the committed stress: |f| within the material's own
    # absolute tolerance — and, for a cross-build bound, within 1e-6 of the
    # yield function's strength scale c*cos(phi).
    tol = max(r["f_absolute_tol"], 1e-6 * r["strength_scale"])
    f = r["f"]
    assert max(abs(v) for v in f) <= tol, f
    # The leg actually yields: the last states sit ON the surface (a purely
    # elastic leg would sit well below it).
    assert abs(f[-1]) <= tol
    assert all(abs(v) <= tol for v in f[-5:])
    sxx, syy, szz = r["last_stress"][:3]
    assert szz < syy < sxx < 0.0                 # compression, distinct


# 2 / 3 ─────────────────────────────────────────────────────────────────────

def test_pre_adr94_superset_deck_is_refused_naming_the_first_foreign_name() -> None:
    r, out, err = _run("superset_refused")
    assert r["refused"] is True, r
    text = _refusal_text(out, err)
    assert "unknown model parameter 'AF_cr'" in text, text[-3000:]
    assert "REJECTED" in text and "ADR-94" in text


def test_missing_phi_is_refused_naming_it() -> None:
    r, out, err = _run("missing_phi_refused")
    assert r["refused"] is True, r
    text = _refusal_text(out, err)
    assert "required model parameter(s) were never given a value" in text
    assert "MC_phi" in text.split("never given a value")[1].splitlines()[0]


# 4 ─────────────────────────────────────────────────────────────────────────

def test_strict_convergence_refuses_on_ladruno_brick_and_is_swallowed_on_std_brick() -> None:
    r, _, err = _run("strict_hosts")
    lb, sb = r["LadrunoBrick"], r["stdBrick"]
    # LadrunoBrick: the very first step is refused (no step commits).
    assert lb["codes"] and all(c != 0 for c in lb["codes"]), lb["codes"]
    assert "REFUSED the trial strain" in err
    # stdBrick: 20/20 "successes" on the identical deck — the swallow.
    assert sb["codes"] == [0] * 20, sb["codes"]
    # ...which is exactly what the D4 gate says at build time, and only
    # on the stdBrick deck.
    assert lb["warnings"] == []
    assert len(sb["warnings"]) == 1 and "ASDPlasticHostWarning" in sb["warnings"][0]
    assert "fork ADR-94 B2" in sb["warnings"][0]


# 5 ─────────────────────────────────────────────────────────────────────────

def test_hoek_brown_uniaxial_tension_plateaus_at_textbook_strength() -> None:
    r, _, _ = _run("hb_tension")
    sigma_t = r["sigma_t"]
    assert sigma_t == pytest.approx(245.0, rel=1e-3)      # the wp/94d number
    # strict off (the fork's rig): the last committed state IS the plateau.
    off = r["strict_0"]
    assert off["n_committed"] >= 1
    assert off["sigma_xx"][-1] == pytest.approx(sigma_t, rel=5e-2), off["sigma_xx"][-1]
    # strict on (the ADR 0105 default): the corner step is REFUSED, the last
    # committed state is the last elastic one, and nothing above sigma_t is
    # ever committed.
    on = r["strict_1"]
    assert on["codes"][-1] != 0
    assert on["n_committed"] == off["n_committed"] - 1
    assert max(on["sigma_xx"]) < sigma_t
    assert on["sigma_xx"] == pytest.approx(off["sigma_xx"][:-1], rel=1e-6)


# 6 ─────────────────────────────────────────────────────────────────────────

def test_f_relative_tol_makes_the_same_problem_complete_in_kpa_and_x1e9() -> None:
    r, _, _ = _run("relative_tol_units")
    assert r["kpa"]["codes"] == [0] * 20, r["kpa"]["codes"]
    assert r["x1e9"]["codes"] == [0] * 20, r["x1e9"]["codes"]
    a = [v * r["unit_gap"] for v in r["kpa"]["last_stress"]]
    b = r["x1e9"]["last_stress"]
    scale = max(abs(v) for v in b)
    assert max(abs(x - y) for x, y in zip(a, b)) <= 1e-6 * scale, (a, b)


# 7 ─────────────────────────────────────────────────────────────────────────

def test_continuum_tangent_costs_no_more_iterations_than_secant() -> None:
    r, _, _ = _run("continuum_vs_secant")
    cont, sec = r["Continuum"], r["Secant"]
    assert cont["codes"] == [0] * 4 and sec["codes"] == [0] * 4
    print(f"[asdplastic_live/continuum_vs_secant] iterations "
          f"Continuum={cont['iterations']} Secant={sec['iterations']}")
    assert cont["iterations"] <= sec["iterations"]
    # Same answer at convergence, tangent-independent.
    for tag, s_cont in cont["stresses"].items():
        s_sec = sec["stresses"][tag]
        scale = max(abs(v) for v in s_sec)
        assert max(abs(x - y) for x, y in zip(s_cont, s_sec)) <= 1e-5 * scale
