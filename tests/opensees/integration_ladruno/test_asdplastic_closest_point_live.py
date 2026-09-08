"""Fork-only — the ADR-97 ``Closest_Point`` / ``Algorithmic`` opt-in pair.

Every case runs in a FRESH subprocess (``_closest_point_cases.py``) with
``stdin=subprocess.DEVNULL``, the same idiom as
``test_asdplastic_live.py``: the fork's ``.pyd`` writes its ``cout`` /
``opserr`` to the process streams, which pytest's ``capfd`` cannot see
but a subprocess pipe can.

**These tests SKIP on a backend older than the ADR-97 closeout.**  The
gate is a capability PROBE (``case_probe`` builds one ``Closest_Point``
material and reports whether the parser took it), never a build-hash
comparison — a bare hash cannot prove ancestry, which is why
``ASDP_CLOSEST_POINT_MIN_BUILD`` is documented and not enforced.  The
skip message names the build that was actually running so a stale
backend is visible in the log rather than mistaken for a pass.

1. The ADR-97 pair converges the same 20-step deviatoric leg the ADR-94
   default converges, to the same committed stress on the same yield
   surface (the fork measured a sub-1 % gap between the two maps).
2. apeGmsh's own ``strict_convergence=True`` default REFUSES that leg
   under ``Closest_Point`` — an absolute-tolerance interaction, not a
   fork defect (ADR 0107).
3. ``material.cp_iterations`` records, reads back as a named Gauss
   component, and steps 0 → 1 as the deck yields.
4. ``tangent_type="Algorithmic"`` without ``Closest_Point`` is refused on
   BOTH sides of the seam: apeGmsh's D2 cross-check before any fork
   process is touched, and the fork's own parser for the same pairing
   pushed through the generic primitive.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apeGmsh.opensees.material.nd import ASDP_CLOSEST_POINT_MIN_BUILD

pytestmark = pytest.mark.ladruno_fork

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_CASES = _HERE / "_closest_point_cases.py"


def _run(case: str) -> tuple[dict, str, str]:
    env = dict(os.environ)
    # ``_closest_point_cases`` imports ``_asdplastic_cases`` as a sibling
    # top-level module, so its own directory must be importable too.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), str(_ROOT), str(_HERE)]
    )
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
    print(f"[closest_point_live/{case}] fork build {build}")
    return result, proc.stdout, proc.stderr


@pytest.fixture(scope="module")
def closest_point_backend() -> str:
    """Skip the module unless the live backend actually knows the token."""
    probe, _, _ = _run("probe")
    build = probe["ladruno_build"]
    if not probe["supported"]:
        pytest.skip(
            f"fork build {build} does not support integration_method "
            f"Closest_Point (needs {ASDP_CLOSEST_POINT_MIN_BUILD} or "
            f"later, fork ADR-97). Parser said: {probe['exception']}"
        )
    return build


# 1 ─────────────────────────────────────────────────────────────────────────

def test_closest_point_converges_the_leg_backward_euler_converges(
    closest_point_backend: str,
) -> None:
    r, _, _ = _run("closest_point_leg")
    cp, be = r["closest_point"], r["backward_euler"]

    assert cp["warnings"] == [], cp["warnings"]
    assert cp["codes"] == [0] * len(cp["codes"]), cp["codes"]
    assert be["codes"] == [0] * len(be["codes"]), be["codes"]

    # Both maps land ON the surface: |f| within the material's own
    # absolute tolerance, or within 1e-6 of the strength scale — a
    # cross-build bound, since 1e-9 pins have failed twice on the fork's
    # Linux CI.
    tol = max(r["f_absolute_tol"], 1e-6 * r["strength_scale"])
    assert max(abs(v) for v in cp["f"]) <= tol, cp["f"]
    assert abs(cp["f"][-1]) <= tol

    # ... and at the SAME state. The two maps are not required to agree
    # bit for bit (ADR-97 D1's byte-identity promise is about
    # Backward_Euler being unchanged by the upgrade, not about the two
    # maps coinciding); the fork measured a sub-1 % gap.
    #
    # Compared PER COMPONENT against that component's own magnitude, not
    # against the strength scale: these stresses are ~3.2e3 while
    # c*cos(phi) is 86.6, so a strength-scale denominator would call an
    # 0.8 % difference a 31 % one. Measured on ff47275fd: sxx and szz
    # agree to ~1e-11 relative; syy differs by 26.7 (0.82 %), which is
    # Backward_Euler's MC_ds corner rounding against Closest_Point's
    # exact corner.
    floor = 1e-6 * max(abs(v) for v in be["last_stress"])
    rel = [
        abs(a - b) / max(abs(a), abs(b), floor)
        for a, b in zip(cp["last_stress"], be["last_stress"])
    ]
    assert max(rel) <= 0.01, (
        f"maps disagree by {max(rel):.3%} on component {rel.index(max(rel))}: "
        f"CP {cp['last_stress']} vs BE {be['last_stress']}"
    )

    # The leg genuinely yields — a purely elastic one would sit well
    # below the surface and prove nothing about the return map. Compared
    # on szz and on sign only: measured on ff47275fd, Closest_Point pins
    # sxx == syy EXACTLY (the true triaxial-compression corner) where
    # Backward_Euler splits them by ~0.8 % through its MC_ds rounding, so
    # an sxx != syy ordering is an artefact of the coarser map, not a
    # property either map must have.
    sxx, syy, szz = cp["last_stress"][:3]
    assert szz < sxx < 0.0 and szz < syy < 0.0, cp["last_stress"]

    # The exact map is the tighter one — that is the whole point of it.
    assert max(abs(v) for v in cp["f"]) < max(abs(v) for v in be["f"])
    print(
        f"[closest_point_live] max|f| CP {max(abs(v) for v in cp['f']):.3e} "
        f"vs BE {max(abs(v) for v in be['f']):.3e}; iters "
        f"CP {sum(cp['iters'])} vs BE {sum(be['iters'])}"
    )


# 1b ────────────────────────────────────────────────────────────────────────

def test_strict_convergence_refuses_the_closest_point_leg(
    closest_point_backend: str,
) -> None:
    """ADR 0107 — apeGmsh's own default is what rejects the better map.

    Not a fork defect: the fork tests a TRIAL state's yield residual
    against the ABSOLUTE f_absolute_tol, and ADR 0105 D2 made that a hard
    refusal by defaulting strict ON. At kPa soil scale 1e-6 absolute is
    ~1.6e-8 relative, tight enough that the MORE accurate map trips it
    where the coarser one happens not to. Pinned so the interaction is
    not rediscovered as a mystery -3.
    """
    r, _, _ = _run("strict_refuses_closest_point")
    assert r["strict_codes"][-1] != 0, r["strict_codes"]
    assert len(r["strict_codes"]) < len(r["relaxed_codes"]), r
    assert r["relaxed_codes"] == [0] * len(r["relaxed_codes"]), r
    # The relaxed run is not merely permitted, it is ACCURATE — so the
    # refusal is about the tolerance's units, not about a bad answer.
    assert r["relaxed_max_f"] < 1e-9, r["relaxed_max_f"]


# 1c ────────────────────────────────────────────────────────────────────────

def test_cp_iterations_records_and_reads_back(
    closest_point_backend: str,
) -> None:
    """ADR 0107 D5 — the material.cp_iterations bucket, end to end.

    Recorder token → .ladruno bucket → reader table → a named Gauss
    component, with the 0-while-elastic / 1-once-yielded transition that
    makes the number mean something.
    """
    r, _, _ = _run("cp_iterations")
    assert r["codes"] == [0] * len(r["codes"]), r["codes"]
    assert r["warnings"] == [], r["warnings"]

    # It reaches available_components() and nothing was dropped.
    assert "cp_iterations" in r["available"], r["available"]
    assert not [w for w in r["read_warnings"] if "Dropped" in w], r["read_warnings"]

    # 0 while the point is elastic, 1 once Mohr-Coulomb yields.
    per_step = r["per_step"]
    assert per_step[0] == [0.0], per_step
    assert per_step[-1] == [1.0], per_step
    assert r["f_last"] is not None and abs(r["f_last"]) < 1e-6, r["f_last"]


# 2 ─────────────────────────────────────────────────────────────────────────

def test_algorithmic_without_closest_point_is_refused_on_both_sides(
    closest_point_backend: str,
) -> None:
    r, out, err = _run("algorithmic_without_cp_refused")

    # Client side: apeGmsh's D2 cross-check, before any fork process.
    assert r["client_message"] is not None, "apeGmsh accepted the pairing"
    assert "ADR-97 D2" in r["client_message"], r["client_message"]
    assert "Closest_Point" in r["client_message"]

    # Server side: the fork's own parser, whose text is what the user
    # sees for everything apeGmsh deliberately does NOT reimplement.
    assert r["server_refused"] is True, r
    text = out + "\n" + err
    # Assert on the PAIRING citation, not merely on the token: a
    # pre-ADR-97 parser also refuses "Algorithmic", but as an unknown
    # tangent_type, which would let this test pass for the wrong reason.
    assert "ADR-97 D2" in text, text[-3000:]
    assert "Closest_Point" in text, text[-3000:]
