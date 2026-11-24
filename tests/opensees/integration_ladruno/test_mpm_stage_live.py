"""The fork side of ``mpm.Stage`` (ADR 0134 Amendment 4, slice #3120).

``mpm stage <matTag> <s>`` (``OPS_LadrunoMPM.cpp``, ``LadrunoMPMSoil.cpp::
stageWalk`` at ``ladruno-v20261123b``) reaches every particle whose material
has the tag through the material's own ``materialState`` parameter and hands
it ``s`` unchanged. Each deck here is emitted by ``apeSees.py`` and run on the
pinned fork in a fresh process, like ``test_mpm_gridvel_live.py``.

The model is a 2-D mode-F column on the default (gimp) grid: a weak
DruckerPrager body ``0.4 x 1.0`` at h = 0.1, ppc = 2 (160 particles), its
sides fixed in x, its base fixed, under gravity and local damping. The staged
deck runs ``mpm stage <dp> 0``, ``mpm analyze``, ``mpm stage <dp> 2``,
``mpm analyze``; the control deck is the same model with no ``Stage``. After
each analyze the test reads every particle's material stress
(``mpm particleResponse stress``) and its sigma_zz (``mpm particleData
sigZZ``), lines the test adds to the emitted deck. Closed forms:

* **stage 0 is elastic.** With the sides fixed and plane strain, the column
  is in uniaxial strain, so an elastic particle holds
  ``sigma_xx / sigma_yy = nu / (1 - nu)`` at every instant. That K0 state is
  outside this DruckerPrager's yield surface below a shallow depth: on it
  ``f1 = |sigma_yy| ((1 - K0) sqrt(2/3) - rho (1 + 2 K0)) - sqrt(2/3) sigmaY``,
  positive once ``|sigma_yy|`` passes :data:`SIGYY_YIELD`.
* **stage 2 is elastoplastic.** After the second analyze every particle is
  on or inside the yield surface, ``f1 <= 0`` (the material's own return
  map, so to round-off).
* **the stage rows did it.** DruckerPrager's default stage is 2
  (``mElastFlag = 2`` in ``DruckerPrager.cpp``), so the control deck is
  elastoplastic from its first step: at the end of each analyze every
  particle is inside the surface, and the deep particles hold the cone's
  lateral ratio, well above K0. So the control ends in another state than
  the stage-0 run.

``mpm stage`` answers the number of particles it reached: 160, both times.

The warrant-row test (``tests/opensees/warrant_rows.py``) is
``test_mpm_stage_walks_the_druckerprager_particles_on_fork``.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import apeGmsh
from apeGmsh.opensees import apeSees
from tests.opensees.mpm._meshes import grid_nodes, make_fem, quads

pytestmark = pytest.mark.ladruno_fork

_FORK_IMPORT = (
    "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n")
#: Added to the emitted deck: a dump of the particle state after each
#: analyze, and the count each ``mpm stage`` answers.
_PROBE = '''
import json as _json
_stage_counts = []
def _q(*a):
    out = ops.mpm(*a)
    return [float(v) for v in (out if isinstance(out, (list, tuple)) else [out])]
def _dump(label):
    with open(label + ".json", "w") as f:
        _json.dump({"id": _q("particleData", "id"), "x": _q("particleData", "x"),
                    "sig": _q("particleResponse", "stress"),
                    "szz": _q("particleData", "sigZZ"),
                    "stage_counts": _stage_counts}, f)
'''

H, PPC, RHO, GRAV = 0.1, 2, 2000.0, 9.81
W, HC = 0.4, 1.0
N_PARTICLES = int(round(W / (H / PPC))) * int(round(HC / (H / PPC)))   # 8 x 20
E, NU = 2.0e7, 0.3
BULK, SHEAR = E / (3.0 * (1.0 - 2.0 * NU)), E / (2.0 * (1.0 + NU))
SIGMA_Y, FRICTION = 1.0e3, 0.1
K0 = NU / (1.0 - NU)
ROOT23 = math.sqrt(2.0 / 3.0)
#: |sigma_yy| above which the elastic K0 state is outside the yield surface.
SIGYY_YIELD = ROOT23 * SIGMA_Y / ((1.0 - K0) * ROOT23 - FRICTION * (1.0 + 2.0 * K0))
DAMPING = 0.8
STEPS, DT = 3000, 1.5e-4
#: The deep particles: below this depth the static |sigma_yy| is above
#: 2 * SIGYY_YIELD (5.9 kPa against 2 x 2.9 kPa).
DEEP = 0.3
#: f1 <= this is "on or inside the yield surface" (the return map closes
#: its residual to 1e-10 absolute; stresses are ~1e4 Pa).
F1_TOL = 1.0e-6 * ROOT23 * SIGMA_Y


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def _bridge(staged: bool) -> apeSees:
    nodes, ids = grid_nodes((2, 5), (0.0, 0.0), (W, HC))
    ops = apeSees(make_fem(nodes, {"Column": quads(ids)}), units="N_m_s",
                  _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.DruckerPrager(
        K=BULK, G=SHEAR, sigmaY=SIGMA_Y, rho=FRICTION, rhoBar=FRICTION,
        Kinf=0.0, Ko=0.0, delta1=0.0, delta2=0.0, H=0.0, theta=1.0,
        name="weak")
    ops.fix(pg="Column", dofs=[1, 1])          # no free FE DOF (fork v1)
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(4, 12))
    ops.mpm.body("Column", material="weak", density=RHO, ppc=PPC)
    ops.mpm.fix("xmin", "x")
    ops.mpm.fix("xmax", "x")
    ops.mpm.fix("ymin", "xy")
    ops.mpm.gravity(0.0, -GRAV)
    ops.mpm.damping(DAMPING)
    if staged:
        ops.mpm.stage("weak", 0)
    ops.mpm.analyze(STEPS, DT)
    if staged:
        ops.mpm.stage("weak", 2)
    ops.mpm.analyze(STEPS, DT)
    return ops


def _probed(text: str) -> str:
    """The emitted deck with the fork import, the dumps after each analyze
    and the stage counts added; every emitted line is kept as it is."""
    stock = "import openseespy.opensees as ops"
    assert stock in text
    text = text.replace(stock, _FORK_IMPORT + _PROBE)
    out, k = [], 0
    for line in text.splitlines():
        if re.match(r"^ops\.mpm\('stage', ", line):
            out.append("_stage_counts += _q(" + line[len("ops.mpm("):])
        elif re.match(r"^ops\.mpm\('analyze', ", line):
            k += 1
            out += [line, f"_dump('after_{k}')"]
        else:
            out.append(line)
    assert k == 2, text
    return "\n".join(out) + "\n"


def _state(path: Path) -> "dict[str, Any]":
    raw = json.loads(path.read_text(encoding="utf-8"))
    n = len(raw["id"])
    sig = np.asarray(raw["sig"]).reshape(n, 3)          # xx, yy, xy
    szz = np.asarray(raw["szz"])
    x = np.asarray(raw["x"]).reshape(n, 2)
    full = np.column_stack([sig[:, 0], sig[:, 1], szz])
    i1 = full.sum(axis=1)
    dev = full - i1[:, None] / 3.0
    norm = np.sqrt((dev ** 2).sum(axis=1) + 2.0 * sig[:, 2] ** 2)
    order = np.argsort(np.asarray(raw["id"]))
    return {"n": n, "x": x[order], "sig": sig[order],
            "f1": (norm + FRICTION * i1 - ROOT23 * SIGMA_Y)[order],
            "stage_counts": raw["stage_counts"]}


def _run(root: Path, name: str, staged: bool) -> "dict[str, Any]":
    ops = _bridge(staged)
    deck_dir, cwd = root / f"{name}_deck", root / f"{name}_cwd"
    deck_dir.mkdir()
    cwd.mkdir()
    deck = deck_dir / f"{name}.py"
    ops.py(str(deck))
    text = deck.read_text(encoding="utf-8")
    deck.write_text(_probed(text), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=cwd, env=_env(),
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return {"deck": text, "tag": ops.tag_for(ops._names["weak"]),
            "after_1": _state(cwd / "after_1.json"),
            "after_2": _state(cwd / "after_2.json")}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    root = tmp_path_factory.mktemp("mpm_stage")
    return {"staged": _run(root, "staged", True),
            "control": _run(root, "control", False)}


def _deep(state: "dict[str, Any]") -> np.ndarray:
    return HC - state["x"][:, 1] >= DEEP


def _ratio(state: "dict[str, Any]") -> np.ndarray:
    return state["sig"][:, 0] / state["sig"][:, 1]


def test_mpm_stage_walks_the_druckerprager_particles_on_fork(
    runs: "dict[str, Any]", record_property: Any,
) -> None:
    """The warrant row: both stage rows reach all 160 particles; stage 0
    holds the K0 ratio in the interior, and after stage 2 every particle is
    on or inside the yield surface."""
    run = runs["staged"]
    tag = run["tag"]
    assert (f"ops.mpm('stage', {tag}, 0)\nops.mpm('analyze', {STEPS}, {DT!r})\n"
            f"ops.mpm('stage', {tag}, 2)\nops.mpm('analyze', {STEPS}, {DT!r})"
            ) in run["deck"]
    one, two = run["after_1"], run["after_2"]
    assert one["n"] == two["n"] == N_PARTICLES
    assert one["stage_counts"] == [N_PARTICLES]
    assert two["stage_counts"] == [N_PARTICLES, N_PARTICLES]

    deep = _deep(one)
    ratio = _ratio(one)[deep]
    record_property("k0_ratio_min", float(ratio.min()))
    record_property("k0_ratio_max", float(ratio.max()))
    assert deep.sum() >= N_PARTICLES // 2
    np.testing.assert_allclose(ratio, K0, rtol=0, atol=0.02)
    # The K0 state is outside the surface at depth: stage 0 is not plastic.
    assert (one["f1"][deep] > 0.5 * ROOT23 * SIGMA_Y).all()

    record_property("f1_max_after_stage_2", float(two["f1"].max()))
    assert (two["f1"] <= F1_TOL).all(), two["f1"].max()


def test_mpm_stage_zero_settles_to_the_geostatic_column_on_fork(
    runs: "dict[str, Any]", record_property: Any,
) -> None:
    """At the end of stage 0 the interior carries its weight:
    sigma_yy = -rho g (H - y) to 10 %."""
    one = runs["staged"]["after_1"]
    deep = _deep(one)
    want = -RHO * GRAV * (HC - one["x"][deep, 1])
    err = np.abs(one["sig"][deep, 1] / want - 1.0)
    record_property("geostatic_rel_err_max", float(err.max()))
    assert err.max() < 0.10


def test_mpm_control_without_stage_ends_elastoplastic_on_fork(
    runs: "dict[str, Any]", record_property: Any,
) -> None:
    """DruckerPrager's default stage is 2, so without the stage rows the
    column is elastoplastic from its first step: the control is never at the
    stage-0 state."""
    run = runs["control"]
    assert "'stage'" not in run["deck"]
    staged_one = runs["staged"]["after_1"]
    for key in ("after_1", "after_2"):
        state = run[key]
        assert state["stage_counts"] == []
        assert (state["f1"] <= F1_TOL).all(), (key, state["f1"].max())
        deep = _deep(state)
        ratio = _ratio(state)[deep]
        record_property(f"control_{key}_ratio_min", float(ratio.min()))
        assert ratio.min() > K0 + 0.05, (key, ratio.min())
        # The stage-0 run at the same instant is outside the surface.
        assert staged_one["f1"][_deep(staged_one)].min() > 0.0
    final = runs["staged"]["after_2"]
    diff = np.abs(run["after_2"]["sig"] - final["sig"]).max()
    record_property("control_vs_staged_final_sig_max_abs_diff", float(diff))
