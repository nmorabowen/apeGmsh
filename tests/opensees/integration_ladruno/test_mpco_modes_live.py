"""#1946 G1 — modes read back from an ``.mpco`` (the stock fallback format).

The same 2-D cantilever as ``test_ladruno_modes_live``, recorded by
``recorder mpco -N modesOfVibration modesOfVibrationRotational`` and read
through ``Results.from_mpco(..., model_h5=)``. It runs on the fork's
in-process module because pip ``openseespy`` 3.8.0 ships no MPCO recorder
(live-stock log, 2026-10-14): on a pip install the only fallback is
``.out``, which has no modes route yet (the slim stock reader, #1961).

The staged case pins the stage-name pairing: a program with one static
stage, then ``eigen``, keeps the program's name on the static stage, and
the mode stages take none (``_bind_stage_names`` pairs positionally).

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import apeGmsh
from apeGmsh.results import Results

from tests.opensees.integration_ladruno.test_ladruno_modes_live import (
    MODAL_TOKENS,
    N_MODES,
    assert_modes_match_live,
    modal_bridge,
    solve_and_record,
)
from tests.opensees.live.test_readback_live import (
    P_TIP,
    build_cantilever_fem,
)

pytestmark = pytest.mark.ladruno_fork


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def _fresh_env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    return env


_STALE_EIGEN_REPRO = """
import sys
from apeGmsh.opensees.emitter.live import get_ops
ops = get_ops()
def model():
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0); ops.node(2, 0.0, 1.0)
    ops.fix(1, 1, 1, 1); ops.mass(2, 1.0, 1.0, 0.0)
    ops.geomTransf("Linear", 1)
    ops.element("elasticBeamColumn", 1, 1, 2, 0.01, 200e9, 1e-4, 1)
model()
if sys.argv[2] == "eigen":
    ops.eigen(1)
model()
ops.recorder("mpco", sys.argv[1], "-N", "modesOfVibration")
ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
ops.load(2, 1.0, 0.0, 0.0)
ops.constraints("Plain"); ops.numberer("Plain"); ops.system("BandGeneral")
ops.test("NormDispIncr", 1e-9, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
ops.remove("recorders")
print("survived")
"""


#: ``exit(-1)`` as the parent sees it: 4294967295 on Windows, 255 on
#: POSIX; 127 was observed when the run went through a POSIX shell.
_EXIT_MINUS_ONE = {-1, 127, 255, 4294967295}
_STALE_EIGEN_MESSAGE = "Eigenvalues were never set"


class StaleEigenExit(Exception):
    """The known fork crash, and nothing else, reached the test."""


def _run_repro(tmp_path: Path, mode: str) -> "subprocess.CompletedProcess[str]":
    script = tmp_path / "repro.py"
    script.write_text(_STALE_EIGEN_REPRO, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script), str(tmp_path / "r.mpco"), mode],
        cwd=tmp_path, env=_fresh_env(), capture_output=True, text=True,
        check=False,
    )


def test_mpco_modes_recorder_runs_without_a_prior_eigen(tmp_path: Path) -> None:
    """Control: the same deck without the first ``eigen`` survives, so
    the pin below is about the eigen-then-wipe sequence only."""
    proc = _run_repro(tmp_path, "control")
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-500:])
    assert "survived" in proc.stdout


def test_mpco_modes_recorder_survives_eigen_then_wipe(tmp_path: Path) -> None:
    """``eigen``, ``wipe``, then an MPCO modal recorder on a fresh model.

    Builds before ``ladruno-v20261015`` gated the modal channel on the
    interpreter's stale ``numEigen`` and called ``Domain::getEigenvalues``,
    which ``exit(-1)``s the process (#1983, pinned on 79e0623670). The fork
    fix (nmorabowen/OpenSees#985) gates on the domain's own eigen count, so
    at FORK_PIN the run survives. The old crash still surfaces by name.
    """
    proc = _run_repro(tmp_path, "eigen")
    output = proc.stdout + proc.stderr
    if proc.returncode in _EXIT_MINUS_ONE and _STALE_EIGEN_MESSAGE in output:
        raise StaleEigenExit(proc.returncode)
    assert proc.returncode == 0 and "survived" in proc.stdout, (
        proc.returncode, proc.stdout[-500:], proc.stderr[-500:])


def test_mpco_modes_match_live_eigen(cantilever: Any, tmp_path: Path) -> None:
    mpco, model_h5 = tmp_path / "modes.mpco", tmp_path / "model.h5"
    ops = modal_bridge(cantilever)
    ops.recorder.MPCO(file=str(mpco), nodal_responses=MODAL_TOKENS)
    ops.h5(str(model_h5))
    lambdas, vectors = solve_and_record(ops)

    res = Results.from_mpco(str(mpco), model_h5=str(model_h5))
    assert_modes_match_live(res, lambdas, vectors)


def test_static_stage_keeps_its_program_name_beside_the_modes(
    cantilever: Any, tmp_path: Path,
) -> None:
    """``ops.stage("gravity")`` (static), then ``eigen``: one MODEL_STAGE
    holds the static steps and the modes. The static stage is named
    ``gravity``; the mode stages stay ``mode_<k>``."""
    mpco, model_h5 = tmp_path / "staged.mpco", tmp_path / "model.h5"
    deck = tmp_path / "staged.py"
    ops = modal_bridge(cantilever)
    ops.recorder.MPCO(
        file=str(mpco), nodal_responses=("displacement", *MODAL_TOKENS),
    )
    solver = ops.analysis.chain(
        algorithm=ops.algorithm.Linear(),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.BandGeneral(),
        analysis=ops.analysis.Static(),
    )
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=ops.nodes.one(label="tip"), forces=(0.0, -P_TIP, 0.0))
        s.analysis(solver, integrator=ops.integrator.LoadControl(dlam=0.5))
        s.run(n_increments=2)
    ops.h5(str(model_h5))
    ops.py(str(deck), progress=False)

    # Run the deck on the module the bridge drives (the fork), then solve
    # the eigenproblem after the stage and record the modes. It runs in a
    # fresh process: after an earlier eigen + wipe in this process the
    # domain keeps a stale eigen count, and the MPCO recorder's modal
    # channel then kills the process at the first analyze step (fork
    # build 79e0623670, "Eigenvalues were never set").
    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()",
    )
    out = tmp_path / "lambdas.json"
    text += (
        "\nimport json\n"
        f"LAMBDAS = ops.eigen({N_MODES})\n"
        "ops.record()\n"
        "ops.remove('recorders')\n"
        f"open({str(out)!r}, 'w').write(json.dumps(LAMBDAS))\n"
    )
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp_path, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    lambdas = json.loads(out.read_text(encoding="utf-8"))

    res = Results.from_mpco(str(mpco), model_h5=str(model_h5))
    names = [(s.name, s.kind) for s in res.stages]
    assert names[0] == ("gravity", "transient")
    assert names[1:] == [(f"mode_{k}", "mode") for k in range(1, N_MODES + 1)]
    assert [m.eigenvalue for m in res.modes] == pytest.approx(lambdas, rel=1e-10)
    # The static stage still reads its own steps (two increments and the
    # record after eigen), not a mode shape.
    assert res.stage("gravity").nodes.get(
        component="displacement_y").values.shape[0] == 3
