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
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp_path, env=env,
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
