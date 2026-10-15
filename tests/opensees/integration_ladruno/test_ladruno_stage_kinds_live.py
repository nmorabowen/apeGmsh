"""#1948 G8 — stage kinds on ``.ladruno`` and MPCO, static then transient.

One 2-D cantilever, two ``ops.stage`` blocks: ``gravity`` (``Static``,
``LoadControl``) then ``shake`` (``Transient``, ``Newmark``, ``dt=0.01``).
The bridge emits a Py deck that records both a ``.ladruno`` and an
``.mpco``; the deck runs on the fork module in a fresh process and dumps
``ops.nodeDisp`` at the end of each stage (before its ``loadConst``).

Oracles: ``.ladruno`` stamps ``KIND`` per stage (``LadrunoReader.stages``);
MPCO is kinded from the archive's ``analysis`` record through
``from_mpco(..., model_h5=)``.  Both must list ``[static, transient]`` and
match ``ops.nodeDisp`` at each stage's last step (rtol 1e-10).  Pip
``openseespy`` has no MPCO recorder, so the MPCO case is fork-only too.
The ``.ladruno`` kind is a strict xfail until the fork stamps KIND per
stage (#2034); its values are checked apart, so the xfail hides nothing.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import Results

from tests.opensees.integration_ladruno.test_ladruno_modes_live import modal_bridge
from tests.opensees.live.test_readback_live import P_TIP, build_cantilever_fem

pytestmark = pytest.mark.ladruno_fork

N_GRAV, N_DYN, DT = 4, 25, 0.01
_LOAD_CONST = "ops.loadConst('-time', 0.0)\n"


def _fresh_env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    return env


def _chain(ops: Any, integrator: Any, analysis: Any) -> "dict[str, Any]":
    return dict(
        test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
        algorithm=ops.algorithm.Linear(), integrator=integrator,
        constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
        system=ops.system.BandGeneral(), analysis=analysis,
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    tmp = tmp_path_factory.mktemp("g8_stage_kinds")
    paths = {k: tmp / f"r.{k}" for k in ("ladruno", "mpco")}
    model_h5, deck = tmp / "model.h5", tmp / "deck.py"
    ops = modal_bridge(build_cantilever_fem())
    ops.recorder.Ladruno(file=str(paths["ladruno"]), nodal_responses=("displacement",))
    ops.recorder.MPCO(file=str(paths["mpco"]), nodal_responses=("displacement",))
    tip = ops.nodes.one(label="tip")
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=tip, forces=(0.0, -P_TIP, 0.0))
        s.analysis(**_chain(ops, ops.integrator.LoadControl(dlam=1.0 / N_GRAV),
                            ops.analysis.Static()))
        s.run(n_increments=N_GRAV)
    with ops.stage(name="shake") as s:
        with s.pattern(series=ops.timeSeries.Constant()) as p:
            p.load(node=tip, forces=(P_TIP, 0.0, 0.0))
        s.analysis(**_chain(ops, ops.integrator.Newmark(gamma=0.5, beta=0.25),
                            ops.analysis.Transient()))
        s.run(n_increments=N_DYN, dt=DT)
    ops.h5(str(model_h5))
    ops.py(str(deck), progress=False)

    # Run the deck on the module the bridge drives (the fork) and dump
    # every node's displacement where each stage ends.
    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text and text.count(_LOAD_CONST) == 2
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n"
        "SNAPS = []\n"
        "def _snap():\n"
        "    SNAPS.append({int(n): list(ops.nodeDisp(int(n)))"
        " for n in ops.getNodeTags()})\n",
    )
    text = text.replace(_LOAD_CONST, "_snap()\n" + _LOAD_CONST)
    out = tmp / "snaps.json"
    text += (
        "\nimport json\n"
        "ops.remove('recorders')\n"
        f"open({str(out)!r}, 'w').write(json.dumps(SNAPS))\n"
    )
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    snaps = json.loads(out.read_text(encoding="utf-8"))
    return {"paths": paths, "model_h5": model_h5,
            "snaps": [{int(k): v for k, v in s.items()} for s in snaps]}


def _open(run: "dict[str, Any]", fmt: str) -> Results:
    path = str(run["paths"][fmt])
    if fmt == "ladruno":
        return Results.from_ladruno(path)
    return Results.from_mpco(path, model_h5=str(run["model_h5"]))


#: The fork's ``.ladruno`` KIND is one recorder-wide ``-kind`` option
#: (default ``static``), stamped on every stage; the bridge passes none,
#: so the transient stage reads ``static``.  The fork fix is #2034.
_LADRUNO_KIND_PER_RECORDER = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#2034: the fork stamps .ladruno KIND per recorder (-kind, "
    "default static), not per stage; the transient stage reads static. "
    "Observed on build 79e0623670.",
)


@pytest.mark.parametrize("fmt", [
    pytest.param("ladruno", marks=_LADRUNO_KIND_PER_RECORDER),
    "mpco",
])
def test_static_then_transient_kinds(run: "dict[str, Any]", fmt: str) -> None:
    stages = [s for s in _open(run, fmt).stages if s.kind != "mode"]
    assert [s.kind for s in stages] == ["static", "transient"]


def test_mpco_stages_take_the_program_names(run: "dict[str, Any]") -> None:
    stages = [s for s in _open(run, "mpco").stages if s.kind != "mode"]
    assert [s.name for s in stages] == ["gravity", "shake"]


@pytest.mark.parametrize("fmt", ["ladruno", "mpco"])
def test_last_step_values_match_node_disp(run: "dict[str, Any]", fmt: str) -> None:
    """Kept apart from the kinds so the #2034 xfail cannot hide a value
    mismatch: the stages pair with the snapshots in run order."""
    res = _open(run, fmt)
    stages = [s for s in res.stages if s.kind != "mode"]
    assert len(stages) == 2
    for stage, snap, n in zip(stages, run["snaps"], (N_GRAV, N_DYN)):
        st = res.stage(stage.id)
        for dof, comp in enumerate(("displacement_x", "displacement_y")):
            slab = st.nodes.get(component=comp)
            values = np.asarray(slab.values)
            assert values.shape[0] == n
            ids = [int(i) for i in np.asarray(slab.node_ids)]
            assert sorted(ids) == sorted(snap)
            want = [snap[i][dof] for i in ids]
            np.testing.assert_allclose(values[-1], want, rtol=1e-10, atol=1e-14)
