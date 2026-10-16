"""#2101 — ``Results.from_ladruno(model_h5=)`` takes the program's stage names.

One 2-D cantilever, two ``ops.stage`` blocks of different kinds:
``gravity`` (``Static``, ``LoadControl``) then ``shake`` (``Transient``,
``Newmark``).  The bridge emits a Py deck recording a ``.ladruno``
(displacement only: no modal channel, so the fork stamps each stage's
``KIND`` with the kind it ran, not ``eigen``, fork #2103); the deck runs on
the fork module in a fresh process.

Oracles:

* beside its own ``model.h5`` the capture's stages take the program
  names and kinds, keep ``MODEL_STAGE[<k>]`` as an alias, and a name
  reads the same slab as its alias;
* beside a ``model.h5`` that declares the same stages in the other order
  the fork's recorded ``KIND``s disagree with the program kinds, so
  nothing is renamed and ``StageKindMismatchWarning`` is the one stage
  warning (#2079 Q7).

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import Results
from apeGmsh.results._bind import (
    DuplicateStageNameWarning,
    ShadowedStageNameWarning,
    StageCountMismatchWarning,
)
from apeGmsh.results.readers._ladruno import StageKindMismatchWarning

from tests.opensees.integration_ladruno.test_ladruno_modes_live import modal_bridge
from tests.opensees.live.test_readback_live import P_TIP, build_cantilever_fem

pytestmark = pytest.mark.ladruno_fork

N_GRAV, N_DYN, DT = 4, 10, 0.01
_STAGE_WARNINGS = (
    StageKindMismatchWarning, StageCountMismatchWarning,
    DuplicateStageNameWarning, ShadowedStageNameWarning,
)


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


def _bridge(fem: Any, order: "tuple[str, ...]", ladruno: Path) -> Any:
    """The cantilever with its stages declared in ``order``."""
    ops = modal_bridge(fem)
    ops.recorder.Ladruno(file=str(ladruno), nodal_responses=("displacement",))
    tip = ops.nodes.one(label="tip")
    for name in order:
        with ops.stage(name=name) as s:
            if name == "gravity":
                with s.pattern(series=ops.timeSeries.Linear()) as p:
                    p.load(node=tip, forces=(0.0, -P_TIP, 0.0))
                s.analysis(**_chain(
                    ops, ops.integrator.LoadControl(dlam=1.0 / N_GRAV),
                    ops.analysis.Static()))
                s.run(n_increments=N_GRAV)
            else:
                with s.pattern(series=ops.timeSeries.Constant()) as p:
                    p.load(node=tip, forces=(P_TIP, 0.0, 0.0))
                s.analysis(**_chain(
                    ops, ops.integrator.Newmark(gamma=0.5, beta=0.25),
                    ops.analysis.Transient()))
                s.run(n_increments=N_DYN, dt=DT)
    return ops


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    tmp = tmp_path_factory.mktemp("stage_binding_2101")
    ladruno, deck = tmp / "r.ladruno", tmp / "deck.py"
    fem = build_cantilever_fem()
    ops = _bridge(fem, ("gravity", "shake"), ladruno)
    ops.h5(str(tmp / "model.h5"))
    ops.py(str(deck), progress=False)
    # The same program with its stages swapped: never run, only bound.
    _bridge(fem, ("shake", "gravity"), tmp / "unused.ladruno").h5(
        str(tmp / "swapped.h5"))

    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n",
    )
    text += "\nops.remove('recorders')\nprint('DECK_DONE')\n"
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0 and "DECK_DONE" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-2000:])
    return {"ladruno": ladruno, "model_h5": tmp / "model.h5",
            "swapped": tmp / "swapped.h5"}


def _open(path: Path, model_h5: "Path | None") -> "tuple[Results, list[type]]":
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = Results.from_ladruno(path, model_h5=model_h5)
    return r, [w.category for w in rec
               if issubclass(w.category, _STAGE_WARNINGS)]


def _hosts(r: Results) -> "list[tuple[str, str, tuple[str, ...]]]":
    return [(s.name, s.kind, s.aliases) for s in r.stages if s.kind != "mode"]


def test_file_records_the_kinds_it_ran(run: "dict[str, Path]") -> None:
    """The precondition: without ``model_h5=`` the fork's own stamps."""
    r, raised = _open(run["ladruno"], None)
    with r:
        assert raised == []
        assert _hosts(r) == [("MODEL_STAGE[1]", "static", ()),
                             ("MODEL_STAGE[2]", "transient", ())]


def test_program_names_and_kinds_bind(run: "dict[str, Path]") -> None:
    r, raised = _open(run["ladruno"], run["model_h5"])
    with r:
        assert raised == []
        assert _hosts(r) == [
            ("gravity", "static", ("MODEL_STAGE[1]",)),
            ("shake", "transient", ("MODEL_STAGE[2]",)),
        ]
        for name, alias, n in (("gravity", "MODEL_STAGE[1]", N_GRAV),
                               ("shake", "MODEL_STAGE[2]", N_DYN)):
            by_name = np.asarray(
                r.stage(name).nodes.get(component="displacement_x").values)
            by_alias = np.asarray(
                r.stage(alias).nodes.get(component="displacement_x").values)
            assert by_name.shape[0] == n
            np.testing.assert_array_equal(by_name, by_alias)


def test_swapped_program_keeps_the_file_names(run: "dict[str, Path]") -> None:
    r, raised = _open(run["ladruno"], run["swapped"])
    with r:
        assert raised == [StageKindMismatchWarning]
        assert _hosts(r) == [("MODEL_STAGE[1]", "static", ()),
                             ("MODEL_STAGE[2]", "transient", ())]
