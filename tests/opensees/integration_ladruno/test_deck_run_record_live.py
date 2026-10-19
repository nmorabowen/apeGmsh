"""#2210 — the deck run record on a real engine (route (c)).

The truss of ``tests/test_deck_run_record.py`` carries a static ramp whose
tip displacement is ``P L / E A``. Its unstaged deck, emitted with
``analyze_steps``, runs in a fresh process with a bridge MPCO recorder:

* the deck writes ``deck.mpco.run.json``, naming the engine it ran on (the
  fork's ``ladrunoBuild`` sha, or ``stock`` / ``unknown``);
* ``read_deck_mpco`` then converts with no warning, and the ``.ladruno`` read
  equals the ``.mpco`` read and the closed form at ``rtol=1e-12``;
* F2 of #2224's review: the same bridge's deck exported, ``ops.h5()``, then
  an in-process gravity, ``loadConst`` and Transient run into the same
  ``.mpco``: no record exists, and ``read_deck_mpco`` warns once (check 1)
  and reads the ``.mpco``.

The fork cases carry ``ladruno_fork`` (the live-fork lane); the stock cases
carry ``live`` and run on stock openseespy (live-stock). The Tcl deck needs
an OpenSees Tcl binary (``$OPENSEES_BIN`` or ``OpenSees`` on ``PATH``),
which no CI lane has; ``tests/test_deck_run_record.py`` runs the Tcl record
block itself in ``tkinter.Tcl`` on every lane.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import Results
from apeGmsh.results.live import DefaultRouteWarning, read_deck_mpco
from tests.test_deck_run_record import STEPS, TIP, truss_bridge

_FORK_IMPORT = (
    "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n")


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def tip_x(res: Results) -> np.ndarray:
    slab = res.stage(res.stages[-1].id).nodes.get(component="displacement_x")
    ids = list(np.asarray(slab.node_ids))
    return np.asarray(slab.values)[:, ids.index(2)]


def _run_deck(tmp_path: Path, dialect: str, *, fork_module: bool,
              ) -> "tuple[Path, Path]":
    """Emit the truss deck (analyze_steps), archive the bridge, run the deck
    in a fresh process exactly as emitted (the py deck's import aside, on the
    fork, whose module is not ``openseespy``)."""
    mpco, model = tmp_path / "deck.mpco", tmp_path / "model.h5"
    deck = tmp_path / f"deck.{dialect}"
    bridge = truss_bridge(mpco)
    bridge.h5(str(model))
    getattr(bridge, dialect)(str(deck), analyze_steps=STEPS, progress=False)
    if dialect == "py":
        if fork_module:
            text = deck.read_text(encoding="utf-8")
            stock = "import openseespy.opensees as ops"
            assert stock in text
            deck.write_text(text.replace(stock, _FORK_IMPORT), encoding="utf-8")
        cmd = [sys.executable, str(deck)]
    else:
        binary = os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")
        if not binary:
            pytest.skip("no OpenSees Tcl binary ($OPENSEES_BIN or OpenSees "
                        "on PATH); the suite runs the Tcl record block in "
                        "tkinter.Tcl")
        cmd = [binary, str(deck)]
    proc = subprocess.run(cmd, cwd=tmp_path, env=_env(), capture_output=True,
                          text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert mpco.exists()
    return mpco, model


def _assert_converts(tmp_path: Path, mpco: Path, model: Path,
                     backend: str, build: str) -> None:
    rec = json.loads((tmp_path / "deck.mpco.run.json").read_text("utf-8"))
    assert (rec["opensees_backend"], rec["opensees_build_sha"]) == (backend, build)
    assert rec["steps_completed"] == rec["analyze_steps"] == STEPS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = Results.from_mpco(str(mpco), model_h5=str(model))
        res = read_deck_mpco(mpco, model_h5=model)
    try:
        assert [s.kind for s in res.stages] == ["static"]
        np.testing.assert_allclose(tip_x(res), tip_x(ref), rtol=1e-12)
        np.testing.assert_allclose(tip_x(res)[-1], TIP, rtol=1e-12)
    finally:
        ref.close()
        res.close()
    assert (tmp_path / "deck.ladruno").exists()


def _fork_build() -> str:
    from apeGmsh.opensees.emitter.live import get_backend_build

    build = get_backend_build()
    assert build is not None
    return build


@pytest.mark.ladruno_fork
def test_fork_py_deck_writes_its_record_and_converts(tmp_path) -> None:
    mpco, model = _run_deck(tmp_path, "py", fork_module=True)
    _assert_converts(tmp_path, mpco, model, "ladruno", _fork_build())


def _stock_with_mpco(where: Path) -> None:
    """Skip unless the bound module is stock and the deck's interpreter
    has an MPCO recorder (stock pip openseespy ships none)."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    if get_backend_info().kind != "stock":
        pytest.skip("the bound module is the fork; the fork case covers it")
    probe = "\n".join([
        "import openseespy.opensees as ops",
        "ops.model('basic', '-ndm', 1, '-ndf', 1)", "ops.node(1, 0.0)",
        "ops.recorder('mpco', 'probe.mpco', '-N', 'displacement')",
        "ops.wipe()",
    ])
    proc = subprocess.run([sys.executable, "-c", probe], cwd=where,
                          capture_output=True, check=False)
    if proc.returncode != 0:
        pytest.skip("this stock openseespy has no MPCO recorder (pip "
                    "openseespy ships none); the fork case covers the route")


@pytest.mark.live
def test_stock_py_deck_writes_its_record_and_converts(tmp_path) -> None:
    _stock_with_mpco(tmp_path)
    mpco, model = _run_deck(tmp_path, "py", fork_module=False)
    _assert_converts(tmp_path, mpco, model, "stock", "unknown")


@pytest.mark.subprocess
def test_tcl_deck_writes_its_record_and_converts(tmp_path) -> None:
    mpco, model = _run_deck(tmp_path, "tcl", fork_module=False)
    rec = json.loads((tmp_path / "deck.mpco.run.json").read_text("utf-8"))
    _assert_converts(tmp_path, mpco, model, rec["opensees_backend"],
                     rec["opensees_build_sha"])
    assert rec["opensees_backend"] in ("ladruno", "stock")


def _in_process_after_export(tmp_path: Path) -> "tuple[Path, Path]":
    """F2: export the deck, archive, then gravity -> loadConst -> Transient
    in this process into the deck's own ``.mpco``."""
    from apeGmsh.opensees.emitter.live import get_ops

    mpco, model = tmp_path / "deck.mpco", tmp_path / "model.h5"
    bridge = truss_bridge(mpco)
    bridge.py(str(tmp_path / "deck.py"), analyze_steps=STEPS, progress=False)
    bridge.h5(str(model))
    opspy: Any = get_ops()
    bridge.run()
    try:
        for _ in range(STEPS):
            assert opspy.analyze(1) == 0
        opspy.loadConst("-time", 0.0)
        opspy.wipeAnalysis()
        opspy.constraints("Plain")
        opspy.numberer("RCM")
        opspy.system("BandGeneral")
        opspy.test("NormDispIncr", 1e-12, 10)
        opspy.algorithm("Linear")
        opspy.integrator("Newmark", 0.5, 0.25)
        opspy.analysis("Transient")
        for _ in range(2):
            assert opspy.analyze(1, 0.01) == 0
    finally:
        opspy.wipe()
    assert mpco.exists()
    return mpco, model


def _assert_refuses_once(tmp_path: Path, mpco: Path, model: Path) -> None:
    assert list(tmp_path.glob("*.run.json*")) == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = read_deck_mpco(mpco, model_h5=model)
    try:
        ours = [w for w in caught if w.category is DefaultRouteWarning]
        assert len(ours) == 1, [str(w.message) for w in caught]
        assert "check 1 (the record)" in str(ours[0].message)
        np.testing.assert_allclose(tip_x(res)[STEPS - 1], TIP, rtol=1e-12)
    finally:
        res.close()
    assert not (tmp_path / "deck.ladruno").exists()


@pytest.mark.ladruno_fork
def test_fork_in_process_run_after_an_export_refuses(tmp_path) -> None:
    _assert_refuses_once(tmp_path, *_in_process_after_export(tmp_path))


@pytest.mark.live
def test_stock_in_process_run_after_an_export_refuses(tmp_path) -> None:
    _stock_with_mpco(tmp_path)
    _assert_refuses_once(tmp_path, *_in_process_after_export(tmp_path))
