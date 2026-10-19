"""#1950 — the default results route on the fork, and on a stock build.

The two-node truss (``tests/test_results_live_default_format.py``) carries
a static load ramp whose tip displacement is ``P L / E A``; every read is
compared to ``ops.nodeDisp`` and to that closed form at ``rtol=1e-12``.

* **fork, in process** (``spec.emit_default``): the fork's own file, with
  ``PROVENANCE@model_hash`` equal to the archive's, read with no warning.
* **stock with MPCO, in process**: the same module with its fork signal
  hidden (``kind == "stock"``). The ``.mpco`` is kept, read with
  ``from_mpco`` after exactly one warning, and never converted.
* **a deck, out of process** (``read_deck_mpco``): the bridge's deck runs
  in a fresh process with a bridge MPCO recorder (the stock MPCO format).
  In one ``ops.stage``, or unstaged with one ``analysis`` call and the
  deck's ``analyze`` call archived (``ops.h5`` before the deck or after it,
  #2210), the ``.mpco`` converts and the converted read equals the
  ``.mpco`` read. Two ``analysis`` calls, or an archive a later deck
  re-wrote to another step count, read with ``from_mpco`` after exactly
  one warning (ruling (ii) on #1950).
* **in process** (``read_deck_mpco`` on an in-process ``.mpco``): a raw
  ``opspy.analyze`` loop, and gravity then ``loadConst`` then a transient
  run (raw, or the bridge's ``ops.analyze`` for the gravity), archive no
  ``analyze`` call: one warning, then ``from_mpco``.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import Results
from apeGmsh.results.live import DefaultRouteWarning, read_deck_mpco
from tests.test_results_live_default_format import (
    TRUSS_STEPS,
    TRUSS_TIP,
    tip_x,
    truss_bridge,
    truss_fem,
    truss_spec,
)

pytestmark = pytest.mark.ladruno_fork


class StockView:
    """The fork module with its fork signal hidden: ``kind == "stock"``."""

    _HIDDEN = frozenset({"ladrunoBuild", "ladrunoSchema"})

    def __init__(self, ops: Any) -> None:
        self._ops = ops

    def __getattr__(self, name: str) -> Any:
        if name in StockView._HIDDEN:
            raise AttributeError(name)
        return getattr(self._ops, name)


def _in_process(tmp_path: Path, *, stock: bool) -> "tuple[Any, list, float, Path]":
    from apeGmsh.opensees._target import backend_info_of
    from apeGmsh.opensees.emitter.live import get_ops

    opspy = get_ops()
    assert backend_info_of(opspy).kind == "fork"
    ops = StockView(opspy) if stock else opspy
    if stock:
        assert backend_info_of(ops).kind == "stock"
    fem = truss_fem()
    bridge = truss_bridge(fem)
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    bridge.run()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with truss_spec(fem).emit_default(tmp_path / "run", model_h5=model,
                                              ops=ops) as run:
                for _ in range(TRUSS_STEPS):
                    assert opspy.analyze(1) == 0
        tip = float(opspy.nodeDisp(2, 1))
    finally:
        opspy.wipe()
    return run, caught, tip, model


def _assert_tip(res: Results, tip: float) -> None:
    x = tip_x(res)
    assert x.shape == (TRUSS_STEPS,)
    np.testing.assert_allclose(x[-1], tip, rtol=1e-12)
    np.testing.assert_allclose(x[-1], TRUSS_TIP, rtol=1e-12)


def test_fork_route_records_stamps_and_reads_its_own_file(tmp_path) -> None:
    run, caught, tip, model = _in_process(tmp_path, stock=False)
    assert run.route == "fork" and caught == []
    assert run.path == tmp_path / "run.ladruno" and run.path.exists()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results()
    try:
        _assert_tip(res, tip)
    finally:
        res.close()
    from apeGmsh.opensees.emitter.live import get_ops

    with h5py.File(run.path, "r") as f, h5py.File(model, "r") as m:
        prov = f["INFO/PROVENANCE"].attrs
        assert str(prov["model_hash"]) == str(m["meta/lineage"].attrs["model_hash"])
        # G10 (#2208): the default route stamps the build that solved it.
        assert str(prov["opensees_build_sha"]) == str(get_ops().ladrunoBuild()).strip()
        assert str(prov["opensees_backend"]) == "ladruno"


def test_stock_in_process_keeps_the_mpco_and_warns_once(tmp_path) -> None:
    run, caught, tip, _ = _in_process(tmp_path, stock=True)
    assert run.route == "mpco" and run.path == tmp_path / "run.mpco"
    assert [w.category for w in caught] == [DefaultRouteWarning]
    assert "not converted" in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results()
    try:
        _assert_tip(res, tip)
    finally:
        res.close()
    assert run.path.exists() and not (tmp_path / "run.ladruno").exists()


# ---------------------------------------------------------------------------
# Out of process: a deck the bridge emitted
# ---------------------------------------------------------------------------


def _fresh_env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    return env


def _deck_run(tmp_path: Path, *, analyses: int = 1, staged: bool = False,
              h5_first: bool = True) -> "tuple[Path, Path, Any]":
    """Emit the truss deck with a bridge MPCO recorder and run it fresh.

    ``h5_first`` writes the archive before the deck (which then re-writes
    it with its ``analyze`` call), else after it."""
    mpco, model, deck = tmp_path / "deck.mpco", tmp_path / "model.h5", tmp_path / "deck.py"
    bridge = truss_bridge(truss_fem(), analyses=analyses, staged=staged,
                          mpco=mpco)
    if h5_first:
        bridge.h5(str(model))
    if staged:
        bridge.py(str(deck), progress=False)
    else:
        bridge.py(str(deck), progress=False, analyze_steps=TRUSS_STEPS)
    if not h5_first:
        bridge.h5(str(model))
    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n",
    )
    text += "\nops.remove('recorders')\nprint('DECK-DONE')\n"
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp_path, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0 and "DECK-DONE" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-2000:])
    assert mpco.exists()
    return mpco, model, bridge


def _assert_converts(tmp_path: Path, mpco: Path, model: Path, kind: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = Results.from_mpco(str(mpco), model_h5=str(model))
        res = read_deck_mpco(mpco, model_h5=model)
    try:
        assert [s.kind for s in res.stages] == [kind]
        np.testing.assert_allclose(tip_x(res), tip_x(ref), rtol=1e-12)
        np.testing.assert_allclose(tip_x(res)[-1], TRUSS_TIP, rtol=1e-12)
    finally:
        ref.close()
        res.close()
    assert mpco.exists() and (tmp_path / "deck.ladruno").exists()


def _assert_falls_back(tmp_path: Path, mpco: Path, model: Path,
                       reason: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = read_deck_mpco(mpco, model_h5=model)
    try:
        assert [w.category for w in caught] == [DefaultRouteWarning]
        assert reason in str(caught[0].message), str(caught[0].message)
        assert len(tip_x(res)) >= TRUSS_STEPS
    finally:
        res.close()
    assert not (tmp_path / "deck.ladruno").exists()


def test_a_staged_deck_converts_and_equals_the_mpco(tmp_path) -> None:
    mpco, model, _ = _deck_run(tmp_path, staged=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = Results.from_mpco(str(mpco), model_h5=str(model))
        res = read_deck_mpco(mpco, model_h5=model)
    try:
        assert [s.kind for s in res.stages] == ["static"]
        np.testing.assert_allclose(tip_x(res), tip_x(ref), rtol=1e-12)
        np.testing.assert_allclose(tip_x(res)[-1], TRUSS_TIP, rtol=1e-12)
    finally:
        ref.close()
        res.close()
    assert mpco.exists() and (tmp_path / "deck.ladruno").exists()


@pytest.mark.parametrize("h5_first", [True, False], ids=["h5-first", "h5-after"])
def test_an_unstaged_deck_converts_and_equals_the_mpco(tmp_path, h5_first) -> None:
    """Ruling (ii), #2210: one ``analysis`` call and the deck's archived
    ``analyze`` call, whichever of ``ops.h5`` / ``ops.py`` ran first."""
    mpco, model, _ = _deck_run(tmp_path, h5_first=h5_first)
    _assert_converts(tmp_path, mpco, model, "static")


def test_an_unstaged_deck_with_two_analysis_calls_warns_once(tmp_path) -> None:
    mpco, model, _ = _deck_run(tmp_path, analyses=2)
    _assert_falls_back(tmp_path, mpco, model, "2 analysis and 1 analyze")


def test_a_deck_whose_archive_a_later_deck_rewrote_warns_once(tmp_path) -> None:
    """A second deck with another step count re-writes the archive (and
    says so); the first deck's run then mismatches it: (d)."""
    mpco, model, bridge = _deck_run(tmp_path)
    with pytest.warns(UserWarning, match="earlier deck's analyze"):
        bridge.py(str(tmp_path / "longer.py"), progress=False,
                  analyze_steps=TRUSS_STEPS + 1)
    _assert_falls_back(tmp_path, mpco, model,
                       f"analyze call runs {TRUSS_STEPS + 1}")


def _in_process_mpco(tmp_path: Path, *, transient: bool,
                     bridge_analyze: bool) -> "tuple[Path, Path]":
    """The truss in process with its bridge MPCO recorder: a static ramp
    (raw ``opspy.analyze`` or the bridge's ``ops.analyze``), then, with
    ``transient``, ``loadConst`` and a raw transient run."""
    from apeGmsh.opensees.emitter.live import get_ops

    opspy = get_ops()
    mpco, model = tmp_path / "deck.mpco", tmp_path / "model.h5"
    bridge = truss_bridge(truss_fem(), mpco=mpco)
    bridge.h5(str(model))
    try:
        if bridge_analyze:
            assert bridge.analyze(steps=TRUSS_STEPS) == 0
        else:
            bridge.run()
            for _ in range(TRUSS_STEPS):
                assert opspy.analyze(1) == 0
        if transient:
            opspy.loadConst("-time", 0.0)
            opspy.integrator("Newmark", 0.5, 0.25)
            opspy.analysis("Transient")
            for _ in range(TRUSS_STEPS):
                assert opspy.analyze(1, 0.01) == 0
        opspy.remove("recorders")
    finally:
        opspy.wipe()
    bridge.h5(str(model))      # after the run: still no deck analyze call
    assert mpco.exists()
    return mpco, model


@pytest.mark.parametrize("transient, bridge_analyze", [
    (False, False), (True, False), (True, True),
], ids=["raw", "gravity-loadConst-transient-raw",
        "gravity-loadConst-transient-bridge-analyze"])
def test_an_in_process_mpco_warns_once_and_reads_the_mpco(
    tmp_path, transient, bridge_analyze,
) -> None:
    mpco, model = _in_process_mpco(tmp_path, transient=transient,
                                   bridge_analyze=bridge_analyze)
    _assert_falls_back(tmp_path, mpco, model, "1 analysis and 0 analyze")
