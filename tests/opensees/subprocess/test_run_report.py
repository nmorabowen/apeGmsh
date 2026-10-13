"""R3-S6 (#1790, ruling D2 on #1372): ``ops.py`` / ``ops.tcl(path,
run=True)`` return a frozen ``RunReport``; a failed run still raises.

ADR 0106 Amendment 1. Real decks, real binaries: the openseespy deck
runs through the resolved python (``$OPENSEES_VENV`` or this one), the
Tcl deck through the resolved ``OpenSees`` binary. The oracles:

- ``increments`` equals the analyze increments the deck drove, and the
  tip deflection read back with ``ops.results(path)`` is that many load
  steps of ``PL^3/3EI`` (``LoadControl(dlam=1.0)``, linear);
- an aborting deck (no support: a singular stiffness) raises on both
  routes. On Tcl ``OpenSees.exe`` exits 0 after the deck's ``error``, so
  only the missing completion line can fail it.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._run import RunReport
from apeGmsh.opensees._target import resolve_opensees_binary
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.live.test_readback_live import (
    E,
    I_BEAM,
    L_BEAM,
    P_TIP,
    build_cantilever_fem,
    cantilever_bridge,
)

pytestmark = pytest.mark.subprocess

UY = -P_TIP * L_BEAM ** 3 / (3.0 * E * I_BEAM)


def _have_tcl_binary() -> bool:
    try:
        resolve_opensees_binary(None, None)
    except Exception:
        return False
    return True


needs_tcl = pytest.mark.skipif(
    not _have_tcl_binary(), reason="no OpenSees binary on this machine")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    pytest.importorskip("openseespy.opensees")
    return build_cantilever_fem()


def _record_tip(ops: apeSees) -> None:
    ops.recorder.Node(file="tip.out", label="tip", dofs=(2,),
                      response="disp", time_format="dt")


def _unsupported(fem: Any) -> apeSees:
    """The cantilever without its support: the first analyze fails."""
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=1e-2, E=E,
                                  Iz=I_BEAM)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=(0.0, -P_TIP, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _assert_two_increment_report(
    ops: apeSees, rep: Any, deck: str, tmp_path: Path,
) -> None:
    assert isinstance(rep, RunReport)
    assert rep.returncode == 0
    assert rep.increments == 2
    assert rep.solver_stats is None          # no -stats asked
    assert rep.elapsed_s > 0.0
    assert rep.log == (tmp_path / (Path(deck).stem + ".log")).resolve()
    assert rep.log.is_file()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.increments = 3                   # type: ignore[misc]
    # The R2-c read still works after the run, and agrees with the count:
    # two steps of dlam=1.0 put twice the load on the tip.
    # The recorder wrote one row per increment.
    tip = ops.results(deck).nodes.get(label="tip", component="displacement_y")
    np.testing.assert_allclose(tip.time, [1.0, 2.0])
    np.testing.assert_allclose(tip.values.ravel(), [UY, 2.0 * UY], rtol=1e-9)


def test_py_run_returns_a_report_and_the_read_still_works(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    _record_tip(ops)
    rep = ops.py("deck.py", run=True, analyze_steps=2, python=sys.executable)
    _assert_two_increment_report(ops, rep, "deck.py", tmp_path)


@needs_tcl
def test_tcl_run_returns_a_report_and_the_read_still_works(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    _record_tip(ops)
    rep = ops.tcl("deck.tcl", run=True, analyze_steps=2)
    _assert_two_increment_report(ops, rep, "deck.tcl", tmp_path)


def test_throttled_markers_still_count_every_increment(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """45 increments print a marker every 2nd one plus the last: the
    count is the loop's ``n``, not the number of markers."""
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    rep = ops.py("deck.py", run=True, analyze_steps=45,
                 python=sys.executable)
    assert rep is not None and rep.increments == 45
    markers = [ln for ln in rep.log.read_text(encoding="utf-8").splitlines()
               if ln.startswith("APEGMSH_PROGRESS")]
    assert len(markers) == 23


def test_without_progress_markers_increments_is_not_established(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    rep = ops.py("deck.py", run=True, analyze_steps=2, progress=False,
                 python=sys.executable)
    assert rep is not None
    assert rep.returncode == 0
    assert rep.increments is None


def test_run_false_still_returns_none(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    assert ops.py("deck.py", analyze_steps=2) is None
    assert ops.tcl("deck.tcl", analyze_steps=2) is None


def test_an_aborting_py_deck_raises(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = _unsupported(cantilever)
    with pytest.raises(RuntimeError, match=r"openseespy subprocess returned"):
        ops.py("deck.py", run=True, analyze_steps=2, python=sys.executable)
    assert "analyze FAILED" in (tmp_path / "deck.log").read_text(
        encoding="utf-8")
    with pytest.raises(BridgeError, match=r"did not finish"):
        ops.results("deck.py")


@needs_tcl
def test_an_aborting_tcl_deck_raises_although_it_exits_0(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = _unsupported(cantilever)
    with pytest.raises(RuntimeError, match=r"exited 0 but the deck did not "
                                           r"run to its end"):
        ops.tcl("deck.tcl", run=True, analyze_steps=2)
    assert "analyze FAILED" in (tmp_path / "deck.log").read_text(
        encoding="utf-8")
    with pytest.raises(BridgeError, match=r"did not finish"):
        ops.results("deck.tcl")
