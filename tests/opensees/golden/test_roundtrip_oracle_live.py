"""R1b oracle, live net (K2c, #1816): a replayed deck must run.

The static declare-before-use check in ``test_roundtrip_oracle.py`` sees
only the references the archive's ``decl_params`` name. It cannot see node
and element tags (connectivity, ``fix``, ``region -node/-ele``, loads,
recorders) or anything inside a ``proc`` body. This module is the second
net: for every replayable corpus model (each one but the S rows), it writes
the bridge's openseespy deck (``ops.py``) and the replayed one
(``from_h5(h5()).build("py")``), and runs each in its own subprocess. It
ran the waived A and B rows until #2194 left none (#2193 cleared the B
rows, #2194 the A rows), so it now runs the corpus the static oracle
proves equal to the bridge deck: the ``py`` target is a second deck that
oracle never compares.

A deck *runs* when the process exits 0 **and** prints the completion
marker appended as its last line: an OpenSees error can still exit 0, so
the marker is what proves the deck reached its end.

- A replayable model: the bridge deck runs, and so does the replay.
- The fatal reorder (#1814): ``_broad_frame``'s replay deck runs, and the
  same deck with ``Parallel 3`` moved above ``Steel01 1`` (the reorder #1814
  fixed, now synthetic) does not, so a C-class reorder really aborts.

A model whose bridge deck does not run on this backend must be one the
archive's solve stamp marks ``requires=("fork",)``; stock openseespy cannot
build it, so there the test asserts only that the bridge deck did not run,
the static check is its only net, and the test skips by name so ``-rs``
shows it.
:data:`_KNOWN_STOCK_FAILURES` names the models whose bridge deck aborts on
stock openseespy although their solve stamp claims ``requires=()``, each
with its reason: #2220 (``Parameter::addComponent`` finds no object for the
parameter; a partitioned stamp that does not name its multi-rank run), or
the fixture fact (``_frame[modal=True]`` declares no supports, so its
``eigen`` cannot run). They are handled like the fork rows: the test asserts
the bridge deck does not run, then skips citing the reason. Once a deck
runs the assertion fails, and the entry must be removed.
Any other bridge deck that does not run fails.

:data:`_STAND_INS` names corpus models the net runs through a stand-in
instead (T ruling on #2195): the golden cell's twin must keep the cell's
``ParallelRCM`` / ``Mumps``, so its stand-in, the same model with
``RCM`` / ``UmfPack`` and a lumped mass, runs in its place. A test pins
that the two decks differ only there and that the twin's deck aborts.

openseespy is never imported into the pytest process: the decks run in
subprocesses, so the gate is ``importlib.util.find_spec`` (#1836).
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel
from tests.opensees.golden.test_roundtrip_oracle import (
    CORPUS,
    Model,
    _move,
    _waivers,
)

pytestmark = pytest.mark.live

_MARKER = "APEGMSH_K2C_DECK_COMPLETE"


_TWIN_ID = "test_h5_decl_params::_broad_partitioned"
_STAND_IN_ID = ("test_h5_decl_params::_two_column_frame_partitioned"
                "[mass=True,stock_solver=True]")
#: Corpus models the net runs through a stand-in (T ruling on #2195). The
#: golden cell's twin must emit the cell's deck, whose ``numberer
#: ParallelRCM`` / ``system Mumps`` no single-process stock build has. Its
#: stand-in is the same model with ``RCM`` / ``UmfPack`` and a lumped mass;
#: :func:`test_a_stand_in_is_its_model_but_for_solver_and_mass` pins that,
#: and that the twin's own bridge deck aborts on that pair.
_STAND_INS: dict[str, str] = {_TWIN_ID: _STAND_IN_ID}
#: Every model whose replay deck must run: the corpus but its S rows (not
#: replayable; ``from_h5`` refuses them by name) and the models a stand-in
#: runs for, in corpus order.
_RUN_ROWS = [m for m in CORPUS
             if (m.id not in _waivers() or _waivers()[m.id].klass != "S")
             and m.id not in _STAND_INS]
#: Models whose bridge deck aborts on stock openseespy while the solve stamp
#: claims ``requires=()``, each with its reason: the issue that tracks it, or
#: the fixture fact that makes it unsolvable. Not a waiver: the net asserts
#: the bridge deck does not run, so a fix fails it until removed.
_FIXTURE_UNSOLVABLE = ("fixture has no supports; eigen cannot run "
                       "(fixture-unsolvable, not a replay fault)")
_KNOWN_STOCK_FAILURES: dict[str, str] = {
    "test_h5_decls::_param_stage_model": "#2220",
    "test_h5_stages_reader::_real_kitchen_sink_bridge": "#2220",
    "test_h5_commands_program::_frame[rayleigh=False,modal=True]":
        _FIXTURE_UNSOLVABLE,
    "test_h5_commands_program::_frame[rayleigh=True,modal=True]":
        _FIXTURE_UNSOLVABLE,
    # The stamp says solve_mode="partitioned" but requires=(): it does not
    # name the multi-rank run its rank-gated deck needs (the #2220 class).
    # In one process rank 1's stage elements are never declared: singular.
    "test_h5_regions_round_trip::_partitioned_staged_with_fan_out": "#2220",
}
#: The model the synthetic fatal reorder (#1814) is cut from.
_FATAL_CASES = [m for m in CORPUS if m.id == "test_h5_decl_params::_broad_frame"]


def _decks(model: Model, tmp: Path) -> tuple[Path, Path, tuple[str, ...]]:
    """The bridge's and the replay's openseespy decks of ``model``, and the
    capabilities its archive's solve stamp says a backend needs. Each deck
    sits in its own directory, its working directory when it runs: the two
    decks write the same output files (a profiler report refuses a run id
    its file already holds). Each directory holds an empty ``out/``: the
    golden recipe's recorders write under it (``out/run.mpco``), OpenSees
    creates no directory, and an MPCO file in a missing one aborts the deck
    with an HDF5 error."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = model.build()
        bridge = tmp / "bridge" / "bridge.py"
        bridge.parent.mkdir()
        (bridge.parent / "out").mkdir()
        ops.py(str(bridge), progress=False)
        archive = tmp / "model.h5"
        ops.h5(str(archive))
        replay = tmp / "replay" / "replay.py"
        replay.parent.mkdir()
        (replay.parent / "out").mkdir()
        model_h5 = OpenSeesModel.from_h5(str(archive))
        model_h5.build("py", out=str(replay))
    stamp = model_h5.solve_stamp
    assert stamp is not None, f"{model.id}: the archive carries no solve stamp"
    return bridge, replay, tuple(stamp.requires)


def _run(deck: Path) -> tuple[bool, str]:
    """Run ``deck`` with a trailing completion marker in a subprocess."""
    text = deck.read_text(encoding="utf-8")
    deck.write_text(f"{text.rstrip()}\nprint({_MARKER!r})\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], capture_output=True, text=True,
        timeout=300, cwd=str(deck.parent),
    )
    out = proc.stdout.strip().splitlines()
    ran = proc.returncode == 0 and bool(out) and out[-1] == _MARKER
    return ran, (f"rc={proc.returncode}\nstdout:\n{proc.stdout[-1500:]}\n"
                 f"stderr:\n{proc.stderr[-1500:]}")


def _openseespy_or_skip() -> None:
    """Skip unless openseespy is installed, without importing it here."""
    if importlib.util.find_spec("openseespy") is None:
        pytest.skip("openseespy is not installed (module 'openseespy' not "
                    "found); the live net runs its decks with it")


@pytest.fixture(autouse=True)
def _openseespy() -> None:
    _openseespy_or_skip()


def _bridge_runs(model: Model, bridge: Path, requires: tuple[str, ...]) -> bool:
    """Whether the bridge deck runs; one that does not must need the fork."""
    ran, log = _run(bridge)
    assert ran or "fork" in requires, (
        f"{model.id}: the bridge deck does not run, and its solve stamp "
        f"requires {requires}:\n{log}")
    return ran


def test_known_stock_failures_name_run_rows() -> None:
    """The table is exactly these five rows, each a row the net runs, each
    citing #2220 or the fixture fact: it can only shrink as a deck is
    fixed, never grow. The S row's twin (#2195) did not grow it: the net
    runs a stand-in for it (:data:`_STAND_INS`)."""
    assert _KNOWN_STOCK_FAILURES == {
        "test_h5_decls::_param_stage_model": "#2220",
        "test_h5_stages_reader::_real_kitchen_sink_bridge": "#2220",
        "test_h5_commands_program::_frame[rayleigh=False,modal=True]":
            _FIXTURE_UNSOLVABLE,
        "test_h5_commands_program::_frame[rayleigh=True,modal=True]":
            _FIXTURE_UNSOLVABLE,
        "test_h5_regions_round_trip::_partitioned_staged_with_fan_out":
            "#2220",
    }
    assert set(_KNOWN_STOCK_FAILURES) <= {m.id for m in _RUN_ROWS}


def test_the_net_runs_some_stock_row() -> None:
    assert _RUN_ROWS and _FATAL_CASES, (
        "the live net would run nothing: no replayable corpus model, or no "
        "fatal-reorder case")
    # Every corpus model but the S rows and the stand-in-run ones: none
    # drops out silently.
    s_rows = {w.model for w in _waivers().values() if w.klass == "S"}
    run = {m.id for m in _RUN_ROWS}
    assert run == {m.id for m in CORPUS} - s_rows - set(_STAND_INS)
    # The ledger has no S row since #2195. The one stand-in is pinned: it
    # runs (no known-failure entry), and it stands in for the twin only.
    assert not s_rows
    assert _STAND_INS == {_TWIN_ID: _STAND_IN_ID}
    assert _STAND_IN_ID in run and _STAND_IN_ID not in _KNOWN_STOCK_FAILURES


def _changed(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    """The lines only ``a`` has and the lines only ``b`` has, in order."""
    gone: list[str] = []
    new: list[str] = []
    for ln in difflib.unified_diff(a, b, lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        (gone if ln[0] == "-" else new).append(ln[1:].strip())
    return gone, new


@pytest.mark.parametrize("twin_id", list(_STAND_INS))
def test_a_stand_in_is_its_model_but_for_solver_and_mass(
        twin_id: str, tmp_path: Path) -> None:
    """The stand-in's bridge deck is the twin's but for the solver pair and
    the ``mass`` lines, and the twin's own bridge deck aborts on this
    backend at that pair. Once a stock build runs it, this fails: delete
    the stand-in and let the net run the twin."""
    by_id = {m.id: m for m in CORPUS}
    twin, stand_in = by_id[twin_id], by_id[_STAND_INS[twin_id]]
    (tmp_path / "twin").mkdir()
    (tmp_path / "stand_in").mkdir()
    t_bridge, _t_replay, _req = _decks(twin, tmp_path / "twin")
    s_bridge, _s_replay, _req = _decks(stand_in, tmp_path / "stand_in")
    gone, new = _changed(t_bridge.read_text(encoding="utf-8").splitlines(),
                         s_bridge.read_text(encoding="utf-8").splitlines())
    assert gone and {ln.split("(")[0] for ln in gone} == {
        "ops.numberer", "ops.system"}
    assert all("'ParallelRCM'" in ln or "'Mumps'" in ln for ln in gone), gone
    assert {ln.split("(")[0] for ln in new} == {
        "ops.numberer", "ops.system", "ops.mass"}
    assert all("'RCM'" in ln or "'UmfPack'" in ln
               for ln in new if not ln.startswith("ops.mass(")), new
    ran, log = _run(t_bridge)
    assert not ran and ("Mumps" in log or "Numberer" in log), (
        f"{twin_id}: the twin's bridge deck no longer aborts on its "
        f"ParallelRCM / Mumps pair; drop its stand-in:\n{log}")


_PROBE_PLUGIN = """import json, os, sys

_OUTCOMES = {}


def pytest_runtest_logreport(report):
    if report.when == "setup" or report.outcome != "passed":
        _OUTCOMES[report.when] = [report.outcome, str(report.longrepr or "")]


def pytest_runtest_teardown(item):
    _OUTCOMES["fixtures"] = sorted(item.fixturenames)


def pytest_sessionfinish(session):
    mods = sorted(k for k in sys.modules
                  if k.split(".")[0] in ("openseespy", "opensees"))
    with open(os.environ["APEGMSH_K_PROBE_OUT"], "w") as f:
        json.dump({"modules": mods, "outcomes": _OUTCOMES}, f)
"""


def test_the_net_never_imports_openseespy_in_process(tmp_path: Path) -> None:
    """Running a test of this module, autouse gate included, leaves
    openseespy out of ``sys.modules`` (#1836). A fresh ``pytest`` process,
    because this one may already hold openseespy from another live test; a
    probe plugin reports the modules at session end. Where openseespy is
    absent the gate skips, and ``sys.modules`` must still be clean."""
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "k_probe_plugin.py").write_text(_PROBE_PLUGIN, encoding="utf-8")
    out = tmp_path / "probe.json"
    target = (f"{Path(__file__).resolve().relative_to(root).as_posix()}"
              "::test_the_net_runs_some_stock_row")
    env = {**os.environ, "LADRUNO_OPENSEES_QUIET": "1",
           "APEGMSH_K_PROBE_OUT": str(out),
           "PYTHONPATH": os.pathsep.join(
               [str(tmp_path), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    # A one-test inner run never selects the warrant rows, so the lane hook would refuse it (rc=4).
    env.pop("APEGMSH_WARRANT_LANE", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-m", "live",
         "-p", "k_probe_plugin", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=300, cwd=str(root), env=env,
    )
    log = f"rc={proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    assert proc.returncode == 0, log
    probe = json.loads(out.read_text(encoding="utf-8"))
    outcomes = probe["outcomes"]
    assert "_openseespy" in outcomes["fixtures"], log
    setup, why = outcomes["setup"]
    assert setup == "passed" or (
        setup == "skipped" and "openseespy is not installed" in why), log
    assert probe["modules"] == [], log


@pytest.mark.parametrize("model", _RUN_ROWS, ids=[m.id for m in _RUN_ROWS])
def test_a_replay_runs_where_the_bridge_deck_runs(
        model: Model, tmp_path: Path) -> None:
    bridge, replay, requires = _decks(model, tmp_path)
    reason = _KNOWN_STOCK_FAILURES.get(model.id)
    if reason is not None:
        ran, log = _run(bridge)
        assert not ran, (
            f"{model.id}: the bridge deck now runs ({reason} is fixed), so "
            f"remove its _KNOWN_STOCK_FAILURES entry:\n{log}")
        pytest.skip(f"{model.id}: known stock failure: {reason} (the bridge "
                    "deck aborts though the stamp requires nothing); "
                    "asserted that the bridge deck did not run here")
    if not _bridge_runs(model, bridge, requires):
        pytest.skip(f"{model.id}: fork-only (solve stamp requires={requires!r}); "
                    "asserted only that the bridge deck did not run here")
    ran, log = _run(replay)
    assert ran, (
        f"{model.id}: the bridge deck runs but the replay deck does not: a "
        "semantic difference the static oracle missed; track it in an "
        f"issue:\n{log}")


@pytest.mark.parametrize("model", _FATAL_CASES, ids=[m.id for m in _FATAL_CASES])
def test_a_fatal_reorder_replay_does_not_run(
        model: Model, tmp_path: Path) -> None:
    bridge, replay, requires = _decks(model, tmp_path)
    assert "fork" not in requires, f"{model.id}: pin a stock-runnable row here"
    ran, log = _run(bridge)
    assert ran, f"{model.id}: the bridge deck does not run:\n{log}"
    fatal = tmp_path / "fatal.py"
    fatal.write_text(_move(
        replay.read_text(encoding="utf-8"), "ops.uniaxialMaterial('Parallel', 3,",
        above="ops.uniaxialMaterial('Steel01', 1,"), encoding="utf-8")
    ran, log = _run(replay)
    assert ran, f"{model.id}: the replay deck does not run (#1814):\n{log}"
    ran, log = _run(fatal)
    assert not ran, (
        f"{model.id}: the deck with Parallel 3 above Steel01 1 still runs; "
        "the live net no longer proves a fatal reorder aborts")
