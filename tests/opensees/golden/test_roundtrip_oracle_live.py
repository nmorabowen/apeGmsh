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

The archive's solve stamp says what a backend needs (``/opensees@requires``,
:data:`~apeGmsh.opensees.emitter.caps.REQUIRES_TOKENS`), and the net reads
it the way ``build('live')`` does (:func:`unmet_requires`, #2372):

- ``"fork"`` or ``"mpco"`` unmet (stock openseespy: no fork commands, no
  MPCO recorder): the bridge deck must abort, an ``"mpco"`` one at the
  MPCO recorder; then the test skips by name, citing the tokens.
- ``"mpi"`` unmet (always, in one process; a partitioned archive, or a
  Mumps-family or ``Parallel*`` ``system``): a rank-gated bridge deck run
  in one process proves nothing either way, so it is not judged. The flat
  replay re-derives its own need (:meth:`OpenSeesModel.replay_requires`):
  when a partitioned mode was the only reason, the replay must run;
  otherwise the test skips by name, citing the token.
- Nothing unmet: the stamp claims this backend solves the model, so a
  bridge deck that aborts **fails** (#2220).

:data:`_KNOWN_STOCK_FAILURES` names the models whose bridge deck aborts
although the stamp claims nothing is unmet, each with the fixture fact that
makes it unsolvable on every build. Not a waiver: the net asserts the
bridge deck does not run, and that the replay fails the same way (at the
same stage and increment for an ``analyze`` failure, at the same command
otherwise), then skips citing the reason. Once a deck runs the assertion
fails, and the entry must be removed.

openseespy is never imported into the pytest process: the decks run in
subprocesses, so the gate is ``importlib.util.find_spec`` (#1836), and the
backend kind is asked of a subprocess too.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.opensees_model import unmet_requires
from tests.opensees.golden.test_roundtrip_oracle import (
    CORPUS,
    Model,
    _move,
    _waivers,
)
from tests.opensees.h5.test_h5_solve_stamp import _SYSTEM_TOKENS

pytestmark = pytest.mark.live

_MARKER = "APEGMSH_K2C_DECK_COMPLETE"

#: Every model whose replay deck must run: the corpus but its S rows (not
#: replayable; ``from_h5`` refuses them by name), in corpus order.
_RUN_ROWS = [m for m in CORPUS
             if m.id not in _waivers() or _waivers()[m.id].klass != "S"]
#: Models whose bridge deck aborts on every build while the solve stamp
#: claims nothing a backend lacks, each with the fixture fact that makes it
#: unsolvable (the #2220 diagnosis: a stock and a fork run both abort
#: there). Not a waiver: the net asserts the bridge deck does not run and
#: that the replay fails the same way, so a fix fails it until removed.
#: Shrink-only.
_FIXTURE_UNSOLVABLE = ("fixture has no supports; eigen cannot run "
                       "(fixture-unsolvable, not a replay fault)")
_KITCHEN_SINK_SINGULAR = (
    "fixture removes the Rock quad and the base x-fix, so the Fill quad "
    "is singular on stock and fork alike (fixture-unsolvable, not a "
    "replay fault; the fixture fix is #2376)")
_KNOWN_STOCK_FAILURES: dict[str, str] = {
    "test_h5_commands_program::_frame[rayleigh=False,modal=True]":
        _FIXTURE_UNSOLVABLE,
    "test_h5_commands_program::_frame[rayleigh=True,modal=True]":
        _FIXTURE_UNSOLVABLE,
    "test_h5_stages_reader::_real_kitchen_sink_bridge":
        _KITCHEN_SINK_SINGULAR,
}
#: One staged frame per LinearSystem member, solving one increment with
#: that ``system`` (#2429): the net runs each deck on the bound build, so
#: the token table ``_SYSTEM_TOKENS`` pins is proven on stock and the fork.
_SYSTEM_ROWS = [Model("test_h5_solve_stamp", "_system_frame",
                      (("name", n), ("staged", True)))
                for n in sorted(_SYSTEM_TOKENS)]
#: The model the synthetic fatal reorder (#1814) is cut from.
_FATAL_CASES = [m for m in CORPUS if m.id == "test_h5_decl_params::_broad_frame"]
#: What a build without the MPCO recorder logs when a deck declares one.
_MPCO_REFUSAL = "recorder type mpco is unknown"
#: The bridge's own abort line when an ``analyze`` fails (``stage_emit``).
_ANALYZE_FAILED = re.compile(
    r"apeGmsh: analyze FAILED at increment \d+/\d+ of stage '[^']*'")


def _decks(model: Model, tmp: Path) -> tuple[Path, Path, tuple[str, ...]]:
    """The bridge's and the replay's openseespy decks of ``model``, and the
    capabilities its archive's solve stamp says a backend needs. Each deck
    sits in its own directory, its working directory when it runs: the two
    decks write the same output files (a profiler report refuses a run id
    its file already holds)."""
    bridge, replay, requires, _flat = _decks_and_replay_needs(model, tmp)
    return bridge, replay, requires


def _decks_and_replay_needs(
    model: Model, tmp: Path,
) -> tuple[Path, Path, tuple[str, ...], tuple[str, ...]]:
    """:func:`_decks`, and what the flat replay deck needs
    (:meth:`OpenSeesModel.replay_requires`: a partitioned mode's ``"mpi"``
    does not apply to it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = model.build()
        bridge = tmp / "bridge" / "bridge.py"
        bridge.parent.mkdir()
        ops.py(str(bridge), progress=False)
        archive = tmp / "model.h5"
        ops.h5(str(archive))
        replay = tmp / "replay" / "replay.py"
        replay.parent.mkdir()
        model_h5 = OpenSeesModel.from_h5(str(archive))
        model_h5.build("py", out=str(replay))
    stamp = model_h5.solve_stamp
    assert stamp is not None, f"{model.id}: the archive carries no solve stamp"
    return bridge, replay, tuple(stamp.requires), model_h5.replay_requires()


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


def _failure(log: str) -> str:
    """Where a deck that did not run stopped: the bridge's ``analyze
    FAILED`` line (stage and increment), else the statement the traceback
    names and the exception it raised, else the log's last line."""
    hit = _ANALYZE_FAILED.search(log)
    if hit:
        return hit.group(0)
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, 0, -1):
        if lines[i - 1].startswith("File ") and ", line " in lines[i - 1]:
            return f"{lines[i]} -> {lines[-1]}"
    return lines[-1] if lines else ""


def _openseespy_or_skip() -> None:
    """Skip unless openseespy is installed, without importing it here."""
    if importlib.util.find_spec("openseespy") is None:
        pytest.skip("openseespy is not installed (module 'openseespy' not "
                    "found); the live net runs its decks with it")


@pytest.fixture(autouse=True)
def _openseespy() -> None:
    _openseespy_or_skip()


_KIND_PROBE = """import openseespy.opensees as o
try:
    b = o.ladrunoBuild()
except Exception:
    b = None
print("fork" if isinstance(b, str) and b else "stock")
"""


@pytest.fixture(scope="module")
def backend_kind() -> str:
    """The kind of openseespy the decks run with (``"fork"`` when
    ``ladrunoBuild()`` answers a sha, as ``get_backend_info`` reads it),
    asked of a subprocess so this process never imports it."""
    _openseespy_or_skip()
    proc = subprocess.run([sys.executable, "-c", _KIND_PROBE],
                          capture_output=True, text=True, timeout=120)
    out = proc.stdout.strip().splitlines()
    assert proc.returncode == 0 and out and out[-1] in ("fork", "stock"), (
        f"backend probe failed: rc={proc.returncode}\n{proc.stdout[-800:]}\n"
        f"{proc.stderr[-800:]}")
    return out[-1]


def _bridge_runs(model: Model, bridge: Path, requires: tuple[str, ...],
                 kind: str) -> bool:
    """Whether the net judges ``model`` here: ``False`` (the caller skips)
    when the stamp names a token this backend does not meet, after
    asserting the bridge deck aborts where the token says it must; else
    the bridge deck must run, since the stamp claims it does (#2220)."""
    unmet = unmet_requires(requires, kind)
    if unmet:
        if "mpi" in unmet and "mpco" not in unmet:
            return False  # one process cannot judge a multi-rank deck
        ran, log = _run(bridge)
        assert not ran, (
            f"{model.id}: the bridge deck runs on this {kind} build although "
            f"its stamp requires {unmet!r}, which it lacks: the stamp "
            f"over-claims\n{log}")
        if "mpco" in unmet:
            assert _MPCO_REFUSAL in log, (
                f"{model.id}: the stamp requires 'mpco', but the bridge deck "
                f"did not abort at its MPCO recorder:\n{log}")
        return False
    ran, log = _run(bridge)
    assert ran, (
        f"{model.id}: the bridge deck does not run on this {kind} build, and "
        f"its solve stamp requires {requires!r}, nothing it lacks: the stamp "
        f"claims a solve this backend cannot do (#2220):\n{log}")
    return True


def test_known_stock_failures_name_run_rows() -> None:
    """The table is exactly these three rows, each a row the net runs,
    each citing its fixture fact: it can only shrink as a fixture is
    fixed, never grow. #2372 removed the #2220 rows: two now stamp what
    their deck needs, one fixture was fixed, and the kitchen sink names
    its fixture fact (its fix is #2376)."""
    assert _KNOWN_STOCK_FAILURES == {
        "test_h5_commands_program::_frame[rayleigh=False,modal=True]":
            _FIXTURE_UNSOLVABLE,
        "test_h5_commands_program::_frame[rayleigh=True,modal=True]":
            _FIXTURE_UNSOLVABLE,
        "test_h5_stages_reader::_real_kitchen_sink_bridge":
            _KITCHEN_SINK_SINGULAR,
    }
    assert set(_KNOWN_STOCK_FAILURES) <= {m.id for m in _RUN_ROWS}
    assert not any("#2220" in r for r in _KNOWN_STOCK_FAILURES.values())


def test_the_net_runs_some_stock_row() -> None:
    assert _RUN_ROWS and _FATAL_CASES, (
        "the live net would run nothing: no replayable corpus model, or no "
        "fatal-reorder case")
    # Every corpus model but the S rows: none drops out silently. The
    # ledger has no S row since #2195.
    s_rows = {w.model for w in _waivers().values() if w.klass == "S"}
    assert {m.id for m in _RUN_ROWS} == {m.id for m in CORPUS} - s_rows
    assert not s_rows


def test_the_failure_reads_stage_and_increment_or_the_command() -> None:
    """:func:`_failure` on the two log shapes the net compares."""
    analyze = ("WARNING ...\nOpenSees > analyze failed\napeGmsh: analyze "
               "FAILED at increment 1/2 of stage 'mutate' (pseudo-time 0) -- "
               "aborting")
    assert _failure(analyze) == (
        "apeGmsh: analyze FAILED at increment 1/2 of stage 'mutate'")
    command = ('Traceback (most recent call last):\n  File "x.py", line 10, '
               "in <module>\n    ops.eigen('-genBandArpack', 2)\n"
               "opensees.OpenSeesError: See stderr output\n")
    assert _failure(command) == ("ops.eigen('-genBandArpack', 2) -> "
                                 "opensees.OpenSeesError: See stderr output")
    assert _failure(analyze.replace("1/2", "2/2")) != _failure(analyze)


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
        model: Model, tmp_path: Path, backend_kind: str) -> None:
    bridge, replay, requires, flat = _decks_and_replay_needs(model, tmp_path)
    reason = _KNOWN_STOCK_FAILURES.get(model.id)
    if reason is not None:
        assert not unmet_requires(requires, backend_kind), (
            f"{model.id}: a known failure must be one the stamp says this "
            f"backend solves; it requires {requires!r}")
        ran, log = _run(bridge)
        assert not ran, (
            f"{model.id}: the bridge deck now runs ({reason} is fixed), so "
            f"remove its _KNOWN_STOCK_FAILURES entry:\n{log}")
        ran_r, log_r = _run(replay)
        assert not ran_r and _failure(log_r) == _failure(log), (
            f"{model.id}: the bridge deck fails at {_failure(log)!r} but the "
            f"replay {'runs' if ran_r else f'fails at {_failure(log_r)!r}'}: "
            f"a semantic difference the static oracle missed:\n{log_r}")
        pytest.skip(f"{model.id}: known failure: {reason}; asserted that the "
                    "bridge deck did not run and that the replay fails at "
                    f"the same place ({_failure(log)})")
    judged = _bridge_runs(model, bridge, requires, backend_kind)
    unmet_flat = unmet_requires(flat, backend_kind)
    if not judged and unmet_flat:
        unmet = unmet_requires(requires, backend_kind)
        pytest.skip(f"{model.id}: the solve stamp requires {unmet!r}, which "
                    f"this {backend_kind} build in one process lacks "
                    f"(requires={requires!r}, flat replay {flat!r}); "
                    + ("a rank-gated deck is not judged in one process"
                       if unmet == ("mpi",) else
                       "asserted only that the bridge deck did not run here"))
    # The bridge deck ran, or only its partitioned mode's "mpi" kept it
    # from being judged: the flat replay needs nothing this backend lacks,
    # so it must run (#2372 F3).
    ran, log = _run(replay)
    assert ran, (
        f"{model.id}: the "
        + ("bridge deck runs" if judged else
           f"stamp requires {requires!r} but the flat replay {flat!r}, "
           "nothing this backend lacks,")
        + " but the replay deck does not run: a semantic difference the "
        f"static oracle missed; track it in an issue:\n{log}")


@pytest.mark.parametrize("model", _SYSTEM_ROWS, ids=[m.id for m in _SYSTEM_ROWS])
def test_each_system_deck_runs_iff_its_stamp_says_so(
        model: Model, tmp_path: Path, backend_kind: str) -> None:
    """A ``system`` member's bridge deck and its replay run on this build
    exactly when the stamp names nothing it lacks; one that does not run
    stops at its ``system`` command. ``Pardiso`` aborts on stock and runs
    on the fork (#2393); every member without ``"mpi"`` runs on both. A
    member that needs ``"mpi"`` (``Mumps``, ``ParallelProfileSPD``) is not
    judged: one process is not an MPI run, and the bridge refuses a serial
    ``Mumps`` deck before writing it."""
    name = dict(model.kwargs)["name"]
    if "mpi" in _SYSTEM_TOKENS[name]:
        pytest.skip(f"{model.id}: system {name} needs an MPI run, which one "
                    "process is not; not judged here")
    bridge, replay, requires, flat = _decks_and_replay_needs(model, tmp_path)
    assert requires == flat == _SYSTEM_TOKENS[name]
    unmet = unmet_requires(requires, backend_kind)
    for deck in (bridge, replay):
        ran, log = _run(deck)
        assert ran == (not unmet), (
            f"{model.id}: the {deck.stem} deck "
            + ("runs" if ran else f"fails at {_failure(log)!r}")
            + f" on this {backend_kind} build, but its stamp requires "
            f"{requires!r} (unmet here: {unmet!r}):\n{log}")
        if not ran:
            assert f"ops.system({name!r}" in _failure(log), (
                f"{model.id}: the {deck.stem} deck did not stop at its "
                f"system command:\n{log}")


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
