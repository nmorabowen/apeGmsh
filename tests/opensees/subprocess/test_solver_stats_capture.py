"""ADR 0106 S3 — ``system Pardiso -stats`` capture through ``stream_run``.

The wiring half of ADR 0106: :func:`stream_run` feeds every tee'd line to
:func:`parse_solver_stats` when the deck asked for statistics, and hands
the finished :class:`RunSolverStats` back through ``apeSees.tcl`` /
``apeSees.py``.

Driven with a fake child process — a tiny throwaway python script run via
``sys.executable`` (the ADR 0095 S12a fixture idiom, same as
``tests/opensees/unit/test_run_streaming.py``) printing a canned
interleaved log. No OpenSees binary, and no PARDISO, is needed to prove
the wiring: the contract is over the *text* the fork prints.

Covers INV-1 (a deck that did not ask does not parse at all), INV-2 (the
tee stays byte-identical to the child's output), INV-6 (a requested block
that never appears warns once, naming ``TIMS_FORK_BATCH_MIN_BUILD``) and
INV-7 (a failed run's log re-parses to the same record).
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._run import SolverStatsWarning, stream_run
from apeGmsh.opensees._solver_stats import RunSolverStats, parse_solver_stats
from apeGmsh.opensees._target import TIMS_FORK_BATCH_MIN_BUILD

from tests.opensees.fixtures.fem_stub import make_two_node_beam

pytestmark = pytest.mark.subprocess


def _block(
    *,
    n: int = 54,
    nnz: int = 1458,
    mtype: int = 2,
    threads: int = 8,
    entries: int,
    peak: int,
    perm: int,
    fact: int,
    mflops: float,
) -> str:
    """One ``PARDISO stats:`` block, labels exactly as fork PR #821 prints."""
    return (
        f"PARDISO stats: n={n} nnz(A)={nnz} matrixType={mtype} "
        f"threads={threads}\n"
        f"  factor entries iparm(18)  = {entries}\n"
        f"  peak memory KB iparm(15)  = {peak}\n"
        f"  perm memory KB iparm(16)  = {perm}\n"
        f"  fact memory KB iparm(17)  = {fact}\n"
        f"  factor Mflops  iparm(19)  = {mflops}"
    )


#: A two-stage run: ``gravity`` refactorises twice (its worst
#: factorisation is the *middle* one on two of the five capacity
#: numbers, which is what the D6 maximum exists for), ``push it`` once
#: (name with a space), plus one block after the last stage closed that
#: belongs to the run-level bucket.
_INTERLEAVED = "\n".join([
    "OpenSees 3.8.0",
    "APEGMSH_STAGE open gravity",
    _block(entries=1836, peak=100, perm=20, fact=300, mflops=0.1),
    "APEGMSH_PROGRESS i=1 n=2 t=1",
    _block(entries=2000, peak=150, perm=20, fact=280, mflops=0.3),
    "APEGMSH_STAGE close gravity",
    "APEGMSH_STAGE open push it",
    _block(n=60, nnz=1600, mtype=11, threads=4,
           entries=900, peak=50, perm=10, fact=120, mflops=0.05),
    "APEGMSH_STAGE close push it",
    _block(entries=42, peak=7, perm=3, fact=9, mflops=0.01),
    "Analysis complete",
])

_NO_BLOCK = "\n".join([
    "OpenSees 3.8.0",
    "APEGMSH_STAGE open gravity",
    "PARDISO: 8 threads",
    "APEGMSH_STAGE close gravity",
    "Analysis complete",
])


def _child(tmp_path: Path, text: str, *, rc: int = 0, name: str = "deck.py") -> str:
    """A fake child that prints *text* verbatim and exits with *rc*."""
    body = (
        "import sys\n"
        f"sys.stdout.write({text!r})\n"
        'sys.stdout.write("\\n")\n'
        f"sys.exit({rc!r})\n"
    )
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def _assert_interleaved(rec: RunSolverStats) -> None:
    assert rec.factorisations == 4
    assert rec.malformed_blocks == 0

    by_name = {s.name: s for s in rec.stages}
    assert set(by_name) == {"gravity", "push it"}

    grav = by_name["gravity"]
    assert grav.factorisations == 2
    # max on the capacity numbers, independently (D6) — peak comes from
    # the second block, fact memory from the first.
    assert grav.max_factor_entries == 2000
    assert grav.max_peak_memory_kb == 150
    assert grav.max_perm_memory_kb == 20
    assert grav.max_fact_memory_kb == 300
    assert grav.max_mflops == pytest.approx(0.3)
    # last seen on the identity numbers
    assert (grav.n, grav.nnz_a, grav.matrix_type, grav.threads) == (54, 1458, 2, 8)

    push = by_name["push it"]
    assert push.factorisations == 1
    assert push.max_factor_entries == 900
    assert (push.n, push.nnz_a, push.matrix_type, push.threads) == (60, 1600, 11, 4)

    # the block after the last close lands in the run-level bucket (INV-4)
    assert rec.run_level.factorisations == 1
    assert rec.run_level.max_factor_entries == 42


# --- the streamed record --------------------------------------------------

def test_streamed_record_matches_the_canned_stages(tmp_path: Path) -> None:
    log = str(tmp_path / "run.log")
    rec = stream_run(
        [sys.executable, _child(tmp_path, _INTERLEAVED)],
        log_path=log, verbose=False, label="run -> deck.py",
        expect_solver_stats=True,
    )
    assert isinstance(rec, RunSolverStats)
    _assert_interleaved(rec)


# --- INV-1: no request, no parse ------------------------------------------

def test_without_stats_returns_none_and_never_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def _spy(source: object) -> RunSolverStats:
        calls.append(source)
        raise AssertionError("parse_solver_stats must not run (INV-1)")

    monkeypatch.setattr("apeGmsh.opensees._run.parse_solver_stats", _spy)
    log = str(tmp_path / "run.log")
    rec = stream_run(
        [sys.executable, _child(tmp_path, _INTERLEAVED)],
        log_path=log, verbose=False, label="run -> deck.py",
    )
    assert rec is None
    assert calls == []


# --- INV-2: the tee is byte-identical -------------------------------------

@pytest.mark.parametrize("expect", [False, True])
def test_tee_matches_the_child_output_byte_for_byte(
    tmp_path: Path, expect: bool
) -> None:
    deck = _child(tmp_path, _INTERLEAVED)
    reference = subprocess.run(
        [sys.executable, deck],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout

    log = tmp_path / "run.log"
    stream_run(
        [sys.executable, deck],
        log_path=str(log), verbose=False, label="run -> deck.py",
        expect_solver_stats=expect,
    )
    assert log.read_bytes() == reference


# --- INV-6: requested but absent ------------------------------------------

def test_missing_block_warns_once_naming_the_constant(tmp_path: Path) -> None:
    log = str(tmp_path / "run.log")
    with pytest.warns(SolverStatsWarning) as caught:
        rec = stream_run(
            [sys.executable, _child(tmp_path, _NO_BLOCK)],
            log_path=log, verbose=False, label="run -> deck.py",
            expect_solver_stats=True,
        )
    assert isinstance(rec, RunSolverStats)
    assert rec.factorisations == 0

    hits = [w for w in caught if issubclass(w.category, SolverStatsWarning)]
    assert len(hits) == 1
    message = str(hits[0].message)
    assert "TIMS_FORK_BATCH_MIN_BUILD" in message
    assert TIMS_FORK_BATCH_MIN_BUILD in message


def test_present_block_does_not_warn(tmp_path: Path) -> None:
    log = str(tmp_path / "run.log")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream_run(
            [sys.executable, _child(tmp_path, _INTERLEAVED)],
            log_path=log, verbose=False, label="run -> deck.py",
            expect_solver_stats=True,
        )
    assert not [
        w for w in caught if issubclass(w.category, SolverStatsWarning)
    ]


# --- INV-7: a failed run's numbers survive on disk -------------------------

def test_failed_run_log_reparses_to_the_same_record(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    with pytest.raises(RuntimeError):
        stream_run(
            [sys.executable, _child(tmp_path, _INTERLEAVED, rc=1)],
            log_path=str(log), verbose=False, label="run -> deck.py",
            expect_solver_stats=True,
        )
    assert log.exists()
    _assert_interleaved(parse_solver_stats(str(log)))


# --- the bridge ------------------------------------------------------------

def _fake_binary(tmp_path: Path, text: str) -> str:
    """A launcher that behaves like an ``OpenSees`` binary for ``bin=``.

    ``apeSees.tcl(run=True)`` invokes ``[binary, deck]``, so the fake has
    to be directly executable: a ``.cmd`` shim on Windows, a ``sh``
    script elsewhere. It ignores the deck it is handed and prints the
    canned log.
    """
    script = _child(tmp_path, text, name="fake_opensees_impl.py")
    if os.name == "nt":
        launcher = tmp_path / "fake_opensees.cmd"
        launcher.write_text(
            f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8"
        )
    else:
        launcher = tmp_path / "fake_opensees.sh"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    return str(launcher)


def test_tcl_run_returns_the_record(tmp_path: Path) -> None:
    """``apeSees.tcl(run=True)`` with ``Pardiso(stats=True)`` returns it."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.Pardiso(stats=True)
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    rec = ops.tcl(
        str(tmp_path / "model.tcl"),
        run=True,
        bin=_fake_binary(tmp_path, _INTERLEAVED),
    )
    assert isinstance(rec, RunSolverStats)
    _assert_interleaved(rec)


def test_tcl_without_stats_returns_none(tmp_path: Path) -> None:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.Pardiso()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    assert ops.tcl(
        str(tmp_path / "model.tcl"),
        run=True,
        bin=_fake_binary(tmp_path, _INTERLEAVED),
    ) is None
