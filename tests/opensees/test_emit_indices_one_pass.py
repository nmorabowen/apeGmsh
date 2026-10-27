"""``_emit_indices`` joins rows to emit indices in one pass (#2544).

``emit_index_of`` scans every program run for each row. Called once per
row, that is quadratic when each constraint is its own call, and it made
four ``tests/test_phase_3f_1.py`` cases (50,001 ``equalDOF`` rows through
``Assembly.bridge``) cost 140 s each on a slow CI runner. The batch join
must give exactly what the per-row join gives, and stay linear.
"""
from __future__ import annotations

import random
import time

import pytest

from apeGmsh.opensees.emitter.h5_reader import ProgramRun, emit_index_of
from apeGmsh.opensees.opensees_model import _emit_indices


def _run(first, count, method, row, stage=-1):
    return ProgramRun(first=first, count=count, method=method,
                      store=method if row >= 0 else "", row=row,
                      stage=stage, decl=-1)


def _program(rng: random.Random, n_rows: int) -> list[ProgramRun]:
    """Runs of ``equalDOF`` covering ``n_rows`` rows in chunks, mixed with
    other verbs, staged runs, ledger runs and partition replicas."""
    runs: list[ProgramRun] = []
    first = 1
    row = 0
    while row < n_rows:
        count = rng.randint(1, 4)
        runs.append(_run(first, count, "equalDOF", row))
        first += count
        if rng.random() < 0.3:  # a partition replica repeats the rows
            runs.append(_run(first, count, "equalDOF", row))
            first += count
        if rng.random() < 0.3:
            runs.append(_run(first, 2, "rigidLink", row))
            first += 2
        if rng.random() < 0.2:
            runs.append(_run(first, count, "equalDOF", row, stage=1))
            first += count
        if rng.random() < 0.2:
            runs.append(_run(first, 1, "equalDOF", -1))  # a ledger call
            first += 1
        row += count
    rng.shuffle(runs)  # first-run-wins must follow program order, not row order
    return runs


@pytest.mark.parametrize("seed", range(20))
def test_batch_join_equals_the_per_row_join(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    program = _program(rng, n)
    for method in ("equalDOF", "rigidLink"):
        covered = [i for i in range(n)
                   if any(r.method == method and r.stage == -1
                          and 0 <= r.row <= i < r.row + r.count
                          for r in program)]
        if len(covered) != n:
            with pytest.raises(LookupError):
                _emit_indices(program, method, n)
            continue
        assert _emit_indices(program, method, n) == [
            emit_index_of(program, method, i) for i in range(n)]


def test_a_row_no_run_covers_raises_the_same_lookup_error() -> None:
    program = [_run(1, 2, "equalDOF", 0)]
    with pytest.raises(LookupError, match="row 2 in stage -1"):
        _emit_indices(program, "equalDOF", 3)


def test_one_call_per_row_is_linear() -> None:
    """60,000 single-row runs: the per-row scan needs ~1.8e9 run visits
    (minutes); one pass needs 60,000 (well under a second)."""
    n = 60_000
    program = [_run(i + 1, 1, "equalDOF", i) for i in range(n)]
    t0 = time.perf_counter()
    got = _emit_indices(program, "equalDOF", n)
    elapsed = time.perf_counter() - t0
    assert got == list(range(1, n + 1))
    assert elapsed < 10.0, f"{elapsed:.1f} s: the join is not linear"


# The callers must use the one-pass join, not the per-row scan: with the
# per-row ``emit_index_of`` made to raise, both still answer on a program
# that covers every row (a reverted call site raises here).


def _forbid_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from apeGmsh.opensees.emitter import h5_reader

    def _per_row(*_a, **_k):
        raise AssertionError("per-row emit_index_of scan in a caller")

    monkeypatch.setattr(h5_reader, "emit_index_of", _per_row)


def test_mp_rows_in_bridge_order_uses_the_one_pass_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.opensees.opensees_model import _mp_rows_in_bridge_order

    program = [_run(1, 2, "equalDOF", 0), _run(3, 1, "rigidLink", 0)]
    _forbid_per_row(monkeypatch)
    got = _mp_rows_in_bridge_order(
        [["e0", "e1"], ["r0"], [], []], program)
    assert got == ((1, "e0"), (2, "e1"), (3, "r0"))


def test_equation_rows_uses_the_one_pass_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from apeGmsh.opensees.opensees_model import _equation_rows

    model = SimpleNamespace(
        equation_constraints=lambda: ("q0", "q1"),
        opensees_zone_version=lambda: SimpleNamespace(
            major=2, minor=35, patch=0),
    )
    program = [_run(5, 2, "equationConstraint", 0)]
    _forbid_per_row(monkeypatch)
    assert _equation_rows(model, program, None) == ((5, "q0"), (6, "q1"))
