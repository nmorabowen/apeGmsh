"""Unit tests for the ADR 0106 S1 ``system Pardiso -stats`` parser.

Pure-parser tests only — no stage markers are emitted anywhere in the
tree yet (that is S2), and nothing here touches ``stream_run`` or the
``apeSees`` bridge (S3). ``parse_solver_stats`` is fed hand-built line
streams that stand in for what the fork's fork PR #821 block (and the
S2 ``APEGMSH_STAGE`` marker it will eventually sit between) looks like
on the wire.

Only ``factor entries iparm(18) = 1836`` for a 54-DOF brick is a
measured fork value (pinned by the fork's own
``tests/test_pardiso_stats.py``, cited in ADR 0106). Every other number
in these fixtures — memory, Mflops, thread counts, the second block's
values — is a fixture value chosen to make a wrong reduction fail, not
a measured one.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._solver_stats import parse_solver_stats

# The fork PR #821 block for a 54-DOF brick (ADR 0106, D1).
_GOOD_BLOCK = """\
PARDISO stats: n=54 nnz(A)=1404 matrixType=2 threads=8
  factor entries iparm(18)  = 1836
  peak memory KB iparm(15)  = 123
  perm memory KB iparm(16)  = 45
  fact memory KB iparm(17)  = 67
  factor Mflops  iparm(19)  = 2
"""

# A second, distinct block: every capacity number is chosen so that
# "last seen" would disagree with "max" (and vice versa for the
# identity numbers), so a reduction that maxes the wrong group fails.
_BLOCK_A = """\
PARDISO stats: n=30 nnz(A)=500 matrixType=5 threads=16
  factor entries iparm(18)  = 500
  peak memory KB iparm(15)  = 50
  perm memory KB iparm(16)  = 20
  fact memory KB iparm(17)  = 30
  factor Mflops  iparm(19)  = 1
"""
_BLOCK_B = """\
PARDISO stats: n=10 nnz(A)=200 matrixType=-2 threads=4
  factor entries iparm(18)  = 300
  peak memory KB iparm(15)  = 80
  perm memory KB iparm(16)  = 10
  fact memory KB iparm(17)  = 60
  factor Mflops  iparm(19)  = 0.5
"""


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_fork_reference_block_parses() -> None:
    """The fork's own 54-DOF fixture: factor_entries == 1836 (measured)."""
    record = parse_solver_stats(_lines(_GOOD_BLOCK))
    assert record.factorisations == 1
    assert record.malformed_blocks == 0
    assert record.stages == ()
    block_bucket = record.run_level
    assert block_bucket.factorisations == 1
    assert block_bucket.max_factor_entries == 1836
    assert block_bucket.n == 54
    assert block_bucket.nnz_a == 1404
    assert block_bucket.matrix_type == 2
    assert block_bucket.threads == 8
    assert block_bucket.max_peak_memory_kb == 123
    assert block_bucket.max_perm_memory_kb == 45
    assert block_bucket.max_fact_memory_kb == 67
    assert block_bucket.max_mflops == 2.0


def test_zero_blocks_is_an_empty_record() -> None:
    record = parse_solver_stats(["not a stats line", "", "OpenSees > "])
    assert record.stages == ()
    assert record.factorisations == 0
    assert record.malformed_blocks == 0
    assert record.run_level.factorisations == 0
    assert record.run_level.max_factor_entries == 0


def test_multi_block_reduction_maxes_capacity_and_keeps_last_identity() -> None:
    """D6: max on the four capacity numbers, last-seen on the four identity ones.

    Block A's factor_entries/perm_memory/mflops are each the larger of
    the two but A is NOT last -- a reduction that (wrongly) took "last"
    for these would report block B's smaller numbers instead. Block B's
    n/nnz_a/matrix_type/threads are each different from A's -- a
    reduction that (wrongly) took "max" for these would disagree with
    B, the actually-last block.
    """
    record = parse_solver_stats(_lines(_BLOCK_A + _BLOCK_B))
    assert record.factorisations == 2
    assert record.malformed_blocks == 0
    bucket = record.run_level
    # identity: last seen (block B), not max.
    assert bucket.n == 10
    assert bucket.nnz_a == 200
    assert bucket.matrix_type == -2
    assert bucket.threads == 4
    # capacity: max across both blocks, not last.
    assert bucket.max_factor_entries == 500
    assert bucket.max_peak_memory_kb == 80
    assert bucket.max_perm_memory_kb == 20
    assert bucket.max_fact_memory_kb == 60
    assert bucket.max_mflops == 1.0


def test_stage_attribution_including_a_name_with_spaces() -> None:
    stream = (
        ['APEGMSH_STAGE open gravity load step']
        + _lines(_BLOCK_A)
        + ['APEGMSH_STAGE close gravity load step']
        + _lines(_BLOCK_B)
    )
    record = parse_solver_stats(stream)
    assert record.factorisations == 2
    assert record.malformed_blocks == 0
    assert len(record.stages) == 1
    stage = record.stages[0]
    assert stage.name == "gravity load step"
    assert stage.factorisations == 1
    assert stage.max_factor_entries == 500
    # The block outside any APEGMSH_STAGE window lands run-level.
    assert record.run_level.factorisations == 1
    assert record.run_level.n == 10


# --- mutation check: six ways a block can fail to parse -------------------
#
# Each mutant starts from a known-good block and breaks exactly one
# thing about it. A parser that accepts any of these is not measuring
# anything (ADR 0106 S1): every one must land as one malformed block and
# zero factorisations, never raise, and never produce a SolverStatsBlock.

_MUTANT_LABEL_TYPO = """\
PARDISO stats: n=54 nnz(A)=1404 matrixType=2 threads=8
  factor entries iparm(8)  = 1836
  peak memory KB iparm(15)  = 123
  perm memory KB iparm(16)  = 45
  fact memory KB iparm(17)  = 67
  factor Mflops  iparm(19)  = 2
"""

_MUTANT_DROPPED_HEADER_FIELD = """\
PARDISO stats: n=54 matrixType=2 threads=8
  factor entries iparm(18)  = 1836
  peak memory KB iparm(15)  = 123
  perm memory KB iparm(16)  = 45
  fact memory KB iparm(17)  = 67
  factor Mflops  iparm(19)  = 2
"""

# "1e" matches the value character class ([\d.eE+-]+) but float("1e")
# raises -- this exercises the parser's own exception handling, not
# just the regex rejecting an obviously non-numeric string.
_MUTANT_NON_NUMERIC_VALUE = """\
PARDISO stats: n=54 nnz(A)=1404 matrixType=2 threads=8
  factor entries iparm(18)  = 1836
  peak memory KB iparm(15)  = 123
  perm memory KB iparm(16)  = 45
  fact memory KB iparm(17)  = 67
  factor Mflops  iparm(19)  = 1e
"""

_MUTANT_TRUNCATED_MISSING_MFLOPS = """\
PARDISO stats: n=54 nnz(A)=1404 matrixType=2 threads=8
  factor entries iparm(18)  = 1836
  peak memory KB iparm(15)  = 123
  perm memory KB iparm(16)  = 45
  fact memory KB iparm(17)  = 67
"""

_MUTANT_TWO_HEADERS_ONE_FIELD_BETWEEN = """\
PARDISO stats: n=54 nnz(A)=1404 matrixType=2 threads=8
  factor entries iparm(18)  = 1836
PARDISO stats: n=10 nnz(A)=20 matrixType=1 threads=1
"""

# Stands in for the pre-#821 once-per-pattern format (not pinned by any
# test, per ADR 0106 D4 -- this line is invented, not measured).
_MUTANT_OLD_FORMAT_LINE = """\
PARDISO stats: n=54 nnz(A)=1404 matrixType=2 threads=8
  nnz and peak memory (per sparsity pattern): 1836, 123
  peak memory KB iparm(15)  = 123
  perm memory KB iparm(16)  = 45
  fact memory KB iparm(17)  = 67
  factor Mflops  iparm(19)  = 2
"""


@pytest.mark.parametrize(
    "mutant",
    [
        _MUTANT_LABEL_TYPO,
        _MUTANT_DROPPED_HEADER_FIELD,
        _MUTANT_NON_NUMERIC_VALUE,
        _MUTANT_TRUNCATED_MISSING_MFLOPS,
        _MUTANT_TWO_HEADERS_ONE_FIELD_BETWEEN,
        _MUTANT_OLD_FORMAT_LINE,
    ],
    ids=[
        "label_typo",
        "dropped_header_field",
        "non_numeric_value",
        "truncated_missing_mflops",
        "two_headers_one_field_between",
        "old_format_line",
    ],
)
def test_malformed_mutants_never_produce_a_block(mutant: str) -> None:
    record = parse_solver_stats(_lines(mutant))
    assert record.malformed_blocks == 1
    assert record.factorisations == 0
