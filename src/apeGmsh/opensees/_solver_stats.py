"""Parse ``system Pardiso -stats`` blocks out of an OpenSees run's output.

Pure, stdlib-only parser for ADR 0106 D1. No wiring lives here — this
module knows nothing about subprocesses, decks, or emitters; S2 adds the
``APEGMSH_STAGE`` marker that gives it stage boundaries and S3 wires it
into :func:`apeGmsh.opensees._run.stream_run` and the ``apeSees``
bridge.

After fork PR #821, ``ops.system.Pardiso(stats=True)`` answers on
stderr after **every** numeric factorisation with a five-line block,
labels exact (pinned by the fork's own ``tests/test_pardiso_stats.py``;
a 54-DOF brick gives ``factor entries iparm(18) = 1836``)::

    PARDISO stats: n=<n> nnz(A)=<nnz> matrixType=<mtype> threads=<nthreads>
      factor entries iparm(18)  = <nnz in L+U>
      peak memory KB iparm(15)  = <peak during symbolic>
      perm memory KB iparm(16)  = <permanent>
      fact memory KB iparm(17)  = <numerical factorization + solve>
      factor Mflops  iparm(19)  = <mflops>

:func:`parse_solver_stats` recognises exactly this block (D4): a header
opens it, the block completes once all five fields have been seen, and
anything else — a missing field, an interrupting second header, a field
with no header, a value that will not parse as a number — counts
against ``malformed_blocks`` instead of raising or guessing. The parser
never raises on input content; see the module's mutation tests.

Complete blocks are attributed to the ``APEGMSH_STAGE open|close
<name>`` window they fall inside (D2's marker, emitted starting S2), or
to the run-level bucket when there is none. Within each bucket the four
capacity numbers reduce to a maximum and the four identity numbers
(``n``, ``nnz(A)``, ``matrixType``, ``threads``) reduce to the last
value seen (D6) — see :class:`StageSolverStats`.

The old once-per-pattern format (pre-#821 fork builds) is not
recognised: its lines match neither pattern below, so they are silently
invisible here rather than guessed at (D4).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

__all__ = [
    "SolverStatsBlock",
    "StageSolverStats",
    "RunSolverStats",
    "parse_solver_stats",
]


@dataclass(frozen=True, slots=True)
class SolverStatsBlock:
    """One complete ``PARDISO stats:`` block — one numeric factorisation."""

    n: int
    nnz_a: int
    matrix_type: int
    threads: int
    factor_entries: int
    peak_memory_kb: int
    perm_memory_kb: int
    fact_memory_kb: int
    mflops: float


@dataclass(frozen=True, slots=True)
class StageSolverStats:
    """The ADR 0106 D6 reduction of every block attributed to one bucket.

    ``name`` is the stage name, or ``None`` for the run-level bucket
    (blocks outside any ``APEGMSH_STAGE`` window — the only bucket a
    flat, unstaged deck ever fills). A bucket with no blocks at all
    (``factorisations == 0``) carries zero/``0.0`` in every other field;
    there is nothing to report a last-seen or maximum of.
    """

    name: str | None
    factorisations: int
    n: int
    nnz_a: int
    matrix_type: int
    threads: int
    max_factor_entries: int
    max_peak_memory_kb: int
    max_perm_memory_kb: int
    max_fact_memory_kb: int
    max_mflops: float


@dataclass(frozen=True, slots=True)
class RunSolverStats:
    """The parsed record for one run.

    ``factorisations`` and ``malformed_blocks`` are run-wide totals
    (INV-3); ``factorisations`` equals the sum of every stage's plus
    ``run_level``'s own count. ``malformed_blocks`` is not attributed to
    a stage — a block that never completed was never confirmed to
    belong anywhere.
    """

    stages: tuple[StageSolverStats, ...]
    run_level: StageSolverStats
    factorisations: int
    malformed_blocks: int


# Anchored on the fork's exact labels (D4) — do not relax.
_HEAD = re.compile(
    r"PARDISO stats:\s+n=(\d+)\s+nnz\(A\)=(\d+)\s+"
    r"matrixType=(-?\d+)\s+threads=(\d+)"
)
_FIELD = re.compile(
    r"^\s+(factor entries iparm\(18\)|peak memory KB iparm\(15\)|"
    r"perm memory KB iparm\(16\)|fact memory KB iparm\(17\)|"
    r"factor Mflops\s+iparm\(19\))\s*=\s*(-?[\d.eE+-]+)\s*$"
)
_STAGE_MARKER = re.compile(r"APEGMSH_STAGE (open|close) (.+)$")

#: The fixed ``iparm(N)`` token embedded in a matched field's label,
#: mapped to the :class:`SolverStatsBlock` attribute it fills. Matching
#: on the token (rather than the label's exact text) tolerates the
#: alignment whitespace the fork pads labels with — ``_FIELD``'s
#: alternatives already differ only in that whitespace.
_FIELD_BY_IPARM = {
    "iparm(18)": "factor_entries",
    "iparm(15)": "peak_memory_kb",
    "iparm(16)": "perm_memory_kb",
    "iparm(17)": "fact_memory_kb",
    "iparm(19)": "mflops",
}
_REQUIRED_FIELDS = frozenset(_FIELD_BY_IPARM.values())


class _Pending:
    """A block under construction: header identity (if any) + fields seen."""

    __slots__ = ("identity", "fields", "stage")

    def __init__(
        self, identity: tuple[int, int, int, int] | None, stage: str | None
    ) -> None:
        self.identity = identity
        self.fields: dict[str, float] = {}
        self.stage = stage


def _empty_stage(name: str | None) -> StageSolverStats:
    return StageSolverStats(
        name=name,
        factorisations=0,
        n=0,
        nnz_a=0,
        matrix_type=0,
        threads=0,
        max_factor_entries=0,
        max_peak_memory_kb=0,
        max_perm_memory_kb=0,
        max_fact_memory_kb=0,
        max_mflops=0.0,
    )


def _reduce(name: str | None, blocks: list[SolverStatsBlock]) -> StageSolverStats:
    """D6: max on the four capacity numbers, last-seen on the four identity ones."""
    if not blocks:
        return _empty_stage(name)
    last = blocks[-1]
    return StageSolverStats(
        name=name,
        factorisations=len(blocks),
        n=last.n,
        nnz_a=last.nnz_a,
        matrix_type=last.matrix_type,
        threads=last.threads,
        max_factor_entries=max(b.factor_entries for b in blocks),
        max_peak_memory_kb=max(b.peak_memory_kb for b in blocks),
        max_perm_memory_kb=max(b.perm_memory_kb for b in blocks),
        max_fact_memory_kb=max(b.fact_memory_kb for b in blocks),
        max_mflops=max(b.mflops for b in blocks),
    )


def parse_solver_stats(
    source: Union[Iterable[str], str, "os.PathLike[str]"],
) -> RunSolverStats:
    """Parse a solve stream (or its tee'd log) into a :class:`RunSolverStats`.

    Two entry points, one function, per D1: an ``Iterable[str]`` of
    already-split lines — the shape ``stream_run`` feeds it line by line
    as it tees — or a path (``str`` / ``os.PathLike``) to a log file to
    read and parse whole, the shape a caller re-parses a failed run's
    ``<deck>.log`` with (INV-7). The parser never raises on input
    content: a line matching neither pattern is ignored, and a block
    that cannot be completed increments ``malformed_blocks`` (D4)
    instead of raising or being guessed at.
    """
    if isinstance(source, (str, os.PathLike)):
        lines: Iterable[str] = Path(source).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    else:
        lines = source

    blocks_by_stage: dict[str | None, list[SolverStatsBlock]] = {}
    malformed_blocks = 0
    current_stage: str | None = None
    pending: _Pending | None = None

    def finalize() -> None:
        nonlocal pending, malformed_blocks
        if pending is None:
            return
        if pending.identity is not None and len(pending.fields) == len(_REQUIRED_FIELDS):
            n, nnz_a, matrix_type, threads = pending.identity
            block = SolverStatsBlock(
                n=n,
                nnz_a=nnz_a,
                matrix_type=matrix_type,
                threads=threads,
                factor_entries=int(pending.fields["factor_entries"]),
                peak_memory_kb=int(pending.fields["peak_memory_kb"]),
                perm_memory_kb=int(pending.fields["perm_memory_kb"]),
                fact_memory_kb=int(pending.fields["fact_memory_kb"]),
                mflops=pending.fields["mflops"],
            )
            blocks_by_stage.setdefault(pending.stage, []).append(block)
        elif pending.fields:
            # A header with 1-4 fields, or an orphan (fields with no
            # header) with any field at all: started, never completed.
            malformed_blocks += 1
        # A header with zero following fields is dropped silently:
        # nothing was ever attempted, so nothing was corrupted.
        pending = None

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")

        stage_match = _STAGE_MARKER.match(line)
        if stage_match is not None:
            action, name = stage_match.group(1), stage_match.group(2)
            current_stage = name if action == "open" else None
            continue

        head_match = _HEAD.match(line)
        if head_match is not None:
            finalize()
            n, nnz_a, matrix_type, threads = (int(g) for g in head_match.groups())
            pending = _Pending((n, nnz_a, matrix_type, threads), current_stage)
            continue

        field_match = _FIELD.match(line)
        if field_match is not None:
            label, raw_value = field_match.groups()
            try:
                value = float(raw_value)
            except ValueError:
                continue
            key = next(v for k, v in _FIELD_BY_IPARM.items() if k in label)
            if pending is None:
                pending = _Pending(None, current_stage)
            pending.fields[key] = value

    finalize()

    stages = tuple(
        _reduce(name, blocks) for name, blocks in blocks_by_stage.items() if name is not None
    )
    run_level = _reduce(None, blocks_by_stage.get(None, []))
    factorisations = sum(s.factorisations for s in stages) + run_level.factorisations

    return RunSolverStats(
        stages=stages,
        run_level=run_level,
        factorisations=factorisations,
        malformed_blocks=malformed_blocks,
    )
