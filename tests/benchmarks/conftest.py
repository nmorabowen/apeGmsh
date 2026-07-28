"""Benchmark-local pytest options.

Hosts the re-baseline switch for :mod:`test_emit_regression_gate`. Kept
here rather than in the root ``conftest`` so the flag only exists where
it means something.
"""
from __future__ import annotations

import pytest

WRITE_BASELINE_FLAG = "--emit-gate-write-baseline"


def pytest_addoption(parser: "pytest.Parser") -> None:
    parser.addoption(
        WRITE_BASELINE_FLAG,
        action="store_true",
        default=False,
        help=(
            "Overwrite tests/benchmarks/emit_gate_baseline.json from this "
            "run instead of asserting against it. Use when a change "
            "legitimately moves emit cost, and commit the new baseline in "
            "the same PR so the diff shows the cost being accepted."
        ),
    )
