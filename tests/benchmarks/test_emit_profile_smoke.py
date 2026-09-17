"""cp1252 console smoke for the ADR 0100 D0 instrument (refuter lens 2).

Windows stdout is cp1252 whenever piped or redirected.  A single U+2248
in a printed f-string killed the bench at the first cell (rc=1) while
5,978 tests stayed green, because nothing exercised the CLI's print
path; ``--help`` was separately broken by a bare ``%`` in an argparse
help string (argparse %-formats help text), and the module docstring —
which argparse prints as its description — carried more non-cp1252
codepoints.  The contract these tests enforce: the module docstring and
every string the instrument prints stay ASCII-only.

Both tests force the child's stdio to cp1252 — on a UTF-8 console they
would prove nothing (the exact reason the breakage survived review).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tests" / "benchmarks" / "emit_throughput_profile.py"


def _cp1252_env() -> "dict[str, str]":
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    # Plain-python child: set PYTHONPATH explicitly — pytest's
    # ``pythonpath=["src"]`` applies only to THIS process, and relying
    # on it in a child is the editable-install trap's pytest cousin.
    env["PYTHONPATH"] = f"{_ROOT / 'src'}{os.pathsep}{_ROOT}"
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    return env


def test_help_renders_on_cp1252_stdout() -> None:
    """``--help`` formats the module docstring plus every ``help=``
    string through argparse (%-formatting included) and prints the lot;
    rc must be 0 on a cp1252 console."""
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True, env=_cp1252_env(), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    out = proc.stdout.decode("cp1252")
    assert "--rss-interval-ms" in out
    # The %%-escape must render as a literal percent, not TypeError.
    assert "~9% low" in out


_VERDICT_PREFIX = "gate verdict: "
_VERDICT_OK = "gate verdict: WITHIN ERROR BOUND"
_VERDICT_DISCARD = "gate verdict: *** DISCARDED - G0a error bound exceeded"


@pytest.mark.bench
def test_mem_cell_report_prints_on_cp1252_stdout(tmp_path: Path) -> None:
    """One tiny --mem cell end-to-end (~45 s): the whole
    ``report_instrumented`` print path must survive a cp1252 console —
    G2/G3 run exactly this pipeline with stdout redirected.

    This is a PRINT-PATH test, not a gate test.  Whether this tiny cell
    passes the G0a error bound is a measurement outcome of the machine
    it runs on — on the shared GitHub runner it discards — so asserting
    only the pass-branch text conflated "the report printed" with "this
    cell passed the gate" (the nightly Benchmarks job was red on that
    conflation from 2026-08-26).  The instrument now prints a verdict
    line in both branches under one prefix, and the assertions below
    require exactly one verdict, one of the two known spellings, and
    agreement with ``gate_status`` in the JSON record the campaign
    aggregator consumes — so the test still proves the verdict print
    path ran, and can no longer be satisfied by a half-printed report.
    """
    json_path = tmp_path / "smoke_cell.json"
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--recipe", "box", "--sizes", "6", "--parts", "2",
            "--mem", "--stream", "--staged", "--repeats", "1",
            "--tm-frames", "4", "--json", str(json_path),
        ],
        capture_output=True, env=_cp1252_env(), timeout=600,
    )
    assert proc.returncode == 0, (
        proc.stderr.decode(errors="replace")[-2000:]
    )
    out = proc.stdout.decode("cp1252")
    for marker in (
        "per-term at anchor",
        "R8 at anchor",
        "G0a(conservative)",
        _VERDICT_PREFIX.strip(),
    ):
        assert marker in out, f"missing {marker!r} in report output"

    # One cell (one size x one repeat) => exactly one verdict, and it
    # must be one of the two spellings the instrument can print.
    n_verdicts = out.count(_VERDICT_PREFIX)
    n_ok = out.count(_VERDICT_OK)
    n_discard = out.count(_VERDICT_DISCARD)
    assert n_verdicts == 1, (
        f"expected exactly 1 verdict line for 1 cell, saw {n_verdicts}"
    )
    assert n_ok + n_discard == 1, (
        "the verdict line matched neither known spelling "
        f"(ok={n_ok}, discarded={n_discard}) - the instrument's verdict "
        "wording changed without this contract being updated"
    )

    # The printed branch must agree with the machine-readable record.
    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(records) == 1, f"expected 1 json record, got {len(records)}"
    expected = "ok" if n_ok else "discarded_error_bound"
    assert records[0].get("gate_status") == expected, (
        f"printed verdict says {expected!r} but the json record says "
        f"{records[0].get('gate_status')!r}"
    )
