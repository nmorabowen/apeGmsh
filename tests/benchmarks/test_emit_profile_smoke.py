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


@pytest.mark.bench
def test_mem_cell_report_prints_on_cp1252_stdout() -> None:
    """One tiny --mem cell end-to-end (~45 s): the whole
    ``report_instrumented`` print path must survive a cp1252 console —
    G2/G3 run exactly this pipeline with stdout redirected."""
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--recipe", "box", "--sizes", "6", "--parts", "2",
            "--mem", "--stream", "--staged", "--repeats", "1",
            "--tm-frames", "4",
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
        "gate verdict",
    ):
        assert marker in out, f"missing {marker!r} in report output"
