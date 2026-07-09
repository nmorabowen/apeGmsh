"""``apeGmsh.opensees._run.stream_run`` — tee to log + console reporting.

Replaces the old ``subprocess.run(capture_output=True)`` that swallowed
output on success and dumped the whole buffer into an exception on
failure. Contract:

* the full raw output is **always** tee'd to ``log_path`` (verbatim,
  markers included) — on success AND on a non-zero exit,
* ``verbose=False`` prints begin / op / end only; ``verbose=True`` adds a
  live step counter parsed from the ``APEGMSH_PROGRESS`` markers plus
  streamed warning lines,
* a non-zero exit raises ``RuntimeError`` carrying the tail + log path,
  never the whole buffer.

Driven with a tiny throwaway python script (via ``sys.executable``) so
no OpenSees binary is needed in CI.
"""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

from apeGmsh.opensees._run import resolve_log_path, run_label, stream_run


def _script(tmp_path, body: str) -> str:
    p = tmp_path / "deck.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


_OK = """
    print("OpenSees 3.8.0")
    for i in (5, 10, 15, 20):
        print("APEGMSH_PROGRESS i=%d n=20 t=%g" % (i, i * 0.1))
        if i == 10:
            print("WARNING: Newton failed to converge")
    print("Analysis complete")
"""

_FAIL = """
    import sys
    print("OpenSees 3.8.0")
    print("WARNING: something off")
    print("apeGmsh: analyze FAILED at increment 3/20")
    sys.exit(1)
"""


# --- helpers --------------------------------------------------------------

def test_resolve_log_path_defaults_next_to_deck() -> None:
    assert resolve_log_path(None, os.path.join("out", "model.tcl")) == \
        os.path.join("out", "model.log")


def test_resolve_log_path_explicit_override() -> None:
    assert resolve_log_path("run.log", "out/model.tcl") == "run.log"


def test_run_label_forms() -> None:
    assert run_label("m.tcl", 6000, 0.002) == "analyze 6000 x dt=0.002  ->  m.tcl"
    assert run_label("m.tcl", 6000, None) == "analyze 6000  ->  m.tcl"
    assert run_label("m.tcl", None, None) == "run  ->  m.tcl"


# --- log is always written ------------------------------------------------

def test_log_written_verbatim_on_success(tmp_path) -> None:
    log = str(tmp_path / "run.log")
    stream_run([sys.executable, _script(tmp_path, _OK)],
               log_path=log, verbose=False, label="analyze 20 -> deck.py")
    txt = open(log, encoding="utf-8").read()
    # verbatim: markers AND the raw warning are on disk
    assert "APEGMSH_PROGRESS i=20 n=20" in txt
    assert "WARNING: Newton failed to converge" in txt
    assert "Analysis complete" in txt


def test_log_written_on_failure(tmp_path) -> None:
    log = str(tmp_path / "run.log")
    with pytest.raises(RuntimeError):
        stream_run([sys.executable, _script(tmp_path, _FAIL)],
                   log_path=log, verbose=False, label="analyze 20 -> deck.py")
    assert os.path.exists(log)
    assert "analyze FAILED at increment 3/20" in open(log, encoding="utf-8").read()


# --- console: minimal vs full --------------------------------------------

def test_minimal_console_is_terse(tmp_path, capsys) -> None:
    stream_run([sys.executable, _script(tmp_path, _OK)],
               log_path=str(tmp_path / "run.log"),
               verbose=False, label="analyze 20 -> deck.py")
    out = capsys.readouterr().out
    assert ">> OpenSees" in out
    assert "[OK] finished" in out
    # no per-step counter in minimal mode
    assert "step" not in out


def test_full_console_shows_counter_and_warning(tmp_path, capsys) -> None:
    stream_run([sys.executable, _script(tmp_path, _OK)],
               log_path=str(tmp_path / "run.log"),
               verbose=True, label="analyze 20 -> deck.py")
    out = capsys.readouterr().out
    assert "step      20/20  100%" in out
    assert "[warn] WARNING: Newton failed to converge" in out
    assert "1 warnings" in out


# --- failure: tail-only, not the whole buffer -----------------------------

def test_failure_exception_carries_tail_and_log_not_whole_buffer(tmp_path) -> None:
    log = str(tmp_path / "run.log")
    # pad the deck with >15 lines of preamble so the tail excludes it
    body = "\n".join(f'    print("preamble line {i}")' for i in range(40))
    body += '\n    import sys\n    print("apeGmsh: analyze FAILED at increment 3/20")\n    sys.exit(1)\n'
    with pytest.raises(RuntimeError) as ei:
        stream_run([sys.executable, _script(tmp_path, body)],
                   log_path=log, verbose=False, label="x")
    msg = str(ei.value)
    assert "returned 1" in msg
    assert log in msg
    assert "analyze FAILED at increment 3/20" in msg
    # the tail is bounded — early preamble is NOT in the exception
    assert "preamble line 0" not in msg
    # but the full log on disk has everything
    assert "preamble line 0" in open(log, encoding="utf-8").read()
