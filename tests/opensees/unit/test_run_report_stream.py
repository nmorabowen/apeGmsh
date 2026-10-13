"""R3-S6 (#1790): ``stream_run`` returns a ``RunReport``.

ADR 0106 Amendment 1. Driven by a throwaway python child (as in
``test_run_streaming.py``) so no OpenSees binary is needed. The child
prints the ``APEGMSH_PROGRESS`` markers an emitted analyze loop prints:
a throttled subset of increments and always the last one (``i == n``).
``expected_loops`` is the deck's own loop sequence, which the bridge
reads with ``deck_progress_loops``.
"""
from __future__ import annotations

import dataclasses
import sys
import textwrap
from pathlib import Path

import pytest

from apeGmsh.opensees._run import RunReport, deck_progress_loops, stream_run

_DONE = "APEGMSH_RUN_COMPLETE"


def _child(tmp_path: Path, body: str) -> str:
    p = tmp_path / "child.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def _loop(n: int, every: int = 1) -> list[str]:
    """The markers of one whole loop, as the emitters print them:
    ``i % every == 0 or i == n``."""
    return [f"APEGMSH_PROGRESS i={i} n={n} t={i}"
            for i in range(1, n + 1) if i % every == 0 or i == n]


def _run(tmp_path: Path, lines: list[str], **kw: object) -> RunReport:
    body = f"print({chr(10).join(lines)!r})\n"
    return stream_run(
        [sys.executable, _child(tmp_path, body)],
        log_path=str(tmp_path / "run.log"), verbose=False, label="t",
        **kw,  # type: ignore[arg-type]
    )


def test_increments_sum_every_whole_loop(tmp_path: Path) -> None:
    rep = _run(tmp_path, _loop(3) + _loop(40, 2) + _loop(1),
               expected_loops=(3, 40, 1))
    assert isinstance(rep, RunReport)
    assert rep.increments == 44
    assert rep.returncode == 0
    assert rep.solver_stats is None
    assert rep.log == (tmp_path / "run.log").resolve()
    assert rep.log.is_file()
    assert rep.elapsed_s > 0.0


def test_the_report_is_frozen(tmp_path: Path) -> None:
    rep = _run(tmp_path, _loop(2), expected_loops=(2,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.increments = 3  # type: ignore[misc]


def test_markers_on_and_no_loop_is_zero(tmp_path: Path) -> None:
    assert _run(tmp_path, ["no analyze here"],
                expected_loops=()).increments == 0


def test_without_markers_the_count_is_not_established(tmp_path: Path) -> None:
    assert _run(tmp_path, _loop(2)).increments is None


@pytest.mark.parametrize("lines", [
    # a loop left at i=2 of 3 before another starts
    _loop(3)[:2] + _loop(2),
    # a loop still open at the end
    _loop(2) + _loop(3)[:1],
    # an increment past its loop
    ["APEGMSH_PROGRESS i=3 n=2 t=3"],
], ids=["abandoned", "open-at-end", "past-n"])
def test_markers_that_are_not_whole_loops_give_none(
    tmp_path: Path, lines: list[str],
) -> None:
    assert _run(tmp_path, lines, expected_loops=(2,)).increments is None


# --- more than one process wrote the stream (the review of #1849) -----------

def test_two_interleaved_processes_of_one_increment_loops_give_none(
    tmp_path: Path,
) -> None:
    """Two ranks of a deck with one 1-increment loop: each ``i=1 n=1``
    is a whole loop on its own, so only the deck's sequence (one loop)
    exposes the second rank. Without the check this reported 2."""
    lines = _loop(1) + _loop(1)
    assert _run(tmp_path, lines, expected_loops=(1,)).increments is None


def test_two_processes_each_printing_a_whole_loop_give_none(
    tmp_path: Path,
) -> None:
    """Two ranks each running the deck's one n=4 loop to its end, their
    output arriving whole: two legitimate-looking loops. Reported 8."""
    lines = _loop(4) + _loop(4)
    assert _run(tmp_path, lines, expected_loops=(4,)).increments is None


def test_interleaved_processes_of_a_longer_loop_give_none(
    tmp_path: Path,
) -> None:
    """Two ranks interleaved line by line: ``i`` repeats inside the loop."""
    a = _loop(4)
    lines = [ln for pair in zip(a, a) for ln in pair]
    assert _run(tmp_path, lines, expected_loops=(4,)).increments is None


def test_more_than_one_completion_line_gives_none(tmp_path: Path) -> None:
    """The Tcl driver's completion line twice: two processes ran it, even
    when the markers alone look like the deck's one loop."""
    lines = _loop(2) + [_DONE, _DONE]
    rep = _run(tmp_path, lines, expected_loops=(2,),
               proves_completion=lambda log: True, completion_line=_DONE)
    assert rep.increments is None
    one = _run(tmp_path, _loop(2) + [_DONE], expected_loops=(2,),
               proves_completion=lambda log: True, completion_line=_DONE)
    assert one.increments == 2


# --- the marker is a whole line ---------------------------------------------

def test_a_stage_named_like_a_marker_does_not_count(tmp_path: Path) -> None:
    """``APEGMSH_STAGE open <name>`` with a marker-shaped name: the
    marker regex is anchored, so the stage line is not ``i=7 n=7``."""
    lines = ["APEGMSH_STAGE open APEGMSH_PROGRESS i=7 n=7 t=0"] + _loop(2)
    assert _run(tmp_path, lines, expected_loops=(2,)).increments == 2


def test_deck_loops_skip_a_stage_named_like_a_marker(tmp_path: Path) -> None:
    deck = tmp_path / "deck.tcl"
    deck.write_text(
        'puts "APEGMSH_STAGE open APEGMSH_PROGRESS i=7 n=7 t=0"\n'
        "for {set _apesees_i 0} {$_apesees_i < 3} {incr _apesees_i} {\n"
        "    if {[expr {($_apesees_i + 1) % 1}] == 0 || "
        "[expr {$_apesees_i + 1}] == 3} {\n"
        '        puts "APEGMSH_PROGRESS i=[expr {$_apesees_i + 1}] n=3 '
        't=[getTime]"\n'
        "    }\n}\n"
        "for _apesees_i in range(0):\n"
        '    print("APEGMSH_PROGRESS i=%d n=0 t=%g" % (1, 0.0))\n'
        '    print("APEGMSH_PROGRESS i=%d n=5 t=%g" % (1, 0.0), flush=True)\n',
        encoding="utf-8",
    )
    # the stage line is not a loop; n=0 prints nothing and is left out
    assert deck_progress_loops(str(deck)) == (3, 5)


# --- the completion proof ---------------------------------------------------

def test_exit_0_without_the_completion_proof_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"exited 0 but the deck did not "
                                           r"run to its end.*run\.log"):
        _run(tmp_path, ["apeGmsh: analyze FAILED at increment 1/2"],
             proves_completion=lambda log: False)


def test_exit_0_with_the_completion_proof_returns(tmp_path: Path) -> None:
    seen: list[str] = []

    def proves(log: str) -> bool:
        seen.append(log)
        return True

    rep = _run(tmp_path, _loop(2), expected_loops=(2,),
               proves_completion=proves)
    assert rep.increments == 2
    assert seen == [str(tmp_path / "run.log")]


def test_a_non_zero_exit_still_raises(tmp_path: Path) -> None:
    body = "import sys\nprint('boom')\nsys.exit(3)\n"
    with pytest.raises(RuntimeError, match=r"subprocess returned 3"):
        stream_run(
            [sys.executable, _child(tmp_path, body)],
            log_path=str(tmp_path / "run.log"), verbose=False, label="t",
            expected_loops=(), proves_completion=lambda log: True,
        )
