"""R3-S6 (#1790): ``stream_run`` returns a ``RunReport``.

ADR 0106 Amendment 1. Driven by a throwaway python child (as in
``test_run_streaming.py``) so no OpenSees binary is needed. The child
prints the ``APEGMSH_PROGRESS`` markers an emitted analyze loop prints:
a throttled subset of increments and always the last one (``i == n``).
"""
from __future__ import annotations

import dataclasses
import sys
import textwrap
from pathlib import Path

import pytest

from apeGmsh.opensees._run import RunReport, stream_run


def _child(tmp_path: Path, body: str) -> str:
    p = tmp_path / "child.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def _markers(*loops: tuple[int, int]) -> str:
    """The markers of whole loops ``(n, every)``, as the emitters print
    them: ``i % every == 0 or i == n``."""
    lines = []
    for n, every in loops:
        for i in range(1, n + 1):
            if i % every == 0 or i == n:
                lines.append(f"APEGMSH_PROGRESS i={i} n={n} t={i}")
    return "\n".join(lines)


def _run(tmp_path: Path, out: str, **kw: object) -> RunReport:
    body = f"print({out!r})\n"
    return stream_run(
        [sys.executable, _child(tmp_path, body)],
        log_path=str(tmp_path / "run.log"), verbose=False, label="t",
        **kw,  # type: ignore[arg-type]
    )


def test_increments_sum_every_whole_loop(tmp_path: Path) -> None:
    rep = _run(tmp_path, _markers((3, 1), (40, 2), (1, 1)),
               count_increments=True)
    assert isinstance(rep, RunReport)
    assert rep.increments == 44
    assert rep.returncode == 0
    assert rep.solver_stats is None
    assert rep.log == (tmp_path / "run.log").resolve()
    assert rep.log.is_file()
    assert rep.elapsed_s > 0.0


def test_the_report_is_frozen(tmp_path: Path) -> None:
    rep = _run(tmp_path, _markers((2, 1)), count_increments=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.increments = 3  # type: ignore[misc]


def test_markers_on_and_no_loop_is_zero(tmp_path: Path) -> None:
    assert _run(tmp_path, "no analyze here",
                count_increments=True).increments == 0


def test_without_markers_the_count_is_not_established(tmp_path: Path) -> None:
    assert _run(tmp_path, _markers((2, 1))).increments is None


@pytest.mark.parametrize("out", [
    # a loop left at i=2 of 3 before another starts
    "APEGMSH_PROGRESS i=1 n=3 t=1\nAPEGMSH_PROGRESS i=2 n=3 t=2\n"
    + _markers((2, 1)),
    # a loop still open at the end
    _markers((2, 1)) + "\nAPEGMSH_PROGRESS i=1 n=3 t=1",
    # two processes printing the same loop, interleaved
    "APEGMSH_PROGRESS i=1 n=2 t=1\nAPEGMSH_PROGRESS i=1 n=2 t=1\n"
    "APEGMSH_PROGRESS i=2 n=2 t=2\nAPEGMSH_PROGRESS i=2 n=2 t=2",
    # an increment past its loop
    "APEGMSH_PROGRESS i=3 n=2 t=3",
])
def test_markers_that_are_not_whole_loops_give_none(
    tmp_path: Path, out: str,
) -> None:
    assert _run(tmp_path, out, count_increments=True).increments is None


def test_exit_0_without_the_completion_proof_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"exited 0 but the deck did not "
                                           r"run to its end.*run\.log"):
        _run(tmp_path, "apeGmsh: analyze FAILED at increment 1/2",
             proves_completion=lambda log: False)


def test_exit_0_with_the_completion_proof_returns(tmp_path: Path) -> None:
    seen: list[str] = []

    def proves(log: str) -> bool:
        seen.append(log)
        return True

    rep = _run(tmp_path, _markers((2, 1)), count_increments=True,
               proves_completion=proves)
    assert rep.increments == 2
    assert seen == [str(tmp_path / "run.log")]


def test_a_non_zero_exit_still_raises(tmp_path: Path) -> None:
    body = "import sys\nprint('boom')\nsys.exit(3)\n"
    with pytest.raises(RuntimeError, match=r"subprocess returned 3"):
        stream_run(
            [sys.executable, _child(tmp_path, body)],
            log_path=str(tmp_path / "run.log"), verbose=False, label="t",
            count_increments=True, proves_completion=lambda log: True,
        )
