"""``pid_alive`` observes a process and never signals it (ADR 0095 S5j).

``read_host``, ``clear_host`` and the busy lock trust it to tell a
crashed host from a live one, so a wrong answer either keeps a dead
claim alive or steals a live one.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from apeGmsh.studio._host_state import pid_alive


@pytest.fixture(autouse=True)
def _no_console_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows ``os.kill(pid, 0)`` is CTRL_C_EVENT, not a probe; on
    Python 3.11 a failed one falls through to TerminateProcess and can
    end this very test run with no summary. Make such a call fail loudly
    instead."""
    if sys.platform != "win32":
        return
    real_kill = os.kill

    def _guarded(pid: int, sig: int) -> None:
        if sig in (signal.CTRL_C_EVENT, signal.CTRL_BREAK_EVENT):
            raise AssertionError(
                f"os.kill({pid}, {sig}) sends a console signal; it is no probe"
            )
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", _guarded)


def test_pid_alive_reads_an_exited_child_as_dead() -> None:
    """While we hold its Popen handle, an exited child's process object
    can still be opened on Windows, so only the exit code tells it is
    gone."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert pid_alive(child.pid)
    finally:
        child.kill()
        child.wait(timeout=10)
    assert not pid_alive(child.pid)


def test_pid_alive_refuses_a_pid_past_a_dword() -> None:
    # Past a DWORD, ctypes would wrap this onto our own (live) pid.
    assert not pid_alive(2**32 + os.getpid())
    assert not pid_alive(2**40)


@pytest.mark.skipif(
    sys.platform == "win32", reason="os.kill overflows only past POSIX pid_t"
)
def test_pid_alive_reads_a_pid_past_pid_t_as_dead() -> None:
    # Under the DWORD cap but past a signed 32-bit pid_t: os.kill raises
    # OverflowError, which must read dead rather than escape the readers.
    assert not pid_alive(2**31 + 7)


def test_pid_alive_counts_a_process_it_may_not_open_as_alive() -> None:
    # System (Windows) and init (POSIX) exist; being refused access
    # means alive.
    assert pid_alive(4 if sys.platform == "win32" else 1)


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only")
def test_pid_alive_leaves_the_shared_kernel32_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ctypes.windll.kernel32`` is shared by the whole process; its
    ``argtypes`` belong to whoever else calls it."""
    import ctypes

    shared = ctypes.windll.kernel32.OpenProcess
    monkeypatch.setattr(shared, "argtypes", None)
    assert pid_alive(os.getpid())
    assert shared.argtypes is None
