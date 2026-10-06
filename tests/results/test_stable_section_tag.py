"""``_stable_section_tag`` gives one name the same tag in every process.

Its docstring promised a "Deterministic positive int tag derived from a
section name", but it took the builtin ``hash()`` of the name, which
Python salts per process (``PYTHONHASHSEED``): measured on the unfixed
source, ``"LayeredShell_A"`` came out ``1509370562`` under seed 1 and
``1288700299`` under seed 2. It is now a CRC-32 of the UTF-8 name.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from apeGmsh.results.capture.spec import _stable_section_tag

SRC = Path(__file__).resolve().parents[2] / "src"
PROBE = (
    "from apeGmsh.results.capture.spec import _stable_section_tag as t; "
    "print(t('LayeredShell_A'))"
)


# Windows-only: a fresh child sometimes dies natively (STATUS_STACK_BUFFER_
# OVERRUN, 0xC000070A) with empty stdout and stderr before running any
# test code (#1376). Retry once on exactly that signature; anything else,
# or a second crash, still fails.
_NATIVE_CRASH = 0xC000070A


def _is_native_crash(done: subprocess.CompletedProcess) -> bool:
    return (
        sys.platform == "win32"
        and done.returncode & 0xFFFFFFFF == _NATIVE_CRASH
        and not done.stdout.strip()
        and not done.stderr.strip()
    )


def _run_child(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    def run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, env=env, capture_output=True, text=True, encoding="utf-8",
        )

    done = run()
    if _is_native_crash(done):
        done = run()
    done.check_returncode()
    return done


def _tag_in_a_fresh_process(seed: str) -> int:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "APEGMSH_QUIET": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])),
    }
    done = _run_child([sys.executable, "-c", PROBE], env)
    return int(done.stdout.strip().splitlines()[-1])


def test_the_tag_is_the_same_under_two_hash_seeds() -> None:
    assert _tag_in_a_fresh_process("1") == _tag_in_a_fresh_process("2")


def test_the_tag_is_pinned() -> None:
    # A CRC-32 is fixed by its definition, so this value holds on every
    # platform and Python version. If it moves, every tag moved with it.
    assert _stable_section_tag("LayeredShell_A") == 1788243603


@pytest.mark.parametrize("name", ["", "a", "LayeredShell_A", "Hormigón"])
def test_the_tag_is_a_positive_opensees_int(name: str) -> None:
    tag = _stable_section_tag(name)
    assert isinstance(tag, int)
    assert 1 <= tag <= 2**31 - 2


def test_the_tag_is_the_crc32_of_the_utf8_name() -> None:
    name = "Hormigón"
    assert _stable_section_tag(name) == (zlib.crc32(name.encode("utf-8")) % (2**31 - 1) or 1)


def test_the_retry_fires_only_on_the_native_crash_signature(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def done(code: int, out: str = "", err: str = ""):
        return subprocess.CompletedProcess([], code, out, err)

    assert _is_native_crash(done(3221227274))
    assert not _is_native_crash(done(3221227274, err="Traceback"))
    assert not _is_native_crash(done(3221227274, out="1"))
    assert not _is_native_crash(done(1))
    monkeypatch.setattr(sys, "platform", "linux")
    assert not _is_native_crash(done(3221227274))


def test_a_second_crash_or_another_failure_still_fails(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []

    def crash(*a, **k):
        calls.append(1)
        return subprocess.CompletedProcess([], 3221227274, "", "")

    monkeypatch.setattr(subprocess, "run", crash)
    with pytest.raises(subprocess.CalledProcessError):
        _run_child(["x"], {})
    assert len(calls) == 2

    calls.clear()

    def other(*a, **k):
        calls.append(1)
        return subprocess.CompletedProcess([], 1, "", "")

    monkeypatch.setattr(subprocess, "run", other)
    with pytest.raises(subprocess.CalledProcessError):
        _run_child(["x"], {})
    assert len(calls) == 1
