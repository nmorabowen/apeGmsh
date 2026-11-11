"""A killed workspace writer leaves the previous file whole (ADR 0132 D6).

As ``test_user_atomic`` does for ``.user.h5``: the writer runs in a real
subprocess, parks at a known point, and is killed by the OS, so no ``finally``
runs. ``mid_write`` parks with the partial half-written; ``before_replace`` parks
with it complete, verified and unpublished. The variants manifest is killed the
same way before its replace.
"""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import h5py
import pytest

from apeGmsh.results.user import (
    SetDef,
    VariantDef,
    ViewDef,
    WorkspaceFile,
    WorkspaceWriteRefused,
    read_variants,
    write_variants,
    write_workspace,
)

from tests.results.user._user_helpers import REPO
from tests.results.user._workspace_helpers import frame

SCRIPT = textwrap.dedent(
    """
    import sys, time
    from apeGmsh.results.user import (
        SetDef, VariantDef, ViewDef, _variants, _workspace, write_variants, write_workspace)

    target, point = sys.argv[1:3]

    def park(*args, **kwargs):
        print("PARKED", flush=True)
        time.sleep(120)

    if point == "mid_write":
        real = _workspace._write_file
        def half(out, meta, tables):
            real(out, meta, tables)
            del out["views"]
            out.flush()       # what is written so far really reaches the partial
            park()
        _workspace._write_file = half
    elif point in ("before_replace", "manifest_before_replace"):
        _workspace._replace = park
        _variants._replace = park
    else:
        raise SystemExit(f"unknown point {point}")

    if point == "manifest_before_replace":
        write_variants(target, [VariantDef("B", "frame_b.h5", {"pass": 2})])
    else:
        write_workspace(target, sets=[SetDef("base", "nodes", [1, 2, 3])],
                        views=[ViewDef("later", {"camera": "top"})])
    print("FINISHED", flush=True)
    """
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partials(target: Path) -> list[Path]:
    return sorted(p for p in target.parent.iterdir()
                  if p.name.startswith(target.name + ".") and p.name.endswith(".partial"))


def _kill_writer_at(point: str, target: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.Popen([sys.executable, "-c", SCRIPT, str(target), point],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert proc.stdout is not None
        line: Optional[str] = proc.stdout.readline().strip()
        assert line == "PARKED", (
            f"the writer never reached {point!r}: {line!r}\n"
            f"{proc.stderr.read() if proc.stderr else ''}")
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()
    assert proc.returncode != 0, "the writer was not killed: the test proved nothing"
    for partial in _partials(target):
        _wait_for_the_dead_writers_lock(partial)


def _wait_for_the_dead_writers_lock(partial: Path) -> None:
    """Windows releases a dead process's file locks some time after it exits."""
    deadline = time.monotonic() + 60.0
    while partial.exists() and time.monotonic() < deadline:
        try:
            partial.read_bytes()
            if ".workspace.h5." in partial.name:
                with h5py.File(partial, "r"):
                    pass
            return
        except OSError as exc:
            if "lock" not in str(exc).lower() and not isinstance(exc, PermissionError):
                return  # a partial truly cut short: the lock is not what stops it
            time.sleep(0.05)


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    frame(tmp_path / "frame.h5")
    return write_workspace(tmp_path / "frame.workspace.h5",
                           sets=[SetDef("roof", "nodes", [31, 32, 33])],
                           views=[ViewDef("iso", {"camera": "iso"})])


@pytest.mark.parametrize("point", ["mid_write", "before_replace"])
def test_a_killed_writer_leaves_the_previous_workspace_whole(saved: Path, point: str) -> None:
    before = _digest(saved)
    _kill_writer_at(point, saved)
    assert _digest(saved) == before, "the previous file changed under a killed writer"
    ws = WorkspaceFile(saved)
    assert ws.revision == 1
    assert [s.name for s in ws.sets] == ["roof"] and [v.name for v in ws.views] == ["iso"]
    assert _partials(saved), "the kill came before any partial: the test proved nothing"


def test_the_partial_really_held_the_new_file_when_the_writer_died(saved: Path) -> None:
    _kill_writer_at("before_replace", saved)
    ws = WorkspaceFile(_partials(saved)[0])
    assert ws.revision == 2 and [v.name for v in ws.views] == ["iso", "later"]


def test_a_later_write_survives_and_sweeps_the_dead_writers_partial(saved: Path) -> None:
    _kill_writer_at("mid_write", saved)
    assert _partials(saved)
    write_workspace(saved, views=[ViewDef("after", {"camera": "top"})])
    assert [v.name for v in WorkspaceFile(saved).views] == ["iso", "after"]
    assert not _partials(saved), "this host's dead partial was not swept"


HOLDERS = {
    "h5py": "import h5py, sys, time; f = h5py.File(sys.argv[1], 'r')",
    "handle": "import sys, time; f = open(sys.argv[1], 'rb')",
}
ON_WINDOWS = pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows blocks replacing a file another process holds open")


@ON_WINDOWS
@pytest.mark.parametrize("holder", sorted(HOLDERS))
def test_a_held_workspace_refuses_the_write_and_stays_whole(saved: Path, holder: str) -> None:
    """#2857: a reader's handle on the old file is a typed refusal, not a raw OSError."""
    before = _digest(saved)
    code = HOLDERS[holder] + "; print('ready', flush=True); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code, str(saved)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        assert line == "ready", f"the holder never opened the file: {line!r}"
        with pytest.raises(WorkspaceWriteRefused) as caught:
            write_workspace(saved, views=[ViewDef("later", {"camera": "top"})])
    finally:
        proc.kill()  # only the holder this test started
        proc.wait(timeout=30)
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()
    assert caught.value.reason == "held-open"
    assert str(saved) in str(caught.value) and "another program holds" in str(caught.value)
    assert isinstance(caught.value.__cause__, PermissionError)
    assert _digest(saved) == before, "the held file changed"
    assert not _partials(saved), "the refused write left its partial behind"
    assert sorted(p.name for p in saved.parent.iterdir()) == ["frame.h5", saved.name]
    write_workspace(saved, views=[ViewDef("later", {"camera": "top"})])  # released: it goes
    assert [v.name for v in WorkspaceFile(saved).views] == ["iso", "later"]


@ON_WINDOWS
def test_a_read_only_workspace_keeps_its_own_error(saved: Path) -> None:
    """A read-only target also answers WinError 5, but nothing holds it: not ``held-open``."""
    before = _digest(saved)
    os.chmod(saved, stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            write_workspace(saved, views=[ViewDef("later", {"camera": "top"})])
    finally:
        os.chmod(saved, stat.S_IREAD | stat.S_IWRITE)
    assert _digest(saved) == before
    assert not _partials(saved)


def test_a_killed_manifest_writer_leaves_the_previous_manifest_whole(saved: Path) -> None:
    manifest = write_variants(saved.with_name("study.variants.json"),
                              [VariantDef("A", "frame.h5", {"pass": 1})])
    before = _digest(manifest)
    _kill_writer_at("manifest_before_replace", manifest)
    assert _digest(manifest) == before
    assert read_variants(manifest).variants[0].params == {"pass": 1}
    assert _partials(manifest)
