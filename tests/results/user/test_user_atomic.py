"""A killed writer leaves the previous file whole (ADR 0130 D2).

The plan's kill criterion: if this fails, the atomic-write premise of D2 does not
hold on this platform. The writer runs in a real subprocess and is killed by the
OS (``TerminateProcess`` / ``SIGKILL``), not by an exception, so no ``finally``
runs. A patched step parks it at a known point so the kill is not a race.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from apeGmsh.results import user as user_package
from apeGmsh.results.user import UserFile, write_user_file
from apeGmsh.results.user import _writer

from .conftest import REPO, nodal

SCRIPT = textwrap.dedent(
    """
    import sys, time
    import numpy as np
    from apeGmsh.results.user import UserResult, _writer, write_user_file

    target, model, point = sys.argv[1:4]

    def park():
        print("PARKED", flush=True)
        time.sleep(120)

    if point == "mid_results":
        real = _writer.write_entry
        count = [0]
        def write_entry(group, r):
            count[0] += 1
            if count[0] == 3:
                group.file.flush()   # what is written so far really reaches the partial
                park()
            real(group, r)
        _writer.write_entry = write_entry
    elif point == "before_replace":
        _writer._replace = lambda partial, dst: park()
    else:
        raise SystemExit(f"unknown point {point}")

    results = [
        UserResult(name=f"new {i}", target="domain", basis="model", components=("x",),
                   data=np.full((1, 1, 1), float(i)))
        for i in range(5)
    ]
    write_user_file(target, model, results=results)
    print("FINISHED", flush=True)
    """
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kill_writer_at(point: str, target: Path, model: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(target), str(model), point],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert proc.stdout is not None
        line: Optional[str] = proc.stdout.readline().strip()
        assert line == "PARKED", (
            f"the writer never reached {point!r}: {line!r}\n{proc.stderr.read() if proc.stderr else ''}"
        )
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
    assert proc.returncode != 0, "the writer was not killed: the test proved nothing"


def test_the_subprocess_imports_this_worktree() -> None:
    """The editable install points at another checkout; the script must not."""
    out = subprocess.run(
        [sys.executable, "-c", "import apeGmsh.results.user as u; print(u.__file__)"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=True,
    ).stdout.strip()
    assert Path(out).resolve() == Path(user_package.__file__).resolve()


@pytest.mark.parametrize("point", ["mid_results", "before_replace"])
def test_a_killed_writer_leaves_the_previous_file_whole(make_model, tmp_path: Path, point: str) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    before = _digest(target)

    _kill_writer_at(point, target, model)

    assert _digest(target) == before, "the previous file changed under a killed writer"
    uf = UserFile(target)
    assert uf.revision == 1
    assert [e.name for e in uf.results().entries] == ["kept"]
    assert np.isnan(uf.read_result("kept").data[0, 1, 0])
    partial = target.with_name(target.name + ".partial")
    assert partial.exists(), "the kill came before any partial: the test proved nothing"


def test_the_partial_really_held_new_data_when_the_writer_died(make_model, tmp_path: Path) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    _kill_writer_at("before_replace", target, model)
    with h5py.File(target.with_name(target.name + ".partial"), "r") as f:
        assert len(f["results"]) == 6, "complete and unpublished"


def test_a_killed_first_write_leaves_no_file(make_model, tmp_path: Path) -> None:
    target = tmp_path / "s.user.h5"
    _kill_writer_at("before_replace", target, make_model())
    assert not target.exists()


def test_a_later_write_survives_the_dead_writers_partial(make_model, tmp_path: Path) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    _kill_writer_at("mid_results", target, model)
    write_user_file(target, model, results=[nodal("later")])
    assert sorted(e.name for e in UserFile(target).results().entries) == ["kept", "later"]
    assert not target.with_name(target.name + ".partial").exists()


def test_an_exception_mid_write_leaves_the_file_and_no_partial(
    make_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    before = _digest(target)

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(_writer, "write_entry", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_user_file(target, model, results=[nodal("new")])
    assert _digest(target) == before
    assert not target.with_name(target.name + ".partial").exists()


def test_a_revision_that_moved_during_the_write_aborts(
    make_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two writers: the second to finish must not erase the first one's save."""
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("a")])
    real = _writer._verify

    def interleave(*a, **k):
        real(*a, **k)
        with h5py.File(target, "a") as f:  # the other writer's save lands here
            f["meta"].attrs["revision"] = np.int32(2)

    monkeypatch.setattr(_writer, "_verify", interleave)
    with pytest.raises(_writer.UserWriteRefused, match="changed while"):
        write_user_file(target, model, results=[nodal("mine")])
    uf = UserFile(target)
    assert uf.revision == 2 and [e.name for e in uf.results().entries] == ["a"]
    assert not target.with_name(target.name + ".partial").exists()


def test_replace_retries_a_transient_lock(
    make_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model()
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("held by a viewer")
        real(src, dst)

    monkeypatch.setattr(_writer.os, "replace", flaky)
    monkeypatch.setattr(_writer, "_REPLACE_WAIT_S", 0.0)
    write_user_file(tmp_path / "s.user.h5", model)
    assert len(calls) == 3 and (tmp_path / "s.user.h5").exists()


def test_a_lock_that_never_clears_raises_and_keeps_the_previous_file(
    make_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model)
    before = _digest(target)

    def locked(src, dst):
        raise PermissionError("held")

    monkeypatch.setattr(_writer.os, "replace", locked)
    monkeypatch.setattr(_writer, "_REPLACE_WAIT_S", 0.0)
    with pytest.raises(PermissionError):
        write_user_file(target, model, results=[nodal("new")])
    assert _digest(target) == before
    assert not target.with_name(target.name + ".partial").exists()
