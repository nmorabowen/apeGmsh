"""A killed writer leaves the previous file whole (ADR 0130 D2).

The plan's kill criterion: if this fails, the atomic-write premise of D2 does not
hold on this platform. The writer runs in a real subprocess and is killed by the
OS (``TerminateProcess`` / ``SIGKILL``), not by an exception, so no ``finally``
runs. A patched step parks it at a known point so the kill is not a race.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from apeGmsh.results import user as user_package
from apeGmsh.results.user import UserFile, write_user_file
from apeGmsh.results.user import _writer

from tests.results.user._user_helpers import REPO, bucket, elements, nodal

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


TWO_WRITERS = textwrap.dedent(
    """
    import sys
    import numpy as np
    from apeGmsh.results.user import UserResult, UserWriteRefused, _writer, write_user_file

    target, model, who, how = sys.argv[1:5]
    base = 100.0 if who == "B" else 0.0

    if how == "park":
        real = _writer.write_entry
        count = [0]
        def write_entry(group, r):
            real(group, r)
            count[0] += 1
            if count[0] == 1:
                group.file.flush()  # A's first entry is really in its partial
                print("PARKED", flush=True)
                sys.stdin.readline()
        _writer.write_entry = write_entry

    results = [
        UserResult(name=f"{who} {i}", target="nodes", basis="model", components=("x", "y", "z"),
                   ids=np.arange(1, 20001, dtype="i4"), data=np.full((1, 20000, 3), base + i))
        for i in range(3)
    ]
    try:
        write_user_file(target, model, results=results)
    except UserWriteRefused as exc:
        print("REFUSED", exc, flush=True)
    else:
        print("WROTE", flush=True)
    """
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partials(target: Path) -> list[Path]:
    """Every partial of this target, whichever writer's."""
    return sorted(p for p in target.parent.iterdir()
                  if p.name.startswith(target.name + ".") and p.name.endswith(".partial"))


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
    for partial in _partials(target):
        _wait_for_the_dead_writers_lock(partial)


def _wait_for_the_dead_writers_lock(partial: Path) -> None:
    """Windows releases a dead process's locks some time after it exits, and how
    long depends on the load (``LockFileEx``). Reading the partial before then
    fails with a lock violation, which is the OS and not the test's subject."""
    deadline = time.monotonic() + 60.0
    while partial.exists() and time.monotonic() < deadline:
        try:
            with h5py.File(partial, "r"):
                return
        except OSError as exc:
            if "lock" not in str(exc):
                return  # a partial truly cut short: the lock is not what stops it
            time.sleep(0.05)


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
    assert _partials(target), "the kill came before any partial: the test proved nothing"


def test_the_partial_really_held_new_data_when_the_writer_died(make_model, tmp_path: Path) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    _kill_writer_at("before_replace", target, model)
    with h5py.File(_partials(target)[0], "r") as f:
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
    assert _partials(target) == [], "the dead writer's partial was not swept"


def _resumable_writer(target: Path, model: Path) -> subprocess.Popen:
    """A writer parked after its first entry, with real data flushed to its partial."""
    proc = subprocess.Popen(
        [sys.executable, "-c", TWO_WRITERS, str(target), str(model), "A", "park"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line == "PARKED", f"{line!r}\n{proc.stderr.read() if proc.stderr else ''}"
    return proc


def _finish(proc: subprocess.Popen) -> tuple[str, str]:
    out, err = proc.communicate(input="go\n", timeout=300)
    return out, err


def test_two_real_writers_neither_damages_the_other_and_the_second_refuses(make_model, tmp_path) -> None:
    """D1, the blocker. A parks with data in its partial; B writes from start to end
    meanwhile. With one shared partial name B truncated A's file, and A then passed
    its checks and published zeros."""
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    a = _resumable_writer(target, model)
    try:
        a_partials = _partials(target)
        assert len(a_partials) == 1 and a_partials[0].stat().st_size > 0
        size = a_partials[0].stat().st_size

        b = subprocess.run(
            [sys.executable, "-c", TWO_WRITERS, str(target), str(model), "B", "straight"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
            timeout=300,
        )
        assert b.returncode == 0 and "WROTE" in b.stdout, b.stderr

        assert a_partials[0].exists(), "B removed a live writer's partial"
        assert a_partials[0].stat().st_size == size, "B touched A's partial"
        uf = UserFile(target)
        assert uf.revision == 2
        assert sorted(e.name for e in uf.results().entries) == ["B 0", "B 1", "B 2", "kept"]

        out, err = _finish(a)
    finally:
        if a.poll() is None:
            a.kill()
            a.communicate()
    assert "REFUSED" in out and "changed while" in out, f"{out}\n{err}"
    uf = UserFile(target)
    assert uf.revision == 2, "the second writer overwrote the first"
    names = sorted(e.name for e in uf.results().entries)
    assert names == ["B 0", "B 1", "B 2", "kept"]
    for i in range(3):
        assert np.all(uf.read_result(f"B {i}").data == 100.0 + i), "the winner's data is not whole"
    assert _partials(target) == [], "A did not delete its own partial"


def test_the_first_writer_to_finish_wins_and_publishes_its_own_data(make_model, tmp_path) -> None:
    """The other order: the parked writer resumes with nobody in the way."""
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    a = _resumable_writer(target, model)
    try:
        out, err = _finish(a)
    finally:
        if a.poll() is None:
            a.kill()
            a.communicate()
    assert "WROTE" in out, f"{out}\n{err}"
    uf = UserFile(target)
    for i in range(3):
        assert np.all(uf.read_result(f"A {i}").data == float(i)), "zeros were published"


def test_a_first_write_that_lost_the_race_refuses(make_model, tmp_path, monkeypatch) -> None:
    model = make_model()
    target = tmp_path / "s.user.h5"
    real = _writer._verify

    def interleave(*a, **k):
        real(*a, **k)
        monkeypatch.setattr(_writer, "_verify", real)
        write_user_file(target, model, results=[nodal("other")])

    monkeypatch.setattr(_writer, "_verify", interleave)
    with pytest.raises(_writer.UserWriteRefused, match="created while"):
        write_user_file(target, model, results=[nodal("mine")])
    assert [e.name for e in UserFile(target).results().entries] == ["other"]
    assert _partials(target) == []


def test_data_that_does_not_read_back_refuses(make_model, tmp_path, monkeypatch) -> None:
    """What D1 published: a well-formed file whose values are zeros."""
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal("kept")])
    before = _digest(target)
    real = _writer.write_entry

    def zeroed(group, r):
        real(group, r)
        group["RESULTS/ON_NODES/values/DATA"][...] = 0.0

    monkeypatch.setattr(_writer, "write_entry", zeroed)
    with pytest.raises(_writer.UserWriteRefused, match="did not read back"):
        write_user_file(target, model, results=[nodal("new")])
    assert _digest(target) == before and _partials(target) == []


def test_element_data_that_does_not_read_back_refuses(make_model, tmp_path, monkeypatch) -> None:
    model = make_model()
    b = bucket()
    real = _writer.write_entry

    def zeroed(group, r):
        real(group, r)
        group[f"RESULTS/ON_ELEMENTS/values/{b.name}/DATA"][...] = 0.0

    monkeypatch.setattr(_writer, "write_entry", zeroed)
    with pytest.raises(_writer.UserWriteRefused, match="did not read back"):
        write_user_file(tmp_path / "s.user.h5", model, results=[elements(b)])


def test_nan_data_compares_by_bits(make_model, tmp_path) -> None:
    """NaN != NaN, so a value compare would refuse every field with a grey cell."""
    field = nodal()
    assert np.isnan(field.data).any()
    target = write_user_file(tmp_path / "s.user.h5", make_model(), results=[field])
    assert UserFile(target).revision == 1


def test_each_writer_names_its_own_partial(make_model, tmp_path, monkeypatch) -> None:
    model = make_model()
    seen = []
    real = _writer._verify

    def spy(partial, *a, **k):
        seen.append(Path(partial).name)
        real(partial, *a, **k)

    monkeypatch.setattr(_writer, "_verify", spy)
    target = tmp_path / "s.user.h5"
    write_user_file(target, model)
    write_user_file(target, model)
    assert len(set(seen)) == 2
    host = _writer._host_token()
    for name in seen:
        assert re.fullmatch(rf"s\.user\.h5\.{host}-{os.getpid()}-[0-9a-f]+\.partial", name), name


def _leftover(target: Path, pid: int, host: Optional[str] = None) -> Path:
    host = _writer._host_token() if host is None else host
    path = target.with_name(f"{target.name}.{host}-{pid}-00ff.partial")
    path.write_bytes(b"left behind")
    return path


@pytest.mark.parametrize(
    "raw, token",
    [("DESKTOP-AB12", "desktop_ab12"), ("ws.office.local", "ws_office_local"),
     ("--", "unknownhost"), ("Ñandú 7", "and_7")],
)
def test_the_host_token_is_filesystem_safe(monkeypatch, raw: str, token: str) -> None:
    monkeypatch.setattr(_writer.socket, "gethostname", lambda: raw)
    got = _writer._host_token()
    assert got == token and re.fullmatch(r"[a-z0-9_]+", got)


def test_a_partial_naming_another_host_is_never_swept(make_model, tmp_path) -> None:
    """A synced folder: that machine's live writer has a PID this one cannot see."""
    target = tmp_path / "s.user.h5"
    elsewhere = _leftover(target, _dead_pid(), host="another_machine")
    mine = _leftover(target, _dead_pid())
    write_user_file(target, make_model())
    assert elsewhere.exists(), "another host's partial was removed"
    assert not mine.exists(), "this host's dead writer's partial was not swept"


def test_a_host_that_cannot_be_told_sweeps_nothing(make_model, tmp_path, monkeypatch) -> None:
    def fail() -> str:
        raise OSError("no name")

    target = tmp_path / "s.user.h5"
    unknown = _leftover(target, _dead_pid(), host="unknownhost")
    monkeypatch.setattr(_writer.socket, "gethostname", fail)
    write_user_file(target, make_model())
    assert unknown.exists()
    assert _partials(target) == [unknown], "the write's own partial was not cleaned up"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=120)
    return proc.pid


def test_a_dead_writers_partial_is_swept_and_a_live_ones_is_not(make_model, tmp_path) -> None:
    target = tmp_path / "s.user.h5"
    dead = _leftover(target, _dead_pid())
    live = _leftover(target, os.getpid())
    other_stem = tmp_path / f"other.user.h5.{_writer._host_token()}-{_dead_pid()}-00ff.partial"
    other_stem.write_bytes(b"not this file's")
    write_user_file(target, make_model())
    assert not dead.exists()
    assert live.exists(), "a live writer's partial was removed"
    assert other_stem.exists(), "another stem's partial was removed"


def test_a_sweep_that_fails_does_not_fail_the_write(make_model, tmp_path, monkeypatch) -> None:
    """The first leftover the sweep meets refuses; the write stands, and the sweep
    goes on to the next leftover."""
    target = tmp_path / "s.user.h5"
    leftovers = {_leftover(target, _dead_pid()), _leftover(target, _dead_pid())}
    real = Path.unlink
    refused: list[Path] = []

    def refuse_first(self, *a, **k):
        if self in leftovers and not refused:
            refused.append(self)
            raise PermissionError("a dead writer's lock outlives it for a moment")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", refuse_first)
    write_user_file(target, make_model())
    assert target.exists() and len(refused) == 1
    assert refused[0].exists()
    (swept,) = leftovers - set(refused)
    assert not swept.exists(), "one failed removal stopped the sweep"


def test_a_failed_write_deletes_only_its_own_partial(make_model, tmp_path, monkeypatch) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model)
    live = _leftover(target, os.getpid())

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(_writer, "write_entry", boom)
    with pytest.raises(RuntimeError):
        write_user_file(target, model, results=[nodal("new")])
    assert _partials(target) == [live]


def test_asking_whether_a_process_is_alive_does_not_kill_it() -> None:
    """``os.kill(pid, 0)`` terminates the process on Windows."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        assert _writer._pid_alive(proc.pid) is True
        time.sleep(0.5)
        assert proc.poll() is None, "the liveness check killed the process"
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert _writer._pid_alive(proc.pid) is False
    assert _writer._pid_alive(os.getpid()) is True


@pytest.mark.parametrize("root, key", [("results", "user_results_schema_version"),
                                       ("combinations", "combinations_schema_version")])
def test_a_zone_group_with_no_version_key_refuses_the_write(make_model, tmp_path, root, key) -> None:
    model = make_model()
    target = write_user_file(tmp_path / "s.user.h5", model, results=[nodal()], combinations=[])
    with h5py.File(target, "a") as f:
        del f["meta"].attrs[key]
    before = _digest(target)
    with pytest.raises(_writer.UserWriteRefused, match=key):
        write_user_file(target, model)
    assert _digest(target) == before and _partials(target) == []


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
    assert _partials(target) == []


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
    assert _partials(target) == []


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
    assert _partials(target) == []
