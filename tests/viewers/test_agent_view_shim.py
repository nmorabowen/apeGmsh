"""The ADR 0128 D7 shim (``apeGmsh.viewers.agent_view``) against a stub executable.

The stub stands in for ``apeGmshViewer`` at its command line, the
contract: it reads ``--render <spec> --out <dir>``, refuses an ``--out``
that is not empty (exit 2, nothing written), and otherwise writes the
manifest and exit code its mode names, as ``apeGmshViewer/src/main/main.ts``
does for ``--mode=render`` (0 every shot drawn, 1 a shot refused or the
run failed, 2 the spec refused). No real viewer runs here.
"""
from __future__ import annotations

import ast
import json
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from apeGmsh.viewers import agent_view
from apeGmsh.viewers.agent_view import AgentViewError, find_viewer, render_stills

STUB = r'''
import json, os, sys, time
from pathlib import Path

args = sys.argv[1:]
Path(os.environ["STUB_LOG"]).write_text(json.dumps(args), encoding="utf-8")
assert args[0] == "--render" and args[2] == "--out", args
spec_path, out = Path(args[1]), Path(args[3])
mode = os.environ.get("STUB_MODE", "ok")
if out.exists() and any(out.iterdir()):
    sys.stderr.write(f"apeGmshViewer: --out refused: {out} is not empty\n")
    sys.exit(2)
spec = json.loads(spec_path.read_text(encoding="utf-8"))
Path(os.environ["STUB_LOG"] + ".spec").write_text(json.dumps(spec), encoding="utf-8")
out.mkdir(parents=True, exist_ok=True)

def entry(name, ok=True):
    return {"name": name, "status": "ok" if ok else "refused",
            "reason": None if ok else f"stage {name} is not in the file",
            "png": f"{name}.png" if ok else None}

def manifest(shots):
    return {"manifest": "apegmsh-manifest/1", "spec": "apegmsh-view/1", "size": spec.get("size"),
            "theme": "light", "shots": shots, "timing": {"shots": {}}}

def write(m):
    (out / "manifest.json").write_text(json.dumps(m), encoding="utf-8")

names = [s["name"] for s in spec["shots"]]
if mode == "ok":
    for n in names:
        (out / f"{n}.png").write_bytes(b"\x89PNG")
    write(manifest([entry(n) for n in names])); sys.exit(0)
if mode == "shot_refused":
    write(manifest([entry(n, ok=(i == 0)) for i, n in enumerate(names)])); sys.exit(1)
if mode == "spec_refused":
    write({"manifest": "apegmsh-manifest/1", "spec": None, "spec_file": str(spec_path),
           "refused": {"key": "shots[0].camera", "reason": "unknown camera 'isometric'"}, "shots": []})
    sys.stderr.write("apeGmshViewer: the spec is refused: shots[0].camera\n"); sys.exit(2)
if mode == "zero_but_refused":
    write(manifest([entry(n, ok=False) for n in names])); sys.exit(0)
if mode == "zero_no_manifest":
    sys.exit(0)
if mode == "zero_not_object":
    (out / "manifest.json").write_text("[1, 2]", encoding="utf-8"); sys.exit(0)
if mode == "crash":
    sys.stderr.write("apeGmshViewer: render failed: the renderer process is gone (crashed)\n"); sys.exit(1)
if mode == "nonzero_clean":
    # main.ts abandonRender: a manifest is already whole, and the run still exits 1.
    write(manifest([entry(n) for n in names])); sys.exit(1)
if mode == "hang":
    # A child of its own that holds the pipes, like Electron's GPU and renderer processes.
    import subprocess
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    time.sleep(20); sys.exit(0)
sys.exit(99)
'''


def _spec(**over):
    spec = {
        "spec": "apegmsh-view/1",
        "files": {"model": "frame.h5"},
        "size": [800, 500],
        "shots": [{"name": "mesh", "camera": "iso"}, {"name": "uz", "stage": "gravity"}],
    }
    spec.update(over)
    return spec


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """A stub executable on ``$APEGMSH_VIEWER``; returns (exe, log path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "stub_viewer.py"
    script.write_text(STUB, encoding="utf-8")
    if sys.platform == "win32":
        exe = bin_dir / "apeGmshViewer.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        exe = bin_dir / "apeGmshViewer"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "argv.json"
    monkeypatch.setenv("APEGMSH_VIEWER", str(exe))
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_MODE", "ok")
    return exe, log


# -- finding the executable -------------------------------------------------------


def test_env_names_the_executable(stub):
    exe, _ = stub
    assert find_viewer() == exe


def test_env_naming_no_file_is_refused_even_with_an_install(tmp_path, monkeypatch):
    local = tmp_path / "local"
    install = local.joinpath("Programs", "apegmsh-viewer", "apeGmshViewer.exe")
    install.parent.mkdir(parents=True)
    install.write_bytes(b"")
    monkeypatch.setattr(agent_view.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("APEGMSH_VIEWER", str(tmp_path / "missing.exe"))
    with pytest.raises(FileNotFoundError, match=r"APEGMSH_VIEWER is .*missing\.exe.*not a file"):
        find_viewer()


def test_the_per_user_install_when_env_is_unset(tmp_path, monkeypatch):
    local = tmp_path / "local"
    install = local.joinpath("Programs", "apegmsh-viewer", "apeGmshViewer.exe")
    install.parent.mkdir(parents=True)
    install.write_bytes(b"")
    monkeypatch.setattr(agent_view.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv("APEGMSH_VIEWER", raising=False)
    assert find_viewer() == install


def test_not_installed_names_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_view.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("APEGMSH_VIEWER", raising=False)
    with pytest.raises(FileNotFoundError, match=r"not installed at .*apegmsh-viewer.*apeGmshViewer\.exe"):
        find_viewer()


def test_no_install_off_windows(monkeypatch):
    monkeypatch.setattr(agent_view.sys, "platform", "linux")
    monkeypatch.delenv("APEGMSH_VIEWER", raising=False)
    with pytest.raises(FileNotFoundError, match="no per-user install"):
        find_viewer()


# -- the spawn contract -----------------------------------------------------------


def test_one_call_returns_the_manifest(stub, tmp_path, monkeypatch):
    _, log = stub
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    spec = _spec()
    out = tmp_path / "stills"
    manifest = render_stills(spec, out)
    argv = json.loads(log.read_text(encoding="utf-8"))
    # The ADR 0128 D4 spelling, `--out` absolute.
    assert argv[0] == "--render" and argv[2] == "--out" and argv[3] == str(out.resolve())
    assert len(argv) == 4
    # The manifest exactly as the app wrote it.
    assert manifest == json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [s["png"] for s in manifest["shots"]] == ["mesh.png", "uz.png"]
    # A dict's relative `files` mean the working folder; nothing else changes, and the caller's dict is untouched.
    written = json.loads(Path(str(log) + ".spec").read_text(encoding="utf-8"))
    assert written["files"] == {"model": str((work / "frame.h5").resolve())}
    assert {k: v for k, v in written.items() if k != "files"} == {k: v for k, v in spec.items() if k != "files"}
    assert spec["files"] == {"model": "frame.h5"}


def test_a_spec_file_is_passed_as_it_is(stub, tmp_path):
    _, log = stub
    spec_file = tmp_path / "views" / "frame.view.json"
    spec_file.parent.mkdir()
    text = json.dumps(_spec())
    spec_file.write_text(text, encoding="utf-8")
    render_stills(spec_file, tmp_path / "stills")
    argv = json.loads(log.read_text(encoding="utf-8"))
    assert argv[1] == str(spec_file.resolve())
    # The app reads its relative paths against the file's own folder: the shim leaves them.
    assert spec_file.read_text(encoding="utf-8") == text


def test_an_explicit_viewer_wins_over_the_env(stub, tmp_path, monkeypatch):
    exe, log = stub
    monkeypatch.setenv("APEGMSH_VIEWER", str(tmp_path / "missing.exe"))
    render_stills(_spec(), tmp_path / "stills", viewer=exe)
    assert log.is_file()


# -- fail closed ------------------------------------------------------------------


def test_a_refused_shot_raises_with_the_manifest(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "shot_refused")
    with pytest.raises(AgentViewError, match=r"exited 1: shot 'uz' is refused: stage uz is not in the file") as e:
        render_stills(_spec(), tmp_path / "stills")
    assert e.value.returncode == 1
    assert e.value.manifest is not None
    assert [s["status"] for s in e.value.manifest["shots"]] == ["ok", "refused"]


def test_a_refused_spec_names_its_key(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "spec_refused")
    with pytest.raises(AgentViewError, match=r"exited 2: the run is refused: shots\[0\]\.camera: unknown camera") as e:
        render_stills(_spec(), tmp_path / "stills")
    assert e.value.returncode == 2
    assert "the spec is refused" in e.value.stderr


def test_an_earlier_manifest_in_out_is_never_read(stub, tmp_path):
    out = tmp_path / "stills"
    out.mkdir()
    earlier = {"manifest": "apegmsh-manifest/1", "shots": [{"name": "old", "status": "ok"}]}
    (out / "manifest.json").write_text(json.dumps(earlier), encoding="utf-8")
    with pytest.raises(AgentViewError, match=r"exited 2 \(.*manifest\.json was there before the run") as e:
        render_stills(_spec(), out)
    assert e.value.manifest is None
    assert "--out refused" in str(e.value)


def test_exit_0_with_a_refused_shot_is_refused(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "zero_but_refused")
    with pytest.raises(AgentViewError, match=r"exited 0, but its manifest states: shot 'mesh' is refused"):
        render_stills(_spec(), tmp_path / "stills")


def test_exit_0_without_a_manifest_is_refused(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "zero_no_manifest")
    with pytest.raises(AgentViewError, match=r"exited 0, but it wrote no manifest\.json"):
        render_stills(_spec(), tmp_path / "stills")


def test_a_manifest_that_is_not_an_object_is_refused(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "zero_not_object")
    with pytest.raises(AgentViewError, match="not a JSON object"):
        render_stills(_spec(), tmp_path / "stills")


def test_a_crash_raises_with_stderr(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "crash")
    # Not one regex over the message: on Windows the stub runs through cmd.exe, whose
    # AutoRun may print to stderr first (seen in a full-suite run).
    with pytest.raises(AgentViewError, match=r"^apeGmshViewer --render exited 1\nstderr: ") as e:
        render_stills(_spec(), tmp_path / "stills")
    assert "the renderer process is gone" in str(e.value)
    assert "the renderer process is gone" in e.value.stderr
    assert e.value.manifest is None


def test_a_nonzero_exit_with_a_clean_manifest_is_refused(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "nonzero_clean")
    with pytest.raises(AgentViewError, match=r"^apeGmshViewer --render exited 1") as e:
        render_stills(_spec(), tmp_path / "stills")
    assert e.value.returncode == 1
    assert [s["status"] for s in e.value.manifest["shots"]] == ["ok", "ok"]


def test_a_run_past_its_timeout_is_killed_with_its_children(stub, tmp_path, monkeypatch):
    """The wrapper (cmd.exe on Windows), the stub and the stub's own child all sleep 20 s:
    the bound holds only when the whole tree is killed, since the child keeps the pipes."""
    monkeypatch.setenv("STUB_MODE", "hang")
    t0 = time.monotonic()
    with pytest.raises(AgentViewError, match="did not exit within 1 s") as e:
        render_stills(_spec(), tmp_path / "stills", timeout=1)
    elapsed = time.monotonic() - t0
    assert e.value.returncode is None
    assert elapsed < 6, f"the timeout of 1 s took {elapsed:.1f} s: a child outlived the kill"


def test_an_interrupt_kills_the_tree_and_propagates_unchanged(stub, tmp_path, monkeypatch):
    """Ctrl-C while the app runs: the whole tree is killed, and the same KeyboardInterrupt goes on."""
    monkeypatch.setenv("STUB_MODE", "hang")
    interrupt = KeyboardInterrupt("test interrupt")
    real_communicate = subprocess.Popen.communicate
    calls = []

    def communicate(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(1.0)  # the stub has started the child that holds the pipes
            raise interrupt
        return real_communicate(self, *args, **kwargs)

    killed = []
    real_kill = agent_view._kill_tree

    def kill_tree(proc):
        killed.append(proc)
        return real_kill(proc)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    monkeypatch.setattr(agent_view, "_kill_tree", kill_tree)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt) as e:
        render_stills(_spec(), tmp_path / "stills")
    elapsed = time.monotonic() - t0
    assert e.value is interrupt
    assert len(killed) == 1 and killed[0].poll() is not None
    assert elapsed < 7, f"the kill took {elapsed:.1f} s: a child outlived it"


def test_nan_in_a_dict_spec_is_refused_before_any_spawn(stub, tmp_path):
    _, log = stub
    with pytest.raises(ValueError):
        render_stills(_spec(size=[float("nan"), 500]), tmp_path / "stills")
    assert not log.exists()


def test_the_shim_imports_nothing_from_the_app():
    """ADR 0128 D7: a process boundary and files, never an import."""
    src = Path(agent_view.__file__).read_text(encoding="utf-8")
    mods = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert mods <= {"__future__", "json", "os", "signal", "subprocess", "sys", "tempfile", "collections.abc", "pathlib", "typing"}, mods
