"""ADR 0095 S5g — host presence + project.json entry script."""
from __future__ import annotations

import json
import os
from pathlib import Path

from apeGmsh.studio import collect_status
from apeGmsh.studio._host_state import claim_host, clear_host, read_host
from apeGmsh.studio._mcp import run_until
from apeGmsh.studio._project import (
    read_project,
    resolve_entry_script,
    write_project,
)


def test_write_and_read_project(tmp_path: Path) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    write_project(tmp_path, script=script)
    proj = read_project(tmp_path)
    assert proj is not None
    assert proj["schema"] == 1
    assert proj["script"] == "box.py"
    assert resolve_entry_script(None, root=tmp_path) == script.resolve()


def test_resolve_entry_script_explicit_wins(tmp_path: Path) -> None:
    write_project(tmp_path, script=tmp_path / "a.py")
    other = tmp_path / "b.py"
    other.write_text("pass\n", encoding="utf-8")
    assert resolve_entry_script(other, root=tmp_path) == other.resolve()


def test_run_until_uses_project_json(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    write_project(tmp_path, script=script)
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = run_until(None, phase="model", root=tmp_path)
    assert result["ok"] is True
    assert str(script.resolve()) in recorded["argv"]


def test_run_until_no_script_no_project(tmp_path: Path) -> None:
    result = run_until(None, phase="model", root=tmp_path)
    assert result["ok"] is False
    assert result["error"]["code"] == "NO_SCRIPT"


def test_claim_and_clear_host(tmp_path: Path) -> None:
    claim_host(tmp_path, phase="mesh")
    host = read_host(tmp_path)
    assert host["running"] is True
    assert host["pid"] == os.getpid()
    assert host["phase"] == "mesh"
    assert host["stale"] is False
    clear_host(tmp_path)
    host = read_host(tmp_path)
    assert host["running"] is False
    assert host["phase"] is None


def test_stale_host_when_pid_dead(tmp_path: Path) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    (habitat / "host.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "running": True,
                "pid": 999_999_999,
                "phase": "model",
                "ts": "2026-08-16T06:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    host = read_host(tmp_path)
    assert host["running"] is False
    assert host["stale"] is True
    assert host["pid"] == 999_999_999


def test_status_includes_host_and_project(tmp_path: Path) -> None:
    write_project(tmp_path, script=tmp_path / "box.py")
    claim_host(tmp_path, phase="model")
    payload = collect_status(tmp_path)
    assert payload["project"]["script"] == "box.py"
    assert payload["host"]["running"] is True
    assert payload["host"]["phase"] == "model"
    clear_host(tmp_path)
