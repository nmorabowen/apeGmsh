"""ADR 0095 S5j — Opus harden follow-up for S5g–S5i."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from apeGmsh.studio import collect_status, format_status
from apeGmsh.studio.__main__ import main
from apeGmsh.studio._busy import (
    _TORN_GRACE_S,
    read_busy,
    release_busy,
    try_acquire_busy,
)
from apeGmsh.studio._contract import CONTRACT_VERSION, load_fixture
from apeGmsh.studio._host_state import claim_host, clear_host, read_host
from apeGmsh.studio._mcp import run_until
from apeGmsh.studio._project import (
    OutsideRootError,
    resolve_entry_script,
    write_project,
)
from apeGmsh.studio._replay import ReplayResult


class _StubSession:
    def __init__(self) -> None:
        self.name = "stub"
        self.is_active = True

    def end(self) -> None:
        self.is_active = False


def test_fixture_contract_version_matches_constant() -> None:
    assert load_fixture("status")["contract_version"] == CONTRACT_VERSION


def test_torn_busy_not_stolen_during_grace(tmp_path: Path) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    busy = habitat / "busy.json"
    busy.write_text("", encoding="utf-8")
    assert try_acquire_busy(tmp_path, op="run_until") is False
    assert read_busy(tmp_path)["stale"] is True


def test_torn_busy_stolen_after_grace(tmp_path: Path) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    busy = habitat / "busy.json"
    busy.write_text("", encoding="utf-8")
    old = time.time() - (_TORN_GRACE_S + 1.0)
    os.utime(busy, (old, old))
    assert try_acquire_busy(tmp_path, op="run_until") is True
    release_busy(tmp_path)


def test_release_busy_ignores_foreign_pid(tmp_path: Path, monkeypatch) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    foreign = os.getpid() + 1_000_003
    (habitat / "busy.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "busy": True,
                "pid": foreign,
                "op": "run_until",
                "phase": "model",
                "script": "box.py",
                "ts": "2026-08-16T06:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "apeGmsh.studio._busy.pid_alive",
        lambda pid: pid == foreign,
    )
    release_busy(tmp_path)
    data = json.loads((habitat / "busy.json").read_text(encoding="utf-8"))
    assert data["busy"] is True
    assert data["pid"] == foreign


def test_clear_host_ignores_foreign_live_pid(tmp_path: Path, monkeypatch) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    foreign = os.getpid() + 1_000_007
    (habitat / "host.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "running": True,
                "pid": foreign,
                "phase": "model",
                "ts": "2026-08-16T06:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "apeGmsh.studio._host_state.pid_alive",
        lambda pid: pid == foreign,
    )
    assert clear_host(tmp_path) is None
    host = read_host(tmp_path)
    assert host["running"] is True
    assert host["pid"] == foreign


def test_claim_host_refuses_foreign_live(tmp_path: Path, monkeypatch) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    foreign = os.getpid() + 1_000_009
    (habitat / "host.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "running": True,
                "pid": foreign,
                "phase": "mesh",
                "ts": "2026-08-16T06:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "apeGmsh.studio._host_state.pid_alive",
        lambda pid: pid == foreign,
    )
    assert claim_host(tmp_path, phase="model") is None
    assert read_host(tmp_path)["pid"] == foreign


def test_project_outside_root_refused(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "box.py"
    outside.parent.mkdir()
    outside.write_text("pass\n", encoding="utf-8")
    habitat = tmp_path / "proj"
    habitat.mkdir()
    write_project(habitat, script=outside)
    with pytest.raises(OutsideRootError):
        resolve_entry_script(None, root=habitat)
    result = run_until(None, phase="model", root=habitat)
    assert result["ok"] is False
    assert result["error"]["code"] == "OUTSIDE_ROOT"


def test_cli_uses_project_json(tmp_path: Path, monkeypatch, capsys) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    write_project(tmp_path, script=script)
    stub = _StubSession()

    def fake_run_until(self, path, *, phase="model"):
        assert Path(path).resolve() == script.resolve()
        return ReplayResult(
            ok=True,
            phase=phase,
            geometry_hash="abc",
            error=None,
            session=stub,
        )

    monkeypatch.setattr(
        "apeGmsh.studio._replay.ReplayRunner.run_until", fake_run_until
    )
    code = main(["--no-viewer", "--root", str(tmp_path)])
    assert code == 0
    assert "replay ok" in capsys.readouterr().out


def test_cli_relative_script_resolves_under_root(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "src" / "box.py"
    script.parent.mkdir()
    script.write_text("pass\n", encoding="utf-8")
    stub = _StubSession()
    seen: list[Path] = []

    def fake_run_until(self, path, *, phase="model"):
        seen.append(Path(path).resolve())
        return ReplayResult(
            ok=True,
            phase=phase,
            geometry_hash="abc",
            error=None,
            session=stub,
        )

    monkeypatch.setattr(
        "apeGmsh.studio._replay.ReplayRunner.run_until", fake_run_until
    )
    monkeypatch.chdir(tmp_path / "src")
    code = main(["src/box.py", "--no-viewer", "--root", str(tmp_path)])
    assert code == 0
    assert seen == [script.resolve()]


def test_format_status_prints_root(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "phase": "model",
                "labels": [],
                "physical_groups": [],
                "counts": {"entities": {}, "elements": {}},
                "entities": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    text = format_status(collect_status(tmp_path))
    assert f"root: {tmp_path.resolve()}" in text
