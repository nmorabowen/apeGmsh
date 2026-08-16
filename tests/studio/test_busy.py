"""ADR 0095 S5h — habitat BUSY lock for concurrent run_until."""
from __future__ import annotations

import json
import os
from pathlib import Path

from apeGmsh.studio import collect_status
from apeGmsh.studio._busy import (
    read_busy,
    release_busy,
    try_acquire_busy,
)
from apeGmsh.studio._mcp import run_until
from apeGmsh.studio._replay import ReplayRunner


def test_acquire_and_release(tmp_path: Path) -> None:
    assert try_acquire_busy(tmp_path, op="run_until", phase="model") is True
    info = read_busy(tmp_path)
    assert info["busy"] is True
    assert info["pid"] == os.getpid()
    assert try_acquire_busy(tmp_path, op="run_until") is False
    release_busy(tmp_path)
    assert read_busy(tmp_path)["busy"] is False
    assert try_acquire_busy(tmp_path, op="run_until") is True
    release_busy(tmp_path)


def test_stale_busy_is_stolen(tmp_path: Path) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    (habitat / "busy.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "busy": True,
                "pid": 999_999_999,
                "op": "run_until",
                "phase": "model",
                "script": "box.py",
                "ts": "2026-08-16T06:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stale = read_busy(tmp_path)
    assert stale["busy"] is False
    assert stale["stale"] is True
    assert try_acquire_busy(tmp_path, op="run_until") is True
    release_busy(tmp_path)


def test_replay_runner_busy(tmp_path: Path) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    assert try_acquire_busy(tmp_path, op="run_until", phase="mesh") is True
    result = ReplayRunner(root=tmp_path).run_until(script, phase="model", root=tmp_path)
    assert result.ok is False
    assert result.error is not None
    assert result.error.startswith("BUSY:")
    release_busy(tmp_path)


def test_mcp_run_until_busy(tmp_path: Path) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    assert try_acquire_busy(tmp_path, op="run_until") is True
    result = run_until(script, phase="model", root=tmp_path)
    assert result["ok"] is False
    assert result["error"]["code"] == "BUSY"
    release_busy(tmp_path)


def test_status_includes_busy(tmp_path: Path) -> None:
    assert try_acquire_busy(tmp_path, op="run_until", phase="model") is True
    payload = collect_status(tmp_path)
    assert payload["busy"]["busy"] is True
    assert payload["busy"]["op"] == "run_until"
    release_busy(tmp_path)
