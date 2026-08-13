"""``python -m apeGmsh.studio`` argparse + dispatch (no Qt)."""
from __future__ import annotations

from pathlib import Path

from apeGmsh.studio.__main__ import main
from apeGmsh.studio._replay import ReplayResult


class _StubSession:
    def __init__(self) -> None:
        self.name = "stub"
        self.is_active = True
        self.ended = False

    def end(self) -> None:
        self.is_active = False
        self.ended = True


def test_missing_script_exits_2(tmp_path: Path) -> None:
    code = main([str(tmp_path / "nope.py")])
    assert code == 2


def test_no_viewer_replays_and_ends(tmp_path: Path, monkeypatch, capsys) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    stub = _StubSession()

    def fake_run_until(self, path, *, phase="model"):
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
    code = main([str(script), "--no-viewer"])
    assert code == 0
    assert stub.ended is True
    out = capsys.readouterr().out
    assert "replay ok" in out


def test_failed_replay_exits_1(tmp_path: Path, monkeypatch, capsys) -> None:
    script = tmp_path / "bad.py"
    script.write_text("pass\n", encoding="utf-8")

    def fake_run_until(self, path, *, phase="model"):
        return ReplayResult(
            ok=False,
            phase=phase,
            geometry_hash=None,
            error="Traceback: boom",
            session=None,
        )

    monkeypatch.setattr(
        "apeGmsh.studio._replay.ReplayRunner.run_until", fake_run_until
    )
    code = main([str(script), "--no-viewer"])
    assert code == 1
    err = capsys.readouterr().err
    assert "boom" in err


def test_no_session_exits_2(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "empty.py"
    script.write_text("print(1)\n", encoding="utf-8")

    def fake_run_until(self, path, *, phase="model"):
        return ReplayResult(
            ok=True,
            phase=phase,
            geometry_hash="abc",
            error=None,
            session=None,
        )

    monkeypatch.setattr(
        "apeGmsh.studio._replay.ReplayRunner.run_until", fake_run_until
    )
    assert main([str(script), "--no-viewer"]) == 2
