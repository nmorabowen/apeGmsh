"""ADR 0095 Amendment 4 / S6a — manual host refresh.

``refresh_host`` is the decision core: busy (INV-18), skip-hash,
failure (INV-4), and success paths, all without touching Qt. The one
``qt``-marked test at the bottom only checks that the toolbar action
and the exposed ``ModelViewer._rebuild_scene`` hook exist on a real
host window — it is skipped headless.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.studio import ReplayResult, ReplayRunner, collect_status, last_run, ledger_path, read_runs
from apeGmsh.studio._host import _format_busy_status, refresh_host

_POC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "box.py"


class _FakeRunner:
    """Stub matching ``ReplayRunner.run_until``'s call shape."""

    def __init__(self, result: ReplayResult) -> None:
        self._result = result
        self.calls: list[tuple[Path, str, str]] = []

    def run_until(self, script: Path, *, phase: str, trigger: str) -> ReplayResult:
        self.calls.append((script, phase, trigger))
        return self._result


def test_format_busy_status() -> None:
    msg = _format_busy_status("BUSY: habitat locked by pid=4321 op=run_until")
    assert msg == "Busy: run_until pid 4321"


def test_refresh_busy_leaves_scene_untouched(tmp_path: Path) -> None:
    busy = ReplayResult(
        ok=False,
        phase="model",
        geometry_hash=None,
        error="BUSY: habitat locked by pid=4321 op=run_until",
        session=None,
    )
    runner = _FakeRunner(busy)
    scene = {"rebuilt": 0}
    outcome = refresh_host(
        runner, tmp_path / "s.py", phase="model",
        on_success=lambda: scene.__setitem__("rebuilt", scene["rebuilt"] + 1),
    )
    assert outcome["status"] == "busy"
    assert outcome["message"] == "Busy: run_until pid 4321"
    assert scene["rebuilt"] == 0
    assert runner.calls == [(tmp_path / "s.py", "model", "refresh")]


def test_refresh_skip_hash_is_noop_ledger_unchanged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    seed = runner.run_until(_POC_FIXTURE, phase="model")
    try:
        assert seed.ok, seed.error
        before = len(read_runs(ledger_path(tmp_path)))

        calls: list[int] = []
        outcome = refresh_host(
            runner, _POC_FIXTURE, phase="model",
            on_success=lambda: calls.append(1),
        )
        assert outcome["status"] == "up_to_date"
        assert outcome["message"] == "Up to date"
        assert calls == []
        assert len(read_runs(ledger_path(tmp_path))) == before
    finally:
        if seed.session is not None and seed.session.is_active:
            seed.session.end()


def test_refresh_failure_keeps_scene_and_ledger_ok_false(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "s.py"
    script.write_text("pass\n", encoding="utf-8")
    runner = ReplayRunner()
    seed = runner.run_until(script, phase="model")
    assert seed.ok, seed.error

    script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    calls: list[int] = []
    outcome = refresh_host(
        runner, script, phase="model",
        on_success=lambda: calls.append(1),
    )
    assert outcome["status"] == "failed"
    assert calls == []

    rec = last_run(ledger_path(tmp_path))
    assert rec is not None
    assert rec["ok"] is False
    assert rec["trigger"] == "refresh"


def test_refresh_success_invokes_callback_and_records_trigger(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "s.py"
    script.write_text("pass\n", encoding="utf-8")
    runner = ReplayRunner()
    seed = runner.run_until(script, phase="model")
    assert seed.ok, seed.error

    script.write_text("x = 1\n", encoding="utf-8")
    calls: list[int] = []
    outcome = refresh_host(
        runner, script, phase="model",
        on_success=lambda: calls.append(1),
    )
    assert outcome["status"] == "refreshed"
    assert outcome["message"] == "Refreshed"
    assert calls == [1]

    rec = last_run(ledger_path(tmp_path))
    assert rec is not None
    assert rec["ok"] is True
    assert rec["trigger"] == "refresh"


def test_refresh_success_without_rebuild_hook_discloses(
    tmp_path: Path, monkeypatch,
) -> None:
    """A host with no scene-rebuild hook (MeshViewer) must not claim
    the view updated — the message discloses the stale scene."""
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "s.py"
    script.write_text("pass\n", encoding="utf-8")
    runner = ReplayRunner()
    seed = runner.run_until(script, phase="model")
    assert seed.ok, seed.error

    script.write_text("x = 1\n", encoding="utf-8")
    outcome = refresh_host(runner, script, phase="model", on_success=None)
    assert outcome["status"] == "refreshed"
    assert outcome["message"] == "Refreshed (reopen host to update view)"


def test_ledger_record_without_trigger_still_parses(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": 1,
        "ts": "2026-01-01T00:00:00Z",
        "script": "s.py",
        "hash": "abc",
        "phase": "model",
        "stopped_at": None,
        "ok": True,
        "session": None,
        "labels": [],
        "physical_groups": [],
        "counts": None,
        "assess": None,
        "visors": [],
        "error": None,
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    rows = read_runs(path)
    assert len(rows) == 1
    assert rows[0].get("trigger") is None

    assert last_run(path)["ok"] is True

    status = collect_status(tmp_path)
    assert status["last_run"]["ok"] is True
    assert status["ledger_error"] is None


@pytest.mark.qt
def test_refresh_action_exists_on_host(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from qtpy import QtCore

    from apeGmsh.studio._host import open_host
    from apeGmsh.viewers.model_viewer import ModelViewer as _RealModelViewer

    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    result = runner.run_until(_POC_FIXTURE, phase="model")
    assert result.ok, result.error

    captured: list = []

    class _CapturingModelViewer(_RealModelViewer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(
        "apeGmsh.viewers.model_viewer.ModelViewer", _CapturingModelViewer,
    )

    seen: dict = {}

    def _check_then_close() -> None:
        try:
            assert captured, "ModelViewer was not constructed"
            viewer = captured[0]
            win = viewer._win
            tooltips = [a.toolTip() for a in win._toolbar.actions()]
            seen["has_refresh_action"] = "Refresh (F5)" in tooltips
            seen["has_rebuild_hook"] = hasattr(viewer, "_rebuild_scene")
        finally:
            try:
                captured[0]._win.window.close()
            except Exception:
                pass

    QtCore.QTimer.singleShot(300, _check_then_close)
    try:
        open_host(
            result.session,
            envelope_path=tmp_path / ".apegmsh" / "selection.json",
            root=tmp_path,
            script=_POC_FIXTURE,
            runner=runner,
        )
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()

    assert seen.get("has_refresh_action") is True
    assert seen.get("has_rebuild_hook") is True
