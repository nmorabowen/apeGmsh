"""ADR 0095 Amendment 4 / S6b — auto-refresh file watch.

``watch_tick`` is the debounce decision core: pure, no Qt, no
filesystem. ``refresh_host`` (S6a) is reused unchanged for the actual
replay — S6b only adds ``trigger="watch"`` and the QTimer poll loop
that feeds it, so the runner-level tests here exercise
``refresh_host(..., trigger="watch")`` directly, the same way
``tests/studio/test_refresh.py`` exercises the manual path. The one
``qt``-marked test at the bottom only checks that the "Watch" toolbar
toggle exists on a real host window — it is skipped headless.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.studio import ReplayResult, ReplayRunner, last_run, ledger_path
from apeGmsh.studio._busy import release_busy, try_acquire_busy
from apeGmsh.studio._host import refresh_host
from apeGmsh.studio._watch import WatchState, seed_watch_state, watch_tick

_POC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "box.py"


# --------------------------------------------------------------------
# Debounce core (pure)
# --------------------------------------------------------------------

def test_seed_watch_state_captures_initial_mtime() -> None:
    state = seed_watch_state(100.0)
    assert state == WatchState(last_seen=100.0, pending=None)


def test_no_change_produces_no_trigger() -> None:
    state = seed_watch_state(100.0)
    for _ in range(5):
        state, settled = watch_tick(state, 100.0)
        assert settled is False
    assert state.pending is None


def test_single_clean_change_produces_one_trigger() -> None:
    state = seed_watch_state(100.0)

    state, settled = watch_tick(state, 101.0)  # change observed
    assert settled is False

    state, settled = watch_tick(state, 101.0)  # same mtime -> settled
    assert settled is True
    assert state.last_seen == 101.0
    assert state.pending is None

    # Further ticks at the same mtime do not fire again.
    state, settled = watch_tick(state, 101.0)
    assert settled is False


def test_flapping_mtime_produces_exactly_one_trigger() -> None:
    state = seed_watch_state(100.0)
    sequence = [101.0, 102.0, 103.0, 103.0, 103.0]
    fires = []
    for mtime in sequence:
        state, settled = watch_tick(state, mtime)
        fires.append(settled)
    # settled only once, on the first tick where the mtime repeats.
    assert fires == [False, False, False, True, False]
    assert state.last_seen == 103.0


def test_missing_file_does_not_settle_and_clears_pending() -> None:
    state = seed_watch_state(100.0)
    state, settled = watch_tick(state, 101.0)
    assert settled is False
    assert state.pending == 101.0

    state, settled = watch_tick(state, None)  # file briefly missing
    assert settled is False
    assert state.pending is None
    assert state.last_seen == 100.0


# --------------------------------------------------------------------
# refresh_host(trigger="watch") — the S6a decision core, reused
# --------------------------------------------------------------------

def test_refresh_host_watch_trigger_records_ledger(
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
        trigger="watch",
    )
    assert outcome["status"] == "refreshed"
    assert calls == [1]

    rec = last_run(ledger_path(tmp_path))
    assert rec is not None
    assert rec["ok"] is True
    assert rec["trigger"] == "watch"


def test_refresh_host_watch_busy_leaves_scene_untouched(tmp_path: Path) -> None:
    """A watch tick that lands while a peer holds busy.json reports
    busy, does not steal, and never calls the scene-rebuild hook
    (INV-18) — same contract as manual refresh, now under trigger="watch"."""
    script = tmp_path / "s.py"
    script.write_text("pass\n", encoding="utf-8")
    assert try_acquire_busy(tmp_path, op="run_until", phase="model") is True
    try:
        runner = ReplayRunner(root=tmp_path)
        calls: list[int] = []
        outcome = refresh_host(
            runner, script, phase="model",
            on_success=lambda: calls.append(1),
            trigger="watch",
        )
        assert outcome["status"] == "busy"
        assert calls == []
    finally:
        release_busy(tmp_path)


def test_refresh_host_rejects_unknown_trigger(tmp_path: Path) -> None:
    class _FakeRunner:
        def run_until(self, script: Path, *, phase: str, trigger: str) -> ReplayResult:
            raise AssertionError("run_until must not be called for a bad trigger")

    with pytest.raises(ValueError):
        refresh_host(_FakeRunner(), tmp_path / "s.py", phase="model", trigger="bogus")


def test_refresh_host_default_trigger_is_still_refresh(tmp_path: Path) -> None:
    """S6a regression guard: omitting trigger= still records "refresh"."""

    class _FakeRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run_until(self, script: Path, *, phase: str, trigger: str) -> ReplayResult:
            self.calls.append(trigger)
            return ReplayResult(
                ok=True, phase=phase, geometry_hash="h", error=None,
                session=None, skipped=True,
            )

    runner = _FakeRunner()
    outcome = refresh_host(runner, tmp_path / "s.py", phase="model")
    assert outcome["status"] == "up_to_date"
    assert runner.calls == ["refresh"]


# --------------------------------------------------------------------
# Qt smoke — Watch toggle exists (skipped headless)
# --------------------------------------------------------------------

@pytest.mark.qt
def test_watch_toggle_exists_on_host(tmp_path: Path, monkeypatch) -> None:
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
            actions = {a.toolTip(): a for a in win._toolbar.actions()}
            seen["has_watch_action"] = "Watch (auto-refresh on save)" in actions
            watch_action = actions.get("Watch (auto-refresh on save)")
            seen["watch_checkable"] = (
                watch_action.isCheckable() if watch_action is not None else False
            )
            seen["watch_checked_by_default"] = (
                watch_action.isChecked() if watch_action is not None else False
            )
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

    assert seen.get("has_watch_action") is True
    assert seen.get("watch_checkable") is True
    assert seen.get("watch_checked_by_default") is True
