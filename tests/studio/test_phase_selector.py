"""ADR 0095 Amendment 4 / S6c — phase selector + INV-8 gates.

``allowed_phases`` is the gate decision core: pure, no Qt, no
filesystem (mirrors ``watch_tick`` in ``test_watch.py``).
``_apply_phase_outcome`` is the runner-level glue that recomputes the
gate after a ``refresh_host`` outcome and writes ``host.json.phase``
through ``_host_state`` — exercised here against a tmp habitat, same
way ``test_refresh.py`` exercises ``refresh_host`` directly. The one
``qt``-marked test at the bottom only checks that the Model / Mesh /
Results toolbar actions exist (and are gated) on a real host window;
it is skipped headless.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.studio import ReplayResult, ReplayRunner, allowed_phases
from apeGmsh.studio._host import (
    _PhaseState,
    _apply_phase_outcome,
    _sync_phase_buttons,
    refresh_host,
)
from apeGmsh.studio._host_state import claim_host, read_host

_POC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "box.py"


def _result(*, phase: str, stopped_at: str | None, ok: bool = True) -> ReplayResult:
    return ReplayResult(
        ok=ok, phase=phase, geometry_hash="h", error=None, stopped_at=stopped_at,
    )


# --------------------------------------------------------------------
# allowed_phases — pure INV-8 gate
# --------------------------------------------------------------------

def test_stopped_at_model_only_model_allowed() -> None:
    result = _result(phase="model", stopped_at="g.mesh.generation.generate")
    assert allowed_phases(result, has_mesh=False) == ("model",)


def test_stopped_at_mesh_or_has_mesh_allows_model_and_mesh() -> None:
    result = _result(phase="mesh", stopped_at="apeSees")
    assert allowed_phases(result, has_mesh=True) == ("model", "mesh")


def test_completed_results_allows_all_three() -> None:
    result = _result(phase="results", stopped_at=None)
    assert allowed_phases(result, has_mesh=True) == ("model", "mesh", "results")


def test_retreat_after_results_then_model_stopped() -> None:
    """INV-8 'must retreat': a later model-stopped replay disables
    mesh / results again after a results-complete replay allowed all
    three."""
    results_done = _result(phase="results", stopped_at=None)
    assert allowed_phases(results_done, has_mesh=True) == ("model", "mesh", "results")

    model_stopped = _result(phase="model", stopped_at="g.mesh.generation.generate")
    assert allowed_phases(model_stopped, has_mesh=False) == ("model",)


def test_failed_result_retreats_to_model_even_with_stale_mesh_evidence() -> None:
    failed = _result(phase="mesh", stopped_at=None, ok=False)
    assert allowed_phases(failed, has_mesh=True) == ("model",)


def test_no_result_yet_falls_back_to_has_mesh() -> None:
    """At host-open time there is no replay result from this runner
    yet (it ran through a different runner upstream); has_mesh alone
    still gates mesh."""
    assert allowed_phases(None, has_mesh=True) == ("model", "mesh")
    assert allowed_phases(None, has_mesh=False) == ("model",)


# --------------------------------------------------------------------
# _sync_phase_buttons — pure Qt-adjacent glue (fake actions, no Qt)
# --------------------------------------------------------------------

class _FakeAction:
    def __init__(self) -> None:
        self.enabled: bool | None = None
        self.checked: bool | None = None

    def setEnabled(self, value: bool) -> None:
        self.enabled = value

    def setChecked(self, value: bool) -> None:
        self.checked = value


def test_sync_phase_buttons_reflects_current_and_allowed() -> None:
    buttons = {"model": _FakeAction(), "mesh": _FakeAction(), "results": _FakeAction()}
    _sync_phase_buttons(buttons, "mesh", ("model", "mesh"))
    assert (buttons["model"].enabled, buttons["model"].checked) == (True, False)
    assert (buttons["mesh"].enabled, buttons["mesh"].checked) == (True, True)
    assert (buttons["results"].enabled, buttons["results"].checked) == (False, False)


# --------------------------------------------------------------------
# _apply_phase_outcome — runner-level, against a tmp habitat
# --------------------------------------------------------------------

def test_phase_switch_replays_and_updates_host_json(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner(root=tmp_path)
    seed = runner.run_until(_POC_FIXTURE, phase="model")
    try:
        assert seed.ok, seed.error
        claim_host(tmp_path, phase="model")
        phase_state = _PhaseState("model")

        outcome = refresh_host(runner, _POC_FIXTURE, phase="mesh", trigger="refresh")
        assert outcome["status"] == "refreshed"
        _apply_phase_outcome(
            outcome, phase_state=phase_state, habitat=tmp_path, requested="mesh",
        )

        assert phase_state.current == "mesh"
        assert phase_state.allowed == ("model", "mesh")
        assert read_host(tmp_path)["phase"] == "mesh"
    finally:
        # The phase="mesh" replay above opened a *second* session
        # (run_until re-execs the script); close both so the live
        # Gmsh kernel does not leak mesh evidence into later tests.
        for sess in (seed.session, outcome["result"].session):
            if sess is not None and sess.is_active:
                sess.end()


def test_requested_phase_not_actually_reached_retreats_anyway(
    tmp_path: Path, monkeypatch,
) -> None:
    """A selector click requesting 'mesh' still retreats to 'model' if
    the replay never actually produced a mesh (INV-8 wins over the
    requested target).

    Opens a real (fresh-model) apeGmsh session with geometry but no
    ``generate()`` call — a bare ``pass`` script would leave the live
    Gmsh kernel's *previously active* model in place, and
    ``session_has_mesh()`` reads whatever model is current, which
    would leak mesh evidence from an earlier test in this same
    process.
    """
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "s.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='s6c_retreat_probe') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n",
        encoding="utf-8",
    )
    runner = ReplayRunner(root=tmp_path)
    seed = runner.run_until(script, phase="model")
    try:
        assert seed.ok, seed.error
        claim_host(tmp_path, phase="model")
        phase_state = _PhaseState("model")

        outcome = refresh_host(runner, script, phase="mesh", trigger="refresh")
        assert outcome["status"] == "refreshed"
        _apply_phase_outcome(
            outcome, phase_state=phase_state, habitat=tmp_path, requested="mesh",
        )

        assert phase_state.current == "model"
        assert phase_state.allowed == ("model",)
        assert read_host(tmp_path)["phase"] == "model"
    finally:
        if outcome["result"].session is not None and outcome["result"].session.is_active:
            outcome["result"].session.end()


def test_busy_outcome_leaves_phase_and_host_json_unchanged(tmp_path: Path) -> None:
    claim_host(tmp_path, phase="model")
    phase_state = _PhaseState("model")
    phase_state.allowed = ("model",)

    busy = ReplayResult(
        ok=False, phase="mesh", geometry_hash=None,
        error="BUSY: habitat locked by pid=1 op=run_until",
    )
    outcome = {"status": "busy", "message": "Busy: run_until pid 1", "result": busy}
    _apply_phase_outcome(outcome, phase_state=phase_state, habitat=tmp_path)

    assert phase_state.current == "model"
    assert phase_state.allowed == ("model",)
    assert read_host(tmp_path)["phase"] == "model"


def test_failed_outcome_leaves_phase_and_host_json_unchanged(tmp_path: Path) -> None:
    claim_host(tmp_path, phase="model")
    phase_state = _PhaseState("model")
    phase_state.allowed = ("model",)

    failed = ReplayResult(ok=False, phase="mesh", geometry_hash="h", error="boom")
    outcome = {"status": "failed", "message": "Refresh failed: boom", "result": failed}
    _apply_phase_outcome(outcome, phase_state=phase_state, habitat=tmp_path)

    assert phase_state.current == "model"
    assert phase_state.allowed == ("model",)
    assert read_host(tmp_path)["phase"] == "model"


def test_apply_phase_outcome_calls_sync_hook_on_every_outcome(tmp_path: Path) -> None:
    """The (optional) button-sync hook fires for busy / failed / refreshed
    outcomes alike — reverted clicks still need the visual snap-back."""
    claim_host(tmp_path, phase="model")
    calls: list[int] = []
    phase_state = _PhaseState("model")
    phase_state.sync = lambda: calls.append(1)

    busy = ReplayResult(ok=False, phase="mesh", geometry_hash=None, error="BUSY: x")
    _apply_phase_outcome(
        {"status": "busy", "message": "Busy", "result": busy},
        phase_state=phase_state, habitat=tmp_path,
    )
    assert calls == [1]


# --------------------------------------------------------------------
# Qt smoke — Model / Mesh / Results actions exist and are gated
# (skipped headless)
# --------------------------------------------------------------------

@pytest.mark.qt
def test_phase_selector_exists_on_host(tmp_path: Path, monkeypatch) -> None:
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
            model_action = actions.get("Model phase")
            mesh_action = actions.get("Mesh phase")
            results_action = actions.get("Results phase")
            seen["has_all_three"] = all(
                a is not None for a in (model_action, mesh_action, results_action)
            )
            seen["model_checked"] = (
                model_action.isChecked() if model_action is not None else False
            )
            seen["model_enabled"] = (
                model_action.isEnabled() if model_action is not None else False
            )
            seen["mesh_enabled"] = (
                mesh_action.isEnabled() if mesh_action is not None else True
            )
            seen["results_enabled"] = (
                results_action.isEnabled() if results_action is not None else True
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

    assert seen.get("has_all_three") is True
    assert seen.get("model_checked") is True
    assert seen.get("model_enabled") is True
    # box.py at phase="model" never reaches generate() — INV-8: no
    # mesh, so mesh/results stay disabled.
    assert seen.get("mesh_enabled") is False
    assert seen.get("results_enabled") is False
