"""ResultsViewer + fem_scene — Phase 0 smoke tests.

Verifies the construction-time path without spinning up a Qt event
loop: scene builder produces a valid grid, ResultsViewer accepts a
bound Results, refusing the unbound case. The full Qt-event-loop
smoke test is gated on Qt + pytest-qt availability and only opens the
window briefly via ``QTimer.singleShot`` to close it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.results_viewer import ResultsViewer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def small_results(g, tmp_path: Path):
    """Tiny native Results with one stage + 3 steps, single-volume mesh."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "small.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": np.zeros((3, n_nodes)),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


# =====================================================================
# fem_scene
# =====================================================================

def test_build_fem_scene_returns_pyvista_grid(small_results):
    scene = build_fem_scene(small_results.fem)
    assert isinstance(scene.grid, pv.UnstructuredGrid)
    assert scene.grid.n_points == small_results.fem.nodes.ids.size
    assert scene.grid.n_cells > 0
    assert scene.model_diagonal > 0.0


def test_fem_scene_has_element_id_cell_data(small_results):
    scene = build_fem_scene(small_results.fem)
    assert "element_id" in scene.grid.cell_data
    assert scene.grid.cell_data["element_id"].size == scene.grid.n_cells


def test_fem_scene_has_node_id_point_data(small_results):
    scene = build_fem_scene(small_results.fem)
    assert "node_id" in scene.grid.point_data
    np.testing.assert_array_equal(
        scene.grid.point_data["node_id"],
        np.asarray(list(small_results.fem.nodes.ids), dtype=np.int64),
    )


def test_fem_scene_id_lookups_are_consistent(small_results):
    scene = build_fem_scene(small_results.fem)
    # Every entry in cell_to_element_id should round-trip
    for cell_idx, eid in enumerate(scene.cell_to_element_id):
        assert scene.element_id_to_cell[int(eid)] == cell_idx
    # Same for nodes
    for nid, idx in scene.node_id_to_idx.items():
        assert scene.node_ids[idx] == nid


def test_fem_scene_node_tree_built_lazily(small_results):
    scene = build_fem_scene(small_results.fem)
    assert scene.node_tree is None
    tree = scene.ensure_node_tree()
    if tree is not None:    # scipy installed (it is, per project deps)
        assert scene.node_tree is tree


# =====================================================================
# ResultsViewer construction
# =====================================================================

def test_results_viewer_construction_requires_bound_fem(small_results):
    # Strip the fem to force the unbound state
    small_results._fem = None
    with pytest.raises(RuntimeError, match="bound FEMData"):
        ResultsViewer(small_results)


def test_results_viewer_default_title_uses_filename(small_results):
    viewer = ResultsViewer(small_results)
    title = viewer._default_title()
    assert title.startswith("Results — ")
    assert title.endswith(".h5")


def test_results_viewer_explicit_title(small_results):
    viewer = ResultsViewer(small_results, title="Custom")
    assert viewer._title == "Custom"


# =====================================================================
# Results.viewer() entry point — subprocess opt-in (Phase 6)
# =====================================================================

def test_subprocess_launches_with_correct_args(small_results, monkeypatch):
    """``blocking=False`` invokes ``python -m apeGmsh.viewers <path>``."""
    import subprocess
    import sys

    captured: dict = {}

    class _FakePopen:
        def __init__(self, args, *_, **__):
            captured["args"] = args

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    handle = small_results.viewer(blocking=False)
    assert isinstance(handle, _FakePopen)
    assert captured["args"][:3] == [sys.executable, "-m", "apeGmsh.viewers"]
    assert captured["args"][3] == str(small_results._path)


def test_subprocess_passes_title(small_results, monkeypatch):
    import subprocess
    captured: dict = {}

    class _FakePopen:
        def __init__(self, args, *_, **__):
            captured["args"] = args

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    small_results.viewer(blocking=False, title="My Run")
    assert "--title" in captured["args"]
    idx = captured["args"].index("--title")
    assert captured["args"][idx + 1] == "My Run"


def test_subprocess_in_memory_raises_clearly(small_results):
    """No path → cannot subprocess; raises with a self-explanatory message."""
    small_results._path = None
    with pytest.raises(RuntimeError, match="In-memory Results"):
        small_results.viewer(blocking=False)


# =====================================================================
# blocking=None — the notebook-safe auto default
# =====================================================================

def _results_module():
    """The ``apeGmsh.results.Results`` *module* (the class shadows it
    as a package attribute, so plain dotted access is ambiguous)."""
    import sys
    return sys.modules[Results.__module__]


def test_notebook_auto_default_spawns_subprocess(small_results, monkeypatch):
    """``blocking=None`` inside a notebook kernel takes the subprocess
    path — the blocking Qt loop would freeze the kernel."""
    import subprocess

    captured: dict = {}

    class _FakePopen:
        def __init__(self, args, *_, **__):
            captured["args"] = args

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        _results_module(), "_in_notebook_kernel", lambda: True,
    )

    handle = small_results.viewer()
    assert isinstance(handle, _FakePopen)
    assert captured["args"][1:3] == ["-m", "apeGmsh.viewers"]


def test_notebook_auto_default_in_memory_falls_back_to_web(
    small_results, monkeypatch,
):
    """In-memory Results cannot subprocess, so the notebook auto
    default opens the web viewer instead of raising (or blocking)."""
    monkeypatch.setattr(
        _results_module(), "_in_notebook_kernel", lambda: True,
    )
    sentinel = object()
    monkeypatch.setattr(
        Results, "show_web", lambda self, **kw: sentinel,
    )
    small_results._path = None
    assert small_results.viewer() is sentinel


def test_script_auto_default_blocks(small_results, monkeypatch):
    """Outside a notebook, ``blocking=None`` keeps the in-process Qt
    path — script / CLI ergonomics unchanged."""
    from apeGmsh.viewers import results_viewer as viewer_mod

    monkeypatch.setattr(
        _results_module(), "_in_notebook_kernel", lambda: False,
    )

    captured: dict = {}

    class _FakeViewer:
        def __init__(self, results, **kwargs):
            captured["results"] = results

        def show(self):
            return self

    monkeypatch.setattr(viewer_mod, "ResultsViewer", _FakeViewer)
    out = small_results.viewer()
    assert isinstance(out, _FakeViewer)
    assert captured["results"] is small_results


# =====================================================================
# Optional: Qt event-loop smoke test
# =====================================================================
#
# Only runs when:
#   * a Qt binding is importable (qtpy resolves)
#   * pyvistaqt is available
#   * pytest-qt is available (provides ``qapp`` fixture)
#
# The test opens the window, then schedules its closure via
# ``QTimer.singleShot`` to keep CI-time bounded. We do not assert
# anything visual — just that the full lifecycle exits cleanly.

def test_show_traps_init_failure_into_session_log(
    small_results, monkeypatch, tmp_path,
):
    """A crash inside ``_show_impl`` must land in the session log
    file with full traceback before propagating, so silent VTK / Qt
    init failures stop disappearing into a closed terminal.
    """
    from apeGmsh.viewers import _log

    # Force a known exception in _show_impl by stubbing one of the
    # early imports. ResultsDirector is constructed right after the
    # imports, so swapping it for a raising fake hits the trap.
    class _Boom(Exception):
        pass

    def _raise(*_a, **_kw):
        raise _Boom("synthetic init failure")

    monkeypatch.setattr(
        "apeGmsh.viewers.diagrams._director.ResultsDirector",
        _raise,
    )

    viewer = ResultsViewer(small_results)
    with pytest.raises(_Boom):
        viewer.show()

    # The session log must exist and contain an init-error entry —
    # this is the whole point of the trap.
    log_path = _log.session_file()
    assert log_path is not None
    text = log_path.read_text(encoding="utf-8")
    assert "init.ResultsViewer.show" in text
    assert "_Boom" in text


@pytest.mark.qt
def test_results_viewer_show_close_lifecycle(small_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    qapp = pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    viewer = ResultsViewer(small_results, title="smoke")

    # Schedule a close via the underlying window. The viewer's
    # ``_on_close`` handler runs and unbinds the director; the
    # ``ViewerWindow.exec`` returns and ``show`` returns ``self``.
    seen: dict = {}

    def _close_after_construction():
        try:
            # Snapshot while the window lives, so the teardown asserts
            # below cannot pass vacuously (install-failed → None).
            seen["scope"] = viewer._scope_gizmo_interactor
            viewer._win.window.close()
        except Exception:
            pass

    QtCore.QTimer.singleShot(250, _close_after_construction)

    result = viewer.show()
    assert result is viewer
    # Director unbound on close
    assert viewer.director is not None
    assert not viewer.director.registry.is_bound
    # Gizmo interactors torn down on close: observers removed, cursor
    # memo withdrawn, refs dropped so a reopen starts clean.
    assert seen["scope"] is not None, (
        "the scope gizmo interactor never installed — the teardown "
        "asserts below would be vacuous"
    )
    assert viewer._clip_gizmo_interactor is None
    assert viewer._scope_gizmo_interactor is None
    assert seen["scope"]._iren is None, "uninstall() did not run"


@pytest.mark.qt
def test_color_map_editor_lives_in_inspector(small_results):
    """ADR 0088 D2 — the color editor is no longer an extension dock:
    it mounts inside the Inspector's diagram context, and the retired
    ``dock_color_map_editor`` objectName is not registered."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    qapp = pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    viewer = ResultsViewer(small_results, title="color-editor-smoke")
    seen: dict = {}

    def _check_then_close():
        try:
            seen["editor"] = viewer._color_editor
            seen["inspector"] = viewer._inspector
            win = viewer._win
            seen["in_inspector_dock"] = (
                win._dock_inspector is not None
                and win._dock_inspector.isAncestorOf(
                    viewer._color_editor.widget,
                )
            )
            try:
                win.extension_dock("dock_color_map_editor")
                seen["retired_dock_registered"] = True
            except KeyError:
                seen["retired_dock_registered"] = False
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(250, _check_then_close)
    viewer.show()

    assert seen.get("editor") is not None
    assert seen.get("inspector") is not None
    assert seen.get("in_inspector_dock") is True
    assert seen.get("retired_dock_registered") is False


@pytest.mark.qt
def test_director_step_drives_active_step(small_results):
    """Plan 04 step 2 cont. — director.set_step routes through the
    director's step subscribers and reaches ``ActiveObjects`` via the
    bridge wired in ResultsViewer._show_impl."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    viewer = ResultsViewer(small_results, title="step-bridge-smoke")
    seen: dict = {"steps": []}

    def _drive_then_close():
        try:
            viewer._active.activeStepChanged.connect(
                lambda s: seen["steps"].append(int(s)),
            )
            # Director.set_step is the canonical entry point for time
            # navigation — invoking it must reach ActiveObjects.
            viewer._director.set_step(1)
            viewer._director.set_step(2)
            # set_step(2) again is a no-op (value unchanged).
            viewer._director.set_step(2)
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(250, _drive_then_close)
    viewer.show()

    # Initial step (0) may have emitted depending on bind timing; just
    # require the two explicit set_step calls to have produced one
    # emission each. Final state is step 2.
    assert 1 in seen["steps"]
    assert 2 in seen["steps"]
    assert viewer._active.active_step == 2


@pytest.mark.qt
def test_director_stage_drives_active_stage(small_results):
    """Plan 04 step 2 cont. — director.set_stage routes through
    ActiveObjects.activeStageChanged."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    viewer = ResultsViewer(small_results, title="stage-bridge-smoke")
    seen: dict = {"stages": []}

    def _drive_then_close():
        try:
            viewer._active.activeStageChanged.connect(
                lambda s: seen["stages"].append(s),
            )
            # small_results fixture has one stage ("grav"). Re-setting
            # the same stage is a no-op via identity check, but
            # set_stage to its own current value still goes through.
            # (Results.stage_ids() bit-rotted while this test was
            # pytest-qt-skipped; the Results.stages property is the
            # live API.)
            stage_ids = [s.id for s in viewer._results.stages]
            assert stage_ids, "fixture must have at least one stage"
            viewer._director.set_stage(stage_ids[0])
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(250, _drive_then_close)
    viewer.show()

    # The bridge fires when set_stage actually changes the active stage
    # (or when the initial bind sets it). Either way, ActiveObjects
    # should hold the fixture's stage id at this point.
    assert viewer._active.active_stage is not None
