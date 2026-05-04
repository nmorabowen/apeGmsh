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
    return Results.from_native(path)


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
    def _close_after_construction():
        try:
            viewer._win.window.close()
        except Exception:
            pass

    QtCore.QTimer.singleShot(250, _close_after_construction)

    result = viewer.show()
    assert result is viewer
    # Director unbound on close
    assert viewer.director is not None
    assert not viewer.director.registry.is_bound
