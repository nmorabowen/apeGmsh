"""fem_scene + the ``Results.viewer()`` door — Phase 0 smoke tests.

Verifies the construction-time path without spinning up a Qt event
loop: the scene builder produces a valid grid, and the viewer door
dispatches to a subprocess or the session window per mode. The legacy
``ResultsViewer`` construction and real-window tests went with that
viewer (X3-d, #1734).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
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


def test_notebook_auto_default_in_memory_raises(
    small_results, monkeypatch,
):
    """In-memory Results cannot subprocess, so the notebook auto
    default raises a clear error instead of blocking or opening the
    legacy web viewer (#1664)."""
    monkeypatch.setattr(
        _results_module(), "_in_notebook_kernel", lambda: True,
    )
    calls: list = []
    monkeypatch.setattr(
        Results, "show_web", lambda self, **kw: calls.append(kw),
    )
    small_results._path = None
    with pytest.raises(RuntimeError, match="in-memory"):
        small_results.viewer()
    assert calls == []


def test_script_auto_default_blocks(small_results, monkeypatch):
    """Outside a notebook, ``blocking=None`` keeps the in-process Qt
    path — script / CLI ergonomics unchanged.

    Flipped at ADR 0098 S6a: the in-process path is now
    ``session().show()``, so this stubs ``ResultsSession.show`` (the one
    seam ``Results._show_session_window`` uses) instead of the retired
    ``ResultsViewer``. What is asserted is unchanged — auto-detect
    outside a kernel means in-process, not a subprocess. The flip's own
    coverage is ``test_viewer_flip.py``.
    """
    from apeGmsh.results.session import ResultsSession

    monkeypatch.setattr(
        _results_module(), "_in_notebook_kernel", lambda: False,
    )

    captured: dict = {}

    def _show(self, *, blocking=True, title=None):
        captured["session"] = self
        captured["blocking"] = blocking
        return None

    monkeypatch.setattr(ResultsSession, "show", _show)
    out = small_results.viewer(restore_session=False, save_session=False)

    assert isinstance(out, ResultsSession)
    assert captured["session"] is out
    assert captured["blocking"] is True
    assert out.results is small_results
