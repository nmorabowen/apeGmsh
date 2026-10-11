"""X2-c (#1664): ``Results.viewer()`` in a notebook kernel, both doors.

Auto mode (``blocking=None``) inside a Jupyter / IPython ZMQ kernel opens
the viewer in a separate process: ``python -m apeGmsh.viewers <path>``.

* A **file-backed** Results (every public ``from_*`` constructor) spawns
  that argv. Pinned here for ``from_native`` and ``from_mpco``, and run
  for real (``subprocess`` marker) with ``APEGMSH_SKIP_VIEWER=1``.
* A **hand-built, in-memory** Results (``_path is None``) has no file for
  the child to open. It used to fall back to ``show_web()`` (the legacy
  trame path X3 deletes); since #1664 it raises ``RuntimeError`` naming
  the ``from_*`` constructors, and spawns nothing. No temp ``.h5``: the
  reader protocol does not carry everything a native file needs, so a
  conversion would lose data (springs, line-station local axes, group
  metadata such as the nodal-force ``frame``).
"""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from tests.conftest import _open_model_from_h5, _stub_model_h5_path

_MPCO = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "results" / "elasticFrame.mpco"
)
_SRC = Path(__file__).resolve().parents[2] / "src"
_CONSTRUCTORS = (
    "from_native", "from_mpco", "from_ladruno", "from_recorders", "from_fem",
)


# =====================================================================
# Fixtures and seams
# =====================================================================

@pytest.fixture
def native_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "notebook.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", stage_id="grav",
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_x": np.zeros((2, node_ids.size))},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def mpco_results():
    return Results.from_mpco(_MPCO, model_h5=_stub_model_h5_path())


@pytest.fixture
def notebook(monkeypatch) -> dict:
    """Pretend to be a ZMQ kernel and catch every way out of ``viewer()``.

    Records the argv handed to ``subprocess.Popen`` and any call to
    ``show_web``; neither launches anything.
    """
    module = sys.modules[Results.__module__]
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    monkeypatch.setattr(module, "_in_notebook_kernel", lambda: True)
    monkeypatch.setattr(
        module, "_start_subprocess_monitor", lambda handle, args: None,
    )
    seen: dict = {"popen": [], "show_web": 0}

    class _FakePopen:
        def __init__(self, args, *_, **__):
            seen["popen"].append(list(args))

    def _show_web(self, **_kw):
        seen["show_web"] += 1
        return object()

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(Results, "show_web", _show_web)
    seen["cls"] = _FakePopen
    seen["undo"] = monkeypatch.undo
    return seen


# =====================================================================
# In-memory: fail closed
# =====================================================================

def test_in_memory_results_raises_and_spawns_nothing(native_results, notebook):
    """Revert-proof: before #1664 this returned ``show_web()``'s handle."""
    native_results._path = None

    with pytest.raises(RuntimeError) as exc:
        native_results.viewer()

    msg = str(exc.value)
    assert "in-memory" in msg
    for name in _CONSTRUCTORS:
        assert f"Results.{name}(" in msg, name
    assert notebook["show_web"] == 0
    assert notebook["popen"] == []


def test_viewer_has_no_path_to_the_web_viewer():
    """Reachability: ``Results.viewer`` names neither ``show_web`` nor
    ``web_viewer`` (ADR 0112 D8; X3 deletes that path)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Results.viewer)))
    names = {
        n.attr if isinstance(n, ast.Attribute) else n.id
        for n in ast.walk(tree)
        if isinstance(n, (ast.Attribute, ast.Name))
    }
    imported = {
        (n.module or "")
        for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    }
    assert not {"show_web", "serve_web", "web_viewer"} & names
    assert not any("web_viewer" in m for m in imported)


def test_explicit_blocking_false_still_refuses_in_memory(
    native_results, notebook,
):
    """Unchanged: the explicit subprocess door refuses an in-memory
    Results too, and never reaches the web viewer."""
    native_results._path = None

    with pytest.raises(RuntimeError, match="In-memory Results"):
        native_results.viewer(blocking=False)
    assert notebook["show_web"] == 0
    assert notebook["popen"] == []


# =====================================================================
# File-backed: the notebook branch spawns python -m apeGmsh.viewers
# =====================================================================

def test_from_native_spawns_the_viewer_module(native_results, notebook):
    path = str(native_results._path)

    handle = native_results.viewer()

    assert isinstance(handle, notebook["cls"])
    assert notebook["show_web"] == 0
    (argv,) = notebook["popen"]
    assert argv[:4] == [sys.executable, "-m", "apeGmsh.viewers", path]


def test_from_mpco_spawns_the_viewer_module(mpco_results, notebook):
    path = str(mpco_results._path)
    model_h5 = str(mpco_results._model_path)

    handle = mpco_results.viewer()

    assert isinstance(handle, notebook["cls"])
    assert notebook["show_web"] == 0
    (argv,) = notebook["popen"]
    assert argv[:4] == [sys.executable, "-m", "apeGmsh.viewers", path]
    # The child exits(2) on an .mpco path without the sibling model.
    assert argv[argv.index("--model-h5") + 1] == model_h5


@pytest.mark.subprocess
def test_notebook_argv_really_runs_headless(native_results, notebook):
    """The argv the notebook branch builds, run for real.

    ``APEGMSH_SKIP_VIEWER=1`` makes the child open the file through the
    CLI's real path and stop at the window boundary, exit 0.
    """
    native_results.viewer()
    (argv,) = notebook["popen"]
    notebook["undo"]()          # restore the real subprocess.Popen

    env = {
        **os.environ,
        "APEGMSH_SKIP_VIEWER": "1",
        "PYTHONPATH": str(_SRC),
    }
    proc = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
