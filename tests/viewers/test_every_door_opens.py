"""The "every door opens" smoke matrix (ADR 0042 test-net keystone).

Every public entry point that OPENS / LOADS / WIRES-UP a Results or model is
driven against the canonical composed ``results.h5`` and asserted to construct
**without launching a window** — using ``APEGMSH_SKIP_VIEWER`` / ``show=False``
/ ``serve`` without ``server.start()`` / a bounded subprocess. None of these
are GPU/rendering paths; they all stop at the render boundary.

This net exists because a single artifact (the composed results.h5) flowed
through N parallel doors and tests covered some but not all — the same bug
shipped three times (PRs #440/#441/#442) plus the web build-but-no-show/start
bugs (#436/#439). Each row below reproduces one of those. A new door must be
appended to ``DOORS`` — a missing door is then a visible omission in one list,
and a layout that a door can't open fails here loudly instead of at a user.

Hard ``timeout=`` on the subprocess door is deliberate: a hung child must fail
fast, not stall CI (an in-session incident: a runner hung ~40 min).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Building the demo + rehydrating the typed model needs the apeSees bridge.
pytest.importorskip("openseespy.opensees", reason="apeSees bridge (demo + readers)")


# Each door: (composed_results_path, sibling_model_path) -> object-or-0.
# It performs the real open/wire-up but stops BEFORE the render boundary.

def _door_opensees_model_from_h5(res: str, mdl: str):
    # #441: must auto-detect the /model neutral zone, not read the root stub.
    from apeGmsh.opensees import OpenSeesModel
    return OpenSeesModel.from_h5(res)


def _door_femdata_from_h5_composed(res: str, mdl: str):
    from apeGmsh.mesh import FEMData
    return FEMData.from_h5(res, root="/model")


def _door_viewerdata_from_h5(res: str, mdl: str):
    # #442: the blocking desktop viewer's read seam — must auto-detect /model.
    from apeGmsh.viewers.data._viewer_data import ViewerData
    return ViewerData.from_h5(res)


def _door_open_results_helper(res: str, mdl: str):
    # #440: CLI dispatch + --model-h5 forwarding for native files.
    from apeGmsh.viewers.__main__ import _open_results
    return _open_results(Path(res), Path(mdl))


def _door_results_from_native(res: str, mdl: str):
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results
    model = OpenSeesModel.from_h5(mdl)
    return Results.from_native(res, model=model)


def _door_show_web_no_display(res: str, mdl: str):
    # #436: build the WebViewer + director, never reach IPython.display.
    pytest.importorskip("pyvista", reason="[viewer] extra")
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results
    from apeGmsh.viewers.web_viewer import show_web
    r = Results.from_native(res, model=OpenSeesModel.from_h5(mdl))
    return show_web(r, show=False)


def _door_serve_web_no_start(res: str, mdl: str):
    # #439: build_app, but APEGMSH_SKIP_VIEWER short-circuits server.start().
    pytest.importorskip("trame.ui.vuetify3", reason="trame-vuetify ([viewer] extra)")
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results
    from apeGmsh.viewers.web_viewer import serve_web
    r = Results.from_native(res, model=OpenSeesModel.from_h5(mdl))
    return serve_web(r)


def _door_subprocess_native(res: str, mdl: str):
    # #440 end-to-end: real argv forwarding through `python -m apeGmsh.viewers`.
    env = {
        **os.environ,
        "APEGMSH_SKIP_VIEWER": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "apeGmsh.viewers", res, "--model-h5", mdl],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.returncode


DOORS = [
    ("opensees_model.from_h5", _door_opensees_model_from_h5),
    ("femdata.from_h5(/model)", _door_femdata_from_h5_composed),
    ("viewerdata.from_h5", _door_viewerdata_from_h5),
    ("_open_results", _door_open_results_helper),
    ("results.from_native", _door_results_from_native),
    ("show_web(show=False)", _door_show_web_no_display),
    ("serve_web(skip-start)", _door_serve_web_no_start),
    ("subprocess --model-h5", _door_subprocess_native),
]


@pytest.fixture(autouse=True)
def _skip_viewer(monkeypatch):
    """In-process doors honour this and never open a Qt/trame window."""
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")


@pytest.mark.parametrize("name,door", DOORS, ids=[d[0] for d in DOORS])
def test_every_door_opens_composed(name, door, composed_results_h5, composed_model_h5):
    """Every load/open door constructs against the composed results.h5."""
    obj = door(str(composed_results_h5), str(composed_model_h5))
    # 0 (subprocess returncode) is not None; None would mean a swallowed
    # failure / a door that returned nothing where it should have built.
    assert obj is not None
