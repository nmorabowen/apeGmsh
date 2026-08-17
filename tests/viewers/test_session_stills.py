"""ADR 0098 S1-A — ``session.render``: offscreen stills of a pane.

Secondary oracle (pixels; skip ≠ pass, ADR 0094): the layer laws are
pinned by ``test_session_realize.py`` on a RecordingBackend; here the
still client itself is exercised — the GL-skip discipline verbatim
from the relocated ``render.py`` path, the file contract, and the
decision-5 parity still against ``render_results(view='contour')``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, ResultsSession
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5

STAGE = "grav"


@pytest.fixture
def still_results(g, tmp_path: Path):
    """One static stage with smooth nodal data (``displacement_z``)."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    n_steps = 3
    disp = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    for t in range(n_steps):
        disp[t] = (t + 1) * coords[:, 2]

    path = tmp_path / "still.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


# =====================================================================
# Skip discipline (no GL needed — runs everywhere)
# =====================================================================


def test_skip_env_writes_nothing(
    still_results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    session = still_results.session()
    out = tmp_path / "stills" / "pane.png"
    assert session.render(out) is None
    assert not out.exists()
    assert not out.parent.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_render_without_results_refuses(tmp_path: Path) -> None:
    session = ResultsSession()
    session.add_view()
    with pytest.raises(RuntimeError, match="no Results bound"):
        session.render(tmp_path / "x.png")


def test_render_two_panes_needs_id(
    still_results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    session = still_results.session()
    session.add_view()
    with pytest.raises(ValueError, match="address one by id"):
        session.render(tmp_path / "x.png")


# =====================================================================
# Pixel stills (GL; skip ≠ pass)
# =====================================================================


def _render_or_skip(session, path, **kwargs):
    out = session.render(path, **kwargs)
    if out is None:
        pytest.skip("no GL context for offscreen stills")
    return out


def test_empty_view_still_writes_png(still_results, tmp_path: Path) -> None:
    session = still_results.session()
    out = _render_or_skip(session, tmp_path / "empty.png")
    assert out.exists() and out.stat().st_size > 0
    # No partial file left behind.
    assert list(tmp_path.glob("*.partial*")) == []


def test_contour_still_writes_png(still_results, tmp_path: Path) -> None:
    session = still_results.session()
    view = session.panes[0]
    view.contour = Contour("displacement_z")
    out = _render_or_skip(session, tmp_path / "contour.png")
    assert out.exists() and out.stat().st_size > 0


def test_empty_view_still_parity_with_render_results_mesh(
    still_results, tmp_path: Path, frames_match_or_skip,
) -> None:
    """The scene-IR substrate emission reproduces the shipped grey
    still (``render_results(view='mesh')``, raw ``add_mesh``)."""
    old = still_results.render(tmp_path / "old.png", view="mesh")
    if old is None:
        pytest.skip("no GL context for offscreen stills")
    session = still_results.session()
    new = _render_or_skip(session, tmp_path / "new.png")

    import matplotlib.image as mpimg

    old_px = (np.asarray(mpimg.imread(old)) * 255).astype(np.uint8)
    new_px = (np.asarray(mpimg.imread(new)) * 255).astype(np.uint8)
    frames_match_or_skip(
        new_px, old_px, what="session substrate vs render_results mesh",
    )


def test_contour_still_parity_with_render_results(
    still_results, tmp_path: Path, frames_match_or_skip,
) -> None:
    """Decision-5 pin at the pixel level: the session's one-shot
    realize (pose before extraction, session-authored legend) and
    ``render_results(view='contour')`` (attach then sync, diagram-
    registered legend) must paint the same picture."""
    old = still_results.render(
        tmp_path / "old.png", view="contour", component="displacement_z",
    )
    if old is None:
        pytest.skip("no GL context for offscreen stills")

    session = still_results.session()
    session.panes[0].contour = Contour("displacement_z")
    new = _render_or_skip(session, tmp_path / "new.png")

    import matplotlib.image as mpimg

    old_px = (np.asarray(mpimg.imread(old)) * 255).astype(np.uint8)
    new_px = (np.asarray(mpimg.imread(new)) * 255).astype(np.uint8)
    frames_match_or_skip(new_px, old_px, what="session vs render_results")
