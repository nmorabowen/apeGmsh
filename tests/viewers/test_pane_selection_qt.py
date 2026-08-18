"""ADR 0098 §8 (S4-1) — pane pick on real GL (``[qt]`` lane).

``test_pane_selection.py`` proves the interpretation below the pick
seam with a fake ``PickBackend``. This file proves the half a fake
cannot: that the real
:class:`~apeGmsh.viewers.backends._pyvista_pick.PyVistaPickBackend` is
installed per pane, that the glyph cloud is pickable **at the actor**
(the scene-IR flag is only a promise until VTK carries it, and
``vtkCellPicker`` skips a non-pickable prop outright), that a real
interactor event resolves to a real node, and that closing the window
takes the observers with it.

Run per-file in a fresh process, under xvfb in CI::

    xvfb-run -a pytest tests/viewers/test_pane_selection_qt.py -m qt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.qt

PANES = 2


@pytest.fixture
def qt_results(g, tmp_path: Path):
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left")
    g.model.geometry.add_box(1, 0, 0, 1, 1, 1, label="right")
    g.physical.add_volume("left", name="Left")
    g.physical.add_volume("right", name="Right")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.zeros((2, node_ids.size))
    disp[1] = np.asarray(fem.nodes.coords, dtype=np.float64)[:, 2] * 10.0

    path = tmp_path / "pane_selection_qt.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", stage_id="grav",
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def live_window(qt_results, tmp_path, monkeypatch):
    """A real :class:`SessionWindow`, node glyphs on in every pane."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from qtpy.QtCore import QSettings

    from apeGmsh.results.session import MeshStyle
    from apeGmsh.viewers.session import SessionResultsWindow, SessionWindow

    ini = str(tmp_path / "ResultsSession.ini")
    monkeypatch.setattr(
        SessionResultsWindow, "_layout_settings",
        staticmethod(lambda: QSettings(ini, QSettings.Format.IniFormat)),
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = qt_results.session()
    window = SessionWindow(session, title="s4-qt")
    window.shell.window.resize(1200, 800)
    window.show(blocking=False)
    app.processEvents()
    for _ in range(PANES - 1):
        session.add_view()
    for pane in session.panes:
        pane.style = MeshStyle(
            mesh=True, outlines=True, nodes=True, gauss=False,
        )
    app.processEvents()
    for frame in window.host.pane_frames:
        frame.pane.reconciler.flush_now()
    app.processEvents()
    try:
        yield app, window, session
    finally:
        window.close()
        app.processEvents()


def _layer(frame, role: str):
    """The realized layer record of one role, by its stable key."""
    key = f"{frame.paneId}:{role}"
    emitted = [rl.key for rl in frame.pane.reconciler.realized.layers]
    for layer in frame.pane.reconciler.realized.layers:
        if layer.key == key:
            return layer
    raise KeyError(f"{key} not emitted; this pane has {emitted}")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(*vector))
    return vector / norm if norm else vector


def _press_release(interactor, x: int, y: int) -> None:
    """One plain LMB click at a display pixel, through real VTK."""
    interactor.SetEventInformation(int(x), int(y), 0, 0, chr(0), 0, None)
    interactor.LeftButtonPressEvent()
    interactor.SetEventInformation(int(x), int(y), 0, 0, chr(0), 0, None)
    interactor.LeftButtonReleaseEvent()


# =====================================================================
# The real backend is installed, per pane
# =====================================================================


def test_each_pane_installs_the_shared_pick_backend(live_window):
    from apeGmsh.viewers.backends._pyvista_pick import PyVistaPickBackend

    _app, window, _session = live_window
    picks = [f.pane.pick for f in window.host.pane_frames]
    assert len(picks) == PANES
    assert all(p is not None and p.installed for p in picks)
    assert all(
        isinstance(p.pick_backend, PyVistaPickBackend) for p in picks
    )
    # One installation per pane — a shared one would route pane 2's
    # clicks into pane 1's picture.
    assert len({id(p.pick_backend) for p in picks}) == PANES
    # Observers really registered (the private tag ledger is the only
    # place the backend records them).
    assert all(p.pick_backend._tags for p in picks)  # noqa: SLF001


def test_the_glyph_cloud_is_pickable_in_vtk(live_window):
    """The scene-IR ``pickable`` flag is a promise; this is the actor.

    ``vtkCellPicker`` skips a non-pickable prop entirely, so until the
    actor carries the flag the glyph cloud is not a pick target at all
    — whatever the layer record says.
    """
    _app, window, _session = live_window
    frame = window.host.pane_frames[0]
    assert _layer(frame, "nodes").handle.actor.GetPickable() == 1
    # Outlines are decoration, not a target: a pickable outline would
    # answer clicks meant for the cloud drawn over it.
    assert _layer(frame, "outlines").handle.actor.GetPickable() == 0
    # Scope note, so the next slice does not over-trust this: the one
    # surface (INV-MESH-2) stays pickable, so a click on a FACE still
    # resolves through the ray-then-snap rule with or without the flag
    # above. This assertion is the pin for "the glyph cloud is itself a
    # target" — the property §8 names and the thing a Gauss pick, which
    # has no surface of its own, actually rides on.


def test_a_real_click_selects_the_node_it_landed_on(live_window):
    """The whole chain on real GL: interactor event → gesture machine →
    ``vtkCellPicker`` ray → nearest drawn target → the ONE store.

    Aimed down -Z at a known pose so the answer is not a matter of
    tolerance: the click lands on the TOP face, a few pixels off one of
    its own nodes, and no other node is anywhere near that world point.
    A back-face node would project to the same neighbourhood and is
    exactly what the ray-then-snap rule is there to exclude.
    """
    app, window, session = live_window
    frame = window.host.pane_frames[0]
    plotter = frame.plotter
    plotter.camera_position = "xy"
    plotter.reset_camera()
    plotter.render()
    app.processEvents()

    targets = frame.pane.reconciler.realized.targets.nodes
    assert targets is not None and targets.ids.size > 0
    display = np.asarray(
        frame.pane.pick.pick_backend.project_points(targets.coords)
    )
    # A node ON the face the camera sees, tie-broken toward the middle
    # of the picture so the ray cannot graze the silhouette.
    top = np.flatnonzero(
        targets.coords[:, 2] >= targets.coords[:, 2].max() - 1e-9
    )
    centre = display.mean(axis=0)
    row = int(top[np.argmin(np.einsum(
        "ij,ij->i", display[top] - centre, display[top] - centre,
    ))])
    # Nudge a few pixels inward — far less than the node spacing, so
    # the nearest target is unchanged, but safely off the silhouette.
    aim = display[row] + 4.0 * _unit(centre - display[row])

    _press_release(plotter.iren.interactor, aim[0], aim[1])
    app.processEvents()

    assert session.selection.kind == "nodes"
    assert session.selection.nodes == (int(targets.ids[row]),)


def test_the_selection_paints_a_layer_on_the_real_pane(live_window):
    """Criterion 11 on real GL: the highlight is an actor, and the
    signature gate let the repaint through."""
    app, window, session = live_window
    frame = window.host.pane_frames[0]
    targets = frame.pane.reconciler.realized.targets.nodes
    before = frame.plotter.renderer.GetActors().GetNumberOfItems()

    session.selection.set_nodes([int(targets.ids[0])])
    app.processEvents()
    frame.pane.reconciler.flush_now()
    app.processEvents()

    highlight = _layer(frame, "selection")
    assert highlight.handle.actor is not None
    assert highlight.handle.actor.GetPickable() == 0
    assert frame.plotter.renderer.GetActors().GetNumberOfItems() > before
    img = np.asarray(frame.plotter.screenshot(return_img=True))
    assert img.std() > 0, "pane rendered a flat frame after the highlight"


def test_closing_the_window_uninstalls_every_pick(qt_results, tmp_path,
                                                  monkeypatch):
    """The observers live on the interactor the pane is about to close.

    The test must NOT uninstall anything itself — that is the S3
    teardown lesson: a test that performs the thing it asserts passes
    against a product that does nothing, and Mesa is what finds out.
    """
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from qtpy.QtCore import QSettings

    from apeGmsh.viewers.session import SessionResultsWindow, SessionWindow

    ini = str(tmp_path / "teardown.ini")
    monkeypatch.setattr(
        SessionResultsWindow, "_layout_settings",
        staticmethod(lambda: QSettings(ini, QSettings.Format.IniFormat)),
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = qt_results.session()
    window = SessionWindow(session, title="s4-qt-teardown")
    window.show(blocking=False)
    app.processEvents()
    session.add_view()
    app.processEvents()
    picks = [f.pane.pick for f in window.host.pane_frames]
    assert len(picks) == 2 and all(p.installed for p in picks)

    window.close()
    app.processEvents()

    for pick in picks:
        assert pick.installed is False
        assert pick.pick_backend._tags == {}  # noqa: SLF001


def test_a_closed_pane_uninstalls_its_own_pick(live_window):
    """The same rule mid-session, on the header's close glyph."""
    app, window, session = live_window
    victim = session.panes[-1].id
    pick = window.host.frame(victim).pane.pick

    window.host.frame(victim).close_button.click()
    app.processEvents()

    assert victim not in [p.id for p in session.panes]
    assert pick.installed is False
    assert pick.pick_backend._tags == {}  # noqa: SLF001
