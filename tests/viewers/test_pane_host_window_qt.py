"""ADR 0098 Amendment 3 (S3) — the pane docks on real GL (``[qt]`` lane).

``test_pane_host_probe.py`` proves the geometry on a SYNTHETIC window
of four bare ``QtInteractor``\\ s. This file is criterion 18: the same
properties on the real :class:`SessionWindow` with four mesh panes,
each carrying a realized session picture — four distinct renderers,
independent cameras, one scalar bar per pane, a non-flat per-pane
screenshot, actors surviving a dock resize, a clean teardown, and the
lifecycle A3.1's evidence table names: **a live pane floated, resized,
re-docked and tabified, keeping its camera pose and a non-flat frame.**

Amendment 3 retired ``T(N)``, so the 4 → 5 re-tile that used to force a
reparent no longer happens. Floating a pane out and dropping it back
does the same thing to the platform window and is the gesture a user
actually performs, so that is what this drives.

Criterion 19 is ADR §11's S3 verify line (two views, different slots
and scopes) and criterion 20 is the toolbar acting on the active pane
only. Both need real cameras and real bars, so both live here.

Run per-file in a fresh process, under xvfb in CI::

    xvfb-run -a pytest tests/viewers/test_pane_host_window_qt.py -m qt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.qt

PANES = 4


@pytest.fixture
def qt_results(g, tmp_path: Path):
    """Two physical groups so the two panes can hold different scopes."""
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

    path = tmp_path / "pane_host_qt.h5"
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
    """A real :class:`SessionWindow` with ``PANES`` mesh panes."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from qtpy.QtCore import QSettings

    from apeGmsh.viewers.session import SessionResultsWindow, SessionWindow

    # Keep the real user's saved arrangement out of this.
    ini = str(tmp_path / "ResultsSession.ini")
    monkeypatch.setattr(
        SessionResultsWindow, "_layout_settings",
        staticmethod(lambda: QSettings(ini, QSettings.Format.IniFormat)),
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = qt_results.session()
    window = SessionWindow(session, title="s3-qt")
    window.shell.window.resize(1400, 900)
    window.show(blocking=False)
    app.processEvents()
    for _ in range(PANES - 1):
        session.add_view()
    app.processEvents()
    for frame in window.host.pane_frames:
        frame.pane.reconciler.flush_now()
    app.processEvents()
    try:
        yield app, window, session
    finally:
        window.close()
        app.processEvents()


def _plotters(window):
    return [f.plotter for f in window.host.pane_frames]


# =====================================================================
# 18 — the probe's properties, on the real window
# =====================================================================


def test_c18_four_panes_own_four_renderers(live_window):
    _app, window, _session = live_window
    plotters = _plotters(window)
    assert len(plotters) == PANES
    assert all(p is not None for p in plotters)
    assert len({id(p.renderer) for p in plotters}) == PANES, "shared renderer"
    counts = [p.renderer.GetActors().GetNumberOfItems() for p in plotters]
    assert all(c >= 1 for c in counts), f"pane lost its actors: {counts}"


def test_c18_pane_cameras_are_independent(live_window):
    app, window, _session = live_window
    plotters = _plotters(window)
    before = [tuple(p.camera.position) for p in plotters]
    plotters[0].camera.position = (99.0, 99.0, 99.0)
    plotters[0].render()
    app.processEvents()
    after = [tuple(p.camera.position) for p in plotters]
    assert after[0] != before[0]
    for k in range(1, PANES):
        assert after[k] == before[k], f"pane{k} camera moved with pane0"


def test_c18_one_scalar_bar_per_pane(live_window):
    """INV-LEGEND-5 on the real thing: each pane carries the legends of
    ITS slots, and the per-plotter registry keeps them apart."""
    from apeGmsh.results.session import Contour

    app, window, session = live_window
    for pane in session.panes:
        pane.contour = Contour("displacement_z")
    app.processEvents()
    for frame in window.host.pane_frames:
        frame.pane.reconciler.flush_now()
    app.processEvents()

    for plotter in _plotters(window):
        assert len(plotter.scalar_bars) == 1


def test_c18_each_pane_screenshots_itself(live_window):
    app, window, _session = live_window
    app.processEvents()
    for plotter in _plotters(window):
        img = np.asarray(plotter.screenshot(return_img=True))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0
        assert img.std() > 0, "pane rendered a flat frame"


def test_c18_panes_survive_a_dock_resize(live_window):
    """The separator drag, restated against dock placement (A3.5)."""
    from qtpy import QtCore

    app, window, session = live_window
    docks = list(window.host.pane_docks)
    window.shell.window.resizeDocks(
        docks, [520, 260, 260, 260][:len(docks)], QtCore.Qt.Horizontal,
    )
    app.processEvents()
    for plotter in _plotters(window):
        plotter.render()
    app.processEvents()
    counts = [
        p.renderer.GetActors().GetNumberOfItems() for p in _plotters(window)
    ]
    assert all(c >= 1 for c in counts), f"lost actors on resize: {counts}"
    assert [d.objectName() for d in window.host.pane_docks] == [
        f"dock_session_pane_{p.id}" for p in session.panes
    ]


def test_c18_live_pane_survives_float_resize_redock_tabify(live_window):
    """The lifecycle A3.1's evidence table claims, on the REAL window.

    Floating a dock out and dropping it back REPARENTS its widget, and
    reparenting a GL-native widget destroys and recreates its platform
    window — the classic VTK black-pane class. The pane must come
    through every step as the same object, still holding its actors,
    its camera pose, and a non-flat frame. That is the amendment's
    whole promise: a pane you can pull out, size on its own, put back,
    and tab onto its neighbour.
    """
    app, window, session = live_window
    victim_id = session.panes[2].id
    frame = window.host.frame(victim_id)
    dock = window.host.dock(victim_id)
    plotter = frame.plotter
    plotter.camera.position = (7.0, 5.0, 3.0)
    plotter.render()
    app.processEvents()
    before_cam = tuple(plotter.camera.position)
    before_actors = plotter.renderer.GetActors().GetNumberOfItems()

    def _still_whole(step: str) -> None:
        for f in window.host.pane_frames:
            f.pane.reconciler.flush_now()
        plotter.render()
        app.processEvents()
        assert window.host.frame(victim_id) is frame, f"rebuilt at {step}"
        assert window.host.dock(victim_id) is dock, f"dock rebuilt at {step}"
        assert frame.plotter is plotter
        assert frame.parent() is not None
        assert plotter.renderer.GetActors().GetNumberOfItems() == (
            before_actors), f"lost actors at {step}"
        assert tuple(plotter.camera.position) == before_cam, (
            f"camera lost at {step}")
        img = np.asarray(plotter.screenshot(return_img=True))
        assert img.std() > 0, f"flat/black frame at {step}"

    dock.setFloating(True)
    app.processEvents()
    _still_whole("floated")

    dock.resize(900, 600)
    app.processEvents()
    assert (dock.width(), dock.height()) == (900, 600)
    _still_whole("resized")

    dock.setFloating(False)
    app.processEvents()
    assert not dock.isFloating()
    _still_whole("re-docked")

    neighbour = window.host.dock(session.panes[0].id)
    window.shell.window.tabifyDockWidget(neighbour, dock)
    window.host.set_active(victim_id)
    app.processEvents()
    assert dock in window.shell.window.tabifiedDockWidgets(neighbour)
    _still_whole("tabified")


def test_c18_teardown_leaves_no_live_interactor(qt_results, tmp_path,
                                                monkeypatch):
    """Closing the window closes every pane's interactor — with no
    shell interactor there is no orphan left over either (caution 1).

    The test must NOT close the plotters itself. An earlier version did,
    and that is exactly why this passed while the rest of the file
    segfaulted on Mesa: nothing in the product closed a pane's GL
    context, so a second window in the same process died on the leaked
    ones. Windows tolerated it; the xvfb lane did not.
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
    window = SessionWindow(session, title="s3-qt-teardown")
    window.show(blocking=False)
    app.processEvents()
    session.add_view()
    app.processEvents()
    plotters = _plotters(window)
    assert len(plotters) == 2

    window.close()
    app.processEvents()

    # Closed BY THE PRODUCT, on the window's own close path.
    for plotter in plotters:
        assert getattr(plotter, "_closed", False) is True, (
            "a pane's GL context outlived its window — the next "
            "interactor in this process is what pays for it"
        )
    # The shell never built one, so there is nothing else to leak.
    assert window.shell.plotter is None


@pytest.mark.parametrize("route", ["frame-glyph", "dock-button"])
def test_c18_a_closed_pane_closes_its_context(live_window, route):
    """The same rule mid-session, on BOTH close affordances: A3.2 made
    the pane docks closable, so the dock's own button has to take the
    pane's GL context with it exactly as the header's glyph does."""
    app, window, session = live_window
    victim = session.panes[3].id
    plotter = window.host.frame(victim).plotter

    if route == "frame-glyph":
        window.host.frame(victim).close_button.click()
    else:
        window.host.dock(victim).close()
    app.processEvents()

    assert victim not in [p.id for p in session.panes]
    assert getattr(plotter, "_closed", False) is True


# =====================================================================
# 19 — ADR §11's S3 verify line
# =====================================================================


def test_c19_two_views_different_slots_and_scopes(live_window):
    """Pane A: a contour → ONE legend. Pane B: deform on, every slot
    empty → ZERO legends (the §5 oracle, per pane). Different scopes,
    so the two pictures are different cell sets too."""
    from apeGmsh.results.session import Contour, Deform, Scope

    app, window, session = live_window
    a, b = session.panes[0], session.panes[1]
    a.contour = Contour("displacement_z")
    a.scope = Scope(axis="physical_groups", names=("Left",))
    b.deform = Deform("displacement")
    b.scope = Scope(axis="physical_groups", names=("Right",))
    app.processEvents()
    for frame in window.host.pane_frames:
        frame.pane.reconciler.flush_now()
    app.processEvents()

    plot_a = window.host.frame(a.id).plotter
    plot_b = window.host.frame(b.id).plotter
    assert len(plot_a.scalar_bars) == 1
    assert len(plot_b.scalar_bars) == 0

    img_a = np.asarray(plot_a.screenshot(return_img=True))
    img_b = np.asarray(plot_b.screenshot(return_img=True))
    assert img_a.std() > 0 and img_b.std() > 0
    assert not np.array_equal(img_a, img_b), "two views, one picture"


# =====================================================================
# 20 — the toolbar acts on the active pane only
# =====================================================================


def test_c20_toolbar_camera_and_fit_act_on_the_active_pane(live_window):
    app, window, session = live_window
    target = session.panes[2].id
    window.host.set_active(target)
    app.processEvents()

    plotters = {f.paneId: f.plotter for f in window.host.pane_frames}
    before = {
        pid: tuple(p.camera.position) for pid, p in plotters.items()
    }
    assert window.shell.plotter is plotters[target]

    window.shell._vw._snap_view("top")  # noqa: SLF001 — the toolbar action
    app.processEvents()
    after = {pid: tuple(p.camera.position) for pid, p in plotters.items()}
    assert after[target] != before[target]
    for pid, position in before.items():
        if pid != target:
            assert after[pid] == position, f"{pid} moved with the toolbar"

    window.shell._vw._fit_view()  # noqa: SLF001
    app.processEvents()
    refit = {pid: tuple(p.camera.position) for pid, p in plotters.items()}
    for pid, position in after.items():
        if pid != target:
            assert refit[pid] == position


def test_c20_screenshot_uses_the_active_pane(live_window, tmp_path):
    """The File → Save screenshot path resolves through the same one
    active object — a still of the window would be the wrong picture."""
    app, window, session = live_window
    target = session.panes[1].id
    window.host.set_active(target)
    app.processEvents()

    out = tmp_path / "active.png"
    window.shell.plotter.screenshot(str(out), transparent_background=False)
    assert out.exists() and out.stat().st_size > 0
