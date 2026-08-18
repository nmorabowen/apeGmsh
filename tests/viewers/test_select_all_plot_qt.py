"""ADR 0098 §6/§9 (S4-2) — the plot pane in a real window (``[qt]``).

``test_select_all_plot.py`` proves everything a canvas can be asked
about without a screen: which arrays each line carries, which gesture
costs which repaint, what the outline menu offers. This file proves
the half that needs a real window and a real Mesa/GL context:

* a mesh pane and a plot pane really do tile together — the chart is a
  live widget with a non-zero size inside the splitter, not a canvas
  that renders to nothing because the frame never gave it room;
* the outline's context menu resolves over REAL item geometry
  (``visualItemRect`` answers a null rect on a tree that was never
  laid out, so the offscreen lane cannot tell a working hit-test from
  one that always lands on row 0);
* the window survives closing with both kinds of pane open. That is
  the S3 lesson (Amendment 1 caution 1): the panes own the GL
  contexts, Windows tolerates a leak and Mesa segfaults on the NEXT
  interactor in the process — and a plot pane, which owns no context
  at all, must not break the path that closes the ones that do.

The teardown assertions deliberately do NOT close anything themselves
(S3's retracted test closed the plotters and then asserted they were
closed, which passes against a product that does nothing).

Run per-file in a fresh process, under xvfb in CI::

    xvfb-run -a pytest tests/viewers/test_select_all_plot_qt.py -m qt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.qt

STAGE = "grav"
T = 4


@pytest.fixture
def qt_results(g, tmp_path: Path):
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left")
    g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="right")
    g.physical.add_volume("left", name="Left")
    g.physical.add_volume("right", name="Right")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.stack([node_ids * 1.0 + t * 10.0 for t in range(T)])

    path = tmp_path / "select_all_qt.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(T, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _window(qt_results, tmp_path, monkeypatch, title: str):
    """A real :class:`SessionWindow` on its own QSettings scope."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("matplotlib")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from qtpy.QtCore import QSettings

    from apeGmsh.viewers.session import SessionResultsWindow, SessionWindow

    ini = str(tmp_path / f"{title}.ini")
    monkeypatch.setattr(
        SessionResultsWindow, "_layout_settings",
        staticmethod(lambda: QSettings(ini, QSettings.Format.IniFormat)),
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = qt_results.session()
    window = SessionWindow(session, title=title)
    window.shell.window.resize(1280, 800)
    window.show(blocking=False)
    app.processEvents()
    return app, window, session


@pytest.fixture
def live_window(qt_results, tmp_path, monkeypatch):
    app, window, session = _window(
        qt_results, tmp_path, monkeypatch, "s4-2-qt",
    )
    try:
        yield app, window, session
    finally:
        window.close()
        app.processEvents()


# =====================================================================
# A plot pane is a pane — with a real size, beside a real mesh pane
# =====================================================================


def test_a_plot_pane_tiles_beside_a_mesh_pane_with_room_to_draw(
    live_window,
):
    """The offscreen lane can assert the canvas exists; only a laid-out
    window can assert it was given the pane's room.

    "Visible with a non-zero size" is NOT the assertion: a canvas that
    is merely re-parented onto the pane is visible and keeps its own
    ~500x300 size hint forever, so that test passes against a pane that
    never put it in a layout. What separates the two is that a laid-out
    canvas FILLS its pane — the layout has zero margins and one
    stretching child — and therefore tracks the tiling.
    """
    from apeGmsh.results.session import PlotSeries, PlotSource
    from apeGmsh.viewers.ui._layout_metrics import LAYOUT

    app, window, session = live_window
    node_id = int(session.results.fem.nodes.ids[0])
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    app.processEvents()

    frame = window.host.frame(plot.id)
    frame.pane.reconciler.flush_now()
    app.processEvents()

    assert len(window.host.pane_frames) == 2
    assert frame.width() >= LAYOUT.pane_min_width
    pane, canvas = frame.pane, frame.pane.chart.widget
    assert canvas.isVisible()
    assert canvas.width() >= pane.width() - 1, (
        f"canvas {canvas.width()}x{canvas.height()} does not fill "
        f"pane {pane.width()}x{pane.height()} — it was never laid out"
    )
    assert canvas.height() >= pane.height() - 1
    # And it drew the series, not an empty axes.
    assert len(frame.pane.chart.axes.lines) >= 1


def test_the_toolbar_finds_no_camera_on_an_active_plot_pane(live_window):
    """0087 INV-2 through the real shell: the camera presets act on the
    ACTIVE pane, and a chart has no camera. The provider must answer
    ``None`` rather than hand the toolbar a mesh pane's plotter."""
    app, window, session = live_window
    plot = session.add_plot()
    app.processEvents()
    window.host.set_active(plot.id)
    app.processEvents()

    assert window.host.active_pane_id == plot.id
    assert window._active_plotter() is None          # noqa: SLF001


# =====================================================================
# The outline's context menu, over real item geometry
# =====================================================================


def test_select_all_from_a_real_outline_row(live_window, monkeypatch):
    """``visualItemRect`` is meaningful only on a laid-out tree, so this
    is the lane that can tell a working hit-test from one that always
    lands on the first row."""
    from qtpy import QtWidgets

    app, window, session = live_window
    tree = window.outline
    widget = tree.widget
    widget.expandAll()
    app.processEvents()

    right = tree._name_items["physical_groups"]["Right"]   # noqa: SLF001
    rect = widget.visualItemRect(right)
    assert rect.height() > 0, "the tree was never laid out"

    chosen: list = []

    def fake_exec(self, *_args, **_kwargs):
        for action in self.actions():
            if action.text().startswith("Select all nodes"):
                chosen.append(action.text())
                return action
        return None

    monkeypatch.setattr(QtWidgets.QMenu, "exec_", fake_exec, raising=False)
    tree._on_context_menu(rect.center())                   # noqa: SLF001

    assert chosen, "no Select all entry was offered"
    assert session.selection.kind == "nodes"
    # "Right" is the far box; the set must be ITS nodes, not the whole
    # model and not the first row's.
    coords = np.asarray(session.results.fem.nodes.coords, dtype=np.float64)
    ids = np.asarray(session.results.fem.nodes.ids, dtype=np.int64)
    far = {int(i) for i in ids[coords[:, 0] >= 2.5]}
    assert set(session.selection.nodes) == far


def test_a_select_all_highlights_in_a_real_pane(live_window, monkeypatch):
    """§8 end to end on GL: the outline fills the set, and the pane that
    HAS its node glyphs on paints the highlight over its own cloud —
    the layer the offscreen lane can only see as a scene-IR record."""
    from qtpy import QtWidgets

    from apeGmsh.results.session import MeshStyle

    app, window, session = live_window
    view = session.panes[0]
    view.style = MeshStyle(mesh=True, outlines=True, nodes=True, gauss=False)
    app.processEvents()
    frame = window.host.frame(view.id)
    frame.pane.reconciler.flush_now()
    app.processEvents()
    assert f"{view.id}:selection" not in _keys(frame)

    tree = window.outline
    tree.widget.expandAll()
    app.processEvents()
    left = tree._name_items["physical_groups"]["Left"]      # noqa: SLF001

    monkeypatch.setattr(
        QtWidgets.QMenu, "exec_",
        lambda self, *_a, **_k: next(
            (a for a in self.actions()
             if a.text().startswith("Select all nodes")), None,
        ),
        raising=False,
    )
    tree._on_context_menu(                                  # noqa: SLF001
        tree.widget.visualItemRect(left).center(),
    )
    app.processEvents()
    frame.pane.reconciler.flush_now()
    app.processEvents()

    assert session.selection.nodes
    assert f"{view.id}:selection" in _keys(frame)


def _keys(frame) -> "set[str]":
    realized = frame.pane.reconciler.realized
    return set() if realized is None else {l.key for l in realized.layers}


# =====================================================================
# Teardown — the plot pane must not break the path that frees contexts
# =====================================================================


def test_closing_the_window_frees_every_mesh_context(
    qt_results, tmp_path, monkeypatch,
):
    """Amendment 1 caution 1, with a plot pane in the mix.

    The panes own the GL contexts and a plot pane owns none, so the
    close path walks a heterogeneous set. Windows tolerates a leaked
    context; Mesa segfaults on the NEXT interactor in the process,
    which is why this assertion belongs in this lane.

    The test does NOT close the panes itself — it closes the WINDOW and
    asks what happened, because a test that performs the thing it
    asserts passes against a product that does nothing.
    """
    app, window, session = _window(
        qt_results, tmp_path, monkeypatch, "s4-2-qt-teardown",
    )
    plot = session.add_plot()
    app.processEvents()

    frames = list(window.host.pane_frames)
    assert len(frames) == 2
    mesh_frames = [f for f in frames if f.is_mesh]
    plot_frames = [f for f in frames if not f.is_mesh]
    assert mesh_frames and plot_frames
    assert all(f.plotter is not None for f in mesh_frames)
    assert all(f.plotter is None for f in plot_frames)
    picks = [f.pane.pick for f in mesh_frames]
    assert all(p is not None and p.installed for p in picks)
    plot_reconciler = plot_frames[0].pane.reconciler
    assert plot_reconciler._on_tick in session._subscribers  # noqa: SLF001

    window.close()
    app.processEvents()

    for pick in picks:
        assert pick.installed is False
    assert plot_reconciler._on_tick not in session._subscribers  # noqa: SLF001
    assert plot.id in [p.id for p in session.panes]  # closing ≠ deleting


def test_closing_a_plot_pane_leaves_the_mesh_pane_alive(live_window):
    """The header's close glyph on a plot must take only that pane —
    the mesh pane beside it keeps its context and its pick."""
    app, window, session = live_window
    plot = session.add_plot()
    app.processEvents()
    mesh_id = session.panes[0].id
    mesh_pick = window.host.frame(mesh_id).pane.pick
    plot_reconciler = window.host.frame(plot.id).pane.reconciler

    window.host.frame(plot.id).close_button.click()
    app.processEvents()

    assert plot.id not in [p.id for p in session.panes]
    assert plot_reconciler._disposed                    # noqa: SLF001
    assert mesh_pick.installed is True
    assert window.host.frame(mesh_id).plotter is not None
