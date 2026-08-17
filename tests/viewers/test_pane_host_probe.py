"""ADR 0098 Amendment 1 probe — N QtInteractors in nested QSplitters.

The implementation plan names multi-interactor driver quirks as
"the single biggest S3 risk" and requires a prototype on Windows AND
under CI Mesa *before* the pane-host amendment is written. This file
is the CI-Mesa half: the `qt-window-tests` lane runs it under
`xvfb-run` with a real xcb platform and Mesa's software GL, which is
the only Linux oracle the project has.

It stays after the spike as S3's regression guard — the day a VTK or
pyvistaqt bump makes four live interactors share a camera or leak a
scalar bar, the pane host breaks and this fails first.

Marked ``qt``: real GL contexts. There is no offscreen variant —
`QT_QPA_PLATFORM=offscreen` gives VTK no usable pixel format (a
single interactor segfaults just the same, on Windows and in the
default CI lane alike), which is precisely why ``MeshPane`` takes an
injectable backend factory.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.qt

PANES = 4


def _mesh(seed: int):
    """A distinct mesh per pane, carrying a distinct scalar range."""
    import pyvista as pv

    sphere = pv.Sphere(
        radius=1.0 + 0.25 * seed, theta_resolution=16, phi_resolution=16,
    )
    sphere["field"] = np.linspace(seed, seed + 10, sphere.n_points)
    return sphere


def _build(n, parent, QtCore, QtInteractor, QtWidgets):
    """n interactors in NESTED splitters (an h-splitter of v-splitters)
    — the workshop's recommended baseline geometry, not a flat row."""
    outer = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, parent)
    interactors = []
    for _ in range((n + 1) // 2):
        inner = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, outer)
        for _ in range(min(2, n - len(interactors))):
            iren = QtInteractor(parent=inner)
            inner.addWidget(iren)
            interactors.append(iren)
        outer.addWidget(inner)
    parent.setCentralWidget(outer)
    return interactors


@pytest.fixture
def pane_host():
    """A live window of PANES interactors in nested splitters, each
    showing its own mesh with its own scalar bar. Torn down whatever
    the test does."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    QtCore = pytest.importorskip("qtpy.QtCore")
    from pyvistaqt import QtInteractor

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.resize(900, 700)
    irens = _build(PANES, win, QtCore, QtInteractor, QtWidgets)
    win.show()
    app.processEvents()
    for k, iren in enumerate(irens):
        iren.add_mesh(
            _mesh(k), scalars="field", scalar_bar_args={"title": f"pane{k}"},
        )
        iren.reset_camera()
    app.processEvents()
    try:
        yield app, win, irens
    finally:
        for iren in irens:
            if iren is not None:
                iren.close()
        app.processEvents()
        win.close()
        app.processEvents()


def test_four_interactors_render_independent_scenes(pane_host):
    """Four live QtInteractors coexist in one window, each owning a
    distinct renderer with its own actors. If N interactors could not
    share a process, S3's pane host would need one render window with
    N VTK viewports instead."""
    _app, _win, irens = pane_host

    assert len(irens) == PANES
    assert len({id(i.renderer) for i in irens}) == PANES, "shared renderer"
    counts = [i.renderer.GetActors().GetNumberOfItems() for i in irens]
    assert all(c >= 1 for c in counts), f"pane lost its actor: {counts}"


def test_pane_cameras_are_independent(pane_host):
    """Moving one pane's camera must not move any other's — per-pane
    pose is the whole point of N panes (§7 unlinked views)."""
    app, _win, irens = pane_host

    before = [tuple(i.camera.position) for i in irens]
    irens[0].camera.position = (99.0, 99.0, 99.0)
    irens[0].render()
    app.processEvents()
    after = [tuple(i.camera.position) for i in irens]

    assert after[0] != before[0], "pane0 camera did not move"
    for k in range(1, PANES):
        assert after[k] == before[k], f"pane{k} camera moved with pane0"


def test_scalar_bars_stay_in_their_own_pane(pane_host):
    """Per-plotter scalar-bar registry (INV-LEGEND-5): the plan sizes
    per-pane bars as 'free' because each plotter keeps its own. Prove
    it — a leak here would make the legend law unimplementable."""
    _app, _win, irens = pane_host

    for k, iren in enumerate(irens):
        assert set(iren.scalar_bars.keys()) == {f"pane{k}"}


def test_each_pane_screenshots_itself(pane_host):
    """Per-pane stills — S1's stills law rides one screenshot per
    pane, not one per window."""
    _app, _win, irens = pane_host

    for iren in irens:
        img = np.asarray(iren.screenshot(return_img=True))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0
        assert img.std() > 0, "pane rendered a flat frame"


def test_panes_survive_a_splitter_resize(pane_host):
    """Dragging a splitter must not cost a pane its GL context."""
    app, win, irens = pane_host

    win.resize(1200, 500)
    app.processEvents()
    for iren in irens:
        iren.render()
    app.processEvents()

    counts = [i.renderer.GetActors().GetNumberOfItems() for i in irens]
    assert all(c >= 1 for c in counts), f"lost actors on resize: {counts}"


def test_a_pane_can_be_removed_and_another_added(pane_host):
    """S3 adds and closes panes at runtime; the surviving panes must
    keep rendering and the new one must come up live."""
    app, win, irens = pane_host
    from pyvistaqt import QtInteractor

    victim = irens[-1]
    irens[-1] = None
    victim.close()
    victim.setParent(None)
    app.processEvents()

    outer = win.centralWidget()
    inner = outer.widget(outer.count() - 1)
    fresh = QtInteractor(parent=inner)
    inner.addWidget(fresh)
    irens[-1] = fresh
    fresh.add_mesh(_mesh(9), scalars="field")
    fresh.reset_camera()
    fresh.render()
    app.processEvents()

    assert fresh.renderer.GetActors().GetNumberOfItems() >= 1
    for k in range(PANES - 1):
        assert irens[k].renderer.GetActors().GetNumberOfItems() >= 1


def test_a_live_interactor_survives_reparent_between_splitters(pane_host):
    """The riskiest GL operation Amendment 1's tiling law mandates.

    `T(N)` re-tiles whenever the column count changes (N = 2, 5, 10…),
    which MOVES a live pane from one splitter to another. Reparenting
    an OpenGL-native widget destroys and recreates its platform
    window — the classic VTK black-pane class, and driver-dependent,
    so Mesa and a real GPU can legitimately disagree here.

    The pane must come through as the SAME object, still holding its
    actors, still holding its camera pose, and still painting a
    non-flat frame. Rebuilding the pane instead of moving it would
    silently lose the camera (backend state, not session state, so
    nothing downstream would catch it) — hence the identity assert.
    """
    app, win, irens = pane_host
    QtCore = pytest.importorskip("qtpy.QtCore")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")

    outer = win.centralWidget()
    victim = irens[0]
    victim.camera.position = (7.0, 5.0, 3.0)
    victim.render()
    app.processEvents()
    before_cam = tuple(victim.camera.position)
    before_actors = victim.renderer.GetActors().GetNumberOfItems()

    # The 4 -> 5 re-tile shape: a third column appears and an existing
    # live pane moves into it.
    new_col = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, outer)
    outer.addWidget(new_col)
    new_col.addWidget(victim)  # <- the reparent under test
    app.processEvents()
    victim.render()
    app.processEvents()

    assert victim is irens[0], "pane was rebuilt, not moved"
    assert victim.parent() is not None
    assert victim.renderer.GetActors().GetNumberOfItems() == before_actors
    assert tuple(victim.camera.position) == before_cam, "camera lost"
    img = np.asarray(victim.screenshot(return_img=True))
    assert img.std() > 0, "reparented pane paints a flat/black frame"

    # The panes that did not move are unaffected.
    for k in range(1, PANES):
        assert irens[k].renderer.GetActors().GetNumberOfItems() >= 1
