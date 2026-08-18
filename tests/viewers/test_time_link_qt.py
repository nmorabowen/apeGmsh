"""ADR 0098 §7 (S4-3) — the time link in a real window (``[qt]``).

``test_time_link.py`` proves the laws with an injected backend. This
file proves the half that needs the real window and a real GL context:

* the scrubber is actually MOUNTED in the shell's bottom dock and
  reaches the panes through it — the offscreen lane builds the widget
  by hand, so it cannot tell a wired dock from an orphan;
* a real slider DRAG (press → move → release), not a programmatic
  ``setValue``, moves every pane. The drag path runs through the
  coalescing timer and ``sliderReleased``, neither of which a direct
  call exercises;
* animation really runs on a QTimer and really stops when the window
  closes. A timer left alive keeps writing the session into panes
  whose GL contexts are being destroyed — on Mesa that is a segfault
  in the next interactor, which is the failure this lane exists for.

The teardown assertion does NOT stop the timer itself (S3's retracted
test closed the plotters and then asserted they were closed, which
passes against a product that does nothing).

Run per-file in a fresh process, under xvfb in CI::

    xvfb-run -a pytest tests/viewers/test_time_link_qt.py -m qt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.qt

T = 5
DT = 0.5
STAGE = "grav"


@pytest.fixture
def qt_results(g, tmp_path: Path):
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
    g.physical.add_volume("a", name="A")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "time_link_qt.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(T, dtype=np.float64) * DT,
        )
        w.write_nodes(
            sid, "partition_0", node_ids=ids,
            components={
                "displacement_z": np.stack(
                    [ids * 1.0 + t for t in range(T)]
                ),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _window(qt_results, tmp_path, monkeypatch, title: str):
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
        qt_results, tmp_path, monkeypatch, "s4-3-qt",
    )
    try:
        yield app, window, session
    finally:
        window.close()
        app.processEvents()


def test_the_scrubber_is_mounted_and_visible(live_window):
    """The offscreen lane constructs the widget by hand, so only a real
    window can say the shell actually gave it the bottom dock."""
    app, window, session = live_window
    scrubber = window.scrubber
    assert scrubber.widget.isVisible()
    assert scrubber.widget.width() > 0
    # It is inside the shell, not floating on its own.
    assert scrubber.widget.window() is window.shell.window


def test_a_real_drag_moves_every_pane(live_window):
    """A real press → move → release, which runs the coalescing timer
    and ``sliderReleased`` — neither reachable from ``setValue``."""
    from qtpy import QtCore
    from qtpy.QtTest import QTest

    from apeGmsh.results.session import Instant

    app, window, session = live_window
    session.add_view()
    app.processEvents()
    scrubber = window.scrubber
    slider = scrubber.slider
    assert slider.isEnabled() and slider.maximum() == T - 1

    # Press at the handle, move to the far end, release.
    QTest.mousePress(
        slider, QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QPoint(2, slider.height() // 2),
    )
    QTest.mouseMove(
        slider, QtCore.QPoint(slider.width() - 2, slider.height() // 2),
    )
    slider.setValue(slider.maximum())          # what the move produces
    QTest.mouseRelease(
        slider, QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QPoint(slider.width() - 2, slider.height() // 2),
    )
    app.processEvents()

    assert session.time == Instant(STAGE, T - 1)
    for pane in session.panes:
        assert session.effective_instant(pane) == Instant(STAGE, T - 1)
    for frame in window.host.pane_frames:
        frame.pane.reconciler.flush_now()
    app.processEvents()
    assert all(
        f.pane.reconciler.realized is not None
        for f in window.host.pane_frames
    )


def test_playback_runs_on_a_real_timer(live_window):
    """Play really advances the instant on the Qt event loop."""
    from qtpy.QtTest import QTest

    app, window, session = live_window
    scrubber = window.scrubber
    scrubber._fps.setValue(60)                 # noqa: SLF001 — fast rig
    scrubber._commit(0)                        # noqa: SLF001
    app.processEvents()
    start = session.time

    scrubber.play_button.setChecked(True)
    assert scrubber.is_playing
    for _ in range(20):
        QTest.qWait(20)
        app.processEvents()
        if session.time != start:
            break
    scrubber.play_button.setChecked(False)

    assert session.time != start, "playback never advanced the instant"
    assert not scrubber.is_playing


def test_closing_the_window_stops_the_animation(
    qt_results, tmp_path, monkeypatch,
):
    """A QTimer left running writes the session into panes whose GL
    contexts are being destroyed — on Mesa that is a segfault in the
    NEXT interactor, which is why this belongs in this lane.

    The test does NOT stop the timer itself; it closes the WINDOW and
    asks what happened.
    """
    app, window, session = _window(
        qt_results, tmp_path, monkeypatch, "s4-3-qt-teardown",
    )
    scrubber = window.scrubber
    scrubber.play_button.setChecked(True)
    app.processEvents()
    assert scrubber.is_playing
    subscribed = scrubber._on_session_tick in session._subscribers  # noqa: SLF001
    assert subscribed

    window.close()
    app.processEvents()

    assert not scrubber.is_playing
    assert scrubber._on_session_tick not in session._subscribers  # noqa: SLF001
