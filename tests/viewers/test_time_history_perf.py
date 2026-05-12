"""TimeHistoryPanel dispatcher migration (UI storm follow-up #4).

Each step change moves a vertical "current step" marker on the
matplotlib chart and triggers a ``canvas.draw_idle``. Without
coalesce, a rapid scrubber drag fires one redraw per slider tick.
The ``attach_dispatcher`` migration collapses the storm to one
redraw per Qt tick.

Same pattern as :class:`OutlineTree` / :class:`DiagramSettingsTab` /
:class:`PickReadoutHUD` — bench file (`@pytest.mark.bench`); two
correctness assertions.
"""
from __future__ import annotations

import os
from typing import Any

import pytest


# Force offscreen Qt before importing qtpy.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _DirectorStub:
    """Director shape for TimeHistoryPanel: step/stage subs + a
    ``read_history`` shim + ``current_time`` for the marker line."""

    def __init__(self):
        self._on_step = []
        self._on_stage = []
        self.stage_id = None
        self._step_index = 0
        self._current_time_val = 0.0

    def subscribe_step(self, cb):
        self._on_step.append(cb)
        return lambda: (
            self._on_step.remove(cb) if cb in self._on_step else None
        )

    def subscribe_stage(self, cb):
        self._on_stage.append(cb)
        return lambda: (
            self._on_stage.remove(cb) if cb in self._on_stage else None
        )

    def read_history(self, node_id, component):
        import numpy as np
        # 10 steps of fake data so refresh() draws a plot instead of
        # rendering the empty-state.
        return np.arange(10, dtype=float), np.zeros(10)

    def current_time(self):
        return self._current_time_val


@pytest.fixture
def panel_and_director():
    """Build TimeHistoryPanel + stub director with proper teardown.

    Without the teardown, the matplotlib FigureCanvas hosted inside
    the panel leaves Qt event-loop state that desynchronises later
    tests' QTimer-driven animations (observed: ``test_time_scrubber_
    animation::test_running_timer_advances_step``).
    """
    from qtpy import QtWidgets
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    director = _DirectorStub()
    from apeGmsh.viewers.ui._time_history import TimeHistoryPanel
    panel = TimeHistoryPanel(director, node_id=1, component="ux")
    yield panel, director
    try:
        panel.close()
    except Exception:
        pass
    try:
        # Close the matplotlib figure + delete the Qt widget so the
        # canvas's Qt event handlers don't linger across tests. The
        # explicit ``processEvents`` drains deletion + matplotlib
        # cleanup callbacks before pytest moves on — without it the
        # next test's QTimer-driven animation desynchronises.
        import matplotlib.pyplot as plt
        plt.close(panel._fig)
        panel._canvas.close()
        panel.widget.close()
        panel.widget.deleteLater()
        from qtpy import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
            app.processEvents()
    except Exception:
        pass


def test_attach_dispatcher_swaps_step_subscription(panel_and_director) -> None:
    """After ``attach_dispatcher``, the panel's legacy subs are gone
    from the Director's subscriber lists."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    panel, director = panel_and_director
    # Initial state: panel holds one step + one stage subscriber.
    assert len(director._on_step) == 1
    assert len(director._on_stage) == 1

    deferred = []
    dispatcher = Dispatcher(
        director=director,
        pump_step=lambda _l: None,
        pump_deform=lambda _l: None,
        pump_gate=lambda: None,
        pump_restack=lambda: None,
        render=lambda: None,
        defer_fn=deferred.append,
    )
    panel.attach_dispatcher(dispatcher)

    # Legacy subs are dropped.
    assert len(director._on_step) == 0
    assert len(director._on_stage) == 0


@pytest.mark.bench
def test_rapid_scrubber_drag_collapses_to_one_redraw(panel_and_director) -> None:
    """100 STEP_CHANGED fires in one tick → exactly one marker
    redraw after the dispatcher's UI flush."""
    from apeGmsh.viewers.diagrams._dispatch import (
        Dispatcher,
        STEP_CHANGED,
    )

    panel, director = panel_and_director
    deferred = []
    dispatcher = Dispatcher(
        director=director,
        pump_step=lambda _l: None,
        pump_deform=lambda _l: None,
        pump_gate=lambda: None,
        pump_restack=lambda: None,
        render=lambda: None,
        defer_fn=deferred.append,
    )
    panel.attach_dispatcher(dispatcher)

    callbacks = [0]
    original = panel._on_step.__func__

    def _counting_on_step(step_index):
        callbacks[0] += 1
        original(panel, step_index)

    panel._on_step = _counting_on_step    # type: ignore[method-assign]

    for step in range(100):
        dispatcher.fire(STEP_CHANGED, payload=None)

    # Pre-drain: nothing fired yet.
    assert callbacks[0] == 0

    for fn in list(deferred):
        fn()

    print(
        f"\n[time_history scrubber storm] 100 STEP_CHANGED fires -> "
        f"{callbacks[0]} marker redraws after coalesce flush"
    )

    # Coalesce contract: same dedup key for all 100 → exactly 1 call.
    assert callbacks[0] == 1, (
        f"Expected 1 redraw after coalesce; got {callbacks[0]} "
        f"for 100 STEP_CHANGED fires"
    )
