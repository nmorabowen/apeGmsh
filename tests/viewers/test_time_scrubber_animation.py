"""Headless coverage for ``TimeScrubberDock`` animation transport.

These tests exercise the Play / FPS / Loop machinery without spinning up
the full Results pipeline. A small ``StubDirector`` mimics the four
methods the scrubber actually uses:

    step_index, set_step, n_steps, current_time, subscribe_step,
    subscribe_stage

The animation logic is exercised by calling ``_on_animation_tick``
directly — that's deterministic and avoids timer-race flakiness — plus
one end-to-end test that drives the QTimer through ``processEvents`` to
confirm the timer wiring itself is correct.
"""
from __future__ import annotations

from typing import Callable, Optional

import pytest


# QApplication is module-scoped: many viewer tests share one and Qt
# really doesn't like a second one in the same process.
@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# Stub director
# =====================================================================


class StubDirector:
    """Mimics the slice of ResultsDirector the scrubber consumes."""

    def __init__(self, n_steps: int = 5) -> None:
        self._n = int(n_steps)
        self._step = 0
        self._step_subs: list[Callable[[int], None]] = []
        self._stage_subs: list[Callable[[str], None]] = []
        self.set_step_calls: list[int] = []

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def n_steps(self) -> int:
        return self._n

    def set_step(self, step: int) -> None:
        clamped = max(0, min(int(step), max(0, self._n - 1)))
        self._step = clamped
        self.set_step_calls.append(clamped)
        for cb in list(self._step_subs):
            cb(clamped)

    def current_time(self) -> Optional[float]:
        return float(self._step)

    def subscribe_step(self, cb: Callable[[int], None]) -> None:
        self._step_subs.append(cb)

    def subscribe_stage(self, cb: Callable[[str], None]) -> None:
        self._stage_subs.append(cb)

    def fire_stage(self, stage_id: str = "stage_2") -> None:
        for cb in list(self._stage_subs):
            cb(stage_id)


@pytest.fixture
def make_dock(qapp):
    """Factory returning ``(dock, stub_director)`` for a given step count."""
    from apeGmsh.viewers.ui._time_scrubber import TimeScrubberDock

    created: list = []

    def _make(n_steps: int = 5):
        director = StubDirector(n_steps=n_steps)
        dock = TimeScrubberDock(director)
        created.append(dock)
        return dock, director

    yield _make
    # No teardown required — Qt cleans widgets when refs drop and the
    # module-scoped QApplication outlives the test.


# =====================================================================
# Loop modes — direct tick
# =====================================================================


def test_once_advances_one_step_per_tick(make_dock):
    dock, d = make_dock(n_steps=4)
    dock._loop_combo.setCurrentIndex(
        dock._loop_combo.findData("once"),
    )
    d.set_step(0)
    d.set_step_calls.clear()

    dock._on_animation_tick()
    dock._on_animation_tick()
    dock._on_animation_tick()

    assert d.set_step_calls == [1, 2, 3]


def test_once_stops_at_last_step(make_dock):
    dock, d = make_dock(n_steps=3)
    dock._loop_combo.setCurrentIndex(dock._loop_combo.findData("once"))

    # Pretend the user pressed Play — start the timer so we can verify
    # the auto-stop releases it.
    dock._anim_timer.start(10)
    assert dock._anim_timer.isActive()

    d.set_step(2)    # at last step already
    d.set_step_calls.clear()

    dock._on_animation_tick()    # should set last + stop
    assert d.set_step_calls == [2]
    assert not dock._anim_timer.isActive()


def test_loop_wraps_to_zero(make_dock):
    dock, d = make_dock(n_steps=3)
    dock._loop_combo.setCurrentIndex(dock._loop_combo.findData("loop"))
    d.set_step(2)
    d.set_step_calls.clear()

    dock._on_animation_tick()
    dock._on_animation_tick()
    dock._on_animation_tick()

    assert d.set_step_calls == [0, 1, 2]


def test_bounce_reverses_at_end_then_at_start(make_dock):
    dock, d = make_dock(n_steps=4)
    dock._loop_combo.setCurrentIndex(dock._loop_combo.findData("bounce"))
    d.set_step(0)
    dock._anim_direction = +1
    d.set_step_calls.clear()

    # Forward to last step.
    dock._on_animation_tick(); dock._on_animation_tick(); dock._on_animation_tick()
    assert d.set_step_calls == [1, 2, 3]
    assert dock._anim_direction == +1

    # Next tick at last step reverses and steps back.
    dock._on_animation_tick()
    assert d.set_step_calls[-1] == 2
    assert dock._anim_direction == -1

    # Continue down to step 0.
    dock._on_animation_tick()
    dock._on_animation_tick()
    assert d.set_step_calls[-2:] == [1, 0]

    # Next tick at step 0 reverses again and steps forward.
    dock._on_animation_tick()
    assert d.set_step_calls[-1] == 1
    assert dock._anim_direction == +1


def test_stage_change_stops_animation(make_dock):
    dock, d = make_dock(n_steps=5)
    dock._loop_combo.setCurrentIndex(dock._loop_combo.findData("loop"))
    dock._anim_timer.start(10)
    dock._btn_play.blockSignals(True)
    dock._btn_play.setChecked(True)
    dock._btn_play.blockSignals(False)

    d.fire_stage("stage_2")

    assert not dock._anim_timer.isActive()
    assert not dock._btn_play.isChecked()


# =====================================================================
# FPS plumbing
# =====================================================================


def test_interval_ms_matches_fps(make_dock):
    dock, _ = make_dock(n_steps=3)
    dock._fps_spin.setValue(30)
    assert dock._interval_ms() == round(1000 / 30)
    dock._fps_spin.setValue(60)
    assert dock._interval_ms() == round(1000 / 60)
    dock._fps_spin.setValue(1)
    assert dock._interval_ms() == 1000


def test_fps_change_while_playing_updates_interval(make_dock):
    dock, _ = make_dock(n_steps=5)
    dock._fps_spin.setValue(30)
    dock._anim_timer.start(dock._interval_ms())
    assert dock._anim_timer.interval() == round(1000 / 30)

    dock._fps_spin.setValue(10)
    assert dock._anim_timer.interval() == 100
    dock._anim_timer.stop()


# =====================================================================
# Toggle behaviour
# =====================================================================


def test_play_does_nothing_when_only_one_step(make_dock):
    dock, d = make_dock(n_steps=1)
    # n_steps=1 ⇒ refresh disables the play button. Still test that
    # programmatic toggling is a no-op so future changes can't sneak
    # through a state where the timer starts on a 1-step file.
    dock._btn_play.setChecked(True)
    assert not dock._btn_play.isChecked()
    assert not dock._anim_timer.isActive()


def test_play_at_last_step_in_once_mode_resets_to_zero(make_dock):
    dock, d = make_dock(n_steps=4)
    dock._loop_combo.setCurrentIndex(dock._loop_combo.findData("once"))
    d.set_step(3)    # last step
    d.set_step_calls.clear()

    dock._btn_play.setChecked(True)
    # _toggle_play(True) should have rewound to 0 and started the timer.
    assert d.set_step_calls == [0]
    assert dock._anim_timer.isActive()
    dock._anim_timer.stop()


# =====================================================================
# End-to-end timer wiring (single round trip)
# =====================================================================


def test_running_timer_advances_step(qapp, make_dock):
    """Confirm the QTimer is actually wired to the tick slot.

    Uses 1 ms interval and ``processEvents`` so the test stays bounded.
    """
    QtCore = pytest.importorskip("qtpy.QtCore")
    dock, d = make_dock(n_steps=5)
    dock._loop_combo.setCurrentIndex(dock._loop_combo.findData("loop"))
    d.set_step(0)
    d.set_step_calls.clear()

    dock._anim_timer.setInterval(1)
    dock._anim_timer.start()
    # Pump the event loop briefly.
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while deadline.elapsed() < 50 and len(d.set_step_calls) < 3:
        qapp.processEvents()
    dock._anim_timer.stop()

    assert len(d.set_step_calls) >= 1
