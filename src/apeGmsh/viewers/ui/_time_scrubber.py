"""Time scrubber dock — bottom-of-window step slider + transport controls.

Single-row layout::

    [<<] [<] [|>] [>] [>>]   step ▮━━━━━━━━○━━━━━━ [ 42 / 100 ]   t=1.234e+00 s   FPS [30] Loop [Once▾]

* ``<<`` / ``>>`` jump to first / last step
* ``<`` / ``>`` step -1 / +1
* ``|>`` toggles a QTimer-driven animation. The timer ticks at
  ``1000 / fps`` ms; on each tick the scrubber asks the Director to
  advance one step. The Director fires ``on_step_changed`` and the
  scrubber updates the slider — the timer never touches the slider
  directly.
* The slider is QTimer-throttled to ~33 ms during drag so step changes
  don't fire faster than the renderer can keep up. The final position
  on release fires a non-throttled update so the rendered step matches
  the slider exactly.

Loop modes:

* ``once`` — play to the last step, then stop.
* ``loop`` — wrap to step 0 after the last step.
* ``bounce`` — reverse direction at end-of-stage; play forward again
  after reaching step 0. Never wraps.

Animation stops automatically on stage change (the new stage may have
a different step count, and the user almost certainly wants to look at
it before resuming).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class TimeScrubberDock:
    """Bottom dock — step slider and transport controls.

    Subscribes to the Director for stage / step changes; emits step
    moves back through ``director.set_step``.
    """

    DRAG_COALESCE_MS = 33    # ~30 fps during drag
    DEFAULT_FPS = 30
    FPS_MIN = 1
    FPS_MAX = 60

    LOOP_MODES = ("once", "loop", "bounce")

    def __init__(self, director: "ResultsDirector") -> None:
        QtWidgets, QtCore = _qt()
        self._director = director
        self._suppress_observer = False

        # Coalescing timer — fires DRAG_COALESCE_MS after the last
        # slider change while dragging. If the user releases the mouse
        # before it fires, sliderReleased forces an immediate update.
        self._drag_timer = QtCore.QTimer()
        self._drag_timer.setSingleShot(True)
        self._drag_timer.setInterval(self.DRAG_COALESCE_MS)
        self._drag_timer.timeout.connect(self._on_drag_timeout)
        self._pending_value: Optional[int] = None

        # Animation timer — drives the Play button. Fires every
        # 1000/fps ms; each tick advances one step honouring the
        # current loop mode. Direction state is only meaningful in
        # ``bounce`` mode; the other modes always step +1.
        self._anim_timer = QtCore.QTimer()
        self._anim_timer.setSingleShot(False)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_direction: int = +1

        # ── Layout ─────────────────────────────────────────────
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self._btn_first = QtWidgets.QToolButton()
        self._btn_first.setText("⏪")    # ⏪
        self._btn_first.setToolTip("Jump to first step")
        self._btn_first.clicked.connect(self._jump_first)

        self._btn_back = QtWidgets.QToolButton()
        self._btn_back.setText("◀")     # ◀
        self._btn_back.setToolTip("Step backward")
        self._btn_back.clicked.connect(lambda: self._step_delta(-1))

        self._btn_play = QtWidgets.QToolButton()
        self._btn_play.setText("▶")     # ▶
        self._btn_play.setCheckable(True)
        self._btn_play.setToolTip("Play / pause animation")
        self._btn_play.toggled.connect(self._toggle_play)

        self._btn_fwd = QtWidgets.QToolButton()
        self._btn_fwd.setText("▶︎")
        self._btn_fwd.setToolTip("Step forward")
        self._btn_fwd.clicked.connect(lambda: self._step_delta(+1))

        self._btn_last = QtWidgets.QToolButton()
        self._btn_last.setText("⏩")     # ⏩
        self._btn_last.setToolTip("Jump to last step")
        self._btn_last.clicked.connect(self._jump_last)

        for b in (self._btn_first, self._btn_back, self._btn_play,
                  self._btn_fwd, self._btn_last):
            layout.addWidget(b)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setTracking(True)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self._slider, stretch=1)

        self._step_label = QtWidgets.QLabel("0 / 0")
        self._step_label.setMinimumWidth(80)
        self._step_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        layout.addWidget(self._step_label)

        self._time_label = QtWidgets.QLabel("t = —")
        self._time_label.setMinimumWidth(140)
        self._time_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        layout.addWidget(self._time_label)

        # ── Animation controls ─────────────────────────────────
        fps_label = QtWidgets.QLabel("FPS")
        layout.addWidget(fps_label)

        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(self.FPS_MIN, self.FPS_MAX)
        self._fps_spin.setValue(self.DEFAULT_FPS)
        self._fps_spin.setToolTip(
            "Animation rate (frames per second).\n"
            "The timer ticks at 1000/fps ms; the renderer may not\n"
            "keep up at high fps for large meshes."
        )
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self._fps_spin)

        loop_label = QtWidgets.QLabel("Loop")
        layout.addWidget(loop_label)

        self._loop_combo = QtWidgets.QComboBox()
        self._loop_combo.addItem("Once",   "once")
        self._loop_combo.addItem("Loop",   "loop")
        self._loop_combo.addItem("Bounce", "bounce")
        self._loop_combo.setToolTip(
            "Once: stop at end of stage\n"
            "Loop: wrap to step 0\n"
            "Bounce: reverse direction at boundaries"
        )
        layout.addWidget(self._loop_combo)

        self._widget = widget

        # Subscribe to director events
        director.subscribe_step(self._on_director_step)
        director.subscribe_stage(self._on_director_stage)

        # Initial state
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        """Return the QWidget holding the scrubber row."""
        return self._widget

    def refresh(self) -> None:
        """Re-read director state and update slider / labels.

        Called on stage change. The slider's max becomes ``n_steps - 1``;
        the current step clamps to the new range.
        """
        n = max(0, int(self._director.n_steps))
        cur = max(0, min(int(self._director.step_index), max(0, n - 1)))
        self._suppress_observer = True
        try:
            self._slider.blockSignals(True)
            self._slider.setMaximum(max(0, n - 1))
            self._slider.setValue(cur)
        finally:
            self._slider.blockSignals(False)
            self._suppress_observer = False

        self._update_labels(cur, n)
        enabled = n > 1
        for b in (self._btn_first, self._btn_back, self._btn_play,
                  self._btn_fwd, self._btn_last):
            b.setEnabled(enabled)
        self._slider.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Slot handlers — slider drag
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        if self._suppress_observer:
            return
        # Coalesce intermediate values while the user drags. The most
        # recent value wins.
        self._pending_value = int(value)
        if not self._drag_timer.isActive():
            self._drag_timer.start()

    def _on_slider_released(self) -> None:
        # Commit immediately on release — bypass the timer so the
        # final rendered step matches the slider position exactly.
        if self._drag_timer.isActive():
            self._drag_timer.stop()
        if self._pending_value is not None:
            v = self._pending_value
            self._pending_value = None
            self._commit_step(v)
        else:
            self._commit_step(self._slider.value())

    def _on_drag_timeout(self) -> None:
        if self._pending_value is None:
            return
        v = self._pending_value
        self._pending_value = None
        self._commit_step(v)

    def _commit_step(self, step: int) -> None:
        n = max(0, int(self._director.n_steps))
        clamped = max(0, min(step, max(0, n - 1)))
        self._director.set_step(clamped)
        # Director will fire ``on_step_changed`` -> _on_director_step
        # which updates the labels.

    # ------------------------------------------------------------------
    # Transport buttons
    # ------------------------------------------------------------------

    def _step_delta(self, delta: int) -> None:
        self._commit_step(self._director.step_index + delta)

    def _jump_first(self) -> None:
        self._commit_step(0)

    def _jump_last(self) -> None:
        n = max(0, int(self._director.n_steps))
        self._commit_step(max(0, n - 1))

    # ------------------------------------------------------------------
    # Animation transport
    # ------------------------------------------------------------------

    def _toggle_play(self, on: bool) -> None:
        if on:
            n = max(0, int(self._director.n_steps))
            if n <= 1:
                # Nothing to animate — release the button.
                self._btn_play.setChecked(False)
                return
            # If we're already at the boundary the user would expect to
            # see, reset to step 0 (forward direction) so Play actually
            # plays. Without this, hitting Play at the last step in
            # ``once`` mode would immediately stop.
            mode = self._loop_mode()
            if mode == "once" and self._director.step_index >= n - 1:
                self._director.set_step(0)
            self._anim_direction = +1
            self._anim_timer.setInterval(self._interval_ms())
            self._anim_timer.start()
        else:
            self._anim_timer.stop()

    def _on_animation_tick(self) -> None:
        n = max(0, int(self._director.n_steps))
        if n <= 1:
            self._stop_animation()
            return
        last = n - 1
        cur = int(self._director.step_index)
        mode = self._loop_mode()

        if mode == "bounce":
            nxt = cur + self._anim_direction
            if nxt > last:
                self._anim_direction = -1
                nxt = last - 1 if last >= 1 else 0
            elif nxt < 0:
                self._anim_direction = +1
                nxt = 1 if last >= 1 else 0
        else:
            nxt = cur + 1
            if nxt > last:
                if mode == "loop":
                    nxt = 0
                else:    # once
                    self._director.set_step(last)
                    self._stop_animation()
                    return

        self._director.set_step(nxt)

    def _stop_animation(self) -> None:
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        # Releasing the button retriggers ``_toggle_play(False)``,
        # which calls ``stop()`` again — harmless.
        if self._btn_play.isChecked():
            self._btn_play.blockSignals(True)
            try:
                self._btn_play.setChecked(False)
            finally:
                self._btn_play.blockSignals(False)

    def _loop_mode(self) -> str:
        mode = self._loop_combo.currentData()
        return mode if mode in self.LOOP_MODES else "once"

    def _interval_ms(self) -> int:
        fps = max(self.FPS_MIN, min(self.FPS_MAX, int(self._fps_spin.value())))
        return max(1, int(round(1000.0 / fps)))

    def _on_fps_changed(self, _value: int) -> None:
        # Live update — if the timer is running, switch interval
        # without disturbing the running state.
        if self._anim_timer.isActive():
            self._anim_timer.setInterval(self._interval_ms())

    # ------------------------------------------------------------------
    # Director callback
    # ------------------------------------------------------------------

    def _on_director_stage(self, _stage_id: str) -> None:
        # Stop any running animation — the new stage may have a
        # different step count and the user almost certainly wants
        # to look at it before resuming.
        self._stop_animation()
        self.refresh()

    def _on_director_step(self, step: int) -> None:
        # Programmatic step changes (jump buttons, etc.) — keep slider
        # in sync without re-firing.
        n = max(0, int(self._director.n_steps))
        self._suppress_observer = True
        try:
            self._slider.blockSignals(True)
            self._slider.setValue(int(step))
        finally:
            self._slider.blockSignals(False)
            self._suppress_observer = False
        self._update_labels(step, n)

    # ------------------------------------------------------------------
    # Label rendering
    # ------------------------------------------------------------------

    def _update_labels(self, step: int, n_steps: int) -> None:
        self._step_label.setText(f"{step} / {max(0, n_steps - 1)}")
        t = self._director.current_time()
        if t is None:
            self._time_label.setText("t = —")
        else:
            self._time_label.setText(f"t = {t:.4g}")
