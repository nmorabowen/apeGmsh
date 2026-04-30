"""Time scrubber dock — bottom-of-window step slider + transport controls.

Single-row layout::

    [<<] [<] [|>] [>] [>>]   step ▮━━━━━━○━━━━━━ [ 42 / 100 ]   t=1.234e+00 s

* ``<<`` / ``>>`` jump to first / last step
* ``<`` / ``>`` step -1 / +1
* ``|>`` toggles animation (Phase 6 — disabled in Phase 0)
* The slider is QTimer-throttled to ~33 ms during drag so step changes
  don't fire faster than the renderer can keep up. The final position
  on release fires a non-throttled update so the rendered step matches
  the slider exactly.

Phase 0 wires the step slider; the play / animation buttons are
present but disabled until Phase 6 lands the animation mode.
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
        self._btn_play.setToolTip("Play / pause animation (Phase 6)")
        self._btn_play.setEnabled(False)
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

        self._widget = widget

        # Subscribe to director events
        director.subscribe_step(self._on_director_step)
        director.subscribe_stage(lambda _id: self.refresh())

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
        for b in (self._btn_first, self._btn_back,
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

    def _toggle_play(self, on: bool) -> None:
        # Phase 6 wires this; Phase 0 keeps the button disabled but
        # provides the slot for forward compatibility.
        if on:
            self._btn_play.setChecked(False)

    # ------------------------------------------------------------------
    # Director callback
    # ------------------------------------------------------------------

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
