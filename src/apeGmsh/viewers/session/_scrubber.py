"""SessionScrubber — the §7 time link's one control (S4-3).

ADR 0098 §7. An **instant** is a ``(stage, step)`` pair, and while the
link is on it is THE instant of every pane: the scrubber, every mesh
view, every plot cursor. "Drag the scrubber → the cursor rides the
curve. Drag the curve → the meshes move. If the link is on and moving
the plot does not move the meshes, the link is a lie."

Adapted from :class:`~..ui._time_scrubber.TimeScrubberDock` — the
transport, the drag coalescing, the fps/loop machinery and the
animation state machine are that widget's, and they were already
director-agnostic. What changes is the binding, exactly as the plan
sized it: ``director.n_steps`` / ``step_index`` / ``set_step`` /
``subscribe_step`` become ``session.time`` and the session's change
tick. The old dock is untouched until S6a.

**One stage at a time** (plan decision 9, settled here). The slider
spans the CURRENT stage's recorded steps and a selector picks the
stage; the director's concatenated multi-stage track is deferred. The
reason is that the scrubber's x-axis is then the same axis every plot
pane draws — the stage's own recorded time — so "the cursor rides the
curve" is literally true rather than true-within-a-segment, and a
history plot (which resolves inside ``results.stage(cursor.stage)``)
swaps curves on an explicit stage change instead of mid-drag. It is
not a one-way door: ``Instant`` is ``(stage, step)`` under either
traversal, so the IR and the S5 snapshot are unaffected.

Mode stages are absent from the selector: a mode pose has no instant
(§4/§7), and ``effective_instant`` returns ``None`` for a mode-posed
view whether the link is on or off.

Reentrancy is the hazard this widget lives with (the plan's own
caution): the session ticks on every write, the tick refreshes this
widget, and a refresh that moved a control would write again. Every
projection therefore runs under ``_suppress_observer`` AND
``blockSignals``, and every write goes through one funnel
(:meth:`_commit`) that no projection path can reach.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .._failures import safe_slot
from ..ui._layout_metrics import LAYOUT


def _qt():
    from qtpy import QtCore, QtWidgets
    return QtWidgets, QtCore


#: What the link toggle says in each state — the §7 law in two lines,
#: because "linked" alone does not tell you what it links.
_LINK_ON = (
    "Time link ON — one instant for the scrubber, every mesh view and "
    "every plot cursor. Click to unlink."
)
_LINK_OFF = (
    "Time link OFF — each pane keeps its own instant, set in the "
    "inspector. Click to link them to this scrubber."
)
_UNLINKED_SLIDER = (
    "The time link is off, so this scrubber drives nothing: each pane "
    "keeps its own instant (set a pane's time in the inspector). Turn "
    "the link on to drive them all from here."
)


class SessionScrubber:
    """Bottom-of-window transport over one session's instant."""

    DRAG_COALESCE_MS = 33    # ~30 fps during drag (the dock's number)
    DEFAULT_FPS = 30
    FPS_MIN = 1
    FPS_MAX = 60
    LOOP_MODES = ("once", "loop", "bounce")

    def __init__(
        self,
        session: Any,
        *,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._session = session
        self._suppress_observer = False
        self._pending_value: Optional[int] = None
        self._anim_direction = +1

        self._drag_timer = QtCore.QTimer()
        self._drag_timer.setSingleShot(True)
        self._drag_timer.setInterval(self.DRAG_COALESCE_MS)
        self._drag_timer.timeout.connect(safe_slot(self._on_drag_timeout))

        self._anim_timer = QtCore.QTimer()
        self._anim_timer.setSingleShot(False)
        self._anim_timer.timeout.connect(safe_slot(self._on_animation_tick))

        widget = QtWidgets.QWidget()
        widget.setObjectName("SessionScrubber")
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        from ..ui._icon_factory import bind_button_glyph

        def button(glyph: str, tip: str, slot) -> Any:
            b = QtWidgets.QToolButton()
            bind_button_glyph(b, glyph)
            b.setToolTip(tip)
            b.clicked.connect(safe_slot(slot))
            row.addWidget(b)
            return b

        self._btn_first = button(
            "skip_first", "Jump to first step", lambda: self._commit(0),
        )
        self._btn_back = button(
            "step_back", "Step backward", lambda: self._step_delta(-1),
        )
        self._btn_play = QtWidgets.QToolButton()
        self._btn_play.setObjectName("SessionScrubberPlay")
        self._btn_play.setCheckable(True)
        bind_button_glyph(self._btn_play, "play")
        self._btn_play.setToolTip("Play / pause the recorded history")
        self._btn_play.toggled.connect(safe_slot(self._toggle_play))
        row.addWidget(self._btn_play)
        self._btn_fwd = button(
            "step_forward", "Step forward", lambda: self._step_delta(+1),
        )
        self._btn_last = button(
            "skip_last", "Jump to last step", self._jump_last,
        )

        # The §7 link. It sits with the transport and not in a menu
        # because it changes what every OTHER control on screen means.
        self._btn_link = QtWidgets.QToolButton()
        self._btn_link.setObjectName("SessionScrubberLink")
        self._btn_link.setCheckable(True)
        # A word, not a glyph: the 0087 roster has no link silhouette,
        # and a 16 px chain is not distinct from the section / probe
        # icons it would sit beside (INV-4 forbids near-duplicates).
        # This row already carries two non-glyph controls anyway.
        self._btn_link.setText("Linked")
        self._btn_link.toggled.connect(safe_slot(self._on_link_toggled))
        row.addWidget(self._btn_link)

        self._stages = QtWidgets.QComboBox()
        self._stages.setObjectName("SessionScrubberStage")
        self._stages.setToolTip(
            "Which recorded stage the slider spans. One stage at a "
            "time: the slider's axis is this stage's own time, the "
            "same axis a plot pane draws."
        )
        self._stages.currentIndexChanged.connect(
            safe_slot(self._on_stage_changed),
        )
        row.addWidget(self._stages)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setObjectName("SessionScrubberSlider")
        self._slider.setMinimum(0)
        self._slider.valueChanged.connect(safe_slot(self._on_slider_changed))
        self._slider.sliderReleased.connect(
            safe_slot(self._on_slider_released),
        )
        self._slider.setMinimumWidth(LAYOUT.scrubber_slider_min_width)
        row.addWidget(self._slider, stretch=1)

        self._step_label = QtWidgets.QLabel()
        self._step_label.setObjectName("SessionScrubberStep")
        self._step_label.setMinimumWidth(LAYOUT.scrubber_step_label_min_width)
        row.addWidget(self._step_label)

        self._time_label = QtWidgets.QLabel()
        self._time_label.setObjectName("SessionScrubberTime")
        self._time_label.setMinimumWidth(LAYOUT.scrubber_time_label_min_width)
        row.addWidget(self._time_label)

        self._fps = QtWidgets.QSpinBox()
        self._fps.setObjectName("SessionScrubberFps")
        self._fps.setRange(self.FPS_MIN, self.FPS_MAX)
        self._fps.setValue(self.DEFAULT_FPS)
        self._fps.setToolTip("Playback frames per second")
        self._fps.valueChanged.connect(safe_slot(self._on_fps_changed))
        row.addWidget(self._fps)

        self._loop = QtWidgets.QComboBox()
        self._loop.setObjectName("SessionScrubberLoop")
        self._loop.addItems(["Once", "Loop", "Bounce"])
        self._loop.setToolTip("What playback does at the last step")
        row.addWidget(self._loop)

        self._widget = widget
        session.subscribe(self._on_session_tick)
        self.refresh()

    # -- surface -------------------------------------------------------

    @property
    def widget(self) -> Any:
        return self._widget

    @property
    def slider(self) -> Any:
        return self._slider

    @property
    def play_button(self) -> Any:
        return self._btn_play

    @property
    def link_button(self) -> Any:
        return self._btn_link

    @property
    def stage_box(self) -> Any:
        return self._stages

    @property
    def is_playing(self) -> bool:
        return bool(self._anim_timer.isActive())

    def dispose(self) -> None:
        """Stop the timers and detach from the session (idempotent).

        The animation timer is the one that matters: a QTimer left
        running on a closed window keeps writing the session, which is
        a repaint per tick into panes that are being torn down.
        """
        self._anim_timer.stop()
        self._drag_timer.stop()
        try:
            self._session.unsubscribe(self._on_session_tick)
        except Exception:
            pass

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Re-read the session and update every control (never writes).

        The whole body runs suppressed: this is the reentrancy hazard
        the plan named — the session ticks on every write, the tick
        lands here, and a control moved here would write again.
        """
        stages = self._stage_ids()
        stage_id, step = self._current()
        n = self._n_steps(stage_id)
        linked = bool(self._session.time_linked)

        self._suppress_observer = True
        try:
            if [self._stages.itemData(i)
                    for i in range(self._stages.count())] != stages:
                self._stages.blockSignals(True)
                try:
                    self._stages.clear()
                    for sid in stages:
                        self._stages.addItem(sid, sid)
                finally:
                    self._stages.blockSignals(False)
            if stage_id is not None:
                index = self._stages.findData(stage_id)
                if index >= 0 and index != self._stages.currentIndex():
                    self._stages.blockSignals(True)
                    try:
                        self._stages.setCurrentIndex(index)
                    finally:
                        self._stages.blockSignals(False)
            self._slider.blockSignals(True)
            try:
                self._slider.setMaximum(max(0, n - 1))
                self._slider.setValue(max(0, min(step, max(0, n - 1))))
            finally:
                self._slider.blockSignals(False)
            self._btn_link.blockSignals(True)
            try:
                self._btn_link.setChecked(linked)
            finally:
                self._btn_link.blockSignals(False)
        finally:
            self._suppress_observer = False

        self._btn_link.setText("Linked" if linked else "Unlinked")
        self._btn_link.setToolTip(_LINK_ON if linked else _LINK_OFF)
        self._update_labels(stage_id, step, n)
        # 0087 INV-2 — with the link off this scrubber drives nothing
        # (§7: each pane keeps its own instant), so it says so instead
        # of moving a slider that changes no picture.
        live = linked and n > 1
        for control in (
            self._btn_first, self._btn_back, self._btn_play,
            self._btn_fwd, self._btn_last, self._slider, self._stages,
            self._fps, self._loop,
        ):
            control.setEnabled(live)
        self._slider.setToolTip("" if linked else _UNLINKED_SLIDER)
        if not live and self.is_playing:
            self._stop_animation()

    # -- session reads -------------------------------------------------

    def _stage_ids(self) -> "list[str]":
        """Every stage an instant can name, in recorded order.

        Mode stages are excluded: a mode pose has NO instant (§4/§7),
        so naming one on the time track would offer a position the
        link cannot take.
        """
        results = self._session.results
        if results is None:
            return []
        try:
            return [
                str(s.id) for s in results.stages
                if getattr(s, "kind", None) != "mode"
            ]
        except Exception:
            return []

    def _n_steps(self, stage_id: "Optional[str]") -> int:
        if stage_id is None or self._session.results is None:
            return 0
        try:
            return max(0, int(self._session.results.stage(stage_id).n_steps))
        except Exception:
            return 0

    def _current(self) -> "tuple[Optional[str], int]":
        """The instant the scrubber sits on — the session's, or the
        first stage's step 0 before anything set one."""
        instant = self._session.time
        if instant is not None:
            return instant.stage, int(instant.step)
        stages = self._stage_ids()
        return (stages[0] if stages else None), 0

    def _time_value(self, stage_id: str, step: int) -> "Optional[float]":
        try:
            import numpy as np

            times = np.asarray(self._session.results.stage(stage_id).time)
            if 0 <= step < times.size:
                return float(times[step])
        except Exception:
            pass
        return None

    def _update_labels(
        self, stage_id: "Optional[str]", step: int, n: int,
    ) -> None:
        if stage_id is None or n <= 0:
            self._step_label.setText("no steps")
            self._time_label.setText("")
            return
        self._step_label.setText(f"Step {step} / {n - 1}")
        value = self._time_value(stage_id, step)
        self._time_label.setText("" if value is None else f"t = {value:g}")

    # -- the ONE write funnel ------------------------------------------

    def _commit(self, step: int) -> None:
        """Write the instant. The only place this widget writes.

        Everything else — transport, drag, animation — resolves to a
        step and comes through here, so there is exactly one place the
        clamp and the §7 law live.
        """
        from apeGmsh.results.session import Instant

        stage_id, _current = self._current()
        if stage_id is None:
            return
        n = self._n_steps(stage_id)
        if n <= 0:
            return
        clamped = max(0, min(int(step), n - 1))
        instant = Instant(stage_id, clamped)
        if instant == self._session.time:
            return
        self._session.time = instant   # ticks → refresh() projects back

    def _step_delta(self, delta: int) -> None:
        _stage, step = self._current()
        self._commit(step + delta)

    def _jump_last(self) -> None:
        stage_id, _step = self._current()
        self._commit(max(0, self._n_steps(stage_id) - 1))

    # -- slots ---------------------------------------------------------

    def _on_session_tick(self) -> None:
        self.refresh()

    def _on_slider_changed(self, value: int) -> None:
        if self._suppress_observer:
            return
        # Coalesce the drag: the most recent value wins, and the
        # session sees one write per ~33 ms instead of one per pixel.
        self._pending_value = int(value)
        if not self._drag_timer.isActive():
            self._drag_timer.start()

    def _on_slider_released(self) -> None:
        if self._drag_timer.isActive():
            self._drag_timer.stop()
        value, self._pending_value = self._pending_value, None
        self._commit(self._slider.value() if value is None else value)

    def _on_drag_timeout(self) -> None:
        value, self._pending_value = self._pending_value, None
        if value is not None:
            self._commit(value)

    def _on_stage_changed(self, _index: int) -> None:
        """Another stage — land on ITS step 0.

        Carrying the step across would silently mean a different time,
        and on a shorter stage it would not exist at all.
        """
        if self._suppress_observer:
            return
        stage_id = self._stages.currentData()
        if stage_id is None:
            return
        from apeGmsh.results.session import Instant

        if self._n_steps(str(stage_id)) <= 0:
            return
        self._stop_animation()
        self._session.time = Instant(str(stage_id), 0)

    def _on_link_toggled(self, checked: bool) -> None:
        if self._suppress_observer:
            return
        self._session.time_linked = bool(checked)

    def _on_fps_changed(self, _value: int) -> None:
        if self._anim_timer.isActive():
            self._anim_timer.setInterval(self._interval_ms())

    # -- animation -----------------------------------------------------

    def _interval_ms(self) -> int:
        return max(1, int(1000 / max(1, int(self._fps.value()))))

    def _loop_mode(self) -> str:
        return self.LOOP_MODES[max(0, self._loop.currentIndex())]

    def _toggle_play(self, on: bool) -> None:
        from ..ui._icon_factory import bind_button_glyph

        bind_button_glyph(self._btn_play, "pause" if on else "play")
        if not on:
            self._anim_timer.stop()
            return
        stage_id, step = self._current()
        n = self._n_steps(stage_id)
        if n <= 1:
            self._btn_play.setChecked(False)
            return
        # Play at the end in ``once`` mode would stop immediately;
        # rewind so the button does what it says.
        if self._loop_mode() == "once" and step >= n - 1:
            self._commit(0)
        self._anim_direction = +1
        self._anim_timer.setInterval(self._interval_ms())
        self._anim_timer.start()

    def _on_animation_tick(self) -> None:
        stage_id, cur = self._current()
        n = self._n_steps(stage_id)
        if n <= 1:
            self._stop_animation()
            return
        last = n - 1
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
                else:
                    self._commit(last)
                    self._stop_animation()
                    return
        self._commit(nxt)

    def _stop_animation(self) -> None:
        from ..ui._icon_factory import bind_button_glyph

        self._anim_direction = +1
        self._anim_timer.stop()
        if self._btn_play.isChecked():
            self._btn_play.blockSignals(True)
            try:
                self._btn_play.setChecked(False)
            finally:
                self._btn_play.blockSignals(False)
        bind_button_glyph(self._btn_play, "play")


__all__ = ["SessionScrubber"]
