"""LayerThicknessPanel dispatcher migration (UI storm follow-up #6).

Each step change re-reads the through-thickness profile at the picked
shell GP and redraws the matplotlib line. Without coalesce a rapid
scrubber drag fires one read + redraw per slider tick. The
``attach_dispatcher`` migration collapses the storm to one redraw per
Qt tick.

Same pattern as :class:`OutlineTree` / :class:`DiagramSettingsTab` /
:class:`PickReadoutHUD` / :class:`TimeHistoryPanel` /
:class:`FiberSectionPanel` — bench file (``@pytest.mark.bench``);
two correctness assertions.
"""
from __future__ import annotations

import os

import pytest


# Force offscreen Qt before importing qtpy.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _DiagramStub:
    """Minimal LayerStackDiagram shape for LayerThicknessPanel.

    ``is_attached=False`` short-circuits ``_populate_gp_picker`` and
    ``_fallback_pick`` so ``refresh()`` lands in the empty-state path
    without touching matplotlib drawing primitives — we only care
    about how many times the panel's refresh is invoked, not what
    the canvas paints.
    """

    is_attached = False

    class _Spec:
        class _Selector:
            component = "stress_xx"
        selector = _Selector()

    spec = _Spec()


class _DirectorStub:
    """Director shape for LayerThicknessPanel: step / picked-gp /
    diagrams subs + a couple of properties the panel reads."""

    def __init__(self):
        self._on_step = []
        self._on_picked = []
        self._on_diagrams = []
        self.picked_gp = None
        self.step_index = 0

    def subscribe_step(self, cb):
        self._on_step.append(cb)
        return lambda: (
            self._on_step.remove(cb) if cb in self._on_step else None
        )

    def subscribe_picked_gp(self, cb):
        self._on_picked.append(cb)
        return lambda: (
            self._on_picked.remove(cb) if cb in self._on_picked else None
        )

    def subscribe_diagrams(self, cb):
        self._on_diagrams.append(cb)
        return lambda: (
            self._on_diagrams.remove(cb) if cb in self._on_diagrams else None
        )

    def set_picked_gp(self, picked):
        self.picked_gp = picked


@pytest.fixture
def panel_and_director():
    """Build LayerThicknessPanel + stub director with explicit teardown.

    The matplotlib FigureCanvas inside the panel leaves Qt event-loop
    state that desynchronises later tests' QTimer-driven animations
    if the figure / canvas / widget aren't torn down. See the
    matching fixture in ``test_time_history_perf.py`` /
    ``test_section_panel_perf.py``.
    """
    from qtpy import QtWidgets
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    director = _DirectorStub()
    diagram = _DiagramStub()
    from apeGmsh.viewers.ui._thickness_panel import LayerThicknessPanel
    panel = LayerThicknessPanel(diagram, director)
    yield panel, director
    try:
        panel.close()
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.close(panel._fig)
        panel._canvas.close()
        panel.widget.close()
        panel.widget.deleteLater()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
            app.processEvents()
    except Exception:
        pass


def test_attach_dispatcher_swaps_step_subscription(panel_and_director) -> None:
    """After ``attach_dispatcher``, the panel's legacy step sub is
    gone from the Director's subscriber list."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    panel, director = panel_and_director
    # Initial state: panel holds exactly one step subscriber.
    assert len(director._on_step) == 1, (
        "LayerThicknessPanel should hold one legacy step sub before "
        "attach_dispatcher"
    )

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

    # Legacy step sub is dropped; picked / diagrams subs stay intact
    # (they're not part of this migration).
    assert len(director._on_step) == 0, (
        "attach_dispatcher must drop the legacy subscribe_step wiring"
    )
    assert len(director._on_picked) == 1
    assert len(director._on_diagrams) == 1


@pytest.mark.bench
def test_rapid_scrubber_drag_collapses_to_one_refresh(panel_and_director) -> None:
    """100 STEP_CHANGED fires in one tick -> exactly one refresh()
    call after the dispatcher's UI flush."""
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

    # Patch refresh() to count callbacks. The dispatcher subscription
    # lambda resolves ``self.refresh`` at call time, so an instance-
    # attribute patch is honoured.
    callbacks = [0]
    original = panel.refresh.__func__

    def _counting_refresh():
        callbacks[0] += 1
        original(panel)

    panel.refresh = _counting_refresh    # type: ignore[method-assign]

    # 100 step fires in one tick.
    for step in range(100):
        dispatcher.fire(STEP_CHANGED, payload=None)

    # Pre-drain: callbacks queued but not fired.
    assert callbacks[0] == 0

    for fn in list(deferred):
        fn()

    print(
        f"\n[thickness_panel scrubber storm] 100 STEP_CHANGED fires -> "
        f"{callbacks[0]} refresh() calls after coalesce flush"
    )

    # Coalesce contract: same dedup key for all 100 -> exactly 1 call.
    assert callbacks[0] == 1, (
        f"Expected 1 refresh after coalesce; got {callbacks[0]} "
        f"for 100 STEP_CHANGED fires"
    )
