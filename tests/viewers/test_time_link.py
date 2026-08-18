"""ADR 0098 §7 (S4-3) — the session time link, offscreen lane.

Criteria (continuing S4-2's numbering):

36. The scrubber writes the session instant, and every pane follows —
    §7's own oracle read forwards: "drag the scrubber → the cursor
    rides the curve".
37. The track is ONE STAGE at a time (plan decision 9): the selector
    lists non-mode stages, and switching lands on that stage's step 0
    rather than carrying a step that means a different time.
38. Unlinked, the scrubber drives nothing and says so; the per-pane
    badge takes over.
39. The badge is CHROME: with a long stage id it must not push the
    pane frame past the A1.4 floor.
40. A mode-posed view shows its frozen state whether linked or not
    (§4/§7: a mode pose has no instant).
41. The inspector's pane-time row writes ``view.time``, and the link
    ignores it while it is on (§9).
42. Playback realizes only the panes that MOVED — the reconciler's
    first sustained-load test. A mode-posed pane is frozen under
    the link (§7), so it is the pane that proves the gate.

Criterion 43 makes the mutation tests mandatory: a badge that is not
squeezable, a stage switch that carries its step, and a scrubber that
writes while unlinked must each make one criterion above FAIL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.session import Deform, Instant, PlotSeries, PlotSource
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import SessionPaneHost
from apeGmsh.viewers.session._scrubber import SessionScrubber
from apeGmsh.viewers.ui._layout_metrics import LAYOUT

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

T = 5
DT = 0.5
LONG_STAGE = "a_very_long_stage_identifier"
ROOMY = (1200, 800)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def timed_results(g, tmp_path: Path):
    """Two static stages with DIFFERENT magnitudes, and a deliberately
    long stage id so the badge's width is exercised for real."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
    g.physical.add_volume("a", name="A")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "timed.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        for stage_id, base in ((LONG_STAGE, 0.0), ("push", 1000.0)):
            sid = w.begin_stage(
                name=stage_id, kind="static", stage_id=stage_id,
                time=np.arange(T, dtype=np.float64) * DT,
            )
            w.write_nodes(
                sid, "partition_0", node_ids=ids,
                components={
                    "displacement_z": np.stack(
                        [ids * 1.0 + base + t for t in range(T)]
                    ),
                },
            )
            w.end_stage()
        # A MODE stage, so "the selector excludes modes" and "a mode
        # pose is frozen" are asserted against one that exists rather
        # than against its absence.
        sid = w.begin_stage(
            name="mode-1", kind="mode", stage_id="mode-1",
            time=np.zeros(1, dtype=np.float64),
            eigenvalue=4.0, frequency_hz=0.318, period_s=3.14,
            mode_index=1,
        )
        w.write_nodes(
            sid, "partition_0", node_ids=ids,
            components={"displacement_z": (ids * 1.0)[None, :]},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def rig(qapp, timed_results):
    """A host + a scrubber on one session, wired the way the window
    wires them."""
    made: list = []

    def factory(_parent):
        backend = RecordingBackend()
        made.append(backend)
        return None, backend

    session = timed_results.session()
    host = SessionPaneHost(
        session, backend_factory=factory, defer_fn=lambda fn: fn(),
    )
    host.resize(*ROOMY)
    host.show()
    qapp.processEvents()
    scrubber = SessionScrubber(session)
    qapp.processEvents()
    yield session, host, scrubber, made
    scrubber.dispose()
    host.dispose()
    host.setParent(None)


def _drag_to(scrubber, step: int) -> None:
    """One slider drag, committed the way a mouse release commits."""
    scrubber.slider.setValue(step)
    scrubber._on_slider_released()          # noqa: SLF001


def _playhead(pane) -> "float | None":
    lines = [
        line for line in pane.chart.axes.lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    return None if not lines else float(lines[-1].get_xdata()[0])


# =====================================================================
# 36-37 — the scrubber writes the instant, one stage at a time
# =====================================================================


def test_c36_the_scrubber_moves_every_pane(rig, qapp):
    """Criterion 36 — §7 forwards. One drag, and the plot's playhead
    sits on that step's recorded TIME, not on its index."""
    session, host, scrubber, _made = rig
    node_id = int(session.results.fem.nodes.ids[0])
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    qapp.processEvents()

    _drag_to(scrubber, 3)
    qapp.processEvents()

    assert session.time == Instant(LONG_STAGE, 3)
    assert _playhead(host.frame(plot.id).pane) == pytest.approx(3 * DT)
    # ...and the mesh pane realized at that instant too.
    view = session.panes[0]
    assert session.effective_instant(view) == Instant(LONG_STAGE, 3)


def test_c37_the_selector_lists_non_mode_stages(rig):
    """Criterion 37. A mode pose has no instant (§4/§7), so a mode
    stage on the time track would offer a position the link cannot
    take."""
    _session, _host, scrubber, _made = rig
    offered = [
        scrubber.stage_box.itemData(i)
        for i in range(scrubber.stage_box.count())
    ]
    assert offered == [LONG_STAGE, "push"]
    assert scrubber.slider.maximum() == T - 1


def test_c37_switching_stage_lands_on_step_zero(rig, qapp):
    """Criterion 37, the other half. Carrying the step across would
    silently mean a different time — and on a shorter stage it would
    not exist at all."""
    session, _host, scrubber, _made = rig
    _drag_to(scrubber, 4)
    qapp.processEvents()
    assert session.time == Instant(LONG_STAGE, 4)

    scrubber.stage_box.setCurrentIndex(1)
    qapp.processEvents()
    assert session.time == Instant("push", 0)


# =====================================================================
# 38-40 — unlinked, and the per-pane badge
# =====================================================================


def test_c38_unlinked_the_scrubber_drives_nothing_and_says_so(rig, qapp):
    """Criterion 38. §7 gives each pane its own instant when the link
    is off, so a live slider here would be a control that changes no
    picture (0087 INV-2)."""
    session, _host, scrubber, _made = rig
    assert scrubber.slider.isEnabled()

    scrubber.link_button.setChecked(False)
    qapp.processEvents()

    assert session.time_linked is False
    assert not scrubber.slider.isEnabled()
    assert "link is off" in scrubber.slider.toolTip()
    assert scrubber.link_button.text() == "Unlinked"


def test_c38_the_badge_appears_only_when_it_is_news(rig, qapp):
    """Criterion 38/39. Linked, every pane sits on the instant the
    scrubber already shows — N copies of one number. The badge earns
    its space only when this pane's instant can differ."""
    session, host, scrubber, _made = rig
    frame = host.pane_frames[0]
    assert not frame._time_badge.isVisible()          # noqa: SLF001

    session.time_linked = False
    session.panes[0].time = Instant("push", 2)
    qapp.processEvents()

    badge = frame._time_badge                          # noqa: SLF001
    assert badge.isVisible()
    assert "2" in badge.text()
    assert "push · step 2" in badge.toolTip()


def test_c39_the_badge_never_breaches_the_pane_floor(rig, qapp):
    """Criterion 39 — trap: the badge is CHROME.

    ``required_extent`` / ``can_add`` multiply by
    ``LAYOUT.pane_min_width`` and ``QSplitter`` will not shrink a child
    below its ``minimumSizeHint``, so a badge that refuses to shrink
    makes the Add gate admit a column the splitter cannot fit.
    Measured before the fix: a long stage id took this frame's floor
    from 207 px to 593.
    """
    session, host, scrubber, _made = rig
    session.panes[0].name = "A deliberately long mesh view name"
    session.time_linked = False
    session.panes[0].time = Instant(LONG_STAGE, T - 1)
    qapp.processEvents()

    frame = host.pane_frames[0]
    assert frame._time_badge.isVisible()               # noqa: SLF001
    assert frame.minimumSizeHint().width() <= LAYOUT.pane_min_width
    # The full value has to survive somewhere legible.
    assert LONG_STAGE in frame._time_badge.toolTip()   # noqa: SLF001


def test_c40_a_mode_posed_view_shows_it_frozen(rig, qapp):
    """Criterion 40. A mode pose has NO instant, linked or not (§4/§7)
    — the one case where the scrubber moves and this pane does not,
    which without a badge reads as a broken pane."""
    session, host, scrubber, _made = rig
    view = session.panes[0]
    view.deform = Deform(field="displacement", mode=1)
    qapp.processEvents()

    badge = host.pane_frames[0]._time_badge            # noqa: SLF001
    assert session.time_linked is True     # frozen even while LINKED
    assert badge.isVisible()
    assert "mode" in badge.text()
    assert session.effective_instant(view) is None


# =====================================================================
# 41 — the inspector's pane time (§9)
# =====================================================================


def test_c41_the_inspector_sets_pane_time(rig, qapp):
    """Criterion 41. §9: "pane time — set it here; the link ignores
    it." So the control writes whenever, and only the note changes
    with the link: the value IS the instant this pane takes the moment
    the link comes off."""
    from apeGmsh.viewers.session import MeshInspectorPage

    session, _host, _scrubber, _made = rig
    view = session.panes[0]
    page = MeshInspectorPage(session, view)
    try:
        page._time_stage.setCurrentIndex(                # noqa: SLF001
            page._time_stage.findData("push"),           # noqa: SLF001
        )
        qapp.processEvents()
        assert view.time == Instant("push", 0)

        page._time_step.setValue(3)                      # noqa: SLF001
        qapp.processEvents()
        assert view.time == Instant("push", 3)
        # The link is on, so the picture still follows the session.
        assert "link is on" in page._time_note.text()    # noqa: SLF001
        assert session.effective_instant(view) == session.time

        session.time_linked = False
        qapp.processEvents()
        assert session.effective_instant(view) == Instant("push", 3)

        page._on_pane_time_clear()                       # noqa: SLF001
        assert view.time is None
    finally:
        page.dispose()


# =====================================================================
# 42 — playback under sustained load
# =====================================================================


def test_c42_playback_realizes_only_the_panes_that_moved(
    rig, qapp, monkeypatch,
):
    """Criterion 42. Animation is the reconciler's first SUSTAINED
    load: every frame writes the session, and the tick is
    session-wide, so without the criterion-12 signature gate each
    frame would cost one realize per pane in the SESSION rather than
    one per pane that actually moved.

    The discriminating pane is a MODE-POSED one. §7 freezes it under
    the link — ``effective_instant`` is ``None`` for it whatever the
    scrubber does — so a frame that costs the static pane a realize
    must cost the mode pane nothing. Counting only panes that DO move
    would pass with the gate ripped out (every tick changes every one
    of them), which is exactly what the mutation pass caught.
    """
    import apeGmsh.viewers.session._reconciler as reconciler_mod

    session, host, scrubber, _made = rig
    static_view = session.panes[0]
    frozen_view = session.add_view()
    frozen_view.deform = Deform(field="displacement", mode=1)
    qapp.processEvents()
    assert session.effective_instant(frozen_view) is None

    real = reconciler_mod.realize_pane
    calls: list = []
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    # The session boots with time=None, so every commit below moves it.
    assert session.time is None
    frames = 4
    for step in range(frames):
        scrubber._commit(step)               # noqa: SLF001 — one tick
        qapp.processEvents()

    assert calls.count(static_view.id) == frames
    assert calls.count(frozen_view.id) == 0, (
        f"the frozen pane realized {calls.count(frozen_view.id)} time(s) "
        f"for a scrubber it does not follow"
    )

    # And a re-commit of the SAME instant costs nothing anywhere —
    # what keeps a 30 fps drag from realizing 30 times a second per
    # pane for pictures that did not move.
    calls.clear()
    scrubber._commit(frames - 1)             # noqa: SLF001
    qapp.processEvents()
    assert calls == []


def test_c42_dispose_stops_a_running_animation(rig, qapp):
    """A QTimer left running on a closed window keeps writing the
    session — a repaint per tick into panes being torn down."""
    _session, _host, scrubber, _made = rig
    scrubber.play_button.setChecked(True)
    assert scrubber.is_playing

    scrubber.dispose()

    assert not scrubber.is_playing


# =====================================================================
# 43 — mutation tests. Each names the criterion it must break.
# =====================================================================


def test_mutation_an_unsqueezable_badge_breaks_c39(rig, qapp):
    """Criterion 39, mutated: give the badge back its natural minimum
    and the long stage id blows the A1.4 floor."""
    from qtpy import QtWidgets

    session, host, _scrubber, _made = rig
    session.time_linked = False
    session.panes[0].time = Instant(LONG_STAGE, T - 1)
    qapp.processEvents()

    badge = host.pane_frames[0]._time_badge             # noqa: SLF001
    badge.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    badge.setText(LONG_STAGE + " · 4")                  # unelided
    qapp.processEvents()

    assert host.pane_frames[0].minimumSizeHint().width() > (
        LAYOUT.pane_min_width
    ), "the unsqueezable badge fitted anyway — the test is not "\
       "measuring the floor"


def test_mutation_a_stage_switch_that_carries_its_step_breaks_c37(
    rig, qapp, monkeypatch,
):
    """Criterion 37, mutated: land on the CURRENT step instead of 0."""
    session, _host, scrubber, _made = rig
    _drag_to(scrubber, 4)
    qapp.processEvents()

    def carry(_index: int) -> None:
        stage_id = scrubber.stage_box.currentData()
        if stage_id is None or scrubber._suppress_observer:  # noqa: SLF001
            return
        session.time = Instant(str(stage_id), 4)
    monkeypatch.setattr(scrubber, "_on_stage_changed", carry)
    scrubber.stage_box.currentIndexChanged.connect(carry)

    scrubber.stage_box.setCurrentIndex(1)
    qapp.processEvents()
    assert session.time != Instant("push", 0), (
        "the mutation did not carry the step — the test is not "
        "measuring the landing rule"
    )
