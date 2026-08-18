"""ADR 0098 §8/§9 + §6 (S4-2) — outline select-all + the plot pane.

The default lane: ``QT_QPA_PLATFORM=offscreen``. A chart needs no GL
at all, so nearly everything below runs here; the ``[qt]`` half
(a real window that actually shows a canvas, and the real interactor
teardown) lives in ``test_select_all_plot_qt.py``.

The criteria (continuing S4-1's numbering):

20. Outline "Select all nodes" writes the ONE store with exactly that
    category's nodes.
21. "Select all Gauss" writes ``(element_id, gp_index)`` pairs in the
    SAME encoding the pane's pick targets use — the outline and the
    highlight cannot disagree about which point is selected.
22. Select-all is the §8 FOURTH writer, not a new store: the XOR law
    applies to it and it is ONE undoable gesture.
23. Select-all does not touch scope. Scope is what is drawn,
    selection is what is plotted (§8).
24. It works with the glyph buttons OFF, and costs a pane with them
    off nothing (§8: "query-from-outline may fill the set with glyphs
    off").
25. A category that recorded no Gauss offers a DISABLED "Select all
    Gauss"; materials is disabled outright (0087 INV-2).
26. The menu's subject is the checked names when the clicked row is
    one of them, and the clicked row alone otherwise.
27. One select-all gesture is not quadratic in the set size — the
    store AND the two per-pane O(n) costs S4-1 rides on it.
28. The plot pane draws the resolved series, and those arrays equal
    what ``results.plot.history`` plots.
29. One gesture, one redraw — and the gate does NOT skip the redraw
    that matters (a series change repaints; a cursor move moves the
    playhead).
30. A cursor move costs no READ: the record is already on screen.
31. A selection write does NOT redraw a plot. Membership is copied at
    creation (§6) — the inverse of the mesh pane's rule.
32. A plot that cannot resolve says so ON the chart.
33. §8's own test: select → New plot, with no Python snippet.
34. A plot pane frame still fits the A1.4 floor, and dies with its
    session subscription.

Criterion 35 makes the mutation tests mandatory: dropping the plot
signature's cursor term, letting select-all write a second store, or
leaving the plot reconciler subscribed on dispose must each make one
criterion above FAIL. They are at the bottom, each naming the
criterion it breaks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.session import (
    Instant, MeshStyle, PlotSeries, PlotSource, Scope,
)
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.scene_ir import Substrate
from apeGmsh.viewers.session import SessionPaneHost, select_all
from apeGmsh.viewers.session import _chart as chart_mod
from apeGmsh.viewers.session import _realize as realize_mod
from apeGmsh.viewers.session._outline import SessionOutline

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"
T = 4
ROOMY = (1200, 800)


# =====================================================================
# Rigs
# =====================================================================


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def session_results(g, tmp_path: Path):
    """Two physical groups; TWO integration points per element so the
    ``gp_index`` half of a Gauss target is not trivially zero."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left")
    g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="right")
    g.physical.add_volume("left", name="Left")
    g.physical.add_volume("right", name="Right")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = np.concatenate(
        [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    )
    disp = np.stack([node_ids * 1.0 + t * 10.0 for t in range(T)])
    sxx = np.zeros((T, elem_ids.size, 2))
    for t in range(T):
        sxx[t, :, 0] = elem_ids + t
        sxx[t, :, 1] = elem_ids * 2.0 + t

    path = tmp_path / "select_all.h5"
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
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids,
            natural_coords=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


class BackendFleet:
    """One recording backend per MESH pane (a plot pane asks for none)."""

    def __init__(self) -> None:
        self.made: list[RecordingBackend] = []

    def __call__(self, _parent):
        backend = RecordingBackend()
        self.made.append(backend)
        return None, backend


@pytest.fixture
def fleet() -> BackendFleet:
    return BackendFleet()


@pytest.fixture
def host(qapp, session_results, fleet):
    session = session_results.session()
    widget = SessionPaneHost(
        session, backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    widget.resize(*ROOMY)
    widget.show()
    qapp.processEvents()
    yield session, widget
    widget.dispose()
    widget.setParent(None)


@pytest.fixture
def outline(host):
    """An outline bound to the host, wired the way the window wires it."""
    session, widget = host
    calls: dict = {"plot_quantity": []}
    tree = SessionOutline(
        session,
        on_pane_selected=widget.set_active,
        on_new_plot_from_selection=calls["plot_quantity"].append,
        can_add=widget.can_add,
    )
    tree.set_active_pane(widget.active_pane_id)
    yield session, widget, tree, calls
    tree.dispose()


def _menu_pick(monkeypatch, wanted: "str | None"):
    """Answer the next ``QMenu.exec_`` with the entry whose text starts
    with ``wanted`` (``None`` dismisses). Returns the list of
    ``(text, enabled)`` the menu offered, so a test can assert on what
    the user was shown as well as on what the choice did."""
    from qtpy import QtWidgets

    offered: list = []

    def fake_exec(self, *_args, **_kwargs):
        chosen = None
        for action in self.actions():
            offered.append((action.text(), action.isEnabled()))
            if wanted is not None and action.text().startswith(wanted):
                chosen = action
        return chosen

    monkeypatch.setattr(QtWidgets.QMenu, "exec_", fake_exec, raising=False)
    monkeypatch.setattr(QtWidgets.QMenu, "exec", fake_exec, raising=False)
    return offered


def _name_item(tree: SessionOutline, axis: str, name: str):
    return tree._name_items[axis][name]        # noqa: SLF001


def _right_click(tree: SessionOutline, item) -> None:
    """Fire the tree's context-menu slot over ``item``'s row."""
    rect = tree.widget.visualItemRect(item)
    tree._on_context_menu(rect.center())       # noqa: SLF001


def _pg_nodes(results, name: str) -> "set[int]":
    from apeGmsh.viewers.data import ViewerData

    from apeGmsh.viewers.session._scope import resolve_scope

    scoped = resolve_scope(
        Scope("physical_groups", (name,)), ViewerData.from_fem(results.fem),
    )
    return {int(i) for i in scoped.node_ids}


# =====================================================================
# 20-21 — what select-all writes
# =====================================================================


def test_c20_select_all_nodes_writes_that_groups_nodes(
    outline, qapp, monkeypatch,
):
    """Criterion 20."""
    session, _widget, tree, _calls = outline
    offered = _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))

    wanted = _pg_nodes(session.results, "Left")
    assert session.selection.kind == "nodes"
    assert set(session.selection.nodes) == wanted
    # The user was told how many, before choosing (0087 INV-2's other
    # half: a control that CAN act says what it will do).
    assert offered[0][0] == f"Select all nodes ({len(wanted)})"


def test_c21_select_all_gauss_matches_the_panes_own_targets(
    outline, qapp, monkeypatch,
):
    """Criterion 21 — the encoding, end to end.

    A ``(element_id, gp_index)`` the outline writes must name the same
    integration point the pane's pick targets do, or the outline would
    select points the highlight marks elsewhere.
    """
    session, widget, tree, _calls = outline
    view = session.panes[0]
    view.style = MeshStyle(mesh=True, outlines=True, nodes=False, gauss=True)
    view.scope = Scope("physical_groups", ("Left",))
    qapp.processEvents()

    _menu_pick(monkeypatch, "Select all Gauss")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))

    assert session.selection.kind == "gauss"
    targets = widget.frame(view.id).pane.reconciler.realized.targets.gauss
    assert targets is not None
    drawn = {
        (int(e), int(g))
        for e, g in zip(targets.element_ids, targets.gp_indices)
    }
    assert set(session.selection.gauss) == drawn
    # Two GPs per element in this rig, so gp_index is not trivially 0.
    assert {gp for _e, gp in session.selection.gauss} == {0, 1}


# =====================================================================
# 22 — the fourth writer, over the ONE store
# =====================================================================


def test_c22_select_all_is_one_gesture_on_the_one_store(
    outline, qapp, monkeypatch,
):
    """Criterion 22. ADR 0045's log records it as ONE undoable SET, and
    the §8 XOR law applies to it like any other writer."""
    session, _widget, tree, _calls = outline
    state = session.selection.state

    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))
    nodes = set(session.selection.nodes)
    assert nodes

    _menu_pick(monkeypatch, "Select all Gauss")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))
    # A write of the other kind REPLACES — not a heterogeneous union.
    assert session.selection.kind == "gauss"
    assert all(t.substrate is Substrate.RESULTS_TOPO for t in state.targets)

    assert state.undo()               # one gesture, not one op per node
    assert set(session.selection.nodes) == nodes


def test_c23_select_all_does_not_touch_scope(outline, qapp, monkeypatch):
    """Criterion 23. "Scope (what is drawn) and selection (what is
    plotted) are different knobs" (§8)."""
    session, _widget, tree, _calls = outline
    view = session.panes[0]
    view.scope = Scope("physical_groups", ("Right",))
    qapp.processEvents()

    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))

    assert view.scope == Scope("physical_groups", ("Right",))
    assert set(session.selection.nodes) == _pg_nodes(session.results, "Left")


def test_c24_select_all_works_and_costs_nothing_with_glyphs_off(
    outline, qapp, monkeypatch,
):
    """Criterion 24. §8: "Query-from-outline may fill the set with
    glyphs off; glyphs on is how you see it." So the set fills, and a
    pane that cannot show it does not repaint for it."""
    session, widget, tree, _calls = outline
    view = session.panes[0]
    view.style = MeshStyle(
        mesh=True, outlines=True, nodes=False, gauss=False,
    )
    qapp.processEvents()

    # The reconciler binds ``realize_pane`` at import, so ITS module
    # global is the one a flush actually calls.
    import apeGmsh.viewers.session._reconciler as reconciler_mod

    realizes: list = []
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *a, **kw: (realizes.append(1), _REAL_REALIZE(*a, **kw))[1],
    )

    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))
    qapp.processEvents()

    assert session.selection.nodes, "the set did not fill with glyphs off"
    assert realizes == [], "a pane with no glyphs repainted for a select-all"


_REAL_REALIZE = realize_mod.realize_pane


# =====================================================================
# 25-26 — what the menu offers
# =====================================================================


def test_c25_gauss_is_disabled_where_none_was_recorded(
    outline, qapp, monkeypatch, session_results,
):
    """Criterion 25. A stage with no Gauss composite has no integration
    points to select, so the entry cannot act and says so."""
    session, _widget, tree, _calls = outline
    monkeypatch.setattr(
        type(session.results.inspect), "components",
        lambda self, **_kw: {"nodes": ["displacement_z"], "gauss": []},
    )
    offered = _menu_pick(monkeypatch, None)
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))

    entries = dict((text.split(" (")[0], ok) for text, ok in offered)
    assert entries["Select all nodes"] is True
    assert entries["Select all Gauss"] is False
    assert session.selection.kind is None


def test_c25_the_materials_axis_offers_no_select_all(
    outline, qapp, monkeypatch,
):
    """The axis the realize layer refuses cannot be selected from
    either — one reason, both knobs (see ``_scope``)."""
    session, _widget, tree, _calls = outline
    offered = _menu_pick(monkeypatch, None)
    _right_click(tree, tree._axis_items["materials"])   # noqa: SLF001
    assert offered and all(not enabled for _text, enabled in offered)
    assert session.selection.kind is None


def test_c26_the_menu_subject_is_the_checked_names(
    outline, qapp, monkeypatch,
):
    """Criterion 26, both halves.

    Right-clicking a CHECKED row means everything checked (that is the
    category the user built); right-clicking an unchecked row means
    that row, because acting on something else would be a silent
    substitution.
    """
    session, _widget, tree, _calls = outline
    left = _name_item(tree, "physical_groups", "Left")
    right = _name_item(tree, "physical_groups", "Right")
    from qtpy import QtCore

    left.setCheckState(0, QtCore.Qt.CheckState.Checked)
    right.setCheckState(0, QtCore.Qt.CheckState.Checked)
    qapp.processEvents()

    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, left)
    both = _pg_nodes(session.results, "Left") | _pg_nodes(
        session.results, "Right",
    )
    assert set(session.selection.nodes) == both

    right.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    qapp.processEvents()
    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, right)
    assert set(session.selection.nodes) == _pg_nodes(
        session.results, "Right",
    )


# =====================================================================
# 27 — the gesture is not quadratic
# =====================================================================


def test_c27_one_set_gesture_is_linear_in_comparisons(monkeypatch):
    """Criterion 27, the complexity oracle — machine-independent.

    ``SelectionState.select_batch`` and ``SelectionLog``'s reducer both
    deduped with ``t not in <list>``, so ONE gesture cost O(n^2)
    equality comparisons. It never showed because ``ResultsViewer``
    never wrote ``SelectionState``; §9's "Select all nodes" is the
    first writer that hands it a whole physical group.
    """
    from apeGmsh.results.session import SessionSelection
    from apeGmsh.viewers.scene_ir import SelectionTarget

    n = 2000
    calls = [0]
    real_eq = SelectionTarget.__eq__

    def counting_eq(self, other):
        calls[0] += 1
        return real_eq(self, other)

    monkeypatch.setattr(SelectionTarget, "__eq__", counting_eq)
    selection = SessionSelection()
    selection.set_nodes(range(n))

    assert len(selection) == n
    # Quadratic would be ~n^2/2 = 2,000,000 here. Linear work still
    # pays a few passes (the no-op guard compares the two lists), so
    # the budget is generous and still 100x under the bug.
    assert calls[0] <= 10 * n, (
        f"{calls[0]} target comparisons for {n} targets — that is "
        f"{calls[0] / n:.1f} per target, not a constant"
    )


def test_c27_the_whole_gesture_stays_under_budget():
    """Criterion 27, the other half — the WHOLE gesture.

    S4-1 deliberately put two O(n) costs on top of the store: the
    reconciler's exact selection signature and the highlight's
    membership test, both per pane per flush. The comparison count
    above cannot see those — it only measures the dedup — so the
    end-to-end cost gets its own budget.

    ``n`` is 25k rather than the 100k a real select-all reaches, on
    purpose: a regression here has to FAIL, not hang. At 100k the
    quadratic dedup ran ~17 minutes and would take the CI job with it;
    at 25k it is about a minute, which a stuck build can still report.
    The complexity itself is pinned above, where the oracle is a
    count and the size does not matter.
    """
    import time

    from apeGmsh.results.session import SessionSelection
    from apeGmsh.viewers.session import NodeTargets, PaneTargets

    n = 25_000
    ids = np.arange(1, n + 1, dtype=np.int64)
    targets = PaneTargets(
        nodes=NodeTargets(ids=ids, coords=np.zeros((n, 3))),
    )
    selection = SessionSelection()

    start = time.perf_counter()
    selection.set_nodes(ids)                       # the store write
    signature = tuple(selection.targets)           # the reconciler term
    layer = realize_mod._selection_layer(          # noqa: SLF001
        "mesh-1", selection, targets,
    )
    elapsed = time.perf_counter() - start

    assert len(signature) == n
    assert layer is not None and layer.points.n_points == n
    assert elapsed < 5.0, (
        f"one {n}-target select-all took {elapsed:.1f} s end to end"
    )



# =====================================================================
# 28 — the chart draws the resolved series
# =====================================================================


def _plot_pane(session, widget, qapp, **kwargs):
    plot = session.add_plot(**kwargs)
    qapp.processEvents()
    return plot, widget.frame(plot.id).pane


def test_c28_the_chart_draws_what_results_plot_history_plots(
    host, qapp,
):
    """Criterion 28. The chart is a second drawing of the SAME query,
    so its lines must carry the same arrays — the direct-equality
    oracle, one level above the resolver's own."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    session, widget = host
    node_id = int(session.results.fem.nodes.ids[0])
    _plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])

    (line,) = pane.chart.axes.lines[:1]
    reference = session.results.plot.history(
        node=node_id, component="displacement_z",
    ).lines[0]
    np.testing.assert_allclose(line.get_xdata(), reference.get_xdata())
    np.testing.assert_allclose(line.get_ydata(), reference.get_ydata())


def test_c28_several_series_are_one_chart(host, qapp):
    """§6: "Several series on one chart are one plot view.\""""
    session, widget = host
    ids = [int(i) for i in session.results.fem.nodes.ids[:3]]
    _plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(i), "displacement_z") for i in ids
    ])
    assert len(pane.chart.axes.lines) == len(ids)


# =====================================================================
# 29-31 — the plot's own repaint path
# =====================================================================


def _count_draws(pane, monkeypatch) -> list:
    drawn: list = []
    real = type(pane.chart).draw
    monkeypatch.setattr(
        type(pane.chart), "draw",
        lambda self, realized: (drawn.append(1), real(self, realized))[1],
    )
    return drawn


def test_c29_a_series_write_repaints_once(host, qapp, monkeypatch):
    """Criterion 29, first half. One gesture, one redraw."""
    session, widget = host
    node_ids = [int(i) for i in session.results.fem.nodes.ids[:2]]
    plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(node_ids[0]), "displacement_z"),
    ])
    drawn = _count_draws(pane, monkeypatch)

    plot.series = [
        PlotSeries(PlotSource.node(i), "displacement_z") for i in node_ids
    ]
    qapp.processEvents()

    assert drawn == [1]
    assert len(pane.chart.axes.lines) == 2


def test_c29_an_unrelated_tick_repaints_nothing(host, qapp, monkeypatch):
    """Criterion 29, the gate. The session ticks on EVERY mutation, so
    another pane's slot fill must not cost this chart a redraw."""
    session, widget = host
    node_id = int(session.results.fem.nodes.ids[0])
    _plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    drawn = _count_draws(pane, monkeypatch)

    session.panes[0].name = "renamed"          # ticks; the chart is same
    qapp.processEvents()
    assert drawn == []


def test_c29_the_gate_does_not_skip_the_cursor(host, qapp):
    """Criterion 29, the half S4-1 got wrong one layer down.

    Moving the session instant ticks the session; if the plot's
    signature omitted the §7 instant it would compare equal and the
    playhead would never move — "if the link is on and moving the plot
    does not move the meshes, the link is a lie", read the other way.
    """
    session, widget = host
    node_id = int(session.results.fem.nodes.ids[0])
    _plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    session.time = Instant(STAGE, 0)
    qapp.processEvents()
    first = _marker_x(pane)

    session.time = Instant(STAGE, T - 1)
    qapp.processEvents()
    moved = _marker_x(pane)

    assert first is not None and moved is not None
    assert moved != first, "the playhead did not follow the instant"
    assert moved == pytest.approx(float(T - 1))


def _marker_x(pane) -> "float | None":
    lines = [
        line for line in pane.chart.axes.lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    return None if not lines else float(lines[-1].get_xdata()[0])


def test_c30_a_cursor_move_reads_nothing(host, qapp, monkeypatch):
    """Criterion 30. The whole record is already on screen (§6), so the
    playhead slides over cached arrays — which is what lets a scrubber
    drag stay interactive in S4-3."""
    session, widget = host
    node_id = int(session.results.fem.nodes.ids[0])
    _plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    session.time = Instant(STAGE, 0)
    qapp.processEvents()

    drawn = _count_draws(pane, monkeypatch)
    accessor = type(session.results.stage(STAGE).nodes)
    real, reads = accessor.get, []
    monkeypatch.setattr(
        accessor, "get",
        lambda self, *a, **kw: (reads.append(1), real(self, *a, **kw))[1],
    )

    session.time = Instant(STAGE, T - 1)
    qapp.processEvents()

    assert reads == [], f"a cursor move cost {len(reads)} read(s)"
    assert drawn == [], "a cursor move redrew the whole chart"
    assert _marker_x(pane) == pytest.approx(float(T - 1))


def test_c31_a_selection_write_does_not_redraw_a_plot(
    host, qapp, monkeypatch,
):
    """Criterion 31 — the INVERSE of the mesh pane's rule.

    §6: "select → New plot COPIES the membership into the PlotView at
    creation — a live alias to the session selection is not a v1
    source." So a later pick changes the set and must NOT change this
    chart; a chart that followed the selection would be a live alias
    by accident.
    """
    session, widget = host
    node_ids = [int(i) for i in session.results.fem.nodes.ids[:3]]
    session.selection.set_nodes(node_ids[:1])
    plot = session.add_plot_from_selection("displacement_z")
    qapp.processEvents()
    pane = widget.frame(plot.id).pane
    drawn = _count_draws(pane, monkeypatch)

    session.selection.set_nodes(node_ids)      # the set moves...
    qapp.processEvents()

    assert drawn == []                          # ...the chart does not
    assert [s.source.key for s in plot.series] == node_ids[:1]
    assert len(pane.chart.axes.lines) == 1


def test_c32_a_plot_that_cannot_resolve_says_so_on_the_chart(
    host, qapp, monkeypatch,
):
    """Criterion 32. Empty axes read as "the model did not move"; the
    author needs the reason where they are looking.

    It is reported as well as drawn (ADR 0084 D4 — a swallowed failure
    is how a stale picture survives), which is why the report has to be
    silenced HERE: the autouse ``pump_failures`` fixture turns one into
    a test failure, and this test drives the refusal on purpose.
    """
    reported: list = []
    monkeypatch.setattr(
        chart_mod, "report", lambda name, exc: reported.append(name),
    )
    session, widget = host
    plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(-12345), "displacement_z"),
    ])
    texts = [t.get_text() for t in pane.chart.axes.texts]
    assert texts and "displacement_z" in texts[0]
    assert pane.reconciler.realized is None
    assert reported == ["pump.session_plot"]

    # And it RETRIES: a refusal must not latch the signature, or fixing
    # the series would leave the message on screen forever.
    plot.series = [PlotSeries(
        PlotSource.node(int(session.results.fem.nodes.ids[0])),
        "displacement_z",
    )]
    qapp.processEvents()
    assert len(pane.chart.axes.lines) == 1
    assert not pane.chart.axes.texts


# =====================================================================
# 33 — §8's own test: select → New plot, no Python
# =====================================================================


def test_c33_select_all_then_new_plot_from_selection(
    outline, qapp, monkeypatch,
):
    """Criterion 33, the whole loop.

    §8: "That set IS the source= of plots. If selection does not
    produce a plot without a Python snippet, selection is still
    broken." Two gestures: right-click a group → Select all nodes;
    click the outline row → pick a quantity.
    """
    session, widget, tree, calls = outline
    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))
    qapp.processEvents()

    row = tree.new_plot_from_selection_item
    assert not row.isDisabled()
    assert f"({len(session.selection.nodes)} nodes)" in row.text(0)

    offered = _menu_pick(monkeypatch, "displacement_z")
    tree._on_item_clicked(row, 0)              # noqa: SLF001
    assert ("displacement_z", True) in offered
    assert calls["plot_quantity"] == ["displacement_z"]

    # What the window does with that quantity (its handler, verbatim).
    plot = session.add_plot_from_selection(calls["plot_quantity"][0])
    qapp.processEvents()
    pane = widget.frame(plot.id).pane
    assert len(pane.chart.axes.lines) == len(session.selection.nodes)


def test_c33_the_selection_stays_plottable_from_a_plot_pane(
    outline, qapp, monkeypatch,
):
    """The quantity vocabulary comes from a MESH view (only a mesh view
    resolves a §7 instant to a stage), so clicking the plot pane you
    just made must not empty the picker for the selection you still
    have. Scope does NOT get that fallback — it may only write the pane
    the user selected."""
    session, widget, tree, calls = outline
    session.selection.set_nodes(
        [int(i) for i in session.results.fem.nodes.ids[:2]],
    )
    plot = session.add_plot()
    widget.set_active(plot.id)
    tree.set_active_pane(plot.id)
    qapp.processEvents()

    offered = _menu_pick(monkeypatch, "displacement_z")
    tree._on_item_clicked(                           # noqa: SLF001
        tree.new_plot_from_selection_item, 0,
    )
    assert ("displacement_z", True) in offered
    assert calls["plot_quantity"] == ["displacement_z"]


def test_c33_the_row_is_gated_on_an_empty_selection(outline, qapp):
    """0087 INV-2: with nothing selected the row cannot act, so it does
    not open a picker whose every choice would refuse."""
    session, _widget, tree, _calls = outline
    row = tree.new_plot_from_selection_item
    assert row.isDisabled()
    assert "Nothing is selected" in row.toolTip(0)


def test_c33_the_row_is_gated_past_the_curve_cap(
    outline, qapp, monkeypatch,
):
    """The cap is enforced where the user can still act on it, not only
    where the reads would happen."""
    from apeGmsh.viewers.session import _plots

    session, _widget, tree, _calls = outline
    monkeypatch.setattr(_plots, "MAX_SERIES", 2)
    session.selection.set_nodes(
        [int(i) for i in session.results.fem.nodes.ids[:5]],
    )
    qapp.processEvents()

    row = tree.new_plot_from_selection_item
    assert row.isDisabled()
    assert "cap" in row.toolTip(0)


# =====================================================================
# 34 — the pane frame
# =====================================================================


def test_c34_a_plot_pane_frame_fits_the_a1_4_floor(host, qapp):
    """Criterion 34. Criterion 18's guard, for the other kind of pane:
    a plot frame carries the same header (minus the mesh buttons) and
    the same reserved instant badge, and ``required_extent`` /
    ``can_add`` still multiply by ``LAYOUT.pane_min_width``."""
    from apeGmsh.viewers.ui._layout_metrics import LAYOUT

    session, widget = host
    plot = session.add_plot(name="A deliberately long plot pane name")
    qapp.processEvents()
    frame = widget.frame(plot.id)
    assert frame.minimumSizeHint().width() <= LAYOUT.pane_min_width


def test_c34_a_closed_plot_pane_stops_subscribing(host, qapp):
    """A pane that closes takes its session subscription with it —
    the plot analogue of criterion 14. A live reconciler on a removed
    pane repaints a chart nobody is looking at, forever."""
    session, widget = host
    plot, pane = _plot_pane(session, widget, qapp)
    assert pane.reconciler._on_tick in session._subscribers  # noqa: SLF001

    session.remove_pane(plot.id)
    qapp.processEvents()

    assert pane.reconciler._on_tick not in session._subscribers  # noqa: SLF001
    assert pane.reconciler._disposed            # noqa: SLF001


def test_the_plot_inspector_lists_and_clears_series(host, qapp):
    """§9's Add / change / clear loop, for a plot: the series ARE the
    occupants, and clearing one writes the record."""
    from apeGmsh.viewers.session import PlotInspectorPage

    session, widget = host
    ids = [int(i) for i in session.results.fem.nodes.ids[:2]]
    plot, _pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(i), "displacement_z") for i in ids
    ])
    page = PlotInspectorPage(session, plot)
    try:
        assert page._rows.count() == 2         # noqa: SLF001
        page._drop(0)                          # noqa: SLF001
        qapp.processEvents()
        assert [s.source.key for s in plot.series] == ids[1:]
        assert page._rows.count() == 1         # noqa: SLF001
    finally:
        page.dispose()


# =====================================================================
# 35 — mutation tests. Each names the criterion it must break.
# =====================================================================


def test_mutation_a_cursorless_signature_breaks_c29(host, qapp, monkeypatch):
    """Criterion 29's cursor half, mutated: drop the §7 instant from
    the plot signature — exactly the S4-1 defect one layer up — and
    the playhead stops following the session instant.

    The mutation is installed BEFORE the pane is built, because that is
    what the defect looks like in the product: every signature this
    reconciler ever computes is cursorless, so the gate compares equal
    on a scrubber move and skips the one repaint that mattered.
    """
    session, widget = host
    session.time = Instant(STAGE, 0)
    monkeypatch.setattr(
        chart_mod.PlotReconciler, "_signature_of",
        lambda self, plot: (
            None if plot is None
            else ((plot.id, plot.kind, plot.series), "cursorless")
        ),
    )
    node_id = int(session.results.fem.nodes.ids[0])
    _plot, pane = _plot_pane(session, widget, qapp, series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    first = _marker_x(pane)
    assert first == pytest.approx(0.0), "the rig drew no playhead at all"

    session.time = Instant(STAGE, T - 1)
    qapp.processEvents()

    assert _marker_x(pane) == pytest.approx(0.0), (
        "the cursorless signature still moved the playhead — the test "
        "is not measuring the gate"
    )


def test_mutation_a_second_store_breaks_c22(outline, qapp, monkeypatch):
    """Criterion 22, mutated: have select-all write its own
    ``SelectionState`` instead of the session's, and the ONE-store law
    (0045 INV-5) fails — the pane's set never sees the gesture."""
    from apeGmsh.viewers.core.selection import SelectionState

    import apeGmsh.viewers.session._select_all as select_all_mod

    session, _widget, tree, _calls = outline
    orphan = SelectionState()

    def mutated(sess, kind, *, axis, names, view=None):
        scoped = select_all_mod._membership(       # noqa: SLF001
            sess.results, axis, names,
        )
        from apeGmsh.results.session import node_target

        orphan.select_batch(
            [node_target(int(i)) for i in scoped.node_ids], replace=True,
        )
        return len(scoped.node_ids)

    monkeypatch.setattr(select_all_mod, "select_all", mutated)
    monkeypatch.setattr(
        "apeGmsh.viewers.session._outline.select_all", mutated,
    )
    _menu_pick(monkeypatch, "Select all nodes")
    _right_click(tree, _name_item(tree, "physical_groups", "Left"))

    assert orphan.targets, "the mutation did not run"
    assert session.selection.kind is None, (
        "the session's set filled anyway — the test is not measuring "
        "which store select-all writes"
    )


def test_mutation_a_live_plot_reconciler_breaks_c34(host, qapp):
    """Criterion 34's teardown half, mutated: skip
    ``PlotReconciler.dispose`` and the removed pane keeps ticking."""
    session, widget = host
    plot, pane = _plot_pane(session, widget, qapp)

    pane.reconciler.dispose = lambda: None       # the mutation
    session.remove_pane(plot.id)
    qapp.processEvents()

    assert pane.reconciler._on_tick in session._subscribers, (  # noqa: SLF001
        "the subscription went away without dispose — the test is not "
        "measuring the teardown"
    )
    type(pane.reconciler).dispose(pane.reconciler)   # clean up
