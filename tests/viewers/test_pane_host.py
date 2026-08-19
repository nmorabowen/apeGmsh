"""ADR 0098 Amendment 3 — the pane docks, offscreen (default lane).

Amendment 1's twenty-four criteria stand except where A3 restates them.
Rewritten with this amendment rather than patched around, because the
file encoded two things A3 reverses: ``centralWidget() is window.host``
and the ``T(N)`` splitter geometry. What replaces them is one
``QDockWidget`` per pane, and the criteria that named splitters are
restated against dock placement (A3.5).

The ``[qt]`` half (18-21) lives in ``test_pane_host_window_qt.py`` and
rides the xvfb lane. Criterion 22 makes the mutation tests mandatory:
dropping the dock floors, rebuilding panes instead of moving them,
pointing two panes at one backend, or giving every dock the same
objectName must each make at least one criterion above FAIL. They are
at the bottom, each naming the criterion it breaks.

Nothing here measures a MAXIMIZED window, and nothing here can: the
lesson A3.1 paid for twice is that a fixed-size ``show()`` reports
success on layouts that are broken maximized. The maximized numbers are
A3.5's, measured on the ``ssi_frame_wall`` bench, and this file's job is
the contract underneath them — which docks exist, what they are called,
what they minimally measure, and who owns them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, MeshStyle, Scope
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import (
    SessionPaneHost,
    SessionResultsWindow,
    SessionWindow,
    pane_dock_name,
    required_extent,
)
from apeGmsh.viewers.session import _reconciler as reconciler_mod
from apeGmsh.viewers.ui._layout_metrics import LAYOUT
from apeGmsh.viewers.ui._results_window import ResultsWindow

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"

#: Comfortably clears the A1.4 floors for a row of six panes
#: (6 x 240 + 5 x 4 = 1460 wide, 200 tall).
ROOMY = (1600, 800)

#: A window wide enough that the middle column takes a row of five
#: (5 x 240 + 4 x 4 = 1216) with both side columns at their minimums.
WIDE = (2200, 900)

#: ...and one that cannot.
NARROW = (900, 700)


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
    """One static stage: nodal displacement_z + 1-GP gauss stress_xx,
    two physical groups so the scope axis has something to check."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left")
    g.model.geometry.add_box(1, 0, 0, 1, 1, 1, label="right")
    g.physical.add_volume("left", name="Left")
    g.physical.add_volume("right", name="Right")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = np.concatenate(
        [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    )
    disp = np.zeros((2, node_ids.size))
    disp[1] = node_ids
    sxx = np.zeros((2, elem_ids.size, 1))
    sxx[1, :, 0] = elem_ids * 10.0

    path = tmp_path / "pane_host.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=np.zeros((1, 3)),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


class CountingBackend(RecordingBackend):
    """``RecordingBackend`` + a render counter — criterion 12 is a
    claim about how many times ``render()`` is called."""

    def __init__(self) -> None:
        super().__init__()
        self.render_count = 0

    def render(self) -> None:
        self.render_count += 1


class BackendFleet:
    """A per-pane ``RecordingBackend`` factory (plan decision 7).

    One backend per pane is the whole point: criterion 11's isolation
    claim is meaningless against a shared one, which is exactly what
    the "two panes, one backend" mutation test exploits.
    """

    def __init__(self) -> None:
        self.made: list[CountingBackend] = []

    def __call__(self, _parent):
        backend = CountingBackend()
        self.made.append(backend)
        return None, backend


@pytest.fixture
def fleet() -> BackendFleet:
    return BackendFleet()


@pytest.fixture
def host(qapp, session_results, fleet):
    """A bare host on a one-pane session.

    With no ``dock_host`` the host IS its own ``QMainWindow`` — the
    pane docks go in it, and a test can drive the projection with no
    shell around it.
    """
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
def settings_scope(tmp_path, monkeypatch):
    """Redirect the session window's QSettings to a temp ini file.

    The real scope is a per-user registry key on Windows; a test that
    wrote there would both pollute the developer's window and read back
    someone else's arrangement.
    """
    from qtpy.QtCore import QSettings

    path = str(tmp_path / "ResultsSession.ini")

    def _settings():
        return QSettings(path, QSettings.Format.IniFormat)

    monkeypatch.setattr(
        SessionResultsWindow, "_layout_settings", staticmethod(_settings),
    )
    return _settings


def _open_window(session, fleet, *, title="s3-off", size=ROOMY):
    from qtpy import QtWidgets

    win = SessionWindow(
        session, title=title,
        backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    win.shell.window.resize(*size)
    win.shell.window.show()
    app = QtWidgets.QApplication.instance()
    app.processEvents()
    return win


@pytest.fixture
def window(qapp, session_results, fleet, settings_scope):
    """The composed window with injected pane backends."""
    win = _open_window(session_results.session(), fleet)
    yield win
    win.close()
    qapp.processEvents()


# =====================================================================
# Helpers
# =====================================================================


def _dock_names(win) -> "list[str]":
    """The pane docks' objectNames, in ``session.panes`` order."""
    return [d.objectName() for d in win.host.pane_docks]


def _left_area(win) -> "list[str]":
    """Every pane dock Qt reports in the LEFT area (A3.2's middle)."""
    from qtpy import QtCore

    shell = win.shell.window
    return [
        d.objectName() for d in win.host.pane_docks
        if shell.dockWidgetArea(d)
        == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
    ]


def _resize(win, qapp, width: int, height: int) -> None:
    """Resize the whole window and re-settle the columns.

    The panes' extent — and therefore the Add gate (A1.4) — is what the
    two side columns leave. Qt hands new slack to whichever dock it
    likes, so the product re-asserts the LayoutMetrics extents on the
    two-pass schedule A3.4 measured; this is that same entry point.
    """
    win.shell.window.resize(width, height)
    for _ in range(4):
        qapp.processEvents()
    win._apply_dock_extents()  # noqa: SLF001 — the product's own pass
    for _ in range(6):
        qapp.processEvents()


def _front_tab(shell, titles) -> "str | None":
    """The text of the raised tab in the group holding ``titles``.

    Read off the tab STRIP, not off ``visibleRegion``: an offscreen
    window never paints, so the painted region is null for every tab
    there and would call the broken case passing. Qt also leaves stale
    tab bars behind after a re-group, so only a visible one counts —
    and the Inspector/Display pair is a tab group of its own, which is
    why the tab texts have to match.
    """
    from qtpy import QtWidgets

    for bar in shell.findChildren(QtWidgets.QTabBar):
        if not bar.isVisible():
            continue
        if {bar.tabText(i) for i in range(bar.count())} == titles:
            return bar.tabText(bar.currentIndex())
    return None


def _add_views(session, n: int) -> None:
    for _ in range(n):
        session.add_view()


# =====================================================================
# 1-2 — the panes are docks, and they are a projection
# =====================================================================


def test_c1_boot_of_a_one_pane_session(window):
    """Criterion 1, restated against dock placement (A3.5)."""
    session = window.session
    assert len(window.host.pane_frames) == 1
    assert window.host.pane_frames[0].paneId == session.panes[0].id
    assert _dock_names(window) == [pane_dock_name(session.panes[0].id)]
    assert _left_area(window) == _dock_names(window)
    assert window.host.dock(session.panes[0].id).isVisible()


def test_c1_the_panes_are_never_the_central_widget(window):
    """A3.1, as a standing guard rather than a story.

    The centre is dropped on the first show and forcing a GL pane host
    into it takes an access violation. Nothing on this path may put one
    there — and the window's centre stays empty, which is what lets the
    dock areas own the whole client rect.
    """
    assert window.shell.window.centralWidget() is None


def test_c1_shell_builds_no_central_interactor(window):
    """A1.1: every GL context in the window belongs to a pane. With no
    pane owning a surface (injected backends) the shell's plotter
    resolves to nothing at all — which is why the criteria above can
    run offscreen."""
    assert window.shell.plotter is None


def test_c2_host_is_a_projection_of_session_panes(window, qapp):
    """Criterion 2 — Python-only writes, in a deliberately mixed order.

    The dock set is the same claim in the other currency: nothing in
    Qt may create or destroy a pane (A3.3.1), so one dock exists per
    ``session.panes`` entry and not one more.
    """
    session = window.session
    a = session.panes[0]
    b = session.add_view("Second")
    p = session.add_plot(name="Curve")
    c = session.add_view("Third")
    session.remove_pane(b.id)
    d = session.add_view("Fourth")
    session.remove_pane(a.id)
    qapp.processEvents()

    assert [f.paneId for f in window.host.pane_frames] == [
        pane.id for pane in session.panes
    ]
    assert [f.paneId for f in window.host.pane_frames] == [
        p.id, c.id, d.id,
    ]
    assert _dock_names(window) == [
        pane_dock_name(pane.id) for pane in session.panes
    ]
    with pytest.raises(KeyError):
        window.host.dock(b.id)


# =====================================================================
# 3-4b — default placement, identity across an Add, the user's own
#        arrangement
# =====================================================================


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_c3_default_placement_is_a_row(window, qapp, n):
    """Criterion 3, restated (A3.2): N panes, N docks, side by side in
    the middle column, left to right in ``session.panes`` order."""
    session = window.session
    _resize(window, qapp, *WIDE)
    _add_views(session, n - 1)
    qapp.processEvents()

    shell = window.shell.window
    assert _dock_names(window) == [
        pane_dock_name(pane.id) for pane in session.panes
    ]
    assert _left_area(window) == _dock_names(window)
    for dock in window.host.pane_docks:
        assert not dock.isFloating()
        # A row, not a tab stack: nothing is tabified until a user says so.
        assert shell.tabifiedDockWidgets(dock) == []


def test_c3_dock_objectnames_are_derived_from_the_pane_id(window, qapp):
    """A3.3.2 — the objectName is the handle ``saveState`` addresses a
    pane by, so it is a pure function of the pane id and nothing else."""
    session = window.session
    _add_views(session, 2)
    qapp.processEvents()
    for pane in session.panes:
        dock = window.host.dock(pane.id)
        assert dock.objectName() == f"dock_session_pane_{pane.id}"
        assert dock.objectName() == pane_dock_name(pane.id)
    assert len({d.objectName() for d in window.host.pane_docks}) == 3


def test_c4_growth_adds_a_dock_without_rebuilding(window, qapp):
    """Criterion 4 — identity, not just arithmetic.

    A host that tore down and rebuilt its panes on every Add would
    satisfy the placement claim while silently destroying every camera
    (backend state, not session state, so nothing else would catch it).
    """
    session = window.session
    _resize(window, qapp, *WIDE)
    _add_views(session, 2)  # N = 3
    qapp.processEvents()
    frames = {f.paneId: f for f in window.host.pane_frames}
    backends = {f.paneId: f.backend for f in window.host.pane_frames}
    docks = {p.id: window.host.dock(p.id) for p in session.panes}

    fourth = session.add_view()
    qapp.processEvents()

    for pane_id, frame in frames.items():
        assert window.host.frame(pane_id) is frame, "pane rebuilt, not kept"
        assert window.host.frame(pane_id).backend is backends[pane_id]
        assert window.host.dock(pane_id) is docks[pane_id]
    assert window.host.dock(fourth.id).objectName() == pane_dock_name(
        fourth.id,
    )
    assert _left_area(window) == _dock_names(window)


def test_c4b_an_arrangement_the_user_built_survives_an_add(window, qapp):
    """Criterion 4b, restated (A3.2).

    A1.2's ratio law existed because an Add that re-derived the whole
    tiling destroyed the one-large-mesh / one-narrow-plot arrangement
    this design exists to support. Qt owns the arrangement now, and the
    same protection has to hold: an Add places the NEW dock and touches
    no other.
    """
    session = window.session
    _resize(window, qapp, *WIDE)
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()

    shell = window.shell.window
    shell.tabifyDockWidget(window.host.dock(a.id), window.host.dock(b.id))
    qapp.processEvents()
    assert window.host.dock(b.id) in shell.tabifiedDockWidgets(
        window.host.dock(a.id),
    )

    session.add_view("C")
    qapp.processEvents()
    assert window.host.dock(b.id) in shell.tabifiedDockWidgets(
        window.host.dock(a.id),
    ), "the Add re-derived the arrangement instead of adding to it"


# =====================================================================
# 5-7 — floors, the Add gate, and the gate's limits
# =====================================================================


def test_c5_floors_are_the_dock_minimums(host, qapp):
    """Criterion 5, restated (A3.2): A1.4's floors survive as dock
    minimum sizes, so a pane can be dragged small and never to zero.

    They reach the dock through the FRAME — a dock derives its minimum
    from its widget plus its title bar — which is why the frame keeps
    them and the dock does not restate them.
    """
    session, widget = host
    _add_views(session, 3)
    qapp.processEvents()

    for frame in widget.pane_frames:
        assert frame.minimumWidth() >= LAYOUT.pane_min_width
        assert frame.minimumHeight() >= LAYOUT.pane_min_height
    for pane in session.panes:
        dock = widget.dock(pane.id)
        assert dock.minimumWidth() >= LAYOUT.pane_min_width
        assert dock.minimumHeight() >= LAYOUT.pane_min_height


def test_c5_a_pane_dock_is_movable_floatable_closable(host, qapp):
    """A3.2's own sentence: *movable, floatable, closable*. The empty
    card is none of those — it is not a view, it is the absence of one."""
    from qtpy import QtWidgets

    session, widget = host
    dock = widget.dock(session.panes[0].id)
    QDW = QtWidgets.QDockWidget.DockWidgetFeature
    for feature in (QDW.DockWidgetMovable, QDW.DockWidgetFloatable,
                    QDW.DockWidgetClosable):
        assert dock.features() & feature
    assert widget.empty_dock.features() == QDW.NoDockWidgetFeatures


def test_c6_add_gate_disables_with_the_reason(window, qapp):
    """Criterion 6, restated: the gate measures the PANE DOCKS' extent,
    because the host itself is only their controller now."""
    session = window.session
    _add_views(session, 3)  # N = 4
    _resize(window, qapp, *NARROW)

    need_w, _need_h = required_extent(5)
    assert need_w == 1216  # a row of 5 x 240 with 4 separators
    assert window.host.panes_extent()[0] < need_w
    ok, why = window.host.can_add()
    assert ok is False and str(need_w) in why
    for item in (window.outline.new_view_item, window.outline.new_plot_item):
        assert item.isDisabled()
        assert str(need_w) in item.toolTip(0)

    _resize(window, qapp, *WIDE)
    assert window.host.panes_extent()[0] >= need_w
    assert window.host.can_add() == (True, "")
    for item in (window.outline.new_view_item, window.outline.new_plot_item):
        assert not item.isDisabled()


def test_c6_the_gate_agrees_with_the_arithmetic(window, qapp):
    """The gate is ``required_extent`` measured against the docks, at
    any size — not a second opinion about them."""
    session = window.session
    _add_views(session, 2)
    for size in (NARROW, ROOMY, WIDE):
        _resize(window, qapp, *size)
        have_w, have_h = window.host.panes_extent()
        need_w, need_h = required_extent(len(session.panes) + 1)
        assert window.host.can_add()[0] is (
            have_w >= need_w and have_h >= need_h
        )


def test_c7_the_gate_does_not_gate_the_ir(window, qapp):
    """Criterion 7 — the IR is truth; the host never refuses a pane
    that exists (a script, or an S5 snapshot restore)."""
    session = window.session
    _add_views(session, 3)
    _resize(window, qapp, *NARROW)
    assert window.outline.new_view_item.isDisabled()

    fifth = session.add_view()
    qapp.processEvents()
    assert len(session.panes) == 5
    assert len(window.host.pane_frames) == 5
    assert window.host.dock(fifth.id) is not None
    assert _dock_names(window) == [
        pane_dock_name(p.id) for p in session.panes
    ]


def test_c8_closing_the_last_pane_lands_on_the_empty_card(window, qapp):
    """Criterion 8 — zero panes is valid IR, so the host renders it."""
    session = window.session
    session.remove_pane(session.panes[0].id)
    qapp.processEvents()

    card = window.host.empty_card
    assert window.host.pane_frames == ()
    assert window.host.empty_dock.isVisible()
    assert window.host.pane_docks == (window.host.empty_dock,)
    assert card.action.text() == "New mesh view"
    assert card.action.isEnabled()
    # 0087 INV-2: one hint, one action, zero enabled DATA controls.
    from qtpy import QtWidgets

    data_controls = card.findChildren(QtWidgets.QComboBox)
    data_controls += card.findChildren(QtWidgets.QAbstractSpinBox)
    assert data_controls == []
    # The inspector says nothing-selected rather than vanishing.
    assert window.current_page is not None
    assert "No pane selected" in window.current_page.widget.text()


def test_c8_the_empty_card_action_creates_a_pane(window, qapp):
    session = window.session
    session.remove_pane(session.panes[0].id)
    qapp.processEvents()
    window.host.empty_card.action.click()
    qapp.processEvents()
    assert len(session.panes) == 1
    assert len(window.host.pane_frames) == 1
    assert window.host.empty_dock.isVisible() is False


# =====================================================================
# 9 — one active pane at all times, and its dock is the raised one
# =====================================================================


def _assert_one_active(win) -> None:
    active = win.host.active_pane_id
    assert active is not None
    assert win.outline.selected_pane_id() == active
    assert win.current_page is win.inspector_page(active)
    lit = [f.paneId for f in win.host.pane_frames if f.is_active]
    assert lit == [active]


def test_c9_one_active_pane_through_every_gesture(window, qapp):
    """Criterion 9 — click a pane, click an outline row, add a pane,
    close the active pane."""
    session = window.session
    _resize(window, qapp, *WIDE)
    _add_views(session, 2)
    qapp.processEvents()
    _assert_one_active(window)

    # Click a pane (what the frame's event filter does).
    third = session.panes[2]
    window.host.frame(third.id).activate()
    qapp.processEvents()
    assert window.host.active_pane_id == third.id
    _assert_one_active(window)

    # Click an outline row.
    first = session.panes[0]
    window.outline.select_pane(first.id)
    qapp.processEvents()
    assert window.host.active_pane_id == first.id
    _assert_one_active(window)

    # Add a pane — it becomes active and outline-selected (A1.6).
    window.outline.new_view_item.setDisabled(False)
    window._on_new_view()  # noqa: SLF001 — the outline's action
    qapp.processEvents()
    assert window.host.active_pane_id == session.panes[-1].id
    _assert_one_active(window)

    # Close the active pane.
    active = window.host.active_pane_id
    window.host.frame(active).close_button.click()
    qapp.processEvents()
    assert active not in [p.id for p in session.panes]
    _assert_one_active(window)


def test_c9_the_active_pane_is_the_raised_tab(window, qapp):
    """A1.5 expressed as dock focus (A3.2).

    Two panes tabbed together is the one arrangement where "which pane
    is active" and "which pane you can see" can disagree — so making a
    pane active has to bring its dock to the front of the group.
    """
    session = window.session
    _resize(window, qapp, *WIDE)
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    shell = window.shell.window
    shell.tabifyDockWidget(window.host.dock(a.id), window.host.dock(b.id))
    qapp.processEvents()

    titles = {a.name or a.id, b.name or b.id}
    for target in (b.id, a.id, b.id):
        window.host.set_active(target)
        qapp.processEvents()
        assert _front_tab(shell, titles) == (
            window.session.pane(target).name or target
        )


def test_c9_the_docks_own_close_button_writes_the_session(window, qapp):
    """A3.2 made the pane docks closable, so a dock now has TWO close
    affordances. Both mean the same thing: the pane leaves the session
    and the widget follows on the tick — never the other way round."""
    session = window.session
    b = session.add_view("B")
    qapp.processEvents()

    window.host.dock(b.id).close()
    qapp.processEvents()
    assert b.id not in [p.id for p in session.panes]
    with pytest.raises(KeyError):
        window.host.dock(b.id)
    _assert_one_active(window)


def test_c9_closed_pane_evicts_its_inspector_page(window, qapp):
    """Amendment 1 caution 9 — the page dies with the pane, or every
    closed pane leaks a live session subscriber refreshing a dead view."""
    session = window.session
    second = session.add_view()
    qapp.processEvents()
    window.outline.select_pane(second.id)
    qapp.processEvents()
    page = window.inspector_page(second.id)
    subscribers_before = len(session._subscribers)  # noqa: SLF001

    session.remove_pane(second.id)
    qapp.processEvents()

    with pytest.raises(KeyError):
        window.inspector_page(second.id)
    assert window.current_page is not page
    assert len(session._subscribers) < subscribers_before  # noqa: SLF001


# =====================================================================
# 10-11 — style buttons and backend isolation
# =====================================================================


def test_c10_style_buttons_are_per_pane_and_two_way(window, qapp):
    """Criterion 10."""
    session = window.session
    a, b = session.panes[0], session.add_view("B")
    qapp.processEvents()
    before_a = a.style

    window.host.frame(b.id).style_button("outlines").click()
    qapp.processEvents()
    assert b.style.outlines is not before_a.outlines
    assert a.style == before_a, "a pane's button moved another pane"

    # Python -> widget, the other direction.
    b.style = MeshStyle(mesh=False, outlines=False, nodes=True, gauss=False)
    qapp.processEvents()
    frame = window.host.frame(b.id)
    assert frame.style_button("mesh").isChecked() is False
    assert frame.style_button("outlines").isChecked() is False
    assert frame.style_button("nodes").isChecked() is True


def test_c10_style_buttons_are_not_restated_in_the_inspector(window, qapp):
    """A1.3: one state, one control — the mesh-view inspector page does
    NOT carry the four buttons."""
    from qtpy import QtWidgets

    session = window.session
    page = window.inspector_page(session.panes[0].id)
    named = {
        w.objectName()
        for w in page.widget.findChildren(QtWidgets.QAbstractButton)
    }
    assert "SessionPaneStyleButton" not in named


def test_c10_nodes_button_paints_a_node_cloud(window, qapp):
    """The button is a picture, not just a record: INV-MESH-4's node
    glyphs are what click-pick will need on screen (§8)."""
    session = window.session
    view = session.panes[0]
    frame = window.host.frame(view.id)
    assert f"{view.id}:nodes" not in frame.backend.layers

    frame.style_button("nodes").click()
    qapp.processEvents()
    assert view.style.nodes is True
    assert f"{view.id}:nodes" in frame.backend.layers


def test_c10_gauss_button_stands_down_for_the_gauss_slot(window, qapp):
    """INV-MESH-4: "They must not draw two clouds." The slot already
    paints those points, with values and a scale."""
    session = window.session
    view = session.panes[0]
    frame = window.host.frame(view.id)
    frame.style_button("gauss").click()
    qapp.processEvents()
    assert f"{view.id}:gauss" in frame.backend.layers

    from apeGmsh.results.session import Gauss

    view.gauss = Gauss("stress_xx")
    qapp.processEvents()
    emitted = {
        layer.key
        for layer in frame.pane.reconciler.realized.layers
    }
    assert f"{view.id}:gauss" in emitted
    # ONE gauss key — the slot's, not the slot's plus the button's.
    assert sum(1 for k in emitted if k.startswith(f"{view.id}:gauss")) == 1
    assert view.style.gauss is True  # the button still describes the view


def test_c11_pane_isolation_at_the_backend(window, qapp):
    """Criterion 11."""
    session = window.session
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    frame_a, frame_b = window.host.frame(a.id), window.host.frame(b.id)
    assert frame_a.backend is not frame_b.backend
    b_layers_before = dict(frame_b.backend.layers)

    a.contour = Contour("displacement_z")
    qapp.processEvents()

    # Stable keys, not raw layer_ids: the backend keys by layer_id,
    # which is an emission detail (S1 contract).
    keys_a = {l.key for l in frame_a.pane.reconciler.realized.layers}
    assert f"{a.id}:contour" in keys_a
    assert frame_b.backend.layers == b_layers_before
    assert frame_b.backend.scalar_bars == {}
    for frame in window.host.pane_frames:
        realized = frame.pane.reconciler.realized
        assert realized is not None
        assert realized.pane_id == frame.paneId


# =====================================================================
# 12 — one gesture, one realize (a GATE, not a target)
# =====================================================================


def test_c12_one_gesture_costs_one_realize(window, qapp, monkeypatch):
    """Criterion 12, first half."""
    session = window.session
    _add_views(session, 3)  # four panes
    qapp.processEvents()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *a, **k: (calls.append(a[1].id), real(*a, **k))[1],
    )
    renders = {
        f.paneId: f.backend.render_count for f in window.host.pane_frames
    }

    session.panes[0].contour = Contour("displacement_z")
    qapp.processEvents()

    assert calls == [session.panes[0].id]
    after = {
        f.paneId: f.backend.render_count for f in window.host.pane_frames
    }
    moved = [pid for pid in renders if after[pid] != renders[pid]]
    assert moved == [session.panes[0].id]


def test_c12_a_theme_change_costs_every_pane(window, qapp, monkeypatch):
    """Criterion 12, second half: the palette is not session state, so
    an equality check over the session would skip exactly the repaint
    that is needed."""
    session = window.session
    _add_views(session, 3)
    qapp.processEvents()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *a, **k: (calls.append(a[1].id), real(*a, **k))[1],
    )
    window.host.request_reconcile()
    qapp.processEvents()

    assert sorted(calls) == sorted(p.id for p in session.panes)


def test_c12_a_no_op_tick_costs_nothing(window, qapp, monkeypatch):
    """The signature guard's own oracle: a tick that moves nothing this
    pane draws must not realize at all."""
    session = window.session
    view = session.panes[0]
    view.contour = Contour("displacement_z")
    qapp.processEvents()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    view.name = "renamed"  # ticks the session; realize reads no name
    qapp.processEvents()
    assert calls == []


# =====================================================================
# 13-15 — persistence (A3.3) and Reset layout
# =====================================================================


def test_c13_close_writes_the_arrangement_into_its_own_scope(
    qapp, session_results, fleet, settings_scope,
):
    """Criterion 13, restated (A3.3.2): the arrangement is the WINDOW's
    dock state now, keyed by each pane's objectName, in the session's
    own QSettings scope.

    The retired ``panes/*`` splitter-ratio record is gone with the
    splitters, and nothing writes it any more.
    """
    from qtpy.QtCore import QSettings

    old_scope = QSettings("apeGmsh", "ResultsViewer")
    before = {k: str(old_scope.value(k)) for k in old_scope.allKeys()}

    win = _open_window(session_results.session(), fleet, title="s3-persist")
    _add_views(win.session, 2)
    qapp.processEvents()
    pane_ids = [p.id for p in win.session.panes]
    win.close()
    qapp.processEvents()

    stored = settings_scope()
    assert int(stored.value("layout/schema_version")) == 2
    assert SessionResultsWindow._LAYOUT_SCHEMA_VERSION == 2
    state = bytes(stored.value("layout/state"))
    for pane_id in pane_ids:
        # QMainWindow.saveState writes objectNames as UTF-16 QStrings.
        assert pane_dock_name(pane_id).encode("utf-16-be") in state
    assert "panes/layout" not in stored.allKeys()

    old_scope = QSettings("apeGmsh", "ResultsViewer")
    after = {k: str(old_scope.value(k)) for k in old_scope.allKeys()}
    assert after == before, "the session window wrote into the old scope"


def test_c14_the_session_decides_existence_the_file_only_placement(
    qapp, session_results, fleet, settings_scope, capsys,
):
    """Criterion 14, restated — A3.3's whole rule, in one round trip.

    Save an arrangement for three panes, then reopen a session that
    dropped one of them and gained one the file never saw. The saved
    entry for the pane that no longer exists is DROPPED (nothing
    resurrects it), the pane with no entry takes DEFAULT placement, and
    neither is an error and neither prompts.
    """
    first = session_results.session()
    win = _open_window(first, fleet, title="s3-save", size=WIDE)
    _add_views(first, 2)
    qapp.processEvents()
    saved_ids = [p.id for p in first.panes]
    win.close()
    qapp.processEvents()

    second = session_results.session()
    second.remove_pane(second.panes[0].id)   # drop the one boot pane
    kept = second.add_view("kept")
    gone = saved_ids[-1]
    capsys.readouterr()

    win2 = _open_window(second, fleet, title="s3-restore", size=WIDE)
    try:
        qapp.processEvents()
        assert [p.id for p in second.panes] == [kept.id]
        assert _dock_names(win2) == [pane_dock_name(kept.id)]
        with pytest.raises(KeyError):
            win2.host.dock(gone)          # dropped, not resurrected
        assert _left_area(win2) == _dock_names(win2)
        assert capsys.readouterr().out == ""
    finally:
        win2.close()
        qapp.processEvents()


def test_c14_a_v1_arrangement_is_discarded_whole(
    qapp, session_results, fleet, settings_scope,
):
    """A3.5 — the dock set changed structurally, so v1 state must not
    half-apply. The schema gate is what stops it, and it is only real
    if a v1 record is actually ignored."""
    stored = settings_scope()
    stored.setValue("layout/schema_version", 1)
    stored.setValue("layout/state", b"not a v2 arrangement")
    stored.sync()

    win = _open_window(session_results.session(), fleet, title="s3-v1")
    try:
        qapp.processEvents()
        assert _dock_names(win) == [
            pane_dock_name(p.id) for p in win.session.panes
        ]
        assert _left_area(win) == _dock_names(win)
    finally:
        win.close()
        qapp.processEvents()


def test_c15_reset_layout_restores_docks_and_default_placement(
    window, qapp,
):
    """Criterion 15, restated: View → Reset layout puts the 0088 D1 dock
    set back (the shell's job) AND the panes back in a row — un-tabbed,
    re-docked, in ``session.panes`` order."""
    session = window.session
    _resize(window, qapp, *WIDE)
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    shell = window.shell.window
    shell.tabifyDockWidget(window.host.dock(a.id), window.host.dock(b.id))
    window.host.dock(b.id).setFloating(True)
    qapp.processEvents()
    assert window.host.dock(b.id).isFloating()

    window.shell.reset_layout()
    qapp.processEvents()

    assert _dock_names(window) == [
        pane_dock_name(p.id) for p in session.panes
    ]
    assert _left_area(window) == _dock_names(window)
    for dock in window.host.pane_docks:
        assert not dock.isFloating()
        assert shell.tabifiedDockWidgets(dock) == []


# =====================================================================
# 16 — guards
# =====================================================================


def test_c16_the_four_style_glyphs_exist_in_the_factory():
    """Criterion 16, glyph half."""
    from apeGmsh.viewers.ui._icon_factory import glyph_names

    assert {"mesh", "outlines", "nodes", "gauss"} <= set(glyph_names())


def test_c16_every_style_button_binds_a_factory_glyph(window):
    """No text-only fallback: a button whose glyph failed to render
    would be an unlabelled empty square, not a legible control."""
    frame = window.host.pane_frames[0]
    for name in ("mesh", "outlines", "nodes", "gauss"):
        button = frame.style_button(name)
        assert not button.icon().isNull()
        assert button.text() == ""
        assert button.toolTip()


def test_c16_new_widgets_use_pascalcase_objectnames(window):
    """0087 INV-7 / caution 7 — and no per-pane objectName on the
    WIDGETS: the pane id rides a Qt property so ONE QSS block styles
    every pane. The DOCK is the exception A3.3.2 requires, and it is
    the dock, not the frame, that carries the id."""
    frame = window.host.pane_frames[0]
    assert window.host.objectName() == "SessionPaneHost"
    assert frame.objectName() == "SessionPaneFrame"
    assert frame.property("paneId") == frame.paneId
    assert frame.property("active") in ("true", "false")


def test_c16_the_retired_pane_dock_name_is_not_reused(window):
    """A3.5's objectName discipline: ``dock_results_panes`` was the
    intermediate single-panel spike and must never name different
    content, or a stale saved state would half-apply to it."""
    from qtpy import QtWidgets

    names = {
        d.objectName()
        for d in window.shell.window.findChildren(QtWidgets.QDockWidget)
    }
    assert "dock_results_panes" not in names
    assert not hasattr(ResultsWindow, "DOCK_PANES")
    source = Path(ResultsWindow.__module__.replace(".", "/"))
    del source


# =====================================================================
# Per-view scope (ADR §11's S3 line, IR half)
# =====================================================================


def test_scope_checkboxes_write_the_active_view(window, qapp):
    session = window.session
    view = session.panes[0]
    item = window.outline._name_items["physical_groups"]["Left"]  # noqa: SLF001
    from qtpy import QtCore

    item.setCheckState(0, QtCore.Qt.CheckState.Checked)
    qapp.processEvents()
    assert view.scope == Scope(axis="physical_groups", names=("Left",))

    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    qapp.processEvents()
    assert view.scope is None


def test_scope_axis_switches_rather_than_intersects(window, qapp):
    """§9: v1 scope is ONE axis — never a boolean ``Left AND hexahedron``."""
    from qtpy import QtCore

    session = window.session
    view = session.panes[0]
    window.outline._name_items["physical_groups"]["Left"].setCheckState(  # noqa: SLF001
        0, QtCore.Qt.CheckState.Checked,
    )
    qapp.processEvents()
    types = list(window.outline._name_items["element_types"])  # noqa: SLF001
    window.outline._name_items["element_types"][types[0]].setCheckState(  # noqa: SLF001
        0, QtCore.Qt.CheckState.Checked,
    )
    qapp.processEvents()
    assert view.scope.axis == "element_types"


def test_materials_axis_is_present_but_disabled(window):
    """0087 INV-2 — no element→material index exists, so the axis does
    not pretend it can act (see ``_scope.py``)."""
    axis = window.outline._axis_items["materials"]  # noqa: SLF001
    assert axis.isDisabled()
    assert "element" in axis.toolTip(0) and "material" in axis.toolTip(0)


def test_two_panes_two_scopes_two_pictures(window, qapp):
    """ADR §11's S3 verify line, IR/layer half: two mesh views with
    different scopes render different cell sets."""
    session = window.session
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    a.scope = Scope(axis="physical_groups", names=("Left",))
    b.scope = Scope(axis="physical_groups", names=("Right",))
    qapp.processEvents()

    cells_a = window.host.frame(a.id).backend.layers[
        f"{a.id}:substrate"
    ].cells.n_cells
    cells_b = window.host.frame(b.id).backend.layers[
        f"{b.id}:substrate"
    ].cells.n_cells
    whole = window.host.frame(a.id).backend.layers[
        f"{a.id}:substrate"
    ]
    assert cells_a > 0 and cells_b > 0
    assert whole is not None
    a.scope = None
    qapp.processEvents()
    cells_all = window.host.frame(a.id).backend.layers[
        f"{a.id}:substrate"
    ].cells.n_cells
    assert cells_all == cells_a + cells_b


def test_two_panes_different_slots_different_legends(window, qapp):
    """The §5 oracle, per pane: a contour pane reports one legend, a
    deform-on/no-slots pane reports zero (INV-LEGEND-5)."""
    from apeGmsh.results.session import Deform

    session = window.session
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    a.contour = Contour("displacement_z")
    b.deform = Deform("displacement")
    qapp.processEvents()

    assert len(window.host.frame(a.id).backend.scalar_bars) == 1
    assert window.host.frame(b.id).backend.scalar_bars == {}


# =====================================================================
# 24 — the old window is untouched
# =====================================================================


def test_c24_the_seam_is_additive_and_opt_in():
    """Criterion 24. The old window never reaches the seam, so its
    construction path is what it always was."""
    import inspect

    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    for cls in (ViewerWindow, ResultsWindow):
        default = inspect.signature(
            cls.__init__
        ).parameters["central_interactor"].default
        assert default is True, f"{cls.__name__} defaults to no interactor"

    from apeGmsh.viewers import results_viewer

    source = Path(results_viewer.__file__).read_text(encoding="utf-8")
    assert "central_interactor" not in source
    assert "set_plotter_provider" not in source


def test_c24_plotter_without_a_provider_is_the_windows_own(window):
    """The identity the old path relies on: with no provider installed
    ``plotter`` IS ``_qt_interactor``, unconditionally."""
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    vw = ViewerWindow.__new__(ViewerWindow)
    vw._qt_interactor = object()  # noqa: SLF001
    vw._plotter_provider = None  # noqa: SLF001
    assert vw.plotter is vw._qt_interactor  # noqa: SLF001


# =====================================================================
# 22 — mutation tests (a green smoke proves nothing until a broken
#      build fails it). Each names the criterion it must break.
# =====================================================================


def test_mutation_no_pane_floors_breaks_c5(
    qapp, session_results, fleet, monkeypatch,
):
    """Take A1.4's floors off the frame -> criterion 5 fails.

    The dock derives its minimum from its widget, so this is the
    mutation that used to swallow ``setChildrenCollapsible``: with the
    frame's floors gone the dock will happily shrink past them.
    """
    import dataclasses

    from apeGmsh.viewers.session import _frame as frame_mod

    monkeypatch.setattr(
        frame_mod, "LAYOUT",
        dataclasses.replace(LAYOUT, pane_min_width=0, pane_min_height=0),
    )
    session = session_results.session()
    widget = SessionPaneHost(
        session, backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    try:
        widget.resize(*ROOMY)
        qapp.processEvents()
        dock = widget.dock(session.panes[0].id)
        with pytest.raises(AssertionError):
            assert dock.minimumWidth() >= LAYOUT.pane_min_width
            assert dock.minimumHeight() >= LAYOUT.pane_min_height
    finally:
        widget.dispose()
        widget.setParent(None)


def test_mutation_one_dock_name_for_every_pane_breaks_c3(
    qapp, session_results, fleet, monkeypatch,
):
    """Give every pane dock the same objectName -> the A3.3.2 naming
    criterion fails, and with it any hope of ``saveState`` addressing
    one pane rather than whichever dock answered first."""
    from apeGmsh.viewers.session import _host as host_mod

    monkeypatch.setattr(
        host_mod, "pane_dock_name", lambda pane_id: "dock_session_pane",
    )
    session = session_results.session()
    widget = SessionPaneHost(
        session, backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    try:
        widget.resize(*ROOMY)
        session.add_view("B")
        qapp.processEvents()
        names = {d.objectName() for d in widget.pane_docks}
        with pytest.raises(AssertionError):
            assert len(names) == len(session.panes)
    finally:
        widget.dispose()
        widget.setParent(None)


def test_mutation_rebuilding_panes_breaks_c4(qapp, session_results, fleet):
    """Rebuild panes instead of keeping them -> criterion 4 fails.

    The dock set still comes out right; only the identity assert catches
    it, which is exactly why criterion 4 is written as identity.
    """
    class RebuildingHost(SessionPaneHost):
        def refresh(self):
            if self._disposed:
                return
            for pane_id in list(self._docks):
                self._drop_dock(pane_id)
            self._listed = []
            super().refresh()

    session = session_results.session()
    widget = RebuildingHost(
        session, backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    try:
        widget.resize(*ROOMY)
        _add_views(session, 2)
        qapp.processEvents()
        before = {f.paneId: f for f in widget.pane_frames}
        session.add_view()
        qapp.processEvents()
        # The dock set survives...
        assert [d.objectName() for d in widget.pane_docks] == [
            pane_dock_name(p.id) for p in session.panes
        ]
        # ...and the identity does not.
        with pytest.raises(AssertionError):
            for pane_id, frame in before.items():
                assert widget.frame(pane_id) is frame
    finally:
        widget.dispose()
        widget.setParent(None)


def test_mutation_one_shared_backend_breaks_c11(qapp, session_results):
    """Point two panes at one backend -> criterion 11 fails."""
    shared = RecordingBackend()
    session = session_results.session()
    widget = SessionPaneHost(
        session,
        backend_factory=lambda _p: (None, shared),
        defer_fn=lambda fn: fn(),
    )
    try:
        widget.resize(*ROOMY)
        a = session.panes[0]
        b = session.add_view("B")
        qapp.processEvents()
        frame_b = widget.frame(b.id)
        b_layers_before = dict(frame_b.backend.layers)
        a.contour = Contour("displacement_z")
        qapp.processEvents()
        with pytest.raises(AssertionError):
            assert frame_b.backend.layers == b_layers_before
    finally:
        widget.dispose()
        widget.setParent(None)
