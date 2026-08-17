"""ADR 0098 Amendment 1 (S3) — the pane host, offscreen (default lane).

Sixteen of the twenty-four acceptance criteria are ``[off]``: they run
under ``QT_QPA_PLATFORM=offscreen`` with ``RecordingBackend``\\ s
injected per pane, and they are honest there only because A1.1's seam
stops the SHELL building a ``QtInteractor`` — under offscreen that
context cannot be created at all, and injecting pane backends does not
touch the shell's own. The ``[qt]`` half (18-21) lives in
``test_pane_host_window_qt.py`` and rides the xvfb lane beside
``test_pane_host_probe.py``.

Criterion 22 makes the mutation tests mandatory: inverting the ``T(N)``
fill order, dropping ``childrenCollapsible``, rebuilding panes instead
of moving them, or pointing two panes at one backend must each make at
least one criterion above FAIL. They are at the bottom, each naming the
criterion it breaks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, MeshStyle, Scope
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import (
    SessionPaneFrame,
    SessionPaneHost,
    SessionResultsWindow,
    SessionWindow,
    required_extent,
    tile_columns,
    tile_shape,
)
from apeGmsh.viewers.session import _host as host_mod
from apeGmsh.viewers.session import _reconciler as reconciler_mod
from apeGmsh.viewers.ui._layout_metrics import LAYOUT

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"

#: Comfortably clears the A1.4 floors for T(6) (3 x 240 + 8 = 728 wide,
#: 2 x 200 + 4 = 404 tall), so the Add gate is open in every test that
#: is not about the gate.
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
    """A bare host on a one-pane session, sized to clear the floors."""
    session = session_results.session()
    widget = SessionPaneHost(
        session, backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    # Shown (offscreen: no real surface) so the splitters get a
    # geometry — setSizes on a zero-extent splitter is a no-op, and
    # the ratio law is measured in sizes.
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


@pytest.fixture
def window(qapp, session_results, fleet, settings_scope):
    """The composed window with injected pane backends."""
    win = SessionWindow(
        session_results.session(),
        title="s3-off",
        backend_factory=fleet,
        defer_fn=lambda fn: fn(),
    )
    # Shown (offscreen: no surface) so the pane splitters get a real
    # geometry — the ratio law is measured in sizes, and setSizes on
    # a zero-extent splitter is a no-op.
    win.shell.window.resize(1600, 1000)
    win.shell.window.show()
    _force_host_size(win, qapp, *ROOMY)
    yield win
    win.close()
    qapp.processEvents()


# =====================================================================
# Helpers
# =====================================================================


def _grid(widget: SessionPaneHost) -> "list[list[str]]":
    """The realized tiling, read off the WIDGET tree (not the law)."""
    return [
        [col.widget(i).paneId for i in range(col.count())]
        for col in widget.column_splitters
    ]


def _cells(widget: SessionPaneHost) -> "dict[str, tuple[int, int]]":
    return {
        pane_id: (j, i)
        for j, column in enumerate(_grid(widget))
        for i, pane_id in enumerate(column)
    }


def _ratios(splitter) -> "list[float]":
    sizes = splitter.sizes()
    total = float(sum(sizes)) or 1.0
    return [s / total for s in sizes]


def _force_host_size(win, qapp, width: int, height: int) -> None:
    """Pin the host to an exact size inside the shell's layout.

    The Add gate is a function of the HOST's extent (A1.4), and a
    central widget inside a QMainWindow otherwise takes whatever the
    dock arithmetic leaves it.
    """
    win.host.setFixedSize(width, height)
    qapp.processEvents()


def _add_views(session, n: int) -> None:
    for _ in range(n):
        session.add_view()


# =====================================================================
# 1-2 — the host IS the centre, and it is a projection
# =====================================================================


def test_c1_boot_of_a_one_pane_session(window):
    """Criterion 1."""
    session = window.session
    assert window.shell.window.centralWidget() is window.host
    root = window.host.root_splitter
    from qtpy import QtCore

    assert root.orientation() == QtCore.Qt.Orientation.Horizontal
    assert root.count() == 1
    assert len(window.host.pane_frames) == 1
    assert window.host.pane_frames[0].paneId == session.panes[0].id


def test_c1_shell_builds_no_central_interactor(window):
    """A1.1: every GL context in the window belongs to a pane. With no
    pane owning a surface (injected backends) the shell's plotter
    resolves to nothing at all — which is why the criteria above can
    run offscreen."""
    assert window.shell.plotter is None


def test_c2_host_is_a_projection_of_session_panes(window):
    """Criterion 2 — Python-only writes, in a deliberately mixed order."""
    session = window.session
    a = session.panes[0]
    b = session.add_view("Second")
    p = session.add_plot(name="Curve")
    c = session.add_view("Third")
    session.remove_pane(b.id)
    d = session.add_view("Fourth")
    session.remove_pane(a.id)

    assert [f.paneId for f in window.host.pane_frames] == [
        pane.id for pane in session.panes
    ]
    assert [f.paneId for f in window.host.pane_frames] == [
        p.id, c.id, d.id,
    ]


# =====================================================================
# 3-4b — T(N), identity across a re-tile, the ratio law
# =====================================================================


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, [["a"]]),
        (2, [["a"], ["b"]]),
        (3, [["a", "c"], ["b"]]),
        (4, [["a", "c"], ["b", "d"]]),
        (5, [["a", "d"], ["b", "e"], ["c"]]),
        (6, [["a", "d"], ["b", "e"], ["c", "f"]]),
    ],
)
def test_c3_tiling_law_is_a_pure_function(n, expected):
    """Criterion 3, on the law itself — row-major fill, C = ceil(sqrt N)."""
    ids = list("abcdef")[:n]
    assert tile_columns(ids) == expected


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_c3_tiling_law_holds_on_the_splitter_tree(host, qapp, n):
    """Criterion 3, on the WIDGETS — the law and the tree must agree."""
    from qtpy import QtCore

    session, widget = host
    _add_views(session, n - 1)
    qapp.processEvents()

    root = widget.root_splitter
    assert root.orientation() == QtCore.Qt.Orientation.Horizontal
    for column in widget.column_splitters:
        assert column.orientation() == QtCore.Qt.Orientation.Vertical
        # Depth is exactly two: a column holds pane frames, never
        # another splitter.
        for i in range(column.count()):
            assert isinstance(column.widget(i), SessionPaneFrame)
    assert _grid(widget) == tile_columns([p.id for p in session.panes])


def test_c4_growth_moves_panes_without_rebuilding(host, qapp):
    """Criterion 4 — identity, not just arithmetic.

    A host that tore down and rebuilt its panes on every Add would
    satisfy the cell positions while silently destroying every camera
    (backend state, not session state, so nothing else would catch it).
    """
    session, widget = host
    _add_views(session, 2)  # N = 3
    qapp.processEvents()
    before_frames = {f.paneId: f for f in widget.pane_frames}
    before_backends = {f.paneId: f.backend for f in widget.pane_frames}
    before_cells = _cells(widget)

    session.add_view()  # 3 -> 4: no column-count change
    qapp.processEvents()
    for pane_id, frame in before_frames.items():
        assert widget.frame(pane_id) is frame
        assert widget.frame(pane_id).backend is before_backends[pane_id]
        assert _cells(widget)[pane_id] == before_cells[pane_id]

    session.add_view()  # 4 -> 5: the column count DOES change
    qapp.processEvents()
    for pane_id, frame in before_frames.items():
        assert widget.frame(pane_id) is frame, "pane rebuilt, not moved"
        assert widget.frame(pane_id).backend is before_backends[pane_id]
    # Panes are allowed to move here — and two of them must.
    assert _cells(widget) != before_cells


def test_c4b_ratio_law_resets_only_the_changed_splitter(host, qapp):
    """Criterion 4b (A1.2).

    The one-large-mesh / one-narrow-plot arrangement this design exists
    to support must survive an Add that never touches the splitter
    holding it.
    """
    session, widget = host
    session.add_view()  # N = 2, two columns
    qapp.processEvents()
    widget.root_splitter.setSizes([700, 300])
    qapp.processEvents()

    session.add_view()  # T(2) -> T(3): column 1 grows, root does not
    qapp.processEvents()
    root = _ratios(widget.root_splitter)
    assert root[0] == pytest.approx(0.70, abs=0.01)
    assert root[1] == pytest.approx(0.30, abs=0.01)
    grown = _ratios(widget.column_splitters[0])
    assert grown == [pytest.approx(0.5, abs=0.01)] * 2

    session.add_view()  # T(3) -> T(4): still two columns
    qapp.processEvents()
    assert _ratios(widget.root_splitter)[0] == pytest.approx(0.70, abs=0.01)

    session.add_view()  # T(4) -> T(5): a third column — the root resets
    qapp.processEvents()
    assert _ratios(widget.root_splitter) == [
        pytest.approx(1 / 3, abs=0.01),
    ] * 3


# =====================================================================
# 5-7 — floors, the Add gate, and the gate's limits
# =====================================================================


def test_c5_floors_and_no_collapse(host, qapp):
    """Criterion 5 — a pane can be dragged small, never to zero."""
    session, widget = host
    _add_views(session, 3)
    qapp.processEvents()

    for frame in widget.pane_frames:
        assert frame.minimumWidth() >= LAYOUT.pane_min_width
        assert frame.minimumHeight() >= LAYOUT.pane_min_height
    for splitter in (widget.root_splitter, *widget.column_splitters):
        assert splitter.childrenCollapsible() is False
        assert splitter.handleWidth() == LAYOUT.splitter_handle_width


def test_c6_add_gate_disables_with_the_reason(window, qapp):
    """Criterion 6 — 640 x 671 with four panes cannot take a fifth."""
    session = window.session
    _add_views(session, 3)
    _force_host_size(window, qapp, 640, 671)

    need_w, _need_h = required_extent(5)
    assert need_w == 728  # 3 x 240 + 2 x 4
    for item in (window.outline.new_view_item, window.outline.new_plot_item):
        assert item.isDisabled()
        assert "728" in item.toolTip(0)

    _force_host_size(window, qapp, 960, 871)
    for item in (window.outline.new_view_item, window.outline.new_plot_item):
        assert not item.isDisabled()


def test_c7_the_gate_does_not_gate_the_ir(window, qapp):
    """Criterion 7 — the IR is truth; the host never refuses a pane
    that exists (a script, or an S5 snapshot restore)."""
    session = window.session
    _add_views(session, 3)
    _force_host_size(window, qapp, 640, 671)
    assert window.outline.new_view_item.isDisabled()

    fifth = session.add_view()
    qapp.processEvents()
    assert len(session.panes) == 5
    assert len(window.host.pane_frames) == 5
    assert window.host.frame(fifth.id) is not None


def test_c8_closing_the_last_pane_lands_on_the_empty_card(window, qapp):
    """Criterion 8 — zero panes is valid IR, so the host renders it."""
    session = window.session
    session.remove_pane(session.panes[0].id)
    qapp.processEvents()

    card = window.host.empty_card
    assert card.isVisible() or not window.host.root_splitter.isVisible()
    assert window.host.pane_frames == ()
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


# =====================================================================
# 9 — one active pane at all times
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
# 13-15 — persistence and Reset layout
# =====================================================================


def test_c13_close_writes_the_pane_layout_into_its_own_scope(
    qapp, session_results, fleet, settings_scope,
):
    """Criterion 13."""
    from qtpy.QtCore import QSettings

    old_scope = QSettings("apeGmsh", "ResultsViewer")
    before = {k: str(old_scope.value(k)) for k in old_scope.allKeys()}

    win = SessionWindow(
        session_results.session(), title="s3-persist",
        backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    win.host.resize(*ROOMY)
    _add_views(win.session, 3)
    qapp.processEvents()
    win.close()
    qapp.processEvents()

    stored = settings_scope()
    assert int(stored.value("panes/schema_version")) == 1
    record = json.loads(str(stored.value("panes/layout")))
    assert record["n"] == 4
    assert len(record["cols"]) == 2 and len(record["rows"]) == 2
    assert SessionResultsWindow._LAYOUT_SCHEMA_VERSION == 1

    old_scope = QSettings("apeGmsh", "ResultsViewer")
    after = {k: str(old_scope.value(k)) for k in old_scope.allKeys()}
    assert after == before, "the session window wrote into the old scope"


def test_c14_restore_reproduces_matching_ratios(host, qapp):
    """Criterion 14, positive half."""
    session, widget = host
    _add_views(session, 3)
    qapp.processEvents()
    widget.root_splitter.setSizes([800, 400])
    qapp.processEvents()
    record = widget.layout_ratios()

    widget.reset_tiling()
    qapp.processEvents()
    assert _ratios(widget.root_splitter)[0] == pytest.approx(0.5, abs=0.01)

    assert widget.apply_ratios(record) is True
    qapp.processEvents()
    assert _ratios(widget.root_splitter)[0] == pytest.approx(
        record["cols"][0], abs=0.005,
    )


@pytest.mark.parametrize(
    "record",
    [
        {"n": 99, "cols": [0.5, 0.5], "rows": [[1.0], [1.0]]},
        {"n": 2, "cols": [0.5], "rows": [[1.0]]},
        {"n": 2},
        "not json at all",
        None,
        {"n": 2, "cols": ["x", "y"], "rows": [[1.0], [1.0]]},
    ],
    ids=["wrong-n", "wrong-shape", "missing-keys", "string", "none", "junk"],
)
def test_c14_mismatched_restore_falls_back_silently(
    host, qapp, capsys, record,
):
    """Criterion 14, negative half — equal ratios, no raise, no console
    message. This is chrome, not §1's session-schema notice."""
    session, widget = host
    session.add_view()
    qapp.processEvents()
    capsys.readouterr()

    assert widget.apply_ratios(record) is False
    assert capsys.readouterr().out == ""


def test_c15_reset_layout_restores_docks_and_retiles(window, qapp):
    """Criterion 15."""
    session = window.session
    _add_views(session, 3)
    qapp.processEvents()
    window.host.root_splitter.setSizes([900, 300])
    qapp.processEvents()
    assert _ratios(window.host.root_splitter)[0] > 0.6

    window.shell.reset_layout()
    qapp.processEvents()

    assert _grid(window.host) == tile_columns([p.id for p in session.panes])
    assert _ratios(window.host.root_splitter) == [
        pytest.approx(0.5, abs=0.01),
    ] * 2
    for column in window.host.column_splitters:
        assert _ratios(column) == [
            pytest.approx(1 / column.count(), abs=0.01),
        ] * column.count()


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
    """0087 INV-7 / caution 7 — and no per-pane objectName: the pane id
    rides a Qt property so ONE QSS block styles every pane."""
    frame = window.host.pane_frames[0]
    assert window.host.objectName() == "SessionPaneHost"
    assert frame.objectName() == "SessionPaneFrame"
    assert frame.property("paneId") == frame.paneId
    assert frame.property("active") in ("true", "false")


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

    from apeGmsh.viewers.ui._results_window import ResultsWindow
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    for cls in (ViewerWindow, ResultsWindow):
        default = inspect.signature(
            cls.__init__
        ).parameters["central_interactor"].default
        assert default is True, f"{cls.__name__} defaults to no interactor"

    root = Path(ResultsWindow.__module__.replace(".", "/")).parent
    del root
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


def test_mutation_column_major_fill_breaks_c3(host, qapp, monkeypatch):
    """Invert the T(N) fill order -> criterion 3 fails."""
    session, widget = host

    def column_major(pane_ids):
        columns, rows = tile_shape(len(pane_ids))
        return [
            pane_ids[j * rows:(j + 1) * rows] for j in range(columns)
        ]

    monkeypatch.setattr(host_mod, "tile_columns", column_major)
    _add_views(session, 2)  # N = 3, where the two orders disagree
    qapp.processEvents()

    with pytest.raises(AssertionError):
        assert _grid(widget) == tile_columns([p.id for p in session.panes])


def test_mutation_no_children_collapsible_breaks_c5(
    qapp, session_results, fleet, monkeypatch,
):
    """Drop the childrenCollapsible call -> criterion 5 fails."""
    from qtpy import QtWidgets

    monkeypatch.setattr(
        QtWidgets.QSplitter, "setChildrenCollapsible",
        lambda self, value: None,
    )
    widget = SessionPaneHost(
        session_results.session(),
        backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    try:
        widget.resize(*ROOMY)
        qapp.processEvents()
        with pytest.raises(AssertionError):
            assert widget.root_splitter.childrenCollapsible() is False
    finally:
        widget.dispose()
        widget.setParent(None)


def test_mutation_rebuilding_panes_breaks_c4(qapp, session_results, fleet):
    """Rebuild panes instead of moving them -> criterion 4 fails.

    The cell arithmetic still passes; only the identity assert catches
    it, which is exactly why criterion 4 is written as identity.
    """
    class RebuildingHost(SessionPaneHost):
        def _tile(self, order):
            for pane_id in list(self._frames):
                dead = self._frames.pop(pane_id)
                dead.dispose()
                dead.setParent(None)
            for pane in self._session.panes:
                self._frames[pane.id] = SessionPaneFrame(
                    self._session, pane,
                    backend_factory=self._backend_factory,
                    defer_fn=self._defer_fn,
                    parent=self,
                )
            super()._tile(order)

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
        # The arithmetic survives...
        assert _grid(widget) == tile_columns([p.id for p in session.panes])
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
