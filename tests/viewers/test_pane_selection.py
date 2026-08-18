"""ADR 0098 §8 (S4-1) — selection surface + pane pick, offscreen lane.

The default lane: ``QT_QPA_PLATFORM=offscreen`` with a per-pane
recording backend that ALSO answers ``supports_picking()``, handing
each pane a fake :class:`PickBackend`. That fake owns the same seam the
real ``PyVistaPickBackend`` owns (``install`` / ``uninstall`` /
``project_points`` and the two geometric callbacks), so everything
below the seam — which target a hit resolves to, which writer runs,
what the pane repaints — is exercised here without GL. The ``[qt]``
half (the real ray-cast, real pickability, real observer teardown)
lives in ``test_pane_selection_qt.py``.

The criteria:

1.  The ONE store — a click writes ADR 0045's ``SelectionState``, and
    its log can undo the gesture.
2.  Click replaces the set with the target nearest the ray.
3.  Ctrl+click extends, and toggles the same target back off.
4.  A Gauss write over a node set REPLACES it (§8's XOR law).
5.  Style button off → the click is a non-event, not a clear.
6.  Plain click on nothing clears; Ctrl+click on nothing keeps.
7.  A window selects what is inside it, scoped to this view's visible
    cells.
8.  Ctrl+window extends.
9.  Two panes may aim differently over the ONE set.
10. The radio is two-way and per-pane, and it never touches the set.
11. The selection is PAINTED, on this pane's pose.
12. The signature gate lets that repaint through — once per pane that
    can show it, never for a pane that cannot.
13. The glyph clouds are pickable; nothing else this slice emits is.
14. A pane that closes takes its pick installation with it.
15. The Gauss target encoding is ``(element_id, gp-within-element)``.

Criterion 16 makes the mutation tests mandatory: dropping the
signature's selection term, flipping the node cloud back to
``pickable=False``, or leaving the pick installed on pane teardown
must each make one criterion above FAIL. They are at the bottom, each
naming the criterion it breaks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.session import Deform, Gauss, MeshStyle, Scope
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.scene_ir import BoxGesture, PickHit, PickModifiers
from apeGmsh.viewers.session import SessionPaneHost
from apeGmsh.viewers.session import _pane as pane_mod
from apeGmsh.viewers.session import _realize as realize_mod
from apeGmsh.viewers.session import _reconciler as reconciler_mod

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"
ROOMY = (1200, 800)

#: The fake camera's world→display scale. Node coords live in
#: ``[0, 2] x [0, 1] x [0, 1]``, so display x spans 0..200 px and the
#: box tests can name a pixel rectangle by hand.
PX = 100.0


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
    """Two physical groups, and TWO integration points per element so
    the ``gp_index`` half of a Gauss target is not trivially zero."""
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
    disp[1] = np.asarray(fem.nodes.coords, dtype=np.float64)[:, 2] * 10.0
    sxx = np.zeros((2, elem_ids.size, 2))
    sxx[1, :, 0] = elem_ids * 10.0
    sxx[1, :, 1] = elem_ids * 20.0

    path = tmp_path / "pane_selection.h5"
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
            element_index=elem_ids,
            natural_coords=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


class FakePickBackend:
    """The ``PickBackend`` seam, without VTK.

    Implements exactly what :class:`PanePick` uses — ``install`` /
    ``uninstall`` / ``project_points`` and the two geometric callbacks
    — plus test drivers that fire those callbacks the way the real
    gesture machine does on a release. The projection is a fixed
    orthographic camera down -Z, which is a legal implementation of
    the seam's contract and makes a pixel rectangle nameable.
    """

    def __init__(self) -> None:
        self.on_pick = None
        self.on_box = None
        self.installed = False
        self.uninstall_calls = 0

    # -- the seam -----------------------------------------------------

    def install(self, *, on_pick, on_hover=None, on_box=None) -> None:
        self.on_pick = on_pick
        self.on_box = on_box
        self.installed = True

    def uninstall(self) -> None:
        self.installed = False
        self.uninstall_calls += 1

    def project_points(self, world) -> np.ndarray:
        w = np.asarray(world, dtype=np.float64)
        return np.column_stack([w[:, 0] * PX, w[:, 1] * PX])

    # -- drivers (what a release does) --------------------------------

    def click(self, world, *, ctrl: bool = False) -> None:
        self.on_pick(
            PickHit(world=tuple(float(c) for c in world), cell_id=0,
                    prop_id=1),
            PickModifiers(ctrl=ctrl),
        )

    def click_nothing(self, *, ctrl: bool = False) -> None:
        self.on_pick(None, PickModifiers(ctrl=ctrl))

    def drag(self, box, *, ctrl: bool = False) -> None:
        self.on_box(BoxGesture(
            box=box, crossing=box[2] < box[0],
            modifiers=PickModifiers(ctrl=ctrl),
        ))


class PickingBackend(RecordingBackend):
    """``RecordingBackend`` + the optional pick capability + a render
    counter (criterion 12 counts realizes and renders)."""

    def __init__(self) -> None:
        super().__init__()
        self.render_count = 0
        self._picker = FakePickBackend()

    def render(self) -> None:
        self.render_count += 1

    def supports_picking(self) -> bool:
        return True

    def picking(self):
        return self._picker


class BackendFleet:
    """One :class:`PickingBackend` per pane (plan decision 7)."""

    def __init__(self) -> None:
        self.made: list[PickingBackend] = []

    def __call__(self, _parent):
        backend = PickingBackend()
        self.made.append(backend)
        return None, backend


@pytest.fixture
def fleet() -> BackendFleet:
    return BackendFleet()


@pytest.fixture
def host(qapp, session_results, fleet):
    """A host on a one-pane session, sized to clear the A1.4 floors."""
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


# =====================================================================
# Helpers
# =====================================================================


def _picker(widget, pane_id):
    return widget.frame(pane_id).backend.picking()


def _targets(widget, pane_id):
    return widget.frame(pane_id).pane.reconciler.realized.targets


def _node_at(widget, pane_id, row: int):
    """``(node_id, world_coords)`` of one drawn node target."""
    nodes = _targets(widget, pane_id).nodes
    return int(nodes.ids[row]), nodes.coords[row]


def _nodes_on(view) -> None:
    view.style = MeshStyle(mesh=True, outlines=True, nodes=True, gauss=False)


def _gauss_on(view) -> None:
    view.style = MeshStyle(mesh=True, outlines=True, nodes=False, gauss=True)


# =====================================================================
# 1 — the ONE store
# =====================================================================


def test_c1_a_click_writes_the_one_selection_state(host, qapp):
    """Criterion 1. ``session.selection`` is the owner-facing surface of
    ADR 0045's store (INV-5), never a second one — so the gesture a
    click makes is in THAT log, and undo reverses it."""
    from apeGmsh.viewers.core.selection import SelectionState

    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    state = session.selection.state
    assert isinstance(state, SelectionState)

    node_id, world = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(world)

    assert session.selection.kind == "nodes"
    assert session.selection.nodes == (node_id,)
    assert [t.key for t in state.targets] == [node_id]
    assert state.undo() is True
    assert session.selection.nodes == ()


def test_c1_no_pick_controller_without_a_picking_backend(
    qapp, session_results,
):
    """A view-only backend is legal (ADR 0042 INV-3): the pane renders
    and simply cannot pick — it must not construct one blindly."""
    widget = SessionPaneHost(
        session_results.session(),
        backend_factory=lambda _p: (None, RecordingBackend()),
        defer_fn=lambda fn: fn(),
    )
    try:
        pane_id = widget.pane_frames[0].paneId
        assert widget.frame(pane_id).pane.pick is None
    finally:
        widget.dispose()
        widget.setParent(None)


# =====================================================================
# 2-3 — click, and Ctrl+click
# =====================================================================


def test_c2_click_replaces_with_the_target_nearest_the_ray(host, qapp):
    """Criterion 2 — and the snap is from the RAY's world point, which
    is what makes a click on a face resolve to a node of that face."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    first, world_a = _node_at(widget, view.id, 0)
    second, world_b = _node_at(widget, view.id, 3)
    picker = _picker(widget, view.id)

    picker.click(world_a)
    assert session.selection.nodes == (first,)

    # A point NEAR the second node, not on it.
    picker.click(np.asarray(world_b) + 1e-3)
    assert session.selection.nodes == (second,), "the set grew, not replaced"


def test_c3_ctrl_click_extends_then_toggles_off(host, qapp):
    """Criterion 3."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    a, world_a = _node_at(widget, view.id, 0)
    b, world_b = _node_at(widget, view.id, 1)
    picker = _picker(widget, view.id)

    picker.click(world_a)
    picker.click(world_b, ctrl=True)
    assert set(session.selection.nodes) == {a, b}

    picker.click(world_b, ctrl=True)
    assert session.selection.nodes == (a,)


# =====================================================================
# 4 — nodes XOR Gauss, last writer replaces
# =====================================================================


def test_c4_a_gauss_write_replaces_a_node_set(host, qapp):
    """Criterion 4 (§8): the set is homogeneous and its kind follows
    the last writer — through the REAL store, not a mock."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    picker = _picker(widget, view.id)
    _node_id, world = _node_at(widget, view.id, 0)
    picker.click(world)
    assert session.selection.kind == "nodes"

    _gauss_on(view)
    view.pick_target = "gauss"
    qapp.processEvents()
    gauss = _targets(widget, view.id).gauss
    picker.click(gauss.coords[0])

    assert session.selection.kind == "gauss"
    assert session.selection.nodes == ()
    assert len(session.selection) == 1


# =====================================================================
# 5-6 — the style gate, and a miss
# =====================================================================


def test_c5_click_pick_requires_the_matching_style_button(host, qapp):
    """Criterion 5 (§8: "click-pick requires the matching style button
    on"). With the button off the pane draws no such cloud, so the
    click is a NON-EVENT — it must not clear a set the user filled from
    the outline either."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    node_id, world = _node_at(widget, view.id, 0)
    picker = _picker(widget, view.id)
    picker.click(world)
    assert session.selection.nodes == (node_id,)

    view.style = MeshStyle(nodes=False, gauss=False)
    qapp.processEvents()
    assert _targets(widget, view.id).nodes is None
    picker.click(world)
    assert session.selection.nodes == (node_id,), "the set was touched"


def test_c6_a_plain_miss_clears_and_a_ctrl_miss_does_not(host, qapp):
    """Criterion 6."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    node_id, world = _node_at(widget, view.id, 0)
    picker = _picker(widget, view.id)

    picker.click(world)
    picker.click_nothing(ctrl=True)
    assert session.selection.nodes == (node_id,)

    picker.click_nothing()
    assert len(session.selection) == 0


# =====================================================================
# 7-8 — the window writer
# =====================================================================


def test_c7_a_window_selects_what_is_inside_it(host, qapp):
    """Criterion 7, first half."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    nodes = _targets(widget, view.id).nodes
    # The left half of the model, in the fake camera's pixels.
    _picker(widget, view.id).drag((-10, -10, int(1.01 * PX), int(1.10 * PX)))

    expected = {
        int(i) for i, c in zip(nodes.ids, nodes.coords) if c[0] <= 1.01
    }
    assert expected, "the rig drew no node in the box"
    assert set(session.selection.nodes) == expected


def test_c7_a_window_is_scoped_to_the_views_visible_cells(host, qapp):
    """Criterion 7, second half (§8: "scoped to this view's visible
    cells"). The scope is the pane's cell set, so a node the pane does
    not draw cannot be windowed even when the rectangle covers it."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    view.scope = Scope(axis="physical_groups", names=("Right",))
    qapp.processEvents()
    drawn = set(int(i) for i in _targets(widget, view.id).nodes.ids)

    _picker(widget, view.id).drag((-10, -10, 10_000, 10_000))
    selected = set(session.selection.nodes)

    assert selected == drawn
    assert all(c[0] >= 0.99 for c in _targets(widget, view.id).nodes.coords)


def test_c8_ctrl_window_extends(host, qapp):
    """Criterion 8."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    picker = _picker(widget, view.id)

    picker.drag((-10, -10, int(1.01 * PX), int(1.10 * PX)))
    left = set(session.selection.nodes)
    picker.drag((-10, -10, 10_000, 10_000), ctrl=True)
    everything = set(session.selection.nodes)

    assert left < everything
    assert everything == set(
        int(i) for i in _targets(widget, view.id).nodes.ids
    )


def test_c8_a_degenerate_rectangle_is_not_a_gesture(host, qapp):
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    node_id, world = _node_at(widget, view.id, 0)
    picker = _picker(widget, view.id)
    picker.click(world)

    picker.drag((50, 50, 50, 90))
    assert session.selection.nodes == (node_id,)


# =====================================================================
# 9-10 — the radio aims, it does not own
# =====================================================================


def test_c9_two_panes_may_aim_differently_over_one_set(host, qapp):
    """Criterion 9 (§8): "two views may aim at different targets over
    the one set"."""
    session, widget = host
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    _nodes_on(a)
    _gauss_on(b)
    b.pick_target = "gauss"
    qapp.processEvents()
    assert a.pick_target == "nodes", "pane B's radio moved pane A's"

    node_id, world = _node_at(widget, a.id, 0)
    _picker(widget, a.id).click(world)
    assert session.selection.nodes == (node_id,)

    gauss = _targets(widget, b.id).gauss
    _picker(widget, b.id).click(gauss.coords[0])
    assert session.selection.kind == "gauss"
    assert len(session.selection) == 1
    # Neither radio moved: the set changed kind, the aims did not.
    assert (a.pick_target, b.pick_target) == ("nodes", "gauss")


def test_c10_the_radio_is_two_way_and_per_pane(host, qapp):
    """Criterion 10 — the widget half."""
    session, widget = host
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    frame_a, frame_b = widget.frame(a.id), widget.frame(b.id)
    assert frame_a.pick_button("nodes").isChecked() is True
    assert frame_a.pick_button("gauss").isChecked() is False

    frame_b.pick_button("gauss").click()
    qapp.processEvents()
    assert b.pick_target == "gauss"
    assert a.pick_target == "nodes", "a pane's radio moved another pane"

    # Python -> widget, the other direction.
    b.pick_target = "nodes"
    qapp.processEvents()
    assert frame_b.pick_button("nodes").isChecked() is True
    assert frame_b.pick_button("gauss").isChecked() is False


def test_c10_the_radio_never_touches_the_set(host, qapp):
    """Criterion 10 — the law half: the radio "neither owns nor clears"
    the selection."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    node_id, world = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(world)

    widget.frame(view.id).pick_button("gauss").click()
    qapp.processEvents()
    assert view.pick_target == "gauss"
    assert session.selection.nodes == (node_id,)


def test_c10_the_gauss_radio_is_disabled_without_gauss_data(
    qapp, g, tmp_path, fleet,
):
    """0087 INV-2 — aiming clicks at a cloud that cannot exist."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="only")
    g.physical.add_volume("only", name="Only")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    path = tmp_path / "nodal_only.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((2, node_ids.size))},
        )
        w.end_stage()
    results = Results.from_native(path, model=_open_model_from_h5(path))

    widget = SessionPaneHost(
        results.session(), backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    try:
        frame = widget.pane_frames[0]
        assert frame.pick_button("gauss").isEnabled() is False
        assert frame.pick_button("nodes").isEnabled() is True
        assert "No Gauss" in frame.pick_button("gauss").toolTip()
    finally:
        widget.dispose()
        widget.setParent(None)


# =====================================================================
# 11-12 — the selection is PAINTED, and the gate lets it through
# =====================================================================


def test_c11_the_selection_is_painted_on_this_panes_glyphs(host, qapp):
    """Criterion 11. Without this the set is invisible and §8's four
    writers write into the dark."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    key = f"{view.id}:selection"
    backend = widget.frame(view.id).backend
    assert key not in backend.layers

    node_id, world = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(world)

    layer = backend.layers[key]
    assert layer.points.n_points == 1
    assert np.allclose(layer.points.coords[0], world)
    assert f"{view.id}:selection" in {
        rl.key for rl in widget.frame(view.id).pane.reconciler.realized.layers
    }
    del node_id


def test_c11_the_highlight_follows_the_pose(host, qapp):
    """Criterion 11, second half (§8: "highlights follow the current
    pose"). The coords come from the same posed cloud the glyphs do."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    node_id, undeformed = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(undeformed)

    view.deform = Deform("displacement", scale=5.0)
    qapp.processEvents()
    nodes = _targets(widget, view.id).nodes
    posed = nodes.coords[int(np.flatnonzero(nodes.ids == node_id)[0])]
    painted = widget.frame(view.id).backend.layers[
        f"{view.id}:selection"
    ].points.coords[0]

    assert np.allclose(painted, posed)
    assert not np.allclose(painted, undeformed), "the rig posed nothing"


def test_c11_a_selection_outside_this_panes_scope_marks_nothing(host, qapp):
    """One set, N panes: a pane marks only what it draws."""
    session, widget = host
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    _nodes_on(a)
    _nodes_on(b)
    a.scope = Scope(axis="physical_groups", names=("Left",))
    b.scope = Scope(axis="physical_groups", names=("Right",))
    qapp.processEvents()

    node_id, world = _node_at(widget, a.id, 0)
    _picker(widget, a.id).click(world)

    assert f"{a.id}:selection" in widget.frame(a.id).backend.layers
    assert f"{b.id}:selection" not in widget.frame(b.id).backend.layers
    del node_id


def test_c12_a_selection_write_costs_one_realize_per_showing_pane(
    host, qapp, monkeypatch,
):
    """Criterion 12. The signature gate is a GATE, not a target: the
    set is session-wide, so every pane that can SHOW it repaints — and
    a pane with both glyph buttons off pays nothing, because realize
    reads no selection there (§8: "query-from-outline may fill the set
    with glyphs off")."""
    session, widget = host
    a = session.panes[0]
    b = session.add_view("B")
    qapp.processEvents()
    _nodes_on(a)          # can show the set
    b.style = MeshStyle(nodes=False, gauss=False)   # cannot
    qapp.processEvents()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *args, **kw: (calls.append(args[1].id), real(*args, **kw))[1],
    )
    _node_id, world = _node_at(widget, a.id, 0)
    _picker(widget, a.id).click(world)

    assert calls == [a.id]


def test_c12_a_second_identical_write_costs_nothing(host, qapp, monkeypatch):
    """The gate's own oracle, on the new term: re-clicking the same
    node changes no membership, so nothing repaints."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    _node_id, world = _node_at(widget, view.id, 0)
    picker = _picker(widget, view.id)
    picker.click(world)

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *args, **kw: (calls.append(1), real(*args, **kw))[1],
    )
    picker.click(world)
    assert calls == []


# =====================================================================
# 13 — the glyph clouds ARE the pick targets
# =====================================================================


def test_c13_the_glyph_clouds_are_pickable(host, qapp):
    """Criterion 13. §8 makes the style button the gate for click-pick,
    which only means anything if the cloud it draws can be hit — with
    mesh and outlines off there is nothing else for the ray to land
    on."""
    session, widget = host
    view = session.panes[0]
    view.style = MeshStyle(mesh=False, outlines=False, nodes=True, gauss=True)
    qapp.processEvents()
    layers = widget.frame(view.id).backend.layers

    assert layers[f"{view.id}:nodes"].pickable is True
    assert layers[f"{view.id}:gauss"].pickable is True


def test_c13_the_highlight_is_not_a_pick_target(host, qapp):
    """A highlight is a mark ON a target, not a second target: a
    pickable one would shadow the glyph it marks."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    _node_id, world = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(world)

    layer = widget.frame(view.id).backend.layers[f"{view.id}:selection"]
    assert layer.pickable is False


def test_c13_the_gauss_slot_keeps_the_cloud_pickable(host, qapp):
    """INV-MESH-4 stands the BUTTON's layer down when the slot occupies
    — but the points are still on screen (with values), so §8 must
    still be able to hit them."""
    session, widget = host
    view = session.panes[0]
    _gauss_on(view)
    view.pick_target = "gauss"
    view.gauss = Gauss("stress_xx")
    qapp.processEvents()

    emitted = {
        rl.key for rl in widget.frame(view.id).pane.reconciler.realized.layers
    }
    assert sum(1 for k in emitted if k.startswith(f"{view.id}:gauss")) == 1
    targets = _targets(widget, view.id).gauss
    assert targets is not None and targets.coords.shape[0] > 0

    _picker(widget, view.id).click(targets.coords[0])
    assert session.selection.kind == "gauss"


# =====================================================================
# 14 — a pane that closes takes its pick with it
# =====================================================================


def test_c14_closing_a_pane_uninstalls_its_pick(host, qapp):
    """Criterion 14 (Amendment 1 caution 1, moved to the pick seam).
    The test must not uninstall anything itself — the assertion is that
    the PRODUCT did, on the pane's own close path."""
    session, widget = host
    session.add_view("B")
    qapp.processEvents()
    victim = session.panes[1]
    picker = _picker(widget, victim.id)
    assert picker.installed is True

    widget.frame(victim.id).close_button.click()
    qapp.processEvents()

    assert victim.id not in [p.id for p in session.panes]
    assert picker.installed is False
    assert picker.uninstall_calls == 1


def test_c14_host_dispose_uninstalls_every_pick(qapp, session_results, fleet):
    session = session_results.session()
    widget = SessionPaneHost(
        session, backend_factory=fleet, defer_fn=lambda fn: fn(),
    )
    session.add_view("B")
    qapp.processEvents()
    pickers = [b.picking() for b in fleet.made]
    assert len(pickers) == 2 and all(p.installed for p in pickers)

    widget.dispose()
    widget.setParent(None)

    assert all(p.installed is False for p in pickers)


def test_c14_dispose_is_idempotent(host, qapp):
    session, widget = host
    view = session.panes[0]
    pane = widget.frame(view.id).pane
    picker = _picker(widget, view.id)
    pane.dispose()
    pane.dispose()
    assert picker.uninstall_calls == 1


# =====================================================================
# 15 — the Gauss target encoding
# =====================================================================


@pytest.mark.parametrize(
    "element_index, expected",
    [
        ([7, 7, 9, 9], [0, 1, 0, 1]),
        ([7, 9, 7, 9], [0, 0, 1, 1]),          # interleaved, slab order
        ([5], [0]),
        ([], []),
    ],
    ids=["grouped", "interleaved", "single", "empty"],
)
def test_c15_gp_index_counts_within_the_element(element_index, expected):
    """Criterion 15, on the derivation. ``GaussSlab`` carries no
    gp_index column, and the §8 target means the index WITHIN the
    element — the same one ``_plots._gauss_series`` resolves."""
    out = realize_mod._gp_index_within_element(  # noqa: SLF001
        np.asarray(element_index, dtype=np.int64),
    )
    assert out.tolist() == expected


def test_c15_a_gauss_click_resolves_to_element_and_gp(host, qapp):
    """Criterion 15, end to end: the pair a click writes is the pair
    ``results.elements.gauss`` answers for that point."""
    session, widget = host
    view = session.panes[0]
    _gauss_on(view)
    view.pick_target = "gauss"
    qapp.processEvents()
    targets = _targets(widget, view.id).gauss
    # A point that is the SECOND integration point of its element.
    rows = np.flatnonzero(targets.gp_indices == 1)
    assert rows.size, "the rig recorded one GP per element"
    row = int(rows[0])

    _picker(widget, view.id).click(targets.coords[row])

    assert session.selection.gauss == (
        (int(targets.element_ids[row]), 1),
    )
    slab = session.results.stage(STAGE).elements.gauss.get(
        ids=[int(targets.element_ids[row])], component="stress_xx",
    )
    assert np.flatnonzero(
        np.asarray(slab.element_index) == int(targets.element_ids[row])
    ).size == 2


def test_c15_the_selection_becomes_a_plot_source(host, qapp):
    """§8: "That set IS the ``source=`` of plots. If selection does not
    produce a plot without a Python snippet, selection is still
    broken." The outline action is S4-2; the surface it will call is
    here and it works off a PICKED set."""
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    node_id, world = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(world)

    plot = session.add_plot_from_selection("displacement_z")
    assert [s.source.key for s in plot.series] == [node_id]


def test_every_pick_target_token_has_a_target_family():
    """The radio addresses a ``PaneTargets`` field BY NAME, so a
    ``pick_target`` token with no family would aim at nothing and say
    nothing — a silent non-event rather than a loud gap."""
    import dataclasses

    from apeGmsh.results.session import PICK_TARGETS
    from apeGmsh.viewers.session import PaneTargets

    families = {f.name for f in dataclasses.fields(PaneTargets)}
    assert families == set(PICK_TARGETS)


# =====================================================================
# 16 — mutation tests. Each names the criterion it must break.
# =====================================================================


def test_mutation_signature_without_selection_breaks_c11(
    host, qapp, monkeypatch,
):
    """Drop the signature's selection term -> criterion 11 fails.

    This is the trap the S3 gate sets for S4: the write ticks the
    session, the signature compares equal, and the flush is SKIPPED —
    so the highlight never paints and nothing else notices.
    """
    session, widget = host
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()
    monkeypatch.setattr(
        reconciler_mod.SessionReconciler, "_selection_term",
        lambda self, pane: (),
    )
    _node_id, world = _node_at(widget, view.id, 0)
    _picker(widget, view.id).click(world)

    assert session.selection.kind == "nodes", "the write itself failed"
    with pytest.raises(AssertionError):
        assert f"{view.id}:selection" in widget.frame(view.id).backend.layers


def test_mutation_unpickable_node_cloud_breaks_c13(host, qapp, monkeypatch):
    """Flip the node cloud back to ``pickable=False`` -> criterion 13
    fails."""
    from dataclasses import replace

    session, widget = host
    real = realize_mod._nodes_layer  # noqa: SLF001
    monkeypatch.setattr(
        realize_mod, "_nodes_layer",
        lambda pane_id, targets: replace(
            real(pane_id, targets), pickable=False,
        ),
    )
    view = session.panes[0]
    _nodes_on(view)
    qapp.processEvents()

    with pytest.raises(AssertionError):
        assert widget.frame(view.id).backend.layers[
            f"{view.id}:nodes"
        ].pickable is True


def test_mutation_pane_dispose_without_the_pick_breaks_c14(
    host, qapp, monkeypatch,
):
    """Leave the pick installed on pane teardown -> criterion 14 fails.

    The mutation is the pre-S4-1 ``dispose`` body: reconciler and
    surface, no pick. On a real interactor that is observers living on
    a closed GL context.
    """
    def old_dispose(self) -> None:
        self._reconciler.dispose()  # noqa: SLF001
        if self._surface is not None:  # noqa: SLF001
            self._surface.close()  # noqa: SLF001

    monkeypatch.setattr(pane_mod.MeshPane, "dispose", old_dispose)
    session, widget = host
    session.add_view("B")
    qapp.processEvents()
    victim = session.panes[1]
    picker = _picker(widget, victim.id)

    widget.frame(victim.id).close_button.click()
    qapp.processEvents()

    with pytest.raises(AssertionError):
        assert picker.installed is False
