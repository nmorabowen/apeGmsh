"""ADR 0098 A5.3 — the pane binds its colour scale to its view.

The Qt half of Amendment 5. The pure record and the snapshot are in
``tests/results/session/test_legend_placement.py``.

These run on ``RecordingBackend``, which has no ``.plotter`` — so
``install_legend_interactor`` finds no VTK interactor and returns
``None``, which is exactly criterion 1's headless case. What is under
test here is everything around the gesture: that the binding follows
the controller when realize swaps it, that a mutation is recorded on
the view and repainted, and that a drag costs no realize. That the
mouse event itself reaches a REAL interactor was measured separately
on the bench (A5.1) and cannot be asserted without a GL context.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, Deform
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session._pane import MeshPane

from .conftest import RecordingBackend
from tests.conftest import _open_model_from_h5

STAGE = "s0"


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def results(g, tmp_path: Path):
    """One box with a nodal field a contour slot can paint."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.zeros((2, node_ids.size))
    disp[1] = np.asarray(fem.nodes.coords, dtype=np.float64)[:, 2] * 10.0

    path = tmp_path / "legend_binding.h5"
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
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def pane(qapp, results):
    """One pane on a contour view, flushed to first paint."""
    session = results.session()
    view = session.panes[0]
    view.contour = Contour("displacement_z")
    backends: list = []

    def factory(_parent):
        backend = RecordingBackend()
        backends.append(backend)
        return None, backend

    widget = MeshPane(
        session, pane_id=view.id,
        backend_factory=factory, defer_fn=lambda fn: fn(),
    )
    widget.reconciler.flush_now()
    qapp.processEvents()
    yield session, view, widget, backends[0]
    widget.dispose()


def _controller(widget):
    realized = widget.reconciler.realized
    return realized.legend_controller if realized else None


def _key(view):
    return ("", view.legends()[0].field)


# ----------------------------------------------------------------------
# A5.6-1 — install
# ----------------------------------------------------------------------

def test_a_headless_backend_installs_no_interactor_and_does_not_raise(pane):
    _session, _view, widget, _backend = pane
    assert _controller(widget) is not None
    # RecordingBackend has no `.plotter`, so there is nothing to observe.
    assert widget._legends._interactor is None
    # ...but the binding still took the dispatcher seat, which is what
    # records and repaints.
    assert _controller(widget).dispatcher is widget._legends


def test_a_view_with_no_legend_binds_nothing(qapp, results):
    session = results.session()
    view = session.panes[0]
    view.deform = Deform("displacement", 2.0)      # pose causes no legend
    widget = MeshPane(
        session, pane_id=view.id,
        backend_factory=lambda _p: (None, RecordingBackend()),
        defer_fn=lambda fn: fn(),
    )
    try:
        widget.reconciler.flush_now()
        qapp.processEvents()
        assert view.legends() == ()
        assert widget._legends._controller is None
    finally:
        widget.dispose()


# ----------------------------------------------------------------------
# A5.6-2 — a mutation is recorded on the VIEW, not just the controller
# ----------------------------------------------------------------------

def test_a_drag_is_recorded_on_the_view(pane):
    _session, view, widget, _backend = pane
    controller, key = _controller(widget), _key(view)

    controller.set_anchor(key, (0.11, 0.22))

    placement = view.legend_placement("displacement_z")
    assert placement is not None
    assert placement.anchor == pytest.approx((0.11, 0.22))


def test_a_resize_is_recorded_on_the_view(pane):
    _session, view, widget, _backend = pane
    controller, key = _controller(widget), _key(view)

    controller.set_anchor(key, (0.11, 0.22))       # undock first
    controller.set_font_scale(key, 1.4)

    assert view.legend_placement("displacement_z").font_scale == \
        pytest.approx(1.4)


def test_a_mutation_repaints(pane):
    """The session has no Dispatcher; without the binding in that seat
    a dragged bar moves in the layout and never on screen."""
    _session, view, widget, backend = pane
    before = backend.renders
    _controller(widget).set_anchor(_key(view), (0.3, 0.3))
    assert backend.renders > before


# ----------------------------------------------------------------------
# A5.6-5 — redock clears the record
# ----------------------------------------------------------------------

def test_redock_clears_the_record(pane):
    _session, view, widget, _backend = pane
    controller, key = _controller(widget), _key(view)

    controller.set_anchor(key, (0.11, 0.22))
    assert view.legend_placement("displacement_z") is not None

    controller.redock(key)
    assert view.legend_placement("displacement_z") is None


# ----------------------------------------------------------------------
# A5.6-3 — it survives a full realize
# ----------------------------------------------------------------------

def test_a_drag_survives_a_full_realize(pane, qapp):
    """The amendment's whole point. Before A5 this measured a THIRD
    position: a new controller laid the bar out from scratch."""
    _session, view, widget, _backend = pane
    _controller(widget).set_anchor(_key(view), (0.11, 0.22))
    first = _controller(widget)

    view.deform = Deform("displacement", 3.0)      # structural -> realize
    widget.reconciler.flush_now()
    qapp.processEvents()

    assert _controller(widget) is not first, "expected a fresh controller"
    entry = _controller(widget).entry(_key(view))
    assert entry.anchor == pytest.approx((0.11, 0.22))
    assert entry.slot is None, "a placed legend must stay undocked"
    assert view.legend_placement("displacement_z").anchor == \
        pytest.approx((0.11, 0.22))


def test_a_resize_survives_a_full_realize(pane, qapp):
    _session, view, widget, _backend = pane
    controller, key = _controller(widget), _key(view)
    controller.set_anchor(key, (0.11, 0.22))
    controller.set_font_scale(key, 1.4)

    view.deform = Deform("displacement", 3.0)
    widget.reconciler.flush_now()
    qapp.processEvents()

    assert _controller(widget).entry(key).font_scale == pytest.approx(1.4)


def test_the_binding_follows_the_new_controller(pane, qapp):
    """A5.3(3). Without the re-bind the pane keeps observing the dead
    controller and the NEXT drag records nothing."""
    _session, view, widget, _backend = pane
    view.deform = Deform("displacement", 3.0)
    widget.reconciler.flush_now()
    qapp.processEvents()

    live = _controller(widget)
    assert widget._legends._controller is live
    assert live.dispatcher is widget._legends

    live.set_anchor(_key(view), (0.44, 0.55))
    assert view.legend_placement("displacement_z").anchor == \
        pytest.approx((0.44, 0.55))


# ----------------------------------------------------------------------
# A5.6-9 — a drag costs no realize
# ----------------------------------------------------------------------

def test_a_drag_costs_no_realize(pane, monkeypatch):
    """Criterion 12 still holds. Placement is deliberately absent from
    the structure signature, so the flush a drag ticks compares equal
    and returns before realize."""
    from apeGmsh.viewers.session import _reconciler as rec

    calls = []
    real = rec.realize_pane
    monkeypatch.setattr(
        rec, "realize_pane",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    _session, view, widget, _backend = pane
    controller, key = _controller(widget), _key(view)

    for i in range(5):                              # a drag is many moves
        controller.set_anchor(key, (0.1 + i / 100, 0.2))
    widget.reconciler.flush_now()

    assert calls == []


# ----------------------------------------------------------------------
# A5.6-8 — placement is per view, not per session
# ----------------------------------------------------------------------

def test_dragging_one_pane_does_not_move_the_other(qapp, results):
    session = results.session()
    first = session.panes[0]
    first.contour = Contour("displacement_z")
    second = session.add_view(name="second")
    second.contour = Contour("displacement_z")

    widgets = []
    for view in (first, second):
        widget = MeshPane(
            session, pane_id=view.id,
            backend_factory=lambda _p: (None, RecordingBackend()),
            defer_fn=lambda fn: fn(),
        )
        widget.reconciler.flush_now()
        widgets.append(widget)
    qapp.processEvents()
    try:
        _controller(widgets[0]).set_anchor(("", "displacement_z"),
                                           (0.11, 0.22))
        assert first.legend_placement("displacement_z") is not None
        assert second.legend_placement("displacement_z") is None
    finally:
        for widget in widgets:
            widget.dispose()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------

def test_dispose_releases_the_dispatcher_seat(pane):
    _session, _view, widget, _backend = pane
    controller = _controller(widget)
    widget.dispose()
    assert controller.dispatcher is None


# ----------------------------------------------------------------------
# The inspector's Line row had no vocabulary at all
# ----------------------------------------------------------------------

@pytest.fixture
def results_with_lines(g, tmp_path: Path):
    """A stage that records the ``line_stations`` family.

    The geometry is irrelevant here — what is under test is the
    component VOCABULARY the inspector reads, which comes straight off
    the file's families. Reusing the box keeps the fixture to the one
    thing that matters: a stage carrying line stations as well as nodes.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = np.concatenate(
        [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    )[:4]
    disp = np.zeros((2, node_ids.size))
    moment = np.zeros((2, elem_ids.size, 2))
    moment[1, :, 0] = elem_ids * 1.0

    path = tmp_path / "line_stations.h5"
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
        w.write_line_stations_group(
            sid, "partition_0", "group_0",
            class_tag=1, int_rule=0,
            element_index=elem_ids,
            station_natural_coord=np.array([0.0, 1.0]),
            components={"bending_moment_z": moment},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def test_line_components_come_from_the_line_stations_family(
    results_with_lines,
):
    """Regression. ``recorded_components`` returns (nodal, gauss), and
    the inspector filtered ``_LINE_COMPONENTS`` against those two — a
    line component appears in neither, so the test was always false,
    the Line row offered nothing, and its Add button was permanently
    dead. Meanwhile ``view.line = Line(...)`` from a script drew the
    diagram perfectly well, which is why no test caught it."""
    from apeGmsh.viewers.session._realize import (
        recorded_components, recorded_line_components,
    )

    session = results_with_lines.session()
    view = session.panes[0]
    nodal, gauss = recorded_components(session, view)
    line = recorded_line_components(session, view)

    assert line, "the fixture records line stations"
    # The bug, stated as a law: these families are disjoint, so the old
    # nodal/gauss filter could never match a line component.
    assert not (line & (nodal | gauss))


# ----------------------------------------------------------------------
# A6 G3 — the layout is only valid for the viewport it was measured on
# ----------------------------------------------------------------------

def test_a_stale_layout_is_detected_and_corrected(pane):
    """A legend's box is derived from pixel font metrics, so it is
    valid only for the viewport it was resolved against — and every
    resize EVENT is unreliable: VTK's ``ConfigureEvent`` does not fire
    for a programmatic resize, and Qt's ``resizeEvent`` lands BEFORE
    the render window has taken the new size, so a re-layout driven
    from it re-measures against the old one.

    So the layout is brought current at the point of use. Measured
    consequence without it: a pane realized at construction size kept a
    box computed against a ~30 px viewport, and the hit test then
    missed a bar plainly on screen."""
    _session, _view, widget, _backend = pane
    controller = _controller(widget)

    controller.ensure_current()
    assert controller.ensure_current() is False, (
        "a current layout must not re-run — this is called from the "
        "hit test, i.e. potentially per mouse-move."
    )

    controller._laid_out_px = (31, 29)      # the state the bug leaves
    assert controller.ensure_current() is True
    assert controller.viewport_px() == controller._laid_out_px


def test_a_viewport_smaller_than_its_margins_yields_no_negative_box(pane):
    """``min(h_px/vh, 1 - 2*my)`` goes NEGATIVE once the viewport is
    smaller than its own margins, handing back a box whose far edge
    precedes its near one — which no hit test can ever match. Found at
    a 30 px viewport, so it is reachable, not theoretical."""
    _session, _view, widget, backend = pane
    controller = _controller(widget)

    backend.viewport = (24, 18)             # smaller than 2 * MARGIN_PX
    controller.ensure_current()

    for entry in controller.entries():
        w, h = entry.extent
        assert w >= 0.0 and h >= 0.0, (
            f"{entry.key} has a negative extent {(w, h)} at "
            f"{controller.viewport_px()}"
        )
