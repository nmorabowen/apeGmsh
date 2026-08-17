"""§1/§3/§6 session surface — stable pane ids, the boot picture, the
change tick, plots copied from selection, and the value records."""
from __future__ import annotations

import pytest

from apeGmsh.results.session import (
    Contour,
    Deform,
    Instant,
    MeshStyle,
    MeshView,
    PlotSeries,
    PlotSource,
    PlotView,
    ResultsSession,
    Scope,
)


# ----------------------------------------------------------------------
# Panes carry a stable id (§1)
# ----------------------------------------------------------------------

def test_pane_ids_are_stable_and_unique() -> None:
    session = ResultsSession()
    v1 = session.add_view()
    p1 = session.add_plot()
    v2 = session.add_view(name="close-up")
    assert (v1.id, p1.id, v2.id) == ("mesh-1", "plot-2", "mesh-3")
    assert v2.name == "close-up"
    assert session.pane("plot-2") is p1
    assert session.panes == (v1, p1, v2)


def test_pane_lookup_miss_is_a_key_error() -> None:
    session = ResultsSession()
    with pytest.raises(KeyError, match="mesh-9"):
        session.pane("mesh-9")


def test_removed_pane_stops_ticking_the_session() -> None:
    # Probe H15: a removed pane a caller still holds must not keep
    # notifying (phantom repaints in S2) — remove detaches the backref.
    session = ResultsSession()
    view = session.add_view()
    session.remove_pane(view.id)
    ticks: list[int] = []
    session.subscribe(lambda: ticks.append(1))
    view.contour = Contour("S")   # mutating the orphan still works...
    assert view.contour == Contour("S")
    assert ticks == []            # ...but the session no longer hears it


def test_removed_pane_id_is_not_reused() -> None:
    session = ResultsSession()
    v1 = session.add_view()
    session.remove_pane(v1.id)
    with pytest.raises(KeyError):
        session.pane(v1.id)
    v2 = session.add_view()
    assert v2.id != v1.id  # ids are names, never recycled


# ----------------------------------------------------------------------
# Boot of a view, no slots — the valid empty picture (§3)
# ----------------------------------------------------------------------

def test_boot_view_is_the_grey_mesh_picture() -> None:
    view = ResultsSession().add_view()
    assert view.style == MeshStyle(
        mesh=True, outlines=True, nodes=False, gauss=False,
    )
    assert view.scope is None          # whole analysis mesh
    assert view.deform is None         # undeformed pose
    assert view.time is None
    assert view.overlay is False
    assert view.pick_target == "nodes"
    assert view.slots == {}
    assert view.legends() == ()        # no legend (§5)
    assert view.clips == ()


# ----------------------------------------------------------------------
# The change tick — one tick per mutating gesture, none for no-ops
# ----------------------------------------------------------------------

def test_every_author_gesture_fires_the_tick() -> None:
    session = ResultsSession()
    ticks: list[int] = []
    session.subscribe(lambda: ticks.append(1))

    view = session.add_view()                     # 1
    view.contour = Contour("S")                   # 2
    view.style = MeshStyle(nodes=True)            # 3
    view.deform = Deform(scale=10.0)              # 4
    session.time = Instant("push", 3)             # 5
    session.time_linked = False                   # 6
    session.selection.set_nodes([1, 2])           # 7
    view.add_clip((0.0, 0.0, 1.0))                # 8
    assert len(ticks) == 8


def test_noop_writes_do_not_tick() -> None:
    session = ResultsSession()
    view = session.add_view()
    view.contour = Contour("S")
    session.time = Instant("push", 3)
    ticks: list[int] = []
    session.subscribe(lambda: ticks.append(1))

    view.contour = Contour("S")           # equal record — replace is a no-op
    view.style = MeshStyle()              # unchanged
    session.time = Instant("push", 3)     # unchanged
    session.time_linked = True            # already linked
    view.overlay = False                  # already off
    assert ticks == []


def test_unsubscribe_stops_the_tick() -> None:
    session = ResultsSession()
    ticks: list[int] = []
    callback = lambda: ticks.append(1)  # noqa: E731
    session.subscribe(callback)
    session.subscribe(callback)  # duplicate registration is one entry
    session.add_view()
    assert len(ticks) == 1
    session.unsubscribe(callback)
    session.add_view()
    assert len(ticks) == 1


# ----------------------------------------------------------------------
# Select → New plot copies the membership (§6/§8)
# ----------------------------------------------------------------------

def test_plot_from_selection_copies_node_membership() -> None:
    session = ResultsSession()
    session.selection.set_nodes([5, 7])
    plot = session.add_plot_from_selection("displacement_x")
    assert plot.kind == "history"
    assert plot.series == (
        PlotSeries(PlotSource.node(5), "displacement_x"),
        PlotSeries(PlotSource.node(7), "displacement_x"),
    )
    # Copy, not alias: mutating the selection later changes nothing.
    session.selection.set_nodes([9])
    assert [s.source.key for s in plot.series] == [5, 7]


def test_plot_from_selection_copies_gauss_membership() -> None:
    session = ResultsSession()
    session.selection.set_gauss([(3, 0), (3, 1)])
    plot = session.add_plot_from_selection("stress_von_mises")
    assert plot.series == (
        PlotSeries(PlotSource.gauss(3, 0), "stress_von_mises"),
        PlotSeries(PlotSource.gauss(3, 1), "stress_von_mises"),
    )


def test_plot_from_empty_selection_refuses() -> None:
    session = ResultsSession()
    with pytest.raises(ValueError, match="Selection is empty"):
        session.add_plot_from_selection("displacement_x")


# ----------------------------------------------------------------------
# Plot views (§6)
# ----------------------------------------------------------------------

def test_plot_kind_vocabulary_is_closed() -> None:
    session = ResultsSession()
    for kind in ("history", "path", "xy"):
        assert session.add_plot(kind=kind).kind == kind
    with pytest.raises(ValueError, match="kind"):
        session.add_plot(kind="scatter")


def test_plot_series_are_typed() -> None:
    session = ResultsSession()
    plot = session.add_plot()
    with pytest.raises(TypeError, match="PlotSeries"):
        plot.series = [("node", 5, "ux")]
    series = PlotSeries(PlotSource.label("crest"), "uz")
    plot.series = [series]
    assert plot.series == (series,)


def test_plot_source_factories_and_validation() -> None:
    assert PlotSource.node(5).key == 5
    assert PlotSource.gauss(7, 2).key == (7, 2)
    assert PlotSource.label("crest").key == "crest"
    assert PlotSource.physical_group("deck").key == "deck"
    with pytest.raises(ValueError, match="kind"):
        PlotSource("selection", "x")  # live alias is not a v1 source
    with pytest.raises(ValueError, match="name string"):
        PlotSource.label("")


# ----------------------------------------------------------------------
# Value records refuse garbage loudly
# ----------------------------------------------------------------------

def test_deform_validation() -> None:
    assert Deform().field == "displacement"  # the §4 default
    with pytest.raises(ValueError, match="Deform.field"):
        Deform(field="mode")
    with pytest.raises(TypeError, match="Deform.mode"):
        Deform(mode=True)


def test_scope_is_one_axis_plus_names_or_all() -> None:
    scope = Scope("physical_groups", names=["deck", "piers"])
    assert scope.names == ("deck", "piers")
    assert Scope("materials").names is None  # all names on the axis
    with pytest.raises(ValueError, match="Scope.axis"):
        Scope("labels")
    with pytest.raises(ValueError, match="Scope.names"):
        Scope("materials", names=[])


def test_view_field_types_are_checked() -> None:
    view = ResultsSession().add_view()
    with pytest.raises(TypeError, match="Scope"):
        view.scope = "deck"
    with pytest.raises(TypeError, match="Deform"):
        view.deform = 1.5
    with pytest.raises(TypeError, match="MeshStyle"):
        view.style = {"mesh": True}


# ----------------------------------------------------------------------
# View clips — 0083 field shape, view-owned (§3)
# ----------------------------------------------------------------------

def test_clip_lifecycle() -> None:
    view = ResultsSession().add_view()
    clip = view.add_clip((0.0, 0.0, 2.0), offset=1.5, name="waterline")
    assert clip.plane_id == "clip-1"
    assert clip.normal == (0.0, 0.0, 1.0)  # normalised
    assert clip.offset == 1.5
    updated = view.set_clip(clip.plane_id, flipped=True, offset=2.0)
    assert updated.flipped is True and updated.offset == 2.0
    assert view.clips == (updated,)
    view.remove_clip(clip.plane_id)
    assert view.clips == ()
    with pytest.raises(KeyError):
        view.remove_clip("clip-1")


def test_view_clip_record_is_a_real_value() -> None:
    # Probe H10: a directly-constructed record (the S5 restore path)
    # must not keep a caller's mutable list behind the frozen shell.
    from apeGmsh.results.session import ViewClip

    clip = ViewClip(plane_id="c1", name="p", normal=[3, 0, 0])
    assert clip.normal == (3.0, 0.0, 0.0)
    assert isinstance(clip.normal, tuple)
    with pytest.raises(ValueError):
        ViewClip(plane_id="c1", name="p", normal=(1.0, 0.0))  # not 3-D


def test_mesh_style_coerces_to_real_bools() -> None:
    # Probe H11: the S2 reconciler diffs records by equality — truthy
    # non-bools must not read as a change.
    assert MeshStyle(mesh="yes", nodes=1) == MeshStyle(mesh=True, nodes=True)


def test_clip_identity_is_not_settable() -> None:
    view = ResultsSession().add_view()
    clip = view.add_clip((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="identity"):
        view.set_clip(clip.plane_id, plane_id="clip-9")


def test_direct_view_construction_needs_no_session() -> None:
    # Views are constructible bare (unit tests, S1 realize fixtures).
    view = MeshView("mesh-x")
    view.contour = Contour("S")
    assert [legend.field for legend in view.legends()] == ["S"]
    plot = PlotView("plot-x", kind="path")
    plot.cursor = Instant("push", 0)
    assert plot.cursor == Instant("push", 0)
