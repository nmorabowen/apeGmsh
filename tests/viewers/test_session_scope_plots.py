"""ADR 0098 S1-B — scope resolution (§3/§9) and plot series (§6).

Scope: one composition axis into ONE cell set (INV-MESH-1). Every
layer of the view — the grey substrate included — is a function of it,
so the assertions here compare emitted cell/point counts against the
scoped subset, not just "it did not crash".

Plots: the resolver's oracle is direct equality with what
``results.plot.history`` plots — same query, same arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    Contour,
    Loads,
    PlotSeries,
    PlotSource,
    ResultsSession,
    Scope,
)
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.data import ViewerData
from apeGmsh.viewers.session import realize_pane, resolve_scope

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"
T = 3


@pytest.fixture
def two_body_results(g, tmp_path: Path):
    """Two disjoint boxes in separate physical groups.

    Two groups make the scope assertions real: a scoped view must emit
    strictly fewer cells than the whole mesh, and the two scopes must
    partition it.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
    g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="b")
    g.physical.add_volume("a", name="BodyA")
    g.physical.add_volume("b", name="BodyB")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    element_ids = np.concatenate(
        [np.asarray(grp.ids, dtype=np.int64) for grp in fem.elements]
    )
    disp = np.stack([node_ids + t * 100.0 for t in range(T)])
    gauss = np.zeros((T, element_ids.size, 1), dtype=np.float64)
    for t in range(T):
        gauss[t, :, 0] = element_ids + t

    path = tmp_path / "two_body.h5"
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
            element_index=element_ids,
            natural_coords=np.zeros((1, 3)),
            components={"stress_xx": gauss},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def one_body_recorded(g, tmp_path: Path):
    """Two groups, but the nodal recorder only covered BodyA — so
    BodyB exists in the model and has nothing recorded."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
    g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="b")
    g.physical.add_volume("a", name="BodyA")
    g.physical.add_volume("b", name="BodyB")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    covered = all_ids[coords[:, 0] <= 1.5]          # BodyA's half only
    disp = np.stack([covered + t * 100.0 for t in range(T)])

    path = tmp_path / "one_body.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(T, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=covered,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _count_node_reads(results, monkeypatch, key=None) -> list:
    """Record every ``results.<stage>.nodes.get`` call. Returns the
    live list — ``key`` records that kwarg instead of a bare marker."""
    accessor = type(results.stage(STAGE).nodes)
    real, reads = accessor.get, []
    monkeypatch.setattr(
        accessor, "get",
        lambda self, *a, **kw: (
            reads.append(1 if key is None else kw.get(key)),
            real(self, *a, **kw),
        )[1],
    )
    return reads


def _realize(results, mutate):
    session = results.session()
    view = session.panes[0]
    mutate(view)
    backend = RecordingBackend()
    return backend, realize_pane(session, view, backend)


# =====================================================================
# Scope resolution (§3 INV-MESH-1, §9)
# =====================================================================


def test_physical_group_axis_partitions_the_mesh(two_body_results):
    view_data = ViewerData.from_fem(two_body_results.fem)
    a = resolve_scope(Scope("physical_groups", ("BodyA",)), view_data)
    b = resolve_scope(Scope("physical_groups", ("BodyB",)), view_data)
    both = resolve_scope(
        Scope("physical_groups", ("BodyA", "BodyB")), view_data,
    )

    assert a.element_ids.size > 0 and b.element_ids.size > 0
    # Disjoint bodies partition the set, and the union is both.
    assert np.intersect1d(a.element_ids, b.element_ids).size == 0
    assert both.element_ids.size == a.element_ids.size + b.element_ids.size
    # Nodes come along with the cells they belong to.
    assert a.node_ids.size > 0
    assert np.intersect1d(a.node_ids, b.node_ids).size == 0


def test_names_none_means_every_name_on_the_axis(two_body_results):
    view_data = ViewerData.from_fem(two_body_results.fem)
    all_names = resolve_scope(Scope("physical_groups"), view_data)
    explicit = resolve_scope(
        Scope("physical_groups", ("BodyA", "BodyB")), view_data,
    )
    np.testing.assert_array_equal(
        all_names.element_ids, explicit.element_ids,
    )


def test_element_type_axis_resolves(two_body_results):
    view_data = ViewerData.from_fem(two_body_results.fem)
    scoped = resolve_scope(Scope("element_types", ("tet4",)), view_data)
    assert scoped.element_ids.size > 0


def test_unscoped_is_not_an_empty_set(two_body_results):
    view_data = ViewerData.from_fem(two_body_results.fem)
    scoped = resolve_scope(None, view_data)
    assert scoped.is_unscoped
    assert scoped.element_ids is None


def test_materials_axis_refuses_loudly(two_body_results):
    """S1-B decision 1: no element→material index is published, so the
    axis refuses rather than silently drawing an unscoped picture."""
    view_data = ViewerData.from_fem(two_body_results.fem)
    with pytest.raises(NotImplementedError, match="materials"):
        resolve_scope(Scope("materials", ("steel",)), view_data)


def test_unknown_scope_name_is_loud(two_body_results):
    view_data = ViewerData.from_fem(two_body_results.fem)
    with pytest.raises(ValueError, match="Unknown physical group"):
        resolve_scope(Scope("physical_groups", ("Nope",)), view_data)


# =====================================================================
# Scope realization — every layer is a function of the ONE cell set
# =====================================================================


def test_scoped_substrate_is_smaller_than_the_whole_mesh(two_body_results):
    whole, _ = _realize(two_body_results, lambda v: None)
    scoped, _ = _realize(
        two_body_results,
        lambda v: setattr(v, "scope", Scope("physical_groups", ("BodyA",))),
    )
    whole_cells = whole.layers["mesh-1:substrate"].cells.n_cells
    scoped_cells = scoped.layers["mesh-1:substrate"].cells.n_cells
    assert 0 < scoped_cells < whole_cells


def test_two_scopes_partition_the_substrate(two_body_results):
    a, _ = _realize(
        two_body_results,
        lambda v: setattr(v, "scope", Scope("physical_groups", ("BodyA",))),
    )
    b, _ = _realize(
        two_body_results,
        lambda v: setattr(v, "scope", Scope("physical_groups", ("BodyB",))),
    )
    whole, _ = _realize(two_body_results, lambda v: None)
    assert (
        a.layers["mesh-1:substrate"].cells.n_cells
        + b.layers["mesh-1:substrate"].cells.n_cells
        == whole.layers["mesh-1:substrate"].cells.n_cells
    )


def test_scoped_contour_paints_only_scoped_nodes(two_body_results):
    """A slot is a function of the cell set too (INV-MESH-1) — the
    contour must not paint the body the scope excluded."""
    view_data = ViewerData.from_fem(two_body_results.fem)
    scoped_set = resolve_scope(
        Scope("physical_groups", ("BodyA",)), view_data,
    )

    def mutate(v):
        v.scope = Scope("physical_groups", ("BodyA",))
        v.contour = Contour("displacement_z")

    backend, realized = _realize(two_body_results, mutate)
    (entry,) = [l for l in realized.layers if l.key.endswith(":contour")]
    values = np.asarray(
        backend.layers[entry.layer_id].field_named("displacement_z").values
    )
    # displacement_z at the default instant (t = T-1) is node_id + 200.
    painted_nodes = np.sort((values - (T - 1) * 100.0).round().astype(np.int64))
    np.testing.assert_array_equal(painted_nodes, np.sort(scoped_set.node_ids))


def test_scope_with_loads_refuses_loudly(two_body_results):
    """The loads kind ignores its selector's ids, so a scoped view
    would draw arrows outside its own cell set (INV-MESH-1)."""
    def mutate(v):
        v.scope = Scope("physical_groups", ("BodyA",))
        v.loads = Loads(pattern="P1")

    with pytest.raises(NotImplementedError, match="loads"):
        _realize(two_body_results, mutate)


# =====================================================================
# Plot series (§6) — the resolver's oracle is results.plot.history
# =====================================================================


def test_node_history_equals_results_plot_history(two_body_results):
    """Direct-equality oracle: the resolver and the matplotlib still
    run the same query, so their arrays must match exactly."""
    import matplotlib

    matplotlib.use("Agg")
    node_id = int(two_body_results.fem.nodes.ids[0])

    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    (series,) = realize_pane(session, plot).series

    ax = two_body_results.plot.history(
        node=node_id, component="displacement_z",
    )
    line = ax.lines[0]
    np.testing.assert_allclose(series.time, line.get_xdata())
    np.testing.assert_allclose(series.values, line.get_ydata())


def test_node_history_values_are_the_recorded_record(two_body_results):
    node_id = int(two_body_results.fem.nodes.ids[0])
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    (series,) = realize_pane(session, plot).series
    np.testing.assert_allclose(
        series.values, [node_id + t * 100.0 for t in range(T)],
    )
    assert series.time.shape == (T,)


def test_node_series_refuses_a_multi_column_read(
    two_body_results, monkeypatch,
):
    """Review finding: the resolver keeps the guard its oracle has.
    ``results.plot.history`` raises rather than index column 0 of a
    multi-column read, because that column is a different node than
    the one asked for."""
    node_id = int(two_body_results.fem.nodes.ids[0])
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])

    class _TwoColumnSlab:
        values = np.zeros((T, 2))
        time = np.arange(T, dtype=float)

    class _Scoped:
        class nodes:  # noqa: N801 - mimics the Results surface
            @staticmethod
            def get(**_kwargs):
                return _TwoColumnSlab()

    monkeypatch.setattr(
        type(two_body_results), "stage", lambda self, _sid: _Scoped(),
    )
    with pytest.raises(ValueError, match="columns"):
        realize_pane(session, plot)


def test_scoped_selector_ids_are_not_pre_converted(two_body_results):
    """Review finding (efficiency): the scoped id array is handed to
    the selector as-is — pre-building a Python tuple here would run
    the same O(N) conversion twice per realize."""
    from apeGmsh.viewers.session._scope import resolve_scope as _rs
    from apeGmsh.viewers.session._specs import _scope_ids

    view_data = ViewerData.from_fem(two_body_results.fem)
    scoped = _rs(Scope("physical_groups", ("BodyA",)), view_data)
    assert isinstance(_scope_ids(scoped, "nodes"), np.ndarray)
    assert _scope_ids(scoped, "none") is None


def test_gauss_history_resolves(two_body_results):
    element_id = int(
        np.concatenate([
            np.asarray(grp.ids) for grp in two_body_results.fem.elements
        ])[0]
    )
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.gauss(element_id, 0), "stress_xx"),
    ])
    (series,) = realize_pane(session, plot).series
    np.testing.assert_allclose(
        series.values, [element_id + t for t in range(T)],
    )


def test_several_series_are_one_plot(two_body_results):
    ids = [int(i) for i in two_body_results.fem.nodes.ids[:3]]
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(i), "displacement_z") for i in ids
    ])
    realized = realize_pane(session, plot)
    assert len(realized.series) == 3
    assert [s.label for s in realized.series] == [
        f"node {i} — displacement_z" for i in ids
    ]


def test_plot_cursor_follows_the_time_link(two_body_results):
    from apeGmsh.results.session import Instant

    node_id = int(two_body_results.fem.nodes.ids[0])
    session = two_body_results.session()
    session.time = Instant(STAGE, 1)
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    realized = realize_pane(session, plot)
    # The whole record is in the series; the cursor is where the
    # playhead sits on it (§7).
    assert realized.cursor == Instant(STAGE, 1)
    assert realized.series[0].values.shape == (T,)


def test_plot_pane_needs_no_backend(two_body_results):
    node_id = int(two_body_results.fem.nodes.ids[0])
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    assert realize_pane(session, plot).pane_id == plot.id


def test_unbound_session_plot_refuses():
    session = ResultsSession()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(1), "displacement_z"),
    ])
    with pytest.raises(RuntimeError, match="no Results bound"):
        realize_pane(session, plot)


@pytest.mark.parametrize("kind", ["path", "xy"])
def test_non_history_plot_kinds_refuse(two_body_results, kind):
    """Plan decision 10 (settled S4-2): path / xy stay refusals, and
    the refusal names the missing IR / authoring surface rather than a
    slice number, so it stays true whenever they land."""
    session = two_body_results.session()
    plot = session.add_plot(kind=kind)
    with pytest.raises(NotImplementedError, match="no v1 authoring"):
        realize_pane(session, plot)


# =====================================================================
# Membership sources (§6, plan decision 10b) — one curve per member
# =====================================================================


def test_physical_group_source_is_one_curve_per_member(two_body_results):
    """Decision 10b. A membership source resolves to exactly the curves
    the same nodes would give as concrete sources, in the slab's own
    column order — the rule ``add_plot_from_selection`` already uses."""
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "displacement_z"),
    ])
    realized = realize_pane(session, plot)

    members = two_body_results.stage(STAGE).nodes.get(
        pg="BodyA", component="displacement_z",
    )
    ids = [int(i) for i in members.node_ids]
    assert len(realized.series) == len(ids)
    assert [s.label for s in realized.series] == [
        f"node {i} — displacement_z" for i in ids
    ]
    # Each curve IS that node's record — the concrete-source oracle.
    for curve, node_id in zip(realized.series, ids):
        (one,) = realize_pane(session, session.add_plot(series=[
            PlotSeries(PlotSource.node(node_id), "displacement_z"),
        ])).series
        np.testing.assert_allclose(curve.values, one.values)
        np.testing.assert_allclose(curve.time, one.time)


def test_a_membership_source_costs_one_read_per_source(
    two_body_results, monkeypatch,
):
    """One curve per member must not mean one READ per member: ``pg=``
    is a first-class filter and a slab is already (T, members). The
    probe read that sizes the membership is the only second read."""
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "displacement_z"),
    ])
    reads = _count_node_reads(two_body_results, monkeypatch)
    realized = realize_pane(session, plot)
    assert len(realized.series) > 2, "the rig resolved too few members"
    assert len(reads) == 2, f"{len(reads)} reads for one membership source"


def test_the_gauss_half_of_a_membership_source_expands(two_body_results):
    """The quantity picks the topology: a Gauss component means the
    group's integration points, addressed the §8 way."""
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "stress_xx"),
    ])
    realized = realize_pane(session, plot)
    slab = two_body_results.stage(STAGE).elements.gauss.get(
        pg="BodyA", component="stress_xx",
    )
    assert len(realized.series) == int(np.asarray(slab.element_index).size)
    assert realized.series[0].label == (
        f"element {int(slab.element_index[0])} gp 0 — stress_xx"
    )
    np.testing.assert_allclose(
        realized.series[0].values, np.asarray(slab.values)[:, 0],
    )


def test_a_quantity_recorded_at_neither_level_refuses(two_body_results):
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "no_such_thing"),
    ])
    with pytest.raises(ValueError, match="recorded at neither"):
        realize_pane(session, plot)


def test_a_quantity_recorded_at_both_levels_refuses(
    two_body_results, monkeypatch,
):
    """A membership source cannot say which topology it means when the
    token exists at both — it refuses rather than picking one."""
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "displacement_z"),
    ])
    monkeypatch.setattr(
        type(two_body_results.inspect), "components",
        lambda self, **_kw: {
            "nodes": ["displacement_z"], "gauss": ["displacement_z"],
        },
    )
    with pytest.raises(ValueError, match="BOTH"):
        realize_pane(session, plot)


def test_an_unknown_membership_name_is_loud(two_body_results):
    """The query layer already refuses a name the model does not have;
    the resolver must not soften that into an empty chart."""
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("nothing-here"),
                   "displacement_z"),
    ])
    with pytest.raises(KeyError, match="nothing-here"):
        realize_pane(session, plot)


def test_a_group_with_no_recorded_members_refuses(one_body_recorded):
    """A group that EXISTS but that the recorder never covered reads
    back as zero columns — the reader intersects silently, so the
    refusal has to happen here. An empty chart would read as "this
    group did not move"."""
    session = one_body_recorded.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyB"), "displacement_z"),
    ])
    with pytest.raises(ValueError, match="no members recording"):
        realize_pane(session, plot)


# =====================================================================
# The curve cap — it must refuse BEFORE reading (decision 10b)
# =====================================================================


def test_a_selection_sized_plot_refuses_before_it_reads(
    two_body_results, monkeypatch,
):
    """"Select all nodes" then "New plot from selection" is one gesture
    away from handing this resolver the whole model, as one concrete
    source PER NODE. The cap has to fire before the reads, not after
    them: the refusal is cheap, the N reads are what hang. The cap
    VALUE is a judgement call, so the test pins the behaviour."""
    from apeGmsh.viewers.session import _plots

    monkeypatch.setattr(_plots, "MAX_SERIES", 4)
    node_ids = [int(i) for i in two_body_results.fem.nodes.ids]
    session = two_body_results.session()
    session.selection.set_nodes(node_ids)
    plot = session.add_plot_from_selection("displacement_z")

    reads = _count_node_reads(two_body_results, monkeypatch)
    with pytest.raises(ValueError, match="the cap is"):
        realize_pane(session, plot)
    assert reads == [], f"the cap fired after {len(reads)} read(s)"


def test_an_over_cap_membership_refuses_after_one_probe(
    two_body_results, monkeypatch,
):
    """A membership size is not known without asking, so ONE
    single-step probe read is the price — never the whole record,
    which for a select-all-sized group is what runs the machine out
    of memory."""
    from apeGmsh.viewers.session import _plots

    monkeypatch.setattr(_plots, "MAX_SERIES", 1)
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "displacement_z"),
    ])
    reads = _count_node_reads(two_body_results, monkeypatch, key="time")
    with pytest.raises(ValueError, match="the cap is"):
        realize_pane(session, plot)
    assert reads == [[0]], f"over-cap membership read {reads}"


def test_the_cap_clears_a_named_physical_group(two_body_results):
    """The cap must not make §9's select-all useless: one named group
    of a real model is what it exists to plot."""
    from apeGmsh.viewers.session._plots import MAX_SERIES

    members = two_body_results.stage(STAGE).nodes.get(
        pg="BodyA", component="displacement_z",
    )
    assert 1 < int(np.asarray(members.node_ids).size) <= MAX_SERIES


def test_plot_pane_still_refuses(two_body_results, tmp_path):
    """A plot pane has no offscreen still yet (§10: the matplotlib
    still of the same queries is results.plot)."""
    node_id = int(two_body_results.fem.nodes.ids[0])
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.node(node_id), "displacement_z"),
    ])
    with pytest.raises(NotImplementedError, match="mesh panes"):
        session.render(tmp_path / "plot.png", plot.id)
