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
    session = two_body_results.session()
    plot = session.add_plot(kind=kind)
    with pytest.raises(NotImplementedError, match="S4-2"):
        realize_pane(session, plot)


def test_membership_sources_refuse_pending_aggregation_rule(two_body_results):
    session = two_body_results.session()
    plot = session.add_plot(series=[
        PlotSeries(PlotSource.physical_group("BodyA"), "displacement_z"),
    ])
    with pytest.raises(NotImplementedError, match="aggregation rule"):
        realize_pane(session, plot)


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
