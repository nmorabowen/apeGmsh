"""ADR 0098 S1-A — realize(): mesh pane into a RecordingBackend.

Primary oracle for the slice (no GL, runs on CI): layer assertions
against the emitted scene IR, straight from the ADR laws —

* deform on, every slot empty → warped mesh, ZERO scalar bars (§5);
* the contour slot replaces the grey substrate (INV-MESH-2);
* averaged vs unaveraged contour route to distinct emissions (§4);
* the bars on the backend match ``view.legends()`` exactly, hide is
  chrome (INV-LEGEND-1..5);
* pose parity with ``render_results``'s composition (S1 decision 5 —
  warp-before-extract emits the same points attach-then-sync does).

Oracles are paired positive/negative (substrate present ↔ replaced,
zero bars ↔ exactly one, visible ↔ hidden) so each assertion is
mutation-tested by its complement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    Contour,
    Deform,
    Instant,
    MeshStyle,
    ResultsSession,
    Scope,
    Vector,
)
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers._pump_set import (
    _compose_substrate_points,
    read_nodal_vector_field,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene
from apeGmsh.viewers.session import realize_pane

from tests.conftest import _open_model_from_h5

STAGE = "grav"


# =====================================================================
# Fixture — one static stage (nodal + 1-GP gauss) after one mode stage
# =====================================================================


def _all_element_ids(fem) -> np.ndarray:
    chunks = [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    return (
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    )


@pytest.fixture
def session_results(g, tmp_path: Path):
    """Native HDF5 with predictable data.

    Stage order is ``[mode_1, grav]`` so the *default* instant (last
    stage, last step) lands on the static stage.

    * grav, step ``t``, node ``nid``: ``displacement_z = nid + t*1000``
    * grav, step ``t``, element ``eid`` (one GP): ``stress_xx = eid*10 + t``
    * mode_1 (kind='mode', mode_index=1), node ``nid``:
      ``displacement_z = 2*nid`` (the shape)
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = _all_element_ids(fem)
    n_steps = 4

    disp = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    for t in range(n_steps):
        disp[t] = node_ids + t * 1000.0
    sxx = np.zeros((n_steps, elem_ids.size, 1), dtype=np.float64)
    for t in range(n_steps):
        sxx[t, :, 0] = elem_ids * 10.0 + t
    shape = (2.0 * node_ids).astype(np.float64)[None, :]

    path = tmp_path / "session_s1a.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="mode_1", kind="mode",
            time=np.zeros(1, dtype=np.float64),
            eigenvalue=4.0, frequency_hz=0.318, period_s=3.14,
            mode_index=1,
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": shape},
        )
        w.end_stage()
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
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


def _session_view(results):
    session = results.session()
    return session, session.panes[0]


def _layer_keys(realized):
    return [layer.key for layer in realized.layers]


def _contour_layer(backend, realized):
    (entry,) = [l for l in realized.layers if l.key.endswith(":contour")]
    return backend.layers[entry.layer_id]


# =====================================================================
# Factory (§1 — default session = one empty mesh view)
# =====================================================================


def test_factory_default_session(session_results):
    session = session_results.session()
    assert isinstance(session, ResultsSession)
    assert session.results is session_results
    (pane,) = session.panes
    assert pane.id == "mesh-1"
    assert pane.slots == {}
    assert pane.legends() == ()


# =====================================================================
# Boot picture + INV-MESH-2
# =====================================================================


def test_empty_view_realizes_the_boot_picture(session_results, backend):
    """§3: the analysis mesh, grey, mesh + outlines on, nodes/Gauss
    off, no legend. Two layers since S3 — the interior grid is the
    surface's ``show_edges``, the feature boundaries are their own
    layer (INV-MESH-4)."""
    session, view = _session_view(session_results)
    realized = realize_pane(session, view, backend)

    assert _layer_keys(realized) == ["mesh-1:substrate", "mesh-1:outlines"]
    layer = backend.layers["mesh-1:substrate"]
    assert layer.color.mode == "solid"
    assert layer.show_edges is True
    assert backend.scalar_bars == {}
    assert realized.scalar_bars == ()
    # Undeformed: the substrate sits at the reference configuration.
    scene = build_fem_scene(session_results.fem)
    np.testing.assert_allclose(
        np.asarray(layer.points.coords, dtype=np.float64),
        np.asarray(scene.reference_points),
        atol=1e-5,
    )


def test_contour_replaces_substrate(session_results, backend):
    """INV-MESH-2: one surface — the contour, never grey under it."""
    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    realized = realize_pane(session, view, backend)

    assert _layer_keys(realized) == ["mesh-1:contour", "mesh-1:outlines"]
    assert not any(k.endswith(":substrate") for k in backend.layers)
    layer = _contour_layer(backend, realized)
    assert layer.color.mode == "by_array"
    # Default instant = last stage, last step (t=3): values nid + 3000.
    field = layer.field_named("displacement_z")
    assert field is not None and field.location == "point"
    values = np.asarray(field.values)
    node_ids = np.sort(
        np.asarray(session_results.fem.nodes.ids, dtype=np.int64)
    )
    np.testing.assert_allclose(
        np.sort(values), node_ids + 3000.0, atol=1e-9,
    )


# =====================================================================
# §5 — legends caused by slots, never by the pose
# =====================================================================


def test_deform_on_slots_empty_zero_bars(session_results, backend):
    """The ADR §5 oracle verbatim: warped mesh, ZERO legends."""
    session, view = _session_view(session_results)
    view.deform = Deform(scale=2.5)
    realized = realize_pane(session, view, backend)

    assert backend.scalar_bars == {}
    assert realized.scalar_bars == ()
    assert realized.legend_controller is None
    # ... and the mesh really is warped: composition parity with the
    # render_results pose (read_nodal_vector_field + compose).
    scene = build_fem_scene(session_results.fem)
    vals = read_nodal_vector_field(
        session_results.stage(STAGE), scene, "displacement", 3,
    )
    expected = _compose_substrate_points(
        scene.reference_points, None, vals, 2.5,
    )
    layer = backend.layers["mesh-1:substrate"]
    np.testing.assert_allclose(
        np.asarray(layer.points.coords, dtype=np.float64), expected,
        rtol=1e-6,
    )


def test_contour_emits_exactly_one_bar(session_results, backend):
    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    realized = realize_pane(session, view, backend)

    assert list(backend.scalar_bars) == ["displacement_z"]
    assert realized.scalar_bars == ("displacement_z",)
    spec = backend.scalar_bars["displacement_z"]
    # INV-LEGEND-4: the scale names the slot quantity.
    assert spec.title == "displacement_z"
    # The scale spans the whole history (visual-store colour limits),
    # not step 0's range: displacement_z runs [min_nid, max_nid + 3000].
    node_ids = np.asarray(session_results.fem.nodes.ids, dtype=np.float64)
    assert spec.lut.vmin == pytest.approx(node_ids.min())
    assert spec.lut.vmax == pytest.approx(node_ids.max() + 3000.0)


def test_bar_range_survives_lut_mirror_failure(
    session_results, backend, monkeypatch,
):
    """Probe P10 regression: the legend range comes from the emitted
    layer's ColorSpec, not the Qt LUT mirror — with ``_init_lut``
    unable to build (Qt-less environment), the bar must still span the
    painted field, never the default 0–1 scale."""
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    monkeypatch.setattr(ContourDiagram, "_init_lut", lambda self: None)
    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    realize_pane(session, view, backend)

    spec = backend.scalar_bars["displacement_z"]
    node_ids = np.asarray(session_results.fem.nodes.ids, dtype=np.float64)
    assert spec.lut.vmin == pytest.approx(node_ids.min())
    assert spec.lut.vmax == pytest.approx(node_ids.max() + 3000.0)


def test_hidden_legend_emits_no_bar_keeps_picture(session_results, backend):
    """INV-LEGEND-3 (hide is chrome) — and the decision-4 mutation
    test: if the null controller failed to suppress the diagram-side
    registration, the contour's own ``show_scalar_bar=True`` register
    would put a bar on the backend and this test fails."""
    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    view.set_legend_hidden("displacement_z")
    realized = realize_pane(session, view, backend)

    assert backend.scalar_bars == {}
    assert realized.scalar_bars == ()
    # The picture stays painted.
    assert _layer_keys(realized) == ["mesh-1:contour", "mesh-1:outlines"]
    assert _contour_layer(backend, realized).color.mode == "by_array"


def test_bars_match_session_legends(session_results, backend):
    """The realize legend law is ``view.legends()`` — nothing else."""
    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    view.deform = Deform()          # pose must not add a scale
    realized = realize_pane(session, view, backend)

    expected = [
        legend.field for legend in view.legends() if not legend.hidden
    ]
    assert list(backend.scalar_bars) == expected
    assert list(realized.scalar_bars) == expected


# =====================================================================
# §4 — averaged vs unaveraged inside the contour slot
# =====================================================================


def test_contour_gauss_averaged_vs_unaveraged(session_results):
    from tests.viewers.conftest import RecordingBackend

    session, view = _session_view(session_results)

    view.contour = Contour("stress_xx", averaging="averaged")
    averaged_backend = RecordingBackend()
    averaged = realize_pane(session, view, averaged_backend)
    averaged_field = _contour_layer(
        averaged_backend, averaged,
    ).field_named("stress_xx")
    assert averaged_field is not None

    view.contour = Contour("stress_xx", averaging="unaveraged")
    discrete_backend = RecordingBackend()
    discrete = realize_pane(session, view, discrete_backend)
    discrete_field = _contour_layer(
        discrete_backend, discrete,
    ).field_named("stress_xx")
    assert discrete_field is not None

    # One GP per element: averaged spreads to corners and averages
    # across neighbours (point data); unaveraged paints flat cell data.
    assert averaged_field.location == "point"
    assert discrete_field.location == "cell"
    # Unaveraged cell values are exactly eid*10 + t at the default
    # instant (t=3).
    elem_ids = _all_element_ids(session_results.fem)
    np.testing.assert_allclose(
        np.sort(np.asarray(discrete_field.values)),
        np.sort(elem_ids * 10.0 + 3.0),
        atol=1e-9,
    )
    # Both are colour-mapped → each backend carries exactly one scale.
    assert list(averaged_backend.scalar_bars) == ["stress_xx"]
    assert list(discrete_backend.scalar_bars) == ["stress_xx"]


def test_contour_nodal_ignores_averaging(session_results):
    """§4 semantic pin: a NODAL quantity has no unaveraged form —
    nodal slabs already carry one value per node — so both averaging
    tokens emit the identical point-located layer."""
    from tests.viewers.conftest import RecordingBackend

    session, view = _session_view(session_results)
    emitted = {}
    for averaging in ("averaged", "unaveraged"):
        view.contour = Contour("displacement_z", averaging=averaging)
        backend = RecordingBackend()
        realized = realize_pane(session, view, backend)
        emitted[averaging] = _contour_layer(backend, realized)
    assert emitted["averaged"].field_named("displacement_z").location == "point"
    assert emitted["unaveraged"].field_named("displacement_z").location == "point"
    np.testing.assert_array_equal(
        emitted["averaged"].field_named("displacement_z").values,
        emitted["unaveraged"].field_named("displacement_z").values,
    )


# =====================================================================
# Decision 5 — pose ordering parity (warp-before-extract ==
# render_results' attach-then-sync)
# =====================================================================


def test_contour_deform_pose_parity(session_results, backend):
    """The contour's emitted points equal the composed pose sampled at
    each submesh point's substrate row — exactly what
    ``render_results``'s ``_apply_deform`` + ``sync_substrate_points``
    produce. Rows are recovered from the painted values (node ids)."""
    session, view = _session_view(session_results)
    scale = 2.5
    step = 3
    view.contour = Contour("displacement_z")
    view.deform = Deform(scale=scale)
    realized = realize_pane(session, view, backend)

    layer = _contour_layer(backend, realized)
    values = np.asarray(layer.field_named("displacement_z").values)
    submesh_node_ids = (values - step * 1000.0).round().astype(np.int64)

    scene = build_fem_scene(session_results.fem)
    vals = read_nodal_vector_field(
        session_results.stage(STAGE), scene, "displacement", step,
    )
    expected_pts = _compose_substrate_points(
        scene.reference_points, None, vals, scale,
    )
    rows = np.asarray(
        [scene.node_id_to_idx[int(n)] for n in submesh_node_ids],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        np.asarray(layer.points.coords, dtype=np.float64),
        expected_pts[rows],
        rtol=1e-6,
    )


def test_deform_auto_scale_rule(session_results, backend):
    """``Deform(scale=None)`` auto-fits with render.py's 0.12 rule."""
    session, view = _session_view(session_results)
    view.deform = Deform()
    realize_pane(session, view, backend)

    scene = build_fem_scene(session_results.fem)
    layer = backend.layers["mesh-1:substrate"]
    offsets = np.linalg.norm(
        np.asarray(layer.points.coords, dtype=np.float64)
        - np.asarray(scene.reference_points),
        axis=1,
    )
    assert offsets.max() == pytest.approx(
        0.12 * float(scene.model_diagonal), rel=1e-5,
    )


# =====================================================================
# §7 — realize consumes the linked-instant law
# =====================================================================


def test_linked_session_time_selects_step(session_results, backend):
    session, view = _session_view(session_results)
    session.time = Instant(STAGE, 1)
    view.contour = Contour("displacement_z")
    realized = realize_pane(session, view, backend)
    values = np.asarray(
        _contour_layer(backend, realized).field_named(
            "displacement_z"
        ).values
    )
    node_ids = np.sort(
        np.asarray(session_results.fem.nodes.ids, dtype=np.int64)
    )
    np.testing.assert_allclose(np.sort(values), node_ids + 1000.0)


def test_unlinked_pane_time_wins(session_results, backend):
    session, view = _session_view(session_results)
    session.time_linked = False
    view.time = Instant(STAGE, 2)
    view.contour = Contour("displacement_z")
    realized = realize_pane(session, view, backend)
    values = np.asarray(
        _contour_layer(backend, realized).field_named(
            "displacement_z"
        ).values
    )
    node_ids = np.sort(
        np.asarray(session_results.fem.nodes.ids, dtype=np.int64)
    )
    np.testing.assert_allclose(np.sort(values), node_ids + 2000.0)


def test_instant_out_of_range_is_loud(session_results, backend):
    session, view = _session_view(session_results)
    session.time = Instant(STAGE, 99)
    with pytest.raises(ValueError, match="out of range"):
        realize_pane(session, view, backend)


# =====================================================================
# Mode pose (§4 — a pose, no instant, frozen under the link)
# =====================================================================


def test_mode_pose_realizes_shape_no_bars(session_results, backend):
    session, view = _session_view(session_results)
    # A mode pose ignores the session instant (frozen under the link).
    session.time = Instant(STAGE, 0)
    view.deform = Deform(scale=1.0, mode=1)
    realized = realize_pane(session, view, backend)

    assert backend.scalar_bars == {}
    assert realized.scalar_bars == ()
    scene = build_fem_scene(session_results.fem)
    layer = backend.layers["mesh-1:substrate"]
    offsets = (
        np.asarray(layer.points.coords, dtype=np.float64)
        - np.asarray(scene.reference_points)
    )
    # The shape is displacement_z = 2*nid at scale 1.
    node_ids = np.asarray(scene.node_ids, dtype=np.float64)
    # PointSet pins coords to float32 — tolerances sized accordingly.
    np.testing.assert_allclose(offsets[:, 2], 2.0 * node_ids, rtol=1e-6)
    np.testing.assert_allclose(offsets[:, :2], 0.0, atol=1e-4)


def test_mode_pose_unknown_mode_is_loud(session_results, backend):
    session, view = _session_view(session_results)
    view.deform = Deform(mode=7)
    with pytest.raises(ValueError, match="mode"):
        realize_pane(session, view, backend)


# =====================================================================
# The S2 contract — stable layer keys, and realize never mutates
# =====================================================================


def test_realize_fires_no_session_tick(session_results, backend):
    """realize() is a pure read of the IR. A realize that ticked the
    session would loop the S2 reconciler (diff → realize → tick →
    diff …), so zero ticks is part of the S1 contract."""
    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    view.deform = Deform()
    ticks = []
    session.subscribe(lambda: ticks.append(1))
    realize_pane(session, view, backend)
    assert ticks == []


def test_layer_keys_stable_across_realizations(session_results):
    from tests.viewers.conftest import RecordingBackend

    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")
    first = realize_pane(session, view, RecordingBackend())
    second = realize_pane(session, view, RecordingBackend())
    assert _layer_keys(first) == _layer_keys(second) == [
        "mesh-1:contour", "mesh-1:outlines",
    ]


# =====================================================================
# Loud refusals — no lying pictures
# =====================================================================


def test_unbound_session_refuses(backend):
    session = ResultsSession()
    view = session.add_view()
    with pytest.raises(RuntimeError, match="no Results bound"):
        realize_pane(session, view, backend)


def test_unknown_quantity_is_loud(session_results, backend):
    session, view = _session_view(session_results)
    view.contour = Contour("nope")
    with pytest.raises(ValueError, match="not recorded"):
        realize_pane(session, view, backend)


def test_foreign_pane_refuses(session_results, backend):
    session, _ = _session_view(session_results)
    other = ResultsSession(results=session_results)
    foreign = other.add_view()
    with pytest.raises((KeyError, ValueError)):
        realize_pane(session, foreign, backend)


def test_realize_rejects_a_non_pane(session_results, backend):
    session, _ = _session_view(session_results)
    with pytest.raises(TypeError, match="MeshView or PlotView"):
        realize_pane(session, object(), backend)


def test_mesh_pane_without_backend_refuses(session_results):
    session, view = _session_view(session_results)
    with pytest.raises(TypeError, match="RenderBackend"):
        realize_pane(session, view)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda v: setattr(v, "overlay", True), "overlay"),
    ],
    ids=["overlay"],
)
def test_unrealized_state_refuses_loudly(
    session_results, backend, mutate, match,
):
    """A still that silently drops session state is a wrong picture.

    The style buttons and the view clips left this list at S3 — they
    realize now (see the INV-MESH-4 and clip tests below).
    """
    session, view = _session_view(session_results)
    mutate(view)
    with pytest.raises(NotImplementedError, match=match):
        realize_pane(session, view, backend)


# =====================================================================
# ADR 0098 A4.1 Bug B — ONE visual store per Results, not per realize
# =====================================================================


def test_realize_shares_one_visual_store_per_results(session_results, backend):
    """Realize used to construct a ``VisualDataStore`` per CALL.

    That store caches full-time slabs so scrubbing reads RAM instead of
    HDF5 — a per-call store is therefore a cache that is empty on every
    frame and dies with it, and since ``ContourDiagram.attach`` pre-warms
    it, every flush paid a full ``(T, N)`` read and threw it away. The
    old viewer's rule is one store per ``Results``; this pins it.
    """
    from apeGmsh.viewers.session._realize import visual_store_for

    session, view = _session_view(session_results)
    view.contour = Contour("displacement_z")

    first = realize_pane(session, view, backend)
    second = realize_pane(session, view, backend)

    stores = {
        id(d._visual_store)  # noqa: SLF001
        for d in (*first.diagrams, *second.diagrams)
        if getattr(d, "_visual_store", None) is not None
    }
    assert len(stores) == 1, "a realize handed its diagrams a fresh store"
    assert stores == {id(visual_store_for(session_results))}


def test_a_second_results_gets_its_own_store(session_results, backend):
    """Weak-keyed per ``Results``: two documents never share a cache."""
    from apeGmsh.viewers.session._realize import visual_store_for

    class _Other:
        pass

    other = _Other()
    assert visual_store_for(session_results) is visual_store_for(
        session_results,
    )
    assert visual_store_for(other) is not visual_store_for(session_results)


def test_the_store_does_not_outlive_its_results(backend):
    """The registry is weak-keyed, so a dropped ``Results`` takes its
    cached slabs with it — this is a cache, not a leak."""
    import gc

    from apeGmsh.viewers.session._realize import _STORES, visual_store_for

    class _Doomed:
        pass

    def _mine() -> int:
        # Count only OUR key: other tests' Results are weak keys too and
        # may be collected at any point, so a total count is not a test.
        return sum(1 for k in list(_STORES.keys()) if isinstance(k, _Doomed))

    doomed = _Doomed()
    visual_store_for(doomed)
    assert _mine() == 1

    del doomed
    gc.collect()
    assert _mine() == 0, "the store outlived its Results"


# =====================================================================
# ADR 0098 A4.2 — the row maps realize records for a cursor re-step
# =====================================================================


@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
def test_recorded_rows_reproduce_the_emitted_layers(
    session_results, backend, scoped,
):
    """A re-step re-points these layers as ``source_points[rows]``.

    If the rows do not reproduce the layer realize actually emitted,
    the fast path draws a different picture from the slow one — so
    pin the identity here, where it is cheap, rather than discovering
    it as a parity failure later.
    """
    session, view = _session_view(session_results)
    view.style = MeshStyle(mesh=True, outlines=True, nodes=True, gauss=False)
    if scoped:
        view.scope = Scope("physical_groups", ("Body",))

    realized = realize_pane(session, view, backend)
    rs = realized.restep
    assert rs is not None, "realize recorded no re-step context"
    by_key = {layer.key: layer for layer in realized.layers}

    scene_pts = np.asarray(rs.scene.grid.points)
    grid_pts = np.asarray(rs.grid.points)

    # The scoped grid is a COPY; its rows must map back onto the scene.
    if scoped:
        assert rs.substrate_rows is not None
        np.testing.assert_allclose(
            grid_pts, scene_pts[rs.substrate_rows], atol=1e-9,
        )
    else:
        assert rs.substrate_rows is None
        assert rs.grid is rs.scene.grid

    outlines = by_key[f"{view.id}:outlines"]
    assert outlines.rows is not None, "no outline row map"
    np.testing.assert_allclose(
        np.asarray(outlines.layer.points.coords),
        grid_pts[outlines.rows],
        atol=1e-9,
        err_msg="outline rows do not reproduce the emitted outline",
    )

    nodes = by_key[f"{view.id}:nodes"]
    assert nodes.rows is not None, "no node row map"
    np.testing.assert_allclose(
        np.asarray(nodes.layer.points.coords),
        scene_pts[nodes.rows],
        atol=1e-9,
        err_msg="node rows do not reproduce the emitted node cloud",
    )


def test_the_outline_row_tag_never_survives_the_extraction(
    session_results, backend,
):
    """The row map rides a point-data tag through two extraction hops.

    Unscoped, the grid it is written onto IS ``scene.grid`` — a leaked
    tag would then be copied by every later extraction and ride into
    layer arrays. It is deleted in a ``finally``; this is the guard.
    """
    from apeGmsh.viewers.session._realize import _OUTLINE_ROWS

    session, view = _session_view(session_results)
    view.style = MeshStyle(mesh=True, outlines=True, nodes=False, gauss=False)
    realized = realize_pane(session, view, backend)

    rs = realized.restep
    assert _OUTLINE_ROWS not in rs.scene.grid.point_data
    assert _OUTLINE_ROWS not in rs.grid.point_data
