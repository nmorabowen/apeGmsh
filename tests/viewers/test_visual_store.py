"""VisualDataStore - eager float32 cache for the post-solve viewer.

Covers the pure-visual performance layer introduced to stop the time
scrubber re-reading the full (T, N) HDF5 dataset every frame:
  * load_stage materializes every node + gauss component as float32
    and records per-component (vmin, vmax) in the same pass;
  * the byte budget gates only the eager pre-fetch (lazy access still
    serves a live request past the cap);
  * ContourDiagram slices the cached float32 row on update_to_step
    and falls back to the per-step read path when no store is stamped.

Headless: uses NativeWriter + an offscreen plotter, no Qt window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._visual_store import VisualDataStore
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


# ---------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------
@pytest.fixture
def results_with_known_disp(g, tmp_path: Path):
    """Native HDF5 with displacement_z = nid + t * 1000 per (node, step)."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_nodes = node_ids.size
    n_steps = 4

    values = np.zeros((n_steps, n_nodes), dtype=np.float64)
    for t in range(n_steps):
        values[t] = node_ids + t * 1000.0

    path = tmp_path / "known_disp.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": values},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _stage_id(results) -> str:
    return results.stages[0].id


def _make_spec(component="displacement_z") -> DiagramSpec:
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(),
    )


# ---------------------------------------------------------------------
# Store unit tests
# ---------------------------------------------------------------------
def test_load_stage_materializes_float32_and_clim(results_with_known_disp):
    store = VisualDataStore()
    sid = _stage_id(results_with_known_disp)
    store.load_stage(results_with_known_disp, sid)

    slab = store.nodes_slab(results_with_known_disp.stage(sid), sid, "displacement_z")
    assert slab is not None
    # float32 resident (the whole point: half-width, slice-not-read).
    assert slab.values.dtype == np.float32
    assert slab.values.shape == (4, len(results_with_known_disp.fem.nodes.ids))

    # color limits match the global finite min/max of the float32 slab.
    clim = store.color_limits(sid, "displacement_z")
    assert clim is not None
    finite = slab.values[np.isfinite(slab.values)]
    assert clim == (float(finite.min()), float(finite.max()))


def test_byte_budget_gates_eager_but_lazy_still_serves(results_with_known_disp):
    # A zero ceiling blocks eager pre-fetch entirely.
    store = VisualDataStore(byte_budget=0)
    sid = _stage_id(results_with_known_disp)
    store.load_stage(results_with_known_disp, sid)
    assert store.loaded_bytes == 0

    # A live request still loads (the cap never refuses a render).
    slab = store.nodes_slab(results_with_known_disp.stage(sid), sid, "displacement_z")
    assert slab is not None
    assert slab.values.dtype == np.float32
    assert store.loaded_bytes > 0


def test_invalidate_stage_drops_only_that_stage(results_with_known_disp):
    store = VisualDataStore()
    sid = _stage_id(results_with_known_disp)
    store.load_stage(results_with_known_disp, sid)
    assert store.loaded_bytes > 0
    store.invalidate_stage(sid)
    assert store.loaded_bytes == 0
    assert store.color_limits(sid, "displacement_z") is None


def test_missing_component_returns_none(results_with_known_disp):
    store = VisualDataStore()
    sid = _stage_id(results_with_known_disp)
    assert store.nodes_slab(results_with_known_disp.stage(sid), sid, "nope") is None


# ---------------------------------------------------------------------
# Regression: large-magnitude demands (stress in Pa) must NOT overflow.
#
# The cache originally cast to float16, whose finite ceiling is 65504.
# SI stress/force (concrete ~3e7, steel ~2.5e8) overflows to +inf, which
# both corrupts the cached row AND collapses color_limits to None (all
# values non-finite). float32 (range ~3.4e38) keeps them finite. These
# tests pin that the cache stays finite on the default node AND gauss
# paths at realistic stress magnitude — the exact case the float16/3030
# happy-path fixture above cannot reach.
# ---------------------------------------------------------------------
_STRESS_PA = 2.5e8    # ~ steel yield in Pa — comfortably over the float16 cap


def _all_element_ids(fem):
    chunks = [np.asarray(grp.ids, dtype=np.int64) for grp in fem.elements]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)


@pytest.fixture
def results_with_stress_magnitude(g, tmp_path: Path):
    """Native HDF5 carrying a node AND a gauss component at ~2.5e8 Pa."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = _all_element_ids(fem)
    n_steps = 3

    # Node component spanning up to ~2.5e8 (linear ramp so min != max).
    pnode = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    for t in range(n_steps):
        pnode[t] = np.linspace(_STRESS_PA * 0.5, _STRESS_PA, node_ids.size) + t

    # Gauss component (1 GP / element) at the same magnitude.
    sxx = np.zeros((n_steps, elem_ids.size, 1), dtype=np.float64)
    for t in range(n_steps):
        sxx[t, :, 0] = np.linspace(_STRESS_PA * 0.5, _STRESS_PA, elem_ids.size) + t
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    path = tmp_path / "stress.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="push", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"pressure": pnode},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def test_node_stress_stays_finite_and_clim_resolves(results_with_stress_magnitude):
    store = VisualDataStore()
    sid = _stage_id(results_with_stress_magnitude)
    scoped = results_with_stress_magnitude.stage(sid)

    slab = store.nodes_slab(scoped, sid, "pressure")
    assert slab is not None
    # float16 would have stamped +inf here (2.5e8 > 65504).
    assert np.all(np.isfinite(slab.values)), "large node demand overflowed the cache"
    clim = store.color_limits(sid, "pressure")
    assert clim is not None, "color_limits collapsed to None (all-inf cache)"
    lo, hi = clim
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    # float32 round-trip keeps ~6-7 sig figs at this magnitude.
    np.testing.assert_allclose(hi, _STRESS_PA, rtol=1e-4)


def test_gauss_stress_stays_finite_and_clim_resolves(results_with_stress_magnitude):
    store = VisualDataStore()
    sid = _stage_id(results_with_stress_magnitude)
    scoped = results_with_stress_magnitude.stage(sid)

    slab = store.gauss_slab(scoped, sid, "stress_xx")
    assert slab is not None
    assert np.all(np.isfinite(slab.values)), "large gauss demand overflowed the cache"
    clim = store.color_limits(sid, "stress_xx")
    assert clim is not None
    lo, hi = clim
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


# ---------------------------------------------------------------------
# Memory budget: LRU eviction bounds resident bytes without refusing a
# live request.
# ---------------------------------------------------------------------
def test_byte_budget_evicts_lru_keeps_live_request(results_with_stress_magnitude):
    sid = _stage_id(results_with_stress_magnitude)
    scoped = results_with_stress_magnitude.stage(sid)

    # Size of a single node component.
    probe = VisualDataStore()
    probe.nodes_slab(scoped, sid, "pressure")
    one = probe.loaded_bytes
    assert one > 0

    # Budget that holds ~1 node entry: loading a 2nd component must evict
    # the colder one rather than growing unbounded.
    store = VisualDataStore(byte_budget=one)
    assert store.nodes_slab(scoped, sid, "pressure") is not None
    assert store.gauss_slab(scoped, sid, "stress_xx") is not None

    cached = {k[1] for k in store._cache}    # noqa: SLF001
    assert "stress_xx" in cached, "the just-loaded live request must survive"
    assert "pressure" not in cached, "the LRU entry should have been evicted"
    # Bounded down to the single live entry: the cold component was dropped
    # rather than letting the cache grow unbounded. (Eviction never empties
    # the cache, so a lone entry larger than the budget is still kept — a
    # render is never refused.)
    assert len(store._cache) == 1
    assert store.loaded_bytes == store._cache[(sid, "stress_xx")].nbytes  # noqa: SLF001


def test_unbounded_store_never_evicts(results_with_stress_magnitude):
    sid = _stage_id(results_with_stress_magnitude)
    scoped = results_with_stress_magnitude.stage(sid)
    store = VisualDataStore()    # default: unbounded
    store.nodes_slab(scoped, sid, "pressure")
    store.gauss_slab(scoped, sid, "stress_xx")
    cached = {k[1] for k in store._cache}    # noqa: SLF001
    assert {"pressure", "stress_xx"} <= cached


# ---------------------------------------------------------------------
# Contour integration: cache hit vs fallback
# ---------------------------------------------------------------------
def _attach_contour(results, headless_plotter, *, store=None):
    scene = build_fem_scene(results.fem)
    diagram = ContourDiagram(_make_spec(), results)
    if store is not None:
        diagram._visual_store = store  # noqa: SLF001 - mimic the registry stamp
    diagram.attach(headless_plotter, results.fem, scene)
    return diagram, scene


def test_contour_uses_visual_store_and_does_not_reread(
    results_with_known_disp, headless_plotter,
):
    results = results_with_known_disp
    sid = _stage_id(results)
    store = VisualDataStore()
    store.load_stage(results, sid)

    diagram, scene = _attach_contour(results, headless_plotter, store=store)

    # The attach precomputed the id->column map into the cached slab.
    assert diagram._visual_node_slab_ref is not None  # noqa: SLF001
    assert diagram._visual_node_cols is not None  # noqa: SLF001

    # Count HDF5 reads through the reader: attach already loaded the
    # component once (lazy load inside _resolve_visual_node_columns).
    reader = results._reader  # noqa: SLF001
    reads = {"n": 0}
    orig = reader.read_nodes

    def counting(*a, **k):
        reads["n"] += 1
        return orig(*a, **k)

    reader.read_nodes = counting
    try:
        for step in range(4):
            diagram.update_to_step(step)
    finally:
        reader.read_nodes = orig

    # No per-step HDF5 read: the cache path slices a float32 row.
    assert reads["n"] == 0, f"expected 0 HDF5 reads during playback, got {reads['n']}"

    # And the painted values still match nid + t*1000 (correctness).
    node_ids = np.asarray(results.fem.nodes.ids, dtype=np.int64)
    grid = scene.grid
    # Map substrate rows -> fem node ids via the contour's own lookup.
    pos = diagram._submesh_pos_of_id  # noqa: SLF001
    arr = np.asarray(diagram._scalar_values)  # noqa: SLF001
    for step in range(4):
        diagram.update_to_step(step)
        expected = node_ids + step * 1000.0
        got = arr[pos[node_ids]]
        # float32 round-trip tolerance (exact for these integer magnitudes).
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_contour_falls_back_without_store(results_with_known_disp, headless_plotter):
    results = results_with_known_disp
    diagram, scene = _attach_contour(results, headless_plotter, store=None)

    # No store stamped -> no cache wiring -> per-step read path.
    assert diagram._visual_node_slab_ref is None  # noqa: SLF001

    node_ids = np.asarray(results.fem.nodes.ids, dtype=np.int64)
    pos = diagram._submesh_pos_of_id  # noqa: SLF001
    arr = np.asarray(diagram._scalar_values)  # noqa: SLF001
    for step in range(4):
        diagram.update_to_step(step)
        expected = node_ids + step * 1000.0
        got = arr[pos[node_ids]]
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------
# Stable colour scale: the contour's auto clim anchors to the store's
# GLOBAL (vmin, vmax) over the whole history instead of step 0's range.
# ---------------------------------------------------------------------
@pytest.fixture
def results_step0_zero(g, tmp_path: Path):
    """displacement_z is 0 at step 0, then 1000*t — so step 0 is the
    degenerate undeformed state that breaks a step-0-anchored clim."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_steps = 4
    values = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    for t in range(1, n_steps):
        values[t] = float(t) * 1000.0    # step 0 stays all-zero
    path = tmp_path / "step0zero.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": values},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def test_contour_stable_clim_uses_global_not_step0(
    results_step0_zero, headless_plotter,
):
    sid = _stage_id(results_step0_zero)
    store = VisualDataStore()
    store.load_stage(results_step0_zero, sid)
    diagram, _ = _attach_contour(results_step0_zero, headless_plotter, store=store)
    lo, hi = diagram._initial_clim    # noqa: SLF001
    # Global demand range is 0..3000; the scale must reflect it, not the
    # degenerate step-0 (0, 1) it would collapse to without the store.
    assert hi == pytest.approx(3000.0, rel=1e-3)
    assert hi > 1.0


def test_contour_clim_falls_back_to_step0_without_store(
    results_step0_zero, headless_plotter,
):
    diagram, _ = _attach_contour(results_step0_zero, headless_plotter, store=None)
    lo, hi = diagram._initial_clim    # noqa: SLF001
    # No store -> per-step range -> step 0 is all-zero -> degenerate (0, 1).
    # (This is exactly the washed-out animation the store fixes.)
    assert (lo, hi) == (0.0, 1.0)