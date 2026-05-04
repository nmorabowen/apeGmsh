"""LoadsDiagram — attach + scale + sync.

Builds a small 2-D mesh with two nodal point loads in a single
pattern, then verifies the diagram populates its ``_source`` PolyData
with the expected force vectors and node positions.

The diagram is constant-magnitude (no timeSeries info in the broker),
so ``update_to_step`` is a no-op — that's tested too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    LoadsDiagram,
    LoadsStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


@pytest.fixture
def loads_results(g, tmp_path: Path):
    """Plate with two labelled corners carrying point loads."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_ids = list(fem.nodes.ids)

    # Minimal Results so we can construct the diagram (the broker
    # carries the loads on .fem, the diagram doesn't read .results).
    n_steps = 1
    node_ids = np.asarray(n_ids, dtype=np.int64)
    components = {
        "displacement_x": np.zeros((n_steps, node_ids.size)),
        "displacement_y": np.zeros((n_steps, node_ids.size)),
        "displacement_z": np.zeros((n_steps, node_ids.size)),
    }
    path = tmp_path / "loads.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="static", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    results = Results.from_native(path)
    # Inject load records on the post-load fem object so the diagram
    # sees them (the writer doesn't persist loads to H5 today).
    from apeGmsh.solvers.Loads import NodalLoadRecord
    fem_after = results.fem
    n_after = list(fem_after.nodes.ids)
    fem_after.nodes.loads._records.append(NodalLoadRecord(
        pattern="P1", node_id=int(n_after[0]),
        force_xyz=(10.0, 0.0, 0.0),
    ))
    fem_after.nodes.loads._records.append(NodalLoadRecord(
        pattern="P1", node_id=int(n_after[1]),
        force_xyz=(0.0, -5.0, 0.0),
    ))
    fem_after.nodes.loads._records.append(NodalLoadRecord(
        pattern="P2", node_id=int(n_after[0]),
        force_xyz=(0.0, 0.0, 7.0),
    ))
    return results


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _spec(pattern: str, scale: float | None = 1.0) -> DiagramSpec:
    return DiagramSpec(
        kind="loads",
        selector=SlabSelector(component=pattern),
        style=LoadsStyle(scale=scale),
    )


def test_construction_requires_loads_style(loads_results):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="loads",
        selector=SlabSelector(component="P1"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="LoadsStyle"):
        LoadsDiagram(bad, loads_results)


def test_attach_requires_scene(loads_results, headless_plotter):
    diagram = LoadsDiagram(_spec("P1"), loads_results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, loads_results.fem)


def test_attach_builds_source_for_pattern(loads_results, headless_plotter):
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P1"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)

    assert diagram._source is not None
    # Two records in P1 → two arrows.
    assert diagram._source.n_points == 2
    vecs = np.asarray(diagram._source.point_data["_vec"])
    mags = np.asarray(diagram._source.point_data["_mag"])
    # Force magnitudes — order-agnostic.
    np.testing.assert_allclose(sorted(mags.tolist()), [5.0, 10.0])
    # Vectors — sums match (10, -5, 0)
    np.testing.assert_allclose(vecs.sum(axis=0), [10.0, -5.0, 0.0])


def test_attach_other_pattern_is_independent(loads_results, headless_plotter):
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P2"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    assert diagram._source is not None
    # P2 only has one record on first node, force = (0, 0, 7)
    assert diagram._source.n_points == 1
    vecs = np.asarray(diagram._source.point_data["_vec"])
    np.testing.assert_allclose(vecs[0], [0.0, 0.0, 7.0])


def test_attach_unknown_pattern_silently_skips(loads_results, headless_plotter):
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("DOES_NOT_EXIST"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    # No records → no source, no actor. Diagram stays attached but inert.
    assert diagram._source is None
    assert diagram._actor is None


def test_update_to_step_is_noop(loads_results, headless_plotter):
    """Constant-magnitude — no timeSeries in broker."""
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P1"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    initial_vecs = np.asarray(diagram._source.point_data["_vec"]).copy()
    diagram.update_to_step(5)
    after = np.asarray(diagram._source.point_data["_vec"])
    np.testing.assert_array_equal(initial_vecs, after)


def test_actor_identity_stable_across_steps(loads_results, headless_plotter):
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P1"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    initial_actor = diagram._actor
    initial_source = diagram._source
    for step in range(3):
        diagram.update_to_step(step)
    assert diagram._actor is initial_actor
    assert diagram._source is initial_source


def test_set_scale_records_runtime_value(loads_results, headless_plotter):
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P1"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


def test_auto_scale_when_none(loads_results, headless_plotter):
    """``scale=None`` → auto-fit so largest arrow ~ fraction × diagonal."""
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P1", scale=None), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    # Largest force is 10.0; auto-fit picks scale = 0.10 * diag / 10.0
    expected = 0.10 * scene.model_diagonal / 10.0
    assert abs(diagram.current_scale() - expected) < 1e-9


def test_detach_clears_state(loads_results, headless_plotter):
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("P1"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    diagram.detach()
    assert diagram._source is None
    assert diagram._actor is None
    assert not diagram.is_attached


def test_zero_force_records_skipped(loads_results, headless_plotter):
    """Records with all-zero force_xyz produce no glyph (degenerate)."""
    from apeGmsh.solvers.Loads import NodalLoadRecord
    loads_results.fem.nodes.loads._records.append(NodalLoadRecord(
        pattern="ZEROES", node_id=int(loads_results.fem.nodes.ids[0]),
        force_xyz=(0.0, 0.0, 0.0),
    ))
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("ZEROES"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    assert diagram._source is None


def test_moment_only_records_skipped(loads_results, headless_plotter):
    """Records with only moments (force_xyz=None) are ignored — moments
    need a different glyph and aren't drawn yet."""
    from apeGmsh.solvers.Loads import NodalLoadRecord
    loads_results.fem.nodes.loads._records.append(NodalLoadRecord(
        pattern="MOMENT_ONLY",
        node_id=int(loads_results.fem.nodes.ids[0]),
        moment_xyz=(0.0, 0.0, 5.0),
    ))
    scene = build_fem_scene(loads_results.fem)
    diagram = LoadsDiagram(_spec("MOMENT_ONLY"), loads_results)
    diagram.attach(headless_plotter, loads_results.fem, scene)
    assert diagram._source is None
