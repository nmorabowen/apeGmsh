"""VectorGlyphDiagram — attach + step + scale.

Builds a small 3-D solid mesh, writes synthetic displacement_x/y/z
per node + step, and verifies the source PolyData carries the
expected vector + magnitude arrays after attach and step changes.

We don't inspect the output glyph PolyData (its layout depends on
the arrow geom + scaling factor). The contract that matters is
the *source* arrays — those are what drive every glyph rebuild.
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
    SlabSelector,
    VectorGlyphDiagram,
    VectorGlyphStyle,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


@pytest.fixture
def vector_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 3
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, n_nodes))
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    components = {
        "displacement_x": base + t * 0.1,
        "displacement_y": base + t * 0.2,
        "displacement_z": base + t * 0.3,
    }

    path = tmp_path / "vec.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _spec() -> DiagramSpec:
    return DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement_x"),
        style=VectorGlyphStyle(scale=1.0),
    )


# =====================================================================
# Construction
# =====================================================================

def test_construction_requires_vector_style(vector_results):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="VectorGlyphStyle"):
        VectorGlyphDiagram(bad, vector_results)


# =====================================================================
# Attach
# =====================================================================

def test_attach_requires_scene(vector_results, headless_plotter):
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, vector_results.fem)


def test_attach_builds_source(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)

    assert diagram._source is not None
    n = vector_results.fem.nodes.ids.size
    assert diagram._source.n_points == n
    # Vectors at step 0 = (nid, nid, nid) for each node
    vecs = np.asarray(diagram._source.point_data["_vec"])
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    np.testing.assert_allclose(vecs[:, 0], fem_ids)
    np.testing.assert_allclose(vecs[:, 1], fem_ids)
    np.testing.assert_allclose(vecs[:, 2], fem_ids)


def test_attach_initial_clim_auto_fits(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    clim = diagram.current_clim()
    assert clim is not None
    lo, hi = clim
    mags = np.linalg.norm(np.asarray(diagram._source.point_data["_vec"]), axis=1)
    assert lo <= mags.min() + 1e-9
    assert hi >= mags.max() - 1e-9


# =====================================================================
# Step update
# =====================================================================

def test_step_update_changes_vectors(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    initial_vecs = np.asarray(diagram._source.point_data["_vec"]).copy()

    diagram.update_to_step(2)
    after = np.asarray(diagram._source.point_data["_vec"])
    assert not np.allclose(initial_vecs, after)


def test_step_2_vectors_match_components(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    diagram.update_to_step(2)
    vecs = np.asarray(diagram._source.point_data["_vec"])
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    # Step 2: dx = nid + 0.2, dy = nid + 0.4, dz = nid + 0.6
    np.testing.assert_allclose(vecs[:, 0], fem_ids + 0.2)
    np.testing.assert_allclose(vecs[:, 1], fem_ids + 0.4)
    np.testing.assert_allclose(vecs[:, 2], fem_ids + 0.6)


# =====================================================================
# Scale
# =====================================================================

def test_set_scale_records_runtime_value(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


# =====================================================================
# In-place mutation
# =====================================================================

def test_actor_identity_stable_across_steps(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    initial_actor = diagram._actor
    initial_source = diagram._source

    for step in range(3):
        diagram.update_to_step(step)

    assert diagram._actor is initial_actor
    assert diagram._source is initial_source


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    diagram.detach()
    assert diagram._source is None
    assert diagram._actor is None
    assert not diagram.is_attached
