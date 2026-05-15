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


def _spec(component: str = "displacement") -> DiagramSpec:
    """Default selector picks the displacement *prefix* (resultant mode).

    Per-axis tests pass ``displacement_x/y/z`` to drive the diagram
    into axis-locked mode.
    """
    return DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component=component),
        style=VectorGlyphStyle(scale=1.0),
    )


# =====================================================================
# Construction
# =====================================================================

def test_construction_requires_vector_style(vector_results):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
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


def test_detach_removes_scalar_bar(vector_results, headless_plotter):
    """Magnitude-colored arrows register a scalar bar; detach must
    drop it so repeated attach/detach cycles don't accumulate bars."""
    scene = build_fem_scene(vector_results.fem)
    for _ in range(3):
        diagram = VectorGlyphDiagram(_spec(), vector_results)
        diagram.attach(headless_plotter, vector_results.fem, scene)
        diagram.detach()
    bars = getattr(headless_plotter, "scalar_bars", {}) or {}
    assert "displacement" not in bars


def test_set_show_scalar_bar_toggles_live(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    assert "displacement" in headless_plotter.scalar_bars

    diagram.set_show_scalar_bar(False)
    assert "displacement" not in headless_plotter.scalar_bars

    diagram.set_show_scalar_bar(True)
    assert "displacement" in headless_plotter.scalar_bars


def test_set_fmt_updates_label_format_live(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    diagram.set_fmt("%.2e")
    assert headless_plotter.scalar_bars["displacement"].GetLabelFormat() == "%.2e"


# =====================================================================
# Axis-locked mode (selector.component matches one of style.components)
# =====================================================================

@pytest.mark.parametrize(
    "component, axis, step0_per_axis",
    [
        # Step 0: dx = nid + 0.0, dy = nid + 0.0, dz = nid + 0.0
        ("displacement_x", 0, ("x", 0.0)),
        ("displacement_y", 1, ("y", 0.0)),
        ("displacement_z", 2, ("z", 0.0)),
    ],
)
def test_axis_mode_zeros_other_components(
    vector_results, headless_plotter, component, axis, step0_per_axis,
):
    """``displacement_x/y/z`` retain only the picked axis in ``_vec``."""
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(component), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    vecs = np.asarray(diagram._source.point_data["_vec"])
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    # Selected axis carries the value, the other two are zero.
    expected = fem_ids + step0_per_axis[1]
    np.testing.assert_allclose(vecs[:, axis], expected)
    for other in (0, 1, 2):
        if other != axis:
            np.testing.assert_allclose(vecs[:, other], 0.0)


def test_axis_mode_step_update(vector_results, headless_plotter):
    """Per-step rescaling still hits only the picked axis."""
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec("displacement_y"), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    diagram.update_to_step(2)
    vecs = np.asarray(diagram._source.point_data["_vec"])
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    # Step 2: dy = nid + 0.4, dx and dz zeroed.
    np.testing.assert_allclose(vecs[:, 1], fem_ids + 0.4)
    np.testing.assert_allclose(vecs[:, 0], 0.0)
    np.testing.assert_allclose(vecs[:, 2], 0.0)


@pytest.mark.parametrize(
    "selection, expected_prefix",
    [
        ("displacement",   "displacement"),
        ("displacement_z", "displacement"),
        ("velocity",       "velocity"),
        ("velocity_y",     "velocity"),
        ("acceleration_x", "acceleration"),
        # Tensor suffix is not a vector axis — falls through unchanged.
        ("stress_xx",      "stress_xx"),
    ],
)
def test_resolve_vector_prefix(selection, expected_prefix):
    from apeGmsh.viewers.diagrams._kind_catalog import resolve_vector_prefix
    assert resolve_vector_prefix(selection) == expected_prefix


@pytest.mark.parametrize(
    "selection, expected_components",
    [
        ("displacement",
            ("displacement_x", "displacement_y", "displacement_z")),
        ("displacement_y",
            ("displacement_x", "displacement_y", "displacement_z")),
        ("velocity",
            ("velocity_x", "velocity_y", "velocity_z")),
        ("velocity_z",
            ("velocity_x", "velocity_y", "velocity_z")),
        ("acceleration_x",
            ("acceleration_x", "acceleration_y", "acceleration_z")),
    ],
)
def test_vector_default_style_derives_components(selection, expected_components):
    """``_vector_default_style`` reads the right field for any prefix."""
    from apeGmsh.viewers.ui._add_diagram_dialog import _vector_default_style
    style = _vector_default_style(selection)
    assert style.components == expected_components


def test_partial_recording_resultant_works(g, tmp_path: Path, headless_plotter):
    """Resultant prefix tolerates a file with only one axis recorded.

    Catches the ≥1 (formerly ≥2) prefix-eligibility threshold: a file
    with only ``displacement_x`` should still offer a usable
    ``displacement`` resultant entry, and the diagram should render
    arrows aligned with x.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 2
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, len(node_ids)))
    components = {"displacement_x": base.copy()}

    path = tmp_path / "vec_x_only.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids, components=components,
        )
        w.end_stage()
    results = Results.from_native(path)

    # Catalog should still offer a "displacement" prefix entry.
    from apeGmsh.viewers.diagrams._kind_catalog import _vector_prefixes
    assert "displacement" in _vector_prefixes(["displacement_x"])

    # Diagram with prefix selection (resultant mode) renders fine.
    spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(scale=1.0),
    )
    scene = build_fem_scene(results.fem)
    diagram = VectorGlyphDiagram(spec, results)
    diagram.attach(headless_plotter, results.fem, scene)
    vecs = np.asarray(diagram._source.point_data["_vec"])
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    # x populated from the slab; y and z stay zero.
    np.testing.assert_allclose(vecs[:, 0], fem_ids)
    np.testing.assert_allclose(vecs[:, 1], 0.0)
    np.testing.assert_allclose(vecs[:, 2], 0.0)


# =====================================================================
# LUT mirror (plan 06)
# =====================================================================


def test_vector_lut_is_none_before_attach(vector_results):
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    assert diagram.lut is None


def test_vector_attach_builds_lut(vector_results, headless_plotter):
    """``use_magnitude_colors=True`` (default) → LUT is built."""
    scene = build_fem_scene(vector_results.fem)
    spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(cmap="plasma", clim=(0.0, 10.0)),
    )
    diagram = VectorGlyphDiagram(spec, vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)

    lut = diagram.lut
    assert lut is not None
    assert lut.array_name == "displacement"
    assert lut.preset == "plasma"
    assert lut.range == (0.0, 10.0)


def test_vector_no_lut_when_magnitude_colors_disabled(
    vector_results, headless_plotter,
):
    """``use_magnitude_colors=False`` → no LUT (nothing to drive)."""
    scene = build_fem_scene(vector_results.fem)
    spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(use_magnitude_colors=False),
    )
    diagram = VectorGlyphDiagram(spec, vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)

    assert diagram.lut is None


def test_vector_set_cmap_routes_through_lut(
    vector_results, headless_plotter,
):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)

    diagram.set_cmap("turbo")
    assert diagram.lut.preset == "turbo"
    assert diagram._runtime_cmap == "turbo"


def test_vector_set_clim_routes_through_lut(
    vector_results, headless_plotter,
):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)

    diagram.set_clim(-2.0, 7.0)
    assert diagram.lut.range == (-2.0, 7.0)
    assert diagram.current_clim() == (-2.0, 7.0)


def test_vector_lut_change_updates_actor_mapper(
    vector_results, headless_plotter,
):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    mapper = diagram._actor.GetMapper()
    sr = mapper.GetScalarRange()
    assert sr[0] == pytest.approx(100.0)
    assert sr[1] == pytest.approx(200.0)


def test_vector_detach_clears_lut(vector_results, headless_plotter):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    assert diagram.lut is not None
    diagram.detach()
    assert diagram.lut is None


def test_vector_lut_changes_after_detach_are_noops(
    vector_results, headless_plotter,
):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(headless_plotter, vector_results.fem, scene)
    held_lut = diagram.lut
    diagram.detach()
    held_lut.set_preset("magma")
    held_lut.set_range(0.0, 1.0)


def test_axis_mode_scale_matches_resultant(vector_results, headless_plotter):
    """``displacement_x`` and the prefix share auto-fit scale."""
    scene = build_fem_scene(vector_results.fem)
    # Build both with auto-fit (scale=None) so the global-norm path runs.
    res_spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(),
    )
    axis_spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement_x"),
        style=VectorGlyphStyle(),
    )
    resultant = VectorGlyphDiagram(res_spec, vector_results)
    resultant.attach(headless_plotter, vector_results.fem, scene)
    pv2 = pv.Plotter(off_screen=True)
    try:
        axis_x = VectorGlyphDiagram(axis_spec, vector_results)
        axis_x.attach(pv2, vector_results.fem, scene)
        assert abs(
            resultant.current_scale() - axis_x.current_scale()
        ) < 1e-9
    finally:
        pv2.close()
