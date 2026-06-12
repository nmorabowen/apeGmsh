"""VectorGlyphDiagram — emits an arrow GlyphLayer through the backend.

Split coverage (ADR 0042 R-B Wave 1 #4, first colour-mapped diagram):

* **Data / emission + LUT-mirror state** via the shared recording stub
  ``backend`` fixture (no GL): assert on the emitted ``GlyphLayer``
  (orientations, by_array colour, scalar bar recorded, set_cmap/set_clim
  routing through the diagram-side LUT mirror).
* **Render integration** via a real offscreen ``PyVistaQtBackend``
  (``pv_backend`` fixture): the scalar bar appears on the plotter and a
  LUT range change reaches the actor's mapper.

Catalog/dialog helper tests are unchanged — they don't touch rendering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    SlabSelector,
    VectorGlyphDiagram,
    VectorGlyphStyle,
)
from apeGmsh.viewers.scene_ir import GlyphLayer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


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
    path = tmp_path / "vec.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": base + t * 0.1,
                "displacement_y": base + t * 0.2,
                "displacement_z": base + t * 0.3,
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _spec(component: str = "displacement") -> DiagramSpec:
    return DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component=component),
        style=VectorGlyphStyle(scale=1.0),
    )


# =====================================================================
# Construction + emission (stub backend)
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


def test_attach_requires_scene(vector_results, backend):
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(backend, vector_results.fem)


def test_attach_emits_arrow_layer_by_magnitude(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)

    layer = diagram._layer
    assert isinstance(layer, GlyphLayer)
    n = vector_results.fem.nodes.ids.size
    assert layer.positions.n_points == n
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    # Step 0: dx = dy = dz = nid.
    np.testing.assert_allclose(layer.orientations[:, 0], fem_ids, atol=1e-3)
    np.testing.assert_allclose(layer.orientations[:, 1], fem_ids, atol=1e-3)
    np.testing.assert_allclose(layer.orientations[:, 2], fem_ids, atol=1e-3)
    # By-magnitude colouring → by_array ColorSpec + raw-magnitude scalar.
    assert layer.color.mode == "by_array"
    assert layer.color_scalar is not None
    # Scalar bar registered on the backend, keyed by layer id.
    assert diagram._handle.layer_id in backend.scalar_bars


def test_attach_initial_clim_auto_fits(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    clim = diagram.current_clim()
    assert clim is not None
    lo, hi = clim
    mags = np.asarray(diagram._layer.color_scalar)
    assert lo <= mags.min() + 1e-6
    assert hi >= mags.max() - 1e-6


def test_step_update_changes_orientations(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    initial = np.asarray(diagram._layer.orientations).copy()
    diagram.update_to_step(2)
    after = np.asarray(diagram._layer.orientations)
    assert not np.allclose(initial, after)
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    np.testing.assert_allclose(after[:, 0], fem_ids + 0.2, atol=1e-3)
    np.testing.assert_allclose(after[:, 1], fem_ids + 0.4, atol=1e-3)
    np.testing.assert_allclose(after[:, 2], fem_ids + 0.6, atol=1e-3)


def test_set_scale_records_runtime_value(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


def test_handle_stable_across_steps(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    initial_handle = diagram._handle
    for step in range(3):
        diagram.update_to_step(step)
    assert diagram._handle is initial_handle


def test_detach_clears_state(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    layer_id = diagram._handle.layer_id
    diagram.detach()
    assert diagram._layer is None
    assert diagram._handle is None
    assert not diagram.is_attached
    assert layer_id in backend.removed
    assert layer_id not in backend.scalar_bars


# =====================================================================
# Axis-locked mode
# =====================================================================


@pytest.mark.parametrize(
    "component, axis",
    [("displacement_x", 0), ("displacement_y", 1), ("displacement_z", 2)],
)
def test_axis_mode_zeros_other_components(vector_results, backend, component, axis):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(component), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    vecs = np.asarray(diagram._layer.orientations)
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    np.testing.assert_allclose(vecs[:, axis], fem_ids, atol=1e-3)
    for other in (0, 1, 2):
        if other != axis:
            np.testing.assert_allclose(vecs[:, other], 0.0, atol=1e-4)


def test_axis_mode_step_update(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec("displacement_y"), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    diagram.update_to_step(2)
    vecs = np.asarray(diagram._layer.orientations)
    fem_ids = np.asarray(scene.node_ids).astype(np.float64)
    np.testing.assert_allclose(vecs[:, 1], fem_ids + 0.4, atol=1e-3)
    np.testing.assert_allclose(vecs[:, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(vecs[:, 2], 0.0, atol=1e-4)


# =====================================================================
# LUT mirror (diagram-side state)
# =====================================================================


def test_vector_lut_is_none_before_attach(vector_results):
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    assert diagram.lut is None


def test_vector_attach_builds_lut(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(cmap="plasma", clim=(0.0, 10.0)),
    )
    diagram = VectorGlyphDiagram(spec, vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    lut = diagram.lut
    assert lut is not None
    assert lut.array_name == "displacement"
    assert lut.preset == "plasma"
    assert lut.range == (0.0, 10.0)


def test_vector_no_lut_when_magnitude_colors_disabled(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(use_magnitude_colors=False),
    )
    diagram = VectorGlyphDiagram(spec, vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    assert diagram.lut is None
    # Solid colour when not painting by magnitude.
    assert diagram._layer.color.mode == "solid"


def test_vector_set_cmap_routes_through_lut(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    diagram.set_cmap("turbo")
    assert diagram.lut.preset == "turbo"
    assert diagram._runtime_cmap == "turbo"


def test_vector_set_clim_routes_through_lut(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    diagram.set_clim(-2.0, 7.0)
    assert diagram.lut.range == (-2.0, 7.0)
    assert diagram.current_clim() == (-2.0, 7.0)


def test_lut_change_pushes_colorspec_through_backend(vector_results, backend):
    """LUT range change → backend.set_layer_color with the new clim."""
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    diagram.lut.set_range(100.0, 200.0)
    color = backend.colors[diagram._handle.layer_id]
    assert color.mode == "by_array"
    assert color.lut.vmin == pytest.approx(100.0)
    assert color.lut.vmax == pytest.approx(200.0)


def test_vector_detach_clears_lut(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    assert diagram.lut is not None
    diagram.detach()
    assert diagram.lut is None


def test_vector_lut_changes_after_detach_are_noops(vector_results, backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(backend, vector_results.fem, scene)
    held_lut = diagram.lut
    diagram.detach()
    held_lut.set_preset("magma")
    held_lut.set_range(0.0, 1.0)


# =====================================================================
# Render integration (real offscreen PyVistaQtBackend)
# =====================================================================


def test_scalar_bar_appears_on_plotter(vector_results, pv_backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(pv_backend, vector_results.fem, scene)
    assert "displacement" in pv_backend.plotter.scalar_bars


def test_set_show_scalar_bar_toggles_live(vector_results, pv_backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(pv_backend, vector_results.fem, scene)
    assert "displacement" in pv_backend.plotter.scalar_bars
    diagram.set_show_scalar_bar(False)
    assert "displacement" not in pv_backend.plotter.scalar_bars
    diagram.set_show_scalar_bar(True)
    assert "displacement" in pv_backend.plotter.scalar_bars


def test_detach_removes_scalar_bar(vector_results, pv_backend):
    scene = build_fem_scene(vector_results.fem)
    for _ in range(3):
        diagram = VectorGlyphDiagram(_spec(), vector_results)
        diagram.attach(pv_backend, vector_results.fem, scene)
        diagram.detach()
    assert "displacement" not in (pv_backend.plotter.scalar_bars or {})


def test_lut_change_updates_actor_mapper(vector_results, pv_backend):
    scene = build_fem_scene(vector_results.fem)
    diagram = VectorGlyphDiagram(_spec(), vector_results)
    diagram.attach(pv_backend, vector_results.fem, scene)
    diagram.lut.set_range(100.0, 200.0)
    mapper = diagram._handle.actor.GetMapper()
    sr = mapper.GetScalarRange()
    assert sr[0] == pytest.approx(100.0)
    assert sr[1] == pytest.approx(200.0)


# =====================================================================
# Catalog / dialog helpers (unchanged — no rendering)
# =====================================================================


@pytest.mark.parametrize(
    "selection, expected_prefix",
    [
        ("displacement", "displacement"),
        ("displacement_z", "displacement"),
        ("velocity", "velocity"),
        ("velocity_y", "velocity"),
        ("acceleration_x", "acceleration"),
        ("stress_xx", "stress_xx"),
    ],
)
def test_resolve_vector_prefix(selection, expected_prefix):
    from apeGmsh.viewers.diagrams._kind_catalog import resolve_vector_prefix
    assert resolve_vector_prefix(selection) == expected_prefix


@pytest.mark.parametrize(
    "selection, expected_components",
    [
        ("displacement", ("displacement_x", "displacement_y", "displacement_z")),
        ("displacement_y", ("displacement_x", "displacement_y", "displacement_z")),
        ("velocity", ("velocity_x", "velocity_y", "velocity_z")),
        ("velocity_z", ("velocity_x", "velocity_y", "velocity_z")),
        ("acceleration_x", ("acceleration_x", "acceleration_y", "acceleration_z")),
    ],
)
def test_vector_default_style_derives_components(selection, expected_components):
    # Default-style factory lives on the kind registry since ADR 0058
    # S0 (declared in _vector_glyph.py, next to the class).
    from apeGmsh.viewers.diagrams._kinds import kind_def
    style = kind_def("vector_glyph").make_default_style(selection)
    assert style.components == expected_components
