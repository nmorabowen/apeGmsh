"""GaussPointDiagram + GaussSlab.global_coords tests.

Verifies:

* Hex8 / quad4 shape functions evaluate correctly (corners + centre).
* GaussSlab.global_coords returns the documented (sum_GP, 3) array.
* GaussPointDiagram emits a sphere ``GlyphLayer`` through the render
  backend (ADR 0042, R-B Wave 2): emission + LUT-mirror state via the
  shared recording stub ``backend`` fixture (no GL); scalar bar / mapper
  / glyph-cell picking via a real offscreen ``PyVistaQtBackend``
  (``pv_backend`` fixture).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results._gauss_world_coords import (
    _hex8_shape_functions,
    _quad4_shape_functions,
    _world_via_bbox,
    compute_global_coords,
)
from apeGmsh.results._slabs import GaussSlab
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    GaussMarkerStyle,
    GaussPointDiagram,
    SlabSelector,
)
from apeGmsh.viewers.scene_ir import GlyphLayer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


# =====================================================================
# Pure shape-fn tests
# =====================================================================

def test_hex8_shape_fns_at_corners():
    """At a corner, the shape function for that corner is 1, others 0."""
    from apeGmsh.results._gauss_world_coords import _HEX8_CORNERS
    for i in range(8):
        N = _hex8_shape_functions(_HEX8_CORNERS[i])
        expected = np.zeros(8)
        expected[i] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-12)


def test_hex8_shape_fns_at_centre():
    """All 8 shape functions = 1/8 at the natural origin."""
    N = _hex8_shape_functions(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(N, np.full(8, 0.125), atol=1e-12)


def test_hex8_shape_fns_partition_of_unity():
    """For any natural point in the cube, sum of N_i = 1."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        nat = rng.uniform(-1, 1, 3)
        N = _hex8_shape_functions(nat)
        assert abs(N.sum() - 1.0) < 1e-12


def test_quad4_shape_fns_at_corners():
    from apeGmsh.results._gauss_world_coords import _QUAD4_CORNERS
    for i in range(4):
        N = _quad4_shape_functions(_QUAD4_CORNERS[i])
        expected = np.zeros(4)
        expected[i] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-12)


def test_quad4_shape_fns_partition_of_unity():
    rng = np.random.default_rng(11)
    for _ in range(20):
        nat = rng.uniform(-1, 1, 2)
        N = _quad4_shape_functions(nat)
        assert abs(N.sum() - 1.0) < 1e-12


def test_world_via_bbox_centre_is_centroid():
    """natural=(0,0,0) -> centroid for any element."""
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],
    ], dtype=np.float64)
    world = _world_via_bbox(np.array([0.0, 0.0, 0.0]), nodes)
    np.testing.assert_allclose(world, nodes.mean(axis=0))


# =====================================================================
# GaussSlab.global_coords integration
# =====================================================================

@pytest.fixture
def gauss_results(g, tmp_path: Path):
    """Solid hex mesh with synthetic 8-GP-per-element gauss values."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Pick all 3-D elements
    eids = []
    for group in fem.elements:
        if group.element_type.dim == 3:
            eids.extend(int(x) for x in group.ids)
    eids = sorted(eids)
    n_elem = len(eids)
    assert n_elem >= 1, "no 3-D elements meshed"

    # Use one GP per element at the natural origin (centre)
    gps_per_elem = 1
    natural_coords = np.array([[0.0, 0.0, 0.0]])
    n_total = n_elem * gps_per_elem

    n_steps = 2
    values = np.zeros(
        (n_steps, n_elem, gps_per_elem), dtype=np.float64,
    )
    for step in range(n_steps):
        for ei in range(n_elem):
            values[step, ei, 0] = ei + step * 100

    path = tmp_path / "gauss.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_gauss_group(
            sid, "partition_0", group_id="g0",
            class_tag=10,
            int_rule=0,
            element_index=np.asarray(eids, dtype=np.int64),
            natural_coords=natural_coords,
            local_axes_quaternion=None,
            components={"stress_xx": values},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path)), eids


def test_gauss_slab_global_coords_returns_correct_shape(gauss_results):
    results, eids = gauss_results
    slab = results.elements.gauss.get(component="stress_xx", time=[0])
    world = slab.global_coords(results.fem)
    assert world.shape == (slab.element_index.size, 3)
    # All GPs at natural=(0,0,0) so they should be at element centroids.
    # Just verify they fall within the cube.
    assert (world[:, 0] >= 0).all() and (world[:, 0] <= 1).all()
    assert (world[:, 1] >= 0).all() and (world[:, 1] <= 1).all()
    assert (world[:, 2] >= 0).all() and (world[:, 2] <= 1).all()


# =====================================================================
# GaussPointDiagram
# =====================================================================

def _make_gauss_spec(**style_kwargs):
    return DiagramSpec(
        kind="gauss_marker",
        selector=SlabSelector(component="stress_xx"),
        style=GaussMarkerStyle(**style_kwargs),
    )


# =====================================================================
# Construction + emission (recording stub backend)
# =====================================================================


def test_diagram_construction_requires_gauss_style(gauss_results):
    results, _ = gauss_results
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="gauss_marker",
        selector=SlabSelector(component="stress_xx"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="GaussMarkerStyle"):
        GaussPointDiagram(bad, results)


def test_attach_requires_scene(gauss_results, backend):
    results, _ = gauss_results
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(backend, results.fem)


def test_attach_emits_sphere_layer(gauss_results, backend):
    results, eids = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)

    layer = diagram._layer
    assert isinstance(layer, GlyphLayer)
    assert layer.kind == "sphere"
    assert layer.positions.n_points == len(eids)
    # Fixed size — one scale per center, all equal.
    assert layer.scales is not None
    assert layer.scales.shape == (len(eids),)
    assert np.allclose(layer.scales, layer.scales[0])
    # Coloured by the GP value.
    assert layer.color.mode == "by_array"
    assert layer.color_scalar is not None
    assert layer.color_scalar.shape == (len(eids),)
    # Scalar bar registered on the backend, keyed by layer id.
    assert diagram.spec.selector.component in backend.scalar_bars


def test_attach_carries_style_opacity(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(opacity=0.4), results)
    diagram.attach(backend, results.fem, scene)
    assert diagram._layer.opacity == pytest.approx(0.4)


def test_attach_initial_clim_auto_fits(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    clim = diagram.current_clim()
    assert clim is not None
    lo, hi = clim
    vals = np.asarray(diagram._layer.color_scalar)
    assert lo <= vals.min() + 1e-6
    assert hi >= vals.max() - 1e-6


def test_step_update_changes_color_scalar(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    initial = np.asarray(diagram._layer.color_scalar).copy()

    diagram.update_to_step(1)
    after = np.asarray(diagram._layer.color_scalar)
    # step 1 values are step*100 larger.
    assert (after - initial).max() == pytest.approx(100.0, rel=1e-6)


def test_handle_stable_across_steps(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    initial_handle = diagram._handle
    for step in range(2):
        diagram.update_to_step(step)
    assert diagram._handle is initial_handle


def test_set_visible_routes_through_backend(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    diagram.set_visible(False)
    assert diagram._handle.visible is False
    diagram.set_visible(True)
    assert diagram._handle.visible is True


def test_diagram_detach_clears_state(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    layer_id = diagram._handle.layer_id
    diagram.detach()
    assert diagram._layer is None
    assert diagram._handle is None
    assert not diagram.is_attached
    assert layer_id in backend.removed
    # The legend retires with its last source layer.
    assert backend.scalar_bars == {}


# =====================================================================
# LUT mirror (diagram-side state, recording stub backend)
# =====================================================================


def test_gauss_lut_is_none_before_attach(gauss_results):
    results, _ = gauss_results
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    assert diagram.lut is None


def test_gauss_attach_builds_lut_from_style(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    spec = _make_gauss_spec(cmap="plasma", clim=(-5.0, 5.0))
    diagram = GaussPointDiagram(spec, results)
    diagram.attach(backend, results.fem, scene)

    lut = diagram.lut
    assert lut is not None
    assert lut.array_name == "stress_xx"
    assert lut.preset == "plasma"
    assert lut.range == (-5.0, 5.0)


def test_gauss_attach_lut_picks_up_autofit_clim(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)

    lut = diagram.lut
    clim = diagram.current_clim()
    assert lut.range == clim


def test_gauss_set_cmap_routes_through_lut(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.set_cmap("turbo")
    assert diagram.lut.preset == "turbo"
    assert diagram._runtime_cmap == "turbo"


def test_gauss_set_clim_routes_through_lut(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.set_clim(-2.0, 7.0)
    assert diagram.lut.range == (-2.0, 7.0)
    assert diagram.current_clim() == (-2.0, 7.0)


def test_gauss_lut_change_pushes_colorspec_through_backend(
    gauss_results, backend,
):
    """LUT range change → backend.set_layer_color with the new clim."""
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    color = backend.colors[diagram._handle.layer_id]
    assert color.mode == "by_array"
    assert color.lut.vmin == pytest.approx(100.0)
    assert color.lut.vmax == pytest.approx(200.0)


def test_gauss_detach_clears_lut(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    assert diagram.lut is not None
    diagram.detach()
    assert diagram.lut is None


def test_gauss_lut_changes_after_detach_are_noops(gauss_results, backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(backend, results.fem, scene)
    held_lut = diagram.lut
    diagram.detach()
    # Must not raise.
    held_lut.set_preset("magma")
    held_lut.set_range(0.0, 1.0)


# =====================================================================
# Render integration (real offscreen PyVistaQtBackend)
# =====================================================================


def test_scalar_bar_appears_on_plotter(gauss_results, pv_backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    assert "stress_xx" in pv_backend.plotter.scalar_bars


def test_gauss_lut_change_updates_actor_mapper(gauss_results, pv_backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    mapper = diagram._handle.actor.GetMapper()
    sr = mapper.GetScalarRange()
    assert sr[0] == pytest.approx(100.0)
    assert sr[1] == pytest.approx(200.0)


def test_detach_removes_scalar_bar(gauss_results, pv_backend):
    """Repeated attach/detach cycles must not accumulate bars."""
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    for _ in range(3):
        diagram = GaussPointDiagram(_make_gauss_spec(), results)
        diagram.attach(pv_backend, results.fem, scene)
        diagram.detach()
    assert "stress_xx" not in (pv_backend.plotter.scalar_bars or {})


def test_set_show_and_fmt_live(gauss_results, pv_backend):
    results, _ = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.set_show_scalar_bar(False)
    assert "stress_xx" not in pv_backend.plotter.scalar_bars

    diagram.set_show_scalar_bar(True)
    assert "stress_xx" in pv_backend.plotter.scalar_bars

    diagram.set_fmt("%.4f")
    bar = pv_backend.plotter.scalar_bars["stress_xx"]
    assert bar.GetLabelFormat() == "%.4f"


def test_resolve_picked_cell_maps_glyph_cell_to_gp(gauss_results, pv_backend):
    """``resolve_picked_cell(cell_id)`` divides by the diagram's fixed
    cells-per-glyph block (derived from the backend's glyph dataset) to
    recover the GP center index, and looks up the matching
    ``element_id`` from the diagram's GP metadata."""
    results, eids = gauss_results
    scene = build_fem_scene(results.fem)
    diagram = GaussPointDiagram(_make_gauss_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    cells_per = diagram._glyph_cells_per_center
    assert cells_per > 0

    # Cell index inside the first glyph block → GP center 0.
    hit = diagram.resolve_picked_cell(0)
    assert hit is not None
    eid_first, gp_idx_first, world_first = hit
    assert gp_idx_first == 0
    assert eid_first == int(diagram._gp_element_index[0])
    assert world_first.shape == (3,)

    # A cell at the boundary of the second block → center 1.
    hit2 = diagram.resolve_picked_cell(cells_per)
    assert hit2 is not None
    _, gp_idx_second, _ = hit2
    assert gp_idx_second == 1

    # Out-of-range cell → None.
    n_centers = diagram._gp_element_index.size
    out_of_range = n_centers * cells_per + 1
    assert diagram.resolve_picked_cell(out_of_range) is None
    assert diagram.resolve_picked_cell(-1) is None
