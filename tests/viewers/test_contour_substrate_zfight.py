"""Contour occludes the grey substrate fill (no z-fighting).

Regression for the grey-bleeds-through-the-contour bug. A ContourDiagram
paints an opaque filled surface on a submesh extracted from the
substrate, coincident with the opaque grey substrate fill. Two opaque
coincident surfaces z-fight (the first-drawn substrate wins the depth
test, so the grey shows through). VTK cross-actor polygon offset does
NOT resolve this (verified), so the viewer instead hides the geometry
substrate FILL while a visible occluding diagram is attached (keeps the
wireframe). ``Diagram.occludes_substrate`` flags which diagrams trigger
the hide; ``_geometries_occluded_by_diagrams`` computes the affected
geometry set; ``_apply_geometry_display`` drops the fill there.
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
from apeGmsh.viewers.diagrams._base import Diagram
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


def _mean_abs_diff(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).mean())


@pytest.fixture
def gradient_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 4, 1, 1, label="bar")
    g.physical.add_volume("bar", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    values = coords[:, 0].reshape(1, node_ids.size).astype(np.float64)
    path = tmp_path / "grad.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="s", kind="static", time=np.array([0.0]))
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_z": values})
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _make_spec() -> DiagramSpec:
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )


# ---------------------------------------------------------------------
# Logic: the occlusion flag + the geometry-set helper (headless)
# ---------------------------------------------------------------------
def test_contour_diagram_is_occluding():
    assert ContourDiagram.occludes_substrate is True
    # Base default leaves non-fill diagrams alone.
    assert Diagram.occludes_substrate is False


def test_occluded_geometry_set_tracks_contour(gradient_results, headless_plotter):
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    from apeGmsh.viewers.results_viewer import _geometries_occluded_by_diagrams

    director = ResultsDirector(gradient_results)
    scene = build_fem_scene(gradient_results.fem)
    director.bind_plotter(
        headless_plotter, scene=scene,
        render_callback=lambda: None,
    )
    geom = director.geometries.active
    assert geom is not None

    # No diagrams -> nothing occluded.
    assert _geometries_occluded_by_diagrams(director) == set()

    # Add the contour the canonical way: registry + composition membership
    # so geometry_for_layer resolves its owning geometry.
    comp = geom.compositions.active or geom.compositions.add("Diagram")
    diagram = ContourDiagram(_make_spec(), gradient_results)
    director.registry.add(diagram)
    geom.compositions.add_layer(comp.id, diagram)
    # Attached + visible contour -> its owning geometry is occluded.
    assert geom.id in _geometries_occluded_by_diagrams(director)

    # Hide the contour (eye off) -> fill should come back, not occluded.
    director.registry.set_visible(diagram, False)
    assert _geometries_occluded_by_diagrams(director) == set()

    # Show again -> occluded again.
    director.registry.set_visible(diagram, True)
    assert geom.id in _geometries_occluded_by_diagrams(director)

    # Remove -> not occluded.
    director.registry.remove(diagram)
    geom.compositions.remove_layer(diagram)
    assert _geometries_occluded_by_diagrams(director) == set()


# ---------------------------------------------------------------------
# Render: hiding the substrate fill kills the z-fight (real pixels)
# ---------------------------------------------------------------------
def test_hiding_substrate_fill_lets_contour_win(gradient_results, headless_plotter):
    plotter = headless_plotter.plotter
    scene = build_fem_scene(gradient_results.fem)

    # Contour only (no substrate) -> its true surface colour.
    diagram = ContourDiagram(_make_spec(), gradient_results)
    diagram.attach(headless_plotter, gradient_results.fem, scene)
    plotter.view_isometric()
    plotter.render()
    img_contour = plotter.screenshot(return_img=True)
    if img_contour is None:
        pytest.skip("no offscreen render context")

    # Substrate grey fill + wireframe ON TOP of the contour (z-fights).
    fill = plotter.add_mesh(scene.grid, color="#bfbfbf", show_edges=False,
                            opacity=1.0, name="results_substrate")
    plotter.add_mesh(scene.grid, style="wireframe", color="#444444",
                     name="results_wireframe", pickable=False)
    plotter.render()
    img_zfight = plotter.screenshot(return_img=True)

    # Hide the substrate fill exactly as _apply_geometry_display now does
    # when an occluding contour is attached (wireframe stays).
    fill.SetVisibility(0)
    plotter.render()
    img_fixed = plotter.screenshot(return_img=True)

    # Before the fix the grey bled through (surface drifted off the
    # contour-only render). After hiding the fill the surface matches
    # the contour-only render -> the grey no longer z-fights through.
    diff_fixed = _mean_abs_diff(img_fixed, img_contour)
    diff_zfight = _mean_abs_diff(img_zfight, img_contour)
    assert diff_fixed < 3.0, (
        f"hidden-fill surface does not match contour-only "
        f"(diff={diff_fixed:.2f})"
    )
    assert diff_zfight > diff_fixed + 5.0, (
        f"z-fighting surface did not differ from fixed "
        f"(zfight={diff_zfight:.2f}, fixed={diff_fixed:.2f})"
    )