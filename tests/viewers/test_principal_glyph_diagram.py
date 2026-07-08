"""PrincipalDirectionDiagram — principal-direction arrows per Gauss point.

Drives the diagram off-screen on a tet-meshed cube carrying a synthetic
6-component stress tensor per element: registration, the 3-arrows-per-GP
layer, per-principal toggles, signed colouring, per-step update,
deform-follow, plane-strain kwargs, and the no-tensor fail-loud path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.backends import PyVistaQtBackend
from apeGmsh.viewers.diagrams import (
    DiagramSpec, NoDataError, PrincipalDirectionDiagram, PrincipalGlyphStyle,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._kinds import kind_def
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5

N_STEPS = 3
_SUF = ("xx", "yy", "zz", "xy", "yz", "xz")


def _continuum_ids(fem) -> np.ndarray:
    ids: list[int] = []
    for grp in fem.elements:
        if grp.element_type.dim == 3:
            ids.extend(int(x) for x in grp.ids)
    return np.asarray(sorted(ids), dtype=np.int64)


@pytest.fixture
def stress_results(g, tmp_path: Path):
    """Tet-meshed cube with a synthetic 6-component stress tensor / GP."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    eids = _continuum_ids(fem)
    n_e = eids.size
    nat = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)   # 1 GP / tet
    # Non-trivial state: σxx and a shear so eigenvectors aren't axis-aligned.
    comp_vals = {"xx": 10.0, "yy": 0.0, "zz": 0.0, "xy": 3.0, "yz": 0.0, "xz": 0.0}
    comps = {
        f"stress_{s}": np.full((N_STEPS, n_e, 1), comp_vals[s], dtype=np.float64)
        for s in _SUF
    }
    path = tmp_path / "stress.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture")
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=eids, natural_coords=nat, components=comps,
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path)), n_e


def _attach(results, style: PrincipalGlyphStyle):
    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = (320, 240)
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="principal_glyph",
        selector=SlabSelector(component="stress_xx"),
        style=style,
    )
    diagram = PrincipalDirectionDiagram(spec, results)
    diagram.attach(PyVistaQtBackend(plotter), results.fem, scene)
    return plotter, scene, diagram


# =====================================================================
# Registration
# =====================================================================

def test_kind_registered():
    entry = kind_def("principal_glyph")
    assert entry is not None
    assert entry.label == "Principal directions (arrows)"
    assert entry.diagram_class is PrincipalDirectionDiagram
    assert entry.style_class is PrincipalGlyphStyle
    assert entry.data_topology == "gauss"


def test_wrong_style_raises(stress_results):
    from apeGmsh.viewers.diagrams import ContourStyle
    results, _ = stress_results
    spec = DiagramSpec(
        kind="principal_glyph",
        selector=SlabSelector(component="stress_xx"),
        style=ContourStyle(),
    )
    with pytest.raises(TypeError, match="PrincipalGlyphStyle"):
        PrincipalDirectionDiagram(spec, results)


# =====================================================================
# Layer construction
# =====================================================================

def test_three_arrows_per_gp(stress_results):
    results, n_e = stress_results
    plotter, _scene, diagram = _attach(results, PrincipalGlyphStyle())
    try:
        assert diagram._layer is not None
        # 3 principals × one GP per tet.
        assert diagram._layer.orientations.shape == (3 * n_e, 3)
        assert diagram._layer.scales.shape == (3 * n_e,)
        assert diagram._values.shape == (n_e, 3)          # descending principals
        # signed colour scalar (compression negative present here: σ3 < 0).
        assert float(diagram._color_values.min()) < 0.0
    finally:
        diagram.detach()
        plotter.close()


def test_principal_toggle_drops_arrows(stress_results):
    results, n_e = stress_results
    plotter, _scene, diagram = _attach(
        results, PrincipalGlyphStyle(show_p2=False),
    )
    try:
        assert diagram._layer.orientations.shape == (2 * n_e, 3)
    finally:
        diagram.detach()
        plotter.close()


def test_scales_track_principal_magnitude(stress_results):
    results, n_e = stress_results
    plotter, _scene, diagram = _attach(results, PrincipalGlyphStyle(scale=1.0))
    try:
        # scale=1 → arrow length = |principal|; block order is p1|p2|p3.
        # (layer scales are stored float32, hence the float32-grade rtol)
        expect = np.abs(np.concatenate(
            [diagram._values[:, 0], diagram._values[:, 1], diagram._values[:, 2]]
        ))
        np.testing.assert_allclose(diagram._layer.scales, expect, rtol=1e-5)
    finally:
        diagram.detach()
        plotter.close()


# =====================================================================
# Update / deform-follow
# =====================================================================

def test_update_and_deform_follow(stress_results):
    results, n_e = stress_results
    plotter, scene, diagram = _attach(results, PrincipalGlyphStyle())
    try:
        diagram.update_to_step(N_STEPS - 1)
        assert diagram._layer.orientations.shape == (3 * n_e, 3)
        before = diagram._coords.copy()
        shifted = np.asarray(scene.grid.points, dtype=np.float64) + np.array([5.0, 0, 0])
        diagram.sync_substrate_points(shifted, scene)
        # anchors moved with the substrate
        assert not np.allclose(before, diagram._coords)
    finally:
        diagram.detach()
        plotter.close()


# =====================================================================
# Plane-strain kwargs + fail-loud
# =====================================================================

def test_plane_strain_style_changes_principals(g, tmp_path: Path):
    # 2-D data (only in-plane components stored) → plane/nu apply.
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    eids = _continuum_ids(fem)
    n_e = eids.size
    nat = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
    comps = {
        "stress_xx": np.full((1, n_e, 1), 10.0),
        "stress_yy": np.full((1, n_e, 1), 4.0),
        "stress_xy": np.zeros((1, n_e, 1)),
    }
    path = tmp_path / "stress2d.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="static", time=np.array([0.0]))
        w.write_gauss_group(sid, "partition_0", "group_0", class_tag=4,
                            int_rule=1, element_index=eids,
                            natural_coords=nat, components=comps)
        w.end_stage()
    results = Results.from_native(path, model=_open_model_from_h5(path))

    _, _, d0 = _attach(results, PrincipalGlyphStyle())
    v_plane_stress = d0._values.copy()      # σ_zz = 0
    d0.detach()
    _, _, d1 = _attach(results, PrincipalGlyphStyle(plane="strain", nu=0.3))
    v_plane_strain = d1._values.copy()      # σ_zz = 0.3·(10+4) = 4.2
    d1.detach()
    # σ_zz = ν(σxx+σyy) ≠ 0 becomes a principal → the set changes.
    assert not np.allclose(v_plane_stress, v_plane_strain)
    # third principal (min) picks up the recovered σ_zz where it's smallest.
    assert np.any(np.isclose(v_plane_strain, 0.3 * (10.0 + 4.0)))


def test_no_tensor_raises(g, tmp_path: Path):
    """A nodes-only file → attach fails loud (no stress tensor)."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    path = tmp_path / "nodesonly.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="static", time=np.array([0.0]))
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": np.zeros((1, node_ids.size))})
        w.end_stage()
    results = Results.from_native(path, model=_open_model_from_h5(path))
    plotter = pv.Plotter(off_screen=True)
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="principal_glyph",
        selector=SlabSelector(component="stress_xx"),
        style=PrincipalGlyphStyle(),
    )
    diagram = PrincipalDirectionDiagram(spec, results)
    try:
        with pytest.raises(NoDataError):
            diagram.attach(PyVistaQtBackend(plotter), results.fem, scene)
    finally:
        plotter.close()
