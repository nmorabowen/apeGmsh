"""SectionCutDiagram + director plumbing — attach / detach / orientation / fan-out.

Builds a minimal model.h5 + a small FEMData by hand so the tag map and
the FEM mesh both exist without driving the full bridge. Keeps the
tests fast (no Gmsh mesh generation) and deterministic.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

# Force offscreen Qt for the Phase D rebuild tests at the bottom of the
# file. Set early so any later qtpy import lands on the right platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pyvista as pv
import pytest

from apeGmsh.cuts import (
    FemToOpsTagMap,
    SectionCutDef,
    SectionSweepDef,
    plane_horizontal,
)
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    SectionCutDiagram,
    SectionCutStyle,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._styles import DiagramStyle
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5
from tests.fixtures.schema import OPENSEES_CURRENT


# ---------------------------------------------------------------------
# Inline fixtures — tag-map h5 + FEMData + Results
# ---------------------------------------------------------------------

def _write_minimal_h5(
    path: Path, *, ops_to_fem: dict[int, int],
    type_token: str = "forceBeamColumn",
    schema_version: str = OPENSEES_CURRENT,
) -> None:
    ids = np.array(list(ops_to_fem.keys()), dtype=np.int64)
    fem_eids = np.array(list(ops_to_fem.values()), dtype=np.int64)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = schema_version
        meta.attrs["apeGmsh_version"] = "0.0.0-test"
        meta.attrs["created_iso"] = "2026-01-01T00:00:00Z"
        meta.attrs["ndm"] = 3
        meta.attrs["ndf"] = 6
        meta.attrs["snapshot_id"] = "test"
        meta.attrs["model_name"] = "test"
        em = f.create_group("opensees/element_meta")
        g = em.create_group(type_token)
        g.attrs["type"] = type_token
        g.create_dataset("ids", data=ids)
        g.create_dataset("fem_eids", data=fem_eids)


@pytest.fixture
def cube_results(g, tmp_path: Path):
    """1×1×1 cube mesh + Results with a tiny stage.

    The cube spans z ∈ [0, 1], so a horizontal cut at z=0.5 slices
    through its mid-height. Element ids land in ``fem.elements.ids``;
    we pair the first three with synthetic OpenSees tags 10, 11, 12 for
    the tag map.
    """
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1.0, 1.0, 1.0, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Tag map: OpenSees tags 10, 11, 12 → first three FEM eids.
    e_ids = sorted(int(e) for e in fem.elements.ids)
    ops_to_fem = {10: e_ids[0], 11: e_ids[1], 12: e_ids[2]}

    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, ops_to_fem=ops_to_fem, type_token="FourNodeTetrahedron")

    # Minimal Results — single stage, zero displacements. The diagram
    # doesn't read .results, but DiagramSpec needs a Results to attach.
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    res_path = tmp_path / "run.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="static", kind="static",
            time=np.array([0.0], dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": np.zeros((1, node_ids.size)),
                "displacement_y": np.zeros((1, node_ids.size)),
                "displacement_z": np.zeros((1, node_ids.size)),
            },
        )
        w.end_stage()
    results = Results.from_native(res_path, model=_open_model_from_h5(res_path))
    return results, fem, h5, ops_to_fem


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _make_cut(
    *, side: str = "positive", with_bounding: bool = False,
    element_ids=(10, 11, 12),
):
    point, normal = plane_horizontal(z=0.5)
    bounding = (
        ((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5), (0.0, 1.0, 0.5))
        if with_bounding else None
    )
    return SectionCutDef(
        plane_point=point,
        plane_normal=normal,
        element_ids=element_ids,
        side=side,
        label="cube mid-cut",
        bounding_polygon=bounding,
    )


def _spec(cut: SectionCutDef) -> DiagramSpec:
    return DiagramSpec(
        kind="section_cut",
        selector=SlabSelector(component=cut.label or "section_cut"),
        style=SectionCutStyle(cut=cut),
    )


# ---------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------

def test_construction_requires_section_cut_style(cube_results):
    results, _fem, _h5, _ = cube_results
    bad = DiagramSpec(
        kind="section_cut",
        selector=SlabSelector(component="cube mid-cut"),
        style=DiagramStyle(),
    )
    tag_map = FemToOpsTagMap.from_h5(_h5)
    with pytest.raises(TypeError, match="SectionCutStyle"):
        SectionCutDiagram(bad, results, tag_map=tag_map)


def test_attach_requires_scene(cube_results, headless_plotter):
    results, fem, h5, _ = cube_results
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(_spec(_make_cut()), results, tag_map=tag_map)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, fem)


def test_attach_requires_tag_map(cube_results, headless_plotter):
    results, fem, _h5, _ = cube_results
    scene = build_fem_scene(fem)
    diagram = SectionCutDiagram(_spec(_make_cut()), results, tag_map=None)
    with pytest.raises(RuntimeError, match="FemToOpsTagMap"):
        diagram.attach(headless_plotter, fem, scene)


# ---------------------------------------------------------------------
# Quad geometry
# ---------------------------------------------------------------------

def test_attach_with_bounding_polygon_uses_polygon_vertices(
    cube_results, backend,
):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    cut = _make_cut(with_bounding=True)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(_spec(cut), results, tag_map=tag_map)
    diagram.attach(backend, fem, scene)

    assert diagram.quad_layer is not None
    # Emitted as a single-polygon cut face.
    assert list(diagram.quad_layer.cells.blocks) == ["polygon"]
    verts = np.asarray(diagram.quad_layer.points.coords, dtype=np.float64)
    # Same set of vertices as the bounding polygon — order may have
    # flipped to align the front face with the kept side.
    expected = np.asarray(cut.bounding_polygon, dtype=np.float64)
    # Sort both by (x, y, z) for a winding-agnostic comparison.
    np.testing.assert_allclose(
        np.array(sorted(verts.tolist())),
        np.array(sorted(expected.tolist())),
        atol=1e-6,
    )
    # Two-tone faces carry the discarded-side back colour.
    assert diagram.quad_layer.back_color == diagram.spec.style.discarded_color


def test_attach_without_bounding_uses_filter_aabb(
    cube_results, backend,
):
    """Quad vertices sit over the AABB of the filter elements, not the
    full model AABB. For a 1×1×1 cube meshed coarsely (lc=1.0), the
    first three tets sample a subset of nodes; the quad's footprint
    should sit inside the model's plan extent.
    """
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    cut = _make_cut(with_bounding=False)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(_spec(cut), results, tag_map=tag_map)
    diagram.attach(backend, fem, scene)

    verts = np.asarray(diagram.quad_layer.points.coords, dtype=np.float64)
    # All vertices lie on the cut plane.
    np.testing.assert_allclose(verts[:, 2], 0.5, atol=1e-6)
    # Quad has 4 vertices.
    assert verts.shape == (4, 3)
    # Footprint is within the model bounds [0, 1] × [0, 1].
    assert verts[:, 0].min() >= -1e-6
    assert verts[:, 0].max() <= 1.0 + 1e-6
    assert verts[:, 1].min() >= -1e-6
    assert verts[:, 1].max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------
# Kept-side orientation
# ---------------------------------------------------------------------

def test_kept_side_winding_positive_side(cube_results, backend):
    """``side="positive"`` → front face normal aligned with +plane_normal."""
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    cut = _make_cut(side="positive", with_bounding=True)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(_spec(cut), results, tag_map=tag_map)
    diagram.attach(backend, fem, scene)

    verts = np.asarray(diagram.quad_layer.points.coords, dtype=np.float64)
    face_normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    # Plane normal is +z; ``positive`` keeps the +z half, so the
    # front face should look up.
    assert face_normal[2] > 0.0


def test_kept_side_winding_negative_side(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    cut = _make_cut(side="negative", with_bounding=True)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(_spec(cut), results, tag_map=tag_map)
    diagram.attach(backend, fem, scene)

    verts = np.asarray(diagram.quad_layer.points.coords, dtype=np.float64)
    face_normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    # ``negative`` keeps the -z half, so the front face should look down.
    assert face_normal[2] < 0.0


# ---------------------------------------------------------------------
# Static behaviour (decision D6)
# ---------------------------------------------------------------------

def test_update_to_step_is_noop(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    initial = np.asarray(diagram.quad_layer.points.coords).copy()
    diagram.update_to_step(3)
    after = np.asarray(diagram.quad_layer.points.coords)
    np.testing.assert_array_equal(initial, after)


def test_sync_substrate_points_does_not_move_quad(
    cube_results, backend,
):
    """Cuts are reference-config geometry — deformation has no effect."""
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    initial = np.asarray(diagram.quad_layer.points.coords).copy()
    # Big warp — should be ignored.
    deformed = np.asarray(scene.grid.points) + np.array([0.0, 0.0, 5.0])
    diagram.sync_substrate_points(deformed, scene)
    after = np.asarray(diagram.quad_layer.points.coords)
    np.testing.assert_array_equal(initial, after)


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------

def test_detach_clears_actors(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    quad_id = diagram.quad_layer.layer_id
    assert quad_id in backend.layers
    diagram.detach()
    # Layers removed from the backend; diagram released its handles.
    assert quad_id in backend.removed
    assert diagram.quad_layer is None
    assert diagram._quad_handle is None
    assert diagram.arrow_layer is None
    assert diagram._arrow_handle is None
    assert diagram._filter_fem_eids is None
    assert not diagram.is_attached


def test_filter_fem_eids_cached_for_phase_1b(
    cube_results, backend,
):
    """``filter_fem_eids`` exposes the resolved FEM ids — Phase 1b
    filter highlight reads it from here."""
    results, fem, h5, ops_to_fem = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    cut = _make_cut(element_ids=(10, 11))
    diagram = SectionCutDiagram(_spec(cut), results, tag_map=tag_map)
    diagram.attach(backend, fem, scene)
    assert diagram.filter_fem_eids is not None
    expected_fem = {ops_to_fem[10], ops_to_fem[11]}
    assert set(int(x) for x in diagram.filter_fem_eids) == expected_fem


# ---------------------------------------------------------------------
# Director plumbing (add_section_cut / add_section_cut_sweep)
# ---------------------------------------------------------------------

def test_director_add_section_cut_without_orientation_raises(cube_results):
    """``add_section_cut`` requires either a bound Results with an
    orientation source OR an explicit ``model_h5=`` kwarg.  ADR 0026
    PR-stretch — the error message points at ``bind_results``."""
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, _fem, _h5, _ = cube_results
    director = ResultsDirector(results)
    with pytest.raises(RuntimeError, match="no model.h5 source available"):
        director.add_section_cut(_make_cut())


def test_director_add_section_cut_routes_to_active_geometry(cube_results):
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, _fem, h5, _ = cube_results
    director = ResultsDirector(results)
    diagram = director.add_section_cut(_make_cut(), model_h5=h5)

    # Registered in the registry.
    assert diagram in list(director.registry)
    # Routed into the active geometry's "Section cuts" composition.
    geom = director.geometries.active
    assert geom is not None
    section_comp = next(
        (c for c in geom.compositions.compositions if c.name == "Section cuts"),
        None,
    )
    assert section_comp is not None
    assert diagram in section_comp.layers


# ---------------------------------------------------------------------
# Phase 1b — filter highlight toggle
# ---------------------------------------------------------------------

def test_show_filter_default_off(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    assert diagram.show_filter is False
    assert diagram.filter_highlight_layer is None
    assert diagram.filter_highlight_handle is None


def test_set_show_filter_on_adds_actor(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    diagram.set_show_filter(True)
    assert diagram.show_filter is True
    assert diagram.filter_highlight_layer is not None
    # The highlight layer is emitted to the backend.
    assert diagram.filter_highlight_layer.layer_id in backend.layers


def test_set_show_filter_off_removes_actor(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    diagram.set_show_filter(True)
    highlight_id = diagram.filter_highlight_layer.layer_id
    diagram.set_show_filter(False)
    assert diagram.show_filter is False
    assert diagram.filter_highlight_layer is None
    assert diagram.filter_highlight_handle is None
    assert highlight_id not in backend.layers


def test_set_show_filter_idempotent(cube_results, backend):
    """Toggling on twice doesn't double up the highlight handle."""
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    diagram.set_show_filter(True)
    first_handle = diagram.filter_highlight_handle
    diagram.set_show_filter(True)    # no-op
    assert diagram.filter_highlight_handle is first_handle


def test_show_filter_initially_bootstraps_at_attach(
    cube_results, backend,
):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    cut = _make_cut(with_bounding=True)
    style = SectionCutStyle(cut=cut, show_filter_initially=True)
    spec = DiagramSpec(
        kind="section_cut",
        selector=SlabSelector(component="cube mid-cut"),
        style=style,
    )
    diagram = SectionCutDiagram(spec, results, tag_map=tag_map)
    diagram.attach(backend, fem, scene)
    assert diagram.show_filter is True
    assert diagram.filter_highlight_layer is not None


def test_set_show_filter_before_attach_is_remembered(
    cube_results, backend,
):
    """Toggle on before attach → flag persists; attach() doesn't act on
    it (only ``show_filter_initially`` does), but a subsequent on/off
    round-trip works."""
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.set_show_filter(True)
    assert diagram.show_filter is True
    assert diagram.filter_highlight_layer is None     # not attached yet
    # Attach the diagram. The runtime flag stays True but the highlight
    # is not auto-built from set_show_filter — only show_filter_initially
    # bootstraps. The toggle still works after attach.
    diagram.attach(backend, fem, scene)
    diagram.set_show_filter(False)
    diagram.set_show_filter(True)
    assert diagram.filter_highlight_layer is not None


def test_detach_clears_filter_highlight(cube_results, backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(backend, fem, scene)
    diagram.set_show_filter(True)
    diagram.detach()
    assert diagram.filter_highlight_layer is None
    assert diagram.filter_highlight_handle is None
    assert diagram.show_filter is False


# ---------------------------------------------------------------------
# Render integration — real offscreen PyVistaQtBackend (ADR 0042)
# ---------------------------------------------------------------------

def test_pv_backend_renders_quad_with_backface(cube_results, pv_backend):
    """The polygon cut face renders through the real backend with a
    two-tone backface property — exercises the ``"polygon"`` token grid
    construction + ``back_color`` actor path (no GL pixels asserted)."""
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(pv_backend, fem, scene)
    handle = diagram._quad_handle
    assert handle.actor is not None
    # One polygon cell built via the explicit cells/celltypes path.
    assert handle.dataset.n_cells == 1
    # Two-tone: backend assigned a backface property.
    assert handle.actor.GetBackfaceProperty() is not None
    diagram.detach()


def test_pv_backend_renders_normal_arrow(cube_results, pv_backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(pv_backend, fem, scene)
    assert diagram._arrow_handle is not None
    assert diagram._arrow_handle.actor is not None
    diagram.detach()


def test_pv_backend_filter_highlight_actor(cube_results, pv_backend):
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(
        _spec(_make_cut(with_bounding=True)), results, tag_map=tag_map,
    )
    diagram.attach(pv_backend, fem, scene)
    diagram.set_show_filter(True)
    assert diagram.filter_highlight_handle is not None
    assert diagram.filter_highlight_handle.actor is not None
    diagram.detach()


# ---------------------------------------------------------------------
# Director — sweep fan-out
# ---------------------------------------------------------------------

def test_director_add_section_cut_sweep_fans_out(cube_results):
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, fem, h5, _ = cube_results
    director = ResultsDirector(results)

    # Three planes, same filter. The cut elements list is arbitrary
    # for the fan-out test — we only care that N defs → N layers.
    cut_a = _make_cut(element_ids=(10,))
    cut_b = _make_cut(element_ids=(10, 11))
    cut_c = _make_cut(element_ids=(10, 11, 12))
    sweep = SectionSweepDef(cuts=(cut_a, cut_b, cut_c))

    diagrams = director.add_section_cut_sweep(
        sweep, model_h5=h5, label_prefix="story",
    )
    assert len(diagrams) == 3
    # All in the same composition.
    geom = director.geometries.active
    assert geom is not None
    section_comp = next(
        (c for c in geom.compositions.compositions if c.name == "Section cuts"),
        None,
    )
    assert section_comp is not None
    assert all(d in section_comp.layers for d in diagrams)
    # Labels follow the prefix scheme.
    assert [d.spec.label for d in diagrams] == [
        "story[0]", "story[1]", "story[2]",
    ]


# ---------------------------------------------------------------------
# Phase D — _on_section_cut_rebuild via DiagramSettingsTab
# (QT_QPA_PLATFORM=offscreen is set at module import — see top of file.)
# ---------------------------------------------------------------------


def _build_settings_tab_against_director(director):
    """Construct a DiagramSettingsTab bound to a real ResultsDirector."""
    from qtpy import QtWidgets
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from apeGmsh.viewers.ui._diagram_settings_tab import DiagramSettingsTab
    return DiagramSettingsTab(director)


def test_rebuild_with_side_flip_swaps_registry_entry(cube_results):
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, _fem, h5, _ = cube_results
    director = ResultsDirector(results)
    cut = _make_cut(side="positive", with_bounding=True)
    diagram = director.add_section_cut(cut, model_h5=h5)
    tab = _build_settings_tab_against_director(director)

    tab._on_section_cut_rebuild(diagram, side="negative")

    # Registry now holds a different diagram.
    registry = list(director.registry)
    assert diagram not in registry
    assert len(registry) == 1
    new_diagram = registry[0]
    # New diagram carries the flipped side.
    assert new_diagram.spec.style.cut.side == "negative"
    # Composition layer list updated too — outline / settings tab see
    # the new instance.
    geom = director.geometries.active
    section_comp = next(
        c for c in geom.compositions.compositions if c.name == "Section cuts"
    )
    assert section_comp.layers == [new_diagram]


def test_rebuild_with_label_change_renames_cut(cube_results):
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, _fem, h5, _ = cube_results
    director = ResultsDirector(results)
    cut = _make_cut(with_bounding=True)
    diagram = director.add_section_cut(cut, model_h5=h5, label="original")
    tab = _build_settings_tab_against_director(director)

    tab._on_section_cut_rebuild(diagram, label="renamed cut")

    registry = list(director.registry)
    assert len(registry) == 1
    new_diagram = registry[0]
    assert new_diagram.spec.style.cut.label == "renamed cut"
    assert new_diagram.spec.label == "renamed cut"


def test_rebuild_preserves_show_filter_runtime_state(
    cube_results, headless_plotter,
):
    """A toggled-on filter overlay survives the side flip — the user
    shouldn't lose their highlight when editing the cut."""
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, fem, h5, _ = cube_results
    scene = build_fem_scene(fem)
    director = ResultsDirector(results)
    director.bind_plotter(headless_plotter, scene=scene)

    cut = _make_cut(side="positive", with_bounding=True)
    diagram = director.add_section_cut(cut, model_h5=h5)
    diagram.set_show_filter(True)
    assert diagram.show_filter is True

    tab = _build_settings_tab_against_director(director)
    tab._on_section_cut_rebuild(diagram, side="negative")

    new_diagram = list(director.registry)[0]
    assert new_diagram is not diagram
    # The new diagram attached with show_filter_initially=True so the
    # overlay is up immediately.
    assert new_diagram.show_filter is True
    assert new_diagram.filter_highlight_layer is not None


def test_rebuild_noop_when_nothing_changes(cube_results):
    """Calling rebuild with the same side / label is a no-op (no
    spurious registry swap on focus-loss with unchanged label).

    The panel renders ``cut.label`` first, falling back to
    ``spec.label`` — so "unchanged" means "matches what the user
    sees in the line edit", which is ``cut.label`` here.
    """
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, _fem, h5, _ = cube_results
    director = ResultsDirector(results)
    cut = _make_cut(side="positive", with_bounding=True)
    diagram = director.add_section_cut(cut, model_h5=h5)
    tab = _build_settings_tab_against_director(director)

    tab._on_section_cut_rebuild(diagram, side="positive")
    tab._on_section_cut_rebuild(diagram, label=cut.label)

    # Same diagram instance still in the registry.
    assert list(director.registry) == [diagram]


def test_rebuild_rejects_invalid_side(cube_results):
    from apeGmsh.viewers.diagrams import ResultsDirector
    results, _fem, h5, _ = cube_results
    director = ResultsDirector(results)
    diagram = director.add_section_cut(
        _make_cut(with_bounding=True), model_h5=h5,
    )
    tab = _build_settings_tab_against_director(director)
    tab._on_section_cut_rebuild(diagram, side="upward")    # invalid
    assert list(director.registry) == [diagram]


# ---------------------------------------------------------------------
# Phase E — session JSON round-trip + restore against a real diagram
# ---------------------------------------------------------------------

def test_deserialized_spec_attaches_against_real_fixture(
    cube_results, headless_plotter,
):
    """A spec that went through serialize → JSON → deserialize must
    still produce a working diagram against the fixture's model.h5.

    The point is to catch any silent dataclass-construction issue
    (tuple→list→tuple coercion on plane_point / element_ids, nested
    SectionCutDef rehydration on the style) that the pure JSON
    round-trip tests in test_session_persistence.py would miss
    because they don't touch the FEM side.
    """
    import json as _json
    from apeGmsh.viewers.diagrams._session import (
        deserialize_spec, serialize_spec,
    )
    results, fem, h5, ops_to_fem = cube_results
    scene = build_fem_scene(fem)

    # The fixture's model.h5 maps OpenSees tags 10, 11, 12 → FEM eids.
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 12),
        side="positive",
        label="cube mid-cut",
        bounding_polygon=(
            (0.0, 0.0, 0.5), (1.0, 0.0, 0.5),
            (1.0, 1.0, 0.5), (0.0, 1.0, 0.5),
        ),
    )
    spec = DiagramSpec(
        kind="section_cut",
        selector=SlabSelector(component=cut.label),
        style=SectionCutStyle(cut=cut, show_filter_initially=True),
        label=cut.label,
    )

    # Full text-mode round-trip: serialize → JSON dump → JSON load →
    # deserialize. Mirrors what ``save_session`` / ``load_session`` do.
    payload = _json.loads(_json.dumps(serialize_spec(spec)))
    restored = deserialize_spec(payload)

    tag_map = FemToOpsTagMap.from_h5(h5)
    diagram = SectionCutDiagram(restored, results, tag_map=tag_map)
    diagram.attach(headless_plotter, fem, scene)

    # The whole point of the round-trip: cut survives intact and the
    # bootstrap filter highlight applied because show_filter_initially
    # round-tripped too.
    assert diagram.cut.element_ids == (10, 11, 12)
    assert diagram.show_filter is True
    assert diagram.filter_highlight_layer is not None
