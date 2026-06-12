"""ADR 0058 S2a — per-geometry scene instances + active switching.

S1 left ``director.scene_for(geometry)`` resolving every geometry to
the single bound scene. S2a makes the seam true: the cache is seeded
with the boot scene at ``bind_plotter`` and misses materialize through
a viewer-injected ``scene_factory`` (the director itself stays
pyvista-free — ADR 0042 INV-2). ``clone_scene`` is the materialization
primitive: deep-copied grid born undeformed, shared index arrays.

Headless tests cover the clone contract + the director cache; the
qt-marked test drives a real viewer through an A→B→A active-geometry
switch and asserts each scene's grid keeps its own deformation state
while ``viewer._scene`` tracks the active geometry. (S2a's
active-only visibility flip was superseded by ADR 0058 S2b: every
geometry with ``visible=True`` renders concurrently, so an active
switch no longer hides the other pair — see
test_scene_instances_s2b.py for the visibility-flag coverage.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import _stub_model_h5_path
from tests.viewers.conftest import RecordingBackend


_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")


# =====================================================================
# clone_scene — the materialization primitive
# =====================================================================

def _tiny_scene():
    """One-tet FEMSceneData with a non-trivial deformation applied."""
    import pyvista as pv
    from apeGmsh.viewers.scene.fem_scene import FEMSceneData

    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    celltypes = np.array([10], dtype=np.uint8)   # VTK_TETRA
    grid = pv.UnstructuredGrid(cells, celltypes, points.copy())
    node_ids = np.array([10, 11, 12, 13], dtype=np.int64)
    grid.point_data["node_id"] = node_ids
    scene = FEMSceneData(
        grid=grid,
        node_ids=node_ids,
        node_id_to_idx={10: 0, 11: 1, 12: 2, 13: 3},
        cell_to_element_id=np.array([7], dtype=np.int64),
        element_id_to_cell={7: 0},
        model_diagonal=float(np.sqrt(3.0)),
        cell_dim=np.array([3], dtype=np.int8),
        reference_points=points.copy(),
    )
    # Deform the source + ghost-hide its cell so the clone contract
    # (born undeformed AND unhidden) is actually exercised.
    grid.points = points + 0.25
    grid.cell_data["vtkGhostType"] = np.array([1], dtype=np.uint8)
    return scene


def test_clone_scene_born_undeformed_while_source_stays_deformed():
    from apeGmsh.viewers.scene.fem_scene import clone_scene

    src = _tiny_scene()
    clone = clone_scene(src)
    np.testing.assert_allclose(
        np.asarray(clone.grid.points), src.reference_points,
    )
    # Source untouched — still deformed.
    assert not np.allclose(np.asarray(src.grid.points), src.reference_points)


def test_clone_scene_shares_index_arrays_owns_points():
    from apeGmsh.viewers.scene.fem_scene import clone_scene

    src = _tiny_scene()
    clone = clone_scene(src)
    # Shared (immutable, every scene indexes the same model).
    assert clone.node_ids is src.node_ids
    assert clone.node_id_to_idx is src.node_id_to_idx
    assert clone.cell_to_element_id is src.cell_to_element_id
    assert clone.element_id_to_cell is src.element_id_to_cell
    assert clone.cell_dim is src.cell_dim
    assert clone.model_diagonal == src.model_diagonal
    # Owned — mutating the clone's grid / baseline can't leak back.
    assert clone.grid is not src.grid
    assert clone.reference_points is not src.reference_points
    np.testing.assert_allclose(clone.reference_points, src.reference_points)
    clone.grid.points = np.asarray(clone.grid.points) + 1.0
    assert not np.allclose(
        np.asarray(src.grid.points), np.asarray(clone.grid.points),
    )


def test_clone_scene_render_fields_start_none_and_ghosts_clear():
    from apeGmsh.viewers.scene.fem_scene import clone_scene

    src = _tiny_scene()
    src.actor = object()
    src.pick_engine = object()
    src.element_visibility = object()
    src.opacity_controller = object()
    clone = clone_scene(src)
    assert clone.actor is None
    assert clone.pick_engine is None
    assert clone.element_visibility is None
    assert clone.opacity_controller is None
    assert clone.node_tree is None
    # Clones are born unhidden — the viewer re-applies view-global
    # hide layers (dim filter / stage activation) at materialization.
    assert int(np.asarray(clone.grid.cell_data["vtkGhostType"]).sum()) == 0
    # Source ghosts untouched.
    assert int(np.asarray(src.grid.cell_data["vtkGhostType"]).sum()) == 1


# =====================================================================
# Director: per-geometry cache + factory + removal
# =====================================================================

@pytest.fixture
def director():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    from apeGmsh.results import Results
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    return ResultsDirector(
        Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path()),
    )


def test_scene_for_distinct_per_geometry_with_factory(director):
    boot_scene = object()
    made: list = []

    def factory(geom):
        s = object()
        made.append((geom.id, s))
        return s

    director.bind_plotter(
        RecordingBackend(), scene=boot_scene, scene_factory=factory,
    )
    active = director.geometries.active
    other = director.geometries.add("Geometry B", make_active=False)
    # Seeded: the boot geometry keeps the bound scene (no factory call).
    assert director.scene_for(active) is boot_scene
    # Miss: materialized through the factory, cached on re-access.
    scene_b = director.scene_for(other)
    assert scene_b is not boot_scene
    assert director.scene_for(other) is scene_b
    assert made == [(other.id, scene_b)]
    assert director.scene_for(active) is not director.scene_for(other)


def test_scene_for_without_factory_keeps_s1_fallback(director):
    boot_scene = object()
    director.bind_plotter(RecordingBackend(), scene=boot_scene)
    other = director.geometries.add("Geometry B", make_active=False)
    assert director.scene_for(other) is boot_scene
    assert director.scene_for(None) is boot_scene


def test_geometry_removed_drops_cached_scene(director):
    boot_scene = object()
    director.bind_plotter(
        RecordingBackend(), scene=boot_scene,
        scene_factory=lambda geom: object(),
    )
    other = director.geometries.add("Geometry B", make_active=False)
    scene_b = director.scene_for(other)
    assert scene_b in director.materialized_scenes()
    assert director.geometries.remove(other.id)
    assert scene_b not in director.materialized_scenes()
    # The boot seed survives.
    assert boot_scene in director.materialized_scenes()


def test_unbind_plotter_clears_scene_cache_and_factory(director):
    director.bind_plotter(
        RecordingBackend(), scene=object(),
        scene_factory=lambda geom: object(),
    )
    other = director.geometries.add("Geometry B", make_active=False)
    director.scene_for(other)
    assert director.materialized_scenes()
    director.unbind_plotter()
    assert director.materialized_scenes() == []
    assert director.scene_for(other) is None


def test_factory_returning_none_falls_back_to_bound_scene(director):
    boot_scene = object()
    director.bind_plotter(
        RecordingBackend(), scene=boot_scene,
        scene_factory=lambda geom: None,
    )
    other = director.geometries.add("Geometry B", make_active=False)
    assert director.scene_for(other) is boot_scene
    # A failed materialization is not cached as a scene.
    assert director.materialized_scenes() == [boot_scene]


# =====================================================================
# Qt — activation flip on a real viewer (local-only; `pytest -m qt`)
# =====================================================================

@pytest.fixture
def deforming_results(g, tmp_path: Path):
    """Tiny native Results whose displacement field is non-zero, so a
    deform-enabled geometry visibly leaves the reference position."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "s2a.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": np.ones((3, n_nodes)),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.mark.qt
def test_active_geometry_switch_flips_substrate_actors(deforming_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        deforming_results, title="s2a-switch",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geoms = director.geometries
            geom_a = geoms.active
            scene_a = director.scene_for(geom_a)
            geom_b = geoms.add("Geometry B", make_active=False)
            geoms.set_deformation(
                geom_b.id, enabled=True,
                field="displacement", scale=2.0,
            )
            geoms.set_active(geom_b.id)
            scene_b = director.scene_for(geom_b)
            seen["distinct_scenes"] = scene_b is not scene_a
            seen["scene_prop_is_b"] = viewer._scene is scene_b
            pair_a = viewer._scene_actors[geom_a.id]
            pair_b = viewer._scene_actors[geom_b.id]
            # ADR 0058 S2b: both geometries default visible — an
            # active switch re-points the editing target but hides
            # nothing (the S2a active-only flip is superseded).
            seen["a_visible"] = all(
                bool(x.GetVisibility()) for x in pair_a
            )
            seen["b_visible"] = all(
                bool(x.GetVisibility()) for x in pair_b
            )
            # B's grid carries B's deform; A's stays at reference.
            seen["b_deformed"] = not np.allclose(
                np.asarray(scene_b.grid.points), scene_b.reference_points,
            )
            seen["a_at_reference"] = np.allclose(
                np.asarray(scene_a.grid.points), scene_a.reference_points,
            )
            # Switch back — handles re-point, A still reference.
            geoms.set_active(geom_a.id)
            seen["a_visible_again"] = all(
                bool(x.GetVisibility()) for x in pair_a
            )
            seen["b_still_visible"] = all(
                bool(x.GetVisibility()) for x in pair_b
            )
            seen["scene_prop_is_a"] = viewer._scene is scene_a
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("distinct_scenes") is True
    assert seen.get("scene_prop_is_b") is True
    assert seen.get("a_visible") is True
    assert seen.get("b_visible") is True
    assert seen.get("b_deformed") is True
    assert seen.get("a_at_reference") is True
    assert seen.get("a_visible_again") is True
    assert seen.get("b_still_visible") is True
    assert seen.get("scene_prop_is_a") is True
