"""Every rendering diagram follows the deformed substrate.

Regression suite for the deform-state pipeline: post-ADR-0042 every
diagram emits backend-owned dataset COPIES, so the old viewer-side
``_sync_layer_grids`` walk over ``d._actors`` (which migrated diagrams
never populate) silently stopped moving substrate-extracted layers.
``Diagram.sync_substrate_points`` is now the ONLY deformation fan-out
and every rendering diagram must implement it.

These tests drive the hook exactly like the viewer's DEFORM pump: shift
every substrate point by +5 in Y, call ``sync_substrate_points``, and
assert the emitted layer's points moved — then pass ``None`` and assert
the layer snapped back to the reference configuration. GL-free (shared
``backend`` recording stub).

line_force / gauss_marker / vector_glyph / reactions / loads already
have deform-follow coverage in their own test files.
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
    FiberSectionDiagram,
    FiberSectionStyle,
    LayerStackDiagram,
    LayerStackStyle,
    SlabSelector,
    SpringForceDiagram,
    SpringForceStyle,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5, _stub_model_h5_path

_SPRING_FIXTURE = Path("tests/fixtures/results/zl_springs.mpco")


# =====================================================================
# Helpers
# =====================================================================

def _shifted(scene, dy: float = 5.0) -> np.ndarray:
    target = np.asarray(scene.grid.points, dtype=np.float64).copy()
    target[:, 1] += dy
    return target


def _layer_points(diagram) -> np.ndarray:
    layer = diagram._layer
    pts = getattr(layer, "points", None) or getattr(layer, "positions", None)
    return np.asarray(pts.coords, dtype=np.float64).copy()


def _assert_follows_and_resets(diagram, scene) -> None:
    """The shared contract: +5Y shift moves every layer point by +5Y;
    ``None`` resets to the reference configuration."""
    before = _layer_points(diagram)
    diagram.sync_substrate_points(_shifted(scene), scene)
    after = _layer_points(diagram)
    np.testing.assert_allclose(
        after - before, np.tile([0.0, 5.0, 0.0], (before.shape[0], 1)),
        atol=1e-5,
    )
    # Reset: the pump restores scene.grid.points to reference before
    # fanning out None — here the grid never moved, so None must land
    # back on the original points.
    diagram.sync_substrate_points(None, scene)
    np.testing.assert_allclose(_layer_points(diagram), before, atol=1e-5)


def _all_element_ids(fem) -> np.ndarray:
    chunks = [
        np.asarray(group.ids, dtype=np.int64) for group in fem.elements
    ]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def cube_results(g, tmp_path: Path):
    """Meshed cube with a nodal scalar + 1-GP and 2-GP gauss groups —
    feeds every contour path."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = _all_element_ids(fem)
    n_steps = 2

    disp = np.tile(node_ids.astype(np.float64), (n_steps, 1))
    sxx_1gp = np.zeros((n_steps, elem_ids.size, 1), dtype=np.float64)
    sxx_1gp[:, :, 0] = elem_ids * 10.0
    sxx_2gp = np.zeros((n_steps, elem_ids.size, 2), dtype=np.float64)
    sxx_2gp[:, :, 0] = elem_ids * 10.0
    sxx_2gp[:, :, 1] = elem_ids * 10.0 + 1.0

    path = tmp_path / "cube.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "g1",
            class_tag=4, int_rule=1,
            element_index=elem_ids,
            natural_coords=np.array([[0.25, 0.25, 0.25]]),
            components={"stress_xx": sxx_1gp},
        )
        w.write_gauss_group(
            sid, "partition_0", "g2",
            class_tag=4, int_rule=2,
            element_index=elem_ids,
            natural_coords=np.array(
                [[0.2, 0.2, 0.2], [0.4, 0.2, 0.2]],
            ),
            components={"stress_yy": sxx_2gp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _contour(results, component, topology, averaging="averaged"):
    return ContourDiagram(
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component=component),
            style=ContourStyle(topology=topology, averaging=averaging),
        ),
        results,
    )


# =====================================================================
# Contour — all dispatch paths
# =====================================================================

def test_contour_nodes_follows_deformation(cube_results, backend) -> None:
    scene = build_fem_scene(cube_results.fem)
    d = _contour(cube_results, "displacement_z", "nodes")
    d.attach(backend, cube_results.fem, scene)
    _assert_follows_and_resets(d, scene)


def test_contour_gauss_cell_follows_deformation(cube_results, backend) -> None:
    # 1 GP + discrete -> cell-data path (extract_cells submesh).
    scene = build_fem_scene(cube_results.fem)
    d = _contour(cube_results, "stress_xx", "gauss", "discrete")
    d.attach(backend, cube_results.fem, scene)
    assert d._effective_topology == "gauss_cell"
    _assert_follows_and_resets(d, scene)


def test_contour_gauss_cell_averaged_follows_deformation(
    cube_results, backend,
) -> None:
    # 1 GP + averaged -> spread-to-corners point path (extract_points).
    scene = build_fem_scene(cube_results.fem)
    d = _contour(cube_results, "stress_xx", "gauss", "averaged")
    d.attach(backend, cube_results.fem, scene)
    assert d._effective_topology == "gauss_cell_averaged"
    _assert_follows_and_resets(d, scene)


def test_contour_gauss_shattered_follows_deformation(
    cube_results, backend,
) -> None:
    # 2 GPs + discrete -> shattered submesh (separate_cells). The
    # vtkOriginalPointIds map must survive the shatter as carried
    # point data for the re-sample to work.
    scene = build_fem_scene(cube_results.fem)
    d = _contour(cube_results, "stress_yy", "gauss", "discrete")
    d.attach(backend, cube_results.fem, scene)
    assert d._effective_topology == "gauss_node_discrete"
    assert d._substrate_rows is not None
    _assert_follows_and_resets(d, scene)


# =====================================================================
# Fiber section — owned dot cloud
# =====================================================================

@pytest.fixture
def fiber_results(g, tmp_path: Path):
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0, label="p0")
    p1 = g.model.geometry.add_point(1.0, 0.0, 0.0, label="p1")
    g.model.geometry.add_line(p0, p1, label="seg")
    g.physical.add_curve(["seg"], name="Beam")
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    line_eids = sorted(
        int(x)
        for group in fem.elements
        if group.element_type.dim == 1
        for x in group.ids
    )
    gps, fibers, n_steps = 2, 4, 2
    rows = [
        (eid, gp, float(fk - 1.5), float(gp - 0.5), 1.0, 0)
        for eid in line_eids
        for gp in range(gps)
        for fk in range(fibers)
    ]
    n_rows = len(rows)
    values = np.tile(
        np.arange(n_rows, dtype=np.float64), (n_steps, 1),
    )
    path = tmp_path / "fibers.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_fibers_group(
            sid, "partition_0", group_id="g0",
            section_tag=10, section_class="FiberSection",
            element_index=np.asarray([r[0] for r in rows], dtype=np.int64),
            gp_index=np.asarray([r[1] for r in rows], dtype=np.int64),
            y=np.asarray([r[2] for r in rows]),
            z=np.asarray([r[3] for r in rows]),
            area=np.asarray([r[4] for r in rows]),
            material_tag=np.asarray([r[5] for r in rows], dtype=np.int64),
            components={"fiber_stress": values},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def test_fiber_cloud_follows_deformation(fiber_results, backend) -> None:
    scene = build_fem_scene(fiber_results.fem)
    d = FiberSectionDiagram(
        DiagramSpec(
            kind="fiber_section",
            selector=SlabSelector(component="fiber_stress"),
            style=FiberSectionStyle(),
        ),
        fiber_results,
    )
    d.attach(backend, fiber_results.fem, scene)
    assert d._points is not None
    # A rigid +5Y substrate shift translates the chord, leaves the
    # frame unchanged, and so translates every fiber dot by +5Y.
    _assert_follows_and_resets(d, scene)


# =====================================================================
# Layer stack — extracted shell submesh
# =====================================================================

@pytest.fixture
def layer_results(g, tmp_path: Path):
    g.model.geometry.add_rectangle(0, 0, 0, 2, 1, label="plate")
    g.physical.add_surface("plate", name="Plate")
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)

    shell_eids = sorted(
        int(x)
        for group in fem.elements
        if group.element_type.dim == 2
        for x in group.ids
    )
    gps, layers, n_steps = 1, 2, 2
    rows = [
        (eid, gp, layer, 0, 0.25)
        for eid in shell_eids
        for gp in range(gps)
        for layer in range(layers)
    ]
    n_rows = len(rows)
    values = np.tile(np.arange(n_rows, dtype=np.float64), (n_steps, 1))
    path = tmp_path / "layers.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_layers_group(
            sid, "partition_0", group_id="g0",
            element_index=np.asarray([r[0] for r in rows], dtype=np.int64),
            gp_index=np.asarray([r[1] for r in rows], dtype=np.int64),
            layer_index=np.asarray([r[2] for r in rows], dtype=np.int64),
            sub_gp_index=np.asarray([r[3] for r in rows], dtype=np.int64),
            thickness=np.asarray([r[4] for r in rows]),
            local_axes_quaternion=np.tile(
                np.array([1.0, 0.0, 0.0, 0.0]), (n_rows, 1),
            ),
            components={"stress_xx": values},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def test_layer_stack_follows_deformation(layer_results, backend) -> None:
    scene = build_fem_scene(layer_results.fem)
    d = LayerStackDiagram(
        DiagramSpec(
            kind="layer_stack",
            selector=SlabSelector(component="stress_xx"),
            style=LayerStackStyle(),
        ),
        layer_results,
    )
    d.attach(backend, layer_results.fem, scene)
    assert d._substrate_rows is not None
    _assert_follows_and_resets(d, scene)


# =====================================================================
# Spring force — owned glyph anchors
# =====================================================================

def test_spring_force_follows_deformation(backend) -> None:
    if not _SPRING_FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_SPRING_FIXTURE}")
    results = Results.from_mpco(
        _SPRING_FIXTURE, model_h5=_stub_model_h5_path(),
    )
    scoped = results.stage(results.stages[0].name)
    scene = build_fem_scene(scoped.fem)
    d = SpringForceDiagram(
        DiagramSpec(
            kind="spring_force",
            selector=SlabSelector(component="spring_force_0"),
            style=SpringForceStyle(scale=1.0),
        ),
        scoped,
    )
    d.attach(backend, scoped.fem, scene)
    assert d._substrate_rows is not None
    assert (d._substrate_rows >= 0).all()
    _assert_follows_and_resets(d, scene)
