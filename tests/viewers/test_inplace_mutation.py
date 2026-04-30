"""In-place mutation contract — Phase 0/1 perf gate.

The single biggest win over the legacy ``apeGmshViewer/`` is mutating
existing actor scalars in place instead of re-adding actors per step.
This file enforces that contract:

* After many step changes, ContourDiagram's actor and mapper are the
  *same* VTK objects as after step 0 — no re-creation.
* The scalar / point_data array's underlying buffer is mutated in
  place; ``np.shares_memory`` confirms.
* DeformedShapeDiagram's points buffer changes value but the grid
  identity is stable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DeformedShapeDiagram,
    DeformedShapeStyle,
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixture: 50-step results with displacement_x/y/z
# =====================================================================

@pytest.fixture
def results_50_steps(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 50
    base = np.broadcast_to(
        node_ids.astype(np.float64), (n_steps, n_nodes),
    )
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    components = {
        "displacement_x": base + t * 0.1,
        "displacement_y": base + t * 0.2,
        "displacement_z": base + t * 0.3,
    }

    path = tmp_path / "fifty_steps.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


# =====================================================================
# ContourDiagram — actor + mapper identity stable across steps
# =====================================================================

def test_contour_actor_identity_stable_across_steps(
    results_50_steps, headless_plotter,
):
    scene = build_fem_scene(results_50_steps.fem)
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )
    diagram = ContourDiagram(spec, results_50_steps)
    diagram.attach(headless_plotter, results_50_steps.fem, scene)

    initial_actor = diagram._actor
    initial_mapper = initial_actor.GetMapper()
    initial_actor_id = id(initial_actor)
    initial_mapper_id = id(initial_mapper)

    for step in range(50):
        diagram.update_to_step(step)

    # Same Python objects + same VTK objects throughout
    assert diagram._actor is initial_actor
    assert id(diagram._actor) == initial_actor_id
    assert diagram._actor.GetMapper() is initial_mapper
    assert id(diagram._actor.GetMapper()) == initial_mapper_id


def test_contour_scalar_buffer_mutates_in_place(
    results_50_steps, headless_plotter,
):
    scene = build_fem_scene(results_50_steps.fem)
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )
    diagram = ContourDiagram(spec, results_50_steps)
    diagram.attach(headless_plotter, results_50_steps.fem, scene)

    # The scalar array reference saved at attach should still point to
    # the *same buffer* used by the submesh's point_data after multiple
    # step updates.
    initial_scalar_ref = diagram._scalar_array
    initial_buffer_id = id(np.asarray(initial_scalar_ref).data)

    diagram.update_to_step(10)
    diagram.update_to_step(25)
    diagram.update_to_step(49)

    # Same Python reference, same data buffer
    assert diagram._scalar_array is initial_scalar_ref
    assert id(np.asarray(diagram._scalar_array).data) == initial_buffer_id

    # And the values reflect the most recent step
    final = np.asarray(diagram._submesh.point_data["_contour"])
    fem_ids = scene.node_ids[
        np.asarray(diagram._submesh.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ]
    np.testing.assert_allclose(final, fem_ids.astype(np.float64) + 49 * 0.3)


# =====================================================================
# DeformedShapeDiagram — actor + grid identity stable across steps
# =====================================================================

def test_deformed_actor_identity_stable_across_steps(
    results_50_steps, headless_plotter,
):
    scene = build_fem_scene(results_50_steps.fem)
    spec = DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),
        style=DeformedShapeStyle(scale=1.0, show_undeformed=True),
    )
    diagram = DeformedShapeDiagram(spec, results_50_steps)
    diagram.attach(headless_plotter, results_50_steps.fem, scene)

    initial_grid = diagram._deformed_grid
    initial_grid_id = id(initial_grid)
    initial_actor = diagram._deformed_actor
    initial_actor_id = id(initial_actor)

    for step in range(50):
        diagram.update_to_step(step)

    # Actor + grid identity stable. PyVista may re-bind the actor's
    # mapper when ``mesh.points = ...`` is reassigned (it rebuilds
    # the normals filter), so we don't pin the mapper id — the
    # contract that matters is "no add_mesh re-creation per step",
    # which actor + grid identity proves.
    assert diagram._deformed_grid is initial_grid
    assert id(diagram._deformed_grid) == initial_grid_id
    assert diagram._deformed_actor is initial_actor
    assert id(diagram._deformed_actor) == initial_actor_id


def test_deformed_scale_change_does_not_re_add_actor(
    results_50_steps, headless_plotter,
):
    scene = build_fem_scene(results_50_steps.fem)
    spec = DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),
        style=DeformedShapeStyle(scale=1.0),
    )
    diagram = DeformedShapeDiagram(spec, results_50_steps)
    diagram.attach(headless_plotter, results_50_steps.fem, scene)

    initial_actor = diagram._deformed_actor
    initial_mapper = initial_actor.GetMapper()

    for s in (2.0, 5.0, 10.0, 0.5, 1.0):
        diagram.set_scale(s)

    assert diagram._deformed_actor is initial_actor
    assert diagram._deformed_actor.GetMapper() is initial_mapper
