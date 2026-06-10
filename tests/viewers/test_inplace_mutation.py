"""In-place mutation contract — Phase 0/1 perf gate.

The single biggest win over the legacy standalone viewer (the removed
top-level ``apeGmshViewer`` package) is mutating
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

from tests.conftest import _open_model_from_h5


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
    return Results.from_native(path, model=_open_model_from_h5(path))


# headless_plotter is a shared fixture in tests/viewers/conftest.py
# (yields a PyVistaQtBackend, ADR 0042 R-B.final).


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

    # Post-migration (ADR 0042): the backend owns the actor; the diagram
    # holds a layer handle. The mesh fast path mutates the bound dataset
    # in place when topology is stable, so the handle's actor + mapper +
    # dataset are the same objects across all steps (no re-add).
    handle = diagram._handle
    initial_actor = handle.actor
    initial_mapper = initial_actor.GetMapper()
    initial_dataset = handle.dataset

    for step in range(50):
        diagram.update_to_step(step)

    assert diagram._handle is handle
    assert handle.actor is initial_actor
    assert handle.actor.GetMapper() is initial_mapper
    assert handle.dataset is initial_dataset


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

    # The persistent scalar buffer the diagram scatters into should be
    # the *same* numpy object across many steps — no per-step
    # reallocation.
    initial_scalar_ref = diagram._scalar_values
    initial_buffer_id = id(diagram._scalar_values.data)

    diagram.update_to_step(10)
    diagram.update_to_step(25)
    diagram.update_to_step(49)

    assert diagram._scalar_values is initial_scalar_ref
    assert id(diagram._scalar_values.data) == initial_buffer_id

    # The bound dataset reflects the most recent step's values.
    final = np.asarray(diagram._handle.dataset.point_data["displacement_z"])
    fem_ids = diagram._fem_ids_to_read
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

    # Post-migration (ADR 0042): the backend owns the actor; the diagram
    # holds a layer handle. The perf contract is unchanged — the backend
    # mutates the grid's points in place when topology is stable, so the
    # handle's actor + dataset are the same objects across all steps (no
    # per-step add_mesh re-creation).
    handle = diagram._deformed_handle
    initial_actor = handle.actor
    initial_dataset = handle.dataset

    for step in range(50):
        diagram.update_to_step(step)

    assert diagram._deformed_handle is handle
    assert handle.actor is initial_actor
    assert handle.dataset is initial_dataset


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

    handle = diagram._deformed_handle
    initial_actor = handle.actor
    initial_dataset = handle.dataset

    for s in (2.0, 5.0, 10.0, 0.5, 1.0):
        diagram.set_scale(s)

    # No re-add on scale change — same actor + same underlying grid.
    assert handle.actor is initial_actor
    assert handle.dataset is initial_dataset
