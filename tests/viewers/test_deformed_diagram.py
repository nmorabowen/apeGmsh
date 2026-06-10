"""DeformedShapeDiagram — emits warped MeshLayers through the backend.

Verifies the warp math on the *emitted* deformed layer:
``deformed_points == base_points + scale * displacement_at_step``.
Uses the recording stub backend (no GL); the ``backend`` fixture is in
tests/viewers/conftest.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DeformedShapeDiagram,
    DeformedShapeStyle,
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.scene_ir import MeshLayer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


@pytest.fixture
def results_with_displacement_xyz(g, tmp_path: Path):
    """Native HDF5 with displacement_x/y/z = nid + t*{1,2,3}."""
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
    path = tmp_path / "disp_xyz.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": base + t * 1.0,
                "displacement_y": base + t * 2.0,
                "displacement_z": base + t * 3.0,
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _make_spec(scale=1.0, show_undeformed=True) -> DiagramSpec:
    return DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),
        style=DeformedShapeStyle(scale=scale, show_undeformed=show_undeformed),
    )


def test_construction_requires_deformed_style(results_with_displacement_xyz):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="DeformedShapeStyle"):
        DeformedShapeDiagram(bad, results_with_displacement_xyz)


def test_attach_emits_warped_deformed_layer(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(scale=1.0), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)

    layer = diagram._deformed_layer
    assert isinstance(layer, MeshLayer)
    base = diagram._base_points
    fem_ids = diagram._fem_ids_to_read.astype(np.float64)
    # Step 0: disp_{x,y,z} = nid.
    expected = base + np.column_stack([fem_ids, fem_ids, fem_ids])
    np.testing.assert_allclose(layer.points.coords, expected, atol=1e-4)


def test_step_change_remixes_displacement(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(scale=1.0), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    base = diagram._base_points
    fem_ids = diagram._fem_ids_to_read.astype(np.float64)

    for t in (1, 2, 0):
        diagram.update_to_step(t)
        expected = base + np.column_stack(
            [fem_ids + t * 1.0, fem_ids + t * 2.0, fem_ids + t * 3.0]
        )
        np.testing.assert_allclose(
            diagram._deformed_layer.points.coords, expected, atol=1e-4
        )


def test_set_scale_re_warps(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(scale=1.0), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    base = diagram._base_points
    fem_ids = diagram._fem_ids_to_read.astype(np.float64)

    diagram.update_to_step(2)
    diagram.set_scale(10.0)
    expected = base + 10.0 * np.column_stack(
        [fem_ids + 2.0, fem_ids + 4.0, fem_ids + 6.0]
    )
    np.testing.assert_allclose(
        diagram._deformed_layer.points.coords, expected, atol=1e-3
    )


def test_current_scale_uses_runtime_override(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(scale=2.0), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    assert diagram.current_scale() == 2.0
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


def test_zero_scale_yields_undeformed(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(scale=0.0), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    np.testing.assert_allclose(
        diagram._deformed_layer.points.coords, diagram._base_points, atol=1e-4
    )


def test_undeformed_layer_present_by_default(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=True), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    assert diagram._undeformed_handle is not None
    assert len(backend.layers) == 2
    # The undeformed layer is a wireframe ghost.
    undef = backend.layers[diagram._undeformed_handle.layer_id]
    assert undef.wireframe is True


def test_undeformed_layer_absent_when_disabled(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=False), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    assert diagram._undeformed_handle is None
    assert len(backend.layers) == 1


def test_set_show_undeformed_toggles_layer(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=True), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    handle = diagram._undeformed_handle
    diagram.set_show_undeformed(False)
    assert handle.visible is False
    diagram.set_show_undeformed(True)
    assert handle.visible is True


def test_detach_removes_layers_and_clears_state(results_with_displacement_xyz, backend):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    diagram.detach()
    assert diagram._deformed_layer is None
    assert diagram._cells is None
    assert not diagram.is_attached
    assert backend.layers == {}


# =====================================================================
# Runtime show_undeformed — gate / show must not resurrect a disabled
# ghost (Phase-1 event/state fixes)
# =====================================================================


def test_set_show_undeformed_records_runtime_flag(results_with_displacement_xyz, backend):
    """The settings tab restores its checkbox from
    ``_runtime_show_undeformed`` — the toggle must write it."""
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=True), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    assert diagram._runtime_show_undeformed is None
    diagram.set_show_undeformed(False)
    assert diagram._runtime_show_undeformed is False
    diagram.set_show_undeformed(True)
    assert diagram._runtime_show_undeformed is True


def test_disabled_ghost_survives_visibility_cycle(results_with_displacement_xyz, backend):
    """set_visible(False→True) must not resurrect a ghost the user
    toggled off at runtime."""
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=True), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    ghost = diagram._undeformed_handle
    diagram.set_show_undeformed(False)
    diagram.set_visible(False)
    diagram.set_visible(True)
    assert diagram._deformed_handle.visible is True
    assert ghost.visible is False


def test_gate_preserves_intent_and_ghost_state(results_with_displacement_xyz, backend):
    """The composition gate (apply_effective_visibility) routes the
    backend handles and leaves both the user-intent flag and the
    runtime ghost toggle untouched."""
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=True), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    ghost = diagram._undeformed_handle
    diagram.set_show_undeformed(False)

    # Gate off (layer not in active composition) …
    diagram.apply_effective_visibility(False)
    assert diagram._deformed_handle.visible is False
    assert diagram.is_visible is True            # intent preserved
    # … and back on: the disabled ghost stays hidden.
    diagram.apply_effective_visibility(True)
    assert diagram._deformed_handle.visible is True
    assert ghost.visible is False


def test_disabled_ghost_survives_reattach(results_with_displacement_xyz, backend):
    """Stage changes detach + re-attach every diagram; a runtime-
    disabled ghost must not come back from style.show_undeformed."""
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(_make_spec(show_undeformed=True), results_with_displacement_xyz)
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    diagram.set_show_undeformed(False)
    diagram.detach()
    diagram.attach(backend, results_with_displacement_xyz.fem, scene)
    assert diagram._undeformed_handle is None
    assert len(backend.layers) == 1
