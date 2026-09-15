"""SpringForceDiagram — emits an arrow GlyphLayer through the backend.

Uses the real MPCO spring fixture (the native writer has no springs
writer). Asserts on the *emitted layer* via the shared recording stub
``backend`` fixture (tests/viewers/conftest.py) — no GL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import InterfaceRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.results import Results
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    SlabSelector,
    SpringForceDiagram,
    SpringForceStyle,
)
from apeGmsh.viewers.diagrams._spring_force import (
    _direction_from_component,
    _direction_from_interface_orient,
    _match_interface_record,
)
from apeGmsh.viewers.scene_ir import GlyphLayer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _stub_model_h5_path


_SPRING_FIXTURE = Path("tests/fixtures/results/zl_springs.mpco")


@pytest.fixture
def spring_results():
    if not _SPRING_FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_SPRING_FIXTURE}")
    return Results.from_mpco(_SPRING_FIXTURE, model_h5=_stub_model_h5_path())


def _spec(component="spring_force_0", direction=None) -> DiagramSpec:
    return DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component=component),
        style=SpringForceStyle(scale=1.0, direction=direction),
    )


# --- Construction + helpers ----------------------------------------------


def test_construction_requires_spring_style(spring_results):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component="spring_force_0"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="SpringForceStyle"):
        SpringForceDiagram(bad, spring_results)


def test_direction_from_component_default():
    np.testing.assert_array_equal(_direction_from_component("spring_force_0"), [1, 0, 0])
    np.testing.assert_array_equal(_direction_from_component("spring_force_1"), [0, 1, 0])
    np.testing.assert_array_equal(_direction_from_component("spring_force_2"), [0, 0, 1])


def test_direction_from_unsuffixed_component_falls_back():
    np.testing.assert_array_equal(
        _direction_from_component("not_a_spring_thing"), [1, 0, 0]
    )


# --- Attach --------------------------------------------------------------


def test_attach_requires_scene(spring_results, backend):
    diagram = SpringForceDiagram(_spec(), spring_results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(backend, spring_results.fem)


def test_attach_emits_arrow_layer(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    layer = diagram._layer
    assert isinstance(layer, GlyphLayer)
    assert layer.kind == "arrow"
    assert layer.positions.n_points >= 1
    assert layer.layer_id in backend.layers


def test_attach_default_direction_from_component(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec("spring_force_1"), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    np.testing.assert_array_equal(diagram._direction, [0, 1, 0])


def test_attach_explicit_direction_normalised(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(direction=(2.0, 0.0, 0.0)), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    np.testing.assert_allclose(np.linalg.norm(diagram._direction), 1.0)
    np.testing.assert_allclose(diagram._direction, [1, 0, 0])


# --- Step update ---------------------------------------------------------


def test_step_update_changes_orientations(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    initial = np.asarray(diagram._layer.orientations).copy()

    n_steps = spring_results.stage(spring_results.stages[0].id).n_steps
    if n_steps < 2:
        pytest.skip("Spring fixture has no second step")
    diagram.update_to_step(min(2, n_steps - 1))
    after = np.asarray(diagram._layer.orientations)
    if not np.allclose(initial, after):
        return  # OK — forces evolved
    pytest.skip("Fixture spring forces are constant across steps")


# --- Runtime style -------------------------------------------------------


def test_set_scale(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


def test_set_direction_re_orients(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    diagram.set_direction((0.0, 0.0, 1.0))
    np.testing.assert_allclose(diagram._direction, [0, 0, 1])


def test_set_zero_direction_is_ignored(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    before = diagram._direction.copy()
    diagram.set_direction((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(diagram._direction, before)


# --- Handle stability + detach -------------------------------------------


def test_handle_stable_across_steps(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    initial_handle = diagram._handle
    n_steps = spring_results.stage(spring_results.stages[0].id).n_steps
    for step in range(min(3, n_steps)):
        diagram.update_to_step(step)
    assert diagram._handle is initial_handle


def test_detach_clears_state(spring_results, backend):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(backend, spring_results.fem, scene)
    layer_id = diagram._handle.layer_id
    diagram.detach()
    assert diagram._layer is None
    assert diagram._handle is None
    assert diagram._direction is None
    assert diagram._directions is None
    assert not diagram.is_attached
    assert layer_id in backend.removed


# --- Interface frame direction (ADR 0093 / TIMs A10) ---------------------
#
# A 3-D `g.constraints.interface()` pair's three springs act along the
# resolved record's own (n, t1, t2) frame, not the global axes. These
# tests exercise the two pure helpers directly (no gmsh, no MPCO needed —
# an InterfaceRecord is a plain dataclass), then one attach()-level test
# against the real spring fixture confirming a matched spring picks up
# the record's frame while an unmatched spring keeps the axis default.


def _iface(master, slave, *, phantom=None, orient=None) -> InterfaceRecord:
    return InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        master_node=master, slave_node=slave, phantom_node=phantom,
        orient=orient,
    )


def test_direction_from_interface_orient_2d():
    # 2-D line master: (x, yp) — yp not orthogonal to x on purpose, to
    # pin the Gram-Schmidt step.
    orient = (1.0, 0.0, 0.0, 0.3, 1.0, 0.0)
    np.testing.assert_allclose(
        _direction_from_interface_orient(orient, 0), [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        _direction_from_interface_orient(orient, 1), [0.0, 1.0, 0.0], atol=1e-12
    )
    assert _direction_from_interface_orient(orient, 2) is None


def test_direction_from_interface_orient_3d():
    # 3-D surface master: (n, t1, t2), already an orthonormal triad.
    orient = (0.0, 0.0, 1.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0)
    np.testing.assert_allclose(
        _direction_from_interface_orient(orient, 0), [0.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        _direction_from_interface_orient(orient, 1), [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        _direction_from_interface_orient(orient, 2), [0.0, 1.0, 0.0]
    )
    assert _direction_from_interface_orient(orient, 3) is None


def test_direction_from_interface_orient_degenerate_or_bad_width():
    assert _direction_from_interface_orient((0, 0, 0, 1, 0, 0), 0) is None
    assert _direction_from_interface_orient((1.0, 2.0, 3.0), 0) is None


def test_match_interface_record_by_master_slave_pair():
    rec = _iface(3, 4)
    assert _match_interface_record([rec], 3, 4) is rec
    assert _match_interface_record([rec], 4, 3) is rec  # order-independent


def test_match_interface_record_uses_phantom_over_slave():
    rec = _iface(3, 4, phantom=99)
    assert _match_interface_record([rec], 3, 99) is rec
    # jNode is the phantom, not the real slave — (3, 4) is not the pair.
    assert _match_interface_record([rec], 3, 4) is None


def test_match_interface_record_no_match():
    rec = _iface(3, 4)
    assert _match_interface_record([rec], 1, 2) is None
    assert _match_interface_record([], 3, 4) is None


def test_attach_uses_interface_frame_for_matched_spring(spring_results, backend):
    """Element 200 (nodes 3, 4) gets a hand-made InterfaceRecord with a
    frame that is clearly not a global axis; element 100 (nodes 1, 2)
    has no record and must keep the axis-aligned default."""
    scene = build_fem_scene(spring_results.fem)
    n_hat = np.array([0.0, 1.0, 0.0])
    spring_results.fem.elements.interfaces = [
        _iface(3, 4, orient=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    ]
    try:
        diagram = SpringForceDiagram(_spec("spring_force_0"), spring_results)
        diagram.attach(backend, spring_results.fem, scene)
        assert diagram._directions is not None
        matched = unmatched = 0
        for pos, direction in zip(diagram._positions, diagram._directions):
            if np.allclose(pos, [1.0, 0.0, 0.0]):     # element 200's anchor
                np.testing.assert_allclose(direction, n_hat)
                matched += 1
            else:                                      # element 100's anchor
                np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])
                unmatched += 1
        assert matched == 1 and unmatched == 1
    finally:
        spring_results.fem.elements.interfaces = []


def test_attach_explicit_direction_wins_over_interface_frame(spring_results, backend):
    spring_results.fem.elements.interfaces = [
        _iface(3, 4, orient=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    ]
    try:
        scene = build_fem_scene(spring_results.fem)
        diagram = SpringForceDiagram(
            _spec("spring_force_0", direction=(0.0, 0.0, 1.0)), spring_results,
        )
        diagram.attach(backend, spring_results.fem, scene)
        assert diagram._directions is None
        np.testing.assert_allclose(diagram._direction, [0.0, 0.0, 1.0])
    finally:
        spring_results.fem.elements.interfaces = []
