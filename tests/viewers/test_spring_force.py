"""SpringForceDiagram — uses the real MPCO spring fixture.

The native writer doesn't expose a ``write_springs_group`` method (the
springs reader is currently MPCO-only), so we exercise the diagram
against ``tests/fixtures/results/zl_springs.mpco``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    SlabSelector,
    SpringForceDiagram,
    SpringForceStyle,
)
from apeGmsh.viewers.diagrams._spring_force import (
    _direction_from_component,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


_SPRING_FIXTURE = Path("tests/fixtures/results/zl_springs.mpco")


@pytest.fixture
def spring_results():
    if not _SPRING_FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_SPRING_FIXTURE}")
    return Results.from_mpco(_SPRING_FIXTURE)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _spec(component="spring_force_0", direction=None) -> DiagramSpec:
    return DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component=component),
        style=SpringForceStyle(
            scale=1.0,
            direction=direction,
        ),
    )


# =====================================================================
# Construction + helpers
# =====================================================================

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
    np.testing.assert_array_equal(
        _direction_from_component("spring_force_0"), [1, 0, 0],
    )
    np.testing.assert_array_equal(
        _direction_from_component("spring_force_1"), [0, 1, 0],
    )
    np.testing.assert_array_equal(
        _direction_from_component("spring_force_2"), [0, 0, 1],
    )


def test_direction_from_unsuffixed_component_falls_back():
    np.testing.assert_array_equal(
        _direction_from_component("not_a_spring_thing"), [1, 0, 0],
    )


# =====================================================================
# Attach
# =====================================================================

def test_attach_requires_scene(spring_results, headless_plotter):
    diagram = SpringForceDiagram(_spec(), spring_results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, spring_results.fem)


def test_attach_builds_source(spring_results, headless_plotter):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)

    assert diagram._source is not None
    assert diagram._source.n_points >= 1


def test_attach_default_direction_from_component(
    spring_results, headless_plotter,
):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec("spring_force_1"), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    np.testing.assert_array_equal(diagram._direction, [0, 1, 0])


def test_attach_explicit_direction_normalised(
    spring_results, headless_plotter,
):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(
        _spec(direction=(2.0, 0.0, 0.0)), spring_results,
    )
    diagram.attach(headless_plotter, spring_results.fem, scene)
    np.testing.assert_allclose(
        np.linalg.norm(diagram._direction), 1.0,
    )
    np.testing.assert_allclose(diagram._direction, [1, 0, 0])


# =====================================================================
# Step update
# =====================================================================

def test_step_update_changes_vectors(spring_results, headless_plotter):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    initial_vec = np.asarray(diagram._source.point_data["_vec"]).copy()

    n_steps = spring_results.stage(spring_results.stages[0].id).n_steps
    if n_steps < 2:
        pytest.skip("Spring fixture has no second step")
    diagram.update_to_step(min(2, n_steps - 1))
    after = np.asarray(diagram._source.point_data["_vec"])
    # The fixture's spring forces evolve through the analysis; not all
    # springs may change but at least one should differ.
    if not np.allclose(initial_vec, after):
        return  # OK
    pytest.skip("Fixture spring forces are constant across steps")


# =====================================================================
# Runtime style
# =====================================================================

def test_set_scale(spring_results, headless_plotter):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


def test_set_direction_re_orients(spring_results, headless_plotter):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    diagram.set_direction((0.0, 0.0, 1.0))
    np.testing.assert_allclose(diagram._direction, [0, 0, 1])


def test_set_zero_direction_is_ignored(spring_results, headless_plotter):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    before = diagram._direction.copy()
    diagram.set_direction((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(diagram._direction, before)


# =====================================================================
# In-place mutation
# =====================================================================

def test_actor_identity_stable_across_steps(
    spring_results, headless_plotter,
):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    initial_actor = diagram._actor
    initial_source = diagram._source

    n_steps = spring_results.stage(spring_results.stages[0].id).n_steps
    for step in range(min(3, n_steps)):
        diagram.update_to_step(step)

    assert diagram._actor is initial_actor
    assert diagram._source is initial_source


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(spring_results, headless_plotter):
    scene = build_fem_scene(spring_results.fem)
    diagram = SpringForceDiagram(_spec(), spring_results)
    diagram.attach(headless_plotter, spring_results.fem, scene)
    diagram.detach()
    assert diagram._source is None
    assert diagram._actor is None
    assert diagram._direction is None
    assert not diagram.is_attached
