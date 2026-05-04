"""ReactionsDiagram — attach + step + family separation.

Builds a small mesh, writes synthetic ``reaction_force_*`` and
``reaction_moment_*`` nodal slabs across multiple steps, and verifies
that the diagram:

* renders both force and moment families when both are recorded,
* drops nodes whose reaction is identically zero across the time-history,
* updates each family's ``_vec`` / ``_mag`` per step,
* keeps actor identity stable across steps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    ReactionsDiagram,
    ReactionsStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


@pytest.fixture
def reactions_results(g, tmp_path: Path):
    """Cube mesh with reactions on the first two nodes only."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    n_steps = 4
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_nodes = node_ids.size

    # Reactions only on the first two nodes; the rest stay 0.
    # Force on node 0, moment on node 1, scaled by step + 1.
    t_scale = (np.arange(n_steps, dtype=np.float64) + 1.0).reshape(-1, 1)

    def _zero():
        return np.zeros((n_steps, n_nodes), dtype=np.float64)

    fx = _zero(); fx[:, 0] = (10.0 * t_scale[:, 0])
    fy = _zero(); fy[:, 0] = (-5.0 * t_scale[:, 0])
    fz = _zero()
    mx = _zero()
    my = _zero()
    mz = _zero(); mz[:, 1] = (3.0 * t_scale[:, 0])

    components = {
        # Need any nodal scalar so the writer is happy; reactions
        # alone is enough but mirror VectorGlyph's setup.
        "reaction_force_x": fx,
        "reaction_force_y": fy,
        "reaction_force_z": fz,
        "reaction_moment_x": mx,
        "reaction_moment_y": my,
        "reaction_moment_z": mz,
    }

    path = tmp_path / "reactions.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _spec(style: ReactionsStyle | None = None) -> DiagramSpec:
    return DiagramSpec(
        kind="reactions",
        # Selector component is a dummy placeholder for the reactions
        # kind — the diagram reads from a fixed component list.
        selector=SlabSelector(component="reactions"),
        style=style or ReactionsStyle(),
    )


def test_construction_requires_reactions_style(reactions_results):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="reactions",
        selector=SlabSelector(component="reactions"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="ReactionsStyle"):
        ReactionsDiagram(bad, reactions_results)


def test_attach_requires_scene(reactions_results, headless_plotter):
    diagram = ReactionsDiagram(_spec(), reactions_results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, reactions_results.fem)


def test_attach_builds_force_and_moment_layers(reactions_results, headless_plotter):
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(_spec(), reactions_results)
    diagram.attach(headless_plotter, reactions_results.fem, scene)

    # Force family: only node 0 has a non-zero reaction → 1 row.
    assert diagram._force.source is not None
    assert diagram._force.source.n_points == 1
    # Moment family: only node 1 → 1 row.
    assert diagram._moment.source is not None
    assert diagram._moment.source.n_points == 1
    # Both actors registered.
    assert len(diagram._actors) == 2


def test_force_only_disables_moment_layer(reactions_results, headless_plotter):
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(
        _spec(ReactionsStyle(show_moments=False)), reactions_results,
    )
    diagram.attach(headless_plotter, reactions_results.fem, scene)
    assert diagram._force.source is not None
    assert diagram._moment.source is None
    assert len(diagram._actors) == 1


def test_moment_only_disables_force_layer(reactions_results, headless_plotter):
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(
        _spec(ReactionsStyle(show_forces=False)), reactions_results,
    )
    diagram.attach(headless_plotter, reactions_results.fem, scene)
    assert diagram._force.source is None
    assert diagram._moment.source is not None
    assert len(diagram._actors) == 1


def test_step_update_rescales_vectors(reactions_results, headless_plotter):
    """Reactions scale linearly with (step+1) in the fixture."""
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(_spec(), reactions_results)
    diagram.attach(headless_plotter, reactions_results.fem, scene)

    vecs0 = np.asarray(diagram._force.source.point_data["_vec"]).copy()
    diagram.update_to_step(3)
    vecs3 = np.asarray(diagram._force.source.point_data["_vec"])
    # step 0: (10, -5, 0). step 3: (40, -20, 0). Ratio = 4.
    np.testing.assert_allclose(vecs3, 4.0 * vecs0)


def test_actor_identity_stable_across_steps(reactions_results, headless_plotter):
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(_spec(), reactions_results)
    diagram.attach(headless_plotter, reactions_results.fem, scene)
    initial_force_actor = diagram._force.actor
    initial_moment_actor = diagram._moment.actor
    initial_force_source = diagram._force.source
    initial_moment_source = diagram._moment.source

    for step in range(4):
        diagram.update_to_step(step)

    assert diagram._force.actor is initial_force_actor
    assert diagram._moment.actor is initial_moment_actor
    assert diagram._force.source is initial_force_source
    assert diagram._moment.source is initial_moment_source


def test_set_force_scale_records_runtime(reactions_results, headless_plotter):
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(_spec(), reactions_results)
    diagram.attach(headless_plotter, reactions_results.fem, scene)
    diagram.set_force_scale(2.5)
    assert diagram.current_force_scale() == 2.5


def test_auto_scale_uses_global_max(reactions_results, headless_plotter):
    """Largest force magnitude over all steps should drive auto-fit."""
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(
        _spec(ReactionsStyle(force_scale=None, auto_scale_fraction=0.10)),
        reactions_results,
    )
    diagram.attach(headless_plotter, reactions_results.fem, scene)
    # Step 3 has force = (40, -20, 0). |force|_max = sqrt(40^2 + 20^2)
    expected_global_max = np.sqrt(40.0 ** 2 + 20.0 ** 2)
    expected = 0.10 * scene.model_diagonal / expected_global_max
    assert abs(diagram.current_force_scale() - expected) < 1e-9


def test_detach_clears_state(reactions_results, headless_plotter):
    scene = build_fem_scene(reactions_results.fem)
    diagram = ReactionsDiagram(_spec(), reactions_results)
    diagram.attach(headless_plotter, reactions_results.fem, scene)
    diagram.detach()
    assert diagram._force.source is None
    assert diagram._force.actor is None
    assert diagram._moment.source is None
    assert diagram._moment.actor is None
    assert not diagram.is_attached
