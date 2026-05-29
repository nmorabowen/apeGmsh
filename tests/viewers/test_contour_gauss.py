"""ContourDiagram — gauss / element-constant path.

Exercises the gauss source paths:

* ``topology="gauss"`` + ``averaging="discrete"`` paints per-cell
  scalars from a ``GaussSlab`` whose elements each carry exactly one
  Gauss point.
* ``topology="gauss"`` + ``averaging="averaged"`` extrapolates GP
  values to corners and averages across elements at shared nodes,
  painting smoothed point data.
* Multi-GP slabs route to the GP→nodal extrapolation path.
* ``update_to_step`` mutates the relevant scalar array in place —
  same in-place contract across all paths.
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
    SlabSelector,
)
from apeGmsh.viewers.diagrams._base import NoDataError
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


# =====================================================================
# Fixtures
# =====================================================================


def _all_element_ids(fem) -> np.ndarray:
    chunks = []
    for group in fem.elements:
        chunks.append(np.asarray(group.ids, dtype=np.int64))
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks)


@pytest.fixture
def results_with_nodes_and_element_constant_gauss(g, tmp_path: Path):
    """Native HDF5 with predictable per-node *and* per-element data.

    For step ``t`` and node ID ``nid``: ``displacement_z = nid + t * 1000``.
    For step ``t`` and element ID ``eid``: ``stress_xx = eid * 10 + t``.

    One Gauss point per element so the gauss path can paint cell-
    constant scalars without further averaging.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    n_steps = 4
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_nodes = node_ids.size
    elem_ids = _all_element_ids(fem)
    n_elem = elem_ids.size

    # Predictable per-node per-step
    disp_z = np.zeros((n_steps, n_nodes), dtype=np.float64)
    for t in range(n_steps):
        disp_z[t] = node_ids + t * 1000.0

    # Predictable per-element per-step (1 GP each → shape (T, E, 1))
    sxx = np.zeros((n_steps, n_elem, 1), dtype=np.float64)
    for t in range(n_steps):
        sxx[t, :, 0] = elem_ids * 10.0 + t

    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)    # one GP

    path = tmp_path / "node_and_gauss.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": disp_z},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def results_with_two_gp_gauss(g, tmp_path: Path):
    """Synthetic data with TWO Gauss points per element.

    Used to assert that the element-constant contour path rejects
    higher-order integration with a clear NoDataError instead of
    silently averaging or scrambling cell values.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    elem_ids = _all_element_ids(fem)
    n_elem = elem_ids.size
    n_steps = 2
    sxx = np.zeros((n_steps, n_elem, 2), dtype=np.float64)    # 2 GPs/elem
    nat = np.array([[-0.5, 0, 0], [0.5, 0, 0]], dtype=np.float64)

    path = tmp_path / "two_gp.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


# headless_plotter is a shared fixture in tests/viewers/conftest.py
# (yields a PyVistaQtBackend, ADR 0042 R-B.final).


# =====================================================================
# Topology resolution
# =====================================================================


def _spec(
    component: str,
    topology: str = "nodes",
    averaging: str = "averaged",
) -> DiagramSpec:
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(topology=topology, averaging=averaging),
    )


def test_nodes_topology_uses_nodal_scalar_path(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(_spec("displacement_z", topology="nodes"), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert diagram._effective_topology == "nodes"
    assert diagram._scalar_location == "point"


def test_gauss_discrete_with_one_gp_uses_cell_data(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert diagram._effective_topology == "gauss_cell"
    assert diagram._scalar_location == "cell"


def test_gauss_averaged_with_one_gp_uses_smoothed_point_data(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="averaged"), r,
    )
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert diagram._effective_topology == "gauss_cell_averaged"
    assert diagram._scalar_location == "point"


def test_explicit_gauss_discrete_uses_cell_data(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    scene = build_fem_scene(r.fem)
    diagram.attach(headless_plotter, r.fem, scene)
    assert diagram._effective_topology == "gauss_cell"
    assert diagram._layer.cells.n_cells == scene.grid.n_cells
    assert diagram._scalar_values.shape[0] == diagram._layer.cells.n_cells


def test_invalid_topology_value_raises(
    results_with_nodes_and_element_constant_gauss,
):
    r = results_with_nodes_and_element_constant_gauss
    with pytest.raises(ValueError, match="topology"):
        d = ContourDiagram(_spec("stress_xx", topology="bogus"), r)
        d._resolve_topology()


# =====================================================================
# Per-cell scalars carry the right values
# =====================================================================


def test_attach_paints_expected_per_cell_values(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    scene = build_fem_scene(r.fem)
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, scene)

    # The diagram caches the per-cell element ids in CellBlocks (grouped)
    # order — the same order as the emitted cell ScalarField. For this
    # single-cell-type cube the grouping is the identity.
    fem_eids_in_submesh = diagram._fem_eids_to_read
    expected = fem_eids_in_submesh.astype(np.float64) * 10.0
    np.testing.assert_array_equal(
        np.asarray(diagram._scalar_values), expected,
    )
    field = diagram._layer.field_named("stress_xx")
    assert field is not None and field.location == "cell"
    np.testing.assert_array_equal(np.asarray(field.values), expected)


def test_update_to_step_mutates_cell_data_in_place(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    scene = build_fem_scene(r.fem)
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, scene)

    buffer_before = diagram._scalar_values
    initial_actor = diagram._handle.actor
    mapper_id_before = id(diagram._handle.actor.GetMapper())

    fem_eids_in_submesh = diagram._fem_eids_to_read

    for step in (1, 2, 3, 0):
        diagram.update_to_step(step)
        expected = fem_eids_in_submesh.astype(np.float64) * 10.0 + step
        np.testing.assert_array_equal(
            np.asarray(diagram._scalar_values), expected,
        )

    # Persistent scalar buffer mutated in place; backend reuses the
    # actor + mapper across steps (mesh fast path).
    assert diagram._scalar_values is buffer_before
    assert diagram._handle.actor is initial_actor
    assert id(diagram._handle.actor.GetMapper()) == mapper_id_before


def test_initial_clim_brackets_step_0_values(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    scene = build_fem_scene(r.fem)
    elem_ids = _all_element_ids(r.fem)
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, scene)
    lo, hi = diagram.current_clim()
    assert lo <= elem_ids.min() * 10.0
    assert hi >= elem_ids.max() * 10.0


def test_autofit_at_current_step_works_for_gauss(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    scene = build_fem_scene(r.fem)
    elem_ids = _all_element_ids(r.fem)
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, scene)
    diagram.update_to_step(2)

    fitted = diagram.autofit_clim_at_current_step()
    assert fitted is not None
    lo, hi = fitted
    assert lo == float(elem_ids.min() * 10 + 2)
    assert hi == float(elem_ids.max() * 10 + 2)


# =====================================================================
# Higher-order integration is rejected with a clear hint
# =====================================================================


def test_two_gp_averaged_routes_to_gauss_node(
    results_with_two_gp_gauss, headless_plotter,
):
    """Multi-GP slab + averaging=averaged → GP→nodal extrapolation
    with cross-element averaging, painted as smoothed point data."""
    r = results_with_two_gp_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="averaged"), r,
    )
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert diagram._effective_topology == "gauss_node"
    assert diagram._scalar_location == "point"


def test_two_gp_discrete_routes_to_shattered_submesh(
    results_with_two_gp_gauss, headless_plotter,
):
    """Multi-GP slab + averaging=discrete → per-element extrapolation
    on a shattered submesh; n_points exceeds the substrate's because
    each cell owns its own copies of its corner points."""
    r = results_with_two_gp_gauss
    scene = build_fem_scene(r.fem)
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, scene)
    assert diagram._effective_topology == "gauss_node_discrete"
    assert diagram._scalar_location == "point"
    # Shattered: each cell has its own copies of its corners, so the
    # emitted layer carries more points than the shared substrate and
    # exactly the cumulative per-cell corner count.
    assert diagram._layer.points.n_points > scene.grid.n_points
    assert (
        diagram._layer.points.n_points
        == int(diagram._discrete_cell_point_offsets[-1])
    )


def test_gauss_node_path_in_place_mutation_across_steps(
    results_with_two_gp_gauss, headless_plotter,
):
    """The extrapolated path must obey the same in-place contract as
    the nodes path: same point_data array, same mapper id across step
    changes."""
    r = results_with_two_gp_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="averaged"), r,
    )
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    buffer_before = diagram._scalar_values
    initial_actor = diagram._handle.actor
    mapper_id_before = id(diagram._handle.actor.GetMapper())

    diagram.update_to_step(1)
    diagram.update_to_step(0)

    assert diagram._scalar_values is buffer_before
    assert diagram._handle.actor is initial_actor
    assert id(diagram._handle.actor.GetMapper()) == mapper_id_before


def test_gauss_discrete_path_in_place_mutation_across_steps(
    results_with_two_gp_gauss, headless_plotter,
):
    """Discrete shattered submesh keeps the same point-data array and
    mapper across step changes."""
    r = results_with_two_gp_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    buffer_before = diagram._scalar_values
    initial_actor = diagram._handle.actor
    mapper_id_before = id(diagram._handle.actor.GetMapper())

    diagram.update_to_step(1)
    diagram.update_to_step(0)

    assert diagram._scalar_values is buffer_before
    assert diagram._handle.actor is initial_actor
    assert id(diagram._handle.actor.GetMapper()) == mapper_id_before


# =====================================================================
# Detach clears both paths' state
# =====================================================================


def test_detach_clears_gauss_state(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(
        _spec("stress_xx", topology="gauss", averaging="discrete"), r,
    )
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    diagram.detach()
    assert diagram._layer is None
    assert diagram._handle is None
    assert diagram._scalar_values is None
    assert diagram._submesh_cell_pos_of_eid is None
    assert diagram._fem_eids_to_read is None
    assert diagram._discrete_cell_point_offsets is None
    assert diagram._effective_topology is None
