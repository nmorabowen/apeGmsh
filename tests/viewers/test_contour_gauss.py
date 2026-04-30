"""ContourDiagram — gauss / element-constant path.

Exercises the new code added in PR 3:

* ``ContourStyle.topology = "gauss"`` paints per-cell scalars from a
  ``GaussSlab`` whose elements each carry exactly one Gauss point.
* ``topology = "auto"`` prefers nodal data when both composites have
  the requested component, and falls through to gauss when only the
  gauss composite has it.
* Multi-GP slabs raise a clear ``NoDataError`` rather than silently
  averaging or rendering nonsense.
* ``update_to_step`` mutates the cell-data array in place — same in-
  place contract as the nodal path.
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
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._base import NoDataError
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


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
    return Results.from_native(path)


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
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


# =====================================================================
# Topology resolution
# =====================================================================


def _spec(component: str, topology: str = "auto") -> DiagramSpec:
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(topology=topology),
    )


def test_auto_prefers_nodes_when_both_have_component(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    # displacement_z is nodal-only; stress_xx is gauss-only.
    # Auto on a nodes-only component → nodes path.
    diagram = ContourDiagram(_spec("displacement_z"), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert diagram._effective_topology == "nodes"
    assert diagram._scalar_array is not None
    assert diagram._cell_scalar_array is None


def test_auto_falls_through_to_gauss_for_element_only_component(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(_spec("stress_xx"), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert diagram._effective_topology == "gauss"
    assert diagram._cell_scalar_array is not None
    assert diagram._scalar_array is None


def test_explicit_gauss_topology_uses_cell_data(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
    scene = build_fem_scene(r.fem)
    diagram.attach(headless_plotter, r.fem, scene)
    assert diagram._effective_topology == "gauss"
    assert diagram._submesh.n_cells == scene.grid.n_cells
    # cell_data array exists and is sized like the submesh
    assert diagram._cell_scalar_array.shape[0] == diagram._submesh.n_cells


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
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
    diagram.attach(headless_plotter, r.fem, scene)

    # vtkOriginalCellIds maps submesh row -> substrate row -> element ID
    orig = np.asarray(
        diagram._submesh.cell_data["vtkOriginalCellIds"], dtype=np.int64,
    )
    fem_eids_in_submesh = scene.cell_to_element_id[orig]
    # Step 0 expected: eid * 10 + 0 == eid * 10
    expected = fem_eids_in_submesh.astype(np.float64) * 10.0
    np.testing.assert_array_equal(
        np.asarray(diagram._cell_scalar_array), expected,
    )


def test_update_to_step_mutates_cell_data_in_place(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    scene = build_fem_scene(r.fem)
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
    diagram.attach(headless_plotter, r.fem, scene)

    array_id_before = id(diagram._cell_scalar_array)
    mapper_id_before = id(diagram._actor.GetMapper())

    orig = np.asarray(
        diagram._submesh.cell_data["vtkOriginalCellIds"], dtype=np.int64,
    )
    fem_eids_in_submesh = scene.cell_to_element_id[orig]

    for step in (1, 2, 3, 0):
        diagram.update_to_step(step)
        expected = fem_eids_in_submesh.astype(np.float64) * 10.0 + step
        np.testing.assert_array_equal(
            np.asarray(diagram._cell_scalar_array), expected,
        )

    assert id(diagram._cell_scalar_array) == array_id_before
    assert id(diagram._actor.GetMapper()) == mapper_id_before


def test_initial_clim_brackets_step_0_values(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    scene = build_fem_scene(r.fem)
    elem_ids = _all_element_ids(r.fem)
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
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
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
    diagram.attach(headless_plotter, r.fem, scene)
    diagram.update_to_step(2)    # values = eid*10 + 2

    fitted = diagram.autofit_clim_at_current_step()
    assert fitted is not None
    lo, hi = fitted
    assert lo == float(elem_ids.min() * 10 + 2)
    assert hi == float(elem_ids.max() * 10 + 2)


# =====================================================================
# Higher-order integration is rejected with a clear hint
# =====================================================================


def test_two_gp_per_element_raises_no_data_error(
    results_with_two_gp_gauss, headless_plotter,
):
    r = results_with_two_gp_gauss
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
    with pytest.raises(NoDataError, match="2 Gauss points per element"):
        diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))


# =====================================================================
# Detach clears both paths' state
# =====================================================================


def test_detach_clears_gauss_state(
    results_with_nodes_and_element_constant_gauss, headless_plotter,
):
    r = results_with_nodes_and_element_constant_gauss
    diagram = ContourDiagram(_spec("stress_xx", topology="gauss"), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    diagram.detach()
    assert diagram._submesh is None
    assert diagram._cell_scalar_array is None
    assert diagram._submesh_cell_pos_of_eid is None
    assert diagram._fem_eids_to_read is None
    assert diagram._effective_topology is None
