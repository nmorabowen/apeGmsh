"""``build_fem_scene`` — coverage of the (dim, npe) fallback for MPCO.

MPCO files don't carry Gmsh element-type codes; ``FEMData.from_mpco`` synthesizes
``code = -class_tag`` (negative). The substrate-mesh builder in
``viewers/scene/fem_scene.py`` keys its primary ``GMSH_LINEAR`` table on positive
Gmsh codes, so MPCO groups would otherwise be skipped wholesale, leaving the
results viewer with a node cloud and no element wireframe.

This file pins:

* the real-fixture path (``elasticFrame.mpco``, ``zl_springs.mpco``) producing
  cells with the expected VTK type;
* the in-memory fallback path on a synthetic FEMData (negative code, dim+npe
  set) building the right cell shape;
* unknown ``(dim, npe)`` combinations still falling cleanly through to
  ``skipped_types``;
* positive Gmsh codes continuing to take the fast path unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results import Results
from apeGmsh.viewers.scene.fem_scene import (
    GMSH_LINEAR,
    GMSH_LINEAR_FALLBACK,
    build_fem_scene,
)


_FIXTURES = Path("tests/fixtures/results")


# =====================================================================
# Helpers
# =====================================================================

def _make_fem(
    *,
    code: int,
    dim: int,
    npe: int,
    coords: np.ndarray,
    connectivity: np.ndarray,
    gmsh_name: str = "synthetic",
) -> FEMData:
    """Build a minimal single-group FEMData for fallback unit tests."""
    n_nodes = coords.shape[0]
    node_ids = np.arange(1, n_nodes + 1, dtype=np.int64)
    info = make_type_info(
        code=code, gmsh_name=gmsh_name, dim=dim, order=1,
        npe=npe, count=connectivity.shape[0],
    )
    elem_ids = np.arange(1, connectivity.shape[0] + 1, dtype=np.int64)
    group = ElementGroup(element_type=info, ids=elem_ids,
                         connectivity=connectivity)

    pg: dict = {}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={info.code: group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(
            n_nodes=n_nodes, n_elems=connectivity.shape[0],
            bandwidth=0, types=[info],
        ),
    )


# =====================================================================
# MPCO fixtures — the bug-this-PR-fixes scenario
# =====================================================================

@pytest.fixture
def elastic_frame_results():
    path = _FIXTURES / "elasticFrame.mpco"
    if not path.exists():
        pytest.skip(f"Missing fixture: {path}")
    return Results.from_mpco(path)


@pytest.fixture
def zl_springs_results():
    path = _FIXTURES / "zl_springs.mpco"
    if not path.exists():
        pytest.skip(f"Missing fixture: {path}")
    return Results.from_mpco(path)


def test_mpco_elastic_frame_substrate_has_cells(elastic_frame_results):
    """ElasticBeam3d (MPCO synthetic code -5) renders as VTK_LINE cells."""
    scene = build_fem_scene(elastic_frame_results.fem)
    assert scene.grid.n_cells > 0
    assert scene.grid.n_cells == sum(
        len(g) for g in elastic_frame_results.fem.elements
    )
    assert set(int(t) for t in scene.grid.celltypes) == {3}     # VTK_LINE
    assert scene.skipped_types == []


def test_mpco_zerolength_substrate_has_cells(zl_springs_results):
    """ZeroLength (MPCO synthetic code -19) renders as VTK_LINE cells."""
    scene = build_fem_scene(zl_springs_results.fem)
    assert scene.grid.n_cells > 0
    assert set(int(t) for t in scene.grid.celltypes) == {3}     # VTK_LINE
    assert scene.skipped_types == []


def test_mpco_cell_to_element_id_round_trip(elastic_frame_results):
    """Per-cell element_id maps round-trip through both directions."""
    scene = build_fem_scene(elastic_frame_results.fem)
    for cell_idx, eid in enumerate(scene.cell_to_element_id):
        assert scene.element_id_to_cell[int(eid)] == cell_idx


# =====================================================================
# Synthetic fallback — one test per shape
# =====================================================================

def test_fallback_negative_code_hex_dim3_npe8():
    """A synthetic group with code=-100, dim=3, npe=8 builds as VTK_HEXAHEDRON."""
    coords = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    fem = _make_fem(code=-100, dim=3, npe=8, coords=coords, connectivity=conn)

    scene = build_fem_scene(fem)
    assert scene.grid.n_cells == 1
    assert int(scene.grid.celltypes[0]) == 12       # VTK_HEXAHEDRON
    assert scene.skipped_types == []


def test_fallback_negative_code_tet_dim3_npe4():
    """Synthetic tet via fallback — dim=3, npe=4 → VTK_TETRA."""
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64,
    )
    conn = np.array([[1, 2, 3, 4]], dtype=np.int64)
    fem = _make_fem(code=-200, dim=3, npe=4, coords=coords, connectivity=conn)
    scene = build_fem_scene(fem)
    assert scene.grid.n_cells == 1
    assert int(scene.grid.celltypes[0]) == 10       # VTK_TETRA


def test_fallback_negative_code_quad_dim2_npe4():
    """A 4-node quad with dim=2 routes to VTK_QUAD, not VTK_TETRA.

    This is the key dim disambiguation case: ``(2, 4)`` is quad but ``(3, 4)``
    is tet — they share npe but differ in dim.
    """
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64,
    )
    conn = np.array([[1, 2, 3, 4]], dtype=np.int64)
    fem = _make_fem(code=-300, dim=2, npe=4, coords=coords, connectivity=conn)
    scene = build_fem_scene(fem)
    assert scene.grid.n_cells == 1
    assert int(scene.grid.celltypes[0]) == 9        # VTK_QUAD


def test_fallback_table_covers_all_directives_shapes():
    """Every ``(dim, npe)`` pair in the directives §FixApproach list resolves.

    Smoke for table completeness: vertex / line / tri / quad / tet / pyramid
    / wedge / hex.
    """
    expected: set[tuple[int, int]] = {
        (0, 1), (1, 2), (2, 3), (2, 4),
        (3, 4), (3, 5), (3, 6), (3, 8),
    }
    assert expected.issubset(GMSH_LINEAR_FALLBACK.keys())


# =====================================================================
# Negative cases — unknown shapes still skip
# =====================================================================

def test_unknown_code_unknown_dim_npe_is_skipped():
    """A negative code with no fallback match lands in ``skipped_types``."""
    coords = np.zeros((7, 3), dtype=np.float64)
    # dim=3, npe=7 isn't in the fallback (no real shape has 7 corner nodes).
    conn = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    fem = _make_fem(code=-999, dim=3, npe=7, coords=coords, connectivity=conn)
    scene = build_fem_scene(fem)
    assert scene.grid.n_cells == 0
    assert scene.skipped_types == [-999]


# =====================================================================
# Regression — positive Gmsh codes still take the primary path
# =====================================================================

def test_positive_gmsh_code_takes_fast_path():
    """Code=4 (Gmsh tet4) hits ``GMSH_LINEAR`` directly — fallback never queried."""
    assert 4 in GMSH_LINEAR
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64,
    )
    conn = np.array([[1, 2, 3, 4]], dtype=np.int64)
    # dim/npe deliberately set to bogus values; if the fast path works,
    # they're never consulted.
    fem = _make_fem(
        code=4, dim=99, npe=99,
        coords=coords, connectivity=conn, gmsh_name="Tetrahedron 4",
    )
    scene = build_fem_scene(fem)
    assert scene.grid.n_cells == 1
    assert int(scene.grid.celltypes[0]) == 10       # VTK_TETRA


# =====================================================================
# Sanity — full-grid integrity after MPCO load
# =====================================================================

def test_mpco_grid_is_valid_pyvista_unstructured(elastic_frame_results):
    scene = build_fem_scene(elastic_frame_results.fem)
    assert isinstance(scene.grid, pv.UnstructuredGrid)
    assert scene.grid.n_points == elastic_frame_results.fem.nodes.ids.size
    assert "element_id" in scene.grid.cell_data
    assert scene.grid.cell_data["element_id"].size == scene.grid.n_cells
    assert "node_id" in scene.grid.point_data
