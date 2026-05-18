"""Phase 8.7 commit 3 — ``ViewerData`` adapter parity tests.

The contract for :class:`ViewerData` is that ``from_fem(fem)`` and
``from_h5(fem.to_h5(path))`` produce indistinguishable accessor
results across the audited viewer surface (nodes, elements, named
indexes, record sets).

Tests build a representative FEMData in memory, build a
``ViewerData`` directly from it, then round-trip it through ``model.h5``
and build a second ``ViewerData`` from the file.  Equivalence checks
walk every audited accessor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh._kernel.records._loads import (
    ElementLoadRecord,
    NodalLoadRecord,
    SPRecord,
)
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh.viewers.data import (
    ElementLoadRow,
    InterpolationRow,
    MassRow,
    NodalLoadRow,
    NodeGroupRow,
    NodePairRow,
    SPRow,
    ViewerData,
    ViewerDataDecodeError,
)


# =====================================================================
# Fixture FEMData — covers every audited accessor
# =====================================================================


def _make_fem_with_selections() -> FEMData:
    """FEMData with one PG, one label, one selection set, plus loads /
    constraints / masses on both sides."""
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)

    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([20], dtype=np.int64),
        connectivity=np.array([[2, 3, 5]], dtype=np.int64),
    )

    pg = {(2, 100): {
        "name": "Slab",
        "node_ids": np.array([2, 3, 5], dtype=np.int64),
        "node_coords": np.array([
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0],
        ]),
        "element_ids": np.array([20], dtype=np.int64),
    }}
    labels = {(1, 200): {
        "name": "edge",
        "node_ids": np.array([1, 2], dtype=np.int64),
        "node_coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "element_ids": np.array([10], dtype=np.int64),
    }}

    np_record = NodePairRecord(
        kind=ConstraintKind.EQUAL_DOF, name="weld",
        master_node=1, slave_node=2, dofs=[1, 2, 3], offset=None,
    )
    ng_record = NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM, name="floor1",
        master_node=1, slave_nodes=[2, 3, 4],
        dofs=[1, 2, 6],
        offsets=np.array([
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    interp_record = InterpolationRecord(
        kind=ConstraintKind.TIE, name="bond",
        slave_node=5, master_nodes=[1, 2, 3, 4],
        weights=np.array([0.25, 0.25, 0.25, 0.25]),
        dofs=[1, 2, 3],
    )

    nl_record = NodalLoadRecord(
        node_id=2, force_xyz=(1000.0, 0.0, 0.0), moment_xyz=None,
        pattern="gravity",
    )
    el_record = ElementLoadRecord(
        element_id=20, load_type="surfacePressure",
        params={"pressure": -250.0, "direction": "normal"},
        pattern="gravity",
    )
    sp_record = SPRecord(
        node_id=1, dof=3, value=0.0, is_homogeneous=True,
    )
    m1 = MassRecord(node_id=4, mass=(100.0, 100.0, 100.0, 0.0, 0.0, 0.0))
    m2 = MassRecord(node_id=5, mass=(50.0, 50.0, 50.0, 1.0, 1.0, 1.0))

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet(labels),
        constraints=[np_record, ng_record],
        loads=[nl_record],
        sp=[sp_record],
        masses=[m1, m2],
    )
    elements = ElementComposite(
        groups={1: line_group, 2: tri_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet(labels),
        constraints=[interp_record],
        loads=[el_record],
    )
    info = MeshInfo(n_nodes=5, n_elems=2, bandwidth=2,
                    types=[line_info, tri_info])
    fem = FEMData(nodes=nodes, elements=elements, info=info)

    # Attach a mesh_selection store with one node-only and one
    # element-bearing entry — covers the 2.4.0 round-trip.
    fem.mesh_selection = MeshSelectionStore({
        (0, 1): {
            "name": "anchor",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            ], dtype=np.float64),
        },
        (2, 1): {
            "name": "slab_picks",
            "node_ids": np.array([2, 3, 5], dtype=np.int64),
            "node_coords": np.array([
                [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0],
            ], dtype=np.float64),
            "element_ids": np.array([20], dtype=np.int64),
            "connectivity": np.array([[2, 3, 5]], dtype=np.int64),
        },
    })
    return fem


@pytest.fixture
def fem() -> FEMData:
    return _make_fem_with_selections()


@pytest.fixture
def viewer_from_fem(fem: FEMData) -> ViewerData:
    return ViewerData.from_fem(fem)


@pytest.fixture
def viewer_from_h5(fem: FEMData, tmp_path: Path) -> ViewerData:
    out = tmp_path / "model.h5"
    fem.to_h5(str(out))
    return ViewerData.from_h5(str(out))


# =====================================================================
# Source-of-truth metadata
# =====================================================================


def test_from_fem_source_kind(viewer_from_fem: ViewerData) -> None:
    assert viewer_from_fem.source_kind == "fem"


def test_from_h5_source_kind(viewer_from_h5: ViewerData) -> None:
    assert viewer_from_h5.source_kind == "h5"


def test_snapshot_id_round_trips(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    """snapshot_id is content-deterministic; from_fem and from_h5
    should agree even though the second went through HDF5."""
    assert viewer_from_fem.snapshot_id
    assert viewer_from_fem.snapshot_id == viewer_from_h5.snapshot_id


# =====================================================================
# Nodes
# =====================================================================


def test_node_ids_and_coords_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    np.testing.assert_array_equal(
        viewer_from_fem.nodes.ids, viewer_from_h5.nodes.ids,
    )
    np.testing.assert_allclose(
        viewer_from_fem.nodes.coords, viewer_from_h5.nodes.coords,
    )


def test_node_index(viewer_from_fem: ViewerData) -> None:
    assert viewer_from_fem.nodes.index(1) == 0
    assert viewer_from_fem.nodes.index(5) == 4
    with pytest.raises(KeyError):
        viewer_from_fem.nodes.index(99)


def test_node_named_indexes_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    np.testing.assert_array_equal(
        viewer_from_fem.nodes.physical.node_ids("Slab"),
        viewer_from_h5.nodes.physical.node_ids("Slab"),
    )
    np.testing.assert_array_equal(
        viewer_from_fem.nodes.labels.node_ids("edge"),
        viewer_from_h5.nodes.labels.node_ids("edge"),
    )
    np.testing.assert_array_equal(
        viewer_from_fem.nodes.selection.node_ids("anchor"),
        viewer_from_h5.nodes.selection.node_ids("anchor"),
    )
    np.testing.assert_array_equal(
        viewer_from_fem.nodes.selection.node_ids("slab_picks"),
        viewer_from_h5.nodes.selection.node_ids("slab_picks"),
    )


def test_selection_missing_returns_empty(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    """``selection.node_ids(missing)`` returns [] on both paths —
    the documented 2.4.0 gap."""
    for v in (viewer_from_fem, viewer_from_h5):
        assert v.nodes.selection.node_ids("does_not_exist").size == 0


def test_physical_missing_raises(viewer_from_fem: ViewerData) -> None:
    with pytest.raises(KeyError):
        viewer_from_fem.nodes.physical.node_ids("does_not_exist")


# =====================================================================
# Elements
# =====================================================================


def test_element_groups_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    fem_groups = list(viewer_from_fem.elements)
    h5_groups = list(viewer_from_h5.elements)
    assert len(fem_groups) == len(h5_groups) == 2

    fem_by_code = {g.element_type.code: g for g in fem_groups}
    h5_by_code = {g.element_type.code: g for g in h5_groups}
    assert set(fem_by_code.keys()) == set(h5_by_code.keys())

    for code, gf in fem_by_code.items():
        gh = h5_by_code[code]
        assert gf.element_type.dim == gh.element_type.dim
        assert gf.element_type.npe == gh.element_type.npe
        np.testing.assert_array_equal(gf.ids, gh.ids)
        np.testing.assert_array_equal(gf.connectivity, gh.connectivity)


def test_element_iteration_unpacking(viewer_from_fem: ViewerData) -> None:
    """ElementGroup.__iter__ yields ``(eid, conn_tuple)`` for solver loops."""
    for group in viewer_from_fem.elements:
        for eid, conn in group:
            assert isinstance(eid, int)
            assert all(isinstance(n, int) for n in conn)
            assert len(conn) == group.npe


def test_element_named_indexes_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    np.testing.assert_array_equal(
        viewer_from_fem.elements.physical.element_ids("Slab"),
        viewer_from_h5.elements.physical.element_ids("Slab"),
    )
    np.testing.assert_array_equal(
        viewer_from_fem.elements.labels.element_ids("edge"),
        viewer_from_h5.elements.labels.element_ids("edge"),
    )
    np.testing.assert_array_equal(
        viewer_from_fem.elements.selection.element_ids("slab_picks"),
        viewer_from_h5.elements.selection.element_ids("slab_picks"),
    )


# =====================================================================
# Nodal loads / masses / SP
# =====================================================================


def test_nodal_loads_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    fem_loads = list(viewer_from_fem.nodes.loads)
    h5_loads = list(viewer_from_h5.nodes.loads)
    assert len(fem_loads) == len(h5_loads) == 1
    assert all(isinstance(r, NodalLoadRow) for r in fem_loads + h5_loads)
    assert fem_loads[0].node_id == h5_loads[0].node_id == 2
    assert fem_loads[0].force_xyz == h5_loads[0].force_xyz == (1000.0, 0.0, 0.0)
    # moment NaN → None on both paths
    assert fem_loads[0].moment_xyz is None
    assert h5_loads[0].moment_xyz is None
    assert fem_loads[0].pattern == h5_loads[0].pattern == "gravity"


def test_nodal_loads_patterns_and_by_pattern(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    for v in (viewer_from_fem, viewer_from_h5):
        assert v.nodes.loads.patterns() == ["gravity"]
        assert len(v.nodes.loads.by_pattern("gravity")) == 1
        assert v.nodes.loads.by_pattern("missing") == []


def test_masses_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    fem_masses = sorted(viewer_from_fem.nodes.masses, key=lambda m: m.node_id)
    h5_masses = sorted(viewer_from_h5.nodes.masses, key=lambda m: m.node_id)
    assert len(fem_masses) == len(h5_masses) == 2
    assert all(isinstance(m, MassRow) for m in fem_masses + h5_masses)
    for a, b in zip(fem_masses, h5_masses):
        assert a.node_id == b.node_id
        np.testing.assert_allclose(a.mass, b.mass)


def test_sp_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    fem_sp = list(viewer_from_fem.nodes.sp)
    h5_sp = list(viewer_from_h5.nodes.sp)
    assert len(fem_sp) == len(h5_sp) == 1
    assert all(isinstance(r, SPRow) for r in fem_sp + h5_sp)
    assert fem_sp[0].node_id == h5_sp[0].node_id == 1
    assert fem_sp[0].dof == h5_sp[0].dof == 3
    assert fem_sp[0].is_homogeneous == h5_sp[0].is_homogeneous is True


# =====================================================================
# Element loads
# =====================================================================


def test_element_loads_parity(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    fem_loads = list(viewer_from_fem.elements.loads)
    h5_loads = list(viewer_from_h5.elements.loads)
    assert len(fem_loads) == len(h5_loads) == 1
    assert all(isinstance(r, ElementLoadRow) for r in fem_loads + h5_loads)
    assert fem_loads[0].element_id == h5_loads[0].element_id == 20
    assert fem_loads[0].load_type == h5_loads[0].load_type == "surfacePressure"
    assert fem_loads[0].params["pressure"] == h5_loads[0].params["pressure"] == -250.0


# =====================================================================
# Constraints (node-side + element-side)
# =====================================================================


def test_node_constraints_parity_counts(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    """Both sources see exactly one NodePairRow + one NodeGroupRow on
    the node side; one InterpolationRow on the element side."""
    fem_node_rows = list(viewer_from_fem.nodes.constraints)
    h5_node_rows = list(viewer_from_h5.nodes.constraints)
    assert len(fem_node_rows) == len(h5_node_rows) == 2
    pair_rows_f = [r for r in fem_node_rows if isinstance(r, NodePairRow)]
    pair_rows_h = [r for r in h5_node_rows if isinstance(r, NodePairRow)]
    assert len(pair_rows_f) == len(pair_rows_h) == 1
    assert pair_rows_f[0].kind == pair_rows_h[0].kind == "equal_dof"

    group_rows_f = [r for r in fem_node_rows if isinstance(r, NodeGroupRow)]
    group_rows_h = [r for r in h5_node_rows if isinstance(r, NodeGroupRow)]
    assert len(group_rows_f) == len(group_rows_h) == 1
    assert group_rows_f[0].slave_nodes == group_rows_h[0].slave_nodes


def test_node_constraints_pairs_expansion(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    """``.pairs()`` flattens NodeGroupRow into 3 NodePairRows + the
    direct NodePairRow → 4 pairs total."""
    for v in (viewer_from_fem, viewer_from_h5):
        pairs = list(v.nodes.constraints.pairs())
        assert len(pairs) == 4
        assert all(isinstance(p, NodePairRow) for p in pairs)


def test_element_constraints_interpolations(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    for v in (viewer_from_fem, viewer_from_h5):
        rows = list(v.elements.constraints.interpolations())
        assert len(rows) == 1
        assert isinstance(rows[0], InterpolationRow)
        assert rows[0].slave_node == 5
        assert rows[0].master_nodes == (1, 2, 3, 4)


def test_element_constraints_couplings_empty(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    for v in (viewer_from_fem, viewer_from_h5):
        assert list(v.elements.constraints.couplings()) == []


def test_phantom_nodes_empty_when_no_node_to_surface(
    viewer_from_fem: ViewerData,
    viewer_from_h5: ViewerData,
) -> None:
    """No NodeToSurface records in the fixture → phantom_nodes()
    returns empty arrays on both sources."""
    for v in (viewer_from_fem, viewer_from_h5):
        ids, coords = v.nodes.constraints.phantom_nodes()
        assert ids.size == 0
        assert coords.shape == (0, 3)


# =====================================================================
# Error surfaces
# =====================================================================


def test_unknown_constraint_record_class_raises() -> None:
    """:func:`constraint_row_from_record` rejects unknown subclasses
    so future writer-side additions surface as a clear error."""
    from apeGmsh.viewers.data._records import constraint_row_from_record

    class _UnknownRec:
        kind = "future_kind"

    with pytest.raises(ViewerDataDecodeError):
        constraint_row_from_record(_UnknownRec())


# =====================================================================
# Empty FEMData → ViewerData with empty composites
# =====================================================================


def test_empty_fem_round_trip(tmp_path: Path) -> None:
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=0,
    )
    nodes = NodeComposite(
        node_ids=np.array([1], dtype=np.int64),
        node_coords=np.array([[0.0, 0.0, 0.0]]),
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={}, physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=1, n_elems=0, bandwidth=0, types=[line_info]),
    )
    out = tmp_path / "empty.h5"
    fem.to_h5(str(out))
    v_fem = ViewerData.from_fem(fem)
    v_h5 = ViewerData.from_h5(str(out))
    for v in (v_fem, v_h5):
        assert v.nodes.ids.tolist() == [1]
        assert len(list(v.elements)) == 0
        assert list(v.nodes.loads) == []
        assert list(v.nodes.masses) == []
        assert list(v.nodes.constraints) == []
        # selection missing → empty
        assert v.nodes.selection.node_ids("nope").size == 0
