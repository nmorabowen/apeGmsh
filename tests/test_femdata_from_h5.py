"""Round-trip coverage of ``FEMData.from_h5`` (session-save Phase 2).

Builds a FEMData with at least one record of every type the writer
handles, writes via ``to_h5``, reads back via ``from_h5``, and asserts
field-level equality.  Companion to ``test_femdata_to_h5`` (which
covers writer correctness alone — this file covers the writer-reader
contract).
"""
from __future__ import annotations

from pathlib import Path

import h5py
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
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh._kernel.records._loads import (
    ElementLoadRecord,
    NodalLoadRecord,
    SPRecord,
)
from apeGmsh._kernel.records._masses import MassRecord


# =====================================================================
# Builder — every record type covered
# =====================================================================


def _make_full_fem() -> FEMData:
    node_ids = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 1.5],
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
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]),
        "element_ids": np.array([20], dtype=np.int64),
    }}
    labels = {(1, 200): {
        "name": "edge",
        "node_ids": np.array([1, 2], dtype=np.int64),
        "node_coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "element_ids": np.array([10], dtype=np.int64),
    }}

    # -- Node-side constraints --
    np_record = NodePairRecord(
        kind=ConstraintKind.EQUAL_DOF, name="weld",
        master_node=1, slave_node=2, dofs=[1, 2, 3], offset=None,
    )
    rb_record = NodePairRecord(
        kind=ConstraintKind.RIGID_BEAM, name="arm",
        master_node=1, slave_node=3, dofs=[1, 2, 3, 4, 5, 6],
        offset=np.array([1.0, 1.0, 0.0]),
    )
    pen_record = NodePairRecord(
        kind=ConstraintKind.PENALTY, name="spring",
        master_node=2, slave_node=3, dofs=[1],
        offset=None, penalty_stiffness=1.0e9,
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
    nts_record = NodeToSurfaceRecord(
        kind=ConstraintKind.NODE_TO_SURFACE, name="hub",
        master_node=6, slave_nodes=[2, 3], phantom_nodes=[100, 101],
        phantom_coords=np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        dofs=[1, 2, 3],
    )

    # -- Element-side constraints --
    interp_record = InterpolationRecord(
        kind=ConstraintKind.TIE, name="tie_a",
        slave_node=5, master_nodes=[2, 3, 4],
        weights=np.array([0.4, 0.4, 0.2]),
        dofs=[1, 2, 3],
        projected_point=np.array([0.7, 0.7, 0.0]),
        parametric_coords=np.array([0.3, 0.3]),
    )
    coup_record = SurfaceCouplingRecord(
        kind=ConstraintKind.MORTAR, name="mortar_face",
        master_nodes=[2, 3], slave_nodes=[6, 7],
        dofs=[1, 2, 3],
        mortar_operator=np.array([[0.5, 0.5], [0.5, 0.5]]),
        slave_records=[
            InterpolationRecord(
                kind=ConstraintKind.TIE,
                slave_node=6, master_nodes=[2, 3],
                weights=np.array([0.5, 0.5]),
                dofs=[1, 2, 3],
                projected_point=np.array([0.5, 0.0, 0.0]),
                parametric_coords=np.array([0.5, 0.0]),
            ),
            InterpolationRecord(
                kind=ConstraintKind.TIE,
                slave_node=7, master_nodes=[2, 3],
                weights=np.array([0.25, 0.75]),
                dofs=[1, 2, 3],
                projected_point=np.array([0.75, 0.0, 0.0]),
                parametric_coords=np.array([0.75, 0.0]),
            ),
        ],
    )

    # -- Loads --
    nl_record = NodalLoadRecord(
        node_id=2, force_xyz=(1000.0, 0.0, 0.0), moment_xyz=None,
        pattern="gravity", name=None,
    )
    nl_record_2 = NodalLoadRecord(
        node_id=3, force_xyz=None, moment_xyz=(0.0, 0.0, 5.0),
        pattern="wind", name=None,
    )
    el_record = ElementLoadRecord(
        element_id=20, load_type="surfacePressure",
        params={"pressure": -250.0, "direction": "normal"},
        pattern="gravity",
    )
    sp_homog = SPRecord(
        node_id=1, dof=3, value=0.0, is_homogeneous=True,
        pattern="default",
    )
    sp_prescribed = SPRecord(
        node_id=4, dof=1, value=0.05, is_homogeneous=False,
        pattern="default",
    )

    # -- Masses --
    m1 = MassRecord(node_id=4, mass=(100.0, 100.0, 100.0, 0.0, 0.0, 0.0))
    m2 = MassRecord(node_id=5, mass=(50.0, 50.0, 50.0, 1.0, 1.0, 1.0))

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet(labels),
        constraints=[np_record, rb_record, pen_record, ng_record, nts_record],
        loads=[nl_record, nl_record_2],
        sp=[sp_homog, sp_prescribed],
        masses=[m1, m2],
    )
    elements = ElementComposite(
        groups={1: line_group, 2: tri_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet(labels),
        constraints=[interp_record, coup_record],
        loads=[el_record],
    )
    info = MeshInfo(
        n_nodes=7, n_elems=2, bandwidth=2,
        types=[line_info, tri_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


# =====================================================================
# Core round-trip
# =====================================================================


def test_round_trip_nodes_and_elements(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out), model_name="rt")

    rebuilt = FEMData.from_h5(str(out))

    np.testing.assert_array_equal(
        rebuilt.nodes.ids.astype(np.int64), fem.nodes.ids.astype(np.int64),
    )
    np.testing.assert_allclose(rebuilt.nodes.coords, fem.nodes.coords)

    # Two element types, preserved by code
    rebuilt_codes = {g.element_type.code for g in rebuilt.elements}
    assert rebuilt_codes == {1, 2}
    tri = next(g for g in rebuilt.elements if g.element_type.code == 2)
    np.testing.assert_array_equal(tri.ids, [20])
    np.testing.assert_array_equal(tri.connectivity, [[2, 3, 5]])


def test_round_trip_physical_groups_and_labels(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    # Both sides see the (dim, tag) entry; element side carries element_ids
    pg_keys = list(rebuilt.nodes.physical.get_all())
    assert (2, 100) in pg_keys
    assert rebuilt.nodes.physical.get_name(2, 100) == "Slab"

    # Element-side element_ids round-trip
    eids = rebuilt.elements.physical.element_ids((2, 100))
    np.testing.assert_array_equal(eids, [20])

    # Labels too
    assert (1, 200) in list(rebuilt.nodes.labels.get_all())
    assert rebuilt.nodes.labels.get_name(1, 200) == "edge"


def test_round_trip_node_pair_records(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    cs = rebuilt.nodes.constraints

    eq = [r for r in cs if r.kind == ConstraintKind.EQUAL_DOF]
    assert len(eq) == 1
    assert eq[0].master_node == 1 and eq[0].slave_node == 2
    assert eq[0].dofs == [1, 2, 3]
    assert eq[0].offset is None
    assert eq[0].penalty_stiffness is None

    rb = [r for r in cs if r.kind == ConstraintKind.RIGID_BEAM]
    assert len(rb) == 1
    np.testing.assert_allclose(rb[0].offset, [1.0, 1.0, 0.0])

    pen = [r for r in cs if r.kind == ConstraintKind.PENALTY]
    assert len(pen) == 1
    assert pen[0].penalty_stiffness == pytest.approx(1.0e9)


def test_round_trip_node_group_record(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    ng = [
        r for r in rebuilt.nodes.constraints
        if r.kind == ConstraintKind.RIGID_DIAPHRAGM
    ]
    assert len(ng) == 1
    assert ng[0].master_node == 1
    assert ng[0].slave_nodes == [2, 3, 4]
    assert ng[0].dofs == [1, 2, 6]
    np.testing.assert_allclose(ng[0].offsets, [
        [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    np.testing.assert_allclose(ng[0].plane_normal, [0.0, 0.0, 1.0])


def test_round_trip_node_to_surface_record(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    nts = [
        r for r in rebuilt.nodes.constraints
        if r.kind == ConstraintKind.NODE_TO_SURFACE
    ]
    assert len(nts) == 1
    r = nts[0]
    assert r.master_node == 6
    assert r.slave_nodes == [2, 3]
    assert r.phantom_nodes == [100, 101]
    np.testing.assert_allclose(r.phantom_coords, [
        [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
    ])
    assert r.dofs == [1, 2, 3]

    # The sub-records are NOT persisted but MUST be re-derived on
    # decode — otherwise every emission iterator returns empty and the
    # reloaded model is silently disconnected (the PR-A bug).
    assert [(p.master_node, p.slave_node) for p in r.rigid_link_records] \
        == [(6, 100), (6, 101)]
    assert all(p.kind == ConstraintKind.RIGID_BEAM
               for p in r.rigid_link_records)
    # offset re-derived from node coords (node_xyz plumbing) — present,
    # finite, shape (3,).
    for p in r.rigid_link_records:
        assert p.offset is not None and np.all(np.isfinite(p.offset))
        assert np.asarray(p.offset).shape == (3,)
    assert [(p.master_node, p.slave_node, tuple(p.dofs))
            for p in r.equal_dof_records] \
        == [(100, 2, (1, 2, 3)), (101, 3, (1, 2, 3))]
    assert all(p.kind == ConstraintKind.EQUAL_DOF
               for p in r.equal_dof_records)

    # End-to-end: the iterators OpenSees emission consumes must be
    # non-empty after reload (this is what was silently broken).
    cons = rebuilt.nodes.constraints
    rl = [(m, s) for m, slaves in cons.rigid_link_groups() for s in slaves]
    assert (6, 100) in rl and (6, 101) in rl
    ed = [(p.master_node, p.slave_node) for p in cons.equal_dofs()]
    assert (100, 2) in ed and (101, 3) in ed


def test_round_trip_interpolation_record(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    ties = list(rebuilt.elements.constraints.interpolations())
    # Three: the standalone TIE + the SurfaceCoupling's two
    # slave_records — which NOW round-trip (CSR sr_* payload fields).
    assert len(ties) == 3
    # The standalone TIE (slave_node=5) is identified by content, not
    # position — ordering of standalone vs coupling-expanded
    # interpolations is not part of the contract.
    rec = next(r for r in ties if r.slave_node == 5)
    assert rec.kind == ConstraintKind.TIE
    assert rec.master_nodes == [2, 3, 4]
    np.testing.assert_allclose(rec.weights, [0.4, 0.4, 0.2])
    np.testing.assert_allclose(rec.projected_point, [0.7, 0.7, 0.0])
    np.testing.assert_allclose(rec.parametric_coords, [0.3, 0.3])
    # The coupling's slave_records also appear (lossless round-trip).
    assert {r.slave_node for r in ties} == {5, 6, 7}


def test_round_trip_surface_coupling_record(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    coups = list(rebuilt.elements.constraints.couplings())
    assert len(coups) == 1
    c = coups[0]
    assert c.kind == ConstraintKind.MORTAR
    assert c.master_nodes == [2, 3]
    assert c.slave_nodes == [6, 7]
    assert c.dofs == [1, 2, 3]
    np.testing.assert_allclose(
        c.mortar_operator, [[0.5, 0.5], [0.5, 0.5]],
    )
    # slave_records now round-trip losslessly (CSR sr_* fields).
    assert len(c.slave_records) == 2
    sr0, sr1 = c.slave_records
    assert sr0.slave_node == 6 and sr0.master_nodes == [2, 3]
    np.testing.assert_allclose(sr0.weights, [0.5, 0.5])
    np.testing.assert_allclose(sr0.projected_point, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(sr0.parametric_coords, [0.5, 0.0])
    assert sr1.slave_node == 7
    np.testing.assert_allclose(sr1.weights, [0.25, 0.75])
    assert [p.dofs for p in c.slave_records] == [[1, 2, 3], [1, 2, 3]]


def test_round_trip_nodal_loads(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    grav = rebuilt.nodes.loads.by_pattern("gravity")
    assert len(grav) == 1
    assert grav[0].node_id == 2
    np.testing.assert_allclose(grav[0].force_xyz, [1000.0, 0.0, 0.0])
    assert grav[0].moment_xyz is None

    wind = rebuilt.nodes.loads.by_pattern("wind")
    assert len(wind) == 1
    assert wind[0].force_xyz is None
    np.testing.assert_allclose(wind[0].moment_xyz, [0.0, 0.0, 5.0])


def test_round_trip_element_loads(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    el = list(rebuilt.elements.loads)
    assert len(el) == 1
    assert el[0].element_id == 20
    assert el[0].load_type == "surfacePressure"
    assert el[0].params["pressure"] == -250.0
    assert el[0].params["direction"] == "normal"
    assert el[0].pattern == "gravity"


def test_round_trip_sp_records(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    sps = list(rebuilt.nodes.sp)
    assert len(sps) == 2

    hom = next(r for r in sps if r.is_homogeneous)
    pre = next(r for r in sps if not r.is_homogeneous)
    assert hom.node_id == 1 and hom.dof == 3 and hom.value == 0.0
    assert pre.node_id == 4 and pre.dof == 1 and pre.value == pytest.approx(0.05)


def test_round_trip_masses(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    assert len(rebuilt.nodes.masses) == 2
    m4 = rebuilt.nodes.masses.by_node(4)
    assert m4 is not None
    assert m4.mass == (100.0, 100.0, 100.0, 0.0, 0.0, 0.0)


def test_round_trip_mesh_selections(tmp_path: Path) -> None:
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore

    fem = _make_full_fem()
    fem.mesh_selection = MeshSelectionStore({
        (0, 1): {
            "name": "base_nodes",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64,
            ),
        },
        (2, 1): {
            "name": "slab_face",
            "node_ids": np.array([2, 3, 5], dtype=np.int64),
            "node_coords": np.array([
                [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0],
            ], dtype=np.float64),
            "element_ids": np.array([20], dtype=np.int64),
        },
    })
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    assert rebuilt.mesh_selection is not None
    keys = list(rebuilt.mesh_selection.get_all())
    assert (0, 1) in keys
    assert (2, 1) in keys
    assert rebuilt.mesh_selection.get_name(2, 1) == "slab_face"


def test_round_trip_mesh_selection_element_connectivity(
    tmp_path: Path,
) -> None:
    """Regression: an element-bearing mesh selection keeps its
    connectivity across the HDF5 round-trip, so the reloaded
    ``MeshSelectionStore.get_elements`` returns data instead of
    raising "no element data" (the pre-fix behaviour where the
    writer dropped connectivity)."""
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore

    fem = _make_full_fem()
    fem.mesh_selection = MeshSelectionStore({
        (2, 1): {
            "name": "slab_face",
            "node_ids": np.array([2, 3, 5], dtype=np.int64),
            "node_coords": np.array([
                [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0],
            ], dtype=np.float64),
            "element_ids": np.array([20], dtype=np.int64),
            "connectivity": np.array([[2, 3, 5]], dtype=np.int64),
        },
    })
    out = tmp_path / "rt_conn.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    assert rebuilt.mesh_selection is not None
    # Pre-fix this raised ValueError("...has no element data...").
    elems = rebuilt.mesh_selection.get_elements(2, 1)
    np.testing.assert_array_equal(elems["element_ids"], [20])
    np.testing.assert_array_equal(elems["connectivity"], [[2, 3, 5]])


# =====================================================================
# Empty composites + snapshot id
# =====================================================================


def test_empty_fem_round_trips(tmp_path: Path) -> None:
    nodes = NodeComposite(
        node_ids=np.array([1], dtype=np.int64),
        node_coords=np.array([[0.0, 0.0, 0.0]]),
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=0,
    )
    elements = ElementComposite(
        groups={}, physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=1, n_elems=0, bandwidth=0, types=[info]),
    )

    out = tmp_path / "empty.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    assert len(rebuilt.nodes.ids) == 1
    assert len(list(rebuilt.elements)) == 0
    assert len(rebuilt.nodes.constraints) == 0
    assert len(rebuilt.nodes.loads) == 0
    assert len(rebuilt.nodes.masses) == 0


def test_snapshot_id_preserved(tmp_path: Path) -> None:
    """Writer + reader preserve enough state that snapshot_id matches."""
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))
    # snapshot_id is a content hash over nodes/elements/PGs/labels —
    # the round-trip should preserve it exactly.
    assert rebuilt.snapshot_id == fem.snapshot_id


# =====================================================================
# Error paths
# =====================================================================


def test_missing_meta_raises(tmp_path: Path) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "bad.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("dummy", data=[1, 2, 3])

    with pytest.raises(MalformedH5Error, match="missing /meta"):
        FEMData.from_h5(str(out))


def test_wrong_schema_major_raises(tmp_path: Path) -> None:
    from apeGmsh.opensees.emitter.h5_reader import SchemaVersionError

    out = tmp_path / "v1.h5"
    with h5py.File(out, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "1.0.0"

    with pytest.raises(SchemaVersionError, match="major 1"):
        FEMData.from_h5(str(out))


# =====================================================================
# Session-level integration — apeGmsh save_to → FEMData.from_h5
# =====================================================================


def test_session_save_then_from_h5(tmp_path: Path) -> None:
    """The autosave path produces a file that FEMData.from_h5 can read."""
    from apeGmsh import apeGmsh

    out = tmp_path / "session.h5"
    with apeGmsh(model_name="s", save_to=out) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
        g.physical.add_volume("body", name="body")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)

    rebuilt = FEMData.from_h5(str(out))
    assert len(rebuilt.nodes.ids) > 0
    assert len(list(rebuilt.elements)) > 0
    # PG "body" survives the round trip
    assert "body" in [
        rebuilt.nodes.physical.get_name(d, t)
        for d, t in rebuilt.nodes.physical.get_all()
    ]
