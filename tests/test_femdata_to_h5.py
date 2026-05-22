"""End-to-end coverage of ``FEMData.to_h5`` (Phase 8.5 commit 2).

Builds a small representative :class:`FEMData` in memory, writes it
to a fresh ``model.h5`` via the new neutral-zone writer, then re-opens
the file with raw ``h5py`` and walks every group asserting field-level
content matches what went in.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh._kernel.records._constraints import NodeGroupRecord, NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from tests.fixtures.schema import NEUTRAL_CURRENT
from apeGmsh._kernel.records._loads import (
    ElementLoadRecord,
    NodalLoadRecord,
    SPRecord,
)
from apeGmsh._kernel.records._masses import MassRecord


# =====================================================================
# Builders
# =====================================================================


def _make_fem() -> FEMData:
    """Construct a small FEMData with nodes, two element types, one PG, one
    label, a NodePair + NodeGroup constraint, a nodal load + element load,
    an SP record, and a couple of mass records.
    """
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)

    # One line2 element + one triangle3 element
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

    # One physical group + one label.  Each carries both node-side and
    # element-side data so we can verify the union-at-root layout.
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

    # Constraints — one NodePair (equal_dof) and one NodeGroup
    # (rigid_diaphragm).
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

    # Loads — one NodalLoad + one ElementLoad.
    nl_record = NodalLoadRecord(
        node_id=2, force_xyz=(1000.0, 0.0, 0.0), moment_xyz=None,
        pattern="gravity", name=None,
    )
    el_record = ElementLoadRecord(
        element_id=20, load_type="surfacePressure",
        params={"pressure": -250.0, "direction": "normal"},
        pattern="gravity",
    )
    sp_record = SPRecord(
        node_id=1, dof=3, value=0.0, is_homogeneous=True,
        pattern="default",
    )

    # Masses.
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
        constraints=None,
        loads=[el_record],
    )
    info = MeshInfo(
        n_nodes=5, n_elems=2, bandwidth=2,
        types=[line_info, tri_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


# =====================================================================
# Round-trip tests
# =====================================================================


def test_to_h5_writes_meta(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "model.h5"
    fem.to_h5(str(out), model_name="demo")
    with h5py.File(out, "r") as f:
        assert "meta" in f
        # Phase 2 bumped neutral 2.4.0 → 2.5.0 (additive: ``name``
        # field on every record dtype, /partitions/, /parts/,
        # snapshot_id verified on read).  Phase 6 (ADR 0021) bumped
        # again 2.5.0 → 2.6.0 for the additive /meta/lineage
        # sub-group.
        assert f["meta"].attrs["schema_version"] == NEUTRAL_CURRENT
        assert int(f["meta"].attrs["ndm"]) == 2  # max element dim
        assert f["meta"].attrs["model_name"] == "demo"


def test_to_h5_writes_nodes(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "nodes.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(
            f["nodes/ids"][:], [1, 2, 3, 4, 5],
        )
        assert f["nodes/coords"].shape == (5, 3)
        np.testing.assert_allclose(
            f["nodes/coords"][0], [0.0, 0.0, 0.0],
        )


def test_to_h5_writes_elements_by_type(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "elements.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "elements" in f
        types = sorted(f["elements"].keys())
        # GMSH alias table: code=1 → "line2", code=2 → "tri3".
        assert "line2" in types
        assert "tri3" in types
        np.testing.assert_array_equal(f["elements/line2/ids"][:], [10])
        np.testing.assert_array_equal(
            f["elements/tri3/connectivity"][:], [[2, 3, 5]],
        )


def test_to_h5_writes_physical_groups_at_root(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "pgs.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "physical_groups/Slab" in f
        slab = f["physical_groups/Slab"]
        assert int(slab.attrs["dim"]) == 2
        assert int(slab.attrs["tag"]) == 100
        np.testing.assert_array_equal(slab["node_ids"][:], [2, 3, 5])
        np.testing.assert_array_equal(slab["element_ids"][:], [20])


def test_to_h5_writes_labels_at_root(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "labels.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "labels/edge" in f
        edge = f["labels/edge"]
        assert int(edge.attrs["dim"]) == 1
        np.testing.assert_array_equal(edge["node_ids"][:], [1, 2])
        np.testing.assert_array_equal(edge["element_ids"][:], [10])


def test_to_h5_writes_constraints_per_kind(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "constraints.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        eq = f["constraints/equal_dof"][:]
        assert len(eq) == 1
        assert eq[0]["target_kind"] == b"node"
        assert eq[0]["payload_kind"] == b"equal_dof"
        np.testing.assert_array_equal(eq[0]["payload"]["dofs"], [1, 2, 3])

        rd = f["constraints/rigid_diaphragm"][:]
        assert len(rd) == 1
        payload = rd[0]["payload"]
        np.testing.assert_array_equal(payload["slave_nodes"], [2, 3, 4])
        # offsets packed flat as 3*n_slaves.
        np.testing.assert_array_equal(
            payload["offsets"].reshape(-1, 3),
            np.array([
                [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            ]),
        )
        np.testing.assert_allclose(
            payload["plane_normal"], [0.0, 0.0, 1.0],
        )


def test_to_h5_writes_loads_per_kind_per_pattern(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "loads.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "loads/nodal/gravity" in f
        assert "loads/element/gravity" in f
        nl = f["loads/nodal/gravity"][:]
        assert int(nl[0]["payload"]["node_id"]) == 2
        np.testing.assert_array_equal(
            nl[0]["payload"]["force_xyz"], [1000.0, 0.0, 0.0],
        )
        # moment_xyz NaN-filled when absent.
        assert np.isnan(nl[0]["payload"]["moment_xyz"]).all()

        el = f["loads/element/gravity"][:]
        # params encoded as JSON string in the payload.
        import json
        params = json.loads(el[0]["payload"]["params_json"])
        assert params["pressure"] == -250.0


def test_to_h5_writes_sp_records(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "sp.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        sp = f["loads/sp/default"][:]
        assert int(sp[0]["payload"]["dof"]) == 3
        assert int(sp[0]["payload"]["is_homogeneous"]) == 1


def test_to_h5_writes_masses(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "masses.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        masses = f["masses"][:]
        assert len(masses) == 2
        m4 = next(m for m in masses if int(m["payload"]["node_id"]) == 4)
        np.testing.assert_array_equal(
            m4["payload"]["mass"], [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        )


def test_to_h5_omits_empty_groups(tmp_path: Path) -> None:
    """A FEMData with no constraints/loads/masses produces no groups for them."""
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
    with h5py.File(out, "r") as f:
        assert "meta" in f
        assert "nodes" in f
        # Element parent group still created (writer always opens it
        # before iterating) but holds no per-type subgroups.
        assert "elements" in f
        assert list(f["elements"].keys()) == []
        # Optional groups absent.
        assert "constraints" not in f
        assert "loads" not in f
        assert "masses" not in f
        assert "physical_groups" not in f
        assert "labels" not in f
        assert "mesh_selections" not in f


def test_to_h5_snapshot_id_in_meta(tmp_path: Path) -> None:
    """``/meta/snapshot_id`` carries the broker's content hash."""
    fem = _make_fem()
    out = tmp_path / "snap.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["snapshot_id"] == fem.snapshot_id


# =====================================================================
# Phase 8.7 commit 2 — /mesh_selections/ round-trip
# =====================================================================


def _attach_mesh_selection_store(fem: FEMData) -> FEMData:
    """Attach a MeshSelectionStore with one node-only + one element-bearing set."""
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore

    sets = {
        (0, 1): {
            "name": "base_nodes",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            ], dtype=np.float64),
        },
        (2, 1): {
            "name": "slab_face",
            "node_ids": np.array([2, 3, 5], dtype=np.int64),
            "node_coords": np.array([
                [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0],
            ], dtype=np.float64),
            "element_ids": np.array([20], dtype=np.int64),
            "connectivity": np.array([[2, 3, 5]], dtype=np.int64),
        },
    }
    fem.mesh_selection = MeshSelectionStore(sets)
    return fem


def test_to_h5_writes_mesh_selections_at_root(tmp_path: Path) -> None:
    """A FEMData with ``mesh_selection`` populates ``/mesh_selections``."""
    fem = _attach_mesh_selection_store(_make_fem())
    out = tmp_path / "selections.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "mesh_selections" in f
        names = sorted(f["mesh_selections"].keys())
        assert names == ["base_nodes", "slab_face"]

        base = f["mesh_selections/base_nodes"]
        assert int(base.attrs["dim"]) == 0
        assert int(base.attrs["tag"]) == 1
        assert base.attrs["name"] == "base_nodes"
        np.testing.assert_array_equal(base["node_ids"][:], [1, 2])
        # dim=0 → no element_ids / connectivity datasets
        assert "element_ids" not in base
        assert "connectivity" not in base

        slab = f["mesh_selections/slab_face"]
        assert int(slab.attrs["dim"]) == 2
        np.testing.assert_array_equal(slab["node_ids"][:], [2, 3, 5])
        np.testing.assert_array_equal(slab["element_ids"][:], [20])
        # connectivity persisted alongside element_ids (rows 1:1)
        np.testing.assert_array_equal(slab["connectivity"][:], [[2, 3, 5]])


def test_to_h5_omits_mesh_selections_when_absent(tmp_path: Path) -> None:
    """No ``fem.mesh_selection`` → no ``/mesh_selections`` group."""
    fem = _make_fem()
    # _make_fem() does not attach a selection store.
    assert getattr(fem, "mesh_selection", None) is None
    out = tmp_path / "no_selections.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "mesh_selections" not in f


# =====================================================================
# Phase 2 — B1: ``name`` field round-trips on every record dtype
# =====================================================================


def test_to_h5_preserves_constraint_name(tmp_path: Path) -> None:
    """Constraint records carry their pre-mesh ``name`` through to_h5.

    Audit gap B1: every record dataclass under ``_kernel/records/*.py``
    has a ``name: str | None`` field but the on-disk payload dtypes
    silently dropped it.  v2.5.0 adds ``name`` to every payload dtype
    so ``fem.inspect.constraint_summary()`` shows the source hint
    after a round-trip.
    """
    fem = _make_fem()
    out = tmp_path / "names.h5"
    fem.to_h5(str(out))

    from apeGmsh.mesh.FEMData import FEMData
    rebuilt = FEMData.from_h5(str(out))

    by_kind = {r.kind: r for r in rebuilt.nodes.constraints}
    assert by_kind[ConstraintKind.EQUAL_DOF].name == "weld"
    assert by_kind[ConstraintKind.RIGID_DIAPHRAGM].name == "floor1"


def test_to_h5_preserves_load_name(tmp_path: Path) -> None:
    """NodalLoad / ElementLoad records carry ``name`` through to_h5."""
    # Build a small FEM with named loads.
    node_ids = np.array([1, 2], dtype=np.int64)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=1,
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2, 1]], dtype=np.int64),
    )
    nl_named = NodalLoadRecord(
        node_id=2, force_xyz=(100.0, 0.0, 0.0), moment_xyz=None,
        pattern="gravity", name="point_load_A",
    )
    el_named = ElementLoadRecord(
        element_id=10, load_type="surfacePressure",
        params={"pressure": -100.0},
        pattern="gravity", name="roof_pressure",
    )
    sp_named = SPRecord(
        node_id=1, dof=3, value=0.0, is_homogeneous=True,
        pattern="default", name="fixed_base",
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        loads=[nl_named], sp=[sp_named],
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        loads=[el_named],
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=2, n_elems=1, bandwidth=1, types=[tri_info]),
    )

    out = tmp_path / "load_names.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    rebuilt_nl = list(rebuilt.nodes.loads)
    assert len(rebuilt_nl) == 1
    assert rebuilt_nl[0].name == "point_load_A"
    rebuilt_el = list(rebuilt.elements.loads)
    assert len(rebuilt_el) == 1
    assert rebuilt_el[0].name == "roof_pressure"
    rebuilt_sp = list(rebuilt.nodes.sp)
    assert len(rebuilt_sp) == 1
    assert rebuilt_sp[0].name == "fixed_base"


def test_to_h5_preserves_mass_name(tmp_path: Path) -> None:
    """MassRecord carries ``name`` through to_h5."""
    node_ids = np.array([1, 2], dtype=np.int64)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    m_named = MassRecord(
        node_id=2, mass=(50.0, 50.0, 50.0, 0.0, 0.0, 0.0),
        name="lumped_floor",
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        masses=[m_named],
    )
    elements = ElementComposite(
        groups={}, physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=2, n_elems=0, bandwidth=0, types=[]),
    )

    out = tmp_path / "mass_names.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    rebuilt_m = list(rebuilt.nodes.masses)
    assert len(rebuilt_m) == 1
    assert rebuilt_m[0].name == "lumped_floor"


# =====================================================================
# Phase 2 — B2: ``/partitions/`` group written when partitioned
# =====================================================================


def test_to_h5_writes_partitions(tmp_path: Path) -> None:
    """Partitioned FEMData emits ``/partitions/{id}/...`` groups.

    Audit gap B2 (write side): the writer didn't emit partitions or
    parts.  v2.5.0 adds ``/partitions/{id}/node_ids`` +
    ``/partitions/{id}/element_ids`` for each partition; absence
    means the mesh wasn't partitioned (omit-empty-groups).
    """
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=2,
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([10, 11], dtype=np.int64),
        connectivity=np.array([[1, 2, 3], [2, 4, 3]], dtype=np.int64),
    )
    partitions = {
        1: {
            "node_ids": np.array([1, 2, 3], dtype=np.int64),
            "element_ids": np.array([10], dtype=np.int64),
        },
        2: {
            "node_ids": np.array([2, 3, 4], dtype=np.int64),
            "element_ids": np.array([11], dtype=np.int64),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        partitions=partitions,
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        partitions=partitions,
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=2, types=[tri_info]),
    )

    out = tmp_path / "partitions.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "partitions" in f
        keys = sorted(f["partitions"].keys())
        assert keys == ["1", "2"]
        np.testing.assert_array_equal(
            f["partitions/1/node_ids"][:], [1, 2, 3])
        np.testing.assert_array_equal(
            f["partitions/1/element_ids"][:], [10])
        np.testing.assert_array_equal(
            f["partitions/2/node_ids"][:], [2, 3, 4])
        np.testing.assert_array_equal(
            f["partitions/2/element_ids"][:], [11])


def test_to_h5_writes_parts(tmp_path: Path) -> None:
    """FEM with parts maps emits ``/parts/{label}/...`` groups."""
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    ], dtype=np.float64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=0,
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        part_node_map={"slab_A": {1, 2}, "slab_B": {3}},
    )
    elements = ElementComposite(
        groups={}, physical=PhysicalGroupSet({}), labels=LabelSet({}),
        part_elem_map={"slab_A": {10}},
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=3, n_elems=0, bandwidth=0, types=[tri_info]),
    )

    out = tmp_path / "parts.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "parts" in f
        assert sorted(f["parts"].keys()) == ["slab_A", "slab_B"]
        np.testing.assert_array_equal(
            f["parts/slab_A/node_ids"][:], [1, 2])
        np.testing.assert_array_equal(
            f["parts/slab_A/element_ids"][:], [10])
        np.testing.assert_array_equal(
            f["parts/slab_B/node_ids"][:], [3])
        # slab_B had no elements → empty element_ids dataset
        np.testing.assert_array_equal(
            f["parts/slab_B/element_ids"][:], [])


def test_to_h5_omits_partitions_and_parts_when_empty(tmp_path: Path) -> None:
    """A non-partitioned FEM with no parts has neither group."""
    fem = _make_fem()
    out = tmp_path / "no_parts.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "partitions" not in f
        assert "parts" not in f
