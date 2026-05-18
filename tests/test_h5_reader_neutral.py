"""Tests for the reference reader's neutral-zone accessors.

Companion to ``tests/test_femdata_to_h5.py``: builds a representative
FEMData via the broker, writes it through ``fem.to_h5(...)``, then
re-opens through :mod:`apeGmsh.opensees.emitter.h5_reader` and walks
the new Phase 8.5 accessors.
"""
from __future__ import annotations

from pathlib import Path

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
from apeGmsh._kernel.records._loads import NodalLoadRecord, SPRecord
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh.opensees.emitter import h5_reader


def _make_fem() -> FEMData:
    """Same shape as tests/test_femdata_to_h5.py::_make_fem."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    pg = {(1, 100): {
        "name": "Edge",
        "node_ids": np.array([1, 2], dtype=np.int64),
        "node_coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "element_ids": np.array([10], dtype=np.int64),
    }}
    np_rec = NodePairRecord(
        kind=ConstraintKind.EQUAL_DOF, master_node=1, slave_node=2,
        dofs=[1, 2, 3],
    )
    ng_rec = NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM, master_node=3,
        slave_nodes=[1, 2, 4], dofs=[1, 2, 6],
    )
    load = NodalLoadRecord(node_id=2, force_xyz=(1.0e3, 0.0, 0.0), pattern="g1")
    sp = SPRecord(node_id=1, dof=1, value=0.0, is_homogeneous=True)
    mass = MassRecord(node_id=4, mass=(100.0,) * 6)

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        constraints=[np_rec, ng_rec],
        loads=[load],
        sp=[sp],
        masses=[mass],
    )
    elements = ElementComposite(
        groups={1: line_group}, physical=PhysicalGroupSet(pg),
        labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=1, bandwidth=0, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


def test_reader_accepts_broker_2_1_0_file(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "broker.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        assert m.schema_version.startswith("2.")


def test_reader_nodes_accessor(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "nodes.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        nodes = m.nodes()
        np.testing.assert_array_equal(nodes["ids"], [1, 2, 3, 4])
        assert nodes["coords"].shape == (4, 3)


def test_reader_element_arrays_accessor(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "elements.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        # elements() still returns attrs-only dict.
        types = m.elements()
        assert "line2" in types
        # The companion accessor exposes the broker datasets.
        arrays = m.element_arrays("line2")
        np.testing.assert_array_equal(arrays["ids"], [10])
        np.testing.assert_array_equal(arrays["connectivity"], [[1, 2]])


def test_reader_physical_groups_accessor(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "pgs.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        pgs = m.physical_groups()
        assert "Edge" in pgs
        edge = pgs["Edge"]
        assert int(edge["dim"]) == 1
        assert int(edge["tag"]) == 100
        np.testing.assert_array_equal(edge["node_ids"], [1, 2])
        np.testing.assert_array_equal(edge["element_ids"], [10])


def test_reader_constraints_accessor(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "c.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        cs = m.constraints()
        assert "equal_dof" in cs
        assert "rigid_diaphragm" in cs
        eq = cs["equal_dof"]
        assert eq[0]["payload_kind"] == b"equal_dof"
        np.testing.assert_array_equal(eq[0]["payload"]["dofs"], [1, 2, 3])


def test_reader_loads_accessor(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "l.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        loads = m.loads()
        # Both nodal-load (from g1 pattern) and SP (under default).
        assert "nodal" in loads
        assert "g1" in loads["nodal"]
        assert "sp" in loads
        assert "default" in loads["sp"]
        nl = loads["nodal"]["g1"]
        assert int(nl[0]["payload"]["node_id"]) == 2


def test_reader_masses_accessor(tmp_path: Path) -> None:
    fem = _make_fem()
    out = tmp_path / "m.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        masses = m.masses()
        assert masses is not None
        assert len(masses) == 1
        np.testing.assert_array_equal(
            masses[0]["payload"]["mass"],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )


def test_reader_mesh_selections_accessor(tmp_path: Path) -> None:
    """Phase 8.7 commit 2: ``/mesh_selections`` reader returns the same
    shape as ``physical_groups()`` / ``labels()``."""
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore

    fem = _make_fem()
    fem.mesh_selection = MeshSelectionStore({
        (0, 1): {
            "name": "anchor",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            ], dtype=np.float64),
        },
        (1, 1): {
            "name": "edge_picks",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            ], dtype=np.float64),
            "element_ids": np.array([10], dtype=np.int64),
            "connectivity": np.array([[1, 2]], dtype=np.int64),
        },
    })
    out = tmp_path / "selections.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        ms = m.mesh_selections()
        assert sorted(ms.keys()) == ["anchor", "edge_picks"]
        anchor = ms["anchor"]
        assert int(anchor["dim"]) == 0
        np.testing.assert_array_equal(anchor["node_ids"], [1, 2])
        # dim=0 → element_ids absent.
        assert "element_ids" not in anchor
        edge = ms["edge_picks"]
        assert int(edge["dim"]) == 1
        np.testing.assert_array_equal(edge["element_ids"], [10])


def test_reader_mesh_selections_empty_when_absent(tmp_path: Path) -> None:
    """Files without ``/mesh_selections`` (pre-2.4.0 or empty store)
    return an empty dict — graceful degradation per the schema's
    additive-bump policy."""
    fem = _make_fem()
    out = tmp_path / "no_selections.h5"
    fem.to_h5(str(out))
    with h5_reader.open(str(out)) as m:
        assert m.mesh_selections() == {}


def test_reader_returns_none_for_absent_masses(tmp_path: Path) -> None:
    """Empty broker → no /masses group → accessor returns None."""
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
    with h5_reader.open(str(out)) as m:
        assert m.masses() is None
        assert m.constraints() == {}
        assert m.loads() == {}
        assert m.physical_groups() == {}
        assert m.labels() == {}
        assert m.mesh_selections() == {}
