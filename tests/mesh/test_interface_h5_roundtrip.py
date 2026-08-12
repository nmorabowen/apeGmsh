"""Neutral-zone H5 round-trip for oriented coincident-pair zeroLength
interfaces (ADR 0093 S6).

``fem.elements.interfaces`` (``g.constraints.interface()``) now persists
through ``FEMData.to_h5`` / ``from_h5`` into a dedicated ``/interfaces``
group (neutral schema 2.29.0). Until S6 the writer *refused* the save
outright (the S3 loud guard, retired here) because there was no
persisted representation at all — this file is that guard flipped from
"asserts refusal" to "asserts survival".

Two fixture styles, deliberately:

* a real two-square mesh driven through the verb, so what round-trips is
  what the resolver actually produces (orientation, tributary areas,
  phantom bridge);
* a hand-built minimal FEMData for the encode-side fail-loud cases,
  where a real mesh buys nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NodePairRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION, read_fem_h5
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)
THICKNESS = 0.5


# ── real-mesh fixture (the verb's own output) ────────────────────────

def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _interface_fem(*, slave_ndf=None, normal=NORMAL, n: int = 2):
    with apeGmsh(model_name="iface_h5", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="rock")
        g.physical.add(2, [right], name="liner")
        g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
        g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
        g.constraints.interface(
            "face", "wire", normal=normal, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=slave_ndf, name="RockLiner")
        return g.mesh.queries.get_fem_data()


def _plain_fem():
    with apeGmsh(model_name="iface_h5_plain", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, surf)], n=2)
        g.mesh.generation.generate(2)
        g.physical.add(2, [surf], name="rock")
        return g.mesh.queries.get_fem_data(dim=2)


def _roundtrip(fem, tmp_path):
    p = str(tmp_path / "m.h5")
    fem.to_h5(p)
    return read_fem_h5(p), p


def _eq(a, b):
    """Field-exact comparison of two :class:`InterfaceRecord`."""
    assert a.kind == b.kind and a.name == b.name
    assert a.master_node == b.master_node
    assert a.slave_node == b.slave_node
    assert a.backing_element == b.backing_element
    assert (a.orient is None) == (b.orient is None)
    if a.orient is not None:
        assert tuple(a.orient) == pytest.approx(tuple(b.orient), abs=0.0)
    assert a.a_trib == pytest.approx(b.a_trib, abs=0.0)
    assert a.normal_law == b.normal_law
    assert a.tangential_law == b.tangential_law
    assert a.phantom_node == b.phantom_node
    assert a.phantom_ndf == b.phantom_ndf
    assert (a.phantom_coords is None) == (b.phantom_coords is None)
    if a.phantom_coords is not None:
        np.testing.assert_array_equal(
            np.asarray(a.phantom_coords, dtype=float),
            np.asarray(b.phantom_coords, dtype=float))
    assert len(a.equal_dof_records) == len(b.equal_dof_records)
    for x, y in zip(a.equal_dof_records, b.equal_dof_records):
        assert x == y


# ======================================================================
# The S3 guard, flipped: saving an interface model now SURVIVES
# ======================================================================
def test_equal_ndf_interfaces_roundtrip_field_exact(tmp_path):
    fem = _interface_fem()
    src = fem.elements.interfaces
    assert len(src) == 2
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.interfaces
    assert len(got) == len(src)
    for a, b in zip(got, src):
        _eq(a, b)
    # the values that matter, read positively rather than by comparison
    assert got[0].name == "RockLiner"
    assert got[0].normal_law == NormalLaw(kind="ent", k_per_area=1.0e9)
    assert got[0].tangential_law == TANGENTIAL
    assert got[0].phantom_node is None
    assert got[0].equal_dof_records == []
    assert tuple(got[0].orient) == pytest.approx((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))


def test_mixed_ndf_phantom_and_nested_equaldof_roundtrip(tmp_path):
    fem = _interface_fem(slave_ndf=3)
    src = fem.elements.interfaces
    assert len(src) == 2 and all(r.phantom_node is not None for r in src)
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.interfaces
    for a, b in zip(got, src):
        _eq(a, b)
    for rec in got:
        assert rec.phantom_ndf == 2
        assert rec.phantom_coords is not None
        assert len(rec.equal_dof_records) == 1
        eq = rec.equal_dof_records[0]
        assert eq.kind == ConstraintKind.EQUAL_DOF
        assert eq.master_node == rec.slave_node       # retained: real beam
        assert eq.slave_node == rec.phantom_node      # constrained: phantom
        assert eq.dofs == [1, 2]
        assert eq.name == "RockLiner"


def test_epp_gap_normal_law_scalars_roundtrip(tmp_path):
    # The law with the most optional scalars (tau_b_n + gap, both NaN
    # sentinels for the other kinds) — and the one whose signs INV-1
    # governs, so silently losing gap<0 would be a real hazard.
    law = NormalLaw(kind="epp_gap", k_per_area=1.0e9, tau_b_n=1.5e5,
                    gap=-1.0e-3)
    fem = _interface_fem(normal=law)
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.interfaces[0]
    assert got.normal_law == law
    assert got.normal_law.gap == pytest.approx(-1.0e-3)
    assert got.normal_law.tau_b_n == pytest.approx(1.5e5)


def test_interface_free_model_omits_group_and_keeps_snapshot(tmp_path):
    import h5py
    fem = _plain_fem()
    assert not fem.elements.interfaces
    back, p = _roundtrip(fem, tmp_path)
    with h5py.File(p, "r") as f:
        assert "interfaces" not in f                   # group omitted
    assert back.snapshot_id == fem.snapshot_id


def test_interface_snapshot_id_stable_on_roundtrip(tmp_path):
    # snapshot_id excludes the interface overlay (consistent with
    # contacts / ties), so an interface model round-trips with an
    # identical id even though the records come back into the broker.
    fem = _interface_fem()
    back, _ = _roundtrip(fem, tmp_path)
    assert back.snapshot_id == fem.snapshot_id
    assert back.elements.interfaces                    # really present


def test_writer_stamps_current_neutral_version():
    from tests.fixtures.schema import NEUTRAL_CURRENT
    assert NEUTRAL_SCHEMA_VERSION == NEUTRAL_CURRENT


def test_reads_prior_minor_file_without_interfaces_group(tmp_path):
    # ADR 0023's two-version window: an in-window 2.28.x file has no
    # /interfaces group at all and must still read → no interfaces.
    import h5py

    from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
    fem = _interface_fem()
    p = str(tmp_path / "old.h5")
    fem.to_h5(p)
    with h5py.File(p, "r+") as f:
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        del f["interfaces"]
    back = read_fem_h5(p)                              # in window → no raise
    assert back.elements.interfaces == []              # absent group ⇒ none


def test_composed_results_embed_path_accepts_interfaces(tmp_path):
    """The ADR 0020 ``/model/`` sub-group path (the other write entry
    point the S3 guard covered) now writes the group too."""
    import h5py

    from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group

    fem = _interface_fem()
    path = tmp_path / "embedded.h5"
    with h5py.File(str(path), "w") as f:
        write_neutral_zone_into_group(fem, f.create_group("model"))
    with h5py.File(str(path), "r") as f:
        assert "interfaces" in f["model"]
        assert f["model/interfaces/interfaces"].shape == (2,)


# ======================================================================
# Hand-built records: encode-side fail-loud hardening
# ======================================================================
def _minimal_fem(interfaces: list | None = None) -> FEMData:
    node_ids = np.array([1, 2], dtype=np.int64)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        interfaces=interfaces,
    )
    info = MeshInfo(n_nodes=2, n_elems=1, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


def _rec(**over) -> InterfaceRecord:
    base = dict(
        kind=ConstraintKind.INTERFACE,
        master_node=1,
        slave_node=2,
        backing_element=10,
        orient=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        a_trib=0.5,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
    )
    base.update(over)
    return InterfaceRecord(**base)


def test_hand_built_record_roundtrips(tmp_path):
    rec = _rec(
        name="hand",
        tangential_law=TangentialLaw(kind="elastic", k_per_area=2.0e7),
        phantom_node=900,
        phantom_coords=np.array([1.0, 2.0, 3.0]),
        phantom_ndf=2,
        equal_dof_records=[NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, name="hand",
            master_node=2, slave_node=900, dofs=[1, 2])],
    )
    back, _ = _roundtrip(_minimal_fem([rec]), tmp_path)
    _eq(back.elements.interfaces[0], rec)


def test_record_with_no_laws_roundtrips_as_none(tmp_path):
    # The "" kind sentinel must decode to None, not to a law object with
    # an empty kind (which NormalLaw.__post_init__ would refuse).
    rec = _rec(normal_law=None, tangential_law=None)
    back, _ = _roundtrip(_minimal_fem([rec]), tmp_path)
    got = back.elements.interfaces[0]
    assert got.normal_law is None and got.tangential_law is None


def test_encode_refuses_half_built_phantom_bridge():
    from apeGmsh.mesh._femdata_h5_io import _encode_interface
    with pytest.raises(ValueError, match="phantom bridge needs both"):
        _encode_interface(_rec(phantom_node=900, phantom_ndf=None,
                               phantom_coords=None))
    with pytest.raises(ValueError, match="all-or-nothing"):
        _encode_interface(_rec(phantom_node=None, phantom_ndf=2))


def test_encode_refuses_more_than_one_nested_equaldof():
    from apeGmsh.mesh._femdata_h5_io import _encode_interface
    eq = NodePairRecord(kind=ConstraintKind.EQUAL_DOF,
                        master_node=2, slave_node=900, dofs=[1, 2])
    with pytest.raises(ValueError, match="at most ONE"):
        _encode_interface(_rec(
            phantom_node=900, phantom_ndf=2,
            phantom_coords=np.array([0.0, 0.0, 0.0]),
            equal_dof_records=[eq, eq]))
