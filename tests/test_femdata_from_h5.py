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

from tests.fixtures.schema import NEUTRAL_CURRENT


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


def test_round_trip_embedded_stiffness_and_flags(tmp_path: Path) -> None:
    """Schema 2.8.0 — ASDEmbeddedNodeElement options on
    InterpolationRecord round-trip through H5.

    Pre-2.8.0 the kernel ``interpolation_payload_dtype`` lacked
    ``stiffness`` / ``stiffness_p`` / ``rotational`` / ``pressure``
    / ``excess``, so an ``embedded`` record with a non-default ``-K``
    silently snapped back to the dataclass default (``1e18``) on
    read — and the bridge then emitted ``-K 1e18`` unconditionally
    (ADR 0035 made ``-K`` always emit).  This locks the fix.
    """
    embedded = InterpolationRecord(
        kind=ConstraintKind.EMBEDDED,
        name="embed_pile",
        slave_node=5,
        master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]),
        dofs=[1, 2, 3],
        projected_point=np.array([0.6, 0.6, 0.0]),
        parametric_coords=np.array([0.3, 0.3]),
        excess=0.0042,
        stiffness=5.0e16,
        stiffness_p=2.5e15,
        rotational=True,
        pressure=True,
    )
    # And a tied_contact-shaped coupling whose slave_records carry
    # non-default options too, exercising the sr_* CSR lane in
    # surface_coupling_payload_dtype.
    sr_a = InterpolationRecord(
        kind=ConstraintKind.TIE, slave_node=6, master_nodes=[2, 3],
        weights=np.array([0.5, 0.5]), dofs=[1, 2, 3],
        projected_point=np.array([0.5, 0.0, 0.0]),
        parametric_coords=np.array([0.5, 0.0]),
        stiffness=7.0e12, stiffness_p=None,
        rotational=False, pressure=True, excess=None,
    )
    sr_b = InterpolationRecord(
        kind=ConstraintKind.TIE, slave_node=7, master_nodes=[2, 3],
        weights=np.array([0.25, 0.75]), dofs=[1, 2, 3],
        projected_point=np.array([0.75, 0.0, 0.0]),
        parametric_coords=np.array([0.75, 0.0]),
        stiffness=3.0e13, stiffness_p=1.5e12,
        rotational=True, pressure=False, excess=1.1e-3,
    )
    coupling = SurfaceCouplingRecord(
        kind=ConstraintKind.TIED_CONTACT, name="tie_face",
        master_nodes=[2, 3], slave_nodes=[6, 7], dofs=[1, 2, 3],
        mortar_operator=None, slave_records=[sr_a, sr_b],
    )

    node_ids = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], [0.5, 0.5, 1.0], [2.0, 0.5, 0.5],
        [2.0, 0.5, 1.5],
    ], dtype=np.float64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=1,
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([20], dtype=np.int64),
        connectivity=np.array([[2, 3, 5]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        constraints=[embedded, coupling],
    )
    info = MeshInfo(n_nodes=7, n_elems=1, bandwidth=0, types=[tri_info])
    fem = FEMData(nodes=nodes, elements=elements, info=info)

    out = tmp_path / "rt_embed.h5"
    fem.to_h5(str(out), model_name="rt_embed")
    rebuilt = FEMData.from_h5(str(out))

    rebuilt_embeds = [
        r for r in rebuilt.elements.constraints
        if isinstance(r, InterpolationRecord)
        and r.kind == ConstraintKind.EMBEDDED
    ]
    assert len(rebuilt_embeds) == 1
    e = rebuilt_embeds[0]
    assert e.stiffness == pytest.approx(5.0e16)
    assert e.stiffness_p == pytest.approx(2.5e15)
    assert e.rotational is True
    assert e.pressure is True
    assert e.excess == pytest.approx(0.0042)

    coups = list(rebuilt.elements.constraints.couplings())
    assert len(coups) == 1
    srs = coups[0].slave_records
    assert len(srs) == 2
    by_node = {r.slave_node: r for r in srs}
    a = by_node[6]
    assert a.stiffness == pytest.approx(7.0e12)
    assert a.stiffness_p is None
    assert a.rotational is False
    assert a.pressure is True
    assert a.excess is None
    b = by_node[7]
    assert b.stiffness == pytest.approx(3.0e13)
    assert b.stiffness_p == pytest.approx(1.5e12)
    assert b.rotational is True
    assert b.pressure is False
    assert b.excess == pytest.approx(1.1e-3)


def test_pre_2_8_0_interpolation_payload_decodes_with_defaults(
    tmp_path: Path,
) -> None:
    """Backward compat: an InterpolationRecord written with the
    pre-2.8.0 payload dtype (no stiffness/stiffness_p/rotational/
    pressure/excess columns) decodes to the dataclass defaults,
    instead of raising on the missing fields.

    The reader probes ``p.dtype.names`` (same structural-detection
    pattern as ``_opt_name`` and the original ``sr_*`` fallback) and
    falls back when the columns are absent.  Synthesised by
    rewriting the ``/constraints/embedded`` dataset with the older
    field layout after a normal write.
    """
    embedded = InterpolationRecord(
        kind=ConstraintKind.EMBEDDED, name="x",
        slave_node=5, master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]),
        dofs=[1, 2, 3],
        projected_point=np.array([0.6, 0.6, 0.0]),
        parametric_coords=np.array([0.3, 0.3]),
        stiffness=5.0e16, stiffness_p=2.5e15,
        rotational=True, pressure=True, excess=0.0042,
    )
    nodes = NodeComposite(
        node_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        node_coords=np.zeros((5, 3), dtype=np.float64),
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=1,
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([20], dtype=np.int64),
        connectivity=np.array([[2, 3, 5]], dtype=np.int64),
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        constraints=[embedded],
    )
    fem = FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=5, n_elems=1, bandwidth=0, types=[tri_info]),
    )

    out = tmp_path / "pre_2_8_0.h5"
    fem.to_h5(str(out))

    # Re-write /constraints/embedded with the pre-2.8.0 payload dtype
    # (just the 7 legacy columns: slave_node, master_nodes, weights,
    # dofs, projected_point, parametric_coords, name).  Mirrors what
    # any file from a 2.7.0 writer would look like.  Built with the
    # public h5py vlen/string helpers — no need to reach into the
    # private ``_record_h5`` helpers.
    from apeGmsh.mesh._record_h5 import make_record_dtype

    utf8 = h5py.string_dtype(encoding="utf-8")
    vlen_i64 = h5py.special_dtype(vlen=np.dtype("int64"))
    vlen_f64 = h5py.special_dtype(vlen=np.dtype("float64"))
    legacy_payload = np.dtype([
        ("slave_node", np.int64),
        ("master_nodes", vlen_i64),
        ("weights", vlen_f64),
        ("dofs", vlen_i64),
        ("projected_point", np.float64, (3,)),
        ("parametric_coords", np.float64, (2,)),
        ("name", utf8),
    ])
    legacy_outer = make_record_dtype(legacy_payload)
    row = np.empty(1, dtype=legacy_outer)
    row[0] = (
        "node", "5", "embedded",
        (
            5,
            np.asarray([2, 3, 4], dtype=np.int64),
            np.asarray([0.3, 0.3, 0.4], dtype=np.float64),
            np.asarray([1, 2, 3], dtype=np.int64),
            (0.6, 0.6, 0.0),
            (0.3, 0.3),
            "x",
        ),
    )
    with h5py.File(str(out), "r+") as f:
        del f["constraints/embedded"]
        f["constraints"].create_dataset("embedded", data=row)
        # Down-stamp the neutral schema version into the prior minor
        # so the two-version reader window still accepts the file.
        # Prior-minor floats with the current schema bump: 2.9.0 →
        # window {2.8.x, 2.9.x}; the test only needs the legacy 7-column
        # InterpolationRecord payload (NOT the 13-column 2.8.0 dtype) to
        # exercise the structural-fallback decoder, and that fallback
        # is keyed on payload-field presence, not on the schema string.
        f["meta"].attrs["schema_version"] = "2.8.0"
        f["meta"].attrs["neutral_schema_version"] = "2.8.0"

    rebuilt = FEMData.from_h5(str(out))
    es = [
        r for r in rebuilt.elements.constraints
        if isinstance(r, InterpolationRecord)
        and r.kind == ConstraintKind.EMBEDDED
    ]
    assert len(es) == 1
    e = es[0]
    # Legacy fields survived.
    assert e.slave_node == 5
    assert e.master_nodes == [2, 3, 4]
    # Missing columns -> InterpolationRecord defaults (NOT the writer's
    # original 5e16 / 2.5e15 / True / True / 0.0042 — that data is
    # gone, but the reader doesn't crash).
    assert e.stiffness == pytest.approx(1.0e18)
    assert e.stiffness_p is None
    assert e.rotational is False
    assert e.pressure is False
    assert e.excess is None


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


def test_loads_zone_with_missing_subgroups(tmp_path: Path) -> None:
    """``/loads/`` may exist with only some of ``{nodal,element,sp}``.

    Per memory ``project_h5py_optional_child_get_hazard``, the reader
    must probe optional sub-groups with ``name in parent`` (H5Lexists),
    not ``Group.get(name)``.  This test removes the ``element`` and
    ``sp`` sub-groups from a freshly written file, then asserts
    ``from_h5`` loads cleanly — no ``AttributeError`` from a ``None``
    returned by ``Group.get``, no spurious entries.
    """
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))

    # Excise the element + sp sub-groups, leaving /loads/nodal/ alone.
    with h5py.File(str(out), "r+") as f:
        assert "loads" in f
        assert "nodal" in f["loads"]
        assert "element" in f["loads"]
        assert "sp" in f["loads"]
        del f["loads/element"]
        del f["loads/sp"]

    rebuilt = FEMData.from_h5(str(out))

    # Nodal loads survive untouched; element + sp collections are empty.
    assert len(list(rebuilt.nodes.loads)) == 2
    assert len(list(rebuilt.elements.loads)) == 0
    assert len(list(rebuilt.nodes.sp)) == 0


def test_round_trip_masses(tmp_path: Path) -> None:
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    assert len(rebuilt.nodes.masses) == 2
    m4 = rebuilt.nodes.masses.by_node(4)
    assert m4 is not None
    assert m4.mass == (100.0, 100.0, 100.0, 0.0, 0.0, 0.0)


# =====================================================================
# Phase 6 / ADR 0021 — INV-1: ``fem.snapshot_id`` byte-identical to
# the stamped ``/meta/lineage/fem_hash``.
# =====================================================================


def test_from_h5_lineage_fem_hash_matches_snapshot_id(tmp_path: Path) -> None:
    """The reloaded ``FEMData.snapshot_id`` matches the stamped
    ``/meta/lineage/fem_hash``.

    Phase 6 (ADR 0021) INV-1: the lineage chain's ``fem_hash`` value
    is byte-identical to today's ``snapshot_id`` semantics for the
    same FEMData.  Both paths converge.
    """
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))
    with h5py.File(str(out), "r") as f:
        assert "meta" in f
        assert "lineage" in f["meta"]
        stamp = f["meta/lineage"].attrs["fem_hash"]
        if isinstance(stamp, bytes):
            stamp = stamp.decode("utf-8", "replace")
    assert rebuilt.snapshot_id == str(stamp)
    # And matches the original fem too.
    assert rebuilt.snapshot_id == fem.snapshot_id


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

    # ADR 0023 — SchemaVersionError message includes the file's version
    # and the reader's supported range; major mismatch surfaces as
    # "different major" with the upgrade-path text.
    with pytest.raises(SchemaVersionError, match="different major"):
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


# =====================================================================
# Phase 2 — B2: partitions + parts round-trip
# =====================================================================


def _make_partitioned_fem() -> FEMData:
    """Hand-built FEMData with two partitions and one Part label."""
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
    part_node_map = {"slab_A": {1, 2}, "slab_B": {3, 4}}
    part_elem_map = {"slab_A": {10}, "slab_B": {11}}

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        partitions=partitions, part_node_map=part_node_map,
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        partitions=partitions, part_elem_map=part_elem_map,
    )
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=2, types=[tri_info]),
    )


def test_round_trip_partitions(tmp_path: Path) -> None:
    """Partitions round-trip through to_h5 / from_h5.

    Audit gap B2: without persistence, ``fem.partitions`` returns []
    on reload and ``select(partition=k)`` raises ``KeyError``.
    """
    fem = _make_partitioned_fem()
    out = tmp_path / "partitions.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    # P2: ``fem.partitions`` is a :class:`PartitionSet` — check ids.
    assert rebuilt.partitions.ids == [1, 2]
    assert rebuilt.nodes.partitions == [1, 2]
    assert rebuilt.elements.partitions == [1, 2]

    sel = rebuilt.nodes.select(partition=1)
    np.testing.assert_array_equal(
        sorted(int(x) for x in sel.ids), [1, 2, 3],
    )
    sel_e = rebuilt.elements.select(partition=1)
    np.testing.assert_array_equal(
        sorted(int(x) for x in sel_e.ids), [10],
    )


def test_round_trip_parts(tmp_path: Path) -> None:
    """Part labels round-trip and ``select(target=part_label)`` resolves.

    Audit gap B2 (parts side): ``select(target=part_label)`` raised
    ``KeyError`` after reload because the part maps weren't
    persisted.
    """
    fem = _make_partitioned_fem()
    out = tmp_path / "parts.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))

    nodes_A = rebuilt.nodes.select(target="slab_A")
    np.testing.assert_array_equal(
        sorted(int(x) for x in nodes_A.ids), [1, 2],
    )
    elems_A = rebuilt.elements.select(target="slab_A")
    np.testing.assert_array_equal(
        sorted(int(x) for x in elems_A.ids), [10],
    )

    nodes_B = rebuilt.nodes.select(target="slab_B")
    np.testing.assert_array_equal(
        sorted(int(x) for x in nodes_B.ids), [3, 4],
    )
    elems_B = rebuilt.elements.select(target="slab_B")
    np.testing.assert_array_equal(
        sorted(int(x) for x in elems_B.ids), [11],
    )


# =====================================================================
# Phase 2 — B3: bandwidth recomputed on read
# =====================================================================


def test_bandwidth_recomputed_on_read(tmp_path: Path) -> None:
    """``info.bandwidth`` is recomputed from connectivity on reload.

    Audit gap B3: pre-v2.5.0 reader hardcoded ``bandwidth=0``.  The
    writer never stored it; the new reader derives it
    deterministically from the reloaded per-type groups.
    """
    fem = _make_full_fem()
    # The fixture's bandwidth value is fed to MeshInfo as a constant;
    # the real bandwidth derived from the connectivity is the one
    # _compute_bandwidth returns.  Compute the expected value the
    # same way from_gmsh would.
    from apeGmsh.mesh.FEMData import _compute_bandwidth
    groups = {g.element_type.code: g for g in fem.elements}
    expected = _compute_bandwidth(groups)
    assert expected > 0  # sanity: the fixture has non-trivial connectivity

    out = tmp_path / "bw.h5"
    fem.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))
    assert rebuilt.info.bandwidth == expected


# =====================================================================
# Phase 2 — B4: snapshot_id verified on read
# =====================================================================


def test_snapshot_id_verified_on_read(tmp_path: Path) -> None:
    """A tampered neutral zone raises ``MalformedH5Error`` on read.

    Audit gap B4 / ADR 0021: FEM round-trip integrity is a hard
    guarantee.  ``/meta/snapshot_id`` is now verified against the
    recomputed hash of the rebuilt FEM; mismatch → raise.
    """
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    fem = _make_full_fem()
    out = tmp_path / "tamper.h5"
    fem.to_h5(str(out))

    # Tamper directly with one node's coords.  The recomputed
    # snapshot_id will not match the stored one.
    with h5py.File(out, "r+") as f:
        coords = f["nodes/coords"][...]
        coords[0] = coords[0] + np.array([1.0, 0.0, 0.0])
        f["nodes/coords"][...] = coords

    with pytest.raises(MalformedH5Error, match="snapshot_id mismatch"):
        FEMData.from_h5(str(out))


# =====================================================================
# Phase 2 — backcompat: 2.4.0 files (no name / partitions / parts)
# =====================================================================


def _make_legacy_2_4_0_h5(path: Path) -> None:
    """Write a synthetic 2.4.0 file with the OLD payload dtype (no name)."""
    import h5py
    from apeGmsh.mesh._record_h5 import make_record_dtype

    old_dt = make_record_dtype(np.dtype([
        ("master_node", np.int64),
        ("slave_node", np.int64),
        ("dofs", h5py.vlen_dtype(np.dtype(np.int64))),
        ("offset", np.float64, (3,)),
        ("penalty_stiffness", np.float64),
    ]))

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        # Per ADR 0023 fixture must be inside the two-version reader
        # window (2.6.x / 2.7.x); the test exercises legacy
        # empty-snapshot_id semantics, not pre-window file handling.
        meta.attrs["schema_version"] = NEUTRAL_CURRENT
        meta.attrs["apeGmsh_version"] = ""
        meta.attrs["created_iso"] = "2025-01-01T00:00:00+00:00"
        meta.attrs["ndm"] = 1
        meta.attrs["ndf"] = 0
        # Empty snapshot_id → reader skips verification (legacy semantic)
        meta.attrs["snapshot_id"] = ""
        meta.attrs["model_name"] = "legacy"

        nodes = f.create_group("nodes")
        nodes.create_dataset("ids", data=np.array([1, 2], dtype=np.int64))
        nodes.create_dataset(
            "coords",
            data=np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float64),
        )
        f.create_group("elements")

        rows = np.empty(1, dtype=old_dt)
        rows[0] = (
            "node", "2", "equal_dof",
            (1, 2, np.array([1, 2, 3], dtype=np.int64),
             (float("nan"),) * 3, float("nan")),
        )
        f.create_group("constraints").create_dataset(
            "equal_dof", data=rows,
        )


def test_legacy_2_4_0_file_reads_without_name(tmp_path: Path) -> None:
    """A 2.4.0 file with the OLD payload dtype still reads.

    Per ADR 0023's two-version window, the 2.5.0 reader accepts
    2.4.x files: missing ``name`` field → decoded as ``None``;
    absent ``/partitions/`` / ``/parts/`` groups → unpartitioned
    FEM with no part maps.
    """
    out = tmp_path / "legacy_2_4_0.h5"
    _make_legacy_2_4_0_h5(out)
    rebuilt = FEMData.from_h5(str(out))

    assert len(rebuilt.nodes.constraints) == 1
    rec = rebuilt.nodes.constraints[0]
    assert rec.name is None
    assert rec.master_node == 1
    assert rec.slave_node == 2
    assert rec.dofs == [1, 2, 3]

    # No partitions and no parts on a legacy file.
    assert len(rebuilt.partitions) == 0


# =====================================================================
# Phase 2 — B7: end-to-end parity (rebuilt vs original)
# =====================================================================


def _assert_fem_equivalent(rebuilt: FEMData, original: FEMData) -> None:
    """Field-level equivalence between ``from_h5(to_h5(fem))`` and ``fem``.

    Covers every surface the writer touches: nodes, elements
    (per-type), info, all constraint kinds (including ``name``),
    all load kinds, masses, mesh_selections, partitions / parts,
    PG / label membership.
    """
    # Nodes ────────────────────────────────────────────────────
    np.testing.assert_array_equal(
        np.asarray(rebuilt.nodes.ids, dtype=np.int64),
        np.asarray(original.nodes.ids, dtype=np.int64),
    )
    np.testing.assert_allclose(rebuilt.nodes.coords, original.nodes.coords)

    # Elements per type ───────────────────────────────────────
    rb_groups = {g.element_type.code: g for g in rebuilt.elements}
    or_groups = {g.element_type.code: g for g in original.elements}
    assert set(rb_groups) == set(or_groups)
    for code, g_o in or_groups.items():
        g_r = rb_groups[code]
        np.testing.assert_array_equal(g_r.ids, g_o.ids)
        np.testing.assert_array_equal(g_r.connectivity, g_o.connectivity)

    # MeshInfo ────────────────────────────────────────────────
    assert rebuilt.info.n_nodes == original.info.n_nodes
    assert rebuilt.info.n_elems == original.info.n_elems
    assert rebuilt.info.bandwidth == original.info.bandwidth
    rb_type_names = sorted(t.name for t in rebuilt.info.types)
    or_type_names = sorted(t.name for t in original.info.types)
    assert rb_type_names == or_type_names

    # Constraints — both node-side and element-side, by kind ───
    def _records_by_kind(seq):
        d: dict[str, list] = {}
        for r in seq:
            d.setdefault(r.kind, []).append(r)
        return d

    rb_nc = _records_by_kind(rebuilt.nodes.constraints)
    or_nc = _records_by_kind(original.nodes.constraints)
    assert set(rb_nc) == set(or_nc)
    for k in or_nc:
        assert len(rb_nc[k]) == len(or_nc[k]), f"node constraint {k}: count differs"
        # Field-by-field equality including ``name``.
        for r_r, r_o in zip(rb_nc[k], or_nc[k]):
            assert r_r.name == r_o.name, f"{k}: name {r_r.name!r} != {r_o.name!r}"

    rb_ec = _records_by_kind(rebuilt.elements.constraints)
    or_ec = _records_by_kind(original.elements.constraints)
    assert set(rb_ec) == set(or_ec)
    for k in or_ec:
        assert len(rb_ec[k]) == len(or_ec[k])
        for r_r, r_o in zip(rb_ec[k], or_ec[k]):
            assert r_r.name == r_o.name

    # Loads ───────────────────────────────────────────────────
    assert sorted(rebuilt.nodes.loads.patterns()) == sorted(
        original.nodes.loads.patterns())
    for pat in original.nodes.loads.patterns():
        rb = rebuilt.nodes.loads.by_pattern(pat)
        org = original.nodes.loads.by_pattern(pat)
        assert len(rb) == len(org)
        for r_r, r_o in zip(rb, org):
            assert r_r.node_id == r_o.node_id
            assert r_r.name == r_o.name

    assert sorted(rebuilt.elements.loads.patterns()) == sorted(
        original.elements.loads.patterns())
    for pat in original.elements.loads.patterns():
        rb = rebuilt.elements.loads.by_pattern(pat)
        org = original.elements.loads.by_pattern(pat)
        assert len(rb) == len(org)
        for r_r, r_o in zip(rb, org):
            assert r_r.element_id == r_o.element_id
            assert r_r.name == r_o.name

    # SP records ──────────────────────────────────────────────
    rb_sp = list(rebuilt.nodes.sp)
    or_sp = list(original.nodes.sp)
    assert len(rb_sp) == len(or_sp)
    for r_r, r_o in zip(rb_sp, or_sp):
        assert r_r.node_id == r_o.node_id
        assert r_r.dof == r_o.dof
        assert r_r.name == r_o.name

    # Masses ──────────────────────────────────────────────────
    rb_m = list(rebuilt.nodes.masses)
    or_m = list(original.nodes.masses)
    assert len(rb_m) == len(or_m)
    for r_r, r_o in zip(rb_m, or_m):
        assert r_r.node_id == r_o.node_id
        assert r_r.name == r_o.name

    # PG / Label memberships ──────────────────────────────────
    assert list(sorted(rebuilt.nodes.physical.get_all())) == list(
        sorted(original.nodes.physical.get_all()))
    assert list(sorted(rebuilt.nodes.labels.get_all())) == list(
        sorted(original.nodes.labels.get_all()))

    # Partitions ──────────────────────────────────────────────
    # P2: compare the integer-id lists; ``fem.partitions`` is a
    # :class:`PartitionSet` so equality is on ``.ids``.
    assert rebuilt.partitions.ids == original.partitions.ids
    for pid in original.partitions.ids:
        rb_ids = sorted(int(x) for x in rebuilt.nodes.select(partition=pid).ids)
        or_ids = sorted(int(x) for x in original.nodes.select(partition=pid).ids)
        assert rb_ids == or_ids

    # Snapshot ID is the strongest single contract — if everything
    # above matches, this should match too.
    assert rebuilt.snapshot_id == original.snapshot_id


def _fixture_simple_frame() -> FEMData:
    """3D frame: nodes + line2 elements + a fixed-base SP."""
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 3.0, 3.0],
    ], dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=np.array([10, 11], dtype=np.int64),
        connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
    )
    sp_base = SPRecord(
        node_id=1, dof=1, value=0.0, is_homogeneous=True,
        pattern="default", name="base_fix",
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        sp=[sp_base],
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=1, types=[line_info]),
    )


def _fixture_plate() -> FEMData:
    """2D plate: triangle3 mesh + an element pressure load."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=2,
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([10, 11], dtype=np.int64),
        connectivity=np.array([[1, 2, 3], [1, 3, 4]], dtype=np.int64),
    )
    el_load = ElementLoadRecord(
        element_id=10, load_type="surfacePressure",
        params={"pressure": -100.0, "direction": "normal"},
        pattern="dead", name="self_weight",
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        loads=[el_load],
    )
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=3, types=[tri_info]),
    )


def _fixture_mixed_dim() -> FEMData:
    """Mixed-dim mesh: line2 + triangle3 + a tie constraint."""
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ], dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 5]], dtype=np.int64),
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([20], dtype=np.int64),
        connectivity=np.array([[2, 3, 4]], dtype=np.int64),
    )
    tie = InterpolationRecord(
        kind=ConstraintKind.TIE, name="beam_to_plate",
        slave_node=5, master_nodes=[2, 3, 4],
        weights=np.array([0.34, 0.33, 0.33]),
        dofs=[1, 2, 3],
        projected_point=np.array([0.6, 0.6, 0.0]),
        parametric_coords=np.array([0.3, 0.3]),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group, 2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        constraints=[tie],
    )
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(
            n_nodes=5, n_elems=2, bandwidth=4,
            types=[line_info, tri_info],
        ),
    )


def _fixture_partitioned() -> FEMData:
    """Partitioned plate — wraps the shared partitioned fixture."""
    return _make_partitioned_fem()


def _fixture_mesh_selection() -> FEMData:
    """FEM with mesh_selections carried through round-trip."""
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore

    fem = _fixture_plate()
    fem.mesh_selection = MeshSelectionStore({
        (2, 1): {
            "name": "plate_face",
            "node_ids": np.array([1, 2, 3, 4], dtype=np.int64),
            "node_coords": fem.nodes.coords,
            "element_ids": np.array([10, 11], dtype=np.int64),
            "connectivity": np.array(
                [[1, 2, 3], [1, 3, 4]], dtype=np.int64),
        },
    })
    return fem


@pytest.mark.parametrize(
    "fixture_id, builder",
    [
        ("simple_frame", _fixture_simple_frame),
        ("plate", _fixture_plate),
        ("mixed_dim", _fixture_mixed_dim),
        ("partitioned", _fixture_partitioned),
        ("mesh_selection", _fixture_mesh_selection),
    ],
)
def test_round_trip_join_equivalent(
    fixture_id: str, builder, tmp_path: Path,
) -> None:
    """End-to-end: ``from_h5(to_h5(fem))`` is field-level equivalent to ``fem``.

    Audit gap B7 (the meta-gap): every prior round-trip test
    compared the rebuilt FEMData against a hand-built fixture.  No
    test compared rebuilt vs the live ``from_gmsh(g)`` it came
    from.  This parametrized parity test covers 5 canonical
    fixtures so any future "writer drops X" or "reader skips Y"
    regression surfaces structurally.
    """
    original = builder()
    out = tmp_path / f"rt_{fixture_id}.h5"
    original.to_h5(str(out))
    rebuilt = FEMData.from_h5(str(out))
    _assert_fem_equivalent(rebuilt, original)


# =====================================================================
# Parameterized sub-group root (ADR 0020 / Phase 4 cleanup)
# =====================================================================


def test_from_h5_with_root_kwarg(tmp_path: Path) -> None:
    """``FEMData.from_h5(path, root='/embedded/model')`` reads the rich
    neutral zone from a sub-group root.

    Phase 4 cleanup (ADR 0020) — composed ``results.h5`` files embed
    the FEMData rich neutral zone under ``/model/`` rather than at the
    file root.  The ``root=`` kwarg lets one reader handle both
    layouts.  Verified by writing the SAME ``write_fem_h5`` content
    under a custom sub-group root and asserting field-level
    equivalence to a standalone (``root="/"``) version.
    """
    from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group

    fem = _make_full_fem()
    nested = tmp_path / "nested.h5"
    with h5py.File(nested, "w") as f:
        sub = f.create_group("embedded/model")
        write_neutral_zone_into_group(
            fem, sub, model_name="nested_demo",
        )

    rebuilt = FEMData.from_h5(str(nested), root="/embedded/model")

    standalone = tmp_path / "standalone.h5"
    fem.to_h5(str(standalone), model_name="nested_demo")
    rebuilt_root = FEMData.from_h5(str(standalone))

    _assert_fem_equivalent(rebuilt, rebuilt_root)


def test_from_h5_default_root_backcompat(tmp_path: Path) -> None:
    """``FEMData.from_h5(path)`` (no ``root=``) keeps byte-identical
    behaviour to the pre-Phase-4 reader.

    Existing call sites must work unchanged — the parameterisation
    is additive.
    """
    fem = _make_full_fem()
    out = tmp_path / "rt.h5"
    fem.to_h5(str(out), model_name="rt")

    # Both calls reach the same internal reader path; the default
    # is ``root="/"``.
    a = FEMData.from_h5(str(out))
    b = FEMData.from_h5(str(out), root="/")
    _assert_fem_equivalent(a, b)
