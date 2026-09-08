"""ADR 0049 — node-pair zeroLength form (``ops.element.ZeroLength(nodes=)``).

The node-pair form wires a single zeroLength-family spring to two explicit
endpoints — at least one typically a ``g.decouple_node`` ground — without a
meshed 2-node "line" physical group.  Covers:

* dual-mode construction validation (pg XOR nodes; ZeroLengthSection forbids
  nodes=);
* :func:`resolve_element_node_pair` endpoint resolution (handle / label /
  int) + the ``i != j`` distinctness guard;
* the G1 equal-endpoint gate firing on a mismatched node-pair spring;
* end-to-end emit (tcl carries ``iNode jNode``) for a spring to a decoupled
  ground sized by ``ops.ndf``;
* H5 round-trip (``inline_connectivity`` persistence — connectivity survives
  ``to_h5 → from_h5 → re-emit`` and ``model_hash`` is stable);
* the partitioned-emit fail-loud guard.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import h5py

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees, ModelData
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.opensees._internal.build import (
    BridgeError,
    resolve_element_node_pair,
)
from apeGmsh.opensees.element.zero_length import (
    CoupledZeroLength,
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
from apeGmsh.opensees.element.two_node_link import TwoNodeLink
from apeGmsh._kernel.defs.decoupled import DecoupledNodeDef


# =====================================================================
# Construction-time dual-mode validation (no FEM needed)
# =====================================================================

_MAT = object()
_MD = (ZeroLengthMatDir(material=_MAT, dof=1),)


def _handle(tag):
    return DecoupledNodeDef(coords=(0.0, 0.0, 0.0), label="gnd", tag=tag)


@pytest.mark.parametrize("cls,kw", [
    (ZeroLength, dict(mat_dirs=_MD)),
    (CoupledZeroLength, dict(material=_MAT, dir1=1, dir2=2)),
    (TwoNodeLink, dict(mat_dirs=_MD)),
])
def test_pg_xor_nodes_required(cls, kw) -> None:
    h = _handle(1)
    # both -> error
    with pytest.raises(ValueError, match="exactly one of pg"):
        cls(pg="x", nodes=(h, "lbl"), **kw)
    # neither -> error
    with pytest.raises(ValueError, match="exactly one of pg"):
        cls(**kw)
    # pg only -> ok
    assert cls(pg="x", **kw).nodes is None
    # nodes only -> ok
    assert cls(nodes=(h, "lbl"), **kw).pg is None


def test_nodes_must_be_pair() -> None:
    with pytest.raises(ValueError, match="2-tuple"):
        ZeroLength(nodes=(_handle(1),), mat_dirs=_MD)  # type: ignore[arg-type]


def test_nodes_reject_bad_ref_type() -> None:
    with pytest.raises(ValueError, match="must be a g.decouple_node"):
        ZeroLength(nodes=(_handle(1), 3.5), mat_dirs=_MD)  # type: ignore[arg-type]


def test_zerolengthsection_forbids_nodes() -> None:
    with pytest.raises(ValueError, match="does not support the node-pair"):
        ZeroLengthSection(nodes=(_handle(1), _handle(2)), section=_MAT)


# =====================================================================
# resolve_element_node_pair — endpoint resolution + i != j guard
# =====================================================================

class _Labels:
    def __init__(self, mapping):
        self._m = mapping

    def node_ids(self, name):
        if name not in self._m:
            raise KeyError(name)
        return self._m[name]


class _Nodes:
    def __init__(self, labels):
        self.labels = _Labels(labels)


class _Fem:
    def __init__(self, labels):
        self.nodes = _Nodes(labels)


def test_resolve_handle_and_label() -> None:
    fem = _Fem({"boundary": (7,)})
    spec = ZeroLength(nodes=(_handle(42), "boundary"), mat_dirs=_MD)
    assert resolve_element_node_pair(fem, spec) == (42, 7)


def test_resolve_int_escape_hatch() -> None:
    fem = _Fem({})
    spec = ZeroLength(nodes=(42, 7), mat_dirs=_MD)
    assert resolve_element_node_pair(fem, spec) == (42, 7)


def test_resolve_unmaterialized_handle_fails_loud() -> None:
    fem = _Fem({})
    spec = ZeroLength(nodes=(_handle(None), 7), mat_dirs=_MD)
    with pytest.raises(BridgeError, match="no resolved tag"):
        resolve_element_node_pair(fem, spec)


def test_resolve_label_multi_node_fails_loud() -> None:
    fem = _Fem({"face": (1, 2, 3)})
    spec = ZeroLength(nodes=(_handle(9), "face"), mat_dirs=_MD)
    with pytest.raises(BridgeError, match="EXACTLY one node"):
        resolve_element_node_pair(fem, spec)


@pytest.mark.parametrize("nodes", [
    (42, 42),                       # int == int
    (_handle(5), 5),                # handle == int
])
def test_resolve_same_node_fails_loud(nodes) -> None:
    fem = _Fem({"lbl": (5,)})
    spec = ZeroLength(nodes=nodes, mat_dirs=_MD)
    with pytest.raises(BridgeError, match="DISTINCT nodes"):
        resolve_element_node_pair(fem, spec)


def test_resolve_label_int_collision_fails_loud() -> None:
    fem = _Fem({"lbl": (5,)})
    spec = ZeroLength(nodes=("lbl", 5), mat_dirs=_MD)
    with pytest.raises(BridgeError, match="DISTINCT nodes"):
        resolve_element_node_pair(fem, spec)


# =====================================================================
# End-to-end — real session, decoupled ground spring
# =====================================================================

def _box_with_ground(label="pile_ground"):
    """1×1×1 tet box (Body, ndf 3) + one decoupled ground node."""
    g = apeGmsh(model_name="np_zl")
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="Body")
    h = g.decouple_node(coords=(0.0, 0.0, 0.0), label=label)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    return g, fem, h


def _spring_ops(fem, h, *, ground_ndf=3):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    k = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    # structural endpoint: a real tet node (int escape hatch); ground: handle.
    mesh_tag = int(fem.nodes.ids[0])
    ops.element.ZeroLength(
        nodes=(mesh_tag, h),
        mat_dirs=(ZeroLengthMatDir(material=k, dof=1),
                  ZeroLengthMatDir(material=k, dof=2),
                  ZeroLengthMatDir(material=k, dof=3)),
    )
    ops.ndf(h, ndf=ground_ndf)
    return ops, mesh_tag


def test_node_pair_spring_emits_inode_jnode(tmp_path: Path) -> None:
    g, fem, h = _box_with_ground()
    try:
        ops, mesh_tag = _spring_ops(fem, h, ground_ndf=3)
        out = tmp_path / "m.tcl"
        ops.tcl(str(out))
        text = out.read_text(encoding="utf-8")
        zl = [ln for ln in text.splitlines()
              if ln.strip().startswith("element zeroLength ")]
        assert len(zl) == 1, text
        toks = zl[0].split()
        # element zeroLength <tag> <iNode> <jNode> -mat ...
        assert int(toks[3]) == mesh_tag
        assert int(toks[4]) == int(h.tag)
        # ground emitted as a node at ndf 3 (equals envelope -> token elided ok)
        assert any(
            ln.strip().startswith(f"node {h.tag} ")
            for ln in text.splitlines()
        )
    finally:
        g.end()


def test_node_pair_g1_fires_on_ndf_mismatch(tmp_path: Path) -> None:
    g, fem, h = _box_with_ground()
    try:
        # ground stated at ndf 2 while the tet structural end infers 3 ->
        # G1 (equal-endpoint gate) must raise: one end below 3 is outside
        # the fork's 3D relaxation at any build.
        ops, _ = _spring_ops(fem, h, ground_ndf=2)
        with pytest.raises(BridgeError, match="differing"):
            ops.tcl(str(tmp_path / "x.tcl"))
    finally:
        g.end()


def test_node_pair_g1_allows_a_3d_pair_with_both_ends_at_least_three():
    """Fork #808 / ADR 96: in 3D a zeroLength takes any pair whose ends
    both carry ndf >= 3, acting on DOFs 1-3 with the rest untouched
    passengers. A ground stated at ndf 4 against a 3-dof tet end is that
    case — G1 used to refuse it, and refusing it now would block the
    u-p interfaces TIMs A10 exists for."""
    g, fem, h = _box_with_ground()
    try:
        ops, _ = _spring_ops(fem, h, ground_ndf=4)
        ops.build()          # G1 runs at build; no raise is the assertion
    finally:
        g.end()


def test_node_pair_partitioned_fails_loud() -> None:
    """A node-pair element in a partitioned (len(partitions)>1) model has no
    fem eid for build_element_partition_owner; v1 fails loud rather than
    silently dropping the spring on every rank (ADR 0049 D6)."""
    from typing import cast

    from apeGmsh.opensees.emitter.tcl import TclEmitter
    from tests.opensees.fixtures.fem_stub import (
        make_two_column_frame_partitioned,
    )

    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    k = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    ops.element.ZeroLength(
        nodes=(1, 2), mat_dirs=(ZeroLengthMatDir(material=k, dof=1),),
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match="partitioned"):
        bm.emit(TclEmitter())


def _model_hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta/lineage"].attrs["model_hash"])


def test_node_pair_inline_connectivity_persisted(tmp_path: Path) -> None:
    g, fem, h = _box_with_ground()
    try:
        ops, mesh_tag = _spring_ops(fem, h, ground_ndf=3)
        out = tmp_path / "m.h5"
        ops.h5(str(out))
        with h5py.File(str(out), "r") as f:
            grp = f["opensees/element_meta/zeroLength"]
            assert "inline_connectivity" in grp
            fem_eids = list(grp["fem_eids"][...])
            row = next(i for i, e in enumerate(fem_eids) if int(e) < 0)
            conn = set(int(c) for c in grp["inline_connectivity"][row])
            assert conn == {mesh_tag, int(h.tag)}
    finally:
        g.end()


def test_node_pair_reader_reconstructs_connectivity(tmp_path: Path) -> None:
    g, fem, h = _box_with_ground()
    try:
        ops, mesh_tag = _spring_ops(fem, h, ground_ndf=3)
        out = tmp_path / "m.h5"
        ops.h5(str(out))
        model = OpenSeesModel.from_h5(str(out))
        recs = [r for r in model.elements() if r.type_token == "zeroLength"]
        assert len(recs) == 1
        assert set(recs[0].connectivity) == {mesh_tag, int(h.tag)}
        # args carry the endpoint prefix again (valid re-emit shape).
        assert set(recs[0].args[:2]) == {mesh_tag, int(h.tag)}
    finally:
        g.end()


def test_node_pair_h5_round_trip_byte_stable(tmp_path: Path) -> None:
    g, fem, h = _box_with_ground()
    try:
        ops, mesh_tag = _spring_ops(fem, h, ground_ndf=3)
        a = tmp_path / "a.h5"
        ops.h5(str(a))
        # H5 -> H5 via the bridge persistence path (ModelData).  The first
        # re-write normalises the apeSees-writer zone into the ModelData
        # layout (the two writers differ pre-existingly, unrelated to ADR
        # 0049 — so a vs b hashes legitimately differ); subsequent re-writes
        # must be byte-stable, which is what locks the node-pair connectivity
        # persistence.
        b, c = tmp_path / "b.h5", tmp_path / "c.h5"
        ModelData.from_h5(str(a)).write(str(b))
        ModelData.from_h5(str(b)).write(str(c))
        assert _model_hash(b) == _model_hash(c)

        # inline_connectivity dataset round-trips with the same endpoints.
        for p in (a, b, c):
            with h5py.File(str(p), "r") as f:
                gz = f["opensees/element_meta/zeroLength"]
                r = next(i for i, e in enumerate(gz["fem_eids"][...])
                         if int(e) < 0)
                conn = set(int(x) for x in gz["inline_connectivity"][r])
                assert conn == {mesh_tag, int(h.tag)}
    finally:
        g.end()
