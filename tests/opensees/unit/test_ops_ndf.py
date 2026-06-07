"""ADR 0049 (DOF half) — ``ops.ndf`` overlay + gates G2/G3 + persistence.

Covers:

* :func:`resolve_ndf_overlay` — the stated-ndf overlay for element-less
  decoupled nodes, and its fail-loud guards (mesh target, element-touched
  target, unresolved handle).
* :func:`validate_constraint_master_ndf` (G2) — exact diaphragm-master ndf +
  per-DOF endpoint checks, broker and stage pools.
* :func:`validate_record_ndf_consistency` (G3) — mass/load exact size, fix /
  support mask length, sp DOF index.
* End-to-end: a real session decoupled node + ``ops.ndf`` emits the stated
  ``-ndf`` and persists it through ``/opensees/nodes_ndf`` (model_hash stable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees._internal.build import (
    BridgeError,
    FixRecord,
    MassRecord,
    NdfRecord,
    SupportRecord,
    resolve_ndf_overlay,
    validate_constraint_master_ndf,
    validate_record_ndf_consistency,
)
from apeGmsh.opensees.pattern.pattern import _LoadRecord, _SPRecord
from apeGmsh._kernel.records._constraints import (
    NodeGroupRecord,
    NodePairRecord,
)


# =====================================================================
# resolve_ndf_overlay — the stated-ndf overlay + its fail-loud guards
# =====================================================================

class _Nodes:
    def __init__(self, decoupled):
        self.decoupled_ids = np.asarray(decoupled, dtype=np.int64)


class _Fem:
    def __init__(self, decoupled):
        self.nodes = _Nodes(decoupled)


class _Handle:
    def __init__(self, tag):
        self.tag = tag


def test_overlay_sets_decoupled_ground_by_tag() -> None:
    fem = _Fem([100])
    ov = resolve_ndf_overlay(
        fem, [NdfRecord(handle=None, tag=100, ndf=6)], {1: 3, 2: 3}, 3,
    )
    assert ov == {100: 6}


def test_overlay_resolves_handle_to_tag() -> None:
    fem = _Fem([100])
    ov = resolve_ndf_overlay(
        fem, [NdfRecord(handle=_Handle(100), tag=None, ndf=3)], {}, 3,
    )
    assert ov == {100: 3}


def test_overlay_unresolved_handle_fails_loud() -> None:
    fem = _Fem([100])
    with pytest.raises(BridgeError, match="no resolved tag"):
        resolve_ndf_overlay(
            fem, [NdfRecord(handle=_Handle(None), tag=None, ndf=3)], {}, 3,
        )


def test_overlay_mesh_node_target_fails_loud() -> None:
    fem = _Fem([100])
    with pytest.raises(BridgeError, match="not a decoupled node"):
        resolve_ndf_overlay(
            fem, [NdfRecord(handle=None, tag=5, ndf=3)], {}, 3,
        )


def test_overlay_element_touched_target_fails_loud() -> None:
    # node 100 is decoupled BUT an element touches it (in *inferred*) — ops.ndf
    # may not restate an inferred ndf (no two-headed model).
    fem = _Fem([100])
    with pytest.raises(BridgeError, match="touched by an element"):
        resolve_ndf_overlay(
            fem, [NdfRecord(handle=None, tag=100, ndf=6)], {100: 3}, 3,
        )


def test_overlay_empty_when_no_records() -> None:
    assert resolve_ndf_overlay(_Fem([100]), [], {1: 3}, 3) == {}


# =====================================================================
# G2 — validate_constraint_master_ndf
# =====================================================================

class _NodesNC:
    def __init__(self, recs):
        # NodeConstraintSet is iterable over its raw records; a list suffices.
        self.constraints = list(recs)


class _FemNC:
    def __init__(self, recs):
        self.nodes = _NodesNC(recs)


def _dia(master, slaves=(2, 3)):
    return NodeGroupRecord(
        kind="rigid_diaphragm", master_node=master, slave_nodes=list(slaves),
        plane_normal=np.array([0.0, 0.0, 1.0]), dofs=None,
    )


def test_g2_diaphragm_master_wrong_ndf_3d_raises() -> None:
    with pytest.raises(BridgeError, match="EXACTLY ndf=6"):
        validate_constraint_master_ndf(_FemNC([_dia(1)]), {1: 3}, 3, 6)


def test_g2_diaphragm_master_ok_3d() -> None:
    validate_constraint_master_ndf(_FemNC([_dia(1)]), {1: 6}, 3, 6)


def test_g2_diaphragm_master_takes_envelope_when_absent() -> None:
    # master 1 absent from the map → envelope 6 → OK.
    validate_constraint_master_ndf(_FemNC([_dia(1)]), {}, 3, 6)


def test_g2_diaphragm_2d_needs_exactly_3() -> None:
    with pytest.raises(BridgeError, match="EXACTLY ndf=3"):
        validate_constraint_master_ndf(_FemNC([_dia(1)]), {1: 2}, 2, 2)


def test_g2_equaldof_dof_index_exceeds_slave_ndf_raises() -> None:
    rec = NodePairRecord(
        kind="equal_dof", master_node=1, slave_node=2, dofs=[1, 2, 3, 4],
    )
    with pytest.raises(BridgeError, match="references DOF 4"):
        validate_constraint_master_ndf(_FemNC([rec]), {1: 6, 2: 3}, 3, 6)


def test_g2_equaldof_within_ndf_ok() -> None:
    rec = NodePairRecord(
        kind="equal_dof", master_node=1, slave_node=2, dofs=[1, 2, 3],
    )
    validate_constraint_master_ndf(_FemNC([rec]), {1: 3, 2: 3}, 3, 3)


def test_g2_diaphragm_with_none_dofs_does_not_crash() -> None:
    # Regression: a rigid_diaphragm carries dofs=None; G2 must not call
    # expand_to_pairs / list(None) on it.
    validate_constraint_master_ndf(_FemNC([_dia(1)]), {1: 6}, 3, 6)


def test_g2_stage_constraint_master_checked() -> None:
    with pytest.raises(BridgeError, match="EXACTLY ndf=6"):
        validate_constraint_master_ndf(
            _FemNC([]), {1: 3}, 3, 6, stage_constraint_records=[_dia(1)],
        )


# =====================================================================
# G3 — validate_record_ndf_consistency
# =====================================================================

# Node-target records never touch fem; a bare object is enough.
_NOFEM = object()


def test_g3_mass_size_must_equal_ndf() -> None:
    rec = MassRecord(pg=None, nodes=(2,), values=(1.0, 1.0))
    with pytest.raises(BridgeError, match="Node::setMass"):
        validate_record_ndf_consistency(_NOFEM, {2: 6}, 3, 6, mass_records=[rec])


def test_g3_mass_exact_ok() -> None:
    rec = MassRecord(pg=None, nodes=(2,), values=(1.0, 1.0))
    validate_record_ndf_consistency(_NOFEM, {2: 2}, 2, 2, mass_records=[rec])


def test_g3_nodal_load_size_must_equal_ndf() -> None:
    rec = _LoadRecord(target_kind="node", target="2", forces=(1.0, 0.0, 0.0))
    with pytest.raises(BridgeError, match="addUnbalancedLoad"):
        validate_record_ndf_consistency(_NOFEM, {2: 6}, 3, 6, load_records=[rec])


def test_g3_fix_short_mask_ok_long_mask_raises() -> None:
    short = FixRecord(pg=None, nodes=(2,), dofs=(1, 1))
    validate_record_ndf_consistency(_NOFEM, {2: 6}, 3, 6, fix_records=[short])
    long = FixRecord(pg=None, nodes=(2,), dofs=(1, 1, 1, 1, 1, 1))
    with pytest.raises(BridgeError, match="addresses 6 DOFs"):
        validate_record_ndf_consistency(_NOFEM, {2: 3}, 3, 3, fix_records=[long])


def test_g3_sp_dof_index_must_fit_ndf() -> None:
    rec = _SPRecord(target_kind="node", target="2", dof=6, value=0.0)
    with pytest.raises(BridgeError, match="addresses DOF 6"):
        validate_record_ndf_consistency(_NOFEM, {2: 3}, 3, 3, sp_records=[rec])


def test_g3_support_mask_length_checked() -> None:
    rec = SupportRecord(pg=None, nodes=(2,), dofs=(1, 1, 1, 1, 1, 1))
    with pytest.raises(BridgeError, match="addresses 6 DOFs"):
        validate_record_ndf_consistency(
            _NOFEM, {2: 3}, 3, 3, support_records=[rec],
        )


# =====================================================================
# End-to-end — real session decoupled node + ops.ndf
# =====================================================================

def _box_with_decoupled(label="control"):
    """1×1×1 tet box (Body, ndf 3) + one element-less decoupled node."""
    g = apeGmsh(model_name="ndf_it")
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="Body")
    h = g.decouple_node(coords=(5.0, 5.0, 5.0), label=label)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    return g, fem, h


def _solid_ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    return ops


def test_ops_ndf_method_validates_args() -> None:
    g, fem, h = _box_with_decoupled()
    try:
        ops = _solid_ops(fem)
        with pytest.raises(ValueError):
            ops.ndf(ndf=3)               # no target
        with pytest.raises(ValueError):
            ops.ndf(h, ndf=0)            # non-positive ndf
    finally:
        g.end()


def test_ops_ndf_emits_stated_ndf_on_decoupled_node(tmp_path: Path) -> None:
    g, fem, h = _box_with_decoupled()
    try:
        ops = _solid_ops(fem)
        ops.ndf(h, ndf=6)               # element-less control node at ndf 6
        out = tmp_path / "m.tcl"
        ops.tcl(str(out))
        text = out.read_text(encoding="utf-8")
        node_line = next(
            ln for ln in text.splitlines()
            if ln.strip().startswith(f"node {h.tag} ")
        )
        assert "-ndf 6" in node_line, node_line
    finally:
        g.end()


def test_ops_ndf_on_mesh_node_fails_loud(tmp_path: Path) -> None:
    g, fem, h = _box_with_decoupled()
    try:
        ops = _solid_ops(fem)
        mesh_tag = next(int(t) for t in fem.nodes.ids if int(t) != h.tag)
        ops.ndf(mesh_tag, ndf=6)        # a tet node — inferred, not decoupled
        with pytest.raises(BridgeError):
            ops.tcl(str(tmp_path / "x.tcl"))
    finally:
        g.end()


def test_ops_ndf_persists_through_h5(tmp_path: Path) -> None:
    g, fem, h = _box_with_decoupled()
    try:
        ops = _solid_ops(fem)
        ops.ndf(h, ndf=6)
        out = tmp_path / "m.h5"
        ops.h5(str(out))                # single write — no round-trip
        with h5_reader.open(str(out)) as model:
            nn = model.nodes_ndf()
            assert nn is not None
            assert nn.get(int(h.tag)) == 6
    finally:
        g.end()
