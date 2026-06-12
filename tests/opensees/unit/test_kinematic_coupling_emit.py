"""Emission contract for RBE2 / kinematic coupling.

``g.constraints.kinematic_coupling`` now emits the Ladruno-fork
``element LadrunoKinematicCoupling`` (class tag 33012) — a penalty
rigid-body driver with moment-arm transport — instead of the old
``equalDOF``-per-slave expansion (which ignored the lever arm and was
correct only for coincident nodes). These tests lock the new emission:

* one ``element LadrunoKinematicCoupling $tag $ref $N $s1..sN`` per record,
* ``-dof`` omitted when the record ties "all" (empty dofs), emitted when
  the user restricts the component list,
* non-kinematic ``NodeGroupRecord`` rows (rigid_diaphragm / rigid_body)
  are NOT touched by this pass,
* partitioned (MPI) emit routes through the SINGLE canonical rank
  (min-of-intersection of the slave owners, mirroring the
  ASDEmbeddedNodeElement rule) with the reference node ghost-declared
  when foreign; a slave set split across partitions fails loud (the
  fork element can't be replicated across ranks like equalDOF was —
  N rank-allocated tags would N-fold over-constrain).
"""
from __future__ import annotations

import pytest

from apeGmsh._kernel.records._constraints import NodeGroupRecord
from apeGmsh.opensees._internal.build import (
    _emit_kinematic_couplings,
    _plan_rank_constraints,
)
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


def _emit(records: list[NodeGroupRecord]) -> RecordingEmitter:
    e = RecordingEmitter()
    _emit_kinematic_couplings(e, records, TagAllocator())
    return e


def test_emits_fork_element_default_ties_all_no_dof_flag() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
    )
    calls = [c for c in _emit([rec]).calls if c[0] == "element"]
    assert len(calls) == 1
    flat = calls[0][1]
    assert flat[0] == "LadrunoKinematicCoupling"
    # flat = (token, ele_tag, refNode, N, s1, s2)
    assert flat[2:] == (1, 2, 2, 3)          # ref=1, N=2, slaves 2 & 3
    assert "-dof" not in flat                  # empty dofs ⇒ no -dof flag


def test_emits_dof_flag_when_restricted() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=10, slave_nodes=[20],
        dofs=[1, 2, 3],
    )
    flat = [c for c in _emit([rec]).calls if c[0] == "element"][0][1]
    assert flat[0] == "LadrunoKinematicCoupling"
    assert flat[2:] == (10, 1, 20, "-dof", 1, 2, 3)   # ref, N=1, slave, -dof


def test_each_record_gets_a_distinct_element_tag() -> None:
    recs = [
        NodeGroupRecord(kind="kinematic_coupling", master_node=1,
                        slave_nodes=[2], dofs=[]),
        NodeGroupRecord(kind="kinematic_coupling", master_node=3,
                        slave_nodes=[4], dofs=[]),
    ]
    calls = [c[1] for c in _emit(recs).calls if c[0] == "element"]
    assert calls[0][1] != calls[1][1]        # ele_tag differs


def test_name_comment_precedes_the_element() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", name="platen", master_node=1,
        slave_nodes=[2], dofs=[],
    )
    kinds = [c[0] for c in _emit([rec]).calls]
    assert kinds == ["mp_constraint_comment", "element"]


def test_non_kinematic_node_group_records_are_skipped() -> None:
    # rigid_diaphragm / rigid_body ride NodeGroupRecord too but are
    # emitted by their own passes — this one must ignore them.
    recs = [
        NodeGroupRecord(kind="rigid_diaphragm", master_node=1,
                        slave_nodes=[2, 3], dofs=[1, 2, 3]),
        NodeGroupRecord(kind="rigid_body", master_node=4,
                        slave_nodes=[5], dofs=[1, 2, 3, 4, 5, 6]),
    ]
    assert [c for c in _emit(recs).calls if c[0] == "element"] == []


def _plan(rec: NodeGroupRecord, rank: int, node_owners: dict[int, set[int]]):
    return _plan_rank_constraints(
        node_constraints=[rec],
        surface_constraints=None,
        partition_rank=rank,
        node_owners=node_owners,
        element_owner={},
        phantom_tags=set(),
    )


def test_partitioned_emits_on_single_canonical_rank_with_ghost_ref() -> None:
    # ref node 1 lives on rank 0; both slaves live on rank 1 ⇒ the
    # canonical rank is 1 (the only rank with every slave present) and
    # the ref is ghost-declared there. Rank 0 emits nothing.
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
    )
    owners = {1: {0}, 2: {1}, 3: {1}}
    plan0 = _plan(rec, 0, owners)
    plan1 = _plan(rec, 1, owners)
    assert not plan0.any()
    assert plan1.allowed_record_ids == frozenset({id(rec)})
    assert plan1.foreign_node_tags == frozenset({1})   # ghost ref


def test_partitioned_canonical_rank_is_min_when_shared() -> None:
    # Slaves boundary-shared on both ranks ⇒ min(intersection) = 0
    # emits; rank 1 stays silent (no double element).
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
    )
    owners = {1: {0}, 2: {0, 1}, 3: {0, 1}}
    plan0 = _plan(rec, 0, owners)
    plan1 = _plan(rec, 1, owners)
    assert plan0.allowed_record_ids == frozenset({id(rec)})
    assert plan0.foreign_node_tags == frozenset()      # all local on rank 0
    assert not plan1.any()


def test_partitioned_split_slaves_fail_loud() -> None:
    # Slaves split across ranks with no common rank ⇒ the single
    # element cannot be assembled anywhere — fail loud on EVERY rank.
    rec = NodeGroupRecord(
        kind="kinematic_coupling", name="lid",
        master_node=1, slave_nodes=[2, 3], dofs=[],
    )
    owners = {1: {0}, 2: {0}, 3: {1}}
    for rank in (0, 1):
        with pytest.raises(ValueError, match="'lid'.*split across partitions"):
            _plan(rec, rank, owners)
