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
* partitioned (MPI) emit fails loud (the fork element can't be safely
  replicated across ranks like equalDOF was).
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


def test_partitioned_emit_fails_loud() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2], dofs=[],
    )
    with pytest.raises(NotImplementedError, match="partitioned"):
        _plan_rank_constraints(
            node_constraints=[rec],
            surface_constraints=None,
            partition_rank=0,
            node_owners={1: {0}, 2: {1}},
            element_owner={},
            phantom_tags=set(),
        )
