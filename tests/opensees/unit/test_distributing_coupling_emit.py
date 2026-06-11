"""Emission contract for RBE3 / distributing coupling.

``g.constraints.distributing_coupling`` resolves to an
:class:`InterpolationRecord` with ``kind="distributing"`` and emits the
Ladruno-fork ``element LadrunoDistributingCoupling`` (class tag 33011) —
the reference (dependent) node R in ``slave_node`` and the independents in
``master_nodes``. These tests lock the kind-branching emit shared by the
serial and partitioned surface-coupling passes:

* a ``distributing`` record → ``element LadrunoDistributingCoupling $tag
  $R $N $i1..iN`` (no ``-w`` when weights are uniform/None; ``-w`` when set),
* the independent count is arbitrary (the 3/4-Rnode embedded guard does
  NOT apply),
* a ``tie`` / ``embedded`` record still routes to ``embeddedNode``.
"""
from __future__ import annotations

import numpy as np

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh.opensees._internal.build import _emit_one_interpolation
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


def _emit(rec: InterpolationRecord) -> RecordingEmitter:
    e = RecordingEmitter()
    _emit_one_interpolation(e, rec, TagAllocator())
    return e


def test_distributing_emits_fork_element_uniform_no_w_flag() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4, 5],
        weights=None,
    )
    calls = [c for c in _emit(rec).calls if c[0] == "element"]
    assert len(calls) == 1
    flat = calls[0][1]
    assert flat[0] == "LadrunoDistributingCoupling"
    # flat = (token, ele_tag, refNode, N, i1, i2, i3, i4)
    assert flat[2:] == (1, 4, 2, 3, 4, 5)     # R=1, N=4, independents 2..5
    assert "-w" not in flat                    # uniform ⇒ no -w


def test_distributing_emits_w_flag_when_weights_set() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=10, master_nodes=[20, 21],
        weights=np.array([2.0, 1.0]),
    )
    flat = [c for c in _emit(rec).calls if c[0] == "element"][0][1]
    assert flat[0] == "LadrunoDistributingCoupling"
    assert flat[2:] == (10, 2, 20, 21, "-w", 2.0, 1.0)


def test_distributing_allows_two_independents_no_rnode_guard() -> None:
    # N=2 would trip the ASDEmbeddedNodeElement 3/4-Rnode guard; the
    # distributing branch must NOT apply it.
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3], weights=None,
    )
    flat = [c for c in _emit(rec).calls if c[0] == "element"][0][1]
    assert flat[0] == "LadrunoDistributingCoupling"
    assert flat[2:] == (1, 2, 2, 3)


def test_name_comment_precedes_the_element() -> None:
    rec = InterpolationRecord(
        kind="distributing", name="anchor", slave_node=1,
        master_nodes=[2, 3, 4], weights=None,
    )
    kinds = [c[0] for c in _emit(rec).calls]
    assert kinds == ["mp_constraint_comment", "element"]


def test_embedded_record_still_routes_to_embeddedNode() -> None:
    # A 3-node tie/embedded interpolation must still emit ASDEmbeddedNodeElement.
    rec = InterpolationRecord(
        kind="embedded", slave_node=7, master_nodes=[8, 9, 10],
        weights=np.array([0.3, 0.3, 0.4]),
    )
    kinds = [c[0] for c in _emit(rec).calls]
    assert "embeddedNode" in kinds
    assert "element" not in kinds
