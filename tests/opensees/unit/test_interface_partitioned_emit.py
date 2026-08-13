"""Owner resolution + tag pre-pass for partitioned ``g.constraints.
interface()`` (ADR 0093 S8 / INV-5).

Pure-kernel tests of the two pieces the per-rank emit consumes:

* :func:`_plan_rank_interfaces` — owner = the rank holding the record's
  stamped ``backing_element``, asserted to live in EXACTLY ONE
  partition's element set (counted directly across the partitions, NOT
  read off ``build_element_partition_owner``'s first-seen dedup); the
  master node asserted native to the owner; the real slave node ghosted
  when foreign; the phantom never foreign.
* :func:`allocate_interface_tags` — every record's
  ``(normal_mat, tangential_mat, element)`` tag triple drawn in flat
  side-list order, so a record's tags do not depend on which rank owns
  it and 1-rank / N-rank decks stay byte-comparable (ADR 0027).

Why this exists at all: an interface pair's two nodes are CO-LOCATED,
so a partition cut hugging the interface replicates both onto both
ranks — the exact all-shared case ADR 0092's node tally refuses as
undecidable. The backing element carries the locality the nodes cannot,
and ADR 0092 measured what happens without a single owner: duplicate
emission converges to a plausible WRONG answer (double stiffness, half
penetration) while base reactions still balance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NodePairRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees._internal.build import (
    BridgeError,
    _plan_rank_interfaces,
    allocate_interface_tags,
)
from apeGmsh.opensees._internal.tag_allocator import TagAllocator

ENT_LAW = NormalLaw(kind="ent", k_per_area=1.0e6)
EPP_LAW = TangentialLaw(kind="epp", k_per_area=1.0e5, tau_b=250.0)
ORIENT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class _Part:
    """The three attributes the planner reads off a PartitionRecord."""

    id: int
    node_ids: tuple[int, ...]
    element_ids: tuple[int, ...]


def _rec(
    master: int, slave: int, *,
    backing: int,
    phantom: int | None = None,
    name: str | None = None,
) -> InterfaceRecord:
    equal_dofs = []
    if phantom is not None:
        equal_dofs = [NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, name=name,
            master_node=slave, slave_node=phantom, dofs=[1, 2],
        )]
    return InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        name=name,
        master_node=master,
        slave_node=slave,
        backing_element=backing,
        orient=ORIENT,
        a_trib=0.25,
        normal_law=ENT_LAW,
        tangential_law=EPP_LAW,
        phantom_node=phantom,
        phantom_coords=(np.asarray((1.0, 0.5, 0.0), dtype=float)
                        if phantom is not None else None),
        phantom_ndf=2 if phantom is not None else None,
        equal_dof_records=equal_dofs,
    )


# =====================================================================
# Owner = the backing element's rank (element-side, INV-5)
# =====================================================================
def test_owner_is_the_rank_holding_the_backing_element():
    parts = [
        _Part(id=1, node_ids=(1, 2, 3), element_ids=(10, 11)),
        _Part(id=2, node_ids=(4, 5, 6), element_ids=(20, 21)),
    ]
    plan = _plan_rank_interfaces(
        [_rec(5, 6, backing=20)], parts,
    )
    assert set(plan) == {1}                       # runtime rank 1 (part id 2)
    (rec, ghosts), = plan[1]
    assert int(rec.master_node) == 5 and ghosts == ()


def test_no_plan_when_no_records():
    assert _plan_rank_interfaces([], [_Part(1, (1,), (10,))]) == {}


def test_all_shared_cut_owner_is_still_exact():
    # The degenerate case the verb was designed around: BOTH pair nodes
    # replicated onto BOTH ranks (the cut hugs the co-located
    # interface), so an ADR 0092-style node tally ties 2-vs-2 and is
    # undecidable. The backing element decides — exactly, with no tie,
    # no heuristic.
    parts = [
        _Part(id=1, node_ids=(1, 2, 3, 4), element_ids=(10,)),
        _Part(id=2, node_ids=(1, 2, 3, 4), element_ids=(20,)),
    ]
    rec = _rec(1, 2, backing=20)
    plan = _plan_rank_interfaces([rec], parts)
    assert set(plan) == {1}
    (_, ghosts), = plan[1]
    # The slave is replicated onto the owner as well — nothing foreign.
    assert ghosts == ()


# =====================================================================
# Ghost set: the foreign real slave, and ONLY it
# =====================================================================
def test_foreign_slave_joins_the_ghost_set():
    parts = [
        _Part(id=1, node_ids=(1, 2), element_ids=(10,)),
        _Part(id=2, node_ids=(3, 4), element_ids=(20,)),
    ]
    plan = _plan_rank_interfaces([_rec(1, 3, backing=10)], parts)
    (_, ghosts), = plan[0]
    assert ghosts == (3,)


def test_native_slave_is_not_ghosted():
    parts = [
        _Part(id=1, node_ids=(1, 2, 3), element_ids=(10,)),
        _Part(id=2, node_ids=(4, 5), element_ids=(20,)),
    ]
    plan = _plan_rank_interfaces([_rec(1, 3, backing=10)], parts)
    (_, ghosts), = plan[0]
    assert ghosts == ()


def test_phantom_is_never_foreign():
    # A mixed-ndf pair's phantom is minted BY the owner rank at emit
    # time — it exists on exactly one rank, appears in no partition's
    # node set, and must never enter the ghost set (only the real
    # slave/beam node can be foreign).
    parts = [
        _Part(id=1, node_ids=(1, 2), element_ids=(10,)),
        _Part(id=2, node_ids=(3, 4), element_ids=(20,)),
    ]
    plan = _plan_rank_interfaces(
        [_rec(1, 3, backing=10, phantom=101)], parts,
    )
    (rec, ghosts), = plan[0]
    assert ghosts == (3,)
    assert 101 not in ghosts


# =====================================================================
# INV-5 preconditions — loud, named, never first-seen
# =====================================================================
def test_backing_element_in_two_partitions_refuses_naming_both():
    # The exact hole INV-5's assertion exists for:
    # build_element_partition_owner documents a FIRST-SEEN tiebreak for
    # duplicated elements, and _extract_partitions genuinely duplicates
    # boundary-entity elements. The planner must count membership
    # directly and refuse — not silently take the first-seen rank.
    parts = [
        _Part(id=1, node_ids=(1, 2), element_ids=(10, 30)),
        _Part(id=2, node_ids=(3, 4), element_ids=(20, 30)),
    ]
    with pytest.raises(BridgeError) as exc:
        _plan_rank_interfaces(
            [_rec(1, 3, backing=30, name="RockLiner")], parts,
        )
    msg = str(exc.value)
    assert "ADR 0093 INV-5" in msg
    assert "'RockLiner'" in msg
    assert "backing element 30" in msg
    assert "[0, 1]" in msg                        # both holding ranks named
    assert "never silently first-seen" in msg


def test_backing_element_in_no_partition_refuses():
    parts = [
        _Part(id=1, node_ids=(1, 2), element_ids=(10,)),
        _Part(id=2, node_ids=(3, 4), element_ids=(20,)),
    ]
    with pytest.raises(BridgeError) as exc:
        _plan_rank_interfaces([_rec(1, 3, backing=99)], parts)
    msg = str(exc.value)
    assert "ADR 0093 INV-5" in msg
    assert "NO partition's element set" in msg


def test_master_not_native_to_the_owner_refuses():
    # extract_partitions puts an element's nodes in its partition's node
    # set, so the master is native to the backing rank by construction —
    # asserted anyway, loudly, so a drift in extraction semantics cannot
    # silently ghost the master onto its own owner.
    parts = [
        _Part(id=1, node_ids=(2,), element_ids=(10,)),   # master 1 missing
        _Part(id=2, node_ids=(1, 3), element_ids=(20,)),
    ]
    with pytest.raises(BridgeError) as exc:
        _plan_rank_interfaces([_rec(1, 3, backing=10)], parts)
    msg = str(exc.value)
    assert "ADR 0093 INV-5" in msg
    assert "does not natively own the master node 1" in msg


def test_violation_names_the_record_position_without_a_name():
    # A nameless record is still identified — by its 1-based side-list
    # position and its node pair.
    parts = [
        _Part(id=1, node_ids=(1,), element_ids=(10,)),
        _Part(id=2, node_ids=(3,), element_ids=(10,)),
    ]
    with pytest.raises(BridgeError, match=r"record #1 \(master=1, slave=3\)"):
        _plan_rank_interfaces([_rec(1, 3, backing=10)], parts)


# =====================================================================
# Plan shape — order preserved, deterministic
# =====================================================================
def test_records_keep_flat_side_list_order_within_a_rank():
    parts = [
        _Part(id=1, node_ids=(1, 2, 3, 4), element_ids=(10, 11)),
        _Part(id=2, node_ids=(5, 6, 7, 8), element_ids=(20, 21)),
    ]
    recs = [
        _rec(1, 2, backing=10),
        _rec(5, 6, backing=20),
        _rec(3, 4, backing=11),
        _rec(7, 8, backing=21),
    ]
    plan = _plan_rank_interfaces(recs, parts)
    assert [r.master_node for r, _ in plan[0]] == [1, 3]
    assert [r.master_node for r, _ in plan[1]] == [5, 7]


def test_same_input_same_output():
    parts = [
        _Part(id=1, node_ids=(1, 2), element_ids=(10,)),
        _Part(id=2, node_ids=(3, 4), element_ids=(20,)),
    ]
    recs = [_rec(1, 3, backing=10), _rec(3, 1, backing=20)]
    first = _plan_rank_interfaces(recs, parts)
    second = _plan_rank_interfaces(recs, parts)
    assert first == second


# =====================================================================
# Tag pre-pass — flat order, per-record triples, shared counters
# =====================================================================
def test_allocate_interface_tags_walks_records_in_order():
    recs = [_rec(1, 3, backing=10), _rec(2, 4, backing=11)]
    tags = TagAllocator()
    plan = allocate_interface_tags(recs, tags)
    # Per record: normal mat, tangential mat (uniaxialMaterial counter),
    # element — the same consumption sequence the inline allocation
    # used, so flat-path tag values are unchanged.
    assert plan[id(recs[0])] == (1, 2, 1)
    assert plan[id(recs[1])] == (3, 4, 2)


def test_allocate_interface_tags_continues_the_shared_namespaces():
    recs = [_rec(1, 3, backing=10)]
    tags = TagAllocator()
    for _ in range(5):
        tags.allocate("element")
    for _ in range(3):
        tags.allocate("uniaxialMaterial")
    plan = allocate_interface_tags(recs, tags)
    assert plan[id(recs[0])] == (4, 5, 6)


def test_allocate_interface_tags_is_rank_independent_by_construction():
    # The whole point of the pre-pass: the triple is keyed by record,
    # allocated before any rank fan-out — so consuming the SAME plan
    # from any rank's block yields the same tags. Two allocations of
    # the same record list from fresh allocators agree.
    recs = [_rec(1, 3, backing=10), _rec(2, 4, backing=20)]
    a = allocate_interface_tags(recs, TagAllocator())
    b = allocate_interface_tags(recs, TagAllocator())
    assert [a[id(r)] for r in recs] == [b[id(r)] for r in recs]
