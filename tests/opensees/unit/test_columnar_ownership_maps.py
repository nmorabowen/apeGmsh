"""Direct unit battery for the B2+B3 columnar ownership containers.

ADR 0065 v2 / plan_emit_memory_columnar.md B2+B3 shipped three
array-backed classes duck-typed to the dicts they replaced
(`FemToOpsTagMap`, `SortedIntToInt`, `NodePartitionOwners`). The emit
behavior is locked by the byte-identity fixtures; this file locks the
MAPPING CONTRACT itself against a reference dict — including the
duplicate-eid tie-break, which the adversarial review found had silently
flipped from the old dict's last-wins to searchsorted first-wins
(review hardening: `_find`/`translate` now use ``side="right" - 1``).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees._internal.build import (
    ElementPlanRows,
    FemToOpsTagMap,
    LazyRankBuckets,
    MISSING_FEM_ELEMENT_ID,
    NodePartitionOwners,
    SortedIntSet,
    SortedIntToInt,
    _plan_owner_ranks,
    bucket_primary_nodes_by_rank,
    build_node_partition_owners,
    node_index_lookup,
)


# ---------------------------------------------------------------------------
# FemToOpsTagMap — parity vs a reference dict
# ---------------------------------------------------------------------------


def _reference_pairs() -> list[tuple[int, int]]:
    rng = np.random.default_rng(42)
    eids = rng.choice(10_000, size=300, replace=False)
    tags = np.arange(1, 301)
    return [(int(e), int(t)) for e, t in zip(eids, tags)]


def test_tag_map_matches_reference_dict() -> None:
    pairs = _reference_pairs()
    ref = dict(pairs)
    m = FemToOpsTagMap.from_pairs(pairs)

    assert len(m) == len(ref)
    assert bool(m) is True
    assert list(m.items()) == list(ref.items())
    assert list(m.keys()) == list(ref.keys())
    assert list(m.values()) == list(ref.values())
    for eid, tag in ref.items():
        assert eid in m
        assert m[eid] == tag
        assert m.get(eid) == tag
    assert 999_999 not in m
    assert m.get(999_999) is None
    assert m.get(999_999, -7) == -7
    with pytest.raises(KeyError):
        m[999_999]


def test_tag_map_empty_is_falsy_like_empty_dict() -> None:
    m = FemToOpsTagMap.from_pairs(())
    assert len(m) == 0
    assert not m
    assert m.get(1) is None
    assert list(m.items()) == []


def test_tag_map_duplicate_eid_is_last_wins_like_the_old_dict() -> None:
    """Overlapping element PGs legally fan one FEM cell from two specs.

    The old ``{eid: tag for ...}`` comprehension resolved the duplicate
    with LAST-wins; the review found searchsorted first-wins had crept
    in, silently retargeting recorder/damping/remove_element selections
    on such models. Lock the old semantics.
    """
    pairs = [(5, 11), (7, 100), (7, 205), (9, 33)]
    ref = dict(pairs)  # last-wins by construction
    m = FemToOpsTagMap.from_pairs(pairs)

    assert ref[7] == 205
    assert m[7] == 205
    assert m.get(7) == 205
    # Vectorised path must agree with the scalar path.
    out = m.translate(np.asarray([5, 7, 9, 8], dtype=np.int64))
    assert out.tolist() == [11, 205, 33, -1]
    # items() intentionally yields BOTH physical rows (plan order) —
    # the reverse ops_tag -> fem_eid map keys by unique tag.
    assert list(m.items()) == pairs


def test_tag_map_translate_matches_get_on_random_queries() -> None:
    pairs = _reference_pairs()
    m = FemToOpsTagMap.from_pairs(pairs)
    rng = np.random.default_rng(7)
    queries = rng.integers(0, 12_000, size=500, dtype=np.int64)
    out = m.translate(queries)
    for q, o in zip(queries.tolist(), out.tolist()):
        assert o == m.get(q, -1)


def test_tag_map_from_plan_drops_sentinel_and_derives_tags() -> None:
    rows_a = ElementPlanRows(
        np.asarray([10, 11, 12], dtype=np.int64),
        np.zeros((3, 4), dtype=np.int64),
        tag_start=100,
    )
    rows_pair = ElementPlanRows(
        np.asarray([MISSING_FEM_ELEMENT_ID], dtype=np.int64),
        np.zeros((1, 2), dtype=np.int64),
        tag_start=103,
    )
    m = FemToOpsTagMap.from_plan(
        [(object(), rows_a), (object(), rows_pair)]  # type: ignore[list-item]
    )
    assert dict(m.items()) == {10: 100, 11: 101, 12: 102}
    assert MISSING_FEM_ELEMENT_ID not in m


# ---------------------------------------------------------------------------
# SortedIntToInt
# ---------------------------------------------------------------------------


def test_sorted_int_to_int_matches_reference_dict() -> None:
    ref = {3: 0, 17: 2, 40: 1, 99: 2}
    keys = np.asarray(sorted(ref), dtype=np.int64)
    vals = np.asarray([ref[int(k)] for k in keys], dtype=np.int64)
    m = SortedIntToInt(keys, vals)

    assert len(m) == 4
    assert m == ref
    for k, v in ref.items():
        assert k in m
        assert m[k] == v
        assert m.get(k, -1) == v
    assert m.get(5) is None
    assert m.get(5, "sentinel") == "sentinel"  # default returned VERBATIM
    assert 5 not in m
    out = m.translate_ranks(np.asarray([3, 5, 99], dtype=np.int64))
    assert out.tolist() == [0, -1, 2]


# ---------------------------------------------------------------------------
# NodePartitionOwners
# ---------------------------------------------------------------------------


class _FakePartition:
    """Runtime rank = enumerate index (runtime_rank_from_partition_record)."""

    def __init__(self, node_ids: list[int]) -> None:
        self.node_ids = np.asarray(node_ids, dtype=np.int64)


class _FakeFem:
    def __init__(self, parts: "list[_FakePartition]") -> None:
        self.partitions = parts


def test_node_partition_owners_csr_matches_reference_sets() -> None:
    fem = _FakeFem([
        _FakePartition([1, 2, 3]),      # rank 0
        _FakePartition([3, 4]),         # rank 1
        _FakePartition([4, 5, 1]),      # rank 2
    ])
    owners = build_node_partition_owners(fem)  # type: ignore[arg-type]
    ref = {1: {0, 2}, 2: {0}, 3: {0, 1}, 4: {1, 2}, 5: {2}}

    assert isinstance(owners, NodePartitionOwners)
    assert len(owners) == len(ref)
    for nid, ranks in ref.items():
        got = owners.get(nid)
        assert isinstance(got, frozenset)
        assert got == ranks
    # Missing node: default coerced to frozenset (documented drift —
    # every consumer only reads membership/intersection).
    assert owners.get(999, set()) == frozenset()
    primary = owners.primary_owner()
    assert {int(k): int(v) for k, v in primary.items()} == {
        1: 0, 2: 0, 3: 0, 4: 1, 5: 2,
    }


# ---------------------------------------------------------------------------
# NodePartitionOwners.resolve_many — batched lookup
# ---------------------------------------------------------------------------
#
# `get` costs a SCALAR np.searchsorted per call (~1.4 us of numpy
# dispatch vs ~90 ns for the dict hash B2 replaced). The MP-constraint
# planner queries one node at a time in a Python loop, which made
# partitioned emit 2-3x slower, growing with rank count.
# `resolve_many` is the batched form; these lock it to `get`'s answers.


def _owners_fixture() -> "NodePartitionOwners":
    return build_node_partition_owners(  # type: ignore[arg-type,return-value]
        _FakeFem([
            _FakePartition([1, 2, 3]),      # rank 0
            _FakePartition([3, 4]),         # rank 1
            _FakePartition([4, 5, 1]),      # rank 2
        ]),
    )


def test_resolve_many_matches_get_including_misses_and_bounds() -> None:
    owners = _owners_fixture()
    # Below the first id, every real id, between ids, and past the last —
    # the past-the-end probe is the one that would index out of bounds
    # without the clip in resolve_many.
    query = [-5, 0, 1, 2, 3, 4, 5, 6, 999]
    got = owners.resolve_many(query)
    for nid in query:
        assert got.get(nid) == owners.get(nid), nid
    assert got[999] == frozenset()
    assert got[1] == {0, 2}


def test_resolve_many_caches_across_calls() -> None:
    owners = _owners_fixture()
    first = owners.resolve_many([1, 2, 3, 4, 5])
    # Second call for already-known ids returns the same cache and must
    # not disturb it — the planner re-runs per rank over the same nodes,
    # and a node's owners don't depend on which rank is asking.
    second = owners.resolve_many([1, 2, 3])
    assert second is first
    assert second[4] == {1, 2}

    # Ids resolved later join the same cache rather than replacing it.
    third = owners.resolve_many([999])
    assert third is first
    assert third[999] == frozenset()
    assert third[1] == {0, 2}


def test_resolve_many_interns_shared_owner_runs() -> None:
    # Nodes 1-3 are owned solely by rank 0, so they share one owner run;
    # node 4 is on both ranks. Interning is what keeps the cache at ~one
    # dict entry per node instead of one set OBJECT per node — the term
    # B2 set out to remove.
    owners = build_node_partition_owners(  # type: ignore[arg-type]
        _FakeFem([
            _FakePartition([1, 2, 3, 4]),   # rank 0
            _FakePartition([4]),            # rank 1
        ]),
    )
    got = owners.resolve_many([1, 2, 3, 4])
    assert got[1] is got[2]
    assert got[2] is got[3]
    assert got[1] == {0}
    assert got[4] == {0, 1}
    assert got[4] is not got[1]


# ---------------------------------------------------------------------------
# _plan_owner_ranks — batched element-owner lookup (B3 companion)
# ---------------------------------------------------------------------------
#
# The element fan-out probed `SortedIntToInt.get` once per element per
# rank — a scalar np.searchsorted apiece. `_plan_owner_ranks` resolves a
# whole plan in one `translate_ranks` call; the plain-dict fallback keeps
# unpartitioned / test callers working.


def _plan_rows() -> "ElementPlanRows":
    return ElementPlanRows(
        np.asarray([10, 11, 12, 13], dtype=np.int64),
        np.zeros((4, 4), dtype=np.int64),
        tag_start=100,
    )


def test_plan_owner_ranks_columnar_matches_scalar_get() -> None:
    rows = _plan_rows()
    # 12 is deliberately absent -> unowned sentinel -1.
    owner = SortedIntToInt(
        np.asarray([10, 11, 13], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
    )
    got = _plan_owner_ranks(rows, owner)
    assert got.tolist() == [0, 1, -1, 0]
    # Parity with the scalar form it replaced.
    assert got.tolist() == [
        owner.get(int(e), -1) for e in rows.eids
    ]


def test_plan_owner_ranks_plain_dict_fallback_agrees() -> None:
    rows = _plan_rows()
    as_dict = {10: 0, 11: 1, 13: 0}
    columnar = SortedIntToInt(
        np.asarray([10, 11, 13], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
    )
    # A plain dict has no translate_ranks and takes the scalar path; both
    # branches must produce the same answer.
    assert (
        _plan_owner_ranks(rows, as_dict).tolist()
        == _plan_owner_ranks(rows, columnar).tolist()
    )


def test_plan_owner_ranks_accepts_a_plain_tuple_plan() -> None:
    # The non-columnar plan shape has no `.eids`, so ids are gathered
    # from the triples instead.
    plan = [(10, (1, 2), 100), (12, (3, 4), 101), (11, (5, 6), 102)]
    owner = SortedIntToInt(
        np.asarray([10, 11, 13], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
    )
    assert _plan_owner_ranks(plan, owner).tolist() == [0, -1, 1]


def test_resolve_many_handles_empty_query_and_empty_map() -> None:
    owners = _owners_fixture()
    assert dict(owners.resolve_many([])) == {}

    empty = NodePartitionOwners(
        np.empty((0,), dtype=np.int64),
        np.zeros((1,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )
    got = empty.resolve_many([7, 8])
    assert got[7] == frozenset()
    assert got[8] == frozenset()
    assert empty.get(7) == frozenset()


# ---------------------------------------------------------------------------
# ADR 0100 P3 — node_index_lookup (D3), SortedIntSet (D2),
# LazyRankBuckets (D4), FemToOpsTagMap.inverse (R8)
# ---------------------------------------------------------------------------


def test_node_index_lookup_matches_reference_dict_on_unsorted_ids() -> None:
    """The broker's ``fem.nodes.ids`` is object-dtype and entity-grouped
    (UNSORTED on every partitioned mesh) — the argsort lookup must agree
    with the dict it replaces including the load-bearing miss -> None."""
    ids = np.asarray([104, 7, 9001, 55, 2, 300], dtype=object)
    ref = {int(nid): i for i, nid in enumerate(ids)}
    m = node_index_lookup(ids)

    assert len(m) == len(ref)
    for nid, idx in ref.items():
        assert m.get(nid) == idx
        assert m[nid] == idx
    assert m.get(9999) is None  # a miss must stay a miss
    with pytest.raises(KeyError):
        m[9999]


def test_node_index_lookup_empty_ids() -> None:
    m = node_index_lookup(np.asarray([], dtype=object))
    assert len(m) == 0
    assert m.get(1) is None


def test_node_index_lookup_duplicate_id_is_last_wins_like_the_old_dict() -> None:
    """Broker node ids are unique today — but "unique today" is what
    FemToOpsTagMap assumed before the review caught its first-wins
    flip.  The dict this replaces resolved a duplicate LAST-wins
    (later enumerate key overwrites); pin the same tie-break."""
    ids = np.asarray([7, 3, 7, 9], dtype=object)
    ref = {int(nid): i for i, nid in enumerate(ids)}
    assert ref[7] == 2  # last-wins by construction
    m = node_index_lookup(ids)
    assert m.get(7) == 2
    assert m[7] == 2
    assert m.get(3) == 1
    assert m.get(9) == 3
    assert len(m) == len(ref) == 3


def test_bucket_primary_nodes_seeds_a_zero_primary_rank() -> None:
    """M7 guard (refuter lens 1): a rank can own ONLY shared nodes
    whose primary owner is another rank.  The seed must still create
    that rank's (empty) entry — the per-rank pattern pass reads
    ``rank_primary_nodes[rank]`` unconditionally, so a dropped seed is
    a KeyError on every deck with such a rank."""
    primary_owner = SortedIntToInt(
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
    )
    out = bucket_primary_nodes_by_rank(primary_owner, [0, 1, 2])
    assert sorted(out) == [0, 1, 2]
    assert list(out[0]) == [1, 3]
    assert list(out[1]) == [2]
    assert list(out[2]) == []
    assert not out[2]
    assert len(out[2]) == 0


def test_sorted_int_set_membership_matches_reference_set() -> None:
    ref = {5, 2, 78, 41, 3}
    s = SortedIntSet.from_ids([5, 2, 78, 41, 3])
    assert len(s) == len(ref)
    assert bool(s)
    for v in range(100):
        assert (v in s) == (v in ref)

    empty = SortedIntSet.from_ids([])
    assert not empty
    assert len(empty) == 0
    assert 1 not in empty

    # ndarray input (the broker's per-rank node_ids arrive as arrays,
    # possibly unsigned); duplicates collapse like set() would.
    s2 = SortedIntSet.from_ids(np.asarray([9, 1, 9], dtype=np.uint64))
    assert sorted(s2) == [1, 9]


def test_sorted_int_set_intersection_and_isdisjoint_match_set_algebra() -> None:
    base = {10, 20, 30, 40}
    s = SortedIntSet.from_ids(sorted(base))
    other = {5, 30, 10, 99}

    assert s.intersection_sorted(other) == sorted(base & other)
    assert s.intersection_sorted(set()) == []
    assert SortedIntSet.from_ids([]).intersection_sorted(other) == []

    assert s.isdisjoint({1, 2})
    assert not s.isdisjoint({40})
    assert s.isdisjoint(set())
    assert SortedIntSet.from_ids([]).isdisjoint({1})


def _eager_buckets_reference(rows: ElementPlanRows, owner) -> dict:
    """The pre-P3 ``bucket_pre_allocated_by_rank`` semantics, inlined as
    the parity oracle: nonzero position order per rank, unowned rows
    dropped."""
    n = len(rows)
    if n == 0:
        return {}
    owners = _plan_owner_ranks(rows, owner)
    out = {}
    for r in np.unique(owners):
        if int(r) < 0:
            continue
        idx = np.nonzero(owners == r)[0]
        out[int(r)] = rows.select_rows(idx)
    return out


def test_lazy_rank_buckets_match_eager_reference_homogeneous() -> None:
    eids = np.arange(1, 11, dtype=np.int64)
    conn = np.arange(30, dtype=np.int64).reshape(10, 3)
    rows = ElementPlanRows(eids, conn, tag_start=50)
    # eid 10 has NO owner -> dropped from every bucket, both sides.
    owner = SortedIntToInt(
        np.arange(1, 10, dtype=np.int64),
        np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64),
    )

    lazy = LazyRankBuckets(rows, owner)
    ref = _eager_buckets_reference(rows, owner)

    assert sorted(ref) == [0, 1, 2]
    for rank, bucket in ref.items():
        assert lazy.count(rank) == len(bucket)
        got = lazy.get(rank)
        assert got is not None
        assert list(got) == list(bucket)  # (eid, conn_tuple, tag) triples
    assert lazy.count(99) == 0
    assert lazy.get(99) is None
    assert lazy.get(99, []) == []


def test_lazy_rank_buckets_mixed_npe_object_conn() -> None:
    """Mixed-npe plans carry object-dtype conn (per-row true-width
    arrays); the lazy buckets must route them identically to the eager
    select_rows copies — widths preserved, no padding leaked."""
    conn_rows = [(1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11)]
    conn = np.empty((4,), dtype=object)
    for i, c in enumerate(conn_rows):
        conn[i] = np.asarray(c, dtype=np.int64)
    rows = ElementPlanRows(
        np.asarray([21, 22, 23, 24], dtype=np.int64), conn, tag_start=7,
    )
    owner = SortedIntToInt(
        np.asarray([21, 22, 23, 24], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
    )

    lazy = LazyRankBuckets(rows, owner)
    ref = _eager_buckets_reference(rows, owner)
    for rank in (0, 1):
        assert list(lazy.get(rank)) == list(ref[rank])
    assert [c for _e, c, _t in lazy.get(0)] == [(1, 2), (6, 7, 8, 9)]


def test_lazy_rank_buckets_empty_plan() -> None:
    rows = ElementPlanRows(
        np.empty((0,), dtype=np.int64),
        np.empty((0, 0), dtype=np.int64),
        tag_start=1,
    )
    lazy = LazyRankBuckets(rows, SortedIntToInt(
        np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64),
    ))
    assert lazy.count(0) == 0
    assert lazy.get(0) is None
    assert lazy.get(0, []) == []


def test_tag_map_inverse_matches_reference_reverse_dict() -> None:
    """R8: the reverse map must reproduce
    ``{int(tag): int(eid) for eid, tag in m.items()}`` exactly —
    including the ``-1`` miss default its sole consumer relies on, and
    dict last-wins on a (pathological) duplicate tag."""
    pairs = [(5, 11), (7, 100), (7, 205), (9, 33)]
    m = FemToOpsTagMap.from_pairs(pairs)
    ref = {int(tag): int(eid) for eid, tag in m.items()}
    inv = m.inverse()
    for tag, eid in ref.items():
        assert inv.get(tag, -1) == eid
    assert inv.get(999, -1) == -1

    # Duplicate TAG (never produced by the allocator, but from_pairs
    # replay accepts it): the old dict comprehension kept the LAST row.
    dup = FemToOpsTagMap.from_pairs([(5, 50), (6, 50), (8, 60)])
    ref_dup = {int(t): int(e) for e, t in dup.items()}
    assert ref_dup[50] == 6
    inv_dup = dup.inverse()
    assert inv_dup.get(50, -1) == 6
    assert inv_dup.get(60, -1) == 8
    assert len(inv_dup) == 2
