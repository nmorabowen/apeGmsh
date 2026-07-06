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
    MISSING_FEM_ELEMENT_ID,
    NodePartitionOwners,
    SortedIntToInt,
    build_node_partition_owners,
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
