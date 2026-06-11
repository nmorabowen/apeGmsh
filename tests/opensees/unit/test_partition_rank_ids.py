"""Unit tests for the 1-based gmsh ``PartitionRecord.id`` → 0-based
OpenSeesMP runtime-rank seam.

Gmsh's :attr:`PartitionRecord.id` is **1-based** (``P1, P2, ...``);
``OpenSeesMP::getPID()`` (and MPI rank) is **0-based** (``0, 1, ...``).
The conversion is the ``enumerate`` index over ``fem.partitions``,
NOT ``record.id - 1`` — the helper
:func:`runtime_rank_from_partition_record` is the single source of
truth for this seam and these tests lock the contract.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from apeGmsh._kernel.records._partitions import PartitionRecord
from apeGmsh.opensees._internal.build import (
    build_element_partition_owner,
    build_node_partition_owners,
    primary_owner_map,
    runtime_rank_from_partition_record,
)


def _make_record(*, id: int, node_ids, element_ids) -> PartitionRecord:
    return PartitionRecord(
        id=int(id),
        node_ids=np.asarray(node_ids, dtype=np.int64),
        element_ids=np.asarray(element_ids, dtype=np.int64),
    )


def test_helper_returns_zero_based_index() -> None:
    """The helper returns its ``index`` argument verbatim — i.e. the
    0-based enumerate index over ``fem.partitions``."""
    rec = _make_record(id=1, node_ids=[], element_ids=[])
    assert runtime_rank_from_partition_record(rec, 0) == 0
    assert runtime_rank_from_partition_record(rec, 1) == 1
    assert runtime_rank_from_partition_record(rec, 2) == 2
    assert runtime_rank_from_partition_record(rec, 7) == 7


def test_helper_ignores_partition_record_id() -> None:
    """The conversion is the ``enumerate`` index — NOT ``record.id - 1``.

    A degenerate ``PartitionRecord`` with a high ``id`` at index 0 still
    maps to runtime rank 0.  This locks the contract against a future
    "fix" that swaps to ``record.id - 1``.
    """
    rec = _make_record(id=99, node_ids=[], element_ids=[])
    assert runtime_rank_from_partition_record(rec, 0) == 0
    # If the helper used record.id - 1 it would return 98 here.
    assert runtime_rank_from_partition_record(rec, 0) != rec.id - 1


def test_build_node_partition_owners_returns_zero_based_ranks() -> None:
    """``build_node_partition_owners`` keys its owner sets by **0-based
    runtime rank**, not by ``PartitionRecord.id``.

    Two records with Gmsh ids ``(1, 2)`` must produce owner sets drawn
    from ``{0, 1}`` — not ``{1, 2}``.
    """
    p1 = _make_record(id=1, node_ids=[10, 11], element_ids=[])
    p2 = _make_record(id=2, node_ids=[11, 12], element_ids=[])
    fem = SimpleNamespace(partitions=(p1, p2))

    owners = build_node_partition_owners(fem)

    all_ranks = set().union(*owners.values())
    assert all_ranks == {0, 1}
    # Shared node 11 belongs to both ranks.
    assert owners[10] == {0}
    assert owners[11] == {0, 1}
    assert owners[12] == {1}


def test_build_element_partition_owner_returns_zero_based_ranks() -> None:
    """``build_element_partition_owner`` maps each element to its
    **0-based runtime rank**, not its ``PartitionRecord.id``.
    """
    p1 = _make_record(id=1, node_ids=[], element_ids=[100, 101])
    p2 = _make_record(id=2, node_ids=[], element_ids=[200, 201])
    fem = SimpleNamespace(partitions=(p1, p2))

    owners = build_element_partition_owner(fem)

    assert set(owners.values()) == {0, 1}
    assert owners[100] == 0
    assert owners[101] == 0
    assert owners[200] == 1
    assert owners[201] == 1


def test_primary_owner_map_picks_lowest_rank_for_shared_nodes() -> None:
    """``primary_owner_map`` reduces multi-rank owner sets to the LOWEST
    owning runtime rank — the single rank where additive nodal
    quantities (mass / load) emit, deterministically.
    """
    p1 = _make_record(id=1, node_ids=[10, 11], element_ids=[])
    p2 = _make_record(id=2, node_ids=[11, 12], element_ids=[])
    fem = SimpleNamespace(partitions=(p1, p2))

    primary = primary_owner_map(build_node_partition_owners(fem))

    assert primary == {10: 0, 11: 0, 12: 1}


def test_primary_owner_map_assigns_each_node_exactly_one_rank() -> None:
    """Every node in the owner map appears exactly once in the primary
    map — the invariant the mass / load dedup rests on."""
    p1 = _make_record(id=1, node_ids=[1, 2, 3], element_ids=[])
    p2 = _make_record(id=2, node_ids=[2, 3, 4], element_ids=[])
    p3 = _make_record(id=3, node_ids=[3, 4, 5], element_ids=[])
    fem = SimpleNamespace(partitions=(p1, p2, p3))

    owners = build_node_partition_owners(fem)
    primary = primary_owner_map(owners)

    assert set(primary) == set(owners)
    for nid, rank in primary.items():
        assert rank in owners[nid]
    # Triple-shared node 3 lands on the lowest rank.
    assert primary[3] == 0


def test_build_helpers_use_enumerate_index_not_record_id() -> None:
    """Locks the convention against the seductive ``record.id - 1``
    alternative.  Two records with Gmsh ids ``(7, 13)`` (non-contiguous,
    non-zero-aligned) still map to ranks ``{0, 1}`` because the seam is
    the ``enumerate`` index.
    """
    p7 = _make_record(id=7, node_ids=[1], element_ids=[1])
    p13 = _make_record(id=13, node_ids=[2], element_ids=[2])
    fem = SimpleNamespace(partitions=(p7, p13))

    node_owners = build_node_partition_owners(fem)
    ele_owner = build_element_partition_owner(fem)

    # If the helper used record.id - 1 we'd see ranks {6, 12}.
    assert set().union(*node_owners.values()) == {0, 1}
    assert set(ele_owner.values()) == {0, 1}
