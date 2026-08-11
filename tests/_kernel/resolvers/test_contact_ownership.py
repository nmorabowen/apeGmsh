"""Unit battery for the contact ownership resolver (ADR 0092 S1).

Pure kernel function: proves owner-rank selection (master-side majority,
lowest-rank tie-break; slave-side for a rigid-plane interaction) and the
ghost node set (whole non-native interface, sorted + deduplicated) against
synthetic :class:`PartitionRecord` fixtures. No Gmsh, no OpenSees, no emit.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import ContactPlaneRecord, ContactRecord
from apeGmsh._kernel.records._partitions import PartitionRecord
from apeGmsh._kernel.resolvers._contact_ownership import (
    ContactOwnership,
    resolve_contact_ownership,
)


def _partition(pid: int, node_ids: list[int]) -> PartitionRecord:
    return PartitionRecord(
        id=pid,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        element_ids=np.empty((0,), dtype=np.int64),
    )


def _nts(master_faces: list[list[int]], slave_nodes: list[int]) -> ContactRecord:
    return ContactRecord(
        kind="contact",
        formulation="nts",
        master_faces=np.asarray(master_faces, dtype=np.int64),
        master_nps=len(master_faces[0]),
        slave_nodes=list(slave_nodes),
    )


def _mortar(
    master_faces: list[list[int]], slave_faces: list[list[int]],
) -> ContactRecord:
    return ContactRecord(
        kind="contact",
        formulation="mortar",
        master_faces=np.asarray(master_faces, dtype=np.int64),
        master_nps=len(master_faces[0]),
        slave_faces=np.asarray(slave_faces, dtype=np.int64),
        slave_nps=len(slave_faces[0]),
    )


def _plane(slave_nodes: list[int]) -> ContactPlaneRecord:
    return ContactPlaneRecord(
        kind="contact_plane",
        slave_nodes=list(slave_nodes),
        normal=(0.0, 0.0, 1.0),
        point=(0.0, 0.0, 0.0),
        kn=1.0e9,
    )


class TestWholeInterfaceOneRank:
    def test_owner_and_empty_ghost_set(self) -> None:
        parts = [_partition(1, [1, 2, 3, 4, 5, 6, 7, 8])]
        rec = _nts([[1, 2, 3, 4]], [5, 6, 7, 8])

        result = resolve_contact_ownership(rec, parts)

        assert result == ContactOwnership(owner_rank=0, ghost_node_ids=())


class TestMasterSlaveOnDifferentRanks:
    def test_owner_is_master_rank_ghosts_are_slave_nodes(self) -> None:
        parts = [
            _partition(1, [1, 2, 3, 4]),   # rank 0 — master
            _partition(2, [5, 6, 7, 8]),   # rank 1 — slave
        ]
        rec = _nts([[1, 2, 3, 4]], [5, 6, 7, 8])

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 0
        assert result.ghost_node_ids == (5, 6, 7, 8)


class TestMasterMajority:
    def test_three_one_split_owner_is_majority_rank(self) -> None:
        parts = [
            _partition(1, [1, 2, 3]),      # rank 0 — 3 master nodes
            _partition(2, [4] + [5, 6]),   # rank 1 — 1 master node + slaves
        ]
        rec = _nts([[1, 2, 3, 4]], [5, 6])

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 0
        # Owner is rank 0: master node 4 and both slave nodes are foreign.
        assert result.ghost_node_ids == (4, 5, 6)


class TestExactTieBreaksToLowestRank:
    def test_two_two_split_owner_is_lowest_rank(self) -> None:
        parts = [
            _partition(1, [1, 2]),
            _partition(2, [3, 4]),
        ]
        rec = _nts([[1, 2, 3, 4]], [])

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 0
        assert result.ghost_node_ids == (3, 4)

    def test_tie_break_is_order_independent(self) -> None:
        # Partitions handed in descending id order — resolver must sort
        # by PartitionRecord.id itself before deriving ranks.
        parts = [
            _partition(2, [3, 4]),
            _partition(1, [1, 2]),
        ]
        rec = _nts([[1, 2, 3, 4]], [])

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 0


class TestThreeRanks:
    def test_owner_is_the_rank_with_the_most_master_nodes(self) -> None:
        parts = [
            _partition(1, [1, 2, 3]),   # rank 0 — 3 master nodes
            _partition(2, [4]),         # rank 1 — 1 master node
            _partition(3, [5, 9]),      # rank 2 — 1 master node (5) + slave (9)
        ]
        rec = _nts([[1, 2, 3, 4, 5]], [9])

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 0
        assert result.ghost_node_ids == (4, 5, 9)


class TestContactPlaneOwnershipFromSlaveNodes:
    def test_owner_from_majority_slave_rank(self) -> None:
        parts = [
            _partition(1, [1]),         # rank 0 — 1 slave node
            _partition(2, [2, 3]),      # rank 1 — 2 slave nodes
        ]
        rec = _plane([1, 2, 3])

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 1
        assert result.ghost_node_ids == (1,)

    def test_plane_wholly_inside_one_rank_has_empty_ghost_set(self) -> None:
        parts = [_partition(1, [1, 2, 3])]
        rec = _plane([1, 2, 3])

        result = resolve_contact_ownership(rec, parts)

        assert result == ContactOwnership(owner_rank=0, ghost_node_ids=())


class TestGhostSetSortedAndDeduplicated:
    def test_mortar_faces_yield_sorted_unique_ghosts(self) -> None:
        parts = [
            _partition(1, [1, 2, 3, 4]),        # rank 0 — master
            _partition(2, [4, 5, 6, 7]),        # rank 1 — slave (node 4 shared)
        ]
        # Slave faces reference node 4 twice across facets, plus node 4 is
        # also a master node — the ghost set must still be sorted+unique.
        rec = _mortar(
            master_faces=[[1, 2, 3, 4]],
            slave_faces=[[4, 5, 6], [4, 6, 7]],
        )

        result = resolve_contact_ownership(rec, parts)

        assert result.owner_rank == 0
        # node 4 is natively owned by rank 0 (in its own partition) even
        # though it's also present on rank 1 — not a ghost. 5,6,7 are.
        assert result.ghost_node_ids == (5, 6, 7)
        assert result.ghost_node_ids == tuple(sorted(set(result.ghost_node_ids)))


class TestErrorCases:
    def test_contact_record_without_master_faces_raises(self) -> None:
        parts = [_partition(1, [1, 2, 3, 4])]
        rec = ContactRecord(kind="contact", formulation="nts", slave_nodes=[1, 2])

        with pytest.raises(ValueError, match="no master nodes"):
            resolve_contact_ownership(rec, parts)

    def test_contact_plane_without_slave_nodes_raises(self) -> None:
        parts = [_partition(1, [1, 2, 3, 4])]
        rec = ContactPlaneRecord(kind="contact_plane", slave_nodes=None)

        with pytest.raises(ValueError, match="no slave nodes"):
            resolve_contact_ownership(rec, parts)

    def test_unsupported_record_type_raises(self) -> None:
        parts = [_partition(1, [1, 2, 3, 4])]

        with pytest.raises(TypeError, match="ContactRecord or ContactPlaneRecord"):
            resolve_contact_ownership(object(), parts)  # type: ignore[arg-type]
