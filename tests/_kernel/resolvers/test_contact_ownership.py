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
    master_backing_element_ids,
    master_node_rank_span,
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


class TestReplicatedBoundaryNodes:
    """Owner selection when partition-boundary nodes are replicated.

    ``extract_partitions`` adds every element's nodes to every partition on the
    entity, so a node on a cut belongs to EVERY touching rank. A raw tally counts
    such a node for all of them, which is how a rank holding no backing solid at
    all could tie and then win the lowest-rank break -- on a model that fully
    honoured INV-4. These pin the corrected behaviour (ADR 0092 INV-1, amended
    after the S1/S2 adversarial review).
    """

    def test_unique_ownership_beats_a_larger_replicated_count(self) -> None:
        # Rank 1 uniquely owns ONE master node (it holds the backing solid).
        # Rank 0 touches three others only by replication -- they are shared, so
        # they say nothing about locality. A raw tally would read 3 vs 4 and pick
        # rank 0; the unique tally correctly picks rank 1.
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[90, 91])
        parts = [
            _partition(1, [11, 12, 13, 80]),          # replication only
            _partition(2, [10, 11, 12, 13, 90, 91]),  # 10 is uniquely its own
        ]

        assert resolve_contact_ownership(rec, parts).owner_rank == 1

    def test_all_replicated_with_an_exact_tie_refuses(self) -> None:
        # Every master node is shared and the raw counts tie. Nothing in the node
        # data distinguishes the backing rank, and the old `min()` silently
        # returned rank 0 -- which an executed counterexample showed can hold ZERO
        # backing solids. Refuse loudly instead of guessing.
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30, 31])
        parts = [
            _partition(1, [10, 11, 12, 13, 20, 21]),
            _partition(2, [10, 11, 12, 13, 30, 31]),
        ]

        with pytest.raises(ValueError, match="cannot choose an owner rank"):
            resolve_contact_ownership(rec, parts)

    def test_all_replicated_without_a_tie_still_resolves(self) -> None:
        # Shared everywhere, but rank 1 touches strictly more master nodes, so
        # there is still a defensible answer. Only the exact tie is undecidable.
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30])
        parts = [
            _partition(1, [10, 11]),
            _partition(2, [10, 11, 12, 13, 30]),
        ]

        assert resolve_contact_ownership(rec, parts).owner_rank == 1

    def test_tie_break_is_by_rank_index_not_node_order(self) -> None:
        # Review finding F3: the original tie test gave the LOWER rank the
        # LOWER-numbered nodes, so an implementation that returned "first rank
        # reaching the best count in tally-insertion order" passed it by
        # coincidence. Here the lower rank owns the HIGHER-numbered nodes, so
        # insertion order and rank order disagree and only a real min() passes.
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[50])
        parts = [
            _partition(1, [12, 13, 50]),   # rank 0 -- higher-numbered nodes
            _partition(2, [10, 11]),       # rank 1 -- lower-numbered nodes
        ]

        assert resolve_contact_ownership(rec, parts).owner_rank == 0


class TestMasterBackingElementIds:
    """Facet → backing-solid resolution (ADR 0092 INV-1, S4 exactness).

    A master facet lies on the body's boundary, so exactly one solid
    element contains all of its nodes — that element is the facet's
    backing solid, and its rank is the ownership property ``-kn auto``
    actually needs. An ambiguous facet yields a per-facet ``None``
    (2026-08-13 review F2 — the old all-or-nothing ``None`` let one
    ambiguous facet disengage the INV-4 backstop for the whole surface);
    a record-level ``None`` survives only for records with no usable
    facet connectivity. Never a guess either way.
    """

    def test_boundary_facets_resolve_to_their_unique_solid(self) -> None:
        # Two stacked hexes; the interface facet [5,6,7,8] belongs to the
        # master body's top face — only element 101 contains all 4 nodes.
        rec = _nts([[5, 6, 7, 8]], slave_nodes=[11, 12, 13, 14])
        groups = [(
            np.asarray([101, 102], dtype=np.int64),
            np.asarray([
                [1, 2, 3, 4, 5, 6, 7, 8],
                [11, 12, 13, 14, 15, 16, 17, 18],
            ], dtype=np.int64),
        )]

        assert master_backing_element_ids(rec, groups) == (101,)

    def test_facet_order_is_preserved_across_multiple_facets(self) -> None:
        rec = _nts([[5, 6, 10], [6, 9, 10]], slave_nodes=[50])
        groups = [(
            np.asarray([7, 8], dtype=np.int64),
            np.asarray([
                [5, 6, 10, 20],     # tet 7 backs facet [5,6,10]
                [6, 9, 10, 21],     # tet 8 backs facet [6,9,10]
            ], dtype=np.int64),
        )]

        assert master_backing_element_ids(rec, groups) == (7, 8)

    def test_uncovered_facet_node_yields_per_facet_none(self) -> None:
        # Node 99 belongs to no element — THAT facet is unresolvable
        # (per-facet None, review F2); the caller decides what to do
        # with the partial map, never guesses.
        rec = _nts([[5, 6, 99]], slave_nodes=[50])
        groups = [(
            np.asarray([7], dtype=np.int64),
            np.asarray([[5, 6, 10, 20]], dtype=np.int64),
        )]

        assert master_backing_element_ids(rec, groups) == (None,)

    def test_interior_face_with_two_candidates_yields_none_entry(self) -> None:
        # Both tets contain the whole facet (an interior face) — ambiguous.
        rec = _nts([[5, 6, 10]], slave_nodes=[50])
        groups = [(
            np.asarray([7, 8], dtype=np.int64),
            np.asarray([
                [5, 6, 10, 20],
                [5, 6, 10, 21],
            ], dtype=np.int64),
        )]

        assert master_backing_element_ids(rec, groups) == (None,)

    def test_one_ambiguous_facet_keeps_the_resolved_neighbours(self) -> None:
        # Review F2's exact shape: the old contract returned None for the
        # WHOLE surface when one facet was ambiguous, silently skipping
        # the INV-4 cut-master check on the resolved rest. Now the
        # resolved facets keep their ids and only the ambiguous one is
        # None.
        rec = _nts([[5, 6, 10], [6, 9, 10], [5, 6, 99]], slave_nodes=[50])
        groups = [(
            np.asarray([7, 8], dtype=np.int64),
            np.asarray([
                [5, 6, 10, 20],     # tet 7 backs facet [5,6,10]
                [6, 9, 10, 21],     # tet 8 backs facet [6,9,10]
            ], dtype=np.int64),
        )]

        assert master_backing_element_ids(rec, groups) == (7, 8, None)

    def test_no_master_faces_returns_none(self) -> None:
        rec = ContactRecord(kind="contact", formulation="nts",
                            slave_nodes=[1, 2])

        assert master_backing_element_ids(rec, []) is None


class TestElementExactOwnerPick:
    """``master_element_ranks`` makes INV-1 exact (second amendment).

    The executed S1 counterexample — every master node replicated with
    tied counts — is undecidable from node data and refuses. With the
    backing solids' ranks supplied, the same partitions resolve exactly.
    """

    def test_element_ranks_decide_what_the_node_tally_cannot(self) -> None:
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30, 31])
        parts = [
            _partition(1, [10, 11, 12, 13, 20, 21]),
            _partition(2, [10, 11, 12, 13, 30, 31]),
        ]
        # Node data alone: refuses (pinned above). Backing solids on
        # rank 1: exact.
        with pytest.raises(ValueError, match="cannot choose an owner rank"):
            resolve_contact_ownership(rec, parts)

        result = resolve_contact_ownership(
            rec, parts, master_element_ranks=(1,),
        )

        assert result.owner_rank == 1
        # Ghost set unchanged by the pick mechanism: whole non-native
        # interface relative to the owner (everything here is native to
        # rank 1).
        assert result.ghost_node_ids == ()

    def test_element_majority_wins_over_node_majority(self) -> None:
        # Rank 0 uniquely owns a master node, but the backing solids sit
        # on rank 1 — elements are the real property, so rank 1 wins.
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30])
        parts = [
            _partition(1, [10, 11, 12, 13]),
            _partition(2, [30]),
        ]

        result = resolve_contact_ownership(
            rec, parts, master_element_ranks=(1, 1),
        )

        assert result.owner_rank == 1
        # Master nodes are native to rank 0 only — ALL of them ghost onto
        # the rank-1 owner (INV-2), while slave node 30 is native there.
        assert result.ghost_node_ids == (10, 11, 12, 13)

    def test_element_rank_tie_breaks_to_lowest(self) -> None:
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30])
        parts = [
            _partition(1, [10, 11]),
            _partition(2, [12, 13, 30]),
        ]

        result = resolve_contact_ownership(
            rec, parts, master_element_ranks=(1, 0),
        )

        assert result.owner_rank == 0

    def test_empty_element_ranks_fall_back_to_node_tally(self) -> None:
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30])
        parts = [
            _partition(1, [10, 11, 12]),
            _partition(2, [13, 30]),
        ]

        result = resolve_contact_ownership(
            rec, parts, master_element_ranks=(),
        )

        assert result.owner_rank == 0   # node majority (3 vs 1)


class TestPlaneUndecidableTieMessage:
    """Review F4: the undecidable-tie refusal must be accurate per lane.

    A ContactPlaneRecord has no master surface and the resolver never
    consults element ownership for it (``master_element_ranks`` is
    ignored), so the old message — "disambiguate with element ownership"
    — was advice the code cannot take. The plane lane names the slave
    tally and suggests not cutting the slave surface instead.
    """

    def _tied_plane_parts(self):
        rec = _plane([1, 2, 3, 4])
        parts = [
            _partition(1, [1, 2, 3, 4, 20]),
            _partition(2, [1, 2, 3, 4, 30]),
        ]
        return rec, parts

    def test_plane_tie_names_the_slave_tally(self) -> None:
        rec, parts = self._tied_plane_parts()

        with pytest.raises(ValueError) as excinfo:
            resolve_contact_ownership(rec, parts)
        msg = str(excinfo.value)
        assert "cannot choose an owner rank" in msg
        assert "SLAVE surface" in msg
        assert "no master surface" in msg
        assert "cutting the slave surface" in msg
        # The master-lane advice must NOT leak into the plane lane:
        # ownership is never consulted for a plane.
        assert "Disambiguate with element ownership" not in msg
        assert "backing solid elements" not in msg

    def test_master_lane_message_unchanged(self) -> None:
        rec = _nts([[10, 11, 12, 13]], slave_nodes=[30, 31])
        parts = [
            _partition(1, [10, 11, 12, 13, 20, 21]),
            _partition(2, [10, 11, 12, 13, 30, 31]),
        ]

        with pytest.raises(ValueError, match="element ownership"):
            resolve_contact_ownership(rec, parts)


class TestMasterNodeRankSpan:
    """Node-view rank span of the master surface (review F2/F3 helper)."""

    def test_single_rank_master(self) -> None:
        rec = _nts([[1, 2, 3, 4]], slave_nodes=[9])
        parts = [
            _partition(1, [1, 2, 3, 4]),
            _partition(2, [9]),
        ]

        assert master_node_rank_span(rec, parts) == (0,)

    def test_replicated_boundary_nodes_span_both_ranks(self) -> None:
        rec = _nts([[1, 2, 3, 4]], slave_nodes=[9])
        parts = [
            _partition(1, [1, 2, 9]),
            _partition(2, [2, 3, 4]),   # node 2 replicated
        ]

        assert master_node_rank_span(rec, parts) == (0, 1)

    def test_nodes_in_no_partition_yield_empty_span(self) -> None:
        rec = _nts([[1, 2, 3, 4]], slave_nodes=[9])

        assert master_node_rank_span(rec, []) == ()
