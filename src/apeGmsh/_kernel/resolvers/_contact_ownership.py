"""Contact ownership resolver — one owner rank per interaction (ADR 0092 S1).

Pure kernel function: given a resolved :class:`~apeGmsh._kernel.records.
_constraints.ContactRecord` / :class:`~apeGmsh._kernel.records._constraints.
ContactPlaneRecord` and the mesh's :class:`~apeGmsh._kernel.records.
_partitions.PartitionRecord` set, decide which rank emits the interaction
and which interface nodes that rank must ghost-declare.

This is S1 of ADR 0092 — **no emit change**. Nothing here is wired into
``apesees.py`` / ``opensees/_internal/build.py`` yet; later slices (S4)
call this resolver from the partitioned emit path. Pure NumPy — no Gmsh,
no OpenSees imports.

Rules (ADR 0092 §Decision, INV-1/INV-2, post adversarial-review
correction):

* **Owner = the rank owning the MASTER surface's nodes** (not the slave).
  The fork's ``-kn auto`` resolves the owning solid of the *master*
  segment, so a master-side owner keeps penalty auto-sizing native. If
  master nodes span several ranks, the rank owning the most master nodes
  wins; exact ties break to the **lowest** rank index for deterministic
  emission.
* **Ghost set = every node of BOTH surfaces the owner does not natively
  own** — the whole interface, not a geometric halo (INV-2). "Natively
  own" mirrors ADR 0027's foreign-node-replication language: a node is
  native to a rank when that rank's own :attr:`PartitionRecord.node_ids`
  contains it.
* :class:`~apeGmsh._kernel.records._constraints.ContactPlaneRecord` has no
  master surface (an analytical rigid plane) — ownership is decided from
  the **slave** nodes instead, and the ghost set is the slave surface
  alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

from apeGmsh._kernel.records._constraints import ContactPlaneRecord, ContactRecord

if TYPE_CHECKING:
    from apeGmsh._kernel.records._partitions import PartitionRecord


__all__ = ["ContactOwnership", "resolve_contact_ownership"]


@dataclass(frozen=True)
class ContactOwnership:
    """Resolved owner rank + ghost set for one contact interaction.

    Attributes
    ----------
    owner_rank
        0-based runtime rank whose emit block carries the
        ``contactSurface`` / ``contact`` (or ``contactPlane``) lines.
    ghost_node_ids
        Sorted, deduplicated tags of every interface node the owner rank
        does not natively own — the nodes S4 must ``node(tag, *xyz, ...)``
        declare (+ SP replay) before emitting the interaction.
    """

    owner_rank: int
    ghost_node_ids: tuple[int, ...]


def _owning_ranks(
    partitions: Iterable["PartitionRecord"],
) -> dict[int, tuple[int, ...]]:
    """``{node_id: (rank, ...)}`` — every rank whose own partition holds it.

    Ranks are the 0-based ``enumerate`` position over ``partitions``
    sorted by :attr:`PartitionRecord.id`, matching the codebase's single
    source of truth for the Gmsh-id → runtime-rank conversion
    (``opensees._internal.build.runtime_rank_from_partition_record``). A
    node present in more than one partition's ``node_ids`` (a replicated
    boundary node) is natively owned by every rank listed.
    """
    ranks_by_node: dict[int, set[int]] = {}
    for rank, part in enumerate(sorted(partitions, key=lambda p: p.id)):
        for nid in part.node_ids:
            ranks_by_node.setdefault(int(nid), set()).add(rank)
    return {nid: tuple(sorted(ranks)) for nid, ranks in ranks_by_node.items()}


def _majority_owner(
    node_ids: Iterable[int], owning_ranks: Mapping[int, tuple[int, ...]],
) -> int:
    """Rank owning the most of ``node_ids``; ties break to the lowest rank."""
    tally: dict[int, int] = {}
    for nid in node_ids:
        for rank in owning_ranks.get(int(nid), ()):
            tally[rank] = tally.get(rank, 0) + 1
    if not tally:
        raise ValueError(
            "resolve_contact_ownership: none of the interaction's owner "
            "nodes appear in any partition"
        )
    best = max(tally.values())
    return min(rank for rank, count in tally.items() if count == best)


def _ghost_nodes(
    interface_node_ids: Iterable[int],
    owner_rank: int,
    owning_ranks: Mapping[int, tuple[int, ...]],
) -> tuple[int, ...]:
    return tuple(sorted({
        int(nid) for nid in interface_node_ids
        if owner_rank not in owning_ranks.get(int(nid), ())
    }))


def _master_node_ids(record: ContactRecord) -> list[int]:
    if record.master_faces is None:
        return []
    faces = np.asarray(record.master_faces)
    return sorted({int(n) for n in faces.reshape(-1)})


def _slave_node_ids(record: "ContactRecord | ContactPlaneRecord") -> list[int]:
    if isinstance(record, ContactPlaneRecord):
        return sorted({int(n) for n in (record.slave_nodes or ())})
    if record.formulation == "mortar":
        if record.slave_faces is None:
            return []
        faces = np.asarray(record.slave_faces)
        return sorted({int(n) for n in faces.reshape(-1)})
    return sorted({int(n) for n in (record.slave_nodes or ())})


def resolve_contact_ownership(
    record: "ContactRecord | ContactPlaneRecord",
    partitions: Iterable["PartitionRecord"],
) -> ContactOwnership:
    """Resolve the owner rank + ghost node set for one contact interaction.

    Parameters
    ----------
    record
        A resolved :class:`ContactRecord` (mesh-to-mesh) or
        :class:`ContactPlaneRecord` (mesh-to-rigid-plane) interaction.
    partitions
        The mesh's partition records (``fem.partitions``); order does not
        matter, this function sorts by :attr:`PartitionRecord.id` itself.

    Returns
    -------
    ContactOwnership
        ``owner_rank`` per ADR 0092 INV-1 (master-side majority, lowest
        rank breaks ties; slave-side for a :class:`ContactPlaneRecord`),
        and ``ghost_node_ids`` per INV-2 (whole non-native interface,
        sorted + deduplicated).
    """
    owning_ranks = _owning_ranks(partitions)

    if isinstance(record, ContactPlaneRecord):
        slave_ids = _slave_node_ids(record)
        if not slave_ids:
            raise ValueError(
                "resolve_contact_ownership: ContactPlaneRecord has no "
                "slave nodes"
            )
        owner_rank = _majority_owner(slave_ids, owning_ranks)
        interface_ids: list[int] = slave_ids
    elif isinstance(record, ContactRecord):
        master_ids = _master_node_ids(record)
        if not master_ids:
            raise ValueError(
                "resolve_contact_ownership: ContactRecord has no master "
                "nodes"
            )
        slave_ids = _slave_node_ids(record)
        owner_rank = _majority_owner(master_ids, owning_ranks)
        interface_ids = sorted(set(master_ids) | set(slave_ids))
    else:
        raise TypeError(
            "resolve_contact_ownership: expected a ContactRecord or "
            f"ContactPlaneRecord, got {type(record).__name__}"
        )

    ghost_ids = _ghost_nodes(interface_ids, owner_rank, owning_ranks)
    return ContactOwnership(owner_rank=owner_rank, ghost_node_ids=ghost_ids)
