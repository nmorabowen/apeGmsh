"""Contact ownership resolver — one owner rank per interaction (ADR 0092 S1).

Pure kernel function: given a resolved :class:`~apeGmsh._kernel.records.
_constraints.ContactRecord` / :class:`~apeGmsh._kernel.records._constraints.
ContactPlaneRecord` and the mesh's :class:`~apeGmsh._kernel.records.
_partitions.PartitionRecord` set, decide which rank emits the interaction
and which interface nodes that rank must ghost-declare.

Born as S1 of ADR 0092 (pure kernel, no emit change); since S4 the
partitioned emit path (``BuiltModel._plan_partitioned_contacts`` in
``opensees/apesees.py``) calls it per interaction, passing the ranks of
the master surface's backing solid elements (via
:func:`master_backing_element_ids` + the emit layer's element→rank map)
to make the owner pick exact where the mesh admits it. Pure NumPy — no
Gmsh, no OpenSees imports.

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


__all__ = [
    "ContactOwnership",
    "master_backing_element_ids",
    "resolve_contact_ownership",
    "soft_family_knobs",
]


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


# ADR 0092 S3 (INV-3): the SOFT-family knobs each record kind can carry.
# `visc` is deliberately NOT here — viscous stabilisation is a damper on the
# owner-local active set, not a mass-sized penalty, so it has no both-sides
# assembled-mass dependency.
_SOFT_KNOBS_BY_KIND: dict[type, tuple[str, ...]] = {
    ContactRecord: ("soft", "edge_soft"),
    ContactPlaneRecord: ("soft",),
}


def soft_family_knobs(
    record: "ContactRecord | ContactPlaneRecord",
) -> tuple[str, ...]:
    """Names of the active SOFT-family knobs on one contact interaction.

    ADR 0092 INV-3 / S3: the explicit Courant-stable SOFT penalty
    (``-soft`` / ``-edgeSoft``) sizes ``k_soft = SOFSCL·4·m_eff/dt²`` from
    the **assembled** mass of BOTH contact surfaces. Under partitioning the
    ghosted side of the interface contributes zero assembled mass on the
    owner rank, so no owner rule can make SOFT correct (fork ADR-78 D4);
    the fork engine refuses ``-soft``/``-edgeSoft`` at handle() time under
    MPI (fork ADR-78 §P2). This predicate is the emit-side half: the
    partitioned emit path refuses at deck-generation time any record for
    which it returns a non-empty tuple.

    A knob is *active* exactly when the args builder would emit its token:
    any value other than ``None``/``False`` (``True`` = fork default
    SOFSCL, a float = explicit SOFSCL — mirroring
    ``opensees.element.contact.contact_args``).

    Returns the active knob names in declaration order (``("soft",)``,
    ``("edge_soft",)``, ``("soft", "edge_soft")``) or ``()`` when the
    record is partition-safe on this axis.
    """
    for kind, knobs in _SOFT_KNOBS_BY_KIND.items():
        if isinstance(record, kind):
            return tuple(
                knob for knob in knobs
                if getattr(record, knob) is not None
                and getattr(record, knob) is not False
            )
    raise TypeError(
        "soft_family_knobs: expected a ContactRecord or "
        f"ContactPlaneRecord, got {type(record).__name__}"
    )


def master_backing_element_ids(
    record: ContactRecord,
    element_groups: "Iterable[tuple[np.ndarray, np.ndarray]]",
) -> "tuple[int, ...] | None":
    """Backing solid element id per master facet, or ``None`` when the
    facet → element map cannot be resolved from the given connectivity.

    ADR 0092 INV-1 (second amendment): node majority is a proxy; the
    property that actually matters is which rank owns the master surface's
    **backing solid elements** (the fork's ``-kn auto`` resolves the owning
    solid of each master segment). A :class:`ContactRecord` carries only
    node connectivity, but the emit layer (S4) has the mesh's element
    connectivity — this helper closes the gap: a master facet lies on the
    body's boundary, so exactly ONE solid element contains all of its
    nodes, and that element is the facet's backing solid.

    Parameters
    ----------
    record
        The contact interaction whose ``master_faces`` to resolve.
    element_groups
        Iterable of ``(ids, connectivity)`` array pairs — one per element
        type — covering the mesh's volume elements (the shape
        ``FEMData``'s element groups expose).

    Returns
    -------
    tuple[int, ...] | None
        One backing element id per master facet (facet order preserved),
        or ``None`` when any facet resolves to zero or several candidate
        elements (an interior face, a non-conforming patch, or
        connectivity that does not cover the surface). ``None`` means
        "fall back to the node tally" — the caller must NOT guess.
    """
    if record.master_faces is None:
        return None
    faces = np.asarray(record.master_faces)
    if faces.ndim == 1:
        if record.master_nps <= 0 or faces.size % record.master_nps:
            return None
        faces = faces.reshape(-1, record.master_nps)
    if faces.size == 0:
        return None

    master_nodes = {int(n) for n in faces.reshape(-1)}
    master_arr = np.fromiter(master_nodes, dtype=np.int64)

    # node -> {touching element ids}, restricted to master-surface nodes
    # (one vectorised isin per element type; never a full-mesh dict).
    incidence: "dict[int, set[int]]" = {}
    for ids, conn in element_groups:
        ids_arr = np.asarray(ids, dtype=np.int64)
        conn_arr = np.asarray(conn)
        if conn_arr.size == 0 or conn_arr.ndim != 2:
            continue
        hit = np.isin(conn_arr, master_arr)
        for row in np.nonzero(hit.any(axis=1))[0]:
            eid = int(ids_arr[row])
            for nid in conn_arr[row][hit[row]]:
                incidence.setdefault(int(nid), set()).add(eid)

    backing: "list[int]" = []
    for facet in faces:
        candidate: "set[int] | None" = None
        for nid in facet:
            touching = incidence.get(int(nid))
            if not touching:
                return None
            candidate = (
                set(touching) if candidate is None
                else candidate & touching
            )
        if candidate is None or len(candidate) != 1:
            return None
        backing.append(next(iter(candidate)))
    return tuple(backing)


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
    """Rank owning the most of ``node_ids``; ties break to the lowest rank.

    Counts **uniquely-owned** nodes first. A partition-boundary node is
    replicated onto every rank whose elements touch it (``extract_partitions``
    adds each element's nodes to each partition on the entity), so a shared node
    carries no locality information at all — counting it for every holder is
    what let a rank with *zero* backing solids tie and then win the lowest-rank
    break. Under ADR 0092 INV-4 the backing rank natively owns every master node
    (each master facet's nodes belong to its backing solid), so if any master
    node is uniquely owned it is uniquely owned by that rank — which makes the
    unique tally exact, not merely a good proxy.

    The all-nodes tally survives only as the fallback for the degenerate case
    where *every* candidate node is replicated and the unique tally is empty.
    """
    unique: dict[int, int] = {}
    shared: dict[int, int] = {}
    seen_any = False
    for nid in node_ids:
        ranks = owning_ranks.get(int(nid), ())
        if not ranks:
            continue
        seen_any = True
        for rank in ranks:
            shared[rank] = shared.get(rank, 0) + 1
        if len(ranks) == 1:
            unique[ranks[0]] = unique.get(ranks[0], 0) + 1
    if not seen_any:
        raise ValueError(
            "resolve_contact_ownership: none of the interaction's owner "
            "nodes appear in any partition"
        )
    if unique:
        best = max(unique.values())
        return min(rank for rank, count in unique.items() if count == best)

    # Every candidate node is replicated, so nothing here distinguishes the rank
    # that holds the backing solids from a neighbour that merely touches the
    # surface. If one rank still leads on the raw count, take it. If the raw
    # count TIES, the answer is genuinely undecidable from node data and the old
    # `min()` silently handed ownership to the lower rank -- which a executed
    # counterexample showed can be a rank with ZERO backing solids, on a model
    # that fully honoured INV-4. Refuse instead: the caller (S4) knows the
    # backing elements and can disambiguate, and a named error at emit time
    # beats a partitioned run that dies later inside `-kn auto`.
    best = max(shared.values())
    leaders = sorted(rank for rank, count in shared.items() if count == best)
    if len(leaders) > 1:
        raise ValueError(
            "resolve_contact_ownership: cannot choose an owner rank -- every "
            f"candidate node is shared, and ranks {leaders} hold equal counts "
            f"({best} each). Node ownership alone cannot say which rank holds "
            "the master surface's backing solid elements, and picking the "
            "lowest can select a rank that holds none (ADR 0092 INV-1). "
            "Disambiguate with element ownership."
        )
    return leaders[0]


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
    *,
    master_element_ranks: "Iterable[int] | None" = None,
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
    master_element_ranks
        Optional — the ranks owning the master surface's **backing solid
        elements** (one entry per facet, from
        :func:`master_backing_element_ids` mapped through the emit
        layer's element→rank ownership). When given and non-empty on a
        :class:`ContactRecord`, the owner pick is **exact** (ADR 0092
        INV-1, second amendment): owner = the rank owning the most
        backing solids, ties to the lowest rank. The node tally — and its
        undecidable-tie refusal — is then bypassed entirely. Ignored for
        a :class:`ContactPlaneRecord` (no master surface).

    Returns
    -------
    ContactOwnership
        ``owner_rank`` per ADR 0092 INV-1 (master-side, element-exact
        when ``master_element_ranks`` is given, node majority otherwise;
        slave-side for a :class:`ContactPlaneRecord`), and
        ``ghost_node_ids`` per INV-2 (whole non-native interface, sorted
        + deduplicated).
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
        element_ranks = (
            tuple(int(r) for r in master_element_ranks)
            if master_element_ranks is not None else ()
        )
        if element_ranks:
            # Element-exact pick: the rank owning the most of the master
            # surface's backing solids. Elements live on exactly one rank
            # each, so this is the real ownership property `-kn auto`
            # needs — no proxy, no undecidable case. A count tie can only
            # mean the master surface itself is cut across ranks (an
            # INV-4 violation the caller checks separately); break to the
            # lowest rank for deterministic emission.
            counts: "dict[int, int]" = {}
            for rank in element_ranks:
                counts[rank] = counts.get(rank, 0) + 1
            best = max(counts.values())
            owner_rank = min(
                rank for rank, count in counts.items() if count == best
            )
        else:
            owner_rank = _majority_owner(master_ids, owning_ranks)
        interface_ids = sorted(set(master_ids) | set(slave_ids))
    else:
        raise TypeError(
            "resolve_contact_ownership: expected a ContactRecord or "
            f"ContactPlaneRecord, got {type(record).__name__}"
        )

    ghost_ids = _ghost_nodes(interface_ids, owner_rank, owning_ranks)
    return ContactOwnership(owner_rank=owner_rank, ghost_node_ids=ghost_ids)
