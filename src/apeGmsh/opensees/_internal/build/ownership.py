"""ownership: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Any, Iterable, Iterator, Mapping, Sequence, cast, overload

import numpy as np

from ..types import Element
from ..tag_resolution import MISSING_FEM_ELEMENT_ID
from .errors import BridgeError
from .records import InitialStressRecord, StageRecord
from .pg_expand import (
    ElementPlanRows,
    expand_pg_to_elements,
    expand_spec_to_elements,
    resolve_ladruno_membrane_context,
)


if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh._kernel.records._partitions import PartitionRecord


# ---------------------------------------------------------------------------
# Partition-aware emission (ADR 0027, P4) — closes the unpartitioned-only
# assumption baked into the original emit pipeline.
# ---------------------------------------------------------------------------


def is_partitioned(fem: "FEMData") -> bool:
    """True iff the FEM snapshot carries more than one partition.

    Single-partition (or unpartitioned) FEMs use the flat emit path —
    bit-identical to the pre-ADR 0027 behaviour, with no
    ``partition_open`` / ``partition_close`` calls and no runtime shim.
    Multi-partition FEMs route through the per-rank fan-out helpers
    that emit per-rank-bracketed output and replicate cross-partition
    MP constraints per ADR 0027.
    """
    parts = getattr(fem, "partitions", None)
    if parts is None:
        return False
    try:
        return len(parts) > 1
    except TypeError:
        return False


def runtime_rank_from_partition_record(
    record: "PartitionRecord", index: int,
) -> int:
    """Return the 0-based OpenSeesMP runtime rank for a ``PartitionRecord``.

    Gmsh-side :attr:`PartitionRecord.id` is **1-based** (``P1, P2, ...``);
    the OpenSeesMP runtime ``getPID()`` (and MPI rank) is **0-based**
    (``0, 1, ...``).  The conversion is the ``enumerate`` index over
    ``fem.partitions`` (which iterates in sorted Gmsh-id order, so the
    assignment is stable and deterministic) — **NOT** ``record.id - 1``.

    This function is the **single source of truth** for the conversion.
    Any future caller that converts the seam elsewhere violates the
    contract — it should call this helper instead.  The body is
    intentionally trivial: the value is the documentation.  A
    maintainer tempted to change to ``record.id - 1`` only has to
    update one function instead of every call site.

    Parameters
    ----------
    record
        The :class:`PartitionRecord` at this position in
        ``fem.partitions``.  Not consulted today — present so the
        call site reads with intent (``runtime_rank_from_partition_record(rec, idx)``
        rather than a bare ``idx``) and so a future change to the
        convention has a record-shaped lever available without
        touching call sites.
    index
        The ``enumerate`` index over ``fem.partitions``.

    Returns
    -------
    int
        The 0-based OpenSeesMP runtime rank.
    """
    del record  # intentionally unused — see docstring
    return index


class SortedIntToInt:
    """Compact ``{int: int}`` map backed by two sorted int64 arrays (B2).

    ADR 0065 v2 / plan_emit_memory_columnar.md B2: a positional array
    replacement for the ``dict[int, int]`` ownership maps
    (``element_owner`` = ``{fem_eid: rank}``, ``primary_owner`` =
    ``{node_id: rank}``) that showed hot in the M0 emit peak (each Python
    ``dict`` boxes both key and value per entry — ~90 B/entry). Here the
    resident form is two int64 arrays; point ``get`` is a
    ``searchsorted``, and :meth:`translate_ranks` resolves a whole array
    of keys in one vectorised pass (used for the per-element owner lookup
    in :class:`LazyRankBuckets`).

    Duck-typed to the old dict for the point-lookup consumers: :meth:`get`
    (``int`` or ``None`` — unknown key stays ``None``, never
    ``KeyError``), :meth:`__contains__`, :meth:`items` (ascending-key
    order), :meth:`__len__`. Read-only.
    """

    __slots__ = ("_keys", "_vals")

    def __init__(self, keys: "np.ndarray", vals: "np.ndarray") -> None:
        # ``keys`` must be sorted+unique; callers below build them that
        # way (np.unique / first-seen dedup). ``vals`` is parallel.
        self._keys = keys
        self._vals = vals

    def __len__(self) -> int:
        return int(self._keys.shape[0])

    def _find(self, key: int) -> int:
        k = self._keys
        i = int(np.searchsorted(k, key))
        if i < k.shape[0] and int(k[i]) == key:
            return i
        return -1

    # ``dict.get``-shaped overloads: an explicit non-None default means
    # the result is always an int — encoding the real contract so e.g.
    # ``owner.get(reverse.get(tag, -1))`` type-checks without a cast.
    @overload
    def get(self, key: int) -> "int | None": ...

    @overload
    def get(self, key: int, default: int) -> int: ...

    def get(self, key: int, default: "int | None" = None) -> "int | None":
        i = self._find(int(key))
        if i < 0:
            return default
        return int(self._vals[i])

    def __getitem__(self, key: int) -> int:
        i = self._find(int(key))
        if i < 0:
            raise KeyError(key)
        return int(self._vals[i])

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (int, np.integer)):
            return False
        return self._find(int(key)) >= 0

    def __iter__(self) -> "Iterator[int]":
        # Dict semantics: iterating yields KEYS. Without this, Python
        # falls back to the legacy __getitem__(0..) iteration protocol
        # and raises KeyError on the first missing index.
        return self.keys()

    def __eq__(self, other: object) -> bool:
        # Mapping-equality against a plain dict (or another instance) —
        # the unit tests assert ``owner_map == {nid: rank, ...}``.
        if isinstance(other, SortedIntToInt):
            return (
                self._keys.shape == other._keys.shape
                and bool(np.array_equal(self._keys, other._keys))
                and bool(np.array_equal(self._vals, other._vals))
            )
        if isinstance(other, dict):
            return len(other) == len(self) and all(
                other.get(k) == v for k, v in self.items()
            )
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # mutable-array-backed; unhashable like dict

    def items(self) -> "Iterator[tuple[int, int]]":
        keys = self._keys
        vals = self._vals
        for i in range(keys.shape[0]):
            yield int(keys[i]), int(vals[i])

    def keys(self) -> "Iterator[int]":
        for i in range(self._keys.shape[0]):
            yield int(self._keys[i])

    def values(self) -> "Iterator[int]":
        for i in range(self._vals.shape[0]):
            yield int(self._vals[i])

    def get_many(
        self, keys: "Any",
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Vectorised :meth:`get`: one ``searchsorted`` for a whole id array.

        Returns ``(vals, found)``, both parallel to ``keys`` and in the
        caller's order (``keys`` need not be sorted). ``found[i]`` is
        True iff ``keys[i]`` is a key, and then ``vals[i]`` equals
        ``self.get(keys[i])``. ``vals`` is undefined where ``found`` is
        False, so a caller reads it only through the mask; a miss is
        never folded into a sentinel value.
        """
        q = np.asarray(keys, dtype=np.int64).reshape(-1)
        k = self._keys
        vals = np.zeros(q.shape[0], dtype=np.int64)
        if k.shape[0] == 0 or q.shape[0] == 0:
            return vals, np.zeros(q.shape[0], dtype=bool)
        pos = np.searchsorted(k, q)
        in_range = pos < k.shape[0]
        pos_c = np.where(in_range, pos, 0)
        found = in_range & (k[pos_c] == q)
        vals[found] = self._vals[pos_c[found]]
        return vals, found

    def translate_ranks(
        self, keys: "np.ndarray", missing: int = -1,
    ) -> "np.ndarray":
        """Vectorised lookup: ``int64[M]`` keys → ``int64[M]`` values.

        Unknown keys map to ``missing`` (default -1). Lets the per-rank
        bucketing resolve every element's owner rank in one call instead
        of a per-element ``.get`` over a Python dict.
        """
        q = np.asarray(keys, dtype=np.int64)
        k = self._keys
        out = np.full(q.shape[0], missing, dtype=np.int64)
        if k.shape[0] == 0 or q.shape[0] == 0:
            return out
        pos = np.searchsorted(k, q)
        in_range = pos < k.shape[0]
        pos_c = np.where(in_range, pos, 0)
        hit = in_range & (k[pos_c] == q)
        out[hit] = self._vals[pos_c[hit]]
        return out


def node_index_lookup(ids: "Any") -> "SortedIntToInt":
    """Columnar ``{node_id: coord_row}`` lookup (ADR 0100 D3).

    Replaces the per-emit ``{int(nid): i for i, nid in enumerate(ids)}``
    dict (~84 B/node of boxed pairs; built up to twice, co-resident, on
    staged partitioned decks) with two int64 arrays — 16 B/node.

    ``fem.nodes.ids`` is object-dtype and entity-grouped — **unsorted on
    every partitioned mesh** (``_fem_extract`` takes
    ``gmsh.model.mesh.getNodes()`` verbatim) — so a raw ``searchsorted``
    over the ids would be wrong; the ids are int64-ified and sorted once,
    and lookups go through the argsort permutation.  Missing-id detection
    is preserved exactly: ``.get`` answers ``None`` on a miss and
    ``[...]`` raises ``KeyError``, like the dict it replaces.  There is
    deliberately NO dict fallback — a fallback would fire on exactly the
    partitioned meshes D3 targets (ADR 0100 §D3, review finding 4).

    LAST-wins on a duplicated id, like the dict it replaces (a later
    ``enumerate`` key overwrites an earlier one).  Broker node ids are
    unique today, but "unique today" is exactly what
    :meth:`FemToOpsTagMap._find` assumed before the review caught its
    silent first-wins flip (ADR 0065 v2 hardening) — same defect class,
    defended the same way: the stable argsort preserves enumerate order
    among equals, and keeping the last row of each duplicate run
    reproduces the dict.  (The dedup lives at build time because
    :class:`SortedIntToInt` contracts sorted+unique keys, where
    FemToOpsTagMap keeps duplicates and resolves them per lookup with
    ``side="right" - 1``.)
    """
    ids_i64 = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids_i64, kind="stable")
    keys = ids_i64[order]
    if keys.shape[0]:
        keep = np.empty(keys.shape[0], dtype=bool)
        keep[:-1] = keys[:-1] != keys[1:]
        keep[-1] = True
        if not keep.all():
            keys = keys[keep]
            order = order[keep]
    return SortedIntToInt(
        keys, order.astype(np.int64, copy=False),
    )


_EMPTY_INT64 = np.empty((0,), dtype=np.int64)


class SortedIntSet:
    """Compact read-only ``set[int]`` backed by one sorted-unique int64
    array (ADR 0100 D2).

    Replaces the per-rank ``set[int]`` membership sets
    (``rank_owned_nodes`` / ``rank_primary_nodes``, ~41 B/node each,
    Σ over ranks growing with np through the shared-boundary factor)
    with 8 B/node.  Duck-typed to the surface the partitioned emit
    passes actually use: ``in`` / ``len`` / truthiness / iteration
    (ascending), plus the two set-algebra forms the staged pass needs —
    :meth:`intersection_sorted` (``sorted(self & other)``) and
    :meth:`isdisjoint`.  Read-only.
    """

    __slots__ = ("_keys",)

    def __init__(self, keys: "np.ndarray") -> None:
        # ``keys`` must be sorted + unique; build via :meth:`from_ids`.
        self._keys = keys

    @classmethod
    def from_ids(cls, ids: "Any") -> "SortedIntSet":
        """Build from any id iterable / array; duplicates collapse."""
        if isinstance(ids, np.ndarray):
            arr = ids.astype(np.int64, copy=False)
        else:
            arr = np.fromiter((int(n) for n in ids), dtype=np.int64)
        if arr.shape[0] == 0:
            return cls(_EMPTY_INT64)
        return cls(np.unique(arr))

    def __len__(self) -> int:
        return int(self._keys.shape[0])

    def __bool__(self) -> bool:
        return bool(self._keys.shape[0] > 0)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (int, np.integer)):
            return False
        k = self._keys
        i = int(np.searchsorted(k, key))
        return i < k.shape[0] and int(k[i]) == int(key)

    def __iter__(self) -> "Iterator[int]":
        return iter(self._keys.tolist())

    def _query_array(self, other: "Any") -> "np.ndarray":
        if isinstance(other, np.ndarray):
            arr = other.astype(np.int64, copy=False)
        else:
            arr = np.fromiter((int(n) for n in other), dtype=np.int64)
        if arr.shape[0] == 0:
            return _EMPTY_INT64
        return np.unique(arr)

    def _member_mask(self, q_sorted: "np.ndarray") -> "np.ndarray":
        k = self._keys
        pos = np.searchsorted(k, q_sorted)
        in_range = pos < k.shape[0]
        pos_c = np.where(in_range, pos, 0)
        return cast("np.ndarray", in_range & (k[pos_c] == q_sorted))

    def intersection_sorted(self, other: "Iterable[int]") -> "list[int]":
        """``sorted(self & set(other))`` — one vectorised pass."""
        q = self._query_array(other)
        if q.shape[0] == 0 or self._keys.shape[0] == 0:
            return []
        return cast("list[int]", q[self._member_mask(q)].tolist())

    def isdisjoint(self, other: "Iterable[int]") -> bool:
        q = self._query_array(other)
        if q.shape[0] == 0 or self._keys.shape[0] == 0:
            return True
        return not bool(self._member_mask(q).any())


def bucket_primary_nodes_by_rank(
    primary_owner: "SortedIntToInt",
    seed_ranks: "Iterable[int]",
) -> "dict[int, SortedIntSet]":
    """Per-rank PRIMARY-owned node-id sets, columnar (ADR 0100 D2).

    Same key set and per-rank membership as the
    ``{rank: set()}`` seed + ``setdefault(owner_rank).add(nid)`` fill
    over ``primary_owner.items()`` it replaces: every seed rank is
    present (possibly empty), and any owner rank absent from the seed
    is created — but each per-rank set is a sorted int64 array instead
    of boxed ints.
    """
    empty = SortedIntSet(_EMPTY_INT64)
    out: "dict[int, SortedIntSet]" = {int(r): empty for r in seed_ranks}
    # Same-module access to the columnar map's parallel arrays: keys
    # ascend, so each per-rank mask subset is already sorted + unique.
    keys = primary_owner._keys
    vals = primary_owner._vals
    for r in np.unique(vals).tolist():
        out[int(r)] = SortedIntSet(keys[vals == r])
    return out


class NodePartitionOwners:
    """Compact ``{node_id: set[rank]}`` map in CSR layout (B2).

    ADR 0065 v2 / plan_emit_memory_columnar.md B2: replaces the
    ``dict[int, set[int]]`` returned by :func:`build_node_partition_owners`
    — the single largest build-side emit-peak term (one Python ``set``
    per node, ~315 B/hex at box-64-rank scale) even though the vast
    majority of nodes belong to exactly one partition.

    The resident form is three int64 arrays (compressed sparse row):
    ``_node_ids`` (sorted, unique), ``_offsets`` (``N+1``), and
    ``_ranks`` (the flat owner-rank runs, each run ascending). A node's
    ranks are ``_ranks[_offsets[i]:_offsets[i+1]]``. Storage is ~2 int64
    per node in the common single-owner case, versus a full ``set``
    object.

    Duck-typed to the old ``dict[int, set[int]]`` for its consumers:
    :meth:`get` returns the node's ranks as a **tuple** (built transiently
    only when queried — the MP-constraint replication paths query a
    handful of constraint nodes, not every node), :meth:`items` yields
    ``(node_id, rank_tuple)``, :meth:`__contains__` / :meth:`__len__`.
    :meth:`primary_owner` builds the ``{node_id: min(rank)}`` reduction
    (:func:`primary_owner_map`) vectorised from the CSR arrays.
    """

    __slots__ = ("_node_ids", "_offsets", "_ranks", "_resolved")

    def __init__(
        self,
        node_ids: "np.ndarray",
        offsets: "np.ndarray",
        ranks: "np.ndarray",
    ) -> None:
        self._node_ids = node_ids
        self._offsets = offsets
        self._ranks = ranks
        # Populated by :meth:`resolve_many`; see there for why this is
        # bounded by the constraint set rather than the model.
        self._resolved: dict[int, frozenset[int]] = {}

    def __len__(self) -> int:
        return int(self._node_ids.shape[0])

    def _find(self, node_id: int) -> int:
        nid = self._node_ids
        i = int(np.searchsorted(nid, node_id))
        if i < nid.shape[0] and int(nid[i]) == node_id:
            return i
        return -1

    def get(
        self,
        node_id: int,
        default: "Iterable[int]" = (),
    ) -> "frozenset[int]":
        """Return the node's owning ranks as a transient ``frozenset``.

        ``frozenset`` (not ``tuple``) because the MP-constraint
        replication consumers run set intersections against the result
        (``intersection & owners`` in ``_canonical_host_rank`` /
        ``_canonical_coupling_rank``). Unknown node → ``default``
        coerced to a ``frozenset`` (callers pass ``set()`` / ``()``,
        both of which behave identically to the old ``dict.get``
        contract under ``in`` / ``&`` / ``sorted`` / iteration).
        """
        i = self._find(int(node_id))
        if i < 0:
            return frozenset(default)
        lo = int(self._offsets[i])
        hi = int(self._offsets[i + 1])
        return frozenset(int(r) for r in self._ranks[lo:hi])

    def resolve_many(
        self, node_ids: "Iterable[int]",
    ) -> "Mapping[int, frozenset[int]]":
        """Resolve many node ids with ONE vectorised ``searchsorted``.

        :meth:`get` costs a *scalar* ``np.searchsorted`` per call — the
        numpy dispatch (``searchsorted`` → ``_wrapfunc`` →
        ``ndarray.searchsorted``) is ~1.4 us, versus ~90 ns for the
        ``dict`` hash this class replaced. The MP-constraint planner
        queries one node at a time inside a Python loop, so that
        per-call overhead dominated partitioned emit (ADR 0065 v2 B2
        traded ~40x per-lookup cost for the memory win).

        Callers that know their key set up front resolve it here in a
        single vectorised probe and then index the result at hash speed.

        Resolutions are **cached on the instance** and only the ids not
        already known are probed. The planner runs once per rank over
        the same constraint records, and a node's owners do not depend
        on which rank is asking — so ranks 1..N-1 are pure cache hits
        and the ``ranks``-fold rebuild disappears.

        The cache holds one ``frozenset`` per *constraint-referenced*
        node, not per node in the model — the distinction B2 rests on,
        and what this class's docstring already assumes ("the
        MP-constraint replication paths query a handful of constraint
        nodes, not every node").

        Returns the cache itself; treat it as **read-only**. It may hold
        more keys than were requested, which is harmless — consumers
        query the ids they care about.
        """
        cache = self._resolved
        missing = [n for n in map(int, node_ids) if n not in cache]
        if not missing:
            return cache

        nid = self._node_ids
        uniq = np.unique(np.asarray(missing, dtype=np.int64))
        if nid.shape[0] == 0:
            cache.update({int(k): frozenset() for k in uniq.tolist()})
            return cache

        pos = np.searchsorted(nid, uniq)
        # Clip before gathering so the `pos == len` past-the-end case
        # (a key above every known id) can't index out of bounds; the
        # `hit` mask discards it either way.
        safe = np.minimum(pos, nid.shape[0] - 1)
        hit = nid[safe] == uniq

        off = self._offsets
        ranks = self._ranks
        keys = uniq.tolist()
        starts = off[safe].tolist()
        stops = off[safe + 1].tolist()
        hits = hit.tolist()

        # Intern by owner-run: the overwhelming majority of nodes belong
        # to exactly one partition, so across a whole model there are
        # only a handful of DISTINCT owner sets (one per rank, plus the
        # few boundary combinations). Sharing one frozenset per distinct
        # run keeps the cache's cost at ~one dict entry per node instead
        # of one set object per node — which is the term B2 set out to
        # remove in the first place.
        interned: dict[tuple[int, ...], frozenset[int]] = {}
        empty: frozenset[int] = frozenset()
        for k in range(len(keys)):
            if not hits[k]:
                cache[keys[k]] = empty
                continue
            run = tuple(ranks[starts[k]:stops[k]].tolist())
            shared = interned.get(run)
            if shared is None:
                shared = frozenset(run)
                interned[run] = shared
            cache[keys[k]] = shared
        return cache

    def __getitem__(self, node_id: int) -> "frozenset[int]":
        i = self._find(int(node_id))
        if i < 0:
            raise KeyError(node_id)
        lo = int(self._offsets[i])
        hi = int(self._offsets[i + 1])
        return frozenset(int(r) for r in self._ranks[lo:hi])

    def __contains__(self, node_id: object) -> bool:
        if not isinstance(node_id, (int, np.integer)):
            return False
        return self._find(int(node_id)) >= 0

    def __iter__(self) -> "Iterator[int]":
        # Dict semantics: iterating yields the node-id KEYS.
        for i in range(self._node_ids.shape[0]):
            yield int(self._node_ids[i])

    def values(self) -> "Iterator[frozenset[int]]":
        off = self._offsets
        ranks = self._ranks
        for i in range(self._node_ids.shape[0]):
            lo = int(off[i])
            hi = int(off[i + 1])
            yield frozenset(int(r) for r in ranks[lo:hi])

    def items(self) -> "Iterator[tuple[int, frozenset[int]]]":
        nid = self._node_ids
        off = self._offsets
        ranks = self._ranks
        for i in range(nid.shape[0]):
            lo = int(off[i])
            hi = int(off[i + 1])
            yield int(nid[i]), frozenset(int(r) for r in ranks[lo:hi])

    def primary_owner(self) -> "SortedIntToInt":
        """Return ``{node_id: min(rank)}`` as a :class:`SortedIntToInt`.

        Vectorised: each node's owner runs are ascending (built that way
        in :func:`build_node_partition_owners`), so the primary (lowest)
        rank is the run's first element — ``_ranks[_offsets[:-1]]``. No
        per-node Python ``min`` over a set.
        """
        n = self._node_ids.shape[0]
        if n == 0:
            empty = np.empty((0,), dtype=np.int64)
            return SortedIntToInt(empty, empty)
        firsts = self._ranks[self._offsets[:-1]]
        return SortedIntToInt(
            self._node_ids, np.asarray(firsts, dtype=np.int64)
        )


def build_node_partition_owners(fem: "FEMData") -> "NodePartitionOwners":
    """Return ``{node_tag: set[rank_id]}`` covering every owning rank.

    A node may belong to multiple partitions (boundary / shared nodes
    that the partitioner replicates across ranks).  The returned dict
    is keyed by every node id that appears in any ``PartitionRecord``;
    unknown / unowned nodes are absent from the map.

    ``rank_id`` is the **0-based runtime rank** — matching
    ``OpenSeesMP::getPID()`` — derived via ``enumerate`` over
    ``fem.partitions`` (which already iterates in sorted Gmsh-id
    order, so the assignment is stable and deterministic).  The
    broker's Gmsh-side 1-based ``PartitionRecord.id`` is preserved
    on the records themselves; only the runtime-rank seam (this
    helper, ``build_element_partition_owner``, and the
    ``partition_rank`` parameter passed to per-rank fan-out helpers)
    is 0-based.

    ADR 0065 v2 / plan_emit_memory_columnar.md B2: returns a compact
    CSR-backed :class:`NodePartitionOwners` instead of a
    ``dict[int, set[int]]`` (one Python ``set`` per node was the largest
    build-side emit-peak term). Built vectorised - concatenate every
    partition's ``(node_id, rank)`` pairs, sort lexicographically by
    ``(node_id, rank)`` so each node's owner run is ascending (the
    :meth:`NodePartitionOwners.primary_owner` reduction relies on that),
    then compress to per-node offsets. Duck-typed to the old dict for its
    ``.get`` / ``.items`` / ``in`` consumers.
    """
    parts = getattr(fem, "partitions", None)
    empty = np.empty((0,), dtype=np.int64)
    if parts is None:
        return NodePartitionOwners(empty, np.zeros(1, dtype=np.int64), empty)
    nid_blocks: "list[np.ndarray]" = []
    rank_blocks: "list[np.ndarray]" = []
    for idx, rec in enumerate(parts):
        rank = runtime_rank_from_partition_record(rec, idx)
        nids = np.asarray(rec.node_ids, dtype=np.int64)
        if nids.shape[0] == 0:
            continue
        nid_blocks.append(nids)
        rank_blocks.append(np.full(nids.shape[0], rank, dtype=np.int64))
    if not nid_blocks:
        return NodePartitionOwners(empty, np.zeros(1, dtype=np.int64), empty)
    all_nids = np.concatenate(nid_blocks)
    all_ranks = np.concatenate(rank_blocks)
    # Lexsort by (node_id, rank): primary key last in np.lexsort. This
    # groups by node and orders each group's ranks ascending, so the
    # first rank in a node's run is its lowest (primary) owner.
    order = np.lexsort((all_ranks, all_nids))
    sorted_nids = all_nids[order]
    sorted_ranks = all_ranks[order]
    # Compress to unique node ids + per-node run offsets. A (node, rank)
    # pair can repeat only if a node appears twice in one partition - the
    # broker de-dups upstream, but guard anyway by dropping exact dup rows.
    uniq_pairs = np.ones(sorted_nids.shape[0], dtype=bool)
    if sorted_nids.shape[0] > 1:
        same = (sorted_nids[1:] == sorted_nids[:-1]) & (
            sorted_ranks[1:] == sorted_ranks[:-1]
        )
        uniq_pairs[1:] = ~same
    sorted_nids = sorted_nids[uniq_pairs]
    sorted_ranks = sorted_ranks[uniq_pairs]
    node_ids, starts = np.unique(sorted_nids, return_index=True)
    offsets = np.empty(node_ids.shape[0] + 1, dtype=np.int64)
    offsets[:-1] = starts
    offsets[-1] = sorted_ranks.shape[0]
    return NodePartitionOwners(node_ids, offsets, sorted_ranks)


def primary_owner_map(node_owners: "NodePartitionOwners") -> "SortedIntToInt":
    """Reduce a multi-rank owner map to ``{node_tag: primary_rank}``.

    **Additive** nodal quantities — ``mass`` lines and pattern ``load``
    lines — must emit on exactly ONE rank: OpenSeesMP merges shared-node
    equations across ranks and SUMS each domain's nodal-mass / load
    contribution during assembly, so the per-owner replication that is
    correct for idempotent lines (``node`` / ``fix`` / ``sp``)
    double-counts interface nodes. Run-verified on an 8-partition model:
    81 massed nodes emitted 177 ``mass`` lines and the partitioned
    transient diverged from the byte-identical sequential run by ~100 %
    of peak velocity; with one ``mass`` line per node the two runs agree
    to machine precision (~5e-15 of peak).

    The primary rank is the **lowest** owning runtime rank — an
    arbitrary but deterministic choice (ADR 0027 §"Tag determinism"
    spirit: same snapshot → same deck bytes).

    ADR 0065 v2 / plan_emit_memory_columnar.md B2: delegates to
    :meth:`NodePartitionOwners.primary_owner`, which reads the lowest
    rank per node straight off the CSR arrays (no per-node ``min`` over a
    Python set) and returns a compact :class:`SortedIntToInt` rather than
    a ``dict[int, int]``.
    """
    return node_owners.primary_owner()


def build_element_partition_owner(fem: "FEMData") -> "SortedIntToInt":
    """Return ``{element_tag: rank_id}`` — each element lives on exactly one rank.

    Unlike nodes, the partitioner gives each element to a single rank
    (interface elements typically belong to a "interface partition" in
    Gmsh; here we honour whatever the broker recorded).  Element ids
    not present in any partition are absent from the map; callers
    interpret "missing key" as "unowned" and either skip emission or
    fall back to an explicit policy.

    ``rank_id`` is the **0-based runtime rank** — matching
    ``OpenSeesMP::getPID()`` — derived via ``enumerate`` over
    ``fem.partitions`` (sorted Gmsh-id order).  See the docstring
    on :func:`build_node_partition_owners` for the rationale.

    ADR 0065 v2 / plan_emit_memory_columnar.md B2: returns a compact
    :class:`SortedIntToInt` (two int64 arrays) instead of a
    ``dict[int, int]`` — the per-element boxed dict was a hot build-side
    emit-peak term. First-seen partition wins for a duplicated element
    (deterministic tiebreak), preserved here by keeping the FIRST
    occurrence when de-duplicating the concatenated ``(eid, rank)`` pairs
    in partition-iteration order.
    """
    parts = getattr(fem, "partitions", None)
    empty = np.empty((0,), dtype=np.int64)
    if parts is None:
        return SortedIntToInt(empty, empty)
    eid_blocks: "list[np.ndarray]" = []
    rank_blocks: "list[np.ndarray]" = []
    for idx, rec in enumerate(parts):
        rank = runtime_rank_from_partition_record(rec, idx)
        eids = np.asarray(rec.element_ids, dtype=np.int64)
        if eids.shape[0] == 0:
            continue
        eid_blocks.append(eids)
        rank_blocks.append(np.full(eids.shape[0], rank, dtype=np.int64))
    if not eid_blocks:
        return SortedIntToInt(empty, empty)
    all_eids = np.concatenate(eid_blocks)
    all_ranks = np.concatenate(rank_blocks)
    # ``np.unique`` on a stably-sorted key keeps the first occurrence in
    # the ORIGINAL order for ``return_index`` — but only if the sort is
    # stable and we pick the min index per group. Do it explicitly: sort
    # by eid (stable), then for each unique eid take the first row, which
    # — because concatenation is in partition-iteration order and the sort
    # is stable — is the first partition that recorded the element (the
    # old ``setdefault`` first-seen tiebreak).
    order = np.argsort(all_eids, kind="stable")
    sorted_eids = all_eids[order]
    sorted_ranks = all_ranks[order]
    uniq_eids, starts = np.unique(sorted_eids, return_index=True)
    uniq_ranks = sorted_ranks[starts]
    return SortedIntToInt(uniq_eids, uniq_ranks)


def node_pair_rank(
    spec: Element, i_tag: int, j_tag: int,
    node_owners: "NodePartitionOwners",
) -> int:
    """The one rank that writes a node-pair element (ADR 0120 D1).

    The lowest rank that holds both endpoints. A pair whose endpoints share
    no rank (two mesh nodes on different partitions) raises: writing it
    would need a ghost endpoint, which is not designed here.
    """
    ri, rj = node_owners.get(i_tag), node_owners.get(j_tag)
    common = ri & rj
    if not common:
        raise BridgeError(
            f"apeSees: the node-pair {type(spec).__name__} between nodes "
            f"{i_tag} (ranks {sorted(ri)}) and {j_tag} (ranks {sorted(rj)}) "
            f"has no rank that holds both endpoints. A node-pair element is "
            f"written on the rank of its endpoints (ADR 0120 D1); one between "
            f"two mesh nodes on different partitions would need a ghost "
            f"endpoint, which is not supported. Reach the second mesh node "
            f"through a decoupled node tied to it by equal_dof, or emit "
            f"serial."
        )
    return min(common)


def macro_element_rank(
    spec: Element, node_tags: Sequence[int],
    node_owners: "NodePartitionOwners",
) -> int:
    """The one rank that writes a multi-node macro-element (e.g. LadrunoMembrane)."""
    common: set[int] | None = None
    for n in node_tags:
        rn = set(node_owners.get(int(n)))
        common = rn if common is None else (common & rn)
    if not common:
        raise BridgeError(
            f"apeSees: {type(spec).__name__} nodes {tuple(node_tags)} do not "
            f"share a single owning partition rank; macro-elements must reside "
            f"wholly on one rank or be emitted serially."
        )
    return min(common)


class FemToOpsTagMap:
    """Columnar ``{fem_eid: ops_element_tag}`` map (B3).

    ADR 0065 v2 / plan_emit_memory_columnar.md B2+B3: the old form was a
    ``dict[int, int]`` built by a comprehension over the whole element
    plan (``{eid: tag for _, sub in plan for eid, _conn, tag in sub}``) —
    one boxed ``int`` key + one boxed ``int`` value per element (~160
    B/element resident, plus the transient boxed ``(eid, conn, tag)``
    triple the comprehension walked). At LOH.1 scale that dict is ~1 GB.

    This map keeps its resident form as two int64 arrays straight off the
    columnar plan (:class:`ElementPlanRows`) — no per-element Python
    boxing survives construction. Point lookups (``get`` / ``in``) use
    ``np.searchsorted`` on a sorted copy of the eids; a vectorised
    :meth:`translate` resolves a whole ``-ele`` list in one call.

    The public read surface is duck-typed to the old dict for the
    unconverted consumers: :meth:`get` (returns ``int`` or ``None`` —
    unknown-eid stays ``None``, never ``KeyError``), :meth:`__contains__`,
    :meth:`items` (plan order, matching the old insertion order), and
    :meth:`__len__`.

    Node-pair specs carry the :data:`MISSING_FEM_ELEMENT_ID` (-1)
    sentinel; it is filtered out at construction exactly as the old
    comprehension's ``if eid != MISSING_FEM_ELEMENT_ID`` guard did, so it
    never becomes a key.

    The map also carries the plan's σ_zz capability (``sigma_zz_blocks``):
    :meth:`from_plan` is the only place in the emit pipeline where an
    element's ops tags and its typed spec (element class + ``plane_type``
    + material) are both in hand, and the recorder fan-out — the consumer
    that needs the answer — already receives this map.  The blocks alias
    the very tag arrays built for the map itself, so carrying them costs
    no extra allocation.
    """

    __slots__ = (
        "_eids", "_tags", "_order", "_sorted_eids", "_sorted_tags",
        "_sigma_zz_blocks", "_sigma_zz_tags",
    )

    def __init__(
        self, eids: "np.ndarray", tags: "np.ndarray",
        sigma_zz_blocks: "Iterable[np.ndarray]" = (),
    ) -> None:
        # ``eids`` / ``tags`` are parallel int64 arrays in PLAN order
        # (so ``items`` reproduces the old dict's insertion order). We
        # also keep a sorted view for O(log N) point membership.
        self._eids = eids
        self._tags = tags
        order = np.argsort(eids, kind="stable")
        self._order = order
        self._sorted_eids = eids[order]
        self._sorted_tags = tags[order]
        # Per-spec ops-tag blocks for the σ_zz-capable specs; concatenated
        # + sorted lazily on the first query so a model nobody records
        # σ_zz on never pays for them.
        self._sigma_zz_blocks: "tuple[np.ndarray, ...]" = tuple(
            sigma_zz_blocks
        )
        self._sigma_zz_tags: "np.ndarray | None" = None

    @classmethod
    def from_pairs(
        cls, pairs: "Iterable[tuple[int, int]]",
    ) -> "FemToOpsTagMap":
        """Build the map from ``(fem_eid, ops_tag)`` pairs.

        The H5 deck-replay path (``_internal/compose.py``) reconstructs
        the map from replayed element records rather than a live plan —
        pair order is preserved as the ``items()`` order, mirroring the
        old dict comprehension's insertion order.
        """
        eids_l: "list[int]" = []
        tags_l: "list[int]" = []
        for e, t in pairs:
            eids_l.append(int(e))
            tags_l.append(int(t))
        return cls(
            np.asarray(eids_l, dtype=np.int64),
            np.asarray(tags_l, dtype=np.int64),
        )

    @classmethod
    def from_plan(
        cls, plan: "Iterable[tuple[Element, ElementPlanRows]]",
    ) -> "FemToOpsTagMap":
        """Build the map from an :func:`allocate_element_tags` plan.

        Concatenates each spec's ``(eids, tag_start + arange)`` in plan
        order, dropping node-pair sentinel rows. Identical key/value set
        to the old ``{eid: tag}`` comprehension, in the same order.

        Each σ_zz-capable spec's tag block is retained by reference (it is
        already built for the map) so the recorder fan-out can gate the
        plane-strain stress promotion — see :meth:`all_sigma_zz_capable`.
        """
        from ..._element_capabilities import element_records_stress_zz

        eid_blocks: "list[np.ndarray]" = []
        tag_blocks: "list[np.ndarray]" = []
        sigma_zz_blocks: "list[np.ndarray]" = []
        for _spec, sub in plan:
            n = len(sub)
            if n == 0:
                continue
            eids = np.asarray(sub.eids, dtype=np.int64)
            if sub.tags is None:
                tags = sub.tag_start + np.arange(n, dtype=np.int64)
            else:
                tags = np.asarray(sub.tags, dtype=np.int64)
            # ADR 0049: drop the node-pair sentinel (-1) so it never
            # becomes a key — matches the old ``if eid != MISSING`` guard.
            keep = eids != MISSING_FEM_ELEMENT_ID
            if not keep.all():
                eids = eids[keep]
                tags = tags[keep]
            if eids.shape[0]:
                eid_blocks.append(eids)
                tag_blocks.append(tags)
                if element_records_stress_zz(_spec):
                    sigma_zz_blocks.append(tags)
        if not eid_blocks:
            empty = np.empty((0,), dtype=np.int64)
            return cls(empty, empty)
        return cls(
            np.concatenate(eid_blocks), np.concatenate(tag_blocks),
            sigma_zz_blocks,
        )

    def all_sigma_zz_capable(self, ops_tags: "Iterable[int]") -> bool:
        """True when every tag in ``ops_tags`` can record a real σ_zz.

        Empty ``ops_tags`` answers ``False`` — there is nothing to promote
        for, and the un-promoted token is always the safe choice.
        """
        if not self._sigma_zz_blocks:
            return False        # no capable spec at all — O(1) for 3-D models
        want = np.asarray(tuple(ops_tags), dtype=np.int64)
        if want.size == 0:
            return False
        if self._sigma_zz_tags is None:
            self._sigma_zz_tags = np.unique(
                np.concatenate(self._sigma_zz_blocks)
            )
        return bool(np.isin(want, self._sigma_zz_tags).all())

    def __len__(self) -> int:
        return int(self._eids.shape[0])

    def __bool__(self) -> bool:
        return int(self._eids.shape[0]) > 0

    def _find(self, eid: int) -> int:
        """Return the index of ``eid`` in the sorted view, or -1.

        LAST-wins on duplicate eids: ``side="right"`` lands one past the
        duplicate run, and the stable argsort in ``__init__`` preserved
        pair (plan) order among equals — so index ``i-1`` is the LATEST
        pair with that eid. This reproduces the old
        ``{eid: tag for ...}`` dict comprehension, where a later
        assignment overwrote an earlier one. Duplicate non-sentinel
        fem_eids are legal (overlapping element PGs fan the same FEM
        cell from two specs — e.g. a truss and a spring on one line
        PG), so the tie-break is behavior, not a corner case
        (review finding, ADR 0065 v2 hardening).
        """
        se = self._sorted_eids
        i = int(np.searchsorted(se, eid, side="right")) - 1
        if i >= 0 and int(se[i]) == eid:
            return i
        return -1

    def get(self, eid: int, default: "int | None" = None) -> "int | None":
        i = self._find(int(eid))
        if i < 0:
            return default
        return int(self._sorted_tags[i])

    def __contains__(self, eid: object) -> bool:
        if not isinstance(eid, (int, np.integer)):
            return False
        return self._find(int(eid)) >= 0

    def __getitem__(self, eid: int) -> int:
        i = self._find(int(eid))
        if i < 0:
            raise KeyError(eid)
        return int(self._sorted_tags[i])

    def items(self) -> "Iterator[tuple[int, int]]":
        """Yield ``(fem_eid, ops_tag)`` in plan (insertion) order."""
        eids = self._eids
        tags = self._tags
        for i in range(eids.shape[0]):
            yield int(eids[i]), int(tags[i])

    def keys(self) -> "Iterator[int]":
        for i in range(self._eids.shape[0]):
            yield int(self._eids[i])

    def values(self) -> "Iterator[int]":
        for i in range(self._tags.shape[0]):
            yield int(self._tags[i])

    def translate(self, eids: "np.ndarray") -> "np.ndarray":
        """Vectorised lookup: ``int64[M]`` fem eids → ``int64[M]`` tags.

        Unknown eids map to -1 (the caller decides whether that is an
        error). Used for the rayleigh / damping region ``-ele`` lists and
        staged ``remove_element`` translation, where the whole selection
        resolves in one ``searchsorted`` instead of a per-eid ``.get``
        loop over the boxed plan.
        """
        q = np.asarray(eids, dtype=np.int64)
        se = self._sorted_eids
        out = np.full(q.shape[0], -1, dtype=np.int64)
        if se.shape[0] == 0 or q.shape[0] == 0:
            return out
        # LAST-wins on duplicates, mirroring _find (side="right" - 1).
        pos = np.searchsorted(se, q, side="right") - 1
        in_range = pos >= 0
        pos_clamped = np.where(in_range, pos, 0)
        hit = in_range & (se[pos_clamped] == q)
        out[hit] = self._sorted_tags[pos_clamped[hit]]
        return out

    def inverse(self) -> "SortedIntToInt":
        """Columnar ``{ops_tag: fem_eid}`` reverse map (ADR 0100 R8).

        Replaces the per-stage
        ``{int(tag): int(eid) for eid, tag in self.items()}`` dict in
        ``_emit_stages_partitioned`` (measured 103-228 B/elem across
        the G0 cells ≈ ~5-12 GB at the 51.0 M-element incident) with
        two int64 arrays.  ``SortedIntToInt.get(tag, -1)`` preserves
        the sole consumer's ``-1`` miss default exactly.

        Tags are unique by construction (block allocation), but
        ``from_pairs`` replay accepts arbitrary pairs — so a duplicate
        tag keeps the LAST pair, exactly as the dict comprehension did.
        """
        order = np.argsort(self._tags, kind="stable")
        t = self._tags[order]
        e = self._eids[order]
        if t.shape[0]:
            # keep-last per duplicate run (stable sort preserved
            # items() order among equals).
            keep = np.empty(t.shape[0], dtype=bool)
            keep[:-1] = t[:-1] != t[1:]
            keep[-1] = True
            if not keep.all():
                t = t[keep]
                e = e[keep]
        return SortedIntToInt(t, e)


def compute_stage_ownership(
    stage_records: "tuple[StageRecord, ...]",
    elements: "Iterable[Element]",
    fem: "FEMData",
) -> "tuple[dict[int, int], dict[int, int]]":
    """Compute element + node ownership maps for Phase SSI-2.B.

    For each stage in registration order, walks the Element primitives
    whose ``pg=`` matches one of the stage's ``activated_pgs`` and
    assigns them (and their referenced nodes) to that stage.

    Returns
    -------
    element_owner : dict[int, int]
        ``{id(element_primitive): stage_index}`` — stage index is the
        position in ``stage_records``.  Element primitives not in any
        stage's activation set are absent from this map (global emit).
    node_owner : dict[int, int]
        ``{fem_node_id: stage_index}`` — node ID is the broker's FEM
        node id.  A node referenced by ANY global element stays global
        (absent from this map).  A node referenced ONLY by stage-bound
        elements is owned by the *lowest* stage index that references
        it.

    Raises
    ------
    BridgeError
        If a PG is activated by more than one stage (ambiguous
        ownership — first-write wins is unsafe; the user clearly
        meant something different).
    """
    # 1. Map PG name → owning stage index.  Raise on conflicts.
    pg_owner: dict[str, int] = {}
    for stage_idx, stage in enumerate(stage_records):
        for pg in stage.activated_pgs:
            if pg in pg_owner:
                raise BridgeError(
                    f"Stage {stage.name!r}: PG {pg!r} is activated by "
                    f"another stage (index {pg_owner[pg]}); PGs may be "
                    "activated by AT MOST one stage."
                )
            pg_owner[pg] = stage_idx

    # 2. Walk elements; map each spec to its owning stage (if any).
    element_owner: dict[int, int] = {}
    # We need to compute node ownership too — walk every element's
    # PG fan-out to collect (eid → set of stages that reference it).
    node_stages: dict[int, set[int]] = {}
    # And: nodes referenced by any GLOBAL element are global,
    # independent of which stages also reference them.
    global_nodes: set[int] = set()
    # Track which element-PG names actually appear on registered
    # Element primitives — used below to validate that every
    # ``s.activate(pgs=)`` PG matches at least one Element.
    seen_element_pgs: set[str] = set()
    for spec in elements:
        spec_pg = getattr(spec, "pg", None)
        if spec_pg:
            seen_element_pgs.add(spec_pg)
        owner_idx = pg_owner.get(spec_pg) if spec_pg else None
        if owner_idx is not None:
            element_owner[id(spec)] = owner_idx
        # ADR 0049: a node-pair spec (spec_pg is None) has owner_idx None, so
        # its two endpoints land in ``global_nodes`` below — forcing a shared
        # endpoint global so no stage can steal it (which would emit the node
        # inside a later stage block, after the global node-pair element that
        # references it -> forward-reference crash).  Node-pair is global-only
        # in v1.
        for _eid, conn in expand_spec_to_elements(fem, spec):
            for node_id in conn:
                if owner_idx is None:
                    global_nodes.add(int(node_id))
                else:
                    node_stages.setdefault(int(node_id), set()).add(owner_idx)
        if type(spec).__name__ == "LadrunoMembrane":
            _fr, supp_tags, hole_loops = resolve_ladruno_membrane_context(
                fem, spec, include_frames=False,
            )
            for node_id in (*supp_tags, *(n for h in hole_loops for n in h)):
                if owner_idx is None:
                    global_nodes.add(int(node_id))
                else:
                    node_stages.setdefault(int(node_id), set()).add(owner_idx)

    # 2b. Validate every activated PG matches at least one registered
    # Element primitive's ``pg=`` (red-team M1).  A typo'd PG name
    # would otherwise silently no-op — no domain_change, no element
    # emit, no error — leaving the user with a wrong-but-runnable
    # deck.
    unknown_pgs: list[tuple[str, str]] = []  # (stage_name, pg)
    for stage_idx, stage in enumerate(stage_records):
        for pg in stage.activated_pgs:
            if pg not in seen_element_pgs:
                unknown_pgs.append((stage.name, pg))
    if unknown_pgs:
        joined = ", ".join(
            f"stage {sn!r} → {pg!r}" for sn, pg in unknown_pgs
        )
        raise BridgeError(
            f"Stage activation references unknown element PGs: "
            f"{joined}.  Each ``s.activate(pgs=...)`` PG must match "
            f"at least one registered Element primitive's ``pg=``.  "
            f"Registered element PGs: {sorted(seen_element_pgs)}."
        )

    # 3. Assemble node_owner: a node is stage-bound to the lowest
    # stage index that references it iff it's NOT referenced by any
    # global element.
    node_owner: dict[int, int] = {}
    for node_id, stages in node_stages.items():
        if node_id in global_nodes:
            continue  # global element references it; stays global.
        node_owner[node_id] = min(stages)

    return element_owner, node_owner


def resolve_initial_stress_elements(
    rec: InitialStressRecord, fem: "FEMData",
) -> tuple[int, ...]:
    """Return the FEM element ids for an :class:`InitialStressRecord`.

    Exactly one of ``rec.pg`` / ``rec.elements`` is non-None
    (validated at the call site in :meth:`apeSees.initial_stress`).
    """
    if rec.elements is not None:
        return rec.elements
    if rec.pg is not None:
        return tuple(eid for eid, _conn in expand_pg_to_elements(fem, rec.pg))
    return ()
