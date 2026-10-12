"""pg_expand: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import weakref
from typing import Any, Iterable, Iterator

import numpy as np

from ..types import Element
from ..tag_resolution import MISSING_FEM_ELEMENT_ID
from .errors import BridgeError
from .records import FixRecord, MassRecord, SupportRecord


if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


# ---------------------------------------------------------------------------
# Element fan-out across a physical group
# ---------------------------------------------------------------------------

#: Per-snapshot memo for the PG fan-outs below (perf). A ``FEMData``
#: snapshot is immutable, so PG → elements / PG → nodes expansion is a
#: pure function of ``(fem, pg)`` — yet a single emit re-runs it several
#: times (ndf inference, tag allocation, ndf-compat validators, fix /
#: mass / load / recorder fan-outs), each pass re-materialising every
#: connectivity row into Python tuples. Keyed weakly on the snapshot so
#: entries die with it; snapshots that refuse weakrefs (exotic test
#: stubs) skip caching. Callers MUST NOT mutate the returned containers.
_PG_FANOUT_CACHE: "weakref.WeakKeyDictionary[object, dict[tuple[str, str], object]]" = (
    weakref.WeakKeyDictionary()
)


def _pg_fanout_cache_for(fem: "FEMData") -> "dict[tuple[str, str], object] | None":
    try:
        return _PG_FANOUT_CACHE.setdefault(fem, {})
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Columnar element containers (ADR 0065 v2 / plan_emit_memory_columnar.md B1+B4)
# ---------------------------------------------------------------------------
#
# WHY these exist: at LOH.1 scale (~6.7M hexes) the element fan-out and the
# element plan were the dominant emit-time RAM term (~4-6 GB). The old form
# built a Python ``list[tuple[int, tuple[int, ...]]]`` per fan-out and a
# ``list[tuple[int, tuple[int, ...], int]]`` per plan-spec, so every element
# cost one boxed ``(eid, conn)`` tuple plus one boxed connectivity ``tuple``
# holding ``npe`` boxed ``int`` objects (~54M boxed ints total). Worse, the
# fan-out result is memoised per snapshot (see :data:`_PG_FANOUT_CACHE`), so
# that boxed graph stayed *resident* for the whole emit.
#
# These two containers keep the RESIDENT form columnar (int64 arrays, straight
# from the FEMData group arrays which are already numpy) and box a row only
# TRANSIENTLY at iteration — the yielded ``(eid, conn_tuple[, tag])`` tuple is
# reclaimed the moment the consumer's loop body finishes with it. Every legacy
# consumer keeps working unchanged because the containers are duck-typed to the
# old list-of-tuples: iterating yields the same tuples in the same order, and
# ``__len__`` / ``__getitem__`` / ``bool`` behave like the old list.
#
# Byte-identity depends on ITERATION ORDER matching the old code exactly:
#   * fan-out order = groups in ``GroupResult`` order, elements in stored id
#     order within each group (the old ``for group in result: for eid, conn_row
#     in group`` walk). We concatenate per-group in that same order.
#   * mixed-npe groups: connectivity is an object-dtype array padded with -1.
#     We store a per-row true-width array so iteration slices each row to its
#     valid width and no -1 padding ever leaks into a yielded tuple — exactly
#     what the old ``tuple(int(n) for n in conn_row)`` produced (``conn_row``
#     there was already the un-padded per-type block row).


class PGElementFanout:
    """Columnar view of a physical-group element fan-out (B4).

    Duck-typed replacement for the old
    ``list[tuple[int, tuple[int, ...]]]`` returned by
    :func:`expand_pg_to_elements`. Holds the fan-out as int64 arrays
    and materialises ``(eid, conn_tuple)`` pairs only transiently at
    iteration, so the memoised resident form (kept in
    :data:`_PG_FANOUT_CACHE`) no longer carries ~54M boxed ints.

    Order is byte-identical to the legacy list: groups in
    ``GroupResult`` order, elements in stored id order within each
    group.

    Attributes
    ----------
    eids : numpy.ndarray
        ``int64[N]`` element ids in fan-out order.
    conn : numpy.ndarray
        Connectivity. Homogeneous fan-outs → ``int64[N, k]``. Mixed
        (multi-npe) fan-outs → object-dtype ``[N]`` of per-row int64
        arrays (already un-padded to their true width), so iteration
        slices are trivial and no -1 sentinel ever leaks.

    Notes
    -----
    Callers MUST treat this as read-only (it is memoised). The
    ``(eid, conn)`` tuples yielded by iteration are fresh per row.
    """

    __slots__ = ("eids", "conn", "_homogeneous")

    def __init__(self, eids: "np.ndarray", conn: "np.ndarray") -> None:
        self.eids = eids
        self.conn = conn
        # A homogeneous fan-out has a rectangular int64[N, k] conn; a
        # mixed one has an object-dtype [N] of per-row arrays. ndim==2
        # distinguishes them for iteration/getitem.
        self._homogeneous = conn.ndim == 2

    def __len__(self) -> int:
        return int(self.eids.shape[0])

    def _row(self, i: int) -> "tuple[int, tuple[int, ...]]":
        return int(self.eids[i]), tuple(int(n) for n in self.conn[i])

    def __iter__(self) -> "Iterator[tuple[int, tuple[int, ...]]]":
        eids = self.eids
        conn = self.conn
        for i in range(eids.shape[0]):
            yield int(eids[i]), tuple(int(n) for n in conn[i])

    def __getitem__(self, i: int) -> "tuple[int, tuple[int, ...]]":
        return self._row(int(i))

    def __bool__(self) -> bool:
        return int(self.eids.shape[0]) > 0


class ElementPlanRows:
    """Columnar element-plan entry for one Element spec (B1).

    Duck-typed replacement for the old
    ``list[tuple[int, tuple[int, ...], int]]`` (the ``sub`` list built
    per spec by :func:`allocate_element_tags`). Holds the spec's rows
    as arrays and yields ``(eid, conn_tuple, ele_tag)`` triples only
    transiently at iteration.

    Tags are contiguous by construction: :func:`allocate_element_tags`
    reserves a block of ``N`` element tags per spec, so row ``i``'s tag
    is ``tag_start + i`` (see :meth:`TagAllocator.allocate_block`).

    Attributes
    ----------
    eids : numpy.ndarray
        ``int64[N]`` element ids (the node-pair sentinel
        :data:`MISSING_FEM_ELEMENT_ID` for a 1-row node-pair spec).
    conn : numpy.ndarray
        Connectivity — same homogeneous / mixed convention as
        :class:`PGElementFanout`.
    tag_start : int
        The OpenSees element tag of row 0; row ``i`` → ``tag_start + i``
        when ``tags`` is ``None`` (the contiguous-block common case).
    tags : numpy.ndarray or None
        Optional ``int64[N]`` explicit per-row tags. Set only for
        rank-bucketed *subsets* (see :class:`LazyRankBuckets`)
        where the selected rows are no longer contiguous, so row ``i``'s
        tag is ``tags[i]``. ``None`` for the full per-spec plan.

    Notes
    -----
    Read-only. Iterating / indexing yields fresh transient tuples.
    """

    __slots__ = ("eids", "conn", "tag_start", "tags", "_homogeneous")

    def __init__(
        self,
        eids: "np.ndarray",
        conn: "np.ndarray",
        tag_start: int,
        tags: "np.ndarray | None" = None,
    ) -> None:
        self.eids = eids
        self.conn = conn
        self.tag_start = int(tag_start)
        self.tags = tags
        self._homogeneous = conn.ndim == 2

    def __len__(self) -> int:
        return int(self.eids.shape[0])

    def _tag(self, i: int) -> int:
        if self.tags is None:
            return self.tag_start + int(i)
        return int(self.tags[i])

    def _row(self, i: int) -> "tuple[int, tuple[int, ...], int]":
        return (
            int(self.eids[i]),
            tuple(int(n) for n in self.conn[i]),
            self._tag(int(i)),
        )

    def __iter__(self) -> "Iterator[tuple[int, tuple[int, ...], int]]":
        eids = self.eids
        conn = self.conn
        if self.tags is None:
            tag_start = self.tag_start
            for i in range(eids.shape[0]):
                yield int(eids[i]), tuple(int(n) for n in conn[i]), tag_start + i
        else:
            tags = self.tags
            for i in range(eids.shape[0]):
                yield int(eids[i]), tuple(int(n) for n in conn[i]), int(tags[i])

    def __getitem__(self, i: int) -> "tuple[int, tuple[int, ...], int]":
        return self._row(int(i))

    def __bool__(self) -> bool:
        return int(self.eids.shape[0]) > 0

    def select_rows(self, idx: "np.ndarray") -> "ElementPlanRows":
        """Return a row-subset view for the integer index array ``idx``.

        Used by :class:`LazyRankBuckets` to build a per-rank
        bucket that is still columnar (arrays indexed by ``idx``) instead
        of a re-materialised list of tuples. The subset carries explicit
        per-row ``tags`` (``tag_start + idx``) because the selected rows
        are no longer contiguous. Order follows ``idx``.
        """
        idx = np.asarray(idx, dtype=np.int64)
        sub_eids = self.eids[idx]
        if self._homogeneous:
            sub_conn = self.conn[idx]
        else:
            # object-dtype [N] of per-row arrays — fancy-index preserves it.
            sub_conn = self.conn[idx]
        if self.tags is None:
            sub_tags = self.tag_start + idx
        else:
            sub_tags = self.tags[idx]
        return ElementPlanRows(sub_eids, sub_conn, 0, tags=sub_tags)


def _fanout_arrays_from_group_result(
    result: "Iterable[Any]",
) -> "tuple[np.ndarray, np.ndarray]":
    """Concatenate a ``GroupResult``'s blocks into ``(eids, conn)`` arrays.

    Order matches the legacy ``for group in result: for eid, conn_row
    in group`` walk: groups in ``GroupResult`` order, elements in stored
    id order within each group. Homogeneous fan-outs (all blocks share
    ``npe``) yield a rectangular ``int64[N, k]``; mixed-npe fan-outs
    yield an object-dtype ``[N]`` of per-row int64 arrays (each already
    at its block's true width — no padding), so iteration never leaks a
    sentinel.
    """
    id_blocks: "list[np.ndarray]" = []
    conn_blocks: "list[np.ndarray]" = []
    for group in result:
        gids = getattr(group, "ids", None)
        gconn = getattr(group, "connectivity", None)
        if gids is not None and gconn is not None:
            # Real ``ElementGroup`` — columnar int64 arrays.
            id_blocks.append(np.asarray(gids, dtype=np.int64))
            conn_blocks.append(np.asarray(gconn, dtype=np.int64))
        else:
            # Duck-typed fallback: a group that only supports the legacy
            # ``for eid, conn_row in group`` pair-iteration protocol (the
            # contract the old ``expand_pg_to_elements`` relied on; some
            # lightweight test stubs expose only this). Materialise its
            # rows into arrays so the columnar container still holds
            # arrays, not the pair tuples.
            g_ids: "list[int]" = []
            g_conn: "list[tuple[int, ...]]" = []
            for eid, conn_row in group:
                g_ids.append(int(eid))
                g_conn.append(tuple(int(n) for n in conn_row))
            if not g_ids:
                continue
            id_blocks.append(np.asarray(g_ids, dtype=np.int64))
            conn_blocks.append(np.asarray(g_conn, dtype=np.int64))
    if not id_blocks:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 0), dtype=np.int64),
        )
    eids = np.concatenate(id_blocks)
    widths = {b.shape[1] for b in conn_blocks}
    if len(widths) == 1:
        return eids, np.concatenate(conn_blocks, axis=0)
    # Mixed npe across blocks: keep per-row arrays at true width so no
    # -1 padding is ever needed and iteration slices trivially.
    rows: "list[np.ndarray]" = []
    for block in conn_blocks:
        for ri in range(block.shape[0]):
            rows.append(block[ri])
    conn_obj = np.empty((len(rows),), dtype=object)
    for i, row_arr in enumerate(rows):
        conn_obj[i] = row_arr
    return eids, conn_obj


def expand_pg_to_elements(
    fem: "FEMData", pg: str,
) -> "PGElementFanout":
    """Return a columnar ``(eid, conn)`` fan-out for ``pg``.

    Order is deterministic: groups iterate in their FEM-snapshot order,
    and within each group elements iterate in id order. Empty PGs
    return an empty fan-out (the caller decides whether that warrants a
    warning; per the Phase-4 spec, empty is permitted and emits
    nothing).

    ADR 0065 v2 / plan_emit_memory_columnar.md B4: the result is a
    :class:`PGElementFanout` (int64 arrays, transient row boxing) rather
    than a resident ``list[tuple[int, tuple[int, ...]]]`` — iterating it
    yields the exact same ``(int, tuple[int, ...])`` pairs in the same
    order, so every ``for eid, conn in expand_pg_to_elements(...)``
    consumer is unchanged, but the memoised form no longer pins ~54M
    boxed connectivity ints.

    The result is memoised per snapshot (see :data:`_PG_FANOUT_CACHE`);
    treat it as read-only.

    Raises
    ------
    BridgeError
        If ``pg`` is not a known PG label / name on the FEM snapshot.
        The error includes the available PG names to help the user.
    """
    if pg is None:
        # Defensive (ADR 0049): ``FEMData.select(pg=None)`` returns the
        # WHOLE mesh, so an un-routed node-pair spec (``spec.pg is None``)
        # reaching here would silently fan out over every element. Fail
        # loud — every element-spec fan-out must route through
        # ``expand_spec_to_elements``, which sends node-pair specs down the
        # explicit-endpoint path and never calls this with ``pg=None``.
        raise BridgeError(
            "expand_pg_to_elements called with pg=None — a node-pair element "
            "spec must be routed through expand_spec_to_elements, not "
            "expanded as a physical group (this is an internal routing bug)."
        )
    cache = _pg_fanout_cache_for(fem)
    if cache is not None:
        hit = cache.get(("elements", pg))
        if hit is not None:
            return hit  # type: ignore[return-value]
    try:
        # selection-unification v2 P3-R / §6.3 §2 #5 (P-GROUPRESULT;
        # m3 — resolution raises at .select()).
        result = fem.elements.select(pg=pg).groups()
    except (KeyError, ValueError) as e:
        # FEMData raises one of these for an unknown PG; surface a
        # bridge-flavored error so the call-site can distinguish.
        available = _available_pg_names(fem)
        raise BridgeError(
            f"physical group {pg!r} not found in FEM snapshot. "
            f"Available element PGs: {sorted(available)}."
        ) from e
    # ADR 0065 v2 / B4: build the fan-out columnar straight from the
    # GroupResult's int64 blocks (concatenated in the legacy walk order)
    # so the memoised form carries arrays, not a boxed tuple graph. The
    # live-mesh element engine can hand back a flat ``{'element_ids',
    # 'connectivity'}`` dict instead of a GroupResult (see the pair-view
    # __iter__ in _mesh_selection.py) — handle both.
    if isinstance(result, dict):
        eids = np.asarray(result["element_ids"], dtype=np.int64)
        conn = np.asarray(result["connectivity"], dtype=np.int64)
        if conn.ndim != 2:
            conn = conn.reshape(eids.shape[0], -1)
        out = PGElementFanout(eids, conn)
    else:
        eids, conn = _fanout_arrays_from_group_result(result)
        out = PGElementFanout(eids, conn)
    if cache is not None:
        cache[("elements", pg)] = out
    return out


def resolve_element_node_pair(
    fem: "FEMData", spec: Element,
) -> tuple[int, int]:
    """Resolve a node-pair element spec's ``nodes=(ref_i, ref_j)`` to tags.

    ADR 0049 node-pair form.  Each endpoint is a :data:`NodeRef`:

    * a ``DecoupledNodeDef`` handle (from ``g.decouple_node``) → its
      ``.tag`` (fail-loud if the handle was never materialized by the FEM
      factory, i.e. ``.tag is None``);
    * a node-label ``str`` → exactly one node via
      :func:`_expand_label_to_nodes` (0 or ≥2 matches fail loud);
    * an ``int`` → the raw node tag (power-user escape hatch — **not**
      compose-safe: ``g.compose`` offsets node tags at the FEM level
      before the bridge runs, so a raw int authored against pre-compose
      numbering binds to the wrong node; prefer a handle or label).

    The two resolved tags must differ — OpenSees has no same-node guard in
    any zeroLength-family element, so an ``i == j`` pair would assemble a
    singular zero-length element silently.  Distinctness is checked here on
    the *resolved* tags (so a label and an int resolving to the same node
    are caught).
    """
    from ...element.zero_length import NodeRef  # noqa: F401  (doc anchor)

    nodes = getattr(spec, "nodes", None)
    cls = type(spec).__name__
    if nodes is None or len(nodes) != 2:
        raise BridgeError(
            f"{cls}: node-pair element has no resolvable nodes= 2-tuple "
            f"(got {nodes!r}) — this is an internal routing bug."
        )

    def _resolve_one(ref: object, which: str) -> int:
        if isinstance(ref, bool):
            raise BridgeError(
                f"{cls} nodes= {which} endpoint must be a g.decouple_node "
                f"handle, a node-label str, or an int tag; got bool {ref!r}."
            )
        if isinstance(ref, int):
            return int(ref)
        if isinstance(ref, str):
            ids = _expand_label_to_nodes(fem, ref)
            if len(ids) != 1:
                raise BridgeError(
                    f"{cls} nodes= {which} endpoint label {ref!r} resolves to "
                    f"{len(ids)} nodes {tuple(ids)} — a node-pair endpoint "
                    f"label must name EXACTLY one node. Use a label bound to a "
                    f"single node (e.g. a g.decouple_node label) or pass the "
                    f"handle directly."
                )
            return int(ids[0])
        tag = getattr(ref, "tag", None)
        if tag is None:
            raise BridgeError(
                f"{cls} nodes= {which} endpoint is a decoupled-node handle "
                f"with no resolved tag — call g.mesh.queries.get_fem_data(...) "
                f"so the FEM factory assigns it a tag before building the "
                f"bridge."
            )
        return int(tag)

    i_tag = _resolve_one(nodes[0], "node_i")
    j_tag = _resolve_one(nodes[1], "node_j")
    if i_tag == j_tag:
        raise BridgeError(
            f"{cls} node-pair endpoints both resolve to node {i_tag} — a "
            f"zeroLength-family element needs two DISTINCT nodes (OpenSees "
            f"would otherwise assemble a singular element silently)."
        )
    return i_tag, j_tag


def expand_spec_to_elements(
    fem: "FEMData", spec: Element,
) -> "PGElementFanout":
    """Unified element-spec fan-out: PG form OR node-pair form (ADR 0049).

    * ``spec.pg is not None`` → fan across the physical group
      (:func:`expand_pg_to_elements`).
    * ``spec.pg is None`` (node-pair form) → a single synthetic element
      ``(MISSING_FEM_ELEMENT_ID, (i_tag, j_tag))`` from the resolved
      ``nodes=`` endpoints.  The sentinel fem-eid marks "no backing gmsh
      cell" — it carries through to the emitter so the H5 record stores the
      connectivity inline (no neutral-mesh cell to source it from).

    ADR 0065 v2 / B4: both branches return a :class:`PGElementFanout`
    (the node-pair path a 1-row one). ``MISSING_FEM_ELEMENT_ID`` (-1)
    fits int64; the sentinel row iterates as ``(-1, (i_tag, j_tag))`` —
    byte-identical to the old 1-element list — and the callers filter it
    out of the tag map by value (``eid != MISSING_FEM_ELEMENT_ID``).

    Branches on ``pg is not None`` by **identity** (never truthiness): an
    empty-string pg is still a PG form, and ``pg=None`` must route
    exclusively to the node-pair path (never to
    ``expand_pg_to_elements``, which would select the whole mesh).
    """
    pg = getattr(spec, "pg", None)
    if pg is not None:
        return expand_pg_to_elements(fem, pg)
    i_tag, j_tag = resolve_element_node_pair(fem, spec)
    return PGElementFanout(
        np.asarray([MISSING_FEM_ELEMENT_ID], dtype=np.int64),
        np.asarray([[i_tag, j_tag]], dtype=np.int64),
    )


def expand_pg_to_nodes(fem: "FEMData", pg: str) -> tuple[int, ...]:
    """Return the node ids for ``pg`` in deterministic order.

    Memoised per snapshot (see :data:`_PG_FANOUT_CACHE`).

    Raises :class:`BridgeError` if ``pg`` is unknown.
    """
    cache = _pg_fanout_cache_for(fem)
    if cache is not None:
        hit = cache.get(("nodes", pg))
        if hit is not None:
            return hit  # type: ignore[return-value]
    try:
        # selection-unification v2 P3-R / §6.3 §2 #6 (P-NODE; m3).
        ids = fem.nodes.select(pg=pg).ids
    except (KeyError, ValueError) as e:
        available = _available_pg_names(fem)
        raise BridgeError(
            f"physical group {pg!r} not found in FEM snapshot. "
            f"Available PGs: {sorted(available)}."
        ) from e
    out = tuple(int(n) for n in ids)
    if cache is not None:
        cache[("nodes", pg)] = out
    return out


def _available_pg_names(fem: "FEMData") -> set[str]:
    """PG names known to the snapshot, for the "not found" errors.

    Helps the user spot a typo without having to re-query the FEM. Both
    ``elements`` and ``nodes`` are read, since the user might have
    asked for a node PG via an element-fan-out call site (or vice
    versa). Reads the public ``PhysicalGroupSet.names()``: the private
    ``_groups`` dict is keyed by ``(dim, tag)``, so filtering its keys
    for strings always listed nothing (#1335).
    """
    return set(fem.elements.physical.names()) | set(
        fem.nodes.physical.names()
    )


def _describe_pg_cells(fem: "FEMData", pg: str) -> str:
    """Best-effort description of what physical group ``pg`` registers.

    Used to build a useful error when a ``pg=`` element fan-out resolves
    to zero elements: the group commonly still HAS cells registered on
    the FEM snapshot's element-group registry (``fem.elements.physical``)
    — e.g. a ``get_fem_data(dim=...)`` call excluded them from
    ``fem.elements`` itself — so naming the registered count/dimension is
    more useful than a bare "0 elements".
    """
    physical = getattr(fem.elements, "physical", None)
    groups = getattr(physical, "_groups", None) if physical is not None else None
    if not isinstance(groups, dict):
        return f"pg {pg!r} has no registered cells on this snapshot"
    matches = [
        (dim, info) for (dim, _tag), info in groups.items()
        if info.get("name") == pg
    ]
    if not matches:
        return f"pg {pg!r} has no registered cells on this snapshot"
    parts = []
    for dim, info in sorted(matches):
        eids = info.get("element_ids")
        n = len(eids) if eids is not None else 0
        parts.append(f"{n} dim-{dim} cell(s)")
    return f"pg {pg!r} has " + ", ".join(parts)


def _record_node_ids(
    fem: "FEMData", rec: "FixRecord | SupportRecord | MassRecord",
) -> "Iterable[int]":
    """The node ids a ``pg`` XOR ``nodes`` bridge record targets."""
    if rec.nodes is not None:
        return rec.nodes
    return expand_pg_to_nodes(fem, rec.pg) if rec.pg is not None else ()


def _expand_label_to_nodes(fem: "FEMData", label_name: str) -> tuple[int, ...]:
    """Return node IDs registered under ``label_name`` on ``fem.nodes.labels``.

    Raises :class:`BridgeError` if the FEM snapshot exposes no labels
    accessor (older fixtures) or the label name is unknown.
    """
    nodes_obj = getattr(fem, "nodes", None)
    labels = getattr(nodes_obj, "labels", None) if nodes_obj is not None else None
    if labels is None:
        raise BridgeError(
            f"label {label_name!r} requested but FEM snapshot has no "
            f"nodes.labels accessor."
        )
    try:
        ids = labels.node_ids(label_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"node label {label_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(n) for n in ids)
