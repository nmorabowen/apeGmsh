"""pg_expand: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
import weakref
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..types import Element
from ..tag_resolution import MISSING_FEM_ELEMENT_ID
from .._stacklevel import _stacklevel_outside_package
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
    * a ``str`` → exactly one node of the node label (a
      ``g.decouple_node`` ``label=`` included) or, failing that, the
      physical group of that name (:func:`node_name_to_nodes`;
      0 or ≥2 matches fail loud);
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
            ids = node_name_to_nodes(
                fem, ref, where=f"{cls} nodes= {which} endpoint")
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


def node_name_to_nodes(
    fem: "FEMData", name: str, *, where: str,
) -> tuple[int, ...]:
    """The nodes a node-argument name selects: a node label (a
    ``g.decouple_node`` label included) or a physical group.

    A name that is both, selecting different nodes, is refused rather
    than picked; one that is neither raises naming both namespaces.
    """
    nodes_obj = getattr(fem, "nodes", None)

    def _lookup(attr: str) -> tuple[int, ...] | None:
        groups = getattr(nodes_obj, attr, None)
        if groups is None:
            return None
        try:
            return tuple(int(n) for n in groups.node_ids(name))
        except KeyError:
            return None  # not a name of this namespace

    by_label = _lookup("labels")
    by_pg = _lookup("physical")
    if by_label is None:
        if by_pg is None:
            raise BridgeError(
                f"{where} {name!r} is neither a node "
                f"label nor a physical group on the FEM snapshot (a "
                f"g.decouple_node label=... is a node label)."
            )
        return by_pg
    if by_pg is not None and set(by_label) != set(by_pg):
        raise BridgeError(
            f"{where} {name!r} names both a label "
            f"(nodes {by_label[:5]}) and a physical group (nodes "
            f"{by_pg[:5]}) — rename one, or pass the node handle."
        )
    return by_label


def _polygon_area_3d(fem: "FEMData", cycle: Sequence[int]) -> float:
    """Enclosed 3D polygon area via Newell's method (fallback to vertex count on stubs)."""
    nodes_obj = getattr(fem, "nodes", None)
    ids = getattr(nodes_obj, "ids", None) if nodes_obj is not None else None
    coords = getattr(nodes_obj, "coords", None) if nodes_obj is not None else None
    if ids is None or coords is None:
        return float(len(cycle))
    try:
        id_arr = np.asarray(ids, dtype=np.int64)
        coord_arr = np.asarray(coords, dtype=np.float64)
        id_to_idx = {int(nid): i for i, nid in enumerate(id_arr.tolist())}
        pts = [coord_arr[id_to_idx[int(n)]] for n in cycle]
    except Exception:
        return float(len(cycle))
    nx = ny = nz = 0.0
    m = len(pts)
    for i in range(m):
        p0 = pts[i]
        p1 = pts[(i + 1) % m]
        x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2]) if len(p0) > 2 else 0.0
        x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2]) if len(p1) > 2 else 0.0
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return 0.5 * float(np.hypot(np.hypot(nx, ny), nz))


def _cycles_are_interior_holes(
    fem: "FEMData",
    outer: Sequence[int],
    holes: Sequence[Sequence[int]],
) -> bool:
    """Return True if all ``holes`` loops lie inside ``outer`` in ``outer``'s plane."""
    nodes_obj = getattr(fem, "nodes", None)
    ids = getattr(nodes_obj, "ids", None) if nodes_obj is not None else None
    coords = getattr(nodes_obj, "coords", None) if nodes_obj is not None else None
    if ids is None or coords is None:
        return True
    try:
        id_arr = np.asarray(ids, dtype=np.int64)
        coord_arr = np.asarray(coords, dtype=np.float64)
        id_to_idx = {int(nid): i for i, nid in enumerate(id_arr.tolist())}
        outer_pts = [coord_arr[id_to_idx[int(n)]] for n in outer]
        hole_pts_list = [[coord_arr[id_to_idx[int(n)]] for n in h] for h in holes]
    except Exception:
        return True

    # Newell normal of outer loop
    nx = ny = nz = 0.0
    m = len(outer_pts)
    for i in range(m):
        p0 = outer_pts[i]
        p1 = outer_pts[(i + 1) % m]
        x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2]) if len(p0) > 2 else 0.0
        x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2]) if len(p1) > 2 else 0.0
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    norm_n = float(np.hypot(np.hypot(nx, ny), nz))
    if norm_n <= 1e-14:
        return False
    n_vec = np.array([nx / norm_n, ny / norm_n, nz / norm_n], dtype=np.float64)
    # Orthonormal basis (e1, e2) in plane
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_vec, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_vec, e1)

    def _proj(pt: np.ndarray) -> tuple[float, float]:
        p3 = np.array(
            [float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0],
            dtype=np.float64,
        )
        return float(np.dot(p3, e1)), float(np.dot(p3, e2))

    poly2d = [_proj(p) for p in outer_pts]

    def _point_in_poly(u: float, v: float) -> bool:
        inside = False
        k = len(poly2d)
        for i in range(k):
            u0, v0 = poly2d[i]
            u1, v1 = poly2d[(i + 1) % k]
            if (v0 > v) != (v1 > v):
                u_int = u0 + (v - v0) * (u1 - u0) / (v1 - v0)
                if u < u_int:
                    inside = not inside
        return inside

    for h_pts in hole_pts_list:
        for hp in h_pts:
            hu, hv = _proj(hp)
            if not _point_in_poly(hu, hv):
                return False
    return True


def extract_pg_polygon_loops(
    fem: "FEMData",
    pg: str,
    fanout: "PGElementFanout | None" = None,
) -> "tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]":
    """Extract ordered ``(outer_loop, hole_loops)`` boundary node cycles from ``pg``.

    Works on 2D surface PGs (triangles/quads: boundary edges are those belonging
    to exactly one 2D cell, chained into closed cycles; the largest-area cycle is
    the outer boundary and any remaining cycles are interior holes) and 1D line
    loop PGs.
    """
    # Memoised per snapshot with the fan-out (an emit asks for a panel's
    # loops from several passes); only for the PG's own fan-out.
    cache = _pg_fanout_cache_for(fem)
    own = fanout is None or fanout is (
        cache.get(("elements", pg)) if cache is not None else None)
    if own and cache is not None:
        hit = cache.get(("loops", pg))
        if hit is not None:
            return hit  # type: ignore[return-value]
    loops = _extract_pg_polygon_loops(fem, pg, fanout)
    if own and cache is not None:
        cache[("loops", pg)] = loops
    return loops


def _extract_pg_polygon_loops(
    fem: "FEMData",
    pg: str,
    fanout: "PGElementFanout | None",
) -> "tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]":
    if fanout is None:
        fanout = expand_pg_to_elements(fem, pg)
    if len(fanout) == 0:
        raise BridgeError(
            f"LadrunoMembrane(pg={pg!r}) selected 0 elements."
        )

    edge_count: dict[tuple[int, int], int] = {}
    first_dir: dict[tuple[int, int], tuple[int, int]] = {}
    order_keys: list[tuple[int, int]] = []

    for _eid, row in fanout:
        m = len(row)
        corners: tuple[int, ...]
        if m in (3, 6):
            corners = (int(row[0]), int(row[1]), int(row[2]))
        elif m in (4, 8, 9):
            corners = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        elif m == 2:
            u, v = int(row[0]), int(row[1])
            key = (u, v) if u < v else (v, u)
            if key not in edge_count:
                edge_count[key] = 0
                first_dir[key] = (u, v)
                order_keys.append(key)
            edge_count[key] += 1
            continue
        elif m >= 3:
            corners = tuple(int(n) for n in row)
        else:
            continue
        k = len(corners)
        for i in range(k):
            u = corners[i]
            v = corners[(i + 1) % k]
            key = (u, v) if u < v else (v, u)
            if key not in edge_count:
                edge_count[key] = 0
                first_dir[key] = (u, v)
                order_keys.append(key)
            edge_count[key] += 1

    boundary_edges = [first_dir[k] for k in order_keys if edge_count[k] == 1]
    if not boundary_edges:
        raise BridgeError(
            f"LadrunoMembrane(pg={pg!r}): could not extract a closed boundary "
            f"polygon loop from physical group {pg!r} (0 boundary edges)."
        )

    next_map: dict[int, list[int]] = {}
    adj: dict[int, list[int]] = {}
    vertices_in_order: list[int] = []
    seen_v: set[int] = set()
    for u, v in boundary_edges:
        next_map.setdefault(u, []).append(v)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        if u not in seen_v:
            seen_v.add(u)
            vertices_in_order.append(u)
        if v not in seen_v:
            seen_v.add(v)
            vertices_in_order.append(v)

    for v_id in vertices_in_order:
        deg = len(adj.get(v_id, ()))
        if deg != 2:
            raise BridgeError(
                f"LadrunoMembrane(pg={pg!r}): boundary vertex {v_id} has "
                f"boundary degree {deg} (expected 2 for a simple closed "
                f"manifold polygon boundary)."
            )

    visited: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for start in vertices_in_order:
        if start in visited:
            continue
        # Try directed walk first (preserves cell winding)
        cycle: list[int] = []
        cur = start
        directed_ok = True
        while cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            outs = [nb for nb in next_map.get(cur, ()) if nb not in visited or nb == start]
            if not outs:
                directed_ok = False
                break
            nxt = outs[0]
            if nxt == start:
                break
            cur = nxt
        if not directed_ok or len(cycle) < 3:
            # Fallback to undirected cycle walk
            for v in cycle:
                visited.discard(v)
            cycle = []
            cur = start
            prev = -1
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                nbs = adj.get(cur, [])
                nxt_candidates = [nb for nb in nbs if nb != prev and (nb not in visited or nb == start)]
                if not nxt_candidates:
                    break
                prev, cur = cur, nxt_candidates[0]
                if cur == start:
                    break
        if len(cycle) < 3:
            raise BridgeError(
                f"LadrunoMembrane(pg={pg!r}): boundary edges did not form a closed "
                f"loop of >= 3 nodes."
            )
        cycles.append(tuple(cycle))

    if not cycles:
        raise BridgeError(
            f"LadrunoMembrane(pg={pg!r}): boundary edges did not form a closed "
            f"loop of >= 3 nodes."
        )
    if len(cycles) == 1:
        return cycles[0], ()

    scored = [(i, _polygon_area_3d(fem, c), c) for i, c in enumerate(cycles)]
    scored.sort(key=lambda item: (-item[1], item[0]))
    outer = scored[0][2]
    holes = tuple(item[2] for item in scored[1:])
    if not _cycles_are_interior_holes(fem, outer, holes):
        raise BridgeError(
            f"LadrunoMembrane(pg={pg!r}): physical group contains multiple "
            f"disconnected polygon boundary loops; each disconnected panel "
            f"must be its own LadrunoMembrane."
        )
    return outer, holes


def _resolve_membrane_node_ref(
    fem: "FEMData", ref: object, where: str,
) -> int:
    if isinstance(ref, bool):
        raise BridgeError(
            f"LadrunoMembrane {where} must be a Node handle, node-label str, "
            f"or int tag; got bool {ref!r}."
        )
    if isinstance(ref, int):
        return int(ref)
    if isinstance(ref, str):
        try:
            ids = _expand_label_to_nodes(fem, ref)
        except BridgeError:
            ids = expand_pg_to_nodes(fem, ref)
        if len(ids) != 1:
            raise BridgeError(
                f"LadrunoMembrane {where} label/PG {ref!r} resolves to {len(ids)} "
                f"nodes {tuple(ids)} — expected a single node."
            )
        return int(ids[0])
    tag = getattr(ref, "tag", None)
    if tag is None:
        raise BridgeError(
            f"LadrunoMembrane {where} is a Node handle with no resolved tag."
        )
    return int(tag)


def resolve_ladruno_membrane_nodes(
    fem: "FEMData", spec: Element,
) -> tuple[int, ...]:
    """Resolve ``LadrunoMembrane(nodes=...)`` to an ordered tuple of node tags."""
    nodes = getattr(spec, "nodes", None)
    if nodes is None or len(nodes) < 3:
        raise BridgeError(
            f"LadrunoMembrane: nodes= requires at least 3 boundary nodes, got {nodes!r}."
        )
    resolved = tuple(
        _resolve_membrane_node_ref(fem, ref, f"nodes[{i}]")
        for i, ref in enumerate(nodes)
    )
    if len(set(resolved)) < 3:
        raise BridgeError(
            f"LadrunoMembrane: nodes= must resolve to at least 3 distinct node tags, "
            f"got {resolved}."
        )
    return resolved


def _node_xyz(fem: "FEMData", ids: "Sequence[int] | np.ndarray") -> "np.ndarray":
    """``float64[n, 3]`` coordinates of ``ids`` (memoised sorted id index).

    Raises :class:`BridgeError` when an id is not a node of the snapshot.
    """
    cache = _pg_fanout_cache_for(fem)
    hit = cache.get(("node_index", "")) if cache is not None else None
    if hit is None:
        all_ids = np.asarray(fem.nodes.ids, dtype=np.int64).reshape(-1)
        coords = np.asarray(fem.nodes.coords, dtype=np.float64)
        if coords.ndim == 2 and coords.shape[1] < 3:
            coords = np.hstack(
                [coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))])
        order = np.argsort(all_ids, kind="stable")
        hit = (all_ids[order], coords[order])
        if cache is not None:
            cache[("node_index", "")] = hit
    sorted_ids: np.ndarray = hit[0]  # type: ignore[index]
    sorted_xyz: np.ndarray = hit[1]  # type: ignore[index]
    want = np.asarray(ids, dtype=np.int64).reshape(-1)
    pos = np.searchsorted(sorted_ids, want)
    pos_c = np.minimum(pos, max(len(sorted_ids) - 1, 0))
    if len(sorted_ids) == 0 or not bool(np.all(sorted_ids[pos_c] == want)):
        bad = want[(len(sorted_ids) == 0) | (sorted_ids[pos_c] != want)]
        raise BridgeError(
            f"LadrunoMembrane: node(s) {bad[:5].tolist()} are not nodes of "
            f"the FEM snapshot."
        )
    return np.asarray(sorted_xyz[pos_c], dtype=np.float64)


#: Relative tolerance of the frame-in-panel test: a point is on the panel's
#: plane / boundary when it is within ``_PANEL_REL_TOL`` x the panel's
#: bounding-box diagonal.  Shared mesh nodes coincide exactly, so this only
#: absorbs round-off.
_PANEL_REL_TOL = 1e-6


class MembraneFramesFilteredWarning(UserWarning):
    """A ``LadrunoMembrane`` ``frames=`` group lost some or all of its
    frames to the per-panel filter (:func:`resolve_ladruno_membrane_context`).

    Expected for a group that spans several panels (a storey's beams); a
    sign of a wrong group or a misplaced frame otherwise, so the build
    names the dropped elements.
    """


class _PanelGeometry:
    """Plane + 2-D polygon of one membrane panel, for the frame filter."""

    def __init__(
        self,
        fem: "FEMData",
        outer: "Sequence[int]",
        holes: "Sequence[Sequence[int]]",
    ) -> None:
        pts = _node_xyz(fem, outer)
        self.origin = pts.mean(axis=0)
        n = np.zeros(3)
        m = len(pts)
        for i in range(m):
            p0, p1 = pts[i], pts[(i + 1) % m]
            n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
            n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
            n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
        nn = float(np.linalg.norm(n))
        if nn <= 0.0:
            raise BridgeError(
                "LadrunoMembrane: the panel's boundary polygon is degenerate "
                "(zero area); cannot place its frames."
            )
        self.normal = n / nn
        ref = (np.array([1.0, 0.0, 0.0]) if abs(self.normal[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        e1 = np.cross(self.normal, ref)
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.normal, self.e1)
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        self.tol = _PANEL_REL_TOL * max(diag, 1e-300)
        self.outer2d = self._proj(pts)
        self.holes2d = [self._proj(_node_xyz(fem, h)) for h in holes]
        self.lo = self.outer2d.min(axis=0) - self.tol
        self.hi = self.outer2d.max(axis=0) + self.tol

    def _proj(self, pts: "np.ndarray") -> "np.ndarray":
        d = pts - self.origin
        return np.stack([d @ self.e1, d @ self.e2], axis=1)

    @staticmethod
    def _on_boundary(p: "np.ndarray", poly: "np.ndarray", tol: float) -> "np.ndarray":
        a = poly
        b = np.roll(poly, -1, axis=0)
        ab = b - a                                   # (m, 2)
        ap = p[:, None, :] - a[None, :, :]           # (n, m, 2)
        den = np.maximum((ab * ab).sum(axis=1), 1e-300)
        t = np.clip((ap * ab[None]).sum(axis=2) / den[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        dist = np.linalg.norm(p[:, None, :] - proj, axis=2)
        return np.asarray((dist <= tol).any(axis=1))

    @staticmethod
    def _inside(p: "np.ndarray", poly: "np.ndarray") -> "np.ndarray":
        a = poly
        b = np.roll(poly, -1, axis=0)
        u, v = p[:, 0:1], p[:, 1:2]
        crosses = (a[None, :, 1] > v) != (b[None, :, 1] > v)
        dv = b[:, 1] - a[:, 1]
        dv = np.where(dv == 0.0, 1e-300, dv)
        u_int = a[None, :, 0] + (v - a[None, :, 1]) * (b[:, 0] - a[:, 0])[None] / dv[None]
        return np.asarray((np.count_nonzero(crosses & (u < u_int), axis=1) % 2) == 1)

    def contains(self, pts: "np.ndarray") -> "np.ndarray":
        """Per point: on the plane, and on the outer boundary or strictly
        inside it, and not strictly inside a hole."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        ok: np.ndarray = np.asarray(
            np.abs((pts - self.origin) @ self.normal) <= self.tol, dtype=bool)
        p2 = self._proj(pts)
        ok &= np.all((p2 >= self.lo) & (p2 <= self.hi), axis=1)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            return ok
        q = p2[idx]
        sub = self._on_boundary(q, self.outer2d, self.tol) | self._inside(q, self.outer2d)
        for h in self.holes2d:
            in_hole = self._inside(q, h) & ~self._on_boundary(q, h, self.tol)
            sub &= ~in_hole
        ok[idx] = sub
        return ok

    def frame_rows_inside(self, fem: "FEMData", conn: "Any") -> "np.ndarray":
        """Per frame connectivity row: every node, and the I-J chord midpoint,
        lie in the panel (:meth:`contains`).  ``conn`` is a rectangular
        ``int[N, k]`` array or a sequence of rows."""
        arr = np.asarray(conn) if not isinstance(conn, np.ndarray) else conn
        if arr.ndim == 2 and arr.dtype != object:
            if arr.shape[0] == 0:
                return np.zeros(0, dtype=bool)
            arr = arr.astype(np.int64, copy=False)
            ok_nodes = self.contains(_node_xyz(fem, arr.reshape(-1))).reshape(arr.shape)
            mids = 0.5 * (_node_xyz(fem, arr[:, 0]) + _node_xyz(fem, arr[:, 1]))
            return np.asarray(ok_nodes.all(axis=1) & self.contains(mids))
        rows = [tuple(int(n) for n in r) for r in conn]
        return np.asarray(
            [bool(self.frame_rows_inside(fem, np.asarray([r], dtype=np.int64))[0])
             for r in rows],
            dtype=bool,
        )


def _membrane_panel_name(spec: Element) -> str:
    pg = getattr(spec, "pg", None)
    if pg is not None:
        return f"LadrunoMembrane(pg={pg!r})"
    return f"LadrunoMembrane(nodes={getattr(spec, 'nodes', None)!r})"


def _plan_lookup(
    element_plan: "Iterable[tuple[Element, ElementPlanRows]]",
    *,
    eids: "Sequence[int] | None" = None,
    tag: "int | None" = None,
) -> "list[tuple[int, tuple[int, ...]]]":
    """``(ops_tag, conn)`` plan rows matching FEM ``eids`` or one ops ``tag``."""
    out: list[tuple[int, tuple[int, ...]]] = []
    want = np.asarray(list(eids), dtype=np.int64) if eids is not None else None
    for _s, sub in element_plan:
        if not len(sub):
            continue
        if want is not None:
            hits = np.nonzero(np.isin(sub.eids, want))[0]
        else:
            t = int(tag) if tag is not None else -1
            if sub.tags is None:
                i = t - sub.tag_start
                hits = np.asarray([i] if 0 <= i < len(sub) else [], dtype=np.int64)
            else:
                hits = np.flatnonzero(np.asarray(sub.tags) == t)
        for i in hits.tolist():
            _eid, conn, ele_tag = sub[i]
            out.append((int(ele_tag), conn))
    return out


def resolve_ladruno_membrane_context(
    fem: "FEMData",
    spec: Element,
    element_plan: "Iterable[tuple[Element, ElementPlanRows]]" = (),
    fem_eid_to_ops_tag: "Any" = None,
    *,
    include_frames: bool = True,
) -> "tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]":
    """Resolve ``(frames, support_nodes, holes)`` for a :class:`LadrunoMembrane`.

    Frames are filtered per panel: the fork kernel goes singular on a
    supporting frame outside the panel (an out-of-plane column projects
    to a point; a beam of the next panel lies outside its polygon), and
    then transfers no load at all.  A frame is kept when every node of it,
    and its I-J chord midpoint, lie on the panel's plane and on its outer
    boundary or strictly inside it, and not strictly inside a hole (within
    :data:`_PANEL_REL_TOL` x the panel's bounding-box diagonal).  A
    ``frames=`` group (a PG / label name, or a ``pg=`` Element spec) is
    FILTERED to the frames of this panel; an explicitly named frame (an int
    tag, or a node-pair Element spec) that falls outside is REFUSED.  When
    ``frames=`` was given, filtering leaves no frame and no
    ``support_nodes=`` was given, the panel would carry its load nowhere:
    refused, naming the panel.
    """
    from ...node import Node, NodeSet

    # 1. Frames — candidate groups as (ops_tags, conn, explicit, source):
    # ``conn`` a rectangular int64[N, k] array (or a row sequence for a
    # mixed-width group).  Kept columnar: a tower-wide ``frames="Beams"``
    # is re-tested against every panel.
    frames_out: list[int] = []
    seen_frames: set[int] = set()
    candidates: "list[tuple[np.ndarray, Any, bool, str]]" = []

    def _translate(eids: "np.ndarray", where: str) -> "np.ndarray":
        if fem_eid_to_ops_tag is None:
            tags = np.full(eids.shape[0], -1, dtype=np.int64)
        elif hasattr(fem_eid_to_ops_tag, "translate"):
            tags = np.asarray(fem_eid_to_ops_tag.translate(eids), dtype=np.int64)
        else:
            tags = np.asarray(
                [int(fem_eid_to_ops_tag.get(int(e), -1)) for e in eids.tolist()],
                dtype=np.int64,
            )
        if eids.shape[0] and bool((tags < 0).any()):
            feid = int(eids[int(np.argmax(tags < 0))])
            raise BridgeError(
                f"LadrunoMembrane: frames= {where} contains FEM "
                f"element {feid}, which has no emitted OpenSees element."
            )
        return tags

    raw_frames = getattr(spec, "frames", None) if include_frames else None
    if raw_frames is not None:
        element_plan = list(element_plan)
        plan_by_spec_id = {id(s): sub for s, sub in element_plan}
        for item in raw_frames:
            if isinstance(item, Element):
                sub = plan_by_spec_id.get(id(item))
                if sub is None or len(sub) == 0:
                    raise BridgeError(
                        f"LadrunoMembrane: frames= Element spec {item!r} has no "
                        f"emitted elements in this model."
                    )
                sub_tags = (
                    sub.tags if sub.tags is not None
                    else sub.tag_start + np.arange(len(sub), dtype=np.int64)
                )
                candidates.append((
                    np.asarray(sub_tags, dtype=np.int64), sub.conn,
                    getattr(item, "pg", None) is None,
                    f"Element spec {type(item).__name__}",
                ))
            elif isinstance(item, str):
                try:
                    fan = expand_pg_to_elements(fem, item)
                    eids_arr = np.asarray(fan.eids, dtype=np.int64)
                    conn_any: "Any" = fan.conn
                except BridgeError:
                    labels = getattr(getattr(fem, "elements", None), "labels", None)
                    if labels is None:
                        raise
                    try:
                        eids_arr = np.asarray(
                            [int(e) for e in labels.element_ids(item)], dtype=np.int64)
                    except (KeyError, ValueError) as exc:
                        raise BridgeError(
                            f"LadrunoMembrane: frames= group/label {item!r} not "
                            f"found in FEM snapshot."
                        ) from exc
                    conn_any = None
                if eids_arr.shape[0] == 0:
                    raise BridgeError(
                        f"LadrunoMembrane: frames= group/label {item!r} matched 0 elements."
                    )
                tags_arr = _translate(eids_arr, f"group {item!r}")
                if conn_any is None:
                    by_tag = dict(_plan_lookup(element_plan, eids=eids_arr.tolist()))
                    missing = [t for t in tags_arr.tolist() if t not in by_tag]
                    if missing:
                        raise BridgeError(
                            f"LadrunoMembrane: frames= label {item!r}: OpenSees "
                            f"element {missing[0]} has no planned connectivity."
                        )
                    conn_any = [by_tag[t] for t in tags_arr.tolist()]
                candidates.append((tags_arr, conn_any, False, f"group {item!r}"))
            elif isinstance(item, int) and not isinstance(item, bool):
                mapped = (
                    fem_eid_to_ops_tag.get(int(item))
                    if fem_eid_to_ops_tag is not None
                    else None
                )
                tag = int(mapped) if mapped is not None else int(item)
                rows = (
                    _plan_lookup(element_plan, eids=[int(item)]) if mapped is not None
                    else _plan_lookup(element_plan, tag=tag)
                )
                if not rows:
                    raise BridgeError(
                        f"LadrunoMembrane: frames= tag {item} matches no element "
                        f"this model emits."
                    )
                candidates.append((
                    np.asarray([tag], dtype=np.int64), [rows[0][1]], True,
                    f"tag {item}",
                ))

    # 2. Support nodes
    supp_out: list[int] = []
    seen_supp: set[int] = set()

    def _add_supp(n: int) -> None:
        ni = int(n)
        if ni not in seen_supp:
            seen_supp.add(ni)
            supp_out.append(ni)

    raw_supp = getattr(spec, "support_nodes", None)
    if raw_supp is not None:
        for item in raw_supp:
            if isinstance(item, int) and not isinstance(item, bool):
                _add_supp(int(item))
            elif isinstance(item, Node):
                _add_supp(_resolve_membrane_node_ref(fem, item, "support_nodes"))
            elif isinstance(item, NodeSet):
                for t in item.tags:
                    _add_supp(int(t))
            elif isinstance(item, str):
                try:
                    nids = expand_pg_to_nodes(fem, item)
                except BridgeError:
                    nids = _expand_label_to_nodes(fem, item)
                if not nids:
                    raise BridgeError(
                        f"LadrunoMembrane: support_nodes= {item!r} matched 0 nodes."
                    )
                for nid in nids:
                    _add_supp(int(nid))

    # 3. Holes (auto-detected from pg mesh + explicit holes=)
    holes_out: list[tuple[int, ...]] = []
    pg = getattr(spec, "pg", None)
    outer: "tuple[int, ...] | None" = None
    if pg is not None:
        outer, pg_holes = extract_pg_polygon_loops(fem, pg)
        holes_out.extend(pg_holes)

    raw_holes = getattr(spec, "holes", None)
    if raw_holes is not None:
        for h_idx, h in enumerate(raw_holes):
            if isinstance(h, str):
                try:
                    h_outer, h_sub = extract_pg_polygon_loops(fem, h)
                    holes_out.append(h_outer)
                    holes_out.extend(h_sub)
                except BridgeError:
                    nids = expand_pg_to_nodes(fem, h)
                    if len(nids) < 3:
                        raise BridgeError(
                            f"LadrunoMembrane: hole PG {h!r} requires at least 3 nodes, "
                            f"got {len(nids)}."
                        )
                    holes_out.append(tuple(int(n) for n in nids))
            else:
                loop = tuple(
                    _resolve_membrane_node_ref(fem, n, f"holes[{h_idx}][{i}]")
                    for i, n in enumerate(h)
                )
                if len(set(loop)) < 3:
                    raise BridgeError(
                        f"LadrunoMembrane: holes[{h_idx}] requires at least 3 "
                        f"distinct nodes, got {loop}."
                    )
                holes_out.append(loop)

    # 4. Per-panel frame filter (see the docstring).
    if candidates:
        if outer is None:
            outer = resolve_ladruno_membrane_nodes(fem, spec)
        panel = _PanelGeometry(fem, outer, holes_out)
        for tags_arr, conn_any, explicit, source in candidates:
            inside = panel.frame_rows_inside(fem, conn_any)
            if explicit and not bool(inside.all()):
                k = int(np.argmin(inside))
                raise BridgeError(
                    f"{_membrane_panel_name(spec)}: frames= {source} (OpenSees "
                    f"element {int(tags_arr[k])}, nodes "
                    f"{tuple(int(n) for n in conn_any[k])}) does not lie in this "
                    f"panel: a supporting frame must lie in the panel's plane, "
                    f"on its outer boundary or inside it, and not inside a "
                    f"hole. The fork kernel goes singular on such a frame and "
                    f"transfers no load. Drop it from this panel's frames=, or "
                    f"name a group (PG / label) and let the bridge keep the "
                    f"panel's own frames."
                )
            if not explicit and int(np.count_nonzero(inside)) < len(inside):
                dropped = tags_arr[~inside].tolist()
                share = (
                    f"all {len(inside)}" if len(dropped) == len(inside)
                    else f"{len(dropped)} of its {len(inside)}"
                )
                warnings.warn(
                    f"{_membrane_panel_name(spec)}: frames= {source}: "
                    f"{share} frame element(s) do "
                    f"not lie in this panel and are not passed to it "
                    f"(OpenSees element(s) {dropped[:10]}"
                    f"{' ...' if len(dropped) > 10 else ''}). Expected when "
                    f"the group spans several panels; otherwise check that "
                    f"those frames lie on the panel's plane, on its outer "
                    f"boundary or inside it, and not inside a hole.",
                    MembraneFramesFilteredWarning,
                    stacklevel=_stacklevel_outside_package(),
                )
            for tag in tags_arr[inside].tolist():
                if tag not in seen_frames:
                    seen_frames.add(tag)
                    frames_out.append(int(tag))
        if not frames_out and not supp_out:
            raise BridgeError(
                f"{_membrane_panel_name(spec)}: none of its frames= "
                f"({', '.join(sorted({s for _t, _c, _x, s in candidates}))}) lies "
                f"in this panel (on its plane, on its outer boundary or inside "
                f"it), and no support_nodes= were given, so the panel's load "
                f"would reach no support. Add the panel's edge beams to frames=, "
                f"or give support_nodes=."
            )

    return tuple(frames_out), tuple(supp_out), tuple(holes_out)


def resolve_ladruno_membrane_diaphragm_nodes(
    fem: "FEMData", spec: Element,
) -> "tuple[int, ...]":
    """``diaphragm_nodes=`` of a :class:`LadrunoMembrane` as node tags.

    The forms of ``support_nodes=``: int tags, ``Node`` handles and PG or
    label names (a PG first, then a node label), in order, first
    occurrence kept. ``()`` when the spec has none.  Any other spec
    raises: only a membrane has diaphragm nodes.
    """
    from ...element.shell import LadrunoMembrane

    if not isinstance(spec, LadrunoMembrane):
        raise BridgeError(
            f"resolve_ladruno_membrane_diaphragm_nodes: {type(spec).__name__} "
            "is not a LadrunoMembrane."
        )
    raw = spec.diaphragm_nodes
    out: list[int] = []
    seen: set[int] = set()

    def _add(n: int) -> None:
        ni = int(n)
        if ni not in seen:
            seen.add(ni)
            out.append(ni)

    for item in raw or ():
        if isinstance(item, str):
            try:
                nids = expand_pg_to_nodes(fem, item)
            except BridgeError:
                nids = _expand_label_to_nodes(fem, item)
            if not nids:
                raise BridgeError(
                    f"LadrunoMembrane: diaphragm_nodes= {item!r} matched 0 nodes."
                )
            for nid in nids:
                _add(int(nid))
        else:
            _add(_resolve_membrane_node_ref(fem, item, "diaphragm_nodes"))
    return tuple(out)


def resolve_ladruno_membrane_emit_context(
    fem: "FEMData",
    spec: Element,
    element_plan: "Iterable[tuple[Element, ElementPlanRows]]" = (),
    fem_eid_to_ops_tag: "Any" = None,
) -> "tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]":
    """The ``(frames, support_nodes, holes, diaphragm_nodes)`` context a
    :class:`LadrunoMembrane` emits with
    (:func:`~apeGmsh.opensees._internal.tag_resolution.set_membrane_context`):
    :func:`resolve_ladruno_membrane_context` and
    :func:`resolve_ladruno_membrane_diaphragm_nodes`.
    """
    return (
        *resolve_ladruno_membrane_context(
            fem, spec, element_plan, fem_eid_to_ops_tag,
        ),
        resolve_ladruno_membrane_diaphragm_nodes(fem, spec),
    )


def ladruno_membrane_named_nodes(
    fem: "FEMData", spec: Element,
) -> "tuple[int, ...]":
    """Every node a :class:`LadrunoMembrane` names besides its outer loop:
    its ``support_nodes=``, hole loops and ``diaphragm_nodes=``, in that
    order (repeats kept).

    The fork gives each one stiffness, so each one is emitted, joins the
    membrane's stage and its rank, and is never skipped as a ``pg=``
    panel's interior node. Its ndf is whatever the rest of the model
    gives it: the class is ndf-adaptive, so ndf inference counts no
    membrane node.
    """
    _fr, supp, holes = resolve_ladruno_membrane_context(
        fem, spec, include_frames=False,
    )
    return (
        *supp,
        *(n for h in holes for n in h),
        *resolve_ladruno_membrane_diaphragm_nodes(fem, spec),
    )


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
    from ...element.shell import LadrunoMembrane

    pg = getattr(spec, "pg", None)
    if isinstance(spec, LadrunoMembrane):
        if pg is not None:
            raw_fanout = expand_pg_to_elements(fem, pg)
            if len(raw_fanout) == 0:
                return raw_fanout
            outer_loop, _holes = extract_pg_polygon_loops(fem, pg, raw_fanout)
            return PGElementFanout(
                np.asarray([MISSING_FEM_ELEMENT_ID], dtype=np.int64),
                np.asarray([outer_loop], dtype=np.int64),
            )
        nodes = resolve_ladruno_membrane_nodes(fem, spec)
        return PGElementFanout(
            np.asarray([MISSING_FEM_ELEMENT_ID], dtype=np.int64),
            np.asarray([nodes], dtype=np.int64),
        )
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
