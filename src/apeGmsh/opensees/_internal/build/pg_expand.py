"""pg_expand: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import weakref
from typing import Any, Iterable, Iterator, Sequence

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


def resolve_ladruno_membrane_context(
    fem: "FEMData",
    spec: Element,
    element_plan: "Iterable[tuple[Element, ElementPlanRows]]" = (),
    fem_eid_to_ops_tag: "Any" = None,
    *,
    include_frames: bool = True,
) -> "tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]":
    """Resolve ``(frames, support_nodes, holes)`` for a :class:`LadrunoMembrane`."""
    from ...node import Node, NodeSet

    # 1. Frames
    frames_out: list[int] = []
    seen_frames: set[int] = set()

    def _add_frame(t: int) -> None:
        ti = int(t)
        if ti not in seen_frames:
            seen_frames.add(ti)
            frames_out.append(ti)

    raw_frames = getattr(spec, "frames", None) if include_frames else None
    if raw_frames is not None:
        plan_by_spec_id = {id(s): sub for s, sub in element_plan}
        for item in raw_frames:
            if isinstance(item, Element):
                sub = plan_by_spec_id.get(id(item))
                if sub is None or len(sub) == 0:
                    raise BridgeError(
                        f"LadrunoMembrane: frames= Element spec {item!r} has no "
                        f"emitted elements in this model."
                    )
                for _eid, _conn, ele_tag in sub:
                    _add_frame(ele_tag)
            elif isinstance(item, str):
                eids_list: list[int] = []
                try:
                    fan = expand_pg_to_elements(fem, item)
                    eids_list = [int(e) for e in fan.eids.tolist()]
                except BridgeError:
                    labels = getattr(getattr(fem, "elements", None), "labels", None)
                    if labels is not None:
                        try:
                            eids_list = [int(e) for e in labels.element_ids(item)]
                        except (KeyError, ValueError) as exc:
                            raise BridgeError(
                                f"LadrunoMembrane: frames= group/label {item!r} not "
                                f"found in FEM snapshot."
                            ) from exc
                    else:
                        raise
                if not eids_list:
                    raise BridgeError(
                        f"LadrunoMembrane: frames= group/label {item!r} matched 0 elements."
                    )
                for feid in eids_list:
                    mapped = (
                        fem_eid_to_ops_tag.get(int(feid))
                        if fem_eid_to_ops_tag is not None
                        else None
                    )
                    if mapped is None:
                        raise BridgeError(
                            f"LadrunoMembrane: frames= group {item!r} contains FEM "
                            f"element {feid}, which has no emitted OpenSees element."
                        )
                    _add_frame(int(mapped))
            elif isinstance(item, int) and not isinstance(item, bool):
                mapped = (
                    fem_eid_to_ops_tag.get(int(item))
                    if fem_eid_to_ops_tag is not None
                    else None
                )
                _add_frame(int(mapped) if mapped is not None else int(item))

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
    if pg is not None:
        _outer, pg_holes = extract_pg_polygon_loops(fem, pg)
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

    return tuple(frames_out), tuple(supp_out), tuple(holes_out)


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
