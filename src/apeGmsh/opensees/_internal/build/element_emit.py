"""element_emit: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import compress
from typing import Any, Iterable, Iterator, Literal, Mapping, cast

import numpy as np

from ..types import Element, Primitive
from ..tag_allocator import TagAllocator
from ..tag_resolution import (
    MISSING_FEM_ELEMENT_ID,
    set_current_fem_element_id,
    set_element_nodes,
    set_membrane_context,
    set_tag_resolver,
)
from ...element.shell import LadrunoMembrane
from .errors import BridgeError
from .pg_expand import (
    ElementPlanRows,
    _describe_pg_cells,
    expand_spec_to_elements,
    resolve_ladruno_membrane_context,
)
from .ownership import SortedIntToInt, _EMPTY_INT64
from .transforms import _element_transf
from .validators.mesh import check_quad_element_on_triangles
from .validators.materials import sweep_asdconcrete_element_size
from .builder_scope import close_builder_ndf_bracket, open_builder_ndf_bracket


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData


#: ``apeSees(fem, element_tags=...)`` modes (ADR 0111 D2).
ElementTagMode = Literal["sequential", "fem"]


ELEMENT_TAG_MODES: "tuple[str, ...]" = ("sequential", "fem")


def reserve_fem_element_tags(
    elements: "Iterable[Element]",
    fem: "FEMData",
    tags: TagAllocator,
) -> int:
    """Reserve the FEM element-id range for ``element_tags="fem"``.

    ADR 0111 D2: with ``"fem"`` every physical-group element's OpenSees
    tag IS its FEM element id. Called once per emit, before any element
    tag is allocated, it checks that those ids can be tags (each ``> 0``,
    no id fanned out by two specs or twice by one) and raises the
    ``"element"`` counter to their maximum (or the snapshot's largest
    element id, if larger), so every element the bridge synthesises
    afterwards (node-pair springs, interface ``zeroLength``,
    embedded / rebar / rigid-body / coupling elements) lands strictly
    above the range and cannot collide with a FEM id. Returns that
    maximum (``0`` for an empty snapshot with no physical-group spec).

    Raises :class:`BridgeError` listing the offending ids.
    """
    blocks: "list[np.ndarray]" = []
    owners: "list[np.ndarray]" = []
    specs: "list[Element]" = []
    for spec in elements:
        if getattr(spec, "pg", None) is None:
            continue                        # node-pair form: no FEM id
        eids = np.asarray(expand_spec_to_elements(fem, spec).eids,
                          dtype=np.int64)
        eids = eids[eids != MISSING_FEM_ELEMENT_ID]
        if eids.size == 0:
            continue
        blocks.append(eids)
        owners.append(np.full(eids.shape[0], len(specs), dtype=np.int64))
        specs.append(spec)
    all_eids = (np.concatenate(blocks) if blocks
                else np.empty((0,), dtype=np.int64))
    bad = all_eids[all_eids <= 0]
    if bad.size:
        raise BridgeError(
            f"apeSees(element_tags='fem'): {bad.size} FEM element id(s) "
            f"are <= 0 (first: {sorted(set(bad.tolist()))[:10]}); an "
            f"OpenSees element tag must be a positive integer. Use "
            f"element_tags='sequential' or renumber the mesh."
        )
    uniq, counts = np.unique(all_eids, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        first = int(dup[0])
        who = sorted({
            f"{type(specs[int(k)]).__name__}(pg={specs[int(k)].pg!r})"  # type: ignore[attr-defined]
            for k in np.concatenate(owners)[all_eids == first]
        })
        raise BridgeError(
            f"apeSees(element_tags='fem'): {dup.size} FEM element id(s) "
            f"would be emitted more than once (first: "
            f"{dup[:10].tolist()}); element {first} is fanned out by "
            f"{', '.join(who)}. With FEM ids as tags every element "
            f"belongs to exactly one element declaration — split the "
            f"physical groups, or use element_tags='sequential'."
        )
    top = int(uniq[-1]) if uniq.size else 0
    # Reserve through the snapshot's largest element id too, not only the
    # emitted ones: a synthesised tag must not reuse the id of a FEM
    # element that exists but is not emitted (a point carrier, an
    # unassigned group), or a results reader keyed by FEM id would take
    # the synthesised element for it.
    snap_ids = getattr(getattr(fem, "elements", None), "ids", None)
    if snap_ids is not None and len(snap_ids):
        top = max(top, int(np.max(np.asarray(snap_ids, dtype=np.int64))))
    tags.reserve_through("element", top)
    return top


def allocate_element_tags(
    elements: "Iterable[Element]",
    fem: "FEMData",
    tags: TagAllocator,
    *,
    element_tags: ElementTagMode = "sequential",
) -> "list[tuple[Element, ElementPlanRows]]":
    """Allocate canonical element tags up-front, return per-spec plan.

    ``element_tags="fem"`` (ADR 0111 D2) makes each physical-group row's
    tag its FEM element id (explicit per-row ``tags``); node-pair specs
    still draw from the counter. :func:`reserve_fem_element_tags` must
    have run on ``tags`` first (``BuiltModel.emit`` does), so the counter
    already sits at or above every FEM id; this function refuses
    otherwise rather than risk a counter tag inside the FEM range.

    Returns a list of ``(spec, ElementPlanRows)`` so the per-rank
    fan-out can simply look up each element's pre-allocated tag instead
    of consuming the allocator per-rank (which would produce diverging
    tag numbering across ranks — ADR 0027 §"Tag determinism").
    Iteration order matches the flat fan-out so tags are byte-identical
    to the unpartitioned path for shared elements.

    ADR 0065 v2 / plan_emit_memory_columnar.md B1: each spec's rows are
    a columnar :class:`ElementPlanRows` (int64 arrays + a ``tag_start``)
    rather than a resident ``list[tuple[int, tuple[int, ...], int]]``.
    Tags stay per-kind sequential: element tags are allocated ONLY here,
    spec by spec in iteration order, so a spec's ``N`` tags are the
    contiguous block ``[tag_start, tag_start + N)`` (verified fact #1 in
    the plan). We reserve that block in one
    :meth:`TagAllocator.allocate_block` call — same counter semantics as
    ``N`` per-element ``allocate("element")`` calls, so tag numbering is
    byte-identical — and the plan derives row ``i``'s tag positionally.
    The fan-out's connectivity arrays are shared by reference (they are
    already read-only, memoised per snapshot), so no per-element boxing
    survives the call.

    Iterating the returned plan yields exactly the same
    ``(eid, conn_tuple, ele_tag)`` triples in the same order as the old
    list-of-tuples form, so every downstream consumer is unchanged.
    """
    plan: list[tuple[Element, ElementPlanRows]] = []
    for spec in elements:
        # ADR 0049: node-pair spec (pg=None) -> single synthetic element
        # (MISSING_FEM_ELEMENT_ID, (i, j)) via expand_spec_to_elements.
        fanout = expand_spec_to_elements(fem, spec)
        n = len(fanout)
        # Fail loud on a PG-form element declaration that fans out to
        # NOTHING (adversarial review of A10): a physical group with no
        # elements of the primitive's dimension in this FEM snapshot used
        # to emit the section/material lines and silently zero element
        # lines. ``pg is None`` is the node-pair form, which always
        # fans out to exactly one synthetic element and can never hit
        # this branch.
        pg = getattr(spec, "pg", None)
        if n == 0 and pg is not None:
            raise BridgeError(
                f"{type(spec).__name__}(pg={pg!r}) selected 0 elements: "
                f"{_describe_pg_cells(fem, pg)}, but none of them are "
                f"present in the FEM snapshot handed to apeSees(fem) — "
                f"check that get_fem_data(dim=...) was not called with a "
                f"dim that excludes this group's cells."
            )
        if pg is not None:
            # #1327: a quad-only element on a triangle PG fails here,
            # naming the PG, instead of per element inside ``_emit``.
            check_quad_element_on_triangles(spec, fanout)
        if (
            element_tags == "fem"
            and pg is not None
            and not (n == 1 and int(fanout.eids[0]) == MISSING_FEM_ELEMENT_ID)
        ):
            eids = np.asarray(fanout.eids, dtype=np.int64)
            if n and int(eids.max()) > tags.last("element"):
                raise BridgeError(
                    "internal: element_tags='fem' but the FEM element-id "
                    "range was not reserved before allocation (call "
                    "reserve_fem_element_tags first); a counter tag "
                    "could collide with a FEM id."
                )
            plan.append(
                (spec, ElementPlanRows(fanout.eids, fanout.conn, 0,
                                       tags=eids))
            )
            continue
        tag_start = tags.allocate_block("element", n)
        # Share the fan-out's arrays by reference — they are read-only
        # (memoised) so the plan and the fan-out cache alias one buffer
        # instead of re-boxing connectivity per spec.
        plan.append(
            (spec, ElementPlanRows(fanout.eids, fanout.conn, tag_start))
        )
    return plan


def _plan_owner_ranks(
    pre_allocated: "ElementPlanRows | list[tuple[int, tuple[int, ...], int]]",
    element_owner: "SortedIntToInt | Mapping[int, int]",
) -> "np.ndarray":
    """Owner rank per plan row as ``int64``, ``-1`` where unowned.

    Resolves the whole plan in ONE vectorised ``searchsorted`` via
    :meth:`SortedIntToInt.translate_ranks` rather than a scalar probe per
    element. ``SortedIntToInt.get`` costs ~1.4 us of numpy dispatch per
    call versus ~90 ns for the ``dict`` B3 replaced, and the element
    fan-out probes it once per element per rank — so the scalar form put
    a flat ~3 x n_elements numpy calls into every partitioned emit.

    ``element_owner`` is duck-typed: tests and unpartitioned callers pass
    a plain ``dict``, which is already hash-speed and takes the scalar
    path unchanged.
    """
    n = len(pre_allocated)
    translate = getattr(element_owner, "translate_ranks", None)
    if translate is not None:
        eids = getattr(pre_allocated, "eids", None)
        if eids is None:
            eids = np.fromiter(
                (int(e) for e, _, _ in pre_allocated),
                dtype=np.int64,
                count=n,
            )
        return cast("np.ndarray", translate(eids, -1))
    # Only a plain Mapping reaches here — SortedIntToInt always carries
    # translate_ranks — so `.get` with a default is a plain int.
    owner_map = cast("Mapping[int, int]", element_owner)
    return np.fromiter(
        (int(owner_map.get(int(e), -1)) for e, _, _ in pre_allocated),
        dtype=np.int64,
        count=n,
    )


class LazyRankBuckets:
    """Lazy per-rank element-plan buckets for ONE spec (ADR 0100 D4).

    The eager form this replaces (``bucket_pre_allocated_by_rank``)
    materialised every rank's ``select_rows`` copy up front — eids +
    connectivity + tags for the whole model, the emit path's **3rd
    connectivity copy** (~80 B/row; 1st = FEMData, 2nd =
    ``_PG_FANOUT_CACHE``).  Here the resident state is one int64
    permutation of the spec's rows grouped by owner rank (8 B/row)
    plus O(ranks) offsets; :meth:`get` materialises a single rank's
    rows on demand and the caller lets them die when that rank's block
    closes.  Rows with no owner (sentinel -1) are dropped exactly as
    the eager bucketing dropped them — they never emitted on any rank.

    Row order inside a bucket is plan order: the stable argsort keeps
    ascending original positions within each rank run, matching the
    eager path's ``np.nonzero`` positions — the emitted deck is
    byte-identical.

    :meth:`count` answers the ADR 0099 S5 hoist pre-pass (per-rank
    presence for ALL ranks *before* the rank loop) without
    materialising any rows.
    """

    __slots__ = ("_plan", "_order", "_ranks", "_starts", "_ends")

    def __init__(
        self,
        pre_allocated: "ElementPlanRows",
        element_owner: "SortedIntToInt | Mapping[int, int]",
    ) -> None:
        self._plan = pre_allocated
        n = len(pre_allocated)
        if n == 0:
            self._order = _EMPTY_INT64
            self._ranks = _EMPTY_INT64
            self._starts = _EMPTY_INT64
            self._ends = _EMPTY_INT64
            return
        owners = _plan_owner_ranks(pre_allocated, element_owner)
        order = np.argsort(owners, kind="stable")
        sorted_owners = owners[order]
        ranks, starts = np.unique(sorted_owners, return_index=True)
        ends = np.append(starts[1:], n)
        keep = ranks >= 0
        self._order = order
        self._ranks = ranks[keep]
        self._starts = starts[keep]
        self._ends = ends[keep]

    def _find(self, rank: int) -> int:
        r = self._ranks
        i = int(np.searchsorted(r, rank))
        if i < r.shape[0] and int(r[i]) == rank:
            return i
        return -1

    def count(self, rank: int) -> int:
        """Row count for ``rank`` — no materialisation."""
        i = self._find(int(rank))
        if i < 0:
            return 0
        return int(self._ends[i] - self._starts[i])

    def get(
        self, rank: int, default: "Any" = None,
    ) -> "ElementPlanRows | Any":
        """Materialise ``rank``'s rows; ``default`` when it owns none.

        Same surface as the eager bucket-dict's ``.get``.  Each call
        builds a FRESH :class:`ElementPlanRows` subset — consume it and
        let it die; retaining every rank's result rebuilds the eager
        residency this class exists to remove.
        """
        i = self._find(int(rank))
        if i < 0:
            return default
        idx = self._order[int(self._starts[i]):int(self._ends[i])]
        return self._plan.select_rows(idx)


#: Rows per owner-probe block in :func:`_owned_mask`.  ``translate_ranks``
#: allocates ~5 ``int64`` temporaries per query row, so probing a whole
#: rank bucket at once peaks at ~40 B/row; a fixed block bounds that.
_OWNER_PROBE_BLOCK = 16_384


def _owned_mask(
    pre_allocated: "ElementPlanRows | list[tuple[int, tuple[int, ...], int]]",
    element_owner: "SortedIntToInt | Mapping[int, int]",
    partition_rank: int,
) -> "np.ndarray":
    """``bool[N]``: row ``i`` is owned by ``partition_rank``.

    Same answer as ``_plan_owner_ranks(...) == partition_rank``.  A
    columnar plan with a columnar owner map is probed in fixed blocks,
    so the owner probe's temporaries stay O(block) rather than O(rows);
    any other pairing takes :func:`_plan_owner_ranks` whole.
    """
    translate = getattr(element_owner, "translate_ranks", None)
    eids = getattr(pre_allocated, "eids", None)
    if translate is None or eids is None:
        return cast(
            "np.ndarray",
            _plan_owner_ranks(pre_allocated, element_owner) == partition_rank,
        )
    n =int(eids.shape[0])
    mask = np.empty(n, dtype=bool)
    for s in range(0, n, _OWNER_PROBE_BLOCK):
        e = min(s + _OWNER_PROBE_BLOCK, n)
        np.equal(translate(eids[s:e], -1), partition_rank, out=mask[s:e])
    return mask


class _OwnedRows:
    """Lazy, re-iterable view of one rank's owned rows of an element plan.

    Holding no per-row Python objects is the point: a list of the owned
    rows would keep every row's node-tag tuple alive for the whole spec
    (ADR 0100).  ``mask`` is ``None`` when
    the whole plan is owned (a rank bucket), else a ``bool[N]`` row
    mask.  Each iteration walks the plan afresh, yielding one transient
    row at a time.

    Iterating the view yields ``(eid, node_tags)`` pairs — the shape
    :func:`sweep_asdconcrete_element_size` reads, with ``len`` as its
    count.  :meth:`triples` yields the full ``(eid, node_tags, ele_tag)``
    rows for the emit loop.
    """

    __slots__ = ("_plan", "_mask", "_n")

    def __init__(
        self,
        plan: "ElementPlanRows | list[tuple[int, tuple[int, ...], int]]",
        mask: "np.ndarray | None",
        n: int,
    ) -> None:
        self._plan = plan
        self._mask = mask
        self._n = n

    def __len__(self) -> int:
        return self._n

    def triples(self) -> "Iterator[tuple[int, tuple[int, ...], int]]":
        if self._mask is None:
            return iter(self._plan)
        return compress(self._plan, self._mask)

    def __iter__(self) -> "Iterator[tuple[int, tuple[int, ...]]]":
        for eid, node_tags, _ in self.triples():
            yield eid, node_tags


def emit_element_spec_partitioned(
    spec: Element,
    emitter: "Emitter",
    fem: "FEMData",
    pre_allocated: "ElementPlanRows | list[tuple[int, tuple[int, ...], int]]",
    base_resolver: object,
    transf_tag_for_element: dict[tuple[int, int], int] | None,
    partition_rank: int,
    element_owner: "SortedIntToInt | Mapping[int, int]",
    ndm: int | None = None,
    envelope_ndf: int | None = None,
    element_plan: "Iterable[tuple[Element, ElementPlanRows]]" = (),
    fem_eid_to_ops_tag: "Any" = None,
) -> None:
    """Per-rank element fan-out (ADR 0027).

    Emits ONLY the elements of ``spec.pg`` whose owner-rank matches
    ``partition_rank``.  Tags come from ``pre_allocated`` (built once
    by :func:`allocate_element_tags`) so cross-rank tag identity is
    preserved verbatim per ADR 0027 §"Tag determinism".

    ``ndm`` / ``envelope_ndf`` enable the builder-ndf bracket
    (:func:`open_builder_ndf_bracket`) for gated element families; the
    bracket is skipped when this rank owns no element of the spec, so
    empty ranks never carry stray ``model`` switches.
    """
    if not pre_allocated:
        return

    # One vectorised owner probe for the whole plan, shared by the ADR
    # 0044 sweep and the emit loop below (see :func:`_plan_owner_ranks`),
    # reduced at once to a 1 B/row ownership mask.  -1 is the unowned
    # sentinel, so it can never equal a real rank.  No per-row Python
    # list (owner ranks or row tuples) is held: rows stream from
    # ``pre_allocated`` to the emitter, so only the row being emitted is
    # alive.
    owned_mask = _owned_mask(pre_allocated, element_owner, partition_rank)
    n_owned = int(np.count_nonzero(owned_mask))
    if n_owned == 0:
        return
    # A rank bucket (``LazyRankBuckets.get``) is owned whole, so it skips
    # the mask walk; a node-pair plan or a stray row keeps the filter.
    owned_rows = _OwnedRows(
        pre_allocated, None if n_owned == len(pre_allocated) else owned_mask,
        n_owned,
    )

    # ADR 0044: sweep only this rank's owned elements (avoids cross-rank
    # duplicate warnings — each rank reports its own over-ceiling
    # elements).  The view re-iterates lazily, and only an ASDConcrete
    # spec iterates it at all.  It runs before any element is written.
    # The cast is duck typing: the sweep reads only ``len`` and
    # ``(eid, node_tags)`` iteration, both of which the view provides.
    sweep_asdconcrete_element_size(
        spec, cast("list[tuple[int, tuple[int, ...]]]", owned_rows), fem)

    transf_spec = _element_transf(spec)
    if isinstance(spec, LadrunoMembrane):
        set_membrane_context(
            emitter,
            resolve_ladruno_membrane_context(
                fem,
                spec,
                element_plan=element_plan,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            ),
        )

    bracketed = (
        ndm is not None and envelope_ndf is not None
        and open_builder_ndf_bracket(
            emitter, spec, ndm=ndm, envelope_ndf=envelope_ndf)
    )

    # Partial writes: a raise from ``spec._emit`` (or the resolver) leaves
    # this spec's earlier rows written and the bracket open, and
    # propagates.  The checks that must precede the first write (owner
    # probe, ADR 0044 sweep) run above, over the whole owned set.
    try:
        for eid, node_tags, ele_tag in owned_rows.triples():
            set_element_nodes(emitter, node_tags)
            set_current_fem_element_id(emitter, eid)

            if (
                transf_spec is not None
                and transf_tag_for_element is not None
                and (id(transf_spec), eid) in transf_tag_for_element
            ):
                override_tag = transf_tag_for_element[(id(transf_spec), eid)]
                base = base_resolver
                override = transf_spec

                def _resolver_with_override(
                    p: Primitive,
                    _base: object = base,
                    _override_spec: Primitive = override,
                    _override_tag: int = override_tag,
                ) -> int:
                    if p is _override_spec:
                        return _override_tag
                    return int(_base(p))  # type: ignore[operator]

                set_tag_resolver(emitter, _resolver_with_override)
                try:
                    spec._emit(emitter, ele_tag)
                finally:
                    set_tag_resolver(emitter, base_resolver)  # type: ignore[arg-type]
            else:
                spec._emit(emitter, ele_tag)
    finally:
        if isinstance(spec, LadrunoMembrane):
            set_membrane_context(emitter, None)

    if bracketed:
        close_builder_ndf_bracket(
            emitter, ndm=ndm, envelope_ndf=envelope_ndf,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Cross-partition MP-constraint replication (ADR 0027 §"Decision")
# ---------------------------------------------------------------------------


def _node_coords_safe(fem: "FEMData", node_id: int) -> tuple[float, float, float]:
    """Best-effort ``(x, y, z)`` lookup for ``node_id``.

    Returns zeros if the node id is unknown to the broker (e.g. a
    phantom tag whose coords live on the source record, not on
    ``fem.nodes``); the caller is responsible for using
    :meth:`_emit_phantom_nodes` for those.
    """
    try:
        idx = fem.nodes.index(int(node_id))
        xyz = fem.nodes.coords[idx]
        return float(xyz[0]), float(xyz[1]), float(xyz[2])
    except (KeyError, IndexError, AttributeError):
        return (0.0, 0.0, 0.0)
