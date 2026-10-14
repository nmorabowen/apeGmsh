"""constraints_ranked: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence, cast
from apeGmsh._kernel.record_sets import PartitionSet

import numpy as np

from .records import StiffnessResolver
from .ndf import (
    _emit_node_with_inferred_ndf,
    node_coords_as_floats,
)
from .ownership import (
    FemToOpsTagMap,
    NodePartitionOwners,
    SortedIntToInt,
    build_node_partition_owners,
    runtime_rank_from_partition_record,
)
from .constraints_plan import (
    _mp_element_tagger,
    _mp_plan,
    mp_constraint_pools,
)
from .constraints import (
    _StageConstraintAdapter,
    _emit_equal_dofs,
    _emit_kinematic_couplings,
    _emit_rigid_body_elements,
    _emit_rigid_diaphragms,
    _emit_rigid_links,
    _gather_phantom_nodes,
    _write_one_interpolation,
)
from .element_emit import _node_coords_safe
from .stage_emit import GhostSPOp, emit_ghost_sp_ops


from .errors import BridgeError

if TYPE_CHECKING:
    from ..types import Primitive
    from .records import StageRecord
    from apeGmsh._kernel.records._constraints import (
        ConstraintRecord,
        InterpolationRecord,
    )
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh._kernel.records._partitions import PartitionRecord
    from ..tag_plan import TagPlan


def plan_partitioned_mp_constraints(
    fem: "FEMData",
    claimed_ids: "frozenset[int]",
    node_owners: "NodePartitionOwners",
    element_owner: "SortedIntToInt",
) -> "tuple[dict[int, tuple[float, float, float]], dict[int, _RankConstraintPlan]]":
    """The global partitioned MP-constraint pass's routing, every rank.

    ``(phantom_coords, {rank: _RankConstraintPlan})``: the phantom nodes
    the pass declares, and which records each rank emits and which
    foreign nodes it declares (:func:`_plan_rank_constraints`, ADR 0027),
    over the ranks in the order the per-rank loop opens them. Empty when
    the FEM carries no constraint container. The tag plan resolves it
    once; :func:`emit_mp_constraints_partitioned` reads it.
    """
    node_constraints, surface_constraints = mp_constraint_pools(
        fem, claimed_ids)
    if node_constraints is None and surface_constraints is None:
        return {}, {}
    phantom_coords = (
        _gather_phantom_nodes(node_constraints)
        if node_constraints is not None
        else {}
    )
    plans: "dict[int, _RankConstraintPlan]" = {}
    for idx, part in enumerate(fem.partitions):
        rank = runtime_rank_from_partition_record(part, idx)
        plans[rank] = _plan_rank_constraints(
            node_constraints=node_constraints,
            surface_constraints=surface_constraints,
            partition_rank=rank,
            node_owners=node_owners,
            element_owner=element_owner,
            phantom_tags=set(phantom_coords.keys()),
            # Only a routed snapshot (ADR 0120 D1) pins element-less
            # nodes; a plain PartitionSet pins none.
            pinned=(fem.partitions.pinned
                    if isinstance(fem.partitions, RoutedPartitionSet)
                    else None),
        )
    return phantom_coords, plans


class RoutedPartitionSet(PartitionSet):
    """``fem.partitions`` with the element-less nodes routed (ADR 0120 D1).

    ``pinned`` maps each routed node to the one runtime rank that declares
    it. The MP-constraint planner reads it: a node-pair record whose
    constrained (slave) node is pinned is written on that rank only, so a
    spring's side-node ``equalDOF`` never crosses ranks.
    """

    def __init__(
        self, records: "dict[int, PartitionRecord]",
        pinned: "Mapping[int, int]",
    ) -> None:
        super().__init__(records)
        self.pinned: dict[int, int] = {
            int(k): int(v) for k, v in pinned.items()}


def route_element_less_nodes(
    fem: "FEMData", node_pairs: "Sequence[tuple[int, int]]",
) -> "FEMData":
    """Give every element-less node of a partitioned FEM one rank (ADR 0120 D1).

    A partition holds the nodes of its elements, so a node no element
    references (a ``g.decouple_node`` node, the ground and side nodes of a
    spring bed) is in no partition and no rank declares it. Such a node is
    routed here when it is decoupled or linked: an endpoint of a node-pair
    element (``node_pairs``) or a node of an MP constraint record. The
    links join the element-less nodes into components; a component takes
    the lowest rank among the partitioned nodes it touches (the primary
    owner of :func:`primary_owner_map`), and the first rank when it touches
    none. Each routed node is added to that one rank's partition, so its
    ``node`` line, ``fix``, ``mass``, regions and the node-pair element
    that reaches it are written on that rank only.

    Returns ``fem`` itself when nothing is routed (an unpartitioned FEM, or
    every node already held), else a shallow copy whose ``partitions`` is a
    :class:`RoutedPartitionSet`. The input is not modified, and the copy
    shares its ``nodes._partitions`` (what ``model.h5`` stores), so an
    archive records the mesh partition, not the routing.
    """
    import copy

    from apeGmsh._kernel.records._partitions import PartitionRecord

    parts = list(getattr(fem, "partitions", None) or ())
    if len(parts) < 2:
        return fem
    owners = build_node_partition_owners(fem)
    primary = owners.primary_owner()
    all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    held = np.isin(all_ids, owners._node_ids)
    unowned = {int(n) for n in all_ids[~held]}
    if not unowned:
        return fem

    links: list[tuple[int, ...]] = [
        (int(i), int(j)) for i, j in node_pairs]
    node_cons, surf_cons = mp_constraint_pools(fem, frozenset())
    if node_cons is not None:
        for rec in node_cons:
            links.append(tuple(_constraint_node_ids(rec)))
    interps = getattr(surf_cons, "interpolations", None)
    if interps is not None:
        for rec in interps():
            links.append((int(rec.slave_node),
                          *(int(m) for m in rec.master_nodes)))

    parent: dict[int, int] = {
        int(n): int(n) for n in getattr(fem.nodes, "decoupled_ids", ())
        if int(n) in unowned}

    def _find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    reach: dict[int, int] = {}      # free node -> lowest held rank it links
    for link in links:
        free = [n for n in link if n in unowned]
        if not free:
            continue
        for n in free:
            parent.setdefault(n, n)
        ranks = [r for r in (primary.get(n) for n in link if n not in unowned)
                 if r is not None]
        if ranks:
            low = min(ranks)
            reach[free[0]] = min(low, reach.get(free[0], low))
        for n in free[1:]:
            a, b = _find(free[0]), _find(n)
            if a != b:
                parent[max(a, b)] = min(a, b)
    if not parent:
        return fem
    best: dict[int, int] = {}
    for n, r in reach.items():
        root = _find(n)
        best[root] = min(r, best.get(root, r))
    first = runtime_rank_from_partition_record(parts[0], 0)
    pinned = {n: best.get(_find(n), first) for n in parent}

    extra: dict[int, list[int]] = {}
    for n, r in pinned.items():
        extra.setdefault(r, []).append(n)
    records: "dict[int, PartitionRecord]" = {}
    for idx, rec in enumerate(parts):
        add = extra.get(runtime_rank_from_partition_record(rec, idx))
        node_ids = np.asarray(rec.node_ids, dtype=np.int64)
        if add:
            node_ids = np.union1d(node_ids, np.asarray(add, dtype=np.int64))
        records[int(rec.id)] = PartitionRecord(
            id=rec.id, node_ids=node_ids, element_ids=rec.element_ids,
            weight_sum=rec.weight_sum,
        )
    view = copy.copy(fem)
    view.partitions = RoutedPartitionSet(records, pinned)
    return view


def _emit_foreign_node_declarations(
    emitter: "Emitter",
    fem: "FEMData",
    plan: "_RankConstraintPlan",
    phantom_coords: "Mapping[int, tuple[float, float, float]]",
    inferred_ndf: "dict[int, int]",
    foreign_node_ndf: int | None,
    ghost_sp_ops: "dict[int, list[GhostSPOp]] | None",
) -> None:
    """Emit this rank's foreign-node declarations (ADR 0027 INV-2).

    Phantoms first, then real foreign (ghost) nodes — the order the
    unpartitioned path uses, kept within the rank block so cross-rank
    text stays byte-identical per INV-1.

    Each ghost node carries **its owner's SP constraints** immediately
    after its ``node`` line. That is not cosmetic: a ghost is declared
    because a constraint reaches across the partition, but it owns no
    elements on this rank, so without its ``fix`` its DOFs are free,
    massless and stiffness-less **here** while the owning rank has them
    constrained. Under ``numberer ParallelPlain`` the two ranks then
    disagree about whether those DOFs are constrained, the declaring
    rank contributes an equation nothing fills, and the global matrix is
    **singular** — measured 2026-07-27 as
    ``MumpsParallelSolver … Error -10 … Matrix is Singular Numerically``
    on both a distributed eigensolve (empty spectrum) and an ordinary
    static ``analyze`` (``returned: -3``). Replicating the ``fix`` makes
    the same deck reproduce the single-process oracle to machine
    precision.

    ``ghost_sp_ops`` is the owner's SP command stream **as of this
    declaration point** (ADR 0027 INV-2, amended 2026-07-28): under a
    staged model that means the global ``ops.fix`` tier plus every
    earlier stage's ``s.fix`` / ``s.remove_sp`` plus the declaring
    stage's own, in emit order.  Keeping it in sync after declaration is
    the caller's job — see ``BuiltModel._emit_stages_partitioned``.

    No de-duplication is needed against the owner-side fix pass:
    :func:`_plan_rank_constraints` excludes owned nodes from
    ``foreign_node_tags`` (``if _owns(t): return``), so a rank never
    declares — and never re-fixes — a node it already emitted.

    **Phantoms are deliberately excluded.** They are bridge-invented
    tags (ADR 0027 §"Phantom-node policy") with no owner and no user
    BCs; there is nothing to replicate.
    """
    for tag in sorted(plan.referenced_phantoms):
        xyz = phantom_coords[tag]
        emitter.node(tag, *node_coords_as_floats(xyz), ndf=6)
    for tag in sorted(plan.foreign_node_tags):
        xyz = _node_coords_safe(fem, tag)
        # Foreign (ghost) nodes are owned by another rank but DO appear
        # in the global inferred map (inference walks every element across
        # ranks), so the ghost takes the SAME inferred ndf its owner emits
        # — cross-rank consistency by determinism (ADR 0048). Envelope
        # fallback for nodes inference can't see.
        _emit_node_with_inferred_ndf(
            emitter, inferred_ndf, int(tag),
            (xyz[0], xyz[1], xyz[2]), int(foreign_node_ndf or 0),
        )
        emit_ghost_sp_ops(
            emitter, int(tag), (ghost_sp_ops or {}).get(int(tag), ()),
        )


@dataclass(frozen=True, slots=True)
class StageConstraintRankPlan:
    """One rank's share of a stage's stage-claimed MP constraints.

    Produced by :func:`plan_stage_mp_constraints_partitioned` *before*
    the rank bracket opens and consumed by
    :func:`emit_stage_mp_constraints_partitioned` *inside* it, so the
    per-rank participation decision is made exactly once (ADR 0034
    empty-bracket skip needs it early; the emit needs it late).
    """

    adapter: "_StageConstraintAdapter"
    phantom_coords: "dict[int, tuple[float, float, float]]"
    plan: "_RankConstraintPlan"


def plan_stage_mp_constraints_partitioned(
    stage_records: "Iterable[ConstraintRecord]",
    *,
    partition_rank: int,
    node_owners: "NodePartitionOwners",
    element_owner: "SortedIntToInt",
) -> "StageConstraintRankPlan | None":
    """Does ``partition_rank`` emit anything for this stage's claimed
    MP constraints?

    Returns ``None`` when it does not — the caller keeps that rank's
    bracket **closed** (ADR 0034 / Phase SSI-2.D empty-bracket skip).
    Otherwise returns the prepared state to hand straight to
    :func:`emit_stage_mp_constraints_partitioned`.

    This exists because the stage's per-rank content gate in
    ``_emit_stages_partitioned`` has to know "does this rank touch the
    stage's constraints?" *before* ``partition_open``, while the answer
    only falls out of :func:`_plan_rank_constraints`.  Splitting the
    plan from the emit keeps the gate exact without planning twice.

    **Pure** — no emitter mutation.  The phantom-tag predicate is
    installed by the emit half, so a rank that contributes nothing
    leaves the emitter untouched.
    """
    adapter = _StageConstraintAdapter(stage_records)
    if not adapter._records:
        return None

    phantom_coords = _gather_phantom_nodes(adapter)
    plan = _plan_rank_constraints(
        node_constraints=adapter,
        surface_constraints=adapter,
        partition_rank=partition_rank,
        node_owners=node_owners,
        element_owner=element_owner,
        phantom_tags=set(phantom_coords.keys()),
    )
    if not plan.any():
        return None
    return StageConstraintRankPlan(adapter, phantom_coords, plan)


def emit_stage_mp_constraints_partitioned(
    rank_plan: "StageConstraintRankPlan",
    emitter: "Emitter",
    fem: "FEMData",
    foreign_node_ndf: int | None,
    inferred_ndf: "dict[int, int]",
    tag_plan: "TagPlan",
    *,
    ndm: int,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    ghost_sp_ops: "dict[int, list[GhostSPOp]] | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
) -> None:
    """Per-rank stage-bound MP-constraint fan-out.

    Mirrors :func:`emit_mp_constraints_partitioned` step-for-step using
    the stage's flat record pool via :class:`_StageConstraintAdapter`.
    Same rank-replication rules (replicate-on-both for node couplings;
    canonical host rank for ``ASDEmbeddedNodeElement``) and same
    foreign-node declaration order (phantoms first, then real foreign
    nodes) as the global partitioned pass.

    ``rank_plan`` comes from
    :func:`plan_stage_mp_constraints_partitioned`, which already
    established that this rank contributes something — there is no
    no-op path here. A copy that a stage's constraint handler would
    multiply was refused before the emit began
    (:func:`refuse_multiplied_mp_constraints`).
    """
    from ..tag_resolution import ATTR_PHANTOM_NODE_TAGS, set_phantom_node_tags

    adapter = rank_plan.adapter
    phantom_coords = rank_plan.phantom_coords
    plan = rank_plan.plan

    if phantom_coords:
        existing: frozenset[int] = getattr(emitter, ATTR_PHANTOM_NODE_TAGS, frozenset())
        set_phantom_node_tags(
            emitter, set(existing) | set(phantom_coords.keys()),
        )

    # Foreign-node declarations — phantoms first, then real foreign
    # nodes (INV-2 within the rank block), each ghost carrying its
    # owner's SP constraints.
    _emit_foreign_node_declarations(
        emitter, fem, plan, phantom_coords, inferred_ndf,
        foreign_node_ndf, ghost_sp_ops,
    )

    # Constraint emission — same ordering as the unpartitioned path.
    ids = plan.allowed_record_ids
    _emit_rigid_links(emitter, adapter, allowed_ids=ids)
    _emit_rigid_body_elements(emitter, adapter, tag_plan, allowed_ids=ids)
    _emit_equal_dofs(emitter, adapter, allowed_ids=ids)
    _emit_rigid_diaphragms(emitter, adapter, allowed_ids=ids)
    _emit_kinematic_couplings(
        emitter, adapter, tag_plan, allowed_ids=ids,
        fem_eid_to_ops_tag=fem_eid_to_ops_tag,
    )

    if plan.embedded_records:
        _emit_surface_couplings_for_rank(
            emitter, plan.embedded_records, tag_plan,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            stiffness_resolver=stiffness_resolver,
        )


def emit_mp_constraints_partitioned(
    emitter: "Emitter",
    fem: "FEMData",
    partition_rank: int,
    foreign_node_ndf: int | None,
    inferred_ndf: "dict[int, int]",
    tag_plan: "TagPlan",
    *,
    ndm: int,
    claimed_ids: "frozenset[int]" = frozenset(),
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    ghost_sp_ops: "dict[int, list[GhostSPOp]] | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
) -> "frozenset[int]":
    """Per-rank MP-constraint fan-out (ADR 0027 §"Decision").

    For ``partition_rank``:

    1. Collect every constraint that touches this rank — meaning any
       node it references lives on this rank (or, for embedded-node /
       ASDEmbeddedNodeElement, the host element lives on this rank).
    2. Compute the set of foreign nodes referenced by those
       constraints (nodes not in this rank's owner set, plus phantoms).
    3. Emit the foreign-node declarations FIRST (INV-2): regular
       foreign nodes via ``node(tag, *xyz[, -ndf K])`` — per-node ndf
       taken from the **global inferred map** (ADR 0048,
       :func:`infer_node_ndf`), falling back to ``foreign_node_ndf``
       (the ``ops.model`` envelope) for nodes the map doesn't cover.
       Because the inferred map is global + deterministic, a ghost node
       resolves to the SAME ndf its owner rank emits — cross-rank
       consistency without per-rank broker state.  Phantoms via
       ``node(tag, *xyz, ndf=6)``.  Each ghost also carries its owner's
       SP lines replayed from ``ghost_sp_ops`` — omitting them leaves a
       free, massless DOF on this rank and makes the global matrix
       singular; see :func:`_emit_foreign_node_declarations`.
    4. Emit the constraints in the same order as the unpartitioned
       :func:`emit_mp_constraints` (phantom-node pre-step → rigidLink
       → equalDOF → rigidDiaphragm → kinematic_coupling →
       ASDEmbeddedNodeElement) so cross-rank text is byte-identical
       per INV-1.

    Returns the real (non-phantom) ghost tags this rank declared, so a
    staged caller can keep their SP state in sync with the owner across
    later stage blocks (ADR 0027 INV-2).

    Steps 1-2 (which records emit on this rank, and which foreign nodes
    it declares) are resolved once per tag plan, every rank at once, by
    :func:`plan_partitioned_mp_constraints` (ADR 0114 D4, amended); this
    reads them, and the element tags, from ``tag_plan`` (:func:`_mp_plan`).

    Replication is exact only under ``Transformation``; a copy another
    handler would multiply was refused before the emit began
    (:func:`refuse_multiplied_mp_constraints`).
    """
    from ..tag_resolution import set_phantom_node_tags

    mp_plan = _mp_plan(tag_plan, fem)
    mp_plan.check_claims(claimed_ids)
    # Stage-claimed records are filtered out — they emit per-stage via
    # ``emit_stage_mp_constraints_partitioned`` instead.
    node_constraints, surface_constraints = mp_constraint_pools(
        fem, claimed_ids)
    if node_constraints is None and surface_constraints is None:
        return frozenset()

    # Which constraints emit on this rank, plus all the foreign-node
    # tags they reference (replicate-on-both / replicate-everywhere-
    # with-a-slave rules from ADR 0027), and the phantom nodes.
    phantom_coords, plan = mp_plan.rank_constraints(partition_rank)

    # Pre-load the phantom-tag predicate on the emitter (ADR 0033 —
    # stateless replacement for the prior phantom-mode flag).
    # Per-rank brokers see identical phantom tags (the resolver
    # canonicalises across ranks per ADR 0027 INV-3), so this set is
    # consistent across the OpenSeesMP fan-out.
    set_phantom_node_tags(emitter, set(phantom_coords.keys()))

    if not plan.any():
        return frozenset()

    # -- 1. Foreign-node declarations (INV-2). ---------------------------
    # Phantoms first (their tags are 6-DOF regardless of model ndf,
    # mirroring the unpartitioned phantom-node-first invariant within
    # this rank's block per ADR 0027 §"Phantom-node policy"), then real
    # ghosts — each with its owner's ndf (ADR 0048) and its owner's SP
    # constraints.
    _emit_foreign_node_declarations(
        emitter, fem, plan, phantom_coords, inferred_ndf,
        foreign_node_ndf, ghost_sp_ops,
    )

    # -- 2. Constraint emission, mirroring the unpartitioned order. ------
    if node_constraints is not None:
        ids = plan.allowed_record_ids
        _emit_rigid_links(emitter, node_constraints, allowed_ids=ids)
        _emit_rigid_body_elements(
            emitter, node_constraints, tag_plan, allowed_ids=ids,
        )
        _emit_equal_dofs(emitter, node_constraints, allowed_ids=ids)
        _emit_rigid_diaphragms(emitter, node_constraints, allowed_ids=ids)
        _emit_kinematic_couplings(
            emitter, node_constraints, tag_plan, allowed_ids=ids,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )

    # ASDEmbeddedNodeElement: only the host-element-owning rank emits.
    if surface_constraints is not None and plan.embedded_records:
        _emit_surface_couplings_for_rank(
            emitter, plan.embedded_records, tag_plan,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            stiffness_resolver=stiffness_resolver,
        )

    # equationConstraint (ADR 0068 Open item 2): replicated on every rank
    # that owns the slave or any master (see _plan_rank_constraints). Each
    # row is byte-identical across ranks — _emit_one_interpolation routes the
    # equation enforce to _emit_equation_tie, which allocates no element tag,
    # so replication across ranks does not perturb the element-tag stream.
    if surface_constraints is not None and plan.equation_records:
        _emit_surface_couplings_for_rank(
            emitter, plan.equation_records, tag_plan,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )

    return plan.foreign_node_tags


def refuse_multiplied_mp_constraints(
    fem: "FEMData",
    tag_plan: "TagPlan",
    node_owners: "NodePartitionOwners",
    element_owner: "SortedIntToInt",
    *,
    pre_element: "Sequence[Primitive]",
    chain_forced: bool,
    has_equation_tie: bool,
    stage_records: "Sequence[StageRecord]",
) -> None:
    """Refuse a partitioned emit whose MP copies a handler would multiply.

    Called once, before the partitioned emit writes anything.

    The partitioned emit writes a replicable record (equalDOF, rigidLink,
    rigidDiaphragm, rigid_body chain, node_to_surface) on every rank that
    holds one of its nodes (ADR 0027). ``Transformation`` needs that: it
    eliminates the constrained DOFs on each rank. ``Penalty``,
    ``Lagrange`` and ``Auto`` instead assemble one penalty or multiplier
    per ``MP_Constraint`` copy, so the partitioned model is stiffer than
    the serial one (measured 4.9e-5 on 2 ranks and 7.3e-5 on 4 on the
    eigenvalues of a rigid diaphragm spanning ranks), and nothing warned.
    Writing each record on one rank instead is exact for a static solve
    but not for ``eigen``: a rank that holds no ``PenaltyMP_FE`` disagrees
    with the others in ``ArpackSOE``'s diagonal-mass flag, and the spectrum
    collapses to 1.0. So the emit refuses. An ``enforce="equation"`` tie
    is written as an ``equationConstraint`` on every rank that holds its
    slave or a master (ADR 0068 Open item 2). Under a declared ``Penalty``
    each copy adds its own penalty term, so a tie spanning ranks is
    refused the same way (#1724 item 1). ``Lagrange``,
    ``LadrunoProjection`` and the auto handler keep ADR 0068's
    replication; ``Transformation`` cannot enforce the tie (INV-4).

    The handler checked is the one each analysis runs under:

    * unstaged: the handler declared in ``pre_element``. When none is
      declared, the bridge auto-emits ``Transformation``, or, when
      ``has_equation_tie``, ``LadrunoProjection`` under an explicit
      integrator and ``Lagrange`` otherwise. When ``chain_forced`` (the
      emitter suppresses the analysis-chain auto-emit: the ARPACK modal
      deck, which forces ``constraints Transformation`` before ``eigen``,
      ADR 0077 INV-10) and ``pre_element`` declares no ``Analysis``, the
      deck runs under ``Transformation``. A declared ``Analysis`` is
      built with the declared handler, so it is still checked;
    * staged: the global records run under every stage's handler, and a
      stage's claimed records under its own and every later stage's
      (a record stays in the domain). A stage that declares none runs
      OpenSees' default ``Plain``.

    ``Transformation`` and ``Plain`` (which ignores MP constraints) keep
    the replication. Any other handler raises :class:`BridgeError` when a
    record it runs under is written on two ranks or more.
    """
    from ...analysis.constraint_handler import Plain, Transformation
    from ...apesees import _is_explicit_integrator
    from ..types import Analysis, ConstraintHandler, Integrator

    def multiplying(handlers: "Iterable[Primitive | None]", *,
                    auto: str) -> "list[str]":
        names = set()
        for h in handlers:
            if h is None:
                if auto:
                    names.add(auto)
            elif not isinstance(h, (Transformation, Plain)):
                names.add(type(h).__name__)
        return sorted(names)

    if stage_records:
        stage_handlers = [st.constraints for st in stage_records]
        global_names = multiplying(stage_handlers, auto="")
    else:
        stage_handlers = []
        declared = next(
            (p for p in pre_element if isinstance(p, ConstraintHandler)), None)
        integrator = next(
            (p for p in pre_element if isinstance(p, Integrator)), None)
        auto = ""
        if has_equation_tie:
            auto = ("LadrunoProjection" if _is_explicit_integrator(integrator)
                    else "Lagrange") + " (auto-emitted for an equation tie)"
        forced = chain_forced and not any(
            isinstance(p, Analysis) for p in pre_element)
        global_names = [] if forced else multiplying((declared,), auto=auto)
    if global_names:
        mp_plan = _mp_plan(tag_plan, fem)
        if mp_plan.rank_plans:
            _refuse_copies(global_names, mp_plan.rank_plans.values())

    ranks = [runtime_rank_from_partition_record(part, idx)
             for idx, part in enumerate(fem.partitions)]
    for i, stage in enumerate(stage_records):
        records = stage.stage_constraint_records
        if not records:
            continue
        names = multiplying(stage_handlers[i:], auto="")
        if not names:
            continue
        plans = []
        for rank in ranks:
            rp = plan_stage_mp_constraints_partitioned(
                records, partition_rank=rank,
                node_owners=node_owners, element_owner=element_owner)
            if rp is not None:
                plans.append(rp.plan)
        _refuse_copies(names, plans)


def _refuse_copies(
    names: "Sequence[str]", plans: "Iterable[_RankConstraintPlan]",
) -> None:
    """Raise when a plan in ``plans`` writes a record another rank writes.

    A replicable record is flagged by its own plan (``replicated_records``).
    An ``enforce="equation"`` tie is written on every rank that holds its
    slave or a master (``equation_records``), so it is copied when two
    plans or more carry the same record. Only ``Penalty`` is refused for
    that (#1724 item 1): each copy adds its own penalty term. ADR 0068
    Open item 2 keeps the replication under the EQ-capable handlers. The
    equation tie is checked first, because ``Transformation``, the
    replicable record's remedy, cannot enforce it (ADR 0068 INV-4).
    """
    plans = list(plans)
    if "Penalty" in names:
        writers: dict[int, int] = {}
        for p in plans:
            for eq in p.equation_records:
                writers[id(eq)] = writers.get(id(eq), 0) + 1
        eq = next((eq for p in plans for eq in p.equation_records
                   if writers[id(eq)] > 1), None)
        if eq is not None:
            name = getattr(eq, "name", None) or "<unnamed>"
            slave = getattr(eq, "slave_node", "?")
            raise BridgeError(
                f"A partitioned emit would write the equationConstraint tie "
                f"{name!r} (slave node {slave}) on {writers[id(eq)]} ranks "
                f"under the constraint handler {', '.join(names)}. Each "
                "rank's copy adds its own penalty term, so the partitioned "
                "model would silently answer stiffer than the serial one "
                "(#1625). Nothing was written. Run serially "
                "(ops.tcl(path, flat=True)), or declare "
                "ops.constraints.Lagrange(), under which the partitioned "
                "emit replicates equation ties across ranks (ADR 0068 Open "
                "item 2)."
            )
    copied = next((p.replicated_records for p in plans
                   if p.replicated_records), ())
    if not copied:
        return
    rec = copied[0]
    kind = getattr(rec, "kind", type(rec).__name__)
    name = getattr(rec, "name", None) or "<unnamed>"
    master = getattr(rec, "master_node", "?")
    raise BridgeError(
        f"A partitioned emit would write the {kind} constraint {name!r} "
        f"(master node {master}) on more than one rank under the "
        f"constraint handler {', '.join(names)}. Each rank's copy assembles "
        "its own penalty or multiplier, so the partitioned model would "
        "silently answer stiffer than the serial one (#1625). Nothing was "
        "written. Declare ops.constraints.Transformation(), which is exact "
        "across ranks, or run serially (ops.tcl(path, flat=True))."
    )


@dataclass(frozen=True, slots=True)
class _RankConstraintPlan:
    """What the per-rank MP-constraint pass will emit on this rank."""

    allowed_record_ids: frozenset[int]
    foreign_node_tags: frozenset[int]
    referenced_phantoms: frozenset[int]
    embedded_records: tuple[object, ...]
    # ADR 0068 Open item 2: enforce="equation" ties replicated on every
    # rank that owns the slave OR any master (domain-level EQ_Constraint,
    # NOT a single-host-rank element). Emitted via _emit_equation_tie.
    equation_records: tuple[object, ...] = ()
    # The replicable records (equalDOF, rigidLink, rigidDiaphragm,
    # rigid_body chain, node_to_surface) this rank writes that another
    # rank writes too: a non-phantom node of theirs is foreign here or
    # also held by another rank.
    replicated_records: tuple[object, ...] = ()

    def any(self) -> bool:
        return (
            bool(self.allowed_record_ids)
            or bool(self.embedded_records)
            or bool(self.equation_records)
        )


def _constraint_node_ids(rec: object) -> "Iterator[int]":
    """Yield every node id a constraint record can reference.

    Deliberately a **superset** gatherer: it reads the union of the
    node-id attributes across record kinds instead of re-deriving the
    per-kind branching in :func:`_plan_rank_constraints`. Over-gathering
    costs one extra dict entry; missing an id would only fall back to a
    scalar lookup. Biasing that way keeps this from silently drifting
    out of step with the planner's dispatch.
    """
    for attr in ("master_node", "slave_node"):
        v = getattr(rec, attr, None)
        if v is not None:
            yield int(v)
    for attr in ("master_nodes", "slave_nodes"):
        vs = getattr(rec, attr, None)
        if vs is not None:
            for v in vs:
                yield int(v)
    for sub in ("rigid_link_records", "equal_dof_records"):
        pairs = getattr(rec, sub, None)
        if pairs is None:
            continue
        for p in pairs:
            yield int(p.master_node)
            yield int(p.slave_node)


def _plan_rank_constraints(
    *,
    node_constraints: Iterable[object] | None,
    surface_constraints: object,
    partition_rank: int,
    node_owners: "NodePartitionOwners | Mapping[int, frozenset[int]]",
    element_owner: "SortedIntToInt",
    phantom_tags: set[int],
    pinned: "Mapping[int, int] | None" = None,
) -> _RankConstraintPlan:
    """Decide which constraint records emit on ``partition_rank``.

    ``pinned`` (ADR 0120 D1, :class:`RoutedPartitionSet`) maps each routed
    element-less node to its one rank: a node-pair record whose slave is
    pinned is written on that rank only, not replicated.
    """
    from apeGmsh._kernel.records._constraints import (
        InterpolationRecord,
        NodeGroupRecord,
        NodePairRecord,
        NodeToSurfaceRecord,
    )

    allowed_ids: set[int] = set()
    foreign_nodes: set[int] = set()
    referenced_phantoms: set[int] = set()
    embedded: list[object] = []
    equation: list[object] = []
    replicated: list[object] = []

    # Materialise both record streams up front so every node id they can
    # reference is resolved against ``node_owners`` in ONE vectorised
    # probe (:meth:`NodePartitionOwners.resolve_many`). The planner's
    # per-node lookups below then run at dict speed; querying the
    # columnar map one node at a time inside these loops costs a scalar
    # ``np.searchsorted`` each (~1.4 us of numpy dispatch) and made
    # partitioned emit 2-3x slower, growing with rank count.
    node_recs: list[object] = (
        [] if node_constraints is None else list(node_constraints)
    )
    interps_iter = getattr(surface_constraints, "interpolations", None)
    interp_recs: "list[InterpolationRecord]" = (
        []
        if surface_constraints is None or interps_iter is None
        else [r for r in interps_iter() if isinstance(r, InterpolationRecord)]
    )

    def _referenced_ids() -> "Iterator[int]":
        for r in node_recs:
            yield from _constraint_node_ids(r)
        # Interpolations are already type-filtered above, so read their
        # two node attributes directly rather than paying the generic
        # gatherer's getattr sweep on what is usually the bulk of the
        # records.
        for r in interp_recs:
            yield int(r.slave_node)
            for mn in r.master_nodes:
                yield int(mn)

    # ``node_owners`` is duck-typed: the build path passes a
    # NodePartitionOwners, but tests and unpartitioned callers pass a
    # plain ``dict``. A dict is already hash-speed, so batching only
    # applies to the columnar form.
    _resolve = getattr(node_owners, "resolve_many", None)
    owners: "Mapping[int, frozenset[int]]" = (
        _resolve(_referenced_ids())
        if _resolve is not None
        else cast("Mapping[int, frozenset[int]]", node_owners)
    )

    def _owns(node_tag: int) -> bool:
        t = int(node_tag)
        o = owners.get(t)
        if o is None:  # not a constraint node — fall back to the map
            o = node_owners.get(t, frozenset())
        return partition_rank in o

    def _is_phantom(node_tag: int) -> bool:
        return int(node_tag) in phantom_tags

    alone = frozenset((partition_rank,))

    def _copied(rec: object, nodes: "Iterable[int]") -> None:
        """Note ``rec`` (written here) when another rank writes it too:
        a non-phantom node of it is not held by this rank alone."""
        for n in nodes:
            t = int(n)
            if _is_phantom(t):
                continue
            o = owners.get(t)
            if o is None:
                o = node_owners.get(t, frozenset())
            if frozenset(o) != alone:
                replicated.append(rec)
                return

    def _add_foreign_or_phantom(node_tag: int) -> None:
        t = int(node_tag)
        if _owns(t):
            return
        if _is_phantom(t):
            referenced_phantoms.add(t)
        else:
            foreign_nodes.add(t)

    if node_recs:
        for rec in node_recs:
            if isinstance(rec, NodePairRecord):
                # equal_dof / rigid_beam / rigid_rod replicate on both
                # owning ranks (ADR 0027 §"Decision" — bullets 1, 2).
                m = int(rec.master_node)
                s = int(rec.slave_node)
                if pinned and s in pinned:
                    # ADR 0120 D1: a routed slave (a spring's side node)
                    # exists on its rank alone; the record goes there.
                    touches = pinned[s] == partition_rank
                else:
                    touches = _owns(m) or _owns(s)
                # Honor phantom slave on rigid_beam side: phantoms are
                # never "owned" by ranks per the broker (they're
                # broker-synthetic). A constraint whose master is on
                # this rank but slave is a phantom must still emit here
                # so that the phantom→slave equalDOF pair (emitted via
                # the NodeToSurface expansion below) sees the master.
                if not touches and (_is_phantom(m) or _is_phantom(s)):
                    # If neither master nor slave is owned by this rank
                    # and neither is a phantom referenced by another
                    # owned constraint, skip.  Pure phantom-phantom
                    # pairs are not expected from the broker.
                    pass
                if touches:
                    allowed_ids.add(id(rec))
                    _add_foreign_or_phantom(m)
                    _add_foreign_or_phantom(s)
                    if not (pinned and s in pinned):
                        _copied(rec, (m, s))
            elif isinstance(rec, NodeGroupRecord):
                # rigid_body / rigid_diaphragm / kinematic_coupling.
                # Per ADR 0027 §"Decision" bullet 3:
                #   "emit on every rank that owns any slave node".
                # The full command line is emitted verbatim on each
                # such rank; slaves are not sharded.
                #
                # kinematic_coupling emits a fork *element*
                # (LadrunoKinematicCoupling), not a replicable equalDOF
                # command — replicating it verbatim on N owning ranks
                # would allocate N distinct element tags ⇒ an N-fold
                # over-constraint. It routes through the SINGLE-
                # canonical-rank rule instead, mirroring the
                # ASDEmbeddedNodeElement ownership below: the one rank
                # where every slave node is present emits the element;
                # the reference node is ghost-declared there when
                # foreign (exactly like the embedded path's constrained
                # node). A slave set split across partitions fails loud.
                from apeGmsh._kernel.records._kinds import ConstraintKind
                if rec.kind == ConstraintKind.KINEMATIC_COUPLING:
                    slaves = [int(s) for s in rec.slave_nodes]
                    canonical = _canonical_coupling_rank(
                        rec, slaves, owners,
                    )
                    if partition_rank == canonical:
                        allowed_ids.add(id(rec))
                        _add_foreign_or_phantom(int(rec.master_node))
                        for s in slaves:
                            _add_foreign_or_phantom(s)
                    continue
                # A rigid_body emitted as the fork LadrunoRigidBody is also
                # a single *element* (allocates a tag) — replicating it on
                # every owning rank would mint N distinct elements ⇒ N-fold
                # over-constraint, exactly like kinematic_coupling above.
                # Route it through the same single-canonical-rank rule. The
                # whole body {master, *slaves} must co-locate (the element
                # binds them all — there is no ghost reference here), so the
                # canonical rank is computed over the FULL body node set; a
                # body split across ranks fails loud. The default
                # rigidLink-chain form (as_element=False) is a replicable MP
                # command and keeps the verbatim "every owning rank" rule.
                if (
                    rec.kind == ConstraintKind.RIGID_BODY
                    and getattr(rec, "as_element", False)
                ):
                    body = [
                        int(rec.master_node),
                        *(int(s) for s in rec.slave_nodes),
                    ]
                    canonical = _canonical_coupling_rank(
                        rec, body, owners,
                    )
                    if partition_rank == canonical:
                        allowed_ids.add(id(rec))
                        for b in body:
                            _add_foreign_or_phantom(b)
                    continue
                m = int(rec.master_node)
                slaves = [int(s) for s in rec.slave_nodes]
                touches = _owns(m) or any(_owns(s) for s in slaves)
                if touches:
                    allowed_ids.add(id(rec))
                    _add_foreign_or_phantom(m)
                    for s in slaves:
                        _add_foreign_or_phantom(s)
                    _copied(rec, (m, *slaves))
            elif isinstance(rec, NodeToSurfaceRecord):
                # NodeToSurface is a compound record. Its rigid_link
                # rows and equal_dof rows reference phantom nodes; we
                # treat the compound as a single bundle and replicate
                # it on every rank that owns the master OR any slave OR
                # any phantom.  Phantoms are "owned" by the rank that
                # owns their slave-side node (the rigid_beam / equalDOF
                # row chains phantom → slave).
                touches = False
                for pair in rec.rigid_link_records:
                    if _owns(int(pair.master_node)) or _owns(int(pair.slave_node)):
                        touches = True
                        break
                if not touches:
                    for pair in rec.equal_dof_records:
                        if (
                            _owns(int(pair.master_node))
                            or _owns(int(pair.slave_node))
                        ):
                            touches = True
                            break
                if touches:
                    allowed_ids.add(id(rec))
                    for pair in rec.rigid_link_records:
                        _add_foreign_or_phantom(int(pair.master_node))
                        _add_foreign_or_phantom(int(pair.slave_node))
                    for pair in rec.equal_dof_records:
                        _add_foreign_or_phantom(int(pair.master_node))
                        _add_foreign_or_phantom(int(pair.slave_node))
                    _copied(rec, (
                        n for pair in (*rec.rigid_link_records,
                                       *rec.equal_dof_records)
                        for n in (pair.master_node, pair.slave_node)))

    if surface_constraints is not None:
        # ASDEmbeddedNodeElement ownership (ADR 0027 §"ASDEmbeddedNode-
        # Element ownership"): emit on the SINGLE rank that can locally
        # assemble the host element — i.e. the rank that owns every
        # one of its master nodes.  Computed as
        # ``min(intersection(node_owners[m] for m in masters))``: the
        # intersection picks ranks where every corner node is present
        # (local or boundary-shared), and ``min`` deterministically
        # selects one canonical owner so every rank's planner agrees on
        # which rank emits.  Empty intersection means the corner nodes
        # split across partitions and the element cannot be assembled
        # on any single rank — fail-loud, naming the offending record.
        if interp_recs:
            for rec in interp_recs:
                # ADR 0068 Open item 2: the equation route is a domain-level
                # EQ_Constraint, NOT an element. The single-canonical-host-
                # rank element rule below is wrong for it — it would drop the
                # constraint on slave-owning ranks and falsely error on a
                # master face cut by a partition. Replicate it on EVERY rank
                # that owns the slave OR any master (the rigidDiaphragm rule):
                # each owning rank ghost-declares the foreign slave/masters
                # and emits identical equationConstraint rows. The active
                # cross-rank EQ-capable handler (LadrunoProjection / Lagrange)
                # resolves the constraint graph across subdomains; an
                # equation is a logical kinematic relation, so replicating it
                # verbatim does not over-constrain (unlike an element).
                if getattr(rec, "enforce", "penalty") == "equation":
                    masters_eq = [int(mn) for mn in rec.master_nodes]
                    slave_eq = int(rec.slave_node)
                    if _owns(slave_eq) or any(_owns(mn) for mn in masters_eq):
                        equation.append(rec)
                        _add_foreign_or_phantom(slave_eq)
                        for mn in masters_eq:
                            _add_foreign_or_phantom(mn)
                    continue
                masters = [int(mn) for mn in rec.master_nodes]
                if not masters:
                    continue
                canonical_rank = _canonical_host_rank(rec, masters, owners)
                if partition_rank == canonical_rank:
                    embedded.append(rec)
                    # Declare any foreign master/slave nodes.
                    for mn in masters:
                        _add_foreign_or_phantom(mn)
                    _add_foreign_or_phantom(int(rec.slave_node))

    return _RankConstraintPlan(
        allowed_record_ids=frozenset(allowed_ids),
        foreign_node_tags=frozenset(foreign_nodes),
        referenced_phantoms=frozenset(referenced_phantoms),
        embedded_records=tuple(embedded),
        equation_records=tuple(equation),
        replicated_records=tuple(replicated),
    )


def _canonical_coupling_rank(
    rec: object,
    slaves: list[int],
    node_owners: "Mapping[int, frozenset[int]]",
) -> int:
    """Return the single rank that emits a kinematic coupling's
    ``LadrunoKinematicCoupling`` element (RBE2).

    Mirrors :func:`_canonical_host_rank`: the canonical rank is
    ``min(intersection(node_owners[s] for s in slaves))`` — every tied
    slave must be present (locally owned or boundary-shared) on the
    chosen rank, and ``min`` picks one deterministic rank when several
    qualify, so every rank's planner agrees on the single emitter.  The
    reference node is NOT required to co-locate: the emit pass declares
    it as a foreign/ghost node when it lives on another rank (the same
    mechanism as the embedded path's constrained node).

    Raises
    ------
    ValueError
        When the intersection is empty — the slave set is split across
        partitions and no single rank can assemble the one coupling
        element.  Same partitioner-input-bug stance as
        :func:`_canonical_host_rank`.
    """
    intersection: set[int] | None = None
    for s in slaves:
        owners = node_owners.get(int(s), frozenset())
        intersection = (
            set(owners) if intersection is None
            else intersection & owners
        )
    if not intersection:
        ref = getattr(rec, "master_node", "?")
        name = getattr(rec, "name", None) or "<unnamed>"
        kind = getattr(rec, "kind", "coupling")
        elem = (
            "LadrunoRigidBody" if kind == "rigid_body"
            else "LadrunoKinematicCoupling"
        )
        raise ValueError(
            f"{kind} {name!r} (ref={ref}, nodes={slaves}) "
            f"has no rank where every node is present — the node "
            f"set is split across partitions, so the single "
            f"{elem} element cannot be assembled on "
            f"one rank. Repartition so the coupled node set stays on "
            f"one rank (for kinematic_coupling the reference node may live "
            f"anywhere — it is ghost-declared; a rigid_body binds its whole "
            f"set), or emit unpartitioned. To repair, "
            f"re-partition from the mesh phase with "
            f"g.mesh.partitioning.partition_explicit(...) placing every "
            f"node's incident elements on one rank (see "
            f"guide_partitioning.md §7.1). Per-node owners: "
            f"{ {s: sorted(node_owners.get(int(s), set())) for s in slaves} }"
        )
    return min(intersection)


def _canonical_host_rank(
    rec: object,
    masters: list[int],
    node_owners: "Mapping[int, frozenset[int]]",
) -> int:
    """Return the single rank that emits the embeddedNode line for ``rec``.

    The canonical host rank is ``min(intersection(node_owners[m] for m
    in masters))``: every master must be present (locally owned or
    boundary-shared) on the chosen rank so OpenSees can assemble the
    element from local + foreign-declared nodes, and ``min`` picks one
    deterministic rank when several qualify.

    Raises
    ------
    ValueError
        When the intersection is empty — the host element's corner
        nodes are split across partitions and no single rank can
        assemble it.  ADR 0027 §"ASDEmbeddedNodeElement ownership"
        treats this as a partitioner-input bug, not a recoverable
        condition.
    """
    intersection: set[int] | None = None
    for m in masters:
        owners = node_owners.get(int(m), frozenset())
        intersection = (
            set(owners) if intersection is None
            else intersection & owners
        )
    if not intersection:
        slave = getattr(rec, "slave_node", "?")
        name = getattr(rec, "name", None) or "<unnamed>"
        raise ValueError(
            f"ASDEmbeddedNodeElement {name!r} (slave={slave}, "
            f"masters={masters}) has no rank that owns every master "
            f"node — host element corners are split across partitions. "
            f"To repair, re-partition from the mesh phase with "
            f"g.mesh.partitioning.partition_explicit(...) placing every "
            f"master node's incident elements on one rank (see "
            f"guide_partitioning.md §7.1). Per-master owners: "
            f"{ {m: sorted(node_owners.get(int(m), set())) for m in masters} }"
        )
    return min(intersection)


def _emit_surface_couplings_for_rank(
    emitter: "Emitter",
    records: tuple[object, ...],
    tag_plan: "TagPlan",
    *,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
) -> None:
    """Emit ASDEmbeddedNodeElement lines for the host-rank surface couplings.

    Element tags come from one ``"element"`` counter shared across all
    ranks of the same emit pass — so a record landing on rank K and a
    different record
    landing on rank K+1 receive distinct globally-unique tags (ADR 0027
    §"Tag determinism").  The previous static ``1_000_000`` base
    collided across ranks because each rank restarted the counter
    independently. The tags come from :func:`plan_mp_elements`
    (:func:`_mp_element_tagger`).
    """
    from apeGmsh._kernel.records._constraints import InterpolationRecord

    tag_of = _mp_element_tagger(tag_plan)
    for rec in records:
        if not isinstance(rec, InterpolationRecord):
            continue
        _write_one_interpolation(
            emitter, rec, tag_of, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            stiffness_resolver=stiffness_resolver,
        )
