"""constraints: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping

import numpy as np

from ..types import Element
from ..tag_resolution import (
    MISSING_FEM_ELEMENT_ID,
    set_current_fem_element_id,
    set_element_nodes,
)
from ..tag_allocator import TagAllocator
from .errors import BridgeError
from .records import EquationConstraintRecord, StiffnessResolver, _emit_name
from .pg_expand import expand_pg_to_nodes
from .ndf import _perp_dirn_from_normal, node_coords_as_floats
from .ownership import FemToOpsTagMap
from .constraints_plan import (
    MPElementEntry,
    _INTERPOLATION_VERBS,
    _interpolation_route,
    _kinematic_coupling_records,
    _mp_element_tagger,
    _mp_plan,
    _rigid_body_element_records,
    embed_tie_records,
    interpolation_records,
    mp_constraint_pools,
    mp_element_entry,
    plan_mp_elements,
    rebar_element_records,
    reinforce_tie_records,
)


if TYPE_CHECKING:
    from apeGmsh._kernel.records._constraints import (
        ConstraintRecord,
        InterpolationRecord,
    )
    from apeGmsh._kernel._coupling_control import CouplingControl
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from ..tag_plan import TagPlan


def emit_equation_constraints(
    emitter: "Emitter",
    fem: "FEMData",
    records: "Iterable[EquationConstraintRecord]",
    *,
    node_ndf: "dict[int, int]",
    default_ndf: int,
) -> None:
    """Emit the user ``equationConstraint`` rows, in declaration order.

    Every node must exist in the snapshot and every DOF must fit that
    node's ndf; OpenSees would otherwise fail late in
    ``EQ_Constraint::setDomain`` or not at all. Runs in the MP-constraint
    pass, after the broker's constraints.
    """
    records = tuple(records)
    if not records:
        return
    known = {int(n) for n in np.asarray(fem.nodes.ids).tolist()}
    for rec in records:
        for node, dof, what in (
            (rec.cnode, rec.cdof, "constrained"),
            *((rn, rd, "retained") for rn, rd, _ in rec.retained),
        ):
            if node not in known:
                raise BridgeError(
                    f"equation_constraint: {what} node {node} is not a node "
                    f"of the FEM snapshot."
                )
            ndf = int(node_ndf.get(node, default_ndf))
            if dof > ndf:
                raise BridgeError(
                    f"equation_constraint: {what} dof {dof} on node {node} "
                    f"exceeds that node's ndf ({ndf})."
                )
        emitter.equationConstraint(
            rec.cnode, rec.cdof, rec.ccoef, list(rec.retained),
        )


# ---------------------------------------------------------------------------
# MP constraint fan-out (Phase 7b, ADR 0022) — closes the §3.3 deferral.
# ---------------------------------------------------------------------------


class PartlyFixedTieWarning(UserWarning):
    """An ``equalDOF`` whose retained node is fixed on some tied DOFs and
    free on others, in a case the bridge cannot rewrite (a stage adds or
    releases that fix). OpenSees' Transformation handler then drops the
    tie on the free DOF at every step (#2396); Lagrange and Penalty are
    correct."""


@dataclass(frozen=True)
class FixedDofs:
    """The homogeneous fixity the global ``fix`` pass emits, for the
    ``equalDOF`` rewrite (#2396): ``fixed[node]`` holds the 1-based DOFs
    fixed from the first step on, ``staged`` every other ``(node, dof)``
    an SP ever acts on (a stage ``s.fix`` / ``s.support`` / ``s.remove_sp``,
    a pattern ``p.sp``), which a static rewrite cannot reproduce, and
    ``ndf`` each node's DOF count."""

    fixed: "Mapping[int, frozenset[int]]"
    staged: "frozenset[tuple[int, int]]"
    ndf: "Mapping[int, int]"
    default_ndf: int
    #: The fix record that fixed each node first, and the bridge's
    #: declaration cursor: a rewritten slave fix is archived under the
    #: master fix's declaration.
    owner: "Mapping[int, object]"
    decl_cursor: "Callable[[Emitter], Any] | None" = None
    #: Nodes that retain, and nodes that are constrained by, some tie.
    retained: "frozenset[int]" = frozenset()
    constrained: "frozenset[int]" = frozenset()
    ndm: int = 3
    #: Whether some analysis runs under Transformation (declared, or
    #: auto-emitted for MP constraints), and whether the deck declares no
    #: Analysis (a modal deck then forces Transformation, ADR 0077 INV-10).
    transformation: bool = False
    no_analysis: bool = False
    #: False for a stage-claimed tie: a ``fix`` added after an analysis
    #: is total, an ``equalDOF`` incremental, so a slave that has moved
    #: would snap back. Such a tie keeps its form and warns.
    rewrite: bool = True

    def in_stage(self) -> "FixedDofs":
        """This map for a stage-claimed tie (``rewrite=False``)."""
        return replace(self, rewrite=False)


def fixed_dof_map(built: Any, inferred_ndf: "Mapping[int, int] | None") -> FixedDofs:
    """The :class:`FixedDofs` of a built model (its global ``fix`` records,
    ``fix_from_model`` included, and every stage's ``s.fix`` /
    ``s.remove_sp``)."""
    from ...analysis.constraint_handler import Transformation
    from ..types import Analysis, ConstraintHandler
    from .ndf import fit_fix_mask

    # A read-only view, not ``dict(...)``: the emit's per-node ndf map is
    # never mutated after inference, and a copy held in ``FixedDofs.ndf``
    # stays resident into the rank blocks at a dict entry per global node
    # (48.5 B/gn; #2527, the emit-profile smoke's residency guard).
    eff = MappingProxyType(inferred_ndf or {})
    default = int(built.ndf)
    fixed: dict[int, set[int]] = {}
    owner: dict[int, object] = {}
    for rec in built.fix_records:
        for node in built._resolve_node_target(rec.pg, rec.nodes):
            n = int(node)
            owner.setdefault(n, rec)
            mask = fit_fix_mask(rec.dofs, int(eff.get(n, default)))
            fixed.setdefault(n, set()).update(
                i + 1 for i, m in enumerate(mask) if int(m))
    staged: set[tuple[int, int]] = set()

    def _pattern_sps(patterns: "Iterable[Any]") -> None:
        for pat in patterns:
            for sp in getattr(pat, "sps", ()) or ():
                nodes = ([int(sp.target)] if sp.target_kind == "node"
                         else built._resolve_node_target(sp.target, None))
                staged.update((int(n), int(sp.dof)) for n in nodes)

    for stage in getattr(built, "stage_records", ()) or ():
        for rec in (*stage.fix_records, *stage.support_records):
            for node in built._resolve_node_target(rec.pg, rec.nodes):
                staged.update((int(node), i + 1)
                              for i, m in enumerate(rec.dofs) if int(m))
        for rem in stage.remove_sp_records:
            for node in built._resolve_node_target(rem.pg, rem.nodes):
                staged.update((int(node), int(d)) for d in rem.dofs)
        _pattern_sps(stage.pattern_specs)
    _pattern_sps(p for p in built.primitives if hasattr(p, "sps"))
    retained: set[int] = set()
    constrained: set[int] = set()
    for rec in getattr(built.fem.nodes, "constraints", None) or ():
        kind = getattr(rec, "kind", None)
        if kind in ("equal_dof", "equal_dof_mixed", "rigid_beam",
                    "rigid_rod", "rigid_body", "rigid_diaphragm"):
            retained.add(int(rec.master_node))
            slaves = (rec.slave_nodes if hasattr(rec, "slave_nodes")
                      else [rec.slave_node])
            constrained.update(int(n) for n in slaves)
    stages = getattr(built, "stage_records", ()) or ()
    prims = list(getattr(built, "primitives", ()) or ())
    declared = next((p for p in prims
                     if isinstance(p, ConstraintHandler)), None)
    global_t = isinstance(declared, Transformation) or (
        declared is None and not built._has_equation_constraints())
    # a stage that declares no handler runs under the global one, which
    # the flat staged emit auto-emits as Transformation for MP constraints
    transformation = (any(isinstance(st.constraints, Transformation)
                          or (st.constraints is None and global_t)
                          for st in stages)
                      if stages else global_t)
    return FixedDofs(
        fixed={n: frozenset(d) for n, d in fixed.items()},
        staged=frozenset(staged), ndf=eff, default_ndf=default,
        owner=owner, decl_cursor=getattr(built, "_decl_cursor", None),
        retained=frozenset(retained), constrained=frozenset(constrained),
        ndm=int(getattr(built, "ndm", 3) or 3),
        transformation=transformation,
        no_analysis=not stages and not any(isinstance(p, Analysis)
                                           for p in prims))


def _split_partly_fixed_tie(
    master: int, slave: int, dofs: "tuple[int, ...]",
    fixed: "FixedDofs | None", name: "str | None",
) -> "tuple[tuple[int, ...], tuple[int, ...]]":
    """``(dofs to tie, dofs to fix on the slave)`` for one ``equalDOF``
    (:func:`_split_partly_fixed_pairs` with each DOF tied to itself)."""
    keep, to_fix = _split_partly_fixed_pairs(
        master, slave, tuple((d, d) for d in dofs), fixed, name)
    return tuple(c for _r, c in keep), to_fix


def _split_partly_fixed_pairs(
    master: int, slave: int, pairs: "tuple[tuple[int, int], ...]",
    fixed: "FixedDofs | None", name: "str | None",
) -> "tuple[tuple[tuple[int, int], ...], tuple[int, ...]]":
    """``(pairs to tie, slave DOFs to fix)`` for one tie of
    ``(retained DOF, constrained DOF)`` pairs (``equalDOF`` /
    ``equalDOF_Mixed``).

    A retained node fixed on some tied DOFs and free on others breaks the
    Transformation handler (``TransformationDOF_Group::enforceSPs`` resets
    the constrained node's free DOF every step, #2396). Tying only the
    pairs whose retained DOF is free, and fixing the slave on the others,
    is the same constraint set under every handler. Left as declared when
    nothing is partly fixed; when a stage or a pattern changes the fix, or
    the tie is part of a chain, warned instead.
    """
    if fixed is None:
        return pairs, ()
    held = fixed.fixed.get(master, frozenset())
    on = tuple(p for p in pairs if p[0] in held)
    tied = {r for r, _c in pairs}
    varying = {r for r in tied if (master, r) in fixed.staged}
    if varying and (varying != tied or on):
        # some retained DOFs change while others hold or stay free
        _warn_partly_fixed(master, slave, name,
                           why="a stage or a pattern SP acts on")
        return pairs, ()
    if not on or len(on) == len(pairs):
        return pairs, ()           # free or fully held: Transformation is exact
    if not fixed.rewrite:
        _warn_partly_fixed(master, slave, name,
                           why="a stage claims the tie (a fix added "
                           "mid-analysis is total, not incremental) on")
        return pairs, ()
    if any((slave, c) in fixed.staged for _r, c in on):
        _warn_partly_fixed(master, slave, name,
                           why="a stage fixes the slave on")
        return pairs, ()
    if slave in fixed.retained or master in fixed.constrained:
        _warn_partly_fixed(master, slave, name,
                           why="the tie is part of a chain of ties through")
        return pairs, ()
    free = tuple(p for p in pairs if p[0] not in held)
    already = fixed.fixed.get(slave, frozenset())
    return free, tuple(c for _r, c in on if c not in already)


def _warn_partly_fixed_rigid_link(
    kind: str, master: int, slave: int, name: "str | None",
    fixed: "FixedDofs | None",
) -> None:
    """Warn when a ``rigidLink``'s retained node is fixed on some of the
    DOFs the link maps and free on others: the same Transformation defect
    as #2396, which a rigid link cannot be split around (its slave DOFs
    depend on the master's rotations). Lagrange and Penalty are correct."""
    if fixed is None:
        return
    ndf = int(fixed.ndf.get(master, fixed.default_ndf))
    mapped = (set(range(1, ndf + 1)) if kind == "beam"
              else set(range(1, min(fixed.ndm, ndf) + 1)))
    held = set(fixed.fixed.get(master, frozenset())) & mapped
    varying = {d for d in mapped if (master, d) in fixed.staged}
    if (held and held != mapped) or (varying and (varying != mapped or held)):
        warnings.warn(
            f"rigidLink {kind} {master} -> {slave}"
            + (f" ({name})" if name else "")
            + ": its retained node is fixed on only some of the DOFs the "
            "link maps. OpenSees' Transformation handler (the bridge "
            "default with MP constraints) then drops the tie on the free "
            "DOFs at every step (#2396, #2406). Use "
            "ops.constraints.Lagrange() or Penalty(...).",
            PartlyFixedTieWarning, stacklevel=4,
        )


def _warn_partly_fixed(master: int, slave: int, name: "str | None", *,
                       why: str) -> None:
    warnings.warn(
        f"equalDOF {master} -> {slave}"
        + (f" ({name})" if name else "")
        + f": its retained node is (or becomes) fixed on only some tied "
        f"DOFs, and {why} a tied DOF, so the tie cannot be rewritten for "
        "the Transformation handler, which then drops the tie on the free "
        "DOF at every step (#2396). Use ops.constraints.Lagrange() or "
        "Penalty(...).",
        PartlyFixedTieWarning, stacklevel=4,
    )


def emit_mp_constraints(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan",
    *, claimed_ids: "frozenset[int]" = frozenset(),
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
    fixed_dofs: "FixedDofs | None" = None,
) -> None:
    """Fan out the broker's MP-constraint records onto ``emitter``.

    Per ADR 0022 INV-5 this pass runs between element emission and
    pattern emission in :meth:`BuiltModel.emit`.

    Ordering (INV-3 / dependency-driven):

    1. **Phantom-node pre-step** — :class:`NodeToSurfaceRecord` rows
       carry synthetic 6-DOF phantom nodes whose tags must exist in
       the OpenSees domain before any constraint references them.
       Emitted via ``emitter.node(tag, *xyz, ndf=6)`` — the standard
       OpenSees per-node ``-ndf`` override pattern.  Tags are
       de-duplicated across records (paranoid; the resolver does not
       collide, but the cost of the set check is negligible).
       ``ndm`` sizes the coordinate list — its default of 3 matches
       :func:`emit_transform_specs`, so a caller that omits it gets
       the historical 3-D line rather than a silently different deck.

    2. **Rigid links** — :meth:`fem.nodes.constraints.rigid_link_groups`
       yields ``(master, slaves)`` tuples covering ``rigid_beam`` /
       ``rigid_rod`` :class:`NodePairRecord` rows, the ``rigid_body``
       slaves on :class:`NodeGroupRecord`, and the phantom-side
       rigid-link rows on :class:`NodeToSurfaceRecord`.  Emitted as
       one ``emitter.rigidLink('beam'|'bar', master, slave)`` per
       slave; ``rigid_rod`` maps to ``"bar"`` per the OpenSees
       vocabulary.

    3. **Equal DOFs** — :meth:`fem.nodes.constraints.equal_dofs`
       yields :class:`NodePairRecord` rows for ``equal_dof`` plus the
       phantom→slave equal-DOF rows nested under
       :class:`NodeToSurfaceRecord`.  Emitted as one
       ``emitter.equalDOF(master, slave, *dofs)`` per record.

    4. **Rigid diaphragms** — :meth:`fem.nodes.constraints.rigid_diaphragms`
       yields ``(perp_dir, master, slaves)`` for
       :class:`NodeGroupRecord` rows with kind ``rigid_diaphragm``.
       Emitted as one ``emitter.rigidDiaphragm(perp_dir, master,
       *slaves)`` per record.

    5. **Kinematic couplings** — :class:`NodeGroupRecord` rows with
       kind ``kinematic_coupling`` are emitted as one fork element
       ``element LadrunoKinematicCoupling`` (RBE2) per record, under its
       planned element tag, by :func:`_emit_kinematic_couplings`. It
       replaced the earlier ``equalDOF``-per-slave expansion.

    6. **Surface couplings** — :meth:`fem.elements.constraints.interpolations`
       yields :class:`InterpolationRecord` rows (one slave node ↔ N
       weighted master nodes from a master element face).  Emitted as
       one ``emitter.embeddedNode(ele_tag, cnode, *args)``
       per record under its planned element tag.  Covers
       ``tie`` / ``distributing`` / ``embedded`` directly and
       ``tied_contact`` / ``mortar`` via the
       :meth:`SurfaceCouplingRecord.slave_records` expansion that
       ``interpolations()`` performs internally.

    Each constraint with a non-empty ``name`` is preceded by
    ``emitter.mp_constraint_comment(name)`` so the user's declaration
    label round-trips into emitted Tcl / Py via the ``# {name}`` line
    (INV-2).

    ``fem_eid_to_ops_tag`` is the bridge-built ``{fem_eid: ops_tag}``
    map — a :class:`CouplingControl` carrying a ``host`` element (the
    ``-k auto`` / ``-wcap`` scalers) stores it as a **FEM eid** and the
    coupling emitters translate it to the emitted OpenSees element tag
    here. Hosted controls fail loud without the map.

    No-op when the FEM snapshot exposes no ``nodes.constraints`` or
    ``elements.constraints`` accessors — broker constraints are
    purely additive on top of any other bridge state.

    The element tags come from :func:`plan_mp_elements` (ADR 0114 D4,
    amended); :func:`_mp_plan` says how this reads them, and a plan made
    for other stage claims than ``claimed_ids`` raises.
    """
    from ..tag_resolution import set_phantom_node_tags

    _mp_plan(tag_plan, fem).check_claims(claimed_ids)
    node_constraints, surface_constraints = mp_constraint_pools(
        fem, claimed_ids)

    # -------------------------------------------------------------------
    # 0. Pre-load the phantom-tag predicate on the emitter so the H5
    #    emitter can classify subsequent ``node()`` calls (ADR 0033 —
    #    stateless replacement for the prior phantom-mode flag).  The
    #    set is computed ONCE from ``NodeToSurfaceRecord.phantom_nodes``
    #    and never mutated; phantom tags are disjoint from real broker
    #    tags so this is safe to install before any emission.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        phantom_tags = set(
            _gather_phantom_nodes(node_constraints).keys()
        )
        set_phantom_node_tags(emitter, phantom_tags)

    # -------------------------------------------------------------------
    # 1. Phantom-node pre-step — emit synthesized phantom nodes BEFORE
    #    any constraint references them.  ADR 0022 INV-3.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_phantom_nodes(emitter, node_constraints)

    # -------------------------------------------------------------------
    # 2. Rigid links — ``emitter.rigidLink(kind, master, slave)`` per
    #    pair.  Walks NodePairRecord rows directly (so the kind / name
    #    survive) plus the rigid_body and node_to_surface compound
    #    expansions.  We don't use ``rigid_link_groups()`` because it
    #    drops the per-pair ``name`` field.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_rigid_links(emitter, node_constraints, fixed=fixed_dofs)

    # -------------------------------------------------------------------
    # 2b. Rigid bodies (as_element) — one fork ``element LadrunoRigidBody``
    #     per NodeGroupRecord(kind=rigid_body, as_element=True). Writes
    #     planned element tags, so it takes ``tag_plan`` (the rigidLink-chain
    #     form in step 2 skips these records).
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_rigid_body_elements(emitter, node_constraints, tag_plan)

    # -------------------------------------------------------------------
    # 3. Equal DOFs — direct NodePairRecord(kind=equal_dof) plus the
    #    NodeToSurfaceRecord.equal_dof_records expansion.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_equal_dofs(emitter, node_constraints, fixed=fixed_dofs)

    # -------------------------------------------------------------------
    # 4. Rigid diaphragms — one ``rigidDiaphragm`` per
    #    NodeGroupRecord(kind=RIGID_DIAPHRAGM).
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_rigid_diaphragms(emitter, node_constraints, fixed=fixed_dofs)

    # -------------------------------------------------------------------
    # 5. Kinematic couplings (RBE2) — one fork
    #    ``element LadrunoKinematicCoupling`` per NodeGroupRecord row.
    #    Carries the moment-arm transport an equalDOF expansion can't
    #    (offset reference). Writes planned element tags, so it takes
    #    ``tag_plan``.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_kinematic_couplings(
            emitter, node_constraints, tag_plan,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )

    # -------------------------------------------------------------------
    # 6. Surface couplings — InterpolationRecord (tie / distributing /
    #    embedded) plus the SurfaceCouplingRecord.slave_records
    #    expansion (tied_contact / mortar).  All go out as
    #    ASDEmbeddedNodeElement.
    # -------------------------------------------------------------------
    if surface_constraints is not None:
        _emit_surface_couplings(
            emitter, surface_constraints, tag_plan,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            stiffness_resolver=stiffness_resolver,
        )


def emit_reinforce_ties(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan",
    *, name_to_tag: "dict[str, int]",
    records: "Iterable[Any] | None" = None,
) -> None:
    """Emit one ``element LadrunoEmbeddedRebar`` per resolved reinforcement
    tie (``g.reinforce``, ADR 20 / R2b).

    Consumes ``fem.elements.reinforce_ties`` —
    :class:`~apeGmsh._kernel.records._constraints.ReinforceTieRecord` rows
    produced by :class:`ReinforcementsComposite` at FEM-build time. Each
    record carries the rebar node, the host node list + shape-function
    weights (the ``-shape`` host-element-tag-free path), the bar axis
    ``-dir``, and the axial law (``-perfect kAxial`` or ``-bond matName``
    + ``-bondScale``). The positional argument list is assembled by the
    R0 ``embedded_rebar_args`` grammar builder (the single source of truth
    for the flag order), and the element takes its planned tag on the
    ``"element"`` counter, so rebar couplings share the global
    element-tag namespace.

    Bond name → tag resolution (Option B): a ``bond`` record holds the
    **name** of a ``LadrunoBondSlip`` material declared separately on the
    bridge. ``name_to_tag`` (the bridge's resolved name-alias map) is
    consulted here; a missing name fails loud — a tie that references an
    unregistered bond material must not silently emit a dangling tag.

    No-op when the FEM snapshot exposes no ``elements.reinforce_ties`` —
    reinforcement is purely additive on top of any other bridge state.

    ``records`` restricts the pass to a subset (the partitioned emit hands
    each rank the ties it owns).

    The element tags come from :func:`plan_mp_elements` (ADR 0114 D4,
    amended), read from ``tag_plan``.
    """
    ties = (
        list(records) if records is not None else reinforce_tie_records(fem))
    if not ties:
        return
    _mp_plan(tag_plan, fem)        # refuses another model's plan
    _write_reinforce_ties(
        emitter, ties, _mp_element_tagger(tag_plan), name_to_tag)


def replay_reinforce_ties(
    emitter: "Emitter", fem: "FEMData", tags: TagAllocator,
    *, name_to_tag: "dict[str, int]",
) -> None:
    """Write ``fem``'s reinforce ties, numbering them from ``tags``.

    The one standalone writer of the MP-element family, for the deck
    replay of an archive below opensees 2.29.0 alone (the
    ``pre-2-29-reinforce-tie-replay`` waiver of the tag-law ledger): such
    an archive stores no tie record, so the replay plans the ties from its
    own allocator through :func:`plan_mp_elements`, the loop the tag plan
    runs. A 2.29.0 archive stores every tie, and its replay never calls
    this (the version gate in ``compose._replay_into``). Every other
    writer reads a :class:`TagPlan`.
    """
    ties = reinforce_tie_records(fem)
    if not ties:
        return

    def standalone(entry: MPElementEntry) -> int:
        (line,) = plan_mp_elements((entry,), tags)
        return line.tag

    _write_reinforce_ties(emitter, ties, standalone, name_to_tag)


def _write_reinforce_ties(
    emitter: "Emitter",
    ties: "list[Any]",
    tag_of: "Callable[[MPElementEntry], int]",
    name_to_tag: "dict[str, int]",
) -> None:
    """Write each reinforce tie under the tag ``tag_of`` gives it."""
    from ...element.embedded_rebar import embedded_rebar_args

    for rec in ties:
        _emit_name(emitter, rec.name)

        if rec.bond is not None:
            bond_tag = name_to_tag.get(rec.bond)
            if bond_tag is None:
                known = ", ".join(sorted(name_to_tag)) or "<none>"
                raise ValueError(
                    f"reinforce: tie at rebar node {rec.rebar_node} "
                    f"references bond material {rec.bond!r}, but no "
                    f"primitive with that name is registered on the "
                    f"bridge. Declare it (e.g. "
                    f"ops.uniaxialMaterial.LadrunoBondSlip(..., "
                    f"name={rec.bond!r})). Known names: {known}."
                )
        else:
            bond_tag = None

        args = embedded_rebar_args(
            rebar_node=int(rec.rebar_node),
            direction=[float(d) for d in rec.direction],
            host_nodes=[int(h) for h in rec.host_nodes],
            shape=[float(w) for w in rec.weights],
            perfect=rec.perfect,
            bond=bond_tag,
            bond_scale=rec.bond_scale,
            kt=rec.kt,
            kt_alpha=rec.kt_alpha,
            corot=rec.corot,
            shape_b=([float(w) for w in rec.shape_b]
                     if rec.shape_b is not None else None),
            enforce=rec.enforce,
            bipenalty=rec.bipenalty,
            dtcr=rec.dtcr,
        )
        ele_tag = tag_of(mp_element_entry("reinforce_tie", rec))
        emitter.embedded_rebar(ele_tag, *args)


def emit_embed_ties(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan",
) -> None:
    """Emit one ``element LadrunoEmbeddedNode`` per resolved embedment tie
    (``g.embed``).

    Consumes ``fem.elements.embed_ties`` —
    :class:`~apeGmsh._kernel.records._constraints.EmbedTieRecord` rows
    produced by :class:`EmbedmentsComposite` at FEM-build time. Each record
    carries the constrained node, the host node list + shape-function
    weights (the ``-shape`` host-element-tag-free path), and the isotropic
    tie parameters. The positional argument list is assembled by the
    ``embedded_node_args`` grammar builder, and the element takes its planned
    tag on the ``"element"`` counter, so embedment couplings share the
    global element-tag namespace.

    No-op when the FEM snapshot exposes no ``elements.embed_ties`` — embedment
    is purely additive on top of any other bridge state. The element tags
    come from :func:`plan_mp_elements`, as in :func:`emit_reinforce_ties`.
    """
    from ...element.embedded_node import embedded_node_args

    ties = embed_tie_records(fem)
    if not ties:
        return
    _mp_plan(tag_plan, fem)        # refuses another model's plan
    tag_of = _mp_element_tagger(tag_plan)

    for rec in ties:
        _emit_name(emitter, rec.name)
        args = embedded_node_args(
            cnode=int(rec.node),
            host_nodes=[int(h) for h in rec.host_nodes],
            shape=[float(w) for w in rec.weights],
            k=rec.k,
            k_alpha=rec.k_alpha,
            enforce=rec.enforce,
            bipenalty=rec.bipenalty,
            dtcr=rec.dtcr,
            staged=rec.staged,
        )
        ele_tag = tag_of(mp_element_entry("embed_tie", rec))
        emitter.embedded_node(ele_tag, *args)


def emit_rebar_elements(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan",
    *, name_to_tag: "dict[str, int]",
    records: "Iterable[Any] | None" = None,
) -> None:
    """Emit the cage's auto-emitted structural rebar elements (ADR 0067
    P5.2 / B1) — one ``CorotTruss`` per line cell of each bar PG.

    Consumes ``fem.elements.rebar_elements`` —
    :class:`~apeGmsh._kernel.records._rebar.RebarElementRecord` rows from
    ``g.rebar.place(emit_elements=True)``. Each record names the bar's
    uniaxial-material **name**, its area, and the bar's resolved line cells
    (``connectivity``, extracted from the live mesh at ``get_fem_data`` — the
    dim-1 cells are dropped from a dim-3 ``FEMData``); a ``CorotTruss`` is
    emitted per line cell. This is the bar's OWN axial element —
    distinct from the ``LadrunoEmbeddedRebar`` coupling (which carries no
    axial stiffness). Each cell takes its planned tag on the ``"element"``
    counter (shared element-tag namespace, like
    :func:`emit_reinforce_ties`).

    Material name → tag resolution (Option B): ``name_to_tag`` (the bridge's
    resolved name-alias map) is consulted; a missing name fails loud — an
    auto-emitted bar that references an unregistered material must not
    silently emit a dangling tag.

    ``element="beam"`` bars raise :class:`NotImplementedError` — beam
    auto-emit (fiber section + ``beamIntegration`` + per-segment
    ``geomTransf`` + ``ndf=6`` + twist) is B1b, not yet wired.

    No-op when the FEM snapshot exposes no ``elements.rebar_elements``.

    ``records`` restricts the pass to a subset (the partitioned emit hands
    each rank records holding only the bar cells it owns).

    The element tags come from :func:`plan_mp_elements`, one per cell,
    keyed by ``(pg, i_node, j_node)`` so a rank's copy of a bar record
    reads the tags planned for its cells (:func:`_mp_element_tagger`).
    """
    recs = (
        list(records) if records is not None else rebar_element_records(fem))
    if not recs:
        return
    _mp_plan(tag_plan, fem)        # refuses another model's plan
    tag_of = _mp_element_tagger(tag_plan)

    for rec in recs:
        if rec.element != "truss":
            raise NotImplementedError(
                f"g.rebar.place(emit_elements=True): auto-emit of "
                f"element={rec.element!r} bars (PG {rec.pg!r}) is not yet "
                f"wired — beam-element (dowel) rebar is B1b (fiber section + "
                f"beamIntegration + per-segment geomTransf + ndf=6 + twist). "
                f"Use element='truss', or hand-emit the beam element."
            )
        mat_tag = name_to_tag.get(rec.material)
        if mat_tag is None:
            known = ", ".join(sorted(name_to_tag)) or "<none>"
            raise ValueError(
                f"g.rebar.place(emit_elements=True): bar PG {rec.pg!r} "
                f"references material {rec.material!r}, but no primitive with "
                f"that name is registered on the bridge. Declare it (e.g. "
                f"ops.uniaxialMaterial.Steel02(..., name={rec.material!r})). "
                f"Known names: {known}."
            )
        for i_node, j_node in rec.connectivity:
            ele_tag = tag_of(mp_element_entry(
                "rebar_cell", rec, (int(i_node), int(j_node))))
            # Minted bar cell — the line cell has a gmsh id, but a dim-3
            # extraction drops it from /elements, so there is no row to
            # join on; args start with the node pair: ADR 0049 node-pair
            # convention (sentinel fem_eid + true endpoints), same
            # rationale as the interface zeroLength above (ADR 0093 S10
            # finding).  Readers join by node pair (h5-schema.md
            # "Bridge-minted rows", #1290).
            set_current_fem_element_id(emitter, MISSING_FEM_ELEMENT_ID)
            set_element_nodes(emitter, (int(i_node), int(j_node)))
            emitter.element(
                "CorotTruss", ele_tag, int(i_node), int(j_node),
                float(rec.area), int(mat_tag),
            )


# ---------------------------------------------------------------------------
# emit_mp_constraints sub-helpers (split for readability + per-kind unit tests)
# ---------------------------------------------------------------------------


def _emit_phantom_nodes(
    emitter: "Emitter", node_constraints: Iterable[object],
) -> None:
    """Emit ``node(tag, *xyz, ndf=6)`` for every phantom node.

    Phantoms only exist on :class:`NodeToSurfaceRecord` rows — the
    resolver synthesizes them at resolve time and stores their tags +
    coords on the record without writing them into ``fem.nodes`` (see
    pre-flight audit in the Phase 7b spec).  De-duplicates tags across
    records — paranoid-cheap; the resolver maintains a single counter,
    but the set check is one line.

    The H5 emitter's phantom-vs-real-broker discriminator is the
    stateless :func:`is_phantom_node` predicate, pre-loaded on the
    emitter by :func:`emit_mp_constraints` before this helper runs
    (S2 / ADR 0033 — real broker nodes can also pass ``ndf=K`` now,
    so the explicit predicate replaces the old "``ndf is not None``"
    heuristic).  No flag flipping needed here.

    **2-D.** The phantom takes ``ndm`` coordinates like every other
    node line — one rule, no exception (:func:`node_coords_as_floats`).
    The consequence is deliberate and worth naming: under ``ndm == 2``
    the ``-ndf 6`` now actually *lands* instead of being swallowed by
    the parser's optional-argument scan, so a 2-D ``node_to_surface``
    stops silently giving its phantom the model envelope and starts
    failing at ``RigidBeam``'s dof-mismatch check.  That construct —
    a 6-DOF master rigid-linked to a surface, ``equalDOF``-ing three
    *translations* — never had a 2-D reading; it only looked like it
    did.  Refusing it by model dimension is a resolve-time gate and
    belongs with the other dimension gates, not in the node emitter,
    which owes the deck exactly what the records say.
    """
    n2s_iter = getattr(node_constraints, "node_to_surfaces", None)
    if n2s_iter is None:
        return
    seen: set[int] = set()
    for rec in n2s_iter():
        coords = rec.phantom_coords
        if coords is None:
            continue
        for tag, xyz in zip(rec.phantom_nodes, coords):
            t = int(tag)
            if t in seen:
                continue
            seen.add(t)
            # Per-node ``-ndf 6`` override — phantoms are 6-DOF even
            # when the surrounding slaves are 3-DOF (standard OpenSees
            # idiom for mixed-ndf models).  ``ndm`` coordinates only,
            # like every other node line: see the 2-D note on
            # :func:`_emit_phantom_nodes`.
            emitter.node(t, *node_coords_as_floats(xyz), ndf=6)


def _emit_rigid_links(
    emitter: "Emitter", node_constraints: Iterable[object],
    *, allowed_ids: frozenset[int] | None = None,
    fixed: "FixedDofs | None" = None,
) -> None:
    """Emit ``rigidLink`` per :class:`NodePairRecord` (rigid_beam /
    rigid_rod) plus the rigid_body and node_to_surface compound
    expansions.  Preserves the per-record ``name`` for INV-2.

    When ``allowed_ids`` is given, only records whose ``id(rec)`` is in
    the set emit — the partitioned paths pass each rank's claimed
    subset (see :class:`_RankConstraintPlan`).
    """
    from apeGmsh._kernel.records._constraints import (
        NodeGroupRecord, NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    rigid_pair_kinds = {
        ConstraintKind.RIGID_BEAM, ConstraintKind.RIGID_ROD,
    }
    for rec in node_constraints:
        if allowed_ids is not None and id(rec) not in allowed_ids:
            continue
        if isinstance(rec, NodePairRecord):
            if rec.kind in rigid_pair_kinds:
                kind: Literal["beam", "bar"] = (
                    "beam" if rec.kind == ConstraintKind.RIGID_BEAM else "bar"
                )
                _emit_name(emitter, rec.name)
                _warn_partly_fixed_rigid_link(
                    kind, int(rec.master_node), int(rec.slave_node),
                    rec.name, fixed)
                emitter.rigidLink(
                    kind, int(rec.master_node), int(rec.slave_node),
                )
        elif isinstance(rec, NodeGroupRecord):
            # Only rigid_body collapses to rigidLink — rigid_diaphragm
            # has its own emit; kinematic_coupling is handled by
            # _emit_kinematic_couplings (DOF-selective).
            #
            # ``as_element`` rigid bodies go out as the fork
            # LadrunoRigidBody element (_emit_rigid_body_elements, which
            # owns the tag allocator) — skip them here so they don't ALSO
            # emit a rigidLink chain.
            if (
                rec.kind == ConstraintKind.RIGID_BODY
                and not getattr(rec, "as_element", False)
            ):
                # Emit the name once for the whole group (one row in
                # H5; one ``# name`` comment in Tcl preceding the first
                # rigidLink line).
                _emit_name(emitter, rec.name)
                if rec.slave_nodes:      # one warning for the body
                    _warn_partly_fixed_rigid_link(
                        "beam", int(rec.master_node),
                        int(rec.slave_nodes[0]), rec.name, fixed)
                for sn in rec.slave_nodes:
                    emitter.rigidLink(
                        "beam", int(rec.master_node), int(sn),
                    )
        elif isinstance(rec, NodeToSurfaceRecord):
            for pair in rec.rigid_link_records:
                if pair.kind in rigid_pair_kinds:
                    pair_kind: Literal["beam", "bar"] = (
                        "beam"
                        if pair.kind == ConstraintKind.RIGID_BEAM
                        else "bar"
                    )
                    _emit_name(emitter, pair.name)
                    _warn_partly_fixed_rigid_link(
                        pair_kind, int(pair.master_node),
                        int(pair.slave_node), pair.name, fixed)
                    emitter.rigidLink(
                        pair_kind, int(pair.master_node), int(pair.slave_node),
                    )


def _emit_rigid_body_elements(
    emitter: "Emitter", node_constraints: Iterable[object],
    tag_plan: "TagPlan",
    *, allowed_ids: frozenset[int] | None = None,
) -> None:
    """Emit ``element LadrunoRigidBody`` per :class:`NodeGroupRecord` row
    with ``kind == 'rigid_body'`` and ``as_element=True`` (ADR 0071).

    The whole node set ``{master_node, *slave_nodes}`` becomes one 6-DOF
    rigid body (fork class tag 33015, 3D only) with a private internal
    CoM node and condensed mass — what the default rigidLink-chain form
    cannot represent. Signature::

        element LadrunoRigidBody $tag $N $s1..$sN [-mass $m]

    ``-internalNode`` is omitted so the fork auto-assigns the CoM node
    (``9000000 + eleTag``, collision-safe). ``rec.mass`` ``None`` ⇒
    ``-mass`` omitted ⇒ the element condenses mass from the slaves.

    **Fork-only:** the line emits on any build; the live emitter gates
    ``LadrunoRigidBody`` through ``_FORK_ONLY_ELEMENTS`` so a stock
    OpenSees build fails loud (it does not know class tag 33015).

    The element tags come from :func:`plan_mp_elements`
    (:func:`_mp_element_tagger`).
    """
    tag_of = _mp_element_tagger(tag_plan)
    for rec in _rigid_body_element_records(node_constraints, allowed_ids):
        _emit_name(emitter, rec.name)
        body_nodes = [int(rec.master_node), *(int(s) for s in rec.slave_nodes)]
        ele_tag = tag_of(mp_element_entry("rigid_body", rec))
        args: list[int | float | str] = [len(body_nodes), *body_nodes]
        if rec.mass is not None:
            args += ["-mass", float(rec.mass)]
        omega = getattr(rec, "omega", None)
        if omega is not None:
            args += ["-omega", *(float(w) for w in omega)]
        emitter.element("LadrunoRigidBody", ele_tag, *args)


def _emit_equal_dofs(
    emitter: "Emitter", node_constraints: Iterable[object],
    *, allowed_ids: frozenset[int] | None = None,
    fixed: "FixedDofs | None" = None,
) -> None:
    """Emit ``equalDOF`` per :class:`NodePairRecord` (equal_dof) plus
    the :attr:`NodeToSurfaceRecord.equal_dof_records` expansion.

    ``allowed_ids`` filters to a rank's claimed subset when given.
    ``fixed`` (the global fix pass's :class:`FixedDofs`) rewrites a tie
    whose retained node is partly fixed (:func:`_split_partly_fixed_tie`).
    """
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if allowed_ids is not None and id(rec) not in allowed_ids:
            continue
        if isinstance(rec, NodePairRecord):
            if rec.kind == ConstraintKind.EQUAL_DOF:
                _emit_name(emitter, rec.name)
                m, sl = int(rec.master_node), int(rec.slave_node)
                tie, to_fix = _split_partly_fixed_tie(
                    m, sl, tuple(int(d) for d in rec.dofs), fixed, rec.name)
                if tie:
                    emitter.equalDOF(m, sl, *tie)
                _emit_slave_fix(emitter, m, sl, to_fix, fixed)
            elif rec.kind == ConstraintKind.EQUAL_DOF_MIXED:
                _emit_name(emitter, rec.name)
                # master_dofs (retained / RDOF) paired index-wise with
                # dofs (constrained / CDOF) — resolver guarantees equal length.
                m, sl = int(rec.master_node), int(rec.slave_node)
                rdofs = rec.master_dofs or []
                kept, to_fix = _split_partly_fixed_pairs(
                    m, sl,
                    tuple((int(r), int(c)) for r, c in zip(rdofs, rec.dofs)),
                    fixed, rec.name)
                if kept:
                    emitter.equalDOF_mixed(m, sl, list(kept))
                _emit_slave_fix(emitter, m, sl, to_fix, fixed)
        elif isinstance(rec, NodeToSurfaceRecord):
            for pair in rec.equal_dof_records:
                _emit_name(emitter, pair.name)
                m, sl = int(pair.master_node), int(pair.slave_node)
                tie, to_fix = _split_partly_fixed_tie(
                    m, sl, tuple(int(d) for d in pair.dofs), fixed, pair.name)
                if tie:
                    emitter.equalDOF(m, sl, *tie)
                _emit_slave_fix(emitter, m, sl, to_fix, fixed)


def _emit_slave_fix(
    emitter: "Emitter", master: int, slave: int,
    to_fix: "tuple[int, ...]", fixed: "FixedDofs | None",
) -> None:
    """``fix(slave, mask)`` for the DOFs a partly-fixed tie hands to the
    slave, declared under the retained node's ``fix`` owner."""
    if not to_fix or fixed is None:
        return
    ndf = int(fixed.ndf.get(slave, fixed.default_ndf))
    cursor = (fixed.decl_cursor(emitter)
              if fixed.decl_cursor is not None else None)
    if cursor is not None:
        cursor.open(fixed.owner[master])
    emitter.fix(slave, *(1 if d in to_fix else 0 for d in range(1, ndf + 1)))
    if cursor is not None:
        cursor.close()


def _emit_rigid_diaphragms(
    emitter: "Emitter", node_constraints: Iterable[object],
    *, allowed_ids: frozenset[int] | None = None,
    fixed: "FixedDofs | None" = None,
) -> None:
    """Emit ``rigidDiaphragm(perp_dir, master, *slaves)`` per
    :class:`NodeGroupRecord` row with ``kind == 'rigid_diaphragm'``.
    Uses the broker's :meth:`rigid_diaphragms` iterator for the
    perp_dir derivation; iterates the raw records in parallel to keep
    the per-record ``name`` aligned with each emit.

    ``allowed_ids`` filters to a rank's claimed subset when given.
    """
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    # Iterate the underlying records directly (not the
    # ``rigid_diaphragms()`` helper) so we still have access to the
    # original record's ``name`` field — the helper drops it.
    for rec in node_constraints:
        if allowed_ids is not None and id(rec) not in allowed_ids:
            continue
        if not (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.RIGID_DIAPHRAGM
        ):
            continue
        perp = _perp_dirn_from_normal(rec.plane_normal)
        _refuse_partly_fixed_diaphragm(emitter, rec, perp, fixed)
        _emit_name(emitter, rec.name)
        emitter.rigidDiaphragm(
            perp, int(rec.master_node),
            *(int(s) for s in rec.slave_nodes),
        )


def _refuse_partly_fixed_diaphragm(
    emitter: "Emitter | None", rec: Any, perp: int, fixed: "FixedDofs | None",
) -> None:
    """Refuse a ``rigidDiaphragm`` whose retained node is held on some of
    its in-plane DOFs and free on others, under Transformation (#2427).

    ``TransformationDOF_Group`` then adds the master's total in-plane
    displacement at every step instead of the increment: the slaves are
    right, the master is not (1, 3, 6, 10 x the increment over four
    linear steps, where Lagrange gives 1, 2, 3, 4). A master free or fully
    held in plane, or fixed only out of plane, is exact.
    """
    if fixed is None:
        return
    forced = fixed.no_analysis and bool(getattr(
        getattr(emitter, "caps", None),
        "suppress_analysis_chain_auto_emit", False))
    if not (fixed.transformation or forced):
        return
    master = int(rec.master_node)
    from .ndf import _DIAPHRAGM_TIED_DOFS

    plane = _DIAPHRAGM_TIED_DOFS.get((fixed.ndm, perp))
    if plane is None:
        return
    held = (set(fixed.fixed.get(master, frozenset()))
            | {d for d in plane if (master, d) in fixed.staged}) & plane
    if not held or held == plane:
        return
    raise BridgeError(
        f"rigidDiaphragm {perp} {master} ..."
        + (f" ({rec.name})" if getattr(rec, "name", None) else "")
        + f": its master node is fixed on in-plane DOF(s) {sorted(held)} "
        f"and free on {sorted(plane - held)}. Under the Transformation "
        "handler (declared, or auto-emitted for MP constraints) OpenSees "
        "then accumulates the master's total in-plane displacement at "
        "every step, a silently wrong master response (#2427). Declare "
        "ops.constraints.Lagrange() or Penalty(...), or drop the master's "
        "in-plane fix (fixing it only out of plane is exact)."
    )


def _coupling_control_flags(
    rec: object,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None",
    *,
    allow_al_update: bool = False,
) -> "list[int | float | str]":
    """Flag tail for a coupling record's :class:`CouplingControl`.

    Translates a hosted control's ``host`` (stored as a **FEM eid** so it
    survives H5 round-trips) into the emitted OpenSees element tag through
    the bridge-built ``fem_eid_to_ops_tag`` map.  Fails loud when the
    control names a host but the map is absent (a legacy direct caller) or
    doesn't contain the eid (the host element never emitted) — emitting the
    raw FEM eid would silently scale the penalty off the wrong element.

    ``allow_al_update`` is the RBE2 gate (fork PR #839): ``-alUpdate`` is a
    ``LadrunoKinematicCoupling``-only token, but :class:`CouplingControl` is
    shared with ``LadrunoDistributingCoupling`` (RBE3) and
    ``LadrunoEmbeddedNode``.  Only :func:`_emit_kinematic_couplings` passes
    ``True``; every other emit path refuses a control carrying it rather
    than emitting a flag the fork's parser rejects.
    """
    control: "CouplingControl | None" = getattr(rec, "control", None)
    if control is None:
        return []
    if getattr(control, "al_update", None) is not None and not allow_al_update:
        name = getattr(rec, "name", None) or "<unnamed>"
        raise ValueError(
            f"coupling {name!r}: al_update (-alUpdate) is a "
            "LadrunoKinematicCoupling-only token (fork PR #839) — this "
            "record emits a different coupling element, whose parser "
            "rejects the flag. Drop al_update, or move the knob to a "
            "kinematic_coupling (RBE2)."
        )
    host = getattr(control, "host", None)
    if host is None:
        return control.emit_flags()
    name = getattr(rec, "name", None) or "<unnamed>"
    if fem_eid_to_ops_tag is None:
        raise ValueError(
            f"coupling {name!r}: control.host={host} (a FEM element id) "
            "needs the bridge's fem_eid_to_ops_tag map to translate into "
            "the emitted OpenSees tag, but this emit pass got none."
        )
    ops_tag = fem_eid_to_ops_tag.get(int(host))
    if ops_tag is None:
        raise ValueError(
            f"coupling {name!r}: control.host={host} is not an emitted "
            "element — the FEM eid is missing from fem_eid_to_ops_tag. "
            "Name a real element of the coupled part as the host."
        )
    return control.emit_flags(host_ops_tag=int(ops_tag))


def _emit_kinematic_couplings(
    emitter: "Emitter", node_constraints: Iterable[object],
    tag_plan: "TagPlan",
    *, allowed_ids: frozenset[int] | None = None,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
) -> None:
    """Emit ``element LadrunoKinematicCoupling`` (RBE2) per
    :class:`NodeGroupRecord` row with ``kind == 'kinematic_coupling'``.

    The Ladruno-fork rigid-body driver (class tag 33012) is a penalty
    coupling that carries the correct moment-arm transport
    ``u_i = u_R + θ_R × d_i``, so an *offset* reference node is coupled
    rigidly.  This replaces the previous ``equalDOF``-per-slave expansion,
    which ignored the lever arm (correct only for coincident nodes).

    Signature::

        element LadrunoKinematicCoupling $tag $refNode $N $s1 ... $sN [-dof $c1 ...]

    The reference (master) node is ``rec.master_node``; the slaves are
    ``rec.slave_nodes``.  ``rec.dofs`` is the dependent-component list:
    an empty list means "every DOF the slave has" (the element's own
    default, ragged-layout aware), so ``-dof`` is **omitted** then; a
    non-empty list emits ``-dof $c1 ...`` to restrict the tie.  Each line
    takes its planned tag on the ``"element"`` counter, like the
    embedded-node path.

    **Fork-only:** the line emits on any build, but the live emitter gates
    ``LadrunoKinematicCoupling`` through ``_FORK_ONLY_ELEMENTS`` so a stock
    OpenSees build fails loud (it does not know class tag 33012).

    ``allowed_ids`` filters to a rank's / stage's claimed subset when given.
    The element tags come from :func:`plan_mp_elements`
    (:func:`_mp_element_tagger`).
    """
    tag_of = _mp_element_tagger(tag_plan)
    for rec in _kinematic_coupling_records(node_constraints, allowed_ids):
        _emit_name(emitter, rec.name)
        slaves = [int(sn) for sn in rec.slave_nodes]
        ele_tag = tag_of(mp_element_entry("kinematic_coupling", rec))
        args: list[int | float | str] = [
            int(rec.master_node), len(slaves), *slaves,
        ]
        if rec.dofs:
            args += ["-dof", *(int(d) for d in rec.dofs)]
        args += _coupling_control_flags(
            rec, fem_eid_to_ops_tag, allow_al_update=True,
        )
        emitter.element("LadrunoKinematicCoupling", ele_tag, *args)


#: ``stiffness="auto"`` scale factor: K = ALPHA · E_host · L_char.  A few
#: orders above the host element stiffness is all the ASD penalty needs
#: (K → ∞ only wrecks conditioning); 1e3 mirrors the fork coupling
#: elements' ``k_alpha`` default and sits **inside** the fork's measured
#: ``1e2…1e4 × k_host`` selection band (PR #839 §3.3), so the value is
#: unchanged by that work.
AUTO_STIFFNESS_ALPHA: float = 1.0e3


def make_auto_stiffness_resolver(
    fem: "FEMData", elements: "Iterable[Element]",
) -> "StiffnessResolver":
    """Build the emit-time resolver for ``stiffness="auto"`` tie records.

    ``K = AUTO_STIFFNESS_ALPHA · E_host · L_char`` where ``E_host`` is
    the largest ``E`` modulus among the materials of the declared
    element specs whose PG touches the record's master nodes, and
    ``L_char`` is the largest pairwise distance among those master
    nodes (the host face / sub-tet size).  apeGmsh never assembles
    stiffness matrices, so this deliberately estimates the host
    diagonal scale from material + geometry rather than reading
    ``max|K(i,i)|`` the way the fork's C++ ``k="auto"`` does.

    Returns a ``resolver(rec) -> float`` that raises
    :class:`BridgeError` when no E-carrying material can be found for a
    record's master nodes — an auto tie with no derivable host modulus
    must fail loud, not guess.

    The node→E and node→xyz maps are built lazily on the first record
    resolved, so handing a resolver to an emit pass that encounters no
    ``"auto"`` record costs nothing.

    **This is a CONDITIONING control, not a rigidity setting** (fork PR
    #839 §3.3).  ``AUTO_STIFFNESS_ALPHA = 1e3`` pins K to the host's own
    order of magnitude — exactly the regime where the residual constraint
    gap is largest — and it **cannot hold a rigid footing**: the fork
    measures the penalty gap as ``err = c/K_t`` exactly, so an auto-scaled
    tie lands around ``1e-4…1e-3`` of the push and stays there.  For a
    genuinely rigid tie use ``enforce="al"`` at a **moderate** ``K_t``
    plus the held-load augmentation sweep
    (:meth:`LiveOpsEmitter.augment`), which drives the gap to the solver
    floor instead.  Cranking ``k`` up is the wrong lever twice over:
    conditioning degrades linearly in ``K_t`` while the rigidity error
    stops improving, and a hand-set numeric ``k`` above ``1e6 × k_host``
    now draws a once-only fork warning whenever a ``-host`` is named.
    """
    specs = tuple(elements)
    maps: dict[str, dict[int, Any]] = {}

    def _build_maps() -> None:
        node_E: dict[int, float] = {}
        for spec in specs:
            pg = getattr(spec, "pg", None)
            mat = getattr(spec, "material", None)
            e_mod = getattr(mat, "E", None) if mat is not None else None
            if pg is None or e_mod is None:
                continue
            try:
                nids = expand_pg_to_nodes(fem, str(pg))
            except Exception:
                continue
            f_e = float(e_mod)
            if f_e <= 0.0:
                continue
            for n in nids:
                n = int(n)
                if f_e > node_E.get(n, 0.0):
                    node_E[n] = f_e
        ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        coords = np.asarray(fem.nodes.coords, dtype=np.float64)
        maps["E"] = node_E
        maps["xyz"] = {int(i): coords[k] for k, i in enumerate(ids)}

    def resolver(rec: "InterpolationRecord") -> float:
        if not maps:
            _build_maps()
        node_E = maps["E"]
        xyz = maps["xyz"]
        masters = [int(m) for m in rec.master_nodes]
        e_vals = [float(node_E[m]) for m in masters if m in node_E]
        if not e_vals:
            raise BridgeError(
                f"stiffness='auto' tie (slave={rec.slave_node}): no "
                f"declared element with an E-carrying material touches "
                f"the master nodes {masters}. Declare the host elements "
                f"with a material exposing .E (e.g. ElasticIsotropic) "
                f"before build(), or pass an explicit stiffness=."
            )
        pts = [xyz[m] for m in masters if m in xyz]
        l_char = 0.0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                l_char = max(
                    l_char, float(np.linalg.norm(pts[a] - pts[b])))
        if l_char <= 0.0:
            raise BridgeError(
                f"stiffness='auto' tie (slave={rec.slave_node}): the "
                f"master nodes {masters} span zero length — cannot "
                f"derive a characteristic host size. Pass an explicit "
                f"stiffness=."
            )
        return AUTO_STIFFNESS_ALPHA * max(e_vals) * l_char

    return resolver


def _emit_surface_couplings(
    emitter: "Emitter", surface_constraints: object, tag_plan: "TagPlan",
    *, fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
) -> None:
    """Emit ``element ASDEmbeddedNodeElement`` per
    :class:`InterpolationRecord` row (covers ``tie`` / ``distributing``
    / ``embedded`` directly; tied_contact / mortar via the
    :meth:`SurfaceCouplingRecord.slave_records` expansion).

    The second positional ``cnode`` is the constrained (embedded /
    slave) node — the ``$Cnode`` slot in the OpenSees signature
    ``element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 ...``.  The
    variadic tail carries the host element's corner node tags
    ($Rnode1..$RnodeN); ASDEmbeddedNodeElement uses isoparametric
    interpolation over those corners internally, so the per-record
    weights from :class:`InterpolationRecord` are NOT emitted here
    (they survive in the FEM record for round-tripping).  Each emitted
    line takes its planned tag on the ``"element"`` counter, so embedded-
    node element tags share the global element-tag namespace and never
    collide with structural elements or with each other under
    partitioned emit (ADR 0027 §"Tag determinism"). The tags come from
    :func:`plan_mp_elements` (:func:`_mp_element_tagger`).
    """
    tag_of = _mp_element_tagger(tag_plan)
    for rec in interpolation_records(surface_constraints):
        _write_one_interpolation(
            emitter, rec, tag_of, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            stiffness_resolver=stiffness_resolver,
        )


def _emit_one_interpolation(
    emitter: "Emitter", rec: "InterpolationRecord", tag_plan: "TagPlan",
    *, fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
) -> None:
    """Emit one :class:`InterpolationRecord` row
    (:func:`_write_one_interpolation`), its element tag read through
    :func:`_mp_element_tagger`."""
    _write_one_interpolation(
        emitter, rec, _mp_element_tagger(tag_plan),
        fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        stiffness_resolver=stiffness_resolver,
    )


def _write_one_interpolation(
    emitter: "Emitter", rec: "InterpolationRecord",
    tag_of: "Callable[[MPElementEntry], int]",
    *, fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
) -> None:
    """Emit one :class:`InterpolationRecord` row, branching on its kind.

    * ``distributing`` (RBE3) → the fork
      ``element LadrunoDistributingCoupling $tag $refNode $N $i1..iN [-w …]``:
      the reference (dependent) node R is ``rec.slave_node`` and the
      independents are ``rec.master_nodes`` (the InterpolationRecord field
      names read backwards for RBE3 — R is the *dependent*). ``rec.weights``
      ``None`` ⇒ ``-w`` omitted ⇒ the element's equal-weight default. The
      independent count is arbitrary (N ≥ 1), so the 3/4-Rnode embedded
      guard does NOT apply here. **Fork-only:** gated via
      ``_FORK_ONLY_ELEMENTS`` in the live emitter.
    * ``tie`` / ``embedded`` → ``element ASDEmbeddedNodeElement $tag $Cnode
      $Rnode1..`` via ``emitter.embeddedNode`` ($Cnode = the constrained
      node; $Rnode* = the host element's 3/4 corner nodes — weights survive
      in the FEM record but aren't emitted, the element interpolates
      isoparametrically over the corners).

    Each element line takes the tag ``tag_of`` gives it (the plan's:
    :func:`_mp_element_tagger`), so
    coupling-element tags share the global element-tag namespace (ADR
    0027 §"Tag determinism"). The route is :func:`_interpolation_route`,
    the one the tag plan reads.
    """
    _emit_name(emitter, rec.name)

    # ADR 0068 — enforce route. The equation route is a domain-level
    # EQ_Constraint (NOT an element), so it must branch BEFORE the
    # element-tag read: it takes no tag, and the plan holds none for it.
    # ``penalty`` / ``penalty_al`` keep the element paths below.
    route = _interpolation_route(rec)
    if route == "equation":
        _emit_equation_tie(emitter, rec)
        return

    ele_tag = tag_of(MPElementEntry(
        "interpolation", rec, id(rec), _INTERPOLATION_VERBS[route]))
    if route == "penalty_al":
        _emit_penalty_al_tie(emitter, rec, ele_tag, fem_eid_to_ops_tag)
        return
    if route == "distributing":
        ref = int(rec.slave_node)
        independents = [int(mn) for mn in rec.master_nodes]
        args: list[int | float | str] = [ref, len(independents), *independents]
        if rec.weights is not None:
            args += ["-w", *(float(w) for w in rec.weights)]
        args += _coupling_control_flags(rec, fem_eid_to_ops_tag)
        emitter.element("LadrunoDistributingCoupling", ele_tag, *args)
        return
    _check_embedded_rnode_count(rec)
    # stiffness="auto" (slice B): resolve to a number from the host
    # material before the emitter sees it — emitters only speak floats.
    stiffness = rec.stiffness
    if isinstance(stiffness, str):
        if stiffness_resolver is None:
            raise BridgeError(
                f"stiffness='auto' tie (slave={rec.slave_node}) reached "
                f"an emit path without an auto-stiffness resolver — "
                f"this emit entry point cannot see the declared "
                f"materials. Pass an explicit stiffness= on the tie."
            )
        stiffness = float(stiffness_resolver(rec))
    # Defect guard (silent-failures slice 3): the ASDEmbeddedNodeElement
    # C++ parity default K=1e18 is unit-blind — against E ~ 2e5 (N/mm/
    # MPa steel) it wrecks the conditioning of the stiffness matrix and
    # Newton stalls (measured: 1e10–1e12 converge, 1e18 does not).  Warn
    # once when the numeric legacy default reaches a penalty emit (old
    # h5 files, or an explicit 1e18); the message is constant so the
    # default warnings filter collapses the per-record repeats.
    elif float(stiffness) == 1.0e18:
        warnings.warn(
            "tie/tied_contact/embedded penalty stiffness left at the "
            "ASDEmbeddedNodeElement default K=1e18, which is unit-blind "
            "and known to destroy conditioning (Newton stalls) in "
            "N/mm/MPa models. Pass a calibrated stiffness (K >> the "
            "host element stiffness suffices; 1e10-1e12 for E~2e5), "
            "use stiffness='auto', or enforce='equation' for an exact "
            "multiplier tie.",
            UserWarning,
            stacklevel=2,
        )
    cnode = int(rec.slave_node)
    master_nodes = _embedded_retained_nodes(rec)
    emitter.embeddedNode(
        ele_tag, cnode, *master_nodes,
        stiffness=stiffness,
        stiffness_p=rec.stiffness_p,
        rotational=rec.rotational,
        pressure=rec.pressure,
    )


def _emit_equation_tie(
    emitter: "Emitter", rec: "InterpolationRecord",
) -> None:
    """Expand one ``enforce="equation"`` :class:`InterpolationRecord` into
    OpenSees ``equationConstraint`` rows — one per tied DOF (ADR 0068 §3).

    For each translational DOF ``d`` in ``rec.dofs`` the tie is the exact
    kinematic relation ``u_d(slave) = Σ_i w_i·u_d(master_i)``, written in
    EQ_Constraint sum-to-zero form::

        1·u_d(slave) + Σ_i (−w_i)·u_d(master_i) = 0

    No element tag is allocated (EQ_Constraint is a domain command, not an
    element); the 3/4-Rnode embedded guard does NOT apply (any face arity
    the shape functions support is fine — ``len(weights)`` masters).
    Rows are emitted in ``dofs`` order for determinism (INV-5).
    """
    weights = rec.weights
    if weights is None:
        raise ValueError(
            f"equation tie (slave={rec.slave_node}) has no interpolation "
            f"weights; the resolver must populate InterpolationRecord."
            f"weights for enforce='equation'."
        )
    master_nodes = [int(m) for m in rec.master_nodes]
    if len(master_nodes) != len(weights):
        raise ValueError(
            f"equation tie (slave={rec.slave_node}): {len(master_nodes)} "
            f"master nodes but {len(weights)} weights — mismatch."
        )
    slave = int(rec.slave_node)
    dofs = [int(d) for d in (rec.dofs or [1, 2, 3])]
    # The equation route ties TRANSLATIONS ONLY: interpolating a rotational
    # DOF from a (translational) master face is meaningless, and the master
    # nodes typically have no DOF >3 — OpenSees would fail late in
    # EQ_Constraint::setDomain ("retained DOF out of bounds"). Fail loud
    # here (ADR 0068 INV-3, dof axis — distinct from the -rot knob).
    bad = [d for d in dofs if d < 1 or d > 3]
    if bad:
        raise ValueError(
            f"equation tie (slave={slave}): dofs {bad} out of range — the "
            f"equation route ties translations only (1..3). Use "
            f"enforce='penalty'/'penalty_al' or kinematic_coupling for "
            f"rotational coupling."
        )
    for d in dofs:
        # Drop zero-weight masters: OpenSees rejects ANY zero rcoef
        # (EQ_Constraint.cpp:98) and aborts the WHOLE equationConstraint
        # line — and a slave projecting onto a master face edge/node
        # legitimately yields N_i=0. Without this filter that routine
        # geometric case silently drops the entire tie.
        retained = [
            (m, d, -float(w))
            for m, w in zip(master_nodes, weights) if float(w) != 0.0
        ]
        if not retained:
            raise ValueError(
                f"equation tie (slave={slave}, dof={d}): all interpolation "
                f"weights are zero — degenerate projection."
            )
        # A self-referential EQ (slave also a retained master) puts u_c on
        # both sides — guard rather than emit a singular row.
        if any(m == slave for m, _, _ in retained):
            raise ValueError(
                f"equation tie: slave node {slave} is also one of its own "
                f"master face nodes — degenerate self-referential tie "
                f"(coincident/duplicate node?)."
            )
        emitter.equationConstraint(slave, d, 1.0, retained)


def _emit_penalty_al_tie(
    emitter: "Emitter", rec: "InterpolationRecord", ele_tag: int,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None",
) -> None:
    """Emit one ``enforce="penalty_al"`` tie as the fork
    ``LadrunoEmbeddedNode`` element (ADR 0068 §1 / P4) — penalty +
    augmented-Lagrange + bipenalty, configured via the record's
    :class:`CouplingControl`.

    Signature (explicit-host form)::

        element LadrunoEmbeddedNode $tag $cNode $h1..$hN -shape $N1..$NN
                [-k {Ku|auto}] [-kAlpha a] [-host eleTag] [-enforce al]
                [-bipenalty -dtcr dt | -wcap beta] [-absolute]

    Unlike the penalty ``ASDEmbeddedNodeElement`` route, the shape weights
    ARE emitted (``-shape``) and any host arity is accepted (no 3/4-Rnode
    guard). Translations only in v1 (the ``-rot``/``-pressure`` element
    features are not wired through the node-to-surface resolver). **Fork-
    only**: gated in the live emitter via ``_FORK_ONLY_ELEMENTS``.
    """
    weights = rec.weights
    if weights is None:
        raise ValueError(
            f"penalty_al tie (slave={rec.slave_node}) has no interpolation "
            f"weights; the resolver must populate them."
        )
    cnode = int(rec.slave_node)
    masters = [int(m) for m in rec.master_nodes]
    if len(masters) != len(weights):
        raise ValueError(
            f"penalty_al tie (slave={cnode}): {len(masters)} master nodes "
            f"but {len(weights)} weights — mismatch."
        )
    args: list[int | float | str] = [
        cnode, *masters, "-shape", *(float(w) for w in weights),
    ]
    args += _coupling_control_flags(rec, fem_eid_to_ops_tag)
    emitter.element("LadrunoEmbeddedNode", ele_tag, *args)


def _embedded_retained_nodes(rec: "InterpolationRecord") -> list[int]:
    """The ``$Rnode`` list of the ``ASDEmbeddedNodeElement`` for ``rec``.

    ``ASDEmbeddedNodeElement`` reads 3 retained nodes as a triangle and
    4 as a **tetrahedron**. A tie record carries the corners of the
    master *face* it projected onto, so a quad4 face gives 4 coplanar
    corners, which the element takes for a zero-volume tet: the
    stiffness is singular and the analysis fails at the first solve
    (measured on the fork). A quad4 tie is therefore written on the
    triangle of the face's (0, 2) diagonal split that holds the
    projected point, the split ``embedded`` applies to quad hosts. The
    bilinear weights decide the side: ``w1 - w3 = (xi - eta) / 2``, so
    the point is in triangle (0, 1, 2) when ``w1 >= w3``, else in
    (0, 2, 3). The record keeps the four corners and their weights (the
    ``equation`` route uses them exactly); only the element changes.
    An ``embedded`` record with 4 nodes is a tet4 host and passes.
    """
    from apeGmsh._kernel.records._kinds import ConstraintKind

    nodes = [int(mn) for mn in rec.master_nodes]
    if rec.kind != ConstraintKind.TIE or len(nodes) != 4:
        return nodes
    w = rec.weights
    if w is None or len(w) != 4:
        raise ValueError(
            f"tie (slave={rec.slave_node}) on a quad4 master face has no "
            f"4 shape-function weights to choose the triangle of the "
            f"ASDEmbeddedNodeElement from; got {w!r}."
        )
    n0, n1, n2, n3 = nodes
    if float(w[1]) >= float(w[3]):
        return [n0, n1, n2]
    return [n0, n2, n3]


def _check_embedded_rnode_count(rec: object) -> None:
    """Fail-loud guard: ASDEmbeddedNodeElement only accepts 3 or 4
    Rnodes (tri host or tet host).

    The C++ parser silently misreads a 5+-Rnode line as a record with
    flag positions in the slots (anything past ``$Rnode4`` is read as
    an option string); a 2-Rnode line aborts in ``setDomain``.  Catch
    here with a clear error rather than at OpenSees runtime.
    """
    n = len(getattr(rec, "master_nodes", ()) or ())
    if n not in (3, 4):
        slave = getattr(rec, "slave_node", "?")
        name = getattr(rec, "name", None) or "<unnamed>"
        raise ValueError(
            f"ASDEmbeddedNodeElement {name!r} (slave={slave}) has "
            f"{n} master nodes; the C++ parser only accepts 3 (tri3 "
            f"host) or 4 (tet4 host).  This typically indicates a "
            f"hand-built InterpolationRecord that bypassed the "
            f"resolver, or a resolver path that produced an unsupported "
            f"host topology."
        )


def _gather_phantom_nodes(node_constraints: object) -> dict[int, tuple[float, float, float]]:
    """Walk ``NodeToSurfaceRecord`` rows and return ``{phantom_tag: xyz}``.

    Phantom tags are broker-derived (one canonical numbering) — per
    ADR 0027 INV-3 the same tag and the same coords appear on every
    rank that hosts a constraint referencing them.
    """
    out: dict[int, tuple[float, float, float]] = {}
    n2s_iter = getattr(node_constraints, "node_to_surfaces", None)
    if n2s_iter is None:
        return out
    for rec in n2s_iter():
        coords = rec.phantom_coords
        if coords is None:
            continue
        for tag, xyz in zip(rec.phantom_nodes, coords):
            t = int(tag)
            if t in out:
                continue
            out[t] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


class _StageConstraintAdapter:
    """Adapter exposing a flat list of resolved constraint records via
    the same container-method surface that :func:`emit_mp_constraints`
    and :func:`emit_mp_constraints_partitioned` consume.

    The bridge's stage builder methods (``s.embedded`` / ``s.equal_dof``
    / ``s.rigid_link`` / ``s.tie`` / ``s.tied_contact`` /
    ``s.kinematic_coupling`` / ``s.node_to_surface``) route resolved
    records into ``_StageBuilder._stage_constraint_records`` (a flat
    ``list[ConstraintRecord]``) instead of ``fem.elements.constraints``
    / ``fem.nodes.constraints``.  This adapter lets the existing
    per-kind emit helpers (``_emit_rigid_links`` /
    ``_emit_equal_dofs`` / etc.) consume that flat list unmodified —
    ``__iter__`` covers the four record-iterating helpers,
    ``node_to_surfaces`` covers the phantom-node pre-step, and
    ``interpolations`` covers the surface-coupling tail.

    Stateless aside from the wrapped list; reused on every emit pass.
    """
    __slots__ = ("_records",)

    def __init__(self, records: "Iterable[ConstraintRecord]") -> None:
        self._records = tuple(records)

    def __iter__(self) -> Iterator[object]:
        return iter(self._records)

    def node_to_surfaces(self) -> Iterator[object]:
        from apeGmsh._kernel.records._constraints import NodeToSurfaceRecord
        for rec in self._records:
            if isinstance(rec, NodeToSurfaceRecord):
                yield rec

    def interpolations(self) -> Iterator[object]:
        from apeGmsh._kernel.records._constraints import (
            InterpolationRecord, SurfaceCouplingRecord,
        )
        for rec in self._records:
            if isinstance(rec, InterpolationRecord):
                yield rec
            elif isinstance(rec, SurfaceCouplingRecord):
                for slave in rec.slave_records:
                    yield slave


def emit_stage_mp_constraints(
    stage_records: "Iterable[ConstraintRecord]",
    emitter: "Emitter",
    tag_plan: "TagPlan",
    *,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    stiffness_resolver: "StiffnessResolver | None" = None,
    fixed_dofs: "FixedDofs | None" = None,
) -> None:
    """Emit a stage's MP constraints inside the stage block (flat path).

    Mirrors :func:`emit_mp_constraints` step-for-step but iterates the
    stage's flat record pool via :class:`_StageConstraintAdapter`
    instead of ``fem.{nodes,elements}.constraints``.  Per ADR 0034 the
    stage's constraint emit runs AFTER stage regions and BEFORE the
    stage's ``domain_change`` gate, so the constrained nodes / elements
    (which emitted at the top of the stage block) are already in the
    OpenSees domain when the constraint references them.

    Phantom-tag predicate update is additive: the existing predicate
    (populated by the global pre-stage pass) is unioned with this
    stage's phantom tags so the H5 emitter classifies subsequent
    ``node()`` calls correctly for both populations.

    No-op when ``stage_records`` is empty.
    """
    from ..tag_resolution import ATTR_PHANTOM_NODE_TAGS, set_phantom_node_tags

    adapter = _StageConstraintAdapter(stage_records)
    if not adapter._records:
        return

    # Additive phantom-tag predicate update — see docstring.
    stage_phantoms = set(_gather_phantom_nodes(adapter).keys())
    if stage_phantoms:
        existing: frozenset[int] = getattr(emitter, ATTR_PHANTOM_NODE_TAGS, frozenset())
        set_phantom_node_tags(emitter, set(existing) | stage_phantoms)

    # Same ordering as emit_mp_constraints (INV-3).
    _emit_phantom_nodes(emitter, adapter)
    staged = fixed_dofs.in_stage() if fixed_dofs is not None else None
    _emit_rigid_links(emitter, adapter, fixed=staged)
    _emit_rigid_body_elements(emitter, adapter, tag_plan)
    _emit_equal_dofs(emitter, adapter, fixed=staged)
    _emit_rigid_diaphragms(emitter, adapter, fixed=staged)
    _emit_kinematic_couplings(
        emitter, adapter, tag_plan, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
    )
    _emit_surface_couplings(
        emitter, adapter, tag_plan, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        stiffness_resolver=stiffness_resolver,
    )
