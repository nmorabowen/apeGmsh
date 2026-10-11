"""records: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from ..types import Damping, Primitive, Recorder
from ...pattern.pattern import Plain
from ..stage_window import warn_stage_series_outside_window


if TYPE_CHECKING:
    from apeGmsh._kernel.records._constraints import (
        ConstraintRecord,
        InterfaceRecord,
        InterpolationRecord,
    )
    from ...emitter.base import Emitter
    from ...analysis.strategy import Ladder


#: Emit-time resolver for ``stiffness="auto"`` tie records — maps one
#: :class:`InterpolationRecord` to the penalty stiffness to emit for it.
#: Produced by :func:`make_auto_stiffness_resolver` and threaded through
#: the MP-constraint emit paths as ``stiffness_resolver=``.
StiffnessResolver = Callable[["InterpolationRecord"], float]


# ---------------------------------------------------------------------------
# Model-level records collected on the bridge between build() calls.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FixRecord:
    """One ``fix`` directive registered through ``apeSees.fix``.

    Either ``pg`` or ``nodes`` is non-None (validated at the call site).
    The build pipeline expands ``pg`` into a per-node fan-out at emit
    time (one ``emitter.fix(node, *dofs)`` per node).
    """

    pg: str | None
    nodes: tuple[int, ...] | None
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EquationConstraintRecord:
    """One user ``equationConstraint`` row from ``apeSees.equation_constraint``.

    The OpenSees ``EQ_Constraint`` relation, in its sum-to-zero form::

        ccoef * u[cdof](cnode) + sum_i rcoef_i * u[rdof_i](rnode_i) = 0

    ``retained`` holds the ``(rnode, rdof, rcoef)`` triples in call order.
    Node tags are FEM node ids (the bridge emits nodes under their FEM
    ids). Validated at declaration (:func:`make_equation_constraint_record`)
    and again against the snapshot's nodes and per-node ndf at emit
    (:func:`emit_equation_constraints`).
    """

    cnode: int
    cdof: int
    ccoef: float
    retained: tuple[tuple[int, int, float], ...]


def make_equation_constraint_record(
    cnode: int, cdof: int, ccoef: float,
    retained: "Iterable[tuple[int, int, float]]",
) -> EquationConstraintRecord:
    """Validate and freeze one ``equationConstraint`` row.

    Refuses what OpenSees refuses late or silently: a zero or non-finite
    coefficient (``EQ_Constraint`` rejects a zero ``rcoef`` and aborts the
    whole line), a DOF below 1, an empty retained set, and a row whose
    constrained ``(node, dof)`` also appears among the retained ones (``u_c``
    on both sides — a singular or trivial row).
    """
    import math

    def _dof(value: object, what: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"equation_constraint: {what} must be an int, got {value!r}.")
        if int(value) < 1:
            raise ValueError(f"equation_constraint: {what} must be >= 1, got {value!r}.")
        return int(value)

    def _coef(value: object, what: str) -> float:
        c = float(value)  # type: ignore[arg-type]
        if not math.isfinite(c) or c == 0.0:
            raise ValueError(
                f"equation_constraint: {what} must be finite and non-zero, "
                f"got {value!r} (OpenSees rejects a zero coefficient and "
                f"drops the whole row)."
            )
        return c

    c_node = _dof(cnode, "constrained node")
    c_dof = _dof(cdof, "constrained dof")
    c_coef = _coef(ccoef, "constrained coefficient")
    rows: list[tuple[int, int, float]] = []
    for i, triple in enumerate(retained):
        try:
            rn, rd, rc = triple
        except (TypeError, ValueError):
            raise ValueError(
                f"equation_constraint: retained[{i}] must be a (node, dof, "
                f"coef) triple, got {triple!r}."
            ) from None
        rows.append((
            _dof(rn, f"retained[{i}] node"),
            _dof(rd, f"retained[{i}] dof"),
            _coef(rc, f"retained[{i}] coef"),
        ))
    if not rows:
        raise ValueError(
            "equation_constraint: retained must hold at least one "
            "(node, dof, coef) triple."
        )
    if any((rn, rd) == (c_node, c_dof) for rn, rd, _ in rows):
        raise ValueError(
            f"equation_constraint: the constrained (node, dof) = "
            f"({c_node}, {c_dof}) also appears among the retained ones — "
            f"u_c would sit on both sides of the equation."
        )
    return EquationConstraintRecord(
        cnode=c_node, cdof=c_dof, ccoef=c_coef, retained=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class SupportRecord:
    """One ``s.support`` directive — a stage-bound HOLD constraint (ADR 0052).

    Shape mirrors :class:`FixRecord` (``pg`` XOR ``nodes`` + a per-ndf
    ``dofs`` 0/1 flag tuple), but the emit is different: instead of
    ``fix`` (absolute, snaps the DOF to ``t = 0``), each flagged DOF
    emits ``sp <node> <dof> [nodeDisp <node> <dof>] -const`` inside the
    stage's dedicated constant HOLD pattern — pinning the DOF at its
    *current deformed* value with zero initial force. The ``nodeDisp``
    capture resolves at runtime in the emitted deck, so no displacement
    value is stored here.
    """

    pg: str | None
    nodes: tuple[int, ...] | None
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MassRecord:
    """One ``mass`` directive registered through ``apeSees.mass`` or
    ``_StageBuilder.mass``.

    ``overwrite`` (Phase SSI-2.E) opts the record out of validator V2's
    cross-tier duplicate-mass check.  V2 normally refuses a second
    mass assignment to a node already mass-assigned in an earlier tier
    because OpenSees ``Domain::setMass`` silently overwrites and the
    physics change would otherwise be silent.  Stage-bound callers that
    deliberately want to mid-run reassign mass — e.g. swap a temporary
    construction mass for a permanent one — pass ``overwrite=True`` to
    acknowledge the overwrite explicitly.  Emits the same ``mass`` line
    either way (OpenSees has no syntactic distinction); the flag is a
    build-time validator-bypass marker only.
    """

    pg: str | None
    nodes: tuple[int, ...] | None
    values: tuple[float, ...]
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class NdfRecord:
    """One ``ops.ndf(target, ndf)`` directive (ADR 0049 — the sole explicit
    per-node ndf channel).

    Targets an element-LESS decoupled node by ``handle`` (a
    ``DecoupledNodeDef`` returned from ``g.decouple_node``) XOR an int
    ``tag``. Handle→tag resolution is deferred to build time
    (:func:`resolve_ndf_overlay`), which raises when the handle's ``.tag`` is
    still ``None`` (un-meshed). ``handle`` is typed ``object`` to avoid a
    session import; the resolver reads ``handle.tag``. ``ops.ndf`` exists
    only for nodes inference cannot reach (no incident element); an attempt
    to target a mesh node or an element-touched node fails loud, preserving
    ADR 0048's no-two-headed-model guarantee.
    """

    handle: object | None
    tag: int | None
    ndf: int


@dataclass(frozen=True, slots=True)
class SPRemovalRecord:
    """One ``s.remove_sp`` directive — releases SP constraints on a set
    of nodes for a stage (Phase SSI-2.E).

    Stage-bound only — no top-level ``apeSees.remove_sp``.  Either
    ``pg`` or ``nodes`` is non-None (validated at the call site).  The
    build pipeline expands ``pg`` into a per-node fan-out at emit time,
    one ``emitter.remove_sp(node, dof)`` per (node, dof) pair.

    Validator V5 (Phase SSI-2.E) refuses any record whose target SP was
    not declared in an earlier scope (global ``apeSees.fix`` pool or a
    strictly-earlier stage's ``s.fix`` pool) or that was already removed
    by an earlier stage's ``s.remove_sp``.  Same-stage ``s.fix`` does
    NOT make an SP available for same-stage removal — the removal emits
    before the fix in the stage block, so the SP doesn't exist yet at
    the remove line.
    """

    pg: str | None
    nodes: tuple[int, ...] | None
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ZeroVelocityRecord:
    """One ``s.zero_velocities(...)`` directive — zeroes the nodal
    velocity AND acceleration state at a stage boundary.

    Stage-bound only — no top-level ``apeSees.zero_velocities``.
    ``nodes=None`` means the whole domain (every node in
    ``fem.nodes.ids``); an explicit tuple restricts the fan-out to that
    node set.  Emit expands to one ``setNodeVel``/``setNodeAccel`` pair
    per (node, DOF), with the DOF range taken from the node's effective
    ndf (a u-p node has 4), immediately before the stage's ``analyze``.
    """

    nodes: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ElementRemovalRecord:
    """One ``s.remove_element`` directive — drops elements from the
    Domain mid-analysis (Phase SSI-2.E).

    Stage-bound only.  Either ``pg`` or ``elements`` is non-None
    (validated at the call site).  The build pipeline expands ``pg``
    via ``expand_pg_to_elements`` into per-element fan-out at emit
    time, one ``emitter.remove_element(tag)`` per element.

    Validator V6 (Phase SSI-2.E) refuses any record whose target
    element was not previously emitted in an earlier scope (globally
    emitted OR activated by a strictly-earlier stage's
    ``s.activate(pgs=)``) or that was already removed by an earlier
    stage.  Element nodes are NOT removed by ``remove element`` — they
    remain in the Domain and may continue to carry SP / mass / load
    declarations from other tiers.
    """

    pg: str | None
    elements: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class MaterialStageRecord:
    """One ``s.update_material_stage`` directive — flips the SANISAND
    elastic/elastoplastic stage flag mid-analysis (Phase SSI-2.E).

    Stage-bound only — no top-level ``apeSees.update_material_stage``.
    ``mat_tags`` are bridge-allocated ``nDMaterial`` tags resolved at
    the call site: unlike the PG-bearing removal records above there is
    nothing to defer, since a material's tag is allocated the moment it
    is registered.  Emit fans out one
    ``emitter.update_material_stage(tag, stage)`` per tag, in the order
    the user listed the materials.

    Validator V7 (Phase SSI-2.E) refuses a record whose target material
    is not referenced by an element that is live in the Domain at or
    before this stage.  ``MaterialStageParameter::setDomain()`` walks
    the Domain's *elements* looking for the material tag; a miss
    prints a WARNING and the flip is a silent no-op.
    """

    mat_tags: tuple[int, ...]
    stage: int


@dataclass(frozen=True, slots=True)
class UpdateParameterRecord:
    """One ``s.update_parameter`` directive — a typed pass-through over
    the OpenSees ``parameter`` / ``addToParameter`` / ``updateParameter``
    primitive that ``s.initial_stress`` and ``s.activate_absorbing``
    already drive internally.

    Stage-bound only.  Exactly one of ``pg`` / ``elements`` is non-None
    (validated at the call site) — the target is ALWAYS an element,
    because ``parameter`` / ``addToParameter`` only address ``node`` /
    ``element`` / ``region`` / ``loadPattern``
    (``OpenSeesParameterCommands.cpp`` ``OPS_Parameter``,
    ``OPS_addToParameter``).  A *material* parameter is reached THROUGH
    an element: the element forwards the unmatched argv to its
    integration-point materials (``LadrunoUP::setParameter``
    ``LadrunoUP.cpp:1948-1971``), and the material matches on
    ``argv[0] == name`` plus ``argv[1] == its own tag``
    (``ManzariDafalias::setParameter`` ``ManzariDafalias.cpp:820-857``).
    ``mat_tag`` carries that trailing tag; ``None`` means the parameter
    is the element's own (``xPerm`` / ``yPerm`` / ``zPerm``).

    No registry of "known" parameter names — the element / material
    ``setParameter`` is the authority, and an unrecognised name is
    already loud there.
    """

    name: str
    value: float
    pg: str | None
    elements: tuple[int, ...] | None
    mat_tag: int | None


@dataclass(frozen=True, slots=True)
class RegionAssignmentRecord:
    """One ``apeSees.region(name=...)`` directive — assigns nodes to a
    named OpenSees Region.

    Either ``pg`` or ``nodes`` is non-None (validated at the call site).
    Multiple records sharing the same ``name`` accumulate into one
    ``region $tag -node n1 n2 ...`` line at emit time: members merge by
    name, one tag is allocated per name, duplicates within a name are
    de-duped while preserving first-seen order.
    """

    name: str
    pg: str | None
    nodes: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class RayleighRecord:
    """One global ``rayleigh`` directive (ADR 0053, D1).

    Carries the four positional coefficients of the OpenSees
    ``rayleigh $alphaM $betaK $betaK0 $betaKc`` command, already resolved:
    a ratio-helper fit (when the ratio form is used) is applied at the call
    site in ``_DampingNS.rayleigh`` before this record is built, so the
    emitter just renders the four numbers.

    ``on`` is the scope (ADR 0053 D2): an empty tuple means **global**
    (a bare ``rayleigh`` line); a non-empty tuple of physical-group names
    means **region-scoped** — each name resolves to its elements at emit
    time and emits one ``region $tag -ele … -rayleigh αM βK βK0 βKc`` line.
    """

    alpha_m: float
    beta_k: float
    beta_k_init: float
    beta_k_comm: float
    on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModalDampingRecord:
    """One ``ops.damping.modal(...)`` directive (ADR 0053 D4).

    Records a bundled ``eigen <solver> <modes>`` + ``modalDamping <f1> [..]``
    emit. ``factors`` is one ratio (uniform across all modes) or ``modes``
    per-mode ratios. Domain-wide — there is no region scope. Emitted
    driver-post, reusing the ``eigen`` emitter so the live emitter runs the
    solve before the factors are set.
    """

    factors: tuple[float, ...]
    modes: int
    solver: str


@dataclass(frozen=True, slots=True)
class DampingAttachRecord:
    """One ``ops.damping.<type>(on=...)`` attachment (ADR 0053 D3).

    ``prim`` is the registered :class:`~apeGmsh.opensees._internal.types.Damping`
    object (it emits its own ``damping <Type> $tag`` line in the pre-element
    definition group; its tag is read back via ``tag_for[id(prim)]`` at the
    attach pass). ``on`` is the tuple of physical-group names whose elements
    the object attaches to, one ``region $tag -ele … -damp $dampTag`` line per
    name.
    """

    prim: Damping
    on: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One ``apeSees.stage(name)`` block — a per-stage analysis chain
    + per-stage ramped initial-stress records + the analyze run params.

    Used by Phase SSI-2.A staged-analysis decks.  Stages emit in the
    order they appear in ``BuiltModel.stage_records``; each one emits:

    1. ``stage_open(name)`` — comment delimiter for deck readability.
    2. Stage's :class:`InitialStressRecord` instances — parameter
       declarations + step_hook_ramp procs + addToParameter calls.
    3. The analysis chain (constraints / numberer / system / test /
       algorithm / integrator / analysis directive) — emitted via
       each primitive's ``_emit``.
    4. ``analyze`` loop — hook-wrapped if any initial_stress
       registered a ramp.
    5. ``stage_close()`` — emits ``loadConst -time 0.0`` +
       ``wipeAnalysis`` + clears the dispatcher lists.

    Analysis-chain primitives stay on the bridge's ``_primitives``
    list (they need tags via the topological-order pass); the stage
    record holds *references* to them, not copies.  ``BuiltModel.emit``
    skips the global pre-element emit of chain primitives when stages
    are declared, so each stage's chain is the only one OpenSees sees
    at run time.

    Phase SSI-2.D adds stage-bound ``fix`` / ``mass`` / ``region`` /
    ``recorder`` pools (``fix_records``, ``mass_records``,
    ``region_records``, ``recorder_specs``).  PR-A ships the dataclass
    slots + V1-V4 validators only — emit wiring lands in PR-B (fix +
    mass) and PR-C (region + recorder).  Existing construction sites
    that don't pass the new fields keep working via the ``()``
    defaults.

    Stage-bound constraint pool (``stage_constraint_records``) holds
    resolved :class:`~apeGmsh._kernel.records._constraints.ConstraintRecord`
    instances authored via ``s.embedded`` / ``s.equal_dof`` /
    ``s.rigid_link`` / ``s.tie`` / ``s.tied_contact`` /
    ``s.kinematic_coupling`` / ``s.node_to_surface``.  Each method
    calls the existing kernel resolver and routes the resolved records
    here instead of into ``fem.elements.constraints`` /
    ``fem.nodes.constraints``, so the global pre-stage constraint
    emit pass skips them and the per-stage emit hook (after regions,
    before ``domain_change``) emits them inside the owning stage's
    block.
    """

    name: str
    initial_stress_records: tuple["InitialStressRecord", ...]
    # Analysis chain references — primitives live on the bridge's
    # ``_primitives`` list; the stage knows which ones to bind for
    # this stage's analyze loop.  Stored as ``Primitive`` (generic)
    # to avoid pulling in the full analysis-chain type hierarchy
    # here; runtime types are enforced by the stage builder.
    test: "Primitive | None"
    algorithm: "Primitive | None"
    integrator: "Primitive | None"
    constraints: "Primitive | None"
    numberer: "Primitive | None"
    system: "Primitive | None"
    analysis: "Primitive | None"
    n_increments: int
    dt: float | None
    # ADR 0057 Phase A: optional solution-strategy ladder for this
    # stage's analyze loop.  Resolved to an emitter-ready StrategySpec
    # at emit time with this record's ``algorithm`` as rung 0.  NOT
    # persisted to H5 in Phase A (declaration persistence is Phase C),
    # so an H5 replay runs the plain fail-loud loop.
    strategy: "Ladder | None" = None
    # Phase SSI-2.B: element-PG names that come online in this stage.
    # The bridge filters Element primitives whose ``pg=`` matches any
    # entry here into the stage's topology-emit block.  Nodes
    # referenced only by stage-bound elements emit alongside them.
    # Multiple stages can NOT share the same PG (first stage wins —
    # later activations of the same PG are validated as errors at
    # build time).  An element whose PG is not activated by any
    # stage stays global (emitted before stage 1).
    activated_pgs: tuple[str, ...] = ()
    # Transient → static handover: nodal velocity / acceleration
    # zeroing (``s.zero_velocities``).  Emitted LAST inside the stage
    # block — after the analysis chain, the stage patterns and the
    # optional ``reset``, immediately before ``analyze`` — so nothing
    # can restore the kinematic state between the zeroing and the step
    # that would otherwise read it.  Default ``()`` keeps existing
    # construction sites working unmodified.
    zero_velocity_records: tuple["ZeroVelocityRecord", ...] = ()
    # Phase SSI-2.D: stage-bound BC + recorder pools.  Populated by
    # ``_StageBuilder.fix / .mass / .region / .recorder`` (PR-B/C).
    # PR-A ships the dataclass slots + the validator surface; emit
    # wiring lands in PR-B (fix + mass) and PR-C (region + recorder).
    # Defaults to empty tuples so existing construction sites and
    # tests continue to work without modification.
    fix_records: tuple[FixRecord, ...] = ()
    mass_records: tuple[MassRecord, ...] = ()
    region_records: tuple[RegionAssignmentRecord, ...] = ()
    recorder_specs: tuple[Recorder, ...] = ()
    # ADR 0053 D5: stage-bound damping pools.  ``rayleigh_records``
    # (``s.damping.rayleigh``) and ``damping_attach_records``
    # (``s.damping.uniform`` / ``sec_stif`` / ``urd`` / ``urd_beta``)
    # emit inside this stage's block (after ``domainChange``, before the
    # analysis chain) so the stage's elements are in the domain when the
    # ``rayleigh`` / ``region -damp`` lines bind.  The Damping objects
    # themselves stay in the bridge's ``_primitives`` (defined once,
    # pre-element); only the attach is stage-scoped.  Modal damping is
    # NOT staged (deferred — eigen / wipeAnalysis interaction).  Default
    # ``()`` keeps existing construction sites working unmodified.
    rayleigh_records: tuple[RayleighRecord, ...] = ()
    damping_attach_records: tuple[DampingAttachRecord, ...] = ()
    # ADR 0051 (BL-3): stage-scoped load patterns.  Populated by
    # ``_StageBuilder.pattern(series=)`` — each is a stage-owned
    # :class:`Plain` (context manager) whose ``load`` / ``sp`` /
    # ``from_model`` lines emit inside this stage's block (after the
    # analysis chain, before ``analyze``) and are frozen by the
    # stage's ``stage_close`` ``loadConst``.  The pattern stays in the
    # bridge's ``_primitives`` (so its tag is allocated), but is
    # claimed via ``apeSees._stage_claimed_pattern_ids`` so the global
    # post-element pattern pass SKIPS it — no double emission.  Default
    # ``()`` keeps existing construction sites working unmodified.
    pattern_specs: tuple[Plain, ...] = ()
    # ADR 0052 slice 1: stage-bound HOLD supports (``s.support``).  Each
    # flagged DOF emits ``sp <node> <dof> [nodeDisp ...] -const`` inside
    # ``support_pattern`` — a dedicated per-stage ``Plain`` bound to a
    # shared ``Constant`` series, claimed via
    # ``apeSees._stage_claimed_pattern_ids`` (so the global + 7b pattern
    # passes skip it) and emitted by a dedicated HOLD block in the BC
    # region of the stage.  ``support_pattern`` is None iff
    # ``support_records`` is empty.  Default ``()`` / ``None`` keeps
    # existing construction sites and tests working unmodified.
    support_records: tuple[SupportRecord, ...] = ()
    support_pattern: "Plain | None" = None
    # Stage-bound constraint pool.  Populated by
    # ``_StageBuilder.embedded`` / ``.equal_dof`` / ``.rigid_link`` /
    # ``.tie`` / ``.tied_contact`` / ``.kinematic_coupling`` /
    # ``.node_to_surface`` — each calls the kernel resolver and
    # appends resolved records here.  Default ``()`` keeps existing
    # construction sites and tests working unmodified.
    stage_constraint_records: tuple["ConstraintRecord", ...] = ()
    # ADR 0093 S7 (INV-6): stage-bound interface pool.  Populated by
    # ``_StageBuilder.interface(name=)``, which CLAIMS resolved
    # :class:`InterfaceRecord` rows off ``fem.elements.interfaces`` (the
    # records stay on the broker — only the emit moves).  Deliberately
    # NOT folded into ``stage_constraint_records``: an interface is not
    # an MP constraint, and its emit shape is the atomic per-pair unit
    # (phantom → equalDOF → two materials → zeroLength) that
    # :func:`emit_stage_interfaces` writes AFTER the stage's MP
    # constraints and BEFORE the stage's ``domain_change``.  Default
    # ``()`` keeps existing construction sites working unmodified.
    stage_interface_records: tuple["InterfaceRecord", ...] = ()
    # Phase SSI-2.E: between-stage Domain mutators.  Removals emit
    # BEFORE the stage's new fix / mass / region lines so a stage can
    # release a prior-stage support and immediately re-apply a new
    # value to the same target.  Validators V5 (remove_sp) and V6
    # (remove_element) gate these at build time — see
    # ``_validate_remove_sp_targets`` / ``_validate_remove_element_targets``.
    remove_sp_records: tuple[SPRemovalRecord, ...] = ()
    remove_element_records: tuple[ElementRemovalRecord, ...] = ()
    # Phase SSI-2.E: SANISAND elastic → elastoplastic stage flips.
    # Emitted AFTER the removals (so a stage can release, re-fix, then
    # flip) and after this stage's element activation — the OpenSees
    # command reaches materials through the Domain's live elements.
    # Validator V7 gates these — see ``_validate_material_stage_targets``.
    update_material_stage_records: tuple[MaterialStageRecord, ...] = ()
    # Phase SSI-2.E: time-state mutators.  ``set_time`` overrides the
    # ``loadConst -time 0.0`` reset that the previous stage's
    # ``stage_close`` emitted (useful when the next stage's pseudo-
    # time should start at a non-zero value); emitted right after
    # ``stage_open``.  ``set_creep_on`` toggles creep for time-
    # dependent concrete materials, emitted alongside ``set_time``.
    # ``pre_analyze_reset`` requests the ``reset`` command right before
    # the stage's ``analyze`` (rarely used; the bare OpenSees
    # ``reset`` wipes Domain state back to the last ``setTime``).
    set_time: float | None = None
    set_creep_on: bool | None = None
    pre_analyze_reset: bool = False
    # TIMs A8: optional per-stage profiler bracket (``s.profile``).
    # ``None`` (default) keeps existing construction sites and tests
    # working unmodified — no bracket emits for a stage that never
    # calls ``s.profile``.
    profile: "ProfileRecord | None" = None
    # ADR 0054 AB-3: ASDAbsorbingBoundary stage flip (``s.activate_absorbing``).
    # Emitted after the analysis chain is established (so the domain holds the
    # stage's elements) — one-shot ``parameter`` / ``addToParameter ... stage`` /
    # ``updateParameter 1`` per record.  Default ``()`` keeps existing
    # construction sites working unmodified.
    activate_absorbing_records: tuple["ActivateAbsorbingRecord", ...] = ()
    # ``s.update_parameter`` — the typed pass-through over the same
    # ``parameter`` / ``addToParameter`` / ``updateParameter`` primitive
    # the two records above drive internally.  Emitted right after the
    # absorbing flip (same slot rationale: the stage's elements are in
    # the Domain, the chain is established, the analyze loop has not
    # started).  Default ``()`` keeps existing construction sites
    # working unmodified.
    update_parameter_records: tuple["UpdateParameterRecord", ...] = ()

    def __post_init__(self) -> None:
        # #1334: a stage pattern whose Path series is zero at every
        # increment of the stage applies no load; warn when the stage
        # closes, before any route emits or runs it.
        warn_stage_series_outside_window(self)


@dataclass(frozen=True, slots=True)
class InitialStressRecord:
    """One ``apeSees.initial_stress(name=...)`` directive — a ramped
    in-situ stress tensor applied to a set of ASDPlasticMaterial3D
    elements via the OpenSees ``parameter`` / ``addToParameter`` /
    ``updateParameter`` mechanism.

    Either ``pg`` or ``elements`` is non-None (validated at the call
    site).  Each record fans out into:

    * Three parameter declarations (XX, YY, ZZ) with bridge-allocated
      tags.
    * One ``addToParameter`` per element (per-rank-scoped in MP).
    * One :class:`StepHookRampRecord` registering the per-step ramp.

    ``lambda_install`` scales the target stress baked into the ramp's
    target_value (target_value = sigma * lambda_install); the per-step
    factor always ramps 0 → 1.0 over ``ramp_steps``.

    Parameters
    ----------
    name
        Unique label for this initial-stress region — used to name the
        Tcl proc / Python function and the per-hook state container.
        MUST be a valid Tcl identifier (alphanumeric + underscore, not
        starting with a digit).
    pg
        Physical group whose elements receive the ramped stress.
    elements
        Explicit list of element tags.  XOR with ``pg``.
    sigma_xx, sigma_yy, sigma_zz
        Target Cauchy stress per component (compression negative).
        Units must match the model's stress unit.
    ramp_steps
        Number of analyze steps over which the factor ramps 0 → 1.0.
        After ``ramp_steps`` analyze calls, the cumulative
        ``updateParameter`` is exactly ``sigma_* * lambda_install``;
        subsequent steps emit ``updateParameter $tag 0.0`` (no-op).
    lambda_install
        Fraction of target stress to install (0 < lambda <= 1).  1.0
        = full install (default).  0.5 = 50% relaxation (intermediate
        convergence-confinement step).
    """

    name: str
    pg: str | None
    elements: tuple[int, ...] | None
    sigma_xx: float
    sigma_yy: float
    sigma_zz: float
    ramp_steps: int
    lambda_install: float


@dataclass(frozen=True, slots=True)
class ActivateAbsorbingRecord:
    """One ``s.activate_absorbing(...)`` directive (ADR 0054, AB-3).

    Flips a set of ``ASDAbsorbingBoundary*`` elements from
    ``Stage_StaticConstraint`` (0) to ``Stage_Absorbing`` (1) via the
    OpenSees ``parameter`` / ``addToParameter ... stage`` / ``updateParameter``
    one-shot sequence — the staged switch between gravity and the transient.
    One-way (0→1); emitted once per stage, after the gravity stages'
    ``loadConst`` and before the transient ``analyze``.

    Exactly one of ``pg`` / ``elements`` is non-None (validated at the call
    site).  ``pg`` is typically the plane-wave skin roll-up
    (``AbsorbingSkinResult.skin_all_pg``).
    """

    pg: str | None
    elements: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """One ``s.profile(...)`` directive (TIMs A8) — brackets THIS
    stage's ``analyze`` loop with the Ladruno fork's stack profiler,
    reported under the stage's own name.

    ``deep`` / ``memory`` / ``per_step`` mirror the three ``start``
    flags on :class:`~apeGmsh.opensees._internal.ns.profiler._ProfilerNS`
    (``-deep`` / ``-memory`` / ``-perStep``) — the bridge-level
    ``ops.profiler.*`` verbs bracket the WHOLE deck's appended
    ``analyze`` call; this record reuses the same
    ``Emitter.profiler(*args)`` Protocol method to bracket a SINGLE
    stage's ``analyze`` loop instead. Emitted as ``profiler start
    [flags]`` immediately before the stage's analyze loop and
    ``profiler stop`` + ``profiler report <stage name>.h5`` immediately after (before
    ``stage_close``) — filename derived from the stage's name so no
    extra kwarg is needed.
    """

    deep: bool = False
    memory: bool = False
    per_step: bool = False


def _emit_name(emitter: "Emitter", name: object) -> None:
    """Emit ``emitter.mp_constraint_comment(name)`` if ``name`` is
    a non-empty string (ADR 0022 INV-2).
    """
    if name is None:
        return
    if not isinstance(name, str):
        return
    if not name:
        return
    emitter.mp_constraint_comment(name)
