"""
``apeSees`` — the bridge class.

Takes a :class:`~apeGmsh.mesh.FEMData` snapshot at construction. Never
imports gmsh. Holds the user's typed primitive declarations and
delegates emission to a separate :class:`BuiltModel` produced by
:meth:`apeSees.build`.

Phase 0 shipped the skeleton (namespace stubs + register + tag
allocator). Phase 4 wires:

  * ``BuiltModel.emit`` drives the emitter end-to-end via the
    fan-out helpers in :mod:`apeGmsh.opensees._internal.build`.
  * Flat methods (``fix``, ``mass``, ``analyze``, ``tcl``, ``py``,
    ``run``) collect records / build a ``BuiltModel`` / pick the
    appropriate :mod:`apeGmsh.opensees.emitter` and drive it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence, TypeVar,
)

from ._internal.artifact_write import BridgeArtifactWriter
from ._internal.build import (
    BridgeError,
    DampingAttachRecord,
    FixRecord,
    InitialStressRecord,
    MassRecord,
    ModalDampingRecord,
    NdfRecord,
    RayleighRecord,
    RegionAssignmentRecord,
    ProfileRecord,
    SPRemovalRecord,
    StageRecord,
    SupportRecord,
    ELEMENT_TAG_MODES,
    ElementTagMode,
    _emit_node_with_inferred_ndf,
    bucket_primary_nodes_by_rank,
    build_element_partition_owner,
    build_node_partition_owners,
    close_builder_ndf_bracket,
    compute_stage_ownership,
    emit_activate_absorbing,
    emit_element_spec_partitioned,
    emit_initial_stress_addtoparameter,
    emit_initial_stress_global,
    EquationConstraintRecord,
    emit_equation_constraints,
    emit_mp_constraints,
    emit_mp_constraints_partitioned,
    make_equation_constraint_record,
    emit_reinforce_ties,
    emit_update_parameters,
    emit_embed_ties,
    emit_contacts,
    emit_contact_planes,
    PlannedContact,
    write_planned_contact,
    emit_interfaces,
    emit_rebar_elements,
    emit_zero_velocities,
    zero_velocity_target_nodes,
    emit_ghost_sp_ops,
    emit_stage_interfaces,
    allocate_interface_tags,
    _emit_interface_record,
    _plan_rank_interfaces,
    _register_interface_phantoms,
    _validate_interface_records,
    emit_stage_mp_constraints,
    emit_stage_mp_constraints_partitioned,
    plan_stage_mp_constraints_partitioned,
    emit_pattern_spec,
    emit_recorder_spec,
    emit_transform_specs,
    expand_pg_to_elements,
    expand_pg_to_nodes,
    ElementPlanRows,
    FemToOpsTagMap,
    LazyRankBuckets,
    NodePartitionOwners,
    SortedIntSet,
    SortedIntToInt,
    is_partitioned,
    node_index_lookup,
    open_builder_ndf_bracket,
    primary_owner_map,
    replay_builder_scoped_declarations,
    runtime_rank_from_partition_record,
    topological_order,
    needs_builder_ndf_bracket,
    validate_builder_scope_ordering,
    validate_node_ndf_element_compat,
    validate_absorbing_quad_geometry,
    validate_body_force_double_count,
    validate_from_model_cases,
    validate_load_basis_vs_elements,
    validate_model_definition_consumed,
    validate_ladruno_up_specs,
    validate_ladruno_up_pressure_dof,
    validate_ladruno_up_solver,
    validate_serial_mumps,
    deck_requests_solver_stats,
    validate_up_pressure_datum,
    validate_manzari_convergence_test,
    validate_manzari_tangent_solver,
    validate_sanisand_substep_cap,
    validate_asdplastic_host,
    infer_node_ndf,
    validate_adaptive_element_endpoints,
    resolve_ndf_overlay,
    validate_constraint_master_ndf,
    validate_diaphragm_master_stiffness,
    validate_record_ndf_consistency,
    fit_dof_vector,
    fit_fix_mask,
    fix_records_from_model,
    broker_mass_components,
    assert_ndm_compatible,
    TagPlanMiss,
)
from ._internal.build import _element_transf as _build_element_transf
from ._element_capabilities import is_builder_scoped
from ._internal.tag_resolution import (
    set_current_fem_element_id,
    set_element_nodes,
    set_stage_owned_node_tags,
)
from ._internal.compose import _compose_model_h5, _path_stem
from .._internal.provenance import ProvenanceStore
from ._internal.ns import (
    _AlgorithmNS,
    _AnalysisNS,
    _BeamIntegrationNS,
    _ConstraintsNS,
    _DampingNS,
    _ElementNS,
    _FaultNS,
    _GeomTransfNS,
    _IntegratorNS,
    _NDMaterialNS,
    _NumbererNS,
    _PatternNS,
    _ProfilerNS,
    _RecorderNS,
    _SectionNS,
    _StrategyNS,
    _SystemNS,
    _TestNS,
    _TimeSeriesNS,
    _UniaxialMaterialNS,
)
from ._internal.tag_allocator import TagAllocator
from ._internal.tag_plan import (
    TagMode,
    TagPlan,
    emit_mode,
    plan_tags,
)
from ._internal.tag_resolution import set_tag_resolver
from ._internal.types import (
    Analysis,
    BeamIntegration,
    ConstraintHandler,
    ConvergenceTest,
    Damping,
    Element,
    GeomTransf,
    Integrator,
    LinearSystem,
    NDMaterial,
    Numberer,
    Pattern,
    Primitive,
    Recorder,
    Section,
    SolutionAlgorithm,
    TimeSeries,
    UniaxialMaterial,
)
from ._run import resolve_log_path, run_label, stream_run
from .emitter.base import Emitter, StrategySpec
from .node import Node, _NodeAccessor, _iter_tags
from .recorder import FilterableRecorder, Ladruno
from .procedures._contact import _ContactQueryMixin
from .procedures._explicit import ExplicitRunResult, _ExplicitMixin
from .procedures._frf import _FrfMixin
from .procedures._modal import _ModalMixin
from .stage._builder import _StageBuilder, _build_initial_stress_record
from .transform import Cartesian, Orientation

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

    # FEMData is the only mesh symbol the bridge depends on (P3, P9).
    # Imported under TYPE_CHECKING so that constructing apeSees does
    # not transitively import gmsh during static analysis.
    # Use the fully-qualified module path to disambiguate from the
    # similarly-named submodule ``apeGmsh.mesh.FEMData`` under mypy.
    from apeGmsh.cuts import SectionCutDef, SectionSweepDef
    from apeGmsh.mesh.FEMData import FEMData
    from ._internal.build import StiffnessResolver
    from .analysis.strategy import Ladder
    from .emitter.tcl import PartitionSpan
    from ._target import OpenSeesCapabilities, OpenSeesTarget
    from ._solver_stats import RunSolverStats
    from .pattern.pattern import Plain
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.results.capture.spec import DomainCaptureSpec
    from apeGmsh.hpc import Cluster, Job

    from .emitter.live import LiveOpsEmitter
    from ._internal.tag_plan import NamedMembers, NamedRegion, RegionSite
    from apeGmsh._kernel.defs.decoupled import DecoupledNodeSetDef
    from ._internal.spring_bed import OrientValues, SpringBed, SpringValues


__all__ = ["apeSees", "BuiltModel", "ExplicitRunResult"]


class OpenSeesAutoEmitWarning(UserWarning):
    """Emitted by the bridge when an MP-aware default is auto-applied.

    Tagged subclass so tests can filter these without losing genuine
    ``UserWarning`` signal. Real users in interactive sessions still see
    the message — the only difference is pytest's default filter ignores
    them (see ``pyproject.toml`` ``[tool.pytest.ini_options]``). Covers
    the constraint-handler auto-emit and the Plain-handler footgun
    warning, plus the parallel-numberer / parallel-system auto-emit
    family in ``_maybe_auto_emit_*`` methods.
    """


class RayleighOverwriteWarning(UserWarning):
    """Emitted when a global ``rayleigh`` and a region-scoped ``rayleigh``
    (``ops.damping.rayleigh(on=...)``) coexist in the same model.

    OpenSees applies element Rayleigh by **overwrite**, not summation: a
    ``region -rayleigh`` replaces the global factors for the elements it
    owns (ADR 0053, verified against the fork reference). A user who expects
    the region damping to *add* to the global damping would be surprised, so
    the emit pass flags the combination. Tagged subclass so pytest's default
    filter ignores it (see ``pyproject.toml``) while interactive users still
    see the message.
    """


class OpenSeesExplicitSolverWarning(UserWarning):
    """Emitted when an explicit integrator is paired with a non-diagonal
    linear system.

    Such a pairing is *mathematically correct* but factors the full mass
    matrix every step, discarding the ``O(N)`` factorization-free advantage
    that is the whole point of explicit integration. Tagged subclass so it
    is filterable; the genuinely-wrong consistent-mass + ``Diagonal`` combo
    is a hard error (``BridgeError``), not this warning.
    """


# Sentinel for "argument not supplied" on ``default_orientation``.
# Using a sentinel (rather than ``None``) lets the user EXPLICITLY pass
# ``None`` to disable the auto-default (typical for 2D models, where
# vecxz is omitted at emit time and an orientation makes no sense),
# while a missing argument still produces the convenience Z-up default
# Cartesian orientation for 3D frame work.
class _UnsetType:
    """Sentinel type for unset constructor arguments."""
    __slots__ = ()
    def __repr__(self) -> str:
        return "<UNSET>"

_UNSET: "_UnsetType" = _UnsetType()


# Bound to Primitive so namespace methods preserve the concrete type:
#   def Steel02(self, ...) -> Steel02:
#       return self._bridge._register(Steel02(...))
_P = TypeVar("_P", bound=Primitive)


# ---------------------------------------------------------------------------
# Tag-allocation kind dispatch
# ---------------------------------------------------------------------------

_KIND_BY_FAMILY: tuple[tuple[type[Primitive], str], ...] = (
    (UniaxialMaterial, "uniaxialMaterial"),
    (NDMaterial,       "nDMaterial"),
    (Section,          "section"),
    (GeomTransf,       "geomTransf"),
    (BeamIntegration,  "beamIntegration"),
    (TimeSeries,       "timeSeries"),
    (Pattern,          "pattern"),
    (Element,          "element"),
    (Damping,          "damping"),
    (Recorder,         "recorder"),
    (ConstraintHandler, "constraints"),
    (Numberer,         "numberer"),
    (LinearSystem,     "system"),
    (ConvergenceTest,  "test"),
    (SolutionAlgorithm, "algorithm"),
    (Integrator,       "integrator"),
    (Analysis,         "analysis"),
)


# Analysis-chain Primitive base types (Phase SSI-2.A) — used by the
# staged-emit path to filter chain primitives out of the global
# pre-element emit (each stage emits its own chain).  Mirrors the
# tuple in :meth:`apeSees._check_analysis_chain_for_analyze`.
_ANALYSIS_CHAIN_BASES: tuple[type[Primitive], ...] = (
    ConstraintHandler,
    Numberer,
    LinearSystem,
    ConvergenceTest,
    SolutionAlgorithm,
    Integrator,
    Analysis,
)



def _is_analysis_chain_primitive(prim: Primitive) -> bool:
    """True iff ``prim`` is one of the seven analysis-chain types."""
    return isinstance(prim, _ANALYSIS_CHAIN_BASES)


def _kind_of(prim: Primitive) -> str:
    """Return the tag-allocator kind string for ``prim``."""
    for base, kind in _KIND_BY_FAMILY:
        if isinstance(prim, base):
            return kind
    raise TypeError(
        f"Primitive {type(prim).__name__} does not inherit from any "
        f"recognized family base (UniaxialMaterial, Section, ...)."
    )


def _planned_element_specs(
    tag_plan: TagPlan, elements: "list[Element]",
) -> "list[tuple[Element, ElementPlanRows]]":
    """The element plan this emit reads, from ``tag_plan``.

    ADR 0114 D4 (amended): element tags are allocated once, by
    ``plan_tags``; the emit paths read them here instead of minting.
    The plan must cover exactly ``elements``, the specs this emit fans
    out, in order; anything else raises :class:`TagPlanMiss`.
    """
    specs = tag_plan.elements.specs
    if [id(s) for s, _ in specs] != [id(e) for e in elements]:
        raise TagPlanMiss(
            f"the element plan holds {len(specs)} specs, but this emit fans "
            f"out {len(elements)}, or in another order: the plan was not "
            "made for this model (ADR 0114 D4, amended)."
        )
    return list(specs)


def _fem_has_contacts(fem: "FEMData") -> bool:
    """True iff the FEM snapshot carries any fork contact interactions that need
    the ``LadrunoContact`` handler — face-to-face (``g.constraints.contact`` →
    ``fem.elements.contacts``) OR rigid-plane (``g.constraints.contact_plane``
    → ``fem.elements.contact_planes``)."""
    elements = getattr(fem, "elements", None)
    if elements is None:
        return False
    return bool(getattr(elements, "contacts", None)
                or getattr(elements, "contact_planes", None))


def _fem_has_interface_equal_dofs(fem: "FEMData") -> bool:
    """True iff any ``g.constraints.interface()`` record carries a nested
    ``equalDOF`` (ADR 0093 D4 — the mixed-ndf phantom bridge).

    Interfaces themselves are handler-independent: a ``zeroLength`` plus
    two uniaxials needs no constraint handler at all. But a **mixed-ndf**
    pair additionally emits an ordinary identity
    ``equalDOF(beam_node, phantom, 1, 2)`` from ``emit_interfaces`` —
    outside ``fem.nodes.constraints``, so neither
    :func:`_fem_has_mp_constraints` nor
    :func:`_fem_has_handler_requiring_mp` could see it. Both consult this
    predicate so a mixed-ndf interface model is treated exactly like any
    other equalDOF-carrying model (ADR 0093 D5's "handler-dependent in
    the ordinary way").

    Equal-ndf interface models return False — they really are
    handler-independent.
    """
    elements = getattr(fem, "elements", None)
    interfaces = (
        getattr(elements, "interfaces", None) if elements is not None else None
    )
    if not interfaces:
        return False
    try:
        for rec in interfaces:
            if getattr(rec, "equal_dof_records", None):
                return True
    except TypeError:
        pass
    return False


def _fem_has_mp_constraints(fem: "FEMData") -> bool:
    """True iff the FEM snapshot carries any MP-constraint records.

    Used by the Phase 8 Transformation auto-emit (the fold-in to
    address the Phase 7b footgun where the default ``Plain`` handler
    silently ignores MP constraints).  Returns True when any of:

    * ``fem.nodes.constraints`` carries ANY records (``equal_dof``,
      ``rigid_beam`` / ``rigid_rod`` / ``rigid_body`` /
      ``rigid_diaphragm`` / ``kinematic_coupling`` /
      ``node_to_surface``).
    * ``fem.elements.constraints.interpolations()`` yields any
      records (``tie`` / ``distributing`` / ``tied_contact`` /
      ``mortar`` / ``embedded``).
    * ``fem.elements.interfaces`` carries a mixed-ndf pair, whose
      nested phantom-bridge ``equalDOF`` (ADR 0093 D4) is emitted by
      ``emit_interfaces`` and lives on neither composite above.

    Returns False when neither composite exists or both are empty
    (defensive on test stubs that don't carry the full broker shape).
    """
    if _fem_has_interface_equal_dofs(fem):
        return True

    nodes = getattr(fem, "nodes", None)
    node_constraints = (
        getattr(nodes, "constraints", None) if nodes is not None else None
    )
    if node_constraints is not None:
        try:
            for _rec in node_constraints:
                return True
        except TypeError:
            # Not iterable — defensive on stubs without __iter__.
            pass

    elements = getattr(fem, "elements", None)
    surface_constraints = (
        getattr(elements, "constraints", None)
        if elements is not None
        else None
    )
    if surface_constraints is not None:
        interps = getattr(surface_constraints, "interpolations", None)
        if interps is not None:
            try:
                for _rec in interps():
                    return True
            except TypeError:
                pass
    return False


def _fem_has_equation_ties(fem: "FEMData") -> bool:
    """True iff the FEM snapshot carries any ``enforce="equation"`` tie
    (ADR 0068).  Such ties emit ``equationConstraint`` (EQ_Constraint),
    which the ``Transformation`` handler CANNOT enforce — so the handler
    auto-emit must upgrade to ``Lagrange`` (implicit) / ``LadrunoProjection``
    (explicit) rather than silently emitting a deck that drops them
    (INV-4).  Defensive on stubs that don't carry the full broker shape.
    """
    elements = getattr(fem, "elements", None)
    surface_constraints = (
        getattr(elements, "constraints", None)
        if elements is not None
        else None
    )
    if surface_constraints is None:
        return False
    interps = getattr(surface_constraints, "interpolations", None)
    if interps is None:
        return False
    try:
        for rec in interps():
            if getattr(rec, "enforce", "penalty") == "equation":
                return True
    except TypeError:
        pass
    return False


def _fem_has_handler_requiring_mp(fem: "FEMData") -> bool:
    """True iff the FEM carries a constraint emitted as a true OpenSees
    ``MP_Constraint`` (``equalDOF`` / ``equalDOF_mixed`` / ``rigidLink`` /
    ``rigidDiaphragm``) that a Plain-style handler (``LadrunoContact``, fork
    P1a) cannot enforce.

    Used by the contact handler-conflict guard (#7) — distinct from the
    broader :func:`_fem_has_mp_constraints` (which also counts penalty-element
    interpolation ties). Node-side records emit MP_Constraints EXCEPT
    ``kinematic_coupling`` (→ ``LadrunoKinematicCoupling`` element) and
    ``penalty`` (g.constraints.penalty → a stiff spring element, NOT an
    MP_Constraint — it has no MP emit path in build.py), which are
    handler-independent. The interpolation ties (``tie`` / ``embedded`` /
    ``distributing``) all emit penalty/coupling ELEMENTS — handler-independent
    too; their only handler-requiring route is ``enforce="equation"``, caught
    separately by :func:`_fem_has_equation_ties`. So only the node side, minus
    those element-emitting kinds, requires a handler. Unknown node kinds
    default to handler-requiring (conservative — better a false fail-loud than
    a silently-unenforced MP constraint).

    ``rigid_body(as_element=True)`` is NOT handler-independent, despite
    emitting a ``LadrunoRigidBody`` element: ``LadrunoRigidBody::setDomain``
    adds one ``MP_Constraint`` per slave (fork
    ``LadrunoRigidBody.cpp:339,360-362``), and ``LadrunoContactHandler`` only
    WARNS about MP constraints — it does not enforce them (fork
    ``LadrunoContactHandler.cpp:706-713``). So contact + rigid_body
    (as_element=True) would silently leave those slaves unconstrained; this
    counts as handler-requiring like the ``as_element=False`` form.

    ADR 0093: a **mixed-ndf** ``g.constraints.interface()`` pair emits a
    real ``equalDOF`` from its own pass, so it counts here too —
    ``LadrunoContactHandler::handle`` warns and moves on without enforcing
    any ``MP_Constraint`` (fork ``LadrunoContactHandler.cpp:340-347``), so
    contact + a phantom-bridged interface would silently leave the phantom
    free. Equal-ndf interfaces emit no MP constraint and stay compatible.
    """
    from apeGmsh._kernel.records._kinds import ConstraintKind

    if _fem_has_interface_equal_dofs(fem):
        return True

    _HANDLER_INDEPENDENT = {
        ConstraintKind.KINEMATIC_COUPLING,   # → LadrunoKinematicCoupling element
        ConstraintKind.PENALTY,              # → stiff spring element
    }

    nodes = getattr(fem, "nodes", None)
    node_constraints = (
        getattr(nodes, "constraints", None) if nodes is not None else None
    )
    if node_constraints is None:
        return False
    try:
        for rec in node_constraints:
            kind = getattr(rec, "kind", None)
            if kind in _HANDLER_INDEPENDENT:
                continue
            return True
    except TypeError:
        # Not iterable — defensive on stubs without __iter__.
        pass
    return False


def _records_have_equation_tie(records: "Iterable[Any] | None") -> bool:
    """True iff ``records`` (a stage's ``stage_constraint_records``) carries
    any ``enforce="equation"`` tie — directly as an ``InterpolationRecord``
    or nested in a ``SurfaceCouplingRecord`` (``tied_contact``).  Used by the
    staged-path EQ handler guard (ADR 0068 Open item 5)."""
    from apeGmsh._kernel.records._constraints import (
        InterpolationRecord, SurfaceCouplingRecord,
    )
    for r in records or ():
        if isinstance(r, InterpolationRecord) and \
                getattr(r, "enforce", "penalty") == "equation":
            return True
        if isinstance(r, SurfaceCouplingRecord):
            for ir in getattr(r, "slave_records", ()) or ():
                if getattr(ir, "enforce", "penalty") == "equation":
                    return True
    return False


def _is_explicit_integrator(integrator: "Primitive | None") -> bool:
    """True iff ``integrator`` is an explicit time integrator.

    Single source of truth shared by the EQ-tie handler auto-detect
    (:meth:`BuiltModel._maybe_auto_emit_constraint_handler`, ADR 0068 Open
    item 1: explicit ⇒ auto-emit ``LadrunoProjection``, implicit ⇒
    ``Lagrange``) and the explicit-solver compat guard
    (:meth:`apeSees._check_explicit_solver_compat`).  ``None`` (no integrator
    registered yet) ⇒ ``False`` (treat as implicit).
    """
    if integrator is None:
        return False
    from .analysis.integrator import (
        CentralDifference,
        CentralDifferenceLadruno,
        ExplicitBathe,
        ExplicitBatheLNVD,
        ExplicitDifference,
    )
    return isinstance(integrator, (
        CentralDifference,
        ExplicitDifference,
        ExplicitBathe,
        ExplicitBatheLNVD,
        CentralDifferenceLadruno,
    ))


# ---------------------------------------------------------------------------
# Partition-aware MPCO recorder plan (ADR 0027 INV-4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _RegionEmit:
    """One auto-emitted ``region`` (a filter region or a Ladruno energy
    region) plus its resolved members, for the partitioned per-rank pass.

    ``elem_ids`` carries **FEM eids** (not OpenSees element tags): the
    per-rank intersection in
    :meth:`BuiltModel._emit_mpco_filter_regions_for_rank` is keyed by
    ``element_owner`` which is FEM-eid keyed; the translation to OpenSees
    tags happens at the final region-emit step on each rank. A filter
    region carries node + element members; an energy region carries
    elements only (the fork auto-derives its nodes).
    """
    tag: int
    node_ids: tuple[int, ...]
    elem_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _MPCOFilterPlan:
    """Pre-resolved regions for one filter-bearing / energy-region recorder.

    Built once before the partitioned per-rank loop in
    :meth:`BuiltModel._emit_partitioned`; consumed by
    :meth:`BuiltModel._emit_mpco_filter_regions_for_rank` (inside each
    rank block) and by the global recorder emit pass after the loop.

    The ``materialised_spec`` carries ``_region_tag`` (and, for a Ladruno
    with ``energy_pg``, ``_energy_region_tags``) populated and the ``*_pg``
    selectors cleared, so the spec's ``_emit`` appends ``-R <tag>`` /
    ``-G energy <tag...>`` without re-entering ``materialize`` (which would
    otherwise allocate fresh tags and re-emit regions globally — the buggy
    pre-ADR-0027 INV-4 behaviour for the partitioned path). The spec is any
    :class:`recorder.FilterableRecorder` (MPCO or Ladruno, ADR 0064).

    ``regions`` holds every region the recorder needs — the value-channel
    filter region (if any) followed by Ladruno energy regions (if any) —
    each emitted per-rank with its owned-member intersection.
    """
    materialised_spec: "FilterableRecorder"
    regions: tuple["_RegionEmit", ...]


def _write_per_rank_tcl(
    path: str, lines: "list[str]", spans: "list[PartitionSpan]",
) -> None:
    """Write a Tcl driver at ``path`` + ``ranks/rank<K>_<seq>.tcl``
    fragments (ADR 0061).

    Each recorded ``if {[getPID] == K} { ... }`` block body moves to a
    rank-local fragment; the driver keeps everything global and
    sequential (materials, analysis chains, stage skeleton) and
    replaces the block with a one-line guard that ``source``s the
    fragment — so at runtime each rank parses only the driver plus its
    own fragments (a brace-quoted ``if`` body is never evaluated on
    non-matching ranks, and ``source`` reads the file only when
    executed). A rank gets one fragment per block it appears in: its
    base-model block plus one per stage (``<seq>`` is the rank's
    0-based block counter).
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    ranks_dir = os.path.join(out_dir, "ranks")
    os.makedirs(ranks_dir, exist_ok=True)

    seq: dict[int, int] = {}
    driver: list[str] = []
    cursor = 0
    for span in spans:
        driver.extend(lines[cursor:span.header])
        n = seq.get(span.rank, 0)
        seq[span.rank] = n + 1
        fname = f"rank{span.rank}_{n}.tcl"
        # Body lines carry the block's one-level (4-space) indent —
        # strip it; lines without the prefix pass through unchanged.
        body = [
            ln.removeprefix("    ")
            for ln in lines[span.body_start:span.body_end]
        ]
        with open(
            os.path.join(ranks_dir, fname), "w", encoding="utf-8",
        ) as f:
            f.write(
                f"# apeGmsh per-rank fragment (ADR 0061): "
                f"rank {span.rank}, block {n}\n"
            )
            if body:
                f.writelines(ln + "\n" for ln in body)
        driver.append(
            f"if {{[getPID] == {span.rank}}} {{ source [file join "
            f"[file dirname [info script]] ranks {fname}] }}"
        )
        driver.append("")
        cursor = span.end
    driver.extend(lines[cursor:])
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(ln + "\n" for ln in driver)



# ---------------------------------------------------------------------------
# BuiltModel — the immutable read-only artifact emitters consume
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuiltModel:
    """Immutable snapshot of declared primitives + tag assignments.

    Drives a frozen :class:`~apeGmsh.opensees.emitter.base.Emitter` via
    :meth:`emit`, dispatching to the per-family fan-out helpers in
    :mod:`apeGmsh.opensees._internal.build`.

    Attributes
    ----------
    primitives
        Tuple of registered primitives in registration order. The
        emit-order topological sort happens inside :meth:`emit`.
    tag_for
        ``id(primitive) -> bridge-allocated tag``.
    ndm, ndf
        Model dimensionality (set via ``apeSees.model``).
    fem
        The FEM snapshot the bridge was built against. Required for
        physical-group fan-out at emit time. Stored on the build
        because the build is the only thing emitters see.
    fix_records, mass_records
        Model-level constraint and mass directives collected through
        ``apeSees.fix`` / ``apeSees.mass``.
    region_records
        Named-region assignments collected through ``apeSees.region``
        (and the ``Node.region`` / ``NodeSet.region`` shortcuts).
        Multiple records sharing the same ``name`` are merged at emit
        time into a single ``region $tag -node ...`` command with one
        freshly-allocated tag.
    initial_stress_records
        :class:`~apeGmsh.opensees._internal.build.InitialStressRecord`
        directives collected through ``apeSees.initial_stress(...)``
        — each is a ramped in-situ stress tensor that fans out into
        parameter declarations + per-rank ``addToParameter`` calls +
        a step-hook ramping proc.  Phase SSI-1.
    """

    primitives:              tuple[Primitive, ...]
    tag_for:                 dict[int, int]
    ndm:                     int
    ndf:                     int
    fem:                     "FEMData"
    fix_records:             tuple[FixRecord, ...]
    mass_records:            tuple[MassRecord, ...]
    region_records:          tuple[RegionAssignmentRecord, ...]
    # ADR 0049 — ``ops.ndf`` directives (the sole explicit per-node ndf
    # channel; element-less decoupled nodes only). Resolved at emit time into
    # the overlay merged over the inferred map (``resolve_ndf_overlay``).
    ndf_records:             tuple[NdfRecord, ...] = ()
    initial_stress_records:  tuple[InitialStressRecord, ...] = ()
    stage_records:           tuple[StageRecord, ...] = ()
    rayleigh_records:        tuple[RayleighRecord, ...] = ()
    damping_attach_records:  tuple[DampingAttachRecord, ...] = ()
    modal_damping_records:   tuple[ModalDampingRecord, ...] = ()
    # User ``equationConstraint`` rows (``apeSees.equation_constraint``),
    # emitted in the MP-constraint pass after the broker's constraints.
    equation_constraint_records: tuple[EquationConstraintRecord, ...] = ()
    # name → bridge-allocated tag, for resolving g.reinforce bond-material
    # references (Option B: the def holds the bond name, the bridge owns
    # the tag). Populated from the name-alias table at build() time.
    name_to_tag:             dict[str, int] = field(default_factory=dict)
    # ADR 0065 Tier 2 — stream per-node masses straight from
    # ``fem.nodes.masses`` at emit instead of materializing one bridge
    # ``MassRecord`` per node (the 7M-object double-store). Set by
    # ``apeSees.mass_from_model()``; consumed in the mass emit paths.
    mass_from_model:         bool = False
    # ADR 0111 D2 — ``"sequential"`` (default: element tags 1, 2, ... in
    # declaration order) or ``"fem"`` (each physical-group element keeps
    # its FEM element id as its OpenSees tag; see ``emit``).
    element_tags:            ElementTagMode = "sequential"
    # ADR 0062 — per-build cache of resolved moment-tensor ``(node, force)``
    # pairs, keyed by ``id(rec)``. The host search in
    # ``resolve_moment_tensor_pairs`` runs against the full FEM snapshot and is
    # rank-independent, but ``_owned_moment_tensor_lines`` is invoked per-rank
    # (and a second time per-rank by the staged pre-check) — without this cache
    # the O(N_elements) search re-runs ~2·np times (the multi-hour emit wall).
    # Fresh per build and mutated in place (frozen-safe: we mutate the dict, not
    # rebind the field), so ``id(rec)`` reuse across builds is not a hazard.
    _mt_pairs_cache:         "dict[int, list[tuple[int, Any]]]" = field(
        default_factory=dict, compare=False)
    # ADR 0114 D4 (amended) — the build-time tag plan, one per emit mode,
    # made by ``plan_tags`` on the first ``emit`` in that mode and reused
    # by every later one. ``init=False``, so ``dataclasses.replace`` gives
    # the copy a fresh memo; ``copy.copy`` still shares it, so
    # ``_tag_plan`` also checks that the memoised plan was made from this
    # model's inputs (``TagPlan.planned_for``) and re-plans if not.
    _tag_plans:              "dict[TagMode, TagPlan]" = field(
        default_factory=dict, init=False, compare=False, repr=False)

    def _tag_plan(self, mode: "TagMode") -> "TagPlan":
        """The memoised :class:`TagPlan` of emit mode ``mode`` for this model."""
        plan = self._tag_plans.get(mode)
        if plan is None or not plan.planned_for(self):
            plan = plan_tags(self, mode)
            self._tag_plans[mode] = plan
        return plan

    def _has_equation_constraints(self) -> bool:
        """True iff the deck carries any ``equationConstraint`` row.

        Either an ``enforce="equation"`` tie on the snapshot (ADR 0068) or a
        user row from ``apeSees.equation_constraint``. Both are
        ``EQ_Constraint``, which ``Transformation`` / ``Auto`` / ``Plain``
        silently drop (INV-4), so every handler guard keys on this.
        """
        return (
            bool(self.equation_constraint_records)
            or _fem_has_equation_ties(self.fem)
        )

    def _claimed_recorder_ids(self) -> "set[int]":
        """``id(...)``-set of recorders claimed by stage builders
        via ``s.recorder(spec)`` (Phase SSI-2.D PR-C).

        Recorders in this set stay in ``self.primitives`` so their
        allocated tag remains available via ``tag_for[id(p)]``, but
        the global post-element recorder emit loop SKIPS them — each
        is emitted instead inside its owning stage's block, after the
        stage's regions and analysis chain, so the recorder line is
        parsed by OpenSees AFTER the stage-bound regions it may
        reference have been declared (recorder member lists cache at
        parse time — TclRecorderCommands.cpp:276, 1331+).
        """
        return {
            id(r)
            for stage in self.stage_records
            for r in stage.recorder_specs
        }

    def _claimed_constraint_ids(self) -> "set[int]":
        """``id(...)``-set of resolved constraint records claimed by
        stage builders via ``s.embedded`` / ``s.equal_dof`` /
        ``s.rigid_link`` / ``s.tie`` / ``s.tied_contact`` /
        ``s.kinematic_coupling`` / ``s.node_to_surface``.

        Records in this set stay on the FEMData broker (so the broker
        view is unmodified across multiple bridges sharing the same
        FEMData), but the global MP-constraint emit pass SKIPS them
        — each is emitted instead inside its owning stage's block,
        AFTER the stage's regions and BEFORE the stage's
        ``domain_change``, so the constrained nodes / elements (which
        emitted at the top of the stage block) are already in the
        OpenSees domain.

        Surface couplings (``s.tied_contact`` → ``SurfaceCouplingRecord``)
        need their nested ``slave_records`` ids added too: the global
        surface-coupling pass consumes ``constraints.interpolations()``,
        which EXPANDS each ``SurfaceCouplingRecord`` into its per-slave
        ``InterpolationRecord`` rows, and the exclusion filter
        (:class:`_ExcludeClaimedConstraints`) matches on those expanded
        slave ids — not the outer record's.  Claiming only the outer id
        would leave the slaves emitting BOTH globally (at ``t = 0``) and
        inside the stage.  The stage adapter expands the same slave
        objects from the claimed outer record, so the in-stage emit is
        unaffected (ADR 0034 follow-up).
        """
        ids: set[int] = set()
        for stage in self.stage_records:
            for r in stage.stage_constraint_records:
                ids.add(id(r))
                slaves = getattr(r, "slave_records", None)
                if slaves:
                    for slave in slaves:
                        ids.add(id(slave))
        return ids

    def _claimed_interface_ids(self) -> "set[int]":
        """``id(...)``-set of ``InterfaceRecord`` rows claimed by stage
        builders via ``s.interface(name=)`` (ADR 0093 S7 / INV-6).

        The records stay on ``fem.elements.interfaces`` (the broker is
        immutable from the bridge's perspective, and every predicate
        that must still count them — ``_fem_has_interface_equal_dofs``
        among others — reads that side-list), but the base
        ``emit_interfaces`` pass SKIPS them: each emits inside its
        owning stage's block — via ``emit_stage_interfaces`` on the
        flat path, via the per-stage owner-rank plan
        (:meth:`_plan_stage_interfaces_partitioned`, ADR 0093 S9) under
        MP — on the equilibrated ground.  Mirrors
        :meth:`_claimed_constraint_ids`;
        no nested-record expansion is needed (an interface record's
        nested ``equal_dof_records`` are emitted by the interface pass
        itself, never by the MP pass).
        """
        return {
            id(r)
            for stage in self.stage_records
            for r in stage.stage_interface_records
        }

    def _claimed_pattern_ids(self) -> "set[int]":
        """``id(...)``-set of load patterns claimed by stage builders
        via ``s.pattern(series=)`` (ADR 0051 BL-3).

        Stage-scoped patterns stay in ``self.primitives`` so their
        allocated tag remains available via ``tag_for[id(p)]``, but
        the global post-element pattern emit loop SKIPS them — each is
        emitted instead inside its owning stage's block, after the
        stage's analysis chain and before ``analyze``, so the pattern's
        loads are frozen by that stage's ``stage_close`` ``loadConst``.
        Mirrors :meth:`_claimed_recorder_ids` /
        :meth:`_claimed_constraint_ids`.
        """
        ids = {
            id(p)
            for stage in self.stage_records
            for p in stage.pattern_specs
        }
        # ADR 0052: the per-stage dedicated HOLD pattern (``s.support``)
        # is stage-scoped too — it emits via the dedicated HOLD block in
        # its stage, never the global / 7b pattern pass, and must not
        # trip the two-mode no-mixing guard.
        for stage in self.stage_records:
            if stage.support_pattern is not None:
                ids.add(id(stage.support_pattern))
        return ids

    def _auto_stiffness_resolver(self) -> "StiffnessResolver":
        """Resolver for ``stiffness="auto"`` tie records (slice B).

        Built from the declared element specs + the broker
        (``K = α·E_host·L_char`` — see
        :func:`make_auto_stiffness_resolver`).  The returned resolver
        initialises its node→E / node→xyz maps lazily on the first
        ``"auto"`` record it sees, so handing one to an emit pass with
        no auto records costs nothing (BuiltModel is frozen+slots, so
        no instance cache — each call returns a fresh lazy resolver).
        """
        from ._internal.build import make_auto_stiffness_resolver
        elements = [p for p in self.primitives if isinstance(p, Element)]
        return make_auto_stiffness_resolver(self.fem, elements)

    # -- ADR 0051 §5 — two-mode no-mixing guard (BL-4) ------------------

    def _validate_two_mode_no_mixing(self) -> None:
        """ADR 0051 §5: a staged model may not also carry a **global**
        load pattern.

        A model is either **non-staged** (a global ``ops.pattern.*`` +
        the analysis chain + ``ops.analyze`` / ``ops.eigen``) **or**
        **staged** (every pattern stage-scoped via ``s.pattern(...)``,
        run through ``ops.tcl`` / ``ops.py``).  Mixing the two is a
        hard error: a global pattern fires in every stage's analyze
        loop (ADR 0031), silently double-applying its loads across the
        staged ``loadConst`` boundaries.

        A "global" pattern is any :class:`Pattern` registered directly
        on the bridge whose id is NOT claimed by a stage via
        ``s.pattern(...)`` (those live in ``_claimed_pattern_ids``).
        """
        if not self.stage_records:
            return
        claimed = self._claimed_pattern_ids()
        globals_ = [
            p for p in self.primitives
            if isinstance(p, Pattern) and id(p) not in claimed
        ]
        if not globals_:
            return
        kinds = ", ".join(sorted({type(p).__name__ for p in globals_}))
        stage_names = ", ".join(repr(s.name) for s in self.stage_records)
        raise BridgeError(
            f"apeSees: cannot mix a global ops.pattern.* registration "
            f"({kinds}) with staged analysis (stages: {stage_names}). "
            "Per ADR 0051 §5 every pattern in a staged model must "
            "be stage-scoped: create it inside the stage via "
            "s.pattern(series=...), not ops.pattern.Plain(...). A model "
            "is either non-staged (global pattern + ops.analyze) OR "
            "staged (per-stage patterns) — never both."
        )

    def emit(self, emitter: Emitter) -> int:
        """Drive ``emitter`` over the model, returning ``analyze``'s exit value.

        Returns ``0`` if no ``analyze`` was registered (the bridge's
        ``apeSees.analyze`` would have populated one); otherwise the
        last ``analyze`` call's return value.

        Topological order rules:
          1. Materials & sections & time series & transforms come
             before elements & patterns & recorders & analysis chain.
          2. Within the topo order: orientation-bearing transforms
             perform a one-shot fan-out across the elements that
             reference them (ADR 0010), producing a per-element
             override map.
          3. Element specs fan out across their physical groups,
             allocating one element tag per element instance.
          4. Pattern / recorder specs resolve ``pg=`` records into
             per-node / per-element calls.
        """
        # Tag resolver: returns the bridge-allocated tag for any
        # primitive in self.primitives. Fan-out helpers may install
        # short-lived element-specific resolvers on top of this; they
        # restore this base resolver before returning.
        def _base_resolver(p: Primitive) -> int:
            try:
                return self.tag_for[id(p)]
            except KeyError as e:
                raise BridgeError(
                    f"primitive {type(p).__name__}({p!r}) is referenced "
                    "as a dependency but was not registered with the "
                    "bridge. Per P11, register all standalone "
                    "primitives via ops.register(prim) before build()."
                ) from e

        set_tag_resolver(emitter, _base_resolver)

        # 1. Model directive.
        emitter.model(ndm=self.ndm, ndf=self.ndf)

        # 2. Topo-sort all registered primitives (and their dependencies).
        ordered = topological_order(self.primitives)

        # 2a. Reachability check (Option A in the Phase-4 spec): every
        # primitive returned by topo sort must itself be registered.
        # The topological_order function walks reachable-from-registered;
        # if it surfaces a primitive whose id is not in self.tag_for,
        # the user constructed a dependency standalone but never
        # registered it.
        for p in ordered:
            if id(p) not in self.tag_for:
                raise BridgeError(
                    f"primitive {type(p).__name__} is reachable through "
                    "another primitive's dependencies() but was never "
                    "registered. Per P11, register all standalone "
                    "primitives via ops.register(prim) before build()."
                )

        # 2b. Validate ``initial_stress`` record names are unique across
        # all stages + the global pool (red-team H2).  Duplicate names
        # would produce two ``proc <name> {...}`` definitions in Tcl:
        # the second overwrites the first, but each was built with
        # different parameter tags + cumulative-state keys, so the
        # surviving proc would reference an uninitialised
        # ``${name}_state(cum_<tag>)`` array element and crash at the
        # first analyze step of the later stage.  Fail loudly at build
        # time instead.
        name_to_owner: dict[str, str] = {}
        for rec in self.initial_stress_records:
            if rec.name in name_to_owner:
                raise BridgeError(
                    f"initial_stress name {rec.name!r} is registered "
                    f"twice (both on {name_to_owner[rec.name]!r}); "
                    "names must be unique across the global pool and "
                    "every stage's records."
                )
            name_to_owner[rec.name] = "global pool"
        for stage in self.stage_records:
            for rec in stage.initial_stress_records:
                if rec.name in name_to_owner:
                    raise BridgeError(
                        f"initial_stress name {rec.name!r} is registered "
                        f"twice (both on {name_to_owner[rec.name]!r} "
                        f"and on stage {stage.name!r}); names must be "
                        "unique across the global pool and every "
                        "stage's records."
                    )
                name_to_owner[rec.name] = f"stage {stage.name!r}"

        # 2c. ADR 0051 (BL-4): two-mode no-mixing guard.  Runs on every
        # emit path (flat / partitioned) before any primitive is
        # emitted — a staged model may not also carry a global pattern.
        self._validate_two_mode_no_mixing()

        # 3. Pre-bin: separate transforms, elements, the rest.
        transforms: list[GeomTransf] = []
        elements:   list[Element]    = []
        rest:       list[Primitive]  = []
        for p in ordered:
            if isinstance(p, GeomTransf):
                transforms.append(p)
            elif isinstance(p, Element):
                elements.append(p)
            else:
                rest.append(p)

        # Fail loud on the shell-on-solid node-sharing trap: a node
        # shared by two elements with disjoint per-node ndf (shell 6 vs
        # solid 3) corrupts OpenSees assembly (FE_Element::setID
        # truncation) and silently loses load.  Runs once here so every
        # emit path (flat / partitioned) is covered before any
        # element is emitted.
        validate_node_ndf_element_compat(self.fem, elements)

        # ADR 0054 (AB-5): ASDAbsorbingBoundary2D has no source-side
        # distortion handling — a skewed quad runs with silently wrong
        # dashpot/stiffness terms.  Fail loud here, once, on every emit
        # path (flat / partitioned).
        validate_absorbing_quad_geometry(self.fem, elements)

        # ADR 0074 (D3 + legality): LadrunoUP shape/perm-dim/stab-on-TH
        # coherence + the Bézier straight-side pre-check (the fork
        # deactivates a curved element at setDomain with only a cryptic
        # analyze() failure to show for it).  Runs once, every emit path.
        validate_ladruno_up_specs(self.fem, elements, self.ndm)

        # ADR 0074: rotation-vs-pressure DOF aliasing — a 2D frame element
        # sharing a saturated equal-order carrier node collides its rotation
        # DOF with the pore-pressure slot (both ndf=ndm+1), passing the
        # count-based gate. Fail loud with the ADR-0069 separate-node fix.
        validate_ladruno_up_pressure_dof(self.fem, elements, self.ndm)

        # ADR 0074 (D4): a LadrunoUP deck that WILL SOLVE must use a general
        # solver — the no-`system` default ProfileSPD silently drops one u-p
        # coupling block (symmetric storage) and returns plausible garbage.
        # Scope the gate to emits that actually drive a u-p solve: skip H5
        # archival and any emit with no analysis chain (model-only export,
        # eigen-only). Staged decks validate each stage's own system
        # (wipeAnalysis re-defaults each stage to ProfileSPD); a globally-
        # registered system is never emitted in staged mode, so it is not
        # counted. A partitioned flat deck with no system rides the ADR-0027
        # INV-5 auto-emit (general Mumps/UmfPack), so it is allowed.
        _staged = bool(self.stage_records)
        _emitter_is_archival = type(emitter).__name__ == "H5Emitter"
        _has_analysis_chain = _staged or any(
            isinstance(p, Analysis) for p in ordered
        )
        _will_partition = is_partitioned(self.fem) and getattr(
            emitter, "supports_partitions", True,
        )
        validate_ladruno_up_solver(
            elements,
            enforce=_has_analysis_chain and not _emitter_is_archival,
            staged=_staged,
            partitioned=_will_partition,
            flat_systems=[p for p in ordered if isinstance(p, LinearSystem)],
            stage_systems=[
                (repr(st.name), st.system) for st in self.stage_records
            ],
        )

        # ADR 0106 D5: an explicit `system Mumps` on a serial (non-
        # partitioned) deck is refused at build time. The fork's desktop
        # targets never compile the serial MumpsSolver, so it would
        # silently answer "unknown system type" and leave the run on
        # whatever SOE was already in place instead of stopping. Gates on
        # the FEM's partition count (D5's `len(fem.partitions) <= 1`), not
        # on `_will_partition`: a partitioned mesh emitted flat carries
        # the parallel chain by design (the ADR 0027 twin decks).
        validate_serial_mumps(
            enforce=_has_analysis_chain and not _emitter_is_archival,
            staged=_staged,
            partitioned=is_partitioned(self.fem),
            flat_systems=[p for p in ordered if isinstance(p, LinearSystem)],
            stage_systems=[
                (repr(st.name), st.system) for st in self.stage_records
            ],
        )

        # A Manzari-family material with the CONSISTENT tangent
        # (tan_type != 0 — now the LadrunoSANISAND default, and the fork
        # parser's own default since PR #792) has a genuinely UNSYMMETRIC
        # tangent.  Same physics as the u-p gate above, same scope rules,
        # but fail-SOFT: the deck runs, it is the answer that is not
        # trustworthy.  Reuses the same _staged / enforce / partitioned
        # facts computed for D4.
        validate_manzari_tangent_solver(
            ordered,
            enforce=_has_analysis_chain and not _emitter_is_archival,
            staged=_staged,
            partitioned=_will_partition,
            flat_systems=[p for p in ordered if isinstance(p, LinearSystem)],
            stage_systems=[
                (repr(st.name), st.system) for st in self.stage_records
            ],
        )

        # ADR 0106 D2: gate the tcl/py runtime stage marker on whether
        # THIS deck asks a solver for ``-stats`` anywhere (flat or in
        # any stage) — reuses the same flat/staged system resolution as
        # the two validators above. A deck with no ``stats=True``
        # anywhere emits byte-identically to today (INV-1).
        emitter._emit_stage_markers = deck_requests_solver_stats(  # type: ignore[attr-defined]
            flat_systems=[p for p in ordered if isinstance(p, LinearSystem)],
            stage_systems=[
                (repr(st.name), st.system) for st in self.stage_records
            ],
        )

        # NormDispIncr is unreachable on a SANISAND deck -- the
        # displacement-increment residual stalls and is not mesh-neutral
        # (measured: a 4.2587e-08 floor against a 1e-8 tolerance in our own
        # live suite). Keyed on the material alone, NOT on tan_type: the
        # stall belongs to the integrator, and it was measured on an
        # elastic-tangent leg.
        validate_manzari_convergence_test(
            ordered,
            staged=_staged,
            flat_tests=[p for p in ordered if isinstance(p, ConvergenceTest)],
            stage_tests=[
                (repr(st.name), st.test) for st in self.stage_records
            ],
        )

        # A max_substeps cap only helps if the element ACTS on the refusal
        # it produces; under one that discards the return code the analysis
        # converges on a partially integrated stress. Raise, not warn --
        # that is a wrong answer, not a slow run.
        validate_sanisand_substep_cap(elements)

        # ADR 0105 D4: an ASDPlasticMaterial3D whose host discards the
        # material return code (stdBrick, fork ADR-94 B2) never sees its
        # own strict_convergence refusal. Warn, not raise -- the vanilla
        # host is legal and was the SSI-1 default.
        validate_asdplastic_host(elements)

        # ADR 0093 S7: a stage-CLAIMED interface cannot ride the staged
        # H5 archive.  The claim itself is bridge-side state — nothing
        # in ``/opensees/stages`` records WHICH interface a stage owns,
        # and the records themselves persist in the neutral zone
        # (S6), where the read side hands them back on
        # ``fem.elements.interfaces`` with no stage attribution.  A
        # replayed archive would therefore emit the whole unit in the
        # BASE pass — the liner installed at t = 0 on the unloaded
        # ground, silently a different model.  The mixed-ndf half also
        # trips the Phase-2 stage phantom guard
        # (``H5Emitter.set_stage_records``), but the equal-ndf half
        # would sail through, so refuse here for both.
        if _emitter_is_archival and any(
            st.stage_interface_records for st in self.stage_records
        ):
            claimed = sorted({
                str(r.name) for st in self.stage_records
                for r in st.stage_interface_records if r.name
            })
            raise BridgeError(
                "apeSees.h5: stage-claimed g.constraints.interface() "
                "records (s.interface(name=...): "
                + (", ".join(repr(n) for n in claimed) or "<unnamed>")
                + ") cannot be archived — the stage CLAIM has no home in "
                "/opensees/stages, so a replayed archive would emit the "
                "interface in the base pass instead of inside its stage "
                "(the liner installed at t = 0 on unloaded ground). "
                "Per-stage interface archival is an ADR 0093 S7 "
                "follow-up. Write the staged deck with ops.tcl(path) / "
                "ops.py(path), or drop the s.interface(name=) claim so "
                "the interface emits in the base pass."
            )

        # ADR 0048 — per-node ndf is INFERRED from the declared element
        # classes (authoritative). Guard ndm against the elements, then
        # resolve the per-node ndf map once; every node-emit site below
        # sources its ``-ndf`` from this map (elided when it equals the
        # ``ops.model`` envelope ``self.ndf``). Nodes absent from the map
        # — element-less / decoupled, or touched only by adaptive
        # elements — take the envelope.
        assert_ndm_compatible(
            [type(spec).__name__ for spec in elements], self.ndm,
        )
        inferred_ndf = infer_node_ndf(self.fem, elements, self.ndm)
        # ADR 0049 — merge the ``ops.ndf`` overlay (the stated ndf of
        # element-less decoupled nodes) over the inferred map BEFORE the G1
        # gate and the node-emit fan-out.  ``effective_ndf`` is a FRESH dict;
        # ``inferred_ndf`` is never mutated in place (guards the shared
        # mutable default ``inferred_ndf={}`` on the stage-emit helpers).
        # ``resolve_ndf_overlay`` fails loud on a mesh / element-touched /
        # unresolved target, so the overlay only ever sizes a node inference
        # could not reach (no two-headed model).
        _overlay = resolve_ndf_overlay(
            self.fem, self.ndf_records, inferred_ndf, self.ndm,
        )
        effective_ndf = {**inferred_ndf, **_overlay}
        # G1 — fail loud if a zeroLength-family element's two ends would emit
        # different ndf (an element-less ground falling to the envelope while
        # its structural partner infers / states a different value).  Reads
        # the EFFECTIVE map so a correct ``ops.ndf(ground, K)`` against a
        # matching structural endpoint passes instead of falsely raising.
        validate_adaptive_element_endpoints(
            self.fem, elements, self.ndm, effective_ndf, self.ndf,
        )
        # G2 — a rigidDiaphragm / rigidLink master must carry an exact ndf and
        # every constrained DOF must fit the endpoint ndf (broker + stage
        # pools); OpenSees warn-and-returns otherwise.
        validate_constraint_master_ndf(
            self.fem, effective_ndf, self.ndm, self.ndf,
            stage_constraint_records=tuple(
                r for st in self.stage_records
                for r in st.stage_constraint_records
            ),
        )
        # G3 — every fix / mass / load / sp record's DOFs must match the
        # node's effective ndf (OpenSees silently drops a mismatched record).
        # from_model() loads are emit-synthesized and out of G3's reach.
        from .pattern.pattern import Plain as _Plain
        # A stage-claimed Plain lives in BOTH self.primitives (for tag
        # allocation) and the stage's pattern_specs — dedup by id so its
        # loads / sps are validated once.
        _plains_by_id: dict[int, _Plain] = {
            id(p): p for p in self.primitives if isinstance(p, _Plain)
        }
        for _st in self.stage_records:
            for _p in _st.pattern_specs:
                _plains_by_id[id(_p)] = _p
        _plains = list(_plains_by_id.values())
        # ADR 0054 close-out: warn if a continuum body_force (always-on,
        # not pattern-gated) overlaps a from_model gravity import on the
        # same nodes along the same axis — self-weight counted twice.
        validate_body_force_double_count(
            self.fem,
            elements,
            tuple(c for p in _plains for c in p.from_model_cases),
        )
        # ADR 0091 — warn when an imported consistent load's basis
        # (lagrange | bernstein, stamped on the record at resolution)
        # mismatches the element family covering its nodes (Bézier
        # control values vs nodal values).  Fail-soft.
        validate_load_basis_vs_elements(
            self.fem,
            elements,
            tuple(c for p in _plains for c in p.from_model_cases),
        )
        # Zero-match from_model guard — an import expanding to zero
        # load/sp lines is a silent no-op at emit (typo, or a case name
        # lost to a pre-2.26.1 model.h5).  Global (whole-broker) check,
        # so per-rank-empty partitioned brackets stay legitimate.
        validate_from_model_cases(
            self.fem,
            tuple(c for p in _plains for c in p.from_model_cases),
            allow_empty=tuple(
                c for p in _plains
                for c in getattr(p, "from_model_allow_empty", ())
            ),
        )
        # ADR 0051 §4 — broker homogeneous SPs (g.constraints.bc) and
        # masses (g.masses) reach the deck only when restated on the
        # bridge; warn when some were not, instead of dropping them in
        # silence.  Archival emits skip it: they never solve, the
        # neutral zone keeps both record sets, and mass_from_model() is
        # deck/live-only, so its advice would fail there.
        if not _emitter_is_archival:
            validate_model_definition_consumed(
                self.fem, effective_ndf, self.ndf, self.ndm,
                fix_records=(
                    *self.fix_records,
                    *(r for st in self.stage_records for r in st.fix_records),
                    *(r for st in self.stage_records
                      for r in st.support_records),
                ),
                mass_records=(
                    *self.mass_records,
                    *(r for st in self.stage_records for r in st.mass_records),
                ),
                mass_from_model=self.mass_from_model,
            )
        # #1333 - a rigidDiaphragm master no element touches has its untied
        # DOFs (uz, rx, ry on a floor) stiffened by nothing: K is singular
        # there.  Warn, naming the fix; archival emits never solve, so skip.
        if not _emitter_is_archival:
            validate_diaphragm_master_stiffness(
                self.fem, elements, self.ndm, self.ndf, effective_ndf,
                fix_records=(
                    *self.fix_records,
                    *(r for st in self.stage_records for r in st.fix_records),
                    *(r for st in self.stage_records
                      for r in st.support_records),
                ),
                sp_records=tuple(sp for p in _plains for sp in p.sps),
                stage_constraint_records=tuple(
                    r for st in self.stage_records
                    for r in st.stage_constraint_records
                ),
            )
        validate_record_ndf_consistency(
            self.fem, effective_ndf, self.ndm, self.ndf,
            fix_records=(
                *self.fix_records,
                *(r for st in self.stage_records for r in st.fix_records),
            ),
            mass_records=(
                *self.mass_records,
                *(r for st in self.stage_records for r in st.mass_records),
            ),
            load_records=tuple(ld for p in _plains for ld in p.loads),
            sp_records=tuple(sp for p in _plains for sp in p.sps),
            support_records=tuple(
                r for st in self.stage_records for r in st.support_records
            ),
        )

        # G4 — a STATIC u-p deck whose pressure DOFs are all free is
        # singular in p and factorises through round-off with rc = 0 and an
        # arbitrary pressure level (fork xfail
        # test_ladruno_up_element_analytic.py:533-548).  Walk the pressure
        # regions and require a datum in each.  Scoped to Static: a sealed
        # region is physically correct under Transient (the storage term
        # regularises the p rows), and archival emits never solve — the
        # same two facts D4 above is scoped on.  Runs AFTER D4 so the
        # solver footgun still reports first on a deck with both.
        from .analysis.analysis import Static as _StaticAnalysis
        validate_up_pressure_datum(
            self.fem, elements, self.ndm,
            enforce=(
                not _emitter_is_archival
                and (
                    any(isinstance(p, _StaticAnalysis) for p in ordered)
                    or any(
                        isinstance(st.analysis, _StaticAnalysis)
                        for st in self.stage_records
                    )
                )
            ),
            fix_records=(
                *self.fix_records,
                *(r for st in self.stage_records for r in st.fix_records),
            ),
            sp_records=tuple(sp for p in _plains for sp in p.sps),
            support_records=tuple(
                r for st in self.stage_records for r in st.support_records
            ),
        )

        # 4. Emit non-element / non-transform primitives in topo order.
        pre_element: list[Primitive] = []
        post_element: list[Primitive] = []
        for p in rest:
            if isinstance(p, (Pattern, Recorder)):
                post_element.append(p)
            else:
                pre_element.append(p)

        # ADR 0099 (INV-1 / INV-3 / INV-4): a gated element's ndf bracket
        # re-issues ``model BasicBuilder``, which deletes the Tcl model
        # builder and purges the process-global timeSeries / geomTransf /
        # beamIntegration / damping registries.  The flat (S2), default
        # partitioned (S5) paths hoist their gated element blocks
        # above those declarations; the remaining paths would emit a deck
        # that dies late — or, for damping, runs to convergence and
        # reports an undamped answer.  Fail loud here, before any
        # primitive is emitted, on every path.
        #
        # ``per_rank`` (ADR 0061) slices the partitioned fan-out into
        # file-per-rank fragments, which puts a FILE boundary where the
        # single-file deck has only a brace — the fix there is the
        # source-line move the flat path got, but applied by the
        # post-emit span writer, which cannot reorder recorded spans
        # yet — so it keeps the refusal and carries its own path token.
        # It is invisible to the emit call (``tcl`` applies it after /
        # around this), hence the emitter attribute — the same seam
        # ``supports_partitions`` uses.
        if is_partitioned(self.fem) and getattr(
            emitter, "supports_partitions", True,
        ):
            _scope_path = (
                "partitioned per_rank"
                if getattr(emitter, "per_rank_fragments", False)
                else "partitioned"
            )
        else:
            _scope_path = "flat"
        validate_builder_scope_ordering(
            elements,
            ordered,
            self.fem,
            ndm=self.ndm,
            envelope_ndf=self.ndf,
            path=_scope_path,
            stage_records=self.stage_records,
        )

        # ADR 0114 D4 (amended): the tags this emit writes come from the
        # build-time tag plan of its mode, made once per mode and
        # memoised.  ``plan_tags`` seeds the planner allocator from the
        # registered primitives, reserves the FEM element-id range under
        # ``element_tags="fem"`` (ADR 0111 D2: every synthesised tag lands
        # above max(FEM id) on every path), plans every migrated family
        # (the element fan-out included) and freezes.  The emit holds the
        # plan itself and hands it to every helper, which reads its
        # family's rows and mints nothing.  The mode matches the dispatch
        # below.
        emitter_can_partition = getattr(emitter, "supports_partitions", True)
        tag_plan = self._tag_plan(emit_mode(
            self, split=False, supports_partitions=emitter_can_partition))

        # ADR 0027: partitioned vs unpartitioned branch.  The
        # unpartitioned path must be **byte-identical** to the pre-ADR
        # 0027 behaviour — no ``partition_open`` / ``partition_close``
        # calls, no runtime shim, no per-rank fan-out.  Single-
        # partition / unpartitioned models keep the flat emit order
        # exactly as it was.
        #
        # A *composed* model is auto-partitioned one-rank-per-module
        # (ADR 0038 §"Rank model"), so it reports as partitioned even
        # though it is one logical structure.  When the emit target
        # cannot drive OpenSeesMP brackets (a single-process target such
        # as the live in-process runner, whose ``partition_open(K!=0)``
        # no-ops), flattening is the only correct behaviour: ``_emit_flat``
        # emits the full unique node / element / constraint set from the
        # snapshot exactly once, i.e. the whole model in one domain.  This
        # is what lets a composed multi-module model emit ALL its nodes and
        # analyze in-process.  Partition-capable emitters (Tcl/Py/MPI
        # writers) keep the per-rank fan-out.
        if not is_partitioned(self.fem) or not emitter_can_partition:
            self._emit_flat(
                emitter=emitter,
                tag_plan=tag_plan,
                transforms=transforms,
                elements=elements,
                inferred_ndf=effective_ndf,
                pre_element=pre_element,
                post_element=post_element,
                base_resolver=_base_resolver,
            )
            return 0

        # Partitioned path — per-rank fan-out per ADR 0027.  Phase
        # SSI-2.C lifted the prior (stages + partitions) gate; staging
        # is now handled inline by :meth:`_emit_partitioned` and
        # :meth:`_emit_stages_partitioned`.
        self._emit_partitioned(
            emitter=emitter,
            tag_plan=tag_plan,
            transforms=transforms,
            elements=elements,
            inferred_ndf=effective_ndf,
            pre_element=pre_element,
            post_element=post_element,
            base_resolver=_base_resolver,
        )
        return 0

    # -- Flat (unpartitioned) emit path -----------------------------------

    def _emit_flat(
        self,
        *,
        emitter: Emitter,
        tag_plan: TagPlan,
        transforms: "list[GeomTransf]",
        elements: "list[Element]",
        inferred_ndf: "dict[int, int]",
        pre_element: "list[Primitive]",
        post_element: "list[Primitive]",
        base_resolver: object,
    ) -> None:
        """Pre-ADR 0027 flat emit path.

        Byte-identical to the original :meth:`emit` body when
        ``len(self.fem.partitions) <= 1``.  No ``partition_open`` /
        ``partition_close`` calls, no runtime shim emission.

        Phase SSI-2.A: when ``stage_records`` is non-empty, the
        analysis-chain primitives in ``pre_element`` are SKIPPED in
        the global pre-element emit and instead emitted per-stage by
        :meth:`_emit_stages_flat` at the end of this method.  This
        keeps every other primitive's emit position byte-identical
        to the non-staged path.
        """
        staged = bool(self.stage_records)

        # Phase SSI-2.B: compute element / node ownership maps when
        # stages are declared.  Stage-bound topology (nodes + elements
        # owned by a stage's activated PGs) emits inside its stage's
        # block; everything else stays in this global pre-stage emit.
        element_owner_stage: dict[int, int] = {}
        node_owner_stage: dict[int, int] = {}
        # Phase SSI-2.E: pre-allocate element tags upfront when staged
        # so V6 (``s.remove_element`` validator) can resolve explicit
        # ``elements=`` user inputs against the live tag map.  Element
        # emit later in this method re-uses ``element_plan`` instead of
        # re-allocating.  TagAllocator is per-kind so this does not
        # disturb the ``geomTransf`` / ``material`` / etc. counters.
        element_plan: "list[tuple[Element, ElementPlanRows]] | None" = None
        fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None
        if staged:
            element_owner_stage, node_owner_stage = compute_stage_ownership(
                self.stage_records, elements, self.fem,
            )
            element_plan = _planned_element_specs(tag_plan, elements)
            # ADR 0065 v2 B3: columnar tag map off the plan (no per-element
            # boxed dict). Node-pair sentinel rows are dropped in from_plan.
            fem_eid_to_ops_tag = FemToOpsTagMap.from_plan(element_plan)
            # Validate that BC pools (global + per-stage) respect the
            # ownership-tier rules — see ``_run_staged_bc_validators``
            # for the H1 / V1 / V2 / V3 / V4 / V5 / V6 surface
            # (Phase SSI-2.D + SSI-2.E).
            self._run_staged_bc_validators(
                node_owner_stage,
                element_owner_stage,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            )
            # ADR 0093 S8 hardening: an UNCLAIMED interface record whose
            # endpoint is a stage-bound node would emit a base-pass
            # zeroLength referencing a node that only exists inside the
            # stage block.
            self._validate_interfaces_not_stage_bound(node_owner_stage)

        # 1a. Nodes — emit every node from the FEM snapshot, EXCEPT
        # nodes bound to a stage (those emit inside that stage's
        # block per Phase SSI-2.B).  S2 (ADR 0033): per-node ``-ndf K``
        # token is sourced from the broker when a declaration covers
        # the node; otherwise the model envelope wins.
        for nid, xyz in zip(self.fem.nodes.ids, self.fem.nodes.coords):
            if int(nid) in node_owner_stage:
                continue
            _emit_node_with_inferred_ndf(
                emitter, inferred_ndf, int(nid),
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                self.ndf,
            )

        # 4a. Materials / sections / analysis chain (excluding patterns
        # + recorders).  Phase SSI-2.A: skip analysis-chain primitives
        # when stages are declared — each stage re-emits its own chain
        # below.
        # ADR 0092 S5 open item: when the user declared an ``analysis``
        # directive, the constraint-handler auto-emit must land BEFORE
        # it — OpenSees constructs the analysis object AT the
        # ``analysis`` command, and a later ``constraints`` line does
        # not retro-propagate into it (silently inert; measured as a
        # plausible-wrong converged answer).  Hoist the auto-emit here
        # and skip the step-7c site.  Decks with no user ``analysis``
        # primitive keep the step-7c position byte-identically.
        #
        # ADR 0099 INV-1: this pass emits only the declarations that
        # SURVIVE a ``model BasicBuilder`` re-issue.  The builder-scoped
        # ones (timeSeries / geomTransf / beamIntegration / damping) are
        # held back to step 4b, below the hoisted gated element blocks,
        # so the LAST ``model`` line in the deck precedes every one of
        # them.  Topo order is preserved within each pass.
        pre_survives: list[Primitive] = []
        pre_scoped:   list[Primitive] = []
        for p in pre_element:
            (pre_scoped if is_builder_scoped(p) else pre_survives).append(p)

        chain_auto_emitted = False
        for p in pre_survives:
            if staged and _is_analysis_chain_primitive(p):
                continue
            if not chain_auto_emitted and isinstance(p, Analysis):
                self._maybe_auto_emit_constraint_handler(
                    emitter, pre_element)
                chain_auto_emitted = True
            tag = self.tag_for[id(p)]
            p._emit(emitter, tag)

        # 5. Elements.  Phase SSI-2.B: pre-allocate ALL element tags
        # upfront (across global + stage-bound elements) so the
        # fem_eid → ops_tag map is complete before any per-stage
        # emit runs.  Then emit only globally-owned elements here;
        # stage-bound elements emit in their stage's block via
        # ``_emit_stages_flat`` below.
        # Phase SSI-2.E: in the staged case the allocation already
        # happened earlier in this method (so V6 could resolve
        # explicit ``elements=`` targets); reuse the prior plan.
        # ADR 0099: allocation moved above the transform fan-out so the
        # hoisted gated pass has a plan.  TagAllocator is per-kind, so
        # the element / geomTransf counters are unaffected by the move —
        # the staged branch above has always allocated here.
        if element_plan is None:
            element_plan = _planned_element_specs(tag_plan, elements)
            # ADR 0065 v2 B3: columnar tag map (see the staged branch above).
            fem_eid_to_ops_tag = FemToOpsTagMap.from_plan(element_plan)
        if fem_eid_to_ops_tag is None:
            raise BridgeError(
                "internal: fem_eid_to_ops_tag not populated — element_plan "
                "allocation must set the map before emit continues."
            )

        def _emit_element_block(
            spec: Element,
            sub: "ElementPlanRows",
            overrides: "dict[tuple[int, int], int] | None",
        ) -> None:
            """Emit one element spec's fan-out, bracketed if its parser gates."""
            transf_spec = _build_element_transf(spec)
            # Builder-ndf bracket for gated upstream parsers (quad/tri6n)
            # under a mixed-ndf envelope — see open_builder_ndf_bracket.
            bracketed = bool(sub) and open_builder_ndf_bracket(
                emitter, spec, ndm=self.ndm, envelope_ndf=self.ndf)
            for eid, node_tags, ele_tag in sub:
                set_element_nodes(emitter, node_tags)
                set_current_fem_element_id(emitter, eid)
                if (
                    transf_spec is not None
                    and overrides is not None
                    and (id(transf_spec), eid) in overrides
                ):
                    override_tag = overrides[(id(transf_spec), eid)]
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
            if bracketed:
                close_builder_ndf_bracket(
                    emitter, ndm=self.ndm, envelope_ndf=self.ndf)

        # 5a. ADR 0099 INV-1 — the gated element blocks, hoisted into one
        # contiguous bracketed run ABOVE every builder-scoped declaration.
        # A gated parser takes an nDMaterial and nothing else (INV-3
        # guards that at emit entry), and nDMaterial survives the model
        # re-issue, so nothing these blocks need has been held back.
        # ``overrides`` is None here: no gated class is transform-bearing.
        global_plan = [
            (spec, sub) for spec, sub in element_plan
            if id(spec) not in element_owner_stage
        ]
        gated_plan = [
            (spec, sub) for spec, sub in global_plan
            if needs_builder_ndf_bracket(
                spec, ndm=self.ndm, envelope_ndf=self.ndf)
        ]
        for spec, sub in gated_plan:
            _emit_element_block(spec, sub, None)

        # 5b. Builder-scoped declarations.  Safe from here down for the
        # NON-staged deck: the flat path emits no further ``model`` line.
        # A STAGE-ACTIVATED gated element does re-issue one inside its
        # stage block — that case is handled by the S7 replay below.
        for p in pre_scoped:
            tag = self.tag_for[id(p)]
            p._emit(emitter, tag)

        # ADR 0099 S7: a stage-activated gated element brackets inside
        # its stage block, where no hoist can reach — the staged pass
        # instead re-declares the builder-scoped declarations at bracket
        # close (``replay_builder_scoped_declarations``).  Armed on the
        # same two conditions as every 0099 hoist — a stage-owned
        # bracket-needing element AND a builder-scoped declaration for
        # the bracket to destroy — so every other deck does not move a
        # byte.  The transform fan-out allocates per-vecxz tags, so the
        # replay re-drives a CAPTURE of the emitted lines (a re-run
        # could not reproduce the tags); ``pre_scoped`` primitives
        # re-emit deterministically and need no capture.
        transf_replay_log: "list[tuple[Any, ...]] | None" = None
        if (
            staged
            and (pre_scoped or transforms)
            and any(
                id(spec) in element_owner_stage
                and needs_builder_ndf_bracket(
                    spec, ndm=self.ndm, envelope_ndf=self.ndf)
                for spec, _sub in element_plan
            )
        ):
            transf_replay_log = []

        # 6. GeomTransf fan-out (builder-scoped — same reasoning as 5b).
        overrides = emit_transform_specs(
            transforms=transforms,
            emitter=emitter,
            fem=self.fem,
            tag_plan=tag_plan,
            spec_to_own_tag=self.tag_for,
            ndm=self.ndm,
            replay_log=transf_replay_log,
        )

        # 6b. The ungated elements, in plan order.
        gated_ids = {id(spec) for spec, _ in gated_plan}
        for spec, sub in global_plan:
            if id(spec) in gated_ids:
                continue
            _emit_element_block(spec, sub, overrides)

        # 7. Fixes / masses / regions.
        # ADR 0051: g.loads no longer auto-emit — loads reach the deck
        # only via an explicit ops.pattern.Plain(...).from_model(case)
        # import (expanded in emit_pattern_spec) or bridge-authored
        # p.load(...).  There is no broker-loads auto-emitter.
        self._emit_fixes(emitter, inferred_ndf)
        self._emit_masses(emitter, inferred_ndf)
        self._emit_regions(emitter, tag_plan)
        self._emit_rayleigh(emitter, tag_plan, fem_eid_to_ops_tag)
        self._emit_damping_attach(emitter, tag_plan, fem_eid_to_ops_tag)
        self._emit_modal_damping(emitter)

        # 7b. MP constraints (Phase 7b, ADR 0022 INV-5).  Records
        # claimed by ``s.embedded`` / ``s.equal_dof`` / ... are
        # SKIPPED here — they emit inside their owning stage's block.
        emit_mp_constraints(
            emitter, self.fem, tag_plan,
            claimed_ids=frozenset(self._claimed_constraint_ids()),
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            stiffness_resolver=self._auto_stiffness_resolver(),
        )
        emit_equation_constraints(
            emitter, self.fem, self.equation_constraint_records,
            node_ndf=inferred_ndf, default_ndf=self.ndf,
        )

        # 7b'. Embedded reinforcement ties (g.reinforce, ADR 20 / R2b).
        # One LadrunoEmbeddedRebar per rebar node; bond names resolve to
        # tags via the bridge name-alias map.
        emit_reinforce_ties(
            emitter, self.fem, tag_plan, name_to_tag=self.name_to_tag,
        )
        # Node-to-host embedment ties (g.embed). One LadrunoEmbeddedNode
        # per constrained node; no material-name resolution needed.
        emit_embed_ties(emitter, self.fem, tag_plan)
        # Face-to-face contact (g.constraints.contact). contactSurface pairs
        # + the contact verb; the LadrunoContact handler is forced by the
        # constraint-handler auto-emit below.
        emit_contacts(emitter, self.fem, tag_plan, ndm=self.ndm)
        emit_contact_planes(emitter, self.fem, tag_plan)
        # Oriented coincident-pair zeroLength interfaces
        # (g.constraints.interface, ADR 0093 D5). Per pair: the mixed-ndf
        # phantom + its equalDOF, two tributary-scaled uniaxials, one
        # zeroLength with the pair's own -orient frame. Handler-
        # independent (no exclusive handler like contact's).
        # Records claimed by ``s.interface(name=)`` are SKIPPED here —
        # they emit inside their owning stage's block (ADR 0093 S7).
        emit_interfaces(
            emitter, self.fem, tag_plan,
            effective_ndf=inferred_ndf,
            envelope_ndf=self.ndf,
            ndm=self.ndm,
            claimed_ids=frozenset(self._claimed_interface_ids()),
        )

        # 7b''. Auto-emitted structural rebar elements (g.rebar.place(
        # emit_elements=True), ADR 0067 P5.2 / B1). One CorotTruss per bar
        # PG line cell — the bar's OWN axial element, distinct from the
        # coupling above. Material resolved by name.
        emit_rebar_elements(
            emitter, self.fem, tag_plan, name_to_tag=self.name_to_tag,
        )

        # 7c. Auto-emit constraint handler when MP constraints present.
        # Skipped when already hoisted before a user ``analysis`` line
        # (ADR 0092 S5 open item — see step 4a).
        if not chain_auto_emitted:
            self._maybe_auto_emit_constraint_handler(emitter, pre_element)

        # 7d. Initial stress (Phase SSI-1).  Emit the step_hook_ramp
        # bundle (dispatcher + parameter decls + proc + lappend), then
        # one addToParameter per element / component.  Single-process
        # path = no ``partition_open`` wrapping.  In staged mode the
        # bridge's ``_initial_stress_records`` should be empty (every
        # record was ``.add()``'d to a stage), but defensively support
        # the case where some records weren't staged — they emit here
        # globally before any stage starts.
        if self.initial_stress_records:
            name_to_param_tags = emit_initial_stress_global(
                self.initial_stress_records, emitter, tag_plan,
            )
            emit_initial_stress_addtoparameter(
                self.initial_stress_records,
                emitter, self.fem,
                name_to_param_tags=name_to_param_tags,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            )

        # 8. Patterns + recorders.  Phase SSI-2.D PR-C: recorders
        # claimed by ``s.recorder(spec)`` are SKIPPED here — they
        # emit inside their owning stage's block.
        claimed_recorder_ids = self._claimed_recorder_ids()
        claimed_pattern_ids = self._claimed_pattern_ids()
        for p in post_element:
            tag = self.tag_for[id(p)]
            if isinstance(p, Pattern):
                # ADR 0051 (BL-3): patterns claimed by ``s.pattern(...)``
                # emit inside their owning stage's block; skip here.
                if id(p) in claimed_pattern_ids:
                    continue
                emit_pattern_spec(
                    p, emitter, tag, self.fem, self.ndf, self.ndm,
                    effective_ndf=inferred_ndf,
                )
            elif isinstance(p, Recorder):
                if id(p) in claimed_recorder_ids:
                    continue
                emit_recorder_spec(
                    p, emitter, tag, self.fem,
                    tag_plan=tag_plan,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )
            else:  # pragma: no cover  - unreachable per partition above
                p._emit(emitter, tag)

        # 9. Phase SSI-2.A / 2.B: per-stage emit block.  Each stage
        # emits its activated topology (Phase 2.B) + initial_stress
        # + analysis chain + analyze loop + stage_close.  No-op for
        # non-staged models.
        if staged:
            self._emit_stages_flat(
                emitter, tag_plan,
                element_plan=element_plan,
                element_owner_stage=element_owner_stage,
                node_owner_stage=node_owner_stage,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                inferred_ndf=inferred_ndf,
                overrides=overrides,
                base_resolver=base_resolver,
                scoped_replay=(
                    (pre_scoped, transf_replay_log)
                    if transf_replay_log is not None else None
                ),
            )

    def _emit_stages_flat(
        self,
        emitter: Emitter,
        tag_plan: TagPlan,
        *,
        element_plan: "list[tuple[Element, ElementPlanRows]]" = (),  # type: ignore[assignment]  # empty tuple is an immutable Sequence[never] default
        element_owner_stage: "dict[int, int]" = {},
        node_owner_stage: "dict[int, int]" = {},
        fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
        inferred_ndf: "dict[int, int]" = {},
        overrides: "dict[tuple[int, int], int] | None" = None,
        base_resolver: object = None,
        scoped_replay: "tuple[list[Primitive], list[tuple[Any, ...]]] | None" = None,
    ) -> None:
        """Phase SSI-2.A / 2.B / 2.D: emit each stage block in registration order.

        Per stage:

        1. ``stage_open(name)`` — comment delimiter.
        2. **(Phase SSI-2.B)** Stage-owned nodes — emit nodes that
           are exclusively referenced by stage-bound elements (per
           the ``node_owner_stage`` map).
        3. **(Phase SSI-2.B)** Stage-owned elements — emit the
           ``element`` commands for elements whose pg is activated
           by this stage.  Tags come from the global ``element_plan``
           (pre-allocated upfront so cross-stage tag identity holds).
           **(ADR 0099 S7)** With ``scoped_replay`` armed, the
           bracket-needing (gated) specs emit FIRST, then — if a
           bracket actually fired and the emitter's runtime actually
           purges (``model_reissue_purges``) — the builder-scoped
           declarations are re-declared, then the ungated specs, whose
           ``element`` lines may resolve a transform / integration /
           damping by tag and must see the replayed registry.  Only
           line positions move; tags come from the same plan.
        4. **(Phase SSI-2.D PR-B)** Stage-bound ``fix`` + ``mass`` —
           emit per-record ``emitter.fix(node, *dofs)`` and
           ``emitter.mass(node, *values)`` for entries in
           ``stage.fix_records`` / ``stage.mass_records``.  Validators
           V1 / V2 (PR-A) already gated these at build time.  PR-C
           will add stage-bound ``region`` emit at this same slot.
        5. ``domain_change()`` — UNCONDITIONAL stage barrier (one per
           stage).  The MPCO/Ladruno recorders open a new
           ``MODEL_STAGE`` group only when the domain-change stamp
           moves; a pure-loading stage never moves it on its own, so
           without the barrier its steps merge into the previous
           stage's group.
        6. Stage's initial_stress records (parameter declarations +
           step_hook_ramp procs + addToParameter calls, exactly the
           same shape as the Phase SSI-1 non-staged global emit).
        7. Analysis-chain primitives — emit each via its ``_emit``
           (the bridge skipped these in the pre_element pass).
        8. ``emitter.analyze(steps=, dt=)`` — auto-wraps with hook
           dispatcher calls if any step_hook_ramp registered this
           stage (the emitter tracks ``_step_hooks_registered``).
        9. ``stage_close()`` — loadConst + wipeAnalysis + hook clear.

        Single-partition dispatch arm.  For MP-partitioned models with
        stages, the dispatch goes through
        ``_emit_partitioned → _emit_stages_partitioned`` instead — see
        that method for the partition-aware emit.
        """
        # ADR 0065 v2 B3: default-empty tag map (the old mutable-{}
        # default) — every production caller passes the real map.
        if fem_eid_to_ops_tag is None:
            fem_eid_to_ops_tag = FemToOpsTagMap.from_plan(())
        # ADR 0068 Open item 5: the global EQ-aware handler auto-emit is
        # skipped for staged models, so validate each stage's declared
        # handler against any equation tie (fail loud, not silent-drop).
        self._validate_staged_eq_handlers()
        # ADR 0092: the same disease, contact edition — a stage's own
        # ``constraints`` line lands after the global auto-emit and
        # before the stage's ``analysis``, so a non-contact handler
        # silently wins and the interaction is never enforced.
        self._validate_staged_contact_handlers()

        # Pre-compute reverse maps: stage_index → list of owned nodes
        # / owned element-spec ids, for efficient per-stage lookup.
        stage_owned_nodes: dict[int, list[int]] = {}
        for nid, sidx in node_owner_stage.items():
            stage_owned_nodes.setdefault(sidx, []).append(int(nid))
        # Within each stage's bucket, emit nodes in FEM-id order so
        # the deck is grep-friendly and cross-run-stable.
        for ids in stage_owned_nodes.values():
            ids.sort()

        # Element plan filtered per stage.  Stage index → list of
        # (spec, sub_records) where sub_records are the (eid, conn,
        # ele_tag) triples already in the global plan.
        stage_owned_specs: dict[int, list[tuple[Element, ElementPlanRows]]] = {}
        for spec, sub in element_plan:
            spec_sidx = element_owner_stage.get(id(spec))
            if spec_sidx is not None:
                stage_owned_specs.setdefault(spec_sidx, []).append((spec, sub))

        # FEM node-id → coord index lookup (mirrors the
        # _emit_partitioned helper inline).  Columnar per ADR 0100 D3.
        node_idx_lookup = node_index_lookup(self.fem.nodes.ids)

        for stage_idx, stage in enumerate(self.stage_records):
            emitter.stage_open(stage.name)

            # Phase SSI-2.E: set_time + set_creep emit right after
            # stage_open so they override the previous stage_close's
            # ``loadConst -time 0.0`` and so the stage's analyze loop
            # sees the right creep state from line 1.
            if stage.set_time is not None:
                emitter.set_time(float(stage.set_time))
            if stage.set_creep_on is not None:
                emitter.set_creep(bool(stage.set_creep_on))

            # 2. Owned nodes.  S2 (ADR 0033): per-node ndf via broker.
            owned_nodes = stage_owned_nodes.get(stage_idx, [])
            for nid in owned_nodes:
                idx = node_idx_lookup.get(nid)
                if idx is None:
                    continue
                xyz = self.fem.nodes.coords[idx]
                _emit_node_with_inferred_ndf(
                    emitter, inferred_ndf, int(nid),
                    (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    self.ndf,
                )

            # 3. Owned elements.
            def _emit_owned_spec(
                spec: Element, sub: "ElementPlanRows",
            ) -> bool:
                """Emit one stage-owned spec's fan-out; True if it bracketed."""
                transf_spec = _build_element_transf(spec)
                # Builder-ndf bracket for gated upstream parsers
                # (quad/tri6n) under a mixed-ndf envelope — see
                # open_builder_ndf_bracket.
                bracketed = bool(sub) and open_builder_ndf_bracket(
                    emitter, spec, ndm=self.ndm, envelope_ndf=self.ndf)
                for eid, node_tags, ele_tag in sub:
                    set_element_nodes(emitter, node_tags)
                    set_current_fem_element_id(emitter, eid)
                    if (
                        transf_spec is not None
                        and overrides is not None
                        and (id(transf_spec), eid) in overrides
                    ):
                        override_tag = overrides[(id(transf_spec), eid)]
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
                if bracketed:
                    close_builder_ndf_bracket(
                        emitter, ndm=self.ndm, envelope_ndf=self.ndf)
                return bracketed

            owned_specs = stage_owned_specs.get(stage_idx, [])
            if scoped_replay is None:
                # No stage-owned bracket / nothing builder-scoped to
                # destroy anywhere in the model: registration order,
                # byte-identical to the pre-S7 emitter.
                for spec, sub in owned_specs:
                    _emit_owned_spec(spec, sub)
            else:
                # ADR 0099 S7: hoist first, replay only what's left —
                # the gated specs bracket at the TOP of the stage block,
                # then the purged declarations are re-declared (on a
                # purging runtime only; measured, the in-process module
                # purges nothing and a replay there collides), then the
                # ungated specs, whose element lines resolve transforms /
                # integrations / dampings by tag.
                gated_owned = [
                    (spec, sub) for spec, sub in owned_specs
                    if needs_builder_ndf_bracket(
                        spec, ndm=self.ndm, envelope_ndf=self.ndf)
                ]
                stage_bracket_fired = False
                for spec, sub in gated_owned:
                    if _emit_owned_spec(spec, sub):
                        stage_bracket_fired = True
                if stage_bracket_fired and getattr(
                    emitter, "model_reissue_purges", False,
                ):
                    scoped_prims, transf_log = scoped_replay
                    replay_builder_scoped_declarations(
                        emitter,
                        scoped_primitives=scoped_prims,
                        tag_for=self.tag_for,
                        transform_log=transf_log,
                    )
                gated_ids = {id(spec) for spec, _sub in gated_owned}
                for spec, sub in owned_specs:
                    if id(spec) not in gated_ids:
                        _emit_owned_spec(spec, sub)

            # Phase SSI-2.E: removals emit BEFORE new BCs so a stage
            # can release a prior-tier support and immediately re-fix
            # the same DOF / re-bind the same element in this stage.
            # Validators V5 / V6 already gated these at build time.
            for sp_rem in stage.remove_sp_records:
                for node_tag in self._resolve_node_target(
                    sp_rem.pg, sp_rem.nodes,
                ):
                    for dof in sp_rem.dofs:
                        emitter.remove_sp(int(node_tag), int(dof))
            for ele_rem in stage.remove_element_records:
                # ``elements=`` from the user is a list of FEM eids
                # (matching the recorder.Element convention); translate
                # to OpenSees ops tags at emit time.
                if ele_rem.pg is not None:
                    fem_eids_for_emit: "Iterable[int]" = (
                        int(eid)
                        for eid, _conn in expand_pg_to_elements(
                            self.fem, ele_rem.pg,
                        )
                    )
                else:
                    fem_eids_for_emit = (
                        int(eid) for eid in (ele_rem.elements or ())
                    )
                for fem_eid in fem_eids_for_emit:
                    ops_tag = fem_eid_to_ops_tag.get(int(fem_eid))
                    if ops_tag is not None:
                        emitter.remove_element(int(ops_tag))

            # Phase SSI-2.E: SANISAND stage flips.  Emitted AFTER the
            # removals (so a stage can release / re-bind first) and
            # necessarily after step 3's element activation above —
            # updateMaterialStage reaches a material through the
            # Domain's LIVE elements, not through the material
            # registry.  Validator V7 already gated this at build time.
            for mat_stage in stage.update_material_stage_records:
                for mat_tag in mat_stage.mat_tags:
                    emitter.update_material_stage(
                        int(mat_tag), int(mat_stage.stage),
                    )

            # 4. Stage-bound BCs (Phase SSI-2.D PR-B + PR-C): fix +
            # mass + region.  Per-record fan-out via _resolve_node_target
            # (same as the global path).  Records emit in registration
            # order; regions group records by name within this stage,
            # allocate one tag per name (V3 guarantees no cross-scope
            # name collision), and emit one ``region $tag -node ...``
            # per name.
            for fix_rec in stage.fix_records:
                for node_tag in self._resolve_node_target(fix_rec.pg, fix_rec.nodes):
                    emitter.fix(int(node_tag), *fix_rec.dofs)
            for mass_rec in stage.mass_records:
                for node_tag in self._resolve_node_target(mass_rec.pg, mass_rec.nodes):
                    node = int(node_tag)
                    emitter.mass(node, *fit_dof_vector(
                        mass_rec.values, int(inferred_ndf.get(node, self.ndf)),
                        kind="mass", node=node))
            self._emit_stage_regions(stage, emitter, tag_plan)
            # Stage-bound MP constraints — emit AFTER regions, BEFORE
            # domain_change so the constrained nodes / elements (which
            # emitted at the top of the stage block) are already in
            # the OpenSees domain when the constraint references them.
            if stage.stage_constraint_records:
                emit_stage_mp_constraints(
                    stage.stage_constraint_records, emitter, tag_plan,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                    stiffness_resolver=self._auto_stiffness_resolver(),
                )

            # ADR 0093 S7 (INV-6): stage-claimed interfaces — the
            # liner-install pattern.  Emitted AFTER the stage MP
            # constraints and BEFORE ``domain_change``, for the same
            # reason: the pair's endpoints (and the phantom this pass
            # mints for a mixed-ndf pair) must be in the Domain when
            # the ``zeroLength`` references them.  Element / material
            # tags continue the shared allocator.
            emit_stage_interfaces(
                stage.stage_interface_records, emitter, tag_plan,
                effective_ndf=inferred_ndf,
                envelope_ndf=self.ndf,
                ndm=self.ndm,
            )

            # ADR 0052: stage-bound HOLD supports — emit AFTER the MP
            # constraints, BEFORE domain_change.  Each flagged DOF emits
            # ``sp <node> <dof> [nodeDisp <node> <dof>] -const`` inside
            # the stage's dedicated ``Plain`` pattern (claimed, so
            # neither the global nor the 7b pattern pass touches it),
            # bound to the shared ``Constant`` series.  ``-const`` pins
            # the value; the ``nodeDisp`` capture resolves at runtime
            # against the prior stage's committed state.
            if stage.support_records and stage.support_pattern is not None:
                pat = stage.support_pattern
                pat_tag = self.tag_for[id(pat)]
                ts_tag = self.tag_for[id(pat.series)]
                emitter.pattern_open("Plain", pat_tag, ts_tag)
                for sup_rec in stage.support_records:
                    for node_tag in self._resolve_node_target(
                        sup_rec.pg, sup_rec.nodes,
                    ):
                        for dof_idx, flag in enumerate(sup_rec.dofs, start=1):
                            if flag:
                                emitter.sp_hold(int(node_tag), dof_idx)
                emitter.pattern_close()

            # 5. domainChange — unconditional stage barrier.  Earlier
            # phases gated this on the stage mutating the domain, but
            # the MPCO/Ladruno recorders open a new MODEL_STAGE group
            # only when the domain-change stamp moves — a pure-loading
            # stage (nodal-load pattern + analyze) never moves it, so
            # its steps were silently appended into the PREVIOUS
            # stage's MODEL_STAGE (stages "lost" in results/viewer).
            # A nodal load does not set the Domain's changed flag
            # (Domain::addNodalLoad), so the barrier must fire every
            # stage.  Single barrier per stage.
            emitter.domain_change()

            # 5b. Stage-bound damping (ADR 0053 D5).  Emitted AFTER
            # domainChange so the stage's elements are in the renumbered
            # domain when ``rayleigh`` binds and ``region -ele … -damp/
            # -rayleigh`` resolves; BEFORE the analysis chain (rayleigh is a
            # domain directive, not part of the chain).  The Damping object
            # definitions themselves emit once, pre-element (global pool);
            # only the attach is stage-scoped.  Modal damping is not staged.
            self._emit_rayleigh(
                emitter, tag_plan, fem_eid_to_ops_tag,
                stage=stage,
            )
            self._emit_damping_attach(
                emitter, tag_plan, fem_eid_to_ops_tag,
                stage=stage,
            )

            # 6. Initial stress.
            if stage.initial_stress_records:
                name_to_param_tags = emit_initial_stress_global(
                    stage.initial_stress_records, emitter, tag_plan,
                )
                emit_initial_stress_addtoparameter(
                    stage.initial_stress_records,
                    emitter, self.fem,
                    name_to_param_tags=name_to_param_tags,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

            # 6b. Absorbing-boundary stage flip (ADR 0054 AB-3).
            if stage.activate_absorbing_records:
                emit_activate_absorbing(
                    stage.activate_absorbing_records,
                    emitter, tag_plan,
                )

            # 6c. ``s.update_parameter`` — the general form of the same
            # primitive, so it shares 6b's slot rationale: the stage's
            # elements are in the Domain and the analyze loop has not
            # started, so the new value is what this stage steps with.
            if stage.update_parameter_records:
                emit_update_parameters(
                    stage.update_parameter_records,
                    emitter, tag_plan,
                )

            # 7. Analysis chain.
            for chain in (
                stage.constraints, stage.numberer, stage.system,
                stage.test, stage.algorithm, stage.integrator,
                stage.analysis,
            ):
                if chain is not None:
                    chain_tag = self.tag_for[id(chain)]
                    chain._emit(emitter, chain_tag)

            # 7b. Stage-scoped patterns (ADR 0051 BL-3) — emit AFTER
            # the chain, BEFORE analyze so the pattern's loads / sps /
            # from_model imports drive THIS stage's analyze loop and
            # are frozen by the stage's ``stage_close`` ``loadConst``.
            # Reuse the flat ``emit_pattern_spec`` so PG fan-out +
            # from_model(case) expansion match the non-staged path.
            for pat in stage.pattern_specs:
                pat_tag = self.tag_for[id(pat)]
                emit_pattern_spec(
                    pat, emitter, pat_tag, self.fem, self.ndf, self.ndm,
                    effective_ndf=inferred_ndf,
                )

            # 8. Stage-bound recorders (Phase SSI-2.D PR-C) — emit
            # AFTER the chain so the recorder sees the bound analysis
            # chain, BEFORE analyze so the recorder captures the
            # stage's analyze steps.  Same emit_recorder_spec helper
            # as the global path; the recorder's tag was allocated
            # at ops.recorder.X(...) call time and remains valid.
            for rec_spec in stage.recorder_specs:
                rec_spec_tag = self.tag_for[id(rec_spec)]
                emit_recorder_spec(
                    rec_spec, emitter, rec_spec_tag, self.fem,
                    tag_plan=tag_plan,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

            # Phase SSI-2.E: pre-analyze reset, if requested.  Emits
            # ``reset`` between the recorder declarations and the
            # analyze loop; wipes the Domain back to the last
            # ``setTime`` call.  Rare; kept for parity with the
            # OpenSees surface.
            if stage.pre_analyze_reset:
                emitter.reset()

            # 8b. Transient → static handover: zero the inherited nodal
            # velocity / acceleration state.  LAST before ``analyze``,
            # and specifically AFTER ``reset`` — ``reset`` reverts the
            # Domain to the last ``setTime``, which would restore the
            # very velocities this is removing.
            if stage.zero_velocity_records:
                emit_zero_velocities(
                    zero_velocity_target_nodes(
                        stage.zero_velocity_records, self.fem.nodes.ids,
                    ),
                    emitter,
                    effective_ndf=inferred_ndf,
                    envelope_ndf=self.ndf,
                )

            # 9. Analyze loop (auto-wraps with hook dispatcher calls).
            # Deck emitters return 0 (their per-increment loops fail
            # loud at RUN time); the live emitter returns the first
            # failing rc — raise rather than run the next stage on a
            # silently partial state.
            # TIMs A8: per-stage profiler bracket (``s.profile``) —
            # ``profiler start [flags]`` immediately before THIS
            # stage's analyze loop only.
            if stage.profile is not None:
                emitter.profiler("start", *_stage_profile_start_flags(stage.profile))
            rc = emitter.analyze(
                steps=stage.n_increments, dt=stage.dt, label=stage.name,
                strategy=_stage_strategy_spec(stage),
            )
            if rc != 0:
                raise BridgeError(
                    f"stage {stage.name!r}: analyze FAILED (rc={rc}) — "
                    f"aborting; running the remaining stages on a "
                    f"partial state is almost never intended."
                )

            # TIMs A8: close the bracket — ``profiler stop`` then
            # ``profiler report <stage name>.h5`` immediately after
            # THIS stage's analyze loop, reported under the stage's own
            # name (``stop`` ends the run the way ``ops.profiler.stop``
            # does at bridge level; ``report`` appends the ended run).
            if stage.profile is not None:
                emitter.profiler("stop")
                emitter.profiler("report", f"{stage.name}.h5")

            # 10. Stage close — loadConst + wipeAnalysis + hook clear.
            emitter.stage_close()

    # -- Partitioned emit path (ADR 0027) ---------------------------------

    def _emit_partitioned(
        self,
        *,
        emitter: Emitter,
        tag_plan: TagPlan,
        transforms: "list[GeomTransf]",
        elements: "list[Element]",
        inferred_ndf: "dict[int, int]",
        pre_element: "list[Primitive]",
        post_element: "list[Primitive]",
        base_resolver: object,
    ) -> None:
        """Per-rank fan-out implementing ADR 0027 (cross-partition MP-constraint
        emission policy).

        Build order:

        1. **Pre-element global primitives** (materials, sections, time series,
           geomTransf, analysis chain). Emitted ONCE outside any
           ``partition_open`` block — these are global script state.
           Phase SSI-2.C: when ``stage_records`` is non-empty, analysis-
           chain primitives are SKIPPED here and re-emitted per-stage by
           :meth:`_emit_stages_partitioned`.
        2. **Per-rank emission** for each rank in ascending partition id:

           * ``partition_open(rank)``
           * Owned nodes (only this rank's ``node_ids`` from the
             ``PartitionRecord``). Phase SSI-2.C: stage-bound nodes are
             SKIPPED here and emitted inside their stage's block.
           * Owned elements (per-rank fan-out across each Element spec;
             non-owned elements still consume a tag slot to preserve
             cross-rank tag identity per ADR 0027 §"Tag determinism").
             Phase SSI-2.C: stage-bound element specs are SKIPPED here
             and emitted inside their stage's block.
           * Owned fixes / masses / regions (with per-rank intersection
             of region member ids per INV-4).
           * Broker nodal loads (re-partitioned per-rank).
           * MP constraints — replicated per ADR 0027 §"Decision" with
             foreign-node declarations preceding each constraint per
             INV-2; phantom-node tags broker-derived per INV-3.
           * Pattern loads / sps (per-rank).
           * ``partition_close()``

        3. **Analysis chain** (constraint handler auto-upgrade,
           numberer / system auto-upgrade per INV-5, pure recorder
           declarations). Phase SSI-2.C: auto-emit handlers are
           SKIPPED in the staged case (each stage validated by
           :class:`_StageBuilder` carries its own complete chain).
        4. **Per-stage blocks** (Phase SSI-2.C) — only when
           ``stage_records`` is non-empty: dispatch to
           :meth:`_emit_stages_partitioned`.
        """
        if self.equation_constraint_records:
            # An EQ row spans arbitrary nodes; routing it to the ranks that
            # own all of them (and ghosting the rest) is not implemented.
            # Fail loud rather than drop or duplicate it.
            raise BridgeError(
                "apeSees.equation_constraint rows are not supported on a "
                "partitioned (OpenSeesMP) emit. Emit an unpartitioned deck, "
                "or run in-process."
            )
        staged = bool(self.stage_records)

        # g.reinforce ties and g.rebar auto-emitted bars are routed per rank
        # (see _plan_partitioned_reinforcement / step 7c-ter).
        elements_comp = getattr(self.fem, "elements", None)

        # g.embed (LadrunoEmbeddedNode ties): like reinforce ties, an embed
        # tie spans the constrained node + its host element's nodes, which may
        # straddle ranks. Per-rank node-ownership routing is deferred — fail
        # loud rather than silently dropping the embedment under MPI emit
        # (the flat path emits these via emit_embed_ties; the partitioned
        # path has no such call).
        if getattr(elements_comp, "embed_ties", None):
            raise BridgeError(
                "apeSees: g.embed embedded-node ties (LadrunoEmbeddedNode) "
                "are not yet supported under partitioned (MPI) emit — per-rank "
                "node-ownership routing of the constrained node + host nodes "
                "is deferred. Emit the embedded model single-process "
                "(non-partitioned), or remove the embedment for the "
                "partitioned run."
            )

        # ADR 0092 S3 (INV-3): `soft=` / `edge_soft=` are structurally
        # incompatible with partitioning — k_soft = SOFSCL·4·m_eff/dt² needs
        # the ASSEMBLED mass of BOTH contact surfaces, and the ghosted side
        # contributes zero on the owner rank, which no owner rule can fix
        # (fork ADR-78 D4; the fork engine likewise refuses -soft/-edgeSoft
        # at handle() time under MPI — fork ADR-78 §P2 LOG). S4 relaxed the
        # old blanket serial-only refusal to the locality contract
        # (INV-1/INV-2, see step 7c below); this check survived it as the
        # one authoring feature partitioned contact never gets. Serial emit
        # is untouched — this runs only on the partitioned path.
        from apeGmsh._kernel.resolvers._contact_ownership import (
            soft_family_knobs,
        )
        for verb, recs in (
            ("g.constraints.contact", getattr(elements_comp, "contacts", None)),
            ("g.constraints.contact_plane",
             getattr(elements_comp, "contact_planes", None)),
        ):
            for idx, rec in enumerate(recs or (), start=1):
                knobs = soft_family_knobs(rec)
                if not knobs:
                    continue
                label = f"#{idx}" + (
                    f" ({rec.name!r})" if getattr(rec, "name", None) else ""
                )
                raise BridgeError(
                    f"apeSees: {verb} interaction {label} sets "
                    f"{' + '.join(k + '=' for k in knobs)} — the explicit "
                    "SOFT penalty is unavailable under partitioned (MPI) "
                    "emit (ADR 0092 INV-3). k_soft = SOFSCL*4*m_eff/dt^2 "
                    "needs the ASSEMBLED mass of BOTH contact surfaces, and "
                    "the ghosted side of the interface contributes zero "
                    "assembled mass on the owner rank — no owner rule can "
                    "fix that (fork ADR-78 D4; the fork engine refuses "
                    "-soft/-edgeSoft under MPI at handle() time, ADR-78 P2). "
                    "Drop the SOFT knob and keep kn='auto' (or an explicit "
                    "penalty), or emit the model serial (non-partitioned, "
                    "or flat=True)."
                )

        # ADR 0092 S4: the blanket "serial-only" contact refusal that stood
        # here is GONE — partitioned contact now emits under the locality
        # contract (INV-1: one owner rank per interaction, master-side;
        # INV-2: the whole non-native interface ghosted as `node` + SP
        # replay). The plan is built below (once element ownership exists,
        # see _plan_partitioned_contacts) and emitted inside the per-rank
        # loop (step 7c); the named refusals for the genuinely unsupported
        # cases (undecidable owner, cut master + auto-sizing, staged decks)
        # live in the planner.

        # ADR 0093 S9: the staged + partitioned interface refusal that
        # stood here is GONE — a stage-claimed interface now emits
        # inside its owner rank's STAGE block (the campaign's actual
        # scenario: the liner installed on the equilibrated ground,
        # under MPI).  The per-stage plans are built below alongside
        # the base-pass plan and consumed by _emit_stages_partitioned.

        # ADR 0093 S8: the blanket interface refusal that stood here is
        # GONE — g.constraints.interface() now emits under partitioning.
        # Each pair is an ATOMIC unit — phantom + nested equalDOF + two
        # materials + one zeroLength — that lands inside exactly one
        # rank's block, owned by the rank holding the pair's backing
        # continuum element (INV-5). The plan is built below (see
        # _plan_partitioned_interfaces — every INV-5 assertion fires
        # before a single line is emitted) and emitted inside the
        # per-rank loop (step 7c-bis); material + element tags are
        # pre-allocated in flat record order before the rank fan-out so
        # 1-rank and N-rank decks stay byte-comparable (ADR 0027).

        # ADR 0049: a node-pair zeroLength-family element
        # (ops.element.*(nodes=...)) has no backing FEM element id, so
        # build_element_partition_owner cannot place it on a rank and
        # emit_element_spec_partitioned would silently drop it on EVERY rank.
        # Per-rank node-ownership routing of an explicit node-pair (whose two
        # endpoints may straddle ranks) is deferred — fail loud rather than
        # emit a partitioned deck missing the spring.  Fires before stage
        # ownership / tag allocation so no node-pair sentinel reaches the
        # per-rank fan-out.
        if len(self.fem.partitions) > 1 and any(
            getattr(spec, "pg", None) is None for spec in elements
        ):
            raise BridgeError(
                "apeSees: node-pair elements (ops.element.<ZeroLength|"
                "CoupledZeroLength|TwoNodeLink>(nodes=...)) are not yet "
                "supported under partitioned (MPI) emit — per-rank "
                "node-ownership routing of an explicit node-pair is deferred "
                "(ADR 0049). Emit single-process (non-partitioned), or wire "
                "the spring through a 2-node physical group (pg=) instead."
            )

        # Phase SSI-2.C: compute stage ownership for partitioned + staged.
        element_owner_stage: dict[int, int] = {}
        node_owner_stage: dict[int, int] = {}
        # Phase SSI-2.E: see ``_emit_flat`` for the pre-allocate
        # rationale.  Same shape under MP — ``allocate_element_tags`` is
        # called once globally and the per-rank fan-out reads back tags
        # from the resulting plan.
        early_element_plan: "list[tuple[Element, ElementPlanRows]] | None" = None
        early_fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None
        if staged:
            element_owner_stage, node_owner_stage = compute_stage_ownership(
                self.stage_records, elements, self.fem,
            )
            early_element_plan = _planned_element_specs(tag_plan, elements)
            # ADR 0065 v2 B3: columnar tag map off the plan (no boxed dict).
            early_fem_eid_to_ops_tag = FemToOpsTagMap.from_plan(
                early_element_plan
            )
            # Phase SSI-2.D + SSI-2.E: run BC ownership-tier validators
            # (H1 / V1 / V2 / V3 / V4 / V5 / V6).  Previously omitted
            # on the partitioned path for H1-V4 — the flat path
            # validated at :meth:`_emit_flat` but the equivalent check
            # on the partitioned path was missing, so a global ``fix``
            # on a stage-bound node would slip through under MP and
            # crash OpenSees at parse time.  Same call as the flat
            # path; V5 + V6 cover the SSI-2.E removal verbs.
            self._run_staged_bc_validators(
                node_owner_stage,
                element_owner_stage,
                fem_eid_to_ops_tag=early_fem_eid_to_ops_tag,
            )
            # ADR 0093 S8 hardening: same UNCLAIMED-interface stage-
            # bound-node check as the flat path — the base per-rank
            # interface pass (step 7c-bis) emits before any stage
            # block.
            self._validate_interfaces_not_stage_bound(node_owner_stage)

        partitions = list(self.fem.partitions)
        node_owners = build_node_partition_owners(self.fem)
        element_owner = build_element_partition_owner(self.fem)

        # ADR 0092 S4 (INV-1): each contact interaction's owner rank +
        # ghost node set were resolved ONCE, with its tags, by the build's
        # tag plan (ADR 0114 D4 amended; _plan_partitioned_contacts) —
        # its named refusals (undecidable owner, cut/partially-resolved
        # master under auto-sizing, staged deck) fired there, before any
        # emission. Read the routing from the plan, repeat its warnings
        # (one per emit, as before the plan), and run the pattern-sp
        # sweep, which needs this emit's ndf map and patterns.
        contact_plan = tag_plan.contacts.for_fem(self.fem)
        if contact_plan.notes:
            import warnings as _warnings
            for note in contact_plan.notes:
                _warnings.warn(note, UserWarning, stacklevel=1)
        # 2026-08-13 review F1: a pattern-borne `sp` on a node the plan
        # ghost-declares would constrain the DOF on its native rank while
        # the owner rank's ghost copy stays FREE — the same constrained-
        # DOF disagreement the interface lane refuses (ADR 0027 INV-2:
        # measured 'Matrix is Singular Numerically'; and the HOLD variant
        # runs CLEAN to a wrong answer). The ghost replay carries the fix
        # tiers only, so refuse here — before any emission — rather than
        # mirror (mirroring pattern sp onto ghosts, correctly under
        # staging, is its own project).
        contact_ghosts = contact_plan.ghost_node_ids()
        if contact_ghosts:
            self._refuse_pattern_sp_on_interface_ghosts(
                contact_ghosts, post_element, inferred_ndf,
                lane="contact",
            )
        contact_lines_by_rank: "dict[int | None, list[PlannedContact]]" = {}
        for contact_line in contact_plan.lines:
            contact_lines_by_rank.setdefault(
                contact_line.owner_rank, []).append(contact_line)

        # ADR 0093 S8 (INV-5): resolve each interface record's owner
        # rank (element-side, via the stamped backing continuum
        # element) + foreign slave ghost set ONCE, before any emission —
        # the single-owner and master-native assertions fire here, and
        # so does the pattern-borne-sp-on-ghost refusal, so a refused
        # model emits nothing. ADR 0093 S9: records claimed by
        # ``s.interface(name=)`` are EXCLUDED from the base-pass plan —
        # each claimed unit is planned per stage below and emits inside
        # its owner rank's stage block (_emit_stages_partitioned),
        # never step 7c-bis. The side-list is read ONCE into a local —
        # the tag pre-pass below keys its plan by id(record), so every
        # consumer must see the same objects.
        interface_records = list(
            getattr(elements_comp, "interfaces", None) or ()
        )
        stage_claimed_interface_ids = frozenset(
            self._claimed_interface_ids()
        )
        unclaimed_interfaces = [
            rec for rec in interface_records
            if id(rec) not in stage_claimed_interface_ids
        ]
        interface_plan_by_rank = self._plan_partitioned_interfaces(
            unclaimed_interfaces, partitions,
            inferred_ndf=inferred_ndf,
            post_element=post_element,
        )
        # g.reinforce ties + g.rebar bars: owner rank + ghost bar nodes,
        # resolved once before any emission.
        reinforcement_plan_by_rank = self._plan_partitioned_reinforcement(
            node_owners,
        )

        # ADR 0093 S9 (INV-6 under INV-5): per-stage owner/ghost plans
        # for the CLAIMED records — the same _plan_rank_interfaces
        # assertions (a claimed pair's owner is still the rank holding
        # its stamped backing continuum element) and the same
        # pattern-sp-on-ghost refusal, all before any emission.
        stage_interface_plans = self._plan_stage_interfaces_partitioned(
            partitions,
            inferred_ndf=inferred_ndf,
            post_element=post_element,
            node_owner_stage=node_owner_stage,
        )

        # Pre-allocate element tags ONCE per element across all PGs
        # (ADR 0027 §"Tag determinism"). Per-rank fan-out then looks
        # tags up rather than re-allocating, so cross-rank tag identity
        # holds for every element (the rank-K block uses the same
        # element tag as the owning rank's block).
        # Phase SSI-2.E: in the staged case the allocation already
        # happened earlier in this method (so V6 could resolve
        # ``s.remove_element`` explicit ``elements=`` targets); reuse
        # the prior plan.
        # ADR 0099 S5: allocation (and the per-rank bucketing that reads
        # it) moved above the pre-element pass so the hoisted gated pass
        # below has a plan.  TagAllocator is per-kind, so the element /
        # geomTransf counters are unaffected by the move — the staged
        # branch above has always allocated here.  Same move ``_emit_flat``
        # made for S2.
        if early_element_plan is not None:
            element_plan = early_element_plan
            if early_fem_eid_to_ops_tag is None:
                raise BridgeError(
                    "internal: early_fem_eid_to_ops_tag not set when "
                    "early_element_plan is set — staged path must populate both."
                )
            fem_eid_to_ops_tag = early_fem_eid_to_ops_tag
        else:
            element_plan = _planned_element_specs(tag_plan, elements)
            # Global fem-eid → ops-tag map; used by the initial_stress
            # per-rank ``addToParameter`` fan-out to translate the user's
            # FEM element selection into OpenSees element tags (Phase
            # SSI-1).  ADR 0065 v2 B3: columnar tag map off the plan
            # (no per-element boxed dict).
            fem_eid_to_ops_tag = FemToOpsTagMap.from_plan(element_plan)

        # Rank-independent lookups, hoisted out of the per-rank loop —
        # each is O(model), so rebuilding them per rank made the whole
        # pass O(model × ranks) (measured: dominant emit cost at
        # production rank counts).  ADR 0100 D3/D4: both are columnar —
        # the node-index lookup is an argsort permutation (16 B/node vs
        # the dict's ~84), and the per-rank element buckets are LAZY
        # (8 B/row of permutation resident; each rank's select_rows
        # copy materialises inside its own block and dies with it).
        node_idx_lookup = node_index_lookup(self.fem.nodes.ids)
        plan_by_rank = {
            id(spec): LazyRankBuckets(sub, element_owner)
            for spec, sub in element_plan
        }

        # -- 1'. ADR 0099 S5 (INV-1) — plan the hoisted gated pass. -------
        # A gated element block is wrapped in a ``model BasicBuilder``
        # re-issue (:func:`open_builder_ndf_bracket`), and that re-issue
        # purges the process-global timeSeries / geomTransf /
        # beamIntegration / damping registries.  Default partitioned emit
        # is ONE file with ``if {[getPID] == K}`` BRACE guards and those
        # declarations global, outside every guard — a brace is not a file
        # boundary, so the ``_emit_flat`` hoist replicates directly: one
        # extra rank-guard block per rank carrying that rank's gated
        # elements and the nodes they need, placed above the declarations.
        #
        # The damage this fixes is RANK-LOCAL: a rank owning no gated
        # element never executes the bracket and runs fine, so the
        # pre-S5 failure is non-deterministic in ``np`` — the same model
        # passes on 2 ranks and dies on 4.
        #
        # Both gates matter.  With no builder-scoped declaration INV-1
        # holds vacuously, and with no OWNED gated row no bracket ever
        # fires; in either case hoisting is pure churn, so the deck keeps
        # the line order it had, byte for byte.
        #
        # The scoped gate counts ``transforms`` alongside ``pre_element``.
        # ``GeomTransf`` is builder-scoped but is pre-binned into its own
        # list by step 3 of ``emit`` and NEVER reaches ``pre_element``, so
        # reading ``pre_element`` alone missed the deck whose only scoped
        # declaration is a geomTransf — gated continuum plus an
        # ``elasticBeamColumn`` (a transform, but no beamIntegration /
        # timeSeries / damping).  That deck skipped the hoist and emitted
        # its ``geomTransf`` above the owning rank's bracket, which is the
        # rank-local INV-1 violation this whole slice exists to prevent.
        # ``pre_scoped`` itself stays ``pre_element``-only: it is the
        # hold-back list for step 1c, and the transform fan-out already
        # emits below the hoist unconditionally.
        pre_scoped: list[Primitive] = [
            p for p in pre_element if is_builder_scoped(p)
        ]
        hoist_by_rank: "dict[int, list[Element]]" = {}
        hoisted_spec_ids: set[int] = set()
        hoisted_nodes_by_rank: dict[int, set[int]] = {}
        if pre_scoped or transforms:
            gated_specs = [
                spec for spec, _sub in element_plan
                if not (staged and id(spec) in element_owner_stage)
                and needs_builder_ndf_bracket(
                    spec, ndm=self.ndm, envelope_ndf=self.ndf)
            ]
            # ADR 0100 D4: the pre-pass gates on per-rank COUNTS (from
            # element_owner, no rows materialised); step 1b materialises
            # each hoisted rank's rows inside its own block.
            for _idx, _part in enumerate(partitions):
                _rank = runtime_rank_from_partition_record(_part, _idx)
                rank_specs: "list[Element]" = [
                    spec for spec in gated_specs
                    if plan_by_rank[id(spec)].count(_rank)
                ]
                if rank_specs:
                    hoist_by_rank[_rank] = rank_specs
            if hoist_by_rank:
                hoisted_spec_ids = {id(spec) for spec in gated_specs}
        if not hoist_by_rank:
            # No hoist: the pre-element pass stays one topo-ordered loop.
            pre_scoped = []
        pre_survives: list[Primitive] = (
            [p for p in pre_element if not is_builder_scoped(p)]
            if pre_scoped else list(pre_element)
        )

        # -- 1. Pre-element global primitives. ----------------------------
        # Phase SSI-2.C: skip analysis-chain primitives when staged —
        # each stage re-emits its own chain inside its stage block.
        # ADR 0092 S5 open item: the INV-5 auto-emits (constraint
        # handler + parallel numberer/system) must precede a user-
        # declared ``analysis`` directive — OpenSees constructs the
        # analysis object AT that command, and ``constraints`` /
        # ``numberer`` do not retro-propagate into it (silently inert;
        # measured: PlainHandler ran and the 2-rank twin converged to a
        # plausible-wrong answer).  Hoist them here when a user
        # ``Analysis`` primitive exists and skip the step-3 site; decks
        # without one keep the step-3 position byte-identically.
        # ADR 0099 S5 (INV-1): this pass emits only the declarations that
        # SURVIVE a ``model BasicBuilder`` re-issue.  The builder-scoped
        # ones are held back to step 1c, below the hoisted gated rank
        # blocks, so the LAST ``model`` line in the deck precedes every
        # one of them.  Topo order is preserved within each pass, and
        # ``pre_scoped`` is empty (so this IS ``pre_element``) whenever
        # the hoist is off.
        suppress_chain_auto = bool(getattr(
            emitter, "suppress_analysis_chain_auto_emit", False))
        chain_auto_emitted = False
        for p in pre_survives:
            if staged and _is_analysis_chain_primitive(p):
                continue
            if (not chain_auto_emitted and not suppress_chain_auto
                    and isinstance(p, Analysis)):
                self._maybe_auto_emit_constraint_handler(
                    emitter, pre_element)
                self._maybe_auto_emit_parallel_numberer(
                    emitter, pre_element)
                self._maybe_auto_emit_parallel_system(
                    emitter, pre_element)
                chain_auto_emitted = True
            tag = self.tag_for[id(p)]
            p._emit(emitter, tag)

        # -- 1b. ADR 0099 S5 (INV-1) — the hoisted gated rank blocks. -----
        # One extra ``if {[getPID] == K}`` guard per rank that OWNS a
        # gated element, carrying the nodes those elements need and then
        # the bracketed element blocks.  Ranks owning none get NO block:
        # an empty guard is dead weight in Tcl and a syntax error in the
        # Python emitter.
        #
        # Every node line here is one the per-rank pass below would have
        # emitted anyway — same source, same node-id order, same skips —
        # so the hoist MOVES lines and never adds or drops one.  A gated
        # parser takes an nDMaterial and nothing else (INV-3 guards that
        # at emit entry), and nDMaterial survives the re-issue, so nothing
        # these blocks need has been held back.  ``transf_tag_for_element``
        # is None: no gated class is transform-bearing, and the geomTransf
        # fan-out has not run yet by construction.
        for idx, part in enumerate(partitions):
            rank = runtime_rank_from_partition_record(part, idx)
            rank_gated_specs = hoist_by_rank.get(rank)
            if not rank_gated_specs:
                continue
            # ADR 0100 D4: materialise THIS rank's gated rows only now,
            # inside its own block (the pre-pass gated on counts).
            gated_rows = [
                (spec, plan_by_rank[id(spec)].get(rank, []))
                for spec in rank_gated_specs
            ]
            needed = {
                int(n)
                for _spec, _sub in gated_rows
                for (_eid, node_tags, _tag) in _sub
                for n in node_tags
            }
            emitted_nodes: set[int] = set()
            emitter.partition_open(rank)
            try:
                for nid in sorted(int(n) for n in part.node_ids):
                    if nid not in needed:
                        continue
                    if staged and nid in node_owner_stage:
                        continue
                    node_idx = node_idx_lookup.get(nid)
                    if node_idx is None:
                        continue
                    xyz = self.fem.nodes.coords[node_idx]
                    _emit_node_with_inferred_ndf(
                        emitter, inferred_ndf, int(nid),
                        (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                        self.ndf,
                    )
                    emitted_nodes.add(nid)
                for ele_spec, ele_rows in gated_rows:
                    emit_element_spec_partitioned(
                        spec=ele_spec,
                        emitter=emitter,
                        fem=self.fem,
                        pre_allocated=ele_rows,
                        base_resolver=base_resolver,
                        transf_tag_for_element=None,
                        partition_rank=rank,
                        element_owner=element_owner,
                        ndm=self.ndm,
                        envelope_ndf=self.ndf,
                    )
            finally:
                emitter.partition_close()
            hoisted_nodes_by_rank[rank] = emitted_nodes

        # -- 1c. Builder-scoped declarations.  Safe from here down: the
        # partitioned path emits no further ``model`` line outside a
        # stage block (a stage-OWNED gated element is refused by INV-4
        # on THIS path — the flat staged path replays instead, ADR 0099
        # S7).
        for p in pre_scoped:
            tag = self.tag_for[id(p)]
            p._emit(emitter, tag)

        # GeomTransf fan-out is GLOBAL — orientation-driven per-element
        # vecxz fan-out emits one geomTransf line per distinct vecxz,
        # which must be declared once on every rank that uses it. The
        # simplest correct path is to emit transforms outside any
        # partition_open block so they're available to every rank.
        overrides = emit_transform_specs(
            transforms=transforms,
            emitter=emitter,
            fem=self.fem,
            tag_plan=tag_plan,
            spec_to_own_tag=self.tag_for,
            ndm=self.ndm,
        )

        # ADR 0093 S8 / ADR 0027 §"Tag determinism": every interface
        # record's material + element tags are allocated HERE, in flat
        # side-list order, before the rank fan-out — a record's tags
        # must not depend on which rank owns it, or 1-rank and N-rank
        # decks stop being byte-comparable ACROSS RANKS. Allocated
        # right after the structural element tags. Flat↔partitioned
        # tag IDENTITY additionally holds only for a model with no
        # other element-minting MP pass: rigid-body / kinematic-
        # coupling / ASDEmbeddedNodeElement tags are minted inside the
        # per-rank 7b pass below, while the flat path mints them
        # BEFORE emit_interfaces — measured drift for a combined
        # model: interface zeroLength tags 21-24 partitioned vs 25-28
        # flat with a coexisting g.constraints.embedded (exactly-once
        # emission and global tag uniqueness hold regardless; ADR 0093
        # INV-5, amended during S8).
        interface_tag_plan: "dict[int, tuple[int, int, int]]" = {}
        if interface_records:
            # ADR 0093 S9: claimed records join the SAME pre-pass, in
            # the sequence the serial-staged deck consumes the
            # allocator — unclaimed rows in flat side-list order first
            # (the base pass), then each stage's claimed rows in stage
            # order (the stage passes). Claimed and unclaimed records
            # therefore draw from ONE deterministic sequence regardless
            # of which path emits them, and serial-staged vs
            # partitioned-staged decks stay tag-comparable under the
            # same conditional documented above (no coexisting
            # element-minting MP pass).
            ordered_interface_records = unclaimed_interfaces + [
                rec
                for stage in self.stage_records
                for rec in stage.stage_interface_records
            ]
            interface_tag_plan = allocate_interface_tags(
                ordered_interface_records, tag_plan,
            )

        # Initial stress — global side (parameter declarations + proc +
        # lappend) emits ONCE outside any ``partition_open`` block.
        # The per-rank ``addToParameter`` fan-out happens inside each
        # rank's block, below.  Per OpenSeesMP semantics the deck is
        # executed by every rank, so the global block runs N times —
        # but parameter / proc / lappend state is rank-local in MP, so
        # each rank ends up with the same local setup.
        init_stress_param_tags: dict[str, tuple[int, int, int]] = {}
        if self.initial_stress_records:
            init_stress_param_tags = emit_initial_stress_global(
                self.initial_stress_records, emitter, tag_plan,
            )

        # Stable per-rank node tags — sort within each rank by node id
        # so cross-rank diffs of the emitted text are grep-friendly.
        # Cache the owned set per rank for the fix / mass / region
        # passes below.  Keyed by the **0-based runtime rank** (what
        # OpenSeesMP's ``getPID()`` returns), derived via ``enumerate``
        # over ``partitions`` (which already iterates in sorted Gmsh-id
        # order).  Broker's ``part.id`` stays Gmsh's 1-based label and
        # is preserved verbatim on the records themselves; only the
        # runtime-rank seam is 0-based.
        # ADR 0100 D2: both membership containers are columnar
        # (sorted int64 arrays, 8 B/node) — membership via searchsorted,
        # and the staged pass's set-algebra sites use the vectorised
        # intersection/isdisjoint forms.
        rank_owned_nodes: dict[int, SortedIntSet] = {}
        for idx, rec in enumerate(partitions):
            rank = runtime_rank_from_partition_record(rec, idx)
            rank_owned_nodes[rank] = SortedIntSet.from_ids(rec.node_ids)

        # ADDITIVE nodal quantities (mass lines, pattern load lines) emit
        # on each node's PRIMARY rank only — OpenSeesMP sums shared-node
        # contributions across ranks, so the every-owner fan-out that is
        # correct for idempotent lines (node / fix / sp) double-counts
        # interface nodes (see primary_owner_map).
        primary_owner = primary_owner_map(node_owners)
        rank_primary_nodes: dict[int, SortedIntSet] = (
            bucket_primary_nodes_by_rank(primary_owner, rank_owned_nodes)
        )

        # ADR 0027 INV-4 (MPCO recorder path): for every MPCO recorder
        # that carries a filter, resolve its full filter ids ONCE and
        # read its planned region tag ONCE — both shared across every rank.
        # The per-rank loop below emits one ``region <tag> -node ... -ele ...``
        # line per rank with the rank's owned subset (or skips the rank
        # entirely when the intersection is empty).  After the per-rank
        # loop, the recorder declaration itself emits ONCE globally with
        # ``-R <tag>`` referencing the shared tag — MPCO post-processing
        # then stitches the per-rank ``.mpco`` outputs by tag identity.
        mpco_filter_plan = self._plan_partitioned_mpco_recorders(
            post_element, tag_plan,
        )

        # Pre-compute claimed-constraint ids ONCE before the per-rank
        # loop — the set is rank-independent (every rank's global
        # constraint pass excludes the same records).
        stage_claimed_constraint_ids = frozenset(
            self._claimed_constraint_ids()
        )

        # ``node_idx_lookup`` / ``plan_by_rank`` — the other two
        # rank-independent lookups — are built above step 1, where the
        # ADR 0099 S5 hoist needs them.
        # ``ghost_sp_ops`` is the rank-independent
        # {node: [("fix", dofs), ...]} view the foreign-node declarations
        # need (ADR 0027 INV-2) — a ghost is not owned by the declaring
        # rank, so its BCs are absent from that rank's bucket and must be
        # replicated alongside the ghost ``node`` line. Built in the same
        # walk (see the method), and only when the model can actually
        # declare a ghost.  This is the GLOBAL tier; a staged model
        # extends it stage by stage in _emit_stages_partitioned.
        # ADR 0092 S4: contact ghosts (INV-2) replay the same
        # rank-independent SP stream MP-constraint ghosts do (ADR 0027
        # INV-2), so the by_node view is also built whenever a contact
        # interaction may declare one.
        # ADR 0093 S8: an interface's foreign slave/beam node ghosts
        # with the same replayed SP stream (INV-5 — geometry + SP only,
        # never mass, per ADR 0092 INV-7), so the by_node view is also
        # built whenever an interface may declare one — including a
        # STAGE-CLAIMED one (ADR 0093 S9), whose ghost is declared
        # inside its owner rank's stage block and must replay the
        # model-level ``fix`` tier plus the stage history.
        fix_plan_by_rank, ghost_sp_ops = self._bucket_fix_targets_by_rank(
            node_owners,
            by_node=(
                _fem_has_mp_constraints(self.fem)
                or bool(contact_lines_by_rank)
                or bool(interface_plan_by_rank)
                or any(bool(p) for p in stage_interface_plans)
            ),
        )
        # Which real (non-phantom) ghosts each rank declared in the
        # GLOBAL constraint pass below.  A staged model has to keep
        # those ghosts' SP state in sync with their owner across later
        # stage blocks (ADR 0027 INV-2), which means knowing who holds
        # a ghost for whom.
        ghost_tags_by_rank: "dict[int, set[int]]" = {}
        mass_plan_by_rank = self._bucket_mass_targets_by_rank(primary_owner)
        # ADR 0065 Tier 2 — model masses streamed from fem.nodes.masses,
        # bucketed by PRIMARY owner (mass is additive under MP, so each node
        # emits on one rank only), in snapshot order so the per-rank lines
        # match an explicit per-node ops.mass loop byte-for-byte.
        model_mass_by_rank: "dict[int, list[Any]]" = {}
        if self.mass_from_model and self._guard_mass_from_model(emitter):
            for _m in self.fem.nodes.masses:
                _prk = primary_owner.get(int(_m.node_id))
                if _prk is not None:
                    model_mass_by_rank.setdefault(_prk, []).append(_m)

        # Pre-compute the post-element rank-local plan for the bridge's
        # fix / mass / region / load passes.  We use the same shapes
        # the flat path uses but pre-intersect with per-rank ownership.
        # ``rank`` is the **0-based runtime rank** matching
        # OpenSeesMP's ``getPID()`` — derived from ``enumerate`` over
        # ``partitions`` so it does not collide with Gmsh's 1-based
        # ``part.id``.  See the bug fix in commit titled
        # ``fix(opensees-bridge): emit 0-based runtime ranks``.
        for idx, part in enumerate(partitions):
            rank = runtime_rank_from_partition_record(part, idx)
            emitter.partition_open(rank)
            try:
                # 1a. Owned nodes — emit in node-id order for stable
                # cross-rank diffs.  Only THIS rank's node_ids are
                # emitted here; foreign-side declarations for cross-
                # partition MP constraints happen in the constraint
                # pass below (INV-2).
                _hoisted: set[int] = hoisted_nodes_by_rank.get(rank) or set()
                for nid in sorted(int(n) for n in part.node_ids):
                    # Phase SSI-2.C: stage-bound nodes emit inside
                    # their stage's block, not in the global pre-stage
                    # per-rank pass.  S2 (ADR 0033): per-node ndf via
                    # broker.
                    if staged and nid in node_owner_stage:
                        continue
                    # ADR 0099 S5: already emitted in this rank's
                    # hoisted gated block (step 1b).
                    if nid in _hoisted:
                        continue
                    node_idx = node_idx_lookup.get(nid)
                    if node_idx is None:
                        continue
                    xyz = self.fem.nodes.coords[node_idx]
                    _emit_node_with_inferred_ndf(
                        emitter, inferred_ndf, int(nid),
                        (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                        self.ndf,
                    )

                # 6. Elements — per-rank fan-out (tags pre-allocated;
                # plan pre-bucketed by owner rank so each rank walks
                # only its own elements).
                # Phase SSI-2.C: stage-bound element specs emit inside
                # their stage's block.
                for ele_spec, _pre_alloc in element_plan:
                    if staged and id(ele_spec) in element_owner_stage:
                        continue
                    # ADR 0099 S5: gated specs emitted in step 1b.
                    if id(ele_spec) in hoisted_spec_ids:
                        continue
                    emit_element_spec_partitioned(
                        spec=ele_spec,
                        emitter=emitter,
                        fem=self.fem,
                        pre_allocated=plan_by_rank[id(ele_spec)].get(
                            rank, []),
                        base_resolver=base_resolver,
                        transf_tag_for_element=overrides,
                        partition_rank=rank,
                        element_owner=element_owner,
                        ndm=self.ndm,
                        envelope_ndf=self.ndf,
                    )

                # 7. Fixes / masses (per-rank ownership, pre-bucketed).
                # Fixes replicate on every owning rank (idempotent);
                # masses are ADDITIVE under MP assembly so each node's
                # mass emits on its primary rank only.
                eff_ndf = inferred_ndf or {}
                for fix_rec, fix_nodes in fix_plan_by_rank.get(rank, ()):
                    for nid in fix_nodes:
                        emitter.fix(nid, *fix_rec.dofs)
                for mass_rec, mass_nodes in mass_plan_by_rank.get(rank, ()):
                    for nid in mass_nodes:
                        emitter.mass(nid, *fit_dof_vector(
                            mass_rec.values,
                            int(eff_ndf.get(nid, self.ndf)),
                            kind="mass", node=nid))
                for _m in model_mass_by_rank.get(rank, ()):
                    _nid = int(_m.node_id)
                    emitter.mass(_nid, *broker_mass_components(
                        _m.mass, int(eff_ndf.get(_nid, self.ndf)),
                        self.ndm, node=_nid))

                # 7-bis. Named regions (per-rank intersection — INV-4).
                self._emit_regions_partitioned(
                    emitter, tag_plan, rank_owned_nodes[rank], rank,
                )

                # 7-ter. MPCO recorder filter regions (INV-4 — internal
                # region resolution).  The recorder DECLARATION itself
                # is emitted globally after the per-rank loop; only the
                # ``region <tag> -node ... -ele ...`` lines vary per
                # rank (and may be skipped when the intersection is
                # empty).  The region tag is the SAME scalar across
                # every emitting rank so MPCO post-merge can stitch by
                # tag identity.
                self._emit_mpco_filter_regions_for_rank(
                    emitter, rank, mpco_filter_plan, rank_owned_nodes[rank],
                    element_owner,
                    fem_eid_to_ops_tag,
                )

                # 7a. (ADR 0051) No broker-loads auto-emit. Per-rank
                # from_model imports expand in _emit_patterns_partitioned.

                # 7b. MP constraints (ADR 0027 — replication policy).
                # Stage-claimed records are SKIPPED — they emit
                # inside their owning stage's block.
                ghost_tags_by_rank[rank] = set(
                    emit_mp_constraints_partitioned(
                        emitter=emitter,
                        fem=self.fem,
                        partition_rank=rank,
                        foreign_node_ndf=int(self.ndf),
                        inferred_ndf=inferred_ndf,
                        tag_plan=tag_plan,
                        ndm=self.ndm,
                        claimed_ids=stage_claimed_constraint_ids,
                        fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                        ghost_sp_ops=ghost_sp_ops,
                        stiffness_resolver=self._auto_stiffness_resolver(),
                    )
                )

                # 7c. Contact interactions owned by THIS rank (ADR 0092
                # S4, INV-1/INV-2/INV-7): ghost `node` declarations + the
                # owner's SP replay first, then the contactSurface pair +
                # contact / contactPlane verb — inside exactly one rank's
                # block (emission on two ranks converges to a plausible
                # WRONG answer with no warning; measured, fork ADR-78
                # P0.d). Runs AFTER 7b so already-declared MP-constraint
                # ghosts are not re-declared.
                self._emit_contacts_partitioned(
                    emitter,
                    contact_lines_by_rank.get(rank, []),
                    declared_ghosts=ghost_tags_by_rank[rank],
                    ghost_sp_ops=ghost_sp_ops,
                    inferred_ndf=inferred_ndf,
                    node_idx_lookup=node_idx_lookup,
                )

                # 7c-bis. Interface units owned by THIS rank (ADR 0093
                # S8, INV-5): per record, the foreign slave's ghost
                # `node` declaration (home ndf) + the owner's SP replay
                # first, then the ATOMIC unit — phantom → equalDOF →
                # materials → zeroLength — inside exactly one rank's
                # block. Duplicate emission converges to a plausible
                # WRONG answer (double stiffness, half penetration)
                # while base reactions still balance (ADR 0092 /
                # fork ADR-78 P0), so single-rank emission is
                # structural: the plan maps each record to exactly one
                # owner and this step reads only plan[rank]. Runs
                # AFTER 7b/7c so already-declared ghosts are not
                # re-declared. Flat-order position parity: the flat
                # path emits interfaces after contacts too. The
                # per-rank phantom re-registration inside this call is
                # LOAD-BEARING: step 7b's emit_mp_constraints_partitioned
                # REPLACES the emitter's phantom-tag set each rank
                # (set_phantom_node_tags, not a union), so interface
                # phantoms must be re-unioned after every rank's 7b or
                # the H5 emitter misclassifies them as real nodes.
                self._emit_interfaces_partitioned(
                    emitter,
                    interface_plan_by_rank.get(rank, []),
                    interface_tag_plan=interface_tag_plan,
                    declared_ghosts=ghost_tags_by_rank[rank],
                    ghost_sp_ops=ghost_sp_ops,
                    inferred_ndf=inferred_ndf,
                    node_idx_lookup=node_idx_lookup,
                )

                # 7c-ter. Embedded reinforcement owned by THIS rank: ghost
                # bar nodes first, then the LadrunoEmbeddedRebar ties and the
                # bar CorotTrusses (g.reinforce / g.rebar).
                self._emit_reinforcement_partitioned(
                    emitter, tag_plan,
                    reinforcement_plan_by_rank.get(rank),
                    declared_ghosts=ghost_tags_by_rank[rank],
                    ghost_sp_ops=ghost_sp_ops,
                    inferred_ndf=inferred_ndf,
                    node_idx_lookup=node_idx_lookup,
                )

                # 7d. Initial stress — per-rank ``addToParameter`` fan-
                # out for owned elements only (Phase SSI-1).  The
                # global step_hook_ramp was emitted before the per-rank
                # loop; this block attaches each owned element's
                # response to the previously declared parameter tags.
                if self.initial_stress_records:
                    emit_initial_stress_addtoparameter(
                        self.initial_stress_records,
                        emitter, self.fem,
                        name_to_param_tags=init_stress_param_tags,
                        fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                        element_owner=element_owner,
                        partition_rank=rank,
                    )

                # 8. Patterns (loads + sps) per-rank.  ADR 0051 (BL-3):
                # stage-claimed patterns emit inside their stage block.
                self._emit_patterns_partitioned(
                    emitter, post_element, rank_owned_nodes[rank],
                    primary_nodes=rank_primary_nodes[rank],
                    inferred_ndf=inferred_ndf,
                    claimed_pattern_ids=frozenset(
                        self._claimed_pattern_ids()
                    ),
                )
            finally:
                emitter.partition_close()

        # -- 2b. Global domain-level damping (ADR 0053).  ``rayleigh`` and
        # the Damping-object ``region -ele … -damp`` attaches are emitted
        # ONCE outside any rank block (every rank binds its locally-owned
        # elements; OpenSeesMP skips foreign -ele tags) — mirroring the
        # flat path's driver-post emit and the stage-bound partitioned
        # pass.  Before this the partitioned path emitted NO global
        # damping, silently dropping a non-stage ``ops.damping.rayleigh``
        # so np>1 ran undamped (the plane-wave handoff's finding #1).
        self._emit_global_damping_partitioned(
            emitter, tag_plan, fem_eid_to_ops_tag,
        )

        # -- 3. Analysis chain — emitted GLOBALLY (outside any rank
        # block).  Auto-emit Transformation handler + ParallelPlain
        # numberer + Mumps system per ADR 0027 §"Constraint handler
        # interaction" (INV-5).
        # Phase SSI-2.C: in the staged case each stage carries a
        # complete user-declared chain (validated by
        # :class:`_StageBuilder`), so the global auto-emit is skipped
        # to avoid emitting a stale fallback chain that would
        # interfere with per-stage state.  Users must declare a
        # parallel-friendly chain (``ParallelPlain`` / ``Mumps`` /
        # ``Transformation``) inside each ``s.analysis(...)``.
        # ADR 0077 Tier 1B: the modal deck FORCES its own eigen preamble
        # right before the captured `eigen` (the `system Mumps` line is
        # load-bearing there, INV-8), so it opts out of the auto-emit
        # rather than carry a second identical numberer/system pair above
        # it. Same emitter-attribute seam as ``supports_partitions``.
        # ADR 0092 S5 open item: skipped when already hoisted before a
        # user-declared ``analysis`` directive in the step-1 pass.
        if not staged and not suppress_chain_auto and not chain_auto_emitted:
            self._maybe_auto_emit_constraint_handler(emitter, pre_element)
            self._maybe_auto_emit_parallel_numberer(emitter, pre_element)
            self._maybe_auto_emit_parallel_system(emitter, pre_element)

        # Recorders — also global (recorders write to disk, not into
        # the model topology; one recorder is sufficient even under
        # MP).  No partition wrapping.
        #
        # MPCO recorders with a filter (INV-4): the per-rank ``region``
        # lines were already emitted inside the rank loop above; here we
        # only emit the ``recorder mpco ... -R <tag>`` declaration line,
        # injecting the planned shared region tag via
        # :func:`dataclasses.replace` so MPCO.materialize's region-emit
        # branch is bypassed (the region was emitted per-rank).  All
        # other recorders (Node / Element / RecorderDeclaration / MPCO
        # without filter) route through emit_recorder_spec unchanged.
        # Phase SSI-2.D PR-C: recorders claimed by ``s.recorder(spec)``
        # are SKIPPED here — they emit inside their owning stage's
        # block.
        claimed_recorder_ids = self._claimed_recorder_ids()
        for p in post_element:
            if not isinstance(p, Recorder):
                continue
            if id(p) in claimed_recorder_ids:
                continue
            tag = self.tag_for[id(p)]
            plan_entry = mpco_filter_plan.get(id(p))
            if plan_entry is not None:
                # Pre-resolved MPCO: build the materialised spec directly
                # so its ``_emit`` appends ``-R <tag>`` without re-
                # entering MPCO.materialize (which would otherwise
                # re-emit the region globally).
                materialised = plan_entry.materialised_spec
                materialised._emit(emitter, tag)
            else:
                emit_recorder_spec(
                    p, emitter, tag, self.fem,
                    tag_plan=tag_plan,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

        # -- 4. Per-stage emit blocks (Phase SSI-2.C). ----------------
        if staged:
            self._emit_stages_partitioned(
                emitter, tag_plan,
                partitions=partitions,
                element_plan=element_plan,
                plan_by_rank=plan_by_rank,
                rank_owned_nodes=rank_owned_nodes,
                rank_primary_nodes=rank_primary_nodes,
                node_idx_lookup=node_idx_lookup,
                element_owner_stage=element_owner_stage,
                node_owner_stage=node_owner_stage,
                element_owner=element_owner,
                node_owners=node_owners,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                inferred_ndf=inferred_ndf,
                overrides=overrides,
                base_resolver=base_resolver,
                ghost_sp_ops=ghost_sp_ops,
                ghost_tags_by_rank=ghost_tags_by_rank,
                interface_tag_plan=interface_tag_plan,
                stage_interface_plans=stage_interface_plans,
            )

    # -- Partitioned staged emit (Phase SSI-2.C) --------------------------

    def _emit_stages_partitioned(
        self,
        emitter: Emitter,
        tag_plan: TagPlan,
        *,
        partitions: "list[Any]",
        element_plan: "list[tuple[Element, ElementPlanRows]]",
        plan_by_rank: "dict[int, LazyRankBuckets]",
        rank_owned_nodes: "dict[int, SortedIntSet]",
        rank_primary_nodes: "dict[int, SortedIntSet]",
        node_idx_lookup: "SortedIntToInt",
        element_owner_stage: "dict[int, int]",
        node_owner_stage: "dict[int, int]",
        element_owner: "SortedIntToInt",
        node_owners: "NodePartitionOwners",
        fem_eid_to_ops_tag: "FemToOpsTagMap",
        inferred_ndf: "dict[int, int]",
        overrides: "dict[tuple[int, int], int] | None",
        base_resolver: object,
        ghost_sp_ops: "dict[int, list[Any]] | None" = None,
        ghost_tags_by_rank: "dict[int, set[int]] | None" = None,
        interface_tag_plan: "dict[int, tuple[int, int, int]] | None" = None,
        stage_interface_plans: (
            "list[dict[int, list[tuple[Any, tuple[int, ...]]]]] | None"
        ) = None,
    ) -> None:
        """Phase SSI-2.C / 2.D: emit each stage block in registration order under MP.

        Per stage:

        1. ``stage_open(name)``.
        2. **(only if the stage activates topology OR carries stage-
           bound BCs — Phase SSI-2.D unified gate)** Per-rank loop:
           ``partition_open(rank)``; emit owned stage-bound nodes +
           owned stage-bound elements + per-rank-filtered stage-bound
           ``fix`` / ``mass`` lines (Phase SSI-2.D PR-B); ``partition_close()``.
           Then a single GLOBAL ``domain_change()`` so every rank
           rebuilds its DOF map — UNCONDITIONAL per stage (recorder
           MODEL_STAGE boundaries key off the domain-change stamp).
           Per-rank brackets are SKIPPED for ranks with no content
           (Phase SSI-2.D) so the Py emitter never produces an empty
           ``if getPID() == K:`` block.
        3. **(only if the stage carries initial-stress records)**
           Initial-stress globals (parameter declarations + step_hook
           procs) emit GLOBALLY, then a per-rank loop emits the
           ``addToParameter`` calls, filtered per rank by
           ``element_owner`` so each element gets exactly one
           ``addToParameter`` per component on its owning rank.
        4. Analysis-chain primitives (global — each rank executes them
           locally at runtime, mirroring the initial-stress
           ``parameter`` / proc semantics).
        5. ``emitter.analyze(steps=, dt=)`` (global — auto-wraps with
           hook dispatcher calls).
        6. ``stage_close()`` (global — ``loadConst -time 0`` +
           ``wipeAnalysis`` + hook-list clear).

        Cross-stage tag identity is preserved because
        :meth:`_emit_partitioned` pre-allocated ALL element tags upfront
        (across global + stage-bound element specs) before the first
        per-rank loop.  Every rank sees the same FEM-eid ↔ OpenSees-tag
        binding across every stage block.
        """
        # ADR 0068 Open item 5: per-stage EQ handler guard (the global
        # auto-emit is skipped for staged models) — same as the flat path.
        self._validate_staged_eq_handlers()
        # ADR 0092: the same disease, contact edition — a stage's own
        # ``constraints`` line lands after the global auto-emit and
        # before the stage's ``analysis``, so a non-contact handler
        # silently wins and the interaction is never enforced.
        self._validate_staged_contact_handlers()

        # ADR 0052: stage-bound HOLD supports (``s.support``) fan out
        # per owning rank below — each rank opens the stage's dedicated
        # ``Plain`` HOLD pattern (shared tag, local copy per rank, same
        # convention as :meth:`_emit_one_pattern_partitioned`) and emits
        # ``sp <node> <dof> [nodeDisp ...] -const`` for its owned target
        # DOFs only (INV-4 fan-out, mirrors ``fix``).

        # ADDITIVE nodal quantities (stage-bound mass, stage-pattern
        # load lines) emit on each node's PRIMARY rank only — same
        # policy as the global per-rank pass (see primary_owner_map).
        primary_owner = primary_owner_map(node_owners)

        # Reverse maps for efficient per-stage iteration.
        stage_owned_nodes: dict[int, set[int]] = {}
        for nid, sidx in node_owner_stage.items():
            stage_owned_nodes.setdefault(sidx, set()).add(int(nid))

        stage_owned_specs: dict[
            int, list[tuple[Element, ElementPlanRows]]
        ] = {}
        for spec, sub in element_plan:
            spec_sidx = element_owner_stage.get(id(spec))
            if spec_sidx is not None:
                stage_owned_specs.setdefault(spec_sidx, []).append((spec, sub))

        # ADR 0100 D3: the node-index lookup arrives as a parameter —
        # the co-resident staged twin (R1 ×2, confirmed deterministic
        # by the G0 campaign) is gone.

        # ADR 0027 INV-2 (amended 2026-07-28) — ghost SP synchronisation
        # across stage boundaries.  A ghost's DOFs must be constrained on
        # the declaring rank exactly when its owner has them constrained;
        # any disagreement breaks ``numberer ParallelPlain``.  Two running
        # pieces of state make that exact in BOTH directions:
        #
        # * ``sp_ops_so_far`` — the owner's SP command stream per node,
        #   replayed onto a ghost the moment it is declared.  Seeded with
        #   the global ``ops.fix`` tier and extended by each stage's
        #   ``s.remove_sp`` then ``s.fix`` (the owner's own emit order).
        #   A ghost first declared in stage N therefore carries stage
        #   N-1's BCs, which it did not before.
        # * ``ghosts_held`` — who already holds a ghost for whom.  Every
        #   LATER stage's fix / remove_sp on that node is mirrored into
        #   the holder's block, so the two ranks do not drift apart after
        #   the declaration either.
        #
        # Both halves are load-bearing and fail differently: a missing
        # ``fix`` leaves a free massless DOF (singular matrix — loud),
        # while a missed ``remove sp`` leaves the ghost MORE constrained
        # than its owner, which is not singular at all, just silently
        # stiffer.  Measured 2026-07-28 on the two-column frame.
        sp_ops_so_far: "dict[int, list[Any]]" = {
            int(k): list(v) for k, v in (ghost_sp_ops or {}).items()
        }
        ghosts_held: "dict[int, set[int]]" = {
            int(r): set(t) for r, t in (ghost_tags_by_rank or {}).items()
        }

        # ADR 0100 R8: columnar ``{ops_tag: fem_eid}`` reverse map,
        # built ONCE (it is stage-invariant; the per-stage dict rebuild
        # it replaces measured 103-228 B/elem resident through every
        # stage block).  Sole consumer: the per-rank ``remove_element``
        # routing below — its ``.get(tag, -1)`` miss default is
        # preserved exactly by SortedIntToInt.
        ops_tag_to_fem_eid = fem_eid_to_ops_tag.inverse()

        for stage_idx, stage in enumerate(self.stage_records):
            emitter.stage_open(stage.name)

            # Phase SSI-2.E: set_time + set_creep emit GLOBALLY right
            # after stage_open (outside any partition_open block —
            # OpenSeesMP executes them locally on every rank).
            if stage.set_time is not None:
                emitter.set_time(float(stage.set_time))
            if stage.set_creep_on is not None:
                emitter.set_creep(bool(stage.set_creep_on))

            owned_nodes_this_stage = stage_owned_nodes.get(stage_idx, set())
            owned_specs_this_stage = stage_owned_specs.get(stage_idx, [])
            # ADR 0055 Phase 5 (P5.1): tell the capture emitter which
            # node declarations inside this stage's rank brackets are
            # the stage's OWN topology — the stage MP pass also
            # declares foreign ghost nodes (ADR 0027 INV-2) that must
            # not enter the stage bucket's ``owned_node_ids``.
            # Cleared at stage close below; non-capture emitters
            # ignore the attribute.
            set_stage_owned_node_tags(emitter, owned_nodes_this_stage)
            has_activation = bool(
                owned_nodes_this_stage or owned_specs_this_stage
            )
            has_bcs = bool(
                stage.fix_records
                or stage.mass_records
                or stage.region_records
                or stage.stage_constraint_records
            )
            # Phase SSI-2.E: removals contribute to the unified
            # domain_change gate and per-rank empty-bracket-skip logic.
            has_removals = bool(
                stage.remove_sp_records
                or stage.remove_element_records
                or stage.update_material_stage_records
            )
            # ADR 0052: stage-bound HOLD supports likewise drive the
            # unified gate (so the per-rank loop + global domain_change
            # fire even when ``s.support`` is the only content in the
            # stage).
            has_supports = bool(stage.support_records)
            # ADR 0093 S9: this stage's claimed-interface owner/ghost
            # plan (built by _plan_stage_interfaces_partitioned before
            # any emission).  Drives the unified gate too — a stage
            # whose ONLY content is a claimed interface must still open
            # the owner rank's bracket, or the unit silently vanishes
            # (the measured s.equal_dof-alone failure mode below).
            stage_iface_plan: (
                "dict[int, list[tuple[Any, tuple[int, ...]]]]"
            ) = (
                stage_interface_plans[stage_idx]
                if stage_interface_plans else {}
            )
            has_stage_interfaces = bool(stage_iface_plan)

            # Phase SSI-2.D PR-B + PR-C: pre-resolve stage-bound BC
            # targets ONCE (rank-independent), then filter per rank
            # below.  Each entry is (record, resolved_node_id).
            fix_targets: list[tuple[FixRecord, int]] = [
                (rec, int(nid))
                for rec in stage.fix_records
                for nid in self._resolve_node_target(rec.pg, rec.nodes)
            ]
            mass_targets: list[tuple[MassRecord, int]] = [
                (rec, int(nid))
                for rec in stage.mass_records
                for nid in self._resolve_node_target(rec.pg, rec.nodes)
            ]
            # Pre-compute the union of all stage-bound region member
            # node ids so each rank can quickly check whether it owns
            # any region member.  Region records merge by name later
            # inside the per-rank fan-out via
            # :meth:`_emit_stage_regions_partitioned`.
            region_target_nodes: set[int] = set()
            for rec in stage.region_records:
                for nid in self._resolve_node_target(rec.pg, rec.nodes):
                    region_target_nodes.add(int(nid))

            # Phase SSI-2.E: pre-resolve removal targets ONCE per stage
            # (rank-independent), then filter per rank.  Same shape as
            # fix_targets / mass_targets above.
            remove_sp_targets: "list[tuple[int, int]]" = []
            for sp_rem in stage.remove_sp_records:
                for nid in self._resolve_node_target(sp_rem.pg, sp_rem.nodes):
                    for dof in sp_rem.dofs:
                        remove_sp_targets.append((int(nid), int(dof)))
            remove_element_targets: "list[int]" = []
            for ele_rem in stage.remove_element_records:
                # ``elements=`` from the user is a list of FEM eids
                # (matching the recorder.Element convention); translate
                # to OpenSees ops tags at emit time.
                if ele_rem.pg is not None:
                    fem_eid_iter: "Iterable[int]" = (
                        int(eid)
                        for eid, _conn in expand_pg_to_elements(
                            self.fem, ele_rem.pg,
                        )
                    )
                else:
                    fem_eid_iter = (
                        int(eid) for eid in (ele_rem.elements or ())
                    )
                for fem_eid in fem_eid_iter:
                    ops_tag = fem_eid_to_ops_tag.get(int(fem_eid))
                    if ops_tag is not None:
                        remove_element_targets.append(int(ops_tag))

            # ADR 0052: pre-resolve HOLD support targets ONCE per stage
            # (rank-independent) to ``(node_id, dof_idx)`` pairs, then
            # filter per rank below.  ``dof_idx`` is 1-based to match the
            # ``sp`` DOF convention; only flagged DOFs are emitted.  Same
            # INV-4 fan-out as ``fix`` — a HOLD ``sp`` replicates on every
            # rank that owns the node.
            support_targets: "list[tuple[int, int]]" = []
            for srec in stage.support_records:
                for snid in self._resolve_node_target(srec.pg, srec.nodes):
                    for dof_idx, flag in enumerate(srec.dofs, start=1):
                        if flag:
                            support_targets.append((int(snid), dof_idx))

            # ADR 0027 INV-2 — this stage's SP delta per node, in the
            # owner's own emit order (removals first, then fixes: the
            # per-rank block below emits them that way so a stage can
            # release a prior support and immediately re-fix it).
            # Reuses the already-resolved target lists — no extra broker
            # walk.
            stage_sp_delta: "dict[int, list[Any]]" = {}
            for _rnid, _rdof in remove_sp_targets:
                stage_sp_delta.setdefault(_rnid, []).append(
                    ("remove", _rdof),
                )
            for _frec, _fnid in fix_targets:
                stage_sp_delta.setdefault(_fnid, []).append(
                    ("fix", _frec.dofs),
                )
            # A ghost DECLARED in this stage replays everything its owner
            # has done up to and including this stage.
            stage_ghost_sp_ops = sp_ops_so_far
            if stage_sp_delta:
                stage_ghost_sp_ops = {
                    k: list(v) for k, v in sp_ops_so_far.items()
                }
                for _snid, _sops in stage_sp_delta.items():
                    stage_ghost_sp_ops.setdefault(_snid, []).extend(_sops)

            # 2. Per-rank topology + BC emit.  Unified gate (Phase
            # SSI-2.D): open the rank's bracket if it has ANY content
            # (owned nodes, owned elements, or owned BCs).  Phase
            # SSI-2.E widens with removals (remove_sp / remove_element).
            # Empty-bracket ranks are skipped so the Py emitter never
            # produces an empty ``if getPID() == K:`` block.
            if (
                has_activation or has_bcs or has_removals or has_supports
                or has_stage_interfaces
            ):
                for idx, part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(part, idx)
                    # Owned-node sets are stage-invariant — computed once
                    # by the caller, not rebuilt per stage × rank.
                    rank_owned = rank_owned_nodes[rank]
                    # ADR 0100 D2: sorted(rank_owned & stage_owned) on
                    # the columnar set, one vectorised pass.
                    rank_stage_nodes = rank_owned.intersection_sorted(
                        owned_nodes_this_stage
                    )
                    rank_fix = [
                        (rec, nid) for rec, nid in fix_targets
                        if nid in rank_owned
                    ]
                    # Mass is ADDITIVE under MP assembly — each node's
                    # stage-bound mass emits on its primary rank only
                    # (fix stays replicated on every owning rank).
                    rank_mass = [
                        (rec, nid) for rec, nid in mass_targets
                        if primary_owner.get(nid) == rank
                    ]
                    rank_has_region_members = not rank_owned.isdisjoint(
                        region_target_nodes
                    )
                    # ADR 0100 D4: presence via count — no rows
                    # materialised for the gate.
                    rank_has_elements = any(
                        plan_by_rank[id(spec)].count(rank)
                        for spec, _sub in owned_specs_this_stage
                    )
                    # Phase SSI-2.E: per-rank removal filtering.
                    # remove_sp replicates on every rank that has the
                    # node (mirrors fix's INV-4 fan-out).  remove_element
                    # fires only on the rank that owns the element
                    # (single owner per fem_eid).
                    rank_remove_sp = [
                        (nid, dof) for nid, dof in remove_sp_targets
                        if nid in rank_owned
                    ]
                    rank_remove_element = [
                        tag for tag in remove_element_targets
                        if element_owner.get(
                            ops_tag_to_fem_eid.get(tag, -1)
                        ) == rank
                    ]
                    # ADR 0052: HOLD ``sp`` targets owned by this rank.
                    rank_support = [
                        (nid, dof) for nid, dof in support_targets
                        if nid in rank_owned
                    ]
                    # ADR 0034 / ADR 0027: a stage-CLAIMED MP constraint
                    # is stage-bound content too.  Without this the gate
                    # skipped a rank whose only stage content was a
                    # constraint it participates in, and the constraint
                    # vanished from the deck — measured 2026-07-28: a
                    # cross-rank ``s.equal_dof`` alone in a stage emitted
                    # ZERO ``equalDOF`` lines, while the same stage plus
                    # an unrelated ``s.fix`` emitted both.  Planned here
                    # (not inside the bracket) so the empty-bracket skip
                    # stays exact: ranks the constraint does not touch
                    # plan to ``None`` and keep their bracket closed.
                    # The plan is handed to the emit below, so the
                    # participation decision is computed ONCE per
                    # (stage, rank).
                    stage_constraint_plan = (
                        plan_stage_mp_constraints_partitioned(
                            stage.stage_constraint_records,
                            partition_rank=rank,
                            node_owners=node_owners,
                            element_owner=element_owner,
                        )
                        if stage.stage_constraint_records
                        else None
                    )
                    # ADR 0027 INV-2 forward half: this rank already
                    # holds a ghost for a node whose OWNER changes its SP
                    # state in this stage.  The owner-side ``rank_fix`` /
                    # ``rank_remove_sp`` filters above are keyed on
                    # ``rank_owned``, which a ghost is by definition not
                    # in — so without this the two ranks silently drift
                    # apart from the stage the owner moved.
                    rank_ghost_sp = [
                        (nid, ops)
                        for nid, ops in stage_sp_delta.items()
                        if nid in ghosts_held.get(rank, ())
                    ]
                    # ADR 0093 S9: the claimed interface units THIS rank
                    # owns in this stage — [] for every non-owner rank,
                    # so the empty-bracket skip stays exact.
                    rank_iface_entries = stage_iface_plan.get(rank, [])
                    rank_has_content = bool(
                        rank_stage_nodes
                        or rank_has_elements
                        or rank_fix
                        or rank_mass
                        or rank_has_region_members
                        or rank_remove_sp
                        or rank_remove_element
                        # Phase SSI-2.E: a SANISAND stage flip is
                        # model-global — every rank is its own process
                        # with its own static mElastFlag, so EVERY rank
                        # must emit the line.
                        or stage.update_material_stage_records
                        or rank_support
                        or rank_ghost_sp
                        or rank_iface_entries
                        or stage_constraint_plan is not None
                    )
                    if not rank_has_content:
                        continue
                    emitter.partition_open(rank)
                    try:
                        for nid in rank_stage_nodes:
                            node_idx = node_idx_lookup.get(nid)
                            if node_idx is None:
                                continue
                            xyz = self.fem.nodes.coords[node_idx]
                            # ADR 0048: per-node ndf from the inferred map.
                            _emit_node_with_inferred_ndf(
                                emitter, inferred_ndf, int(nid),
                                (
                                    float(xyz[0]),
                                    float(xyz[1]),
                                    float(xyz[2]),
                                ),
                                self.ndf,
                            )
                        # Per-rank element fan-out across this stage's
                        # specs — plan pre-bucketed by owner rank, so
                        # each rank walks only its own elements.
                        for ele_spec, _pre_alloc in owned_specs_this_stage:
                            emit_element_spec_partitioned(
                                spec=ele_spec,
                                emitter=emitter,
                                fem=self.fem,
                                pre_allocated=plan_by_rank[
                                    id(ele_spec)].get(rank, []),
                                base_resolver=base_resolver,
                                transf_tag_for_element=overrides,
                                partition_rank=rank,
                                element_owner=element_owner,
                                ndm=self.ndm,
                                envelope_ndf=self.ndf,
                            )
                        # Phase SSI-2.E: per-rank removals emit BEFORE
                        # new BCs so a stage can release a prior-tier
                        # support and immediately re-fix in the same
                        # stage.  remove_sp replicates on every rank
                        # that has the node (INV-4 fan-out, mirrors
                        # fix); remove_element fires only on the rank
                        # owning the element.
                        for nid, dof in rank_remove_sp:
                            emitter.remove_sp(nid, dof)
                        for ops_tag in rank_remove_element:
                            emitter.remove_element(ops_tag)
                        # Phase SSI-2.E: SANISAND stage flips, after
                        # this rank's element fan-out above (the command
                        # reaches materials through the Domain's live
                        # elements) and replicated on every rank.
                        for mat_stage in (
                            stage.update_material_stage_records
                        ):
                            for mat_tag in mat_stage.mat_tags:
                                emitter.update_material_stage(
                                    int(mat_tag), int(mat_stage.stage),
                                )
                        # Phase SSI-2.D PR-B + PR-C: per-rank stage-
                        # bound BCs (fix + mass + region).  Targets
                        # pre-resolved above; per-rank filter via
                        # ``rank_owned`` intersection mirrors the
                        # existing INV-4 fan-out convention.  Every
                        # contributing rank writes the one tag the
                        # plan gave each region name.
                        for fix_rec, nid in rank_fix:
                            emitter.fix(nid, *fix_rec.dofs)
                        # ADR 0027 INV-2 forward half: mirror this
                        # stage's SP delta onto the ghosts this rank
                        # already holds.  Each node's ops are already in
                        # the owner's order (removals, then fixes).
                        for _gnid, _gops in rank_ghost_sp:
                            emit_ghost_sp_ops(emitter, _gnid, _gops)
                        for mass_rec, nid in rank_mass:
                            emitter.mass(int(nid), *fit_dof_vector(
                                mass_rec.values,
                                int(inferred_ndf.get(int(nid), self.ndf)),
                                kind="mass", node=int(nid)))
                        self._emit_stage_regions_partitioned(
                            stage, emitter, tag_plan,
                            owned_nodes=rank_owned, rank=rank,
                        )
                        # Stage-bound MP constraints — per-rank fan-
                        # out using the same replication rules as the
                        # global partitioned constraint pass.  The plan
                        # was computed by the content gate above; a
                        # ``None`` plan means this rank contributes
                        # nothing (and, if nothing else did either, the
                        # bracket was never opened).
                        if stage_constraint_plan is not None:
                            emit_stage_mp_constraints_partitioned(
                                stage_constraint_plan,
                                emitter=emitter,
                                fem=self.fem,
                                foreign_node_ndf=int(self.ndf),
                                inferred_ndf=inferred_ndf,
                                tag_plan=tag_plan,
                                ndm=self.ndm,
                                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                                ghost_sp_ops=stage_ghost_sp_ops,
                                stiffness_resolver=(
                                    self._auto_stiffness_resolver()),
                            )
                            # Ghosts declared HERE already replayed the
                            # owner's full history via
                            # ``stage_ghost_sp_ops``; from the NEXT stage
                            # on they take the per-stage mirror instead.
                            ghosts_held.setdefault(rank, set()).update(
                                stage_constraint_plan.plan.foreign_node_tags
                            )
                        # ADR 0093 S9 (INV-6 under INV-5): the claimed
                        # interface units this rank owns — emitted AFTER
                        # the stage MP constraints and BEFORE the HOLD
                        # supports / domain_change, mirroring the flat
                        # staged order (emit_stage_interfaces).  Same
                        # per-record shape as the base 7c-bis pass: a
                        # foreign slave/beam node is ghost-declared
                        # first (home ndf) with the owner's SP stream
                        # replayed up to AND INCLUDING this stage
                        # (``stage_ghost_sp_ops`` — the stage-bound
                        # ``s.fix`` tier crosses the boundary per
                        # ADR 0027 INV-2), then the atomic unit consumes
                        # its pre-allocated tag triple.
                        # ``declared_ghosts=ghosts_held[rank]`` both
                        # skips already-held ghosts (7b/7c/7c-bis and
                        # earlier stages, plus this stage's MP pass) and
                        # registers new ones, so LATER stages' SP deltas
                        # mirror onto them via ``rank_ghost_sp`` — the
                        # forward half of INV-2.  The per-rank phantom
                        # re-registration inside the call is
                        # load-bearing here too: the stage MP pass above
                        # replaces the emitter's phantom-tag set.
                        if rank_iface_entries:
                            self._emit_interfaces_partitioned(
                                emitter, rank_iface_entries,
                                interface_tag_plan=(
                                    interface_tag_plan or {}
                                ),
                                declared_ghosts=ghosts_held.setdefault(
                                    rank, set(),
                                ),
                                ghost_sp_ops=stage_ghost_sp_ops,
                                inferred_ndf=inferred_ndf,
                                node_idx_lookup=node_idx_lookup,
                            )
                        # ADR 0052: stage-bound HOLD supports — emit AFTER
                        # the MP constraints, mirroring the flat path
                        # order.  Each rank that owns at least one HOLD
                        # target opens the stage's dedicated ``Plain``
                        # pattern locally (shared tag + shared ``Constant``
                        # series, same per-rank pattern-open/close idiom as
                        # :meth:`_emit_one_pattern_partitioned`) and emits
                        # ``sp <node> <dof> [nodeDisp ...] -const`` for its
                        # owned target DOFs.  ``rank_support`` is non-empty
                        # only when this rank owns a target, so the bracket
                        # is never empty here.
                        if rank_support and stage.support_pattern is not None:
                            pat = stage.support_pattern
                            pat_tag = self.tag_for[id(pat)]
                            ts_tag = self.tag_for[id(pat.series)]
                            emitter.pattern_open("Plain", pat_tag, ts_tag)
                            for nid, dof in rank_support:
                                emitter.sp_hold(nid, dof)
                            emitter.pattern_close()
                    finally:
                        emitter.partition_close()

            # This stage's SP delta is now history: a ghost declared in
            # any LATER stage replays it too (ADR 0027 INV-2).  Folded
            # after the rank loop so the loop's declarations see the
            # stage-inclusive ``stage_ghost_sp_ops`` and its mirrors see
            # the pre-stage ``ghosts_held`` — no node gets both.
            for _snid, _sops in stage_sp_delta.items():
                sp_ops_so_far.setdefault(_snid, []).extend(_sops)

            # 3. Global ``domain_change`` — rebuild DOF map on every
            # rank.  UNCONDITIONAL (outside the content gate above):
            # the MPCO/Ladruno recorders open a new MODEL_STAGE only
            # when the domain-change stamp moves, and a pure-loading
            # stage never moves it on its own — without this barrier
            # such a stage's steps merge into the previous stage's
            # MODEL_STAGE group.  Single global call; OpenSeesMP
            # executes it locally on each rank.
            emitter.domain_change()

            # 3b. Stage-bound damping (ADR 0053 D5) — single global emit
            # after domainChange, mirroring the global partitioned damping
            # pass: ``rayleigh`` / ``region -ele … -damp/-rayleigh`` lists
            # every PG element; OpenSeesMP binds only the elements each
            # rank owns.  Modal damping is not staged.
            self._emit_rayleigh(
                emitter, tag_plan, fem_eid_to_ops_tag,
                stage=stage,
            )
            self._emit_damping_attach(
                emitter, tag_plan, fem_eid_to_ops_tag,
                stage=stage,
            )

            # 4. Initial-stress globals + per-rank ``addToParameter``.
            if stage.initial_stress_records:
                name_to_param_tags = emit_initial_stress_global(
                    stage.initial_stress_records, emitter, tag_plan,
                )
                for idx, _part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(_part, idx)
                    emitter.partition_open(rank)
                    try:
                        emit_initial_stress_addtoparameter(
                            stage.initial_stress_records,
                            emitter, self.fem,
                            name_to_param_tags=name_to_param_tags,
                            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                            element_owner=element_owner,
                            partition_rank=rank,
                        )
                    finally:
                        emitter.partition_close()

            # 4b. Absorbing-boundary stage flip (ADR 0054 AB-3) — per rank.
            if stage.activate_absorbing_records:
                for idx, _part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(_part, idx)
                    emitter.partition_open(rank)
                    try:
                        emit_activate_absorbing(
                            stage.activate_absorbing_records,
                            emitter, tag_plan, partition_rank=rank,
                        )
                    finally:
                        emitter.partition_close()

            # 4c. ``s.update_parameter`` — per rank, same ownership
            # filter as 4b (only the rank owning an element may address
            # it; an unowned eid is skipped, not an error).
            if stage.update_parameter_records:
                for idx, _part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(_part, idx)
                    emitter.partition_open(rank)
                    try:
                        emit_update_parameters(
                            stage.update_parameter_records,
                            emitter, tag_plan, partition_rank=rank,
                        )
                    finally:
                        emitter.partition_close()

            # 5. Analysis chain — global; each rank executes locally.
            for chain in (
                stage.constraints, stage.numberer, stage.system,
                stage.test, stage.algorithm, stage.integrator,
                stage.analysis,
            ):
                if chain is not None:
                    chain_tag = self.tag_for[id(chain)]
                    chain._emit(emitter, chain_tag)

            # 5b. Stage-scoped patterns (ADR 0051 BL-3) — per-rank
            # fan-out.  Unlike recorders (which write to disk and emit
            # once globally), a pattern's ``load`` / ``sp`` lines target
            # owned nodes, so they must live inside ``partition_open(rank)``
            # blocks — same convention as the global per-rank pattern
            # pass.  Pre-check owned content per rank to skip an empty
            # ``if getPID()==K:`` bracket (an empty body is a Python
            # SyntaxError on the Py emitter).  Emit AFTER the chain and
            # BEFORE analyze so the loads drive THIS stage's analyze loop
            # and freeze under the stage's ``stage_close`` ``loadConst``.
            if stage.pattern_specs:
                for idx, part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(part, idx)
                    # ADR 0100 D2 completeness (found by the mypy rebind
                    # flag): this pass used to rebuild a full per-rank
                    # owned SET plus a primary set per stage x rank —
                    # one scalar primary_owner.get per owned node, the
                    # exact construct D2 removed elsewhere.  Membership
                    # is all the pattern pre-check and emit consume, and
                    # both columnar containers carry identical
                    # membership (a node's primary rank is always an
                    # owning rank), so the deck is byte-identical.
                    rank_owned = rank_owned_nodes[rank]
                    rank_primary = rank_primary_nodes[rank]
                    if not self._stage_pattern_specs_have_owned_content(
                        stage.pattern_specs, rank_owned,
                        primary_nodes=rank_primary,
                        inferred_ndf=inferred_ndf,
                    ):
                        continue
                    emitter.partition_open(rank)
                    try:
                        for pat in stage.pattern_specs:
                            self._emit_one_pattern_partitioned(
                                emitter, pat, rank_owned,
                                primary_nodes=rank_primary,
                                inferred_ndf=inferred_ndf,
                            )
                    finally:
                        emitter.partition_close()

            # 6. Stage-bound recorders (Phase SSI-2.D PR-C) — emit
            # GLOBALLY (no per-rank wrap; recorders write to disk,
            # one declaration is sufficient under MP, same convention
            # as the global recorder pass).  Emit AFTER the chain so
            # the recorder sees the bound analysis chain; BEFORE
            # analyze so the recorder captures the stage's analyze
            # steps.
            for rec_spec in stage.recorder_specs:
                rec_spec_tag = self.tag_for[id(rec_spec)]
                emit_recorder_spec(
                    rec_spec, emitter, rec_spec_tag, self.fem,
                    tag_plan=tag_plan,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

            # Phase SSI-2.E: pre-analyze reset (global; emits ``reset``
            # outside any partition block — each rank applies locally).
            if stage.pre_analyze_reset:
                emitter.reset()

            # 6b. Transient → static handover: zero the inherited nodal
            # velocity / acceleration state.  Per-rank (INV-4): a rank
            # can only reach the nodes in its own subdomain, so each
            # rank emits its owned slice; a node held by two ranks is
            # zeroed on both (idempotent).  Emitted after ``reset``
            # and immediately before ``analyze``, same as the flat path.
            if stage.zero_velocity_records:
                zero_vel_nodes = zero_velocity_target_nodes(
                    stage.zero_velocity_records, self.fem.nodes.ids,
                )
                for idx, part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(part, idx)
                    rank_owned = rank_owned_nodes[rank]
                    rank_zero_vel = [
                        nid for nid in zero_vel_nodes if nid in rank_owned
                    ]
                    if not rank_zero_vel:
                        continue
                    emitter.partition_open(rank)
                    try:
                        emit_zero_velocities(
                            rank_zero_vel, emitter,
                            effective_ndf=inferred_ndf,
                            envelope_ndf=self.ndf,
                        )
                    finally:
                        emitter.partition_close()

            # 7. Analyze loop (auto-wraps with hook dispatcher calls).
            # See the flat path: deck emitters fail loud at RUN time;
            # a live rc != 0 raises here rather than running the next
            # stage on a silently partial state.
            # TIMs A8: per-stage profiler bracket (``s.profile``) —
            # ``profiler start [flags]`` immediately before THIS
            # stage's analyze loop only.
            if stage.profile is not None:
                emitter.profiler("start", *_stage_profile_start_flags(stage.profile))
            rc = emitter.analyze(
                steps=stage.n_increments, dt=stage.dt, label=stage.name,
                strategy=_stage_strategy_spec(stage),
            )
            if rc != 0:
                raise BridgeError(
                    f"stage {stage.name!r}: analyze FAILED (rc={rc}) — "
                    f"aborting; running the remaining stages on a "
                    f"partial state is almost never intended."
                )

            # TIMs A8: close the bracket — ``profiler stop`` then
            # ``profiler report <stage name>.h5`` immediately after
            # THIS stage's analyze loop, reported under the stage's own
            # name (``stop`` ends the run the way ``ops.profiler.stop``
            # does at bridge level; ``report`` appends the ended run).
            if stage.profile is not None:
                emitter.profiler("stop")
                emitter.profiler("report", f"{stage.name}.h5")

            # 8. Stage close — loadConst + wipeAnalysis + hook clear.
            set_stage_owned_node_tags(emitter, None)
            emitter.stage_close()

    # -- Model-level fix / mass fan-out -----------------------------------

    # -- Staged-BC validators (Phase SSI-2.D, PR-A) -----------------------

    def _run_staged_bc_validators(
        self,
        node_owner_stage: "dict[int, int]",
        element_owner_stage: "dict[int, int] | None" = None,
        *,
        fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    ) -> None:
        """Run every BC-tier validator in order; raise on the first
        offender batch.

        Called once at build time when the model is staged (i.e.
        ``self.stage_records`` is non-empty), from both
        :meth:`_emit_flat` and :meth:`_emit_partitioned`.  Each
        validator is a no-op when its scope is empty:

        - **H1** — global pool targets stage-bound nodes (red-team
          hardening from #312).
        - **V1** — stage N's BC targets a node owned by stage M > N
          (Phase SSI-2.D, PR-A).
        - **V2** — duplicate ``(node, DOF)`` fix or duplicate ``node``
          mass across global + per-stage tiers (Phase SSI-2.D, PR-A).
          Phase SSI-2.E: subtracts stage-bound ``s.remove_sp``
          targets from the fix alive set (atomic-replace pattern);
          ``mass`` records with ``overwrite=True`` bypass the
          duplicate-mass refusal.
        - **V3** — region ``name=`` collision across scopes (Phase
          SSI-2.D, PR-A).
        - **V4** — stage N's recorder targets a node owned by stage
          M > N OR an element owned by stage M > N (Phase SSI-2.D,
          PR-C).  Recorder lines parse at deck-read time and bind to
          the topology that exists at that point; a recorder
          referencing not-yet-emitted topology would crash OpenSees
          at parse time.
        - **V5** — stage N's ``s.remove_sp`` targets an SP that is
          not alive in any earlier scope at this point (Phase SSI-2.E).
          An SP is "alive" if declared in the global ``apeSees.fix``
          pool or in a strictly-earlier stage's ``s.fix`` pool AND
          not already removed by an earlier stage.  Same-stage
          ``s.fix`` does NOT count — fix emits AFTER remove_sp in
          the same stage's block.
        - **V6** — stage N's ``s.remove_element`` targets an element
          that is not alive at this point (Phase SSI-2.E).  An
          element is alive if globally emitted OR activated by this
          stage / a strictly-earlier stage AND not already removed.

        ``element_owner_stage`` is optional only for backwards-
        compatibility with the PR-A signature; callers that exercise
        recorder validation must supply it (both
        :meth:`_emit_flat` and :meth:`_emit_partitioned` do).
        ``fem_eid_to_ops_tag`` is required for V6 to validate explicit
        ``elements=`` targets against the live tag map; when None,
        V6 falls back to PG-only validation.
        """
        self._validate_no_stage_bound_node_targets(node_owner_stage)
        self._validate_stage_bound_node_targets(node_owner_stage)
        self._validate_no_duplicate_fix_mass_across_tiers(node_owner_stage)
        self._validate_region_scope_invariants()
        self._validate_stage_bound_recorder_targets(
            node_owner_stage,
            element_owner_stage or {},
        )
        self._validate_remove_sp_targets()
        # Explicit None-check (not ``or``): an EMPTY FemToOpsTagMap is
        # falsy, and ``or {}`` would silently swap it for a dict.
        self._validate_remove_element_targets(
            element_owner_stage or {},
            fem_eid_to_ops_tag
            if fem_eid_to_ops_tag is not None
            else FemToOpsTagMap.from_plan(()),
        )
        self._validate_material_stage_targets(element_owner_stage or {})

    # -- Ownership-tier helpers (PR-A: shared by H1 + V1) -----------------

    def _records_as_targets(
        self,
        records: "Iterable[FixRecord | MassRecord | RegionAssignmentRecord | SupportRecord]",
        kind: str,
    ) -> "list[tuple[str, str, str | None, tuple[int, ...] | None]]":
        """Normalise a record iterable into ``(kind, label, pg, nodes)``
        tuples ready for ``_collect_ownership_offenders``.

        ``label`` is the user-facing handle the validator surfaces in
        offender lines: the record's PG name or explicit nodes tuple
        for fix / mass, the region's ``name`` for region records.
        """
        out: "list[tuple[str, str, str | None, tuple[int, ...] | None]]" = []
        for rec in records:
            if isinstance(rec, RegionAssignmentRecord):
                label = rec.name
            else:
                label = str(rec.pg or rec.nodes)
            out.append((kind, label, rec.pg, rec.nodes))
        return out

    def _collect_ownership_offenders(
        self,
        targets: "Iterable[tuple[str, str, str | None, tuple[int, ...] | None]]",
        is_allowed: "Callable[[int | None], bool]",
        node_owner_stage: "dict[int, int]",
    ) -> "list[tuple[str, str, int, int | None]]":
        """Resolve each ``(kind, label, pg, nodes)`` to node ids and
        collect those failing the ``is_allowed`` ownership predicate.

        ``is_allowed`` receives the node's ``node_owner_stage`` lookup
        (``None`` for a globally-emitted node, an ``int`` stage index
        for a stage-bound node) and returns ``True`` if the BC is
        permitted to target that node from the caller's scope.

        Returns ``(kind, label, node_id, owner_stage_idx)`` tuples.
        """
        offenders: "list[tuple[str, str, int, int | None]]" = []
        for kind, label, pg, nodes in targets:
            for node_tag in self._resolve_node_target(pg, nodes):
                owner = node_owner_stage.get(int(node_tag))
                if not is_allowed(owner):
                    offenders.append((kind, label, int(node_tag), owner))
        return offenders

    def _render_offender_line(
        self,
        kind: str,
        label: str,
        node_id: int,
        owner_stage_idx: "int | None",
    ) -> str:
        """Single offender line used by H1 / V1 error rendering."""
        if owner_stage_idx is None:
            owner_text = "globally-emitted"
        else:
            owner_text = (
                f"owned by stage index {owner_stage_idx} "
                f"({self.stage_records[owner_stage_idx].name!r})"
            )
        return f"  • {kind} ({label!r}) targets node {node_id} → {owner_text}"

    def _validate_no_stage_bound_node_targets(
        self, node_owner_stage: "dict[int, int]",
    ) -> None:
        """Refuse global fix / mass / region directives that target
        stage-bound nodes (red-team H1).

        The pre-stage global emit fires before any ``stage_open``, so
        a ``fix 5 1 1`` line targeting a node that only emits inside
        stage 2's block would reference a non-existent node and crash
        OpenSees at parse time.  Validate upfront with the ownership
        map and a clear error message.

        PR-A (Phase SSI-2.D) refactor: shares the
        :meth:`_collect_ownership_offenders` helper with the new V1
        validator (which mirrors this check for the *stage-bound*
        pools that PR-B / PR-C will populate).  Behaviour is
        byte-identical to the pre-refactor implementation; the user-
        facing hint is updated to point at the SSI-2.D ``s.fix`` /
        ``s.mass`` / ``s.region`` verbs.
        """
        if not node_owner_stage:
            return
        targets: "list[tuple[str, str, str | None, tuple[int, ...] | None]]" = []
        targets.extend(self._records_as_targets(self.fix_records, "fix"))
        targets.extend(self._records_as_targets(self.mass_records, "mass"))
        targets.extend(self._records_as_targets(self.region_records, "region"))
        offenders = self._collect_ownership_offenders(
            targets,
            is_allowed=lambda owner: owner is None,
            node_owner_stage=node_owner_stage,
        )
        if not offenders:
            return
        lines = [
            self._render_offender_line(kind, label, nid, sidx)
            for kind, label, nid, sidx in offenders[:10]
        ]
        extra = (
            f"\n  • ... and {len(offenders) - 10} more"
            if len(offenders) > 10 else ""
        )
        raise BridgeError(
            "Stage-bound nodes referenced by GLOBAL fix / mass / "
            "region directives — those directives would emit before "
            "the stage's ``stage_open`` and reference a non-existent "
            "OpenSees node, crashing at parse time.  Either move the "
            "BC onto a globally-emitted node, or declare it inside "
            "the owning stage's ``with ops.stage(...) as s`` block "
            "via ``s.fix(...)`` / ``s.mass(...)`` / ``s.region(...)`` "
            "(Phase SSI-2.D).  Offenders:\n"
            + "\n".join(lines) + extra
        )

    def _validate_stage_bound_node_targets(
        self, node_owner_stage: "dict[int, int]",
    ) -> None:
        """Refuse stage N's fix / mass / region directives that target
        nodes owned by a LATER stage M > N (V1).

        Each stage's BCs emit inside that stage's block, after the
        stage's topology + ``domain_change``.  A stage-N ``s.fix``
        targeting a node that only comes online in stage M > N would
        reference a non-existent node at stage-N parse time, same
        failure mode as the H1 case but inverted in scope.

        Globally-emitted nodes are always legal targets (the node
        exists from the pre-stage block onward).  Nodes owned by
        stage M < N are legal too (they were emitted in stage M's
        block and persist across ``wipeAnalysis``).  Nodes owned by
        stage N itself are legal (they emit *before* the stage's BC
        block within the same stage).  Nodes owned by stage M > N
        are illegal.

        PR-A ships the validator; the stage-bound pools it iterates
        (``stage.fix_records`` / ``mass_records`` / ``region_records``)
        are populated by PR-B / PR-C builders.  Until then this
        method is a no-op on every existing test fixture.
        """
        if not self.stage_records:
            return
        offenders_per_stage: "list[tuple[str, list[tuple[str, str, int, int | None]]]]" = []
        for stage_idx, stage in enumerate(self.stage_records):
            if not (
                stage.fix_records
                or stage.mass_records
                or stage.region_records
                or stage.support_records
            ):
                continue
            targets: "list[tuple[str, str, str | None, tuple[int, ...] | None]]" = []
            targets.extend(self._records_as_targets(stage.fix_records, "s.fix"))
            targets.extend(self._records_as_targets(stage.mass_records, "s.mass"))
            targets.extend(
                self._records_as_targets(stage.region_records, "s.region")
            )
            # ADR 0052: HOLD supports are stage-bound BCs too — a stage-N
            # support targeting a node owned by a LATER stage M > N would
            # emit before the node exists, same failure mode as s.fix.
            targets.extend(
                self._records_as_targets(stage.support_records, "s.support")
            )
            _n = stage_idx
            offenders = self._collect_ownership_offenders(
                targets,
                # Allowed: globally-emitted (None) or owned by stage M <= N.
                is_allowed=lambda owner: (owner is None or owner <= _n),
                node_owner_stage=node_owner_stage,
            )
            if offenders:
                offenders_per_stage.append((stage.name, offenders))
        if not offenders_per_stage:
            return
        chunks: list[str] = []
        for stage_name, offenders in offenders_per_stage:
            lines = [
                self._render_offender_line(kind, label, nid, sidx)
                for kind, label, nid, sidx in offenders[:10]
            ]
            extra = (
                f"\n  • ... and {len(offenders) - 10} more"
                if len(offenders) > 10 else ""
            )
            chunks.append(
                f"Stage {stage_name!r} BCs reference nodes owned by a "
                f"LATER stage:\n" + "\n".join(lines) + extra
            )
        raise BridgeError(
            "Stage-bound BCs reference nodes that only come online in "
            "a later stage — the BC would emit before the target node "
            "exists, crashing OpenSees at parse time.  Move the BC to "
            "the later stage's ``with ops.stage(...) as s`` block, or "
            "split the owning PG so the target node activates earlier."
            "\n\n" + "\n\n".join(chunks)
        )

    def _validate_no_duplicate_fix_mass_across_tiers(
        self, node_owner_stage: "dict[int, int]",
    ) -> None:
        """Refuse duplicate ``(node, DOF)`` fix or duplicate ``(node)``
        mass targets across global + per-stage tiers (V2).

        OpenSees ``Domain::addSP_Constraint`` rejects duplicate
        ``(node, DOF)`` SP constraints with an error (verified at
        SRC/domain/domain/Domain.cpp:589-605); ``Domain::setMass``
        silently overwrites a node's mass with the latest value — so
        a stage-2 ``s.mass(...)`` on a node already mass-assigned in
        stage 1 (or globally) silently changes the physics.  Refuse
        both at build time.

        Tiers are: ``"global"`` (the bridge's own
        ``fix_records`` / ``mass_records``) and one tier per stage
        (``stage_records[i].fix_records`` / ``mass_records``).  The
        first occurrence of a ``(node, DOF)`` pair wins; the second
        is reported as the offender with both source tiers named.

        PR-A ships the validator; until PR-B populates the stage-
        bound pools this is a no-op on existing test fixtures.
        """
        # (node, DOF_index_1_based) → tier label of first occurrence.
        fix_owner: dict[tuple[int, int], str] = {}
        # node → tier label of first occurrence.
        mass_owner: dict[int, str] = {}
        offenders: list[str] = []

        def _scan_fix(
            records: "Iterable[FixRecord | SupportRecord]",
            tier: str,
            kind: str = "fix",
        ) -> None:
            # ADR 0052: ``s.support`` (HOLD) records share this scan with
            # ``fix`` — both create a single-point constraint on a
            # ``(node, DOF)``, and OpenSees rejects two SPs on the same
            # DOF.  ``kind`` only labels the offender message; the
            # ``fix_owner`` map is shared so fix↔support collisions
            # (across tiers or within a stage) are caught too.
            for rec in records:
                for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                    for dof_idx, flag in enumerate(rec.dofs, start=1):
                        if not flag:
                            continue
                        key = (int(node_tag), dof_idx)
                        prior = fix_owner.get(key)
                        if prior is not None:
                            offenders.append(
                                f"  • {kind} on node {node_tag} DOF "
                                f"{dof_idx} declared in {prior!r} AND in "
                                f"{tier!r}"
                            )
                        else:
                            fix_owner[key] = tier

        def _scan_mass(records: "Iterable[MassRecord]", tier: str) -> None:
            for rec in records:
                for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                    prior = mass_owner.get(int(node_tag))
                    if prior is not None:
                        # Phase SSI-2.E: overwrite=True opts out of V2.
                        # The user is acknowledging the OpenSees setMass
                        # overwrite is intentional.  Update the owner so a
                        # subsequent record on the same node still gets
                        # checked against THIS one.
                        if rec.overwrite:
                            mass_owner[int(node_tag)] = tier
                            continue
                        offenders.append(
                            f"  • mass on node {node_tag} "
                            f"declared in {prior!r} AND in {tier!r}"
                        )
                    else:
                        mass_owner[int(node_tag)] = tier

        def _scan_remove_sp(records: "Iterable[SPRemovalRecord]") -> None:
            """Phase SSI-2.E: a stage-bound s.remove_sp invalidates the
            prior alive (node, DOF) registration so a same-stage
            s.fix(...) on that target doesn't trip V2.  Removal emits
            BEFORE fix within a stage block, so the ordering is
            consistent with the actual OpenSees command sequence.
            """
            for rec in records:
                for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                    for dof in rec.dofs:
                        fix_owner.pop((int(node_tag), int(dof)), None)

        _scan_fix(self.fix_records, "global pool")
        _scan_mass(self.mass_records, "global pool")
        for stage in self.stage_records:
            tier = f"stage {stage.name!r}"
            # Same-stage emission order: removals BEFORE new fix.
            _scan_remove_sp(stage.remove_sp_records)
            _scan_fix(stage.fix_records, tier)
            # ADR 0052: HOLD supports share the SP (node, DOF) namespace.
            _scan_fix(stage.support_records, tier, kind="support")
            _scan_mass(stage.mass_records, tier)
        if offenders:
            preview = offenders[:10]
            extra = (
                f"\n  • ... and {len(offenders) - 10} more"
                if len(offenders) > 10 else ""
            )
            raise BridgeError(
                "Duplicate fix / mass targets across global + stage "
                "tiers — OpenSees rejects duplicate SP constraints "
                "(Domain::addSP_Constraint) and silently overwrites "
                "mass on repeated setMass (Domain::setMass), so the "
                "later declaration would either crash or silently "
                "change physics.  Consolidate to a single declaration "
                "per (node, DOF):\n" + "\n".join(preview) + extra
            )

    def _validate_region_scope_invariants(self) -> None:
        """Refuse cross-scope region ``name=`` collisions and mixed-
        tier region membership (V3).

        OpenSees ``Domain::addRegion`` silently appends on duplicate
        region tag (verified at SRC/domain/domain/Domain.cpp:2679-
        2697); the first region keeps the tag from
        ``Domain::getRegion`` lookups — the second is silently
        orphaned.  apeGmsh's :class:`TagAllocator` makes tag collision
        impossible for our own allocations, but two ``region`` records
        sharing the same ``name=`` across scope (global + stage, or
        stage A + stage B) silently produce two regions with
        different tags but the same user-facing name — confusing
        post-processing tooling and contradicting users' expectation
        of "name = identity".

        Refuse same-``name`` regions across distinct scopes at build
        time.  Multiple region records sharing the same name *within*
        a single scope still accumulate into one ``region`` line at
        emit time (existing behaviour preserved).

        PR-A ships the validator; until PR-C populates
        ``stage.region_records`` this is a no-op on existing test
        fixtures.
        """
        # name → tier label of first occurrence.
        name_owner: dict[str, str] = {}
        offenders: list[str] = []

        def _scan(records: "Iterable[RegionAssignmentRecord]", tier: str) -> None:
            for rec in records:
                prior = name_owner.get(rec.name)
                if prior is not None and prior != tier:
                    offenders.append(
                        f"  • region name {rec.name!r} declared in "
                        f"{prior!r} AND in {tier!r}"
                    )
                else:
                    name_owner.setdefault(rec.name, tier)

        _scan(self.region_records, "global pool")
        for stage in self.stage_records:
            _scan(stage.region_records, f"stage {stage.name!r}")
        if offenders:
            preview = offenders[:10]
            extra = (
                f"\n  • ... and {len(offenders) - 10} more"
                if len(offenders) > 10 else ""
            )
            raise BridgeError(
                "Region ``name=`` collision across scopes — OpenSees "
                "allocates a separate tag per declaration "
                "(Domain::addRegion silently appends on duplicate "
                "tag), so two regions with the same user-facing name "
                "but different scopes silently produce two regions "
                "with different tags.  Mangle the name to make scope "
                "explicit (e.g. ``lining_rayleigh_stage2``):\n"
                + "\n".join(preview) + extra
            )

    def _recorder_node_targets(self, spec: "Recorder") -> "tuple[int, ...]":
        """Return the FEM node ids a recorder spec resolves to.

        Handles :class:`recorder.Node` (``pg`` / ``nodes``) and any
        :class:`recorder.FilterableRecorder` — MPCO / Ladruno —
        (``nodes_pg`` / ``nodes``).  Returns an empty tuple for recorder
        kinds that don't target nodes (e.g. :class:`recorder.Element`-only
        or :class:`RecorderDeclaration`).
        """
        from .recorder import FilterableRecorder as FilterableRec
        from .recorder import Node as NodeRec
        if isinstance(spec, NodeRec):
            if spec.pg is not None:
                return expand_pg_to_nodes(self.fem, spec.pg)
            return spec.nodes or ()
        if isinstance(spec, FilterableRec):
            if spec.nodes_pg is not None:
                return expand_pg_to_nodes(self.fem, spec.nodes_pg)
            return spec.nodes or ()
        return ()

    def _recorder_element_targets(
        self, spec: "Recorder",
    ) -> "tuple[int, ...]":
        """Return the FEM element ids a recorder spec resolves to.

        Handles :class:`recorder.Element` (``pg`` / ``elements``) and any
        :class:`recorder.FilterableRecorder` — MPCO / Ladruno —
        (``elements_pg`` / ``elements``).  Returns an empty tuple for
        recorder kinds that don't target elements.

        ``expand_pg_to_elements`` returns ``[(eid, conn), ...]`` —
        this helper extracts just the eid component.
        """
        from .recorder import Element as ElementRec
        from .recorder import FilterableRecorder as FilterableRec
        if isinstance(spec, ElementRec):
            if spec.pg is not None:
                return tuple(
                    int(eid) for eid, _conn in
                    expand_pg_to_elements(self.fem, spec.pg)
                )
            return spec.elements or ()
        if isinstance(spec, FilterableRec):
            if spec.elements_pg is not None:
                return tuple(
                    int(eid) for eid, _conn in
                    expand_pg_to_elements(self.fem, spec.elements_pg)
                )
            return spec.elements or ()
        return ()

    def _build_fem_eid_owner_stage_map(
        self,
        element_owner_stage: "dict[int, int]",
    ) -> "dict[int, int]":
        """Invert ``element_owner_stage`` (keyed by ``id(Element)``) into
        a ``fem_eid → stage_index`` map for V4's element-target checks.

        Walks ``self.primitives``, picks out registered Element
        specs, and for each spec whose ``id(...)`` lives in
        ``element_owner_stage``, expands its ``pg=`` to FEM eids and
        maps each eid to the owning stage index.

        Globally-emitted elements are absent from the map.
        """
        out: dict[int, int] = {}
        for spec in self.primitives:
            if not isinstance(spec, Element):
                continue
            sidx = element_owner_stage.get(id(spec))
            if sidx is None:
                continue
            pg = getattr(spec, "pg", None)
            if not pg:
                continue
            for eid, _conn in expand_pg_to_elements(self.fem, pg):
                out[int(eid)] = sidx
        return out

    def _validate_stage_bound_recorder_targets(
        self,
        node_owner_stage: "dict[int, int]",
        element_owner_stage: "dict[int, int]",
    ) -> None:
        """V4: stage N's recorders may target only globally-emitted
        topology or topology owned by stage M <= N.

        Recorder lines (``recorder Node ...`` / ``Element ...`` /
        ``mpco ...``) parse at deck-read time and bind to the
        topology that exists at that point.  A stage-N recorder
        targeting a node owned by stage M > N would reference a not-
        yet-emitted node and crash OpenSees at parse time.

        Same ownership-tier rule as V1 / H1 but applied to:

        - :class:`recorder.Node` (node targets)
        - :class:`recorder.Element` (element targets)
        - :class:`recorder.MPCO` (both node and element targets)

        Builds the ``fem_eid → stage_index`` reverse map ad-hoc since
        ``compute_stage_ownership`` returns ``element_owner_stage``
        keyed by ``id(spec)``, not by FEM eid.

        :class:`RecorderDeclaration` instances are silently passed —
        Phase 9's declaration shape doesn't carry a direct
        ``pg`` / ``nodes`` / ``elements`` selector validated here.
        """
        if not self.stage_records:
            return
        fem_eid_to_stage = self._build_fem_eid_owner_stage_map(
            element_owner_stage,
        )
        offenders_per_stage: "list[tuple[str, list[str]]]" = []
        for stage_idx, stage in enumerate(self.stage_records):
            if not stage.recorder_specs:
                continue
            stage_offenders: list[str] = []
            for spec in stage.recorder_specs:
                spec_label = type(spec).__name__
                # Node targets.
                for nid in self._recorder_node_targets(spec):
                    owner = node_owner_stage.get(int(nid))
                    if owner is not None and owner > stage_idx:
                        owner_name = self.stage_records[owner].name
                        stage_offenders.append(
                            f"  • {spec_label} recorder targets node "
                            f"{nid} → owned by LATER stage {owner_name!r} "
                            f"(index {owner})"
                        )
                # Element targets.
                for eid in self._recorder_element_targets(spec):
                    owner = fem_eid_to_stage.get(int(eid))
                    if owner is not None and owner > stage_idx:
                        owner_name = self.stage_records[owner].name
                        stage_offenders.append(
                            f"  • {spec_label} recorder targets element "
                            f"{eid} → owned by LATER stage {owner_name!r} "
                            f"(index {owner})"
                        )
            if stage_offenders:
                offenders_per_stage.append((stage.name, stage_offenders))
        if not offenders_per_stage:
            return
        chunks: list[str] = []
        for stage_name, lines in offenders_per_stage:
            preview = lines[:10]
            extra = (
                f"\n  • ... and {len(lines) - 10} more"
                if len(lines) > 10 else ""
            )
            chunks.append(
                f"Stage {stage_name!r} recorders reference topology "
                f"owned by a LATER stage:\n" + "\n".join(preview) + extra
            )
        raise BridgeError(
            "Stage-bound recorders reference topology that only "
            "comes online in a later stage — the ``recorder`` line "
            "parses at deck-read time and would bind to non-existent "
            "nodes/elements, crashing OpenSees at parse time.  Move "
            "the recorder onto a later-stage ``s.recorder(...)`` "
            "binding, or use globally-emitted targets only."
            "\n\n" + "\n\n".join(chunks)
        )

    # -- V5 / V6 — Phase SSI-2.E removal-target validators ---------------

    def _validate_remove_sp_targets(self) -> None:
        """V5: every ``s.remove_sp`` target must reference an SP that
        is alive at the point where the ``remove sp`` line emits.

        An SP is "alive" at the start of stage N if:

        * declared in the global ``apeSees.fix`` pool, OR
        * declared in a strictly-earlier stage's ``s.fix`` pool,

        AND it was not removed by a strictly-earlier stage's
        ``s.remove_sp``.  Same-stage ``s.fix`` does NOT count — fix
        emits AFTER remove_sp in the same stage's block, so the SP
        does not yet exist when remove_sp parses.

        Within a single stage's ``remove_sp_records`` list, the same
        ``(node, dof)`` may not appear twice (the second emit would
        target an already-removed SP).
        """
        if not self.stage_records:
            return
        # (node, DOF_1based) currently in the OpenSees Domain as an SP.
        alive: dict[tuple[int, int], str] = {}

        # Seed with the global pool's SPs (fix dofs with flag==1).
        for rec in self.fix_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                for dof_idx, flag in enumerate(rec.dofs, start=1):
                    if flag:
                        alive[(int(node_tag), dof_idx)] = "global pool"

        offenders_per_stage: "list[tuple[str, list[str]]]" = []
        for stage in self.stage_records:
            stage_offenders: list[str] = []
            for sp_rem_rec in stage.remove_sp_records:
                for node_tag in self._resolve_node_target(sp_rem_rec.pg, sp_rem_rec.nodes):
                    for dof in sp_rem_rec.dofs:
                        key = (int(node_tag), int(dof))
                        prior = alive.pop(key, None)
                        if prior is None:
                            stage_offenders.append(
                                f"  • s.remove_sp targets node "
                                f"{node_tag} DOF {dof} — no active SP "
                                "at this point (not declared in an "
                                "earlier scope, or already removed by "
                                "an earlier stage / earlier record)"
                            )
            if stage_offenders:
                offenders_per_stage.append((stage.name, stage_offenders))
            # After this stage's removals, its own fix records add to
            # the alive set so later stages see them.  Same-stage fix
            # does NOT count toward same-stage removal (see seed
            # comment above).
            for rec in stage.fix_records:
                for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                    for dof_idx, flag in enumerate(rec.dofs, start=1):
                        if flag:
                            alive[(int(node_tag), dof_idx)] = (
                                f"stage {stage.name!r}"
                            )
        if not offenders_per_stage:
            return
        chunks: list[str] = []
        for stage_name, lines in offenders_per_stage:
            preview = lines[:10]
            extra = (
                f"\n  • ... and {len(lines) - 10} more"
                if len(lines) > 10 else ""
            )
            chunks.append(
                f"Stage {stage_name!r} s.remove_sp targets:\n"
                + "\n".join(preview) + extra
            )
        raise BridgeError(
            "Stage-bound s.remove_sp targets an SP that doesn't exist "
            "at the point of removal — OpenSees ``remove sp`` would "
            "error at parse time.  Either declare the SP in the "
            "global ``ops.fix`` pool or in a strictly-earlier stage's "
            "``s.fix(...)``, or drop the s.remove_sp call.  Same-stage "
            "``s.fix`` does NOT count: fix emits AFTER remove_sp in "
            "the same stage block."
            "\n\n" + "\n\n".join(chunks)
        )

    def _validate_remove_element_targets(
        self,
        element_owner_stage: "dict[int, int]",
        fem_eid_to_ops_tag: "FemToOpsTagMap",
    ) -> None:
        """V6: every ``s.remove_element`` target must reference an
        element that is alive at the point where the ``remove element``
        line emits.

        Validates in FEM-eid space (matching the recorder convention —
        :class:`recorder.Element` also takes ``elements=[fem_eid, ...]``
        from the user and translates to OpenSees tags at emit time).

        An element is "alive" at the point a stage's removal block
        emits if it was emitted in an earlier deck position AND not
        previously removed.  Earlier positions are:

        * the global pre-stage element fan-out (specs NOT in
          ``element_owner_stage``), OR
        * a strictly-earlier stage's activation, OR
        * this same stage's activation (activation emits BEFORE the
          removal block within the stage).

        Within a single stage's ``remove_element_records`` list, the
        same FEM eid may not appear twice.
        """
        if not self.stage_records:
            return
        # fem_eid → tier label currently alive in the Domain.
        alive_fem: dict[int, str] = {}

        # Build stage_idx → list[fem_eid] for stage-activated elements.
        stage_activated_fem: dict[int, list[int]] = {}
        for spec in self.primitives:
            if not isinstance(spec, Element):
                continue
            pg = getattr(spec, "pg", None)
            if not pg:
                continue
            sidx = element_owner_stage.get(id(spec))
            if sidx is None:
                # Globally-emitted spec — alive from the start.
                for fem_eid, _conn in expand_pg_to_elements(self.fem, pg):
                    # ``fem_eid_to_ops_tag`` is a sanity check the
                    # element survived tag allocation; the alive set
                    # itself is keyed by fem_eid.
                    if int(fem_eid) in fem_eid_to_ops_tag:
                        alive_fem[int(fem_eid)] = "global pool"
            else:
                for fem_eid, _conn in expand_pg_to_elements(self.fem, pg):
                    if int(fem_eid) in fem_eid_to_ops_tag:
                        stage_activated_fem.setdefault(sidx, []).append(
                            int(fem_eid),
                        )

        offenders_per_stage: "list[tuple[str, list[str]]]" = []
        for stage_idx, stage in enumerate(self.stage_records):
            # Stage activation emits BEFORE the removal block in the
            # same stage, so a stage may remove what it just activated.
            for fem_eid in stage_activated_fem.get(stage_idx, []):
                alive_fem[fem_eid] = f"stage {stage.name!r}"
            stage_offenders: list[str] = []
            for rec in stage.remove_element_records:
                fem_eids_to_check: list[int] = []
                if rec.pg is not None:
                    try:
                        for fem_eid, _conn in expand_pg_to_elements(
                            self.fem, rec.pg,
                        ):
                            fem_eids_to_check.append(int(fem_eid))
                    except BridgeError:
                        stage_offenders.append(
                            f"  • s.remove_element pg={rec.pg!r} — "
                            "physical group not found on the FEM "
                            "snapshot"
                        )
                        continue
                elif rec.elements is not None:
                    fem_eids_to_check.extend(int(t) for t in rec.elements)
                for fem_eid in fem_eids_to_check:
                    prior = alive_fem.pop(fem_eid, None)
                    if prior is None:
                        stage_offenders.append(
                            f"  • s.remove_element targets FEM eid "
                            f"{fem_eid} — no active element at this "
                            "point (not declared in an earlier scope, "
                            "not activated by this stage or an earlier "
                            "stage, or already removed by an earlier "
                            "stage / earlier record)"
                        )
            if stage_offenders:
                offenders_per_stage.append((stage.name, stage_offenders))
        if not offenders_per_stage:
            return
        chunks: list[str] = []
        for stage_name, lines in offenders_per_stage:
            preview = lines[:10]
            extra = (
                f"\n  • ... and {len(lines) - 10} more"
                if len(lines) > 10 else ""
            )
            chunks.append(
                f"Stage {stage_name!r} s.remove_element targets:\n"
                + "\n".join(preview) + extra
            )
        raise BridgeError(
            "Stage-bound s.remove_element targets an element that "
            "doesn't exist at the point of removal — OpenSees "
            "``remove element`` would error at runtime.  Either "
            "declare the element globally / via an earlier stage's "
            "``s.activate(pgs=...)``, or drop the s.remove_element "
            "call."
            "\n\n" + "\n\n".join(chunks)
        )

    def _validate_material_stage_targets(
        self,
        element_owner_stage: "dict[int, int]",
    ) -> None:
        """V7: every ``s.update_material_stage`` target must be used by
        an element that is LIVE in the Domain where the
        ``updateMaterialStage`` line emits.

        ``MaterialStageParameter::setDomain()`` walks the Domain's
        elements looking for the material tag; on a miss it prints
        ``no effect with material tag N``, the command still reports
        success, and the flip silently does nothing.  This validator
        turns that into a build-time error.

        An element is live at the flip if it emitted in the global
        pre-stage fan-out (spec NOT in ``element_owner_stage``) or was
        activated by this stage or a strictly-earlier one — the flip
        block emits after the stage's own element fan-out.

        Known gap: ``s.remove_element`` is NOT modelled here.  A deck
        that removes every element using a material and then flips that
        material still passes V7 and silently no-ops at run time.  That
        needs a per-stage live-set walk rather than this earliest-live
        map; it is not worth the machinery until a real deck hits it.
        """
        if not self.stage_records:
            return
        if not any(
            stage.update_material_stage_records
            for stage in self.stage_records
        ):
            return
        # nDMaterial tag → earliest stage index at which some element
        # using it is live.  -1 == globally emitted (live from line 1).
        first_live: dict[int, int] = {}
        for spec in self.primitives:
            if not isinstance(spec, Element):
                continue
            sidx = element_owner_stage.get(id(spec), -1)
            # Walk the spec's dependency closure — an ND material may
            # sit directly on the element or behind a section wrapper.
            seen: set[int] = set()
            frontier: list[Primitive] = list(spec.dependencies())
            while frontier:
                dep = frontier.pop()
                if id(dep) in seen:
                    continue
                seen.add(id(dep))
                frontier.extend(dep.dependencies())
                if not isinstance(dep, NDMaterial):
                    continue
                mat_tag = self.tag_for.get(id(dep))
                if mat_tag is None:
                    continue
                prior = first_live.get(int(mat_tag))
                if prior is None or sidx < prior:
                    first_live[int(mat_tag)] = sidx
        offenders: list[str] = []
        for stage_idx, stage in enumerate(self.stage_records):
            for rec in stage.update_material_stage_records:
                for mat_tag in rec.mat_tags:
                    live_at = first_live.get(int(mat_tag))
                    if live_at is not None and live_at <= stage_idx:
                        continue
                    offenders.append(
                        f"  • stage {stage.name!r} flips nDMaterial tag "
                        f"{mat_tag} to stage {rec.stage}, but no element "
                        "using that material is live in the Domain at "
                        "that point"
                        + (
                            " (it is not used by any element)"
                            if live_at is None else
                            f" (its elements are activated later, by "
                            f"stage "
                            f"{self.stage_records[live_at].name!r})"
                        )
                    )
        if not offenders:
            return
        preview = offenders[:10]
        extra = (
            f"\n  • ... and {len(offenders) - 10} more"
            if len(offenders) > 10 else ""
        )
        raise BridgeError(
            "s.update_material_stage targets a material with no live "
            "element — OpenSees resolves updateMaterialStage through "
            "the Domain's elements, so the flip would print "
            "``no effect with material tag N`` and silently do "
            "nothing.  Move the call to a stage at or after the one "
            "that activates the material's elements."
            "\n\n" + "\n".join(preview) + extra
        )

    def _emit_fixes(
        self, emitter: Emitter,
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> None:
        eff = inferred_ndf or {}
        for rec in self.fix_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                node = int(node_tag)
                emitter.fix(node, *fit_fix_mask(
                    rec.dofs, int(eff.get(node, self.ndf))))

    def _guard_mass_from_model(self, emitter: Emitter) -> bool:
        """Validate a ``mass_from_model`` emit (ADR 0065 Tier 2).

        Returns True when the caller should stream the snapshot masses,
        False when it should not: there are none (a no-op declaration), or
        the emitter is the H5 archival emitter. The masses already persist
        in ``model.h5``'s neutral zone (``/masses``), so the archive skips
        the redundant stream, which would rebuild the 7M-object
        materialization the feature exists to avoid, and instead marks
        ``/opensees/bcs@mass_from_model`` so replay re-streams them (ADR
        0112 amendment 5, #1304).

        Raises ``BridgeError`` on every emitter, the H5 one included, if any
        node carries both a snapshot mass and an explicit ``ops.mass``
        (additive under MP assembly, so a silent double-count).
        """
        masses = getattr(self.fem.nodes, "masses", None)
        if not masses:
            return False
        if self.mass_records:
            explicit = {
                int(n)
                for rec in self.mass_records
                for n in self._resolve_node_target(rec.pg, rec.nodes)
            }
            overlap = sorted(
                int(m.node_id) for m in masses if int(m.node_id) in explicit
            )
            if overlap:
                raise BridgeError(
                    "mass_from_model() and explicit ops.mass(...) both target "
                    f"node(s) {overlap[:5]} (+{max(0, len(overlap) - 5)} more) "
                    "— nodal mass is additive under MP assembly, so emitting "
                    "both would double-count. Use exactly one mass channel."
                )
        from .emitter.h5 import H5Emitter
        if isinstance(emitter, H5Emitter):
            # Checked after the overlap guard, which holds on every emitter.
            emitter.mark_mass_from_model()
            return False
        return True

    def _emit_masses(
        self, emitter: Emitter,
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> None:
        eff = inferred_ndf or {}
        for rec in self.mass_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                node = int(node_tag)
                emitter.mass(node, *fit_dof_vector(
                    rec.values, int(eff.get(node, self.ndf)),
                    kind="mass", node=node))
        if self.mass_from_model and self._guard_mass_from_model(emitter):
            for m in self.fem.nodes.masses:
                nid = int(m.node_id)
                emitter.mass(nid, *broker_mass_components(
                    m.mass, int(eff.get(nid, self.ndf)),
                    self.ndm, node=nid))

    def _emit_fixes_partitioned(
        self, emitter: Emitter, owned_nodes: set[int],
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> None:
        """Per-rank fix fan-out (ADR 0027).

        Only emit ``fix`` lines for nodes owned by this rank.  A fix on
        a non-owned node is silently skipped on this rank's block —
        the owning rank handles it.
        """
        eff = inferred_ndf or {}
        for rec in self.fix_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if int(node_tag) in owned_nodes:
                    node = int(node_tag)
                    emitter.fix(node, *fit_fix_mask(
                        rec.dofs, int(eff.get(node, self.ndf))))

    def _emit_masses_partitioned(
        self, emitter: Emitter, owned_nodes: set[int],
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> None:
        """Per-rank mass fan-out (ADR 0027).

        Unlike :meth:`_emit_fixes_partitioned` (idempotent ``fix`` lines,
        replicated on every owning rank), nodal mass is ADDITIVE under
        OpenSeesMP assembly — callers must pass the rank's
        **primary-owned** node set (each node on exactly one rank; see
        :func:`primary_owner_map`), not the full per-rank node set, or
        shared interface nodes carry their mass once per owning rank.
        """
        eff = inferred_ndf or {}
        for rec in self.mass_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if int(node_tag) in owned_nodes:
                    node = int(node_tag)
                    emitter.mass(node, *fit_dof_vector(
                        rec.values, int(eff.get(node, self.ndf)),
                        kind="mass", node=node))
        if self.mass_from_model and self._guard_mass_from_model(emitter):
            for m in self.fem.nodes.masses:
                nid = int(m.node_id)
                if nid in owned_nodes:
                    emitter.mass(nid, *broker_mass_components(
                        m.mass, int(eff.get(nid, self.ndf)),
                        self.ndm, node=nid))

    def _bucket_fix_targets_by_rank(
        self, node_owners: "NodePartitionOwners", *, by_node: bool = False,
    ) -> "tuple[dict[int, list[tuple[FixRecord, list[int]]]], dict[int, list[Any]]]":
        """Resolve every global fix record's targets ONCE and bucket by rank.

        Replaces the per-rank :meth:`_emit_fixes_partitioned` pass in the
        base partitioned loop, whose ``_resolve_node_target`` call re-ran
        the full PG → nodes broker expansion once per rank —
        O(records × nodes × ranks).  Fixes replicate on EVERY owning
        rank (idempotent lines, ADR 0027 INV-4 fan-out), so a node
        lands in each of its owners' buckets.  Record order and
        within-record node order are preserved per rank, keeping the
        emitted deck byte-identical.

        Returns ``(by_rank, by_node)``:

        * ``by_rank`` — ``{rank: [(record, [node ids]), ...]}``, what the
          per-rank fix pass emits.
        * ``by_node`` — ``{node id: [("fix", dofs), ...]}`` in record
          order, the **rank-independent** view the foreign-node (ghost)
          declarations need (ADR 0027 INV-2). A ghost is by definition
          NOT owned by the declaring rank, so it never appears in that
          rank's ``by_rank`` bucket and its BCs would otherwise be missing
          there — see :func:`~apeGmsh.opensees._internal.build.emit_mp_constraints_partitioned`.
          Tagged ``("fix", …)`` because a staged model extends the same
          stream with ``("remove", dof)`` entries per stage.
          Built inside this same walk rather than by a second
          ``_resolve_node_target`` pass, which would double the
          O(records × nodes) broker expansion this method exists to
          avoid.

        ``by_node=False`` (the default) returns it empty. Only a model
        with MP constraints can declare a ghost, and on a large model
        this map is one dict entry per FIXED node — the kind of
        per-node boxing ADR 0065 took out of the emit peak, so it is
        not built speculatively.
        """
        out: "dict[int, list[tuple[FixRecord, list[int]]]]" = {}
        fix_by_node: "dict[int, list[Any]]" = {}
        for rec in self.fix_records:
            per_rank: "dict[int, list[int]]" = {}
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                nid = int(node_tag)
                if by_node:
                    fix_by_node.setdefault(nid, []).append(("fix", rec.dofs))
                for rank in node_owners.get(nid, ()):
                    per_rank.setdefault(rank, []).append(nid)
            for rank, nodes_list in per_rank.items():
                out.setdefault(rank, []).append((rec, nodes_list))
        return out, fix_by_node

    def _bucket_mass_targets_by_rank(
        self, primary_owner: "SortedIntToInt",
    ) -> "dict[int, list[tuple[MassRecord, list[int]]]]":
        """Resolve every global mass record's targets ONCE and bucket by rank.

        Mass counterpart of :meth:`_bucket_fix_targets_by_rank` — nodal
        mass is ADDITIVE under OpenSeesMP assembly, so each node lands
        in its PRIMARY owner's bucket only (see :func:`primary_owner_map`).
        """
        out: "dict[int, list[tuple[MassRecord, list[int]]]]" = {}
        for rec in self.mass_records:
            per_rank: "dict[int, list[int]]" = {}
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                nid = int(node_tag)
                rank = primary_owner.get(nid)
                if rank is not None:
                    per_rank.setdefault(rank, []).append(nid)
            for rank, nodes_list in per_rank.items():
                out.setdefault(rank, []).append((rec, nodes_list))
        return out

    # -- Partitioned contact (ADR 0092 S4) --------------------------------

    def _plan_partitioned_contacts(
        self,
        partitions: "list[Any]",
        *,
        staged: bool,
    ) -> "tuple[dict[int, list[tuple[str, Any, tuple[int, ...]]]], tuple[str, ...]]":
        """Resolve owner rank + ghost set for every contact interaction
        (ADR 0092 S4, INV-1/INV-2), or refuse with a NAMED error (INV-5).

        Returns ``({owner_rank: [(kind, record, ghost_node_ids), ...]},
        notes)`` with ``kind`` in ``{"contact", "contact_plane"}`` — what
        step 7c of the per-rank loop emits inside the owner's block —
        and the warnings the routing raises, which every partitioned
        emit repeats. Empty when the model carries no contact
        interactions.

        The build's tag plan calls this once per plan
        (``tag_plan.plan_tags``, ADR 0114 D4 amended), and numbers the
        routed interactions rank by rank; the emit reads the routing
        from the plan. The pattern-borne ``sp`` sweep over the routed
        ghosts (2026-08-13 review F1) needs the emit's own ndf map and
        patterns, so :meth:`_emit_partitioned` runs it, still before any
        emission.

        Owner exactness (INV-1, second amendment): where the mesh's
        element connectivity resolves each master facet to its backing
        solid (:func:`master_backing_element_ids`) and every backing
        element is partition-owned, the ranks of those backing solids are
        passed to the resolver and the pick is **exact** — the node-tally
        proxy (and its undecidable-tie refusal) never engages. Where the
        facet → element map is not resolvable (stub FEMs without global
        connectivity, non-conforming patches), the resolver falls back to
        the unique-owner node tally and refuses the genuinely undecidable
        tie rather than guessing.

        Named refusals raised here (all before ANY emission):

        * **undecidable owner** — the resolver's tie refusal, wrapped
          with the interaction's index + name;
        * **cut master surface + auto-sizing** — backing solids straddle
          ranks while ``kn``/``eps_n``/``eps_t``/``edge_kn`` is
          ``"auto"``: the fork resolves the owning solid of each master
          segment on the emitting rank, so off-rank backing silently
          skips those facets' auto penalty (INV-4 / fork ADR-78 D5.2);
        * **partially-resolvable backing under auto-sizing** (2026-08-13
          review, F2/F3) — some facet's backing solid cannot be
          identified (ambiguous facet → element resolution, or the
          element is absent from every ``PartitionRecord``) while an
          auto knob is active AND the master's nodes span several ranks
          in the node view: the unresolved facets could hide exactly the
          off-rank backing the previous refusal exists for, so falling
          back to the node tally would re-open the silent-partial-
          interface hole. A partial element→rank ownership map alone
          (no auto knob, or a one-rank master) degrades to the tally
          with a loud warning instead (a note, which each emit warns);
        * **staged model** — the partitioned staged pipeline skips the
          analysis-chain auto-emit (each stage carries its own chain), so
          the forced ``LadrunoContact`` handler would never be emitted
          and the contact would be silently unenforced; contact-ghost SP
          sync across stage blocks is likewise unproven. Deferred, not
          designed around.

        No ghost-cost bound is enforced: the 2026-08-11 adversarial
        review WITHDREW the ghost budget (ADR 0092 §Sign-off Q2 — ~50 B
        per ghost line puts a 10^4-node interface at ~0.2% of a 100^3
        deck; the real cost is the runtime shared-DOF gather, measured at
        fork ADR-78 P4, not deck text).
        """
        elements_comp = getattr(self.fem, "elements", None)
        contacts = list(getattr(elements_comp, "contacts", None) or ())
        planes = list(getattr(elements_comp, "contact_planes", None) or ())
        if not contacts and not planes:
            return {}, ()
        element_owner = build_element_partition_owner(self.fem)
        notes: "list[str]" = []

        from apeGmsh._kernel.resolvers._contact_ownership import (
            master_backing_element_ids,
            master_node_rank_span,
            resolve_contact_ownership,
        )

        if staged:
            raise BridgeError(
                "apeSees: g.constraints.contact / contact_plane "
                "interactions are not supported on a STAGED partitioned "
                "model (ADR 0092 S4) — the partitioned staged pipeline "
                "skips the analysis-chain auto-emit (each stage declares "
                "its own chain), so the 'LadrunoContact' handler contact "
                "requires would never be emitted and the interaction "
                "would be silently unenforced. Emit the contact model "
                "unstaged, or serial (non-partitioned / flat=True)."
            )

        # 2D contact under MPI: parallel / DDM 2D contact is explicitly
        # OUT OF SCOPE fork-side (fork ADR-85), so the ownership resolver's
        # generic `record.master_nps` reshape would produce a plan the
        # engine cannot honour. Refuse by name here, before any emission —
        # same idiom as the `soft=` INV-3 refusal above.
        for idx, rec in enumerate(contacts, start=1):
            if int(getattr(rec, "master_nps", 0)) != 2:
                continue
            label = f"#{idx}" + (
                f" ({rec.name!r})" if getattr(rec, "name", None) else ""
            )
            raise BridgeError(
                f"apeSees: g.constraints.contact interaction {label} is a "
                f"2D line-segment contact (master_nps=2), and 2D contact is "
                f"not supported under partitioned (MPI) emit — parallel / "
                f"DDM 2D contact is explicitly out of scope in the fork's "
                f"own ADR-85. Emit the 2D contact model serial "
                f"(non-partitioned / flat=True)."
            )

        # The rigid-plane lane needs its OWN gate, not an extension of the
        # loop above: `ContactPlaneRecord` carries no `master_nps`, so the
        # 2D discriminator that works for `contact` does not exist here and
        # the dimension has to come from the model. Without this a 2D
        # `contact_plane` fell straight through into the generic ownership
        # path — the exact silent kind of pass-through the refusal above is
        # there to prevent, and the same fork scope applies to both lanes.
        if planes and int(self.ndm) == 2:
            idx, rec = 1, planes[0]
            label = f"#{idx}" + (
                f" ({rec.name!r})" if getattr(rec, "name", None) else ""
            )
            raise BridgeError(
                f"apeSees: g.constraints.contact_plane interaction {label} "
                f"is in a 2D model (ndm=2), and 2D contact is not supported "
                f"under partitioned (MPI) emit — parallel / DDM 2D contact "
                f"is explicitly out of scope in the fork's own ADR-85, for "
                f"the rigid-plane lane as much as the NTS one. Emit the 2D "
                f"model serial (non-partitioned / flat=True)."
            )

        # The mesh's element connectivity, if this FEM snapshot exposes
        # it (real FEMData iterates ElementGroup objects; hand-rolled
        # test stubs may not) — the facet -> backing-solid input that
        # makes the owner pick exact.
        element_groups: "list[tuple[Any, Any]]" = []
        try:
            for grp in elements_comp or ():
                ids = getattr(grp, "ids", None)
                conn = getattr(grp, "connectivity", None)
                if ids is not None and conn is not None:
                    element_groups.append((ids, conn))
        except TypeError:
            element_groups = []

        plan: "dict[int, list[tuple[str, Any, tuple[int, ...]]]]" = {}
        for verb, kind, recs in (
            ("g.constraints.contact", "contact", contacts),
            ("g.constraints.contact_plane", "contact_plane", planes),
        ):
            for idx, rec in enumerate(recs, start=1):
                label = f"#{idx}" + (
                    f" ({rec.name!r})" if getattr(rec, "name", None) else ""
                )
                auto_knobs = [
                    knob for knob in
                    ("kn", "eps_n", "eps_t", "edge_kn")
                    if getattr(rec, knob, None) == "auto"
                ]
                backing_ranks: "tuple[int, ...] | None" = None
                unresolved_facets = 0   # no unique backing element (F2)
                unowned_backing = 0     # element in no PartitionRecord (F3)
                if kind == "contact" and element_groups:
                    backing = master_backing_element_ids(rec, element_groups)
                    if backing is not None:
                        resolved_ranks: "list[int]" = []
                        for eid in backing:
                            if eid is None:
                                unresolved_facets += 1
                                continue
                            rank = element_owner.get(int(eid))
                            if rank is None:
                                unowned_backing += 1
                                continue
                            resolved_ranks.append(int(rank))
                        if resolved_ranks:
                            backing_ranks = tuple(resolved_ranks)
                if (backing_ranks and len(set(backing_ranks)) > 1
                        and auto_knobs):
                    raise BridgeError(
                        f"apeSees: {verb} interaction {label} — the "
                        "partitioner CUT the master surface (its "
                        "backing solid elements straddle ranks "
                        f"{sorted(set(backing_ranks))}) while "
                        f"{' + '.join(k + '=' for k in auto_knobs)} is "
                        "'auto' (ADR 0092 INV-4). The fork resolves "
                        "the owning solid of each master segment on "
                        "the emitting rank, so facets whose backing "
                        "solid lives off-rank would silently SKIP the "
                        "auto penalty sizing (fork ADR-78 D5.2). "
                        "Re-partition with the master surface's "
                        "backing elements declared uncuttable "
                        "(g.mesh.partitioning.partition(n, "
                        "uncuttable_elements=...)), or use an "
                        "explicit numeric penalty."
                    )
                # 2026-08-13 review F2/F3: the INV-4 backstop above runs
                # on the RESOLVED facet subset. Where the map is only
                # partial (an ambiguous facet, or a backing element
                # missing from every PartitionRecord) AND an auto knob
                # is live AND the master's nodes span several ranks,
                # the unresolved facets could hide exactly the off-rank
                # backing that backstop exists to catch — refuse instead
                # of silently degrading to the node tally.
                if (unresolved_facets or unowned_backing) and auto_knobs:
                    span = master_node_rank_span(rec, partitions)
                    if len(span) > 1:
                        raise BridgeError(
                            f"apeSees: {verb} interaction {label} — "
                            f"{unresolved_facets + unowned_backing} master "
                            "facet(s) cannot be traced to a "
                            "partition-owned backing solid element "
                            f"({unresolved_facets} ambiguous/uncovered in "
                            "the facet-to-element map, "
                            f"{unowned_backing} whose backing element is "
                            "absent from every PartitionRecord) while "
                            f"{' + '.join(k + '=' for k in auto_knobs)} "
                            "is 'auto' and the master surface's nodes "
                            f"span ranks {list(span)} (ADR 0092 INV-4). "
                            "The cut-master + auto-sizing backstop "
                            "cannot verify that every facet's backing "
                            "solid is owner-rank-local, and the fork "
                            "silently SKIPS auto penalty sizing on "
                            "facets whose backing solid lives off-rank "
                            "(fork ADR-78 D5.2) — a silently PARTIAL "
                            "interface. Use an explicit numeric penalty, "
                            "or re-partition with the master surface's "
                            "backing elements declared uncuttable "
                            "(g.mesh.partitioning.partition(n, "
                            "uncuttable_elements=...))."
                        )
                if unowned_backing:
                    # F3: a partial element→rank ownership map is never
                    # silent — the exact owner pick quietly degrading to
                    # the node tally is how a mis-owned interaction slips
                    # through. (With an auto knob + multi-rank master it
                    # refused above instead.) The routing runs once per
                    # tag plan, so the warning is a note that every
                    # partitioned emit repeats (:meth:`_emit_partitioned`).
                    notes.append(
                        f"apeSees: {verb} interaction {label} — "
                        f"{unowned_backing} master facet backing "
                        "element(s) are absent from every "
                        "PartitionRecord (partial element-ownership "
                        "map); the element-exact owner pick (ADR 0092 "
                        "INV-1) degrades to the node tally for this "
                        "interaction."
                    )
                try:
                    ownership = resolve_contact_ownership(
                        rec, partitions,
                        master_element_ranks=backing_ranks,
                    )
                except ValueError as exc:
                    raise BridgeError(
                        f"apeSees: {verb} interaction {label} cannot be "
                        "emitted under partitioned (MPI) emit — "
                        f"{exc} (ADR 0092 INV-1: each interaction needs "
                        "exactly one owner rank; emitting on two ranks "
                        "converges to a plausible WRONG answer — half "
                        "the penetration — with no warning, fork ADR-78 "
                        "P0.d). Re-partition so one rank owns the "
                        "interaction's deciding surface — the master "
                        "surface, or the slave surface for a "
                        "contact_plane — (g.mesh.partitioning."
                        "partition(n, uncuttable_elements=...)), or emit "
                        "serial (non-partitioned / flat=True)."
                    ) from exc
                plan.setdefault(ownership.owner_rank, []).append(
                    (kind, rec, ownership.ghost_node_ids),
                )
        return plan, tuple(notes)

    def _emit_contacts_partitioned(
        self,
        emitter: Emitter,
        lines: "list[PlannedContact]",
        *,
        declared_ghosts: "set[int]",
        ghost_sp_ops: "dict[int, list[Any]]",
        inferred_ndf: "dict[int, int]",
        node_idx_lookup: "SortedIntToInt",
    ) -> None:
        """Emit this rank's owned contact interactions (ADR 0092 S4).

        For each planned line this rank owns (the tag plan's routing, in
        plan order): first declare every ghost node — ``node tag x y z``
        with the same inferred/envelope ndf its owner emits, immediately
        followed by the owner's replayed SP stream (ADR 0027 INV-2
        machinery) — then the ``contactSurface`` pair + ``contact`` /
        ``contactPlane`` verb under the line's planned tags
        (:func:`write_planned_contact`).

        INV-7 holds structurally: a ghost gets a ``node`` line + ``fix``
        replay and NOTHING else — mass buckets by primary owner, loads by
        primary owner, elements by element owner, all of which point at
        the ghost's native rank, never here.

        ``declared_ghosts`` is this rank's live ghost registry
        (``ghost_tags_by_rank[rank]``, already holding step 7b's
        MP-constraint ghosts): a node already declared is not re-declared
        (a duplicate ``node`` line is an OpenSees parse error), and every
        ghost declared here is registered back into it.
        """
        for line in lines:
            for nid_raw in line.ghost_node_ids:
                nid = int(nid_raw)
                if nid in declared_ghosts:
                    continue
                node_idx = node_idx_lookup.get(nid)
                if node_idx is None:
                    raise BridgeError(
                        f"apeSees: contact interface node {nid} is not in "
                        "the FEM snapshot — cannot ghost-declare it on the "
                        "interaction's owner rank (ADR 0092 INV-2). The "
                        "contact surface references a node the mesh does "
                        "not carry."
                    )
                xyz = self.fem.nodes.coords[node_idx]
                _emit_node_with_inferred_ndf(
                    emitter, inferred_ndf, nid,
                    (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    self.ndf,
                )
                emit_ghost_sp_ops(
                    emitter, nid, ghost_sp_ops.get(nid, ()),
                )
                declared_ghosts.add(nid)
            write_planned_contact(emitter, line, ndm=self.ndm)

    def _plan_partitioned_reinforcement(
        self, node_owners: "Any",
    ) -> "dict[int, tuple[list[Any], list[Any], tuple[int, ...]]]":
        """Owner rank of every embedded-reinforcement item under a
        partitioned emit: ``{rank: (ties, bar_records, ghost_node_ids)}``.

        * A ``LadrunoEmbeddedRebar`` tie goes to the lowest rank that owns
          all its host nodes (the host element's rank). If that rank does
          not own the rebar node, the node is ghost-declared there, the
          same shared-node mechanism as ``ASDEmbeddedNodeElement``.
        * A bar ``CorotTruss`` cell goes to the lowest rank owning both of
          its nodes (a Gmsh-partitioned line cell always has one); a cell
          straddling ranks goes to its first node's rank and the second
          node is ghost-declared.

        Every item lands on exactly one rank, so nothing is duplicated.
        Empty when the model has no embedded reinforcement.
        """
        from dataclasses import replace

        elements = getattr(self.fem, "elements", None)
        ties = list(getattr(elements, "reinforce_ties", None) or ())
        bars = list(getattr(elements, "rebar_elements", None) or ())
        if not ties and not bars:
            return {}

        def ranks(nid: int) -> "set[int]":
            owners = node_owners.get(int(nid))
            if not owners:
                raise BridgeError(
                    f"apeSees: embedded-reinforcement node {nid} is not "
                    "owned by any partition; partition the mesh after the "
                    "bars are meshed."
                )
            return set(int(r) for r in owners)

        tie_plan: "dict[int, list[Any]]" = {}
        bar_plan: "dict[int, dict[int, list[tuple[int, int]]]]" = {}
        ghosts: "dict[int, set[int]]" = {}

        for rec in ties:
            common = set.intersection(*(ranks(h) for h in rec.host_nodes))
            if not common:
                raise BridgeError(
                    f"apeSees: the host nodes of the tie at rebar node "
                    f"{rec.rebar_node} share no partition; cannot place "
                    "the LadrunoEmbeddedRebar element."
                )
            owner = min(common)
            tie_plan.setdefault(owner, []).append(rec)
            if owner not in ranks(rec.rebar_node):
                ghosts.setdefault(owner, set()).add(int(rec.rebar_node))

        for k, rec in enumerate(bars):
            for i_node, j_node in rec.connectivity:
                ri, rj = ranks(i_node), ranks(j_node)
                common = ri & rj
                owner = min(common) if common else min(ri)
                bar_plan.setdefault(owner, {}).setdefault(k, []).append(
                    (int(i_node), int(j_node)))
                if owner not in rj:
                    ghosts.setdefault(owner, set()).add(int(j_node))

        plan: "dict[int, tuple[list[Any], list[Any], tuple[int, ...]]]" = {}
        for rank in set(tie_plan) | set(bar_plan):
            rank_bars = [
                replace(bars[k], connectivity=tuple(cells))
                for k, cells in sorted(bar_plan.get(rank, {}).items())
            ]
            plan[rank] = (
                tie_plan.get(rank, []),
                rank_bars,
                tuple(sorted(ghosts.get(rank, ()))),
            )
        return plan

    def _emit_reinforcement_partitioned(
        self,
        emitter: Emitter,
        tag_plan: TagPlan,
        entry: "tuple[list[Any], list[Any], tuple[int, ...]] | None",
        *,
        declared_ghosts: "set[int]",
        ghost_sp_ops: "dict[int, list[Any]]",
        inferred_ndf: "dict[int, int]",
        node_idx_lookup: "SortedIntToInt",
    ) -> None:
        """Emit this rank's embedded reinforcement: ghost bar nodes (with
        the owner's replayed SPs), then its ``LadrunoEmbeddedRebar`` ties
        and bar ``CorotTruss`` cells, planned by
        :meth:`_plan_partitioned_reinforcement`."""
        if not entry:
            return
        rank_ties, rank_bars, ghost_ids = entry
        for nid in ghost_ids:
            if nid in declared_ghosts:
                continue
            node_idx = node_idx_lookup.get(nid)
            if node_idx is None:
                raise BridgeError(
                    f"apeSees: rebar node {nid} is not in the FEM snapshot; "
                    "cannot ghost-declare it."
                )
            xyz = self.fem.nodes.coords[node_idx]
            _emit_node_with_inferred_ndf(
                emitter, inferred_ndf, nid,
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                self.ndf,
            )
            emit_ghost_sp_ops(emitter, nid, ghost_sp_ops.get(nid, ()))
            declared_ghosts.add(nid)
        emit_reinforce_ties(
            emitter, self.fem, tag_plan, name_to_tag=self.name_to_tag,
            records=rank_ties,
        )
        emit_rebar_elements(
            emitter, self.fem, tag_plan, name_to_tag=self.name_to_tag,
            records=rank_bars,
        )

    def _plan_partitioned_interfaces(
        self,
        records: "list[Any]",
        partitions: "list[Any]",
        *,
        inferred_ndf: "dict[int, int]",
        post_element: "list[Primitive]",
    ) -> "dict[int, list[tuple[Any, tuple[int, ...]]]]":
        """Resolve owner rank + ghost set for every interface record
        (ADR 0093 S8, INV-5), or refuse with a NAMED error.

        Returns ``{owner_rank: [(record, ghost_node_ids), ...]}`` —
        what step 7c-bis of the per-rank loop emits inside the owner's
        block. Empty when the model carries no interface records.

        Validation-before-emission, same discipline as the flat pass:
        the whole pool runs :func:`_validate_interface_records` (the
        ndf gate, the orient refusal, the phantom-coords guard), then
        :func:`_plan_rank_interfaces` (the INV-5 single-owner and
        master-native assertions), then the pattern-borne-sp sweep
        (:meth:`_refuse_pattern_sp_on_interface_ghosts` — a prescribed
        displacement on a node this plan will ghost cannot be honoured
        yet) — all before a single line is emitted, so a refused model
        emits nothing.

        Stage-claimed records are planned separately
        (:meth:`_plan_stage_interfaces_partitioned`, ADR 0093 S9) —
        the caller passes only the UNCLAIMED subset here, so every
        record in ``records`` is base-pass.
        """
        if not records:
            return {}
        _validate_interface_records(
            records, effective_ndf=inferred_ndf,
            envelope_ndf=self.ndf, ndm=self.ndm,
        )
        plan = _plan_rank_interfaces(records, partitions)
        ghost_ids = {
            int(nid)
            for entries in plan.values()
            for _rec, ghosts in entries
            for nid in ghosts
        }
        if ghost_ids:
            self._refuse_pattern_sp_on_interface_ghosts(
                ghost_ids, post_element, inferred_ndf,
            )
        return plan

    def _plan_stage_interfaces_partitioned(
        self,
        partitions: "list[Any]",
        *,
        inferred_ndf: "dict[int, int]",
        post_element: "list[Primitive]",
        node_owner_stage: "dict[int, int]",
    ) -> "list[dict[int, list[tuple[Any, tuple[int, ...]]]]]":
        """Owner/ghost plans for every stage's CLAIMED interface records
        (ADR 0093 S9 — INV-5 ownership applied inside INV-6's stage
        claim), or refuse with a NAMED error.

        Returns one plan per stage, indexed like ``self.stage_records``
        — each the ``{owner_rank: [(record, ghost_node_ids), ...]}``
        shape :meth:`_emit_stages_partitioned` consumes inside the
        stage's per-rank loop. Empty dicts for stages that claimed
        nothing; an empty list only when the model has no stages.

        Validation-before-emission, same discipline as the base-pass
        plan: per stage, the claimed subset runs
        :func:`_validate_interface_records`, then
        :func:`_plan_rank_interfaces` — the INV-5 single-owner and
        master-native assertions apply unchanged, because a claimed
        pair's owner is still the rank holding its stamped backing
        continuum element. The pattern-sp-on-ghost refusal sweeps the
        UNION of every stage's ghost set (the sweep itself already
        covers both the global pattern pool and each stage's
        ``pattern_specs``), so a stage pattern prescribing displacement
        on a slave that another stage's plan will ghost is refused too.

        One stage-specific refusal lives here: a claimed record whose
        master or slave node is owned by a LATER stage than the
        claiming one. On the flat path that deck fails loud at OpenSees
        parse time (the stage block references a node not yet
        declared); under MP the owner rank would ghost-declare the
        node stages before its home rank brings it online, the deck
        RUNS, and the interface pushes on a floating ghost — the
        ADR 0092 plausible-wrong-answer class, so it refuses at plan
        time instead.
        """
        plans: "list[dict[int, list[tuple[Any, tuple[int, ...]]]]]" = []
        all_ghosts: "set[int]" = set()
        for stage_idx, stage in enumerate(self.stage_records):
            recs = list(stage.stage_interface_records)
            if not recs:
                plans.append({})
                continue
            _validate_interface_records(
                recs, effective_ndf=inferred_ndf,
                envelope_ndf=self.ndf, ndm=self.ndm,
            )
            for pos, rec in enumerate(recs, start=1):
                name = getattr(rec, "name", None)
                label = f"#{pos}" + (f" ({name!r})" if name else "")
                for role, nid in (
                    ("master", int(rec.master_node)),
                    ("slave", int(rec.slave_node)),
                ):
                    sidx = node_owner_stage.get(nid)
                    if sidx is not None and sidx > stage_idx:
                        raise BridgeError(
                            f"apeSees: stage {stage.name!r} claims "
                            f"interface record {label}, but its {role} "
                            f"node {nid} is owned by the LATER stage "
                            f"{self.stage_records[sidx].name!r} — the "
                            "claiming stage's block would reference a "
                            "node whose topology only comes online "
                            "stages later (under MP the owner rank "
                            "would ghost-declare it early and the "
                            "interface would push on a floating ghost "
                            "— a plausible-wrong answer, ADR 0092 / "
                            "ADR 0093 S9). Claim the interface into "
                            "the stage that activates its topology, "
                            "or a later one."
                        )
            plan = _plan_rank_interfaces(recs, partitions)
            plans.append(plan)
            all_ghosts.update(
                int(nid)
                for entries in plan.values()
                for _rec, ghosts in entries
                for nid in ghosts
            )
        if all_ghosts:
            self._refuse_pattern_sp_on_interface_ghosts(
                all_ghosts, post_element, inferred_ndf,
            )
        return plans

    def _refuse_pattern_sp_on_interface_ghosts(
        self,
        ghost_ids: "set[int]",
        post_element: "list[Primitive]",
        inferred_ndf: "dict[int, int]",
        *,
        lane: str = "interface",
    ) -> None:
        """Refuse a pattern-borne ``sp`` targeting a node an interface
        plan will ghost-declare (ADR 0027 INV-2 / ADR 0093 INV-5).

        ``lane`` selects the wording only — ``"interface"`` (the ADR
        0093 pairs this sweep was written for) or ``"contact"`` (the
        ADR 0092 contact/contact_plane ghost sets, 2026-08-13 review
        F1). The machinery is record-shape-agnostic: it needs nothing
        but the ghost id set, so both planners run the identical sweep
        and only the error's header/invariant naming differs.

        The ghost replay stream carries the ``fix`` tiers only — the
        model-level ``ops.fix`` pool and, under staging, each stage's
        ``s.fix`` / ``s.remove_sp`` history (``stage_ghost_sp_ops``).
        Pattern ``sp`` lines fan out on the node's NATIVE ranks
        (:func:`_emit_pattern_sp_partitioned` filters on
        ``owned_nodes``) and are never mirrored onto a ghost. A
        prescribed displacement on a ghosted interface slave would
        therefore constrain the DOF on the home rank while the owner
        rank's ghost copy stays free — the exact constrained-DOF
        disagreement ADR 0027 INV-2 documents as a measured Mumps
        "Matrix is Singular Numerically" under ``numberer
        ParallelPlain``. Folding pattern sp into the ghost stream is a
        named follow-up; until it lands, refusing loudly at plan time
        is the honest behaviour.

        ADR 0052 HOLD supports (``s.support``) are pattern-borne sp
        too — a per-stage dedicated ``Plain`` pattern of ``sp <node>
        <dof> [nodeDisp …] -const`` lines, fanned out on native ranks
        only (``rank_support`` filters on ``rank_owned``). Worse than
        the singular-matrix case: a HOLD on a ghosted slave leaves the
        owner rank's copy free and the 2-rank deck runs CLEAN to a
        plausible wrong answer — measured (S9 probe, ``mpiexec -n 2``)
        26.4% error on the base reaction and 12.5% on the master
        displacement, while ``s.fix`` in the same slot agrees with
        serial to 1e-16. Mirroring the HOLD sp onto a ghost would need
        the runtime ``[nodeDisp …]`` capture replayed on the owner
        rank — a named follow-up; refused here.

        Sweeps every :class:`Plain` pattern — the global pool
        (``post_element``) and each stage's ``pattern_specs`` — over
        its explicit ``sp`` records (node and pg targets) and its
        ``from_model(case)`` sp imports, then every stage's
        ``support_records``. Called with the UNION-of-ghosts of the
        base (unclaimed, S8) plan and of every stage's claimed (S9)
        plan, so both paths are covered.
        """
        from .pattern.pattern import Plain

        if lane == "contact":
            what = "g.constraints.contact / g.constraints.contact_plane"
            ghost_why = (
                "their interaction's owner rank (contact interface "
                "ghosts, ADR 0092 INV-2)"
            )
            mirroring = (
                "Mirroring pattern sp into the contact ghost replay — "
                "correctly, including under staging — is its own "
                "project and is not wired yet."
            )
        else:
            what = "g.constraints.interface()"
            ghost_why = (
                "their pair's owner rank (foreign slave/beam nodes, "
                "ADR 0093 INV-5)"
            )
            mirroring = (
                "Mirroring pattern sp into the ghost replay is not "
                "wired yet."
            )

        def _refuse(label: str, hits: "set[int]", consequence: str) -> None:
            raise BridgeError(
                f"apeSees: {what} — node(s) "
                f"{sorted(hits)} are ghost-declared on "
                f"{ghost_why}, but {label} prescribes sp "
                "displacement(s) on "
                "them. Pattern-borne sp lines emit only on a node's "
                f"NATIVE ranks, so the ghost copy would stay "
                f"unconstrained on the owner rank — {consequence} "
                f"{mirroring} Re-partition so the slave "
                "side is native to the owner rank "
                "(g.mesh.partitioning.partition(n, "
                "uncuttable_elements=...)), use ops.fix / s.fix for a "
                "homogeneous constraint (the fix tier IS replayed "
                "onto ghosts), or emit serial (non-partitioned / "
                "flat=True)."
            )

        pools: "list[tuple[str, Any]]" = [
            ("pattern", p) for p in post_element if isinstance(p, Plain)
        ]
        pools += [
            (f"stage {st.name!r} pattern", p)
            for st in self.stage_records
            for p in st.pattern_specs
            if isinstance(p, Plain)
        ]
        for kind, p in pools:
            hits: "set[int]" = set()
            for sp_rec in p.sps:
                if sp_rec.target_kind == "node":
                    try:
                        nid = int(sp_rec.target)
                    except (TypeError, ValueError):
                        continue
                    if nid in ghost_ids:
                        hits.add(nid)
                    continue
                try:
                    ids = self.fem.nodes.select(pg=sp_rec.target).ids
                except (KeyError, ValueError, AttributeError):
                    continue
                hits.update(
                    int(n) for n in ids if int(n) in ghost_ids
                )
            if not hits and p.from_model_cases:
                _fm_loads, fm_sps = self._owned_from_model_lines(
                    p.from_model_cases,
                    load_nodes=set(),
                    sp_nodes=set(ghost_ids),
                    inferred_ndf=inferred_ndf,
                )
                hits.update(int(nid) for nid, _dof, _val in fm_sps)
            if not hits:
                continue
            tag = self.tag_for.get(id(p))
            label = f"{kind} (tag {tag})" if tag is not None else kind
            _refuse(
                label, hits,
                "the constrained-DOF disagreement ADR 0027 INV-2 "
                "documents as a measured 'Matrix is Singular "
                "Numerically' under numberer ParallelPlain.",
            )

        # ADR 0052 HOLD supports — see the docstring: on a ghosted
        # slave the deck runs CLEAN to the wrong answer (measured
        # 26.4% / 12.5%), the ADR 0092 silent-error class.
        for st in self.stage_records:
            hold_hits: "set[int]" = set()
            for srec in st.support_records:
                if not any(srec.dofs):
                    continue
                for nid in self._resolve_node_target(srec.pg, srec.nodes):
                    if int(nid) in ghost_ids:
                        hold_hits.add(int(nid))
            if hold_hits:
                _refuse(
                    f"stage {st.name!r} s.support (an ADR 0052 HOLD "
                    "is a pattern-borne sp)",
                    hold_hits,
                    "the 2-rank deck runs CLEAN and converges to a "
                    "plausible wrong answer (measured: 26.4% error on "
                    "the base reaction, 12.5% on the master "
                    "displacement — the ADR 0092 silent-error class).",
                )

    def _validate_interfaces_not_stage_bound(
        self, node_owner_stage: "dict[int, int]",
    ) -> None:
        """Refuse UNCLAIMED interface records whose endpoints are
        stage-bound nodes (ADR 0093 S8 hardening; shared by the flat
        and partitioned paths).

        The base interface pass emits before any ``stage_open``, so an
        unclaimed record whose master/slave node only comes online
        inside a stage block (``s.activate(pgs=[...])``) would emit a
        ``zeroLength`` (and, mixed-ndf, an ``equalDOF``) referencing a
        non-existent OpenSees node — a parse-time crash at best. Same
        failure mode as the H1 global-BC case
        (:meth:`_validate_no_stage_bound_node_targets`). Claimed
        records are exempt: they emit inside their owning stage's
        block, after the stage's activated topology (ADR 0093 INV-6).
        """
        if not node_owner_stage:
            return
        elements_comp = getattr(self.fem, "elements", None)
        records = list(getattr(elements_comp, "interfaces", None) or ())
        if not records:
            return
        claimed = self._claimed_interface_ids()
        offenders: "list[str]" = []
        for pos, rec in enumerate(records, start=1):
            if id(rec) in claimed:
                continue
            name = getattr(rec, "name", None)
            label = f"#{pos}" + (f" ({name!r})" if name else "")
            for role, nid in (
                ("master", int(rec.master_node)),
                ("slave", int(rec.slave_node)),
            ):
                sidx = node_owner_stage.get(int(nid))
                if sidx is None:
                    continue
                stage_name = self.stage_records[sidx].name
                offenders.append(
                    f"  • interface {label}: {role} node {nid} is "
                    f"owned by stage {stage_name!r}"
                )
        if not offenders:
            return
        raise BridgeError(
            "Stage-bound nodes referenced by UNCLAIMED "
            "g.constraints.interface() records — the base interface "
            "pass emits before any stage_open, so the zeroLength (and "
            "a mixed-ndf pair's equalDOF) would reference a "
            "non-existent OpenSees node, crashing at parse time. "
            "Claim the interface into the stage that activates its "
            "topology: s.interface(name=...) inside the owning "
            "``with ops.stage(...) as s`` block (ADR 0093 INV-6). "
            "Offenders:\n" + "\n".join(offenders)
        )

    def _emit_interfaces_partitioned(
        self,
        emitter: Emitter,
        entries: "list[tuple[Any, tuple[int, ...]]]",
        *,
        interface_tag_plan: "dict[int, tuple[int, int, int]]",
        declared_ghosts: "set[int]",
        ghost_sp_ops: "dict[int, list[Any]]",
        inferred_ndf: "dict[int, int]",
        node_idx_lookup: "SortedIntToInt",
    ) -> None:
        """Emit this rank's owned interface units (ADR 0093 S8/S9).

        Two call sites, one shape: the base per-rank pass (step 7c-bis
        of :meth:`_emit_partitioned`, unclaimed records) and the
        stage-claimed pass inside a stage's per-rank bracket
        (:meth:`_emit_stages_partitioned`, ADR 0093 S9 — where
        ``ghost_sp_ops`` is the stage-inclusive
        ``stage_ghost_sp_ops`` stream and ``declared_ghosts`` is the
        rank's live ``ghosts_held`` registry, so ghost declarations
        replay the owner's SP history up to and including the claiming
        stage and later stages mirror their deltas onto them).

        For each planned ``(record, ghost_node_ids)`` entry: first
        declare the foreign slave/beam node — ``node tag x y z`` with
        the SAME inferred/envelope ndf its home rank emits (INV-5 ghost
        ndf parity: mixed-ndf models are exactly where per-node ndf is
        live, so the declaration goes through the same
        ``_emit_node_with_inferred_ndf`` the home rank's native pass
        uses), immediately followed by the owner's replayed SP stream
        (ADR 0027 INV-2 machinery) — then the atomic unit via the
        shared :func:`_emit_interface_record` core, consuming this
        record's pre-allocated tag triple.

        ADR 0092 INV-7 holds structurally: a ghost gets a ``node`` line
        + SP replay and NOTHING else — mass buckets by primary owner,
        loads by primary owner, elements by element owner, all of which
        point at the ghost's native rank, never here. The phantom is
        minted here (owner-only) and registered with the phantom-tag
        predicate before its ``node()`` call, same as the S5/S7 passes.

        The nested equalDOF emits on the owner rank ONLY — deviating
        from the ADR 0027 MP replication rule for the reason INV-5
        states: the constrained node is a phantom that exists on
        exactly one rank (the owner mints it), so there is nothing to
        replicate; the retained beam node's DOFs are shared by tag
        through the ghost mechanism.

        ``declared_ghosts`` is this rank's live ghost registry
        (``ghost_tags_by_rank[rank]``, already holding 7b's
        MP-constraint ghosts and 7c's contact ghosts): a node already
        declared is not re-declared (a duplicate ``node`` line is an
        OpenSees parse error), and every ghost declared here is
        registered back into it so a staged model keeps its SP state
        in sync across later stage blocks.
        """
        if not entries:
            return
        _register_interface_phantoms(
            emitter, [rec for rec, _ghosts in entries],
        )
        for rec, ghost_ids in entries:
            for nid_raw in ghost_ids:
                nid = int(nid_raw)
                if nid in declared_ghosts:
                    continue
                node_idx = node_idx_lookup.get(nid)
                if node_idx is None:
                    raise BridgeError(
                        f"apeSees: interface slave node {nid} is not in "
                        "the FEM snapshot — cannot ghost-declare it on "
                        "the pair's owner rank (ADR 0093 INV-5). The "
                        "interface record references a node the mesh "
                        "does not carry."
                    )
                xyz = self.fem.nodes.coords[node_idx]
                _emit_node_with_inferred_ndf(
                    emitter, inferred_ndf, nid,
                    (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    self.ndf,
                )
                emit_ghost_sp_ops(
                    emitter, nid, ghost_sp_ops.get(nid, ()),
                )
                declared_ghosts.add(nid)
            _emit_interface_record(
                emitter, rec, interface_tag_plan[id(rec)],
            )

    def _emit_rayleigh(
        self,
        emitter: Emitter,
        tag_plan: "TagPlan | None" = None,
        fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
        *,
        stage: "StageRecord | None" = None,
    ) -> None:
        """Emit Rayleigh damping declarations (ADR 0053, D1 + D2 + D5).

        Domain-level commands, emitted driver-post (after the model is built)
        alongside fixes / masses / regions. **Globals emit first, then
        region-scoped forms** so a region refines the global ("region wins"),
        matching OpenSees' OVERWRITE-per-element semantics. When a global and
        any region-scoped record coexist a :class:`RayleighOverwriteWarning`
        fires — the region replaces (does not add to) the global damping for
        its elements.

        Global records (``on == ()``) render a bare ``rayleigh αM βK βK0
        βKc``. Region records render one ``region $tag -ele … -rayleigh …``
        per ``on`` physical-group name, with ``-ele`` membership because βK
        is stiffness-proportional. ``tag_plan`` / ``fem_eid_to_ops_tag`` are
        required only when region records are present (the emit driver always
        supplies them); each region's tag is the one the build's tag plan
        gave it (``("rayleigh", scope)`` sites of
        :meth:`_region_sites`).

        ``stage`` selects the source pool — D5 passes the stage whose
        ``rayleigh_records`` emit inside its block; ``None`` uses the global
        (non-staged) pool.
        """
        recs = self.rayleigh_records if stage is None else stage.rayleigh_records
        if not recs:
            return
        import warnings as _warnings

        globals_ = [r for r in recs if not r.on]
        scoped = [r for r in recs if r.on]
        if globals_ and scoped:
            _warnings.warn(
                RayleighOverwriteWarning(
                    "A global ops.damping.rayleigh and a region-scoped one "
                    "(on=...) coexist; OpenSees overwrites element Rayleigh "
                    "per element, so elements in the region take the region's "
                    "factors (NOT the sum of global + region).",
                ),
                stacklevel=2,
            )
        for rec in globals_:
            emitter.rayleigh(
                rec.alpha_m, rec.beta_k, rec.beta_k_init, rec.beta_k_comm,
            )
        if not scoped:
            return
        region_tags = self._planned_damping_region_tags(
            "rayleigh", stage, recs, tag_plan)
        for i, rec in enumerate(recs):
            for j, name in enumerate(rec.on):
                ele_tags = self._resolve_damping_on_elements(
                    name, fem_eid_to_ops_tag,
                )
                tag = region_tags[(i, j)]
                emitter.region(
                    tag, "-ele", *ele_tags, "-rayleigh",
                    rec.alpha_m, rec.beta_k, rec.beta_k_init, rec.beta_k_comm,
                )

    def _emit_global_damping_partitioned(
        self,
        emitter: Emitter,
        tag_plan: TagPlan,
        fem_eid_to_ops_tag: "FemToOpsTagMap",
    ) -> None:
        """Emit global (non-stage) damping under partitioned (MPI) emit.

        Mirrors the stage-bound partitioned pass
        (:meth:`_emit_stages_partitioned` step 3b): ``rayleigh`` and the
        Damping-object ``region -ele … -damp`` attaches are emitted ONCE
        outside any ``partition_open`` block, and OpenSeesMP binds only the
        elements each rank owns —
        ``MeshRegion::setElements`` keeps "only those elements in the
        domain" (foreign element tags from other ranks are silently
        skipped), so a global ``region -ele <all tags>`` line is correct on
        every rank.  A bare global ``rayleigh αM βK βK0 βKc`` likewise
        applies to each rank's local domain.  This is the same routing the
        flat path uses driver-post — :meth:`_emit_rayleigh` /
        :meth:`_emit_damping_attach` handle both the bare and the
        region-scoped (``on=``) forms.

        This drains only the bridge's GLOBAL pools
        (``self.rayleigh_records`` / ``self.damping_attach_records``);
        stage-bound damping routes through the owning stage's pool in
        :meth:`_emit_stages_partitioned` and is untouched here.

        Before this, the partitioned path emitted **no** global damping at
        all — a global ``ops.damping.rayleigh(...)`` declared outside a
        stage was silently dropped, so an np>1 run came out undamped (the
        plane-wave handoff's load-bearing finding #1).

        Modal damping (``ops.damping.modal`` → ``eigen`` + ``modalDamping``)
        is the one form that does **not** carry over, and it fails loud
        rather than emit a silently-incorrect deck — exactly as the stage
        path refuses per-stage modal.  Use Rayleigh damping under MPI, or
        emit single-process.

        **This guard got MORE load-bearing, not less, on 2026-07-27.**
        Its original reason was that a bare ``eigen`` under OpenSeesMP
        solved each rank's LOCAL subdomain — so the eigensolve itself
        failed the user, and refusing was belt-and-braces.  Fork PR #668
        fixed the eigensolve (ADR 0077 Tier 1B: with ``system Mumps``
        the ARPACK route is now a correct distributed eigensolve), which
        removes that accidental protection.  What is still broken is
        ``modalDamping`` itself: its modal projection of velocity is a
        rank-local partial sum, so the assembled damping force is
        ``Σ_r β_r M_r φ`` rather than ``β M φ`` and the damped response
        changes with rank count — with no warning and no error
        (fork ADR-1000 §34.4).  Do not relax this guard on the grounds
        that "the eigen works now"; the eigen working is precisely what
        makes it the last line of defence.
        """
        if self.modal_damping_records:
            raise BridgeError(
                "apeSees: modal damping (ops.damping.modal) is not "
                "supported under partitioned (MPI) emit — modalDamping's "
                "modal projection is a rank-local partial sum, so the "
                "assembled damping force is sum_r(beta_r M_r phi) instead "
                "of beta M phi and the damped response silently changes "
                "with rank count (fork ADR-1000 section 34.4). This is "
                "true even on a build where the preceding eigen is a "
                "correct distributed solve (fork PR #668 / ADR 0077 Tier "
                "1B). Use Rayleigh damping (ops.damping.rayleigh, any "
                "on=) under MPI, or emit single-process (non-partitioned)."
            )
        # Rayleigh (bare global + region-scoped) and Damping-object region
        # attaches — emitted once, outside any partition block, exactly as
        # the stage-bound partitioned pass does (every rank binds only its
        # locally-owned elements; foreign -ele tags are skipped).
        self._emit_rayleigh(emitter, tag_plan, fem_eid_to_ops_tag)
        self._emit_damping_attach(emitter, tag_plan, fem_eid_to_ops_tag)

    def _resolve_damping_on_elements(
        self,
        pg: str,
        fem_eid_to_ops_tag: "FemToOpsTagMap | None",
    ) -> tuple[int, ...]:
        """Resolve a damping ``on=`` physical-group name to OpenSees element
        tags (fail-loud on an empty / unmapped group). Shared by region
        Rayleigh (D2) and the Damping-object attach (D3)."""
        from ._internal.build import expand_pg_to_elements

        if fem_eid_to_ops_tag is None:
            raise BridgeError(
                "ops.damping(on=...) needs the element-tag map; this is an "
                "internal emit-wiring error.",
            )
        ops_tags: list[int] = []
        for eid, _conn in expand_pg_to_elements(self.fem, pg):
            ops_tag = fem_eid_to_ops_tag.get(int(eid))
            if ops_tag is None:
                raise BridgeError(
                    f"ops.damping(on={pg!r}): element {eid} has no emitted "
                    "OpenSees tag (is the group meshed / emitted?).",
                )
            ops_tags.append(int(ops_tag))
        if not ops_tags:
            raise ValueError(
                f"ops.damping(on={pg!r}): the group resolved to zero elements "
                "— region-scoped damping needs elements (βK / -damp act on "
                "elements).",
            )
        return tuple(ops_tags)

    def _emit_damping_attach(
        self,
        emitter: Emitter,
        tag_plan: "TagPlan | None" = None,
        fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
        *,
        stage: "StageRecord | None" = None,
    ) -> None:
        """Attach each ``damping`` object to its ``on`` groups (ADR 0053 D3).

        The object itself already emitted its ``damping <Type> $tag`` line in
        the pre-element definition group; here, driver-post, we emit one
        ``region $tag -ele … -damp $dampTag`` per ``on`` physical group, with
        ``-ele`` membership. The object's tag is read back from ``tag_for``;
        each region's tag is the one the build's tag plan gave it
        (``("damping", scope)`` sites of :meth:`_region_sites`).

        ``stage`` selects the source pool — D5 passes the stage whose
        ``damping_attach_records`` attach inside its block (the object
        definition still emits once, pre-element); ``None`` uses the global
        (non-staged) pool.
        """
        recs = (self.damping_attach_records if stage is None
                else stage.damping_attach_records)
        if not recs:
            return
        region_tags = self._planned_damping_region_tags(
            "damping", stage, recs, tag_plan)
        for i, rec in enumerate(recs):
            damp_tag = self.tag_for[id(rec.prim)]
            for j, name in enumerate(rec.on):
                ele_tags = self._resolve_damping_on_elements(
                    name, fem_eid_to_ops_tag,
                )
                emitter.region(
                    region_tags[(i, j)], "-ele", *ele_tags, "-damp", damp_tag,
                )

    @staticmethod
    def _damping_region_keys(
        recs: "Sequence[RayleighRecord | DampingAttachRecord]",
    ) -> tuple[tuple[int, int], ...]:
        """The region keys of a damping pool: ``(record index, on index)``
        of every ``on`` name, in the order the pool writes them."""
        return tuple(
            (i, j) for i, rec in enumerate(recs) for j in range(len(rec.on)))

    def _planned_damping_region_tags(
        self,
        kind: str,
        stage: "StageRecord | None",
        recs: "Sequence[RayleighRecord | DampingAttachRecord]",
        tag_plan: "TagPlan | None",
    ) -> dict[object, int]:
        """The planned region tag of each ``on`` name of a damping pool."""
        if tag_plan is None:
            raise BridgeError(
                "apeSees: region-scoped damping (on=...) needs the emit's "
                "tag plan; this is an internal emit-wiring error."
            )
        return tag_plan.regions.tags_for(
            self.fem, (kind, None if stage is None else id(stage)),
            self._damping_region_keys(recs),
        )

    def _emit_modal_damping(self, emitter: Emitter) -> None:
        """Emit bundled ``eigen`` + ``modalDamping`` (ADR 0053 D4).

        Domain-level directive, emitted driver-post (after the model is built,
        so the mass matrix exists). For each record: ``eigen <solver> <modes>``
        (reusing :meth:`Emitter.eigen` — the live emitter runs the solve here,
        exactly when ``modalDamping`` needs the computed modes) followed by
        ``modalDamping <f1> [..]``. A scalar factor applies uniformly to all
        modes; ``modes`` factors apply per-mode.
        """
        for rec in self.modal_damping_records:
            emitter.eigen(rec.modes, solver=rec.solver)
            emitter.modal_damping(*rec.factors)

    def _emit_regions(self, emitter: Emitter, tag_plan: TagPlan) -> None:
        """Fan named-region assignments out into ``emitter.region`` calls.

        One ``region $tag -node ...`` line per name, in first-seen order.
        The build's tag plan merged each name's members (every record of
        the name, PG resolution and explicit tuples alike, deduped by node
        tag in first-seen order) and gave each name with members one
        ``"region"`` tag (:meth:`_region_sites`); this writes them.
        """
        if not self.region_records:
            return
        for region in self._planned_named_regions(tag_plan, None):
            if region.nodes:
                emitter.region(region.planned_tag(), "-node", *region.nodes)

    def _emit_stage_regions(
        self,
        stage: "StageRecord",
        emitter: Emitter,
        tag_plan: TagPlan,
    ) -> None:
        """Single-partition fan-out for one stage's region pool.

        :meth:`_emit_regions` over ``stage.region_records``: one
        ``region $tag -node n1 n2 ...`` per name, under the tag the plan
        gave the name at this stage's site.

        V3 (Phase SSI-2.D PR-A) guarantees no name collision across
        scopes, so each stage's name set is disjoint from every other
        stage's and from the global pool; the plan keys each stage's
        names by the stage, so their tags are disjoint by construction.
        """
        if not stage.region_records:
            return
        for region in self._planned_named_regions(tag_plan, stage):
            if region.nodes:
                emitter.region(region.planned_tag(), "-node", *region.nodes)

    def _emit_stage_regions_partitioned(
        self,
        stage: "StageRecord",
        emitter: Emitter,
        tag_plan: TagPlan,
        owned_nodes: "set[int] | SortedIntSet",
        rank: int,
    ) -> None:
        """Per-rank fan-out for one stage's region pool (MP path).

        Mirrors :meth:`_emit_regions_partitioned` over
        ``stage.region_records``: same merged members, per-rank
        ``owned_nodes`` intersection, INV-4 empty-intersection skip.
        Every rank that emits a region writes the one tag the plan gave
        it, which the plan numbered on the first rank that emits it.
        """
        if not stage.region_records:
            return
        for region in self._planned_named_regions(tag_plan, stage):
            owned_list = [n for n in region.nodes if int(n) in owned_nodes]
            if not owned_list:
                region.check_unheld_on(rank)
                continue
            emitter.region(region.tag_on_rank(rank), "-node", *owned_list)

    def _emit_regions_partitioned(
        self,
        emitter: Emitter,
        tag_plan: TagPlan,
        owned_nodes: "set[int] | SortedIntSet",
        rank: int,
    ) -> None:
        """Per-rank named-region fan-out (ADR 0027 §"Regions interaction" /
        INV-4).

        Same merged members as :meth:`_emit_regions`, but each is
        intersected with ``owned_nodes`` before emission.  Empty
        intersection ⇒ no ``region`` line emitted on this rank (INV-4).
        The tag is the same scalar on every rank that emits the region
        (INV-4 §"region tag is the same scalar across every rank that
        does emit"): the plan gave it once, numbered on the FIRST rank
        that emits the region.
        """
        if not self.region_records:
            return
        for region in self._planned_named_regions(tag_plan, None):
            owned_list = [n for n in region.nodes if int(n) in owned_nodes]
            if not owned_list:
                # INV-4: empty intersection → no region line on this rank.
                region.check_unheld_on(rank)
                continue
            emitter.region(region.tag_on_rank(rank), "-node", *owned_list)

    def _planned_named_regions(
        self, tag_plan: TagPlan, stage: "StageRecord | None",
    ) -> "tuple[NamedRegion, ...]":
        """The named regions the plan holds for ``stage``'s pool (``None``:
        the global pool), with their merged members and planned tags.

        The plan must hold exactly the names this pool's records declare,
        in first-seen order; a plan made for another model raises.
        """
        records = self.region_records if stage is None else stage.region_records
        names = tuple(dict.fromkeys(rec.name for rec in records))
        return tag_plan.regions.named_for(
            self.fem, ("named", None if stage is None else id(stage)), names)

    def _merged_region_members(
        self, records: "Sequence[RegionAssignmentRecord]",
    ) -> dict[str, tuple[int, ...]]:
        """Each region name's members, in first-seen name order.

        Every record of a name joins it (PG resolution and explicit
        tuples alike); members dedupe by node tag, in first-seen order.
        """
        by_name: dict[str, list[int]] = {}
        seen_per_name: dict[str, set[int]] = {}
        for rec in records:
            bucket = by_name.setdefault(rec.name, [])
            seen = seen_per_name.setdefault(rec.name, set())
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if node_tag not in seen:
                    seen.add(node_tag)
                    bucket.append(node_tag)
        return {name: tuple(nodes) for name, nodes in by_name.items()}

    def _region_sites(
        self, mode: TagMode, ordered: "Sequence[Primitive]",
    ) -> "tuple[list[tuple[RegionSite, tuple[object, ...]]], list[tuple[RegionSite, tuple[object, ...]]], dict[RegionSite, NamedMembers]]":
        """Every region site, in the order the flat walk mints their tags
        and in the order ``mode``'s emit writes them, and the merged
        members of every named-region site.

        Returns ``(sites, written, named)``. ``sites`` is the canonical
        mint order, the flat emit's in every mode, so a region has one tag
        whatever the mode (ADR 0114 D4, amended): the global named regions
        (each name that has members), the global region-scoped Rayleigh,
        the global damping attaches, then the global pass's filtered
        recorders (``ordered``'s order, the stage-claimed ones skipped);
        then, staged, each stage in turn: its named regions, its Rayleigh,
        its damping attaches, its claimed filtered recorders.

        ``written`` is ``sites`` on a flat emit. A partitioned emit writes
        the global pass's filtered recorders first (their regions are
        written inside every rank block), then the global named regions,
        each first on the first rank (in partition order) that holds one
        of its members, then the global damping; then each stage as above,
        its named regions in first-holder order. A named region no rank
        holds is not written there.

        A stage-claimed recorder's regions belong to its stage alone. The
        partitioned global pass once planned them too, so each was written
        twice under two tags (#1446).
        """
        partitioned = mode.partitioned
        sites: list[tuple[RegionSite, tuple[object, ...]]] = []
        written: list[tuple[RegionSite, tuple[object, ...]]] = []
        named: dict[RegionSite, NamedMembers] = {}
        rank_nodes: "list[tuple[int, SortedIntSet]] | None" = None
        if partitioned and (self.region_records or any(
                st.region_records for st in self.stage_records)):
            rank_nodes = [
                (runtime_rank_from_partition_record(part, idx),
                 SortedIntSet.from_ids(part.node_ids))
                for idx, part in enumerate(self.fem.partitions)
            ]

        def add(site: RegionSite, keys: "tuple[object, ...]",
                written_keys: "tuple[object, ...] | None" = None) -> None:
            sites.append((site, keys))
            written.append(
                (site, keys if written_keys is None else written_keys))

        def add_named(
            scope: "int | None", records: "Sequence[RegionAssignmentRecord]",
        ) -> None:
            if not records:
                return
            site: RegionSite = ("named", scope)
            members = self._merged_region_members(records)
            first: dict[str, int] = {}
            keys = tuple(name for name, nodes in members.items() if nodes)
            written_keys = keys
            if rank_nodes is not None:
                # Written first on the first rank, in partition order,
                # that holds a member; first-seen order within a rank.
                for rank, owned in rank_nodes:
                    for name, nodes in members.items():
                        if name not in first and any(
                                int(n) in owned for n in nodes):
                            first[name] = rank
                written_keys = tuple(first)
            named[site] = tuple(
                (name, nodes, first.get(name))
                for name, nodes in members.items()
            )
            add(site, keys, written_keys)

        def add_damping(
            kind: str, scope: "int | None",
            recs: "Sequence[RayleighRecord | DampingAttachRecord]",
        ) -> None:
            keys = self._damping_region_keys(recs)
            if keys:
                add((kind, scope), keys)

        def recorder_sites(
            specs: "Iterable[object]",
        ) -> "list[tuple[RegionSite, tuple[object, ...]]]":
            return [
                (("recorder", id(spec)), spec.region_keys())
                for spec in specs
                if isinstance(spec, FilterableRecorder) and spec.region_keys()
            ]

        claimed = self._claimed_recorder_ids()
        global_recorders = recorder_sites(
            p for p in ordered
            if isinstance(p, Recorder) and id(p) not in claimed
        )
        if partitioned:
            written += global_recorders
        add_named(None, self.region_records)
        add_damping("rayleigh", None, self.rayleigh_records)
        add_damping("damping", None, self.damping_attach_records)
        sites += global_recorders
        if not partitioned:
            written += global_recorders
        for stage in self.stage_records:
            add_named(id(stage), stage.region_records)
            add_damping("rayleigh", id(stage), stage.rayleigh_records)
            add_damping("damping", id(stage), stage.damping_attach_records)
            for site_keys in recorder_sites(stage.recorder_specs):
                add(*site_keys)
        return sites, written, named

    # -- MPCO recorder filter regions (ADR 0027 INV-4 — internal regions) --

    def _plan_partitioned_mpco_recorders(
        self,
        post_element: "list[Primitive]",
        tag_plan: TagPlan,
    ) -> "dict[int, _MPCOFilterPlan]":
        """Pre-resolve the region(s) of every region-bearing recorder of
        the global pass, under the tags the build's tag plan gave them.

        Returns a dict keyed by ``id(spec)`` carrying the recorder's
        regions (value-channel filter region + any Ladruno ``energy_pg``
        energy region, each with resolved FEM-eid members) plus the
        materialised spec (``_region_tag`` / ``_energy_region_tags``
        populated, ``*_pg`` selectors cleared, ready for ``_emit``).
        Recorders with neither a filter nor an energy region are absent
        from the dict; the caller routes them through the unchanged
        :func:`emit_recorder_spec` global pass.

        Stage-claimed recorders are absent too: their stage writes their
        regions, inside the stage block (after the stage's elements are in
        the domain), under the tags the plan gave them there. Planning
        them here as well wrote each of their regions twice, under two
        tags, the first never referenced (#1446).

        The plan is built ONCE before the per-rank loop so:

        1. Each region tag is the SAME scalar across every rank that
           emits its rank-intersection of the region (INV-4: stitching
           by tag identity).
        2. ``_emit`` is bypass-safe — the materialised spec carries the
           shared tags directly, so the global recorder pass after the
           per-rank loop simply forwards ``-R <tag>`` / ``-G energy
           <tag>`` without re-reading tags or re-emitting regions.
        """
        claimed = self._claimed_recorder_ids()
        plan: dict[int, _MPCOFilterPlan] = {}
        for p in post_element:
            # FilterableRecorder = MPCO + Ladruno (ADR 0064): both share
            # the has_filter()/resolve_filter_ids()/-R region machinery,
            # so the per-rank region pass covers both recorder kinds.
            if not isinstance(p, FilterableRecorder) or id(p) in claimed:
                continue
            if not p.region_keys():
                continue
            region_tags = p.planned_region_tags(self.fem, tag_plan)
            regions: list[_RegionEmit] = []
            materialised: FilterableRecorder = p

            # Value-channel filter region (-R), shared by MPCO + Ladruno.
            if p.has_filter():
                node_ids, elem_ids = p.resolve_filter_ids(self.fem)
                region_tag = region_tags["filter"]
                regions.append(_RegionEmit(region_tag, node_ids, elem_ids))
                materialised = replace(
                    materialised,
                    nodes_pg=None,
                    elements_pg=None,
                    nodes=node_ids if node_ids else None,
                    elements=elem_ids if elem_ids else None,
                    _region_tag=region_tag,
                )

            # Decoupled energy region (-G energy $tag), Ladruno only
            # (ADR 0064 §4). Independent of the value filter: its own tag,
            # its own per-rank fan-out.
            if isinstance(p, Ladruno) and p.energy_pg is not None:
                e_eids = p.resolve_energy_ids(self.fem)
                energy_tag = region_tags["energy"]
                regions.append(_RegionEmit(energy_tag, (), e_eids))
                assert isinstance(materialised, Ladruno)
                materialised = replace(
                    materialised,
                    energy_pg=None,
                    _energy_region_tags=(energy_tag,),
                )

            plan[id(p)] = _MPCOFilterPlan(
                materialised_spec=materialised,
                regions=tuple(regions),
            )
        return plan

    def _emit_mpco_filter_regions_for_rank(
        self,
        emitter: Emitter,
        rank: int,
        plan: "dict[int, _MPCOFilterPlan]",
        owned_nodes: "set[int] | SortedIntSet",
        element_owner: "SortedIntToInt",
        fem_eid_to_ops_tag: "FemToOpsTagMap",
    ) -> None:
        """Per-rank emission of MPCO recorder filter regions (INV-4).

        For every filter-bearing MPCO recorder, intersects the resolved
        node ids with ``owned_nodes`` and the resolved element ids with
        the rank's owned elements (via ``element_owner``).  When BOTH
        intersections are empty the recorder's region is omitted on
        this rank (INV-4 §"empty intersection ⇒ no region emitted on
        that rank").

        ``entry.elem_ids`` carries **FEM eids** (the planner deliberately
        calls ``resolve_filter_ids(fem)`` without the
        ``fem_eid_to_ops_tag`` map so the per-rank ``element_owner``
        intersection — keyed by FEM eid — stays correct).  The
        per-rank FEM-eid subset is translated to OpenSees element tags
        via ``fem_eid_to_ops_tag`` just before emission so the
        ``region <tag> -ele ...`` line carries OpenSees tags (which is
        what the region command expects), not raw FEM eids.  Lookup
        miss → :class:`BridgeError`, mirroring the
        :meth:`Element.materialize` policy.

        The region tag is the SAME scalar across every emitting rank
        (carried on each plan entry); MPCO post-processing stitches the
        per-rank ``.mpco`` files by tag identity, so a rank with an
        empty intersection that simply omits its region line is fine —
        MPCO handles the missing per-rank contribution gracefully and
        the recorder declaration's ``-R <tag>`` still resolves on the
        ranks that did emit.
        """
        if not plan:
            return
        from ._internal.build import BridgeError
        for entry in plan.values():
            kind = type(entry.materialised_spec).__name__
            # A recorder may carry several regions (value-channel filter +
            # Ladruno energy region); emit each one's per-rank intersection.
            for region in entry.regions:
                # Per-rank node intersection — preserves declaration order.
                rank_node_ids = tuple(
                    n for n in region.node_ids if int(n) in owned_nodes
                )
                # Per-rank element intersection — keep elements whose owner
                # is this rank.  Element ownership is single-rank
                # (build_element_partition_owner), so a missing key means
                # the element isn't on any rank — skip silently.
                rank_fem_eids = tuple(
                    e for e in region.elem_ids
                    if element_owner.get(int(e)) == rank
                )
                if not rank_node_ids and not rank_fem_eids:
                    # INV-4: empty intersection on this rank → no region.
                    continue
                region_args: list[int | float | str] = []
                if rank_node_ids:
                    region_args += ["-node", *rank_node_ids]
                if rank_fem_eids:
                    rank_ops_tags: list[int] = []
                    for eid in rank_fem_eids:
                        ops_tag = fem_eid_to_ops_tag.get(int(eid))
                        if ops_tag is None:
                            raise BridgeError(
                                f"{kind} recorder region (rank {rank}): "
                                f"resolves to FEM eid {eid} but no element "
                                "was emitted at that eid — declare an "
                                "``ops.element.X(pg=...)`` primitive whose "
                                f"pg includes the {kind} recorder's "
                                "elements_pg / energy_pg."
                            )
                        rank_ops_tags.append(int(ops_tag))
                    region_args += ["-ele", *rank_ops_tags]
                emitter.region(region.tag, *region_args)

    def _resolve_node_target(
        self, pg: str | None, nodes: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if pg is not None:
            return expand_pg_to_nodes(self.fem, pg)
        assert nodes is not None  # exactly-one-of validated at apeSees.fix
        return nodes

    # ADR 0051: the broker nodal-load auto-emitters (_emit_broker_loads
    # / _emit_broker_loads_partitioned / _broker_load_components) were
    # removed. g.loads reach the deck only via an explicit
    # ops.pattern.Plain(...).from_model(case) import — expanded in
    # _internal/build.py::emit_pattern_spec (flat) and
    # _emit_patterns_partitioned (per-rank). The DOF-agnostic 3D→ndf
    # mapping now lives in build.py::broker_load_components.

    def _emit_patterns_partitioned(
        self,
        emitter: Emitter,
        post_element: "list[Primitive]",
        owned_nodes: "set[int] | SortedIntSet",
        *,
        primary_nodes: "set[int] | SortedIntSet",
        inferred_ndf: "dict[int, int] | None" = None,
        claimed_pattern_ids: "frozenset[int]" = frozenset(),
    ) -> None:
        """Per-rank pattern fan-out (ADR 0027).

        Walks every :class:`Pattern` primitive (skipping recorders,
        which emit globally outside any partition block) and emits
        only the ``p.load`` / ``p.sp`` rows targeting nodes owned by
        this rank.  ``load`` lines are ADDITIVE under OpenSeesMP
        assembly and filter on ``primary_nodes`` (each node on exactly
        one rank — see :func:`primary_owner_map`); ``sp`` lines are
        idempotent constraints and keep the full ``owned_nodes``
        fan-out (every domain holding the node needs the constraint).
        Non-Plain patterns delegate verbatim — they have no per-node
        fan-out to filter, and OpenSeesMP handles them with their own
        per-rank semantics (e.g. ``UniformExcitation`` applies on
        every rank simultaneously).

        ADR 0051 (BL-3): patterns claimed by ``s.pattern(...)`` are
        SKIPPED here — they emit inside their owning stage's per-rank
        block via :meth:`_emit_stages_partitioned`.
        """
        for p in post_element:
            if not isinstance(p, Pattern):
                continue
            if id(p) in claimed_pattern_ids:
                continue
            self._emit_one_pattern_partitioned(
                emitter, p, owned_nodes,
                primary_nodes=primary_nodes,
                inferred_ndf=inferred_ndf,
            )

    def _emit_one_pattern_partitioned(
        self,
        emitter: Emitter,
        p: "Pattern",
        owned_nodes: "set[int] | SortedIntSet",
        *,
        primary_nodes: "set[int] | SortedIntSet",
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> bool:
        """Emit one pattern's rank-owned ``load`` / ``sp`` lines.

        ``load`` lines filter on ``primary_nodes`` (additive under MP
        assembly — one rank per node); ``sp`` lines filter on
        ``owned_nodes`` (idempotent constraints — every owning rank).

        Returns ``True`` if a ``pattern_open`` block was emitted (the
        rank owns at least one load / sp / from_model line, or the
        pattern is non-Plain and emits on every rank), ``False`` if the
        pattern had no content for this rank (so the caller can skip an
        empty ``partition_open`` bracket — an empty ``if getPID()==K:``
        body is a Python ``SyntaxError`` on the Py emitter).
        """
        from .pattern.pattern import Plain
        from ._internal.tag_resolution import resolve_tag

        tag = self.tag_for[id(p)]
        if not isinstance(p, Plain):
            # Non-Plain pattern (UniformExcitation etc.) — emit on
            # every rank verbatim.  Per ADR 0027 these patterns have
            # no per-node fan-out the bridge can filter; the OpenSeesMP
            # semantics for them are pattern-class-specific.
            p._emit(emitter, tag)
            return True

        eff = inferred_ndf or {}

        def ndf_of(n: int) -> int:
            return int(eff.get(int(n), self.ndf))

        ts_tag = resolve_tag(emitter, p.series)
        # Pre-filter loads / sps so we don't open an empty pattern.
        owned_loads = [
            rec for rec in p.loads
            if _pattern_record_owned(rec, primary_nodes, self.fem)
        ]
        owned_sps = [
            rec for rec in p.sps
            if _pattern_record_owned(rec, owned_nodes, self.fem)
        ]
        # ADR 0051: from_model(case) imports, expanded + rank-filtered.
        fm_loads, fm_sps = self._owned_from_model_lines(
            p.from_model_cases,
            load_nodes=primary_nodes,
            sp_nodes=owned_nodes,
            inferred_ndf=eff,
        )
        # ADR 0062: moment-tensor sources → rank-owned nodal load lines.
        mt_loads = self._owned_moment_tensor_lines(
            p.moment_tensors,
            load_nodes=primary_nodes,
            inferred_ndf=eff,
        )
        if (not owned_loads and not owned_sps
                and not fm_loads and not fm_sps and not mt_loads):
            return False
        emitter.pattern_open("Plain", tag, ts_tag)
        for load_rec in owned_loads:
            _emit_pattern_load_partitioned(
                load_rec, emitter, self.fem, primary_nodes, ndf_of,
            )
        for sp_rec in owned_sps:
            _emit_pattern_sp_partitioned(
                sp_rec, emitter, self.fem, owned_nodes,
            )
        for node_id, comps in fm_loads:
            emitter.load(node_id, *comps)
        for node_id, dof, value in fm_sps:
            emitter.sp(node_id, dof, value)
        for node_id, comps in mt_loads:
            emitter.load(node_id, *comps)
        emitter.pattern_close()
        return True

    def _stage_pattern_specs_have_owned_content(
        self,
        specs: "tuple[Plain, ...]",
        owned_nodes: "set[int] | SortedIntSet",
        *,
        primary_nodes: "set[int] | SortedIntSet",
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> bool:
        """Pure pre-check: would any ``specs`` pattern emit a line for
        this rank?  Used to skip opening an empty ``partition_open``
        bracket in the staged partitioned pattern pass (BL-3).

        Mirrors :meth:`_emit_one_pattern_partitioned` exactly: loads
        check against ``primary_nodes``, sps against ``owned_nodes`` —
        the pre-check must match what would actually emit, or a rank
        whose only content is a non-primary shared loaded node opens an
        empty bracket (a Python ``SyntaxError`` on the Py emitter).
        """
        from .pattern.pattern import Plain

        for p in specs:
            if not isinstance(p, Plain):
                return True  # non-Plain emits on every rank
            if any(
                _pattern_record_owned(rec, primary_nodes, self.fem)
                for rec in p.loads
            ):
                return True
            if any(
                _pattern_record_owned(rec, owned_nodes, self.fem)
                for rec in p.sps
            ):
                return True
            fm_loads, fm_sps = self._owned_from_model_lines(
                p.from_model_cases,
                load_nodes=primary_nodes,
                sp_nodes=owned_nodes,
                inferred_ndf=inferred_ndf,
            )
            if fm_loads or fm_sps:
                return True
            if self._owned_moment_tensor_lines(
                p.moment_tensors,
                load_nodes=primary_nodes,
                inferred_ndf=inferred_ndf,
            ):
                return True
        return False

    def _owned_from_model_lines(
        self,
        cases: "tuple[str, ...]",
        *,
        load_nodes: "set[int] | SortedIntSet",
        sp_nodes: "set[int] | SortedIntSet",
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> "tuple[list[tuple[int, tuple[float, ...]]], list[tuple[int, int, float]]]":
        """Expand from_model ``cases`` to rank-owned (load, sp) lines.

        Mirrors the flat ``emit_pattern_spec`` from_model expansion, but
        filters to nodes owned by the current rank (ADR 0027 / 0051).
        ``load`` lines filter on ``load_nodes`` (the rank's PRIMARY-owned
        set — loads are additive under MP assembly, one rank per node;
        see :func:`primary_owner_map`); ``sp`` lines filter on
        ``sp_nodes`` (the rank's full owned set — constraints replicate
        on every owning rank). The DOF-agnostic spatial load is mapped
        onto **each node's** effective ndf (``inferred_ndf`` — envelope
        fallback), not the model envelope.
        """
        from ._internal.build import broker_load_components

        eff = inferred_ndf or {}

        fm_loads: list[tuple[int, tuple[float, ...]]] = []
        fm_sps: list[tuple[int, int, float]] = []
        if not cases:
            return fm_loads, fm_sps
        nodes = getattr(self.fem, "nodes", None)
        if nodes is None:
            return fm_loads, fm_sps
        load_set = getattr(nodes, "loads", None)
        sp_set = getattr(nodes, "sp", None)
        for case in cases:
            if load_set is not None:
                for rec in load_set.by_pattern(case):
                    if int(rec.node_id) in load_nodes:
                        node_ndf = int(eff.get(int(rec.node_id), self.ndf))
                        fm_loads.append(
                            (int(rec.node_id),
                             broker_load_components(rec, node_ndf, self.ndm)),
                        )
            if sp_set is not None:
                for rec in sp_set.prescribed():
                    if rec.pattern == case and int(rec.node_id) in sp_nodes:
                        fm_sps.append((int(rec.node_id), rec.dof, rec.value))
        return fm_loads, fm_sps

    def _owned_moment_tensor_lines(
        self,
        moment_tensors: "tuple[Any, ...]",
        *,
        load_nodes: "set[int] | SortedIntSet",
        inferred_ndf: "dict[int, int] | None" = None,
    ) -> "list[tuple[int, tuple[float, ...]]]":
        """Resolve moment-tensor sources to rank-owned nodal load lines (ADR 0062).

        Mirrors :meth:`_owned_from_model_lines`: the source's
        representation-theorem nodal forces are additive MP loads (one
        rank per node), so each ``(node, force)`` pair is kept only when
        the node is in this rank's PRIMARY-owned set ``load_nodes``. The
        host search and force build run against the full FEM snapshot (the
        host element may straddle a partition boundary); only the emitted
        lines are rank-filtered, so the per-rank union reproduces the same
        deck the flat path emits. Returns ``[(node, components), ...]``.
        """
        from ._internal.build import (
            broker_load_components,
            resolve_moment_tensor_pairs,
        )
        from apeGmsh._kernel.records._loads import NodalLoadRecord

        eff = inferred_ndf or {}
        out: list[tuple[int, tuple[float, ...]]] = []
        for rec in moment_tensors:
            # Resolve the rank-independent host search ONCE per build; every
            # other rank (and the staged pre-check) reuses the cached pairs.
            pairs = self._mt_pairs_cache.get(id(rec))
            if pairs is None:
                pairs = resolve_moment_tensor_pairs(rec, self.fem)
                self._mt_pairs_cache[id(rec)] = pairs
            for node, force in pairs:
                if int(node) in load_nodes:
                    node_ndf = int(eff.get(int(node), self.ndf))
                    nl = NodalLoadRecord(
                        node_id=int(node),
                        force_xyz=(
                            float(force[0]), float(force[1]), float(force[2]),
                        ),
                    )
                    out.append(
                        (int(node),
                         broker_load_components(nl, node_ndf, self.ndm)),
                    )
        return out

    # -- Auto-emit constraint handler (Phase 8 fold-in) ----------------

    def _maybe_auto_emit_constraint_handler(
        self,
        emitter: Emitter,
        pre_element: "list[Primitive]",
    ) -> None:
        """Auto-emit ``constraints("Transformation")`` when MP
        constraints are present in the FEM AND the user did not
        explicitly declare a constraint handler.

        Addresses the Phase 7b footgun: the default OpenSees handler
        ``Plain`` silently ignores ``equalDOF`` / ``rigidLink`` /
        ``rigidDiaphragm`` / surface-coupling records.

        Behaviour matrix (Phase 8 fold-in):

        +--------------------------------+-------------------------------+
        | User declared handler          | MP constraints present?       |
        +--------------------------------+-------------------------------+
        | (none)                         | yes -> auto-emit Transformation |
        |                                |        + UserWarning            |
        +--------------------------------+-------------------------------+
        | ``Plain``                      | yes -> UserWarning (different   |
        |                                |        message; user's choice   |
        |                                |        respected, Plain already |
        |                                |        emitted)                 |
        +--------------------------------+-------------------------------+
        | ``Penalty`` / ``Transformation``| no warning, no auto-emit       |
        | / ``Lagrange``                 |                               |
        +--------------------------------+-------------------------------+
        | (any)                          | no MP -> no-op                  |
        +--------------------------------+-------------------------------+
        """
        import warnings as _warnings

        # Contact requires the LadrunoContact handler (it injects the contact
        # FE adapters into the assembly). It supersedes the Transformation
        # auto-emit: emit it whenever any contact interaction is present.
        if _fem_has_contacts(self.fem):
            # An enforce="equation" tie (EQ_Constraint) needs the Lagrange
            # (implicit) or LadrunoProjection (explicit) handler — but contact
            # needs LadrunoContact, and only ONE constraint handler can be
            # active. The two are mutually exclusive. Without this guard the
            # contact branch would emit LadrunoContact and return, silently
            # dropping the equation tie (LadrunoContact cannot enforce
            # EQ_Constraint). Fail loud rather than emit a broken deck.
            if self._has_equation_constraints():
                raise BridgeError(
                    "Contact interactions and an enforce='equation' tie "
                    "(EQ_Constraint) are both present, but they require "
                    "different, mutually exclusive constraint handlers — "
                    "contact needs 'LadrunoContact' while an equation tie "
                    "needs 'Lagrange' (implicit) or 'LadrunoProjection' "
                    "(explicit). A single OpenSees constraint handler cannot "
                    "enforce both. Split the model into separate runs, or "
                    "switch the tie to enforce='penalty' / 'penalty_al'."
                )
            # The LadrunoContact handler is Plain-style for MP constraints
            # (fork P1a): with contact active, true MP_Constraints (equalDOF /
            # equalDOF_mixed / rigidLink / rigidDiaphragm) are NOT enforced.
            # Only one constraint handler can be active, and it must be
            # LadrunoContact for the contact FE adapters — so a model with both
            # contact and a handler-requiring MP constraint would silently drop
            # the MP enforcement. Fail loud rather than emit a silently-wrong
            # deck. NB this checks only HANDLER-REQUIRING constraints: penalty/
            # penalty_al ties + kinematic/distributing couplings emit
            # handler-INDEPENDENT elements and coexist with contact fine;
            # equation ties are caught above. rigid_body(as_element=True) is
            # NOT handler-independent — LadrunoRigidBody::setDomain still adds
            # one MP_Constraint per slave, which LadrunoContact only warns
            # about instead of enforcing (fork LadrunoContactHandler.cpp:
            # 706-713) — so it trips this guard the same as as_element=False.
            if _fem_has_handler_requiring_mp(self.fem):
                raise BridgeError(
                    "Contact interactions and a handler-requiring MP "
                    "constraint (equalDOF / equalDOF_mixed / rigidLink / "
                    "rigidDiaphragm / rigid_body, including "
                    "rigid_body(as_element=True) — LadrunoRigidBody still "
                    "emits an MP_Constraint per slave) are both present, but "
                    "the LadrunoContact handler required for contact is "
                    "Plain-style for MP constraints (fork P1a) — the MP "
                    "constraint would NOT be enforced. Only one constraint "
                    "handler can be active. Split the model into separate "
                    "runs, or remove the MP constraint from the contact run. "
                    "(Penalty/penalty_al ties and kinematic/distributing "
                    "couplings are handler-independent elements and are fine "
                    "with contact.)"
                )
            from .analysis.constraint_handler import (
                LadrunoContact as _LadrunoContactHandler,
            )
            declared = next(
                (p for p in pre_element if isinstance(p, ConstraintHandler)), None)
            # Declaring 'LadrunoContact' yourself is agreement, not conflict —
            # warning there cried wolf on the one correct choice (and on every
            # staged contact model, where _validate_staged_contact_handlers now
            # REQUIRES each stage to declare exactly this handler).
            if declared is not None and not isinstance(
                declared, _LadrunoContactHandler,
            ):
                _warnings.warn(
                    "Contact interactions are present but a constraint handler "
                    f"('{type(declared).__name__}') was declared — contact "
                    "requires 'LadrunoContact', so it is (re)emitted here. NOTE "
                    "the emit ORDER decides which one the analysis is built "
                    "with: a later declaration (e.g. a stage's own chain) wins. "
                    "Remove the explicit handler, or declare "
                    "ops.constraints.LadrunoContact(), to silence this.",
                    OpenSeesAutoEmitWarning, stacklevel=2,
                )
            emitter.constraints("LadrunoContact")
            return

        if not (
            _fem_has_mp_constraints(self.fem)
            or self.equation_constraint_records
        ):
            return

        # Find any user-declared ConstraintHandler in pre_element.
        # (Constraint handlers go to pre_element because they're not
        # Pattern or Recorder.)
        from .analysis.constraint_handler import (
            Auto as ConstraintsAuto,
            Lagrange as ConstraintsLagrange,
            Plain as ConstraintsPlain,
            Transformation as ConstraintsTransformation,
        )

        import warnings as _warnings

        # ADR 0068 (INV-4): an enforce="equation" tie emits
        # equationConstraint (EQ_Constraint), which the Transformation
        # handler CANNOT enforce — it would silently drop the tie. When
        # any equation tie is present the handler must be Lagrange
        # (implicit, exact) or the fork LadrunoProjection (explicit).
        has_eq = self._has_equation_constraints()

        declared_handler: "ConstraintHandler | None" = None
        for p in pre_element:
            if isinstance(p, ConstraintHandler):
                declared_handler = p
                break

        if declared_handler is None:
            if has_eq:
                # Equation ties present (ADR 0068 Open item 1): auto-detect
                # implicit vs explicit from the registered integrator and emit
                # the right EQ-capable handler. Explicit → LadrunoProjection
                # (fork; Δt-neutral, momentum-conserving — a Lagrange
                # multiplier's massless DOF would break the explicit mass
                # solve). Implicit / no integrator → Lagrange (exact).
                # 'Transformation' cannot enforce EQ_Constraint and is never
                # auto-emitted here (INV-4).
                integrator = next(
                    (p for p in pre_element if isinstance(p, Integrator)), None,
                )
                if _is_explicit_integrator(integrator):
                    _warnings.warn(
                        "An enforce='equation' tie (EQ_Constraint) is present "
                        f"with an explicit integrator "
                        f"({type(integrator).__name__}). Auto-emitting the "
                        "fork 'LadrunoProjection' constraint handler "
                        "(Δt-neutral, momentum-conserving). It is fork-only — "
                        "a stock build fails loud. To override, declare "
                        "ops.constraints.X() before build().",
                        OpenSeesAutoEmitWarning,
                        stacklevel=2,
                    )
                    emitter.constraints("LadrunoProjection")
                    return
                _warnings.warn(
                    "An enforce='equation' tie (EQ_Constraint) is present. "
                    "Auto-emitting 'Lagrange' constraint handler (exact; "
                    "correct for implicit analysis). For an EXPLICIT "
                    "transient run, register an explicit integrator (e.g. "
                    "ops.integrator.CentralDifferenceLadruno()) before build() "
                    "so the fork 'LadrunoProjection' is auto-emitted instead, "
                    "or declare ops.constraints.LadrunoProjection() yourself "
                    "(Δt-neutral). 'Transformation' cannot enforce "
                    "EQ_Constraint and is never auto-emitted here.",
                    OpenSeesAutoEmitWarning,
                    stacklevel=2,
                )
                emitter.constraints("Lagrange")
                self._warn_lagrange_norm_disp_incr(pre_element)
                return
            # No user-declared handler — auto-emit Transformation.
            _warnings.warn(
                "MP constraints are present in the model (equalDOF, "
                "rigidLink, rigidDiaphragm, or surface couplings). "
                "Auto-emitting 'Transformation' constraint handler. "
                "To override, explicitly declare ops.constraints.X() "
                "before build().",
                OpenSeesAutoEmitWarning,
                stacklevel=2,
            )
            emitter.constraints("Transformation")
            return

        if isinstance(declared_handler, ConstraintsPlain):
            # User explicitly declared Plain + MP constraints present.
            # Plain is already emitted (pre_element pass); just warn.
            _warnings.warn(
                "MP constraints present but Plain handler explicitly "
                "declared — MP constraints will be silently ignored. "
                "Did you mean Transformation/Lagrange/Penalty?",
                OpenSeesAutoEmitWarning,
                stacklevel=2,
            )
            return

        # ADR 0068 INV-4: fail loud on Transformation/Auto + equation tie —
        # NEITHER handler enforces EQ_Constraint (Transformation has no EQ
        # path; AutoConstraintHandler iterates only SP + MP, never getEQs()),
        # so the deck would run but silently drop the tie.
        if has_eq and isinstance(
            declared_handler, (ConstraintsTransformation, ConstraintsAuto)
        ):
            kind = type(declared_handler).__name__
            raise ValueError(
                f"Constraint handler '{kind}' was declared, but the model "
                "has an enforce='equation' tie (EQ_Constraint), which "
                f"'{kind}' cannot enforce — the deck would silently drop the "
                "tie. Declare ops.constraints.Lagrange() (implicit) or "
                "ops.constraints.LadrunoProjection() (explicit, fork), or "
                "switch the tie to enforce='penalty'."
            )

        # ADR 0068 Open item 1 (soft): a user-declared 'Lagrange' with an
        # explicit integrator + equation tie is enforceable but hazardous —
        # Lagrange adds massless multiplier DOFs the explicit central-
        # difference mass solve cannot invert. Warn (don't fail) and point at
        # the Δt-neutral LadrunoProjection.
        if has_eq and isinstance(declared_handler, ConstraintsLagrange):
            integrator = next(
                (p for p in pre_element if isinstance(p, Integrator)), None,
            )
            if _is_explicit_integrator(integrator):
                _warnings.warn(
                    "Constraint handler 'Lagrange' was declared with an "
                    f"explicit integrator ({type(integrator).__name__}) and an "
                    "enforce='equation' tie. Lagrange introduces massless "
                    "multiplier DOFs that an explicit mass solve cannot invert "
                    "— prefer ops.constraints.LadrunoProjection() (fork; "
                    "Δt-neutral) for explicit runs.",
                    OpenSeesAutoEmitWarning,
                    stacklevel=2,
                )
            self._warn_lagrange_norm_disp_incr(pre_element)
            return
        # Any other explicit handler — no warning, no auto-emit.

    @staticmethod
    def _warn_lagrange_norm_disp_incr(
        pre_element: "list[Primitive]",
    ) -> None:
        """Warn on Lagrange + an absolute ``NormDispIncr`` test.

        Under the Lagrange handler the multiplier DOFs enter the
        solution vector with force-like scaling (interface forces, e.g.
        ~1e7 N) while the real DOFs carry displacements (e.g. mm), so an
        absolute displacement-increment norm can be numerically
        unreachable however exact the solve is — the analysis reports
        failure-to-converge on a converged solution.  Silent-failures
        program, defect-2 companion.
        """
        import warnings as _warnings

        from .analysis.test import NormDispIncr as _NDI

        test = next((p for p in pre_element if isinstance(p, _NDI)), None)
        if test is None:
            return
        _warnings.warn(
            "Constraint handler 'Lagrange' with a NormDispIncr "
            "convergence test: the Lagrange-multiplier DOFs enter the "
            "displacement-increment norm with force-like magnitudes, so "
            "a tight absolute tolerance can be unreachable even on an "
            "exactly converged solve. Prefer ops.test.NormUnbalance(...) "
            "or a relative test under Lagrange.",
            OpenSeesAutoEmitWarning,
            stacklevel=3,
        )

    def _validate_staged_eq_handlers(self) -> None:
        """ADR 0068 Open item 5 — staged-path EQ handler guard.

        The global EQ-aware auto-emit (:meth:`_maybe_auto_emit_constraint_
        handler`) does NOT run for staged models: each stage declares its
        own analysis chain (incl. ``stage.constraints``).  Without this
        guard an ``enforce="equation"`` tie (an ``equationConstraint`` that
        lives in the domain across all stages, or a stage-bound one) is
        SILENTLY DROPPED by any stage whose handler is ``Transformation`` /
        ``Auto`` / ``Plain`` (or absent → OpenSees default ``Plain``) — the
        same failure mode INV-4 fails loud on for the non-staged path.

        Per stage that must enforce an equation tie, require an EQ-capable
        handler (``Lagrange`` / ``Penalty`` / ``LadrunoProjection``); fail
        loud otherwise.  No-op when no equation ties exist (MP-only staged
        models are unaffected).
        """
        from .analysis.constraint_handler import (
            Lagrange as _Lag, LadrunoProjection as _Proj, Penalty as _Pen,
        )
        # Read the field directly (not _has_equation_constraints): the guard
        # is also driven on duck-typed stand-ins that carry only fem +
        # stage_records (tests/test_staged_eq_handler_guard.py).
        global_eq = _fem_has_equation_ties(self.fem) or bool(
            getattr(self, "equation_constraint_records", ()),
        )
        eq_ok = (_Lag, _Pen, _Proj)
        for stage in self.stage_records:
            needs_eq = global_eq or _records_have_equation_tie(
                getattr(stage, "stage_constraint_records", ()))
            if not needs_eq:
                continue
            handler = stage.constraints
            if isinstance(handler, eq_ok):
                continue
            if handler is None:
                detail = (
                    "declares no constraint handler (OpenSees defaults to "
                    "'Plain', which drops EQ_Constraint)")
            else:
                detail = (
                    f"declares constraint handler "
                    f"'{type(handler).__name__}', which cannot enforce "
                    f"EQ_Constraint")
            raise ValueError(
                f"stage {stage.name!r}: an enforce='equation' tie is "
                f"present but the stage {detail} — the deck would silently "
                f"drop the tie. Declare s.constraints.Lagrange() (implicit) "
                f"or s.constraints.LadrunoProjection() (explicit, fork) on "
                f"the stage, or switch the tie to enforce='penalty'."
            )

    def _validate_staged_contact_handlers(self) -> None:
        """ADR 0092 — staged-path CONTACT handler guard.

        The contact analogue of :meth:`_validate_staged_eq_handlers`,
        and the same failure mode the staged **partitioned** refusal
        already names: "the partitioned staged pipeline skips the
        analysis-chain auto-emit … so the forced ``LadrunoContact``
        handler would never be emitted and the interaction would be
        silently unenforced". That reasoning is not partition-specific
        — the SERIAL staged path had the same hole, hidden because the
        global auto-emit *does* emit ``constraints LadrunoContact`` and
        so looks like cover. It is not: every stage re-declares the
        whole chain, so the stage's own ``constraints`` line lands
        AFTER the global one and BEFORE the stage's ``analysis``, and
        the analysis is constructed with the stage's handler.

        Measured on a staged two-block contact deck whose stage
        declared ``Transformation``::

            contact 1 1 2 …            ← the interaction
            constraints LadrunoContact ← global auto-emit
            constraints Transformation ← the stage's handler
            analysis Static            ← constructed with Transformation

        The contact FE adapters are never injected, so the interaction
        does nothing and nothing says so — the exact silent-wrong class
        ADR 0092 exists to prevent. Since ``s.analysis()`` *requires*
        ``constraints=``, the wrong handler is always reachable.

        Fail loud, naming the stage and its handler. No-op when the
        model carries no contact interaction.
        """
        if not self.stage_records or not _fem_has_contacts(self.fem):
            return
        from .analysis.constraint_handler import (
            LadrunoContact as _LadrunoContact,
        )
        for stage in self.stage_records:
            handler = stage.constraints
            if isinstance(handler, _LadrunoContact):
                continue
            detail = (
                "declares no constraint handler"
                if handler is None
                else f"declares constraint handler "
                     f"'{type(handler).__name__}'"
            )
            raise BridgeError(
                f"stage {stage.name!r}: a g.constraints.contact / "
                f"contact_plane interaction is present but the stage "
                f"{detail} — contact requires 'LadrunoContact' (it injects "
                f"the contact FE adapters into the assembly), and a stage "
                f"re-declares the whole analysis chain, so the stage's "
                f"handler is the one the stage's 'analysis' is built with. "
                f"The deck would run with the interaction SILENTLY "
                f"UNENFORCED. Declare s.constraints.LadrunoContact() on "
                f"every stage of a contact model."
            )

    # -- Auto-emit parallel numberer / system (ADR 0027 INV-5) -----------

    def _maybe_auto_emit_parallel_numberer(
        self,
        emitter: Emitter,
        pre_element: "list[Primitive]",
    ) -> None:
        """Auto-emit ``numberer ParallelPlain`` under partitioning when
        the user has not explicitly declared a numberer (ADR 0027 INV-5).

        Behaviour:

        * No user numberer + ``len(fem.partitions) > 1`` → emit
          ``numberer ParallelPlain`` (single ``UserWarning``).
        * User declared ``Plain`` / ``RCM`` (serial) +
          ``len(fem.partitions) > 1`` → ``UserWarning`` flagging the
          MP-incompatibility; the user's choice is preserved verbatim
          (already emitted by the pre_element pass).
        * User declared ``ParallelPlain`` / ``ParallelRCM`` → no
          warning, no auto-emit.
        """
        import warnings as _warnings

        declared_numberer: "Numberer | None" = None
        for p in pre_element:
            if isinstance(p, Numberer):
                declared_numberer = p
                break

        if declared_numberer is None:
            _warnings.warn(
                "len(fem.partitions) > 1 with no user-declared numberer; "
                "auto-emitting runtime-conditional 'numberer ParallelPlain' "
                "with 'RCM' fallback so the deck runs under both OpenSeesMP "
                "and single-process OpenSees (ADR 0027 INV-5).  Explicitly "
                "declare ops.numberer.<Plain|RCM|ParallelPlain|ParallelRCM>() "
                "before build() to override.",
                OpenSeesAutoEmitWarning,
                stacklevel=2,
            )
            emitter.parallel_runtime_fallback_numberer(
                "ParallelPlain", "RCM",
            )
            return

        # User-declared numberer — check MP compatibility.
        token = type(declared_numberer).__name__
        if token in {"Plain", "RCM", "AMD"}:
            _warnings.warn(
                f"len(fem.partitions) > 1 with serial numberer "
                f"{token!r} explicitly declared — OpenSeesMP requires "
                "a parallel numberer ('ParallelPlain' or 'ParallelRCM') "
                "for correct DOF numbering across ranks. The user's "
                "choice is preserved; switch to ops.numberer.ParallelPlain() "
                "or ops.numberer.ParallelRCM() for a runnable parallel "
                "deck.",
                OpenSeesAutoEmitWarning,
                stacklevel=2,
            )

    def _maybe_auto_emit_parallel_system(
        self,
        emitter: Emitter,
        pre_element: "list[Primitive]",
    ) -> None:
        """Auto-emit ``system Mumps`` under partitioning when the user
        has not explicitly declared a system of equations (ADR 0027
        INV-5).  Mirror of :meth:`_maybe_auto_emit_parallel_numberer`.
        """
        import warnings as _warnings

        declared_system: "LinearSystem | None" = None
        for p in pre_element:
            if isinstance(p, LinearSystem):
                declared_system = p
                break

        if declared_system is None:
            _warnings.warn(
                "len(fem.partitions) > 1 with no user-declared system; "
                "auto-emitting runtime-conditional 'system Mumps' with "
                "'UmfPack' fallback so the deck runs under both OpenSeesMP "
                "and single-process OpenSees (ADR 0027 INV-5).  Explicitly "
                "declare ops.system.<Mumps|MumpsParallel>() before build() "
                "to override.",
                OpenSeesAutoEmitWarning,
                stacklevel=2,
            )
            emitter.parallel_runtime_fallback_system(
                "Mumps", "UmfPack",
            )
            return

        # User-declared system — check MP compatibility.
        token = type(declared_system).__name__
        # Heuristic incompatibility list (the serial systems most users
        # default to).  Mumps / MumpsParallel are MP-OK; everything else
        # warns.
        if token in {
            "BandSPD", "BandGen", "ProfileSPD", "SparseGeneral",
            "SparseSPD", "FullGeneral", "UmfPack", "Pardiso",
        }:
            _warnings.warn(
                f"len(fem.partitions) > 1 with serial system "
                f"{token!r} explicitly declared — OpenSeesMP requires "
                "a parallel system ('Mumps' typically). The user's "
                "choice is preserved; switch to ops.system.Mumps() for "
                "a runnable parallel deck.",
                OpenSeesAutoEmitWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Module-level pattern record helpers (ADR 0027) — keep imports off the
# hot path and the bridge body small.
# ---------------------------------------------------------------------------


def _pattern_record_owned(
    rec: "Any", owned_nodes: "set[int] | SortedIntSet", fem: "FEMData",
) -> bool:
    """True iff ``rec``'s ``pg``/``node`` targets include any owned node."""
    target_kind = getattr(rec, "target_kind", "node")
    target = getattr(rec, "target", None)
    if target_kind == "node":
        if target is None:
            return False
        try:
            return int(target) in owned_nodes
        except (TypeError, ValueError):
            return False
    # PG target — at least one of the PG's nodes must be owned.
    try:
        ids = fem.nodes.select(pg=target).ids
    except (KeyError, ValueError, AttributeError):
        return False
    for nid in ids:
        if int(nid) in owned_nodes:
            return True
    return False


def _emit_pattern_load_partitioned(
    rec: "Any",
    emitter: Emitter,
    fem: "FEMData",
    owned_nodes: "set[int] | SortedIntSet",
    ndf_of: "Callable[[int], int]",
) -> None:
    """Per-rank version of the inner load fan-out."""
    if rec.target_kind == "node":
        node = int(rec.target)
        if node in owned_nodes:
            emitter.load(node, *fit_dof_vector(
                rec.forces, ndf_of(node), kind="nodal load", node=node))
        return
    # PG — fan out only owned nodes.
    try:
        ids = fem.nodes.select(pg=rec.target).ids
    except (KeyError, ValueError, AttributeError):
        return
    for node_tag in ids:
        if int(node_tag) in owned_nodes:
            emitter.load(int(node_tag), *fit_dof_vector(
                rec.forces, ndf_of(int(node_tag)), kind="nodal load",
                node=int(node_tag)))


def _emit_pattern_sp_partitioned(
    rec: "Any", emitter: Emitter, fem: "FEMData",
    owned_nodes: "set[int] | SortedIntSet",
) -> None:
    """Per-rank version of the inner sp fan-out."""
    if rec.target_kind == "node":
        if int(rec.target) in owned_nodes:
            emitter.sp(int(rec.target), rec.dof, rec.value)
        return
    try:
        ids = fem.nodes.select(pg=rec.target).ids
    except (KeyError, ValueError, AttributeError):
        return
    for node_tag in ids:
        if int(node_tag) in owned_nodes:
            emitter.sp(int(node_tag), rec.dof, rec.value)


# ---------------------------------------------------------------------------
# apeSees — the bridge
# ---------------------------------------------------------------------------

def _warn_if_shown(message: str, category: type[Warning]) -> bool:
    """Warn at the first caller outside apeGmsh; ``True`` iff the active
    filters let it through to the user.

    The automatic ``model.h5`` write silences ``H5FeatureDeferredWarning``,
    so whether a deferred warning was seen depends on the caller's
    filters. The warning is recorded under those same filters and then
    re-issued to the real handler.
    """
    import warnings as _warnings

    from ._internal.build import _stacklevel_outside_package

    with _warnings.catch_warnings(record=True) as seen:
        _warnings.warn(
            message, category, stacklevel=_stacklevel_outside_package())
    for w in seen:
        _warnings.warn_explicit(
            w.message, w.category, w.filename, w.lineno, source=w.source)
    return bool(seen)


class apeSees(_ContactQueryMixin, _ModalMixin, _FrfMixin, _ExplicitMixin):
    """The OpenSees bridge.

    Construct with a :class:`~apeGmsh.mesh.FEMData` snapshot:

    .. code-block:: python

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
        ...

    The bridge holds **declared** state. ``apeSees.build()`` returns a
    :class:`BuiltModel` (immutable) that emitters consume.

    Parameters
    ----------
    fem
        The FEM snapshot the bridge is built against.
    default_orientation
        Orientation field substituted on any
        ``ops.geomTransf.<Type>()`` call where the user supplied
        neither ``orientation=`` nor ``vecxz=``. Defaults to
        ``Cartesian()`` (Z-up) which matches the prevailing structural
        convention. Pass an explicit ``None`` for 2D models, where
        vecxz is omitted at emit time and an orientation field makes
        no sense. Pass a custom orientation (e.g.
        ``Cartesian(reference_axis=(0,1,0))`` for a Y-up CAD import)
        to set the model-wide default once.
    element_tags
        How emitted elements are numbered. ``"sequential"`` (default)
        tags them 1, 2, ... in declaration order. ``"fem"`` (ADR 0111
        D2) keeps every physical-group element's FEM element id as its
        OpenSees tag, so a deck translated from another pre-processor
        (STKO) keeps that tool's element ids; elements the bridge
        synthesises (node-pair springs, interface ``zeroLength``,
        embedded / rebar / rigid-body / coupling elements) are numbered
        above the largest FEM id. ``"fem"`` refuses an id ``<= 0`` and
        an element fanned out by two declarations (``BridgeError`` at
        emit).
    _artifacts
        ADR 0112 D1: every terminal emit and live build leaves the full
        ``model.h5`` at the session's conventional path
        (:mod:`~apeGmsh.opensees._internal.artifact_write`).  ``False``
        is for the library's own bridges only (importers, ``strut_tie``,
        the emit-cost bench); it is not a user-facing opt-out.
    """

    def __init__(
        self,
        fem: "FEMData",
        *,
        default_orientation: Orientation | None | _UnsetType = _UNSET,
        opensees: "OpenSeesTarget | None" = None,
        element_tags: ElementTagMode = "sequential",
        _artifacts: bool = True,
    ) -> None:
        if element_tags not in ELEMENT_TAG_MODES:
            raise ValueError(
                f"apeSees: element_tags must be one of {ELEMENT_TAG_MODES}, "
                f"not {element_tags!r}."
            )
        self._fem: "FEMData" = fem
        # ADR 0111 D2 — how element tags are numbered at emit (BuiltModel).
        self._element_tags: ElementTagMode = element_tags
        # Last live emitter from :meth:`analyze` — retained so post-run live
        # queries (e.g. :meth:`ladruno_projection_tie_force`) can reach the
        # openseespy session that just ran. ``None`` until a live analyze runs.
        self._live_emitter: "LiveOpsEmitter | None" = None
        # Which OpenSees runtime the subprocess paths bind, and the live
        # fork expectation.  ``None`` → env-var / PATH fallback (the
        # pre-target behaviour).  See :mod:`apeGmsh.opensees._target`.
        self._opensees: "OpenSeesTarget | None" = opensees
        self._primitives: list[Primitive] = []
        # name -> primitive alias table (bridge-side; the primitive
        # stays pure/tag-less, so names never touch the lineage hash or
        # the h5 schema).  Populated by ``_register(..., name=...)`` and
        # read by ``_resolve`` so reference kwargs accept a name string
        # as well as the object handle.
        self._names: dict[str, Primitive] = {}
        # ADR 0112 D3 (V2d): where in the user's source each primitive
        # was declared.  Filled by ``_register`` through the shared
        # capture helper, one record per user call, keyed
        # ``opensees/<kind>/<name|#k>``; ``h5()`` appends the table to
        # the snapshot's own ``/provenance``.  Never hashed.
        self._provenance = ProvenanceStore()
        # ADR 0112 D1 (V2d-4b): the automatic model.h5 write, called once
        # at the end of each terminal emit / live build (not ``h5()``).
        self._artifacts = BridgeArtifactWriter(enabled=_artifacts)
        # ADR 0114 Q3: each distinct set of ``ledger`` verbs ``h5()``
        # warned about, so the automatic write and an explicit ``h5()``
        # warn once per set rather than on every emit.
        self._ledger_warned: set[frozenset[str]] = set()
        # Call ordinal of ``imposed_displacement``: the ``<name>`` of its
        # synthesised records when the call gives no ``name=``.
        self._imposed_displacement_calls = 0
        self._tags = TagAllocator()
        self._ndm: int | None = None
        self._ndf: int | None = None
        self._fix_records: list[FixRecord] = []
        self._equation_constraint_records: list[EquationConstraintRecord] = []
        self._mass_records: list[MassRecord] = []
        # ADR 0065 Tier 2 — opt-in: stream per-node masses from the snapshot
        # at emit instead of one bridge MassRecord per node. Set by
        # ``mass_from_model()``; threaded into the BuiltModel.
        self._mass_from_model: bool = False
        # ADR 0051 §4 — opt-in: fix every homogeneous SP on the snapshot.
        # Set by ``fix_from_model()``; materialized into FixRecords at build.
        self._fix_from_model: bool = False
        # ADR 0049 — ``ops.ndf`` directives (element-less decoupled nodes only).
        self._ndf_records: list[NdfRecord] = []
        self._region_records: list[RegionAssignmentRecord] = []
        self._rayleigh_records: list[RayleighRecord] = []
        self._damping_attach_records: list[DampingAttachRecord] = []
        self._modal_damping_records: list[ModalDampingRecord] = []
        self._initial_stress_records: list[InitialStressRecord] = []
        # Phase SSI-2.A: closed StageRecord instances accumulate here as
        # ``with ops.stage(name) as s:`` blocks exit.  ``stage_records``
        # being non-empty switches BuiltModel.emit into the staged
        # emission path (per-stage analyze loops with loadConst /
        # wipeAnalysis / hook-list clear between).
        self._stage_records: list[StageRecord] = []
        # Ladruno-fork stack profiler: ordered ``(verb, args)`` control
        # entries recorded by ``ops.profiler.<verb>(...)``.  The deck
        # emitters (tcl / py) flush these bracketing the appended
        # ``analyze`` line — ``start`` / ``reset`` before, ``stop`` /
        # ``report`` / ``memory`` after (see ``_split_profiler_records``).
        # Live single-call profiling does NOT consume this; it is driven by
        # the ``profile=`` kwarg family on :meth:`analyze`.
        self._profiler_records: list[
            tuple[str, tuple[int | float | str, ...]]
        ] = []
        # Tracks the currently-open _StageBuilder, if any, so
        # ``apeSees.stage()`` can refuse nested ``with`` blocks
        # (post-merge cleanup, red-team M4).  None when no stage is
        # being built.  Cleared by ``_StageBuilder.__exit__``.
        self._open_stage_builder: "_StageBuilder | None" = None
        # Phase SSI-2.D (PR-C) recorder claiming: when ``s.recorder(rec)``
        # PULLs a registered recorder spec into a stage's pool, the
        # spec's ``id(...)`` lands here so the global post-element
        # emit loop knows to SKIP it (the stage's emit will drive
        # ``_emit_recorder_spec`` inside the stage block instead).
        # The recorder stays in ``_primitives`` so its allocated tag
        # remains discoverable via ``tag_for[id(p)]``.
        self._stage_claimed_recorder_ids: set[int] = set()
        # Stage-bound constraint claiming: when ``s.embedded(name=...)``
        # / ``s.equal_dof(name=...)`` / etc. CLAIMS a resolved
        # constraint record from ``fem.{nodes,elements}.constraints``,
        # the record's ``id(...)`` lands here so the global MP-
        # constraint emit loop SKIPS it.  The record stays on the
        # FEMData broker (broker is immutable from the bridge's
        # perspective) but emits inside the owning stage's block
        # via ``emit_stage_mp_constraints``.  Doubles as the
        # double-claim detector across stage builders.
        self._stage_claimed_constraint_ids: set[int] = set()
        # ADR 0093 S7 (INV-6) stage-bound INTERFACE claiming, kept as a
        # PARALLEL id-set rather than folded into the MP set above: the
        # two pools are released and re-emitted by different passes
        # (``emit_stage_mp_constraints`` vs ``emit_stage_interfaces``),
        # and a shared set would let an interface claim satisfy — or
        # collide with — an MP claim.  Same semantics otherwise: the
        # record stays on ``fem.elements.interfaces`` and the base
        # ``emit_interfaces`` pass SKIPS it; doubles as the double-claim
        # detector across stage builders.
        self._stage_claimed_interface_ids: set[int] = set()
        # ADR 0051 (BL-3) stage-scoped pattern claiming: when
        # ``s.pattern(series=)`` creates a stage-owned ``Plain``, its
        # ``id(...)`` lands here so the global post-element pattern emit
        # loop SKIPS it (the stage's emit drives ``emit_pattern_spec`` /
        # ``_emit_one_pattern_partitioned`` inside the stage block).
        # The pattern stays in ``_primitives`` so its tag remains
        # discoverable via ``tag_for[id(p)]``.  Mirrors
        # ``_stage_claimed_recorder_ids``.
        self._stage_claimed_pattern_ids: set[int] = set()
        # ADR 0052 slice 1: the single shared ``Constant`` series (factor
        # 1.0) that every stage's HOLD pattern references.  Created
        # lazily on the first ``s.support(...)`` across any stage (so a
        # model with no supports emits no extra series), then reused —
        # one series, not one per stage (ADR 0052 Resolved decision §3).
        self._hold_series: "TimeSeries | None" = None
        # Resolve the sentinel: unset → Cartesian() (Z-up). Explicit
        # None disables the auto-default (2D models).
        if isinstance(default_orientation, _UnsetType):
            self._default_orientation: Orientation | None = Cartesian()
        else:
            self._default_orientation = default_orientation

        # Namespaces.
        self.uniaxialMaterial = _UniaxialMaterialNS(self)
        self.nDMaterial       = _NDMaterialNS(self)
        self.section          = _SectionNS(self)
        self.geomTransf       = _GeomTransfNS(self)
        self.beamIntegration  = _BeamIntegrationNS(self)
        self.timeSeries       = _TimeSeriesNS(self)
        self.pattern          = _PatternNS(self)
        self.fault            = _FaultNS(self)
        self.element          = _ElementNS(self)
        self.recorder         = _RecorderNS(self)
        self.profiler         = _ProfilerNS(self)
        self.damping          = _DampingNS(self)

        # FEM-aware aggregates (Phase 5A) — query-and-act over fem.nodes.
        self.nodes            = _NodeAccessor(self)
        self.constraints      = _ConstraintsNS(self)
        self.numberer         = _NumbererNS(self)
        self.system           = _SystemNS(self)
        self.test             = _TestNS(self)
        self.algorithm        = _AlgorithmNS(self)
        self.integrator       = _IntegratorNS(self)
        self.analysis         = _AnalysisNS(self)
        self.strategy         = _StrategyNS(self)

    # -- Read-only access to the FEM snapshot ----------------------------
    @property
    def fem(self) -> "FEMData":
        return self._fem

    # -- OpenSees runtime target / capabilities --------------------------
    @property
    def opensees(self) -> "OpenSeesTarget | None":
        """The :class:`OpenSeesTarget` bound on construction, or ``None``."""
        return self._opensees

    def capabilities(self) -> "OpenSeesCapabilities":
        """Probe the in-process openseespy build (live path).

        Imports openseespy in the active interpreter and reports whether
        it looks like the Ladruno fork (``has_fork``), exposes the
        fork-only ``profiler`` command, its ``version()`` string, and its
        ``build`` stamp (the exact git hash the binary was compiled from,
        on fork builds that ship ``ladrunoBuild``).
        Raises if openseespy is not installed.  This introspects the
        **live** runtime only — the subprocess paths bind their own
        interpreter / binary via :class:`OpenSeesTarget`.
        """
        from ._target import probe_live_capabilities

        return probe_live_capabilities()

    def _assert_fork_if_required(self) -> None:
        """Fail loud at the live boundary if ``require_fork`` is unmet.

        Called before driving any live :class:`LiveOpsEmitter`.  No-op
        unless an :class:`OpenSeesTarget` with ``require_fork=True`` was
        bound on construction.
        """
        target = self._opensees
        if target is None or not target.require_fork:
            return
        if not self.capabilities().has_fork:
            raise RuntimeError(
                "OpenSeesTarget(require_fork=True) but the in-process "
                "openseespy build is not the Ladruno fork (its "
                "BackendInfo reads kind='stock': ladrunoBuild() did not "
                "answer a git sha). Launch "
                "this script under a python whose openseespy is the fork "
                "build, or drop require_fork to run on stock OpenSees."
            )

    # -- Read-only union views over global + stage-bound BC pools --------
    # Phase SSI-2.D PR-B introspection symmetry (Red #19).  Tooling
    # that inspects ``bridge._fix_records`` to count fix declarations
    # would otherwise silently miss stage-bound fixes registered via
    # ``s.fix(...)``.  These properties return frozen tuples carrying
    # the union, with each entry tagged by its origin tier.

    @property
    def all_fix_records(self) -> "tuple[tuple[str, FixRecord], ...]":
        """All fix records — global + every stage's pool.

        Returns a tuple of ``(origin, record)`` pairs where ``origin``
        is either ``"global"`` or ``f"stage {stage.name!r}"``.  Order:
        global pool first (in registration order), then each stage in
        ``stage_records`` order, then each record within a stage in
        registration order.
        """
        out: list[tuple[str, FixRecord]] = [
            ("global", rec) for rec in self._fix_records
        ]
        for stage in self._stage_records:
            origin = f"stage {stage.name!r}"
            out.extend((origin, rec) for rec in stage.fix_records)
        return tuple(out)

    @property
    def all_mass_records(self) -> "tuple[tuple[str, MassRecord], ...]":
        """All mass records — global + every stage's pool.

        Same shape as :attr:`all_fix_records`.
        """
        out: list[tuple[str, MassRecord]] = [
            ("global", rec) for rec in self._mass_records
        ]
        for stage in self._stage_records:
            origin = f"stage {stage.name!r}"
            out.extend((origin, rec) for rec in stage.mass_records)
        return tuple(out)

    @property
    def all_region_records(
        self,
    ) -> "tuple[tuple[str, RegionAssignmentRecord], ...]":
        """All region records — global + every stage's pool.

        Phase SSI-2.D PR-C introspection symmetry (matches the
        :attr:`all_fix_records` / :attr:`all_mass_records` shape).
        Validator V3 (PR-A) guarantees no ``name=`` collision across
        scopes, so the user-facing name is unambiguous per
        ``(origin, record)`` pair.
        """
        out: list[tuple[str, RegionAssignmentRecord]] = [
            ("global", rec) for rec in self._region_records
        ]
        for stage in self._stage_records:
            origin = f"stage {stage.name!r}"
            out.extend((origin, rec) for rec in stage.region_records)
        return tuple(out)

    @property
    def all_recorder_specs(self) -> "tuple[tuple[str, Recorder], ...]":
        """All recorder specs — global + every stage's pool.

        Global recorders are sourced from ``self._primitives``
        filtered to :class:`Recorder` instances and EXCLUDING any
        spec claimed by ``s.recorder(...)``; the per-stage entries
        come from each :class:`StageRecord`'s ``recorder_specs``.
        Origin is ``"global"`` or ``f"stage {stage.name!r}"``.
        """
        out: list[tuple[str, Recorder]] = []
        for prim in self._primitives:
            if not isinstance(prim, Recorder):
                continue
            if id(prim) in self._stage_claimed_recorder_ids:
                continue
            out.append(("global", prim))
        for stage in self._stage_records:
            origin = f"stage {stage.name!r}"
            out.extend((origin, rec) for rec in stage.recorder_specs)
        return tuple(out)

    # -- Flat methods ----------------------------------------------------

    def model(self, *, ndm: int, ndf: int) -> None:
        """Set the model dimensionality (``ndm``) and the envelope ``ndf``.

        Per-node ``ndf`` is **inferred** from the declared element
        classes (ADR 0048) — ``ndf`` here is only the OpenSees model
        **envelope** (``model BasicBuilder -ndm K -ndf N``) and the
        **fallback** for nodes inference cannot see: element-less /
        decoupled nodes, and nodes touched only by adaptive elements
        (the zeroLength family). Element-attached nodes get their
        inferred value as a per-node ``-ndf`` override, emitted only
        where it differs from this envelope. There is no per-node
        ``ndf`` to declare on the geometry session — ``g.node_ndf``
        was removed; the elements you declare determine it.
        """
        self._ndm = ndm
        self._ndf = ndf

    def domain_capture(
        self,
        spec: "DomainCaptureSpec",
        *,
        path: "str | Path",
        ops: Any = None,
    ) -> "DomainCapture":
        """Open a :class:`DomainCapture` for in-process recording.

        Live entry point that resolves the supplied
        :class:`DomainCaptureSpec` against the bridge's ``fem``
        snapshot using the bridge's ``ndm`` / ``ndf``, then returns a
        :class:`DomainCapture` context manager writing to ``path``.

        Per Phase 9 D8 ``ndm`` / ``ndf`` are sourced implicitly from
        the bridge — the user must have called ``ops.model(ndm=,
        ndf=)`` first. Use :meth:`DomainCapture.from_h5` instead when
        no live bridge is available (sources ``ndm`` / ``ndf`` from a
        ``model.h5`` ``/meta`` block).

        Example::

            ops.model(ndm=3, ndf=6)
            from apeGmsh.opensees.emitter.live import get_ops
            osp = get_ops()   # the module the bridge drives; never import openseespy
            spec = DomainCaptureSpec(opensees=ops)
            spec.nodes(pg="Top", components=["displacement"])
            with ops.domain_capture(spec, path="run.h5", ops=osp) as cap:
                cap.begin_stage("gravity", kind="static")
                for _ in range(n):
                    osp.analyze(1, 1.0)   # the backend module, not the bridge
                    cap.step(t=osp.getTime())
                cap.end_stage()

        Raises
        ------
        RuntimeError
            If ``ops.model(ndm=, ndf=)`` has not been called yet.
        """
        if self._ndm is None or self._ndf is None:
            raise RuntimeError(
                "ops.domain_capture: ops.model(ndm=, ndf=) must be "
                "called before opening a DomainCapture (Phase 9 D8 "
                "binds ndm/ndf at resolve time)."
            )
        from ..results.capture._domain import DomainCapture
        resolved = spec._resolve_with_explicit_ndm_ndf(
            self._fem, ndm=self._ndm, ndf=self._ndf,
        )
        # Pass the live bridge through so DomainCapture materialises a
        # sidecar model.h5 and composes its ``/opensees/`` zone into the
        # run file (ADR 0020 Composed-file pattern).  Without this the
        # capture file carries only ``/model/`` + ``/stages/`` — and the
        # broker's neutral ``/model/meta`` has no bridge ``ndf`` (the
        # broker doesn't know the OpenSees envelope), so
        # ``OpenSeesModel.from_h5(path, fem_root="/model")`` would read
        # ``ndf=0``.  Forwarding the bridge lets
        # ``NativeWriter.write_opensees_from`` propagate the envelope
        # ndf onto ``/model/meta`` so mixed-ndf models round-trip through
        # ``Results.from_native``.
        #
        # The sidecar is written via ``self.h5(...)``.  Every build
        # forwards the bridge now: ADR 0055 Phase 5 lifted the
        # partitioned-staged ``h5()`` guard (P5.1), so the Composed run
        # file carries ``/opensees/stages`` + ``/opensees/partitions``
        # + the envelope ndf for partitioned staged captures too
        # (P5.3) — the feedstock the stage-aware viewer reads.  The
        # one remaining staged raise site (stage-claimed phantom
        # nodes, emitter gate-2) is handled by ``DomainCapture``'s
        # __enter__ degrade: it warns and proceeds sidecar-less.
        return DomainCapture(resolved, path, self._fem, ops=ops, bridge=self)

    def fix(
        self,
        *,
        pg: str | None = None,
        nodes: Iterable[int | Node] | None = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Apply homogeneous SP constraints (``fix``).

        Exactly one of ``pg`` / ``nodes`` must be supplied. ``nodes``
        accepts a mix of plain integer tags and :class:`Node`
        instances (from ``ops.nodes.get(...)``); both are normalized
        to tags. The build pipeline expands ``pg`` to a per-node
        fan-out at emit time.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                "apeSees.fix: supply exactly one of pg= or nodes= "
                f"(got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._fix_records.append(
            FixRecord(pg=pg, nodes=nodes_tuple, dofs=tuple(dofs)),
        )

    def equation_constraint(
        self,
        *,
        constrained: "tuple[int | Node, int]",
        retained: "Iterable[tuple[int | Node, int, float]]",
        coef: float = 1.0,
    ) -> None:
        """Declare one ``equationConstraint`` row (OpenSees ``EQ_Constraint``).

        The row is the linear relation, in OpenSees' sum-to-zero form::

            coef * u[cdof](cnode) + sum_i rcoef_i * u[rdof_i](rnode_i) = 0

        with ``constrained=(cnode, cdof)`` and ``retained=[(rnode, rdof,
        rcoef), ...]``. So ``u_c = 0.5 u_a + 0.5 u_b`` is
        ``constrained=(c, 1), retained=[(a, 1, -0.5), (b, 1, -0.5)]``.
        Nodes are FEM node ids (or :class:`Node` handles); DOFs are
        1-based.

        Emitted in the MP-constraint pass, after the snapshot's
        constraints. ``EQ_Constraint`` is enforced by ``Lagrange``,
        ``Penalty`` and the fork's ``LadrunoProjection`` only:
        ``Transformation``, ``Auto`` and ``Plain`` drop it without a word.
        So with no declared handler the bridge auto-emits ``Lagrange``
        (implicit) or ``LadrunoProjection`` (explicit integrator), and a
        declared ``Transformation`` / ``Auto`` raises — the same rules as
        an ``enforce="equation"`` tie (ADR 0068 INV-4).

        Validated here (non-zero finite coefficients, DOFs >= 1, a
        non-empty retained set, the constrained DOF not among the retained
        ones) and at emit (nodes exist, DOFs fit each node's ndf). The
        in-process run needs a build with ``equationConstraint``
        (openseespy >= 3.8.0 — one such model per process, since stock
        ``wipe()`` keeps the rows — or the fork); a partitioned emit
        refuses the rows, and
        ``ops.h5(...)`` does not archive them (``H5FeatureDeferredWarning``).
        """
        try:
            cnode, cdof = constrained
        except (TypeError, ValueError):
            raise ValueError(
                "apeSees.equation_constraint: constrained must be a "
                f"(node, dof) pair, got {constrained!r}."
            ) from None
        rows: list[tuple[int, int, float]] = []
        for triple in retained:
            try:
                rn, rd, rc = triple
            except (TypeError, ValueError):
                raise ValueError(
                    "apeSees.equation_constraint: each retained entry must "
                    f"be a (node, dof, coef) triple, got {triple!r}."
                ) from None
            rows.append((_iter_tags([rn])[0], rd, rc))
        self._equation_constraint_records.append(
            make_equation_constraint_record(
                _iter_tags([cnode])[0], cdof, coef, rows,
            ),
        )

    def mass(
        self,
        *,
        pg: str | None = None,
        nodes: Iterable[int | Node] | None = None,
        values: tuple[float, ...],
        overwrite: bool = False,
    ) -> None:
        """Attach lumped nodal mass.

        Exactly one of ``pg`` / ``nodes`` must be supplied. ``nodes``
        accepts plain integers or :class:`Node` instances.

        ``overwrite`` (Phase SSI-2.E) opts the record out of validator
        V2's cross-tier duplicate-mass check.  Rare at the global tier
        but kept for symmetry with the stage-bound :meth:`_StageBuilder.mass`
        — see that method for the typical use case.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                "apeSees.mass: supply exactly one of pg= or nodes= "
                f"(got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._mass_records.append(
            MassRecord(
                pg=pg, nodes=nodes_tuple, values=tuple(values),
                overwrite=bool(overwrite),
            ),
        )

    def mass_from_model(self) -> None:
        """Stream per-node lumped masses straight from the model snapshot.

        In 3-D, equivalent to looping ``ops.mass(nodes=[m.node_id], values=m.mass)``
        over every entry in ``fem.nodes.masses`` (e.g. the per-node tributary
        masses produced by ``g.masses.volume(...)``), but **without
        materializing one bridge ``MassRecord`` per node** — the snapshot
        masses are streamed at emit time. On a multi-million-node model this
        avoids a multi-GB resident list and millions of small objects (ADR
        0065 Tier 2). Honours per-node ``ndf``; each broker mass is spatially
        ordered ``(mx, my, mz, Ixx, Iyy, Izz)`` and is mapped onto the node's
        DOFs by ``broker_mass_components`` — byte-identical to the explicit
        loop in 3-D, and ``(mx, my[, Izz])`` on a 2-D (``ndm=2``) node.

        Model-wide declaration (no arguments). May be combined with explicit
        :meth:`mass` calls only on *disjoint* node sets — overlap raises at
        emit (nodal mass is additive under MP assembly). Deck/live emit only;
        the H5 archival emitter rejects it (masses already persist in
        ``model.h5`` via ``fem.nodes.masses``).
        """
        self._mass_from_model = True

    def fix_from_model(self) -> None:
        """Fix every homogeneous SP on the model snapshot (ADR 0051 §4).

        The support twin of :meth:`mass_from_model`: equivalent to a
        :meth:`fix` for each node carrying homogeneous records on
        ``fem.nodes.sp`` (``g.constraints.bc(...)``, or a zero-valued
        ``g.displacements``), with the node's restrained DOFs folded
        into one mask, so two targets sharing a node emit one ``fix``.
        The records' spatial components (``ux uy uz rx ry rz``) land on
        the node's deck DOFs by ``ndm`` and ndf, and one the node lacks
        is left out: ``bc``'s default ``[1, 1, 1]`` pins x and y on a
        2-D solid or frame and leaves a frame's ``rz`` free. Prescribed
        (non-zero) SPs are untouched: ``p.from_model(case)`` imports
        those.

        Model-wide declaration (no arguments), materialized into
        ordinary fix records at :meth:`build`, so every emit path treats
        them like :meth:`fix`. May be combined with explicit
        :meth:`fix` / ``s.fix`` / ``s.support`` only on *disjoint*
        (node, DOF) pairs: an overlap raises at build, since OpenSees
        refuses a second SP on a constrained DOF.
        """
        self._fix_from_model = True

    def ndf(self, target: object = None, *, ndf: int) -> None:
        """State the per-node ``ndf`` of an element-LESS decoupled node
        (ADR 0049 — the sole explicit per-node ndf channel).

        Every other node's ndf is **inferred** from its incident element
        classes (ADR 0048). ``ops.ndf`` exists only for nodes inference cannot
        reach — a spring/dashpot **ground**, a control node, or a mass anchor
        created via ``g.decouple_node(...)`` that no element touches.

        Parameters
        ----------
        target
            The decoupled-node handle returned by ``g.decouple_node(...)`` (a
            ``DecoupledNodeDef``) **or** its integer node tag. The handle is
            resolved to its tag at **build** time (so a handle materialized
            after meshing resolves correctly); a still-unmeshed handle fails
            loud at build.
        ndf
            The DOF count to assign the node.

        Raises (at build) :class:`BridgeError` if *target* is a mesh node, an
        element-touched node (its ndf is inferred — restating it would create
        a two-headed model), or an unresolved handle. The stated value is also
        checked by gates G1–G3 (adaptive endpoints, constraint masters,
        referenced fix/mass/load/sp DOFs).
        """
        if target is None:
            raise ValueError(
                "apeSees.ndf: a target is required — pass the decoupled-node "
                "handle from g.decouple_node(...) or its integer tag."
            )
        if not isinstance(ndf, int) or isinstance(ndf, bool) or ndf < 1:
            raise ValueError(
                f"apeSees.ndf: ndf must be a positive int (got {ndf!r})."
            )
        if isinstance(target, bool):
            raise ValueError(
                f"apeSees.ndf: target must be a decoupled-node handle or an "
                f"int tag (got bool {target!r})."
            )
        if isinstance(target, int):
            self._ndf_records.append(NdfRecord(handle=None, tag=int(target), ndf=ndf))
        else:
            # A handle (DecoupledNodeDef) — store it raw; resolve_ndf_overlay
            # dereferences ``.tag`` at build (fail-loud on a None tag).
            self._ndf_records.append(NdfRecord(handle=target, tag=None, ndf=ndf))

    def spring_bed(
        self,
        ground: "DecoupledNodeSetDef",
        *,
        at: "DecoupledNodeSetDef | None" = None,
        k: "SpringValues",
        c: "SpringValues | None" = None,
        orient: "OrientValues | None" = None,
        tributary: str | None = None,
        dirs: "Sequence[int]" = (1, 2, 3),
        do_rayleigh: bool = False,
        fix: bool = True,
        ndf: int = 3,
        name: str | None = None,
    ) -> "SpringBed":
        """A grounded zeroLength spring (and dashpot) on every node of a
        decoupled node set (ADR 0119 D2) — a distributed foundation bed.

        Spring ``i`` joins the ``ground`` node ``i`` (a
        ``g.decouple_node_set(...)`` handle) to the ``at`` node with the
        same source node — the side nodes of a second set declared with
        ``tie_dofs`` — or, with ``at=None``, to the source node itself.
        Per spring and direction the bed registers
        ``uniaxialMaterial Parallel <m> <E1> <V1> -factors k c`` over one
        ``Elastic 1.0`` and one ``Viscous 1.0 1.0`` shared by every bed of
        the bridge (``-factors k`` alone when ``c`` is ``None``), one
        ``zeroLength`` per spring, ``fix`` on the ground nodes (all
        ``ndf`` DOFs) and the ``ndf`` of the ground and ``at`` nodes.

        Parameters
        ----------
        ground, at
            Resolved ``g.decouple_node_set`` handles on one source.
        k, c
            Stiffness and dashpot coefficient per spring and direction:
            an ``(n, len(dirs))`` array, ``len(dirs)`` values for every
            spring, or a callable ``f(xyz, area)`` returning the array
            (``xyz``: the ``(n, 3)`` source coordinates in source order;
            ``area``: the tributary areas, ``None`` without
            ``tributary=``). Values must be finite and ``>= 0``.
        orient
            ``-orient``: a 6-tuple, an ``(n, 6)`` array or ``f(xyz)``;
            ``None`` keeps the global axes.
        tributary
            A 2-D label or physical group whose element areas, shared
            equally among each element's corners, give ``area``.
        dirs
            The ``-dir`` DOFs, distinct, each ``<= ndf``.
        do_rayleigh
            ``False`` (default) keeps the springs out of Rayleigh
            damping: the dashpots carry the foundation damping.
        fix
            Fix the ground nodes (default ``True``).
        ndf
            The DOF count stated for the ground and ``at`` nodes; it must
            equal the ``ndf`` of the node each spring reaches.
        name
            A label for the returned record.

        Returns the :class:`~apeGmsh.opensees._internal.spring_bed.SpringBed`
        record (node tags, ``k``, ``c``, areas, orientations, specs).
        Partitioned emit refuses node-pair elements (ADR 0049), so a bed
        emits single-process only.
        """
        from ._internal.spring_bed import build_spring_bed

        return build_spring_bed(
            self, ground, at=at, k=k, c=c, orient=orient,
            tributary=tributary, dirs=dirs, do_rayleigh=do_rayleigh,
            fix=fix, ndf=ndf, name=name,
        )

    def initial_stress(
        self,
        *,
        name: str,
        pg: str | None = None,
        elements: Iterable[int] | None = None,
        sigma_xx: float,
        sigma_yy: float,
        sigma_zz: float,
        ramp_steps: int,
        lambda_install: float = 1.0,
    ) -> "InitialStressRecord":
        """Initialize an in-situ stress tensor on ASDPlasticMaterial3D elements.

        Emits the OpenSees ``parameter`` / ``addToParameter`` /
        ``updateParameter`` ramp pattern that STKO uses to inject a
        pre-stressed state without applying gravity-driven body loads.
        The factor ramps linearly 0 → 1 over ``ramp_steps`` analyze
        calls and plateaus at 1.0 thereafter; the target stress baked
        into the ramp is ``sigma_* × lambda_install``, so passing
        ``lambda_install < 1.0`` produces a partial-installation
        (convergence-confinement) result.

        Exactly one of ``pg`` / ``elements`` must be supplied.  This
        primitive is **declarative only** — the actual stress
        advancement happens at analyze time, via the per-step
        dispatcher this primitive registers with.  Call
        ``ops.analyze(steps=ramp_steps, dt=...)`` or pass
        ``analyze_steps=ramp_steps`` to :meth:`tcl` / :meth:`py` for
        the ramp to take effect.

        Parameters
        ----------
        name
            Unique Tcl-identifier-safe label.  Used to name the
            emitted proc / state container.
        pg
            Physical group whose elements receive the ramped stress.
        elements
            Explicit list of FEM element ids.  XOR with ``pg``.
        sigma_xx, sigma_yy, sigma_zz
            Target Cauchy stress per component (compression negative).
        ramp_steps
            Number of analyze steps over which the factor reaches 1.0.
            Must be ``>= 1``.
        lambda_install
            Fraction of target to install (default 1.0).  Must be in
            ``(0, 1]``.
        """
        record = _build_initial_stress_record(
            source_label="apeSees.initial_stress",
            name=name, pg=pg, elements=elements,
            sigma_xx=sigma_xx, sigma_yy=sigma_yy, sigma_zz=sigma_zz,
            ramp_steps=ramp_steps, lambda_install=lambda_install,
        )
        self._initial_stress_records.append(record)
        # Phase SSI-2.A: return the record so callers can pass it to
        # ``with ops.stage(...) as s: s.add(record)`` which moves it
        # from this bridge-global pool into the stage's pool.
        # Non-staged callers can ignore the return value — the record
        # is already registered and will emit in the flat path.
        return record

    def convergence_confinement(
        self,
        *,
        name: str,
        pg: str | None = None,
        elements: Iterable[int] | None = None,
        sigma_xx: float = 0.0,
        sigma_yy: float = 0.0,
        sigma_zz: float = 0.0,
        lambda_target: float,
        n_steps: int,
    ) -> "InitialStressRecord":
        """Convergence-confinement helper (Phase SSI-3).

        Thin wrapper over :meth:`initial_stress` for the tunnelling
        convergence-confinement pattern: ramp a target stress on a
        boundary region to ``lambda_target`` × ``sigma`` over
        ``n_steps`` analyze steps.  Matches the
        ``_stressCtrl_11``-style proc from
        ``SSI/Interaccion/analysis_steps.tcl:19753-19767``.

        Differs from :meth:`initial_stress` in two cosmetic ways:

        * ``lambda_target`` (renamed from ``lambda_install``) — more
          natural reading at the call site for confinement / relaxation
          contexts.
        * ``n_steps`` (renamed from ``ramp_steps``) — matches the
          spec's naming.

        At least one of ``sigma_xx`` / ``sigma_yy`` / ``sigma_zz`` must
        be non-zero (typically only one — single-component relaxation
        is the canonical SSI use case).

        Returns the underlying :class:`InitialStressRecord`; pass it to
        ``s.add(...)`` inside a stage block to bind to that stage.

        Parameters
        ----------
        name
            Unique Tcl-identifier-safe label.
        pg, elements
            Same XOR semantics as :meth:`initial_stress`.
        sigma_xx, sigma_yy, sigma_zz
            Target Cauchy stress per component (compression negative).
            At least one must be non-zero.
        lambda_target
            Fraction of target stress to install — i.e. the relaxation
            (or confinement) coefficient.  Must be in ``(0, 1]``.
        n_steps
            Number of analyze steps over which the factor reaches 1.0
            internally.  After the cap, the cumulative is
            ``sigma × lambda_target``.
        """
        if sigma_xx == 0.0 and sigma_yy == 0.0 and sigma_zz == 0.0:
            raise ValueError(
                "apeSees.convergence_confinement: at least one of "
                "sigma_xx / sigma_yy / sigma_zz must be non-zero."
            )
        return self.initial_stress(
            name=name,
            pg=pg,
            elements=elements,
            sigma_xx=sigma_xx,
            sigma_yy=sigma_yy,
            sigma_zz=sigma_zz,
            ramp_steps=n_steps,
            lambda_install=lambda_target,
        )

    def imposed_displacement(
        self,
        *,
        pg: str | None = None,
        nodes: Iterable[int] | None = None,
        ux: float | None = None,
        uy: float | None = None,
        uz: float | None = None,
        pattern_factor: float = 1.0,
        series: "TimeSeries | None" = None,
        name: str | None = None,
    ) -> "Plain":
        """Imposed-displacement pattern helper (Phase SSI-3).

        Emits one ``pattern Plain`` containing ``sp NODE DOF VALUE``
        prescribed-displacement entries for every (node, dof) pair
        where the corresponding ``ux`` / ``uy`` / ``uz`` is non-None.
        Used for fault-slip kinematics, support-settlement scenarios,
        and any other prescribed-displacement driver.

        STKO equivalent:
        ``pattern Plain N tsTag -fact F { sp NODE DOF VAL ... }``
        from ``SSI/Interaccion y Falla/analysis_steps.tcl:22832-23253``.
        Where STKO uses ``-fact F`` on the pattern, this helper folds
        the same scaling into the auto-created ``Linear(factor=F)``
        time series — numerically identical, simpler API.

        Parameters
        ----------
        pg, nodes
            XOR: exactly one of ``pg`` (physical-group name) or
            ``nodes`` (iterable of FEM node ids) must be supplied.
        ux, uy, uz
            Scalar broadcast: every targeted node gets the same
            prescribed displacement in this DOF.  ``None`` (default)
            skips the DOF.  At least one of the three must be set.
        pattern_factor
            Multiplier folded into the auto-created ``Linear`` time
            series.  Default ``1.0`` (no scaling).  Matches STKO's
            ``-fact F`` semantics: the actual applied displacement
            at simulation-time ``t`` is
            ``value × pattern_factor × t``.
        series
            Optional explicit :class:`TimeSeries` to use.  Must be
            already registered with the bridge.  When supplied,
            ``pattern_factor`` is ignored — the user is in full
            control of the time-history shape.
        name
            Optional bridge-side alias for the pattern (as ``name=`` on
            ``ops.pattern.Plain``).  It is also the ``<name>`` of the
            call's provenance records,
            ``opensees/pattern/imposed_displacement:<name>`` and, for the
            auto-created series,
            ``opensees/timeSeries/imposed_displacement:<name>``; without
            it ``<name>`` is ``#<k>``, the call's 1-based ordinal on this
            bridge, so a name may not start with ``#``.  A refused or
            taken name raises before anything is registered.  Both
            records carry ``origin = "synthesised"`` and point at this
            call (ADR 0112 D3, #1378).

        Returns
        -------
        Plain
            The registered :class:`Plain` pattern.  This is a **global**
            (non-staged) pattern: it is valid only in a non-staged deck
            (global pattern + ``ops.analyze``).  Per ADR 0051 §5 a model
            may not mix a global pattern with stages — combining this
            with ``ops.stage(...)`` raises :class:`BridgeError` at build.
            For prescribed motion inside a staged deck, author the ``sp``
            on a stage pattern instead (``with s.pattern(series=...) as
            p: p.sp(...)``).

        Notes
        -----
        Per-node-varying displacements are NOT supported in v1 —
        every targeted node gets the same scalar.  For different
        values per node, call ``imposed_displacement`` multiple times
        with disjoint ``nodes=`` lists, or construct the ``Plain``
        pattern manually via ``ops.pattern.Plain(...)``.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                "apeSees.imposed_displacement: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        if ux is None and uy is None and uz is None:
            raise ValueError(
                "apeSees.imposed_displacement: at least one of ux / "
                "uy / uz must be supplied."
            )
        if pattern_factor == 0.0:
            raise ValueError(
                "apeSees.imposed_displacement: pattern_factor must be "
                "non-zero (a zero factor produces an inert pattern)."
            )

        # DOF-index validation against the model's ndf (red-team H3).
        # ``uz`` maps to DOF 3, which only exists on ndf>=3 models;
        # emitting ``sp NODE 3 VALUE`` on an ndf=2 model produces an
        # OpenSees parse error ("invalid dof").  Catch upfront with a
        # clear error pointing at the offending kwarg.
        if self._ndf is not None:
            dof_kwargs = (("ux", 1, ux), ("uy", 2, uy), ("uz", 3, uz))
            for kw, dof_idx, val in dof_kwargs:
                if val is not None and dof_idx > self._ndf:
                    raise ValueError(
                        f"apeSees.imposed_displacement: {kw}= targets "
                        f"DOF {dof_idx}, but the model's ndf is "
                        f"{self._ndf}.  Drop {kw}= or call "
                        f"ops.model(..., ndf={dof_idx}) first."
                    )

        from .pattern.pattern import Plain as _Plain
        from .time_series.time_series import Linear as _Linear

        # Every refusal happens here, before anything is registered, so
        # a bad call leaves no series, pattern, alias or record behind.
        # An empty name is no name (as ``_register`` and ``capture``
        # read it): the call takes its ordinal key.
        name = name or None
        if name is not None:
            if name.startswith("#"):
                raise ValueError(
                    f"apeSees.imposed_displacement: name={name!r} may not "
                    "start with '#': that prefix marks the ordinal key of "
                    "an unnamed call (imposed_displacement:#<k>)."
                )
            existing = self._names.get(name)
            if existing is not None:
                raise ValueError(
                    f"apeSees: name {name!r} is already registered to a "
                    f"{type(existing).__name__}; names must be unique per "
                    "bridge.  Pick a different name= (or pass the object "
                    "handle directly)."
                )
        if series is not None:
            # A name string resolves through the alias table like every
            # other reference kwarg (an unknown name or a non-series
            # fails loud).
            series = self._resolve(series, base=TimeSeries)

        # Provenance key of this call's synthesised objects
        # (``<verb>:<name>`` or ``<verb>:#<k>``; see ``name`` above).
        # Both prospective keys are checked before either object is
        # registered: user names share the key space, and a refusal
        # must leave no series, pattern or record behind.
        _key = (
            f"imposed_displacement:"
            f"{name if name is not None else f'#{self._imposed_displacement_calls + 1}'}"
        )
        for _family, _needed in (("timeSeries", series is None),
                                 ("pattern", True)):
            if _needed and self._provenance.has("opensees", _family, _key):
                raise ValueError(
                    "apeSees.imposed_displacement: the provenance key "
                    f"'opensees/{_family}/{_key}' of the {_family} it would "
                    "create already has a record (a declaration was named "
                    "like it); pass another name=.  Nothing was registered."
                )
        self._imposed_displacement_calls += 1

        # Default time series: Linear scaled by pattern_factor.
        # Folds STKO's ``-fact F`` semantics into the time-series
        # factor instead of an explicit ``-fact`` on the pattern
        # (apeGmsh's Plain pattern primitive doesn't carry one).
        if series is None:
            series = self._register(
                _Linear(factor=float(pattern_factor)), synthesised=_key,
            )

        # Register + tag the Plain pattern directly (not through the
        # namespace) so its provenance record carries the verb's key.
        plain = self._register(_Plain(series=series), name=name,
                               synthesised=_key)
        # Populate the sp records.  Plain's recording API accepts
        # either pg= or node=; we route based on the helper's input.
        dof_values: tuple[tuple[int, float | None], ...] = (
            (1, ux), (2, uy), (3, uz),
        )
        with plain:
            if pg is not None:
                for dof, value in dof_values:
                    if value is None:
                        continue
                    plain.sp(pg=pg, dof=dof, value=float(value))
            else:
                assert nodes is not None
                for node in nodes:
                    for dof, value in dof_values:
                        if value is None:
                            continue
                        plain.sp(node=int(node), dof=dof, value=float(value))
        return plain

    def stage(self, name: str) -> "_StageBuilder":
        """Open a staged-analysis block (Phase SSI-2.A).

        Nested ``with ops.stage(...)`` blocks are NOT supported —
        opening a second stage builder while another is still open
        raises ``RuntimeError``.  The lexical-vs-emit-order semantics
        would otherwise be confusing (the inner builder's __exit__
        fires first, registering the inner stage BEFORE the outer in
        ``_stage_records``, which is the opposite of what readers
        expect).

        Usage::

            with ops.stage(name="insitu") as s:
                s.add(ops.initial_stress(name="rock", ..., ramp_steps=10))
                s.analysis(
                    test=ops.test.NormDispIncr(tol=1e-4, max_iter=150),
                    algorithm=ops.algorithm.Newton(),
                    integrator=ops.integrator.LoadControl(dlam=0.1),
                    constraints=ops.constraints.Plain(),
                    numberer=ops.numberer.RCM(),
                    system=ops.system.UmfPack(),
                    analysis=ops.analysis.Static(),
                )
                s.run(n_increments=10, dt=0.1)

        Each stage emits its own analysis-chain primitives, its own
        analyze loop (hook-wrapped if any ``s.add(initial_stress(...))``
        registered a ramp), and a between-stages cleanup block
        (``loadConst -time 0.0`` + ``wipeAnalysis`` + hook-list clear).

        Multiple ``with ops.stage(...)`` blocks accumulate in
        registration order; they emit in that order at deck-emit time.

        Validation happens on ``with`` exit: every stage must have a
        complete analysis chain (all six chain kwargs + the analysis
        directive) and an ``s.run(...)`` call.

        Returns
        -------
        _StageBuilder
            Context manager that collects per-stage records and emits
            a :class:`StageRecord` to the bridge on close.
        """
        if not name:
            raise ValueError("apeSees.stage: name= must be non-empty.")
        if self._open_stage_builder is not None:
            raise RuntimeError(
                "apeSees.stage: a stage is already open "
                f"(name={self._open_stage_builder._name!r}).  Close it "
                "before opening another — nested ``with ops.stage(...)``"
                " blocks would register stages in lexically-reversed "
                "order at emit time."
            )
        builder = _StageBuilder(self, str(name))
        self._open_stage_builder = builder
        return builder

    def region(
        self,
        *,
        name: str,
        pg: str | None = None,
        nodes: Iterable[int | Node] | None = None,
    ) -> None:
        """Assign nodes to a named OpenSees Region.

        Each ``name`` collects all nodes registered against it
        (across multiple calls, across explicit ``nodes=`` and
        ``pg=`` resolutions) and emits a single
        ``region $tag -node n1 n2 ...`` line at build time with a
        region tag from the build's tag plan.  Useful for damping
        assignments and any future recorder that filters by region.

        Exactly one of ``pg`` / ``nodes`` must be supplied; ``nodes``
        accepts a mix of plain integer tags and :class:`Node`
        instances (matching :meth:`fix` / :meth:`mass`).

        End users typically call this through :meth:`Node.region` or
        :meth:`NodeSet.region` rather than directly.
        """
        if not name:
            raise ValueError("apeSees.region: name= must be non-empty.")
        if (pg is None) == (nodes is None):
            raise ValueError(
                "apeSees.region: supply exactly one of pg= or nodes= "
                f"(got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._region_records.append(
            RegionAssignmentRecord(
                name=str(name), pg=pg, nodes=nodes_tuple,
            ),
        )

    def _split_profiler_records(
        self,
    ) -> tuple[
        list[tuple[str, tuple[int | float | str, ...]]],
        list[tuple[str, tuple[int | float | str, ...]]],
    ]:
        """Split recorded profiler verbs into (before-analyze, after-analyze).

        ``start`` / ``reset`` bracket *open* (emitted before the ``analyze``
        line); ``stop`` / ``report`` / ``memory`` bracket *close* (after).
        Recorded order is preserved within each side. Consumed by the deck
        emitters (:meth:`tcl` / :meth:`py`).
        """
        pre: list[tuple[str, tuple[int | float | str, ...]]] = []
        post: list[tuple[str, tuple[int | float | str, ...]]] = []
        for verb, vargs in self._profiler_records:
            if verb in ("start", "reset"):
                pre.append((verb, vargs))
            else:
                post.append((verb, vargs))
        return pre, post

    def analyze(
        self,
        *,
        steps: int,
        dt: float | None = None,
        strategy: "Ladder | None" = None,
        profile: str | None = None,
        profile_run: str | None = None,
        profile_deep: bool = False,
        profile_memory: bool = False,
        profile_per_step: bool = False,
    ) -> int:
        """Build + emit + run the analysis chain via the live emitter.

        Builds a :class:`BuiltModel`, drives a
        :class:`~apeGmsh.opensees.emitter.live.LiveOpsEmitter` end-to-
        end, then issues the ``analyze`` call. Returns the openseespy
        ``analyze`` return value (0 on success).

        ``strategy`` (ADR 0057 Phase A) attaches a solution-strategy
        ladder to the analyze loop — on a failed increment the live
        runner escalates through the ladder's algorithm rungs (the
        declared chain algorithm is rung 0), restoring rung 0 after a
        rescue and logging escalations to the live emitter's
        ``strategy_events``.  Exhaustion returns the failing rc.

        When ``profile`` is given, the live run is bracketed by the Ladruno
        fork's stack profiler: ``profiler start [flags]`` before the analyze
        loop and ``profiler report <profile> [-run profile_run]`` after,
        with ``profile_deep`` / ``profile_memory`` / ``profile_per_step``
        toggling the ``start`` flags. Requires the fork build — the live
        emitter raises a clear error on stock openseespy. (Deck-mode
        profiling uses the explicit ``ops.profiler.*`` verbs instead, and
        does NOT consume the ``profile=`` kwargs here.)

        Raises :class:`BridgeError` if the analysis chain is incomplete
        (one or more of constraints / numberer / system / test /
        algorithm / integrator / analysis is missing).

        Phase SSI-2.A: staged models (``ops.stage(...)`` blocks
        declared) are NOT supported by live execution.  Emit a Tcl
        or Py deck via :meth:`tcl` / :meth:`py` and run it via the
        OpenSees binary / openseespy subprocess instead.
        """
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.analyze: live execution does not support "
                "staged models in Phase SSI-2.A "
                f"(got {len(self._stage_records)} stage(s)).  Use "
                "ops.tcl(path, run=True) or ops.py(path, run=True) to "
                "emit a staged deck and run it via the OpenSees binary "
                "/ openseespy subprocess instead."
            )
        self._check_analysis_chain_for_analyze()
        self._check_explicit_solver_compat()

        # Local import — keeps openseespy out of import-time for users
        # who only emit Tcl / py.
        from .emitter.live import LiveOpsEmitter

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        # Retain for post-run live queries (e.g. ladruno_projection_tie_force);
        # the in-process openseespy session stays alive after analyze returns.
        self._live_emitter = live_emitter
        bm.emit(live_emitter)
        self._artifacts.after_emit(self)
        if profile is not None:
            start_flags: list[str] = []
            if profile_deep:
                start_flags.append("-deep")
            if profile_memory:
                start_flags.append("-memory")
            if profile_per_step:
                start_flags.append("-perStep")
            live_emitter.profiler("start", *start_flags)
        spec: StrategySpec | None = None
        if strategy is not None:
            # Rung 0 = the flat chain's declared algorithm (the last
            # one registered wins, matching emission order).
            base = next(
                (p for p in reversed(self._primitives)
                 if isinstance(p, SolutionAlgorithm)),
                None,
            )
            spec = strategy.to_spec(base=base)
        result: int = int(
            live_emitter.analyze(steps=steps, dt=dt, strategy=spec)
        )
        if profile is not None:
            report_args: list[str] = [profile]
            if profile_run is not None:
                report_args += ["-run", profile_run]
            live_emitter.profiler("report", *report_args)
        return result

    def ladruno_projection_tie_force(self, node: int, dof: int) -> float:
        """Tie force ``f = M(a_raw - a_proj)`` at ``(node, dof)`` from the last
        projection step (≈ LS-DYNA ``*DATABASE_NCFORC``).

        Recovers the interface force a non-matching equation-tied interface
        (``g.constraints.tie(..., enforce="equation")``) carries, via the fork
        ``ladrunoProjectionTieForce`` query (ADR-30 P3 / ADR 0068 P5). ``dof``
        is 1-based (OpenSees convention).

        Requires a prior **live** :meth:`analyze` with a ``LadrunoProjection``
        constraint handler active. Fork-only: a stock build raises
        ``RuntimeError`` (see :data:`~apeGmsh.opensees.emitter.live.
        _TIE_FORCE_FORK_REQUIRED`).

        For a recorded **time history** of the tie force instead of a single
        post-run value, use the recorder route:
        ``ops.recorder.Ladruno(nodal_responses=("constraintTieForce",))`` and
        read it back with
        ``results.nodes.get(component="constraint_tie_force_x")`` (explicit
        analyses only — the recorder channel is scattered by the explicit
        ``CentralDifferenceLadruno`` integrator).
        """
        if self._live_emitter is None:
            raise BridgeError(
                "apeSees.ladruno_projection_tie_force: no live analysis has "
                "run. Call analyze(...) first (the live path); the query reads "
                "the last projection step. For a recorded time history, record "
                "ops.recorder.Ladruno(nodal_responses=('constraintTieForce',)) "
                "and read results.nodes.get(component='constraint_tie_force_x')."
            )
        return self._live_emitter.ladruno_projection_tie_force(node, dof)

    def augment(
        self, *, element: int, tol: float = 1.0e-8, max_passes: int = 10,
    ) -> "AbstractContextManager[list[float]]":
        """Run an ADR-41 held-load augmentation sweep on a live run and
        yield its per-pass constraint-violation history.

        The supported way to close an ``enforce="al"`` coupling's constraint
        gap **within** one step (fork PR #839 §3.2): ``enforce="al"`` alone
        advances its Uzawa recursion once per *committed* step, so a single
        push leaves the penalty gap ``c/K_t`` standing. Forwards to
        :meth:`~apeGmsh.opensees.emitter.live.LiveOpsEmitter.augment`, which
        owns the whole recipe (``ladrunoBeginAugment`` →
        ``integrator LoadControl 0.0`` → held passes polling
        ``constraintViolation`` → ``ladrunoEndAugment`` + the caller's
        integrator restored). Read that docstring for the semantics; the
        passes all run on **entry**, so the ``with`` body is a placeholder
        and displacements must be read **after** the block::

            ops.analyze(steps=1)                 # the real step
            with ops.augment(element=tag, tol=1e-8) as gaps:
                pass
            assert gaps[-1] < 1e-8

        Parameters
        ----------
        element : int
            Emitted OpenSees tag of the ``LadrunoKinematicCoupling``.
            apeGmsh has **no name → emitted-tag lookup** (the constraint
            name rides the comment channel only), so the tag has to come
            from the live domain: ``max(openseespy.opensees.getEleTags())``
            when the coupling is the last element emitted, or a scan of a
            written ``model.h5`` for the ``"LadrunoKinematicCoupling"``
            ``type_token`` (``OpenSeesModel.elements()``).
        tol : float, default 1e-8
            Stop once the violation falls below this.
        max_passes : int, default 10
            Hard cap on held-load passes; exceeding it raises.

        Raises
        ------
        BridgeError
            When no live analysis has run; when no integrator was recorded
            through the live emitter; when ``tol`` is not met within
            ``max_passes``. Also on a stock (non-fork) build, or a nested
            sweep (``RuntimeError``).
        """
        if self._live_emitter is None:
            raise BridgeError(
                "apeSees.augment: no live analysis has run. Call "
                "analyze(...) first (the live path) — the sweep drives the "
                "in-process domain that step left standing. There is no "
                "deck-emission equivalent: ladrunoBeginAugment / "
                "ladrunoEndAugment are issued interactively around held "
                "passes, not written into a Tcl / py deck."
            )
        return self._live_emitter.augment(
            element=element, tol=tol, max_passes=max_passes,
        )

    def tcl(
        self,
        path: str,
        *,
        run: bool = False,
        bin: str | None = None,
        analyze_steps: int | None = None,
        analyze_dt: float | None = None,
        per_rank: bool = False,
        flat: bool = False,
        stream: bool = False,
        verbose: bool = False,
        log: str | None = None,
        progress: bool = True,
    ) -> "RunSolverStats | None":
        """Emit a Tcl deck to ``path``; optionally subprocess OpenSees.

        When ``run=True`` the OpenSees subprocess output is **always**
        tee'd to a log file — ``log`` (a path) overrides, otherwise it
        is ``<path>.log`` next to the deck. Console output is opt-in via
        ``verbose``: ``False`` (default) prints begin / op / end only;
        ``True`` adds a live step counter (parsed from the
        ``APEGMSH_PROGRESS`` markers ``progress=True`` injects into the
        analyze loop) plus streamed warning lines. A non-zero exit
        raises ``RuntimeError`` carrying the log tail + path, never the
        whole buffer. ``verbose`` / ``log`` / ``progress`` are inert
        when ``run=False``.

        Returns a :class:`~apeGmsh.opensees._solver_stats.RunSolverStats`
        (ADR 0106 D1) when the deck declared ``Pardiso(stats=True)``
        anywhere and the run finished — the per-stage capacity record
        parsed out of the solver's own ``PARDISO stats:`` blocks.
        ``None`` otherwise, which includes every deck that did not ask
        for statistics and every ``run=False`` call.

        When ``analyze_steps`` is supplied, an ``analyze`` line is
        appended to the deck after every other primitive — wrapped in
        a hook-dispatching for-loop if any
        :meth:`initial_stress` calls registered step hooks (Phase
        SSI-1).  Without ``analyze_steps``, the emitted deck declares
        the model but does not drive an analysis.

        ``per_rank=True`` (ADR 0061) writes a driver deck at ``path``
        plus one ``ranks/rank<K>_<seq>.tcl`` fragment per
        ``if {[getPID] == K} { ... }`` block; the driver guards each
        fragment behind a one-line ``source`` so every MPI rank parses
        only the driver plus its own fragments — O(global + model/np)
        instead of O(model) per rank.  Layout-only: the deck semantics
        (including the single-process rank-0 fallback) are unchanged.
        Requires a partitioned model (``len(fem.partitions) > 1``).

        ``flat=True`` forces the single-domain (serial) emit even when
        the model carries partitions — e.g. a composed model, which is
        auto-partitioned one-rank-per-module (ADR 0038 §"Rank model")
        and would otherwise take the per-rank fan-out.  The deck
        declares the whole model in one domain with no ``getPID``
        brackets, exactly as the live in-process runner and modal decks
        emit it.  This is the Tcl route for serial-only records
        (``g.embed`` ties; fork contact before ADR 0092 S4 landed
        partitioned emit, and still the escape hatch for the contact
        cases the partitioned path refuses) on a composed model.
        Mutually exclusive with ``per_rank``; a no-op on
        an already-unpartitioned model.

        ``stream=True`` (ADR 0065 Tier 2 / plan_emit_memory_columnar.md
        A1–A3) writes the deck through a live file sink instead of
        accumulating the line buffer, so peak emit memory stops scaling
        with deck size. Output is byte-identical to the default list
        mode, including under ``per_rank=True``, where the fragment
        files are live-routed (``partition_open`` switches the sink)
        rather than sliced post-hoc. Everything goes to ``.tmp``
        siblings promoted atomically on clean completion — a mid-emit
        exception never leaves a half-written deck.
        """
        from .emitter.tcl import TclEmitter, deck_backend

        if flat and per_rank:
            raise ValueError(
                "apeSees.tcl: flat=True and per_rank=True are mutually "
                "exclusive — flat forces the single-domain (serial) "
                "emit; per_rank splits the partitioned fan-out "
                "(ADR 0061). Drop one of the two flags."
            )
        bm = self.build()
        emitter = TclEmitter(backend=deck_backend(self._opensees))
        emitter._emit_progress = bool(progress)
        # ADR 0099 S5: ``per_rank`` is applied around / after ``bm.emit``
        # (live fragment routing under ``stream``, post-hoc span slicing
        # otherwise), so the INV-4 gate inside emit cannot see it. Stamp
        # it on the emitter — the same seam ``supports_partitions`` uses.
        emitter.per_rank_fragments = bool(per_rank)  # type: ignore[attr-defined]
        if flat:
            # Force the single-domain emit for a partition-carrying fem
            # (e.g. a composed model auto-partitioned one-rank-per-module,
            # ADR 0038) — the same seam the live in-process runner and
            # modal decks use. Serial-only records (g.embed ties, plus
            # the contact cases ADR 0092 S4's partitioned fan-out
            # refuses — SOFT, staged, undecidable owner) emit on this
            # path.
            emitter.supports_partitions = False  # type: ignore[attr-defined]
        pre_prof, post_prof = self._split_profiler_records()
        if stream:
            # ADR 0065 Tier 2: write-through sink; per-rank
            # fragment files are live-routed by
            # partition_open/partition_close.
            emitter.stream_to(path, per_rank=per_rank)
        try:
            bm.emit(emitter)
            for _verb, _vargs in pre_prof:
                emitter.profiler(_verb, *_vargs)
            if analyze_steps is not None:
                emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
            for _verb, _vargs in post_prof:
                emitter.profiler(_verb, *_vargs)
            if stream and per_rank and (
                emitter.stream_fragment_count() == 0
            ):
                raise ValueError(
                    "apeSees.tcl: per_rank=True requires a "
                    "partitioned model (len(fem.partitions) > 1) — "
                    "the emitted deck has no per-rank blocks to "
                    "split out. Partition the mesh "
                    "(g.mesh.partitioning) or drop per_rank."
                )
            if stream:
                # Promotion runs INSIDE the guarded region (review
                # hardening): a failing os.replace mid-promotion
                # (Windows file lock / antivirus) routes to
                # stream_abort below, which removes every remaining
                # .tmp. Fragments already promoted by the partial
                # loop stay in place — the driver is promoted LAST,
                # so no deck entry point exists until everything it
                # sources does; a clean re-run overwrites the
                # leftovers via os.replace.
                emitter.stream_finish()
        except BaseException:
            if stream:
                # Leave no half-written deck: remove every .tmp
                # (final paths are only ever created by a COMPLETE
                # promotion pass, except fragments promoted before
                # a mid-promotion failure — see stream_finish;
                # ADR 0065 Tier 2 Decision §4).
                emitter.stream_abort()
            raise
        if not stream and per_rank:
            spans = emitter.partition_spans()
            if not spans:
                raise ValueError(
                    "apeSees.tcl: per_rank=True requires a "
                    "partitioned model (len(fem.partitions) > 1) — "
                    "the emitted deck has no per-rank blocks to "
                    "split out. Partition the mesh "
                    "(g.mesh.partitioning) or drop per_rank."
                )
            # line_buffer(): read-only, no deck-sized copy (ADR 0065 A0).
            _write_per_rank_tcl(path, emitter.line_buffer(), spans)
        elif not stream:
            with open(path, "w", encoding="utf-8") as f:
                emitter.write_to(f)
        self._artifacts.after_emit(self)

        if not run:
            return None

        binary = _resolve_opensees_binary(bin, self._opensees)
        return stream_run(
            [binary, path],
            log_path=resolve_log_path(log, path),
            verbose=verbose,
            label=run_label(path, analyze_steps, analyze_dt),
            header="OpenSees",
            deck_path=path,
            expect_solver_stats=_deck_requested_solver_stats(emitter),
        )

    def modal_deck(
        self,
        path: str,
        *,
        solver: str = "feast",
        band: "tuple[float, float] | None" = None,
        num_modes: int | None = None,
        certify: bool = False,
        target: str = "tcl",
        out: str = "eigenvalues.out",
    ) -> None:
        """Emit a distributed modal deck (ADR 0077 Tier 1) — two backends.

        ``solver="feast"`` (default) emits the **replicated** FEAST deck
        described below; ``solver="arpack"`` emits the **partitioned**
        ARPACK deck (Tier 1B) — see
        :meth:`_modal_deck_arpack` for that half. They invert each
        other on the two facts that matter (flat vs partitioned emit;
        ``system`` inert vs load-bearing), so the docs are kept apart
        rather than merged.

        **Which to use.** ``"arpack"`` when the model does not fit on one
        node: it is the only backend where both the storage *and* the
        factorization are distributed. ``"feast"`` when you want a
        frequency window rather than the lowest N, or ``-certify``
        completeness. **Neither** for a model that fits on one node —
        Tier 0 (:meth:`eigen` / :meth:`modal_properties` on the
        unpartitioned build) is faster at every size measured and is the
        only route to correct participation factors and effective modal
        mass.

        FEAST backend (``solver="feast"``, needs ``band=``)
        --------------------------------------------------
        Writes a **flat** Tcl deck — every MPI rank builds the FULL model —
        that runs band-targeted FEAST under ``OpenSeesMP``: ``eigen -feast
        band[0] band[1] -rci`` routes each contour solve through the
        distributed ``dmumps`` kernel (fork ADR 43, **L3-only**: every rank
        holds the full ``(K, M)`` CSR and the kernel slices the 2n block
        system's triplets across ranks). Distribution lives inside the RCI
        kernel, **not** in domain decomposition — a partitioned
        ``if {[getPID]==K}`` deck fails ``FeastEigenSOE::setSize`` (P2
        live finding), so partitions on the model are ignored here (the
        deck is emitted flat) and the deck's ``system`` line plays no part
        in the FEAST solve. RAM trade-off: the full model is assembled on
        every rank (the documented L3 regime, ~1e5–1e6 DOF).

        The band (Hz) defines the mode count; there is no ``num_modes``.
        The deck is the HPC entry point (``ops.run_remote`` /
        ``Cluster.submit``, ADR 0060) and also runs single-process under
        plain ``OpenSees`` (serial FEAST — the ``getPID`` shim makes the
        rank-0 write-out unconditional).

        ``modalProperties`` is **not** emitted: it is MPI-blind upstream
        (wrong effective mass under any multi-rank run; ADR 0077 INV-2).
        For participation factors run the single-process
        :meth:`modal_properties` (Tier 0). Harvest with
        :meth:`ParallelModalResult.from_job` — eigenvalues from the
        rank-0 write-out plus mode shapes (ADR 0077 P3): the deck
        records one ``mode_shape_<k>.out`` per found mode from rank 0
        (the replicated model puts ALL nodes on every rank) with a
        ``mode_shapes.json`` sidecar pinning the node→column map
        (sorted mesh node tags × ``ndf`` DOFs).

        Parameters
        ----------
        path
            Deck output path.
        solver
            ``"feast"`` (default, replicated band solve) or ``"arpack"``
            (partitioned lowest-N solve, Tier 1B).
        band
            ``(f_min, f_max)`` frequency band in Hz; needs
            ``0 <= f_min < f_max``. **FEAST only** — rejected for
            ``solver="arpack"``, whose selection axis is a mode count.
        num_modes
            Number of modes to extract. **ARPACK only** — rejected for
            ``solver="feast"``, where the contour *is* the band and the
            found count is dynamic.
        certify
            Emit ``-certify`` (fork Sturm/inertia completeness check).
            **FEAST only.**
        target
            ``"tcl"`` (the classic-Tcl deck). Each solver needs its own
            fork build: FEAST the classic-Tcl ``-feast`` parity build
            (fork PR #578), ARPACK the MP eigen wiring (fork PR #668,
            ``5a522b03b``). ``"pymp"`` (an
            OpenSeesMP-Python deck, ADR 0077 unlock 2a) raises for both
            solvers — for ARPACK because the modern interpreter still
            builds its ``ArpackSOE`` bare (the same latent F1 defect;
            #668 is classic-Tcl only).
        out
            Rank-0 eigenvalue write-out filename (read by
            :meth:`ParallelModalResult.from_job`).

        Raises
        ------
        ValueError
            If ``solver`` is unknown, if the band / mode-count arguments
            do not match the chosen solver, if ``band`` is invalid, if
            ``solver="arpack"`` is given an unpartitioned model, or if a
            non-``Mumps`` system is declared on an ARPACK deck.
        NotImplementedError
            If ``target != "tcl"`` or the model has registered stages.
        """
        from .emitter.tcl import TclEmitter, deck_backend

        if solver not in ("feast", "arpack"):
            raise ValueError(
                "apeSees.modal_deck: solver must be 'feast' (replicated "
                "band solve) or 'arpack' (partitioned lowest-N solve), "
                f"got {solver!r}."
            )
        if target != "tcl":
            raise NotImplementedError(
                "apeSees.modal_deck: target='pymp' (an OpenSeesMP-Python "
                "deck) is ADR 0077 unlock 2a and not implemented yet; use "
                "target='tcl'. For solver='arpack' this is not just "
                "unimplemented: the modern interpreter (openseespy / PyMP) "
                "still builds its ArpackSOE bare, so the distributed-eigen "
                "wiring of fork PR #668 is classic-Tcl only."
            )
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.modal_deck: staged models are not supported "
                "(per-stage parallel modal is deferred, ADR 0077 / "
                f"SSI-2.A) (got {len(self._stage_records)} stage(s))."
            )
        self._guard_modal_deck_constraint_handler()
        if solver == "arpack":
            if band is not None or certify:
                raise ValueError(
                    "apeSees.modal_deck: band= / certify= are FEAST-only "
                    "(the contour is the band). solver='arpack' selects "
                    "the lowest num_modes= modes; drop band=/certify= or "
                    "use solver='feast'."
                )
            if num_modes is None or int(num_modes) < 1:
                raise ValueError(
                    "apeSees.modal_deck: solver='arpack' needs "
                    f"num_modes >= 1, got {num_modes!r}."
                )
            self._modal_deck_arpack(
                path, num_modes=int(num_modes), out=out,
            )
            return

        if num_modes is not None:
            raise ValueError(
                "apeSees.modal_deck: num_modes= is ARPACK-only — for "
                "solver='feast' the contour IS the band and the found "
                "mode count is dynamic. Pass band= only, or use "
                "solver='arpack'."
            )
        if band is None:
            raise ValueError(
                "apeSees.modal_deck: solver='feast' needs band="
                "(f_min, f_max) in Hz."
            )
        f_min, f_max = band
        if not (0.0 <= f_min < f_max):
            raise ValueError(
                "apeSees.modal_deck: need 0 <= band[0] < band[1], got "
                f"{band!r}."
            )

        bm = self.build()
        emitter = TclEmitter(backend=deck_backend(self._opensees))
        # L3 FEAST needs the FULL model on every rank — force the flat
        # (replicated) emit even for a partition-authored fem, exactly as
        # the live emitter does (ADR 0077 P2 live finding).
        emitter.supports_partitions = False  # type: ignore[attr-defined]
        bm.emit(emitter)
        # Deterministic eigen preamble (every rank identical). The handler
        # matters (Penalty pollutes M, Lagrange injects zero-mass DOFs →
        # spurious modes); the numberer must number identically on every
        # rank (RCM); the system line is NOT in the FEAST solve path — a
        # serial UmfPack is correct even for the distributed run.
        emitter.constraints("Transformation")
        emitter.numberer("RCM")
        emitter.system("UmfPack")
        # P3 mode-shape harvest: pin the recorder column order to the
        # sorted mesh node tags (the sidecar the deck writes lets
        # ParallelModalResult.from_job map columns back without the deck).
        # Mesh nodes only — bridge-declared extra nodes (decoupled nodes)
        # are not harvested.
        shape_tags = tuple(sorted(int(t) for t in bm.fem.nodes.ids))
        emitter.eigen_feast_parallel(
            f_min, f_max, certify=certify, out=out,
            shape_nodes=shape_tags, shape_ndf=bm.ndf, shape_ndm=bm.ndm,
        )

        with open(path, "w", encoding="utf-8") as f:
            emitter.write_to(f)
        self._artifacts.after_emit(self)

    def _guard_modal_deck_constraint_handler(self) -> None:
        """Refuse a modal deck whose model needs a constraint handler
        other than ``Transformation`` (ADR 0077 INV-4 / INV-10).

        Both Tier-1 backends **force** ``constraints Transformation``
        after the model, because the alternatives corrupt an eigensolve:
        ``Penalty`` pollutes M with penalty mass and ``Lagrange`` injects
        zero-mass DOFs, either of which fabricates spurious modes. Two
        constructs need a *different* handler and are therefore mutually
        exclusive with a modal deck:

        * **``enforce="equation"`` ties** (ADR 0068 INV-4) emit
          ``equationConstraint`` rows, which ``Transformation`` cannot
          enforce — it drops them **silently**. The normal
          (``ops.tcl``) path auto-upgrades the handler to ``Lagrange`` /
          ``LadrunoProjection`` for exactly this reason.
        * **contact interactions**, which need ``LadrunoContact`` to
          inject the contact FE adapters into the assembly.

        Without this guard the forced ``Transformation`` line simply wins
        — it is emitted last — and the deck runs to completion producing
        a spectrum for a *different structure* than the user described,
        with no warning anywhere. Found by the ADR 0077 P6 adversarial
        pass, which measured a modal deck emitting six
        ``equationConstraint`` rows under a bare ``constraints
        Transformation``.

        Only one handler can be active, so there is no emit that
        satisfies both requirements: fail loud, exactly as the
        contact-plus-equation-tie combination already does.
        """
        if self._equation_constraint_records or _fem_has_equation_ties(
            self._fem,
        ):
            raise BridgeError(
                "apeSees.modal_deck: the model carries an enforce='equation' "
                "tie, which needs the 'Lagrange' (implicit) or "
                "'LadrunoProjection' (explicit) constraint handler — but a "
                "modal deck FORCES 'constraints Transformation' (ADR 0077 "
                "INV-4 / INV-10), because Lagrange injects zero-mass DOFs "
                "that fabricate spurious modes. Only one handler can be "
                "active, so the two are mutually exclusive: Transformation "
                "would silently DROP the equationConstraint rows and hand "
                "you a spectrum for a different structure. Switch the tie to "
                "enforce='penalty' / 'penalty_al' (handler-independent "
                "elements) for the modal run, or run the eigensolve "
                "single-process via ops.eigen(...) (Tier 0), which emits no "
                "forced handler."
            )
        if _fem_has_contacts(self._fem):
            raise BridgeError(
                "apeSees.modal_deck: the model carries contact interactions, "
                "which need the 'LadrunoContact' handler to inject the "
                "contact FE adapters — but a modal deck FORCES 'constraints "
                "Transformation' (ADR 0077 INV-4 / INV-10). Only one handler "
                "can be active, so the contact would be silently unenforced. "
                "Remove the contact for the modal run (a linear eigensolve "
                "about the undeformed state does not see it anyway), or run "
                "single-process via ops.eigen(...) (Tier 0)."
            )

    def _modal_deck_arpack(
        self, path: str, *, num_modes: int, out: str,
    ) -> None:
        """Emit a PARTITIONED distributed-ARPACK modal deck
        (ADR 0077 Tier 1B). Driven by :meth:`modal_deck`.

        The ordinary ``_emit_partitioned`` output — ``if {[getPID]==K}``
        blocks and all — plus a forced eigen preamble and one captured
        plain ``eigen $num_modes``. Each rank holds only its slice of the
        model **and** the ``(K−σM)`` factor+solve is distributed across
        ranks, which is why this is the backend for a model that does not
        fit on one node. Contrast the FEAST deck, which is replicated:
        every rank there assembles the full ``(K, M)``.

        **Runtime precondition — no feature probe exists.** This deck
        requires an ``OpenSeesMP.exe`` built from the fork's ``ladruno``
        branch at or after ``5a522b03b`` (PR #668), which gives
        ``ArpackSOE`` its ``setProcessID``/``setChannels`` and applies
        them when the analysis ``LinearSOE`` is ``MumpsParallelSOE``.
        Against an older binary the deck does **not** fail cleanly: it
        hangs, or returns an empty spectrum on rank 0 while rank 1
        deadlocks. Treat the build as a deployment precondition and gate
        on the fork's own smoke
        (``Ladruno_scripts/verify_arpack_mp_mumps.tcl``, run under a
        timeout, requiring ``cases=4`` on every rank).

        **``system Mumps`` is load-bearing here** (ADR 0077 INV-8) — the
        exact opposite of the FEAST deck, whose ``system`` line plays no
        part in the solve. The fork wires the ``ArpackSOE`` collectives
        *only* for ``MumpsParallelSOE``; emit anything else and the
        wiring silently does not engage, putting the run back on the
        broken global-K / local-M path. ``ParallelProfileSPD`` and
        ``MPIDiagonal`` are the trap — genuinely distributed, own
        collectives live, so the deck runs while the eigensolver stays
        rank-local. A user-declared non-``Mumps`` system therefore
        **raises** rather than being silently overridden. The numberer is
        treated differently on purpose: the forced ``ParallelPlain`` line
        comes last and wins over a declared serial one, which *rescues*
        the deck (a serial numberer fails ``LinearSOE::setSize`` before
        ``eigen`` runs) rather than quietly changing what the user asked
        for. Overriding a deliberate solver choice is not in the same
        category, so that one is refused.

        Nodal mass needs no special handling here, but only because the
        partitioned emit already does the right thing: the ``M*v`` merge
        sums per-rank contributions, so a boundary node given mass on two
        ranks would count twice — and since the Tcl ``eigen`` path always
        has ``shift = 0``, K would stay exactly right while M went wrong,
        surfacing as a plausible spectrum biased low rather than as an
        error. ``_emit_partitioned`` routes ``mass`` (like pattern
        ``load``) through ``primary_owner_map``, one owning rank per node
        (ADR 0077 INV-12).

        Harvest with :meth:`ParallelModalResult.from_job`: the rank-0
        eigenvalue write-out plus **per-rank** mode-shape sidecars
        (``mode_shapes_rank<P>.json`` + ``mode_shape_<k>_rank<P>.out``),
        which the reader merges. The FEAST rank-0 recorder cannot be
        reused — it would capture rank 0's slice only and return a
        partial field with no error.

        Cross-rank MP constraints (`equalDOF` / `rigidLink` /
        `rigidDiaphragm` / surface couplings straddling a partition) are
        supported, but **required an ADR 0027 fix landed alongside this
        backend** (INV-2, 2026-07-27): the foreign-node declaration used
        to emit the ghost node without the owner's ``fix``, which left a
        free massless DOF on the non-owning rank and made the assembled
        matrix singular — an empty spectrum here, and a failed
        ``analyze`` in any partitioned static run. A build predating
        that fix returns ``n_modes == 0`` for such a model.

        **This deck is NOT its own serial oracle** — unlike the FEAST
        one, which is replicated and therefore runs identically at any
        rank count. Run single-process, the ``getPID`` shim returns 0 and
        only rank 0's block executes, so ``OpenSees`` builds rank 0's
        **submodel** and solves that. Observed on the 8-mass chain split
        in two: the 4-DOF submodel fails ARPACK's ``NCV`` constraint and
        ``eigen`` returns an EMPTY list, which the deck happily writes —
        ``from_job`` then reports ``n_modes == 0``, the only visible
        signal. Run it under ``mpiexec``; for a serial oracle build the
        model unpartitioned and use Tier 0 (:meth:`eigen`).
        """
        from .emitter.tcl import TclEmitter, deck_backend

        if len(self._fem.partitions) < 2:
            raise ValueError(
                "apeSees.modal_deck: solver='arpack' needs a PARTITIONED "
                "model (len(fem.partitions) > 1) — the whole point of "
                "this backend is that each rank holds only its slice, "
                f"and it has no reason to exist unpartitioned (got "
                f"{len(self._fem.partitions)} partition(s)). Partition "
                "the mesh (g.mesh.partitioning), or use Tier 0 — the "
                "single-process ops.eigen(...) / ops.modal_properties("
                "...), which is faster at every size that fits on one "
                "node and is the only route to participation factors."
            )

        bm = self.build()
        declared_system = next(
            (p for p in bm.primitives if isinstance(p, LinearSystem)), None,
        )
        if declared_system is not None and (
            type(declared_system).__name__ != "Mumps"
        ):
            raise ValueError(
                "apeSees.modal_deck: solver='arpack' requires "
                "'system Mumps' — the fork wires the distributed "
                "ArpackSOE collectives ONLY when the analysis LinearSOE "
                "is MumpsParallelSOE (fork PR #668), so a "
                f"{type(declared_system).__name__!r} system silently "
                "leaves the eigensolve rank-local (ADR 0077 INV-8). This "
                "is the opposite of the FEAST deck, where the system "
                "line is inert. Declare ops.system.Mumps() (or none — "
                "the deck forces it), or use solver='feast'."
            )

        emitter = TclEmitter(backend=deck_backend(self._opensees))
        # The Tier-1B preamble is FORCED below, so suppress the ADR 0027
        # INV-5 auto-emit that would otherwise put a second (identical)
        # numberer/system pair above it. Nothing is lost: the auto-emit's
        # constraint handler is Transformation, which is what we force.
        #
        # CONTRACT (ADR 0092 review, F6): this seam is honored ONLY by
        # the partitioned emit path — `_emit_partitioned` reads it; the
        # flat auto-emit sites never consult it. That is sound
        # here because this deck REQUIRES len(partitions) > 1 (guarded
        # above), so the flat lane is unreachable. Any future producer
        # that can reach `_emit_flat` must first wire
        # the flag through those sites' `_maybe_auto_emit_*` calls, or
        # the suppression will silently not happen there.
        emitter.suppress_analysis_chain_auto_emit = True  # type: ignore[attr-defined]
        bm.emit(emitter)
        # Forced eigen preamble, emitted adjacent to the solve so the deck
        # reads as one unit. `constraints Transformation` unconditionally
        # (ADR 0077 INV-10: the auto-emit only fires when MP constraints
        # exist, while Penalty pollutes M with penalty mass and Lagrange
        # injects zero-mass DOFs — either fabricates spurious modes). The
        # numberer/system pair is the ADR 0027 INV-5 runtime-conditional
        # form: ParallelPlain/Mumps under OpenSeesMP (which is what
        # engages the #668 wiring — INV-8), degrading to RCM/UmfPack
        # under single-process OpenSees so the deck still PARSES there.
        # It does not solve the right problem there — see the "not a
        # serial oracle" note in the docstring.
        emitter.constraints("Transformation")
        emitter.parallel_runtime_fallback_numberer("ParallelPlain", "RCM")
        emitter.parallel_runtime_fallback_system("Mumps", "UmfPack")
        # Per-rank mode-shape harvest (ADR 0077 INV-9 / INV-12). Each
        # rank records the nodes IT owns, in sorted order; shared
        # boundary nodes land in two sidecars on purpose so from_job can
        # cross-check them. Mesh nodes only — bridge-declared extra nodes
        # (decoupled nodes) are not harvested, same as the FEAST path.
        mesh_nodes = {int(t) for t in bm.fem.nodes.ids}
        shape_nodes_by_rank = {
            runtime_rank_from_partition_record(rec, idx): tuple(sorted(
                n for n in (int(t) for t in rec.node_ids) if n in mesh_nodes
            ))
            for idx, rec in enumerate(bm.fem.partitions)
        }
        emitter.eigen_parallel(
            num_modes, out=out,
            shape_nodes_by_rank=shape_nodes_by_rank,
            shape_ndf=bm.ndf, shape_ndm=bm.ndm,
        )

        with open(path, "w", encoding="utf-8") as f:
            emitter.write_to(f)
        self._artifacts.after_emit(self)

    def py(
        self,
        path: str,
        *,
        run: bool = False,
        analyze_steps: int | None = None,
        analyze_dt: float | None = None,
        python: str | None = None,
        stream: bool = False,
        verbose: bool = False,
        log: str | None = None,
        progress: bool = True,
    ) -> "RunSolverStats | None":
        """Emit an openseespy Python deck to ``path``; optionally run it.

        ``run=True`` streams the openseespy subprocess exactly like
        :meth:`tcl` — full output always tee'd to a log (``log`` path
        override, else ``<path>.log``), console opt-in via ``verbose``,
        a live step counter from the ``progress`` markers, and a
        tail-only ``RuntimeError`` on a non-zero exit. ``verbose`` /
        ``log`` / ``progress`` are inert when ``run=False``. The
        ``RunSolverStats | None`` return is :meth:`tcl`'s (ADR 0106 D1).

        ``analyze_steps`` / ``analyze_dt`` semantics mirror :meth:`tcl`
        (Phase SSI-1).

        ``stream=True`` is out of scope for the Python deck emitter
        (v1) and fails loud — the HPC path is Tcl (ADR 0065 Tier 2 /
        plan_emit_memory_columnar.md A1–A3); use
        ``ops.tcl(path, stream=True)``.
        """
        from .emitter.py import PyEmitter
        from .emitter.tcl import deck_backend

        if stream:
            raise ValueError(
                "apeSees.py: stream=True is not supported for the "
                "Python deck emitter (v1) — the HPC path is Tcl "
                "(ADR 0065 Tier 2 / plan_emit_memory_columnar.md "
                "A1–A3); use ops.tcl(path, stream=True) instead."
            )
        bm = self.build()
        emitter = PyEmitter(backend=deck_backend(self._opensees))
        emitter._emit_progress = bool(progress)
        pre_prof, post_prof = self._split_profiler_records()
        bm.emit(emitter)
        for _verb, _vargs in pre_prof:
            emitter.profiler(_verb, *_vargs)
        if analyze_steps is not None:
            emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
        for _verb, _vargs in post_prof:
            emitter.profiler(_verb, *_vargs)
        with open(path, "w", encoding="utf-8") as f:
            emitter.write_to(f)
        self._artifacts.after_emit(self)

        if not run:
            return None

        python_bin = _resolve_python_binary(python, self._opensees)
        # PYTHONUNBUFFERED so the child's stdout streams live through the
        # pipe rather than block-buffering until exit (the tee + live
        # counter depend on it).
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        return stream_run(
            [python_bin, path],
            log_path=resolve_log_path(log, path),
            verbose=verbose,
            label=run_label(path, analyze_steps, analyze_dt),
            header="openseespy",
            env=child_env,
            deck_path=path,
            expect_solver_stats=_deck_requested_solver_stats(emitter),
        )

    def run(self, *, wipe: bool = True) -> None:
        """Drive an in-process LiveOpsEmitter through the full deck.

        This emits every primitive but does NOT call ``analyze`` —
        that is the user's call (or :meth:`analyze`'s). Useful when
        the user wants to declare a model, populate openseespy state,
        and then run their own analysis driver.
        """
        from .emitter.live import LiveOpsEmitter

        bm = self.build()
        emitter = LiveOpsEmitter(wipe=wipe)
        bm.emit(emitter)
        self._artifacts.after_emit(self)

    def run_remote(
        self,
        job_dir: str,
        *,
        cluster: "str | Cluster",
        np: int | None = None,
        name: str | None = None,
        deck: str = "main.tcl",
        binary: str | None = None,
        walltime: str | None = None,
        analyze_steps: int | None = None,
        analyze_dt: float | None = None,
        wait: bool = True,
        poll: float = 15.0,
        timeout: float | None = None,
        overwrite: bool = False,
    ) -> "Job":
        """Emit the Tcl deck and run it on a SLURM cluster (ADR 0060 sugar).

        One call for the whole loop: emit into ``job_dir`` -> push ->
        ``sbatch`` -> poll to completion -> fetch results back into
        ``job_dir``. Wraps :class:`apeGmsh.hpc.Cluster` /
        :class:`apeGmsh.hpc.Job`; use those directly for finer control
        (or pass ``wait=False`` to get the live :class:`Job` handle back
        right after submission).

        Parameters
        ----------
        job_dir
            Local directory the deck is emitted into and results are
            fetched back into. Created if missing.
        cluster
            Cluster name in ``~/.apegmsh/clusters.toml`` (e.g.
            ``"esmeralda"``) or a constructed ``Cluster``.
        np
            MPI ranks. Defaults to the model's partition count
            (``len(fem.partitions)``), or 1 for a flat model.
        analyze_steps / analyze_dt
            Forwarded to :meth:`tcl` — appends the ``analyze`` drive
            line exactly as the local emit would.
        wait
            ``True`` (default) blocks until the job ends and fetches.
            ``False`` returns the submitted ``Job`` immediately;
            poll/fetch it yourself (it survives sessions via
            ``Job.load(job_dir)``).

        Raises
        ------
        HPCError
            If the job ends in any state other than ``COMPLETED``
            (results and logs are still fetched first; the message
            carries the stderr tail).
        """
        from pathlib import Path as _Path

        from ..hpc import Cluster as _Cluster
        from ..hpc import HPCError as _HPCError
        from ..hpc import JobStatus as _JobStatus

        resolved = (
            _Cluster.load(cluster) if isinstance(cluster, str) else cluster
        )
        ranks = np if np is not None else max(1, len(self._fem.partitions))
        path = _Path(job_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.tcl(
            str(path / deck),
            analyze_steps=analyze_steps,
            analyze_dt=analyze_dt,
        )
        job = resolved.submit(
            path,
            np=ranks,
            name=name,
            deck=deck,
            binary=binary,
            walltime=walltime,
            overwrite=overwrite,
        )
        if not wait:
            return job
        status = job.wait(poll=poll, timeout=timeout)
        # Fetch BEFORE the verdict: on failure the logs are the evidence.
        job.fetch()
        if status is not _JobStatus.COMPLETED:
            raise _HPCError(
                f"remote job {job.name!r} (slurm {job.slurm_id}) ended "
                f"{status.value}; logs fetched into {job.local_dir}.\n"
                f"--- stderr tail ---\n{job.tail(30, stream='err')}"
            )
        return job

    def h5(
        self,
        path: str,
        *,
        model_name: str | None = None,
        cuts: "Sequence[SectionCutDef]" = (),
        sweeps: "Sequence[SectionSweepDef]" = (),
    ) -> None:
        """Emit a model-definition HDF5 archive at ``path``.

        Phase 8.5 composes the file in two layers:

        1. The **broker** (``self._fem``) writes ``/meta`` + the
           neutral zone (``/nodes``, ``/elements/{type}``,
           ``/physical_groups``, ``/labels``, ``/constraints/{kind}``,
           ``/loads/{kind}/{pattern}``, ``/masses``).  Broker writers
           live in :mod:`apeGmsh.mesh._femdata_h5_io`.
        2. The **bridge** (an :class:`H5Emitter` driven through the
           :class:`BuiltModel`) appends ``/opensees/...`` enrichment.
        3. apeGmsh.cuts v4: if ``cuts`` and / or ``sweeps`` are
           supplied, they're persisted under ``/opensees/cuts/`` and
           ``/opensees/sweeps/`` (writer in
           :mod:`apeGmsh.cuts._h5_io`).

        If ``self._fem`` does not expose a real :class:`FEMData`
        surface (e.g. integration tests using a hand-rolled stub),
        the broker step is skipped: the file ends up with the
        bridge's own ``/meta`` plus ``/opensees/...``, but no neutral
        zone.  Real callers always get the full file shape.

        Parameters
        ----------
        path
            File path to write the HDF5 archive to.
        model_name
            Optional human-readable name written to ``/meta/model_name``.
            Defaults to the path's stem.
        cuts
            Optional sequence of :class:`apeGmsh.cuts.SectionCutDef`
            to persist under ``/opensees/cuts/cut_{i}``.  Each cut
            travels with the model definition; the viewer auto-loads
            them from the file the next time ``Results.viewer(...)``
            is opened against a results.h5 carrying the same
            ``/opensees/`` zone (Phase 8 / ADR 0020 Composed-file
            pattern).
        sweeps
            Optional sequence of :class:`apeGmsh.cuts.SectionSweepDef`
            to persist under ``/opensees/sweeps/sweep_{i}``.  Each
            sweep group carries its own ``cuts/`` sub-group in sweep
            order (see ``apeGmsh/cuts/ARCHITECTURE.md`` "## v4").

        """
        # ADR 0055 Phase 2 + Phase 5 (P5.1, schema 2.19.0): staged
        # builds archive — the H5 emitter captures the per-stage emit
        # stream into ``/opensees/stages`` (see ``set_stage_records``
        # below).  For PARTITIONED staged builds the capture is
        # rank-agnostic by construction: replicated per-rank emission
        # dedupes on record identity, per-rank pattern/region
        # fragments merge by tag, and foreign ghost-node declarations
        # are filtered out of the stage buckets via the
        # ``set_stage_owned_node_tags`` side-channel — the stage zone
        # carries the flat logical program (rank-major capture order),
        # while the per-rank shape stays derivable from the neutral
        # ``/partitions`` zone.  The one staged remainder is the
        # phantom-node degrade (stage-claimed ``node_to_surface``),
        # which ``set_stage_records`` keeps fail-loud for flat and
        # partitioned builds alike.
        # ADR 0055 Phase 1: GLOBAL ``ops.initial_stress(...)`` archival is
        # supported — the records persist declaratively to
        # ``/opensees/initial_stress`` and replay re-runs the emit helpers
        # (see ``set_initial_stress_records`` below).  Per-stage
        # initial-stress persists with its stage under
        # ``/opensees/stages`` (Phase 2).

        from .emitter.h5 import H5Emitter

        snapshot_id = ""
        try:
            snapshot_id = str(self._fem.snapshot_id)
        except Exception:
            # FEM snapshots produced by some legacy paths may not have
            # a snapshot_id; tolerate gracefully (the H5 emitter writes
            # an empty string into /meta/snapshot_id, which the schema
            # already allows).
            snapshot_id = ""

        name = model_name or _path_stem(path)
        bm = self.build()
        emitter = H5Emitter(model_name=name, snapshot_id=snapshot_id)
        bm.emit(emitter)
        # Ledgered verbs whose own deferred warning reached the user; the
        # ledger warning below skips only those (one dropped row, one
        # warning, on the explicit and the automatic path alike).
        deferred_shown: set[str] = set()
        if bm.equation_constraint_records:
            # The deck zone has no equationConstraint record and, unlike an
            # enforce="equation" tie, a bridge-level row has no neutral-zone
            # twin either — so it does not survive a from_h5 round-trip.
            from .emitter.h5 import H5FeatureDeferredWarning
            if _warn_if_shown(
                f"ops.h5: {len(bm.equation_constraint_records)} "
                "apeSees.equation_constraint row(s) are NOT archived — the "
                "model.h5 has no record for them, so a model rebuilt from it "
                "runs without the constraint. Emit Tcl / openseespy (or run "
                "in-process) for the complete model.",
                H5FeatureDeferredWarning,
            ):
                deferred_shown.add("equationConstraint")

        # ADR 0055 Phase 1: hand the declarative global initial-stress
        # records to the emitter via the side-channel (the Protocol
        # ``step_hook_ramp`` / ``addToParameter`` calls bm.emit just drove
        # were no-op'd on H5 — they carry the resolved form).  ``bm.emit``
        # only emits the GLOBAL bucket at 7d; per-stage records ride the
        # stage side-channel below (ADR 0055 Phase 2).
        emitter.set_initial_stress_records(self._initial_stress_records)

        # ADR 0055 Phase 2: attach the declarative per-stage complement
        # (activated_pgs, per-stage initial-stress, activate_absorbing)
        # to the stage buckets the emitter captured in-band during
        # ``bm.emit``, and fail loud on any capture/record drift.
        # Called UNCONDITIONALLY (gate-2): a zero-record build that
        # somehow captured brackets must trip the count cross-check,
        # not silently write orphan buckets.
        emitter.set_stage_records(bm.stage_records)

        # ADR 0048 / 0049 — recompute the EFFECTIVE per-node ndf map (the same
        # deterministic inputs bm.emit used: inferred ∪ the ops.ndf overlay)
        # so the persisted /opensees/nodes_ndf matches the emitted deck exactly
        # and model_hash stays stable across a from_h5 → to_h5 round-trip.  The
        # overlay must fold in here too, else a STATED decoupled-node ndf is
        # lost on the FIRST write (not just round-trip).
        _elements = [p for p in bm.primitives if isinstance(p, Element)]
        _inferred = infer_node_ndf(self._fem, _elements, bm.ndm)
        _overlay = resolve_ndf_overlay(
            self._fem, bm.ndf_records, _inferred, bm.ndm,
        )
        _nodes_ndf = {**_inferred, **_overlay}

        # Single composition path, shared with ModelData.write (ADR
        # 0018 / _internal.compose).  apeSees passes snapshot_id=None:
        # the broker / bridge meta write is authoritative here, so
        # this stays byte-invariant with the pre-extraction code.
        # ADR 0112 D3: the bridge's declaration provenance rides along
        # and is appended to the snapshot's own table in /provenance.
        _compose_model_h5(
            self._fem, emitter, path,
            model_name=name,
            ndm=int(bm.ndm),
            ndf=int(self._ndf or 0),
            cuts=cuts,
            sweeps=sweeps,
            names=self._name_records(),
            computed_sections=self._computed_section_records(bm.primitives),
            nodes_ndf=_nodes_ndf,
            provenance=self._provenance.snapshot(),
        )
        # A verb whose deferred warning the user saw above is skipped; the
        # automatic write silences that class, so there the ledger warns.
        self._warn_ledger(
            emitter.ledger_counts, already_warned=frozenset(deferred_shown),
        )

    def _warn_ledger(
        self, counts: "Mapping[str, int]", *,
        already_warned: "frozenset[str]" = frozenset(),
    ) -> None:
        """Warn once per distinct set of ``ledger`` verbs ``h5()`` dropped.

        ADR 0114 Q3: a ``ledger`` call (contact, rebar, embed,
        ``equationConstraint``) leaves no ``/opensees`` record, so a deck
        replayed from the file omits it. The guard is per instance and
        per verb set, so a model that writes on every emit warns once,
        and again only when a new ledgered verb appears. Verbs in
        ``already_warned`` raised a dedicated warning of their own and
        are left out, so no dropped row warns twice.
        """
        verbs = frozenset(
            v for v, n in counts.items() if n and v not in already_warned)
        if not verbs or verbs in self._ledger_warned:
            return
        self._ledger_warned.add(verbs)
        import warnings as _warnings

        from ._internal.build import _stacklevel_outside_package
        from .emitter.h5 import H5LedgerWarning
        detail = ", ".join(f"{v} x{counts[v]}" for v in sorted(verbs))
        # The explicit ``ops.h5()`` and the automatic write after an emit
        # both reach here; point at the user's call either way.
        _warnings.warn(
            f"model.h5: {sum(counts[v] for v in verbs)} call(s) to ledgered "
            f"verbs ({detail}) are not carried by the /opensees archive, "
            "so a deck replayed from it omits them unless the neutral zone "
            "re-derives them (ADR 0114 Q3).",
            H5LedgerWarning,
            stacklevel=_stacklevel_outside_package(),
        )

    # -- Registration -----------------------------------------------------

    def _register(
        self, prim: _P, *, name: str | None = None,
        synthesised: str | None = None,
    ) -> _P:
        """Add ``prim`` to the bridge, allocate its tag, return it.

        When ``name`` is given, register it as a bridge-side alias for
        ``prim`` so reference kwargs (``series=``, ``material=``,
        ``transf=``, …) can later refer to it by string as well as by
        the returned handle.  Names are unique per bridge instance; a
        duplicate raises ``ValueError`` (fail-loud — no silent
        last-wins).

        ADR 0112 D3: the registration records its declaration
        provenance as ``opensees/<kind>/<name|#k>``, where ``<kind>``
        is the tag-allocator kind (the OpenSees command: ``element``,
        ``uniaxialMaterial``, ``pattern``, ...).  The helper keeps one
        record per user call.  A primitive the bridge synthesises inside
        a verb the user called (the HOLD series and pattern of
        ``s.support``, the series and pattern of ``imposed_displacement``)
        is registered with ``synthesised=<verb>:<owner>[/<role>]`` and
        gets a record under that key, pointing at the verb call, with
        ``origin = "synthesised"`` and no ``#k`` number (maintainer
        ruling on #1378).
        """
        kind = _kind_of(prim)
        # An empty name is no name, exactly as ``capture`` reads it: the
        # primitive is unnamed for the key, the alias table and the
        # record alike (#1378 round 4).
        name = name or None
        # Every refusal runs before the tag is allocated and the
        # primitive appended, so a refused call leaves no primitive, no
        # tag and no record.  A primitive registered before (P11:
        # ``allocate_for`` is idempotent on the object) keeps its record
        # and is not checked again.
        already = self._tags.tag_for(prim) is not None
        if name is not None:
            existing = self._names.get(name)
            if existing is not None and existing is not prim:
                raise ValueError(
                    f"apeSees: name {name!r} is already registered to a "
                    f"{type(existing).__name__}; names must be unique per "
                    "bridge.  Pick a different name= (or pass the object "
                    "handle directly)."
                )
        if not already:
            # User names and synthesised keys share one key space per
            # family; a collision on either side fails here.
            if synthesised is not None:
                key, what = synthesised, "synthesised key"
            elif name is not None:
                key, what = name, "name"
            else:
                key = self._provenance.next_unnamed_key("opensees", kind)
                what = "unnamed key"
            if self._provenance.has("opensees", kind, key):
                raise ValueError(
                    f"apeSees: the {what} {key!r} of this "
                    f"{type(prim).__name__} collides with the existing "
                    f"provenance record 'opensees/{kind}/{key}' (user names "
                    "and synthesised keys share one key space per family); "
                    "rename one of them.  Nothing was registered."
                )
        self._tags.allocate_for(prim, kind)
        self._primitives.append(prim)
        if name is not None:
            self._names[name] = prim
        if not already:
            if synthesised is not None:
                self._provenance.capture_synthesised(
                    "opensees", kind, synthesised)
            else:
                self._provenance.capture(
                    "opensees", kind, name, on_existing="raise")
        return prim

    def register(self, prim: _P) -> _P:
        """Register a standalone primitive with the bridge (P11).

        Idempotent on the object: registering an instance the bridge
        already holds (a namespace handle, or a second ``register``)
        returns it unchanged, so it stays once in the primitive list
        (#1409).
        """
        if self._tags.tag_for(prim) is not None:
            return prim
        return self._register(prim)

    def _resolve(
        self,
        ref: "_P | str",
        *,
        base: type[Primitive] = Primitive,
    ) -> "_P":
        """Resolve a reference that may be a primitive handle OR a name.

        This is what makes every reference kwarg dual-mode (object or
        name), mirroring the session side where composites return an
        object that can also be addressed by its registered name.  A
        non-string ``ref`` is returned untouched (the common
        pass-the-handle path).  A ``str`` is looked up in the alias
        table; an unknown name or a kind mismatch fails loud.
        """
        if not isinstance(ref, str):
            return ref
        prim = self._names.get(ref)
        if prim is None:
            known = ", ".join(sorted(self._names)) or "<none registered>"
            raise KeyError(
                f"apeSees: no primitive registered under name {ref!r}.  "
                f"Known names: {known}.  Pass name= when constructing the "
                "primitive, or hand the object handle directly."
            )
        if not isinstance(prim, base):
            raise TypeError(
                f"apeSees: name {ref!r} refers to a {type(prim).__name__}, "
                f"but a {base.__name__} is required here."
            )
        return prim  # type: ignore[return-value]

    def tag_for(self, prim: Primitive) -> int | None:
        """Return ``prim``'s allocated tag, or ``None`` if unregistered."""
        return self._tags.tag_for(prim)

    def _name_records(self) -> tuple[tuple[str, str, int], ...]:
        """Resolve the name-alias table to ``(name, kind, tag)`` records.

        Sorted by name for a deterministic ``/opensees/names`` layout.
        Skips any alias whose primitive somehow lost its tag (defensive
        — registration always allocates one).
        """
        records: list[tuple[str, str, int]] = []
        for name, prim in self._names.items():
            tag = self._tags.tag_for(prim)
            if tag is None:
                continue
            records.append((name, _kind_of(prim), int(tag)))
        records.sort(key=lambda r: r[0])
        return tuple(records)

    def _computed_section_records(
        self, primitives: "tuple[Primitive, ...]",
    ) -> tuple[tuple[int, str, str], ...]:
        """Provenance rows for every registered ``ComputedSection``
        (ADR 0078 Amendment A1): ``(tag, analyzer_name, payload)``.

        Called after a successful ``bm.emit`` — the analyses backing
        the payload are already memoized, so this never re-solves.
        Sorted by tag for a deterministic sidecar layout.
        """
        from .section.computed import ComputedSection
        from ._internal._computed_sections_h5 import computed_section_payload

        records: list[tuple[int, str, str]] = []
        for prim in primitives:
            if not isinstance(prim, ComputedSection):
                continue
            tag = self._tags.tag_for(prim)
            if tag is None:  # pragma: no cover - registration allocates
                continue
            records.append((
                int(tag),
                prim.analysis.name or "",
                computed_section_payload(prim),
            ))
        records.sort(key=lambda r: r[0])
        return tuple(records)

    # -- Build -----------------------------------------------------------

    def build(self) -> BuiltModel:
        """Freeze the declarations into a :class:`BuiltModel`."""
        if self._ndm is None or self._ndf is None:
            raise RuntimeError(
                "apeSees.model(ndm=..., ndf=...) must be called before "
                "build()."
            )
        self._check_damping_attached()

        tag_for: dict[int, int] = {
            id(p): self._tags.tag_for(p) or 0 for p in self._primitives
        }
        fix_records = tuple(self._fix_records)
        if self._fix_from_model:
            fix_records += fix_records_from_model(
                self._fem,
                [p for p in self._primitives if isinstance(p, Element)],
                self._ndm, self._ndf,
                ndf_records=self._ndf_records,
                explicit=(
                    *fix_records,
                    *(r for st in self._stage_records for r in st.fix_records),
                    *(r for st in self._stage_records
                      for r in st.support_records),
                ),
            )
        return BuiltModel(
            primitives=tuple(self._primitives),
            tag_for=tag_for,
            ndm=self._ndm,
            ndf=self._ndf,
            fem=self._fem,
            fix_records=fix_records,
            mass_records=tuple(self._mass_records),
            region_records=tuple(self._region_records),
            ndf_records=tuple(self._ndf_records),
            initial_stress_records=tuple(self._initial_stress_records),
            stage_records=tuple(self._stage_records),
            rayleigh_records=tuple(self._rayleigh_records),
            damping_attach_records=tuple(self._damping_attach_records),
            modal_damping_records=tuple(self._modal_damping_records),
            equation_constraint_records=tuple(
                self._equation_constraint_records,
            ),
            name_to_tag={
                nm: tag for nm, _kind, tag in self._name_records()
            },
            mass_from_model=self._mass_from_model,
            element_tags=self._element_tags,
        )

    # -- Internal helpers ------------------------------------------------

    def _check_damping_attached(self) -> None:
        """Fail loud on a ``damping`` object attached to nothing (ADR 0053).

        A ``Damping`` primitive only dissipates once it is bound to elements
        — either via a region (``ops.damping.<type>(on=...)`` → a
        :class:`DampingAttachRecord`) or directly on a supported element
        (``ops.element.<Type>(..., damp=obj)``). There is no global
        ``-damp`` in OpenSees, so a registered object referenced by neither
        would emit a dangling ``damping <Type>`` line that damps nothing.
        We catch it here (the one point where both attach routes are known)
        rather than at the ``ops.damping.*`` call, since the element route
        is declared afterward.
        """
        region_attached = {
            id(rec.prim) for rec in self._damping_attach_records
        }
        # D5: a stage-bound object attaches inside its stage's pool.
        for stage in self._stage_records:
            region_attached.update(
                id(rec.prim) for rec in stage.damping_attach_records
            )
        element_attached = {
            id(damp)
            for p in self._primitives
            if isinstance(p, Element)
            for damp in (getattr(p, "damp", None),)
            if damp is not None
        }
        for prim in self._primitives:
            if not isinstance(prim, Damping):
                continue
            if id(prim) in region_attached or id(prim) in element_attached:
                continue
            raise BridgeError(
                f"ops.damping.{type(prim).__name__.lower()}: this damping "
                "object attaches to nothing — pass on= (region attach) or "
                "hand it to a supported element's damp= kwarg. There is no "
                "global -damp.",
            )

    def _check_explicit_solver_compat(self) -> None:
        """Guard the explicit integrator / linear-system / mass pairings.

        Two checks (from the explicit-dynamics design review):

        * **RAISE** (silently-wrong): ``Diagonal`` / ``MPIDiagonal`` solves
          only the diagonal of the assembled matrix, so an element with
          *consistent* mass (``c_mass=True``) would have its off-diagonal
          mass dropped with no error — wrong results. apeGmsh cannot reach
          OpenSees' ``-lumped`` row-sum salvage, so this is a hard error.
        * **WARN** (correct-but-slow): an explicit integrator paired with a
          non-diagonal system factors the full mass every step, losing the
          ``O(N)`` explicit advantage.
        """
        from .analysis.system import Diagonal, MPIDiagonal

        system = next(
            (p for p in self._primitives if isinstance(p, LinearSystem)), None,
        )
        is_diagonal = isinstance(system, (Diagonal, MPIDiagonal))

        if is_diagonal:
            consistent = sorted({
                type(p).__name__
                for p in self._primitives
                if isinstance(p, Element) and getattr(p, "c_mass", False)
            })
            if consistent:
                raise BridgeError(
                    f"system {type(system).__name__} solves only the DIAGONAL "
                    "of the mass matrix, but these elements use consistent "
                    f"mass (c_mass=True): {', '.join(consistent)}. The "
                    "off-diagonal mass would be silently discarded, giving "
                    "wrong results. Drop c_mass=True (use lumped mass) with a "
                    "diagonal solver, or choose a non-diagonal system "
                    "(e.g. ops.system.ProfileSPD())."
                )

        integrator = next(
            (p for p in self._primitives if isinstance(p, Integrator)), None,
        )
        if (
            _is_explicit_integrator(integrator)
            and system is not None
            and not is_diagonal
        ):
            import warnings as _warnings
            _warnings.warn(
                f"Explicit integrator {type(integrator).__name__} paired with "
                f"system {type(system).__name__}: correct, but it factors the "
                "full mass matrix every step, losing the O(N) advantage of "
                "explicit integration. Use ops.system.Diagonal() (lumped "
                "diagonal mass) for explicit runs.",
                OpenSeesExplicitSolverWarning,
                stacklevel=3,
            )

    def _warn_if_unguarded_explicit_run(self) -> None:
        """Warn that :meth:`analyze_explicit` sizes ``dt`` once at ``t=0``.

        The fixed step is blind to a stiffening tangent. If the registered
        integrator has neither ``cfl_abort`` nor ``recompute`` set, a
        mid-run CFL violation would diverge silently instead of aborting —
        surface that so the user can opt into the fork's guard.
        """
        integrator = next(
            (p for p in self._primitives if isinstance(p, Integrator)), None,
        )
        guarded = bool(
            getattr(integrator, "cfl_abort", False)
            or getattr(integrator, "recompute", None)
        )
        if not guarded:
            import warnings as _warnings
            _warnings.warn(
                "analyze_explicit sizes dt once on the initial stiffness and "
                "holds it for the whole run. On a stiffening model (contact "
                "closing, geometric / material stiffening) the critical step "
                "shrinks and a fixed dt can diverge. Construct the integrator "
                "with cfl_abort=True (and recompute=N) so a mid-run CFL "
                "violation aborts and is re-raised, instead of diverging.",
                OpenSeesExplicitSolverWarning,
                stacklevel=3,
            )

    def _check_analysis_chain_for_analyze(self) -> None:
        """Raise :class:`BridgeError` if the analysis chain is incomplete."""
        required: tuple[tuple[type[Primitive], str], ...] = (
            (ConstraintHandler,  "constraints"),
            (Numberer,           "numberer"),
            (LinearSystem,       "system"),
            (ConvergenceTest,    "test"),
            (SolutionAlgorithm,  "algorithm"),
            (Integrator,         "integrator"),
            (Analysis,           "analysis"),
        )
        missing: list[str] = []
        for base, name in required:
            if not any(isinstance(p, base) for p in self._primitives):
                missing.append(name)
        if missing:
            raise BridgeError(
                "apeSees.analyze: analysis chain is incomplete; "
                f"missing: {', '.join(missing)}. Register the missing "
                "primitives via ops.<family>.<Type>(...) before calling "
                "analyze()."
            )


# ---------------------------------------------------------------------------
# TIMs A8 — per-stage profiler bracket
# ---------------------------------------------------------------------------

def _stage_profile_start_flags(profile: "ProfileRecord") -> list[str]:
    """Build the ``profiler start`` flag list for a stage's bracket.

    Mirrors ``_ProfilerNS.start`` exactly (``-deep`` / ``-memory`` /
    ``-perStep``) — see ``_internal/ns/profiler.py``.
    """
    flags: list[str] = []
    if profile.deep:
        flags.append("-deep")
    if profile.memory:
        flags.append("-memory")
    if profile.per_step:
        flags.append("-perStep")
    return flags


# ---------------------------------------------------------------------------
# ADR 0057 Phase A — stage strategy resolution
# ---------------------------------------------------------------------------

def _stage_strategy_spec(stage: "StageRecord") -> StrategySpec | None:
    """Resolve a stage's ADR 0057 ladder to its emitter-ready spec.

    Rung 0 = the stage chain's own algorithm.  The stage builder types
    the chain slots as generic ``Primitive``; the isinstance narrows it
    back to :class:`SolutionAlgorithm` (always true for a chain built
    through ``s.analysis(algorithm=...)``).
    """
    if stage.strategy is None:
        return None
    base = (
        stage.algorithm
        if isinstance(stage.algorithm, SolutionAlgorithm) else None
    )
    return stage.strategy.to_spec(base=base)


def _deck_requested_solver_stats(emitter: object) -> bool:
    """Did the deck just emitted through *emitter* ask for ``-stats``?

    ADR 0106 D2/D4: "the same predicate answers D4's *was a block
    expected?*". :meth:`apeSees.emit` already ran
    :func:`deck_requests_solver_stats` over this deck's flat / staged
    system declarations and stamped the answer on the emitter to gate
    the ``APEGMSH_STAGE`` marker — reading it back is the same facts,
    not a second resolution that could disagree with the bytes.
    """
    return bool(getattr(emitter, "_emit_stage_markers", False))


# ---------------------------------------------------------------------------
# Binary resolution helpers
# ---------------------------------------------------------------------------

def _resolve_opensees_binary(
    explicit: str | None, target: "OpenSeesTarget | None" = None
) -> str:
    """Resolve the OpenSees Tcl binary path (see :mod:`._target`)."""
    from ._target import resolve_opensees_binary

    return resolve_opensees_binary(explicit, target)


def _resolve_python_binary(
    explicit: str | None = None, target: "OpenSeesTarget | None" = None
) -> str:
    """Resolve the python interpreter for an openseespy script (see :mod:`._target`)."""
    from ._target import resolve_python_binary

    return resolve_python_binary(explicit, target)
