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
import re
import subprocess
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, TypeVar

from ._internal.build import (
    BridgeError,
    DampingAttachRecord,
    ActivateAbsorbingRecord,
    ElementRemovalRecord,
    FixRecord,
    InitialStressRecord,
    MassRecord,
    ModalDampingRecord,
    NdfRecord,
    RayleighRecord,
    RegionAssignmentRecord,
    SPRemovalRecord,
    StageRecord,
    SupportRecord,
    _emit_node_with_inferred_ndf,
    allocate_element_tags,
    bucket_pre_allocated_by_rank,
    build_element_partition_owner,
    build_node_partition_owners,
    close_builder_ndf_bracket,
    compute_stage_ownership,
    emit_activate_absorbing,
    emit_element_spec_partitioned,
    emit_initial_stress_addtoparameter,
    emit_initial_stress_global,
    emit_mp_constraints,
    emit_mp_constraints_partitioned,
    emit_reinforce_ties,
    emit_stage_mp_constraints,
    emit_stage_mp_constraints_partitioned,
    emit_pattern_spec,
    emit_recorder_spec,
    emit_transform_specs,
    expand_pg_to_elements,
    expand_pg_to_nodes,
    is_partitioned,
    open_builder_ndf_bracket,
    primary_owner_map,
    runtime_rank_from_partition_record,
    topological_order,
    validate_node_ndf_element_compat,
    validate_absorbing_quad_geometry,
    validate_body_force_double_count,
    infer_node_ndf,
    validate_adaptive_element_endpoints,
    resolve_ndf_overlay,
    validate_constraint_master_ndf,
    validate_record_ndf_consistency,
    fit_dof_vector,
    assert_ndm_compatible,
)
from ._internal.build import _element_transf as _build_element_transf
from ._internal.tag_resolution import (
    MISSING_FEM_ELEMENT_ID,
    set_current_fem_element_id,
    set_element_nodes,
    set_stage_owned_node_tags,
)
from ._internal.compose import _compose_model_h5, _path_stem
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
    _StageDampingNS,
    _StrategyNS,
    _SystemNS,
    _TestNS,
    _TimeSeriesNS,
    _UniaxialMaterialNS,
)
from ._internal.tag_allocator import TagAllocator
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
from .emitter.base import Emitter, StrategySpec
from .node import Node, _NodeAccessor, _iter_tags
from .recorder import FilterableRecorder, Ladruno
from .transform import Cartesian, Orientation

if TYPE_CHECKING:
    from pathlib import Path

    # FEMData is the only mesh symbol the bridge depends on (P3, P9).
    # Imported under TYPE_CHECKING so that constructing apeSees does
    # not transitively import gmsh during static analysis.
    # Use the fully-qualified module path to disambiguate from the
    # similarly-named submodule ``apeGmsh.mesh.FEMData`` under mypy.
    from apeGmsh.cuts import SectionCutDef, SectionSweepDef
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh._kernel.records._constraints import ConstraintRecord
    from .analysis.strategy import Ladder
    from .emitter.tcl import PartitionSpan
    from ._target import OpenSeesCapabilities, OpenSeesTarget
    from .pattern.pattern import Plain
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.results.capture.spec import DomainCaptureSpec
    from apeGmsh.hpc import Cluster, Job

    from .analysis.eigen import EigenResult


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

    Returns False when neither composite exists or both are empty
    (defensive on test stubs that don't carry the full broker shape).
    """
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


# ---------------------------------------------------------------------------
# Split-emit layout (ADR 0043 slice 1.1, mode A)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _SplitLayout:
    """Line-span map produced by :meth:`BuiltModel._emit_split`.

    Carves the single emitter buffer into a contiguous per-module band
    ``[module_start, module_end)`` and, within it, each module's
    ``(label, start, end)`` sub-span.  The Tcl / Py writers slice the
    buffer with this map: ``lines[start:end]`` is module ``label``'s
    fragment body; ``lines[:module_start]`` is the driver preamble
    (model + definitions + transforms) and ``lines[module_end:]`` is the
    driver tail (interface + loads + patterns + recorders).
    """

    module_start: int
    module_end: int
    modules: "list[tuple[str, int, int]]"


def _split_safe_name(label: str, used: "set[str]") -> str:
    """Map a compose module label to a collision-free fragment stem.

    Empty (host) label → ``"host"``; any character outside
    ``[0-9A-Za-z_-]`` (e.g. the nested-compose ``/`` separator) →
    ``_``; duplicates are disambiguated with a numeric suffix.
    """
    base = label if label != "" else "host"
    safe = re.sub(r"[^0-9A-Za-z_-]", "_", base) or "host"
    candidate = safe
    i = 1
    while candidate in used:
        candidate = f"{safe}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _write_split_tcl(
    path: str, lines: "list[str]", layout: "_SplitLayout",
) -> None:
    """Write a Tcl driver at ``path`` + ``parts/<label>.tcl`` fragments.

    The driver ``source``s each fragment (relative to its own
    location, so the deck runs from any cwd) between the definitions
    preamble and the interface / loads / recorders tail.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    used: set[str] = set()
    source_lines: list[str] = []
    for label, start, end in layout.modules:
        safe = _split_safe_name(label, used)
        body = lines[start:end]
        with open(
            os.path.join(parts_dir, f"{safe}.tcl"), "w", encoding="utf-8",
        ) as f:
            f.write(f"# apeGmsh split fragment: {label or 'host'}\n")
            if body:
                f.writelines(ln + "\n" for ln in body)
        source_lines.append(
            f"source [file join [file dirname [info script]] "
            f"parts {safe}.tcl]"
        )

    driver = (
        lines[: layout.module_start]
        + ["", "# --- module fragments (ADR 0043 split='parts') ---"]
        + source_lines
        + [""]
        + lines[layout.module_end:]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(ln + "\n" for ln in driver)


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


def _write_split_py(
    path: str, lines: "list[str]", layout: "_SplitLayout",
) -> None:
    """Write a Py driver at ``path`` + ``parts/<label>.py`` fragments.

    Each fragment exposes ``def build(ops): ...``; the driver loads
    each fragment by explicit file path via ``importlib`` (no
    ``sys.path`` mutation, no bare-module-name collisions) and calls
    ``build(ops)`` against the driver's own ``ops`` handle.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    used: set[str] = set()
    call_lines: list[str] = []
    for label, start, end in layout.modules:
        safe = _split_safe_name(label, used)
        body = lines[start:end]
        with open(
            os.path.join(parts_dir, f"{safe}.py"), "w", encoding="utf-8",
        ) as f:
            f.write(f"# apeGmsh split fragment: {label or 'host'}\n")
            f.write("def build(ops):\n")
            if body:
                for ln in body:
                    f.write(f"    {ln}\n")
            else:
                f.write("    pass\n")
        # Load each fragment by explicit file path (no sys.path
        # mutation, no bare-module-name collisions) and call its
        # ``build(ops)`` against the driver's own ops handle.
        call_lines.append(
            f"_apesees_load('_apesees_frag_{safe}', '{safe}.py').build(ops)"
        )

    inject = (
        [
            "",
            "# --- module fragments (ADR 0043 split='parts') ---",
            "import importlib.util as _ilu, os as _os",
            "def _apesees_load(_name, _file):",
            "    _path = _os.path.join(_os.path.dirname("
            "_os.path.abspath(__file__)), 'parts', _file)",
            "    _spec = _ilu.spec_from_file_location(_name, _path)",
            "    _mod = _ilu.module_from_spec(_spec)",
            "    _spec.loader.exec_module(_mod)",
            "    return _mod",
        ]
        + call_lines
        + [""]
    )
    driver = lines[: layout.module_start] + inject + lines[layout.module_end:]
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
    # name → bridge-allocated tag, for resolving g.reinforce bond-material
    # references (Option B: the def holds the bond name, the bridge owns
    # the tag). Populated from the name-alias table at build() time.
    name_to_tag:             dict[str, int] = field(default_factory=dict)
    # ADR 0065 Tier 2 — stream per-node masses straight from
    # ``fem.nodes.masses`` at emit instead of materializing one bridge
    # ``MassRecord`` per node (the 7M-object double-store). Set by
    # ``apeSees.mass_from_model()``; consumed in the mass emit paths.
    mass_from_model:         bool = False
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

    def emit(
        self, emitter: Emitter, *, split: bool = False,
    ) -> "int | _SplitLayout":
        """Drive ``emitter`` over the model, returning ``analyze``'s exit value.

        Returns ``0`` if no ``analyze`` was registered (the bridge's
        ``apeSees.analyze`` would have populated one); otherwise the
        last ``analyze`` call's return value.

        When ``split=True`` (ADR 0043 slice 1.1, mode A) the bridge
        drives the module-grouped :meth:`_emit_split` path instead of
        the flat / partitioned paths and returns a :class:`_SplitLayout`
        for the Tcl / Py writers to slice the single buffer into
        per-module fragments + a driver.  ``split`` is honoured only by
        the Tcl / Py emit targets; every other path leaves it ``False``
        and is byte-identical to the pre-0043 behaviour.

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
        # Re-create a TagAllocator seeded with the bridge's existing
        # primitive-tag assignments. Element fan-out + orientation
        # override tags allocate freshly during emit; the seeded
        # counters keep those allocations from colliding with
        # primitive-own tags.
        tags = TagAllocator()
        for prim in self.primitives:
            tags.allocate_for(prim, _kind_of(prim))
        # tag_for already mirrors the assignments; nothing else to do
        # for the seeded primitives.

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
        # emit path (flat / split / partitioned) before any primitive is
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
        # emit path (flat / split / partitioned) is covered before any
        # element is emitted.
        validate_node_ndf_element_compat(self.fem, elements)

        # ADR 0054 (AB-5): ASDAbsorbingBoundary2D has no source-side
        # distortion handling — a skewed quad runs with silently wrong
        # dashpot/stiffness terms.  Fail loud here, once, on every emit
        # path (flat / split / partitioned).
        validate_absorbing_quad_geometry(self.fem, elements)

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

        # 4. Emit non-element / non-transform primitives in topo order.
        pre_element: list[Primitive] = []
        post_element: list[Primitive] = []
        for p in rest:
            if isinstance(p, (Pattern, Recorder)):
                post_element.append(p)
            else:
                pre_element.append(p)

        # ADR 0043 slice 1.1: split (mode A) dispatch.  Routed before
        # the partitioned branch so the split guards (which fail loud
        # on partitioned / staged / initial_stress / non-composed
        # models) own the decision.  The single-file paths below are
        # untouched when ``split`` is ``False``.
        if split:
            return self._emit_split(
                emitter=emitter,
                tags=tags,
                transforms=transforms,
                elements=elements,
                inferred_ndf=effective_ndf,
                pre_element=pre_element,
                post_element=post_element,
                base_resolver=_base_resolver,
            )

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
        emitter_can_partition = getattr(emitter, "supports_partitions", True)
        if not is_partitioned(self.fem) or not emitter_can_partition:
            self._emit_flat(
                emitter=emitter,
                tags=tags,
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
            tags=tags,
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
        tags: TagAllocator,
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
        element_plan: "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]] | None" = None
        fem_eid_to_ops_tag: dict[int, int] | None = None
        if staged:
            element_owner_stage, node_owner_stage = compute_stage_ownership(
                self.stage_records, elements, self.fem,
            )
            element_plan = allocate_element_tags(elements, self.fem, tags)
            fem_eid_to_ops_tag = {
                eid: ele_tag
                for _, sub in element_plan
                for eid, _conn, ele_tag in sub
                if eid != MISSING_FEM_ELEMENT_ID  # ADR 0049: node-pair sentinel
            }
            # Validate that BC pools (global + per-stage) respect the
            # ownership-tier rules — see ``_run_staged_bc_validators``
            # for the H1 / V1 / V2 / V3 / V4 / V5 / V6 surface
            # (Phase SSI-2.D + SSI-2.E).
            self._run_staged_bc_validators(
                node_owner_stage,
                element_owner_stage,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            )

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

        # 4a. Materials / sections / time series / analysis chain
        # (excluding patterns + recorders).  Phase SSI-2.A: skip
        # analysis-chain primitives when stages are declared — each
        # stage re-emits its own chain below.
        for p in pre_element:
            if staged and _is_analysis_chain_primitive(p):
                continue
            tag = self.tag_for[id(p)]
            p._emit(emitter, tag)

        # 5. GeomTransf fan-out.
        overrides = emit_transform_specs(
            transforms=transforms,
            elements=elements,
            emitter=emitter,
            fem=self.fem,
            tags=tags,
            spec_to_own_tag=self.tag_for,
            ndm=self.ndm,
        )

        # 6. Elements.  Phase SSI-2.B: pre-allocate ALL element tags
        # upfront (across global + stage-bound elements) so the
        # fem_eid → ops_tag map is complete before any per-stage
        # emit runs.  Then emit only globally-owned elements here;
        # stage-bound elements emit in their stage's block via
        # ``_emit_stages_flat`` below.
        # Phase SSI-2.E: in the staged case the allocation already
        # happened earlier in this method (so V6 could resolve
        # explicit ``elements=`` targets); reuse the prior plan.
        if element_plan is None:
            element_plan = allocate_element_tags(elements, self.fem, tags)
            fem_eid_to_ops_tag = {
                eid: ele_tag
                for _, sub in element_plan
                for eid, _conn, ele_tag in sub
                if eid != MISSING_FEM_ELEMENT_ID  # ADR 0049: node-pair sentinel
            }
        if fem_eid_to_ops_tag is None:
            raise BridgeError(
                "internal: fem_eid_to_ops_tag not populated — element_plan "
                "allocation must set the map before emit continues."
            )
        for spec, sub in element_plan:
            if id(spec) in element_owner_stage:
                continue  # stage-bound — emit inside the stage block.
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

        # 7. Fixes / masses / regions.
        # ADR 0051: g.loads no longer auto-emit — loads reach the deck
        # only via an explicit ops.pattern.Plain(...).from_model(case)
        # import (expanded in emit_pattern_spec) or bridge-authored
        # p.load(...).  There is no broker-loads auto-emitter.
        self._emit_fixes(emitter)
        self._emit_masses(emitter, inferred_ndf)
        self._emit_regions(emitter, tags)
        self._emit_rayleigh(emitter, tags, fem_eid_to_ops_tag)
        self._emit_damping_attach(emitter, tags, fem_eid_to_ops_tag)
        self._emit_modal_damping(emitter)

        # 7b. MP constraints (Phase 7b, ADR 0022 INV-5).  Records
        # claimed by ``s.embedded`` / ``s.equal_dof`` / ... are
        # SKIPPED here — they emit inside their owning stage's block.
        emit_mp_constraints(
            emitter, self.fem, tags,
            claimed_ids=frozenset(self._claimed_constraint_ids()),
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )

        # 7b'. Embedded reinforcement ties (g.reinforce, ADR 20 / R2b).
        # One LadrunoEmbeddedRebar per rebar node; bond names resolve to
        # tags via the bridge name-alias map.
        emit_reinforce_ties(
            emitter, self.fem, tags, name_to_tag=self.name_to_tag,
        )

        # 7c. Auto-emit constraint handler when MP constraints present.
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
                self.initial_stress_records, emitter, tags,
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
                    tags=tags,
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
                emitter, tags,
                element_plan=element_plan,
                element_owner_stage=element_owner_stage,
                node_owner_stage=node_owner_stage,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                inferred_ndf=inferred_ndf,
                overrides=overrides,
                base_resolver=base_resolver,
            )

    # -- Split (mode A, ADR 0043 slice 1.1) emit path ---------------------

    def _emit_split(
        self,
        *,
        emitter: Emitter,
        tags: TagAllocator,
        transforms: "list[GeomTransf]",
        elements: "list[Element]",
        inferred_ndf: "dict[int, int]",
        pre_element: "list[Primitive]",
        post_element: "list[Primitive]",
        base_resolver: object,
    ) -> "_SplitLayout":
        """Module-grouped emit for ``split="parts"`` (ADR 0043 mode A).

        Drives a single ``emitter`` in three bands:

        * **driver-pre** — definitions (materials / sections / time
          series / beamIntegration) + the analysis chain + the
          ``geomTransf`` fan-out.  Module-agnostic; lands in the driver.
        * **per module** — for each composed module label, that
          module's ``node`` + ``element`` + ``mass`` + intra-part
          ``fix`` lines, emitted contiguously.  The ``[start, end)``
          line span is recorded so the writer carves the fragment file.
        * **driver-post** — regions, broker loads, the cross-module
          MP-constraint interface, the auto constraint handler, then
          patterns + recorders.  All land in the driver.

        Returns the :class:`_SplitLayout` describing the contiguous
        module band + each module's sub-span.

        Fail-loud for slice-1.1 out-of-scope models (partitioned,
        staged, ``initial_stress``, non-composed): the split seam is a
        single-Domain, single-pass, write-only export that does not
        compose with those axes yet.
        """
        from .emitter.tcl import TclEmitter
        from .emitter.py import PyEmitter
        if not isinstance(emitter, (TclEmitter, PyEmitter)):
            raise BridgeError(
                "split='parts': emitter must be a buffered emitter "
                "(TclEmitter or PyEmitter) — only buffered emitters "
                "support the split line-span protocol."
            )
        if is_partitioned(self.fem):
            raise BridgeError(
                "split='parts' does not support partitioned models "
                "(ADR 0043 slice 1.1).  Partition emit (ADR 0027) is an "
                "orthogonal split axis; emit the single-file deck instead."
            )
        if self.stage_records:
            raise BridgeError(
                "split='parts' does not support staged models "
                "(ADR 0043 slice 1.1).  Emit the single-file deck instead."
            )
        if self.initial_stress_records:
            raise BridgeError(
                "split='parts' does not support initial_stress models "
                "(ADR 0043 slice 1.1).  Emit the single-file deck instead."
            )

        node_label_arr = self.fem.nodes.module_label
        elem_label_by_id = self.fem.elements.module_label_by_id()
        if node_label_arr is None or elem_label_by_id is None:
            raise BridgeError(
                "split='parts' requires a composed model (g.compose); "
                "this model carries no per-row module labels.  Emit the "
                "single-file deck instead."
            )

        nid_to_label: dict[int, str] = {
            int(nid): str(lbl)
            for nid, lbl in zip(self.fem.nodes.ids, node_label_arr)
        }
        present = set(nid_to_label.values()) | set(elem_label_by_id.values())
        if not any(lbl != "" for lbl in present):
            raise BridgeError(
                "split='parts' requires a composed model with at least "
                "one composed source module; every row is host-owned "
                "(empty label).  Emit the single-file deck instead."
            )

        # Host ("") first, then composed sources alphabetically — a
        # deterministic source order, stable across runs.
        ordered_labels = sorted(present, key=lambda s: (s != "", s))

        # -- driver-pre: definitions + analysis chain (no nodes —
        #    nodes are per-module).  Mirrors _emit_flat step 4a; staged
        #    skip is unreachable here (gated out above).
        for p in pre_element:
            p._emit(emitter, self.tag_for[id(p)])

        overrides = emit_transform_specs(
            transforms=transforms,
            elements=elements,
            emitter=emitter,
            fem=self.fem,
            tags=tags,
            spec_to_own_tag=self.tag_for,
            ndm=self.ndm,
        )

        element_plan = allocate_element_tags(elements, self.fem, tags)
        fem_eid_to_ops_tag = {
            eid: ele_tag
            for _, sub in element_plan
            for eid, _conn, ele_tag in sub
            if eid != MISSING_FEM_ELEMENT_ID  # ADR 0049: node-pair sentinel
        }

        # Fail loud if any element's module label disagrees with its
        # connectivity nodes' module (red/blue review, Finding B).  A
        # silent host-default ('') for an element whose nodes live in a
        # composed module would route that element into the ``host``
        # fragment — emitted FIRST — referencing nodes not yet defined
        # (they live in a later fragment), producing a deck that fails
        # to load.  ``g.compose`` never produces cross-module element
        # connectivity (every module is offset into a disjoint tag
        # namespace), so this guards against partial / inconsistent
        # module-label metadata, not normal composed models.
        for _spec, sub in element_plan:
            for eid, conn, _ele_tag in sub:
                node_labels = {
                    nid_to_label[int(n)]
                    for n in conn
                    if int(n) in nid_to_label
                }
                if len(node_labels) > 1:
                    raise BridgeError(
                        f"split='parts': element fem_eid={eid} spans "
                        f"modules {sorted(node_labels)} through its "
                        "connectivity. Every element's nodes must belong "
                        "to one module; cross-module coupling must go "
                        "through interface constraints, not shared "
                        "element connectivity."
                    )
                elem_label = elem_label_by_id.get(int(eid), "")
                if node_labels and elem_label not in node_labels:
                    owner = next(iter(node_labels))
                    raise BridgeError(
                        f"split='parts': element fem_eid={eid} carries "
                        f"module label {elem_label!r} but its nodes "
                        f"belong to module {owner!r} (inconsistent / "
                        "partial compose metadata). Refusing to emit a "
                        "fragment that would reference undefined nodes."
                    )

        # -- per-module band.
        module_start = len(emitter.lines())
        modules: list[tuple[str, int, int]] = []
        node_idx = {int(nid): i for i, nid in enumerate(self.fem.nodes.ids)}
        for label in ordered_labels:
            span_start = len(emitter.lines())
            owned_nodes = {
                nid for nid, lbl in nid_to_label.items() if lbl == label
            }
            # Nodes — FEM-id order for a grep-friendly, stable fragment.
            for nid in sorted(owned_nodes):
                xyz = self.fem.nodes.coords[node_idx[nid]]
                _emit_node_with_inferred_ndf(
                    emitter, inferred_ndf, int(nid),
                    (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    self.ndf,
                )
            # Elements owned by this module.
            self._emit_element_subset(
                emitter,
                element_plan=element_plan,
                eid_label=elem_label_by_id,
                label=label,
                overrides=overrides,
                base_resolver=base_resolver,
            )
            # Intra-part fix + mass (reuse the owned-node-set filter).
            self._emit_fixes_partitioned(emitter, owned_nodes)
            self._emit_masses_partitioned(emitter, owned_nodes, inferred_ndf)
            modules.append((label, span_start, len(emitter.lines())))
        module_end = len(emitter.lines())

        # -- driver-post: regions, interface, patterns, recorders.
        # ADR 0051: no broker-loads auto-emit — loads ride from_model.
        self._emit_regions(emitter, tags)
        self._emit_rayleigh(emitter, tags, fem_eid_to_ops_tag)
        self._emit_damping_attach(emitter, tags, fem_eid_to_ops_tag)
        self._emit_modal_damping(emitter)
        emit_mp_constraints(
            emitter, self.fem, tags,
            claimed_ids=frozenset(self._claimed_constraint_ids()),
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )
        emit_reinforce_ties(
            emitter, self.fem, tags, name_to_tag=self.name_to_tag,
        )
        self._maybe_auto_emit_constraint_handler(emitter, pre_element)

        claimed_recorder_ids = self._claimed_recorder_ids()
        for p in post_element:
            tag = self.tag_for[id(p)]
            if isinstance(p, Pattern):
                emit_pattern_spec(
                    p, emitter, tag, self.fem, self.ndf, self.ndm,
                    effective_ndf=inferred_ndf,
                )
            elif isinstance(p, Recorder):
                if id(p) in claimed_recorder_ids:
                    continue
                emit_recorder_spec(
                    p, emitter, tag, self.fem,
                    tags=tags,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

        return _SplitLayout(
            module_start=module_start,
            module_end=module_end,
            modules=modules,
        )

    def _emit_element_subset(
        self,
        emitter: Emitter,
        *,
        element_plan: "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]",
        eid_label: "dict[int, str]",
        label: str,
        overrides: "dict[tuple[int, int], int] | None",
        base_resolver: object,
    ) -> None:
        """Emit ``element`` lines for elements owned by ``label``.

        Factored from the :meth:`_emit_flat` element loop so the split
        path reuses the exact orientation-override resolver dance
        (ADR 0010) without duplicating it.
        """
        for spec, sub in element_plan:
            transf_spec = _build_element_transf(spec)
            rows = [
                row for row in sub
                if eid_label.get(int(row[0]), "") == label
            ]
            if not rows:
                continue
            # Builder-ndf bracket for gated upstream parsers (quad/tri6n)
            # under a mixed-ndf envelope — see open_builder_ndf_bracket.
            bracketed = open_builder_ndf_bracket(
                emitter, spec, ndm=self.ndm, envelope_ndf=self.ndf)
            for eid, node_tags, ele_tag in rows:
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

    def _emit_stages_flat(
        self,
        emitter: Emitter,
        tags: TagAllocator,
        *,
        element_plan: "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]" = (),  # type: ignore[assignment]  # empty tuple is an immutable Sequence[never] default
        element_owner_stage: "dict[int, int]" = {},
        node_owner_stage: "dict[int, int]" = {},
        fem_eid_to_ops_tag: "dict[int, int]" = {},
        inferred_ndf: "dict[int, int]" = {},
        overrides: "dict[tuple[int, int], int] | None" = None,
        base_resolver: object = None,
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
        stage_owned_specs: dict[int, list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]] = {}
        for spec, sub in element_plan:
            spec_sidx = element_owner_stage.get(id(spec))
            if spec_sidx is not None:
                stage_owned_specs.setdefault(spec_sidx, []).append((spec, sub))

        # FEM node-id → coord index lookup (mirrors the
        # _emit_partitioned helper inline).  Cheap to build once.
        node_idx_lookup = {
            int(nid): i for i, nid in enumerate(self.fem.nodes.ids)
        }

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
            owned_specs = stage_owned_specs.get(stage_idx, [])
            for spec, sub in owned_specs:
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
            self._emit_stage_regions(stage, emitter, tags)
            # Stage-bound MP constraints — emit AFTER regions, BEFORE
            # domain_change so the constrained nodes / elements (which
            # emitted at the top of the stage block) are already in
            # the OpenSees domain when the constraint references them.
            if stage.stage_constraint_records:
                emit_stage_mp_constraints(
                    stage.stage_constraint_records, emitter, tags,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
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
                emitter, tags, fem_eid_to_ops_tag,
                records=stage.rayleigh_records,
            )
            self._emit_damping_attach(
                emitter, tags, fem_eid_to_ops_tag,
                records=stage.damping_attach_records,
            )

            # 6. Initial stress.
            if stage.initial_stress_records:
                name_to_param_tags = emit_initial_stress_global(
                    stage.initial_stress_records, emitter, tags,
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
                    emitter, self.fem,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                    tags=tags,
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
                    tags=tags,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

            # Phase SSI-2.E: pre-analyze reset, if requested.  Emits
            # ``reset`` between the recorder declarations and the
            # analyze loop; wipes the Domain back to the last
            # ``setTime`` call.  Rare; kept for parity with the
            # OpenSees surface.
            if stage.pre_analyze_reset:
                emitter.reset()

            # 9. Analyze loop (auto-wraps with hook dispatcher calls).
            # Deck emitters return 0 (their per-increment loops fail
            # loud at RUN time); the live emitter returns the first
            # failing rc — raise rather than run the next stage on a
            # silently partial state.
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

            # 10. Stage close — loadConst + wipeAnalysis + hook clear.
            emitter.stage_close()

    # -- Partitioned emit path (ADR 0027) ---------------------------------

    def _emit_partitioned(
        self,
        *,
        emitter: Emitter,
        tags: TagAllocator,
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
        staged = bool(self.stage_records)

        # g.reinforce (ADR 20 / R2b): partitioned emission of
        # LadrunoEmbeddedRebar ties needs per-rank node-ownership routing
        # (a tie spans the rebar node + its host element's nodes, which
        # may straddle ranks). That routing is deferred; fail loud rather
        # than silently dropping the reinforcement under MPI emit.
        elements_comp = getattr(self.fem, "elements", None)
        if getattr(elements_comp, "reinforce_ties", None):
            raise BridgeError(
                "apeSees: g.reinforce embedded-reinforcement ties are not "
                "yet supported under partitioned (MPI) emit — per-rank "
                "node-ownership routing of LadrunoEmbeddedRebar is deferred "
                "(ADR 20 / R2). Emit the reinforced model single-process "
                "(non-partitioned), or remove the reinforcement for the "
                "partitioned run."
            )

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
        early_element_plan: "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]] | None" = None
        early_fem_eid_to_ops_tag: dict[int, int] | None = None
        if staged:
            element_owner_stage, node_owner_stage = compute_stage_ownership(
                self.stage_records, elements, self.fem,
            )
            early_element_plan = allocate_element_tags(elements, self.fem, tags)
            early_fem_eid_to_ops_tag = {
                eid: ele_tag
                for _, sub in early_element_plan
                for eid, _conn, ele_tag in sub
                if eid != MISSING_FEM_ELEMENT_ID  # ADR 0049: node-pair sentinel
            }
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

        partitions = list(self.fem.partitions)
        node_owners = build_node_partition_owners(self.fem)
        element_owner = build_element_partition_owner(self.fem)

        # -- 1. Pre-element global primitives. ----------------------------
        # Phase SSI-2.C: skip analysis-chain primitives when staged —
        # each stage re-emits its own chain inside its stage block.
        for p in pre_element:
            if staged and _is_analysis_chain_primitive(p):
                continue
            tag = self.tag_for[id(p)]
            p._emit(emitter, tag)

        # GeomTransf fan-out is GLOBAL — orientation-driven per-element
        # vecxz fan-out emits one geomTransf line per distinct vecxz,
        # which must be declared once on every rank that uses it. The
        # simplest correct path is to emit transforms outside any
        # partition_open block so they're available to every rank.
        overrides = emit_transform_specs(
            transforms=transforms,
            elements=elements,
            emitter=emitter,
            fem=self.fem,
            tags=tags,
            spec_to_own_tag=self.tag_for,
            ndm=self.ndm,
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
        if early_element_plan is not None:
            element_plan = early_element_plan
            if early_fem_eid_to_ops_tag is None:
                raise BridgeError(
                    "internal: early_fem_eid_to_ops_tag not set when "
                    "early_element_plan is set — staged path must populate both."
                )
            fem_eid_to_ops_tag = early_fem_eid_to_ops_tag
        else:
            element_plan = allocate_element_tags(elements, self.fem, tags)
            # Global fem-eid → ops-tag map; used by the initial_stress
            # per-rank ``addToParameter`` fan-out to translate the user's
            # FEM element selection into OpenSees element tags (Phase
            # SSI-1).
            fem_eid_to_ops_tag = {}
            for _, sub in element_plan:
                for eid, _conn, ele_tag in sub:
                    if eid == MISSING_FEM_ELEMENT_ID:
                        continue  # ADR 0049: node-pair sentinel — not a fem eid
                    fem_eid_to_ops_tag[int(eid)] = int(ele_tag)

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
                self.initial_stress_records, emitter, tags,
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
        rank_owned_nodes: dict[int, set[int]] = {}
        for idx, rec in enumerate(partitions):
            rank = runtime_rank_from_partition_record(rec, idx)
            rank_owned_nodes[rank] = {int(n) for n in rec.node_ids}

        # ADDITIVE nodal quantities (mass lines, pattern load lines) emit
        # on each node's PRIMARY rank only — OpenSeesMP sums shared-node
        # contributions across ranks, so the every-owner fan-out that is
        # correct for idempotent lines (node / fix / sp) double-counts
        # interface nodes (see primary_owner_map).
        primary_owner = primary_owner_map(node_owners)
        rank_primary_nodes: dict[int, set[int]] = {
            rank: set() for rank in rank_owned_nodes
        }
        for nid, owner_rank in primary_owner.items():
            rank_primary_nodes.setdefault(owner_rank, set()).add(nid)

        # Cross-rank tag identity cache (region tags, ADR 0027
        # §"Tag determinism").
        region_tag_cache: dict[str, int] = {}

        # ADR 0027 INV-4 (MPCO recorder path): for every MPCO recorder
        # that carries a filter, resolve its full filter ids ONCE and
        # allocate the region tag ONCE — both shared across every rank.
        # The per-rank loop below emits one ``region <tag> -node ... -ele ...``
        # line per rank with the rank's owned subset (or skips the rank
        # entirely when the intersection is empty).  After the per-rank
        # loop, the recorder declaration itself emits ONCE globally with
        # ``-R <tag>`` referencing the shared tag — MPCO post-processing
        # then stitches the per-rank ``.mpco`` outputs by tag identity.
        mpco_filter_plan = self._plan_partitioned_mpco_recorders(
            post_element, tags,
        )

        # Pre-compute claimed-constraint ids ONCE before the per-rank
        # loop — the set is rank-independent (every rank's global
        # constraint pass excludes the same records).
        stage_claimed_constraint_ids = frozenset(
            self._claimed_constraint_ids()
        )

        # Rank-independent lookups, hoisted out of the per-rank loop —
        # each is O(model), so rebuilding them per rank made the whole
        # pass O(model × ranks) (measured: dominant emit cost at
        # production rank counts).
        node_idx_lookup = {
            int(nid): i for i, nid in enumerate(self.fem.nodes.ids)
        }
        plan_by_rank = {
            id(spec): bucket_pre_allocated_by_rank(sub, element_owner)
            for spec, sub in element_plan
        }
        fix_plan_by_rank = self._bucket_fix_targets_by_rank(node_owners)
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
                for nid in sorted(int(n) for n in part.node_ids):
                    # Phase SSI-2.C: stage-bound nodes emit inside
                    # their stage's block, not in the global pre-stage
                    # per-rank pass.  S2 (ADR 0033): per-node ndf via
                    # broker.
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

                # 6. Elements — per-rank fan-out (tags pre-allocated;
                # plan pre-bucketed by owner rank so each rank walks
                # only its own elements).
                # Phase SSI-2.C: stage-bound element specs emit inside
                # their stage's block.
                for ele_spec, _pre_alloc in element_plan:
                    if staged and id(ele_spec) in element_owner_stage:
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
                    emitter.mass(_nid, *fit_dof_vector(
                        _m.mass, int(eff_ndf.get(_nid, self.ndf)),
                        kind="mass", node=_nid))

                # 7-bis. Named regions (per-rank intersection — INV-4).
                self._emit_regions_partitioned(
                    emitter, tags, rank_owned_nodes[rank], rank,
                    region_tag_cache,
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
                emit_mp_constraints_partitioned(
                    emitter=emitter,
                    fem=self.fem,
                    partition_rank=rank,
                    node_owners=node_owners,
                    element_owner=element_owner,
                    foreign_node_ndf=int(self.ndf),
                    inferred_ndf=inferred_ndf,
                    tags=tags,
                    claimed_ids=stage_claimed_constraint_ids,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
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
        if not staged:
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
        # injecting the pre-allocated shared region tag via
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
                # re-allocate a tag and re-emit the region globally).
                materialised = plan_entry.materialised_spec
                materialised._emit(emitter, tag)
            else:
                emit_recorder_spec(
                    p, emitter, tag, self.fem,
                    tags=tags,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

        # -- 4. Per-stage emit blocks (Phase SSI-2.C). ----------------
        if staged:
            self._emit_stages_partitioned(
                emitter, tags,
                partitions=partitions,
                element_plan=element_plan,
                plan_by_rank=plan_by_rank,
                rank_owned_nodes=rank_owned_nodes,
                element_owner_stage=element_owner_stage,
                node_owner_stage=node_owner_stage,
                element_owner=element_owner,
                node_owners=node_owners,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                inferred_ndf=inferred_ndf,
                overrides=overrides,
                base_resolver=base_resolver,
            )

    # -- Partitioned staged emit (Phase SSI-2.C) --------------------------

    def _emit_stages_partitioned(
        self,
        emitter: Emitter,
        tags: TagAllocator,
        *,
        partitions: "list[Any]",
        element_plan: "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]",
        plan_by_rank: "dict[int, dict[int, list[tuple[int, tuple[int, ...], int]]]]",
        rank_owned_nodes: "dict[int, set[int]]",
        element_owner_stage: "dict[int, int]",
        node_owner_stage: "dict[int, int]",
        element_owner: "dict[int, int]",
        node_owners: "dict[int, set[int]]",
        fem_eid_to_ops_tag: "dict[int, int]",
        inferred_ndf: "dict[int, int]",
        overrides: "dict[tuple[int, int], int] | None",
        base_resolver: object,
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
            int, list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]
        ] = {}
        for spec, sub in element_plan:
            spec_sidx = element_owner_stage.get(id(spec))
            if spec_sidx is not None:
                stage_owned_specs.setdefault(spec_sidx, []).append((spec, sub))

        node_idx_lookup = {
            int(nid): i for i, nid in enumerate(self.fem.nodes.ids)
        }

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
                stage.remove_sp_records or stage.remove_element_records
            )
            # ADR 0052: stage-bound HOLD supports likewise drive the
            # unified gate (so the per-rank loop + global domain_change
            # fire even when ``s.support`` is the only content in the
            # stage).
            has_supports = bool(stage.support_records)

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

            # Per-stage region tag cache (Phase SSI-2.D PR-C): the
            # SAME tag must survive across every rank that emits its
            # rank-intersection of a stage-bound region.  Fresh per
            # stage so stage-2's regions don't accidentally re-use
            # stage-1's tags.
            stage_region_tag_cache: dict[str, int] = {}

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
            # Build a fem_eid lookup for explicit ops_tag → fem_eid
            # mapping (needed for per-rank routing of remove_element).
            ops_tag_to_fem_eid: dict[int, int] = {
                int(v): int(k)
                for k, v in fem_eid_to_ops_tag.items()
            }

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

            # 2. Per-rank topology + BC emit.  Unified gate (Phase
            # SSI-2.D): open the rank's bracket if it has ANY content
            # (owned nodes, owned elements, or owned BCs).  Phase
            # SSI-2.E widens with removals (remove_sp / remove_element).
            # Empty-bracket ranks are skipped so the Py emitter never
            # produces an empty ``if getPID() == K:`` block.
            if has_activation or has_bcs or has_removals or has_supports:
                for idx, part in enumerate(partitions):
                    rank = runtime_rank_from_partition_record(part, idx)
                    # Owned-node sets are stage-invariant — computed once
                    # by the caller, not rebuilt per stage × rank.
                    rank_owned = rank_owned_nodes[rank]
                    rank_stage_nodes = sorted(
                        rank_owned & owned_nodes_this_stage
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
                    rank_has_region_members = bool(
                        region_target_nodes & rank_owned
                    )
                    rank_has_elements = any(
                        plan_by_rank[id(spec)].get(rank)
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
                    rank_has_content = bool(
                        rank_stage_nodes
                        or rank_has_elements
                        or rank_fix
                        or rank_mass
                        or rank_has_region_members
                        or rank_remove_sp
                        or rank_remove_element
                        or rank_support
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
                        # Phase SSI-2.D PR-B + PR-C: per-rank stage-
                        # bound BCs (fix + mass + region).  Targets
                        # pre-resolved above; per-rank filter via
                        # ``rank_owned`` intersection mirrors the
                        # existing INV-4 fan-out convention.  Region
                        # emit threads the per-stage tag cache so all
                        # contributing ranks emit the SAME tag for
                        # each region name.
                        for fix_rec, nid in rank_fix:
                            emitter.fix(nid, *fix_rec.dofs)
                        for mass_rec, nid in rank_mass:
                            emitter.mass(int(nid), *fit_dof_vector(
                                mass_rec.values,
                                int(inferred_ndf.get(int(nid), self.ndf)),
                                kind="mass", node=int(nid)))
                        self._emit_stage_regions_partitioned(
                            stage, emitter, tags,
                            owned_nodes=rank_owned,
                            region_tag_cache=stage_region_tag_cache,
                        )
                        # Stage-bound MP constraints — per-rank fan-
                        # out using the same replication rules as the
                        # global partitioned constraint pass.  No-op
                        # when the stage has no constraints or none
                        # touch this rank.
                        if stage.stage_constraint_records:
                            emit_stage_mp_constraints_partitioned(
                                stage.stage_constraint_records,
                                emitter=emitter,
                                fem=self.fem,
                                partition_rank=rank,
                                node_owners=node_owners,
                                element_owner=element_owner,
                                foreign_node_ndf=int(self.ndf),
                                inferred_ndf=inferred_ndf,
                                tags=tags,
                                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
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
                emitter, tags, fem_eid_to_ops_tag,
                records=stage.rayleigh_records,
            )
            self._emit_damping_attach(
                emitter, tags, fem_eid_to_ops_tag,
                records=stage.damping_attach_records,
            )

            # 4. Initial-stress globals + per-rank ``addToParameter``.
            if stage.initial_stress_records:
                name_to_param_tags = emit_initial_stress_global(
                    stage.initial_stress_records, emitter, tags,
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
                            emitter, self.fem,
                            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                            tags=tags,
                            element_owner=element_owner,
                            partition_rank=rank,
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
                    rank_owned = {int(n) for n in part.node_ids}
                    rank_primary = {
                        nid for nid in rank_owned
                        if primary_owner.get(nid) == rank
                    }
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
                    tags=tags,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )

            # Phase SSI-2.E: pre-analyze reset (global; emits ``reset``
            # outside any partition block — each rank applies locally).
            if stage.pre_analyze_reset:
                emitter.reset()

            # 7. Analyze loop (auto-wraps with hook dispatcher calls).
            # See the flat path: deck emitters fail loud at RUN time;
            # a live rc != 0 raises here rather than running the next
            # stage on a silently partial state.
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
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
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
        self._validate_remove_element_targets(
            element_owner_stage or {},
            fem_eid_to_ops_tag or {},
        )

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
        fem_eid_to_ops_tag: "dict[int, int]",
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

    def _emit_fixes(self, emitter: Emitter) -> None:
        for rec in self.fix_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                emitter.fix(node_tag, *rec.dofs)

    def _guard_mass_from_model(self, emitter: Emitter) -> bool:
        """Validate a ``mass_from_model`` emit (ADR 0065 Tier 2).

        Returns True when there are snapshot masses to stream, False when
        none (a no-op declaration). Raises ``BridgeError`` if the emitter is
        the H5 archival emitter (masses already persist in ``model.h5`` via
        ``fem.nodes.masses`` — re-streaming would rebuild the 7M-object
        materialization the feature exists to avoid), or if any node carries
        both a snapshot mass and an explicit ``ops.mass`` (additive under MP
        assembly → silent double-count).
        """
        masses = getattr(self.fem.nodes, "masses", None)
        if not masses:
            return False
        if hasattr(emitter, "write_opensees_into"):
            raise BridgeError(
                "mass_from_model() is deck/live-only — nodal masses already "
                "persist in model.h5 via fem.nodes.masses; do not re-stream "
                "them through the H5 emitter."
            )
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
                emitter.mass(nid, *fit_dof_vector(
                    m.mass, int(eff.get(nid, self.ndf)),
                    kind="mass", node=nid))

    def _emit_fixes_partitioned(
        self, emitter: Emitter, owned_nodes: set[int],
    ) -> None:
        """Per-rank fix fan-out (ADR 0027).

        Only emit ``fix`` lines for nodes owned by this rank.  A fix on
        a non-owned node is silently skipped on this rank's block —
        the owning rank handles it.
        """
        for rec in self.fix_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if int(node_tag) in owned_nodes:
                    emitter.fix(node_tag, *rec.dofs)

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
        (The composed ``split='parts'`` caller passes module-exclusive
        sets, which are primary by construction.)
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
                    emitter.mass(nid, *fit_dof_vector(
                        m.mass, int(eff.get(nid, self.ndf)),
                        kind="mass", node=nid))

    def _bucket_fix_targets_by_rank(
        self, node_owners: "dict[int, set[int]]",
    ) -> "dict[int, list[tuple[FixRecord, list[int]]]]":
        """Resolve every global fix record's targets ONCE and bucket by rank.

        Replaces the per-rank :meth:`_emit_fixes_partitioned` pass in the
        base partitioned loop, whose ``_resolve_node_target`` call re-ran
        the full PG → nodes broker expansion once per rank —
        O(records × nodes × ranks).  Fixes replicate on EVERY owning
        rank (idempotent lines, ADR 0027 INV-4 fan-out), so a node
        lands in each of its owners' buckets.  Record order and
        within-record node order are preserved per rank, keeping the
        emitted deck byte-identical.
        """
        out: "dict[int, list[tuple[FixRecord, list[int]]]]" = {}
        for rec in self.fix_records:
            per_rank: "dict[int, list[int]]" = {}
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                nid = int(node_tag)
                for rank in node_owners.get(nid, ()):
                    per_rank.setdefault(rank, []).append(nid)
            for rank, nodes_list in per_rank.items():
                out.setdefault(rank, []).append((rec, nodes_list))
        return out

    def _bucket_mass_targets_by_rank(
        self, primary_owner: "dict[int, int]",
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

    def _emit_rayleigh(
        self,
        emitter: Emitter,
        tags: "TagAllocator | None" = None,
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
        *,
        records: "Sequence[RayleighRecord] | None" = None,
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
        is stiffness-proportional. ``tags`` / ``fem_eid_to_ops_tag`` are
        required only when region records are present (the emit driver always
        supplies them).

        ``records`` overrides the source pool — D5 passes a stage's
        ``rayleigh_records`` so stage-bound Rayleigh emits inside the stage
        block; ``None`` uses the global (non-staged) pool.
        """
        recs = self.rayleigh_records if records is None else tuple(records)
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
        for rec in scoped:
            for name in rec.on:
                ele_tags = self._resolve_damping_on_elements(
                    name, fem_eid_to_ops_tag,
                )
                assert tags is not None  # always supplied when scoped present
                tag = tags.allocate("region")
                emitter.region(
                    tag, "-ele", *ele_tags, "-rayleigh",
                    rec.alpha_m, rec.beta_k, rec.beta_k_init, rec.beta_k_comm,
                )

    def _resolve_damping_on_elements(
        self,
        pg: str,
        fem_eid_to_ops_tag: "dict[int, int] | None",
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
        tags: "TagAllocator | None" = None,
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
        *,
        records: "Sequence[DampingAttachRecord] | None" = None,
    ) -> None:
        """Attach each ``damping`` object to its ``on`` groups (ADR 0053 D3).

        The object itself already emitted its ``damping <Type> $tag`` line in
        the pre-element definition group; here, driver-post, we emit one
        ``region $tag -ele … -damp $dampTag`` per ``on`` physical group, with
        ``-ele`` membership. The object's tag is read back from ``tag_for``.

        ``records`` overrides the source pool — D5 passes a stage's
        ``damping_attach_records`` so the ``region -damp`` attach emits
        inside the stage block (the object definition still emits once,
        pre-element); ``None`` uses the global (non-staged) pool.
        """
        recs = self.damping_attach_records if records is None else tuple(records)
        if not recs:
            return
        for rec in recs:
            damp_tag = self.tag_for[id(rec.prim)]
            for name in rec.on:
                ele_tags = self._resolve_damping_on_elements(
                    name, fem_eid_to_ops_tag,
                )
                assert tags is not None  # always supplied when records present
                tag = tags.allocate("region")
                emitter.region(
                    tag, "-ele", *ele_tags, "-damp", damp_tag,
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

    def _emit_regions(self, emitter: Emitter, tags: TagAllocator) -> None:
        """Fan named-region assignments out into ``emitter.region`` calls.

        Groups records by name in first-seen order; within each name,
        merges all member nodes (across PG resolution and explicit
        tuples) into a single ordered, deduped tuple; allocates one
        ``"region"`` tag per name; emits one ``region $tag -node ...``
        line per name.

        Multiple records with the same name dedupe by node tag; node
        order within the emitted ``-node`` list is first-seen across
        records.
        """
        if not self.region_records:
            return
        by_name: dict[str, list[int]] = {}
        seen_per_name: dict[str, set[int]] = {}
        for rec in self.region_records:
            bucket = by_name.setdefault(rec.name, [])
            seen = seen_per_name.setdefault(rec.name, set())
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if node_tag not in seen:
                    seen.add(node_tag)
                    bucket.append(node_tag)
        for name, node_list in by_name.items():
            if not node_list:
                continue
            tag = tags.allocate("region")
            emitter.region(tag, "-node", *node_list)

    def _emit_stage_regions(
        self,
        stage: "StageRecord",
        emitter: Emitter,
        tags: TagAllocator,
    ) -> None:
        """Single-partition fan-out for one stage's region pool.

        Mirrors :meth:`_emit_regions` shape but operates on
        ``stage.region_records`` rather than ``self.region_records``:
        groups records by ``name`` in first-seen order, merges members
        across same-name records (de-duping by node tag, first-seen
        order preserved), allocates ONE region tag per name, emits one
        ``region $tag -node n1 n2 ...`` per name.

        V3 (Phase SSI-2.D PR-A) guarantees no name collision across
        scopes, so each stage's name set is disjoint from every other
        stage's and from the global pool — tag allocation through the
        shared ``tags.allocate("region")`` counter produces disjoint
        tags by construction.
        """
        if not stage.region_records:
            return
        by_name: dict[str, list[int]] = {}
        seen_per_name: dict[str, set[int]] = {}
        for rec in stage.region_records:
            bucket = by_name.setdefault(rec.name, [])
            seen = seen_per_name.setdefault(rec.name, set())
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if node_tag not in seen:
                    seen.add(node_tag)
                    bucket.append(node_tag)
        for name, node_list in by_name.items():
            if not node_list:
                continue
            tag = tags.allocate("region")
            emitter.region(tag, "-node", *node_list)

    def _emit_stage_regions_partitioned(
        self,
        stage: "StageRecord",
        emitter: Emitter,
        tags: TagAllocator,
        owned_nodes: set[int],
        region_tag_cache: dict[str, int],
    ) -> None:
        """Per-rank fan-out for one stage's region pool (MP path).

        Mirrors :meth:`_emit_regions_partitioned`: same name-merging
        semantics, per-rank ``owned_nodes`` intersection, INV-4 empty-
        intersection skip.  The ``region_tag_cache`` is shared across
        the per-rank loop FOR THIS STAGE so all contributing ranks
        agree on the tag — the cache is keyed by region NAME (within
        this stage), not stage-mangled, because V3 already guarantees
        no cross-stage name collision.

        The caller (``_emit_stages_partitioned``) constructs a FRESH
        cache per stage so stage-2's regions don't accidentally
        adopt stage-1's tags.
        """
        if not stage.region_records:
            return
        by_name: dict[str, list[int]] = {}
        seen_per_name: dict[str, set[int]] = {}
        for rec in stage.region_records:
            bucket = by_name.setdefault(rec.name, [])
            seen = seen_per_name.setdefault(rec.name, set())
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if node_tag not in seen:
                    seen.add(node_tag)
                    bucket.append(node_tag)
        for name, node_list in by_name.items():
            owned_list = [n for n in node_list if int(n) in owned_nodes]
            if not owned_list:
                continue
            tag = region_tag_cache.get(name)
            if tag is None:
                tag = tags.allocate("region")
                region_tag_cache[name] = tag
            emitter.region(tag, "-node", *owned_list)

    def _emit_regions_partitioned(
        self,
        emitter: Emitter,
        tags: TagAllocator,
        owned_nodes: set[int],
        rank: int,
        region_tag_cache: dict[str, int],
    ) -> None:
        """Per-rank named-region fan-out (ADR 0027 §"Regions interaction" /
        INV-4).

        Same name-merging semantics as :meth:`_emit_regions`, but the
        merged node tuple is intersected with ``owned_nodes`` before
        emission.  Empty intersection ⇒ no ``region`` line emitted on
        this rank (INV-4).  The region tag is allocated **once** on the
        FIRST rank that emits the region — ``region_tag_cache`` is
        shared across the per-rank loop so the same tag survives across
        every rank that emits the region (INV-4 §"region tag is the
        same scalar across every rank that does emit").
        """
        if not self.region_records:
            return

        by_name: dict[str, list[int]] = {}
        seen_per_name: dict[str, set[int]] = {}
        for rec in self.region_records:
            bucket = by_name.setdefault(rec.name, [])
            seen = seen_per_name.setdefault(rec.name, set())
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if node_tag not in seen:
                    seen.add(node_tag)
                    bucket.append(node_tag)
        for name, node_list in by_name.items():
            owned_list = [n for n in node_list if int(n) in owned_nodes]
            if not owned_list:
                # INV-4: empty intersection → no region line on this rank.
                continue
            tag = region_tag_cache.get(name)
            if tag is None:
                tag = tags.allocate("region")
                region_tag_cache[name] = tag
            emitter.region(tag, "-node", *owned_list)

    # -- MPCO recorder filter regions (ADR 0027 INV-4 — internal regions) --

    def _plan_partitioned_mpco_recorders(
        self,
        post_element: "list[Primitive]",
        tags: TagAllocator,
    ) -> "dict[int, _MPCOFilterPlan]":
        """Pre-resolve + allocate region tag(s) per region-bearing recorder.

        Returns a dict keyed by ``id(spec)`` carrying the recorder's
        regions (value-channel filter region + any Ladruno ``energy_pg``
        energy region, each with resolved FEM-eid members) plus the
        materialised spec (``_region_tag`` / ``_energy_region_tags``
        populated, ``*_pg`` selectors cleared, ready for ``_emit``).
        Recorders with neither a filter nor an energy region are absent
        from the dict; the caller routes them through the unchanged
        :func:`emit_recorder_spec` global pass.

        The plan is built ONCE before the per-rank loop so:

        1. Each region tag is the SAME scalar across every rank that
           emits its rank-intersection of the region (INV-4: stitching
           by tag identity).
        2. ``_emit`` is bypass-safe — the materialised spec carries the
           shared tags directly, so the global recorder pass after the
           per-rank loop simply forwards ``-R <tag>`` / ``-G energy
           <tag>`` without re-allocating tags or re-emitting regions.
        """
        plan: dict[int, _MPCOFilterPlan] = {}
        for p in post_element:
            # FilterableRecorder = MPCO + Ladruno (ADR 0064): both share
            # the has_filter()/resolve_filter_ids()/-R region machinery,
            # so the per-rank region pass covers both recorder kinds.
            if not isinstance(p, FilterableRecorder):
                continue
            regions: list[_RegionEmit] = []
            materialised: FilterableRecorder = p

            # Value-channel filter region (-R), shared by MPCO + Ladruno.
            if p.has_filter():
                node_ids, elem_ids = p.resolve_filter_ids(self.fem)
                region_tag = tags.allocate("region")
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
                energy_tag = tags.allocate("region")
                regions.append(_RegionEmit(energy_tag, (), e_eids))
                assert isinstance(materialised, Ladruno)
                materialised = replace(
                    materialised,
                    energy_pg=None,
                    _energy_region_tags=(energy_tag,),
                )

            if not regions:
                continue
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
        owned_nodes: set[int],
        element_owner: dict[int, int],
        fem_eid_to_ops_tag: dict[int, int],
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
        owned_nodes: set[int],
        *,
        primary_nodes: set[int],
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
        owned_nodes: set[int],
        *,
        primary_nodes: set[int],
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
        owned_nodes: set[int],
        *,
        primary_nodes: set[int],
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
        load_nodes: set[int],
        sp_nodes: set[int],
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
        load_nodes: set[int],
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
        if not _fem_has_mp_constraints(self.fem):
            return

        # Find any user-declared ConstraintHandler in pre_element.
        # (Constraint handlers go to pre_element because they're not
        # Pattern or Recorder.)
        from .analysis.constraint_handler import (
            Plain as ConstraintsPlain,
            Transformation as ConstraintsTransformation,
        )

        import warnings as _warnings

        # ADR 0068 (INV-4): an enforce="equation" tie emits
        # equationConstraint (EQ_Constraint), which the Transformation
        # handler CANNOT enforce — it would silently drop the tie. When
        # any equation tie is present the handler must be Lagrange
        # (implicit, exact) or the fork LadrunoProjection (explicit).
        has_eq = _fem_has_equation_ties(self.fem)

        declared_handler: "ConstraintHandler | None" = None
        for p in pre_element:
            if isinstance(p, ConstraintHandler):
                declared_handler = p
                break

        if declared_handler is None:
            if has_eq:
                # Equation ties present → auto-emit Lagrange (exact,
                # correct for implicit static/transient). For an EXPLICIT
                # transient run declare ops.constraints.LadrunoProjection()
                # explicitly (Δt-neutral, momentum-conserving); auto
                # implicit/explicit detection is ADR 0068 Open item 1.
                _warnings.warn(
                    "An enforce='equation' tie (EQ_Constraint) is present. "
                    "Auto-emitting 'Lagrange' constraint handler (exact; "
                    "correct for implicit analysis). For an EXPLICIT "
                    "transient run, declare ops.constraints."
                    "LadrunoProjection() (fork) before build() — it is "
                    "Δt-neutral. 'Transformation' cannot enforce "
                    "EQ_Constraint and is never auto-emitted here.",
                    OpenSeesAutoEmitWarning,
                    stacklevel=2,
                )
                emitter.constraints("Lagrange")
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

        # ADR 0068 INV-4: fail loud on Transformation + equation tie —
        # the deck would run but silently drop the EQ_Constraint.
        if has_eq and isinstance(declared_handler, ConstraintsTransformation):
            raise ValueError(
                "Constraint handler 'Transformation' was declared, but the "
                "model has an enforce='equation' tie (EQ_Constraint), which "
                "Transformation cannot enforce — the deck would silently "
                "drop the tie. Declare ops.constraints.Lagrange() "
                "(implicit) or ops.constraints.LadrunoProjection() "
                "(explicit, fork), or switch the tie to enforce='penalty'."
            )
        # Any other explicit handler — no warning, no auto-emit.

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
            "SparseSPD", "FullGeneral", "UmfPack",
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
    rec: "Any", owned_nodes: set[int], fem: "FEMData",
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
    owned_nodes: set[int],
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
    rec: "Any", emitter: Emitter, fem: "FEMData", owned_nodes: set[int],
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

class apeSees:
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
    """

    def __init__(
        self,
        fem: "FEMData",
        *,
        default_orientation: Orientation | None | _UnsetType = _UNSET,
        opensees: "OpenSeesTarget | None" = None,
    ) -> None:
        self._fem: "FEMData" = fem
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
        self._tags = TagAllocator()
        self._ndm: int | None = None
        self._ndf: int | None = None
        self._fix_records: list[FixRecord] = []
        self._mass_records: list[MassRecord] = []
        # ADR 0065 Tier 2 — opt-in: stream per-node masses from the snapshot
        # at emit instead of one bridge MassRecord per node. Set by
        # ``mass_from_model()``; threaded into the BuiltModel.
        self._mass_from_model: bool = False
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
        fork-only ``profiler`` command, and its ``version()`` string.
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
                "openseespy build does not look like the Ladruno fork "
                "(the fork-only 'profiler' command is absent). Launch "
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
            spec = DomainCaptureSpec(opensees=ops)
            spec.nodes(pg="Top", components=["displacement"])
            with ops.domain_capture(spec, path="run.h5") as cap:
                cap.begin_stage("gravity", kind="static")
                for _ in range(n):
                    ops.analyze(1, 1.0)
                    cap.step(t=ops.getTime())
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

        Equivalent to looping ``ops.mass(nodes=[m.node_id], values=m.mass)``
        over every entry in ``fem.nodes.masses`` (e.g. the per-node tributary
        masses produced by ``g.masses.volume(...)``), but **without
        materializing one bridge ``MassRecord`` per node** — the snapshot
        masses are streamed at emit time. On a multi-million-node model this
        avoids a multi-GB resident list and millions of small objects (ADR
        0065 Tier 2). Emits byte-identical deck lines and honours per-node
        ``ndf`` via the same ``fit_dof_vector`` as :meth:`mass`.

        Model-wide declaration (no arguments). May be combined with explicit
        :meth:`mass` calls only on *disjoint* node sets — overlap raises at
        emit (nodal mass is additive under MP assembly). Deck/live emit only;
        the H5 archival emitter rejects it (masses already persist in
        ``model.h5`` via ``fem.nodes.masses``).
        """
        self._mass_from_model = True

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

        # Default time series: Linear scaled by pattern_factor.
        # Folds STKO's ``-fact F`` semantics into the time-series
        # factor instead of an explicit ``-fact`` on the pattern
        # (apeGmsh's Plain pattern primitive doesn't carry one).
        if series is None:
            series = self.timeSeries.Linear(factor=float(pattern_factor))

        # Construct the Plain pattern via the namespace so it gets
        # registered + tagged.
        plain = self.pattern.Plain(series=series)
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
        freshly allocated region tag.  Useful for damping
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
        bm.emit(live_emitter)
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

    def eigen(
        self,
        num_modes: int,
        *,
        solver: str = "-genBandArpack",
    ) -> "EigenResult":
        """Build + emit + run a one-shot ``eigen`` solve via the live emitter.

        Builds a :class:`BuiltModel`, drives a
        :class:`~apeGmsh.opensees.emitter.live.LiveOpsEmitter` end-to-
        end (model + nodes + elements + bcs + mass), then issues the
        single ``eigen`` call and returns an :class:`EigenResult`
        carrying the eigenvalues plus a back-reference to the live
        emitter for lazy mode-shape access.

        Unlike :meth:`analyze`, ``eigen`` does NOT require an analysis
        chain (constraints / numberer / system / test / algorithm /
        integrator / analysis): it only needs the assembled stiffness
        and mass matrices.

        Parameters
        ----------
        num_modes
            Number of modes to compute. Must be ``>= 1``.
        solver
            OpenSees eigen-solver flag, one of ``-genBandArpack``
            (default), ``-symmBandLapack``, ``-fullGenLapack``,
            ``-frequency``, ``-standard``. Passed through verbatim to
            ``ops.eigen(solver, num_modes)``.

        Returns
        -------
        EigenResult
            Carries ``eigenvalues`` (``λ_i = ω_i²``) plus derived
            ``omega`` / ``freq`` / ``periods`` and a
            :meth:`EigenResult.mode_shape` accessor.

        Raises
        ------
        ValueError
            If ``num_modes < 1``.
        NotImplementedError
            If the model has any registered stages — live execution
            of staged models is unsupported (Phase SSI-2.A).
        """
        if num_modes < 1:
            raise ValueError(
                f"apeSees.eigen: num_modes must be >= 1, got {num_modes}."
            )
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.eigen: live execution does not support staged "
                "models (Phase SSI-2.A) "
                f"(got {len(self._stage_records)} stage(s)).  Eigen "
                "analyses are typically run against an unstaged build; "
                "either drop the stage blocks or emit Tcl/Py and run "
                "the eigen command there."
            )

        # Local imports — keep openseespy + numpy out of bridge import
        # time for Tcl/Py/H5-only users.
        from .analysis.eigen import EigenResult
        from .emitter.live import LiveOpsEmitter
        import numpy as np

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen(num_modes, solver=solver)
        return EigenResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            _live=live_emitter,
        )

    def critical_time_step(self) -> float:
        """Query the active explicit integrator's critical time step ``dt_cr``.

        **Fork-only** (Ladruno): builds + emits a throwaway live model
        (like :meth:`eigen`), primes one tiny step to trigger the
        integrator's ``dt_cr`` computation, then returns the usable
        (Noh-Bathe) limit.

        Requires a complete analysis chain with an **explicit**
        integrator constructed with ``cfl=True`` (e.g.
        ``ops.integrator.ExplicitBathe(cfl=True)``), a ``Transient``
        analysis, and **element mass density** (``-rho`` / ``-mass``) —
        the ``dt_cr`` eigensolve uses element mass+stiffness, not
        ``ops.mass`` nodal mass.

        Raises
        ------
        BridgeError
            If the analysis chain is incomplete.
        NotImplementedError
            If the model has registered stages (live execution of staged
            models is unsupported — emit Tcl/Py instead).
        ValueError
            If ``dt_cr`` is not usable (no ``cfl`` flag, a non-explicit
            integrator, or a pure nodal-mass model).
        """
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.critical_time_step: live execution does not "
                "support staged models "
                f"(got {len(self._stage_records)} stage(s)). Emit Tcl/Py "
                "and query criticalTimeStep() there instead."
            )
        self._check_analysis_chain_for_analyze()
        self._check_explicit_solver_compat()

        from .emitter.live import LiveOpsEmitter

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        # Prime one negligible step so the integrator computes dt_cr.
        live_emitter.analyze(steps=1, dt=_DTCR_PRIME_DT)
        return _dtcr_or_raise(live_emitter.critical_time_step())

    def analyze_explicit(
        self,
        *,
        duration: float,
        safety: float = 0.9,
        dt_max: float | None = None,
    ) -> "ExplicitRunResult":
        """Run an explicit transient over ``duration``, auto-sized to ``dt_cr``.

        **Fork-only** (Ladruno) driver implementing the explicit-dynamics
        sub-stepping recipe (ADR D5): build + emit, prime one tiny step,
        query the critical time step, then integrate ``duration`` in
        ``n = ceil(duration / (safety * dt_cr))`` equal sub-steps via a
        single ``analyze(n, duration / n)``.

        .. warning::
           ``dt_cr`` is queried **once**, on the initial stiffness. For a
           model whose tangent *stiffens* mid-run (contact closing,
           geometric / material stiffening) the true critical step shrinks
           and a fixed ``dt`` can go supercritical and diverge. Guard such
           runs by constructing the integrator with ``cfl_abort=True`` (and
           ``recompute=N``) so a recomputed CFL violation aborts the run —
           this method then re-raises that abort as an error rather than
           returning silently. A one-shot run with an unguarded integrator
           emits :class:`OpenSeesExplicitSolverWarning`.

        Parameters
        ----------
        duration
            Total physical time to integrate (``> 0``).
        safety
            Fraction of ``dt_cr`` used as the step (``0 < safety <= 1``;
            default ``0.9``). Scales the value ``criticalTimeStep()``
            returns — do not re-base it on any larger Noh-Bathe bound.
        dt_max
            Optional upper bound on the sub-step — use a step finer than
            stability requires (e.g. for output resolution). ``> 0``.

        Returns
        -------
        ExplicitRunResult
            ``(n, dt, dt_cr)`` — the sub-step count, the step actually used,
            and the queried critical time step.

        Raises
        ------
        BridgeError / NotImplementedError / ValueError
            As for :meth:`critical_time_step`, plus ``ValueError`` for an
            out-of-range ``duration`` / ``safety`` / ``dt_max``.
        RuntimeError
            If the explicit ``analyze`` returns non-zero (divergence, or a
            mid-run ``-cflAbort`` when the integrator is guarded).
        """
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.analyze_explicit: live execution does not support "
                f"staged models (got {len(self._stage_records)} stage(s)). "
                "Emit Tcl/Py and drive the explicit run there instead."
            )
        self._check_analysis_chain_for_analyze()
        self._check_explicit_solver_compat()
        self._warn_if_unguarded_explicit_run()

        from .emitter.live import LiveOpsEmitter

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        # Prime, query dt_cr, then size + run the sub-stepped analysis on
        # the SAME emitter (the prime step's tiny dt is stable).
        live_emitter.analyze(steps=1, dt=_DTCR_PRIME_DT)
        dtcr = _dtcr_or_raise(live_emitter.critical_time_step())
        n, dt = _explicit_substep_count(
            duration, dtcr, safety=safety, dt_max=dt_max,
        )
        ret = int(live_emitter.analyze(steps=n, dt=dt))
        if ret != 0:
            raise RuntimeError(
                f"apeSees.analyze_explicit: explicit run failed (analyze "
                f"returned {ret}) after sizing dt={dt:.3e} from "
                f"dt_cr={dtcr:.3e} over {n} sub-steps. The solution likely "
                "diverged — on a stiffening model the critical step can fall "
                "below dt mid-run. Lower safety=, pass a smaller dt_max=, or "
                "construct the integrator with cfl_abort=True / recompute=N."
            )
        return ExplicitRunResult(n=n, dt=dt, dt_cr=dtcr)

    def tcl(
        self,
        path: str,
        *,
        run: bool = False,
        bin: str | None = None,
        analyze_steps: int | None = None,
        analyze_dt: float | None = None,
        split: bool = False,
        per_rank: bool = False,
    ) -> None:
        """Emit a Tcl deck to ``path``; optionally subprocess OpenSees.

        When ``analyze_steps`` is supplied, an ``analyze`` line is
        appended to the deck after every other primitive — wrapped in
        a hook-dispatching for-loop if any
        :meth:`initial_stress` calls registered step hooks (Phase
        SSI-1).  Without ``analyze_steps``, the emitted deck declares
        the model but does not drive an analysis.

        ``split=True`` (ADR 0043 slice 1.1, mode A) writes a driver
        deck at ``path`` plus one ``parts/<module>.tcl`` fragment per
        composed module (``g.compose``); the driver ``source``s each
        fragment.  The split is canonical — by compose module, no
        free-form carve — and changes only the on-disk layout: the
        default ``split=False`` writes the single self-contained deck,
        byte-identical to the pre-0043 output.  Requires a composed
        model; partitioned / staged / ``initial_stress`` models are not
        supported under ``split``.

        ``per_rank=True`` (ADR 0061) writes a driver deck at ``path``
        plus one ``ranks/rank<K>_<seq>.tcl`` fragment per
        ``if {[getPID] == K} { ... }`` block; the driver guards each
        fragment behind a one-line ``source`` so every MPI rank parses
        only the driver plus its own fragments — O(global + model/np)
        instead of O(model) per rank.  Layout-only: the deck semantics
        (including the single-process rank-0 fallback) are unchanged.
        Requires a partitioned model (``len(fem.partitions) > 1``);
        mutually exclusive with ``split``.
        """
        from .emitter.tcl import TclEmitter

        if split and per_rank:
            raise ValueError(
                "apeSees.tcl: split=True and per_rank=True are mutually "
                "exclusive — split carves by compose module (ADR 0043), "
                "per_rank by partition rank (ADR 0061)."
            )
        bm = self.build()
        emitter = TclEmitter()
        pre_prof, post_prof = self._split_profiler_records()
        if not split:
            bm.emit(emitter)
            for _verb, _vargs in pre_prof:
                emitter.profiler(_verb, *_vargs)
            if analyze_steps is not None:
                emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
            for _verb, _vargs in post_prof:
                emitter.profiler(_verb, *_vargs)
            if per_rank:
                spans = emitter.partition_spans()
                if not spans:
                    raise ValueError(
                        "apeSees.tcl: per_rank=True requires a "
                        "partitioned model (len(fem.partitions) > 1) — "
                        "the emitted deck has no per-rank blocks to "
                        "split out. Partition the mesh "
                        "(g.mesh.partitioning) or drop per_rank."
                    )
                _write_per_rank_tcl(path, emitter.lines(), spans)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    emitter.write_to(f)
        else:
            layout = bm.emit(emitter, split=True)
            for _verb, _vargs in pre_prof:
                emitter.profiler(_verb, *_vargs)
            if analyze_steps is not None:
                emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
            for _verb, _vargs in post_prof:
                emitter.profiler(_verb, *_vargs)
            _write_split_tcl(path, emitter.lines(), layout)  # type: ignore[arg-type]

        if not run:
            return

        binary = _resolve_opensees_binary(bin, self._opensees)
        proc = subprocess.run(
            [binary, path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenSees subprocess returned {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

    def py(
        self,
        path: str,
        *,
        run: bool = False,
        analyze_steps: int | None = None,
        analyze_dt: float | None = None,
        split: bool = False,
        python: str | None = None,
    ) -> None:
        """Emit an openseespy Python deck to ``path``; optionally run it.

        ``analyze_steps`` / ``analyze_dt`` semantics mirror :meth:`tcl`
        (Phase SSI-1).

        ``split=True`` (ADR 0043 slice 1.1, mode A) writes a driver
        script at ``path`` plus one ``parts/<module>.py`` fragment per
        composed module; each fragment exposes ``def build(ops): ...``
        and the driver loads + calls them.  The default ``split=False``
        writes the single self-contained script, byte-identical to the
        pre-0043 output.  Same composed-model requirement as
        :meth:`tcl`.
        """
        from .emitter.py import PyEmitter

        bm = self.build()
        emitter = PyEmitter()
        pre_prof, post_prof = self._split_profiler_records()
        if not split:
            bm.emit(emitter)
            for _verb, _vargs in pre_prof:
                emitter.profiler(_verb, *_vargs)
            if analyze_steps is not None:
                emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
            for _verb, _vargs in post_prof:
                emitter.profiler(_verb, *_vargs)
            with open(path, "w", encoding="utf-8") as f:
                emitter.write_to(f)
        else:
            layout = bm.emit(emitter, split=True)
            for _verb, _vargs in pre_prof:
                emitter.profiler(_verb, *_vargs)
            if analyze_steps is not None:
                emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
            for _verb, _vargs in post_prof:
                emitter.profiler(_verb, *_vargs)
            _write_split_py(path, emitter.lines(), layout)  # type: ignore[arg-type]

        if not run:
            return

        python_bin = _resolve_python_binary(python, self._opensees)
        proc = subprocess.run(
            [python_bin, path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"openseespy subprocess returned {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
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
        _compose_model_h5(
            self._fem, emitter, path,
            model_name=name,
            ndf=int(self._ndf or 0),
            cuts=cuts,
            sweeps=sweeps,
            names=self._name_records(),
            nodes_ndf=_nodes_ndf,
        )

    # -- Registration -----------------------------------------------------

    def _register(self, prim: _P, *, name: str | None = None) -> _P:
        """Add ``prim`` to the bridge, allocate its tag, return it.

        When ``name`` is given, register it as a bridge-side alias for
        ``prim`` so reference kwargs (``series=``, ``material=``,
        ``transf=``, …) can later refer to it by string as well as by
        the returned handle.  Names are unique per bridge instance; a
        duplicate raises ``ValueError`` (fail-loud — no silent
        last-wins).
        """
        kind = _kind_of(prim)
        self._tags.allocate_for(prim, kind)
        self._primitives.append(prim)
        if name is not None:
            existing = self._names.get(name)
            if existing is not None and existing is not prim:
                raise ValueError(
                    f"apeSees: name {name!r} is already registered to a "
                    f"{type(existing).__name__}; names must be unique per "
                    "bridge.  Pick a different name= (or pass the object "
                    "handle directly)."
                )
            self._names[name] = prim
        return prim

    def register(self, prim: _P) -> _P:
        """Register a standalone primitive with the bridge (P11)."""
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
        return BuiltModel(
            primitives=tuple(self._primitives),
            tag_for=tag_for,
            ndm=self._ndm,
            ndf=self._ndf,
            fem=self._fem,
            fix_records=tuple(self._fix_records),
            mass_records=tuple(self._mass_records),
            region_records=tuple(self._region_records),
            ndf_records=tuple(self._ndf_records),
            initial_stress_records=tuple(self._initial_stress_records),
            stage_records=tuple(self._stage_records),
            rayleigh_records=tuple(self._rayleigh_records),
            damping_attach_records=tuple(self._damping_attach_records),
            modal_damping_records=tuple(self._modal_damping_records),
            name_to_tag={
                nm: tag for nm, _kind, tag in self._name_records()
            },
            mass_from_model=self._mass_from_model,
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
        from .analysis.integrator import (
            CentralDifference,
            CentralDifferenceLadruno,
            ExplicitBathe,
            ExplicitBatheLNVD,
            ExplicitDifference,
        )
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

        explicit_types = (
            CentralDifference, ExplicitDifference, ExplicitBathe,
            ExplicitBatheLNVD, CentralDifferenceLadruno,
        )
        integrator = next(
            (p for p in self._primitives if isinstance(p, Integrator)), None,
        )
        if (
            isinstance(integrator, explicit_types)
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
# Explicit critical-time-step (dt_cr) helpers (Ladruno fork).  Pure functions
# — the math + sentinel handling, unit-testable without openseespy.
# ---------------------------------------------------------------------------

# Tiny priming step: triggers the integrator's domainChanged() so dt_cr is
# computed, without meaningfully advancing the solution (negligible vs any
# real integration duration).  Per the explicit-dynamics ADR (D5) recipe.
# NOTE: required on the deployed fork build (605affeb) — criticalTimeStep()
# returns the 0.0 NOT_COMPUTED sentinel until the first step runs.
_DTCR_PRIME_DT = 1e-12


@dataclass(frozen=True, slots=True)
class ExplicitRunResult:
    """Result of :meth:`apeSees.analyze_explicit`.

    ``n`` sub-steps of size ``dt`` tiled the requested duration; ``dt_cr``
    is the critical time step queried on the initial stiffness (``dt`` is
    ``safety * dt_cr``, optionally capped by ``dt_max``).
    """

    n: int
    dt: float
    dt_cr: float


def _dtcr_or_raise(dtcr: float) -> float:
    """Return ``dtcr`` if it is a usable (> 0) critical time step, else raise.

    ``ops.criticalTimeStep()`` sentinels: ``0.0`` not-computed (no prior
    ``analyze`` / ``domainChanged``), ``-1.0`` not-applicable or disabled
    (no ``cfl`` flag, a non-explicit integrator, or a model whose *elements*
    yield no finite estimate — e.g. a pure nodal-mass model).
    """
    if dtcr > 0:
        return dtcr
    if dtcr == 0.0:
        raise ValueError(
            "critical_time_step: dt_cr not computed yet (sentinel 0.0). "
            "An explicit integrator with cfl=True must be registered and "
            "the domain primed — this is handled internally, so seeing 0.0 "
            "means the priming step did not run."
        )
    raise ValueError(
        "critical_time_step: dt_cr not applicable (sentinel -1.0). Requires "
        "an explicit integrator (ExplicitBathe / ExplicitBatheLNVD / "
        "CentralDifferenceLadruno) with cfl=True, and element mass density "
        "(e.g. element -rho / -mass). The dt_cr eigensolve loops ELEMENTS and "
        "uses element mass + stiffness; ops.mass nodal mass is excluded (the "
        "estimate is computed on a different mass operator than the run uses), "
        "so a pure nodal-mass model yields no finite estimate."
    )


def _explicit_substep_count(
    duration: float,
    dtcr: float,
    *,
    safety: float,
    dt_max: float | None,
) -> tuple[int, float]:
    """Stable sub-step count for an explicit run of length ``duration``.

    ``dt_stable = safety * dtcr`` (capped by ``dt_max`` if given);
    ``n = max(1, ceil(duration / dt))``; returns ``(n, duration / n)`` so
    the steps tile ``duration`` exactly.  Assumes ``dtcr > 0`` (the caller
    routes through :func:`_dtcr_or_raise` first).
    """
    import math

    if duration <= 0:
        raise ValueError(
            f"analyze_explicit: duration must be > 0, got {duration}."
        )
    if not (0.0 < safety <= 1.0):
        raise ValueError(
            f"analyze_explicit: safety must be in (0, 1], got {safety}."
        )
    if dt_max is not None and dt_max <= 0:
        raise ValueError(
            f"analyze_explicit: dt_max must be > 0, got {dt_max}."
        )
    # ``safety`` scales the value criticalTimeStep() returns — NOT any
    # larger Noh-Bathe bound. Empirically 0.9 of the returned value runs
    # stable for ExplicitBathe; re-basing it on a bigger bound would erase
    # the margin.
    dt = safety * dtcr
    if dt_max is not None:
        dt = min(dt, dt_max)
    n = max(1, math.ceil(duration / dt))
    return n, duration / n


# ---------------------------------------------------------------------------
# Shared validation for initial_stress (used by ops.initial_stress PULL +
# s.initial_stress PUSH).
# ---------------------------------------------------------------------------


def _build_initial_stress_record(
    *,
    source_label: str,
    name: str,
    pg: str | None,
    elements: "Iterable[int] | None",
    sigma_xx: float,
    sigma_yy: float,
    sigma_zz: float,
    ramp_steps: int,
    lambda_install: float,
) -> "InitialStressRecord":
    """Validate inputs and construct an :class:`InitialStressRecord`.

    Shared by :meth:`apeSees.initial_stress` (the bridge-global PULL
    factory) and :meth:`_StageBuilder.initial_stress` (the stage-bound
    PUSH method).  ``source_label`` prefixes every error message so
    users see which API surface they violated.

    Validation rules (identical to the historical inline checks):

    * Exactly one of ``pg=`` / ``elements=``.
    * ``name`` non-empty and a valid Tcl identifier.
    * ``ramp_steps >= 1``.
    * ``lambda_install in (0, 1]``.
    """
    if (pg is None) == (elements is None):
        raise ValueError(
            f"{source_label}: supply exactly one of pg= or "
            f"elements= (got pg={pg!r}, elements={elements!r})."
        )
    if not name:
        raise ValueError(
            f"{source_label}: name= must be non-empty."
        )
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(
            f"{source_label}: name must be a valid Tcl identifier "
            "(alphanumeric + underscore, not starting with a digit). "
            f"Got name={name!r}."
        )
    if ramp_steps < 1:
        raise ValueError(
            f"{source_label}: ramp_steps must be >= 1, "
            f"got {ramp_steps}."
        )
    if not (0.0 < lambda_install <= 1.0):
        raise ValueError(
            f"{source_label}: lambda_install must be in (0, 1], "
            f"got {lambda_install}."
        )
    elements_tuple = (
        tuple(int(e) for e in elements) if elements is not None else None
    )
    return InitialStressRecord(
        name=str(name),
        pg=pg,
        elements=elements_tuple,
        sigma_xx=float(sigma_xx),
        sigma_yy=float(sigma_yy),
        sigma_zz=float(sigma_zz),
        ramp_steps=int(ramp_steps),
        lambda_install=float(lambda_install),
    )


# ---------------------------------------------------------------------------
# _StageBuilder — context manager backing ops.stage(name) (Phase SSI-2.A)
# ---------------------------------------------------------------------------


class _StageBuilder:
    """Collects per-stage records inside a ``with ops.stage(...) as s:``
    block; emits a :class:`StageRecord` to the bridge on context-exit.

    Lifecycle:

    1. Constructed by :meth:`apeSees.stage` — holds a back-reference
       to the bridge.
    2. Inside the ``with`` block, the user calls ``s.add(record)``,
       ``s.analysis(test=, algorithm=, ...)``, and ``s.run(n=, dt=)``.
    3. On ``__exit__`` (clean exit only), validates that all required
       fields are populated and appends a frozen :class:`StageRecord`
       to ``bridge._stage_records``.  On exception, the stage is
       discarded (caller's exception propagates).

    The builder is NOT a typed primitive — it does not register a tag
    with the bridge.  The records / analysis-chain primitives it
    references ARE registered (independently, via their own
    namespace calls) and therefore appear in
    :attr:`apeSees._primitives` for the topological emit pass.  The
    stage record holds REFERENCES into those primitives, not copies.
    """

    __slots__ = (
        "_bridge", "_name",
        "_initial_stress_records",
        "_activate_absorbing_records",
        "_activated_pgs",
        # Phase SSI-2.D (PR-B + PR-C): stage-bound BC + recorder pools.
        "_fix_records",
        "_mass_records",
        "_region_records",
        "_recorder_specs",
        # ADR 0053 D5: stage-bound damping pools + the ``s.damping`` namespace.
        "_rayleigh_records",
        "_damping_attach_records",
        "damping",
        # ADR 0051 (BL-3): stage-scoped load patterns created via
        # ``s.pattern(series=)``.
        "_pattern_specs",
        # ADR 0052 slice 1: stage-bound HOLD supports (``s.support``) +
        # the lazily-created per-stage ``Plain`` HOLD pattern.
        "_support_records",
        "_support_pattern",
        # Stage-bound constraint pool — populated by s.embedded /
        # s.equal_dof / s.rigid_link / s.tie / s.tied_contact /
        # s.kinematic_coupling / s.node_to_surface.
        "_stage_constraint_records",
        # Phase SSI-2.E: between-stage Domain mutators.  Removal pools
        # emit BEFORE the stage's new fix / mass / region lines; the
        # three scalar fields emit at well-defined slots (set_time +
        # set_creep right after stage_open; pre_analyze_reset right
        # before analyze).
        "_remove_sp_records",
        "_remove_element_records",
        "_set_time",
        "_set_creep_on",
        "_pre_analyze_reset",
        "_test", "_algorithm", "_integrator",
        "_constraints", "_numberer", "_system", "_analysis",
        "_n_increments", "_dt",
        # ADR 0057 Phase A: optional solution-strategy ladder for the
        # stage's analyze loop (set via ``s.run(strategy=)``).
        "_strategy",
        "_analysis_set", "_run_set",
    )

    def __init__(self, bridge: "apeSees", name: str) -> None:
        self._bridge = bridge
        self._name = name
        self._initial_stress_records: list[InitialStressRecord] = []
        self._activate_absorbing_records: list[ActivateAbsorbingRecord] = []
        self._activated_pgs: list[str] = []
        # Phase SSI-2.D PR-B: stage-bound BC pools (fix + mass).
        self._fix_records: list[FixRecord] = []
        self._mass_records: list[MassRecord] = []
        # Phase SSI-2.D PR-C: stage-bound region + recorder pools.
        self._region_records: list[RegionAssignmentRecord] = []
        self._recorder_specs: list[Recorder] = []
        # ADR 0053 D5: stage-bound damping pools + the ``s.damping``
        # namespace (rayleigh + object forms; modal raises — deferred).
        self._rayleigh_records: list[RayleighRecord] = []
        self._damping_attach_records: list[DampingAttachRecord] = []
        self.damping = _StageDampingNS(bridge, self)
        # ADR 0051 (BL-3): stage-scoped load patterns.
        self._pattern_specs: list["Plain"] = []
        # ADR 0052 slice 1: stage-bound HOLD supports + the dedicated
        # per-stage ``Plain`` HOLD pattern (created lazily on the first
        # ``s.support`` call in this stage; None until then).
        self._support_records: list["SupportRecord"] = []
        self._support_pattern: "Plain | None" = None
        # Stage-bound constraint pool — flat list of resolved
        # ConstraintRecord instances.  Emit-time dispatches by
        # isinstance into the six per-kind emit helpers.
        self._stage_constraint_records: list["ConstraintRecord"] = []
        # Phase SSI-2.E: between-stage Domain mutators.
        self._remove_sp_records: list[SPRemovalRecord] = []
        self._remove_element_records: list[ElementRemovalRecord] = []
        self._set_time: float | None = None
        self._set_creep_on: bool | None = None
        self._pre_analyze_reset: bool = False
        self._test: Primitive | None = None
        self._algorithm: Primitive | None = None
        self._integrator: Primitive | None = None
        self._constraints: Primitive | None = None
        self._numberer: Primitive | None = None
        self._system: Primitive | None = None
        self._analysis: Primitive | None = None
        self._n_increments: int = 0
        self._dt: float | None = None
        self._strategy: "Ladder | None" = None
        self._analysis_set: bool = False
        self._run_set: bool = False

    def __enter__(self) -> "_StageBuilder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        # Clear the bridge's open-builder slot regardless of how we
        # exit (exception or clean close) so subsequent
        # ``ops.stage(...)`` calls work.  Set in
        # ``apeSees.stage(name)``.
        self._bridge._open_stage_builder = None
        if exc_type is not None:
            # Don't swallow user's exception; just drop the in-progress
            # stage (no records appended to the bridge).
            return
        # Validate: every stage MUST have a complete analysis chain
        # and a run() call.  Missing-piece errors are caller errors.
        if not self._analysis_set:
            raise ValueError(
                f"Stage {self._name!r}: missing s.analysis(...) — "
                "every stage must declare its analysis chain (test, "
                "algorithm, integrator, constraints, numberer, system, "
                "analysis)."
            )
        if not self._run_set:
            raise ValueError(
                f"Stage {self._name!r}: missing s.run(n_increments=, "
                "dt=) — every stage must declare its analyze loop."
            )
        record = StageRecord(
            name=self._name,
            initial_stress_records=tuple(self._initial_stress_records),
            test=self._test,
            algorithm=self._algorithm,
            integrator=self._integrator,
            constraints=self._constraints,
            numberer=self._numberer,
            system=self._system,
            analysis=self._analysis,
            n_increments=int(self._n_increments),
            dt=None if self._dt is None else float(self._dt),
            strategy=self._strategy,
            activated_pgs=tuple(self._activated_pgs),
            fix_records=tuple(self._fix_records),
            mass_records=tuple(self._mass_records),
            region_records=tuple(self._region_records),
            recorder_specs=tuple(self._recorder_specs),
            rayleigh_records=tuple(self._rayleigh_records),
            damping_attach_records=tuple(self._damping_attach_records),
            pattern_specs=tuple(self._pattern_specs),
            support_records=tuple(self._support_records),
            support_pattern=self._support_pattern,
            stage_constraint_records=tuple(self._stage_constraint_records),
            remove_sp_records=tuple(self._remove_sp_records),
            remove_element_records=tuple(self._remove_element_records),
            set_time=self._set_time,
            set_creep_on=self._set_creep_on,
            pre_analyze_reset=self._pre_analyze_reset,
            activate_absorbing_records=tuple(self._activate_absorbing_records),
        )
        self._bridge._stage_records.append(record)

    # -- Stage population -------------------------------------------------

    def add(self, record: InitialStressRecord) -> None:
        """Bind a previously-registered record to this stage.

        Currently supports :class:`InitialStressRecord` only.  The
        record is removed from the bridge's global ``_initial_stress_records``
        pool (so it does not also emit in the flat-emit zone) and
        added to this stage's pool.

        Passing a record that's NOT in the bridge's global pool
        raises ``ValueError`` — usually a sign of double-``add``ing
        the same record across stages.
        """
        if isinstance(record, InitialStressRecord):
            try:
                self._bridge._initial_stress_records.remove(record)
            except ValueError as e:
                raise ValueError(
                    f"Stage {self._name!r}.add: InitialStressRecord "
                    f"name={record.name!r} not in the bridge's global "
                    "pool — was it already added to a different stage "
                    "or registered through a different bridge instance?"
                ) from e
            self._initial_stress_records.append(record)
            return
        raise TypeError(
            f"Stage {self._name!r}.add: unsupported record type "
            f"{type(record).__name__!r}.  Phase SSI-2.A supports "
            "InitialStressRecord only; future versions may extend."
        )

    def initial_stress(
        self,
        *,
        name: str,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
        sigma_xx: float,
        sigma_yy: float,
        sigma_zz: float,
        ramp_steps: int,
        lambda_install: float = 1.0,
    ) -> "InitialStressRecord":
        """Stage-bound PUSH mirror of :meth:`apeSees.initial_stress`.

        Signature is identical to ``ops.initial_stress(...)``; the
        record is created and appended directly to this stage's pool
        instead of the bridge's global pool — no intermediate
        ``s.add(record)`` step required.

        Equivalent to:

            record = ops.initial_stress(name=..., ...)
            s.add(record)

        but in one call, mirroring the
        ``s.fix`` / ``s.mass`` / ``s.embedded`` PUSH builder methods.
        The existing :meth:`add` PULL path remains supported for
        callers that build records globally and bind them later.

        Per-stage emission ordering is unchanged: this stage's
        initial-stress records emit AFTER the stage's analysis chain
        is established, regardless of which API surface (PUSH vs PULL)
        the record came in through.

        Returns the constructed record so callers can inspect or pass
        it to validators (mirroring ``ops.initial_stress``'s return).
        """
        record = _build_initial_stress_record(
            source_label=f"Stage {self._name!r}.initial_stress",
            name=name, pg=pg, elements=elements,
            sigma_xx=sigma_xx, sigma_yy=sigma_yy, sigma_zz=sigma_zz,
            ramp_steps=ramp_steps, lambda_install=lambda_install,
        )
        self._initial_stress_records.append(record)
        return record

    def activate_absorbing(
        self,
        *,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
    ) -> "ActivateAbsorbingRecord":
        """Flip this stage's absorbing-boundary elements to absorbing mode.

        Emits the one-way ``ASDAbsorbingBoundary`` stage switch (0→1) — the
        OpenSees ``parameter`` / ``addToParameter ... stage`` /
        ``updateParameter 1`` sequence (ADR 0054 AB-3) — once, after this
        stage's analysis chain is established and before its ``analyze`` loop,
        so the gravity stage has already held the boundary by penalty.  Target
        the elements by ``pg`` (typically
        ``AbsorbingSkinResult.skin_all_pg``) or an explicit ``elements`` list;
        exactly one is required.  Per-partition emission is automatic.

        Usage::

            with ops.stage("dynamic") as s:
                s.activate_absorbing(pg=skin.skin_all_pg)
                s.analysis(...); s.run(...)
        """
        if (pg is None) == (elements is None):
            raise ValueError(
                f"Stage {self._name!r}.activate_absorbing: supply exactly one "
                "of pg= or elements=."
            )
        record = ActivateAbsorbingRecord(
            pg=pg,
            elements=(
                tuple(int(e) for e in elements) if elements is not None else None
            ),
        )
        self._activate_absorbing_records.append(record)
        return record

    # -- Stage-bound constraints (CLAIM by name) -------------------------

    def embedded(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``embedded`` constraint records by name for
        this stage.

        Constraint declaration happens at apeGmsh time via
        ``g.constraints.embedded(host_label=..., embedded_label=...,
        name=...)``, which produces resolved ``InterpolationRecord``
        rows on ``fem.elements.constraints``.  This method finds the
        rows matching ``name`` and:

        * appends them to this stage's constraint pool (so they emit
          inside the stage's block, AFTER stage regions and BEFORE the
          stage's ``domain_change``);
        * records their ``id(...)`` in the bridge's
          ``_stage_claimed_constraint_ids`` set so the global MP-
          constraint pass SKIPS them (no double emission).

        The shipped contract is **claim-by-name**, not direct create:
        the kernel resolver runs at apeGmsh / FEMData-build time
        (needs gmsh + parts), so by bridge time the records already
        exist on the FEMData broker.  See ADR 0034 §"Stage-bound
        constraints".

        Parameters
        ----------
        name
            Unique constraint name passed to
            ``g.constraints.embedded(name=...)`` at apeGmsh time.

        Returns
        -------
        tuple[ConstraintRecord, ...]
            The claimed records, in registration order on the broker.

        Raises
        ------
        ValueError
            * No record on ``fem.elements.constraints`` matches
              ``name`` (typo, or missing ``name=`` at declaration).
            * The matched record is already claimed by a different
              stage (double-claim).
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.embedded",
            kind="embedded",
            scope="elements",
        )

    def tie(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``tie`` constraint records by name (claim-by-
        name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.tie",
            kind="tie",
            scope="elements",
        )

    def distributing(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``distributing`` constraint records by name
        (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.distributing",
            kind="distributing",
            scope="elements",
        )

    def equal_dof(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``equal_dof`` constraint records by name
        (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.equal_dof",
            kind="equal_dof",
            scope="nodes",
        )

    def rigid_link(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved rigid-link constraint records by name.

        Spans ``rigid_beam``, ``rigid_rod``, and ``rigid_body`` since
        ``g.constraints.rigid_link(...)`` may produce any of the three
        depending on the user's flag (see ConstraintsComposite).
        Claim-by-name semantics; see :meth:`embedded` for the contract.
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.rigid_link",
            kind=frozenset({"rigid_beam", "rigid_rod", "rigid_body"}),
            scope="nodes",
        )

    def rigid_diaphragm(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``rigid_diaphragm`` constraint records by
        name (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.rigid_diaphragm",
            kind="rigid_diaphragm",
            scope="nodes",
        )

    def kinematic_coupling(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``kinematic_coupling`` constraint records by
        name (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.kinematic_coupling",
            kind="kinematic_coupling",
            scope="nodes",
        )

    def node_to_surface(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``node_to_surface`` constraint records by
        name (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.node_to_surface",
            kind="node_to_surface",
            scope="nodes",
        )

    def node_to_surface_spring(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``node_to_surface_spring`` constraint records
        by name (claim-by-name; see :meth:`embedded` for the contract).
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.node_to_surface_spring",
            kind="node_to_surface_spring",
            scope="nodes",
        )

    def tied_contact(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim a resolved ``tied_contact`` surface coupling by name for
        this stage (claim-by-name; see :meth:`embedded` for the contract).

        ``g.constraints.tied_contact(master_label=..., slave_label=...,
        name=...)`` resolves at apeGmsh time to a single
        :class:`SurfaceCouplingRecord` on ``fem.elements.constraints``
        whose ``slave_records`` hold one ``InterpolationRecord`` per slave
        node.  Claiming it routes the whole coupling into this stage's
        block: the stage adapter expands the nested slaves on emit, and
        :meth:`_claimed_constraint_ids` registers those same slave ids so
        the global surface-coupling pass (which sees the expanded slaves,
        not the outer record) correctly skips them — no double emission.

        Both the flat and partitioned stage emit paths handle the
        expansion (``_StageConstraintAdapter.interpolations`` /
        ``_emit_surface_couplings_for_rank``).
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.tied_contact",
            kind="tied_contact",
            scope="elements",
        )

    # NOTE: s.mortar is intentionally out of scope — mortar is not
    # implemented kernel-side (``g.constraints.mortar`` raises
    # NotImplementedError at apeGmsh time), so there is no resolved
    # SurfaceCouplingRecord to claim.

    # -- Internal claim helper -------------------------------------------

    def _claim_constraints_by_name(
        self,
        *,
        name: str,
        method_label: str,
        kind: "str | frozenset[str] | None",
        scope: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Walk the FEMData constraint broker, claim matches by name.

        Parameters
        ----------
        name
            Constraint name to match (from
            ``g.constraints.<kind>(..., name=name)``).
        method_label
            Display name for error messages (e.g. ``"s.embedded"``).
        kind
            If a ``str``, filter records by ``rec.kind == kind``.
            If a ``frozenset[str]``, filter by membership (used for
            ``s.rigid_link`` which spans rigid_beam / rigid_rod /
            rigid_body).  ``None`` skips the kind check.
        scope
            ``"elements"`` (walk ``fem.elements.constraints``) or
            ``"nodes"`` (walk ``fem.nodes.constraints``).
        """
        if not name:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: name= must be "
                "non-empty (claim-by-name requires the user to have "
                "passed a unique name= to g.constraints.X at apeGmsh "
                "time)."
            )
        fem = self._bridge._fem
        if scope == "elements":
            container = getattr(
                getattr(fem, "elements", None), "constraints", None,
            )
            scope_attr = "fem.elements.constraints"
        elif scope == "nodes":
            container = getattr(
                getattr(fem, "nodes", None), "constraints", None,
            )
            scope_attr = "fem.nodes.constraints"
        else:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: invalid scope "
                f"{scope!r} (internal bug)."
            )
        if container is None:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: {scope_attr} "
                f"is None on this FEMData — no constraint broker to "
                f"claim from."
            )

        kind_check: "Callable[[object], bool] | None"
        kind_label: "str | None"
        if isinstance(kind, str):
            kind_str = kind

            def kind_check(k: object) -> bool:
                return k == kind_str

            kind_label = repr(kind)
        elif kind is not None:
            kind_set = kind

            def kind_check(k: object) -> bool:
                return k in kind_set

            kind_label = repr(sorted(kind))
        else:
            kind_check = None
            kind_label = None
        matched: list["ConstraintRecord"] = []
        for rec in container:
            if getattr(rec, "name", None) != name:
                continue
            if kind_check is not None and not kind_check(
                getattr(rec, "kind", None)
            ):
                continue
            matched.append(rec)
        if not matched:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: no resolved "
                f"constraint records found with name={name!r}"
                + (f" and kind in {kind_label}" if kind_label else "")
                + f" on {scope_attr}. Did you pass name={name!r} to "
                "the matching g.constraints.X(...) call at apeGmsh "
                "time?"
            )
        already = self._bridge._stage_claimed_constraint_ids
        for rec in matched:
            if id(rec) in already:
                raise ValueError(
                    f"Stage {self._name!r}.{method_label}: constraint "
                    f"name={name!r} is already claimed by another "
                    "stage — each named constraint may bind to at "
                    "most one stage."
                )
        for rec in matched:
            already.add(id(rec))
            self._stage_constraint_records.append(rec)
        return tuple(matched)

    def fix(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Apply homogeneous SP constraints (``fix``) bound to this stage.

        Signature mirrors :meth:`apeSees.fix` verbatim: exactly one of
        ``pg`` / ``nodes`` must be supplied; ``nodes`` accepts a mix
        of plain integer tags and :class:`Node` instances; ``dofs``
        is a tuple of 0/1 flags per ndf.  The bridge expands ``pg``
        to a per-node fan-out at emit time, same as the global
        :meth:`apeSees.fix` path.

        Stage-bound fix lines emit **inside this stage's block**, after
        the stage's topology (nodes + elements activated via
        :meth:`activate`) and before the stage's initial-stress
        records.  Per-rank fan-out under MP follows the same INV-4
        rules as the global path: each rank only emits ``fix`` for
        nodes it owns.

        Validators V1 / V2 (Phase SSI-2.D PR-A) gate this at build
        time — see :meth:`apeSees._run_staged_bc_validators`.

        Reference frame — absolute (ANCHOR).  ``fix`` is a homogeneous
        single-point constraint at value 0, so adding it mid-stage to a
        node that has already drifted to ``u = d`` drives that DOF back
        toward its ``t = 0`` reference position on the next ``analyze``:
        the node is *moved*, and the attached elements pick up the
        corresponding (physical, not spurious) forces.  Use this when
        you genuinely want the DOF returned to the undeformed position.
        To instead *hold* the node at its current deformed position with
        zero initial force, use :meth:`support` (ADR 0052); ``fix``
        cannot express that, as a homogeneous SP has no value lever.

        Parameters
        ----------
        pg
            Physical group whose nodes receive the fix.  XOR with
            ``nodes``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg``.
        dofs
            ``ndf``-length tuple of 0/1 flags — ``1`` means fix that
            DOF, ``0`` leaves it free.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``nodes`` is supplied.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.fix: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._fix_records.append(
            FixRecord(pg=pg, nodes=nodes_tuple, dofs=tuple(dofs)),
        )

    def support(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Install a stage-bound support that HOLDS the current deformed
        position with zero initial force (ADR 0052).

        The staged-construction counterpart to :meth:`fix`.  Where
        ``fix`` is *absolute* — it drives the DOF back to its ``t = 0``
        reference position — ``support`` *holds* the DOF at wherever it
        has drifted to by the start of this stage.  Each flagged DOF
        emits, inside this stage's dedicated constant pattern::

            sp <node> <dof> [nodeDisp <node> <dof>] -const

        The ``nodeDisp`` value is captured **at runtime** in the emitted
        deck (after the prior stage's ``analyze`` + ``loadConst``), so
        the support is satisfied the instant it is added — zero residual,
        zero jump, zero spurious force.  ``-const`` pins the value so it
        is never scaled by a load factor.

        Signature mirrors :meth:`fix`: exactly one of ``pg`` / ``nodes``;
        ``dofs`` is an ``ndf``-length tuple of 0/1 flags (``1`` = hold
        that DOF).  No value is supplied — it is read from the model at
        runtime.  Emits inside this stage's block (BC region, before
        ``domain_change``) into a per-stage ``Plain`` pattern bound to a
        single shared ``Constant`` series; the pattern is claimed so the
        global / stage-load-pattern passes never double-emit it.

        Reference frame — deformed (HOLD).  Use this for the usual
        staged-construction intent: you add a support to hold what is
        already there.  To instead return a DOF to its ``t = 0`` position
        (a physical restoring force, e.g. releasing then re-anchoring),
        use :meth:`fix`.

        Transient caveat — momentum-kill, not value-jump.  A HOLD support
        introduces no displacement jump (the value equals the current
        position), so it is exactly zero-force in a static stage.  In a
        *transient* stage, however, rigidly pinning a moving DOF cuts its
        velocity in one step — the reaction absorbs the momentum and the
        kinetic energy is removed discontinuously (an impulse).  This is
        **not** fixable by ramping (there is no value trajectory to
        ramp).  If that impulse matters, install the support at a
        quiescent instant, or model the support as a stiff
        spring + dashpot (a ``zeroLength`` element) so the momentum
        bleeds off instead of being cut.

        Validators V1 / V2 gate this at build time: a ``support`` and a
        ``fix`` (or two ``support`` directives) on the same ``(node,
        DOF)`` across tiers is refused — a DOF can carry only one
        single-point constraint.  ``s.remove_sp`` on that target clears
        the registration so a same-stage re-support is allowed.

        Parameters
        ----------
        pg
            Physical group whose nodes are held.  XOR with ``nodes``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg``.
        dofs
            ``ndf``-length tuple of 0/1 flags — ``1`` means hold that
            DOF at its current displacement, ``0`` leaves it free.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``nodes`` is supplied.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.support: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        # Lazily create the shared Constant series (once across all
        # stages) and this stage's dedicated Plain HOLD pattern (once
        # per stage), then claim the pattern so neither the global
        # post-element pattern pass nor the 7b stage-load-pattern pass
        # double-emits it — the dedicated HOLD block drives its emit.
        if self._support_pattern is None:
            if self._bridge._hold_series is None:
                self._bridge._hold_series = self._bridge.timeSeries.Constant(
                    factor=1.0,
                )
            self._support_pattern = self._bridge.pattern.Plain(
                series=self._bridge._hold_series,
            )
            self._bridge._stage_claimed_pattern_ids.add(
                id(self._support_pattern),
            )
        self._support_records.append(
            SupportRecord(pg=pg, nodes=nodes_tuple, dofs=tuple(dofs)),
        )

    def mass(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        values: tuple[float, ...],
        overwrite: bool = False,
    ) -> None:
        """Attach lumped nodal mass bound to this stage.

        Signature mirrors :meth:`apeSees.mass` verbatim, plus the
        Phase SSI-2.E ``overwrite=`` flag.  Stage-bound mass lines
        emit alongside stage-bound fix lines (see :meth:`fix` for the
        emit-position rationale).

        OpenSees ``setMass`` silently OVERWRITES a node's mass on
        repeated calls.  Validator V2 (Phase SSI-2.D PR-A) refuses
        the same node receiving mass in more than one tier (global +
        any stage, or stage A + stage B) at build time so the
        physics change is not silent.

        Pass ``overwrite=True`` to opt out of V2 for this record only —
        the user is acknowledging the OpenSees ``setMass`` overwrite is
        intentional (e.g. swapping a temporary construction mass for a
        permanent one between stages).  The emitted ``mass`` line is
        byte-identical with or without the flag; the difference is
        purely a build-time validator-bypass marker.

        Parameters
        ----------
        pg
            Physical group whose nodes receive the mass.  XOR with
            ``nodes``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg``.
        values
            ``ndf``-length tuple of mass values per DOF.
        overwrite
            When ``True``, V2 skips the cross-tier duplicate-mass
            check for this record.  Defaults to ``False`` (V2 active).

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``nodes`` is supplied.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.mass: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._mass_records.append(
            MassRecord(
                pg=pg, nodes=nodes_tuple, values=tuple(values),
                overwrite=bool(overwrite),
            ),
        )

    # -- Phase SSI-2.E: between-stage Domain mutators --------------------

    def remove_sp(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Release prior-tier SP constraints on a set of nodes / DOFs
        within this stage (Phase SSI-2.E).

        Stage-bound only.  The emitted ``remove sp $node $dof`` lines
        fire BEFORE the stage's new ``fix`` / ``mass`` / ``region``
        lines, so a stage can release a prior-stage support and then
        re-fix the same DOF with a new value in the same stage block.

        Validator V5 (Phase SSI-2.E) refuses targets whose SP was not
        declared in an earlier scope (global pool OR strictly-earlier
        stage's ``s.fix`` pool), or that was already removed by an
        earlier stage's ``s.remove_sp``.

        Parameters
        ----------
        pg
            Physical group whose nodes have SPs released.  XOR with
            ``nodes``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg``.
        dofs
            DOF indices to release per node.  Per OpenSees convention,
            DOFs are 1-based — ``(1, 2, 3)`` releases the first three
            DOFs at every resolved node.  Unlike :meth:`fix`, these
            are DOF *indices* (one ``remove sp`` line per index), not
            a fixity flag vector.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``nodes`` is supplied, or
            if ``dofs`` is empty.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.remove_sp: supply exactly one "
                f"of pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        dofs_tuple = tuple(int(d) for d in dofs)
        if not dofs_tuple:
            raise ValueError(
                f"Stage {self._name!r}.remove_sp: dofs= must contain "
                "at least one DOF index."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._remove_sp_records.append(
            SPRemovalRecord(
                pg=pg, nodes=nodes_tuple, dofs=dofs_tuple,
            ),
        )

    def remove_bc(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Release prior-tier boundary conditions on a set of nodes /
        DOFs within this stage — the ``g.constraints.bc``-reading alias
        of :meth:`remove_sp` (ADR 0051 §8).

        Verbatim delegate: ``s.remove_bc(...)`` and ``s.remove_sp(...)``
        produce identical :class:`SPRemovalRecord` rows and identical
        ``remove sp $node $dof`` deck lines.  ``remove_bc`` reads more
        naturally when the released constraint was declared with
        ``g.constraints.bc(...)``; ``remove_sp`` is retained because
        shipped decks and tests reference it.

        DOF convention (unchanged, easy to trip over): ``dofs=`` here are
        **1-based DOF indices** — one ``remove sp`` line per index — NOT
        the 0/1 fixity flag vector that ``ops.fix`` / ``s.fix`` take.
        ``(1, 2, 3)`` releases the first three DOFs at every resolved
        node.

        See :meth:`remove_sp` for the full parameter / validator (V5)
        contract.
        """
        self.remove_sp(pg=pg, nodes=nodes, dofs=dofs)

    def remove_element(
        self,
        *,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
    ) -> None:
        """Drop elements from the Domain mid-analysis within this stage
        (Phase SSI-2.E).

        Stage-bound only.  The emitted ``remove element $tag`` lines
        fire BEFORE the stage's new ``fix`` / ``mass`` / ``region`` /
        MP-constraint lines so the same stage can release legacy
        elements and immediately bind new BCs to the survivors.

        Element nodes are NOT removed — they remain in the Domain and
        may continue to carry SP / mass / load declarations from other
        tiers.  Use :meth:`remove_sp` separately if you also want to
        drop SP constraints from those orphaned nodes.

        Validator V6 (Phase SSI-2.E) refuses targets that were not
        previously emitted in an earlier scope (globally emitted OR
        activated by a strictly-earlier stage's ``s.activate(pgs=)``),
        or that were already removed by an earlier stage.

        Parameters
        ----------
        pg
            Physical group whose elements are removed.  XOR with
            ``elements``.
        elements
            Explicit list of FEM element ids (NOT OpenSees ops tags)
            — matches the :class:`recorder.Element` convention.  The
            bridge translates FEM eids to OpenSees ops tags via the
            pre-allocated ``fem_eid_to_ops_tag`` map at emit time, so
            the emitted ``remove element $tag`` line carries the
            OpenSees tag the rest of the deck uses.  XOR with ``pg``.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``elements`` is supplied.
        """
        if (pg is None) == (elements is None):
            raise ValueError(
                f"Stage {self._name!r}.remove_element: supply exactly "
                f"one of pg= or elements= (got pg={pg!r}, "
                f"elements={elements!r})."
            )
        elements_tuple = (
            None if elements is None else tuple(int(e) for e in elements)
        )
        self._remove_element_records.append(
            ElementRemovalRecord(pg=pg, elements=elements_tuple),
        )

    def set_time(self, t: float) -> None:
        """Override the stage's starting pseudo-time (Phase SSI-2.E).

        Emits ``setTime $t`` at the top of the stage block — right
        after ``stage_open``.  Overrides the ``loadConst -time 0.0``
        reset that the previous stage's ``stage_close`` emitted.
        Useful when the dynamic clock of a transient stage should
        begin at a non-zero value (e.g. continuing simulated time
        across multi-record ground motion runs).

        Idempotent at the call-site level: a second ``s.set_time(...)``
        in the same stage overwrites the prior value (only the last
        wins).  Per OpenSees semantics, ``setTime`` does NOT reset
        committed state — node displacements / element forces survive.
        """
        self._set_time = float(t)

    def set_creep(self, on: bool) -> None:
        """Toggle creep for time-dependent concrete materials in this
        stage (Phase SSI-2.E).

        Emits ``setCreep 1`` or ``setCreep 0`` near the top of the
        stage block (after ``set_time``).  Sticky on the OpenSees side
        — apeSees does NOT auto-reset between stages.  Re-assert the
        desired state per stage if you need it scoped.

        A second call in the same stage overwrites the prior value.
        """
        self._set_creep_on = bool(on)

    def reset(self) -> None:
        """Request a ``reset`` command right before this stage's
        ``analyze`` (Phase SSI-2.E).

        Emits the bare OpenSees ``reset`` command, which wipes the
        Domain state back to the last ``setTime`` call.  Rarely
        needed — kept for parity with the OpenSees surface so unusual
        workflows don't have to drop to raw Tcl.

        Idempotent: multiple ``s.reset()`` calls on the same stage
        produce a single emitted ``reset`` line.
        """
        self._pre_analyze_reset = True

    def region(
        self,
        *,
        name: str,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
    ) -> None:
        """Assign nodes to a named OpenSees Region bound to this stage.

        Signature mirrors :meth:`apeSees.region` verbatim.  Stage-bound
        regions emit **inside this stage's block**, alongside the
        stage's ``fix`` / ``mass`` lines and before the
        ``domain_change`` barrier.

        Under MP the per-stage region tag is allocated once on the
        first rank that contributes members, then re-used across every
        rank that owns members of the same region — same INV-4
        convention the global path uses, but cached per-stage so two
        stages with regions named ``"foo"`` get distinct tags.
        Validator V3 (Phase SSI-2.D PR-A) refuses same-``name``
        regions across scopes, so within any single stage every
        ``name=`` resolves to one unique tag.

        Parameters
        ----------
        name
            Region label.  Must be unique across global + every
            stage's region pool (V3).  Mangle the label to make scope
            explicit when the same conceptual region appears in
            multiple stages (e.g. ``lining_rayleigh_stage2``).
        pg
            Physical group whose nodes join the region.  XOR with
            ``nodes``.
        nodes
            Explicit node list.  XOR with ``pg``.

        Raises
        ------
        ValueError
            If ``name`` is empty or if both / neither of
            ``pg`` / ``nodes`` is supplied.
        """
        if not name:
            raise ValueError(
                f"Stage {self._name!r}.region: name= must be non-empty."
            )
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.region: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._region_records.append(
            RegionAssignmentRecord(
                name=str(name), pg=pg, nodes=nodes_tuple,
            ),
        )

    def recorder(self, spec: Recorder) -> None:
        """Bind a previously-registered recorder to this stage (PULL).

        ``spec`` is a :class:`Recorder` constructed and registered via
        ``ops.recorder.Node(...)`` / ``ops.recorder.Element(...)`` /
        ``ops.recorder.MPCO(...)``.  The recorder keeps its allocated
        tag and stays in the bridge's ``_primitives`` list, but its
        ``id(...)`` lands in
        :attr:`apeSees._stage_claimed_recorder_ids` so the global
        post-element emit loop SKIPS it — the stage's emit pass
        invokes :func:`emit_recorder_spec` inside the stage block
        instead, AFTER the stage's region declarations and analysis
        chain so the recorder sees fully-populated regions when
        OpenSees parses the ``recorder`` line.

        Mirrors the :meth:`add` PULL semantics for
        :class:`InitialStressRecord`; stage-bound recorders ARE
        bridge primitives with registration side effects (tag
        allocation), so PULL is the natural shape.

        Parameters
        ----------
        spec
            A :class:`Recorder` instance already registered with the
            bridge.

        Raises
        ------
        TypeError
            If ``spec`` is not a :class:`Recorder` instance.
        ValueError
            If ``spec`` is not in the bridge's ``_primitives`` (never
            registered through ``ops.recorder.X(...)``) or has
            already been claimed by another stage (double-add).
        """
        if not isinstance(spec, Recorder):
            raise TypeError(
                f"Stage {self._name!r}.recorder: expected a Recorder "
                f"instance (constructed via ops.recorder.Node / "
                f"Element / MPCO); got {type(spec).__name__!r}."
            )
        if id(spec) in self._bridge._stage_claimed_recorder_ids:
            raise ValueError(
                f"Stage {self._name!r}.recorder: recorder spec already "
                "claimed by another stage — each recorder may bind to "
                "at most one stage."
            )
        if spec not in self._bridge._primitives:
            raise ValueError(
                f"Stage {self._name!r}.recorder: recorder spec not in "
                "the bridge's _primitives — was it registered through "
                "this bridge's ``ops.recorder.X(...)`` namespace?"
            )
        self._bridge._stage_claimed_recorder_ids.add(id(spec))
        self._recorder_specs.append(spec)

    def pattern(
        self,
        *,
        series: "TimeSeries | str",
        name: str | None = None,
    ) -> "Plain":
        """Create a stage-scoped ``Plain`` load pattern (ADR 0051 §6).

        Returns a stage-owned :class:`~apeGmsh.opensees.pattern.pattern.Plain`
        that is **both** a typed primitive (registered with the bridge,
        so it gets a tag) **and** a context manager — open it with a
        ``with`` block and call ``p.load(...)`` / ``p.sp(...)`` /
        ``p.from_model(case)`` to populate it, exactly like the global
        ``ops.pattern.Plain(...)``::

            with ops.stage(name="push") as s:
                ts = ops.timeSeries.Linear()
                with s.pattern(series=ts) as p:
                    p.from_model("live")
                    p.load(node=99, forces=(50.0, 0.0, 0.0))
                s.analysis(...)
                s.run(n_increments=10, dt=0.1)

        The pattern emits **inside this stage's block** — after the
        stage's analysis chain and before its ``analyze`` loop — so its
        loads / prescribed displacements drive only this stage and are
        frozen as the permanent baseline by the stage's
        ``stage_close`` ``loadConst``.  It is claimed via
        :attr:`apeSees._stage_claimed_pattern_ids` so the global
        post-element pattern pass SKIPS it (no double emission), exactly
        mirroring how :meth:`recorder` claims recorders.

        The existing **global** ``ops.pattern.Plain(...)`` remains the
        non-staged path; per ADR 0051 §5 a model may not mix a global
        pattern with stages — that no-mixing guard lands in BL-4.

        Parameters
        ----------
        series
            The :class:`~apeGmsh.opensees._internal.types.TimeSeries`
            scaling this pattern's loads — a handle, or the ``name=``
            a series was registered under (dual-mode, same as
            ``ops.pattern.Plain``).
        name
            Optional bridge-side alias for the pattern (see
            ``ops.pattern.Plain``).
        """
        # Delegate construction to the pattern namespace so the series
        # name resolution + registration + tag allocation are identical
        # to the global ``ops.pattern.Plain(...)`` path; then claim it
        # for this stage.
        plain = self._bridge.pattern.Plain(series=series, name=name)
        self._bridge._stage_claimed_pattern_ids.add(id(plain))
        self._pattern_specs.append(plain)
        return plain

    def activate(self, *, pgs: "Iterable[str]") -> None:
        """Mark element PGs as activated by this stage (Phase SSI-2.B).

        Elements whose ``pg=`` matches any activated PG emit their
        ``node`` + ``element`` commands **inside this stage's block**
        (between ``stage_open`` and ``domain_change``), not in the
        global pre-stage emit.  Nodes referenced exclusively by
        stage-activated elements move into the stage's block too;
        nodes shared with global elements stay global.

        May be called multiple times per stage (PGs accumulate as a
        set; duplicates collapse).  Same PG activated in two
        different stages is a build-time error (first-write wins
        is unsafe — the user clearly meant something different).

        Parameters
        ----------
        pgs
            Iterable of element-PG names (e.g. ``["cimbra"]``,
            ``["rock", "lining"]``).  Each must be a non-empty string.
        """
        for pg in pgs:
            if not isinstance(pg, str) or not pg:
                raise ValueError(
                    f"Stage {self._name!r}.activate: pgs= must be an "
                    "iterable of non-empty strings, got "
                    f"{pg!r}."
                )
            if pg not in self._activated_pgs:
                self._activated_pgs.append(pg)

    def analysis(
        self,
        *,
        test: Primitive,
        algorithm: Primitive,
        integrator: Primitive,
        constraints: Primitive,
        numberer: Primitive,
        system: Primitive,
        analysis: Primitive,
    ) -> None:
        """Bind the analysis chain for this stage.

        All seven arguments are required.  Each must be a primitive
        already registered with the bridge (e.g. via
        ``ops.test.NormDispIncr(...)``); the stage holds a reference
        only, not a copy.  Multiple stages may share the same
        primitive instance (e.g. the same ``constraints.Plain()``
        across all stages) — the bridge emits each primitive exactly
        once per stage in which it's referenced, so OpenSees gets a
        fresh ``constraints Plain`` line per stage as required.
        """
        if self._analysis_set:
            raise ValueError(
                f"Stage {self._name!r}.analysis: already called; "
                "stages support one analysis chain each."
            )
        self._test = test
        self._algorithm = algorithm
        self._integrator = integrator
        self._constraints = constraints
        self._numberer = numberer
        self._system = system
        self._analysis = analysis
        self._analysis_set = True

    def run(
        self, *, n_increments: int, dt: float | None = None,
        strategy: "Ladder | None" = None,
    ) -> None:
        """Set the analyze-loop length + step size for this stage.

        ``strategy`` (ADR 0057 Phase A) attaches a solution-strategy
        ladder to this stage's analyze loop: on a failed increment the
        emitted loop escalates through the ladder's algorithm rungs
        (the chain's own algorithm is rung 0) and restores rung 0
        after a rescue; exhausting the ladder aborts fail-loud.
        Build one via ``ops.strategy.profile("non-smooth")`` or
        ``ops.strategy.Ladder(rungs=[...])``.
        """
        if self._run_set:
            raise ValueError(
                f"Stage {self._name!r}.run: already called; "
                "stages support one analyze loop each."
            )
        if n_increments < 1:
            raise ValueError(
                f"Stage {self._name!r}.run: n_increments must be >= 1, "
                f"got {n_increments}."
            )
        self._n_increments = int(n_increments)
        self._dt = None if dt is None else float(dt)
        self._strategy = strategy
        self._run_set = True


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
