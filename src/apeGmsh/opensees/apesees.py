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
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeVar

from ._internal.build import (
    BridgeError,
    FixRecord,
    InitialStressRecord,
    MassRecord,
    RegionAssignmentRecord,
    StageRecord,
    allocate_element_tags,
    build_element_partition_owner,
    build_node_partition_owners,
    compute_stage_ownership,
    emit_element_spec,
    emit_element_spec_partitioned,
    emit_initial_stress_addtoparameter,
    emit_initial_stress_global,
    emit_mp_constraints,
    emit_mp_constraints_partitioned,
    emit_pattern_spec,
    emit_recorder_spec,
    emit_transform_specs,
    expand_pg_to_nodes,
    is_partitioned,
    topological_order,
)
from ._internal.build import _element_transf as _build_element_transf
from ._internal.tag_resolution import (
    set_current_fem_element_id,
    set_element_nodes,
)
from ._internal.compose import _compose_model_h5, _path_stem
from ._internal.ns import (
    _AlgorithmNS,
    _AnalysisNS,
    _BeamIntegrationNS,
    _ConstraintsNS,
    _ElementNS,
    _GeomTransfNS,
    _IntegratorNS,
    _NDMaterialNS,
    _NumbererNS,
    _PatternNS,
    _RecorderNS,
    _SectionNS,
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
from .emitter.base import Emitter
from .node import Node, _NodeAccessor, _iter_tags
from .recorder import MPCO
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
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.results.capture.spec import DomainCaptureSpec

    from .analysis.eigen import EigenResult


__all__ = ["apeSees", "BuiltModel"]


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


# ---------------------------------------------------------------------------
# Partition-aware MPCO recorder plan (ADR 0027 INV-4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _MPCOFilterPlan:
    """Pre-resolved filter ids + shared region tag for one MPCO recorder.

    Built once before the partitioned per-rank loop in
    :meth:`BuiltModel._emit_partitioned`; consumed by
    :meth:`BuiltModel._emit_mpco_filter_regions_for_rank` (inside each
    rank block) and by the global recorder emit pass after the loop.

    The ``materialised_spec`` carries ``_region_tag`` populated and
    ``nodes_pg`` / ``elements_pg`` cleared so the spec's ``_emit``
    appends ``-R <region_tag>`` to the ``recorder mpco`` line without
    re-entering ``MPCO.materialize`` (which would otherwise allocate a
    new region tag and re-emit the region globally — the buggy pre-
    ADR-0027 INV-4 behaviour for the partitioned path).
    """
    region_tag: int
    node_ids: tuple[int, ...]
    elem_ids: tuple[int, ...]
    materialised_spec: "MPCO"


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
    initial_stress_records:  tuple[InitialStressRecord, ...] = ()
    stage_records:           tuple[StageRecord, ...] = ()

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

        # 4. Emit non-element / non-transform primitives in topo order.
        pre_element: list[Primitive] = []
        post_element: list[Primitive] = []
        for p in rest:
            if isinstance(p, (Pattern, Recorder)):
                post_element.append(p)
            else:
                pre_element.append(p)

        # ADR 0027: partitioned vs unpartitioned branch.  The
        # unpartitioned path must be **byte-identical** to the pre-ADR
        # 0027 behaviour — no ``partition_open`` / ``partition_close``
        # calls, no runtime shim, no per-rank fan-out.  Single-
        # partition / unpartitioned models keep the flat emit order
        # exactly as it was.
        if not is_partitioned(self.fem):
            self._emit_flat(
                emitter=emitter,
                tags=tags,
                transforms=transforms,
                elements=elements,
                pre_element=pre_element,
                post_element=post_element,
                base_resolver=_base_resolver,
            )
            return 0

        # Partitioned path — per-rank fan-out per ADR 0027.  Phase
        # SSI-2.A: the (stages + partitions) combo is not yet
        # supported; the per-stage emit assumes the flat path.  Lift
        # in a follow-up when an MP-staged use case appears.
        if self.stage_records:
            raise NotImplementedError(
                "apeSees: combining staged builds (Phase SSI-2.A) "
                "with MP-partitioned FEMs (ADR 0027) is not yet "
                f"supported (got {len(self.stage_records)} stage(s) "
                f"and {len(self.fem.partitions)} partitions).  Use "
                "single-partition FEMs for staged decks, or non-staged "
                "builds for MP-partitioned ones."
            )
        self._emit_partitioned(
            emitter=emitter,
            tags=tags,
            transforms=transforms,
            elements=elements,
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
        if staged:
            element_owner_stage, node_owner_stage = compute_stage_ownership(
                self.stage_records, elements, self.fem,
            )

        # 1a. Nodes — emit every node from the FEM snapshot, EXCEPT
        # nodes bound to a stage (those emit inside that stage's
        # block per Phase SSI-2.B).
        for nid, xyz in zip(self.fem.nodes.ids, self.fem.nodes.coords):
            if int(nid) in node_owner_stage:
                continue
            emitter.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

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
        element_plan = allocate_element_tags(elements, self.fem, tags)
        fem_eid_to_ops_tag: dict[int, int] = {
            eid: ele_tag
            for _, sub in element_plan
            for eid, _conn, ele_tag in sub
        }
        for spec, sub in element_plan:
            if id(spec) in element_owner_stage:
                continue  # stage-bound — emit inside the stage block.
            transf_spec = _build_element_transf(spec)
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

        # 7. Fixes / masses / regions / broker loads.
        self._emit_fixes(emitter)
        self._emit_masses(emitter)
        self._emit_regions(emitter, tags)
        self._emit_broker_loads(emitter, tags)

        # 7b. MP constraints (Phase 7b, ADR 0022 INV-5).
        emit_mp_constraints(emitter, self.fem)

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

        # 8. Patterns + recorders.
        for p in post_element:
            tag = self.tag_for[id(p)]
            if isinstance(p, Pattern):
                emit_pattern_spec(p, emitter, tag, self.fem)
            elif isinstance(p, Recorder):
                emit_recorder_spec(p, emitter, tag, self.fem, tags=tags)
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
                overrides=overrides,
                base_resolver=base_resolver,
            )

    def _emit_stages_flat(
        self,
        emitter: Emitter,
        tags: TagAllocator,
        *,
        element_plan: "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]" = (),  # type: ignore[assignment]
        element_owner_stage: "dict[int, int]" = {},  # type: ignore[assignment]
        node_owner_stage: "dict[int, int]" = {},  # type: ignore[assignment]
        fem_eid_to_ops_tag: "dict[int, int]" = {},  # type: ignore[assignment]
        overrides: "dict[tuple[int, int], int] | None" = None,
        base_resolver: object = None,
    ) -> None:
        """Phase SSI-2.A / 2.B: emit each stage block in registration order.

        Per stage:

        1. ``stage_open(name)`` — comment delimiter.
        2. **(Phase SSI-2.B)** Stage-owned nodes — emit nodes that
           are exclusively referenced by stage-bound elements (per
           the ``node_owner_stage`` map).
        3. **(Phase SSI-2.B)** Stage-owned elements — emit the
           ``element`` commands for elements whose pg is activated
           by this stage.  Tags come from the global ``element_plan``
           (pre-allocated upfront so cross-stage tag identity holds).
        4. **(Phase SSI-2.B)** ``domain_change()`` — if any topology
           was added in this stage, tell OpenSees to rebuild its
           DOF map before the analysis chain emits.
        5. Stage's initial_stress records (parameter declarations +
           step_hook_ramp procs + addToParameter calls, exactly the
           same shape as the Phase SSI-1 non-staged global emit).
        6. Analysis-chain primitives — emit each via its ``_emit``
           (the bridge skipped these in the pre_element pass).
        7. ``emitter.analyze(steps=, dt=)`` — auto-wraps with hook
           dispatcher calls if any step_hook_ramp registered this
           stage (the emitter tracks ``_step_hooks_registered``).
        8. ``stage_close()`` — loadConst + wipeAnalysis + hook clear.

        Single-process path only.  The (stages + partitions) combo
        is not yet supported; ``_emit_partitioned`` raises
        ``NotImplementedError`` when stages are present.
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
            sidx = element_owner_stage.get(id(spec))
            if sidx is not None:
                stage_owned_specs.setdefault(sidx, []).append((spec, sub))

        # FEM node-id → coord index lookup (mirrors the
        # _emit_partitioned helper inline).  Cheap to build once.
        node_idx_lookup = {
            int(nid): i for i, nid in enumerate(self.fem.nodes.ids)
        }

        for stage_idx, stage in enumerate(self.stage_records):
            emitter.stage_open(stage.name)

            # 2. Owned nodes.
            owned_nodes = stage_owned_nodes.get(stage_idx, [])
            for nid in owned_nodes:
                idx = node_idx_lookup.get(nid)
                if idx is None:
                    continue
                xyz = self.fem.nodes.coords[idx]
                emitter.node(
                    nid,
                    float(xyz[0]), float(xyz[1]), float(xyz[2]),
                )

            # 3. Owned elements.
            owned_specs = stage_owned_specs.get(stage_idx, [])
            for spec, sub in owned_specs:
                transf_spec = _build_element_transf(spec)
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

            # 4. domainChange — only if this stage added topology.
            if owned_nodes or owned_specs:
                emitter.domain_change()

            # 5. Initial stress.
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

            # 6. Analysis chain.
            for chain in (
                stage.constraints, stage.numberer, stage.system,
                stage.test, stage.algorithm, stage.integrator,
                stage.analysis,
            ):
                if chain is not None:
                    chain_tag = self.tag_for[id(chain)]
                    chain._emit(emitter, chain_tag)

            # 7. Analyze loop (auto-wraps with hook dispatcher calls).
            emitter.analyze(steps=stage.n_increments, dt=stage.dt)

            # 8. Stage close — loadConst + wipeAnalysis + hook clear.
            emitter.stage_close()

    # -- Partitioned emit path (ADR 0027) ---------------------------------

    def _emit_partitioned(
        self,
        *,
        emitter: Emitter,
        tags: TagAllocator,
        transforms: "list[GeomTransf]",
        elements: "list[Element]",
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
        2. **Per-rank emission** for each rank in ascending partition id:

           * ``partition_open(rank)``
           * Owned nodes (only this rank's ``node_ids`` from the
             ``PartitionRecord``).
           * Owned elements (per-rank fan-out across each Element spec;
             non-owned elements still consume a tag slot to preserve
             cross-rank tag identity per ADR 0027 §"Tag determinism").
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
           declarations).
        """
        partitions = list(self.fem.partitions)
        node_owners = build_node_partition_owners(self.fem)
        element_owner = build_element_partition_owner(self.fem)

        # -- 1. Pre-element global primitives. ----------------------------
        for p in pre_element:
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
        element_plan = allocate_element_tags(elements, self.fem, tags)
        # Global fem-eid → ops-tag map; used by the initial_stress
        # per-rank ``addToParameter`` fan-out to translate the user's
        # FEM element selection into OpenSees element tags (Phase
        # SSI-1).
        fem_eid_to_ops_tag: dict[int, int] = {}
        for _, sub in element_plan:
            for eid, _conn, ele_tag in sub:
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
        rank_owned_nodes: dict[int, set[int]] = {
            rank: {int(n) for n in rec.node_ids}
            for rank, rec in enumerate(partitions)
        }

        # Cross-rank tag identity caches (region tags, broker
        # timeSeries / pattern tags, ADR 0027 §"Tag determinism").
        region_tag_cache: dict[str, int] = {}
        broker_ts_tag_cache: dict[str, int] = {}
        broker_pat_tag_cache: dict[str, int] = {}

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

        # Pre-compute the post-element rank-local plan for the bridge's
        # fix / mass / region / load passes.  We use the same shapes
        # the flat path uses but pre-intersect with per-rank ownership.
        # ``rank`` is the **0-based runtime rank** matching
        # OpenSeesMP's ``getPID()`` — derived from ``enumerate`` over
        # ``partitions`` so it does not collide with Gmsh's 1-based
        # ``part.id``.  See the bug fix in commit titled
        # ``fix(opensees-bridge): emit 0-based runtime ranks``.
        for rank, part in enumerate(partitions):
            emitter.partition_open(rank)
            try:
                # 1a. Owned nodes — emit in node-id order for stable
                # cross-rank diffs.  Only THIS rank's node_ids are
                # emitted here; foreign-side declarations for cross-
                # partition MP constraints happen in the constraint
                # pass below (INV-2).
                node_idx_lookup = {
                    int(nid): i for i, nid in enumerate(self.fem.nodes.ids)
                }
                for nid in sorted(int(n) for n in part.node_ids):
                    idx = node_idx_lookup.get(nid)
                    if idx is None:
                        continue
                    xyz = self.fem.nodes.coords[idx]
                    emitter.node(
                        nid,
                        float(xyz[0]), float(xyz[1]), float(xyz[2]),
                    )

                # 6. Elements — per-rank fan-out (tags pre-allocated).
                for ele_spec, pre_alloc in element_plan:
                    emit_element_spec_partitioned(
                        spec=ele_spec,
                        emitter=emitter,
                        fem=self.fem,
                        pre_allocated=pre_alloc,
                        base_resolver=base_resolver,
                        transf_tag_for_element=overrides,
                        partition_rank=rank,
                        element_owner=element_owner,
                    )

                # 7. Fixes / masses (per-rank ownership).
                self._emit_fixes_partitioned(emitter, rank_owned_nodes[rank])
                self._emit_masses_partitioned(emitter, rank_owned_nodes[rank])

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
                )

                # 7a. Broker nodal loads (re-partitioned per-rank).
                self._emit_broker_loads_partitioned(
                    emitter, tags, rank_owned_nodes[rank],
                    broker_ts_tag_cache, broker_pat_tag_cache,
                )

                # 7b. MP constraints (ADR 0027 — replication policy).
                emit_mp_constraints_partitioned(
                    emitter=emitter,
                    fem=self.fem,
                    partition_rank=rank,
                    node_owners=node_owners,
                    element_owner=element_owner,
                    foreign_node_ndf=int(self.ndf),
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

                # 8. Patterns (loads + sps) per-rank.
                self._emit_patterns_partitioned(
                    emitter, post_element, rank_owned_nodes[rank],
                )
            finally:
                emitter.partition_close()

        # -- 3. Analysis chain — emitted GLOBALLY (outside any rank
        # block).  Auto-emit Transformation handler + ParallelPlain
        # numberer + Mumps system per ADR 0027 §"Constraint handler
        # interaction" (INV-5).
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
        for p in post_element:
            if not isinstance(p, Recorder):
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
                emit_recorder_spec(p, emitter, tag, self.fem, tags=tags)

    # -- Model-level fix / mass fan-out -----------------------------------

    def _emit_fixes(self, emitter: Emitter) -> None:
        for rec in self.fix_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                emitter.fix(node_tag, *rec.dofs)

    def _emit_masses(self, emitter: Emitter) -> None:
        for rec in self.mass_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                emitter.mass(node_tag, *rec.values)

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
    ) -> None:
        """Per-rank mass fan-out (ADR 0027). Mirror of :meth:`_emit_fixes_partitioned`."""
        for rec in self.mass_records:
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                if int(node_tag) in owned_nodes:
                    emitter.mass(node_tag, *rec.values)

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
        """Pre-resolve filter ids + allocate one region tag per filter-bearing MPCO.

        Returns a dict keyed by ``id(spec)`` carrying the resolved
        ``(node_ids, elem_ids)`` plus the shared region tag and the
        materialised spec (with ``_region_tag`` populated, filters
        cleared, ready for ``_emit``).  Non-MPCO recorders and
        whole-model MPCO recorders are absent from the dict; the
        caller routes them through the unchanged
        :func:`emit_recorder_spec` global pass.

        The plan is built ONCE before the per-rank loop so:

        1. The region tag is the SAME scalar across every rank that
           emits its rank-intersection of the region (INV-4: stitching
           by tag identity).
        2. ``_emit`` is bypass-safe — the materialised spec carries the
           shared ``_region_tag`` directly, so the global recorder pass
           after the per-rank loop simply forwards ``-R <tag>`` onto the
           ``recorder mpco`` line without re-allocating a tag or re-
           emitting the region globally.
        """
        plan: dict[int, _MPCOFilterPlan] = {}
        for p in post_element:
            if not isinstance(p, MPCO):
                continue
            if not p.has_filter():
                continue
            node_ids, elem_ids = p.resolve_filter_ids(self.fem)
            region_tag = tags.allocate("region")
            materialised = replace(
                p,
                nodes_pg=None,
                elements_pg=None,
                nodes=node_ids if node_ids else None,
                elements=elem_ids if elem_ids else None,
                _region_tag=region_tag,
            )
            plan[id(p)] = _MPCOFilterPlan(
                region_tag=region_tag,
                node_ids=node_ids,
                elem_ids=elem_ids,
                materialised_spec=materialised,
            )
        return plan

    def _emit_mpco_filter_regions_for_rank(
        self,
        emitter: Emitter,
        rank: int,
        plan: "dict[int, _MPCOFilterPlan]",
        owned_nodes: set[int],
        element_owner: dict[int, int],
    ) -> None:
        """Per-rank emission of MPCO recorder filter regions (INV-4).

        For every filter-bearing MPCO recorder, intersects the resolved
        node ids with ``owned_nodes`` and the resolved element ids with
        the rank's owned elements (via ``element_owner``).  When BOTH
        intersections are empty the recorder's region is omitted on
        this rank (INV-4 §"empty intersection ⇒ no region emitted on
        that rank").

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
        for entry in plan.values():
            # Per-rank node intersection — preserves original declaration order.
            rank_node_ids = tuple(
                n for n in entry.node_ids if int(n) in owned_nodes
            )
            # Per-rank element intersection — keep elements whose owner
            # is this rank.  Element ownership is single-rank
            # (build_element_partition_owner), so a missing key means
            # the element isn't on any rank — skip silently.
            rank_elem_ids = tuple(
                e for e in entry.elem_ids
                if element_owner.get(int(e)) == rank
            )
            if not rank_node_ids and not rank_elem_ids:
                # INV-4: empty intersection on this rank → no region line.
                continue
            region_args: list[int | float | str] = []
            if rank_node_ids:
                region_args += ["-node", *rank_node_ids]
            if rank_elem_ids:
                region_args += ["-ele", *rank_elem_ids]
            emitter.region(entry.region_tag, *region_args)

    def _resolve_node_target(
        self, pg: str | None, nodes: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if pg is not None:
            return expand_pg_to_nodes(self.fem, pg)
        assert nodes is not None  # exactly-one-of validated at apeSees.fix
        return nodes

    # -- Broker nodal-load fan-out (ADR 0001) -----------------------------

    def _emit_broker_loads(
        self, emitter: Emitter, tags: TagAllocator,
    ) -> None:
        """Emit ``fem.nodes.loads`` as synthesized Plain patterns.

        No-op when the FEM snapshot exposes no ``nodes.loads`` (e.g.
        hand-rolled test stubs) — broker loads are purely additive on
        top of any registered bridge primitives.
        """
        nodes = getattr(self.fem, "nodes", None)
        load_set = getattr(nodes, "loads", None)
        if load_set is None:
            return
        by_pattern: dict[str, list[Any]] = {}
        for rec in load_set:
            by_pattern.setdefault(rec.pattern, []).append(rec)
        for recs in by_pattern.values():
            if not recs:
                continue
            ts_tag = tags.allocate("timeSeries")
            pat_tag = tags.allocate("pattern")
            emitter.timeSeries("Linear", ts_tag)
            emitter.pattern_open("Plain", pat_tag, ts_tag)
            for rec in recs:
                emitter.load(
                    int(rec.node_id), *self._broker_load_components(rec),
                )
            emitter.pattern_close()

    def _broker_load_components(self, rec: Any) -> tuple[float, ...]:
        """Map a DOF-agnostic ``NodalLoadRecord`` onto this model's ndf.

        apeGmsh records store pure 3-D spatial force/moment vectors;
        the bridge is the only layer that knows ``ndf``, so the DOF
        mapping lives here (per the records' DOF-agnostic contract).
        """
        fx, fy, fz = rec.force_xyz or (0.0, 0.0, 0.0)
        mx, my, mz = rec.moment_xyz or (0.0, 0.0, 0.0)
        if self.ndf == 2:
            return (fx, fy)
        if self.ndf == 3:                 # planar frame: ux, uy, rz
            return (fx, fy, mz)
        if self.ndf == 6:
            return (fx, fy, fz, mx, my, mz)
        return (fx, fy, fz, mx, my, mz)[: self.ndf]

    def _emit_broker_loads_partitioned(
        self,
        emitter: Emitter,
        tags: TagAllocator,
        owned_nodes: set[int],
        ts_tag_cache: dict[str, int],
        pat_tag_cache: dict[str, int],
    ) -> None:
        """Per-rank broker nodal-load fan-out (ADR 0027).

        Mirrors :meth:`_emit_broker_loads` but emits only loads
        targeting nodes owned by this rank. Pattern / time-series tags
        are cached across the per-rank loop so ranks that emit the
        same pattern share the same tag (cross-rank tag identity per
        ADR 0027 §"Tag determinism").
        """
        nodes = getattr(self.fem, "nodes", None)
        load_set = getattr(nodes, "loads", None)
        if load_set is None:
            return

        by_pattern: dict[str, list[Any]] = {}
        for rec in load_set:
            by_pattern.setdefault(rec.pattern, []).append(rec)
        if not by_pattern:
            return

        for pattern_name, recs in by_pattern.items():
            owned_recs = [
                r for r in recs if int(r.node_id) in owned_nodes
            ]
            if not owned_recs:
                continue
            ts_tag = ts_tag_cache.get(pattern_name)
            if ts_tag is None:
                ts_tag = tags.allocate("timeSeries")
                ts_tag_cache[pattern_name] = ts_tag
            pat_tag = pat_tag_cache.get(pattern_name)
            if pat_tag is None:
                pat_tag = tags.allocate("pattern")
                pat_tag_cache[pattern_name] = pat_tag
            emitter.timeSeries("Linear", ts_tag)
            emitter.pattern_open("Plain", pat_tag, ts_tag)
            for rec in owned_recs:
                emitter.load(
                    int(rec.node_id), *self._broker_load_components(rec),
                )
            emitter.pattern_close()

    def _emit_patterns_partitioned(
        self,
        emitter: Emitter,
        post_element: "list[Primitive]",
        owned_nodes: set[int],
    ) -> None:
        """Per-rank pattern fan-out (ADR 0027).

        Walks every :class:`Pattern` primitive (skipping recorders,
        which emit globally outside any partition block) and emits
        only the ``p.load`` / ``p.sp`` rows targeting nodes owned by
        this rank.  Non-Plain patterns delegate verbatim — they have
        no per-node fan-out to filter, and OpenSeesMP handles them
        with their own per-rank semantics (e.g. ``UniformExcitation``
        applies on every rank simultaneously).
        """
        from .pattern.pattern import Plain, _LoadRecord, _SPRecord
        from ._internal.tag_resolution import resolve_tag

        for p in post_element:
            if not isinstance(p, Pattern):
                continue
            tag = self.tag_for[id(p)]
            if not isinstance(p, Plain):
                # Non-Plain pattern (UniformExcitation etc.) — emit on
                # every rank verbatim.  Per ADR 0027 these patterns
                # have no per-node fan-out the bridge can filter; the
                # OpenSeesMP semantics for them are pattern-class-
                # specific.
                p._emit(emitter, tag)
                continue

            ts_tag = resolve_tag(emitter, p.series)
            # Pre-filter loads / sps so we don't open an empty pattern.
            owned_loads = [
                rec for rec in p.loads
                if _pattern_record_owned(rec, owned_nodes, self.fem)
            ]
            owned_sps = [
                rec for rec in p.sps
                if _pattern_record_owned(rec, owned_nodes, self.fem)
            ]
            if not owned_loads and not owned_sps:
                continue
            emitter.pattern_open("Plain", tag, ts_tag)
            for rec in owned_loads:
                _emit_pattern_load_partitioned(
                    rec, emitter, self.fem, owned_nodes,
                )
            for rec in owned_sps:
                _emit_pattern_sp_partitioned(
                    rec, emitter, self.fem, owned_nodes,
                )
            emitter.pattern_close()

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
        from .analysis.constraint_handler import Plain as ConstraintsPlain

        import warnings as _warnings

        declared_handler: "ConstraintHandler | None" = None
        for p in pre_element:
            if isinstance(p, ConstraintHandler):
                declared_handler = p
                break

        if declared_handler is None:
            # No user-declared handler — auto-emit Transformation.
            _warnings.warn(
                "MP constraints are present in the model (equalDOF, "
                "rigidLink, rigidDiaphragm, or surface couplings). "
                "Auto-emitting 'Transformation' constraint handler. "
                "To override, explicitly declare ops.constraints.X() "
                "before build().",
                UserWarning,
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
                UserWarning,
                stacklevel=2,
            )
            return
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
                UserWarning,
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
                UserWarning,
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
                UserWarning,
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
                UserWarning,
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
    rec: "Any", emitter: Emitter, fem: "FEMData", owned_nodes: set[int],
) -> None:
    """Per-rank version of the inner load fan-out."""
    if rec.target_kind == "node":
        if int(rec.target) in owned_nodes:
            emitter.load(int(rec.target), *rec.forces)
        return
    # PG — fan out only owned nodes.
    try:
        ids = fem.nodes.select(pg=rec.target).ids
    except (KeyError, ValueError, AttributeError):
        return
    for node_tag in ids:
        if int(node_tag) in owned_nodes:
            emitter.load(int(node_tag), *rec.forces)


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
    ) -> None:
        self._fem: "FEMData" = fem
        self._primitives: list[Primitive] = []
        self._tags = TagAllocator()
        self._ndm: int | None = None
        self._ndf: int | None = None
        self._fix_records: list[FixRecord] = []
        self._mass_records: list[MassRecord] = []
        self._region_records: list[RegionAssignmentRecord] = []
        self._initial_stress_records: list[InitialStressRecord] = []
        # Phase SSI-2.A: closed StageRecord instances accumulate here as
        # ``with ops.stage(name) as s:`` blocks exit.  ``stage_records``
        # being non-empty switches BuiltModel.emit into the staged
        # emission path (per-stage analyze loops with loadConst /
        # wipeAnalysis / hook-list clear between).
        self._stage_records: list[StageRecord] = []
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
        self.element          = _ElementNS(self)
        self.recorder         = _RecorderNS(self)

        # FEM-aware aggregates (Phase 5A) — query-and-act over fem.nodes.
        self.nodes            = _NodeAccessor(self)
        self.constraints      = _ConstraintsNS(self)
        self.numberer         = _NumbererNS(self)
        self.system           = _SystemNS(self)
        self.test             = _TestNS(self)
        self.algorithm        = _AlgorithmNS(self)
        self.integrator       = _IntegratorNS(self)
        self.analysis         = _AnalysisNS(self)

    # -- Read-only access to the FEM snapshot ----------------------------
    @property
    def fem(self) -> "FEMData":
        return self._fem

    # -- Flat methods ----------------------------------------------------

    def model(self, *, ndm: int, ndf: int) -> None:
        """Set the model dimensionality (``ndm``) and DOFs/node (``ndf``)."""
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
        return DomainCapture(resolved, path, self._fem, ops=ops)

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
    ) -> None:
        """Attach lumped nodal mass.

        Exactly one of ``pg`` / ``nodes`` must be supplied. ``nodes``
        accepts plain integers or :class:`Node` instances.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                "apeSees.mass: supply exactly one of pg= or nodes= "
                f"(got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        self._mass_records.append(
            MassRecord(pg=pg, nodes=nodes_tuple, values=tuple(values)),
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
        if (pg is None) == (elements is None):
            raise ValueError(
                "apeSees.initial_stress: supply exactly one of pg= or "
                f"elements= (got pg={pg!r}, elements={elements!r})."
            )
        if not name:
            raise ValueError(
                "apeSees.initial_stress: name= must be non-empty."
            )
        # Tcl identifier safety: alphanumeric + underscore, not starting
        # with a digit.  Keeps the emitted ``proc <name>`` parsable.
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError(
                "apeSees.initial_stress: name must be a valid Tcl "
                "identifier (alphanumeric + underscore, not starting "
                f"with a digit). Got name={name!r}."
            )
        if ramp_steps < 1:
            raise ValueError(
                "apeSees.initial_stress: ramp_steps must be >= 1, "
                f"got {ramp_steps}."
            )
        if not (0.0 < lambda_install <= 1.0):
            raise ValueError(
                "apeSees.initial_stress: lambda_install must be in "
                f"(0, 1], got {lambda_install}."
            )
        elements_tuple = (
            tuple(int(e) for e in elements) if elements is not None else None
        )
        record = InitialStressRecord(
            name=str(name),
            pg=pg,
            elements=elements_tuple,
            sigma_xx=float(sigma_xx),
            sigma_yy=float(sigma_yy),
            sigma_zz=float(sigma_zz),
            ramp_steps=int(ramp_steps),
            lambda_install=float(lambda_install),
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
            The registered :class:`Plain` pattern.  Phase SSI-3 ships
            the pattern as a global registration; if used inside a
            staged deck, it fires in every stage's analyze loop
            (gate via the time series if that's not desired).

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
        return _StageBuilder(self, str(name))

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

    def analyze(self, *, steps: int, dt: float | None = None) -> int:
        """Build + emit + run the analysis chain via the live emitter.

        Builds a :class:`BuiltModel`, drives a
        :class:`~apeGmsh.opensees.emitter.live.LiveOpsEmitter` end-to-
        end, then issues the ``analyze`` call. Returns the openseespy
        ``analyze`` return value (0 on success).

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

        # Local import — keeps openseespy out of import-time for users
        # who only emit Tcl / py.
        from .emitter.live import LiveOpsEmitter

        bm = self.build()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        result: int = int(live_emitter.analyze(steps=steps, dt=dt))
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
        """
        if num_modes < 1:
            raise ValueError(
                f"apeSees.eigen: num_modes must be >= 1, got {num_modes}."
            )

        # Local imports — keep openseespy + numpy out of bridge import
        # time for Tcl/Py/H5-only users.
        from .analysis.eigen import EigenResult
        from .emitter.live import LiveOpsEmitter
        import numpy as np

        bm = self.build()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen(num_modes, solver=solver)
        return EigenResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            _live=live_emitter,
        )

    def tcl(
        self,
        path: str,
        *,
        run: bool = False,
        bin: str | None = None,
        analyze_steps: int | None = None,
        analyze_dt: float | None = None,
    ) -> None:
        """Emit a Tcl deck to ``path``; optionally subprocess OpenSees.

        When ``analyze_steps`` is supplied, an ``analyze`` line is
        appended to the deck after every other primitive — wrapped in
        a hook-dispatching for-loop if any
        :meth:`initial_stress` calls registered step hooks (Phase
        SSI-1).  Without ``analyze_steps``, the emitted deck declares
        the model but does not drive an analysis.
        """
        from .emitter.tcl import TclEmitter

        bm = self.build()
        emitter = TclEmitter()
        bm.emit(emitter)
        if analyze_steps is not None:
            emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(emitter.lines()) + "\n")

        if not run:
            return

        binary = _resolve_opensees_binary(bin)
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
    ) -> None:
        """Emit an openseespy Python deck to ``path``; optionally run it.

        ``analyze_steps`` / ``analyze_dt`` semantics mirror :meth:`tcl`
        (Phase SSI-1).
        """
        from .emitter.py import PyEmitter

        bm = self.build()
        emitter = PyEmitter()
        bm.emit(emitter)
        if analyze_steps is not None:
            emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(emitter.lines()) + "\n")

        if not run:
            return

        python_bin = _resolve_python_binary()
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
        )

    # -- Registration -----------------------------------------------------

    def _register(self, prim: _P) -> _P:
        """Add ``prim`` to the bridge, allocate its tag, return it."""
        kind = _kind_of(prim)
        self._tags.allocate_for(prim, kind)
        self._primitives.append(prim)
        return prim

    def register(self, prim: _P) -> _P:
        """Register a standalone primitive with the bridge (P11)."""
        return self._register(prim)

    def tag_for(self, prim: Primitive) -> int | None:
        """Return ``prim``'s allocated tag, or ``None`` if unregistered."""
        return self._tags.tag_for(prim)

    # -- Build -----------------------------------------------------------

    def build(self) -> BuiltModel:
        """Freeze the declarations into a :class:`BuiltModel`."""
        if self._ndm is None or self._ndf is None:
            raise RuntimeError(
                "apeSees.model(ndm=..., ndf=...) must be called before "
                "build()."
            )

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
            initial_stress_records=tuple(self._initial_stress_records),
            stage_records=tuple(self._stage_records),
        )

    # -- Internal helpers ------------------------------------------------

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
        "_activated_pgs",
        "_test", "_algorithm", "_integrator",
        "_constraints", "_numberer", "_system", "_analysis",
        "_n_increments", "_dt",
        "_analysis_set", "_run_set",
    )

    def __init__(self, bridge: "apeSees", name: str) -> None:
        self._bridge = bridge
        self._name = name
        self._initial_stress_records: list[InitialStressRecord] = []
        self._activated_pgs: list[str] = []
        self._test: Primitive | None = None
        self._algorithm: Primitive | None = None
        self._integrator: Primitive | None = None
        self._constraints: Primitive | None = None
        self._numberer: Primitive | None = None
        self._system: Primitive | None = None
        self._analysis: Primitive | None = None
        self._n_increments: int = 0
        self._dt: float | None = None
        self._analysis_set: bool = False
        self._run_set: bool = False

    def __enter__(self) -> "_StageBuilder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if exc_type is not None:
            # Don't swallow user's exception; just drop the in-progress
            # stage (no records appended to the bridge).
            return False
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
            activated_pgs=tuple(self._activated_pgs),
        )
        self._bridge._stage_records.append(record)
        return False

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

    def run(self, *, n_increments: int, dt: float | None = None) -> None:
        """Set the analyze-loop length + step size for this stage."""
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
        self._run_set = True


# ---------------------------------------------------------------------------
# Binary resolution helpers
# ---------------------------------------------------------------------------

def _resolve_opensees_binary(explicit: str | None) -> str:
    """Resolve the OpenSees Tcl binary path.

    Search order: explicit ``bin=`` argument, ``$OPENSEES_BIN``,
    ``shutil.which("OpenSees")``. Raises :class:`FileNotFoundError`
    if all three are unset / not found.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("OPENSEES_BIN")
    if env:
        return env
    on_path = shutil.which("OpenSees")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "OpenSees Tcl binary not found. Tried: bin= argument, "
        "$OPENSEES_BIN environment variable, shutil.which('OpenSees'). "
        "Set $OPENSEES_BIN or install OpenSees on PATH."
    )


def _resolve_python_binary() -> str:
    """Resolve the python interpreter to run an openseespy script.

    Search order: ``$OPENSEES_VENV``'s python, ``shutil.which("python")``,
    ``sys.executable``. Falls back to the running interpreter if no
    explicit venv is configured.
    """
    venv = os.environ.get("OPENSEES_VENV")
    if venv:
        if os.name == "nt":
            candidate = os.path.join(venv, "Scripts", "python.exe")
        else:
            candidate = os.path.join(venv, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    on_path = shutil.which("python")
    if on_path:
        return on_path
    return sys.executable
