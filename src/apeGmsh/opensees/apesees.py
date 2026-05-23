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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeVar

from ._internal.build import (
    BridgeError,
    FixRecord,
    MassRecord,
    RegionAssignmentRecord,
    allocate_element_tags,
    build_element_partition_owner,
    build_node_partition_owners,
    emit_element_spec,
    emit_element_spec_partitioned,
    emit_mp_constraints,
    emit_mp_constraints_partitioned,
    emit_pattern_spec,
    emit_recorder_spec,
    emit_transform_specs,
    expand_pg_to_nodes,
    is_partitioned,
    topological_order,
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
    """

    primitives:      tuple[Primitive, ...]
    tag_for:         dict[int, int]
    ndm:             int
    ndf:             int
    fem:             "FEMData"
    fix_records:     tuple[FixRecord, ...]
    mass_records:    tuple[MassRecord, ...]
    region_records:  tuple[RegionAssignmentRecord, ...]

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

        # Partitioned path — per-rank fan-out per ADR 0027.
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
        """
        # 1a. Nodes — emit every node from the FEM snapshot.
        for nid, xyz in zip(self.fem.nodes.ids, self.fem.nodes.coords):
            emitter.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

        # 4a. Materials / sections / time series / analysis chain
        # (excluding patterns + recorders).
        for p in pre_element:
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

        # 6. Elements.
        for ele_spec in elements:
            emit_element_spec(
                spec=ele_spec,
                emitter=emitter,
                fem=self.fem,
                tags=tags,
                base_resolver=base_resolver,
                transf_tag_for_element=overrides,
            )

        # 7. Fixes / masses / regions / broker loads.
        self._emit_fixes(emitter)
        self._emit_masses(emitter)
        self._emit_regions(emitter, tags)
        self._emit_broker_loads(emitter, tags)

        # 7b. MP constraints (Phase 7b, ADR 0022 INV-5).
        emit_mp_constraints(emitter, self.fem)

        # 7c. Auto-emit constraint handler when MP constraints present.
        self._maybe_auto_emit_constraint_handler(emitter, pre_element)

        # 8. Patterns + recorders.
        for p in post_element:
            tag = self.tag_for[id(p)]
            if isinstance(p, Pattern):
                emit_pattern_spec(p, emitter, tag, self.fem)
            elif isinstance(p, Recorder):
                emit_recorder_spec(p, emitter, tag, self.fem, tags=tags)
            else:  # pragma: no cover  - unreachable per partition above
                p._emit(emitter, tag)

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

        # Stable per-rank node tags — sort within each rank by node id
        # so cross-rank diffs of the emitted text are grep-friendly.
        # Cache the owned set per rank for the fix / mass / region
        # passes below.
        rank_owned_nodes: dict[int, set[int]] = {
            int(rec.id): {int(n) for n in rec.node_ids}
            for rec in partitions
        }

        # Cross-rank tag identity caches (region tags, broker
        # timeSeries / pattern tags, ADR 0027 §"Tag determinism").
        region_tag_cache: dict[str, int] = {}
        broker_ts_tag_cache: dict[str, int] = {}
        broker_pat_tag_cache: dict[str, int] = {}

        # Pre-compute the post-element rank-local plan for the bridge's
        # fix / mass / region / load passes.  We use the same shapes
        # the flat path uses but pre-intersect with per-rank ownership.
        for part in partitions:
            rank = int(part.id)
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
        for p in post_element:
            if isinstance(p, Recorder):
                tag = self.tag_for[id(p)]
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
                "auto-emitting 'numberer ParallelPlain' (OpenSeesMP-"
                "compatible).  Explicitly declare "
                "ops.numberer.<Plain|RCM|ParallelPlain|ParallelRCM>() "
                "before build() to override.",
                UserWarning,
                stacklevel=2,
            )
            emitter.numberer("ParallelPlain")
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
                "auto-emitting 'system Mumps' (OpenSeesMP-compatible "
                "parallel sparse-direct solver).  Explicitly declare "
                "ops.system.<Mumps|MumpsParallel>() before build() to "
                "override.",
                UserWarning,
                stacklevel=2,
            )
            emitter.system("Mumps")
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
        """
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
    ) -> None:
        """Emit a Tcl deck to ``path``; optionally subprocess OpenSees."""
        from .emitter.tcl import TclEmitter

        bm = self.build()
        emitter = TclEmitter()
        bm.emit(emitter)
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

    def py(self, path: str, *, run: bool = False) -> None:
        """Emit an openseespy Python deck to ``path``; optionally run it."""
        from .emitter.py import PyEmitter

        bm = self.build()
        emitter = PyEmitter()
        bm.emit(emitter)
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
