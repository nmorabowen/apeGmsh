"""Partition and stage emission blocks for :mod:`apeGmsh.opensees.emitter.h5`."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from .._internal.typed_records import (
    EmbeddedNodeRecord as _EmbeddedNodeRecord,
    EqualDOFRecord as _EqualDOFRecord,
    FixRecord as _FixRecord,
    MassRecord as _MassRecord,
    PatternRecord as _PatternRecord,
    RecorderRecord as _RecorderRecord,
    RegionRecord as _RegionRecord,
    RigidDiaphragmRecord as _RigidDiaphragmRecord,
    RigidLinkRecord as _RigidLinkRecord,
)

# ---------------------------------------------------------------------------
# Partition emission (ADR 0027, schema 2.10.0)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PartitionEmitBlock:
    """In-memory accumulator for a single ``partition_open(rank)`` block.

    Lives only inside :class:`H5Emitter`; the read-side reuses
    :class:`apeGmsh.opensees.emitter.h5_reader.PartitionEmittedRecord`
    which is the public surface.  The two are intentionally separate —
    the writer carries mutable accumulators (lists; node_set for dedupe),
    the reader carries immutable tuples.
    """

    #: OpenSeesMP rank id this block scopes.
    rank: int
    #: Node tags emitted under this block, in insertion order, with
    #: duplicates dropped (a phantom node declared twice in one rank
    #: block from two MP constraints would still count once).
    node_ids: list[int] = field(default_factory=list)
    #: Element tags emitted under this block, in insertion order.
    element_ids: list[int] = field(default_factory=list)
    #: O(1) dedupe membership test for ``node_ids`` (parallel
    #: shadow — never read directly, only via ``add_node``).
    _node_set: set[int] = field(default_factory=set)

    def add_node(self, tag: int) -> None:
        """Append ``tag`` if not already present (dedupe within block)."""
        if tag not in self._node_set:
            self._node_set.add(tag)
            self.node_ids.append(tag)

    def add_element(self, tag: int) -> None:
        """Append ``tag`` to the block's element list."""
        self.element_ids.append(tag)


# ---------------------------------------------------------------------------
# Stage emission (ADR 0055 Phase 2, schema 2.18.0)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StageEmitBlock:
    """In-memory capture bucket for one ``stage_open(name)`` block.

    ADR 0055 Phase 2 — while a stage bracket is open, the emitter
    routes in-band Protocol calls here instead of (or, for topology,
    in addition to) the global buffers:

    * **Redirected** (would otherwise pollute the global zones and
      double-apply on replay): ``fix`` / ``mass`` / ``region`` / MP
      constraints / pattern brackets (loads / sps / sp_holds) /
      ``recorder`` / ``rayleigh`` / analysis-chain attrs / ``analyze``
      / the SSI-2.E mutators.  Redirecting the chain calls is what
      prevents the phantom global ``/opensees/analysis`` leak — the
      last stage's chain would otherwise land in ``_analysis_attrs``.
    * **Dual-appended** (global buffers must stay complete — element
      metadata and phantom-node bookkeeping cover ALL elements/nodes):
      ``node`` / ``element``, which additionally record the stage's
      owned tags here, in emit order.

    The declarative complement that never flows through the resolved
    Protocol stream (``activated_pgs``, per-stage initial-stress,
    ``activate_absorbing`` — their emit drives only no-op'd
    ``step_hook_ramp`` / ``addToParameter`` / ``flip_element_stage``
    calls) is attached afterwards by
    :meth:`H5Emitter.set_stage_records` (Phase-1 side-channel pattern).

    Tri-state capture is by *presence*: a mutator that was never
    called leaves its field at the sentinel and the writer omits the
    attribute entirely — never-set is structurally distinct from
    value-0 with no presence companion needed.
    """

    name: str
    # Owned topology (dual-append, emit order — also the replay order).
    owned_node_ids: list[int] = field(default_factory=list)
    owned_element_ids: list[int] = field(default_factory=list)
    # Phantom nodes emitted INSIDE this stage (stage-claimed
    # node_to_surface constraints synthesize them).  Their coordinates
    # land only in the write-only global ``_node_coords`` buffer, so a
    # staged archive listing them in ``owned_node_ids`` would be
    # irreplayable — ``set_stage_records`` fails loud on a non-empty
    # list until a per-stage phantom store exists (gate-2 finding).
    phantom_node_tags: list[int] = field(default_factory=list)
    # Per-stage emit-sequence counter + the slots stamped with it.
    # Regions arrive from four producers (stage regions, region-scoped
    # rayleigh, damping attach, recorder pg-filter) whose interleaving
    # with the global-form ``rayleigh`` call and the stage patterns
    # carries OpenSees overwrite semantics ("region refines global") —
    # the ``emit_index`` stamps preserve the original relative order so
    # replay can reproduce it (gate-2 finding; the lists alone lose it).
    emit_seq: int = 0
    region_seq: list[int] = field(default_factory=list)
    rayleigh_seq: list[int] = field(default_factory=list)
    pattern_seq: list[int] = field(default_factory=list)
    # Redirected stage-scoped pools (same record classes as the
    # global buffers so the per-group writers are reusable verbatim).
    fixes: "list[_FixRecord]" = field(default_factory=list)
    masses: "list[_MassRecord]" = field(default_factory=list)
    regions: "list[_RegionRecord]" = field(default_factory=list)
    equal_dofs: "list[_EqualDOFRecord]" = field(default_factory=list)
    rigid_links: "list[_RigidLinkRecord]" = field(default_factory=list)
    rigid_diaphragms: "list[_RigidDiaphragmRecord]" = field(
        default_factory=list,
    )
    embedded_nodes: "list[_EmbeddedNodeRecord]" = field(
        default_factory=list,
    )
    # ADR 0055 P2.3: per-MP-record emit_index. The bridge interleaves
    # the four MP kinds across one emit pass — crucially a
    # kinematic_coupling emits its equalDOF AFTER rigidDiaphragm
    # (build/ _emit_kinematic_couplings, step 5), so the genuine vs
    # kinematic equalDOFs straddle the rigid_diaphragm emit.  Four
    # separate typed buckets lose that cross-bucket order; the seq
    # stamps recover it (merge-sort on replay, same design as
    # region_seq / rayleigh_seq).
    equal_dof_seq: list[int] = field(default_factory=list)
    rigid_link_seq: list[int] = field(default_factory=list)
    rigid_diaphragm_seq: list[int] = field(default_factory=list)
    embedded_node_seq: list[int] = field(default_factory=list)
    patterns_complete: "list[_PatternRecord]" = field(default_factory=list)
    open_pattern: "_PatternRecord | None" = None
    recorders: "list[_RecorderRecord]" = field(default_factory=list)
    # K1-6 (schema 2.25.0): the declaration of each fix / mass / recorder
    # record above, parallel to its list (``/opensees/decls/rows``).
    fix_decls: list[int] = field(default_factory=list)
    mass_decls: list[int] = field(default_factory=list)
    recorder_decls: list[int] = field(default_factory=list)
    remove_sp_decls: list[int] = field(default_factory=list)
    remove_element_decls: list[int] = field(default_factory=list)
    update_material_stage_decls: list[int] = field(default_factory=list)
    rayleigh_decls: list[int] = field(default_factory=list)
    # The HOLD pattern's ``sp_holds`` rows, by the pattern's tag.
    sp_hold_decls: "dict[int, list[int]]" = field(default_factory=dict)
    # Global-form stage rayleigh (``on=()``) — four raw coefficients
    # per call.  Region-scoped rayleigh / damping attaches arrive as
    # ``region`` calls with the ``-rayleigh`` / ``-damp`` tail and are
    # captured in ``regions`` above.
    rayleighs: "list[tuple[float, float, float, float]]" = field(
        default_factory=list,
    )
    # SSI-2.E mutators (resolved targets, emit order).
    remove_sps: "list[tuple[int, int]]" = field(default_factory=list)
    remove_elements: "list[int]" = field(default_factory=list)
    #: SANISAND stage flips — ``(mat_tag, stage)`` pairs, emit order.
    update_material_stages: "list[tuple[int, int]]" = field(
        default_factory=list,
    )
    # Per-stage analysis chain + analyze loop (the ``_write_analysis``
    # attr shape, scoped to this stage).
    chain_attrs: "dict[str, Any]" = field(default_factory=dict)
    analyze_call: "tuple[int, float | None] | None" = None
    domain_changed: bool = False
    # Presence-captured time-state mutators (None == never called).
    set_time: "float | None" = None
    set_creep_on: "bool | None" = None
    pre_analyze_reset: bool = False
    # Declarative complement — attached by ``set_stage_records``.
    activated_pgs: "tuple[str, ...]" = ()
    initial_stress_records: "tuple[Any, ...]" = ()
    activate_absorbing_records: "tuple[Any, ...]" = ()
    #: ``s.update_parameter`` / ``s.zero_velocities`` records (schema
    #: 2.27.0, K1-8), attached by ``set_stage_records``.
    update_parameter_records: "tuple[Any, ...]" = ()
    zero_velocity_records: "tuple[Any, ...]" = ()
    # K1-8 (schema 2.27.0): the parameter tags the build planned, as the
    # in-band calls handed them in. ``ramp_tags`` maps an initial
    # stress's name to its (XX, YY, ZZ) ramp tags; ``flip_calls`` and
    # ``update_calls`` hold one ``(rank, pid)`` per call (``rank`` ``-1``
    # outside a partition bracket), and ``update_calls`` also the call's
    # ``(args, value)`` for the cross-check. ``node_kinematics_calls``
    # counts the ``setNodeVel`` / ``setNodeAccel`` calls.
    ramp_tags: "dict[str, tuple[int, int, int]]" = field(default_factory=dict)
    flip_calls: "list[tuple[int, int]]" = field(default_factory=list)
    update_calls: "list[tuple[int, int, tuple[str | int, ...], float]]" = (
        field(default_factory=list))
    node_kinematics_calls: int = 0
    # The stored tags, one entry per declarative record above
    # (``set_stage_records`` attributes the calls; the echo path
    # restores them): an initial stress's three ramp tags, and every
    # ``(rank, pid)`` a flip or update was written with.
    initial_stress_param_tags: "tuple[tuple[int, int, int], ...]" = ()
    absorbing_param_tags: "tuple[tuple[tuple[int, int], ...], ...]" = ()
    update_parameter_param_tags: "tuple[tuple[tuple[int, int], ...], ...]" = ()

    def next_emit_index(self) -> int:
        """Monotone per-stage sequence number (1-based, emit order)."""
        self.emit_seq += 1
        return self.emit_seq


def _ramp_tags_by_name(
    records: "Sequence[Any]",
    held: "Mapping[str, tuple[int, int, int]]",
    where: str,
) -> "tuple[tuple[int, int, int], ...]":
    """Each initial-stress record's ramp tags, read from ``held`` by name
    (schema 2.27.0, K1-8).

    ``held`` is what the scope's ``step_hook_ramp`` calls handed in. A
    record with no ramp, or a ramp with no record, is capture drift and
    refuses: the archive would store a tag its record never had.
    """
    names = [str(rec.name) for rec in records]
    if len(set(names)) != len(names) or set(names) != set(held):
        raise RuntimeError(
            f"H5Emitter: {where} declares initial stress {sorted(names)} "
            f"but its step_hook_ramp calls named {sorted(held)}; each record "
            "stores the tags of its own ramp (schema 2.27.0)."
        )
    return tuple(held[name] for name in names)


def _attribute_param_calls(
    where: str,
    verb: str,
    records: "Sequence[Any]",
    calls: "Sequence[tuple[int, int]]",
) -> "tuple[tuple[tuple[int, int], ...], ...]":
    """Each record's ``(rank, pid)`` rows, from the stage's calls in order
    (schema 2.27.0, K1-8).

    The build gives each flip or update record one ``parameter`` tag,
    minted in record order (the canonical flat walk,
    ``tag_plan._plan_parameters``), and a partitioned emit writes that tag
    on every rank holding the record's elements. So the distinct tags,
    ascending, are the records in order. A record that addresses no
    element has no tag and no call, and leaves the attribution ambiguous:
    that refuses rather than store a tag under the wrong record.
    """
    pids = sorted({int(pid) for _rank, pid in calls})
    if len(pids) != len(records):
        targets = ", ".join(_param_target(rec) for rec in records)
        raise RuntimeError(
            f"H5Emitter.set_stage_records: {where}: s.{verb} declares "
            f"{len(records)} record(s) (targets: {targets}) but its calls "
            f"carry {len(pids)} distinct parameter tag(s) {pids}, so a "
            "target holds no element. Each record stores its own tag "
            "(schema 2.27.0), so the tags cannot be attributed; drop the "
            "record whose target is empty, or use ops.tcl(path) / "
            "ops.py(path)."
        )
    slot = {pid: k for k, pid in enumerate(pids)}
    rows: "list[list[tuple[int, int]]]" = [[] for _ in records]
    for rank, pid in calls:
        rows[slot[int(pid)]].append((int(rank), int(pid)))
    return tuple(tuple(r) for r in rows)


def _param_target(rec: Any) -> str:
    """A flip or update record's target, as the user wrote it."""
    if rec.elements is not None:
        return f"elements={list(rec.elements)!r}"
    return f"pg={rec.pg!r}"


def _with_param_rows(
    stage: str,
    verb: str,
    records: "Sequence[Any]",
    param_tags: "Sequence[tuple[tuple[int, int], ...]]",
) -> "list[tuple[Any, tuple[tuple[int, int], ...]]]":
    """Pair each flip or update record with its stored ``(rank, pid)``
    rows; a record without its rows refuses (a 2.27.0 archive never
    leaves its replay a tag to mint).

    An empty run (no row) is a record that wrote no flip: the rewrite of a
    pre-2.27 archive whose record targets no element. The authoring path
    never makes one, since :func:`_attribute_param_calls` refuses such a
    record first; the replay checks that the record still resolves no
    element."""
    if len(param_tags) != len(records):
        raise RuntimeError(
            f"H5Emitter._write_stages: stage {stage!r} holds "
            f"{len(records)} {verb} record(s) but {len(param_tags)} "
            "param_tags entries; every record stores its tag rows "
            "(schema 2.27.0)."
        )
    for rows in param_tags:
        if rows and len({pid for _rank, pid in rows}) != 1:
            raise RuntimeError(
                f"H5Emitter._write_stages: stage {stage!r}: a {verb} "
                f"record's param_tags rows {tuple(rows)!r} do not name one "
                "parameter tag."
            )
    return list(zip(records, param_tags))


def _write_param_rows(
    group: Any, rows: "Sequence[tuple[int, int]]",
) -> None:
    """Write a record's ``param_tags`` (k, 2) int64 dataset: one
    ``(rank, pid)`` row per call (``rank`` ``-1`` for a flat build)."""
    import numpy as np

    group.create_dataset(
        "param_tags",
        data=np.asarray(
            [[int(r), int(p)] for r, p in rows], dtype=np.int64,
        ).reshape(len(rows), 2),
    )


def _check_update_calls(
    where: str,
    records: "Sequence[Any]",
    param_tags: "Sequence[tuple[tuple[int, int], ...]]",
    calls: "Sequence[tuple[int, int, tuple[str | int, ...], float]]",
) -> None:
    """Every ``updateParameter`` call says what its attributed record says
    (its name, material tag and value), or the attribution is drift."""
    owner = {
        pid: rec for rec, rows in zip(records, param_tags)
        for _rank, pid in rows}
    for _rank, pid, args, value in calls:
        rec = owner[int(pid)]
        want = ((rec.name,) if rec.mat_tag is None
                else (rec.name, int(rec.mat_tag)))
        if tuple(args) != want or float(value) != float(rec.value):
            raise RuntimeError(
                f"H5Emitter.set_stage_records: {where}: updateParameter "
                f"{pid} writes {tuple(args)!r} = {value!r}, but the record "
                f"its tag belongs to is {want!r} = {rec.value!r} — the emit "
                "stream and the build diverged."
            )


def _merge_node_region_args(
    old: "tuple[int | float | str, ...]",
    new: "tuple[int | float | str, ...]",
) -> "tuple[int | float | str, ...] | None":
    """Union two ``("-node", n1, n2, ...)`` region arg tails (P5.1).

    Returns the merged tail (first-occurrence member order preserved,
    duplicates dropped) or ``None`` when either tail is not the plain
    ``-node`` member-list form — the caller then appends the record
    unmerged rather than corrupting a flag tail it cannot parse.
    """
    def _is_node_form(t: "tuple[int | float | str, ...]") -> bool:
        return (
            len(t) >= 1
            and t[0] == "-node"
            and all(isinstance(v, int) for v in t[1:])
        )

    if not _is_node_form(old) or not _is_node_form(new):
        return None
    seen = set(old[1:])
    merged: "list[int | float | str]" = list(old)
    for v in new[1:]:
        if v not in seen:
            seen.add(v)
            merged.append(v)
    return tuple(merged)


def _partition_blocks_to_ro(
    blocks: "Sequence[_PartitionEmitBlock]",
) -> "list[Any]":
    """Freeze writer partition blocks into read-side
    ``PartitionEmittedRecord`` values (P5.0b).

    Used by ``OpenSeesModel.from_compose_buffers`` so a model
    materialised straight from emitter buffers carries the same
    partition view a write-then-``from_h5`` round trip would produce.
    ``boundary_node_ids`` is computed with the same one-pass symmetric
    intersection :meth:`H5Emitter._write_partitions` uses.
    """
    from .h5_reader import PartitionEmittedRecord

    node_sets = [set(blk.node_ids) for blk in blocks]
    out: "list[Any]" = []
    for idx, blk in enumerate(blocks):
        others: set[int] = set()
        for other_idx, other_set in enumerate(node_sets):
            if other_idx == idx:
                continue
            others.update(other_set)
        out.append(PartitionEmittedRecord(
            rank=int(blk.rank),
            element_ids=tuple(int(e) for e in blk.element_ids),
            node_ids=tuple(int(n) for n in blk.node_ids),
            boundary_node_ids=tuple(
                int(n) for n in blk.node_ids if n in others
            ),
        ))
    return out


def _stage_block_to_ro(blk: "_StageEmitBlock") -> "Any":
    """Freeze a capture bucket into a read-side ``StageRecordRO``.

    Used by ``OpenSeesModel.from_compose_buffers`` so a model
    materialised straight from emitter buffers carries the same
    staged view a write-then-``from_h5`` round trip would produce.
    The bucket's declarative complement must already be attached
    (``set_stage_records`` ran).
    """
    from .._internal.typed_records import StageRecordRO

    if blk.analyze_call is None:
        raise RuntimeError(
            f"stage {blk.name!r}: no analyze call captured — "
            "malformed bracket cannot freeze to a StageRecordRO."
        )
    steps, dt = blk.analyze_call
    return StageRecordRO(
        name=blk.name,
        analyze_steps=int(steps),
        analyze_dt=dt,
        set_time=blk.set_time,
        set_creep_on=blk.set_creep_on,
        pre_analyze_reset=blk.pre_analyze_reset,
        domain_changed=blk.domain_changed,
        activated_pgs=tuple(blk.activated_pgs),
        owned_node_ids=tuple(blk.owned_node_ids),
        owned_element_ids=tuple(blk.owned_element_ids),
        fixes=tuple(blk.fixes),
        masses=tuple(blk.masses),
        regions=tuple(blk.regions),
        region_seq=tuple(blk.region_seq),
        equal_dofs=tuple(blk.equal_dofs),
        rigid_links=tuple(blk.rigid_links),
        rigid_diaphragms=tuple(blk.rigid_diaphragms),
        embedded_nodes=tuple(blk.embedded_nodes),
        equal_dof_seq=tuple(blk.equal_dof_seq),
        rigid_link_seq=tuple(blk.rigid_link_seq),
        rigid_diaphragm_seq=tuple(blk.rigid_diaphragm_seq),
        embedded_node_seq=tuple(blk.embedded_node_seq),
        patterns=tuple(blk.patterns_complete),
        pattern_seq=tuple(blk.pattern_seq),
        recorders=tuple(blk.recorders),
        rayleighs=tuple(blk.rayleighs),
        rayleigh_seq=tuple(blk.rayleigh_seq),
        remove_sps=tuple(blk.remove_sps),
        remove_elements=tuple(blk.remove_elements),
        update_material_stages=tuple(blk.update_material_stages),
        chain_attrs=dict(blk.chain_attrs),
        initial_stress=tuple(blk.initial_stress_records),
        activate_absorbing=tuple(
            (rec.pg, None if rec.elements is None else tuple(rec.elements))
            for rec in blk.activate_absorbing_records
        ),
        update_parameters=tuple(blk.update_parameter_records),
        zero_velocities=tuple(blk.zero_velocity_records),
        initial_stress_param_tags=tuple(blk.initial_stress_param_tags),
        absorbing_param_tags=tuple(blk.absorbing_param_tags),
        update_parameter_param_tags=tuple(blk.update_parameter_param_tags),
    )


def _ro_to_stage_block(ro: "Any") -> "_StageEmitBlock":
    """Thaw a ``StageRecordRO`` back into a capture bucket — the echo
    half of the ``from_h5 → to_h5`` round trip.  Field-for-field
    inverse of :func:`_stage_block_to_ro`; the writer then re-emits
    equivalent stage bytes (hash stability is the acceptance test,
    not raw file bytes — attr creation order may differ)."""
    from .._internal.build.records import ActivateAbsorbingRecord

    blk = _StageEmitBlock(name=str(ro.name))
    blk.analyze_call = (
        int(ro.analyze_steps),
        None if ro.analyze_dt is None else float(ro.analyze_dt),
    )
    blk.set_time = ro.set_time
    blk.set_creep_on = ro.set_creep_on
    blk.pre_analyze_reset = bool(ro.pre_analyze_reset)
    blk.domain_changed = bool(ro.domain_changed)
    blk.activated_pgs = tuple(ro.activated_pgs)
    blk.owned_node_ids = list(ro.owned_node_ids)
    blk.owned_element_ids = list(ro.owned_element_ids)
    blk.fixes = list(ro.fixes)
    blk.masses = list(ro.masses)
    blk.regions = list(ro.regions)
    blk.region_seq = list(ro.region_seq)
    blk.equal_dofs = list(ro.equal_dofs)
    blk.rigid_links = list(ro.rigid_links)
    blk.rigid_diaphragms = list(ro.rigid_diaphragms)
    blk.embedded_nodes = list(ro.embedded_nodes)
    blk.equal_dof_seq = list(ro.equal_dof_seq)
    blk.rigid_link_seq = list(ro.rigid_link_seq)
    blk.rigid_diaphragm_seq = list(ro.rigid_diaphragm_seq)
    blk.embedded_node_seq = list(ro.embedded_node_seq)
    blk.patterns_complete = list(ro.patterns)
    blk.pattern_seq = list(ro.pattern_seq)
    blk.recorders = list(ro.recorders)
    blk.rayleighs = list(ro.rayleighs)
    blk.rayleigh_seq = list(ro.rayleigh_seq)
    blk.remove_sps = [(int(n), int(d)) for n, d in ro.remove_sps]
    blk.remove_elements = [int(t) for t in ro.remove_elements]
    blk.update_material_stages = [
        (int(m), int(v)) for m, v in ro.update_material_stages
    ]
    blk.chain_attrs = dict(ro.chain_attrs)
    blk.initial_stress_records = tuple(ro.initial_stress)
    blk.activate_absorbing_records = tuple(
        ActivateAbsorbingRecord(
            pg=pg, elements=None if elements is None else tuple(elements),
        )
        for pg, elements in ro.activate_absorbing
    )
    blk.update_parameter_records = tuple(ro.update_parameters)
    blk.zero_velocity_records = tuple(ro.zero_velocities)
    blk.initial_stress_param_tags = tuple(ro.initial_stress_param_tags)
    blk.absorbing_param_tags = tuple(ro.absorbing_param_tags)
    blk.update_parameter_param_tags = tuple(ro.update_parameter_param_tags)
    return blk
