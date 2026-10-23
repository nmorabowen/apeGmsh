"""stage_emit: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
)

from ..tag_allocator import TagAllocator
from .errors import BridgeError, TagPlanMiss, plan_owner
from .records import (
    ActivateAbsorbingRecord,
    InitialStressRecord,
    UpdateParameterRecord,
    ZeroVelocityRecord,
)
from .pg_expand import expand_pg_to_elements
from .ownership import (
    FemToOpsTagMap,
    SortedIntToInt,
    resolve_initial_stress_elements,
)


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from ..tag_plan import TagPlan


class DeclCursor(Protocol):
    """Opens the declaration of the record whose calls follow (K1-6).

    The bridge's ``BuiltModel._decl_cursor`` gives an archival emitter one
    that forwards to ``H5Emitter.set_declaration``; a deck's does nothing.
    The writers below bracket each record's calls with :meth:`open` /
    :meth:`close`, so ``/opensees/program`` records the declaration that
    made each ``step_hook_ramp``, ``addToParameter``,
    ``flip_element_stage``, ``update_parameter``, ``set_node_vel`` and
    ``set_node_accel`` call: a proof at emit time, as for ``fix``.
    """

    def open(self, owner: object) -> None:
        """The calls that follow are ``owner``'s declaration's."""

    def close(self) -> None:
        """The calls that follow belong to no declaration."""


class _NoDeclCursor:
    """The cursor that opens nothing: a deck, and the archive replays of
    ``compose.py``, whose records carry no declaration row."""

    __slots__ = ()

    def open(self, owner: object) -> None:
        del owner

    def close(self) -> None:
        pass


#: The no-op :class:`DeclCursor`, the default where ``compose.py`` calls in.
NO_DECL_CURSOR: DeclCursor = _NoDeclCursor()


class ParameterSite(NamedTuple):
    """One place an emit declares ``parameter`` tags.

    ``record`` declares them, ``rank`` is the runtime rank whose
    ``partition_open`` block writes them (``None`` outside any block),
    ``verb`` is the emitter verb that writes them, and ``n_tags`` is how
    many it declares there: three for an initial stress (its XX, YY and
    ZZ ramps), one for a stage flip or update that resolves an element on
    that rank, none for one that resolves none. ``ele_tags`` are the
    OpenSees element tags a flip or update addresses on that rank, as the
    plan resolved them (empty for an initial stress).
    """

    record: Any
    rank: int | None
    verb: str
    n_tags: int
    ele_tags: tuple[int, ...] = ()


class PlannedParameter(NamedTuple):
    """One :class:`ParameterSite` as the tag plan numbers it.

    ``ele_tags`` are the site's resolved element tags, which the flip and
    update writers read instead of resolving the record again.
    """

    record: Any
    rank: int | None
    verb: str
    tags: tuple[int, ...]
    ele_tags: tuple[int, ...] = ()


#: The verbs that declare ``parameter`` tags, and how many each declares
#: per site: an initial stress's ramp (XX, YY, ZZ), and the one-shot
#: absorbing flip and ``s.update_parameter`` (one, or none on a rank that
#: holds none of the record's elements).
PARAMETER_VERBS: dict[str, frozenset[int]] = {
    "step_hook_ramp": frozenset({3}),
    "flip_element_stage": frozenset({0, 1}),
    "update_parameter": frozenset({0, 1}),
}


def initial_stress_sites(
    records: "Iterable[InitialStressRecord]",
) -> list[ParameterSite]:
    """The parameter site of each initial stress: three ramp tags, written
    once, outside any partition block."""
    return [ParameterSite(rec, None, "step_hook_ramp", 3) for rec in records]


def parameter_flip_sites(
    verb: str,
    records: "Sequence[ActivateAbsorbingRecord | UpdateParameterRecord]",
    ele_tags: "Sequence[tuple[int, ...]]",
    rank: int | None,
) -> list[ParameterSite]:
    """The parameter site of each flip or update record on ``rank``: one
    tag if it resolves an element there (``ele_tags``), else none."""
    if verb not in ("flip_element_stage", "update_parameter"):
        raise BridgeError(f"parameter_flip_sites: unknown verb {verb!r}.")
    if len(records) != len(ele_tags):
        raise BridgeError(
            "parameter_flip_sites: one element list per record is needed.")
    return [
        ParameterSite(rec, rank, verb, 1 if tags_ else 0, tuple(tags_))
        for rec, tags_ in zip(records, ele_tags)
    ]


def plan_parameters(
    sites: "Iterable[ParameterSite]", tags: TagAllocator,
) -> tuple[PlannedParameter, ...]:
    """Mint each site's ``parameter`` tags, in order.

    The allocation loop of the initial-stress, absorbing-flip and
    ``update_parameter`` emits, moved out of them (ADR 0114 D4, amended).
    The build's tag plan runs it once per emit mode over every site the
    mode's emit writes (``tag_plan._plan_parameters``); the two waived
    replay writers (:func:`replay_initial_stress_global`,
    :func:`replay_activate_absorbing`) run it over their own sites.
    """
    out: list[PlannedParameter] = []
    for site in sites:
        if site.n_tags not in PARAMETER_VERBS.get(site.verb, ()):
            raise BridgeError(
                f"plan_parameters: a {site.verb!r} site declares "
                f"{site.n_tags} parameter tags; the verbs and their counts "
                f"are {dict(PARAMETER_VERBS)}."
            )
        minted: list[int] = []
        for _ in range(site.n_tags):
            minted.append(tags.allocate("parameter"))
        out.append(PlannedParameter(
            site.record, site.rank, site.verb, tuple(minted), site.ele_tags))
    return tuple(out)


def number_parameters(
    sites: "Iterable[ParameterSite]",
    canonical: "Iterable[PlannedParameter]",
) -> tuple[PlannedParameter, ...]:
    """Each site's ``parameter`` tags, read from the canonical walk.

    ``canonical`` is :func:`plan_parameters`'s result over the flat
    emit's sites, one per record. A partitioned emit writes a flip or an
    update once per rank that holds its elements, each time with the tag
    the flat walk gave its record, so a tag is rank-invariant. A site
    that declares tags its record holds none of (or holds for another
    verb) raises :class:`TagPlanMiss`.
    """
    by_record: dict[int, PlannedParameter] = {}
    for line in canonical:
        if by_record.setdefault(id(line.record), line) is not line:
            raise TagPlanMiss(
                f"number_parameters: the canonical walk holds "
                f"{plan_owner(line.record)} twice; it declares one site per "
                "record (ADR 0114 D4, amended)."
            )
    out: list[PlannedParameter] = []
    for site in sites:
        if site.n_tags not in PARAMETER_VERBS.get(site.verb, ()):
            raise BridgeError(
                f"number_parameters: a {site.verb!r} site declares "
                f"{site.n_tags} parameter tags; the verbs and their counts "
                f"are {dict(PARAMETER_VERBS)}."
            )
        held: tuple[int, ...] = ()
        if site.n_tags:
            minted = by_record.get(id(site.record))
            if (minted is None or minted.record is not site.record
                    or minted.verb != site.verb
                    or len(minted.tags) != site.n_tags):
                raise TagPlanMiss(
                    f"the {site.verb!r} site of {plan_owner(site.record)} at "
                    f"rank {site.rank} declares {site.n_tags} parameter "
                    "tag(s) the canonical (flat-order) walk did not number "
                    "(ADR 0114 D4, amended)."
                )
            held = minted.tags
        out.append(PlannedParameter(
            site.record, site.rank, site.verb, held, site.ele_tags))
    return tuple(out)


def _planned_parameters(
    sites: "Sequence[ParameterSite]", tag_plan: "TagPlan",
) -> list[PlannedParameter]:
    """Each site as ``tag_plan`` numbers it, in order.

    The plan must hold each site's record at its rank, written by its
    verb, with as many tags as the site declares, or it raises
    :class:`TagPlanMiss` (ADR 0114 D4, amended).
    """
    return [tag_plan.parameters.line_at(site) for site in sites]


def write_planned_ramps(
    emitter: "Emitter", lines: "Iterable[PlannedParameter]",
    *, cursor: DeclCursor = NO_DECL_CURSOR,
) -> dict[str, tuple[int, int, int]]:
    """Write each planned initial-stress ramp; mint nothing.

    One :meth:`Emitter.step_hook_ramp` per line, under the line's three
    planned tags (XX, YY, ZZ), with the line's record's declaration open
    on ``cursor``. Returns ``{record_name: (xx, yy, zz)}``.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for line in lines:
        if line.verb != "step_hook_ramp" or len(line.tags) != 3:
            raise BridgeError(
                f"write_planned_ramps: a {line.verb!r} line with "
                f"{len(line.tags)} tags is not an initial-stress ramp.")
        rec = line.record
        xx_tag, yy_tag, zz_tag = line.tags
        targets = (
            (xx_tag, rec.sigma_xx * rec.lambda_install),
            (yy_tag, rec.sigma_yy * rec.lambda_install),
            (zz_tag, rec.sigma_zz * rec.lambda_install),
        )
        cursor.open(rec)
        try:
            emitter.step_hook_ramp(
                name=rec.name,
                targets=targets,
                n_steps_to_full=float(rec.ramp_steps),
                phase="before",
            )
        finally:
            cursor.close()
        out[rec.name] = (xx_tag, yy_tag, zz_tag)
    return out


def emit_initial_stress_global(
    records: "Iterable[InitialStressRecord]",
    emitter: "Emitter",
    tag_plan: "TagPlan",
    *, cursor: DeclCursor,
) -> dict[str, tuple[int, int, int]]:
    """Emit the global side of each :class:`InitialStressRecord`.

    For each record, takes its three parameter tags (XX, YY, ZZ) from
    the build's tag plan (:func:`_planned_parameters`), then calls
    :meth:`Emitter.step_hook_ramp` (:func:`write_planned_ramps`), which
    bundles the dispatcher boilerplate (once), the parameter
    declarations, the per-step proc, and the dispatcher registration.

    Returns the mapping ``{record_name: (xx_tag, yy_tag, zz_tag)}`` so
    the per-rank ``addToParameter`` fan-out (see
    :func:`emit_initial_stress_addtoparameter`) can reach the same
    tags without re-allocating.
    """
    return write_planned_ramps(emitter, _planned_parameters(
        initial_stress_sites(records), tag_plan), cursor=cursor)


def replay_initial_stress_global(
    records: "Iterable[InitialStressRecord]",
    emitter: "Emitter",
    tags: TagAllocator,
) -> dict[str, tuple[int, int, int]]:
    """:func:`emit_initial_stress_global`, numbering the ramps from ``tags``.

    The standalone ramp writer, for the deck replay alone (the
    ``initial-stress-replay`` and ``staged-replay-params`` waivers of the
    tag-law ledger): the archive stores the declarative record, not its
    parameter tags, so the replay plans them from its own allocator
    through :func:`plan_parameters`, the loop the tag plan runs.
    """
    return write_planned_ramps(emitter, plan_parameters(
        initial_stress_sites(records), tags))


def emit_initial_stress_addtoparameter(
    records: "Iterable[InitialStressRecord]",
    emitter: "Emitter",
    fem: "FEMData",
    name_to_param_tags: dict[str, tuple[int, int, int]],
    fem_eid_to_ops_tag: "FemToOpsTagMap",
    element_owner: "SortedIntToInt | None" = None,
    partition_rank: int | None = None,
    *, cursor: DeclCursor = NO_DECL_CURSOR,
) -> None:
    """Emit ``addToParameter`` for each element covered by each record.

    Each record's calls run with its declaration open on ``cursor``.

    Per-rank semantics: if ``partition_rank`` is supplied (MP-mode
    emission inside a ``partition_open`` block), elements are filtered
    against ``element_owner`` and only owned elements emit.  Single-
    partition / flat callers pass ``partition_rank=None`` and the
    filter is skipped.

    Elements whose FEM id is in ``fem_eid_to_ops_tag`` but does NOT
    match this rank (in MP mode) are silently skipped — they belong
    to a different rank.  Elements whose FEM id is ABSENT from
    ``fem_eid_to_ops_tag`` entirely (e.g. the user passed an
    ``elements=[bad_id]`` that doesn't match any registered Element
    primitive) raise :class:`BridgeError` in single-partition mode
    so the user sees the mistake (red-team M6).  Under MP the same
    eid might legitimately be missing on this rank because it's
    owned by another rank — there we keep the silent-skip behaviour.
    """
    components = (
        ("commitStressIncrementXX", 0),
        ("commitStressIncrementYY", 1),
        ("commitStressIncrementZZ", 2),
    )
    is_partitioned_mode = partition_rank is not None
    for rec in records:
        param_tags = name_to_param_tags[rec.name]
        cursor.open(rec)
        try:
            for eid in resolve_initial_stress_elements(rec, fem):
                if is_partitioned_mode and element_owner is not None:
                    owner = element_owner.get(int(eid))
                    if owner is None or owner != partition_rank:
                        continue
                ops_tag = fem_eid_to_ops_tag.get(int(eid))
                if ops_tag is None:
                    if is_partitioned_mode:
                        continue  # owned by another rank; silent skip OK.
                    raise BridgeError(
                        f"initial_stress {rec.name!r}: element id {int(eid)} "
                        "is not registered with any Element primitive "
                        "(would silently no-op the addToParameter response).  "
                        "Either remove it from ``elements=`` or register the "
                        "matching Element primitive via "
                        "``ops.element.<Type>(pg=...)``."
                    )
                for response, idx in components:
                    emitter.addToParameter(
                        int(param_tags[idx]), int(ops_tag), response,
                    )
        finally:
            cursor.close()


def _flip_ele_tags(
    rec: "ActivateAbsorbingRecord | UpdateParameterRecord",
    fem: "FEMData",
    fem_eid_to_ops_tag: "FemToOpsTagMap",
    element_owner: "SortedIntToInt | None",
    partition_rank: int | None,
    unregistered: "Callable[[int], str]",
) -> tuple[int, ...]:
    """The OpenSees tags of ``rec``'s elements that ``partition_rank`` flips.

    ``rec`` targets its elements by ``pg`` or explicit ``elements``. In MP
    mode (``partition_rank`` set) only this rank's owned elements count,
    and an eid absent from ``fem_eid_to_ops_tag`` is skipped (it lives on
    another rank). In single-partition mode an absent eid raises
    :class:`BridgeError` with the message ``unregistered(eid)`` (the user
    named an element no primitive emitted).
    """
    if rec.elements is not None:
        eids: tuple[int, ...] = rec.elements
    elif rec.pg is not None:
        eids = tuple(eid for eid, _conn in expand_pg_to_elements(fem, rec.pg))
    else:  # pragma: no cover — validated at the call site
        eids = ()
    is_partitioned_mode = partition_rank is not None
    ops_tags: list[int] = []
    for eid in eids:
        if is_partitioned_mode and element_owner is not None:
            owner = element_owner.get(int(eid))
            if owner is None or owner != partition_rank:
                continue
        ops_tag = fem_eid_to_ops_tag.get(int(eid))
        if ops_tag is None:
            if is_partitioned_mode:
                continue  # owned by another rank; silent skip OK.
            raise BridgeError(unregistered(int(eid)))
        ops_tags.append(int(ops_tag))
    return tuple(ops_tags)


def update_parameter_ele_tags(
    rec: "UpdateParameterRecord",
    fem: "FEMData",
    fem_eid_to_ops_tag: "FemToOpsTagMap",
    element_owner: "SortedIntToInt | None" = None,
    partition_rank: int | None = None,
) -> tuple[int, ...]:
    """The OpenSees tags ``rec``'s ``updateParameter`` addresses on
    ``partition_rank`` (:func:`_flip_ele_tags`)."""
    def unregistered(eid: int) -> str:
        return (
            f"update_parameter {rec.name!r}: element id {eid} "
            "is not registered with any Element primitive (the "
            "updateParameter would silently no-op).  Either drop it "
            "from elements= or declare the matching Element "
            "primitive via ops.element.<Type>(pg=...)."
        )
    return _flip_ele_tags(rec, fem, fem_eid_to_ops_tag, element_owner,
                          partition_rank, unregistered)


def absorbing_ele_tags(
    rec: "ActivateAbsorbingRecord",
    fem: "FEMData",
    fem_eid_to_ops_tag: "FemToOpsTagMap",
    element_owner: "SortedIntToInt | None" = None,
    partition_rank: int | None = None,
) -> tuple[int, ...]:
    """The OpenSees tags ``rec``'s stage flip addresses on
    ``partition_rank`` (:func:`_flip_ele_tags`)."""
    def unregistered(eid: int) -> str:
        return (
            f"activate_absorbing: element id {eid} is not "
            "registered with any Element primitive (the stage flip "
            "would silently no-op).  Emit the absorbing elements via "
            "``ops.element.absorbing_boundary(skin=...)`` first."
        )
    return _flip_ele_tags(rec, fem, fem_eid_to_ops_tag, element_owner,
                          partition_rank, unregistered)


def emit_update_parameters(
    records: "Iterable[UpdateParameterRecord]",
    emitter: "Emitter",
    tag_plan: "TagPlan",
    partition_rank: int | None = None,
    *, cursor: DeclCursor,
) -> None:
    """Emit ``s.update_parameter`` for each record.

    Same per-rank contract as :func:`emit_activate_absorbing`: the two
    verbs drive the same OpenSees primitive, only the argv tail and the
    value differ. Each (record, rank) that addresses an element writes its
    own ``parameter`` tag, so each block is self-contained and a later
    stage may re-declare. The tag and the elements it addresses on
    ``partition_rank`` (:func:`update_parameter_ele_tags`, resolved once,
    by the build's tag plan) are the plan's; :func:`write_planned_flips`
    writes them.
    """
    lines = tag_plan.parameters.flip_lines(
        "update_parameter", tuple(records), partition_rank)
    write_planned_flips(emitter, lines, [line.ele_tags for line in lines],
                        cursor=cursor)


def emit_activate_absorbing(
    records: "Iterable[ActivateAbsorbingRecord]",
    emitter: "Emitter",
    tag_plan: "TagPlan",
    partition_rank: int | None = None,
    *, cursor: DeclCursor,
) -> None:
    """Emit the absorbing-boundary stage flip for each record (ADR 0054 AB-3).

    For each record, the one-shot ``parameter`` / ``addToParameter ...
    stage`` / ``updateParameter 1`` / ``remove parameter`` block via
    :meth:`Emitter.flip_element_stage`, over the OpenSees tags of its
    elements (``pg`` or explicit ``elements``) on ``partition_rank``.

    The build's tag plan resolves those elements once
    (:func:`absorbing_ele_tags`): in MP mode (``partition_rank`` set) only
    the rank's owned elements are flipped, and an eid absent from the
    element plan is skipped (it lives on another rank); in single-partition
    mode an absent eid is a hard :class:`BridgeError` (the user named an
    element no primitive emitted). Each (record, rank) that flips an
    element writes its own ``parameter`` tag, which the plan gives it, so
    each block is self-contained; :func:`write_planned_flips` writes it.
    """
    lines = tag_plan.parameters.flip_lines(
        "flip_element_stage", tuple(records), partition_rank)
    write_planned_flips(emitter, lines, [line.ele_tags for line in lines],
                        cursor=cursor)


def replay_activate_absorbing(
    records: "Iterable[ActivateAbsorbingRecord]",
    emitter: "Emitter",
    fem: "FEMData",
    fem_eid_to_ops_tag: "FemToOpsTagMap",
    tags: TagAllocator,
) -> None:
    """:func:`emit_activate_absorbing` on a flat deck, numbering the flips
    from ``tags``.

    The standalone flip writer, for the staged deck replay alone (the
    ``staged-replay-params`` waiver of the tag-law ledger): the archive
    stores no parameter tag, so the replay resolves each record's elements
    (:func:`absorbing_ele_tags`) and plans its tag from its own allocator
    through :func:`plan_parameters`, the loop the tag plan runs.
    """
    records = tuple(records)
    resolved = [
        absorbing_ele_tags(rec, fem, fem_eid_to_ops_tag) for rec in records]
    lines = plan_parameters(parameter_flip_sites(
        "flip_element_stage", records, resolved, None), tags)
    write_planned_flips(emitter, lines, resolved)


def write_planned_flips(
    emitter: "Emitter",
    lines: "Sequence[PlannedParameter]",
    ele_tags: "Sequence[tuple[int, ...]]",
    *, cursor: DeclCursor = NO_DECL_CURSOR,
) -> None:
    """Write each planned absorbing flip or update; mint nothing.

    Each call runs with its line's record's declaration open on
    ``cursor``.

    ``ele_tags`` holds, per line, the OpenSees tags its record addresses
    on the line's rank (:func:`absorbing_ele_tags`,
    :func:`update_parameter_ele_tags`). A line with a tag writes one
    :meth:`Emitter.flip_element_stage` or :meth:`Emitter.update_parameter`
    under it; a line without one (no element on its rank) writes nothing.
    """
    if len(lines) != len(ele_tags):
        raise BridgeError(
            "write_planned_flips: one element list per planned line is "
            "needed.")
    for line, ops_tags in zip(lines, ele_tags):
        if line.verb not in ("flip_element_stage", "update_parameter"):
            raise BridgeError(
                f"write_planned_flips: unknown verb {line.verb!r}; it "
                "writes flip_element_stage and update_parameter lines.")
        if len(line.tags) != (1 if ops_tags else 0):
            raise BridgeError(
                f"write_planned_flips: a {line.verb!r} line holds "
                f"{len(line.tags)} tags for {len(ops_tags)} elements; a "
                "line takes one tag iff it addresses an element.")
        if not line.tags:
            continue
        (pid,) = line.tags
        rec = line.record
        cursor.open(rec)
        try:
            if line.verb == "flip_element_stage":
                emitter.flip_element_stage(pid, ops_tags)
                continue
            args: tuple[str | int, ...] = (
                (rec.name,) if rec.mat_tag is None
                else (rec.name, int(rec.mat_tag))
            )
            emitter.update_parameter(pid, ops_tags, args, float(rec.value))
        finally:
            cursor.close()


def zero_velocity_targets_by_record(
    records: "Iterable[ZeroVelocityRecord]",
    all_node_ids: "Iterable[int]",
) -> "list[tuple[ZeroVelocityRecord, list[int]]]":
    """Resolve a stage's ``s.zero_velocities`` pool, record by record.

    ``nodes=None`` expands to ``all_node_ids`` (the whole domain). Each
    record keeps the nodes no earlier record named, in first-seen order,
    so a node named by two records is the first one's (the rule of
    ``BuiltModel._claim_ghost_fixes``) and the deck zeroes it once.
    Concatenated, the lists are :func:`zero_velocity_target_nodes`.
    """
    out: list[tuple[ZeroVelocityRecord, list[int]]] = []
    seen: set[int] = set()
    domain: tuple[int, ...] | None = None
    for rec in records:
        if rec.nodes is None:
            if domain is None:
                domain = tuple(int(n) for n in all_node_ids)
            targets: "Iterable[int]" = domain
        else:
            targets = rec.nodes
        mine: list[int] = []
        for nid in targets:
            n = int(nid)
            if n not in seen:
                seen.add(n)
                mine.append(n)
        out.append((rec, mine))
    return out


def zero_velocity_target_nodes(
    records: "Iterable[ZeroVelocityRecord]",
    all_node_ids: "Iterable[int]",
) -> "list[int]":
    """Resolve a stage's ``s.zero_velocities`` pool to its node list.

    ``nodes=None`` expands to ``all_node_ids`` (the whole domain).
    Order is first-seen; duplicates across records collapse, so calling
    the verb twice on overlapping sets does not double the deck.
    """
    return [n for _rec, nodes in zero_velocity_targets_by_record(
        records, all_node_ids) for n in nodes]


def emit_zero_velocities(
    nodes: "Iterable[int]",
    emitter: "Emitter",
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
) -> None:
    """Emit the nodal velocity / acceleration zeroing for ``nodes``.

    Per node, one ``setNodeVel`` then one ``setNodeAccel`` per DOF of
    that node's effective ndf (``effective_ndf`` is the ADR 0048
    inferred map; nodes absent from it fall back to the ``ops.model``
    envelope).  Both commands carry ``-commit`` — see
    :meth:`Emitter.set_node_vel` for why that is mandatory rather than
    cosmetic.
    """
    for nid in nodes:
        node = int(nid)
        ndf = int(effective_ndf.get(node, envelope_ndf))
        for dof in range(1, ndf + 1):
            emitter.set_node_vel(node, dof, 0.0)
        for dof in range(1, ndf + 1):
            emitter.set_node_accel(node, dof, 0.0)


def emit_zero_velocities_by_record(
    targets: "Iterable[tuple[ZeroVelocityRecord, Sequence[int]]]",
    emitter: "Emitter",
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
    *, cursor: DeclCursor,
) -> None:
    """:func:`emit_zero_velocities` over each record's nodes
    (:func:`zero_velocity_targets_by_record`), with the record's
    declaration open on ``cursor``. A record with no node left writes
    nothing; the deck is :func:`emit_zero_velocities` over the
    concatenated nodes."""
    for rec, nodes in targets:
        if not nodes:
            continue
        cursor.open(rec)
        try:
            emit_zero_velocities(nodes, emitter, effective_ndf, envelope_ndf)
        finally:
            cursor.close()


#: The nDMaterial classes whose flip to a plastic stage runs a
#: per-instance ``Elastic2Plastic()`` that ``updateMaterialStage`` gives
#: the first carrying element only.  ``stage/_builder.py`` re-exports
#: it as ``FIRST_ELEMENT_ONLY_PLASTIC_INIT``.
SWEPT_MATERIAL_CLASSES: frozenset[str] = frozenset(
    {"ManzariDafalias", "LadrunoSANISAND"},
)

#: The element type tokens, as the ``element`` verb receives them, whose
#: element hands ``materialState <tag>`` to every integration-point
#: material, mapped to where the nDMaterial tag sits in the verb's argv:
#: an index, or ``None`` for "the integer right before the first option
#: string" (``LadrunoUP``, whose node count varies with its shape).  The
#: tokens of ``stage/_builder.MATERIAL_STATE_ELEMENTS``, one per class
#: (``SixNodeTri`` emits ``tri6n``).
MATERIAL_STATE_ELEMENT_TOKENS: dict[str, int | None] = {
    "tri6n": 8,
    "FourNodeTetrahedron": 4,
    "SSPquad": 4,
    "SSPbrick": 8,
    "BezierTri6": 8,
    "BezierTet10": 10,
    "LadrunoUP": None,
}


#: The nDMaterial wrappers, by type token (the bridge class has the same
#: name), whose ``setParameter`` hands every name on to the material they
#: wrap, mapped to the argv index of that material's tag.  An element of
#: a wrapper over a swept material is a carrier of the swept material:
#: its ``materialState <tag>`` reaches the wrapped integration points.
#: Each entry is measured against the per-element reference
#: (``tests/opensees/integration/test_md_stage_route_elements.py``):
#: ``PlaneStrain`` on stock 3.8.0 and the fork, ``StagedStrain`` on the
#: fork build 79e062367.  ``LogStrain``, ``LogStrain2D`` and
#: ``InitDefGrad`` forward ``setParameter`` in the fork source too, but
#: are not listed: over ManzariDafalias the first two end the process on
#: that build, and ``InitDefGrad`` takes only a finite-strain inner.  The
#: stage close refuses a multi-element plastic flip reached through any
#: primitive outside this table, so it is the one resolution both sides
#: use.
MATERIAL_STATE_WRAPPERS: dict[str, int] = {
    "PlaneStrain": 0,
    "StagedStrain": 0,
}


class ForkBuildFloor(NamedTuple):
    """A Ladruno fork build floor that an emitted deck checks at its start.

    A deck cannot order two build shas, so the floor is proven by a
    ``probe`` command: one that every fork build at or after the floor
    registers, in the classic Tcl and the Python engines alike, while
    stock OpenSees and every older fork build lack it.  The deck that
    needs the floor stops with :func:`fork_floor_message` when the
    running build has no such command (#2504).
    """

    #: The command the running build must expose.
    probe: str
    #: The floor, as the message names it.
    floor: str
    #: What goes wrong below the floor, as the message names it.
    failure: str


#: The ``materialState`` forwarding of the Ladruno continuum solids.  Fork
#: commit ``64a0341a6`` (WP-124 C10) lets ``materialState`` through
#: ``LadrunoShell::isMaterialPointToken`` to ``forwardToMaterials``.
#: Without it these elements read the name as a per-point address and
#: drop it, so a flip reaches the first element only
#: (``updateMaterialStage``) or none (``setParameter``,
#: ``updateParameter``): measured on fork build 79e062367, and forwarding
#: bit for bit at ``FORK_PIN`` 100eb9ae.  ``ladrunoSchema`` enters the
#: fork in ``d9ff64c03``, a descendant of ``64a0341a6``, so every build
#: that registers it forwards.  A build that forwards but predates the
#: command is refused (fail closed).
LADRUNO_SOLID_MATERIAL_STATE_FLOOR = ForkBuildFloor(
    probe="ladrunoSchema",
    floor="a Ladruno fork build at or after FORK_PIN 100eb9ae "
          "(fork PR #859)",
    failure="the switch reaches the first element or none, and the run "
            "gives a wrong answer without a warning (apeGmsh issues #2445, "
            "#2504)",
)

#: Element class -> the fork build floor a ``materialState`` update of
#: one of its elements needs.  The five classes emit their class name as
#: their ``element`` token, so the key is both.  One table for both sides:
#: ``stage/_builder.py`` derives its fork-floor classes from it, and the
#: deck emitters guard on it (:class:`MaterialStageCarriers`).
MATERIAL_STATE_FORK_FLOOR: dict[str, ForkBuildFloor] = {
    cls: LADRUNO_SOLID_MATERIAL_STATE_FLOOR
    for cls in ("LadrunoQuad", "LadrunoCST", "LadrunoLST", "LadrunoBrick",
                "LadrunoBrick20")
}


def fork_floor_message(floor: ForkBuildFloor) -> str:
    """The error a deck stops with below ``floor``.

    One line of plain text, so both targets quote it as is: no Tcl
    substitution character and no backslash.
    """
    classes = ", ".join(sorted(
        cls for cls, f in MATERIAL_STATE_FORK_FLOOR.items() if f == floor))
    message = (
        f"apeGmsh: this deck switches materialState on {classes} "
        f"elements, which hand it to their materials only on "
        f"{floor.floor}. The running OpenSees has no {floor.probe} "
        "command, so it is stock OpenSees or an older fork build, where "
        f"{floor.failure}. Rebuild the Ladruno fork at or after FORK_PIN."
    )
    _refuse_tcl_syntax("fork_floor_message", message)
    return message


def _refuse_tcl_syntax(who: str, message: str) -> None:
    """Raise when ``message`` holds a character Tcl would substitute:
    both targets quote a guard message as is."""
    bad = sorted(set(message) & set('[]{}$"\\\n'))
    if bad:
        raise BridgeError(
            f"{who}: {bad!r} in the guard message would be read as Tcl "
            "syntax; reword it.")


#: The Ladruno fork commit that stops the ManzariDafalias constructors
#: resetting their class-wide stage flag.
MD_STATIC_FIX_COMMIT = "43b235bbb753113c3fdee7922e6e037c9ed138df"

#: Fork builds measured for that reset, by exact ``ladrunoBuild()`` sha:
#: does the build carry :data:`MD_STATIC_FIX_COMMIT`?  Ancestry checked on
#: the fork (GitHub compare); each row is measured by the live reset test
#: (``tests/opensees/stage/test_class_wide_stage_flag.py``).  A sha cannot
#: be ordered, so a build joins only once it is measured, and the deck
#: guard (:data:`MD_STATIC_RESET_GUARD`) refuses every fork build not
#: listed ``True``.  A ``FORK_PIN`` bump adds its row (a unit lock fails
#: otherwise).
FORK_BUILD_HAS_MD_STATIC_FIX: dict[str, bool] = {
    # FORK_PIN ladruno-v20261023, which merges the fix: no reset.
    "bdb69cea5a9fb17f474cb526e745ac5efb89b0be": True,
    # The maintainer's venv build, an ancestor of the fix: resets.
    "79e0623670c61c768357a01dff1477a4a0ef7e06": False,
}


class ForkBuildAllowlist(NamedTuple):
    """Ladruno fork builds that an emitted deck accepts by exact sha.

    The deck asks the running build for ``ladrunoBuild``.  A build without
    the command runs on: stock OpenSees, which this guard says is right.
    A build with it must answer one of :attr:`builds`, else the deck stops
    with :attr:`message`: an unknown, unmeasured or unanswering fork build
    is refused (fail closed).  Used where no command marks the fix, so no
    :class:`ForkBuildFloor` probe can (#2516).
    """

    #: The accepted ``ladrunoBuild()`` answers, 40 lowercase hex each.
    builds: tuple[str, ...]
    #: The error the deck stops with.
    message: str


#: A check an emitted deck runs on the build at its start.
ForkBuildGuard: TypeAlias = "ForkBuildFloor | ForkBuildAllowlist"


def _md_static_reset_message() -> str:
    message = (
        "apeGmsh: this deck builds an element or material after a stage "
        "flipped the class-wide stage flag ManzariDafalias::mElastFlag of "
        "ManzariDafalias and LadrunoSANISAND, and solves before flipping "
        "it again. A Ladruno fork build without fork commit "
        f"{MD_STATIC_FIX_COMMIT[:9]} resets that flag in the material "
        "constructor, so every one of these materials would go back to "
        "elastic with no warning (apeGmsh issues #2444, #2516). The "
        "running OpenSees is a fork build whose ladrunoBuild sha is not "
        "one measured to carry the fix. Rebuild the Ladruno fork at "
        "FORK_PIN, flip the whole group again after the activation, or "
        "activate these elements before the flip."
    )
    _refuse_tcl_syntax("MD_STATIC_RESET_GUARD", message)
    return message


def _known_fixed_builds() -> tuple[str, ...]:
    builds = tuple(sorted(
        sha for sha, fixed in FORK_BUILD_HAS_MD_STATIC_FIX.items() if fixed))
    for sha in builds:
        if len(sha) != 40 or set(sha) - set("0123456789abcdef"):
            raise BridgeError(
                f"FORK_BUILD_HAS_MD_STATIC_FIX: {sha!r} is not a 40-hex "
                "ladrunoBuild sha.")
    return builds


#: The guard a deck whose solve could see the ManzariDafalias constructor
#: reset runs (:meth:`MaterialStageCarriers.solve_fork_guards`).  The one
#: hook for a fork capability marker of the fix: when the fork
#: registers one, this becomes a :class:`ForkBuildFloor` on that command
#: (with its own message) and the sha table goes.
MD_STATIC_RESET_GUARD = ForkBuildAllowlist(
    builds=_known_fixed_builds(), message=_md_static_reset_message())


def fork_guard_lines(
    guard: ForkBuildGuard, dialect: 'Literal["py", "tcl"]',
) -> list[str]:
    """The deck lines that run ``guard`` in ``dialect``, unindented.

    A :class:`ForkBuildFloor` stops the deck when the build lacks its
    probe command; a :class:`ForkBuildAllowlist` when the build has
    ``ladrunoBuild`` and it does not answer a listed sha.  Any other
    object raises.
    """
    if isinstance(guard, ForkBuildFloor):
        message = fork_floor_message(guard)
        if dialect == "tcl":
            return [f'if {{[info commands {guard.probe}] eq ""}} {{',
                    f'    error "{message}"',
                    "}"]
        return [f"if not hasattr(ops, {guard.probe!r}):",
                f"    raise SystemExit({message!r})"]
    if isinstance(guard, ForkBuildAllowlist):
        if dialect == "tcl":
            return ['if {[info commands ladrunoBuild] ne ""} {',
                    "    if {[catch {ladrunoBuild} _apesees_build] || "
                    f"[lsearch -exact {{{' '.join(guard.builds)}}} "
                    "[string trim $_apesees_build]] < 0} {",
                    f'        error "{guard.message}"',
                    "    }",
                    "}"]
        return ["if hasattr(ops, 'ladrunoBuild'):",
                "    try:",
                "        _apesees_build = str(ops.ladrunoBuild()).strip()",
                "    except Exception:",
                "        _apesees_build = ''",
                f"    if _apesees_build not in {tuple(guard.builds)!r}:",
                f"        raise SystemExit({guard.message!r})"]
    raise BridgeError(
        f"fork_guard_lines: {type(guard).__name__} is not a fork build "
        "guard (ForkBuildFloor or ForkBuildAllowlist).")


def _carried_material_tag(
    token: str, argv: "Sequence[int | float | str]",
) -> int:
    """The nDMaterial tag an element of ``token`` carries in ``argv``.

    Raises :class:`BridgeError` when ``argv`` does not hold an integer
    where :data:`MATERIAL_STATE_ELEMENT_TOKENS` says the tag sits.
    """
    where = MATERIAL_STATE_ELEMENT_TOKENS[token]
    if where is None:
        first_str = next(
            (i for i, v in enumerate(argv) if isinstance(v, str)), None)
        where = -1 if first_str is None else first_str - 1
    value = argv[where] if 0 <= where < len(argv) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeError(
            f"element {token}: no nDMaterial tag at argv position {where} "
            f"of {tuple(argv)!r}; the stage-flip sweep cannot tell which "
            "material the element carries.  Update "
            "MATERIAL_STATE_ELEMENT_TOKENS with the element's grammar."
        )
    return int(value)


class MaterialStageCarriers:
    """Which emitted elements carry each swept material (#2445).

    Every deck emitter keeps one and feeds it its ``nDMaterial``,
    ``element``, ``remove_element`` and ``partition_open`` /
    ``partition_close`` calls.  ``updateMaterialStage`` alone reaches a
    material through ``MaterialStageParameter::setDomain``, which stops at
    the FIRST element that accepts it, so a :data:`SWEPT_MATERIAL_CLASSES`
    flip to a plastic stage would run ``Elastic2Plastic()`` on that
    element's integration points only.  :meth:`flip_commands` therefore
    precedes the flip of such a material on two or more carriers with::

        setParameter -val <stage> -ele <every carrier> materialState <tag>

    ``setParameter`` with an element list builds an
    ``ElementStateParameter`` that offers ``materialState <tag>`` to each
    listed element and updates each one that accepts it, so every
    integration-point material of the tag runs its own stage update,
    ``Elastic2Plastic()`` included.  OpenSees numbers the transient
    parameter itself and removes it, as ``updateMaterialStage`` does, so
    the deck carries no parameter tag.  The list names the carriers only:
    DruckerPrager, BoundingCamClay and the stress-density models accept
    ``materialState`` without checking the tag, so a sweep that reached
    their elements would switch them too.  An element of another rank is
    skipped by OpenSees (it is not in that rank's domain).

    The ``updateMaterialStage`` line still follows every flip: it sets the
    class-wide flag, reaches the elements that refresh their stage caches
    only on it (``LadrunoUP``), and its second ``Elastic2Plastic()`` on the
    first element repeats the first.  A stage ``0`` flip, a material of
    any other class, and a material on one carrier keep the single
    ``updateMaterialStage`` line.  A carrier of a class outside
    :data:`MATERIAL_STATE_ELEMENT_TOKENS` is not tracked: the stage close
    refuses a multi-element plastic flip of such a material.

    The carriers are the elements emitted before the flip, so an element
    a later stage activates is not listed; the archive replays the same
    calls in the same order, so its deck lists the same elements.

    It also notes each element of a :data:`MATERIAL_STATE_FORK_FLOOR`
    class.  :meth:`flip_fork_floors` and :meth:`fork_floors` name the
    floors a sweep or a ``materialState`` update of such an element
    needs, and the deck emitters check each at the deck's start
    (#2504).  The check is derived from the calls, like the lines it
    guards, so ``model_hash`` does not see it.

    And it watches for the ManzariDafalias constructor reset (#2516).
    On a fork build without :data:`MD_STATIC_FIX_COMMIT` the constructor
    of a ``FORK_RESETS_STAGE_STATIC`` class sets the class-wide stage
    flag to elastic, and every new element copies its material through
    it.  So an ``element`` or ``nDMaterial`` emitted after a user flip
    left the group at a non-zero stage, with no flip of the group again
    before the next solve verb, gives that build's answer and no other.
    :meth:`solve_fork_guards` then names :data:`MD_STATIC_RESET_GUARD`.
    Every element counts, whatever it carries: an element of an
    unknown grammar could copy the material, so the watch never misses
    one (fail closed), and a staged deck builds its sections and
    materials before its first stage block.  A flip of the group after
    the construction, and before the solve, makes every build agree
    (``Elastic2Plastic()`` runs on each flip, whatever the flag held).
    """

    def __init__(self) -> None:
        self._swept: set[int] = set()
        # wrapper tag -> the tag it wraps (MATERIAL_STATE_WRAPPERS only).
        self._wraps: dict[int, int] = {}
        # material tag -> rank (None: outside any partition block) ->
        # element tags, in emission order.
        self._carriers: dict[int, dict[int | None, dict[int, None]]] = {}
        self._rank: int | None = None
        # element tag -> its MATERIAL_STATE_FORK_FLOOR floor.
        self._floors: dict[int, ForkBuildFloor] = {}
        from ...stage._builder import (
            CLASS_WIDE_STAGE_FLAG,
            FORK_RESETS_STAGE_STATIC,
            STAGE_FLAG_PARAMETERS,
        )
        # The constructor-reset watch: the class-wide group of each
        # FORK_RESETS_STAGE_STATIC class, the stage a user flip left each
        # group at, and the groups a construction reset since.
        self._reset_classes: dict[str, str] = {
            cls: CLASS_WIDE_STAGE_FLAG[cls] for cls in FORK_RESETS_STAGE_STATIC}
        self._flag_params = STAGE_FLAG_PARAMETERS
        self._reset_group: dict[int, str] = {}
        self._group_stage: dict[str, int] = {}
        self._reset_pending: set[str] = set()

    def _note_construction(self) -> None:
        """An ``element`` or ``nDMaterial``: on a resetting build every
        group a user flip left non-zero is now elastic."""
        self._reset_pending.update(
            g for g, stage in self._group_stage.items() if stage != 0)

    def note_flip(self, mat_tag: int, stage: int) -> None:
        """A user flip of ``mat_tag`` to ``stage`` (``updateMaterialStage``
        or a stage-flag ``updateParameter``): it sets its group's flag on
        every build."""
        group = self._reset_group.get(int(mat_tag))
        if group is not None:
            self._group_stage[group] = int(stage)
            self._reset_pending.discard(group)

    def note_parameter(
        self, args: "Sequence[int | float | str]", value: float,
    ) -> None:
        """An ``updateParameter`` on ``args``: a stage-flag name sets the
        flag of the material it names.  One with no material tag reaches
        every material of its elements, so it may set any group's flag
        and heals none."""
        if not args or args[0] not in self._flag_params:
            return
        if len(args) > 1:
            self.note_flip(int(args[1]), int(value))
            return
        if int(value) != 0:
            for group in set(self._reset_group.values()):
                self._group_stage[group] = int(value)

    def solve_fork_guards(self) -> tuple[ForkBuildGuard, ...]:
        """The guards a solve verb needs: :data:`MD_STATIC_RESET_GUARD`
        once a construction reset a group on a resetting build and no
        flip set it again."""
        return (MD_STATIC_RESET_GUARD,) if self._reset_pending else ()

    def note_material(
        self, mat_type: str, tag: int,
        params: "Sequence[int | float | str]" = (),
    ) -> None:
        tag = int(tag)
        self._note_construction()
        if mat_type in self._reset_classes:
            self._reset_group[tag] = self._reset_classes[mat_type]
        else:
            self._reset_group.pop(tag, None)
        self._wraps.pop(tag, None)
        if mat_type in SWEPT_MATERIAL_CLASSES:
            self._swept.add(tag)
            return
        self._swept.discard(tag)
        if mat_type in MATERIAL_STATE_WRAPPERS:
            where = MATERIAL_STATE_WRAPPERS[mat_type]
            inner = params[where] if where < len(params) else None
            if isinstance(inner, bool) or not isinstance(inner, int):
                raise BridgeError(
                    f"nDMaterial {mat_type} {tag}: no wrapped material tag "
                    f"at argv position {where} of {tuple(params)!r}; the "
                    "stage-flip update cannot tell what it wraps.  Update "
                    "MATERIAL_STATE_WRAPPERS with the wrapper's grammar."
                )
            self._wraps[tag] = int(inner)

    def _resolve(self, mat: int) -> int:
        """The material ``mat`` wraps, through every forwarding wrapper."""
        seen = {mat}
        while mat in self._wraps:
            mat = self._wraps[mat]
            if mat in seen:  # pragma: no cover - OpenSees refuses a cycle
                break
            seen.add(mat)
        return mat

    def note_element(
        self, token: str, tag: int, argv: "Sequence[int | float | str]",
    ) -> None:
        self._note_construction()
        floor = MATERIAL_STATE_FORK_FLOOR.get(token)
        if floor is not None:
            self._floors[int(tag)] = floor
        if token not in MATERIAL_STATE_ELEMENT_TOKENS:
            return
        mat = self._resolve(_carried_material_tag(token, argv))
        if mat in self._swept:
            by_rank = self._carriers.setdefault(mat, {})
            by_rank.setdefault(self._rank, {})[int(tag)] = None

    def note_removed(self, tag: int) -> None:
        self._floors.pop(int(tag), None)
        for by_rank in self._carriers.values():
            for tags in by_rank.values():
                tags.pop(int(tag), None)

    def partition_open(self, rank: int) -> None:
        self._rank = int(rank)

    def partition_close(self) -> None:
        self._rank = None

    def _carriers_of(self, mat_tag: int) -> tuple[int, ...]:
        by_rank = self._carriers.get(int(mat_tag), {})
        if self._rank is None:
            ranks: "Iterable[int | None]" = by_rank
        else:
            ranks = (None, self._rank)
        out: dict[int, None] = {}
        for rank in ranks:
            out.update(by_rank.get(rank, {}))
        return tuple(out)

    def _sweep(self, tag: int, value: int) -> tuple[int, ...]:
        """The carriers a flip of ``tag`` to ``value`` updates one by
        one; empty when the flip is the bare ``updateMaterialStage``."""
        carriers = self._carriers_of(tag)
        if value < 1 or tag not in self._swept or len(carriers) < 2:
            return ()
        return carriers

    def fork_floors(
        self, ele_tags: "Iterable[int]",
    ) -> tuple[ForkBuildFloor, ...]:
        """The floors a ``materialState`` update of ``ele_tags`` needs
        (:data:`MATERIAL_STATE_FORK_FLOOR`), each once, in first-seen
        order."""
        out: dict[ForkBuildFloor, None] = {}
        for et in ele_tags:
            floor = self._floors.get(int(et))
            if floor is not None:
                out[floor] = None
        return tuple(out)

    def flip_fork_floors(
        self, mat_tag: int, stage: int,
    ) -> tuple[ForkBuildFloor, ...]:
        """The floors the flip :meth:`flip_commands` emits needs: those
        of its swept carriers.  The bare ``updateMaterialStage`` needs
        none: on one carrier it reaches the only element on any build."""
        return self.fork_floors(self._sweep(int(mat_tag), int(stage)))

    def flip_commands(
        self, mat_tag: int, stage: int,
    ) -> tuple[tuple[str, tuple[int | float | str, ...]], ...]:
        """The commands one flip of ``mat_tag`` to ``stage`` emits, as
        ``(command, argv)``; every target renders them alike."""
        tag = int(mat_tag)
        value = int(stage)
        ums = ("updateMaterialStage", ("-material", tag, "-stage", value))
        carriers = self._sweep(tag, value)
        if not carriers:
            return (ums,)
        return (
            ("setParameter", (
                "-val", float(value), "-ele", *carriers,
                "materialState", str(tag))),
            ums,
        )


#: One SP-constraint operation in a node's owner-side command stream,
#: replayed onto that node's ghost copies (ADR 0027 INV-2).  Either
#: ``("fix", flags)`` — the 0/1 fixity vector ``ops.fix`` / ``s.fix``
#: took — or ``("remove", dof)`` — the 1-based DOF index ``s.remove_sp``
#: took.  Ordered: a ``fix`` never releases, so only the sequence
#: distinguishes "fixed then freed" from "never fixed".
GhostSPOp: "TypeAlias" = tuple[str, Any]


def emit_ghost_sp_ops(
    emitter: "Emitter", tag: int, ops: "Iterable[GhostSPOp]",
) -> None:
    """Replay one node's SP-constraint op stream onto its ghost copy.

    ``("fix", flags)`` renders ``fix <tag> *flags``; ``("remove", dof)``
    renders ``remove sp <tag> <dof>``.  Replaying the owner's commands
    **in order** — rather than reducing them to a net fixity vector —
    is what keeps the ghost exact when a stage releases a constraint it
    set earlier: ``fix`` is additive per flagged DOF and never releases,
    so only the ordered stream carries the difference between "fixed
    then freed" and "never fixed".
    """
    for kind, payload in ops:
        if kind == "fix":
            emitter.fix(int(tag), *payload)
        else:
            emitter.remove_sp(int(tag), int(payload))
