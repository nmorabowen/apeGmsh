"""The build-time tag plan (ADR 0114 D4, amended).

A tag is written once, by the bridge's build, and the archive keeps it.
:func:`plan_tags` is where the build writes them: it seeds a planner
:class:`TagAllocator` from the registered primitives, plans every derived
tag family, and freezes the allocator. The emit paths then read the plan
instead of minting.

The migration moved each family's allocation loop out of its emit helper
and into this module one family at a time, so every step left the decks
byte-identical. A family not yet moved has a *pending* sub-plan, whose
:meth:`FamilyTagPlan.stream` raises. Every family is planned now, and the
emit holds the :class:`TagPlan` itself, which has no allocation API: its
helpers take the plan, read their family's rows, and mint nothing. An
owner the plan does not hold raises :class:`~.build.TagPlanMiss`, naming
the family and the owner.

The plan is keyed by :class:`TagMode`, the ``(split, partitioned,
staged)`` triple, because the split, partitioned and staged decks each
number their derived tags in their own order today. The plan for one
mode is what that mode's emit writes.

A plan row is ``(kind, tag)`` in the emit's verb vocabulary: the verb,
joined by ``:`` to its type token when the planner knows it
(``("region", 3)``). That is the vocabulary of the stream the emitter
records, so the oracle can compare the two. An element spec picks its
type token inside its own ``_emit``, so an element row carries the bare
verb, ``("element", 7)``, and the oracle compares element rows by verb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from .tag_allocator import TagAllocator, TagLawError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ..apesees import BuiltModel
    from .build import (
        ContactPlan,
        ElementPlanRows,
        InterfacePlan,
        MPElementPlan,
        ParameterSite,
        PlannedParameter,
        TransformFanout,
    )
    from .types import Element, GeomTransf

#: One planned emission: ``(kind, tag)`` in the emit's verb vocabulary.
TagRow = tuple[str, int]

#: A helper handed a fork of another model's plan says so first, rather
#: than report a count or order mismatch that would hide the cause.
_FOREIGN_PLAN = (
    "the {family} plan was made over another FEM snapshot than the one "
    "this emit walks: the tag plan is another model's. Emit each model "
    "through its own BuiltModel.emit (ADR 0114 D4, amended)."
)


def _miss(message: str) -> Exception:
    """A plan miss: the plan holds no row for an owner this emit writes.

    The error is a :class:`~.build.TagPlanMiss`, a :class:`~.build.BridgeError`
    and a :class:`TagLawError` at once; ``message`` names the family and
    the owner.
    """
    from .build import TagPlanMiss

    return TagPlanMiss(message)


class TagMode(NamedTuple):
    """The emit mode a plan is keyed by."""

    split: bool
    partitioned: bool
    staged: bool


def emit_mode(
    bm: BuiltModel, *, split: bool, supports_partitions: bool,
) -> TagMode:
    """The :class:`TagMode` that ``bm.emit(emitter, split=split)`` takes.

    ``supports_partitions`` is the emitter's own flag (an emitter that
    cannot drive OpenSeesMP brackets emits a partitioned mesh flat).
    ``split`` wins over partitioning, as in the emit dispatch.
    """
    from .build import is_partitioned

    return TagMode(
        split=bool(split),
        partitioned=(
            not split and supports_partitions and is_partitioned(bm.fem)
        ),
        staged=bool(bm.stage_records),
    )


# ---------------------------------------------------------------------------
# The per-family sub-plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FamilyTagPlan:
    """The tags one derived family is planned to write, in emit order.

    ``KINDS`` are the allocator kinds the family mints. ``MIGRATED`` turns
    ``True`` in the slice that moves the family's allocation loop into
    :func:`plan_tags`; until then the sub-plan is pending and holds no
    rows.
    """

    FAMILY: ClassVar[str]
    KINDS: ClassVar[frozenset[str]]
    MIGRATED: ClassVar[bool] = False

    rows: tuple[TagRow, ...] = ()

    def __post_init__(self) -> None:
        if not self.MIGRATED and self.rows:
            raise TagLawError(
                f"{self.FAMILY}: a pending sub-plan cannot carry rows; set "
                "MIGRATED in the slice that moves its allocation into "
                "plan_tags."
            )

    def stream(self) -> tuple[TagRow, ...]:
        """The family's planned ``(kind, tag)`` rows.

        Raises :class:`NotImplementedError` while the family is pending:
        its tags are still minted by the emit path, so the plan has no
        rows to give.
        """
        if not self.MIGRATED:
            raise NotImplementedError(
                f"tag family {self.FAMILY!r} is not planned yet: its tags "
                "are still minted at emit time (ADR 0114 D4, amended)."
            )
        return self.rows


@dataclass(frozen=True, slots=True)
class ElementTagPlan(FamilyTagPlan):
    """Element-spec fan-out: one tag per element instance of every spec.

    ``specs`` is :func:`~.build.allocate_element_tags`'s plan, made once
    by :func:`plan_tags`: each element spec, in the emit's topological
    order, with its columnar rows (FEM id, connectivity, tag). The emit
    paths read it instead of allocating. :meth:`stream` derives the
    ``("element", tag)`` rows on demand, so the plan holds no Python
    object per element.
    """

    FAMILY: ClassVar[str] = "elements"
    KINDS: ClassVar[frozenset[str]] = frozenset({"element"})
    MIGRATED: ClassVar[bool] = True

    specs: tuple[tuple[Element, ElementPlanRows], ...] = field(
        default=(), compare=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "elements: the element plan derives its rows from specs; "
                "pass specs, not rows."
            )

    def stream(self) -> tuple[TagRow, ...]:
        """One ``("element", tag)`` row per planned element, spec by spec."""
        return tuple(
            ("element", tag)
            for _spec, sub in self.specs
            for _eid, _conn, tag in sub
        )


@dataclass(frozen=True, slots=True)
class TransformTagPlan(FamilyTagPlan):
    """Orientation fan-out: one ``geomTransf`` per distinct ``vecxz``.

    ``fanout`` is :func:`~.build.plan_transform_specs`'s result, made once
    by :func:`plan_tags`: each transform spec, in the emit's topological
    order, with its planned lines, plus the per-element override map.
    :func:`~.build.emit_transform_specs` writes it instead of allocating.
    :meth:`stream` holds only the planned tags: a fan-out's first line
    reuses the spec's own (registered) tag, which no family mints.
    """

    FAMILY: ClassVar[str] = "transforms"
    KINDS: ClassVar[frozenset[str]] = frozenset({"geomTransf"})
    MIGRATED: ClassVar[bool] = True

    fanout: TransformFanout | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "transforms: the transform plan derives its rows from its "
                "fan-out; pass fanout, not rows."
            )

    def _planned(self) -> TransformFanout:
        if self.fanout is None:
            raise _miss(
                "transforms: this plan carries no fan-out; plan_tags "
                "plans one for every mode (ADR 0114 D4, amended)."
            )
        return self.fanout

    def stream(self) -> tuple[TagRow, ...]:
        """One ``("geomTransf:<type>", tag)`` row per planned tag."""
        from .build import _TRANSF_TYPE_TOKEN

        rows: list[TagRow] = []
        for transf, lines in self._planned().specs:
            if not lines:
                continue
            token = _TRANSF_TYPE_TOKEN[type(transf)]
            rows.extend((f"geomTransf:{token}", tag) for tag, _ in lines[1:])
        return tuple(rows)

    def fanout_for(
        self, transforms: list[GeomTransf], fem: object,
    ) -> TransformFanout:
        """The planned fan-out, which must cover exactly ``transforms``.

        ``transforms`` are the specs this emit walks, in order, over the
        FEM snapshot ``fem``. A plan made over another FEM (a fork of
        another model's plan), for other specs, or in another order,
        raises :class:`TagLawError`.
        """
        fanout = self._planned()
        if fem is not fanout.fem:
            raise _miss(_FOREIGN_PLAN.format(family="transform"))
        if [id(t) for t, _ in fanout.specs] != [id(t) for t in transforms]:
            raise _miss(
                f"the transform plan holds {len(fanout.specs)} specs, but "
                f"this emit walks {len(transforms)}, or in another order: "
                "the plan was not made for this model (ADR 0114 D4, "
                "amended)."
            )
        return fanout


#: Where an emit writes a region: the site's kind and its scope. The scope
#: is ``None`` for a model-wide pool, ``id(stage)`` for a stage's pool, and
#: ``id(spec)`` for a recorder.
RegionSite = tuple[str, "int | None"]

#: The kinds of region site, and what keys a region at each one:
#:
#: ``named``     a named region (``ops.region`` / ``s.region``), by name;
#: ``rayleigh``  a region-scoped Rayleigh, by ``(record index, on index)``;
#: ``damping``   a damping attach, by ``(record index, on index)``;
#: ``recorder``  a filtered recorder, by region key (``filter``, ``energy``).
REGION_SITE_KINDS: frozenset[str] = frozenset(
    {"named", "rayleigh", "damping", "recorder"})


#: Every region name a named-region site declares, in first-seen order:
#: ``(name, merged member nodes, first rank)``. The first rank is the
#: runtime rank a partitioned emit numbers the region on (the first, in
#: partition order, that holds a member), and ``None`` on a flat emit or
#: when no rank holds one.
NamedMembers = tuple[tuple[str, tuple[int, ...], "int | None"], ...]


class PlannedRegion(NamedTuple):
    """One planned ``region`` tag: the site that writes it, its key there."""

    site: RegionSite
    key: object
    tag: int


class NamedRegion(NamedTuple):
    """A named region as the plan holds it: its merged member nodes, in
    first-seen order, its planned tag and, under a partitioned emit, the
    rank it was numbered on.

    ``tag`` is ``None`` for a region no emit writes (it has no members
    or, under a partitioned emit, no rank holds one). :meth:`planned_tag`
    and :meth:`tag_on_rank` are how a writer reads it.
    """

    name: str
    nodes: tuple[int, ...]
    tag: int | None
    first_rank: int | None = None

    def tag_on_rank(self, rank: int) -> int:
        """The tag ``rank``, which holds members of this region, writes.

        The plan numbered the region on its first holder rank, so no
        earlier rank may hold a member.
        """
        if self.first_rank is None or rank < self.first_rank:
            raise _miss(
                f"named region {self.name!r} has members on rank {rank}, "
                f"but the region plan numbered it on rank {self.first_rank}: "
                "the plan was not made for this emit (ADR 0114 D4, amended)."
            )
        return self.planned_tag()

    def check_unheld_on(self, rank: int) -> None:
        """Raise if the plan numbered this region on ``rank``, which holds
        none of its members."""
        if rank == self.first_rank:
            raise _miss(
                f"the region plan numbered named region {self.name!r} on "
                f"rank {rank}, which holds none of its members: the plan "
                "was not made for this emit (ADR 0114 D4, amended)."
            )

    def planned_tag(self) -> int:
        """The tag to write; raises if the plan gave this region none."""
        if self.tag is None:
            raise _miss(
                f"the region plan gives named region {self.name!r} no tag, "
                "but this emit writes it: the plan was not made for this "
                "emit (ADR 0114 D4, amended)."
            )
        return self.tag


def plan_regions(
    sites: "Iterable[tuple[RegionSite, Sequence[object]]]",
    tags: TagAllocator,
) -> tuple[PlannedRegion, ...]:
    """Mint one region tag per key of every site, in order.

    The region allocation loop, moved out of the emit (ADR 0114 D4,
    amended). ``sites`` are the region sites of one emit, in the order it
    writes them, each with its keys in the order it mints them
    (``BuiltModel._region_sites``). The build's tag plan runs it once per
    emit mode with the planner allocator; a recorder handed a plain
    allocator runs it over its own site
    (:meth:`~apeGmsh.opensees.recorder.FilterableRecorder.planned_region_tags`).
    """
    rows: list[PlannedRegion] = []
    for site, keys in sites:
        if site[0] not in REGION_SITE_KINDS:
            raise TagLawError(
                f"plan_regions: unknown region site {site[0]!r}; the kinds "
                f"are {sorted(REGION_SITE_KINDS)}."
            )
        for key in keys:
            rows.append(PlannedRegion(site, key, tags.allocate("region")))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class RegionTagPlan(FamilyTagPlan):
    """Named regions, damping regions and recorder filter/energy regions.

    ``regions`` is :func:`plan_regions`'s result, made once by
    :func:`plan_tags` over ``BuiltModel._region_sites``: every region tag
    the mode's emit writes, in the order it was minted, keyed by its site.
    ``named`` holds, per named-region site, every declared region name in
    first-seen order with its merged member nodes, so the emit writes the
    members the plan resolved. ``partitioned`` says the named regions were
    numbered rank by rank, each on the first rank that holds a member.
    ``fem`` is the FEM snapshot it was made over. The emit's writers read
    their tags here (:meth:`tags_for`, :meth:`named_for`) and mint none.
    """

    FAMILY: ClassVar[str] = "regions"
    KINDS: ClassVar[frozenset[str]] = frozenset({"region"})
    MIGRATED: ClassVar[bool] = True

    regions: tuple[PlannedRegion, ...] | None = field(
        default=None, compare=False)
    named: "Mapping[RegionSite, NamedMembers]" = field(
        default_factory=dict, compare=False)
    partitioned: bool = False
    fem: object = field(default=None, compare=False)
    _by_site: "dict[RegionSite, dict[object, int]]" = field(
        default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "regions: the region plan derives its rows from its planned "
                "regions; pass regions, not rows."
            )
        if self.regions is None:
            return
        last = 0
        for row in self.regions:
            if row.site[0] not in REGION_SITE_KINDS:
                raise TagLawError(
                    f"regions: unknown region site {row.site[0]!r}.")
            if row.tag <= last:
                raise TagLawError(
                    f"regions: tag {row.tag} of {row.site}/{row.key!r} "
                    f"follows tag {last}; the plan holds its regions in the "
                    "order it minted them, so its tags only rise."
                )
            last = row.tag
            at_site = self._by_site.setdefault(row.site, {})
            if row.key in at_site:
                raise TagLawError(
                    f"regions: {row.site}/{row.key!r} is planned twice.")
            at_site[row.key] = row.tag
        for site, at_site in self._by_site.items():
            if site[0] != "named":
                continue
            members = {
                name: nodes for name, nodes, _ in self.named.get(site, ())}
            for name in at_site:
                if not isinstance(name, str) or not members.get(name):
                    raise TagLawError(
                        f"regions: named region {name!r} has a tag but no "
                        f"members at {site}.")

    def _planned(self, fem: object) -> tuple[PlannedRegion, ...]:
        if self.regions is None:
            raise _miss(
                "regions: this plan carries no regions; plan_tags plans "
                "them for every mode (ADR 0114 D4, amended)."
            )
        if fem is not self.fem:
            raise _miss(_FOREIGN_PLAN.format(family="region"))
        return self.regions

    def stream(self) -> tuple[TagRow, ...]:
        """One ``("region", tag)`` row per planned region, in mint order."""
        return tuple(("region", row.tag) for row in self._planned(self.fem))

    def tags_for(
        self, fem: object, site: RegionSite, keys: "Sequence[object]",
    ) -> dict[object, int]:
        """The planned tag of each of ``keys`` at ``site``.

        ``keys`` are the regions this emit writes at ``site``, in the order
        it writes them, derived from the records it walks. The plan must
        hold exactly those keys there, in that order (a damping pool and a
        recorder write their regions in the order the plan minted them): a
        plan made for other records, one that dropped a region, or one
        that swapped two, raises :class:`TagLawError`.
        """
        self._planned(fem)
        planned = self._by_site.get(site, {})
        if list(planned) != list(keys):
            raise _miss(
                f"the region plan holds {list(planned)} at {site}, but this "
                f"emit writes {list(keys)}: the plan was not made for this "
                "emit (ADR 0114 D4, amended)."
            )
        return dict(planned)

    def named_for(
        self, fem: object, site: RegionSite, names: "Sequence[str]",
    ) -> tuple[NamedRegion, ...]:
        """The named regions of ``site``, in first-seen order.

        ``names`` are the region names this emit's records declare at
        ``site``, in first-seen order; the plan must hold exactly them,
        in that order. Each comes with its planned members and tag.

        A flat or split emit numbers the names that have members in
        first-seen order. A partitioned emit numbers them rank by rank,
        each on its first holder rank, in first-seen order within a rank;
        its writers check each first rank (:meth:`NamedRegion.tag_on_rank`,
        :meth:`NamedRegion.check_unheld_on`). Either way the planned tags
        must be exactly those names', rising in that order: a plan that
        dropped or swapped two raises :class:`TagLawError`.
        """
        self._planned(fem)
        members = tuple(self.named.get(site, ()))
        if [name for name, _, _ in members] != list(names):
            raise _miss(
                f"the region plan holds the named regions "
                f"{[name for name, _, _ in members]} at {site}, but this "
                f"emit declares {list(names)}: the plan was not made for "
                "this emit (ADR 0114 D4, amended)."
            )
        if self.partitioned:
            ranked = [(first, i, name)
                      for i, (name, _, first) in enumerate(members)
                      if first is not None]
            numbered = [name for _, _, name in sorted(ranked)]
        else:
            numbered = [name for name, nodes, _ in members if nodes]
        planned = self._by_site.get(site, {})
        if list(planned) != numbered:
            raise _miss(
                f"the region plan numbers the named regions {list(planned)} "
                f"at {site}, but this emit numbers {numbered}, in that "
                "order: the plan was not made for this emit (ADR 0114 D4, "
                "amended)."
            )
        return tuple(
            NamedRegion(name, nodes, planned.get(name), first)
            for name, nodes, first in members
        )


#: A parameter site's key: its record and the runtime rank whose block
#: writes it (``None`` outside any partition block).
ParameterKey = tuple[object, "int | None"]


@dataclass(frozen=True, slots=True)
class ParameterTagPlan(FamilyTagPlan):
    """Initial-stress, absorbing-flip and ``s.update_parameter`` parameter
    ids.

    ``lines`` is :func:`~.build.plan_parameters`'s result, made once by
    :func:`plan_tags` over every parameter site the mode's emit writes, in
    the order it writes them: the global initial stresses, then stage by
    stage its initial stresses, its absorbing flips and its updates (each
    rank by rank under a partitioned emit, a site with no element on its
    rank holding no tag). ``owners`` is every ``(record, rank)`` the emit
    walks, derived from the model's records alone; the lines must be
    exactly those, by identity and in that order, or the plan raises when
    it is made. The writers read their lines (:meth:`line_at`,
    :meth:`flip_lines`) and mint nothing; a flip or update line carries
    the element tags the plan resolved, so the writers do not resolve them
    again.
    """

    FAMILY: ClassVar[str] = "parameters"
    KINDS: ClassVar[frozenset[str]] = frozenset({"parameter"})
    MIGRATED: ClassVar[bool] = True

    lines: tuple[PlannedParameter, ...] | None = field(
        default=None, compare=False)
    owners: tuple[ParameterKey, ...] = field(default=(), compare=False)
    _index: "dict[tuple[int, int | None], PlannedParameter]" = field(
        default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        from .build import PARAMETER_VERBS

        if self.rows:
            raise TagLawError(
                "parameters: the parameter plan derives its rows from its "
                "planned sites; pass lines, not rows."
            )
        if self.lines is None:
            return
        last = 0
        for line in self.lines:
            if len(line.tags) not in PARAMETER_VERBS.get(line.verb, ()):
                raise TagLawError(
                    f"parameters: a {line.verb!r} line holds "
                    f"{len(line.tags)} tags; the verbs and their counts are "
                    f"{dict(PARAMETER_VERBS)}."
                )
            key = (id(line.record), line.rank)
            if key in self._index:
                raise TagLawError(
                    f"parameters: a {type(line.record).__name__} is planned "
                    f"twice at rank {line.rank}.")
            self._index[key] = line
            for tag in line.tags:
                if tag <= last:
                    raise TagLawError(
                        f"parameters: tag {tag} follows tag {last}; the plan "
                        "holds its sites in the order it minted them, so its "
                        "tags only rise."
                    )
                last = tag
        planned = [(line.record, line.rank) for line in self.lines]
        if len(planned) != len(self.owners) or any(
            a is not b or r != s
            for (a, r), (b, s) in zip(planned, self.owners)
        ):
            raise TagLawError(
                f"the parameter plan holds {len(planned)} sites, but the "
                f"model's records declare {len(self.owners)}, or other "
                "ones, or in another order: the plan does not cover the "
                "model (ADR 0114 D4, amended)."
            )

    def planned(self) -> tuple[PlannedParameter, ...]:
        """The planned sites; a sub-plan that carries none raises."""
        if self.lines is None:
            raise _miss(
                "parameters: this plan carries no parameter plan; plan_tags "
                "plans one for every mode (ADR 0114 D4, amended)."
            )
        return self.lines

    def stream(self) -> tuple[TagRow, ...]:
        """One ``(verb, tag)`` row per planned tag, in mint order. The
        verb is the one that writes it: ``step_hook_ramp`` (an initial
        stress's three ramps), ``flip_element_stage`` or
        ``update_parameter``."""
        return tuple(
            (line.verb, tag) for line in self.planned() for tag in line.tags)

    def _line(self, record: object, rank: int | None) -> PlannedParameter:
        """The line the plan holds for ``record`` at ``rank``.

        Looked up by the record's identity: a record the plan does not
        hold at that rank (another model's, or one the plan dropped)
        raises :class:`~.build.TagPlanMiss`.
        """
        self.planned()
        line = self._index.get((id(record), rank))
        if line is None or line.record is not record:
            raise _miss(
                f"the parameter plan holds no {type(record).__name__} at "
                f"rank {rank}: the plan was not made for this emit (ADR "
                "0114 D4, amended)."
            )
        return line

    def __getitem__(self, key: ParameterKey) -> tuple[int, ...]:
        """The tags the plan gives ``record`` at ``rank``."""
        record, rank = key
        return self._line(record, rank).tags

    def line_at(self, site: ParameterSite) -> PlannedParameter:
        """The line of ``site``: its record's at its rank, with as many
        tags as the site declares, written by its verb, or
        :class:`~.build.TagPlanMiss`."""
        line = self._line(site.record, site.rank)
        if line.verb != site.verb or len(line.tags) != site.n_tags:
            raise _miss(
                f"the parameter plan gives a {type(site.record).__name__} "
                f"at rank {site.rank} {len(line.tags)} {line.verb!r} tag(s), "
                f"but this emit writes {site.n_tags} {site.verb!r} tag(s) "
                "there: the plan was not made for this emit (ADR 0114 D4, "
                "amended)."
            )
        return line

    def flip_lines(
        self, verb: str, records: "Sequence[object]", rank: int | None,
    ) -> tuple[PlannedParameter, ...]:
        """The lines of a flip or update pass: each record's at ``rank``,
        written by ``verb``, with the element tags the plan resolved for
        it there (:attr:`PlannedParameter.ele_tags`).

        A record the plan does not hold at ``rank``, or holds for another
        verb, raises :class:`~.build.TagPlanMiss`.
        """
        lines = tuple(self._line(rec, rank) for rec in records)
        for line in lines:
            if line.verb != verb:
                raise _miss(
                    f"the parameter plan holds a "
                    f"{type(line.record).__name__} at rank {rank} for "
                    f"{line.verb!r}, but this emit writes it by {verb!r}: "
                    "the plan was not made for this emit (ADR 0114 D4, "
                    "amended)."
                )
        return lines


@dataclass(frozen=True, slots=True)
class MPElementTagPlan(FamilyTagPlan):
    """Elements the MP constraints synthesise: ties, rebar, rigid bodies,
    couplings.

    ``mp`` is the :class:`~.build.MPElementPlan` :func:`plan_tags` makes
    once per mode: every element in the order its mode's emit writes it
    (rigid bodies, kinematic couplings and interpolation ties; reinforce
    and embed ties; rebar cells; stage by stage, and rank by rank under a
    partitioned emit), and, under a partitioned emit, the global
    MP-constraint pass's routing. The writers read it instead of
    allocating.
    """

    FAMILY: ClassVar[str] = "mp_elements"
    KINDS: ClassVar[frozenset[str]] = frozenset({"element"})
    MIGRATED: ClassVar[bool] = True

    mp: MPElementPlan | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "mp_elements: the MP-element plan derives its rows from "
                "its planned elements; pass mp, not rows."
            )

    def planned(self) -> MPElementPlan:
        """The MP-element plan; a sub-plan that carries none raises."""
        if self.mp is None:
            raise _miss(
                "mp_elements: this plan carries no MP-element plan; "
                "plan_tags plans one for every mode (ADR 0114 D4, "
                "amended)."
            )
        return self.mp

    def stream(self) -> tuple[TagRow, ...]:
        """One ``(verb, tag)`` row per planned element, in emit order.

        The verb is the one its writer emits: ``element``,
        ``embeddedNode``, ``embedded_rebar`` or ``embedded_node``.
        """
        return tuple((line.verb, line.tag) for line in self.planned().lines)


@dataclass(frozen=True, slots=True)
class InterfaceTagPlan(FamilyTagPlan):
    """Interface ``zeroLength`` elements and their two uniaxial materials.

    ``interfaces`` is the :class:`~.build.InterfacePlan` :func:`plan_tags`
    makes once per mode: every interface record with its ``(normal,
    tangential, element)`` tags, in the order its mode's emit takes
    them. :func:`~.build.allocate_interface_tags` reads it instead of
    allocating.
    """

    FAMILY: ClassVar[str] = "interfaces"
    KINDS: ClassVar[frozenset[str]] = frozenset(
        {"element", "uniaxialMaterial"})
    MIGRATED: ClassVar[bool] = True

    interfaces: InterfacePlan | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "interfaces: the interface plan derives its rows from its "
                "planned records; pass interfaces, not rows."
            )

    def planned(self) -> InterfacePlan:
        """The interface plan; a sub-plan that carries none raises."""
        if self.interfaces is None:
            raise _miss(
                "interfaces: this plan carries no interface plan; "
                "plan_tags plans one for every mode (ADR 0114 D4, "
                "amended)."
            )
        return self.interfaces

    def stream(self) -> tuple[TagRow, ...]:
        """Each planned record's two ``uniaxialMaterial`` rows, then its
        ``element`` row (the ``zeroLength``), in plan order."""
        rows: list[TagRow] = []
        for line in self.planned().lines:
            n_tag, t_tag, ele_tag = line.tags
            rows += [("uniaxialMaterial", n_tag), ("uniaxialMaterial", t_tag),
                     ("element", ele_tag)]
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class ContactTagPlan(FamilyTagPlan):
    """Contact surfaces and contact interactions (face and rigid plane).

    ``contacts`` is :func:`~.build.plan_contacts`'s result, made once by
    :func:`plan_tags`: every interaction in emit order with its planned
    tags and, under a partitioned emit, its routing (owner rank and
    ghost nodes, ADR 0092 S4). The emit paths write it instead of
    allocating, and the partitioned path reads its routing instead of
    resolving owners again.
    """

    FAMILY: ClassVar[str] = "contacts"
    KINDS: ClassVar[frozenset[str]] = frozenset({"contactSurface", "contact"})
    MIGRATED: ClassVar[bool] = True

    contacts: ContactPlan | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "contacts: the contact plan derives its rows from its "
                "planned interactions; pass contacts, not rows."
            )

    def _planned(self) -> ContactPlan:
        if self.contacts is None:
            raise _miss(
                "contacts: this plan carries no contact plan; plan_tags "
                "plans one for every mode (ADR 0114 D4, amended)."
            )
        return self.contacts

    def stream(self) -> tuple[TagRow, ...]:
        """Each planned interaction's ``contact_surface`` rows, then the
        row of its verb (``contact`` or ``contact_plane``, the record
        kind's own name), in emit order."""
        rows: list[TagRow] = []
        for line in self._planned().lines:
            if line.kind not in ("contact", "contact_plane"):
                raise TagLawError(
                    f"contacts: unknown contact kind {line.kind!r}.")
            *surfaces, contact = line.tags
            rows.extend(("contact_surface", tag) for tag in surfaces)
            rows.append((line.kind, contact))
        return tuple(rows)

    def for_fem(self, fem: object) -> ContactPlan:
        """The contact plan, which must have been made over ``fem``.

        ``fem`` is the FEM snapshot this emit walks. A plan made over
        another one (a fork of another model's plan), or one that does
        not hold each of its contact records exactly once, raises
        :class:`TagLawError`.
        """
        planned = self._planned()
        if fem is not planned.fem:
            raise _miss(_FOREIGN_PLAN.format(family="contact"))
        planned.check_covers(fem)
        return planned


#: The sub-plan class of every family, in :class:`TagPlan` field order.
FAMILY_PLANS: dict[str, type[FamilyTagPlan]] = {
    cls.FAMILY: cls
    for cls in (
        ElementTagPlan, TransformTagPlan, RegionTagPlan, ParameterTagPlan,
        MPElementTagPlan, InterfaceTagPlan, ContactTagPlan,
    )
}

#: Family names, in :class:`TagPlan` field order.
FAMILIES: tuple[str, ...] = tuple(FAMILY_PLANS)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TagPlan:
    """Every tag one emit mode writes, planned before the emit.

    ``allocator`` is the planner allocator, frozen: it holds the seeded
    primitive tags and every planned family's mints. ``inputs`` is
    :func:`plan_inputs` of the model the plan was made for, so a memoised
    plan can tell whether it still describes a model
    (:meth:`planned_for`).

    The emit holds the plan itself and hands it to every helper; the plan
    has no allocation API, so an emit can read a tag but never mint one.
    """

    mode: TagMode
    allocator: TagAllocator
    elements: ElementTagPlan
    transforms: TransformTagPlan
    regions: RegionTagPlan
    parameters: ParameterTagPlan
    mp_elements: MPElementTagPlan
    interfaces: InterfaceTagPlan
    contacts: ContactTagPlan
    inputs: tuple[object, ...] = field(default=(), compare=False, repr=False)

    def planned_for(self, bm: BuiltModel) -> bool:
        """``True`` iff this plan was made from exactly ``bm``'s inputs.

        Every public field of ``bm`` must be the very object the plan
        read (identity, not equality): a model copied with
        :func:`dataclasses.replace` or :func:`copy.copy` and then changed
        is not described by its source's plan.
        """
        now = plan_inputs(bm)
        return len(now) == len(self.inputs) and all(
            a is b for a, b in zip(now, self.inputs))

    def __post_init__(self) -> None:
        if not self.allocator.frozen:
            raise TagLawError(
                "TagPlan: the planner allocator must be frozen; plan_tags "
                "freezes it as its last step."
            )

    def family(self, name: str) -> FamilyTagPlan:
        """The sub-plan of family ``name``; an unknown name raises."""
        if name not in FAMILY_PLANS:
            raise KeyError(
                f"unknown tag family {name!r}; the families are {FAMILIES}."
            )
        plan: FamilyTagPlan = getattr(self, name)
        return plan

    @property
    def migrated(self) -> tuple[str, ...]:
        """The families whose tags this plan writes, in field order."""
        return tuple(f for f in FAMILIES if FAMILY_PLANS[f].MIGRATED)

    @property
    def frozen_kinds(self) -> frozenset[str]:
        """Allocator kinds no emit path may mint any more.

        A kind is frozen once every family that mints it has moved into
        the plan; a kind shared with a pending family stays open.
        """
        kinds: set[str] = set()
        for cls in FAMILY_PLANS.values():
            kinds |= cls.KINDS
        return frozenset(
            k for k in kinds
            if all(
                cls.MIGRATED for cls in FAMILY_PLANS.values()
                if k in cls.KINDS
            )
        )

    def stream(self) -> list[TagRow]:
        """Every family's planned rows, family by family.

        Raises :class:`NotImplementedError` while any family is pending.
        """
        out: list[TagRow] = []
        for name in FAMILIES:
            out.extend(self.family(name).stream())
        return out


def plan_inputs(bm: BuiltModel) -> tuple[object, ...]:
    """Every public field of ``bm``, in field order: what a plan reads.

    The planner reads the primitives, their tags, the FEM snapshot,
    ``element_tags`` and the records (stage, region, initial-stress and
    parameter records); taking every public field covers them all.
    """
    import dataclasses

    return tuple(
        getattr(bm, f.name) for f in dataclasses.fields(bm)
        if not f.name.startswith("_")
    )


def plan_tags(bm: BuiltModel, mode: TagMode) -> TagPlan:
    """Plan every tag ``bm``'s emit in ``mode`` writes, then freeze.

    The planner allocator is seeded as the emit seeds its own: each
    registered primitive in registration order, then, under
    ``element_tags="fem"``, the FEM element-id range on the ``"element"``
    counter (ADR 0111 D2). A seeded tag that disagrees with
    ``bm.tag_for`` raises :class:`TagLawError`: the plan would not
    describe the tags the model was registered with.
    """
    from ..apesees import _kind_of
    from .build import (
        allocate_element_tags,
        plan_transform_specs,
        reserve_fem_element_tags,
        topological_order,
    )
    from .types import Element, GeomTransf

    if not isinstance(mode, TagMode):
        raise TypeError(
            f"plan_tags: mode must be a TagMode, got {type(mode).__name__}."
        )
    tags = TagAllocator()
    for prim in bm.primitives:
        tag = tags.allocate_for(prim, _kind_of(prim))
        if bm.tag_for.get(id(prim)) != tag:
            raise TagLawError(
                f"plan_tags: {type(prim).__name__} seeds as tag {tag}, but "
                f"the model registered it as {bm.tag_for.get(id(prim))}."
            )
    if bm.element_tags == "fem":
        reserve_fem_element_tags(
            [p for p in bm.primitives if isinstance(p, Element)],
            bm.fem, tags,
        )

    ordered = topological_order(bm.primitives)
    element_specs = [p for p in ordered if isinstance(p, Element)]

    # Elements: every spec's fan-out, in the emit's topological order.
    # Each emit path made this allocation once, before any other
    # element-kind mint, so planning it here numbers it as they did.
    elements = ElementTagPlan(specs=tuple(allocate_element_tags(
        element_specs, bm.fem, tags, element_tags=bm.element_tags,
    )))

    # Transforms: the orientation fan-out, over the transform specs in
    # the same topological order. Every emit path ran it once, and no
    # other family mints ``geomTransf``, so every mode numbers it alike.
    transforms = TransformTagPlan(fanout=plan_transform_specs(
        [p for p in ordered if isinstance(p, GeomTransf)],
        element_specs, bm.fem, tags, bm.tag_for, ndm=bm.ndm,
    ))

    # MP elements and interfaces: one ``element`` counter, after the
    # element fan-out, interleaved as the mode's emit interleaves them;
    # the interfaces also number their ``uniaxialMaterial`` pairs.
    mp, interfaces = _plan_mp_elements_and_interfaces(bm, mode, tags)

    # Contacts: every interaction, in the order its mode's emit writes
    # it. No other family mints ``contactSurface`` or ``contact``.
    contacts = ContactTagPlan(contacts=_plan_contacts(bm, mode, tags))

    # Regions: every named, damping and recorder region, in the order its
    # mode's emit writes them. No other family mints ``region``.
    sites, named = bm._region_sites(mode, ordered)
    regions = RegionTagPlan(
        regions=plan_regions(sites, tags), named=named,
        partitioned=mode.partitioned, fem=bm.fem)

    # Parameters: every initial-stress ramp, absorbing flip and update,
    # in the order the mode's emit writes them. No other family mints
    # ``parameter``.
    parameters = _plan_parameters(bm, mode, elements, tags)

    # Every family is planned: the frozen planner holds every tag the
    # emit writes, and nothing mints after this.
    tags.freeze()
    return TagPlan(
        mode=mode,
        allocator=tags,
        elements=elements,
        transforms=transforms,
        regions=regions,
        parameters=parameters,
        mp_elements=MPElementTagPlan(mp=mp),
        interfaces=InterfaceTagPlan(interfaces=interfaces),
        contacts=contacts,
        inputs=plan_inputs(bm),
    )


def _plan_mp_elements_and_interfaces(
    bm: BuiltModel, mode: TagMode, tags: TagAllocator,
) -> tuple[MPElementPlan, InterfacePlan]:
    """The MP-element and interface plans of ``bm``'s emit in ``mode``.

    Both families mint ``element`` tags, so they are planned in one walk,
    in the order the mode's emit writes them:

    * flat: the global MP-constraint pass (rigid bodies,
      kinematic couplings, interpolation ties), the reinforce and embed
      ties, the unclaimed interfaces, the rebar cells; then, stage by
      stage, the stage's claimed MP constraints and claimed interfaces;
    * partitioned: every interface in one pre-pass (unclaimed, then each
      stage's claimed ones: ADR 0093 S8/S9); then rank by rank the global
      MP-constraint pass, routed once here
      (:func:`~.build.plan_partitioned_mp_constraints`), and the rank's
      reinforce ties and rebar cells
      (``BuiltModel._plan_partitioned_reinforcement``); then stage by
      stage, rank by rank, the stage's claimed MP constraints.

    The partitioned emit refuses ``g.embed``, so no embed tie is planned
    there. Each plan checks, as it is made, that it holds every element
    and record the emit of ``bm.fem`` writes, once.
    """
    from .build import (
        InterfacePlan,
        MPElementEntry,
        MPElementPlan,
        PlannedInterface,
        PlannedMPElement,
        _StageConstraintAdapter,
        build_element_partition_owner,
        build_node_partition_owners,
        constraint_pass_entries,
        embed_tie_records,
        interface_records,
        interpolation_records,
        mp_constraint_pools,
        mp_element_entry,
        plan_interface_tags,
        plan_mp_elements,
        plan_partitioned_mp_constraints,
        plan_stage_mp_constraints_partitioned,
        rebar_cell_entries,
        rebar_element_records,
        reinforce_tie_records,
        runtime_rank_from_partition_record,
    )

    fem = bm.fem
    claimed = frozenset(bm._claimed_constraint_ids())
    claimed_interfaces = frozenset(bm._claimed_interface_ids())
    unclaimed = [r for r in interface_records(fem)
                 if id(r) not in claimed_interfaces]
    node_constraints, surface_constraints = mp_constraint_pools(fem, claimed)
    mp: list[PlannedMPElement] = []
    ifaces: list[PlannedInterface] = []

    def ties(records: list[Any]) -> list[MPElementEntry]:
        return [mp_element_entry("reinforce_tie", r) for r in records]

    if not mode.partitioned:
        mp += plan_mp_elements([
            *constraint_pass_entries(
                node_constraints, interpolation_records(surface_constraints)),
            *ties(reinforce_tie_records(fem)),
            *(mp_element_entry("embed_tie", r)
              for r in embed_tie_records(fem)),
        ], tags)
        ifaces += plan_interface_tags(unclaimed, tags)
        mp += plan_mp_elements(
            rebar_cell_entries(rebar_element_records(fem)), tags)
        for stage in bm.stage_records:
            if stage.stage_constraint_records:
                adapter = _StageConstraintAdapter(
                    stage.stage_constraint_records)
                mp += plan_mp_elements(constraint_pass_entries(
                    adapter, interpolation_records(adapter)), tags)
            ifaces += plan_interface_tags(stage.stage_interface_records, tags)
        return (
            MPElementPlan(fem=fem, lines=tuple(mp), claimed_ids=claimed),
            InterfacePlan(fem=fem, lines=tuple(ifaces)),
        )

    ifaces += plan_interface_tags([
        *unclaimed,
        *(r for stage in bm.stage_records
          for r in stage.stage_interface_records),
    ], tags)
    partitions = list(fem.partitions)
    node_owners = build_node_partition_owners(fem)
    element_owner = build_element_partition_owner(fem)
    phantom_coords, rank_plans = plan_partitioned_mp_constraints(
        fem, claimed, node_owners, element_owner)
    reinforcement = bm._plan_partitioned_reinforcement(node_owners)
    ranks = [runtime_rank_from_partition_record(part, idx)
             for idx, part in enumerate(partitions)]
    for rank in ranks:
        # ``rank_plans`` is empty when the FEM has no constraint
        # container, else it routes every rank.
        if rank_plans and rank_plans[rank].any():
            routed = rank_plans[rank]
            mp += plan_mp_elements(constraint_pass_entries(
                node_constraints, routed.embedded_records,
                allowed_ids=routed.allowed_record_ids,
            ), tags)
        # ``reinforcement`` holds only the ranks that own a tie or a cell.
        if rank in reinforcement:
            rank_ties, rank_bars, _ghosts = reinforcement[rank]
            mp += plan_mp_elements(
                [*ties(rank_ties), *rebar_cell_entries(rank_bars)], tags)
    for stage in bm.stage_records:
        if not stage.stage_constraint_records:
            continue
        for rank in ranks:
            staged = plan_stage_mp_constraints_partitioned(
                stage.stage_constraint_records, partition_rank=rank,
                node_owners=node_owners, element_owner=element_owner,
            )
            if staged is None:
                continue
            mp += plan_mp_elements(constraint_pass_entries(
                staged.adapter, staged.plan.embedded_records,
                allowed_ids=staged.plan.allowed_record_ids,
            ), tags)
    return (
        MPElementPlan(
            fem=fem, lines=tuple(mp), partitioned=True, claimed_ids=claimed,
            phantom_coords=phantom_coords, rank_plans=rank_plans,
        ),
        InterfacePlan(fem=fem, lines=tuple(ifaces)),
    )


def parameter_owners(
    bm: BuiltModel, ranks: "Sequence[int | None]",
) -> tuple[ParameterKey, ...]:
    """Every ``(record, rank)`` whose parameter site ``bm``'s emit walks,
    in emit order, from the model's records alone.

    The global initial stresses, then stage by stage: its initial
    stresses (outside any partition block), its absorbing flips and its
    updates, each over ``ranks`` (``[None]`` on a flat emit, the runtime
    ranks in partition order on a partitioned one).
    """
    out: list[ParameterKey] = [(rec, None) for rec in bm.initial_stress_records]
    for stage in bm.stage_records:
        out += [(rec, None) for rec in stage.initial_stress_records]
        for records in (stage.activate_absorbing_records,
                        stage.update_parameter_records):
            out += [(rec, rank) for rank in ranks for rec in records]
    return tuple(out)


def _plan_parameters(
    bm: BuiltModel, mode: TagMode, elements: ElementTagPlan,
    tags: TagAllocator,
) -> ParameterTagPlan:
    """The parameter plan of ``bm``'s emit in ``mode``.

    The emit writes an initial stress's three ramp tags once, outside any
    partition block: first the global pool's, then each stage's. Each
    stage then writes its absorbing flips, then its updates, record by
    record; a partitioned emit writes each pass rank by rank, and a
    record takes a tag on a rank only if it addresses an element that rank
    owns. The element tags are the element plan's (``elements``), as the
    emit maps them; the rank ownership is the emit's
    (:func:`~.build.build_element_partition_owner`).
    """
    from .build import (
        FemToOpsTagMap,
        absorbing_ele_tags,
        build_element_partition_owner,
        initial_stress_sites,
        parameter_flip_sites,
        plan_parameters,
        runtime_rank_from_partition_record,
        update_parameter_ele_tags,
    )

    fem = bm.fem
    ranks: list[int | None] = [None]
    if mode.partitioned:
        ranks = [runtime_rank_from_partition_record(part, idx)
                 for idx, part in enumerate(fem.partitions)]
    sites: list[ParameterSite] = initial_stress_sites(bm.initial_stress_records)
    flips = any(
        stage.activate_absorbing_records or stage.update_parameter_records
        for stage in bm.stage_records)
    # The emit's own maps, built only when a flip or update reads them.
    eid_to_tag = FemToOpsTagMap.from_plan(elements.specs if flips else ())
    owner = (build_element_partition_owner(fem)
             if flips and mode.partitioned else None)
    for stage in bm.stage_records:
        sites += initial_stress_sites(stage.initial_stress_records)
        absorbing = stage.activate_absorbing_records
        for rank in ranks:
            sites += parameter_flip_sites("flip_element_stage", absorbing, [
                absorbing_ele_tags(rec, fem, eid_to_tag, owner, rank)
                for rec in absorbing], rank)
        updates = stage.update_parameter_records
        for rank in ranks:
            sites += parameter_flip_sites("update_parameter", updates, [
                update_parameter_ele_tags(rec, fem, eid_to_tag, owner, rank)
                for rec in updates], rank)
    return ParameterTagPlan(
        lines=plan_parameters(sites, tags),
        owners=parameter_owners(bm, ranks),
    )


def _plan_contacts(
    bm: BuiltModel, mode: TagMode, tags: TagAllocator,
) -> ContactPlan:
    """The contact plan of ``bm``'s emit in ``mode``.

    A flat or split emit writes every contact, then every contact plane.
    A partitioned emit writes each interaction inside its owner rank's
    block, rank by rank (ADR 0092 S4), so the routing is resolved here,
    once per plan: ``BuiltModel._plan_partitioned_contacts`` picks each
    owner rank and ghost set and raises every routing refusal before any
    emission. Its warnings ride on the plan, and each emit repeats them.
    """
    from .build import (
        flat_contact_entries,
        partitioned_contact_entries,
        plan_contacts,
    )

    if not mode.partitioned:
        return plan_contacts(bm.fem, flat_contact_entries(bm.fem), tags)
    partitions = list(bm.fem.partitions)
    routing, notes = bm._plan_partitioned_contacts(
        partitions, staged=mode.staged)
    return plan_contacts(
        bm.fem, partitioned_contact_entries(routing, partitions), tags,
        notes=notes,
    )
