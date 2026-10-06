"""The build-time tag plan (ADR 0114 D4, amended).

A tag is written once, by the bridge's build, and the archive keeps it.
:func:`plan_tags` is where the build writes them: it seeds a planner
:class:`TagAllocator` from the registered primitives, plans every derived
tag family, and freezes the allocator. The emit paths then read the plan
instead of minting.

The migration moves each family's allocation loop out of its emit helper
and into this module one family at a time, so every step leaves the decks
byte-identical. Until a family moves, its sub-plan is *pending*: its
:meth:`FamilyTagPlan.stream` raises, and the emit path keeps minting that
family's tags from :meth:`TagPlan.emit_allocator`, a fork of the frozen
planner allocator in which every kind whose minting families have all
moved is frozen. A minting site the migration missed then raises
:class:`TagLawError` where it is.

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

The emit helpers that are handed only the emit allocator reach the plan
through it: :func:`plan_of` reads the plan that an allocator from
:meth:`TagPlan.emit_allocator` was forked for.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from .tag_allocator import TagAllocator, TagLawError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ..apesees import BuiltModel
    from .build import ContactPlan, ElementPlanRows, TransformFanout
    from .types import Element, GeomTransf

#: One planned emission: ``(kind, tag)`` in the emit's verb vocabulary.
TagRow = tuple[str, int]

#: A helper handed a fork of another model's plan says so first, rather
#: than report a count or order mismatch that would hide the cause.
_FOREIGN_PLAN = (
    "the {family} plan was made over another FEM snapshot than the one "
    "this emit walks: the emit allocator is a fork of another model's tag "
    "plan. Emit each model through its own BuiltModel.emit (ADR 0114 D4, "
    "amended)."
)


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
            raise TagLawError(
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
            raise TagLawError(_FOREIGN_PLAN.format(family="transform"))
        if [id(t) for t, _ in fanout.specs] != [id(t) for t in transforms]:
            raise TagLawError(
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


class PlannedRegion(NamedTuple):
    """One planned ``region`` tag: the site that writes it, its key there."""

    site: RegionSite
    key: object
    tag: int


class NamedRegion(NamedTuple):
    """A named region as the plan holds it: its merged member nodes, in
    first-seen order, and its planned tag.

    ``tag`` is ``None`` for a region no emit writes (it has no members
    or, under a partitioned emit, no rank holds one). :meth:`planned_tag`
    is how a writer reads it.
    """

    name: str
    nodes: tuple[int, ...]
    tag: int | None

    def planned_tag(self) -> int:
        """The tag to write; raises if the plan gave this region none."""
        if self.tag is None:
            raise TagLawError(
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
    members the plan resolved. ``fem`` is the FEM snapshot it was made
    over. The emit's writers read their tags here (:meth:`tags_for`,
    :meth:`named_for`) and mint none.
    """

    FAMILY: ClassVar[str] = "regions"
    KINDS: ClassVar[frozenset[str]] = frozenset({"region"})
    MIGRATED: ClassVar[bool] = True

    regions: tuple[PlannedRegion, ...] | None = field(
        default=None, compare=False)
    named: "Mapping[RegionSite, tuple[tuple[str, tuple[int, ...]], ...]]" = (
        field(default_factory=dict, compare=False))
    fem: object = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.rows:
            raise TagLawError(
                "regions: the region plan derives its rows from its planned "
                "regions; pass regions, not rows."
            )
        if self.regions is None:
            return
        last = 0
        seen: set[tuple[RegionSite, object]] = set()
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
            if (row.site, row.key) in seen:
                raise TagLawError(
                    f"regions: {row.site}/{row.key!r} is planned twice.")
            seen.add((row.site, row.key))
            if row.site[0] == "named" and row.key not in dict(
                    self.named.get(row.site, ())):
                raise TagLawError(
                    f"regions: named region {row.key!r} has a tag but no "
                    f"members at {row.site}.")

    def _planned(self, fem: object) -> tuple[PlannedRegion, ...]:
        if self.regions is None:
            raise TagLawError(
                "regions: this plan carries no regions; plan_tags plans "
                "them for every mode (ADR 0114 D4, amended)."
            )
        if fem is not self.fem:
            raise TagLawError(_FOREIGN_PLAN.format(family="region"))
        return self.regions

    def stream(self) -> tuple[TagRow, ...]:
        """One ``("region", tag)`` row per planned region, in mint order."""
        return tuple(("region", row.tag) for row in self._planned(self.fem))

    def tags_for(
        self, fem: object, site: RegionSite, keys: "Sequence[object]",
    ) -> dict[object, int]:
        """The planned tag of each of ``keys`` at ``site``.

        ``keys`` are the regions this emit writes at ``site``, derived
        from the records it walks. The plan must hold exactly those keys
        there (by identity of key, not by count): a plan made for other
        records, or one that dropped a region, raises
        :class:`TagLawError`.
        """
        planned = {
            row.key: row.tag for row in self._planned(fem) if row.site == site}
        if Counter(list(planned)) != Counter(keys):
            raise TagLawError(
                f"the region plan holds {list(planned)} at {site}, but this "
                f"emit writes {list(keys)}: the plan was not made for this "
                "emit (ADR 0114 D4, amended)."
            )
        return {key: planned[key] for key in keys}

    def named_for(
        self, fem: object, site: RegionSite, names: "Sequence[str]",
    ) -> tuple[NamedRegion, ...]:
        """The named regions of ``site``, in first-seen order.

        ``names`` are the region names this emit's records declare at
        ``site``, in first-seen order; the plan must hold exactly them,
        in that order. Each comes with its planned members and tag.
        """
        planned = self._planned(fem)
        members = tuple(self.named.get(site, ()))
        if [name for name, _ in members] != list(names):
            raise TagLawError(
                f"the region plan holds the named regions "
                f"{[name for name, _ in members]} at {site}, but this emit "
                f"declares {list(names)}: the plan was not made for this "
                "emit (ADR 0114 D4, amended)."
            )
        tags = {row.key: row.tag for row in planned if row.site == site}
        return tuple(
            NamedRegion(name, nodes, tags.get(name))
            for name, nodes in members
        )


@dataclass(frozen=True, slots=True)
class ParameterTagPlan(FamilyTagPlan):
    """Initial-stress and staged ``updateParameter`` parameter ids."""

    FAMILY: ClassVar[str] = "parameters"
    KINDS: ClassVar[frozenset[str]] = frozenset({"parameter"})


@dataclass(frozen=True, slots=True)
class MPElementTagPlan(FamilyTagPlan):
    """Elements the MP constraints synthesise: ties, rebar, rigid bodies,
    couplings."""

    FAMILY: ClassVar[str] = "mp_elements"
    KINDS: ClassVar[frozenset[str]] = frozenset({"element"})


@dataclass(frozen=True, slots=True)
class InterfaceTagPlan(FamilyTagPlan):
    """Interface ``zeroLength`` elements and their two uniaxial materials."""

    FAMILY: ClassVar[str] = "interfaces"
    KINDS: ClassVar[frozenset[str]] = frozenset(
        {"element", "uniaxialMaterial"})


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
            raise TagLawError(
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
            raise TagLawError(_FOREIGN_PLAN.format(family="contact"))
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

    def emit_allocator(self) -> TagAllocator:
        """The allocator the emit mints its pending families from.

        A fork of the frozen planner allocator: it continues every
        counter from the plan, and every :attr:`frozen_kinds` kind raises
        :class:`TagLawError` on a mint.
        """
        return self.allocator.fork(self.frozen_kinds, origin=self)


def plan_of(tags: TagAllocator) -> TagPlan:
    """The plan ``tags`` was forked from by :meth:`TagPlan.emit_allocator`.

    The emit helpers that are handed only the emit allocator read the
    plan here. Any other allocator (a fresh one, the frozen planner
    allocator, a plain :meth:`TagAllocator.fork`) raises
    :class:`TagLawError`: no plan drove that emit.
    """
    # The fork's origin is set by TagAllocator.fork and read only here
    # (a sibling module of tag_allocator in this package).
    plan = tags._origin
    if not isinstance(plan, TagPlan):
        raise TagLawError(
            "plan_of: this allocator did not come from "
            "TagPlan.emit_allocator(), so it carries no tag plan; an emit "
            "is driven by BuiltModel.emit (ADR 0114 D4, amended)."
        )
    return plan


def plan_or_standalone(tags: TagAllocator) -> TagPlan | None:
    """The plan of a bridge emit's allocator, or ``None`` for a plain one.

    The two-way emit helpers (ADR 0114 D4, amended; until K1-3d S6 drops
    ``tags`` from their signatures) read the plan when handed
    :meth:`TagPlan.emit_allocator`'s fork, and plan their own rows from
    a plain :class:`TagAllocator` (a direct caller, or the compose
    replay under its ledger waiver) through the same ``plan_*`` loop.
    ``None`` therefore means "plan from ``tags``", never "skip". Any other
    allocator (a plain :meth:`TagAllocator.fork`, a fork for another
    origin, the frozen planner allocator) raises :class:`TagLawError`.
    """
    # A fresh allocator is neither forked nor frozen; ``_forked`` is set
    # by TagAllocator.fork and read only here and in plan_of's sibling
    # check (a module of the same package as tag_allocator).
    if not tags._forked and not tags.frozen:
        return None
    return plan_of(tags)


def plan_inputs(bm: BuiltModel) -> tuple[object, ...]:
    """Every public field of ``bm``, in field order: what a plan reads.

    The planner reads the primitives, their tags, the FEM snapshot and
    ``element_tags`` today, and the families still to migrate read the
    records; taking every public field covers them all.
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

    # Contacts: every interaction, in the order its mode's emit writes
    # it. No other family mints ``contactSurface`` or ``contact``.
    contacts = ContactTagPlan(contacts=_plan_contacts(bm, mode, tags))

    # Regions: every named, damping and recorder region, in the order its
    # mode's emit writes them. No other family mints ``region``.
    sites, named = bm._region_sites(mode, ordered)
    regions = RegionTagPlan(
        regions=plan_regions(sites, tags), named=named, fem=bm.fem)

    # The other families are pending: their tags are still minted at
    # emit time, from TagPlan.emit_allocator().
    tags.freeze()
    return TagPlan(
        mode=mode,
        allocator=tags,
        elements=elements,
        transforms=transforms,
        regions=regions,
        parameters=ParameterTagPlan(),
        mp_elements=MPElementTagPlan(),
        interfaces=InterfaceTagPlan(),
        contacts=contacts,
        inputs=plan_inputs(bm),
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
