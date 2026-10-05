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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from .tag_allocator import TagAllocator, TagLawError

if TYPE_CHECKING:
    from ..apesees import BuiltModel
    from .build import ElementPlanRows
    from .types import Element

#: One planned emission: ``(kind, tag)`` in the emit's verb vocabulary.
TagRow = tuple[str, int]


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
    """Orientation fan-out: one ``geomTransf`` per distinct ``vecxz``."""

    FAMILY: ClassVar[str] = "transforms"
    KINDS: ClassVar[frozenset[str]] = frozenset({"geomTransf"})


@dataclass(frozen=True, slots=True)
class RegionTagPlan(FamilyTagPlan):
    """Named regions, damping regions and recorder filter/energy regions."""

    FAMILY: ClassVar[str] = "regions"
    KINDS: ClassVar[frozenset[str]] = frozenset({"region"})


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
    """Contact surfaces and contact interactions (face and rigid plane)."""

    FAMILY: ClassVar[str] = "contacts"
    KINDS: ClassVar[frozenset[str]] = frozenset({"contactSurface", "contact"})


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
    primitive tags and every planned family's mints.
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
        reserve_fem_element_tags,
        topological_order,
    )
    from .types import Element

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

    # Elements: every spec's fan-out, in the emit's topological order.
    # Each emit path made this allocation once, before any other
    # element-kind mint, so planning it here numbers it as they did.
    elements = ElementTagPlan(specs=tuple(allocate_element_tags(
        [p for p in topological_order(bm.primitives)
         if isinstance(p, Element)],
        bm.fem, tags, element_tags=bm.element_tags,
    )))

    # The other families are pending: their tags are still minted at
    # emit time, from TagPlan.emit_allocator().
    tags.freeze()
    return TagPlan(
        mode=mode,
        allocator=tags,
        elements=elements,
        transforms=TransformTagPlan(),
        regions=RegionTagPlan(),
        parameters=ParameterTagPlan(),
        mp_elements=MPElementTagPlan(),
        interfaces=InterfaceTagPlan(),
        contacts=ContactTagPlan(),
    )
