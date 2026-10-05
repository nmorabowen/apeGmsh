"""The tag plan's oracle (ADR 0114 D4, amended): the plan is what the emit writes.

For every model the K1-3 tag-law pins drive (``_tag_streams.models()``,
plus the split frame) and the emit mode it takes, ``plan_tags`` must
plan exactly the ``(kind, tag)`` rows the emit writes. The oracle is the
emitted stream itself, read by the K1-3 tap: for each tag family,
``sorted(plan.<family>.stream())`` equals the tapped rows that family
owns. A family's case is ``xfail(strict=True)`` until the slice that moves
its allocation loop into ``plan_tags`` sets its ``MIGRATED`` flag; the
marker is read from that flag, so the case turns into a real comparison
in the same commit.

A tapped row belongs to a family by its verb (``FAMILY_VERBS``), less the
rows a registered primitive emits under its own tag (a transform's first
``geomTransf``, a material), less the rows of the other migrated families
sharing that verb. The ``element`` verb is shared by the element,
MP-element and interface families, and a pending sharer's rows cannot be
told apart from the rest. So while any family sharing one of its verbs is
pending, a migrated family's plan must be a sub-multiset of its rows (no
planned row the emit does not write); once every sharer has migrated the
comparison is exact. ``test_whole_plan_equals_tapped_stream`` is the
exact check over every family at once, and turns on with the last
migration.

Beside the oracle, this file pins what S1 ships: ``TagAllocator.freeze()``
and ``fork()``, ``TagLawError``, and a plan seeded as the emit seeds its
allocator (the emit's first mint in each kind follows the plan's counter).
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import pytest

from apeGmsh.opensees._internal.tag_allocator import TagAllocator, TagLawError
from apeGmsh.opensees._internal.tag_plan import (
    FAMILIES,
    FAMILY_PLANS,
    TagMode,
    TagPlan,
    emit_mode,
    plan_tags,
)
from apeGmsh.opensees.apesees import _kind_of
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.contract import _tag_streams as ts

#: The tap verbs each family's tags are emitted under.
FAMILY_VERBS: dict[str, frozenset[str]] = {
    "elements": frozenset({"element"}),
    "transforms": frozenset({"geomTransf"}),
    "regions": frozenset({"region"}),
    "parameters": frozenset({"addToParameter", "update_parameter"}),
    "mp_elements": frozenset(
        {"element", "embeddedNode", "embedded_rebar", "embedded_node"}),
    "interfaces": frozenset({"element", "uniaxialMaterial"}),
    "contacts": frozenset({"contact_surface", "contact", "contact_plane"}),
}

#: Tap verbs a registered primitive also emits under its own tag, with the
#: allocator kind it was registered in.
SEEDED_VERBS: dict[str, str] = {
    "geomTransf": "geomTransf",
    "uniaxialMaterial": "uniaxialMaterial",
}

#: Allocator kind -> the tap verbs whose tags that kind's mints carry.
KIND_VERBS: dict[str, frozenset[str]] = {
    "element": frozenset({"element"}),
    "geomTransf": frozenset({"geomTransf"}),
    "region": frozenset({"region"}),
    "parameter": frozenset({"addToParameter", "update_parameter"}),
}

_MODELS = ts.models()
_SPLIT = "two_module_frame/split"
CASES: tuple[str, ...] = (*sorted(_MODELS), _SPLIT)


def _verb(kind: str) -> str:
    return kind.split(":", 1)[0]


@lru_cache(maxsize=None)
def _case(name: str) -> tuple[Any, tuple[tuple[str, int], ...], TagPlan]:
    """``(BuiltModel, tapped stream, plan)`` for one case, built once."""
    if name == _SPLIT:
        bm = ts.split_model().build()
        cls: type = TclEmitter
        split = True
    else:
        bm = _MODELS[name]().build()
        cls = RecordingEmitter
        split = False
    stream = ts.emit_stream(bm, cls, split=split)
    mode = emit_mode(
        bm, split=split,
        supports_partitions=getattr(cls, "supports_partitions", True),
    )
    return bm, tuple(stream), plan_tags(bm, mode)


def _seeded(bm: Any) -> set[tuple[str, int]]:
    """``(allocator kind, tag)`` of every registered primitive."""
    return {(_kind_of(p), bm.tag_for[id(p)]) for p in bm.primitives}


def _derived_rows(name: str, verbs: frozenset[str]) -> Counter[tuple[str, int]]:
    """Tapped rows under ``verbs``, less those of registered primitives."""
    bm, stream, _ = _case(name)
    seeded = _seeded(bm)
    return Counter(
        r for r in stream
        if _verb(r[0]) in verbs
        and not (
            _verb(r[0]) in SEEDED_VERBS
            and (SEEDED_VERBS[_verb(r[0])], r[1]) in seeded
        )
    )


def _family_rows(
    name: str, family: str, plan: TagPlan,
) -> tuple[list[tuple[str, int]], bool]:
    """The tapped rows ``family`` owns, and whether they are exact.

    Not exact while a family sharing one of its verbs is pending: those
    rows then hold the pending family's rows too (module docstring).
    """
    verbs = FAMILY_VERBS[family]
    rows = _derived_rows(name, verbs)
    exact = True
    for other in FAMILIES:
        if other == family or not FAMILY_VERBS[other] & verbs:
            continue
        if FAMILY_PLANS[other].MIGRATED:
            rows -= Counter(plan.family(other).stream())
        else:
            exact = False
    return sorted(rows.elements()), exact


def _family_param(name: str, family: str) -> Any:
    marks = []
    if not FAMILY_PLANS[family].MIGRATED:
        marks.append(pytest.mark.xfail(
            raises=NotImplementedError, strict=True,
            reason=f"{family}: tags still minted at emit time",
        ))
    return pytest.param(name, family, marks=marks, id=f"{name}-{family}")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, family",
    [_family_param(n, f) for n in CASES for f in FAMILIES],
)
def test_plan_equals_tapped_stream(name: str, family: str) -> None:
    _, _, plan = _case(name)
    planned = Counter(plan.family(family).stream())
    tapped, exact = _family_rows(name, family, plan)
    extra = sorted((planned - Counter(tapped)).elements())
    missing = sorted((Counter(tapped) - planned).elements())
    assert not extra, (
        f"{name} [{plan.mode}]: the {family} plan has rows the emit does "
        f"not write: {extra}"
    )
    if exact:
        assert not missing, (
            f"{name} [{plan.mode}]: the emit writes {family} rows the plan "
            f"does not hold: {missing}"
        )


_ALL_MIGRATED = all(cls.MIGRATED for cls in FAMILY_PLANS.values())


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=[] if _ALL_MIGRATED else [pytest.mark.xfail(
        raises=NotImplementedError, strict=True,
        reason="families still pending: " + ", ".join(
            f for f in FAMILIES if not FAMILY_PLANS[f].MIGRATED),
    )])
    for n in CASES
])
def test_whole_plan_equals_tapped_stream(name: str) -> None:
    """``sorted(plan.stream())`` is every derived row the emit writes."""
    _, _, plan = _case(name)
    planned = sorted(plan.stream())
    verbs = frozenset().union(*FAMILY_VERBS.values())
    assert planned == sorted(_derived_rows(name, verbs).elements()), (
        f"{name} [{plan.mode}]: the plan and the emit disagree"
    )


@pytest.mark.parametrize("name", CASES)
def test_plan_seed_is_the_emit_seed(name: str) -> None:
    """The emit's mints in each kind run on from the plan's counter.

    ``plan_tags`` seeds its allocator as ``BuiltModel.emit`` seeds its
    own. While a kind is minted at emit time, its emitted tags (less any
    registered primitive's) are therefore the contiguous run that starts
    one above the plan's counter.
    """
    bm, stream, plan = _case(name)
    seeded = _seeded(bm)
    for kind, verbs in KIND_VERBS.items():
        minted = sorted({
            tag for k, tag in stream
            if _verb(k) in verbs and (kind, tag) not in seeded
        })
        if not minted:
            continue
        first = plan.allocator.last(kind) + 1
        assert minted == list(range(first, first + len(minted))), (
            f"{name}: emitted {kind} tags {minted} are not the run from "
            f"{first}, one above the plan's seeded counter"
        )


def test_corpus_reaches_every_mode() -> None:
    modes = {_case(n)[2].mode for n in CASES}
    for split, partitioned, staged in (
        (False, False, False), (False, True, False),
        (False, False, True), (False, True, True), (True, False, False),
    ):
        assert TagMode(split, partitioned, staged) in modes


def test_two_rank_regions_number_differently_by_mode() -> None:
    """The invariant-8 fixture: rank order is not declaration order.

    ``east`` (rank 1's nodes) is declared before ``west`` (rank 0's).
    The flat deck numbers the named regions in declaration order; the
    partitioned deck numbers the MPCO filter region first and then each
    named region on the first rank that emits it. The region plan has to
    reproduce both, until the canonical-numbering slice makes them one.
    """
    def regions(name: str) -> list[int]:
        return [t for k, t in _case(name)[1] if k == "region"]

    assert regions("two_rank_regions/flat") == [1, 2, 3]
    assert regions("two_rank_regions/partitioned") == [2, 1, 3, 1]


# ---------------------------------------------------------------------------
# plan_tags
# ---------------------------------------------------------------------------


def test_plan_tags_freezes_its_allocator() -> None:
    _, _, plan = _case("two_column_frame/flat")
    assert plan.allocator.frozen
    with pytest.raises(TagLawError):
        plan.allocator.allocate("element")


def test_plan_seeds_every_registered_tag() -> None:
    bm, _, plan = _case("initial_stress_frame/flat")
    for prim in bm.primitives:
        assert plan.allocator.tag_for(prim) == bm.tag_for[id(prim)]


def test_pending_plan_freezes_no_kind() -> None:
    _, _, plan = _case("two_column_frame/flat")
    assert plan.migrated == tuple(
        f for f in FAMILIES if FAMILY_PLANS[f].MIGRATED)
    if not plan.migrated:
        assert plan.frozen_kinds == frozenset()
        with pytest.raises(NotImplementedError):
            plan.stream()


def test_emit_allocator_continues_the_plan() -> None:
    _, _, plan = _case("arch_with_orientation_fan_out/flat")
    tags = plan.emit_allocator()
    assert not tags.frozen
    assert tags.frozen_kinds == plan.frozen_kinds
    for kind in ("element", "geomTransf", "region"):
        if kind not in plan.frozen_kinds:
            assert tags.allocate(kind) == plan.allocator.last(kind) + 1
    # The plan's own allocator is untouched by the fork's mints.
    assert plan.allocator.last("element") == 1


def test_unknown_family_raises() -> None:
    _, _, plan = _case("two_column_frame/flat")
    with pytest.raises(KeyError, match="unknown tag family"):
        plan.family("nodes")


def test_plan_tags_refuses_a_non_mode() -> None:
    bm, _, _ = _case("two_column_frame/flat")
    with pytest.raises(TypeError, match="TagMode"):
        plan_tags(bm, (False, False, False))  # type: ignore[arg-type]


def test_tag_plan_refuses_an_open_allocator() -> None:
    _, _, plan = _case("two_column_frame/flat")
    fields = {f: getattr(plan, f) for f in FAMILIES}
    with pytest.raises(TagLawError, match="must be frozen"):
        TagPlan(mode=plan.mode, allocator=TagAllocator(), **fields)


def test_pending_subplan_cannot_carry_rows() -> None:
    for name, cls in FAMILY_PLANS.items():
        if cls.MIGRATED:
            continue
        with pytest.raises(TagLawError, match="pending"):
            cls(rows=(("region", 1),))


# ---------------------------------------------------------------------------
# TagAllocator.freeze() / fork()
# ---------------------------------------------------------------------------


def _seeded_allocator() -> TagAllocator:
    tags = TagAllocator()
    tags.allocate("element")
    tags.allocate_block("element", 3)
    tags.allocate("region")
    return tags


@pytest.mark.parametrize("call", [
    lambda t: t.allocate("element"),
    lambda t: t.allocate("brand_new_kind"),
    lambda t: t.allocate_block("element", 2),
    lambda t: t.allocate_block("element", 0),
    lambda t: t.allocate_for(object(), "element"),
    lambda t: t.reserve_through("element", 1),
    lambda t: t.reset(),
], ids=["allocate", "allocate_new_kind", "allocate_block",
        "allocate_block_0", "allocate_for", "reserve_through", "reset"])
def test_frozen_allocator_refuses_every_mint(call: Any) -> None:
    tags = _seeded_allocator()
    tags.freeze()
    with pytest.raises(TagLawError):
        call(tags)
    assert tags.last("element") == 4 and tags.last("region") == 1


def test_frozen_allocator_refuses_allocate_for_of_a_known_primitive() -> None:
    tags = TagAllocator()
    prim = object()
    assert tags.allocate_for(prim, "uniaxialMaterial") == 1
    tags.freeze()
    with pytest.raises(TagLawError):
        tags.allocate_for(prim, "uniaxialMaterial")
    assert tags.tag_for(prim) == 1


def test_freeze_is_idempotent_and_reads_stay_legal() -> None:
    tags = _seeded_allocator()
    assert not tags.frozen
    tags.freeze()
    tags.freeze()
    assert tags.frozen
    assert tags.last("element") == 4
    assert tags.last("never_minted") == 0


def test_fork_freezes_only_the_named_kinds() -> None:
    parent = _seeded_allocator()
    parent.freeze()
    child = parent.fork({"element"})
    assert not child.frozen
    assert child.frozen_kinds == frozenset({"element"})
    with pytest.raises(TagLawError, match="planned and frozen"):
        child.allocate("element")
    with pytest.raises(TagLawError):
        child.allocate_block("element", 1)
    with pytest.raises(TagLawError):
        child.reserve_through("element", 99)
    assert child.allocate("region") == 2
    assert child.allocate("geomTransf") == 1
    # The parent is unchanged by the child's mints.
    assert parent.last("region") == 1
    assert parent.last("geomTransf") == 0


def test_fork_keeps_assignments_and_inherits_frozen_kinds() -> None:
    parent = TagAllocator()
    prim = object()
    parent.allocate_for(prim, "section")
    child = parent.fork({"region"})
    assert child.tag_for(prim) == 1
    grandchild = child.fork({"element"})
    assert grandchild.frozen_kinds == frozenset({"region", "element"})
    with pytest.raises(TagLawError):
        grandchild.allocate("region")


def test_fork_with_no_kinds_is_an_open_copy() -> None:
    parent = _seeded_allocator()
    parent.freeze()
    child = parent.fork()
    assert child.allocate("element") == 5
    assert child.allocate_for(object(), "element") == 6


@pytest.mark.parametrize("bad", ["", 3, None])
def test_fork_refuses_a_non_kind(bad: Any) -> None:
    with pytest.raises(TypeError, match="non-empty str"):
        TagAllocator().fork({bad})


def test_unfrozen_allocator_is_unchanged() -> None:
    """Before a freeze the allocator counts exactly as it always has."""
    tags = TagAllocator()
    assert tags.allocate("element") == 1
    assert tags.allocate_block("element", 3) == 2
    assert tags.allocate_block("element", 0) == 5
    tags.reserve_through("element", 10)
    assert tags.allocate("element") == 11
    prim = object()
    assert tags.allocate_for(prim, "region") == 1
    assert tags.allocate_for(prim, "region") == 1
    tags.reset()
    assert tags.last("element") == 0 and tags.tag_for(prim) is None
