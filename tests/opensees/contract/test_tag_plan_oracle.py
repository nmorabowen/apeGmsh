"""The tag plan's oracle (ADR 0114 D4, amended): the plan is what the emit writes.

For every model the K1-3 tag-law pins drive (``_tag_streams.models()``,
plus the split frame) and the emit mode it takes, the plan the emit read
(``BuiltModel.emit`` memoises one per mode) must plan exactly the derived
``(kind, tag)`` rows the emit writes. The oracle is the emit itself, read
twice:

* the K1-3 tap records the ``(kind, tag)`` row of every emitted verb;
* a mint log records every ``TagAllocator`` call the emit makes, with the
  function that made it. The calls on the plan's own allocator are the
  planning (the seeding, then the migrated families' mints); every other
  call is an emit-time mint. ``_MINT_SITES`` names the family of each
  minting function (an unknown one raises), so every tag has an owner.

A tapped row of a derived verb (``VERB_KIND``) then belongs to:

* nobody, when it is a registered primitive's own tag (a material, a
  transform's first ``geomTransf``);
* the family whose function minted it at emit time;
* while the element family is pending, the element family, for an
  ``element`` row nobody minted inside the emit's reserved range (a FEM
  id under ``element_tags="fem"``);
* otherwise the plan: a tag the emit read rather than minted, which some
  migrated family's plan must hold.

``test_plan_equals_tapped_stream`` runs per family. A pending family's
case is ``xfail(strict=True)``: its ``stream()`` raises. A migrated
family's case fails if the emit still mints any of its tags, and otherwise
compares its plan, exactly, with the planned rows of its kinds less the
other migrated families' plans in those kinds. Rows compare by
``(verb, tag)``: an element spec picks its type token in its own
``_emit``, so the plan's element rows carry the bare verb. The
``MIGRATED`` flag in ``tag_plan.py`` drives the marker, so a migration
slice turns its cases into real comparisons in the same commit;
``test_every_family_has_rows`` keeps every family's comparison
non-vacuous, and ``test_short_or_empty_element_plan_fails_the_oracle``
proves the comparison catches a plan that drops rows.

Beside the oracle, this file pins what S1 and S2 ship:
``TagAllocator.freeze()`` and ``fork()``, ``TagLawError``, a plan seeded
exactly as the emit used to seed its allocator (``element_tags="fem"``
included), the plan each emit path reads (``plan_of``), and the refusal
of ``reset()`` on every fork.
"""
from __future__ import annotations

import inspect
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import pytest

from apeGmsh.opensees._internal.tag_allocator import TagAllocator, TagLawError
from apeGmsh.opensees._internal.tag_plan import (
    FAMILIES,
    FAMILY_PLANS,
    TagMode,
    TagPlan,
    emit_mode,
    plan_of,
    plan_tags,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.contract import _tag_streams as ts

Row = tuple[str, int]

#: Tap verb -> the allocator kind of its tag, for every verb whose tag the
#: build mints (or, for a primitive's own verb, may mint).
VERB_KIND: dict[str, str] = {
    "element": "element",
    "embeddedNode": "element",
    "embedded_rebar": "element",
    "embedded_node": "element",
    "geomTransf": "geomTransf",
    "uniaxialMaterial": "uniaxialMaterial",
    "region": "region",
    "addToParameter": "parameter",
    "update_parameter": "parameter",
    "flip_element_stage": "parameter",
    "contact_surface": "contactSurface",
    "contact": "contact",
    "contact_plane": "contact",
}

#: Tap verbs whose tag is a node id, a registered primitive's tag that
#: no family mints, or a reference to a tag minted elsewhere.
NON_DERIVED_VERBS: frozenset[str] = frozenset({
    "node", "fix", "mass", "load", "sp", "remove_element", "timeSeries",
    "pattern_open", "nDMaterial", "section", "section_open",
    "beamIntegration", "damping",
})

#: The function that mints at emit time -> the family its tags belong to.
_MINT_SITES: dict[str, str] = {
    "allocate_element_tags": "elements",
    "emit_element_spec": "elements",
    "plan_transform_specs": "transforms",
    "_emit_rayleigh": "regions",
    "_emit_damping_attach": "regions",
    "_emit_regions": "regions",
    "_emit_stage_regions": "regions",
    "_emit_stage_regions_partitioned": "regions",
    "_emit_regions_partitioned": "regions",
    "_plan_partitioned_mpco_recorders": "regions",
    "materialize": "regions",
    "emit_initial_stress_global": "parameters",
    "emit_update_parameters": "parameters",
    "emit_activate_absorbing": "parameters",
    "emit_reinforce_ties": "mp_elements",
    "emit_embed_ties": "mp_elements",
    "emit_rebar_elements": "mp_elements",
    "_emit_rigid_body_elements": "mp_elements",
    "_emit_kinematic_couplings": "mp_elements",
    "_emit_one_interpolation": "mp_elements",
    "allocate_interface_tags": "interfaces",
    "plan_contacts": "contacts",
}

#: The owner of a row some migrated family's plan must hold.
PLANNED = "<planned>"


def _contact_ranks_fem(n_ranks: int, *, partitioned: bool = True) -> Any:
    """A truss stub with one contact and one contact plane per rank.

    Rank ``r`` natively holds the master pad ``b+1..b+4`` and the plane's
    slave node ``b+7`` (``b = 100 (r + 1)``), so it owns contact ``c{r}``
    and plane ``p{r}``. The contact's slave nodes ``b+5, b+6`` live on the
    next rank, so the owner declares them as ghosts. Contacts and planes
    are declared in reverse rank order, so the partitioned deck, which
    numbers them rank by rank, numbers them differently from the flat
    one. The stub's elements are not iterable, so the owner pick is the
    node tally (ADR 0092 INV-1).
    """
    from apeGmsh._kernel.records._constraints import (
        ContactPlaneRecord,
        ContactRecord,
    )

    from tests.opensees.fixtures.fem_stub import (
        FEMStub,
        _ElementGroupView,
        _ElementsStub,
        _NodesStub,
    )

    ids: list[int] = []
    coords: list[tuple[float, float, float]] = []
    bar_ids: list[int] = []
    bars: list[tuple[int, int]] = []
    rank_nodes: dict[int, list[int]] = {r: [] for r in range(n_ranks)}
    rank_elems: dict[int, list[int]] = {r: [] for r in range(n_ranks)}
    contacts: list[Any] = []
    planes: list[Any] = []
    for r in range(n_ranks):
        b, x, nxt = 100 * (r + 1), 10.0 * r, (r + 1) % n_ranks
        ids += [b + k for k in range(1, 8)]
        coords += [(x, 0.0, 0.0), (x + 1, 0.0, 0.0), (x + 1, 1.0, 0.0),
                   (x, 1.0, 0.0), (x + 0.2, 0.2, 0.5), (x + 0.8, 0.8, 0.5),
                   (x, 0.0, -1.0)]
        rank_nodes[r] += [b + 1, b + 2, b + 3, b + 4, b + 7]
        rank_nodes[nxt] += [b + 5, b + 6]
        e = 10 * (r + 1)
        for eid, conn, rank in ((e + 1, (b + 1, b + 2), r),
                                (e + 2, (b + 3, b + 7), r),
                                (e + 3, (b + 5, b + 6), nxt)):
            bar_ids.append(eid)
            bars.append(conn)
            rank_elems[rank].append(eid)
        contacts.insert(0, ContactRecord(
            kind="contact", name=f"c{r}", formulation="nts",
            master_faces=np.array([[b + 1, b + 2, b + 3, b + 4]],
                                  dtype=np.int64),
            master_nps=4, slave_nodes=[b + 5, b + 6],
            kn=1.0e6, kt=0.0, mu=0.0,
        ))
        planes.insert(0, ContactPlaneRecord(
            kind="contact_plane", name=f"p{r}", slave_nodes=[b + 7],
            normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, -2.0), kn=1.0e6,
        ))
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs={}),
        elements=_ElementsStub(elem_pgs={"Bars": _ElementGroupView(
            ids=tuple(bar_ids), connectivity=tuple(bars))}),
    )
    if partitioned:
        fem.set_partitions([
            (r, rank_nodes[r], rank_elems[r]) for r in range(n_ranks)])
    fem.elements.contacts = contacts
    fem.elements.contact_planes = planes
    return fem


def _contact_ranks(n_ranks: int, *, partitioned: bool = True) -> Any:
    from typing import cast

    from apeGmsh.opensees import apeSees

    ops = apeSees(cast(Any, _contact_ranks_fem(
        n_ranks, partitioned=partitioned)))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    return ops


#: The contact cases: 2 and 4 ranks, and the 4-rank stub emitted flat.
_CONTACT_MODELS: dict[str, Callable[[], Any]] = {
    "contact_ranks_2/partitioned": lambda: _contact_ranks(2),
    "contact_ranks_4/partitioned": lambda: _contact_ranks(4),
    "contact_ranks_4/flat": lambda: _contact_ranks(4, partitioned=False),
}

_MODELS = {**ts.models(), **_CONTACT_MODELS}
_SPLIT = "two_module_frame/split"
CASES: tuple[str, ...] = (*sorted(_MODELS), _SPLIT)


def _verb(kind: str) -> str:
    return kind.split(":", 1)[0]


# ---------------------------------------------------------------------------
# The mint log
# ---------------------------------------------------------------------------


class _Op(NamedTuple):
    allocator: int          # id() of the TagAllocator
    method: str
    args: tuple[Any, ...]
    result: Any
    caller: str             # the function that called the method


_LOGGED = ("allocate", "allocate_block", "allocate_for", "reserve_through")


@contextmanager
def _mint_log() -> Iterator[list[_Op]]:
    """Record every outermost ``TagAllocator`` call made inside the block."""
    log: list[_Op] = []
    depth = [0]
    originals = {m: getattr(TagAllocator, m) for m in _LOGGED}

    def wrap(method: str, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: TagAllocator, *args: Any) -> Any:
            caller = sys._getframe(1).f_code.co_name
            depth[0] += 1
            try:
                out = orig(self, *args)
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                log.append(_Op(id(self), method, args, out, caller))
            return out
        return wrapper

    for m, f in originals.items():
        setattr(TagAllocator, m, wrap(m, f))
    try:
        yield log
    finally:
        for m, f in originals.items():
            setattr(TagAllocator, m, f)


class Case(NamedTuple):
    bm: Any
    stream: tuple[Row, ...]
    plan: TagPlan                         # the plan the emit read
    seed: dict[str, int]                  # the plan's counters before its mints
    seeded: frozenset[tuple[str, int]]    # (kind, tag) of each seeded primitive
    planned: dict[str, list[int]]         # kind -> tags the planner minted
    mints: dict[tuple[str, int], str]     # (kind, tag) minted at emit -> family
    minted: dict[str, list[int]]          # kind -> tags minted at emit, in order


def _mint_range(op: _Op) -> range:
    first = int(op.result)
    n = 1 if op.method == "allocate" else int(op.args[1])
    return range(first, first + n)


def _mint_family(name: str, op: _Op) -> str:
    if op.caller not in _MINT_SITES:
        raise AssertionError(
            f"{name}: {op.caller}() mints {op.args[0]!r}; name its family "
            "in _MINT_SITES"
        )
    return _MINT_SITES[op.caller]


def _read_log(name: str, log: list[_Op], plan: TagPlan) -> tuple[
    dict[str, int], frozenset[tuple[str, int]], dict[str, list[int]],
    dict[tuple[str, int], str], dict[str, list[int]],
]:
    """Split the log into the planning and the emit-time mints."""
    planner = id(plan.allocator)
    assert any(op.allocator == planner for op in log), (
        f"{name}: the emit's plan was not made inside this emit")
    emit_allocators = {op.allocator for op in log} - {planner}
    assert len(emit_allocators) <= 1, (
        f"{name}: the emit minted from {len(emit_allocators)} allocators "
        "besides the plan's; it mints from one emit_allocator() fork"
    )
    seed: dict[str, int] = {}
    seeded: set[tuple[str, int]] = set()
    planned: dict[str, list[int]] = {}
    mints: dict[tuple[str, int], str] = {}
    minted: dict[str, list[int]] = {}
    for op in log:
        if op.allocator != planner:
            assert op.method not in ("allocate_for", "reserve_through"), (
                f"{name}: {op.caller}() seeds the emit allocator with "
                f"{op.method}; the plan seeds it")
            family = _mint_family(name, op)
            for tag in _mint_range(op):
                mints[(op.args[0], tag)] = family
                minted.setdefault(op.args[0], []).append(tag)
            continue
        if op.method in ("allocate_for", "reserve_through"):
            assert not planned, (
                f"{name}: {op.method} after the planner's first mint")
            if op.method == "allocate_for":
                kind, tag = op.args[1], int(op.result)
                seeded.add((kind, tag))
            else:
                kind, tag = op.args[0], int(op.args[1])
            seed[kind] = max(seed.get(kind, 0), tag)
            continue
        family = _mint_family(name, op)
        assert FAMILY_PLANS[family].MIGRATED, (
            f"{name}: the planner mints for pending family {family}")
        planned.setdefault(op.args[0], []).extend(_mint_range(op))
    return seed, frozenset(seeded), planned, mints, minted


@lru_cache(maxsize=None)
def _case(name: str) -> Case:
    """Everything one case's oracle reads, built once."""
    if name == _SPLIT:
        bm = ts.split_model().build()
        cls: type = TclEmitter
        split = True
    else:
        bm = _MODELS[name]().build()
        cls = RecordingEmitter
        split = False
    mode = emit_mode(
        bm, split=split,
        supports_partitions=getattr(cls, "supports_partitions", True),
    )
    assert mode not in bm._tag_plans
    with _mint_log() as log:
        stream = ts.emit_stream(bm, cls, split=split)
    plan = bm._tag_plans[mode]
    seed, seeded, planned, mints, minted = _read_log(name, log, plan)
    return Case(bm, tuple(stream), plan, seed, seeded, planned, mints,
                minted)


def _owner(case: Case, row: Row) -> str | None:
    """The family a tapped row belongs to (module docstring)."""
    kind = VERB_KIND.get(_verb(row[0]))
    if kind is None:
        return None
    key = (kind, row[1])
    if key in case.seeded and _verb(row[0]) != "element":
        return None
    if key in case.mints:
        return case.mints[key]
    if (kind == "element" and not FAMILY_PLANS["elements"].MIGRATED
            and row[1] <= case.seed.get("element", 0)):
        return "elements"      # a FEM id, inside the reserved range
    return PLANNED


def _rows(case: Case, owner: str, kinds: frozenset[str]) -> Counter[Row]:
    """The tapped rows of ``owner`` in ``kinds``, as ``(verb, tag)``."""
    return Counter(
        (_verb(r[0]), r[1]) for r in case.stream
        if _owner(case, r) == owner and VERB_KIND[_verb(r[0])] in kinds
    )


def _in_kinds(rows: Any, kinds: frozenset[str]) -> Counter[Row]:
    """Planned ``rows`` in ``kinds``, as ``(verb, tag)``."""
    return Counter(
        (_verb(r[0]), r[1]) for r in rows if VERB_KIND[_verb(r[0])] in kinds)


def _family_mismatch(
    case: Case, family: str, planned_rows: Any,
) -> str | None:
    """Why ``planned_rows`` is not ``family``'s rows of ``case``, or None."""
    kinds = FAMILY_PLANS[family].KINDS
    still_minted = _rows(case, family, kinds)
    if still_minted:
        return (f"{family} is MIGRATED, but the emit still mints its rows "
                f"{sorted(still_minted.elements())}")
    others: Counter[Row] = Counter()
    for other in FAMILIES:
        cls = FAMILY_PLANS[other]
        if other != family and cls.MIGRATED and cls.KINDS & kinds:
            others += _in_kinds(case.plan.family(other).stream(), kinds)
    pool = _rows(case, PLANNED, kinds)
    if others - pool:
        return ("other families plan rows the emit does not write: "
                f"{sorted((others - pool).elements())}")
    expected = pool - others
    planned = _in_kinds(planned_rows, kinds)
    if planned != expected:
        return (f"[{case.plan.mode}] the {family} plan and the emit "
                f"disagree: planned only "
                f"{sorted((planned - expected).elements())}, emitted only "
                f"{sorted((expected - planned).elements())}")
    return None


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
    case = _case(name)
    planned = case.plan.family(family).stream()
    problem = _family_mismatch(case, family, planned)
    assert problem is None, f"{name}: {problem}"


def test_short_or_empty_element_plan_fails_the_oracle() -> None:
    """The element comparison is not vacuous: a dropped row is caught.

    Every case with elements must reject the element plan with its last
    row dropped, and with every row dropped, while accepting the real one.
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.elements.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "elements", rows) is None, name
        assert _family_mismatch(case, "elements", rows[:-1]), name
        assert _family_mismatch(case, "elements", ()), name
    assert checked >= 10


def test_short_or_empty_transform_plan_fails_the_oracle() -> None:
    """The transform comparison is not vacuous: a dropped row is caught.

    Every case whose orientation fan-out plans a tag must reject the
    transform plan with its last row dropped, and with every row
    dropped, while accepting the real one.
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.transforms.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "transforms", rows) is None, name
        assert _family_mismatch(case, "transforms", rows[:-1]), name
        assert _family_mismatch(case, "transforms", ()), name
    assert checked >= 4      # the arch fixture, in each of its four modes


# ---------------------------------------------------------------------------
# The transform family (K1-3d S3a)
# ---------------------------------------------------------------------------


def _arch_transform_inputs(case: Case) -> tuple[list[Any], list[Any]]:
    from apeGmsh.opensees._internal.build import topological_order
    from apeGmsh.opensees._internal.types import Element, GeomTransf

    ordered = topological_order(case.bm.primitives)
    return ([p for p in ordered if isinstance(p, GeomTransf)],
            [p for p in ordered if isinstance(p, Element)])


def test_geomtransf_is_frozen_in_the_emit_fork() -> None:
    """A stray ``geomTransf`` mint at emit time raises where it is."""
    plan = _case("arch_with_orientation_fan_out/flat").plan
    assert "geomTransf" in plan.frozen_kinds
    tags = plan.emit_allocator()
    with pytest.raises(TagLawError, match="planned and frozen"):
        tags.allocate("geomTransf")
    with pytest.raises(TagLawError):
        tags.allocate_block("geomTransf", 1)


def test_emit_transform_specs_is_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    The plain-allocator path (a direct caller, until K1-3d S6) writes the
    same lines and overrides as the planned path; any other fork raises.
    """
    from apeGmsh.opensees._internal.build import emit_transform_specs

    case = _case("arch_with_orientation_fan_out/flat")
    bm, plan = case.bm, case.plan
    transforms, elements = _arch_transform_inputs(case)

    def run(tags: TagAllocator) -> tuple[list[Row], Any]:
        em = ts.tapped(RecordingEmitter)()
        em.tap = []
        overrides = emit_transform_specs(
            transforms, elements, em, bm.fem, tags, bm.tag_for, ndm=bm.ndm)
        return list(em.tap), overrides

    planned_rows, planned_over = run(plan.emit_allocator())
    plain_rows, plain_over = run(_seeded_like_the_planner(bm))
    assert planned_rows == plain_rows and planned_over == plain_over
    assert planned_over == plan.transforms.fanout_for(
        transforms, bm.fem).overrides
    assert planned_over      # the arch fans out past the spec's own tag

    for other in (plan.allocator.fork(), plan.allocator,
                  plan.allocator.fork({"geomTransf"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            run(other)


def test_emit_transform_specs_refuses_a_fork_of_another_models_plan() -> None:
    """A real foreign plan's fork is refused as such, not as a miscount.

    The same arch recipe built twice gives two models with their own FEM
    snapshots. Model B's emit handed model A's emit allocator must say
    the plan is another model's, before any count or order check.
    """
    from apeGmsh.opensees._internal.build import emit_transform_specs

    case = _case("arch_with_orientation_fan_out/flat")
    other = _MODELS["arch_with_orientation_fan_out/flat"]().build()
    assert other.fem is not case.bm.fem
    o_transforms, o_elements = _arch_transform_inputs(_Inputs(other))
    em = RecordingEmitter()
    with pytest.raises(TagLawError, match="another model's tag plan"):
        emit_transform_specs(
            o_transforms, o_elements, em, other.fem,
            case.plan.emit_allocator(), other.tag_for, ndm=other.ndm)
    assert not em.calls


class _Inputs(NamedTuple):
    """A model wrapped like a :class:`Case`, for the input helpers."""

    bm: Any


def _seeded_like_the_planner(bm: Any) -> TagAllocator:
    """A plain allocator seeded with ``bm``'s primitives, as a direct
    caller of an emit helper seeds one."""
    from apeGmsh.opensees.apesees import _kind_of

    tags = TagAllocator()
    for prim in bm.primitives:
        tags.allocate_for(prim, _kind_of(prim))
    return tags


def test_emit_refuses_a_transform_plan_for_other_specs() -> None:
    case = _case("arch_with_orientation_fan_out/flat")
    transforms, _ = _arch_transform_inputs(case)
    assert transforms
    sub = case.plan.transforms
    fem = case.bm.fem
    assert sub.fanout_for(transforms, fem) is sub.fanout
    for wrong in ([], [*transforms, transforms[0]], transforms[:-1]):
        with pytest.raises(TagLawError, match="transform plan holds"):
            sub.fanout_for(wrong, fem)
    with pytest.raises(TagLawError, match="another model's tag plan"):
        sub.fanout_for(transforms, ts.synthesised_elements_fem())


def test_transform_plan_derives_its_rows_from_its_fanout() -> None:
    from apeGmsh.opensees._internal.tag_plan import TransformTagPlan

    with pytest.raises(TagLawError, match="fanout, not rows"):
        TransformTagPlan(rows=(("geomTransf", 2),))
    with pytest.raises(TagLawError, match="no fan-out"):
        TransformTagPlan().stream()


# ---------------------------------------------------------------------------
# The contact family (K1-3d S3b)
# ---------------------------------------------------------------------------


_CONTACT_CASES = (
    "synthesised_elements/flat", "synthesised_elements_fem_ids/flat",
    *_CONTACT_MODELS,
)


def _contact_rows(stream: Any) -> list[Row]:
    return [(_verb(k), t) for k, t in stream
            if _verb(k) in ("contact_surface", "contact", "contact_plane")]


def test_short_or_empty_contact_plan_fails_the_oracle() -> None:
    """The contact comparison is not vacuous: a dropped row is caught.

    Every case with contacts (flat, and partitioned over 2 and 4 ranks)
    must reject the contact plan with its last row dropped, and with
    every row dropped, while accepting the real one.
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.contacts.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "contacts", rows) is None, name
        assert _family_mismatch(case, "contacts", rows[:-1]), name
        assert _family_mismatch(case, "contacts", ()), name
    assert checked == len(_CONTACT_CASES)


@pytest.mark.parametrize("name", _CONTACT_CASES)
def test_contact_rows_are_written_in_plan_order(name: str) -> None:
    """The emit writes the planned contact rows in the plan's own order."""
    case = _case(name)
    assert _contact_rows(case.stream) == list(case.plan.contacts.stream())


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_partitioned_contacts_number_rank_by_rank(n_ranks: int) -> None:
    """The partitioned plan numbers contacts rank by rank, as the deck did.

    Rank ``r`` owns ``c{r}`` and ``p{r}``, declared in reverse rank
    order. The partitioned deck writes each rank's block in turn, so
    rank ``r``'s contact takes surfaces ``3r+1, 3r+2`` and contact
    ``2r+1``, and its plane surface ``3r+3`` and contact ``2r+2``. The
    flat deck numbers every contact, then every plane, in declaration
    order. Each owner declares its contact's slave nodes as ghosts.
    """
    case = _case(f"contact_ranks_{n_ranks}/partitioned")
    lines = case.plan.contacts.contacts
    assert lines is not None
    got = {line.record.name: (line.tags, line.owner_rank)
           for line in lines.lines}
    want: dict[str, Any] = {}
    for r in range(n_ranks):
        want[f"c{r}"] = ((3 * r + 1, 3 * r + 2, 2 * r + 1), r)
        want[f"p{r}"] = ((3 * r + 3, 2 * r + 2), r)
    assert got == want
    for line in lines.lines:
        b = 100 * (line.owner_rank + 1)
        ghosts = (b + 5, b + 6) if line.kind == "contact" else ()
        assert tuple(sorted(line.ghost_node_ids)) == ghosts

    if n_ranks == 4:
        flat = _case("contact_ranks_4/flat").plan.contacts.contacts
        assert flat is not None
        assert [(ln.record.name, ln.tags) for ln in flat.lines] == [
            ("c3", (1, 2, 1)), ("c2", (3, 4, 2)), ("c1", (5, 6, 3)),
            ("c0", (7, 8, 4)), ("p3", (9, 5)), ("p2", (10, 6)),
            ("p1", (11, 7)), ("p0", (12, 8)),
        ]
        assert all(ln.owner_rank is None for ln in flat.lines)


def test_contact_kinds_are_frozen_on_every_emit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emit path's allocator refuses a ``contactSurface`` or
    ``contact`` mint: flat, split, staged, and partitioned over 1, 2 and
    4 ranks."""
    from apeGmsh.opensees.apesees import BuiltModel

    seen: list[tuple[str, TagAllocator]] = []

    def spy(path: str) -> Callable[..., Any]:
        orig = getattr(BuiltModel, path)
        sig = inspect.signature(orig)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append((path, sig.bind(self, *args, **kwargs)
                         .arguments["tags"]))
            return orig(self, *args, **kwargs)
        return wrapper

    for path in _PATHS:
        monkeypatch.setattr(BuiltModel, path, spy(path))
    names = {*_first_case_per_mode().values(), *_CONTACT_MODELS}
    reached: set[str] = set()
    for name in sorted(names):
        if name == _SPLIT:
            ts.emit_stream(ts.split_model().build(), TclEmitter, split=True)
        else:
            ts.emit_stream(_MODELS[name]().build(), RecordingEmitter)
    for path, tags in seen:
        reached.add(path)
        assert {"contactSurface", "contact"} <= tags.frozen_kinds, path
        for kind in ("contactSurface", "contact"):
            with pytest.raises(TagLawError, match="planned and frozen"):
                tags.allocate(kind)
            with pytest.raises(TagLawError):
                tags.allocate_block(kind, 1)
    assert reached == set(_PATHS)


@pytest.mark.parametrize("name", _CONTACT_CASES)
def test_a_stray_contact_mint_raises(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: an emit that mints a contact tag raises where it mints.

    The flat and split paths are mutated back to minting their contact
    tags (``_planned_contact_lines`` plans from the emit allocator); the
    partitioned path's writer is mutated to mint one ``contact`` tag per
    line, over 2 and 4 ranks. Each emit must raise, not write a deck.
    """
    import apeGmsh.opensees.apesees as apesees_mod
    from apeGmsh.opensees._internal import build
    from apeGmsh.opensees.apesees import BuiltModel

    def minting_lines(fem: Any, tags: TagAllocator, kind: str) -> Any:
        entries = [(kind, r, None, ()) for r in build.contact_records(fem, kind)]
        return build.plan_contacts(fem, entries, tags).lines

    monkeypatch.setattr(build, "_planned_contact_lines", minting_lines)

    fork: list[TagAllocator] = []
    orig_partitioned = BuiltModel._emit_partitioned

    def spy(self: Any, **kwargs: Any) -> Any:
        fork.append(kwargs["tags"])
        return orig_partitioned(self, **kwargs)

    orig_writer = apesees_mod.write_planned_contact

    def minting_writer(emitter: Any, line: Any, *, ndm: int) -> None:
        fork[-1].allocate("contact")
        orig_writer(emitter, line, ndm=ndm)

    monkeypatch.setattr(BuiltModel, "_emit_partitioned", spy)
    monkeypatch.setattr(apesees_mod, "write_planned_contact", minting_writer)

    with pytest.raises(TagLawError, match="planned and frozen"):
        ts.emit_stream(_MODELS[name]().build(), RecordingEmitter)
    assert bool(fork) == name.endswith("/partitioned")


@pytest.mark.parametrize("name", _CONTACT_CASES)
@pytest.mark.parametrize("cut", ["short", "empty"])
def test_emit_refuses_a_short_or_empty_contact_plan(
    name: str, cut: str,
) -> None:
    """Mutation: the emit refuses a contact plan that dropped interactions.

    The memoised plan's contact lines are cut (the last one dropped, or
    all of them); the next emit, flat or partitioned, must raise rather
    than write a deck that lacks them.
    """
    import dataclasses

    from apeGmsh.opensees._internal.tag_plan import ContactTagPlan

    bm = _MODELS[name]().build()
    mode = emit_mode(bm, split=False, supports_partitions=True)
    plan = bm._tag_plan(mode)
    contacts = plan.contacts.contacts
    assert contacts is not None and contacts.lines
    lines = contacts.lines[:-1] if cut == "short" else ()
    bm._tag_plans[mode] = dataclasses.replace(
        plan, contacts=ContactTagPlan(
            contacts=dataclasses.replace(contacts, lines=lines)))
    with pytest.raises(TagLawError, match="contact plan holds"):
        ts.emit_stream(bm, RecordingEmitter)


def test_emit_contacts_is_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    The plain-allocator path (a direct caller, until K1-3d S6) writes the
    same rows as the planned path. A plain fork, the frozen planner
    allocator and a fork for another origin carry no plan; a fork of a
    real foreign plan is refused as another model's.
    """
    from apeGmsh.opensees._internal.build import (
        emit_contact_planes,
        emit_contacts,
    )

    case = _case("synthesised_elements/flat")
    bm, plan = case.bm, case.plan

    def run(tags: TagAllocator, fem: Any = None) -> list[Row]:
        em = ts.tapped(RecordingEmitter)()
        em.tap = []
        emit_contacts(em, bm.fem if fem is None else fem, tags, ndm=bm.ndm)
        emit_contact_planes(em, bm.fem if fem is None else fem, tags)
        return list(em.tap)

    planned = run(plan.emit_allocator())
    assert planned == run(_seeded_like_the_planner(bm))
    assert planned == list(plan.contacts.stream())
    assert planned       # one contact and one plane

    for other in (plan.allocator.fork(), plan.allocator,
                  plan.allocator.fork({"contact"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            run(other)
    foreign = _case("contact_ranks_4/flat").plan
    with pytest.raises(TagLawError, match="another model's tag plan"):
        run(foreign.emit_allocator())
    with pytest.raises(TagLawError, match="another model's tag plan"):
        run(plan.emit_allocator(), fem=foreign.contacts.contacts.fem)


def test_emit_contacts_refuses_a_plan_for_other_records() -> None:
    from apeGmsh.opensees._internal.build import ContactPlan

    flat = _case("contact_ranks_4/flat").plan.contacts.for_fem(
        _case("contact_ranks_4/flat").bm.fem)
    part = _case("contact_ranks_4/partitioned").plan.contacts.contacts
    assert isinstance(part, ContactPlan)
    records = [ln.record for ln in flat.lines if ln.kind == "contact"]
    assert flat.lines_for("contact", records)
    for wrong in ([], records[:-1], [*records, records[0]], records[::-1]):
        with pytest.raises(TagLawError, match="contact plan holds"):
            flat.lines_for("contact", wrong)
    # A partitioned plan numbers rank by rank: not the flat walk's order.
    part_records = [r for r in part.fem.elements.contacts]
    with pytest.raises(TagLawError, match="contact plan holds"):
        part.lines_for("contact", part_records)


def test_contact_plan_derives_its_rows_from_its_lines() -> None:
    from apeGmsh.opensees._internal.tag_plan import ContactTagPlan

    with pytest.raises(TagLawError, match="contacts, not rows"):
        ContactTagPlan(rows=(("contact", 1),))
    with pytest.raises(TagLawError, match="no contact plan"):
        ContactTagPlan().stream()
    with pytest.raises(TagLawError, match="no contact plan"):
        ContactTagPlan().for_fem(object())


def test_partitioned_routing_runs_once_and_warns_once_per_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The routing runs once per plan; its warning fires once per emit.

    Before the plan, every partitioned emit re-ran
    ``_plan_partitioned_contacts`` and warned once (the F3 partial
    element-ownership warning). Now the plan runs it once, and each emit
    repeats the warning it noted: two emits, one routing, two warnings.
    """
    import warnings

    from apeGmsh.opensees.apesees import BuiltModel

    from tests.opensees.integration.test_contact_partitioned_review_fixes import (
        _cut_master_stub,
        _stub_ops,
    )

    calls: list[int] = []
    orig = BuiltModel._plan_partitioned_contacts

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(BuiltModel, "_plan_partitioned_contacts", spy)
    bm = _stub_ops(_cut_master_stub(kn=1.0e6, second_facet="unowned")).build()
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bm.emit(RecordingEmitter())
        hits = [w for w in caught
                if "absent from every PartitionRecord" in str(w.message)]
        assert len(hits) == 1
        assert hits[0].category is UserWarning
    assert len(calls) == 1


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
    case = _case(name)
    planned = sorted((_verb(k), t) for k, t in case.plan.stream())
    derived = sorted(
        (_verb(k), t) for k, t in case.stream
        if _owner(case, (k, t)) is not None)
    assert planned == derived, (
        f"{name} [{case.plan.mode}]: the plan and the emit disagree")


def test_every_family_has_rows() -> None:
    """Each family's comparison runs on at least one case with rows."""
    for family in FAMILIES:
        kinds = FAMILY_PLANS[family].KINDS
        assert any(
            _rows(_case(n), family, kinds) or (
                FAMILY_PLANS[family].MIGRATED
                and _case(n).plan.family(family).stream())
            for n in CASES
        ), f"no corpus case writes a {family} tag; the oracle is vacuous"


@pytest.mark.parametrize("name", CASES)
def test_every_derived_row_has_an_owner(name: str) -> None:
    """Every tagged verb is classified, and a planned row has a planner."""
    case = _case(name)
    migrated_kinds: set[str] = set()
    for cls in FAMILY_PLANS.values():
        if cls.MIGRATED:
            migrated_kinds |= cls.KINDS
    for row in case.stream:
        verb = _verb(row[0])
        assert verb in VERB_KIND or verb in NON_DERIVED_VERBS, (
            f"{name}: classify the tagged verb {verb!r}")
        if _owner(case, row) == PLANNED:
            assert VERB_KIND[verb] in migrated_kinds, (
                f"{name}: {row} was neither minted at emit nor seeded, and "
                "no migrated family plans its kind"
            )


def test_mint_sites_name_families() -> None:
    assert set(_MINT_SITES.values()) <= set(FAMILIES)


@pytest.mark.parametrize("name", CASES)
def test_plan_seed_is_the_emit_seed(name: str) -> None:
    """The planner seeds, then mints its families, then the emit runs on.

    The planner's seed is read from the mint log: its counters after the
    primitive seeding and the ``element_tags="fem"`` reservation, before
    its first mint (``test_plan_seeds_every_registered_tag`` holds the
    seed to ``bm.tag_for``). Its mints in a kind run on, without a gap,
    from one above the seed, and the frozen counter ends there. Every
    emit-time mint in a kind then runs on, without a gap, from one above
    the plan's counter.
    """
    case = _case(name)
    kinds = set(case.seed) | {
        k for cls in FAMILY_PLANS.values() for k in cls.KINDS}
    for kind in sorted(kinds):
        first = case.seed.get(kind, 0) + 1
        tags = case.planned.get(kind, [])
        assert sorted(tags) == list(range(first, first + len(tags))), (
            f"{name}: planned {kind} tags are not the run from {first}")
        assert case.plan.allocator.last(kind) == first - 1 + len(tags), (
            f"{name}: the plan's {kind!r} counter is "
            f"{case.plan.allocator.last(kind)}, its seed and mints end at "
            f"{first - 1 + len(tags)}"
        )
    for kind, tags in case.minted.items():
        first = case.plan.allocator.last(kind) + 1
        assert sorted(tags) == list(range(first, first + len(tags))), (
            f"{name}: emitted {kind} tags {sorted(tags)} are not the run "
            f"from {first}, one above the plan's counter"
        )


def test_fem_ids_case_reserves_past_the_carrier() -> None:
    """The ``element_tags="fem"`` case: FEM ids from 33, mints from 51."""
    case = _case("synthesised_elements_fem_ids/flat")
    assert case.plan.allocator.last("element") == ts._SYNTH_CARRIER_EID
    assert min(case.minted["element"]) == ts._SYNTH_CARRIER_EID + 1
    elements = sorted(t for k, t in case.stream if k == "element:Truss")
    assert elements[0] == ts._SYNTH_FIRST_EID


def test_corpus_reaches_every_mode() -> None:
    modes = {_case(n).plan.mode for n in CASES}
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
        return [t for k, t in _case(name).stream if k == "region"]

    assert regions("two_rank_regions/flat") == [1, 2, 3]
    assert regions("two_rank_regions/partitioned") == [2, 1, 3, 1]


# ---------------------------------------------------------------------------
# plan_tags
# ---------------------------------------------------------------------------


def test_plan_tags_freezes_its_allocator() -> None:
    plan = _case("two_column_frame/flat").plan
    assert plan.allocator.frozen
    with pytest.raises(TagLawError):
        plan.allocator.allocate("element")


def test_plan_seeds_every_registered_tag() -> None:
    case = _case("initial_stress_frame/flat")
    bm, plan = case.bm, case.plan
    for prim in bm.primitives:
        assert plan.allocator.tag_for(prim) == bm.tag_for[id(prim)]


def test_a_kind_shared_with_a_pending_family_stays_open() -> None:
    """``element`` is frozen only once every family minting it has moved.

    The elements family is planned, but the MP-element and interface
    families still mint ``element`` tags at emit time, so the emit fork
    leaves ``element`` open; the oracle's still-minted check is then what
    catches an element-spec mint the migration missed.
    """
    plan = _case("two_column_frame/flat").plan
    assert plan.migrated == tuple(
        f for f in FAMILIES if FAMILY_PLANS[f].MIGRATED)
    assert "elements" in plan.migrated
    sharing = [f for f, cls in FAMILY_PLANS.items()
               if "element" in cls.KINDS and not cls.MIGRATED]
    assert ("element" in plan.frozen_kinds) == (not sharing)
    if len(plan.migrated) < len(FAMILIES):
        with pytest.raises(NotImplementedError):
            plan.stream()


def test_element_plan_rows_come_from_its_specs() -> None:
    from apeGmsh.opensees._internal.tag_plan import ElementTagPlan

    with pytest.raises(TagLawError, match="specs, not rows"):
        ElementTagPlan(rows=(("element", 1),))
    plan = _case("two_column_frame/flat").plan
    assert plan.elements.stream() == tuple(
        ("element", t) for _s, sub in plan.elements.specs
        for _e, _c, t in sub)
    assert plan.elements.stream()


# ---------------------------------------------------------------------------
# The plan each emit path reads (plan_of)
# ---------------------------------------------------------------------------


_PATHS = ("_emit_flat", "_emit_split", "_emit_partitioned",
          "_emit_stages_flat", "_emit_stages_partitioned")


def _first_case_per_mode() -> dict[TagMode, str]:
    out: dict[TagMode, str] = {}
    for name in CASES:
        out.setdefault(_case(name).plan.mode, name)
    return out


def test_every_emit_path_reads_the_memoised_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plan_of(tags)`` is the memoised plan, on every path and every emit.

    Each emit path (flat, split, staged flat, partitioned, staged
    partitioned) receives a fresh ``emit_allocator()`` fork of the one plan
    ``BuiltModel.emit`` memoises for its mode; a second emit reuses it.
    """
    from apeGmsh.opensees.apesees import BuiltModel

    seen: list[tuple[str, TagAllocator]] = []

    def spy(path: str) -> Callable[..., Any]:
        orig = getattr(BuiltModel, path)
        sig = inspect.signature(orig)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(self, *args, **kwargs)
            seen.append((path, bound.arguments["tags"]))
            return orig(self, *args, **kwargs)
        return wrapper

    for path in _PATHS:
        monkeypatch.setattr(BuiltModel, path, spy(path))

    reached: set[str] = set()
    for mode, name in _first_case_per_mode().items():
        if name == _SPLIT:
            bm = ts.split_model().build()
            cls: type = TclEmitter
        else:
            bm = _MODELS[name]().build()
            cls = RecordingEmitter
        forks: list[TagAllocator] = []
        plans: list[TagPlan] = []
        for _ in range(2):
            seen.clear()
            ts.emit_stream(bm, cls, split=mode.split)
            assert seen, f"{name}: no emit path ran"
            memo = bm._tag_plans[mode]
            for path, tags in seen:
                reached.add(path)
                assert plan_of(tags) is memo, f"{name}: {path}"
            forks.append(seen[0][1])
            plans.append(plan_of(seen[0][1]))
        # The second emit reads the plan the first one made.
        assert plans[0] is plans[1], f"{name}: re-planned on the second emit"
        assert forks[0] is not forks[1], f"{name}: one fork, two emits"
        assert list(bm._tag_plans) == [mode]
    assert reached == set(_PATHS)


def test_plan_of_refuses_an_allocator_without_a_plan() -> None:
    plan = _case("two_column_frame/flat").plan
    for tags in (TagAllocator(), plan.allocator, plan.allocator.fork(),
                 plan.allocator.fork({"element"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            plan_of(tags)
    assert plan_of(plan.emit_allocator()) is plan


def test_emit_refuses_an_element_plan_for_other_specs() -> None:
    from apeGmsh.opensees.apesees import _planned_element_specs

    plan = _case("kitchen_sink_absorbing/staged").plan
    specs = [s for s, _ in plan.elements.specs]
    assert len({id(s) for s in specs}) == len(specs) >= 2
    tags = plan.emit_allocator()
    got = [s for s, _ in _planned_element_specs(tags, specs)]
    assert [id(s) for s in got] == [id(s) for s in specs]
    wrongs = {
        "short": specs[:-1],
        "extra": [*specs, specs[0]],
        "rotated": [specs[-1], *specs[:-1]],
        "swapped": [specs[1], specs[0], *specs[2:]],
        "reversed": specs[::-1],
        "empty": [],
    }
    for label, wrong in wrongs.items():
        with pytest.raises(TagLawError, match="element plan"):
            _planned_element_specs(tags, wrong)
            pytest.fail(f"{label}: accepted")


def _fresh_stream(name: str) -> list[Row]:
    return list(ts.emit_stream(_MODELS[name]().build(), RecordingEmitter))


def test_replace_gives_a_fresh_memo_and_fresh_tags() -> None:
    """``dataclasses.replace`` never reuses its source's plan.

    The source is emitted first, so its memo is populated; the copy, with
    ``element_tags`` or ``fem`` replaced, must emit what a freshly built
    model of the same inputs emits.
    """
    import dataclasses

    bm = _MODELS["synthesised_elements/flat"]().build()
    ts.emit_stream(bm, RecordingEmitter)
    assert bm._tag_plans

    as_fem = dataclasses.replace(bm, element_tags="fem")
    assert as_fem._tag_plans == {}
    assert (ts.emit_stream(as_fem, RecordingEmitter)
            == _fresh_stream("synthesised_elements_fem_ids/flat"))

    new_fem = dataclasses.replace(bm, fem=ts.synthesised_elements_fem())
    assert new_fem._tag_plans == {}
    assert (ts.emit_stream(new_fem, RecordingEmitter)
            == _fresh_stream("synthesised_elements/flat"))
    (mode,) = bm._tag_plans
    assert not bm._tag_plans[mode].planned_for(new_fem)


def test_copy_shares_the_memo_only_while_the_inputs_match() -> None:
    """``copy.copy`` shares the memo dict; the plan checks its inputs.

    A copy emitted as is reuses the plan and writes the same tags. A copy
    whose input changes (here through ``object.__setattr__``, the only way
    to change a frozen model in place) re-plans instead of reading a plan
    made for other inputs.
    """
    import copy

    bm = _MODELS["synthesised_elements/flat"]().build()
    first = ts.emit_stream(bm, RecordingEmitter)
    (mode,) = bm._tag_plans
    plan = bm._tag_plans[mode]

    same = copy.copy(bm)
    assert same._tag_plans is bm._tag_plans
    assert ts.emit_stream(same, RecordingEmitter) == first
    assert bm._tag_plans[mode] is plan

    changed = copy.copy(bm)
    object.__setattr__(changed, "element_tags", "fem")
    assert (ts.emit_stream(changed, RecordingEmitter)
            == _fresh_stream("synthesised_elements_fem_ids/flat"))
    assert not plan.planned_for(changed)


def test_emit_allocator_continues_the_plan() -> None:
    plan = _case("arch_with_orientation_fan_out/flat").plan
    tags = plan.emit_allocator()
    assert not tags.frozen
    assert tags.frozen_kinds == plan.frozen_kinds
    before = {k: plan.allocator.last(k)
              for k in ("element", "geomTransf", "region")}
    for kind in before:
        if kind not in plan.frozen_kinds:
            assert tags.allocate(kind) == before[kind] + 1
    # The plan's own allocator is untouched by the fork's mints.
    assert {k: plan.allocator.last(k) for k in before} == before
    assert before["element"] > 1      # the planned element fan-out


def test_unknown_family_raises() -> None:
    plan = _case("two_column_frame/flat").plan
    with pytest.raises(KeyError, match="unknown tag family"):
        plan.family("nodes")


def test_plan_tags_refuses_a_non_mode() -> None:
    bm = _case("two_column_frame/flat").bm
    with pytest.raises(TypeError, match="TagMode"):
        plan_tags(bm, (False, False, False))  # type: ignore[arg-type]


def test_tag_plan_refuses_an_open_allocator() -> None:
    plan = _case("two_column_frame/flat").plan
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


def test_fork_with_a_frozen_kind_refuses_reset() -> None:
    child = _seeded_allocator().fork({"region"})
    with pytest.raises(TagLawError, match="frozen kinds"):
        child.reset()
    assert child.allocate("element") == 5


def test_fork_with_no_kinds_is_an_open_copy() -> None:
    parent = _seeded_allocator()
    parent.freeze()
    child = parent.fork()
    assert child.allocate("element") == 5
    assert child.allocate_for(object(), "element") == 6


@pytest.mark.parametrize("make", [
    lambda: _seeded_allocator().fork(),
    lambda: _seeded_allocator().fork({"region"}),
    lambda: _case("two_column_frame/flat").plan.emit_allocator(),
], ids=["no_frozen_kinds", "frozen_kind", "emit_allocator"])
def test_every_fork_refuses_reset(make: Any) -> None:
    """A fork never clears: that would re-mint tags its parent handed out.

    Before S2 a fork with no frozen kinds still reset; now that the emit
    mints from ``plan.emit_allocator()``, every fork refuses.
    """
    child = make()
    last = child.last("element")
    with pytest.raises(TagLawError, match="forked allocator"):
        child.reset()
    assert child.last("element") == last


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
