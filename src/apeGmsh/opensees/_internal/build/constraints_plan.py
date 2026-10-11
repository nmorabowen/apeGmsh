"""constraints_plan: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    cast,
)

from ..tag_allocator import TagAllocator, TagLawError
from .errors import BridgeError, TagPlanMiss, plan_owner


if TYPE_CHECKING:
    from collections import Counter
    from apeGmsh.mesh.FEMData import FEMData
    from ..tag_plan import TagPlan
    from .constraints_ranked import (
        _RankConstraintPlan,
    )


# ---------------------------------------------------------------------------
# MP elements (ADR 0114 D4, amended): planned tags
#
# The tag plan numbers every element the MP constraints synthesise once per
# emit mode: rigid bodies, kinematic couplings, interpolation ties, reinforce
# and embed ties, and auto-emitted rebar cells. :func:`plan_mp_elements` is
# the allocation loop; the writers below read each element's planned tag
# (:func:`_mp_element_tagger`) instead of minting one.
# ---------------------------------------------------------------------------

#: The MP-element sites, each a writer below.
MP_ELEMENT_SITES: tuple[str, ...] = (
    "rigid_body", "kinematic_coupling", "interpolation",
    "reinforce_tie", "embed_tie", "rebar_cell",
)


#: The tap verb each site but ``interpolation`` writes; an
#: interpolation's verb depends on its route (:data:`_INTERPOLATION_VERBS`).
_SITE_VERBS: dict[str, str] = {
    "rigid_body": "element",
    "kinematic_coupling": "element",
    "reinforce_tie": "embedded_rebar",
    "embed_tie": "embedded_node",
    "rebar_cell": "element",
}


#: An interpolation's route -> the tap verb it writes. The ``equation``
#: route writes ``equationConstraint`` rows and no element (ADR 0068).
_INTERPOLATION_VERBS: dict[str, str] = {
    "penalty_al": "element",
    "distributing": "element",
    "embedded": "embeddedNode",
}


def _interpolation_route(rec: Any) -> str:
    """How :func:`_emit_one_interpolation` writes ``rec``.

    ``"equation"`` (``equationConstraint`` rows, no element),
    ``"penalty_al"`` (``LadrunoEmbeddedNode``), ``"distributing"``
    (``LadrunoDistributingCoupling``) or ``"embedded"``
    (``ASDEmbeddedNodeElement``).
    """
    from apeGmsh._kernel.records._kinds import ConstraintKind

    enforce = getattr(rec, "enforce", "penalty")
    if enforce == "equation":
        return "equation"
    if enforce == "penalty_al":
        return "penalty_al"
    if rec.kind == ConstraintKind.DISTRIBUTING:
        return "distributing"
    return "embedded"


class MPElementEntry(NamedTuple):
    """One MP element to plan: its site, record, key and tap verb.

    ``key`` is ``id(record)``, except for a rebar cell, whose key is
    ``(pg, i_node, j_node)``: the partitioned emit hands each rank copies
    of the bar records that hold only its cells.
    """

    site: str
    record: Any
    key: Hashable
    verb: str


class PlannedMPElement(NamedTuple):
    """One MP element as the tag plan numbers it."""

    site: str
    record: Any
    key: Hashable
    verb: str
    tag: int


def mp_element_entry(
    site: str, record: Any, cell: "tuple[int, int] | None" = None,
) -> MPElementEntry:
    """The plan entry of one MP element; an unknown site raises."""
    if site not in MP_ELEMENT_SITES:
        raise BridgeError(
            f"unknown MP-element site {site!r}; the sites are "
            f"{MP_ELEMENT_SITES}."
        )
    if (site == "rebar_cell") != (cell is not None):
        raise BridgeError(
            "mp_element_entry: a rebar_cell entry, and only a rebar_cell "
            "entry, names its cell."
        )
    if cell is not None:
        key: Hashable = (str(record.pg), int(cell[0]), int(cell[1]))
        return MPElementEntry(site, record, key, _SITE_VERBS[site])
    if site == "interpolation":
        route = _interpolation_route(record)
        if route not in _INTERPOLATION_VERBS:
            raise BridgeError(
                f"an {route!r} interpolation writes no element, so it "
                "takes no element tag (ADR 0068)."
            )
        return MPElementEntry(
            site, record, id(record), _INTERPOLATION_VERBS[route])
    return MPElementEntry(site, record, id(record), _SITE_VERBS[site])


def plan_mp_elements(
    entries: "Iterable[MPElementEntry]", tags: TagAllocator,
) -> tuple[PlannedMPElement, ...]:
    """Plan the MP elements' tags, one ``element`` tag each, in order.

    The allocation loop of every MP-element writer, moved out of the
    emit (ADR 0114 D4, amended): the build's tag plan runs it once per
    emit mode with the planner allocator, in the order that mode's emit
    writes the elements; :func:`replay_reinforce_ties` alone runs it with
    the deck replay's allocator.
    """
    out: list[PlannedMPElement] = []
    for entry in entries:
        out.append(PlannedMPElement(*entry, tags.allocate("element")))
    return tuple(out)


def number_mp_elements(
    entries: "Iterable[MPElementEntry]",
    canonical: "Mapping[tuple[str, Hashable], PlannedMPElement]",
) -> tuple[PlannedMPElement, ...]:
    """The MP elements ``entries``, in their order, with the tags the
    canonical (flat-order) walk minted for them.

    ``canonical`` maps each ``(site, key)`` of that walk
    (:func:`plan_mp_elements`) to its line. A partitioned emit writes the
    same elements rank by rank, so it reads their tags here and mints
    none. An element the canonical walk did not number raises
    :class:`TagPlanMiss`.
    """
    out: list[PlannedMPElement] = []
    for entry in entries:
        line = canonical.get((entry.site, entry.key))
        if line is None:
            raise TagPlanMiss(
                f"the partitioned emit writes the {entry.site} element "
                f"{entry.key!r}, which the canonical (flat-order) walk did "
                "not number (ADR 0114 D4, amended)."
            )
        out.append(PlannedMPElement(*entry, line.tag))
    return tuple(out)


def number_constraint_pass(
    node_constraints: "Iterable[object] | None",
    interpolations: "Iterable[Any]",
    canonical: "Mapping[tuple[str, Hashable], PlannedMPElement]",
    *,
    allowed_ids: "frozenset[int] | None" = None,
) -> list[PlannedMPElement]:
    """:func:`constraint_pass_entries` over one rank's records, numbered by
    the canonical (flat-order) walk's lines.

    The same elements, in the same order, as
    ``number_mp_elements(constraint_pass_entries(...), tags)``, but an
    interpolation reuses its canonical line, so its route is read only
    when that walk did not number it: an ``equation`` tie writes no
    element and is skipped; any other raises :class:`TagPlanMiss`.
    """
    from apeGmsh._kernel.records._constraints import InterpolationRecord

    out: list[PlannedMPElement] = []
    if node_constraints is not None:
        for site, records in (
            ("rigid_body", _rigid_body_element_records(
                node_constraints, allowed_ids)),
            ("kinematic_coupling", _kinematic_coupling_records(
                node_constraints, allowed_ids)),
        ):
            for rec in records:
                line = canonical.get((site, id(rec)))
                if line is None or line.record is not rec:
                    raise TagPlanMiss(
                        f"the partitioned emit writes the {site} element of "
                        f"{plan_owner(rec)}, which the canonical (flat-order) "
                        "walk did not number (ADR 0114 D4, amended)."
                    )
                out.append(line)
    for rec in interpolations:
        if not isinstance(rec, InterpolationRecord):
            continue
        line = canonical.get(("interpolation", id(rec)))
        if line is not None and line.record is rec:
            out.append(line)
            continue
        if _interpolation_route(rec) == "equation":
            continue
        raise TagPlanMiss(
            f"the partitioned emit writes the interpolation element of "
            f"{plan_owner(rec)}, which the canonical (flat-order) walk did "
            "not number (ADR 0114 D4, amended)."
        )
    return out


def mp_constraint_pools(
    fem: "FEMData", claimed_ids: "frozenset[int]",
) -> "tuple[Any, Any]":
    """``(node_constraints, surface_constraints)`` of ``fem``, less the
    stage-claimed records; ``None`` where the snapshot has no such
    container (broker constraints are purely additive)."""
    nodes = getattr(fem, "nodes", None)
    elements = getattr(fem, "elements", None)
    node_constraints = (
        getattr(nodes, "constraints", None) if nodes is not None else None
    )
    surface_constraints = (
        getattr(elements, "constraints", None)
        if elements is not None
        else None
    )
    # Filter out records claimed by stage builders so they don't
    # double-emit in the global block and again inside their owning
    # stage's block.  Wrap once at the orchestrator; helpers consume
    # the adapter unchanged.
    if claimed_ids and node_constraints is not None:
        node_constraints = _ExcludeClaimedConstraints(
            node_constraints, claimed_ids,
        )
    if claimed_ids and surface_constraints is not None:
        surface_constraints = _ExcludeClaimedConstraints(
            surface_constraints, claimed_ids,
        )
    return node_constraints, surface_constraints


def _rigid_body_element_records(
    node_constraints: Iterable[object],
    allowed_ids: "frozenset[int] | None" = None,
) -> Iterator[Any]:
    """The ``rigid_body`` rows emitted as ``LadrunoRigidBody`` elements
    (``as_element=True``), in container order."""
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if allowed_ids is not None and id(rec) not in allowed_ids:
            continue
        if (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.RIGID_BODY
            and getattr(rec, "as_element", False)
        ):
            yield rec


def _kinematic_coupling_records(
    node_constraints: Iterable[object],
    allowed_ids: "frozenset[int] | None" = None,
) -> Iterator[Any]:
    """The ``kinematic_coupling`` rows, in container order."""
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if allowed_ids is not None and id(rec) not in allowed_ids:
            continue
        if (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.KINEMATIC_COUPLING
        ):
            yield rec


def interpolation_records(surface_constraints: object) -> list[Any]:
    """Every :class:`InterpolationRecord` ``surface_constraints`` yields
    (``tied_contact`` / ``mortar`` slaves expanded), in order."""
    from apeGmsh._kernel.records._constraints import InterpolationRecord

    interps = getattr(surface_constraints, "interpolations", None)
    if surface_constraints is None or interps is None:
        return []
    return [r for r in interps() if isinstance(r, InterpolationRecord)]


def constraint_pass_entries(
    node_constraints: "Iterable[object] | None",
    interpolations: "Iterable[Any]",
    *,
    allowed_ids: "frozenset[int] | None" = None,
) -> list[MPElementEntry]:
    """The elements one MP-constraint pass writes, in its write order.

    Rigid bodies, then kinematic couplings (both over
    ``node_constraints``, filtered to ``allowed_ids`` when given), then
    every interpolation in ``interpolations`` that writes an element: the
    order of :func:`emit_mp_constraints`, of its partitioned and staged
    variants, which take their records from the same walks.
    """
    from apeGmsh._kernel.records._constraints import InterpolationRecord

    out: list[MPElementEntry] = []
    if node_constraints is not None:
        out.extend(
            mp_element_entry("rigid_body", rec)
            for rec in _rigid_body_element_records(
                node_constraints, allowed_ids))
        out.extend(
            mp_element_entry("kinematic_coupling", rec)
            for rec in _kinematic_coupling_records(
                node_constraints, allowed_ids))
    for rec in interpolations:
        if not isinstance(rec, InterpolationRecord):
            continue
        # The ``mp_element_entry`` of an element-writing interpolation,
        # its route read once (a gate cell walks 10^4 of them).
        route = _interpolation_route(rec)
        if route != "equation":
            out.append(MPElementEntry(
                "interpolation", rec, id(rec), _INTERPOLATION_VERBS[route]))
    return out


def _fem_side_list(fem: "FEMData", name: str) -> list[Any]:
    """``fem.elements.<name>`` as a list; empty when it has none."""
    elements = getattr(fem, "elements", None)
    if elements is None:
        return []
    return list(getattr(elements, name, None) or ())


def reinforce_tie_records(fem: "FEMData") -> list[Any]:
    """``fem.elements.reinforce_ties`` (``g.reinforce``)."""
    return _fem_side_list(fem, "reinforce_ties")


def embed_tie_records(fem: "FEMData") -> list[Any]:
    """``fem.elements.embed_ties`` (``g.embed``)."""
    return _fem_side_list(fem, "embed_ties")


def rebar_element_records(fem: "FEMData") -> list[Any]:
    """``fem.elements.rebar_elements`` (``g.rebar.place(emit_elements=True)``)."""
    return _fem_side_list(fem, "rebar_elements")


def interface_records(fem: "FEMData") -> list[Any]:
    """``fem.elements.interfaces`` (``g.constraints.interface``)."""
    return _fem_side_list(fem, "interfaces")


def rebar_cell_entries(records: "Iterable[Any]") -> list[MPElementEntry]:
    """One entry per line cell of each bar record, in write order."""
    return [
        mp_element_entry("rebar_cell", rec, (int(i), int(j)))
        for rec in records
        for i, j in rec.connectivity
    ]


def mp_element_keys(
    fem: "FEMData", *, partitioned: bool,
) -> "Counter[tuple[str, Hashable]]":
    """Every ``(site, key)`` one emit of ``fem`` writes, as a multiset.

    Read straight off the FEM snapshot, past any stage claim, rank or
    pass routing, so a plan whose routing drops or doubles an element
    disagrees with it. A partitioned emit writes no embed tie (the path
    refuses ``g.embed``) and skips an interpolation with no master node
    (no rank can host it).
    """
    from collections import Counter

    node_constraints, surface_constraints = mp_constraint_pools(
        fem, frozenset())
    interps = interpolation_records(surface_constraints)
    if partitioned:
        interps = [r for r in interps if len(r.master_nodes)]
    entries = constraint_pass_entries(node_constraints, interps)
    entries += [
        mp_element_entry("reinforce_tie", r)
        for r in reinforce_tie_records(fem)]
    if not partitioned:
        entries += [
            mp_element_entry("embed_tie", r) for r in embed_tie_records(fem)]
    entries += rebar_cell_entries(rebar_element_records(fem))
    return Counter((e.site, e.key) for e in entries)


@dataclass(frozen=True, slots=True)
class MPElementPlan:
    """The MP elements' tags, in the order the emit writes them.

    ``fem`` is the FEM snapshot that was planned and ``lines`` the
    elements in emit order. ``claimed_ids`` are the stage-claimed records
    the global passes skip. A partitioned plan also carries the global
    MP-constraint pass's routing, resolved once (ADR 0027):
    ``phantom_coords`` and each rank's :class:`_RankConstraintPlan`,
    which :func:`emit_mp_constraints_partitioned` reads instead of
    resolving it again.

    A plan holds each element once and covers every element the emit of
    ``fem`` writes (:meth:`check_covers`); either failure raises
    :class:`TagLawError` when the plan is made.
    """

    fem: Any
    lines: tuple[PlannedMPElement, ...]
    partitioned: bool = False
    claimed_ids: frozenset[int] = frozenset()
    phantom_coords: "Mapping[int, tuple[float, float, float]]" = field(
        default_factory=dict)
    rank_plans: "Mapping[int, _RankConstraintPlan]" = field(
        default_factory=dict)
    _index: "dict[tuple[str, Hashable], int]" = field(
        init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        index: "dict[tuple[str, Hashable], int]" = {}
        for line in self.lines:
            k = (line.site, line.key)
            if k in index:
                raise TagLawError(
                    f"the MP-element plan holds the {line.site} element "
                    f"{line.key!r} twice: an emit writes each once (ADR "
                    "0114 D4, amended)."
                )
            index[k] = line.tag
        object.__setattr__(self, "_index", index)
        self.check_covers()

    def check_covers(self) -> None:
        """Raise :class:`TagLawError` unless the plan holds exactly the
        MP elements the emit of its FEM writes (:func:`mp_element_keys`).

        The writers look each element's tag up by record, so a plan that
        dropped a record's element would only fail when that record is
        written; this check refuses it, and a plan that holds another
        model's record in its place, when the plan is made.
        """
        from collections import Counter

        planned = Counter((line.site, line.key) for line in self.lines)
        expected = mp_element_keys(self.fem, partitioned=self.partitioned)
        if planned != expected:
            lost = sorted((expected - planned).elements(), key=repr)
            extra = sorted((planned - expected).elements(), key=repr)
            owner = (f"holds no {lost[0][0]} element for {lost[0][1]!r}"
                     if lost else
                     f"holds a {extra[0][0]} element for {extra[0][1]!r} "
                     "that the emit of its FEM does not write")
            raise TagPlanMiss(
                f"the MP-element plan {owner}: it holds "
                f"{sum(planned.values())} elements, the emit of its FEM "
                f"writes {sum(expected.values())}; the plan does not cover "
                "the model (ADR 0114 D4, amended)."
            )

    def for_fem(self, fem: "FEMData") -> "MPElementPlan":
        """The plan, which must have been made over ``fem``; a plan made
        over another FEM snapshot raises :class:`TagLawError`."""
        if fem is not self.fem:
            raise TagPlanMiss(_FOREIGN_FEM.format(family="MP-element"))
        return self

    def check_claims(self, claimed_ids: "frozenset[int]") -> None:
        """Raise :class:`TagLawError` unless the plan's global passes
        skipped exactly the stage claims ``claimed_ids``."""
        if frozenset(claimed_ids) != self.claimed_ids:
            raise TagPlanMiss(
                "the MP-element plan was made for other stage claims than "
                "this emit passes (ADR 0114 D4, amended)."
            )

    def tag_of(self, entry: MPElementEntry) -> int:
        """The planned tag of ``entry``; a record the plan does not hold
        raises :class:`TagLawError`."""
        tag = self._index.get((entry.site, entry.key))
        if tag is None:
            raise TagPlanMiss(
                f"the MP-element plan holds no {entry.site} element for "
                f"{entry.key!r}: the plan was not made for this emit (ADR "
                "0114 D4, amended)."
            )
        return tag

    def rank_constraints(
        self, rank: int,
    ) -> "tuple[Mapping[int, tuple[float, float, float]], _RankConstraintPlan]":
        """The global pass's phantom coordinates and rank ``rank``'s
        routing; a rank the plan did not route raises."""
        if not self.partitioned:
            raise TagPlanMiss(
                "a flat MP-element plan carries no rank routing; the "
                "partitioned emit reads a partitioned plan (ADR 0114 D4, "
                "amended)."
            )
        if rank not in self.rank_plans:
            raise TagPlanMiss(
                f"the MP-element plan routes no rank {rank}: the plan was "
                "not made for this emit (ADR 0114 D4, amended)."
            )
        return self.phantom_coords, self.rank_plans[rank]


#: A writer handed another model's plan says so first.
_FOREIGN_FEM = (
    "the {family} plan was made over another FEM snapshot than the one "
    "this emit walks: the tag plan is another model's. Emit each model "
    "through its own BuiltModel.emit (ADR 0114 D4, amended)."
)


def _mp_plan(tag_plan: "TagPlan", fem: "FEMData") -> "MPElementPlan":
    """The MP-element plan ``tag_plan`` holds, made over ``fem``.

    A plan made over another FEM raises :class:`TagPlanMiss`.
    """
    return tag_plan.mp_elements.planned().for_fem(fem)


def _mp_element_tagger(
    tag_plan: "TagPlan",
) -> "Callable[[MPElementEntry], int]":
    """How a writer gets one MP element's tag: the tag ``tag_plan`` holds
    for it (:meth:`MPElementPlan.tag_of`)."""
    return tag_plan.mp_elements.planned().tag_of


# ---------------------------------------------------------------------------
# Stage-bound MP constraints — flat-list adapter + per-stage emit helpers.
# ---------------------------------------------------------------------------


class _ExcludeClaimedConstraints:
    """Wrap a constraint container, hiding records whose ``id()`` is
    in ``claimed_ids`` (records claimed by stage builders).

    The MP-constraint emit pass calls this once at the orchestrator
    level (``emit_mp_constraints`` / ``emit_mp_constraints_partitioned``)
    to filter ``fem.{nodes,elements}.constraints`` before handing
    the result to the per-kind helpers.  Stage-claimed records emit
    inside their owning stage's block instead (see
    :func:`emit_stage_mp_constraints`), so excluding them from the
    global pass is what produces the per-stage routing.

    Stateless aside from the wrapped container + frozen claim set;
    reused on every emit pass.  Delegation surface mirrors
    :class:`_StageConstraintAdapter` for symmetry with the per-kind
    helpers.
    """
    __slots__ = ("_inner", "_claimed")

    def __init__(
        self, inner: object, claimed: "frozenset[int]",
    ) -> None:
        self._inner = inner
        self._claimed = claimed

    def __iter__(self) -> Iterator[object]:
        for rec in cast(Iterable[object], self._inner):
            if id(rec) not in self._claimed:
                yield rec

    def node_to_surfaces(self) -> Iterator[object]:
        inner_method = getattr(self._inner, "node_to_surfaces", None)
        if inner_method is None:
            return
        for rec in inner_method():
            if id(rec) not in self._claimed:
                yield rec

    def interpolations(self) -> Iterator[object]:
        inner_method = getattr(self._inner, "interpolations", None)
        if inner_method is None:
            return
        for rec in inner_method():
            if id(rec) not in self._claimed:
                yield rec
