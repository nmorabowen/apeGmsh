"""contacts: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NamedTuple, Sequence, TypeAlias

from ..tag_allocator import TagAllocator
from .errors import BridgeError, TagPlanMiss, _coverage_miss, plan_owner
from .records import _emit_name
from .ownership import runtime_rank_from_partition_record


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from ..tag_plan import TagPlan


# ---------------------------------------------------------------------------
# Contacts (g.constraints.contact / contact_plane, ADR 0092): planned tags
#
# The tag plan numbers every contact interaction once per emit mode
# (ADR 0114 D4, amended): :func:`plan_contacts` is the allocation loop,
# and the emit writes the planned lines (:func:`write_planned_contact`).
# ---------------------------------------------------------------------------

#: The contact record kinds, in the order a flat emit walks them, and the
#: ``fem.elements`` stream each one is read from.
CONTACT_STREAMS: dict[str, str] = {
    "contact": "contacts",
    "contact_plane": "contact_planes",
}


class PlannedContact(NamedTuple):
    """One contact interaction as the tag plan numbers it.

    ``kind`` is ``"contact"`` (face to face: a master and a slave
    ``contactSurface``, then ``contact``) or ``"contact_plane"`` (a rigid
    plane: a slave ``contactSurface``, then ``contactPlane``). ``tags``
    are the surface tags, then the contact tag. ``owner_rank`` and
    ``ghost_node_ids`` are the partitioned routing (ADR 0092 S4): the
    rank whose block writes the interaction, and the interface nodes
    declared there as ghosts first. A flat emit has ``None`` and ``()``.
    """

    kind: str
    record: Any
    tags: tuple[int, ...]
    owner_rank: "int | None"
    ghost_node_ids: tuple[int, ...]


#: One interaction to plan: ``(kind, record, owner_rank, ghost_node_ids)``.
ContactEntry: TypeAlias = "tuple[str, Any, int | None, tuple[int, ...]]"


@dataclass(frozen=True, slots=True)
class ContactPlan:
    """:func:`plan_contacts`'s result: every contact's tags, in emit order.

    ``fem`` is the FEM snapshot whose records were planned. ``lines`` are
    the interactions in the order the emit writes them: every contact,
    then every contact plane, on a flat emit; rank by rank, as routed, on
    a partitioned one. ``notes`` are the routing's warnings, which every
    partitioned emit repeats, so each emit warns once, as before the plan.
    """

    fem: Any
    lines: tuple[PlannedContact, ...]
    notes: tuple[str, ...] = ()

    def lines_for(
        self, kind: str, records: "Sequence[Any]",
    ) -> tuple[PlannedContact, ...]:
        """The planned lines of ``kind``, which must be exactly ``records``.

        ``records`` is the pool a flat emit walks, in order. A plan made
        for other records, or in another order (a partitioned plan numbers
        them rank by rank), raises :class:`TagLawError`.
        """
        planned = tuple(line for line in self.lines if line.kind == kind)
        if [id(line.record) for line in planned] != [id(r) for r in records]:
            raise TagPlanMiss(
                f"{_coverage_miss('contact', [ln.record for ln in planned], records)}"
                f" in this emit's order: the contact plan holds "
                f"{len(planned)} {kind} records, but this emit walks "
                f"{len(records)}, or in another order; the plan was not made "
                "for this emit (ADR 0114 D4, amended)."
            )
        return planned

    def check_covers(self, fem: "FEMData") -> None:
        """Raise :class:`TagLawError` unless the plan holds every record
        of ``fem``'s contact streams exactly once, and nothing else.

        The partitioned emit writes the plan's routed lines and walks no
        record pool of its own, so a plan that dropped a record would
        drop the interaction from the deck without this check.
        """
        for kind in CONTACT_STREAMS:
            planned = [line.record for line in self.lines if line.kind == kind]
            records = contact_records(fem, kind)
            if sorted(map(id, planned)) != sorted(map(id, records)):
                raise TagPlanMiss(
                    f"{_coverage_miss('contact', planned, records)}: the contact "
                    f"plan holds {len(planned)} {kind} records, but the FEM "
                    f"carries {len(records)}; the plan was not made for this "
                    "emit (ADR 0114 D4, amended)."
                )

    def ghost_node_ids(self) -> set[int]:
        """Every node a routed interaction declares as a ghost."""
        return {nid for line in self.lines for nid in line.ghost_node_ids}


def contact_records(fem: "FEMData", kind: str) -> list[Any]:
    """The records of contact ``kind`` on ``fem``; empty when it has none."""
    if kind not in CONTACT_STREAMS:
        raise BridgeError(
            f"unknown contact kind {kind!r}; the kinds are "
            f"{tuple(CONTACT_STREAMS)}."
        )
    elements = getattr(fem, "elements", None)
    if elements is None:
        return []
    return list(getattr(elements, CONTACT_STREAMS[kind], None) or ())


def flat_contact_entries(fem: "FEMData") -> list[ContactEntry]:
    """Every contact, then every contact plane: the flat emit's order."""
    return [
        (kind, rec, None, ())
        for kind in CONTACT_STREAMS
        for rec in contact_records(fem, kind)
    ]


def partitioned_contact_entries(
    routing: "Mapping[int, Sequence[tuple[str, Any, tuple[int, ...]]]]",
    partitions: "Sequence[Any]",
) -> list[ContactEntry]:
    """The routed interactions, rank by rank: the partitioned emit's order.

    ``routing`` is ``BuiltModel._plan_partitioned_contacts``'s
    ``{owner_rank: [(kind, record, ghost_node_ids), ...]}``. The ranks run
    in the order the per-rank loop opens them (ADR 0027: ``enumerate``
    over ``partitions``). A record routed to a rank that no partition
    runs raises, rather than drop out of the deck.
    """
    out: list[ContactEntry] = []
    ranks: set[int] = set()
    for idx, part in enumerate(partitions):
        rank = runtime_rank_from_partition_record(part, idx)
        ranks.add(rank)
        out.extend(
            (kind, rec, rank, tuple(int(n) for n in ghosts))
            for kind, rec, ghosts in routing.get(rank, ())
        )
    stray = sorted(set(routing) - ranks)
    if stray:
        raise BridgeError(
            f"apeSees: contact interactions are routed to rank(s) {stray}, "
            "which no partition runs (ADR 0092 INV-1); they would drop out "
            "of the partitioned deck."
        )
    return out


def plan_contacts(
    fem: "FEMData", entries: "Iterable[ContactEntry]", tags: TagAllocator,
    *, notes: "Iterable[str]" = (),
) -> ContactPlan:
    """Plan the contact tags, in the order the emit writes them.

    The allocation loop of :func:`emit_contacts` and
    :func:`emit_contact_planes`, moved out of the emit (ADR 0114 D4,
    amended): the build's tag plan runs it once per emit mode with the
    planner allocator, and the helpers write what it planned. Each
    ``contact`` takes two
    ``contactSurface`` tags (master, slave) and one ``contact`` tag; each
    ``contact_plane`` takes one ``contactSurface`` tag (slave) and one
    ``contact`` tag.
    """
    lines: list[PlannedContact] = []
    for kind, rec, owner_rank, ghosts in entries:
        if kind == "contact":
            surfaces: tuple[int, ...] = (
                tags.allocate("contactSurface"),
                tags.allocate("contactSurface"),
            )
        elif kind == "contact_plane":
            surfaces = (tags.allocate("contactSurface"),)
        else:
            raise BridgeError(
                f"plan_contacts: unknown contact kind {kind!r}; the kinds "
                f"are {tuple(CONTACT_STREAMS)}."
            )
        lines.append(PlannedContact(
            kind, rec, (*surfaces, tags.allocate("contact")),
            owner_rank, tuple(ghosts),
        ))
    return ContactPlan(fem=fem, lines=tuple(lines), notes=tuple(notes))


def route_contacts(
    canonical: ContactPlan, entries: "Iterable[ContactEntry]",
    *, notes: "Iterable[str]" = (),
) -> ContactPlan:
    """The routed interactions ``entries``, in their order, with the tags
    the canonical (flat-order) plan gave them.

    ``canonical`` is :func:`plan_contacts`'s result over
    :func:`flat_contact_entries`. A partitioned emit writes the same
    interactions rank by rank (:func:`partitioned_contact_entries`), so
    each keeps its flat tags and gains its owner rank and ghosts. An
    interaction the canonical plan does not hold, or holds as another
    kind, raises :class:`TagPlanMiss`.
    """
    by_record = {id(line.record): line for line in canonical.lines}
    lines: list[PlannedContact] = []
    for kind, rec, owner_rank, ghosts in entries:
        line = by_record.get(id(rec))
        if line is None or line.record is not rec or line.kind != kind:
            raise TagPlanMiss(
                f"the partitioned emit writes the {kind} {plan_owner(rec)}, "
                "which the canonical (flat-order) contact plan does not "
                "number (ADR 0114 D4, amended)."
            )
        lines.append(PlannedContact(
            kind, rec, line.tags, owner_rank, tuple(ghosts)))
    return ContactPlan(
        fem=canonical.fem, lines=tuple(lines), notes=tuple(notes))


def _planned_contact_lines(
    fem: "FEMData", tag_plan: "TagPlan", kind: str,
) -> tuple[PlannedContact, ...]:
    """The planned lines of ``kind`` for a flat emit of ``fem``.

    Read from ``tag_plan`` (ADR 0114 D4, amended); a plan made for
    another model or for other records raises :class:`TagPlanMiss`.
    """
    records = contact_records(fem, kind)
    return tag_plan.contacts.for_fem(fem).lines_for(kind, records)


def write_planned_contact(
    emitter: "Emitter", line: PlannedContact, *, ndm: int,
) -> None:
    """Write one planned interaction under its planned tags.

    ``ndm`` gates a face-to-face ``contact`` (:func:`_write_contact`); a
    ``contact_plane`` carries no dimension of its own
    (:func:`_write_contact_plane`). An unknown kind raises.
    """
    if line.kind == "contact":
        _write_contact(emitter, line, ndm=ndm)
    elif line.kind == "contact_plane":
        _write_contact_plane(emitter, line)
    else:
        raise BridgeError(
            f"write_planned_contact: unknown contact kind {line.kind!r}; "
            f"the kinds are {tuple(CONTACT_STREAMS)}."
        )


def _write_contact(
    emitter: "Emitter", line: PlannedContact, *, ndm: int,
) -> None:
    """Write a face-to-face contact: master and slave surface, ``contact``.

    ``ndm`` cross-checks the record's own dimension (``master_nps == 2`` ⇔
    2D) against what the user declared in ``ops.model(ndm=, ndf=)``. The
    two are genuinely independent sources of truth — the resolve-time gate
    reads ``gmsh.model.getDimension()``, this one reads the declaration —
    so the check catches a 2D mesh declared into a 3D model, and the
    ``compose`` / ``from_h5`` hole where an archived 2D record lands in a
    3D assembly, which nothing else covers.

    It also sets the ``-outward`` ARITY (two components in 2D, three in
    3D) — the fork reads the arity from the referenced nodes' coordinate
    size and rejects the other form, so this is a correctness parameter,
    not a formatting one.
    """
    from ...element.contact import contact_args, contact_surface_args

    rec = line.record
    m_nps = int(rec.master_nps)
    if (m_nps == 2) != (int(ndm) == 2):
        who = repr(rec.name) if rec.name else "(unnamed)"
        raise BridgeError(
            f"apeSees: contact interaction {who} carries "
            f"master_nps={m_nps}, i.e. a "
            f"{'2D line-segment' if m_nps == 2 else '3D faceted'} "
            f"surface, but the model was declared ndm={int(ndm)}. The "
            f"fork derives the contact lane from the referenced nodes' "
            f"coordinate size and aborts on a mismatch; a 2D surface in "
            f"a 3D model (or the reverse) cannot be emitted. Rebuild the "
            f"contact against a {int(ndm)}D mesh, or declare "
            f"ops.model(ndm={2 if m_nps == 2 else 3}, ...)."
        )
    _emit_name(emitter, rec.name)
    m_tag, s_tag, c_tag = line.tags

    # Master: always a faceted surface (flat connectivity + stride).
    m_flat = [int(n) for n in rec.master_faces.reshape(-1)]
    emitter.contact_surface(
        m_tag, *contact_surface_args("master", m_flat, rec.master_nps))

    # Slave: NTS node set vs mortar faceted.
    if rec.formulation == "nts":
        emitter.contact_surface(
            s_tag,
            *contact_surface_args("slave", [int(n) for n in rec.slave_nodes]))
    else:
        s_flat = [int(n) for n in rec.slave_faces.reshape(-1)]
        emitter.contact_surface(
            s_tag,
            *contact_surface_args("slave-segments", s_flat, rec.slave_nps))

    emitter.contact(c_tag, *contact_args(
        m_tag, s_tag, rec.formulation,
        kn=rec.kn, kt=rec.kt, mu=rec.mu,
        eps_n=rec.eps_n, eps_t=rec.eps_t,
        cohesion=rec.cohesion, tau_max=rec.tau_max,
        aug_tol=rec.aug_tol, max_aug=rec.max_aug, ngp=rec.ngp,
        tie=rec.tie, thickness=rec.thickness,
        soft=rec.soft, visc=rec.visc,
        consistent_tan=rec.consistent_tan, geom_tan=rec.geom_tan,
        cell=rec.cell,
        edge_edge=rec.edge_edge, edge_kn=rec.edge_kn,
        edge_band=rec.edge_band, edge_mu=rec.edge_mu, edge_kt=rec.edge_kt,
        edge_cohesion=rec.edge_cohesion, edge_tau_max=rec.edge_tau_max,
        edge_consistent_tan=rec.edge_consistent_tan,
        edge_soft=rec.edge_soft, edge_alm=rec.edge_alm,
        edge_aug_tol=rec.edge_aug_tol,
        outward=rec.outward, ndm=int(ndm),
    ))


def _write_contact_plane(emitter: "Emitter", line: PlannedContact) -> None:
    """Write a rigid-plane contact: slave surface, then ``contactPlane``."""
    from ...element.contact import contact_plane_args, contact_surface_args

    rec = line.record
    _emit_name(emitter, rec.name)
    s_tag, c_tag = line.tags
    emitter.contact_surface(
        s_tag,
        *contact_surface_args("slave", [int(n) for n in rec.slave_nodes]))
    emitter.contact_plane(c_tag, *contact_plane_args(
        s_tag, rec.normal, rec.point, rec.kn,
        visc=rec.visc, soft=rec.soft,
    ))


def emit_contacts(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan", *, ndm: int,
) -> None:
    """Emit the fork `contactSurface` + `contact` pair per contact interaction
    (`g.constraints.contact`).

    Consumes ``fem.elements.contacts`` —
    :class:`~apeGmsh._kernel.records._constraints.ContactRecord` rows produced
    by :class:`ConstraintsComposite` at FEM-build time. Each record emits two
    `contactSurface` defs (master faceted + slave node-set/faceted) and the
    `contact` verb (:func:`_write_contact`, which also cross-checks the
    record's dimension against ``ndm``). The `LadrunoContact` handler is
    emitted separately by the bridge's constraint-handler auto-emit.

    The tags come from :func:`plan_contacts` (ADR 0114 D4, amended);
    :func:`_planned_contact_lines` says how this reads them. The
    partitioned path (ADR 0092 S4) does not come through here: it writes
    the plan's routed lines inside each owner rank's block. No-op when the
    FEM carries no contacts.
    """
    for line in _planned_contact_lines(fem, tag_plan, "contact"):
        _write_contact(emitter, line, ndm=ndm)


def emit_contact_planes(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan",
) -> None:
    """Emit one fork ``contactSurface -slave`` + ``contactPlane`` per
    rigid-plane contact (`g.constraints.contact_plane`).

    Consumes ``fem.elements.contact_planes`` — :class:`ContactPlaneRecord` rows.
    Each record emits one ``contactSurface -slave <nodes>`` (the slave node set)
    and one ``contactPlane <tag> <slaveSurfTag> nx ny nz px py pz kn [-visc]
    [-soft]``. The ``LadrunoContact`` handler is auto-emitted by the bridge when
    contacts OR contact planes are present. The tags come from
    :func:`plan_contacts`, as in :func:`emit_contacts`, and the partitioned
    path (ADR 0092 S4) writes the plan's routed lines instead. No-op when the
    FEM carries no contact planes.
    """
    for line in _planned_contact_lines(fem, tag_plan, "contact_plane"):
        _write_contact_plane(emitter, line)
