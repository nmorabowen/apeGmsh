"""interfaces: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple, Sequence, cast

import numpy as np

from ..tag_resolution import (
    MISSING_FEM_ELEMENT_ID,
    set_current_fem_element_id,
    set_element_nodes,
)
from ..tag_allocator import TagAllocator, TagLawError
from .errors import BridgeError, TagPlanMiss, _coverage_miss, plan_owner
from .records import _emit_name
from .ndf import node_coords_as_floats
from .ownership import (
    runtime_rank_from_partition_record,
)
from .constraints_plan import _FOREIGN_FEM, interface_records


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh._kernel.records._constraints import InterfaceRecord
    from apeGmsh._kernel.records._partitions import PartitionRecord
    from ..tag_plan import TagPlan
    from ..types import UniaxialMaterial


def _interface_normal_material(a_trib: float, law: object) -> "UniaxialMaterial":
    """Translate a :class:`NormalLaw` to its typed uniaxial primitive
    (ADR 0093 D1 translation table, normal half).

    **The translation owns the signs, never the user (INV-1).** The
    record's law fields are positive-magnitude physical quantities
    (``NormalLaw.__post_init__`` enforces ``tau_b_n > 0`` and
    ``gap <= 0``); the compression-side convention is applied here:
    under ``strain = x_hat . (u_j - u_i)`` with local-x the master's
    OUTWARD normal, separation is *positive* elongation, so a
    compression-carrying gap law must have ``Fy < 0`` **and**
    ``gap <= 0``. Both are forced with ``-abs(...)`` rather than passed
    through, so a future law-schema change that relaxes the record-side
    validation cannot silently produce a tension-only interface
    (``EPPGapMaterial::setTrialStrain`` branches on ``sign(fy)`` and only
    *warns* on a mismatched pair, `EPPGapMaterial.cpp:109-168`).

    ``gap == 0`` is normalised to ``+0.0`` (``-abs(0.0)`` is ``-0.0``,
    which would render a pointless ``-0.0`` in the deck); the S2
    primitive exempts ``gap == 0`` from its sign check either way.
    """
    from ...material.uniaxial import ENT, ElasticMaterial, ElasticPPGap

    kind = getattr(law, "kind", None)
    k = float(getattr(law, "k_per_area"))
    if kind == "ent":
        return ENT(E=k * a_trib)
    if kind == "elastic":
        return ElasticMaterial(E=k * a_trib)
    if kind == "epp_gap":
        gap = -abs(float(getattr(law, "gap")))
        if gap == 0.0:
            gap = 0.0                      # normalise -0.0 out of the deck
        return ElasticPPGap(
            E=k * a_trib,
            Fy=-abs(float(getattr(law, "tau_b_n")) * a_trib),
            gap=gap,
        )
    raise BridgeError(
        f"interface: unknown NormalLaw kind {kind!r} — the emit-time "
        f"translation table (ADR 0093 D1) covers 'ent', 'epp_gap' and "
        f"'elastic'. A new law kind needs a row here."
    )


def _interface_tangential_material(a_trib: float, law: object) -> "UniaxialMaterial":
    """Translate a :class:`TangentialLaw` to its typed uniaxial
    primitive (ADR 0093 D1 translation table, tangential half).

    ``epp`` is the correction the ADR review forced: ``ElasticPP`` takes
    a yield **strain**, not a force (``uniaxialMaterial ElasticPP $tag
    $E $epsyP``, fork `ElasticPPMaterial.cpp:52-76`). So
    ``epsyP = tau_b / k_per_area`` — ``A_trib`` cancels, and the
    physical yield force ``tau_b * A_trib`` is the emergent product
    ``E * epsyP``. ``epsyP`` is therefore the SAME on every pair of one
    interface while ``E`` scales with the tributary area.
    """
    from ...material.uniaxial import ElasticMaterial, ElasticPP

    kind = getattr(law, "kind", None)
    k = float(getattr(law, "k_per_area"))
    if kind == "elastic":
        return ElasticMaterial(E=k * a_trib)
    if kind == "epp":
        return ElasticPP(
            E=k * a_trib,
            epsyP=float(getattr(law, "tau_b")) / k,
        )
    raise BridgeError(
        f"interface: unknown TangentialLaw kind {kind!r} — the emit-time "
        f"translation table (ADR 0093 D1) covers 'epp' and 'elastic'. A "
        f"new law kind needs a row here."
    )


def _interface_is_3d(rec: object) -> bool:
    """Is this a 3-D (dim-2 surface master) interface record?

    Read off the record's OWN frame width, never off ``ndm``: six floats
    is the 2-D line master's ``-orient`` argument, nine is the 3-D
    surface master's ``(n, t1, t2)`` triad (ADR 0093 D2 / TIMs A10 S2).
    The record is what carries the dimension, so a record reaching emit
    through ``g.compose``, an h5 reload or a stage claim is classified
    the same way on every route — and a record whose width disagrees
    with the model's ``ndm`` is refused by :func:`_validate_interface_ndf`
    rather than silently emitting the other dimension's element shape.
    """
    orient = getattr(rec, "orient", None)
    return orient is not None and len(orient) == 9


def _validate_interface_ndf(
    rec: object,
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
    ndm: int,
) -> None:
    """ADR 0093 D4 / S4-refinement gate — the pair's declared bridging
    must match the endpoints' *actual* ndf.

    ``ZeroLength::setDomain`` (fork ``ZeroLength.cpp:611-673``) errors on
    ``dofNd1 != dofNd2``, and ``infer_node_ndf`` deliberately skips the
    zeroLength family (build.py:437) — so an interface element
    contributes no ndf of its own and both real endpoints must get
    theirs from the structural elements they belong to. Two ways the
    declaration can be wrong, both silent-wrong-model without this gate:

    * ``slave_ndf`` left at the default while the slave really is a
      3-dof beam node — the deck emits a mixed-ndf zeroLength that the
      engine refuses (or, worse, a future engine silently truncates).
    * ``slave_ndf=3`` declared against a 2-dof slave — the phantom and
      its equalDOF are pure noise, and the `equalDOF` would tie dofs the
      user never meant to bridge.

    In 3-D (a nine-float record, TIMs A10 S3) there is no phantom to
    declare — fork #808 / ADR 96 joins the mixed pair directly — so the
    gate becomes ADR 96's own rule, imported from the resolver as
    ``accepts_3d_ndf_pair`` rather than restated here.
    """
    name = getattr(rec, "name", None)
    label = f" {name!r}" if name else ""
    master = int(getattr(rec, "master_node"))
    slave = int(getattr(rec, "slave_node"))
    phantom = getattr(rec, "phantom_node", None)

    def ndf_of(n: int) -> int:
        return int(effective_ndf.get(int(n), int(envelope_ndf)))

    if _interface_is_3d(rec):
        _validate_interface_ndf_3d(
            label, master, slave, phantom,
            ndf_of(master), ndf_of(slave), ndm,
        )
        return

    if int(ndm) != 2:
        raise BridgeError(
            f"interface{label}: this record carries a six-float frame, "
            f"i.e. a 2D dim-1 line master (ADR 0093 D2), but the model is "
            f"ndm={int(ndm)}. A 3D interface resolves a dim-2 surface "
            f"master and carries a nine-float (n, t1, t2) frame."
        )

    m_ndf, s_ndf = ndf_of(master), ndf_of(slave)
    if m_ndf != 2:
        raise BridgeError(
            f"interface{label}: master node {master} has ndf={m_ndf}, but "
            f"the interface master must be a 2-dof 2D continuum node "
            f"(ADR 0093 INV-1 — iNode is always the real continuum node). "
            f"Check which label you passed as the master."
        )
    if phantom is None:
        if s_ndf != 2:
            raise BridgeError(
                f"interface{label}: pair (master={master}, slave={slave}) "
                f"was resolved WITHOUT a phantom bridge (slave_ndf omitted "
                f"or =2), but the slave node's actual ndf is "
                f"slave_ndf={s_ndf}. A zeroLength cannot join a 2-dof "
                f"master to a {s_ndf}-dof slave — ZeroLength::setDomain "
                f"refuses dofNd1 != dofNd2. Declare "
                f"g.constraints.interface(..., slave_ndf={s_ndf}) so the "
                f"resolver mints the phantom bridge (ADR 0093 D4)."
            )
        return

    if s_ndf != 3:
        raise BridgeError(
            f"interface{label}: pair (master={master}, slave={slave}) was "
            f"resolved WITH a phantom bridge (slave_ndf=3), but the slave "
            f"node's actual ndf is {s_ndf}. The phantom + equalDOF would "
            f"bridge nothing. Drop slave_ndf= from "
            f"g.constraints.interface() so the pair connects directly "
            f"(ADR 0093 D4)."
        )
    p_ndf = int(getattr(rec, "phantom_ndf", 0) or 0)
    if p_ndf != m_ndf:
        raise BridgeError(
            f"interface{label}: phantom node {int(phantom)} carries "
            f"ndf={p_ndf} but the master node {master} has ndf={m_ndf} — "
            f"the zeroLength joins those two, and the engine refuses "
            f"dofNd1 != dofNd2 (ADR 0093 D4)."
        )


def _validate_interface_ndf_3d(
    label: str,
    master: int,
    slave: int,
    phantom: "int | None",
    m_ndf: int,
    s_ndf: int,
    ndm: int,
) -> None:
    """The 3-D half of :func:`_validate_interface_ndf` (TIMs A10 S3).

    The rule is IMPORTED from the resolver
    (:func:`~apeGmsh._kernel.resolvers._interface_resolver.accepts_3d_ndf_pair`),
    never restated: the resolver's ``slave_ndf`` gate and this emit-time
    gate must agree by construction, and a second copy is a second thing
    to drift. The fork joins any 3-D pair with both ends ndf >= 3,
    acting on DOFs 1-3 with every DOF past the third an untouched
    passenger (fork #808 / ADR 96); a pair below that is a warning plus
    an inert element there, so it is refused here, before a line is
    written.

    It is the *rule* and not ADR 96's list of example pairs: the list
    omits ``(4, 6)`` — a u-p soil master under a shell raft — which the
    fork takes like any other, and which the resolver's
    ``slave_ndf=6`` already promises (adversarial review F1, measured on
    build ``1652f945c``).

    D4 does not cross over: in 3-D no phantom is minted at all (that is
    what the fork's relaxation retires), so a record carrying one is a
    resolver-contract violation rather than a user mistake.
    """
    from apeGmsh._kernel.resolvers._interface_resolver import (
        _MIN_3D_NDF,
        _NAMED_3D_NDF_PAIRS,
        accepts_3d_ndf_pair,
    )
    from ..._target import TIMS_FORK_BATCH_MIN_BUILD

    if int(ndm) != 3:
        raise BridgeError(
            f"interface{label}: this record carries a nine-float "
            f"(n, t1, t2) frame, i.e. a 3D dim-2 surface master (ADR 0093 "
            f"D2 / TIMs A10), but the model is ndm={int(ndm)}. A 2D "
            f"interface carries six floats."
        )
    if phantom is not None:
        raise BridgeError(
            f"interface{label}: pair (master={master}, slave={slave}) "
            f"carries phantom node {int(phantom)}, but a 3D interface "
            f"mints no phantom — fork #808 / ADR 96 joins the mixed pair "
            f"directly, which is what retires the ADR 0093 D4 bridge in "
            f"3D. A phantom here means the record and the resolver "
            f"disagree."
        )
    if not accepts_3d_ndf_pair(m_ndf, s_ndf):
        raise BridgeError(
            f"interface{label}: pair (master={master}, slave={slave}) has "
            f"ndf=({m_ndf}, {s_ndf}), which no 3D zeroLength accepts. The "
            f"fork joins a 3D pair whose ends BOTH carry ndf >= "
            f"{_MIN_3D_NDF}, acting on DOFs 1-3 with every DOF past the "
            f"third an untouched passenger (fork #808 / ADR 96, minimum "
            f"build {TIMS_FORK_BATCH_MIN_BUILD} = "
            f"TIMS_FORK_BATCH_MIN_BUILD) — e.g. "
            f"{list(_NAMED_3D_NDF_PAIRS)}, and any other pair over that "
            f"floor. Below it the engine warns and leaves the element "
            f"inert, so the interface would silently do nothing."
        )


#: How far the record's ``t2`` may sit from ``n x t1`` before the 3-D
#: frame is refused (TIMs A10 S3).  Tight, because the resolver builds
#: the triad exactly and ``g.compose`` only rotates it — anything looser
#: would be tolerating a real error rather than float noise.
_ORIENT_TRIAD_TOL = 1e-9


def _validate_interface_orient_triad(rec: "InterfaceRecord") -> None:
    """A 3-D record's frame must be orthonormal, and its third vector
    must BE ``n x t1``.

    ``zeroLength -orient x1 x2 x3 yp1 yp2 yp3`` takes only TWO vectors
    and derives the third itself: local-1 is ``x``, local-2 is the part
    of ``yp`` orthogonal to ``x``, local-3 is ``1 x 2``
    (``ZeroLength::setUp``).  So emitting the record's ``(n, t1)`` gives
    the element ``(n, t1, n x t1)`` — which is the record's own ``t2``
    only if the record's triad is right-handed.  S1's ``_tangent_pair``
    builds it that way; asserted here rather than trusted, because a
    record can reach emit through ``g.compose`` (which rotates every
    stacked vector), an h5 reload or a hand build, and a flipped ``t2``
    would put the ``-dir 3`` slider on the opposite tangent with no
    other symptom in the deck.

    **Orthonormality is checked first (TIMs A10 S4).** The ``t2 == n x
    t1`` rule alone passes a ``t1`` tilted OUT of the tangent plane, as
    long as ``t2`` was built from the same skewed ``t1``: the S3
    adversarial review MEASURED a 10-degree skew sailing through and
    then being silently re-orthogonalised by ``ZeroLength::setUp``,
    which gives the right answer for the WRONG frame — the springs act
    along a triad the record does not describe, so a per-pair
    ``spring_force_1`` no longer means what the record says it means.
    Refused here instead, on the same 1e-9 budget.
    """
    frame = np.asarray(rec.orient, dtype=float).reshape(3, 3)
    n, t1, t2 = frame[0], frame[1], frame[2]
    n_err = abs(float(np.linalg.norm(n)) - 1.0)
    t1_err = abs(float(np.linalg.norm(t1)) - 1.0)
    dot_err = abs(float(np.dot(n, t1)))
    if max(n_err, t1_err, dot_err) > _ORIENT_TRIAD_TOL:
        name = getattr(rec, "name", None)
        label = f" {name!r}" if name else ""
        raise BridgeError(
            f"interface{label}: pair (master={int(rec.master_node)}, "
            f"slave={int(rec.slave_node)}) carries a frame that is not "
            f"ORTHONORMAL — |n|-1={n_err:.3e}, |t1|-1={t1_err:.3e}, "
            f"n.t1={dot_err:.3e} (budget {_ORIENT_TRIAD_TOL:g}) for "
            f"n={tuple(n)}, t1={tuple(t1)}. The zeroLength -orient "
            f"argument is (x, yp) and ZeroLength::setUp silently "
            f"re-orthogonalises it, so a skewed frame runs to a "
            f"plausible answer whose springs act along a triad the "
            f"record does not describe (ADR 0093 D2 / TIMs A10 S4)."
        )
    err = float(np.linalg.norm(np.cross(n, t1) - t2))
    if err > _ORIENT_TRIAD_TOL:
        name = getattr(rec, "name", None)
        label = f" {name!r}" if name else ""
        raise BridgeError(
            f"interface{label}: pair (master={int(rec.master_node)}, "
            f"slave={int(rec.slave_node)}) carries a frame whose t2="
            f"{tuple(t2)} is not n x t1={tuple(np.cross(n, t1))} "
            f"(off by {err:.3e} > {_ORIENT_TRIAD_TOL:g}). The zeroLength "
            f"-orient argument is only (n, t1) and the engine derives "
            f"local-3 as 1 x 2, so a left-handed record would put the "
            f"second tangential slider on -t2 while the record says t2 "
            f"(ADR 0093 D2 / TIMs A10 S3)."
        )


def _validate_interface_records(
    records: "Sequence[InterfaceRecord]",
    *,
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
    ndm: int,
) -> None:
    """Gate a whole interface pool before a single line is emitted — a
    deck half-written then aborted is worse than one never started.

    Shared by the base pass (:func:`emit_interfaces`, which validates
    the WHOLE side-list including stage-claimed rows), the stage pass
    (:func:`emit_stage_interfaces`) and the partitioned owner-rank plan
    — so a refusal here reaches every route into
    :func:`_emit_interface_record`, on every dimension.
    """
    for rec in records:
        _validate_interface_ndf(rec, effective_ndf, envelope_ndf, ndm)
        if rec.orient is None:
            raise BridgeError(
                f"interface: record for pair (master="
                f"{int(rec.master_node)}, slave={int(rec.slave_node)}) "
                f"carries no orient 6-tuple. Emitting the zeroLength "
                f"without -orient would silently fall back to the GLOBAL "
                f"frame, i.e. a normal law acting along global x — the "
                f"silent sign error ADR 0093 INV-1 exists to kill."
            )
        if _interface_is_3d(rec):
            _validate_interface_orient_triad(rec)
        if rec.phantom_node is not None and rec.phantom_coords is None:
            raise BridgeError(
                f"interface: record for pair (master="
                f"{int(rec.master_node)}, slave={int(rec.slave_node)}) "
                f"carries phantom_node={int(rec.phantom_node)} but no "
                f"phantom_coords — the phantom cannot be declared."
            )


def _register_interface_phantoms(
    emitter: "Emitter", records: "Sequence[InterfaceRecord]",
) -> None:
    """ADR 0093 D4(b): interface phantoms must join the phantom-tag
    predicate BEFORE their ``node()`` calls, or the H5 emitter
    classifies them as ordinary (real broker) nodes.

    Additive union — the MP pass (``emit_mp_constraints`` step 0) may
    already have installed its own set; same pattern as
    :func:`emit_stage_mp_constraints`.  The predicate is consulted
    **per ``node()`` call** (``H5Emitter.node`` → :func:`is_phantom_node`),
    so each pass registering its own records right before emitting them
    is sufficient: a stage-claimed phantom registers inside the stage
    block, immediately before the ``node()`` that declares it.
    """
    from ..tag_resolution import ATTR_PHANTOM_NODE_TAGS, set_phantom_node_tags

    phantoms = {
        int(rec.phantom_node) for rec in records
        if rec.phantom_node is not None
    }
    if phantoms:
        existing: "frozenset[int]" = getattr(
            emitter, ATTR_PHANTOM_NODE_TAGS, frozenset())
        set_phantom_node_tags(emitter, set(existing) | phantoms)


class PlannedInterface(NamedTuple):
    """One interface record as the tag plan numbers it:
    ``(normal_mat_tag, tangential_mat_tag, element_tag)``."""

    record: Any
    tags: tuple[int, int, int]


def plan_interface_tags(
    records: "Iterable[InterfaceRecord]", tags: TagAllocator,
) -> tuple[PlannedInterface, ...]:
    """Plan every record's material + element tags, in the given order.

    The allocation loop of :func:`allocate_interface_tags`, moved out of
    the emit (ADR 0114 D4, amended). Per record the allocator is consumed
    in the ``(uniaxialMaterial, uniaxialMaterial, element)`` sequence the
    pre-S8 inline allocation used, so the per-kind counters advance
    exactly as before.
    """
    out: list[PlannedInterface] = []
    for rec in records:
        n_tag = tags.allocate("uniaxialMaterial")
        t_tag = tags.allocate("uniaxialMaterial")
        ele_tag = tags.allocate("element")
        out.append(PlannedInterface(rec, (n_tag, t_tag, ele_tag)))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class InterfacePlan:
    """The interface records' tags, in the order the emit planned them.

    ``fem`` is the FEM snapshot whose ``elements.interfaces`` were
    planned, and ``lines`` each record with its ``(normal, tangential,
    element)`` tags: on a flat emit the unclaimed records, then each
    stage's claimed ones, as their passes write them; on a partitioned
    emit the same sequence as one pre-pass (ADR 0093 S8/S9). A plan
    holds each record once and every record of ``fem`` (:meth:`check_covers`),
    or raises :class:`TagLawError` when it is made.
    """

    fem: Any
    lines: tuple[PlannedInterface, ...]
    _index: "dict[int, tuple[int, int, int]]" = field(
        init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        index: "dict[int, tuple[int, int, int]]" = {}
        for line in self.lines:
            if id(line.record) in index:
                raise TagLawError(
                    "the interface plan holds one record twice: an emit "
                    "writes each once (ADR 0114 D4, amended)."
                )
            index[id(line.record)] = line.tags
        object.__setattr__(self, "_index", index)
        self.check_covers()

    def check_covers(self) -> None:
        """Raise :class:`TagLawError` unless the plan holds exactly the
        records of its FEM's ``elements.interfaces``.

        The writers look each record's tags up by identity, so a plan
        that dropped a record would only fail when that record is written;
        this check refuses it, and a plan that holds another model's
        record in its place, when the plan is made.
        """
        planned = [line.record for line in self.lines]
        records = interface_records(self.fem)
        if sorted(map(id, planned)) != sorted(map(id, records)):
            raise TagPlanMiss(
                f"{_coverage_miss('interface', planned, records)}: the "
                f"interface plan holds {len(planned)} records, but the FEM "
                f"carries {len(records)}; the plan does not cover the model "
                "(ADR 0114 D4, amended)."
            )

    def for_fem(self, fem: "FEMData") -> "InterfacePlan":
        """The plan, which must have been made over ``fem``; a plan made
        over another FEM snapshot raises :class:`TagLawError`."""
        if fem is not self.fem:
            raise TagPlanMiss(_FOREIGN_FEM.format(family="interface"))
        return self

    def tags_for(
        self, records: "Iterable[InterfaceRecord]",
    ) -> "dict[int, tuple[int, int, int]]":
        """``{id(record): tags}`` for ``records``; a record the plan does
        not hold raises :class:`TagLawError`."""
        out: "dict[int, tuple[int, int, int]]" = {}
        for rec in records:
            planned = self._index.get(id(rec))
            if planned is None:
                raise TagPlanMiss(
                    f"the interface plan holds no tags for {plan_owner(rec)}, "
                    "which this emit writes: the plan was not made for this "
                    "emit (ADR 0114 D4, amended)."
                )
            out[id(rec)] = planned
        return out


def allocate_interface_tags(
    records: "Sequence[InterfaceRecord]", tag_plan: "TagPlan",
) -> "dict[int, tuple[int, int, int]]":
    """Every record's material + element tags, keyed by ``id(record)`` —
    ADR 0093 INV-5 / ADR 0027 §"Tag determinism".

    Returns ``{id(record): (normal_mat_tag, tangential_mat_tag,
    element_tag)}``, the tags ``tag_plan`` holds for ``records``
    (:meth:`InterfacePlan.tags_for`; ADR 0114 D4, amended). It mints
    nothing: a record the plan does not hold raises :class:`TagPlanMiss`.

    Splitting allocation from emission is what lets the partitioned
    path allocate ALL interface tags before the per-rank fan-out (so a
    record's tags do not depend on which rank owns it, and 1-rank and
    N-rank decks stay byte-comparable), while the flat and stage passes
    consume the SAME plan so the two paths cannot drift.
    """
    return tag_plan.interfaces.planned().tags_for(records)


def _plan_rank_interfaces(
    records: "Sequence[InterfaceRecord]",
    partitions: "Iterable[object]",
) -> "dict[int, list[tuple[InterfaceRecord, tuple[int, ...]]]]":
    """Resolve each interface record's owner rank + ghost node set
    (ADR 0093 INV-5 — element-side ownership).

    Owner = the rank owning the record's ``backing_element`` (a
    highest-dimension domain continuum element, stamped at resolve
    time).  Node-tally ownership (ADR 0092 INV-1) is wrong here *by
    construction*: the pair's nodes are co-located, so a cut hugging
    the interface replicates both onto both ranks and the tally is
    undecidable — the backing element is the locality information the
    nodes cannot carry.

    Two loud preconditions make the pick exact rather than heuristic
    (INV-5, as scoped by the adversarial review):

    * **Single owner, asserted directly.** The backing element must
      appear in EXACTLY ONE partition's ``element_ids``.  Membership is
      counted across ``fem.partitions[*]`` here — NOT read off
      :func:`build_element_partition_owner`, whose documented
      first-seen tiebreak would silently hide a duplicated element.
      ``_extract_partitions`` replicates multi-partition boundary-entity
      elements across every partition holding the entity; a stamped
      element that turns out replicated (or absent) is a resolver-
      contract violation and refuses with the record, the element and
      the partitions holding it.
    * **Master node native to the owner.** The backing element lives on
      the owner rank, and an element's nodes are in its partition's
      node set by ``extract_partitions`` construction — asserted anyway,
      loudly, so a drift in extraction semantics cannot silently ghost
      the master.

    The ghost set is the record's real slave node when the owner does
    not natively own it (the foreign beam/slave node of ADR 0093 INV-5,
    ghosted as geometry + SP only per ADR 0092 INV-7).  The phantom is
    minted by the owner rank and is never foreign.

    Returns ``{owner_rank: [(record, ghost_node_ids), ...]}`` with each
    rank's records in the given (flat side-list) order.  Pure — no
    emitter access, no tag allocation.
    """
    recs = list(records)
    if not recs:
        return {}
    parts = list(partitions)

    backing = np.asarray(
        sorted({int(r.backing_element) for r in recs}), dtype=np.int64,
    )
    needed_nodes = np.asarray(
        sorted({int(r.master_node) for r in recs}
               | {int(r.slave_node) for r in recs}),
        dtype=np.int64,
    )

    # element -> [every rank whose partition holds it] (NOT first-seen),
    # and rank -> {the master/slave nodes it natively owns}.
    holder_ranks: "dict[int, list[int]]" = {}
    native_nodes: "dict[int, set[int]]" = {}
    for idx, part in enumerate(parts):
        # Duck-typed seam (stub FEMs) — the record itself is not
        # consulted by the conversion; see the helper's contract.
        rank = runtime_rank_from_partition_record(
            cast("PartitionRecord", part), idx)
        eids = np.asarray(
            getattr(part, "element_ids", ()), dtype=np.int64,
        )
        if eids.size:
            for eid in backing[np.isin(backing, eids)]:
                holder_ranks.setdefault(int(eid), []).append(rank)
        nids = np.asarray(getattr(part, "node_ids", ()), dtype=np.int64)
        native_nodes[rank] = (
            {int(n) for n in needed_nodes[np.isin(needed_nodes, nids)]}
            if nids.size else set()
        )

    plan: "dict[int, list[tuple[InterfaceRecord, tuple[int, ...]]]]" = {}
    for pos, rec in enumerate(recs, start=1):
        name = getattr(rec, "name", None)
        label = f"#{pos}" + (f" ({name!r})" if name else "")
        eid = int(rec.backing_element)
        holders = holder_ranks.get(eid, [])
        if len(holders) != 1:
            where = (
                "NO partition's element set" if not holders else
                f"{len(holders)} partitions' element sets (runtime ranks "
                f"{holders})"
            )
            raise BridgeError(
                f"apeSees: g.constraints.interface() record {label} "
                f"(master={int(rec.master_node)}, "
                f"slave={int(rec.slave_node)}) stamps backing element "
                f"{eid}, which appears in {where}. ADR 0093 INV-5 "
                "requires the stamped backing element to be a "
                "highest-dimension DOMAIN continuum element owned by "
                "exactly one partition — top-dimension elements are "
                "never replicated across partitions, so this indicates "
                "a boundary-entity element was stamped (or the "
                "partition records are inconsistent with the resolved "
                "interface). The owner pick is exact under this "
                "asserted precondition, and explicit when it fails — "
                "never silently first-seen."
            )
        owner = holders[0]
        master = int(rec.master_node)
        if master not in native_nodes.get(owner, ()):
            raise BridgeError(
                f"apeSees: g.constraints.interface() record {label} — "
                f"owner rank {owner} holds backing element {eid} but "
                f"does not natively own the master node {master} "
                "(ADR 0093 INV-5). extract_partitions puts every "
                "element's nodes in its partition's node set, so the "
                "master node of a pair is native to the backing "
                "element's rank by construction — a violation means "
                "the partition records and the resolved interface "
                "disagree, and emitting would ghost the master onto "
                "its own owner."
            )
        slave = int(rec.slave_node)
        ghosts: "tuple[int, ...]" = (
            () if slave in native_nodes.get(owner, set()) else (slave,)
        )
        plan.setdefault(owner, []).append((rec, ghosts))
    return plan


def _emit_interface_record(
    emitter: "Emitter", rec: "InterfaceRecord",
    pre_allocated: "tuple[int, int, int]",
) -> None:
    """Emit ONE record's atomic unit — phantom ``node`` → nested
    ``equalDOF`` → the two tributary-scaled uniaxials → the
    ``zeroLength``.

    The single per-record core shared by the base pass
    (:func:`emit_interfaces`), the stage-block pass
    (:func:`emit_stage_interfaces`, ADR 0093 S7) and the partitioned
    owner-rank pass (ADR 0093 S8), so the D1 translation table and
    INV-1's node order exist in exactly one place.

    Two element shapes, chosen off the record's frame width
    (:func:`_interface_is_3d`) — never off ``ndm``, which the record may
    outlive: a 2-D line master emits ``-mat mN mT -dir 1 2`` with the
    record's six-float ``-orient``, a 3-D surface master (TIMs A10 S3)
    emits ``-mat mN mT mT -dir 1 2 3`` with the first six of its nine.
    See the comment at the ``args`` fork for why the tangential tag is
    repeated rather than minted twice, and for what "two uncoupled
    sliders" costs.

    ``pre_allocated`` is this record's ``(normal_mat_tag,
    tangential_mat_tag, element_tag)`` triple from
    :func:`allocate_interface_tags` — allocation is separated from
    emission so the partitioned path can allocate every record's tags
    in flat order BEFORE the rank fan-out (ADR 0027 tag determinism).

    ``ndm`` sizes the phantom's coordinate list
    (:func:`node_coords_as_floats`).  This is the interface lane's own
    live bug, not a precaution: the resolver that mints the phantom is
    2-D-ONLY by construction (``_interface_resolver._PHANTOM_NDF = 2``),
    so before this parameter existed EVERY mixed-ndf interface phantom
    went out as ``node <tag> x y 0.0 -ndf <n>`` and the ``-ndf`` was
    swallowed — the phantom silently took the model envelope instead of
    the record's own ndf, and the zeroLength then joined two endpoints
    of different dof counts.
    """
    _emit_name(emitter, rec.name)

    if rec.phantom_node is not None:
        # ``phantom_coords`` / ``phantom_ndf`` / ``orient`` below are
        # Optional on the record but PROVEN non-None by
        # :func:`_validate_interface_records`, which every caller runs
        # over the whole pool before the first line is emitted — hence
        # the narrow ignores rather than re-raising here.
        xyz = node_coords_as_floats(
            rec.phantom_coords,  # type: ignore[arg-type]
        )
        emitter.node(
            int(rec.phantom_node), *xyz,
            ndf=int(rec.phantom_ndf),  # type: ignore[arg-type]
        )
        for pair in rec.equal_dof_records:
            emitter.equalDOF(
                int(pair.master_node), int(pair.slave_node),
                *(int(d) for d in pair.dofs),
            )
        j_node = int(rec.phantom_node)
    else:
        j_node = int(rec.slave_node)

    a_trib = float(rec.a_trib)
    m_normal = _interface_normal_material(a_trib, rec.normal_law)
    m_tangential = _interface_tangential_material(
        a_trib, rec.tangential_law)
    n_tag, t_tag, ele_tag = pre_allocated
    m_normal._emit(emitter, n_tag)
    m_tangential._emit(emitter, t_tag)

    args: "list[int | float | str]"
    if _interface_is_3d(rec):
        # TWO UNCOUPLED COULOMB SLIDERS, not a circular slip surface:
        # dir 2 and dir 3 each carry the tangential law in full, so
        # sliding along t1 and along t2 yield independently at
        # ``tau_b * A_trib`` rather than on a combined
        # ``|tau| <= tau_b * A_trib`` radius (the slip locus is a square
        # in the tangent plane, not a circle, and is up to sqrt(2) too
        # strong on the diagonal). That is the plan's own choice for S3
        # — a uniaxial bundle is what ADR 0093 D1 translates to — and
        # S4 measures what it costs.
        #
        # ONE tangential tag on both slots, not two: ZeroLength deep-
        # copies every ``-mat`` entry (``ZeroLength.cpp:405``,
        # ``theMaterial1d[i] = theMat[i]->getCopy()``), so the two
        # sliders carry fully independent state from a single declared
        # material — a second identical uniaxialMaterial line would buy
        # nothing and double the deck's material count.
        #
        # Only the first six floats go out: ``-orient`` IS (n, t1), and
        # the engine derives local-3 as ``1 x 2``, which
        # :func:`_validate_interface_orient_triad` has already proven
        # equals the record's t2.
        args = [
            int(rec.master_node), j_node,
            "-mat", n_tag, t_tag, t_tag,
            "-dir", 1, 2, 3,
            "-orient",
            *(float(v) for v in rec.orient[:6]),  # type: ignore[index]
        ]
    else:
        args = [
            int(rec.master_node), j_node,
            "-mat", n_tag, t_tag,
            "-dir", 1, 2,
            "-orient", *(float(v) for v in rec.orient),  # type: ignore[union-attr]
        ]
    # ADR 0049 node-pair convention for a minted (mesh-less) element:
    # sentinel fem_eid + the TRUE endpoint pair.  Without this the H5
    # emitter's sticky side channels leak the last mesh row's fem_eid
    # and connectivity onto every interface zeroLength — mislabeling
    # the per-pair springs channels in ``Results`` and corrupting the
    # argstack slicing (ADR 0093 S10 finding).
    set_current_fem_element_id(emitter, MISSING_FEM_ELEMENT_ID)
    set_element_nodes(emitter, (int(rec.master_node), j_node))
    emitter.element("zeroLength", ele_tag, *args)


def emit_stage_interfaces(
    records: "Sequence[InterfaceRecord]",
    emitter: "Emitter", tag_plan: "TagPlan",
    *,
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
    ndm: int,
) -> None:
    """Emit a stage's CLAIMED interface records inside the stage block
    (flat path, ADR 0093 S7 / INV-6).

    The liner-install pattern: ``g.constraints.interface(...,
    name="RockLiner")`` + ``s.interface(name="RockLiner")`` inside an
    ``ops.stage(...)`` block moves the whole per-pair unit out of the
    base pass and into the stage's block, so the interface is installed
    on the ALREADY-EQUILIBRATED ground rather than at ``t = 0``.

    Emitted AFTER the stage's activated topology and stage MP
    constraints, BEFORE the stage's ``domain_change`` barrier — the
    ``zeroLength``'s two endpoints (and, for a mixed-ndf pair, the
    phantom this pass mints) must be in the Domain when the element
    references them.  Element and material tags come from the SAME
    plan as the base pass's (:func:`allocate_interface_tags`), which
    numbers each stage's claimed records after the base pass, continuing
    the shared namespace (the :func:`emit_stage_mp_constraints`
    element-in-stage precedent).

    No-op when the stage claimed nothing.
    """
    if not records:
        return
    recs = list(records)
    _validate_interface_records(
        recs, effective_ndf=effective_ndf,
        envelope_ndf=envelope_ndf, ndm=ndm,
    )
    _register_interface_phantoms(emitter, recs)
    iface_tags = allocate_interface_tags(recs, tag_plan)
    for rec in recs:
        _emit_interface_record(emitter, rec, iface_tags[id(rec)])


def emit_interfaces(
    emitter: "Emitter", fem: "FEMData", tag_plan: "TagPlan",
    *,
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
    ndm: int,
    claimed_ids: "frozenset[int]" = frozenset(),
) -> None:
    """Emit one oriented coincident-pair ``zeroLength`` per resolved
    interface record (``g.constraints.interface``, ADR 0093 D5).

    Consumes ``fem.elements.interfaces`` —
    :class:`~apeGmsh._kernel.records._constraints.InterfaceRecord` rows
    produced by :class:`ConstraintsComposite` at FEM-build time. An
    additive side-list pass in the shape of :func:`emit_contacts`: it
    bypasses the ``_DISPATCH`` MP pipeline entirely and owns its own
    tag allocation. Per record, in this order (the golden tests pin it):

    1. the mixed-ndf **phantom** ``node(tag, x, y, z, ndf=<phantom_ndf>)``
       — the record's own ndf, NOT :func:`_emit_phantom_nodes`'s
       hardcoded 6 (ADR 0093 D4(a));
    2. the nested ``equalDOF(retained=beam node, constrained=phantom,
       1 2)``;
    3. the pair's **two** tributary-scaled uniaxial materials (normal
       then tangential), translated from the record's declarative laws
       by the D1 table above;
    4. the ``zeroLength`` itself, ``-mat mN mT -dir 1 2 -orient …``.

    **INV-1 (the whole point of the verb):** ``iNode`` is the real
    continuum ``master_node`` and ``jNode`` is the ``phantom_node`` when
    one was minted, else the real ``slave_node``. ZeroLength deformation
    is ``x_hat . (u_j - u_i)`` (fork ``ZeroLength.cpp:1991-2009``) and
    local-x is the master face's outward normal, so separation elongates
    and an ENT normal law carries zero force. Swapping the two endpoints
    converges just fine — as a tension-only interface. The golden tests
    pin the node order literally for exactly that reason.

    Materials go out through the typed S2/S1 primitives' own ``_emit``
    (they need no tag resolution, so this is byte-identical to a
    user-declared material and inherits every sign / range guard); the
    ``zeroLength`` line is hand-built like :func:`emit_contacts` rather
    than routed through the :class:`ZeroLength` primitive, because that
    primitive's ``_emit`` reads its endpoints from the emitter's
    element-nodes context and resolves material tags through the
    bridge's primitive-identity resolver — neither exists for records.

    Records claimed by ``s.interface(name=)`` (ADR 0093 S7) are SKIPPED
    here — their ``id(...)`` is in ``claimed_ids`` and they emit inside
    their owning stage's block via :func:`emit_stage_interfaces`.  They
    are still VALIDATED here, so a bad record fails loud before any
    line is written regardless of which pass owns its emit.

    Flat-path only: the partitioned path plans ownership per record
    (:func:`_plan_rank_interfaces`, ADR 0093 S8 / INV-5) and emits each
    atomic unit inside its owner rank's block via the same
    :func:`_emit_interface_record` core. No-op when the FEM snapshot
    exposes no ``elements.interfaces``.

    The tags come from :func:`plan_interface_tags` (ADR 0114 D4,
    amended); :func:`allocate_interface_tags` says how this reads them.
    """
    elements = getattr(fem, "elements", None)
    interfaces = (
        getattr(elements, "interfaces", None)
        if elements is not None else None
    )
    if not interfaces:
        return

    # Validate the WHOLE side-list — claimed rows included — before
    # emitting a single line.
    all_records = list(interfaces)
    _validate_interface_records(
        all_records, effective_ndf=effective_ndf,
        envelope_ndf=envelope_ndf, ndm=ndm,
    )

    unclaimed = [rec for rec in all_records if id(rec) not in claimed_ids]
    if not unclaimed:
        return
    tag_plan.interfaces.planned().for_fem(fem)   # refuses another model's plan
    _register_interface_phantoms(emitter, unclaimed)
    iface_tags = allocate_interface_tags(unclaimed, tag_plan)
    for rec in unclaimed:
        _emit_interface_record(emitter, rec, iface_tags[id(rec)])
