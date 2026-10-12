"""ndf: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
from typing import Container, Iterable, Mapping, Sequence

import numpy as np

from ..types import Element
from ...pattern.pattern import _LoadRecord, _SPRecord
from .._stacklevel import _stacklevel_outside_package
from .errors import BridgeError
from .records import (
    FixRecord,
    MassRecord,
    NdfRecord,
    SupportRecord,
)
from .pg_expand import (
    _record_node_ids,
    expand_pg_to_nodes,
    expand_spec_to_elements,
)


if TYPE_CHECKING:
    from apeGmsh._kernel.records._constraints import ConstraintRecord
    from ...emitter.base import Emitter
    from .ownership import SortedIntToInt
    from apeGmsh.mesh.FEMData import FEMData


def validate_node_ndf_element_compat(
    fem: "FEMData", elements: "Iterable[Element]",
) -> None:
    """Fail loud when one mesh node is shared by two elements whose
    per-node DOF requirements are mutually incompatible.

    Root cause (verified against OpenSees ``FE_Element::setID``,
    ``SRC/analysis/fe_ele/FE_Element.cpp``): an element sizes its
    equation-map array ``myID`` to its OWN ``numDOF`` (e.g. a 4-node
    tetrahedron declares ``numDOF = 12``).  When one of its nodes
    carries MORE DOFs than the element expects — the canonical case is
    a shell node (``ndf=6``) shared with a solid element (``ndf=3``),
    produced by fragmenting a shell surface onto a solid volume so they
    share interface nodes — the cumulative DOF count overflows
    ``numDOF``, ``setID`` returns ``-3``, and the element's stiffness is
    assembled into the WRONG global equations.  The result is a silent
    equilibrium violation: the structure deflects plausibly but only a
    fraction of the applied load reaches the supports
    (``Σ reactions ≠ Σ loads``).  No constraint handler or rotation
    clamp can repair it — the assembly itself is corrupt.

    The check is conservative: it fires ONLY when two elements with
    DISJOINT ``ndf_ok`` sets (e.g. shell ``{6}`` vs solid ``{3}``)
    genuinely share a node — a configuration OpenSees can never
    assemble (whatever single ndf the shared node is given, one of the
    two elements mis-maps).  Element types absent from the capability
    registry resolve to ``ndf_ok = None`` and are skipped (treated as
    unconstrained), so the guard never raises a false positive; it also
    never flags multi-ndf elements (beams / trusses carry ``{3, 6}`` /
    ``{2, 3, 6}``, which intersect both solids and shells) — a beam
    sharing a node with a solid is therefore NOT caught here (a rarer,
    distinct mistake; out of scope for this shell-on-solid guard).  The
    correct idiom for a real shell-on-solid interface is SEPARATE
    coincident nodes (shell ``ndf=6`` + solid ``ndf=3``) tied by
    ``g.constraints.equal_dof`` / ``tie`` on the translational DOFs
    (plus a shell-edge rotation clamp for the line-hinge), never shared
    nodes — see ADR 0033 / 0046.
    """
    from ..._element_capabilities import element_class_ndf_ok

    # Classify each spec once (skip the unclassifiable — fail-safe).
    classified: list[tuple["Element", str, "frozenset[int]"]] = []
    for spec in elements:
        etype = type(spec).__name__
        ndf_ok = element_class_ndf_ok(etype)
        if ndf_ok is not None:
            classified.append((spec, etype, ndf_ok))

    # Fast path: the per-node walk can only find a conflict if two
    # distinct ndf_ok families are mutually disjoint (shell {6} vs
    # solid {3}).  Single-family models (all solids, all shells, solids
    # + beams, ...) — the overwhelming common case — can never trip the
    # guard, so skip the O(E) connectivity expansion entirely.
    families = list({nf for _s, _e, nf in classified})
    has_disjoint_pair = any(
        not (families[i] & families[j])
        for i in range(len(families))
        for j in range(i + 1, len(families))
    )
    if not has_disjoint_pair:
        return None

    # node tag -> (running ndf_ok intersection, element type that last
    # narrowed it).  ``owner`` tracks the narrowing element so the error
    # names the element actually responsible for the surviving set, not
    # merely the first one seen (matters when a compatible multi-ndf
    # element is processed between two incompatible ones).
    acc: dict[int, "frozenset[int]"] = {}
    owner: dict[int, str] = {}
    for spec, etype, ndf_ok in classified:
        # ADR 0049: route via expand_spec_to_elements so a node-pair spec
        # (pg=None) contributes its two endpoints here.  For the adaptive
        # zeroLength family ndf_ok={1..6} is intersection-inert, so this is
        # a no-op for correctness — it runs only to keep every fan-out site
        # uniform (no caller ever passes pg=None to expand_pg_to_elements).
        for _eid, node_tags in expand_spec_to_elements(fem, spec):
            for raw in node_tags:
                t = int(raw)
                prev = acc.get(t)
                if prev is None:
                    acc[t] = ndf_ok
                    owner[t] = etype
                    continue
                inter = prev & ndf_ok
                if not inter:
                    raise BridgeError(
                        f"mesh node {t} is shared by element type "
                        f"{owner[t]!r} (per-node ndf {sorted(prev)}) and "
                        f"{etype!r} (per-node ndf {sorted(ndf_ok)}), whose "
                        f"DOF requirements are incompatible. OpenSees cannot "
                        f"assemble a node shared between elements with "
                        f"disjoint ndf — FE_Element::setID truncates the "
                        f"element's equation map, silently corrupting "
                        f"assembly (lost load / equilibrium violation). This "
                        f"is the classic shell-on-solid trap: a shell surface "
                        f"fragmented onto a solid volume so they share "
                        f"interface nodes. Fix: give the interface SEPARATE "
                        f"coincident nodes (do NOT fragment the shell into "
                        f"the solid) and tie their translational DOFs with "
                        f"g.constraints.equal_dof (conformal) or "
                        f"g.constraints.tie (non-matching), then clamp the "
                        f"shell-edge rotations for the line hinge. See ADR "
                        f"0033 / 0046 / the shell-on-solid idiom."
                    )
                if inter != prev:
                    acc[t] = inter
                    owner[t] = etype
    return None


#: ``ndf_ok`` of the adaptive zeroLength family — elements that accept any
#: per-node ndf (``{1..6}``) and therefore carry no inference opinion. A node
#: whose only incident elements are adaptive is omitted from the inferred map
#: and falls back to the ``ops.model`` envelope (see :func:`_infer_ndf_from_incidence`).
_ADAPTIVE_NDF_OK: "frozenset[int]" = frozenset({1, 2, 3, 4, 5, 6})


def _infer_ndf_from_incidence(
    node_to_classes: "dict[int, list[str]]", ndm: int,
) -> "dict[int, int]":
    """Pure core of ADR 0048 per-node ndf inference.

    ``node_to_classes`` maps a node tag -> the OpenSees element CLASS names
    incident on it. Returns ``{tag: ndf}`` where ``ndf`` is the ``max`` of each
    incident element's :func:`element_required_floor`, validated against every
    incident element's ``ndf_ok`` (the shell-on-solid / quad+beam ``∩`` gate).

    Raises :class:`BridgeError` when an incident element class is unclassifiable
    (no registry entry) or the chosen ``ndf`` is rejected by some incident
    element's ``ndf_ok`` — the latter being STRICTER than the disjoint-set check
    in :func:`validate_node_ndf_element_compat` (it catches a beam needing 6
    sharing a solid node accepting only 3, where ``{3,6} ∩ {3} = {3}`` is
    non-empty yet the node still cannot be assembled). The fix is always
    SEPARATE coincident nodes + ``equalDOF`` (ADR 0046 / 0048).
    """
    from ..._element_capabilities import (
        element_class_ndf_ok,
        element_required_floor,
    )

    result: dict[int, int] = {}
    for tag, classes in node_to_classes.items():
        incident: list[tuple[str, "frozenset[int]"]] = []
        floor = 0
        for cls in classes:
            ndf_ok = element_class_ndf_ok(cls)
            fl = element_required_floor(cls, ndm)
            if ndf_ok is None or fl is None:
                raise BridgeError(
                    f"node {tag}: element class {cls!r} is not in the "
                    f"capability registry, so its per-node ndf cannot be "
                    f"inferred. Add an _ELEM_REGISTRY entry (ndf_ok + "
                    f"ndf_required) for {cls!r} before emitting."
                )
            incident.append((cls, ndf_ok))
            # Adaptive elements (the zeroLength family: ``ndf_ok == {1..6}``)
            # carry no real per-node opinion — their partner / the structural
            # side supplies the count (the floor of 1 is a placeholder that
            # must never inflate the max). They are skipped here; a node
            # touched ONLY by adaptive elements gets no inferred value
            # (``floor`` stays 0) and is omitted below, so it falls back to
            # the ``ops.model`` envelope at emit (ADR 0048 / 0049).
            if ndf_ok == _ADAPTIVE_NDF_OK:
                continue
            if fl > floor:
                floor = fl
        if floor == 0:
            # Every incident element is adaptive → no inferred opinion.
            # Omit the node; the envelope supplies its ndf at emit time.
            continue
        for cls, ndf_ok in incident:
            if floor not in ndf_ok:
                raise BridgeError(
                    f"mesh node {tag}: its incident elements require "
                    f"ndf={floor} but {cls!r} only accepts ndf "
                    f"{sorted(ndf_ok)}. OpenSees cannot assemble a shared node "
                    f"whose ndf any incident element rejects "
                    f"(FE_Element::setID truncates / mis-maps the element's "
                    f"equation array). Give the interface SEPARATE coincident "
                    f"nodes and tie the shared DOFs with g.constraints.equal_dof "
                    f"(conformal) or g.constraints.tie (non-matching). See "
                    f"ADR 0046 / 0048."
                )
        result[tag] = floor
    return result


def infer_node_ndf(
    fem: "FEMData", elements: "Iterable[Element]", ndm: int,
) -> "dict[int, int]":
    """ADR 0048 per-node ndf inference over the declared element specs.

    Walks each element spec's physical group to its mesh nodes (via
    :func:`expand_pg_to_elements`), then applies
    :func:`_infer_ndf_from_incidence`. Returns ``{node_tag: ndf}`` for every
    node touched by >= 1 declared element. Nodes touched by NO declared
    element — and nodes touched only by adaptive elements — are **absent**
    from the map; in this pragmatic variant (ADR 0048 PR-1) they fall back
    to the ``ops.model`` envelope at emit (the purist clean break's
    fail-loud-on-orphan-mesh-node and ``ops.ndf`` decoupled channel are
    deferred).

    Implementation note (perf): semantics are exactly
    :func:`_infer_ndf_from_incidence` — the unit-tested reference core,
    kept above — but evaluated per element CLASS with numpy instead of
    per (element × node) incidence in Python. A node's inferred ndf
    depends only on the SET of classes incident on it, so one registry
    probe per class plus a per-class unique-node pass is equivalent,
    and it was the dominant flat-emit cost at large element counts.

    ADR 0074 extends this with per-SLOT contributions: a class carrying
    ``ndf_floor_per_slot`` (``LadrunoUP`` Taylor–Hood shapes) splits each
    fan-out chunk into slot-floor groups — vertex slots contribute
    ``ndm+1``, mid-edge slots ``ndm`` — each validated STRICTLY (the ok
    set is ``{floor}``; the fork element loud-errors on any deviation).
    Position-uniform shapes of a strict class likewise validate against
    ``{scalar floor}``.  Per-slot classes resolve only through THIS
    function; the reference core's class-name signature cannot carry
    slot information.
    """
    from ..._element_capabilities import (
        element_class_ndf_ok,
        element_ndf_slot_floors,
        element_ndf_strict,
        element_required_floor,
    )

    # ADR 0049: a node-pair spec (pg=None) contributes its two endpoints
    # via expand_spec_to_elements.  Inert for correctness on the adaptive
    # zeroLength family (skipped below, mirroring the reference core), so
    # a decoupled ground stays absent from the inferred map and remains
    # ops.ndf-eligible; routed only to avoid an expand_pg_to_elements(None)
    # whole-mesh fan-out.
    class_chunks: dict[str, list[tuple[int, ...]]] = {}
    for spec in elements:
        chunks = class_chunks.setdefault(type(spec).__name__, [])
        for _eid, node_tags in expand_spec_to_elements(fem, spec):
            chunks.append(node_tags)

    # One registry probe + one unique-node pass per class.
    structural: list[tuple[str, "frozenset[int]", int, np.ndarray]] = []
    for cls, chunks in class_chunks.items():
        if not chunks:
            continue
        ndf_ok = element_class_ndf_ok(cls)
        fl = element_required_floor(cls, ndm)
        if ndf_ok is None or fl is None:
            tag = int(chunks[0][0])
            raise BridgeError(
                f"node {tag}: element class {cls!r} is not in the "
                f"capability registry, so its per-node ndf cannot be "
                f"inferred. Add an _ELEM_REGISTRY entry (ndf_ok + "
                f"ndf_required) for {cls!r} before emitting."
            )
        # Adaptive classes carry no per-node opinion (no floor, and
        # their ndf_ok = {1..6} can never reject a floor) — drop them
        # here; adaptive-only nodes stay absent from the result.
        if ndf_ok == _ADAPTIVE_NDF_OK:
            continue
        # ADR 0074: split a strict per-slot class's chunks into
        # slot-floor groups; shapes without a slot map take the scalar
        # floor.  Strict classes validate against {floor} exactly.
        strict = element_ndf_strict(cls)
        scalar_conns: list[tuple[int, ...]] = []
        slot_ids: dict[int, list[int]] = {}
        if strict:
            for conn in chunks:
                slot_floors = element_ndf_slot_floors(cls, len(conn))
                if slot_floors is None:
                    scalar_conns.append(conn)
                    continue
                for n, f in zip(conn, slot_floors):
                    slot_ids.setdefault(int(f), []).append(int(n))
        else:
            scalar_conns = chunks
        if scalar_conns:
            ids = np.unique(np.fromiter(
                (n for conn in scalar_conns for n in conn), dtype=np.int64,
            ))
            ok_set = frozenset({int(fl)}) if strict else ndf_ok
            structural.append((cls, ok_set, int(fl), ids))
        for f, id_list in sorted(slot_ids.items()):
            ids = np.unique(np.asarray(id_list, dtype=np.int64))
            structural.append((cls, frozenset({int(f)}), int(f), ids))

    if not structural:
        return {}

    # Per-node floor = max of the incident structural classes' floors.
    all_ids = np.unique(
        np.concatenate([ids for _cls, _ok, _fl, ids in structural])
    )
    floors = np.zeros(len(all_ids), dtype=np.int64)
    for _cls, _ndf_ok, fl, ids in structural:
        pos = np.searchsorted(all_ids, ids)
        floors[pos] = np.maximum(floors[pos], fl)

    # The chosen floor must sit inside EVERY incident class's ndf_ok
    # (the shell-on-solid / beam-on-brick strict gate of the core).
    for cls, ndf_ok, _fl, ids in structural:
        pos = np.searchsorted(all_ids, ids)
        vals = floors[pos]
        ok = np.isin(vals, np.fromiter(ndf_ok, dtype=np.int64))
        if not bool(ok.all()):
            bad = int(np.argmax(~ok))
            raise BridgeError(
                f"mesh node {int(ids[bad])}: its incident elements require "
                f"ndf={int(vals[bad])} but {cls!r} only accepts ndf "
                f"{sorted(ndf_ok)}. OpenSees cannot assemble a shared node "
                f"whose ndf any incident element rejects "
                f"(FE_Element::setID truncates / mis-maps the element's "
                f"equation array). Give the interface SEPARATE coincident "
                f"nodes and tie the shared DOFs with g.constraints.equal_dof "
                f"(conformal) or g.constraints.tie (non-matching). See "
                f"ADR 0046 / 0048."
            )

    return {
        int(t): int(f)
        for t, f in zip(all_ids.tolist(), floors.tolist())
    }


def validate_adaptive_element_endpoints(
    fem: "FEMData",
    elements: "Iterable[Element]",
    ndm: int,
    inferred: "dict[int, int]",
    envelope_ndf: int,
) -> None:
    """Fail loud when an adaptive element's endpoints resolve to different ndf.

    Adaptive elements (the zeroLength family, ``ndf_ok == {1..6}``) accept
    any per-node ndf but OpenSees requires **both** ends of a
    ``zeroLength`` / ``twoNodeLink`` / ``CoupledZeroLength`` to carry the
    SAME ndf (``ZeroLength.cpp``: ``dofNd1 == dofNd2``; on mismatch
    ``setDomain`` bails and the element is **silently absent** — analysis
    proceeds with no spring).

    Inference omits adaptive-only nodes (a spring-to-ground node touched by
    no structural element falls to the ``ops.model`` envelope), so a spring
    whose structural end infers a value != envelope would silently emit
    mismatched ends. This guard catches that — and the symmetric case of a
    spring straddling two structural regions of differing ndf — at build
    time, naming the element and the fix.

    ADR 0049 — this is the correctness spine of the node-pair zeroLength
    form: routing the fan-out through :func:`expand_spec_to_elements` feeds
    a node-pair spec's two endpoints (one typically a decoupled ground sized
    by ``ops.ndf``) into this exact check.  **It is load-bearing that the
    ``inferred`` argument is the *effective* map (inferred ∪ ``ops.ndf``
    overlay)** — the build site passes ``effective_ndf``; if it ever passed
    the raw inferred map, a correct ``ops.ndf(ground, K != envelope)`` would
    fall to the envelope here and falsely raise.  ``ZeroLengthSection`` is
    intentionally NOT covered (it is non-adaptive; node-pair is forbidden
    for it).
    """
    from ..._element_capabilities import element_class_ndf_ok

    for spec in elements:
        cls = type(spec).__name__
        if element_class_ndf_ok(cls) != _ADAPTIVE_NDF_OK:
            continue
        for eid, node_tags in expand_spec_to_elements(fem, spec):
            eff = {
                int(n): inferred.get(int(n), int(envelope_ndf))
                for n in node_tags
            }
            if len(set(eff.values())) > 1:
                # Fork #808 / ADR 96: a 3-D zeroLength-family element now
                # takes ANY pair whose two ends both carry ndf >= 3. It
                # acts on DOFs 1-3 and every DOF past the third — a u-p
                # node's pore pressure, a shell node's rotations — rides
                # as an untouched passenger. In 2-D, and below the
                # minimum build, the mismatch is still a warning plus an
                # inert element: the silent no-spring this guard exists
                # to catch.
                if int(ndm) == 3 and min(eff.values()) >= 3:
                    continue
                raise BridgeError(
                    f"{cls} element {eid} connects nodes with differing "
                    f"effective ndf {eff} — OpenSees requires equal ndf at "
                    f"both ends of a zeroLength-family element (it is "
                    f"silently dropped otherwise). Typically one end is an "
                    f"element-less / ground node taking the ops.model "
                    f"envelope ndf={int(envelope_ndf)} while the structural "
                    f"end infers a different value. Fix: set the model "
                    f"envelope to match the structural side, attach an "
                    f"element to the ground node, or use separate coincident "
                    f"nodes + g.constraints.equal_dof on the shared DOFs. "
                    f"(The one exemption is a 3-D pair whose ends BOTH "
                    f"carry ndf >= 3 — fork #808 / ADR 96, minimum build "
                    f"TIMS_FORK_BATCH_MIN_BUILD — which this pair is not.)"
                )


def assert_ndm_compatible(class_names: "Iterable[str]", ndm: int) -> None:
    """ADR 0048 ``ndm`` compatibility guard.

    ``ndm`` must lie in the intersection of every declared element's
    ``ndm_ok``. An empty intersection (a 2D ``quad`` declared alongside a 3D
    ``stdBrick``) raises :class:`BridgeError` — you cannot mix coordinate
    dimensions in one OpenSees domain. Unclassifiable element types are skipped
    (conservative). A non-empty intersection that excludes ``ndm`` also raises.
    """
    from ..._element_capabilities import element_class_ndm_ok

    inter: "set[int] | None" = None
    seen: list[str] = []
    for cls in class_names:
        ok = element_class_ndm_ok(cls)
        if ok is None:
            continue
        if inter is None:
            inter = set(ok)
        else:
            narrowed = inter & set(ok)
            if not narrowed:
                raise BridgeError(
                    f"element {cls!r} (ndm {sorted(ok)}) cannot share a model "
                    f"with the already-declared {seen!r} (common ndm "
                    f"{sorted(inter)}): you cannot mix 2D and 3D elements in "
                    f"one OpenSees domain."
                )
            inter = narrowed
        seen.append(cls)
    if inter is not None and ndm not in inter:
        raise BridgeError(
            f"ops.model(ndm={ndm}) is incompatible with the declared elements "
            f"{seen!r}, whose common ndm is {sorted(inter)}."
        )


def resolve_ndf_overlay(
    fem: "FEMData",
    ndf_records: "Iterable[NdfRecord]",
    inferred: "dict[int, int]",
    ndm: int,
) -> "dict[int, int]":
    """ADR 0049 — resolve ``ops.ndf`` directives into a ``{tag: ndf}`` overlay.

    Each :class:`NdfRecord` states the ndf of an element-LESS decoupled node
    (a spring/dashpot ground, control node, or mass anchor). The resolved id
    must be a decoupled node (``fem.nodes.decoupled_ids``) that NO element
    touches (absent from *inferred*). A target that is a mesh node, an
    element-touched node (already in *inferred*), or an unresolved handle
    (``.tag is None``) fails loud — ``ops.ndf`` is the sole explicit ndf
    channel and may only size a node inference cannot reach, preserving ADR
    0048's no-two-headed-model guarantee.

    Returns ``{tag: ndf}`` (empty when there are no records). The caller
    merges this over the inferred map (``{**inferred, **overlay}``) BEFORE the
    adaptive-endpoint gate (G1) and the node-emit fan-out run.
    """
    records = tuple(ndf_records)
    if not records:
        return {}

    nodes = getattr(fem, "nodes", None)
    decoupled: "set[int]" = set()
    if nodes is not None:
        try:
            decoupled = {int(t) for t in nodes.decoupled_ids}
        except Exception:
            decoupled = set()

    overlay: dict[int, int] = {}
    for rec in records:
        tag = rec.tag
        if tag is None and rec.handle is not None:
            tag = getattr(rec.handle, "tag", None)
        if tag is None:
            raise BridgeError(
                "ops.ndf: the target decoupled node has no resolved tag — its "
                "g.decouple_node handle was not materialized. Call "
                "g.mesh.queries.get_fem_data(...) so the FEM factory assigns "
                "the node its tag before constructing the bridge."
            )
        tag = int(tag)
        if tag in inferred:
            raise BridgeError(
                f"ops.ndf may only state the ndf of an element-LESS decoupled "
                f"node, but node {tag} is touched by an element (its ndf is "
                f"inferred as {inferred[tag]}). Drop ops.ndf for node {tag} — "
                f"the incident element class already determines its ndf "
                f"(ADR 0048 / 0049)."
            )
        if tag not in decoupled:
            raise BridgeError(
                f"ops.ndf target node {tag} is not a decoupled node. ops.ndf "
                f"is restricted to element-less nodes created via "
                f"g.decouple_node(...); a mesh node's ndf is inferred from its "
                f"incident elements and cannot be overridden (ADR 0048 / 0049)."
            )
        overlay[tag] = int(rec.ndf)
    return overlay


def validate_constraint_master_ndf(
    fem: "FEMData",
    effective: "dict[int, int]",
    ndm: int,
    envelope_ndf: int,
    stage_constraint_records: "Iterable[ConstraintRecord]" = (),
) -> None:
    """ADR 0049 G2 — fail loud when a constraint master / endpoint carries an
    ndf OpenSees silently rejects.

    A ``rigidDiaphragm`` retained node must be EXACTLY ndf 6 (3D) / 3 (2D)
    (``RigidDiaphragm.cpp:94-100`` warns-and-returns otherwise — the
    constraint silently vanishes). Every DOF index an ``equalDOF`` /
    ``rigidLink`` / ``kinematic_coupling`` references must be ``<=`` the ndf
    of BOTH endpoints. The ``∩ ndf_ok`` element gate
    (:func:`validate_node_ndf_element_compat`) is structurally blind to these
    — a diaphragm master or a decoupled constrained node is element-less — so
    G2 is the constraint-side half. Covers broker constraints AND
    stage-claimed constraints, which leave ``fem.*.constraints`` for
    :attr:`StageRecord.stage_constraint_records` (build.py:670).

    Operates over the resolved *effective* ndf map (inferred ∪ ``ops.ndf``
    overlay), falling absent nodes to the ``ops.model`` *envelope_ndf*.
    """
    from apeGmsh._kernel.records._kinds import ConstraintKind as _CK
    from apeGmsh._kernel.records._constraints import (
        NodeGroupRecord,
        NodePairRecord,
    )

    floor = 6 if int(ndm) == 3 else 3
    dof_selective = frozenset({
        _CK.EQUAL_DOF, _CK.KINEMATIC_COUPLING, _CK.RIGID_BEAM,
        _CK.RIGID_BEAM_STIFF, _CK.RIGID_ROD, _CK.RIGID_BODY,
    })

    def ndf_of(n: int) -> int:
        return int(effective.get(int(n), int(envelope_ndf)))

    # A1 — ``kinematic_coupling`` with ``dofs=None`` ties EVERY DOF the
    # slave has, by COUNT: ``LadrunoKinematicCoupling.cpp:260-275`` walks
    # c = 1..ndm+nrot and keeps each c the slave carries, so an ndf-4 u-p
    # slave in 3D has its pore pressure (slot 4) tied to the master's θx
    # by ``buildB`` (:335-350) — silently (the only warning sits behind
    # ``!useDefault``).  Only a pure translation (ndm) or translation +
    # rotation (ndm + nrot) layout is safe under the default; anything
    # else must name its ``dofs=``.  A 2D u-p node (ndf 3) is
    # indistinguishable BY COUNT from a (u, v, θ) node — this gate does
    # not see it.
    nrot = 3 if int(ndm) == 3 else 1
    rigid_layouts = frozenset({int(ndm), int(ndm) + nrot})

    def _check_default_coupling_slave(slave: int, name: object) -> None:
        k = ndf_of(slave)
        if k in rigid_layouts:
            return
        label = f" {name!r}" if name else ""
        raise BridgeError(
            f"kinematic_coupling{label}: slave node {slave} has ndf {k}, "
            f"which is neither a translation-only ({int(ndm)}) nor a "
            f"translation+rotation ({int(ndm) + nrot}) layout in "
            f"{int(ndm)}D — e.g. a u-p node whose DOF {int(ndm) + 1} is "
            f"pore pressure. With dofs=None the fork ties every DOF the "
            f"slave has BY COUNT (LadrunoKinematicCoupling.cpp:260-275), "
            f"so that DOF would be tied to the master's rotation. Pass "
            f"dofs= explicitly (e.g. dofs=[1, 2, 3] for translations only)."
        )

    def _check_diaphragm(master: int) -> None:
        if ndf_of(master) != floor:
            raise BridgeError(
                f"rigidDiaphragm master node {int(master)} has ndf "
                f"{ndf_of(master)}, but OpenSees requires EXACTLY ndf={floor} "
                f"for a {int(ndm)}D diaphragm retained node "
                f"(RigidDiaphragm.cpp:94-100 warns-and-returns otherwise, "
                f"silently dropping the diaphragm). Give the master ndf="
                f"{floor} via ops.ndf(master, {floor}) (decoupled master) or "
                f"by attaching elements that carry ndf={floor}."
            )

    def _check_pair(
        master: int, slave: int,
        dofs: "Iterable[int]", kind: str,
    ) -> None:
        for d in dofs:
            for endpoint in (int(master), int(slave)):
                if int(d) > ndf_of(endpoint):
                    raise BridgeError(
                        f"constraint {kind!r} references DOF {int(d)} on node "
                        f"{int(endpoint)} whose ndf is {ndf_of(endpoint)} "
                        f"(< {int(d)}). OpenSees cannot constrain a DOF the "
                        f"node does not carry. Raise the node's ndf "
                        f"(ops.ndf for a decoupled node, or the element / "
                        f"ops.model envelope ndf) or drop DOF {int(d)} from "
                        f"the constraint."
                    )

    def _walk(records: "Iterable[ConstraintRecord]") -> None:
        # Iterate RAW records with isinstance dispatch (NOT NodeGroupRecord.
        # expand_to_pairs / NodeConstraintSet.pairs, which crash on a
        # rigid_diaphragm whose ``dofs is None``). Per-DOF kinds carry their
        # own ``slave_nodes`` / ``slave_node``; surface / interpolation /
        # node-to-surface records have no scalar master and are skipped (the
        # ∩ element gate and phantom-node machinery already cover them).
        for rec in records:
            if isinstance(rec, NodeGroupRecord):
                if rec.kind == _CK.RIGID_DIAPHRAGM:
                    _check_diaphragm(int(rec.master_node))
                elif rec.kind in dof_selective and rec.dofs:
                    for slave in rec.slave_nodes:
                        _check_pair(
                            int(rec.master_node), int(slave), rec.dofs,
                            rec.kind,
                        )
                elif rec.kind == _CK.KINEMATIC_COUPLING:
                    # empty dofs ⇒ the element's count-based default (A1)
                    for slave in rec.slave_nodes:
                        _check_default_coupling_slave(
                            int(slave), getattr(rec, "name", None),
                        )
            elif isinstance(rec, NodePairRecord):
                if rec.kind in dof_selective and rec.dofs:
                    _check_pair(
                        int(rec.master_node), int(rec.slave_node), rec.dofs,
                        rec.kind,
                    )

    nodes = getattr(fem, "nodes", None)
    nc = getattr(nodes, "constraints", None) if nodes is not None else None
    if nc is not None:
        _walk(nc)  # NodeConstraintSet is iterable over its raw records.
    _walk(stage_constraint_records)


class DetachedDiaphragmMasterWarning(UserWarning):
    """A ``rigid_diaphragm`` master that no element touches has DOFs
    nothing stiffens (#1333).

    ``rigidDiaphragm`` ties only the in-plane DOFs of the slaves to the
    master (``ux, uy, rz`` for a horizontal floor).  A master standing
    alone at the floor's centre of mass - the documented pattern - then
    carries ``uz, rx, ry`` that no element, constraint or ``fix``
    stiffens.  The stiffness matrix is singular there: OpenSees prints
    "matrix singular", a static ``analyze`` can still return 0 with
    garbage displacements, and ``eigen`` reports modes with periods of
    ~1e4 s or ~1e-154 s.  The warning names the ``ops.fix`` mask that
    closes the hole; attaching the master to an element also silences it.
    """


#: 1-based DOFs ``rigidDiaphragm`` ties on a slave, per ``(ndm, perpDirn)``
#: (``RigidDiaphragm.cpp:142-212``; ``perpPlaneConstrained = perpDirn - 1``).
#: The master is the retained node, so these are the master DOFs the
#: slaves' stiffness reaches.
_DIAPHRAGM_TIED_DOFS: "dict[tuple[int, int], frozenset[int]]" = {
    (3, 3): frozenset({1, 2, 6}),
    (3, 2): frozenset({1, 3, 5}),
    (3, 1): frozenset({2, 3, 4}),
    (2, 3): frozenset({1, 2, 3}),
    (2, 2): frozenset({2}),
    (2, 1): frozenset({1}),
}


_SPATIAL_DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


def _single_node_pg(fem: "FEMData", node: int) -> "str | None":
    """Name of a physical group whose only node is ``node``, so a warning
    can suggest the ``ops.fix(pg=...)`` form the user wrote the model in;
    ``None`` when no such group exists (the ``nodes=`` form is always
    valid, so that is a result, not a skipped lookup).  Every dim is
    walked: only a point group can hold exactly one node, and the public
    :meth:`PhysicalGroupSet.names` is the one surface the FEM stub mirrors."""
    for name in fem.nodes.physical.names():
        ids = expand_pg_to_nodes(fem, name)
        if len(ids) == 1 and int(ids[0]) == int(node):
            return str(name)
    return None


def _constraint_record_node_tags(rec: object) -> "set[int]":
    """Every node tag a constraint record references, from its
    ADR 0038 ``tag_rewrite_spec`` cover set."""
    spec = type(rec).tag_rewrite_spec  # type: ignore[attr-defined]
    out: set[int] = set()
    for f in spec["tag_fields_scalar"]:
        v = getattr(rec, f)
        if v is not None:
            out.add(int(v))
    for f in spec["tag_fields_array"]:
        v = getattr(rec, f)
        if v is not None:
            out.update(int(n) for n in np.asarray(v).reshape(-1))
    return out


def validate_diaphragm_master_stiffness(
    fem: "FEMData",
    elements: "Iterable[Element]",
    ndm: int,
    envelope_ndf: int,
    effective_ndf: "Mapping[int, int]",
    *,
    fix_records: "Iterable[FixRecord | SupportRecord]" = (),
    sp_records: "Iterable[_SPRecord]" = (),
    stage_constraint_records: "Iterable[ConstraintRecord]" = (),
) -> None:
    """#1333 - warn when a ``rigid_diaphragm`` master that no element
    touches has DOFs nothing stiffens.

    For each diaphragm master absent from every declared element's
    connectivity, the DOFs ``1..ndf`` are reduced by what reaches them:
    the DOFs the diaphragm ties (:data:`_DIAPHRAGM_TIED_DOFS`, from the
    emitted ``perpDirn``, not the record's ``dofs``), the DOFs of any
    other node-pair / node-group constraint the master takes part in
    (``master_dofs`` when it is the retained node of an
    ``equal_dof_mixed``; every DOF when the record names none, which is
    the fork couplings' count-based default), and the DOFs a ``fix`` /
    ``s.support`` / pattern ``sp`` holds.  A master any other record kind
    references (a surface coupling, an embedment, a contact) is left
    alone: those records carry no per-DOF set the bridge can read, and a
    false warning on a legitimately coupled master would teach users to
    filter the category.  A master an element touches is out of scope
    whatever it stiffens - this gate is about the element-less master
    the ``rigid_diaphragm`` docstring recommends.

    One aggregated :class:`DetachedDiaphragmMasterWarning` names every
    detached master, its free DOFs and the ``ops.fix`` mask that closes
    them; a model whose masters are all attached or fully held stays
    silent.  Covers broker constraints and stage-claimed ones.
    """
    from apeGmsh._kernel.records._kinds import ConstraintKind as _CK
    from apeGmsh._kernel.records._constraints import (
        NodeGroupRecord,
        NodePairRecord,
    )

    nodes = getattr(fem, "nodes", None)
    nc = getattr(nodes, "constraints", None) if nodes is not None else None
    node_records: list[object] = [
        *(nc if nc is not None else ()), *stage_constraint_records,
    ]
    diaphragms = [
        r for r in node_records
        if isinstance(r, NodeGroupRecord) and r.kind == _CK.RIGID_DIAPHRAGM
    ]
    if not diaphragms:
        return
    masters = {int(r.master_node) for r in diaphragms}

    # Attached masters: any declared element's connectivity holds the tag.
    master_arr = np.fromiter(masters, dtype=np.int64)
    attached: set[int] = set()
    for spec in elements:
        fan = expand_spec_to_elements(fem, spec)
        if not fan:
            continue
        conn = fan.conn
        if conn.ndim == 2:
            hit = conn[np.isin(conn, master_arr)]
        else:
            hit = np.concatenate([
                np.asarray(row, dtype=np.int64) for row in conn
            ])
            hit = hit[np.isin(hit, master_arr)]
        attached.update(int(n) for n in hit)
        if attached >= masters:
            return
    detached = masters - attached

    def ndf_of(n: int) -> int:
        return int(effective_ndf.get(int(n), int(envelope_ndf)))

    held: dict[int, set[int]] = {m: set() for m in detached}

    # The diaphragms themselves - the DOFs OpenSees ties for the
    # emitted perpDirn.
    for rec in diaphragms:
        m = int(rec.master_node)
        if m in held:
            perp = _perp_dirn_from_normal(rec.plane_normal)
            held[m] |= _DIAPHRAGM_TIED_DOFS[(3 if int(ndm) == 3 else 2, perp)]

    # Every other node-side constraint the master takes part in.
    elements_c = getattr(fem, "elements", None)
    ec = (
        getattr(elements_c, "constraints", None)
        if elements_c is not None else None
    )
    diaphragm_ids = {id(r) for r in diaphragms}
    for rec in (*node_records, *(ec if ec is not None else ())):
        if id(rec) in diaphragm_ids:  # by identity: records hold arrays
            continue
        touched = _constraint_record_node_tags(rec) & detached
        if not touched:
            continue
        if isinstance(rec, (NodePairRecord, NodeGroupRecord)):
            dofs = {int(d) for d in rec.dofs} if rec.dofs else None
            for m in touched:
                if (
                    isinstance(rec, NodePairRecord)
                    and rec.master_dofs is not None
                    and int(rec.master_node) == m
                ):
                    held[m] |= {int(d) for d in rec.master_dofs}
                elif dofs is None:
                    held[m] |= set(range(1, ndf_of(m) + 1))
                else:
                    held[m] |= dofs
        else:
            # Surface coupling / embedment / contact: no per-DOF set the
            # bridge can read - assume the master is stiffened.
            for m in touched:
                held[m] |= set(range(1, ndf_of(m) + 1))

    # fix / support masks and pattern sps on the master.
    for frec in fix_records:
        for n in _record_node_ids(fem, frec):
            if int(n) in held:
                held[int(n)] |= {
                    d for d, flag in enumerate(frec.dofs, start=1) if flag
                }
    for srec in sp_records:
        targets = (
            expand_pg_to_nodes(fem, srec.target)
            if srec.target_kind == "pg" else (int(srec.target),)
        )
        for n in targets:
            if int(n) in held:
                held[int(n)].add(int(srec.dof))

    issues: list[str] = []
    for rec in diaphragms:
        m = int(rec.master_node)
        if m not in held:
            continue
        ndf = ndf_of(m)
        free = sorted(set(range(1, ndf + 1)) - held[m])
        if not free:
            continue
        del held[m]  # one entry per master, whatever the diaphragm count
        layout = _load_dof_layout(ndf, int(ndm))
        names = ", ".join(
            _SPATIAL_DOF_NAMES[layout[d - 1]] if d - 1 < len(layout)
            else f"DOF {d}"
            for d in free
        )
        mask = tuple(1 if d in free else 0 for d in range(1, ndf + 1))
        label = f" {rec.name!r}" if rec.name else ""
        pg = _single_node_pg(fem, m)
        target = f"pg={pg!r}" if pg is not None else f"nodes=({m},)"
        issues.append(
            f"rigid_diaphragm{label}: master node {m} is attached to no "
            f"element, and its DOFs {names} ({', '.join(map(str, free))}) "
            f"are stiffened by nothing - not the diaphragm, not another "
            f"constraint, not a fix. Fix them with ops.fix({target}, "
            f"dofs={mask}) or attach the master to an element"
        )
    if issues:
        warnings.warn(
            "; ".join(issues) + ". The stiffness matrix is singular on "
            "those DOFs: OpenSees prints 'matrix singular', a static "
            "analyze can still return 0 with garbage displacements, and "
            "eigen reports periods of ~1e4 s or ~1e-154 s (#1333).",
            DetachedDiaphragmMasterWarning,
            stacklevel=_stacklevel_outside_package(),
        )


def fit_dof_vector(
    values: "Iterable[float]",
    node_ndf: int,
    *,
    kind: str,
    node: int,
) -> tuple[float, ...]:
    """Fit a DOF-ordered ``load`` / ``mass`` vector to a node's ``ndf``.

    The per-node ``ndf`` is authoritative (ADR 0048); the user sets it via the
    declared elements / ``ops.ndf`` and we trust it. A vector SHORTER than the
    node ndf is zero-padded on the trailing DOFs (no force / mass on the higher
    DOFs). A vector LONGER than the node ndf is accepted only when the overflow
    is all-zero (it is trimmed); a **non-zero** overflow component addresses a
    DOF the node does not have and OpenSees ``Node::addUnbalancedLoad`` /
    ``setMass`` (``Node.cpp:940`` / ``:1272``) would drop the WHOLE vector — so
    it fails loud.
    """
    vals = [float(v) for v in values]
    if len(vals) > int(node_ndf):
        overflow = vals[int(node_ndf):]
        if any(v != 0.0 for v in overflow):
            lost = ", ".join(
                f"DOF {int(node_ndf) + i + 1}={v:g}"
                for i, v in enumerate(overflow)
                if v != 0.0
            )
            raise BridgeError(
                f"{kind} on node {node} has {len(vals)} components but the "
                f"node's ndf is {int(node_ndf)}; component(s) {lost} address "
                f"DOFs the node does not have. OpenSees "
                f"Node::addUnbalancedLoad / setMass (Node.cpp:940/1272) drop a "
                f"size-mismatched vector wholesale — the load / mass would be "
                f"silently lost. Drop the extra component(s) or raise the "
                f"node's ndf (ops.ndf for a decoupled node)."
            )
        vals = vals[:int(node_ndf)]
    return tuple(vals) + (0.0,) * (int(node_ndf) - len(vals))


def fit_fix_mask(
    dofs: "Iterable[int]",
    node_ndf: int,
) -> tuple[int, ...]:
    """Fit a ``fix`` DOF mask to a node's ``ndf`` by zero-padding.

    The ``fix`` arity rule is the OPPOSITE of the ``load`` / ``mass`` one,
    which is easy to get backwards. ``SP_Constraint.cpp:74``::

        if (vals.Size()-1 < ndf) { "invalid # of constraint values"; return -1; }
        for (int i = 0; i < ndf; i++) { ... }

    A mask SHORTER than the node's ndf is a hard error — OpenSees refuses
    the whole command, it does not fix the leading DOFs. A mask LONGER than
    ndf is silently truncated by the loop bound.

    So a short mask is padded here with ``0`` (= DOF left free), which is
    what "I did not mention DOF k" means; the caller has already rejected a
    too-long mask carrying real intent (see
    :func:`validate_record_ndf_consistency`), and any residual overflow is
    trimmed to match what OpenSees would do anyway.
    """
    mask = [int(d) for d in dofs]
    n = int(node_ndf)
    return tuple(mask[:n]) + (0,) * (n - len(mask))


def validate_record_ndf_consistency(
    fem: "FEMData",
    effective: "dict[int, int]",
    ndm: int,
    envelope_ndf: int,
    fix_records: "Iterable[FixRecord]" = (),
    mass_records: "Iterable[MassRecord]" = (),
    load_records: "Iterable[_LoadRecord]" = (),
    sp_records: "Iterable[_SPRecord]" = (),
    support_records: "Iterable[SupportRecord]" = (),
) -> None:
    """ADR 0049 G3 — fail loud when a ``fix`` / ``mass`` / ``load`` / ``sp``
    record addresses DOFs the node's effective ndf cannot carry.

    OpenSees ``Node::addUnbalancedLoad`` (``Node.cpp:940``) and
    ``Node::setMass`` (``Node.cpp:1272``) warn-and-return on a vector whose
    size ``!=`` the node's numDOF, silently dropping the WHOLE load / mass.
    The bridge therefore **fits** every mass / load vector to the node's ndf
    at emit (:func:`fit_dof_vector`): a short vector is zero-padded on the
    trailing DOFs, so the only unrecoverable case is a vector LONGER than the
    node ndf carrying a non-zero overflow component. Hence:

      * ``mass`` / nodal-``load`` vectors may be SHORTER than the node ndf
        (zero-padded), but a non-zero component beyond the node ndf raises;
      * a ``fix`` / ``support`` DOF-mask must not EXCEED the node ndf. A
        SHORT mask is fine here but is **not** something OpenSees accepts —
        ``SP_Constraint.cpp:74`` rejects the whole command when the mask is
        shorter than ndf — so emit pads it via :func:`fit_fix_mask`;
      * an ``sp`` DOF index (1-based) must be ``<=`` the node ndf.

    Operates over the *effective* ndf map (inferred ∪ ``ops.ndf`` overlay),
    falling absent nodes to the *envelope_ndf*.

    Known limitation: ``g.loads`` + ``p.from_model(case)`` loads are expanded
    into per-node lines inside the bridge at emit (``Plain.from_model_cases``,
    pattern.py:176-183) and are NOT :class:`_LoadRecord` instances — they are
    out of this guard's reach, but the emit-time mapping
    (:func:`broker_load_components` at the per-node ndf) fits + fail-loud-checks
    them directly.
    """
    def ndf_of(n: int) -> int:
        return int(effective.get(int(n), int(envelope_ndf)))

    def _pg_or_nodes(rec: "FixRecord | MassRecord | SupportRecord") -> "list[int]":
        if rec.nodes is not None:
            return [int(n) for n in rec.nodes]
        if rec.pg is not None:
            return list(expand_pg_to_nodes(fem, rec.pg))
        return []

    def _target_nodes(rec: "_LoadRecord | _SPRecord") -> "list[int]":
        # _LoadRecord / _SPRecord: target_kind ('pg' | 'node') + target (str).
        if rec.target_kind == "pg":
            return list(expand_pg_to_nodes(fem, rec.target))
        return [int(rec.target)]

    # mass — fit to node ndf; a non-zero overflow beyond ndf raises.
    for mrec in mass_records:
        for node in _pg_or_nodes(mrec):
            fit_dof_vector(
                mrec.values, ndf_of(node), kind="mass", node=node,
            )

    # nodal load — fit to node ndf; a non-zero overflow beyond ndf raises.
    for lrec in load_records:
        for node in _target_nodes(lrec):
            fit_dof_vector(
                lrec.forces, ndf_of(node), kind="nodal load", node=node,
            )

    # fix / support — DOF-mask must not exceed ndf.
    fix_like: "list[FixRecord | SupportRecord]" = [
        *fix_records, *support_records,
    ]
    for frec in fix_like:
        n_mask = len(frec.dofs)
        for node in _pg_or_nodes(frec):
            if n_mask > ndf_of(node):
                raise BridgeError(
                    f"fix / support on node {node} addresses {n_mask} DOFs but "
                    f"the node's ndf is only {ndf_of(node)}. The DOF mask "
                    f"cannot be longer than the node's ndf. Shorten the mask "
                    f"or raise the node's ndf (ops.ndf for a decoupled node)."
                )

    # sp — 1-based DOF index must be <= ndf.
    for srec in sp_records:
        d = int(srec.dof)
        for node in _target_nodes(srec):
            if d > ndf_of(node):
                raise BridgeError(
                    f"sp constraint on node {node} addresses DOF {d} but the "
                    f"node's ndf is only {ndf_of(node)}. The 1-based DOF index "
                    f"must be <= the node ndf. Lower the DOF or raise the "
                    f"node's ndf (ops.ndf for a decoupled node)."
                )


def node_coords_as_floats(
    coords: "Sequence[float]",
) -> "tuple[float, float, float]":
    """Coerce a stored ``(x, y, z)`` triple to plain Python floats.

    **Trimming to ``ndm`` is deliberately NOT done here.** That invariant —
    a 2-D ``node`` line carries exactly two coordinates, because a padded
    third desynchronises OpenSees' optional-argument scan and silently
    swallows the following ``-ndf K`` — is enforced once, at the emitter,
    by :func:`~apeGmsh.opensees.emitter.base.trim_coords_to_ndm` (ADR 0099).
    That trim runs last on every text and live emitter, so a second copy
    here bought nothing and gave one rule two enforcement points.

    What survives is the coercion, and it is load-bearing rather than
    cosmetic. The broker hands out ``numpy`` scalars, and
    :class:`~apeGmsh.opensees.emitter.tcl.TclEmitter` renders an unknown
    numeric via ``repr`` — under numpy 2.x ``repr(np.float64(0.0))`` is
    ``'np.float64(0.0)'``, so a stray numpy coordinate emits the literal
    text ``node 1 np.float64(0.0) np.float64(0.0)``. It also misses the
    emitter's plain-float fast path. Hence: plain floats, always, and the
    dimension question belongs to whoever writes the line.
    """
    return (float(coords[0]), float(coords[1]), float(coords[2]))


def _emit_node_with_inferred_ndf(
    emitter: "Emitter",
    inferred: "dict[int, int]",
    tag: int,
    coords: tuple[float, float, float],
    envelope_ndf: int,
) -> None:
    """Emit one ``node(tag, *coords)`` call with inference-sourced ndf
    (ADR 0048 — element-class inference is authoritative).

    The per-node ndf comes from the precomputed *inferred* map
    (:func:`infer_node_ndf`); nodes absent from it — element-less /
    decoupled nodes, and nodes touched only by adaptive elements —
    take *envelope_ndf* (the ``ops.model`` value). The ``-ndf K``
    token is **elided** when the resolved value equals the envelope,
    since the OpenSees ``model ... -ndf K`` directive already supplies
    it; this keeps homogeneous decks free of redundant ``-ndf`` lines.

    ``ndm`` selects how many coordinates go on the line — see
    :func:`node_coords_as_floats`.  It is required, not defaulted: a
    caller that forgot it would emit a 2-D deck whose ``-ndf`` tokens
    are all inert, which no test can see from the text.
    """
    cs = node_coords_as_floats(coords)
    ndf_val = inferred.get(int(tag), int(envelope_ndf))
    if ndf_val == int(envelope_ndf):
        emitter.node(int(tag), *cs)
    else:
        emitter.node(int(tag), *cs, ndf=int(ndf_val))



def emit_owned_nodes_with_inferred_ndf(
    emitter: "Emitter",
    inferred: "dict[int, int]",
    node_ids: "Iterable[int] | np.ndarray",
    node_idx_lookup: "SortedIntToInt",
    coords: "np.ndarray",
    envelope_ndf: int,
    exclude: "Sequence[Container[int]]" = (),
) -> None:
    """Emit one rank's owned ``node`` lines in ascending id order (#1670).

    The array form of the per-node loop
    ``for nid in sorted(node_ids): ... _emit_node_with_inferred_ndf(...)``:
    the ids are sorted once, resolved to coordinate rows with ONE
    vectorised :meth:`SortedIntToInt.get_many`, and the coordinates are
    read with one indexed gather and converted to plain Python floats
    by ``.tolist()``, the same values ``float(xyz[i])`` gives. Each node
    still goes through ``emitter.node`` one call at a time, so the
    emitter's ``ndm`` trim and its float fast path see exactly what they
    saw before, and the lines, their order and their bytes are unchanged.

    An id absent from ``node_idx_lookup`` is skipped, as before. An id
    in any set of ``exclude`` is skipped (stage-bound nodes, and nodes a
    hoisted block already declared). Duplicate ids emit once per
    occurrence, as ``sorted`` kept them.
    """
    ids = np.sort(np.asarray(node_ids, dtype=np.int64).reshape(-1))
    rows, found = node_idx_lookup.get_many(ids)
    ids_l = ids[found].tolist()
    xyz_l = np.asarray(coords[rows[found]], dtype=np.float64)[:, :3].tolist()
    env = int(envelope_ndf)
    node = emitter.node
    for nid, (x, y, z) in zip(ids_l, xyz_l):
        if exclude and any(nid in s for s in exclude):
            continue
        ndf_val = inferred.get(nid, env)
        if ndf_val == env:
            node(nid, x, y, z)
        else:
            node(nid, x, y, z, ndf=int(ndf_val))

def _load_dof_layout(ndf: int, ndm: int) -> tuple[int, ...]:
    """Spatial-component index placed at each DOF, for an ``(ndm, ndf)`` model.

    Indices address ``(Fx, Fy, Fz, Mx, My, Mz)``. ``ndf == 3`` is the
    ambiguous case: a 2D planar frame is ``(ux, uy, rz)`` → ``(Fx, Fy,
    Mz)``, but a 3D solid is ``(ux, uy, uz)`` → ``(Fx, Fy, Fz)`` — so
    ``ndm`` decides. Unrecognised ``ndf`` falls back to the leading
    ``ndf`` components (legacy behaviour).
    """
    if ndf == 1:
        return (0,)
    if ndf == 2:
        return (0, 1)
    if ndf == 3:
        return (0, 1, 5) if ndm == 2 else (0, 1, 2)
    if ndf == 6:
        return (0, 1, 2, 3, 4, 5)
    return tuple(range(min(ndf, 6)))


def _perp_dirn_from_normal(normal: object) -> int:
    """Map a diaphragm plane normal to OpenSees ``perpDirn`` (1|2|3).

    Mirrors the implementation in
    :mod:`apeGmsh._kernel.record_sets._perp_dirn` (we don't import
    that module directly to keep the bridge build pipeline independent
    of broker-private helpers).
    """
    if normal is None:
        return 3
    arr = np.abs(np.asarray(normal, dtype=float).reshape(-1))
    if arr.size < 3 or not np.any(np.isfinite(arr)) or not np.any(arr):
        return 3
    return int(np.argmax(arr[:3])) + 1
