"""loads: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
from typing import Iterable, Mapping

import numpy as np

from ...types import Element
from ..._stacklevel import _stacklevel_outside_package
from ..errors import BridgeError
from ..records import FixRecord, MassRecord, NdfRecord, SupportRecord
from ..pg_expand import _record_node_ids, expand_pg_to_nodes
from ..ndf import _load_dof_layout, infer_node_ndf, resolve_ndf_overlay
from .up import _iter_element_groups


if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


class WarnBodyForceDoubleCount(UserWarning):
    """A continuum element's ``body_force`` overlaps an imported gravity case.

    A continuum element's constructor ``body_force`` (``b1 b2 b3`` /
    ``-bodyForce``) is applied **unconditionally every step** — it is NOT
    pattern-gated (verified live + against ``Brick.cpp:1268-1274`` /
    ``FourNodeQuad.cpp:890-906``: the ``applyLoad == 0`` branch integrates the
    constructor ``b`` whether or not an ``eleLoad`` exists).  So if a
    ``p.from_model(case)`` also imports a geometry gravity case whose resolved
    nodal loads land on the same nodes **along the same axis**, that region
    carries its self-weight **twice**.  Fail-soft — both may be intentional,
    but it is almost always a mistake.

    Only a nodal load reduced from a body definition (``g.loads.gravity`` /
    ``g.loads.volume``) counts; a collinear line, surface, point or face
    load is a boundary load, not a second self-weight (#1338). A record
    whose source is unknown (a ``model.h5`` older than neutral 2.35.0)
    still counts, because it may be either.
    """


def validate_body_force_double_count(
    fem: "FEMData",
    elements: "Iterable[Element]",
    from_model_cases: "Iterable[str]",
) -> None:
    """ADR 0054 close-out — warn on silently double-counted self-weight.

    Detects the trap where a continuum element carries a constructor
    ``body_force`` (always-on, see :class:`WarnBodyForceDoubleCount`) **and**
    a ``p.from_model(case)`` import drives a gravity load onto the same nodes.
    To avoid false positives on the legitimate *lateral-load + self-weight*
    combo, the overlap only counts a node whose imported nodal load is
    **collinear** with the element's body force (i.e. the same line of action
    — double self-weight, not an orthogonal push).  Fail-soft (one aggregated
    warning).

    Collinearity alone cannot tell a reduced self-weight from a vertical
    footing line load (#1338), so the overlap also reads each record's
    ``source`` — the definition kind the resolver stamped
    (:class:`~apeGmsh._kernel.records._kinds.NodalLoadSource`). A body
    source (``gravity`` / ``body``) counts; a boundary source (``point``,
    ``line``, ``surface``, ``face_load``, ...) never does; ``None`` (a
    synthesized record, or a file older than neutral 2.35.0) counts because
    it may be either; any other value is outside the vocabulary and raises.

    LadrunoUP names its always-on solid self-weight ``body`` (an
    ACCELERATION, not a force density) rather than ``body_force``; the
    collinearity test compares directions only, so the unit difference is
    irrelevant and the guard reads either attribute.
    """
    from apeGmsh._kernel.records._kinds import NodalLoadSource
    # (pg, class_name, body_force_3d) — pg pulled via getattr so the loop
    # stays typed against the abstract ``Element`` (no ``.pg`` attribute).
    bf_specs: list[tuple[str, str, np.ndarray]] = []
    for spec in elements:
        # The always-on-self-weight vector: ``body_force`` on the standard
        # continuum elements, ``body`` on LadrunoUP (u-p accelerations).
        bf = getattr(spec, "body_force", None)
        if bf is None:
            bf = getattr(spec, "body", None)
        pg = getattr(spec, "pg", None)
        if bf is None or pg is None:
            continue
        vec = np.zeros(3, dtype=float)
        vec[: len(bf)] = [float(c) for c in bf]
        if float(np.linalg.norm(vec)) == 0.0:
            continue
        bf_specs.append((str(pg), type(spec).__name__, vec))
    cases = [c for c in dict.fromkeys(from_model_cases)]  # de-dup, keep order
    if not bf_specs or not cases:
        return

    nodes = getattr(fem, "nodes", None)
    load_set = getattr(nodes, "loads", None) if nodes is not None else None
    if load_set is None:
        return

    # case -> {node_id: (force_xyz, source)} — only loads carrying a real
    # force from a source that can be a self-weight: a body reduction, or
    # unknown.  A boundary-source record (#1338) is skipped here, and a
    # source outside the vocabulary fails closed.
    case_loads: dict[str, dict[int, tuple[np.ndarray, str | None]]] = {}
    for case in cases:
        per_node: dict[int, tuple[np.ndarray, str | None]] = {}
        for rec in load_set.by_pattern(case):
            f = getattr(rec, "force_xyz", None)
            if f is None:
                continue
            source = rec.source
            if source in NodalLoadSource.BOUNDARY_KINDS:
                continue
            if source is not None and source not in NodalLoadSource.BODY_KINDS:
                raise BridgeError(
                    f"nodal load on node {int(rec.node_id)} in case {case!r} "
                    f"carries source {source!r}, which is not a "
                    f"NodalLoadSource kind {sorted(NodalLoadSource.ALL)}; the "
                    f"double-count guard cannot classify it."
                )
            fv = np.asarray(f, dtype=float)
            if float(np.linalg.norm(fv)) == 0.0:
                continue
            per_node[int(rec.node_id)] = (fv, source)
        if per_node:
            case_loads[case] = per_node

    if not case_loads:
        return

    collisions: list[str] = []
    for pg, cls_name, vec in bf_specs:
        bf_dir = vec / np.linalg.norm(vec)
        spec_nodes = set(expand_pg_to_nodes(fem, pg))
        for case, per_node in case_loads.items():
            n_hit = 0
            hit_sources: set[str | None] = set()
            for nid in spec_nodes & per_node.keys():
                fv, source = per_node[nid]
                cos = float(abs(np.dot(fv / np.linalg.norm(fv), bf_dir)))
                if cos > 0.999:        # collinear -> same line of action
                    n_hit += 1
                    hit_sources.add(source)
            if n_hit:
                named = ", ".join(sorted(s for s in hit_sources if s is not None))
                if None in hit_sources:
                    named = (
                        (named + ", " if named else "")
                        + "unknown: a record without a definition kind, "
                        "such as a model.h5 older than neutral 2.35.0"
                    )
                collisions.append(
                    f"pg {pg!r} ({cls_name}, body_force) "
                    f"shares {n_hit} loaded node(s) with from_model case "
                    f"{case!r} (source {named})"
                )

    if collisions:
        joined = "; ".join(collisions)
        warnings.warn(
            f"self-weight may be double-counted: {joined}. A continuum "
            "element's body_force is applied every step regardless of any "
            "load pattern, so importing a gravity case onto the same region "
            "applies its weight twice. Drop one — either remove body_force= "
            "and keep the from_model gravity case (pattern-controlled, "
            "loadConst-freezable, the staged-SSI idiom), or remove the "
            "gravity case and keep body_force= (always-on, not rampable).",
            WarnBodyForceDoubleCount,
            stacklevel=2,
        )


class WarnLoadBasisMismatch(UserWarning):
    """An imported consistent load's basis conflicts with the element family.

    ADR 0091: a consistent reduction integrates the field against a
    shape-function basis chosen at *authoring* time
    (``g.loads.…(reduction="consistent", basis=…)``), while the element
    family is chosen later on the bridge.  A **Lagrange**-consistent
    vector applied to Bézier CONTROL values (BezierTet10 / BezierTri6 /
    the LadrunoUP tri6/tet10 Taylor–Hood variant) represents a strongly
    oscillatory traction — exact resultant, local spikes — which can
    drive near-surface Gauss points of a pressure-sensitive material
    into apex/tension (the TIMs T2 strip-footing divergence).  The
    reverse (a **Bernstein**-consistent vector on nodal-value elements)
    is the same distribution error mirrored.  Fail-soft: the resultant
    is exact either way and elastic answers stay plausible, so this is
    a warning, not a :class:`BridgeError`.
    """


#: Bridge element classes whose DOFs are Bernstein CONTROL values
#: (Ladruno Bézier family).  ``LadrunoUP`` is handled separately — it is
#: control-value only on its tri6/tet10 (Taylor–Hood) meshes.
_CONTROL_VALUE_ELEMENT_CLASSES: tuple[str, ...] = (
    "BezierTet10", "BezierTri6",
)


def validate_load_basis_vs_elements(
    fem: "FEMData",
    elements: "Iterable[Element]",
    from_model_cases: "Iterable[str]",
) -> None:
    """ADR 0091 — warn when an imported load's basis mismatches its target.

    Reads the ``basis`` tag consistent reductions stamp on their
    :class:`NodalLoadRecord`s and cross-references the node coverage of
    the deck's element declarations:

    * ``basis="lagrange"`` records on nodes owned **exclusively** by
      control-value elements → the T2 mechanism;
    * ``basis="bernstein"`` records on nodes owned **exclusively** by
      nodal-value elements → the mirrored mismatch.

    Exclusive ownership keeps interface nodes (shared between a Bézier
    and a Lagrange region) from producing false positives — a load
    resolved for one region legitimately touches the other's boundary
    nodes there.  Records with ``basis=None`` (point / tributary /
    resultant / gravity equal split) and nodes covered by no declared
    element are skipped.  One aggregated warning
    (:class:`WarnLoadBasisMismatch`).
    """
    cases = [c for c in dict.fromkeys(from_model_cases)]
    nodes = getattr(fem, "nodes", None)
    load_set = getattr(nodes, "loads", None) if nodes is not None else None
    if not cases or load_set is None:
        return

    # case -> [(node_id, basis)] for basis-tagged records only.
    tagged: dict[str, list[tuple[int, str]]] = {}
    for case in cases:
        recs = [
            (int(rec.node_id), str(basis))
            for rec in load_set.by_pattern(case)
            if (basis := getattr(rec, "basis", None)) is not None
        ]
        if recs:
            tagged[case] = recs
    if not tagged:
        return

    control_nodes: set[int] = set()
    nodal_nodes: set[int] = set()
    for spec in elements:
        pg = getattr(spec, "pg", None)
        if pg is None:
            continue
        cls_name = type(spec).__name__
        if cls_name == "LadrunoUP":
            # Bézier Taylor–Hood on tri6/tet10 meshes (quadratic
            # Bernstein u); every other LadrunoUP mesh is nodal-value.
            try:
                result = fem.elements.select(pg=str(pg)).groups()
            except Exception:
                continue
            for group in _iter_element_groups(result):
                conn = np.asarray(group.connectivity)
                ids = {int(n) for n in conn.reshape(-1)}
                width = conn.shape[1] if conn.ndim == 2 else 0
                (control_nodes if width in (6, 10) else nodal_nodes).update(ids)
            continue
        try:
            ids = set(expand_pg_to_nodes(fem, str(pg)))
        except BridgeError:
            continue
        if cls_name in _CONTROL_VALUE_ELEMENT_CLASSES:
            control_nodes.update(ids)
        else:
            nodal_nodes.update(ids)

    only_control = control_nodes - nodal_nodes
    only_nodal = nodal_nodes - control_nodes
    if not only_control and not only_nodal:
        return

    issues: list[str] = []
    for case, recs in tagged.items():
        lag_hits = {n for n, b in recs if b == "lagrange" and n in only_control}
        bern_hits = {n for n, b in recs if b == "bernstein" and n in only_nodal}
        if lag_hits:
            issues.append(
                f"case {case!r}: {len(lag_hits)} Lagrange-consistent loaded "
                f"node(s) on Bézier control-value elements — re-author with "
                f"basis='bernstein'"
            )
        if bern_hits:
            issues.append(
                f"case {case!r}: {len(bern_hits)} Bernstein-consistent loaded "
                f"node(s) on nodal-value elements — drop basis= (the "
                f"Lagrange default)"
            )

    if issues:
        joined = "; ".join(issues)
        warnings.warn(
            f"consistent-load basis mismatches the target element family: "
            f"{joined}. The resultant is exact either way, but the load's "
            f"local distribution is wrong for these DOFs — on Bézier "
            f"control values a Lagrange-consistent vector is a strongly "
            f"oscillatory traction that can drive near-surface Gauss "
            f"points of a pressure-sensitive material into apex/tension "
            f"(ADR 0091, TIMs T2).",
            WarnLoadBasisMismatch,
            stacklevel=2,
        )


def validate_from_model_cases(
    fem: "FEMData",
    from_model_cases: "Iterable[str]",
    allow_empty: "Iterable[str]" = (),
) -> None:
    """Fail loud when a ``from_model(case)`` import matches zero records.

    ``_emit_from_model_case`` expands a case into ``load`` lines (nodal
    loads tagged with the case) and ``sp`` lines (prescribed,
    non-homogeneous SPs tagged with it).  A case matching neither is a
    silent no-op at emit — historically a documented gap (ADR 0051),
    in practice always a typo or a case name lost to a pre-2.26.1
    ``model.h5`` whose SP writer flattened every case to ``default``.

    The check is **global** (whole-broker), deliberately not per-rank:
    a rank legitimately owning zero records of a case is normal in
    partitioned emit and must not trip this.

    Raises
    ------
    BridgeError
        For each imported case with zero importable records, unless the
        case was imported with ``from_model(case, allow_empty=True)``.
        The message distinguishes a case that exists but carries only
        homogeneous (hold) records — those are model-level by design
        and never importable — from a name with no records at all.
    """
    skip = set(allow_empty)
    cases = [c for c in dict.fromkeys(from_model_cases) if c not in skip]
    if not cases:
        return
    nodes = getattr(fem, "nodes", None)
    load_set = getattr(nodes, "loads", None) if nodes is not None else None
    sp_set = getattr(nodes, "sp", None) if nodes is not None else None

    load_patterns: set[str] = (
        set(load_set.patterns()) if load_set is not None else set()
    )
    sp_prescribed: set[str] = (
        {r.pattern for r in sp_set.prescribed()}
        if sp_set is not None else set()
    )
    sp_homogeneous_only: set[str] = (
        {r.pattern for r in sp_set.homogeneous()}
        if sp_set is not None else set()
    ) - sp_prescribed - load_patterns

    problems: list[str] = []
    for case in cases:
        if case in load_patterns or case in sp_prescribed:
            continue
        if case in sp_homogeneous_only:
            problems.append(
                f"case {case!r} exists but carries only homogeneous "
                f"(hold/fix) SP records, which from_model never imports "
                f"— holds are model-level; use ops.fix(...) instead"
            )
        else:
            problems.append(f"case {case!r} matches no record at all")
    if not problems:
        return

    available = sorted(load_patterns | sp_prescribed)
    hint = ""
    if (
        sp_set is not None
        and any(not r.is_homogeneous for r in sp_set.by_pattern("default"))
    ):
        hint = (
            "  Note: this broker carries prescribed SP records under "
            "'default' — a model.h5 saved before schema 2.26.1 flattened "
            "every displacement case name to 'default'; re-save the file "
            "from its source session to recover the case bindings."
        )
    raise BridgeError(
        f"from_model import(s) would silently apply nothing: "
        f"{'; '.join(problems)}.  Importable cases in this model: "
        f"{available or '(none)'}.  Pass "
        f"from_model(case, allow_empty=True) to permit a deliberately "
        f"empty import.{hint}"
    )


class UnconsumedModelDefinitionWarning(UserWarning):
    """The model defines supports or masses that the deck never emits.

    ``g.constraints.bc(...)`` (and a zero-valued ``g.displacements``)
    resolve into homogeneous SP records on ``fem.nodes.sp``;
    ``g.masses.*`` resolve into ``fem.nodes.masses``.  The bridge emits
    neither on its own (ADR 0051 §4): a deck carries them only when it
    restates them — ``ops.fix`` / ``s.fix`` / ``s.support`` /
    ``ops.fix_from_model()`` for the supports, ``ops.mass`` /
    ``s.mass`` / ``ops.mass_from_model()`` for the masses.  A deck that
    restates none of them runs unsupported and massless with no error.
    Filter this category for a deck that leaves them out on purpose.
    """


def _sp_deck_dof(spatial_dof: int, node_ndf: int, ndm: int) -> "int | None":
    """Deck DOF (1-based) carrying a broker SP's spatial component.

    Broker SP records index the spatial vector ``(ux, uy, uz, rx, ry,
    rz)`` (``g.constraints.bc``'s mask, ``g.displacements``' ``dofs``);
    a deck ``fix`` mask is positional over the node's DOFs.  The two
    agree only in 3-D; a 2-D frame (``ndm=2, ndf=3``) carries ``rz`` at
    DOF 3, so this maps through :func:`_load_dof_layout`, the layout the
    load path uses.  ``None`` when the node has no DOF for the component
    (``uz`` on any 2-D node, rotations on a solid).
    """
    layout = _load_dof_layout(int(node_ndf), int(ndm))
    idx = int(spatial_dof) - 1
    return layout.index(idx) + 1 if idx in layout else None


def validate_model_definition_consumed(
    fem: "FEMData",
    effective_ndf: "Mapping[int, int]",
    envelope_ndf: int,
    ndm: int,
    *,
    fix_records: "Iterable[FixRecord | SupportRecord]",
    mass_records: "Iterable[MassRecord]",
    mass_from_model: bool,
) -> None:
    """ADR 0051 §4 — warn when broker supports / masses miss the deck.

    A homogeneous SP record is consumed when a bridge fix-like record
    (``ops.fix`` / ``s.fix`` / ``s.support``, which is where
    ``ops.fix_from_model()`` lands too) flags the deck DOF that carries
    its spatial component on that node (:func:`_sp_deck_dof`, so ``uz``
    of a 2-D frame is not mistaken for its ``rz``).  A component the
    node has no DOF for is skipped: no deck can carry it, so nothing is
    dropped.  A mass record is consumed when a bridge ``mass`` record
    targets its node, or wholesale by ``mass_from_model``.  Prescribed
    (non-zero) SPs are load-case records that ``p.from_model(case)``
    imports, not model definition, and are out of scope.

    One aggregated :class:`UnconsumedModelDefinitionWarning` names the
    unconsumed counts and the verbs that restate them; a deck that
    restates every record stays silent.
    """
    nodes = getattr(fem, "nodes", None)
    sp_set = getattr(nodes, "sp", None) if nodes is not None else None
    mass_set = getattr(nodes, "masses", None) if nodes is not None else None
    issues: list[str] = []

    # (node, deck DOF, case) per homogeneous record the node can carry.
    homogeneous: list[tuple[int, int, str]] = []
    for r in sp_set if sp_set is not None else ():
        if not r.is_homogeneous:
            continue
        n = int(r.node_id)
        d = _sp_deck_dof(
            int(r.dof), int(effective_ndf.get(n, envelope_ndf)), ndm)
        if d is not None:
            homogeneous.append((n, d, str(r.pattern)))
    if homogeneous:
        fixed = {
            (int(n), d)
            for rec in fix_records
            for n in _record_node_ids(fem, rec)
            for d, flag in enumerate(rec.dofs, start=1)
            if flag
        }
        missed = [h for h in homogeneous if (h[0], h[1]) not in fixed]
        if missed:
            cases = sorted({c for _, _, c in missed if c != "default"})
            source = ", ".join(
                (["g.constraints.bc"]
                 if any(c == "default" for _, _, c in missed) else [])
                + ([f"zero-valued g.displacements holds in case(s) "
                    f"{', '.join(repr(c) for c in cases)}"] if cases else [])
            )
            issues.append(
                f"{len(missed)} homogeneous SP record(s) on "
                f"{len({n for n, _, _ in missed})} node(s) ({source}) have "
                f"no fix — restate them with ops.fix(pg=..., dofs=...) or "
                f"ops.fix_from_model()"
            )

    if mass_set is not None and len(mass_set) and not mass_from_model:
        ids = np.asarray(mass_set.node_ids(), dtype=np.int64)
        covered = np.fromiter(
            {int(n) for rec in mass_records for n in _record_node_ids(fem, rec)},
            dtype=np.int64,
        )
        n_missed = int(np.count_nonzero(~np.isin(ids, covered)))
        if n_missed:
            issues.append(
                f"{n_missed} nodal mass record(s) (g.masses) have no mass "
                f"— restate them with ops.mass_from_model() or "
                f"ops.mass(pg=..., values=...)"
            )

    if issues:
        warnings.warn(
            "the model defines supports / masses that this deck never "
            f"emits: {'; '.join(issues)}.  apeSees does not emit "
            "g.constraints.bc or g.masses on its own (ADR 0051 §4); "
            "without them the deck runs unsupported / massless, with no "
            "error.  If this deck leaves them out on purpose, filter "
            "UnconsumedModelDefinitionWarning.",
            UnconsumedModelDefinitionWarning,
            stacklevel=_stacklevel_outside_package(),
        )


def fix_records_from_model(
    fem: "FEMData",
    elements: "Iterable[Element]",
    ndm: int,
    envelope_ndf: int,
    *,
    ndf_records: "Iterable[NdfRecord]" = (),
    explicit: "Iterable[FixRecord | SupportRecord]" = (),
) -> tuple[FixRecord, ...]:
    """``ops.fix_from_model()`` — one ``fix`` per node from the snapshot.

    Folds every homogeneous SP record on ``fem.nodes.sp`` into one DOF
    mask per node, so two ``g.constraints.bc`` targets sharing a node
    give one ``fix`` line, not two (OpenSees refuses a second SP on a
    constrained DOF).  Nodes sharing a mask share one record.  Each
    record's spatial component lands on the deck DOF that carries it
    (:func:`_sp_deck_dof`, by ``ndm`` and the node's effective ndf:
    inferred from *elements*, plus the ``ops.ndf`` overlay, else the
    envelope), and a component the node lacks is left out.  So ``bc``'s
    default ``[1, 1, 1]`` (ux, uy, uz) pins x and y on a 2-D frame and
    leaves ``rz`` free, exactly what
    :func:`validate_model_definition_consumed` counts.  Prescribed
    records are not touched: ``p.from_model(case)`` imports those.

    Raises
    ------
    BridgeError
        If a stage's ``s.fix`` / ``s.support`` (``explicit``) already
        constrains a DOF the snapshot fixes: the stage is another tier,
        so the second SP on it would fail when the deck runs.  A global
        ``ops.fix`` never clashes: it shares this tier and merges with
        it (ADR 0115 D6/Q1).
    """
    sp_set = getattr(getattr(fem, "nodes", None), "sp", None)
    spatial_by_node: dict[int, set[int]] = {}
    for r in sp_set if sp_set is not None else ():
        if r.is_homogeneous:
            spatial_by_node.setdefault(int(r.node_id), set()).add(int(r.dof))
    if not spatial_by_node:
        return ()

    inferred = infer_node_ndf(fem, elements, ndm)
    node_ndf = {
        **inferred, **resolve_ndf_overlay(fem, ndf_records, inferred, ndm),
    }
    dofs_by_node: dict[int, set[int]] = {}
    for n, spatial in spatial_by_node.items():
        ndf_n = int(node_ndf.get(n, envelope_ndf))
        deck = {
            d for d in (_sp_deck_dof(s, ndf_n, ndm) for s in spatial)
            if d is not None
        }
        if deck:
            dofs_by_node[n] = deck

    clash = sorted({
        (int(n), d)
        for rec in explicit
        for n in _record_node_ids(fem, rec)
        for d, flag in enumerate(rec.dofs, start=1)
        if flag and d in dofs_by_node.get(int(n), ())
    })
    if clash:
        shown = ", ".join(f"node {n} DOF {d}" for n, d in clash[:5])
        more = f" (+{len(clash) - 5} more)" if len(clash) > 5 else ""
        raise BridgeError(
            f"fix_from_model() and a stage's s.fix / s.support both "
            f"constrain {shown}{more}. The stage is another tier, so "
            f"they do not merge the way a global ops.fix does (ADR 0115 "
            f"D6), and OpenSees refuses a second SP on a constrained DOF: "
            f"the deck would fail when it runs. Drop the stage's fix / "
            f"support on those DOFs, or drop fix_from_model() and restate "
            f"every support explicitly."
        )

    by_mask: dict[tuple[int, ...], list[int]] = {}
    for n in sorted(dofs_by_node):
        ds = dofs_by_node[n]
        mask = tuple(1 if d in ds else 0 for d in range(1, max(ds) + 1))
        by_mask.setdefault(mask, []).append(n)
    return tuple(
        FixRecord(pg=None, nodes=tuple(ns), dofs=mask)
        for mask, ns in by_mask.items()
    )
