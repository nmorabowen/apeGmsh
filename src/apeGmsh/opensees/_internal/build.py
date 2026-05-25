"""
Phase-4 build pipeline helpers.

The bridge's ``BuiltModel.emit`` orchestrates emission, but the
non-trivial work — dependency-sorted ordering, element fan-out across
physical groups, orientation-derived per-element ``vecxz`` fan-out,
and pattern / recorder ``pg=`` fan-out — lives here as pure helpers
so the orchestration in :mod:`apesees` stays small and readable.

The helpers in this module never import :mod:`openseespy`; they speak
only to the frozen :class:`Emitter` Protocol via the bridge-attached
tag resolver and element-nodes context (see
:mod:`apeGmsh.opensees._internal.tag_resolution`).

Three deferred contracts are resolved here:

  1. **Element fan-out** across a physical group's element ids and
     connectivity (the bridge writes the per-element node tags into the
     emitter's ``_current_element_nodes`` slot, allocates a per-element
     tag, and drives ``spec._emit`` once per element).

  2. **orientation-derived per-element vecxz fan-out** (ADR 0010):
     when a ``Linear`` / ``PDelta`` / ``Corotational`` GeomTransf is
     constructed with ``orientation=`` rather than an explicit
     ``vecxz=``, the bridge computes the local tangent for each
     element in the transform-bearing PGs, queries the orientation
     triad at the element midpoint, resolves the per-element
     ``vecxz`` via :func:`resolve_vecxz`, and emits one
     ``geomTransf`` line per distinct ``vecxz`` (within a ``1e-9``
     tolerance), reusing the same geomTransf tag for elements whose
     vecxz matches.

  3. **Pattern / recorder ``pg=`` fan-out** to per-node and per-element
     tags: the bridge resolves ``pg=`` records on :class:`Plain`
     patterns (loads + sps) and on Node / Element recorders into
     concrete tag lists before driving each primitive's ``_emit``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .._orientation import resolve_vecxz

from ..element.beam_column import (
    ElasticTimoshenkoBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from ..pattern.pattern import Plain, _LoadRecord, _SPRecord
from ..recorder import RecorderDeclaration, RecorderRecord
from ..transform import Corotational, Linear, PDelta
from .tag_allocator import TagAllocator
from .tag_resolution import (
    resolve_tag,
    set_current_fem_element_id,
    set_element_nodes,
    set_tag_resolver,
)
from .types import Element, GeomTransf, Primitive, Recorder

if TYPE_CHECKING:
    # Use the fully-qualified module path to disambiguate from the
    # similarly-named submodule ``apeGmsh.mesh.FEMData`` under mypy.
    from apeGmsh.mesh.FEMData import FEMData

    from ..emitter.base import Emitter


__all__ = [
    "BridgeError",
    "FixRecord",
    "InitialStressRecord",
    "MassRecord",
    "RegionAssignmentRecord",
    "StageRecord",
    "VECXZ_TOL",
    "compute_stage_ownership",
    "allocate_element_tags",
    "build_element_partition_owner",
    "build_node_partition_owners",
    "compute_vecxz_for_element",
    "emit_element_spec",
    "emit_element_spec_partitioned",
    "emit_initial_stress_addtoparameter",
    "emit_initial_stress_global",
    "resolve_initial_stress_elements",
    "emit_mp_constraints",
    "emit_mp_constraints_partitioned",
    "emit_pattern_spec",
    "emit_recorder_spec",
    "expand_pg_to_elements",
    "expand_pg_to_nodes",
    "is_orientation_transform",
    "is_partitioned",
    "topological_order",
]


#: Tolerance for considering two ``vecxz`` triples equal during the
#: orientation-derived fan-out's deduplication step. Two elements
#: whose per-element ``vecxz`` agrees to this tolerance share one
#: ``geomTransf`` line.
VECXZ_TOL: float = 1e-9


class BridgeError(RuntimeError):
    """Build-pipeline error — a primitive's dependency is unregistered,
    a PG is missing, or a fan-out cannot proceed for a structural
    reason. Distinct from :class:`ValueError` (caller error during
    primitive construction)."""


def validate_envelope_covers_broker_ndf(
    fem: "FEMData", envelope_ndf: int,
) -> None:
    """Raise :class:`BridgeError` when the envelope ``ndf`` is smaller
    than the largest per-node ``ndf`` declared on the broker.

    Runs at three sites per ADR 0033: :meth:`apeSees.model`,
    :meth:`OpenSeesModel.from_h5`, and
    :meth:`OpenSeesModel.from_compose_buffers`.  Sentinel zeros and
    a missing ``_ndf`` channel are no-ops — the check fires only
    when the user actually declared per-node ndf values.

    The message names the offending node + the fix (raise the
    envelope ``ndf`` in ``apeSees(fem).model(...)``) so the user
    doesn't have to grep the broker.
    """
    import numpy as np
    nodes = getattr(fem, "nodes", None)
    if nodes is None:
        return
    ndf_arr = getattr(nodes, "_ndf", None)
    if ndf_arr is None:
        return
    arr = np.asarray(ndf_arr, dtype=np.int8)
    if arr.size == 0:
        return
    max_declared = int(arr.max())
    if max_declared <= 0:
        return
    if int(envelope_ndf) >= max_declared:
        return
    # Find one offending node ID for the error message.
    idx = int(np.argmax(arr))
    try:
        nid = int(nodes.ids[idx])
    except Exception:
        nid = -1
    raise BridgeError(
        f"OpenSeesModel(ndf={int(envelope_ndf)}) cannot host node "
        f"nid={nid} declared with ndf={max_declared} via "
        f"g.node_ndf.set(...); raise ndf in "
        f"apeSees(fem).model(ndf=...) to at least {max_declared}."
    )


def _emit_node_with_broker_ndf(
    emitter: "Emitter",
    fem: "FEMData",
    tag: int,
    coords: tuple[float, float, float],
    *,
    envelope_ndf: int | None = None,
) -> None:
    """Emit one ``node(tag, *coords)`` call with broker-sourced ndf
    (S2 — ADR 0033 emit-side semantics).

    Looks up the per-node ndf via :meth:`fem.nodes.ndf_for` and
    passes ``ndf=K`` to the emitter only when a declaration covers
    the node.  On :class:`LookupError` (no declaration), emits
    without the override — the OpenSees envelope (``model ... -ndf
    K`` set via :meth:`apeSees.model`) supplies the value, matching
    the OpenSees-native semantics of ``model -ndf K`` + per-node
    ``-ndf J``.

    The optional ``envelope_ndf`` is consulted only when the
    fallback path itself wants to force a value (e.g. foreign-node
    declarations in the partitioned MP-constraint emit, where the
    bridge's model-wide envelope is the documented fallback even
    when no per-node declaration covers the foreign node).  Pass
    ``None`` for the common "envelope wins via the model directive"
    case at globally-owned and stage-owned emit sites.
    """
    x, y, z = coords
    try:
        ndf_val = fem.nodes.ndf_for(int(tag))
    except LookupError:
        if envelope_ndf is None:
            emitter.node(int(tag), float(x), float(y), float(z))
        else:
            emitter.node(
                int(tag), float(x), float(y), float(z),
                ndf=int(envelope_ndf),
            )
        return
    emitter.node(
        int(tag), float(x), float(y), float(z), ndf=int(ndf_val),
    )


# ---------------------------------------------------------------------------
# Model-level records collected on the bridge between build() calls.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FixRecord:
    """One ``fix`` directive registered through ``apeSees.fix``.

    Either ``pg`` or ``nodes`` is non-None (validated at the call site).
    The build pipeline expands ``pg`` into a per-node fan-out at emit
    time (one ``emitter.fix(node, *dofs)`` per node).
    """

    pg: str | None
    nodes: tuple[int, ...] | None
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MassRecord:
    """One ``mass`` directive registered through ``apeSees.mass``."""

    pg: str | None
    nodes: tuple[int, ...] | None
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegionAssignmentRecord:
    """One ``apeSees.region(name=...)`` directive — assigns nodes to a
    named OpenSees Region.

    Either ``pg`` or ``nodes`` is non-None (validated at the call site).
    Multiple records sharing the same ``name`` accumulate into one
    ``region $tag -node n1 n2 ...`` line at emit time: members merge by
    name, one tag is allocated per name, duplicates within a name are
    de-duped while preserving first-seen order.
    """

    name: str
    pg: str | None
    nodes: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One ``apeSees.stage(name)`` block — a per-stage analysis chain
    + per-stage ramped initial-stress records + the analyze run params.

    Used by Phase SSI-2.A staged-analysis decks.  Stages emit in the
    order they appear in ``BuiltModel.stage_records``; each one emits:

    1. ``stage_open(name)`` — comment delimiter for deck readability.
    2. Stage's :class:`InitialStressRecord` instances — parameter
       declarations + step_hook_ramp procs + addToParameter calls.
    3. The analysis chain (constraints / numberer / system / test /
       algorithm / integrator / analysis directive) — emitted via
       each primitive's ``_emit``.
    4. ``analyze`` loop — hook-wrapped if any initial_stress
       registered a ramp.
    5. ``stage_close()`` — emits ``loadConst -time 0.0`` +
       ``wipeAnalysis`` + clears the dispatcher lists.

    Analysis-chain primitives stay on the bridge's ``_primitives``
    list (they need tags via the topological-order pass); the stage
    record holds *references* to them, not copies.  ``BuiltModel.emit``
    skips the global pre-element emit of chain primitives when stages
    are declared, so each stage's chain is the only one OpenSees sees
    at run time.
    """

    name: str
    initial_stress_records: tuple["InitialStressRecord", ...]
    # Analysis chain references — primitives live on the bridge's
    # ``_primitives`` list; the stage knows which ones to bind for
    # this stage's analyze loop.  Stored as ``Primitive`` (generic)
    # to avoid pulling in the full analysis-chain type hierarchy
    # here; runtime types are enforced by the stage builder.
    test: "Primitive | None"
    algorithm: "Primitive | None"
    integrator: "Primitive | None"
    constraints: "Primitive | None"
    numberer: "Primitive | None"
    system: "Primitive | None"
    analysis: "Primitive | None"
    n_increments: int
    dt: float | None
    # Phase SSI-2.B: element-PG names that come online in this stage.
    # The bridge filters Element primitives whose ``pg=`` matches any
    # entry here into the stage's topology-emit block.  Nodes
    # referenced only by stage-bound elements emit alongside them.
    # Multiple stages can NOT share the same PG (first stage wins —
    # later activations of the same PG are validated as errors at
    # build time).  An element whose PG is not activated by any
    # stage stays global (emitted before stage 1).
    activated_pgs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InitialStressRecord:
    """One ``apeSees.initial_stress(name=...)`` directive — a ramped
    in-situ stress tensor applied to a set of ASDPlasticMaterial3D
    elements via the OpenSees ``parameter`` / ``addToParameter`` /
    ``updateParameter`` mechanism.

    Either ``pg`` or ``elements`` is non-None (validated at the call
    site).  Each record fans out into:

    * Three parameter declarations (XX, YY, ZZ) with bridge-allocated
      tags.
    * One ``addToParameter`` per element (per-rank-scoped in MP).
    * One :class:`StepHookRampRecord` registering the per-step ramp.

    ``lambda_install`` scales the target stress baked into the ramp's
    target_value (target_value = sigma * lambda_install); the per-step
    factor always ramps 0 → 1.0 over ``ramp_steps``.

    Parameters
    ----------
    name
        Unique label for this initial-stress region — used to name the
        Tcl proc / Python function and the per-hook state container.
        MUST be a valid Tcl identifier (alphanumeric + underscore, not
        starting with a digit).
    pg
        Physical group whose elements receive the ramped stress.
    elements
        Explicit list of element tags.  XOR with ``pg``.
    sigma_xx, sigma_yy, sigma_zz
        Target Cauchy stress per component (compression negative).
        Units must match the model's stress unit.
    ramp_steps
        Number of analyze steps over which the factor ramps 0 → 1.0.
        After ``ramp_steps`` analyze calls, the cumulative
        ``updateParameter`` is exactly ``sigma_* * lambda_install``;
        subsequent steps emit ``updateParameter $tag 0.0`` (no-op).
    lambda_install
        Fraction of target stress to install (0 < lambda <= 1).  1.0
        = full install (default).  0.5 = 50% relaxation (intermediate
        convergence-confinement step).
    """

    name: str
    pg: str | None
    elements: tuple[int, ...] | None
    sigma_xx: float
    sigma_yy: float
    sigma_zz: float
    ramp_steps: int
    lambda_install: float


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------

def topological_order(
    primitives: Iterable[Primitive],
) -> tuple[Primitive, ...]:
    """Return ``primitives`` sorted so each one's dependencies appear
    before it.

    Per Phase 4 Step 1b: the bridge must emit materials before sections,
    sections before elements, and time series before patterns. The
    topological sort traverses each primitive's :meth:`Primitive.dependencies`
    transitively, ordering parents before children.

    The returned tuple is stable: input ordering is preserved among
    primitives that have no mutual dependency (Kahn's algorithm with
    a per-primitive insertion-order queue).

    Per ADR P11 (Option A in the Phase-4 spec), this function does
    NOT auto-register reachable dependencies — the caller is responsible
    for ensuring every primitive returned by another's
    :meth:`dependencies` is itself registered. The caller (the bridge's
    build flow) checks the resulting tuple against its registered set
    and raises :class:`BridgeError` when it spots an unregistered
    dependency. This module only emits the order; it does not police
    registration.
    """
    # Kahn's algorithm with a stable order: ``order_seen`` records the
    # FIRST time each primitive id appears so the output preserves
    # input order among primitives with no mutual dependency. The
    # closure walk in ``_collect_reachable`` adds dependencies after
    # their dependents — Kahn's pass below reverses that to get a
    # parents-before-children sequence.
    seen: dict[int, Primitive] = {}
    for p in primitives:
        _collect_reachable(p, seen)

    # Build adjacency: for each primitive, list its dependencies.
    deps: dict[int, list[int]] = {
        i: [id(d) for d in p.dependencies()]
        for i, p in seen.items()
    }
    in_degree: dict[int, int] = {i: 0 for i in seen}
    for i, ds in deps.items():
        for d in ds:
            # d may not be in seen if a primitive returns dependencies()
            # that are not transitively reachable — should not happen,
            # but guard anyway.
            if d in in_degree:
                in_degree[i] += 1

    # The "depends-on" graph: edges go child -> parent (a child depends
    # on its parent). For Kahn's algorithm we want parents (in-degree
    # 0) emitted first. With the encoding above, a primitive's
    # in-degree counts how many parents it has; we emit in-degree-0
    # primitives, then "remove" their outbound edges (edges from
    # CHILDREN that point at them). To do that, build an inverted
    # index: parent_id -> list of child_ids that depend on it.
    children_of: dict[int, list[int]] = {i: [] for i in seen}
    for child_id, parent_ids in deps.items():
        for parent_id in parent_ids:
            if parent_id in children_of:
                children_of[parent_id].append(child_id)

    # Stable insertion order — iterate seen.values() in their first-
    # encounter order, but pick only primitives whose in-degree is 0.
    queue: list[int] = [i for i in seen if in_degree[i] == 0]
    out: list[Primitive] = []
    while queue:
        node = queue.pop(0)
        out.append(seen[node])
        for child in children_of[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(out) != len(seen):
        raise BridgeError(
            "topological_order: cycle detected in primitive dependency "
            f"graph. Reached {len(out)} of {len(seen)} primitives."
        )
    return tuple(out)


def _collect_reachable(
    p: Primitive, seen: dict[int, Primitive],
) -> None:
    """Walk ``p`` and its transitive dependencies, adding every
    primitive to ``seen`` keyed by ``id``. Insertion order is the
    walk order — this is what gives :func:`topological_order` its
    stability when no dependency relationship dictates otherwise."""
    pid = id(p)
    if pid in seen:
        return
    seen[pid] = p
    for d in p.dependencies():
        _collect_reachable(d, seen)


# ---------------------------------------------------------------------------
# orientation → per-element vecxz computation
# ---------------------------------------------------------------------------

_AnyTransf = Linear | PDelta | Corotational

_TRANSF_TYPE_TOKEN: dict[type[GeomTransf], str] = {
    Linear:       "Linear",
    PDelta:       "PDelta",
    Corotational: "Corotational",
}


def is_orientation_transform(t: Primitive) -> bool:
    """True if ``t`` is a Linear / PDelta / Corotational with an
    ``orientation=`` parameter set (and hence needs per-element
    vecxz fan-out at build time).

    A transform with explicit ``vecxz=`` does NOT need fan-out: it
    emits one ``geomTransf`` line with the spec's allocated tag. The
    bridge checks this flag to decide whether to drive the fan-out or
    let ``spec._emit`` run once with the spec's tag.
    """
    if not isinstance(t, (Linear, PDelta, Corotational)):
        return False
    return t.orientation is not None and t.vecxz is None


def compute_vecxz_for_element(
    transf: _AnyTransf,
    p_i: np.ndarray,
    p_j: np.ndarray,
) -> tuple[float, float, float]:
    """Return the per-element ``vecxz`` for one element under ``transf``.

    Reads the orientation triad at the element midpoint, computes the
    unit tangent from ``p_i``/``p_j``, and runs the orientation rule
    (ADR 0010) via :func:`resolve_vecxz`. ``transf`` MUST be an
    orientation-bearing transform (caller checks via
    :func:`is_orientation_transform` first).
    """
    if transf.orientation is None:
        raise BridgeError(
            f"compute_vecxz_for_element: transform {transf!r} has no "
            "orientation; caller should have used the explicit vecxz "
            "path."
        )
    p_i = np.asarray(p_i, dtype=float)
    p_j = np.asarray(p_j, dtype=float)
    edge = p_j - p_i
    norm = float(np.linalg.norm(edge))
    if norm <= 0.0:
        raise BridgeError(
            "compute_vecxz_for_element: element has zero-length edge; "
            f"p_i={p_i}, p_j={p_j}."
        )
    tangent = edge / norm
    midpoint = 0.5 * (p_i + p_j)
    e1, e2, e3 = transf.orientation.triad_at(midpoint)
    return resolve_vecxz(tangent, e1, e2, e3, transf.roll_deg)


def _vecxz_key(v: tuple[float, float, float]) -> tuple[int, int, int]:
    """Quantize a ``vecxz`` triple to integer cells of size :data:`VECXZ_TOL`
    so dict keys agree on the dedupe."""
    inv = 1.0 / VECXZ_TOL
    return (int(round(v[0] * inv)), int(round(v[1] * inv)), int(round(v[2] * inv)))


# ---------------------------------------------------------------------------
# Element fan-out across a physical group
# ---------------------------------------------------------------------------

def expand_pg_to_elements(
    fem: "FEMData", pg: str,
) -> list[tuple[int, tuple[int, ...]]]:
    """Return ``[(element_id, (node_ids, ...)), ...]`` for ``pg``.

    Order is deterministic: groups iterate in their FEM-snapshot order,
    and within each group elements iterate in id order. Empty PGs
    return an empty list (the caller decides whether that warrants a
    warning; per the Phase-4 spec, empty is permitted and emits
    nothing).

    Raises
    ------
    BridgeError
        If ``pg`` is not a known PG label / name on the FEM snapshot.
        The error includes the available PG names to help the user.
    """
    try:
        # selection-unification v2 P3-R / §6.3 §2 #5 (P-GROUPRESULT;
        # m3 — resolution raises at .select()).
        result = fem.elements.select(pg=pg).groups()
    except (KeyError, ValueError) as e:
        # FEMData raises one of these for an unknown PG; surface a
        # bridge-flavored error so the call-site can distinguish.
        available = _available_pg_names(fem)
        raise BridgeError(
            f"physical group {pg!r} not found in FEM snapshot. "
            f"Available element PGs: {sorted(available)}."
        ) from e
    out: list[tuple[int, tuple[int, ...]]] = []
    for group in result:
        for eid, conn_row in group:
            out.append((int(eid), tuple(int(n) for n in conn_row)))
    return out


def expand_pg_to_nodes(fem: "FEMData", pg: str) -> tuple[int, ...]:
    """Return the node ids for ``pg`` in deterministic order.

    Raises :class:`BridgeError` if ``pg`` is unknown.
    """
    try:
        # selection-unification v2 P3-R / §6.3 §2 #6 (P-NODE; m3).
        ids = fem.nodes.select(pg=pg).ids
    except (KeyError, ValueError) as e:
        available = _available_pg_names(fem)
        raise BridgeError(
            f"physical group {pg!r} not found in FEM snapshot. "
            f"Available PGs: {sorted(available)}."
        ) from e
    return tuple(int(n) for n in ids)


def _available_pg_names(fem: "FEMData") -> set[str]:
    """Best-effort enumeration of PG names known to the snapshot.

    Used in error messages — helps the user spot a typo without having
    to re-query the FEM. We probe both ``elements`` and ``nodes``
    composites since the user might have asked for a node PG via an
    element-fan-out call site (or vice versa).
    """
    out: set[str] = set()
    for composite_name in ("elements", "nodes"):
        composite = getattr(fem, composite_name, None)
        if composite is None:
            continue
        physical = getattr(composite, "physical", None)
        if physical is None:
            continue
        groups = getattr(physical, "_groups", None)
        if isinstance(groups, dict):
            for key in groups.keys():
                if isinstance(key, str):
                    out.add(key)
    return out


def emit_element_spec(
    spec: Element,
    emitter: "Emitter",
    fem: "FEMData",
    tags: TagAllocator,
    base_resolver: object,
    transf_tag_for_element: dict[tuple[int, int], int] | None = None,
    tag_recorder: dict[int, int] | None = None,
) -> None:
    """Drive the per-PG fan-out for one :class:`Element` typed spec.

    Parameters
    ----------
    spec
        The element typed primitive (carries ``pg=``).
    emitter
        Target emitter; the per-element node tags are pushed via
        :func:`set_element_nodes` before each ``spec._emit``.
    fem
        FEM snapshot the spec fans out over.
    tags
        Allocator — produces a fresh element tag for each fan-out instance.
    base_resolver
        The bridge's base tag resolver (callable). The fan-out installs
        an *element-specific* resolver on top of it when the element's
        transform requires orientation-driven per-element vecxz
        overrides.
    transf_tag_for_element
        Dict keyed ``(id(transf_spec), element_id)`` → per-element
        ``geomTransf`` tag. Filled by :func:`emit_transform_specs` for
        orientation-bearing transforms; ``None`` (or missing keys)
        means use the spec's own resolver path.
    tag_recorder
        Optional ``{fem_eid: ops_tag}`` dict the fan-out mutates as
        each element is emitted.  Used by downstream emit passes that
        need to look up the OpenSees element tag for a FEM element id
        (Phase SSI-1: initial_stress' addToParameter fan-out).
    """
    elements = expand_pg_to_elements(fem, spec.pg)  # type: ignore[attr-defined]
    if not elements:
        return

    transf_spec = _element_transf(spec)

    for eid, node_tags in elements:
        ele_tag = tags.allocate("element")
        if tag_recorder is not None:
            tag_recorder[int(eid)] = int(ele_tag)
        set_element_nodes(emitter, node_tags)
        # Phase 8.6: pass the FEM element id through the side channel
        # so the H5 emitter can record the (fem_eid, ops_tag) mapping
        # under /opensees/element_meta/{type_token}/fem_eids.
        set_current_fem_element_id(emitter, eid)

        if (
            transf_spec is not None
            and transf_tag_for_element is not None
            and (id(transf_spec), eid) in transf_tag_for_element
        ):
            override_tag = transf_tag_for_element[(id(transf_spec), eid)]

            # Wrap the base resolver so a lookup of the element's
            # transform spec returns its per-element override tag while
            # all other primitives resolve normally.
            base = base_resolver
            override = transf_spec

            def _resolver_with_override(
                p: Primitive,
                _base: object = base,
                _override_spec: Primitive = override,
                _override_tag: int = override_tag,
            ) -> int:
                if p is _override_spec:
                    return _override_tag
                # base is callable in practice (set by the bridge); cast
                # via a runtime call.
                return int(_base(p))  # type: ignore[operator]

            set_tag_resolver(emitter, _resolver_with_override)
            try:
                spec._emit(emitter, ele_tag)
            finally:
                # Restore the base resolver so subsequent primitives
                # see the unwrapped lookup.
                set_tag_resolver(emitter, base_resolver)  # type: ignore[arg-type]
        else:
            spec._emit(emitter, ele_tag)


def _element_transf(spec: Element) -> GeomTransf | None:
    """Return ``spec.transf`` for elements that compose a transform; else None.

    Truss / shell / solid elements have no transform — only beam-column
    family elements do. We isinstance-dispatch rather than reading
    ``getattr(spec, "transf", None)`` so we don't accidentally pick up
    an attribute on a future Element family that means something else.
    """
    if isinstance(
        spec,
        (
            elasticBeamColumn,
            forceBeamColumn,
            dispBeamColumn,
            ElasticTimoshenkoBeam,
        ),
    ):
        return spec.transf
    return None


# ---------------------------------------------------------------------------
# orientation-bearing transform fan-out — emit one geomTransf line per
# distinct vecxz observed across the elements that reference the spec.
# ---------------------------------------------------------------------------

def emit_transform_specs(
    transforms: Iterable[GeomTransf],
    elements: Iterable[Element],
    emitter: "Emitter",
    fem: "FEMData",
    tags: TagAllocator,
    spec_to_own_tag: dict[int, int],
    ndm: int = 3,
) -> dict[tuple[int, int], int]:
    """Emit ``geomTransf`` lines for every transform spec.

    For non-orientation transforms (explicit ``vecxz=``), one line
    per spec using the spec's own allocated tag — that's the path
    :class:`Linear` / :class:`PDelta` / :class:`Corotational` already
    handle in their ``_emit``. When ``ndm == 2`` and such a transform
    has neither ``vecxz`` nor ``orientation``, the bare 2-D form
    ``geomTransf <Type> $tag`` is emitted here instead (the primitive
    ``_emit`` requires a ``vecxz`` and doesn't know ``ndm``).

    For orientation-bearing transforms, the bridge:

      1. Walks every element spec whose ``transf`` IS this transform.
      2. For each element in the spec's PG, computes the per-element
         vecxz via :func:`compute_vecxz_for_element`.
      3. Deduplicates across all elements: distinct vecxz keys produce
         distinct ``geomTransf`` tags. The first encountered vecxz
         reuses the spec's own allocated tag (so the spec's tag is
         never wasted); subsequent distinct vecxz get freshly-allocated
         transform tags.
      4. Emits one ``geomTransf`` line per distinct ``vecxz``.
      5. Returns a per-element override map so the element fan-out can
         install element-specific resolvers for elements whose transform
         vecxz is not the spec's "own" tag.

    Returns
    -------
    dict[(id(transf_spec), element_id), int]
        Per-element override tags. Elements not in the dict use the
        spec's own resolver lookup (which yields the spec's own tag).
    """
    # Pre-bin elements by transform spec.
    elems_by_transf: dict[int, list[Element]] = {}
    for ele in elements:
        t = _element_transf(ele)
        if t is None:
            continue
        elems_by_transf.setdefault(id(t), []).append(ele)

    overrides: dict[tuple[int, int], int] = {}

    for transf in transforms:
        own_tag = spec_to_own_tag[id(transf)]
        if not is_orientation_transform(transf):
            # No orientation fan-out. Either an explicit vecxz= (3D —
            # one line, the spec's own _emit) or the bare 2D form
            # (``geomTransf <Type> $tag`` with no vecxz vector, which
            # is required in 2D and invalid in 3D). The primitive's
            # _emit can't take this branch because it doesn't know ndm.
            bare_2d = (
                ndm == 2
                and type(transf) in _TRANSF_TYPE_TOKEN
                and getattr(transf, "vecxz", None) is None
                and getattr(transf, "orientation", None) is None
            )
            if bare_2d:
                emitter.geomTransf(_TRANSF_TYPE_TOKEN[type(transf)], own_tag)
            else:
                transf._emit(emitter, own_tag)
            continue

        # Guard: orientation= is meaningless in OpenSees 2-D.  The
        # 2-D ``geomTransf <Type> $tag`` command takes no trailing
        # vecxz argument; if we proceed into the fan-out with ndm=2,
        # the emitter would silently produce the 3-D form (three
        # extra floats) which OpenSees rejects at parse time.
        # Refuse loudly with a clear message instead of producing
        # invalid output.  Lifting this restriction (supporting
        # in-plane Cylindrical(axis=(0,0,1)) etc.) is tracked in
        # ``architecture/_DEFERRED.md`` § "Cylindrical / Spherical
        # in 2-D models".
        if ndm == 2:
            raise BridgeError(
                f"geomTransf {type(transf).__name__}: orientation= is "
                "not supported with ndm=2 (OpenSees 2-D transforms "
                "take no vecxz argument). Drop the orientation= "
                "kwarg, or construct the bridge with "
                "``apeSees(fem, default_orientation=None)`` so the "
                "default Cartesian is not auto-applied.  See "
                "architecture/_DEFERRED.md § \"Cylindrical / Spherical "
                "in 2-D models\" for the planned lift."
            )

        # FEM-aware orientations (e.g. AlongBeam) declare a bind_fem
        # hook that materializes any FEM-derived state (reference-curve
        # segments, tangents) into the orientation instance before
        # per-element queries. The fixed-geometry orientations
        # (Cartesian, Cylindrical, Spherical) don't declare it — the
        # hasattr check makes the hook opt-in without polluting the
        # simple cases with a no-op method.
        # is_orientation_transform already narrowed `transf` to a
        # Linear / PDelta / Corotational that has a non-None
        # orientation, but the static GeomTransf base type doesn't
        # advertise the attribute.
        orient = transf.orientation  # type: ignore[attr-defined]
        if hasattr(orient, "bind_fem"):
            orient.bind_fem(fem)

        # orientation path: walk every element whose transf IS this
        # transform, compute per-element vecxz, dedupe.
        type_token = _TRANSF_TYPE_TOKEN[type(transf)]
        elems = elems_by_transf.get(id(transf), [])
        if not elems:
            # No elements reference this transform — emit nothing. The
            # spec is effectively a dead declaration; the user can
            # find this with introspection.
            continue

        # Gather per-element vecxz, keyed by element id.
        per_element_vecxz: list[tuple[int, tuple[float, float, float]]] = []
        for ele_spec in elems:
            for eid, node_ids in expand_pg_to_elements(fem, ele_spec.pg):  # type: ignore[attr-defined]
                if len(node_ids) != 2:
                    pg = ele_spec.pg  # type: ignore[attr-defined]
                    raise BridgeError(
                        f"orientation transform {type(transf).__name__}: "
                        f"element {eid} in PG {pg!r} has "
                        f"{len(node_ids)} nodes; orientation-driven "
                        "vecxz fan-out requires line elements (2 nodes)."
                    )
                p_i = _node_coord(fem, int(node_ids[0]))
                p_j = _node_coord(fem, int(node_ids[1]))
                vec = compute_vecxz_for_element(transf, p_i, p_j)  # type: ignore[arg-type]
                per_element_vecxz.append((eid, vec))

        # Dedupe by quantized key. First-seen vecxz reuses the spec's
        # own tag; later distinct vecxz claim fresh transform tags.
        key_to_tag: dict[tuple[int, int, int], int] = {}
        for eid, vec in per_element_vecxz:
            k = _vecxz_key(vec)
            if k not in key_to_tag:
                if not key_to_tag:
                    # First distinct vecxz — reuse the spec's own tag.
                    key_to_tag[k] = own_tag
                else:
                    key_to_tag[k] = tags.allocate("geomTransf")
                emitter.geomTransf(type_token, key_to_tag[k], *vec)

            assigned = key_to_tag[k]
            if assigned != own_tag:
                overrides[(id(transf), eid)] = assigned

    return overrides


def _node_coord(fem: "FEMData", node_id: int) -> np.ndarray:
    """Return the 3-D coordinates of ``node_id`` from the FEM snapshot."""
    idx = fem.nodes.index(node_id)
    return np.asarray(fem.nodes.coords[idx], dtype=float)


# ---------------------------------------------------------------------------
# Pattern fan-out — Plain pattern PG records resolved into per-node calls
# ---------------------------------------------------------------------------

def emit_pattern_spec(
    spec: Primitive,
    emitter: "Emitter",
    tag: int,
    fem: "FEMData",
) -> None:
    """Drive a pattern's emit, expanding any ``pg=`` records to per-node calls.

    For a :class:`Plain` pattern: emit ``pattern_open`` ourselves so we
    can intervene between it and ``pattern_close``, replacing the
    spec's :meth:`_emit` body. Records with ``target_kind == "node"``
    pass through unchanged; ``target_kind == "pg"`` records are
    fanned out into per-node ``emitter.load`` / ``emitter.sp`` calls.

    For non-Plain patterns (``UniformExcitation``, etc.) we delegate to
    the spec's own ``_emit`` since they have no PG-bearing records.
    """
    if not isinstance(spec, Plain):
        spec._emit(emitter, tag)
        return

    ts_tag = resolve_tag(emitter, spec.series)
    emitter.pattern_open("Plain", tag, ts_tag)
    for rec in spec.loads:
        _emit_load_record(rec, emitter, fem)
    for sp_rec in spec.sps:
        _emit_sp_record(sp_rec, emitter, fem)
    emitter.pattern_close()


def _emit_load_record(
    rec: _LoadRecord, emitter: "Emitter", fem: "FEMData",
) -> None:
    if rec.target_kind == "node":
        emitter.load(int(rec.target), *rec.forces)
        return
    # PG fan-out.
    for node_tag in expand_pg_to_nodes(fem, rec.target):
        emitter.load(node_tag, *rec.forces)


def _emit_sp_record(
    rec: _SPRecord, emitter: "Emitter", fem: "FEMData",
) -> None:
    if rec.target_kind == "node":
        emitter.sp(int(rec.target), rec.dof, rec.value)
        return
    for node_tag in expand_pg_to_nodes(fem, rec.target):
        emitter.sp(node_tag, rec.dof, rec.value)


# ---------------------------------------------------------------------------
# Recorder fan-out — Node / Element recorders with pg= resolved to ids
# ---------------------------------------------------------------------------

def emit_recorder_spec(
    spec: Recorder,
    emitter: "Emitter",
    tag: int,
    fem: "FEMData",
    *,
    tags: "TagAllocator | None" = None,
    fem_eid_to_ops_tag: "dict[int, int] | None" = None,
) -> None:
    """Drive a recorder's emit through its :meth:`Recorder.materialize`.

    :class:`RecorderDeclaration` follows a different shape (one
    declaration fans out to many ``recorder`` lines) and stays on its
    own branch.  Every other recorder routes through
    :meth:`Recorder.materialize` which resolves any ``pg=``-style
    selectors against ``fem``, emits any auxiliary declarations
    (e.g. MPCO's ``region`` line) on ``emitter`` directly, and
    returns a clone of itself with the build-time selectors cleared.
    The dispatcher then invokes ``_emit`` on the materialised spec.

    Recorders that carry no build-time selectors (e.g. a Node recorder
    constructed with explicit ``nodes=``) inherit the default
    no-op :meth:`Recorder.materialize` and pass through unchanged.

    ``fem_eid_to_ops_tag`` is the bridge-built ``{fem_eid: ops_tag}``
    map — element-targeting recorders (``Element``) use it to translate
    FEM eids resolved from ``pg=`` into the actual OpenSees element
    tags emitted by the element fan-out.  Without it (legacy direct
    callers), ``Element`` recorders fall back to FEM eids verbatim,
    which silently writes recorder lines that target the wrong
    elements whenever an element primitive consumed an allocator slot
    in ``_register`` — the bridge always supplies the map on the
    recorder emit pass.
    """
    if isinstance(spec, RecorderDeclaration):
        _emit_recorder_declaration(spec, emitter, fem)
        return
    materialised = spec.materialize(
        emitter, fem, tags, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
    )
    materialised._emit(emitter, tag)


# ---------------------------------------------------------------------------
# RecorderDeclaration emit fan-out (Phase 9 commit 3)
# ---------------------------------------------------------------------------

def _emit_recorder_declaration(
    decl: RecorderDeclaration,
    emitter: "Emitter",
    fem: "FEMData",
) -> None:
    """Walk a :class:`RecorderDeclaration` and emit one
    ``emitter.recorder(...)`` call per (ops_token, target_set) group.

    Per Phase 9 commit 3a, only the ``"nodes"`` category is handled
    end-to-end. Other categories raise :class:`NotImplementedError`
    pointing at the follow-up commits.
    """
    from .._recorder_translate import (
        element_record_response_tokens,
        group_node_components_by_ops_token,
    )

    for record in decl.records:
        if record.category in ("fibers", "layers", "modal"):
            raise NotImplementedError(
                f"RecorderDeclaration record(category={record.category!r}) "
                "is not file-emit-able; use DomainCapture instead (Phase 9 "
                "commit 5 provides the bridge-friendly entry point)."
            )
        if record.category not in (
            "nodes", "elements", "line_stations", "gauss",
        ):  # pragma: no cover  - guarded by RecorderRecord validation
            raise ValueError(
                f"unrecognized category {record.category!r} on "
                f"RecorderDeclaration record"
            )

        # Schema 2.3.0: bracket every fan-out with declaration
        # metadata so the H5 emitter can archive the original intent
        # alongside each emitted ``recorder(...)`` call. Lean emitters
        # (Tcl / Py / Live / Recording) treat these calls as no-ops.
        emitter.recorder_declaration_begin(
            declaration_name=decl.name,
            record_name=record.name,
            category=record.category,
            components=record.components,
            raw=record.raw,
            pg=record.pg,
            label=record.label,
            selection=record.selection,
            ids=record.ids,
            dt=record.dt,
            n_steps=record.n_steps,
            file_root=decl.file_root,
        )
        try:
            if record.category == "nodes":
                _emit_nodes_record(record, decl, emitter, fem,
                                   group_node_components_by_ops_token)
            else:
                _emit_element_level_record(
                    record, decl, emitter, fem,
                    element_record_response_tokens,
                )
        finally:
            emitter.recorder_declaration_end()


def _emit_nodes_record(
    record: RecorderRecord,
    decl: RecorderDeclaration,
    emitter: "Emitter",
    fem: "FEMData",
    group: object,  # callable; passed in to keep imports local to caller
) -> None:
    """Emit one node-level :class:`RecorderRecord`.

    Resolves selectors (``pg``, ``label``, ``selection``, ``ids``) to
    a flat node-tag tuple, groups canonical components by their
    OpenSees recorder token, and emits one ``recorder Node`` call per
    canonical (``ops_token``, ``target_set``) group plus one extra
    ``recorder Node`` per ``raw=`` token (with dofs defaulting to all
    DOFs from ``decl.ndf``).

    File path convention:
      * canonical: ``<file_root>/<decl.name>__<record_name>__<token>.out``
      * raw: ``<file_root>/<decl.name>__<record_name>__raw_<token>.out``
    """
    node_ids = _resolve_node_targets(record, fem)
    if not node_ids:
        return  # nothing to record — silent skip mirrors typed-primitive behavior

    # Group canonical components by ops token (e.g. "disp": (1, 2)).
    # Caller passes the translator to keep its import scoped to the
    # _emit_recorder_declaration function.
    grouped = group(record.components) if record.components else {}  # type: ignore[operator]
    record_name = record.name or "default"

    for ops_token, dofs in grouped.items():
        file_path = _recorder_file_path(
            decl.file_root, decl.name, record_name, ops_token,
        )
        args: list[int | float | str] = ["-file", file_path]
        if record.dt is not None:
            args += ["-dT", record.dt]
        # Default time_format is "dt" for declared records — they're
        # broader-vocabulary and time-aware consumers (results) expect
        # the leading time column.
        args += ["-time"]
        args += ["-node", *node_ids]
        args += ["-dof", *dofs]
        args.append(ops_token)
        emitter.recorder("Node", *args)

    # Raw escape hatch: one extra recorder Node per raw token. Dofs
    # default to all DOFs (1..ndf) since raw tokens bypass the
    # canonical→dof translation.
    if record.raw:
        all_dofs = tuple(range(1, decl.ndf + 1))
        for raw_token in record.raw:
            file_path = _recorder_file_path(
                decl.file_root, decl.name, record_name,
                f"raw_{_sanitize_raw_token(raw_token)}",
            )
            args = ["-file", file_path]
            if record.dt is not None:
                args += ["-dT", record.dt]
            args += ["-time"]
            args += ["-node", *node_ids]
            args += ["-dof", *all_dofs]
            args.append(raw_token)
            emitter.recorder("Node", *args)


def _resolve_node_targets(
    record: RecorderRecord, fem: "FEMData",
) -> tuple[int, ...]:
    """Resolve a node-category :class:`RecorderRecord`'s selectors to
    a flat tuple of node tags.

    Supports ``ids=`` (mutex with named selectors) and the named
    selectors ``pg=`` / ``label=`` / ``selection=`` (composable — the
    resulting target sets are unioned and deduplicated, mirroring the
    legacy ``Recorders`` helper semantics).
    """
    if record.ids is not None:
        return tuple(int(i) for i in record.ids)

    chunks: list[Iterable[int]] = []
    for pg_name in record.pg:
        chunks.append(expand_pg_to_nodes(fem, pg_name))
    for label_name in record.label:
        chunks.append(_expand_label_to_nodes(fem, label_name))
    for sel_name in record.selection:
        chunks.append(_expand_selection_to_nodes(fem, sel_name))

    if not chunks:
        return ()
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        for tag in chunk:
            t = int(tag)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return tuple(out)


def _expand_label_to_nodes(fem: "FEMData", label_name: str) -> tuple[int, ...]:
    """Return node IDs registered under ``label_name`` on ``fem.nodes.labels``.

    Raises :class:`BridgeError` if the FEM snapshot exposes no labels
    accessor (older fixtures) or the label name is unknown.
    """
    nodes_obj = getattr(fem, "nodes", None)
    labels = getattr(nodes_obj, "labels", None) if nodes_obj is not None else None
    if labels is None:
        raise BridgeError(
            f"label {label_name!r} requested but FEM snapshot has no "
            f"nodes.labels accessor."
        )
    try:
        ids = labels.node_ids(label_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"node label {label_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(n) for n in ids)


def _expand_selection_to_nodes(fem: "FEMData", sel_name: str) -> tuple[int, ...]:
    """Return node IDs registered under ``sel_name`` on ``fem.mesh_selection``.

    Raises :class:`BridgeError` if the FEM snapshot has no
    ``mesh_selection`` store or the selection name is unknown.
    """
    store = getattr(fem, "mesh_selection", None)
    if store is None:
        raise BridgeError(
            f"selection {sel_name!r} requested but FEM snapshot has no "
            f"mesh_selection store (no post-mesh selections were declared "
            f"on the session)."
        )
    try:
        ids = store.node_ids(sel_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"node selection {sel_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(n) for n in ids)


# ---------------------------------------------------------------------------
# Element-level emit (Phase 9 commit 3b)
# ---------------------------------------------------------------------------


def _emit_element_level_record(
    record: RecorderRecord,
    decl: RecorderDeclaration,
    emitter: "Emitter",
    fem: "FEMData",
    response_tokens: object,  # callable; passed in to keep imports local
) -> None:
    """Emit one element-level :class:`RecorderRecord` (elements / gauss /
    line_stations).

    Resolves selectors to element IDs, picks the OpenSees response
    phrase based on the record's components (via the catalog-driven
    ``element_record_response_tokens`` helper), and issues one
    ``emitter.recorder("Element", ...)`` call for the canonical group
    plus one per ``raw=`` token.

    For ``line_stations`` records, also emits a paired
    ``integrationPoints`` recorder writing to ``<file>_gpx.out`` —
    consumed by the .out transcoder when reading the line-station
    results back into a :class:`LineStationSlab`.
    """
    elem_ids = _resolve_element_targets(record, fem)
    if not elem_ids:
        return

    record_name = record.name or "default"
    emitted_canonical = False

    if record.components:
        tokens = response_tokens(  # type: ignore[operator]
            record.category, record.components, record_name=record.name,
        )
        if tokens is not None:
            file_path = _recorder_file_path(
                decl.file_root, decl.name, record_name, record.category,
            )
            args: list[int | float | str] = ["-file", file_path]
            if record.dt is not None:
                args += ["-dT", record.dt]
            args += ["-time"]
            args += ["-ele", *elem_ids]
            args += list(tokens)
            emitter.recorder("Element", *args)
            emitted_canonical = True

    # Raw escape hatch: one extra recorder Element per raw token.
    if record.raw:
        for raw_token in record.raw:
            file_path = _recorder_file_path(
                decl.file_root, decl.name, record_name,
                f"raw_{_sanitize_raw_token(raw_token)}",
            )
            args = ["-file", file_path]
            if record.dt is not None:
                args += ["-dT", record.dt]
            args += ["-time"]
            args += ["-ele", *elem_ids]
            args.append(raw_token)
            emitter.recorder("Element", *args)

    # line_stations IP pairing: the .out transcoder needs per-element
    # integration-point positions to map the section.force samples back
    # to physical xi*L coordinates. Emit one gpx file per record (shared
    # across canonical + raw tokens — the GP geometry is independent of
    # the response token).
    if record.category == "line_stations" and (
        emitted_canonical or record.raw
    ):
        canonical_path = _recorder_file_path(
            decl.file_root, decl.name, record_name, record.category,
        )
        gpx_path = _line_station_gpx_path(canonical_path)
        args = ["-file", gpx_path]
        if record.dt is not None:
            args += ["-dT", record.dt]
        args += ["-time"]
        args += ["-ele", *elem_ids]
        args.append("integrationPoints")
        emitter.recorder("Element", *args)


def _resolve_element_targets(
    record: RecorderRecord, fem: "FEMData",
) -> tuple[int, ...]:
    """Resolve an element-level :class:`RecorderRecord`'s selectors to
    a flat tuple of element tags.

    Supports ``ids=`` (mutex with named selectors) and the named
    selectors ``pg=`` / ``label=`` / ``selection=`` (composable — same
    union/dedup contract as :func:`_resolve_node_targets`).
    """
    if record.ids is not None:
        return tuple(int(i) for i in record.ids)

    chunks: list[Iterable[int]] = []
    for pg_name in record.pg:
        chunks.append(eid for eid, _conn in expand_pg_to_elements(fem, pg_name))
    for label_name in record.label:
        chunks.append(_expand_label_to_elements(fem, label_name))
    for sel_name in record.selection:
        chunks.append(_expand_selection_to_elements(fem, sel_name))

    if not chunks:
        return ()
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        for tag in chunk:
            t = int(tag)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return tuple(out)


def _expand_label_to_elements(
    fem: "FEMData", label_name: str,
) -> tuple[int, ...]:
    """Return element IDs registered under ``label_name`` on
    ``fem.elements.labels``."""
    elements_obj = getattr(fem, "elements", None)
    labels = (
        getattr(elements_obj, "labels", None)
        if elements_obj is not None
        else None
    )
    if labels is None:
        raise BridgeError(
            f"label {label_name!r} requested but FEM snapshot has no "
            f"elements.labels accessor."
        )
    try:
        ids = labels.element_ids(label_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"element label {label_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(e) for e in ids)


def _expand_selection_to_elements(
    fem: "FEMData", sel_name: str,
) -> tuple[int, ...]:
    """Return element IDs registered under ``sel_name`` on
    ``fem.mesh_selection``."""
    store = getattr(fem, "mesh_selection", None)
    if store is None:
        raise BridgeError(
            f"selection {sel_name!r} requested but FEM snapshot has no "
            f"mesh_selection store (no post-mesh selections were declared "
            f"on the session)."
        )
    try:
        ids = store.element_ids(sel_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"element selection {sel_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(e) for e in ids)


# ---------------------------------------------------------------------------
# File-path helpers for RecorderDeclaration emit
# ---------------------------------------------------------------------------

def _recorder_file_path(file_root: str, *parts: str) -> str:
    """Build a recorder ``.out`` path from ``file_root`` and a sequence
    of basename parts joined by ``__``.

    Mirrors the legacy ``_build_file_path`` convention in
    ``apeGmsh.results.spec._emit`` — handles empty ``file_root`` (no
    prefix) and trailing slashes on the directory portion.
    """
    fname = "__".join(parts) + ".out"
    if not file_root:
        return fname
    sep = "" if file_root.endswith(("/", "\\")) else "/"
    return f"{file_root}{sep}{fname}"


def _line_station_gpx_path(line_station_file_path: str) -> str:
    """Return the paired ``integrationPoints`` recorder path for a
    line-stations file.

    Replaces the ``.out`` suffix with ``_gpx.out``. Matches the
    convention defined in ``apeGmsh.results.spec._emit.line_station_gpx_path``
    so the legacy .out transcoder locates the paired file unchanged.
    """
    if line_station_file_path.endswith(".out"):
        return line_station_file_path[:-4] + "_gpx.out"
    return line_station_file_path + "_gpx"


def _sanitize_raw_token(token: str) -> str:
    """Return a filename-safe form of ``token`` for raw= file paths.

    Replaces any character that isn't alphanumeric or underscore with
    ``_``. Raw tokens are user-supplied OpenSees response strings that
    may contain spaces, hyphens, or other shell-sensitive chars; the
    sanitized form is used only in the output ``.out`` filename — the
    raw token reaches OpenSees verbatim as the recorder response.
    """
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in token)


# ---------------------------------------------------------------------------
# MP constraint fan-out (Phase 7b, ADR 0022) — closes the §3.3 deferral.
# ---------------------------------------------------------------------------


def emit_mp_constraints(
    emitter: "Emitter", fem: "FEMData",
) -> None:
    """Fan out the broker's MP-constraint records onto ``emitter``.

    Per ADR 0022 INV-5 this pass runs between element emission and
    pattern emission in :meth:`BuiltModel.emit`.

    Ordering (INV-3 / dependency-driven):

    1. **Phantom-node pre-step** — :class:`NodeToSurfaceRecord` rows
       carry synthetic 6-DOF phantom nodes whose tags must exist in
       the OpenSees domain before any constraint references them.
       Emitted via ``emitter.node(tag, *xyz, ndf=6)`` — the standard
       OpenSees per-node ``-ndf`` override pattern.  Tags are
       de-duplicated across records (paranoid; the resolver does not
       collide, but the cost of the set check is negligible).

    2. **Rigid links** — :meth:`fem.nodes.constraints.rigid_link_groups`
       yields ``(master, slaves)`` tuples covering ``rigid_beam`` /
       ``rigid_rod`` :class:`NodePairRecord` rows, the ``rigid_body``
       slaves on :class:`NodeGroupRecord`, and the phantom-side
       rigid-link rows on :class:`NodeToSurfaceRecord`.  Emitted as
       one ``emitter.rigidLink('beam'|'bar', master, slave)`` per
       slave; ``rigid_rod`` maps to ``"bar"`` per the OpenSees
       vocabulary.

    3. **Equal DOFs** — :meth:`fem.nodes.constraints.equal_dofs`
       yields :class:`NodePairRecord` rows for ``equal_dof`` plus the
       phantom→slave equal-DOF rows nested under
       :class:`NodeToSurfaceRecord`.  Emitted as one
       ``emitter.equalDOF(master, slave, *dofs)`` per record.

    4. **Rigid diaphragms** — :meth:`fem.nodes.constraints.rigid_diaphragms`
       yields ``(perp_dir, master, slaves)`` for
       :class:`NodeGroupRecord` rows with kind ``rigid_diaphragm``.
       Emitted as one ``emitter.rigidDiaphragm(perp_dir, master,
       *slaves)`` per record.

    5. **Kinematic couplings** — :class:`NodeGroupRecord` rows with
       kind ``kinematic_coupling`` are emitted as one ``equalDOF``
       per ``(master, slave)`` pair (the per-DOF selectivity makes
       ``rigidLink`` / ``rigidDiaphragm`` wrong for this family —
       see the docstring on :meth:`rigid_link_groups`).

    6. **Surface couplings** — :meth:`fem.elements.constraints.interpolations`
       yields :class:`InterpolationRecord` rows (one slave node ↔ N
       weighted master nodes from a master element face).  Emitted as
       one ``emitter.embeddedNode(ele_tag, cnode, *args)``
       per record using a freshly allocated element tag.  Covers
       ``tie`` / ``distributing`` / ``embedded`` directly and
       ``tied_contact`` / ``mortar`` via the
       :meth:`SurfaceCouplingRecord.slave_records` expansion that
       ``interpolations()`` performs internally.

    Each constraint with a non-empty ``name`` is preceded by
    ``emitter.mp_constraint_comment(name)`` so the user's declaration
    label round-trips into emitted Tcl / Py via the ``# {name}`` line
    (INV-2).

    No-op when the FEM snapshot exposes no ``nodes.constraints`` or
    ``elements.constraints`` accessors — broker constraints are
    purely additive on top of any other bridge state.
    """
    from .tag_resolution import set_phantom_node_tags

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

    # -------------------------------------------------------------------
    # 0. Pre-load the phantom-tag predicate on the emitter so the H5
    #    emitter can classify subsequent ``node()`` calls (ADR 0033 —
    #    stateless replacement for the prior phantom-mode flag).  The
    #    set is computed ONCE from ``NodeToSurfaceRecord.phantom_nodes``
    #    and never mutated; phantom tags are disjoint from real broker
    #    tags so this is safe to install before any emission.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        phantom_tags = set(
            _gather_phantom_nodes(node_constraints).keys()
        )
        set_phantom_node_tags(emitter, phantom_tags)

    # -------------------------------------------------------------------
    # 1. Phantom-node pre-step — emit synthesized phantom nodes BEFORE
    #    any constraint references them.  ADR 0022 INV-3.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_phantom_nodes(emitter, node_constraints)

    # -------------------------------------------------------------------
    # 2. Rigid links — ``emitter.rigidLink(kind, master, slave)`` per
    #    pair.  Walks NodePairRecord rows directly (so the kind / name
    #    survive) plus the rigid_body and node_to_surface compound
    #    expansions.  We don't use ``rigid_link_groups()`` because it
    #    drops the per-pair ``name`` field.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_rigid_links(emitter, node_constraints)

    # -------------------------------------------------------------------
    # 3. Equal DOFs — direct NodePairRecord(kind=equal_dof) plus the
    #    NodeToSurfaceRecord.equal_dof_records expansion.
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_equal_dofs(emitter, node_constraints)

    # -------------------------------------------------------------------
    # 4. Rigid diaphragms — one ``rigidDiaphragm`` per
    #    NodeGroupRecord(kind=RIGID_DIAPHRAGM).
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_rigid_diaphragms(emitter, node_constraints)

    # -------------------------------------------------------------------
    # 5. Kinematic couplings — DOF-selective; emitted as equalDOF per
    #    (master, slave) pair.  These ride NodeGroupRecord rows but
    #    cannot collapse onto ``rigidDiaphragm`` (would over-constrain
    #    by ignoring rec.dofs).
    # -------------------------------------------------------------------
    if node_constraints is not None:
        _emit_kinematic_couplings(emitter, node_constraints)

    # -------------------------------------------------------------------
    # 6. Surface couplings — InterpolationRecord (tie / distributing /
    #    embedded) plus the SurfaceCouplingRecord.slave_records
    #    expansion (tied_contact / mortar).  All go out as
    #    ASDEmbeddedNodeElement.
    # -------------------------------------------------------------------
    if surface_constraints is not None:
        _emit_surface_couplings(emitter, surface_constraints)


# ---------------------------------------------------------------------------
# emit_mp_constraints sub-helpers (split for readability + per-kind unit tests)
# ---------------------------------------------------------------------------


def _emit_phantom_nodes(
    emitter: "Emitter", node_constraints: object,
) -> None:
    """Emit ``node(tag, *xyz, ndf=6)`` for every phantom node.

    Phantoms only exist on :class:`NodeToSurfaceRecord` rows — the
    resolver synthesizes them at resolve time and stores their tags +
    coords on the record without writing them into ``fem.nodes`` (see
    pre-flight audit in the Phase 7b spec).  De-duplicates tags across
    records — paranoid-cheap; the resolver maintains a single counter,
    but the set check is one line.

    The H5 emitter's phantom-vs-real-broker discriminator is the
    stateless :func:`is_phantom_node` predicate, pre-loaded on the
    emitter by :func:`emit_mp_constraints` before this helper runs
    (S2 / ADR 0033 — real broker nodes can also pass ``ndf=K`` now,
    so the explicit predicate replaces the old "``ndf is not None``"
    heuristic).  No flag flipping needed here.
    """
    n2s_iter = getattr(node_constraints, "node_to_surfaces", None)
    if n2s_iter is None:
        return
    seen: set[int] = set()
    for rec in n2s_iter():
        coords = rec.phantom_coords
        if coords is None:
            continue
        for tag, xyz in zip(rec.phantom_nodes, coords):
            t = int(tag)
            if t in seen:
                continue
            seen.add(t)
            x, y, z = (float(c) for c in xyz)
            # Per-node ``-ndf 6`` override — phantoms are 6-DOF even
            # when the surrounding slaves are 3-DOF (standard OpenSees
            # idiom for mixed-ndf models).
            emitter.node(t, x, y, z, ndf=6)


def _emit_rigid_links(
    emitter: "Emitter", node_constraints: object,
) -> None:
    """Emit ``rigidLink`` per :class:`NodePairRecord` (rigid_beam /
    rigid_rod) plus the rigid_body and node_to_surface compound
    expansions.  Preserves the per-record ``name`` for INV-2.
    """
    from apeGmsh._kernel.records._constraints import (
        NodeGroupRecord, NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    rigid_pair_kinds = {
        ConstraintKind.RIGID_BEAM, ConstraintKind.RIGID_ROD,
    }
    for rec in node_constraints:
        if isinstance(rec, NodePairRecord):
            if rec.kind in rigid_pair_kinds:
                kind = "beam" if rec.kind == ConstraintKind.RIGID_BEAM else "bar"
                _emit_name(emitter, rec.name)
                emitter.rigidLink(
                    kind, int(rec.master_node), int(rec.slave_node),
                )
        elif isinstance(rec, NodeGroupRecord):
            # Only rigid_body collapses to rigidLink — rigid_diaphragm
            # has its own emit; kinematic_coupling is handled by
            # _emit_kinematic_couplings (DOF-selective).
            if rec.kind == ConstraintKind.RIGID_BODY:
                # Emit the name once for the whole group (one row in
                # H5; one ``# name`` comment in Tcl preceding the first
                # rigidLink line).
                _emit_name(emitter, rec.name)
                for sn in rec.slave_nodes:
                    emitter.rigidLink(
                        "beam", int(rec.master_node), int(sn),
                    )
        elif isinstance(rec, NodeToSurfaceRecord):
            for pair in rec.rigid_link_records:
                if pair.kind in rigid_pair_kinds:
                    kind = (
                        "beam"
                        if pair.kind == ConstraintKind.RIGID_BEAM
                        else "bar"
                    )
                    _emit_name(emitter, pair.name)
                    emitter.rigidLink(
                        kind, int(pair.master_node), int(pair.slave_node),
                    )


def _emit_equal_dofs(
    emitter: "Emitter", node_constraints: object,
) -> None:
    """Emit ``equalDOF`` per :class:`NodePairRecord` (equal_dof) plus
    the :attr:`NodeToSurfaceRecord.equal_dof_records` expansion.
    """
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if isinstance(rec, NodePairRecord):
            if rec.kind == ConstraintKind.EQUAL_DOF:
                _emit_name(emitter, rec.name)
                emitter.equalDOF(
                    int(rec.master_node), int(rec.slave_node),
                    *(int(d) for d in rec.dofs),
                )
        elif isinstance(rec, NodeToSurfaceRecord):
            for pair in rec.equal_dof_records:
                _emit_name(emitter, pair.name)
                emitter.equalDOF(
                    int(pair.master_node), int(pair.slave_node),
                    *(int(d) for d in pair.dofs),
                )


def _emit_rigid_diaphragms(
    emitter: "Emitter", node_constraints: object,
) -> None:
    """Emit ``rigidDiaphragm(perp_dir, master, *slaves)`` per
    :class:`NodeGroupRecord` row with ``kind == 'rigid_diaphragm'``.
    Uses the broker's :meth:`rigid_diaphragms` iterator for the
    perp_dir derivation; iterates the raw records in parallel to keep
    the per-record ``name`` aligned with each emit.
    """
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    # Iterate the underlying records directly (not the
    # ``rigid_diaphragms()`` helper) so we still have access to the
    # original record's ``name`` field — the helper drops it.
    for rec in node_constraints:
        if not (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.RIGID_DIAPHRAGM
        ):
            continue
        perp = _perp_dirn_from_normal(rec.plane_normal)
        _emit_name(emitter, rec.name)
        emitter.rigidDiaphragm(
            perp, int(rec.master_node),
            *(int(s) for s in rec.slave_nodes),
        )


def _emit_kinematic_couplings(
    emitter: "Emitter", node_constraints: object,
) -> None:
    """Emit ``equalDOF`` per (master, slave) pair for
    :class:`NodeGroupRecord` rows with ``kind == 'kinematic_coupling'``.

    Kinematic coupling is DOF-selective (the user picks which DOFs to
    couple); collapsing onto ``rigidDiaphragm`` would over-constrain
    by promoting all 6 DOFs.  ``equalDOF`` with the explicit dofs
    list is the correct OpenSees primitive.
    """
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if not (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.KINEMATIC_COUPLING
        ):
            continue
        _emit_name(emitter, rec.name)
        for sn in rec.slave_nodes:
            emitter.equalDOF(
                int(rec.master_node), int(sn),
                *(int(d) for d in rec.dofs),
            )


def _emit_surface_couplings(
    emitter: "Emitter", surface_constraints: object,
) -> None:
    """Emit ``element ASDEmbeddedNodeElement`` per
    :class:`InterpolationRecord` row (covers ``tie`` / ``distributing``
    / ``embedded`` directly; tied_contact / mortar via the
    :meth:`SurfaceCouplingRecord.slave_records` expansion).

    The second positional ``cnode`` is the constrained (embedded /
    slave) node — the ``$Cnode`` slot in the OpenSees signature
    ``element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 ...``.  The
    variadic tail carries the host element's corner node tags
    ($Rnode1..$RnodeN); ASDEmbeddedNodeElement uses isoparametric
    interpolation over those corners internally, so the per-record
    weights from :class:`InterpolationRecord` are NOT emitted here
    (they survive in the FEM record for round-tripping).  Each
    emitted line allocates a fresh integer element tag from a
    per-call counter — these tags are file-internal and do not
    collide with bridge-allocated element tags because the
    ASDEmbeddedNodeElement vocabulary uses the ``element`` family
    namespace.
    """
    from apeGmsh._kernel.records._constraints import InterpolationRecord

    interps = getattr(surface_constraints, "interpolations", None)
    if interps is None:
        return

    # File-internal counter — start high enough to avoid colliding with
    # whatever the bridge has allocated for structural elements.  The
    # actual tag values are only meaningful to the emitter (they need
    # to be distinct per emit pass).
    next_tag = _allocate_embedded_tag_base(emitter)

    for rec in interps():
        if not isinstance(rec, InterpolationRecord):
            continue
        _emit_name(emitter, rec.name)
        ele_tag = next_tag
        next_tag += 1
        # ASDEmbeddedNodeElement OpenSees signature:
        #   element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 $Rnode2 $Rnode3 <$Rnode4>
        # where $Cnode is the CONSTRAINED (embedded / slave) node and
        # $Rnode* are the host element's corner nodes. The Emitter
        # Protocol forwards as: embeddedNode(ele_tag, $Cnode, *$Rnodes).
        # — weights are recorded under the FEM record but not emitted
        # here; ASDEmbeddedNodeElement uses isoparametric interpolation
        # over the host element's corners internally.
        cnode = int(rec.slave_node)
        args: list[int | float] = [int(mn) for mn in rec.master_nodes]
        emitter.embeddedNode(ele_tag, cnode, *args)


def _allocate_embedded_tag_base(emitter: "Emitter") -> int:
    """Return a starting element tag for ASDEmbeddedNodeElement emits.

    The base is offset above any bridge-allocated element tag so
    embedded-node tags never collide.  Static offset chosen by the
    spec; emitters with stricter tag-collision needs can override by
    pre-populating a higher counter.
    """
    # Lazy peek at any ``_orientation_tag_counter`` (H5Emitter) or
    # similar hint on the emitter; default to 1_000_000 which is well
    # outside the typical bridge-allocated range for mesh elements.
    base = 1_000_000
    return base


def _emit_name(emitter: "Emitter", name: object) -> None:
    """Emit ``emitter.mp_constraint_comment(name)`` if ``name`` is
    a non-empty string (ADR 0022 INV-2).
    """
    if name is None:
        return
    if not isinstance(name, str):
        return
    if not name:
        return
    emitter.mp_constraint_comment(name)


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


# ---------------------------------------------------------------------------
# Partition-aware emission (ADR 0027, P4) — closes the unpartitioned-only
# assumption baked into the original emit pipeline.
# ---------------------------------------------------------------------------


def is_partitioned(fem: "FEMData") -> bool:
    """True iff the FEM snapshot carries more than one partition.

    Single-partition (or unpartitioned) FEMs use the flat emit path —
    bit-identical to the pre-ADR 0027 behaviour, with no
    ``partition_open`` / ``partition_close`` calls and no runtime shim.
    Multi-partition FEMs route through the per-rank fan-out helpers
    that emit per-rank-bracketed output and replicate cross-partition
    MP constraints per ADR 0027.
    """
    parts = getattr(fem, "partitions", None)
    if parts is None:
        return False
    try:
        return len(parts) > 1
    except TypeError:
        return False


def build_node_partition_owners(fem: "FEMData") -> dict[int, set[int]]:
    """Return ``{node_tag: set[rank_id]}`` covering every owning rank.

    A node may belong to multiple partitions (boundary / shared nodes
    that the partitioner replicates across ranks).  The returned dict
    is keyed by every node id that appears in any ``PartitionRecord``;
    unknown / unowned nodes are absent from the map.

    ``rank_id`` is the **0-based runtime rank** — matching
    ``OpenSeesMP::getPID()`` — derived via ``enumerate`` over
    ``fem.partitions`` (which already iterates in sorted Gmsh-id
    order, so the assignment is stable and deterministic).  The
    broker's Gmsh-side 1-based ``PartitionRecord.id`` is preserved
    on the records themselves; only the runtime-rank seam (this
    helper, ``build_element_partition_owner``, and the
    ``partition_rank`` parameter passed to per-rank fan-out helpers)
    is 0-based.
    """
    owners: dict[int, set[int]] = {}
    parts = getattr(fem, "partitions", None)
    if parts is None:
        return owners
    for rank, rec in enumerate(parts):
        for nid in rec.node_ids:
            owners.setdefault(int(nid), set()).add(rank)
    return owners


def build_element_partition_owner(fem: "FEMData") -> dict[int, int]:
    """Return ``{element_tag: rank_id}`` — each element lives on exactly one rank.

    Unlike nodes, the partitioner gives each element to a single rank
    (interface elements typically belong to a "interface partition" in
    Gmsh; here we honour whatever the broker recorded).  Element ids
    not present in any partition are absent from the map; callers
    interpret "missing key" as "unowned" and either skip emission or
    fall back to an explicit policy.

    ``rank_id`` is the **0-based runtime rank** — matching
    ``OpenSeesMP::getPID()`` — derived via ``enumerate`` over
    ``fem.partitions`` (sorted Gmsh-id order).  See the docstring
    on :func:`build_node_partition_owners` for the rationale.
    """
    owners: dict[int, int] = {}
    parts = getattr(fem, "partitions", None)
    if parts is None:
        return owners
    for rank, rec in enumerate(parts):
        for eid in rec.element_ids:
            # If an element appears in two partitions (degenerate input),
            # the first-seen partition wins.  This is a deterministic
            # tiebreak; callers can rely on it.
            owners.setdefault(int(eid), rank)
    return owners


def _intersect_with_partition(
    ids: "Iterable[int]", partition_ids: "Iterable[int]",
) -> tuple[int, ...]:
    """Return ``ids`` intersected with ``partition_ids``, preserving order.

    Helper for per-rank region intersection (ADR 0027 INV-4) and any
    other per-rank "owned subset" pruning.  Order is the order of
    ``ids`` (the supplied iteration order is preserved); the partition
    side is hashed for O(1) membership.
    """
    part_set = set(int(p) for p in partition_ids)
    out: list[int] = []
    seen: set[int] = set()
    for i in ids:
        ii = int(i)
        if ii in part_set and ii not in seen:
            seen.add(ii)
            out.append(ii)
    return tuple(out)


def allocate_element_tags(
    elements: "Iterable[Element]",
    fem: "FEMData",
    tags: TagAllocator,
) -> "list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]]":
    """Allocate canonical element tags up-front, return per-spec plan.

    Returns a list of ``(spec, [(eid, conn, ele_tag), ...])`` so the
    per-rank fan-out can simply look up each element's pre-allocated
    tag instead of consuming the allocator per-rank (which would
    produce diverging tag numbering across ranks — ADR 0027 §"Tag
    determinism").  Iteration order matches the flat fan-out so tags
    are byte-identical to the unpartitioned path for shared elements.
    """
    plan: list[tuple[Element, list[tuple[int, tuple[int, ...], int]]]] = []
    for spec in elements:
        sub: list[tuple[int, tuple[int, ...], int]] = []
        for eid, conn in expand_pg_to_elements(fem, spec.pg):  # type: ignore[attr-defined]
            ele_tag = tags.allocate("element")
            sub.append((int(eid), tuple(int(n) for n in conn), int(ele_tag)))
        plan.append((spec, sub))
    return plan


def compute_stage_ownership(
    stage_records: "tuple[StageRecord, ...]",
    elements: "Iterable[Element]",
    fem: "FEMData",
) -> "tuple[dict[int, int], dict[int, int]]":
    """Compute element + node ownership maps for Phase SSI-2.B.

    For each stage in registration order, walks the Element primitives
    whose ``pg=`` matches one of the stage's ``activated_pgs`` and
    assigns them (and their referenced nodes) to that stage.

    Returns
    -------
    element_owner : dict[int, int]
        ``{id(element_primitive): stage_index}`` — stage index is the
        position in ``stage_records``.  Element primitives not in any
        stage's activation set are absent from this map (global emit).
    node_owner : dict[int, int]
        ``{fem_node_id: stage_index}`` — node ID is the broker's FEM
        node id.  A node referenced by ANY global element stays global
        (absent from this map).  A node referenced ONLY by stage-bound
        elements is owned by the *lowest* stage index that references
        it.

    Raises
    ------
    BridgeError
        If a PG is activated by more than one stage (ambiguous
        ownership — first-write wins is unsafe; the user clearly
        meant something different).
    """
    # 1. Map PG name → owning stage index.  Raise on conflicts.
    pg_owner: dict[str, int] = {}
    for stage_idx, stage in enumerate(stage_records):
        for pg in stage.activated_pgs:
            if pg in pg_owner:
                raise BridgeError(
                    f"Stage {stage.name!r}: PG {pg!r} is activated by "
                    f"another stage (index {pg_owner[pg]}); PGs may be "
                    "activated by AT MOST one stage."
                )
            pg_owner[pg] = stage_idx

    # 2. Walk elements; map each spec to its owning stage (if any).
    element_owner: dict[int, int] = {}
    # We need to compute node ownership too — walk every element's
    # PG fan-out to collect (eid → set of stages that reference it).
    node_stages: dict[int, set[int]] = {}
    # And: nodes referenced by any GLOBAL element are global,
    # independent of which stages also reference them.
    global_nodes: set[int] = set()
    # Track which element-PG names actually appear on registered
    # Element primitives — used below to validate that every
    # ``s.activate(pgs=)`` PG matches at least one Element.
    seen_element_pgs: set[str] = set()
    for spec in elements:
        spec_pg = getattr(spec, "pg", None)
        if spec_pg:
            seen_element_pgs.add(spec_pg)
        owner_idx = pg_owner.get(spec_pg) if spec_pg else None
        if owner_idx is not None:
            element_owner[id(spec)] = owner_idx
        for _eid, conn in expand_pg_to_elements(fem, spec_pg):
            for node_id in conn:
                if owner_idx is None:
                    global_nodes.add(int(node_id))
                else:
                    node_stages.setdefault(int(node_id), set()).add(owner_idx)

    # 2b. Validate every activated PG matches at least one registered
    # Element primitive's ``pg=`` (red-team M1).  A typo'd PG name
    # would otherwise silently no-op — no domain_change, no element
    # emit, no error — leaving the user with a wrong-but-runnable
    # deck.
    unknown_pgs: list[tuple[str, str]] = []  # (stage_name, pg)
    for stage_idx, stage in enumerate(stage_records):
        for pg in stage.activated_pgs:
            if pg not in seen_element_pgs:
                unknown_pgs.append((stage.name, pg))
    if unknown_pgs:
        joined = ", ".join(
            f"stage {sn!r} → {pg!r}" for sn, pg in unknown_pgs
        )
        raise BridgeError(
            f"Stage activation references unknown element PGs: "
            f"{joined}.  Each ``s.activate(pgs=...)`` PG must match "
            f"at least one registered Element primitive's ``pg=``.  "
            f"Registered element PGs: {sorted(seen_element_pgs)}."
        )

    # 3. Assemble node_owner: a node is stage-bound to the lowest
    # stage index that references it iff it's NOT referenced by any
    # global element.
    node_owner: dict[int, int] = {}
    for node_id, stages in node_stages.items():
        if node_id in global_nodes:
            continue  # global element references it; stays global.
        node_owner[node_id] = min(stages)

    return element_owner, node_owner


def resolve_initial_stress_elements(
    rec: InitialStressRecord, fem: "FEMData",
) -> tuple[int, ...]:
    """Return the FEM element ids for an :class:`InitialStressRecord`.

    Exactly one of ``rec.pg`` / ``rec.elements`` is non-None
    (validated at the call site in :meth:`apeSees.initial_stress`).
    """
    if rec.elements is not None:
        return rec.elements
    if rec.pg is not None:
        return tuple(eid for eid, _conn in expand_pg_to_elements(fem, rec.pg))
    return ()


def emit_initial_stress_global(
    records: "Iterable[InitialStressRecord]",
    emitter: "Emitter",
    tags: TagAllocator,
) -> dict[str, tuple[int, int, int]]:
    """Emit the global side of each :class:`InitialStressRecord`.

    For each record, allocates three parameter tags (XX, YY, ZZ) from
    the bridge allocator, then calls :meth:`Emitter.step_hook_ramp`,
    which bundles the dispatcher boilerplate (once), the parameter
    declarations, the per-step proc, and the dispatcher registration.

    Returns the mapping ``{record_name: (xx_tag, yy_tag, zz_tag)}`` so
    the per-rank ``addToParameter`` fan-out (see
    :func:`emit_initial_stress_addtoparameter`) can reach the same
    tags without re-allocating.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for rec in records:
        xx_tag = tags.allocate("parameter")
        yy_tag = tags.allocate("parameter")
        zz_tag = tags.allocate("parameter")
        targets = (
            (xx_tag, rec.sigma_xx * rec.lambda_install),
            (yy_tag, rec.sigma_yy * rec.lambda_install),
            (zz_tag, rec.sigma_zz * rec.lambda_install),
        )
        emitter.step_hook_ramp(
            name=rec.name,
            targets=targets,
            n_steps_to_full=float(rec.ramp_steps),
            phase="before",
        )
        out[rec.name] = (xx_tag, yy_tag, zz_tag)
    return out


def emit_initial_stress_addtoparameter(
    records: "Iterable[InitialStressRecord]",
    emitter: "Emitter",
    fem: "FEMData",
    name_to_param_tags: dict[str, tuple[int, int, int]],
    fem_eid_to_ops_tag: dict[int, int],
    element_owner: dict[int, int] | None = None,
    partition_rank: int | None = None,
) -> None:
    """Emit ``addToParameter`` for each element covered by each record.

    Per-rank semantics: if ``partition_rank`` is supplied (MP-mode
    emission inside a ``partition_open`` block), elements are filtered
    against ``element_owner`` and only owned elements emit.  Single-
    partition / flat callers pass ``partition_rank=None`` and the
    filter is skipped.

    Elements whose FEM id is in ``fem_eid_to_ops_tag`` but does NOT
    match this rank (in MP mode) are silently skipped — they belong
    to a different rank.  Elements whose FEM id is ABSENT from
    ``fem_eid_to_ops_tag`` entirely (e.g. the user passed an
    ``elements=[bad_id]`` that doesn't match any registered Element
    primitive) raise :class:`BridgeError` in single-partition mode
    so the user sees the mistake (red-team M6).  Under MP the same
    eid might legitimately be missing on this rank because it's
    owned by another rank — there we keep the silent-skip behaviour.
    """
    components = (
        ("commitStressIncrementXX", 0),
        ("commitStressIncrementYY", 1),
        ("commitStressIncrementZZ", 2),
    )
    is_partitioned_mode = partition_rank is not None
    for rec in records:
        param_tags = name_to_param_tags[rec.name]
        for eid in resolve_initial_stress_elements(rec, fem):
            if is_partitioned_mode and element_owner is not None:
                owner = element_owner.get(int(eid))
                if owner is None or owner != partition_rank:
                    continue
            ops_tag = fem_eid_to_ops_tag.get(int(eid))
            if ops_tag is None:
                if is_partitioned_mode:
                    continue  # owned by another rank; silent skip OK.
                raise BridgeError(
                    f"initial_stress {rec.name!r}: element id {int(eid)} "
                    "is not registered with any Element primitive "
                    "(would silently no-op the addToParameter response).  "
                    "Either remove it from ``elements=`` or register the "
                    "matching Element primitive via "
                    "``ops.element.<Type>(pg=...)``."
                )
            for response, idx in components:
                emitter.addToParameter(
                    int(param_tags[idx]), int(ops_tag), response,
                )


def emit_element_spec_partitioned(
    spec: Element,
    emitter: "Emitter",
    fem: "FEMData",
    pre_allocated: "list[tuple[int, tuple[int, ...], int]]",
    base_resolver: object,
    transf_tag_for_element: dict[tuple[int, int], int] | None,
    partition_rank: int,
    element_owner: dict[int, int],
) -> None:
    """Per-rank element fan-out (ADR 0027).

    Emits ONLY the elements of ``spec.pg`` whose owner-rank matches
    ``partition_rank``.  Tags come from ``pre_allocated`` (built once
    by :func:`allocate_element_tags`) so cross-rank tag identity is
    preserved verbatim per ADR 0027 §"Tag determinism".
    """
    if not pre_allocated:
        return
    transf_spec = _element_transf(spec)

    for eid, node_tags, ele_tag in pre_allocated:
        owner = element_owner.get(int(eid))
        if owner is None or owner != partition_rank:
            continue
        set_element_nodes(emitter, node_tags)
        set_current_fem_element_id(emitter, eid)

        if (
            transf_spec is not None
            and transf_tag_for_element is not None
            and (id(transf_spec), eid) in transf_tag_for_element
        ):
            override_tag = transf_tag_for_element[(id(transf_spec), eid)]
            base = base_resolver
            override = transf_spec

            def _resolver_with_override(
                p: Primitive,
                _base: object = base,
                _override_spec: Primitive = override,
                _override_tag: int = override_tag,
            ) -> int:
                if p is _override_spec:
                    return _override_tag
                return int(_base(p))  # type: ignore[operator]

            set_tag_resolver(emitter, _resolver_with_override)
            try:
                spec._emit(emitter, ele_tag)
            finally:
                set_tag_resolver(emitter, base_resolver)  # type: ignore[arg-type]
        else:
            spec._emit(emitter, ele_tag)


# ---------------------------------------------------------------------------
# Cross-partition MP-constraint replication (ADR 0027 §"Decision")
# ---------------------------------------------------------------------------


def _node_coords_safe(fem: "FEMData", node_id: int) -> tuple[float, float, float]:
    """Best-effort ``(x, y, z)`` lookup for ``node_id``.

    Returns zeros if the node id is unknown to the broker (e.g. a
    phantom tag whose coords live on the source record, not on
    ``fem.nodes``); the caller is responsible for using
    :meth:`_emit_phantom_nodes` for those.
    """
    try:
        idx = fem.nodes.index(int(node_id))
        xyz = fem.nodes.coords[idx]
        return float(xyz[0]), float(xyz[1]), float(xyz[2])
    except (KeyError, IndexError, AttributeError):
        return (0.0, 0.0, 0.0)


def _gather_phantom_nodes(node_constraints: object) -> dict[int, tuple[float, float, float]]:
    """Walk ``NodeToSurfaceRecord`` rows and return ``{phantom_tag: xyz}``.

    Phantom tags are broker-derived (one canonical numbering) — per
    ADR 0027 INV-3 the same tag and the same coords appear on every
    rank that hosts a constraint referencing them.
    """
    out: dict[int, tuple[float, float, float]] = {}
    n2s_iter = getattr(node_constraints, "node_to_surfaces", None)
    if n2s_iter is None:
        return out
    for rec in n2s_iter():
        coords = rec.phantom_coords
        if coords is None:
            continue
        for tag, xyz in zip(rec.phantom_nodes, coords):
            t = int(tag)
            if t in out:
                continue
            out[t] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


def emit_mp_constraints_partitioned(
    emitter: "Emitter",
    fem: "FEMData",
    partition_rank: int,
    node_owners: dict[int, set[int]],
    element_owner: dict[int, int],
    foreign_node_ndf: int | None,
) -> None:
    """Per-rank MP-constraint fan-out (ADR 0027 §"Decision").

    For ``partition_rank``:

    1. Collect every constraint that touches this rank — meaning any
       node it references lives on this rank (or, for embedded-node /
       ASDEmbeddedNodeElement, the host element lives on this rank).
    2. Compute the set of foreign nodes referenced by those
       constraints (nodes not in this rank's owner set, plus phantoms).
    3. Emit the foreign-node declarations FIRST (INV-2): regular
       foreign nodes via ``node(tag, *xyz[, -ndf K])`` — per-node ndf
       sourced from the FEM broker via ``fem.nodes.ndf_for(tag)``
       (S2 / ADR 0033); on ``LookupError`` (no declaration covering
       the foreign node) the call falls back to ``foreign_node_ndf``
       — the bridge envelope.  Phantoms via
       ``node(tag, *xyz, ndf=6)``.
    4. Emit the constraints in the same order as the unpartitioned
       :func:`emit_mp_constraints` (phantom-node pre-step → rigidLink
       → equalDOF → rigidDiaphragm → kinematic_coupling →
       ASDEmbeddedNodeElement) so cross-rank text is byte-identical
       per INV-1.
    """
    from .tag_resolution import set_phantom_node_tags
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

    if node_constraints is None and surface_constraints is None:
        return

    # Build the lookup tables used during constraint replication.
    phantom_coords = (
        _gather_phantom_nodes(node_constraints)
        if node_constraints is not None
        else {}
    )

    # Pre-load the phantom-tag predicate on the emitter (ADR 0033 —
    # stateless replacement for the prior phantom-mode flag).
    # Per-rank brokers see identical phantom tags (the resolver
    # canonicalises across ranks per ADR 0027 INV-3), so this set is
    # consistent across the OpenSeesMP fan-out.
    set_phantom_node_tags(emitter, set(phantom_coords.keys()))

    # Collect which constraints will emit on this rank, plus all the
    # foreign-node tags they reference (replicate-on-both / replicate-
    # everywhere-with-a-slave rules from ADR 0027).
    plan = _plan_rank_constraints(
        node_constraints=node_constraints,
        surface_constraints=surface_constraints,
        partition_rank=partition_rank,
        node_owners=node_owners,
        element_owner=element_owner,
        phantom_tags=set(phantom_coords.keys()),
    )
    if not plan.any():
        return

    # -- 1. Foreign-node declarations (INV-2). ---------------------------
    # Phantoms first (their tags are 6-DOF regardless of model ndf,
    # mirroring the unpartitioned phantom-node-first invariant within
    # this rank's block per ADR 0027 §"Phantom-node policy").
    for tag in sorted(plan.referenced_phantoms):
        xyz = phantom_coords[tag]
        emitter.node(tag, xyz[0], xyz[1], xyz[2], ndf=6)
    # Foreign nodes — per-node ndf via broker lookup; envelope
    # fallback on LookupError (S2 / ADR 0033).  OpenSeesMP per-rank
    # brokers carry identical ``_ndf`` arrays (lineage-folded per
    # ADR 0021), so this lookup resolves consistently across ranks.
    for tag in sorted(plan.foreign_node_tags):
        xyz = _node_coords_safe(fem, tag)
        _emit_node_with_broker_ndf(
            emitter, fem, int(tag),
            (xyz[0], xyz[1], xyz[2]),
            envelope_ndf=foreign_node_ndf,
        )

    # -- 2. Constraint emission, mirroring the unpartitioned order. ------
    if node_constraints is not None:
        _emit_rigid_links_filtered(
            emitter, node_constraints, plan.allowed_record_ids,
        )
        _emit_equal_dofs_filtered(
            emitter, node_constraints, plan.allowed_record_ids,
        )
        _emit_rigid_diaphragms_filtered(
            emitter, node_constraints, plan.allowed_record_ids,
        )
        _emit_kinematic_couplings_filtered(
            emitter, node_constraints, plan.allowed_record_ids,
        )

    # ASDEmbeddedNodeElement: only the host-element-owning rank emits.
    if surface_constraints is not None and plan.embedded_records:
        _emit_surface_couplings_for_rank(
            emitter, plan.embedded_records,
        )


@dataclass(frozen=True, slots=True)
class _RankConstraintPlan:
    """What the per-rank MP-constraint pass will emit on this rank."""

    allowed_record_ids: frozenset[int]
    foreign_node_tags: frozenset[int]
    referenced_phantoms: frozenset[int]
    embedded_records: tuple[object, ...]

    def any(self) -> bool:
        return bool(self.allowed_record_ids) or bool(self.embedded_records)


def _plan_rank_constraints(
    *,
    node_constraints: object,
    surface_constraints: object,
    partition_rank: int,
    node_owners: dict[int, set[int]],
    element_owner: dict[int, int],
    phantom_tags: set[int],
) -> _RankConstraintPlan:
    """Decide which constraint records emit on ``partition_rank``."""
    from apeGmsh._kernel.records._constraints import (
        InterpolationRecord,
        NodeGroupRecord,
        NodePairRecord,
        NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    allowed_ids: set[int] = set()
    foreign_nodes: set[int] = set()
    referenced_phantoms: set[int] = set()
    embedded: list[object] = []

    def _owns(node_tag: int) -> bool:
        return partition_rank in node_owners.get(int(node_tag), set())

    def _is_phantom(node_tag: int) -> bool:
        return int(node_tag) in phantom_tags

    def _add_foreign_or_phantom(node_tag: int) -> None:
        t = int(node_tag)
        if _owns(t):
            return
        if _is_phantom(t):
            referenced_phantoms.add(t)
        else:
            foreign_nodes.add(t)

    if node_constraints is not None:
        for rec in node_constraints:
            if isinstance(rec, NodePairRecord):
                # equal_dof / rigid_beam / rigid_rod replicate on both
                # owning ranks (ADR 0027 §"Decision" — bullets 1, 2).
                m = int(rec.master_node)
                s = int(rec.slave_node)
                touches = _owns(m) or _owns(s)
                # Honor phantom slave on rigid_beam side: phantoms are
                # never "owned" by ranks per the broker (they're
                # broker-synthetic). A constraint whose master is on
                # this rank but slave is a phantom must still emit here
                # so that the phantom→slave equalDOF pair (emitted via
                # the NodeToSurface expansion below) sees the master.
                if not touches and (_is_phantom(m) or _is_phantom(s)):
                    # If neither master nor slave is owned by this rank
                    # and neither is a phantom referenced by another
                    # owned constraint, skip.  Pure phantom-phantom
                    # pairs are not expected from the broker.
                    pass
                if touches:
                    allowed_ids.add(id(rec))
                    _add_foreign_or_phantom(m)
                    _add_foreign_or_phantom(s)
            elif isinstance(rec, NodeGroupRecord):
                # rigid_body / rigid_diaphragm / kinematic_coupling.
                # Per ADR 0027 §"Decision" bullet 3:
                #   "emit on every rank that owns any slave node".
                # The full command line is emitted verbatim on each
                # such rank; slaves are not sharded.
                m = int(rec.master_node)
                slaves = [int(s) for s in rec.slave_nodes]
                touches = _owns(m) or any(_owns(s) for s in slaves)
                if touches:
                    allowed_ids.add(id(rec))
                    _add_foreign_or_phantom(m)
                    for s in slaves:
                        _add_foreign_or_phantom(s)
            elif isinstance(rec, NodeToSurfaceRecord):
                # NodeToSurface is a compound record. Its rigid_link
                # rows and equal_dof rows reference phantom nodes; we
                # treat the compound as a single bundle and replicate
                # it on every rank that owns the master OR any slave OR
                # any phantom.  Phantoms are "owned" by the rank that
                # owns their slave-side node (the rigid_beam / equalDOF
                # row chains phantom → slave).
                touches = False
                for pair in rec.rigid_link_records:
                    if _owns(int(pair.master_node)) or _owns(int(pair.slave_node)):
                        touches = True
                        break
                if not touches:
                    for pair in rec.equal_dof_records:
                        if (
                            _owns(int(pair.master_node))
                            or _owns(int(pair.slave_node))
                        ):
                            touches = True
                            break
                if touches:
                    allowed_ids.add(id(rec))
                    for pair in rec.rigid_link_records:
                        _add_foreign_or_phantom(int(pair.master_node))
                        _add_foreign_or_phantom(int(pair.slave_node))
                    for pair in rec.equal_dof_records:
                        _add_foreign_or_phantom(int(pair.master_node))
                        _add_foreign_or_phantom(int(pair.slave_node))

    if surface_constraints is not None:
        # ASDEmbeddedNodeElement ownership: bound to the rank that owns
        # the host element. We allocate file-internal element tags
        # canonically (per-call counter) and only the host-element-
        # owning rank emits.  Per ADR 0027 §"ASDEmbeddedNodeElement
        # ownership". Since the host element of an
        # ASDEmbeddedNodeElement is an *implied* element (not in
        # fem.elements), we use the first master node id as the proxy
        # for ownership — the surface constraint's interpolation
        # touches a real master element, so its corner nodes all live
        # on the same rank (the partitioner does not split element
        # connectivity).
        interps_iter = getattr(surface_constraints, "interpolations", None)
        if interps_iter is not None:
            for rec in interps_iter():
                if not isinstance(rec, InterpolationRecord):
                    continue
                # Host ownership: the rank that owns ALL master nodes
                # of the host element. In the partitioner this is
                # exactly the rank that owns the host element since the
                # master nodes form one element's connectivity. We pick
                # the first master node's owner-set; any rank in that
                # set is the host's rank (and the masters are
                # guaranteed co-resident).
                masters = [int(mn) for mn in rec.master_nodes]
                if not masters:
                    continue
                host_owners = node_owners.get(masters[0], set())
                if partition_rank in host_owners:
                    embedded.append(rec)
                    # Declare any foreign master/slave nodes.
                    for mn in masters:
                        _add_foreign_or_phantom(mn)
                    _add_foreign_or_phantom(int(rec.slave_node))

    return _RankConstraintPlan(
        allowed_record_ids=frozenset(allowed_ids),
        foreign_node_tags=frozenset(foreign_nodes),
        referenced_phantoms=frozenset(referenced_phantoms),
        embedded_records=tuple(embedded),
    )


def _emit_rigid_links_filtered(
    emitter: "Emitter", node_constraints: object,
    allowed_ids: frozenset[int],
) -> None:
    """Subset of :func:`_emit_rigid_links` honoring ``allowed_ids``."""
    from apeGmsh._kernel.records._constraints import (
        NodeGroupRecord, NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    rigid_pair_kinds = {
        ConstraintKind.RIGID_BEAM, ConstraintKind.RIGID_ROD,
    }
    for rec in node_constraints:
        if id(rec) not in allowed_ids:
            continue
        if isinstance(rec, NodePairRecord):
            if rec.kind in rigid_pair_kinds:
                kind = "beam" if rec.kind == ConstraintKind.RIGID_BEAM else "bar"
                _emit_name(emitter, rec.name)
                emitter.rigidLink(
                    kind, int(rec.master_node), int(rec.slave_node),
                )
        elif isinstance(rec, NodeGroupRecord):
            if rec.kind == ConstraintKind.RIGID_BODY:
                _emit_name(emitter, rec.name)
                for sn in rec.slave_nodes:
                    emitter.rigidLink(
                        "beam", int(rec.master_node), int(sn),
                    )
        elif isinstance(rec, NodeToSurfaceRecord):
            for pair in rec.rigid_link_records:
                if pair.kind in rigid_pair_kinds:
                    kind = (
                        "beam"
                        if pair.kind == ConstraintKind.RIGID_BEAM
                        else "bar"
                    )
                    _emit_name(emitter, pair.name)
                    emitter.rigidLink(
                        kind, int(pair.master_node), int(pair.slave_node),
                    )


def _emit_equal_dofs_filtered(
    emitter: "Emitter", node_constraints: object,
    allowed_ids: frozenset[int],
) -> None:
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if id(rec) not in allowed_ids:
            continue
        if isinstance(rec, NodePairRecord):
            if rec.kind == ConstraintKind.EQUAL_DOF:
                _emit_name(emitter, rec.name)
                emitter.equalDOF(
                    int(rec.master_node), int(rec.slave_node),
                    *(int(d) for d in rec.dofs),
                )
        elif isinstance(rec, NodeToSurfaceRecord):
            for pair in rec.equal_dof_records:
                _emit_name(emitter, pair.name)
                emitter.equalDOF(
                    int(pair.master_node), int(pair.slave_node),
                    *(int(d) for d in pair.dofs),
                )


def _emit_rigid_diaphragms_filtered(
    emitter: "Emitter", node_constraints: object,
    allowed_ids: frozenset[int],
) -> None:
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if id(rec) not in allowed_ids:
            continue
        if not (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.RIGID_DIAPHRAGM
        ):
            continue
        perp = _perp_dirn_from_normal(rec.plane_normal)
        _emit_name(emitter, rec.name)
        emitter.rigidDiaphragm(
            perp, int(rec.master_node),
            *(int(s) for s in rec.slave_nodes),
        )


def _emit_kinematic_couplings_filtered(
    emitter: "Emitter", node_constraints: object,
    allowed_ids: frozenset[int],
) -> None:
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for rec in node_constraints:
        if id(rec) not in allowed_ids:
            continue
        if not (
            isinstance(rec, NodeGroupRecord)
            and rec.kind == ConstraintKind.KINEMATIC_COUPLING
        ):
            continue
        _emit_name(emitter, rec.name)
        for sn in rec.slave_nodes:
            emitter.equalDOF(
                int(rec.master_node), int(sn),
                *(int(d) for d in rec.dofs),
            )


def _emit_surface_couplings_for_rank(
    emitter: "Emitter", records: tuple[object, ...],
) -> None:
    """Emit ASDEmbeddedNodeElement lines for the host-rank surface couplings."""
    from apeGmsh._kernel.records._constraints import InterpolationRecord

    next_tag = _allocate_embedded_tag_base(emitter)
    for rec in records:
        if not isinstance(rec, InterpolationRecord):
            continue
        _emit_name(emitter, rec.name)
        ele_tag = next_tag
        next_tag += 1
        cnode = int(rec.slave_node)
        args: list[int | float] = [int(mn) for mn in rec.master_nodes]
        emitter.embeddedNode(ele_tag, cnode, *args)
