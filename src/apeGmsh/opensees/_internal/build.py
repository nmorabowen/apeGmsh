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
    "MassRecord",
    "VECXZ_TOL",
    "compute_vecxz_for_element",
    "emit_element_spec",
    "emit_mp_constraints",
    "emit_pattern_spec",
    "emit_recorder_spec",
    "expand_pg_to_elements",
    "expand_pg_to_nodes",
    "is_orientation_transform",
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
    """
    elements = expand_pg_to_elements(fem, spec.pg)  # type: ignore[attr-defined]
    if not elements:
        return

    transf_spec = _element_transf(spec)

    for eid, node_tags in elements:
        ele_tag = tags.allocate("element")
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
    """
    if isinstance(spec, RecorderDeclaration):
        _emit_recorder_declaration(spec, emitter, fem)
        return
    spec.materialize(emitter, fem, tags)._emit(emitter, tag)


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
