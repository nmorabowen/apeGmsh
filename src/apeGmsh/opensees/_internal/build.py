"""
Phase-4 build pipeline helpers.

The bridge's ``BuiltModel.emit`` orchestrates emission, but the
non-trivial work — dependency-sorted ordering, element fan-out across
physical groups, csys-derived per-element ``vecxz`` fan-out, and
pattern / recorder ``pg=`` fan-out — lives here as pure helpers so
the orchestration in :mod:`apesees` stays small and readable.

The helpers in this module never import :mod:`openseespy`; they speak
only to the frozen :class:`Emitter` Protocol via the bridge-attached
tag resolver and element-nodes context (see
:mod:`apeGmsh.opensees._internal.tag_resolution`).

Three deferred contracts are resolved here:

  1. **Element fan-out** across a physical group's element ids and
     connectivity (the bridge writes the per-element node tags into the
     emitter's ``_current_element_nodes`` slot, allocates a per-element
     tag, and drives ``spec._emit`` once per element).

  2. **csys-derived per-element vecxz fan-out** (ADR 0010): when a
     ``Linear`` / ``PDelta`` / ``Corotational`` GeomTransf is constructed
     with ``csys=`` rather than an explicit ``vecxz=``, the bridge
     computes the local tangent for each element in the
     transform-bearing PGs, queries the CS triad at the element midpoint,
     resolves the per-element ``vecxz`` via :func:`resolve_vecxz`, and
     emits one ``geomTransf`` line per distinct ``vecxz`` (within a
     ``1e-9`` tolerance), reusing the same geomTransf tag for elements
     whose vecxz matches.

  3. **Pattern / recorder ``pg=`` fan-out** to per-node and per-element
     tags: the bridge resolves ``pg=`` records on :class:`Plain`
     patterns (loads + sps) and on Node / Element recorders into
     concrete tag lists before driving each primitive's ``_emit``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .._csys import resolve_vecxz

from ..element.beam_column import (
    ElasticTimoshenkoBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from ..pattern.pattern import Plain, _LoadRecord, _SPRecord
from ..recorder import Element as ElementRecorder
from ..recorder import Node as NodeRecorder
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
    "emit_pattern_spec",
    "emit_recorder_spec",
    "expand_pg_to_elements",
    "expand_pg_to_nodes",
    "is_csys_transform",
    "topological_order",
]


#: Tolerance for considering two ``vecxz`` triples equal during the
#: csys-derived fan-out's deduplication step. Two elements whose
#: per-element ``vecxz`` agrees to this tolerance share one
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
# csys → per-element vecxz computation
# ---------------------------------------------------------------------------

_AnyTransf = Linear | PDelta | Corotational

_TRANSF_TYPE_TOKEN: dict[type[GeomTransf], str] = {
    Linear:       "Linear",
    PDelta:       "PDelta",
    Corotational: "Corotational",
}


def is_csys_transform(t: Primitive) -> bool:
    """True if ``t`` is a Linear / PDelta / Corotational with a
    ``csys=`` parameter set (and hence needs per-element vecxz fan-out
    at build time).

    A transform with explicit ``vecxz=`` does NOT need fan-out: it
    emits one ``geomTransf`` line with the spec's allocated tag. The
    bridge checks this flag to decide whether to drive the fan-out or
    let ``spec._emit`` run once with the spec's tag.
    """
    if not isinstance(t, (Linear, PDelta, Corotational)):
        return False
    return t.csys is not None and t.vecxz is None


def compute_vecxz_for_element(
    transf: _AnyTransf,
    p_i: np.ndarray,
    p_j: np.ndarray,
) -> tuple[float, float, float]:
    """Return the per-element ``vecxz`` for one element under ``transf``.

    Reads the CS triad at the element midpoint, computes the unit
    tangent from ``p_i``/``p_j``, and runs the CS rule (ADR 0010) via
    :func:`resolve_vecxz`. ``transf`` MUST be a csys-bearing transform
    (caller checks via :func:`is_csys_transform` first).
    """
    if transf.csys is None:
        raise BridgeError(
            f"compute_vecxz_for_element: transform {transf!r} has no "
            "csys; caller should have used the explicit vecxz path."
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
    e1, e2, e3 = transf.csys.triad_at(midpoint)
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
        result = fem.elements.get(pg=pg)
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
        ids = fem.nodes.get(pg=pg).ids
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
        transform requires csys-driven per-element vecxz overrides.
    transf_tag_for_element
        Dict keyed ``(id(transf_spec), element_id)`` → per-element
        ``geomTransf`` tag. Filled by :func:`emit_transform_specs` for
        csys-bearing transforms; ``None`` (or missing keys) means use
        the spec's own resolver path.
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
# csys-bearing transform fan-out — emit one geomTransf line per distinct
# vecxz observed across the elements that reference the transform spec.
# ---------------------------------------------------------------------------

def emit_transform_specs(
    transforms: Iterable[GeomTransf],
    elements: Iterable[Element],
    emitter: "Emitter",
    fem: "FEMData",
    tags: TagAllocator,
    spec_to_own_tag: dict[int, int],
) -> dict[tuple[int, int], int]:
    """Emit ``geomTransf`` lines for every transform spec.

    For non-csys transforms (explicit ``vecxz=``), one line per spec
    using the spec's own allocated tag — that's the path :class:`Linear`
    / :class:`PDelta` / :class:`Corotational` already handle in their
    ``_emit``.

    For csys-bearing transforms, the bridge:

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
        if not is_csys_transform(transf):
            # Explicit vecxz path — drive the spec's own _emit once.
            transf._emit(emitter, own_tag)
            continue

        # csys path: walk every element whose transf IS this transform,
        # compute per-element vecxz, dedupe.
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
                    raise BridgeError(
                        f"csys transform {type(transf).__name__}: element "
                        f"{eid} in PG {ele_spec.pg!r} has {len(node_ids)} "  # type: ignore[attr-defined]
                        "nodes; csys-driven vecxz fan-out requires line "
                        "elements (2 nodes)."
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
) -> None:
    """Drive a recorder's emit, expanding ``pg=`` to explicit id lists.

    For Node / Element recorders the type-system already requires
    exactly-one-of ``pg`` / ``nodes`` (or ``elements``); we resolve the
    ``pg`` form into an in-memory replica with the equivalent explicit
    list, then drive the replica's ``_emit``. Other recorder kinds
    (MPCO) have no PG resolution and pass through unchanged.
    """
    if isinstance(spec, NodeRecorder) and spec.pg is not None:
        from dataclasses import replace
        node_ids = expand_pg_to_nodes(fem, spec.pg)
        node_replaced = replace(spec, pg=None, nodes=node_ids)
        node_replaced._emit(emitter, tag)
        return
    if isinstance(spec, ElementRecorder) and spec.pg is not None:
        # Element recorders accept only element ids, not connectivity;
        # reuse expand_pg_to_elements but extract just the ids.
        from dataclasses import replace
        elem_ids = tuple(
            eid for eid, _conn in expand_pg_to_elements(fem, spec.pg)
        )
        elem_replaced = replace(spec, pg=None, elements=elem_ids)
        elem_replaced._emit(emitter, tag)
        return
    spec._emit(emitter, tag)
