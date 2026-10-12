"""topo: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations

from typing import Iterable

from ..types import Primitive
from .errors import BridgeError


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
