"""Chain-phase routing helpers — Phase 3B.2d / Compose v1.1-A.

When a session is in *chain phase* (``g._fem is not None``) the broker
is canonical and the legacy "store def → re-extract on next
get_fem_data()" pattern is broken: chain-phase ``get_fem_data()`` short-
circuits to the cached FEMData and never re-resolves the def lists.

This module provides the bridge: given a session whose latest broker
snapshot is ``g._fem``, plus a freshly-built definition (``BCDef`` /
``PointMassDef`` / ``EqualDOFDef`` / etc.), it resolves the def against
the FEMData via :class:`FEMDataSource` and returns a new :class:`FEMData`
with the resulting records appended via the broker's ``with_*``
transforms.

Scope
-----
The chain-phase router covers def types whose resolution can be answered
out of the broker without a live gmsh session:

* ``BCDef`` → one :class:`SPRecord` per restrained DOF per node.
* ``PointMassDef`` → one :class:`MassRecord` per resolved node.
* ``PointLoadDef`` → one :class:`NodalLoadRecord` per resolved node.

Plus the node-only interface-bridging constraints (Compose v1.1-A):

* ``EqualDOFDef`` → one :class:`NodePairRecord` per co-located pair.
* ``RigidLinkDef`` → one :class:`NodePairRecord` per slave node.
* ``RigidDiaphragmDef`` → one :class:`NodeGroupRecord`.

These use :class:`FEMDataSource.nodes_for` to resolve master/slave
labels into node-id sets, then run the same pure-Python
:class:`~apeGmsh._kernel.resolvers._constraint_resolver.ConstraintResolver`
the build-phase path uses.  No gmsh state required.

Deferred (v1.1-A.2)
~~~~~~~~~~~~~~~~~~~
``EmbeddedDef`` and ``TiedContactDef`` need element-connectivity and
face-connectivity queries respectively — those require new
:class:`FEMDataSource` methods (and, for tied-contact, synthesis of
face connectivity from volume elements).  They continue to fall back
to the bump-counter pattern; the def is stored on
``constraint_defs`` but not applied to ``_fem`` until a build-phase
``get_fem_data()`` re-extraction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


def try_chain_phase_route(session, defn) -> bool:
    """Composite-side entry point: try routing ``defn`` against ``session._fem``.

    Returns ``True`` when the def was successfully resolved + applied to
    ``session._fem`` (which is replaced with a new snapshot in place);
    returns ``False`` when ``defn``'s shape exceeds this minimum-viable
    router's coverage or the session is in build phase (``_fem is None``).
    The caller falls back to the bump-counter pattern in the False case.

    Catches :class:`KeyError` from name resolution so a missing target
    surfaces only at extraction time (preserves backward-compat with
    the existing build-phase behaviour where defs may reference names
    that get created later in the session).
    """
    fem = getattr(session, "_fem", None)
    if fem is None:
        return False
    try:
        new_fem = route_def_to_fem(fem, defn)
    except (KeyError, TypeError):
        # Name resolution failure or unsupported target shape — fall
        # back to bump-counter.  In chain phase the def silently won't
        # be applied (documented limitation of this slice).
        return False
    if new_fem is None:
        return False
    session._fem = new_fem
    # Mark the cache fresh so the next ``get_fem_data()`` returns the
    # updated snapshot without an extraction attempt (the broker is
    # already in sync via the ``with_*`` transform).
    if hasattr(session, "_mark_fem_fresh"):
        session._mark_fem_fresh()
    return True


def route_def_to_fem(fem: "FEMData", defn) -> "FEMData | None":
    """Try to resolve ``defn`` directly into ``fem`` via ``with_*``.

    Returns a new :class:`FEMData` with the records appended on
    success; returns ``None`` when ``defn``'s shape needs geometry-
    aware reduction this router does not cover yet (the caller falls
    back to the bump-counter pattern).

    No exceptions escape — resolution failures (e.g. unknown target
    name) propagate as :class:`KeyError` from the underlying source
    adapter; type-mismatch falls through as ``None``.
    """
    from apeGmsh._kernel.defs.constraints import (
        BCDef,
        EqualDOFDef,
        RigidDiaphragmDef,
        RigidLinkDef,
    )
    from apeGmsh._kernel.defs.masses import PointMassDef
    from apeGmsh._kernel.defs.loads import PointLoadDef
    from apeGmsh._kernel.records._loads import (
        NodalLoadRecord,
        SPRecord,
    )
    from apeGmsh._kernel.records._masses import MassRecord
    from ._source import FEMDataSource

    source = FEMDataSource(fem)

    # ── BCDef → SPRecord ──────────────────────────────────────────
    if isinstance(defn, BCDef):
        node_ids = _resolve_target_to_node_ids(source, defn.target)
        new_fem = fem
        for nid in sorted(node_ids):
            for d_idx, mask in enumerate(defn.dofs):
                if mask != 1:
                    continue
                rec = SPRecord(
                    name=defn.name,
                    node_id=int(nid),
                    dof=d_idx + 1,
                    value=0.0,
                    is_homogeneous=True,
                )
                new_fem = new_fem.with_load(rec)
        return new_fem

    # ── PointMassDef → MassRecord ────────────────────────────────
    if isinstance(defn, PointMassDef):
        node_ids = _resolve_target_to_node_ids(source, defn.target)
        rot = defn.rotational or (0.0, 0.0, 0.0)
        dofs = defn.dofs or [1, 2, 3]
        mass_by_dof = {d: defn.mass for d in dofs}
        new_fem = fem
        for nid in sorted(node_ids):
            translational = tuple(
                float(mass_by_dof.get(i + 1, 0.0)) for i in range(3)
            )
            mass6 = (
                translational[0], translational[1], translational[2],
                float(rot[0]), float(rot[1]), float(rot[2]),
            )
            rec = MassRecord(
                name=defn.name,
                node_id=int(nid),
                mass=mass6,
            )
            new_fem = new_fem.with_mass(rec)
        return new_fem

    # ── PointLoadDef → NodalLoadRecord ───────────────────────────
    if isinstance(defn, PointLoadDef):
        node_ids = _resolve_target_to_node_ids(source, defn.target)
        f = defn.force_xyz
        m = defn.moment_xyz
        force_xyz = (
            (float(f[0]), float(f[1]), float(f[2]))
            if f is not None and any(abs(float(v)) > 0.0 for v in f)
            else None
        )
        moment_xyz = (
            (float(m[0]), float(m[1]), float(m[2]))
            if m is not None and any(abs(float(v)) > 0.0 for v in m)
            else None
        )
        if force_xyz is None and moment_xyz is None:
            # Zero load — nothing to apply.
            return fem
        new_fem = fem
        for nid in sorted(node_ids):
            rec = NodalLoadRecord(
                pattern=defn.pattern,
                name=defn.name,
                node_id=int(nid),
                force_xyz=force_xyz,
                moment_xyz=moment_xyz,
            )
            new_fem = new_fem.with_load(rec)
        return new_fem

    # ── EqualDOFDef → NodePairRecord ──────────────────────────────
    if isinstance(defn, EqualDOFDef):
        return _route_equal_dof(fem, source, defn)

    # ── RigidLinkDef → NodePairRecord ─────────────────────────────
    if isinstance(defn, RigidLinkDef):
        return _route_rigid_link(fem, source, defn)

    # ── RigidDiaphragmDef → NodeGroupRecord ───────────────────────
    if isinstance(defn, RigidDiaphragmDef):
        return _route_rigid_diaphragm(fem, source, defn)

    # ── Unsupported def shape ─────────────────────────────────────
    return None


def _route_equal_dof(fem: "FEMData", source, defn) -> "FEMData":
    """Resolve ``EqualDOFDef`` against ``fem`` and append node-pair records.

    Build-phase parity — uses the same
    :class:`~apeGmsh._kernel.resolvers._constraint_resolver.ConstraintResolver`
    that the meshed path uses; only the master/slave node sets come
    from :meth:`FEMDataSource.nodes_for` rather than from
    ``g.parts.build_node_map``.
    """
    from apeGmsh._kernel.resolvers._constraint_resolver import (
        ConstraintResolver,
    )

    master_nodes = {
        int(t) for t in source.nodes_for(defn.master_label)
    }
    slave_nodes = {
        int(t) for t in source.nodes_for(defn.slave_label)
    }
    resolver = _build_resolver(fem, ConstraintResolver)
    records = resolver.resolve_equal_dof(defn, master_nodes, slave_nodes)
    new_fem = fem
    for rec in records:
        new_fem = new_fem.with_constraint(rec)
    return new_fem


def _route_rigid_link(fem: "FEMData", source, defn) -> "FEMData":
    """Resolve ``RigidLinkDef`` against ``fem`` and append node-pair records."""
    from apeGmsh._kernel.resolvers._constraint_resolver import (
        ConstraintResolver,
    )

    master_nodes = {
        int(t) for t in source.nodes_for(defn.master_label)
    }
    slave_nodes = {
        int(t) for t in source.nodes_for(defn.slave_label)
    }
    resolver = _build_resolver(fem, ConstraintResolver)
    records = resolver.resolve_rigid_link(defn, master_nodes, slave_nodes)
    new_fem = fem
    for rec in records:
        new_fem = new_fem.with_constraint(rec)
    return new_fem


def _route_rigid_diaphragm(fem: "FEMData", source, defn) -> "FEMData":
    """Resolve ``RigidDiaphragmDef`` against ``fem`` and append a group record.

    Mirrors :meth:`ConstraintsComposite._resolve_diaphragm` — the
    diaphragm gathers ``master_label`` ∪ ``slave_label`` nodes and
    filters by plane proximity inside the resolver.
    """
    from apeGmsh._kernel.resolvers._constraint_resolver import (
        ConstraintResolver,
    )

    m = {int(t) for t in source.nodes_for(defn.master_label)}
    s = {int(t) for t in source.nodes_for(defn.slave_label)}
    all_in = m | s
    resolver = _build_resolver(fem, ConstraintResolver)
    record = resolver.resolve_rigid_diaphragm(defn, all_in)
    # The build-phase resolver returns an empty NodeGroupRecord (no
    # master_node, no slaves) when no nodes survive the plane filter
    # — preserve that behaviour by not appending an empty record (it
    # would otherwise produce a phantom constraint with master_node=0).
    if not record.slave_nodes and record.master_node == 0:
        return fem
    return fem.with_constraint(record)


def _build_resolver(fem: "FEMData", resolver_cls):
    """Build a :class:`ConstraintResolver` from FEMData arrays.

    The resolver only needs ``node_tags`` + ``node_coords`` for the
    three node-only constraint paths (no element connectivity).  Pass
    them in directly so we avoid the cost of materialising the full
    element table from the broker.
    """
    return resolver_cls(
        node_tags=np.asarray(fem.nodes.ids, dtype=np.int64),
        node_coords=np.asarray(fem.nodes.coords, dtype=np.float64),
    )


def _resolve_target_to_node_ids(source, target) -> np.ndarray:
    """Coerce a ``target`` field (str, int, list of ...) to int64 ids.

    Strings route through :meth:`FEMDataSource.nodes_for`; raw ints
    (or lists thereof) are taken at face value.  This is intentionally
    narrow — defs with mesh-selection sentinels, ``(dim, tag)`` lists,
    or other complex targets fall through to ``None`` at the caller.
    """
    if isinstance(target, str):
        return source.nodes_for(target)
    if isinstance(target, int):
        return np.array([int(target)], dtype=np.int64)
    if isinstance(target, (list, tuple)):
        ints: list[int] = []
        for x in target:
            if isinstance(x, int):
                ints.append(int(x))
            elif (
                isinstance(x, tuple) and len(x) == 2
                and all(isinstance(y, int) for y in x)
            ):
                # (dim, tag) — for nodes only (dim=0).  Otherwise we
                # do not have enough info to resolve here; let the
                # caller fall back to bump-counter.
                d, t = x
                if d == 0:
                    ints.append(int(t))
                else:
                    raise TypeError(
                        f"chain-phase routing: (dim, tag) target with "
                        f"dim={d} requires element-connectivity walk "
                        f"not yet wired in chain phase."
                    )
            else:
                raise TypeError(
                    f"chain-phase routing: unsupported target element "
                    f"{x!r} (type {type(x).__name__})."
                )
        return np.array(sorted(set(ints)), dtype=np.int64)
    raise TypeError(
        f"chain-phase routing: unsupported target type "
        f"{type(target).__name__}; pass a label/PG name (str), a bare "
        f"node id (int), or a list of those."
    )
