"""
_fem_factory — Factory functions for FEMData construction.
===========================================================

Implements ``FEMData.from_gmsh()`` and ``FEMData.from_msh()`` by
orchestrating the raw Gmsh extraction helpers from ``_fem_extract.py``
and splitting resolved records into node-side vs element-side
sub-composites.
"""

from __future__ import annotations

import logging

import numpy as np

from ._element_types import ElementGroup, make_type_info
from ._fem_extract import (
    extract_raw, extract_physical_groups, extract_labels,
    extract_partitions,
)
from ._group_set import PhysicalGroupSet, LabelSet

_log = logging.getLogger(__name__)


# =====================================================================
# Constraint splitting
# =====================================================================

def _split_constraints(records: list) -> tuple[list, list]:
    """Split resolved constraint records into node-level and surface-level."""
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        InterpolationRecord, SurfaceCouplingRecord,
    )

    node_recs = []
    surface_recs = []

    for rec in records:
        if isinstance(rec, (NodePairRecord, NodeGroupRecord,
                            NodeToSurfaceRecord)):
            node_recs.append(rec)
        elif isinstance(rec, (InterpolationRecord,
                              SurfaceCouplingRecord)):
            surface_recs.append(rec)
        else:
            _log.warning(
                "Unknown constraint record type %s (kind=%r) — "
                "placed in node-level set as fallback.",
                type(rec).__name__, getattr(rec, 'kind', '?'))
            node_recs.append(rec)

    return node_recs, surface_recs


# =====================================================================
# Load splitting
# =====================================================================

def _split_loads(records: list) -> tuple[list, list, list]:
    """Split resolved load records into nodal, element, and SP."""
    from apeGmsh._kernel.records._loads import NodalLoadRecord, ElementLoadRecord, SPRecord

    nodal = []
    element = []
    sp = []

    for rec in records:
        if isinstance(rec, NodalLoadRecord):
            nodal.append(rec)
        elif isinstance(rec, ElementLoadRecord):
            element.append(rec)
        elif isinstance(rec, SPRecord):
            sp.append(rec)
        else:
            _log.warning(
                "Unknown load record type %s (kind=%r) — "
                "placed in nodal set as fallback.",
                type(rec).__name__, getattr(rec, 'kind', '?'))
            nodal.append(rec)

    return nodal, element, sp


# =====================================================================
# Constraint-connected node collection
# =====================================================================

def _collect_constraint_nodes(
    node_constraints: list,
    surface_constraints: list,
    nodal_loads: list,
    sp_records: list,
    mass_records: list,
) -> set[int]:
    """Collect every mesh node ID referenced by resolved BCs."""
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        InterpolationRecord, SurfaceCouplingRecord,
    )

    ids: set[int] = set()

    for rec in node_constraints:
        if isinstance(rec, NodePairRecord):
            ids.add(rec.master_node)
            ids.add(rec.slave_node)
        elif isinstance(rec, NodeGroupRecord):
            ids.add(rec.master_node)
            ids.update(rec.slave_nodes)
        elif isinstance(rec, NodeToSurfaceRecord):
            ids.add(rec.master_node)
            ids.update(rec.slave_nodes)

    for rec in surface_constraints:
        if isinstance(rec, InterpolationRecord):
            ids.add(rec.slave_node)
            ids.update(rec.master_nodes)
        elif isinstance(rec, SurfaceCouplingRecord):
            ids.update(rec.master_nodes)
            ids.update(rec.slave_nodes)

    for rec in nodal_loads:
        ids.add(rec.node_id)
    for rec in sp_records:
        ids.add(rec.node_id)
    for rec in mass_records:
        ids.add(rec.node_id)

    return ids


# =====================================================================
# Build ElementGroup dict from raw groups
# =====================================================================

def _build_element_groups(raw_groups: dict[int, dict]) -> dict[int, ElementGroup]:
    """Convert raw extraction groups to ElementGroup objects."""
    result: dict[int, ElementGroup] = {}
    for etype_code, info in raw_groups.items():
        type_info = make_type_info(
            code=etype_code,
            gmsh_name=info['gmsh_name'],
            dim=info['dim'],
            order=info['order'],
            npe=info['npe'],
            count=len(info['ids']),
        )
        result[etype_code] = ElementGroup(
            element_type=type_info,
            ids=info['ids'],
            connectivity=info['conn'],
        )
    return result


def _flat_connectivity(groups: dict[int, ElementGroup]) -> np.ndarray:
    """Temporary flat connectivity for resolver kwargs.

    Resolvers receive this but never read it.  If types have
    different npe, pad shorter rows with -1.
    """
    if not groups:
        return np.empty((0, 0), dtype=np.int64)

    blocks = [g.connectivity for g in groups.values() if len(g) > 0]
    if not blocks:
        return np.empty((0, 0), dtype=np.int64)

    max_npe = max(b.shape[1] for b in blocks)
    padded = []
    for b in blocks:
        if b.shape[1] < max_npe:
            pad = np.full(
                (b.shape[0], max_npe - b.shape[1]), -1, dtype=np.int64)
            padded.append(np.hstack([b, pad]))
        else:
            padded.append(b)
    return np.vstack(padded)


def _flat_elem_tags(groups: dict[int, ElementGroup]) -> np.ndarray:
    """Concatenated element tags from all groups."""
    if not groups:
        return np.array([], dtype=np.int64)
    return np.concatenate([g.ids for g in groups.values()])


# =====================================================================
# Per-node ndf populator (shell-to-solid coupling feature, S1b)
# =====================================================================
#
# Explicit-only contract: every node that needs an ``ndf`` must be
# covered by a declaration on ``session.node_ndf`` — either a targeted
# :meth:`set` call or the blanket :meth:`set_default`.  apeGmsh
# refuses to infer ``ndf`` from element class; the populator writes
# ``0`` (sentinel) for any node not covered by a declaration, and
# :meth:`NodeComposite.ndf_for` raises ``LookupError`` on the
# sentinel with a message that names both fixes.


def _populate_node_ndf(
    node_ids: np.ndarray,
    session,
    node_map: dict | None,
) -> np.ndarray | None:
    """Build the per-node ``ndf`` ``int8`` array from session defs.

    Returns ``None`` when no ``NodeNDFComposite`` is wired (e.g. a
    session built without it, or the ``from_msh`` path with no
    session at all).  Returns a sentinel-initialised array (``0`` =
    undeclared) when the composite exists but carries no defs — this
    distinguishes "broker came from a session that knows about ndf
    but the user hadn't declared anything" from "broker has no
    ndf metadata channel at all".

    Resolver behaviour follows the dimensional resolution contract:
    a missing target raises ``KeyError``, which propagates through
    this function and up out of ``get_fem_data()`` rather than being
    silently swallowed.
    """
    if session is None:
        return None
    composite = getattr(session, "node_ndf", None)
    if composite is None:
        return None

    n = int(np.asarray(node_ids).size)
    ndf = np.zeros(n, dtype=np.int8)

    # Empty composite: return the sentinel array so the broker still
    # advertises "ndf was wired but undeclared" — every ndf_for() will
    # raise the helpful LookupError.
    if not composite._defs:
        return ndf

    id_to_idx = {int(t): i for i, t in enumerate(np.asarray(node_ids))}

    targeted = composite._targeted_defs()
    default = composite._default_def()

    # Apply targeted defs in declaration order; later defs win on
    # overlap (consistent with the imperative ``set_*`` semantics
    # documented on NodeNDFComposite).
    for defn in targeted:
        target_nodes = _resolve_ndf_target_to_node_ids(
            session, defn.target, node_map=node_map,
        )
        if not target_nodes:
            _log.warning(
                "g.node_ndf.set target %r resolved to zero nodes — "
                "the declaration has no effect.",
                defn.target,
            )
            continue
        ndf_value = np.int8(defn.ndf)
        missing: list[int] = []
        for tag in target_nodes:
            idx = id_to_idx.get(int(tag))
            if idx is None:
                # Tag references a node not in the broker — typically
                # an orphan filtered out by ``remove_orphans=True``.
                # Log the cohort once at the end rather than failing
                # loud per node: the resolver itself already failed
                # loud at the target level (no-such-label / wrong-dim);
                # an orphan-filtered node is a downstream artifact of
                # the orphan-removal pass the user explicitly opted
                # into.
                missing.append(int(tag))
                continue
            ndf[idx] = ndf_value
        if missing:
            _log.warning(
                "g.node_ndf.set target %r resolved to %d node(s) "
                "absent from the FEM broker (likely orphan-filtered): "
                "%s%s",
                defn.target,
                len(missing),
                missing[:5],
                "..." if len(missing) > 5 else "",
            )

    # Fill remaining sentinels with the default, if one was declared.
    if default is not None:
        ndf[ndf == 0] = np.int8(default.ndf)

    return ndf


def _resolve_ndf_target_to_node_ids(
    session,
    target,
    *,
    node_map: dict | None,
) -> set[int]:
    """Resolve a ``NodeNDFDef`` target to a set of mesh node IDs.

    Mirrors the precedence chain used by
    :meth:`LoadsComposite._target_nodes` and
    :meth:`MassesComposite`: raw DimTag list / mesh-selection
    sentinel / label / PG / part label.  ``KeyError`` from the
    shared resolver propagates per the dimensional resolution
    contract — a missing target must fail loud, not silently
    bind zero nodes.
    """
    import gmsh

    from apeGmsh.core._resolution import resolve_target

    # Part-label fast path — consistent with NodeComposite._resolve_one_target
    # and LoadsComposite._target_nodes.
    parts = getattr(session, "parts", None)
    if (isinstance(target, str)
            and parts is not None
            and target in getattr(parts, "_instances", {})
            and node_map is not None
            and target in node_map):
        return {int(n) for n in node_map[target]}

    dts = resolve_target(
        session, target, source="auto",
        not_found_prefix="g.node_ndf.set target",
        noun="node_ndf",
    )

    # Mesh-selection sentinel — fail loud on missing set, mirrors
    # LoadsComposite._target_nodes (the silent ``return set()`` was
    # promoted to KeyError there for the same reason).
    if dts and isinstance(dts[0], tuple) and dts[0] and dts[0][0] == "__ms__":
        _, dim, tag = dts[0]
        ms = getattr(session, "mesh_selection", None)
        info = None if ms is None else ms._sets.get((dim, tag))
        if info is None:
            raise KeyError(
                f"g.node_ndf.set target {target!r} resolved to a "
                f"mesh-selection sentinel ('__ms__', {dim}, {tag}), "
                f"but that set is absent from g.mesh_selection._sets."
            )
        return {int(n) for n in info.get("node_ids", [])}

    nodes: set[int] = set()
    for d, t in dts:
        nt, _, _ = gmsh.model.mesh.getNodes(
            dim=int(d), tag=int(t),
            includeBoundary=True, returnParametricCoord=False,
        )
        nodes.update(int(n) for n in nt)
    return nodes


# =====================================================================
# Shared extraction core
# =====================================================================

def _extract_mesh_core(dim: int | None):
    """Shared extraction: raw arrays → element groups + PGs + labels.

    Returns
    -------
    tuple
        (node_tags, node_coords, elem_tags, groups,
         used_tags, physical, labels, partitions)
    """
    raw = extract_raw(dim=dim)

    node_tags   = raw['node_tags']
    node_coords = raw['node_coords']
    used_tags   = raw['used_tags']

    groups = _build_element_groups(raw['groups'])
    elem_tags = _flat_elem_tags(groups)

    physical = PhysicalGroupSet(extract_physical_groups())
    labels   = LabelSet(extract_labels())
    partitions = extract_partitions(dim)

    return (node_tags, node_coords, elem_tags, groups,
            used_tags, physical, labels, partitions)


# =====================================================================
# from_gmsh
# =====================================================================

def _from_gmsh(
    cls,
    *,
    dim: int | None,
    session=None,
    ndf: int = 6,
    remove_orphans: bool = False,
):
    """Build a FEMData from the live Gmsh session.

    Parameters
    ----------
    cls : type
        The FEMData class.
    dim : int or None
        Element dimension to extract.  None = all dims.
    session : apeGmsh session, optional
        Provides constraints, loads, masses composites.
    ndf : int
        DOFs per node for load/mass padding.
    remove_orphans : bool
        If True, remove orphan nodes.  Default False.
    """
    from .FEMData import (
        NodeComposite, ElementComposite, MeshInfo, _compute_bandwidth,
    )

    # ── 1. Extract ────────────────────────────────────────────
    (node_tags, node_coords, elem_tags, groups,
     used_tags, physical, labels, partitions) = _extract_mesh_core(dim)

    node_ids = np.asarray(node_tags, dtype=int)
    node_coords_all = node_coords

    # ── 2. Resolve BCs ────────────────────────────────────────
    node_constraints: list = []
    surface_constraints: list = []
    nodal_loads: list = []
    element_loads: list = []
    sp_records: list = []
    mass_records: list = []

    if session is not None:
        parts_comp = getattr(session, "parts", None)
        node_map = None
        face_map = None
        if (parts_comp is not None
                and getattr(parts_comp, "_instances", None)):
            # No broad swallow: a node/face-map build failure must
            # surface with its real cause rather than degrade to
            # None and resurface later as a vaguer constraint error.
            node_map = parts_comp.build_node_map(
                node_ids, node_coords_all)
            face_map = parts_comp.build_face_map(node_map)

        # Build temp flat connectivity for resolver kwargs
        flat_conn = _flat_connectivity(groups)
        resolve_kw = dict(
            elem_tags=elem_tags,
            connectivity=flat_conn,
            node_map=node_map,
            face_map=face_map,
        )

        # Constraints / loads / masses.
        #
        # These resolve() calls are deliberately NOT wrapped in a
        # broad ``except Exception: log.warning`` swallow.  The
        # resolvers raise precise, actionable ValueError/KeyError when
        # a reference is wrong-dimension, multi-dim, unresolved, or
        # would otherwise silently bind the wrong node/face set.  A
        # structural model that silently drops a tie / load / mass is
        # worse than one that errors — get_fem_data() must fail loud
        # so the user fixes the model, not discover it post-analysis.
        constraints_comp = getattr(session, "constraints", None)
        if (constraints_comp is not None
                and getattr(constraints_comp, "constraint_defs", None)):
            all_constraints = constraints_comp.resolve(
                node_ids, node_coords_all, **resolve_kw)
            node_constraints, surface_constraints = \
                _split_constraints(all_constraints)

        loads_comp = getattr(session, "loads", None)
        if (loads_comp is not None
                and getattr(loads_comp, "load_defs", None)):
            all_loads = loads_comp.resolve(
                node_ids, node_coords_all, **resolve_kw)
            nodal_loads, element_loads, sp_records = _split_loads(all_loads)

        # Single-point constraints declared via g.constraints.bc().
        # Kept separate from constraints_comp.resolve() above: BCDefs
        # have no master/slave and resolve to homogeneous SPRecords in
        # fem.nodes.sp, not to fem.nodes.constraints.  Independent of
        # the constraint_defs guard so a BC-only model still resolves.
        if (constraints_comp is not None
                and getattr(constraints_comp, "_bc_defs", None)):
            sp_records.extend(
                constraints_comp.resolve_bcs(
                    node_ids, node_map=node_map))

        masses_comp = getattr(session, "masses", None)
        if (masses_comp is not None
                and getattr(masses_comp, "mass_defs", None)):
            mass_records = masses_comp.resolve(
                node_ids, node_coords_all,
                ndf=ndf, **resolve_kw)

    # ── 3. Orphan filtering ───────────────────────────────────
    if remove_orphans:
        protected = _collect_constraint_nodes(
            node_constraints, surface_constraints,
            nodal_loads, sp_records, mass_records,
        )
        node_ids, node_coords_all = _filter_orphans(
            node_tags, node_coords, used_tags, protected)

    # ── 4. Build MeshInfo ─────────────────────────────────────
    type_list = [g.element_type for g in groups.values()]
    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=int(sum(len(g) for g in groups.values())),
        bandwidth=_compute_bandwidth(groups),
        types=type_list,
    )

    # ── 5. Build composites ───────────────────────────────────
    # If the session carries a parts registry, snapshot its
    # label -> {mesh-node-ids} and label -> {mesh-element-ids} maps
    # now so fem.nodes.get(target=part_label) and
    # fem.elements.get(target=part_label) can resolve without
    # needing a live Gmsh session later.
    part_node_map: dict[str, set[int]] = {}
    part_elem_map: dict[str, set[int]] = {}
    if session is not None:
        parts = getattr(session, "parts", None)
        if parts is not None and getattr(parts, "_instances", None):
            import gmsh  # local import — gmsh is alive during factory
            # Fail loud, mirroring the constraint-path policy above
            # (the same build_node_map call there is deliberately not
            # swallowed): a part-map build failure must surface with
            # its real cause, not degrade to an empty map that
            # resurfaces later as a misleading "part not found".
            part_node_map = parts.build_node_map(
                node_ids, node_coords_all,
            ) or {}
            # Element map: iterate each part instance's DimTags and
            # ask Gmsh for the elements on each entity (the registry
            # has no element-map builder today).
            #
            # inst.entities can hold tags that fragment_all() / boolean
            # ops have retagged out of existence (skill pitfall 7.6 —
            # OCC renumbers entities; the coordinate-based node map is
            # the robust contract).  Skip absent entities *explicitly*
            # by pre-filtering against the live model — not via a
            # blanket ``except`` — so a genuine getElements failure on
            # an entity the model DOES have still fails loud.
            present = set(gmsh.model.getEntities())
            for label, inst in parts._instances.items():
                e_ids: set[int] = set()
                for d in sorted(inst.entities.keys(), reverse=True):
                    for t in inst.entities[d]:
                        if (int(d), int(t)) not in present:
                            continue
                        _, etags_list, _ = gmsh.model.mesh.getElements(
                            int(d), int(t))
                        for arr in etags_list:
                            e_ids.update(int(x) for x in arr)
                if e_ids:
                    part_elem_map[label] = e_ids

    # ── 5b. Per-node ndf populator (S1b explicit-only) ────────
    # Walks session.node_ndf defs and writes the int8 ndf vector.
    # KeyError from a missing resolver target propagates per the
    # dimensional resolution contract (test_resolution_contract.py).
    node_ndf = _populate_node_ndf(node_ids, session, node_map=part_node_map)

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords_all,
        physical=physical,
        labels=labels,
        constraints=node_constraints or None,
        loads=nodal_loads or None,
        sp=sp_records or None,
        masses=mass_records or None,
        partitions=partitions or None,
        part_node_map=part_node_map or None,
        ndf=node_ndf,
    )
    elements = ElementComposite(
        groups=groups,
        physical=physical,
        labels=labels,
        constraints=surface_constraints or None,
        loads=element_loads or None,
        partitions=partitions or None,
        part_elem_map=part_elem_map or None,
    )

    # Stamp the post-extraction flag on the session so any further
    # ``g.node_ndf.set(...)`` call after this point warns (the
    # broker is cached; later defs won't appear in this FEMData).
    if session is not None:
        try:
            session._fem_built = True
        except AttributeError:
            pass  # not a vanilla session — skip silently

    # ── 6. Snapshot mesh selections ───────────────────────────
    ms_store = None
    if session is not None:
        ms_comp = getattr(session, "mesh_selection", None)
        if ms_comp is not None and len(ms_comp) > 0:
            # Fail loud (see the note above): a snapshot failure must
            # not silently drop every mesh-selection set and resurface
            # later as a misleading "selection not found".
            ms_store = ms_comp._snapshot()

    return cls(
        nodes=nodes,
        elements=elements,
        info=info,
        mesh_selection=ms_store,
    )


# =====================================================================
# Orphan filtering
# =====================================================================

def _filter_orphans(
    node_tags: np.ndarray,
    node_coords: np.ndarray,
    used_tags: set[int],
    protected: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove orphan nodes, optionally protecting some."""
    keep = np.isin(node_tags, list(used_tags))

    if protected:
        also_keep = np.isin(node_tags, list(protected))
        keep = keep | also_keep

    orphan_mask = ~keep
    n_orphans = int(orphan_mask.sum())
    if n_orphans > 0:
        orphan_tags = node_tags[orphan_mask]
        orphan_coords = node_coords[orphan_mask]
        detail = ", ".join(
            f"{int(t)} ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})"
            for t, c in zip(orphan_tags[:20], orphan_coords[:20])
        )
        _log.warning(
            "%d orphan node(s) removed (not connected to any element). "
            "First: [%s]%s",
            n_orphans,
            detail,
            f" ... (+{n_orphans - 20} more)" if n_orphans > 20 else "")

    node_ids = np.asarray(node_tags[keep], dtype=int)
    node_coords_filtered = node_coords[keep]
    return node_ids, node_coords_filtered


# =====================================================================
# from_msh
# =====================================================================

def _from_msh(
    cls,
    *,
    path: str,
    dim: int | None = 2,
    remove_orphans: bool = False,
):
    """Build a FEMData from an external ``.msh`` file."""
    import gmsh
    from .FEMData import (
        NodeComposite, ElementComposite, MeshInfo, _compute_bandwidth,
    )
    from .._session import _gmsh_acquire, _gmsh_release

    _gmsh_acquire()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(path))

        (node_tags, node_coords, elem_tags, groups,
         used_tags, physical, labels, partitions) = _extract_mesh_core(dim)

        if remove_orphans:
            node_ids, node_coords = _filter_orphans(
                node_tags, node_coords, used_tags)
        else:
            node_ids = np.asarray(node_tags, dtype=int)

        type_list = [g.element_type for g in groups.values()]
        info = MeshInfo(
            n_nodes=len(node_ids),
            n_elems=int(sum(len(g) for g in groups.values())),
            bandwidth=_compute_bandwidth(groups),
            types=type_list,
        )

        # Per-node ndf — leave at ``None`` (no NodeNDFComposite in
        # the ``from_msh`` path).  Per the locked S2 design, the hash
        # fold in ``_femdata_hash`` skips when ``_ndf is None`` OR is
        # all-sentinel, so this stays hash-symmetric with
        # ``from_gmsh`` of a model with no declarations (which folds
        # an all-zero array → also skipped).  The emit-side falls
        # back to the apeSees envelope ``ndf=K`` via the
        # ``try/except LookupError`` pattern, so ``from_msh``-built
        # broker FEMs emit correctly under any uniform-ndf model.
        nodes = NodeComposite(
            node_ids=node_ids, node_coords=node_coords,
            physical=physical, labels=labels,
            partitions=partitions or None,
        )
        elements = ElementComposite(
            groups=groups,
            physical=physical, labels=labels,
            partitions=partitions or None,
        )
    finally:
        _gmsh_release()

    return cls(nodes=nodes, elements=elements, info=info)
