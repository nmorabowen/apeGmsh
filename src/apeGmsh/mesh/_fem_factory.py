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

from .._kernel.records._node_ndf import IMPLICIT_NDF_BY_DIM
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
# Per-node ndf populator (shell-to-solid coupling feature)
# =====================================================================


def _build_implicit_ndf(
    node_ids: np.ndarray,
    groups: dict[int, ElementGroup],
) -> np.ndarray:
    """Compute the per-node implicit ``ndf`` vector.

    For each node, take the max of :data:`IMPLICIT_NDF_BY_DIM` over
    the dim of every incident element group.  Nodes with no incident
    element (orphans that survive the orphan filter, e.g. because a
    BC pins them) fall back to the largest implicit value in the
    table — currently 6 — so a downstream solver that needs DOFs at
    a constraint-only node won't silently lose them.

    Returns an ``int8`` array aligned 1:1 with ``node_ids``.
    """
    n = int(np.asarray(node_ids).size)
    # Initialise to 0 so the max-reduction over incident groups
    # cleanly accumulates.  Nodes that never appear in any group
    # are patched to the fallback at the end.
    ndf = np.zeros(n, dtype=np.int8)

    # Build a tag -> index map so we can scatter quickly.
    id_to_idx = {int(t): i for i, t in enumerate(np.asarray(node_ids))}

    for grp in groups.values():
        if grp.connectivity.size == 0:
            continue
        dim_ndf = IMPLICIT_NDF_BY_DIM.get(int(grp.dim))
        if dim_ndf is None:
            continue
        # Unique node ids in this group's connectivity.
        unique_tags = np.unique(np.asarray(grp.connectivity).reshape(-1))
        for tag in unique_tags:
            idx = id_to_idx.get(int(tag))
            if idx is None:
                continue
            if dim_ndf > int(ndf[idx]):
                ndf[idx] = np.int8(dim_ndf)

    # Patch any node that didn't appear in any group's connectivity.
    # The most defensible default is the largest table value so the
    # full DOF set is available for whatever BC pins them.  Tests
    # that build degenerate fixtures with no elements will see this.
    fallback = max(IMPLICIT_NDF_BY_DIM.values()) if IMPLICIT_NDF_BY_DIM else 6
    ndf[ndf == 0] = np.int8(fallback)
    return ndf


def _apply_explicit_ndf_overrides(
    ndf: np.ndarray,
    ndf_source: np.ndarray,
    node_ids: np.ndarray,
    session,
    node_map: dict | None = None,
) -> None:
    """Apply ``g.model.set_node_ndf(...)`` overrides in-place.

    Walks the session's ``model._node_ndf_defs`` list (each entry is
    a :class:`NodeNDFDef`); resolves each def's target to a node-id
    set via the shared loads/masses resolver
    (:func:`apeGmsh.core._resolution.resolve_target`) and overwrites
    the implicit value at every resolved index, marking the source
    byte ``1`` (= explicit).
    """
    model = getattr(session, "model", None)
    if model is None:
        return
    defs = getattr(model, "_node_ndf_defs", None) or []
    if not defs:
        return

    id_to_idx = {int(t): i for i, t in enumerate(np.asarray(node_ids))}

    # Local helper that turns a NodeNDFDef target into a node-id set.
    # Reuses the same precedence chain (label → PG → part) used by
    # loads/masses via the shared resolver, plus the parts node-map
    # fast path so part labels resolve without a live Gmsh probe.
    import gmsh

    from apeGmsh.core._resolution import resolve_target

    parts = getattr(session, "parts", None)

    def _resolve_to_node_ids(target) -> set[int]:
        # Part-label fast path — consistent with NodeComposite._resolve_one_target.
        if (isinstance(target, str)
                and parts is not None
                and target in getattr(parts, "_instances", {})
                and node_map is not None
                and target in node_map):
            return {int(n) for n in node_map[target]}
        # Mesh-selection sentinel + raw DimTag list + label/PG/part union
        # all flow through the shared resolver.
        dts = resolve_target(
            session, target, source="auto",
            not_found_prefix="set_node_ndf target",
            noun="node_ndf",
        )
        if dts and isinstance(dts[0], tuple) and dts[0] and dts[0][0] == "__ms__":
            _, dim, tag = dts[0]
            ms = getattr(session, "mesh_selection", None)
            info = None if ms is None else ms._sets.get((dim, tag))
            if info is None:
                raise KeyError(
                    f"set_node_ndf target {target!r} resolved to a mesh "
                    f"selection that is absent from g.mesh_selection."
                )
            return {int(n) for n in info.get("node_ids", [])}

        nodes: set[int] = set()
        for d, t in dts:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(d), tag=int(t),
                    includeBoundary=True, returnParametricCoord=False,
                )
                nodes.update(int(n) for n in nt)
            except Exception:
                pass
        return nodes

    # Apply each def in declaration order; later defs win on overlap.
    for defn in defs:
        try:
            target_nodes = _resolve_to_node_ids(defn.target)
        except KeyError:
            # Re-raise: silent drop here would resurface much later as a
            # misleading "wrong ndf on node X" debugging session.
            raise
        if not target_nodes:
            _log.warning(
                "set_node_ndf target %r resolved to zero nodes — "
                "the override has no effect.",
                defn.target,
            )
            continue
        for tag in target_nodes:
            idx = id_to_idx.get(int(tag))
            if idx is None:
                # Node target points at a tag that isn't in the broker
                # — likely an orphan filtered out.  Skip silently;
                # there is no node to assign ndf to.
                continue
            ndf[idx] = np.int8(defn.ndf)
            ndf_source[idx] = np.int8(1)


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

    # ── 5b. Per-node ndf (shell-to-solid coupling feature) ────
    # Compute the implicit vector from element-class dim, then apply
    # any explicit ``g.model.set_node_ndf(...)`` overrides.  Lives
    # here (after orphan filtering, before composite construction) so
    # the arrays are aligned 1:1 with the final ``node_ids``.
    node_ndf = _build_implicit_ndf(node_ids, groups)
    node_ndf_source = np.zeros(node_ndf.shape, dtype=np.int8)
    if session is not None:
        _apply_explicit_ndf_overrides(
            node_ndf, node_ndf_source, node_ids, session,
            node_map=node_map if 'node_map' in locals() else None,
        )

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
        ndf_source=node_ndf_source,
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

    gmsh.initialize()
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

        # Per-node implicit ndf (no session for explicit overrides
        # in from_msh — explicit ndf only flows through from_gmsh).
        node_ndf = _build_implicit_ndf(node_ids, groups)
        node_ndf_source = np.zeros(node_ndf.shape, dtype=np.int8)

        nodes = NodeComposite(
            node_ids=node_ids, node_coords=node_coords,
            physical=physical, labels=labels,
            partitions=partitions or None,
            ndf=node_ndf,
            ndf_source=node_ndf_source,
        )
        elements = ElementComposite(
            groups=groups,
            physical=physical, labels=labels,
            partitions=partitions or None,
        )
    finally:
        gmsh.finalize()

    return cls(nodes=nodes, elements=elements, info=info)
