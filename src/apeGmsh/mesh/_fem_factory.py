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

from ._fem_extract import (
    extract_raw, extract_physical_groups, extract_labels,
    extract_partitions,
)
from ._element_types import ElementGroup, make_type_info

_log = logging.getLogger(__name__)
from ._group_set import PhysicalGroupSet, LabelSet


# =====================================================================
# Constraint splitting
# =====================================================================

def _split_constraints(records: list) -> tuple[list, list]:
    """Split resolved constraint records into node-level and surface-level."""
    from apeGmsh.solvers.Constraints import (
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
    from apeGmsh.solvers.Loads import NodalLoadRecord, ElementLoadRecord, SPRecord

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
    from apeGmsh.solvers.Constraints import (
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
            try:
                node_map = parts_comp.build_node_map(
                    node_ids, node_coords_all)
                face_map = parts_comp.build_face_map(node_map)
            except Exception:
                node_map = None
                face_map = None

        # Build temp flat connectivity for resolver kwargs
        flat_conn = _flat_connectivity(groups)
        resolve_kw = dict(
            elem_tags=elem_tags,
            connectivity=flat_conn,
            node_map=node_map,
            face_map=face_map,
        )

        # Constraints
        constraints_comp = getattr(session, "constraints", None)
        if (constraints_comp is not None
                and getattr(constraints_comp, "constraint_defs", None)):
            try:
                all_constraints = constraints_comp.resolve(
                    node_ids, node_coords_all, **resolve_kw)
                node_constraints, surface_constraints = \
                    _split_constraints(all_constraints)
            except Exception as exc:
                _log.warning("Constraint resolve failed: %s", exc)

        # Loads
        loads_comp = getattr(session, "loads", None)
        if (loads_comp is not None
                and getattr(loads_comp, "load_defs", None)):
            try:
                all_loads = loads_comp.resolve(
                    node_ids, node_coords_all, **resolve_kw)
                nodal_loads, element_loads, sp_records = _split_loads(all_loads)
            except Exception as exc:
                _log.warning("Load resolve failed: %s", exc)

        # Masses
        masses_comp = getattr(session, "masses", None)
        if (masses_comp is not None
                and getattr(masses_comp, "mass_defs", None)):
            try:
                mass_records = masses_comp.resolve(
                    node_ids, node_coords_all,
                    ndf=ndf, **resolve_kw)
            except Exception as exc:
                _log.warning("Mass resolve failed: %s", exc)

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
    )
    elements = ElementComposite(
        groups=groups,
        physical=physical,
        labels=labels,
        constraints=surface_constraints or None,
        loads=element_loads or None,
        partitions=partitions or None,
    )

    # ── 6. Snapshot mesh selections ───────────────────────────
    ms_store = None
    if session is not None:
        ms_comp = getattr(session, "mesh_selection", None)
        if ms_comp is not None and len(ms_comp) > 0:
            try:
                ms_store = ms_comp._snapshot()
            except Exception:
                pass

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
        gmsh.finalize()

    return cls(nodes=nodes, elements=elements, info=info)
