"""
_fem_extract — Shared FEM data extraction from a live Gmsh session.
====================================================================

Standalone helper functions that pull node/element/physical-group data
straight from the ``gmsh`` API and package it for FEMData construction.

Used by:

* ``Mesh.get_fem_data()``   — the normal apeGmsh pipeline
* ``MshLoader.from_msh()``  — loading an external ``.msh`` file

Having the logic here avoids duplicating Gmsh API calls across modules.
"""

from __future__ import annotations

import gmsh
import numpy as np
from numpy import ndarray


# =====================================================================
# Raw extraction (dict of arrays — no FEMData yet)
# =====================================================================

def extract_raw(dim: int | None = 2) -> dict:
    """Pull raw FEM arrays from the current Gmsh session.

    Parameters
    ----------
    dim : int or None
        Element dimension to extract (1 = lines, 2 = tri/quad,
        3 = tet/hex).  ``None`` extracts all dimensions.

    Returns
    -------
    dict
        Keys: ``node_tags``, ``node_coords``, ``groups``,
        ``elem_tags``, ``used_tags``.

        ``groups`` is ``dict[int, dict]`` keyed by Gmsh element type
        code, with each value containing ``'ids'``, ``'conn'``,
        ``'gmsh_name'``, ``'dim'``, ``'order'``, ``'npe'``.
    """
    # --- nodes (full mesh) ---
    raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
    node_tags   = np.array(raw_tags, dtype=np.int64)
    node_coords = np.array(raw_coords).reshape(-1, 3)

    # --- elements, grouped by type ---
    gmsh_dim = -1 if dim is None else dim
    elem_types, elem_tags_list, node_tags_list = \
        gmsh.model.mesh.getElements(dim=gmsh_dim, tag=-1)

    groups: dict[int, dict] = {}
    all_elem_tags: list[int] = []

    for etype, etags, enodes in zip(
        elem_types, elem_tags_list, node_tags_list
    ):
        etype = int(etype)
        props = gmsh.model.mesh.getElementProperties(etype)
        # props: (name, dim, order, n_nodes, local_coords, n_primary)
        npe = props[3]
        ids = np.array(etags, dtype=np.int64)
        conn = np.array(enodes, dtype=np.int64).reshape(-1, npe)

        if etype in groups:
            # Multiple entities of the same type — concatenate
            prev = groups[etype]
            prev['ids'] = np.concatenate([prev['ids'], ids])
            prev['conn'] = np.concatenate([prev['conn'], conn])
        else:
            groups[etype] = {
                'ids':       ids,
                'conn':      conn,
                'gmsh_name': props[0],
                'dim':       int(props[1]),
                'order':     int(props[2]),
                'npe':       npe,
            }

        all_elem_tags.extend(int(t) for t in etags)

    # --- used_tags from connectivity (all dims >= 1) ---
    used_tags: set[int] = set()
    for g in groups.values():
        if g['conn'].size > 0:
            used_tags.update(g['conn'].ravel().tolist())
    # Also include nodes from other structural dims (1D, 2D, 3D)
    if dim is not None:
        for d in range(1, 4):
            if d == dim:
                continue
            _, _, d_enodes = gmsh.model.mesh.getElements(dim=d, tag=-1)
            for enodes_block in d_enodes:
                used_tags.update(int(n) for n in enodes_block)

    return {
        'node_tags':   node_tags,
        'node_coords': node_coords,
        'groups':      groups,
        'elem_tags':   all_elem_tags,
        'used_tags':   used_tags,
    }


# =====================================================================
# Shared per-type extraction for PGs and labels
# =====================================================================

def _extract_entity_elements(pg_dim: int, pg_tag: int) -> tuple[
    ndarray, dict[int, dict],
]:
    """Extract element IDs and per-type groups for a physical group.

    Returns
    -------
    (flat_elem_ids, groups_dict)
        flat_elem_ids: ndarray of all element IDs (all types combined)
        groups_dict: ``{etype_code: {'ids': ndarray, 'conn': ndarray}}``
    """
    entity_tags = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
    flat_ids: list[int] = []
    groups: dict[int, dict] = {}

    for ent_tag in entity_tags:
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
            dim=pg_dim, tag=int(ent_tag))
        for etype, etags_arr, enodes in zip(
            etypes, etags_list, enodes_list
        ):
            etype = int(etype)
            props = gmsh.model.mesh.getElementProperties(etype)
            npe = props[3]
            ids = np.array(etags_arr, dtype=np.int64)
            conn = np.array(enodes, dtype=np.int64).reshape(-1, npe)
            flat_ids.extend(int(t) for t in etags_arr)

            if etype in groups:
                prev = groups[etype]
                prev['ids'] = np.concatenate([prev['ids'], ids])
                prev['conn'] = np.concatenate([prev['conn'], conn])
            else:
                groups[etype] = {
                    'ids':       ids,
                    'conn':      conn,
                    'gmsh_name': props[0],
                    'dim':       int(props[1]),
                    'order':     int(props[2]),
                    'npe':       npe,
                }

    flat_arr = np.array(flat_ids, dtype=np.int64) if flat_ids else np.array(
        [], dtype=np.int64)
    return flat_arr, groups


# =====================================================================
# Physical-group snapshot
# =====================================================================

def extract_physical_groups() -> dict[tuple[int, int], dict]:
    """Snapshot every physical group in the current Gmsh session.

    Returns
    -------
    dict
        ``{(dim, pg_tag): {'name', 'node_ids', 'node_coords',
        'element_ids'?, 'groups'?}}``

        ``groups`` is ``dict[int, dict]`` keyed by element type code
        (same shape as in ``extract_raw``).
    """
    pg_data: dict[tuple[int, int], dict] = {}

    from apeGmsh.core.Labels import is_label_pg

    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        if is_label_pg(name):
            continue
        pg_node_tags, pg_coords = \
            gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)

        entry: dict = {
            'name':        name,
            'node_ids':    np.array(pg_node_tags, dtype=np.int64),
            'node_coords': np.array(pg_coords).reshape(-1, 3),
        }

        if pg_dim >= 1:
            elem_ids, groups = _extract_entity_elements(pg_dim, pg_tag)
            if len(elem_ids) > 0:
                entry['element_ids'] = elem_ids
                entry['groups'] = groups

        pg_data[(pg_dim, pg_tag)] = entry

    return pg_data


# =====================================================================
# Label snapshot (Tier 1 — _label: prefixed PGs)
# =====================================================================

def extract_labels() -> dict[tuple[int, int], dict]:
    """Snapshot every label (``_label:``-prefixed PG) in the session.

    Same structure as :func:`extract_physical_groups` but captures
    only label PGs and strips the ``_label:`` prefix from names.

    Returns
    -------
    dict
        ``{(dim, pg_tag): {'name', 'node_ids', 'node_coords',
        'element_ids'?, 'groups'?}}``
    """
    from apeGmsh.core.Labels import LABEL_PREFIX, is_label_pg

    lbl_data: dict[tuple[int, int], dict] = {}

    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        if not is_label_pg(name):
            continue

        clean_name = name[len(LABEL_PREFIX):]

        pg_node_tags, pg_coords = \
            gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)

        entry: dict = {
            'name':        clean_name,
            'node_ids':    np.array(pg_node_tags, dtype=np.int64),
            'node_coords': np.array(pg_coords).reshape(-1, 3),
        }

        if pg_dim >= 1:
            elem_ids, groups = _extract_entity_elements(pg_dim, pg_tag)
            if len(elem_ids) > 0:
                entry['element_ids'] = elem_ids
                entry['groups'] = groups

        lbl_data[(pg_dim, pg_tag)] = entry

    return lbl_data


# =====================================================================
# Partition snapshot
# =====================================================================

def extract_partitions(dim: int | None) -> dict[int, dict]:
    """Per-partition node/element membership from the live Gmsh session.

    Parameters
    ----------
    dim : int or None
        Element dimension to collect.  ``None`` collects all dims.

    Returns
    -------
    dict[int, dict]
        ``{partition_id: {'node_ids': ndarray, 'element_ids': ndarray}}``
        Empty dict if the mesh is not partitioned.
    """
    try:
        n_parts = gmsh.model.getNumberOfPartitions()
    except (AttributeError, Exception):
        return {}
    if n_parts == 0:
        return {}

    part_elems: dict[int, list[int]] = {}
    part_nodes: dict[int, set[int]] = {}

    dims = [dim] if dim is not None else list(range(4))
    for d in dims:
        for ent_dim, ent_tag in gmsh.model.getEntities(d):
            try:
                pparts = gmsh.model.getPartitions(ent_dim, ent_tag)
            except Exception:
                continue
            if len(pparts) == 0:
                continue

            etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
                ent_dim, ent_tag)
            for _etype, etags, enodes in zip(etypes, etags_list, enodes_list):
                for p in pparts:
                    p = int(p)
                    part_elems.setdefault(p, []).extend(
                        int(t) for t in etags)
                    part_nodes.setdefault(p, set()).update(
                        int(n) for n in enodes)

    result: dict[int, dict] = {}
    for p in sorted(part_elems):
        result[p] = {
            'element_ids': np.array(sorted(part_elems[p]),
                                    dtype=np.int64),
            'node_ids': np.array(sorted(part_nodes.get(p, [])),
                                 dtype=np.int64),
        }
    return result
