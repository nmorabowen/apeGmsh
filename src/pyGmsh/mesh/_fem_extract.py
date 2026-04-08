"""
_fem_extract — Shared FEM data extraction from a live Gmsh session.
====================================================================

Standalone helper functions that pull node/element/physical-group data
straight from the ``gmsh`` API and package it into a :class:`FEMData`.

Used by:

* ``Mesh.get_fem_data()``   — the normal pyGmsh pipeline
* ``MshLoader.from_msh()``  — loading an external ``.msh`` file

Having the logic here avoids duplicating ~90 lines of Gmsh API calls
across two modules.
"""

from __future__ import annotations

import gmsh
import numpy as np
from numpy import ndarray


# =====================================================================
# Raw extraction (dict of arrays — no FEMData yet)
# =====================================================================

def extract_raw(dim: int = 2) -> dict:
    """
    Pull raw FEM arrays from the current Gmsh session.

    Parameters
    ----------
    dim : int
        Element dimension to extract (1 = lines, 2 = tri/quad,
        3 = tet/hex).

    Returns
    -------
    dict
        Keys: ``node_tags``, ``node_coords``, ``connectivity``,
        ``elem_tags``, ``elem_type_codes``, ``elem_type_info``,
        ``used_tags``.
    """
    # --- nodes (full mesh) ---
    raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
    node_tags   = np.array(raw_tags, dtype=np.int64)
    node_coords = np.array(raw_coords).reshape(-1, 3)

    # --- elements of requested dimension ---
    elem_types, elem_tags_list, node_tags_list = \
        gmsh.model.mesh.getElements(dim=dim, tag=-1)

    conn_blocks:     list[ndarray] = []
    elem_tags:       list[int]     = []
    elem_type_codes: list[int]     = []
    elem_type_info:  dict[int, tuple] = {}

    for etype, etags, enodes in zip(
        elem_types, elem_tags_list, node_tags_list
    ):
        props = gmsh.model.mesh.getElementProperties(int(etype))
        # props: (name, dim, order, n_nodes, local_coords, n_primary)
        npe = props[3]
        conn_blocks.append(
            np.array(enodes, dtype=np.int64).reshape(-1, npe)
        )
        n_this = len(etags)
        elem_tags.extend(int(t) for t in etags)
        elem_type_codes.extend([int(etype)] * n_this)
        elem_type_info[int(etype)] = (props[0], props[1], props[3])

    connectivity = (
        np.vstack(conn_blocks) if conn_blocks
        else np.empty((0, 0), dtype=int)
    )

    # --- used_tags from connectivity (all dims >= 1) ---
    # A node is "used" if it appears in any element's connectivity
    # (lines, surfaces, volumes).  Nodes only referenced by dim=0
    # point elements (arc centres, construction points) are orphans.
    used_tags: set[int] = set()
    # Start with the target-dim connectivity we already extracted
    if connectivity.size > 0:
        used_tags.update(connectivity.ravel().tolist())
    # Also include nodes from other structural dims (1D, 2D, 3D)
    # in case they don't appear in the target-dim connectivity.
    for d in range(1, 4):
        if d == dim:
            continue  # already covered above
        _, _, d_enodes = gmsh.model.mesh.getElements(dim=d, tag=-1)
        for enodes_block in d_enodes:
            used_tags.update(int(n) for n in enodes_block)

    return {
        'node_tags':       node_tags,
        'node_coords':     node_coords,
        'connectivity':    connectivity,
        'elem_tags':       elem_tags,
        'elem_type_codes': elem_type_codes,
        'elem_type_info':  elem_type_info,
        'used_tags':       used_tags,
    }


# =====================================================================
# Physical-group snapshot
# =====================================================================

def extract_physical_groups() -> dict[tuple[int, int], dict]:
    """
    Snapshot every physical group in the current Gmsh session.

    Returns
    -------
    dict
        ``{(dim, pg_tag): {'name', 'node_ids', 'node_coords',
        'element_ids'?, 'connectivity'?}}``
    """
    pg_data: dict[tuple[int, int], dict] = {}

    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        pg_node_tags, pg_coords = \
            gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)

        entry: dict = {
            'name':        name,
            'node_ids':    np.array(pg_node_tags, dtype=np.int64),
            'node_coords': np.array(pg_coords).reshape(-1, 3),
        }

        # Capture element data for dim >= 1 physical groups
        if pg_dim >= 1:
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(
                pg_dim, pg_tag
            )
            pg_elem_tags:   list[int]     = []
            pg_conn_blocks: list[ndarray] = []

            for ent_tag in entity_tags:
                etypes, etags_list, enodes_list = \
                    gmsh.model.mesh.getElements(
                        dim=pg_dim, tag=int(ent_tag)
                    )
                for etype, etags_arr, enodes in zip(
                    etypes, etags_list, enodes_list
                ):
                    props = gmsh.model.mesh.getElementProperties(
                        int(etype)
                    )
                    npe = props[3]
                    pg_elem_tags.extend(int(t) for t in etags_arr)
                    pg_conn_blocks.append(
                        np.array(enodes, dtype=np.int64).reshape(-1, npe)
                    )

            if pg_conn_blocks:
                entry['element_ids']  = np.array(
                    pg_elem_tags, dtype=np.int64
                )
                entry['connectivity'] = np.vstack(pg_conn_blocks)

        pg_data[(pg_dim, pg_tag)] = entry

    return pg_data


# =====================================================================
# Full FEMData assembly
# =====================================================================

def build_fem_data(dim: int = 2):
    """
    Extract a complete :class:`FEMData` from the live Gmsh session.

    Combines :func:`extract_raw` and :func:`extract_physical_groups`
    into a single self-contained ``FEMData`` object.

    Parameters
    ----------
    dim : int
        Element dimension to extract.

    Returns
    -------
    FEMData
    """
    from .FEMData import (
        FEMData, MeshInfo, PhysicalGroupSet, _compute_bandwidth,
    )

    raw = extract_raw(dim=dim)

    node_tags    = raw['node_tags']
    node_coords  = raw['node_coords']
    connectivity = raw['connectivity']
    elem_tags    = np.asarray(raw['elem_tags'], dtype=int)
    used_tags    = raw['used_tags']

    # Filter to only nodes referenced by elements
    mask      = np.isin(node_tags, list(used_tags))
    n_total   = len(node_tags)
    node_ids  = np.asarray(node_tags[mask], dtype=int)
    node_coords = node_coords[mask]
    n_orphans = n_total - len(node_ids)
    if n_orphans > 0:
        orphan_tags = node_tags[~mask]
        print(
            f"[FEMData] WARNING: {n_orphans} orphan node(s) removed "
            f"(not connected to any element). "
            f"Tags: {orphan_tags.tolist()[:20]}"
            + (f" ... (+{n_orphans - 20} more)" if n_orphans > 20 else "")
        )

    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=len(elem_tags),
        bandwidth=_compute_bandwidth(connectivity),
    )

    physical = PhysicalGroupSet(extract_physical_groups())

    result = FEMData(
        node_ids=node_ids,
        node_coords=node_coords,
        element_ids=elem_tags,
        connectivity=connectivity,
        info=info,
        physical=physical,
    )

    return result
