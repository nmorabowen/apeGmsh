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

import hashlib

import gmsh
import numpy as np
from numpy import ndarray


# =====================================================================
# Connectivity integrity guard
# =====================================================================

def _validate_connectivity(
    groups: dict[int, dict],
    node_tags: np.ndarray,
) -> None:
    """Fail loud if any element references a node absent from the mesh.

    Every connectivity tag must be a real node returned by
    ``gmsh.model.mesh.getNodes()``.  A reference to node tag ``0`` — or
    any other tag not in the node set — means the mesh is corrupt, and
    passing it downstream produces the cryptic OpenSees abort
    ``Domain::addElement - ... no Node 0 exists in the domain``.

    The dominant cause is a **global 3-D recombine**: Gmsh's
    ``mesh.recombine()`` (exposed as ``g.mesh.structured.recombine()``)
    is a 2-D surface operation.  Running it while 3-D tetrahedra exist
    deletes interior nodes but leaves the tets that referenced them
    pointing at node ``0``.  We catch that here, at extraction, rather
    than letting the solver fail with an opaque message.
    """
    if node_tags.size == 0:
        return

    valid = node_tags  # already int64 from getNodes
    bad_by_type: list[tuple[str, np.ndarray]] = []
    total_bad_refs = 0
    has_zero = False

    for info in groups.values():
        conn = info['conn']
        if conn.size == 0:
            continue
        uniq = np.unique(conn)
        missing = uniq[~np.isin(uniq, valid)]
        if missing.size:
            bad_by_type.append((info['gmsh_name'], missing))
            total_bad_refs += int(np.isin(conn, missing).sum())
            if (missing == 0).any():
                has_zero = True

    if not bad_by_type:
        return

    detail = "; ".join(
        f"{name}: {sorted(int(t) for t in tags)[:10]}"
        f"{' ...' if tags.size > 10 else ''}"
        for name, tags in bad_by_type
    )
    zero_hint = ""
    if has_zero:
        zero_hint = (
            " Node tag 0 is never a valid node -- this almost always "
            "means the mesh was corrupted by a global 3-D recombine. "
            "`g.mesh.structured.recombine()` (Gmsh's `mesh.recombine()`) "
            "is a 2-D surface operation; running it while 3-D tetrahedra "
            "exist deletes interior nodes and leaves the tets that "
            "referenced them pointing at node 0. To build a hexahedral "
            "volume mesh, use a structured/transfinite setup "
            "(`g.mesh.structured.set_transfinite(...)`) instead of the "
            "global recombine."
        )
    raise ValueError(
        f"Corrupt mesh: {total_bad_refs} element-connectivity "
        f"reference(s) point at node tag(s) absent from the mesh node "
        f"set ({detail})." + zero_hint
    )


# =====================================================================
# Raw extraction (dict of arrays — no FEMData yet)
# =====================================================================

def _model_entity(dim: int, tag: int) -> tuple[int, int]:
    """The model entity ``(dim, tag)`` was cut from, following ``getParent``.

    A partitioned mesh moves elements onto partition entities; Gmsh's
    ``getParent`` names the entity each one was cut from, and
    ``(-1, -1)`` on a model entity. The chain is followed to its end; a
    model entity is its own result.
    """
    cur_dim, cur_tag = dim, tag
    seen = {(cur_dim, cur_tag)}
    while True:
        p_dim, p_tag = gmsh.model.getParent(cur_dim, cur_tag)
        p_dim, p_tag = int(p_dim), int(p_tag)
        if p_dim < 0:
            break
        if (p_dim, p_tag) in seen:
            raise RuntimeError(
                f"Gmsh getParent cycles at entity ({p_dim}, {p_tag}) "
                f"from ({dim}, {tag}); cannot resolve the elements' "
                f"source entity."
            )
        seen.add((p_dim, p_tag))
        cur_dim, cur_tag = p_dim, p_tag
    return cur_dim, cur_tag


def _is_partition_artifact(dim: int, tag: int) -> bool:
    """True when ``(dim, tag)`` holds elements partitioning added.

    Two kinds of entity carry elements that are not elements of the
    mesh, and both are skipped at extraction so a partitioned FEMData
    holds each mesh element once, exactly as the unpartitioned mesh
    does (#2582):

    * a **partition-interface** entity
      (``Mesh.PartitionCreateTopology``, on by default): Gmsh meshes
      the cut between two partitions with new lower-dimensional
      elements (lines between surfaces, triangles between volumes,
      points where a cut meets a curve). Its parent is the entity of
      *higher* dimension the cut lies in, so the chain ends on a model
      entity of another dimension.
    * a **ghost** entity (``Mesh.PartitionCreateGhostCells``): it holds
      copies, under the same element ids, of the neighbouring
      partition's elements, which are already extracted from the
      partition that owns them.

    A model entity (no parent) is never an artifact, whatever its tag:
    a MSH2 file may classify real elements on elementary tag ``0``.
    """
    if gmsh.model.getType(dim, tag).startswith("Ghost"):
        return True
    m_dim, m_tag = _model_entity(dim, tag)
    return (m_dim, m_tag) != (dim, tag) and m_dim != dim


def _source_entity(dim: int, tag: int) -> int:
    """The source-entity tag of elements classified on ``(dim, tag)``.

    The tag of the model entity ``(dim, tag)`` was cut from
    (:func:`_model_entity`) when that entity has dimension ``dim``,
    else ``-1``. Extraction skips partition-interface and ghost
    entities (:func:`_is_partition_artifact`) before asking, so ``-1``
    remains only for a tag that is not positive (a MSH2 file may
    classify elements on elementary tag ``0``); a ghost or interface
    entity still answers ``-1`` here.
    """
    if gmsh.model.getType(dim, tag).startswith("Ghost"):
        return -1
    m_dim, m_tag = _model_entity(dim, tag)
    return m_tag if m_dim == dim and m_tag > 0 else -1


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
        code (ascending), with each value containing ``'ids'``,
        ``'conn'``, ``'entity'``, ``'gmsh_name'``, ``'dim'``,
        ``'order'``, ``'npe'``.  ``'entity'`` is the ``(E,) int32``
        source-entity stream: row-parallel to ``'ids'``, the
        tag of the model entity of the element's own dimension the
        element is classified on, a partition entity resolved to its
        parent (:func:`_source_entity`).

        Partition-interface and ghost elements are not extracted
        (:func:`_is_partition_artifact`): a partitioned mesh yields the
        same element ids as the unpartitioned one.
    """
    # --- nodes (full mesh) ---
    raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
    node_tags   = np.array(raw_tags, dtype=np.int64)
    node_coords = np.array(raw_coords).reshape(-1, 3)

    # --- elements, per model entity, grouped by type ---
    # One ``getElements(d, tag)`` per entity rather than one global
    # ``getElements(dim, -1)``: Gmsh builds the global call by walking
    # the same entities, in ``getEntities`` order, inside each ascending
    # type code. Concatenating the per-entity blocks in that order
    # reproduces its ids and connectivity byte for byte, and the loop
    # learns each block's entity at no extra Gmsh call per element.
    gmsh_dim = -1 if dim is None else dim
    blocks: dict[int, list[tuple[ndarray, ndarray, int]]] = {}
    for ent_dim, ent_tag in gmsh.model.getEntities(gmsh_dim):
        if _is_partition_artifact(int(ent_dim), int(ent_tag)):
            continue
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
            dim=ent_dim, tag=ent_tag)
        if len(etypes) == 0:
            continue
        source = _source_entity(int(ent_dim), int(ent_tag))
        for etype, etags, enodes in zip(etypes, etags_list, enodes_list):
            if len(etags) == 0:
                continue
            blocks.setdefault(int(etype), []).append((etags, enodes, source))

    groups: dict[int, dict] = {}
    all_elem_tags: list[int] = []

    for etype in sorted(blocks):
        props = gmsh.model.mesh.getElementProperties(etype)
        # props: (name, dim, order, n_nodes, local_coords, n_primary)
        npe = props[3]
        parts = blocks.pop(etype)
        if len(parts) == 1:
            etags, enodes, source = parts[0]
            ids = np.array(etags, dtype=np.int64)
            conn = np.array(enodes, dtype=np.int64).reshape(-1, npe)
            entity = np.full(ids.size, source, dtype=np.int32)
        else:
            ids = np.concatenate(
                [np.asarray(p[0], dtype=np.int64) for p in parts])
            conn = np.concatenate(
                [np.asarray(p[1], dtype=np.int64) for p in parts]
            ).reshape(-1, npe)
            entity = np.repeat(
                np.array([p[2] for p in parts], dtype=np.int32),
                [len(p[0]) for p in parts],
            )
        del parts
        groups[etype] = {
            'ids':       ids,
            'conn':      conn,
            'entity':    entity,
            'gmsh_name': props[0],
            'dim':       int(props[1]),
            'order':     int(props[2]),
            'npe':       npe,
        }
        all_elem_tags.extend(ids.tolist())

    # --- integrity guard: every connectivity tag must be a real node ---
    _validate_connectivity(groups, node_tags)

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

        Partition-interface and ghost entities are skipped
        (:func:`_is_partition_artifact`), as in :func:`extract_raw`:
        each element belongs to the one partition that owns it.
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
            if _is_partition_artifact(int(ent_dim), int(ent_tag)):
                continue
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


# =====================================================================
# Producing-model identity
# =====================================================================

def gmsh_model_identity() -> tuple[str, str]:
    """Name and cheap fingerprint of the current Gmsh model.

    Recorded on a FEMData snapshot at extraction and recomputed before
    a raw ``(dim, tag)`` selection asks live Gmsh, so the snapshot never
    answers from a different model.  The name alone cannot identify the
    model: two sessions may share one (the default is the running
    script's stem) and Gmsh accepts duplicate names.  The fingerprint
    folds the entity list, the max
    node and element tags, and the nodes on geometry points (tags and
    coordinates); every call is O(1) or O(entities), independent of
    mesh size.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(gmsh.model.getEntities(), dtype=np.int64).tobytes())
    h.update(np.asarray(
        [gmsh.model.mesh.getMaxNodeTag(), gmsh.model.mesh.getMaxElementTag()],
        dtype=np.int64).tobytes())
    tags, coords, _ = gmsh.model.mesh.getNodes(
        dim=0, tag=-1, returnParametricCoord=False)
    h.update(np.asarray(tags, dtype=np.int64).tobytes())
    h.update(np.asarray(coords, dtype=np.float64).tobytes())
    return gmsh.model.getCurrent(), h.hexdigest()
