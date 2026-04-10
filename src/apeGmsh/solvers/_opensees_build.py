"""
_build — internal implementation of :meth:`OpenSees.build`.

Split out of ``OpenSees.py`` because the build routine is ~280 lines
of validation + mesh extraction logic that does not benefit from
living inside the class body.  It's invoked from
``OpenSees.build()`` as a free function that takes the parent
:class:`OpenSees` instance and mutates its internal tables.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import gmsh
import numpy as np
import pandas as pd

from ._element_specs import _ETYPE_INFO, _ELEM_REGISTRY

if TYPE_CHECKING:
    from .OpenSees import OpenSees


def run_build(ops: "OpenSees") -> None:
    """
    Extract the active gmsh mesh and construct all model tables on *ops*.

    Performs full validation:

    * element types match the physical-group topology
    * material references point to the correct registry
    * geomTransf is declared for beam elements
    * ``ndf`` / ``ndm`` are compatible with each element spec
    * warns when higher-order gmsh nodes are downgraded to first-order

    Must be called after ``g.mesh.generation.generate()`` and all
    declarations.
    """
    # ── 1. Global node numbering ──────────────────────────────────────
    # Use getNodes() with no args → returns the unique node cache.
    # getNodes(dim=-1, ..., includeBoundary=True) can return the same
    # physical node multiple times (once per entity it touches),
    # producing duplicates that break the model.
    raw_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    if len(raw_tags) == 0:
        raise RuntimeError(
            "OpenSees.build(): no mesh nodes found — "
            "call g.mesh.generation.generate() first."
        )
    coords_arr = np.array(coords_flat).reshape(-1, 3)
    ops._node_map = {
        int(gt): ops_id
        for ops_id, gt in enumerate(raw_tags, start=1)
    }
    ops._nodes_df = pd.DataFrame({
        'ops_id': np.arange(1, len(raw_tags) + 1, dtype=np.int64),
        'x'     : coords_arr[:, 0],
        'y'     : coords_arr[:, 1],
        'z'     : coords_arr[:, 2],
    }).set_index('ops_id')
    ops._log(f"build(): {len(raw_tags)} nodes mapped")

    # ── 2. Sequential registry tags ───────────────────────────────────
    ops._nd_mat_tags  = {n: i for i, n in enumerate(ops._nd_materials,  start=1)}
    ops._uni_mat_tags = {n: i for i, n in enumerate(ops._uni_materials, start=1)}
    ops._sec_tags     = {n: i for i, n in enumerate(ops._sections,      start=1)}
    ops._transf_tags  = {n: i for i, n in enumerate(ops._geom_transfs,  start=1)}

    # ── 3. Elements per assigned physical group ───────────────────────
    elem_rows: list[dict] = []
    ops_elem_id = 1
    _warned_ho: set[str] = set()  # track high-order warnings per PG

    for pg_name, asgn in ops._elem_assignments.items():
        ops_type    = asgn["ops_type"]
        mat_name    = asgn["material"]
        transf_name = asgn["geom_transf"]
        hint_dim    = asgn["dim"]
        extra       = asgn["extra"]
        spec        = _ELEM_REGISTRY[ops_type]  # already validated in assign

        # ── ndm / ndf check ───────────────────────────────────────────
        if ops._ndm not in spec.ndm_ok:
            raise ValueError(
                f"elements.assign({pg_name!r}, {ops_type!r}): "
                f"ndm={ops._ndm} is not valid for this element type "
                f"(allowed: {sorted(spec.ndm_ok)}). "
                f"Call set_model() with the correct ndm."
            )
        if ops._ndf not in spec.ndf_ok:
            raise ValueError(
                f"elements.assign({pg_name!r}, {ops_type!r}): "
                f"ndf={ops._ndf} is not valid for this element type "
                f"(allowed: {sorted(spec.ndf_ok)}). "
                f"Call set_model() with the correct ndf."
            )

        # ── material reference check ──────────────────────────────────
        mat_tag = sec_tag = None
        if spec.mat_family == "nd":
            if mat_name is None or mat_name not in ops._nd_materials:
                raise ValueError(
                    f"elements.assign({pg_name!r}, {ops_type!r}): "
                    f"requires an nDMaterial; "
                    f"{mat_name!r} not found in materials.add_nd_material registry. "
                    f"Available: {sorted(ops._nd_materials)}"
                )
            mat_tag = ops._nd_mat_tags[mat_name]
        elif spec.mat_family == "uni":
            if mat_name is None or mat_name not in ops._uni_materials:
                raise ValueError(
                    f"elements.assign({pg_name!r}, {ops_type!r}): "
                    f"requires a uniaxialMaterial; "
                    f"{mat_name!r} not found in materials.add_uni_material registry. "
                    f"Available: {sorted(ops._uni_materials)}"
                )
            mat_tag = ops._uni_mat_tags[mat_name]
        elif spec.mat_family == "section":
            if mat_name is None or mat_name not in ops._sections:
                raise ValueError(
                    f"elements.assign({pg_name!r}, {ops_type!r}): "
                    f"requires a section; "
                    f"{mat_name!r} not found in materials.add_section registry. "
                    f"Available: {sorted(ops._sections)}"
                )
            sec_tag = ops._sec_tags[mat_name]
        # spec.mat_family == "none": no material needed

        # ── geomTransf check ──────────────────────────────────────────
        transf_tag = None
        if spec.needs_transf:
            if transf_name is None or transf_name not in ops._geom_transfs:
                raise ValueError(
                    f"elements.assign({pg_name!r}, {ops_type!r}): "
                    f"requires a geomTransf; "
                    f"{transf_name!r} not found in elements.add_geom_transf registry. "
                    f"Available: {sorted(ops._geom_transfs)}"
                )
            transf_tag = ops._transf_tags[transf_name]

        # ── physical group lookup ─────────────────────────────────────
        expected_dim = hint_dim if hint_dim is not None else spec.expected_pg_dim
        pg_dim, pg_tag = ops._find_pg(pg_name, expected_dim)
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)

        # ── extract elements ──────────────────────────────────────────
        slots = spec.get_slots(ops._ndm)

        for ent_tag in entity_tags:
            etypes, elem_tags_list, enodes_list = \
                gmsh.model.mesh.getElements(dim=pg_dim, tag=ent_tag)

            for etype, elem_tags_arr, enodes in zip(
                etypes, elem_tags_list, enodes_list
            ):
                # topology compatibility check
                if etype not in spec.gmsh_etypes:
                    raise ValueError(
                        f"elements.assign({pg_name!r}, {ops_type!r}): "
                        f"physical group contains gmsh element type {etype} "
                        f"which is incompatible with {ops_type!r}. "
                        f"Expected gmsh types: {sorted(spec.gmsh_etypes)}. "
                        f"Check that the physical-group dimension and mesh "
                        f"algorithm match the element formulation."
                    )

                n_corner, _ = _ETYPE_INFO[etype]
                n_per       = len(enodes) // len(elem_tags_arr)
                reorder     = spec.node_reorder.get(etype, tuple(range(n_corner)))

                # high-order warning (issued once per PG)
                if n_per > n_corner and pg_name not in _warned_ho:
                    warnings.warn(
                        f"OpenSees.build(): physical group {pg_name!r} "
                        f"contains {n_per}-node elements (gmsh type {etype}), "
                        f"but {ops_type!r} uses only {n_corner} corner nodes. "
                        f"Mid-side nodes are discarded — the exported model "
                        f"is first-order only.",
                        UserWarning, stacklevel=2,
                    )
                    _warned_ho.add(pg_name)

                for k in range(len(elem_tags_arr)):
                    raw_ns    = enodes[k * n_per:(k + 1) * n_per]
                    corner_ns = [raw_ns[i] for i in reorder]
                    ops_nodes = tuple(
                        ops._node_map[int(ni)]
                        for ni in corner_ns
                        if int(ni) in ops._node_map
                    )
                    if len(ops_nodes) != n_corner:
                        continue  # unmapped node — skip silently

                    elem_rows.append({
                        'ops_id'    : ops_elem_id,
                        'gmsh_id'   : int(elem_tags_arr[k]),
                        'ops_type'  : ops_type,
                        'pg_name'   : pg_name,
                        'mat_name'  : mat_name,
                        'mat_tag'   : mat_tag,
                        'sec_tag'   : sec_tag,
                        'transf_tag': transf_tag,
                        'n_nodes'   : n_corner,
                        'nodes'     : ops_nodes,
                        'slots'     : slots,
                        'extra'     : extra,
                    })
                    ops_elem_id += 1

    # ── 3b. Filter to model-active nodes only ─────────────────────────
    # Gmsh's getNodes(dim=-1) returns ALL mesh nodes including
    # geometric vertex points that are not part of any element.
    # We prune down to nodes referenced by at least one element OR by a
    # boundary-condition / load physical group, then renumber sequentially.
    reverse_map = {v: k for k, v in ops._node_map.items()}
    connected_gmsh: set[int] = set()
    for row in elem_rows:
        for oid in row['nodes']:
            connected_gmsh.add(reverse_map[oid])

    # Also keep nodes needed by boundary conditions and loads —
    # these may sit on physical groups (e.g. base supports at dim=0)
    # that have no assigned OpenSees element.
    all_gmsh_tags = set(ops._node_map.keys())
    for pg_name, bc in ops._bcs.items():
        pg_dim, pg_tag = ops._find_pg(pg_name, bc.get("dim"))
        raw_pg, _ = gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)
        connected_gmsh.update(
            int(t) for t in raw_pg if int(t) in all_gmsh_tags
        )

    for loads in ops._load_patterns.values():
        for load_def in loads:
            if load_def["type"] == "nodal":
                pg_name = load_def["pg_name"]
                pg_dim, pg_tag = ops._find_pg(pg_name, load_def.get("dim"))
                raw_pg, _ = gmsh.model.mesh.getNodesForPhysicalGroup(
                    pg_dim, pg_tag
                )
                connected_gmsh.update(
                    int(t) for t in raw_pg if int(t) in all_gmsh_tags
                )
            elif load_def["type"] == "nodal_direct":
                nid = int(load_def["node_id"])
                if nid in all_gmsh_tags:
                    connected_gmsh.add(nid)

    n_total  = len(ops._node_map)
    n_pruned = n_total - len(connected_gmsh)
    if n_pruned > 0:
        ops._log(
            f"build(): pruned {n_pruned} disconnected node(s) "
            f"({n_total} → {len(connected_gmsh)})"
        )

        # Build new sequential mapping (gmsh tag → new ops id)
        sorted_connected = sorted(connected_gmsh)
        new_node_map = {
            gt: new_id
            for new_id, gt in enumerate(sorted_connected, 1)
        }

        # Remap element node tuples from old ops ids to new ones
        old_to_new = {
            ops._node_map[gt]: new_node_map[gt]
            for gt in connected_gmsh
        }
        for row in elem_rows:
            row['nodes'] = tuple(old_to_new[oid] for oid in row['nodes'])

        # Rebuild coordinate table for connected nodes only
        raw_tag_to_idx = {int(t): i for i, t in enumerate(raw_tags)}
        connected_indices = [raw_tag_to_idx[gt] for gt in sorted_connected]
        cc = coords_arr[connected_indices]

        ops._node_map = new_node_map
        ops._nodes_df = pd.DataFrame({
            'ops_id': np.arange(1, len(sorted_connected) + 1,
                                dtype=np.int64),
            'x': cc[:, 0],
            'y': cc[:, 1],
            'z': cc[:, 2],
        }).set_index('ops_id')

    cols = [
        'gmsh_id', 'ops_type', 'pg_name', 'mat_name', 'mat_tag',
        'sec_tag', 'transf_tag', 'n_nodes', 'nodes', 'slots', 'extra',
    ]
    ops._elements_df = (
        pd.DataFrame(elem_rows).set_index('ops_id')
        if elem_rows
        else pd.DataFrame(columns=cols)
    )
    ops._log(
        f"build(): {len(elem_rows)} elements from "
        f"{len(ops._elem_assignments)} group(s)"
    )
    ops._built = True
