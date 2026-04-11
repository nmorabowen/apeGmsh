from __future__ import annotations

import gmsh
import numpy as np
import pandas as pd
from numpy import ndarray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

class Inspect:
    """
    Composite introspection helper attached to a apeGmsh instance as ``self.inspect``.

    All geometry-query logic lives here so that apeGmsh itself stays focused on
    model lifecycle / IO / visualisation.

    Parameters
    ----------
    parent : _SessionBase
        The owning apeGmsh instance (provides ``_verbose``).
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_geometry_info(self) -> tuple[dict, pd.DataFrame]:
        """
        Build a nested mapping with flat DataFrames and type summaries per
        dimension.  Pure geometric introspection — no mesh, no sampling.

        Returns
        -------
        mapping : dict
            {
                label: {
                    'df'      : pd.DataFrame,   # flat entity table (one row per entity)
                    'summary' : pd.DataFrame,   # type counts for this dimension
                    'entities': { tag: dict }   # raw per-entity data
                }
            }
        global_summary : pd.DataFrame
            Single table with entity counts and types across all dimensions.
        """
        entity_labels: dict[int, str] = {
            0: 'points',
            1: 'curves',
            2: 'surfaces',
            3: 'volumes',
        }

        mapping: dict[str, dict] = {
            label: {'df': None, 'summary': None, 'entities': {}}
            for label in entity_labels.values()
        }

        global_rows: list[dict] = []

        for dim, label in entity_labels.items():
            entities         = gmsh.model.getEntities(dim=dim)
            rows: list[dict] = []

            for _, tag in entities:

                entity_type: str         = gmsh.model.getType(dim, tag)
                boundary                 = gmsh.model.getBoundary([(dim, tag)], oriented=False)
                boundary_tags: list[int] = [t for _, t in boundary]
                bounds                   = gmsh.model.getParametrizationBounds(dim, tag)

                entry: dict = {
                    'type'              : entity_type,
                    'boundary_tags'     : boundary_tags,
                    'parametric_bounds' : bounds,
                }

                # --- dim 0: points ---
                if dim == 0:
                    coords: ndarray  = np.array(gmsh.model.getValue(0, tag, []))
                    entry['coords']  = coords
                    rows.append({
                        'name': f'Point {tag}',
                        'tag' : tag,
                        'type': entity_type,
                        'x'   : coords[0],
                        'y'   : coords[1],
                        'z'   : coords[2],
                    })

                # --- dim 1: curves ---
                elif dim == 1:
                    u_min: float = bounds[0][0]
                    u_max: float = bounds[1][0]
                    u_mid: float = 0.5 * (u_min + u_max)

                    start   : ndarray = np.array(gmsh.model.getValue(1, tag, [u_min]))
                    end     : ndarray = np.array(gmsh.model.getValue(1, tag, [u_max]))
                    midpoint: ndarray = np.array(gmsh.model.getValue(1, tag, [u_mid]))
                    length  : float   = gmsh.model.occ.getMass(1, tag)
                    curvature: float  = gmsh.model.getCurvature(1, tag, [u_mid])

                    entry['length']           = length
                    entry['start']            = start
                    entry['end']              = end
                    entry['midpoint']         = midpoint
                    entry['curvature_at_mid'] = curvature

                    rows.append({
                        'name'            : f'Curve {tag}',
                        'tag'             : tag,
                        'type'            : entity_type,
                        'start_pt'        : boundary_tags[0] if boundary_tags else -1,
                        'end_pt'          : boundary_tags[-1] if boundary_tags else -1,
                        'length'          : length,
                        'start_x'         : start[0],
                        'start_y'         : start[1],
                        'start_z'         : start[2],
                        'end_x'           : end[0],
                        'end_y'           : end[1],
                        'end_z'           : end[2],
                        'mid_x'           : midpoint[0],
                        'mid_y'           : midpoint[1],
                        'mid_z'           : midpoint[2],
                        'curvature_at_mid': curvature,
                        'u_min'           : u_min,
                        'u_max'           : u_max,
                    })

                # --- dim 2: surfaces ---
                elif dim == 2:
                    u_mid_2d: list[float] = [
                        0.5 * (bounds[0][i] + bounds[1][i]) for i in range(2)
                    ]

                    area           = float(gmsh.model.occ.getMass(2, tag))
                    center         = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
                    normal         = np.array(gmsh.model.getNormal(tag, u_mid_2d))
                    curvature      = float(gmsh.model.getCurvature(2, tag, u_mid_2d))
                    k1, k2, d1, d2          = gmsh.model.getPrincipalCurvatures(tag, u_mid_2d)

                    entry['area']                 = area
                    entry['center_of_mass']       = center
                    entry['normal_at_mid']        = normal
                    entry['curvature_at_mid']     = curvature
                    entry['principal_curvatures'] = {
                        'k1': k1, 'k2': k2,
                        'd1': np.array(d1), 'd2': np.array(d2)
                    }

                    rows.append({
                        'name'              : f'Surface {tag}',
                        'tag'               : tag,
                        'type'              : entity_type,
                        'n_boundary_curves' : len(boundary_tags),
                        'boundary_curves'   : str(boundary_tags),
                        'area'              : area,
                        'cx'                : center[0],
                        'cy'                : center[1],
                        'cz'                : center[2],
                        'nx'                : normal[0],
                        'ny'                : normal[1],
                        'nz'                : normal[2],
                        'curvature_at_mid'  : curvature,
                        'k1'                : k1,
                        'k2'                : k2,
                    })

                # --- dim 3: volumes ---
                else:
                    volume = float(gmsh.model.occ.getMass(3, tag))
                    center = np.array(gmsh.model.occ.getCenterOfMass(3, tag))
                    inertia: ndarray = np.array(
                        gmsh.model.occ.getMatrixOfInertia(3, tag)
                    ).reshape(3, 3)

                    entry['volume']         = volume
                    entry['center_of_mass'] = center
                    entry['inertia']        = inertia

                    rows.append({
                        'name'               : f'Volume {tag}',
                        'tag'                : tag,
                        'type'               : entity_type,
                        'n_boundary_surfaces': len(boundary_tags),
                        'boundary_surfaces'  : str(boundary_tags),
                        'volume'             : volume,
                        'cx'                 : center[0],
                        'cy'                 : center[1],
                        'cz'                 : center[2],
                        'ixx'                : inertia[0, 0],
                        'iyy'                : inertia[1, 1],
                        'izz'                : inertia[2, 2],
                        'ixy'                : inertia[0, 1],
                        'ixz'                : inertia[0, 2],
                        'iyz'                : inertia[1, 2],
                    })

                mapping[label]['entities'][tag] = entry

            # --- flat entity DataFrame ---
            mapping[label]['df'] = (
                pd.DataFrame(rows).set_index('name') if rows else pd.DataFrame()
            )

            # --- per-dimension type summary ---
            if rows:
                mapping[label]['summary'] = (
                    pd.DataFrame(rows)
                    .groupby('type')
                    .size()
                    .reset_index(name='count')
                    .set_index('type')
                )
            else:
                mapping[label]['summary'] = pd.DataFrame()

            # --- accumulate into global summary ---
            for row in rows:
                global_rows.append({
                    'entity': label,
                    'type'  : row.get('type', 'Vertex'),
                })

        # --- global summary ---
        global_summary: pd.DataFrame = (
            pd.DataFrame(global_rows)
            .groupby(['entity', 'type'])
            .size()
            .reset_index(name='count')
            .set_index(['entity', 'type'])
        )

        if self._parent._verbose:
            print("\n--- Global Geometry Summary ---")
            print(global_summary.to_string())
            for label, data in mapping.items():
                df: pd.DataFrame = data['df']
                if not df.empty:
                    print(f"\n--- {label.capitalize()} Entities ---")
                    print(df.to_string())

        return mapping, global_summary

    def get_mesh_info(self) -> tuple[dict, pd.DataFrame]:
        """
        Introspect the current mesh.  Returns counts, per-entity breakdowns
        and a per-element-type quality summary.  No geometric re-sampling —
        only what ``gmsh.model.mesh`` can report directly.

        Returns
        -------
        mapping : dict
            {
                'nodes'    : {'count': int, 'df': pd.DataFrame},
                'elements' : {
                    'df'      : pd.DataFrame,  # one row per (dim, tag, elem_type)
                    'summary' : pd.DataFrame,  # counts per element type
                    'quality' : pd.DataFrame,  # min/mean/max SICN per elem type
                },
            }
        global_summary : pd.DataFrame
            Single table with node / element counts indexed by dim.
        """
        # --- nodes -----------------------------------------------------
        node_tags, node_xyz, _ = gmsh.model.mesh.getNodes(
            dim=-1, tag=-1, includeBoundary=True
        )
        n_nodes: int = len(node_tags)
        nodes_df: pd.DataFrame = (
            pd.DataFrame({
                'tag': node_tags,
                'x'  : node_xyz[0::3],
                'y'  : node_xyz[1::3],
                'z'  : node_xyz[2::3],
            }).set_index('tag')
            if n_nodes else pd.DataFrame()
        )

        # --- elements, per entity -------------------------------------
        elem_rows   : list[dict] = []
        quality_rows: list[dict] = []
        global_rows : list[dict] = [{'kind': 'nodes', 'dim': -1, 'count': n_nodes}]

        for dim in range(4):
            dim_elem_count = 0
            for _, tag in gmsh.model.getEntities(dim=dim):
                etypes, etags, _ = gmsh.model.mesh.getElements(dim=dim, tag=tag)
                for et, tags in zip(etypes, etags):
                    name, edim, order, n_per_elem, *_ = (
                        gmsh.model.mesh.getElementProperties(et)
                    )
                    n = len(tags)
                    dim_elem_count += n
                    elem_rows.append({
                        'dim'         : dim,
                        'entity_tag'  : tag,
                        'elem_type'   : name,
                        'order'       : order,
                        'nodes_per_el': n_per_elem,
                        'count'       : n,
                    })

                    # quality (SICN) — only defined for dim>=2 solid/surface elements
                    if dim >= 2 and n > 0:
                        try:
                            q = np.asarray(
                                gmsh.model.mesh.getElementQualities(
                                    list(tags), qualityName='minSICN'
                                )
                            )
                            quality_rows.append({
                                'elem_type'   : name,
                                'count'       : n,
                                'sicn_min'    : float(q.min()),
                                'sicn_mean'   : float(q.mean()),
                                'sicn_max'    : float(q.max()),
                            })
                        except Exception:
                            pass

            global_rows.append({
                'kind': 'elements', 'dim': dim, 'count': dim_elem_count,
            })

        # --- assemble return dicts ---
        elem_df = (
            pd.DataFrame(elem_rows).set_index(['dim', 'entity_tag'])
            if elem_rows else pd.DataFrame()
        )
        elem_summary = (
            pd.DataFrame(elem_rows)
            .groupby('elem_type')['count']
            .sum()
            .reset_index()
            .set_index('elem_type')
            if elem_rows else pd.DataFrame()
        )
        quality_df = (
            pd.DataFrame(quality_rows).set_index('elem_type')
            if quality_rows else pd.DataFrame()
        )
        global_summary = pd.DataFrame(global_rows).set_index('kind')

        mapping = {
            'nodes': {'count': n_nodes, 'df': nodes_df},
            'elements': {
                'df': elem_df,
                'summary': elem_summary,
                'quality': quality_df,
            },
        }

        if self._parent._verbose:
            print("\n--- Mesh Info ---")
            print(f"Nodes: {n_nodes}")
            if not elem_summary.empty:
                print(elem_summary.to_string())
            if not quality_df.empty:
                print("\n--- Element Quality (SICN) ---")
                print(quality_df.to_string())

        return mapping, global_summary

    # ------------------------------------------------------------------
    # Model summary — true introspection
    # ------------------------------------------------------------------

    def print_summary(self) -> str:
        """
        Print a comprehensive model summary by introspecting the live
        Gmsh session.  Two data sources are distinguished:

        * **[gmsh]** — read directly from the Gmsh API (ground truth).
        * **[tracked]** — recorded by apeGmsh's ``Mesh`` wrapper for
          settings that Gmsh exposes no getter for (transfinite,
          per-entity sizes, recombine, fields, per-entity algorithm).

        The summary covers:

        1. **Geometry** — entity counts and types per dimension
        2. **Physical groups** — names, dimensions, member counts
        3. **Mesh options** — global settings readable via
           ``gmsh.option.getNumber``
        4. **Mesh directives** — write-only settings tracked by apeGmsh
        5. **Mesh statistics** — node/element counts, element types,
           quality (if mesh has been generated)

        Returns
        -------
        str
            The formatted summary text (also printed to stdout).
        """
        import math as _math
        lines: list[str] = []
        _hr = "=" * 72

        def _section(title: str) -> None:
            lines.append("")
            lines.append(_hr)
            lines.append(f"  {title}")
            lines.append(_hr)

        def _sub(title: str) -> None:
            lines.append(f"\n  --- {title} ---")

        _DIM_LABEL = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}
        _ALGO_2D = {
            1: "MeshAdapt", 2: "Automatic", 3: "InitialMeshOnly",
            5: "Delaunay", 6: "Frontal-Delaunay", 7: "BAMG",
            8: "Frontal-Delaunay for Quads",
            9: "Packing of Parallelograms", 11: "Quasi-Structured Quad",
        }
        _ALGO_3D = {
            1: "Delaunay", 3: "InitialMeshOnly", 4: "Frontal",
            7: "MMG3D", 9: "R-tree", 10: "HXT",
        }

        # ==============================================================
        # 1. GEOMETRY  [gmsh]
        # ==============================================================
        _section("GEOMETRY  [gmsh]")
        total_ents = 0
        for dim in range(4):
            ents = gmsh.model.getEntities(dim=dim)
            n = len(ents)
            total_ents += n
            if n == 0:
                continue
            # Count by type
            types: dict[str, int] = {}
            for _, tag in ents:
                try:
                    t = gmsh.model.getType(dim, tag)
                except Exception:
                    t = "Unknown"
                types[t] = types.get(t, 0) + 1
            type_str = ", ".join(f"{v} {k}" for k, v in sorted(types.items()))
            lines.append(f"  {_DIM_LABEL[dim]:10s}: {n:>5d}  ({type_str})")

        # Bounding box of entire model
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
            lines.append(
                f"  {'BBox':10s}: "
                f"({bb[0]:.4g}, {bb[1]:.4g}, {bb[2]:.4g}) → "
                f"({bb[3]:.4g}, {bb[4]:.4g}, {bb[5]:.4g})"
            )
            diag = float(np.linalg.norm(
                [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
            ))
            lines.append(f"  {'Diagonal':10s}: {diag:.4g}")
        except Exception:
            pass

        lines.append(f"  {'Total':10s}: {total_ents} entities")

        # ==============================================================
        # 2. PHYSICAL GROUPS  [gmsh]
        # ==============================================================
        _section("PHYSICAL GROUPS  [gmsh]")
        pgs = gmsh.model.getPhysicalGroups()
        if not pgs:
            lines.append("  (none)")
        else:
            for dim, pg_tag in pgs:
                name = gmsh.model.getPhysicalName(dim, pg_tag) or "(unnamed)"
                try:
                    members = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
                    n_mem = len(members)
                    tags_str = ", ".join(str(t) for t in members[:8])
                    if n_mem > 8:
                        tags_str += f" ... (+{n_mem - 8} more)"
                except Exception:
                    n_mem = 0
                    tags_str = "?"
                lines.append(
                    f"  dim={dim}  pg_tag={pg_tag:>3d}  "
                    f"{name!r:30s}  "
                    f"{n_mem:>4d} entities  [{tags_str}]"
                )
            # Reverse: entities with no physical group
            all_ents = set(gmsh.model.getEntities())
            assigned: set[tuple[int, int]] = set()
            for dim, pg_tag in pgs:
                try:
                    for t in gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag):
                        assigned.add((dim, int(t)))
                except Exception:
                    pass
            unassigned = all_ents - assigned
            if unassigned:
                by_dim: dict[int, int] = {}
                for d, _ in unassigned:
                    by_dim[d] = by_dim.get(d, 0) + 1
                parts = ", ".join(
                    f"{n} {_DIM_LABEL[d].lower()}" for d, n in sorted(by_dim.items())
                )
                lines.append(f"\n  Unassigned entities: {len(unassigned)} ({parts})")

        # ==============================================================
        # 3. MESH OPTIONS  [gmsh]
        # ==============================================================
        _section("MESH OPTIONS  [gmsh]")
        _opts: list[tuple[str, str, str]] = [
            ("Mesh.MeshSizeMin",              "Size min",              ""),
            ("Mesh.MeshSizeMax",              "Size max",              ""),
            ("Mesh.MeshSizeFactor",           "Size factor",           ""),
            ("Mesh.MeshSizeFromCurvature",    "From curvature",        "elems per 2π"),
            ("Mesh.MeshSizeFromPoints",       "From points",           "bool"),
            ("Mesh.MeshSizeExtendFromBoundary", "Extend from boundary", "bool"),
            ("Mesh.Algorithm",                "Algorithm 2D",          "code"),
            ("Mesh.Algorithm3D",              "Algorithm 3D",          "code"),
            ("Mesh.ElementOrder",             "Element order",         ""),
            ("Mesh.RecombineAll",             "Recombine all",         "bool"),
            ("Mesh.RecombinationAlgorithm",   "Recombine algorithm",   "code"),
            ("Mesh.Optimize",                 "Optimize",              "bool"),
            ("Mesh.OptimizeNetgen",           "Optimize Netgen",       "bool"),
            ("Mesh.SubdivisionAlgorithm",     "Subdivision",           "code"),
            ("Mesh.MinimumCurveNodes",        "Min curve nodes",       ""),
        ]
        for opt_key, label, hint in _opts:
            try:
                val = gmsh.option.getNumber(opt_key)
                extra = ""
                if opt_key == "Mesh.Algorithm":
                    extra = f"  ({_ALGO_2D.get(int(val), '?')})"
                elif opt_key == "Mesh.Algorithm3D":
                    extra = f"  ({_ALGO_3D.get(int(val), '?')})"
                elif hint == "bool":
                    extra = f"  ({'ON' if val else 'OFF'})"
                elif hint == "elems per 2π" and val > 0:
                    extra = f"  ({int(val)} elements per 2π)"
                lines.append(f"  {label:28s}: {val:>12g}{extra}")
            except Exception:
                pass

        # ==============================================================
        # 4. MESH DIRECTIVES  [tracked by apeGmsh]
        # ==============================================================
        _section("MESH DIRECTIVES  [tracked by apeGmsh]")
        mesh_composite = getattr(self._parent, 'mesh', None)
        directives = getattr(mesh_composite, '_directives', []) if mesh_composite else []

        if not directives:
            lines.append("  (no directives recorded this session)")
        else:
            # Group by kind for readability
            by_kind: dict[str, list[dict]] = {}
            for d in directives:
                by_kind.setdefault(d['kind'], []).append(d)

            for kind, items in by_kind.items():
                _sub(f"{kind} ({len(items)} directive{'s' if len(items) != 1 else ''})")

                if kind == 'set_size':
                    for d in items:
                        lines.append(
                            f"    dim={d['dim']}  tags={d['tags']}  "
                            f"size={d['size']}"
                        )
                elif kind == 'set_size_all_points':
                    for d in items:
                        lines.append(
                            f"    size={d['size']}  "
                            f"({d['n_points']} points)"
                        )
                elif kind == 'set_size_callback':
                    for d in items:
                        lines.append(f"    callback={d['func_name']}")
                elif kind == 'transfinite_curve':
                    for d in items:
                        lines.append(
                            f"    curve={d['tag']}  n_nodes={d['n_nodes']}  "
                            f"type={d['mesh_type']}  coef={d['coef']}"
                        )
                elif kind == 'transfinite_surface':
                    for d in items:
                        corners = d['corners'] or "auto"
                        lines.append(
                            f"    surface={d['tag']}  "
                            f"arrangement={d['arrangement']}  "
                            f"corners={corners}"
                        )
                elif kind == 'transfinite_volume':
                    for d in items:
                        corners = d['corners'] or "auto"
                        lines.append(
                            f"    volume={d['tag']}  corners={corners}"
                        )
                elif kind == 'transfinite_automatic':
                    for d in items:
                        angle_deg = _math.degrees(d['corner_angle'])
                        lines.append(
                            f"    corner_angle={angle_deg:.1f}°  "
                            f"recombine={d['recombine']}  "
                            f"dim_tags={d['dim_tags'] or 'all'}"
                        )
                elif kind == 'recombine':
                    for d in items:
                        lines.append(
                            f"    dim={d['dim']}  tag={d['tag']}  "
                            f"angle={d['angle']}°"
                        )
                elif kind == 'smoothing':
                    for d in items:
                        lines.append(
                            f"    dim={d['dim']}  tag={d['tag']}  "
                            f"passes={d['val']}"
                        )
                elif kind == 'algorithm':
                    for d in items:
                        alg_name = ""
                        if d['dim'] == 2:
                            alg_name = _ALGO_2D.get(d['algorithm'], '?')
                        elif d['dim'] == 3:
                            alg_name = _ALGO_3D.get(d['algorithm'], '?')
                        lines.append(
                            f"    dim={d['dim']}  tag={d['tag']}  "
                            f"algorithm={d['algorithm']} ({alg_name})"
                        )
                elif kind == 'field_add':
                    for d in items:
                        lines.append(
                            f"    field_tag={d['field_tag']}  "
                            f"type={d['field_type']}"
                        )
                elif kind == 'field_background':
                    for d in items:
                        lines.append(
                            f"    background_field={d['field_tag']}"
                        )
                else:
                    for d in items:
                        lines.append(f"    {d}")

        # ==============================================================
        # 5. MESH STATISTICS  [gmsh]
        # ==============================================================
        _section("MESH STATISTICS  [gmsh]")
        try:
            node_tags, _, _ = gmsh.model.mesh.getNodes(
                dim=-1, tag=-1, includeBoundary=True
            )
            n_nodes = len(node_tags)
        except Exception:
            n_nodes = 0

        if n_nodes == 0:
            lines.append("  (no mesh generated yet)")
        else:
            lines.append(f"  Total nodes: {n_nodes}")

            # Per-dim element breakdown
            _sub("Elements by type")
            type_totals: dict[str, int] = {}
            for dim in range(4):
                for _, tag in gmsh.model.getEntities(dim=dim):
                    etypes, etags, _ = gmsh.model.mesh.getElements(
                        dim=dim, tag=tag,
                    )
                    for et, tags_arr in zip(etypes, etags):
                        name, _, order, n_per, *_ = (
                            gmsh.model.mesh.getElementProperties(et)
                        )
                        n = len(tags_arr)
                        type_totals[name] = type_totals.get(name, 0) + n

            total_elems = 0
            for ename, count in sorted(type_totals.items()):
                lines.append(f"    {ename:30s}: {count:>8d}")
                total_elems += count
            lines.append(f"    {'TOTAL':30s}: {total_elems:>8d}")

            # Per-physical-group node counts
            pgs = gmsh.model.getPhysicalGroups()
            if pgs:
                _sub("Nodes per physical group")
                for dim, pg_tag in pgs:
                    name = gmsh.model.getPhysicalName(dim, pg_tag) or f"(tag={pg_tag})"
                    try:
                        pg_nodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(
                            dim, pg_tag,
                        )
                        lines.append(
                            f"    dim={dim}  {name!r:30s}: "
                            f"{len(pg_nodes):>8d} nodes"
                        )
                    except Exception:
                        lines.append(
                            f"    dim={dim}  {name!r:30s}: (error)"
                        )

            # Quality snapshot (aggregate by element type, dim>=2 only)
            _sub("Element quality (SICN)")
            has_quality = False
            for dim in (2, 3):
                for _, tag in gmsh.model.getEntities(dim=dim):
                    etypes, etags, _ = gmsh.model.mesh.getElements(
                        dim=dim, tag=tag,
                    )
                    for et, tags_arr in zip(etypes, etags):
                        if len(tags_arr) == 0:
                            continue
                        name, *_ = gmsh.model.mesh.getElementProperties(et)
                        try:
                            q = np.asarray(
                                gmsh.model.mesh.getElementQualities(
                                    list(tags_arr), qualityName='minSICN',
                                )
                            )
                            has_quality = True
                            lines.append(
                                f"    {name:24s}  "
                                f"n={len(tags_arr):>6d}  "
                                f"min={q.min():.4f}  "
                                f"mean={q.mean():.4f}  "
                                f"max={q.max():.4f}"
                            )
                        except Exception:
                            pass
            if not has_quality:
                lines.append("    (no 2D/3D elements to measure)")

        # ==============================================================
        # Footer
        # ==============================================================
        lines.append("")
        lines.append(_hr)

        text = "\n".join(lines)
        print(text)
        return text