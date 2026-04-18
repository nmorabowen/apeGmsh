"""
Mesh Scene Builder — construct batched VTK actors from Gmsh mesh data.

Extracts per-entity element connectivity via ``getElements``, merges
all entities of each dimension into one ``UnstructuredGrid`` with
``cell_data["colors"]`` and ``cell_data["entity_tag"]``.

Also builds the node cloud, KD-tree, and element bookkeeping data.

Usage::

    from apeGmsh.viewers.scene.mesh_scene import build_mesh_scene
    scene = build_mesh_scene(plotter, dims=[1, 2, 3])
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import gmsh
import numpy as np
import pyvista as pv

from apeGmsh._types import DimTag
from ..core.entity_registry import EntityRegistry
from ..ui.theme import THEME
from .glyph_points import build_node_cloud


# ======================================================================
# Gmsh -> VTK element type mapping
# ======================================================================

GMSH_TO_VTK: dict[int, int] = {
    1:  3,    # 2-node line          -> VTK_LINE
    2:  5,    # 3-node triangle      -> VTK_TRIANGLE
    3:  9,    # 4-node quad          -> VTK_QUAD
    4:  10,   # 4-node tet           -> VTK_TETRA
    5:  12,   # 8-node hex           -> VTK_HEXAHEDRON
    6:  13,   # 6-node prism         -> VTK_WEDGE
    7:  14,   # 5-node pyramid       -> VTK_PYRAMID
    8:  21,   # 6-node tri (2nd)     -> VTK_QUADRATIC_TRIANGLE
    9:  23,   # 8-node quad (2nd)    -> VTK_QUADRATIC_QUAD
    11: 24,   # 10-node tet (2nd)    -> VTK_QUADRATIC_TETRA
    15: 1,    # 1-node point         -> VTK_VERTEX
}

# Element type name -> color palette key
ELEM_TYPE_COLORS: dict[str, str] = {
    "Triangle":      "#4363d8",
    "Quadrilateral": "#3cb44b",
    "Tetrahedron":   "#e6194b",
    "Hexahedron":    "#f58231",
    "Prism":         "#911eb4",
    "Pyramid":       "#42d4f4",
    "Line":          "#aaaaaa",
    "Point":         "#ffffff",
}

PARTITION_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

def elem_type_category(name: str) -> str:
    """Map a Gmsh element type name to a colour-palette key."""
    low = name.lower()
    for key in ("triangle", "quadrilateral", "quad", "tetrahedron",
                "hexahedron", "prism", "pyramid", "line", "point"):
        if key in low:
            return key.capitalize()
    return "Line"


# ======================================================================
# Element properties cache (module-level)
# ======================================================================

_elem_props_cache: dict[int, tuple] = {}


def _get_elem_props(etype: int) -> tuple:
    """Cached ``gmsh.model.mesh.getElementProperties(etype)``."""
    cached = _elem_props_cache.get(etype)
    if cached is not None:
        return cached
    props = gmsh.model.mesh.getElementProperties(int(etype))
    _elem_props_cache[etype] = props
    return props


# ======================================================================
# MeshSceneData
# ======================================================================

@dataclass
class MeshSceneData:
    """Everything the mesh viewer needs from the scene build."""

    registry: EntityRegistry
    node_tags: np.ndarray
    node_coords: np.ndarray
    node_tag_to_idx: dict[int, int]
    tag_to_idx: np.ndarray                         # dense lookup array
    model_diagonal: float

    # Element bookkeeping
    elem_data: dict[int, dict] = field(default_factory=dict)
    elem_to_brep: dict[int, DimTag] = field(default_factory=dict)
    brep_to_elems: dict[DimTag, list[int]] = field(default_factory=dict)
    brep_dominant_type: dict[DimTag, str] = field(default_factory=dict)
    batch_cell_to_elem: dict[int, dict[int, int]] = field(default_factory=dict)

    # Node cloud
    node_cloud: pv.PolyData | None = None
    node_actor: Any = None
    node_tree: Any = None  # scipy.spatial.cKDTree or None

    # Physical group mappings
    group_to_breps: dict[str, list[DimTag]] = field(default_factory=dict)
    brep_to_group: dict[DimTag, str] = field(default_factory=dict)


# ======================================================================
# Cell extraction (per entity)
# ======================================================================

def _collect_entity_cells(
    dim: int,
    tag: int,
    tag_to_idx: np.ndarray,
    *,
    elem_data_out: dict[int, dict],
    elem_to_brep_out: dict[int, DimTag],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], str] | None:
    """Extract VTK cells for one BRep entity.

    Returns ``(cell_parts, type_parts, elem_tags, dominant_type)``
    or ``None`` if empty.
    """
    dt: DimTag = (dim, tag)
    try:
        elem_types, elem_tags_list, elem_node_tags_list = (
            gmsh.model.mesh.getElements(dim, tag)
        )
    except Exception:
        return None
    # Gmsh returns ``elem_types`` as a numpy ndarray — plain Python
    # truthiness (``if not elem_types``) raises ``ValueError`` on the
    # empty case because numpy refuses to coerce an empty array to
    # ``bool``.  Use ``len() == 0`` which works for both list and
    # ndarray and short-circuits on unmeshed entities (e.g. a dim=3
    # volume when the user only meshed up to dim=2).
    if len(elem_types) == 0:
        return None

    all_cells: list[np.ndarray] = []
    all_types: list[np.ndarray] = []
    brep_elem_tags: list[int] = []
    dominant_type_name = ""
    dominant_count = 0

    for etype, etags, enodes in zip(
        elem_types, elem_tags_list, elem_node_tags_list,
    ):
        vtk_type = GMSH_TO_VTK.get(int(etype))
        if vtk_type is None:
            continue
        props = _get_elem_props(int(etype))
        type_name: str = props[0]
        n_nodes: int = props[3]

        etags_arr = np.asarray(etags, dtype=np.int64)
        enodes_arr = np.asarray(enodes, dtype=np.int64)
        n_elems = len(etags_arr)
        if n_elems == 0:
            continue
        if n_elems > dominant_count:
            dominant_count = n_elems
            dominant_type_name = type_name

        node_rows = enodes_arr.reshape(n_elems, n_nodes)
        if tag_to_idx is not None and len(tag_to_idx) > 0:
            max_allowed = len(tag_to_idx) - 1
            in_range = (
                np.all(node_rows <= max_allowed)
                and np.all(node_rows >= 0)
            )
            if in_range:
                idx_arr = tag_to_idx[node_rows]
                valid_mask = np.all(idx_arr >= 0, axis=1)
            else:
                clipped = np.clip(node_rows, 0, max_allowed)
                idx_arr = tag_to_idx[clipped]
                valid_mask = (
                    np.all(idx_arr >= 0, axis=1)
                    & np.all(node_rows >= 0, axis=1)
                    & np.all(node_rows <= max_allowed, axis=1)
                )
        else:
            idx_arr = np.zeros_like(node_rows)
            valid_mask = np.zeros(n_elems, dtype=bool)

        valid_idx_arr = idx_arr[valid_mask]
        valid_etags = etags_arr[valid_mask]
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            continue

        prefix = np.full((n_valid, 1), n_nodes, dtype=np.int64)
        all_cells.append(np.hstack([prefix, valid_idx_arr]).ravel())
        all_types.append(np.full(n_valid, vtk_type, dtype=np.uint8))

        valid_node_rows = node_rows[valid_mask]
        for ri in range(n_valid):
            elem_tag = int(valid_etags[ri])
            brep_elem_tags.append(elem_tag)
            elem_to_brep_out[elem_tag] = dt
            elem_data_out[elem_tag] = {
                "type_name": type_name,
                "nodes": valid_node_rows[ri].tolist(),
                "dim": dim,
                "brep_dt": dt,
            }

    if not all_cells:
        return None
    return all_cells, all_types, brep_elem_tags, dominant_type_name


# ======================================================================
# Public API
# ======================================================================

def build_mesh_scene(
    plotter: pv.Plotter,
    dims: list[int],
    *,
    line_width: float = 3.0,
    surface_opacity: float = 1.0,
    show_surface_edges: bool = True,
    node_marker_size: float = 6.0,
    node_color: str | None = None,
    edge_color: str | None = None,
    verbose: bool = False,
) -> MeshSceneData:
    """Build batched mesh actors and return a :class:`MeshSceneData`.

    Always uses the batched path (one actor per dim).
    ``node_color`` / ``edge_color`` default to the active palette's
    ``node_accent`` / ``mesh_edge_color`` when left as ``None``.
    """
    from ..ui.preferences_manager import PREFERENCES as _PREF
    _pref = _PREF.current

    _pal = THEME.current
    if node_color is None:
        node_color = _pal.node_accent
    if edge_color is None:
        edge_color = _pal.mesh_edge_color

    t0 = time.perf_counter()
    registry = EntityRegistry()

    # ── fetch all mesh nodes ────────────────────────────────────────
    node_tags_raw, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags_raw, dtype=np.int64)
    node_coords = np.asarray(node_coords_flat, dtype=np.float64).reshape(-1, 3)

    if len(node_tags) > 0:
        max_tag = int(node_tags.max())
        tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
        tag_to_idx[node_tags] = np.arange(len(node_tags), dtype=np.int64)
    else:
        tag_to_idx = np.array([], dtype=np.int64)

    try:
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm(
            [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
        ))
        if diag <= 0.0:
            diag = 1.0
        origin = np.array([
            (bb[0] + bb[3]) * 0.5,
            (bb[1] + bb[4]) * 0.5,
            (bb[2] + bb[5]) * 0.5,
        ])
    except Exception:
        diag = 1.0
        origin = np.zeros(3)

    # Shift node coordinates to origin for numerical stability
    node_coords -= origin
    registry.origin_shift = origin

    t_setup = time.perf_counter() - t0

    # ── element bookkeeping ─────────────────────────────────────────
    elem_data: dict[int, dict] = {}
    elem_to_brep: dict[int, DimTag] = {}
    brep_to_elems: dict[DimTag, list[int]] = {}
    brep_dominant_type: dict[DimTag, str] = {}
    batch_cell_to_elem: dict[int, dict[int, int]] = {}

    # ── build batched actors per dim ────────────────────────────────
    t_actors = time.perf_counter()
    n_actors = 0

    for dim in sorted(dims):
        all_cells_parts: list[np.ndarray] = []
        all_types_parts: list[np.ndarray] = []
        all_entity_tags: list[int] = []
        all_elem_tags_flat: list[int] = []
        cell_to_dt: dict[int, DimTag] = {}
        centroids_dim: dict[DimTag, np.ndarray] = {}
        cell_offset = 0

        for _, tag in gmsh.model.getEntities(dim=dim):
            dt: DimTag = (dim, tag)
            result = _collect_entity_cells(
                dim, tag, tag_to_idx,
                elem_data_out=elem_data,
                elem_to_brep_out=elem_to_brep,
            )
            if result is None:
                continue
            cell_parts, type_parts, be_tags, dom_type = result

            n_entity_cells = len(be_tags)
            brep_to_elems[dt] = be_tags
            brep_dominant_type[dt] = elem_type_category(dom_type)

            cell_indices = list(range(cell_offset, cell_offset + n_entity_cells))
            for ci in cell_indices:
                cell_to_dt[ci] = dt

            all_cells_parts.extend(cell_parts)
            all_types_parts.extend(type_parts)
            all_entity_tags.extend([tag] * n_entity_cells)
            all_elem_tags_flat.extend(be_tags)

            try:
                ntags, ncoords, _ = gmsh.model.mesh.getNodes(
                    dim=dim, tag=tag, includeBoundary=True,
                )
                if len(ntags) > 0:
                    pts = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
                    centroids_dim[dt] = pts.mean(axis=0)
            except Exception:
                pass

            cell_offset += n_entity_cells

        if not all_cells_parts:
            continue

        cells_flat = np.concatenate(all_cells_parts)
        cell_types_flat = np.concatenate(all_types_parts)
        grid = pv.UnstructuredGrid(cells_flat, cell_types_flat, node_coords)
        _idle_rgb = np.array(
            _pal.dim_pt if dim == 0 else
            _pal.dim_crv if dim == 1 else
            _pal.dim_srf if dim == 2 else
            _pal.dim_vol,
            dtype=np.uint8,
        )
        colors = np.tile(_idle_rgb, (grid.n_cells, 1))
        grid["colors"] = colors
        grid.cell_data["entity_tag"] = np.array(all_entity_tags, dtype=np.int64)

        show_edges = show_surface_edges and dim >= 2
        opacity = surface_opacity if dim >= 2 else 1.0
        dim_kwargs: dict[str, Any] = dict(
            scalars="colors", rgb=True,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=line_width if dim == 1 else 0.5,
            render_lines_as_tubes=(dim == 1),
            smooth_shading=_pref.smooth_shading, pickable=True,
        )
        # Flat matte + silhouette on dim=2/3 — mirrors brep_scene so the
        # two viewers present the same CAD-style outline.
        if dim >= 2:
            dim_kwargs.update(
                diffuse=0.9, specular=0.0,
                silhouette=dict(
                    color=_pal.outline_color,
                    line_width=(
                        _pal.outline_silhouette_px if dim == 3
                        else _pal.outline_feature_px
                    ),
                    feature_angle=_pref.feature_angle,
                ),
            )
        actor = plotter.add_mesh(grid, reset_camera=False, **dim_kwargs)

        registry.register_dim(
            dim, grid, actor, cell_to_dt, centroids_dim,
            add_mesh_kwargs=dim_kwargs,
        )
        batch_cell_to_elem[dim] = {
            i: etag for i, etag in enumerate(all_elem_tags_flat)
        }
        n_actors += 1

    plotter.reset_camera()  # type: ignore[call-arg]  # pyvista stub quirk
    t_actors_elapsed = time.perf_counter() - t_actors

    # ── node cloud ──────────────────────────────────────────────────
    # Show ALL mesh nodes — including embedded reference points that
    # may not appear in the displayed element connectivity.
    filt_tags = node_tags
    filt_coords = node_coords

    node_cloud = None
    node_actor = None
    if len(filt_coords) > 0:
        node_cloud, node_actor = build_node_cloud(
            plotter, filt_coords,
            model_diagonal=diag,
            marker_size=node_marker_size,
            color=node_color,
        )

    # KD-tree for nearest-node picking
    node_tree = None
    if len(filt_coords) > 0:
        try:
            from scipy.spatial import cKDTree
            node_tree = cKDTree(filt_coords)
        except ImportError:
            pass

    # ── physical group mappings ─────────────────────────────────────
    group_to_breps: dict[str, list[DimTag]] = {}
    brep_to_group: dict[DimTag, str] = {}
    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        except Exception:
            pg_name = f"Group_{pg_dim}_{pg_tag}"
        ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
        breps = [(pg_dim, int(t)) for t in ents]
        group_to_breps[pg_name] = breps
        for dt in breps:
            if dt not in brep_to_group:
                brep_to_group[dt] = pg_name

    # ── profiling ───────────────────────────────────────────────────
    if verbose:
        total = time.perf_counter() - t0
        n_nodes = len(filt_tags)
        entities = {d: len(gmsh.model.getEntities(d)) for d in dims}
        print(f"\n[mesh_scene] Built in {total:.2f}s  "
              f"({n_actors} actors, {n_nodes} nodes)")
        print(f"  Entities: {entities}")
        print(f"  Node setup    : {t_setup:.3f}s")
        print(f"  Actor creation: {t_actors_elapsed:.3f}s")
        print(f"  Remainder     : {total - t_setup - t_actors_elapsed:.3f}s")

    return MeshSceneData(
        registry=registry,
        node_tags=filt_tags,
        node_coords=filt_coords,
        node_tag_to_idx={int(t): i for i, t in enumerate(filt_tags)},
        tag_to_idx=tag_to_idx,
        model_diagonal=diag,
        elem_data=elem_data,
        elem_to_brep=elem_to_brep,
        brep_to_elems=brep_to_elems,
        brep_dominant_type=brep_dominant_type,
        batch_cell_to_elem=batch_cell_to_elem,
        node_cloud=node_cloud,
        node_actor=node_actor,
        node_tree=node_tree,
        group_to_breps=group_to_breps,
        brep_to_group=brep_to_group,
    )
