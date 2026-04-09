"""
BRep Scene Builder — construct batched VTK actors from Gmsh BRep geometry.

Generates a temporary coarse mesh (if none exists), then extracts
per-entity triangulation via ``getNodes``/``getElements``.  Produces
one merged actor per dimension with ``cell_data["colors"]`` for
per-entity recoloring and ``cell_data["entity_tag"]`` for picking.

Usage::

    from pyGmsh.viewers.scene.brep_scene import build_brep_scene
    registry = build_brep_scene(plotter, dims=[0, 1, 2])
"""
from __future__ import annotations

import time
from typing import Any

import gmsh
import numpy as np
import pyvista as pv

from ..core.entity_registry import DimTag, EntityRegistry
from ..core.color_manager import IDLE_COLORS
from .glyph_points import build_point_glyphs


# ======================================================================
# Surface tessellation from global mesh
# ======================================================================

def _surface_polydata_from_global_mesh(
    node_coords: np.ndarray,
    tag_to_idx: np.ndarray,
    elem_types,
    enodes_list,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Build one surface entity from global mesh nodes + connectivity.

    Uses the global node array so triangles referencing nodes on
    embedded curves/points are resolved correctly (no holes).
    """
    if len(tag_to_idx) == 0:
        return np.empty((0, 3), dtype=np.float64), [], 0

    face_specs: list[tuple[int, np.ndarray]] = []
    n_cells = 0
    max_tag = len(tag_to_idx) - 1

    for etype, enodes in zip(elem_types, enodes_list):
        etype = int(etype)
        if etype == 2:
            npe = 3
        elif etype == 3:
            npe = 4
        else:
            continue

        enodes_arr = np.asarray(enodes, dtype=np.int64)
        if len(enodes_arr) == 0:
            continue

        n_elems = len(enodes_arr) // npe
        node_mat = enodes_arr.reshape(n_elems, npe)
        in_range = (
            np.all(node_mat >= 0, axis=1)
            & np.all(node_mat <= max_tag, axis=1)
        )
        if not np.any(in_range):
            continue

        node_mat = node_mat[in_range]
        idx_mat = tag_to_idx[node_mat]
        valid = np.all(idx_mat >= 0, axis=1)
        if not np.any(valid):
            continue

        node_mat = node_mat[valid]
        face_specs.append((npe, node_mat))
        n_cells += len(node_mat)

    if not face_specs:
        return np.empty((0, 3), dtype=np.float64), [], 0

    all_tags = np.concatenate([nm.ravel() for _, nm in face_specs])
    unique_tags, inverse = np.unique(all_tags, return_inverse=True)
    local_pts = node_coords[tag_to_idx[unique_tags]]

    faces_parts: list[np.ndarray] = []
    cursor = 0
    for npe, node_mat in face_specs:
        count = node_mat.size
        local_idx = inverse[cursor:cursor + count].reshape(-1, npe)
        prefix = np.full((len(local_idx), 1), npe, dtype=np.int64)
        faces_parts.append(np.hstack([prefix, local_idx]).ravel())
        cursor += count

    return local_pts, faces_parts, n_cells


# ======================================================================
# Temp mesh generation
# ======================================================================

def _generate_temp_mesh(diag: float) -> None:
    """Generate a coarse 2D mesh with per-curve adaptive sizing."""
    _saved: dict[str, float] = {}
    for key in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax",
                "Mesh.Algorithm", "Mesh.MeshSizeExtendFromBoundary"):
        try:
            _saved[key] = gmsh.option.getNumber(key)
        except Exception:
            pass
    try:
        max_size = diag * 0.03
        for _, ctag in gmsh.model.getEntities(dim=1):
            try:
                mass = gmsh.model.occ.getMass(1, ctag)
                curve_size = min(max_size, mass / 3.0)
                if curve_size > 0:
                    bnd = gmsh.model.getBoundary([(1, ctag)], combined=False)
                    pt_tags = [(d, abs(t)) for d, t in bnd if d == 0]
                    if pt_tags:
                        gmsh.model.mesh.setSize(pt_tags, curve_size)
            except Exception:
                pass

        gmsh.option.setNumber("Mesh.MeshSizeMin", diag * 0.0005)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
    except Exception:
        pass
    finally:
        try:
            all_pts = gmsh.model.getEntities(dim=0)
            if all_pts:
                gmsh.model.mesh.setSize(all_pts, 0.0)
        except Exception:
            pass
        for key, val in _saved.items():
            try:
                gmsh.option.setNumber(key, val)
            except Exception:
                pass


# ======================================================================
# Public API
# ======================================================================

def _compute_entity_bboxes(dims: list[int]) -> dict:
    """Compute AABB corners for all entities from Gmsh bounding boxes."""
    bboxes = {}
    for dim in dims:
        for _, tag in gmsh.model.getEntities(dim=dim):
            try:
                bb = gmsh.model.getBoundingBox(dim, tag)
                xmin, ymin, zmin = bb[0], bb[1], bb[2]
                xmax, ymax, zmax = bb[3], bb[4], bb[5]
                corners = np.array([
                    [xmin, ymin, zmin], [xmax, ymin, zmin],
                    [xmin, ymax, zmin], [xmax, ymax, zmin],
                    [xmin, ymin, zmax], [xmax, ymin, zmax],
                    [xmin, ymax, zmax], [xmax, ymax, zmax],
                ])
                bboxes[(dim, tag)] = corners
            except Exception:
                pass
    return bboxes


def build_brep_scene(
    plotter: pv.Plotter,
    dims: list[int],
    *,
    point_size: float = 10.0,
    line_width: float = 6.0,
    surface_opacity: float = 0.35,
    show_surface_edges: bool = False,
) -> EntityRegistry:
    """Build batched BRep actors and return an :class:`EntityRegistry`.

    Parameters
    ----------
    plotter : pv.Plotter
        Target plotter / QtInteractor.
    dims : list[int]
        Which entity dimensions to render (e.g. ``[0, 1, 2]``).
    point_size, line_width, surface_opacity, show_surface_edges
        Visual properties.

    Returns
    -------
    EntityRegistry
    """
    t0 = time.perf_counter()
    registry = EntityRegistry()

    # ── model diagonal ──────────────────────────────────────────────
    try:
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm(
            [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
        ))
        if diag <= 0.0:
            diag = 1.0
    except Exception:
        diag = 1.0

    # ── generate temp mesh if needed ────────────────────────────────
    t_mesh = time.perf_counter()
    had_mesh = False
    try:
        existing, _, _ = gmsh.model.mesh.getNodes()
        had_mesh = len(existing) > 0
    except Exception:
        pass

    if not had_mesh:
        _generate_temp_mesh(diag)
    t_mesh_elapsed = time.perf_counter() - t_mesh

    # ── fetch global node array ─────────────────────────────────────
    try:
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
    except Exception:
        return registry
    if len(all_tags) == 0:
        return registry

    all_node_tags = np.asarray(all_tags, dtype=np.int64)
    all_node_coords = np.asarray(all_coords, dtype=np.float64).reshape(-1, 3)
    global_tag_to_idx = np.full(
        int(all_node_tags.max()) + 1, -1, dtype=np.int64,
    )
    global_tag_to_idx[all_node_tags] = np.arange(
        len(all_node_tags), dtype=np.int64,
    )

    # ── precompute bounding boxes from Gmsh ───────────────────────
    all_bboxes = _compute_entity_bboxes(dims)

    n_entities = 0

    # ── dim=0: point glyphs ─────────────────────────────────────────
    t_dim = time.perf_counter()
    n_d0 = 0
    if 0 in dims:
        centers, tags_d0 = [], []
        for _, tag in gmsh.model.getEntities(dim=0):
            try:
                ntags, ncoords, _ = gmsh.model.mesh.getNodes(dim=0, tag=tag)
                if len(ntags) == 0:
                    continue
                xyz = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)[0]
                centers.append(xyz)
                tags_d0.append(tag)
                n_d0 += 1
            except Exception:
                pass
        if centers:
            mesh, actor, cell_to_dt, centroids = build_point_glyphs(
                plotter, np.array(centers), tags_d0,
                model_diagonal=diag,
                point_size=point_size,
                idle_color=IDLE_COLORS[0],
            )
            d0_bboxes = {dt: all_bboxes[dt] for dt in centroids if dt in all_bboxes}
            registry.register_dim(0, mesh, actor, cell_to_dt, centroids, d0_bboxes)
    t_d0 = time.perf_counter() - t_dim
    n_entities += n_d0

    # ── dim=1: merged curves ────────────────────────────────────────
    t_dim = time.perf_counter()
    n_d1 = 0
    if 1 in dims:
        pts_parts: list[np.ndarray] = []
        lines_parts: list[np.ndarray] = []
        etags: list[int] = []
        dt_cells: dict[DimTag, list[int]] = {}
        centroids_d1: dict[DimTag, np.ndarray] = {}
        cell_off = 0
        pt_off = 0
        for _, tag in gmsh.model.getEntities(dim=1):
            try:
                ntags, ncoords, nparams = gmsh.model.mesh.getNodes(
                    dim=1, tag=tag, includeBoundary=True,
                )
                if len(ntags) < 2:
                    continue
                pts = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
                # Sort nodes by parametric coordinate along the curve
                params = np.asarray(nparams, dtype=np.float64)
                if len(params) == len(pts):
                    order = np.argsort(params)
                    pts = pts[order]
                n = len(pts)
                n_lines = n - 1
                idx = np.arange(n_lines, dtype=np.int64)
                lines = np.empty(n_lines * 3, dtype=np.int64)
                lines[0::3] = 2
                lines[1::3] = idx + pt_off
                lines[2::3] = idx + pt_off + 1
                pts_parts.append(pts)
                lines_parts.append(lines)
                cell_indices = list(range(cell_off, cell_off + n_lines))
                dt: DimTag = (1, tag)
                dt_cells[dt] = cell_indices
                etags.extend([tag] * n_lines)
                centroids_d1[dt] = pts.mean(axis=0)
                cell_off += n_lines
                pt_off += n
                n_d1 += 1
            except Exception:
                pass
        if pts_parts:
            merged_pts = np.vstack(pts_parts)
            merged_lines = np.concatenate(lines_parts)
            poly = pv.PolyData()
            poly.points = merged_pts
            poly.lines = merged_lines
            colors = np.tile(IDLE_COLORS[1], (len(etags), 1))
            poly.cell_data["entity_tag"] = np.array(etags, dtype=np.int64)
            poly.cell_data["colors"] = colors
            actor = plotter.add_mesh(
                poly, scalars="colors", rgb=True,
                line_width=line_width,
                render_lines_as_tubes=True,
                pickable=True, reset_camera=False,
            )
            cell_to_dt_d1 = {}
            for dt, cells in dt_cells.items():
                for ci in cells:
                    cell_to_dt_d1[ci] = dt
            d1_bboxes = {dt: all_bboxes[dt] for dt in centroids_d1 if dt in all_bboxes}
            registry.register_dim(1, poly, actor, cell_to_dt_d1, centroids_d1, d1_bboxes)
    t_d1 = time.perf_counter() - t_dim
    n_entities += n_d1

    # ── dim=2: merged surfaces ──────────────────────────────────────
    t_dim = time.perf_counter()
    n_d2 = 0
    if 2 in dims:
        pts_parts = []
        faces_parts: list[np.ndarray] = []
        etags = []
        dt_cells = {}
        centroids_d2: dict[DimTag, np.ndarray] = {}
        cell_off = 0
        pt_off = 0
        for _, tag in gmsh.model.getEntities(dim=2):
            try:
                ets, _, enl = gmsh.model.mesh.getElements(2, tag)
                lpts, entity_faces, n_cells_e = (
                    _surface_polydata_from_global_mesh(
                        all_node_coords, global_tag_to_idx, ets, enl,
                    )
                )
                if n_cells_e == 0:
                    continue
                dt = (2, tag)
                cell_indices = list(range(cell_off, cell_off + n_cells_e))
                dt_cells[dt] = cell_indices
                centroids_d2[dt] = lpts.mean(axis=0)
                pts_parts.append(lpts)
                for f in entity_faces:
                    shifted = f.copy()
                    shifted.reshape(-1, shifted[0] + 1)[:, 1:] += pt_off
                    faces_parts.append(shifted)
                etags.extend([tag] * n_cells_e)
                cell_off += n_cells_e
                pt_off += len(lpts)
                n_d2 += 1
            except Exception:
                pass
        if pts_parts:
            merged_pts = np.vstack(pts_parts)
            merged_faces = np.concatenate(faces_parts)
            poly = pv.PolyData(merged_pts, faces=merged_faces)
            colors = np.tile(IDLE_COLORS[2], (len(etags), 1))
            poly.cell_data["entity_tag"] = np.array(etags, dtype=np.int64)
            poly.cell_data["colors"] = colors
            actor = plotter.add_mesh(
                poly, scalars="colors", rgb=True,
                opacity=surface_opacity,
                show_edges=show_surface_edges,
                edge_color="#2C4A6E",
                line_width=0.5,
                smooth_shading=True,
                pickable=True, reset_camera=False,
            )
            cell_to_dt_d2 = {}
            for dt, cells in dt_cells.items():
                for ci in cells:
                    cell_to_dt_d2[ci] = dt
            d2_bboxes = {dt: all_bboxes[dt] for dt in centroids_d2 if dt in all_bboxes}
            registry.register_dim(2, poly, actor, cell_to_dt_d2, centroids_d2, d2_bboxes)
    t_d2 = time.perf_counter() - t_dim
    n_entities += n_d2

    # dim=3 not implemented yet (same merge pattern on boundary surfaces)

    plotter.reset_camera()

    # ── cleanup temp mesh ───────────────────────────────────────────
    if not had_mesh:
        try:
            gmsh.model.mesh.clear()
        except Exception:
            pass

    # ── profiling ───────────────────────────────────────────────────
    total = time.perf_counter() - t0
    print(f"\n[brep_scene] Built in {total:.2f}s  "
          f"({len(registry.dims)} actors, {n_entities} entities)")
    print(f"  Mesh generate : {t_mesh_elapsed:.3f}s"
          f"  {'(existing)' if had_mesh else '(temp coarse)'}")
    print(f"  dim0 points   : {t_d0:.3f}s  ({n_d0})")
    print(f"  dim1 curves   : {t_d1:.3f}s  ({n_d1})")
    print(f"  dim2 surfaces : {t_d2:.3f}s  ({n_d2})")

    return registry
