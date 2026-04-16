"""
constraint_overlay — Build PyVista actors for constraint visualization.
========================================================================

Pure functions that take FEM data + visual parameters and return
PyVista meshes ready for ``plotter.add_mesh()``.  No Qt, no plotter
reference, no closures — testable in isolation.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pyvista as pv

from apeGmsh.mesh._record_set import ConstraintKind

_log = logging.getLogger(__name__)


# =====================================================================
# Coordinate helper
# =====================================================================

def _node_coords_shifted(fem, nid: int, origin: np.ndarray):
    """Look up shifted coordinates for a node ID.  Returns None on miss."""
    try:
        return fem.nodes.coords[fem.nodes.index(int(nid))] - origin
    except (KeyError, IndexError):
        return None


# =====================================================================
# Node-pair constraint geometry
# =====================================================================

def build_node_pair_actors(
    fem,
    active_kinds: set[str],
    origin: np.ndarray,
    marker_radius: float,
    line_width: int,
    color_fn,
) -> list[tuple]:
    """Build PyVista meshes for node-pair constraints.

    Returns a list of ``(mesh_or_glyphs, add_mesh_kwargs)`` tuples.
    The caller does ``plotter.add_mesh(mesh, **kwargs)`` for each.

    Single pass over ``pairs()`` — records grouped by kind first,
    then geometry built per kind.
    """
    from apeGmsh.solvers.Constraints import (
        NodePairRecord, NodeToSurfaceRecord,
    )

    # ── Collect node-pair records (expanded) ────────────────────
    # NodeToSurfaceRecord.expand() yields sub-records with
    # kind="rigid_beam" / "equal_dof", not "node_to_surface".
    # We need to map those back so the checkbox filter works.
    by_kind: dict[str, list] = defaultdict(list)
    for rec in fem.nodes.constraints.pairs():
        if rec.kind in active_kinds:
            by_kind[rec.kind].append(rec)

    # ── node_to_surface: draw master→slave lines directly ──────
    # The expanded sub-records (rigid_beam, equal_dof) are for
    # solvers; for visualisation we want the high-level topology.
    if ConstraintKind.NODE_TO_SURFACE in active_kinds:
        for raw in fem.nodes.constraints:
            if isinstance(raw, NodeToSurfaceRecord):
                for slave_tag in raw.slave_nodes:
                    by_kind["node_to_surface"].append(
                        NodePairRecord(
                            kind="node_to_surface",
                            master_node=raw.master_node,
                            slave_node=slave_tag,
                        ))

    result: list[tuple] = []

    for kind, records in by_kind.items():
        line_pts = []
        line_cells = []
        master_positions: dict[int, np.ndarray] = {}
        idx = 0

        for rec in records:
            p1 = _node_coords_shifted(fem, rec.master_node, origin)
            p2 = _node_coords_shifted(fem, rec.slave_node, origin)
            if p1 is None or p2 is None:
                continue
            line_pts.extend([p1, p2])
            line_cells.extend([2, idx, idx + 1])
            idx += 2
            if rec.master_node not in master_positions:
                master_positions[rec.master_node] = p1

        color = color_fn(kind)

        # Line segments
        if line_pts:
            pts_arr = np.array(line_pts, dtype=float)
            cells_arr = np.array(line_cells, dtype=np.int64)
            poly = pv.PolyData(pts_arr, lines=cells_arr)
            result.append((poly, dict(
                color=color, line_width=line_width,
                render_lines_as_tubes=True,
                name=f"_cst_lines_{kind}",
                reset_camera=False, pickable=False,
            )))

        # Master node spheres
        if master_positions:
            cloud = pv.PolyData(
                np.array(list(master_positions.values()), dtype=float))
            sphere = pv.Sphere(
                radius=marker_radius,
                theta_resolution=8, phi_resolution=8)
            glyphs = cloud.glyph(geom=sphere, orient=False, scale=False)
            result.append((glyphs, dict(
                color=color, lighting=False,
                name=f"_cst_masters_{kind}",
                reset_camera=False, pickable=False,
            )))

    # Phantom node markers are rendered permanently by MeshViewer
    # (diamond glyphs in the node cloud), so no overlay needed here.

    return result


# =====================================================================
# Surface constraint geometry
# =====================================================================

def build_surface_actors(
    fem,
    active_kinds: set[str],
    origin: np.ndarray,
    line_width: int,
    color_fn,
) -> list[tuple]:
    """Build PyVista meshes for surface constraints.

    Returns a list of ``(mesh_or_glyphs, add_mesh_kwargs)`` tuples.
    """
    # Single pass: group interpolations by kind
    by_kind: dict[str, list] = defaultdict(list)
    for rec in fem.elements.constraints.interpolations():
        if rec.kind in active_kinds:
            by_kind[rec.kind].append(rec)

    result: list[tuple] = []

    for kind, records in by_kind.items():
        interp_pts = []
        interp_cells = []
        idx = 0

        for rec in records:
            slave_pt = _node_coords_shifted(fem, rec.slave_node, origin)
            if slave_pt is None:
                continue
            master_pts = []
            for mnid in rec.master_nodes:
                mp = _node_coords_shifted(fem, mnid, origin)
                if mp is not None:
                    master_pts.append(mp)
            if not master_pts:
                continue
            weights = rec.weights
            if weights is not None and len(weights) == len(master_pts):
                centroid = np.average(
                    master_pts, axis=0, weights=weights)
            else:
                centroid = np.mean(master_pts, axis=0)
            interp_pts.extend([slave_pt, centroid])
            interp_cells.extend([2, idx, idx + 1])
            idx += 2

        color = color_fn(kind)

        if interp_pts:
            pts_arr = np.array(interp_pts, dtype=float)
            cells_arr = np.array(interp_cells, dtype=np.int64)
            poly = pv.PolyData(pts_arr, lines=cells_arr)
            result.append((poly, dict(
                color=color, line_width=line_width,
                render_lines_as_tubes=True, opacity=0.7,
                name=f"_cst_interp_{kind}",
                reset_camera=False, pickable=False,
            )))

    # Surface coupling highlights
    for coup in fem.elements.constraints.couplings():
        if coup.kind not in active_kinds:
            continue
        color = color_fn(coup.kind)
        for node_set, suffix, opac in [
            (coup.master_nodes, "master", 0.25),
            (coup.slave_nodes, "slave", 0.25),
        ]:
            face_pts = []
            for nid in node_set:
                pt = _node_coords_shifted(fem, nid, origin)
                if pt is not None:
                    face_pts.append(pt)
            if len(face_pts) < 3:
                continue
            cloud = pv.PolyData(np.array(face_pts, dtype=float))
            try:
                surf = cloud.delaunay_2d()
            except Exception as exc:
                _log.warning(
                    "delaunay_2d failed for %s %s coupling "
                    "(%d points): %s",
                    coup.kind, suffix, len(face_pts), exc,
                )
                continue
            result.append((surf, dict(
                color=color, opacity=opac,
                name=f"_cst_surf_{coup.kind}_{suffix}_{id(coup)}",
                reset_camera=False, pickable=False,
            )))

    return result
