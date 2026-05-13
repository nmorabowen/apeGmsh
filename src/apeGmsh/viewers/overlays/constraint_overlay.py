"""
constraint_overlay — Build PyVista actors for constraint visualization.
========================================================================

Pure functions that take a :class:`ViewerData` snapshot + visual
parameters and return PyVista meshes ready for ``plotter.add_mesh()``.
No Qt, no plotter reference, no closures — testable in isolation.

Phase 8.7 commit 6 retargets the inputs from raw :class:`FEMData` /
:class:`apeGmsh.mesh.records._constraints` types onto
:class:`apeGmsh.viewers.data.ViewerData` plus the read-side row
dataclasses.  Constraint ``kind`` values are plain strings; the
``NODE_TO_SURFACE_KIND`` sentinel and the
``NODE_PAIR_KINDS`` / ``SURFACE_KINDS`` classifier frozensets are
re-exported from :mod:`apeGmsh.viewers.data` so consumers don't reach
back into ``apeGmsh.mesh``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from apeGmsh.viewers.data import (
    NODE_TO_SURFACE_KIND,
    NodePairRow,
    NodeToSurfaceRow,
)

if TYPE_CHECKING:
    from apeGmsh.viewers.data import ViewerData

_log = logging.getLogger(__name__)


# =====================================================================
# Coordinate helper
# =====================================================================

def _node_coords_shifted(
    view: "ViewerData", nid: int, origin: np.ndarray,
) -> "np.ndarray | None":
    """Look up shifted coordinates for a node ID.  Returns None on miss."""
    try:
        return view.nodes.coords[view.nodes.index(int(nid))] - origin
    except (KeyError, IndexError):
        return None


# =====================================================================
# Node-pair constraint geometry
# =====================================================================

def build_node_pair_actors(
    view: "ViewerData",
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
    # ── Collect node-pair rows (expanded) ───────────────────────
    # NodeConstraintView.pairs() expands NodeGroup and NodeToSurface
    # rows into per-slave NodePairRow instances — same emission order
    # the FEMData side documented.
    by_kind: dict[str, list[NodePairRow]] = defaultdict(list)
    for row in view.nodes.constraints.pairs():
        if row.kind in active_kinds:
            by_kind[row.kind].append(row)

    # ── node_to_surface: draw master→slave lines directly ──────
    # The expanded sub-rows (rigid_beam, equal_dof) are for solver
    # emission; for visualisation we want the high-level topology.
    if NODE_TO_SURFACE_KIND in active_kinds:
        for raw in view.nodes.constraints:
            if isinstance(raw, NodeToSurfaceRow):
                for slave_tag in raw.slave_nodes:
                    by_kind[NODE_TO_SURFACE_KIND].append(
                        NodePairRow(
                            kind=NODE_TO_SURFACE_KIND,
                            master_node=raw.master_node,
                            slave_node=slave_tag,
                            dofs=(),
                        )
                    )

    result: list[tuple] = []

    for kind, rows in by_kind.items():
        line_pts = []
        line_cells = []
        master_positions: dict[int, np.ndarray] = {}
        idx = 0

        for row in rows:
            p1 = _node_coords_shifted(view, row.master_node, origin)
            p2 = _node_coords_shifted(view, row.slave_node, origin)
            if p1 is None or p2 is None:
                continue
            line_pts.extend([p1, p2])
            line_cells.extend([2, idx, idx + 1])
            idx += 2
            if row.master_node not in master_positions:
                master_positions[row.master_node] = p1

        color = color_fn(kind)

        # Line segments
        if line_pts:
            pts_arr = np.array(line_pts, dtype=float)
            cells_arr = np.array(line_cells, dtype=int)
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
    view: "ViewerData",
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
    for row in view.elements.constraints.interpolations():
        if row.kind in active_kinds:
            by_kind[row.kind].append(row)

    result: list[tuple] = []

    for kind, rows in by_kind.items():
        interp_pts = []
        interp_cells = []
        idx = 0

        for row in rows:
            slave_pt = _node_coords_shifted(view, row.slave_node, origin)
            if slave_pt is None:
                continue
            master_pts = []
            for mnid in row.master_nodes:
                mp = _node_coords_shifted(view, mnid, origin)
                if mp is not None:
                    master_pts.append(mp)
            if not master_pts:
                continue
            weights = row.weights
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
            cells_arr = np.array(interp_cells, dtype=int)
            poly = pv.PolyData(pts_arr, lines=cells_arr)
            result.append((poly, dict(
                color=color, line_width=line_width,
                render_lines_as_tubes=True, opacity=0.7,
                name=f"_cst_interp_{kind}",
                reset_camera=False, pickable=False,
            )))

    # Surface coupling highlights
    for coup in view.elements.constraints.couplings():
        if coup.kind not in active_kinds:
            continue
        color = color_fn(coup.kind)
        for node_set, suffix, opac in [
            (coup.master_nodes, "master", 0.25),
            (coup.slave_nodes, "slave", 0.25),
        ]:
            face_pts = []
            for nid in node_set:
                pt = _node_coords_shifted(view, nid, origin)
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
