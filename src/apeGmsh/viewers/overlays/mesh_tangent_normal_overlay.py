"""
Mesh tangent / normal overlay runtime manager.

Per-entity orientation glyphs computed from **mesh node positions**
(not BRep parametrics). One arrow per geometric entity:

* Curve tangents (dim=1) — length-weighted mean of per-element edge
  tangents, placed at the centroid of element midpoints.
* Surface normals (dim=2) — area-weighted mean of per-element corner
  normals (cross-products of corner edges), placed at the centroid of
  element centroids.

Use this when you want to verify the actual meshed orientation; for
the smooth CAD orientation see ``TangentNormalOverlay`` (BRep version).

Only corner (primary) nodes are used, so high-order elements (Q9, Q8,
T6, line3, …) work correctly.

Usage::

    overlay = MeshTangentNormalOverlay(
        plotter,
        origin_shift=registry.origin_shift,
        model_diagonal=diag,
    )
    overlay.set_show_tangents(True)
    overlay.set_show_normals(True)
"""
from __future__ import annotations

from typing import Any

import gmsh
import numpy as np


class MeshTangentNormalOverlay:
    """Live manager for the mesh-derived tangent / normal overlay."""

    def __init__(
        self,
        plotter: Any,
        *,
        origin_shift: np.ndarray,
        model_diagonal: float,
        show_tangents: bool = False,
        show_normals: bool = False,
        scale: float = 0.05,
    ) -> None:
        self.plotter = plotter
        self.origin_shift = np.asarray(origin_shift, dtype=np.float64)
        self.model_diagonal = float(model_diagonal) or 1.0
        self.show_tangents = bool(show_tangents)
        self.show_normals = bool(show_normals)
        self.scale = float(scale)

        self._tangent_actor: Any = None
        self._normal_actor: Any = None

        if self.show_tangents:
            self._build_tangent_actor()
        if self.show_normals:
            self._build_normal_actor()

    # ── public API ───────────────────────────────────────────────────

    def set_show_tangents(self, b: bool) -> None:
        self.show_tangents = bool(b)
        if self.show_tangents:
            if self._tangent_actor is None:
                self._build_tangent_actor()
            else:
                self._tangent_actor.SetVisibility(True)
        elif self._tangent_actor is not None:
            self._tangent_actor.SetVisibility(False)
        self._render()

    def set_show_normals(self, b: bool) -> None:
        self.show_normals = bool(b)
        if self.show_normals:
            if self._normal_actor is None:
                self._build_normal_actor()
            else:
                self._normal_actor.SetVisibility(True)
        elif self._normal_actor is not None:
            self._normal_actor.SetVisibility(False)
        self._render()

    def set_scale(self, s: float) -> None:
        s = float(s)
        if s == self.scale:
            return
        self.scale = s
        self._invalidate(rebuild_visible=True)

    def set_origin_shift(self, shift: np.ndarray) -> None:
        self.origin_shift = np.asarray(shift, dtype=np.float64)
        self._invalidate(rebuild_visible=True)

    def set_model_diagonal(self, diag: float) -> None:
        self.model_diagonal = float(diag) or 1.0
        self._invalidate(rebuild_visible=True)

    def refresh_theme(self) -> None:
        """Rebuild visible actors so they pick up the new theme palette."""
        self._invalidate(rebuild_visible=True)

    # ── internal: lifecycle ──────────────────────────────────────────

    def _invalidate(self, *, rebuild_visible: bool) -> None:
        self._remove_tangent_actor()
        self._remove_normal_actor()
        if rebuild_visible:
            if self.show_tangents:
                self._build_tangent_actor()
            if self.show_normals:
                self._build_normal_actor()
        self._render()

    def _remove_tangent_actor(self) -> None:
        if self._tangent_actor is not None:
            try:
                self.plotter.remove_actor(self._tangent_actor)
            except Exception:
                pass
            self._tangent_actor = None

    def _remove_normal_actor(self) -> None:
        if self._normal_actor is not None:
            try:
                self.plotter.remove_actor(self._normal_actor)
            except Exception:
                pass
            self._normal_actor = None

    def _render(self) -> None:
        try:
            self.plotter.render()
        except Exception:
            pass

    # ── internal: node-coord cache ───────────────────────────────────

    def _node_coord_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(tag_to_idx, coords)`` for fast tag → xyz lookup.

        Uses ``-1`` as the sentinel for unused slots in ``tag_to_idx``.
        """
        tags_raw, coords_flat, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags_raw, dtype=np.int64)
        coords = np.asarray(coords_flat, dtype=np.float64).reshape(-1, 3)
        if len(tags) == 0:
            return np.array([], dtype=np.int64), coords
        max_tag = int(tags.max())
        tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
        tag_to_idx[tags] = np.arange(len(tags), dtype=np.int64)
        return tag_to_idx, coords

    @staticmethod
    def _primary_count(etype: int) -> int:
        """Number of corner (primary) nodes for a Gmsh element type."""
        try:
            props = gmsh.model.mesh.getElementProperties(int(etype))
            return int(props[5])
        except Exception:
            return 0

    # ── internal: builders ───────────────────────────────────────────

    def _build_tangent_actor(self) -> None:
        tag_to_idx, coords = self._node_coord_map()
        if len(tag_to_idx) == 0:
            return

        positions, vectors, lengths = [], [], []
        for _, tag in gmsh.model.getEntities(dim=1):
            tangent, centroid = self._entity_tangent(tag, tag_to_idx, coords)
            if tangent is None:
                continue
            length = self._entity_arrow_length(1, tag)
            positions.append(centroid - self.origin_shift)
            vectors.append(tangent)
            lengths.append(length)

        self._tangent_actor = self._make_glyph_actor(
            positions, vectors, lengths, color_key="success",
        )

    def _build_normal_actor(self) -> None:
        tag_to_idx, coords = self._node_coord_map()
        if len(tag_to_idx) == 0:
            return

        positions, vectors, lengths = [], [], []
        for _, tag in gmsh.model.getEntities(dim=2):
            normal, centroid = self._entity_normal(tag, tag_to_idx, coords)
            if normal is None:
                continue
            length = self._entity_arrow_length(2, tag)
            positions.append(centroid - self.origin_shift)
            vectors.append(normal)
            lengths.append(length)

        self._normal_actor = self._make_glyph_actor(
            positions, vectors, lengths, color_key="warning",
        )

    # ── internal: per-entity computation ─────────────────────────────

    def _entity_tangent(
        self,
        tag: int,
        tag_to_idx: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Length-weighted mean tangent of all line elements on a curve."""
        try:
            etypes, _, enode_lists = gmsh.model.mesh.getElements(1, tag)
        except Exception:
            return None, None
        if len(etypes) == 0:
            return None, None

        accum_vec = np.zeros(3, dtype=np.float64)
        accum_pos = np.zeros(3, dtype=np.float64)
        accum_w = 0.0

        max_tag = len(tag_to_idx) - 1
        for etype, enodes in zip(etypes, enode_lists):
            n_prim = self._primary_count(int(etype))
            if n_prim < 2:
                continue
            n_nodes = int(gmsh.model.mesh.getElementProperties(int(etype))[3])
            arr = np.asarray(enodes, dtype=np.int64).reshape(-1, n_nodes)
            corner = arr[:, :n_prim]
            if max_tag < 0 or np.any(corner > max_tag) or np.any(corner < 0):
                continue
            idx = tag_to_idx[corner]
            if np.any(idx < 0):
                continue
            p0 = coords[idx[:, 0]]
            p1 = coords[idx[:, 1]]
            edge = p1 - p0
            length = np.linalg.norm(edge, axis=1)
            valid = length > 1e-12
            if not np.any(valid):
                continue
            edge = edge[valid]
            length = length[valid]
            unit = edge / length[:, None]
            mid = 0.5 * (p0[valid] + p1[valid])
            accum_vec += (unit * length[:, None]).sum(axis=0)
            accum_pos += (mid * length[:, None]).sum(axis=0)
            accum_w += float(length.sum())

        if accum_w < 1e-12:
            return None, None
        mag = np.linalg.norm(accum_vec)
        if mag < 1e-12:
            return None, None
        return accum_vec / mag, accum_pos / accum_w

    def _entity_normal(
        self,
        tag: int,
        tag_to_idx: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Area-weighted mean normal of all face elements on a surface."""
        try:
            etypes, _, enode_lists = gmsh.model.mesh.getElements(2, tag)
        except Exception:
            return None, None
        if len(etypes) == 0:
            return None, None

        accum_vec = np.zeros(3, dtype=np.float64)
        accum_pos = np.zeros(3, dtype=np.float64)
        accum_w = 0.0

        max_tag = len(tag_to_idx) - 1
        for etype, enodes in zip(etypes, enode_lists):
            n_prim = self._primary_count(int(etype))
            if n_prim < 3:
                continue
            n_nodes = int(gmsh.model.mesh.getElementProperties(int(etype))[3])
            arr = np.asarray(enodes, dtype=np.int64).reshape(-1, n_nodes)
            corner = arr[:, :n_prim]
            if max_tag < 0 or np.any(corner > max_tag) or np.any(corner < 0):
                continue
            idx = tag_to_idx[corner]
            if np.any(idx < 0):
                continue

            p0 = coords[idx[:, 0]]
            p1 = coords[idx[:, 1]]
            # For tris use node 2 as second edge end; for quads use the
            # opposite corner (node 3) so the cross-product spans the
            # full diagonal — robust for skewed quads.
            opp = 2 if n_prim == 3 else 3
            p_opp = coords[idx[:, opp]]
            n_vec = np.cross(p1 - p0, p_opp - p0)
            area2 = np.linalg.norm(n_vec, axis=1)
            valid = area2 > 1e-12
            if not np.any(valid):
                continue
            n_vec = n_vec[valid]
            area2 = area2[valid]
            unit = n_vec / area2[:, None]
            centroid = coords[idx].mean(axis=1)[valid]
            accum_vec += (unit * area2[:, None]).sum(axis=0)
            accum_pos += (centroid * area2[:, None]).sum(axis=0)
            accum_w += float(area2.sum())

        if accum_w < 1e-12:
            return None, None
        mag = np.linalg.norm(accum_vec)
        if mag < 1e-12:
            return None, None
        return accum_vec / mag, accum_pos / accum_w

    # ── internal: helpers ────────────────────────────────────────────

    def _entity_arrow_length(self, dim: int, tag: int) -> float:
        """Per-entity arrow length: bbox diag * 0.5, capped by global scale."""
        global_len = self.model_diagonal * self.scale
        try:
            bb = gmsh.model.getBoundingBox(dim, tag)
            ent_diag = float(np.linalg.norm(
                [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
            ))
            if ent_diag > 0:
                return min(global_len, ent_diag * 0.5)
        except Exception:
            pass
        return global_len

    def _make_glyph_actor(
        self,
        positions: list,
        vectors: list,
        lengths: list,
        *,
        color_key: str,
    ) -> Any:
        if not positions:
            return None
        import pyvista as pv
        from ..ui.theme import THEME

        pts = np.asarray(positions, dtype=np.float64)
        vecs = np.asarray(vectors, dtype=np.float64)
        scl = np.asarray(lengths, dtype=np.float64)

        cloud = pv.PolyData(pts)
        cloud["vectors"] = vecs
        cloud["scale"] = scl

        arrow = pv.Arrow(
            tip_length=0.25,
            tip_radius=0.10,
            shaft_radius=0.035,
        )
        glyphs = cloud.glyph(
            orient="vectors",
            scale="scale",
            factor=1.0,
            geom=arrow,
        )

        color = getattr(THEME.current, color_key)
        actor = self.plotter.add_mesh(
            glyphs,
            color=color,
            lighting=False,
            pickable=False,
            render=False,
        )
        return actor
