"""
Tangent / normal overlay runtime manager.

Geometry-orientation glyphs for the BRep viewer:

* Curve tangents (dim=1) — one arrow per curve at its mid-parameter
  point, oriented along the parametric derivative.
* Surface normals (dim=2) — one arrow per surface at its centre of
  mass, oriented along ``gmsh.model.getNormal``.

Both glyph sets are decorative — they live outside the entity registry,
are never pickable, and are toggled independently. The overlay caches
its actors and only rebuilds when geometry, origin shift, scale, or the
active theme change.

Usage::

    overlay = TangentNormalOverlay(
        plotter,
        origin_shift=registry.origin_shift,
        model_diagonal=diag,
    )
    overlay.set_show_tangents(True)
    overlay.set_show_normals(True)
    overlay.set_origin_shift(new_shift)   # after parts fuse
"""
from __future__ import annotations

from typing import Any

import gmsh
import numpy as np


class TangentNormalOverlay:
    """Live manager for the curve-tangent / surface-normal overlay."""

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

    # ── internal: builders ───────────────────────────────────────────

    def _build_tangent_actor(self) -> None:
        positions, vectors, lengths = [], [], []
        for _, tag in gmsh.model.getEntities(dim=1):
            try:
                lo, hi = gmsh.model.getParametrizationBounds(1, tag)
                u_mid = 0.5 * (lo[0] + hi[0])
                pos = np.asarray(
                    gmsh.model.getValue(1, tag, [u_mid]), dtype=np.float64,
                ).reshape(3)
                d = np.asarray(
                    gmsh.model.getDerivative(1, tag, [u_mid]),
                    dtype=np.float64,
                ).reshape(3)
                n = np.linalg.norm(d)
                if not np.isfinite(n) or n < 1e-12:
                    continue
                tangent = d / n
                length = self._entity_arrow_length(1, tag)
            except Exception:
                continue
            positions.append(pos - self.origin_shift)
            vectors.append(tangent)
            lengths.append(length)

        self._tangent_actor = self._make_glyph_actor(
            positions, vectors, lengths, color_key="info",
        )

    def _build_normal_actor(self) -> None:
        positions, vectors, lengths = [], [], []
        for _, tag in gmsh.model.getEntities(dim=2):
            try:
                com = np.asarray(
                    gmsh.model.occ.getCenterOfMass(2, tag), dtype=np.float64,
                ).reshape(3)
                n = np.asarray(
                    gmsh.model.getNormal(tag, [0.5, 0.5]), dtype=np.float64,
                ).reshape(3)
                nm = np.linalg.norm(n)
                if not np.isfinite(nm) or nm < 1e-12:
                    continue
                normal = n / nm
                length = self._entity_arrow_length(2, tag)
            except Exception:
                continue
            positions.append(com - self.origin_shift)
            vectors.append(normal)
            lengths.append(length)

        self._normal_actor = self._make_glyph_actor(
            positions, vectors, lengths, color_key="error",
        )

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

        # Unit arrow (length 1) with proportions tuned for visibility.
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
