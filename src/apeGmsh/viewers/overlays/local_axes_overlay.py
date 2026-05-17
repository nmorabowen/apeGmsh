"""Local-axes overlay — beam-element orientation triads.

Draws the OpenSees ``geomTransf`` local frame of every 1-D (line)
element as three colour-coded arrow glyphs at the element midpoint:

* **local-x** (red)   — element axis, node i → node j
* **local-y** (green) — strong/weak bending axis
* **local-z** (blue)  — the ``vecxz`` plane direction

This is the implementation of the "Local-axis glyph overlay" feature
specified in ``opensees/architecture/viewer-integration.md`` (the
x=red / y=green / z=blue convention is taken from there).  The real
per-element ``vecxz`` arrives via :class:`ViewerData` (the
``transforms`` ↔ ``element_meta`` join lives in
``opensees.emitter.h5_reader``); when a model carries no OpenSees
enrichment the frame falls back to the structural default and the
overlay still renders a sensible triad.

Pure PyVista — no Qt, no gmsh, no ``apeGmsh.mesh`` / ``apeGmsh.opensees``
imports (the Phase 8.7 pure-h5-consumer invariant; see
``tests/test_viewers_pure_h5_consumer.py``).  Lifecycle mirrors
:class:`apeGmsh.viewers.overlays.probe_overlay.ProbeOverlay`:
constructed after ``bind_plotter`` with ``(plotter, scene, director)``;
actors are name-managed so a rebuild is idempotent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv

from ..diagrams._beam_geometry import iter_local_frames

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector
    from ..scene.fem_scene import FEMSceneData


class LocalAxesOverlay:
    """Owns the three local-axis glyph actors; one instance per viewer."""

    #: apeGmsh axis convention (viewer-integration.md).
    AXIS_COLORS = {"x": "#FF4136", "y": "#2ECC40", "z": "#0074D9"}
    #: Glyph length as a fraction of the model diagonal, capped per
    #: element at ``GLYPH_ELEMENT_FRACTION`` of the element length so a
    #: triad never visually swamps a short member.
    GLYPH_DIAGONAL_FRACTION = 0.04
    GLYPH_ELEMENT_FRACTION = 0.45

    def __init__(
        self,
        plotter: Any,
        scene: "FEMSceneData",
        director: "ResultsDirector",
    ) -> None:
        self._plotter = plotter
        self._scene = scene
        self._director = director
        self._visible = False
        # FEM node id -> substrate row, so endpoints read off the live
        # (possibly deformed) ``scene.grid.points``.
        self._node_row = {
            int(nid): i for i, nid in enumerate(scene.node_ids)
        }
        self._actor_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, show: bool) -> None:
        show = bool(show)
        if show == self._visible and not show:
            return
        self._visible = show
        self._remove_actors()
        if show:
            self._build()
        self._render()

    def refresh(self) -> None:
        """Rebuild from the current substrate (call after a deform /
        step / theme change)."""
        if not self._visible:
            return
        self._remove_actors()
        self._build()
        self._render()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _node_coord(self, nid: int) -> "np.ndarray | None":
        row = self._node_row.get(int(nid))
        if row is None:
            return None
        try:
            pts = self._scene.grid.points
            if row >= len(pts):
                return None
            return np.asarray(pts[row], dtype=np.float64)
        except Exception:
            return None

    def _build(self) -> None:
        view = getattr(self._director, "view", None)
        if view is None:
            return

        origins: list[np.ndarray] = []
        vecs = {"x": [], "y": [], "z": []}  # type: dict[str, list]
        scales: list[float] = []

        diag = float(getattr(self._scene, "model_diagonal", 0.0)) or 1.0
        global_len = diag * self.GLYPH_DIAGONAL_FRACTION

        for frame in iter_local_frames(view, self._node_coord):
            glyph_len = global_len
            if frame.length > 0:
                glyph_len = min(
                    global_len, frame.length * self.GLYPH_ELEMENT_FRACTION,
                )
            origins.append(frame.origin)
            vecs["x"].append(frame.x)
            vecs["y"].append(frame.y)
            vecs["z"].append(frame.z)
            scales.append(glyph_len)

        if not origins:
            return

        pts = np.asarray(origins, dtype=np.float64)
        scl = np.asarray(scales, dtype=np.float64)
        arrow = pv.Arrow(tip_length=0.25, tip_radius=0.10, shaft_radius=0.035)

        for axis in ("x", "y", "z"):
            cloud = pv.PolyData(pts.copy())
            cloud["vectors"] = np.asarray(vecs[axis], dtype=np.float64)
            cloud["scale"] = scl
            glyphs = cloud.glyph(
                orient="vectors", scale="scale", factor=1.0, geom=arrow,
            )
            name = f"_localaxes_{axis}"
            try:
                self._plotter.add_mesh(
                    glyphs,
                    color=self.AXIS_COLORS[axis],
                    lighting=False,
                    pickable=False,
                    reset_camera=False,
                    name=name,
                )
                self._actor_names.append(name)
            except Exception:
                continue

    def _remove_actors(self) -> None:
        for name in self._actor_names:
            try:
                self._plotter.remove_actor(name)
            except Exception:
                pass
        self._actor_names.clear()

    def _render(self) -> None:
        try:
            self._plotter.render()
        except Exception:
            pass
