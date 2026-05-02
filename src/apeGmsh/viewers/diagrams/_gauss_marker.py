"""GaussPointDiagram — sphere markers at GP world positions.

Reads :class:`GaussSlab`, calls ``slab.global_coords(fem)`` to map
natural coords to world coords (proper shape fns for hex8 / quad4;
centroid + bbox approximation otherwise), and renders one **real**
sphere glyph per GP — sized in world units off the model diagonal.

We deliberately don't use ``render_points_as_spheres=True`` here: on
co-planar (z=0) 2-D models that flag tends to lose its billboards to
z-fighting with the substrate fill, and the only mitigation
(``SetResolveCoincidentTopology…``) is global VTK state and ends up
also disturbing the wireframe overlay. Real sphere geometry sits a
finite radius above / below the plane and renders unambiguously.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import GaussMarkerStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_SCALAR_NAME = "_gp_value"


class GaussPointDiagram(Diagram):
    """Sphere markers at Gauss-point world positions, colored by value."""

    kind = "gauss_marker"
    topology = "gauss"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, GaussMarkerStyle):
            raise TypeError(
                "GaussPointDiagram requires a GaussMarkerStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._cloud: Optional[pv.PolyData] = None         # input centers
        self._glyphs: Optional[pv.PolyData] = None        # actor's input
        self._actor: Any = None
        self._scalar_array: Optional[ndarray] = None      # per-center scalar
        self._glyph_scalar_array: Optional[ndarray] = None  # per-glyph-point
        self._pts_per_center: int = 0
        self._element_ids_to_read: tuple[int, ...] = ()
        self._initial_clim: Optional[tuple[float, float]] = None
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_cmap: Optional[str] = None
        # Cached at attach so deformation sync can re-evaluate shape
        # functions against deformed substrate coords without
        # re-reading the slab from disk.
        self._gp_element_index: Optional[ndarray] = None
        self._gp_natural_coords: Optional[ndarray] = None
        # Reference world coords + base glyph points captured at attach;
        # used by sync_substrate_points to translate each sphere when
        # the substrate deforms.
        self._gp_centers_at_build: Optional[ndarray] = None
        self._base_glyph_pts: Optional[ndarray] = None

    # ------------------------------------------------------------------
    # Attach / detach / update
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        fem: "FEMData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "GaussPointDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, fem, scene)
        style: GaussMarkerStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (continuum: 2-D + 3-D) ──────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_continuum_element_ids(fem)
        if element_ids.size == 0:
            return
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Step 0 read ─────────────────────────────────────────────
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.gauss.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[0],
            )
        except Exception as exc:
            raise RuntimeError(
                f"GaussPointDiagram could not read gauss slab: {exc}"
            )
        if slab.values.size == 0 or slab.element_index.size == 0:
            return

        # ── World positions via the slab's shape-fn helper ──────────
        # Cache the per-GP arrays so deformation sync can recompute
        # world coords later against deformed substrate points without
        # re-reading the slab from disk.
        self._gp_element_index = np.asarray(
            slab.element_index, dtype=np.int64,
        ).copy()
        self._gp_natural_coords = np.asarray(
            slab.natural_coords, dtype=np.float64,
        ).copy()

        try:
            world = slab.global_coords(fem)
        except Exception:
            # Defensive: fall back to a zero array — we'd rather show
            # something than crash the viewer.
            world = np.zeros((slab.element_index.size, 3), dtype=np.float64)

        center_scalars = np.asarray(slab.values[0], dtype=np.float64).copy()
        cloud = pv.PolyData(world)
        cloud.point_data[_SCALAR_NAME] = center_scalars
        self._cloud = cloud
        self._scalar_array = cloud.point_data[_SCALAR_NAME]
        self._gp_centers_at_build = np.asarray(world, dtype=np.float64).copy()

        # Initial clim
        if style.clim is not None:
            self._initial_clim = (
                float(style.clim[0]), float(style.clim[1]),
            )
        else:
            data = np.asarray(self._scalar_array)
            finite = data[np.isfinite(data)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        # ── Real sphere glyphs at each GP center ────────────────────
        # ``style.point_size`` keeps its prior semantic (bigger = more
        # visible) but is now used to scale a world-space sphere
        # radius off the model diagonal — same convention as the
        # pre-solve mesh viewer's node cloud. This dodges the brittle
        # ``render_points_as_spheres=True`` pixel-billboard path that
        # was losing markers to z-fighting on planar models.
        diag = float(getattr(scene, "model_diagonal", 0.0)) or 1.0
        radius = 0.003 * diag * max(0.1, float(style.point_size) / 10.0)
        sphere = pv.Sphere(
            radius=radius, theta_resolution=10, phi_resolution=10,
        )
        n_per_center = sphere.n_points
        glyphs = cloud.glyph(geom=sphere, scale=False, orient=False)
        # pyvista propagates input point_data through ``glyph()``,
        # but the resulting attribute is per-glyph-point. Re-stamp our
        # canonical name in case the propagation skipped it (e.g. when
        # an active scalar collision happens), then keep a handle to
        # the per-glyph array for in-place per-step updates.
        glyph_scalars = np.repeat(center_scalars, n_per_center)
        glyphs.point_data[_SCALAR_NAME] = glyph_scalars
        self._glyphs = glyphs
        self._glyph_scalar_array = glyphs.point_data[_SCALAR_NAME]
        self._pts_per_center = n_per_center
        # Snapshot the unwarped glyph point coords so sync can translate
        # each sphere by ``deformed[i] - centers_at_build[i]`` without
        # rebuilding the glyph PolyData.
        self._base_glyph_pts = np.asarray(
            glyphs.points, dtype=np.float64,
        ).copy()

        actor = plotter.add_mesh(
            glyphs,
            scalars=_SCALAR_NAME,
            cmap=self._runtime_cmap or style.cmap,
            clim=self._runtime_clim or self._initial_clim,
            opacity=style.opacity,
            show_scalar_bar=style.show_scalar_bar,
            scalar_bar_args={
                "title": self.spec.selector.component,
            } if style.show_scalar_bar else None,
            name=self._actor_name(),
            reset_camera=False,
            smooth_shading=True,
            lighting=True,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if (
            self._scalar_array is None
            or self._glyph_scalar_array is None
            or self._glyphs is None
            or self._pts_per_center == 0
        ):
            return
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.gauss.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return
        if slab.values.size == 0:
            return
        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if slab_values.size != self._scalar_array.size:
            return
        self._scalar_array[:] = slab_values
        # Tile the per-center scalar to per-glyph-point so every
        # vertex on each sphere paints with the same color.
        self._glyph_scalar_array[:] = np.repeat(
            slab_values, self._pts_per_center,
        )
        try:
            self._glyphs.Modified()
        except Exception:
            pass

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Translate every GP sphere to follow the deformed substrate.

        Re-evaluates the per-GP world coords against ``deformed_pts``
        (or ``fem.nodes.coords`` when ``None``), then shifts each
        sphere's glyph points by ``new_center[i] - center_at_build[i]``.
        """
        if (
            self._glyphs is None
            or self._cloud is None
            or self._fem is None
            or self._gp_element_index is None
            or self._gp_natural_coords is None
            or self._gp_centers_at_build is None
            or self._base_glyph_pts is None
            or self._pts_per_center == 0
        ):
            return
        try:
            from apeGmsh.results._gauss_world_coords import (
                compute_global_coords_from_arrays,
            )
            new_centers = compute_global_coords_from_arrays(
                self._gp_element_index,
                self._gp_natural_coords,
                self._fem,
                node_coords_override=deformed_pts,
            )
            shifts = new_centers - self._gp_centers_at_build
            shifts_tiled = np.repeat(shifts, self._pts_per_center, axis=0)
            self._glyphs.points = self._base_glyph_pts + shifts_tiled
            # Keep the input cloud's centers in sync so any code that
            # reads ``self._cloud.points`` still sees the current state.
            self._cloud.points = new_centers
        except Exception:
            pass

    def detach(self) -> None:
        self._cloud = None
        self._glyphs = None
        self._actor = None
        self._scalar_array = None
        self._glyph_scalar_array = None
        self._pts_per_center = 0
        self._element_ids_to_read = ()
        self._initial_clim = None
        self._gp_element_index = None
        self._gp_natural_coords = None
        self._gp_centers_at_build = None
        self._base_glyph_pts = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style
    # ------------------------------------------------------------------

    def set_clim(self, vmin: float, vmax: float) -> None:
        if vmin == vmax:
            vmax = vmin + 1.0
        self._runtime_clim = (float(vmin), float(vmax))
        if self._actor is not None:
            try:
                self._actor.GetMapper().SetScalarRange(*self._runtime_clim)
            except Exception:
                pass

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        if self._scalar_array is None:
            return None
        data = np.asarray(self._scalar_array)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    def set_cmap(self, cmap: str) -> None:
        self._runtime_cmap = cmap
        if self._actor is None:
            return
        try:
            lut = pv.LookupTable(cmap)
            clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
            lut.scalar_range = clim
            mapper = self._actor.GetMapper()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(*clim)
        except Exception:
            pass

    def current_clim(self) -> Optional[tuple[float, float]]:
        return self._runtime_clim or self._initial_clim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_gauss_{id(self):x}"

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _collect_continuum_element_ids(fem: "FEMData") -> ndarray:
        """All 2-D and 3-D element IDs (continuum types)."""
        ids: list[int] = []
        for group in fem.elements:
            if group.element_type.dim in (2, 3):
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)
