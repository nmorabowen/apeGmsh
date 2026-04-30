"""GaussPointDiagram — sphere markers at GP world positions.

Reads :class:`GaussSlab`, calls ``slab.global_coords(fem)`` to map
natural coords to world coords (proper shape fns for hex8 / quad4;
centroid + bbox approximation otherwise), and adds a colored point
cloud per GP.
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

        self._cloud: Optional[pv.PolyData] = None
        self._actor: Any = None
        self._scalar_array: Optional[ndarray] = None
        self._element_ids_to_read: tuple[int, ...] = ()
        self._initial_clim: Optional[tuple[float, float]] = None
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_cmap: Optional[str] = None

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
        try:
            world = slab.global_coords(fem)
        except Exception:
            # Defensive: fall back to a zero array — we'd rather show
            # something than crash the viewer.
            world = np.zeros((slab.element_index.size, 3), dtype=np.float64)

        cloud = pv.PolyData(world)
        cloud.point_data[_SCALAR_NAME] = (
            np.asarray(slab.values[0], dtype=np.float64).copy()
        )
        self._cloud = cloud
        self._scalar_array = cloud.point_data[_SCALAR_NAME]

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

        actor = plotter.add_mesh(
            cloud,
            scalars=_SCALAR_NAME,
            cmap=self._runtime_cmap or style.cmap,
            clim=self._runtime_clim or self._initial_clim,
            opacity=style.opacity,
            render_points_as_spheres=True,
            point_size=style.point_size,
            show_scalar_bar=style.show_scalar_bar,
            scalar_bar_args={
                "title": self.spec.selector.component,
            } if style.show_scalar_bar else None,
            name=self._actor_name(),
            reset_camera=False,
            lighting=False,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if self._cloud is None or self._scalar_array is None:
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
        try:
            self._cloud.Modified()
        except Exception:
            pass

    def detach(self) -> None:
        self._cloud = None
        self._actor = None
        self._scalar_array = None
        self._element_ids_to_read = ()
        self._initial_clim = None
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
