"""ContourDiagram — paint scalar values from a NodeSlab on the substrate.

Phase 1 surface — nodal kinematics and scalars only. Continuum
gauss-interpolated contour arrives in Phase 4 as an extension to this
diagram.

Performance contract (locked in Phase 0, validated here):

* Selector resolves to FEM node IDs **once at attach**. The submesh
  is extracted from the substrate grid once; per-step reads only
  refresh ``point_data["_contour"]`` in place.
* Per-step update is one h5py read for the selected IDs at the active
  step, one numpy scatter into the submesh point-data array, one
  ``Modified()`` mark. No actor re-creation.
* The mapper id seen by ``actor.GetMapper()`` is stable across step
  changes — the in-place mutation test asserts this.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import ContourStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_SCALAR_NAME = "_contour"


class ContourDiagram(Diagram):
    """Per-node scalar contour painted on a slice of the substrate mesh.

    The selector picks which nodes / elements carry data; everything
    else stays as the gray substrate. Multiple contour diagrams compose
    naturally — each owns its own submesh actor.
    """

    kind = "contour"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, ContourStyle):
            raise TypeError(
                "ContourDiagram requires a ContourStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        # Runtime state populated by attach()
        self._submesh: Any = None              # pv.UnstructuredGrid slice
        self._actor: Any = None
        self._scalar_array: Optional[ndarray] = None
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._fem_ids_to_read: Optional[ndarray] = None
        self._initial_clim: Optional[tuple[float, float]] = None

        # Mutable runtime overrides (style is frozen)
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_opacity: Optional[float] = None
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
                "ContourDiagram.attach requires a FEMSceneData (the "
                "viewer's substrate mesh). The Director must call "
                "bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, fem, scene)

        # ── Resolve selector to substrate point indices ─────────────
        node_ids = self._resolved_node_ids
        if node_ids is None:
            # Unrestricted -> full mesh
            point_indices = np.arange(scene.grid.n_points, dtype=np.int64)
        else:
            point_indices = self._fem_ids_to_substrate_indices(scene, node_ids)
            if point_indices.size == 0:
                from ._base import NoDataError
                raise NoDataError(
                    f"ContourDiagram: selector resolved to {node_ids.size} "
                    f"node(s) but none of them are in the substrate mesh "
                    f"(selector={self.spec.selector!r})."
                )

        # ── Extract a submesh containing those points + their cells ─
        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0:
            from ._base import NoDataError
            raise NoDataError(
                "ContourDiagram: substrate submesh is empty for this "
                "selector — nothing to color."
            )

        # vtkOriginalPointIds maps submesh point index -> substrate index
        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        # FEM node IDs in submesh-point order
        fem_ids_in_submesh = scene.node_ids[orig_indices]

        # Build a numpy lookup table from FEM node ID -> submesh position
        max_id = int(fem_ids_in_submesh.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[fem_ids_in_submesh] = np.arange(
            fem_ids_in_submesh.size, dtype=np.int64,
        )

        self._submesh = submesh
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh

        # ── Initial scalar array filled from step 0 ─────────────────
        scalars = np.zeros(submesh.n_points, dtype=np.float64)
        submesh.point_data[_SCALAR_NAME] = scalars
        self._scalar_array = submesh.point_data[_SCALAR_NAME]

        values_at_step_0 = self._fetch_step_values(0)
        if values_at_step_0 is None:
            from ._base import NoDataError
            raise NoDataError(
                f"ContourDiagram: no nodal data for component "
                f"{self.spec.selector.component!r} at step 0. Use "
                f"`results.inspect.diagnose("
                f"{self.spec.selector.component!r})` to see which "
                f"buckets were checked."
            )
        self._scatter_into_scalar(values_at_step_0[0], values_at_step_0[1])

        # ── Resolve initial clim (style or auto-fit at step 0) ──────
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        if style.clim is not None:
            self._initial_clim = (float(style.clim[0]), float(style.clim[1]))
        else:
            data = np.asarray(self._scalar_array)
            finite = data[np.isfinite(data)]
            if finite.size:
                lo = float(finite.min())
                hi = float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        # ── Add the actor with scalar coloring ──────────────────────
        cmap = self._runtime_cmap or style.cmap
        opacity = (
            self._runtime_opacity
            if self._runtime_opacity is not None else style.opacity
        )
        clim = self._runtime_clim or self._initial_clim

        actor = plotter.add_mesh(
            submesh,
            scalars=_SCALAR_NAME,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            show_edges=style.show_edges,
            show_scalar_bar=style.show_scalar_bar,
            scalar_bar_args={
                "title": self.spec.selector.component,
            } if style.show_scalar_bar else None,
            name=self._actor_name(),
            reset_camera=False,
            lighting=True,
            smooth_shading=False,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if self._submesh is None or self._scalar_array is None:
            return
        fetched = self._fetch_step_values(step_index)
        if fetched is None:
            return
        slab_node_ids, slab_values = fetched
        self._scatter_into_scalar(slab_node_ids, slab_values)

    def detach(self) -> None:
        self._submesh = None
        self._actor = None
        self._scalar_array = None
        self._submesh_pos_of_id = None
        self._fem_ids_to_read = None
        self._initial_clim = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style adjustments (used by the settings tab)
    # ------------------------------------------------------------------

    def set_clim(self, vmin: float, vmax: float) -> None:
        """Override the colormap range. Live update."""
        if vmin == vmax:
            vmax = vmin + 1.0
        self._runtime_clim = (float(vmin), float(vmax))
        self._apply_clim()

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        """Re-fit clim to the current step's value range."""
        if self._scalar_array is None:
            return None
        data = np.asarray(self._scalar_array)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        lo = float(finite.min())
        hi = float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    def set_opacity(self, opacity: float) -> None:
        self._runtime_opacity = float(opacity)
        if self._actor is not None:
            try:
                self._actor.GetProperty().SetOpacity(float(opacity))
            except Exception:
                pass

    def set_cmap(self, cmap: str) -> None:
        """Switch the colormap. Mutates the lookup table on the active actor."""
        self._runtime_cmap = cmap
        if self._actor is None:
            return
        # PyVista's add_mesh creates a vtkScalarsToColors — the cleanest
        # way to swap cmap without re-adding the actor is to rebuild
        # the lookup table via PyVista's helper.
        try:
            import pyvista as pv
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_contour_{id(self):x}"

    def _fetch_step_values(
        self, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray]]:
        """Read one step's slab. Returns ``(node_ids, values)`` or None."""
        if self._fem_ids_to_read is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        ids = self._fem_ids_to_read
        slab = results.nodes.get(
            ids=ids,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return None
        # values shape (1, M)
        return (np.asarray(slab.node_ids, dtype=np.int64),
                np.asarray(slab.values[0], dtype=np.float64))

    def _scatter_into_scalar(
        self, slab_node_ids: ndarray, slab_values: ndarray,
    ) -> None:
        if self._submesh_pos_of_id is None or self._scalar_array is None:
            return
        positions = self._submesh_pos_of_id[slab_node_ids]
        valid = positions >= 0
        # In-place assignment — no array re-allocation.
        self._scalar_array[positions[valid]] = slab_values[valid]
        try:
            self._submesh.Modified()
        except Exception:
            pass

    def _apply_clim(self) -> None:
        if self._actor is None:
            return
        clim = self._runtime_clim or self._initial_clim
        if clim is None:
            return
        try:
            mapper = self._actor.GetMapper()
            mapper.SetScalarRange(*clim)
        except Exception:
            pass

    def _scoped_results(self) -> "Optional[Results]":
        """Return a Results scoped to the diagram's stage (or the spec's)."""
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        # No stage on spec — let Results auto-resolve (works for
        # single-stage files; raises for multi-stage). The Director
        # is responsible for setting the spec.stage_id when adding.
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _fem_ids_to_substrate_indices(
        scene: "FEMSceneData", fem_ids: ndarray,
    ) -> ndarray:
        """Map a FEM-id array to substrate point indices, dropping misses."""
        max_id = max(int(fem_ids.max()), int(scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(scene.node_ids.size, dtype=np.int64)
        idx = lookup[fem_ids]
        return idx[idx >= 0]
