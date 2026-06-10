"""DeformedShapeDiagram — warp the substrate by a displacement vector.

Reads ``displacement_x`` / ``_y`` / ``_z`` (configurable via
``DeformedShapeStyle.components``) at each step and adds
``scale * displacement`` to the deformed mesh's point coordinates.
Optional undeformed wireframe ghost stays visible behind the warped
mesh.

Render seam (ADR 0042, R-B): emits two :class:`MeshLayer`s through
``self._backend`` — a solid *deformed* layer (its points re-warped per
step / scale change via ``update_layer``) and an optional *undeformed*
wireframe ghost. The diagram holds no VTK actors. The submesh is still
extracted via pyvista (transitional) and re-expressed as neutral IR
through ``cellblocks_from_grid``; that extraction moves behind a
pyvista-free scene accessor at R-B.final.

Performance contract:

* Selector + component list + cell topology resolved once at attach.
* Per-step update reshapes displacement to ``(n_points, ndim)``,
  recomputes ``base + scale * disp``, and updates the deformed layer in
  place (the backend reuses the actor when topology is unchanged).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import DeformedShapeStyle
from ..scene_ir import ColorSpec, MeshLayer, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData
    from ..scene_ir import CellBlocks


class DeformedShapeDiagram(Diagram):
    """Warp the substrate by a displacement vector."""

    kind = "deformed_shape"
    topology = "nodes"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, DeformedShapeStyle):
            raise TypeError(
                "DeformedShapeDiagram requires a DeformedShapeStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        # Topology + warp inputs, resolved once at attach.
        self._cells: "Optional[CellBlocks]" = None
        self._base_points: Optional[ndarray] = None
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._fem_ids_to_read: Optional[ndarray] = None
        self._displacement_buffer: Optional[ndarray] = None  # (n_points, ndim)

        # Backend handles + last-emitted layers (no VTK actor held).
        self._deformed_handle: Any = None
        self._undeformed_handle: Any = None
        self._deformed_layer: Optional[MeshLayer] = None

        # Mutable runtime overrides
        self._runtime_scale: Optional[float] = None
        # Runtime override of style.show_undeformed (None = no override).
        # Read back by the settings tab to restore its checkbox state.
        self._runtime_show_undeformed: Optional[bool] = None
        self._last_step: int = 0

    # ------------------------------------------------------------------
    # Attach / detach / update
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "DeformedShapeDiagram.attach requires a FEMSceneData. "
                "The Director must call bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, view, scene)

        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve selector to substrate point indices ─────────────
        node_ids = self._resolved_node_ids
        if node_ids is None:
            point_indices = np.arange(scene.grid.n_points, dtype=np.int64)
        else:
            point_indices = self._fem_ids_to_substrate_indices(scene, node_ids)
            if point_indices.size == 0:
                self._cells = None
                return

        # Submesh extraction is still pyvista (transitional) — re-expressed
        # as neutral IR below via cellblocks_from_grid.
        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0:
            self._cells = None
            return

        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        fem_ids_in_submesh = scene.node_ids[orig_indices]

        max_id = int(fem_ids_in_submesh.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[fem_ids_in_submesh] = np.arange(
            fem_ids_in_submesh.size, dtype=np.int64,
        )

        base_points = np.asarray(submesh.points, dtype=np.float64).copy()
        ndim = len(style.components)

        from ..backends.pyvista_qt import cellblocks_from_grid

        self._cells = cellblocks_from_grid(submesh)
        self._base_points = base_points
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh
        self._displacement_buffer = np.zeros(
            (base_points.shape[0], ndim), dtype=np.float64,
        )

        # ── Deformed layer (warped at step 0) ───────────────────────
        warped = self._warped_points(0)
        self._deformed_layer = self._mesh_layer(
            warped, "deformed", style.color, opacity=1.0, wireframe=False,
        )
        self._deformed_handle = self._backend.add_layer(self._deformed_layer)

        # ── Undeformed reference (wireframe ghost) ──────────────────
        # Honour a runtime toggle across detach/re-attach (stage change)
        # so a ghost the user turned off doesn't resurrect.
        if self._effective_show_undeformed():
            undef = self._mesh_layer(
                base_points, "undeformed", style.undeformed_color,
                opacity=style.undeformed_opacity, wireframe=True,
            )
            self._undeformed_handle = self._backend.add_layer(undef)
        else:
            self._undeformed_handle = None

    def update_to_step(self, step_index: int) -> None:
        if self._cells is None or self._deformed_handle is None:
            return
        warped = self._warped_points(int(step_index))
        self._deformed_layer = self._mesh_layer(
            warped, "deformed",
            self.spec.style.color,            # type: ignore[attr-defined]
            opacity=1.0, wireframe=False,
        )
        self._backend.update_layer(self._deformed_handle, self._deformed_layer)

    def detach(self) -> None:
        for handle in (self._deformed_handle, self._undeformed_handle):
            if self._backend is not None and handle is not None:
                self._backend.remove_layer(handle)
        self._deformed_handle = None
        self._undeformed_handle = None
        self._deformed_layer = None
        self._cells = None
        self._base_points = None
        self._submesh_pos_of_id = None
        self._fem_ids_to_read = None
        self._displacement_buffer = None
        super().detach()

    # ------------------------------------------------------------------
    # Visibility (backend-routed)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is None:
            return
        if self._deformed_handle is not None:
            self._backend.set_layer_visible(
                self._deformed_handle, bool(visible),
            )
        if self._undeformed_handle is not None:
            # The ghost only shows when the diagram is visible AND the
            # user hasn't toggled it off — a plain show must not
            # resurrect a disabled undeformed reference.
            self._backend.set_layer_visible(
                self._undeformed_handle,
                bool(visible) and self._effective_show_undeformed(),
            )

    # ------------------------------------------------------------------
    # Runtime style adjustments
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        """Update the warp scale. Live update without re-attach."""
        self._runtime_scale = float(scale)
        if self._cells is not None and self._deformed_handle is not None:
            self.update_to_step(self._last_step)

    def set_show_undeformed(self, show: bool) -> None:
        """Toggle the undeformed reference visibility."""
        self._runtime_show_undeformed = bool(show)
        if self._backend is not None and self._undeformed_handle is not None:
            self._backend.set_layer_visible(
                self._undeformed_handle,
                bool(show) and bool(self._visible),
            )

    def _effective_show_undeformed(self) -> bool:
        """Runtime toggle when set, else the attach-time style flag."""
        if self._runtime_show_undeformed is not None:
            return self._runtime_show_undeformed
        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]
        return bool(style.show_undeformed)

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]
        return style.scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_id(self, suffix: str) -> str:
        return f"deformed_{id(self):x}_{suffix}"

    def _mesh_layer(
        self,
        points: ndarray,
        suffix: str,
        color: Any,
        *,
        opacity: float,
        wireframe: bool,
    ) -> MeshLayer:
        assert self._cells is not None
        return MeshLayer(
            layer_id=self._layer_id(suffix),
            points=PointSet(points),
            cells=self._cells,
            color=ColorSpec(mode="solid", solid_rgb=color),
            opacity=opacity,
            wireframe=wireframe,
        )

    def _warped_points(self, step_index: int) -> ndarray:
        """Return ``base + scale * displacement`` for ``step_index``.

        Reads each configured component for the step, scatters it into
        the displacement buffer, pads to 3-D, and applies the scale.
        Falls back to the base points if results can't be read.
        """
        assert (
            self._base_points is not None
            and self._displacement_buffer is not None
            and self._fem_ids_to_read is not None
            and self._submesh_pos_of_id is not None
        )
        self._last_step = step_index
        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]
        results = self._scoped_results()
        if results is None:
            return self._base_points

        self._displacement_buffer.fill(0.0)
        for axis, component in enumerate(style.components):
            try:
                slab = results.nodes.get(
                    ids=self._fem_ids_to_read,
                    component=component,
                    time=[int(step_index)],
                )
            except Exception:
                continue
            if slab.values.size == 0:
                continue
            slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
            slab_values = np.asarray(slab.values[0], dtype=np.float64)
            positions = self._submesh_pos_of_id[slab_ids]
            valid = positions >= 0
            self._displacement_buffer[positions[valid], axis] = (
                slab_values[valid]
            )

        if self._displacement_buffer.shape[1] < 3:
            disp_3d = np.zeros((self._base_points.shape[0], 3), dtype=np.float64)
            disp_3d[:, : self._displacement_buffer.shape[1]] = (
                self._displacement_buffer
            )
        else:
            disp_3d = self._displacement_buffer[:, :3]

        return self._base_points + self.current_scale() * disp_3d

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    @staticmethod
    def _fem_ids_to_substrate_indices(
        scene: "FEMSceneData", fem_ids: ndarray,
    ) -> ndarray:
        max_id = max(int(fem_ids.max()), int(scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(scene.node_ids.size, dtype=np.int64)
        idx = lookup[fem_ids]
        return idx[idx >= 0]
