"""DeformedShapeDiagram — warp the substrate by a displacement vector.

Reads ``displacement_x`` / ``_y`` / ``_z`` (configurable via
``DeformedShapeStyle.components``) at each step and adds
``scale * displacement`` to the deformed mesh's point coordinates.
Optional undeformed wireframe ghost stays visible behind the warped
mesh.

Performance contract:

* Selector + component list resolved once at attach.
* Per-step update reads N components × selected nodes (small for
  ``pg=`` slices, full mesh for unrestricted), reshapes to
  ``(n_points, ndim)``, mutates ``deformed_grid.points`` in place.
* The actor is re-used across steps; only the points array changes.

Note on selectors: ``selector.component`` is the *primary* component
for the diagram (used for the display label). The actual warp uses
the full ``style.components`` triple — this keeps the SlabSelector
single-component contract while letting deformed shapes warp by 2-3
components.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import DeformedShapeStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


class DeformedShapeDiagram(Diagram):
    """Warp the substrate by a displacement vector."""

    kind = "deformed_shape"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, DeformedShapeStyle):
            raise TypeError(
                "DeformedShapeDiagram requires a DeformedShapeStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        # Runtime state populated by attach()
        self._deformed_grid: Any = None
        self._deformed_actor: Any = None
        self._undeformed_actor: Any = None
        self._base_points: Optional[ndarray] = None
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._fem_ids_to_read: Optional[ndarray] = None
        self._displacement_buffer: Optional[ndarray] = None  # (n_points, ndim)

        # Mutable runtime overrides
        self._runtime_scale: Optional[float] = None
        self._runtime_show_undeformed: Optional[bool] = None

        # Cached for re-applying scale on step or scale change
        self._last_step: int = 0

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
                "DeformedShapeDiagram.attach requires a FEMSceneData. "
                "The Director must call bind_plotter(plotter, "
                "scene=scene)."
            )
        super().attach(plotter, fem, scene)

        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve selector to substrate point indices ─────────────
        node_ids = self._resolved_node_ids
        if node_ids is None:
            point_indices = np.arange(scene.grid.n_points, dtype=np.int64)
        else:
            point_indices = self._fem_ids_to_substrate_indices(scene, node_ids)
            if point_indices.size == 0:
                self._deformed_grid = None
                return

        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0:
            self._deformed_grid = None
            return

        # vtkOriginalPointIds maps submesh point index -> substrate index
        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        fem_ids_in_submesh = scene.node_ids[orig_indices]

        max_id = int(fem_ids_in_submesh.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[fem_ids_in_submesh] = np.arange(
            fem_ids_in_submesh.size, dtype=np.int64,
        )

        # ── Save base points (undeformed) for warp math ─────────────
        base_points = np.asarray(submesh.points, dtype=np.float64).copy()

        # ── Deformed grid is a separate copy — we mutate its points ─
        deformed = submesh.copy()

        ndim = len(style.components)
        disp_buffer = np.zeros((base_points.shape[0], ndim), dtype=np.float64)

        self._deformed_grid = deformed
        self._base_points = base_points
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh
        self._displacement_buffer = disp_buffer

        # ── Initial warp at step 0 ──────────────────────────────────
        self._apply_step(0)

        # ── Add deformed mesh actor ─────────────────────────────────
        deformed_actor = plotter.add_mesh(
            deformed,
            color=style.color,
            opacity=1.0,
            show_edges=False,
            lighting=True,
            smooth_shading=False,
            name=self._actor_name("deformed"),
            reset_camera=False,
        )
        self._deformed_actor = deformed_actor

        # ── Undeformed reference (wireframe ghost) ──────────────────
        if style.show_undeformed:
            undef_actor = plotter.add_mesh(
                submesh,
                color=style.undeformed_color,
                style="wireframe",
                line_width=1,
                opacity=style.undeformed_opacity,
                show_edges=False,
                lighting=False,
                name=self._actor_name("undeformed"),
                reset_camera=False,
            )
            self._undeformed_actor = undef_actor
        else:
            self._undeformed_actor = None

        actors = [deformed_actor]
        if self._undeformed_actor is not None:
            actors.append(self._undeformed_actor)
        self._actors = actors

    def update_to_step(self, step_index: int) -> None:
        if self._deformed_grid is None:
            return
        self._apply_step(int(step_index))

    def detach(self) -> None:
        self._deformed_grid = None
        self._deformed_actor = None
        self._undeformed_actor = None
        self._base_points = None
        self._submesh_pos_of_id = None
        self._fem_ids_to_read = None
        self._displacement_buffer = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style adjustments
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        """Update the warp scale. Live update without re-attach."""
        self._runtime_scale = float(scale)
        if self._deformed_grid is not None:
            self._apply_step(self._last_step)

    def set_show_undeformed(self, show: bool) -> None:
        """Toggle the undeformed reference visibility."""
        self._runtime_show_undeformed = bool(show)
        if self._undeformed_actor is not None:
            try:
                self._undeformed_actor.SetVisibility(bool(show))
            except Exception:
                pass

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]
        return style.scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _actor_name(self, suffix: str) -> str:
        return f"diagram_deformed_{id(self):x}_{suffix}"

    def _apply_step(self, step_index: int) -> None:
        """Read all components for ``step_index`` and update points in place."""
        if (
            self._deformed_grid is None
            or self._base_points is None
            or self._displacement_buffer is None
            or self._fem_ids_to_read is None
            or self._submesh_pos_of_id is None
        ):
            return

        style: DeformedShapeStyle = self.spec.style    # type: ignore[assignment]
        results = self._scoped_results()
        if results is None:
            return

        # Reset displacement buffer to zero before scattering each
        # component — missing components contribute nothing.
        self._displacement_buffer.fill(0.0)

        for axis, component in enumerate(style.components):
            try:
                slab = results.nodes.get(
                    ids=self._fem_ids_to_read,
                    component=component,
                    time=[int(step_index)],
                )
            except Exception:
                # Component may not be available — skip silently.
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

        scale = self.current_scale()
        # Pad to 3D if components < 3 (e.g., 2D model with x/y only).
        if self._displacement_buffer.shape[1] < 3:
            disp_3d = np.zeros(
                (self._base_points.shape[0], 3), dtype=np.float64,
            )
            disp_3d[:, : self._displacement_buffer.shape[1]] = (
                self._displacement_buffer
            )
        else:
            disp_3d = self._displacement_buffer[:, :3]

        new_points = self._base_points + scale * disp_3d
        # In-place mutation of the VTK points array — VTK observes
        # the change via Modified.
        self._deformed_grid.points = new_points
        try:
            self._deformed_grid.Modified()
        except Exception:
            pass
        self._last_step = step_index

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
