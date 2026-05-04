"""LoadsDiagram — arrows at nodes from applied nodal force vectors (per pattern).

Reads constant force vectors from ``fem.nodes.loads.by_pattern(...)``
and draws one arrow per loaded node, oriented and scaled by the force.

Time-series scaling is intentionally *not* implemented: the broker
records reference magnitudes only and does not carry the OpenSees
timeSeries function (Linear / Path / Constant), so the load history
cannot be reconstructed at viewer time. The arrows are constant
across steps. When timeSeries info lands in the broker, ``update_to_step``
becomes the place to re-scale.

Moments are intentionally not drawn — they need a different glyph
(curved / double-shaft arrow) and will be handled in a follow-up.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import LoadsStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


class LoadsDiagram(Diagram):
    """Arrows at nodes from applied nodal forces in a load pattern."""

    kind = "loads"
    # Virtual topology — data comes from ``fem.nodes.loads`` rather
    # than a Results composite. The catalog's data combo is populated
    # with pattern names, not canonical components.
    topology = "loads"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, LoadsStyle):
            raise TypeError(
                "LoadsDiagram requires a LoadsStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)
        self._source: Optional[pv.PolyData] = None
        self._actor: Any = None
        # Substrate row indices the arrow tail points were sampled
        # from — cached so ``sync_substrate_points`` can rebuild the
        # source when the active geometry deforms.
        self._substrate_idx: Optional[ndarray] = None
        self._initial_scale: float = 1.0
        self._runtime_scale: Optional[float] = None

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
            raise RuntimeError("LoadsDiagram.attach requires a FEMSceneData.")
        super().attach(plotter, fem, scene)
        style: LoadsStyle = self.spec.style    # type: ignore[assignment]
        pattern = self.spec.selector.component

        # Collect (node_id, force) pairs for this pattern. Skip records
        # without a force vector and zero-force records — drawing a
        # zero-length arrow renders as a degenerate glyph.
        node_ids: list[int] = []
        forces: list[tuple[float, float, float]] = []
        try:
            records = list(fem.nodes.loads.by_pattern(pattern))
        except Exception:
            records = []
        for r in records:
            f = getattr(r, "force_xyz", None)
            if f is None:
                continue
            if f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0:
                continue
            node_ids.append(int(r.node_id))
            forces.append((float(f[0]), float(f[1]), float(f[2])))
        if not node_ids:
            return

        node_ids_arr = np.asarray(node_ids, dtype=np.int64)
        forces_arr = np.asarray(forces, dtype=np.float64)

        # Map FEM node ids to substrate row indices.
        max_id = int(max(node_ids_arr.max(), scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(
            scene.node_ids.size, dtype=np.int64,
        )
        substrate_idx = lookup[node_ids_arr]
        valid = substrate_idx >= 0
        if not valid.any():
            return
        substrate_idx = substrate_idx[valid]
        forces_arr = forces_arr[valid]

        coords = np.asarray(scene.grid.points)[substrate_idx].copy()
        source = pv.PolyData(coords)
        source.point_data["_vec"] = forces_arr
        source.point_data["_mag"] = np.linalg.norm(forces_arr, axis=1)
        self._source = source
        self._substrate_idx = substrate_idx.copy()

        # Auto-fit: largest arrow reaches ``auto_scale_fraction`` of
        # the model diagonal. Mirror VectorGlyph's convention.
        mag_max = float(source.point_data["_mag"].max())
        if style.scale is None:
            if mag_max > 0.0 and scene.model_diagonal > 0.0:
                self._initial_scale = (
                    style.auto_scale_fraction * scene.model_diagonal / mag_max
                )
            else:
                self._initial_scale = 1.0
        else:
            self._initial_scale = float(style.scale)

        glyph = self._build_glyph(self.current_scale())
        actor = plotter.add_mesh(
            glyph,
            name=self._actor_name(),
            color=style.color,
            reset_camera=False,
            lighting=False,
            # Decorative overlay — picks pass through to the substrate.
            pickable=False,
            show_scalar_bar=False,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        # Constant — broker has no timeSeries info to drive a per-step
        # multiplier. Step changes are silently ignored.
        return

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Move arrow tails to follow the deformed substrate."""
        if (
            self._source is None
            or self._actor is None
            or self._substrate_idx is None
        ):
            return
        try:
            target_pts = (
                np.asarray(deformed_pts, dtype=np.float64)
                if deformed_pts is not None
                else np.asarray(scene.grid.points, dtype=np.float64)
            )
            self._source.points = target_pts[self._substrate_idx]
            glyph = self._build_glyph(self.current_scale())
            mapper = self._actor.GetMapper()
            mapper.SetInputData(glyph)
            mapper.Modified()
        except Exception:
            pass

    def detach(self) -> None:
        self._source = None
        self._actor = None
        self._substrate_idx = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime adjustments
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        self._runtime_scale = float(scale)
        if self._source is not None and self._actor is not None:
            try:
                glyph = self._build_glyph(self.current_scale())
                self._actor.GetMapper().SetInputData(glyph)
                self._actor.GetMapper().Modified()
            except Exception:
                pass

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_loads_{id(self):x}"

    def _build_glyph(self, scale: float) -> pv.PolyData:
        if self._source is None:
            return pv.PolyData()
        style: LoadsStyle = self.spec.style    # type: ignore[assignment]
        try:
            return self._source.glyph(
                orient="_vec",
                scale="_mag",
                factor=float(scale),
                geom=pv.Arrow(tip_length=style.arrow_tip_fraction),
            )
        except Exception:
            return pv.PolyData(np.asarray(self._source.points))
