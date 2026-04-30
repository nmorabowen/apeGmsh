"""VectorGlyphDiagram — arrows at nodes from N components of a NodeSlab.

Same pattern as ``DeformedShapeDiagram``: ``selector.component`` is the
diagram's display label; the actual vector field is built from
``style.components`` (default: 3-D translational displacement).

Per-step update reads N components, recomputes the glyph PolyData,
and rebinds the actor's mapper input — actor identity stable across
steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import VectorGlyphStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


class VectorGlyphDiagram(Diagram):
    """Arrows at nodes, oriented and scaled by an N-component vector field."""

    kind = "vector_glyph"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, VectorGlyphStyle):
            raise TypeError(
                "VectorGlyphDiagram requires a VectorGlyphStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._source: Optional[pv.PolyData] = None
        self._actor: Any = None
        self._fem_ids_to_read: Optional[ndarray] = None
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._initial_scale: float = 1.0
        self._initial_clim: Optional[tuple[float, float]] = None

        self._runtime_scale: Optional[float] = None
        self._runtime_clim: Optional[tuple[float, float]] = None

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
                "VectorGlyphDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, fem, scene)
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve node IDs ────────────────────────────────────────
        node_ids = self._resolved_node_ids
        if node_ids is None:
            node_ids = scene.node_ids.copy()
        if node_ids.size == 0:
            return

        # Map FEM ids to substrate point indices
        max_id = int(max(node_ids.max(), scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(
            scene.node_ids.size, dtype=np.int64,
        )
        substrate_idx = lookup[node_ids]
        valid = substrate_idx >= 0
        if not valid.any():
            return
        node_ids = node_ids[valid]
        substrate_idx = substrate_idx[valid]

        coords = np.asarray(scene.grid.points)[substrate_idx].copy()

        # Lookup table FEM id -> position in our source PolyData
        n = node_ids.size
        max_id = int(node_ids.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[node_ids] = np.arange(n, dtype=np.int64)
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = node_ids

        source = pv.PolyData(coords)
        source.point_data["_vec"] = np.zeros((n, 3), dtype=np.float64)
        source.point_data["_mag"] = np.zeros(n, dtype=np.float64)
        self._source = source

        # Read step-0 vectors
        vecs0 = self._read_vectors(0)
        source.point_data["_vec"][:] = vecs0
        mags0 = np.linalg.norm(vecs0, axis=1)
        source.point_data["_mag"][:] = mags0

        # Determine scale (auto-fit if not provided)
        if style.scale is None:
            max_abs = float(mags0.max()) if mags0.size else 0.0
            if max_abs > 0.0 and scene.model_diagonal > 0.0:
                self._initial_scale = (
                    style.auto_scale_fraction * scene.model_diagonal / max_abs
                )
            else:
                self._initial_scale = 1.0
        else:
            self._initial_scale = float(style.scale)

        # Determine initial clim (over magnitudes)
        if style.clim is not None:
            self._initial_clim = (
                float(style.clim[0]), float(style.clim[1]),
            )
        else:
            finite = mags0[np.isfinite(mags0)]
            if finite.size:
                lo = float(finite.min())
                hi = float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        # Build glyph PolyData
        glyph = self._build_glyph(self.current_scale())

        kwargs: dict[str, Any] = dict(
            name=self._actor_name(),
            reset_camera=False,
            lighting=False,
        )
        if style.use_magnitude_colors:
            kwargs.update(
                scalars="_mag",
                cmap=style.cmap,
                clim=self._runtime_clim or self._initial_clim,
                show_scalar_bar=True,
                scalar_bar_args={"title": self.spec.selector.component},
            )
        else:
            kwargs.update(
                color=style.color,
                show_scalar_bar=False,
            )

        actor = plotter.add_mesh(glyph, **kwargs)
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if self._source is None or self._actor is None:
            return
        vecs = self._read_vectors(int(step_index))
        if vecs is None:
            return
        mags = np.linalg.norm(vecs, axis=1)
        self._source.point_data["_vec"][:] = vecs
        self._source.point_data["_mag"][:] = mags
        try:
            self._source.Modified()
        except Exception:
            pass
        # Rebuild the glyph and rebind the mapper input. We don't
        # ``add_mesh`` again — the actor is reused.
        glyph = self._build_glyph(self.current_scale())
        try:
            mapper = self._actor.GetMapper()
            mapper.SetInputData(glyph)
            mapper.Modified()
        except Exception:
            pass

    def detach(self) -> None:
        self._source = None
        self._actor = None
        self._fem_ids_to_read = None
        self._submesh_pos_of_id = None
        self._initial_clim = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime adjustments
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        self._runtime_scale = float(scale)
        # Re-apply current step values with new scale
        if self._source is not None:
            try:
                glyph = self._build_glyph(self.current_scale())
                self._actor.GetMapper().SetInputData(glyph)
                self._actor.GetMapper().Modified()
            except Exception:
                pass

    def set_clim(self, vmin: float, vmax: float) -> None:
        if vmin == vmax:
            vmax = vmin + 1.0
        self._runtime_clim = (float(vmin), float(vmax))
        if self._actor is not None:
            try:
                mapper = self._actor.GetMapper()
                mapper.SetScalarRange(*self._runtime_clim)
            except Exception:
                pass

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        if self._source is None:
            return None
        mags = np.asarray(self._source.point_data["_mag"])
        finite = mags[np.isfinite(mags)]
        if finite.size == 0:
            return None
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    def set_cmap(self, cmap: str) -> None:
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

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    def current_clim(self) -> Optional[tuple[float, float]]:
        return self._runtime_clim or self._initial_clim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_vector_{id(self):x}"

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    def _read_vectors(self, step_index: int) -> Optional[ndarray]:
        """Read each component for ``step_index``, return ``(N, 3)``."""
        if self._fem_ids_to_read is None or self._submesh_pos_of_id is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]

        n = self._fem_ids_to_read.size
        out = np.zeros((n, 3), dtype=np.float64)
        for axis, comp in enumerate(style.components[:3]):
            try:
                slab = results.nodes.get(
                    ids=self._fem_ids_to_read,
                    component=comp,
                    time=[step_index],
                )
            except Exception:
                continue
            if slab.values.size == 0:
                continue
            slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
            slab_vals = np.asarray(slab.values[0], dtype=np.float64)
            positions = self._submesh_pos_of_id[slab_ids]
            valid = positions >= 0
            out[positions[valid], axis] = slab_vals[valid]
        return out

    def _build_glyph(self, scale: float) -> pv.PolyData:
        """Return a freshly-built arrow PolyData from ``self._source``."""
        if self._source is None:
            return pv.PolyData()
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]
        # PyVista's glyph() respects orient + scale; ``factor`` is the
        # global multiplier we want (already includes the auto-fit).
        try:
            return self._source.glyph(
                orient="_vec",
                scale="_mag",
                factor=float(scale),
                geom=pv.Arrow(tip_length=style.arrow_tip_fraction),
            )
        except Exception:
            return pv.PolyData(np.asarray(self._source.points))
