"""ReactionsDiagram — arrows at constrained nodes from recorded reactions.

Renders BOTH ``reaction_force_*`` and ``reaction_moment_*`` from the
nodes composite of the stage-scoped Results. Forces are drawn with
straight arrows (``pv.Arrow``); moments with the curved-arrow glyph
from :mod:`..overlays.moment_glyph`. Each family auto-fits its own
scale because forces and torques have different units — coupling
them would distort one of the two.

Unlike :class:`LoadsDiagram`, reactions ARE step-resolved (the
recorder writes them per step), so ``update_to_step`` re-reads the
slabs and rebuilds the glyphs — same pattern as
:class:`VectorGlyphDiagram`.

Auto-filter: at attach, the global ``(T, N)`` magnitude slab is
reduced to one max per node per family; rows below
``zero_tol × global_max`` are dropped so free-interior nodes (where
the recorder typically writes effective zero) don't clutter the
scene with tiny arrows. Forces and moments are filtered
independently — a node may host a moment but no force, or vice
versa.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import ReactionsStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_REACTION_FORCE_COMPONENTS: tuple[str, str, str] = (
    "reaction_force_x",
    "reaction_force_y",
    "reaction_force_z",
)

_REACTION_MOMENT_COMPONENTS: tuple[str, str, str] = (
    "reaction_moment_x",
    "reaction_moment_y",
    "reaction_moment_z",
)


class _GlyphLayer:
    """Per-family render state (force or moment).

    Encapsulates the source PolyData + actor + cached substrate
    indices + the FEM ids feeding the slab read. Detach is the
    caller's job — the layer doesn't own a plotter reference.
    """

    __slots__ = (
        "source", "actor", "fem_ids", "substrate_idx",
        "initial_scale", "runtime_scale",
    )

    def __init__(self) -> None:
        self.source: Optional[pv.PolyData] = None
        self.actor: Any = None
        self.fem_ids: Optional[ndarray] = None
        self.substrate_idx: Optional[ndarray] = None
        self.initial_scale: float = 1.0
        self.runtime_scale: Optional[float] = None

    @property
    def current_scale(self) -> float:
        return (
            self.runtime_scale
            if self.runtime_scale is not None
            else self.initial_scale
        )

    def clear(self) -> None:
        self.source = None
        self.actor = None
        self.fem_ids = None
        self.substrate_idx = None


class ReactionsDiagram(Diagram):
    """Arrows + curved arrows at constrained nodes for recorded reactions."""

    kind = "reactions"
    topology = "nodes"  # data lives in results.nodes

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, ReactionsStyle):
            raise TypeError(
                "ReactionsDiagram requires a ReactionsStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)
        self._force = _GlyphLayer()
        self._moment = _GlyphLayer()

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
                "ReactionsDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, fem, scene)
        style: ReactionsStyle = self.spec.style    # type: ignore[assignment]

        # Resolve nodes from the selector. ``None`` = all nodes; the
        # auto-filter below trims to nodes with non-zero recorded
        # reactions per family.
        node_ids = self._resolved_node_ids
        if node_ids is None:
            node_ids = scene.node_ids.copy()
        if node_ids.size == 0:
            return

        if style.show_forces:
            self._build_layer(
                self._force,
                node_ids,
                _REACTION_FORCE_COMPONENTS,
                user_scale=style.force_scale,
                auto_fraction=style.auto_scale_fraction,
                color=style.force_color,
                geom_kind="arrow",
                style=style,
                plotter=plotter,
                scene=scene,
                actor_suffix="force",
            )
        if style.show_moments:
            self._build_layer(
                self._moment,
                node_ids,
                _REACTION_MOMENT_COMPONENTS,
                user_scale=style.moment_scale,
                auto_fraction=style.auto_scale_fraction,
                color=style.moment_color,
                geom_kind="moment",
                style=style,
                plotter=plotter,
                scene=scene,
                actor_suffix="moment",
            )

        actors = [
            layer.actor for layer in (self._force, self._moment)
            if layer.actor is not None
        ]
        self._actors = actors

    def update_to_step(self, step_index: int) -> None:
        for layer, comps, geom_kind in (
            (self._force, _REACTION_FORCE_COMPONENTS, "arrow"),
            (self._moment, _REACTION_MOMENT_COMPONENTS, "moment"),
        ):
            if layer.source is None or layer.actor is None:
                continue
            vecs = self._read_vectors(layer.fem_ids, comps, int(step_index))
            if vecs is None:
                continue
            layer.source.point_data["_vec"][:] = vecs
            layer.source.point_data["_mag"][:] = np.linalg.norm(vecs, axis=1)
            try:
                layer.source.Modified()
            except Exception:
                pass
            glyph = self._build_glyph(layer, geom_kind)
            try:
                mapper = layer.actor.GetMapper()
                mapper.SetInputData(glyph)
                mapper.Modified()
            except Exception:
                pass

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        target_pts: Optional[ndarray]
        try:
            target_pts = (
                np.asarray(deformed_pts, dtype=np.float64)
                if deformed_pts is not None
                else np.asarray(scene.grid.points, dtype=np.float64)
            )
        except Exception:
            return
        for layer, geom_kind in (
            (self._force, "arrow"),
            (self._moment, "moment"),
        ):
            if (
                layer.source is None
                or layer.actor is None
                or layer.substrate_idx is None
            ):
                continue
            try:
                layer.source.points = target_pts[layer.substrate_idx]
                glyph = self._build_glyph(layer, geom_kind)
                mapper = layer.actor.GetMapper()
                mapper.SetInputData(glyph)
                mapper.Modified()
            except Exception:
                pass

    def detach(self) -> None:
        self._force.clear()
        self._moment.clear()
        super().detach()

    # ------------------------------------------------------------------
    # Runtime adjustments
    # ------------------------------------------------------------------

    def set_force_scale(self, scale: float) -> None:
        self._set_scale(self._force, "arrow", scale)

    def set_moment_scale(self, scale: float) -> None:
        self._set_scale(self._moment, "moment", scale)

    def _set_scale(
        self, layer: _GlyphLayer, geom_kind: str, scale: float,
    ) -> None:
        layer.runtime_scale = float(scale)
        if layer.source is not None and layer.actor is not None:
            try:
                glyph = self._build_glyph(layer, geom_kind)
                layer.actor.GetMapper().SetInputData(glyph)
                layer.actor.GetMapper().Modified()
            except Exception:
                pass

    def current_force_scale(self) -> float:
        return self._force.current_scale

    def current_moment_scale(self) -> float:
        return self._moment.current_scale

    # ------------------------------------------------------------------
    # Internal — build one family's actor
    # ------------------------------------------------------------------

    def _build_layer(
        self,
        layer: _GlyphLayer,
        node_ids: ndarray,
        components: tuple[str, str, str],
        *,
        user_scale: Optional[float],
        auto_fraction: float,
        color: str,
        geom_kind: str,
        style: ReactionsStyle,
        plotter: Any,
        scene: "FEMSceneData",
        actor_suffix: str,
    ) -> None:
        """Read full slab → filter zero rows → build PolyData + actor."""
        full_vecs = self._read_vectors_all_steps(node_ids, components)
        if full_vecs is None or full_vecs.size == 0:
            return
        per_node_max = np.linalg.norm(full_vecs, axis=2).max(axis=0)
        global_max = float(per_node_max.max())
        if global_max <= 0.0:
            return
        keep = per_node_max > (style.zero_tol * global_max)
        if not keep.any():
            return
        kept_ids = node_ids[keep]

        # Map FEM ids → substrate row indices.
        max_id = int(max(kept_ids.max(), scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(
            scene.node_ids.size, dtype=np.int64,
        )
        substrate_idx = lookup[kept_ids]
        valid = substrate_idx >= 0
        if not valid.any():
            return
        kept_ids = kept_ids[valid]
        substrate_idx = substrate_idx[valid]

        coords = np.asarray(scene.grid.points)[substrate_idx].copy()
        n = kept_ids.size
        source = pv.PolyData(coords)
        source.point_data["_vec"] = np.zeros((n, 3), dtype=np.float64)
        source.point_data["_mag"] = np.zeros(n, dtype=np.float64)
        layer.source = source
        layer.fem_ids = kept_ids.astype(np.int64)
        layer.substrate_idx = substrate_idx.copy()

        # Step-0 vectors for the initial render.
        vecs0 = self._read_vectors(layer.fem_ids, components, 0)
        if vecs0 is not None:
            source.point_data["_vec"][:] = vecs0
            source.point_data["_mag"][:] = np.linalg.norm(vecs0, axis=1)

        # Auto-fit against the worst step in the time-history.
        if user_scale is None:
            if global_max > 0.0 and scene.model_diagonal > 0.0:
                layer.initial_scale = (
                    auto_fraction * scene.model_diagonal / global_max
                )
            else:
                layer.initial_scale = 1.0
        else:
            layer.initial_scale = float(user_scale)

        glyph = self._build_glyph(layer, geom_kind)
        actor = plotter.add_mesh(
            glyph,
            name=self._actor_name(actor_suffix),
            color=color,
            reset_camera=False,
            lighting=False,
            pickable=False,
            show_scalar_bar=False,
        )
        layer.actor = actor

    # ------------------------------------------------------------------
    # Internal — slab reads
    # ------------------------------------------------------------------

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            try:
                return self._results.stage(self.spec.stage_id)
            except Exception:
                return None
        return self._results

    def _read_vectors(
        self,
        fem_ids: Optional[ndarray],
        components: tuple[str, str, str],
        step_index: int,
    ) -> Optional[ndarray]:
        """Read one step of the (N, 3) reaction slab for ``fem_ids``."""
        if fem_ids is None or fem_ids.size == 0:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        n = fem_ids.size
        out = np.zeros((n, 3), dtype=np.float64)
        max_id = int(fem_ids.max()) + 1
        pos = np.full(max_id + 1, -1, dtype=np.int64)
        pos[fem_ids] = np.arange(n, dtype=np.int64)
        for axis, comp in enumerate(components):
            try:
                slab = results.nodes.get(
                    ids=fem_ids,
                    component=comp,
                    time=[step_index],
                )
            except Exception:
                continue
            if slab.values.size == 0:
                continue
            slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
            slab_vals = np.asarray(slab.values[0], dtype=np.float64)
            positions = pos[slab_ids]
            valid = positions >= 0
            out[positions[valid], axis] = slab_vals[valid]
        return out

    def _read_vectors_all_steps(
        self, node_ids: ndarray, components: tuple[str, str, str],
    ) -> Optional[ndarray]:
        """Return ``(T, N, 3)`` reaction slab across every step."""
        results = self._scoped_results()
        if results is None:
            return None
        slabs: list[ndarray] = []
        slab_ids_ref: Optional[ndarray] = None
        n_t = 0
        for comp in components:
            try:
                slab = results.nodes.get(
                    ids=node_ids,
                    component=comp,
                    time=None,
                )
            except Exception:
                slabs.append(np.zeros((0, 0)))
                continue
            v = np.asarray(slab.values, dtype=np.float64)
            if v.ndim != 2 or v.size == 0:
                slabs.append(np.zeros((0, 0)))
                continue
            slabs.append(v)
            n_t = max(n_t, v.shape[0])
            if slab_ids_ref is None:
                slab_ids_ref = np.asarray(slab.node_ids, dtype=np.int64)
        if slab_ids_ref is None or n_t == 0:
            return None
        n = int(node_ids.size)
        out = np.zeros((n_t, n, 3), dtype=np.float64)
        max_id = int(max(node_ids.max(), slab_ids_ref.max())) + 1
        pos = np.full(max_id + 1, -1, dtype=np.int64)
        pos[node_ids] = np.arange(n, dtype=np.int64)
        positions = pos[slab_ids_ref]
        valid = positions >= 0
        for axis, v in enumerate(slabs):
            if v.size == 0 or v.shape[0] == 0:
                continue
            t = min(n_t, v.shape[0])
            out[:t, positions[valid], axis] = v[:t][:, valid]
        return out

    # ------------------------------------------------------------------
    # Internal — glyph build
    # ------------------------------------------------------------------

    def _actor_name(self, suffix: str) -> str:
        return f"diagram_reactions_{id(self):x}_{suffix}"

    def _build_glyph(
        self, layer: _GlyphLayer, geom_kind: str,
    ) -> pv.PolyData:
        if layer.source is None:
            return pv.PolyData()
        style: ReactionsStyle = self.spec.style    # type: ignore[assignment]
        try:
            if geom_kind == "moment":
                from ..overlays.moment_glyph import make_moment_glyph
                geom = make_moment_glyph(
                    arc_degrees=float(style.moment_arc_degrees),
                )
            else:
                geom = pv.Arrow(tip_length=style.arrow_tip_fraction)
            return layer.source.glyph(
                orient="_vec",
                scale="_mag",
                factor=float(layer.current_scale),
                geom=geom,
            )
        except Exception:
            return pv.PolyData(np.asarray(layer.source.points))
