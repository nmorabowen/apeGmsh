"""ReactionsDiagram — arrows at constrained nodes from recorded reactions.

Renders BOTH ``reaction_force_*`` and ``reaction_moment_*`` from the
nodes composite of the stage-scoped Results. Forces are drawn with
straight arrow glyphs; moments with the curved-arrow ``kind="moment"``
glyph. Each family auto-fits its own scale because forces and torques
have different units — coupling them would distort one of the two.

Render seam (ADR 0042, R-B Wave 3 #2). Each family emits one
:class:`GlyphLayer` through ``self._backend``; the diagram holds no VTK
objects. The moment family uses the additive ``kind="moment"`` glyph —
the curved-arrow geometry (and its ``arc_degrees``) is built backend-
side, so the diagram only carries the arc spec on the IR.

Unlike :class:`LoadsDiagram`, reactions ARE step-resolved (the
recorder writes them per step), so ``update_to_step`` re-reads the
slabs and re-emits the glyph layers — same pattern as
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
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._kinds import register_diagram_kind
from ._styles import ReactionsStyle
from ..scene_ir import ColorSpec, GlyphLayer, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
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

# selector.component → which axis to render. ``None`` keeps the
# resultant 3-vector (combined force + moment families). An axis index
# selects a single force component and silences moments — the user
# wired this through the Data combo to compare force balance per axis.
_AXIS_FROM_COMPONENT: dict[str, Optional[int]] = {
    "reactions":   None,
    "reaction_x":  0,
    "reaction_y":  1,
    "reaction_z":  2,
}


class _Family:
    """Per-family render state (force or moment).

    Holds the emitted :class:`GlyphLayer` + backend handle + cached
    substrate indices, FEM ids, world coords, and the current step's
    vectors. No VTK objects. Detach is the caller's job.
    """

    __slots__ = (
        "handle", "layer", "fem_ids", "substrate_idx", "coords", "vecs",
        "color", "geom_kind", "suffix", "initial_scale", "runtime_scale",
    )

    def __init__(self) -> None:
        self.handle: Any = None
        self.layer: Optional[GlyphLayer] = None
        self.fem_ids: Optional[ndarray] = None
        self.substrate_idx: Optional[ndarray] = None
        self.coords: Optional[ndarray] = None
        self.vecs: Optional[ndarray] = None
        self.color: str = "#FFFFFF"
        self.geom_kind: str = "arrow"
        self.suffix: str = "force"
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
        self.handle = None
        self.layer = None
        self.fem_ids = None
        self.substrate_idx = None
        self.coords = None
        self.vecs = None


@register_diagram_kind(label="Reactions", style_class=ReactionsStyle, order=100)
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
        self._force = _Family()
        self._moment = _Family()
        comp = getattr(spec.selector, "component", "") or ""
        self._axis: Optional[int] = _AXIS_FROM_COMPONENT.get(comp, None)

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
                "ReactionsDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, view, scene)
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
            self._build_family(
                self._force,
                node_ids,
                _REACTION_FORCE_COMPONENTS,
                user_scale=style.force_scale,
                auto_fraction=style.auto_scale_fraction,
                color=style.force_color,
                geom_kind="arrow",
                style=style,
                scene=scene,
                suffix="force",
            )
        # Axis-locked modes show forces only — three curved-arrow
        # glyphs alongside an axis-aligned force read as visual noise.
        if style.show_moments and self._axis is None:
            self._build_family(
                self._moment,
                node_ids,
                _REACTION_MOMENT_COMPONENTS,
                user_scale=style.moment_scale,
                auto_fraction=style.auto_scale_fraction,
                color=style.moment_color,
                geom_kind="moment",
                style=style,
                scene=scene,
                suffix="moment",
            )

    def update_to_step(self, step_index: int) -> None:
        for family, comps in (
            (self._force, _REACTION_FORCE_COMPONENTS),
            (self._moment, _REACTION_MOMENT_COMPONENTS),
        ):
            if family.handle is None:
                continue
            vecs = self._read_vectors(family.fem_ids, comps, int(step_index))
            if vecs is None:
                continue
            family.vecs = vecs
            self._rebuild_and_push(family)

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
        for family in (self._force, self._moment):
            if family.handle is None or family.substrate_idx is None:
                continue
            try:
                family.coords = target_pts[family.substrate_idx]
            except Exception:
                continue
            self._rebuild_and_push(family)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is None:
            return
        for family in (self._force, self._moment):
            if family.handle is not None:
                self._backend.set_layer_visible(family.handle, bool(visible))

    def detach(self) -> None:
        if self._backend is not None:
            for family in (self._force, self._moment):
                if family.handle is not None:
                    self._backend.remove_layer(family.handle)
        self._force.clear()
        self._moment.clear()
        super().detach()

    # ------------------------------------------------------------------
    # Runtime adjustments
    # ------------------------------------------------------------------

    def set_force_scale(self, scale: float) -> None:
        self._set_scale(self._force, scale)

    def set_moment_scale(self, scale: float) -> None:
        self._set_scale(self._moment, scale)

    def _set_scale(self, family: _Family, scale: float) -> None:
        family.runtime_scale = float(scale)
        if family.handle is not None:
            self._rebuild_and_push(family)

    def current_force_scale(self) -> float:
        return self._force.current_scale

    def current_moment_scale(self) -> float:
        return self._moment.current_scale

    # ------------------------------------------------------------------
    # Internal — build one family's actor
    # ------------------------------------------------------------------

    def _build_family(
        self,
        family: _Family,
        node_ids: ndarray,
        components: tuple[str, str, str],
        *,
        user_scale: Optional[float],
        auto_fraction: float,
        color: str,
        geom_kind: str,
        style: ReactionsStyle,
        scene: "FEMSceneData",
        suffix: str,
    ) -> None:
        """Read full slab → filter zero rows → emit a GlyphLayer."""
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
        family.fem_ids = kept_ids.astype(np.int64)
        family.substrate_idx = substrate_idx.copy()
        family.coords = coords
        family.color = color
        family.geom_kind = geom_kind
        family.suffix = suffix

        # Step-0 vectors for the initial render.
        vecs0 = self._read_vectors(family.fem_ids, components, 0)
        family.vecs = (
            vecs0 if vecs0 is not None
            else np.zeros((n, 3), dtype=np.float64)
        )

        # Auto-fit against the worst step in the time-history.
        if user_scale is None:
            if global_max > 0.0 and scene.model_diagonal > 0.0:
                family.initial_scale = (
                    auto_fraction * scene.model_diagonal / global_max
                )
            else:
                family.initial_scale = 1.0
        else:
            family.initial_scale = float(user_scale)

        family.layer = self._build_family_layer(family)
        family.handle = self._backend.add_layer(family.layer)

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
        """Read one step of the (N, 3) reaction slab for ``fem_ids``.

        In axis-locked modes (``self._axis is not None``) the off-axis
        columns are zeroed *after* the read, so the glyph mapper sees
        ``_mag = |component|`` and culls zero-length arrows. The full
        slab is still read at attach time (via ``_read_vectors_all_steps``)
        for auto-fit scaling, which keeps reaction_x / reaction_y / etc.
        rendered at comparable lengths.
        """
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
            if self._axis is not None and axis != self._axis:
                continue
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

    def _layer_id(self, suffix: str) -> str:
        return f"reactions_{id(self):x}_{suffix}"

    def _build_family_layer(self, family: _Family) -> GlyphLayer:
        """Build the family's GlyphLayer from its cached coords + vectors.

        Glyph **size** = ``|vec| × scale`` (folded into ``scales``);
        orientation = the raw reaction vector. The moment family uses
        the curved-arrow ``kind="moment"`` glyph; the backend builds its
        geometry from ``arc_degrees``.
        """
        style: ReactionsStyle = self.spec.style    # type: ignore[assignment]
        assert family.coords is not None and family.vecs is not None
        vecs = family.vecs
        mags = np.linalg.norm(vecs, axis=1)
        kind = "moment" if family.geom_kind == "moment" else "arrow"
        return GlyphLayer(
            layer_id=self._layer_id(family.suffix),
            positions=PointSet(family.coords),
            kind=kind,
            orientations=vecs,
            scales=mags * float(family.current_scale),
            color=ColorSpec(mode="solid", solid_rgb=family.color),
            arc_degrees=(
                float(style.moment_arc_degrees) if kind == "moment" else None
            ),
        )

    def _rebuild_and_push(self, family: _Family) -> None:
        if family.handle is None or self._backend is None:
            return
        family.layer = self._build_family_layer(family)
        self._backend.update_layer(family.handle, family.layer)
