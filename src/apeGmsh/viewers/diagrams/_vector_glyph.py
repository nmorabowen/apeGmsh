"""VectorGlyphDiagram — arrows at nodes from N components of a NodeSlab.

``selector.component`` is the diagram's display label; the actual
vector field is built from ``style.components`` (default: 3-D
translational displacement).

Render seam (ADR 0042, R-B Wave 1 #4 — first colour-mapped diagram).
Emits one arrow :class:`GlyphLayer` via the backend; holds no VTK
objects. Glyph **size** uses ``magnitude × scale`` (folded into
``scales``); when ``use_magnitude_colors`` the glyph **colour** uses the
raw magnitude (``color_scalar`` + ``ColorSpec(by_array)``). The Qt LUT
mirror stays diagram-side — its ``changed`` signal is translated into a
plain ``ColorSpec`` and pushed through ``backend.set_layer_color`` (the
backend never sees Qt).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import VectorGlyphStyle
from ..scene_ir import ColorSpec, GlyphLayer, LutSpec, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


def _default_style(component: str) -> VectorGlyphStyle:
    """Build a default ``VectorGlyphStyle`` for the user's selection.

    The catalog offers each vector prefix plus its per-axis options
    (``displacement``, ``displacement_x``, ...). Either form names a
    field; we resolve both back to the prefix so ``components`` reads
    the *correct* x/y/z triple — picking ``velocity`` reads velocity,
    not a hardcoded displacement default.
    """
    from ._kind_catalog import resolve_vector_prefix
    prefix = resolve_vector_prefix(component) if component else "displacement"
    return VectorGlyphStyle(components=(
        f"{prefix}_x", f"{prefix}_y", f"{prefix}_z",
    ))


@register_diagram_kind(
    label="Vector glyph (arrows)",
    style_class=VectorGlyphStyle,
    style_factory=_default_style,
    order=60,
)
class VectorGlyphDiagram(ScalarColorSupport, Diagram):
    """Arrows at nodes, oriented and scaled by an N-component vector field."""

    kind = "vector_glyph"
    topology = "nodes"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, VectorGlyphStyle):
            raise TypeError(
                "VectorGlyphDiagram requires a VectorGlyphStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._layer: Optional[GlyphLayer] = None
        self._handle: Any = None
        self._coords: Optional[ndarray] = None
        self._fem_ids_to_read: Optional[ndarray] = None
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._substrate_idx: Optional[ndarray] = None
        self._initial_scale: float = 1.0
        self._runtime_scale: Optional[float] = None
        # Scalar-bar + runtime colour state + LUT mirror (mixin; the
        # LUT is built only when colouring by magnitude).
        self._init_scalar_color_state()

        comp = getattr(spec.selector, "component", "") or ""
        style: VectorGlyphStyle = spec.style    # type: ignore[assignment]
        try:
            self._axis: Optional[int] = list(style.components).index(comp)
        except ValueError:
            self._axis = None

    def _scalar_bar_is_enabled(self) -> bool:
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]
        return bool(getattr(style, "use_magnitude_colors", False))

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
            raise RuntimeError("VectorGlyphDiagram.attach requires a FEMSceneData.")
        super().attach(plotter, view, scene)
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]

        node_ids = self._resolved_node_ids
        if node_ids is None:
            node_ids = scene.node_ids.copy()
        if node_ids.size == 0:
            return

        max_id = int(max(node_ids.max(), scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(scene.node_ids.size, dtype=np.int64)
        substrate_idx = lookup[node_ids]
        valid = substrate_idx >= 0
        if not valid.any():
            return
        node_ids = node_ids[valid]
        substrate_idx = substrate_idx[valid]
        coords = np.asarray(scene.grid.points)[substrate_idx].copy()

        n = node_ids.size
        max_id = int(node_ids.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[node_ids] = np.arange(n, dtype=np.int64)
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = node_ids
        self._substrate_idx = substrate_idx.copy()
        self._coords = coords

        vecs0 = self._read_vectors(0)
        mags0 = np.linalg.norm(vecs0, axis=1)
        global_mag_max = self._read_global_mag_max()

        if style.scale is None:
            max_abs = global_mag_max if global_mag_max > 0.0 else float(
                mags0.max() if mags0.size else 0.0
            )
            if max_abs > 0.0 and scene.model_diagonal > 0.0:
                self._initial_scale = (
                    style.auto_scale_fraction * scene.model_diagonal / max_abs
                )
            else:
                self._initial_scale = 1.0
        else:
            self._initial_scale = float(style.scale)

        if style.clim is not None:
            self._initial_clim = (float(style.clim[0]), float(style.clim[1]))
        else:
            hi = (
                global_mag_max if global_mag_max > 0.0
                else float(mags0[np.isfinite(mags0)].max())
                if mags0.size and np.isfinite(mags0).any()
                else 1.0
            )
            lo = 0.0
            if hi <= lo:
                hi = lo + 1.0
            self._initial_clim = (lo, hi)

        self._layer = self._build_layer(vecs0, mags0, self.current_scale())
        self._handle = self._backend.add_layer(self._layer)

        # LUT mirror + scalar bar only when colouring by magnitude.
        if style.use_magnitude_colors:
            self._init_lut()
            if self._effective_show_scalar_bar():
                self._backend.add_scalar_bar(
                    self._handle, self._make_scalar_bar_spec(),
                )

    def update_to_step(self, step_index: int) -> None:
        if self._layer is None or self._handle is None:
            return
        vecs = self._read_vectors(int(step_index))
        if vecs is None:
            return
        mags = np.linalg.norm(vecs, axis=1)
        self._layer = self._build_layer(vecs, mags, self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Move arrow tails to follow the deformed substrate."""
        if (
            self._layer is None
            or self._handle is None
            or self._substrate_idx is None
        ):
            return
        target_pts = (
            np.asarray(deformed_pts, dtype=np.float64)
            if deformed_pts is not None
            else np.asarray(scene.grid.points, dtype=np.float64)
        )
        self._coords = target_pts[self._substrate_idx]
        # Rebuild at the new tail positions from the layer's existing
        # orientations + magnitudes (arrows orient/scale unchanged).
        orientations = np.asarray(self._layer.orientations, dtype=np.float64)
        mags = (
            np.asarray(self._layer.color_scalar, dtype=np.float64)
            if self._layer.color_scalar is not None
            else np.linalg.norm(orientations, axis=1)
        )
        self._layer = self._build_layer(orientations, mags, self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._coords = None
        self._fem_ids_to_read = None
        self._submesh_pos_of_id = None
        self._substrate_idx = None
        self._initial_clim = None
        super().detach()

    # ------------------------------------------------------------------
    # Visibility (backend-routed)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Runtime adjustments
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        self._runtime_scale = float(scale)
        if self._layer is not None and self._handle is not None:
            mags = (
                np.asarray(self._layer.color_scalar, dtype=np.float64)
                if self._layer.color_scalar is not None
                else np.linalg.norm(
                    np.asarray(self._layer.orientations, dtype=np.float64), axis=1
                )
            )
            vecs = np.asarray(self._layer.orientations, dtype=np.float64)
            self._layer = self._build_layer(vecs, mags, self.current_scale())
            self._backend.update_layer(self._handle, self._layer)

    # clim/cmap/LUT handling comes from ScalarColorSupport; the LUT is
    # built only when colouring by magnitude (the _init_lut call site
    # in attach is gated on ``use_magnitude_colors``).

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        if self._layer is None:
            return None
        return self._layer.color_scalar

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"vector_{id(self):x}"

    def _color_array_name(self) -> str:
        return self.spec.selector.component or "magnitude"

    def _build_layer(
        self, vecs: ndarray, mags: ndarray, scale: float,
    ) -> GlyphLayer:
        """Arrow glyph: size = magnitude × scale; colour (when enabled) =
        raw magnitude through the LUT."""
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]
        assert self._coords is not None
        if style.use_magnitude_colors:
            clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
            cmap = self._runtime_cmap or style.cmap
            color = ColorSpec(
                mode="by_array",
                array_name=self._color_array_name(),
                lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
            )
            color_scalar = mags
        else:
            color = ColorSpec(mode="solid", solid_rgb=style.color)
            color_scalar = None
        return GlyphLayer(
            layer_id=self._layer_id(),
            positions=PointSet(self._coords),
            kind="arrow",
            orientations=vecs,
            scales=mags * float(scale),
            color_scalar=color_scalar,
            color=color,
        )

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
            if self._axis is not None and axis != self._axis:
                continue
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

    def _read_global_mag_max(self) -> float:
        """Maximum |vector| across every step. ``0.0`` if unavailable."""
        if self._fem_ids_to_read is None or self._submesh_pos_of_id is None:
            return 0.0
        results = self._scoped_results()
        if results is None:
            return 0.0
        style: VectorGlyphStyle = self.spec.style    # type: ignore[assignment]

        slabs: list[ndarray] = []
        n_t = 0
        slab_ids_ref: Optional[ndarray] = None
        for comp in style.components[:3]:
            try:
                slab = results.nodes.get(
                    ids=self._fem_ids_to_read,
                    component=comp,
                    time=None,
                )
            except Exception:
                continue
            if slab.values.size == 0:
                continue
            v = np.asarray(slab.values, dtype=np.float64)
            if v.ndim != 2:
                continue
            slabs.append(v)
            if v.shape[0] > n_t:
                n_t = v.shape[0]
            if slab_ids_ref is None:
                slab_ids_ref = np.asarray(slab.node_ids, dtype=np.int64)
        if not slabs or n_t == 0:
            return 0.0
        ssq = np.zeros_like(slabs[0])
        for v in slabs:
            t = min(ssq.shape[0], v.shape[0])
            ssq[:t] += v[:t] ** 2
        try:
            mag_max = float(np.sqrt(ssq).max())
        except ValueError:
            return 0.0
        return mag_max if np.isfinite(mag_max) else 0.0
