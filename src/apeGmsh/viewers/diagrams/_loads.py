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

Render seam (ADR 0042, R-B.0): this diagram is the first migrated to
the backend-neutral IR. It emits a single :class:`GlyphLayer` of arrow
glyphs through ``self._backend`` and holds no VTK objects — the
backend owns the actor. The arrow size is folded into the layer's
per-glyph ``scales`` (= magnitude × current scale) so the backend's
generic glyph path reproduces the legacy ``factor``-scaled arrows.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._kinds import register_diagram_kind
from ._styles import LoadsStyle
from ..scene_ir import ColorSpec, GlyphLayer, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


@register_diagram_kind(label="Applied loads", style_class=LoadsStyle, order=90)
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
        # Reference-config arrow tail coords (substrate sample), the
        # force vectors, and their magnitudes — cached so set_scale /
        # sync_substrate_points can rebuild the layer.
        self._coords: Optional[ndarray] = None
        self._forces: Optional[ndarray] = None
        self._mags: Optional[ndarray] = None
        # Substrate row indices the arrow tails sampled — for
        # sync_substrate_points re-positioning under deformation.
        self._substrate_idx: Optional[ndarray] = None
        self._initial_scale: float = 1.0
        self._runtime_scale: Optional[float] = None
        # Backend handle + the last emitted layer (the latter exposed
        # for tests / introspection — there is no VTK actor to hold).
        self._handle: Any = None
        self._layer: Optional[GlyphLayer] = None

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
            raise RuntimeError("LoadsDiagram.attach requires a FEMSceneData.")
        super().attach(plotter, view, scene)
        style: LoadsStyle = self.spec.style    # type: ignore[assignment]
        pattern = self.spec.selector.component

        # Collect (node_id, force) pairs for this pattern. Skip records
        # without a force vector and zero-force records — a zero-length
        # arrow renders as a degenerate glyph.
        node_ids: list[int] = []
        forces: list[tuple[float, float, float]] = []
        try:
            records = list(view.nodes.loads.by_pattern(pattern))
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
        mags = np.linalg.norm(forces_arr, axis=1)
        self._coords = coords
        self._forces = forces_arr
        self._mags = mags
        self._substrate_idx = substrate_idx.copy()

        # Auto-fit: largest arrow reaches ``auto_scale_fraction`` of the
        # model diagonal. Mirror VectorGlyph's convention.
        mag_max = float(mags.max())
        if style.scale is None:
            if mag_max > 0.0 and scene.model_diagonal > 0.0:
                self._initial_scale = (
                    style.auto_scale_fraction * scene.model_diagonal / mag_max
                )
            else:
                self._initial_scale = 1.0
        else:
            self._initial_scale = float(style.scale)

        self._layer = self._build_layer(coords, self.current_scale())
        self._handle = self._backend.add_layer(self._layer)

    def update_to_step(self, step_index: int) -> None:
        # Constant — broker has no timeSeries info to drive a per-step
        # multiplier. Step changes are silently ignored.
        return

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Move arrow tails to follow the deformed substrate."""
        if (
            self._forces is None
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
        self._layer = self._build_layer(self._coords, self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def detach(self) -> None:
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._handle = None
        self._layer = None
        self._coords = None
        self._forces = None
        self._mags = None
        self._substrate_idx = None
        super().detach()

    # ------------------------------------------------------------------
    # Visibility (backend-routed — no VTK actor held here)
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
        if self._coords is not None and self._handle is not None:
            self._layer = self._build_layer(self._coords, self.current_scale())
            self._backend.update_layer(self._handle, self._layer)

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"loads_{id(self):x}"

    def _build_layer(self, coords: ndarray, scale: float) -> GlyphLayer:
        """Build the arrow GlyphLayer for the current coords + scale.

        The per-glyph scale is ``magnitude × scale`` so the backend's
        generic glyph path (scale by the ``scales`` array, orient by
        ``orientations``) reproduces the legacy ``factor``-scaled arrows.
        """
        style: LoadsStyle = self.spec.style    # type: ignore[assignment]
        assert self._forces is not None and self._mags is not None
        return GlyphLayer(
            layer_id=self._layer_id(),
            positions=PointSet(coords),
            kind="arrow",
            orientations=self._forces,
            scales=self._mags * float(scale),
            color=ColorSpec(mode="solid", solid_rgb=style.color),
        )
