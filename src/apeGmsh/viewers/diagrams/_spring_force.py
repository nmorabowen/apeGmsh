"""SpringForceDiagram — arrows on zero-length springs.

For each spring (a 2-node element whose two nodes coincide), we render
one arrow at the spring's location, oriented along the configured
spring direction (deduced from the canonical component name suffix or
overridden via ``SpringForceStyle.direction``), with length scaled by
the absolute force value.

The ZeroLength element's spring directions are not carried by FEMData
generically — apeGmsh stores connectivity but not the per-spring
direction vectors. We therefore default to axis-aligned directions
(suffix 0 → x, 1 → y, 2 → z) and let the user override per diagram if
their model uses skew springs.

Render seam (ADR 0042, R-B): emits one arrow :class:`GlyphLayer` via
``self._backend`` and holds no VTK objects. Signed force flips the
arrow (orientation = value × direction); the absolute value drives the
glyph scale. Springs sit at the reference configuration (no
deformation sync). Same shape as the migrated LoadsDiagram.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import SpringForceStyle
from ..scene_ir import ColorSpec, GlyphLayer, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


_DEFAULT_AXES = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
)

_COMPONENT_SUFFIX = re.compile(r"^spring_(force|deformation)_(\d+)$")


def _direction_from_component(component: str) -> ndarray:
    """Default unit direction for ``spring_force_<n>`` style components."""
    m = _COMPONENT_SUFFIX.match(component)
    if m is None:
        return _DEFAULT_AXES[0].copy()
    idx = int(m.group(2))
    if 0 <= idx < 3:
        return _DEFAULT_AXES[idx].copy()
    return _DEFAULT_AXES[0].copy()


class SpringForceDiagram(Diagram):
    """Force / deformation arrow on each zero-length spring."""

    kind = "spring_force"
    topology = "springs"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, SpringForceStyle):
            raise TypeError(
                "SpringForceDiagram requires a SpringForceStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._layer: Optional[GlyphLayer] = None
        self._handle: Any = None
        self._element_ids_to_read: tuple[int, ...] = ()
        self._positions: Optional[ndarray] = None   # (n, 3) spring coords
        self._values: Optional[ndarray] = None      # (n,) signed force/def
        self._direction: Optional[ndarray] = None
        self._initial_scale: float = 1.0

        # Mapping from slab position -> spring index in our layer order
        self._slab_to_spring_pos: Optional[ndarray] = None

        # Anchor-node substrate row per glyph (-1 = not on substrate),
        # for sync_substrate_points deformation-following.
        self._substrate_rows: Optional[ndarray] = None

        self._runtime_scale: Optional[float] = None

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
                "SpringForceDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, view, scene)
        style: SpringForceStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (zero-length springs are 1-D elements) ─
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_zero_length_ids(view)
        if element_ids.size == 0:
            from ._base import NoDataError
            raise NoDataError(
                f"SpringForceDiagram: no zero-length spring elements "
                f"found in the selection (selector={self.spec.selector!r})."
            )
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Spring world positions (use node i — i and j coincide) ──
        anchors = self._collect_spring_anchors(view, element_ids)
        if not anchors:
            from ._base import NoDataError
            raise NoDataError(
                "SpringForceDiagram: could not resolve world positions "
                "for the spring elements."
            )

        # ── Step-0 read ─────────────────────────────────────────────
        results = self._scoped_results()
        if results is None:
            from ._base import NoDataError
            raise NoDataError(
                "SpringForceDiagram: results scope unresolved (no stage)."
            )
        try:
            slab = results.elements.springs.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SpringForceDiagram could not read springs slab: {exc}"
            )
        if slab.values.size == 0:
            from ._base import NoDataError
            raise NoDataError(
                f"SpringForceDiagram: no spring data for component "
                f"{self.spec.selector.component!r}. Use "
                f"`results.inspect.diagnose("
                f"{self.spec.selector.component!r})`."
            )

        slab_eids = np.asarray(slab.element_index, dtype=np.int64)
        slab_values_all = np.asarray(slab.values, dtype=np.float64)  # (T, N)
        slab_values = slab_values_all[0] if slab_values_all.size else slab_values_all
        n_slab = slab_eids.size

        # Reorder positions to match slab ordering
        positions_in_slab_order = np.zeros((n_slab, 3), dtype=np.float64)
        anchor_nids = np.zeros(n_slab, dtype=np.int64)
        valid_mask = np.zeros(n_slab, dtype=bool)
        for k in range(n_slab):
            anchor = anchors.get(int(slab_eids[k]))
            if anchor is not None:
                anchor_nids[k] = anchor[0]
                positions_in_slab_order[k] = anchor[1]
                valid_mask[k] = True

        if not valid_mask.any():
            return

        positions_in_slab_order = positions_in_slab_order[valid_mask]
        anchor_nids = anchor_nids[valid_mask]
        slab_values = slab_values[valid_mask]
        slab_values_all = (
            slab_values_all[:, valid_mask] if slab_values_all.ndim == 2
            else slab_values_all
        )
        self._positions = positions_in_slab_order
        self._slab_to_spring_pos = np.where(valid_mask)[0]

        # Anchor-node substrate rows so sync_substrate_points can
        # re-sample the deformed substrate (-1 = node not on substrate;
        # that glyph stays at its reference anchor).
        scene_ids = np.asarray(scene.node_ids, dtype=np.int64)
        max_sid = int(max(
            scene_ids.max() if scene_ids.size else 0,
            anchor_nids.max() if anchor_nids.size else 0,
        ))
        nid_to_sub = np.full(max_sid + 2, -1, dtype=np.int64)
        nid_to_sub[scene_ids] = np.arange(scene_ids.size, dtype=np.int64)
        self._substrate_rows = nid_to_sub[anchor_nids]

        # Direction
        if style.direction is not None:
            d = np.asarray(style.direction, dtype=np.float64)
            norm = float(np.linalg.norm(d))
            if norm < 1e-12:
                d = _DEFAULT_AXES[0].copy()
            else:
                d = d / norm
        else:
            d = _direction_from_component(self.spec.selector.component)
        self._direction = d

        # Auto scale at attach — global max-abs across every step
        if style.scale is None:
            max_abs = (
                float(np.abs(slab_values_all).max())
                if slab_values_all.size else 0.0
            )
            if max_abs > 0.0 and scene.model_diagonal > 0.0:
                self._initial_scale = (
                    style.auto_scale_fraction * scene.model_diagonal / max_abs
                )
            else:
                self._initial_scale = 1.0
        else:
            self._initial_scale = float(style.scale)

        # Emit the arrow glyph layer through the backend.
        self._values = slab_values
        self._layer = self._build_layer(slab_values, self.current_scale())
        self._handle = self._backend.add_layer(self._layer)

    def update_to_step(self, step_index: int) -> None:
        if self._layer is None or self._handle is None or self._direction is None:
            return
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.springs.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return
        if slab.values.size == 0:
            return
        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if (
            self._slab_to_spring_pos is None
            or slab_values.size <= int(self._slab_to_spring_pos.max())
        ):
            return
        ours = slab_values[self._slab_to_spring_pos]
        self._values = ours
        self._layer = self._build_layer(ours, self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Move the arrow anchors to follow the deformed substrate
        (the glyph positions are owned geometry, sampled from the
        reference node coords at attach)."""
        if (
            self._handle is None
            or self._positions is None
            or self._values is None
            or self._substrate_rows is None
        ):
            return
        try:
            target = (
                np.asarray(deformed_pts, dtype=np.float64)
                if deformed_pts is not None
                else np.asarray(scene.grid.points, dtype=np.float64)
            )
        except Exception:
            return
        rows = self._substrate_rows
        on_sub = (rows >= 0) & (rows < target.shape[0])
        if not on_sub.any():
            return
        new_pos = self._positions.copy()
        new_pos[on_sub] = target[rows[on_sub]]
        self._positions = new_pos
        self._layer = self._build_layer(self._values, self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def detach(self) -> None:
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._element_ids_to_read = ()
        self._positions = None
        self._values = None
        self._direction = None
        self._slab_to_spring_pos = None
        self._substrate_rows = None
        super().detach()

    # ------------------------------------------------------------------
    # Visibility (backend-routed)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Runtime style
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        self._runtime_scale = float(scale)
        if self._values is not None and self._handle is not None:
            self._layer = self._build_layer(self._values, self.current_scale())
            self._backend.update_layer(self._handle, self._layer)

    def set_direction(
        self, direction: tuple[float, float, float],
    ) -> None:
        d = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            return
        self._direction = d / norm
        if self._values is None or self._handle is None:
            return
        self._layer = self._build_layer(self._values, self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"spring_{id(self):x}"

    def _build_layer(self, values: ndarray, scale: float) -> GlyphLayer:
        """Arrow glyph layer: orientation = value × dir (sign flips the
        arrow), scale = |value| × ``scale``."""
        style: SpringForceStyle = self.spec.style    # type: ignore[assignment]
        assert self._positions is not None and self._direction is not None
        orientations = values[:, None] * self._direction[None, :]
        scales = np.abs(values) * float(scale)
        return GlyphLayer(
            layer_id=self._layer_id(),
            positions=PointSet(self._positions),
            kind="arrow",
            orientations=orientations,
            scales=scales,
            color=ColorSpec(mode="solid", solid_rgb=style.color),
        )

    @staticmethod
    def _collect_zero_length_ids(view: "ViewerData") -> ndarray:
        """All 1-D element IDs (springs are 1-D, dim==1)."""
        ids: list[int] = []
        for group in view.elements:
            if group.element_type.dim == 1:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _collect_spring_anchors(
        view: "ViewerData", element_ids: ndarray,
    ) -> "dict[int, tuple[int, ndarray]]":
        """``eid -> (node_i_id, node_i_position)`` for resolvable springs.

        Node i anchors the glyph (i == j for a zero-length element).
        Returning a dict keyed by eid (rather than a positions array
        positionally zipped against ``element_ids``) keeps the mapping
        correct when some spring's node is absent from the view, and
        hands ``sync_substrate_points`` the node id it needs to follow
        the deformed substrate.
        """
        eid_set = {int(e) for e in element_ids}
        node_ids_arr = np.asarray(list(view.nodes.ids), dtype=np.int64)
        coords_arr = np.asarray(view.nodes.coords, dtype=np.float64)
        if node_ids_arr.size == 0:
            return {}
        max_nid = int(node_ids_arr.max())
        nid_to_idx = np.full(max_nid + 2, -1, dtype=np.int64)
        nid_to_idx[node_ids_arr] = np.arange(
            node_ids_arr.size, dtype=np.int64,
        )
        out: dict[int, tuple[int, ndarray]] = {}
        for group in view.elements:
            if group.element_type.dim != 1:
                continue
            ids = np.asarray(group.ids, dtype=np.int64)
            conn = np.asarray(group.connectivity, dtype=np.int64)
            for k in range(len(group)):
                eid = int(ids[k])
                if eid not in eid_set:
                    continue
                nid_i = int(conn[k, 0])
                ii = nid_to_idx[nid_i]
                if ii < 0:
                    continue
                out[eid] = (nid_i, coords_arr[ii].copy())
        return out
