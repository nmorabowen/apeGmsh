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
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._styles import SpringForceStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
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

        self._source: Optional[pv.PolyData] = None
        self._actor: Any = None
        self._element_ids_to_read: tuple[int, ...] = ()
        self._spring_positions: Optional[ndarray] = None
        self._direction: Optional[ndarray] = None
        self._initial_scale: float = 1.0

        # Mapping from slab position -> spring index in our PolyData
        self._slab_to_spring_pos: Optional[ndarray] = None

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
            raise RuntimeError(
                "SpringForceDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, fem, scene)
        style: SpringForceStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (zero-length springs are 1-D elements) ─
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_zero_length_ids(fem)
        if element_ids.size == 0:
            from ._base import NoDataError
            raise NoDataError(
                f"SpringForceDiagram: no zero-length spring elements "
                f"found in the selection (selector={self.spec.selector!r})."
            )
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Spring world positions (use node i — i and j coincide) ──
        positions = self._collect_spring_positions(fem, element_ids)
        if positions.shape[0] == 0:
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
        eid_to_pos = {
            int(eid): pos for eid, pos in zip(element_ids, positions)
        }
        valid_mask = np.zeros(n_slab, dtype=bool)
        for k in range(n_slab):
            p = eid_to_pos.get(int(slab_eids[k]))
            if p is not None:
                positions_in_slab_order[k] = p
                valid_mask[k] = True

        if not valid_mask.any():
            return

        positions_in_slab_order = positions_in_slab_order[valid_mask]
        slab_values = slab_values[valid_mask]
        slab_values_all = (
            slab_values_all[:, valid_mask] if slab_values_all.ndim == 2
            else slab_values_all
        )
        n = positions_in_slab_order.shape[0]
        self._spring_positions = positions_in_slab_order
        self._slab_to_spring_pos = np.where(valid_mask)[0]

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

        # Build source PolyData with vector field
        source = pv.PolyData(positions_in_slab_order)
        source.point_data["_vec"] = np.tile(d, (n, 1))
        source.point_data["_mag"] = np.abs(slab_values)
        # Encode signed magnitude in vector length so negative values flip
        source.point_data["_vec"][:] = (
            slab_values[:, None] * d[None, :]
        )
        # Magnitude column stores absolute value for glyph scale
        source.point_data["_mag"] = np.abs(slab_values)
        self._source = source

        glyph = self._build_glyph(self.current_scale())

        actor = plotter.add_mesh(
            glyph,
            color=style.color,
            opacity=1.0,
            lighting=False,
            name=self._actor_name(),
            reset_camera=False,
            show_scalar_bar=False,
            # Decorative overlay — picks pass through to the substrate.
            pickable=False,
        )
        self._actor = actor
        self._actors = [actor]

    def update_to_step(self, step_index: int) -> None:
        if self._source is None or self._actor is None:
            return
        if self._direction is None:
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
        self._source.point_data["_vec"][:] = (
            ours[:, None] * self._direction[None, :]
        )
        self._source.point_data["_mag"][:] = np.abs(ours)
        try:
            self._source.Modified()
        except Exception:
            pass
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
        self._element_ids_to_read = ()
        self._spring_positions = None
        self._direction = None
        self._slab_to_spring_pos = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style
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

    def set_direction(
        self, direction: tuple[float, float, float],
    ) -> None:
        d = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            return
        self._direction = d / norm
        if self._source is None:
            return
        # Re-orient existing vectors using the latest magnitude column.
        mags = np.asarray(self._source.point_data["_mag"])
        # Recover signed values: we don't have the sign on hand; assume
        # positive (caller can re-step to refresh).
        self._source.point_data["_vec"][:] = mags[:, None] * self._direction[None, :]
        try:
            self._source.Modified()
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
        return f"diagram_spring_{id(self):x}"

    def _scoped_results(self) -> "Optional[Results]":
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        try:
            return self._results
        except Exception:
            return None

    def _build_glyph(self, scale: float) -> pv.PolyData:
        if self._source is None:
            return pv.PolyData()
        style: SpringForceStyle = self.spec.style    # type: ignore[assignment]
        try:
            return self._source.glyph(
                orient="_vec",
                scale="_mag",
                factor=float(scale),
                geom=pv.Arrow(tip_length=style.arrow_tip_fraction),
            )
        except Exception:
            return pv.PolyData(np.asarray(self._source.points))

    @staticmethod
    def _collect_zero_length_ids(fem: "FEMData") -> ndarray:
        """All 1-D element IDs (springs are 1-D, dim==1)."""
        ids: list[int] = []
        for group in fem.elements:
            if group.element_type.dim == 1:
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _collect_spring_positions(
        fem: "FEMData", element_ids: ndarray,
    ) -> ndarray:
        """Return ``(n_elements, 3)`` of node-i positions (i == j for ZL)."""
        eid_set = {int(e) for e in element_ids}
        node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
        coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
        if node_ids_arr.size == 0:
            return np.zeros((0, 3))
        max_nid = int(node_ids_arr.max())
        nid_to_idx = np.full(max_nid + 2, -1, dtype=np.int64)
        nid_to_idx[node_ids_arr] = np.arange(
            node_ids_arr.size, dtype=np.int64,
        )
        out_eids: list[int] = []
        out_pos: list[ndarray] = []
        for group in fem.elements:
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
                out_eids.append(eid)
                out_pos.append(coords_arr[ii].copy())

        # Reorder to match input order
        eid_to_pos = dict(zip(out_eids, out_pos))
        ordered = [
            eid_to_pos.get(int(eid))
            for eid in element_ids
        ]
        ordered = [p for p in ordered if p is not None]
        if not ordered:
            return np.zeros((0, 3))
        return np.vstack(ordered)
