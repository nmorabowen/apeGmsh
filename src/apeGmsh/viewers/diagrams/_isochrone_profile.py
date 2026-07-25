"""IsochroneProfileDiagram — a family of profiles, one curve per instant.

The classical "isochrone" plot: a response plotted against position
along a path, with one curve per time instant, so the whole evolution
of the profile reads in a single static chart. Terzaghi's consolidation
isochrones are the canonical example (excess pore pressure vs depth at
successive times); the same picture serves a soil column under shaking,
a story-drift profile over a building's height, or a wave travelling
down a pile.

The diagram's real product is 2-D, so it lives mostly in its
``make_side_panel`` chart (see
:class:`~apeGmsh.viewers.ui._isochrone_panel.IsochroneProfilePanel`).
Its 3-D presence is deliberately minimal — the sampled path drawn as a
polyline — because the one thing a chart cannot tell you is *which
nodes it came from*.

Path model (kept intentionally narrow):

* Nodes are ordered by one coordinate axis (``path_axis``, ``"auto"`` =
  the axis of largest extent), and that coordinate IS the position
  abscissa. This is exact for the profiles people actually draw — a
  soil column, a storey stack, a horizontal line of nodes.
* It is **not** arc length along an arbitrary curve. A selection that
  doubles back on the chosen axis isn't a function of position and will
  read as a zig-zag; the fix is a better selection, not a cleverer
  ordering, so the diagram doesn't try to guess one.

Render seam (ADR 0042): emits one line-cell :class:`MeshLayer` through
the backend and holds no VTK objects. The chart reads its data through
this diagram's :meth:`read_profile` so the panel never touches Results
internals — the same split the fiber-section panel uses.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec, NoDataError
from ._isochrone_math import AXIS_NAMES, dominant_axis, pick_step_indices
from ._kinds import register_diagram_kind
from ._styles import IsochroneProfileStyle
from ..scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


@register_diagram_kind(
    label="Isochrone profile (curve family)",
    style_class=IsochroneProfileStyle,
    order=14,
)
class IsochroneProfileDiagram(Diagram):
    """Sampled path in 3-D + a one-curve-per-instant chart in the plot pane."""

    kind = "isochrone_profile"
    topology = "nodes"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, IsochroneProfileStyle):
            raise TypeError(
                "IsochroneProfileDiagram requires an "
                f"IsochroneProfileStyle; got {type(spec.style).__name__}."
            )
        style = spec.style
        if style.path_axis not in ("auto",) + AXIS_NAMES:
            raise ValueError(
                f"IsochroneProfileStyle.path_axis must be 'auto' or one "
                f"of {AXIS_NAMES}; got {style.path_axis!r}."
            )
        if style.value_axis not in ("auto", "horizontal", "vertical"):
            raise ValueError(
                "IsochroneProfileStyle.value_axis must be 'auto', "
                f"'horizontal', or 'vertical'; got {style.value_axis!r}."
            )
        super().__init__(spec, results)

        self._handle: Any = None
        self._layer: Optional[MeshLayer] = None
        self._points: Optional[PointSet] = None
        self._cells: Optional[CellBlocks] = None
        # Path state, resolved once at attach and ordered along the axis.
        self._path_node_ids: Optional[ndarray] = None
        self._path_axis_index: int = 2
        self._substrate_rows: Optional[ndarray] = None

    # ------------------------------------------------------------------
    # Attach / update / detach
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "IsochroneProfileDiagram.attach requires a FEMSceneData "
                "(the viewer's substrate mesh). The Director must call "
                "bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, view, scene)

        rows, node_ids, axis = self._resolve_path(scene)
        self._substrate_rows = rows
        self._path_node_ids = node_ids
        self._path_axis_index = axis

        # A profile needs the component to actually be readable, and the
        # user finds out here rather than via an empty chart.
        self._assert_component_available()

        base = np.asarray(scene.grid.points, dtype=np.float64)[rows]
        self._points = PointSet(base)
        self._cells = self._polyline_cells(rows.size)
        self._layer = self._build_layer()
        self._handle = self._backend.add_layer(self._layer)

    def update_to_step(self, step_index: int) -> None:
        """No 3-D work — the path doesn't move with the cursor.

        The chart's current-step highlight is driven by the panel's own
        step subscription, so there is nothing to push here.
        """
        return None

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Re-sample the path polyline from the (deformed) substrate."""
        if (
            self._handle is None
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
        if rows.size == 0 or int(rows.max()) >= target.shape[0]:
            return
        self._points = PointSet(target[rows])
        self._push_update()

    def detach(self) -> None:
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._handle = None
        self._layer = None
        self._points = None
        self._cells = None
        self._path_node_ids = None
        self._substrate_rows = None
        super().detach()

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Side panel (the diagram's actual product)
    # ------------------------------------------------------------------

    def make_side_panel(self, director: Any) -> Any:
        if not self.is_attached:
            return None
        try:
            from ..ui._isochrone_panel import IsochroneProfilePanel
        except ImportError:
            # matplotlib / Qt absent — the 3-D path still renders.
            return None
        return IsochroneProfilePanel(self, director)

    # ------------------------------------------------------------------
    # Data access for the panel
    # ------------------------------------------------------------------

    @property
    def path_axis_name(self) -> str:
        """``"x"`` / ``"y"`` / ``"z"`` — the resolved ordering axis."""
        return AXIS_NAMES[self._path_axis_index]

    @property
    def path_node_ids(self) -> "ndarray | None":
        """The sampled nodes in path order, or ``None`` before attach."""
        return self._path_node_ids

    def value_on_horizontal(self) -> bool:
        """Whether the chart should put the response on the x-axis.

        ``value_axis="auto"`` resolves to True for a ``z`` path so a
        depth profile draws upright (depth vertical) in the geotechnical
        convention; every other axis keeps position horizontal.
        """
        style: IsochroneProfileStyle = self.spec.style  # type: ignore[assignment]
        if style.value_axis == "horizontal":
            return True
        if style.value_axis == "vertical":
            return False
        return self._path_axis_index == 2

    def read_profile(self) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
        """``(position, times, values)`` for the whole curve family.

        Returns
        -------
        ``position`` is ``(P,)`` — the path coordinate of each sampled
        node, ascending. ``times`` is ``(C,)`` — the drawn instants.
        ``values`` is ``(C, P)`` — the response at each instant along
        the path, aligned to ``position``. ``None`` when the read fails
        or yields nothing, which the panel renders as an empty state
        rather than raising into the Qt event loop.
        """
        if self._path_node_ids is None or self._view is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        style: IsochroneProfileStyle = self.spec.style  # type: ignore[assignment]
        time = self._stage_time_vector(
            self.spec.selector.component, int(self._path_node_ids[0]),
        )
        if time is None or time.size == 0:
            return None
        steps = pick_step_indices(int(time.size), int(style.n_curves))
        if steps.size == 0:
            return None
        try:
            slab = results.nodes.get(
                ids=self._path_node_ids,
                component=self.spec.selector.component,
                time=[int(s) for s in steps],
            )
        except Exception:
            return None
        values = np.asarray(slab.values, dtype=np.float64)
        if values.size == 0:
            return None

        # Re-order slab columns into path order.
        slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
        cols = self._align_columns(self._path_node_ids, slab_ids)
        keep = cols >= 0
        if not keep.any():
            return None
        position = self._path_positions()[keep]
        return (
            position,
            np.asarray(slab.time, dtype=np.float64),
            values[:, cols[keep]],
        )

    def read_profile_at_step(
        self, step_index: int,
    ) -> "Optional[tuple[ndarray, ndarray]]":
        """``(position, values)`` for one step — the live highlight curve.

        One narrow slab read (the path's nodes at a single step), which
        is what lets the panel show where "now" sits inside the drawn
        family as the user scrubs. ``None`` on any failure; the panel
        then just omits the highlight.
        """
        if self._path_node_ids is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        try:
            slab = results.nodes.get(
                ids=self._path_node_ids,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return None
        values = np.asarray(slab.values, dtype=np.float64)
        if values.size == 0:
            return None
        cols = self._align_columns(
            self._path_node_ids, np.asarray(slab.node_ids, dtype=np.int64),
        )
        keep = cols >= 0
        if not keep.any():
            return None
        return (self._path_positions()[keep], values[0, cols[keep]])

    def _path_positions(self) -> ndarray:
        """Path coordinate of each sampled node (undeformed reference).

        Read from the scene's undeformed baseline, not the live (possibly
        warped) grid: a profile's abscissa is a material position, so it
        must not stretch as the model deforms.
        """
        scene = getattr(self, "_scene", None)
        rows = self._substrate_rows
        if scene is None or rows is None:
            return np.zeros(0, dtype=np.float64)
        base = getattr(scene, "reference_points", None)
        if base is None:
            base = scene.grid.points
        pts = np.asarray(base, dtype=np.float64)
        return pts[rows, self._path_axis_index]

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(
        self, scene: "FEMSceneData",
    ) -> tuple[ndarray, ndarray, int]:
        """``(substrate_rows, node_ids, axis_index)`` ordered along the axis."""
        node_ids = self._resolved_node_ids
        if node_ids is None or node_ids.size == 0:
            raise NoDataError(
                "IsochroneProfileDiagram: a profile needs an explicit "
                "path — select a physical group, label, or node id set. "
                "'All nodes' has no meaningful ordering."
            )
        rows = self._fem_ids_to_substrate_rows(scene, node_ids)
        if rows.size < 2:
            raise NoDataError(
                f"IsochroneProfileDiagram: the selector resolved to "
                f"{rows.size} substrate node(s); a profile needs at "
                f"least 2 (selector={self.spec.selector!r})."
            )

        coords = np.asarray(scene.grid.points, dtype=np.float64)[rows]
        style: IsochroneProfileStyle = self.spec.style  # type: ignore[assignment]
        if style.path_axis == "auto":
            axis = dominant_axis(coords)
        else:
            axis = AXIS_NAMES.index(style.path_axis)

        # Sort by the path axis, breaking ties with the other two
        # coordinates. A line-like selection has no ties and the
        # secondary keys never matter; a selection several nodes wide
        # (a whole physical group, the easiest thing to pick) has many,
        # and ordering those by mesh id would walk the polyline back and
        # forth across the section at random. Lexicographic order at
        # least traverses each level coherently — and makes the path
        # depend only on geometry, not on node numbering.
        others = [i for i in range(3) if i != axis]
        order = np.lexsort((
            coords[:, others[1]], coords[:, others[0]], coords[:, axis],
        ))
        rows = rows[order]
        return (rows, scene.node_ids[rows], axis)

    def _assert_component_available(self) -> None:
        """Fail loud at attach when the component can't be read.

        Without this the diagram would attach fine, render a path, and
        hand the panel an empty chart — the "silently blank" failure the
        NoDataError contract exists to prevent.
        """
        component = self.spec.selector.component
        results = self._scoped_results()
        if results is None:
            raise NoDataError(
                "IsochroneProfileDiagram: could not scope Results to a "
                "stage — the diagram needs a resolvable stage."
            )
        try:
            slab = results.nodes.get(
                ids=self._path_node_ids, component=component, time=[0],
            )
        except Exception as exc:
            raise NoDataError(
                f"IsochroneProfileDiagram: could not read "
                f"{component!r}: {exc}"
            ) from exc
        if slab.values.size == 0:
            raise NoDataError(
                f"IsochroneProfileDiagram: no nodal data for component "
                f"{component!r} on the selected path. Use "
                f"`results.inspect.diagnose({component!r})` to see "
                f"which buckets were checked."
            )

    @staticmethod
    def _fem_ids_to_substrate_rows(
        scene: "FEMSceneData", fem_ids: ndarray,
    ) -> ndarray:
        """Substrate rows for ``fem_ids``, dropping ids not in the scene."""
        if fem_ids.size == 0 or scene.node_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        max_id = max(int(fem_ids.max()), int(scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(
            scene.node_ids.size, dtype=np.int64,
        )
        idx = lookup[fem_ids]
        return idx[idx >= 0]

    @staticmethod
    def _align_columns(fem_ids: ndarray, slab_ids: ndarray) -> ndarray:
        """For each ``fem_ids`` entry, its column in the slab (or -1)."""
        if slab_ids.size == 0:
            return np.full(fem_ids.size, -1, dtype=np.int64)
        max_id = int(max(int(fem_ids.max()), int(slab_ids.max()))) + 1
        col_of_id = np.full(max_id + 1, -1, dtype=np.int64)
        col_of_id[slab_ids] = np.arange(slab_ids.size, dtype=np.int64)
        safe = np.clip(fem_ids, 0, max_id)
        cols = col_of_id[safe]
        cols[(fem_ids < 0) | (fem_ids > max_id)] = -1
        return cols

    # ------------------------------------------------------------------
    # Layer build / emit
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"isochrone_profile_{id(self):x}"

    @staticmethod
    def _polyline_cells(n_points: int) -> CellBlocks:
        """Consecutive 2-node line cells through the ordered path."""
        if n_points < 2:
            return CellBlocks({})
        idx = np.arange(n_points - 1, dtype=np.int64)
        conn = np.column_stack((idx, idx + 1))
        return CellBlocks({"line": conn})

    def _build_layer(self) -> MeshLayer:
        style: IsochroneProfileStyle = self.spec.style  # type: ignore[assignment]
        assert self._points is not None and self._cells is not None
        return MeshLayer(
            layer_id=self._layer_id(),
            points=self._points,
            cells=self._cells,
            color=ColorSpec(mode="solid", solid_rgb=style.path_color),
            line_width=style.path_line_width,
            opacity=1.0 if style.show_path else 0.0,
            pickable=False,
        )

    def _push_update(self) -> None:
        if self._handle is None:
            return
        self._layer = self._build_layer()
        self._backend.update_layer(self._handle, self._layer)
