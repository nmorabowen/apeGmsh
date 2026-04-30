"""Probe overlay — point / line / plane probes for the results viewer.

Mined from ``apeGmshViewer/visualization/probes.py`` (frozen sibling
package, MIT, same author) and refactored to consume the integrated
viewer's ``FEMSceneData`` substrate and ``ResultsDirector`` instead
of raw VTU ``point_data`` dicts.

Result vocabulary
-----------------
Three frozen result records:

* :class:`PointProbeResult` — picked world position, nearest FEM node,
  and current-step values for the active diagrams' components.
* :class:`LineProbeResult` — N evenly-spaced sample points between
  two picked endpoints; each sample carries the same per-component
  values (looked up at its nearest node).
* :class:`PlaneProbeResult` — geometric slice of the substrate mesh.
  Phase 5b v1 returns the slice geometry only; coloring by a
  diagram's scalar lands in a follow-up.

Interactive entry points wrap PyVista's ``enable_point_picking`` —
clicks on the substrate mesh fire the corresponding callback. The
overlay maintains the marker actors and result history; ``clear()``
removes them all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pyvista as pv
from numpy import ndarray

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector
    from ..scene.fem_scene import FEMSceneData


# =====================================================================
# Result records
# =====================================================================

@dataclass(frozen=True)
class PointProbeResult:
    """Single-point sample of all active diagrams' components."""
    position: ndarray            # (3,) picked world coord
    closest_node_id: int         # FEM node ID
    closest_coord: ndarray       # (3,) snapped node coord
    distance: float              # Euclidean distance pick → snapped node
    step_index: int
    field_values: dict[str, float]   # component name -> scalar value

    def summary(self) -> str:
        head = (
            f"Point @ ({self.position[0]:.4g}, {self.position[1]:.4g}, "
            f"{self.position[2]:.4g})\n"
            f"  nearest node: {self.closest_node_id}  "
            f"d = {self.distance:.4g}\n"
            f"  step = {self.step_index}"
        )
        if not self.field_values:
            return head + "\n  (no components — add diagrams to populate)"
        rows = [f"  {name:24s} = {val:.6g}"
                for name, val in self.field_values.items()]
        return head + "\n" + "\n".join(rows)


@dataclass(frozen=True)
class LineProbeResult:
    """N samples along a line between two picked points."""
    point_a: ndarray             # (3,)
    point_b: ndarray             # (3,)
    n_samples: int
    arc_length: ndarray          # (N,) distance from A
    positions: ndarray           # (N, 3) sample positions
    closest_node_ids: ndarray    # (N,) per-sample nearest FEM node
    step_index: int
    field_values: dict[str, ndarray]   # component -> (N,) values
    sampled_mesh: Optional[pv.PolyData] = None

    @property
    def total_length(self) -> float:
        return float(np.linalg.norm(self.point_b - self.point_a))


@dataclass(frozen=True)
class PlaneProbeResult:
    """Slice of the substrate mesh by an arbitrary plane."""
    origin: ndarray              # (3,) point on plane
    normal: ndarray              # (3,) unit normal
    slice_mesh: pv.PolyData      # the cut geometry
    n_points: int


# =====================================================================
# Probe overlay
# =====================================================================

class ProbeMode(Enum):
    NONE = auto()
    POINT = auto()
    LINE_START = auto()
    LINE_END = auto()


class ProbeOverlay:
    """Owns probe markers + result history; one instance per viewer."""

    PROBE_COLOR = "#FF6B6B"
    LINE_COLOR = "#FFD166"
    SLICE_CMAP = "turbo"
    LINE_WIDTH = 3
    MARKER_FRACTION = 0.008      # of model diagonal

    def __init__(
        self,
        plotter: Any,
        scene: "FEMSceneData",
        director: "ResultsDirector",
    ) -> None:
        self._plotter = plotter
        self._scene = scene
        self._director = director

        # Make sure we have a node tree for nearest-node lookups
        scene.ensure_node_tree()

        # Result history (callers introspect, UI displays the latest)
        self.point_results: list[PointProbeResult] = []
        self.line_results: list[LineProbeResult] = []
        self.plane_results: list[PlaneProbeResult] = []

        # Visual marker bookkeeping (actor names — pyvista resolves)
        self._point_actor_names: list[str] = []
        self._label_actor_names: list[str] = []
        self._line_actor_name: Optional[str] = None
        self._slice_actor_name: Optional[str] = None

        # Interactive state
        self._mode = ProbeMode.NONE
        self._line_start: Optional[ndarray] = None

        # Callbacks (set by UI)
        self.on_point_result: Optional[Callable[[PointProbeResult], None]] = None
        self.on_line_result: Optional[Callable[[LineProbeResult], None]] = None
        self.on_plane_result: Optional[Callable[[PlaneProbeResult], None]] = None

    # ------------------------------------------------------------------
    # Programmatic probes (no Qt event loop required)
    # ------------------------------------------------------------------

    def probe_at_point(
        self, position: "ndarray | tuple[float, float, float]",
    ) -> PointProbeResult:
        """Sample at one world position; snap to nearest mesh node.

        Field values come from the Director — one per component used
        by any active diagram, evaluated at the current step.
        """
        pos = np.asarray(position, dtype=np.float64).ravel()[:3]
        node_id, snapped, distance = self._snap_to_nearest_node(pos)
        components = self._collect_active_components()
        values = self._director.read_at_pick(node_id, components)
        result = PointProbeResult(
            position=pos,
            closest_node_id=int(node_id),
            closest_coord=snapped,
            distance=distance,
            step_index=self._director.step_index,
            field_values=values,
        )
        self.point_results.append(result)
        self._add_point_marker(pos, f"P{len(self.point_results)}")
        return result

    def probe_along_line(
        self,
        point_a: "ndarray | tuple",
        point_b: "ndarray | tuple",
        *,
        n_samples: int = 50,
    ) -> LineProbeResult:
        """Sample fields along a straight line between two world points.

        For each of ``n_samples`` evenly-spaced sample points, snap
        to the nearest FEM node and pull current-step values from
        the active diagrams' components. Returns one ``(N,)`` array
        per component.
        """
        a = np.asarray(point_a, dtype=np.float64).ravel()[:3]
        b = np.asarray(point_b, dtype=np.float64).ravel()[:3]
        n = max(2, int(n_samples))
        ts = np.linspace(0.0, 1.0, n)
        positions = a[None, :] * (1.0 - ts[:, None]) + b[None, :] * ts[:, None]
        diffs = np.diff(positions, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        arc_length = np.zeros(n, dtype=np.float64)
        arc_length[1:] = np.cumsum(seg_lengths)

        node_ids = np.empty(n, dtype=np.int64)
        for k in range(n):
            nid, _, _ = self._snap_to_nearest_node(positions[k])
            node_ids[k] = nid

        components = self._collect_active_components()
        per_component: dict[str, ndarray] = {}
        for c in components:
            vals = np.zeros(n, dtype=np.float64)
            # Read once per unique node, then scatter
            unique, inv = np.unique(node_ids, return_inverse=True)
            for i, nid in enumerate(unique):
                got = self._director.read_at_pick(int(nid), [c])
                if c in got:
                    vals[inv == i] = got[c]
            per_component[c] = vals

        # Sampled-mesh stub for visualization (PyVista line — no fields)
        try:
            sampled = pv.Line(a, b, resolution=n - 1)
        except Exception:
            sampled = None

        result = LineProbeResult(
            point_a=a, point_b=b,
            n_samples=n,
            arc_length=arc_length,
            positions=positions,
            closest_node_ids=node_ids,
            step_index=self._director.step_index,
            field_values=per_component,
            sampled_mesh=sampled,
        )
        self.line_results.append(result)
        self._add_line_visual(a, b)
        return result

    def probe_with_plane(
        self,
        origin: "ndarray | tuple | None" = None,
        normal: "ndarray | tuple | str" = "x",
    ) -> Optional[PlaneProbeResult]:
        """Slice the substrate mesh; render the cut surface."""
        grid = self._scene.grid
        if grid.n_cells == 0:
            return None
        if origin is None:
            org = np.asarray(grid.center, dtype=np.float64)
        else:
            org = np.asarray(origin, dtype=np.float64).ravel()[:3]
        if isinstance(normal, str):
            mapping = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
            }
            n = mapping.get(normal.lower())
            if n is None:
                raise ValueError(f"Unknown normal {normal!r}")
        else:
            n = np.asarray(normal, dtype=np.float64).ravel()[:3]
            n_norm = float(np.linalg.norm(n))
            if n_norm < 1e-12:
                return None
            n = n / n_norm

        try:
            sliced = grid.slice(normal=tuple(n), origin=tuple(org))
        except Exception:
            return None
        if sliced is None or sliced.n_points == 0:
            return None

        result = PlaneProbeResult(
            origin=org,
            normal=n,
            slice_mesh=sliced,
            n_points=int(sliced.n_points),
        )
        self.plane_results.append(result)
        self._add_slice_actor(sliced)
        return result

    # ------------------------------------------------------------------
    # Interactive entry points (require a Qt event loop)
    # ------------------------------------------------------------------

    def start_point_probe(self) -> None:
        self._safe_disable_picking()
        self._mode = ProbeMode.POINT
        try:
            self._plotter.enable_point_picking(
                callback=self._on_interactive_point,
                show_message="Click to probe — picks the nearest mesh node",
                color=self.PROBE_COLOR,
                point_size=12,
                show_point=True,
                tolerance=0.025,
                pickable_window=False,
            )
        except Exception as exc:
            import sys
            print(
                f"[ProbeOverlay] enable_point_picking failed: {exc}",
                file=sys.stderr,
            )
            self._mode = ProbeMode.NONE

    def start_line_probe(self) -> None:
        self._safe_disable_picking()
        self._mode = ProbeMode.LINE_START
        self._line_start = None
        try:
            self._plotter.enable_point_picking(
                callback=self._on_interactive_line,
                show_message="Click point A for line probe",
                color=self.LINE_COLOR,
                point_size=12,
                show_point=True,
                tolerance=0.025,
                pickable_window=False,
            )
        except Exception as exc:
            import sys
            print(
                f"[ProbeOverlay] line probe init failed: {exc}",
                file=sys.stderr,
            )
            self._mode = ProbeMode.NONE

    def stop(self) -> None:
        """Cancel any active interactive probe."""
        self._safe_disable_picking()
        self._mode = ProbeMode.NONE
        self._line_start = None

    # ------------------------------------------------------------------
    # Interactive callbacks
    # ------------------------------------------------------------------

    def _on_interactive_point(self, point) -> None:
        if point is None:
            return
        result = self.probe_at_point(np.asarray(point))
        self._mode = ProbeMode.NONE
        self._safe_disable_picking()
        if self.on_point_result is not None:
            try:
                self.on_point_result(result)
            except Exception:
                pass

    def _on_interactive_line(self, point) -> None:
        if point is None:
            return
        pos = np.asarray(point)
        if self._mode == ProbeMode.LINE_START:
            self._line_start = pos
            self._mode = ProbeMode.LINE_END
            self._add_point_marker(pos, "A")
            self._safe_disable_picking()
            try:
                self._plotter.enable_point_picking(
                    callback=self._on_interactive_line,
                    show_message="Click point B to complete the line",
                    color=self.LINE_COLOR,
                    point_size=12,
                    show_point=True,
                    tolerance=0.025,
                    pickable_window=False,
                )
            except Exception:
                self._mode = ProbeMode.NONE
        elif self._mode == ProbeMode.LINE_END:
            if self._line_start is None:
                self._mode = ProbeMode.NONE
                return
            self._add_point_marker(pos, "B")
            result = self.probe_along_line(self._line_start, pos)
            self._mode = ProbeMode.NONE
            self._line_start = None
            self._safe_disable_picking()
            if self.on_line_result is not None:
                try:
                    self.on_line_result(result)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all probe markers + clear result history."""
        for name in self._point_actor_names + self._label_actor_names:
            try:
                self._plotter.remove_actor(name)
            except Exception:
                pass
        if self._line_actor_name is not None:
            try:
                self._plotter.remove_actor(self._line_actor_name)
            except Exception:
                pass
        if self._slice_actor_name is not None:
            try:
                self._plotter.remove_actor(self._slice_actor_name)
            except Exception:
                pass
        self._point_actor_names.clear()
        self._label_actor_names.clear()
        self._line_actor_name = None
        self._slice_actor_name = None
        self.point_results.clear()
        self.line_results.clear()
        self.plane_results.clear()

    # ------------------------------------------------------------------
    # Internal — visualization
    # ------------------------------------------------------------------

    def _add_point_marker(self, pos: ndarray, label: str) -> None:
        radius = max(self._scene.model_diagonal * self.MARKER_FRACTION, 1e-6)
        sphere = pv.Sphere(radius=radius, center=pos)
        actor_name = f"_probe_pt_{len(self._point_actor_names)}"
        try:
            self._plotter.add_mesh(
                sphere, color=self.PROBE_COLOR, name=actor_name,
                reset_camera=False, lighting=False,
            )
            self._point_actor_names.append(actor_name)
        except Exception:
            return

        label_name = f"_probe_label_{len(self._label_actor_names)}"
        try:
            self._plotter.add_point_labels(
                pv.PolyData(pos.reshape(1, 3)),
                [label],
                name=label_name,
                point_size=0,
                font_size=14,
                text_color=self.PROBE_COLOR,
                shape=None,
                reset_camera=False,
                always_visible=True,
            )
            self._label_actor_names.append(label_name)
        except Exception:
            pass

    def _add_line_visual(self, a: ndarray, b: ndarray) -> None:
        if self._line_actor_name is not None:
            try:
                self._plotter.remove_actor(self._line_actor_name)
            except Exception:
                pass
        line = pv.Line(a, b)
        try:
            self._plotter.add_mesh(
                line, color=self.LINE_COLOR,
                line_width=self.LINE_WIDTH,
                name="_probe_line",
                reset_camera=False, render_lines_as_tubes=True,
            )
            self._line_actor_name = "_probe_line"
        except Exception:
            pass

    def _add_slice_actor(self, sliced: pv.PolyData) -> None:
        if self._slice_actor_name is not None:
            try:
                self._plotter.remove_actor(self._slice_actor_name)
            except Exception:
                pass
        try:
            self._plotter.add_mesh(
                sliced, name="_probe_slice",
                color="lightblue", opacity=0.6,
                show_edges=True, edge_color="#4682B4",
                reset_camera=False, lighting=False,
            )
            self._slice_actor_name = "_probe_slice"
        except Exception:
            pass

    def _safe_disable_picking(self) -> None:
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal — data
    # ------------------------------------------------------------------

    def _snap_to_nearest_node(
        self, world: ndarray,
    ) -> tuple[int, ndarray, float]:
        """Return ``(fem_node_id, snapped_coord, distance)``."""
        scene = self._scene
        coords = np.asarray(scene.grid.points)
        node_tree = scene.ensure_node_tree()
        if node_tree is not None:
            distance, idx = node_tree.query(world)
            idx = int(idx)
        else:
            diffs = coords - world[None, :]
            idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
            distance = float(np.linalg.norm(coords[idx] - world))
        snapped = coords[idx]
        fem_id = int(scene.node_ids[idx])
        return fem_id, snapped, float(distance)

    def _collect_active_components(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for d in self._director.registry.diagrams():
            if not d.is_attached:
                continue
            comp = d.spec.selector.component
            if comp not in seen:
                seen.add(comp)
                out.append(comp)
        return out
