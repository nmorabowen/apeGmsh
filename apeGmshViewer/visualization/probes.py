"""
Probes — Interactive sampling tools for FEM field inspection.

Three probe types:
    1. Point Probe  — click on the mesh to sample all field values at that location
    2. Line Probe   — define two endpoints, sample fields along the line (N samples)
    3. Plane Probe  — slice the mesh with an arbitrary plane, show field on the cut

Each probe returns a ProbeResult that can be displayed in the properties panel
or plotted as a chart (line probe -> XY plot, plane probe -> contour on slice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import pyvista as pv

from apeGmsh.viewers.ui.theme import RED, YELLOW
from apeGmshViewer.loaders.vtu_loader import MeshData


# ─────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PointProbeResult:
    """Result of sampling fields at a single point."""

    position: np.ndarray                 # (3,) probe location
    closest_point: np.ndarray            # (3,) nearest mesh point
    closest_point_id: int                # node index in the mesh
    distance: float                      # distance from probe to closest node
    field_values: dict[str, Any]         # field_name -> scalar or vector value

    def summary(self) -> str:
        lines = [
            f"Point Probe at ({self.position[0]:.4f}, "
            f"{self.position[1]:.4f}, {self.position[2]:.4f})",
            f"Nearest node: {self.closest_point_id} "
            f"(d = {self.distance:.4e})",
        ]
        for name, val in self.field_values.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1:
                mag = np.linalg.norm(val)
                lines.append(f"  {name}: {val}  (|v| = {mag:.6e})")
            else:
                lines.append(f"  {name}: {val:.6e}")
        return "\n".join(lines)


@dataclass
class LineProbeResult:
    """Result of sampling fields along a line."""

    point_a: np.ndarray                  # (3,) start point
    point_b: np.ndarray                  # (3,) end point
    n_samples: int
    arc_length: np.ndarray               # (n_samples,) distance from start
    positions: np.ndarray                # (n_samples, 3) sampled positions
    field_values: dict[str, np.ndarray]  # field_name -> (n_samples,) or (n_samples, 3)
    sampled_mesh: pv.PolyData | None = None  # the pyvista sampled line

    @property
    def total_length(self) -> float:
        return float(np.linalg.norm(self.point_b - self.point_a))


@dataclass
class PlaneProbeResult:
    """Result of slicing the mesh with a plane."""

    origin: np.ndarray                   # (3,) plane origin
    normal: np.ndarray                   # (3,) plane normal
    slice_mesh: pv.PolyData              # the sliced geometry
    field_names: list[str]               # available fields on the slice


class ProbeMode(Enum):
    NONE = auto()
    POINT = auto()
    LINE_START = auto()        # waiting for first click
    LINE_END = auto()          # waiting for second click
    PLANE = auto()


# ─────────────────────────────────────────────────────────────────────────
# Probe engine
# ─────────────────────────────────────────────────────────────────────────

class ProbeEngine:
    """Manages probe interactions on the VTK viewport.

    Works with the ViewportRenderer's plotter to add interactive probes
    and display results as overlay actors.
    """

    # Visual constants
    PROBE_COLOR = RED                # pink/red for probe markers
    LINE_COLOR = YELLOW              # yellow for probe line
    SLICE_COLOR = "jet"              # colormap for plane probe
    MARKER_SIZE = 14
    LINE_WIDTH = 3

    def __init__(self, plotter: pv.Plotter):
        self._plotter = plotter
        self._mode = ProbeMode.NONE
        self._active_mesh: pv.UnstructuredGrid | None = None
        self._active_mesh_name: str | None = None

        # Probe state
        self._point_actors: list[str] = []
        self._line_actor: str | None = None
        self._slice_actor: str | None = None
        self._label_actors: list[str] = []
        self._chart_actor: str | None = None

        # Line probe: store first point while waiting for second
        self._line_start: np.ndarray | None = None

        # Callbacks for the UI
        self._on_point_result = None
        self._on_line_result = None
        self._on_plane_result = None

        # History of probe results
        self.point_results: list[PointProbeResult] = []
        self.line_results: list[LineProbeResult] = []
        self.plane_results: list[PlaneProbeResult] = []

    # ── Configuration ────────────────────────────────────────────────

    def set_active_mesh(self, mesh: pv.UnstructuredGrid, name: str) -> None:
        """Set which mesh the probes operate on."""
        self._active_mesh = mesh
        self._active_mesh_name = name

    def set_callbacks(
        self,
        on_point=None,
        on_line=None,
        on_plane=None,
    ) -> None:
        """Register callbacks for probe results."""
        self._on_point_result = on_point
        self._on_line_result = on_line
        self._on_plane_result = on_plane

    # ── Point Probe ──────────────────────────────────────────────────

    def _safe_disable_picking(self) -> None:
        """Disable any active picker before enabling a new one."""
        try:
            self._plotter.disable_picking()
        except Exception:
            pass
        self._set_nav_picking(False)

    def _set_nav_picking(self, active: bool) -> None:
        """Tell the navigation layer whether LMB picking is active."""
        fn = getattr(self._plotter, '_nav_set_picking', None)
        if fn is not None:
            fn(active)

    def start_point_probe(self) -> None:
        """Enter point probe mode — next click samples all fields."""
        self._safe_disable_picking()
        self._mode = ProbeMode.POINT
        self._set_nav_picking(True)
        self._plotter.enable_point_picking(
            callback=self._on_point_picked,
            show_message="Click on mesh to probe field values",
            color=self.PROBE_COLOR,
            point_size=self.MARKER_SIZE,
            show_point=True,
            tolerance=0.025,
            pickable_window=False,
        )

    def probe_at_point(self, position: np.ndarray) -> PointProbeResult | None:
        """Programmatically probe at a specific world coordinate.

        Uses VTK's probe filter to interpolate field values at the
        given position (doesn't require the point to be on a node).
        """
        if self._active_mesh is None:
            return None

        pos = np.asarray(position, dtype=np.float64).ravel()[:3]
        mesh = self._active_mesh

        # Create a single-point dataset at the probe location
        probe_point = pv.PolyData(pos.reshape(1, 3))
        # Sample the mesh at this point (interpolates fields)
        sampled = probe_point.sample(mesh)

        # Also find the closest actual mesh node
        closest_id = mesh.find_closest_point(pos)
        closest_pt = np.array(mesh.points[closest_id])
        distance = float(np.linalg.norm(pos - closest_pt))

        # Extract all field values at the sampled point
        field_values = {}
        for name in mesh.point_data:
            arr = sampled.point_data.get(name)
            if arr is not None and len(arr) > 0:
                val = arr[0]
                field_values[name] = val

        for name in mesh.cell_data:
            arr = sampled.cell_data.get(name)
            if arr is not None and len(arr) > 0:
                val = arr[0]
                field_values[f"{name} (cell)"] = val

        result = PointProbeResult(
            position=pos,
            closest_point=closest_pt,
            closest_point_id=closest_id,
            distance=distance,
            field_values=field_values,
        )
        self.point_results.append(result)

        # Add visual marker
        self._add_point_marker(pos, f"P{len(self.point_results)}")

        return result

    def _on_point_picked(self, point) -> None:
        """Internal callback for interactive point picking."""
        if point is None:
            return
        pos = np.array(point)
        result = self.probe_at_point(pos)
        if result and self._on_point_result:
            self._on_point_result(result)

    def _add_point_marker(self, pos: np.ndarray, label: str) -> None:
        """Add a visual marker sphere at the probe location."""
        sphere = pv.Sphere(radius=0.02, center=pos)
        # Use auto-scaling based on mesh bounds
        if self._active_mesh is not None:
            bounds = self._active_mesh.bounds
            diag = np.sqrt(
                (bounds[1] - bounds[0])**2 +
                (bounds[3] - bounds[2])**2 +
                (bounds[5] - bounds[4])**2
            )
            sphere = pv.Sphere(radius=diag * 0.008, center=pos)

        actor_name = f"_probe_pt_{len(self._point_actors)}"
        self._plotter.add_mesh(
            sphere,
            color=self.PROBE_COLOR,
            name=actor_name,
            reset_camera=False,
            lighting=False,
        )
        self._point_actors.append(actor_name)

        # Add text label
        label_name = f"_probe_label_{len(self._label_actors)}"
        self._plotter.add_point_labels(
            pv.PolyData(pos.reshape(1, 3)),
            [label],
            name=label_name,
            point_size=0,
            font_size=14,
            text_color=self.PROBE_COLOR,
            shape=None,
            render_points_as_spheres=False,
            reset_camera=False,
            always_visible=True,
        )
        self._label_actors.append(label_name)

    # ── Line Probe ───────────────────────────────────────────────────

    def start_line_probe(self) -> None:
        """Enter line probe mode — two clicks define the sampling line."""
        self._safe_disable_picking()
        self._mode = ProbeMode.LINE_START
        self._line_start = None
        self._set_nav_picking(True)
        self._plotter.enable_point_picking(
            callback=self._on_line_point_picked,
            show_message="Click FIRST point for line probe",
            color="#f9e2af",
            point_size=self.MARKER_SIZE,
            show_point=True,
            tolerance=0.025,
            pickable_window=False,
        )

    def probe_along_line(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        n_samples: int = 100,
    ) -> LineProbeResult | None:
        """Sample fields along a line between two points.

        Uses VTK's sample-over-line filter to interpolate field values
        at evenly-spaced points along the line.
        """
        if self._active_mesh is None:
            return None

        a = np.asarray(point_a, dtype=np.float64).ravel()[:3]
        b = np.asarray(point_b, dtype=np.float64).ravel()[:3]

        # Create the sampling line
        line = pv.Line(a, b, resolution=n_samples - 1)
        sampled = line.sample(self._active_mesh)

        # Extract positions and arc length
        positions = np.array(sampled.points)
        diffs = np.diff(positions, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        arc_length = np.zeros(len(positions))
        arc_length[1:] = np.cumsum(seg_lengths)

        # Extract all field values along the line
        field_values = {}
        for name in sampled.point_data:
            arr = np.array(sampled.point_data[name])
            field_values[name] = arr

        result = LineProbeResult(
            point_a=a,
            point_b=b,
            n_samples=len(positions),
            arc_length=arc_length,
            positions=positions,
            field_values=field_values,
            sampled_mesh=sampled,
        )
        self.line_results.append(result)

        # Visualize the probe line
        self._add_line_visual(a, b, sampled)

        return result

    def _on_line_point_picked(self, point) -> None:
        """Handle clicks during line probe definition."""
        if point is None:
            return

        pos = np.array(point)

        if self._mode == ProbeMode.LINE_START:
            self._line_start = pos
            self._mode = ProbeMode.LINE_END
            self._add_point_marker(pos, "A")
            # Disable current picker before re-enabling for second point
            self._safe_disable_picking()
            self._set_nav_picking(True)
            self._plotter.enable_point_picking(
                callback=self._on_line_point_picked,
                show_message="Click SECOND point for line probe",
                color="#f9e2af",
                point_size=self.MARKER_SIZE,
                show_point=True,
                tolerance=0.025,
                pickable_window=False,
            )
        elif self._mode == ProbeMode.LINE_END:
            self._add_point_marker(pos, "B")
            result = self.probe_along_line(self._line_start, pos)
            self._mode = ProbeMode.NONE
            self._set_nav_picking(False)
            self._plotter.disable_picking()
            if result and self._on_line_result:
                self._on_line_result(result)

    def _add_line_visual(
        self,
        a: np.ndarray,
        b: np.ndarray,
        sampled: pv.PolyData,
    ) -> None:
        """Draw the probe line on the viewport."""
        # Remove old line
        if self._line_actor:
            self._plotter.remove_actor(self._line_actor)

        line_name = "_probe_line"

        # If sampled data has an active scalar, color the line by it
        if sampled.point_data and len(sampled.point_data) > 0:
            first_field = list(sampled.point_data.keys())[0]
            self._plotter.add_mesh(
                sampled,
                scalars=first_field,
                cmap="turbo",
                line_width=self.LINE_WIDTH + 1,
                name=line_name,
                show_scalar_bar=False,
                reset_camera=False,
                render_lines_as_tubes=True,
            )
        else:
            line = pv.Line(a, b)
            self._plotter.add_mesh(
                line,
                color=self.LINE_COLOR,
                line_width=self.LINE_WIDTH,
                name=line_name,
                reset_camera=False,
                render_lines_as_tubes=True,
            )
        self._line_actor = line_name

    # ── Plane Probe (Slice) ──────────────────────────────────────────

    def probe_with_plane(
        self,
        origin: np.ndarray | None = None,
        normal: np.ndarray | str = "x",
    ) -> PlaneProbeResult | None:
        """Slice the mesh with a plane and return the cut surface.

        Parameters
        ----------
        origin : array-like (3,), optional
            Point on the plane. Defaults to mesh center.
        normal : array-like (3,) or str
            Plane normal vector, or one of "x", "y", "z".
        """
        if self._active_mesh is None:
            return None

        mesh = self._active_mesh

        # Default origin = mesh center
        if origin is None:
            origin = np.array(mesh.center)
        else:
            origin = np.asarray(origin, dtype=np.float64).ravel()[:3]

        # Parse normal
        if isinstance(normal, str):
            normal_map = {
                "x": np.array([1, 0, 0]),
                "y": np.array([0, 1, 0]),
                "z": np.array([0, 0, 1]),
            }
            normal_vec = normal_map.get(normal.lower(), np.array([1, 0, 0]))
        else:
            normal_vec = np.asarray(normal, dtype=np.float64).ravel()[:3]
            normal_vec = normal_vec / np.linalg.norm(normal_vec)

        # Perform the slice
        sliced = mesh.slice(normal=normal_vec, origin=origin)

        if sliced.n_points == 0:
            return None

        field_names = list(sliced.point_data.keys()) + list(sliced.cell_data.keys())

        result = PlaneProbeResult(
            origin=origin,
            normal=normal_vec,
            slice_mesh=sliced,
            field_names=field_names,
        )
        self.plane_results.append(result)

        # Visualize the slice
        self._add_slice_visual(sliced)

        return result

    def start_interactive_plane(self) -> None:
        """Start an interactive plane widget for slicing.

        The user can drag the plane origin and rotate the normal.
        """
        if self._active_mesh is None:
            return

        self._safe_disable_picking()
        self._mode = ProbeMode.PLANE
        self._set_nav_picking(True)

        def _on_plane_moved(normal, origin):
            result = self.probe_with_plane(origin=origin, normal=normal)
            if result and self._on_plane_result:
                self._on_plane_result(result)

        self._plotter.add_plane_widget(
            _on_plane_moved,
            normal="x",
            origin=self._active_mesh.center,
            bounds=self._active_mesh.bounds,
            factor=1.2,
            color=self.PROBE_COLOR,
            tubing=True,
            outline_translation=True,
            origin_translation=True,
            implicit=False,
        )

    def _add_slice_visual(self, sliced: pv.PolyData) -> None:
        """Add the slice surface to the viewport."""
        if self._slice_actor:
            self._plotter.remove_actor(self._slice_actor)

        slice_name = "_probe_slice"

        # Color by the first available scalar
        kwargs = {
            "name": slice_name,
            "reset_camera": False,
            "show_edges": True,
            "edge_color": "#585b70",
            "line_width": 1,
            "lighting": True,
            "opacity": 0.95,
        }

        if sliced.point_data and len(sliced.point_data) > 0:
            first_field = list(sliced.point_data.keys())[0]
            kwargs["scalars"] = first_field
            kwargs["cmap"] = "turbo"
            kwargs["show_scalar_bar"] = True
            kwargs["scalar_bar_args"] = {
                "title": f"Slice: {first_field}",
                "color": "white",
                "title_font_size": 12,
                "label_font_size": 10,
                "n_labels": 5,
                "fmt": "%.3e",
                "position_x": 0.05,
                "position_y": 0.1,
                "width": 0.08,
                "height": 0.5,
            }
        else:
            kwargs["color"] = self.PROBE_COLOR
            kwargs["show_scalar_bar"] = False

        self._plotter.add_mesh(sliced, **kwargs)
        self._slice_actor = slice_name

    # ── Cleanup ──────────────────────────────────────────────────────

    def clear_point_probes(self) -> None:
        """Remove all point probe markers."""
        for name in self._point_actors:
            self._plotter.remove_actor(name)
        for name in self._label_actors:
            self._plotter.remove_actor(name)
        self._point_actors.clear()
        self._label_actors.clear()
        self.point_results.clear()

    def clear_line_probe(self) -> None:
        """Remove the line probe visual."""
        if self._line_actor:
            self._plotter.remove_actor(self._line_actor)
            self._line_actor = None
        self.line_results.clear()

    def clear_plane_probe(self) -> None:
        """Remove the plane probe and widget."""
        if self._slice_actor:
            self._plotter.remove_actor(self._slice_actor)
            self._slice_actor = None
        try:
            self._plotter.clear_plane_widgets()
        except Exception:
            pass
        self.plane_results.clear()

    def clear_all(self) -> None:
        """Remove all probe visuals and reset state."""
        self.clear_point_probes()
        self.clear_line_probe()
        self.clear_plane_probe()
        self._mode = ProbeMode.NONE
        self._set_nav_picking(False)

    def stop(self) -> None:
        """Deactivate current probe mode."""
        self._mode = ProbeMode.NONE
        self._set_nav_picking(False)
        try:
            self._plotter.disable_picking()
        except Exception:
            pass
        try:
            self._plotter.clear_plane_widgets()
        except Exception:
            pass

    @property
    def mode(self) -> ProbeMode:
        return self._mode
