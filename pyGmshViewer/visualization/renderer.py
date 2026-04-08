"""
Renderer — Manages the PyVista/VTK viewport and all visual actors.

Handles mesh display modes (surface, wireframe, edges), contour plots,
deformed shapes, colorbars, and camera control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import pyvista as pv

from pyGmshViewer.loaders.vtu_loader import MeshData, create_deformed_mesh


class DisplayMode(Enum):
    SURFACE = auto()
    WIREFRAME = auto()
    SURFACE_WITH_EDGES = auto()
    POINTS = auto()


# ── Color palettes for structural visualization ──────────────────────────
# Inspired by Abaqus/Paraview defaults for FEM work
COLORMAPS = {
    "Jet": "jet",
    "Viridis": "viridis",
    "Coolwarm": "coolwarm",
    "Turbo": "turbo",
    "RdBu": "RdBu",
    "Plasma": "plasma",
    "Spectral": "Spectral",
    "Rainbow": "rainbow",
}

DEFAULT_MESH_COLOR = "#5B8DB8"      # steel blue
DEFAULT_EDGE_COLOR = "#2C4A6E"      # dark navy
DEFAULT_BG_TOP = "#1a1a2e"          # dark gradient top
DEFAULT_BG_BOTTOM = "#16213e"       # dark gradient bottom
DEFORMED_COLOR = "#E05C00"          # orange for deformed overlay
UNDEFORMED_COLOR = "#AAAAAA"        # light grey for reference
LINE_MESH_COLOR = "#F5A623"         # amber for line-only meshes (frames, links)
LINE_MESH_WIDTH = 4                 # thicker default for line cells


# VTK cell type codes for "line-like" cells
_LINE_CELL_TYPES = {3, 4, 21}  # VTK_LINE, VTK_POLY_LINE, VTK_QUADRATIC_EDGE


def _is_line_only(mesh: pv.UnstructuredGrid) -> bool:
    """True if the mesh contains only 1-D line cells (no surfaces/volumes)."""
    try:
        ctypes = set(int(t) for t in mesh.celltypes)
    except Exception:
        return False
    return bool(ctypes) and ctypes.issubset(_LINE_CELL_TYPES)


@dataclass
class MeshActor:
    """Tracks a displayed mesh and its visualization state."""

    mesh_data: MeshData
    actor: Any = None
    display_mode: DisplayMode = DisplayMode.SURFACE_WITH_EDGES
    visible: bool = True
    opacity: float = 1.0
    color: str = DEFAULT_MESH_COLOR

    # Contour state
    active_scalar: str | None = None
    scalar_range: tuple[float, float] | None = None
    colormap: str = "jet"

    # Deformation state
    deformed_actor: Any = None
    deformed_mesh: pv.UnstructuredGrid | None = None
    scale_factor: float = 1.0
    show_deformed: bool = False
    show_undeformed_ref: bool = True

    # Line-only meshes (1D cells) need a thicker line width to be visible
    is_line_mesh: bool = False


class ViewportRenderer:
    """Manages the 3D viewport and all visual actors.

    This class wraps a PyVista BackgroundPlotter (or QtInteractor)
    and provides high-level methods for FEM visualization.
    """

    def __init__(self, plotter: pv.Plotter) -> None:
        self._plotter = plotter
        self._actors: dict[str, MeshActor] = {}
        self._setup_viewport()

    def _setup_viewport(self) -> None:
        """Configure default viewport appearance."""
        self._plotter.set_background(DEFAULT_BG_TOP, top=DEFAULT_BG_BOTTOM)
        self._plotter.add_axes(
            interactive=False,
            line_width=2,
            color="white",
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        self._plotter.enable_anti_aliasing("ssaa")

    # ── Mesh Management ──────────────────────────────────────────────────

    def add_mesh(self, mesh_data: MeshData, name: str | None = None) -> str:
        """Add a mesh to the viewport.

        Parameters
        ----------
        mesh_data : MeshData
            Loaded mesh data from the loader module.
        name : str, optional
            Display name. Defaults to mesh_data.name.

        Returns
        -------
        The name key used to reference this mesh actor.
        """
        name = name or mesh_data.name
        # Remove existing actor with same name
        if name in self._actors:
            self.remove_mesh(name)

        line_only = _is_line_only(mesh_data.mesh)
        mesh_color = LINE_MESH_COLOR if line_only else DEFAULT_MESH_COLOR
        actor = self._plotter.add_mesh(
            mesh_data.mesh,
            color=mesh_color,
            show_edges=not line_only,
            edge_color=DEFAULT_EDGE_COLOR,
            line_width=LINE_MESH_WIDTH if line_only else 1,
            opacity=1.0,
            label=name,
            reset_camera=True,
            lighting=not line_only,
            smooth_shading=not line_only,
            render_lines_as_tubes=line_only,
            name=name,
        )

        self._actors[name] = MeshActor(
            mesh_data=mesh_data,
            actor=actor,
            display_mode=DisplayMode.SURFACE if line_only else DisplayMode.SURFACE_WITH_EDGES,
            color=mesh_color,
            is_line_mesh=line_only,
        )

        self._plotter.reset_camera()
        return name

    def remove_mesh(self, name: str) -> None:
        """Remove a mesh from the viewport."""
        if name in self._actors:
            ma = self._actors[name]
            self._plotter.remove_actor(name)
            if ma.deformed_actor is not None:
                self._plotter.remove_actor(f"{name}_deformed")
            del self._actors[name]

    def clear_all(self) -> None:
        """Remove all meshes from the viewport."""
        for name in list(self._actors.keys()):
            self.remove_mesh(name)
        self._plotter.clear()
        self._setup_viewport()

    # ── Display Modes ────────────────────────────────────────────────────

    def set_display_mode(self, name: str, mode: DisplayMode) -> None:
        """Change how a mesh is rendered."""
        if name not in self._actors:
            return

        ma = self._actors[name]
        ma.display_mode = mode

        # Re-add with new style
        self._plotter.remove_actor(name)
        kwargs = self._display_kwargs(ma)
        ma.actor = self._plotter.add_mesh(
            ma.mesh_data.mesh, name=name, **kwargs
        )

    def _display_kwargs(self, ma: MeshActor) -> dict:
        """Build keyword args for plotter.add_mesh based on display state."""
        base = {
            "opacity": ma.opacity,
            "lighting": True,
            "smooth_shading": True,
            "reset_camera": False,
        }

        # Contour mode
        if ma.active_scalar:
            base["scalars"] = ma.active_scalar
            base["cmap"] = ma.colormap
            base["show_scalar_bar"] = True
            base["scalar_bar_args"] = {
                "title": ma.active_scalar,
                "color": "white",
                "title_font_size": 14,
                "label_font_size": 12,
                "n_labels": 7,
                "fmt": "%.3e",
                "position_x": 0.85,
                "position_y": 0.1,
                "width": 0.08,
                "height": 0.7,
            }
            if ma.scalar_range:
                base["clim"] = ma.scalar_range
        else:
            base["color"] = ma.color
            base["show_scalar_bar"] = False

        # Display mode
        mode = ma.display_mode
        if mode == DisplayMode.SURFACE:
            base["style"] = "surface"
            base["show_edges"] = False
        elif mode == DisplayMode.WIREFRAME:
            base["style"] = "wireframe"
            base["show_edges"] = False
            base["color"] = DEFAULT_EDGE_COLOR
        elif mode == DisplayMode.SURFACE_WITH_EDGES:
            base["style"] = "surface"
            base["show_edges"] = True
            base["edge_color"] = DEFAULT_EDGE_COLOR
            base["line_width"] = 1
        elif mode == DisplayMode.POINTS:
            base["style"] = "points"
            base["point_size"] = 5
            base["show_edges"] = False

        # Line-only meshes: always render as thick tubes so the 1D cells
        # (frame elements, rigid links) are actually visible.
        if ma.is_line_mesh:
            base["line_width"] = LINE_MESH_WIDTH
            base["render_lines_as_tubes"] = True
            base["lighting"] = False
            base["smooth_shading"] = False
            base["show_edges"] = False

        return base

    # ── Time-Step Navigation ───────────────────────────────────────────────

    def set_active_time_step(self, name: str, step: int) -> None:
        """Switch the displayed mesh data to a specific time step.

        Replaces the mesh actor's underlying grid with the
        ``step_meshes[step]`` grid from MeshData, preserving the
        current display mode, colormap, and contour settings.

        Parameters
        ----------
        name : str
            Mesh actor name.
        step : int
            Time step index into ``mesh_data.step_meshes``.
        """
        if name not in self._actors:
            return

        ma = self._actors[name]
        md = ma.mesh_data

        if not md.step_meshes or step >= len(md.step_meshes):
            return

        step_mesh = md.step_meshes[step]
        md.mesh = step_mesh
        md.point_field_names = list(step_mesh.point_data.keys())
        md.cell_field_names = list(step_mesh.cell_data.keys())

        # Re-render with same display state
        self._plotter.remove_actor(name)
        kwargs = self._display_kwargs(ma)
        ma.actor = self._plotter.add_mesh(step_mesh, name=name, **kwargs)

        # Refresh deformed overlay if active
        if ma.show_deformed:
            disp_field = None
            for f in md.point_field_names:
                fl = f.lower()
                if "disp" in fl or "modeshape" in fl or f == "Displacement":
                    disp_field = f
                    break
            if disp_field:
                self.show_deformed(
                    name,
                    displacement_field=disp_field,
                    scale_factor=ma.scale_factor,
                    time_step=step,
                    show_undeformed=ma.show_undeformed_ref,
                )

    # ── Contour Plots ────────────────────────────────────────────────────

    def set_scalar_field(
        self,
        name: str,
        field_name: str | None,
        colormap: str = "jet",
        clim: tuple[float, float] | None = None,
    ) -> None:
        """Apply a scalar contour plot to a mesh.

        Parameters
        ----------
        name : str
            Mesh actor name.
        field_name : str or None
            Scalar field name (from point_data or cell_data). None to clear.
        colormap : str
            Matplotlib colormap name.
        clim : tuple, optional
            (min, max) for color range. None for auto.
        """
        if name not in self._actors:
            return

        ma = self._actors[name]
        ma.active_scalar = field_name
        ma.colormap = colormap
        ma.scalar_range = clim

        if field_name:
            # Activate the scalar on the mesh object
            mesh = ma.mesh_data.mesh
            if field_name in mesh.point_data:
                mesh.set_active_scalars(field_name, preference="point")
            elif field_name in mesh.cell_data:
                mesh.set_active_scalars(field_name, preference="cell")

        # Rebuild the actor
        self._plotter.remove_actor(name)
        kwargs = self._display_kwargs(ma)
        ma.actor = self._plotter.add_mesh(
            ma.mesh_data.mesh, name=name, **kwargs
        )

    # ── Deformed Shape ───────────────────────────────────────────────────

    def show_deformed(
        self,
        name: str,
        displacement_field: str = "Displacement",
        scale_factor: float = 1.0,
        time_step: int | None = None,
        show_undeformed: bool = True,
    ) -> None:
        """Show deformed mesh overlay.

        Parameters
        ----------
        name : str
            Mesh actor name.
        displacement_field : str
            Name of the displacement vector field.
        scale_factor : float
            Amplification factor.
        time_step : int, optional
            Time step index for time-series data.
        show_undeformed : bool
            Whether to keep the undeformed mesh visible as reference.
        """
        if name not in self._actors:
            return

        ma = self._actors[name]
        ma.scale_factor = scale_factor
        ma.show_deformed = True
        ma.show_undeformed_ref = show_undeformed

        # Create deformed mesh
        ma.deformed_mesh = create_deformed_mesh(
            ma.mesh_data,
            displacement_field=displacement_field,
            scale_factor=scale_factor,
            time_step=time_step,
        )

        # Remove old deformed actor
        deformed_name = f"{name}_deformed"
        self._plotter.remove_actor(deformed_name)

        # Build kwargs for deformed mesh
        kwargs = {
            "opacity": ma.opacity,
            "lighting": not ma.is_line_mesh,
            "smooth_shading": not ma.is_line_mesh,
            "reset_camera": False,
            "show_edges": not ma.is_line_mesh,
            "edge_color": "#803D00",
            "line_width": LINE_MESH_WIDTH if ma.is_line_mesh else 1,
            "render_lines_as_tubes": ma.is_line_mesh,
            "name": deformed_name,
        }

        # If there's an active scalar, show it on the deformed mesh too
        if ma.active_scalar and ma.active_scalar in ma.deformed_mesh.point_data:
            kwargs["scalars"] = ma.active_scalar
            kwargs["cmap"] = ma.colormap
            kwargs["show_scalar_bar"] = False  # Only one scalar bar
            if ma.scalar_range:
                kwargs["clim"] = ma.scalar_range
        else:
            kwargs["color"] = DEFORMED_COLOR
            kwargs["show_scalar_bar"] = False

        ma.deformed_actor = self._plotter.add_mesh(
            ma.deformed_mesh, **kwargs
        )

        # Update undeformed reference appearance
        if show_undeformed:
            self._plotter.remove_actor(name)
            ma.actor = self._plotter.add_mesh(
                ma.mesh_data.mesh,
                color=UNDEFORMED_COLOR,
                style="wireframe",
                line_width=LINE_MESH_WIDTH if ma.is_line_mesh else 1,
                opacity=0.3,
                show_edges=False,
                render_lines_as_tubes=ma.is_line_mesh,
                name=name,
                reset_camera=False,
            )
        else:
            self._plotter.remove_actor(name)
            ma.actor = None

    def hide_deformed(self, name: str) -> None:
        """Remove the deformed overlay and restore original display."""
        if name not in self._actors:
            return

        ma = self._actors[name]
        ma.show_deformed = False

        deformed_name = f"{name}_deformed"
        self._plotter.remove_actor(deformed_name)
        ma.deformed_actor = None
        ma.deformed_mesh = None

        # Restore original mesh
        self._plotter.remove_actor(name)
        kwargs = self._display_kwargs(ma)
        ma.actor = self._plotter.add_mesh(
            ma.mesh_data.mesh, name=name, **kwargs
        )

    def update_scale_factor(self, name: str, scale_factor: float) -> None:
        """Update the deformation scale factor (live update)."""
        if name not in self._actors:
            return
        ma = self._actors[name]
        if not ma.show_deformed:
            return

        # Determine displacement field
        disp_field = None
        for f in ma.mesh_data.point_field_names:
            if "disp" in f.lower() or "modeshape" in f.lower() or f == "Displacement":
                disp_field = f
                break

        if disp_field:
            self.show_deformed(
                name,
                displacement_field=disp_field,
                scale_factor=scale_factor,
                show_undeformed=ma.show_undeformed_ref,
            )

    # ── Visibility / Opacity ─────────────────────────────────────────────

    def set_visibility(self, name: str, visible: bool) -> None:
        """Toggle mesh visibility."""
        if name not in self._actors:
            return
        ma = self._actors[name]
        ma.visible = visible
        if ma.actor:
            ma.actor.SetVisibility(visible)
        if ma.deformed_actor:
            ma.deformed_actor.SetVisibility(visible)

    def set_opacity(self, name: str, opacity: float) -> None:
        """Change mesh opacity (0.0 – 1.0)."""
        if name not in self._actors:
            return
        ma = self._actors[name]
        ma.opacity = opacity
        # Rebuild
        self._plotter.remove_actor(name)
        kwargs = self._display_kwargs(ma)
        ma.actor = self._plotter.add_mesh(
            ma.mesh_data.mesh, name=name, **kwargs
        )

    # ── Camera ───────────────────────────────────────────────────────────

    def reset_camera(self) -> None:
        self._plotter.reset_camera()

    def view_xy(self) -> None:
        self._plotter.view_xy()

    def view_xz(self) -> None:
        self._plotter.view_xz()

    def view_yz(self) -> None:
        self._plotter.view_yz()

    def view_isometric(self) -> None:
        self._plotter.view_isometric()

    # ── Node / Element picking ───────────────────────────────────────────

    def enable_point_picking(self, callback=None) -> None:
        """Enable interactive node picking."""
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

        def _on_pick(point):
            if callback:
                callback(point)

        self._plotter.enable_point_picking(
            callback=_on_pick,
            show_message=True,
            color="red",
            point_size=12,
            show_point=True,
            tolerance=0.025,
        )

    def enable_cell_picking(self, callback=None) -> None:
        """Enable interactive element picking."""
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

        def _on_pick(cell):
            if callback:
                callback(cell)

        self._plotter.enable_cell_picking(
            callback=_on_pick,
            show_message=True,
            color="red",
            show=True,
        )

    def disable_picking(self) -> None:
        """Disable all picking modes."""
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

    # ── Screenshot ───────────────────────────────────────────────────────

    def screenshot(self, filename: str = "screenshot.png") -> str:
        """Save a screenshot of the current view."""
        self._plotter.screenshot(filename)
        return filename

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def actor_names(self) -> list[str]:
        return list(self._actors.keys())

    def get_actor(self, name: str) -> MeshActor | None:
        return self._actors.get(name)

    # ------------------------------------------------------------------
    # Vector field glyphs
    # ------------------------------------------------------------------

    def show_vectors(
        self,
        name: str,
        vector_field: str,
        *,
        scale: float = 1.0,
        cmap: str = "viridis",
        time_step: int | None = None,
    ) -> None:
        """Show vector arrows (glyphs) at nodes.

        Parameters
        ----------
        name : str
            Name of the mesh to show vectors on.
        vector_field : str
            Name of the vector field in point_data.
        scale : float
            Arrow scale factor.
        cmap : str
            Colormap for arrow magnitude.
        time_step : int, optional
            Time step to use (for PVD series).
        """
        mesh_actor = self._actors.get(name)
        if mesh_actor is None:
            return
        self.hide_vectors(name)
        if time_step is not None and mesh_actor.mesh_data.step_meshes:
            source = mesh_actor.mesh_data.step_meshes[time_step]
        else:
            source = mesh_actor.mesh_data.mesh
        if vector_field not in source.point_data:
            return
        vec = np.asarray(source.point_data[vector_field])
        if vec.ndim == 1:
            vec = vec.reshape(-1, 1)
        if vec.shape[1] == 2:
            vec = np.column_stack([vec, np.zeros(vec.shape[0])])
        mag = np.linalg.norm(vec, axis=1)
        source["_vec_magnitude"] = mag
        source["_vec_field"] = vec
        arrows = source.glyph(
            orient="_vec_field",
            scale="_vec_magnitude",
            factor=scale,
        )
        actor = self._plotter.add_mesh(
            arrows, scalars="_vec_magnitude", cmap=cmap,
            name=f"_vectors_{name}",
        )
        mesh_actor._vector_actor = actor

    def hide_vectors(self, name: str) -> None:
        """Remove vector arrow glyphs."""
        mesh_actor = self._actors.get(name)
        if mesh_actor is None:
            return
        actor = getattr(mesh_actor, "_vector_actor", None)
        if actor is not None:
            self._plotter.remove_actor(actor)
            mesh_actor._vector_actor = None

    @property
    def plotter(self) -> pv.Plotter:
        return self._plotter