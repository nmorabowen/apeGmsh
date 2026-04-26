"""
ClippingController — Interactive section/clipping plane for the mesh viewer.

Uses a single ``vtkPlane`` attached to every per-dim mapper via
``AddClippingPlane``. Render-time clipping — no mesh recomputation when
the user drags the widget, so it stays interactive on large meshes.

The plane widget is the standard PyVista 3D handle; its callback only
mutates the shared plane's normal/origin and re-renders.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import vtk

if TYPE_CHECKING:
    from .entity_registry import EntityRegistry


class ClippingController:
    """Toggle a render-time clipping plane across all per-dim actors."""

    __slots__ = ("_plotter", "_registry", "_plane", "_enabled")

    def __init__(self, plotter, registry: "EntityRegistry") -> None:
        self._plotter = plotter
        self._registry = registry
        self._plane = vtk.vtkPlane()
        self._plane.SetNormal(1.0, 0.0, 0.0)
        self._plane.SetOrigin(0.0, 0.0, 0.0)
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> bool:
        """Flip on/off; return new state."""
        if self._enabled:
            self.disable()
        else:
            self.enable()
        return self._enabled

    def enable(self) -> None:
        if self._enabled:
            return
        for dim in self._registry.dims:
            for actor in (
                self._registry.dim_actors.get(dim),
                self._registry.dim_wire_actors.get(dim),
            ):
                if actor is None:
                    continue
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.AddClippingPlane(self._plane)

        try:
            self._plotter.add_plane_widget(
                callback=self._on_plane_changed,
                normal=(1.0, 0.0, 0.0),
                origin=(0.0, 0.0, 0.0),
                interaction_event="end",
                normal_rotation=True,
                origin_translation=True,
            )
        except Exception:
            pass
        self._enabled = True
        self._plotter.render()

    def disable(self) -> None:
        if not self._enabled:
            return
        for dim in self._registry.dims:
            for actor in (
                self._registry.dim_actors.get(dim),
                self._registry.dim_wire_actors.get(dim),
            ):
                if actor is None:
                    continue
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.RemoveAllClippingPlanes()

        try:
            self._plotter.clear_plane_widgets()
        except Exception:
            pass
        self._enabled = False
        self._plotter.render()

    def reset(self) -> None:
        """Recentre the plane at the model origin with +X normal."""
        self._plane.SetNormal(1.0, 0.0, 0.0)
        self._plane.SetOrigin(0.0, 0.0, 0.0)
        if self._enabled:
            try:
                self._plotter.clear_plane_widgets()
                self._plotter.add_plane_widget(
                    callback=self._on_plane_changed,
                    normal=(1.0, 0.0, 0.0),
                    origin=(0.0, 0.0, 0.0),
                    interaction_event="end",
                    normal_rotation=True,
                    origin_translation=True,
                )
            except Exception:
                pass
            self._plotter.render()

    def _on_plane_changed(self, normal, origin) -> None:
        self._plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
        self._plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
        self._plotter.render()
