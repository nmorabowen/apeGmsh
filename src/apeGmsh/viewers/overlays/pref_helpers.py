"""
pref_helpers — Shared preference callback factories.
=====================================================

Functions that create preference callbacks for any viewer that has
an ``EntityRegistry`` and a ``plotter``.  Both ModelViewer and
MeshViewer call these instead of duplicating the logic.

Each factory returns a plain callable ``(value) -> None`` suitable
for passing directly to ``PreferencesTab(on_line_width=..., ...)``.
"""

from __future__ import annotations

from typing import Any, Callable


def make_line_width_cb(
    registry: Any,
    plotter: Any,
) -> Callable[[float], None]:
    """Line width callback: updates dim=1 actors + stored kwargs."""
    def _cb(v: float) -> None:
        for dim, actor in registry.dim_actors.items():
            if dim == 1:
                actor.GetProperty().SetLineWidth(v)
                kw = registry._add_mesh_kwargs.get(dim, {})
                kw['line_width'] = v
                registry._add_mesh_kwargs[dim] = kw
        plotter.render()
    return _cb


def make_opacity_cb(
    registry: Any,
    plotter: Any,
) -> Callable[[float], None]:
    """Surface opacity callback: updates dim>=2 actors + stored kwargs."""
    def _cb(v: float) -> None:
        for dim, actor in registry.dim_actors.items():
            if dim >= 2:
                actor.GetProperty().SetOpacity(v)
                kw = registry._add_mesh_kwargs.get(dim, {})
                kw['opacity'] = v
                registry._add_mesh_kwargs[dim] = kw
        plotter.render()
    return _cb


def make_edges_cb(
    registry: Any,
    plotter: Any,
) -> Callable[[bool], None]:
    """Edge visibility callback: updates dim>=2 actors + stored kwargs."""
    def _cb(show: bool) -> None:
        for dim, actor in registry.dim_actors.items():
            if dim >= 2:
                actor.GetProperty().SetEdgeVisibility(show)
                kw = registry._add_mesh_kwargs.get(dim, {})
                kw['show_edges'] = show
                registry._add_mesh_kwargs[dim] = kw
        plotter.render()
    return _cb
