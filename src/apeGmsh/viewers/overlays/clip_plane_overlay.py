"""
ClipPlaneOverlay — non-destructive section view via VTK clipping planes.

Attaches a single ``vtkPlane`` to the mappers of every BRep dim actor in
the registry.  VTK's clipping discards rasterized fragments on the
positive side of the plane (``n . (p - origin) > 0``); flipping the
normal reverses which half is hidden.  Geometry is untouched — Gmsh
state and tags are unchanged, so closing the section view restores the
full model.

This is *visual only*.  For an actual cut that produces new entities,
use ``g.model.geometry.add_cutting_plane`` + ``slice``.

Coordinate system
-----------------
The viewer renders actors in *shifted* coordinates
(``world - origin_shift``).  The panel exposes a *world*-coordinate
position; this overlay converts on every update.

Lifecycle
---------
After ``_rebuild_scene`` (e.g. parts fuse) the registry's actors are
replaced.  Call :meth:`rebind` so the plane reattaches to the fresh
mappers.  If the registry's ``origin_shift`` also changed, call
:meth:`set_origin_shift` first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..core.entity_registry import EntityRegistry


_AXIS_NORMALS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


class ClipPlaneOverlay:
    """Single axis-aligned clipping plane for the BRep viewer.

    Parameters
    ----------
    plotter
        PyVista QtInteractor — used only for ``render()``.
    registry
        :class:`EntityRegistry` whose ``dim_actors`` we clip.
    origin_shift
        The registry's coordinate shift ``(sx, sy, sz)``.
    """

    def __init__(
        self,
        plotter,
        registry: "EntityRegistry",
        origin_shift: Sequence[float],
    ) -> None:
        from vtkmodules.vtkCommonDataModel import vtkPlane

        self._plotter = plotter
        self._registry = registry
        self._origin_shift = list(origin_shift)
        self._plane = vtkPlane()

        self._enabled = False
        self._axis: str = "X"
        self._world_position: float = 0.0
        self._flipped: bool = False
        self._attached_mappers: list = []

        self._update_plane()

    # ------------------------------------------------------------------
    # Public API — the panel calls these
    # ------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._detach()
        if enabled:
            self._attach()
        self._plotter.render()

    def set_axis(self, axis: str) -> None:
        if axis not in _AXIS_NORMALS:
            return
        self._axis = axis
        self._update_plane()
        self._plotter.render()

    def set_world_position(self, world_pos: float) -> None:
        self._world_position = float(world_pos)
        self._update_plane()
        self._plotter.render()

    def set_flipped(self, flipped: bool) -> None:
        self._flipped = bool(flipped)
        self._update_plane()
        self._plotter.render()

    def set_origin_shift(self, origin_shift: Sequence[float]) -> None:
        """Track a new registry origin shift (after scene rebuild)."""
        self._origin_shift = list(origin_shift)
        self._update_plane()

    def rebind(self) -> None:
        """Reattach to fresh mappers after a scene rebuild.

        After ``_rebuild_scene``, the actors in ``registry.dim_actors``
        are replaced; the mappers we cached are stale.  Drop them and
        — if the plane was active — bind to the new ones.
        """
        self._attached_mappers.clear()
        if self._enabled:
            self._attach()
        self._plotter.render()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _axis_normal(self) -> tuple[float, float, float]:
        nx, ny, nz = _AXIS_NORMALS[self._axis]
        if self._flipped:
            return (-nx, -ny, -nz)
        return (nx, ny, nz)

    def _update_plane(self) -> None:
        """Recompute origin + normal in shifted coordinates."""
        nx, ny, nz = self._axis_normal()
        # Origin sits on the cut along the active axis; other axes are 0
        # (any point on the plane works for vtkPlane).
        idx = "XYZ".index(self._axis)
        origin = [0.0, 0.0, 0.0]
        origin[idx] = self._world_position - self._origin_shift[idx]
        self._plane.SetOrigin(*origin)
        self._plane.SetNormal(nx, ny, nz)

    def _attach(self) -> None:
        for actor in self._registry.dim_actors.values():
            if actor is None:
                continue
            try:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue
                mapper.AddClippingPlane(self._plane)
                self._attached_mappers.append(mapper)
            except Exception:
                # Some actors (e.g. point glyph composites) may not
                # support clipping planes; skip silently rather than
                # losing the whole feature.
                pass

    def _detach(self) -> None:
        for mapper in self._attached_mappers:
            try:
                mapper.RemoveClippingPlane(self._plane)
            except Exception:
                pass
        self._attached_mappers.clear()


__all__ = ["ClipPlaneOverlay"]
