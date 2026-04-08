"""
VisibilityManager — Hide / isolate / reveal entities via cell colors.

Uses ``ColorManager`` to set hidden entities to ``[0, 0, 0]`` RGB.
No ``actor.VisibilityOff()`` — consistent for batched actors where
all entities of a dimension share one actor.

Usage::

    vis = VisibilityManager(registry, color_mgr, selection)
    vis.hide()          # hide current picks
    vis.isolate()       # hide everything except current picks
    vis.reveal_all()    # show everything
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .color_manager import ColorManager
    from .entity_registry import DimTag, EntityRegistry
    from .selection import SelectionState


class VisibilityManager:
    """Cell-color-based entity visibility."""

    __slots__ = ("_registry", "_color_mgr", "_selection", "_hidden", "on_changed")

    def __init__(
        self,
        registry: "EntityRegistry",
        color_mgr: "ColorManager",
        selection: "SelectionState",
    ) -> None:
        self._registry = registry
        self._color_mgr = color_mgr
        self._selection = selection
        self._hidden: set["DimTag"] = set()
        self.on_changed: list[Callable[[], None]] = []

    @property
    def hidden(self) -> frozenset["DimTag"]:
        return frozenset(self._hidden)

    def is_hidden(self, dt: "DimTag") -> bool:
        return dt in self._hidden

    def hide(self) -> None:
        """Hide every currently picked entity, then clear picks."""
        picks = self._selection.picks
        if not picks:
            return
        for dt in picks:
            self._color_mgr.set_entity_state(dt, hidden=True)
            self._hidden.add(dt)
        self._selection.clear()
        self._fire()

    def isolate(self) -> None:
        """Hide everything except the currently picked entities."""
        picks = set(self._selection.picks)
        if not picks:
            return
        for dt in self._registry.all_entities():
            if dt not in picks:
                self._color_mgr.set_entity_state(dt, hidden=True)
                self._hidden.add(dt)
        self._fire()

    def reveal_all(self) -> None:
        """Restore all hidden entities to idle colors."""
        if not self._hidden:
            return
        self._hidden.clear()
        self._color_mgr.reset_all_idle()
        self._fire()

    def _fire(self) -> None:
        for cb in self.on_changed:
            try:
                cb()
            except Exception:
                pass
