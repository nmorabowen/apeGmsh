"""PickEngine for ResultsViewer — actor inventory + mode routing.

Diagrams that own a 3-D pickable actor register themselves on attach so
the results-pick controller can route a vtkCellPicker hit back to the
diagram-specific ``(element_id, sub_index, …)`` result without walking
every active diagram on every click.

Currently registered:

* :class:`GaussPointDiagram` with ``kind="gp"``. Reverse-map returns
  ``(element_id, gp_index, world_xyz)``.

:class:`FiberSectionDiagram` is intentionally NOT registered — its
3-D point cloud sets ``pickable=False`` so picks pass through to the
substrate. Fiber selection happens through the 2-D side panel + the
director's ``picked_gp`` channel.

Mode routing (Phase 3.2): :func:`PickEngine.set_pick_mode` walks the
inventory and flips each actor's ``SetPickable`` per the mode's
allow-list. The pick controller then sees only the actors relevant to
the current mode. ``Alt``-pick-through (a context manager on the
engine) temporarily restores all pickability for one click so the
user can reach back to the substrate without leaving GP/FIBER mode.
"""
from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Iterator, Optional


# Reverse-map signature: ``cell_id -> (element_id, sub_index, *)`` or
# ``None`` if the cell index falls outside the diagram's range. The
# tail of the tuple is diagram-defined (e.g., world coords for GP); the
# pick controller destructures by position.
ReverseMapFn = Callable[[int], Optional[tuple]]


class PickMode(str, Enum):
    """What the user is currently clicking *for*.

    Drives :meth:`PickEngine.set_pick_mode`, which restricts which
    inventory actors stay pickable. Substrate pickability is independent
    of this — the controller still runs ``vtkCellPicker.Pick`` on every
    visible actor and routes by mode after the hit.
    """
    NODE = "node"
    ELEMENT = "element"
    GP = "gp"
    FIBER = "fiber"


# Mapping mode → set of inventory-actor kinds that stay pickable.
# Empty allow-list means "no inventory actors are pickable" — the
# user is targeting the substrate (node-snap or cell-resolve) and any
# overlay glyphs (GP markers) should let the click pass through.
_MODE_ALLOW: dict[PickMode, frozenset[str]] = {
    PickMode.NODE: frozenset(),
    PickMode.ELEMENT: frozenset(),
    PickMode.GP: frozenset({"gp"}),
    PickMode.FIBER: frozenset({"fiber"}),
}


class PickEngine:
    """Actor inventory for the Results viewer's pick controller.

    One instance per :class:`ResultsViewer`; stashed on
    ``FEMSceneData.pick_engine`` so diagrams can find it during
    ``attach``. The pick controller (``results_pick.install_results_pick``)
    consults the engine to translate a picker hit on a registered
    actor into the diagram's ``(eid, sub_index, …)`` triple.
    """

    def __init__(self) -> None:
        # ``id(vtkProp)`` -> ``(kind, reverse_map_fn, actor)``.
        # Keying by ``id`` avoids vtkObject hashing surprises; we
        # retain the actor reference so ``set_pick_mode`` can walk the
        # inventory and call SetPickable on each one.
        self._actors: dict[int, tuple[str, ReverseMapFn, Any]] = {}
        # Currently active mode. ``set_pick_mode`` updates this and
        # reapplies the allow-list to every registered actor.
        self._mode: PickMode = PickMode.NODE
        # Dispatcher used to publish PICK_MODE_CHANGED on transition.
        # Wired by ResultsViewer (``engine.dispatcher = ...``); None
        # in headless tests where no event bus exists.
        self.dispatcher: Any = None

    def register_actor(
        self,
        actor: Any,
        kind: str,
        reverse_map_fn: ReverseMapFn,
    ) -> None:
        """Register ``actor`` so the picker can route hits + mode flips.

        Subsequent calls with the same actor overwrite the prior
        registration (used by diagrams that re-attach across stage /
        step changes).

        ``kind`` is one of the documented kinds (``"gp"``, ``"fiber"``,
        ``"element"``, ``"node"``). Phase 3.2's ``PickMode`` →
        allow-list mapping decides which kinds remain pickable per mode.

        ``reverse_map_fn`` takes a VTK cell index and returns the
        diagram's interpretation of that cell — typically
        ``(element_id, sub_index)`` plus diagram-specific data such as
        world coords. The pick controller destructures the leading
        elements by position.
        """
        if actor is None:
            return
        kind_s = str(kind)
        self._actors[id(actor)] = (kind_s, reverse_map_fn, actor)
        # Apply the current mode immediately — a diagram that attaches
        # while the viewer is already in GP mode should land pickable,
        # not inherit the actor's default (True) regardless of mode.
        self._set_actor_pickable(actor, kind_s in _MODE_ALLOW[self._mode])

    def unregister_actor(self, actor: Any) -> None:
        """Drop ``actor`` from the inventory. No-op when unregistered."""
        if actor is None:
            return
        self._actors.pop(id(actor), None)

    def resolve_pick(
        self, actor: Any, cell_id: int,
    ) -> Optional[tuple]:
        """Translate a picker hit to the diagram's reverse-map result.

        Returns ``None`` when ``actor`` isn't in the inventory or its
        reverse_map_fn returns ``None`` for ``cell_id``.
        """
        if actor is None:
            return None
        entry = self._actors.get(id(actor))
        if entry is None:
            return None
        _kind, reverse_map_fn, _actor = entry
        try:
            return reverse_map_fn(int(cell_id))
        except Exception:
            return None

    def kind_for_actor(self, actor: Any) -> Optional[str]:
        """Return the registered ``kind`` for ``actor`` or ``None``."""
        if actor is None:
            return None
        entry = self._actors.get(id(actor))
        return entry[0] if entry is not None else None

    def registered_actors(self) -> list[tuple[str, Any]]:
        """Snapshot ``(kind, actor)`` for every registered entry."""
        return [(k, a) for (k, _r, a) in self._actors.values()]

    def is_registered(self, actor: Any) -> bool:
        return id(actor) in self._actors

    def __len__(self) -> int:
        return len(self._actors)

    # ------------------------------------------------------------------
    # Mode routing (Phase 3.2)
    # ------------------------------------------------------------------

    @property
    def mode(self) -> PickMode:
        return self._mode

    def set_pick_mode(self, mode: PickMode) -> None:
        """Switch the active pick mode + flip inventory pickability.

        For each registered actor, sets ``actor.SetPickable(kind in
        _MODE_ALLOW[mode])``. Fires ``PICK_MODE_CHANGED`` through the
        injected dispatcher on transition (no-op on same-mode set so
        the storm guard stays cheap).
        """
        if not isinstance(mode, PickMode):
            mode = PickMode(mode)
        if mode is self._mode:
            return
        self._mode = mode
        allow = _MODE_ALLOW[mode]
        for kind, _rev, actor in self._actors.values():
            self._set_actor_pickable(actor, kind in allow)
        if self.dispatcher is not None:
            try:
                from ..diagrams._dispatch import PICK_MODE_CHANGED
                self.dispatcher.fire(PICK_MODE_CHANGED, payload=mode.value)
            except Exception:
                pass

    @contextmanager
    def with_pick_through(self) -> Iterator[None]:
        """Temporarily make every inventory actor pickable.

        Used by the pick controller for the ``Alt``-modifier
        ``pick-through`` gesture: the user wants a single click to
        ignore the current mode filter (e.g., reach back to the
        substrate while in GP mode). On exit, the previous mode's
        allow-list is reapplied.
        """
        # Snapshot current pickable state per actor so a nested
        # with_pick_through restores to whatever was set when the
        # outer block entered.
        prior: dict[int, bool] = {}
        for k, entry in self._actors.items():
            _kind, _rev, actor = entry
            try:
                prior[k] = bool(actor.GetPickable())
            except Exception:
                prior[k] = True
            self._set_actor_pickable(actor, True)
        try:
            yield
        finally:
            for k, entry in self._actors.items():
                _kind, _rev, actor = entry
                self._set_actor_pickable(actor, prior.get(k, True))

    def _set_actor_pickable(self, actor: Any, pickable: bool) -> None:
        """Best-effort SetPickable — swallow VTK errors silently."""
        try:
            actor.SetPickable(bool(pickable))
        except Exception:
            pass


__all__ = ["PickEngine", "PickMode", "ReverseMapFn"]
