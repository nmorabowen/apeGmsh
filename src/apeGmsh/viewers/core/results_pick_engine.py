"""PickInventory for ResultsViewer — actor inventory for pick routing.

Diagrams that own a 3-D pickable actor register themselves on attach so
the results-pick controller can route a ``vtkCellPicker`` hit back to the
diagram-specific ``(element_id, sub_index, …)`` result without walking
every active diagram on every click.

Currently registered:

* :class:`GaussPointDiagram` with ``kind="gp"``. Reverse-map returns
  ``(element_id, gp_index, world_xyz)``.

:class:`FiberSectionDiagram` is intentionally NOT registered — its
3-D point cloud sets ``pickable=False`` so picks pass through to the
substrate. Fiber selection happens through the 2-D side panel.

ADR 0047 R-D.2b — clean redesign: the per-mode allow-list + ``SetPickable``
flipping (``set_pick_mode`` / ``_MODE_ALLOW``) and the ``Alt``-pick-through
context manager are **gone**. They were dead in production: ``set_pick_mode``
had zero callers, so ``register_actor`` left every GP marker stuck
non-pickable (``_MODE_ALLOW[NODE]`` is empty) and GP picking only worked
when the user held ``Alt`` (which force-enabled everything). Registered
actors now simply stay pickable; the pick controller routes a hit by the
current mode + which actor was hit at *resolve* time. This is a pure
inventory — no VTK, no mode state.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


# Reverse-map signature: ``cell_id -> (element_id, sub_index, *)`` or
# ``None`` if the cell index falls outside the diagram's range. The
# tail of the tuple is diagram-defined (e.g., world coords for GP); the
# pick controller destructures by position.
ReverseMapFn = Callable[[int], Optional[tuple]]


class PickInventory:
    """Actor inventory for the Results viewer's pick controller.

    One instance per :class:`ResultsViewer`; stashed on
    ``FEMSceneData.pick_engine`` so diagrams can find it during
    ``attach``. The pick controller (``results_pick.install_results_pick``)
    consults it to translate a picker hit on a registered actor into the
    diagram's ``(eid, sub_index, …)`` triple.
    """

    def __init__(self) -> None:
        # ``id(vtkProp)`` -> ``(kind, reverse_map_fn, actor)``. Keying by
        # ``id`` avoids vtkObject hashing surprises and matches the
        # ``prop_id`` a PickBackend hit carries (ADR 0047).
        self._actors: dict[int, tuple[str, ReverseMapFn, Any]] = {}

    def register_actor(
        self,
        actor: Any,
        kind: str,
        reverse_map_fn: ReverseMapFn,
    ) -> None:
        """Register ``actor`` so the picker can route hits to its diagram.

        Subsequent calls with the same actor overwrite the prior
        registration (used by diagrams that re-attach across stage / step
        changes). ``kind`` is one of ``"gp"`` / ``"fiber"`` / ``"element"``
        / ``"node"``. ``reverse_map_fn`` takes a VTK cell index and returns
        the diagram's interpretation (typically ``(element_id, sub_index,
        world)``).

        The actor's pickability is left untouched — registered overlay
        actors stay pickable and are routed by mode at resolve time
        (ADR 0047 R-D.2b).
        """
        if actor is None:
            return
        self._actors[id(actor)] = (str(kind), reverse_map_fn, actor)

    def unregister_actor(self, actor: Any) -> None:
        """Drop ``actor`` from the inventory. No-op when unregistered."""
        if actor is None:
            return
        self._actors.pop(id(actor), None)

    def resolve_pick(self, actor: Any, cell_id: int) -> Optional[tuple]:
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

    # -- prop_id-keyed lookups (ADR 0047 R-D.2b-ii) -------------------
    # A PickBackend hit carries ``prop_id`` (``id(actor)``), not the actor
    # object, so the controller resolves overlay hits through these.

    def resolve(self, prop_id: Optional[int], cell_id: int) -> Optional[tuple]:
        """Like :meth:`resolve_pick` but keyed by ``prop_id`` (``id(actor)``)."""
        if prop_id is None:
            return None
        entry = self._actors.get(int(prop_id))
        if entry is None:
            return None
        _kind, reverse_map_fn, _actor = entry
        try:
            return reverse_map_fn(int(cell_id))
        except Exception:
            return None

    def registered_actors(self) -> list[tuple[str, Any]]:
        """Snapshot ``(kind, actor)`` for every registered entry."""
        return [(k, a) for (k, _r, a) in self._actors.values()]

    def is_registered(self, actor: Any) -> bool:
        return id(actor) in self._actors

    def __len__(self) -> int:
        return len(self._actors)


__all__ = ["PickInventory", "ReverseMapFn"]
