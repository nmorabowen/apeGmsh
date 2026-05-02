"""Composition manager — named groups of layers (the user-facing "diagrams").

A *Composition* is a named bundle of :class:`Diagram` instances (the
internal class is still ``Diagram``; in the UI we call it a "Layer").
The user picks a composition in the outline, and the details dock
renders that composition's layer stack.

Compositions are a UI/grouping abstraction layered on top of the flat
:class:`DiagramRegistry`. Each composition keeps direct refs to its
member Diagram instances; removal/teardown calls the registry. The
registry remains the single source of truth for *which layers exist*;
the manager tracks *how they're grouped* and *which composition is
active*.

In the post-Geometry refactor, each :class:`CompositionManager` lives
inside one :class:`Geometry` (see :mod:`._geometries`) — it owns the
compositions of that geometry only. The previously bootstrapped locked
"Geometry" composition is gone; the geometry concept now sits one
level above.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from ._base import Diagram


# ---------------------------------------------------------------------
# Special composition ids (legacy)
# ---------------------------------------------------------------------

# Kept as a string constant for back-compat with old session JSON that
# referenced it; new code should not depend on it.
GEOMETRY_ID = "geometry"


# ---------------------------------------------------------------------
# Composition record
# ---------------------------------------------------------------------

@dataclass
class Composition:
    """One named bundle of Diagram instances.

    Attributes
    ----------
    id
        Stable identifier — UUID for user-created compositions.
    name
        Display name (mutable). Manager.rename() updates it.
    layers
        Direct refs to the Diagram instances belonging to this
        composition. Order matches the layer-stack z-order.
    locked
        Reserved — currently always ``False``. Kept on the dataclass so
        old code paths that read it don't break; the new model has no
        locked compositions (the locked-Geometry concept moved up to
        :class:`Geometry`).
    """
    id: str
    name: str
    layers: list = field(default_factory=list)
    locked: bool = False


# ---------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------

class CompositionManager:
    """Registry of compositions + active-composition pointer + observers.

    Bootstrap state: empty. The owning :class:`Geometry` decides
    whether to pre-create a default composition. Observers fire on
    add / remove / rename / set_active / layer-membership changes.
    """

    def __init__(self) -> None:
        self._compositions: list[Composition] = []
        self._active_id: Optional[str] = None
        self._on_changed: list[Callable[[], None]] = []
        self._parent_notify: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # Iteration / lookup
    # ------------------------------------------------------------------

    @property
    def compositions(self) -> list[Composition]:
        """Snapshot copy of the composition list (UI-order)."""
        return list(self._compositions)

    @property
    def active(self) -> Optional[Composition]:
        return self.find(self._active_id) if self._active_id else None

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def find(self, comp_id: Optional[str]) -> Optional[Composition]:
        if comp_id is None:
            return None
        for c in self._compositions:
            if c.id == comp_id:
                return c
        return None

    def composition_for_layer(self, layer: "Diagram") -> Optional[Composition]:
        """Which composition does ``layer`` belong to (if any)?"""
        for c in self._compositions:
            if layer in c.layers:
                return c
        return None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(
        self, name: str = "Diagram", *, make_active: bool = True,
    ) -> Composition:
        """Append a new composition with a unique name."""
        unique_name = self._unique_name(name)
        comp = Composition(id=str(uuid.uuid4()), name=unique_name)
        self._compositions.append(comp)
        if make_active:
            self._active_id = comp.id
        self._notify()
        return comp

    def duplicate(self, comp_id: str) -> Optional[Composition]:
        """Clone the composition's name + layer membership.

        Note: the cloned composition references the SAME Diagram
        instances as the original. The caller is responsible for
        deep-cloning the underlying layers if independent state is
        wanted.
        """
        src = self.find(comp_id)
        if src is None:
            return None
        new_comp = Composition(
            id=str(uuid.uuid4()),
            name=self._unique_name(src.name + " (copy)"),
            layers=list(src.layers),
        )
        self._compositions.append(new_comp)
        self._active_id = new_comp.id
        self._notify()
        return new_comp

    def remove(self, comp_id: str) -> bool:
        """Remove a composition. Returns True on success.

        Caller is responsible for tearing down the layers (calling
        ``registry.remove`` on each) before invoking this; the manager
        only drops the grouping.
        """
        for i, c in enumerate(self._compositions):
            if c.id == comp_id:
                del self._compositions[i]
                if self._active_id == comp_id:
                    self._active_id = (
                        self._compositions[0].id
                        if self._compositions else None
                    )
                self._notify()
                return True
        return False

    def rename(self, comp_id: str, new_name: str) -> bool:
        """Rename a composition. Empty / collision names are no-ops."""
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        comp = self.find(comp_id)
        if comp is None:
            return False
        if comp.name == new_name:
            return False
        if any(
            c.id != comp_id and c.name == new_name for c in self._compositions
        ):
            new_name = self._unique_name(new_name)
        comp.name = new_name
        self._notify()
        return True

    def set_active(self, comp_id: Optional[str]) -> None:
        """Set the active composition (or None for "no selection")."""
        if comp_id is not None and self.find(comp_id) is None:
            return
        if comp_id == self._active_id:
            return
        self._active_id = comp_id
        self._notify()

    def add_layer(self, comp_id: str, layer: "Diagram") -> None:
        """Tag ``layer`` with composition ``comp_id``.

        No-op if the layer is already a member of any composition.
        """
        if self.composition_for_layer(layer) is not None:
            return
        comp = self.find(comp_id)
        if comp is None:
            return
        comp.layers.append(layer)
        self._notify()

    @property
    def active_accepts_layers(self) -> bool:
        """Whether ``+ Add layer`` should target the active composition.

        ``False`` when no composition is active. Callers branch to
        "create a new composition first" in that case.
        """
        return self.active is not None

    def remove_layer(self, layer: "Diagram") -> None:
        """Drop ``layer`` from whichever composition owns it."""
        for c in self._compositions:
            if layer in c.layers:
                c.layers.remove(layer)
                self._notify()
                return

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        self._on_changed.append(callback)
        def _unsub() -> None:
            if callback in self._on_changed:
                self._on_changed.remove(callback)
        return _unsub

    def set_parent_notifier(
        self, callback: Optional[Callable[[], None]],
    ) -> None:
        """Bridge composition events up to the owning Geometry's manager.

        :class:`GeometryManager` calls this on construction so its own
        ``subscribe()`` callbacks see every composition change without
        each subscriber having to attach to every per-Geometry manager.
        """
        self._parent_notify = callback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unique_name(self, base: str) -> str:
        existing = {c.name for c in self._compositions}
        if base not in existing:
            return base
        n = 2
        while f"{base} {n}" in existing:
            n += 1
        return f"{base} {n}"

    def _notify(self) -> None:
        for cb in list(self._on_changed):
            try:
                cb()
            except Exception:
                pass
        if self._parent_notify is not None:
            try:
                self._parent_notify()
            except Exception:
                pass
