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

Bootstrap state: one always-present "Geometry" composition with no
layers — the user's base view of just mesh + nodes. ``Esc`` returns
to it. The user can rename Geometry but not delete it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Optional

if TYPE_CHECKING:
    from ._base import Diagram


# ---------------------------------------------------------------------
# Special composition ids
# ---------------------------------------------------------------------

GEOMETRY_ID = "geometry"     # the locked, always-present base composition


# ---------------------------------------------------------------------
# Composition record
# ---------------------------------------------------------------------

@dataclass
class Composition:
    """One named bundle of Diagram instances.

    Attributes
    ----------
    id
        Stable identifier — UUID for user-created compositions, the
        constant :data:`GEOMETRY_ID` for the locked base composition.
    name
        Display name (mutable). Manager.rename() updates it.
    layers
        Direct refs to the Diagram instances belonging to this
        composition. Order matches the layer-stack z-order.
    locked
        ``True`` for Geometry — the manager refuses ``remove(id)`` for
        locked compositions.
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

    The Geometry composition is created on construction and made
    active by default. Observers fire on add / remove / rename /
    set_active / layer-membership changes — the OutlineTree subscribes
    to repaint, the DetailsPanel subscribes to refresh the stack view.
    """

    def __init__(self) -> None:
        self._compositions: list[Composition] = []
        self._active_id: Optional[str] = None
        self._on_changed: list[Callable[[], None]] = []
        # Bootstrap: locked Geometry composition.
        geom = Composition(
            id=GEOMETRY_ID, name="Geometry", layers=[], locked=True,
        )
        self._compositions.append(geom)
        self._active_id = GEOMETRY_ID

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

    @property
    def geometry(self) -> Composition:
        """The locked base composition — guaranteed to exist."""
        comp = self.find(GEOMETRY_ID)
        assert comp is not None, "Geometry composition was deleted somehow"
        return comp

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

    def add(self, name: str = "Diagram", *, make_active: bool = True) -> Composition:
        """Append a new composition with a unique name (auto-numbers collisions)."""
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
        wanted (currently the user just wants a quick copy of the
        same layers; a true deep copy would re-attach actors and is
        out of scope for v1).
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
        """Remove a composition. Refuses when it's locked.

        Returns True if a composition was removed. The caller should
        teardown the layers (call ``registry.remove`` on each) before
        invoking this — the manager only drops the grouping, not the
        underlying Diagram instances.
        """
        for i, c in enumerate(self._compositions):
            if c.id == comp_id:
                if c.locked:
                    return False
                del self._compositions[i]
                if self._active_id == comp_id:
                    self._active_id = (
                        GEOMETRY_ID if self.find(GEOMETRY_ID) is not None
                        else (self._compositions[0].id if self._compositions
                              else None)
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
        # Avoid duplicates with other compositions.
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

        No-op if the layer is already a member of any composition or
        if the target composition is locked (Geometry doesn't accept
        layers — it's the base mesh-only view).
        """
        if self.composition_for_layer(layer) is not None:
            return
        comp = self.find(comp_id)
        if comp is None:
            return
        if comp.locked:
            return
        comp.layers.append(layer)
        self._notify()

    @property
    def active_accepts_layers(self) -> bool:
        """Whether ``+ Add layer`` should target the active composition.

        ``False`` when no composition is active or when the active one
        is locked (Geometry). Callers can branch to "create a new
        diagram first" in that case.
        """
        active = self.active
        return active is not None and not active.locked

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
