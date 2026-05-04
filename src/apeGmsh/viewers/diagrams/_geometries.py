"""Geometry manager — top-level grouping above compositions.

A *Geometry* is a deformation-bearing container that owns a list of
compositions (the user-facing "Diagrams"). Multiple geometries can
coexist, each with its own deformation state (field + scale). The
viewport renders only the active geometry; switching geometry
re-applies that geometry's deformation to the substrate and routes
all per-Geometry state to its compositions.

Hierarchy::

    GeometryManager           ← director.geometries
    └── Geometry              ← outline first level
        └── CompositionManager
            └── Composition   ← outline second level (UI: "Diagram")
                └── Diagram   ← layer (Contour, VectorGlyph, …)

Bootstrap: one always-present "Geometry 1" with an empty
:class:`CompositionManager`. The user can add more geometries; the
manager refuses ``remove`` on the last surviving geometry so the
viewer always has somewhere to land.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from ._compositions import CompositionManager


@dataclass
class Geometry:
    """One named deformation context + its compositions.

    Attributes
    ----------
    id
        Stable UUID.
    name
        Display name (mutable via :meth:`GeometryManager.rename`).
    deform_enabled
        Whether the substrate is currently warped by this geometry's
        nodal vector field.
    deform_field
        Vector prefix driving the warp (``"displacement"`` /
        ``"velocity"`` / ``"acceleration"``). ``None`` when no field
        has been picked.
    deform_scale
        Scalar multiplier on the warp.
    show_mesh
        Whether the substrate fill + wireframe are visible while this
        geometry is active. Per-geometry so a "deformed shell" view
        can hide the wireframe while a "diagnostic" view keeps it.
    show_nodes
        Whether the node-cloud overlay is visible while this geometry
        is active.
    display_opacity
        Single 0..1 alpha applied to substrate fill + wireframe + node
        cloud when this geometry is active. Lets the user dim the
        substrate so on-top diagrams (contour, line force) read better.
    compositions
        Per-geometry composition manager. Always non-null.
    """
    id: str
    name: str
    deform_enabled: bool = False
    deform_field: Optional[str] = None
    deform_scale: float = 1.0
    show_mesh: bool = True
    show_nodes: bool = True
    display_opacity: float = 1.0
    compositions: CompositionManager = field(default_factory=CompositionManager)


class GeometryManager:
    """Registry of geometries + active-geometry pointer + observers.

    One observer chain spans both levels: when any geometry's
    composition manager fires (add / remove / rename / set_active /
    layer-membership), this manager re-fires its own observers. UI
    consumers (outline, settings tab) need only one subscription.
    """

    def __init__(self) -> None:
        self._geometries: list[Geometry] = []
        self._active_id: Optional[str] = None
        self._on_changed: list[Callable[[], None]] = []
        # Bootstrap: one geometry. Bridge its composition manager
        # events up so consumers only subscribe at the top level.
        boot = self._make_geometry("Geometry 1")
        self._geometries.append(boot)
        self._active_id = boot.id

    # ------------------------------------------------------------------
    # Iteration / lookup
    # ------------------------------------------------------------------

    @property
    def geometries(self) -> list[Geometry]:
        """Snapshot copy of the geometry list (UI-order)."""
        return list(self._geometries)

    @property
    def active(self) -> Optional[Geometry]:
        return self.find(self._active_id) if self._active_id else None

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def find(self, geom_id: Optional[str]) -> Optional[Geometry]:
        if geom_id is None:
            return None
        for g in self._geometries:
            if g.id == geom_id:
                return g
        return None

    def geometry_for_composition(self, comp_id: str) -> Optional[Geometry]:
        """Return the geometry that owns the composition ``comp_id``."""
        for g in self._geometries:
            if g.compositions.find(comp_id) is not None:
                return g
        return None

    def geometry_for_layer(self, layer) -> Optional[Geometry]:
        """Return the geometry whose composition contains ``layer``."""
        for g in self._geometries:
            if g.compositions.composition_for_layer(layer) is not None:
                return g
        return None

    # ------------------------------------------------------------------
    # Mutations — geometry list
    # ------------------------------------------------------------------

    def add(
        self, name: str = "Geometry", *, make_active: bool = True,
    ) -> Geometry:
        """Append a new geometry with a unique name."""
        geom = self._make_geometry(self._unique_name(name))
        self._geometries.append(geom)
        if make_active:
            self._active_id = geom.id
        self._notify()
        return geom

    def duplicate(self, geom_id: str) -> Optional[Geometry]:
        """Clone the geometry's deformation state + composition shells.

        The cloned geometry has *no* compositions to start — copying
        layer membership across geometries is ambiguous (the same
        Diagram instance can't live in two geometries' submeshes
        without re-attaching). v1: shallow clone of deformation state
        only.
        """
        src = self.find(geom_id)
        if src is None:
            return None
        new_geom = self._make_geometry(self._unique_name(src.name + " (copy)"))
        new_geom.deform_enabled = src.deform_enabled
        new_geom.deform_field = src.deform_field
        new_geom.deform_scale = src.deform_scale
        new_geom.show_mesh = src.show_mesh
        new_geom.show_nodes = src.show_nodes
        new_geom.display_opacity = src.display_opacity
        self._geometries.append(new_geom)
        self._active_id = new_geom.id
        self._notify()
        return new_geom

    def remove(self, geom_id: str) -> bool:
        """Remove a geometry. Refuses when it would empty the list.

        Caller is responsible for tearing down every layer in every
        composition before invoking this — the manager only drops the
        grouping.
        """
        if len(self._geometries) <= 1:
            return False
        for i, g in enumerate(self._geometries):
            if g.id == geom_id:
                del self._geometries[i]
                if self._active_id == geom_id:
                    self._active_id = (
                        self._geometries[0].id
                        if self._geometries else None
                    )
                self._notify()
                return True
        return False

    def rename(self, geom_id: str, new_name: str) -> bool:
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        geom = self.find(geom_id)
        if geom is None or geom.name == new_name:
            return False
        if any(
            g.id != geom_id and g.name == new_name for g in self._geometries
        ):
            new_name = self._unique_name(new_name)
        geom.name = new_name
        self._notify()
        return True

    def set_active(self, geom_id: Optional[str]) -> None:
        """Switch the active geometry (or None to deselect entirely)."""
        if geom_id is not None and self.find(geom_id) is None:
            return
        if geom_id == self._active_id:
            return
        self._active_id = geom_id
        self._notify()

    # ------------------------------------------------------------------
    # Mutations — deformation state on a geometry
    # ------------------------------------------------------------------

    def set_deformation(
        self,
        geom_id: str,
        *,
        enabled: Optional[bool] = None,
        field: Optional[str] = None,
        scale: Optional[float] = None,
    ) -> bool:
        """Update one or more deformation fields on a geometry.

        Pass only the fields you want to change; ``None`` leaves the
        existing value. Returns True if anything changed.
        """
        geom = self.find(geom_id)
        if geom is None:
            return False
        changed = False
        if enabled is not None and bool(enabled) != geom.deform_enabled:
            geom.deform_enabled = bool(enabled)
            changed = True
        if field is not None and field != geom.deform_field:
            geom.deform_field = str(field) if field else None
            changed = True
        if scale is not None and float(scale) != geom.deform_scale:
            geom.deform_scale = float(scale)
            changed = True
        if changed:
            self._notify()
        return changed

    # ------------------------------------------------------------------
    # Mutations — substrate display state on a geometry
    # ------------------------------------------------------------------

    def set_display(
        self,
        geom_id: str,
        *,
        show_mesh: Optional[bool] = None,
        show_nodes: Optional[bool] = None,
        display_opacity: Optional[float] = None,
    ) -> bool:
        """Update one or more display fields on a geometry.

        Mirrors :meth:`set_deformation` — pass only what you want to
        change; ``None`` leaves the existing value. Returns True if
        anything changed. Observers fire once per call regardless of
        how many fields were updated.
        """
        geom = self.find(geom_id)
        if geom is None:
            return False
        changed = False
        if show_mesh is not None and bool(show_mesh) != geom.show_mesh:
            geom.show_mesh = bool(show_mesh)
            changed = True
        if show_nodes is not None and bool(show_nodes) != geom.show_nodes:
            geom.show_nodes = bool(show_nodes)
            changed = True
        if display_opacity is not None:
            clamped = max(0.0, min(1.0, float(display_opacity)))
            if clamped != geom.display_opacity:
                geom.display_opacity = clamped
                changed = True
        if changed:
            self._notify()
        return changed

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to ANY state change (geometry list / active /
        per-geometry composition list / active composition / rename
        / layer membership)."""
        self._on_changed.append(callback)

        def _unsub() -> None:
            if callback in self._on_changed:
                self._on_changed.remove(callback)
        return _unsub

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_geometry(self, name: str) -> Geometry:
        comp_mgr = CompositionManager()
        comp_mgr.set_parent_notifier(self._notify)
        return Geometry(id=str(uuid.uuid4()), name=name, compositions=comp_mgr)

    def _unique_name(self, base: str) -> str:
        existing = {g.name for g in self._geometries}
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


__all__ = ["Geometry", "GeometryManager"]
