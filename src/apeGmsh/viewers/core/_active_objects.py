"""ActiveObjects — per-viewer signal-driven state coordinator.

Single source of truth for "what is currently selected / active" in
one viewer window. Panels subscribe to its Qt signals to update
themselves; no panel calls another's ``refresh()`` / ``update()``
method directly.

Inspired by ParaView's ``pqActiveObjects`` ([`Qt/Components/pqActiveObjects.h:34`])
with three deliberate simplifications:

* **Per-viewer instance**, not process-global. One ``ActiveObjects``
  per ``ViewerWindow``, owned by it. Different viewers in the same
  process don't share state — keeps tests parallelizable and avoids
  cross-window leakage.
* **Opaque object refs**. The same class works for ``model.viewer``
  (BRep entities), ``mesh.viewer`` (elements / nodes), and
  ``results.viewer`` (Diagram layers, Geometries, Stages). The
  signal payload type is ``object`` so subscribers cast as needed.
* **Identity-based no-op**. ``set_X(same_ref)`` short-circuits
  before emitting — prevents redundant cascades when callers
  defensively call setters with the current value. Steps (ints)
  use ``==`` comparison instead since they're value types.

Usage::

    active = ActiveObjects(parent=window)
    active.activeLayerChanged.connect(properties_panel.bind_layer)
    active.activeLayerChanged.connect(color_editor.bind_layer)
    # ...later, from anywhere:
    active.set_active_layer(some_diagram)
    # → both subscribers re-bind, no direct wiring between them.

Plan 04 step 1: this file ships standalone with tests. Subsequent
sub-tasks migrate the three viewers' ad-hoc callback graphs onto it
one viewer at a time.
"""
from __future__ import annotations

from typing import Any, Optional


def _build_class():
    """Construct the ActiveObjects class at first import.

    Done lazily so importing this module doesn't pull qtpy. The first
    access constructs the class once and stashes it in module globals.
    """
    from qtpy import QtCore

    class ActiveObjects(QtCore.QObject):
        """Per-viewer central state. See module docstring."""

        # ── Universal signals (every viewer can use these) ───────
        selectionChanged       = QtCore.Signal(object)
        activeViewChanged      = QtCore.Signal(object)

        # ── Results-viewer signals ───────────────────────────────
        activeLayerChanged       = QtCore.Signal(object)
        activeCompositionChanged = QtCore.Signal(object)
        activeGeometryChanged    = QtCore.Signal(object)
        activeStageChanged       = QtCore.Signal(object)
        activeStepChanged        = QtCore.Signal(int)

        # ── Mesh-viewer signals ──────────────────────────────────
        activePickModeChanged  = QtCore.Signal(str)

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self._selection: Any = None
            self._active_view: Any = None
            self._active_layer: Any = None
            self._active_composition: Any = None
            self._active_geometry: Any = None
            self._active_stage: Any = None
            self._active_step: int = -1
            self._active_pick_mode: str = ""

        # ── Universal ────────────────────────────────────────────
        @property
        def selection(self) -> Any:
            return self._selection

        def set_selection(self, sel: Any) -> None:
            """Update the active selection and emit if it changed.

            Identity-comparison no-op: ``set_selection(same)`` returns
            without emitting. Callers can re-fire by passing a freshly
            constructed object if they need to force a refresh after
            in-place mutation.
            """
            if sel is self._selection:
                return
            self._selection = sel
            self.selectionChanged.emit(sel)

        @property
        def active_view(self) -> Any:
            return self._active_view

        def set_active_view(self, view: Any) -> None:
            if view is self._active_view:
                return
            self._active_view = view
            self.activeViewChanged.emit(view)

        # ── Results viewer ───────────────────────────────────────
        @property
        def active_layer(self) -> Any:
            return self._active_layer

        def set_active_layer(self, layer: Any) -> None:
            if layer is self._active_layer:
                return
            self._active_layer = layer
            self.activeLayerChanged.emit(layer)

        @property
        def active_composition(self) -> Any:
            return self._active_composition

        def set_active_composition(self, composition: Any) -> None:
            """Update the active composition (a.k.a. "Diagram" group)
            and emit if it changed.

            Identity comparison so opaque ids (strings, ints, refs)
            all work. ``None`` clears the active composition."""
            if composition is self._active_composition:
                return
            self._active_composition = composition
            self.activeCompositionChanged.emit(composition)

        @property
        def active_geometry(self) -> Any:
            return self._active_geometry

        def set_active_geometry(self, geometry: Any) -> None:
            if geometry is self._active_geometry:
                return
            self._active_geometry = geometry
            self.activeGeometryChanged.emit(geometry)

        @property
        def active_stage(self) -> Any:
            return self._active_stage

        def set_active_stage(self, stage: Any) -> None:
            if stage is self._active_stage:
                return
            self._active_stage = stage
            self.activeStageChanged.emit(stage)

        @property
        def active_step(self) -> int:
            return self._active_step

        def set_active_step(self, step: int) -> None:
            """Steps are value-typed (int), so use ``==`` comparison.

            ``-1`` is the sentinel for "no step active".
            """
            step_i = int(step)
            if step_i == self._active_step:
                return
            self._active_step = step_i
            self.activeStepChanged.emit(step_i)

        # ── Mesh viewer ──────────────────────────────────────────
        @property
        def active_pick_mode(self) -> str:
            """One of ``""``, ``"brep"``, ``"element"``, ``"node"``."""
            return self._active_pick_mode

        def set_active_pick_mode(self, mode: str) -> None:
            mode_s = str(mode) if mode else ""
            if mode_s == self._active_pick_mode:
                return
            self._active_pick_mode = mode_s
            self.activePickModeChanged.emit(mode_s)

        # ── Convenience ──────────────────────────────────────────
        def snapshot(self) -> dict:
            """Return a debug-friendly dict of the current state.

            For tests, logging, status-bar inspection. Not a contract
            for serialization — field names may shift as the viewer
            grows. Callers wanting persistent state should use the
            individual getters.
            """
            return {
                "selection":          self._selection,
                "active_view":        self._active_view,
                "active_layer":       self._active_layer,
                "active_composition": self._active_composition,
                "active_geometry":    self._active_geometry,
                "active_stage":       self._active_stage,
                "active_step":        self._active_step,
                "active_pick_mode":   self._active_pick_mode,
            }

    return ActiveObjects


_ActiveObjectsClass: Optional[type] = None


def __getattr__(name: str):
    """Lazy-construct ``ActiveObjects`` on first attribute access.

    Avoids pulling qtpy at module import time. Once built, the class
    is cached for the lifetime of the process.
    """
    global _ActiveObjectsClass
    if name == "ActiveObjects":
        if _ActiveObjectsClass is None:
            _ActiveObjectsClass = _build_class()
        return _ActiveObjectsClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
