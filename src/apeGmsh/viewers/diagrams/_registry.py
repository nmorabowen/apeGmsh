"""DiagramRegistry — ordered list of active diagrams.

Owns the in-memory collection of attached / detached diagrams. The
Director routes step changes through the registry; the UI's Diagrams
tab subscribes to ``on_changed`` to repaint the list.

Operations are sequential and side-effect-only — adding a diagram
attaches it; removing detaches it; reordering rebuilds the internal
list. The registry does not coalesce renders itself; the Director
calls ``plotter.render()`` once per logical event.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional

from ._base import Diagram

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from ..scene.fem_scene import FEMSceneData


class DiagramRegistry:
    """Ordered collection of Diagrams plus add/remove/toggle/reorder.

    The registry is plotter-aware: it calls ``attach`` /
    ``detach`` on diagrams as they are added / removed / re-attached
    (e.g., on stage change).
    """

    def __init__(self) -> None:
        self._diagrams: list[Diagram] = []
        self._plotter: Any = None
        self._fem: "FEMData | None" = None
        self._scene: "FEMSceneData | None" = None
        self.on_changed: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Plotter binding
    # ------------------------------------------------------------------

    def bind(
        self,
        plotter: Any,
        fem: "FEMData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        """Bind to a plotter + FEMData (+ optional substrate scene).

        Future ``add(...)`` calls attach immediately to this plotter.

        Idempotent — calling ``bind`` again with a new plotter detaches
        every diagram from the old plotter (if attached) and re-attaches
        to the new one.
        """
        if self._plotter is not None and self._plotter is not plotter:
            for d in self._diagrams:
                if d.is_attached:
                    d.detach()
        self._plotter = plotter
        self._fem = fem
        self._scene = scene
        for d in self._diagrams:
            if not d.is_attached:
                d.attach(plotter, fem, scene)

    def unbind(self) -> None:
        """Detach all diagrams and forget the plotter binding."""
        for d in self._diagrams:
            if d.is_attached:
                d.detach()
        self._plotter = None
        self._fem = None
        self._scene = None

    @property
    def is_bound(self) -> bool:
        return self._plotter is not None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, diagram: Diagram) -> Diagram:
        """Append ``diagram`` and attach it if the registry is bound."""
        self._diagrams.append(diagram)
        if self.is_bound and not diagram.is_attached:
            diagram.attach(self._plotter, self._fem, self._scene)  # type: ignore[arg-type]
        self._notify()
        return diagram

    def remove(self, diagram: Diagram) -> None:
        """Detach and drop ``diagram`` from the list. No-op if absent."""
        if diagram not in self._diagrams:
            return
        if diagram.is_attached:
            diagram.detach()
        self._diagrams.remove(diagram)
        self._notify()

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self._diagrams):
            self.remove(self._diagrams[index])

    def clear(self) -> None:
        for d in list(self._diagrams):
            if d.is_attached:
                d.detach()
        self._diagrams.clear()
        self._notify()

    def move(self, index: int, new_index: int) -> None:
        """Reorder a diagram. Used by the Diagrams tab Up / Down buttons."""
        if not (0 <= index < len(self._diagrams)):
            return
        new_index = max(0, min(new_index, len(self._diagrams) - 1))
        if new_index == index:
            return
        d = self._diagrams.pop(index)
        self._diagrams.insert(new_index, d)
        self._notify()

    def set_visible(self, diagram: Diagram, visible: bool) -> None:
        diagram.set_visible(visible)
        self._notify()

    # ------------------------------------------------------------------
    # Time / stage routing
    # ------------------------------------------------------------------

    def update_to_step(self, step_index: int) -> None:
        """Forward a step change to every visible attached diagram.

        The Director calls this once per logical step change. The
        registry does not call ``plotter.render()`` — that is the
        Director's responsibility (one render per coalesced batch).
        """
        for d in self._diagrams:
            if d.is_attached and d.is_visible:
                d.update_to_step(step_index)

    def reattach_all(self) -> None:
        """Detach + re-attach every diagram. Used on stage change.

        Subclasses' ``attach`` re-resolves the selector against the
        (potentially new) FEM and rebuilds initial actors. This is the
        cold path — accept the cost.
        """
        if not self.is_bound:
            return
        for d in self._diagrams:
            if d.is_attached:
                d.detach()
        for d in self._diagrams:
            d.attach(self._plotter, self._fem, self._scene)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Iteration / inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._diagrams)

    def __iter__(self) -> Iterator[Diagram]:
        return iter(self._diagrams)

    def __getitem__(self, index: int) -> Diagram:
        return self._diagrams[index]

    def index_of(self, diagram: Diagram) -> Optional[int]:
        try:
            return self._diagrams.index(diagram)
        except ValueError:
            return None

    def diagrams(self) -> list[Diagram]:
        """Live snapshot copy of the diagram list."""
        return list(self._diagrams)

    def visible_diagrams(self) -> list[Diagram]:
        return [d for d in self._diagrams if d.is_attached and d.is_visible]

    # ------------------------------------------------------------------
    # Observer plumbing
    # ------------------------------------------------------------------

    def _notify(self) -> None:
        for cb in list(self.on_changed):
            try:
                cb()
            except Exception as exc:
                import sys
                print(
                    f"[DiagramRegistry] observer raised: {exc}",
                    file=sys.stderr,
                )

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a no-arg callback fired on add/remove/move/visibility.

        Returns an unsubscribe thunk.
        """
        self.on_changed.append(callback)
        def _unsub() -> None:
            if callback in self.on_changed:
                self.on_changed.remove(callback)
        return _unsub
