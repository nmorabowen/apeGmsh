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
    from apeGmsh.viewers.data import ViewerData
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
        self._backend: Any = None
        self._view: "ViewerData | None" = None
        self._scene: "FEMSceneData | None" = None
        # ADR 0058 S1 — optional per-diagram scene resolution. When
        # set (by ResultsDirector.bind_plotter), every attach resolves
        # its scene through the owning geometry instead of the single
        # bound ``_scene``. None = legacy constant-scene behaviour
        # (standalone registries in unit tests).
        self._scene_resolver: "Callable[[Diagram], FEMSceneData | None] | None" = None
        # ADR 0058 S2b — optional scalar-bar title prefix resolution.
        # When set (by ResultsDirector.bind_plotter), every attach
        # stamps the resolver on the diagram so ScalarBarSupport can
        # prefix the bar title with the owning geometry's name while
        # more than one geometry is visible. None = unprefixed titles.
        self._bar_prefix_resolver: "Callable[[Diagram], Optional[str]] | None" = None
        # ADR 0058 S3b — optional stage-pin resolution (mirror of the
        # bar-prefix resolver). When set (by
        # ResultsDirector.bind_plotter), every attach stamps the
        # resolver on the diagram so ``Diagram._scoped_results`` can
        # follow the owning geometry's stage pin at read time. None =
        # reads stay on the active stage.
        self._stage_pin_resolver: "Callable[[Diagram], Optional[str]] | None" = None
        self.on_changed: list[Callable[[], None]] = []
        # Injected by ResultsDirector at construction (ADR 0056
        # Part 2): owner mutators fire their own dispatcher events.
        # None only for a standalone registry built outside a director
        # (unit tests).
        self.dispatcher: Any = None
        # Optional visual float16 cache forwarded from the
        # director via bind(); stamped on each diagram at attach so
        # per-step reads can slice a float16 row from RAM. None for
        # standalone registries (unit tests) -> diagrams fall back to
        # the per-step results.*.get(...) read path.
        self._visual_store: Any = None

    # ------------------------------------------------------------------
    # Plotter binding
    # ------------------------------------------------------------------

    def bind(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
        *,
        scene_resolver: (
            "Callable[[Diagram], FEMSceneData | None] | None"
        ) = None,
        bar_prefix_resolver: (
            "Callable[[Diagram], Optional[str]] | None"
        ) = None,
        stage_pin_resolver: (
            "Callable[[Diagram], Optional[str]] | None"
        ) = None,
        visual_store: Any = None,
    ) -> None:
        """Bind to a plotter + ViewerData (+ optional substrate scene).

        Future ``add(...)`` calls attach immediately to this plotter.

        This is the render-seam binding boundary (ADR 0042, R-B.final):
        the raw pyvista ``plotter`` is wrapped into a ``RenderBackend``
        **once** here, and every ``Diagram.attach`` is handed that
        backend — diagrams never see the raw plotter through this path.
        An already-wrapped backend is accepted as-is so callers may
        inject an alternate backend.

        Idempotent — calling ``bind`` again with a new plotter detaches
        every diagram from the old plotter (if attached) and re-attaches
        to the new one.
        """
        if self._plotter is not None and self._plotter is not plotter:
            for d in self._diagrams:
                if d.is_attached:
                    d.detach()
        self._plotter = plotter
        self._backend = self._as_backend(plotter)
        self._view = view
        self._scene = scene
        self._scene_resolver = scene_resolver
        self._bar_prefix_resolver = bar_prefix_resolver
        self._stage_pin_resolver = stage_pin_resolver
        self._visual_store = visual_store
        for d in self._diagrams:
            self._stamp_bar_prefix(d)
            self._stamp_stage_pin(d)
            self._stamp_visual_store(d)
            if not d.is_attached:
                d.attach(self._backend, view, self._scene_for(d))

    def _stamp_bar_prefix(self, diagram: Diagram) -> None:
        """Hand the diagram the bar-prefix resolver (ADR 0058 S2b).

        ``ScalarBarSupport._scalar_bar_title`` consults the attribute
        on demand, so the prefix is always computed against the
        CURRENT visible-geometry count — at bar add and remove time
        alike. Stamped (not passed per-call) so the viewer's restack
        path, which calls ``diagram.attach`` directly, needs no
        changes.
        """
        if self._bar_prefix_resolver is None:
            return
        try:
            diagram._bar_prefix_resolver = self._bar_prefix_resolver  # noqa: SLF001
        except Exception:
            pass

    def _stamp_stage_pin(self, diagram: Diagram) -> None:
        """Hand the diagram the stage-pin resolver (ADR 0058 S3b).

        ``Diagram._effective_stage_id`` consults the attribute on
        demand, so the pin is always resolved against the CURRENT
        owning geometry — at attach and at every read alike. Stamped
        (not passed per-call) for the same reason as the bar-prefix
        resolver: the viewer's restack / re-attach paths call
        ``diagram.attach`` directly and need no changes.
        """
        if self._stage_pin_resolver is None:
            return
        try:
            diagram._stage_pin_resolver = self._stage_pin_resolver  # noqa: SLF001
        except Exception:
            pass

    def _stamp_visual_store(self, diagram: Diagram) -> None:
        """Hand the diagram the director visual float16 cache.

        When present, per-step diagrams slice a float16 row
        from RAM instead of re-reading HDF5 every frame
        (see viewers.diagrams._visual_store). Stamped (not
        passed per-call) for the same reason as the stage-pin
        resolver: the viewer restack / re-attach paths call
        diagram.attach directly and need no changes. None for
        standalone registries leaves diagrams on the per-step
        read path (byte-identical to pre-cache behaviour).
        """
        if self._visual_store is None:
            return
        try:
            diagram._visual_store = self._visual_store  # noqa: SLF001
        except Exception:
            pass

    def _scene_for(self, diagram: Diagram) -> "FEMSceneData | None":
        """Scene this diagram attaches against (ADR 0058 S1 seam).

        Resolved through the owning geometry when a resolver was bound;
        the constant bound scene otherwise.
        """
        if self._scene_resolver is not None:
            try:
                resolved = self._scene_resolver(diagram)
            except Exception:
                resolved = None
            if resolved is not None:
                return resolved
        return self._scene

    @staticmethod
    def _as_backend(plotter: Any) -> Any:
        """Wrap a raw pyvista plotter into a ``PyVistaQtBackend``.

        Pass-through when ``plotter`` already satisfies the
        ``RenderBackend`` Protocol (has ``add_layer``) — lets a caller
        inject an alternate backend (e.g. a trame backend, or a test
        recording backend).
        """
        if hasattr(plotter, "add_layer"):
            return plotter
        from ..backends import PyVistaQtBackend
        return PyVistaQtBackend(plotter)

    def unbind(self) -> None:
        """Detach all diagrams and forget the plotter binding."""
        for d in self._diagrams:
            if d.is_attached:
                d.detach()
        self._plotter = None
        self._backend = None
        self._view = None
        self._scene = None
        self._scene_resolver = None
        self._bar_prefix_resolver = None
        self._stage_pin_resolver = None

    @property
    def is_bound(self) -> bool:
        return self._plotter is not None

    @property
    def backend(self) -> Any:
        """The ``RenderBackend`` this registry binds diagrams to.

        ``None`` until :meth:`bind`. Exposed so the viewer's restack /
        re-attach paths inject the same backend rather than the raw
        plotter (ADR 0042, R-B.final).
        """
        return self._backend

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, diagram: Diagram) -> Diagram:
        """Append ``diagram`` and attach it if the registry is bound.

        If ``attach()`` raises (e.g. ``NoDataError``), the diagram is
        rolled out of the list before the exception propagates so the
        registry never holds an un-attached diagram in an active list.
        """
        self._diagrams.append(diagram)
        self._stamp_bar_prefix(diagram)
        self._stamp_stage_pin(diagram)
        self._stamp_visual_store(diagram)
        if self.is_bound and not diagram.is_attached:
            try:
                diagram.attach(self._backend, self._view, self._scene_for(diagram))  # type: ignore[arg-type]
            except Exception:
                # Roll back the append; the caller (dialog) surfaces the
                # error to the user.
                self._diagrams.pop()
                raise
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

    def replace(self, old: Diagram, new: Diagram) -> Diagram:
        """Swap ``old`` for ``new`` at the same registry index.

        Used by the Layers panel to live-edit a diagram's Kind or Data
        without losing its z-position. Detaches ``old`` and attaches
        ``new`` (when bound). If ``new.attach()`` raises, ``old`` is
        re-attached and the exception propagates so the caller can
        surface it.
        """
        idx = self.index_of(old)
        if idx is None:
            # Treat as a plain add to keep callers simple.
            return self.add(new)
        was_attached = old.is_attached
        if was_attached:
            old.detach()
        self._diagrams[idx] = new
        self._stamp_bar_prefix(new)
        self._stamp_stage_pin(new)
        self._stamp_visual_store(new)
        if self.is_bound and not new.is_attached:
            try:
                new.attach(self._backend, self._view, self._scene_for(new))  # type: ignore[arg-type]
            except Exception:
                # Roll back: restore old at the same index and re-attach.
                self._diagrams[idx] = old
                if was_attached:
                    try:
                        old.attach(self._backend, self._view, self._scene_for(old))  # type: ignore[arg-type]
                    except Exception:
                        pass
                raise
        self._notify()
        return new

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
        """Set a diagram's user-intent visibility.

        Owner-fired (ADR 0056 Part 2): applies the change AND fires
        ``LAYER_VISIBILITY_CHANGED`` so the composition gate re-runs —
        call sites never fire it themselves. Idempotent per call: a
        no-op write (value unchanged) skips both the notify and the
        fire. Bulk cascades wrap their loop in
        ``dispatcher.gesture_batch()`` so N writes cost one gate pump.
        """
        if bool(diagram.is_visible) == bool(visible):
            return
        diagram.set_visible(visible)
        self._notify()
        if self.dispatcher is not None:
            from ._dispatch import LAYER_VISIBILITY_CHANGED
            self.dispatcher.fire(LAYER_VISIBILITY_CHANGED)

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
            d.attach(self._backend, self._view, self._scene_for(d))  # type: ignore[arg-type]

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
