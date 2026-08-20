"""Give one pane's colour scales their drag and resize back.

ADR 0098 Amendment 5. Two facts make this a module rather than one call
in ``SessionPane.__init__``:

* **The controller is not stable.** ``_realize_legends`` constructs a
  fresh :class:`LegendController` on every full realize, so an
  interactor bound once observes an object the next slot edit throws
  away. The binding therefore re-installs whenever the controller
  changes identity (A5.3(3)).
* **Nothing repaints on the session path.** ``LegendController``
  mutators call ``_reconcile_and_fire``, which pushes the new geometry
  to the backend and then fires ``LEGEND_CHANGED`` on its
  ``dispatcher`` so the old viewer's Dispatcher coalesces a render. The
  session has no Dispatcher, so ``dispatcher`` is ``None`` and a
  dragged bar would move in the layout and never on screen. This class
  IS that dispatcher — the seam ``DiagramRegistry.bind`` uses for the
  same purpose.

The same ``fire`` that repaints is where the gesture becomes session
state (A5.3(4)): every mutator funnels through it, so move, resize and
re-dock are all recorded by one hook rather than three.
"""
from __future__ import annotations

from typing import Any, Optional


class PaneLegendBinding:
    """Bind a pane's legend controller to its view's placement record.

    Owns at most one :class:`LegendInteractor` at a time. Idempotent:
    binding the controller already bound does nothing, which matters
    because a re-step flush calls it on every scrub frame.
    """

    def __init__(self, backend: Any, reconciler: Any) -> None:
        self._backend = backend
        self._reconciler = reconciler
        self._controller: Any = None
        self._interactor: Any = None
        #: Guards the fire -> set_legend_placement -> session tick path
        #: from re-entering itself.
        self._recording = False

    # -- binding -------------------------------------------------------

    def bind(self, controller: Optional[Any]) -> None:
        """Attach to ``controller``, replacing any previous binding.

        ``None`` (a pane whose view causes no legend) tears the previous
        binding down rather than leaving observers on the interactor
        pointed at a dead controller.
        """
        if controller is self._controller:
            return
        self._detach()
        self._controller = controller
        if controller is None:
            return
        from ..core._legend_interactor import install_legend_interactor

        # Offscreen and headless backends have no interactor; install
        # returns None and the pane simply has no gesture (A5.6-1).
        self._interactor = install_legend_interactor(self._backend, controller)
        controller.dispatcher = self
        # A6 G3 - re-layout when the render window is resized. The
        # layout resolves a legend's box from pixel font metrics, so it
        # is a function of the viewport; computed once at attach it is
        # stale the moment the pane is sized, and a pane realized
        # before its first real resize keeps a box measured against a
        # ~30 px viewport. That is not only ugly: the hit test reads
        # the same extent, so the gesture misses a bar the user can
        # plainly see. Called only from diagrams/_registry.py until
        # now - the old path - which is why the session window never
        # had it.
        try:
            controller.install_resize_hook()
        except Exception:
            pass
        # ...and re-layout ONCE right now. The hook only covers resizes
        # from here on, and the box this controller was just built with
        # was measured during realize — before the pane was necessarily
        # at its real size. Without this the first paint keeps the
        # stale box until something else happens to resize the window.
        try:
            controller.refresh()
        except Exception:
            pass

    def on_viewport_resized(self) -> None:
        """Re-layout after the pane changed size.

        Driven from ``MeshPane.resizeEvent`` rather than left to
        ``install_resize_hook`` alone: that hook observes VTK's
        ``ConfigureEvent``, which does not reliably fire for a Qt-driven
        resize (measured — a pane resized and shown keeps a box
        measured against its ~30 px construction size, and the hit test
        then misses a bar the user can see). Qt's own resize event is
        the signal that always arrives; the VTK hook stays for resizes
        Qt does not mediate.
        """
        controller = self._controller
        if controller is None:
            return
        try:
            # ensure_current, not refresh: a live window drag fires
            # resizeEvent continuously, and only the ticks where the
            # render window actually took a new size need a re-layout.
            controller.ensure_current()
        except Exception:
            pass

    def _detach(self) -> None:
        if self._interactor is not None:
            try:
                self._interactor.uninstall()
            except Exception:
                pass
            self._interactor = None
        if self._controller is not None:
            # Only disown the dispatcher if it is still ours; a
            # controller adopted elsewhere keeps whatever it was given.
            if getattr(self._controller, "dispatcher", None) is self:
                self._controller.dispatcher = None
            self._controller = None

    def dispose(self) -> None:
        """Drop the interactor and the dispatcher seat. Idempotent.

        Called from ``SessionPane.dispose`` for the reason the pick
        installation is: these observers live on the interactor the
        pane is about to close.
        """
        self._detach()

    # -- the dispatcher seam -------------------------------------------

    def fire(self, _event: Any = None) -> None:
        """A legend mutated: record it on the view, then repaint.

        Record first. A repaint that beats the record would still show
        the right picture, but a realize landing in between would
        rebuild from a record that does not yet know about the gesture.
        """
        self._record()
        try:
            self._backend.render()
        except Exception:
            pass

    def _record(self) -> None:
        """Mirror every hand-placed entry onto the view (A5.3(4)).

        Undocked (``slot is None``) means the user placed it; docked
        means it belongs to the automatic stack, so any stored
        placement is cleared — which is how ``redock`` is recorded
        without a second hook.
        """
        controller, view = self._controller, self._view()
        if controller is None or view is None or self._recording:
            return
        self._recording = True
        try:
            live = {legend.field for legend in view.legends()}
            for entry in controller.entries():
                field = entry.key[1] if len(entry.key) > 1 else None
                if field is None or field not in live:
                    # A legend the view no longer causes: its placement
                    # is pruned by the view itself when the slot goes.
                    continue
                if entry.slot is None:
                    view.set_legend_placement(
                        field, entry.anchor, entry.font_scale,
                    )
                else:
                    view.clear_legend_placement(field)
        finally:
            self._recording = False

    def _view(self) -> Optional[Any]:
        try:
            return self._reconciler.view
        except Exception:
            return None


__all__ = ["PaneLegendBinding"]
