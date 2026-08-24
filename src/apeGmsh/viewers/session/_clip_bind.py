"""Give one pane's section planes their gizmos back.

ADR 0098 Amendment 7 (R1). The sibling of :mod:`._legend_bind`, and it
is a module for the same reason: three facts make this more than one
call in ``MeshPane.__init__``.

* **The gizmo layers must NOT go through the reconciler's ledger.**
  Realize emits through a ``LedgerBackend``, and ``_teardown`` removes
  **every** handle that ledger recorded. A gizmo added through it would
  be swept on the next slot edit — the actors would vanish while the
  plane stayed. So the renderer is built on ``ledger.inner``, the raw
  backend, and the gizmo outlives realize by construction. This is the
  one piece of the design that cannot be discovered from the gizmo
  code, only from the reconciler's.

* **The interactor's priority-12 seat is order-sensitive, and realize
  disturbs it.** VTK invokes same-priority observers in insertion
  order, and ADR 0083 relies on that: the legend interactor is
  installed FIRST so a colour bar overlapping a gizmo wins the press.
  But ``_realize_legends`` builds a fresh ``LegendController`` every
  full realize, so :class:`PaneLegendBinding` uninstalls and
  re-installs its observers — landing them *after* ours. Left alone,
  the gizmo would silently start taking presses over the bar from the
  first slot edit onward. :meth:`reseat` re-installs ours behind the
  legend's, and is called after every legend re-bind.

* **The renderer is a reconciler of its own.** ``refresh()`` diffs
  desired gizmos against drawn ones and uses ``update_layer``'s
  in-place fast path for same-topology moves, which is what keeps a
  drag from rebuilding actors (the ADR 0081 L2 flicker lesson). So the
  renderer must be LONG-LIVED — one per pane, surviving realize —
  rather than rebuilt per flush like the legend controller. Handles are
  strong refs by contract (``_clip_gizmo.py:376``), and this object is
  what holds them.

The controller is :class:`~._clip_control.ViewClipController`, so every
gesture writes ``MeshView.set_clip`` and a dragged plane is session
state — it survives realize, snapshot and restore with no work here.
"""
from __future__ import annotations

from typing import Any, Optional


class PaneClipBinding:
    """Own one pane's clip controller, gizmo renderer and interactor.

    Idempotent where it matters: :meth:`refresh` is called on every
    flush, including a re-step frame, and must be cheap when nothing
    about the planes moved.
    """

    def __init__(self, backend: Any, reconciler: Any) -> None:
        # The RAW backend, deliberately — see the module docstring.
        # ``LedgerBackend`` exposes its wrapped target as ``inner``;
        # a plain backend is its own.
        self._backend = getattr(backend, "inner", backend)
        self._reconciler = reconciler
        self._controller: Any = None
        self._gizmos: Any = None
        self._interactor: Any = None
        self._view: Any = None
        self._bounds: Optional[tuple] = None

    # -- introspection (tests and the inspector) -----------------------

    @property
    def controller(self) -> Any:
        """The five-member clip contract over this pane's view."""
        return self._controller

    @property
    def gizmos(self) -> Any:
        """The renderer, or ``None`` before the first refresh."""
        return self._gizmos

    @property
    def interactor(self) -> Any:
        """The installed gesture handler, or ``None`` on a headless /
        offscreen backend that has no interactor to observe."""
        return self._interactor

    # -- the flush hook ------------------------------------------------

    def refresh(self, realized: Optional[Any]) -> None:
        """Reconcile the gizmos against the view. Called every flush.

        ``realized`` supplies the reference bounds the quad is sized
        from. A pane that has realized nothing, or whose realize could
        not resolve bounds, draws no gizmo — rather than guessing a box
        and putting a grab handle somewhere the model is not.
        """
        view = self._current_view()
        bounds = getattr(realized, "reference_bounds", None)
        if view is None or bounds is None:
            self.clear()
            return

        if self._controller is None or view is not self._view:
            # A pane retargeted at another view (``set_pane``) must not
            # keep cutting with the old one's planes.
            self._rebuild(view, bounds)
        elif bounds != self._bounds:
            # The model's reference extent changed under us — a scope
            # flip, or the first realize after a boot. The renderer
            # bakes the bbox in at construction, so it is replaced
            # rather than mutated.
            self._rebuild(view, bounds)

        if self._gizmos is not None:
            try:
                self._gizmos.refresh()
            except Exception:
                pass

    def reseat(self) -> None:
        """Re-install our observers so they sit BEHIND the legend's.

        Called after the legend binding re-binds, which it does on every
        full realize (``_realize_legends`` builds a new controller each
        time). Without this the gizmo drifts to the front of the
        priority-12 insertion order and starts winning presses over a
        colour bar the user can plainly see — a defect that would only
        appear *after the first slot edit*, which is exactly the kind
        that survives manual testing.
        """
        if self._interactor is None:
            return
        try:
            self._interactor.uninstall()
            self._interactor.install()
        except Exception:
            pass

    # -- lifecycle -----------------------------------------------------

    def _rebuild(self, view: Any, bounds: tuple) -> None:
        from ..core._clip_gizmo import ClipGizmoRenderer
        from ..core._clip_gizmo_interactor import install_clip_gizmo_interactor
        from ._clip_control import ViewClipController

        self.clear()
        self._view = view
        self._bounds = bounds
        self._controller = ViewClipController(view)
        self._gizmos = ClipGizmoRenderer(
            self._backend, self._controller, bbox=bounds,
        )
        # Returns None for a backend with no interactor (headless,
        # offscreen); the pane then simply has no gesture, which is a
        # supported state, not a failure.
        self._interactor = install_clip_gizmo_interactor(
            self._backend, self._controller, self._gizmos,
        )
        if self._interactor is not None:
            # A drag mutates the view, which ticks the session, which
            # flushes — so the picture already follows. What a release
            # needs is a repaint of anything that recomputes on the
            # gesture BOUNDARY rather than per tick (ADR 0083 S3's
            # cut-face pass), and a render for the final frame.
            self._interactor.on_drag_end = self._on_drag_end

    def clear(self) -> None:
        """Drop the gizmo layers and the observers. Idempotent."""
        if self._interactor is not None:
            try:
                self._interactor.uninstall()
            except Exception:
                pass
            self._interactor = None
        if self._gizmos is not None:
            try:
                self._gizmos.clear()
            except Exception:
                pass
            self._gizmos = None
        self._controller = None
        self._view = None
        self._bounds = None

    def dispose(self) -> None:
        """Called from ``MeshPane.dispose``, for the reason the pick and
        legend installations are: these observers live on the
        interactor the pane is about to close, and the layers live in
        the context it is about to release."""
        self.clear()

    # -- internals -----------------------------------------------------

    def _on_drag_end(self) -> None:
        try:
            self._backend.render()
        except Exception:
            pass

    def _current_view(self) -> Optional[Any]:
        try:
            return self._reconciler.view
        except Exception:
            return None


__all__ = ["PaneClipBinding"]
