"""SessionReconciler — session diff → backend ops → ONE coalesced render.

ADR 0098 S2. The session IR is truth; this module keeps one live
``RenderBackend`` painting it. A NEW reconciler under ADR 0084's guard
discipline — the Dispatcher stays with the old ontology and is never
touched; what is imitated is its *batching shape*: mutations mark
dirty and schedule ONE deferred flush (``QTimer.singleShot(0)``
idiom, injectable for tests), so an N-write gesture costs one
realize and exactly one ``backend.render()``.

The S2 diff protocol (plan decision 6, settled here):

* ``realize()`` stays ONE-SHOT (S1 contract): each flush tears down
  exactly what the previous flush emitted and re-realizes the
  projected pane into the live backend. The stable layer keys
  (``"mesh-1:contour"``) are the diff ledger; update-in-place by key
  is deliberately NOT the v1 protocol — scene-IR records carry numpy
  payloads with no cheap, well-defined equality, and a wrong
  "unchanged" verdict silently paints a stale picture (the worst
  ADR 0084 failure class). S4's sustained playback drives
  ``update_to_step`` through ``RealizedPane.diagrams`` instead of
  re-realizing; that widening rides these keys.
* Scalar-bar sync: teardown removes the previous flush's bar keys;
  realize re-registers bars through a fresh ``LegendController``
  bound to the live backend, measured against the live viewport.
  After a successful flush the reconciler ADOPTS that controller
  (``adopt_controller``) so ``controller_for(backend)`` resolves to
  the real bar owner, not the null realize installs (the S1 runway).
* Granular session events are NOT the v1 protocol —
  ``ResultsSession.subscribe`` is a bare change tick and the
  reconciler re-derives everything from ``realize()`` output.

Teardown is exact even when a realize fails mid-emission: the
reconciler wraps the backend in a :class:`LedgerBackend` that records
every layer handle and bar key that passes through it, so a partial
emission is swept on the next flush instead of leaking actors.

Error discipline (ADR 0084 D4): a flush failure must not kill the Qt
event loop, but must never be silent — it reports through
``viewers._failures`` as ``pump.session_reconcile``, which the
autouse ``pump_failures`` test fixture turns into a test failure and
the window surfaces on its status bar.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .._failures import report
from ..core._legend import adopt_controller
from ._realize import realize_pane

_Listener = Callable[[Optional[Any]], None]


def _default_defer(fn: Callable[[], None]) -> None:
    """Schedule ``fn`` on the Qt event loop; run inline without Qt.

    The Dispatcher's UI-lane idiom: ``QTimer.singleShot(0, fn)`` so a
    storm of session ticks inside one gesture coalesces into one
    flush on the next loop turn. Tests inject their own ``defer_fn``
    (immediate, or a queue they drain by hand) for determinism.
    """
    try:
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, fn)
    except Exception:
        fn()


class LedgerBackend:
    """Records exactly what this reconciler emitted onto ``inner``.

    A transparent forwarder (``__getattr__``) that additionally keeps
    the layer handles and scalar-bar keys added through it — the
    teardown ledger. Realize sees THIS object as its backend, so even
    a realize that raises mid-emission leaves an exact record of what
    reached the backend, and the next flush sweeps it instead of
    leaking actors it has no handles for.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.layer_handles: list[Any] = []
        self.bar_keys: list[str] = []

    @property
    def inner(self) -> Any:
        """The wrapped live backend."""
        return self._inner

    def add_layer(self, layer: Any) -> Any:
        handle = self._inner.add_layer(layer)
        self.layer_handles.append(handle)
        return handle

    def remove_layer(self, handle: Any) -> None:
        self._inner.remove_layer(handle)
        try:
            self.layer_handles.remove(handle)
        except ValueError:
            pass

    def add_scalar_bar(self, handle: Any, spec: Any) -> None:
        self._inner.add_scalar_bar(handle, spec)
        if spec.key not in self.bar_keys:
            self.bar_keys.append(spec.key)

    def remove_scalar_bar(self, bar_key: str) -> None:
        self._inner.remove_scalar_bar(bar_key)
        if bar_key in self.bar_keys:
            self.bar_keys.remove(bar_key)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class SessionReconciler:
    """Keeps one live backend painting one mesh pane of one session.

    Subscribes to the session's bare change tick; every tick marks
    dirty and schedules ONE deferred flush. A flush tears down the
    previous emission (by ledger), realizes the projected pane
    one-shot into the backend, adopts the realization's legend
    controller, then renders exactly once.

    ``pane_id=None`` projects the session's first mesh view resolved
    at flush time — the default one-empty-view session needs no
    addressing, and panes added or removed from Python retarget
    automatically. ``set_pane`` aims the projection (the outline's
    selection in the S2 window); a stale id falls back to the first
    mesh view rather than crashing the viewport.
    """

    def __init__(
        self,
        session: Any,
        backend: Any,
        *,
        pane_id: Optional[str] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> None:
        self._session = session
        self._ledger = LedgerBackend(backend)
        self._pane_id = pane_id
        self._defer = defer_fn or _default_defer
        self._dirty = False
        self._scheduled = False
        self._disposed = False
        self._realized: Optional[Any] = None
        self._listeners: list[_Listener] = []
        session.subscribe(self._on_tick)
        # Boot flush: the valid empty picture (§3) paints without any
        # session mutation having fired yet.
        self.schedule()

    # -- surface -------------------------------------------------------

    @property
    def backend(self) -> LedgerBackend:
        """The ledger the pane emits through (wraps the live backend)."""
        return self._ledger

    @property
    def realized(self) -> Optional[Any]:
        """The last flush's ``RealizedPane`` (``None`` before the first
        successful flush, after a failed one, or with nothing to
        project). The inspector reads per-key emissions off it to show
        an occupied-but-empty slot as "(nothing to draw)" — ADR §4."""
        return self._realized

    @property
    def pane_id(self) -> Optional[str]:
        return self._pane_id

    def set_pane(self, pane_id: Optional[str]) -> None:
        """Aim the projection at a pane (the outline's selection)."""
        if pane_id == self._pane_id:
            return
        self._pane_id = pane_id
        self.schedule()

    def add_listener(self, callback: _Listener) -> None:
        """Register a post-flush callback ``cb(realized_or_None)``.

        Listeners are projections (inspector state, camera fit) — they
        must not mutate the session from inside the callback path
        without an equality guard, or they schedule the next flush
        themselves.
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: _Listener) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def schedule(self) -> None:
        """Mark dirty and schedule one deferred flush (coalescing)."""
        if self._disposed:
            return
        self._dirty = True
        if self._scheduled:
            return
        self._scheduled = True
        self._defer(self._flush)

    def flush_now(self) -> None:
        """Run a flush synchronously if one is pending (test hook and
        the stills-of-a-live-window path)."""
        self._flush()

    def dispose(self) -> None:
        """Detach from the session. The backend dies with its window;
        no teardown is attempted here."""
        if self._disposed:
            return
        self._disposed = True
        self._session.unsubscribe(self._on_tick)
        self._listeners.clear()

    # -- internals -----------------------------------------------------

    def _on_tick(self) -> None:
        self.schedule()

    def _resolve_pane(self) -> Optional[Any]:
        from apeGmsh.results.session import MeshView

        if self._pane_id is not None:
            try:
                pane = self._session.pane(self._pane_id)
            except KeyError:
                pane = None
            if isinstance(pane, MeshView):
                return pane
            # Stale or non-mesh target: fall back, don't crash the
            # viewport — the outline re-aims on its next selection.
            self._pane_id = None
        for pane in self._session.panes:
            if isinstance(pane, MeshView):
                return pane
        return None

    def _teardown(self) -> None:
        """Remove exactly what this reconciler emitted, by ledger."""
        for bar_key in list(self._ledger.bar_keys):
            self._ledger.remove_scalar_bar(bar_key)
        for handle in list(self._ledger.layer_handles):
            self._ledger.remove_layer(handle)
        self._realized = None

    def _flush(self) -> None:
        self._scheduled = False
        if self._disposed or not self._dirty:
            return
        self._dirty = False
        try:
            self._teardown()
            pane = self._resolve_pane()
            if pane is not None:
                self._realized = realize_pane(
                    self._session, pane, self._ledger,
                )
                controller = self._realized.legend_controller
                if controller is not None:
                    # The S1 runway: realize adopts a null controller
                    # for its backend; the pane must own its real
                    # legend owner or controller_for() answers null.
                    adopt_controller(self._ledger, controller)
        except Exception as exc:
            # The viewport must survive (the ledger swept, or sweeps
            # on the next flush, whatever was partially emitted) but
            # the failure is never silent: pump_failures fails the
            # test, the window surfaces it on the status bar.
            self._realized = None
            report("pump.session_reconcile", exc)
        try:
            self._ledger.render()
        except Exception as exc:
            report("pump.session_render", exc)
        for callback in list(self._listeners):
            try:
                callback(self._realized)
            except Exception as exc:
                report("pump.session_listener", exc)


__all__ = ["LedgerBackend", "SessionReconciler"]
