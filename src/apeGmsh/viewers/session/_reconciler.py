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
* **One gesture, one realize** (ADR 0098 Amendment 1 caution 4,
  criterion 12 — a GATE, not a target). The tick is session-wide, so
  with N panes one slot fill would otherwise cost N realizes. Each
  reconciler therefore keeps a **pane-level signature** over exactly
  the inputs ``realize_pane`` reads — the pane identity, its effective
  instant (§7, which folds in ``session.time`` / ``time_linked``), its
  scope, pose, style, overlay, clips, occupied slots, derived
  legends, and (S4-1) the §8 selection membership whenever a glyph
  button is on. Every one of those is a frozen record, a scalar, or a
  tuple of frozen targets, so equality is cheap and total. An
  unchanged signature skips the flush outright: no realize, no
  ``render()``. The escape hatch is
  :meth:`schedule_forced` — the theme / density repaint, whose input
  is a palette the session cannot see, so it must not be compared
  away.

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
from ._chart import _same_stage
from ._realize import can_restep, realize_pane, restep_pane

_Listener = Callable[[Optional[Any]], None]

#: Sentinel for "no flush has run yet" — distinct from any real
#: signature, including the ``None`` a removed pane produces.
_NO_SIGNATURE = object()


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
    automatically.

    An EXPLICIT ``pane_id`` is a hard binding: a stale or non-mesh id
    paints **nothing**. S2 could fall back to the first mesh view
    because there was one viewport and the outline re-aimed it on the
    next selection; under a pane host that fallback is a hazard
    (Amendment 1 caution 10) — the pane closing on this very tick
    would repaint view 1 into its dying backend, and a mis-wired pane
    would mirror pane 1 instead of failing loudly.
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
        self._bound = pane_id is not None
        self._defer = defer_fn or _default_defer
        self._dirty = False
        self._scheduled = False
        self._disposed = False
        self._forced = False
        self._signature: Any = _NO_SIGNATURE
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
    def view(self) -> Optional[Any]:
        """The ``MeshView`` this reconciler currently paints, or ``None``.

        Resolved on every read rather than cached, for the reason
        ``_resolve_pane`` exists: a bound pane whose view is gone must
        answer ``None``, not the view it used to have."""
        return self._resolve_pane()

    @property
    def pane_id(self) -> Optional[str]:
        return self._pane_id

    def set_pane(self, pane_id: Optional[str]) -> None:
        """Aim the projection at another pane.

        Vestigial under a pane host (Amendment 1 caution 3): each pane
        is constructed with its own id and never re-aimed — the click
        selects, it does not re-point a shared viewport. Kept because
        the unbound (``pane_id=None``) reconciler is still the
        no-window realize path's shape, and tests aim it.
        """
        if pane_id == self._pane_id:
            return
        self._pane_id = pane_id
        self._bound = pane_id is not None
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

    def schedule_forced(self) -> None:
        """Schedule a flush that skips the signature guard.

        For changes the session cannot see — the theme palette and the
        density tokens realize bakes into its layers. Criterion 12's
        other half: a theme change costs every pane a repaint, and it
        must, because nothing in the session IR moved.
        """
        self._forced = True
        self.schedule()

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
            if self._bound:
                # Amendment 1 caution 10: a bound pane whose view is
                # gone paints NOTHING. Falling back to the first mesh
                # view is right for one shared viewport and wrong for a
                # host — a pane closing on this tick would repaint view
                # 1 into its dying backend, and a mis-wired pane would
                # mirror pane 1 instead of failing.
                return None
            self._pane_id = None
        for pane in self._session.panes:
            if isinstance(pane, MeshView):
                return pane
        return None

    def _pane_signature(self, pane: Optional[Any]) -> Any:
        """Everything ``realize_pane`` reads off the session, split into
        ``(structure, cursor)`` — the criterion-12 gate, and since
        ADR 0098 A4.2 the fast-path gate too.

        ``structure`` is what the picture is BUILT from; ``cursor`` is
        where in time it is shown. A change to the cursor alone can be
        re-stepped in place; anything in ``structure`` re-realizes.
        Same shape as ``PlotReconciler._signature_of``, which has split
        this way since S4-2.

        ``deform`` sits in ``structure`` on purpose: a scale change is
        rare, and a full realize is the conservative answer. So does
        the selection term — the S4-1 defect (a selection write ticks
        the session, the signature compares equal, and the repaint the
        user just asked for is skipped) stays fixed.

        Deliberately NOT a hash: these are small frozen records and
        tuples, so ``==`` is exact. A hash could collide and a
        collision here paints a stale picture — the worst ADR 0084
        failure class.
        """
        if pane is None:
            return None
        return (
            (
                pane.id,
                pane.scope,
                pane.deform,
                pane.style,
                pane.overlay,
                pane.clips,
                tuple(pane.slots.items()),
                pane.legends(),
                self._selection_term(pane),
            ),
            self._session.effective_instant(pane),
        )

    def _selection_term(self, pane: Any) -> Any:
        """The §8 selection, as this pane's signature reads it.

        The selection highlight is realize OUTPUT (S4-1), so without a
        term here a selection write ticks the session, the signature
        compares equal, and the repaint the user just asked for is
        skipped — the gate working exactly as designed, against the
        picture. The term is the MEMBERSHIP, not a count: two different
        five-node sets must not compare equal.

        Read only when a glyph button is on, because that is when
        realize reads it: with both off there is no cloud to mark, so
        an outline "select all" costs this pane nothing (§8 —
        "query-from-outline may fill the set with glyphs off").

        Cost is O(set size) per flush attempt — a 100k-target gesture
        builds a 100k tuple on every pane that could show it. That is
        the price of an EXACT comparison, and it stays well under the
        realize it gates; a cheaper proxy (a count, a log revision)
        would either collide or reach into the store's internals, and
        a collision here paints a stale picture.
        """
        if not (pane.style.nodes or pane.style.gauss):
            return ()
        return tuple(self._session.selection.targets)

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
        forced, self._forced = self._forced, False
        try:
            pane = self._resolve_pane()
            signature = self._pane_signature(pane)
        except Exception as exc:
            # Signature composition reads the session; a failure there
            # is a real failure, not a reason to skip silently.
            self._signature = _NO_SIGNATURE
            report("pump.session_reconcile", exc)
            return
        previous = self._signature
        if (
            not forced
            and signature == previous
            and previous is not _NO_SIGNATURE
        ):
            # Nothing this pane draws moved — criterion 12: the other
            # panes of an N-pane session cost nothing when one pane's
            # slot is filled.
            return
        self._signature = signature

        # ADR 0098 A4.2 — the cursor moved and NOTHING else did: re-step
        # this pane in place rather than tearing it down and rebuilding
        # it. Guarded by ``forced`` (theme / density rebuild the colours
        # realize bakes in, which the session cannot see) and by the
        # STAGE, because the diagrams' cached arrays are a function of
        # the stage — the same limit ``_same_stage`` already enforces
        # for plots. Any doubt falls through to the full path below.
        restepped = False
        if (
            not forced
            and pane is not None
            and previous is not _NO_SIGNATURE
            and previous is not None
            and signature is not None
            and signature[0] == previous[0]
            and _same_stage(signature[1], previous[1])
            and can_restep(self._realized, pane)
        ):
            try:
                self._realized = restep_pane(
                    self._session, pane, self._realized, self._ledger,
                )
            except Exception as exc:
                # Never leave a HALF re-stepped pane on screen: report,
                # then fall through to the full realize in THIS flush.
                # ``restep_pane`` only updates layers in place, so a
                # failed one leaves no orphaned handle behind.
                report("pump.session_restep", exc)
            else:
                restepped = True

        try:
            if not restepped:
                self._teardown()
                if pane is not None:
                    self._realized = realize_pane(
                        self._session, pane, self._ledger,
                    )
                    controller = self._realized.legend_controller
                    if controller is not None:
                        # The S1 runway: realize adopts a null
                        # controller for its backend; the pane must own
                        # its real legend owner or controller_for()
                        # answers null.
                        adopt_controller(self._ledger, controller)
        except Exception as exc:
            # The viewport must survive (the ledger swept, or sweeps
            # on the next flush, whatever was partially emitted) but
            # the failure is never silent: pump_failures fails the
            # test, the window surfaces it on the status bar. The
            # signature is dropped too — a failed flush must be RETRIED
            # by the next tick, never skipped as "unchanged".
            self._realized = None
            self._signature = _NO_SIGNATURE
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
