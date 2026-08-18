"""ResultsSession — the presentation document (ADR 0098 §1, §7).

"What is on screen, at which time, with which pictures, with which
pick." Panes, one time link, one selection. VTK actors, docks and
widgets are not truth; the session is — the Qt client (S2) projects
it, ``realize()`` (S1) turns it into scene IR, the snapshot (S5)
serialises it.

Observer surface: ``subscribe(cb)`` — a bare change tick, fired once
per mutating gesture anywhere in the session (session fields, view
mutators, slot fills, selection writes). The S2 reconciler coalesces
on top of this; granular events are deliberately NOT the v1 protocol.
Ticks are loud: a subscriber that raises propagates to the mutator's
caller (state is already committed) — no silent pump (ADR 0084).
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

from ._selection import SessionSelection
from ._time import Instant
from ._views import (
    MeshView,
    PlotSeries,
    PlotSource,
    PlotView,
    _next_counter,
)

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results

Pane = Union[MeshView, PlotView]


class ResultsSession:
    """The presentation library's root object.

    Constructing one gives you presentation with no window (§1);
    ``render()`` / ``realize()`` (S1) and ``show()`` (S2) are clients.
    A session of zero panes is valid IR — the ``results.session()``
    factory owns the default-one-empty-mesh-view sugar (and the
    ``results=`` binding the realize clients need).
    """

    def __init__(self, results: "Optional[Results]" = None) -> None:
        self._results = results
        self._panes: list[Pane] = []
        self._ids = itertools.count(1)
        self._time: Optional[Instant] = None
        self._time_linked: bool = True
        self._subscribers: list[Callable[[], None]] = []
        self._selection = SessionSelection(on_changed=self._tick)

    @property
    def results(self) -> "Optional[Results]":
        """The data broker this session presents (§1: ``Results``
        answers "what did the solver write?"; the session answers what
        is on screen). ``None`` for a bare IR-only session — realize
        and render then refuse loudly."""
        return self._results

    # -- panes ---------------------------------------------------------

    @property
    def panes(self) -> tuple[Pane, ...]:
        """Every pane, in creation order (the outline's panes group)."""
        return tuple(self._panes)

    def pane(self, pane_id: str) -> Pane:
        """The pane with this stable id. A miss is a ``KeyError`` — the
        id is how render / MCP / snapshot address a pane, so a stale id
        must fail loudly, never guess."""
        for pane in self._panes:
            if pane.id == pane_id:
                return pane
        raise KeyError(
            f"No pane {pane_id!r} in this session (have: "
            f"{[p.id for p in self._panes]})."
        )

    def add_view(self, name: Optional[str] = None) -> MeshView:
        """A new mesh view, booted to the valid empty picture (§3)."""
        view = MeshView(
            pane_id=f"mesh-{next(self._ids)}",
            name=name,
            on_changed=self._tick,
        )
        self._panes.append(view)
        self._tick()
        return view

    def add_plot(
        self,
        kind: str = "history",
        series: Sequence[PlotSeries] = (),
        name: Optional[str] = None,
    ) -> PlotView:
        """A new plot pane (§6)."""
        plot = PlotView(
            pane_id=f"plot-{next(self._ids)}",
            kind=kind,
            series=series,
            name=name,
            on_changed=self._tick,
        )
        self._panes.append(plot)
        self._tick()
        return plot

    def add_plot_from_selection(
        self,
        quantity: str,
        kind: str = "history",
        name: Optional[str] = None,
    ) -> PlotView:
        """The "select → New plot" law (§6/§8): the selection set IS
        the source. Membership is COPIED into concrete node/gauss
        sources at creation — mutating the selection afterwards does
        not touch the plot (a live alias is not v1). An empty selection
        has nothing to plot and refuses loudly."""
        sel = self._selection
        if sel.kind == "nodes":
            sources = [PlotSource.node(i) for i in sel.nodes]
        elif sel.kind == "gauss":
            sources = [PlotSource.gauss(e, gp) for e, gp in sel.gauss]
        else:
            raise ValueError(
                "Selection is empty — select nodes or Gauss points "
                "before creating a plot from the selection."
            )
        series = tuple(
            PlotSeries(source=s, quantity=quantity) for s in sources
        )
        return self.add_plot(kind=kind, series=series, name=name)

    def remove_pane(self, pane_id: str) -> None:
        pane = self.pane(pane_id)
        self._panes.remove(pane)
        # Detach the notify backref: a removed pane a caller still holds
        # must neither tick this session (phantom repaints in S2) nor
        # keep it alive.
        pane._notify = None
        self._tick()

    # -- snapshot restore (S5a) ----------------------------------------

    def _adopt_pane(self, pane: Pane) -> None:
        """Append a pane restored from a snapshot, keeping ITS id.

        ``add_view`` / ``add_plot`` mint an id; a restore must keep the
        one the file names, because that id is how the snapshot, the
        outline and the MCP verb address the pane. Wiring ``_notify``
        here is what makes the restored pane a real member of this
        session rather than a detached copy that never repaints.
        """
        pane._notify = self._tick
        self._panes.append(pane)

    def _reseed_ids(self) -> None:
        """Re-seed the pane-id counter past every restored pane id.

        TWO counters must be re-seeded by a restore, not one: this one
        (``mesh-N`` / ``plot-N``, both drawn from the single sequence)
        and each mesh view's own ``clip-N`` counter, in
        ``MeshView._adopt_clips``. Restore a session holding ``mesh-3``,
        add a view, and without this you get a SECOND ``mesh-1`` — two
        panes with one id, and every id-addressed client picks whichever
        comes first.
        """
        self._ids = _next_counter(
            [pane.id for pane in self._panes], "mesh-", "plot-",
        )

    # -- time (§7) -----------------------------------------------------

    @property
    def time(self) -> Optional[Instant]:
        """The session instant. While the link is on this is THE
        instant of every pane (§7)."""
        return self._time

    @time.setter
    def time(self, value: Optional[Instant]) -> None:
        if value is not None and not isinstance(value, Instant):
            raise TypeError(
                f"ResultsSession.time takes an Instant or None; got "
                f"{type(value).__name__}."
            )
        if value == self._time:
            return
        self._time = value
        self._tick()

    @property
    def time_linked(self) -> bool:
        """Linked: one instant — scrubber, every mesh view, every plot
        cursor. Unlinked: each pane keeps its own. If the link is on
        and moving the plot does not move the meshes, the link is a
        lie (§7)."""
        return self._time_linked

    @time_linked.setter
    def time_linked(self, value: bool) -> None:
        value = bool(value)
        if value == self._time_linked:
            return
        self._time_linked = value
        self._tick()

    def effective_instant(self, pane: "Pane | str") -> Optional[Instant]:
        """THE instant this pane is at — the §7 law in one place.

        * A mode-posed mesh view has no instant, linked or not: the
          scrubber moves instants and a mode has none (§4/§7).
        * Link on → the session instant, for every other pane; pane
          time is ignored (§9 — set it here; the link ignores it).
        * Link off → the pane's own ``time`` / ``cursor``.
        """
        if isinstance(pane, str):
            pane = self.pane(pane)
        if isinstance(pane, MeshView):
            if pane.is_mode_posed:
                return None
            return self._time if self._time_linked else pane.time
        return self._time if self._time_linked else pane.cursor

    # -- selection (§8) ------------------------------------------------

    @property
    def selection(self) -> SessionSelection:
        """The one selection set — nodes XOR Gauss, last writer wins.
        The owner-facing surface of the one ``SelectionState`` (0045
        INV-5), never a second store."""
        return self._selection

    # -- realize clients (S1) ------------------------------------------

    def realize(
        self, backend: Any = None, pane: "Pane | str | None" = None,
    ):
        """Realize one pane (S1).

        A **mesh** pane is one-shot: it emits its complete layer set
        into ``backend`` (an ADR 0042 ``RenderBackend``) and returns a
        ``RealizedPane`` whose layers carry stable keys — the S2
        reconciler's diff surface. A **plot** pane resolves its series
        to arrays and returns a ``RealizedPlot``; it needs no backend
        (§6: its client draws the numbers). With one pane no id is
        needed; otherwise address by pane id.

        The projection lives in ``apeGmsh.viewers.session`` (imported
        lazily here) — the session package itself stays free of the
        diagram/Qt/VTK machinery per the S0 purity guard.
        """
        from apeGmsh.viewers.session import realize_pane, resolve_pane

        return realize_pane(self, resolve_pane(self, pane), backend)

    def render(
        self,
        path: "str | Path",
        pane: "Pane | str | None" = None,
        *,
        camera: Optional[str] = None,
        window_size: tuple[int, int] = (1280, 720),
    ) -> Optional[Path]:
        """Write one offscreen still of a pane (§1). Path or ``None``.

        Same skip discipline as ``results.render`` (ADR 0094):
        ``APEGMSH_SKIP_VIEWER=1`` or no GL context prints the
        ``[skip viewer]`` notice, writes no file and returns ``None``.
        ``camera=`` defaults to ``xy`` for a planar model, ``iso``
        otherwise.
        """
        from apeGmsh.viewers.session import render_still

        return render_still(
            self, pane, path, camera=camera, window_size=window_size,
        )

    def show(
        self, *, blocking: bool = True, title: "Optional[str]" = None,
    ):
        """Open the Qt client on this session (ADR 0098 §1, S2).

        The window is a projection: the outline lists the panes, the
        inspector edits the selected one through the §9 Add / change /
        clear loop, and the viewport reconciles the selected mesh view
        — every gesture writes THIS session, same as a script would.
        N tiled panes arrive with S3's pane host; until then one mesh
        view shows at a time.

        ``blocking=False`` presents the window on an already-running
        Qt loop and returns immediately. ``APEGMSH_SKIP_VIEWER=1``
        prints the standard skip notice and returns ``None``. Returns
        the ``SessionWindow`` handle otherwise.

        The Qt client lives in ``apeGmsh.viewers.session`` (imported
        lazily here) — the session package itself stays free of the
        Qt/VTK machinery per the S0 purity guard.
        """
        from apeGmsh.viewers.session import show_session

        return show_session(self, blocking=blocking, title=title)

    # -- snapshot clients (S5) -----------------------------------------

    def snapshot(self) -> dict:
        """This session as a JSON-safe dict (ADR 0098 §11 S5).

        Panes, slots, pose, the time link AND every pane's own instant,
        the one selection set. Nothing derived (legends are a function
        of the slots, §5) and nothing about a window: the snapshot is
        the document, so an agent can draw a still of what a human
        arranged without Qt.
        """
        from ._snapshot import snapshot

        return snapshot(self)

    def save_snapshot(self, path: "str | Path | None" = None) -> Path:
        """Write :meth:`snapshot` atomically; returns the path written.

        ``path=None`` defaults to ``<results>.session.json`` beside the
        results file — deliberately NOT the old viewer's
        ``<results>.viewer-session.json``, which today's window still
        owns and overwrites on close (plan decision 11). The paths
        merge at the S6 flip, behind the ``.legacy`` rename-aside.
        """
        from ._snapshot import save_snapshot

        return save_snapshot(self, path)

    # -- observer surface ----------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Register a change-tick subscriber (no payload — the v1
        protocol; the S2 reconciler diffs realize() output instead of
        consuming granular events)."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _tick(self) -> None:
        for callback in list(self._subscribers):
            callback()

    def __repr__(self) -> str:
        return (
            f"<ResultsSession {len(self._panes)} panes, "
            f"linked={self._time_linked}, time={self._time}>"
        )


__all__ = ["ResultsSession", "Pane"]
