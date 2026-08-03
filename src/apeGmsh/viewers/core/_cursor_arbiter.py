"""One owner for the window cursor, shared by the hover interactors.

Two priority-12 interactors now hover on the same ``MouseMoveEvent``
and neither aborts on a miss (that is the whole point — a miss must
leave picking and navigation untouched), so **both** run on every
move and both used to write ``SetCurrentCursor`` directly. That is
one resource with two owners, and it failed exactly the way ADR 0056
says it must (S2 review finding C2):

* hover a gizmo — the gizmo sets a translate cursor and remembers it;
* slide straight onto a legend — the legend sets its move cursor, then
  the gizmo's handler runs (it is installed second), misses, and resets
  the window to the default, discarding what the legend just asked for;
* the next move writes nothing at all, because each interactor's
  "only write when it changes" memo now matches. The legend keeps a
  default cursor until the pointer leaves and comes back.

So the memo moves here, off the individual interactors, and becomes a
**request** per requester. The arbiter applies the winner: the
highest-priority requester with a live request, or the default when
nobody wants anything.
"""
from __future__ import annotations

from typing import Any, Optional
from weakref import WeakKeyDictionary


#: Requester priority, highest first — declared, not inferred from who
#: happened to hover first, so an overlap always resolves the way the
#: *press* does (the legend interactor installs first and therefore
#: wins a same-priority press; ADR 0083 Consequences). An unknown token
#: sorts last rather than raising.
_PRIORITY = ("legend", "clip_gizmo", "scope_gizmo")

#: ``window -> {token: cursor_constant}``. Weak on the render window so
#: a closed viewer's entry disappears with it.
_REQUESTS: "WeakKeyDictionary[Any, dict[str, Optional[int]]]" = (
    WeakKeyDictionary()
)


def request_cursor(window: Any, token: str, cursor: Optional[int]) -> None:
    """Ask for ``cursor`` on ``window``; ``None`` withdraws the request.

    Applies whichever live request was registered first, so a hover
    handler that misses can never erase the cursor another handler is
    still asking for. Non-fatal throughout — a cursor is feedback, and
    failing to set one must never break a gesture.
    """
    if window is None:
        return
    try:
        requests = _REQUESTS.get(window)
        if requests is None:
            requests = {}
            _REQUESTS[window] = requests
    except TypeError:          # a window that cannot be weak-referenced
        return

    if token in requests and requests[token] == cursor:
        return
    requests[token] = cursor

    def _rank(item: "tuple[str, Optional[int]]") -> int:
        name = item[0]
        return _PRIORITY.index(name) if name in _PRIORITY else len(_PRIORITY)

    winner = next(
        (c for _t, c in sorted(requests.items(), key=_rank) if c is not None),
        None,
    )
    try:
        import vtk

        window.SetCurrentCursor(
            winner if winner is not None else vtk.VTK_CURSOR_DEFAULT,
        )
    except Exception:
        pass


__all__ = ["request_cursor"]
