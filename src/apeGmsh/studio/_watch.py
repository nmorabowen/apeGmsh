"""File-watch debounce core (ADR 0095 Amendment 4 / S6b).

Pure: no Qt, no filesystem. The host's QTimer loop polls the entry
script's mtime (``_host.py``, ~1 s period) and feeds each observation
through :func:`watch_tick`; this module only decides when a changed
mtime has *settled* — the same value seen on two consecutive ticks.
A save that flaps the mtime across several ticks (editors that write
then touch, or write twice) yields exactly one settled observation
once it stops moving; an mtime that keeps changing never settles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatchState:
    """Debounce state between polls.

    ``last_seen`` is the mtime already acted on (seeded at watch
    start, updated when a change settles). ``pending`` is a candidate
    mtime observed on the previous tick that has not yet repeated.
    """

    last_seen: float | None = None
    pending: float | None = None


def seed_watch_state(mtime: float | None) -> WatchState:
    """Initial state for a fresh watch loop — the current mtime, no candidate."""
    return WatchState(last_seen=mtime, pending=None)


def watch_tick(state: WatchState, mtime: float | None) -> tuple[WatchState, bool]:
    """One poll observation. Returns ``(new_state, settled)``.

    ``settled`` is ``True`` exactly when *mtime* repeats unchanged
    from the previous tick and differs from ``last_seen`` — the file
    changed and then held still for one full tick. A missing file
    (``mtime is None``) drops any pending candidate without settling
    (a save that briefly removes-then-recreates the file does not
    fire on the gap).
    """
    if mtime is None:
        return WatchState(last_seen=state.last_seen, pending=None), False
    if mtime == state.last_seen:
        return WatchState(last_seen=state.last_seen, pending=None), False
    if state.pending == mtime:
        return WatchState(last_seen=mtime, pending=None), True
    return WatchState(last_seen=state.last_seen, pending=mtime), False
