"""FilterController — the dimensional select-mode owner (ADR 0045 S2).

One per viewer. The single, sticky source of truth for *which entity
dimensions are active* — the `0/1/2/3/4` keystroke contract and the
checkbox filter panel are **two front-ends writing the same frozenset**
(INV-4), consumed identically by every viewer. ``PickEngine._pickable_
dims`` becomes a *derived mirror* fed from here, not a second source of
truth.

Ratified keystroke semantics (ADR 0045 §Resolved decisions 1):
**multi-select by default** — a bare dim key *toggles* that dimension in
or out of the active set (:meth:`toggle`); the checkbox panel *replaces*
the set wholesale (:meth:`set_active`); ``4`` selects all
(:meth:`select_all`); :meth:`clear` empties it. There is no single-dim
REPLACE key mode.

Pure Python — no Qt, no VTK — so the state machine is fully unit-testable
headless. A viewer wires :meth:`on_change` to fan the active set out to
its three coupled effects (pick-resolution gate, actor pickability,
visibility feedback); the *policy* of those effects stays viewer-side,
only the *source of truth* lives here.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional


class FilterController:
    """Owns the active dimension set; notifies one ``on_change`` sink.

    Parameters
    ----------
    dims
        The dimensions this viewer can filter (e.g. ``[0, 1, 2, 3]`` for
        the model viewer, ``[1, 2, 3]`` for a typical mesh). A
        toggle/set of a dimension not in this list is ignored — a bare
        ``0`` keypress in a viewer with no dim-0 substrate is a harmless
        no-op.
    initial
        The initially-active set; defaults to **all** ``dims`` (matching
        the all-checked checkbox panel today). Values outside ``dims``
        are dropped.
    on_change
        ``fn(active: frozenset[int]) -> None`` invoked whenever the
        active set actually changes. The viewer's fan-out (set_pickable_
        dims + visibility feedback + panel sync) lives here. Public and
        settable, so a viewer that must build its checkbox panel before
        its pick engine exists can construct the controller first, point
        the panel at :meth:`set_active`, and attach the sink later.
    """

    def __init__(
        self,
        dims: Iterable[int],
        *,
        initial: Optional[Iterable[int]] = None,
        on_change: Optional[Callable[[frozenset], None]] = None,
    ) -> None:
        self._dims: list[int] = sorted({int(d) for d in dims})
        if initial is None:
            self._active: frozenset = frozenset(self._dims)
        else:
            self._active = frozenset(
                int(d) for d in initial if int(d) in self._dims
            )
        self.on_change = on_change

    # -- state ---------------------------------------------------------

    @property
    def dims(self) -> list[int]:
        """The filterable dimensions (sorted)."""
        return list(self._dims)

    @property
    def active(self) -> frozenset:
        """The currently-active dimension set."""
        return self._active

    def is_active(self, dim: int) -> bool:
        return int(dim) in self._active

    # -- mutators ------------------------------------------------------

    def set_active(self, dims: Iterable[int], *, emit: bool = True) -> None:
        """Replace the active set wholesale (the checkbox-panel path).

        Values outside :attr:`dims` are dropped. ``on_change`` fires only
        when the set actually changes (so a redundant set is a cheap
        no-op and cannot storm the fan-out)."""
        new = frozenset(int(d) for d in dims if int(d) in self._dims)
        if new == self._active:
            return
        self._active = new
        if emit and self.on_change is not None:
            self.on_change(self._active)

    def toggle(self, dim: int, *, emit: bool = True) -> None:
        """Flip one dimension in/out of the active set (the bare-key path,
        multi-select). A dimension not in :attr:`dims` is ignored."""
        d = int(dim)
        if d not in self._dims:
            return
        new = self._active - {d} if d in self._active else self._active | {d}
        self.set_active(new, emit=emit)

    def select_all(self) -> None:
        """Activate every filterable dimension (the ``4`` key / "All")."""
        self.set_active(self._dims)

    def clear(self) -> None:
        """Deactivate every dimension (the "None" button)."""
        self.set_active(())


__all__ = ["FilterController"]
