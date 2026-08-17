"""Instant — the ``(stage, step)`` time identity (ADR 0098 §7).

An instant names one recorded step of one stage. It is a pure value:
snapping a dragged scrubber/curve position to the *nearest recorded*
step needs the stage's time vector and happens in the clients
(realize / Qt), never here — the IR always stores a concrete recorded
step so two equal instants compare equal.

A mode-posed mesh view (``Deform.mode`` set) has **no** instant and is
frozen under the session time link; that law lives in
:meth:`~apeGmsh.results.session.ResultsSession.effective_instant`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instant:
    """One recorded time sample: ``(stage, step)``.

    ``stage`` is the stage id string (``Results`` ``StageInfo.id``);
    ``step`` is the 0-based recorded step index within that stage.
    Negative indices are rejected: an instant is a *name*, and letting
    ``-1`` alias the last step would give one instant two unequal names
    (poison for the linked-time comparison and the S5 snapshot).
    """

    stage: str
    step: int

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError(
                f"Instant.stage must be a non-empty stage id string; "
                f"got {self.stage!r}."
            )
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError(
                f"Instant.step must be an int step index; got "
                f"{self.step!r}."
            )
        if self.step < 0:
            raise ValueError(
                f"Instant.step must be >= 0 (no negative aliases); got "
                f"{self.step}."
            )


__all__ = ["Instant"]
