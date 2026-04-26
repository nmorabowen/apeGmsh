"""Time-slice resolution.

Converts a ``TimeSlice`` (None / int / list / slice / float) plus a
time vector into a 1-D ndarray of step indices. The reader uses
this to select time samples before reading from disk.

Slice semantics
---------------
- ``None`` → all steps.
- ``int`` → single step index (negative indexing supported).
- ``list[int]`` → explicit step indices.
- ``float`` → nearest step (``argmin(|t - target|)``).
- ``slice`` → mixed semantics following pandas convention:
    - integer endpoints → step indices, numpy half-open
    - float endpoints   → time-value search, half-open ``[start, stop)``
- A ``slice`` with mixed types (e.g. ``slice(0, 5.0)``) is invalid.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from .readers._protocol import TimeSlice


def resolve_time_slice(
    time_slice: "TimeSlice", time_vector: ndarray,
) -> ndarray:
    """Return a 1-D int64 ndarray of step indices into ``time_vector``."""
    n_steps = len(time_vector)

    if time_slice is None:
        return np.arange(n_steps, dtype=np.int64)

    # Bool-typed slices are caught by isinstance(int) — handle order matters.
    if isinstance(time_slice, bool):
        raise TypeError(
            f"Boolean is not a valid time_slice (got {time_slice!r})."
        )

    if isinstance(time_slice, int):
        idx = time_slice
        if idx < 0:
            idx += n_steps
        if not 0 <= idx < n_steps:
            raise IndexError(
                f"Step index {time_slice} out of range "
                f"(time vector has {n_steps} steps)."
            )
        return np.array([idx], dtype=np.int64)

    if isinstance(time_slice, float):
        if n_steps == 0:
            raise IndexError("Cannot resolve nearest step on empty time vector.")
        idx = int(np.argmin(np.abs(time_vector - time_slice)))
        return np.array([idx], dtype=np.int64)

    if isinstance(time_slice, list):
        arr = np.asarray(time_slice, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(
                "list time_slice must be 1-D step indices."
            )
        if arr.size and (arr.min() < 0 or arr.max() >= n_steps):
            raise IndexError(
                f"List time_slice contains out-of-range indices "
                f"(range 0..{n_steps - 1})."
            )
        return arr

    if isinstance(time_slice, slice):
        return _resolve_slice(time_slice, time_vector, n_steps)

    raise TypeError(
        f"Unsupported time_slice type: {type(time_slice).__name__}. "
        f"Must be None, int, list[int], slice, or float."
    )


def _resolve_slice(
    s: slice, time_vector: ndarray, n_steps: int,
) -> ndarray:
    start, stop, step = s.start, s.stop, s.step

    # Step must be int (or None).
    if step is not None and not isinstance(step, int):
        raise TypeError(
            f"slice step must be int or None (got {type(step).__name__})."
        )

    # Determine endpoint mode: integer (step indices) or float (time values).
    has_float = isinstance(start, float) or isinstance(stop, float)
    has_int = (
        (start is not None and not isinstance(start, float)
         and isinstance(start, int))
        or (stop is not None and not isinstance(stop, float)
            and isinstance(stop, int))
    )

    if has_float and has_int:
        raise TypeError(
            f"slice endpoints must be the same type "
            f"(got start={start!r}, stop={stop!r})."
        )

    if has_float:
        # Time-value search, half-open.
        i_start = (
            0 if start is None
            else int(np.searchsorted(time_vector, start, side="left"))
        )
        i_stop = (
            n_steps if stop is None
            else int(np.searchsorted(time_vector, stop, side="left"))
        )
        return np.arange(i_start, i_stop, step or 1, dtype=np.int64)

    # Integer endpoints → numpy index semantics.
    return np.arange(*s.indices(n_steps), dtype=np.int64)
