"""Generic two-column ``time accel`` parser.

Handles whitespace-separated ASCII files of the form::

    0.000000   -1.503467e-03
    0.005000   -1.499866e-03
    ...

Comment lines starting with ``#``, ``%``, ``//`` or ``;`` are skipped.
``dt`` is inferred from the time column; uniform and non-uniform
sampling are both accepted (the latter preserves the full time vector).

This is the format of the homogenised records under
``examples/Records/03_Selected/`` — PEER NGA, Chilean Maule 2010,
Italian Amatrice 2016, and 1985 Chilean records all share it.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .._motion import GroundMotion


_COMMENT_CHARS = ("#", "%", "/", ";")


def _strip_comments(path: str) -> list[str]:
    """Return non-comment, non-blank lines from *path*."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [
            line for line in fh
            if line.strip() and not line.lstrip().startswith(_COMMENT_CHARS)
        ]


def read_two_column(
    path: str | os.PathLike[str],
    *,
    scale_factor: float = 1.0,
    uniform_rtol: float = 1e-4,
) -> GroundMotion:
    """Parse a two-column ``time accel`` record file.

    Both uniform and non-uniform records are accepted; the parser
    inspects the time column and chooses the right representation.

    Parameters
    ----------
    path
        Path to the record file.
    scale_factor
        Multiplier applied to the acceleration column at read time.
        Use it for unit conversion (``9.81`` to push g → m/s²) or for
        amplitude scaling. Default ``1.0`` (values stored as-is).
    uniform_rtol
        Relative tolerance for treating a record as uniform
        (``max|Δt - mean(Δt)| / mean(Δt) ≤ uniform_rtol``).

    Returns
    -------
    A :class:`GroundMotion` snapshot.

    Raises
    ------
    ValueError
        If the file has fewer than two data rows, more than two
        columns, or a non-monotonic time column.
    """
    path_str = os.fspath(path)
    rows = _strip_comments(path_str)
    if len(rows) < 2:
        raise ValueError(
            f"{path_str}: need at least 2 data rows, found {len(rows)}"
        )
    data = np.loadtxt(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"{path_str}: expected 2 columns, got shape {data.shape}"
        )

    t = data[:, 0]
    accel = data[:, 1] * scale_factor

    dt_samples = np.diff(t)
    if np.any(dt_samples <= 0):
        raise ValueError(f"{path_str}: time column is not strictly increasing")
    dt_mean = float(np.mean(dt_samples))
    max_dev = float(np.max(np.abs(dt_samples - dt_mean)))
    is_uniform = (max_dev / dt_mean) <= uniform_rtol

    metadata: dict[str, Any] = {
        "format": "two_column",
        "t0": float(t[0]),
        "uniform": is_uniform,
        "scale_factor": scale_factor,
    }
    if not is_uniform:
        metadata["max_dt_deviation"] = max_dev
        metadata["dt_rel_deviation"] = max_dev / dt_mean

    return GroundMotion(
        accel=accel,
        dt=dt_mean,
        time=None if is_uniform else t,
        source=os.path.basename(path_str),
        metadata=metadata,
    )
