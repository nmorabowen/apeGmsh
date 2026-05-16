"""Generic one-column accel-only parser.

Reads a whitespace-separated ASCII file that contains acceleration
values only — either one per line or multiple per line (PEER-style
fixed-width also works as long as the header has been stripped).

``dt`` is *not* in the file and must be supplied explicitly.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .._motion import GroundMotion


_COMMENT_CHARS = ("#", "%", "/", ";")


def _strip_comments(path: str, skiprows: int) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        all_lines = list(fh)
    body = all_lines[skiprows:]
    return [
        line for line in body
        if line.strip() and not line.lstrip().startswith(_COMMENT_CHARS)
    ]


def read_one_column(
    path: str | os.PathLike[str],
    *,
    dt: float,
    scale_factor: float = 1.0,
    skiprows: int = 0,
) -> GroundMotion:
    """Parse a one-column (or PEER-style multi-per-line) accel file.

    Parameters
    ----------
    path
        Path to the record file.
    dt
        Time step in seconds. Required — not encoded in this format.
    scale_factor
        Multiplier applied to the acceleration values at read time.
        Default ``1.0`` (values stored as-is).
    skiprows
        Lines to discard at the top of the file before parsing data.
        Comment lines (``# % / ;``) are skipped automatically and do
        not need to be counted here.

    Returns
    -------
    A :class:`GroundMotion` snapshot.
    """
    path_str = os.fspath(path)
    rows = _strip_comments(path_str, skiprows=skiprows)
    if not rows:
        raise ValueError(f"{path_str}: no data rows found")

    flat: list[float] = []
    for line in rows:
        for tok in line.split():
            flat.append(float(tok))
    if len(flat) < 2:
        raise ValueError(
            f"{path_str}: need at least 2 samples, found {len(flat)}"
        )
    accel = np.asarray(flat, dtype=np.float64) * scale_factor

    metadata: dict[str, Any] = {
        "format": "one_column",
        "scale_factor": scale_factor,
    }
    return GroundMotion(
        accel=accel,
        dt=dt,
        source=os.path.basename(path_str),
        metadata=metadata,
    )
