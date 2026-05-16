"""Format sniffer for :class:`GroundMotion.from_file`.

The sniffer inspects the head of a file and returns one of:

- ``"peer_at2"`` — PEER NGA AT2 (line 1 carries the PEER banner)
- ``"itaca"`` — ITACA / ESM keyword-value header
- ``"two_column"`` — generic ``time accel`` rows
- ``"one_column"`` — generic accel-only (requires ``dt`` from caller)
- ``"unknown"`` — no first-party parser matches

``from_file()`` then dispatches to the matching reader, or delegates
to obspy if installed.
"""
from __future__ import annotations

import os
from typing import Literal

from ._motion import GroundMotion
from ._parsers import (
    is_itaca,
    is_peer_at2,
    read_itaca,
    read_obspy,
    read_one_column,
    read_peer_at2,
    read_two_column,
)


FormatTag = Literal["peer_at2", "itaca", "two_column", "one_column", "unknown"]

_COMMENT_CHARS = ("#", "%", "/", ";")


def _first_data_line(path: str) -> str | None:
    """Return the first non-blank, non-comment line — or None."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(_COMMENT_CHARS):
                    continue
                return stripped
    except OSError:
        return None
    return None


def _column_count(line: str) -> int | None:
    """Number of numeric tokens on *line*, or None if any non-numeric."""
    toks = line.split()
    if not toks:
        return 0
    try:
        for t in toks:
            float(t)
    except ValueError:
        return None
    return len(toks)


def sniff_format(path: str | os.PathLike[str]) -> FormatTag:
    """Identify a record's format from header content alone."""
    path_str = os.fspath(path)

    if is_peer_at2(path_str):
        return "peer_at2"
    if is_itaca(path_str):
        return "itaca"

    first = _first_data_line(path_str)
    if first is None:
        return "unknown"
    cols = _column_count(first)
    if cols == 2:
        return "two_column"
    if cols == 1:
        return "one_column"
    if cols is not None and cols > 2:
        return "one_column"
    return "unknown"


def from_file(
    path: str | os.PathLike[str],
    *,
    scale_factor: float = 1.0,
    dt: float | None = None,
    obspy_component: int = 0,
) -> GroundMotion:
    """Identify the format of *path* and parse it.

    Parameters
    ----------
    path
        Record file.
    scale_factor
        Multiplier applied to the acceleration values. Default ``1.0``.
    dt
        Time step. **Required** for ``one_column`` records (and
        PEER-like no-header multi-per-line data that the sniffer
        reports as such). Ignored for formats that encode their own
        ``dt``.
    obspy_component
        Forwarded to :func:`read_obspy` when delegating. Default ``0``
        picks the first trace.

    Returns
    -------
    A :class:`GroundMotion` snapshot.

    Raises
    ------
    ValueError
        If ``dt`` is missing for a one-column record.
    ImportError
        If the format is unknown to the first-party parsers and obspy
        is not installed.
    """
    tag = sniff_format(path)

    if tag == "peer_at2":
        return read_peer_at2(path, scale_factor=scale_factor)

    if tag == "itaca":
        return read_itaca(path, scale_factor=scale_factor)

    if tag == "two_column":
        return read_two_column(path, scale_factor=scale_factor)

    if tag == "one_column":
        if dt is None:
            raise ValueError(
                "One-column records do not encode dt. "
                "Pass `dt=<seconds>`."
            )
        return read_one_column(path, dt=dt, scale_factor=scale_factor)

    # tag == "unknown" — try obspy as the long-tail delegate.
    return read_obspy(
        path, scale_factor=scale_factor, component=obspy_component
    )
