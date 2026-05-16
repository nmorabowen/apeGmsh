"""ITACA / ESM ASCII acceleration record parser.

The ITalian ACcelerometric Archive (ITACA) and the European Engineering
Strong Motion database (ESM) ship records as ASCII files with a long
keyword-value header followed by a single column of acceleration
values::

    EVENT_NAME: CENTRAL_ITALY
    EVENT_DATE_YYYYMMDD: 20161030
    ...
    SAMPLING_INTERVAL_S: 0.005
    NPTS: 50000
    UNITS: cm/s^2
    ...
    0.001234
    0.001345
    ...

The unit declaration is preserved in ``metadata["units_declared"]`` for
inspection but NOT applied to values — apply your own ``scale_factor``
if you need conversion (e.g. ``0.01`` for ``cm/s^2`` → ``m/s^2``).
"""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from .._motion import GroundMotion


# A header line: optional whitespace + KEY (letters/digits/underscore)
# + colon + value. Anything else (or a bare numeric line) ends the header.
_HEADER_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")

# Canonical ITACA / ESM keys for DT, NPTS, UNITS. Real files vary by
# vintage — match generously.
_DT_KEYS = (
    "SAMPLING_INTERVAL_S",
    "SAMPLING_INTERVAL_(S)",
    "SAMPLING_INTERVAL",
    "TIME_STEP",
    "DT",
)
_NPTS_KEYS = ("NDATA", "NPTS", "NUM_SAMPLES", "NPOINTS")
_UNITS_KEYS = ("UNITS", "UNIT_MEASURE", "DATA_UNITS")


def _looks_numeric(line: str) -> bool:
    """True if *line* contains only numeric tokens (one or many)."""
    toks = line.split()
    if not toks:
        return False
    try:
        for t in toks:
            float(t)
        return True
    except ValueError:
        return False


def is_itaca(path: str | os.PathLike[str]) -> bool:
    """Quick header-peek: does the file start with ITACA-style keywords?"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 10:
                    break
                m = _HEADER_LINE.match(line)
                if not m:
                    continue
                key = m.group(1).upper()
                if (
                    key.startswith(("EVENT_", "NETWORK", "STATION_CODE"))
                    or key in _DT_KEYS
                    or key in _NPTS_KEYS
                ):
                    return True
    except OSError:
        return False
    return False


def read_itaca(
    path: str | os.PathLike[str],
    *,
    scale_factor: float = 1.0,
) -> GroundMotion:
    """Parse an ITACA / ESM ASCII acceleration file.

    Parameters
    ----------
    path
        Path to the ITACA / ESM ``.asc`` file.
    scale_factor
        Multiplier applied to the acceleration values at read time.
        ITACA defaults to ``cm/s^2`` — pass ``0.01`` to convert to
        ``m/s^2``, or leave at ``1.0`` for native units.

    Returns
    -------
    A :class:`GroundMotion` snapshot. The full keyword-value header
    is preserved under ``metadata["header"]``; the declared units
    string is also surfaced as ``metadata["units_declared"]``.

    Raises
    ------
    ValueError
        If DT cannot be found in the header, or the data block is empty.
    """
    path_str = os.fspath(path)
    header: dict[str, str] = {}
    data_lines: list[str] = []

    with open(path_str, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if data_lines or _looks_numeric(stripped):
                data_lines.append(stripped)
                continue
            m = _HEADER_LINE.match(line)
            if m:
                header[m.group(1).upper()] = m.group(2)

    if not data_lines:
        raise ValueError(f"{path_str}: no numeric data block found")

    # Locate DT.
    dt: float | None = None
    for key in _DT_KEYS:
        if key in header:
            try:
                dt = float(header[key])
                break
            except ValueError:
                continue
    if dt is None:
        raise ValueError(
            f"{path_str}: no sampling interval found in header "
            f"(looked for {_DT_KEYS})"
        )

    # Locate declared units (kept for inspection only — not transformed).
    units_declared: str | None = None
    for key in _UNITS_KEYS:
        if key in header:
            units_declared = header[key].strip()
            break

    # Locate declared NPTS (optional — used as a sanity cross-check).
    npts_declared: int | None = None
    for key in _NPTS_KEYS:
        if key in header:
            try:
                npts_declared = int(header[key])
                break
            except ValueError:
                continue

    flat: list[float] = []
    for line in data_lines:
        for tok in line.split():
            flat.append(float(tok))
    if len(flat) < 2:
        raise ValueError(
            f"{path_str}: need at least 2 samples, found {len(flat)}"
        )
    if npts_declared is not None and len(flat) < npts_declared:
        raise ValueError(
            f"{path_str}: header declares NPTS={npts_declared}, "
            f"but file contains only {len(flat)} samples"
        )
    if npts_declared is not None:
        flat = flat[:npts_declared]
    accel = np.asarray(flat, dtype=np.float64) * scale_factor

    metadata: dict[str, Any] = {
        "format": "itaca",
        "header": header,
        "scale_factor": scale_factor,
    }
    if npts_declared is not None:
        metadata["npts_declared"] = npts_declared
    if units_declared is not None:
        metadata["units_declared"] = units_declared

    return GroundMotion(
        accel=accel,
        dt=dt,
        source=os.path.basename(path_str),
        metadata=metadata,
    )
