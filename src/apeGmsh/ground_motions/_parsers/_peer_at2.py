"""PEER NGA AT2 acceleration record parser.

PEER NGA-West2 ``.AT2`` files follow a fixed 4-line ASCII header
followed by acceleration values in FORTRAN-style fixed-width::

    PEER NGA STRONG MOTION DATABASE RECORD
    IMPERIAL VALLEY 10/15/79, BONDS CORNER, 140
    ACCELERATION TIME HISTORY IN UNITS OF G
    NPTS=  3164, DT=   .0050 SEC
      .1234E-04  .2345E-04 ...

The 4th line encodes NPTS and DT in a forgiving keyword=value format.
The units declared on line 3 are preserved in
``metadata["units_declared"]`` for inspection but NOT applied to
values — apply your own ``scale_factor`` if you need conversion.
"""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from .._motion import GroundMotion


# ``NPTS = 3164,  DT = .0050  SEC`` — case-insensitive, whitespace-free.
_HEADER_RE = re.compile(
    r"npts\s*=\s*(\d+).*?dt\s*=\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)

_PEER_BANNER = "PEER NGA"


def is_peer_at2(path: str | os.PathLike[str]) -> bool:
    """Quick header-peek: does the first line carry the PEER banner?"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    return _PEER_BANNER in first.upper()


def read_peer_at2(
    path: str | os.PathLike[str],
    *,
    scale_factor: float = 1.0,
) -> GroundMotion:
    """Parse a PEER NGA ``.AT2`` acceleration file.

    Parameters
    ----------
    path
        Path to the ``.AT2`` file.
    scale_factor
        Multiplier applied to the acceleration values at read time.
        PEER records are typically in ``g`` — pass ``9.81`` to push
        into an SI model, or leave at ``1.0`` to keep native units.

    Returns
    -------
    A :class:`GroundMotion` snapshot. ``metadata`` carries the raw
    header lines under keys ``header1``..``header4`` and the unit
    declaration under ``units_declared``.

    Raises
    ------
    ValueError
        If line 4 does not contain a parseable ``NPTS=`` / ``DT=``
        pair, or if the data block is shorter than ``NPTS``.
    """
    path_str = os.fspath(path)
    with open(path_str, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if len(lines) < 5:
        raise ValueError(
            f"{path_str}: PEER files have a 4-line header + data; "
            f"got only {len(lines)} lines"
        )

    h1, h2, h3, h4 = lines[:4]
    m = _HEADER_RE.search(h4)
    if not m:
        raise ValueError(
            f"{path_str}: line 4 does not match PEER header pattern "
            f"'NPTS=..., DT=...'; got: {h4!r}"
        )
    npts = int(m.group(1))
    dt = float(m.group(2))

    flat: list[float] = []
    for line in lines[4:]:
        for tok in line.split():
            flat.append(float(tok))
    if len(flat) < npts:
        raise ValueError(
            f"{path_str}: header declares NPTS={npts}, "
            f"but file contains only {len(flat)} samples"
        )
    accel = np.asarray(flat[:npts], dtype=np.float64) * scale_factor

    metadata: dict[str, Any] = {
        "format": "peer_at2",
        "header1": h1.rstrip(),
        "header2": h2.rstrip(),
        "header3": h3.rstrip(),
        "header4": h4.rstrip(),
        "npts_declared": npts,
        "units_declared": h3.strip(),
        "scale_factor": scale_factor,
    }
    return GroundMotion(
        accel=accel,
        dt=dt,
        source=os.path.basename(path_str),
        metadata=metadata,
    )
