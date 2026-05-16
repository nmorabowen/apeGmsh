"""``apeGmsh.ground_motions`` — seismic record loaders.

A small domain module for loading acceleration time histories and
feeding them into an :class:`apeGmsh.opensees.apeSees` session.

The module is **unit-agnostic**: values come out of parsers as they
were stored in the file, modulo an optional ``scale_factor`` that
gets multiplied at read time. The caller is responsible for knowing
what units the record is in and what units the model expects::

    from apeGmsh.ground_motions import GroundMotion

    # Sniff and dispatch (recommended) — values stored as-is
    gm = GroundMotion.from_file("RSN160.AT2")

    # Convert at read time via scale_factor (PEER records are in g)
    gm = GroundMotion.from_peer_at2("RSN160.AT2", scale_factor=9.81)

    # Or convert at emit time via the OpenSees Path primitive's factor=
    ts = gm.to_time_series(ops, factor=9.81)
    with ops.pattern.UniformExcitation(direction=1, series=ts):
        pass

Supported formats:

==================  ===========================  ==================
Format              Constructor                  Dependency
==================  ===========================  ==================
PEER NGA AT2        ``from_peer_at2`` / sniff    none
ITACA / ESM ASCII   ``from_itaca`` / sniff       none
Two-column ``time accel``  ``from_two_column``   none
One-column accel    ``from_one_column``          none
K-NET / SAC / ...   ``from_obspy`` /             ``obspy``
                    ``from_file`` fallback       (opt-in)
==================  ===========================  ==================

Where a file's source format declares units (PEER line 3, ITACA
``UNITS:`` header), the string is preserved in
``GroundMotion.metadata["units_declared"]`` for inspection — but the
parser never applies it.
"""
from __future__ import annotations

from ._motion import GroundMotion
from ._sniffer import from_file, sniff_format

__all__ = [
    "GroundMotion",
    "from_file",
    "sniff_format",
]
