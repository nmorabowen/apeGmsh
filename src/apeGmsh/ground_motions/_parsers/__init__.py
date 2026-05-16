"""Parser surface for :mod:`apeGmsh.ground_motions`.

Each parser is a free function that takes a file path and returns a
:class:`apeGmsh.ground_motions.GroundMotion`. Format-specific
detectors (``is_*``) live alongside their parsers and are used by
the sniffer.
"""
from __future__ import annotations

from ._itaca import is_itaca, read_itaca
from ._obspy_bridge import from_obspy_trace, read_obspy
from ._one_column import read_one_column
from ._peer_at2 import is_peer_at2, read_peer_at2
from ._two_column import read_two_column

__all__ = [
    "read_two_column",
    "read_one_column",
    "read_peer_at2",
    "read_itaca",
    "read_obspy",
    "from_obspy_trace",
    "is_peer_at2",
    "is_itaca",
]
