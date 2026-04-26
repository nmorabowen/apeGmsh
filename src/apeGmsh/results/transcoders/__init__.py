"""Recorder output transcoders.

Parse OpenSees ``.out`` / ``.xml`` recorder files into apeGmsh
native HDF5 (Strategy A — Tcl recorders → user runs OpenSees → we
parse the output here).

Phase 6 v1 supports text format (``.out``) for nodal records.
Element-level transcoding shares unflattening logic with the
Phase 7 element-level capture; both will land together in a
follow-up.
"""
from ._recorder import RecorderTranscoder

__all__ = ["RecorderTranscoder"]
