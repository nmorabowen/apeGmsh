"""Live recorder execution — emit recorders against the running ops domain.

Two strategies, both consume a :class:`ResolvedRecorderSpec`:

- :class:`LiveRecorders` — classic ``recorder Node`` / ``recorder
  Element`` emission with per-stage filenames; read back via
  :meth:`Results.from_recorders` (one stage at a time).
- :class:`LiveMPCO` — single ``recorder mpco`` writing one HDF5 file
  containing all stages; read back via :meth:`Results.from_mpco`.
  Requires an openseespy build with the MPCO recorder compiled in.

Both spawn no subprocess: the output files land on disk while the
notebook keeps running.
"""
from ._mpco import LiveMPCO
from ._recorders import LiveRecorders

__all__ = ["LiveRecorders", "LiveMPCO"]
