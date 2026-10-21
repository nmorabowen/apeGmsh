"""Live recorder execution — emit recorders against the running ops domain.

Three strategies, all consume a :class:`ResolvedRecorderSpec`:

- :class:`LiveRecorders` — classic ``recorder Node`` / ``recorder
  Element`` emission with per-stage filenames; read back via
  :meth:`Results.from_recorders` (one stage at a time).
- :class:`LiveMPCO` — single ``recorder mpco`` writing one HDF5 file
  containing all stages; read back via :meth:`Results.from_mpco`.
  Requires an openseespy build with the MPCO recorder compiled in.
- :class:`LiveDefault` — the default route: picks one of the above, or
  the fork's own recorder, from the bound build (``emit_default``).
  :func:`read_deck_mpco` is its deck-side twin for a ``.mpco``.

:func:`capture_live_modes` writes the modes of a plain openseespy deck
(``ops.eigen`` on the live domain) as native ``mode_<k>`` stages.

Both spawn no subprocess: the output files land on disk while the
notebook keeps running.
"""
from ._default import DefaultRouteWarning, LiveDefault, read_deck_mpco
from ._modes import capture_live_modes
from ._mpco import LiveMPCO
from ._recorders import LiveRecorders

__all__ = [
    "DefaultRouteWarning", "LiveDefault", "LiveRecorders", "LiveMPCO",
    "capture_live_modes", "read_deck_mpco",
]
