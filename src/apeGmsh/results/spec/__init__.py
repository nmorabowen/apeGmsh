"""apeGmsh.results.spec — resolved recorder spec + emit helpers.

Phase 8.3b relocated the recorder cluster out of ``apeGmsh.solvers``
into this sub-package. Phase 9 then unified recorder declaration on
the bridge:

- :func:`apeGmsh.opensees.ops.recorder.declare` is the canonical
  declaration entry point. It produces a typed
  :class:`apeGmsh.opensees.recorder.RecorderDeclaration` that drives
  the file-emit path through the bridge build pipeline.
- The in-process capture counterpart (Phase 9 commit 5) was deleted in
  #2191; modes come from ``ops.eigen(...)`` (``EigenResult``) and
  plain-openseespy decks from ``results.live.capture_live_modes``.

What lives in this sub-package post commit 5:

- :mod:`apeGmsh.results.spec._resolved` — resolved containers
  (:class:`ResolvedRecorderSpec`, :class:`ResolvedRecorderRecord`)
  consumed by the legacy file-emit / live-recorder / transcoder
  pipeline. New code paths construct these directly when needed.
- :mod:`apeGmsh.results.spec._emit` — emission helpers
  (:func:`emit_logical`, :func:`to_ops_args`, :func:`mpco_ops_args`,
  :func:`line_station_gpx_path`, …) that translate a resolved spec
  into OpenSees-native recorder commands.

The Phase 9 commit 4 ``Recorders`` fluent helper relocation was
reverted in commit 5: the transitional helper was deleted entirely
because the canonical Phase 9 path
(:func:`ops.recorder.declare`) covers the same surface without an
opensees → results layering inversion.
"""
from __future__ import annotations

from ._resolved import ResolvedRecorderRecord, ResolvedRecorderSpec

__all__ = [
    "ResolvedRecorderRecord",
    "ResolvedRecorderSpec",
]
