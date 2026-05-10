"""
Emitter package — single seam between primitives and emit targets.

Phase 0 ships the frozen Protocol (``base.Emitter``) and the in-memory
test fixture (``recording.RecordingEmitter``). The three production
emitters (TclEmitter, PyEmitter, LiveOpsEmitter) and the H5 emitter
land in later phases; each implements this same Protocol.
"""
from __future__ import annotations

from .base import Emitter
from .recording import RecordingEmitter

__all__ = ["Emitter", "RecordingEmitter"]
