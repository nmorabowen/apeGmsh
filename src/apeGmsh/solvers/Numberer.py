"""Deprecation shim for the Numberer relocation (Phase 8.1).

The canonical home is now :mod:`apeGmsh.mesh._numberer`.  This module
re-exports the public surface so legacy imports keep working for one
release cycle, with a one-shot :class:`DeprecationWarning`.
"""
from __future__ import annotations

import warnings

from apeGmsh.mesh._numberer import Numberer, NumberedMesh

warnings.warn(
    "apeGmsh.solvers.Numberer is deprecated; import Numberer and "
    "NumberedMesh from apeGmsh.mesh._numberer (or the top-level "
    "apeGmsh package) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Numberer", "NumberedMesh"]
