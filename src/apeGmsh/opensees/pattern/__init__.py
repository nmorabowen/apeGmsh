"""
``apeGmsh.opensees.pattern`` — typed ``pattern`` primitives.

Re-exports every concrete class from :mod:`.pattern` so that user code
can ``from apeGmsh.opensees.pattern import Plain`` for standalone
construction (P11) without reaching into the implementation module.

See :mod:`.pattern` for per-class documentation, ADR 0005 for the
explicit-context-manager decision, and ADR 0007 for why
``time_series/`` lives in a sibling subfolder rather than nested
inside ``pattern/`` (mirroring the OpenSees source layout would have
coupled two conceptually distinct concepts).
"""
from __future__ import annotations

from .pattern import (
    H5DRM,
    Plain,
    UniformExcitation,
    _LoadRecord,
    _SPRecord,
)

__all__ = [
    "Plain",
    "UniformExcitation",
    "H5DRM",
    "_LoadRecord",
    "_SPRecord",
]
