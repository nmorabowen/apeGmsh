"""apeGmsh.results._vocabulary — deprecation shim (Phase 9).

The canonical vocabulary moved to :mod:`apeGmsh._vocabulary` so the
OpenSees bridge (declaration-side) and the results module
(consumer-side) can both import without a layering inversion.

This shim fires a :class:`DeprecationWarning` once on first import
and re-exports the canonical names for one release cycle. Internal
apeGmsh code imports from :mod:`apeGmsh._vocabulary` directly; only
external callers see the warning.
"""
from __future__ import annotations

import warnings as _warnings

from apeGmsh._vocabulary import *  # noqa: F401, F403

_warnings.warn(
    "apeGmsh.results._vocabulary moved to apeGmsh._vocabulary in "
    "Phase 9; import from apeGmsh._vocabulary directly. This shim "
    "will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
