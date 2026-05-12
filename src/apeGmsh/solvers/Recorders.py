"""Deprecation shim for the Recorders declaration helper relocation (Phase 8.3b).

Canonical home is :mod:`apeGmsh.results.spec.declaration`.  The
declaration helper is paired with :mod:`apeGmsh.results.spec._resolved`
(its output type) and :mod:`apeGmsh.results.spec._emit` (the emit path
its output drives), so all three live next to the results-side
machinery that consumes them.
"""
from __future__ import annotations

import warnings
from typing import Any

from apeGmsh.results.spec import declaration as _canonical


def __getattr__(name: str) -> Any:
    """Forward attribute access to the canonical module.

    Used so that ``from apeGmsh.solvers.Recorders import _X`` keeps
    working even for underscore-prefixed names that a plain
    star-import shim could not re-export.
    """
    return getattr(_canonical, name)


__all__ = getattr(_canonical, "__all__", [
    name for name in vars(_canonical) if not name.startswith("_")
])

warnings.warn(
    "apeGmsh.solvers.Recorders is deprecated; import Recorders from "
    "apeGmsh.results.spec (or apeGmsh.results.spec.declaration) instead.",
    DeprecationWarning,
    stacklevel=2,
)
