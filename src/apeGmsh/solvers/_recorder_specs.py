"""Deprecation shim for the resolved recorder spec relocation (Phase 8.3b).

Canonical home is :mod:`apeGmsh.results.spec._resolved`.  The
:class:`ResolvedRecorderSpec` and :class:`ResolvedRecorderRecord`
containers (output of :class:`Recorders.resolve`) live next to the
results-side machinery that consumes them — capture, live driver,
MPCO emit, transcoders, writers.
"""
from __future__ import annotations

import warnings
from typing import Any

from apeGmsh.results.spec import _resolved as _canonical


def __getattr__(name: str) -> Any:
    """Forward attribute access to the canonical module.

    Used so that ``from apeGmsh.solvers._recorder_specs import _X``
    keeps working even for underscore-prefixed names that a plain
    star-import shim could not re-export.
    """
    return getattr(_canonical, name)


__all__ = getattr(_canonical, "__all__", [
    name for name in vars(_canonical) if not name.startswith("_")
])

warnings.warn(
    "apeGmsh.solvers._recorder_specs is deprecated; import from "
    "apeGmsh.results.spec._resolved instead.",
    DeprecationWarning,
    stacklevel=2,
)
