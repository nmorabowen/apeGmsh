"""
``apeGmsh.opensees.time_series`` — typed ``timeSeries`` primitives.

Re-exports every concrete class from :mod:`.time_series` so that user
code can ``from apeGmsh.opensees.time_series import Path`` for
material-study / standalone construction (P11) without reaching into
the implementation module.

See :mod:`.time_series` for per-class documentation and ADR 0007 for
the rationale behind separating ``time_series/`` from ``pattern/``.
"""
from __future__ import annotations

from .time_series import (
    Constant,
    Linear,
    Path,
    Pulse,
    Trig,
)

__all__ = [
    "Linear",
    "Constant",
    "Path",
    "Trig",
    "Pulse",
]
