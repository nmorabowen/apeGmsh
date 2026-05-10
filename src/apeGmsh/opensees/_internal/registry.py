"""
Type-token registry — light scaffold.

The architecture references this; Phase 0 ships the slot. Subsequent
phases may register their concrete primitive classes here so callers
can look up "the Python class for this OpenSees ``(family,
type_token)`` pair." Today nothing populates it; today nothing reads
from it. The slot exists so Phase 1+ agents do not invent a parallel
mechanism.

Keying is ``(family, type_token)`` because OpenSees re-uses some
type tokens across families (e.g. ``Linear`` appears as a
``geomTransf``, a ``timeSeries``, and an ``algorithm``). The family
prefix disambiguates.
"""
from __future__ import annotations

from .types import Primitive


__all__ = ["register", "lookup"]


# (family, type_token) -> primitive class.
_REGISTRY: dict[tuple[str, str], type[Primitive]] = {}


def register(
    family: str, type_token: str, cls: type[Primitive]
) -> None:
    """Register a primitive class under ``(family, type_token)``."""
    _REGISTRY[(family, type_token)] = cls


def lookup(family: str, type_token: str) -> type[Primitive] | None:
    """Return the registered class for ``(family, type_token)``, or None."""
    return _REGISTRY.get((family, type_token))
