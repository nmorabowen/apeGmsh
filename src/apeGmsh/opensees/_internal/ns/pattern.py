"""
``_PatternNS`` — backs ``ops.pattern.<Type>(...)``.

Phase 3A populates the workhorse :class:`Plain` and the
ground-motion :class:`UniformExcitation`. Each method mirrors the
matching dataclass signature exactly and registers the constructed
primitive with the bridge.

:class:`MultiSupport` is deferred to a follow-up.
"""
from __future__ import annotations

from ...pattern.pattern import Plain, UniformExcitation
from ..types import TimeSeries
from ._base import _BridgeNamespace


__all__ = ["_PatternNS"]


# ``series=`` accepts the abstract ``TimeSeries`` base (or a registered
# name), NOT a concrete union. A concrete union was tried first and
# rotted within weeks: PR #558's Ricker / ASCE41Protocol /
# ModifiedATC24Protocol / FEMA461Protocol never made it into the union,
# so type-checked callers were falsely rejected, and internal stage
# helpers (typed against the base) couldn't forward at all. Every
# TimeSeries subclass is a valid series — the base IS the contract.
_AnyTimeSeries = TimeSeries


class _PatternNS(_BridgeNamespace):
    """``ops.pattern.<Type>(...)`` — typed methods for Phase 3A."""

    # -- Plain ----------------------------------------------------------
    def Plain(
        self,
        *,
        series: _AnyTimeSeries | str,
        name: str | None = None,
    ) -> Plain:
        """Construct + register a ``pattern Plain``.

        ``series=`` accepts either a TimeSeries handle or the name a
        TimeSeries was registered under (``ops.timeSeries.Linear(name=...)``).

        The returned instance is a context manager: open it with a
        ``with`` block and call ``p.load(...)`` / ``p.sp(...)`` to
        record the loads / prescribed displacements that the pattern
        carries.
        """
        series = self._bridge._resolve(series, base=TimeSeries)
        return self._bridge._register(Plain(series=series), name=name)

    # -- UniformExcitation ---------------------------------------------
    def UniformExcitation(
        self,
        *,
        direction: int,
        series: _AnyTimeSeries | str,
        name: str | None = None,
    ) -> UniformExcitation:
        """Construct + register a ``pattern UniformExcitation``.

        Ground-motion pattern; the ``direction`` is a 1-based DOF index
        (1, 2, 3 = translations; 4, 5, 6 = rotations) per the OpenSees
        manual. ``series=`` accepts a TimeSeries handle or its
        registered name. The returned instance is technically a context
        manager for symmetry with :class:`Plain`, but the body is empty
        — the pattern's payload is the acceleration history itself.
        """
        series = self._bridge._resolve(series, base=TimeSeries)
        return self._bridge._register(
            UniformExcitation(direction=direction, series=series),
            name=name,
        )
