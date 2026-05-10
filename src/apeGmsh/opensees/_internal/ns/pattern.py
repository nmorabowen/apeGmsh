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
from ...time_series.time_series import (
    Constant,
    Linear,
    Path,
    Pulse,
    Trig,
)
from ._base import _BridgeNamespace


__all__ = ["_PatternNS"]


# Union of TimeSeries concrete types accepted on ``series=``. Mirrors
# the pattern used elsewhere (transform's ``_AnyTransf``): a concrete
# union lets mypy check call sites against the actual shipping
# TimeSeries subclasses rather than the abstract base.
_AnyTimeSeries = Linear | Constant | Path | Trig | Pulse


class _PatternNS(_BridgeNamespace):
    """``ops.pattern.<Type>(...)`` — typed methods for Phase 3A."""

    # -- Plain ----------------------------------------------------------
    def Plain(self, *, series: _AnyTimeSeries) -> Plain:
        """Construct + register a ``pattern Plain``.

        The returned instance is a context manager: open it with a
        ``with`` block and call ``p.load(...)`` / ``p.sp(...)`` to
        record the loads / prescribed displacements that the pattern
        carries.
        """
        return self._bridge._register(Plain(series=series))

    # -- UniformExcitation ---------------------------------------------
    def UniformExcitation(
        self,
        *,
        direction: int,
        series: _AnyTimeSeries,
    ) -> UniformExcitation:
        """Construct + register a ``pattern UniformExcitation``.

        Ground-motion pattern; the ``direction`` is a 1-based DOF index
        (1, 2, 3 = translations; 4, 5, 6 = rotations) per the OpenSees
        manual. The returned instance is technically a context manager
        for symmetry with :class:`Plain`, but the body is empty —
        the pattern's payload is the acceleration history itself.
        """
        return self._bridge._register(
            UniformExcitation(direction=direction, series=series)
        )
