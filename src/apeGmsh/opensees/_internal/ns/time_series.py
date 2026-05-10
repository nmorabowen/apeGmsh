"""
``_TimeSeriesNS`` — backs ``ops.timeSeries.<Type>(...)``.

Phase 1D-extra populates the OpenSees core 5: Linear, Constant, Path,
Trig, Pulse. Each method mirrors the matching dataclass signature
exactly and registers the constructed primitive with the bridge.

Cyclic-loading-protocol classes (ASCE41Protocol, FEMA461Protocol,
ATC24Protocol) are deferred to a follow-up; when they land they slot
into this namespace alongside the existing methods.
"""
from __future__ import annotations

from ...time_series.time_series import (
    Constant,
    Linear,
    Path,
    Pulse,
    Trig,
)
from ._base import _BridgeNamespace


__all__ = ["_TimeSeriesNS"]


class _TimeSeriesNS(_BridgeNamespace):
    """``ops.timeSeries.<Type>(...)`` — typed methods for Phase 1D-extra."""

    # -- Linear ---------------------------------------------------------
    def Linear(self, *, factor: float = 1.0) -> Linear:
        """Construct + register a ``timeSeries Linear`` (linear ramp)."""
        return self._bridge._register(Linear(factor=factor))

    # -- Constant -------------------------------------------------------
    def Constant(self, *, factor: float = 1.0) -> Constant:
        """Construct + register a ``timeSeries Constant`` (step)."""
        return self._bridge._register(Constant(factor=factor))

    # -- Path -----------------------------------------------------------
    def Path(
        self,
        *,
        file: str | None = None,
        values: tuple[float, ...] | None = None,
        time: tuple[float, ...] | None = None,
        dt: float | None = None,
        factor: float = 1.0,
        start_time: float = 0.0,
        prepend_zero: bool = False,
    ) -> Path:
        """Construct + register a ``timeSeries Path`` (time-history).

        Exactly one of ``file`` or ``values`` must be supplied; when
        ``values`` is supplied, exactly one of ``dt`` or ``time`` is
        required. See :class:`apeGmsh.opensees.time_series.time_series.Path`.
        """
        return self._bridge._register(
            Path(
                file=file,
                values=values,
                time=time,
                dt=dt,
                factor=factor,
                start_time=start_time,
                prepend_zero=prepend_zero,
            )
        )

    # -- Trig -----------------------------------------------------------
    def Trig(
        self,
        *,
        t_start: float,
        t_end: float,
        period: float,
        factor: float = 1.0,
        shift: float = 0.0,
        zero_shift: float = 0.0,
    ) -> Trig:
        """Construct + register a ``timeSeries Trig`` (sinusoidal)."""
        return self._bridge._register(
            Trig(
                t_start=t_start,
                t_end=t_end,
                period=period,
                factor=factor,
                shift=shift,
                zero_shift=zero_shift,
            )
        )

    # -- Pulse ----------------------------------------------------------
    def Pulse(
        self,
        *,
        t_start: float,
        t_end: float,
        period: float,
        width: float,
        factor: float = 1.0,
        shift: float = 0.0,
        zero_shift: float = 0.0,
    ) -> Pulse:
        """Construct + register a ``timeSeries Pulse`` (square wave)."""
        return self._bridge._register(
            Pulse(
                t_start=t_start,
                t_end=t_end,
                period=period,
                width=width,
                factor=factor,
                shift=shift,
                zero_shift=zero_shift,
            )
        )
