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
    ASCE41Protocol,
    Constant,
    FEMA461Protocol,
    Linear,
    ModifiedATC24Protocol,
    Path,
    Pulse,
    Ricker,
    Trig,
)
from ._base import _BridgeNamespace


__all__ = ["_TimeSeriesNS"]


class _TimeSeriesNS(_BridgeNamespace):
    """``ops.timeSeries.<Type>(...)`` — typed methods for Phase 1D-extra."""

    # -- Linear ---------------------------------------------------------
    def Linear(self, *, factor: float = 1.0, name: str | None = None) -> Linear:
        """Construct + register a ``timeSeries Linear`` (linear ramp)."""
        return self._bridge._register(Linear(factor=factor), name=name)

    # -- Constant -------------------------------------------------------
    def Constant(
        self, *, factor: float = 1.0, name: str | None = None
    ) -> Constant:
        """Construct + register a ``timeSeries Constant`` (step)."""
        return self._bridge._register(Constant(factor=factor), name=name)

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
        name: str | None = None,
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
            ),
            name=name,
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
        name: str | None = None,
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
            ),
            name=name,
        )

    # -- Ricker ---------------------------------------------------------
    def Ricker(
        self,
        *,
        f_n: float,
        t_total: float,
        dt: float,
        t_center: float = 0.0,
        kind: str = "acceleration",
        factor: float = 1.0,
        name: str | None = None,
    ) -> Ricker:
        """Construct + register a Ricker wavelet (emits a ``timeSeries Path``).

        Samples a Gaussian-derivative wavelet onto a uniform ``dt`` grid
        and emits it as a tabulated path. ``kind`` is ``"acceleration"``
        (the Ricker / Mexican-hat) or ``"velocity"`` (1st-derivative
        form); ``t_center`` places the peak. See
        :class:`apeGmsh.opensees.time_series.time_series.Ricker`.
        """
        return self._bridge._register(
            Ricker(
                f_n=f_n,
                t_total=t_total,
                dt=dt,
                t_center=t_center,
                kind=kind,
                factor=factor,
            ),
            name=name,
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
        name: str | None = None,
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
            ),
            name=name,
        )

    # -- Cyclic loading protocols ---------------------------------------
    def ASCE41Protocol(
        self, *, factor: float = 1.0, name: str | None = None
    ) -> ASCE41Protocol:
        """Construct + register an ASCE 41 cyclic protocol (emits a Path).

        Normalized ±1 displacement history; ``factor`` is the peak
        displacement / strain. See
        :class:`apeGmsh.opensees.time_series.time_series.ASCE41Protocol`.
        """
        return self._bridge._register(ASCE41Protocol(factor=factor), name=name)

    def ModifiedATC24Protocol(
        self, *, factor: float = 1.0, name: str | None = None
    ) -> ModifiedATC24Protocol:
        """Construct + register a Modified ATC-24 cyclic protocol (Path).

        Normalized ±1 displacement history; ``factor`` is the peak
        displacement / strain. See
        :class:`apeGmsh.opensees.time_series.time_series.ModifiedATC24Protocol`.
        """
        return self._bridge._register(
            ModifiedATC24Protocol(factor=factor), name=name
        )

    def FEMA461Protocol(
        self,
        *,
        factor: float = 1.0,
        alpha: float = 0.4,
        start_fraction: float = 0.01,
        name: str | None = None,
    ) -> FEMA461Protocol:
        """Construct + register a FEMA 461 cyclic protocol (emits a Path).

        Two cycles per amplitude with geometric growth by ``(1 + alpha)``;
        normalized ±1, ``factor`` is the peak displacement / strain. See
        :class:`apeGmsh.opensees.time_series.time_series.FEMA461Protocol`.
        """
        return self._bridge._register(
            FEMA461Protocol(
                factor=factor, alpha=alpha, start_fraction=start_fraction
            ),
            name=name,
        )
