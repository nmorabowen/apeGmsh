"""
Typed ``damping`` objects (ADR 0053, D3).

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)`` mirroring
the OpenSees ``damping <Type> $tag ...`` command. The matching
:class:`apeGmsh.opensees._internal.ns.damping._DampingNS` methods take the same
kwargs, ``_register`` the primitive, and record a ``region -damp`` attachment.

These are frequency-band viscous dissipators (Yuli Huang et al., 2020-2022) —
**not** Rayleigh. They are inert until attached to elements via a region.

D3a ships the two scalar types:

  * :class:`Uniform`  — constant ζ across a frequency band
    (``damping Uniform $tag $zeta $freq1 $freq2``)
  * :class:`SecStif`  — committed-stiffness-proportional
    (``damping SecStif $tag $beta``)

The multi-point types (``URD`` / ``URDbeta``) and the ``-factor`` TimeSeries
scale are deferred to a follow-up (D3b).

**Time window** — all types accept ``activate_time`` / ``deactivate_time``
(``-activateTime`` / ``-deactivateTime``) so the object can be switched off
during a quasi-static gravity stage and on for the dynamic phase; without
windowing a damping object dissipates from t=0 and corrupts the initial state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Damping, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = ["Uniform", "SecStif"]


def _window_args(
    activate_time: float | None,
    deactivate_time: float | None,
) -> list[float | str]:
    """Trailing ``-activateTime`` / ``-deactivateTime`` flags (ADR 0053).

    Emitted only when set — OpenSees defaults are ``0.0`` / ``1e20``.
    """
    args: list[float | str] = []
    if activate_time is not None:
        args += ["-activateTime", activate_time]
    if deactivate_time is not None:
        args += ["-deactivateTime", deactivate_time]
    return args


@dataclass(frozen=True, kw_only=True, slots=True)
class Uniform(Damping):
    """``damping Uniform $tag $zeta $freq1 $freq2 [-activateTime ta]
    [-deactivateTime td]`` — constant damping ratio ``zeta`` across the
    band ``[freq1, freq2]`` (Hz).

    ``zeta`` is the **physical target ratio** — OpenSees applies the
    internal factor of two; do not pre-divide (ADR 0053).
    """

    zeta: float
    freq1: float
    freq2: float
    activate_time: float | None = None
    deactivate_time: float | None = None

    def __post_init__(self) -> None:
        if self.zeta < 0.0:
            raise ValueError(f"Uniform: zeta must be >= 0, got {self.zeta!r}")
        if self.freq1 <= 0.0 or self.freq2 <= 0.0:
            raise ValueError(
                "Uniform: freq1/freq2 must be positive (Hz), got "
                f"freq1={self.freq1!r}, freq2={self.freq2!r}"
            )
        if self.freq2 <= self.freq1:
            raise ValueError(
                "Uniform: freq2 must be > freq1, got "
                f"freq1={self.freq1!r}, freq2={self.freq2!r}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        emitter.damping(
            "Uniform", tag, self.zeta, self.freq1, self.freq2,
            *_window_args(self.activate_time, self.deactivate_time),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class SecStif(Damping):
    """``damping SecStif $tag $beta [-activateTime ta] [-deactivateTime td]``
    — committed (secant) stiffness-proportional damping.

    (OpenSees also accepts the ``SecStiff`` spelling; the bridge emits the
    ``SecStif`` form, which both the Tcl and openseespy parsers accept.)
    """

    beta: float
    activate_time: float | None = None
    deactivate_time: float | None = None

    def __post_init__(self) -> None:
        if self.beta < 0.0:
            raise ValueError(f"SecStif: beta must be >= 0, got {self.beta!r}")

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        emitter.damping(
            "SecStif", tag, self.beta,
            *_window_args(self.activate_time, self.deactivate_time),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
