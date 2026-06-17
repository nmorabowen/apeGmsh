"""``_FaultNS`` — backs ``ops.fault.from_shakermaker`` (MT-4, ADR 0062).

A finite fault is a loop of point moment-tensor sources, each with its own
location, scalar moment, mechanism, and rupture onset ``t0``. Because
OpenSees binds one ``timeSeries`` per pattern, each source becomes **one
``Plain`` pattern**: a ``Yoffe`` moment function shifted to the source's
rupture time (``t0``) carrying a single ``moment_tensor`` (whose own ``t0``
stays 0 — the delay lives in the series). Returns the list of patterns.

Per ADR 0062 decision 8 this imports nothing from ShakerMaker:
``from_shakermaker`` duck-types the ``FaultSource`` (iterates it; reads each
source's ``.x`` [km], ``.angles`` [**radians** → degrees], ``.tt`` [onset]).

The FFSP ``get_subfaults()`` adapter (``from_ffsp``) is **deferred to MT-5**:
its unit contract is source-verified to differ from this module's earlier
assumption (FFSP coords are metres not km; ``peak_time`` is the dimensionless
ratio ``pktm/(rstm+pktm)`` not seconds; ``slip`` is rescaled by ``is_moment``),
and a correct ``M0`` needs validation against a real FFSP run.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ._base import _BridgeNamespace

if TYPE_CHECKING:
    from ...pattern.pattern import Plain


__all__ = [
    "_FaultNS",
    "WarnFaultSubfaultSkipped",
    "WarnFaultPeakClamped",
    "WarnFaultSubfaultTruncated",
]

# Yoffe requires peak_time < rise_time/2 (the regularization half-width).
# Clamp just below to keep the series constructible; warn when it bites.
_YOFFE_PEAK_FRACTION = 0.49


class WarnFaultSubfaultSkipped(UserWarning):
    """Subfaults skipped (non-positive moment / rise / peak time)."""


class WarnFaultPeakClamped(UserWarning):
    """A subfault's ``peak_time`` was clamped below ``rise_time/2`` to satisfy
    the Yoffe regularization bound, shortening its slip-rate pulse."""


class WarnFaultSubfaultTruncated(UserWarning):
    """A subfault's rupture onset (+rise) falls outside the analysis window
    ``t_total``, so its moment is partly or wholly never released."""


@dataclass(frozen=True, slots=True)
class FaultSubsource:
    """One point moment-tensor source of a finite fault (deck units)."""

    position: tuple[float, float, float]
    M0: float
    strike: float
    dip: float
    rake: float
    t0: float
    rise_time: float
    peak_time: float


def _deck_position(
    x_km: float, y_km: float, z_km: float, *, length_scale: float, frame: str,
) -> tuple[float, float, float]:
    """Map a (km, depth-down) source point to deck coordinates.

    ``length_scale`` is deck-length per km. For a ``"z-up"`` mesh (free
    surface at ``z = 0``, depth positive downward) the depth becomes a
    negative deck ``z``; ``"z-down"`` keeps it positive.
    """
    if frame not in ("z-up", "z-down"):
        raise ValueError(
            f"_deck_position: frame must be 'z-up' or 'z-down', got {frame!r}."
        )
    zx = float(z_km) * length_scale
    z_deck = -zx if frame == "z-up" else zx
    return (float(x_km) * length_scale, float(y_km) * length_scale, z_deck)


def _per_source(value: "float | Sequence[float]", i: int, n: int) -> float:
    """Broadcast a scalar (incl. a 0-d array), or index a length-``n`` seq."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.ravel()
    if len(flat) != n:
        raise ValueError(
            f"from_shakermaker: per-source value has length {len(flat)} but "
            f"the fault has {n} sources."
        )
    return float(flat[i])


class _FaultNS(_BridgeNamespace):
    """``ops.fault.<from_*>(...)`` — finite-fault moment-tensor sources."""

    def from_shakermaker(
        self,
        fault: Any,
        *,
        frame: str,
        M0: "float | Sequence[float]",
        rise_time: "float | Sequence[float]",
        peak_time: "float | Sequence[float]",
        dt: float,
        t_total: float,
        length_scale: float,
        method: str = "consistent",
        f_max: float | None = None,
        region: str | None = None,
    ) -> "list[Plain]":
        """Build one ``Plain`` pattern per ``PointSource`` of a ``FaultSource``.

        Duck-types the fault (iterates it; reads each source's ``.x``
        [km], ``.angles`` [**radians** — converted to degrees here], and
        ``.tt`` [rupture onset]). A ``PointSource`` carries no scalar
        moment or rise/peak time, so ``M0`` / ``rise_time`` / ``peak_time``
        are supplied here — a scalar (shared) or a per-source sequence.
        ``length_scale`` (deck-length per km) is required.
        """
        sources = list(fault)
        n = len(sources)
        specs: list[FaultSubsource] = []
        for i, ps in enumerate(sources):
            x = np.asarray(ps.x, dtype=float).ravel()
            ang = np.asarray(ps.angles, dtype=float).ravel()  # RADIANS
            if x.shape[0] != 3 or ang.shape[0] < 3:
                raise ValueError(
                    f"from_shakermaker: source {i} needs x (len 3) and "
                    f"angles (len 3); got x={x.tolist()}, angles={ang.tolist()}."
                )
            pos = _deck_position(
                x[0], x[1], x[2], length_scale=length_scale, frame=frame,
            )
            specs.append(FaultSubsource(
                position=pos,
                M0=_per_source(M0, i, n),
                strike=math.degrees(float(ang[0])),
                dip=math.degrees(float(ang[1])),
                rake=math.degrees(float(ang[2])),
                t0=float(getattr(ps, "tt", 0.0)),
                rise_time=_per_source(rise_time, i, n),
                peak_time=_per_source(peak_time, i, n),
            ))
        return self._build_patterns(
            specs, frame=frame, dt=dt, t_total=t_total,
            method=method, f_max=f_max, region=region,
        )

    # -- shared pattern builder ----------------------------------------

    def _build_patterns(
        self,
        specs: "list[FaultSubsource]",
        *,
        frame: str,
        dt: float,
        t_total: float,
        method: str,
        f_max: float | None,
        region: str | None = None,
    ) -> "list[Plain]":
        patterns: list[Plain] = []
        skipped = clamped = truncated = never_fired = 0
        for s in specs:
            if s.M0 <= 0 or s.rise_time <= 0 or s.peak_time <= 0:
                skipped += 1
                continue
            # A source whose onset is past the window never fires; one whose
            # rise tail spills past it under-releases its moment.
            if s.t0 >= t_total:
                never_fired += 1
            elif s.t0 + s.rise_time > t_total:
                truncated += 1
            # Yoffe requires peak_time < rise_time/2; clamp + flag.
            peak = s.peak_time
            if peak > _YOFFE_PEAK_FRACTION * s.rise_time:
                peak = _YOFFE_PEAK_FRACTION * s.rise_time
                clamped += 1
            series = self._bridge.timeSeries.Yoffe(
                rise_time=s.rise_time, peak_time=peak, t0=s.t0,
                t_total=t_total, dt=dt, f_max=f_max,
            )
            p = self._bridge.pattern.Plain(series=series)
            p.moment_tensor(
                position=s.position, frame=frame, M0=s.M0,
                mech=dict(strike=s.strike, dip=s.dip, rake=s.rake),
                method=method, region=region,
            )
            patterns.append(p)
        self._emit_build_warnings(
            len(specs), skipped, clamped, truncated, never_fired,
        )
        return patterns

    @staticmethod
    def _emit_build_warnings(
        total: int, skipped: int, clamped: int, truncated: int,
        never_fired: int,
    ) -> None:
        if skipped:
            warnings.warn(
                f"ops.fault: skipped {skipped} of {total} subfaults with "
                f"non-positive moment / rise / peak time (taper edges carry "
                f"no radiated moment).",
                WarnFaultSubfaultSkipped, stacklevel=3,
            )
        if never_fired or truncated:
            warnings.warn(
                f"ops.fault: {never_fired} subfault(s) have a rupture onset "
                f"at/after t_total (their moment is never released) and "
                f"{truncated} have a rise tail past t_total (moment partly "
                f"truncated). Lengthen t_total to capture the full rupture.",
                WarnFaultSubfaultTruncated, stacklevel=3,
            )
        if clamped:
            warnings.warn(
                f"ops.fault: clamped peak_time below rise_time/2 on {clamped} "
                f"of {total} subfaults (Yoffe regularization bound) — their "
                f"slip-rate pulse is shorter (higher corner frequency) than "
                f"requested.",
                WarnFaultPeakClamped, stacklevel=3,
            )
