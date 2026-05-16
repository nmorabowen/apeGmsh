"""``GroundMotion`` — immutable seismic time-history snapshot.

A :class:`GroundMotion` is a plain dataclass holding an acceleration
time history. The class is unit-agnostic: values are stored as-is, and
parsers accept a ``scale_factor`` kwarg that multiplies values at read
time. It is the caller's responsibility to know what units the record
is in and what units the consuming model expects.

Parsers in :mod:`apeGmsh.ground_motions._parsers` construct instances.
The :meth:`to_time_series` adapter declares them onto an
:class:`apeGmsh.opensees.apeSees` session.

The class is frozen — to apply a different scale factor or change
metadata, re-parse or construct a new instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apeGmsh.opensees import apeSees


@dataclass(frozen=True, slots=True, kw_only=True)
class GroundMotion:
    """Immutable acceleration time history.

    Values are stored as-is (no unit normalisation). The caller is
    responsible for tracking whether the numbers are in ``g``,
    ``m/s^2``, ``cm/s^2`` or anything else — typically by passing a
    ``scale_factor`` to the parser, or by post-multiplying with a
    ``factor`` at emit time.

    Parameters
    ----------
    accel
        1-D ``float64`` array of acceleration samples.
    dt
        Time step in seconds. For uniform sampling, the constant step;
        for non-uniform records, the mean Δt (kept for sizing/display).
        Optional — if ``time`` is given and ``dt`` is not, it is
        computed automatically.
    time
        Optional 1-D array of sample times in seconds. Pass when the
        record is non-uniformly sampled. ``None`` (default) means the
        record is uniform at :attr:`dt`. When given, must match
        ``len(accel)`` and be strictly increasing.
    source
        Human-readable provenance string (typically the filename).
        Empty by default.
    metadata
        Optional dict of raw header fields preserved by the parser.
        Read-only; do not mutate. Parsers that find unit declarations
        in the source file store them under ``metadata["units_declared"]``
        for inspection (the values are not transformed).
    """

    accel: np.ndarray
    dt: float | None = None
    time: np.ndarray | None = None
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise inputs. Use object.__setattr__ because dataclass is frozen.
        arr = np.ascontiguousarray(self.accel, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"accel must be 1-D, got shape {arr.shape}"
            )
        if arr.size < 2:
            raise ValueError("accel must contain at least 2 samples")

        time_arr: np.ndarray | None = None
        if self.time is not None:
            time_arr = np.ascontiguousarray(self.time, dtype=np.float64)
            if time_arr.ndim != 1:
                raise ValueError(
                    f"time must be 1-D, got shape {time_arr.shape}"
                )
            if time_arr.size != arr.size:
                raise ValueError(
                    f"time and accel must be the same length; "
                    f"got {time_arr.size} and {arr.size}"
                )
            if np.any(np.diff(time_arr) <= 0):
                raise ValueError("time must be strictly increasing")

        # Resolve dt. If time given, derive mean Δt; otherwise dt is required.
        if time_arr is not None:
            derived_dt = float(np.mean(np.diff(time_arr)))
            resolved_dt = float(self.dt) if self.dt is not None else derived_dt
        else:
            if self.dt is None:
                raise ValueError(
                    "dt is required when time is not given "
                    "(uniform-sampling case)"
                )
            resolved_dt = float(self.dt)

        if not np.isfinite(resolved_dt) or resolved_dt <= 0:
            raise ValueError(f"dt must be positive, got {resolved_dt}")

        object.__setattr__(self, "accel", arr)
        object.__setattr__(self, "time", time_arr)
        object.__setattr__(self, "dt", resolved_dt)

    # ── Derived properties ────────────────────────────────────────────
    @property
    def npts(self) -> int:
        return int(self.accel.size)

    @property
    def is_uniform(self) -> bool:
        """``True`` if the record has no stored time vector."""
        return self.time is None

    @property
    def time_vector(self) -> np.ndarray:
        """Sample times in seconds.

        For uniform records this is synthesised on demand. For
        non-uniform records, the stored :attr:`time` is returned.
        """
        if self.time is not None:
            return self.time
        return np.arange(self.npts, dtype=np.float64) * self.dt  # type: ignore[operator]

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self.time is not None:
            return float(self.time[-1] - self.time[0])
        return (self.npts - 1) * float(self.dt)  # type: ignore[arg-type]

    @property
    def pga(self) -> float:
        """Peak ground acceleration in the record's native units
        (absolute value)."""
        return float(np.max(np.abs(self.accel)))

    # ── OpenSees adapter ──────────────────────────────────────────────
    def to_time_series(
        self,
        ops: "apeSees",
        *,
        factor: float = 1.0,
        prepend_zero: bool = False,
    ):
        """Declare an ``ops.timeSeries.Path`` from this record.

        Parameters
        ----------
        ops
            The :class:`apeGmsh.opensees.apeSees` instance to declare
            against.
        factor
            Multiplier passed to ``ops.timeSeries.Path(factor=...)``.
            Use this for unit conversion at emit time (e.g.
            ``factor=9.81`` to push a record-in-``g`` into an SI model).
            OpenSees applies the factor at runtime, so the underlying
            values stay readable. Defaults to ``1.0`` (no scaling).
        prepend_zero
            Forwarded to ``ops.timeSeries.Path`` — prepend a ``(0, 0)``
            sample so the structure starts at rest. Useful for records
            that begin at non-zero acceleration.

        Returns
        -------
        The :class:`Path` primitive returned by ``ops.timeSeries.Path``.
        """
        kwargs: dict[str, Any] = {
            "values": tuple(self.accel.tolist()),
            "prepend_zero": prepend_zero,
        }
        if factor != 1.0:
            kwargs["factor"] = factor
        # Uniform → dt=, non-uniform → time= (the Path primitive
        # accepts exactly one of the two).
        if self.is_uniform:
            kwargs["dt"] = self.dt
        else:
            kwargs["time"] = tuple(self.time.tolist())  # type: ignore[union-attr]
        return ops.timeSeries.Path(**kwargs)

    # ── Constructors (classmethods) ───────────────────────────────────
    #
    # Each delegates to a parser in ``apeGmsh.ground_motions._parsers``.
    # Imports are local to avoid a circular dependency at module load.

    @classmethod
    def from_two_column(
        cls,
        path,
        *,
        scale_factor: float = 1.0,
        uniform_rtol: float = 1e-4,
    ) -> "GroundMotion":
        """Read a generic ``time accel`` two-column file."""
        from ._parsers import read_two_column
        return read_two_column(
            path, scale_factor=scale_factor, uniform_rtol=uniform_rtol
        )

    @classmethod
    def from_one_column(
        cls,
        path,
        *,
        dt: float,
        scale_factor: float = 1.0,
        skiprows: int = 0,
    ) -> "GroundMotion":
        """Read a generic accel-only file (one or many values per line)."""
        from ._parsers import read_one_column
        return read_one_column(
            path, dt=dt, scale_factor=scale_factor, skiprows=skiprows
        )

    @classmethod
    def from_peer_at2(
        cls,
        path,
        *,
        scale_factor: float = 1.0,
    ) -> "GroundMotion":
        """Read a PEER NGA ``.AT2`` record (4-line header)."""
        from ._parsers import read_peer_at2
        return read_peer_at2(path, scale_factor=scale_factor)

    @classmethod
    def from_itaca(
        cls,
        path,
        *,
        scale_factor: float = 1.0,
    ) -> "GroundMotion":
        """Read an ITACA / ESM ASCII record (keyword-value header)."""
        from ._parsers import read_itaca
        return read_itaca(path, scale_factor=scale_factor)

    @classmethod
    def from_obspy(
        cls,
        path,
        *,
        scale_factor: float = 1.0,
        component: int = 0,
    ) -> "GroundMotion":
        """Read *path* via obspy (K-NET, SAC, MiniSEED, ...). Requires obspy."""
        from ._parsers import read_obspy
        return read_obspy(
            path, scale_factor=scale_factor, component=component
        )

    @classmethod
    def from_obspy_trace(
        cls,
        trace,
        *,
        scale_factor: float = 1.0,
    ) -> "GroundMotion":
        """Wrap an existing :class:`obspy.Trace` as a :class:`GroundMotion`."""
        from ._parsers import from_obspy_trace
        return from_obspy_trace(trace, scale_factor=scale_factor)

    @classmethod
    def from_file(
        cls,
        path,
        *,
        scale_factor: float = 1.0,
        dt: float | None = None,
        obspy_component: int = 0,
    ) -> "GroundMotion":
        """Sniff the file format and dispatch to the right parser.

        Falls back to the obspy bridge for unrecognised formats (which
        requires obspy installed).
        """
        from ._sniffer import from_file
        return from_file(
            path,
            scale_factor=scale_factor,
            dt=dt,
            obspy_component=obspy_component,
        )

    # ── Repr ──────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        src = f" source={self.source!r}" if self.source else ""
        sampling = "uniform" if self.is_uniform else "non-uniform"
        return (
            f"GroundMotion(npts={self.npts}, dt={self.dt:g} ({sampling}), "
            f"duration={self.duration:g}s, pga={self.pga:g}{src})"
        )
