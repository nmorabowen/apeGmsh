"""Opt-in ObsPy bridge for the long tail of seismic formats.

ObsPy ships native readers for ~30 strong-motion formats (K-NET,
KiK-net, SAC, MiniSEED, SEISAN, GSE2, GCF, ...). Rather than
re-implement them, we delegate via a lazy import.

ObsPy is a heavy dependency (~50 MB on disk, pulls in scipy). Users
who never touch obspy formats pay nothing — the import only happens
when one of these functions is called.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .._motion import GroundMotion

if TYPE_CHECKING:  # pragma: no cover - typing only
    import obspy


_OBSPY_INSTALL_HINT = (
    "ObsPy is required for this format. Install it with "
    "`pip install obspy` and retry."
)


def _import_obspy():
    try:
        import obspy  # noqa: F401
    except ImportError as exc:
        raise ImportError(_OBSPY_INSTALL_HINT) from exc
    return obspy


def from_obspy_trace(
    trace: "obspy.Trace",
    *,
    scale_factor: float = 1.0,
) -> GroundMotion:
    """Wrap an existing :class:`obspy.Trace` in a :class:`GroundMotion`.

    Parameters
    ----------
    trace
        An obspy ``Trace`` whose ``data`` is the acceleration history.
    scale_factor
        Multiplier applied to ``trace.data`` at wrap time. Default
        ``1.0`` (values stored as-is). ObsPy's unit conventions vary
        by format — apply the conversion you need at this seam.

    Returns
    -------
    A :class:`GroundMotion` snapshot. Selected trace stats are
    preserved under ``metadata``.
    """
    data = np.ascontiguousarray(trace.data, dtype=np.float64) * scale_factor
    stats = trace.stats
    dt = float(stats.delta)

    metadata: dict[str, Any] = {
        "format": "obspy",
        "network": str(getattr(stats, "network", "")),
        "station": str(getattr(stats, "station", "")),
        "channel": str(getattr(stats, "channel", "")),
        "location": str(getattr(stats, "location", "")),
        "starttime": str(getattr(stats, "starttime", "")),
        "scale_factor": scale_factor,
    }
    return GroundMotion(
        accel=data,
        dt=dt,
        source=f"obspy:{metadata['station']}.{metadata['channel']}",
        metadata=metadata,
    )


def read_obspy(
    path: str | os.PathLike[str],
    *,
    scale_factor: float = 1.0,
    component: int = 0,
) -> GroundMotion:
    """Read *path* with ``obspy.read()`` and return the *component*-th trace.

    Parameters
    ----------
    path
        Any file format obspy can read (K-NET ``.NS`` / ``.EW`` /
        ``.UD``, SAC, MiniSEED, SEISAN, GSE2, ...).
    scale_factor
        Multiplier applied to the trace data. Default ``1.0``.
    component
        Index into the obspy ``Stream`` to use. Multi-component
        formats return more than one trace; this picks one.

    Returns
    -------
    A :class:`GroundMotion` snapshot.

    Raises
    ------
    ImportError
        If obspy is not installed.
    ValueError
        If the file contains no traces, or *component* is out of range.
    """
    obspy = _import_obspy()
    stream = obspy.read(os.fspath(path))
    if len(stream) == 0:
        raise ValueError(f"{path}: obspy read returned no traces")
    if component >= len(stream):
        raise ValueError(
            f"{path}: requested component {component} but stream has "
            f"only {len(stream)} traces"
        )
    return from_obspy_trace(stream[component], scale_factor=scale_factor)
