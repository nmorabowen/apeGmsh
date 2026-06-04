"""Reader for the Ladruno live-telemetry **Monitor** sink (``recorder Monitor``).

The Monitor is a *lightweight SWMR-HDF5 sidecar* — **not** the canonical
``.ladruno`` recorder and **not** a :class:`~apeGmsh.results.Results` object.
It carries only a handful of selected nodal scalars (one column per
node × dof × response), so it surfaces as a thin time-history rather than a
full FEM result:

* :func:`read_monitor` — read a finished / at-rest sink into
  :class:`MonitorData`.
* :func:`tail_monitor` — *follow* a sink another process is still writing,
  yielding each frame as it lands (the live path the Monitor exists for).

On-disk layout (``FORMAT="ladruno-monitor"``, ``FORMAT_VERSION=1``)::

    /                attrs: FORMAT, FORMAT_VERSION, GENERATOR
      COLUMNS  [nCols]   vlen str — channel labels ``node<N>.<resp>.dof<D>``
      STEP     [T]       int32     — commitTag per frame
      TIME     [T]       float64   — pseudo-time per frame
      FRAMES   [T,nCols] float64   — one appended row per frame

Channels are in **node-major** order (``node2.dof1, node2.dof2,
node3.dof1, …``); the labels are self-describing, so consumers key on the
label, never the position.

**Cross-process tailing.** The sink is written in HDF5 SWMR mode. A reader
in a *separate process* from the running solver may need
``HDF5_USE_FILE_LOCKING=FALSE`` in its environment *before* Python imports
``h5py`` (a libhdf5 quirk on networked / locked filesystems). Within one
process — or for at-rest reads — no special handling is needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import h5py
    import pandas as pd

_MONITOR_FORMAT = "ladruno-monitor"


def _decode(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _read_columns(f: "h5py.File") -> tuple[str, ...]:
    return tuple(_decode(c) for c in f["COLUMNS"][:])


def _validate_monitor(f: "h5py.File", path: Path) -> None:
    fmt = _decode(f.attrs["FORMAT"]) if "FORMAT" in f.attrs else ""
    if fmt != _MONITOR_FORMAT:
        raise ValueError(
            f"{path} is not a Ladruno monitor sink: FORMAT is {fmt!r}, "
            f"expected {_MONITOR_FORMAT!r}. read_monitor expects a file "
            "written by 'recorder Monitor'."
        )
    for ds in ("COLUMNS", "STEP", "TIME", "FRAMES"):
        if ds not in f:
            raise ValueError(f"{path}: monitor sink missing dataset {ds!r}.")


@dataclass(frozen=True)
class MonitorData:
    """An at-rest snapshot of a Monitor sink.

    ``frames`` is ``[T, nCols]`` aligned with ``columns`` (channel labels)
    and the per-frame ``step`` / ``time`` axes.
    """

    columns: tuple[str, ...]
    step: ndarray
    time: ndarray
    frames: ndarray

    @property
    def n_frames(self) -> int:
        return int(self.step.size)

    def channel(self, label: str) -> ndarray:
        """The ``[T]`` time-history of one channel by its label
        (``"node2.disp.dof1"``). Raises ``KeyError`` if absent."""
        try:
            col = self.columns.index(label)
        except ValueError:
            raise KeyError(
                f"channel {label!r} not in monitor sink; available: "
                f"{list(self.columns)}."
            ) from None
        return self.frames[:, col]

    def to_dataframe(self, *, index: str = "time") -> "pd.DataFrame":
        """A :class:`pandas.DataFrame` (one column per channel label).

        ``index="time"`` (default) indexes by pseudo-time; ``index="step"``
        by the recorder commitTag.
        """
        if index not in ("time", "step"):
            raise ValueError(
                f"index must be 'time' or 'step', got {index!r}."
            )
        import pandas as pd

        axis = self.time if index == "time" else self.step
        return pd.DataFrame(
            self.frames,
            columns=list(self.columns),
            index=pd.Index(axis, name=index),
        )


def read_monitor(path: "str | Path") -> MonitorData:
    """Read a finished / at-rest Monitor sink into :class:`MonitorData`.

    Raises :class:`ValueError` if ``path`` is not a ``ladruno-monitor``
    file (e.g. a ``.ladruno`` or foreign HDF5).
    """
    import h5py

    p = Path(path)
    with h5py.File(p, "r") as f:
        _validate_monitor(f, p)
        return MonitorData(
            columns=_read_columns(f),
            step=np.asarray(f["STEP"][:], dtype=np.int64).flatten(),
            time=np.asarray(f["TIME"][:], dtype=np.float64).flatten(),
            frames=np.asarray(f["FRAMES"][:], dtype=np.float64).reshape(
                f["STEP"].shape[0], -1,
            ),
        )


def tail_monitor(
    path: "str | Path",
    *,
    start: int = 0,
    poll: float = 0.1,
    timeout: "Optional[float]" = None,
) -> Iterator[tuple[int, float, ndarray]]:
    """Follow a Monitor sink another process is still writing (SWMR).

    Opens ``path`` in SWMR-read mode and yields ``(step, time, frame_row)``
    for each frame appended after index ``start``, refreshing the datasets
    between polls. Works equally on a still-growing file and a finished one
    (it drains whatever is already there first).

    Stops when ``timeout`` seconds elapse with no new frame (``None`` =
    follow forever — the caller breaks out of the loop). ``poll`` is the
    sleep between refresh checks. ``frame_row`` is the ``[nCols]`` row in
    ``columns`` order; read the column labels once via :func:`read_monitor`
    or open the file yourself.

    For a reader in a *separate process* from the solver, set
    ``HDF5_USE_FILE_LOCKING=FALSE`` before importing ``h5py`` (see the
    module docstring).
    """
    import h5py

    p = Path(path)
    with h5py.File(p, "r", swmr=True) as f:
        _validate_monitor(f, p)
        step_ds = f["STEP"]
        time_ds = f["TIME"]
        frame_ds = f["FRAMES"]
        n = int(start)
        idle = 0.0
        while True:
            step_ds.id.refresh()
            total = step_ds.shape[0]
            if total > n:
                time_ds.id.refresh()
                frame_ds.id.refresh()
                steps = np.asarray(step_ds[n:total], dtype=np.int64).flatten()
                times = np.asarray(time_ds[n:total], dtype=np.float64).flatten()
                rows = np.asarray(frame_ds[n:total], dtype=np.float64)
                for i in range(total - n):
                    yield int(steps[i]), float(times[i]), rows[i]
                n = total
                idle = 0.0
                continue
            if timeout is not None and idle >= timeout:
                return
            time.sleep(poll)
            idle += poll


__all__ = ["MonitorData", "read_monitor", "tail_monitor"]
