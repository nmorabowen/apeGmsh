"""LiveMPCO — in-process STKO MPCO recorder emission.

Issues a single ``ops.recorder("mpco", path, -N ..., -E ..., -T ...)``
call into the running openseespy domain on ``__enter__`` and removes
it on ``__exit__`` (which is what flushes the HDF5 file). The MPCO
recorder writes one file containing all stages — there is no
``begin_stage``/``end_stage`` ceremony, since pseudoTime in the
OpenSees domain handles stage scoping internally.

Build-gate
----------
Vanilla ``openseespy`` distributions do **not** include the MPCO
recorder. STKO ships its own openseespy build that does. If the
recorder is unavailable, ``__enter__`` raises with a pointer to the
two workable options (STKO's distribution, or ``spec.emit_recorders``
on the classic recorder path).

Usage
-----
::

    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    with spec.emit_mpco("run.mpco"):
        for _ in range(n_steps):
            ops.analyze(1, dt)

    results = Results.from_mpco("run.mpco")
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...solvers._recorder_emit import mpco_ops_args

if TYPE_CHECKING:
    from ...solvers._recorder_specs import ResolvedRecorderSpec


class LiveMPCO:
    """Context manager that owns a single in-process MPCO recorder.

    Parameters
    ----------
    spec
        The :class:`ResolvedRecorderSpec` whose records to emit.
    path
        Output ``.mpco`` HDF5 file path. Parent directory created on
        ``__enter__`` if missing.
    ops
        The openseespy module (or a stand-in for testing). Defaults
        to ``openseespy.opensees`` resolved lazily on ``__enter__``.

    Raises
    ------
    RuntimeError
        On ``__enter__`` if ``ops.recorder('mpco', ...)`` fails — most
        commonly because the openseespy build does not include the
        MPCO recorder.
    """

    def __init__(
        self,
        spec: "ResolvedRecorderSpec",
        path: "str | Path",
        *,
        ops=None,
    ) -> None:
        self._spec = spec
        self._path = Path(path)
        self._ops = ops

        self._opened = False
        self._exited = False
        self._tag: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "LiveMPCO":
        if self._opened:
            raise RuntimeError(
                "LiveMPCO is single-use; create a new instance."
            )
        self._opened = True

        if self._ops is None:
            import openseespy.opensees as ops_module
            self._ops = ops_module

        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)

        # Build the args. The leading "mpco" is included in the tuple.
        # Path is split into output_dir + filename so mpco_ops_args
        # produces a forward-slash joined path that ops.recorder is
        # happy with on every platform.
        out_dir = str(self._path.parent) if str(self._path.parent) != "." else ""
        args = mpco_ops_args(
            self._spec.records,
            output_dir=out_dir,
            filename=self._path.name,
        )

        try:
            tag = self._ops.recorder(*args)
        except Exception as exc:
            self._opened = False     # allow re-use after fixing the build
            raise RuntimeError(
                "ops.recorder('mpco', ...) failed. The most likely "
                "cause is that the active openseespy build does not "
                "include the MPCO recorder; vanilla openseespy "
                "distributions do not ship it. Workable options:\n"
                "  - run inside STKO's bundled Python distribution\n"
                "  - use spec.emit_recorders(...) for classic "
                "recorders + Results.from_recorders(...)\n"
                "  - export with g.opensees.export.tcl(..., "
                "recorders=spec, mpco=True) and run with STKO loaded"
            ) from exc

        if isinstance(tag, int):
            self._tag = tag

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Removing the recorder is what flushes the HDF5 file. Don't
        # raise from cleanup — preserve any in-flight exception.
        if self._tag is not None:
            try:
                self._ops.remove("recorder", self._tag)
            except Exception:  # noqa: BLE001 — best-effort flush
                warnings.warn(
                    f"LiveMPCO: failed to remove recorder tag "
                    f"{self._tag}; the .mpco file may not be flushed.",
                    stacklevel=2,
                )

        self._exited = True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def tag(self) -> Optional[int]:
        """The recorder tag returned by ``ops.recorder``, if any."""
        return self._tag

    @property
    def path(self) -> Path:
        return self._path
