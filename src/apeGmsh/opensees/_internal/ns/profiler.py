"""
``_ProfilerNS`` — backs ``ops.profiler.<verb>(...)``.

The Ladruno fork's stack profiler is a *control* command that brackets
the analyze loop — not a model-definition primitive and not a recorder.
So, unlike the other namespaces (which build + ``_register`` a typed
primitive that emits during the model fan-out), each verb here records an
ordered ``(verb, args)`` entry on ``bridge._profiler_records``. The deck
emitters (:meth:`apeGmsh.opensees.apeSees.tcl` / :meth:`~.py`) flush those
records around the appended ``analyze`` line at emit time — ``start`` /
``reset`` before it, ``stop`` / ``report`` / ``memory`` after.

The fork command grammar (``ladruno:SRC/interpreter/OpenSeesCommands.cpp``
``OPS_profiler()``)::

    profiler start [-deep] [-memory] [-perStep]
    profiler stop
    profiler reset
    profiler report <filename> [-run <id>]
    profiler memory

The profiler exists only in the Ladruno fork build; emitting the deck text
works on any build, and the fork requirement is gated at run time (the live
emitter re-raises a friendly error; ``ops.tcl(run=True)`` surfaces the Tcl
interpreter's own error). Live single-call profiling is driven by the
``profile=`` kwarg family on :meth:`apeGmsh.opensees.apeSees.analyze`, not
by these recorded verbs — see ``internal_docs/plan_profiler_integration.md``.
"""
from __future__ import annotations

from ._base import _BridgeNamespace


__all__ = ["_ProfilerNS"]


class _ProfilerNS(_BridgeNamespace):
    """``ops.profiler.<verb>(...)`` — Ladruno-fork stack profiler control.

    Each verb records an ordered entry on ``bridge._profiler_records``;
    the deck emitters replay them bracketing the ``analyze`` line.
    """

    def start(
        self,
        *,
        deep: bool = False,
        memory: bool = False,
        per_step: bool = False,
    ) -> None:
        """``profiler start [-deep] [-memory] [-perStep]`` — begin a run.

        Coarse phase timing is always captured; ``deep`` adds fine-grained
        seam timing, ``memory`` enables the memory counters, ``per_step``
        records a per-step time series. Emitted *before* the ``analyze``
        line in the deck.
        """
        flags: list[str] = []
        if deep:
            flags.append("-deep")
        if memory:
            flags.append("-memory")
        if per_step:
            flags.append("-perStep")
        self._bridge._profiler_records.append(("start", tuple(flags)))

    def stop(self) -> None:
        """``profiler stop`` — end the run. Emitted *after* ``analyze``."""
        self._bridge._profiler_records.append(("stop", ()))

    def reset(self) -> None:
        """``profiler reset`` — clear trees/series (config kept).

        Emitted *before* ``analyze`` (it prepares a fresh run).
        """
        self._bridge._profiler_records.append(("reset", ()))

    def report(self, filename: str, *, run: str | None = None) -> None:
        """``profiler report <filename> [-run <id>]`` — append run to HDF5.

        Writes the profiled run to ``filename`` under run id ``run``.
        Emitted *after* ``analyze``. Read ``filename`` with the fork's
        out-of-tree ``Ladruno_tools/profiler_viewer`` (``ProfilerResults``
        headless API or the React viewer) — apeGmsh ships no reader.
        """
        args: tuple[str, ...] = (filename,)
        if run is not None:
            args = (filename, "-run", run)
        self._bridge._profiler_records.append(("report", args))

    def memory(self) -> None:
        """``profiler memory`` — emit a memory-snapshot line.

        Emitted *after* ``analyze``. The fork command also *returns* peak
        bytes when called live; that live-return path is deferred (this verb
        only records the deck line). See the plan's ``memory()`` decision.
        """
        self._bridge._profiler_records.append(("memory", ()))
