"""Live-properties build controller (ADR 0080 B6).

The builder GUI edits a :class:`~apeGmsh.sections.SectionDocument`; the
live properties panel wants the analyzer numbers for the current
document state. Building + analyzing a continuum section runs a private
Gmsh session and a warping/plastic solve — **far too heavy for the UI
thread** (the S6 no-solve-on-the-UI-thread law). This module runs that
work in a background thread and marshals the result back to the UI
thread, with:

* **memoization** by canonical document state — an identical state
  never rebuilds;
* **coalescing** — a burst of edits while a build is in flight collapses
  to a single follow-up build of the *latest* state (N edits → ≤ N
  builds, last state wins);
* **staleness dropping** — a result for a state that is no longer the
  latest requested is cached but not delivered.

The controller is Qt-light: it owns a ``QTimer`` that drains a
thread-safe result queue on the UI thread. Tests inject a blocking
builder and drive :meth:`PropertiesController.drain` manually for
determinism. The heavy build itself is :func:`build_document`, injected
so tests never touch Gmsh.
"""
from __future__ import annotations

import contextlib
import gc
import json
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from ._analysis import SectionProperties

__all__ = [
    "BuildResult",
    "PropertiesController",
    "build_document",
    "fiber_identities",
]


@dataclass
class BuildResult:
    """The outcome of building one document state off the UI thread.

    Exactly one of (``analysis``, ``identities``) is set on success;
    ``error`` is set instead when the build failed. ``worker_thread_id``
    is the id of the thread that ran the build — the
    no-solve-on-the-UI-thread proof.
    """

    key: str
    kind: str
    analysis: "SectionProperties | None" = None
    stress_available: bool = False
    identities: "dict[str, Any] | None" = None
    error: "str | None" = None
    worker_thread_id: "int | None" = None


# ─────────────────────────────────────────────────────────────────────
# Qt-safety: the cyclic GC must not run on the worker thread
# ─────────────────────────────────────────────────────────────────────

#: Serializes the pause/resume bookkeeping below.
_GC_GUARD = threading.Lock()
#: How many workers are currently inside :func:`_no_cyclic_gc`.
_GC_HOLDERS = 0
#: Whether the cyclic GC was enabled when the first holder arrived.
_GC_WAS_ENABLED = False


@contextlib.contextmanager
def _no_cyclic_gc() -> "Iterator[None]":
    """Pause automatic cyclic collection for the length of one build.

    **Why this exists.** Qt objects belong to the thread that created
    them: a PySide6 wrapper finalized off the GUI thread is handed to
    ``Shiboken::BindingManager``'s deferred-deletion queue, which the
    main thread later drains through ``Py_AddPendingCall``. By then the
    queued entry no longer describes a live object, and the destructor
    call jumps through it — SIGSEGV inside ``runDeletionInMainThread()``,
    with no Python traceback and below the reach of any ``except``.

    :func:`_work` already keeps Qt out of everything this thread can
    *reach* (see its docstring). That is necessary and was not
    sufficient: the cyclic collector runs on whichever thread happens to
    trip the allocation threshold, and this one allocates hard —
    document parse, ``FEMData`` unpickle, the NumPy solve. One gen-2
    pass landing here while some earlier builder window's widget tree is
    unreachable finalizes *that* Qt graph on this thread, and the main
    thread dies draining the queue at whatever test it had reached.

    Refcount-driven frees are unaffected, so this does not leak: it only
    defers *cyclic* garbage to the next collection after the build, on
    whatever thread runs it then. Overlapping builds are refcounted, and
    an interpreter that already had the collector off keeps it off.
    """
    global _GC_HOLDERS, _GC_WAS_ENABLED
    with _GC_GUARD:
        if _GC_HOLDERS == 0:
            _GC_WAS_ENABLED = gc.isenabled()
            gc.disable()
        _GC_HOLDERS += 1
    try:
        yield
    finally:
        with _GC_GUARD:
            _GC_HOLDERS -= 1
            if _GC_HOLDERS == 0 and _GC_WAS_ENABLED:
                gc.enable()


def canonical_state(doc_dict: "dict[str, Any]") -> str:
    """A stable string key for a document dict (memoization key)."""
    return json.dumps(doc_dict, sort_keys=True)


def fiber_identities(recipe: Any) -> "dict[str, Any]":
    """Exact fiber-sum identities for a :class:`FiberRecipe` (cheap, no
    solve): total area, per-material area, item counts, and ``GJ``."""
    areas = recipe.areas_by_material()
    return {
        "total_area": sum(areas.values()),
        "areas_by_material": dict(areas),
        "n_patches": len(recipe.patches),
        "n_layers": len(recipe.layers),
        "n_points": len(recipe.points),
        "GJ": recipe.GJ,
    }


def build_document(doc_dict: "dict[str, Any]") -> BuildResult:
    """Build + analyze one document state (the default heavy builder).

    Continuum: mesh → analyzer → geometric/warping/plastic + unit stress
    fields (via
    :func:`~apeGmsh.sections._inspector.precompute_analyses`). Fiber: the
    deterministic recipe expansion → :func:`fiber_identities`. Any
    failure (unset mesh, disconnected section, mesh error) is captured
    as ``error`` rather than raised — a bad edit greys the panel, it
    does not crash the worker.

    **The meshing runs in a child process; the solve runs here.** Gmsh
    is one process-global, non-reentrant C++ runtime: driving it from
    this worker while the main thread drives its own sessions aborts the
    interpreter outright, below the reach of the ``except`` below.
    Serializing with the runtime lock would be safe but not *useful*
    here — a session the user left open holds that lock for its whole
    lifetime, so the panel would never refresh. Meshing out of process
    removes the contention instead (:mod:`._mesh_proc`).

    The analyzer stays in-process on this worker thread: it is pure
    NumPy over the returned snapshot, it is the expensive half, and
    keeping it here leaves the panel a **live** ``SectionProperties`` to
    drive interactively.
    """
    from ._document import SectionDocument
    from ._inspector import precompute_analyses
    from ._mesh_proc import mesh_document

    key = canonical_state(doc_dict)
    try:
        doc = SectionDocument(doc_dict)
        if doc.kind == "continuum":
            fem = mesh_document(doc_dict)               # child process
            analysis = doc.analysis_from_fem(fem)
            stress_ok = precompute_analyses(analysis)   # here, no gmsh
            return BuildResult(
                key=key, kind="continuum",
                analysis=analysis, stress_available=stress_ok,
            )
        recipe = doc.build()        # fiber lane opens no session
        return BuildResult(
            key=key, kind="fiber",
            identities=fiber_identities(recipe),
        )
    except Exception as exc:  # worker isolation — never propagate
        return BuildResult(key=key, kind=doc_dict.get("kind", "?"),
                           error=str(exc))


def _work(
    builder: "Callable[[dict[str, Any]], BuildResult]",
    results: "queue.Queue[BuildResult]",
    key: str,
    doc_dict: "dict[str, Any]",
) -> None:
    """Runs on the worker thread — the only off-UI-thread code.

    **A module-level function on purpose, not a method.** A bound method
    would make the *running thread* an owner of the controller, and the
    controller reaches Qt: its ``QTimer``, and through ``on_result`` the
    whole builder window. A build that outlives its window then drops
    the last reference to that Qt graph on this thread. PySide6 defers
    such deletions to the main thread (Shiboken's
    ``runDeletionInMainThread``) and the deferred pass segfaults the
    interpreter — exit 139, no Python traceback, reported at whichever
    code the main thread happened to reach rather than here. Taking the
    builder and the queue as plain arguments keeps this thread's
    reachable set Qt-free, so the last drop always lands on the UI
    thread.

    **That is necessary but not sufficient**, which is why the body runs
    under :func:`_no_cyclic_gc`. Reachability governs only what *this*
    thread's own references can free; the cyclic collector can run here
    and finalize a Qt graph this thread never touched. See #1080 — the
    crash survived the reachability fix and the core still named
    ``runDeletionInMainThread``.
    """
    with _no_cyclic_gc():
        try:
            res = builder(doc_dict)
        except Exception as exc:  # pragma: no cover - builder isolation
            res = BuildResult(key=key, kind="?", error=str(exc))
        res.key = key
        res.worker_thread_id = threading.get_ident()
        results.put(res)


class PropertiesController:
    """Runs document builds off the UI thread and delivers fresh results
    back on it.

    ``on_result`` is invoked on the UI thread with the freshest
    :class:`BuildResult` whenever a build for the latest-requested state
    completes (or is served from cache). ``builder`` is injectable
    (tests supply a blocking stub so no Gmsh runs); ``poll_ms`` sets the
    result-drain cadence of the internal ``QTimer``.
    """

    def __init__(
        self,
        *,
        builder: "Callable[[dict[str, Any]], BuildResult] | None" = None,
        on_result: "Callable[[BuildResult], None] | None" = None,
        poll_ms: int = 40,
        autostart_timer: bool = True,
    ) -> None:
        self._builder = builder or build_document
        self._on_result = on_result
        self._cache: dict[str, BuildResult] = {}
        self._results: "queue.Queue[BuildResult]" = queue.Queue()
        self._latest_key: str | None = None
        self._running = False
        self._pending: "tuple[str, dict[str, Any]] | None" = None
        self._threads: list[threading.Thread] = []
        #: total number of heavy builds actually dispatched (the
        #: coalescing/memoization test surface).
        self.build_count = 0

        self._timer: Any = None
        if autostart_timer:
            self._start_timer(poll_ms)

    def _start_timer(self, poll_ms: int) -> None:
        from qtpy.QtCore import QTimer

        self._timer = QTimer()
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self.drain)
        self._timer.start()

    # ── request / dispatch ───────────────────────────────────────────

    def request(self, doc_dict: "dict[str, Any]") -> None:
        """Ask for the properties of ``doc_dict``. Cheap and non-blocking
        — the heavy build runs on a worker thread; ``on_result`` fires
        later on the UI thread."""
        key = canonical_state(doc_dict)
        self._latest_key = key
        if self._running:
            self._pending = (key, doc_dict)   # coalesce — keep latest
            return
        self._launch_or_serve(key, doc_dict)

    def _launch_or_serve(self, key: str, doc_dict: "dict[str, Any]") -> None:
        cached = self._cache.get(key)
        if cached is not None:
            if key == self._latest_key and self._on_result is not None:
                self._on_result(cached)      # memoized — no build
            return
        self._running = True
        self.build_count += 1
        t = threading.Thread(
            target=_work,                       # module-level: see above
            args=(self._builder, self._results, key, doc_dict),
            daemon=True,
        )
        self._threads.append(t)
        t.start()

    # ── drain (UI thread: QTimer tick or manual in tests) ────────────

    def drain(self) -> int:
        """Deliver any completed results on the UI thread; dispatch the
        coalesced pending build if the current one just finished.
        Returns the number of results drained."""
        n = 0
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                break
            n += 1
            self._cache[res.key] = res
            self._running = False
            if res.key == self._latest_key and self._on_result is not None:
                self._on_result(res)
        if not self._running and self._pending is not None:
            key, doc_dict = self._pending
            self._pending = None
            self._launch_or_serve(key, doc_dict)
        return n

    def join(self, timeout: "float | None" = None) -> None:
        """Join every worker thread started so far (test helper)."""
        for t in list(self._threads):
            t.join(timeout)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
