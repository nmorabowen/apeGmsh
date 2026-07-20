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

import json
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover
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

    Continuum: a private Gmsh session → analyzer → geometric/warping/
    plastic + unit stress fields (via
    :func:`~apeGmsh.sections._inspector.precompute_analyses`). Fiber: the
    deterministic recipe expansion → :func:`fiber_identities`. Any
    failure (unset mesh, disconnected section, mesh error) is captured
    as ``error`` rather than raised — a bad edit greys the panel, it
    does not crash the worker.
    """
    from ._document import SectionDocument
    from ._inspector import precompute_analyses

    key = canonical_state(doc_dict)
    try:
        doc = SectionDocument(doc_dict)
        if doc.kind == "continuum":
            analysis = doc.build()
            stress_ok = precompute_analyses(analysis)
            return BuildResult(
                key=key, kind="continuum",
                analysis=analysis, stress_available=stress_ok,
            )
        recipe = doc.build()
        return BuildResult(
            key=key, kind="fiber",
            identities=fiber_identities(recipe),
        )
    except Exception as exc:  # worker isolation — never propagate
        return BuildResult(key=key, kind=doc_dict.get("kind", "?"),
                           error=str(exc))


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
            target=self._work, args=(key, doc_dict), daemon=True,
        )
        self._threads.append(t)
        t.start()

    def _work(self, key: str, doc_dict: "dict[str, Any]") -> None:
        """Runs on the worker thread — the only off-UI-thread code."""
        try:
            res = self._builder(doc_dict)
        except Exception as exc:  # pragma: no cover - builder isolation
            res = BuildResult(key=key, kind="?", error=str(exc))
        res.key = key
        res.worker_thread_id = threading.get_ident()
        self._results.put(res)

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
