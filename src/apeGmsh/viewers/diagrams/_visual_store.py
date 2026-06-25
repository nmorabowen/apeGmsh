"""VisualDataStore - eager float32 cache for the post-solve viewer.

Pure-visual performance layer for the post-solve viewer. Caches
full-time slabs as float32 in RAM so the time scrubber slices a row
from memory instead of re-reading HDF5 every frame, and accumulates
per-component vmin/vmax during the single load pass so color limits
never need a rescan (ShakerMaker vmax-sidecar idea, computed
live - no sidecar file).

Why float32
-----------
The viewer per-step diagrams feed a VTK/PyVista mapper whose scalars
are float32 anyway, so float32 is the native mapper width and is the
exact precision the colour pipeline consumes. float32 halves the
resident footprint versus the float64 the public readers return, so
more components stay cached at once - while staying finite for the
full demand range (stress in Pa overflows float16, which tops out at
65504).

What is NOT touched
-------------------
The public readers read_nodes / read_gauss stay float64 -
results.plot (publication matplotlib), inspect and exporters
keep full precision. This store is opt-in: the director owns one and
the registry stamps it on each diagram at attach. Diagrams fall back
to the per-step read path when no store is stamped (headless tests,
pre-bind, or a component the store could not load).

Memory policy
-------------
Eager by default - load_stage materializes every node + gauss
component for the requested stage. An optional
APEGMSH_VIEWER_CACHE_BYTES cap (or byte_budget= ctor kwarg)
stops further eager pre-fetch once the budget is hit; missing
components then load lazily on first access. A lazy (live) load always
succeeds — after storing it, the cache evicts the LEAST-recently-used
entries until back under the budget, but never the entry just loaded
and never the last remaining entry, so a render is never refused. The
LRU stamp is bumped on every slab access, so the hot (currently
scrubbed) stage/component survives while cold ones are dropped first.
Default (no env, no kwarg) = unbounded, no eviction.
"""
from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results

# Env override for the eager-prefetch byte ceiling. Absent = unbounded.
_ENV_CAP = "APEGMSH_VIEWER_CACHE_BYTES"

# Opt-in eager pre-fetch of EVERY component on bind. Default off: the
# store lazy-loads each component on first access (one full-time read
# per component, then 30-fps playback from RAM). Set to "1" to preload
# the whole stage up-front - fine for small runs, but for megapesados
# (e.g. big .ladruno soil files) it blocks the viewer open for a long
# time and blows memory, so it stays off by default.
_ENV_EAGER = "APEGMSH_VIEWER_EAGER"


def _env_eager() -> bool:
    return os.environ.get(_ENV_EAGER, "").strip() in ("1", "true", "yes", "on")


def _env_cap() -> Optional[int]:
    raw = os.environ.get(_ENV_CAP)
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


class _Entry:
    """One cached full-time slab (float32 values) + colour limits.

    slab is the reader slab with .values replaced by a float32 array
    (same shape, same metadata). vmin / vmax are the finite min/max of
    that float32 array, computed once during the load pass.
    """

    __slots__ = ("slab", "vmin", "vmax", "kind", "nbytes")

    def __init__(self, slab: Any, vmin: float, vmax: float, kind: str) -> None:
        self.slab = slab
        self.vmin = vmin
        self.vmax = vmax
        self.kind = kind
        self.nbytes = int(np.asarray(slab.values).nbytes)


def _finite_minmax(arr: ndarray) -> "tuple[float, float]":
    """Finite min/max of a (possibly float32) array; (nan, nan) if none."""
    a = np.asarray(arr).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(finite.min()), float(finite.max()))


def _replace_values(slab: Any, values: ndarray) -> Any:
    """Return a copy of slab with .values swapped (frozen dataclass)."""
    return dataclasses.replace(slab, values=values)


class VisualDataStore:
    """Eager float32 cache of full-time slabs, keyed by (stage, component).

    One instance lives on the ResultsDirector for the viewer session
    and is shared by every diagram (one Results -> one director -> one
    store). Diagrams never touch HDF5 through this store; they ask for
    a slab and slice a row.
    """

    def __init__(self, *, byte_budget: "int | None" = None) -> None:
        # None = unbounded eager. The env cap is the fallback so tests
        # and scripts can throttle without touching call sites.
        self._budget: Optional[int] = (
            byte_budget if byte_budget is not None else _env_cap()
        )
        # (stage_id, component) -> _Entry  (kind recorded on the entry)
        self._cache: "dict[tuple[str, str], _Entry]" = {}
        self._loaded_bytes: int = 0
        # LRU bookkeeping: monotone tick stamped on every slab access/store.
        # Used only when a byte budget is set (default None = no eviction).
        self._tick: int = 0
        self._last_access: "dict[tuple[str, str], int]" = {}

    def _touch(self, key: "tuple[str, str]") -> None:
        self._tick += 1
        self._last_access[key] = self._tick

    def _evict_to_budget(self, protect: "tuple[str, str]") -> None:
        """Evict least-recently-used entries until under the byte budget.

        Never evicts ``protect`` (the entry a live request just loaded) and
        never empties the cache — so a render is never refused, only older
        cached stages/components are dropped. No-op when no budget is set.
        """
        if self._budget is None:
            return
        while (
            self._loaded_bytes > self._budget
            and len(self._cache) > 1
        ):
            victim = min(
                (k for k in self._cache if k != protect),
                key=lambda k: self._last_access.get(k, 0),
                default=None,
            )
            if victim is None:
                break
            entry = self._cache.pop(victim)
            self._last_access.pop(victim, None)
            self._loaded_bytes -= entry.nbytes
        if self._loaded_bytes < 0:
            self._loaded_bytes = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop every cached entry (full reset)."""
        self._cache.clear()
        self._last_access.clear()
        self._loaded_bytes = 0

    def invalidate_stage(self, stage_id: str) -> None:
        """Drop only the entries belonging to stage_id."""
        for key in [k for k in self._cache if k[0] == stage_id]:
            entry = self._cache.pop(key)
            self._last_access.pop(key, None)
            self._loaded_bytes -= entry.nbytes
        if self._loaded_bytes < 0:
            self._loaded_bytes = 0

    @property
    def loaded_bytes(self) -> int:
        return self._loaded_bytes

    @property
    def byte_budget(self) -> Optional[int]:
        return self._budget

    def _under_budget(self) -> bool:
        return self._budget is None or self._loaded_bytes < self._budget

    # ------------------------------------------------------------------
    # Eager load
    # ------------------------------------------------------------------
    def load_stage(self, results: "Results", stage_id: str) -> None:
        """Eagerly materialize every node + gauss component for stage_id.

        Stops once the byte budget is hit (remaining components load
        lazily on first access). Errors per-component are swallowed so
        one unreadable component never blocks the rest of the stage.
        """
        try:
            scoped = results.stage(stage_id)
        except Exception:
            return
        # Nodes first - they are the hot path for demand playback.
        for component in self._available(scoped, "nodes"):
            if not self._under_budget():
                break
            self._load_nodes(scoped, stage_id, component)
        for component in self._available(scoped, "gauss"):
            if not self._under_budget():
                break
            self._load_gauss(scoped, stage_id, component)

    @staticmethod
    def _available(scoped: "Results", kind: str) -> "list[str]":
        try:
            if kind == "nodes":
                return list(scoped.nodes.available_components())
            return list(scoped.elements.gauss.available_components())
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Lazy accessors (the diagram hot path)
    # ------------------------------------------------------------------
    def nodes_slab(self, scoped: "Results", stage_id: str, component: str) -> Any:
        """Full-time NodeSlab with float32 values, or None.

        Lazily loads on first miss (a live request always renders, even
        past the eager budget). Returns None if the component is not
        recorded in this stage.
        """
        key = (stage_id, component)
        entry = self._cache.get(key)
        if entry is None:
            if not self._load_nodes(scoped, stage_id, component):
                return None
            entry = self._cache.get(key)
        if entry is not None:
            self._touch(key)
        return entry.slab if entry is not None else None

    def gauss_slab(self, scoped: "Results", stage_id: str, component: str) -> Any:
        """Full-time GaussSlab with float32 values, or None."""
        key = (stage_id, component)
        entry = self._cache.get(key)
        if entry is None:
            if not self._load_gauss(scoped, stage_id, component):
                return None
            entry = self._cache.get(key)
        if entry is not None:
            self._touch(key)
        return entry.slab if entry is not None else None

    def color_limits(self, stage_id: str, component: str) -> "Optional[tuple[float, float]]":
        """Finite (vmin, vmax) cached for (stage, component), or None."""
        entry = self._cache.get((stage_id, component))
        if entry is None:
            return None
        lo, hi = entry.vmin, entry.vmax
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        if lo == hi:
            hi = lo + 1.0
        return (lo, hi)

    # ------------------------------------------------------------------
    # Loaders - the ONLY place that reads HDF5 through this store
    # ------------------------------------------------------------------
    def _load_nodes(self, scoped: "Results", stage_id: str, component: str) -> bool:
        slab = self._safe_read(lambda: scoped.nodes.get(component=component, time=None))
        if slab is None or np.asarray(slab.values).size == 0:
            return False
        f32 = np.asarray(slab.values, dtype=np.float32)
        new_slab = _replace_values(slab, f32)
        vmin, vmax = _finite_minmax(f32)
        self._store(stage_id, component, new_slab, vmin, vmax, "nodes")
        return True

    def _load_gauss(self, scoped: "Results", stage_id: str, component: str) -> bool:
        slab = self._safe_read(lambda: scoped.elements.gauss.get(component=component, time=None))
        if slab is None or np.asarray(slab.values).size == 0:
            return False
        f32 = np.asarray(slab.values, dtype=np.float32)
        new_slab = _replace_values(slab, f32)
        vmin, vmax = _finite_minmax(f32)
        self._store(stage_id, component, new_slab, vmin, vmax, "gauss")
        return True

    def _store(
        self, stage_id: str, component: str,
        slab: Any, vmin: float, vmax: float, kind: str,
    ) -> None:
        key = (stage_id, component)
        existing = self._cache.get(key)
        if existing is not None:
            self._loaded_bytes -= existing.nbytes
        entry = _Entry(slab, vmin, vmax, kind)
        self._cache[key] = entry
        self._loaded_bytes += entry.nbytes
        self._touch(key)
        # Bound resident memory: drop LRU entries until under the budget,
        # but never the one just stored and never the last entry.
        self._evict_to_budget(protect=key)

    @staticmethod
    def _safe_read(fn: Any) -> Any:
        try:
            return fn()
        except Exception:
            return None