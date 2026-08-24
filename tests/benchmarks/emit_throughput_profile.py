"""Emit-throughput / phase-resolved profiling tool (ADR 0065 Tier 2, Cost
Center B) — rebuilt as the ADR 0100 D0 instrument so it can answer gate G0.

NOT a pytest test (no ``test_`` prefix → not collected). A standalone CLI.

Two jobs:

1. The original ADR 0065 job: locate the deck-emit wall with phase-resolved
   wall-clock timers (mesh / partition / get_fem_data / build / emit / write)
   plus an optional cProfile attribution pass (``--profile``).

2. The ADR 0100 D0 job (``--mem`` / ``--rss-only``): attribute the emit-time
   *resident* memory growth to the seven candidate terms R1..R7 of ADR 0100,
   producing gate G0's numbers:

   * G0a(sampled) = Σ(R1..R4, R6, R7) / (anchor traced-current −
     pre_build traced-current) — the form the ADR's literal wording
     describes.  G0a(conservative) divides by the TRUE counter peak,
     charging every unsampled excursion to unattributed — it is the
     GATE number, chosen deliberately because it cannot flatter.  A
     cell with ANY non-zero shortfall (best sampled state below the
     true counter peak) is DISCARDED — it has no G0a at all: the
     allocation-driven trigger misses the same instant identically on
     every run, so a missed peak yields a stable wrong number that
     looks converged.  The numerator is EXACTLY the ADR's
     authorised set; CACHE (the broker fan-out memo), R8
     (ops_tag_to_fem_eid, absent from the ADR), and the process floor
     are attributed and reported but never counted.
   * G0b(slope) = slope of RSS-peak growth vs hexes (from a paired
     --rss-only run via --pair-rss-json) over slope of traced-peak
     growth vs hexes.  The per-cell RSS/traced ratio is printed but
     demoted — at bench scale it is offset-dominated noise.

   The old instrument could not answer G0: it snapshotted only *after* emit
   returned (when the loop-local structures were already dead), its RSS
   statistic was a Windows-only process-*lifetime* peak dominated by the
   meshing phase (and silently ``0.0`` on Linux), and it retained the
   previous size's model across sizes.  This rebuild takes hooked snapshots
   *inside* the emit (pre_build / ndf_chunks_peak / post_build /
   partition_open#first / #mid / post_emit / post_write), samples RSS from a
   background thread on every platform (or says plainly that it can't), and
   releases each size's model before the next one starts.

   HONESTY RULES (do not "fix" these away):
   * G0a is never clamped, normalised, or capped.  >1 means the term spans
     double-count and the first-match order needs revisiting; 0.1 means the
     R-table does not own the growth.  Either way the real number prints.
   * A missing anchor ABORTS the run — a silently-rotted span reads as
     "unattributed", which corrupts G0a.
   * RSS prints "unavailable" where it cannot be measured, never 0.0.

Two recipes:
  --recipe box        structured hex box (fast, clean, scalable knob = nodes/edge)
  --recipe planewave  the ADR loh1-mirror: add_plane_wave_box (ASDAbsorbing skin)
                      + per-layer masses.volume + stdBrick per soil PG + staged
                      activate_absorbing — the config that produced 670 hex/s.

Run (venv):
  python tests/benchmarks/emit_throughput_profile.py \
      --recipe box --sizes 25,30,35 --parts 16 --mem --stream --staged

Set PYTHONPATH=src when running against the worktree source.
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import gc
import io
import json
import math
import os
import statistics
import pstats
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Path

# Extrapolation targets (hex counts).  51M is the incident deck that
# OOM-killed a 60 GB node at ~61.3 GB RSS (ADR 0100); the rest are the
# campaign's forward-looking sizes.
EXTRAP_TARGETS = (51_000_000, 71_300_000, 100_000_000, 139_000_000)


# ----------------------------------------------------------------------------
# model construction (phase-timed)
# ----------------------------------------------------------------------------
def build_box(n_nodes_edge: int, parts: int, with_masses: bool, ph: dict):
    """Structured hex box: (n_nodes_edge-1)^3 hexes. Returns (fem, soil_pgs)."""
    g = apeGmsh(model_name=f"box_{n_nodes_edge}", verbose=False)
    g.begin()
    try:
        t = time.perf_counter()
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        if with_masses:
            g.masses.volume("soil", density=2400.0)
        g.mesh.structured.set_transfinite("soil", n=n_nodes_edge, recombine=True)
        ph["geom"] += time.perf_counter() - t

        t = time.perf_counter()
        g.mesh.generation.generate(dim=3)
        ph["mesh"] += time.perf_counter() - t

        t = time.perf_counter()
        if parts > 1:
            g.mesh.partitioning.partition(parts)
        ph["partition"] += time.perf_counter() - t

        t = time.perf_counter()
        fem = g.mesh.queries.get_fem_data()
        ph["get_fem"] += time.perf_counter() - t
    finally:
        g.end()
    return fem, ("soil",), None


def build_planewave(nxy: int, nz_layers, parts: int, with_masses: bool, ph: dict):
    """ADR loh1-mirror: add_plane_wave_box + per-layer masses.volume.

    Returns (fem, soil_pgs, res) where res is the AbsorbingSkinResult (carries
    skin PGs for the absorbing_boundary element)."""
    g = apeGmsh(model_name=f"pwb_{nxy}", verbose=False)
    g.begin()
    try:
        t = time.perf_counter()
        z = [(d, n) for (d, n) in nz_layers]
        res = g.parts.add_plane_wave_box(x=(600.0, nxy), y=(600.0, nxy), z=z)
        if with_masses:
            for pg in res.soil_pgs:
                g.masses.volume(pg, density=2400.0)
        ph["geom"] += time.perf_counter() - t

        t = time.perf_counter()
        g.mesh.generation.generate(dim=3)
        ph["mesh"] += time.perf_counter() - t

        t = time.perf_counter()
        if parts > 1:
            g.mesh.partitioning.partition(parts)
        ph["partition"] += time.perf_counter() - t

        t = time.perf_counter()
        fem = g.mesh.queries.get_fem_data()
        ph["get_fem"] += time.perf_counter() - t
    finally:
        g.end()
    return fem, tuple(res.soil_pgs), res


def _full_chain(ops):
    return {
        "test": ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Transformation(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def make_ops(fem, soil_pgs, res, staged: bool, mass_mode: str):
    """mass_mode: none | density | from_model | explicit_loop"""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    rho = 2000.0 if mass_mode == "density" else 0.0
    mat = ElasticIsotropic(E=1.0e7, nu=0.25, rho=rho)
    ops.register(mat)
    for pg in soil_pgs:
        ops.element.stdBrick(pg=pg, material=mat)
    if res is not None:
        ts = ops.register(Path(values=(0.0, 1.0, 0.0), dt=0.1))
        ops.element.absorbing_boundary(
            skin=res, material=mat, base_series=ts, base_dirs=("x",))
    if mass_mode == "from_model":
        ops.mass_from_model()
    elif mass_mode == "explicit_loop":
        for m in fem.nodes.masses:
            ops.mass(nodes=[int(m.node_id)], values=tuple(m.mass))
    if staged:
        with ops.stage(name="gravity") as s:
            s.analysis(**_full_chain(ops))
            s.run(n_increments=2)
        with ops.stage(name="dynamic") as s:
            if res is not None:
                s.activate_absorbing(pg=res.skin_all_pg)
            s.analysis(**_full_chain(ops))
            s.run(n_increments=2)
    return ops


def n_hexes(fem, soil_pgs) -> int:
    total = 0
    for pg in soil_pgs:
        try:
            total += sum(len(g.ids) for g in fem.elements.select(pg=pg).groups())
        except Exception:
            pass
    return total


# ----------------------------------------------------------------------------
# ADR 0100 R-term table — spans located by ANCHOR TEXT, never by hardcoded
# line number.  apesees.py is a hot file edited by concurrent sessions;
# a rotted hardcoded span reads as "unattributed", which corrupts G0a.
# A missing anchor raises and ABORTS — never silently skips a term.
# ----------------------------------------------------------------------------
def _locate(path: str, anchor: str, *, occurrence: int = 1,
            before: int = 1, after: int = 6) -> "tuple[str, int, int]":
    """(path, lo, hi) line span around the Nth occurrence of ``anchor``."""
    seen = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if anchor in line:
                seen += 1
                if seen == occurrence:
                    return (path, max(1, i - before), i + after)
    raise RuntimeError(
        f"G0 R-term anchor NOT FOUND: {anchor!r} (occurrence {occurrence}) "
        f"in {path}. The source has drifted — re-verify the ADR 0100 "
        f"R-term table before trusting any attribution number."
    )


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


# Attribution claim order.  First-match-wins is a real modelling choice:
# the deepest / most-specific terms claim first so that e.g. the R7 chunk
# tuples (allocated under infer_node_ndf, whose CALL SITE is also an R6
# anchor) land on R7, not R6.  The order is printed with every report.
#
# CACHE claims FIRST: the broker fan-out memo (_PG_FANOUT_CACHE arrays,
# concatenated in build._fanout_arrays_from_group_result) is the ADR's
# separate "2nd connectivity copy", NOT an R-term — its allocation frames
# pass through R7's span (the expand_spec_to_elements call at ~build:423),
# so without this pseudo-term R7 silently charges ~72 B/hex of cache to
# the numerator and, at the anchor (where the real class_chunks tuples
# are long dead), R7 collapses to ~pure cache — inverting the D4-vs-D5
# priority a reader would take from the report.
#
# CAVEAT (measured, do not "fix"): once CACHE claims first, the R6/R7
# split is entirely a TERM_ORDER artefact — R7's remaining frames sit
# inside R6's call-site span, so R6 and R7 individually carry no
# independent information; only their SUM is order-invariant.
TERM_ORDER = ("CACHE", "R7", "R6", "R8", "R1", "R3", "R2", "R4")
# The G0a numerator is EXACTLY the set ADR 0100's decision rule was
# written over: R1-R4, R6, R7.  CACHE, R8, and the process floor (R5)
# are attributed and reported but NEVER counted — widening the
# numerator beyond the authorised set would RAISE the gate number
# (measured: R8 alone is ~+0.13 at 13.8k hexes, more than cancelling
# the cache correction), which is precisely the failure mode this
# instrument exists to prevent.  R8's inclusion goes through an ADR
# amendment or not at all.
NUMERATOR_TERMS = ("R7", "R6", "R1", "R3", "R2", "R4")

_TERMS_CACHE: "list[tuple[str, list[tuple[str, int, int]]]] | None" = None


def _term_table() -> "list[tuple[str, list[tuple[str, int, int]]]]":
    """[(term, [(normed_path, lo, hi), ...])] in claim order, cached.

    Line numbers in the comments are as-of-writing documentation only;
    the anchors are what locate the spans.
    """
    global _TERMS_CACHE
    if _TERMS_CACHE is not None:
        return _TERMS_CACHE

    import apeGmsh.opensees._internal.build as _build_mod
    import apeGmsh.opensees.apesees as _apesees_mod
    ap = _apesees_mod.__file__
    bd = _build_mod.__file__

    specs: "list[tuple[str, str, str, int, int, int]]" = [
        # (term, path, anchor, occurrence, before, after)
        # CACHE — the broker fan-out memo's persistent arrays: the
        # np.concatenate outputs (uniform-npe path) and the object-dtype
        # container (mixed-npe path) in _fanout_arrays_from_group_result
        # (~1876 / 1879 / 1886).  Claims before R7 (see TERM_ORDER).
        ("CACHE", bd, "eids = np.concatenate(id_blocks)", 1, 0, 0),
        ("CACHE", bd, "return eids, np.concatenate(conn_blocks, axis=0)",
         1, 0, 0),
        ("CACHE", bd, "conn_obj = np.empty((len(rows),), dtype=object)",
         1, 0, 2),
        # R7 — class_chunks dict + the chunks.append(node_tags) loop
        # (build.py ~420-424).  The node_tags tuples allocate inside the
        # expand_spec_to_elements generator, but the caller frame at the
        # for-loop line is in this span, so nframe>=2 catches them.
        ("R7", bd, "class_chunks: dict[str, list[tuple[int, ...]]] = {}",
         1, 0, 4),
        # R6 — the inferred per-node-ndf dict.  Three faces: the emit-side
        # call site (~1251), the co-resident effective_ndf merge (~1263),
        # and the return-dict comprehension inside build.infer_node_ndf
        # (~504-507).
        ("R6", ap, "inferred_ndf = infer_node_ndf(", 1, 0, 0),
        ("R6", ap, "effective_ndf = {**inferred_ndf", 1, 0, 0),
        ("R6", bd, "for t, f in zip(all_ids.tolist(), floors.tolist())",
         1, 2, 1),
        # R8 — ops_tag_to_fem_eid (~3674): a full per-element reverse
        # tag map built inside _emit_stages_partitioned (its boxed ints
        # come from FemToOpsTagMap.items(), whose caller frame is in
        # this span).  ~128 B/elem ≈ 6.5 GB at incident scale; ABSENT
        # from the ADR's R-table, so it is attributed and REPORTED but
        # NOT in the gate numerator (see NUMERATOR_TERMS).
        ("R8", ap, "ops_tag_to_fem_eid: dict[int, int] = {", 1, 0, 3),
        # R1 — node_idx_lookup and its siblings.  Occurrence order in the
        # file: 1 = staged-flat (~2209), 2 = partitioned base (~2806),
        # 3 = staged partitioned co-resident twin (~3527).  Windows kept
        # tight (after=2) because R3's plan_by_rank starts one line after
        # the base dict — the overlap assertion below enforces it.
        ("R1", ap, "node_idx_lookup = {", 2, 0, 2),
        ("R1", ap, "node_idx_lookup = {", 3, 0, 2),
        ("R1", ap, "node_idx_lookup = {", 1, 0, 2),
        # ADR 0100's table names a 4th sibling near ~1962 (split='parts');
        # its real spelling is ``node_idx = {...}``, not node_idx_lookup —
        # anchored on the actual text.
        ("R1", ap,
         "node_idx = {int(nid): i for i, nid in "
         "enumerate(self.fem.nodes.ids)}", 1, 0, 0),
        # R3 — plan_by_rank (~2809); its bytes are actually allocated in
        # build.py's bucket_pre_allocated_by_rank (~7564-7600) and
        # ElementPlanRows.select_rows (~1810-1830).
        ("R3", ap, "plan_by_rank = {", 1, 0, 3),
        ("R3", bd, "def bucket_pre_allocated_by_rank(", 1, 0, 36),
        ("R3", bd, "def select_rows(", 1, 0, 20),
        # R2 — per-rank owned / primary node-set dicts + fill loops
        # (~3044-3047, ~3055-3059).
        ("R2", ap, "rank_owned_nodes: dict[int, set[int]] = {}", 1, 0, 3),
        ("R2", ap, "rank_primary_nodes: dict[int, set[int]] = {", 1, 0, 4),
        # R4 — model_mass_by_rank dict + fill loop (~3127-3132); only
        # populated under --mass from_model.
        ("R4", ap, 'model_mass_by_rank: "dict[int, list[Any]]" = {}',
         1, 0, 5),
    ]

    by_term: "dict[str, list[tuple[str, int, int]]]" = {
        t: [] for t in TERM_ORDER
    }
    all_spans: "list[tuple[str, int, int, str]]" = []
    for term, path, anchor, occ, before, after in specs:
        p, lo, hi = _locate(path, anchor, occurrence=occ,
                            before=before, after=after)
        np_ = _norm(p)
        by_term[term].append((np_, lo, hi))
        all_spans.append((np_, lo, hi, term))

    # No two term spans may overlap — an overlap double-counts bytes and
    # G0a inflates silently.
    all_spans.sort()
    for (f1, lo1, hi1, t1), (f2, lo2, hi2, t2) in zip(all_spans,
                                                      all_spans[1:]):
        if f1 == f2 and lo2 <= hi1:
            raise RuntimeError(
                f"G0 R-term spans OVERLAP: {t1} [{lo1},{hi1}] and "
                f"{t2} [{lo2},{hi2}] in {f1}. Tighten the windows before "
                f"trusting any attribution number."
            )

    _TERMS_CACHE = [(t, by_term[t]) for t in TERM_ORDER]
    return _TERMS_CACHE


@functools.lru_cache(maxsize=8192)
def _norm_frame(p: str) -> str:
    """Frame filenames get the SAME normalisation as the span table.

    normcase alone is not enough: a runtime sys.path entry that is
    absolute but unnormalised yields co_filenames that never match spans
    built from module __file__ — measured as per-term all-zeros and a
    G0a of 0.000 printed as the gate.  Memoised: one normpath per
    distinct filename, not per frame."""
    return _norm(p)


def _match_term(traceback_obj, terms) -> "str | None":
    """First term (in table order) with a frame inside one of its spans."""
    frames = [
        (_norm_frame(fr.filename), fr.lineno) for fr in traceback_obj
    ]
    for name, spans in terms:
        for fn, ln in frames:
            for sfn, lo, hi in spans:
                if fn == sfn and lo <= ln <= hi:
                    return name
    return None


def _attribute_snapshot(snap, terms) -> "dict[str, int]":
    """Per-term byte totals from an absolute traceback-grouped snapshot.

    The R-term spans all lie in emit-only code paths, so floor (pre-build)
    allocations can never match a term — absolute statistics are safe here;
    only the *unattributed site list* needs the pre_build diff.
    """
    per_term = dict.fromkeys(TERM_ORDER, 0)
    for stat in snap.statistics("traceback"):
        t = _match_term(stat.traceback, terms)
        if t is not None:
            per_term[t] += stat.size
    return per_term


def _short_path(p: str) -> str:
    parts = p.replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def _top_unattributed(pre_path: str, anchor_path: str, terms,
                      n: int) -> "list[str]":
    """Top-n unattributed GROWTH sites: anchor snapshot minus the
    pre_build snapshot (so fem/session floor sites don't drown the list),
    term-matched lines removed.  This list is the most valuable output of
    the whole tool if attribution comes out low."""
    pre = tracemalloc.Snapshot.load(pre_path)
    cur = tracemalloc.Snapshot.load(anchor_path)
    diffs = cur.compare_to(pre, "traceback")
    del pre, cur
    rows = [d for d in diffs if _match_term(d.traceback, terms) is None]
    rows.sort(key=lambda d: d.size_diff, reverse=True)
    out = []
    for d in rows[:n]:
        frames = list(d.traceback)  # oldest → most recent
        tail = frames[-3:]
        loc = " <- ".join(
            f"{_short_path(fr.filename)}:{fr.lineno}" for fr in reversed(tail)
        )
        out.append(
            f"{d.size_diff / 1e6:>+10.2f} MB  n={d.count_diff:>9}  {loc}"
        )
    return out


# ----------------------------------------------------------------------------
# Cross-platform sampled RSS (replaces the old Windows-only, process-
# lifetime ``_rss_peak_mb`` — whose peak was dominated by the meshing
# phase and silently 0.0 on Linux).
# ----------------------------------------------------------------------------
def _make_rss_reader():
    """Return a zero-arg callable → current RSS in bytes, or None where
    unsupported.  stdlib only — must run on rigs without psutil."""
    if sys.platform.startswith("linux"):
        def _read_linux() -> "int | None":
            with open("/proc/self/status", encoding="ascii",
                      errors="replace") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
            return None
        try:
            if _read_linux() is None:
                return None
        except OSError:
            return None
        return _read_linux

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            # GetCurrentProcess() through ctypes truncates the -1
            # pseudo-handle to 32 bits (ERROR_INVALID_HANDLE on x64) —
            # build the pseudo-handle directly.
            handle = ctypes.c_void_p(-1)
            # Win8+: psapi functions live in kernel32 as K32*; older
            # systems keep the psapi.dll export.
            fn = getattr(ctypes.windll.kernel32,
                         "K32GetProcessMemoryInfo", None)
            if fn is None:
                fn = getattr(ctypes.windll.psapi,
                             "GetProcessMemoryInfo", None)
            if fn is None:
                return None

            def _read_windows() -> "int | None":
                pmc = _PMC()
                pmc.cb = ctypes.sizeof(_PMC)
                if fn(handle, ctypes.byref(pmc), pmc.cb):
                    # WorkingSetSize — the CURRENT resident set.  The old
                    # helper read PeakWorkingSetSize, a process-lifetime
                    # statistic the meshing phase dominates.
                    return int(pmc.WorkingSetSize)
                return None

            if _read_windows() is None:
                return None
            return _read_windows
        except Exception:
            return None

    return None


def _linux_vm_hwm() -> "int | None":
    """VmHWM (process-lifetime RSS peak, kB→bytes) — Linux cross-check."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/status", encoding="ascii",
                  errors="replace") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class TracedPeakTracker:
    """Thread-side tracemalloc peak sampler (mem mode).

    Ticked every ``--tm-poll-ms`` from the RSS sampler's daemon thread,
    so it sees states the main thread never announces: an excursion that
    rises and falls between two emitted lines (a dict comprehension over
    all nodes emits nothing) was structurally invisible to the
    append-anchored poll this replaces — the thread tick does not depend
    on the main thread reaching any particular statement.
    ``get_traced_memory()`` / ``take_snapshot()`` hold the GIL, so a
    thread-side snapshot is a consistent view.

    Every tick banks the peak-counter reading into ``counter_max``
    BEFORE the ``reset_peak`` that swallows the tracker's own snapshot
    transient (a tens-of-MB trace-table copy).  The report's
    ``G0a(conservative)`` divides by this banked TRUE peak, charging any
    unsampled remainder to unattributed — it can never flatter the
    hypothesis.

    ``tick`` never raises into the measurement, skips itself while a
    previous tick's snapshot is still in flight, and ``finalize_lock``
    lets the main thread wait out an in-flight snapshot before loading
    the dump.
    """

    def __init__(self, snap_path: str, grow: float, sampler, hooks_ref):
        self.snap_path = snap_path
        self.grow = grow
        self.sampler = sampler
        self.hooks_ref = hooks_ref
        # Seeded with the attach-time floor so the first snapshot fires
        # on real growth, not on the floor itself.
        self.last_snap_cur = float(tracemalloc.get_traced_memory()[0])
        self.counter_max = 0
        self.entry: "dict | None" = None
        self.idx = 0
        self.snaps = 0
        self.ticks = 0
        self._lock = threading.Lock()

    def tick(self):
        if not self._lock.acquire(blocking=False):
            return  # previous tick's snapshot still in flight
        try:
            if not tracemalloc.is_tracing():
                return
            self.ticks += 1
            cur, peak = tracemalloc.get_traced_memory()
            if peak > self.counter_max:
                self.counter_max = peak
            if cur > self.last_snap_cur * self.grow:
                rss = (self.sampler._sample("peak_tracked")
                       if self.sampler is not None else None)
                cur2, peak2 = tracemalloc.get_traced_memory()
                if peak2 > self.counter_max:
                    self.counter_max = peak2
                snap = tracemalloc.take_snapshot()
                snap.dump(self.snap_path)
                del snap
                # Re-bank IMMEDIATELY before the reset: snap.dump's file
                # write releases the GIL, so main-thread growth during it
                # would otherwise be erased by the reset (inflating G0a).
                # The peak READING here is unusable — it includes the
                # tracker's own snapshot transient — so bank the re-read
                # CURRENT instead (the snapshot is freed by now): real
                # main-thread residency shows in cur3, instrument bytes
                # do not.  A main-thread rise-AND-fall inside the dump
                # window is the one shape still lost; it is inseparable
                # from the tracker's own bytes in a single counter.
                cur3 = tracemalloc.get_traced_memory()[0]
                if cur3 > self.counter_max:
                    self.counter_max = cur3
                tracemalloc.reset_peak()
                self.last_snap_cur = cur2
                self.snaps += 1
                self.idx = len(self.hooks_ref)
                self.entry = {
                    "label": "peak_tracked", "rss": rss,
                    "traced_cur": cur2, "traced_peak": peak2,
                    "per_term": None, "term_sum": None,
                }
        except Exception:
            pass  # sampling must never raise into the measurement
        finally:
            self._lock.release()

    def finalize_lock(self):
        return self._lock


class RssSampler:
    """Daemon thread polling process RSS; ``mark(label)`` timestamps phase
    boundaries with a synchronous sample.  Sampling never raises into the
    measurement.  ``available`` False ⇒ this platform cannot be read; the
    tool then prints "RSS: unavailable", never 0.0.

    In --mem mode a :class:`TracedPeakTracker` can be attached; the loop
    then ticks at ``tm_interval_s`` (finer) and thins RSS samples back to
    ``interval_s``."""

    def __init__(self, interval_s: float = 0.05):
        self._reader = _make_rss_reader()
        self.available = self._reader is not None
        self.interval_s = interval_s
        self.samples: "list[tuple[float, int, str]]" = []  # (t, rss, mark)
        self._t0 = time.perf_counter()
        self._stop_evt = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._tm: "TracedPeakTracker | None" = None
        self.tm_interval_s = 0.005

    def attach_tm(self, tracker: "TracedPeakTracker", tm_interval_s: float):
        self.tm_interval_s = tm_interval_s
        self._tm = tracker

    def detach_tm(self):
        self._tm = None

    def _sample(self, mark: str = "") -> "int | None":
        if not self.available:
            return None
        try:
            rss = self._reader()
        except Exception:
            return None
        if rss is None:
            return None
        self.samples.append((time.perf_counter() - self._t0, rss, mark))
        return rss

    def _loop(self):
        next_rss = 0.0
        while True:
            # Tick fine (tm poll) while a tracker is attached, coarse
            # (RSS interval) otherwise; RSS samples are thinned back to
            # ``interval_s`` either way.
            tm = self._tm
            tick = self.tm_interval_s if tm is not None else self.interval_s
            if self._stop_evt.wait(tick):
                return
            if tm is not None:
                tm.tick()
            now = time.perf_counter()
            if now >= next_rss:
                self._sample("")
                next_rss = now + self.interval_s

    def start(self):
        # The thread runs even when RSS is unreadable on this platform —
        # in --mem mode it still ticks the TracedPeakTracker; ``_sample``
        # guards RSS availability itself.
        self._sample("start")
        self._thread = threading.Thread(
            target=self._loop, name="rss-sampler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample("stop")

    def mark(self, label: str) -> "int | None":
        return self._sample(label)

    def _mark_time(self, label: str) -> "float | None":
        for t, _rss, m in self.samples:
            if m == label:
                return t
        return None

    def value_at(self, label: str) -> "int | None":
        for _t, rss, m in self.samples:
            if m == label:
                return rss
        return None

    def peak_between(self, a: str, b: str) -> "int | None":
        ta, tb = self._mark_time(a), self._mark_time(b)
        if ta is None or tb is None:
            return None
        window = [rss for t, rss, _m in self.samples if ta <= t <= tb]
        return max(window) if window else None

    def write_trace(self, path: str):
        """CSV trajectory — the desktop analogue of the cluster memlog
        that produced the ADR 0100 incident record."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("t_s,rss_bytes,mark\n")
            for t, rss, m in self.samples:
                f.write(f"{t:.4f},{rss},{m}\n")


# ----------------------------------------------------------------------------
# plain (untraced) emit — the fast path, unchanged semantics
# ----------------------------------------------------------------------------
def emit_phases(ops, no_gc: bool, ph: dict, deck_path: str,
                stream: bool = False, per_rank: bool = False):
    """Replicate ops.tcl() decomposed into build / emit / write, each timed.

    ``stream=True`` replicates ``ops.tcl(path, stream=True)`` instead
    (ADR 0065 Tier 2 / plan_emit_memory_columnar.md A1–A3): the emit
    phase writes through the live sink and the write phase is the
    ``stream_finish`` promotion — the line buffer never exists.

    ``deck_path`` must live inside a caller-owned temp DIRECTORY — the
    per-rank streaming path writes a sibling ``ranks/`` directory the old
    per-file cleanup leaked; the caller's ``shutil.rmtree`` owns both.
    """
    gc.collect()
    if no_gc:
        gc.disable()
    try:
        t = time.perf_counter()
        bm = ops.build()
        ph["build"] += time.perf_counter() - t

        emitter = TclEmitter()
        if stream:
            emitter.stream_to(deck_path, per_rank=per_rank)
        t = time.perf_counter()
        bm.emit(emitter)
        ph["emit"] += time.perf_counter() - t

        t = time.perf_counter()
        if stream:
            emitter.stream_finish()
        else:
            with open(deck_path, "w", encoding="utf-8") as f:
                emitter.write_to(f)
        ph["write"] += time.perf_counter() - t
    finally:
        if no_gc:
            gc.enable()


# ----------------------------------------------------------------------------
# hooked, instrumented emit (ADR 0100 D0)
# ----------------------------------------------------------------------------
def emit_instrumented(
    ops, ph: dict, deck_path: str, *, stream: bool, per_rank: bool,
    parts: int, sampler: "RssSampler | None", mem: bool, no_gc: bool,
    snap_dir: str, tm_poll_ms: float = 5.0,
    peak_snap_grow: float = 1.05,
) -> dict:
    """Emit with in-flight hooks. Returns {"hooks": [...], "po_calls": n,
    "po_ranks": k, "snap_paths": {label: path}}.

    Hooks (in fired order): pre_build, post_build, ndf_chunks_peak,
    partition_open#first, partition_open#mid, post_emit, post_write —
    plus, in --mem mode, a synthetic ``peak_tracked`` hook (see below).
    ndf_chunks_peak fires DURING emit — infer_node_ndf lives inside
    BuiltModel.emit(), not apeSees.build() (measured: build() is +0.0 MB).

    Every hook records (traced_current, traced_peak, rss); in --mem mode it
    additionally takes a tracemalloc snapshot, attributes it against the
    R-term table, and dumps the raw snapshot to ``snap_dir`` so the anchor's
    unattributed-site list can be diffed against pre_build afterwards.
    Dumping (instead of holding snapshots in memory) keeps the instrument's
    own snapshots out of later hooks' traced-current.

    traced_peak at each hook is the absolute peak since the PREVIOUS hook
    (``reset_peak`` runs after each capture, after the snapshot work, so
    the instrument's own snapshot spike never surfaces in the next hook).

    ``peak_tracked`` — the named hooks alone CANNOT see the true resident
    peak: measured on staged partitioned runs, the global traced peak
    lives in windows no named hook samples (inside the per-rank loop, and
    the staged pass between #mid and post_emit), so a named-hook anchor
    understates the G0a denominator.  A :class:`TracedPeakTracker` ticked
    every ``tm_poll_ms`` from the sampler's daemon thread reads
    traced-current; when it exceeds the last snapshotted value by
    ``peak_snap_grow``x, a snapshot is taken and dumped (one file,
    replaced each time, so only the highest survives).  The final peak
    entry joins ``hooks`` at its chronological position and competes for
    the anchor like any named hook.  Any excursion the tracker still
    misses is banked from the peak COUNTER and charged to unattributed by
    the report's ``G0a(conservative)`` — the gate number can therefore
    never be flattered by under-sampling.
    """
    hooks: "list[dict]" = []
    snap_paths: "dict[str, str]" = {}
    terms = _term_table() if mem else None

    def capture(label: str):
        rss = sampler.mark(label) if sampler is not None else None
        entry: dict = {
            "label": label, "rss": rss,
            "traced_cur": None, "traced_peak": None,
            "per_term": None, "term_sum": None,
        }
        if mem:
            # Serialize with the thread tracker: a tick that lands while
            # THIS capture's take_snapshot is allocating its tens-of-MB
            # trace-table copy would read the spike as emit growth,
            # snapshot the instrument's own snapshot-in-progress, and
            # bank a phantom peak (measured: a 69 MB "peak" whose
            # per-term split was all zeros and whose top site was
            # tracemalloc.py itself).  Holding the tracker's lock makes
            # ticks skip — and bank nothing — until the reset below has
            # swallowed the spike.  No real emit signal is lost: the
            # main thread is in here, not emitting.
            lk = tracker._lock if tracker is not None else None
            if lk is not None:
                lk.acquire()
            try:
                cur, peak = tracemalloc.get_traced_memory()
                entry["traced_cur"], entry["traced_peak"] = cur, peak
                snap = tracemalloc.take_snapshot()
                per_term = _attribute_snapshot(snap, terms)
                entry["per_term"] = per_term
                entry["term_sum"] = sum(
                    per_term[t] for t in NUMERATOR_TERMS)
                p = os.path.join(
                    snap_dir,
                    f"{len(hooks):02d}_{label.replace('#', '_')}.tm")
                snap.dump(p)
                snap_paths[label] = p
                del snap
                # Reset AFTER the snapshot work so the instrument's own
                # allocation spike is swallowed here, not billed to the
                # next inter-hook interval.
                tracemalloc.reset_peak()
            finally:
                if lk is not None:
                    lk.release()
        hooks.append(entry)

    # -- hook 2 mechanism: patch the SOURCE module's element_class_ndf_ok.
    # build.py's infer_node_ndf imports it function-locally at call time
    # (build.py ~407), so patching the source module is picked up; patching
    # build's module globals would NOT be.  The first element_class_ndf_ok
    # call inside infer_node_ndf happens right after class_chunks is
    # complete — R7 at its peak.  NOTE (measured, contra the naive plan):
    # validate_node_ndf_element_compat runs EARLIER in emit() and also
    # calls element_class_ndf_ok via a function-local import, so a bare
    # fire-on-first-call patch fires too early — the wrapper therefore
    # gates on the immediate caller being infer_node_ndf.
    import apeGmsh.opensees._element_capabilities as _ecap
    orig_ndf_ok = _ecap.element_class_ndf_ok
    ndf_fired = [False]

    def _ndf_ok_wrapper(class_name):
        if not ndf_fired[0]:
            if sys._getframe(1).f_code.co_name == "infer_node_ndf":
                ndf_fired[0] = True
                capture("ndf_chunks_peak")
        return orig_ndf_ok(class_name)

    # -- staged-pass DIAGNOSTIC hooks (ADR 0100 D0): the tracker's peak
    # snapshot landing inside the staged pass is TIMING, not mechanism —
    # measured spread 0.12 on G0a across identical runs depending on
    # whether the snapshot happened to fire after the R1 twin was built.
    # ``stages_enter`` (at _emit_stages_partitioned entry) and
    # ``stage_open#first`` (the emitter's first stage_open) were built to
    # bracket the twin, but MEASURED they fire BELOW the tracked peak —
    # they do not close the sampling lottery.  They stay because they
    # make the miss visible in the hook table; the load-bearing fix is
    # the shortfall-discard rule in the report (a cell whose best
    # sampled state sits below the true counter peak has no G0a).
    import apeGmsh.opensees.apesees as _apesees_mod
    _BM = _apesees_mod.BuiltModel
    orig_stages = _BM._emit_stages_partitioned

    def _stages_wrapper(bm_self, *a, **k):
        capture("stages_enter")
        return orig_stages(bm_self, *a, **k)

    emitter = TclEmitter()
    if stream:
        emitter.stream_to(deck_path, per_rank=per_rank)

    orig_so = emitter.stage_open
    so_fired = [False]

    def _so_wrapper(name):
        if not so_fired[0]:
            so_fired[0] = True
            capture("stage_open#first")
        return orig_so(name)

    # -- peak tracking (mem mode): a TracedPeakTracker ticked from the
    # RSS sampler's daemon thread.  The earlier append-anchored poll was
    # structurally blind to excursions that rise and fall between two
    # emitted lines (a dict comprehension over all nodes emits nothing);
    # the thread tick is blind to nothing because it does not depend on
    # the main thread reaching any particular statement.
    peak_snap_path = os.path.join(snap_dir, "peak_tracked.tm")
    tracker: "TracedPeakTracker | None" = None
    if mem and sampler is not None:
        tracker = TracedPeakTracker(
            peak_snap_path, peak_snap_grow, sampler, hooks)

    # -- hooks 4: wrap the emitter INSTANCE's partition_open.  #mid fires
    # on the ⌊parts/2⌋-th DISTINCT rank — the staged path re-opens
    # partitions, so distinct ranks are counted, and the total call count
    # is recorded too.
    orig_po = emitter.partition_open
    seen_ranks: "set[int]" = set()
    po_calls = [0]
    mid_target = parts // 2

    def _po_wrapper(rank: int):
        po_calls[0] += 1
        if rank not in seen_ranks:
            seen_ranks.add(rank)
            if len(seen_ranks) == 1:
                capture("partition_open#first")
            elif len(seen_ranks) == mid_target and mid_target > 1:
                capture("partition_open#mid")
        return orig_po(rank)

    gc.collect()
    if no_gc:
        gc.disable()
    _ecap.element_class_ndf_ok = _ndf_ok_wrapper
    emitter.partition_open = _po_wrapper
    emitter.stage_open = _so_wrapper
    _BM._emit_stages_partitioned = _stages_wrapper
    if tracker is not None:
        sampler.attach_tm(tracker, tm_poll_ms / 1000.0)
    try:
        capture("pre_build")

        t = time.perf_counter()
        bm = ops.build()
        ph["build"] += time.perf_counter() - t
        capture("post_build")

        t = time.perf_counter()
        bm.emit(emitter)
        ph["emit"] += time.perf_counter() - t
        capture("post_emit")

        t = time.perf_counter()
        if stream:
            emitter.stream_finish()
        else:
            with open(deck_path, "w", encoding="utf-8") as f:
                emitter.write_to(f)
        ph["write"] += time.perf_counter() - t
        capture("post_write")
    finally:
        # Restore EVERY patch — a leaked monkeypatch poisons the next
        # size in the same process.
        _ecap.element_class_ndf_ok = orig_ndf_ok
        _BM._emit_stages_partitioned = orig_stages
        if tracker is not None:
            sampler.detach_tm()
        try:
            del emitter.partition_open
        except AttributeError:
            pass
        try:
            del emitter.stage_open
        except AttributeError:
            pass
        if no_gc:
            gc.enable()

    # Finalize the tracked peak: attribute its (last, highest) snapshot
    # and slot the entry into ``hooks`` at its chronological position so
    # it competes for the anchor.  Runs after the named captures, so the
    # load's transient allocations pollute nothing.  The tracker is
    # detached above; taking its lock waits out any in-flight snapshot
    # before we read the dump.
    if tracker is not None and tracker.entry is not None:
        with tracker.finalize_lock():
            entry = tracker.entry
            try:
                snap = tracemalloc.Snapshot.load(peak_snap_path)
                per_term = _attribute_snapshot(snap, terms)
                del snap
                entry["per_term"] = per_term
                entry["term_sum"] = sum(
                    per_term[t] for t in NUMERATOR_TERMS)
                snap_paths["peak_tracked"] = peak_snap_path
                hooks.insert(min(tracker.idx, len(hooks)), entry)
            except Exception as exc:  # diagnostic aid must not kill the run
                print(f"  (peak_tracked attribution failed: {exc!r})")

    return {
        "hooks": hooks,
        "po_calls": po_calls[0],
        "po_ranks": len(seen_ranks),
        "snap_paths": snap_paths,
        "peak_snaps": tracker.snaps if tracker is not None else 0,
        "peak_ticks": tracker.ticks if tracker is not None else 0,
        "peak_counter_max": (tracker.counter_max
                             if tracker is not None else 0),
    }


def _mb(x) -> str:
    return "      -" if x is None else f"{x / 1e6:>9.1f}"


def report_instrumented(args, sz: int, hx: int, nn: int, result: dict,
                        sampler: "RssSampler | None", rep: int = 0) -> dict:
    """Print the G0 report for one measured cell; return the JSON record."""
    hooks = result["hooks"]
    by_label = {h["label"]: h for h in hooks}
    pre = hooks[0]
    assert pre["label"] == "pre_build"
    mem = args.mem

    rss_pre = pre["rss"]
    rss_post = by_label.get("post_write", {}).get("rss")
    rss_peak = (sampler.peak_between("pre_build", "post_write")
                if sampler is not None and sampler.available else None)
    rss_growth = (rss_post - rss_pre
                  if rss_post is not None and rss_pre is not None else None)
    rss_peak_growth = (rss_peak - rss_pre
                       if rss_peak is not None and rss_pre is not None
                       else None)

    mode = "--mem (tracemalloc ON)" if mem else "--rss-only (tracemalloc OFF)"
    print(f"\n-- G0 instrument (hexes={hx:,} nodes={nn:,}) --  {mode}")
    if sampler is not None and not sampler.available:
        print("  RSS: unavailable on this platform")
    if mem:
        print(f"  attribution order (first match wins): "
              f"{', '.join(TERM_ORDER)}")

    anchor = None
    g0a = None
    g0a_sampled = None
    g0a_cons = None
    g0b = None
    per_term_anchor: "dict[str, int] | None" = None
    unattributed = None
    traced_growth = None
    global_peak = None
    named_anchor_label = None
    pt_cur = None
    shortfall = None
    gate_status = None
    cons_growth = None

    if mem:
        pre_cur = pre["traced_cur"]
        anchor = max(hooks, key=lambda h: h["traced_cur"])
        # A per-term of zero for every term at every hook is never a
        # real measurement (path-normalisation drift or span rot) —
        # refusing to print a G0a beats printing 0.000 as the gate.
        if all(not h["term_sum"] for h in hooks):
            raise RuntimeError(
                "G0 attribution returned ZERO bytes for every R-term at "
                "every hook. That is never a real measurement — check "
                "frame-filename normalisation and the span table before "
                "trusting anything this process printed."
            )
        # Named-hook peaks cover only the intervals since their previous
        # reset; the poll banks its counter readings (pre-reset) in
        # peak_counter_max — the true global peak is the max of both.
        # pre_build's own traced_peak is EXCLUDED: it covers the
        # mesh/model-build window (pre-emit transients, measured 2.6 MB)
        # and G0a is about emit growth.
        global_peak = max(
            max(h["traced_peak"] for h in hooks[1:]),
            result.get("peak_counter_max") or 0,
        )

        print(f"  {'hook':<22} {'traced-cur':>10} {'d-pre':>9} "
              f"{'traced-peak':>11} {'RSS':>9} {'G0a(s)':>7}")
        for h in hooks:
            growth = h["traced_cur"] - pre_cur
            if h is pre or growth <= 0:
                g0a_h = "-"
            else:
                g0a_h = f"{h['term_sum'] / growth:.2f}"
            mark = " <-anchor" if h is anchor else ""
            print(f"  {h['label']:<22} {_mb(h['traced_cur'])} "
                  f"{growth / 1e6:>+9.1f} {_mb(h['traced_peak']):>11} "
                  f"{_mb(h['rss'])} {g0a_h:>7}{mark}")
        # Full per-term row at EVERY hook — the anchor-only view once
        # inverted the R7 story (real chunk tuples live at
        # ndf_chunks_peak; by the anchor only the cache remains).
        print(f"  {'per-term (MB)':<22} "
              + " ".join(f"{t:>6}" for t in TERM_ORDER))
        for h in hooks:
            row = h["per_term"] or {}
            print(f"  {h['label']:<22} "
                  + " ".join(f"{row.get(t, 0) / 1e6:>6.1f}"
                             for t in TERM_ORDER))

        traced_growth = anchor["traced_cur"] - pre_cur
        per_term_anchor = anchor["per_term"]
        term_sum = anchor["term_sum"]
        # G0a(sampled): against the best state a snapshot actually saw —
        # this is the form ADR 0100's literal wording ("a snapshot taken
        # at the peak") describes.  G0a(conservative): against the TRUE
        # counter peak — the entire unsampled excursion is charged to
        # unattributed.  Conservative is chosen DELIBERATELY as the gate
        # (the ADR amendment will say so): it is the only form that
        # cannot flatter the hypothesis.  An unqualified "G0a" must
        # never appear anywhere in the output.
        g0a_sampled = (term_sum / traced_growth
                       if traced_growth > 0 else float("nan"))
        cons_growth = global_peak - pre_cur
        g0a_cons = (term_sum / cons_growth
                    if cons_growth > 0 else float("nan"))
        g0a = g0a_cons
        shortfall = global_peak - anchor["traced_cur"]
        # Unattributed = growth minus EVERYTHING claimed (numerator terms
        # AND the non-numerator CACHE) — cache bytes are attributed, just
        # never counted toward G0a.
        unattributed = cons_growth - sum(per_term_anchor.values())

        print(f"  anchor hook: {anchor['label']} "
              f"(traced-cur {anchor['traced_cur'] / 1e6:,.1f} MB; "
              f"global traced peak {global_peak / 1e6:,.1f} MB)")

        # Tracked peak vs the best NAMED hook — the named hooks alone
        # cannot see a peak that lives between them (measured: the
        # staged pass between partition_open#mid and post_emit).
        named = [h for h in hooks if h["label"] != "peak_tracked"]
        best_named = max(named, key=lambda h: h["traced_cur"])
        named_anchor_label = best_named["label"]
        pt = by_label.get("peak_tracked")
        pt_cur = pt["traced_cur"] if pt is not None else None
        if pt is not None:
            print(f"  named-hook anchor: {best_named['label']} "
                  f"({best_named['traced_cur'] / 1e6:,.1f} MB)   "
                  f"tracked peak: {pt['traced_cur'] / 1e6:,.1f} MB "
                  f"({result.get('peak_snaps', 0)} peak snapshots)")
            if pt["traced_cur"] > best_named["traced_cur"] * 1.05:
                gap = (pt["traced_cur"] / best_named["traced_cur"] - 1.0) * 100
                print(f"  NOTE: the tracked peak exceeds every named hook "
                      f"by {gap:.0f}% - the resident peak lives BETWEEN "
                      f"the named hooks (staged pass); G0a above is "
                      f"computed there, not at a named hook.")
            # (under-sampling is no longer a footnote here — a >5%
            # shortfall REFUSES the gate below.)
        else:
            print("  (no peak_tracked snapshot fired - emit too small for "
                  "the poll cadence; anchor is named-hook only)")
        terms_txt = "  ".join(
            f"{t}={per_term_anchor[t] / 1e6:,.1f}MB" for t in TERM_ORDER
        )
        print(f"  per-term at anchor: {terms_txt}")
        print(f"  CACHE at anchor (broker fan-out memo, the ADR's separate "
              f"'2nd connectivity copy' - reported, never in the "
              f"numerator): {per_term_anchor.get('CACHE', 0) / 1e6:,.1f} MB")
        print(f"  R8 at anchor (ops_tag_to_fem_eid reverse tag map, "
              f"~128 B/elem at incident scale - measured and reported, "
              f"but NOT in the G0a numerator: ADR 0100 does not authorise "
              f"it; queued for amendment): "
              f"{per_term_anchor.get('R8', 0) / 1e6:,.1f} MB")
        print(f"  process floor (pre_build traced-cur: imports + session + "
              f"fem pin; NOT R5 alone, never in the numerator): "
              f"{pre_cur / 1e6:,.1f} MB")
        cons_share = (unattributed / cons_growth * 100.0
                      if cons_growth > 0 else float("nan"))
        # GATE VALIDITY — the load-bearing rule (ADR 0100 D0): the
        # sampler trigger is allocation-driven, so a cell that misses
        # the peak misses it IDENTICALLY on every run — a stable wrong
        # number that looks converged (measured: reproducible to ±0.001
        # across repeats while sitting 0.031 from the <0.40
        # abandon-the-program threshold).  The named twin hooks do NOT
        # close this (measured: they fire below the tracked peak — the
        # twin dict is built after both fire); they are DIAGNOSTICS that
        # make the miss visible, not the fix.  ANY non-zero shortfall
        # means the cell DID NOT MEASURE THE PEAK and is DISCARDED: no
        # gate number exists for it, in the report or the JSON.
        # Repeats are no defence — a missed peak repeats identically.
        gate_ok = shortfall == 0
        gate_status = "ok" if gate_ok else "discarded_shortfall"
        if gate_ok:
            print(f"  G0a(sampled)      = {g0a_sampled:.3f}  "
                  f"(sum(R-terms) {term_sum / 1e6:,.1f} MB / sampled "
                  f"growth {traced_growth / 1e6:,.1f} MB)")
            print(f"  G0a(conservative) = {g0a_cons:.3f}  "
                  f"(sum(R-terms) {term_sum / 1e6:,.1f} MB / true-peak "
                  f"growth {cons_growth / 1e6:,.1f} MB; unattributed "
                  f"{unattributed / 1e6:,.1f} MB = {cons_share:.0f}%)"
                  f"  <- GATE NUMBER")
            if abs(g0a_sampled - g0a_cons) <= 0.05:
                print("  G0a(sampled) and G0a(conservative) agree within "
                      "0.05. (This checks sampled-vs-conservative "
                      "agreement ONLY - not a global soundness "
                      "certificate.)")
        else:
            g0a = None
            print(f"  *** did not measure the peak - cell DISCARDED "
                  f"(shortfall {shortfall:,} B = "
                  f"{shortfall / 1e6:,.2f} MB between the true counter "
                  f"peak and the best sampled state). A discarded cell "
                  f"has no G0a - the ratio fields stay in the JSON as "
                  f"diagnostics only, g0a is null. ***")
        if g0a_sampled > 1.0:
            print("  *** WARNING: G0a(sampled) > 1 - the R-term spans "
                  "double-count; "
                  "the first-match order / span windows need revisiting. "
                  "The number above is printed UNCLAMPED. ***")
        # G0b per-cell — AUDITED: the old form divided end-to-end RSS
        # growth by the ANCHOR traced growth (mismatched instants; it
        # printed a non-physical 0.68).  Both ends now sit at
        # pre_build→post_write.  Still demoted: at bench scale the ratio
        # is dominated by size-independent RSS offsets (interpreter,
        # numpy, allocator arenas), not by the ADR's x~1.9 amplification
        # SLOPE — the slope-based G0b printed after the size sweep is
        # the number the ADR's "G0b ~= 2" branch rule means.
        traced_growth_end = (
            by_label["post_write"]["traced_cur"] - pre_cur
            if "post_write" in by_label else None)
        if (rss_growth is not None and traced_growth_end is not None
                and traced_growth_end > 0):
            g0b = rss_growth / traced_growth_end
            print(f"  G0b(per-cell) = {g0b:.2f}  (end-to-end RSS growth "
                  f"{rss_growth / 1e6:,.1f} MB / end-to-end traced growth "
                  f"{traced_growth_end / 1e6:,.1f} MB) [OFFSET-DOMINATED "
                  f"at bench scale and tracemalloc-inflated - use the "
                  f"slope-based G0b printed after the size sweep]")
        print(f"  partition_open: {result['po_calls']} calls, "
              f"{result['po_ranks']} distinct ranks")
        print("  standing limitation: a main-thread allocation that rises "
              "AND falls inside a tracker dump window cannot be witnessed "
              "- instrument and main-thread bytes share one counter, so "
              "such a transient is absent from BOTH G0a forms.")

        # Top unattributed growth sites (anchor vs pre_build) — the most
        # valuable output of the whole tool if attribution comes out low.
        try:
            sites = _top_unattributed(
                result["snap_paths"]["pre_build"],
                result["snap_paths"][anchor["label"]],
                _term_table(), args.mem_top,
            )
            print(f"  top {args.mem_top} unattributed growth sites "
                  f"(anchor minus pre_build):")
            for ln in sites:
                print(f"    {ln}")
        except Exception as exc:  # diagnostic aid must not kill the run
            print(f"  (unattributed-site diff failed: {exc!r})")
    else:
        # --rss-only: phase-resolved RSS deltas, uninflated.
        print(f"  {'hook':<22} {'RSS':>9} {'d-pre':>9}")
        for h in hooks:
            d = (h["rss"] - rss_pre
                 if h["rss"] is not None and rss_pre is not None else None)
            d_txt = f"{d / 1e6:>+9.1f}" if d is not None else "        -"
            print(f"  {h['label']:<22} {_mb(h['rss'])} {d_txt}")
        if rss_peak_growth is not None:
            print(f"  RSS peak (pre_build->post_write): "
                  f"{rss_peak / 1e6:,.1f} MB  (+{rss_peak_growth / 1e6:,.1f} "
                  f"MB over pre_build; {rss_peak_growth / max(hx, 1):,.0f} "
                  f"B/hex)")
        anchor = (max((h for h in hooks if h["rss"] is not None),
                      key=lambda h: h["rss"], default=None))
        if anchor is not None:
            print(f"  max-RSS hook: {anchor['label']}")
        print(f"  partition_open: {result['po_calls']} calls, "
              f"{result['po_ranks']} distinct ranks")
        print("  G0a / G0b: n/a in --rss-only (tracemalloc off; pair with "
              "a --mem run)")

    hwm = _linux_vm_hwm()
    if hwm is not None:
        print(f"  VmHWM cross-check (process-lifetime peak): "
              f"{hwm / 1e6:,.1f} MB")

    # Extrapolations — both traced-implied and RSS-implied at each target.
    b_hex_traced = ((global_peak - pre["traced_cur"]) / hx
                    if mem and hx else None)
    b_hex_rss = (rss_peak_growth / hx
                 if rss_peak_growth is not None and hx else None)
    if b_hex_rss is not None and hx < 500_000:
        print("  NOTE: RSS-implied extrapolation from a cell below 0.5M "
              "hexes is UNRELIABLE - size-independent RSS offsets "
              "(interpreter, numpy, allocator arenas) dominate the "
              "per-hex figure; treat as an upper bound at best.")
    extrap = {}
    for tgt in EXTRAP_TARGETS:
        tr = b_hex_traced * tgt / 1e9 if b_hex_traced is not None else None
        rs = b_hex_rss * tgt / 1e9 if b_hex_rss is not None else None
        extrap[str(tgt)] = {"traced_gb": tr, "rss_gb": rs}
        tr_txt = f"{tr:.1f}" if tr is not None else "-"
        rs_txt = f"{rs:.1f}" if rs is not None else "-"
        print(f"  extrapolated @ {tgt / 1e6:>5.1f}M hexes: "
              f"traced ~{tr_txt} GB   rss ~{rs_txt} GB")

    return {
        "size": sz,
        "recipe": args.recipe,
        "parts": args.parts,
        "staged": args.staged,
        "mass": args.mass,
        "stream": args.stream,
        "per_rank": args.per_rank,
        "hexes": hx,
        "n_nodes": nn,
        "traced_peak_b": global_peak,
        "traced_growth_b": traced_growth,
        "floor_traced_b": pre["traced_cur"],
        "rss_growth_b": rss_growth,
        "rss_peak_growth_b": rss_peak_growth,
        "b_per_hex_traced": b_hex_traced,
        "b_per_hex_rss": b_hex_rss,
        # "g0a" is the GATE number = g0a_conservative (Σ R-terms over the
        # true counter peak), or null when the cell was DISCARDED
        # (gate_status == "discarded_shortfall": the tracker did not
        # sample the peak; a discarded cell has no G0a).  The
        # g0a_sampled / g0a_conservative fields remain as diagnostics —
        # their names are unambiguous; only "g0a" is ever a gate number.
        "g0a": g0a,
        "g0a_sampled": g0a_sampled,
        "g0a_conservative": g0a_cons,
        "gate_status": gate_status,
        "discarded_reason": (None if gate_status in (None, "ok")
                             else "peak_shortfall"),
        "repeat": rep,
        "cons_growth_b": cons_growth,
        "cache_bytes": (per_term_anchor or {}).get("CACHE"),
        "g0b_percell": g0b,
        "anchor_hook": anchor["label"] if anchor is not None else None,
        "named_anchor_hook": named_anchor_label,
        "peak_tracked_cur_b": pt_cur,
        "peak_counter_b": global_peak,
        "peak_shortfall_b": shortfall,
        "peak_snaps": result.get("peak_snaps", 0),
        "per_term_bytes": per_term_anchor,
        "unattributed_bytes": unattributed,
        "partition_open_calls": result["po_calls"],
        "partition_open_ranks": result["po_ranks"],
        "extrapolations": extrap,
        "hooks": {
            h["label"]: {
                "traced_cur": h["traced_cur"],
                "traced_peak": h["traced_peak"],
                "rss": h["rss"],
            } for h in hooks
        },
        "per_term_by_hook": (
            {h["label"]: h["per_term"] for h in hooks} if mem else None
        ),
    }


def _fit_slope(points) -> "tuple[float, float]":
    """Least-squares ``y = a*x + b`` over (x, y); returns (a, R^2).

    The intercept b absorbs the size-independent offsets that make the
    per-cell G0b ratio meaningless at bench scale — the slope ratio is
    the amplification factor the ADR's "G0b ~= 2" branch rule means."""
    import numpy as np
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), r2


def profile_emit(ops, top: int, deck_path: str) -> str:
    pr = cProfile.Profile()
    gc.collect()
    pr.enable()
    ops.tcl(deck_path)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(top)
    return s.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="box", choices=["box", "planewave"])
    ap.add_argument("--sizes", default="30,45",
                    help="box: nodes/edge (hexes=(n-1)^3); planewave: nxy per side")
    ap.add_argument("--planewave-z", default="3,5",
                    help="planewave: comma list of per-layer z element counts")
    ap.add_argument("--parts", type=int, default=16)
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--mass", default="none",
                    choices=["none", "density", "from_model", "explicit_loop"])
    ap.add_argument("--no-gc", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile attribution on the largest size")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--mem", action="store_true",
                    help="ADR 0100 D0: hooked tracemalloc attribution of the "
                         "R1..R7 terms + G0a/G0b. Distorts wall-clock — "
                         "don't read throughput from a --mem run.")
    ap.add_argument("--mem-top", type=int, default=15,
                    help="--mem: unattributed growth sites to list at the "
                         "anchor hook")
    ap.add_argument("--tm-frames", type=int, default=12,
                    help="--mem: tracemalloc nframe. 1 cannot attribute "
                         "R3/R6/R7 (their bytes allocate below the owning "
                         "site); default 12.")
    ap.add_argument("--tm-poll-ms", type=float, default=5.0,
                    help="--mem: sampler-thread tick for the peak_tracked "
                         "tracker (default 5 ms). Thread-side, so it sees "
                         "excursions that rise and fall between two "
                         "emitted lines - the append-anchored poll it "
                         "replaces was structurally blind to those.")
    ap.add_argument("--peak-snap-grow", type=float, default=1.05,
                    help="--mem: re-snapshot the tracked peak when "
                         "traced-current exceeds the last snapshot by this "
                         "factor (default 1.05)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="repeat each size N times; the per-size summary "
                         "reports median/min/max G0a(conservative) over "
                         "gate-ok repeats (default 3 under --mem, else 1). "
                         "Repeats defend against ORDINARY noise only - the "
                         "allocation-driven sampler trigger locks onto the "
                         "same instant every run, so a missed peak is "
                         "missed identically; the shortfall gate-refusal "
                         "is the defence against that, not repeats.")
    ap.add_argument("--rss-only", action="store_true",
                    help="sampler on, tracemalloc OFF: phase-resolved RSS "
                         "deltas + B/hex-RSS, uninflated. Pair with a --mem "
                         "run for an honest G0b.")
    ap.add_argument("--rss-interval-ms", type=int, default=50,
                    help="RSS sampler poll interval (default 50)")
    ap.add_argument("--rss-trace", default=None, metavar="PATH",
                    help="dump the RSS trajectory as CSV (t_s,rss_bytes,mark)"
                         "; with multiple sizes, _sz<N> is appended per size")
    ap.add_argument("--per-rank", action="store_true",
                    help="stream_to(path, per_rank=True); requires --stream "
                         "and parts > 1")
    ap.add_argument("--stream", action="store_true",
                    help="emit through the ADR 0065 Tier 2 write-through "
                         "sink (ops.tcl(stream=True) equivalent, "
                         "plan A1–A3) instead of accumulating the line "
                         "buffer")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="write one flat record per measured cell (the ADR "
                         "0100 campaign aggregator's input)")
    ap.add_argument("--pair-rss-json", default=None, metavar="PATH",
                    help="--mem: JSON from a paired --rss-only run of the "
                         "SAME cells; its uninflated RSS-peak growths give "
                         "the RSS slope for G0b(slope). Without it the "
                         "slope G0b is declared not computable rather than "
                         "derived from tracemalloc-inflated RSS.")
    args = ap.parse_args()

    if args.mem and args.rss_only:
        ap.error("--mem and --rss-only are mutually exclusive: --rss-only "
                 "exists precisely to measure RSS without tracemalloc's "
                 "bookkeeping inflating it.")
    if args.per_rank and (not args.stream or args.parts <= 1):
        ap.error("--per-rank requires --stream and --parts > 1 (per-rank "
                 "fragments only exist on the streamed partitioned path).")

    sizes = [int(x) for x in args.sizes.split(",")]
    z_layers = [(100.0, int(n)) for n in args.planewave_z.split(",")]
    want_masses = args.mass in ("from_model", "explicit_loop")
    instrumented = args.mem or args.rss_only
    largest = max(sizes)

    print(f"== emit throughput profile ==  recipe={args.recipe} parts={args.parts} "
          f"staged={args.staged} mass={args.mass} no_gc={args.no_gc} "
          f"stream={args.stream} per_rank={args.per_rank}")
    hdr = (f"{'hexes':>10} {'mesh':>7} {'partn':>7} {'getfem':>7} "
           f"{'build':>7} {'emit':>7} {'write':>7} {'EMIT/hexs':>10}")
    print(hdr)
    if args.mem:
        print("(wall-clock below is tracemalloc-inflated - throughput from "
              "a plain run only)")

    repeats = (args.repeats if args.repeats is not None
               else (3 if args.mem else 1))
    rows = []
    records = []
    for sz in sizes:
      size_recs: "list[dict]" = []
      for rep in range(repeats):
        if repeats > 1:
            print(f"\n--- size {sz}: repeat {rep + 1}/{repeats} ---")
        ph = dict.fromkeys(
            ["geom", "mesh", "partition", "get_fem", "build", "emit", "write"], 0.0)
        tmpdir = tempfile.mkdtemp(prefix="emit_prof_")
        deck = os.path.join(tmpdir, "deck.tcl")
        snap_dir = os.path.join(tmpdir, "tmsnap")
        sampler: "RssSampler | None" = None
        tm_started = False
        try:
            if instrumented:
                os.makedirs(snap_dir, exist_ok=True)
                sampler = RssSampler(args.rss_interval_ms / 1000.0)
                sampler.start()
            if args.mem:
                # Started BEFORE the model build so pre_build's
                # traced-current carries the floor (session + FEMData,
                # i.e. R5) — G0a measures growth ABOVE that floor.
                tracemalloc.start(args.tm_frames)
                tm_started = True

            if args.recipe == "box":
                fem, soil_pgs, res = build_box(
                    sz, args.parts, want_masses, ph)
            else:
                fem, soil_pgs, res = build_planewave(
                    sz, z_layers, args.parts, want_masses, ph)
            ops = make_ops(fem, soil_pgs, res, args.staged, args.mass)

            if instrumented:
                result = emit_instrumented(
                    ops, ph, deck, stream=args.stream,
                    per_rank=args.per_rank, parts=args.parts,
                    sampler=sampler, mem=args.mem, no_gc=args.no_gc,
                    snap_dir=snap_dir,
                    tm_poll_ms=args.tm_poll_ms,
                    peak_snap_grow=args.peak_snap_grow,
                )
            else:
                emit_phases(ops, args.no_gc, ph, deck,
                            stream=args.stream, per_rank=args.per_rank)

            # Post-measurement: n_hexes warms the broker fan-out cache, so
            # it runs AFTER the hooks (running it before would change the
            # emit's own allocation profile vs production).
            hx = n_hexes(fem, soil_pgs)
            try:
                nn = int(len(fem.nodes.ids))
            except Exception:
                nn = 0
            emit_total = ph["build"] + ph["emit"] + ph["write"]
            rate = hx / emit_total if emit_total else 0.0
            print(f"{hx:>10} {ph['mesh']:>7.2f} {ph['partition']:>7.2f} "
                  f"{ph['get_fem']:>7.2f} {ph['build']:>7.2f} {ph['emit']:>7.2f} "
                  f"{ph['write']:>7.2f} {rate:>10.0f}")

            if instrumented:
                rec = report_instrumented(
                    args, sz, hx, nn, result, sampler, rep)
                records.append(rec)
                size_recs.append(rec)
            if rep == repeats - 1:
                rows.append((hx, emit_total))

            # --profile runs INSIDE the largest size's iteration (the old
            # tool kept ``last = (ops, sz)`` alive across sizes, pinning
            # the previous FEMData + fan-out cache through the next size's
            # measurement — the cross-size retention this rebuild drops).
            if args.profile and sz == largest and rep == repeats - 1:
                if tm_started:
                    tracemalloc.stop()
                    tm_started = False
                    print("(--profile: tracemalloc stopped before profiling;"
                          " post-release traced-current unavailable for "
                          "this size)")
                print(f"\n===== cProfile attribution (size={sz}) =====")
                print(profile_emit(
                    ops, args.top, os.path.join(tmpdir, "profile_deck.tcl")))

            # Release THIS size's model before the next size starts, and
            # print the residue so a future retention regression is
            # visible in the output rather than silent.
            del ops, fem, res, soil_pgs
            if instrumented:
                del result
            gc.collect()
            if tm_started:
                residue = tracemalloc.get_traced_memory()[0]
                print(f"  traced-current after release: "
                      f"{residue / 1e6:,.1f} MB")
            elif sampler is not None and sampler.available:
                r = sampler.mark("post_release")
                if r is not None:
                    print(f"  RSS after release: {r / 1e6:,.1f} MB")
        finally:
            if tm_started:
                tracemalloc.stop()
            if sampler is not None:
                sampler.stop()
                if args.rss_trace and sampler.available:
                    tp = args.rss_trace
                    if len(sizes) > 1 or repeats > 1:
                        root, ext = os.path.splitext(tp)
                        tp = f"{root}_sz{sz}_r{rep}{ext or '.csv'}"
                    try:
                        sampler.write_trace(tp)
                        print(f"  rss trace: {tp} "
                              f"({len(sampler.samples)} samples)")
                    except OSError as exc:
                        print(f"  rss trace write failed: {exc!r}")
            shutil.rmtree(tmpdir, ignore_errors=True)

      # -- per-size summary over repeats.  NOTE: median-of-repeats
      # defends against ORDINARY noise only — the allocation-driven
      # trigger locks onto the same instant every run, so a missed peak
      # is missed identically; the shortfall gate-refusal is the
      # defence against that, not repeats.
      if args.mem and repeats > 1 and size_recs:
        statuses = [r.get("gate_status") for r in size_recs]
        ok_vals = [r["g0a_conservative"] for r in size_recs
                   if r.get("gate_status") == "ok"
                   and r.get("g0a_conservative") is not None]
        print(f"\n== size {sz} over {repeats} repeats ==")
        print(f"  gate status per repeat: {statuses}")
        if ok_vals:
            print(f"  G0a(conservative) over gate-ok repeats: median "
                  f"{statistics.median(ok_vals):.3f} "
                  f"(min {min(ok_vals):.3f}, max {max(ok_vals):.3f}, "
                  f"n={len(ok_vals)}/{repeats})")
        else:
            print("  G0a(conservative): NO gate-ok repeat - this size did "
                  "not measure a gate number.")
        r1_twin = [
            round((((r.get("per_term_by_hook") or {})
                    .get("stage_open#first") or {}).get("R1", 0)) / 1e6, 2)
            for r in size_recs
        ]
        r1_mid = [
            round((((r.get("per_term_by_hook") or {})
                    .get("partition_open#mid") or {}).get("R1", 0)) / 1e6, 2)
            for r in size_recs
        ]
        print(f"  R1 (MB) at stage_open#first per repeat: {r1_twin}  vs at "
              f"partition_open#mid: {r1_mid}")
        print("  (median defends against ordinary noise ONLY - a missed "
              "peak repeats identically; see the gate-refusal rule.)")

    if len(rows) >= 2 and rows[0][1] and rows[0][0]:
        sh = rows[-1][0] / rows[0][0]
        st = rows[-1][1] / rows[0][1]
        print(f"\nemit linearity: hexes x{sh:.1f} -> emit-time x{st:.1f} "
              f"(exponent ~{math.log(st)/math.log(sh) if sh > 1 else 0:.2f})")

    # -- shortfall-zero census: how many cells actually measured the
    # peak.  The campaign's gate needs >=3 sizes x >=2 np of SURVIVING
    # cells — if the surviving band is thinner than that, the driver
    # decides whether to spend the plan's one permitted library-side
    # exception (a mid-emit snapshot hook); this tool only reports.
    if args.mem and records:
        ok_n = sum(1 for r in records if r.get("gate_status") == "ok")
        print(f"\nshortfall-zero census: {ok_n}/{len(records)} cells "
              f"measured the peak (only survivors carry gate numbers)")
        by_sz_c: "dict[int, list]" = {}
        for r in records:
            by_sz_c.setdefault(r["size"], []).append(r)
        for s in sorted(by_sz_c):
            rs = by_sz_c[s]
            n_ok = sum(1 for x in rs if x.get("gate_status") == "ok")
            sf = [x.get("peak_shortfall_b") for x in rs]
            print(f"  size {s} (hexes {rs[0]['hexes']:,}): {n_ok}/{len(rs)} "
                  f"survived; shortfalls (B): {sf}")

    # -- G0b(slope): slope of RSS-peak growth vs hexes over slope of
    # traced-peak growth vs hexes.  The RSS side must come from a paired
    # --rss-only run (--pair-rss-json) — this process's own RSS is
    # tracemalloc-inflated, and the per-cell ratio is offset-dominated
    # (measured 0.68–87 across cells: noise).  Without the pair the tool
    # says "not computable" rather than printing an indefensible number.
    g0b_slope = None
    g0b_r2_rss = None
    g0b_r2_traced = None
    if args.mem:
        by_sz: "dict[int, list[dict]]" = {}
        for r in records:
            by_sz.setdefault(r["size"], []).append(r)
        pts_traced = []
        for s in sorted(by_sz):
            vals = [x["cons_growth_b"] for x in by_sz[s]
                    if x.get("cons_growth_b") is not None]
            if vals:
                pts_traced.append(
                    (by_sz[s][0]["hexes"], statistics.median(vals)))
        pts_rss = []
        if args.pair_rss_json:
            try:
                with open(args.pair_rss_json, encoding="utf-8") as f:
                    pair = json.load(f)
                pair_by_sz: "dict[int, list]" = {}
                for r in pair:
                    if r.get("rss_peak_growth_b") is not None:
                        pair_by_sz.setdefault(r["hexes"], []).append(
                            r["rss_peak_growth_b"])
                pts_rss = [(h, statistics.median(v))
                           for h, v in sorted(pair_by_sz.items())]
            except (OSError, ValueError, KeyError) as exc:
                print(f"\nG0b(slope): pair json unreadable ({exc!r})")
        if len(pts_traced) >= 3 and len(pts_rss) >= 3:
            a_tr, g0b_r2_traced = _fit_slope(pts_traced)
            a_rss, g0b_r2_rss = _fit_slope(pts_rss)
            g0b_slope = a_rss / a_tr if a_tr > 0 else None
            def fmt(pts):
                return " ".join(
                    f"({h},{v / 1e6:.1f}MB)" for h, v in pts)

            val = "n/a" if g0b_slope is None else f"{g0b_slope:.2f}"
            print(f"\nG0b(slope) = {val}  "
                  f"(uninflated RSS slope from paired --rss-only run / "
                  f"traced slope from this run)")
            print(f"  fit: R2 rss={g0b_r2_rss:.3f} over {fmt(pts_rss)}")
            print(f"  fit: R2 traced={g0b_r2_traced:.3f} over "
                  f"{fmt(pts_traced)}")
        else:
            print(f"\nG0b(slope): NOT COMPUTABLE in this invocation - "
                  f"needs >=3 sizes on both series (have traced="
                  f"{len(pts_traced)}, rss={len(pts_rss)}; the RSS series "
                  f"comes from --pair-rss-json, a paired --rss-only run). "
                  f"The per-cell G0b is offset-dominated - do not gate "
                  f"on it.")
        for r in records:
            r["g0b_slope"] = g0b_slope
            r["g0b_slope_r2_rss"] = g0b_r2_rss
            r["g0b_slope_r2_traced"] = g0b_r2_traced

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"\njson: {args.json} ({len(records)} records)")


if __name__ == "__main__":
    main()
