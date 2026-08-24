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

   * G0a = Σ(R1..R4, R6, R7) / (anchor traced-current − pre_build
     traced-current).  R5 (the BuiltModel ``fem`` pin — the legitimate
     floor) is reported separately and NEVER counted in the numerator.
   * G0b = RSS growth / traced growth over pre_build → post_write.

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
import gc
import io
import json
import math
import os
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
TERM_ORDER = ("R7", "R6", "R1", "R3", "R2", "R4")

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


def _match_term(traceback_obj, terms) -> "str | None":
    """First term (in table order) with a frame inside one of its spans."""
    frames = [
        (os.path.normcase(fr.filename), fr.lineno) for fr in traceback_obj
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


class RssSampler:
    """Daemon thread polling process RSS; ``mark(label)`` timestamps phase
    boundaries with a synchronous sample.  Sampling never raises into the
    measurement.  ``available`` False ⇒ this platform cannot be read; the
    tool then prints "RSS: unavailable", never 0.0."""

    def __init__(self, interval_s: float = 0.05):
        self._reader = _make_rss_reader()
        self.available = self._reader is not None
        self.interval_s = interval_s
        self.samples: "list[tuple[float, int, str]]" = []  # (t, rss, mark)
        self._t0 = time.perf_counter()
        self._stop_evt = threading.Event()
        self._thread: "threading.Thread | None" = None

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
        while not self._stop_evt.wait(self.interval_s):
            self._sample("")

    def start(self):
        if not self.available:
            return
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
    snap_dir: str,
) -> dict:
    """Emit with in-flight hooks. Returns {"hooks": [...], "po_calls": n,
    "po_ranks": k, "snap_paths": {label: path}}.

    Hooks (in fired order): pre_build, post_build, ndf_chunks_peak,
    partition_open#first, partition_open#mid, post_emit, post_write.
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
            cur, peak = tracemalloc.get_traced_memory()
            entry["traced_cur"], entry["traced_peak"] = cur, peak
            snap = tracemalloc.take_snapshot()
            per_term = _attribute_snapshot(snap, terms)
            entry["per_term"] = per_term
            entry["term_sum"] = sum(per_term.values())
            p = os.path.join(
                snap_dir, f"{len(hooks):02d}_{label.replace('#', '_')}.tm")
            snap.dump(p)
            snap_paths[label] = p
            del snap
            # Reset AFTER the snapshot work so the instrument's own
            # allocation spike is swallowed here, not billed to the next
            # inter-hook interval.
            tracemalloc.reset_peak()
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

    emitter = TclEmitter()
    if stream:
        emitter.stream_to(deck_path, per_rank=per_rank)

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
        try:
            del emitter.partition_open
        except AttributeError:
            pass
        if no_gc:
            gc.enable()

    return {
        "hooks": hooks,
        "po_calls": po_calls[0],
        "po_ranks": len(seen_ranks),
        "snap_paths": snap_paths,
    }


def _mb(x) -> str:
    return "      -" if x is None else f"{x / 1e6:>9.1f}"


def report_instrumented(args, sz: int, hx: int, nn: int, result: dict,
                        sampler: "RssSampler | None") -> dict:
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
    g0b = None
    per_term_anchor: "dict[str, int] | None" = None
    unattributed = None
    traced_growth = None
    global_peak = None

    if mem:
        pre_cur = pre["traced_cur"]
        anchor = max(hooks, key=lambda h: h["traced_cur"])
        global_peak = max(h["traced_peak"] for h in hooks)

        print(f"  {'hook':<22} {'traced-cur':>10} {'d-pre':>9} "
              f"{'traced-peak':>11} {'RSS':>9} {'G0a':>6}")
        for h in hooks:
            growth = h["traced_cur"] - pre_cur
            if h is pre or growth <= 0:
                g0a_h = "-"
            else:
                g0a_h = f"{h['term_sum'] / growth:.2f}"
            mark = " <-anchor" if h is anchor else ""
            print(f"  {h['label']:<22} {_mb(h['traced_cur'])} "
                  f"{growth / 1e6:>+9.1f} {_mb(h['traced_peak']):>11} "
                  f"{_mb(h['rss'])} {g0a_h:>6}{mark}")

        traced_growth = anchor["traced_cur"] - pre_cur
        per_term_anchor = anchor["per_term"]
        term_sum = anchor["term_sum"]
        g0a = term_sum / traced_growth if traced_growth > 0 else float("nan")
        unattributed = traced_growth - term_sum

        print(f"  anchor hook: {anchor['label']} "
              f"(traced-cur {anchor['traced_cur'] / 1e6:,.1f} MB; "
              f"global traced peak {global_peak / 1e6:,.1f} MB)")
        terms_txt = "  ".join(
            f"{t}={per_term_anchor[t] / 1e6:,.1f}MB" for t in TERM_ORDER
        )
        print(f"  per-term at anchor: {terms_txt}")
        print(f"  R5 floor (pre_build traced-cur, incl. fem pin - NEVER in "
              f"the G0a numerator): {pre_cur / 1e6:,.1f} MB")
        share = (unattributed / traced_growth * 100.0
                 if traced_growth > 0 else float("nan"))
        print(f"  G0a = {g0a:.3f}   "
              f"(sum(terms) {term_sum / 1e6:,.1f} MB / growth "
              f"{traced_growth / 1e6:,.1f} MB; unattributed "
              f"{unattributed / 1e6:,.1f} MB = {share:.0f}%)")
        if g0a > 1.0:
            print("  *** WARNING: G0a > 1 - the R-term spans double-count; "
                  "the first-match order / span windows need revisiting. "
                  "The number above is printed UNCLAMPED. ***")
        if rss_growth is not None and traced_growth > 0:
            g0b = rss_growth / traced_growth
            print(f"  G0b = {g0b:.2f}  (RSS growth {rss_growth / 1e6:,.1f} MB"
                  f" / traced growth {traced_growth / 1e6:,.1f} MB) "
                  f"[tracemalloc-INFLATED — pair with a --rss-only run for "
                  f"the honest G0b]")
        print(f"  partition_open: {result['po_calls']} calls, "
              f"{result['po_ranks']} distinct ranks")

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
        "g0a": g0a,
        "g0b": g0b,
        "anchor_hook": anchor["label"] if anchor is not None else None,
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
    }


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

    rows = []
    records = []
    for sz in sizes:
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
                records.append(
                    report_instrumented(args, sz, hx, nn, result, sampler))
            rows.append((hx, emit_total))

            # --profile runs INSIDE the largest size's iteration (the old
            # tool kept ``last = (ops, sz)`` alive across sizes, pinning
            # the previous FEMData + fan-out cache through the next size's
            # measurement — the cross-size retention this rebuild drops).
            if args.profile and sz == largest:
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
                    if len(sizes) > 1:
                        root, ext = os.path.splitext(tp)
                        tp = f"{root}_sz{sz}{ext or '.csv'}"
                    try:
                        sampler.write_trace(tp)
                        print(f"  rss trace: {tp} "
                              f"({len(sampler.samples)} samples)")
                    except OSError as exc:
                        print(f"  rss trace write failed: {exc!r}")
            shutil.rmtree(tmpdir, ignore_errors=True)

    if len(rows) >= 2 and rows[0][1] and rows[0][0]:
        sh = rows[-1][0] / rows[0][0]
        st = rows[-1][1] / rows[0][1]
        print(f"\nemit linearity: hexes x{sh:.1f} -> emit-time x{st:.1f} "
              f"(exponent ~{math.log(st)/math.log(sh) if sh > 1 else 0:.2f})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"\njson: {args.json} ({len(records)} records)")


if __name__ == "__main__":
    main()
