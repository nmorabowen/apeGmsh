"""Emit-throughput / phase-resolved profiling tool (ADR 0065 Tier 2, Cost Center B).

NOT a pytest test (no ``test_`` prefix → not collected). A standalone CLI that
LOCATES the deck-emit wall by (a) phase-resolved wall-clock timers separating
mesh / partition / get_fem_data / build / emit / write, and (b) a cProfile
attribution pass on the emit. The adversarial workflow + an earlier sweep proved
the emit *algorithm* is linear at ~50-90k hex/s (NOT the claimed ~670 hex/s),
so the point of this tool is to find which phase actually carries the wall on a
real model — without tracemalloc, whose per-allocation overhead is the leading
suspect for the original 670 hex/s figure.

Two recipes:
  --recipe box        structured hex box (fast, clean, scalable knob = nodes/edge)
  --recipe planewave  the ADR loh1-mirror: add_plane_wave_box (ASDAbsorbing skin)
                      + per-layer masses.volume + stdBrick per soil PG + staged
                      activate_absorbing — the config that produced 670 hex/s.

Run (venv):
  C:/Users/nmora/venv/opensees_venv/Scripts/python.exe \
      tests/benchmarks/emit_throughput_profile.py \
      --recipe planewave --sizes 30,45 --parts 16 --mass explicit_loop --profile

Set PYTHONPATH=src when running against the worktree source.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import io
import math
import os
import pstats
import tempfile
import time

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Path


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


def emit_phases(ops, no_gc: bool, ph: dict):
    """Replicate ops.tcl() decomposed into build / emit / write, each timed."""
    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    gc.collect()
    if no_gc:
        gc.disable()
    try:
        t = time.perf_counter()
        bm = ops.build()
        ph["build"] += time.perf_counter() - t

        emitter = TclEmitter()
        t = time.perf_counter()
        bm.emit(emitter)
        ph["emit"] += time.perf_counter() - t

        t = time.perf_counter()
        with open(path, "w", encoding="utf-8") as f:
            emitter.write_to(f)
        ph["write"] += time.perf_counter() - t
    finally:
        if no_gc:
            gc.enable()
        try:
            os.remove(path)
        except OSError:
            pass


def _rss_peak_mb() -> float:
    """Whole-process peak working set in MB (0.0 where unavailable).

    Complements tracemalloc (Python allocations only) — the gap between
    the two is the heap-fragmentation / native-allocation multiplier
    from the ADR 0065 ledger.
    """
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

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        # GetCurrentProcess() through ctypes truncates the -1
        # pseudo-handle to 32 bits (ERROR_INVALID_HANDLE on x64) —
        # build the pseudo-handle directly.
        h = ctypes.c_void_p(-1)
        # Win8+: psapi functions live in kernel32 as K32*; older
        # systems keep the psapi.dll export.
        for fn in (
            getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None),
            getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None),
        ):
            if fn is not None and fn(h, ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 1e6
    except Exception:
        pass
    return 0.0


def emit_phases_mem(ops, ph: dict, mem: dict, top: int) -> str:
    """``emit_phases`` + tracemalloc attribution (ADR 0065 v2, M0).

    Traces ONLY build / emit / write — the ledger terms are
    emitter-side; session/mesh allocations are deliberately out of
    scope. Records per phase the ABSOLUTE traced (current, peak) pair
    so the overall peak is ``max(peaks)``, and returns the top-``top``
    allocation sites at the post-emit resident point (line buffer +
    any plan still referenced, before write).

    Wall-clock from a --mem run is meaningless (tracemalloc per-alloc
    overhead) — use a plain run for throughput.
    """
    import tracemalloc

    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    gc.collect()
    tracemalloc.start()
    snap_txt = ""
    try:
        def _phase(name, fn):
            gc.collect()
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            t = time.perf_counter()
            out = fn()
            ph[name] += time.perf_counter() - t
            cur, peak = tracemalloc.get_traced_memory()
            mem[name] = {"before": before, "cur": cur, "peak": peak}
            return out

        bm = _phase("build", lambda: ops.build())
        emitter = TclEmitter()
        _phase("emit", lambda: bm.emit(emitter))
        snap = tracemalloc.take_snapshot()
        snap_txt = "\n".join(
            str(s) for s in snap.statistics("lineno")[:top]
        )
        del snap

        def _write():
            with open(path, "w", encoding="utf-8") as f:
                emitter.write_to(f)

        _phase("write", _write)
        mem["rss_peak_mb"] = _rss_peak_mb()
    finally:
        tracemalloc.stop()
        try:
            os.remove(path)
        except OSError:
            pass
    return snap_txt


def profile_emit(ops, top: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    pr = cProfile.Profile()
    gc.collect()
    pr.enable()
    ops.tcl(path)
    pr.disable()
    os.remove(path)
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
                    help="tracemalloc + RSS-peak attribution on build/emit/"
                         "write (ADR 0065 v2 M0). Distorts wall-clock — "
                         "don't read throughput from a --mem run.")
    ap.add_argument("--mem-top", type=int, default=15,
                    help="--mem: allocation sites to list at the post-emit "
                         "resident point")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    z_layers = [(100.0, int(n)) for n in args.planewave_z.split(",")]
    want_masses = args.mass in ("from_model", "explicit_loop")

    print(f"== emit throughput profile ==  recipe={args.recipe} parts={args.parts} "
          f"staged={args.staged} mass={args.mass} no_gc={args.no_gc}")
    hdr = (f"{'hexes':>10} {'mesh':>7} {'partn':>7} {'getfem':>7} "
           f"{'build':>7} {'emit':>7} {'write':>7} {'EMIT/hexs':>10}")
    print(hdr)

    rows = []
    last = None
    for sz in sizes:
        ph = dict.fromkeys(
            ["geom", "mesh", "partition", "get_fem", "build", "emit", "write"], 0.0)
        if args.recipe == "box":
            fem, soil_pgs, res = build_box(sz, args.parts, want_masses, ph)
        else:
            fem, soil_pgs, res = build_planewave(sz, z_layers, args.parts, want_masses, ph)
        ops = make_ops(fem, soil_pgs, res, args.staged, args.mass)
        mem: dict = {}
        snap_txt = ""
        if args.mem:
            snap_txt = emit_phases_mem(ops, ph, mem, args.mem_top)
        else:
            emit_phases(ops, args.no_gc, ph)
        hx = n_hexes(fem, soil_pgs)
        emit_total = ph["build"] + ph["emit"] + ph["write"]
        rate = hx / emit_total if emit_total else 0.0
        print(f"{hx:>10} {ph['mesh']:>7.2f} {ph['partition']:>7.2f} "
              f"{ph['get_fem']:>7.2f} {ph['build']:>7.2f} {ph['emit']:>7.2f} "
              f"{ph['write']:>7.2f} {rate:>10.0f}")
        if args.mem and hx:
            peak_abs = max(
                mem[n]["peak"] for n in ("build", "emit", "write") if n in mem
            )
            print(f"\n-- memory (hexes={hx}) --  traced peak "
                  f"{peak_abs/1e6:,.0f} MB   RSS peak "
                  f"{mem.get('rss_peak_mb', 0.0):,.0f} MB")
            for name in ("build", "emit", "write"):
                d = mem.get(name)
                if not d:
                    continue
                transient = d["peak"] - d["before"]
                resident = d["cur"] - d["before"]
                print(f"  {name:>5}: resident {resident/1e6:>+9.1f} MB   "
                      f"phase-peak {transient/1e6:>+9.1f} MB   "
                      f"({transient/hx:,.0f} B/hex)")
            for tgt in (6_710_000, 11_000_000):
                print(f"  extrapolated traced peak @ {tgt/1e6:>4.1f}M hexes: "
                      f"~{peak_abs/hx*tgt/1e9:.1f} GB")
            print("  top sites at post-emit resident point:")
            for ln in snap_txt.splitlines():
                print(f"    {ln}")
            print()
        rows.append((hx, emit_total))
        last = (ops, sz)

    if len(rows) >= 2 and rows[0][1] and rows[0][0]:
        sh = rows[-1][0] / rows[0][0]
        st = rows[-1][1] / rows[0][1]
        print(f"\nemit linearity: hexes x{sh:.1f} -> emit-time x{st:.1f} "
              f"(exponent ~{math.log(st)/math.log(sh) if sh > 1 else 0:.2f})")

    if args.profile and last is not None:
        ops, sz = last
        print(f"\n===== cProfile attribution (size={sz}) =====")
        print(profile_emit(ops, args.top))


if __name__ == "__main__":
    main()
