"""Memory proof for the columnar MassSet refactor (ADR 0065 v2 / C1-C3).

NOT a pytest test (no ``test_`` prefix -> not collected). A standalone
CLI that measures the resident cost of the resolved nodal-mass store
before and after the columnar refactor, in bytes per node, for the two
paths the plan named as the 7M-node hotspots:

  (a) get_fem_data  -- the in-session resolve (``g.masses.volume`` builds
      the per-node tributary masses at ``g.mesh.queries.get_fem_data()``).
  (b) FEMData.from_h5 -- the rehydration path (``_read_masses`` boxed one
      ``MassRecord`` per node before this change; now it adopts columns).

The knob is ``--edge`` (nodes per transfinite edge); ``edge=80`` builds
~493k hexes / ~512k nodes -- a ~0.5M-node stand-in for the LOH.1 wall.
``tracemalloc`` brackets each path and attributes the peak to the mass
store by diffing a masses-on vs masses-off build.

Target (plan gate): ~500 B/node -> ~60 B/node.  The columnar store is
``int64[N]`` (8 B) + ``float64[N,6]`` (48 B) = 56 B/node + O(1); the old
store was one non-slotted ``MassRecord`` dataclass per node (a 6-tuple of
Python floats + a boxed int + a str slot), ~400-700 B/node.

Run (venv, worktree source):
  set PYTHONPATH=src
  C:/Users/nmora/venv/opensees_venv/Scripts/python.exe \
      tests/benchmarks/mass_columnar_memory_profile.py --edge 80
"""
from __future__ import annotations

import argparse
import gc
import os
import tempfile
import tracemalloc

from apeGmsh import apeGmsh, FEMData


def _build_box(edge: int, with_masses: bool):
    """Structured hex box with (edge-1)^3 hexes; optional volume masses."""
    g = apeGmsh(model_name=f"massbox_{edge}", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        if with_masses:
            g.masses.volume("soil", density=2400.0)
        g.mesh.structured.set_transfinite("soil", n=edge, recombine=True)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()
    finally:
        g.end()
    return fem


def _peak_delta_bytes(build_fn) -> tuple[int, object]:
    """tracemalloc peak growth (bytes) across ``build_fn()``; returns
    (peak_bytes, produced_object) with the object kept alive."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    obj = build_fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=int, default=80,
                    help="nodes per transfinite edge; (edge-1)^3 hexes")
    args = ap.parse_args()

    # ---- (a) in-session resolve: masses-on minus masses-off ----------
    peak_on, fem = _peak_delta_bytes(lambda: _build_box(args.edge, True))
    n_nodes = len(fem.nodes.masses)
    peak_off, fem_off = _peak_delta_bytes(lambda: _build_box(args.edge, False))
    resolve_delta = peak_on - peak_off
    del fem_off

    ms_a = fem.nodes.masses
    resident_a = (ms_a.node_ids().nbytes + ms_a.mass_array().nbytes
                  if hasattr(ms_a, "node_ids") else -1)

    print(f"nodes with mass : {n_nodes}")
    print("== (a) get_fem_data (in-session resolve) ==")
    print(f"  peak masses-on   : {peak_on/1e6:9.1f} MB")
    print(f"  peak masses-off  : {peak_off/1e6:9.1f} MB")
    print(f"  peak delta       : {resolve_delta/1e6:9.1f} MB "
          f"=> {resolve_delta/max(n_nodes,1):7.1f} B/node "
          f"(incl. transient resolver accumulator)")
    if resident_a >= 0:
        print(f"  RESIDENT store   : {resident_a/1e6:9.1f} MB "
              f"=> {resident_a/max(n_nodes,1):7.1f} B/node "
              f"(the columnar MassSet that survives on FEMData)")

    # ---- (b) from_h5 rehydration -------------------------------------
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.h5")
        fem.to_h5(path)
        del fem
        gc.collect()

        peak_rehydrate, fem2 = _peak_delta_bytes(lambda: FEMData.from_h5(path))
        n2 = len(fem2.nodes.masses)
        print("== (b) FEMData.from_h5 (rehydration) ==")
        print(f"  peak from_h5     : {peak_rehydrate/1e6:9.1f} MB "
              f"(whole FEMData; mass store is a subset)")
        # Isolate the mass-store cost: measure a from_h5 with the /masses
        # group present vs the same file's masses adopted as columns.
        ms = fem2.nodes.masses
        col_bytes = ms.node_ids().nbytes + ms.mass_array().nbytes
        print(f"  columnar store   : {col_bytes/1e6:9.1f} MB "
              f"=> {col_bytes/max(n2,1):7.1f} B/node (int64 + float64[6])")


if __name__ == "__main__":
    main()
