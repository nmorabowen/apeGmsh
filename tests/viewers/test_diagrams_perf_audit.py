"""Throwaway perf bench — diagrams subsystem audit.

Run with::

    pytest tests/viewers/test_diagrams_perf_audit.py -m bench -s

Measures per-step update time on a non-trivial scene for the diagrams
flagged by the audit:

* VectorGlyphDiagram — rebuilds the entire glyph PolyData each step.
* DeformedShapeDiagram — issues N independent h5py reads per step,
  one per displacement component.
* ContourDiagram — nodes path (cheap baseline).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DeformedShapeDiagram,
    DeformedShapeStyle,
    DiagramSpec,
    SlabSelector,
    VectorGlyphDiagram,
    VectorGlyphStyle,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


@pytest.fixture
def big_solid_results(g, tmp_path: Path):
    """3-D box meshed densely. Targets ~1k nodes / many tetras."""
    g.model.geometry.add_box(0, 0, 0, 5, 5, 5, label="block")
    g.physical.add_volume("block", name="Body")
    g.mesh.sizing.set_global_size(0.45)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_nodes = node_ids.size

    n_steps = 80
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n_steps, n_nodes)) * 0.01
    components = {
        "displacement_x": base.copy(),
        "displacement_y": base.copy() + 0.02,
        "displacement_z": base.copy() + 0.03,
    }

    path = tmp_path / "big_solid.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    print(f"\n[fixture] n_nodes={n_nodes} n_steps={n_steps}")
    return Results.from_native(path, model=_open_model_from_h5(path)), n_nodes


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _bench(label: str, fn, n_iters: int = 50) -> float:
    # Warmup
    for i in range(5):
        fn(i % n_iters)
    start = time.perf_counter()
    for i in range(n_iters):
        fn(i % n_iters)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / n_iters) * 1000.0
    print(f"  {label:42s}  avg={avg_ms:7.2f} ms  (n={n_iters})")
    return avg_ms


@pytest.mark.bench
def test_audit_diagrams_perf(big_solid_results, headless_plotter):
    results, n_nodes = big_solid_results
    scene = build_fem_scene(results.fem)

    print(f"\n=== Diagrams perf audit (n_nodes={n_nodes}) ===")

    # ---------- ContourDiagram (nodes) ---------------------------
    contour = ContourDiagram(
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="displacement_x"),
            style=ContourStyle(),
        ),
        results,
    )
    contour.attach(headless_plotter, results.fem, scene)
    _bench("ContourDiagram(nodes) update_to_step",
           contour.update_to_step)
    contour.detach()

    # ---------- DeformedShapeDiagram -----------------------------
    deformed = DeformedShapeDiagram(
        DiagramSpec(
            kind="deformed_shape",
            selector=SlabSelector(component="displacement_x"),
            style=DeformedShapeStyle(scale=1.0),
        ),
        results,
    )
    t0 = time.perf_counter()
    deformed.attach(headless_plotter, results.fem, scene)
    print(f"  DeformedShape attach                       "
          f"{(time.perf_counter()-t0)*1000:.2f} ms")
    _bench("DeformedShape update_to_step (3 reads/step)",
           deformed.update_to_step)
    deformed.detach()

    # ---------- VectorGlyphDiagram -------------------------------
    glyph = VectorGlyphDiagram(
        DiagramSpec(
            kind="vector_glyph",
            selector=SlabSelector(component="displacement_x"),
            style=VectorGlyphStyle(scale=1.0),
        ),
        results,
    )
    t0 = time.perf_counter()
    glyph.attach(headless_plotter, results.fem, scene)
    print(f"  VectorGlyph attach                         "
          f"{(time.perf_counter()-t0)*1000:.2f} ms")
    _bench("VectorGlyph update_to_step (rebuilds glyph)",
           glyph.update_to_step)
    glyph.detach()

    # No assertion — this is an audit, numbers go to stdout.
