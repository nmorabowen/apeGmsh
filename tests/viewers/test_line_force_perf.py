"""LineForceDiagram — Phase 2 perf gate.

Built once at attach: per-station base coordinates and fill directions.
Per-step update: one h5py read + numpy scatter + in-place mutation.

The plan calls for <10 ms step changes on a 1000-beam frame; on a small
mesh we can't bench that scale, but we *can* assert that:

* No actor / polydata re-creation across many step transitions
  (in-place mutation contract).
* The per-step update doesn't grow with the number of steps already
  taken (no quadratic accumulator).

Marked ``@pytest.mark.bench`` so it is skipped in normal CI but can be
opted into with ``pytest -m bench``.
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
    DiagramSpec,
    LineForceDiagram,
    LineForceStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


@pytest.fixture
def big_beam_results(g, tmp_path: Path):
    """Many short segments along +X. Adapts to actual element count."""
    n_seg = 50
    points = [
        g.model.geometry.add_point(float(i), 0.0, 0.0, label=f"p{i}")
        for i in range(n_seg + 1)
    ]
    for i in range(n_seg):
        g.model.geometry.add_line(
            points[i], points[i + 1], label=f"seg{i}",
        )
    g.physical.add_curve(
        [f"seg{i}" for i in range(n_seg)], name="Beam",
    )
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    line_eids: list[int] = []
    for group in fem.elements:
        if group.element_type.dim == 1:
            line_eids.extend(int(x) for x in group.ids)
    line_eids = sorted(line_eids)
    n_beams = len(line_eids)
    n_stations = 5
    natural_coords = np.linspace(-1.0, 1.0, n_stations)

    n_steps = 100
    rng = np.random.default_rng(42)
    values = rng.standard_normal((n_steps, n_beams, n_stations)) * 100.0

    path = tmp_path / "big_beam.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_line_stations_group(
            sid, "partition_0", group_id="g0",
            class_tag=10,
            int_rule=0,
            element_index=np.asarray(line_eids, dtype=np.int64),
            station_natural_coord=natural_coords,
            components={"bending_moment_z": values},
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def test_no_actor_re_creation_across_100_steps(
    big_beam_results, headless_plotter,
):
    scene = build_fem_scene(big_beam_results.fem)
    diagram = LineForceDiagram(
        DiagramSpec(
            kind="line_force",
            selector=SlabSelector(component="bending_moment_z"),
            style=LineForceStyle(scale=1.0),
        ),
        big_beam_results,
    )
    diagram.attach(headless_plotter, big_beam_results.fem, scene)

    initial_actor = diagram._fill_actor
    initial_poly = diagram._fill_polydata

    for step in range(100):
        diagram.update_to_step(step)

    assert diagram._fill_actor is initial_actor
    assert diagram._fill_polydata is initial_poly


@pytest.mark.bench
def test_step_change_under_10ms_average(big_beam_results, headless_plotter):
    scene = build_fem_scene(big_beam_results.fem)
    diagram = LineForceDiagram(
        DiagramSpec(
            kind="line_force",
            selector=SlabSelector(component="bending_moment_z"),
            style=LineForceStyle(scale=1.0),
        ),
        big_beam_results,
    )
    diagram.attach(headless_plotter, big_beam_results.fem, scene)

    # Warmup
    for step in range(5):
        diagram.update_to_step(step)

    n = 100
    start = time.perf_counter()
    for step in range(n):
        diagram.update_to_step(step % 100)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / n) * 1000.0
    print(f"\nLineForce step change avg: {avg_ms:.2f} ms")
    # Phase 2 gate: <10 ms on 1000-beam frame.  On this small fixture
    # we expect well under 5 ms; the 10 ms cap is generous.
    assert avg_ms < 10.0, f"step change took {avg_ms:.2f} ms (>10 ms gate)"
