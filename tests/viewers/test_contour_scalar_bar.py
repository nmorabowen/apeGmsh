"""ContourDiagram — scalar-bar lifecycle and runtime overrides.

Exercises the live show/hide toggle, the fmt override, and the leak
fix on detach. Uses an off-screen pyvista plotter — no Qt required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram, ContourStyle, DiagramSpec, SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixtures (mirror test_contour_diagram.py — minimal nodal data)
# =====================================================================

@pytest.fixture
def results_with_known_disp(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 2
    values = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    for t in range(n_steps):
        values[t] = node_ids + t * 1000.0

    path = tmp_path / "known_disp.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": values},
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _spec(component: str = "displacement_z") -> DiagramSpec:
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(),
    )


def _bar_titles(plotter) -> list[str]:
    bars = getattr(plotter, "scalar_bars", None)
    if bars is None:
        return []
    return list(bars.keys())


# =====================================================================
# Default attach registers the bar with the requested title
# =====================================================================

def test_attach_creates_bar_with_component_title(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert "displacement_z" in _bar_titles(headless_plotter)


def test_default_fmt_lands_on_bar_actor(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    bar = headless_plotter.scalar_bars["displacement_z"]
    assert bar.GetLabelFormat() == "%.3g"


# =====================================================================
# Live show/hide toggle
# =====================================================================

def test_set_show_scalar_bar_false_removes_bar(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert "displacement_z" in _bar_titles(headless_plotter)

    diagram.set_show_scalar_bar(False)
    assert "displacement_z" not in _bar_titles(headless_plotter)
    assert diagram._runtime_show_scalar_bar is False


def test_set_show_scalar_bar_true_re_adds_bar(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_show_scalar_bar(False)
    assert "displacement_z" not in _bar_titles(headless_plotter)

    diagram.set_show_scalar_bar(True)
    assert "displacement_z" in _bar_titles(headless_plotter)


# =====================================================================
# Live fmt override
# =====================================================================

def test_set_fmt_updates_label_format_live(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_fmt("%.2e")
    bar = headless_plotter.scalar_bars["displacement_z"]
    assert bar.GetLabelFormat() == "%.2e"
    assert diagram._runtime_fmt == "%.2e"


def test_set_fmt_persists_through_show_toggle(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_fmt("%.4f")
    diagram.set_show_scalar_bar(False)
    diagram.set_show_scalar_bar(True)

    bar = headless_plotter.scalar_bars["displacement_z"]
    assert bar.GetLabelFormat() == "%.4f"


# =====================================================================
# Leak fix — detach removes the bar
# =====================================================================

def test_detach_removes_scalar_bar(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert "displacement_z" in _bar_titles(headless_plotter)

    diagram.detach()
    assert "displacement_z" not in _bar_titles(headless_plotter)


def test_repeated_attach_detach_does_not_accumulate_bars(
    results_with_known_disp, headless_plotter,
):
    """Three attach/detach cycles should leave zero residual bars."""
    r = results_with_known_disp
    scene = build_fem_scene(r.fem)
    for _ in range(3):
        d = ContourDiagram(_spec(), r)
        d.attach(headless_plotter, r.fem, scene)
        d.detach()
    assert "displacement_z" not in _bar_titles(headless_plotter)


# =====================================================================
# Initial show=False on style suppresses the bar
# =====================================================================

def test_attach_with_show_scalar_bar_false_creates_no_bar(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(show_scalar_bar=False),
    )
    diagram = ContourDiagram(spec, r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert "displacement_z" not in _bar_titles(headless_plotter)
