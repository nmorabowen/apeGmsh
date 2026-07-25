"""ContourDiagram — its registration with the viewer's legends.

Post-ADR-0081 the diagram no longer owns a bar: it registers a layer
with the ``LegendController`` and forwards the settings-tab setters.
So these cover the *seam* — attach registers, detach retires, the
setters reach the controller, and the style's four scalar-bar fields
act as initial preferences. Layout itself (box geometry, fit, overlap)
is ``test_legend_layout.py``'s job.

Uses an off-screen pyvista plotter — no Qt required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram, ContourStyle, DiagramSpec, SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


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
    return Results.from_native(path, model=_open_model_from_h5(path))


# headless_plotter is a shared fixture in tests/viewers/conftest.py
# (yields a PyVistaQtBackend, ADR 0042 R-B.final).


def _spec(component: str = "displacement_z") -> DiagramSpec:
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(),
    )


def _bar_titles(backend) -> list[str]:
    # Accept either a RenderBackend (ADR 0042 — the headless_plotter
    # fixture) or a raw plotter; scalar bars live on the pyvista plotter.
    plotter = getattr(backend, "plotter", backend)
    bars = getattr(plotter, "scalar_bars", None)
    if bars is None:
        return []
    return list(bars.keys())


def _legends(backend):
    from apeGmsh.viewers.core._legend import controller_for
    return controller_for(backend)


# =====================================================================
# Attach registers a legend; detach retires it
# =====================================================================

def test_attach_creates_bar_with_component_title(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert "displacement_z" in _bar_titles(headless_plotter)
    assert diagram._legend_key == ("", "displacement_z")


def test_default_fmt_lands_on_bar_actor(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetLabelFormat() == "%.3g"


def test_detach_removes_scalar_bar(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))
    assert "displacement_z" in _bar_titles(headless_plotter)

    diagram.detach()
    assert "displacement_z" not in _bar_titles(headless_plotter)
    assert _legends(headless_plotter).entries() == []


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
    assert _legends(headless_plotter).entries() == []


# =====================================================================
# The setters reach the controller
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
    assert diagram._effective_show_scalar_bar() is False


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


def test_set_fmt_updates_label_format_live(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_fmt("%.2e")
    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetLabelFormat() == "%.2e"
    assert diagram._effective_fmt() == "%.2e"


def test_set_fmt_persists_through_show_toggle(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_fmt("%.4f")
    diagram.set_show_scalar_bar(False)
    diagram.set_show_scalar_bar(True)

    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetLabelFormat() == "%.4f"


def test_set_scalar_bar_vertical_flips_orientation(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    # Vertical is the default (ADR 0081 L0) — VTK's correct layout.
    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetOrientation() == 1

    diagram.set_scalar_bar_vertical(False)
    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetOrientation() == 0

    diagram.set_scalar_bar_vertical(True)
    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetOrientation() == 1


def test_set_scalar_bar_scale_scales_the_text_and_the_box(
    results_with_known_disp, headless_plotter,
):
    """The size knob drives the fonts, and the box follows.

    Pre-0081 it multiplied the box alone while the text stayed pinned
    at 18 pt, so shrinking produced a legend whose title overlapped its
    own tick labels (ADR 0081 D3).
    """
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_scalar_bar_scale(0.6)
    small = headless_plotter.plotter.scalar_bars["displacement_z"]
    small_pt = small.GetTitleTextProperty().GetFontSize()
    small_box = (small.GetWidth(), small.GetHeight())

    diagram.set_scalar_bar_scale(2.0)
    big = headless_plotter.plotter.scalar_bars["displacement_z"]
    big_pt = big.GetTitleTextProperty().GetFontSize()
    big_box = (big.GetWidth(), big.GetHeight())

    assert big_pt > small_pt, "font size must follow the size knob"
    assert big_box[0] > small_box[0] and big_box[1] > small_box[1]


def test_layout_overrides_survive_show_toggle(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    diagram = ContourDiagram(_spec(), r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    diagram.set_scalar_bar_vertical(True)
    diagram.set_scalar_bar_scale(0.75)
    diagram.set_show_scalar_bar(False)
    diagram.set_show_scalar_bar(True)

    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetOrientation() == 1
    assert diagram._effective_bar_scale() == pytest.approx(0.75)


# =====================================================================
# The style's scalar-bar fields are initial preferences
# =====================================================================

def test_style_scalar_bar_layout_applies_at_attach(
    results_with_known_disp, headless_plotter,
):
    r = results_with_known_disp
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(scalar_bar_vertical=True, scalar_bar_scale=1.5),
    )
    diagram = ContourDiagram(spec, r)
    diagram.attach(headless_plotter, r.fem, build_fem_scene(r.fem))

    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetOrientation() == 1
    assert diagram._effective_bar_scale() == pytest.approx(1.5)


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


def test_a_second_attach_does_not_redecorate_a_live_legend(
    results_with_known_disp, headless_plotter,
):
    """Style preferences seed a legend; they never overwrite one.

    A re-attach (stage change, restack) would otherwise undo every edit
    the user had made — the ADR 0081 D2 failure mode in a different
    disguise.
    """
    r = results_with_known_disp
    scene = build_fem_scene(r.fem)
    first = ContourDiagram(_spec(), r)
    first.attach(headless_plotter, r.fem, scene)
    first.set_scalar_bar_vertical(True)

    second = ContourDiagram(
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="displacement_z"),
            style=ContourStyle(scalar_bar_vertical=False),
        ),
        r,
    )
    second.attach(headless_plotter, r.fem, scene)

    bar = headless_plotter.plotter.scalar_bars["displacement_z"]
    assert bar.GetOrientation() == 1
