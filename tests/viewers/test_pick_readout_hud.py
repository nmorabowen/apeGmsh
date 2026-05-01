"""PickReadoutHUD — top-left viewport overlay tests.

The HUD updates from two signals: ``ProbeOverlay.on_point_result`` and
``ResultsDirector.subscribe_step``. These tests drive the HUD through
its callback (no Qt event loop, no interactive picking required) and
read back the rendered labels.

Note: ``ProbeOverlay.probe_at_point`` does not fire
``on_point_result`` — that callback is invoked by the interactive
picking path. To exercise the HUD we synthesize a ``PointProbeResult``
and feed it to the chained callback the HUD installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DiagramSpec,
    ResultsDirector,
    SlabSelector,
)
from apeGmsh.viewers.overlays.probe_overlay import (
    PointProbeResult,
    ProbeOverlay,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def results_for_hud(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_steps = 3
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, n_nodes))
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    components = {"displacement_z": base + t * 0.1}
    path = tmp_path / "hud.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids, components=components,
        )
        w.end_stage()
    return Results.from_native(path)


def _make_setup(qapp, results, *, with_diagram=True):
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    plotter = pv.Plotter(off_screen=True)
    director.bind_plotter(plotter, scene=scene)
    if with_diagram:
        spec = DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="displacement_z"),
            style=ContourStyle(),
        )
        diagram = ContourDiagram(spec, results)
        director.registry.add(diagram)
    overlay = ProbeOverlay(plotter, scene, director)
    viewport = QtWidgets.QWidget()
    viewport.resize(800, 600)
    viewport.show()
    from apeGmsh.viewers.ui._pick_readout_hud import PickReadoutHUD
    hud = PickReadoutHUD(viewport, overlay, director)
    return hud, overlay, director, scene, plotter


def _synth_result(scene, director, target_idx, values):
    coord = np.asarray(scene.grid.points)[target_idx]
    return PointProbeResult(
        position=coord,
        closest_node_id=int(scene.node_ids[target_idx]),
        closest_coord=coord,
        distance=0.0,
        step_index=director.step_index,
        field_values=values,
    )


# =====================================================================
# Construction
# =====================================================================

def test_hud_starts_in_empty_state(qapp, results_for_hud):
    hud, _, _, _, plotter = _make_setup(qapp, results_for_hud)
    assert hud._header.text() == "(no pick)"
    assert not hud._coords.isVisible()
    assert not hud._values.isVisible()
    plotter.close()


def test_hud_chains_existing_point_callback(qapp, results_for_hud):
    """If something else (e.g. ProbePaletteHUD) already wired
    ``on_point_result``, the HUD must wrap it — not replace it."""
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    scene = build_fem_scene(results_for_hud.fem)
    director = ResultsDirector(results_for_hud)
    plotter = pv.Plotter(off_screen=True)
    director.bind_plotter(plotter, scene=scene)
    overlay = ProbeOverlay(plotter, scene, director)

    seen = []
    overlay.on_point_result = lambda r: seen.append(r)

    viewport = QtWidgets.QWidget()
    viewport.resize(400, 300)
    viewport.show()
    from apeGmsh.viewers.ui._pick_readout_hud import PickReadoutHUD
    hud = PickReadoutHUD(viewport, overlay, director)

    # Simulate the overlay firing a result through the chain.
    result = _synth_result(scene, director, 0, {"displacement_z": 1.5})
    overlay.on_point_result(result)
    assert len(seen) == 1
    assert hud._header.text().startswith("node ")
    plotter.close()


# =====================================================================
# Point-pick rendering
# =====================================================================

def test_hud_updates_on_point_pick(qapp, results_for_hud):
    hud, overlay, director, scene, plotter = _make_setup(qapp, results_for_hud)
    target_idx = 3
    nid = int(scene.node_ids[target_idx])
    overlay.on_point_result(
        _synth_result(scene, director, target_idx, {"displacement_z": 7.5})
    )
    assert hud._header.text() == f"node {nid}"
    assert "displacement_z" in hud._values.text()
    plotter.close()


def test_hud_renders_value_with_six_significant_digits(
    qapp, results_for_hud,
):
    hud, overlay, director, scene, plotter = _make_setup(qapp, results_for_hud)
    overlay.on_point_result(
        _synth_result(scene, director, 1, {"displacement_z": 3.14159265})
    )
    assert "3.14159" in hud._values.text()
    plotter.close()


# =====================================================================
# Step-change re-read
# =====================================================================

def test_hud_tracks_step_changes(qapp, results_for_hud):
    hud, overlay, director, scene, plotter = _make_setup(qapp, results_for_hud)
    target_idx = 2
    nid = int(scene.node_ids[target_idx])
    director.set_step(0)
    # Real pick at step 0 — values come from director.read_at_pick.
    real_at_0 = director.read_at_pick(nid, ["displacement_z"])
    overlay.on_point_result(
        _synth_result(scene, director, target_idx, dict(real_at_0))
    )
    text_step_0 = hud._values.text()

    # Step to a different time. The step-change subscription should
    # re-read values for the cached node id.
    director.set_step(2)
    text_step_2 = hud._values.text()

    # The fixture writes a different value at step 2 vs step 0, so the
    # HUD's rendered values must differ.
    assert text_step_0 != text_step_2
    plotter.close()


# =====================================================================
# Empty-component fallback
# =====================================================================

def test_hud_empty_components_shows_hint(qapp, results_for_hud):
    """With no diagrams attached, a pick result with empty
    ``field_values`` must show the 'add diagrams' hint."""
    hud, overlay, director, scene, plotter = _make_setup(
        qapp, results_for_hud, with_diagram=False,
    )
    overlay.on_point_result(_synth_result(scene, director, 0, {}))
    assert "add diagrams" in hud._values.text()
    plotter.close()


# =====================================================================
# Stage change clears cached pick
# =====================================================================

def test_hud_clears_on_stage_change(qapp, results_for_hud):
    hud, overlay, director, scene, plotter = _make_setup(qapp, results_for_hud)
    overlay.on_point_result(
        _synth_result(scene, director, 0, {"displacement_z": 1.0})
    )
    assert hud._header.text().startswith("node ")
    # Fire stage-changed callback directly — only one stage in fixture
    # so calling set_stage with the same id won't fire, but invoking
    # the subscriber list directly exercises the handler.
    for cb in director.on_stage_changed:
        cb("any")
    assert hud._header.text() == "(no pick)"
    plotter.close()
