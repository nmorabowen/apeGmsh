"""ProbeOverlay — mock-driven interactive-callback tests.

Bypasses PyVista's ``enable_point_picking`` (which needs a real Qt
event loop and a live interactor to actually fire) by calling the
overlay's internal ``_on_interactive_*`` slots directly with synthetic
world coordinates. This exercises:

* The single-click point flow: callback receives a ``PointProbeResult``,
  mode resets, picking gets disabled.
* The two-click line flow: first click advances ``LINE_START`` →
  ``LINE_END`` and stashes the start coord; second click fires
  ``on_line_result`` with both endpoints.
* The ``stop()`` / state-reset paths.
* Robustness — observers that raise don't propagate, ``None``
  picks are no-ops.

Production callsite is ``plotter.enable_point_picking(callback=...)``
in ``start_point_probe`` / ``start_line_probe``; PyVista hands
``callback(world_xyz)`` to the same slots we drive here.
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
    LineProbeResult,
    PointProbeResult,
    ProbeMode,
    ProbeOverlay,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixture
# =====================================================================

@pytest.fixture
def overlay_with_diagram(g, tmp_path: Path):
    """Real off-screen plotter with one ContourDiagram attached so probes
    have a component to read."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 2
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, n_nodes))
    components = {
        "displacement_z": base + np.arange(n_steps, dtype=np.float64
                                           ).reshape(-1, 1) * 0.1,
    }

    path = tmp_path / "interactive.h5"
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
    results = Results.from_native(path)

    plotter = pv.Plotter(off_screen=True)
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.bind_plotter(plotter, scene=scene)

    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )
    diagram = ContourDiagram(spec, results)
    director.registry.add(diagram)

    overlay = ProbeOverlay(plotter, scene, director)

    yield overlay, scene, plotter

    plotter.close()


def _picked_world_for(scene, idx: int) -> np.ndarray:
    """Return a world coord very close to substrate node ``idx``."""
    return np.asarray(scene.grid.points)[idx] + np.array([0.001, 0.0, 0.0])


# =====================================================================
# Point-probe callback
# =====================================================================

def test_point_callback_fires_observer_with_result(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    received: list[PointProbeResult] = []
    overlay.on_point_result = received.append

    # Avoid hitting plotter.enable_point_picking — fake the entry-point
    # so we can assert state without a live picker.
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    overlay.start_point_probe()
    assert overlay._mode is ProbeMode.POINT

    # PyVista's picker would call this with the world XYZ
    fake_pick = _picked_world_for(scene, 3)
    overlay._on_interactive_point(fake_pick)

    assert overlay._mode is ProbeMode.NONE
    assert len(received) == 1
    assert isinstance(received[0], PointProbeResult)
    assert received[0].closest_node_id == int(scene.node_ids[3])
    # The result also lands on the overlay's history
    assert len(overlay.point_results) == 1


def test_point_callback_handles_none_input(
    overlay_with_diagram, monkeypatch,
):
    overlay, _, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)
    received: list = []
    overlay.on_point_result = received.append

    overlay._on_interactive_point(None)
    assert received == []
    assert len(overlay.point_results) == 0


def test_point_callback_observer_exception_doesnt_propagate(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    def _bad_observer(_result):
        raise RuntimeError("boom")

    overlay.on_point_result = _bad_observer
    # Should not raise
    overlay._on_interactive_point(_picked_world_for(scene, 0))
    # Result still recorded despite observer failure
    assert len(overlay.point_results) == 1


# =====================================================================
# Line-probe callback
# =====================================================================

def test_line_first_click_advances_to_end_state(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    enable_calls: list[dict] = []

    def _capture_enable(**kwargs):
        enable_calls.append(kwargs)

    monkeypatch.setattr(plotter, "enable_point_picking", _capture_enable)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    overlay.start_line_probe()
    # The first start_line_probe call enables picking once.
    assert overlay._mode is ProbeMode.LINE_START
    assert len(enable_calls) == 1

    a = _picked_world_for(scene, 1)
    overlay._on_interactive_line(a)

    assert overlay._mode is ProbeMode.LINE_END
    np.testing.assert_array_equal(overlay._line_start, a)
    # First click also re-enables picking with a new prompt
    assert len(enable_calls) == 2
    assert "Click point B" in enable_calls[1].get("show_message", "")


def test_line_second_click_completes_probe(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    received: list[LineProbeResult] = []
    overlay.on_line_result = received.append

    overlay.start_line_probe()
    a = _picked_world_for(scene, 0)
    b = _picked_world_for(scene, 5)
    overlay._on_interactive_line(a)
    overlay._on_interactive_line(b)

    assert overlay._mode is ProbeMode.NONE
    assert overlay._line_start is None
    assert len(received) == 1
    np.testing.assert_array_equal(received[0].point_a, a)
    np.testing.assert_array_equal(received[0].point_b, b)
    assert len(overlay.line_results) == 1


def test_line_second_click_uses_default_n_samples(
    overlay_with_diagram, monkeypatch,
):
    """Default sample count for the interactive flow is 50 (per
    ``probe_along_line``'s default)."""
    overlay, scene, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    received: list[LineProbeResult] = []
    overlay.on_line_result = received.append
    overlay.start_line_probe()
    overlay._on_interactive_line(_picked_world_for(scene, 0))
    overlay._on_interactive_line(_picked_world_for(scene, 1))

    assert received[0].n_samples == 50


def test_line_callback_handles_none_at_either_step(
    overlay_with_diagram, monkeypatch,
):
    overlay, _, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    overlay.start_line_probe()
    overlay._on_interactive_line(None)
    # State unchanged — still waiting on the first click
    assert overlay._mode is ProbeMode.LINE_START

    overlay._on_interactive_line(np.array([0.5, 0.5, 0.5]))
    overlay._on_interactive_line(None)
    # State unchanged — still waiting on the second click
    assert overlay._mode is ProbeMode.LINE_END


def test_line_observer_exception_doesnt_propagate(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    def _bad_observer(_result):
        raise RuntimeError("boom")

    overlay.on_line_result = _bad_observer
    overlay.start_line_probe()
    overlay._on_interactive_line(_picked_world_for(scene, 0))
    overlay._on_interactive_line(_picked_world_for(scene, 1))
    # Should not raise; result recorded
    assert len(overlay.line_results) == 1
    assert overlay._mode is ProbeMode.NONE


# =====================================================================
# stop()
# =====================================================================

def test_stop_after_point_probe_resets_mode(
    overlay_with_diagram, monkeypatch,
):
    overlay, _, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    disable_calls: list[int] = []
    monkeypatch.setattr(
        plotter, "disable_picking",
        lambda: disable_calls.append(1),
    )

    overlay.start_point_probe()
    assert overlay._mode is ProbeMode.POINT
    overlay.stop()
    assert overlay._mode is ProbeMode.NONE
    # disable_picking gets called both at start (safe-disable) and at stop
    assert disable_calls    # at least one


def test_stop_after_first_line_click_clears_start(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    overlay.start_line_probe()
    overlay._on_interactive_line(_picked_world_for(scene, 0))
    assert overlay._mode is ProbeMode.LINE_END
    assert overlay._line_start is not None

    overlay.stop()
    assert overlay._mode is ProbeMode.NONE
    assert overlay._line_start is None


def test_stop_when_idle_is_safe(overlay_with_diagram, monkeypatch):
    overlay, _, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)
    overlay.stop()    # should not raise
    assert overlay._mode is ProbeMode.NONE


# =====================================================================
# clear() after interactive pick
# =====================================================================

def test_clear_after_interactive_pick_drops_results_and_markers(
    overlay_with_diagram, monkeypatch,
):
    overlay, scene, plotter = overlay_with_diagram
    monkeypatch.setattr(plotter, "enable_point_picking", lambda **kw: None)
    monkeypatch.setattr(plotter, "disable_picking", lambda: None)

    overlay.start_point_probe()
    overlay._on_interactive_point(_picked_world_for(scene, 0))
    assert len(overlay.point_results) == 1
    assert len(overlay._point_actor_names) >= 1

    overlay.clear()
    assert len(overlay.point_results) == 0
    assert len(overlay._point_actor_names) == 0
