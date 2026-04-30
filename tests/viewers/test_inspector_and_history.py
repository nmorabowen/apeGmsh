"""Phase 5 v1 — Director read helpers + Inspector data path.

These tests exercise the data accessors without spinning up Qt:

* ``Director.read_at_pick(node, components, step=)`` returns
  the right scalar values keyed by component.
* ``Director.read_history(node, component)`` returns
  ``(time, values)`` for the full stage at one node.

The Inspector and TimeHistoryPanel widgets themselves are Qt
widgets — covered by an importability sanity check (full UI
testing needs pytest-qt + a live event loop).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import ResultsDirector


# =====================================================================
# Fixture: small Results with multiple components per node
# =====================================================================

@pytest.fixture
def multi_component_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 5
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, n_nodes))
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    components = {
        "displacement_x": base + t * 0.1,
        "displacement_y": base + t * 0.2,
        "displacement_z": base + t * 0.3,
    }

    path = tmp_path / "multi.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.linspace(0.0, 1.0, n_steps),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    return Results.from_native(path), node_ids, components


# =====================================================================
# read_at_pick
# =====================================================================

def test_read_at_pick_no_stage_returns_empty(multi_component_results):
    results, _, _ = multi_component_results
    director = ResultsDirector(results)
    director._stage_id = None    # force unscoped
    out = director.read_at_pick(1, ["displacement_x"])
    assert out == {}


def test_read_at_pick_returns_scalars_per_component(
    multi_component_results,
):
    results, node_ids, components = multi_component_results
    director = ResultsDirector(results)
    # Single stage auto-picked
    target_node = int(node_ids[0])
    out = director.read_at_pick(
        target_node,
        ["displacement_x", "displacement_y", "displacement_z"],
    )
    assert set(out.keys()) == {
        "displacement_x", "displacement_y", "displacement_z",
    }
    # Step 0 values: dx = nid + 0, dy = nid + 0, dz = nid + 0
    assert out["displacement_x"] == pytest.approx(float(target_node))
    assert out["displacement_y"] == pytest.approx(float(target_node))
    assert out["displacement_z"] == pytest.approx(float(target_node))


def test_read_at_pick_uses_explicit_step(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    target_node = int(node_ids[0])
    out = director.read_at_pick(
        target_node, ["displacement_x"], step=3,
    )
    # Step 3 dx = nid + 3 * 0.1 = nid + 0.3
    assert out["displacement_x"] == pytest.approx(
        float(target_node) + 0.3,
    )


def test_read_at_pick_uses_current_step_default(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    director.set_step(2)
    target_node = int(node_ids[0])
    out = director.read_at_pick(target_node, ["displacement_x"])
    # Step 2 dx = nid + 0.2
    assert out["displacement_x"] == pytest.approx(
        float(target_node) + 0.2,
    )


def test_read_at_pick_skips_unknown_component(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    out = director.read_at_pick(
        int(node_ids[0]),
        ["displacement_x", "nonexistent_component"],
    )
    assert "displacement_x" in out
    assert "nonexistent_component" not in out


# =====================================================================
# read_history
# =====================================================================

def test_read_history_returns_full_stage(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    target_node = int(node_ids[0])
    data = director.read_history(target_node, "displacement_x")
    assert data is not None
    time, values = data
    # 5 steps in fixture
    assert time.size == 5
    assert values.size == 5
    # Time vector spans 0..1
    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(1.0)


def test_read_history_values_match_fixture(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    target_node = int(node_ids[0])
    data = director.read_history(target_node, "displacement_z")
    assert data is not None
    _, values = data
    expected = np.array(
        [float(target_node) + step * 0.3 for step in range(5)],
    )
    np.testing.assert_allclose(values, expected)


def test_read_history_no_stage_returns_none(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    director._stage_id = None
    assert director.read_history(int(node_ids[0]), "displacement_x") is None


def test_read_history_unknown_component_returns_none(multi_component_results):
    results, node_ids, _ = multi_component_results
    director = ResultsDirector(results)
    data = director.read_history(int(node_ids[0]), "nonexistent_xyz")
    assert data is None


# =====================================================================
# Importability of UI modules (no Qt event loop, just module-load)
# =====================================================================

def test_inspector_tab_module_imports():
    # Importing the module itself shouldn't require Qt — we use lazy
    # imports.
    from apeGmsh.viewers.ui import _inspector_tab    # noqa: F401


def test_time_history_panel_module_imports():
    from apeGmsh.viewers.ui import _time_history    # noqa: F401
