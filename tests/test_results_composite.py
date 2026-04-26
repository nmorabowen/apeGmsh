"""Phase 2 — Results composite API.

Selection by pg/label/ids, component access, time slicing, stage scoping.
The fixture writes a small synthetic native HDF5 for fast tests.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter


# =====================================================================
# Test fixture — synthetic native HDF5 with one stage, two components
# =====================================================================

def _make_synthetic(tmp_path: Path) -> Path:
    """Write a synthetic single-stage results file (no FEMData embedded)."""
    path = tmp_path / "synthetic.h5"
    time = np.array([0.0, 1.0, 2.0, 3.0])
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ux = np.tile(np.arange(5.0), (4, 1)) * np.array([[1.0], [2.0], [3.0], [4.0]])
    uy = np.full((4, 5), 0.1)
    elem_idx = np.array([10, 20], dtype=np.int64)
    nat = np.array([[0.0, 0.0]], dtype=np.float64)        # 1 GP per element
    sxx = np.array([[[1.0], [2.0]], [[1.5], [2.5]],
                     [[2.0], [3.0]], [[2.5], [3.5]]])

    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="dynamic", kind="transient", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": ux, "displacement_y": uy})
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_idx, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return path


# =====================================================================
# Single-stage: stage auto-resolves
# =====================================================================

def test_single_stage_auto_resolves(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        # No explicit stage needed when there's exactly one
        slab = r.nodes.get(component="displacement_x")
        assert slab.values.shape == (4, 5)
        assert slab.node_ids.tolist() == [1, 2, 3, 4, 5]


def test_get_by_ids(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        slab = r.nodes.get(ids=[2, 4], component="displacement_x")
        assert slab.node_ids.tolist() == [2, 4]


def test_get_by_pg_requires_fem(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        # No fem embedded → pg= raises
        with pytest.raises(RuntimeError, match="bound FEMData"):
            r.nodes.get(pg="Top", component="displacement_x")


def test_combining_selectors_raises(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        with pytest.raises(ValueError, match="not multiple"):
            r.nodes.get(ids=[1], pg="Top", component="displacement_x")


# =====================================================================
# Time slicing through the composite
# =====================================================================

def test_time_int_index(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        slab = r.nodes.get(component="displacement_x", time=2)
        assert slab.values.shape == (1, 5)
        # ux at step 2: [0, 3, 6, 9, 12]
        np.testing.assert_allclose(slab.values[0], [0, 3, 6, 9, 12])


def test_time_float_nearest(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        slab = r.nodes.get(component="displacement_x", time=1.4)
        # nearest to 1.4 is index 1 (t=1.0)
        np.testing.assert_allclose(slab.time, [1.0])


def test_time_float_slice(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        slab = r.nodes.get(
            component="displacement_x", time=slice(1.0, 3.0),
        )
        # half-open: t = 1.0, 2.0
        np.testing.assert_allclose(slab.time, [1.0, 2.0])


# =====================================================================
# Element / gauss composites delegate correctly
# =====================================================================

def test_gauss_composite(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        slab = r.elements.gauss.get(component="stress_xx")
        # 2 elements * 1 GP = 2 entries
        assert slab.values.shape == (4, 2)
        np.testing.assert_array_equal(slab.element_index, [10, 20])


def test_gauss_filter_by_ids(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        slab = r.elements.gauss.get(ids=[20], component="stress_xx")
        assert slab.element_index.tolist() == [20]


# =====================================================================
# Multi-stage — explicit picking required
# =====================================================================

def _make_two_stage(tmp_path: Path) -> Path:
    path = tmp_path / "two_stage.h5"
    node_ids = np.array([1, 2], dtype=np.int64)
    with NativeWriter(path) as w:
        w.open()
        sid_g = w.begin_stage(name="gravity", kind="static",
                               time=np.array([0.0]))
        w.write_nodes(sid_g, "partition_0", node_ids=node_ids,
                      components={"displacement_x": np.array([[1.0, 2.0]])})
        w.end_stage()
        sid_d = w.begin_stage(name="dynamic", kind="transient",
                               time=np.array([0.0, 1.0]))
        w.write_nodes(sid_d, "partition_0", node_ids=node_ids,
                      components={"displacement_x": np.array([[10.0, 20.0],
                                                                [11.0, 21.0]])})
        w.end_stage()
    return path


def test_multi_stage_no_default(tmp_path: Path) -> None:
    path = _make_two_stage(tmp_path)
    with Results.from_native(path) as r:
        with pytest.raises(RuntimeError, match="Multiple stages"):
            r.nodes.get(component="displacement_x")


def test_stage_scope_by_name(tmp_path: Path) -> None:
    path = _make_two_stage(tmp_path)
    with Results.from_native(path) as r:
        gravity = r.stage("gravity")
        assert gravity.name == "gravity"
        assert gravity.kind == "static"
        assert gravity.n_steps == 1
        slab = gravity.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values, [[1.0, 2.0]])


def test_stage_scope_by_id(tmp_path: Path) -> None:
    path = _make_two_stage(tmp_path)
    with Results.from_native(path) as r:
        # Stage IDs are stage_0, stage_1 in write order
        s = r.stage("stage_1")
        assert s.name == "dynamic"
        assert s.n_steps == 2


def test_explicit_stage_kwarg(tmp_path: Path) -> None:
    """Top-level Results lets reads pass stage= explicitly."""
    path = _make_two_stage(tmp_path)
    with Results.from_native(path) as r:
        slab = r.nodes.get(component="displacement_x", stage="dynamic")
        assert slab.values.shape == (2, 2)


def test_unknown_stage(tmp_path: Path) -> None:
    path = _make_two_stage(tmp_path)
    with Results.from_native(path) as r:
        with pytest.raises(KeyError, match="No stage matches"):
            r.stage("not_a_stage")


# =====================================================================
# Stage-only properties raise on unscoped Results
# =====================================================================

def test_stage_props_raise_on_unscoped(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        with pytest.raises(AttributeError, match="stage-scoped"):
            _ = r.kind
        with pytest.raises(AttributeError, match="stage-scoped"):
            _ = r.time


def test_stage_props_on_scoped(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        s = r.stage("dynamic")
        assert s.kind == "transient"
        assert s.n_steps == 4
        np.testing.assert_allclose(s.time, [0.0, 1.0, 2.0, 3.0])


# =====================================================================
# Available components / inspect summary
# =====================================================================

def test_available_components(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        comps = r.nodes.available_components()
        assert set(comps) == {"displacement_x", "displacement_y"}
        gcomps = r.elements.gauss.available_components()
        assert set(gcomps) == {"stress_xx"}


def test_inspect_summary_runs(tmp_path: Path) -> None:
    path = _make_synthetic(tmp_path)
    with Results.from_native(path) as r:
        s = r.inspect.summary()
        assert "Stages" in s or "stage" in s.lower()
        assert "dynamic" in s
