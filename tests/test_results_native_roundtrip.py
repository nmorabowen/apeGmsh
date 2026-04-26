"""Phase 1 — synthetic round-trip through NativeWriter / NativeReader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.readers import NativeReader, ResultLevel
from apeGmsh.results.writers import NativeWriter


# =====================================================================
# Nodal round-trip
# =====================================================================

def test_nodes_full_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.linspace(0.0, 1.0, 11)        # 11 steps
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    ux = np.outer(time, np.array([0.1, 0.2, 0.3, 0.4]))   # (11, 4)
    uy = np.outer(time, np.array([1.0, 2.0, 3.0, 4.0]))

    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="static", kind="static", time=time)
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_x": ux, "displacement_y": uy},
        )
        w.end_stage()

    with NativeReader(path) as r:
        stages = r.stages()
        assert len(stages) == 1
        assert stages[0].name == "static"
        assert stages[0].kind == "static"
        assert stages[0].n_steps == 11

        comps = r.available_components(stages[0].id, ResultLevel.NODES)
        assert set(comps) == {"displacement_x", "displacement_y"}

        slab = r.read_nodes(stages[0].id, "displacement_x")
        np.testing.assert_array_equal(slab.node_ids, node_ids)
        np.testing.assert_allclose(slab.values, ux)
        np.testing.assert_allclose(slab.time, time)


def test_nodes_filter_by_ids(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.array([0.0, 1.0, 2.0])
    node_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    ux = np.tile(np.arange(5.0), (3, 1))     # (3, 5) — distinct values per node

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": ux})
        w.end_stage()

    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x",
                            node_ids=np.array([20, 50], dtype=np.int64))
        np.testing.assert_array_equal(slab.node_ids, [20, 50])
        # Columns 1 and 4 of the original (zero-indexed)
        np.testing.assert_allclose(slab.values, ux[:, [1, 4]])


def test_nodes_time_slice(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.linspace(0.0, 10.0, 11)
    node_ids = np.array([1, 2], dtype=np.int64)
    ux = np.outer(time, np.array([1.0, 2.0]))   # (11, 2)

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": ux})
        w.end_stage()

    with NativeReader(path) as r:
        # Single step by index
        s = r.read_nodes("stage_0", "displacement_x", time_slice=5)
        np.testing.assert_allclose(s.values, ux[5:6])
        np.testing.assert_allclose(s.time, [time[5]])

        # Slice by float values [3.0, 7.0)
        s = r.read_nodes("stage_0", "displacement_x",
                          time_slice=slice(3.0, 7.0))
        # time is [0, 1, ..., 10], so indices 3..6 inclusive
        expected_idx = np.array([3, 4, 5, 6])
        np.testing.assert_allclose(s.values, ux[expected_idx])
        np.testing.assert_allclose(s.time, time[expected_idx])

        # Single time value (nearest)
        s = r.read_nodes("stage_0", "displacement_x", time_slice=4.7)
        # Nearest step to t=4.7 is index 5 (t=5.0)
        np.testing.assert_allclose(s.values, ux[5:6])


# =====================================================================
# Gauss point round-trip
# =====================================================================

def test_gauss_full_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.array([0.0, 0.5, 1.0])
    elem_idx = np.array([1, 2, 3], dtype=np.int64)            # 3 elements
    nat_coords = np.array([[-1.0, -1.0, -1.0],                # 4 GPs (tet4)
                            [ 1.0, -1.0, -1.0],
                            [-1.0,  1.0, -1.0],
                            [-1.0, -1.0,  1.0]], dtype=np.float64)
    # values shape (T, E, n_gp) = (3, 3, 4)
    sxx = np.arange(36, dtype=np.float64).reshape(3, 3, 4)

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4,
            int_rule=1,
            element_index=elem_idx,
            natural_coords=nat_coords,
            components={"stress_xx": sxx},
        )
        w.end_stage()

    with NativeReader(path) as r:
        comps = r.available_components("stage_0", ResultLevel.GAUSS)
        assert comps == ["stress_xx"]

        slab = r.read_gauss("stage_0", "stress_xx")
        # sum_GP = 3 elements * 4 GPs = 12
        assert slab.values.shape == (3, 12)
        assert slab.element_index.shape == (12,)
        # Each element appears 4 times (once per GP)
        np.testing.assert_array_equal(slab.element_index,
                                       np.repeat(elem_idx, 4))
        # Natural coords are the same 4 GPs tiled per element
        np.testing.assert_allclose(slab.natural_coords[:4], nat_coords)
        np.testing.assert_allclose(slab.natural_coords[4:8], nat_coords)
        assert slab.local_axes_quaternion is None


def test_gauss_with_local_axes_quaternion(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.array([0.0, 1.0])
    elem_idx = np.array([5, 7], dtype=np.int64)
    nat = np.array([[-0.5, -0.5], [0.5, -0.5]], dtype=np.float64)
    quat = np.array([[1, 0, 0, 0], [0.7, 0.7, 0, 0]], dtype=np.float64)
    sxx = np.zeros((2, 2, 2))

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=203, int_rule=200,
            element_index=elem_idx,
            natural_coords=nat,
            components={"stress_xx": sxx},
            local_axes_quaternion=quat,
        )
        w.end_stage()

    with NativeReader(path) as r:
        slab = r.read_gauss("stage_0", "stress_xx")
        assert slab.local_axes_quaternion is not None
        # Quaternion repeats per GP within element.
        assert slab.local_axes_quaternion.shape == (4, 4)
        # Both rows for element 0 carry quat[0]; both rows for element 1 carry quat[1].
        np.testing.assert_allclose(
            slab.local_axes_quaternion[:2], np.tile(quat[0], (2, 1)),
        )
        np.testing.assert_allclose(
            slab.local_axes_quaternion[2:], np.tile(quat[1], (2, 1)),
        )


def test_gauss_filter_by_element_ids(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.array([0.0])
    elem_idx = np.array([1, 2, 3, 4], dtype=np.int64)
    nat = np.array([[0.0]], dtype=np.float64)                  # 1 GP per element
    sxx = np.array([[[1.1], [2.2], [3.3], [4.4]]], dtype=np.float64)  # (1, 4, 1)

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=time)
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_idx,
            natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()

    with NativeReader(path) as r:
        slab = r.read_gauss("stage_0", "stress_xx",
                             element_ids=np.array([2, 4]))
        np.testing.assert_array_equal(slab.element_index, [2, 4])
        np.testing.assert_allclose(slab.values, [[2.2, 4.4]])


# =====================================================================
# Fibers round-trip
# =====================================================================

def test_fibers_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    time = np.array([0.0, 1.0])
    # 2 elements, 2 GPs each, 3 fibers per GP → 12 fiber rows total
    n = 12
    eidx = np.repeat([1, 2], 6)
    gidx = np.tile(np.repeat([0, 1], 3), 2)
    y = np.tile([-1.0, 0.0, 1.0], 4)
    z = np.zeros(n)
    area = np.full(n, 0.5)
    mat = np.tile([100, 200, 200], 4)
    sigma = np.arange(2 * n, dtype=np.float64).reshape(2, n)

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_fibers_group(
            sid, "partition_0", "group_0",
            section_tag=1, section_class="FiberSection",
            element_index=eidx, gp_index=gidx,
            y=y, z=z, area=area, material_tag=mat,
            components={"fiber_stress": sigma},
        )
        w.end_stage()

    with NativeReader(path) as r:
        slab = r.read_fibers("stage_0", "fiber_stress")
        np.testing.assert_array_equal(slab.element_index, eidx)
        np.testing.assert_array_equal(slab.gp_index, gidx)
        np.testing.assert_allclose(slab.y, y)
        np.testing.assert_allclose(slab.area, area)
        np.testing.assert_array_equal(slab.material_tag, mat)
        np.testing.assert_allclose(slab.values, sigma)
