"""Phase 1 — multi-partition stitching, on legacy files.

``NativeWriter`` refuses a second partition per stage (ADR 0123 D5), so
the multi-partition files these reads need are built with raw h5py
(``tests/results/_native_legacy.py``). The reader keeps stitching them
(D4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apeGmsh.results.readers import NativeReader

from tests.results._native_legacy import write_multi_partition


def _write_partitioned(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a 3-partition file. Returns (all_node_ids, all_displacement_x)."""
    time = np.array([0.0, 1.0])

    # Partition 0: nodes 1, 2, 3
    p0_ids = np.array([1, 2, 3], dtype=np.int64)
    p0_ux = np.array([[0.10, 0.20, 0.30], [0.11, 0.22, 0.33]])

    # Partition 1: nodes 4, 5
    p1_ids = np.array([4, 5], dtype=np.int64)
    p1_ux = np.array([[0.40, 0.50], [0.44, 0.55]])

    # Partition 2: nodes 6, 7, 8, 9
    p2_ids = np.array([6, 7, 8, 9], dtype=np.int64)
    p2_ux = np.array([[0.60, 0.70, 0.80, 0.90],
                       [0.66, 0.77, 0.88, 0.99]])

    write_multi_partition(path, {
        "partition_0": (p0_ids, {"displacement_x": p0_ux}),
        "partition_1": (p1_ids, {"displacement_x": p1_ux}),
        "partition_2": (p2_ids, {"displacement_x": p2_ux}),
    }, time=time, kind="transient")

    all_ids = np.concatenate([p0_ids, p1_ids, p2_ids])
    all_ux = np.concatenate([p0_ux, p1_ux, p2_ux], axis=1)
    return all_ids, all_ux


def test_partitions_listed(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    _write_partitioned(path)
    with NativeReader(path) as r:
        assert r.partitions("stage_0") == [
            "partition_0", "partition_1", "partition_2",
        ]


def test_full_read_stitches_across_partitions(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    all_ids, all_ux = _write_partitioned(path)
    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x")
    # Order of stitching matches sorted partition_id order, which matches
    # the write order here.
    np.testing.assert_array_equal(slab.node_ids, all_ids)
    np.testing.assert_allclose(slab.values, all_ux)


def test_filter_spans_partitions(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    _write_partitioned(path)
    # Pick one node from each partition
    requested = np.array([2, 5, 8], dtype=np.int64)
    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x", node_ids=requested)
    np.testing.assert_array_equal(slab.node_ids, [2, 5, 8])
    # Step 0 values for nodes 2, 5, 8
    np.testing.assert_allclose(slab.values[0], [0.20, 0.50, 0.80])
    # Step 1
    np.testing.assert_allclose(slab.values[1], [0.22, 0.55, 0.88])


def test_filter_skips_partitions_with_no_match(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    _write_partitioned(path)
    # Only nodes from partition_0
    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x",
                             node_ids=np.array([1, 3]))
    np.testing.assert_array_equal(slab.node_ids, [1, 3])
    np.testing.assert_allclose(slab.values[0], [0.10, 0.30])


def test_shared_boundary_node_deduped(tmp_path: Path) -> None:
    # A boundary node replicated across partition domains must yield one
    # column, not one per partition (F1). Its copies agree, so it reads
    # once (#2360; disagreeing copies refuse, test_native_repeated_node_ids).
    path = tmp_path / "run.h5"
    time = np.array([0.0, 1.0])
    # Node 3 is shared by both partitions with identical kinematics.
    p0_ids = np.array([1, 2, 3], dtype=np.int64)
    p0_ux = np.array([[0.10, 0.20, 0.30], [0.11, 0.22, 0.33]])
    p1_ids = np.array([3, 4], dtype=np.int64)
    p1_ux = np.array([[0.30, 0.40], [0.33, 0.44]])

    write_multi_partition(path, {
        "partition_0": (p0_ids, {"displacement_x": p0_ux}),
        "partition_1": (p1_ids, {"displacement_x": p1_ux}),
    }, time=time, kind="transient")

    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x")

    # Node 3 appears exactly once; ids are the unique set.
    np.testing.assert_array_equal(slab.node_ids, [1, 2, 3, 4])
    assert slab.values.shape == (2, 4)
    # Column for node 3 carries the (consistent) shared value.
    col3 = slab.values[:, list(slab.node_ids).index(3)]
    np.testing.assert_allclose(col3, [0.30, 0.33])


def test_native_springs_returns_empty_slab(tmp_path: Path) -> None:
    # Native files have no spring-recording path; read_springs returns an
    # empty slab rather than raising AttributeError from the composite
    # layer (F3).
    path = tmp_path / "run.h5"
    _write_partitioned(path)
    with NativeReader(path) as r:
        slab = r.read_springs("stage_0", "spring_force_0")
    assert slab.element_index.size == 0
    assert slab.values.shape == (2, 0)
    np.testing.assert_allclose(slab.time, [0.0, 1.0])
