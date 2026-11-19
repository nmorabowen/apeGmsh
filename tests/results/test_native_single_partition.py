"""ADR 0123 D5: a native ``results.h5`` stage holds at most one partition.

``NativeWriter._require_partition``, which every ``write_*`` passes
through, refuses a partition id other than the one the stage already
holds, by name (file, stage, existing id, refused id), before creating any
group. The rule is about count, not name: a stage whose only partition is
``partition_3`` is legal, a stage's writes re-enter its partition freely,
and each stage has its own partition. Legacy multi-partition files keep
reading (D4): ``test_native_repeated_node_ids.py`` and
``test_results_native_partitions.py``, on the raw-h5py builder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.writers import NativeWriter
from apeGmsh.results.writers._native import NativeSecondPartition

_TIME = np.array([0.0, 0.5, 1.0])
_T = _TIME.size


@pytest.fixture
def writer(tmp_path: Path) -> Iterator[tuple[NativeWriter, str, Path]]:
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        yield w, sid, path


def _partitions(path: Path, sid: str = "stage_0") -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(f[f"/stages/{sid}/partitions"].keys())


# Every write_* entry, each writing one small, valid group into ``pid``.
_WRITES: dict[str, Callable[[NativeWriter, str, str], None]] = {
    "nodes": lambda w, sid, pid: w.write_nodes(
        sid, pid, node_ids=np.array([1, 2]),
        components={"displacement_x": np.zeros((_T, 2))},
    ),
    "gauss": lambda w, sid, pid: w.write_gauss_group(
        sid, pid, "g0", class_tag=4, int_rule=1,
        element_index=np.array([10, 11]),
        natural_coords=np.zeros((1, 2)),
        components={"stress_xx": np.zeros((_T, 2, 1))},
    ),
    "fibers": lambda w, sid, pid: w.write_fibers_group(
        sid, pid, "f0", section_tag=1, section_class="Fiber",
        element_index=np.array([10, 10]), gp_index=np.array([0, 1]),
        y=np.zeros(2), z=np.zeros(2), area=np.ones(2),
        material_tag=np.array([1, 1]),
        components={"fiber_stress": np.zeros((_T, 2))},
    ),
    "layers": lambda w, sid, pid: w.write_layers_group(
        sid, pid, "l0",
        element_index=np.array([10, 10]), gp_index=np.array([0, 0]),
        layer_index=np.array([0, 1]), sub_gp_index=np.array([0, 0]),
        thickness=np.ones(2),
        local_axes_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        components={"fiber_stress": np.zeros((_T, 2))},
    ),
    "line_stations": lambda w, sid, pid: w.write_line_stations_group(
        sid, pid, "s0", class_tag=64,
        element_index=np.array([10, 11]),
        station_natural_coord=np.array([-1.0, 1.0]),
        components={"axial_force": np.zeros((_T, 2, 2))},
    ),
    "nodal_forces": lambda w, sid, pid: w.write_nodal_forces_group(
        sid, pid, "n0", class_tag=5, frame="global",
        element_index=np.array([10, 11]),
        components={"nodal_resisting_force_x": np.zeros((_T, 2, 2))},
    ),
    "element_ids": lambda w, sid, pid: w.write_element_ids(
        sid, pid, np.array([10, 11]),
    ),
}


# ---------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------

def test_a_second_partition_refuses_by_name(writer) -> None:
    w, sid, path = writer
    _WRITES["nodes"](w, sid, "partition_0")
    with pytest.raises(NativeSecondPartition) as info:
        w.write_nodes(
            sid, "partition_1", node_ids=np.array([5, 9]),
            components={"displacement_x": np.ones((_T, 2))},
        )
    err = info.value
    assert isinstance(err, ValueError)
    assert (err.file, err.stage_id, err.existing, err.refused) == (
        path, sid, "partition_0", "partition_1",
    )
    msg = str(err)
    assert str(path) in msg and f"{sid!r}" in msg
    assert "'partition_0'" in msg and "'partition_1'" in msg
    assert "ADR 0123" in msg
    w.end_stage()
    w.close()
    # The refused write created nothing: the file holds its one
    # partition, opens, and reads the first write.
    assert _partitions(path) == ["partition_0"]
    with NativeReader(path) as r:
        assert r.partitions(sid) == ["partition_0"]
        slab = r.read_nodes(sid, "displacement_x")
    np.testing.assert_array_equal(slab.node_ids, [1, 2])
    np.testing.assert_array_equal(slab.values, np.zeros((_T, 2)))


@pytest.mark.parametrize("second", sorted(_WRITES))
@pytest.mark.parametrize("first", ["nodes", "element_ids", "gauss"])
def test_every_write_refuses_a_second_partition(
    writer, first: str, second: str,
) -> None:
    # Each write_* passes through _require_partition, so each refuses,
    # whichever write created the stage's partition.
    w, sid, path = writer
    _WRITES[first](w, sid, "partition_0")
    with pytest.raises(NativeSecondPartition) as info:
        _WRITES[second](w, sid, "partition_1")
    assert (info.value.existing, info.value.refused) == (
        "partition_0", "partition_1",
    )
    w.end_stage()
    w.close()
    assert _partitions(path) == ["partition_0"]
    NativeReader(path).close()


# ---------------------------------------------------------------------
# What stays legal
# ---------------------------------------------------------------------

def test_one_partition_of_any_name_is_legal(writer) -> None:
    # Count, not name: partition_3 alone is one partition. A guard that
    # refused every id but partition_0 fails here.
    w, sid, path = writer
    for name in sorted(_WRITES):
        _WRITES[name](w, sid, "partition_3")
    w.end_stage()
    w.close()
    assert _partitions(path) == ["partition_3"]
    with NativeReader(path) as r:
        assert r.partitions(sid) == ["partition_3"]
        np.testing.assert_array_equal(
            r.read_nodes(sid, "displacement_x").node_ids, [1, 2],
        )


def test_writes_re_enter_the_stage_partition(writer) -> None:
    # Nodes, then a gauss group, then the element ids, then a second
    # nodes call with another component set: one partition throughout.
    w, sid, path = writer
    _WRITES["nodes"](w, sid, "partition_0")
    _WRITES["gauss"](w, sid, "partition_0")
    _WRITES["element_ids"](w, sid, "partition_0")
    w.write_nodes(sid, "partition_0", node_ids=np.array([1, 2]),
                  components={"displacement_y": np.ones((_T, 2))})
    w.end_stage()
    w.close()
    assert _partitions(path) == ["partition_0"]


def test_each_stage_has_its_own_partition(tmp_path: Path) -> None:
    # The rule is per stage: a second stage's partition is its first.
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open()
        a = w.begin_stage(name="a", kind="static", time=_TIME)
        _WRITES["nodes"](w, a, "partition_0")
        w.end_stage()
        b = w.begin_stage(name="b", kind="static", time=_TIME)
        _WRITES["nodes"](w, b, "partition_1")
        w.end_stage()
    assert _partitions(path, a) == ["partition_0"]
    assert _partitions(path, b) == ["partition_1"]
    with NativeReader(path) as r:
        assert r.partitions(a) == ["partition_0"]
        assert r.partitions(b) == ["partition_1"]


# ---------------------------------------------------------------------
# Single-partition round trip (D2: no PARTITION_REDUCTION in native)
# ---------------------------------------------------------------------

def _attr_names(path: Path) -> set[str]:
    names: set[str] = set()
    with h5py.File(path, "r") as f:
        names.update(f.attrs.keys())

        def visit(_: str, obj: Any) -> None:
            names.update(obj.attrs.keys())

        f.visititems(visit)
    return names


def test_a_single_partition_file_round_trips_exactly(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    ids = np.array([3, 1, 7], dtype=np.int64)
    ux = np.arange(_T * 3, dtype=np.float64).reshape(_T, 3) / 7.0
    sxx = np.arange(_T * 2 * 4, dtype=np.float64).reshape(_T, 2, 4) * 1.5
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="transient", time=_TIME)
        w.write_nodes(sid, "partition_0", node_ids=ids,
                      components={"displacement_x": ux})
        w.write_gauss_group(
            sid, "partition_0", "g0", class_tag=4, int_rule=1,
            element_index=np.array([10, 11]),
            natural_coords=np.zeros((4, 2)),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    with h5py.File(path, "r") as f:
        part = f[f"/stages/{sid}/partitions/partition_0"]
        np.testing.assert_array_equal(part["nodes/_ids"][...], ids)
        assert np.array_equal(part["nodes/displacement_x"][...], ux)
        assert np.array_equal(
            part["elements/gauss_points/g0/stress_xx"][...], sxx,
        )
    with NativeReader(path) as r:
        slab = r.read_nodes(sid, "displacement_x", node_ids=np.array([7]))
    np.testing.assert_array_equal(slab.values[:, 0], ux[:, 2])
    assert not any("PARTITION_REDUCTION" in n.upper() for n in _attr_names(path))
