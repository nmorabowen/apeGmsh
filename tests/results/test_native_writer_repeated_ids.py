"""#2361: ``NativeWriter`` refuses an id repeated within one partition
chunk, the writer side of #2360.

Since #2360 the native reader refuses, at open, a file whose partition
``nodes/_ids`` lists a node twice. The writer used to produce that file
without a word (``write_nodes(..., node_ids=[7, 7])``). It now refuses the
write by name (file, stage, partition, dataset, repeated ids) before any
of the call reaches the file. The element id writes have the same gap and
the same refusal: ``elements/_ids`` and the ``_element_index`` of the
gauss-point, line-station and nodal-force groups, and the (element, Gauss
point, layer, sub-point) row key of a layers group. A fibers group lists
one row per fiber, so its ``_element_index`` repeats by design.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.readers._native import NativeRepeatedNodeId
from apeGmsh.results.writers import NativeWriter
from apeGmsh.results.writers._native import NativeWriteRepeatedId

_TIME = np.array([0.0, 1.0])
_PART = "/stages/stage_0/partitions/partition_0"


@pytest.fixture
def writer(tmp_path: Path) -> Iterator[tuple[NativeWriter, str, Path]]:
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        yield w, sid, path


def _values(*shape: int) -> np.ndarray:
    return np.zeros((len(_TIME), *shape), dtype=np.float64)


def _partition(path: Path) -> list[str]:
    """Everything the file holds under ``partition_0`` (empty when the
    refused call left nothing behind)."""
    names: list[str] = []
    with h5py.File(path, "r") as f:
        if _PART in f:
            f[_PART].visit(names.append)
    return names


# ---------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------

def test_write_nodes_refuses_a_repeated_node_id_by_name(writer) -> None:
    w, sid, path = writer
    with pytest.raises(NativeWriteRepeatedId) as info:
        w.write_nodes(
            sid, "partition_0", node_ids=np.array([7, 9, 7]),
            components={"displacement_x": _values(3)},
        )
    err = info.value
    assert (err.file, err.stage_id, err.partition_id) == (
        path, sid, "partition_0",
    )
    assert err.dataset == f"{_PART}/nodes/_ids"
    assert err.repeated == {7: [0, 2]}
    msg = str(err)
    assert str(path) in msg and "'partition_0'" in msg and "'stage_0'" in msg
    assert "nodes/_ids" in msg and "{7: [0, 2]}" in msg
    w.end_stage()
    w.close()
    # Nothing of the refused call reached the file, and the file opens.
    assert _partition(path) == []
    NativeReader(path).close()


def test_the_two_element_premise_case_refuses(writer) -> None:
    # The card's premise, verbatim: node_ids=[7, 7].
    w, sid, _ = writer
    with pytest.raises(NativeWriteRepeatedId) as info:
        w.write_nodes(
            sid, "partition_0", node_ids=[7, 7],
            components={"displacement_x": _values(2)},
        )
    assert info.value.repeated == {7: [0, 1]}


def test_a_node_held_by_two_partitions_is_not_a_repeat(writer) -> None:
    # A boundary node replicated across partitions is each partition's
    # own row; the reader checks the copies agree (#2360).
    w, sid, path = writer
    w.write_nodes(sid, "partition_0", node_ids=np.array([1, 5]),
                  components={"displacement_x": _values(2)})
    w.write_nodes(sid, "partition_1", node_ids=np.array([5, 9]),
                  components={"displacement_x": _values(2)})
    w.end_stage()
    w.close()
    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x")
    np.testing.assert_array_equal(slab.node_ids, [1, 5, 9])


def test_message_shows_five_repeats_then_counts_the_rest(writer) -> None:
    w, sid, _ = writer
    ids = np.repeat(np.arange(1, 8), 2)          # 7 ids, each twice
    with pytest.raises(NativeWriteRepeatedId) as info:
        w.write_nodes(sid, "partition_0", node_ids=ids,
                      components={"displacement_x": _values(ids.size)})
    assert len(info.value.repeated) == 7
    assert "and 2 more" in str(info.value)


@pytest.mark.parametrize("ids", [
    [7, 7], [1, 2, 1, 2], [3, 1, 2], [5], [], [2, 1, 2, 3, 3],
    [0, 2**63 - 1, -2, 0],
])
def test_writer_refuses_exactly_what_the_reader_refuses(
    tmp_path: Path, ids: list[int],
) -> None:
    """Cross-check: the writer refuses an id list iff the reader refuses
    a file holding it, naming the same rows. The reader's file is made
    with h5py, so it does not depend on the writer."""
    ids_arr = np.array(ids, dtype=np.int64)

    reader_file = tmp_path / "reader.h5"
    with NativeWriter(reader_file) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        w.write_nodes(sid, "partition_0",
                      node_ids=np.arange(ids_arr.size, dtype=np.int64),
                      components={"displacement_x": _values(ids_arr.size)})
        w.end_stage()
    with h5py.File(reader_file, "r+") as f:
        f[f"{_PART}/nodes/_ids"][...] = ids_arr
    try:
        NativeReader(reader_file).close()
        reader_refused = None
    except NativeRepeatedNodeId as exc:
        reader_refused = exc.repeated

    with NativeWriter(tmp_path / "writer.h5") as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        try:
            w.write_nodes(sid, "partition_0", node_ids=ids_arr,
                          components={"displacement_x": _values(ids_arr.size)})
            writer_refused = None
        except NativeWriteRepeatedId as exc:
            writer_refused = exc.repeated

    assert writer_refused == reader_refused


# ---------------------------------------------------------------------
# element ids
# ---------------------------------------------------------------------

def test_write_element_ids_refuses_a_repeated_element_id(writer) -> None:
    w, sid, path = writer
    with pytest.raises(NativeWriteRepeatedId) as info:
        w.write_element_ids(sid, "partition_0", np.array([4, 4, 6]))
    assert info.value.dataset == f"{_PART}/elements/_ids"
    assert info.value.repeated == {4: [0, 1]}
    assert "element id(s)" in str(info.value)
    w.end_stage()
    w.close()
    assert _partition(path) == []


def _gauss(w: NativeWriter, sid: str, eidx: list[int]) -> None:
    w.write_gauss_group(
        sid, "partition_0", "g0", class_tag=4, int_rule=1,
        element_index=np.array(eidx), natural_coords=np.zeros((1, 2)),
        components={"stress_xx": _values(len(eidx), 1)},
    )


def _line_stations(w: NativeWriter, sid: str, eidx: list[int]) -> None:
    w.write_line_stations_group(
        sid, "partition_0", "g0", class_tag=64,
        element_index=np.array(eidx),
        station_natural_coord=np.array([-1.0, 1.0]),
        components={"axial_force": _values(len(eidx), 2)},
    )


def _nodal_forces(w: NativeWriter, sid: str, eidx: list[int]) -> None:
    w.write_nodal_forces_group(
        sid, "partition_0", "g0", class_tag=5, frame="global",
        element_index=np.array(eidx),
        components={"nodal_resisting_force_x": _values(len(eidx), 2)},
    )


@pytest.mark.parametrize("write, category", [
    (_gauss, "gauss_points"),
    (_line_stations, "line_stations"),
    (_nodal_forces, "nodal_forces"),
])
def test_element_group_refuses_a_repeated_element_id(
    writer, write, category: str,
) -> None:
    w, sid, path = writer
    with pytest.raises(NativeWriteRepeatedId) as info:
        write(w, sid, [3, 8, 3, 8])
    err = info.value
    assert err.dataset == f"{_PART}/elements/{category}/g0/_element_index"
    assert err.repeated == {3: [0, 2], 8: [1, 3]}
    assert (err.stage_id, err.partition_id) == (sid, "partition_0")
    # The refused group was never created, so the same group id writes.
    write(w, sid, [3, 8])
    w.end_stage()
    w.close()
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{_PART}/elements/{category}/g0/_element_index"][...], [3, 8],
        )


def _layers(w: NativeWriter, sid: str, key: list[tuple[int, int, int, int]],
            ) -> None:
    cols = np.array(key, dtype=np.int64).T
    n = cols.shape[1]
    w.write_layers_group(
        sid, "partition_0", "g0",
        element_index=cols[0], gp_index=cols[1],
        layer_index=cols[2], sub_gp_index=cols[3],
        thickness=np.ones(n), local_axes_quaternion=np.tile([1.0, 0, 0, 0], (n, 1)),
        components={"stress_xx": _values(n)},
    )


def test_layers_refuse_a_repeated_row_key(writer) -> None:
    w, sid, path = writer
    with pytest.raises(NativeWriteRepeatedId) as info:
        _layers(w, sid, [(1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)])
    assert info.value.repeated == {(1, 0, 0, 0): [0, 2]}
    assert "layer row id(s)" in str(info.value)
    assert f"{_PART}/elements/layers/g0" in info.value.dataset
    w.end_stage()
    w.close()
    assert _partition(path) == []


def test_layers_repeat_an_element_across_their_distinct_rows(writer) -> None:
    # One element, two Gauss points, two layers: the element id repeats,
    # the row key does not.
    w, sid, path = writer
    _layers(w, sid, [(1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 0)])
    w.end_stage()
    w.close()
    with h5py.File(path, "r") as f:
        assert f[f"{_PART}/elements/layers/g0/stress_xx"].shape == (2, 4)


def test_fibers_repeat_their_element_id_by_design(writer) -> None:
    # One row per fiber: two fibers of element 2 at section 0.
    w, sid, path = writer
    w.write_fibers_group(
        sid, "partition_0", "g0", section_tag=1, section_class="Fiber",
        element_index=np.array([2, 2]), gp_index=np.array([0, 0]),
        y=np.array([-0.1, 0.1]), z=np.zeros(2), area=np.ones(2),
        material_tag=np.ones(2, dtype=np.int64),
        components={"fiber_stress": _values(2)},
    )
    w.end_stage()
    w.close()
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{_PART}/elements/fibers/g0/_element_index"][...], [2, 2],
        )
