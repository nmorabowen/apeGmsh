"""#2360: the native reader refuses a node id repeated within one
partition at open, and reads a node held by several partitions only when
every copy agrees.

Native files carry no partition-reduction rule, so a boundary node whose
copies differ (a per-rank partial reaction: 3.0 on one rank, 4.0 on the
other, true value 7.0) used to read as the first copy, 3.0, silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.readers._native import (
    NativePartitionCopiesDisagree,
    NativeRepeatedNodeId,
)
from apeGmsh.results.writers import NativeWriter

_TIME = np.array([0.0, 1.0])


def _write(path: Path, partitions: dict[str, tuple[list[int], dict]]) -> None:
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        for pid, (ids, comps) in partitions.items():
            w.write_nodes(
                sid, pid, node_ids=np.array(ids, dtype=np.int64),
                components={k: np.asarray(v, dtype=np.float64)
                            for k, v in comps.items()},
            )
        w.end_stage()


def test_repeat_within_one_partition_refuses_at_open(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    _write(path, {
        "partition_0": ([1, 2], {"displacement_x": [[0.1, 0.2], [0.1, 0.2]]}),
        "partition_1": ([7, 9, 7], {"displacement_x": [[1.0, 2.0, 3.0]] * 2}),
    })
    with pytest.raises(NativeRepeatedNodeId) as info:
        NativeReader(path)
    err = info.value
    assert err.file == path
    assert err.dataset == "/stages/stage_0/partitions/partition_1/nodes/_ids"
    assert err.repeated == {7: [0, 2]}
    assert str(path) in str(err) and "partition_1/nodes/_ids" in str(err)
    # The refusal closed the file it opened.
    path.unlink()


def test_partial_reaction_copies_refuse(tmp_path: Path) -> None:
    # Node 5 sits on the rank boundary; each rank holds its own share of
    # the reaction (3.0 + 4.0), while the displacement copies agree.
    path = tmp_path / "run.h5"
    _write(path, {
        "partition_0": ([1, 5], {
            "displacement_x": [[0.0, 0.5], [0.0, 0.6]],
            "reaction_force_x": [[1.0, 3.0], [1.0, 3.0]],
        }),
        "partition_1": ([5, 9], {
            "displacement_x": [[0.5, 0.0], [0.6, 0.0]],
            "reaction_force_x": [[4.0, 2.0], [4.0, 2.0]],
        }),
    })
    with NativeReader(path) as r:
        with pytest.raises(NativePartitionCopiesDisagree) as info:
            r.read_nodes("stage_0", "reaction_force_x")
        disp = r.read_nodes("stage_0", "displacement_x")
    err = info.value
    assert (err.stage_id, err.component) == ("stage_0", "reaction_force_x")
    assert err.copies == {5: ["partition_0", "partition_1"]}
    assert "reaction_force_x" in str(err)
    assert "no partition-reduction rule" in str(err)
    # The agreeing component of the same node still reads.
    np.testing.assert_array_equal(disp.node_ids, [1, 5, 9])
    np.testing.assert_array_equal(disp.values[:, 1], [0.5, 0.6])


def test_disagreement_outside_the_requested_slice_still_refuses(
    tmp_path: Path,
) -> None:
    # The copies agree at step 0 and differ at step 1: a read of step 0,
    # filtered to the shared node, refuses all the same.
    path = tmp_path / "run.h5"
    _write(path, {
        "partition_0": ([1, 5], {"reaction_force_x": [[1.0, 0.0], [1.0, 3.0]]}),
        "partition_1": ([5, 9], {"reaction_force_x": [[0.0, 2.0], [4.0, 2.0]]}),
    })
    with NativeReader(path) as r:
        with pytest.raises(NativePartitionCopiesDisagree):
            r.read_nodes(
                "stage_0", "reaction_force_x",
                node_ids=np.array([5]), time_slice=0,
            )


def test_agreeing_boundary_copies_read_first_wins(tmp_path: Path) -> None:
    # Guard against over-refusal: replicated copies that agree bit for
    # bit (NaN included) read as one column, as before.
    path = tmp_path / "run.h5"
    _write(path, {
        "partition_0": ([1, 3], {"displacement_x": [[0.1, 0.3], [0.1, np.nan]]}),
        "partition_1": ([3, 4], {"displacement_x": [[0.3, 0.4], [np.nan, 0.4]]}),
        "partition_2": ([3, 6], {"displacement_x": [[0.3, 0.6], [np.nan, 0.6]]}),
    })
    with NativeReader(path) as r:
        slab = r.read_nodes("stage_0", "displacement_x")
        one = r.read_nodes("stage_0", "displacement_x", node_ids=np.array([3]))
    np.testing.assert_array_equal(slab.node_ids, [1, 3, 4, 6])
    np.testing.assert_array_equal(slab.values[0], [0.1, 0.3, 0.4, 0.6])
    assert np.isnan(slab.values[1, 1])
    np.testing.assert_array_equal(one.node_ids, [3])
    np.testing.assert_array_equal(one.values[0], [0.3])
