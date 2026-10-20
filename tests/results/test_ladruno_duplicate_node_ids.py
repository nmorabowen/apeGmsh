"""A ``.ladruno`` that lists one node id twice is refused at open (#2300).

The Python twin of the viewer's #2299. Before the fix
``LadrunoReader.read_nodes`` returned ``ON_NODES/<res>/ID`` as written, so
a repeated id read as two columns of one node, and a part set that sums
partition copies (``PARTITION_REDUCTION=SUM``) added both, counting the
node twice. Now :class:`LadrunoReader` raises
:class:`LadrunoRepeatedNodeId` when the file opens, naming the file, the
``ID`` dataset and the repeated id with its rows. The check is per file,
so a part set refuses through the part that repeats the id, and a node in
*two part files* (the copy SUM exists for) still opens and sums.

The model's own node list, ``MODEL/NODES/ID``, is checked the same way
(#2307): a repeated id there gives one node two coordinates, and the
FEMData built from the file two nodes with one id.

The oracle numbers are the fork's reproduction (nmorabowen/OpenSees#861),
as in ``tests/results/readers/test_partition_reduction.py``: support node
1 is shared by two ranks, part-0 holds reaction ``(0, 10)``, part-1 holds
``(20, 20)``, and the serial reaction is ``(20, 30)``. A part-0 that lists
node 1 twice would read ``(20, 40)`` under SUM.

The files are synthetic, written with ``h5py``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno import (
    LadrunoReader,
    LadrunoRepeatedNodeId,
)
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader

_TIME = np.array([0.5, 1.0])

_REPO = Path(__file__).resolve().parents[2]


def _result(
    on_nodes: "h5py.Group", name: str, components: str,
    rows: Sequence[tuple[int, tuple[float, float]]], reduction: str,
) -> None:
    """One result group; ``rows`` keeps its order and any repeated id."""
    ids = np.array([n for n, _ in rows], dtype=np.int64)
    last = np.array([v for _, v in rows], dtype=np.float64)
    grp = on_nodes.create_group(name)
    grp.attrs["COMPONENTS"] = np.array([components.encode()])
    grp.attrs["PARTITION_REDUCTION"] = np.array(
        [reduction.encode()], dtype="S16",
    )
    grp.create_dataset("ID", data=ids.reshape(-1, 1))
    grp.create_dataset("DATA", data=np.stack([0.5 * last, last]))
    grp.create_dataset("TIME", data=_TIME)
    grp.create_dataset("STEP", data=np.arange(_TIME.size))


def _file(
    path: Path, *,
    reaction: Sequence[tuple[int, tuple[float, float]]],
    disp: Sequence[tuple[int, tuple[float, float]]],
    part: "int | None" = None,
) -> Path:
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = 1
        if part is not None:
            info.attrs["PARTITIONED"] = 1
            info.attrs["PARTITION_ID"] = part
            info.attrs["NUM_PARTITIONS"] = 2
        stage = f.create_group("MODEL_STAGE[1]")
        stage.attrs["KIND"] = np.bytes_(b"static")
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        _result(on_nodes, "REACTION_FORCE", "Rx,Ry", reaction, "SUM")
        _result(on_nodes, "DISPLACEMENT", "Ux,Uy", disp, "NONE")
    return path


_REACTION_PART0 = [(1, (0.0, 10.0)), (2, (0.0, 5.0))]
_REACTION_PART1 = [(1, (20.0, 20.0)), (3, (0.0, 15.0))]
_DISP_PART0 = [(1, (0.0, 0.0)), (2, (0.0, -0.1))]
_DISP_PART1 = [(1, (0.0, 0.0)), (3, (0.0, -0.2))]


def _assert_closed(path: Path) -> None:
    """The refused file is unlocked: an ``"r+"`` open succeeds."""
    with h5py.File(path, "r+"):
        pass


# ---------------------------------------------------------------------
# A single file
# ---------------------------------------------------------------------

def test_single_file_repeated_node_id_refuses_at_open(tmp_path: Path) -> None:
    path = _file(
        tmp_path / "run.ladruno",
        reaction=_REACTION_PART0,
        disp=[(1, (0.0, 0.0)), (2, (0.0, -0.1)), (1, (0.0, 0.0))],
    )
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    err = info.value
    assert err.file == path
    assert err.dataset == "/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/ID"
    assert err.repeated == {1: [0, 2]}
    msg = str(err)
    assert str(path) in msg
    assert "/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/ID" in msg
    assert "{1: [0, 2]}" in msg
    _assert_closed(path)


def test_public_entry_refuses_by_name(tmp_path: Path) -> None:
    path = _file(
        tmp_path / "run.ladruno",
        reaction=[(4, (0.0, 1.0)), (4, (0.0, 2.0))],
        disp=[(4, (0.0, 0.0))],
    )
    with pytest.raises(LadrunoRepeatedNodeId, match="REACTION_FORCE/ID"):
        Results.from_ladruno(path)


def test_repeated_id_in_a_later_stage_refuses(tmp_path: Path) -> None:
    """Every stage is checked, not only the first one read."""
    path = _file(
        tmp_path / "run.ladruno", reaction=_REACTION_PART0, disp=_DISP_PART0,
    )
    with h5py.File(path, "a") as f:
        stage = f.create_group("MODEL_STAGE[2]")
        stage.attrs["KIND"] = np.bytes_(b"static")
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        _result(on_nodes, "DISPLACEMENT", "Ux,Uy",
                [(7, (0.0, 0.0)), (8, (0.0, 0.0)), (7, (0.0, 0.0)),
                 (8, (0.0, 0.0)), (7, (0.0, 0.0))], "NONE")
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    assert info.value.dataset == (
        "/MODEL_STAGE[2]/RESULTS/ON_NODES/DISPLACEMENT/ID"
    )
    assert info.value.repeated == {7: [0, 2, 4], 8: [1, 3]}


def test_repeated_id_in_node_envelope_refuses(tmp_path: Path) -> None:
    path = _file(
        tmp_path / "run.ladruno", reaction=_REACTION_PART0, disp=_DISP_PART0,
    )
    with h5py.File(path, "a") as f:
        env = f["MODEL_STAGE[1]"].create_group(
            "RESULTS/ENVELOPES/ON_NODES/DISPLACEMENT",
        )
        env.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
        env.create_dataset("ID", data=np.array([[5], [6], [5]]))
        for name in ("MIN", "MAX", "ABSMAX"):
            env.create_dataset(name, data=np.zeros((3, 2)))
        env.create_dataset("ARG_STEP", data=np.zeros((3, 2), dtype=np.int64))
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    assert info.value.dataset == (
        "/MODEL_STAGE[1]/RESULTS/ENVELOPES/ON_NODES/DISPLACEMENT/ID"
    )
    assert info.value.repeated == {5: [0, 2]}


def test_repeated_id_in_modal_group_refuses(tmp_path: Path) -> None:
    path = _file(
        tmp_path / "run.ladruno", reaction=_REACTION_PART0, disp=_DISP_PART0,
    )
    with h5py.File(path, "a") as f:
        modal = f["MODEL_STAGE[1]/RESULTS/ON_NODES"].create_group(
            "MODES_OF_VIBRATION(U)",
        )
        modal.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
        modal.create_dataset("ID", data=np.array([[1], [2], [2]]))
        mode = modal.create_dataset(
            "DATA/STEP_0/MODE_0", data=np.zeros((3, 2)),
        )
        mode.attrs["LAMBDA"] = 4.0
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    assert info.value.dataset == (
        "/MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)/ID"
    )
    assert info.value.repeated == {2: [1, 2]}


def test_unsorted_distinct_ids_still_open(tmp_path: Path) -> None:
    """Ids out of order are not repeats: the sorted fast path falls back
    to a full check and the file opens."""
    path = _file(
        tmp_path / "run.ladruno",
        reaction=[(3, (0.0, 1.0)), (1, (0.0, 2.0)), (2, (0.0, 3.0))],
        disp=[(2, (0.0, 0.0)), (1, (0.0, 0.0))],
    )
    with LadrunoReader(path) as r:
        ry = r.read_nodes("stage_0", "reaction_force_y")
    assert ry.node_ids.tolist() == [3, 1, 2]
    np.testing.assert_allclose(ry.values[-1], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------
# The model's node list, MODEL/NODES/ID (#2307)
# ---------------------------------------------------------------------

_QUAD2D = _REPO / "tests/fixtures/ladruno/quad2d.ladruno"


def _quad2d_with_model_ids(tmp_path: Path, ids: Sequence[int]) -> Path:
    """A copy of ``quad2d.ladruno`` (4 nodes) with ``MODEL/NODES/ID``
    rewritten; its coordinates and results are untouched."""
    path = tmp_path / "quad2d.ladruno"
    shutil.copy(_QUAD2D, path)
    with h5py.File(path, "a") as f:
        node_id = f["MODEL_STAGE[1]/MODEL/NODES/ID"]
        assert node_id[...].ravel().tolist() == [1, 2, 3, 4]
        node_id[...] = np.asarray(ids, dtype=node_id.dtype).reshape(
            node_id.shape,
        )
    return path


def test_repeated_model_node_id_refuses_at_open(tmp_path: Path) -> None:
    """4 nodes, 3 unique ids: node 1 is listed at rows 0 and 3. Without
    the refusal the FEMData read from the file holds id 1 twice, at
    (0, 0) and (0, 1), and no node 4 for the quad that names it."""
    path = _quad2d_with_model_ids(tmp_path, [1, 2, 3, 1])
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    err = info.value
    assert err.model is True
    assert err.file == path
    assert err.dataset == "/MODEL_STAGE[1]/MODEL/NODES/ID"
    assert err.repeated == {1: [0, 3]}
    msg = str(err)
    assert "/MODEL_STAGE[1]/MODEL/NODES/ID" in msg
    assert "{1: [0, 3]}" in msg
    assert "two coordinates" in msg
    _assert_closed(path)


def test_repeated_model_node_id_refuses_through_public_entry(
    tmp_path: Path,
) -> None:
    path = _quad2d_with_model_ids(tmp_path, [1, 2, 3, 1])
    with pytest.raises(LadrunoRepeatedNodeId, match="MODEL/NODES/ID"):
        Results.from_ladruno(path)
    _assert_closed(path)


def test_unsorted_distinct_model_ids_still_open(tmp_path: Path) -> None:
    path = _quad2d_with_model_ids(tmp_path, [4, 2, 3, 1])
    with LadrunoReader(path) as r:
        fem = r.fem()
    assert fem is not None
    assert sorted(int(n) for n in fem.nodes.ids) == [1, 2, 3, 4]


def test_result_id_check_still_names_a_result(tmp_path: Path) -> None:
    """A result repeat is not reported as a model repeat."""
    path = _file(
        tmp_path / "run.ladruno",
        reaction=_REACTION_PART0,
        disp=[(1, (0.0, 0.0)), (1, (0.0, 0.0))],
    )
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    assert info.value.model is False
    assert "two columns" in str(info.value)


# A list a subtracting fast path (``np.diff(ids) > 0``) reads as strictly
# increasing: 2**63 - 1 - 0 > 0, -2 - (2**63 - 1) wraps to 2**63 - 1 > 0,
# and 0 - (-2) = 2 > 0. Id 0 repeats at rows 0 and 3. The uint64 form
# reads as the same int64 list.
_B = 2**63 - 1
_WRAP_CASES = {
    "int64": np.array([0, _B, -2, 0], dtype=np.int64),
    "uint64": np.array([0, _B, 2**64 - 2, 0], dtype=np.uint64),
}


def _replace_id(path: Path, dataset: str, ids: np.ndarray) -> None:
    with h5py.File(path, "a") as f:
        shape = (ids.size, 1) if f[dataset].ndim == 2 else (ids.size,)
        del f[dataset]
        f.create_dataset(dataset, data=ids.reshape(shape))


@pytest.mark.parametrize("dtype", sorted(_WRAP_CASES))
def test_wrapping_result_ids_refuse(tmp_path: Path, dtype: str) -> None:
    path = _file(
        tmp_path / "run.ladruno", reaction=_REACTION_PART0,
        disp=[(n, (0.0, 0.0)) for n in (1, 2, 3, 4)],
    )
    dataset = "/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/ID"
    _replace_id(path, dataset, _WRAP_CASES[dtype])
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    assert info.value.model is False
    assert info.value.dataset == dataset
    assert info.value.repeated == {0: [0, 3]}
    _assert_closed(path)


@pytest.mark.parametrize("dtype", sorted(_WRAP_CASES))
def test_wrapping_model_ids_refuse(tmp_path: Path, dtype: str) -> None:
    path = _quad2d_with_model_ids(tmp_path, [1, 2, 3, 4])
    dataset = "/MODEL_STAGE[1]/MODEL/NODES/ID"
    _replace_id(path, dataset, _WRAP_CASES[dtype])
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoReader(path)
    assert info.value.model is True
    assert info.value.dataset == dataset
    assert info.value.repeated == {0: [0, 3]}
    _assert_closed(path)


# ---------------------------------------------------------------------
# A part set
# ---------------------------------------------------------------------

def test_part_set_where_one_part_repeats_a_node_refuses(tmp_path: Path) -> None:
    """Without the refusal the SUM stitch reads node 1 as (20, 40)."""
    p0 = _file(
        tmp_path / "run.part-0.ladruno", part=0,
        reaction=[*_REACTION_PART0, (1, (0.0, 10.0))], disp=_DISP_PART0,
    )
    p1 = _file(
        tmp_path / "run.part-1.ladruno", part=1,
        reaction=_REACTION_PART1, disp=_DISP_PART1,
    )
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoMultiPartitionReader([p0, p1])
    assert info.value.file == p0
    assert info.value.dataset == (
        "/MODEL_STAGE[1]/RESULTS/ON_NODES/REACTION_FORCE/ID"
    )
    assert info.value.repeated == {1: [0, 2]}
    _assert_closed(p0)
    _assert_closed(p1)


def test_part_set_where_a_later_part_repeats_closes_earlier_parts(
    tmp_path: Path,
) -> None:
    """The bad part opens second, so part-0 is already open when part-1
    refuses: the façade closes it before the error propagates."""
    p0 = _file(
        tmp_path / "run.part-0.ladruno", part=0,
        reaction=_REACTION_PART0, disp=_DISP_PART0,
    )
    p1 = _file(
        tmp_path / "run.part-1.ladruno", part=1,
        reaction=[*_REACTION_PART1, (3, (0.0, 15.0))], disp=_DISP_PART1,
    )
    with pytest.raises(LadrunoRepeatedNodeId) as info:
        LadrunoMultiPartitionReader([p0, p1])
    assert info.value.file == p1
    assert info.value.repeated == {3: [1, 2]}
    # ``info`` holds the traceback, and with it the façade and its open
    # part-0 reader; only the close-on-error path unlocks part-0 here.
    _assert_closed(p0)
    _assert_closed(p1)


def test_node_shared_across_parts_still_sums(tmp_path: Path) -> None:
    """The legal copy: node 1 once in each part file. SUM gives the
    serial reaction (20, 30); NONE keeps the first copy."""
    paths = [
        _file(tmp_path / "run.part-0.ladruno", part=0,
              reaction=_REACTION_PART0, disp=_DISP_PART0),
        _file(tmp_path / "run.part-1.ladruno", part=1,
              reaction=_REACTION_PART1, disp=_DISP_PART1),
    ]
    with LadrunoMultiPartitionReader(paths) as r:
        rx = r.read_nodes("stage_0", "reaction_force_x")
        ry = r.read_nodes("stage_0", "reaction_force_y")
        uy = r.read_nodes("stage_0", "displacement_y")
    assert rx.node_ids.tolist() == [1, 2, 3]
    np.testing.assert_allclose(rx.values[-1], [20.0, 0.0, 0.0])
    np.testing.assert_allclose(ry.values[-1], [30.0, 5.0, 15.0])
    np.testing.assert_allclose(ry.values[0], [15.0, 2.5, 7.5])
    np.testing.assert_allclose(uy.values[-1], [0.0, -0.1, -0.2])


# ---------------------------------------------------------------------
# Every committed .ladruno still opens
# ---------------------------------------------------------------------

_COMMITTED = sorted(
    p for d in ("tests/fixtures", "apeGmshViewer/fixtures")
    for p in (_REPO / d).rglob("*.ladruno")
)


def test_committed_fixtures_were_found() -> None:
    assert len(_COMMITTED) >= 20, _COMMITTED


@pytest.mark.parametrize(
    "path", _COMMITTED, ids=[p.relative_to(_REPO).as_posix() for p in _COMMITTED],
)
def test_committed_ladruno_opens(path: Path) -> None:
    with LadrunoReader(path):
        pass
