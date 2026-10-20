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

The oracle numbers are the fork's reproduction (nmorabowen/OpenSees#861),
as in ``tests/results/readers/test_partition_reduction.py``: support node
1 is shared by two ranks, part-0 holds reaction ``(0, 10)``, part-1 holds
``(20, 20)``, and the serial reaction is ``(20, 30)``. A part-0 that lists
node 1 twice would read ``(20, 40)`` under SUM.

The files are synthetic, written with ``h5py``.
"""
from __future__ import annotations

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
