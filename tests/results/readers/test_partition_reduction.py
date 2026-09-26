"""Partitioned node stitch honours ``PARTITION_REDUCTION`` (WP-126).

Under OpenSeesMP a node on a partition interface is written by every rank
that holds it. Kinematics are the same in every copy, but a reaction (or
an unbalanced force) is only the partial from that rank's own elements:
the true value is the sum of the copies. The Ladruno recorder labels each
result group with ``PARTITION_REDUCTION`` = ``NONE`` / ``SUM`` /
``UNSUPPORTED`` (fork schema ``ladruno_schema_v1.md`` §7.1). Files without
the attribute (older ``.ladruno``, every ``.mpco``) fall back to the result
name.

The numbers are the fork's reproduction (nmorabowen/OpenSees#861): two
trusses into a support node 1 shared by two ranks. Serial reaction
``(20, 30)``; part-0 holds ``(0, 10)``, part-1 holds ``(20, 20)``. The old
first-copy stitch returned ``(0, 10)``.

The files are synthetic, written with ``h5py``, so no fork build or MPI
run is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader

# Two steps, so the sum is checked per step and not only at the last one.
_TIME = np.array([0.5, 1.0])

# node -> (Rx, Ry) at the last step; step 0 is half of it.
_REACTION_PART0 = {1: (0.0, 10.0), 2: (0.0, 5.0)}
_REACTION_PART1 = {1: (20.0, 20.0), 3: (0.0, 15.0)}

# Displacement copies of shared node 1 deliberately differ, so the test
# can tell which copy the stitch kept.
_DISP_PART0 = {1: (0.0, 0.0), 2: (0.0, -0.1)}
_DISP_PART1 = {1: (9.0, 9.0), 3: (0.0, -0.2)}


def _rows(table: dict[int, tuple[float, float]]):
    ids = np.array(sorted(table), dtype=np.int64)
    last = np.array([table[int(n)] for n in ids], dtype=np.float64)
    data = np.stack([0.5 * last, last])  # (T, n, nComp)
    return ids, data


# ---------------------------------------------------------------------
# .ladruno builder
# ---------------------------------------------------------------------

def _ladruno_result(
    on_nodes, name: str, components: str,
    table: dict[int, tuple[float, float]], reduction: Optional[str],
) -> None:
    ids, data = _rows(table)
    grp = on_nodes.create_group(name)
    grp.attrs["COMPONENTS"] = np.array([components.encode()])
    if reduction is not None:
        # The recorder writes a rank-1 fixed-length string, as for
        # COMPONENTS.
        grp.attrs["PARTITION_REDUCTION"] = np.array(
            [reduction.encode()], dtype="S16",
        )
    grp.create_dataset("ID", data=ids.reshape(-1, 1))
    grp.create_dataset("DATA", data=data)
    grp.create_dataset("TIME", data=_TIME)
    grp.create_dataset("STEP", data=np.arange(_TIME.size))


def _ladruno_part(
    path: Path, part: int, *,
    reaction: dict, disp: dict,
    reaction_reduction: Optional[str], disp_reduction: Optional[str],
) -> Path:
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = 1
        info.attrs["PARTITIONED"] = 1
        info.attrs["PARTITION_ID"] = part
        info.attrs["NUM_PARTITIONS"] = 2
        stage = f.create_group("MODEL_STAGE[1]")
        stage.attrs["KIND"] = np.bytes_(b"static")
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        _ladruno_result(
            on_nodes, "REACTION_FORCE", "Rx,Ry", reaction, reaction_reduction,
        )
        _ladruno_result(on_nodes, "DISPLACEMENT", "Ux,Uy", disp, disp_reduction)
    return path


def _ladruno_pair(
    tmp_path: Path, *,
    reaction_reduction: Optional[str] = "SUM",
    disp_reduction: Optional[str] = "NONE",
    reaction_reduction_part1: Optional[str] = None,
) -> list[Path]:
    """``run.part-0.ladruno`` + ``run.part-1.ladruno``. ``None`` leaves
    the attribute out (a file written before WP-126)."""
    r1 = (reaction_reduction if reaction_reduction_part1 is None
          else reaction_reduction_part1)
    return [
        _ladruno_part(
            tmp_path / "run.part-0.ladruno", 0,
            reaction=_REACTION_PART0, disp=_DISP_PART0,
            reaction_reduction=reaction_reduction,
            disp_reduction=disp_reduction,
        ),
        _ladruno_part(
            tmp_path / "run.part-1.ladruno", 1,
            reaction=_REACTION_PART1, disp=_DISP_PART1,
            reaction_reduction=r1, disp_reduction=disp_reduction,
        ),
    ]


def _stitched(reader, component: str, **kw) -> dict[int, np.ndarray]:
    slab = reader.read_nodes("stage_0", component, **kw)
    return {int(n): slab.values[:, i] for i, n in enumerate(slab.node_ids)}


# ---------------------------------------------------------------------
# Per-file reader reports the reduction kind
# ---------------------------------------------------------------------

def test_ladruno_reader_reports_attribute(tmp_path: Path) -> None:
    p0, _ = _ladruno_pair(tmp_path)
    with LadrunoReader(p0) as r:
        assert r.node_partition_reduction("stage_0", "reaction_force_y") == "SUM"
        assert r.node_partition_reduction("stage_0", "displacement_x") == "NONE"
        # A component the file does not record has no reduction.
        assert r.node_partition_reduction("stage_0", "velocity_x") is None


def test_ladruno_reader_falls_back_to_result_name(tmp_path: Path) -> None:
    p0, _ = _ladruno_pair(
        tmp_path, reaction_reduction=None, disp_reduction=None,
    )
    with LadrunoReader(p0) as r:
        assert r.node_partition_reduction("stage_0", "reaction_force_x") == "SUM"
        assert r.node_partition_reduction("stage_0", "displacement_x") == "NONE"


def test_ladruno_reader_rejects_unknown_attribute_value(tmp_path: Path) -> None:
    p0, _ = _ladruno_pair(tmp_path, reaction_reduction="MEAN")
    with LadrunoReader(p0) as r:
        with pytest.raises(ValueError, match="PARTITION_REDUCTION"):
            r.node_partition_reduction("stage_0", "reaction_force_x")


# ---------------------------------------------------------------------
# Ladruno stitch
# ---------------------------------------------------------------------

def test_shared_support_reaction_is_summed(tmp_path: Path) -> None:
    """The WP-126 reproduction: (0, 10) + (20, 20) -> (20, 30)."""
    with LadrunoMultiPartitionReader(_ladruno_pair(tmp_path)) as r:
        rx = _stitched(r, "reaction_force_x")
        ry = _stitched(r, "reaction_force_y")
    np.testing.assert_allclose(rx[1], [10.0, 20.0])
    np.testing.assert_allclose(ry[1], [15.0, 30.0])
    # A node held by one part keeps its own value.
    np.testing.assert_allclose(ry[2], [2.5, 5.0])
    np.testing.assert_allclose(ry[3], [7.5, 15.0])


def test_reaction_sum_honours_node_and_time_filters(tmp_path: Path) -> None:
    with LadrunoMultiPartitionReader(_ladruno_pair(tmp_path)) as r:
        slab = r.read_nodes(
            "stage_0", "reaction_force_y",
            node_ids=np.array([1]), time_slice=-1,
        )
    assert slab.node_ids.tolist() == [1]
    np.testing.assert_allclose(slab.values, [[30.0]])


def test_shared_displacement_keeps_first_copy(tmp_path: Path) -> None:
    """``NONE`` is unchanged: the first partition's copy wins."""
    with LadrunoMultiPartitionReader(_ladruno_pair(tmp_path)) as r:
        ux = _stitched(r, "displacement_x")
    np.testing.assert_allclose(ux[1], [0.0, 0.0])
    assert sorted(ux) == [1, 2, 3]


def test_reaction_without_attribute_sums_by_name(tmp_path: Path) -> None:
    """A pre-WP-126 file has no attribute; REACTION_FORCE still sums."""
    paths = _ladruno_pair(
        tmp_path, reaction_reduction=None, disp_reduction=None,
    )
    with LadrunoMultiPartitionReader(paths) as r:
        ry = _stitched(r, "reaction_force_y")
        ux = _stitched(r, "displacement_x")
    np.testing.assert_allclose(ry[1], [15.0, 30.0])
    np.testing.assert_allclose(ux[1], [0.0, 0.0])


def test_unsupported_result_is_refused(tmp_path: Path) -> None:
    paths = _ladruno_pair(tmp_path, reaction_reduction="UNSUPPORTED")
    with LadrunoMultiPartitionReader(paths) as r:
        with pytest.raises(ValueError, match="UNSUPPORTED"):
            r.read_nodes("stage_0", "reaction_force_x")


def test_partitions_disagreeing_on_reduction_are_refused(tmp_path: Path) -> None:
    paths = _ladruno_pair(
        tmp_path, reaction_reduction="SUM", reaction_reduction_part1="NONE",
    )
    with LadrunoMultiPartitionReader(paths) as r:
        with pytest.raises(ValueError, match="disagree"):
            r.read_nodes("stage_0", "reaction_force_x")


def test_partitioned_energy_is_refused(tmp_path: Path) -> None:
    """Each rank's energy balance is a partial that no sum recovers."""
    with LadrunoMultiPartitionReader(_ladruno_pair(tmp_path)) as r:
        with pytest.raises(ValueError, match="UNSUPPORTED"):
            r.read_energy("stage_0")


# ---------------------------------------------------------------------
# MPCO stitch (name fallback only: .mpco has no attribute)
# ---------------------------------------------------------------------

def _mpco_result(on_nodes, name: str, components: str, table: dict) -> None:
    ids, data = _rows(table)
    grp = on_nodes.create_group(name)
    grp.attrs["COMPONENTS"] = np.array([components.encode()])
    grp.create_dataset("ID", data=ids.reshape(-1, 1).astype(np.int32))
    steps = grp.create_group("DATA")
    for k in range(_TIME.size):
        ds = steps.create_dataset(f"STEP_{k}", data=data[k])
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = _TIME[k]


def _mpco_part(path: Path, reaction: dict, disp: dict) -> Path:
    ids = np.array(sorted(set(reaction) | set(disp)), dtype=np.int32)
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.create_dataset("SPATIAL_DIM", data=2)
        stage = f.create_group("MODEL_STAGE[1]")
        stage.attrs["STEP"] = 0
        stage.attrs["TIME"] = 0.0
        nodes = stage.create_group("MODEL/NODES")
        nodes.create_dataset("ID", data=ids.reshape(-1, 1))
        nodes.create_dataset("COORDINATES", data=np.zeros((ids.size, 2)))
        stage.create_group("MODEL/ELEMENTS")
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        _mpco_result(on_nodes, "REACTION_FORCE", "Rx,Ry", reaction)
        _mpco_result(on_nodes, "DISPLACEMENT", "Ux,Uy", disp)
    return path


def test_mpco_shared_reaction_is_summed(tmp_path: Path) -> None:
    paths = [
        _mpco_part(tmp_path / "run.part-0.mpco", _REACTION_PART0, _DISP_PART0),
        _mpco_part(tmp_path / "run.part-1.mpco", _REACTION_PART1, _DISP_PART1),
    ]
    reader = MPCOMultiPartitionReader(paths)
    try:
        ry = _stitched(reader, "reaction_force_y")
        ux = _stitched(reader, "displacement_x")
    finally:
        reader.close()
    np.testing.assert_allclose(ry[1], [15.0, 30.0])
    np.testing.assert_allclose(ux[1], [0.0, 0.0])
